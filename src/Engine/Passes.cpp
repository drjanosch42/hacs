// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Passes.cpp -- see Passes.h.
#include "Passes.h"
#include "EngineTrace.h"

#include <chrono>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Recorder.h"

namespace engine {
namespace tape {

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------
const char* op_name(Op op) {
    switch (op) {
        case Op::GEMM: return "GEMM";               case Op::BATCHED_GEMM: return "BATCHED_GEMM";
        case Op::AXPY: return "AXPY";               case Op::SCAL: return "SCAL";
        case Op::TRANSPOSE: return "TRANSPOSE";     case Op::CANONICALIZE: return "CANONICALIZE";
        case Op::PANEL_RREF: return "PANEL_RREF";   case Op::PIVOT_SEARCH: return "PIVOT_SEARCH";
        case Op::TRSM: return "TRSM";               case Op::SCHUR_UPDATE: return "SCHUR_UPDATE";
        case Op::GATHER: return "GATHER";           case Op::SCATTER: return "SCATTER";
        case Op::COMPARE: return "COMPARE";
        case Op::H2D: return "H2D";                 case Op::D2H: return "D2H";
    }
    return "?";
}

const char* device_policy_name(DevicePolicy p) {
    switch (p) {
        case DevicePolicy::Cpu: return "cpu";
        case DevicePolicy::Gpu: return "gpu";
        case DevicePolicy::Hybrid: return "hybrid";
    }
    return "?";
}

ThreadClass thread_class(Op op) {
    switch (op) {
        case Op::GEMM: case Op::BATCHED_GEMM: case Op::SCHUR_UPDATE: case Op::TRSM:
            return ThreadClass::ComputeBound;
        case Op::PANEL_RREF: case Op::PIVOT_SEARCH:
            return ThreadClass::Serial;
        default:
            return ThreadClass::MemoryBound;
    }
}

namespace {

bool env_flag(const char* name, bool dflt) {
    const char* e = std::getenv(name);
    if (e == nullptr || e[0] == '\0') return dflt;
    return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0 || std::strcmp(e, "false") == 0);
}


bool whole_buffer(const BufferDesc& d, const ViewRef& v) {
    return !v.transposed && v.row0 == 0 && v.col0 == 0 && v.rows == d.rows && v.cols == d.cols;
}
// Does the node read its output view before writing it (accumulation)?
template<class Elt>
bool reads_output(const Node<Elt>& n) {
    switch (n.op) {
        case Op::GEMM: case Op::BATCHED_GEMM: return n.gop != kernels::GemmOp::Set;
        case Op::SCAL: return n.alpha != 0;
        case Op::GATHER: return n.accumulate;
        case Op::TRSM: case Op::TRANSPOSE: case Op::SCATTER: case Op::PIVOT_SEARCH: case Op::D2H: return false;
        default: return true;
    }
}

template<class Elt>
BufId new_temp(Program<Elt>& prog, uint32_t rows, uint32_t cols, Layout layout, bool fixed, uint64_t key) {
    BufferDesc d; d.kind = BufKind::Temp; d.rows = rows; d.cols = cols; d.layout = layout;
    d.layout_fixed = fixed; d.content_key = key;
    prog.bufs.push_back(d);
    return static_cast<BufId>(prog.bufs.size() - 1);
}

// Fresh, EXACT content identities (never hashes): bit 63 = unique (recorder /
// pass temps), bit 62 = CSE class (index into the CSE table), bit 61 = pool
// block (Recorder::pool_input).  A caller-supplied key must keep bits 61-63
// clear (ENGINE_CHECKed by Recorder::input).
template<class Elt>
uint64_t unique_key(Program<Elt>& prog) { return 0x8000000000000000ull | (++prog.next_key); }

// Rows of B contiguous under this view (the GEMM fast-path requirement)?
template<class Elt>
bool row_contiguous(const Program<Elt>& prog, const ViewRef& v) {
    const Layout l = prog.bufs[v.buf].layout;
    return v.transposed ? (l == Layout::ColMajor) : (l == Layout::RowMajor);
}
// The physical layout that would make `v` row-contiguous.
Layout wanted_layout(const ViewRef& v) { return v.transposed ? Layout::ColMajor : Layout::RowMajor; }

template<class Elt>
double node_cost(const Node<Elt>& n) {
    const double m = n.c.lrows(), nn = n.c.lcols();
    switch (n.op) {
        case Op::GEMM: case Op::SCHUR_UPDATE:
            if (n.kinds) return m * nn * static_cast<double>(n.kb1 != 0 ? n.a.lcols() : n.c.lrows());
            return m * nn * static_cast<double>(n.a.lcols());
        case Op::TRSM:
            return n.pivot_kind == PivotKind::General ? m * nn * static_cast<double>(n.b.lrows()) : m * nn;
        case Op::PANEL_RREF: return m * nn * static_cast<double>(std::min(n.c.lrows(), n.c.lcols()));
        case Op::PIVOT_SEARCH: return static_cast<double>(n.a.lrows()) * n.a.lcols() * std::min(n.a.lrows(), n.a.lcols());
        case Op::COMPARE: return static_cast<double>(n.a.lrows()) * n.a.lcols();
        default: return m * nn;
    }
}

} // namespace

PassOptions PassOptions::from_env() {
    PassOptions o;
    o.batch  = env_flag("ENGINE_BATCH", true);
    o.ksplit = env_flag("ENGINE_KSPLIT", false);
    o.arena  = env_flag("ENGINE_ARENA", true);
    o.cse    = env_flag("ENGINE_CSE", true);
    o.fuse   = env_flag("ENGINE_FUSE", true);
    o.layout = env_flag("ENGINE_LAYOUT", true);
    const char* d = std::getenv("ENGINE_DEVICE");
    o.device_policy = Device::CPU; o.device_model = false; o.policy = DevicePolicy::Cpu;
    if (d != nullptr && std::strcmp(d, "gpu") == 0) {
        o.device_policy = Device::GPU; o.device_model = true; o.policy = DevicePolicy::Gpu;
    } else if (d != nullptr && std::strcmp(d, "hybrid") == 0) {
        o.device_policy = Device::GPU; o.device_model = true; o.policy = DevicePolicy::Hybrid;
    }
    return o;
}

// ---------------------------------------------------------------------------
// Access sets
// ---------------------------------------------------------------------------
template<class Elt>
void node_accesses(const Program<Elt>& prog, const Node<Elt>& n, std::vector<Access>& out) {
    out.clear();
    auto rd = [&](const ViewRef& v) { if (v.valid() && v.buf != kIdentityBuf) out.push_back(Access{v, false}); };
    auto wr = [&](const ViewRef& v) { out.push_back(Access{v, true}); };
    switch (n.op) {
        case Op::GEMM:
            if (n.kinds) { if (n.kb1 != 0) rd(n.a); if (n.kb2 != 0) rd(n.b); }
            else { rd(n.a); rd(n.b); }
            if (n.gop != kernels::GemmOp::Set) rd(n.c);
            wr(n.c);
            break;
        case Op::BATCHED_GEMM:
            for (uint32_t i = n.batch_begin; i < n.batch_end; ++i) {
                const GemmInstance& g = prog.instances[i];
                rd(g.a); rd(g.b);
                if (n.gop != kernels::GemmOp::Set) rd(g.c);
                wr(g.c);
            }
            break;
        case Op::SCHUR_UPDATE: rd(n.a); rd(n.b); rd(n.c); wr(n.c); break;
        case Op::TRSM: if (n.pivot_kind == PivotKind::General) rd(n.a); rd(n.b); wr(n.c); break;
        case Op::AXPY: rd(n.a); rd(n.c); wr(n.c); break;
        case Op::SCAL: if (n.alpha != 0) rd(n.c); wr(n.c); break;
        case Op::TRANSPOSE: case Op::SCATTER: rd(n.a); wr(n.c); break;
        case Op::CANONICALIZE: rd(n.c); wr(n.c); break;
        case Op::GATHER: rd(n.a); if (n.accumulate) rd(n.c); wr(n.c); break;
        case Op::PANEL_RREF: rd(n.c); wr(n.c); break;
        case Op::PIVOT_SEARCH: rd(n.a); wr(n.c); break;
        case Op::COMPARE: rd(n.a); rd(n.b); break;
        case Op::H2D: rd(n.c); break;                 // reads the host copy
        case Op::D2H: wr(n.c); break;                 // writes the host copy
    }
}

// ---------------------------------------------------------------------------
// Structural hash
// ---------------------------------------------------------------------------
namespace {
uint64_t mix_view(uint64_t h, const ViewRef& v) {
    h = fnv_mix(h, v.buf); h = fnv_mix(h, v.row0); h = fnv_mix(h, v.rows);
    h = fnv_mix(h, v.col0); h = fnv_mix(h, v.cols); h = fnv_mix(h, v.transposed ? 1 : 0);
    return h;
}
}

template<class Elt>
uint64_t structural_hash(const Program<Elt>& prog) {
    uint64_t h = fnv_init();
    h = fnv_mix(h, sizeof(Elt));
    h = fnv_mix(h, static_cast<uint64_t>(prog.ctx.regime));
    std::map<uint64_t, uint32_t> canon;
    for (const BufferDesc& d : prog.bufs) {
        h = fnv_mix(h, static_cast<uint64_t>(d.kind)); h = fnv_mix(h, d.rows); h = fnv_mix(h, d.cols);
        h = fnv_mix(h, static_cast<uint64_t>(d.layout)); h = fnv_mix(h, d.layout_fixed ? 1 : 0);
        auto it = canon.find(d.content_key);
        if (it == canon.end()) it = canon.emplace(d.content_key, static_cast<uint32_t>(canon.size())).first;
        h = fnv_mix(h, it->second);
    }
    for (const Node<Elt>& n : prog.nodes) {
        if (n.dead) continue;
        h = fnv_mix(h, static_cast<uint64_t>(n.op));
        h = mix_view(h, n.a); h = mix_view(h, n.b); h = mix_view(h, n.c);
        h = fnv_mix(h, n.alpha); h = fnv_mix(h, n.beta);
        h = fnv_mix(h, n.kinds ? 1 : 0);
        h = fnv_mix(h, n.ka1); h = fnv_mix(h, n.kb1); h = fnv_mix(h, n.ka2); h = fnv_mix(h, n.kb2);
        h = fnv_mix(h, static_cast<uint64_t>(n.gop)); h = fnv_mix(h, n.subtract ? 1 : 0);
        h = fnv_mix(h, n.accumulate ? 1 : 0); h = fnv_mix(h, static_cast<uint64_t>(n.pivot_kind));
        h = fnv_mix(h, n.aux); h = fnv_mix(h, n.index_list); h = fnv_mix(h, n.row_list); h = fnv_mix(h, n.slot_in); h = fnv_mix(h, n.slot_out);
        h = fnv_mix(h, n.batch_begin); h = fnv_mix(h, n.batch_end);
    }
    for (const GemmInstance& g : prog.instances) { h = mix_view(h, g.a); h = mix_view(h, g.b); h = mix_view(h, g.c); }
    for (const auto& l : prog.index_lists) { h = fnv_mix(h, l.size()); for (uint32_t c : l) h = fnv_mix(h, c); }
    return h;
}

template<class Elt>
Program<Elt> Recorder<Elt>::finish() {
    prog_.stats.recorded = static_cast<uint32_t>(prog_.nodes.size());
    for (const Node<Elt>& n : prog_.nodes)
        ENGINE_CHECK_MSG(n.op == Op::COMPARE || prog_.bufs[n.c.buf].kind != BufKind::Input,
                         "tape: a node writes an Input buffer");
    prog_.structural_hash = structural_hash(prog_);
    Program<Elt> out = std::move(prog_);
    prog_ = Program<Elt>();
    prog_.ctx = out.ctx;
    return out;
}

template<class Elt>
std::string Program<Elt>::summary() const {
    std::ostringstream s;
    s << "program: " << bufs.size() << " buffers (" << n_inputs << " in, " << n_outputs << " out), "
      << nodes.size() << " nodes recorded, " << live_nodes() << " live, " << levels.size() << " levels, "
      << "arena " << arena_elts << " elts; passes: fold " << stats.folded_scalar << "+" << stats.folded_axpy
      << ", cse -" << stats.cse_removed << ", transposes +" << stats.transposes_inserted
      << ", ksplit " << stats.ksplit_nodes << ", batches " << stats.batches << " (" << stats.batched_instances
      << " inst), gpu " << stats.gpu_nodes << " (" << stats.gpu_groups << " groups, " << stats.forced_cpu << " forced cpu), "
      << "xfer h2d " << stats.h2d_nodes << " (" << static_cast<uint64_t>(stats.h2d_bytes) << " B, " << stats.hoisted_h2d << " hoisted) d2h "
      << stats.d2h_nodes << " (" << static_cast<uint64_t>(stats.d2h_bytes) << " B)";
    return s.str();
}

// ---------------------------------------------------------------------------
// Pass 1 -- constant-fold on kinds
// ---------------------------------------------------------------------------
namespace {

template<class Elt>
void pass_fold(Program<Elt>& prog, const PassOptions& opts) {
    using GemmOp = kernels::GemmOp;
    const ModCtx<Elt>& ctx = prog.ctx;
    std::vector<Node<Elt>> out;
    out.reserve(prog.nodes.size() * 2);
    for (const Node<Elt>& n0 : prog.nodes) {
        if (n0.dead || !(n0.op == Op::GEMM && n0.kinds)) { out.push_back(n0); continue; }
        const Node<Elt>& n = n0;
        const uint32_t dr = n.c.lrows(), dc = n.c.lcols();
        const uint32_t dk = (n.kb1 != 0) ? n.a.lcols() : (n.kb2 != 0 ? n.b.lrows() : dr);
        auto mk = [&](Op op) { Node<Elt> m; m.op = op; m.origin = n.origin; m.c = n.c; return m; };
        if (!opts.fold) {
            // Unfolded arm: materialise both factors densely, then one plain GEMM.
            auto materialise = [&](Elt ka, Elt kb, const ViewRef& M, uint32_t r, uint32_t c) {
                const BufId t = new_temp(prog, r, c, Layout::RowMajor, false, unique_key(prog));
                Node<Elt> g = mk(Op::GATHER);
                g.c = ViewRef{t, 0, r, 0, c, false};
                g.alpha = ka; g.beta = kb; g.aux = 0; g.index_list = kNoList; g.accumulate = false;
                g.a = (kb != 0) ? M : ViewRef{kIdentityBuf, 0, r, 0, c, false};
                out.push_back(g);
                return g.c;
            };
            const ViewRef A = materialise(n.ka1, n.kb1, n.a, dr, dk);
            const ViewRef B = materialise(n.ka2, n.kb2, n.b, dk, dc);
            Node<Elt> g = mk(Op::GEMM); g.a = A; g.b = B; g.gop = n.gop; g.alpha = Elt(1);
            out.push_back(g);
            continue;
        }
        const Elt s0 = ctx.mul(n.ka1, n.ka2), s1 = ctx.mul(n.ka1, n.kb2), s2 = ctx.mul(n.kb1, n.ka2), s3 = ctx.mul(n.kb1, n.kb2);
        const int nz = (s0 != 0) + (s1 != 0) + (s2 != 0) + (s3 != 0);
        bool sub = (n.gop == GemmOp::Sub);
        if (n.gop == GemmOp::Set) {
            if (nz == 1 && s3 != 0) {
                Node<Elt> g = mk(Op::GEMM); g.a = n.a; g.b = n.b; g.gop = GemmOp::Set; g.alpha = s3;
                out.push_back(g);
                continue;
            }
            Node<Elt> z = mk(Op::SCAL); z.alpha = Elt(0); out.push_back(z);
            sub = false;
        }
        if (s0 != 0) {
            Node<Elt> x = mk(Op::AXPY); x.a = ViewRef{kIdentityBuf, 0, dr, 0, dc, false};
            x.alpha = s0; x.subtract = sub; out.push_back(x); ++prog.stats.folded_scalar;
        }
        if (s1 != 0) {
            Node<Elt> x = mk(Op::AXPY); x.a = n.b; x.alpha = s1; x.subtract = sub; out.push_back(x);
            ++prog.stats.folded_axpy;
        }
        if (s2 != 0) {
            Node<Elt> x = mk(Op::AXPY); x.a = n.a; x.alpha = s2; x.subtract = sub; out.push_back(x);
            ++prog.stats.folded_axpy;
        }
        if (s3 != 0) {
            // product into a temp (CSE-able), then AXPY; pass 2b fuses un-shared ones back
            const BufId t = new_temp(prog, dr, dc, Layout::RowMajor, false, unique_key(prog));
            Node<Elt> g = mk(Op::GEMM); g.a = n.a; g.b = n.b; g.c = ViewRef{t, 0, dr, 0, dc, false};
            g.gop = GemmOp::Set; g.alpha = Elt(1); out.push_back(g);
            Node<Elt> x = mk(Op::AXPY); x.a = g.c; x.alpha = s3; x.subtract = sub; out.push_back(x);
        }
        if (nz == 0 && n.gop == GemmOp::Set) { /* zero fill already emitted */ }
    }
    prog.nodes.swap(out);
}

// ---------------------------------------------------------------------------
// Pass 2 -- CSE + fusion
// ---------------------------------------------------------------------------
struct CseKey {
    std::vector<uint64_t> w;
    bool operator<(const CseKey& o) const { return w < o.w; }
};

template<class Elt>
void write_counts(const Program<Elt>& prog, std::vector<uint32_t>& wc, std::vector<uint32_t>& rc) {
    wc.assign(prog.bufs.size(), 0); rc.assign(prog.bufs.size(), 0);
    std::vector<Access> acc;
    for (const Node<Elt>& n : prog.nodes) {
        if (n.dead) continue;
        node_accesses(prog, n, acc);
        for (const Access& a : acc) { if (a.write) ++wc[a.v.buf]; else ++rc[a.v.buf]; }
    }
}

template<class Elt>
void remap_views(Program<Elt>& prog, const std::vector<BufId>& alias) {
    auto fix = [&](ViewRef& v) { if (v.valid() && v.buf != kIdentityBuf) v.buf = alias[v.buf]; };
    for (Node<Elt>& n : prog.nodes) { fix(n.a); fix(n.b); fix(n.c); }
    for (GemmInstance& g : prog.instances) { fix(g.a); fix(g.b); fix(g.c); }
}

template<class Elt>
void pass_cse(Program<Elt>& prog, const PassOptions& opts) {
    using GemmOp = kernels::GemmOp;
    std::vector<uint32_t> wc, rc;
    write_counts(prog, wc, rc);
    std::vector<BufId> alias(prog.bufs.size());
    for (BufId b = 0; b < alias.size(); ++b) alias[b] = b;
    auto view_key = [&](CseKey& k, const ViewRef& v) {
        if (!v.valid()) { k.w.push_back(0); return; }
        if (v.buf == kIdentityBuf) { k.w.push_back(1); k.w.push_back(v.rows); k.w.push_back(v.cols); return; }
        const BufId b = alias[v.buf];
        k.w.push_back(prog.bufs[b].content_key);
        k.w.push_back(v.row0); k.w.push_back(v.rows); k.w.push_back(v.col0); k.w.push_back(v.cols);
        k.w.push_back(v.transposed ? 1 : 0);
    };
    if (opts.cse) {
        std::map<CseKey, BufId> table;
        for (Node<Elt>& n : prog.nodes) {
            if (n.dead) continue;
            auto fix = [&](ViewRef& v) { if (v.valid() && v.buf != kIdentityBuf) v.buf = alias[v.buf]; };
            fix(n.a); fix(n.b); fix(n.c);
            bool eligible = false;
            switch (n.op) {
                case Op::GEMM:      eligible = (n.gop == GemmOp::Set); break;
                case Op::TRANSPOSE: case Op::SCATTER: eligible = true; break;
                case Op::GATHER:    eligible = !n.accumulate; break;
                default: break;
            }
            const BufferDesc& d = prog.bufs[n.c.buf];
            if (!eligible || d.kind != BufKind::Temp || wc[n.c.buf] != 1 || !whole_buffer(d, n.c)) continue;
            CseKey k;
            k.w.push_back(static_cast<uint64_t>(n.op)); k.w.push_back(n.alpha); k.w.push_back(n.beta);
            k.w.push_back(n.aux); k.w.push_back(n.index_list); k.w.push_back(n.row_list); k.w.push_back(static_cast<uint64_t>(n.gop));
            view_key(k, n.a); view_key(k, n.b);
            k.w.push_back(d.rows); k.w.push_back(d.cols);
            auto it = table.find(k);
            if (it != table.end() && it->second != n.c.buf) {
                alias[n.c.buf] = it->second;
                n.dead = true;
                ++prog.stats.cse_removed;
            } else {
                // exact identity: the CSE class index (the key vector fully
                // describes the producer, recursively through operand keys)
                prog.bufs[n.c.buf].content_key = 0x4000000000000000ull | static_cast<uint64_t>(table.size() + 1);
                table.emplace(k, n.c.buf);
            }
        }
        remap_views(prog, alias);
        write_counts(prog, wc, rc);
    }
    if (opts.fuse) {
        // T = A.B (GEMM Set, alpha 1, whole Temp, single write) read exactly once by
        // AXPY(c +/-= s * T) -> c gop= s * A.B, T unused.
        std::vector<uint32_t> producer(prog.bufs.size(), 0xFFFFFFFFu);
        for (uint32_t i = 0; i < prog.nodes.size(); ++i) {
            const Node<Elt>& n = prog.nodes[i];
            if (n.dead || n.op != Op::GEMM || n.gop != GemmOp::Set || n.alpha != Elt(1)) continue;
            const BufferDesc& d = prog.bufs[n.c.buf];
            if (d.kind != BufKind::Temp || wc[n.c.buf] != 1 || rc[n.c.buf] != 1 || !whole_buffer(d, n.c)) continue;
            auto stable = [&](const ViewRef& v) {
                const BufferDesc& s = prog.bufs[v.buf];
                return s.kind == BufKind::Input || (s.kind == BufKind::Temp && wc[v.buf] <= 1);
            };
            if (!stable(n.a) || !stable(n.b)) continue;
            producer[n.c.buf] = i;
        }
        for (uint32_t j = 0; j < prog.nodes.size(); ++j) {
            Node<Elt>& x = prog.nodes[j];
            if (x.dead || x.op != Op::AXPY || !x.a.valid() || x.a.buf == kIdentityBuf) continue;
            const uint32_t i = producer[x.a.buf];
            if (i == 0xFFFFFFFFu || i >= j || !whole_buffer(prog.bufs[x.a.buf], x.a)) continue;
            Node<Elt>& g = prog.nodes[i];
            Node<Elt> f = g;
            f.c = x.c; f.gop = x.subtract ? GemmOp::Sub : GemmOp::Add; f.alpha = x.alpha; f.origin = x.origin;
            x = f;
            g.dead = true;
            producer[x.a.buf] = 0xFFFFFFFFu;
        }
    }
}

// ---------------------------------------------------------------------------
// Costs (needed by pass 3's votes and pass 6)
// ---------------------------------------------------------------------------
template<class Elt>
void assign_costs(Program<Elt>& prog) {
    for (Node<Elt>& n : prog.nodes) if (!n.dead) n.cost = node_cost(n);
}

// ---------------------------------------------------------------------------
// Pass 3 -- layout assignment + TRANSPOSE insertion (D3 policy)
// ---------------------------------------------------------------------------
template<class Elt>
void pass_layout(Program<Elt>& prog, const PassOptions& opts) {
    if (!opts.layout) return;
    std::vector<std::array<double, 2>> vote(prog.bufs.size(), std::array<double, 2>{0.0, 0.0});
    auto cast = [&](const ViewRef& v, Layout want, double w) {
        if (!v.valid() || v.buf == kIdentityBuf) return;
        vote[v.buf][static_cast<int>(want)] += w;
    };
    for (const Node<Elt>& n : prog.nodes) {
        if (n.dead) continue;
        switch (n.op) {
            case Op::GEMM: case Op::SCHUR_UPDATE:
                cast(n.b, wanted_layout(n.b), n.cost);
                cast(n.c, wanted_layout(n.c), n.cost);
                break;
            case Op::TRSM:
                if (n.pivot_kind == PivotKind::General) { cast(n.b, wanted_layout(n.b), n.cost); cast(n.c, wanted_layout(n.c), n.cost); }
                break;
            case Op::PANEL_RREF: cast(n.c, wanted_layout(n.c), n.cost); break;
            case Op::PIVOT_SEARCH: cast(n.a, wanted_layout(n.a), n.cost); break;
            default: break;
        }
    }
    for (BufId b = 0; b < prog.bufs.size(); ++b) {
        BufferDesc& d = prog.bufs[b];
        if (d.kind != BufKind::Temp || d.layout_fixed) continue;
        const double r = vote[b][0], c = vote[b][1];
        if (r > c) d.layout = Layout::RowMajor;
        else if (c > r) d.layout = Layout::ColMajor;
        // tie: keep the producer's (recorded) preference, RowMajor by default
    }
    // Mismatched GEMM B operands: one physical relayout per (buffer, layout),
    // hoisted to the producer (Temps) or to the program start (Inputs).
    std::vector<uint32_t> wc, rc;
    write_counts(prog, wc, rc);
    std::vector<uint32_t> last_write(prog.bufs.size(), 0xFFFFFFFFu);
    {
        std::vector<Access> acc;
        for (uint32_t i = 0; i < prog.nodes.size(); ++i) {
            if (prog.nodes[i].dead) continue;
            node_accesses(prog, prog.nodes[i], acc);
            for (const Access& a : acc) if (a.write) last_write[a.v.buf] = i;
        }
    }
    std::map<std::pair<BufId, int>, BufId> twins;
    std::vector<std::pair<uint32_t, Node<Elt>>> inserts;   // (insert before node index, node)
    for (uint32_t i = 0; i < prog.nodes.size(); ++i) {
        Node<Elt>& n = prog.nodes[i];
        if (n.dead) continue;
        const bool gemm_like = (n.op == Op::GEMM || n.op == Op::SCHUR_UPDATE ||
                                (n.op == Op::TRSM && n.pivot_kind == PivotKind::General));
        if (!gemm_like || !n.b.valid()) continue;
        if (row_contiguous(prog, n.b)) continue;
        const BufId src = n.b.buf;
        const BufferDesc sd = prog.bufs[src];   // by value: new_temp() below grows prog.bufs
        if (sd.kind == BufKind::Output || wc[src] > 1) continue;       // may be modified later: leave it to the kernel's pack
        if (sd.kind == BufKind::Temp && (last_write[src] == 0xFFFFFFFFu || last_write[src] >= i)) continue;
        const Layout want = wanted_layout(n.b);
        const auto key = std::make_pair(src, static_cast<int>(want));
        auto it = twins.find(key);
        if (it == twins.end()) {
            const BufId t = new_temp(prog, sd.rows, sd.cols, want, true, sd.content_key);
            Node<Elt> tr; tr.op = Op::TRANSPOSE; tr.origin = n.origin;
            tr.a = ViewRef{src, 0, sd.rows, 0, sd.cols, false};
            tr.c = ViewRef{t, 0, sd.rows, 0, sd.cols, false};
            tr.cost = static_cast<double>(sd.rows) * sd.cols;
            const uint32_t pos = (sd.kind == BufKind::Input) ? 0u : last_write[src] + 1;
            inserts.emplace_back(pos, tr);
            it = twins.emplace(key, t).first;
            ++prog.stats.transposes_inserted;
        }
        n.b.buf = it->second;
    }
    if (!inserts.empty()) {
        std::stable_sort(inserts.begin(), inserts.end(),
                         [](const auto& x, const auto& y) { return x.first < y.first; });
        std::vector<Node<Elt>> out;
        out.reserve(prog.nodes.size() + inserts.size());
        size_t k = 0;
        for (uint32_t i = 0; i <= prog.nodes.size(); ++i) {
            while (k < inserts.size() && inserts[k].first == i) out.push_back(inserts[k++].second);
            if (i < prog.nodes.size()) out.push_back(prog.nodes[i]);
        }
        prog.nodes.swap(out);
    }
}

// ---------------------------------------------------------------------------
// Pass 4 -- k-blocking
// ---------------------------------------------------------------------------
template<class Elt>
void pass_kblock(Program<Elt>& prog, const PassOptions& opts) {
    using GemmOp = kernels::GemmOp;
    const uint64_t kmax = std::max<uint64_t>(1, prog.ctx.k_max);
    std::vector<Node<Elt>> out;
    out.reserve(prog.nodes.size());
    for (const Node<Elt>& n : prog.nodes) {
        if (n.dead || !(n.op == Op::GEMM || n.op == Op::SCHUR_UPDATE || (n.op == Op::TRSM && n.pivot_kind == PivotKind::General))) {
            out.push_back(n); continue;
        }
        const uint32_t k = (n.op == Op::TRSM) ? n.b.lrows() : n.a.lcols();
        Node<Elt> m = n;
        m.k_bound = static_cast<uint32_t>(std::min<uint64_t>(k, kmax));
        if (!opts.ksplit || k <= kmax || n.op == Op::TRSM) { out.push_back(m); continue; }
        const uint32_t mm = n.c.lrows(), nn = n.c.lcols();
        for (uint32_t k0 = 0; k0 < k; k0 += static_cast<uint32_t>(kmax)) {
            const uint32_t kk = static_cast<uint32_t>(std::min<uint64_t>(kmax, k - k0));
            Node<Elt> s = m;
            s.a = n.a.sub(0, mm, k0, kk);
            s.b = n.b.sub(k0, kk, 0, nn);
            s.k_bound = kk;
            if (k0 > 0) s.gop = (n.gop == GemmOp::Sub) ? GemmOp::Sub : GemmOp::Add;
            out.push_back(s);
        }
        ++prog.stats.ksplit_nodes;
    }
    prog.nodes.swap(out);
}

// ---------------------------------------------------------------------------
// Dependencies and wavefront levels
// ---------------------------------------------------------------------------
struct Hist { uint32_t node; ViewRef v; bool write; };

template<class Elt>
void analyse_levels(Program<Elt>& prog) {
    std::vector<std::vector<Hist>> hist(prog.bufs.size());
    std::vector<uint32_t> slot_node(prog.n_slots, 0xFFFFFFFFu);
    std::vector<Access> acc;
    for (uint32_t i = 0; i < prog.nodes.size(); ++i) {
        Node<Elt>& n = prog.nodes[i];
        if (n.dead) continue;
        node_accesses(prog, n, acc);
        uint32_t level = 0;
        bool has_dep = false;
        for (const Access& a : acc) {
            for (const Hist& h : hist[a.v.buf]) {
                if ((h.write || a.write) && h.v.overlaps(a.v)) {
                    has_dep = true;
                    level = std::max(level, prog.nodes[h.node].level + 1);
                }
            }
        }
        if (n.slot_in != kNoSlot) {
            const uint32_t p = slot_node[n.slot_in];
            ENGINE_CHECK_MSG(p != 0xFFFFFFFFu, "tape: node reads an unwritten result slot");
            has_dep = true;
            level = std::max(level, prog.nodes[p].level + 1);
        }
        n.level = has_dep ? level : 0;
        if (n.slot_out != kNoSlot) slot_node[n.slot_out] = i;
        for (const Access& a : acc) {
            std::vector<Hist>& hv = hist[a.v.buf];
            if (a.write) {
                // entries fully covered by this write are dominated (any later
                // overlap with them overlaps this write, which is at a higher level)
                const ViewRef& w = a.v;
                hv.erase(std::remove_if(hv.begin(), hv.end(), [&](const Hist& h) {
                    return h.v.row0 >= w.row0 && h.v.row0 + h.v.rows <= w.row0 + w.rows &&
                           h.v.col0 >= w.col0 && h.v.col0 + h.v.cols <= w.col0 + w.cols;
                }), hv.end());
            }
            hv.push_back(Hist{i, a.v, a.write});
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 6 -- device assignment (Phase 5: the real thing)
// ---------------------------------------------------------------------------
//   cpu    : every node CPU (the golden path).
//   gpu    : every device-eligible node GPU.
//   hybrid : the D4 inequality per GROUP of same-key compute nodes of one
//            level (the groups pass 5 will batch), with the measured constants
//            of EngineCostModel.h, the residency of the operands (simulated in
//            level order: an operand a previous GPU group already uses costs
//            no transfer) and one launch per group; memory-class nodes follow
//            the buffers the GPU compute touches, so a zero-fill of a GPU
//            accumulator or an AXPY into it runs on the device instead of
//            forcing a transfer.  Everything decided here is recorded in the
//            node (`device`), so the schedule is a pure function of
//            (program, options): the same program always runs the same way.
struct BatchKey {
    std::array<uint64_t, 12> w{};
    bool operator<(const BatchKey& o) const { return w < o.w; }
};

template<class Elt>
int layout_class(const Program<Elt>& prog, const ViewRef& v) {
    return (prog.bufs[v.buf].layout == Layout::RowMajor ? 0 : 1) + (v.transposed ? 2 : 0);
}

template<class Elt>
bool batchable(const Node<Elt>& n) {
    return !n.dead && (n.op == Op::GEMM || n.op == Op::SCHUR_UPDATE) &&
           n.c.lrows() != 0 && n.c.lcols() != 0 && n.a.lcols() != 0;
}
template<class Elt>
BatchKey batch_key(const Program<Elt>& prog, const Node<Elt>& n, bool with_device) {
    BatchKey k;
    k.w[0] = static_cast<uint64_t>(n.op); k.w[1] = static_cast<uint64_t>(n.gop); k.w[2] = n.alpha;
    k.w[3] = n.c.lrows(); k.w[4] = n.c.lcols(); k.w[5] = n.a.lcols();
    k.w[6] = static_cast<uint64_t>(layout_class(prog, n.a)); k.w[7] = static_cast<uint64_t>(layout_class(prog, n.b));
    k.w[8] = static_cast<uint64_t>(layout_class(prog, n.c)); k.w[9] = sizeof(Elt);
    k.w[10] = n.k_bound; k.w[11] = with_device ? static_cast<uint64_t>(n.device) : 0;
    return k;
}

template<class Elt>
bool compute_node(const Node<Elt>& n) {
    return n.op == Op::GEMM || n.op == Op::SCHUR_UPDATE || n.op == Op::BATCHED_GEMM ||
           (n.op == Op::TRSM && n.pivot_kind == PivotKind::General);
}

template<class Elt>
double buf_bytes(const Program<Elt>& prog, BufId b) {
    const BufferDesc& d = prog.bufs[b];
    return static_cast<double>(d.rows) * d.cols * sizeof(Elt);
}

template<class Elt>
void pass_device(Program<Elt>& prog, const PassOptions& opts) {
    const DevicePolicy pol = opts.effective_policy();
    for (Node<Elt>& n : prog.nodes) if (!n.dead) n.device = Device::CPU;
    if (pol == DevicePolicy::Cpu) return;
    if (pol == DevicePolicy::Gpu) {
        for (Node<Elt>& n : prog.nodes) if (!n.dead && device_eligible_node(n)) n.device = Device::GPU;
        return;
    }
    // ---- hybrid ----
    const CostConstants& K = cost_constants();
    const bool u64 = sizeof(Elt) == 8;
    const Regime regime = prog.ctx.regime;
    const double r_cpu_dense = u64 ? K.cpu_modmac_u64_1t * 16.0 : K.cpu_modmac_16t;
    const double r_cpu_small = u64 ? K.cpu_modmac_u64_1t * 16.0 : K.cpu_modmac_16t_small;
    double r_gpu_dense = K.gpu_modmac, r_gpu_small = K.gpu_modmac_small;
    if (u64) r_gpu_dense = r_gpu_small = K.gpu_modmac_u64;
    else if (regime == Regime::U32_Shoup) r_gpu_dense = r_gpu_small = K.gpu_modmac_shoup;
    else if (regime == Regime::U32_Barrett64) r_gpu_dense = r_gpu_small = K.gpu_modmac_barrett64;

    // groups: (level, key without device) -> members, in first-member order
    struct Group { uint32_t level; std::vector<uint32_t> members; };
    std::vector<Group> groups;
    {
        std::map<std::pair<uint32_t, BatchKey>, uint32_t> gidx;
        for (uint32_t i = 0; i < prog.nodes.size(); ++i) {
            const Node<Elt>& n = prog.nodes[i];
            if (n.dead || !compute_node(n)) continue;
            if (batchable(n)) {
                const auto key = std::make_pair(n.level, batch_key(prog, n, false));
                auto it = gidx.find(key);
                if (it == gidx.end()) { it = gidx.emplace(key, static_cast<uint32_t>(groups.size())).first; groups.push_back(Group{n.level, {}}); }
                groups[it->second].members.push_back(i);
            } else {
                groups.push_back(Group{n.level, {i}});
            }
        }
        std::stable_sort(groups.begin(), groups.end(), [](const Group& x, const Group& y) {
            if (x.level != y.level) return x.level < y.level;
            return x.members[0] < y.members[0];
        });
    }
    // Simulated residency in level order: resident[b] = the device copy is
    // current; dev_newer[b] = the device copy is the ONLY current one (a GPU
    // group wrote it since the last CPU write).  Choosing the CPU for a group
    // whose operands are dev_newer costs their download, choosing the GPU for
    // one whose operands are not resident costs their upload -- both sides of
    // the D4 inequality carry the transfer term, so an accumulator that went
    // to the device stays there instead of ping-ponging level by level.
    std::vector<char> resident(prog.bufs.size(), 0), dev_newer(prog.bufs.size(), 0), d2h_charged(prog.bufs.size(), 0);
    std::vector<Access> acc;
    // Upload charges that the memory-class sweep below would remove anyway:
    //   * a buffer whose FIRST access is a whole-buffer Set write by a
    //     memory-class node (zero(), a SCATTER copy) needs no upload -- that
    //     writer follows the compute onto the device (`free_first[b]`);
    //   * a relayout twin (pass 3's TRANSPOSE of an Input) costs its source's
    //     bytes, once: `twin_of[b]` = the Input it copies.
    std::vector<char> free_first(prog.bufs.size(), 0);
    std::vector<BufId> twin_of(prog.bufs.size(), kNoBuf);
    {
        std::vector<char> seen(prog.bufs.size(), 0);
        std::vector<uint32_t> order;
        for (uint32_t i = 0; i < prog.nodes.size(); ++i) if (!prog.nodes[i].dead) order.push_back(i);
        std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return prog.nodes[x].level < prog.nodes[y].level; });
        for (uint32_t i : order) {
            const Node<Elt>& n = prog.nodes[i];
            node_accesses(prog, n, acc);
            for (const Access& a : acc) {
                const BufId b = a.v.buf;
                if (!seen[b]) {
                    seen[b] = 1;
                    if (a.write && !compute_node(n) && device_eligible_node(n) && whole_buffer(prog.bufs[b], a.v) && !reads_output(n)) {
                        free_first[b] = 1;
                        if ((n.op == Op::TRANSPOSE || n.op == Op::SCATTER) && n.a.valid() && n.a.buf != kIdentityBuf &&
                            prog.bufs[n.a.buf].kind == BufKind::Input && whole_buffer(prog.bufs[n.a.buf], n.a))
                            twin_of[b] = n.a.buf;
                    }
                }
            }
        }
    }
    double t_cpu_total = 0, t_gpu_total = 0;
    const bool trace = std::getenv("ENGINE_TRACE_PASSES") != nullptr;
    for (const Group& g : groups) {
        double W = 0, bytes_h2d = 0, bytes_d2h = 0, bytes_back = 0, max_tiles = 0;
        std::vector<BufId> touched, written, sources;
        for (uint32_t i : g.members) {
            const Node<Elt>& n = prog.nodes[i];
            W += n.cost;
            node_accesses(prog, n, acc);
            for (const Access& a : acc) { touched.push_back(a.v.buf); if (a.write) written.push_back(a.v.buf); }
            const double tiles = (n.op == Op::BATCHED_GEMM) ? 0.0
                : std::ceil(n.c.lrows() / 64.0) * std::ceil(n.c.lcols() / 64.0);
            max_tiles = std::max(max_tiles, tiles);
        }
        auto uniq = [](std::vector<BufId>& v) { std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); };
        uniq(touched); uniq(written);
        for (BufId b : touched) {
            if (!resident[b]) {
                if (twin_of[b] != kNoBuf) { if (!resident[twin_of[b]]) sources.push_back(twin_of[b]); }
                else if (!free_first[b]) bytes_h2d += buf_bytes(prog, b);
            }
            if (dev_newer[b]) bytes_back += buf_bytes(prog, b);
            if (prog.bufs[b].kind == BufKind::Output && !d2h_charged[b]) bytes_d2h += buf_bytes(prog, b);
        }
        uniq(sources);
        for (BufId b : sources) if (std::find(touched.begin(), touched.end(), b) == touched.end()) bytes_h2d += buf_bytes(prog, b);
        const bool batched = g.members.size() >= 2;
        double t_cpu = W / (batched ? r_cpu_small : r_cpu_dense) + static_cast<double>(g.members.size()) * K.cpu_node_s;
        t_cpu += bytes_back / K.d2h_bytes_s;
        double t_gpu;
        if (batched) t_gpu = W / r_gpu_small + K.gpu_launch_s;
        else t_gpu = std::max(W / (max_tiles >= K.gpu_dense_tiles ? r_gpu_dense : r_gpu_small), K.gpu_kernel_floor_s) + K.gpu_launch_s;
        t_gpu += bytes_h2d / K.h2d_bytes_s + bytes_d2h / K.d2h_bytes_s;
        const bool regime_gpu = u64 && regime != Regime::U32_Delayed;   // ell >= 2^32: always GPU (D4)
        const bool gpu = regime_gpu || (t_gpu < t_cpu);
        if (trace)
            std::cout << "    [pass6] level " << g.level << " group of " << g.members.size() << " " << op_name(prog.nodes[g.members[0]].op)
                      << " W=" << W << " h2d=" << bytes_h2d << " d2h=" << bytes_d2h << " back=" << bytes_back
                      << " t_cpu=" << t_cpu * 1e3 << "ms t_gpu=" << t_gpu * 1e3 << "ms -> " << (gpu ? "GPU" : "CPU") << std::endl;
        if (gpu) {
            for (uint32_t i : g.members) prog.nodes[i].device = Device::GPU;
            for (BufId b : touched) { resident[b] = 1; if (prog.bufs[b].kind == BufKind::Output) d2h_charged[b] = 1; }
            for (BufId b : sources) resident[b] = 1;
            for (BufId b : written) dev_newer[b] = 1;
            t_gpu_total += t_gpu; ++prog.stats.gpu_groups;
        } else {
            for (BufId b : touched) dev_newer[b] = 0;      // downloaded (charged above)
            for (BufId b : written) resident[b] = 0;       // the device copy is stale now
            t_cpu_total += t_cpu;
        }
    }
    // memory-class nodes follow the GPU-touched buffers
    for (Node<Elt>& n : prog.nodes) {
        if (n.dead || compute_node(n) || !device_eligible_node(n)) continue;
        node_accesses(prog, n, acc);
        bool all = !acc.empty();
        for (const Access& a : acc) if (!resident[a.v.buf]) all = false;
        if (all) n.device = Device::GPU;
    }
    prog.stats.model_t_cpu = t_cpu_total; prog.stats.model_t_gpu = t_gpu_total;
}

// ---------------------------------------------------------------------------
// Pass 7 -- residency: transfer nodes (Phase 5)
// ---------------------------------------------------------------------------
// Whole-buffer residency per program buffer, simulated level by level:
//   Host  (host copy newer)  -> a GPU access needs an H2D before the level
//   Device (device newer)    -> a CPU access needs a D2H before the level,
//                               unless it is a whole-buffer Set write
//   Both / None              -> nothing
// After a level, the writers set the state; two devices writing one buffer
// in one level would leave neither copy whole, so such GPU writers are moved
// to the CPU first (fixed point; stats.forced_cpu).  H2D nodes are hoisted
// to the level after the buffer's last host write (level 0 for Inputs) so
// the DMA overlaps the CPU levels in between; D2H nodes stay at the level of
// the first CPU reader.  Outputs left on the device are downloaded in an
// epilogue level.  Transfer nodes are ordinary nodes (op H2D / D2H, view c =
// the whole buffer, device GPU, cost = bytes) appended to prog.nodes, so they
// are visible in the program and counted by node_accesses / lifetimes.
template<class Elt>
void pass_residency(Program<Elt>& prog, const PassOptions& /*opts*/) {
    bool any_gpu = false;
    for (const Node<Elt>& n : prog.nodes) if (!n.dead && n.device == Device::GPU) { any_gpu = true; break; }
    if (!any_gpu) return;
    uint32_t nl = 0;
    for (const Node<Elt>& n : prog.nodes) if (!n.dead) nl = std::max(nl, n.level + 1);
    std::vector<std::vector<uint32_t>> levels(nl);
    for (uint32_t i = 0; i < prog.nodes.size(); ++i) if (!prog.nodes[i].dead) levels[prog.nodes[i].level].push_back(i);
    std::vector<Access> acc;
    // 1. conflicts: mixed-device writers of one buffer in one level -> CPU
    for (bool changed = true; changed;) {
        changed = false;
        for (const auto& lv : levels) {
            std::map<BufId, std::pair<bool, bool>> writers;   // buf -> (cpu writes, gpu writes)
            for (uint32_t i : lv) {
                const Node<Elt>& n = prog.nodes[i];
                node_accesses(prog, n, acc);
                for (const Access& a : acc) if (a.write) {
                    auto& w = writers[a.v.buf];
                    if (n.device == Device::GPU) w.second = true; else w.first = true;
                }
            }
            for (uint32_t i : lv) {
                Node<Elt>& n = prog.nodes[i];
                if (n.device != Device::GPU) continue;
                node_accesses(prog, n, acc);
                for (const Access& a : acc) if (a.write && writers[a.v.buf].first) {
                    n.device = Device::CPU; ++prog.stats.forced_cpu; changed = true; break;
                }
            }
        }
    }
    // 2. simulate residency, collect transfers
    enum class St : uint8_t { None, Host, Device, Both };
    std::vector<St> st(prog.bufs.size(), St::None);
    for (BufId b = 0; b < prog.bufs.size(); ++b)
        if (prog.bufs[b].kind != BufKind::Temp) st[b] = St::Host;
    std::vector<uint32_t> last_host_write(prog.bufs.size(), 0xFFFFFFFFu);   // level of the last CPU write
    struct Xfer { Op op; BufId buf; uint32_t level; };
    std::vector<Xfer> xfers;
    for (uint32_t L = 0; L < nl; ++L) {
        std::vector<BufId> h2d, d2h, gpu_w, cpu_w;
        for (uint32_t i : levels[L]) {
            const Node<Elt>& n = prog.nodes[i];
            node_accesses(prog, n, acc);
            for (const Access& a : acc) {
                const BufId b = a.v.buf;
                // a whole-buffer write with Set semantics replaces every byte: the
                // stale copy on the writing side needs no transfer first
                const bool whole_set = a.write && whole_buffer(prog.bufs[b], a.v) && !reads_output(n);
                if (n.device == Device::GPU) {
                    if (st[b] == St::Host && !whole_set) h2d.push_back(b);
                    if (a.write) gpu_w.push_back(b);
                } else {
                    if (st[b] == St::Device && !whole_set) d2h.push_back(b);
                    if (a.write) cpu_w.push_back(b);
                }
            }
        }
        auto uniq = [](std::vector<BufId>& v) { std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); };
        uniq(h2d); uniq(d2h); uniq(gpu_w); uniq(cpu_w);
        for (BufId b : h2d) {
            // hoist: after the last host write (Inputs: level 0)
            uint32_t at = (last_host_write[b] == 0xFFFFFFFFu) ? 0u : last_host_write[b] + 1;
            at = std::min(at, L);
            if (at < L) ++prog.stats.hoisted_h2d;
            xfers.push_back(Xfer{Op::H2D, b, at});
            st[b] = St::Both;
        }
        for (BufId b : d2h) { xfers.push_back(Xfer{Op::D2H, b, L}); st[b] = St::Both; }
        for (BufId b : gpu_w) st[b] = St::Device;
        for (BufId b : cpu_w) { st[b] = St::Host; last_host_write[b] = L; }
    }
    for (BufId b = 0; b < prog.bufs.size(); ++b)
        if (prog.bufs[b].kind == BufKind::Output && st[b] == St::Device) xfers.push_back(Xfer{Op::D2H, b, nl});
    std::stable_sort(xfers.begin(), xfers.end(), [](const Xfer& x, const Xfer& y) {
        if (x.level != y.level) return x.level < y.level;
        if (x.op != y.op) return x.op == Op::H2D;   // uploads before downloads inside a level
        return x.buf < y.buf;
    });
    for (const Xfer& x : xfers) {
        const BufferDesc& d = prog.bufs[x.buf];
        Node<Elt> n; n.op = x.op; n.device = Device::GPU; n.level = x.level;
        n.c = ViewRef{x.buf, 0, d.rows, 0, d.cols, false};
        n.cost = buf_bytes(prog, x.buf);
        n.origin = 0xFFFFFFFFu;
        prog.nodes.push_back(n);
        if (x.op == Op::H2D) { ++prog.stats.h2d_nodes; prog.stats.h2d_bytes += n.cost; }
        else { ++prog.stats.d2h_nodes; prog.stats.d2h_bytes += n.cost; }
    }
}

// ---------------------------------------------------------------------------
// Pass 5 -- batching
// ---------------------------------------------------------------------------
template<class Elt>
void pass_batch(Program<Elt>& prog, const PassOptions& opts) {
    if (!opts.batch) return;
    std::map<std::pair<uint32_t, BatchKey>, std::vector<uint32_t>> groups;
    for (uint32_t i = 0; i < prog.nodes.size(); ++i) {
        const Node<Elt>& n = prog.nodes[i];
        if (!batchable(n)) continue;
        groups[std::make_pair(n.level, batch_key(prog, n, true))].push_back(i);
    }
    for (auto& g : groups) {
        std::vector<uint32_t>& members = g.second;
        if (members.size() < 2) continue;
        std::vector<GemmInstance> inst;
        inst.reserve(members.size());
        for (uint32_t i : members) inst.push_back(GemmInstance{prog.nodes[i].a, prog.nodes[i].b, prog.nodes[i].c});
        std::stable_sort(inst.begin(), inst.end(), [](const GemmInstance& x, const GemmInstance& y) {
            if (x.c.buf != y.c.buf) return x.c.buf < y.c.buf;
            if (x.c.row0 != y.c.row0) return x.c.row0 < y.c.row0;
            return x.c.col0 < y.c.col0;
        });
        Node<Elt> bnode = prog.nodes[members[0]];
        bnode.op = Op::BATCHED_GEMM;
        bnode.batch_begin = static_cast<uint32_t>(prog.instances.size());
        prog.instances.insert(prog.instances.end(), inst.begin(), inst.end());
        bnode.batch_end = static_cast<uint32_t>(prog.instances.size());
        bnode.cost = 0;
        for (uint32_t i : members) { bnode.cost += prog.nodes[i].cost; prog.nodes[i].dead = true; }
        bnode.a = bnode.b = bnode.c = ViewRef{};
        // The batch node takes its first member's slot and the members' common
        // level (see compile()).
        bnode.level = g.first.first;
        prog.nodes[members[0]] = bnode;
        ++prog.stats.batches;
        prog.stats.batched_instances += static_cast<uint32_t>(members.size());
    }
}

// ---------------------------------------------------------------------------
// Schedule: levels, lifetimes, arena
// ---------------------------------------------------------------------------
// (slot-9 WP-7, lever L3) The arena's first-fit placement in O(log L) per
// temp, L the number of live temps.  THE PLACEMENT RULE IS UNCHANGED; it is
// stated once here and implemented twice (the linear reference below it):
//   * the temps are visited in (first_level, footprint desc, id) order;
//   * before a temp is placed, every live temp with last_level < its
//     first_level is released;
//   * the temp goes to the LOWEST offset `off` in {0} u {the 16-aligned end of
//     a live temp} with [off, off + footprint) meeting no live temp, i.e. to
//     the start of the leftmost gap (between 0 or a live temp's aligned end
//     and the next live temp's start) of length >= footprint, or after the
//     last live temp if there is none.
// The pre-WP-7 code kept the live set in an offset-sorted vector and walked it
// (and its expiry, a remove_if) linearly per temp: O(T * L).  At U_10007,
// n = 18, the fused program 1 at q = 1 has ~10^6 temps, and that walk was
// ~17 s of an ~18 s compile (slot9/WP-7_report.md, the profile table).  Here
// the live temps sit in a treap keyed by offset whose every node carries the
// gap before it and its subtree's maximum gap, so "the leftmost gap >= k" is
// one descent; the expiry is a min-heap on last_level.  Every answer is a
// function of the live SET (never of the treap's shape or its priorities),
// and the set evolves exactly as the vector did, so every arena_offset --
// hence the compiled program -- is identical to the linear scan's.
// Footprints and offsets are multiples of 16 (ld is rounded to 16), so no gap
// is negative.  Under !NDEBUG schedule() also runs ArenaFirstFitLinear and
// ENGINE_DEBUG_CHECKs every offset against it; EngineArenaFirstFitValidation
// (Registry/Slot9/WP7.cpp) re-derives every offset of randomized programs with
// an independent linear scan, in every build.
class ArenaFirstFit {
public:
    static constexpr size_t kAlign = 16;
    static size_t aligned_end(size_t off, size_t size) { return (off + size + kAlign - 1) / kAlign * kAlign; }

    // Release every live temp with last < level.
    void release_before(uint32_t level) {
        while (!expiry_.empty() && expiry_.top().first < level) {
            const size_t off = expiry_.top().second;
            expiry_.pop();
            erase(off);
        }
    }
    // Place [off, off + size) at the first fit (size > 0); returns off.
    size_t place(size_t size, uint32_t last) {
        size_t off = 0;
        const uint32_t hit = leftmost_gap_at_least(root_, size);
        if (hit != kNil) off = pool_[hit].off - pool_[hit].gap;   // the gap starts at the predecessor's aligned end (or 0)
        else if (!by_off_.empty()) off = aligned_end(by_off_.rbegin()->first, pool_[by_off_.rbegin()->second].size);
        insert(off, size);
        expiry_.push(std::make_pair(last, off));
        return off;
    }

private:
    static constexpr uint32_t kNil = 0xFFFFFFFFu;
    struct TNode { size_t off, size, gap, mx; uint32_t l, r, pri; };
    std::vector<TNode> pool_;
    std::vector<uint32_t> free_;
    uint32_t root_ = kNil;
    uint32_t seed_ = 0x9E3779B9u;          // fixed seed: priorities shape the treap only, never an answer
    std::map<size_t, uint32_t> by_off_;    // offset -> node: the neighbours of an offset
    std::priority_queue<std::pair<uint32_t, size_t>, std::vector<std::pair<uint32_t, size_t>>,
                        std::greater<std::pair<uint32_t, size_t>>> expiry_;   // (last, offset), smallest first

    size_t mx(uint32_t t) const { return t == kNil ? 0 : pool_[t].mx; }
    void pull(uint32_t t) { TNode& n = pool_[t]; n.mx = std::max(n.gap, std::max(mx(n.l), mx(n.r))); }
    uint32_t next_pri() { seed_ ^= seed_ << 13; seed_ ^= seed_ >> 17; seed_ ^= seed_ << 5; return seed_; }
    // t -> (keys < key, keys >= key)
    void split(uint32_t t, size_t key, uint32_t& a, uint32_t& b) {
        if (t == kNil) { a = b = kNil; return; }
        if (pool_[t].off < key) { split(pool_[t].r, key, pool_[t].r, b); a = t; }
        else { split(pool_[t].l, key, a, pool_[t].l); b = t; }
        pull(t);
    }
    uint32_t merge(uint32_t a, uint32_t b) {
        if (a == kNil) return b;
        if (b == kNil) return a;
        if (pool_[a].pri > pool_[b].pri) { pool_[a].r = merge(pool_[a].r, b); pull(a); return a; }
        pool_[b].l = merge(a, pool_[b].l); pull(b); return b;
    }
    // The in-order-first node whose gap is >= k, or kNil.
    uint32_t leftmost_gap_at_least(uint32_t t, size_t k) const {
        while (t != kNil) {
            if (mx(pool_[t].l) >= k) t = pool_[t].l;
            else if (pool_[t].gap >= k) return t;
            else if (mx(pool_[t].r) >= k) t = pool_[t].r;
            else return kNil;
        }
        return kNil;
    }
    void set_gap(size_t off, size_t gap) {   // one node's gap; the maxima above it re-pulled
        uint32_t a, b, c;
        split(root_, off, a, b);
        split(b, off + 1, b, c);
        ENGINE_CHECK(b != kNil);
        pool_[b].gap = gap; pull(b);
        root_ = merge(merge(a, b), c);
    }
    void insert(size_t off, size_t size) {
        const auto nx = by_off_.lower_bound(off);
        ENGINE_CHECK(nx == by_off_.end() || nx->first != off);
        size_t pred_end = 0;
        if (nx != by_off_.begin()) { const auto pv = std::prev(nx); pred_end = aligned_end(pv->first, pool_[pv->second].size); }
        ENGINE_CHECK(pred_end <= off);
        uint32_t id;
        if (!free_.empty()) { id = free_.back(); free_.pop_back(); }
        else { id = static_cast<uint32_t>(pool_.size()); pool_.emplace_back(); }
        pool_[id] = TNode{off, size, off - pred_end, off - pred_end, kNil, kNil, next_pri()};
        uint32_t a, b;
        split(root_, off, a, b);
        root_ = merge(merge(a, id), b);
        if (nx != by_off_.end()) {
            ENGINE_CHECK(aligned_end(off, size) <= nx->first);
            set_gap(nx->first, nx->first - aligned_end(off, size));
        }
        by_off_.emplace(off, id);
    }
    void erase(size_t off) {
        const auto it = by_off_.find(off);
        ENGINE_CHECK(it != by_off_.end());
        size_t pred_end = 0;
        if (it != by_off_.begin()) { const auto pv = std::prev(it); pred_end = aligned_end(pv->first, pool_[pv->second].size); }
        const auto nx = std::next(it);
        const bool has_nx = nx != by_off_.end();
        const size_t nx_off = has_nx ? nx->first : 0;
        uint32_t a, b, c;
        split(root_, off, a, b);
        split(b, off + 1, b, c);
        ENGINE_CHECK(b == it->second);
        free_.push_back(b);
        root_ = merge(a, c);
        by_off_.erase(it);
        if (has_nx) set_gap(nx_off, nx_off - pred_end);
    }
};

#ifndef NDEBUG
// The pre-WP-7 linear scan, its rule verbatim: the debug cross-check.
struct ArenaFirstFitLinear {
    struct Live { size_t off, size; uint32_t last; };
    std::vector<Live> live;   // sorted by offset
    size_t place(size_t footprint, uint32_t first, uint32_t last) {
        live.erase(std::remove_if(live.begin(), live.end(), [&](const Live& l) { return l.last < first; }), live.end());
        size_t off = 0;
        for (size_t i = 0; i <= live.size(); ++i) {
            const size_t end = (i < live.size()) ? live[i].off : ~static_cast<size_t>(0);
            if (end - off >= footprint) {
                live.insert(live.begin() + static_cast<long>(i), Live{off, footprint, last});
                return off;
            }
            if (i < live.size()) off = ArenaFirstFit::aligned_end(live[i].off, live[i].size);
        }
        ENGINE_CHECK(false);
        return 0;
    }
};
#endif

template<class Elt>
void schedule(Program<Elt>& prog, const PassOptions& opts) {
    uint32_t nl = 0;
    for (const Node<Elt>& n : prog.nodes) if (!n.dead) nl = std::max(nl, n.level + 1);
    prog.levels.assign(nl, {});
    for (uint32_t i = 0; i < prog.nodes.size(); ++i)
        if (!prog.nodes[i].dead) prog.levels[prog.nodes[i].level].push_back(i);
    for (BufferDesc& d : prog.bufs) { d.first_level = 0xFFFFFFFFu; d.last_level = 0; d.footprint = 0; d.arena_offset = 0; }
    std::vector<Access> acc;
    for (const Node<Elt>& n : prog.nodes) {
        if (n.dead) continue;
        node_accesses(prog, n, acc);
        for (const Access& a : acc) {
            BufferDesc& d = prog.bufs[a.v.buf];
            d.first_level = std::min(d.first_level, n.level);
            d.last_level  = std::max(d.last_level, n.level);
        }
    }
    // footprints (ld rounded to 16 elements: rows / columns start 64-byte aligned)
    std::vector<BufId> temps;
    size_t total = 0;
    for (BufId b = 0; b < prog.bufs.size(); ++b) {
        BufferDesc& d = prog.bufs[b];
        if (d.kind != BufKind::Temp) continue;
        const uint32_t minor = (d.layout == Layout::RowMajor) ? d.cols : d.rows;
        const uint32_t major = (d.layout == Layout::RowMajor) ? d.rows : d.cols;
        d.ld = (minor + 15u) & ~15u;
        d.footprint = (d.rows == 0 || d.cols == 0 || d.first_level == 0xFFFFFFFFu) ? 0 : static_cast<size_t>(major) * d.ld;
        total += d.footprint;
        if (d.footprint > 0) temps.push_back(b);
    }
    prog.stats.temp_elts = total;
    std::stable_sort(temps.begin(), temps.end(), [&](BufId x, BufId y) {
        const BufferDesc& a = prog.bufs[x]; const BufferDesc& b = prog.bufs[y];
        if (a.first_level != b.first_level) return a.first_level < b.first_level;
        if (a.footprint != b.footprint) return a.footprint > b.footprint;
        return x < y;
    });
    size_t arena = 0;
    const size_t align = ArenaFirstFit::kAlign;
    ArenaFirstFit fit;   // (slot-9 WP-7, L3) see ArenaFirstFit: the same rule, O(log L) per temp
#ifndef NDEBUG
    ArenaFirstFitLinear fit_ref;
#endif
    for (BufId b : temps) {
        BufferDesc& d = prog.bufs[b];
        if (opts.arena) {
            fit.release_before(d.first_level);
            const size_t off = fit.place(d.footprint, d.last_level);
#ifndef NDEBUG
            ENGINE_DEBUG_CHECK_MSG(fit_ref.place(d.footprint, d.first_level, d.last_level) == off,
                                   "tape schedule: the arena first-fit differs from the linear-scan reference");
#endif
            d.arena_offset = off;
        } else {
            d.arena_offset = arena;
        }
        arena = std::max(arena, (d.arena_offset + d.footprint + align - 1) / align * align);
    }
    prog.arena_elts = arena;
    prog.stats.arena_elts = arena;
    prog.stats.levels = nl;
    prog.stats.live_nodes = prog.live_nodes();
}

} // namespace

uint64_t next_program_uid() {
    static std::atomic<uint64_t> c{0};
    return ++c;
}

template<class Elt>
void compile(Program<Elt>& prog, const PassOptions& opts) {
    if (prog.compiled) return;
    struct CompileTrace {
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~CompileTrace() { if (trace_nodes_enabled()) trace_add_compile(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()); }
    } ctrace;
    prog.opts = opts;
    pass_fold(prog, opts);          // 1
    assign_costs(prog);
    pass_cse(prog, opts);           // 2 (+ fusion)
    assign_costs(prog);
    pass_layout(prog, opts);        // 3
    pass_kblock(prog, opts);        // 4
    assign_costs(prog);
    analyse_levels(prog);
    pass_device(prog, opts);        // 6 -- per group of would-be batch members (the batching key needs it)
    pass_batch(prog, opts);         // 5 -- a batch keeps its members' level: they were mutually
                                    //      independent with all deps below and all dependents
                                    //      above, so no other level moves (list order is NOT a
                                    //      topological order after this point; nothing uses it)
    pass_residency(prog, opts);     // 7 -- conflicts, transfer nodes, epilogue downloads
    prog.stats.gpu_nodes = 0; prog.stats.gpu_compute_nodes = 0;
    for (const Node<Elt>& n : prog.nodes) if (!n.dead && n.device == Device::GPU && !is_transfer(n.op)) {
        ++prog.stats.gpu_nodes;
        if (compute_node(n)) ++prog.stats.gpu_compute_nodes;
    }
    schedule(prog, opts);
    prog.uid = next_program_uid();
    prog.compiled = true;
}

template<class Elt>
void compile(Program<Elt>& prog) { compile(prog, PassOptions::from_env()); }

#define TAPE_PASSES_INSTANTIATE(Elt)                                                             \
    template void compile<Elt>(Program<Elt>&, const PassOptions&);                               \
    template void compile<Elt>(Program<Elt>&);                                                   \
    template void node_accesses<Elt>(const Program<Elt>&, const Node<Elt>&, std::vector<Access>&); \
    template uint64_t structural_hash<Elt>(const Program<Elt>&);                                 \
    template Program<Elt> Recorder<Elt>::finish();                                               \
    template std::string Program<Elt>::summary() const;
TAPE_PASSES_INSTANTIATE(uint32_t)
TAPE_PASSES_INSTANTIATE(uint64_t)
#undef TAPE_PASSES_INSTANTIATE

} // namespace tape
} // namespace engine
