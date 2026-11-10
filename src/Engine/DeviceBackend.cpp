// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DeviceBackend.cpp -- see DeviceBackend.h.
#include "DeviceBackend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>

#include "../Cuda/CudaCtx.h"
#include "../Cuda/CudaKernels.h"
#include "../Cuda/DeviceBuf.h"
#include "Passes.h"

namespace engine {
namespace tape {

namespace cu = engine::cuda;
using kernels::GemmOp;

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// ENGINE_CUDA_GRAPH: 0 = never capture, 1 (default) = capture on the second
// run of a program (the first run of a one-shot program is not worth a
// capture), 2 = capture on the first run.  A tuning knob: never changes a value.
int graph_mode() {
    const char* e = std::getenv("ENGINE_CUDA_GRAPH");   // re-read per level: the validations switch it per arm
    if (e == nullptr || e[0] == '\0') return 1;
    return std::atoi(e);
}
size_t program_cache_cap() {
    static const size_t c = [] {
        const char* e = std::getenv("ENGINE_CUDA_PROGRAM_CACHE");
        const long v = (e != nullptr) ? std::atol(e) : 0;
        return v <= 0 ? static_cast<size_t>(64) : static_cast<size_t>(v);
    }();
    return c;
}
bool trace() {
    static const bool t = std::getenv("ENGINE_TRACE_DEVICE") != nullptr;
    return t;
}
// ENGINE_CUDA_MERGE_LEVELS=0 disables the merging of consecutive GPU-only
// wavefronts into one graph (a tuning knob; never changes a value).
bool merge_levels() {
    const char* e = std::getenv("ENGINE_CUDA_MERGE_LEVELS");
    return !(e != nullptr && e[0] == '0');
}

constexpr int kStreams = 3;
// (slot-22 G1, audit R1) the stream slot of the t-th GPU node of a level with nlev GPU nodes.
// The ONE formula: `build` (which deals each IMMA-eligible node its slot's workspace) and
// `enqueue_level` (which picks the node's stream) both call it, and `enqueue` checks that the
// node's imma_slot equals the slot it is launched on.
inline int stream_slot(size_t t, size_t nlev) {
    const int ns = static_cast<int>(std::min<size_t>(nlev, kStreams));
    return ns > 1 ? static_cast<int>(t % static_cast<size_t>(ns)) : 0;
}
// (slot-22 G1) the cap of one IMMA block: a GEMM whose whole-K workspace is larger runs in K chunks
// (cuda::imma_k_chunk) inside it, so a backend holds at most kStreams x 128 MiB of IMMA workspace.
constexpr size_t kImmaBlockCap = static_cast<size_t>(128) << 20;

// (slot-22 G1) one IMMA operand workspace (device memory). Shared by the programs of a
// backend that were built while it was the backend's current block for its stream slot;
// freed when the last of them (and the backend) lets go -- so a captured graph never
// outlives the block it points into.
struct ImmaBlock {
    void*  p = nullptr;
    size_t bytes = 0;
    explicit ImmaBlock(size_t b) : bytes(b) { p = cu::CudaCtx::get().dalloc(b); }
    ~ImmaBlock() { if (p != nullptr && cu::available()) cu::CudaCtx::get().dfree(p, bytes); }
    ImmaBlock(const ImmaBlock&) = delete;
    ImmaBlock& operator=(const ImmaBlock&) = delete;
};

} // namespace

// ---------------------------------------------------------------------------
// Per-program device state
// ---------------------------------------------------------------------------
template<class Elt>
struct DeviceProgram {
    uint64_t uid = 0, structural_hash = 0;
    size_t   n_nodes = 0;
    std::vector<cu::DeviceBuf<Elt>*> dbuf;                    // per BufId (null: untouched by the device)
    std::vector<std::unique_ptr<cu::DeviceBuf<Elt>>> owned;   // the program's own mirrors (session mirrors are not owned)
    std::vector<char> is_session;                             // per BufId: bound to an elimination-session mirror (WP-19a)
    std::vector<int>  cpu_last;                               // per BufId: last level with a CPU node touching it (-1: none)
    bool     has_session = false;
    std::vector<uint32_t> mirror_ld;                          // ld the mirror was built with
    std::vector<std::vector<uint32_t>> gpu_nodes, xfer_nodes; // per level
    std::vector<std::vector<BufId>> gpu_written, gpu_touched, cpu_written;   // per level, sorted unique
    struct NodePrep {
        std::vector<std::unique_ptr<cu::GemmBatch<Elt>>> batches;
        std::vector<std::vector<uint32_t>> batch_inst;   // instance ids per sub-batch (for the scaled epilogue)
        void*  ws = nullptr; size_t ws_bytes = 0; uint32_t splits = 1;
        Elt*   scratch = nullptr; size_t scratch_elts = 0;   // scaled products: m x n row-major slab(s)
        int    imma_slot = -1;   // (slot-22 G1) the stream slot whose IMMA workspace this node uses (-1: none)
    };
    std::map<uint32_t, NodePrep> prep;
    std::map<uint32_t, std::pair<uint32_t*, size_t>> index_lists;   // program list id -> device array
    std::vector<std::unique_ptr<cu::GraphCapture>> graphs;        // per run start level
    // (slot-22 G1) the IMMA operand workspace of each stream slot, >= the largest IMMA-eligible
    // GEMM-class node dealt to that slot. Nodes of one slot run in stream order -- within a run,
    // inside a captured graph, and across the programs of the backend, which share its streams
    // and run one at a time (one Executor, one thread) -- so they can share it.
    // INVARIANT (audit R1): ONE DRIVER PER BACKEND -- a DeviceBackend (its streams, its programs and
    // these blocks) is driven by one host thread at a time; two threads enqueueing programs of the
    // same backend concurrently would let two GEMMs convert into the same A8 / B8 slab. The slot
    // formula itself is the single stream_slot(), checked at every enqueue.
    std::shared_ptr<ImmaBlock> imma_ws[kStreams];
    // Consecutive levels with no CPU node and no D2H node form one RUN: their
    // dependencies are stream order, so the whole run is enqueued (and
    // captured) at once and the host synchronises once at its end.  H2D
    // nodes inside a run are issued at its start (legal: no host write can
    // occur inside a GPU-only run, and pass 7 hoisted them past the last one).
    std::vector<uint32_t> run_start, run_end;                     // per level: [start, end] of its run (end inclusive)
    uint64_t runs = 0, last_use = 0;
    size_t   bytes = 0;
    bool     merged = false;

    ~DeviceProgram() {
        cu::CudaCtx& cx = cu::CudaCtx::get();
        graphs.clear();
        for (auto& kv : prep) {
            if (kv.second.ws) cx.dfree(kv.second.ws, kv.second.ws_bytes);
            if (kv.second.scratch) cx.dfree(kv.second.scratch, kv.second.scratch_elts * sizeof(Elt));
        }
        prep.clear();
        for (auto& kv : index_lists) cx.dfree(kv.second.first, kv.second.second * sizeof(uint32_t));
        index_lists.clear();
        dbuf.clear();
        owned.clear();
    }
};

template<class Elt>
struct DeviceBackend<Elt>::Impl {
    cu::Stream streams[kStreams] = {};
    cu::Event  ev_fork{}, ev_join[kStreams] = {}, ev_done{};
    bool       created = false;
    std::map<uint64_t, std::unique_ptr<DeviceProgram<Elt>>> programs;
    std::shared_ptr<ImmaBlock> imma_cur[kStreams];   // (slot-22 G1) the current IMMA block per stream slot
    uint64_t   use_counter = 0;
    DeviceStats stats;
    // current run
    const Program<Elt>* prog = nullptr;
    DeviceProgram<Elt>* dp = nullptr;
    std::vector<DenseBuf<Elt>>* host = nullptr;
    const ModCtx<Elt>* ctx = nullptr;
    uint32_t cur_level = 0;
    std::vector<uint32_t> empty;
    bool verify = false;
    // (slot-18 WP-19a) the elimination session: adopted host buffers keyed by
    // base pointer (std::map: no hash containers on result paths).
    struct SessionEntry {
        DenseBuf<Elt> host;                              // non-owning alias of the adopted storage
        bool keep = true;
        std::unique_ptr<cu::DeviceBuf<Elt>> mirror;      // created lazily by the first program binding it
    };
    bool session_on = false;
    std::map<const Elt*, SessionEntry> session;

    static size_t owned_elts(const cu::DeviceBuf<Elt>& b) {
        // the bytes a window binding really owns: (major - 1) * ld + minor
        return static_cast<size_t>(b.major_dim() - 1) * b.ld + b.minor_dim();
    }
    // The session mirror for a host binding: same base, ld and layout, and a
    // prefix window of the adopted buffer (so buffer coordinates coincide).
    cu::DeviceBuf<Elt>* session_mirror(const DenseBuf<Elt>& h) {
        if (!session_on || h.data() == nullptr || h.footprint() == 0) return nullptr;
        auto it = session.find(h.data());
        if (it == session.end()) return nullptr;
        SessionEntry& e = it->second;
        if (e.host.ld != h.ld || e.host.layout != h.layout || h.rows > e.host.rows || h.cols > e.host.cols) return nullptr;
        if (!e.mirror) {
            e.mirror = std::make_unique<cu::DeviceBuf<Elt>>(e.host);   // host side current
            ++stats.session_mirrors;
        }
        return e.mirror.get();
    }
    // The EXISTING session mirror for a binding (prefix rule), never created.
    cu::DeviceBuf<Elt>* session_existing(const DenseBuf<Elt>& h) {
        if (!session_on || h.data() == nullptr || h.footprint() == 0) return nullptr;
        auto it = session.find(h.data());
        if (it == session.end() || !it->second.mirror) return nullptr;
        const SessionEntry& e = it->second;
        if (e.host.ld != h.ld || e.host.layout != h.layout || h.rows > e.host.rows || h.cols > e.host.cols) return nullptr;
        return it->second.mirror.get();
    }
    // A run with no device work inside a session: every bound adopted buffer
    // whose mirror is device-newer is downloaded first, and every one the
    // program writes becomes host-newer (no GPU node runs, so marking up front
    // is the same as marking after).
    void session_host_run(const Program<Elt>& p, std::vector<DenseBuf<Elt>>& hb) {
        std::vector<char> written(p.bufs.size(), 0);
        std::vector<Access> acc;
        for (const Node<Elt>& n : p.nodes) {
            if (n.dead) continue;
            node_accesses(p, n, acc);
            for (const Access& a : acc) if (a.write && a.v.buf < p.bufs.size()) written[a.v.buf] = 1;
        }
        for (BufId b = 0; b < p.bufs.size(); ++b) {
            if (p.bufs[b].kind == BufKind::Temp) continue;
            cu::DeviceBuf<Elt>* m = session_existing(hb[b]);
            if (m == nullptr) continue;
            if (m->residency() == cu::Residency::Device) {
                cu::CudaCtx::get().download(m->host()->data(), m->dptr(), owned_elts(*m) * sizeof(Elt), streams[0]);
                m->mark_synced();
                ++stats.session_syncs;
            }
            if (written[b]) m->mark_host_written();
        }
    }
    void session_download(SessionEntry& e) {
        cu::DeviceBuf<Elt>* b = e.mirror.get();
        if (b == nullptr || b->residency() != cu::Residency::Device) return;
        const size_t elts = owned_elts(*b);
        cu::CudaCtx::get().download(e.host.data(), b->dptr(), elts * sizeof(Elt), streams[0]);
        b->mark_synced();
        ++stats.session_release_d2h; stats.session_release_d2h_bytes += elts * sizeof(Elt);
    }

    void ensure_streams() {
        if (created) return;
        cu::CudaCtx& cx = cu::CudaCtx::get();
        for (int i = 0; i < kStreams; ++i) streams[i] = cx.create_stream();
        ev_fork = cx.create_event(); ev_done = cx.create_event();
        for (int i = 0; i < kStreams; ++i) ev_join[i] = cx.create_event();
        created = true;
    }
    ~Impl() {
        programs.clear();
        for (int i = 0; i < kStreams; ++i) imma_cur[i].reset();
        if (created && cu::available()) {
            cu::CudaCtx& cx = cu::CudaCtx::get();
            cx.destroy(ev_fork); cx.destroy(ev_done);
            for (int i = 0; i < kStreams; ++i) { cx.destroy(ev_join[i]); cx.destroy_stream(streams[i]); }
        }
    }

    // ---- views ----------------------------------------------------------------
    cu::DeviceView<Elt> dv(const ViewRef& v) {
        cu::DeviceBuf<Elt>* b = dp->dbuf[v.buf];
        if (b == nullptr || v.rows == 0 || v.cols == 0) {
            cu::DeviceView<Elt> e; e.rows = v.lrows(); e.cols = v.lcols(); return e;
        }
        cu::DeviceView<Elt> w = b->view().sub(v.row0, v.rows, v.col0, v.cols);
        return v.transposed ? w.transposed_view() : w;
    }
    cu::DeviceView<const Elt> dcv(const ViewRef& v) { return cu::DeviceView<const Elt>(dv(v)); }

    // ---- (slot-22 G1) the IMMA block of stream slot i grows to >= need ------------
    void imma_grow(int i, size_t need) {
        cu::CudaCtx& cx = cu::CudaCtx::get();
        size_t sz = static_cast<size_t>(1) << 20;
        while (sz < need) sz <<= 1;
        if (imma_cur[i]) {
            for (int k = 0; k < kStreams; ++k) cx.sync(streams[k]);   // nothing in flight reads the old block
            const ImmaBlock* old = imma_cur[i].get();
            for (auto& kv : programs) {
                DeviceProgram<Elt>& q = *kv.second;
                if (q.imma_ws[i].get() != old) continue;
                q.imma_ws[i].reset();
                for (auto& g : q.graphs) g.reset();
                ++stats.imma_graph_drops;
            }
            imma_cur[i].reset();   // the last holder: freed here
        }
        try {
            imma_cur[i] = std::make_shared<ImmaBlock>(sz);
        } catch (const cu::CudaError&) {
            imma_cur[i].reset();
            return;
        }
        ++stats.imma_blocks;
        for (auto& kv : programs) {   // re-point the programs that lost the old block
            DeviceProgram<Elt>& q = *kv.second;
            bool uses = false;
            for (const auto& pk : q.prep) if (pk.second.imma_slot == i) { uses = true; break; }
            if (uses && !q.imma_ws[i]) q.imma_ws[i] = imma_cur[i];
        }
        if (trace()) std::cout << "    [device] IMMA slot " << i << ": block " << (sz >> 20) << " MiB (need " << (need >> 20) << " MiB)" << std::endl;
    }

    // ---- build ----------------------------------------------------------------
    DeviceProgram<Elt>& build(const Program<Elt>& p, std::vector<DenseBuf<Elt>>& hb) {
        cu::CudaCtx& cx = cu::CudaCtx::get();
        auto up = std::make_unique<DeviceProgram<Elt>>();
        DeviceProgram<Elt>& d = *up;
        d.uid = p.uid; d.structural_hash = p.structural_hash; d.n_nodes = p.nodes.size();
        d.merged = !verify && merge_levels();   // ENGINE_VERIFY_DEVICE compares level by level: no merging
        const size_t nl = p.levels.size();
        d.gpu_nodes.assign(nl, {}); d.xfer_nodes.assign(nl, {});
        d.gpu_written.assign(nl, {}); d.gpu_touched.assign(nl, {}); d.cpu_written.assign(nl, {});
        d.dbuf.assign(p.bufs.size(), nullptr); d.mirror_ld.assign(p.bufs.size(), 0);
        d.is_session.assign(p.bufs.size(), 0); d.cpu_last.assign(p.bufs.size(), -1);
        std::vector<char> need(p.bufs.size(), 0);
        std::vector<Access> acc;
        for (size_t L = 0; L < nl; ++L) {
            for (uint32_t ni : p.levels[L]) {
                const Node<Elt>& n = p.nodes[ni];
                if (is_transfer(n.op)) { d.xfer_nodes[L].push_back(ni); need[n.c.buf] = 1; continue; }
                node_accesses(p, n, acc);
                if (n.device == Device::GPU) {
                    d.gpu_nodes[L].push_back(ni);
                    for (const Access& a : acc) {
                        need[a.v.buf] = 1;
                        d.gpu_touched[L].push_back(a.v.buf);
                        if (a.write) d.gpu_written[L].push_back(a.v.buf);
                    }
                } else {
                    for (const Access& a : acc) {
                        if (a.write) d.cpu_written[L].push_back(a.v.buf);
                        if (a.v.buf < p.bufs.size()) d.cpu_last[a.v.buf] = static_cast<int>(L);
                    }
                }
            }
            auto uniq = [](std::vector<BufId>& v) { std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); };
            uniq(d.gpu_written[L]); uniq(d.gpu_touched[L]); uniq(d.cpu_written[L]);
        }
        // (WP-19a) a binding of an adopted buffer that already HAS a session
        // mirror is mapped even when no GPU or transfer node touches it here:
        // a CPU write must make the mirror host-newer (host_wrote), and a CPU
        // access of a device-newer mirror must download it first (begin_run).
        if (session_on)
            for (BufId b = 0; b < p.bufs.size(); ++b)
                if (!need[b] && p.bufs[b].kind != BufKind::Temp && session_existing(hb[b]) != nullptr) need[b] = 1;
        // CPU-written lists only matter for mirrored buffers
        for (size_t L = 0; L < nl; ++L) {
            auto& v = d.cpu_written[L];
            v.erase(std::remove_if(v.begin(), v.end(), [&](BufId b) { return !need[b]; }), v.end());
        }
        // runs of GPU-only levels
        d.run_start.assign(nl, 0); d.run_end.assign(nl, 0);
        {
            std::vector<char> gpu_only(nl, 0);
            for (size_t L = 0; L < nl; ++L) {
                bool ok = !d.gpu_nodes[L].empty();
                for (uint32_t ni : p.levels[L]) {
                    const Node<Elt>& n = p.nodes[ni];
                    if (n.op == Op::D2H || (!is_transfer(n.op) && n.device != Device::GPU)) { ok = false; break; }
                }
                gpu_only[L] = ok ? 1 : 0;
            }
            size_t L = 0;
            while (L < nl) {
                size_t e = L;
                if (gpu_only[L] && d.merged) while (e + 1 < nl && gpu_only[e + 1]) ++e;
                for (size_t k = L; k <= e; ++k) { d.run_start[k] = static_cast<uint32_t>(L); d.run_end[k] = static_cast<uint32_t>(e); }
                L = e + 1;
            }
        }
        for (BufId b = 0; b < p.bufs.size(); ++b) {
            if (!need[b]) continue;
            DenseBuf<Elt>& h = hb[b];
            if (h.footprint() == 0) continue;
            d.mirror_ld[b] = h.ld;
            if (p.bufs[b].kind != BufKind::Temp) {
                if (cu::DeviceBuf<Elt>* sm = session_mirror(h)) {   // (WP-19a) the session's mirror, not owned
                    d.dbuf[b] = sm; d.is_session[b] = 1; d.has_session = true;
                    continue;
                }
            }
            d.owned.push_back(std::make_unique<cu::DeviceBuf<Elt>>(h));
            d.dbuf[b] = d.owned.back().get();
            d.bytes += d.dbuf[b]->bytes();
        }
        // per-node preparation
        // (slot-22 G1) the IMMA route: decided per program build (eligible modulus, supported
        // device, ENGINE_GPU_IMMA); the kernel re-checks at every call, so a stale decision can
        // only fall back to the u32 kernel -- never change a value.
        const bool imma_on = cu::imma_eligible(*ctx) && cu::imma_policy() && cu::imma_supported();
        size_t imma_need[kStreams] = {};
        for (size_t L = 0; L < nl; ++L) {
            const size_t nlev = d.gpu_nodes[L].size();
            for (size_t t = 0; t < nlev; ++t) {
                const uint32_t ni = d.gpu_nodes[L][t];
                const int slot = stream_slot(t, nlev);   // == enqueue_level's stream choice (the one formula)
                const Node<Elt>& n = p.nodes[ni];
                switch (n.op) {
                    case Op::GEMM: case Op::SCHUR_UPDATE: case Op::TRSM: {
                        if (n.op == Op::TRSM && n.pivot_kind != PivotKind::General) break;
                        const uint32_t m = n.c.lrows(), nn = n.c.lcols(), k = (n.op == Op::TRSM) ? n.b.lrows() : n.a.lcols();
                        if (m == 0 || nn == 0 || k == 0) break;
                        typename DeviceProgram<Elt>::NodePrep& pr = d.prep[ni];
                        if (imma_on) {
                            pr.imma_slot = slot;
                            const size_t full = cu::imma_workspace_bytes(m, nn, k);
                            const size_t chunk = cu::imma_workspace_bytes(m, nn, std::min<uint32_t>(k, 64));
                            imma_need[slot] = std::max(imma_need[slot], std::max(std::min(full, kImmaBlockCap), chunk));
                            ++stats.imma_nodes;
                        }
                        pr.splits = cu::suggest_splits(m, nn, k);
                        if (pr.splits > 1) {
                            pr.ws_bytes = cu::gemm_workspace_bytes(m, nn, pr.splits, sizeof(Elt));
                            pr.ws = cx.dalloc(pr.ws_bytes);
                            d.bytes += pr.ws_bytes;
                        }
                        if (n.alpha != Elt(1) && n.op != Op::TRSM) {
                            pr.scratch_elts = static_cast<size_t>(m) * nn;
                            pr.scratch = static_cast<Elt*>(cx.dalloc(pr.scratch_elts * sizeof(Elt)));
                            d.bytes += pr.scratch_elts * sizeof(Elt);
                        }
                        break;
                    }
                    case Op::BATCHED_GEMM: {
                        typename DeviceProgram<Elt>::NodePrep& pr = d.prep[ni];
                        const bool scaled = (n.alpha != Elt(1));
                        const uint32_t ninst = n.batch_end - n.batch_begin;
                        const GemmInstance& g0 = p.instances[n.batch_begin];
                        const uint32_t m = g0.c.lrows(), nn = g0.c.lcols();
                        if (scaled) {
                            pr.scratch_elts = static_cast<size_t>(m) * nn * ninst;
                            pr.scratch = static_cast<Elt*>(cx.dalloc(pr.scratch_elts * sizeof(Elt)));
                            d.bytes += pr.scratch_elts * sizeof(Elt);
                        }
                        // sub-batches by stride tuple (GemmBatch shares strides across instances)
                        std::map<std::array<size_t, 6>, size_t> groups;
                        for (uint32_t i = n.batch_begin; i < n.batch_end; ++i) {
                            const GemmInstance& g = p.instances[i];
                            cu::DeviceView<const Elt> A = dcv_of(d, g.a), B = dcv_of(d, g.b);
                            cu::DeviceView<Elt> C = scaled
                                ? cu::DeviceView<Elt>(pr.scratch + static_cast<size_t>(i - n.batch_begin) * m * nn, nn, 1, m, nn)
                                : dv_of(d, g.c);
                            const std::array<size_t, 6> key{A.rs, A.cs, B.rs, B.cs, C.rs, C.cs};
                            auto it = groups.find(key);
                            if (it == groups.end()) {
                                it = groups.emplace(key, pr.batches.size()).first;
                                pr.batches.push_back(std::make_unique<cu::GemmBatch<Elt>>());
                                pr.batch_inst.emplace_back();
                                pr.batches.back()->set_shape(A, B, C);
                            }
                            pr.batches[it->second]->push(A, B, C);
                            pr.batch_inst[it->second].push_back(i);
                        }
                        size_t bneed = 0;
                        for (auto& b : pr.batches) {
                            b->upload(streams[0]);
                            bneed = std::max(bneed, b->imma_bytes_needed());   // sub-batches run one after the other
                        }
                        // (G1) the slot's workspace serves every sub-batch; a batch above the block cap stays on the u32 kernel
                        if (imma_on && bneed > 0 && bneed <= kImmaBlockCap) {
                            pr.imma_slot = slot;
                            imma_need[slot] = std::max(imma_need[slot], bneed);
                            ++stats.imma_nodes;
                        }
                        break;
                    }
                    case Op::GATHER: {
                        // the column list and (slot-18 WP-19a) the row list
                        for (const uint32_t li : {n.index_list, n.row_list}) {
                            if (li == kNoList || d.index_lists.count(li)) continue;
                            const std::vector<uint32_t>& l = p.index_lists[li];
                            const size_t cnt = std::max<size_t>(l.size(), 1);
                            uint32_t* dl = static_cast<uint32_t*>(cx.dalloc(cnt * sizeof(uint32_t)));
                            if (!l.empty()) cx.upload(l.data(), dl, l.size() * sizeof(uint32_t), streams[0]);
                            d.index_lists.emplace(li, std::make_pair(dl, cnt));
                            d.bytes += cnt * sizeof(uint32_t);
                        }
                        break;
                    }
                    default: break;
                }
            }
        }
        // (G1) the per-slot IMMA workspaces: ONE block per stream slot and backend, grown on demand
        // to the next power of two >= the need (so the arena sees few distinct sizes). A growth first
        // retires the old block everywhere: the streams are synchronised, every cached program that
        // points at it is re-pointed to the new block and loses its captured graphs (they recorded
        // the old pointer; the next run re-captures), and the old block is freed. So the IMMA memory
        // of a backend is at most kStreams blocks. A failed allocation leaves the slot on the u32 kernel.
        for (int i = 0; i < kStreams; ++i) {
            if (imma_need[i] == 0) continue;
            if (!imma_cur[i] || imma_cur[i]->bytes < imma_need[i]) imma_grow(i, imma_need[i]);
            d.imma_ws[i] = imma_cur[i];
        }
        d.graphs.resize(nl);
        cx.sync(streams[0]);   // batch tables / index lists landed
        ++stats.programs_built;
        // LRU eviction
        while (programs.size() >= program_cache_cap() && !programs.empty()) {
            auto victim = programs.begin();
            for (auto it = programs.begin(); it != programs.end(); ++it)
                if (it->second->last_use < victim->second->last_use) victim = it;
            programs.erase(victim);
            ++stats.programs_evicted;
        }
        DeviceProgram<Elt>* raw = up.get();
        programs[p.uid] = std::move(up);
        return *raw;
    }
    // view helpers usable during build (dp not yet set)
    cu::DeviceView<Elt> dv_of(DeviceProgram<Elt>& d, const ViewRef& v) {
        cu::DeviceBuf<Elt>* b = d.dbuf[v.buf];
        if (b == nullptr || v.rows == 0 || v.cols == 0) { cu::DeviceView<Elt> e; e.rows = v.lrows(); e.cols = v.lcols(); return e; }
        cu::DeviceView<Elt> w = b->view().sub(v.row0, v.rows, v.col0, v.cols);
        return v.transposed ? w.transposed_view() : w;
    }
    cu::DeviceView<const Elt> dcv_of(DeviceProgram<Elt>& d, const ViewRef& v) { return cu::DeviceView<const Elt>(dv_of(d, v)); }

    // ---- node execution ----------------------------------------------------------
    // (slot-22 G1) the node's IMMA workspace (its stream slot's); the kernel decides the route.
    void imma_opts(const typename DeviceProgram<Elt>::NodePrep& pr, cu::GemmOpts& o) const {
        if (pr.imma_slot < 0 || !dp->imma_ws[pr.imma_slot]) return;
        o.imma_workspace = dp->imma_ws[pr.imma_slot]->p;
        o.imma_workspace_bytes = dp->imma_ws[pr.imma_slot]->bytes;
    }
    void gemm_like(const Node<Elt>& n, uint32_t ni, const ViewRef& A, const ViewRef& B, const ViewRef& C, cu::Stream s) {
        if (C.lrows() == 0 || C.lcols() == 0) return;
        cu::DeviceView<Elt> c = dv(C);
        const uint32_t k = A.lcols();
        if (k == 0) {
            if (n.gop == GemmOp::Set) cu::fill<Elt>(c, Elt(0), s);
            return;
        }
        cu::GemmOpts o;
        auto it = dp->prep.find(ni);
        if (it != dp->prep.end() && it->second.splits > 1) { o.splits = it->second.splits; o.workspace = it->second.ws; o.workspace_bytes = it->second.ws_bytes; }
        if (it != dp->prep.end()) imma_opts(it->second, o);
        if (n.alpha == Elt(1)) { cu::gemm<Elt>(*ctx, dcv(A), dcv(B), c, n.gop, s, o); return; }
        ENGINE_CHECK_MSG(it != dp->prep.end() && it->second.scratch != nullptr, "DeviceBackend: scaled GEMM without scratch");
        cu::DeviceView<Elt> sc(it->second.scratch, C.lcols(), 1, C.lrows(), C.lcols());
        cu::gemm<Elt>(*ctx, dcv(A), dcv(B), sc, GemmOp::Set, s, o);
        if (n.gop == GemmOp::Set) { cu::copy<Elt>(cu::DeviceView<const Elt>(sc), c, s); cu::scal<Elt>(*ctx, n.alpha, c, s); }
        else cu::axpy<Elt>(*ctx, n.alpha, cu::DeviceView<const Elt>(sc), c, n.gop == GemmOp::Sub, s);
    }

    void enqueue(uint32_t ni, int slot) {
        ENGINE_CHECK_MSG(slot >= 0 && slot < kStreams, "DeviceBackend: stream slot out of range");
        const cu::Stream s = streams[slot];
        {   // (G1, audit R1) a node's IMMA workspace belongs to the slot it is launched on
            auto pit = dp->prep.find(ni);
            ENGINE_CHECK_MSG(pit == dp->prep.end() || pit->second.imma_slot < 0 || pit->second.imma_slot == slot,
                             "DeviceBackend: node's IMMA slot differs from its stream slot");
        }
        const Node<Elt>& n = prog->nodes[ni];
        switch (n.op) {
            case Op::GEMM: case Op::SCHUR_UPDATE:
                gemm_like(n, ni, n.a, n.b, n.c, s);
                break;
            case Op::BATCHED_GEMM: {
                auto it = dp->prep.find(ni);
                ENGINE_CHECK_MSG(it != dp->prep.end(), "DeviceBackend: batch not prepared");
                typename DeviceProgram<Elt>::NodePrep& pr = it->second;
                const bool scaled = (n.alpha != Elt(1));
                cu::GemmOpts io;   // (G1) the node's slot workspace, shared by its sub-batches
                imma_opts(pr, io);
                for (size_t g = 0; g < pr.batches.size(); ++g) {
                    pr.batches[g]->run(*ctx, scaled ? GemmOp::Set : n.gop, s, 0, -1, io.imma_workspace, io.imma_workspace_bytes);
                    if (!scaled) continue;
                    const uint32_t m = pr.batches[g]->m(), nn = pr.batches[g]->n();
                    for (uint32_t i : pr.batch_inst[g]) {
                        cu::DeviceView<const Elt> sc(pr.scratch + static_cast<size_t>(i - n.batch_begin) * m * nn, nn, 1, m, nn);
                        cu::DeviceView<Elt> c = dv(prog->instances[i].c);
                        if (n.gop == GemmOp::Set) { cu::copy<Elt>(sc, c, s); cu::scal<Elt>(*ctx, n.alpha, c, s); }
                        else cu::axpy<Elt>(*ctx, n.alpha, sc, c, n.gop == GemmOp::Sub, s);
                    }
                }
                break;
            }
            case Op::TRSM: {
                if (n.c.lrows() == 0 || n.c.lcols() == 0) break;
                if (n.pivot_kind == PivotKind::General) {
                    if (n.b.lrows() == 0) { cu::fill<Elt>(dv(n.c), Elt(0), s); break; }
                    cu::GemmOpts o;
                    auto it = dp->prep.find(ni);
                    if (it != dp->prep.end() && it->second.splits > 1) { o.splits = it->second.splits; o.workspace = it->second.ws; o.workspace_bytes = it->second.ws_bytes; }
                    if (it != dp->prep.end()) imma_opts(it->second, o);
                    cu::gemm<Elt>(*ctx, dcv(n.a), dcv(n.b), dv(n.c), GemmOp::Set, s, o);
                } else {
                    cu::copy<Elt>(dcv(n.b), dv(n.c), s);
                    if (n.pivot_kind == PivotKind::Scalar && n.alpha != Elt(1)) cu::scal<Elt>(*ctx, n.alpha, dv(n.c), s);
                }
                break;
            }
            case Op::AXPY:
                if (n.c.lrows() == 0 || n.c.lcols() == 0) break;
                if (n.a.buf == kIdentityBuf) cu::axpy_identity<Elt>(*ctx, n.alpha, dv(n.c), n.subtract, s);
                else cu::axpy<Elt>(*ctx, n.alpha, dcv(n.a), dv(n.c), n.subtract, s);
                break;
            case Op::SCAL:
                if (n.c.lrows() == 0 || n.c.lcols() == 0) break;
                if (n.alpha == Elt(0)) cu::fill<Elt>(dv(n.c), Elt(0), s);
                else cu::scal<Elt>(*ctx, n.alpha, dv(n.c), s);
                break;
            case Op::TRANSPOSE: case Op::SCATTER:
                if (n.c.lrows() == 0 || n.c.lcols() == 0) break;
                cu::copy<Elt>(dcv(n.a), dv(n.c), s);
                break;
            case Op::CANONICALIZE:
                if (n.c.lrows() == 0 || n.c.lcols() == 0) break;
                cu::canonicalise<Elt>(*ctx, dv(n.c), s);
                break;
            case Op::GATHER: {
                if (n.c.lrows() == 0 || n.c.lcols() == 0) break;
                const bool ident = (n.a.buf == kIdentityBuf);
                cu::DeviceView<const Elt> a;
                if (!ident) a = dcv(n.a);
                const uint32_t* cols = nullptr;
                if (n.index_list != kNoList) {
                    auto it = dp->index_lists.find(n.index_list);
                    ENGINE_CHECK_MSG(it != dp->index_lists.end(), "DeviceBackend: index list not uploaded");
                    cols = it->second.first;
                }
                const uint32_t* rows = nullptr;   // (slot-18 WP-19a) the row-list GATHER
                if (n.row_list != kNoList) {
                    auto it = dp->index_lists.find(n.row_list);
                    ENGINE_CHECK_MSG(it != dp->index_lists.end(), "DeviceBackend: row list not uploaded");
                    rows = it->second.first;
                }
                cu::gather<Elt>(*ctx, n.alpha, ident ? Elt(0) : n.beta, a, n.aux, cols, dv(n.c), n.accumulate, s, rows);
                break;
            }
            default:
                ENGINE_CHECK_MSG(false, "DeviceBackend: op is not device-eligible");
        }
    }

    void enqueue_level(uint32_t L) {
        cu::CudaCtx& cx = cu::CudaCtx::get();
        const std::vector<uint32_t>& nodes = dp->gpu_nodes[L];
        const int ns = static_cast<int>(std::min<size_t>(nodes.size(), kStreams));
        if (ns > 1) {
            cx.record(ev_fork, streams[0]);
            for (int i = 1; i < ns; ++i) cx.wait(streams[i], ev_fork);
        }
        for (size_t t = 0; t < nodes.size(); ++t) enqueue(nodes[t], stream_slot(t, nodes.size()));
        for (int i = 1; i < ns; ++i) { cx.record(ev_join[i], streams[i]); cx.wait(streams[0], ev_join[i]); }
    }
    // every level of the run [L, run_end[L]] in stream order
    void enqueue_run(uint32_t L) {
        for (uint32_t k = L; k <= dp->run_end[L]; ++k) enqueue_level(k);
    }

    void transfer(const Node<Elt>& n) {
        cu::CudaCtx& cx = cu::CudaCtx::get();
        cu::DeviceBuf<Elt>* b = dp->dbuf[n.c.buf];
        if (b == nullptr) return;
        DenseBuf<Elt>& h = *b->host();   // this run's binding, or (WP-19a) the session's adopted buffer
        const size_t elts = owned_elts(*b);
        const cu::Residency r = b->residency();
        const bool sess = dp->is_session[n.c.buf] != 0;
        if (n.op == Op::H2D) {
            // (WP-19a) a session mirror left device-newer by an earlier program of
            // the session: no host write happened since (the session contract), so
            // the device copy is the current one -- the plan assumed a fresh binding.
            if (sess && r == cu::Residency::Device) { ++stats.session_skipped_h2d; return; }
            ENGINE_CHECK_MSG(r != cu::Residency::Device, "DeviceBackend: H2D of a device-newer buffer (residency plan)");
            if (r == cu::Residency::Both) { ++stats.skipped_transfers; return; }
            cx.upload(h.data(), b->dptr(), elts * sizeof(Elt), streams[0]);
            b->mark_synced();
            ++stats.h2d; stats.h2d_bytes += elts * sizeof(Elt);
        } else {
            ENGINE_CHECK_MSG(r != cu::Residency::Host, "DeviceBackend: D2H of a host-newer buffer (residency plan)");
            if (r == cu::Residency::Both || r == cu::Residency::None) { ++stats.skipped_transfers; return; }
            // (WP-19a) no CPU node of this program touches it from here on: the
            // download is deferred to the next CPU access, release or session end.
            if (sess && dp->cpu_last[n.c.buf] < static_cast<int>(cur_level)) { ++stats.session_deferred_d2h; return; }
            cx.download(h.data(), b->dptr(), elts * sizeof(Elt), streams[0]);
            b->mark_synced();
            ++stats.d2h; stats.d2h_bytes += elts * sizeof(Elt);
        }
    }
};

// ---------------------------------------------------------------------------
template<class Elt>
DeviceBackend<Elt>::DeviceBackend() : impl_(new Impl()) {}
template<class Elt>
DeviceBackend<Elt>::~DeviceBackend() = default;

template<class Elt>
bool DeviceBackend<Elt>::usable() { return cu::available(); }

template<class Elt>
bool DeviceBackend<Elt>::begin_run(const Program<Elt>& prog, std::vector<DenseBuf<Elt>>& host, const ModCtx<Elt>& ctx, bool verify) {
    Impl& I = *impl_;
    I.verify = verify;
    if (prog.stats.gpu_nodes == 0 && prog.stats.h2d_nodes == 0 && prog.stats.d2h_nodes == 0) {
        if (I.session_on) I.session_host_run(prog, host);   // (WP-19a) a CPU-only program inside a session
        return false;
    }
    ENGINE_CHECK_MSG(prog.compiled && prog.uid != 0, "DeviceBackend: program not compiled");
    I.ensure_streams();
    I.prog = &prog; I.host = &host; I.ctx = &ctx;
    auto it = I.programs.find(prog.uid);
    DeviceProgram<Elt>* dp = (it != I.programs.end()) ? it->second.get() : nullptr;
    if (dp != nullptr) {
        bool ok = dp->structural_hash == prog.structural_hash && dp->n_nodes == prog.nodes.size();
        for (BufId b = 0; ok && b < prog.bufs.size(); ++b)
            if (dp->dbuf[b] && dp->mirror_ld[b] != host[b].ld) ok = false;
        if (!ok) { I.programs.erase(it); dp = nullptr; ++I.stats.programs_evicted; }
    }
    if (dp != nullptr && dp->merged != (!verify && merge_levels())) { I.programs.erase(it); dp = nullptr; ++I.stats.programs_evicted; }
    // (WP-19a) a cached program never mixes with a session: one that holds
    // session mirrors (of an earlier session) or binds an adopted buffer
    // through its own mirror is rebuilt.
    if (dp != nullptr) {
        bool stale = dp->has_session;
        for (BufId b = 0; !stale && I.session_on && b < prog.bufs.size(); ++b)
            if (dp->dbuf[b] && prog.bufs[b].kind != BufKind::Temp && I.session.count(host[b].data())) stale = true;
        if (stale) { I.programs.erase(prog.uid); dp = nullptr; ++I.stats.programs_evicted; }
    }
    if (dp == nullptr) dp = &I.build(prog, host);
    I.dp = dp;
    dp->last_use = ++I.use_counter;
    for (BufId b = 0; b < prog.bufs.size(); ++b) {
        if (!dp->dbuf[b]) continue;
        if (dp->is_session[b]) {
            // (WP-19a) a CPU node of this program touches a device-newer session
            // buffer: make the host copy current first (the plan assumed it).
            cu::DeviceBuf<Elt>* sb = dp->dbuf[b];
            if (dp->cpu_last[b] >= 0 && sb->residency() == cu::Residency::Device) {
                cu::CudaCtx::get().download(sb->host()->data(), sb->dptr(), Impl::owned_elts(*sb) * sizeof(Elt), I.streams[0]);
                sb->mark_synced();
                ++I.stats.session_syncs;
            }
            continue;
        }
        dp->dbuf[b]->rebind_host(host[b], prog.bufs[b].kind != BufKind::Temp);
    }
    ++I.stats.runs;
    return true;
}

template<class Elt>
void DeviceBackend<Elt>::transfers(uint32_t L) {
    Impl& I = *impl_;
    DeviceProgram<Elt>& d = *I.dp;
    if (L >= d.xfer_nodes.size()) return;
    if (d.run_start[L] != L) return;                    // issued at the run start
    const auto t0 = Clock::now();
    for (uint32_t k = L; k <= d.run_end[L]; ++k) {
        I.cur_level = k;
        for (uint32_t ni : d.xfer_nodes[k]) I.transfer(I.prog->nodes[ni]);
    }
    I.stats.transfer_seconds += since(t0);
}

template<class Elt>
void DeviceBackend<Elt>::launch(uint32_t L) {
    Impl& I = *impl_;
    DeviceProgram<Elt>& d = *I.dp;
    if (L >= d.gpu_nodes.size() || d.gpu_nodes[L].empty()) return;
    if (d.run_start[L] != L) return;                    // inside a merged run: enqueued at its start
    const auto t0 = Clock::now();
    cu::CudaCtx& cx = cu::CudaCtx::get();
    const int gm = graph_mode();
    const bool want_graph = (gm == 2) || (gm == 1 && d.runs >= 1);
    if (want_graph && !d.graphs[L]) {
        d.graphs[L] = std::make_unique<cu::GraphCapture>();
        d.graphs[L]->begin(I.streams[0]);
        I.enqueue_run(L);
        d.graphs[L]->end();
        ++I.stats.captures;
    }
    if (d.graphs[L]) { d.graphs[L]->launch(I.streams[0]); ++I.stats.graph_launches; }
    else { I.enqueue_run(L); ++I.stats.direct_launches; }
    cx.record(I.ev_done, I.streams[0]);
    size_t nn = 0;
    for (uint32_t k = L; k <= d.run_end[L]; ++k) nn += d.gpu_nodes[k].size();
    I.stats.gpu_nodes += nn;
    I.stats.gpu_levels += d.run_end[L] - L + 1;
    if (d.run_end[L] > L) I.stats.merged_levels += d.run_end[L] - L;
    I.stats.launch_seconds += since(t0);
    if (trace()) std::cout << "    [device] levels " << L << ".." << d.run_end[L] << ": " << nn << " gpu nodes, "
                           << (d.graphs[L] ? "graph" : "direct") << std::endl;
}

template<class Elt>
void DeviceBackend<Elt>::wait(uint32_t L) {
    Impl& I = *impl_;
    DeviceProgram<Elt>& d = *I.dp;
    if (L >= d.gpu_nodes.size() || d.gpu_nodes[L].empty()) return;
    if (d.run_end[L] != L) return;                      // the run's last level synchronises
    const auto t0 = Clock::now();
    cu::CudaCtx::get().sync(I.ev_done);
    for (uint32_t k = d.run_start[L]; k <= L; ++k)
        for (BufId b : d.gpu_written[k]) if (d.dbuf[b]) d.dbuf[b]->mark_device_written();
    I.stats.wait_seconds += since(t0);
}

template<class Elt>
void DeviceBackend<Elt>::host_wrote(uint32_t L) {
    Impl& I = *impl_;
    DeviceProgram<Elt>& d = *I.dp;
    if (L >= d.cpu_written.size()) return;
    for (BufId b : d.cpu_written[L]) if (d.dbuf[b]) d.dbuf[b]->mark_host_written();
}

template<class Elt>
void DeviceBackend<Elt>::end_run() {
    Impl& I = *impl_;
    if (I.dp) ++I.dp->runs;
    // (WP-19a) a program bound to session mirrors is one-shot by construction
    // (the elimination driver records per panel): drop its device state now,
    // so no cached program ever points at a released session mirror.
    const bool drop = I.dp != nullptr && I.dp->has_session;
    const uint64_t uid = I.dp ? I.dp->uid : 0;
    I.dp = nullptr; I.prog = nullptr; I.host = nullptr; I.ctx = nullptr;
    if (drop) I.programs.erase(uid);
}

template<class Elt>
bool DeviceBackend<Elt>::level_has_gpu(uint32_t L) const {
    const Impl& I = *impl_;
    return I.dp != nullptr && L < I.dp->gpu_nodes.size() && !I.dp->gpu_nodes[L].empty();
}
template<class Elt>
const std::vector<uint32_t>& DeviceBackend<Elt>::gpu_nodes(uint32_t L) const {
    const Impl& I = *impl_;
    if (I.dp == nullptr || L >= I.dp->gpu_nodes.size()) return I.empty;
    return I.dp->gpu_nodes[L];
}

template<class Elt>
void DeviceBackend<Elt>::verify_sync_host(uint32_t L) {
    Impl& I = *impl_;
    DeviceProgram<Elt>& d = *I.dp;
    if (L >= d.gpu_touched.size()) return;
    cu::CudaCtx& cx = cu::CudaCtx::get();
    for (BufId b : d.gpu_touched[L]) {
        cu::DeviceBuf<Elt>* db = d.dbuf[b];
        if (db == nullptr || db->residency() != cu::Residency::Device) continue;
        DenseBuf<Elt>& h = *db->host();
        const size_t elts = static_cast<size_t>(db->major_dim() - 1) * db->ld + db->minor_dim();
        cx.download(h.data(), db->dptr(), elts * sizeof(Elt), I.streams[0]);
        db->mark_synced();
    }
}
template<class Elt>
void DeviceBackend<Elt>::verify_snapshot(uint32_t L, std::vector<std::pair<BufId, DenseBuf<Elt>>>& out) {
    Impl& I = *impl_;
    DeviceProgram<Elt>& d = *I.dp;
    out.clear();
    if (L >= d.gpu_written.size()) return;
    for (BufId b : d.gpu_written[L]) {
        cu::DeviceBuf<Elt>* db = d.dbuf[b];
        if (db == nullptr) continue;
        DenseBuf<Elt> snap(db->rows, db->cols, db->layout, db->ld);
        db->download_to(snap.data(), snap.footprint(), I.streams[0]);
        out.emplace_back(b, std::move(snap));
    }
    I.stats.verified_nodes += d.gpu_nodes[L].size();
}
template<class Elt>
void DeviceBackend<Elt>::verify_mark_synced(uint32_t L) {
    Impl& I = *impl_;
    DeviceProgram<Elt>& d = *I.dp;
    if (L >= d.gpu_written.size()) return;
    for (BufId b : d.gpu_written[L]) if (d.dbuf[b]) d.dbuf[b]->mark_synced();
}

template<class Elt>
void DeviceBackend<Elt>::release() {
    Impl& I = *impl_;
    ENGINE_CHECK_MSG(I.dp == nullptr, "DeviceBackend::release inside a run");
    I.programs.clear();
    for (int i = 0; i < kStreams; ++i) I.imma_cur[i].reset();   // (G1) nothing else holds a block now
}
template<class Elt>
size_t DeviceBackend<Elt>::device_bytes() const {
    size_t b = 0;
    for (const auto& kv : impl_->programs) b += kv.second->bytes;
    {   // (G1) every distinct IMMA block, once
        std::vector<const ImmaBlock*> seen;
        auto add = [&](const std::shared_ptr<ImmaBlock>& blk) {
            if (!blk || std::find(seen.begin(), seen.end(), blk.get()) != seen.end()) return;
            seen.push_back(blk.get()); b += blk->bytes;
        };
        for (int i = 0; i < kStreams; ++i) add(impl_->imma_cur[i]);
        for (const auto& kv : impl_->programs) for (int i = 0; i < kStreams; ++i) add(kv.second->imma_ws[i]);
    }
    for (const auto& kv : impl_->session) if (kv.second.mirror) b += kv.second.mirror->bytes();
    return b;
}

// ---- (slot-18 WP-19a, GPU G1) the elimination session ------------------------
template<class Elt>
void DeviceBackend<Elt>::session_begin() {
    Impl& I = *impl_;
    ENGINE_CHECK_MSG(!I.session_on && I.dp == nullptr, "DeviceBackend::session_begin: a session is open or a run is active");
    I.session_on = true;
    ++I.stats.sessions;
}
template<class Elt>
bool DeviceBackend<Elt>::session_open() const { return impl_->session_on; }
template<class Elt>
void DeviceBackend<Elt>::session_host_run(const Program<Elt>& prog, std::vector<DenseBuf<Elt>>& host) {
    Impl& I = *impl_;
    if (I.session_on) I.session_host_run(prog, host);
}
template<class Elt>
void DeviceBackend<Elt>::session_adopt(const DenseBuf<Elt>& h, bool keep) {
    Impl& I = *impl_;
    ENGINE_CHECK_MSG(I.session_on && I.dp == nullptr, "DeviceBackend::session_adopt: no open session, or inside a run");
    if (h.data() == nullptr || h.footprint() == 0) return;
    auto it = I.session.find(h.data());
    if (it != I.session.end()) {
        const DenseBuf<Elt>& e = it->second.host;
        const bool same = e.rows == h.rows && e.cols == h.cols && e.ld == h.ld && e.layout == h.layout;
        // Two read-only adoptions of one base (two borrowed pool windows): the
        // first one stands -- a binding outside its prefix gets a program
        // mirror, which is safe because neither is ever written.  A writable
        // buffer must be released before its storage is re-adopted.
        ENGINE_CHECK_MSG(same || (!keep && !it->second.keep),
                         "DeviceBackend::session_adopt: base re-adopted with other metadata (release it first)");
        return;
    }
    typename Impl::SessionEntry e;
    const size_t cap = static_cast<size_t>(h.layout == Layout::RowMajor ? h.rows : h.cols) * h.ld;
    // Tags as the executor's own binding of a buffer (Executor.cpp: Canonical /
    // Generic, whatever the caller's DenseBuf says), so a session mirror and a
    // program mirror present the same DeviceView to every kernel.
    e.host = DenseBuf<Elt>(const_cast<Elt*>(h.data()), std::max<size_t>(cap, 1), h.rows, h.cols, h.layout, h.ld);
    e.keep = keep;
    I.session.emplace(h.data(), std::move(e));
}
template<class Elt>
void DeviceBackend<Elt>::session_release(const Elt* base) {
    Impl& I = *impl_;
    ENGINE_CHECK_MSG(I.dp == nullptr, "DeviceBackend::session_release inside a run");
    auto it = I.session.find(base);
    if (it == I.session.end()) return;
    if (it->second.keep) I.session_download(it->second);
    I.session.erase(it);
}
template<class Elt>
void DeviceBackend<Elt>::session_end(bool download) {
    Impl& I = *impl_;
    if (download)
        for (auto& kv : I.session) if (kv.second.keep) I.session_download(kv.second);
    I.session.clear();
    I.session_on = false;
}
template<class Elt>
const DeviceStats& DeviceBackend<Elt>::stats() const { return impl_->stats; }

template class DeviceBackend<uint32_t>;
template class DeviceBackend<uint64_t>;

} // namespace tape
} // namespace engine
