// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Executor.cpp -- see Executor.h.
#include "Executor.h"
#include "../ModArithmetic/RetiredDeviceLevers.h"   // (S25 LEVERS-25) ENGINE_GPU_LEAN

#include <atomic>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/Kernels.h"
#include "Passes.h"
#include "EngineTrace.h"
#include "EngineRuntime.h"   // (slot-23 P7) note_lean_host_run()

namespace engine {
namespace tape {

namespace { std::atomic<uint64_t> g_lean_host_runs{0}; }
bool gpu_lean_enabled() { return engine::lever::gpu_lean_lever().enabled(); }   // (S25 LEVERS-25) RetiredDeviceLevers.h
uint64_t gpu_lean_host_runs() { return g_lean_host_runs.load(std::memory_order_relaxed); }

namespace {

using kernels::GemmOp;

struct Task {
    uint32_t node;
    uint32_t r0, nr, c0, nc;   // output tile (logical coordinates of the node's / instance's c)
    uint32_t inst;             // BATCHED_GEMM instance index (absolute), else 0
    uint32_t tile;             // tile ordinal inside the node (COMPARE flags)
};

template<class Elt>
std::vector<Elt>& scratch_vec() {
    static thread_local std::vector<Elt> s;
    return s;
}

bool trace_levels() {
    static const bool t = std::getenv("ENGINE_TRACE_LEVELS") != nullptr;
    return t;
}
uint32_t tile_mult() {
    static const uint32_t m = [] {
        const char* e = std::getenv("ENGINE_TILE_MULT");
        const uint32_t v = (e != nullptr) ? static_cast<uint32_t>(std::atoi(e)) : 0;
        return v == 0 ? 4u : v;   // tiles per thread (measured best: 4), tuning knob; never affects a value
    }();
    return m;
}
uint32_t tile_cols() {
    static const uint32_t c = [] {
        const char* e = std::getenv("ENGINE_TILE_COLS");
        const uint32_t v = (e != nullptr) ? static_cast<uint32_t>(std::atoi(e)) : 0;
        return v == 0 ? 256u : v;   // column panel of an output tile (measured best: 256)
    }();
    return c;
}

// Output tiling of a GEMM-shaped node: never splits k.
void gemm_tiles(uint32_t m, uint32_t n, uint32_t k, uint32_t target, std::vector<std::array<uint32_t, 4>>& out) {
    out.clear();
    if (m == 0 || n == 0) return;
    const double work = static_cast<double>(m) * n * std::max<uint32_t>(k, 1);
    if (target <= 1 || work < 65536.0) { out.push_back({0, m, 0, n}); return; }
    const uint32_t tc = tile_cols();
    const uint32_t ncc = (n + tc - 1) / tc;
    uint32_t nrc = (target + ncc - 1) / ncc;
    nrc = std::max<uint32_t>(1, std::min<uint32_t>(nrc, (m + 7) / 8));
    const uint32_t rb = (((m + nrc - 1) / nrc) + 7) & ~7u;
    const uint32_t cb = (n + ncc - 1) / ncc;
    for (uint32_t r = 0; r < m; r += rb)
        for (uint32_t c = 0; c < n; c += cb)
            out.push_back({r, std::min(rb, m - r), c, std::min(cb, n - c)});
}

void row_chunks(uint32_t rows, uint32_t cols, uint32_t target, std::vector<std::array<uint32_t, 4>>& out) {
    out.clear();
    if (rows == 0 || cols == 0) { out.push_back({0, rows, 0, cols}); return; }
    if (target <= 1 || static_cast<double>(rows) * cols < 8192.0 || rows < 2) { out.push_back({0, rows, 0, cols}); return; }
    const uint32_t nrc = std::max<uint32_t>(1, std::min<uint32_t>(target, (rows + 15) / 16));
    const uint32_t rb = (rows + nrc - 1) / nrc;
    for (uint32_t r = 0; r < rows; r += rb) out.push_back({r, std::min(rb, rows - r), 0, cols});
}

} // namespace

template<class Elt>
struct Executor<Elt>::Impl {
    const Program<Elt>&  prog;
    const ModCtx<Elt>&   ctx;
    std::vector<DenseBuf<Elt>> bufs;
    std::vector<ResultSlot>& slots;
    std::vector<uint8_t> cmp_flags;      // per (node, tile): 1 = differs
    std::vector<uint32_t> cmp_base;      // node -> first flag index
    std::vector<uint32_t> cmp_count;     // node -> number of tiles

    // ENGINE_TRACE_NODES: per-worker, per-op busy seconds and task counts
    // (each worker writes its own row; merged on the calling thread after
    // the run).
    bool trace = false;
    std::vector<double>   trace_busy;    // [worker * kTraceOps + op]
    std::vector<uint64_t> trace_tasks;

    Impl(const Program<Elt>& p, const ModCtx<Elt>& c, std::vector<ResultSlot>& s) : prog(p), ctx(c), slots(s) {}

    void run_task_traced(const Task& t, uint32_t worker) {
        if (!trace) { run_task(t); return; }
        const auto b0 = std::chrono::steady_clock::now();
        run_task(t);
        const double b = std::chrono::duration<double>(std::chrono::steady_clock::now() - b0).count();
        const uint32_t op = static_cast<uint32_t>(prog.nodes[t.node].op);
        trace_busy[worker * kTraceOps + op] += b;
        ++trace_tasks[worker * kTraceOps + op];
    }

    DenseView<Elt> view(const ViewRef& v) {
        return DenseView<Elt>(bufs[v.buf], v.row0, v.rows, v.col0, v.cols, v.transposed);
    }
    DenseView<const Elt> cv(const ViewRef& v) { return DenseView<const Elt>(view(v)); }

    void gemm_tile(const Node<Elt>& n, const ViewRef& A, const ViewRef& B, const ViewRef& C, const Task& t) {
        const ViewRef a = A.sub(t.r0, t.nr, 0, A.lcols());
        const ViewRef b = B.sub(0, B.lrows(), t.c0, t.nc);
        const ViewRef c = C.sub(t.r0, t.nr, t.c0, t.nc);
        if (a.lcols() == 0) {
            if (n.gop == GemmOp::Set) BlockMat<Elt>::zero_view(view(c));
            return;
        }
        DenseView<Elt> cvw = view(c);
        // Fast path: plain kernel with the vectorised row-major epilogue.  A
        // scaled product, or an output tile whose rows are not contiguous
        // (column-major C: the kernel would reduce cell by cell), goes through
        // a row-major thread-local scratch and one exact copy / scale / axpy.
        if (n.alpha == Elt(1) && cvw.row_contiguous()) { kernels::gemm<Elt>(ctx, cv(a), cv(b), cvw, n.gop); return; }
        std::vector<Elt>& s = scratch_vec<Elt>();
        const size_t need = static_cast<size_t>(t.nr) * t.nc;
        if (s.size() < need) s.resize(need);
        DenseBuf<Elt> sb(s.data(), s.size(), t.nr, t.nc, Layout::RowMajor);
        kernels::gemm<Elt>(ctx, cv(a), cv(b), engine::view(sb), GemmOp::Set);
        if (n.gop == GemmOp::Set) {
            if (n.alpha == Elt(1)) kernels::copy<Elt>(cview(sb), cvw);
            else
                for (uint32_t i = 0; i < t.nr; ++i)
                    for (uint32_t j = 0; j < t.nc; ++j) cvw.at(i, j) = ctx.mul(n.alpha, sb.at(i, j));
        } else {
            kernels::axpy<Elt>(ctx, n.alpha, cview(sb), cvw, n.gop == GemmOp::Sub);
        }
    }

    void run_task(const Task& t) {
        const Node<Elt>& n = prog.nodes[t.node];
        switch (n.op) {
            case Op::GEMM: case Op::SCHUR_UPDATE:
                gemm_tile(n, n.a, n.b, n.c, t);
                break;
            case Op::BATCHED_GEMM: {
                const GemmInstance& g = prog.instances[t.inst];
                gemm_tile(n, g.a, g.b, g.c, t);
                break;
            }
            case Op::TRSM: {
                const ViewRef b = n.b.sub(t.r0, t.nr, t.c0, t.nc);
                const ViewRef c = n.c.sub(t.r0, t.nr, t.c0, t.nc);
                if (n.pivot_kind == PivotKind::General) {
                    const ViewRef bb = n.b.sub(0, n.b.lrows(), t.c0, t.nc);
                    const ViewRef aa = n.a.sub(t.r0, t.nr, 0, n.a.lcols());
                    kernels::gemm<Elt>(ctx, cv(aa), cv(bb), view(c), GemmOp::Set);
                } else if (n.pivot_kind == PivotKind::Identity) {
                    kernels::copy<Elt>(cv(b), view(c));
                } else {
                    DenseView<const Elt> bv = cv(b); DenseView<Elt> cw = view(c);
                    for (uint32_t i = 0; i < t.nr; ++i)
                        for (uint32_t j = 0; j < t.nc; ++j) cw.at(i, j) = ctx.mul(n.alpha, bv.at(i, j));
                }
                break;
            }
            case Op::AXPY: {
                const ViewRef c = n.c.sub(t.r0, t.nr, t.c0, t.nc);
                DenseView<Elt> cw = view(c);
                if (n.a.buf == kIdentityBuf) {
                    // diagonal entries of the tile: logical (i, i) with i in [r0, r0+nr) and j == i
                    for (uint32_t i = 0; i < t.nr; ++i) {
                        const uint32_t g = t.r0 + i;
                        if (g < t.c0 || g >= t.c0 + t.nc) continue;
                        Elt& x = cw.at(i, g - t.c0);
                        x = n.subtract ? ctx.sub(x, n.alpha) : ctx.add(x, n.alpha);
                    }
                } else {
                    kernels::axpy<Elt>(ctx, n.alpha, cv(n.a.sub(t.r0, t.nr, t.c0, t.nc)), cw, n.subtract);
                }
                break;
            }
            case Op::SCAL: {
                DenseView<Elt> cw = view(n.c.sub(t.r0, t.nr, t.c0, t.nc));
                if (n.alpha == 0) BlockMat<Elt>::zero_view(cw);
                else kernels::scal<Elt>(ctx, n.alpha, cw);
                break;
            }
            case Op::TRANSPOSE: case Op::SCATTER:
                kernels::copy<Elt>(cv(n.a.sub(t.r0, t.nr, t.c0, t.nc)), view(n.c.sub(t.r0, t.nr, t.c0, t.nc)));
                break;
            case Op::CANONICALIZE:
                kernels::canonicalise<Elt>(ctx, view(n.c.sub(t.r0, t.nr, t.c0, t.nc)));
                break;
            case Op::GATHER: {
                DenseView<Elt> cw = view(n.c.sub(t.r0, t.nr, 0, n.c.lcols()));
                const bool ident = (n.a.buf == kIdentityBuf);
                const std::vector<uint32_t>* rows = (n.row_list == kNoList) ? nullptr : &prog.index_lists[n.row_list];
                DenseView<const Elt> av;
                if (!ident) av = rows ? cv(n.a) : cv(n.a.sub(t.r0, t.nr, 0, n.a.lcols()));
                const std::vector<uint32_t>* cols = (n.index_list == kNoList) ? nullptr : &prog.index_lists[n.index_list];
                const uint32_t nc = cols ? static_cast<uint32_t>(cols->size()) : n.a.lcols();
                for (uint32_t i = 0; i < t.nr; ++i) {
                    const uint32_t ar = rows ? (*rows)[t.r0 + i] : i;        // row of av
                    const uint32_t grow = n.aux + (rows ? (*rows)[t.r0 + i] : t.r0 + i);
                    for (uint32_t tt = 0; tt < nc; ++tt) {
                        const uint32_t col = cols ? (*cols)[tt] : tt;
                        Elt v = 0;
                        if (!ident && n.beta != 0) v = (n.beta == Elt(1)) ? av.at(ar, col) : ctx.mul(n.beta, av.at(ar, col));
                        if (n.alpha != 0 && grow == col) v = ctx.add(v, n.alpha);
                        Elt& x = cw.at(i, tt);
                        x = n.accumulate ? ctx.add(x, v) : v;
                    }
                }
                break;
            }
            case Op::PANEL_RREF: {
                const elim::RrefResult r = elim::rref<Elt>(ctx, view(n.c));
                ResultSlot& s = slots[n.slot_out];
                s.rank = r.rank; s.pivots = r.pivots; s.written = true;
                break;
            }
            case Op::PIVOT_SEARCH: {
                const ResultSlot& q = slots[n.slot_in];
                ENGINE_CHECK_MSG(q.written, "PIVOT_SEARCH: pivot slot not written");
                const uint32_t kj = static_cast<uint32_t>(q.pivots.size());
                DenseView<const Elt> S = cv(n.a);
                const uint32_t nc = S.rows();
                ResultSlot& s = slots[n.slot_out];
                s.pivots.clear(); s.rank = kj; s.written = true;
                if (kj == 0) break;
                DenseBuf<Elt> Pt(kj, nc);
                for (uint32_t tt = 0; tt < kj; ++tt)
                    for (uint32_t i = 0; i < nc; ++i) Pt.at(tt, i) = S.at(i, q.pivots[tt]);
                // The row rank profile of S[:, Q] is the column rank profile of
                // its transpose: the PLU's pivots alone, no back-substitution.
                const elim::RrefResult rp = elim::column_rank_profile<Elt>(ctx, engine::view(Pt));
                ENGINE_CHECK_MSG(rp.rank == kj, "PIVOT_SEARCH: row rank profile lost rank");
                s.pivots = rp.pivots;
                DenseView<Elt> B = view(n.c);
                for (uint32_t tt = 0; tt < kj; ++tt)
                    for (uint32_t u = 0; u < kj; ++u) B.at(tt, u) = S.at(s.pivots[tt], q.pivots[u]);
                break;
            }
            case Op::COMPARE: {
                DenseView<const Elt> a = cv(n.a.sub(t.r0, t.nr, 0, n.a.lcols()));
                DenseView<const Elt> b = cv(n.b.sub(t.r0, t.nr, 0, n.b.lcols()));
                uint8_t differs = 0;
                if (a.rows() != b.rows() || a.cols() != b.cols()) differs = 1;
                else
                    for (uint32_t i = 0; i < a.rows() && !differs; ++i)
                        for (uint32_t j = 0; j < a.cols(); ++j)
                            if (a.at(i, j) != b.at(i, j)) { differs = 1; break; }
                cmp_flags[cmp_base[t.node] + t.tile] = differs;
                break;
            }
            case Op::H2D: case Op::D2H:
                break;   // never expanded into a task (see expand())
        }
    }

    // Expand a node into tasks; `target` = the thread count this node may use.
    void expand(uint32_t ni, uint32_t target, std::vector<Task>& tasks, std::vector<std::array<uint32_t, 4>>& tmp) {
        const Node<Elt>& n = prog.nodes[ni];
        auto push_tiles = [&](uint32_t inst) {
            uint32_t k = 0;
            for (const auto& tl : tmp) tasks.push_back(Task{ni, tl[0], tl[1], tl[2], tl[3], inst, k++});
        };
        switch (n.op) {
            case Op::GEMM: case Op::SCHUR_UPDATE:
                gemm_tiles(n.c.lrows(), n.c.lcols(), n.a.lcols(), target * tile_mult(), tmp); push_tiles(0); break;
            case Op::TRSM:
                if (n.pivot_kind == PivotKind::General) gemm_tiles(n.c.lrows(), n.c.lcols(), n.b.lrows(), target * tile_mult(), tmp);
                else row_chunks(n.c.lrows(), n.c.lcols(), target, tmp);
                push_tiles(0); break;
            case Op::BATCHED_GEMM: {
                const uint32_t ninst = n.batch_end - n.batch_begin;
                const uint32_t per = (ninst >= target) ? 1 : (target * tile_mult() + ninst - 1) / ninst;
                for (uint32_t i = n.batch_begin; i < n.batch_end; ++i) {
                    const GemmInstance& g = prog.instances[i];
                    gemm_tiles(g.c.lrows(), g.c.lcols(), g.a.lcols(), per, tmp); push_tiles(i);
                }
                break;
            }
            case Op::AXPY: case Op::SCAL: case Op::TRANSPOSE: case Op::SCATTER: case Op::CANONICALIZE:
                row_chunks(n.c.lrows(), n.c.lcols(), target, tmp); push_tiles(0); break;
            case Op::GATHER:
                row_chunks(n.c.lrows(), n.c.lcols(), target, tmp); push_tiles(0); break;
            case Op::COMPARE:
                row_chunks(n.a.lrows(), n.a.lcols(), target, tmp);
                cmp_base[ni] = static_cast<uint32_t>(cmp_flags.size());
                cmp_count[ni] = static_cast<uint32_t>(tmp.size());
                cmp_flags.resize(cmp_flags.size() + tmp.size(), 0);
                push_tiles(0); break;
            case Op::PANEL_RREF: case Op::PIVOT_SEARCH:
                tasks.push_back(Task{ni, 0, 0, 0, 0, 0, 0}); break;
            case Op::H2D: case Op::D2H:
                break;   // transfer nodes are the backend's; nothing to do without a device
        }
    }
};

template<class Elt>
Executor<Elt>::Executor(ThreadPool* pool) : pool_(pool != nullptr ? pool : &ThreadPool::global()) {
    // No device probe here: see device_usable().
}

template<class Elt>
Executor<Elt>::~Executor() = default;

template<class Elt>
bool Executor<Elt>::device_usable() const {
    if (!backend_ && DeviceBackend<Elt>::usable()) backend_.reset(new DeviceBackend<Elt>());
    return backend_ != nullptr;
}
template<class Elt>
const DeviceStats& Executor<Elt>::device_stats() const {
    static const DeviceStats none;
    return backend_ ? backend_->stats() : none;
}
template<class Elt>
void Executor<Elt>::release_device_state() { if (backend_) backend_->release(); }
template<class Elt>
size_t Executor<Elt>::device_bytes() const { return backend_ ? backend_->device_bytes() : 0; }
template<class Elt>
bool Executor<Elt>::elim_session_begin() {
    if (!device_usable() || backend_->session_open()) return false;
    backend_->session_begin();
    return true;
}
template<class Elt>
bool Executor<Elt>::elim_session_open() const { return backend_ && backend_->session_open(); }
template<class Elt>
void Executor<Elt>::elim_session_adopt(const DenseBuf<Elt>& host, bool keep) {
    if (backend_ && backend_->session_open()) backend_->session_adopt(host, keep);
}
template<class Elt>
void Executor<Elt>::elim_session_release(const Elt* base) {
    if (backend_ && backend_->session_open()) backend_->session_release(base);
}
template<class Elt>
void Executor<Elt>::elim_session_end(bool download) {
    if (backend_ && backend_->session_open()) backend_->session_end(download);
}

// ENGINE_VERIFY_DEVICE: the GPU nodes of `level` have run and the host
// waited; re-execute them on the CPU (into the host buffers, which
// verify_sync_host made current before the launch) and compare every window
// they write with the device copy, entrywise.  A difference is a bug, never
// noise (plan Sec. 4.3): throw.
template<class Elt>
void Executor<Elt>::verify_level(Impl& st, uint32_t level) {
    std::vector<std::pair<BufId, DenseBuf<Elt>>> snap;
    backend_->verify_snapshot(level, snap);
    const std::vector<uint32_t>& gpu = backend_->gpu_nodes(level);
    std::vector<Task> compute, memory;
    std::vector<std::array<uint32_t, 4>> tmp;
    const uint32_t nthreads_c = std::min<uint32_t>(pool_->size(), kThreadsComputeBound);
    const uint32_t nthreads_m = std::min<uint32_t>(pool_->size(), kThreadsMemoryBound);
    for (uint32_t ni : gpu) {
        if (thread_class(st.prog.nodes[ni].op) == ThreadClass::ComputeBound) st.expand(ni, nthreads_c, compute, tmp);
        else st.expand(ni, nthreads_m, memory, tmp);
    }
    if (!compute.empty()) pool_->parallel_for(static_cast<uint32_t>(compute.size()), nthreads_c, [&](uint32_t t, uint32_t) { st.run_task(compute[t]); });
    if (!memory.empty()) pool_->parallel_for(static_cast<uint32_t>(memory.size()), nthreads_m, [&](uint32_t t, uint32_t) { st.run_task(memory[t]); });
    std::vector<Access> acc;
    for (uint32_t ni : gpu) {
        const Node<Elt>& n = st.prog.nodes[ni];
        node_accesses(st.prog, n, acc);
        for (const Access& a : acc) {
            if (!a.write) continue;
            const DenseBuf<Elt>* dev = nullptr;
            for (const auto& kv : snap) if (kv.first == a.v.buf) { dev = &kv.second; break; }
            ENGINE_CHECK_MSG(dev != nullptr, "verify_device: no device snapshot for a written buffer");
            const DenseView<const Elt> h = st.cv(a.v);
            const DenseView<const Elt> d(*dev, a.v.row0, a.v.rows, a.v.col0, a.v.cols, a.v.transposed);
            if (!same_entries(h, d)) {
                std::string msg = "ENGINE_VERIFY_DEVICE: GPU node " + std::to_string(ni) + " (" + op_name(n.op) +
                                  ") at level " + std::to_string(level) + " differs from the CPU result in buffer " +
                                  std::to_string(a.v.buf) + " window [" + std::to_string(a.v.row0) + "+" + std::to_string(a.v.rows) +
                                  ", " + std::to_string(a.v.col0) + "+" + std::to_string(a.v.cols) + "]";
                throw std::logic_error(msg);
            }
        }
    }
    backend_->verify_mark_synced(level);
}

template<class Elt>
RunResult Executor<Elt>::run(Program<Elt>& prog, const Bindings<Elt>& b, const PassOptions& opts) {
    if (!prog.compiled) compile(prog, opts);
    RunResult out;
    execute(prog, b, out);
    return out;
}

template<class Elt>
RunResult Executor<Elt>::run(const Program<Elt>& prog, const Bindings<Elt>& b) {
    ENGINE_CHECK_MSG(prog.compiled, "Executor::run: program not compiled (call compile() or the mutable overload)");
    RunResult out;
    execute(prog, b, out);
    return out;
}

template<class Elt>
void Executor<Elt>::execute(const Program<Elt>& prog, const Bindings<Elt>& b, RunResult& out) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    const ModCtx<Elt>& ctx = b.has_ctx ? b.ctx : prog.ctx;
    if (b.has_ctx) {
        ENGINE_CHECK_MSG(b.ctx.regime == prog.ctx.regime, "Executor::run: replay under a different prime regime");
        // ENGINE_API.md Sec. 13.7: a replay never crosses the declared ring --
        // a field program never runs on a composite (Z/ell^k) binding, nor the
        // reverse.  Structural only: no hash or key input changes.
        ENGINE_CHECK_MSG(b.ctx.is_field() == prog.ctx.is_field(),
                         "Executor::run: replay across the declared ring (field vs Z/ell^k, ENGINE_API.md Sec. 13.7)");
        for (const Node<Elt>& n : prog.nodes)
            if (!n.dead) ENGINE_CHECK_MSG(n.k_bound <= b.ctx.k_max, "Executor::run: replay prime has a smaller k_max than the program's k_bound");
    }
    ENGINE_CHECK_MSG(b.inputs.size() == prog.n_inputs && b.outputs.size() == prog.n_outputs,
                     "Executor::run: binding count does not match the program");
    out.slots.assign(prog.n_slots, ResultSlot());
    Impl st(prog, ctx, out.slots);
    st.bufs.reserve(prog.bufs.size());
    st.cmp_base.assign(prog.nodes.size(), 0);
    st.cmp_count.assign(prog.nodes.size(), 0);

    // arena
    if (prog.opts.arena) {
        if (arena_.size() < prog.arena_elts) arena_.resize(prog.arena_elts);
        if (poison_) {
            const Elt p = ctx.p;
            for (size_t i = 0; i < arena_.size(); ++i) arena_[i] = static_cast<Elt>((i * 2654435761ull + 12345ull) % p);
        }
    } else {
        own_.clear();
    }
    for (BufId bid = 0; bid < prog.bufs.size(); ++bid) {
        const BufferDesc& d = prog.bufs[bid];
        switch (d.kind) {
            case BufKind::Input: {
                const DenseView<const Elt>& v = b.inputs[d.slot];
                const ViewShape sh = view_shape(v);
                ENGINE_CHECK_MSG(sh.rows == d.rows && sh.cols == d.cols && sh.layout == d.layout,
                                 "Executor::run: an input binding's shape or layout differs from the recording");
                Elt* base = const_cast<Elt*>(v.data());
                const size_t cap = static_cast<size_t>(sh.layout == Layout::RowMajor ? sh.rows : sh.cols) * sh.ld;
                st.bufs.emplace_back(base, std::max<size_t>(cap, 1), d.rows, d.cols, sh.layout, sh.ld);
                break;
            }
            case BufKind::Output: {
                const DenseView<Elt>& v = b.outputs[d.slot];
                const ViewShape sh = view_shape(v);
                ENGINE_CHECK_MSG(sh.rows == d.rows && sh.cols == d.cols && sh.layout == d.layout,
                                 "Executor::run: an output binding's shape or layout differs from the recording");
                const size_t cap = static_cast<size_t>(sh.layout == Layout::RowMajor ? sh.rows : sh.cols) * sh.ld;
                st.bufs.emplace_back(v.data(), std::max<size_t>(cap, 1), d.rows, d.cols, sh.layout, sh.ld);
                break;
            }
            case BufKind::Temp: {
                if (d.footprint == 0) {
                    // unreferenced (a pass removed every use) or zero-sized: no storage
                    if (d.rows == 0 || d.cols == 0) st.bufs.emplace_back(d.rows, d.cols, d.layout);
                    else st.bufs.emplace_back();
                    break;
                }
                if (prog.opts.arena) {
                    ENGINE_CHECK(d.arena_offset + d.footprint <= arena_.size());
                    st.bufs.emplace_back(arena_.data() + d.arena_offset, d.footprint, d.rows, d.cols, d.layout, d.ld);
                } else {
                    own_.emplace_back(d.rows, d.cols, d.layout, d.ld);
                    DenseBuf<Elt>& o = own_.back();
                    st.bufs.emplace_back(o.data(), o.footprint(), d.rows, d.cols, d.layout, d.ld);
                }
                break;
            }
        }
    }

    std::vector<Task> compute, memory, serial;
    std::vector<std::array<uint32_t, 4>> tmp;
    const uint32_t nthreads_c = std::min<uint32_t>(pool_->size(), kThreadsComputeBound);
    const uint32_t nthreads_m = std::min<uint32_t>(pool_->size(), kThreadsMemoryBound);
    uint32_t ntasks = 0;
    st.trace = trace_nodes_enabled();
    if (st.trace) {
        st.trace_busy.assign(static_cast<size_t>(pool_->size()) * kTraceOps, 0.0);
        st.trace_tasks.assign(static_cast<size_t>(pool_->size()) * kTraceOps, 0);
    }
    double trace_class_wall[kTraceClasses] = {0.0, 0.0, 0.0};
    double trace_class_busy[kTraceClasses] = {0.0, 0.0, 0.0};
    uint64_t trace_class_regions[kTraceClasses] = {0, 0, 0};
    uint64_t trace_class_threads[kTraceClasses] = {0, 0, 0};
    auto busy_snapshot = [&]() {
        double b = 0.0;
        if (st.trace) for (double x : st.trace_busy) b += x;
        return b;
    };
    // Phase 5: the device arm.  Without a device every GPU-assigned node runs
    // on the CPU and the transfer nodes are skipped (a value never depends on
    // the device: plan D4).
    bool want_gpu = prog.stats.gpu_nodes > 0 || prog.stats.h2d_nodes > 0 || prog.stats.d2h_nodes > 0;
    bool gpu = false;
    if (want_gpu && prog.stats.gpu_compute_nodes == 0 && !(backend_ && backend_->session_open()) && gpu_lean_enabled()) {
        // (slot-23 P7) the VRAM-lean rule (Executor.h): memory-class GPU nodes only -> the host path
        want_gpu = false;
        ++stats_.lean_host_runs;
        g_lean_host_runs.fetch_add(1, std::memory_order_relaxed);
        note_lean_host_run();   // (audit R4) the thread's DeviceTally, if any
    }
    if (want_gpu) {
        const bool wants_device = prog.stats.gpu_nodes != 0 || prog.stats.h2d_nodes != 0 || prog.stats.d2h_nodes != 0;
        if (wants_device && device_usable()) gpu = backend_->begin_run(prog, st.bufs, ctx, verify_);
        else stats_.fallback_nodes += prog.stats.gpu_nodes;
    } else if (backend_ && backend_->session_open()) {
        // (slot-18 WP-19a) a host-only program inside an elimination session:
        // the adopted buffers it binds are synchronised (DeviceBackend.h).
        backend_->session_host_run(prog, st.bufs);
    }
    for (uint32_t L = 0; L < prog.levels.size(); ++L) {
        const std::vector<uint32_t>& level = prog.levels[L];
        compute.clear(); memory.clear(); serial.clear();
        if (gpu) {
            backend_->transfers(L);
            if (verify_) backend_->verify_sync_host(L);
        }
        for (uint32_t ni : level) {
            const Node<Elt>& n = prog.nodes[ni];
            if (is_transfer(n.op)) continue;
            if (gpu && n.device == Device::GPU) continue;
            const ThreadClass tc = thread_class(n.op);
            if (tc == ThreadClass::ComputeBound) st.expand(ni, nthreads_c, compute, tmp);
            else if (tc == ThreadClass::Serial) st.expand(ni, 1, serial, tmp);
            else st.expand(ni, nthreads_m, memory, tmp);
        }
        if (gpu) backend_->launch(L);
        const auto tl0 = Clock::now();
        const double bs0 = busy_snapshot();
        if (!compute.empty())
            pool_->parallel_for(static_cast<uint32_t>(compute.size()), nthreads_c,
                                [&](uint32_t t, uint32_t w) { st.run_task_traced(compute[t], w); });
        const auto tl1 = Clock::now();
        if (st.trace && !compute.empty()) {
            const double bs1 = busy_snapshot();
            trace_class_wall[1] += std::chrono::duration<double>(tl1 - tl0).count();
            trace_class_busy[1] += bs1 - bs0; ++trace_class_regions[1];
            trace_class_threads[1] += std::min<uint32_t>(nthreads_c, static_cast<uint32_t>(compute.size()));
        }
        const double bs2 = busy_snapshot();
        if (!memory.empty())
            pool_->parallel_for(static_cast<uint32_t>(memory.size()), nthreads_m,
                                [&](uint32_t t, uint32_t w) { st.run_task_traced(memory[t], w); });
        if (st.trace && !memory.empty()) {
            const double bs3 = busy_snapshot();
            trace_class_wall[0] += std::chrono::duration<double>(Clock::now() - tl1).count();
            trace_class_busy[0] += bs3 - bs2; ++trace_class_regions[0];
            trace_class_threads[0] += std::min<uint32_t>(nthreads_m, static_cast<uint32_t>(memory.size()));
        }
        if (!serial.empty()) {
            // Host-serial nodes (PANEL_RREF / PIVOT_SEARCH) run on the calling
            // thread, OUTSIDE a pool region, with the elimination's inner
            // parallel hook bound to the pool: the panel RREF's trailing
            // updates then split over the workers (DenseElim.h).
            const elim::ParallelFor hook = [&](uint32_t nt, const std::function<void(uint32_t)>& fn) {
                pool_->parallel_for(nt, nthreads_c, [&](uint32_t t, uint32_t) { fn(t); });
            };
            elim::ScopedParallelHook sh(&hook);
            const auto ts0 = Clock::now();
            for (const Task& t : serial) st.run_task_traced(t, 0);
            const double sw = std::chrono::duration<double>(Clock::now() - ts0).count();
            out.serial_seconds += sw;
            if (st.trace) { trace_class_wall[2] += sw; trace_class_busy[2] += sw; ++trace_class_regions[2]; trace_class_threads[2] += 1; }
        }
        if (gpu) {
            backend_->wait(L);
            backend_->host_wrote(L);
            if (verify_ && backend_->level_has_gpu(L)) verify_level(st, L);
        }
        if (trace_levels()) {
            const auto tl2 = Clock::now();
            std::cout << "    level " << L << ": " << level.size() << " nodes, compute "
                      << compute.size() << " tasks " << std::chrono::duration<double>(tl1 - tl0).count() * 1e3 << " ms, memory "
                      << memory.size() << " tasks " << std::chrono::duration<double>(tl2 - tl1).count() * 1e3 << " ms"
                      << (gpu && backend_->level_has_gpu(L) ? " (+ device)" : "") << std::endl;
        }
        ntasks += static_cast<uint32_t>(compute.size() + memory.size());
        // COMPARE reductions of this level (host, deterministic)
        for (uint32_t ni : level) {
            const Node<Elt>& n = prog.nodes[ni];
            if (n.op != Op::COMPARE) continue;
            bool eq = true;
            for (uint32_t k = 0; k < st.cmp_count[ni]; ++k)
                if (st.cmp_flags[st.cmp_base[ni] + k]) eq = false;
            out.slots[n.slot_out].equal = eq;
            out.slots[n.slot_out].written = true;
        }
    }
    if (gpu) backend_->end_run();
    out.levels = static_cast<uint32_t>(prog.levels.size());
    out.tasks = ntasks;
    out.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    ++stats_.runs; stats_.nodes += prog.live_nodes(); stats_.tasks += ntasks; stats_.seconds += out.seconds;
    if (st.trace) {
        trace_add_execute(out.seconds);
        for (uint32_t c = 0; c < kTraceClasses; ++c)
            if (trace_class_regions[c]) trace_add_class(c, trace_class_wall[c], trace_class_busy[c], static_cast<uint32_t>(trace_class_threads[c]));
        uint64_t nodes_by_op[kTraceOps] = {};
        for (const Node<Elt>& n : prog.nodes) if (!n.dead) ++nodes_by_op[static_cast<uint32_t>(n.op)];
        for (uint32_t op = 0; op < kTraceOps; ++op) {
            double b = 0.0; uint64_t k = 0;
            for (uint32_t w = 0; w < pool_->size(); ++w) { b += st.trace_busy[w * kTraceOps + op]; k += st.trace_tasks[w * kTraceOps + op]; }
            if (nodes_by_op[op] || k) trace_add_op(op, nodes_by_op[op], k, b);
        }
    }
}

template class Executor<uint32_t>;
template class Executor<uint64_t>;

} // namespace tape
} // namespace engine
