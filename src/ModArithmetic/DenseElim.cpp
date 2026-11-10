// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "DenseElim.h"

#include <atomic>
#include <type_traits>

#include <chrono>
#include <mutex>
#include <thread>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "EngineCheck.h"
#include "Kernels.h"
#include "Binary/BitMatrix.h"
#include "Binary/TritMatrix.h"

// See DenseElim.h for the algorithm and the uniqueness argument.

namespace engine {
namespace elim {

// ---------------------------------------------------------------------------
// Tunables / backend switch
// ---------------------------------------------------------------------------
namespace {

uint32_t env_u32(const char* name, uint32_t dflt, uint32_t lo, uint32_t hi) {
    const char* e = std::getenv(name);
    if (e == nullptr || e[0] == '\0') return dflt;
    const long v = std::strtol(e, nullptr, 10);
    if (v < static_cast<long>(lo) || v > static_cast<long>(hi)) return dflt;
    return static_cast<uint32_t>(v);
}

} // namespace

namespace {
Params& mutable_params() {
    static Params p = [] {
        Params q;
        q.panel     = env_u32("ENGINE_ELIM_PANEL", q.panel, 1, 1u << 16);
        q.base      = env_u32("ENGINE_ELIM_BASE", q.base, 1, 1u << 12);
        q.trsm_base = env_u32("ENGINE_ELIM_TRSM_BASE", q.trsm_base, 1, 1u << 12);
        if (q.base > q.panel) q.base = q.panel;
        return q;
    }();
    return p;
}
} // namespace

const Params& params() { return mutable_params(); }

void set_params(const Params& p) {
    Params q = p;
    if (q.panel == 0) q.panel = 1;
    if (q.base == 0) q.base = 1;
    if (q.trsm_base == 0) q.trsm_base = 1;
    if (q.base > q.panel) q.base = q.panel;
    mutable_params() = q;
}

#if OC_ENABLE_LEGACY
#include "legacy/DenseElimLegacy.inc"   // (S24 CL-L2) ScopedLegacy, legacy_forced()
#endif

namespace {
thread_local const ParallelFor* g_parallel_hook = nullptr;
// Set while a hook task runs on this thread: a nested hook call from inside
// a task would re-enter the pool from a worker (deadlock), so it runs serially.
thread_local bool g_in_hook_task = false;
}

namespace {
std::mutex& plu_mutex() { static std::mutex m; return m; }
PluPhaseStats& plu_totals() { static PluPhaseStats t; return t; }
bool plu_trace_on() { static const bool on = std::getenv("ENGINE_TRACE_NODES") != nullptr; return on; }
using PluClock = std::chrono::steady_clock;
inline double plu_secs(PluClock::time_point a) { return std::chrono::duration<double>(PluClock::now() - a).count(); }
}
PluPhaseStats plu_phase_totals() { std::lock_guard<std::mutex> lk(plu_mutex()); return plu_totals(); }
void reset_plu_phase_totals() { std::lock_guard<std::mutex> lk(plu_mutex()); plu_totals() = PluPhaseStats(); }
const ParallelFor* parallel_hook() { return g_in_hook_task ? nullptr : g_parallel_hook; }
ScopedParallelHook::ScopedParallelHook(const ParallelFor* hook) : saved_(g_parallel_hook) { g_parallel_hook = hook; }
ScopedParallelHook::~ScopedParallelHook() { g_parallel_hook = saved_; }
ScopedInHookTask::ScopedInHookTask() : saved_(g_in_hook_task) { g_in_hook_task = true; }
ScopedInHookTask::~ScopedInHookTask() { g_in_hook_task = saved_; }

namespace {

using kernels::GemmOp;

// Runs fn(t) for t < ntasks over the installed hook, marking the tasks so a
// nested hook call falls back to serial execution.  Every caller splits
// DISJOINT rows or columns whose entries are exact modular values computed
// by the same operation sequence whatever the split, so no tiling decision
// can move a bit (ENGINE_API Sec. 3 / D4).
inline void run_hook(const ParallelFor* par, uint32_t ntasks, const std::function<void(uint32_t)>& fn) {
    (*par)(ntasks, [&](uint32_t t) {
        const bool saved = g_in_hook_task;
        g_in_hook_task = true;
        struct Restore { bool s; ~Restore() { g_in_hook_task = s; } } restore{saved};
        fn(t);
    });
}
// Tile count for splitting `extent` (rows or columns) of a step costing
// `work` modular MACs over the hook: 0 = run serially.
inline uint32_t hook_tiles(const ParallelFor* par, uint32_t extent, size_t work, uint32_t min_extent, size_t min_work, uint32_t min_tile) {
    if (par == nullptr || extent < min_extent || work < min_work) return 0;
    return std::max<uint32_t>(2, std::min<uint32_t>(16, extent / min_tile));
}
// (S24 PAR-L1) the module routes' tile count is also bounded by the WORK: at least mod_tile_work() modular MACs per tile, and
// serial below two such tiles.  Measured: at (N, n) = (1, 60) the descent record's torsion modules opened ~170 000 hook regions
// of ~15 tiles of ~8 000 MACs each (howell_rows, cyclic_decomposition), whose pool wake-up / registry traffic made 16 threads
// 40 % slower than 1 (PARL1_report.md).  Tiling is value-neutral (run_hook above), so the count cannot move a bit.  Rollback
// lever HECKE_MODULE_TILE_WORK: exactly 0 restores the extent-only count of hook_tiles verbatim; 1 or unset = the default bound
// kModTileWork; a number >= 2 = that bound (measurement knob); anything else warns and keeps the default.  Read once per process.  The PLU hooks (the tape's) are untouched.
constexpr size_t kModTileWork = size_t(1) << 20;   // ~1 M MACs (~0.3-1 ms) per tile: measured, PARL1_report.md §3
std::atomic<uint64_t> g_mod_tiles_capped{0}, g_mod_tiles_serial{0};
std::atomic<long long> g_mod_tile_work_scoped{-1};   // >= 0: ScopedModuleTileWork
size_t mod_tile_work_env() {
    static const size_t w = [] {
        const char* e = std::getenv("HECKE_MODULE_TILE_WORK");
        if (e == nullptr || std::string(e) == "1") return kModTileWork;
        if (std::string(e) == "0") return size_t(0);
        char* end = nullptr;   // a number >= 2: that bound (a measurement knob; tiling cannot move a value)
        const unsigned long long v = std::strtoull(e, &end, 10);
        if (end != e && *end == '\0' && v >= 2) return static_cast<size_t>(v);
        std::fprintf(stderr, "WARNING HECKE_MODULE_TILE_WORK=%s is not a number: the default work bound stays on\n", e);
        return kModTileWork;
    }();
    return w;
}
size_t mod_tile_work() {
    const long long o = g_mod_tile_work_scoped.load(std::memory_order_relaxed);
    return o >= 0 ? static_cast<size_t>(o) : mod_tile_work_env();
}
void print_module_tile_work_effective_at_exit() {
    const char* e = std::getenv("HECKE_MODULE_TILE_WORK");
    std::fprintf(stdout, "EFFECTIVE HECKE_MODULE_TILE_WORK mode=%s source=env (HECKE_MODULE_TILE_WORK=%s) min_tile_work=%zu capped=%llu serial=%llu\n",
                 mod_tile_work() ? "bounded" : "extent", e ? e : "", mod_tile_work(),
                 static_cast<unsigned long long>(g_mod_tiles_capped.load()), static_cast<unsigned long long>(g_mod_tiles_serial.load()));
    std::fflush(stdout);
}
[[maybe_unused]] const bool g_mod_tile_work_noted_at_load = [] {
    if (std::getenv("HECKE_MODULE_TILE_WORK") != nullptr) std::atexit(&print_module_tile_work_effective_at_exit);
    return true;
}();
inline uint32_t mod_tiles(const ParallelFor* par, uint32_t extent, size_t work, uint32_t min_extent, size_t min_work, uint32_t min_tile) {
    const uint32_t t = hook_tiles(par, extent, work, min_extent, min_work, min_tile);
    const size_t w = mod_tile_work();
    if (t == 0 || w == 0) return t;
    const size_t cap = work / w;
    if (cap < 2) { g_mod_tiles_serial.fetch_add(1, std::memory_order_relaxed); return 0; }
    if (cap < t) { g_mod_tiles_capped.fetch_add(1, std::memory_order_relaxed); return static_cast<uint32_t>(cap); }
    return t;
}
// (S24 MT-1) cyclic_decomposition's split thresholds; under ScopedModuleMtForceSplit (validation only) every step with
// at least 2 rows / columns splits (2-16 tiles), so the split code is exercised on small inputs.  Tiling is
// value-neutral either way.
std::atomic<int> g_mt_force_split{0};
inline uint32_t cyc_tiles(const ParallelFor* par, uint32_t extent, size_t work, uint32_t min_extent, size_t min_work, uint32_t min_tile) {
    if (par != nullptr && g_mt_force_split.load(std::memory_order_relaxed) != 0)
        return extent < 2 ? 0u : std::max<uint32_t>(2, std::min<uint32_t>(16, extent));
    return mod_tiles(par, extent, work, min_extent, min_work, min_tile);   // (S24 PAR-L1)
}

// ---------------------------------------------------------------------------
// Exact delayed-accumulation helpers.  Every accumulator holds at most
// ctx.k_max products of canonical residues (ENGINE_API.md Sec. 5).
// ---------------------------------------------------------------------------

// sum_{s<k} row[idx[s]] * col[s * cs]   (indexed gather on the row side)
template<class Elt>
inline Elt dot_indexed(const ModCtx<Elt>& ctx, const Elt* row, const uint32_t* idx,
                       const Elt* col, size_t cs, uint32_t k) {
    using Accum = typename ModCtx<Elt>::Accum;
    Elt total = 0;
    uint32_t s = 0;
    while (s < k) {
        const uint32_t kb = static_cast<uint32_t>(std::min<uint64_t>(k - s, ctx.k_max));
        Accum acc = 0;
        for (uint32_t t = s; t < s + kb; ++t) {
            acc += static_cast<Accum>(row[idx[t]]) * col[static_cast<size_t>(t) * cs];
        }
        total = ctx.add(total, ctx.reduce(acc));
        s += kb;
    }
    return total;
}

// target[j] -= sum_{s<k} coef[s * coef_stride] * rows[s * rs + j],  j < w.
// Row-vectorised: an exact Accum row of width w, reduced once per k-block.
template<class Elt>
void row_sub_combination(const ModCtx<Elt>& ctx, Elt* target, const Elt* coef, size_t coef_stride,
                         const Elt* rows, size_t rs, uint32_t k, uint32_t w) {
    using Accum = typename ModCtx<Elt>::Accum;
    if (k == 0 || w == 0) return;
    thread_local std::vector<Accum> acc;
    thread_local std::vector<Elt> tot;
    acc.assign(w, Accum(0));
    const bool blocked = ctx.k_max < k;
    if (blocked) tot.assign(w, Elt(0));
    uint32_t s = 0;
    while (s < k) {
        const uint32_t kb = static_cast<uint32_t>(std::min<uint64_t>(k - s, ctx.k_max));
        for (uint32_t t = s; t < s + kb; ++t) {
            const Elt c = coef[static_cast<size_t>(t) * coef_stride];
            if (c == 0) continue;
            const Elt* r = rows + static_cast<size_t>(t) * rs;
            const Accum ca = static_cast<Accum>(c);
            for (uint32_t j = 0; j < w; ++j) acc[j] += ca * r[j];
        }
        if (blocked) {
            for (uint32_t j = 0; j < w; ++j) {
                tot[j] = ctx.add(tot[j], ctx.reduce(acc[j]));
                acc[j] = 0;
            }
        }
        s += kb;
    }
    if (blocked) {
        for (uint32_t j = 0; j < w; ++j) target[j] = ctx.sub(target[j], tot[j]);
    } else {
        for (uint32_t j = 0; j < w; ++j) target[j] = ctx.sub(target[j], ctx.reduce(acc[j]));
    }
}

template<class Elt>
void zero_rows(DenseView<Elt> v) {
    const auto s = v.strides();
    for (uint32_t i = 0; i < v.rows(); ++i) {
        Elt* row = s.base + i * s.rs;
        for (uint32_t j = 0; j < v.cols(); ++j) row[j * s.cs] = 0;
    }
}

// ---------------------------------------------------------------------------
// Triangular solves, in place on B (k x w), B row-contiguous.
//   lower: B = L^{-1} B with L unit lower (strict lower triangle read)
//   upper: B = U^{-1} B with U unit upper (strict upper triangle read)
// Recursive: leaf = row-delayed substitution, split = TRSM / GEMM / TRSM.
// ---------------------------------------------------------------------------
template<class Elt>
void trsm_lower_unit(const ModCtx<Elt>& ctx, DenseView<const Elt> L, DenseView<Elt> B) {
    const uint32_t k = L.rows(), w = B.cols();
    ENGINE_CHECK(L.cols() == k && B.rows() == k);
    if (k <= 1 || w == 0) return;
    if (k <= params().trsm_base) {
        const auto ls = L.strides();
        const auto bs = B.strides();
        ENGINE_CHECK(bs.cs == 1);
        for (uint32_t i = 1; i < k; ++i) {
            row_sub_combination<Elt>(ctx, bs.base + i * bs.rs, ls.base + i * ls.rs, ls.cs,
                                     bs.base, bs.rs, i, w);
        }
        return;
    }
    const uint32_t k1 = k / 2, k2 = k - k1;
    trsm_lower_unit<Elt>(ctx, L.sub(0, k1, 0, k1), B.sub(0, k1, 0, w));
    kernels::gemm<Elt>(ctx, L.sub(k1, k2, 0, k1), B.sub(0, k1, 0, w), B.sub(k1, k2, 0, w), GemmOp::Sub);
    trsm_lower_unit<Elt>(ctx, L.sub(k1, k2, k1, k2), B.sub(k1, k2, 0, w));
}

template<class Elt>
void trsm_upper_unit(const ModCtx<Elt>& ctx, DenseView<const Elt> U, DenseView<Elt> B) {
    const uint32_t k = U.rows(), w = B.cols();
    ENGINE_CHECK(U.cols() == k && B.rows() == k);
    if (k <= 1 || w == 0) return;
    if (k <= params().trsm_base) {
        const auto us = U.strides();
        const auto bs = B.strides();
        ENGINE_CHECK(bs.cs == 1);
        for (uint32_t i = k - 1; i-- > 0;) {
            row_sub_combination<Elt>(ctx, bs.base + i * bs.rs,
                                     us.base + i * us.rs + (i + 1) * us.cs, us.cs,
                                     bs.base + (i + 1) * bs.rs, bs.rs, k - 1 - i, w);
        }
        return;
    }
    const uint32_t k1 = k / 2, k2 = k - k1;
    trsm_upper_unit<Elt>(ctx, U.sub(k1, k2, k1, k2), B.sub(k1, k2, 0, w));
    kernels::gemm<Elt>(ctx, U.sub(0, k1, k1, k2), B.sub(k1, k2, 0, w), B.sub(0, k1, 0, w), GemmOp::Sub);
    trsm_upper_unit<Elt>(ctx, U.sub(0, k1, 0, k1), B.sub(0, k1, 0, w));
}

// Column-tiled entry points: the columns of B are independent (each is its
// own triangular solve), so a wide B is split over the hook.
template<class Elt>
void trsm_lower_unit_par(const ModCtx<Elt>& ctx, DenseView<const Elt> L, DenseView<Elt> B) {
    const uint32_t k = L.rows(), w = B.cols();
    const uint32_t tiles = hook_tiles(parallel_hook(), w, static_cast<size_t>(k) * k * w / 2, 128, size_t(1) << 19, 32);
    if (tiles == 0) { trsm_lower_unit<Elt>(ctx, L, B); return; }
    run_hook(parallel_hook(), tiles, [&](uint32_t t) {
        const uint32_t c0 = static_cast<uint32_t>((static_cast<uint64_t>(w) * t) / tiles);
        const uint32_t c1 = static_cast<uint32_t>((static_cast<uint64_t>(w) * (t + 1)) / tiles);
        if (c1 > c0) trsm_lower_unit<Elt>(ctx, L, B.sub(0, k, c0, c1 - c0));
    });
}
template<class Elt>
void trsm_upper_unit_par(const ModCtx<Elt>& ctx, DenseView<const Elt> U, DenseView<Elt> B) {
    const uint32_t k = U.rows(), w = B.cols();
    const uint32_t tiles = hook_tiles(parallel_hook(), w, static_cast<size_t>(k) * k * w / 2, 128, size_t(1) << 19, 32);
    if (tiles == 0) { trsm_upper_unit<Elt>(ctx, U, B); return; }
    run_hook(parallel_hook(), tiles, [&](uint32_t t) {
        const uint32_t c0 = static_cast<uint32_t>((static_cast<uint64_t>(w) * t) / tiles);
        const uint32_t c1 = static_cast<uint32_t>((static_cast<uint64_t>(w) * (t + 1)) / tiles);
        if (c1 > c0) trsm_upper_unit<Elt>(ctx, U, B.sub(0, k, c0, c1 - c0));
    });
}

// ---------------------------------------------------------------------------
// The packed PLU driver.
// ---------------------------------------------------------------------------
template<class Elt>
struct Plu {
    const ModCtx<Elt>& ctx;
    DenseView<Elt> A;          // row-contiguous
    Elt* base;
    size_t rs;
    uint32_t m, n;
    uint32_t r = 0;            // pivots found so far; pivot i sits in row i
    std::vector<uint32_t> piv; // piv[i] = column of pivot i
    const Params& prm;

    PluPhaseStats ph;          // diagnostic (ENGINE_TRACE_NODES)
    const bool tr;

    Plu(const ModCtx<Elt>& c, DenseView<Elt> a)
        : ctx(c), A(a), base(a.strides().base), rs(a.strides().rs),
          m(a.rows()), n(a.cols()), prm(params()), tr(plu_trace_on()) {
        ENGINE_CHECK_MSG(a.strides().cs == 1, "Plu: view must be row-contiguous");
        piv.reserve(std::min(m, n));
    }
    ~Plu() {
        if (!tr) return;
        std::lock_guard<std::mutex> lk(plu_mutex());
        PluPhaseStats& t = plu_totals();
        t.base_panel += ph.base_panel; t.base_rows += ph.base_rows; t.base_rows_par += ph.base_rows_par;
        t.trsm += ph.trsm; t.trailing_gemm += ph.trailing_gemm; t.to_rref += ph.to_rref; t.total += ph.total; ++t.calls;
    }

    Elt* row(uint32_t i) const { return base + static_cast<size_t>(i) * rs; }

    void swap_rows(uint32_t i1, uint32_t i2) {
        std::swap_ranges(row(i1), row(i1) + n, row(i2));
    }

    // Left-looking (Crout) factorisation of columns [c0, c1) over rows [r, m),
    // using only the pivots found inside this call: on entry these columns
    // are already the Schur complement w.r.t. every earlier pivot.
    void base_panel(uint32_t c0, uint32_t c1) {
        const PluClock::time_point t0 = tr ? PluClock::now() : PluClock::time_point();
        struct Acc { double& d; PluClock::time_point t0; bool on; ~Acc() { if (on) d += plu_secs(t0); } } acc{ph.base_panel, t0, tr};
        const uint32_t r0 = r;
        // The multipliers of the most recent pivot column are scaled LAZILY:
        // instead of a strided pass over the column right after the pivot
        // choice, each active row scales its own entry at the start of the
        // next column's Schur step (the row is in cache there), and a final
        // pass covers the last pivot of the panel.  The scaled value is the
        // same product either way.
        uint32_t pend_col = 0; Elt pend_inv = 0; bool pending = false;
        auto scale_pending_rows = [&](uint32_t i0, uint32_t i1) {
            if (!pending) return;
            Elt* col = base + pend_col;
            for (uint32_t i = i0; i < i1; ++i) {
                Elt& v = col[static_cast<size_t>(i) * rs];
                if (v != 0) v = ctx.mul(v, pend_inv);
            }
        };
        // Every column of the panel is visited, even once every row has become
        // a pivot row (r == m): step (1) still has to turn the remaining
        // columns of the pivot rows into U entries.
        for (uint32_t c = c0; c < c1; ++c) {
            const uint32_t kk = r - r0;
            Elt* colc = base + c;
            const uint32_t* lp = kk ? &piv[r0] : nullptr;
            // (1) U entries of the local pivot rows (forward substitution)
            for (uint32_t t = 1; t < kk; ++t) {
                Elt* ri = row(r0 + t);
                const Elt s = dot_indexed<Elt>(ctx, ri, lp, colc + static_cast<size_t>(r0) * rs, rs, t);
                ri[c] = ctx.sub(ri[c], s);
            }
            if (r >= m) continue;                          // no active rows left
            // (2) Schur complement of the active rows in this column, fused
            //     with the pending multiplier scaling and the pivot scan (3):
            //     rows are independent, so the pass is split over the hook.
            uint32_t ip = m;
            const Elt* ucol = colc + static_cast<size_t>(r0) * rs;
            auto rows_pass = [&](uint32_t i0, uint32_t i1) -> uint32_t {
                scale_pending_rows(i0, i1);
                uint32_t first = m;
                for (uint32_t i = i0; i < i1; ++i) {
                    Elt* ri = row(i);
                    if (kk > 0) {
                        const Elt s = dot_indexed<Elt>(ctx, ri, lp, ucol, rs, kk);
                        ri[c] = ctx.sub(ri[c], s);
                    }
                    if (first == m && ri[c] != 0) first = i;
                }
                return first;
            };
            const uint32_t nact = m - r;
            const uint32_t tiles = hook_tiles(parallel_hook(), nact, static_cast<size_t>(nact) * std::max<uint32_t>(kk, 1), 1024, size_t(1) << 14, 128);
            const PluClock::time_point tp0 = tr ? PluClock::now() : PluClock::time_point();
            struct AccP { double& d; PluClock::time_point t0; bool on; uint32_t par; double& dp; ~AccP() { if (on) { const double s = plu_secs(t0); d += s; if (par) dp += s; } } } accp{ph.base_rows, tp0, tr, tiles, ph.base_rows_par};
            if (tiles == 0) {
                ip = rows_pass(r, m);
            } else {
                std::vector<uint32_t> firsts(tiles, m);
                run_hook(parallel_hook(), tiles, [&](uint32_t t) {
                    const uint32_t i0 = r + static_cast<uint32_t>((static_cast<uint64_t>(nact) * t) / tiles);
                    const uint32_t i1 = r + static_cast<uint32_t>((static_cast<uint64_t>(nact) * (t + 1)) / tiles);
                    firsts[t] = rows_pass(i0, i1);
                });
                for (uint32_t t = 0; t < tiles; ++t) if (firsts[t] < ip) ip = firsts[t];
            }
            pending = false;
            // (3) legacy pivot rule: topmost non-zero row from r downward
            if (ip == m) continue;                     // non-pivot column
            if (ip != r) swap_rows(ip, r);
            pend_inv = ctx.inv(colc[static_cast<size_t>(r) * rs]);
            pend_col = c; pending = true;              // multipliers scaled by the next pass
            piv.push_back(c);
            ++r;
        }
        scale_pending_rows(r, m);
    }

    // Apply the k pivots in rows [r0, r0 + k) to columns [c1, c2):
    //   U12 = L11^{-1} U12 (TRSM), A22 -= L21 U12 (GEMM).
    void trailing_update(uint32_t r0, uint32_t k, uint32_t c1, uint32_t c2) {
        const uint32_t w = c2 - c1;
        if (k == 0 || w == 0) return;
        // gather the packed L: unit lower k x k on top, (m - r0 - k) x k below
        const uint32_t mb = m - r0 - k;
        DenseBuf<Elt> Lk(k, k);
        for (uint32_t i = 1; i < k; ++i) {
            const Elt* ri = row(r0 + i);
            Elt* li = Lk.data() + static_cast<size_t>(i) * Lk.ld;
            for (uint32_t j = 0; j < i; ++j) li[j] = ri[piv[r0 + j]];
        }
        DenseView<Elt> U12 = A.sub(r0, k, c1, w);
        {
            const PluClock::time_point t0 = tr ? PluClock::now() : PluClock::time_point();
            trsm_lower_unit_par<Elt>(ctx, cview(Lk), U12);
            if (tr) ph.trsm += plu_secs(t0);
        }
        if (mb == 0) return;
        const PluClock::time_point tg0 = tr ? PluClock::now() : PluClock::time_point();
        struct Acc { double& d; PluClock::time_point t0; bool on; ~Acc() { if (on) d += plu_secs(t0); } } acc{ph.trailing_gemm, tg0, tr};
        // Row tiles of A22 are independent (each output row depends on its own
        // L21 row and the shared U12), so the update is split over the
        // installed hook when it is large enough to amortise a pool region.
        const ParallelFor* par = parallel_hook();
        const uint32_t tiles = (par != nullptr && mb >= 256 && static_cast<size_t>(mb) * w * k >= (size_t(1) << 20))
                                   ? std::min<uint32_t>(16, mb / 128) : 1;
        // The multipliers L21 sit at the pivot columns of the panel; when
        // those are contiguous (every panel column a pivot -- the common
        // case) L21 is a window of A and is read in place, else gathered.
        bool contiguous = true;
        for (uint32_t j = 1; j < k && contiguous; ++j) contiguous = (piv[r0 + j] == piv[r0] + j);
        const uint32_t lcol = piv[r0];
        auto tile = [&](uint32_t t) {
            const uint32_t i0 = static_cast<uint32_t>((static_cast<uint64_t>(mb) * t) / tiles);
            const uint32_t i1 = static_cast<uint32_t>((static_cast<uint64_t>(mb) * (t + 1)) / tiles);
            const uint32_t rows = i1 - i0;
            if (rows == 0) return;
            if (contiguous) {
                kernels::gemm<Elt>(ctx, DenseView<const Elt>(A.sub(r0 + k + i0, rows, lcol, k)), DenseView<const Elt>(U12),
                                   A.sub(r0 + k + i0, rows, c1, w), GemmOp::Sub);
                return;
            }
            DenseBuf<Elt> L21(rows, k);
            for (uint32_t i = 0; i < rows; ++i) {
                const Elt* ri = row(r0 + k + i0 + i);
                Elt* li = L21.data() + static_cast<size_t>(i) * L21.ld;
                for (uint32_t j = 0; j < k; ++j) li[j] = ri[piv[r0 + j]];
            }
            kernels::gemm<Elt>(ctx, cview(L21), DenseView<const Elt>(U12), A.sub(r0 + k + i0, rows, c1, w), GemmOp::Sub);
        };
        if (tiles == 1) tile(0);
        else (*par)(tiles, tile);
    }

    // Bisection inside a panel.
    void rec_panel(uint32_t c0, uint32_t c1) {
        if (r >= m || c0 >= c1) return;
        if (c1 - c0 <= prm.base) { base_panel(c0, c1); return; }
        const uint32_t cm = c0 + (c1 - c0) / 2;
        const uint32_t r_before = r;
        rec_panel(c0, cm);
        trailing_update(r_before, r - r_before, cm, c1);
        rec_panel(cm, c1);
    }

    void factor() {
        const PluClock::time_point t0 = tr ? PluClock::now() : PluClock::time_point();
        struct Acc { double& d; PluClock::time_point t0; bool on; ~Acc() { if (on) d += plu_secs(t0); } } acc{ph.total, t0, tr};
        for (uint32_t c0 = 0; c0 < n && r < m; c0 += prm.panel) {
            const uint32_t c1 = std::min<uint32_t>(c0 + prm.panel, n);
            const uint32_t r_before = r;
            rec_panel(c0, c1);
            trailing_update(r_before, r - r_before, c1, n);
        }
    }

    // Packed PLU -> RREF in place.
    void to_rref() {
        const PluClock::time_point t0 = tr ? PluClock::now() : PluClock::time_point();
        struct Acc { double& d0; double& d1; PluClock::time_point t0; bool on; ~Acc() { if (on) { const double s = plu_secs(t0); d0 += s; d1 += s; } } } acc{ph.to_rref, ph.total, t0, tr};
        const uint32_t k = r;
        if (k == 0) { zero_rows(A); return; }
        std::vector<uint32_t> np;
        np.reserve(n - k);
        {
            std::vector<char> is_piv(n, 0);
            for (uint32_t c : piv) is_piv[c] = 1;
            for (uint32_t c = 0; c < n; ++c) if (!is_piv[c]) np.push_back(c);
        }
        const uint32_t nnp = static_cast<uint32_t>(np.size());
        std::vector<Elt> d(k);
        for (uint32_t i = 0; i < k; ++i) d[i] = row(i)[piv[i]];
        ctx.batch_inv(d);   // pivots are non-zero by construction
        DenseBuf<Elt> X;
        if (nnp > 0) {
            // unit-upper U1' = D^{-1} U1 (strict upper only) and X = D^{-1} U[:, non-pivots]
            DenseBuf<Elt> Uk(k, k);
            for (uint32_t i = 0; i < k; ++i) {
                const Elt* ri = row(i);
                Elt* ui = Uk.data() + static_cast<size_t>(i) * Uk.ld;
                for (uint32_t j = i + 1; j < k; ++j) ui[j] = ctx.mul(ri[piv[j]], d[i]);
            }
            X = DenseBuf<Elt>(k, nnp);
            for (uint32_t i = 0; i < k; ++i) {
                const Elt* ri = row(i);
                Elt* xi = X.data() + static_cast<size_t>(i) * X.ld;
                for (uint32_t t = 0; t < nnp; ++t) xi[t] = ctx.mul(ri[np[t]], d[i]);
            }
            trsm_upper_unit_par<Elt>(ctx, cview(Uk), view(X));
        }
        for (uint32_t i = 0; i < k; ++i) {
            Elt* ri = row(i);
            for (uint32_t j = 0; j < n; ++j) ri[j] = 0;
            ri[piv[i]] = 1;
            if (nnp > 0) {
                const Elt* xi = X.data() + static_cast<size_t>(i) * X.ld;
                for (uint32_t t = 0; t < nnp; ++t) ri[np[t]] = xi[t];
            }
        }
        for (uint32_t i = k; i < m; ++i) {
            Elt* ri = row(i);
            for (uint32_t j = 0; j < n; ++j) ri[j] = 0;
        }
    }
};

std::atomic<DeviceRrefHook> g_device_rref_hook{nullptr};

// (slot-22 BIN-1) the native F_2 route, Sec. 8.1.  (S24 CL-5) ENGINE_BINARY_F2 / ENGINE_TERNARY_F3 on EnvLever: opt-in, exactly
// "1" = on, anything else = off, silently; read once; ScopedBinaryF2 / ScopedTernaryF3 override.
constexpr lever::Policy kOptInPolicy{false, lever::Empty::Unset, lever::Match::Exact01, lever::Cache::Once, nullptr};
lever::EnvLever g_binary_f2_lever("ENGINE_BINARY_F2", kOptInPolicy);
lever::EnvLever g_ternary_f3_lever("ENGINE_TERNARY_F3", kOptInPolicy);
std::atomic<uint64_t> g_binary_f2_rrefs{0}, g_binary_f2_profiles{0};

// A is row-contiguous or not; want_rref: write RREF(A) back (zero rows at the bottom), else leave
// A untouched (column_rank_profile: its contents are unspecified anyway).
template<class Elt>
bool binary_f2_route(const ModCtx<Elt>& ctx, DenseView<Elt> A, bool want_rref, RrefResult& res) {
    if (ctx.p != Elt(2) || !ctx.is_field() || !binary_f2_enabled()) return false;
    binary::BitMatrix B = binary::BitMatrix::from_dense<Elt>(DenseView<const Elt>(A));
    const binary::BitRref br = want_rref ? binary::rref(B) : binary::echelon(B);
    if (want_rref) B.to_dense<Elt>(A);
    res.rank = br.rank;
    res.pivots = br.pivots;
    (want_rref ? g_binary_f2_rrefs : g_binary_f2_profiles).fetch_add(1, std::memory_order_relaxed);
    return true;
}

// (slot-23 TER-1) the native F_3 route, Sec. 8.3 (the lever: above).
std::atomic<uint64_t> g_ternary_f3_rrefs{0}, g_ternary_f3_profiles{0};

// The twin of binary_f2_route at p == 3 (bit-sliced binary::TritMatrix, Gauss--Jordan).
template<class Elt>
bool ternary_f3_route(const ModCtx<Elt>& ctx, DenseView<Elt> A, bool want_rref, RrefResult& res) {
    if (ctx.p != Elt(3) || !ctx.is_field() || !ternary_f3_enabled()) return false;
    binary::TritMatrix T = binary::TritMatrix::from_dense<Elt>(DenseView<const Elt>(A));
    const binary::TritRref tr = want_rref ? binary::rref(T) : binary::echelon(T);
    if (want_rref) T.to_dense<Elt>(A);
    res.rank = tr.rank;
    res.pivots = tr.pivots;
    (want_rref ? g_ternary_f3_rrefs : g_ternary_f3_profiles).fetch_add(1, std::memory_order_relaxed);
    return true;
}

template<class Elt>
RrefResult rref_contiguous(const ModCtx<Elt>& ctx, DenseView<Elt> A) {
    RrefResult res;
    if (A.rows() == 0 || A.cols() == 0) return res;
    if constexpr (std::is_same<Elt, uint32_t>::value) {   // (slot-22 G2) the device route, Sec. 11.9
        const DeviceRrefHook h = g_device_rref_hook.load(std::memory_order_acquire);
        if (h != nullptr && h(ctx, A, true, res)) return res;
    }
    if (binary_f2_route<Elt>(ctx, A, true, res)) return res;   // (slot-22 BIN-1) Sec. 8.1, opt-in
    if (ternary_f3_route<Elt>(ctx, A, true, res)) return res;  // (slot-23 TER-1) Sec. 8.3, opt-in
    Plu<Elt> plu(ctx, A);
    plu.factor();
    plu.to_rref();
    res.rank = plu.r;
    res.pivots = std::move(plu.piv);
    return res;
}

template<class Elt>
void fill_from(DenseView<Elt> dst, DenseView<const Elt> src, uint32_t r0, uint32_t c0) {
    for (uint32_t i = 0; i < src.rows(); ++i)
        for (uint32_t j = 0; j < src.cols(); ++j) dst.at(r0 + i, c0 + j) = src.at(i, j);
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry points
//
// ENGINE_API.md Sec. 13.3: every entry point below that inverts a pivot or
// defines its output as an RREF is FIELD-ONLY and carries one O(1)
// ENGINE_CHECK on ctx.is_field() at its entry -- nothing else in its body
// changes, and the check is true on every call the field route makes (13.8).
// eliminate_against / sub_product / image_basis are modulus-only (13.2, 13.3).
// ---------------------------------------------------------------------------

void set_device_rref_hook(DeviceRrefHook h) { g_device_rref_hook.store(h, std::memory_order_release); }
DeviceRrefHook device_rref_hook() { return g_device_rref_hook.load(std::memory_order_acquire); }

const lever::EnvLever& binary_f2_lever() { return g_binary_f2_lever; }
const lever::EnvLever& ternary_f3_lever() { return g_ternary_f3_lever; }
bool binary_f2_enabled() { return g_binary_f2_lever.enabled(); }
bool ternary_f3_enabled() { return g_ternary_f3_lever.enabled(); }
TernaryF3Totals ternary_f3_totals() {
    TernaryF3Totals t;
    t.rrefs = g_ternary_f3_rrefs.load(std::memory_order_relaxed);
    t.rank_profiles = g_ternary_f3_profiles.load(std::memory_order_relaxed);
    return t;
}
BinaryF2Totals binary_f2_totals() {
    BinaryF2Totals t;
    t.rrefs = g_binary_f2_rrefs.load(std::memory_order_relaxed);
    t.rank_profiles = g_binary_f2_profiles.load(std::memory_order_relaxed);
    return t;
}

// (S24 CL-5, C-08) The six default-on module levers on EnvLever (DenseElim.h): exactly "0" = off, "1" = on, anything else
// ("" included: the copies tested e == nullptr only) warns once on stderr and stays on, reported as source=env; read once per
// process; a Scoped* override wins.  Each prints its EFFECTIVE line at exit when the variable is set (registered at load).
namespace {
constexpr lever::Policy module_policy(const char* warning) {
    return lever::Policy{true, lever::Empty::Invalid, lever::Match::Exact01, lever::Cache::Once, warning, "default", "env", "env"};
}
lever::EnvArmRetiredLever g_module_stream_lever(
    "HECKE_MODULE_STREAM", module_policy("WARNING HECKE_MODULE_STREAM=%s is neither 0 nor 1: the stream route stays on\n"));
lever::EnvArmRetiredLever g_module_eval_lever(
    "HECKE_MODULE_EVAL", module_policy("WARNING HECKE_MODULE_EVAL=%s is neither 0 nor 1: the fast evaluation stays on\n"));
lever::EnvArmRetiredLever g_module_parallel_lever(
    "HECKE_MODULE_PARALLEL", module_policy("WARNING HECKE_MODULE_PARALLEL=%s is neither 0 nor 1: the parallel assembly stays on\n"));
lever::EnvLever g_module_mt_lever("HECKE_MODULE_MT",
                                  module_policy("WARNING HECKE_MODULE_MT=%s is neither 0 nor 1: the module MT splits stay on\n"));
lever::EnvArmRetiredLever g_route2_shared_lever(
    "HECKE_ROUTE2_SHARED_MEMO", module_policy("WARNING HECKE_ROUTE2_SHARED_MEMO=%s is neither 0 nor 1: the shared memo stays on\n"));
lever::EnvLever g_lean_peak_lever(
    "HECKE_MODULE_LEAN_PEAK", module_policy("WARNING HECKE_MODULE_LEAN_PEAK=%s is neither 0 nor 1: the lean-peak route stays on\n"));
// ENGINE_HOST_ELIM_HOOK (read here as tape::host_elim_hook_enabled() reads it; DenseElim does not link the tape): exactly "0" = off
lever::EnvLever g_elim_hook_lever("ENGINE_HOST_ELIM_HOOK",
                                  lever::Policy{true, lever::Empty::Unset, lever::Match::Exact01, lever::Cache::Once, nullptr});
// "mode=<on|off> source=<scoped|env|default>[ (NAME=<v>)]": the common head of the effective texts
std::string lever_head(const lever::EnvLever& l, const char* on_text, const char* off_text) {
    std::string s = std::string("mode=") + (l.enabled() ? on_text : off_text) + " source=" + l.source();
    if (const char* v = l.raw()) s += std::string(" (") + l.name() + "=" + v + ")";
    return s;
}
void at_exit_if_set(const lever::EnvLever& l, void (*fn)()) {
    if (l.is_set()) std::atexit(fn);
}
} // namespace
const lever::EnvLever& module_stream_lever() { return g_module_stream_lever; }
const lever::EnvLever& module_eval_lever() { return g_module_eval_lever; }
const lever::EnvLever& module_parallel_lever() { return g_module_parallel_lever; }
const lever::EnvLever& module_mt_lever() { return g_module_mt_lever; }
const lever::EnvLever& route2_shared_memo_lever() { return g_route2_shared_lever; }
const lever::EnvLever& module_lean_peak_lever() { return g_lean_peak_lever; }

// (slot-23 G2) the module-stream lever -- DenseElim.h.
namespace {
std::atomic<int> g_readoff_corruption{0};   // validation only
std::atomic<int> g_lean_buffer_corruption{0};   // validation only (S23 G2-G audit m2)
std::atomic<uint64_t> g_ms_readoff{0}, g_ms_howell{0}, g_ms_legacy{0}, g_ms_dd_gemm{0}, g_ms_dd_scalar{0},
    g_ms_cyc_new{0}, g_ms_cyc_old{0};
std::atomic<uint64_t> g_ms_cyc_lean{0}, g_ms_cyc_lean_legacy{0}, g_ms_invf_lean{0}, g_ms_invf_legacy{0};   // (S23 G2-G)
std::atomic<uint64_t> g_ms_event[13] = {};   // (S23 G2-C/D/E, G2-F, G2-T) indexed by ModuleStreamEvent
void print_module_eval_effective_at_exit();
void note_module_eval_consulted() {
    static const bool registered = (at_exit_if_set(g_module_eval_lever, &print_module_eval_effective_at_exit), true);
    (void)registered;
}
// (G2-T audit F3) registered at load, not at the first consultation: a set HECKE_MODULE_EVAL that is never consulted
// (HECKE_MODULE_STREAM=0, or a run with no module-stream operator) still prints its line, with eval=fast:0,legacy:0.
[[maybe_unused]] const bool g_module_eval_noted_at_load = (note_module_eval_consulted(), true);
// (audit m2) The EFFECTIVE convention: when HECKE_MODULE_STREAM is set, ONE line per process, printed at
// exit with the final counters, registered the first time the lever is consulted ANYWHERE (Theta_eq, X_q,
// H_class_coordinates, d o d, cyclic_decomposition -- also a cohomology-only or Integral Subquotient run).
void print_module_stream_effective_at_exit() {
    std::fprintf(stdout, "EFFECTIVE HECKE_MODULE_STREAM %s\n", module_stream_effective_text().c_str());
    std::fflush(stdout);
}
void note_module_stream_consulted() {
    static const bool registered = (at_exit_if_set(g_module_stream_lever, &print_module_stream_effective_at_exit), true);
    (void)registered;
}
} // namespace
bool module_stream_enabled() {
    note_module_stream_consulted();
    return g_module_stream_lever.enabled();
}
bool module_eval_enabled() {
    note_module_eval_consulted();
    return g_module_eval_lever.enabled();
}
std::string module_eval_effective_text() {
    std::string s = lever_head(g_module_eval_lever, "fast", "legacy");
    s += " eval=fast:" + std::to_string(g_ms_event[11].load()) + ",legacy:" + std::to_string(g_ms_event[12].load());
    return s;
}
namespace {
void print_module_eval_effective_at_exit() {
    std::fprintf(stdout, "EFFECTIVE HECKE_MODULE_EVAL %s\n", module_eval_effective_text().c_str());
    std::fflush(stdout);
}
// (S23 G2-P) the module-parallel lever -- DenseElim.h.
std::atomic<uint64_t> g_mp_parallel{0}, g_mp_serial{0};
std::atomic<uint32_t> g_mp_threads{0};   // the largest pool size seen by compute_module_stream
std::atomic<uint64_t> g_mp_howell{0}, g_mp_readoff{0};   // hook regions of howell_rows / the product read-off
std::atomic<int> g_readoff_worker_fault{0};             // validation only (G2-P audit F1): 0 off, 1 residual, 2 throw
std::atomic<uint64_t> g_readoff_worker_faults{0};
// ENGINE_HOST_ELIM_HOOK=0: no elimination hook is installed, so howell_rows / the product read-off stay serial (G2-P audit F2).
bool elim_hook_env_off() { return !g_elim_hook_lever.env_enabled(); }
void print_module_parallel_effective_at_exit() {
    std::fprintf(stdout, "EFFECTIVE HECKE_MODULE_PARALLEL %s\n", module_parallel_effective_text().c_str());
    std::fflush(stdout);
}
// registered at load (the G2-T audit F3 rule): a set HECKE_MODULE_PARALLEL that is never consulted still prints
[[maybe_unused]] const bool g_module_parallel_noted_at_load = [] {
    if (g_module_parallel_lever.is_set() || elim_hook_env_off()) std::atexit(&print_module_parallel_effective_at_exit);
    return true;
}();
} // namespace
bool module_parallel_enabled() { return g_module_parallel_lever.enabled(); }
void module_parallel_count(bool parallel, uint64_t degrees, uint32_t threads) {
    (parallel ? g_mp_parallel : g_mp_serial).fetch_add(degrees);
    uint32_t t = g_mp_threads.load();
    while (threads > t && !g_mp_threads.compare_exchange_weak(t, threads)) {}
}
uint64_t module_parallel_degrees(bool parallel) { return (parallel ? g_mp_parallel : g_mp_serial).load(); }
uint64_t module_parallel_howell_regions() { return g_mp_howell.load(); }
uint64_t module_parallel_readoff_regions() { return g_mp_readoff.load(); }
ScopedReadoffWorkerFault::ScopedReadoffWorkerFault(int mode) : saved_(g_readoff_worker_fault.exchange(mode)) {}
ScopedReadoffWorkerFault::~ScopedReadoffWorkerFault() { g_readoff_worker_fault.store(saved_); }
uint64_t readoff_worker_faults() { return g_readoff_worker_faults.load(); }
std::string module_parallel_effective_text() {
    std::string s = lever_head(g_module_parallel_lever, "on", "off");
    s += " threads=" + std::to_string(g_mp_threads.load());
    s += " par=parallel:" + std::to_string(g_mp_parallel.load()) + ",serial:" + std::to_string(g_mp_serial.load());
    s += " hook=howell:" + std::to_string(g_mp_howell.load()) + ",readoff:" + std::to_string(g_mp_readoff.load());
    s += elim_hook_env_off() ? " elim_hook=off howell_readoff=serial" : " elim_hook=on";   // (G2-P audit F2)
    return s;
}
// (S24 MT-1) the module-MT lever -- DenseElim.h.
namespace {
std::atomic<uint64_t> g_mt_event[6] = {};
void print_module_mt_effective_at_exit() {
    std::fprintf(stdout, "EFFECTIVE HECKE_MODULE_MT %s\n", module_mt_effective_text().c_str());
    std::fflush(stdout);
}
// registered at load (the G2-T audit F3 rule): a set HECKE_MODULE_MT that is never consulted still prints its line
[[maybe_unused]] const bool g_module_mt_noted_at_load = (at_exit_if_set(g_module_mt_lever, &print_module_mt_effective_at_exit), true);
} // namespace
bool module_mt_enabled() { return g_module_mt_lever.enabled(); }
const ParallelFor* module_mt_hook() {
    return (module_mt_enabled() && module_parallel_enabled()) ? parallel_hook() : nullptr;
}
size_t module_tile_work() { return mod_tile_work(); }
uint64_t module_tile_work_capped() { return g_mod_tiles_capped.load(); }
uint64_t module_tile_work_serial() { return g_mod_tiles_serial.load(); }
ScopedModuleTileWork::ScopedModuleTileWork(size_t work) : saved_(g_mod_tile_work_scoped.exchange(static_cast<long long>(work))) {}
ScopedModuleTileWork::~ScopedModuleTileWork() { g_mod_tile_work_scoped.store(saved_); }
ScopedModuleMtForceSplit::ScopedModuleMtForceSplit() : saved_(g_mt_force_split.exchange(1)) {}
ScopedModuleMtForceSplit::~ScopedModuleMtForceSplit() { g_mt_force_split.store(saved_); }
void module_mt_count(ModuleMtEvent e, uint64_t k) { g_mt_event[static_cast<uint32_t>(e)].fetch_add(k, std::memory_order_relaxed); }
uint64_t module_mt_regions(ModuleMtEvent e) { return g_mt_event[static_cast<uint32_t>(e)].load(); }
std::string module_mt_effective_text() {
    std::string s = lever_head(g_module_mt_lever, "on", "off");
    s += " regions=cyc_search:" + std::to_string(module_mt_regions(ModuleMtEvent::CycSearch)) +
         ",cyc_rows:" + std::to_string(module_mt_regions(ModuleMtEvent::CycRows)) +
         ",gemm:" + std::to_string(module_mt_regions(ModuleMtEvent::Gemm)) +
         ",check:" + std::to_string(module_mt_regions(ModuleMtEvent::Check)) +
         ",assembly:" + std::to_string(module_mt_regions(ModuleMtEvent::Assembly)) +
         " pivot_exit=" + std::to_string(module_mt_regions(ModuleMtEvent::PivotExit));
    // (MT-1 audit F2) part (0) needs only the lever and the lean route; the splits also need HECKE_MODULE_PARALLEL
    if (!module_stream_enabled()) s += " (HECKE_MODULE_STREAM off: inert)";
    else if (!module_parallel_enabled()) s += " (HECKE_MODULE_PARALLEL off: splits inert, pivot-search exit on)";
    return s;
}
// (S23 G2-T2) the route-2 memo lever, HECKE_ROUTE2_SHARED_MEMO (DenseElim.h).
namespace {
std::atomic<uint64_t> g_route2_shared_deg{0}, g_route2_chunked_deg{0};
void print_route2_shared_effective_at_exit() {
    std::fprintf(stdout, "EFFECTIVE HECKE_ROUTE2_SHARED_MEMO %s\n", route2_shared_memo_effective_text().c_str());
    std::fflush(stdout);
}
// registered at load (as HECKE_MODULE_EVAL's, G2-T audit F3): a set HECKE_ROUTE2_SHARED_MEMO always prints its line
[[maybe_unused]] const bool g_route2_shared_noted_at_load =
    (at_exit_if_set(g_route2_shared_lever, &print_route2_shared_effective_at_exit), true);
} // namespace
bool route2_shared_memo_enabled() { return g_route2_shared_lever.enabled(); }
void route2_memo_count(bool shared) { (shared ? g_route2_shared_deg : g_route2_chunked_deg).fetch_add(1, std::memory_order_relaxed); }
Route2MemoCounts route2_memo_counts() {
    Route2MemoCounts c;
    c.shared = g_route2_shared_deg.load();
    c.chunked = g_route2_chunked_deg.load();
    return c;
}
std::string route2_shared_memo_effective_text() {
    std::string s = lever_head(g_route2_shared_lever, "shared", "chunked");
    const Route2MemoCounts c = route2_memo_counts();
    s += " route2_degrees=shared:" + std::to_string(c.shared) + ",chunked:" + std::to_string(c.chunked);
    return s;
}
// (S24 G2-H) the lean-peak lever, HECKE_MODULE_LEAN_PEAK (DenseElim.h).
namespace {
std::atomic<uint64_t> g_lean_peak_event[6] = {};   // indexed by LeanPeakEvent
void print_lean_peak_effective_at_exit() {
    std::fprintf(stdout, "EFFECTIVE HECKE_MODULE_LEAN_PEAK %s\n", module_lean_peak_effective_text().c_str());
    std::fflush(stdout);
}
// registered at load (as HECKE_ROUTE2_SHARED_MEMO's): a set HECKE_MODULE_LEAN_PEAK always prints its line
[[maybe_unused]] const bool g_lean_peak_noted_at_load = (at_exit_if_set(g_lean_peak_lever, &print_lean_peak_effective_at_exit), true);
} // namespace
bool module_lean_peak_enabled() { return g_lean_peak_lever.enabled(); }
void lean_peak_count(LeanPeakEvent e, uint64_t k) {
    if (k != 0) g_lean_peak_event[static_cast<uint32_t>(e)].fetch_add(k, std::memory_order_relaxed);
}
uint64_t lean_peak_events(LeanPeakEvent e) { return g_lean_peak_event[static_cast<uint32_t>(e)].load(); }
std::string module_lean_peak_effective_text() {
    std::string s = lever_head(g_lean_peak_lever, "on", "off");
    s += " howell=adopt:" + std::to_string(lean_peak_events(LeanPeakEvent::HowellAdopt)) +
         ",copy:" + std::to_string(lean_peak_events(LeanPeakEvent::HowellCopy)) +
         " residual=" + std::to_string(lean_peak_events(LeanPeakEvent::Residual)) +
         " route2_inplace=" + std::to_string(lean_peak_events(LeanPeakEvent::Route2InPlace)) +
         " early_release=" + std::to_string(lean_peak_events(LeanPeakEvent::EarlyRelease)) +
         " length_only=" + std::to_string(lean_peak_events(LeanPeakEvent::LengthOnly));
    if (!module_stream_enabled()) s += " (HECKE_MODULE_STREAM off: inert)";
    return s;
}
const char* module_stream_env_value() { return g_module_stream_lever.raw(); }
ScopedReadoffCorruption::ScopedReadoffCorruption(bool on) : saved_(g_readoff_corruption.exchange(on ? 1 : 0)) {}
ScopedReadoffCorruption::~ScopedReadoffCorruption() { g_readoff_corruption.store(saved_); }
ScopedLeanBufferCorruption::ScopedLeanBufferCorruption(bool on) : saved_(g_lean_buffer_corruption.exchange(on ? 1 : 0)) {}
ScopedLeanBufferCorruption::~ScopedLeanBufferCorruption() { g_lean_buffer_corruption.store(saved_); }
ModuleStreamTotals module_stream_totals() {
    ModuleStreamTotals t;
    t.readoff = g_ms_readoff.load(); t.howell = g_ms_howell.load(); t.legacy = g_ms_legacy.load();
    t.dd_gemm = g_ms_dd_gemm.load(); t.dd_scalar = g_ms_dd_scalar.load();
    t.cyc_interchanged = g_ms_cyc_new.load(); t.cyc_legacy = g_ms_cyc_old.load();
    t.omega_streamed = g_ms_event[0].load(); t.omega_table = g_ms_event[1].load();
    t.lr_grouped = g_ms_event[2].load(); t.lr_scalar = g_ms_event[3].load();
    t.route2_blocked = g_ms_event[4].load(); t.route2_full = g_ms_event[5].load();
    t.w_nonpivot = g_ms_event[6].load(); t.lean_released = g_ms_event[7].load();
    t.lr_early_flush = g_ms_event[8].load();
    t.tau_chains = g_ms_event[9].load(); t.tau_pooled = g_ms_event[10].load();
    t.eval_fast = g_ms_event[11].load(); t.eval_legacy = g_ms_event[12].load();
    t.cyc_lean = g_ms_cyc_lean.load(); t.cyc_lean_legacy = g_ms_cyc_lean_legacy.load();
    t.invf_lean = g_ms_invf_lean.load(); t.invf_legacy = g_ms_invf_legacy.load();
    return t;
}
void module_stream_count_solve(bool stream_on, ReadoffStatus st, bool readoff_tried) {
    if (!stream_on) { g_ms_legacy.fetch_add(1, std::memory_order_relaxed); return; }
    if (readoff_tried && st == ReadoffStatus::Ok) g_ms_readoff.fetch_add(1, std::memory_order_relaxed);
    else g_ms_howell.fetch_add(1, std::memory_order_relaxed);
}
void module_stream_count(ModuleStreamEvent e, uint64_t n) {
    if (n != 0) g_ms_event[static_cast<int>(e)].fetch_add(n, std::memory_order_relaxed);
}
void module_stream_count_dd(bool gemm) { (gemm ? g_ms_dd_gemm : g_ms_dd_scalar).fetch_add(1, std::memory_order_relaxed); }
std::string module_stream_effective_text() {
    const ModuleStreamTotals t = module_stream_totals();
    note_module_stream_consulted();   // as module_stream_enabled() did here before
    std::string s = lever_head(g_module_stream_lever, "stream", "legacy");
    s += " solve=readoff:" + std::to_string(t.readoff) + ",howell:" + std::to_string(t.howell) +
         ",legacy:" + std::to_string(t.legacy) + " dd=gemm:" + std::to_string(t.dd_gemm) +
         ",scalar:" + std::to_string(t.dd_scalar) + " cyc=interchanged:" + std::to_string(t.cyc_interchanged) +
         ",legacy:" + std::to_string(t.cyc_legacy) + " omega=streamed:" + std::to_string(t.omega_streamed) +
         ",table:" + std::to_string(t.omega_table) + " lr=grouped:" + std::to_string(t.lr_grouped) +
         ",scalar:" + std::to_string(t.lr_scalar) +
         ",early_flush:" + std::to_string(t.lr_early_flush) + " route2=blocked:" + std::to_string(t.route2_blocked) +
         ",full:" + std::to_string(t.route2_full) + " w=nonpivot:" + std::to_string(t.w_nonpivot) +
         " retention=lean:" + std::to_string(t.lean_released) +
         " cyc_buffers=lean:" + std::to_string(t.cyc_lean) + ",legacy:" + std::to_string(t.cyc_lean_legacy) +
         " invf=lean:" + std::to_string(t.invf_lean) + ",legacy:" + std::to_string(t.invf_legacy) +
         " tau=chains:" + std::to_string(t.tau_chains) + ",pooled:" + std::to_string(t.tau_pooled) +
         " eval=fast:" + std::to_string(t.eval_fast) + ",legacy:" + std::to_string(t.eval_legacy);
    return s;
}

template<class Elt>
RrefResult rref(const ModCtx<Elt>& ctx, DenseView<Elt> A) {
    ENGINE_CHECK_MSG(ctx.is_field(), "elim::rref: field-only; use the Z/ell^k entry points of ENGINE_API.md Sec. 13");
    ENGINE_CHECK_MSG(A.invariant() == Invariant::Canonical, "elim::rref: operand must be Canonical");
    if (A.row_contiguous()) return rref_contiguous<Elt>(ctx, A);
    DenseBuf<Elt> tmp(A.rows(), A.cols());
    kernels::copy<Elt>(DenseView<const Elt>(A), view(tmp));
    RrefResult res = rref_contiguous<Elt>(ctx, view(tmp));
    kernels::copy<Elt>(cview(tmp), A);
    return res;
}

template<class Elt>
RrefResult column_rank_profile(const ModCtx<Elt>& ctx, DenseView<Elt> A) {
    ENGINE_CHECK_MSG(ctx.is_field(), "elim::column_rank_profile: field-only; use the Z/ell^k entry points of ENGINE_API.md Sec. 13");
    ENGINE_CHECK_MSG(A.invariant() == Invariant::Canonical, "elim::column_rank_profile: operand must be Canonical");
    ENGINE_CHECK_MSG(A.row_contiguous(), "elim::column_rank_profile: operand must be row-contiguous");
    RrefResult res;
    if (A.rows() == 0 || A.cols() == 0) return res;
    if constexpr (std::is_same<Elt, uint32_t>::value) {   // (slot-22 G2) the device route; A is left untouched (its contents are unspecified)
        const DeviceRrefHook h = g_device_rref_hook.load(std::memory_order_acquire);
        if (h != nullptr && h(ctx, A, false, res)) return res;
    }
    if (binary_f2_route<Elt>(ctx, A, false, res)) return res;   // (slot-22 BIN-1) Sec. 8.1, opt-in
    if (ternary_f3_route<Elt>(ctx, A, false, res)) return res;  // (slot-23 TER-1) Sec. 8.3, opt-in
    Plu<Elt> plu(ctx, A);
    plu.factor();
    res.rank = plu.r;
    res.pivots = std::move(plu.piv);
    return res;
}

template<class Elt>
RrefResult rref_with_transform(const ModCtx<Elt>& ctx, DenseView<const Elt> A,
                               DenseView<Elt> R, DenseView<Elt> U) {
    ENGINE_CHECK_MSG(ctx.is_field(), "elim::rref_with_transform: field-only; use the Z/ell^k entry points of ENGINE_API.md Sec. 13");
    const uint32_t m = A.rows(), n = A.cols();
    ENGINE_CHECK(R.rows() == m && R.cols() == n && U.rows() == m && U.cols() == m);
    DenseBuf<Elt> aug(m, n + m);
    fill_from<Elt>(view(aug), A, 0, 0);
    for (uint32_t i = 0; i < m; ++i) aug.at(i, n + i) = 1;
    RrefResult full = rref_contiguous<Elt>(ctx, view(aug));
    kernels::copy<Elt>(cview(aug).sub(0, m, 0, n), R);
    kernels::copy<Elt>(cview(aug).sub(0, m, n, m), U);
    RrefResult res;
    for (uint32_t c : full.pivots) if (c < n) res.pivots.push_back(c);
    res.rank = static_cast<uint32_t>(res.pivots.size());
    return res;
}

template<class Elt>
DenseBuf<Elt> kernel_basis(const ModCtx<Elt>& ctx, DenseView<const Elt> R,
                           const std::vector<uint32_t>& pivots) {
    ENGINE_CHECK_MSG(ctx.is_field(), "elim::kernel_basis: field-only; use the Z/ell^k entry points of ENGINE_API.md Sec. 13");
    const uint32_t n = R.cols();
    const uint32_t k = static_cast<uint32_t>(pivots.size());
    ENGINE_CHECK(k <= n && k <= R.rows());
    std::vector<char> is_piv(n, 0);
    for (uint32_t c : pivots) { ENGINE_CHECK(c < n); is_piv[c] = 1; }
    DenseBuf<Elt> K(n - k, n);
    uint32_t t = 0;
    for (uint32_t jf = 0; jf < n; ++jf) {
        if (is_piv[jf]) continue;
        Elt* kt = K.data() + static_cast<size_t>(t) * K.ld;
        kt[jf] = 1;
        for (uint32_t i = 0; i < k; ++i) kt[pivots[i]] = ctx.neg(R.at(i, jf));
        ++t;
    }
    return K;
}

template<class Elt>
DenseBuf<Elt> image_basis(DenseView<const Elt> A, const std::vector<uint32_t>& pivots) {
    const uint32_t m = A.rows();
    DenseBuf<Elt> I(static_cast<uint32_t>(pivots.size()), m);
    for (uint32_t t = 0; t < pivots.size(); ++t) {
        ENGINE_CHECK(pivots[t] < A.cols());
        Elt* it = I.data() + static_cast<size_t>(t) * I.ld;
        for (uint32_t i = 0; i < m; ++i) it[i] = A.at(i, pivots[t]);
    }
    return I;
}

template<class Elt>
bool inverse(const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<Elt> Ainv) {
    ENGINE_CHECK_MSG(ctx.is_field(), "elim::inverse: field-only; use the Z/ell^k entry points of ENGINE_API.md Sec. 13");
    const uint32_t n = A.rows();
    if (n == 0 || A.cols() != n) return false;
    ENGINE_CHECK(Ainv.rows() == n && Ainv.cols() == n);
    DenseBuf<Elt> aug(n, 2 * n);
    fill_from<Elt>(view(aug), A, 0, 0);
    for (uint32_t i = 0; i < n; ++i) aug.at(i, n + i) = 1;
    const RrefResult res = rref_contiguous<Elt>(ctx, view(aug));
    if (res.rank < n || res.pivots[n - 1] != n - 1) return false;   // pivots == 0..n-1
    kernels::copy<Elt>(cview(aug).sub(0, n, n, n), Ainv);
    return true;
}

template<class Elt>
std::vector<std::vector<Elt>> solve_multi_rhs(const ModCtx<Elt>& ctx, DenseView<const Elt> B,
                                              DenseView<const Elt> Y) {
    ENGINE_CHECK_MSG(ctx.is_field(), "elim::solve_multi_rhs: field-only; use the Z/ell^k entry points of ENGINE_API.md Sec. 13");
    const uint32_t a = B.rows(), nb = B.cols(), ny = Y.cols();
    ENGINE_CHECK(Y.rows() == a);
    std::vector<std::vector<Elt>> result(ny, std::vector<Elt>(nb, Elt(0)));
    if (ny == 0 || a == 0) return result;
    DenseBuf<Elt> aug(a, nb + ny);
    fill_from<Elt>(view(aug), B, 0, 0);
    fill_from<Elt>(view(aug), Y, 0, nb);
    const RrefResult res = rref_contiguous<Elt>(ctx, view(aug));
    for (uint32_t i = 0; i < res.rank; ++i) {
        const uint32_t c = res.pivots[i];
        if (c >= nb) {
            throw std::runtime_error("Vector is not in image of basis (right-hand side " +
                                     std::to_string(c - nb) + ").");
        }
        for (uint32_t t = 0; t < ny; ++t) result[t][c] = aug.at(i, nb + t);
    }
    return result;
}

template<class Elt>
std::vector<uint32_t> complement_basis(const ModCtx<Elt>& ctx, DenseView<const Elt> Rl,
                                       DenseView<const Elt> G, DenseBuf<Elt>* projector) {
    ENGINE_CHECK_MSG(ctx.is_field(), "elim::complement_basis: field-only; use the Z/ell^k entry points of ENGINE_API.md Sec. 13");
    const uint32_t a = G.rows(), nr = Rl.cols(), ng = G.cols();
    ENGINE_CHECK(Rl.rows() == a);
    DenseBuf<Elt> aug(a, nr + ng + a);
    fill_from<Elt>(view(aug), Rl, 0, 0);
    fill_from<Elt>(view(aug), G, 0, nr);
    for (uint32_t i = 0; i < a; ++i) aug.at(i, nr + ng + i) = 1;
    const RrefResult res = rref_contiguous<Elt>(ctx, view(aug));
    uint32_t independent_relations = 0;
    std::vector<uint32_t> gens;
    for (uint32_t c : res.pivots) {
        if (c < nr) ++independent_relations;
        else if (c < nr + ng) gens.push_back(c - nr);
    }
    if (projector != nullptr) {
        const uint32_t dim_h = static_cast<uint32_t>(gens.size());
        *projector = DenseBuf<Elt>(dim_h, a);
        for (uint32_t i = 0; i < dim_h; ++i)
            for (uint32_t j = 0; j < a; ++j)
                projector->at(i, j) = aug.at(independent_relations + i, nr + ng + j);
    }
    return gens;
}

template<class Elt>
DenseBuf<Elt> eliminate_against(const ModCtx<Elt>& ctx, DenseView<Elt> B, DenseView<const Elt> R,
                                const std::vector<uint32_t>& pivots, uint32_t l_cols) {
    const uint32_t m = B.rows(), n = B.cols();
    const uint32_t k = static_cast<uint32_t>(pivots.size());
    ENGINE_CHECK(R.cols() == n && k <= R.rows() && k <= l_cols);
    DenseBuf<Elt> L(m, l_cols);
    for (uint32_t i = 0; i < m; ++i) {
        Elt* li = L.data() + static_cast<size_t>(i) * L.ld;
        for (uint32_t j = 0; j < k; ++j) li[j] = B.at(i, pivots[j]);
    }
    if (k > 0 && m > 0 && n > 0) {
        kernels::gemm<Elt>(ctx, cview(L).sub(0, m, 0, k), R.sub(0, k, 0, n), B, GemmOp::Sub);
    }
    return L;
}

template<class Elt>
void sub_product(const ModCtx<Elt>& ctx, DenseView<Elt> C, DenseView<const Elt> L,
                 DenseView<const Elt> R) {
    if (C.rows() == 0 || C.cols() == 0 || L.cols() == 0) return;
    kernels::gemm<Elt>(ctx, L, R, C, GemmOp::Sub);
}

// ===========================================================================
// Z/ell^k: the Howell form and its derived entry points (ENGINE_API.md
// Sec. 13.5 / 13.6).  Every entry point ENGINE_CHECKs !ctx.is_field(); the
// field route above never reaches this code (Sec. 13.8).
//
// Arithmetic: all entries stay canonical (< ell^k).  valuation / unit_part /
// divide_exact and the reduce-above quotient are integer operations on Elt;
// every product goes through ctx.mul (Sec. 13.5 "Overflow").  A row update is
// an AXPY, which is modulus-only (Sec. 13.2).
//
// (slot-11 WP-12, F-1) A factor applied to a whole row or column -- the AXPY
// multiplier q, the scaling unit's inverse, the closure factor ell^{k-v} --
// is prepared ONCE with ctx.mul_prep and applied with ctx.mul_prepped, which
// equals ctx.mul bit for bit (ModCtx.h): one Montgomery REDC per product in
// the U64_Montgomery regime instead of a __umodti3, and literally ctx.mul in
// every other regime (all of u32, U64_Wide).  Canonical in, canonical out:
// every stored entry, every branch and every result is unchanged by
// construction (ModCtxMontgomeryPairValidation checks it entrywise).
// ===========================================================================
#define ENGINE_COMPOSITE_ONLY(name) \
    name ": composite-only (Z/ell^k); a field context uses the Sec. 8 entry points (ENGINE_API.md Sec. 13.6)"

namespace {

// row[c0..n) -= q * src[c0..n)   (exact, canonical)
template<class Elt>
inline void row_sub_scaled(const ModCtx<Elt>& ctx, Elt* row, const Elt* src, Elt q, uint32_t c0, uint32_t n) {
    if (q == 0) return;
    const Elt qp = ctx.mul_prep(q);
    for (uint32_t j = c0; j < n; ++j)
        if (src[j] != 0) row[j] = ctx.sub(row[j], ctx.mul_prepped(src[j], qp));
}

// The Sec. 13.5 algorithm on an n-column row-major row store holding the m
// rows of A; the store grows by the closure rows.
// (S24 G2-H) pivots_only: return after the pivot loop with H = 0 x n -- the pivots and their valuations are final
// there (the reduce-above below changes no pivot and no valuation), which is all length() reads.
template<class Elt>
HowellResult<Elt> howell_rows(const ModCtx<Elt>& ctx, std::vector<Elt> store, uint32_t m, uint32_t n,
                              bool pivots_only = false) {
    HowellResult<Elt> out;
    // The ordered ACTIVE list: initially the rows of A, top to bottom.
    std::vector<uint32_t> active(m);
    for (uint32_t i = 0; i < m; ++i) active[i] = i;
    uint32_t n_rows = m;
    std::vector<uint32_t> piv_rows;               // store index of each pivot row, in column order
    // (S23 G2-P) HECKE_MODULE_PARALLEL with an elimination hook installed (compute_cohomology_module /
    // compute_module_stream install one when the consumer pool has > 1 worker): steps 3 and the reduce-above
    // split their DISJOINT target rows into contiguous tiles over the hook.  Every target row receives the same
    // single update from the same (unchanged, read-only) source row, so no tiling can move a bit (run_hook).
    const ParallelFor* hpar = module_parallel_enabled() ? parallel_hook() : nullptr;
    for (uint32_t c = 0; c < n && !active.empty(); ++c) {
        // 1. Minimal valuation in column c; ties to the TOPMOST active row.
        size_t best = active.size();
        uint32_t best_v = ctx.k;
        for (size_t t = 0; t < active.size(); ++t) {
            const Elt e = store[static_cast<size_t>(active[t]) * n + c];
            if (e == 0) continue;
            const uint32_t v = ctx.valuation(e);
            if (v < best_v) { best_v = v; best = t; if (v == 0) break; }
        }
        if (best == active.size()) continue;       // I_c = 0: not a pivot column
        const uint32_t prow = active[best];
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(best));
        // 2. Scale so that the row leads with exactly ell^v.
        {
            Elt* P = store.data() + static_cast<size_t>(prow) * n;
            const Elt u = ctx.unit_part(P[c]);
            if (u != 1) {
                const Elt uip = ctx.mul_prep(ctx.inv(u));
                for (uint32_t j = c; j < n; ++j) P[j] = ctx.mul_prepped(P[j], uip);
            }
        }
        const Elt lead = ctx.ell_pow(best_v);
        // 3. Eliminate every other active row exactly (valuation >= v there).
        const uint32_t tiles3 = mod_tiles(hpar, static_cast<uint32_t>(active.size()),
                                           active.size() * static_cast<size_t>(n - c), 64, size_t(1) << 17, 16);
        if (tiles3 == 0) {
            for (uint32_t r : active) {
                Elt* R = store.data() + static_cast<size_t>(r) * n;
                if (R[c] == 0) continue;
                const Elt q = ctx.divide_exact(R[c], lead);
                row_sub_scaled(ctx, R, store.data() + static_cast<size_t>(prow) * n, q, c, n);
                ENGINE_DEBUG_CHECK(R[c] == 0);
            }
        } else {
            const size_t na = active.size();
            g_mp_howell.fetch_add(1, std::memory_order_relaxed);
            run_hook(hpar, tiles3, [&](uint32_t t) {
                const size_t a0 = na * t / tiles3, a1 = na * (t + 1) / tiles3;
                for (size_t x = a0; x < a1; ++x) {
                    Elt* R = store.data() + static_cast<size_t>(active[x]) * n;
                    if (R[c] == 0) continue;
                    const Elt q = ctx.divide_exact(R[c], lead);
                    row_sub_scaled(ctx, R, store.data() + static_cast<size_t>(prow) * n, q, c, n);
                    ENGINE_DEBUG_CHECK(R[c] == 0);
                }
            });
        }
        // 4. Howell closure: v > 0 appends ell^{k-v} * (pivot row), zero in
        //    column c, at the BOTTOM of the active list.
        if (best_v > 0) {
            const Elt fp = ctx.mul_prep(ctx.ell_pow(ctx.k - best_v));
            store.resize(store.size() + n);
            const Elt* P = store.data() + static_cast<size_t>(prow) * n;
            Elt* Cl = store.data() + static_cast<size_t>(n_rows) * n;
            for (uint32_t j = 0; j < n; ++j) Cl[j] = (j <= c) ? Elt(0) : ctx.mul_prepped(P[j], fp);
            active.push_back(n_rows++);
        }
        piv_rows.push_back(prow);
        out.pivots.push_back(c);
        out.pivot_v.push_back(best_v);
    }
    // Reduce above, pivot columns left to right: H[i'][j_i] = q*ell^{v_i} + r,
    // r in [0, ell^{v_i}) (integer division); subtract q * row i.  This sets
    // column j_i of row i' to r and changes only its columns > j_i besides
    // (review m2), which later pivots reduce in turn.
    if (pivots_only) {
        out.H = DenseBuf<Elt>(0, n);
        return out;
    }
    const uint32_t h = static_cast<uint32_t>(piv_rows.size());
    // (S24 G2-H) HECKE_MODULE_LEAN_PEAK (a): no closure row was appended (n_rows == m) and every store row is a pivot
    // row (h == m), so piv_rows is a permutation of the store's rows: permute them in place (cycle by cycle, one row
    // of scratch) and adopt the store as H -- row i of H is the old row piv_rows[i], as in the copy below.
    bool adopted = false;
    if (module_lean_peak_enabled() && module_stream_enabled()) {
        if (h > 0 && n > 0 && n_rows == m && h == m && store.size() == static_cast<size_t>(h) * n) {
            std::vector<uint8_t> done(h, 0);
            std::vector<Elt> tmp(n);
            for (uint32_t i = 0; i < h; ++i) {
                if (done[i]) continue;
                done[i] = 1;
                if (piv_rows[i] == i) continue;
                Elt* base = store.data();
                std::copy(base + static_cast<size_t>(i) * n, base + static_cast<size_t>(i + 1) * n, tmp.begin());
                uint32_t j = i;
                for (;;) {                                  // row j <- old row piv_rows[j], along the cycle of i
                    const uint32_t k = piv_rows[j];
                    if (k == i) {
                        std::copy(tmp.begin(), tmp.end(), base + static_cast<size_t>(j) * n);
                        break;
                    }
                    std::copy(base + static_cast<size_t>(k) * n, base + static_cast<size_t>(k + 1) * n,
                              base + static_cast<size_t>(j) * n);
                    done[k] = 1;
                    j = k;
                }
            }
            out.H = DenseBuf<Elt>::adopt(std::move(store), h, n);
            adopted = true;
            lean_peak_count(LeanPeakEvent::HowellAdopt);
        } else {
            lean_peak_count(LeanPeakEvent::HowellCopy);
        }
    }
    if (!adopted) {
    out.H = DenseBuf<Elt>(h, n);
    for (uint32_t i = 0; i < h; ++i) {
        const Elt* src = store.data() + static_cast<size_t>(piv_rows[i]) * n;
        std::copy(src, src + n, out.H.data() + static_cast<size_t>(i) * out.H.ld);
    }
    }   // (S24 G2-H) !adopted
    for (uint32_t i = 0; i < h; ++i) {
        const uint32_t ji = out.pivots[i];
        const Elt d = ctx.ell_pow(out.pivot_v[i]);
        const Elt* Ri = out.H.data() + static_cast<size_t>(i) * out.H.ld;
        const uint32_t tilesr = mod_tiles(hpar, i, static_cast<size_t>(i) * (n - ji), 64, size_t(1) << 17, 16);
        if (tilesr == 0) {
            for (uint32_t ip = 0; ip < i; ++ip) {
                Elt* Rp = out.H.data() + static_cast<size_t>(ip) * out.H.ld;
                const Elt q = Rp[ji] / d;
                row_sub_scaled(ctx, Rp, Ri, q, ji, n);
            }
        } else {   // (S23 G2-P) rows ip < i are disjoint targets of the read-only row i
            g_mp_howell.fetch_add(1, std::memory_order_relaxed);
            run_hook(hpar, tilesr, [&](uint32_t t) {
                const uint32_t p0 = static_cast<uint32_t>(static_cast<uint64_t>(i) * t / tilesr);
                const uint32_t p1 = static_cast<uint32_t>(static_cast<uint64_t>(i) * (t + 1) / tilesr);
                for (uint32_t ip = p0; ip < p1; ++ip) {
                    Elt* Rp = out.H.data() + static_cast<size_t>(ip) * out.H.ld;
                    const Elt q = Rp[ji] / d;
                    row_sub_scaled(ctx, Rp, Ri, q, ji, n);
                }
            });
        }
    }
    return out;
}

template<class Elt>
std::vector<Elt> rows_of(DenseView<const Elt> A) {
    std::vector<Elt> s(static_cast<size_t>(A.rows()) * A.cols());
    for (uint32_t i = 0; i < A.rows(); ++i)
        for (uint32_t j = 0; j < A.cols(); ++j) s[static_cast<size_t>(i) * A.cols() + j] = A.at(i, j);
    return s;
}

// [A^T | I_n]  (n x (m + n)), row-major store.
template<class Elt>
std::vector<Elt> transpose_augmented(DenseView<const Elt> A) {
    const uint32_t m = A.rows(), n = A.cols(), w = m + n;
    std::vector<Elt> s(static_cast<size_t>(n) * w, Elt(0));
    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t i = 0; i < m; ++i) s[static_cast<size_t>(j) * w + i] = A.at(i, j);
        s[static_cast<size_t>(j) * w + m + j] = 1;
    }
    return s;
}

// (S23 G2-G) kernel_module's step 2: the rows of a Howell form of [A^T | I_n] (A m x n) whose leading column
// is >= m, restricted to their last n columns.  Shared by kernel_module and the lean cyclic_decomposition /
// invariant_factors, which take ONE Howell form of [Gamma^T | I] for both the kernel and the coordinates.
template<class Elt>
HowellResult<Elt> kernel_from_augmented(const HowellResult<Elt>& full, uint32_t m, uint32_t n) {
    HowellResult<Elt> out;
    uint32_t first = 0;
    while (first < full.rows() && full.pivots[first] < m) ++first;
    const uint32_t h = full.rows() - first;
    out.H = DenseBuf<Elt>(h, n);
    for (uint32_t i = 0; i < h; ++i) {
        for (uint32_t j = 0; j < n; ++j) out.H.at(i, j) = full.H.at(first + i, m + j);
        out.pivots.push_back(full.pivots[first + i] - m);
        out.pivot_v.push_back(full.pivot_v[first + i]);
    }
    return out;
}

// (S24 MT-1) one right-hand side t of solve_from_augmented below: its serial loop body, verbatim (u is a work vector
// of length m + nb; x = result[t] of length nb).
template<class Elt>
void solve_one_from_augmented(const ModCtx<Elt>& ctx, const HowellResult<Elt>& H, uint32_t m, uint32_t nb,
                              DenseView<const Elt> Y, uint32_t t, std::vector<Elt>& u, std::vector<Elt>& x) {
    const uint32_t w = m + nb;
    std::fill(u.begin(), u.end(), Elt(0));
    for (uint32_t i = 0; i < m; ++i) u[i] = Y.at(i, t);
    uint32_t pi = 0;
    for (uint32_t j = 0; j < m; ++j) {
        const bool is_piv = pi < H.rows() && H.pivots[pi] == j;
        if (!is_piv) {
            if (u[j] != 0)
                throw std::runtime_error("Vector is not in image of basis (right-hand side " +
                                         std::to_string(t) + ").");
            continue;
        }
        if (u[j] != 0) {
            if (ctx.valuation(u[j]) < H.pivot_v[pi])
                throw std::runtime_error("Vector is not in image of basis (right-hand side " +
                                         std::to_string(t) + ").");
            const Elt q = ctx.divide_exact(u[j], ctx.ell_pow(H.pivot_v[pi]));
            row_sub_scaled(ctx, u.data(), H.H.data() + static_cast<size_t>(pi) * H.H.ld, q, j, w);
        }
        ++pi;
    }
    for (uint32_t j = 0; j < nb; ++j) x[j] = ctx.neg(u[m + j]);
    for (uint32_t i = pi; i < H.rows(); ++i) {
        const uint32_t j = H.pivots[i] - m;
        const Elt q = x[j] / ctx.ell_pow(H.pivot_v[i]);
        row_sub_scaled(ctx, x.data(), H.H.data() + static_cast<size_t>(i) * H.H.ld + m, q, j, nb);
    }
}

// (S23 G2-G) solve_multi_rhs_module's step 2 on a Howell form H of [B^T | I_nb] (B m x nb) already in hand.
template<class Elt>
std::vector<std::vector<Elt>> solve_from_augmented(const ModCtx<Elt>& ctx, const HowellResult<Elt>& H, uint32_t m,
                                                   uint32_t nb, DenseView<const Elt> Y) {
    const uint32_t ny = Y.cols(), w = m + nb;
    std::vector<std::vector<Elt>> result(ny, std::vector<Elt>(nb, Elt(0)));
    // (S24 MT-1) HECKE_MODULE_MT with a hook: the right-hand sides are independent (each reads H and its own column
    // of Y, writes its own result[t]), so they run as hook tasks over contiguous t ranges, each with its own u; a
    // task stops at its first failing t and records the message, and the SMALLEST failing t's message is thrown --
    // the serial outcome (m8).
    const ParallelFor* mpar = module_mt_hook();
    if (mpar != nullptr && ny >= 8 && static_cast<uint64_t>(ny) * w * std::max<uint32_t>(1, H.rows()) >= (uint64_t(1) << 22)) {
        module_mt_count(ModuleMtEvent::Gemm);
        const uint32_t ntask = std::min<uint32_t>(ny, 64);
        std::vector<uint32_t> fail_t(ntask, ~0u);
        std::vector<std::string> fail_msg(ntask);
        run_hook(mpar, ntask, [&](uint32_t x) {
            const uint32_t t0 = static_cast<uint32_t>(static_cast<uint64_t>(ny) * x / ntask);
            const uint32_t t1 = static_cast<uint32_t>(static_cast<uint64_t>(ny) * (x + 1) / ntask);
            std::vector<Elt> u(w);
            for (uint32_t t = t0; t < t1; ++t) {
                try {
                    solve_one_from_augmented<Elt>(ctx, H, m, nb, Y, t, u, result[t]);
                } catch (const std::runtime_error& e) {
                    fail_t[x] = t;
                    fail_msg[x] = e.what();
                    return;
                }
            }
        });
        for (uint32_t x = 0; x < ntask; ++x)
            if (fail_t[x] != ~0u) throw std::runtime_error(fail_msg[x]);
        return result;
    }
    std::vector<Elt> u(w);
    for (uint32_t t = 0; t < ny; ++t) {   // ascending t: the SMALLEST failing t throws (m8)
        std::fill(u.begin(), u.end(), Elt(0));
        for (uint32_t i = 0; i < m; ++i) u[i] = Y.at(i, t);
        uint32_t pi = 0;
        for (uint32_t j = 0; j < m; ++j) {
            const bool is_piv = pi < H.rows() && H.pivots[pi] == j;
            if (!is_piv) {
                if (u[j] != 0)
                    throw std::runtime_error("Vector is not in image of basis (right-hand side " +
                                             std::to_string(t) + ").");
                continue;
            }
            if (u[j] != 0) {
                if (ctx.valuation(u[j]) < H.pivot_v[pi])
                    throw std::runtime_error("Vector is not in image of basis (right-hand side " +
                                             std::to_string(t) + ").");
                const Elt q = ctx.divide_exact(u[j], ctx.ell_pow(H.pivot_v[pi]));
                row_sub_scaled(ctx, u.data(), H.H.data() + static_cast<size_t>(pi) * H.H.ld, q, j, w);
            }
            ++pi;
        }
        std::vector<Elt>& x = result[t];
        for (uint32_t j = 0; j < nb; ++j) x[j] = ctx.neg(u[m + j]);
        // reduce modulo ker B, the kernel's pivot columns left to right
        for (uint32_t i = pi; i < H.rows(); ++i) {
            const uint32_t j = H.pivots[i] - m;
            const Elt q = x[j] / ctx.ell_pow(H.pivot_v[i]);
            row_sub_scaled(ctx, x.data(), H.H.data() + static_cast<size_t>(i) * H.H.ld + m, q, j, nb);
        }
    }
    return result;
}

// (S23 G2-G) rows_of([Rl | G]^T) -- the store image_module([Rl | G]) builds -- without forming [Rl | G]:
// row j is column j of Rl (j < nr), else column j - nr of G.  The same bytes, so the same Howell form.
template<class Elt>
std::vector<Elt> columns_of_pair(DenseView<const Elt> Rl, DenseView<const Elt> G) {
    const uint32_t m = G.rows(), nr = Rl.cols(), ng = G.cols();
    std::vector<Elt> s(static_cast<size_t>(nr + ng) * m);
    for (uint32_t j = 0; j < nr; ++j)
        for (uint32_t i = 0; i < m; ++i) s[static_cast<size_t>(j) * m + i] = Rl.at(i, j);
    for (uint32_t j = 0; j < ng; ++j)
        for (uint32_t i = 0; i < m; ++i) s[static_cast<size_t>(nr + j) * m + i] = G.at(i, j);
    return s;
}

// (S23 G2-G) transpose_augmented(Gen) for Gen = H^T (m x s), H the s x m Howell rows: row g = [H row g | e_g].
template<class Elt>
std::vector<Elt> rows_augmented(const DenseBuf<Elt>& H) {
    const uint32_t s = H.rows, m = H.cols, w = m + s;
    std::vector<Elt> st(static_cast<size_t>(s) * w, Elt(0));
    for (uint32_t g = 0; g < s; ++g) {
        for (uint32_t i = 0; i < m; ++i) st[static_cast<size_t>(g) * w + i] = H.at(g, i);
        st[static_cast<size_t>(g) * w + m + g] = 1;
    }
    return st;
}

} // namespace

template<class Elt>
HowellResult<Elt> howell(const ModCtx<Elt>& ctx, DenseView<const Elt> A) {
    ENGINE_CHECK_MSG(!ctx.is_field(), ENGINE_COMPOSITE_ONLY("elim::howell"));
    return howell_rows<Elt>(ctx, rows_of<Elt>(A), A.rows(), A.cols());
}

// Sec. 13.6: the rows of howell([A^T | I_n]) with leading column >= m,
// restricted to their last n columns.  They generate {(0, x) : x A^T = 0}
// (the Howell property), and conditions 1-3 carry over to the restriction.
template<class Elt>
HowellResult<Elt> kernel_module(const ModCtx<Elt>& ctx, DenseView<const Elt> A) {
    ENGINE_CHECK_MSG(!ctx.is_field(), ENGINE_COMPOSITE_ONLY("elim::kernel_module"));
    const uint32_t m = A.rows(), n = A.cols();
    const HowellResult<Elt> full = howell_rows<Elt>(ctx, transpose_augmented<Elt>(A), n, m + n);
    return kernel_from_augmented<Elt>(full, m, n);
}

// Sec. 13.6: howell(A^T).
template<class Elt>
HowellResult<Elt> image_module(const ModCtx<Elt>& ctx, DenseView<const Elt> A) {
    ENGINE_CHECK_MSG(!ctx.is_field(), ENGINE_COMPOSITE_ONLY("elim::image_module"));
    return howell_rows<Elt>(ctx, rows_of<Elt>(A.transposed_view()), A.cols(), A.rows());
}

// (S24 G2-H) length(image_module(A)), the lean decompositions' len_rl: with HECKE_MODULE_LEAN_PEAK on (stream route)
// from the pivot loop alone (howell_rows' pivots_only: the same pivots and valuations, no H, no reduce-above).
template<class Elt>
uint32_t image_length(const ModCtx<Elt>& ctx, DenseView<const Elt> A) {
    if (!(module_lean_peak_enabled() && module_stream_enabled())) return length<Elt>(ctx, image_module<Elt>(ctx, A));
    ENGINE_CHECK_MSG(!ctx.is_field(), ENGINE_COMPOSITE_ONLY("elim::image_module"));
    lean_peak_count(LeanPeakEvent::LengthOnly);
    return length<Elt>(ctx, howell_rows<Elt>(ctx, rows_of<Elt>(A.transposed_view()), A.cols(), A.rows(), true));
}

// Sec. 13.6.  One Howell form H of rowspan[B^T | I_nb] = {((B z)^T, z^T)}
// serves both halves:
//  * the rows with leading column < m reduce (y_t^T, 0): walking the columns
//    j < m left to right, a pivot column needs ell^{v_j} | u_j (else y_t is
//    not in im B -- I_j = ell^{v_j} R), a non-pivot column needs u_j = 0
//    (I_j = 0); the x-part of the subtracted combination is then
//    x = -(last nb entries of u);
//  * the rows with leading column >= m are kernel_module(B) (above), and x is
//    reduced by them left to right: x[j] in [0, ell^{v_j}).  The reduced
//    representative of x + ker B is unique (Sec. 13.5's argument on d in K).
template<class Elt>
std::vector<std::vector<Elt>> solve_multi_rhs_module(const ModCtx<Elt>& ctx, DenseView<const Elt> B,
                                                     DenseView<const Elt> Y) {
    ENGINE_CHECK_MSG(!ctx.is_field(), ENGINE_COMPOSITE_ONLY("elim::solve_multi_rhs_module"));
    const uint32_t m = B.rows(), nb = B.cols(), ny = Y.cols();
    ENGINE_CHECK(Y.rows() == m);
    if (ny == 0) return std::vector<std::vector<Elt>>();
    const uint32_t w = m + nb;
    const HowellResult<Elt> H = howell_rows<Elt>(ctx, transpose_augmented<Elt>(B), nb, w);
    return solve_from_augmented<Elt>(ctx, H, m, nb, Y);
}

// (S24 MT-1) DenseElim.h: C = A B in row blocks over the hook.  Each block is kernels::gemm on the block's rows of
// A and C with the whole of B: every entry of C is the same exact dot product mod ell^k whatever the split
// (ENGINE_API.md Sec. 5), and the blocks write disjoint rows.
template<class Elt>
void gemm_rows_hook(const ParallelFor* par, const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<const Elt> B,
                    DenseView<Elt> C) {
    const uint32_t m = A.rows(), l = A.cols(), ny = B.cols();
    ENGINE_CHECK(B.rows() == l && C.rows() == m && C.cols() == ny);
    const uint32_t tiles = mod_tiles(par, m, static_cast<size_t>(m) * l * ny, 128, size_t(1) << 22, 64);
    if (tiles == 0) {
        kernels::gemm<Elt>(ctx, A, B, C);
        return;
    }
    module_mt_count(ModuleMtEvent::Gemm);
    run_hook(par, tiles, [&](uint32_t t) {
        const uint32_t r0 = static_cast<uint32_t>(static_cast<uint64_t>(m) * t / tiles);
        const uint32_t r1 = static_cast<uint32_t>(static_cast<uint64_t>(m) * (t + 1) / tiles);
        if (r1 > r0) kernels::gemm<Elt>(ctx, A.sub(r0, r1 - r0, 0, l), B, C.sub(r0, r1 - r0, 0, ny));
    });
}

// (slot-23 G2-A) DenseElim.h: X = Y[piv, :] when B[piv, :] = I, the residual
// Y - B X recomputed exactly over every row, in row blocks.
template<class Elt>
ReadoffStatus solve_unit_pivot_readoff(const ModCtx<Elt>& ctx, DenseView<const Elt> B,
                                       const std::vector<uint32_t>& piv, DenseView<const Elt> Y,
                                       DenseBuf<Elt>& X, uint32_t row_block) {
    const uint32_t m = B.rows(), nb = B.cols(), ny = Y.cols();
    ENGINE_CHECK(Y.rows() == m);
    if (piv.size() != nb) return ReadoffStatus::NotIdentity;
    for (uint32_t c = 0; c < nb; ++c)
        if (piv[c] >= m || (c > 0 && piv[c] <= piv[c - 1])) return ReadoffStatus::NotIdentity;
    for (uint32_t r = 0; r < nb; ++r)                       // B[piv, :] == I, O(nb^2)
        for (uint32_t c = 0; c < nb; ++c)
            if (B.at(piv[r], c) != (r == c ? Elt(1) : Elt(0))) return ReadoffStatus::NotIdentity;
    X = DenseBuf<Elt>(nb, ny);
    for (uint32_t r = 0; r < nb; ++r)
        for (uint32_t t = 0; t < ny; ++t) X.at(r, t) = Y.at(piv[r], t);
    if (nb > 0 && ny > 0 && g_readoff_corruption.load(std::memory_order_relaxed) != 0)   // validation only
        X.at(0, 0) = ctx.add(X.at(0, 0), Elt(1));
    if (m == 0 || ny == 0) return ReadoffStatus::Ok;
    if (row_block == 0) row_block = 1;
    const uint32_t rb = std::min(row_block, m);
    // (S24 MT-1) HECKE_MODULE_MT with a hook: the residual's row blocks as hook tasks, each with its own product
    // buffer (the same GEMM on the same operands per block; read-only X, B, Y).  Residual iff some block fails --
    // the serial outcome (the serial loop stops at its first failing block; the status carries no position).
    const ParallelFor* mpar = module_mt_hook();
    const uint32_t nblk = (m + rb - 1) / rb;
    if (mpar != nullptr && nblk >= 2 && static_cast<uint64_t>(m) * nb * ny >= (uint64_t(1) << 20)) {
        module_mt_count(ModuleMtEvent::Gemm);
        const uint32_t ntask = std::min<uint32_t>(nblk, 64);
        std::vector<uint8_t> bad(ntask, 0);
        run_hook(mpar, ntask, [&](uint32_t x) {
            DenseBuf<Elt> Pt(rb, ny);
            for (uint32_t blk = x; blk < nblk && !bad[x]; blk += ntask) {
                const uint32_t r0 = blk * rb, nr = std::min(rb, m - r0);
                DenseView<Elt> Pv = view(Pt).sub(0, nr, 0, ny);
                if (nb > 0) {
                    kernels::gemm<Elt>(ctx, B.sub(r0, nr, 0, nb), cview(X), Pv);
                } else {
                    for (uint32_t i = 0; i < nr; ++i)
                        for (uint32_t t = 0; t < ny; ++t) Pt.at(i, t) = 0;
                }
                for (uint32_t i = 0; i < nr && !bad[x]; ++i)
                    for (uint32_t t = 0; t < ny; ++t)
                        if (Pt.at(i, t) != Y.at(r0 + i, t)) { bad[x] = 1; break; }
            }
        });
        for (uint8_t b : bad) if (b) return ReadoffStatus::Residual;
        return ReadoffStatus::Ok;
    }
    DenseBuf<Elt> P(rb, ny);
    for (uint32_t r0 = 0; r0 < m; r0 += rb) {
        const uint32_t nr = std::min(rb, m - r0);
        DenseView<Elt> Pv = view(P).sub(0, nr, 0, ny);
        if (nb > 0) {
            kernels::gemm<Elt>(ctx, B.sub(r0, nr, 0, nb), cview(X), Pv);
        } else {
            for (uint32_t i = 0; i < nr; ++i)
                for (uint32_t t = 0; t < ny; ++t) P.at(i, t) = 0;
        }
        for (uint32_t i = 0; i < nr; ++i)
            for (uint32_t t = 0; t < ny; ++t)
                if (P.at(i, t) != Y.at(r0 + i, t)) return ReadoffStatus::Residual;
    }
    return ReadoffStatus::Ok;
}

// (slot-23 G2-E) DenseElim.h: the read-off on a product Y = A G, Y never materialised.
template<class Elt>
ReadoffStatus solve_unit_pivot_readoff_product(const ModCtx<Elt>& ctx, DenseView<const Elt> B,
                                               const std::vector<uint32_t>& piv, DenseView<const Elt> A,
                                               DenseView<const Elt> G, DenseBuf<Elt>& X, uint32_t row_block,
                                               uint32_t perturb_row) {
    const uint32_t m = B.rows(), nb = B.cols(), l = A.cols(), ny = G.cols();
    ENGINE_CHECK(A.rows() == m && G.rows() == l);
    if (piv.size() != nb) return ReadoffStatus::NotIdentity;
    for (uint32_t c = 0; c < nb; ++c)
        if (piv[c] >= m || (c > 0 && piv[c] <= piv[c - 1])) return ReadoffStatus::NotIdentity;
    for (uint32_t r = 0; r < nb; ++r)                       // B[piv, :] == I, O(nb^2)
        for (uint32_t c = 0; c < nb; ++c)
            if (B.at(piv[r], c) != (r == c ? Elt(1) : Elt(0))) return ReadoffStatus::NotIdentity;
    if (row_block == 0) row_block = 1;
    X = DenseBuf<Elt>(nb, ny);
    DenseBuf<Elt> Ynp(m - nb, ny);                          // the non-pivot rows of Y, in row order
    if (m == 0 || ny == 0) return ReadoffStatus::Ok;
    const uint32_t rb = std::min(row_block, m);
    // (S23 G2-P) HECKE_MODULE_PARALLEL with an elimination hook installed (compute_module_stream): the row blocks of
    // both passes run as hook tasks.  Each block's destination rows (pivot -> X, other -> Ynp) are fixed by piv, so
    // the blocks write disjoint rows with the same GEMM on the same operands; pass 2 reports Residual iff some
    // block does, which is the serial outcome (the serial loop returns at its first failing block).
    const ParallelFor* hpar = module_parallel_enabled() ? parallel_hook() : nullptr;
    const uint32_t nblk = (m + rb - 1) / rb;
    if (hpar != nullptr && nblk >= 2 && static_cast<size_t>(m) * ny * std::max<uint32_t>(l, 1) >= (size_t(1) << 20)) {
        g_mp_readoff.fetch_add(1, std::memory_order_relaxed);
        std::vector<uint32_t> dest(m);                      // row -> index in X (pivot) or in Ynp (others)
        std::vector<uint8_t> is_piv(m, 0);
        {
            uint32_t u = 0, w = 0;
            for (uint32_t row = 0; row < m; ++row) {
                if (u < nb && piv[u] == row) { is_piv[row] = 1; dest[row] = u++; }
                else dest[row] = w++;
            }
            ENGINE_CHECK(u == nb && w == m - nb);
        }
        run_hook(hpar, nblk, [&](uint32_t bi) {
            const uint32_t r0 = bi * rb, nr = std::min(rb, m - r0);
            DenseBuf<Elt> Pb(nr, ny);
            if (l > 0) kernels::gemm<Elt>(ctx, A.sub(r0, nr, 0, l), G, view(Pb));
            if (perturb_row >= r0 && perturb_row < r0 + nr)     // validation only
                Pb.at(perturb_row - r0, 0) = ctx.add(Pb.at(perturb_row - r0, 0), Elt(1));
            for (uint32_t i = 0; i < nr; ++i) {
                const uint32_t row = r0 + i;
                Elt* dst = is_piv[row] ? &X.at(dest[row], 0) : &Ynp.at(dest[row], 0);
                for (uint32_t t = 0; t < ny; ++t) dst[t] = Pb.at(i, t);
            }
        });
        const Elt y00 = (nb > 0) ? X.at(0, 0) : Elt(0);
        if (nb > 0 && g_readoff_corruption.load(std::memory_order_relaxed) != 0)   // validation only
            X.at(0, 0) = ctx.add(X.at(0, 0), Elt(1));
        std::atomic<int> bad{0};
        // validation only (G2-P audit F1): one fault on a block run by a pool worker, not by the calling thread
        const int fault = g_readoff_worker_fault.load(std::memory_order_relaxed);
        const std::thread::id caller = std::this_thread::get_id();
        std::atomic<bool> fault_fired{false};
        run_hook(hpar, nblk, [&](uint32_t bi) {
            if (fault != 0) {
                if (std::this_thread::get_id() != caller) {
                    if (!fault_fired.exchange(true)) {
                        g_readoff_worker_faults.fetch_add(1, std::memory_order_relaxed);
                        if (fault == 2)
                            throw std::runtime_error("solve_unit_pivot_readoff_product: injected worker fault (validation only)");
                        bad.store(1, std::memory_order_relaxed);
                        return;
                    }
                } else {   // let a worker take a block first (bounded: no wedge if none ever joins)
                    const auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                    while (!fault_fired.load() && std::chrono::steady_clock::now() < t_end) std::this_thread::yield();
                }
            }
            const uint32_t r0 = bi * rb, nr = std::min(rb, m - r0);
            DenseBuf<Elt> Pb(nr, ny);
            if (nb > 0) kernels::gemm<Elt>(ctx, B.sub(r0, nr, 0, nb), cview(X), view(Pb));
            for (uint32_t i = 0; i < nr; ++i) {
                const uint32_t row = r0 + i;
                const Elt* y = is_piv[row] ? &X.at(dest[row], 0) : &Ynp.at(dest[row], 0);
                const bool first_pivot = nb > 0 && piv[0] == row;
                for (uint32_t t = 0; t < ny; ++t)
                    if (Pb.at(i, t) != ((first_pivot && t == 0) ? y00 : y[t])) { bad.store(1, std::memory_order_relaxed); return; }
            }
        });
        return bad.load() != 0 ? ReadoffStatus::Residual : ReadoffStatus::Ok;
    }
    DenseBuf<Elt> P(rb, ny);
    // pass 1: Y in row blocks; pivot rows -> X, the other rows -> Ynp
    uint32_t u = 0, w = 0;                                  // next pivot index, next non-pivot row of Ynp
    for (uint32_t r0 = 0; r0 < m; r0 += rb) {
        const uint32_t nr = std::min(rb, m - r0);
        DenseView<Elt> Pv = view(P).sub(0, nr, 0, ny);
        if (l > 0) {
            kernels::gemm<Elt>(ctx, A.sub(r0, nr, 0, l), G, Pv);
        } else {
            for (uint32_t i = 0; i < nr; ++i)
                for (uint32_t t = 0; t < ny; ++t) P.at(i, t) = 0;
        }
        if (perturb_row >= r0 && perturb_row < r0 + nr)     // validation only
            P.at(perturb_row - r0, 0) = ctx.add(P.at(perturb_row - r0, 0), Elt(1));
        for (uint32_t i = 0; i < nr; ++i) {
            const uint32_t row = r0 + i;
            Elt* dst;
            if (u < nb && piv[u] == row) dst = &X.at(u++, 0);
            else dst = &Ynp.at(w++, 0);
            for (uint32_t t = 0; t < ny; ++t) dst[t] = P.at(i, t);
        }
    }
    ENGINE_CHECK(u == nb && w == m - nb);
    // Y(piv[0], 0) as formed: the pivot rows of Y live only in X, so the residual of row piv[0] is taken against
    // this copy (what solve_unit_pivot_readoff compares with Y itself), which the corruption hook leaves untouched.
    const Elt y00 = (nb > 0) ? X.at(0, 0) : Elt(0);
    if (nb > 0 && g_readoff_corruption.load(std::memory_order_relaxed) != 0)   // validation only
        X.at(0, 0) = ctx.add(X.at(0, 0), Elt(1));
    // pass 2: the residual Y - B X over every row (a pivot row against its copy in X, the others against Ynp)
    u = 0; w = 0;
    for (uint32_t r0 = 0; r0 < m; r0 += rb) {
        const uint32_t nr = std::min(rb, m - r0);
        DenseView<Elt> Pv = view(P).sub(0, nr, 0, ny);
        if (nb > 0) {
            kernels::gemm<Elt>(ctx, B.sub(r0, nr, 0, nb), cview(X), Pv);
        } else {
            for (uint32_t i = 0; i < nr; ++i)
                for (uint32_t t = 0; t < ny; ++t) P.at(i, t) = 0;
        }
        for (uint32_t i = 0; i < nr; ++i) {
            const uint32_t row = r0 + i;
            const Elt* y;
            const bool first_pivot = u == 0 && nb > 0 && piv[0] == row;
            if (u < nb && piv[u] == row) y = &X.at(u++, 0);
            else y = &Ynp.at(w++, 0);
            for (uint32_t t = 0; t < ny; ++t)
                if (P.at(i, t) != ((first_pivot && t == 0) ? y00 : y[t])) return ReadoffStatus::Residual;
        }
    }
    return ReadoffStatus::Ok;
}

// (S24 G2-H) DenseElim.h: the read-off's certificate on a product, X and Y never formed.
template<class Elt>
ReadoffStatus unit_pivot_residual_product(const ModCtx<Elt>& ctx, DenseView<const Elt> B,
                                          const std::vector<uint32_t>& piv, DenseView<const Elt> A,
                                          DenseView<const Elt> G, uint32_t row_block, uint32_t perturb_row) {
    const uint32_t m = B.rows(), nb = B.cols(), l = A.cols(), ny = G.cols();
    ENGINE_CHECK(A.rows() == m && G.rows() == l);
    // the product read-off's checks, in its order
    if (piv.size() != nb) return ReadoffStatus::NotIdentity;
    for (uint32_t c = 0; c < nb; ++c)
        if (piv[c] >= m || (c > 0 && piv[c] <= piv[c - 1])) return ReadoffStatus::NotIdentity;
    for (uint32_t r = 0; r < nb; ++r)                       // B[piv, :] == I, O(nb^2)
        for (uint32_t c = 0; c < nb; ++c)
            if (B.at(piv[r], c) != (r == c ? Elt(1) : Elt(0))) return ReadoffStatus::NotIdentity;
    if (row_block == 0) row_block = 1;
    if (m == 0 || ny == 0) return ReadoffStatus::Ok;
    // validation only: the read-off's X(0, 0) + 1 makes row piv[0] of its residual non-zero
    if (nb > 0 && g_readoff_corruption.load(std::memory_order_relaxed) != 0) return ReadoffStatus::Residual;
    const uint32_t mn = m - nb;
    if (mn == 0) return ReadoffStatus::Ok;                  // every row is a pivot row: the residual is zero
    std::vector<uint32_t> np;                               // the non-pivot rows, ascending
    np.reserve(mn);
    {
        uint32_t u = 0;
        for (uint32_t row = 0; row < m; ++row) {
            if (u < nb && piv[u] == row) ++u;
            else np.push_back(row);
        }
        ENGINE_CHECK(u == nb && np.size() == mn);
    }
    // the perturbation's site (validation only): a non-pivot row index w, or a pivot index u
    uint32_t pert_w = ~0u, pert_u = ~0u;
    if (perturb_row < m) {
        const auto it = std::lower_bound(piv.begin(), piv.end(), perturb_row);
        if (it != piv.end() && *it == perturb_row) pert_u = static_cast<uint32_t>(it - piv.begin());
        else pert_w = static_cast<uint32_t>(std::lower_bound(np.begin(), np.end(), perturb_row) - np.begin());
    }
    const ParallelFor* hpar = module_parallel_enabled() ? parallel_hook() : nullptr;
    // pivot blocks of <= 64 rows (the gathered A[piv, :] block is <= 64 x l); tiles of <= 128 non-pivot rows (row_block
    // caps both further): per tile Rt = A[tile, :] - B[tile, :] A[piv, :] (the pivot blocks in ascending order, one
    // GEMM each), then P = Rt G with the perturbation applied, P == 0 checked.  Buffers O(128 (l + ny)), not O(mn l).
    const uint32_t pb = std::min(std::min<uint32_t>(row_block, 64), std::max<uint32_t>(nb, 1));
    const uint32_t st = std::min(std::min<uint32_t>(row_block, 128), mn);
    // true iff some non-pivot row in [w0, w1) has a non-zero residual (stops at the first failing tile)
    auto range_bad = [&](uint32_t w0, uint32_t w1) -> bool {
        if (w1 <= w0) return false;
        DenseBuf<Elt> Rt(st, l), Ap(pb, l), Bs(st, pb), P(st, ny);
        for (uint32_t s0 = w0; s0 < w1; s0 += st) {
            const uint32_t nw = std::min(st, w1 - s0);
            for (uint32_t w = 0; w < nw; ++w)
                for (uint32_t j = 0; j < l; ++j) Rt.at(w, j) = A.at(np[s0 + w], j);
            if (l > 0) {
                for (uint32_t u0 = 0; u0 < nb; u0 += pb) {
                    const uint32_t nu = std::min(pb, nb - u0);
                    for (uint32_t x = 0; x < nu; ++x)
                        for (uint32_t j = 0; j < l; ++j) Ap.at(x, j) = A.at(piv[u0 + x], j);
                    for (uint32_t w = 0; w < nw; ++w)
                        for (uint32_t x = 0; x < nu; ++x) Bs.at(w, x) = B.at(np[s0 + w], u0 + x);
                    kernels::gemm<Elt>(ctx, cview(Bs).sub(0, nw, 0, nu), cview(Ap).sub(0, nu, 0, l),
                                       view(Rt).sub(0, nw, 0, l), GemmOp::Sub);
                }
                kernels::gemm<Elt>(ctx, cview(Rt).sub(0, nw, 0, l), G, view(P).sub(0, nw, 0, ny));
            } else {
                for (uint32_t i = 0; i < nw; ++i)
                    for (uint32_t t = 0; t < ny; ++t) P.at(i, t) = 0;
            }
            if (pert_w >= s0 && pert_w < s0 + nw) P.at(pert_w - s0, 0) = ctx.add(P.at(pert_w - s0, 0), Elt(1));
            if (pert_u != ~0u)
                for (uint32_t i = 0; i < nw; ++i) P.at(i, 0) = ctx.sub(P.at(i, 0), B.at(np[s0 + i], pert_u));
            for (uint32_t i = 0; i < nw; ++i)
                for (uint32_t t = 0; t < ny; ++t)
                    if (P.at(i, t) != 0) return true;
        }
        return false;
    };
    // HECKE_MODULE_PARALLEL with a hook: contiguous ranges of non-pivot rows as hook tasks (each with its own tile
    // buffers); Residual iff some range fails -- the serial outcome (the status carries no position).
    const uint32_t tiles = mod_tiles(hpar, mn, static_cast<size_t>(mn) * std::max<uint32_t>(l, 1) * (nb + ny), 64,
                                      size_t(1) << 20, 32);
    if (tiles == 0) return range_bad(0, mn) ? ReadoffStatus::Residual : ReadoffStatus::Ok;
    g_mp_readoff.fetch_add(1, std::memory_order_relaxed);
    std::vector<uint8_t> bad(tiles, 0);
    run_hook(hpar, tiles, [&](uint32_t t) {
        bad[t] = range_bad(static_cast<uint32_t>(static_cast<uint64_t>(mn) * t / tiles),
                           static_cast<uint32_t>(static_cast<uint64_t>(mn) * (t + 1) / tiles)) ? 1 : 0;
    });
    for (uint8_t x : bad) if (x) return ReadoffStatus::Residual;
    return ReadoffStatus::Ok;
}

// (slot-23 G2-B; audit m1) DenseElim.h: the d o d predicate A B = 0, by GEMM or the legacy scalar loop.
template<class Elt>
bool product_is_zero(const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<const Elt> B, bool gemm) {
    ENGINE_CHECK(A.cols() == B.rows());
    if (gemm) {
        if (A.rows() == 0 || B.cols() == 0 || A.cols() == 0) return true;   // the product is empty or zero
        DenseBuf<Elt> P(A.rows(), B.cols());
        kernels::gemm<Elt>(ctx, A, B, view(P));
        for (uint32_t i = 0; i < P.rows; ++i)
            for (uint32_t j = 0; j < P.cols; ++j)
                if (P.at(i, j) != 0) return false;
        return true;
    }
    for (uint32_t i = 0; i < A.rows(); ++i)
        for (uint32_t j = 0; j < B.cols(); ++j) {
            Elt acc = 0;
            for (uint32_t l = 0; l < A.cols(); ++l) acc = ctx.add(acc, ctx.mul(A.at(i, l), B.at(l, j)));
            if (acc != 0) return false;
        }
    return true;
}

template<class Elt>
uint32_t length(const ModCtx<Elt>& ctx, const HowellResult<Elt>& h) {
    ENGINE_CHECK_MSG(!ctx.is_field(), ENGINE_COMPOSITE_ONLY("elim::length"));
    uint32_t L = 0;
    for (uint32_t v : h.pivot_v) { ENGINE_CHECK(v < ctx.k); L += ctx.k - v; }
    return L;
}

template<class Elt>
uint32_t length(const ModCtx<Elt>& ctx, DenseView<const Elt> A) {
    ENGINE_CHECK_MSG(!ctx.is_field(), ENGINE_COMPOSITE_ONLY("elim::length"));
    return length<Elt>(ctx, howell<Elt>(ctx, A));
}

// Sec. 13.6's rule, step by step:
//  1. present M = R^s / Rel: the s generators are the Howell rows of
//     N = colspan[Rl | G] (image_module); Rel is generated by kernel_module of
//     that generator matrix Gen (m x s) and by the coordinates
//     (solve_multi_rhs_module) of the columns of Rl in Gen;
//  2. the Smith form of the s x r relation matrix (rows = generators, one
//     column per relation) by MINIMAL-VALUATION pivoting -- smallest valuation
//     anywhere, then leftmost column, then topmost row -- clearing the pivot's
//     row first (column operations) and its column second (row operations),
//     each by the divide_exact quotient, and scaling the pivot to ell^sigma
//     (review M3's rule; only the valuations are read here);
//  3. a diagonal ell^sigma with 1 <= sigma < k gives e = sigma; each of the s
//     generators whose diagonal entry is zero or missing gives e = k; unit
//     diagonal entries give nothing;
//  4. check: sum e_i = length(N) - length(colspan Rl)  (ENGINE_CHECKed).
// The result is unique by the structure theorem for finite Z/ell^k-modules.
template<class Elt>
std::vector<uint32_t> invariant_factors(const ModCtx<Elt>& ctx, DenseView<const Elt> G,
                                        DenseView<const Elt> Rl) {
    ENGINE_CHECK_MSG(!ctx.is_field(), ENGINE_COMPOSITE_ONLY("elim::invariant_factors"));
    const uint32_t m = G.rows(), ng = G.cols(), nr = Rl.cols();
    ENGINE_CHECK_MSG(Rl.rows() == m, "elim::invariant_factors: G and Rl must have the same row count");
    // (S23 G2-G) HECKE_MODULE_STREAM: the lean work buffers -- N = [Rl | G] and Gen are never formed (their Howell
    // stores are built directly), one Howell form of [Gen^T | I] serves the kernel and the coordinates, and each
    // buffer is dropped once read; length(colspan Rl) right after s = 0 (audit n2).  The same stores, Howell forms, the
    // same relation matrix: bit-identical (HeckeModuleStreamValidation (C)).  Off: the legacy buffers.
    const bool lean = module_stream_enabled();
    (lean ? g_ms_invf_lean : g_ms_invf_legacy).fetch_add(1, std::memory_order_relaxed);
    uint32_t len_rl = 0;
    DenseBuf<Elt> N;                                    // N = [Rl | G] (legacy only)
    HowellResult<Elt> Nh;
    if (lean) {
        Nh = howell_rows<Elt>(ctx, columns_of_pair<Elt>(Rl, G), nr + ng, m);
    } else {
        N = DenseBuf<Elt>(m, nr + ng);
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < nr; ++j) N.at(i, j) = Rl.at(i, j);
            for (uint32_t j = 0; j < ng; ++j) N.at(i, nr + j) = G.at(i, j);
        }
        Nh = image_module<Elt>(ctx, cview(N));
    }
    const uint32_t s = Nh.rows();
    std::vector<uint32_t> e;
    if (s == 0) return e;
    // (audit n2) after the s = 0 return (s = 0 forces Rl = 0) and after N's elimination, as on the legacy route
    if (lean && nr > 0) len_rl = image_length<Elt>(ctx, Rl);   // (S24 G2-H) the pivot loop alone, lever on
    DenseBuf<Elt> Gen;                                  // m x s (legacy only)
    HowellResult<Elt> K;
    std::vector<std::vector<Elt>> coords;
    if (lean) {
        const HowellResult<Elt> full = howell_rows<Elt>(ctx, rows_augmented<Elt>(Nh.H), s, m + s);
        if (nr > 0) coords = solve_from_augmented<Elt>(ctx, full, m, s, Rl);   // Rl lies in span Gen
        K = kernel_from_augmented<Elt>(full, m, s);
        Nh.H = DenseBuf<Elt>();                         // length(Nh) below reads pivot_v only
    } else {
        Gen = DenseBuf<Elt>(m, s);
        for (uint32_t g = 0; g < s; ++g)
            for (uint32_t i = 0; i < m; ++i) Gen.at(i, g) = Nh.H.at(g, i);
        K = kernel_module<Elt>(ctx, cview(Gen));
        if (nr > 0) coords = solve_multi_rhs_module<Elt>(ctx, cview(Gen), Rl);   // Rl lies in span Gen
    }
    const uint32_t r = K.rows() + nr;
    std::vector<Elt> Rm(static_cast<size_t>(s) * r, Elt(0));
    auto at = [&](uint32_t i, uint32_t j) -> Elt& { return Rm[static_cast<size_t>(i) * r + j]; };
    const uint32_t kr = K.rows();
    for (uint32_t c = 0; c < kr; ++c)
        for (uint32_t i = 0; i < s; ++i) at(i, c) = K.H.at(c, i);
    for (uint32_t c = 0; c < nr; ++c)
        for (uint32_t i = 0; i < s; ++i) at(i, kr + c) = coords[c][i];
    if (lean) { K = HowellResult<Elt>(); std::vector<std::vector<Elt>>().swap(coords); }
    if (lean && s > 0 && r > 0 && g_lean_buffer_corruption.load(std::memory_order_relaxed) != 0)   // validation only
        Rm[0] = ctx.add(Rm[0], Elt(1));
    std::vector<char> row_used(s, 0), col_used(r, 0);
    uint32_t n_piv = 0;
    for (;;) {
        uint32_t pr = s, pc = r, pv = ctx.k;
        for (uint32_t j = 0; j < r; ++j) {              // leftmost column first ...
            if (col_used[j]) continue;
            for (uint32_t i = 0; i < s; ++i) {          // ... topmost row first; strict < keeps the first
                if (row_used[i]) continue;
                const Elt a = at(i, j);
                if (a == 0) continue;
                const uint32_t v = ctx.valuation(a);
                if (v < pv) { pv = v; pr = i; pc = j; }
            }
        }
        if (pr == s) break;
        const Elt d = ctx.ell_pow(pv);
        const Elt uinv = ctx.inv(ctx.unit_part(at(pr, pc)));
        // clear the pivot's row: col_j -= q * col_pc, q = a_{pr,j} / pivot
        for (uint32_t j = 0; j < r; ++j) {
            if (j == pc || col_used[j] || at(pr, j) == 0) continue;
            const Elt qp = ctx.mul_prep(ctx.mul(ctx.divide_exact(at(pr, j), d), uinv));
            for (uint32_t i = 0; i < s; ++i) {
                if (row_used[i] || at(i, pc) == 0) continue;
                at(i, j) = ctx.sub(at(i, j), ctx.mul_prepped(at(i, pc), qp));
            }
            ENGINE_DEBUG_CHECK(at(pr, j) == 0);
        }
        // clear the pivot's column: row_i -= q * row_pr.  Row pr is now zero
        // off the pivot on every remaining column, so this zeroes (i, pc) and
        // changes nothing else.
        for (uint32_t i = 0; i < s; ++i)
            if (i != pr && !row_used[i]) at(i, pc) = 0;
        at(pr, pc) = d;                                 // the pivot scaled to ell^sigma
        row_used[pr] = 1; col_used[pc] = 1; ++n_piv;
        if (pv >= 1) e.push_back(pv);
    }
    for (uint32_t i = n_piv; i < s; ++i) e.push_back(ctx.k);
    std::sort(e.begin(), e.end(), [](uint32_t a, uint32_t b) { return a > b; });
    uint32_t sum = 0;
    for (uint32_t x : e) sum += x;
    if (lean) std::vector<Elt>().swap(Rm);
    else len_rl = (nr == 0) ? 0u : length<Elt>(ctx, image_module<Elt>(ctx, Rl));
    ENGINE_CHECK_MSG(sum + len_rl == length<Elt>(ctx, Nh),
                     "elim::invariant_factors: sum e_i != length(N) - length(colspan Rl) (ENGINE_API.md Sec. 13.6 step 4)");
    return e;
}

template<class Elt>
std::vector<uint32_t> invariant_factors(const ModCtx<Elt>& ctx, DenseView<const Elt> A) {
    const DenseBuf<Elt> none(A.rows(), 0);
    return invariant_factors<Elt>(ctx, A, cview(none));
}

// (slot-8 WP-3) [E] sec:ring (c), step by step -- see DenseElim.h.
template<class Elt>
CyclicDecomposition<Elt> cyclic_decomposition(const ModCtx<Elt>& ctx, DenseView<const Elt> G,
                                              DenseView<const Elt> Rl) {
    ENGINE_CHECK_MSG(!ctx.is_field(), ENGINE_COMPOSITE_ONLY("elim::cyclic_decomposition"));
    const uint32_t m = G.rows(), ng = G.cols(), nr = Rl.cols();
    ENGINE_CHECK_MSG(Rl.rows() == m, "elim::cyclic_decomposition: G and Rl must have the same row count");
    CyclicDecomposition<Elt> out;
    // (S23 G2-G) HECKE_MODULE_STREAM: the lean work buffers.  N = [Rl | G] is never formed (image_module's
    // store rows_of(N^T) is built from Rl and G directly); Nh.H is dropped once copied into Gamma (length(N)
    // reads pivot_v only); ONE Howell form of [Gamma^T | I_s] gives both kernel_module(Gamma) and the
    // coordinates of Rl (each of kernel_module and solve_multi_rhs_module computes exactly that form); K and
    // the coordinates are dropped once copied into the relation matrix, the relation matrix once (c2)-(c3)
    // is done; length(colspan Rl) is taken right after the s = 0 return (audit n2).  The same stores, the same
    // Howell forms, the same relation matrix, the same operations: bit-identical (HeckeModuleStreamValidation
    // (C)).  Off: the legacy buffers.
    const bool lean = module_stream_enabled();
    (lean ? g_ms_cyc_lean : g_ms_cyc_lean_legacy).fetch_add(1, std::memory_order_relaxed);
    uint32_t len_rl = 0;
    // (c1) Gamma: the Howell generators of N = colspan[Rl | G].
    DenseBuf<Elt> N;                                   // legacy only
    HowellResult<Elt> Nh;
    if (lean) {
        Nh = howell_rows<Elt>(ctx, columns_of_pair<Elt>(Rl, G), nr + ng, m);
    } else {
        N = DenseBuf<Elt>(m, nr + ng);
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < nr; ++j) N.at(i, j) = Rl.at(i, j);
            for (uint32_t j = 0; j < ng; ++j) N.at(i, nr + j) = G.at(i, j);
        }
        Nh = image_module<Elt>(ctx, cview(N));
    }
    const uint32_t s = Nh.rows();
    out.Gamma = DenseBuf<Elt>(m, s);
    for (uint32_t g = 0; g < s; ++g)
        for (uint32_t i = 0; i < m; ++i) out.Gamma.at(i, g) = Nh.H.at(g, i);
    if (lean) Nh.H = DenseBuf<Elt>();
    out.U = DenseBuf<Elt>(s, s);
    for (uint32_t i = 0; i < s; ++i) out.U.at(i, i) = 1;
    out.basis = DenseBuf<Elt>(m, 0);
    if (s == 0) return out;
    // (audit n2) after the s = 0 return (s = 0 forces Rl = 0) and after N's elimination, as on the legacy route
    if (lean && nr > 0) len_rl = image_length<Elt>(ctx, Rl);   // (S24 G2-H) the pivot loop alone, lever on
    HowellResult<Elt> K;
    std::vector<std::vector<Elt>> coords;
    if (lean) {
        const HowellResult<Elt> full = howell_rows<Elt>(ctx, transpose_augmented<Elt>(cview(out.Gamma)), s, m + s);
        if (nr > 0) coords = solve_from_augmented<Elt>(ctx, full, m, s, Rl);
        K = kernel_from_augmented<Elt>(full, m, s);
    } else {
        K = kernel_module<Elt>(ctx, cview(out.Gamma));
        if (nr > 0) coords = solve_multi_rhs_module<Elt>(ctx, cview(out.Gamma), Rl);
    }
    const uint32_t kr = K.rows();
    const uint32_t r = kr + nr;
    std::vector<Elt> Rm(static_cast<size_t>(s) * r, Elt(0));
    auto at = [&](uint32_t i, uint32_t j) -> Elt& { return Rm[static_cast<size_t>(i) * r + j]; };
    for (uint32_t c = 0; c < kr; ++c)
        for (uint32_t i = 0; i < s; ++i) at(i, c) = K.H.at(c, i);
    for (uint32_t c = 0; c < nr; ++c)
        for (uint32_t i = 0; i < s; ++i) at(i, kr + c) = coords[c][i];
    if (lean) { K = HowellResult<Elt>(); std::vector<std::vector<Elt>>().swap(coords); }
    if (lean && s > 0 && r > 0 && g_lean_buffer_corruption.load(std::memory_order_relaxed) != 0)   // validation only
        Rm[0] = ctx.add(Rm[0], Elt(1));
    DenseBuf<Elt> Gm = std::move(out.Gamma);   // mirrored copy: becomes the transformed generators
    out.Gamma = DenseBuf<Elt>(m, s);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t g = 0; g < s; ++g) out.Gamma.at(i, g) = Gm.at(i, g);
    // (c2)-(c3)
    std::vector<uint32_t> sigma;             // sigma[t]: the valuation of diagonal position t
    const bool interchange = lean;   // (slot-23 G2-B) HECKE_MODULE_STREAM, read once per call (G2-G audit n1); value-neutral
    (interchange ? g_ms_cyc_new : g_ms_cyc_old).fetch_add(1, std::memory_order_relaxed);
    std::vector<uint32_t> qcols;
    std::vector<Elt> qprep;
    const uint32_t tmax = std::min(s, r);
    uint32_t t = 0;
    // (S24 MT-1) HECKE_MODULE_MT on the lean route: the pivot search exits at the first unit (any thread count);
    // with an elimination hook (and HECKE_MODULE_PARALLEL) the remaining columns and the row operations are split
    // over the pool.  Off (=0): the full scan and the serial loops below, verbatim.
    const bool mt = interchange && module_mt_enabled();
    const ParallelFor* cpar = mt ? module_mt_hook() : nullptr;
    std::vector<uint32_t> tile_v, tile_i, tile_j;
    std::vector<std::pair<uint32_t, Elt>> clear_list;
    if (mt) module_mt_count(ModuleMtEvent::PivotExit);   // (MT-1 audit F2) calls on the exit search (every thread count)
    for (; t < tmax; ++t) {
        uint32_t pr = s, pc = r, pv = ctx.k;
        if (mt) {
            // The serial order is column-major from (t, t) with a strict <: the pivot is the FIRST entry (in that
            // order) of minimal valuation.  A unit (v = 0) is minimal and no later entry is strictly smaller, so
            // the scan may stop at the first unit; before it, the scan below is the serial one.
            for (uint32_t i = t; i < s; ++i) {             // column t first
                const Elt a = at(i, t);
                if (a == 0) continue;
                const uint32_t v = ctx.valuation(a);
                if (v < pv) { pv = v; pr = i; pc = t; if (v == 0) break; }
            }
            if (pv != 0 && t + 1 < r) {
                const uint32_t ncols = r - t - 1;
                const uint32_t tiles = cyc_tiles(cpar, ncols, static_cast<size_t>(ncols) * (s - t), 64, size_t(1) << 16, 16);
                if (tiles == 0) {
                    for (uint32_t j = t + 1; j < r && pv != 0; ++j)
                        for (uint32_t i = t; i < s; ++i) {
                            const Elt a = at(i, j);
                            if (a == 0) continue;
                            const uint32_t v = ctx.valuation(a);
                            if (v < pv) { pv = v; pr = i; pc = j; if (v == 0) break; }
                        }
                } else {
                    // per tile of consecutive columns: its first minimum in the serial order (exit at a unit);
                    // combined in tile order with the same strict <, which is the serial first minimum
                    module_mt_count(ModuleMtEvent::CycSearch);
                    tile_v.assign(tiles, ctx.k); tile_i.assign(tiles, s); tile_j.assign(tiles, r);
                    run_hook(cpar, tiles, [&](uint32_t x) {
                        const uint32_t j0 = t + 1 + static_cast<uint32_t>(static_cast<uint64_t>(ncols) * x / tiles);
                        const uint32_t j1 = t + 1 + static_cast<uint32_t>(static_cast<uint64_t>(ncols) * (x + 1) / tiles);
                        uint32_t bv = ctx.k, bi = s, bj = r;
                        for (uint32_t j = j0; j < j1 && bv != 0; ++j)
                            for (uint32_t i = t; i < s; ++i) {
                                const Elt a = at(i, j);
                                if (a == 0) continue;
                                const uint32_t v = ctx.valuation(a);
                                if (v < bv) { bv = v; bi = i; bj = j; if (v == 0) break; }
                            }
                        tile_v[x] = bv; tile_i[x] = bi; tile_j[x] = bj;
                    });
                    for (uint32_t x = 0; x < tiles; ++x)
                        if (tile_v[x] < pv) { pv = tile_v[x]; pr = tile_i[x]; pc = tile_j[x]; }
                }
            }
        } else
        for (uint32_t j = t; j < r; ++j) {                 // leftmost column first ...
            for (uint32_t i = t; i < s; ++i) {             // ... topmost row; strict < keeps the first
                const Elt a = at(i, j);
                if (a == 0) continue;
                const uint32_t v = ctx.valuation(a);
                if (v < pv) { pv = v; pr = i; pc = j; }
            }
        }
        if (pr == s) break;                                // the trailing submatrix is zero
        if (pr != t) {                                     // row swap, mirrored on Gamma and U
            for (uint32_t j = 0; j < r; ++j) std::swap(at(t, j), at(pr, j));
            for (uint32_t i = 0; i < m; ++i) std::swap(Gm.at(i, t), Gm.at(i, pr));
            for (uint32_t j = 0; j < s; ++j) std::swap(out.U.at(t, j), out.U.at(pr, j));
        }
        if (pc != t)                                       // column swap: relations only
            for (uint32_t i = 0; i < s; ++i) std::swap(at(i, t), at(i, pc));
        const Elt d = ctx.ell_pow(pv);
        const Elt uinv = ctx.inv(ctx.unit_part(at(t, t)));
        if (uinv != 1) {                                   // scale the pivot's COLUMN (a relation op)
            const Elt uinvp = ctx.mul_prep(uinv);
            for (uint32_t i = t; i < s; ++i) at(i, t) = ctx.mul_prepped(at(i, t), uinvp);
        }
        ENGINE_DEBUG_CHECK(at(t, t) == d);
        if (interchange) {
            // (slot-23 G2-B) the same column operations, i-outer / j-inner (row-major access).
            // Column t is not written by them, and row t's entries at j > t are read before
            // any is changed (the quotients are taken first), so every (i, j) receives the
            // same single update with the same operands: bit-identical.
            qcols.clear();
            qprep.clear();
            for (uint32_t j = t + 1; j < r; ++j) {
                if (at(t, j) == 0) continue;
                qcols.push_back(j);
                qprep.push_back(ctx.mul_prep(ctx.divide_exact(at(t, j), d)));
            }
            // (S24 MT-1) rows i are disjoint targets; each reads only its own column-t entry (never written here)
            // and the quotients, so a row split gives every entry the same single update
            const uint32_t rtiles = qcols.empty() ? 0u
                : cyc_tiles(cpar, s - t, static_cast<size_t>(s - t) * qcols.size(), 64, size_t(1) << 17, 16);
            auto row_ops = [&](uint32_t i0, uint32_t i1) {
                for (uint32_t i = i0; i < i1; ++i) {
                    const Elt a = at(i, t);
                    if (a == 0) continue;
                    Elt* rowp = &Rm[static_cast<size_t>(i) * r];
                    for (size_t u = 0; u < qcols.size(); ++u) {
                        Elt& e = rowp[qcols[u]];
                        e = ctx.sub(e, ctx.mul_prepped(a, qprep[u]));
                    }
                }
            };
            if (rtiles > 0) {
                module_mt_count(ModuleMtEvent::CycRows);
                const uint32_t ns_ = s - t;
                run_hook(cpar, rtiles, [&](uint32_t x) {
                    row_ops(t + static_cast<uint32_t>(static_cast<uint64_t>(ns_) * x / rtiles),
                            t + static_cast<uint32_t>(static_cast<uint64_t>(ns_) * (x + 1) / rtiles));
                });
            } else if (!qcols.empty())
                row_ops(t, s);
            for (uint32_t j : qcols) { (void)j; ENGINE_DEBUG_CHECK(at(t, j) == 0); }
        } else
        for (uint32_t j = t + 1; j < r; ++j) {             // clear the pivot's ROW (column ops)
            if (at(t, j) == 0) continue;
            const Elt qp = ctx.mul_prep(ctx.divide_exact(at(t, j), d));
            for (uint32_t i = t; i < s; ++i)
                if (at(i, t) != 0) at(i, j) = ctx.sub(at(i, j), ctx.mul_prepped(at(i, t), qp));
            ENGINE_DEBUG_CHECK(at(t, j) == 0);
        }
        if (cpar != nullptr) {
            // (S24 MT-1) the same column clear, split: the quotients and the zeroing first (serially, in i order;
            // no step below reads column t of the relations), then the U rows i (disjoint targets of the read-only
            // row t) and the Gamma rows (each row's column t receives the serial sequence over i, in i order;
            // columns i > t are only read) over row tiles
            clear_list.clear();
            for (uint32_t i = t + 1; i < s; ++i) {
                if (at(i, t) == 0) continue;
                clear_list.emplace_back(i, ctx.mul_prep(ctx.divide_exact(at(i, t), d)));
                at(i, t) = 0;
            }
            const uint32_t nl = static_cast<uint32_t>(clear_list.size());
            const uint32_t ut = cyc_tiles(cpar, nl, static_cast<size_t>(nl) * s, 64, size_t(1) << 17, 16);
            auto u_ops = [&](uint32_t x0, uint32_t x1) {
                for (uint32_t x = x0; x < x1; ++x) {
                    const uint32_t i = clear_list[x].first;
                    const Elt qp = clear_list[x].second;
                    for (uint32_t j = 0; j < s; ++j)
                        if (out.U.at(t, j) != 0) out.U.at(i, j) = ctx.sub(out.U.at(i, j), ctx.mul_prepped(out.U.at(t, j), qp));
                }
            };
            if (ut > 0) {
                module_mt_count(ModuleMtEvent::CycRows);
                run_hook(cpar, ut, [&](uint32_t y) {
                    u_ops(static_cast<uint32_t>(static_cast<uint64_t>(nl) * y / ut), static_cast<uint32_t>(static_cast<uint64_t>(nl) * (y + 1) / ut));
                });
            } else {
                u_ops(0, nl);
            }
            const uint32_t gt = nl == 0 ? 0u : cyc_tiles(cpar, m, static_cast<size_t>(nl) * m, 64, size_t(1) << 17, 16);
            auto g_ops = [&](uint32_t r0, uint32_t r1) {
                for (uint32_t row = r0; row < r1; ++row) {
                    Elt acc = Gm.at(row, t);
                    for (uint32_t x = 0; x < nl; ++x) {
                        const Elt g = Gm.at(row, clear_list[x].first);
                        if (g != 0) acc = ctx.add(acc, ctx.mul_prepped(g, clear_list[x].second));
                    }
                    Gm.at(row, t) = acc;
                }
            };
            if (gt > 0) {
                module_mt_count(ModuleMtEvent::CycRows);
                run_hook(cpar, gt, [&](uint32_t y) {
                    g_ops(static_cast<uint32_t>(static_cast<uint64_t>(m) * y / gt), static_cast<uint32_t>(static_cast<uint64_t>(m) * (y + 1) / gt));
                });
            } else if (nl > 0) {
                g_ops(0, m);
            }
        } else
        for (uint32_t i = t + 1; i < s; ++i) {             // clear the pivot's COLUMN (row ops, mirrored)
            if (at(i, t) == 0) continue;
            const Elt qp = ctx.mul_prep(ctx.divide_exact(at(i, t), d));
            at(i, t) = 0;                                  // row t is zero off the pivot now
            for (uint32_t j = 0; j < s; ++j)               // U: row_i -= q row_t
                if (out.U.at(t, j) != 0) out.U.at(i, j) = ctx.sub(out.U.at(i, j), ctx.mul_prepped(out.U.at(t, j), qp));
            for (uint32_t row = 0; row < m; ++row)         // Gamma: gamma_t += q gamma_i
                if (Gm.at(row, i) != 0) Gm.at(row, t) = ctx.add(Gm.at(row, t), ctx.mul_prepped(Gm.at(row, i), qp));
        }
        sigma.push_back(pv);
    }
    if (lean) std::vector<Elt>().swap(Rm);
    // (c4) read-off, then the stable descending sort.
    std::vector<std::pair<uint32_t, uint32_t>> summ;       // (e, position)
    for (uint32_t i = 0; i < s; ++i) {
        if (i < sigma.size()) {
            if (sigma[i] >= 1) summ.emplace_back(sigma[i], i);
        } else {
            summ.emplace_back(ctx.k, i);
        }
    }
    std::stable_sort(summ.begin(), summ.end(),
                     [](const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b) {
                         return a.first > b.first;
                     });
    out.basis = DenseBuf<Elt>(m, static_cast<uint32_t>(summ.size()));
    uint32_t sum = 0;
    for (uint32_t c = 0; c < summ.size(); ++c) {
        out.exponents.push_back(summ[c].first);
        out.position.push_back(summ[c].second);
        sum += summ[c].first;
        for (uint32_t i = 0; i < m; ++i) out.basis.at(i, c) = Gm.at(i, summ[c].second);
    }
    if (!lean) len_rl = (nr == 0) ? 0u : length<Elt>(ctx, image_module<Elt>(ctx, Rl));
    ENGINE_CHECK_MSG(sum + len_rl == length<Elt>(ctx, Nh),
                     "elim::cyclic_decomposition: sum e_i != length(N) - length(colspan Rl) ([E] sec:ring (c4))");
    return out;
}

template<class Elt>
std::vector<std::vector<Elt>> cyclic_coordinates(const ModCtx<Elt>& ctx, const CyclicDecomposition<Elt>& d,
                                                 DenseView<const Elt> Y) {
    ENGINE_CHECK_MSG(!ctx.is_field(), ENGINE_COMPOSITE_ONLY("elim::cyclic_coordinates"));
    const uint32_t ny = Y.cols(), s = d.Gamma.cols, t = d.rows();
    ENGINE_CHECK_MSG(Y.rows() == d.Gamma.rows, "elim::cyclic_coordinates: row count mismatch");
    std::vector<std::vector<Elt>> out(ny, std::vector<Elt>(t, Elt(0)));
    if (ny == 0 || t == 0) {
        if (ny > 0 && s > 0) (void)solve_multi_rhs_module<Elt>(ctx, cview(d.Gamma), Y);   // membership still checked
        return out;
    }
    const std::vector<std::vector<Elt>> x = solve_multi_rhs_module<Elt>(ctx, cview(d.Gamma), Y);
    for (uint32_t c = 0; c < ny; ++c) {
        for (uint32_t i = 0; i < t; ++i) {
            const uint32_t row = d.position[i];
            Elt acc = 0;
            for (uint32_t j = 0; j < s; ++j)
                if (d.U.at(row, j) != 0 && x[c][j] != 0) acc = ctx.add(acc, ctx.mul(d.U.at(row, j), x[c][j]));
            const uint32_t e = d.exponents[i];
            out[c][i] = (e >= ctx.k) ? acc : static_cast<Elt>(acc % ctx.ell_pow(e));
        }
    }
    return out;
}

#undef ENGINE_COMPOSITE_ONLY

#define ENGINE_ELIM_INSTANTIATE(Elt)                                                             \
    template RrefResult rref<Elt>(const ModCtx<Elt>&, DenseView<Elt>);                            \
    template RrefResult column_rank_profile<Elt>(const ModCtx<Elt>&, DenseView<Elt>);             \
    template RrefResult rref_with_transform<Elt>(const ModCtx<Elt>&, DenseView<const Elt>,        \
                                                 DenseView<Elt>, DenseView<Elt>);                 \
    template DenseBuf<Elt> kernel_basis<Elt>(const ModCtx<Elt>&, DenseView<const Elt>,            \
                                             const std::vector<uint32_t>&);                       \
    template DenseBuf<Elt> image_basis<Elt>(DenseView<const Elt>, const std::vector<uint32_t>&);  \
    template bool inverse<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, DenseView<Elt>);         \
    template std::vector<std::vector<Elt>> solve_multi_rhs<Elt>(const ModCtx<Elt>&,               \
                                                                DenseView<const Elt>,             \
                                                                DenseView<const Elt>);            \
    template std::vector<uint32_t> complement_basis<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, \
                                                         DenseView<const Elt>, DenseBuf<Elt>*);   \
    template DenseBuf<Elt> eliminate_against<Elt>(const ModCtx<Elt>&, DenseView<Elt>,             \
                                                  DenseView<const Elt>,                           \
                                                  const std::vector<uint32_t>&, uint32_t);        \
    template void sub_product<Elt>(const ModCtx<Elt>&, DenseView<Elt>, DenseView<const Elt>,      \
                                   DenseView<const Elt>);                                         \
    template HowellResult<Elt> howell<Elt>(const ModCtx<Elt>&, DenseView<const Elt>);             \
    template HowellResult<Elt> kernel_module<Elt>(const ModCtx<Elt>&, DenseView<const Elt>);      \
    template HowellResult<Elt> image_module<Elt>(const ModCtx<Elt>&, DenseView<const Elt>);       \
    template std::vector<std::vector<Elt>> solve_multi_rhs_module<Elt>(const ModCtx<Elt>&,        \
                                                                       DenseView<const Elt>,      \
                                                                       DenseView<const Elt>);     \
    template void gemm_rows_hook<Elt>(const ParallelFor*, const ModCtx<Elt>&, DenseView<const Elt>,   \
                                      DenseView<const Elt>, DenseView<Elt>);                       \
    template ReadoffStatus solve_unit_pivot_readoff<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, \
                                                         const std::vector<uint32_t>&,            \
                                                         DenseView<const Elt>, DenseBuf<Elt>&,    \
                                                         uint32_t);                               \
    template ReadoffStatus solve_unit_pivot_readoff_product<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, \
                                                                 const std::vector<uint32_t>&,            \
                                                                 DenseView<const Elt>, DenseView<const Elt>, \
                                                                 DenseBuf<Elt>&, uint32_t, uint32_t);       \
    template ReadoffStatus unit_pivot_residual_product<Elt>(const ModCtx<Elt>&, DenseView<const Elt>,           \
                                                            const std::vector<uint32_t>&, DenseView<const Elt>, \
                                                            DenseView<const Elt>, uint32_t, uint32_t);         \
    template bool product_is_zero<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, DenseView<const Elt>, bool); \
    template uint32_t length<Elt>(const ModCtx<Elt>&, DenseView<const Elt>);                      \
    template uint32_t length<Elt>(const ModCtx<Elt>&, const HowellResult<Elt>&);                  \
    template std::vector<uint32_t> invariant_factors<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, \
                                                          DenseView<const Elt>);                  \
    template std::vector<uint32_t> invariant_factors<Elt>(const ModCtx<Elt>&, DenseView<const Elt>);   \
    template CyclicDecomposition<Elt> cyclic_decomposition<Elt>(const ModCtx<Elt>&, DenseView<const Elt>, \
                                                                DenseView<const Elt>);                    \
    template std::vector<std::vector<Elt>> cyclic_coordinates<Elt>(const ModCtx<Elt>&,                    \
                                                                   const CyclicDecomposition<Elt>&,       \
                                                                   DenseView<const Elt>);

ENGINE_ELIM_INSTANTIATE(uint32_t)
ENGINE_ELIM_INSTANTIATE(uint64_t)
#undef ENGINE_ELIM_INSTANTIATE

// (hecke-extensions slot 11, WP-18a; ENGINE_API.md Sec. 13.12.3) the u128
// element type is COMPOSITE-ONLY: the Sec. 13.6 entry points alone are
// instantiated for it -- the bodies above, unchanged, on ModCtx<u128>'s
// mul_prep / mul_prepped pair (ModCtxU128.h).  No field entry point (rref,
// kernel_basis, inverse, solve_multi_rhs, complement_basis, ...) exists for
// u128, so a field use of it does not link.
template HowellResult<u128> howell<u128>(const ModCtx<u128>&, DenseView<const u128>);
template HowellResult<u128> kernel_module<u128>(const ModCtx<u128>&, DenseView<const u128>);
template HowellResult<u128> image_module<u128>(const ModCtx<u128>&, DenseView<const u128>);
template std::vector<std::vector<u128>> solve_multi_rhs_module<u128>(const ModCtx<u128>&, DenseView<const u128>,
                                                                     DenseView<const u128>);
template uint32_t length<u128>(const ModCtx<u128>&, DenseView<const u128>);
template uint32_t length<u128>(const ModCtx<u128>&, const HowellResult<u128>&);
template std::vector<uint32_t> invariant_factors<u128>(const ModCtx<u128>&, DenseView<const u128>, DenseView<const u128>);
template std::vector<uint32_t> invariant_factors<u128>(const ModCtx<u128>&, DenseView<const u128>);
template CyclicDecomposition<u128> cyclic_decomposition<u128>(const ModCtx<u128>&, DenseView<const u128>,
                                                              DenseView<const u128>);
template std::vector<std::vector<u128>> cyclic_coordinates<u128>(const ModCtx<u128>&, const CyclicDecomposition<u128>&,
                                                                 DenseView<const u128>);

} // namespace elim
} // namespace engine
