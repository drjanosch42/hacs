// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, package Z3K) Smith / Howell / kernel_module over Z/3^k (dense blocked and small-modulus variants) -- see Ring3k.h.

#include "Ring3k.h"

#include "SmithPar.h"
#include "TritMatrix.h"
#include "../ModArithmetic/EngineCheck.h"
#include "../ModArithmetic/Kernels.h"
#include "../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <utility>

namespace engine {
namespace ring3k {

namespace {

std::atomic<uint32_t> g_panel{64}, g_reduce_block{64};
std::atomic<uint64_t> g_lu_calls{0}, g_lu_pivots{0}, g_closure_rows{0}, g_smith_levels{0}, g_gemm_calls{0};

using U32 = uint32_t;

uint32_t pow3(uint32_t e) {
    uint32_t p = 1;
    for (uint32_t i = 0; i < e; ++i) p *= 3u;
    return p;
}

// The engine context of Z/3^k (k >= 2) or F_3 (k = 1): the modulus-only layer (gemm, mul, sub) is exact on both.
ModCtx<U32> ctx_for(uint32_t k) { return ModCtx<U32>::chain(3u, k); }

// The 3-adic valuation of a canonical x in [0, 3^k), capped at k (valuation(0) = k).
inline uint32_t val3(U32 x, uint32_t k) {
    if (x == 0) return k;
    uint32_t v = 0;
    while (x % 3u == 0) { x /= 3u; ++v; }
    return v;
}

// The inverse mod 3^k of a unit u (u mod 3 != 0), canonical; the engine's unit inverse (ModCtx::inv, extended Euclid;
// S24 CL-G3, CODEAUDIT C-04: it replaced the Mod3k<1> Montgomery power -- the inverse of a unit is unique).
struct UnitInverter {
    ModCtx<U32> R;
    explicit UnitInverter(uint32_t k) : R(ctx_for(k)) {}
    U32 inv(U32 u) const { return R.inv(u); }
};

// Compact row-major working matrix over Z/3^k.
struct Work {
    uint32_t rows = 0, cols = 0;
    std::vector<U32> a;
    U32*       row(uint32_t i)       { return a.data() + static_cast<size_t>(i) * cols; }
    const U32* row(uint32_t i) const { return a.data() + static_cast<size_t>(i) * cols; }
    U32&       at(uint32_t i, uint32_t j)       { return a[static_cast<size_t>(i) * cols + j]; }
};

Work to_work(const DenseBuf<U32>& A, uint32_t k) {
    const U32 M = pow3(k);
    Work W;
    W.rows = A.rows;
    W.cols = A.cols;
    W.a.resize(static_cast<size_t>(A.rows) * A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) {
            const U32 x = A.at(i, j);
            ENGINE_CHECK_MSG(x < M, "ring3k: input entry not canonical mod 3^k");
            W.a[static_cast<size_t>(i) * A.cols + j] = x;
        }
    return W;
}

// r[0, n) -= f * p[0, n) mod ctx.p (canonical in, canonical out).  Shoup form of f (ctx.p <= 2^31 always here).
inline void axpy_sub(const ModCtx<U32>& ctx, U32* r, const U32* p, U32 f, size_t n) {
    if (f == 0) return;
    const U32 fs = ctx.shoup_precompute(f);
    for (size_t j = 0; j < n; ++j)
        if (p[j] != 0) r[j] = ctx.sub(r[j], ctx.shoup_mul(p[j], f, fs));
}

// A row-major parent matrix (rows x ld, contiguous); windows of it are DenseViews of an external DenseBuf over it.
struct Parent {
    U32* p;
    uint32_t rows, ld;
};
// Rows [cr0, cr0 + mrows) of X, columns [c0, c0 + ncols), -= L (rows [lr0, lr0 + mrows), columns [0, np) of Lp) *
// (rows [ur0, ur0 + np), columns [c0, c0 + ncols) of X), all over ctx: one kernels::gemm (C and U never overlap).
void gemm_sub(const ModCtx<U32>& ctx, const Parent& X, uint32_t cr0, uint32_t mrows, uint32_t ur0, uint32_t c0, uint32_t ncols,
              const Parent& Lp, uint32_t lr0, uint32_t np) {
    if (mrows == 0 || ncols == 0 || np == 0) return;
    DenseBuf<U32> Xb(X.p, static_cast<size_t>(X.rows) * X.ld, X.rows, X.ld, Layout::RowMajor, X.ld);
    DenseBuf<U32> Lb(Lp.p, static_cast<size_t>(Lp.rows) * Lp.ld, Lp.rows, Lp.ld, Layout::RowMajor, Lp.ld);
    const DenseView<U32> C = view(Xb).sub(cr0, mrows, c0, ncols);
    const DenseView<const U32> U = cview(Xb).sub(ur0, np, c0, ncols);
    const DenseView<const U32> L = cview(Lb).sub(lr0, mrows, 0, np);
    kernels::gemm<U32>(ctx, L, U, C, kernels::GemmOp::Sub);
    g_gemm_calls.fetch_add(1, std::memory_order_relaxed);
}

// (slot-24 PAR-2K) gemm_sub split into disjoint tiles of C (row tiles; row x column tiles when the rows are few), one
// kernels::gemm per tile (its scratch is thread-local).  Every C entry is the same canonical residue as in one call.
void gemm_sub_mt(uint32_t T, const ModCtx<U32>& ctx, const Parent& X, uint32_t cr0, uint32_t mrows, uint32_t ur0, uint32_t c0,
                 uint32_t ncols, const Parent& Lp, uint32_t lr0, uint32_t np) {
    const uint64_t work = static_cast<uint64_t>(mrows) * ncols * np;
    const uint64_t mw = smithpar::min_work();
    if (T <= 1 || mrows == 0 || ncols == 0 || np == 0 || work < 2 * mw) {
        gemm_sub(ctx, X, cr0, mrows, ur0, c0, ncols, Lp, lr0, np);
        return;
    }
    uint64_t want = 4ull * T;
    if (mw > 0 && work / mw < want) want = work / mw;
    if (want < 2) want = 2;
    const uint32_t row_cap = (mrows + 15) / 16;
    const uint32_t rt = static_cast<uint32_t>(want < row_cap ? want : row_cap);
    uint32_t ct = static_cast<uint32_t>((want + rt - 1) / rt);
    const uint32_t col_cap = (ncols + 63) / 64;
    if (ct > col_cap) ct = col_cap;
    if (ct == 0) ct = 1;
    smithpar::parallel_for(rt * ct, [&](uint32_t t) {
        const uint32_t ri = t / ct, ci = t % ct;
        const uint32_t a0 = smithpar::split_at(mrows, rt, ri, 8), a1 = smithpar::split_at(mrows, rt, ri + 1, 8);
        const uint32_t b0 = smithpar::split_at(ncols, ct, ci, 64), b1 = smithpar::split_at(ncols, ct, ci + 1, 64);
        if (a1 > a0 && b1 > b0) gemm_sub(ctx, X, cr0 + a0, a1 - a0, ur0, c0 + b0, b1 - b0, Lp, lr0 + a0, np);
    });
}

// The panel TRSM over the columns [0, w) (relative), split into column tiles: body(a, b) solves the columns [a, b).
template<class F>
void trsm_mt(uint32_t T, uint32_t w, uint32_t np, const F& body) {
    const uint64_t work = static_cast<uint64_t>(np) * np / 2 * w;
    const uint64_t mw = smithpar::min_work();
    uint64_t want = 2ull * T;
    if (mw > 0 && work / mw < want) want = work / mw;
    const uint32_t cap = (w + 63) / 64;
    const uint32_t tiles = static_cast<uint32_t>(want < cap ? want : cap);
    if (T <= 1 || np < 2 || work < 2 * mw || tiles < 2) { body(0, w); return; }
    smithpar::parallel_for(tiles, [&](uint32_t t) {
        const uint32_t a = smithpar::split_at(w, tiles, t, 64), b = smithpar::split_at(w, tiles, t + 1, 64);
        if (b > a) body(a, b);
    });
}

enum class PivotRule : uint8_t { FirstUnit, MinValuation };
struct EchelonInfo {
    uint32_t rank = 0;
    std::vector<uint32_t> pivots, pivot_v;
};

void swap_rows(Work& A, uint32_t i, uint32_t j) {
    if (i == j) return;
    std::swap_ranges(A.row(i), A.row(i) + A.cols, A.row(j));
}

// Blocked right-looking LU over Z/3^k (Ring2k.h's echelon_blocked, with "unit" for "odd").  Rows [0, rank) are the
// echelon rows on return (row t leads at pivots[t] with valuation pivot_v[t], exactly zero in every earlier pivot
// column); rows [rank, m) are zero in every pivot column and (FirstUnit) 0 mod 3 / (MinValuation) zero elsewhere.
// Row operations only, all unimodular.  Pivots are sought in [0, col_limit); under FirstUnit a column without a unit
// is an ENGINE_CHECK failure (the caller puts the F_3 rank profile first and passes col_limit = the F_3 rank).
EchelonInfo echelon_blocked(Work& A, uint32_t k, PivotRule rule, uint32_t panel, uint32_t col_limit = 0xffffffffu) {
    EchelonInfo info;
    const uint32_t m = A.rows, n = A.cols;
    if (m == 0 || n == 0) return info;
    if (panel == 0) panel = 1;
    const ModCtx<U32> ctx = ctx_for(k);
    const UnitInverter inv(k);
    const uint32_t climit = col_limit < n ? col_limit : n;
    std::vector<U32> L(static_cast<size_t>(m) * panel, 0u);
    uint32_t r = 0, c = 0;
    uint64_t piv_count = 0;
    // (slot-24 PAR-2K) T > 1: the threaded route -- the left-looking ("lazy") panel below (Ring2k.h's), trsm_mt, gemm_sub_mt.
    // Every entry gets the serial arithmetic (canonical residues: one answer), every pivot rule reads the same values.
    const uint32_t T = smithpar::threads();
    std::vector<uint32_t> lvl, pj;
    std::vector<U32> pd, pui;   // the pivots' 3^v and unit inverses
    while (c < climit && r < m) {
        const uint32_t c1 = c + panel < climit ? c + panel : climit;
        uint32_t np = 0;
        if (T <= 1) {
            for (uint32_t j = c; j < c1 && r + np < m; ++j) {
                const uint32_t prow = r + np;
                uint32_t best = m, bv = k;
                for (uint32_t i = prow; i < m; ++i) {
                    const U32 x = A.at(i, j);
                    if (x == 0) continue;
                    if (rule == PivotRule::FirstUnit) {
                        if (x % 3u != 0) { best = i; bv = 0; break; }
                    } else {
                        const uint32_t v = val3(x, k);
                        if (v < bv) { bv = v; best = i; if (v == 0) break; }
                    }
                }
                if (best == m) {
                    ENGINE_CHECK_MSG(rule != PivotRule::FirstUnit,
                                     "ring3k::echelon_blocked: FirstUnit met a column without a unit (put the F_3 rank profile "
                                     "first and pass col_limit = the F_3 rank)");
                    continue;
                }
                if (best != prow) {
                    swap_rows(A, best, prow);
                    std::swap_ranges(L.data() + static_cast<size_t>(best) * panel, L.data() + static_cast<size_t>(best + 1) * panel,
                                     L.data() + static_cast<size_t>(prow) * panel);
                }
                const U32* P = A.row(prow);
                const U32 d = pow3(bv);
                const U32 ui = inv.inv(P[j] / d);   // P[j] = 3^bv u, u a unit
                for (uint32_t i = prow + 1; i < m; ++i) {
                    U32* R = A.row(i);
                    U32& slot = L[static_cast<size_t>(i) * panel + np];
                    if (R[j] == 0) { slot = 0; continue; }
                    // v(R[j]) >= bv (minimal valuation, or bv = 0): f * 3^bv u == R[j] mod 3^k.
                    ENGINE_DEBUG_CHECK(R[j] % d == 0);
                    const U32 f = ctx.mul(R[j] / d, ui);
                    slot = f;
                    axpy_sub(ctx, R + j, P + j, f, c1 - j);
                    ENGINE_DEBUG_CHECK(R[j] == 0);
                }
                info.pivots.push_back(j);
                info.pivot_v.push_back(bv);
                ++np;
            }
        } else {
            // THE LAZY PANEL (Ring2k.h's echelon_blocked): each row receives the panel pivots in order, exactly the serial
            // steps, when the pivot search reads it or, for the rest, at the end of the panel in one parallel region.
            lvl.assign(m - r, 0);
            pj.clear();
            pd.clear();
            pui.clear();
            auto bring = [&](uint32_t i, uint32_t to) {
                uint32_t& li = lvl[i - r];
                if (li >= to) return;
                U32* R = A.row(i);
                U32* Li = L.data() + static_cast<size_t>(i) * panel;
                for (uint32_t t = li; t < to; ++t) {
                    const uint32_t j = pj[t];
                    if (R[j] == 0) { Li[t] = 0; continue; }
                    ENGINE_DEBUG_CHECK(R[j] % pd[t] == 0);
                    const U32 f = ctx.mul(R[j] / pd[t], pui[t]);
                    Li[t] = f;
                    axpy_sub(ctx, R + j, A.row(r + t) + j, f, c1 - j);
                    ENGINE_DEBUG_CHECK(R[j] == 0);
                }
                li = to;
            };
            for (uint32_t j = c; j < c1 && r + np < m; ++j) {
                const uint32_t prow = r + np;
                uint32_t best = m, bv = k;
                for (uint32_t i = prow; i < m; ++i) {
                    bring(i, np);
                    const U32 x = A.at(i, j);
                    if (x == 0) continue;
                    if (rule == PivotRule::FirstUnit) {
                        if (x % 3u != 0) { best = i; bv = 0; break; }
                    } else {
                        const uint32_t v = val3(x, k);
                        if (v < bv) { bv = v; best = i; if (v == 0) break; }
                    }
                }
                if (best == m) {
                    ENGINE_CHECK_MSG(rule != PivotRule::FirstUnit,
                                     "ring3k::echelon_blocked: FirstUnit met a column without a unit (put the F_3 rank profile "
                                     "first and pass col_limit = the F_3 rank)");
                    continue;
                }
                if (best != prow) {
                    swap_rows(A, best, prow);
                    std::swap_ranges(L.data() + static_cast<size_t>(best) * panel, L.data() + static_cast<size_t>(best + 1) * panel,
                                     L.data() + static_cast<size_t>(prow) * panel);
                    std::swap(lvl[best - r], lvl[prow - r]);
                }
                const U32 d = pow3(bv);
                pj.push_back(j);
                pd.push_back(d);
                pui.push_back(inv.inv(A.at(prow, j) / d));
                info.pivots.push_back(j);
                info.pivot_v.push_back(bv);
                ++np;
            }
            if (r + np < m) {
                smithpar::count_panel_split();
                const uint32_t rows = m - (r + np), r1 = r + np, npp = np;
                smithpar::for_rows(rows, static_cast<uint64_t>(rows) * npp * (c1 - c) / 2,
                                   [&](uint32_t i0, uint32_t i1) { for (uint32_t i = i0; i < i1; ++i) bring(r1 + i, npp); });
            }
        }
        if (np > 0 && c1 < n) {
            // Unit-lower TRSM on the panel's pivot rows, then one GEMM on the rows below.
            auto trsm = [&](uint32_t a, uint32_t b) {
                for (uint32_t t = 1; t < np; ++t) {
                    const U32* lt = L.data() + static_cast<size_t>(r + t) * panel;
                    for (uint32_t t2 = 0; t2 < t; ++t2)
                        if (lt[t2] != 0) axpy_sub(ctx, A.row(r + t) + a, A.row(r + t2) + a, lt[t2], b - a);
                }
            };
            if (T <= 1) {
                trsm(c1, n);
                if (r + np < m)
                    gemm_sub(ctx, Parent{A.a.data(), m, n}, r + np, m - (r + np), r, c1, n - c1, Parent{L.data(), m, panel}, r + np, np);
            } else {
                trsm_mt(T, n - c1, np, [&](uint32_t a, uint32_t b) { trsm(c1 + a, c1 + b); });
                if (r + np < m)
                    gemm_sub_mt(T, ctx, Parent{A.a.data(), m, n}, r + np, m - (r + np), r, c1, n - c1, Parent{L.data(), m, panel},
                                r + np, np);
            }
        }
        piv_count += np;
        r += np;
        c = c1;
    }
    info.rank = r;
    g_lu_calls.fetch_add(1, std::memory_order_relaxed);
    g_lu_pivots.fetch_add(piv_count, std::memory_order_relaxed);
    return info;
}

// The Sec. 13.5 pass (active list, minimal valuation / topmost, scale to 3^v, exact elimination, closure rows at the
// bottom), then reduce above in blocks.  On A itself (howell_naive) or on the echelon rows (howell).
HowellForm howell_pass(Work A, uint32_t k, uint32_t reduce_block) {
    const uint32_t m = A.rows, n = A.cols;
    const ModCtx<U32> ctx = ctx_for(k);
    const UnitInverter inv(k);
    HowellForm out;
    out.k = k;
    std::vector<std::vector<U32>> closure;
    auto row = [&](uint32_t id) -> U32* { return id < m ? A.row(id) : closure[id - m].data(); };
    std::vector<uint32_t> active(m);
    for (uint32_t i = 0; i < m; ++i) active[i] = i;
    std::vector<uint32_t> prow_id;
    uint64_t closures = 0;
    for (uint32_t c = 0; c < n && !active.empty(); ++c) {
        size_t best = active.size();
        uint32_t bv = k;
        for (size_t t = 0; t < active.size(); ++t) {
            const U32 x = row(active[t])[c];
            if (x == 0) continue;
            const uint32_t v = val3(x, k);
            if (v < bv) { bv = v; best = t; if (v == 0) break; }
        }
        if (best == active.size()) continue;
        const uint32_t pid = active[best];
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(best));
        U32* P = row(pid);
        const U32 d = pow3(bv);
        const U32 ui = inv.inv(P[c] / d);
        if (ui != 1) {
            const U32 us = ctx.shoup_precompute(ui);
            for (uint32_t j = c; j < n; ++j) if (P[j] != 0) P[j] = ctx.shoup_mul(P[j], ui, us);
        }
        ENGINE_CHECK_MSG(P[c] == d, "ring3k::howell_pass: the scaled pivot is not 3^v");
        for (uint32_t id : active) {
            U32* R = row(id);
            if (R[c] == 0) continue;
            ENGINE_DEBUG_CHECK(R[c] % d == 0);
            axpy_sub(ctx, R + c, P + c, R[c] / d, n - c);
            ENGINE_DEBUG_CHECK(R[c] == 0);
        }
        if (bv > 0) {
            ENGINE_CHECK_MSG(closure.size() < n, "ring3k::howell_pass: more closure rows than columns");
            closure.emplace_back(n, 0u);
            P = row(pid);   // emplace_back may move the closure rows (a closure pivot row lives there)
            U32* Z = closure.back().data();
            const U32 f = pow3(k - bv);   // 3^(k - v) * P: zero in column c (3^k = 0)
            for (uint32_t j = c + 1; j < n; ++j) Z[j] = ctx.mul(P[j], f);
            active.push_back(m + static_cast<uint32_t>(closure.size() - 1));
            ++closures;
        }
        prow_id.push_back(pid);
        out.pivots.push_back(c);
        out.pivot_v.push_back(bv);
    }
    const uint32_t h = static_cast<uint32_t>(prow_id.size());
    Work H;
    H.rows = h;
    H.cols = n;
    H.a.resize(static_cast<size_t>(h) * n);
    for (uint32_t i = 0; i < h; ++i) std::copy(row(prow_id[i]), row(prow_id[i]) + n, H.row(i));
    std::vector<U32>().swap(A.a);
    std::vector<std::vector<U32>>().swap(closure);
    // Reduce above, pivots left to right, in blocks [t0, t1): H[i][p_t] = q 3^v_t + r, r in [0, 3^v_t); row i -= q row t.
    if (reduce_block == 0) reduce_block = 1;
    std::vector<U32> Q, vals;
    for (uint32_t t0 = 0; t0 < h; t0 += reduce_block) {
        const uint32_t t1 = t0 + reduce_block < h ? t0 + reduce_block : h;
        const uint32_t b = t1 - t0;
        const uint32_t c0 = out.pivots[t0];
        if (t0 > 0) {
            // Rows [0, t0): the floor quotients against the block, sequentially on the block's pivot columns only,
            // then one GEMM over the columns >= c0.
            Q.assign(static_cast<size_t>(t0) * b, 0u);
            vals.assign(b, 0u);
            bool nonzero = false;
            for (uint32_t i = 0; i < t0; ++i) {
                const U32* hi = H.row(i);
                for (uint32_t t = 0; t < b; ++t) vals[t] = hi[out.pivots[t0 + t]];
                U32* q = Q.data() + static_cast<size_t>(i) * b;
                for (uint32_t t = 0; t < b; ++t) {
                    const U32 qt = vals[t] / pow3(out.pivot_v[t0 + t]);
                    q[t] = qt;
                    if (qt == 0) continue;
                    nonzero = true;
                    const U32* ht = H.row(t0 + t);
                    for (uint32_t t3 = t + 1; t3 < b; ++t3) vals[t3] = ctx.sub(vals[t3], ctx.mul(qt, ht[out.pivots[t0 + t3]]));
                }
            }
            if (nonzero) gemm_sub(ctx, Parent{H.a.data(), h, n}, 0, t0, t0, c0, n - c0, Parent{Q.data(), t0, b}, 0, b);
        }
        // The triangle inside the block.
        for (uint32_t t = t0 + 1; t < t1; ++t) {
            const uint32_t ct = out.pivots[t];
            const U32 dt = pow3(out.pivot_v[t]);
            for (uint32_t i = t0; i < t; ++i) {
                U32* hi = H.row(i);
                const U32 q = hi[ct] / dt;
                if (q != 0) axpy_sub(ctx, hi + ct, H.row(t) + ct, q, n - ct);
            }
        }
    }
    out.H = DenseBuf<U32>(h, n);
    for (uint32_t i = 0; i < h; ++i) std::copy(H.row(i), H.row(i) + n, out.H.data() + static_cast<size_t>(i) * out.H.ld);
    g_closure_rows.fetch_add(closures, std::memory_order_relaxed);
    return out;
}

// ---------------------------------------------------------------------------
// The SMALL-MODULUS rows (k <= kSmallK, 3^k <= 243): uint16_t entries, the Sec. 13.5 pass on the input rows themselves
// (no LU first), a row skipped when its entry in the pivot column is zero -- the engine's own sparse-friendly order --
// with each row update one vectorised loop whose modulus is a compile-time constant (x % 3^k by multiply-shift).
// The census coboundaries are sparse: there the dense blocked LU above does more work than this (Z3K_report.md Sec. 2).
// ---------------------------------------------------------------------------
using U16 = uint16_t;

// Not under ThreadSanitizer: GCC's ifunc resolver for target_clones is itself instrumented and runs at relocation time,
// before libtsan is initialised, so __tsan_func_entry dereferences NULL and the process dies before main (KNOWN_ISSUES TSan
// startup crash). Same arithmetic, scalar-dispatch only: no value changes.
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && !defined(__SANITIZE_THREAD__)
#define RING3K_CLONES __attribute__((target_clones("arch=skylake-avx512", "arch=haswell", "default")))
#else
#define RING3K_CLONES
#endif

// r[0, n) = (r - f p) mod M, canonical in / out; M^2 + M < 2^16 so the u16 sum never wraps.  (S24 CL-5, C-06) Each body twice:
// the target_clones dispatch and a plain portable copy (the clones' "default" body), taken under ENGINE_KERNEL=portable.
#define RING3K_SMALL_AXPY_BODY(M)                                                                  \
    for (size_t j = 0; j < n; ++j)                                                                 \
        r[j] = static_cast<U16>(static_cast<U16>(r[j] + static_cast<U16>((M) * (M)) - static_cast<U16>(f * p[j])) % (M));
#define RING3K_SMALL_AXPY(M)                                                                                \
    RING3K_CLONES void axpy_small_##M(U16* r, const U16* p, U16 f, size_t n) { RING3K_SMALL_AXPY_BODY(M) } \
    void axpy_small_portable_##M(U16* r, const U16* p, U16 f, size_t n) { RING3K_SMALL_AXPY_BODY(M) }
RING3K_SMALL_AXPY(3)
RING3K_SMALL_AXPY(9)
RING3K_SMALL_AXPY(27)
RING3K_SMALL_AXPY(81)
RING3K_SMALL_AXPY(243)
#undef RING3K_SMALL_AXPY
#undef RING3K_SMALL_AXPY_BODY

using SmallAxpy = void (*)(U16*, const U16*, U16, size_t);
SmallAxpy small_axpy(uint32_t k) {
    const bool portable = kernels::isa_ceiling() == cpu::IsaPath::Portable;   // (S24 CL-5) ENGINE_KERNEL=portable
    switch (k) {
        case 1: return portable ? &axpy_small_portable_3 : &axpy_small_3;
        case 2: return portable ? &axpy_small_portable_9 : &axpy_small_9;
        case 3: return portable ? &axpy_small_portable_27 : &axpy_small_27;
        case 4: return portable ? &axpy_small_portable_81 : &axpy_small_81;
        case 5: return portable ? &axpy_small_portable_243 : &axpy_small_243;
        default: ENGINE_CHECK_MSG(false, "ring3k: the small-modulus rows need k <= 5"); return nullptr;
    }
}

constexpr uint32_t kSmallK = 5;

// The Sec. 13.5 pass on the m x n row-major store (u16, canonical mod 3^k) -- the engine's howell_rows step by step:
// columns left to right; the active entry of minimal valuation, ties to the topmost; scale to 3^v; eliminate every
// other active row with a non-zero entry there; the closure row 3^(k-v) P at the bottom of the active list.  With
// `reduce` the reduce-above (pivots left to right, floor quotients, rows above in order) follows; without it only the
// pivots and valuations are produced (all a length needs).  Every value is exact, so the output is elim::howell's.
HowellForm howell_small(std::vector<U16> store, uint32_t m, uint32_t n, uint32_t k, bool reduce) {
    const uint32_t M = pow3(k);
    const SmallAxpy axpy = small_axpy(k);
    const UnitInverter inv(k);
    std::vector<uint8_t> vt(M);   // the valuation of every residue (vt[0] = k)
    for (uint32_t x = 0; x < M; ++x) vt[x] = static_cast<uint8_t>(val3(x, k));
    HowellForm out;
    out.k = k;
    std::vector<std::vector<U16>> closure;
    auto row = [&](uint32_t id) -> U16* { return id < m ? store.data() + static_cast<size_t>(id) * n : closure[id - m].data(); };
    std::vector<uint32_t> active(m);
    for (uint32_t i = 0; i < m; ++i) active[i] = i;
    std::vector<uint32_t> prow_id;
    uint64_t closures = 0;
    for (uint32_t c = 0; c < n && !active.empty(); ++c) {
        size_t best = active.size();
        uint32_t bv = k;
        for (size_t t = 0; t < active.size(); ++t) {
            const U16 x = row(active[t])[c];
            if (x == 0) continue;
            const uint32_t v = vt[x];
            if (v < bv) { bv = v; best = t; if (v == 0) break; }
        }
        if (best == active.size()) continue;
        const uint32_t pid = active[best];
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(best));
        U16* P = row(pid);
        const uint32_t d = pow3(bv);
        const uint32_t ui = inv.inv(P[c] / d);
        if (ui != 1)
            for (uint32_t j = c; j < n; ++j) P[j] = static_cast<U16>((static_cast<uint32_t>(P[j]) * ui) % M);
        ENGINE_CHECK_MSG(P[c] == d, "ring3k::howell_small: the scaled pivot is not 3^v");
        for (uint32_t id : active) {
            U16* R = row(id);
            if (R[c] == 0) continue;
            ENGINE_DEBUG_CHECK(R[c] % d == 0);
            axpy(R + c, P + c, static_cast<U16>(R[c] / d), n - c);
            ENGINE_DEBUG_CHECK(R[c] == 0);
        }
        if (bv > 0) {
            ENGINE_CHECK_MSG(closure.size() < n, "ring3k::howell_small: more closure rows than columns");
            closure.emplace_back(n, U16(0));
            P = row(pid);   // emplace_back may move the closure rows (a closure pivot row lives there)
            U16* Z = closure.back().data();
            const uint32_t f = pow3(k - bv);
            for (uint32_t j = c + 1; j < n; ++j) Z[j] = static_cast<U16>((static_cast<uint32_t>(P[j]) * f) % M);
            active.push_back(m + static_cast<uint32_t>(closure.size() - 1));
            ++closures;
        }
        prow_id.push_back(pid);
        out.pivots.push_back(c);
        out.pivot_v.push_back(bv);
    }
    g_closure_rows.fetch_add(closures, std::memory_order_relaxed);
    const uint32_t h = static_cast<uint32_t>(prow_id.size());
    if (!reduce) {
        out.H = DenseBuf<U32>(0, n);
        return out;
    }
    std::vector<U16> H(static_cast<size_t>(h) * n);
    for (uint32_t i = 0; i < h; ++i) std::copy(row(prow_id[i]), row(prow_id[i]) + n, H.data() + static_cast<size_t>(i) * n);
    std::vector<U16>().swap(store);
    std::vector<std::vector<U16>>().swap(closure);
    for (uint32_t i = 0; i < h; ++i) {
        const uint32_t ji = out.pivots[i];
        const uint32_t d = pow3(out.pivot_v[i]);
        const U16* Ri = H.data() + static_cast<size_t>(i) * n;
        for (uint32_t ip = 0; ip < i; ++ip) {
            U16* Rp = H.data() + static_cast<size_t>(ip) * n;
            const uint32_t q = Rp[ji] / d;
            if (q != 0) axpy(Rp + ji, Ri + ji, static_cast<U16>(q), n - ji);
        }
    }
    out.H = DenseBuf<U32>(h, n);
    for (uint32_t i = 0; i < h; ++i)
        for (uint32_t j = 0; j < n; ++j) out.H.at(i, j) = H[static_cast<size_t>(i) * n + j];
    return out;
}

std::vector<U16> small_rows(const DenseBuf<U32>& A, uint32_t k, bool transpose_augment) {
    const U32 M = pow3(k);
    if (!transpose_augment) {
        std::vector<U16> s(static_cast<size_t>(A.rows) * A.cols);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) {
                const U32 x = A.at(i, j);
                ENGINE_CHECK_MSG(x < M, "ring3k: input entry not canonical mod 3^k");
                s[static_cast<size_t>(i) * A.cols + j] = static_cast<U16>(x);
            }
        return s;
    }
    // [A^T | I_n] (n x (m + n)), the Sec. 13.6 kernel rule.
    const uint32_t m = A.rows, n = A.cols, w = m + n;
    std::vector<U16> s(static_cast<size_t>(n) * w, U16(0));
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            const U32 x = A.at(i, j);
            ENGINE_CHECK_MSG(x < M, "ring3k: input entry not canonical mod 3^k");
            s[static_cast<size_t>(j) * w + i] = static_cast<U16>(x);
        }
    for (uint32_t j = 0; j < n; ++j) s[static_cast<size_t>(j) * w + m + j] = 1;
    return s;
}

void check_k(uint32_t k, const char* who) {
    ENGINE_CHECK_MSG(k >= 1 && k <= kMaxK, who);
}

} // namespace

Params params() {
    Params p;
    p.panel = g_panel.load(std::memory_order_relaxed);
    p.reduce_block = g_reduce_block.load(std::memory_order_relaxed);
    return p;
}
void set_params(const Params& p) {
    g_panel.store(p.panel ? p.panel : 1, std::memory_order_relaxed);
    g_reduce_block.store(p.reduce_block ? p.reduce_block : 1, std::memory_order_relaxed);
}
Counters counters() {
    Counters c;
    c.lu_calls = g_lu_calls.load();
    c.lu_pivots = g_lu_pivots.load();
    c.closure_rows = g_closure_rows.load();
    c.smith_levels = g_smith_levels.load();
    c.gemm_calls = g_gemm_calls.load();
    return c;
}
void reset_counters() {
    g_lu_calls = 0; g_lu_pivots = 0; g_closure_rows = 0; g_smith_levels = 0; g_gemm_calls = 0;
}

SmithResult smith_levels(DenseBuf<U32> Ain, uint32_t k) {
    check_k(k, "ring3k::smith_levels: need 1 <= k <= 19");
    Work A = to_work(Ain, k);
    DenseBuf<U32>().swap(Ain);
    SmithResult res;
    res.k = k;
    const uint32_t panel = smithpar::route_panel(params().panel);   // (PAR-UNIFY) 128 on the threaded route
    std::vector<U32> tmp;
    std::vector<uint8_t> trits;
    for (uint32_t s = 0; s < k && A.rows > 0 && A.cols > 0; ++s) {
        const uint32_t kk = k - s;
        const bool f2r = smithpar::f2_route();   // (PAR-F2) the lazy F_3 echelon and the threaded level passes
        bool any = false;
        {
            smithpar::PhaseTimer tm(smithpar::Phase::LevelPass);
            if (!f2r) {
                for (U32 x : A.a) if (x != 0) { any = true; break; }
            } else {
                // (PAR-F2) the zero test over row tiles (a boolean: whichever tile decides it)
                std::atomic<bool> hit{false};
                const uint32_t nc = A.cols;
                smithpar::for_rows(A.rows, static_cast<uint64_t>(A.rows) * nc / 4u, [&](uint32_t i0, uint32_t i1) {
                    for (uint32_t i = i0; i < i1 && !hit.load(std::memory_order_relaxed); ++i) {
                        const U32* ai = A.a.data() + static_cast<size_t>(i) * nc;
                        for (uint32_t j = 0; j < nc; ++j)
                            if (ai[j] != 0) { hit.store(true, std::memory_order_relaxed); return; }
                    }
                });
                any = hit.load();
            }
        }
        if (!any) break;
        g_smith_levels.fetch_add(1, std::memory_order_relaxed);
        // The F_3 column rank profile of A mod 3 (TER-1's TritMatrix echelon), its columns moved to the front: the
        // unit-pivot LU then finds a pivot in each of the first r0 columns and never skips one.
        uint32_t r0 = 0;
        std::vector<uint32_t> order;
        {
            binary::TritMatrix T;
            if (f2r) {
                // (PAR-F2) the trit planes straight from A over row tiles (from_u8's bits: plane one = (x mod 3 == 1),
                // plane two = (x mod 3 == 2)), without the m x n byte buffer
                smithpar::PhaseTimer tm(smithpar::Phase::F3Pack);
                T = binary::TritMatrix(A.rows, A.cols);
                const uint32_t nc = A.cols, WP = T.words_per_plane();
                smithpar::for_rows(A.rows, static_cast<uint64_t>(A.rows) * nc, [&](uint32_t i0, uint32_t i1) {
                    for (uint32_t i = i0; i < i1; ++i) {
                        const U32* ai = A.a.data() + static_cast<size_t>(i) * nc;
                        uint64_t* t1 = T.row(i);
                        uint64_t* t2 = t1 + WP;
                        for (uint32_t j0 = 0; j0 < nc; j0 += 64) {
                            const uint32_t e = std::min<uint32_t>(64, nc - j0);
                            uint64_t w1 = 0, w2 = 0;
                            for (uint32_t q = 0; q < e; ++q) {
                                const U32 v = ai[j0 + q] % 3u;
                                w1 |= static_cast<uint64_t>(v == 1u) << q;
                                w2 |= static_cast<uint64_t>(v == 2u) << q;
                            }
                            t1[j0 >> 6] = w1;
                            t2[j0 >> 6] = w2;
                        }
                    }
                });
            } else {
            smithpar::PhaseTimer tm(smithpar::Phase::F3Pack);
            trits.resize(A.a.size());
            if (smithpar::threads() <= 1) {
                for (size_t t = 0; t < A.a.size(); ++t) trits[t] = static_cast<uint8_t>(A.a[t] % 3u);
            } else {
                // (PAR-2K) row tiles
                const uint32_t nc = A.cols;
                smithpar::for_rows(A.rows, static_cast<uint64_t>(A.rows) * nc, [&](uint32_t i0, uint32_t i1) {
                    for (size_t t = static_cast<size_t>(i0) * nc; t < static_cast<size_t>(i1) * nc; ++t)
                        trits[t] = static_cast<uint8_t>(A.a[t] % 3u);
                });
            }
            T = binary::TritMatrix::from_u8(trits.data(), A.rows, A.cols, A.cols);
            }
            smithpar::PhaseTimer tme(smithpar::Phase::F3Echelon);
            const binary::TritRref tr = f2r ? binary::echelon_lazy(T) : binary::echelon(T);
            r0 = tr.rank;
            order.assign(tr.pivots.begin(), tr.pivots.end());
            order.reserve(A.cols);
            for (uint32_t j = 0, t = 0; j < A.cols; ++j) {
                if (t < tr.pivots.size() && tr.pivots[t] == j) { ++t; continue; }
                order.push_back(j);
            }
        }
        res.level_ranks.resize(s + 1, 0);
        res.level_ranks[s] = r0;
        for (uint32_t t = 0; t < r0; ++t) res.sigma.push_back(s);
        if (r0 > 0) {
            {
            smithpar::PhaseTimer tmp_(smithpar::Phase::Perm);
            if (smithpar::threads() <= 1) {
                tmp.resize(A.cols);
                for (uint32_t i = 0; i < A.rows; ++i) {
                    U32* ai = A.row(i);
                    for (uint32_t j = 0; j < A.cols; ++j) tmp[j] = ai[order[j]];
                    std::copy(tmp.begin(), tmp.end(), ai);
                }
            } else {
                // (PAR-2K) row tiles, a row buffer per tile
                smithpar::for_rows(A.rows, static_cast<uint64_t>(A.rows) * A.cols, [&](uint32_t i0, uint32_t i1) {
                    std::vector<U32> rb(A.cols);
                    for (uint32_t i = i0; i < i1; ++i) {
                        U32* ai = A.row(i);
                        for (uint32_t j = 0; j < A.cols; ++j) rb[j] = ai[order[j]];
                        std::copy(rb.begin(), rb.end(), ai);
                    }
                });
            }
            }
            EchelonInfo e;
            {
                smithpar::PhaseTimer tml(smithpar::Phase::LU);
                e = echelon_blocked(A, kk, PivotRule::FirstUnit, panel, r0);
            }
            ENGINE_CHECK_MSG(e.rank == r0 && (r0 == 0 || e.pivots.back() == r0 - 1),
                             "ring3k::smith: the unit-pivot LU disagrees with the F_3 rank profile");
        }
        smithpar::PhaseTimer tms(smithpar::Phase::Schur);
        // The Schur complement (rows [r0, m), columns [r0, n)) is 0 mod 3: divide by 3 exactly, compact in place
        // (destination index <= source index, so the forward copy is safe).
        const uint32_t sr = A.rows - r0, sc = A.cols - r0;
        if (smithpar::threads() <= 1) {
            for (uint32_t i = 0; i < sr; ++i) {
                const size_t src0 = static_cast<size_t>(r0 + i) * A.cols + r0;
                const size_t dst0 = static_cast<size_t>(i) * sc;
                for (uint32_t jj = 0; jj < sc; ++jj) {
                    const U32 x = A.a[src0 + jj];
                    ENGINE_DEBUG_CHECK_MSG(x % 3u == 0, "ring3k::smith: Schur complement not 0 mod 3");
                    A.a[dst0 + jj] = x / 3u;
                }
            }
        } else {
            // (PAR-2K) Ring2k.h's two passes: in-row division by 3 into the front of each source row (row tiles), then
            // the rows move to their compact place in ascending order (dst(i) ends before source row i + 1 starts).
            const uint32_t n0 = A.cols;
            smithpar::for_rows(sr, static_cast<uint64_t>(sr) * sc, [&](uint32_t i0, uint32_t i1) {
                for (uint32_t i = i0; i < i1; ++i) {
                    U32* row = A.a.data() + static_cast<size_t>(r0 + i) * n0;
                    for (uint32_t jj = 0; jj < sc; ++jj) {
                        const U32 x = row[r0 + jj];
                        ENGINE_DEBUG_CHECK_MSG(x % 3u == 0, "ring3k::smith: Schur complement not 0 mod 3");
                        row[jj] = x / 3u;
                    }
                }
            });
            if (f2r) {
                smithpar::compact_rows(A.a.data(), r0, n0, sr, sc);   // (PAR-F2) the same move, over row tiles
            } else {
                for (uint32_t i = 0; i < sr; ++i) {
                    const size_t src0 = static_cast<size_t>(r0 + i) * n0, dst0 = static_cast<size_t>(i) * sc;
                    if (src0 != dst0) std::memmove(A.a.data() + dst0, A.a.data() + src0, sizeof(U32) * sc);
                }
            }
        }
        A.rows = sr;
        A.cols = sc;
        A.a.resize(static_cast<size_t>(sr) * sc);
    }
    return res;
}

HowellForm howell_blocked(DenseBuf<U32> Ain, uint32_t k) {
    check_k(k, "ring3k::howell_blocked: need 1 <= k <= 19");
    Work A = to_work(Ain, k);
    DenseBuf<U32>().swap(Ain);
    const EchelonInfo e = echelon_blocked(A, k, PivotRule::MinValuation, smithpar::route_panel(params().panel));
    A.a.resize(static_cast<size_t>(e.rank) * A.cols);
    A.rows = e.rank;
    return howell_pass(std::move(A), k, params().reduce_block);
}

SmithResult smith_lengths(const DenseBuf<U32>& A, uint32_t k) {
    ENGINE_CHECK_MSG(k >= 1 && k <= kSmallK, "ring3k::smith_lengths: need 1 <= k <= 5");
    SmithResult res;
    res.k = k;
    if (A.rows == 0 || A.cols == 0) return res;
    // L_j = length of rowspan(A mod 3^j) = sum_i max(0, j - sigma_i), so L_j - L_(j-1) = #{sigma < j}.  L_1 = the F_3
    // rank (TritMatrix echelon); L_j, j >= 2, = sum (j - v) over the Howell pivots of A mod 3^j (pass only).  The
    // orientation with the shorter rows is taken (rowspan and colspan have one length).
    const bool tr = A.cols > A.rows;
    const uint32_t m = tr ? A.cols : A.rows, n = tr ? A.rows : A.cols;
    std::vector<uint32_t> L(k + 1, 0);
    {
        const U32 M = pow3(k);
        std::unique_ptr<smithpar::PhaseTimer> tm(new smithpar::PhaseTimer(smithpar::Phase::F3Pack));
        std::vector<uint8_t> trits(static_cast<size_t>(m) * n);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) {
                const U32 x = A.at(i, j);
                ENGINE_CHECK_MSG(x < M, "ring3k: input entry not canonical mod 3^k");
                trits[tr ? static_cast<size_t>(j) * n + i : static_cast<size_t>(i) * n + j] = static_cast<uint8_t>(x % 3u);
            }
        binary::TritMatrix T = binary::TritMatrix::from_u8(trits.data(), m, n, n);
        tm.reset();
        const smithpar::PhaseTimer te(smithpar::Phase::F3Echelon);
        // (PAR-F2) the lazy blocked Gauss over row tiles on the threaded route: echelon's rank, bit for bit
        L[1] = smithpar::f2_route() ? binary::echelon_lazy(T).rank : binary::echelon(T).rank;
    }
    for (uint32_t j = 2; j <= k; ++j) {
        const U32 Mj = pow3(j);
        std::vector<U16> s(static_cast<size_t>(m) * n);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t c = 0; c < A.cols; ++c)
                s[tr ? static_cast<size_t>(c) * n + i : static_cast<size_t>(i) * n + c] = static_cast<U16>(A.at(i, c) % Mj);
        const HowellForm h = howell_small(std::move(s), m, n, j, false);
        L[j] = h.length();
    }
    g_smith_levels.fetch_add(k, std::memory_order_relaxed);
    // count(sigma = s) = D_(s+1) - D_s, D_j = L_j - L_(j-1) = #{sigma < j}, D_0 = 0.
    uint32_t prev = 0;
    for (uint32_t s = 0; s < k; ++s) {
        const uint32_t Dn = L[s + 1] - L[s];
        ENGINE_CHECK_MSG(Dn >= prev, "ring3k::smith_lengths: the lengths are not convex");
        res.level_ranks.push_back(Dn - prev);
        for (uint32_t t = 0; t < Dn - prev; ++t) res.sigma.push_back(s);
        prev = Dn;
    }
    return res;
}

SmithResult smith(DenseBuf<U32> A, uint32_t k) {
    check_k(k, "ring3k::smith: need 1 <= k <= 19");
    if (k <= kSmallK) return smith_lengths(A, k);
    return smith_levels(std::move(A), k);
}

HowellForm howell(DenseBuf<U32> A, uint32_t k) {
    check_k(k, "ring3k::howell: need 1 <= k <= 19");
    if (k <= kSmallK) {
        const uint32_t m = A.rows, n = A.cols;
        return howell_small(small_rows(A, k, false), m, n, k, true);
    }
    return howell_blocked(std::move(A), k);
}

HowellForm kernel_module(const DenseBuf<U32>& A, uint32_t k) {
    check_k(k, "ring3k::kernel_module: need 1 <= k <= 19");
    const uint32_t m = A.rows, n = A.cols;
    HowellForm h;
    if (k <= kSmallK) {
        h = howell_small(small_rows(A, k, true), n, m + n, k, true);
    } else {
        DenseBuf<U32> B(n, m + n);
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < n; ++j) B.at(j, i) = A.at(i, j);
        for (uint32_t j = 0; j < n; ++j) B.at(j, m + j) = 1u;
        h = howell_blocked(std::move(B), k);
    }
    HowellForm out;
    out.k = k;
    uint32_t first = 0;
    while (first < h.pivots.size() && h.pivots[first] < m) ++first;
    const uint32_t r = static_cast<uint32_t>(h.pivots.size()) - first;
    out.H = DenseBuf<U32>(r, n);
    for (uint32_t t = 0; t < r; ++t) {
        out.pivots.push_back(h.pivots[first + t] - m);
        out.pivot_v.push_back(h.pivot_v[first + t]);
        for (uint32_t j = 0; j < n; ++j) out.H.at(t, j) = h.H.at(first + t, m + j);
    }
    return out;
}

SmithResult smith_naive(DenseBuf<U32> Ain, uint32_t k) {
    check_k(k, "ring3k::smith_naive: need 1 <= k <= 19");
    Work A = to_work(Ain, k);
    const ModCtx<U32> ctx = ctx_for(k);
    const UnitInverter inv(k);
    SmithResult res;
    res.k = k;
    const uint32_t m = A.rows, n = A.cols;
    for (uint32_t r = 0; r < m && r < n; ++r) {
        uint32_t bi = m, bj = n, bv = k;
        for (uint32_t i = r; i < m && bv > 0; ++i)
            for (uint32_t j = r; j < n; ++j) {
                const U32 x = A.at(i, j);
                if (x == 0) continue;
                const uint32_t v = val3(x, k);
                if (v < bv) { bv = v; bi = i; bj = j; if (v == 0) break; }
            }
        if (bi == m) break;
        swap_rows(A, bi, r);
        if (bj != r) for (uint32_t i = 0; i < m; ++i) std::swap(A.at(i, bj), A.at(i, r));
        const U32* P = A.row(r);
        const U32 d = pow3(bv);
        const U32 ui = inv.inv(P[r] / d);
        for (uint32_t i = r + 1; i < m; ++i) {
            U32* R = A.row(i);
            if (R[r] == 0) continue;
            axpy_sub(ctx, R + r, P + r, ctx.mul(R[r] / d, ui), n - r);
        }
        // The pivot has minimal valuation in the whole block: clearing its row by column operations touches only the
        // pivot row, so the remaining block is unchanged.
        res.sigma.push_back(bv);
    }
    return res;
}

HowellForm howell_naive(DenseBuf<U32> Ain, uint32_t k) {
    check_k(k, "ring3k::howell_naive: need 1 <= k <= 19");
    Work A = to_work(Ain, k);
    DenseBuf<U32>().swap(Ain);
    return howell_pass(std::move(A), k, 1);
}

} // namespace ring3k
} // namespace engine
