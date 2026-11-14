// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// PAR-D2: the blocked, threaded, SIMD replay of D2's minimal-valuation Smith form with factored transforms.  See the header and
// docs/implementation/slot24/PARD2_report.md.
#include "DescentSmith.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <system_error>
#include <thread>

#include "DescentSmithKernels.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Binary/SmithPar.h"
#include "../../../ModArithmetic/Binary/ParWorker.h"

namespace integral_descent {
namespace maps {

namespace {

using clk = std::chrono::steady_clock;

// ---------------------------------------------------------------------------------------------------------------------------
// the lever
// ---------------------------------------------------------------------------------------------------------------------------
// (S24 CL-5) on EnvLever: "" = unset; an invalid value warns once and reads as the default.
// (S26 LEVERS-26) The env arm HECKE_DESCENT_SMITH_BLOCKED=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1: EnvLever.h
// EnvArmRetiredLever); SmithFormOracle stays in every build (C-13) via ScopedDescentSmithBlocked.
engine::lever::EnvArmRetiredLever g_blocked("HECKE_DESCENT_SMITH_BLOCKED",
                                  engine::lever::Policy{true, engine::lever::Empty::Unset, engine::lever::Match::Exact01, engine::lever::Cache::Once, "WARNING: HECKE_DESCENT_SMITH_BLOCKED=%s is neither 0 nor 1; the blocked Smith form stays on (default)\n", "default", "env", "default"});
int simd_env() {
    static const int v = [] {
        const char* e = std::getenv("HECKE_DESCENT_SMITH_SIMD");
        if (!e || !*e) return -1;
        if (std::strcmp(e, "scalar") == 0 || std::strcmp(e, "0") == 0) return 0;
        if (std::strcmp(e, "avx2") == 0) return 1;
        if (std::strcmp(e, "avx512") == 0) return 2;
        std::fprintf(stderr, "WARNING: HECKE_DESCENT_SMITH_SIMD=%s is not scalar|avx2|avx512; auto\n", e);
        return -1;
    }();
    return v;
}

thread_local SmithStats t_stats;
std::atomic<SmithCapture*> g_capture{nullptr};
std::mutex g_capture_mu;   // (PAR-ASM audit F2) guards the capture vectors (SmithCapture and CallerCapture)

// ---------------------------------------------------------------------------------------------------------------------------
// (PAR-UNIFY) the threads: the process's ONE descent pool (engine::smithpar, src/ModArithmetic/Binary/SmithPar.h), shared with the
// ring2k / ring3k Smith forms.  No pool is created per factorisation or per product any more: a call above the work threshold
// runs one region on the shared pool (the caller takes part; exceptions are carried to the caller); a call below it, a nested
// call, or a call while another thread holds the pool runs serially (the same values: every task writes its own rows).
// ---------------------------------------------------------------------------------------------------------------------------
class Pool {
public:
    explicit Pool(uint32_t n) : n_(std::max(1u, n)) {}
    uint32_t size() const { return n_; }
    void run(uint32_t ntasks, const std::function<void(uint32_t)>& fn) { engine::smithpar::parallel_for_n(n_, ntasks, fn); }

private:
    uint32_t n_;
};

// The work threshold (multiply-adds), shared with the ring2k / ring3k route: below it a step stays serial.  A function of the
// shape only (deterministic), never of a value.
inline bool worth(uint64_t work) { return work >= 2 * engine::smithpar::min_work(); }

// ---------------------------------------------------------------------------------------------------------------------------
// the factorisation
// ---------------------------------------------------------------------------------------------------------------------------
struct Engine {
    const ZMod& Z;
    smithk::KZ kz;
    smithk::Kernels K;
    uint32_t r, c;
    uint32_t* A;
    bool want_g;
    Pool* pool;
    uint32_t nb;
    // per position (moves with row swaps)
    std::vector<uint32_t> cur;      // the row is updated by the pivots < cur
    std::vector<uint32_t> fr;       // the row has the column swaps of the steps < fr
    std::vector<uint32_t> lb;       // a lower bound of the row's minimal valuation on the trailing columns (k: zero)
    std::vector<uint32_t> lbcol;    // the first column of minimal valuation at the last scan
    std::vector<uint32_t> seen;     // step + 1 of the last full scan (lbcol exact within that step)
    std::vector<uint32_t> rowOrig;
    // per step
    std::vector<uint32_t> sw, val, uinv;
    std::vector<uint32_t> colOrig;
    SmithStats st;

    Engine(const ZMod& Z_, uint32_t r_, uint32_t c_, uint32_t* A_, bool want_g_, Pool* pool_, uint32_t nb_, int simd)
        : Z(Z_), kz(smithk::MakeKZ(Z_)), K(smithk::Select(simd)), r(r_), c(c_), A(A_), want_g(want_g_), pool(pool_), nb(nb_),
          cur(r_, 0), fr(r_, 0), lb(r_, 0), lbcol(r_, 0), seen(r_, 0), rowOrig(r_), colOrig(c_) {
        for (uint32_t i = 0; i < r; ++i) rowOrig[i] = i;
        for (uint32_t j = 0; j < c; ++j) colOrig[j] = j;
        const uint32_t mn = std::min(r, c);
        sw.reserve(mn);
        val.reserve(mn);
        uinv.reserve(mn);
    }
    uint32_t* row(uint32_t i) const { return A + size_t(i) * c; }

    void apply_frames(uint32_t i, uint32_t upto) {
        uint32_t* a = row(i);
        if (upto > fr[i] + 4)   // the targets are scattered over the row: issue their loads first
            for (uint32_t x = fr[i]; x < upto; ++x) __builtin_prefetch(a + sw[x], 1);
        for (uint32_t x = fr[i]; x < upto; ++x) {
            const uint32_t y = sw[x];
            if (y != x) std::swap(a[x], a[y]);
        }
        if (fr[i] < upto) fr[i] = upto;
    }
    // Bring row i up to date with the pivots < t (and the column frame t).  Scratch: f, U (>= t entries).
    void bring(uint32_t i, uint32_t t, uint32_t* f, const uint32_t** U) {
        if (cur[i] >= t) {
            apply_frames(i, t);
            return;
        }
        if (lb[i] >= Z.k) {   // a zero trailing row stays zero (every multiplier vanishes); column swaps permute zeros
            fr[i] = cur[i] = t;
            return;
        }
        apply_frames(i, t);
        uint32_t* a = row(i);
        uint32_t nf = 0;
        for (uint32_t s = cur[i]; s < t; ++s) {
            const uint32_t x = a[s];
            if (!x) continue;
            const uint32_t fv = smithk::Mul(kz, smithk::DivPowExact(kz, x, val[s]), uinv[s]);
            a[s] = fv;
            const uint32_t* us = row(s);
            if (s + 1 < t) {
                const uint32_t* u1 = us + s + 1;
                K.upd(kz, a + s + 1, &u1, &fv, 1, t - s - 1);
            }
            f[nf] = fv;
            U[nf] = us + t;
            ++nf;
        }
        if (nf && t < c) K.upd(kz, a + t, U, f, nf, c - t);
        cur[i] = t;
        ++st.row_updates;
    }
    // Rows [lo, hi) up to date with the pivots < t, in parallel tiles; the trailing update is chunked over columns so that the
    // panel of pivot rows stays in cache.
    void bring_all(uint32_t lo, uint32_t hi, uint32_t t) {
        if (lo >= hi) return;
        const uint32_t tile = 8;
        const uint32_t ntiles = (hi - lo + tile - 1) / tile;
        auto job = [&](uint32_t k) {
            const uint32_t a0 = lo + k * tile, a1 = std::min(hi, a0 + tile);
            uint32_t span = 1;
            for (uint32_t i = a0; i < a1; ++i)
                if (cur[i] < t) span = std::max(span, t - cur[i]);
            std::vector<uint32_t> fbuf(size_t(tile) * span);
            std::vector<const uint32_t*> ubuf(size_t(tile) * span);
            uint32_t nfs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint32_t idx[8];
            uint32_t m = 0;
            // phase 1 (multipliers on the panel columns) per row; collect the trailing update
            for (uint32_t i = a0; i < a1; ++i) {
                if (cur[i] >= t) {
                    apply_frames(i, t);
                    continue;
                }
                if (lb[i] >= Z.k) {
                    fr[i] = cur[i] = t;
                    continue;
                }
                apply_frames(i, t);
                uint32_t* a = row(i);
                uint32_t* f = fbuf.data() + size_t(m) * span;
                const uint32_t** U = ubuf.data() + size_t(m) * span;
                uint32_t nf = 0;
                for (uint32_t s = cur[i]; s < t; ++s) {
                    const uint32_t x = a[s];
                    if (!x) continue;
                    const uint32_t fv = smithk::Mul(kz, smithk::DivPowExact(kz, x, val[s]), uinv[s]);
                    a[s] = fv;
                    const uint32_t* us = row(s);
                    if (s + 1 < t) {
                        const uint32_t* u1 = us + s + 1;
                        K.upd(kz, a + s + 1, &u1, &fv, 1, t - s - 1);
                    }
                    f[nf] = fv;
                    U[nf] = us + t;
                    ++nf;
                }
                cur[i] = t;
                if (nf) {
                    nfs[m] = nf;
                    idx[m] = i;
                    ++m;
                }
            }
            // phase 2: the trailing columns, chunked
            if (m && t < c) {
                const uint32_t chunk = 2048;
                std::vector<const uint32_t*> Uc(span);
                for (uint32_t j0 = t; j0 < c; j0 += chunk) {
                    const uint32_t n = std::min(chunk, c - j0);
                    for (uint32_t q = 0; q < m; ++q) {
                        const uint32_t** U = ubuf.data() + size_t(q) * span;
                        for (uint32_t x = 0; x < nfs[q]; ++x) Uc[x] = U[x] + (j0 - t);
                        K.upd(kz, row(idx[q]) + j0, Uc.data(), fbuf.data() + size_t(q) * span, nfs[q], n);
                    }
                }
            }
        };
        if (pool && pool->size() > 1 && ntiles > 1 && worth(uint64_t(hi - lo) * (c - std::min(c, t) + 1) * std::min(nb, t + 1)))
            pool->run(ntiles, job);
        else
            for (uint32_t k = 0; k < ntiles; ++k) job(k);
        st.row_updates += hi - lo;
    }
    // Scan the (up to date) row i on the columns >= t: the first unit, else the first entry of minimal valuation.
    // Returns the minimal valuation (0: a unit at lbcol[i]); sets lb[i], lbcol[i].
    uint32_t scan(uint32_t i, uint32_t t) {
        const uint32_t* a = row(i);
        uint32_t best = Z.k, col = 0;
        smithk::ScanRow(kz, K, a, t, c, best, col);
        lb[i] = best;
        lbcol[i] = col;
        seen[i] = t + 1;
        ++st.rows_scanned;
        return best;
    }

    void run() {
        std::vector<uint32_t> f(std::max(r, c) + 1);
        std::vector<const uint32_t*> U(std::max(r, c) + 1);
        uint32_t t = 0, t0 = 0;
        while (t < r && t < c) {
            // ---- the pivot search (exactly the oracle's choice) ----
            uint32_t best = Z.k, bi = 0, bj = 0;
            bool unit = false;
            // pass 1: the first unit in row-major order; a row with lb >= 1 has no unit (valuations only grow)
            for (uint32_t i = t; i < r; ++i) {
                if (lb[i] >= 1) {
                    ++st.rows_skipped;
                    continue;
                }
                bring(i, t, f.data(), U.data());
                if (scan(i, t) == 0) {
                    unit = true;
                    best = 0;
                    bi = i;
                    bj = lbcol[i];
                    break;
                }
            }
            if (!unit) {
                // pass 2: no unit anywhere; the first entry of minimal valuation (rows with lb >= best cannot win; 1 is the floor)
                for (uint32_t i = t; i < r && best > 1; ++i) {
                    if (lb[i] >= best) {
                        ++st.rows_skipped;
                        continue;
                    }
                    if (seen[i] != t + 1 || cur[i] < t) {
                        bring(i, t, f.data(), U.data());
                        scan(i, t);
                    }
                    if (lb[i] < best) {
                        best = lb[i];
                        bi = i;
                        bj = lbcol[i];
                    }
                }
            }
            if (best >= Z.k) break;
            // ---- the swaps ----
            if (bi != t) {
                std::swap_ranges(row(t), row(t) + c, row(bi));
                std::swap(cur[t], cur[bi]);
                std::swap(fr[t], fr[bi]);
                std::swap(lb[t], lb[bi]);
                std::swap(lbcol[t], lbcol[bi]);
                std::swap(seen[t], seen[bi]);
                std::swap(rowOrig[t], rowOrig[bi]);
            }
            sw.push_back(bj);
            if (bj != t) {
                std::swap(colOrig[t], colOrig[bj]);
                ++st.col_swaps;
            }
            // the panel's pivot rows (and the new one) follow the column swap eagerly; every other row lazily (apply_frames)
            for (uint32_t i = t0; i <= t; ++i) apply_frames(i, t + 1);
            const uint32_t pv = row(t)[t];
            ENGINE_DEBUG_CHECK(pv != 0);
            val.push_back(best);
            uinv.push_back(Z.inv(smithk::DivPowExact(kz, pv, best)));
            ++t;
            if (t - t0 >= nb && t < r) {
                bring_all(t, r, t);
                ++st.panels;
                t0 = t;
            }
        }
        // the multipliers of the rows below the last pivot (the trailing values are zero or absent)
        if (t < r) bring_all(t, r, t);
        const uint32_t s = t;
        if (want_g) {
            auto job = [&](uint32_t i) {
                apply_frames(i, s);
                uint32_t* a = row(i);
                const uint32_t v = val[i], u = uinv[i];
                for (uint32_t j = i + 1; j < c; ++j)
                    if (a[j]) a[j] = smithk::Mul(kz, smithk::DivPowExact(kz, a[j], v), u);
            };
            if (pool && pool->size() > 1 && worth(uint64_t(s) * c * 4))
                pool->run(s, job);
            else
                for (uint32_t i = 0; i < s; ++i) job(i);
        }
    }
};

// (PAR-UNIFY) the process's descent thread budget (ENGINE_THREADS, else the CPU affinity count; SmithPar.h)
uint32_t default_threads() {
    // (PAR-ASM audit F4) on a worker of an outer region (src/ModArithmetic/Binary/ParWorker.h): serial, no nested pool
    if (engine::parworker::active()) return 1;
    return engine::smithpar::default_threads();
}

inline uint32_t neg(const ZMod& Z, uint32_t x) { return Z.two ? ((0u - x) & Z.mask) : (x ? static_cast<uint32_t>(Z.m - x) : 0u); }

// run fn over [0, n) tasks with the factors' thread count when the work is large enough
void par(const SmithFactors& S, uint32_t ntasks, uint64_t work, const std::function<void(uint32_t)>& fn) {
    if (S.threads > 1 && ntasks > 1 && worth(work)) {
        engine::smithpar::parallel_for_n(std::min(S.threads, ntasks), ntasks, fn);
    } else {
        for (uint32_t k = 0; k < ntasks; ++k) fn(k);
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------
// public
// ---------------------------------------------------------------------------------------------------------------------------
const engine::lever::EnvLever& DescentSmithBlockedLever() { return g_blocked; }
bool DescentSmithBlockedEnabled() { return g_blocked.enabled(); }

uint32_t DescentSmithDefaultThreads() { return default_threads(); }
std::string DescentSmithSimdPath(int simd) {
    if (simd < 0) simd = simd_env();
    return smithk::PathName(smithk::Clamp(simd));
}

int DescentSmithClampSimd(int simd) { return smithk::Clamp(simd); }

uint32_t SmithFactors::smax() const {
    uint32_t s = 0;
    for (uint32_t v : sigma) s = std::max(s, v);
    return s;
}

void SetSmithCapture(SmithCapture* cap) { g_capture.store(cap); }
void SmithCaptureNote(const ZMod& Z, const Mat& A) {
    SmithCapture* cap = g_capture.load();
    if (!cap) return;
    if (uint64_t(A.r) * A.c < cap->min_entries) return;
    std::lock_guard<std::mutex> lk(g_capture_mu);
    if (cap->items.size() >= cap->max_items) return;
    cap->items.emplace_back(Z, A);
}

namespace {
std::atomic<CallerCapture*> g_callers{nullptr};
}
void SetCallerCapture(CallerCapture* cap) { g_callers.store(cap); }
bool CaptureActive() { return g_capture.load() != nullptr || g_callers.load() != nullptr; }
void CallerCaptureBuildPres(const ZMod& Z, const Mat& P, uint32_t rankP, const Mat& Nx, uint32_t c, uint32_t b) {
    CallerCapture* cap = g_callers.load();
    if (!cap) return;
    if (std::max(uint64_t(P.r) * P.c, uint64_t(Nx.r) * Nx.c) > cap->max_entries) return;
    std::lock_guard<std::mutex> lk(g_capture_mu);
    if (cap->bp.size() >= cap->max_items) return;
    cap->bp.push_back({Z, P, rankP, Nx, c, b});
}
void CallerCaptureInducedMap(const Pres& X, const Pres& Y, const Mat& f, uint32_t rankF) {
    CallerCapture* cap = g_callers.load();
    if (!cap) return;
    if (uint64_t(f.r) * f.c > cap->max_entries) return;
    std::lock_guard<std::mutex> lk(g_capture_mu);
    if (cap->im.size() >= cap->max_items) return;
    cap->im.push_back({X, Y, f, rankF});
}

SmithFactors SmithFactor(const ZMod& Z, Mat A, bool want_g, const SmithEngineOptions& opt) {
    const auto t0 = clk::now();
    ENGINE_CHECK_MSG(Z.m < (uint64_t(1) << 31), "SmithFactor: ell^k < 2^31 (the certified accumulator bound)");
    SmithFactors S;
    S.Z = Z;
    S.r = A.r;
    S.c = A.c;
    S.threads = engine::parworker::active() ? 1u : (opt.threads ? opt.threads : default_threads());   // (PAR-ASM audit F4)
    S.simd = smithk::Clamp(opt.simd >= 0 ? opt.simd : simd_env());
    std::unique_ptr<Pool> pool;
    if (S.threads > 1) pool.reset(new Pool(S.threads));   // a handle on the shared pool (no threads created here)
    {   // reduce the input (in parallel blocks of rows when large)
        const uint32_t rb = 64, nb = (A.r + rb - 1) / rb;
        auto job = [&](uint32_t b) {
            uint32_t* p = A.a.data() + size_t(b) * rb * A.c;
            uint32_t* e = A.a.data() + std::min(size_t(A.r), size_t(b + 1) * rb) * A.c;
            if (Z.two)
                for (; p < e; ++p) *p &= Z.mask;
            else
                for (; p < e; ++p) *p = Z.red(*p);
        };
        if (pool && worth(uint64_t(A.r) * A.c))
            pool->run(nb, job);
        else
            for (uint32_t b = 0; b < nb; ++b) job(b);
    }
    Engine E(Z, A.r, A.c, A.a.data(), want_g, pool.get(), opt.panel ? opt.panel : 128u, S.simd);
    E.run();
    S.sigma = E.val;
    S.uinv = std::move(E.uinv);
    S.rowOrig = std::move(E.rowOrig);
    S.colOrig = std::move(E.colOrig);
    S.A = std::move(A);
    S.g_ready = want_g;
    E.st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    t_stats = E.st;
    return S;
}

// ---- transforms ---------------------------------------------------------------------------------------------------------
Mat FactorsLinvTRows(const SmithFactors& S, const std::vector<uint32_t>& rows) {
    const uint32_t r = S.r, s = S.rank();
    Mat O(static_cast<uint32_t>(rows.size()), r);
    for (uint32_t q = 0; q < rows.size(); ++q) {
        const uint32_t t = rows[q];
        ENGINE_CHECK(t < r);
        uint32_t* o = O.row(q);
        o[S.rowOrig[t]] = 1;
        if (t < s)
            for (uint32_t i = t + 1; i < r; ++i) o[S.rowOrig[i]] = S.A.at(i, t);
    }
    return O;
}
Mat FactorsLinvT(const SmithFactors& S) {
    std::vector<uint32_t> rows(S.r);
    for (uint32_t i = 0; i < S.r; ++i) rows[i] = i;
    return FactorsLinvTRows(S, rows);
}
Mat FactorsRinvRows(const SmithFactors& S, const std::vector<uint32_t>& rows) {
    ENGINE_CHECK_MSG(S.g_ready, "FactorsRinvRows: the factorisation kept no G");
    const uint32_t c = S.c, s = S.rank();
    Mat O(static_cast<uint32_t>(rows.size()), c);
    for (uint32_t q = 0; q < rows.size(); ++q) {
        const uint32_t t = rows[q];
        ENGINE_CHECK(t < c);
        uint32_t* o = O.row(q);
        o[S.colOrig[t]] = 1;
        if (t < s) {
            const uint32_t* g = S.A.row(t);
            for (uint32_t j = t + 1; j < c; ++j) o[S.colOrig[j]] = g[j];
        }
    }
    return O;
}
Mat FactorsRinv(const SmithFactors& S) {
    std::vector<uint32_t> rows(S.c);
    for (uint32_t i = 0; i < S.c; ++i) rows[i] = i;
    return FactorsRinvRows(S, rows);
}

// L = F^{-1} P: row i of F^{-1} is y with y F = e_i: y_i = 1, y_j = 0 (j > i or s <= j != i), and for j < min(i, s), descending,
// y_j = -(F[i][j] + sum_{j < u < min(i, s)} y_u F[u][j])  ("push": acc := F[i][.]; at u, y_u = -acc[u], acc[0..u) -= acc[u] F[u][0..u)).
Mat FactorsLRows(const SmithFactors& S, const std::vector<uint32_t>& rows) {
    const uint32_t r = S.r, s = S.rank();
    const smithk::KZ kz = smithk::MakeKZ(S.Z);
    const smithk::Kernels K = smithk::Select(S.simd);
    Mat O(static_cast<uint32_t>(rows.size()), r);
    uint64_t work = 0;
    for (uint32_t i : rows) work += uint64_t(std::min(i, s)) * std::min(i, s) / 2;
    par(S, static_cast<uint32_t>(rows.size()), work, [&](uint32_t q) {
        const uint32_t i = rows[q];
        ENGINE_CHECK(i < r);
        uint32_t* o = O.row(q);
        o[S.rowOrig[i]] = 1;
        const uint32_t h = std::min(i, s);
        if (h == 0) return;
        std::vector<uint32_t> acc(S.A.row(i), S.A.row(i) + h);
        // blocked push: within a block of u's push only into the block, then one multi-pivot update below it
        const uint32_t bw = 32;
        std::vector<uint32_t> fco(bw);
        std::vector<const uint32_t*> Up(bw);
        uint32_t u1 = h;
        while (u1 > 0) {
            const uint32_t u0 = u1 > bw ? u1 - bw : 0;
            uint32_t nf = 0;
            for (uint32_t u = u1; u-- > u0;) {
                const uint32_t a = acc[u];
                if (!a) continue;
                const uint32_t* fu = S.A.row(u);
                if (u > u0) {
                    const uint32_t* p = fu + u0;
                    K.upd(kz, acc.data() + u0, &p, &a, 1, u - u0);
                }
                fco[nf] = a;
                Up[nf] = fu;
                ++nf;
            }
            if (nf && u0 > 0) K.upd(kz, acc.data(), Up.data(), fco.data(), nf, u0);
            u1 = u0;
        }
        for (uint32_t u = 0; u < h; ++u) o[S.rowOrig[u]] = neg(S.Z, acc[u]);
    });
    return O;
}

// L (all rows): F^{-1} by column blocks (independent; X = the block's columns of F^{-1}, rows i >= j0), each row
// X[i] = e_i - sum_{j0 <= t < min(i, s)} F[i][t] X[t].
Mat FactorsL(const SmithFactors& S) {
    const uint32_t r = S.r, s = S.rank();
    const smithk::KZ kz = smithk::MakeKZ(S.Z);
    const smithk::Kernels K = smithk::Select(S.simd);
    Mat O(r, r);
    for (uint32_t i = s; i < r; ++i) O.at(i, S.rowOrig[i]) = 1;
    const uint32_t w = 64;
    const uint32_t nblk = (s + w - 1) / w;
    par(S, nblk, uint64_t(r) * r * std::max(1u, s) / 6, [&](uint32_t b) {
        const uint32_t j0 = b * w, j1 = std::min(s, j0 + w), bw = j1 - j0;
        std::vector<uint32_t> X(size_t(r - j0) * bw, 0);
        std::vector<uint32_t> fco(r);
        std::vector<const uint32_t*> Up(r);
        for (uint32_t i = j0; i < r; ++i) {
            uint32_t* xi = X.data() + size_t(i - j0) * bw;
            if (i < j1) xi[i - j0] = 1;
            const uint32_t h = std::min(i, s);
            const uint32_t* fi = S.A.row(i);
            uint32_t nf = 0;
            for (uint32_t t = j0; t < h; ++t)
                if (fi[t]) {
                    fco[nf] = fi[t];
                    Up[nf] = X.data() + size_t(t - j0) * bw;
                    ++nf;
                }
            if (nf) K.upd(kz, xi, Up.data(), fco.data(), nf, bw);
            for (uint32_t q = 0; q < bw; ++q) O.at(i, S.rowOrig[j0 + q]) = xi[q];
        }
    });
    return O;
}

// R^T rows t (= columns t of R = Q G^{-1}): z = column t of G^{-1}: z_t = 1, z_u = 0 (u > t, or s <= u != t), and for
// j < min(t, s), descending, z_j = -(G[j][t] + sum_{j < u < min(t, s)} G[j][u] z_u).  A block of t's at once (Zb: rows j, one
// column per t), each row Zb[j] = -(G[j][t_q])_q - sum_{j < u < jmax} G[j][u] Zb[u] with Zb[u][q] fixed to delta for u >= t_q.
Mat FactorsRTRows(const SmithFactors& S, const std::vector<uint32_t>& rows) {
    ENGINE_CHECK_MSG(S.g_ready, "FactorsRTRows: the factorisation kept no G");
    const uint32_t c = S.c, s = S.rank();
    const smithk::KZ kz = smithk::MakeKZ(S.Z);
    const smithk::Kernels K = smithk::Select(S.simd);
    Mat O(static_cast<uint32_t>(rows.size()), c);
    const uint32_t w = 64;
    const uint32_t nblk = static_cast<uint32_t>((rows.size() + w - 1) / w);
    uint64_t work = uint64_t(rows.size()) * s * s / 2;
    par(S, nblk, work, [&](uint32_t b) {
        const uint32_t q0 = b * w, q1 = std::min<uint32_t>(static_cast<uint32_t>(rows.size()), q0 + w), bw = q1 - q0;
        uint32_t maxt = 0;
        for (uint32_t q = q0; q < q1; ++q) {
            ENGINE_CHECK(rows[q] < c);
            maxt = std::max(maxt, rows[q]);
        }
        const uint32_t jmax = std::min(s, maxt + 1);
        std::vector<uint32_t> Zb(size_t(jmax) * bw, 0);
        std::vector<uint32_t> fco(jmax + 1);
        std::vector<const uint32_t*> Up(jmax + 1);
        for (uint32_t j = jmax; j-- > 0;) {
            uint32_t* zj = Zb.data() + size_t(j) * bw;
            const uint32_t* g = S.A.row(j);
            for (uint32_t q = 0; q < bw; ++q) {
                const uint32_t t = rows[q0 + q];
                zj[q] = t >= s ? neg(S.Z, g[t]) : 0u;   // t < s: the u = t term below carries G[j][t]
            }
            uint32_t nf = 0;
            for (uint32_t u = j + 1; u < jmax; ++u)
                if (g[u]) {
                    fco[nf] = g[u];
                    Up[nf] = Zb.data() + size_t(u) * bw;
                    ++nf;
                }
            if (nf) K.upd(kz, zj, Up.data(), fco.data(), nf, bw);
            for (uint32_t q = 0; q < bw; ++q) {
                const uint32_t t = rows[q0 + q];
                if (t <= j) zj[q] = t == j ? 1u : 0u;
            }
        }
        for (uint32_t q = 0; q < bw; ++q) {
            const uint32_t t = rows[q0 + q];
            uint32_t* o = O.row(q0 + q);
            for (uint32_t j = 0; j < jmax; ++j) o[S.colOrig[j]] = Zb[size_t(j) * bw + q];
            if (t >= s) o[S.colOrig[t]] = 1;
        }
    });
    return O;
}
Mat FactorsRT(const SmithFactors& S) {
    std::vector<uint32_t> rows(S.c);
    for (uint32_t i = 0; i < S.c; ++i) rows[i] = i;
    return FactorsRTRows(S, rows);
}

// L B = F^{-1} (P B): Y[i] = (P B)[i] - sum_{t < min(i, s)} F[i][t] Y[t].
Mat FactorsApplyL(const SmithFactors& S, const Mat& B) {
    ENGINE_CHECK_MSG(B.r == S.r, "FactorsApplyL: shape");
    const uint32_t r = S.r, s = S.rank(), k = B.c;
    const smithk::KZ kz = smithk::MakeKZ(S.Z);
    const smithk::Kernels K = smithk::Select(S.simd);
    Mat Y(r, k);
    if (k == 0) return Y;
    for (uint32_t j = 0; j < r; ++j) std::copy(B.row(S.rowOrig[j]), B.row(S.rowOrig[j]) + k, Y.row(j));
    std::vector<uint32_t> fco(r);
    std::vector<const uint32_t*> Up(r);
    for (uint32_t i = 1; i < r; ++i) {
        const uint32_t h = std::min(i, s);
        const uint32_t* fi = S.A.row(i);
        uint32_t nf = 0;
        for (uint32_t t = 0; t < h; ++t)
            if (fi[t]) {
                fco[nf] = fi[t];
                Up[nf] = Y.row(t);
                ++nf;
            }
        if (nf) K.upd(kz, Y.row(i), Up.data(), fco.data(), nf, k);
    }
    return Y;
}

// ---- products ------------------------------------------------------------------------------------------------------------
namespace {
// row i of A B into out (length B.c): out := -(sum_k A[i][k] B[k]) by the kernel, then negated (exact)
void ProductRow(const smithk::KZ& kz, const smithk::Kernels& K, const ZMod& Z, const Mat& A, const Mat& B, uint32_t i, uint32_t* out,
                std::vector<uint32_t>& f, std::vector<const uint32_t*>& U, bool reduce_a = false) {
    std::fill(out, out + B.c, 0u);
    const uint32_t* a = A.row(i);
    uint32_t nf = 0;
    for (uint32_t k = 0; k < A.c; ++k) {
        const uint32_t x = reduce_a ? Z.red(a[k]) : a[k];
        if (x) {
            f[nf] = x;
            U[nf] = B.row(k);
            ++nf;
        }
    }
    if (!nf) return;
    K.upd(kz, out, U.data(), f.data(), nf, B.c);
    for (uint32_t j = 0; j < B.c; ++j) out[j] = neg(Z, out[j]);
}
}  // namespace

namespace {
// the SIMD bodies need reduced operands at odd ell (the Montgomery bound); at ell = 2 the arithmetic is exact mod 2^32 anyway
void CheckReduced(const ZMod& Z, const Mat& A, const char* what) {
    if (Z.two) return;
    for (uint32_t x : A.a) ENGINE_CHECK_MSG(x < Z.m, what);
}
}  // namespace

Mat MulMod(const ZMod& Z, const Mat& A, const Mat& B, bool reduce_a) {
    ENGINE_CHECK_MSG(A.c == B.r, "maps::MulMod: shape");
    Mat C(A.r, B.c);
    if (A.r == 0 || B.c == 0) return C;
    if (!reduce_a) CheckReduced(Z, A, "maps::MulMod: an operand is not reduced");
    CheckReduced(Z, B, "maps::MulMod: an operand is not reduced");
    const smithk::KZ kz = smithk::MakeKZ(Z);
    const smithk::Kernels K = smithk::Select(simd_env());
    const uint32_t blk = 32;
    const uint32_t nb = (A.r + blk - 1) / blk;
    SmithFactors meta;
    meta.threads = default_threads();
    par(meta, nb, uint64_t(A.r) * A.c + uint64_t(A.r) * B.c * 8, [&](uint32_t b) {
        std::vector<uint32_t> f(A.c);
        std::vector<const uint32_t*> U(A.c);
        for (uint32_t i = b * blk; i < std::min(A.r, (b + 1) * blk); ++i) ProductRow(kz, K, Z, A, B, i, C.row(i), f, U, reduce_a);
    });
    return C;
}

bool ProductsEqual(const ZMod& Z, const Mat& A1, const Mat& B1, const Mat& A2, const Mat& B2) {
    ENGINE_CHECK_MSG(A1.c == B1.r && A2.c == B2.r, "maps::ProductsEqual: shape");
    if (A1.r != A2.r || B1.c != B2.c) return false;
    if (A1.r == 0 || B1.c == 0) return true;
    for (const Mat* M : {&A1, &B1, &A2, &B2}) CheckReduced(Z, *M, "maps::ProductsEqual: an operand is not reduced");
    const smithk::KZ kz = smithk::MakeKZ(Z);
    const smithk::Kernels K = smithk::Select(simd_env());
    uint64_t nnz = 0;
    for (uint32_t x : A1.a) nnz += x != 0;
    for (uint32_t x : A2.a) nnz += x != 0;
    const uint32_t blk = 32;
    const uint32_t nb = (A1.r + blk - 1) / blk;
    std::atomic<bool> same{true};
    SmithFactors meta;
    meta.threads = default_threads();
    par(meta, nb, nnz * B1.c, [&](uint32_t b) {
        if (!same.load(std::memory_order_relaxed)) return;
        std::vector<uint32_t> f(std::max(A1.c, A2.c));
        std::vector<const uint32_t*> U(std::max(A1.c, A2.c));
        std::vector<uint32_t> r1(B1.c), r2(B1.c);
        for (uint32_t i = b * blk; i < std::min(A1.r, (b + 1) * blk); ++i) {
            ProductRow(kz, K, Z, A1, B1, i, r1.data(), f, U);
            ProductRow(kz, K, Z, A2, B2, i, r2.data(), f, U);
            if (r1 != r2) {
                same.store(false);
                return;
            }
        }
    });
    return same.load();
}

Smith FactorsToSmith(SmithFactors&& S, bool left, bool right) {
    Smith O;
    O.sigma = S.sigma;
    if (left) {
        O.L = FactorsL(S);
        O.LinvT = FactorsLinvT(S);
    }
    if (right) {
        O.RT = FactorsRT(S);
        O.Rinv = FactorsRinv(S);
    }
    return O;
}

Smith SmithFormBlocked(const ZMod& Z, Mat A, bool left, bool right, const SmithEngineOptions& opt) {
    return FactorsToSmith(SmithFactor(Z, std::move(A), right, opt), left, right);
}

}  // namespace maps
}  // namespace integral_descent
