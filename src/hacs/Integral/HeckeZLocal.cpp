// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZLocal -- slot-25 HZ-LOC: the local reads (see the header).

#include "HeckeZLocal.h"

#include "Descent/DescentModuleExport.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../Engine/EngineRuntime.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace heckez {
namespace local {

using u128 = unsigned __int128;

// ---- (slot-26 Z9) the route -----------------------------------------------------------------------------------------------------
namespace {
bool LeverDefaultPar() {
    static const bool par = [] {
        const char* e = std::getenv("HECKEZ_LOCAL_PAR");
        return !(e && std::strcmp(e, "0") == 0);
    }();
    return par;
}
std::atomic<bool> g_override{false};
std::atomic<bool> g_override_par{true};
std::atomic<uint32_t> g_override_threads{0};
std::atomic<bool> g_override_force{false};
std::atomic<uint64_t> g_par_reduce_calls{0}, g_par_matmul_calls{0};
thread_local bool t_in_task = false;   // inside a ParFor task: everything below runs serially (no nested regions)

uint32_t RouteThreads(const ParRoute& r) {
    const uint32_t pool = engine::tape::consumer_pool().size();
    const uint32_t t = r.threads ? std::min(r.threads, pool) : pool;
    return std::max<uint32_t>(1, t);
}

// fn(task) for task in [0, n), in parallel over at most `threads` threads; every task writes disjoint memory
void ParFor(uint32_t n, uint32_t threads, const std::function<void(uint32_t)>& fn) {
    if (threads <= 1 || n <= 1 || t_in_task) {
        for (uint32_t i = 0; i < n; ++i) fn(i);
        return;
    }
    engine::tape::consumer_pool().parallel_for(n, threads, [&](uint32_t task, uint32_t) {
        const bool saved = t_in_task;
        t_in_task = true;
        try {
            fn(task);
        } catch (...) {
            t_in_task = saved;
            throw;
        }
        t_in_task = saved;
    });
}

// threads for a top-level parallel region (1 inside a task or on the serial route)
uint32_t TopThreads() {
    if (t_in_task) return 1;
    const ParRoute r = CurrentParRoute();
    return r.par ? RouteThreads(r) : 1;
}
}  // namespace

ParRoute CurrentParRoute() {
    ParRoute r;
    if (g_override.load(std::memory_order_acquire)) {
        r.par = g_override_par.load(std::memory_order_relaxed);
        r.threads = g_override_threads.load(std::memory_order_relaxed);
        r.force = g_override_force.load(std::memory_order_relaxed);
    } else {
        r.par = LeverDefaultPar();
    }
    return r;
}

ParCallCounts ParCalls() { return ParCallCounts{g_par_reduce_calls.load(), g_par_matmul_calls.load()}; }

const char* DefaultParRouteName() { return LeverDefaultPar() ? "par" : "serial(HECKEZ_LOCAL_PAR=0)"; }

ScopedParRoute::ScopedParRoute(ParRoute r) : had_(g_override.load()), saved_(CurrentParRoute()) {
    g_override_par.store(r.par);
    g_override_threads.store(r.threads);
    g_override_force.store(r.force);
    g_override.store(true, std::memory_order_release);
}

ScopedParRoute::~ScopedParRoute() {
    g_override_par.store(saved_.par);
    g_override_threads.store(saved_.threads);
    g_override_force.store(saved_.force);
    g_override.store(had_, std::memory_order_release);
}

// ---- Z/ell^K --------------------------------------------------------------------------------------------------------------------
bool Zlk::Fits(uint64_t ell, uint32_t K) {
    if (ell < 2 || K < 1) return false;
    u128 x = 1;
    for (uint32_t i = 0; i < K; ++i) {
        x *= ell;
        if (x >= static_cast<u128>(kWordModulusLimit)) return false;
    }
    return true;
}

Zlk Zlk::Make(uint64_t ell, uint32_t K) {
    ENGINE_CHECK_MSG(Fits(ell, K), "HeckeZLocal: ell^K must be < 2^63 (refused(word_modulus) upstream)");
    Zlk R;
    R.ell = ell;
    R.K = K;
    R.ctx = engine::ModCtx<uint64_t>::chain(ell, K);
    R.q = R.ctx.p;
    return R;
}

uint32_t Zlk::val(uint64_t x) const {
    x %= q;
    if (x == 0) return K;
    uint32_t v = 0;
    while (x % ell == 0) {
        x /= ell;
        ++v;
    }
    return v;
}

uint64_t Zlk::pw(uint32_t v) const {
    if (v >= K) return 0;
    uint64_t x = 1;
    for (uint32_t i = 0; i < v; ++i) x *= ell;
    return x;
}

uint64_t Zlk::inv_unit(uint64_t u) const {
    // extended Euclid on (u, q), q = ell^K, gcd(u, ell) = 1
    u %= q;
    ENGINE_CHECK_MSG(u % ell != 0, "HeckeZLocal: inv_unit of a non-unit");
    __int128 a = static_cast<__int128>(u), b = static_cast<__int128>(q), x0 = 1, x1 = 0;
    while (b != 0) {
        const __int128 t = a / b;
        __int128 r = a - t * b;
        a = b;
        b = r;
        r = x0 - t * x1;
        x0 = x1;
        x1 = r;
    }
    ENGINE_CHECK_MSG(a == 1, "HeckeZLocal: inv_unit: gcd != 1");
    __int128 m = x0 % static_cast<__int128>(q);
    if (m < 0) m += static_cast<__int128>(q);
    const uint64_t r = static_cast<uint64_t>(m);
    ENGINE_CHECK(mul(r, u) == 1 % q);
    return r;
}

std::string Zlk::regime() const { return engine::regime_name(ctx.regime); }

// ---- matrices -------------------------------------------------------------------------------------------------------------------
Mat MatIdentity(size_t D) {
    Mat I(D * D, 0);
    for (size_t i = 0; i < D; ++i) I[i * D + i] = 1;
    return I;
}

namespace {
// rows [i0, i1) of C = A B (C pre-zeroed); the slot-25 loop, unchanged per row
void MatMulRows(const Zlk& R, const Mat& A, const Mat& B, size_t D, Mat& C, size_t i0, size_t i1) {
    for (size_t i = i0; i < i1; ++i)
        for (size_t k = 0; k < D; ++k) {
            const uint64_t a = A[i * D + k];
            if (a == 0) continue;
            const uint64_t* b = &B[k * D];
            uint64_t* c = &C[i * D];
            for (size_t j = 0; j < D; ++j)
                if (b[j]) c[j] = R.add(c[j], R.mul(a, b[j]));
        }
}
constexpr size_t kParMatMulMinD = 64;
}  // namespace

Mat MatMul(const Zlk& R, const Mat& A, const Mat& B, size_t D) {
    ENGINE_CHECK(A.size() == D * D && B.size() == D * D);
    Mat C(D * D, 0);
    const uint32_t th = (D >= kParMatMulMinD || CurrentParRoute().force) ? TopThreads() : 1;
    if (th <= 1) {
        MatMulRows(R, A, B, D, C, 0, D);
        return C;
    }
    g_par_matmul_calls.fetch_add(1, std::memory_order_relaxed);
    // (Z9) row-parallel: every output row is the slot-25 loop on its own row, so C is identical
    const uint32_t tasks = static_cast<uint32_t>(std::min<size_t>(D, size_t(th) * 4));
    ParFor(tasks, th, [&](uint32_t t) {
        const size_t i0 = D * t / tasks, i1 = D * (t + 1) / tasks;
        MatMulRows(R, A, B, D, C, i0, i1);
    });
    return C;
}

Mat MatAdd(const Zlk& R, const Mat& A, const Mat& B) {
    ENGINE_CHECK(A.size() == B.size());
    Mat C(A.size());
    for (size_t i = 0; i < A.size(); ++i) C[i] = R.add(A[i], B[i]);
    return C;
}

Mat MatScale(const Zlk& R, const Mat& A, uint64_t s) {
    Mat C(A.size());
    s = R.red(s);
    for (size_t i = 0; i < A.size(); ++i) C[i] = R.mul(A[i], s);
    return C;
}

Mat MatMinusScalar(const Zlk& R, const Mat& A, uint64_t s, size_t D) {
    Mat C = A;
    s = R.red(s);
    for (size_t i = 0; i < D; ++i) C[i * D + i] = R.sub(C[i * D + i], s);
    return C;
}

Mat MatPolyEval(const Zlk& R, const std::vector<uint64_t>& p, const Mat& A, size_t D) {
    Mat X(D * D, 0);
    for (size_t t = p.size(); t-- > 0;) {
        X = MatMul(R, X, A, D);
        const uint64_t c = R.red(p[t]);
        for (size_t i = 0; i < D; ++i) X[i * D + i] = R.add(X[i * D + i], c);
    }
    return X;
}

bool MatIsZero(const Mat& A) {
    for (uint64_t x : A)
        if (x) return false;
    return true;
}

// ---- polynomials ----------------------------------------------------------------------------------------------------------------
Poly Charpoly(const Zlk& R, const Mat& A, size_t D) {
    // Berkowitz, high -> low: C = [1, c_1, .., c_r] = det(x I - A_r), A_r the leading r x r block
    std::vector<uint64_t> C{1};
    for (size_t r = 0; r < D; ++r) {
        // A_{r+1} = [[A_r, col], [row, a]]
        const uint64_t a = A[r * D + r];
        std::vector<uint64_t> t(r + 2, 0);
        t[0] = 1;
        t[1] = R.neg(a);
        std::vector<uint64_t> v(r);
        for (size_t i = 0; i < r; ++i) v[i] = A[i * D + r];   // col
        for (size_t j = 0; j < r; ++j) {
            // t_{j+2} = - row . A_r^j col
            uint64_t s = 0;
            for (size_t i = 0; i < r; ++i)
                if (v[i]) s = R.add(s, R.mul(A[r * D + i], v[i]));
            t[j + 2] = R.neg(s);
            if (j + 1 < r) {
                std::vector<uint64_t> w(r, 0);
                for (size_t i = 0; i < r; ++i)
                    for (size_t k = 0; k < r; ++k)
                        if (v[k]) w[i] = R.add(w[i], R.mul(A[i * D + k], v[k]));
                v.swap(w);
            }
        }
        std::vector<uint64_t> Cn(r + 2, 0);
        for (size_t i = 0; i < r + 2; ++i)
            for (size_t j = 0; j <= std::min(i, r); ++j)
                if (C[j] && t[i - j]) Cn[i] = R.add(Cn[i], R.mul(t[i - j], C[j]));
        C.swap(Cn);
    }
    Poly p(C.rbegin(), C.rend());   // low -> high
    return p;
}

bool DivideByRoot(const Zlk& R, Poly p, uint64_t a, uint32_t r, Poly* out) {
    a = R.red(a);
    for (uint32_t t = 0; t < r; ++t) {
        if (p.size() < 2) return false;
        // synthetic division, high -> low
        const size_t deg = p.size() - 1;
        Poly qt(deg, 0);
        uint64_t acc = 0;
        for (size_t i = deg + 1; i-- > 0;) {
            acc = R.add(R.mul(acc, a), p[i]);
            if (i > 0) qt[i - 1] = acc;
        }
        if (acc != 0) return false;   // the remainder p(a)
        p.swap(qt);
    }
    *out = p;
    return true;
}

bool MonicSqrt(const Zlk& R, const Poly& p, Poly* out) {
    ENGINE_CHECK_MSG(R.ell % 2 == 1, "HeckeZLocal: MonicSqrt needs ell odd");
    if (p.empty() || (p.size() - 1) % 2 != 0 || R.red(p.back()) != 1 % R.q) return false;
    const size_t m = (p.size() - 1) / 2;
    Poly c(m + 1, 0);
    c[m] = 1 % R.q;
    const uint64_t inv2 = R.inv_unit(2);
    for (size_t j = 1; j <= m; ++j) {
        // coefficient of x^{2m - j}: sum_{i + i' = 2m - j} c_i c_i' = 2 c_m c_{m-j} + sum_{i = m-j+1}^{m-1} c_i c_{2m-j-i}
        uint64_t s = R.red(p[2 * m - j]);
        for (size_t i = m - j + 1; i <= m - 1 && i < m; ++i) s = R.sub(s, R.mul(c[i], c[2 * m - j - i]));
        c[m - j] = R.mul(s, inv2);
    }
    // verify c^2 == p
    Poly sq(2 * m + 1, 0);
    for (size_t i = 0; i <= m; ++i)
        for (size_t j = 0; j <= m; ++j) sq[i + j] = R.add(sq[i + j], R.mul(c[i], c[j]));
    for (size_t i = 0; i < sq.size(); ++i)
        if (sq[i] != R.red(p[i])) return false;
    *out = c;
    return true;
}

uint64_t PolyEval(const Zlk& R, const Poly& p, uint64_t x) {
    x = R.red(x);
    uint64_t acc = 0;
    for (size_t i = p.size(); i-- > 0;) acc = R.add(R.mul(acc, x), R.red(p[i]));
    return acc;
}

uint64_t HexResidue(const Zlk& R, const std::string& hex) {
    size_t i = 0;
    bool neg = false;
    if (i < hex.size() && hex[i] == '-') {
        neg = true;
        ++i;
    }
    if (hex.compare(i, 2, "0x") != 0) throw std::invalid_argument("HeckeZLocal: HexResidue: not a canonical hex integer: " + hex);
    i += 2;
    u128 acc = 0;
    for (; i < hex.size(); ++i) {
        const char ch = hex[i];
        uint32_t d;
        if (ch >= '0' && ch <= '9') d = static_cast<uint32_t>(ch - '0');
        else if (ch >= 'a' && ch <= 'f') d = static_cast<uint32_t>(ch - 'a' + 10);
        else if (ch >= 'A' && ch <= 'F') d = static_cast<uint32_t>(ch - 'A' + 10);
        else throw std::invalid_argument("HeckeZLocal: HexResidue: bad digit in " + hex);
        acc = (acc * 16 + d) % R.q;
    }
    const uint64_t r = static_cast<uint64_t>(acc);
    return neg ? R.neg(r) : r;
}

// ---- Smith valuations -----------------------------------------------------------------------------------------------------------
std::vector<uint32_t> SmithValuations(const Zlk& R, std::vector<uint64_t> A, size_t r, size_t c) {
    ENGINE_CHECK(A.size() == r * c);
    for (uint64_t& x : A) x = R.red(x);
    std::vector<uint32_t> out;
    std::vector<size_t> rows(r), cols(c);
    for (size_t i = 0; i < r; ++i) rows[i] = i;
    for (size_t j = 0; j < c; ++j) cols[j] = j;
    while (!rows.empty() && !cols.empty()) {
        // the entry of minimal valuation
        uint32_t best = R.K;
        size_t bi = 0, bj = 0;
        for (size_t ii = 0; ii < rows.size() && best > 0; ++ii)
            for (size_t jj = 0; jj < cols.size(); ++jj) {
                const uint64_t x = A[rows[ii] * c + cols[jj]];
                if (!x) continue;
                const uint32_t v = R.val(x);
                if (v < best) {
                    best = v;
                    bi = ii;
                    bj = jj;
                    if (v == 0) break;
                }
            }
        if (best >= R.K) break;
        out.push_back(best);
        const size_t pr = rows[bi], pc = cols[bj];
        const uint64_t piv = A[pr * c + pc];
        const uint64_t lv = R.pw(best);
        const uint64_t uinv = R.inv_unit(piv / lv);
        // clear column pc below/above: row_i -= (A[i][pc] / ell^best) * uinv * row_pr  (only rows still active matter)
        for (size_t ii = 0; ii < rows.size(); ++ii) {
            if (ii == bi) continue;
            const size_t i = rows[ii];
            const uint64_t x = A[i * c + pc];
            if (!x) continue;
            const uint64_t f = R.mul(x / lv, uinv);
            for (size_t jj = 0; jj < cols.size(); ++jj) {
                const size_t j = cols[jj];
                const uint64_t y = A[pr * c + j];
                if (y) A[i * c + j] = R.sub(A[i * c + j], R.mul(f, y));
            }
            ENGINE_DEBUG_CHECK(A[i * c + pc] == 0);
        }
        // the column operations clearing row pr touch row pr only: drop row pr and column pc
        rows.erase(rows.begin() + static_cast<std::ptrdiff_t>(bi));
        cols.erase(cols.begin() + static_cast<std::ptrdiff_t>(bj));
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ---- Howell spans ---------------------------------------------------------------------------------------------------------------
Span::Span(const Zlk& R, size_t m) : R_(&R), m_(m) {}

namespace {
constexpr size_t kParReduceMinM = size_t(1) << 14;   // D >= 128
}  // namespace

bool Span::reduce(std::vector<uint64_t>& v, size_t* conflict) const {
    // true: reduced to zero; false: v != 0 left, *conflict = index of the blocking row (rows_.size(): no row at its first column)
    if (rows_.size() >= 2 && (m_ >= kParReduceMinM || CurrentParRoute().force)) {
        const uint32_t th = TopThreads();
        if (th > 1) {
            g_par_reduce_calls.fetch_add(1, std::memory_order_relaxed);
            return reduce_par(v, conflict, th);
        }
    }
    return reduce_serial(v, conflict);
}

// (Z9) The serial reduce, exactly, in three phases.  Rows are zero left of their pivot and pivots are distinct and ascending, so
// the factor f_i of row i depends only on the pivot columns (phase A, serial, O(r^2)).  Phase B applies v -= sum f_i row_i over
// column chunks in parallel (rows up to phase A's valuation conflict) and finds the first non-zero non-pivot column j_np; its
// value is the serial one, because the rows with pivot > j_np are zero there.  If the serial loop would have stopped at j_np
// (before phase A's conflict), phase C adds the rows with pivot > j_np back.  Residues are canonical, so the order of the
// additions does not matter: v and *conflict are those of reduce_serial.
bool Span::reduce_par(std::vector<uint64_t>& v, size_t* conflict, uint32_t threads) const {
    const size_t r = rows_.size();
    // ---- phase A
    std::vector<uint64_t> vp(r), f(r, 0);
    for (size_t i = 0; i < r; ++i) vp[i] = v[rows_[i].piv];
    size_t stopA = r;
    for (size_t i = 0; i < r; ++i) {
        const uint64_t x = vp[i];
        if (x == 0) continue;
        const Row& row = rows_[i];
        if (R_->val(x) < row.v) {
            stopA = i;
            break;
        }
        const uint64_t fi = x / R_->pw(row.v);
        f[i] = fi;
        for (size_t k = i + 1; k < r; ++k) {
            const uint64_t a = row.a[rows_[k].piv];
            if (a) vp[k] = R_->sub(vp[k], R_->mul(fi, a));
        }
    }
    const size_t jA = stopA < r ? rows_[stopA].piv : m_;
    // ---- phase B
    const uint32_t nch = static_cast<uint32_t>(std::min<size_t>(m_, size_t(threads) * 8));
    std::vector<size_t> first_np(nch, m_);
    ParFor(nch, threads, [&](uint32_t c) {
        const size_t c0 = m_ * c / nch, c1 = m_ * (c + 1) / nch;
        for (size_t i = 0; i < stopA && rows_[i].piv < c1; ++i) {
            const uint64_t fi = f[i];
            if (!fi) continue;
            const uint64_t* a = rows_[i].a.data();
            for (size_t t = std::max(c0, rows_[i].piv); t < c1; ++t)
                if (a[t]) v[t] = R_->sub(v[t], R_->mul(fi, a[t]));
        }
        // the first non-zero non-pivot column in [c0, c1)
        auto it = std::lower_bound(rows_.begin(), rows_.end(), c0, [](const Row& rw, size_t col) { return rw.piv < col; });
        for (size_t t = c0; t < c1; ++t) {
            while (it != rows_.end() && it->piv < t) ++it;
            if (it != rows_.end() && it->piv == t) continue;
            if (v[t]) {
                first_np[c] = t;
                break;
            }
        }
    });
    size_t jnp = m_;
    for (size_t x : first_np) jnp = std::min(jnp, x);
    if (jnp < jA) {
        // ---- phase C: the serial loop stops at j_np (no row there): undo the rows with pivot > j_np
        ParFor(nch, threads, [&](uint32_t c) {
            const size_t c0 = m_ * c / nch, c1 = m_ * (c + 1) / nch;
            for (size_t i = 0; i < stopA && rows_[i].piv < c1; ++i) {
                if (rows_[i].piv < jnp) continue;
                const uint64_t fi = f[i];
                if (!fi) continue;
                const uint64_t* a = rows_[i].a.data();
                for (size_t t = std::max(c0, rows_[i].piv); t < c1; ++t)
                    if (a[t]) v[t] = R_->add(v[t], R_->mul(fi, a[t]));
            }
        });
        *conflict = r;
        return false;
    }
    if (stopA < r) {
        *conflict = stopA;
        return false;
    }
    ENGINE_DEBUG_CHECK(std::all_of(v.begin(), v.end(), [](uint64_t x) { return x == 0; }));
    return true;
}

bool Span::reduce_serial(std::vector<uint64_t>& v, size_t* conflict) const {
    size_t ri = 0;
    for (size_t j = 0; j < m_; ++j) {
        if (v[j] == 0) continue;
        while (ri < rows_.size() && rows_[ri].piv < j) ++ri;
        if (ri == rows_.size() || rows_[ri].piv != j) {
            *conflict = rows_.size();
            return false;
        }
        const Row& row = rows_[ri];
        const uint32_t vv = R_->val(v[j]);
        if (vv < row.v) {
            *conflict = ri;
            return false;
        }
        const uint64_t f = v[j] / R_->pw(row.v);   // exact: ell^{row.v} | v[j] as integers
        for (size_t t = j; t < m_; ++t)
            if (row.a[t]) v[t] = R_->sub(v[t], R_->mul(f, row.a[t]));
        ENGINE_DEBUG_CHECK(v[j] == 0);
    }
    return true;
}

bool Span::insert(std::vector<uint64_t> v0) {
    ENGINE_CHECK(v0.size() == m_);
    for (uint64_t& x : v0) x = R_->red(x);
    bool grew = false;
    std::vector<std::vector<uint64_t>> pending{std::move(v0)};
    while (!pending.empty()) {
        std::vector<uint64_t> v = std::move(pending.back());
        pending.pop_back();
        size_t conflict = 0;
        if (reduce(v, &conflict)) continue;
        size_t j = 0;
        while (v[j] == 0) ++j;
        const uint32_t vv = R_->val(v[j]);
        const uint64_t lv = R_->pw(vv);
        const uint64_t uinv = R_->inv_unit(v[j] / lv);
        for (size_t t = j; t < m_; ++t)
            if (v[t]) v[t] = R_->mul(v[t], uinv);
        ENGINE_DEBUG_CHECK(v[j] == lv);
        Row nr{j, vv, v};
        // the annihilator multiple ell^{K - vv} * row: zero at the pivot, maybe not elsewhere
        if (vv > 0) {
            std::vector<uint64_t> ann(m_, 0);
            const uint64_t s = R_->pw(R_->K - vv);
            bool nz = false;
            for (size_t t = j + 1; t < m_; ++t) {
                ann[t] = R_->mul(v[t], s);
                nz = nz || ann[t];
            }
            if (nz) pending.push_back(std::move(ann));
        }
        grew = true;
        if (conflict < rows_.size()) {
            // the new row has the smaller valuation at this pivot: it takes the slot, the old row is re-inserted
            std::vector<uint64_t> old = std::move(rows_[conflict].a);
            rows_[conflict] = std::move(nr);
            pending.push_back(std::move(old));
        } else {
            auto it = std::lower_bound(rows_.begin(), rows_.end(), j, [](const Row& r, size_t c) { return r.piv < c; });
            rows_.insert(it, std::move(nr));
        }
    }
    return grew;
}

bool Span::contains(std::vector<uint64_t> v) const {
    ENGINE_CHECK(v.size() == m_);
    for (uint64_t& x : v) x = R_->red(x);
    size_t conflict = 0;
    return reduce(v, &conflict);
}

uint32_t Span::length() const {
    uint32_t L = 0;
    for (const Row& r : rows_) L += R_->K - r.v;
    return L;
}

std::vector<std::vector<uint64_t>> Span::rows() const {
    std::vector<std::vector<uint64_t>> out;
    for (const Row& r : rows_) out.push_back(r.a);
    return out;
}

uint32_t Span::generators() const {
    std::vector<uint64_t> A;
    for (const Row& r : rows_) A.insert(A.end(), r.a.begin(), r.a.end());
    return static_cast<uint32_t>(SmithValuations(*R_, A, rows_.size(), m_).size());
}

uint32_t Span::length_by_smith() const {
    std::vector<uint64_t> A;
    for (const Row& r : rows_) A.insert(A.end(), r.a.begin(), r.a.end());
    uint32_t L = 0;
    for (uint32_t s : SmithValuations(*R_, A, rows_.size(), m_)) L += R_->K - s;
    return L;
}

namespace {
// (Z9) the par route of AlgebraClosure: the products of a pass in parallel (index-owned slots), inserted serially in the (work
// item, generator) order; the next pass multiplies only the product vectors whose insert grew the span (S_new = S_old + the
// inserted vectors, and S_old g lies in S_new already), so the closure is the same span (Z9_predictions.md Sec. 1 item 3).
Span AlgebraClosurePar(const Zlk& R, const std::vector<Mat>& gens, size_t D, uint32_t threads) {
    Span S(R, D * D);
    const Mat I = MatIdentity(D);
    S.insert(I);
    std::vector<Mat> work{I};
    const size_t G = gens.size();
    const size_t chunk = std::max<size_t>(1, size_t(threads) * 2);
    for (uint32_t pass = 0;; ++pass) {
        ENGINE_CHECK_MSG(pass <= D * D * R.K + 2, "HeckeZLocal: the algebra closure did not terminate");
        std::vector<Mat> next;
        bool changed = false;
        const size_t items = work.size() * G;
        for (size_t base = 0; base < items; base += chunk) {
            const size_t cnt = std::min(chunk, items - base);
            std::vector<Mat> prod(cnt);
            ParFor(static_cast<uint32_t>(cnt), threads, [&](uint32_t k) {
                const size_t idx = base + k;
                prod[k] = MatMul(R, work[idx / G], gens[idx % G], D);
            });
            for (size_t k = 0; k < cnt; ++k)
                if (S.insert(prod[k])) {
                    changed = true;
                    next.push_back(std::move(prod[k]));
                }
        }
        if (!changed) break;
        work.swap(next);
    }
    ENGINE_CHECK_MSG(S.length() == S.length_by_smith(), "HeckeZLocal: Howell length != Smith length (span defect)");
    return S;
}
}  // namespace

Span AlgebraClosure(const Zlk& R, const std::vector<Mat>& gens, size_t D) {
    {
        const uint32_t th = TopThreads();
        if (CurrentParRoute().par && !t_in_task) return AlgebraClosurePar(R, gens, D, th);
    }
    Span S(R, D * D);
    S.insert(MatIdentity(D));
    for (uint32_t pass = 0;; ++pass) {
        ENGINE_CHECK_MSG(pass <= D * D * R.K + 2, "HeckeZLocal: the algebra closure did not terminate");
        bool changed = false;
        const auto rows = S.rows();
        for (const auto& x : rows)
            for (const Mat& g : gens)
                if (S.insert(MatMul(R, x, g, D))) changed = true;
        if (!changed) break;
    }
    ENGINE_CHECK_MSG(S.length() == S.length_by_smith(), "HeckeZLocal: Howell length != Smith length (span defect)");
    return S;
}

Span PowerSpan(const Zlk& R, const Mat& theta, size_t D) {
    Span S(R, D * D);
    Mat X = MatIdentity(D);
    for (size_t i = 0; i < D; ++i) {
        S.insert(X);
        X = MatMul(R, X, theta, D);
    }
    return S;
}

// ---- the read set (audit F1) ---------------------------------------------------------------------------------------------------
std::vector<uint32_t> ReadPrimes(uint32_t N, uint64_t ell, const std::vector<uint32_t>& P, std::string* t_ell) {
    std::vector<uint32_t> out = P;
    std::string w;
    if (std::find(P.begin(), P.end(), ell) != P.end()) w = "in_P";
    else if (N % ell == 0) w = "omitted(ell|N:U_ell_not_exported)";
    else if (ell > kTEllCap) w = "omitted(cap:ell>" + std::to_string(kTEllCap) + ")";
    else {
        out.push_back(static_cast<uint32_t>(ell));
        w = "added";
    }
    if (t_ell) *t_ell = w;
    return out;
}

// ---- the lattice module from the export -----------------------------------------------------------------------------------------
LatticeModule LatticeFromExport(uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, bool allow_bad_primes) {
    LatticeModule M;
    M.N = N;
    M.n = n;
    M.ell = ell;
    M.K = K;
    M.P = P;
    M.source = "export";
    M.word = "export";
    if (ell >= (uint64_t(1) << 31)) {
        M.status = "refused(export_word)";
        M.detail = "the export's word ring needs ell < 2^31";
        return M;
    }
    integral_descent::modexport::ExportOptions eo;
    eo.allow_bad_primes = allow_bad_primes;   // (S26 Z1) default false: the default options, as before
    const integral_descent::modexport::ModuleExport E =
        integral_descent::modexport::ExportModule(N, n, static_cast<uint32_t>(ell), P, eo);
    M.K_max = E.K_max;
    if (!E.ok) {
        M.status = "blocked(export)";
        M.detail = E.refusal;
        return M;
    }
    if (K > E.K_max) {
        M.status = ell >= 46341 ? "refused(export_word)" : "open(precision)";
        M.detail = "K = " + std::to_string(K) + " > the export's K_max = " + std::to_string(E.K_max);
        return M;
    }
    if (!Zlk::Fits(ell, K)) {
        M.status = "refused(word_modulus)";
        return M;
    }
    const integral_descent::modexport::ModuleAtK A = E.at(K);
    M.D = A.free_rank;
    M.h1_rank = A.rank();
    for (size_t o = 0; o < A.ps.size(); ++o) {
        const auto F = A.free_block(o);
        ENGINE_CHECK(F.r == M.D && F.c == M.D);
        Mat T(F.a.begin(), F.a.end());
        M.T.push_back(std::move(T));
    }
    ENGINE_CHECK_MSG(A.ps == P, "HeckeZLocal: the export's operator list differs from P");
    M.regime = Zlk::Make(ell, K).regime();
    M.is_lattice = true;
    M.ok = true;
    M.status = "ok";
    return M;
}

LatticeModule ReduceLattice(const LatticeModule& M, uint32_t K) {
    ENGINE_CHECK(M.ok && K >= 1 && K <= M.K);
    LatticeModule X = M;
    X.K = K;
    const Zlk R = Zlk::Make(M.ell, K);
    for (Mat& T : X.T)
        for (uint64_t& x : T) x = R.red(x);
    X.regime = R.regime();
    return X;
}

// ---- the reads ------------------------------------------------------------------------------------------------------------------
namespace {

uint64_t PowU(const Zlk& R, uint64_t b, uint64_t e) {
    uint64_t r = 1 % R.q;
    b = R.red(b);
    while (e) {
        if (e & 1) r = R.mul(r, b);
        b = R.mul(b, b);
        e >>= 1;
    }
    return r;
}

std::vector<Mat> ReduceAll(const Zlk& R, const std::vector<Mat>& T) {
    std::vector<Mat> out = T;
    for (Mat& A : out)
        for (uint64_t& x : A) x = R.red(x);
    return out;
}

Mat Combination(const Zlk& R, const std::vector<Mat>& T, const std::vector<uint32_t>& c, size_t D) {
    Mat X(D * D, 0);
    for (size_t i = 0; i < T.size() && i < c.size(); ++i)
        if (c[i]) X = MatAdd(R, X, MatScale(R, T[i], c[i]));
    return X;
}

// polynomials over the field F_ell (Zlk with K = 1), low -> high, trimmed
void Trim(Poly& p) {
    while (!p.empty() && p.back() == 0) p.pop_back();
}
Poly PMul(const Zlk& F, const Poly& a, const Poly& b) {
    if (a.empty() || b.empty()) return {};
    Poly c(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) c[i + j] = F.add(c[i + j], F.mul(a[i], b[j]));
    Trim(c);
    return c;
}
Poly PSub(const Zlk& F, Poly a, const Poly& b) {
    if (a.size() < b.size()) a.resize(b.size(), 0);
    for (size_t i = 0; i < b.size(); ++i) a[i] = F.sub(a[i], b[i]);
    Trim(a);
    return a;
}
void PDivMod(const Zlk& F, Poly a, const Poly& b, Poly* qo, Poly* ro) {
    ENGINE_CHECK(!b.empty());
    Trim(a);
    Poly q(a.size() >= b.size() ? a.size() - b.size() + 1 : 0, 0);
    const uint64_t binv = F.inv_unit(b.back());
    while (a.size() >= b.size() && !a.empty()) {
        const size_t s = a.size() - b.size();
        const uint64_t f = F.mul(a.back(), binv);
        q[s] = f;
        for (size_t i = 0; i < b.size(); ++i) a[s + i] = F.sub(a[s + i], F.mul(f, b[i]));
        Trim(a);
    }
    Trim(q);
    *qo = q;
    *ro = a;
}
// u with u * a == 1 mod b (gcd 1), else empty
Poly PInvMod(const Zlk& F, const Poly& a, const Poly& b) {
    Poly r0 = b, r1 = a, s0{}, s1{1};
    Trim(r1);
    while (!r1.empty()) {
        Poly q, r;
        PDivMod(F, r0, r1, &q, &r);
        Poly s = PSub(F, s0, PMul(F, q, s1));
        r0 = r1;
        r1 = r;
        s0 = s1;
        s1 = s;
    }
    if (r0.size() != 1) return {};
    const uint64_t c = F.inv_unit(r0[0]);
    for (uint64_t& x : s0) x = F.mul(x, c);
    Poly q, r;
    PDivMod(F, s0, b, &q, &r);
    return r;
}

std::string JoinU(const std::vector<uint32_t>& v) {
    std::ostringstream o;
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    return o.str();
}

}  // namespace

bool LocalRead::certified_any() const {
    return c3 == "certified" || root == "closes" || (c3 == "certified" && mono != "none" && !mono.empty()) ||
           e_status.rfind("stopped", 0) == 0 || p26 == "eis+cusp" || p26 == "eis" || p26 == "cusp";
}

std::string LocalRead::text() const {
    std::ostringstream o;
    o << "hz_local/v1 N=" << N << " n=" << n << " ell=" << ell << " K=" << K << " source=" << source
      << " lattice=" << (is_lattice ? "yes" : "no") << " D=" << D << " d=" << d << " P=[" << JoinU(P) << "] regime=" << regime
      << " word=" << word << " t_ell=" << t_ell << " dim_A1=" << dim_A1 << " c3=" << c3 << " mono=" << mono << " e=" << e << " e_op=" << e_op << " e_status=" << e_status
      << " K_gen=" << K_gen << " p26=" << p26 << " p26_op=" << p26_op << " eis_block_rank=" << eis_block_rank << " roots=[";
    for (size_t i = 0; i < roots.size(); ++i) {
        const RootRead& r = roots[i];
        o << (i ? ";" : "") << r.op << ":" << r.annihilator << ":v=" << r.v_c_lambda << ":a=[" << JoinU(r.a) << "]:eta=["
          << JoinU(r.eta) << "]:" << r.status;
    }
    o << "] v_gP_local=" << v_gP_local << " v_gP=" << v_gP << "(" << v_gP_source << ") v_eta=" << v_eta << " eta=[" << JoinU(eta)
      << "] root=" << root << " C_T=" << C_T;
    return o.str();
}

uint64_t LocalRead::digest() const { return engine::FnvBytes(engine::kFnvOffset64, text()); }

LocalRead ReadLocal(const LatticeModule& M, const ReadOptions& opt) {
    ENGINE_CHECK_MSG(M.ok, "HeckeZLocal: ReadLocal on a refused lattice module");
    ENGINE_CHECK(M.T.size() == M.P.size() && M.K >= 1);
    {   // (Z9 fix1, SF-1) once per process: which route the record path ran (C stderr, not std::cout/std::cerr: the --serve cell capture swaps both stream buffers, so a served cell log never sees it; never a record field)
        static std::once_flag once;
        std::call_once(once, [] {
            const ParRoute r = CurrentParRoute();
            std::fprintf(stderr, "EFFECTIVE HECKEZ_LOCAL_PAR route=%s threads=%u source=%s\n", r.par ? "par" : "serial",
                         static_cast<unsigned>(r.par ? RouteThreads(r) : 1u),
                         std::getenv("HECKEZ_LOCAL_PAR") ? "env" : "default");
            std::fflush(stderr);
        });
    }
    LocalRead L;
    L.N = M.N;
    L.n = M.n;
    L.ell = M.ell;
    L.K = M.K;
    L.source = M.source;
    L.is_lattice = M.is_lattice;
    L.D = M.D;
    L.d = opt.d_target;
    L.P = M.P;
    const size_t D = M.D;
    const Zlk R = Zlk::Make(M.ell, M.K);
    const Zlk F = Zlk::Make(M.ell, 1);
    L.regime = R.regime();
    L.t_ell = M.t_ell;
    L.word = M.word;
    const std::vector<Mat> T = ReduceAll(R, M.T);
    const std::vector<Mat> T1 = ReduceAll(F, M.T);

    // ---- (C3)
    const Span A1 = AlgebraClosure(F, T1, D);
    L.dim_A1 = A1.length();
    ENGINE_CHECK_MSG(static_cast<int64_t>(L.dim_A1) <= opt.d_target || !M.is_lattice,
                     "HeckeZLocal: dim A'_1 exceeds the a-priori d (a wrong d or a wrong module)");
    const bool by_theory = M.N == 1 && (M.ell == 2 || M.ell == 3) && opt.d_target >= 4;
    if (!M.is_lattice) L.c3 = "A1_of_computed";
    else if (by_theory) L.c3 = "refused(by_theory)";
    else if (static_cast<int64_t>(L.dim_A1) == opt.d_target) L.c3 = "certified";
    else L.c3 = "open(sandwich)";

    // ---- theta candidates: HZ-RAT's theta, then each T_p
    struct Cand { std::string name; Mat m; std::vector<uint32_t> coeffs; };
    std::vector<Cand> cands;
    if (opt.rat.present && !opt.rat.theta.empty()) cands.push_back({"theta", Combination(R, T, opt.rat.theta, D), opt.rat.theta});
    for (size_t i = 0; i < T.size(); ++i) {
        std::vector<uint32_t> c(T.size(), 0);
        c[i] = 1;
        cands.push_back({"T_" + std::to_string(M.P[i]), T[i], c});
    }
    auto lambda_of = [&](const std::vector<uint32_t>& c) {
        uint64_t s = 0;
        for (size_t i = 0; i < c.size(); ++i)
            if (c[i]) s = R.add(s, R.mul(c[i] % R.q, R.add(1 % R.q, PowU(R, M.P[i], opt.k_weight - 1))));
        return s;
    };

    if (!M.is_lattice) {
        L.mono = "A1_of_computed";
        L.e_status = "A1_of_computed";
        L.p26 = "A1_of_computed";
        L.root = "A1_of_computed";
        return L;
    }

    // ---- (2.4) / (2.5)
    if (L.c3 == "certified") {
        L.mono = "none";
        for (const Cand& c : cands) {
            Mat m1 = c.m;
            for (uint64_t& x : m1) x = F.red(x);
            if (static_cast<int64_t>(PowerSpan(F, m1, D).length()) == opt.d_target) {
                L.mono = c.name;
                break;
            }
        }
        // the stopping rule for the first candidate
        const Cand& c0 = cands.front();
        L.e_op = c0.name;
        L.e_status = "open(precision)";
        const bool reuse_A1 = CurrentParRoute().par;   // (Z9) at Kc = 1 the ring and generators are those of A1
        for (uint32_t Kc = 1; Kc <= M.K; ++Kc) {
            const Zlk Rc = Zlk::Make(M.ell, Kc);
            std::optional<Span> A_own;
            if (!(reuse_A1 && Kc == 1)) A_own.emplace(AlgebraClosure(Rc, ReduceAll(Rc, M.T), D));
            const Span& A = A_own ? *A_own : A1;
            Mat th = c0.m;
            for (uint64_t& x : th) x = Rc.red(x);
            const Span B = PowerSpan(Rc, th, D);
            uint32_t t = 0;
            for (; t < Kc; ++t) {
                bool all = true;
                const uint64_t s = Rc.pw(t);
                for (const auto& row : A.rows()) {
                    std::vector<uint64_t> w(row.size());
                    for (size_t i = 0; i < row.size(); ++i) w[i] = Rc.mul(row[i], s);
                    if (!B.contains(w)) {
                        all = false;
                        break;
                    }
                }
                if (all) break;
            }
            if (t < Kc) {
                L.e = static_cast<int32_t>(t);
                L.e_status = "stopped(K=" + std::to_string(Kc) + ")";
                break;
            }
        }
    } else if (L.c3 == "open(sandwich)") {
        L.e_status = "not_saturated";
        L.mono = "open(cap)";
        for (uint32_t Kc = 2; Kc <= M.K; ++Kc) {
            const Zlk Rc = Zlk::Make(M.ell, Kc);
            const Span A = AlgebraClosure(Rc, ReduceAll(Rc, M.T), D);
            if (static_cast<int64_t>(A.generators()) != opt.d_target) continue;
            L.K_gen = Kc;
            L.mono = "AK_unsaturated(K=" + std::to_string(Kc) + "):none";
            Span ellA(Rc, D * D);
            for (const auto& row : A.rows()) {
                std::vector<uint64_t> w(row.size());
                for (size_t i = 0; i < row.size(); ++i) w[i] = Rc.mul(row[i], M.ell % Rc.q);
                ellA.insert(w);
            }
            const uint32_t len0 = ellA.length();
            for (const Cand& c : cands) {
                Span S = ellA;
                Mat th = c.m;
                for (uint64_t& x : th) x = Rc.red(x);
                Mat X = MatIdentity(D);
                for (size_t i = 0; i < D; ++i) {
                    S.insert(X);
                    X = MatMul(Rc, X, th, D);
                }
                if (static_cast<int64_t>(S.length() - len0) == opt.d_target) {
                    L.mono = "AK_unsaturated(K=" + std::to_string(Kc) + "):" + c.name;
                    break;
                }
            }
            break;
        }
    } else {
        L.mono = "not_run";
        L.e_status = "not_run";
    }

    if (!opt.eis_rational) {
        L.p26 = "not_run";
        L.root = "general_case";
        return L;
    }

    // ---- annihilators of the cuspidal part, per candidate
    struct Ann { std::string kind; Poly c; bool sharp; bool ok; };
    auto annihilator = [&](const Cand& cd) -> Ann {
        if (cd.name == "theta") {
            Poly c;
            for (const std::string& h : opt.rat.c_hex) c.push_back(HexResidue(R, h));
            return {"hz_rat_cusp_part", c, true, true};
        }
        const Poly chi = Charpoly(R, cd.m, D);
        Poly q;
        if (!DivideByRoot(R, chi, lambda_of(cd.coeffs), opt.r_eis, &q)) return {"not_exact", {}, false, false};
        if (M.ell % 2 == 1) {
            Poly c;
            if (!MonicSqrt(R, q, &c)) return {"not_exact", {}, false, false};
            return {"local_charpoly_S_k", c, true, true};
        }
        return {"local_chi/(x-l)^r", q, false, true};
    };

    // ---- (2.6) on the theta-block at lambda (theta = the first candidate whose c(lambda) is read below K)
    {
        size_t pick = 0;
        for (size_t i = 0; i < cands.size(); ++i) {
            const Ann ai = annihilator(cands[i]);
            if (ai.ok && R.val(PolyEval(R, ai.c, lambda_of(cands[i].coeffs))) < M.K) {
                pick = i;
                break;
            }
        }
        const Cand& c0 = cands[pick];
        L.p26_op = c0.name;
        const Ann an = annihilator(c0);
        const uint64_t lam = lambda_of(c0.coeffs);
        if (!an.ok) {
            L.p26 = "not_exact";
        } else {
            const uint64_t cl = PolyEval(R, an.c, lam);
            const uint32_t v = R.val(cl);
            // the idempotent mod ell: chi-bar = (x - lam)^a h, p = w h with w h = 1 mod (x - lam)^a
            Mat th1 = c0.m;
            for (uint64_t& x : th1) x = F.red(x);
            Poly chi1 = Charpoly(F, th1, D);
            uint32_t a = 0;
            Poly h = chi1;
            for (;;) {
                Poly qd;
                if (!DivideByRoot(F, h, lam, 1, &qd)) break;
                h = qd;
                ++a;
            }
            if (a == 0 || v >= M.K) {
                L.p26 = a == 0 ? "none" : "open(precision)";
            } else {
                Poly s{1};
                for (uint32_t i = 0; i < a; ++i) s = PMul(F, s, Poly{F.neg(F.red(lam)), 1});
                const Poly w = PInvMod(F, h, s);
                ENGINE_CHECK_MSG(!w.empty(), "HeckeZLocal: the block polynomial is not invertible (coprimality defect)");
                const Poly pe = PMul(F, w, h);
                Mat e = MatPolyEval(R, pe, c0.m, D);
                for (int it = 0;; ++it) {
                    ENGINE_CHECK_MSG(it < 70, "HeckeZLocal: the idempotent lift did not converge");
                    const Mat e2 = MatMul(R, e, e, D);
                    if (e2 == e) break;
                    const Mat e3 = MatMul(R, e2, e, D);
                    e = MatAdd(R, MatScale(R, e2, 3), MatScale(R, e3, R.q - (2 % R.q)));
                }
                {
                    Mat e1 = e;
                    for (uint64_t& x : e1) x = F.red(x);
                    L.eis_block_rank = static_cast<uint32_t>(SmithValuations(F, e1, D, D).size());
                }
                const bool eis = !MatIsZero(MatMul(R, MatPolyEval(R, an.c, c0.m, D), e, D));
                const bool cusp = !MatIsZero(MatMul(R, MatMinusScalar(R, c0.m, lam, D), e, D));
                L.p26 = eis && cusp ? "eis+cusp" : eis ? "eis" : cusp ? "cusp" : "none";
            }
        }
    }

    // ---- (5.4) the at-root reads
    const uint32_t r = opt.r_eis;
    std::vector<uint32_t> eta_ref;
    bool have = false, inconsistent = false;
    int32_t v_t_ell = -1;            // v_ell c_ell(lambda_ell) of the T_ell read (audit F1), -1: not read below K
    const std::string t_ell_name = "T_" + std::to_string(M.ell);
    for (const Cand& cd : cands) {
        RootRead rr;
        rr.op = cd.name;
        const Ann an = annihilator(cd);
        rr.annihilator = an.kind;
        if (!an.ok) {
            rr.status = "not_exact";
            inconsistent = true;
            L.roots.push_back(rr);
            continue;
        }
        const uint64_t cl = PolyEval(R, an.c, lambda_of(cd.coeffs));
        const uint32_t v = R.val(cl);
        if (v >= M.K) {
            rr.status = "open(precision)";
            L.roots.push_back(rr);
            continue;
        }
        rr.v_c_lambda = static_cast<int32_t>(v);
        if (cd.name == "theta" && !opt.rat.lambda_hex.empty() && HexResidue(R, opt.rat.lambda_hex) != lambda_of(cd.coeffs)) {
            rr.status = "inconsistent(lambda_theta vs HZ-RAT)";
            inconsistent = true;
            L.roots.push_back(rr);
            continue;
        }
        if (cd.name == "theta" && opt.rat.v_c_lambda >= 0 && opt.rat.v_c_lambda != static_cast<int32_t>(v)) {
            rr.status = "inconsistent(v_c_lambda vs HZ-RAT)";
            inconsistent = true;
            L.roots.push_back(rr);
            continue;
        }
        if (an.sharp && cd.name != "theta") {
            if (L.v_gP_local < 0 || static_cast<int32_t>(v) < L.v_gP_local) L.v_gP_local = static_cast<int32_t>(v);
            if (cd.name == t_ell_name) v_t_ell = static_cast<int32_t>(v);
        }
        rr.a = SmithValuations(R, MatPolyEval(R, an.c, cd.m, D), D, D);
        if (rr.a.size() != r) {
            rr.status = "rank_mismatch";
            inconsistent = true;
            L.roots.push_back(rr);
            continue;
        }
        for (uint32_t ai : rr.a) {
            ENGINE_CHECK_MSG(ai <= v, "HeckeZLocal: a Smith valuation exceeds v(c(lambda)) (c(lambda) L_S is in c(T) L)");
            if (v > ai) rr.eta.push_back(v - ai);
        }
        std::sort(rr.eta.rbegin(), rr.eta.rend());
        rr.status = "ok";
        if (!have) {
            eta_ref = rr.eta;
            have = true;
        } else if (rr.eta != eta_ref) {
            inconsistent = true;
        }
        L.roots.push_back(rr);
    }
    if (opt.rat.present && opt.rat.v_gP >= 0) {
        // g_{P u {ell}} = gcd(g_P, c_ell(lambda_ell)) ell-wise (Prop. 5.3 holds for every P; audit F1)
        L.v_gP = opt.rat.v_gP;
        L.v_gP_source = "hz_rat";
        if (v_t_ell >= 0 && v_t_ell < L.v_gP) {
            L.v_gP = v_t_ell;
            L.v_gP_source = "hz_rat+T_ell";
        }
    } else if (L.v_gP_local >= 0) {
        L.v_gP = L.v_gP_local;
        L.v_gP_source = "local_charpoly";
    } else {
        L.v_gP_source = "none";
    }
    if (inconsistent) {
        L.root = "inconsistent";
    } else if (!have) {
        L.root = "open(precision)";
    } else {
        L.eta = eta_ref;
        L.v_eta = eta_ref.empty() ? 0 : static_cast<int32_t>(eta_ref.front());
        const std::string ell_s = std::to_string(M.ell);
        if (L.v_gP < 0) {
            L.root = "open(precision)";
        } else if (L.v_eta >= L.v_gP) {
            ENGINE_CHECK_MSG(L.v_eta == L.v_gP, "HeckeZLocal: v(exp eta_L) > v(g_P) contradicts exp eta_L | g_P");
            L.root = "closes";
            L.C_T = ell_s + "^" + std::to_string(L.v_gP) + "_exact";
        } else {
            L.root = "upper_bound_certified";
            L.C_T = ell_s + "^<=" + std::to_string(L.v_gP) + "_upper_bound(eta=" + ell_s + "^" + std::to_string(L.v_eta) + ")";
        }
    }
    return L;
}

}  // namespace local
}  // namespace heckez
