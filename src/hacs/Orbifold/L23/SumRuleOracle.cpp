// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, L23-3) the level-1 q-expansion sum-rule oracle -- see SumRuleOracle.h.

#include "SumRuleOracle.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <utility>

#include "../../NumberField/BigInt.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace l23 {

uint32_t dim_cusp_forms(uint32_t k) {
    if (k % 2 || k < 12) return 0;
    return k / 12 - ((k % 12 == 2) ? 1u : 0u);
}

namespace {

using nf::BigInt;
using Series = std::vector<BigInt>;

// Residues modulo M = ell^K, kept in [0, M).
struct ModRing {
    uint32_t ell = 0, K = 0;
    BigInt M;
    ModRing(uint32_t ell_, uint32_t K_) : ell(ell_), K(K_), M(BigInt::pow(BigInt(static_cast<int64_t>(ell_)), K_)) {}
    BigInt red(const BigInt& x) const { return x.floor_mod(M); }
    // v_ell(x) for a residue, capped at K (0 -> K).
    uint32_t val(const BigInt& x) const {
        if (x.is_zero()) return K;
        uint32_t v = 0;
        if (ell == 2) {
            for (uint32_t limb : x.magnitude()) {
                if (limb == 0) { v += 32; continue; }
                v += static_cast<uint32_t>(__builtin_ctz(limb));
                break;
            }
        } else {
            constexpr uint32_t kChunk = 3486784401u;   // 3^20
            BigInt y = x;
            while (y.mod_u32(kChunk) == 0) { y = y.div_exact_u32(kChunk); v += 20; }
            while (y.mod_u32(ell) == 0) { y = y.div_exact_u32(ell); ++v; }
        }
        return std::min(v, K);
    }
    BigInt pow_mod(uint64_t b, uint64_t e) const {
        BigInt r(1), x = red(BigInt::from_u64(b));
        while (e) {
            if (e & 1u) r = red(r * x);
            e >>= 1;
            if (e) x = red(x * x);
        }
        return r;
    }
    // u^-1 mod M for a unit u (Newton: x <- x (2 - u x), precision doubling from mod ell).
    BigInt inv_unit(const BigInt& u) const {
        const uint32_t u0 = u.mod_u32(ell);
        ENGINE_CHECK_MSG(u0 != 0, "l23::sum_rule_oracle: inverting a non-unit");
        BigInt x(static_cast<int64_t>(ell == 2 ? 1u : (u0 == 1 ? 1u : 2u)));   // u0^-1 mod ell (ell in {2, 3})
        for (uint32_t prec = 1; prec < K; prec *= 2) x = red(x * red(BigInt(2) - red(u * x)));
        return x;
    }
};

// c = a * b truncated to N terms, mod M (accumulate, reduce once per coefficient).
Series mul(const ModRing& R, const Series& a, const Series& b, size_t N) {
    size_t a0 = 0, b0 = 0;
    while (a0 < a.size() && a[a0].is_zero()) ++a0;
    while (b0 < b.size() && b[b0].is_zero()) ++b0;
    Series c(N);
    for (size_t m = a0 + b0; m < N; ++m) {
        BigInt acc;
        for (size_t i = a0; i + b0 <= m && i < a.size(); ++i) {
            const size_t j = m - i;
            if (j >= b.size() || a[i].is_zero() || b[j].is_zero()) continue;
            acc += a[i] * b[j];
        }
        c[m] = R.red(acc);
    }
    return c;
}

uint64_t sigma_r(uint32_t r, uint64_t m) {
    uint64_t s = 0;
    for (uint64_t d = 1; d <= m; ++d)
        if (m % d == 0) {
            uint64_t t = 1;
            for (uint32_t i = 0; i < r; ++i) t *= d;
            s += t;
        }
    return s;
}

// Smith valuations of a square matrix over Z/ell^K by global minimum-valuation pivoting (exact for every invariant < K;
// an all-zero remainder contributes K's).  Rows are consumed in place.
std::vector<uint32_t> smith_vals(const ModRing& R, std::vector<Series> A) {
    const size_t d = A.size();
    std::vector<uint32_t> out;
    std::vector<bool> row_done(d, false), col_done(d, false);
    for (size_t step = 0; step < d; ++step) {
        uint32_t best = R.K;
        size_t bi = d, bj = d;
        for (size_t i = 0; i < d && best > 0; ++i) {
            if (row_done[i]) continue;
            for (size_t j = 0; j < d; ++j) {
                if (col_done[j]) continue;
                const uint32_t v = R.val(A[i][j]);
                if (v < best) { best = v; bi = i; bj = j; if (v == 0) break; }
            }
        }
        if (bi == d) {   // the remainder is zero mod ell^K
            for (size_t t = step; t < d; ++t) out.push_back(R.K);
            break;
        }
        out.push_back(best);
        row_done[bi] = col_done[bj] = true;
        const BigInt lv = BigInt::pow(BigInt(static_cast<int64_t>(R.ell)), best);
        const BigInt uinv = R.inv_unit(A[bi][bj] / lv);
        for (size_t r = 0; r < d; ++r) {
            if (row_done[r] || A[r][bj].is_zero()) continue;
            const BigInt f = R.red((A[r][bj] / lv) * uinv);   // exact: v(A[r][bj]) >= best
            for (size_t j = 0; j < d; ++j) {
                if (col_done[j] && j != bj) continue;
                if (A[bi][j].is_zero()) continue;
                A[r][j] = R.red(A[r][j] - f * A[bi][j]);
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// One attempt at precision K; returns false if some invariant reached K.
bool attempt(uint32_t ell, uint32_t n, const std::vector<uint32_t>& primes, uint32_t K, uint32_t mutate, SumRuleRow& row) {
    const ModRing R(ell, K);
    const uint32_t k = n + 2, d = dim_cusp_forms(k);
    row.max_invariant = 0;
    row.vdet_tell = row.vdet_tell_minus_1 = 0;
    row.vdet_tp.clear();
    row.vdet_eis.clear();
    row.v_pn1.clear();
    bool ok = true;
    for (uint32_t p : primes) {
        if (p == ell) continue;
        // v(p^n - 1) (p^n mod ell^K; capped at K)
        const BigInt x = R.red(R.pow_mod(p, n) - BigInt(1));
        const uint32_t v = R.val(x);
        if (v >= K) ok = false;
        row.v_pn1[p] = v;
    }
    if (d == 0) {   // det of the empty matrix: 1
        for (uint32_t p : primes) if (p != ell) { row.vdet_tp[p] = 0; row.vdet_eis[p] = 0; }
        return ok;
    }
    uint32_t pmax = ell;
    for (uint32_t p : primes) pmax = std::max(pmax, p);
    const size_t N = static_cast<size_t>(pmax) * d + 1;
    // E_4, E_6, Delta = q prod (1 - q^m)^24
    Series E4(N), E6(N);
    for (size_t m = 0; m < N; ++m) {
        E4[m] = m == 0 ? BigInt(1) : R.red(BigInt::from_u64(240) * BigInt::from_u64(sigma_r(3, m)));
        E6[m] = m == 0 ? BigInt(1) : R.red(BigInt(-504) * BigInt::from_u64(sigma_r(5, m)));
    }
    std::vector<int64_t> P(N, 0);
    P[0] = 1;
    for (size_t m = 1; m < N; ++m)
        for (size_t i = N - 1; i >= m; --i) P[i] -= P[i - m];   // Euler: coefficients in {-1, 0, 1}
    Series Pb(N);
    for (size_t i = 0; i < N; ++i) Pb[i] = R.red(BigInt(P[i]));
    const Series P2 = mul(R, Pb, Pb, N), P4 = mul(R, P2, P2, N), P8 = mul(R, P4, P4, N), P16 = mul(R, P8, P8, N);
    const Series P24 = mul(R, P16, P8, N);
    Series D(N);
    for (size_t i = 1; i < N; ++i) D[i] = P24[i - 1];
    const uint32_t r = k - 12 * d;
    const uint32_t b = (r % 4 == 0) ? 0 : 1, a = (r - 6 * b) / 4;
    ENGINE_CHECK_MSG(4 * a + 6 * b == r, "l23::sum_rule_oracle: weight split");
    Series one(N);
    one[0] = BigInt(1);
    Series G = one;
    for (uint32_t i = 0; i < a; ++i) G = mul(R, G, E4, N);
    if (b) G = mul(R, G, E6, N);
    const Series E43 = mul(R, mul(R, E4, E4, N), E4, N);
    // Y_i = E_4^{3i} G (i = 0 .. d - 1), Delta^j (j = 1 .. d), f_j = Delta^j Y_{d-j}
    std::vector<Series> Y(d);
    Y[0] = G;
    for (uint32_t i = 1; i < d; ++i) Y[i] = mul(R, Y[i - 1], E43, N);
    std::vector<Series> f(d);
    Series Dj = one;
    for (uint32_t j = 1; j <= d; ++j) {
        Dj = mul(R, Dj, D, N);
        f[j - 1] = mul(R, Dj, Y[d - j], N);
    }
    for (uint32_t i = 0; i < d; ++i) {
        for (uint32_t t = 0; t <= i; ++t) ENGINE_CHECK_MSG(f[i][t].is_zero() || t == i + 1, "l23::sum_rule_oracle: basis not unitriangular");
        ENGINE_CHECK_MSG(f[i][i + 1] == BigInt(1), "l23::sum_rule_oracle: leading coefficient");
    }
    // echelonise on q^1 .. q^d (descending: f_j, j > i, is already reduced)
    for (uint32_t i = d; i-- > 0;)
        for (uint32_t j = i + 1; j < d; ++j) {
            const BigInt c = f[i][j + 1];
            if (c.is_zero()) continue;
            for (size_t t = 0; t < N; ++t)
                if (!f[j][t].is_zero()) f[i][t] = R.red(f[i][t] - c * f[j][t]);
        }
    auto hecke = [&](uint32_t p) {
        const BigInt pk = R.pow_mod(p, k - 1);
        std::vector<Series> Mt(d, Series(d));
        for (uint32_t i = 0; i < d; ++i)
            for (uint32_t m = 1; m <= d; ++m) {
                BigInt v = f[i][static_cast<size_t>(p) * m];
                if (m % p == 0 && mutate != 1) v += pk * f[i][m / p];
                Mt[i][m - 1] = R.red(v);
            }
        return Mt;
    };
    auto shifted = [&](std::vector<Series> Mt, const BigInt& c) {
        for (uint32_t i = 0; i < d; ++i) Mt[i][i] = R.red(Mt[i][i] - c);
        return Mt;
    };
    auto vdet = [&](std::vector<Series> Mt) {
        const std::vector<uint32_t> s = smith_vals(R, std::move(Mt));
        uint32_t sum = 0;
        for (uint32_t x : s) {
            if (x >= K) ok = false;
            row.max_invariant = std::max(row.max_invariant, x);
            sum += x;
        }
        return sum;
    };
    {
        const std::vector<Series> Tl = hecke(ell);
        row.vdet_tell = vdet(Tl);
        row.vdet_tell_minus_1 = vdet(shifted(Tl, BigInt(1)));
    }
    for (uint32_t p : primes) {
        if (p == ell) continue;
        const std::vector<Series> Tp = hecke(p);
        row.vdet_tp[p] = vdet(shifted(Tp, BigInt(static_cast<int64_t>(1 + p))));
        row.vdet_eis[p] = vdet(shifted(Tp, R.red(BigInt(1) + R.pow_mod(p, n + 1))));
    }
    return ok;
}

} // namespace

SumRuleRow sum_rule_oracle(uint32_t ell, uint32_t n, const std::vector<uint32_t>& primes, uint32_t k_start, uint32_t k_cap,
                           uint32_t mutate) {
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "l23::sum_rule_oracle: ell in {2, 3}");
    for (uint32_t p : primes) ENGINE_CHECK_MSG(p >= 2 && p < 65536, "l23::sum_rule_oracle: primes in [2, 2^16)");
    const auto t0 = std::chrono::steady_clock::now();
    SumRuleRow row;
    row.ell = ell;
    row.n = n;
    row.k = n + 2;
    row.dim_s = dim_cusp_forms(n + 2);
    uint32_t K = k_start ? k_start : std::max<uint32_t>(64, 2 * ((n + 1) / (ell + 1)) + 64);
    K = std::min(K, k_cap);
    for (;;) {
        const auto ta = std::chrono::steady_clock::now();
        row.resolved = attempt(ell, n, primes, K, mutate, row);
        row.K = K;
        ++row.attempts;
        char buf[256];
        std::snprintf(buf, sizeof(buf), "SUMRULE_ORACLE ell=%u n=%u dimS=%u K=%u max_invariant=%u -> %s (%.3f s)", ell, n, row.dim_s, K,
                      row.max_invariant, row.resolved ? "resolved" : (K >= k_cap ? "UNRESOLVED at k_cap" : "double K"),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count());
        row.log.emplace_back(buf);
        if (row.resolved || K >= k_cap) break;
        K = std::min(2 * K, k_cap);
    }
    row.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return row;
}

} // namespace l23
