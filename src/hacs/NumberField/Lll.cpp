// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Lll.cpp -- see Lll.h (slot-21 T1). Indices are 0-based here; Cohen's are
// 1-based (his k = 2 is our k = 1, his d_0 = 1 is our d[0], his d_i our d[i]).
#include "Lll.h"

#include <utility>

#include "NfCheck.h"

namespace nf {

namespace {

// floor(a / b) for b > 0 (BigInt division truncates toward zero).
BigInt floor_div(const BigInt& a, const BigInt& b) {
    BigInt q, r;
    BigInt::divmod(a, b, q, r);
    if (r.is_neg()) q -= BigInt(1);
    return q;
}

// Exact division; the integral LLL's quotients are integers (Cohen Prop. 2.6.5).
BigInt div_exact(const BigInt& a, const BigInt& b) {
    BigInt q, r;
    BigInt::divmod(a, b, q, r);
    NF_DEBUG_CHECK_MSG(r.is_zero(), "integral LLL: inexact division");
    return q;
}

bool rat_less(const BigRational& a, const BigRational& b) { return a.num() * b.den() < b.num() * a.den(); }

void check_delta(const BigRational& delta) {
    // 1/4 < delta <= 1, i.e. 4 num > den and num <= den (den > 0).
    NF_CHECK_MSG(BigInt(4) * delta.num() > delta.den() && delta.num() <= delta.den(),
                 "LLL: delta must satisfy 1/4 < delta <= 1");
}

template <typename M>
void check_square_symmetric(const M& G) {
    const std::size_t n = G.size();
    for (std::size_t i = 0; i < n; ++i) {
        NF_CHECK_MSG(G[i].size() == n, "LLL: Gram matrix not square");
        for (std::size_t j = 0; j < i; ++j) NF_CHECK_MSG(G[i][j] == G[j][i], "LLL: Gram matrix not symmetric");
    }
}

IntMat identity(std::size_t n) {
    IntMat H(n, std::vector<BigInt>(n, BigInt(0)));
    for (std::size_t i = 0; i < n; ++i) H[i][i] = BigInt(1);
    return H;
}

// b_k . b_j for k > kmax (row k of H is still e_k): sum_m H[j][m] G[m][k].
template <typename T>
T dot_untouched(const IntMat& H, const std::vector<std::vector<T>>& G, std::size_t k, std::size_t j) {
    if (j == k) return G[k][k];
    T s = T(0);
    for (std::size_t m = 0; m < G.size(); ++m)
        if (!H[j][m].is_zero()) s += T(H[j][m]) * G[m][k];
    return s;
}

void row_axpy(std::vector<BigInt>& dst, const BigInt& q, const std::vector<BigInt>& src) {  // dst -= q src
    for (std::size_t m = 0; m < dst.size(); ++m)
        if (!src[m].is_zero()) dst[m] -= q * src[m];
}

}  // namespace

BigRational lll_default_delta() { return BigRational(BigInt(3), BigInt(4)); }

// ---------------------------------------------------------------- Alg. 2.6.3
LllResult lll_gram_rational(const RatMat& G, const BigRational& delta) {
    check_delta(delta);
    check_square_symmetric(G);
    const std::size_t n = G.size();
    LllResult R;
    R.H = identity(n);
    R.B.assign(n, BigRational(0));
    R.mu.assign(n, std::vector<BigRational>(n, BigRational(0)));
    if (n == 0) return R;
    IntMat& H = R.H;
    std::vector<BigRational>& B = R.B;
    RatMat& mu = R.mu;

    auto red = [&](std::size_t k, std::size_t l) {
        const BigRational& m = mu[k][l];
        // |mu| <= 1/2: nothing to do.
        if (BigInt(2) * m.num().abs() <= m.den()) return;
        const BigInt q = floor_div(BigInt(2) * m.num() + m.den(), BigInt(2) * m.den());  // floor(mu + 1/2)
        row_axpy(H[k], q, H[l]);
        mu[k][l] -= BigRational(q);
        for (std::size_t i = 0; i < l; ++i) mu[k][i] -= BigRational(q) * mu[l][i];
        ++R.reductions;
    };
    auto swap = [&](std::size_t k, std::size_t kmax) {
        std::swap(H[k], H[k - 1]);
        for (std::size_t j = 0; j + 1 < k; ++j) std::swap(mu[k][j], mu[k - 1][j]);
        const BigRational m = mu[k][k - 1];
        const BigRational Bn = B[k] + m * m * B[k - 1];
        mu[k][k - 1] = m * B[k - 1] / Bn;
        B[k] = B[k - 1] * B[k] / Bn;
        B[k - 1] = Bn;
        for (std::size_t i = k + 1; i <= kmax; ++i) {
            const BigRational t = mu[i][k];
            mu[i][k] = mu[i][k - 1] - m * t;
            mu[i][k - 1] = t + mu[k][k - 1] * mu[i][k];
        }
        ++R.swaps;
    };

    B[0] = G[0][0];
    NF_CHECK_MSG(B[0].num().sign() > 0, "LLL: the vectors do not form a basis (B_1 <= 0)");
    std::size_t k = 1, kmax = 0;
    std::vector<BigRational> a(n);
    while (k < n) {
        if (k > kmax) {  // step 2: incremental Gram--Schmidt (Remark (2) form)
            kmax = k;
            for (std::size_t j = 0; j < k; ++j) {
                BigRational s = dot_untouched(H, G, k, j);
                for (std::size_t i = 0; i < j; ++i) s -= mu[j][i] * a[i];
                a[j] = s;
                mu[k][j] = s / B[j];
            }
            BigRational s = G[k][k];
            for (std::size_t j = 0; j < k; ++j) s -= mu[k][j] * a[j];
            B[k] = s;
            NF_CHECK_MSG(s.num().sign() > 0, "LLL: the vectors do not form a basis (B_k <= 0)");
        }
        // step 3: the Lovasz condition
        red(k, k - 1);
        if (rat_less(B[k], (delta - mu[k][k - 1] * mu[k][k - 1]) * B[k - 1])) {
            swap(k, kmax);
            if (k > 1) --k;
        } else {
            for (std::size_t l = k - 1; l-- > 0;) red(k, l);
            ++k;
        }
    }
    return R;
}

// ---------------------------------------------------------------- Alg. 2.6.7
LllResult lll_gram_integral(const IntMat& G, const BigRational& delta) {
    check_delta(delta);
    check_square_symmetric(G);
    const std::size_t n = G.size();
    LllResult R;
    R.H = identity(n);
    R.B.assign(n, BigRational(0));
    R.mu.assign(n, std::vector<BigRational>(n, BigRational(0)));
    if (n == 0) return R;
    IntMat& H = R.H;
    // d[0] = 1, d[i + 1] = d_i of vector i; lam[i][j] = lambda_{ij} (j < i).
    std::vector<BigInt> d(n + 1, BigInt(0));
    IntMat lam(n, std::vector<BigInt>(n, BigInt(0)));
    const BigInt& da = delta.num();
    const BigInt& db = delta.den();

    auto redi = [&](std::size_t k, std::size_t l) {
        const BigInt& dl = d[l + 1];
        if (BigInt(2) * lam[k][l].abs() <= dl) return;
        const BigInt q = floor_div(BigInt(2) * lam[k][l] + dl, BigInt(2) * dl);
        row_axpy(H[k], q, H[l]);
        lam[k][l] -= q * dl;
        for (std::size_t i = 0; i < l; ++i) lam[k][i] -= q * lam[l][i];
        ++R.reductions;
    };
    auto swapi = [&](std::size_t k, std::size_t kmax) {
        std::swap(H[k], H[k - 1]);
        for (std::size_t j = 0; j + 1 < k; ++j) std::swap(lam[k][j], lam[k - 1][j]);
        const BigInt l = lam[k][k - 1];
        // Cohen's d_{k-2}, d_{k-1}, d_k are our d[k-1], d[k], d[k+1].
        const BigInt Bn = div_exact(d[k - 1] * d[k + 1] + l * l, d[k]);
        for (std::size_t i = k + 1; i <= kmax; ++i) {
            const BigInt t = lam[i][k];
            lam[i][k] = div_exact(d[k + 1] * lam[i][k - 1] - l * t, d[k]);
            lam[i][k - 1] = div_exact(Bn * t + l * lam[i][k], d[k + 1]);
        }
        d[k] = Bn;
        ++R.swaps;
    };

    d[0] = BigInt(1);
    d[1] = G[0][0];
    NF_CHECK_MSG(d[1].sign() > 0, "LLL: the vectors do not form a basis (d_1 <= 0)");
    std::size_t k = 1, kmax = 0;
    while (k < n) {
        if (k > kmax) {  // step 2
            kmax = k;
            for (std::size_t j = 0; j <= k; ++j) {
                BigInt u = dot_untouched(H, G, k, j);
                for (std::size_t i = 0; i < j; ++i) u = div_exact(d[i + 1] * u - lam[k][i] * lam[j][i], d[i]);
                if (j < k) lam[k][j] = u;
                else d[k + 1] = u;
            }
            NF_CHECK_MSG(d[k + 1].sign() > 0, "LLL: the vectors do not form a basis (d_k <= 0)");
        }
        // step 3: d_k d_{k-2} < delta d_{k-1}^2 - lambda^2, times delta's denominator
        redi(k, k - 1);
        const BigInt& lk = lam[k][k - 1];
        if (db * d[k + 1] * d[k - 1] < da * d[k] * d[k] - db * lk * lk) {
            swapi(k, kmax);
            if (k > 1) --k;
        } else {
            for (std::size_t l = k - 1; l-- > 0;) redi(k, l);
            ++k;
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        R.B[i] = BigRational(d[i + 1], d[i]);
        for (std::size_t j = 0; j < i; ++j) R.mu[i][j] = BigRational(lam[i][j], d[j + 1]);
    }
    return R;
}

// ---------------------------------------------------------------- helpers
IntMat gram_of_rows(const IntMat& rows) {
    const std::size_t n = rows.size();
    IntMat G(n, std::vector<BigInt>(n, BigInt(0)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            NF_CHECK(rows[i].size() == rows[j].size());
            BigInt s(0);
            for (std::size_t m = 0; m < rows[i].size(); ++m) s += rows[i][m] * rows[j][m];
            G[i][j] = s;
            G[j][i] = s;
        }
    return G;
}

IntMat int_mat_mul(const IntMat& A, const IntMat& B) {
    const std::size_t r = A.size(), inner = B.size(), c = B.empty() ? 0 : B[0].size();
    IntMat C(r, std::vector<BigInt>(c, BigInt(0)));
    for (std::size_t i = 0; i < r; ++i) {
        NF_CHECK(A[i].size() == inner);
        for (std::size_t t = 0; t < inner; ++t) {
            if (A[i][t].is_zero()) continue;
            for (std::size_t j = 0; j < c; ++j)
                if (!B[t][j].is_zero()) C[i][j] += A[i][t] * B[t][j];
        }
    }
    return C;
}

namespace {
template <typename T>
std::vector<std::vector<T>> transform_gram_t(const IntMat& H, const std::vector<std::vector<T>>& G) {
    const std::size_t n = H.size(), m = G.size();
    std::vector<std::vector<T>> HG(n, std::vector<T>(m, T(0)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t t = 0; t < m; ++t)
            if (!H[i][t].is_zero())
                for (std::size_t j = 0; j < m; ++j) HG[i][j] += T(H[i][t]) * G[t][j];
    std::vector<std::vector<T>> R(n, std::vector<T>(n, T(0)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            T s(0);
            for (std::size_t t = 0; t < m; ++t)
                if (!H[j][t].is_zero()) s += HG[i][t] * T(H[j][t]);
            R[i][j] = s;
            R[j][i] = s;
        }
    return R;
}
}  // namespace

IntMat transform_gram(const IntMat& H, const IntMat& G) { return transform_gram_t(H, G); }
RatMat transform_gram(const IntMat& H, const RatMat& G) { return transform_gram_t(H, G); }

BigInt det_bareiss(IntMat A) {
    const std::size_t n = A.size();
    if (n == 0) return BigInt(1);
    for (const auto& r : A) NF_CHECK(r.size() == n);
    BigInt prev(1);
    int sign = 1;
    for (std::size_t k = 0; k + 1 < n; ++k) {
        if (A[k][k].is_zero()) {
            std::size_t p = k + 1;
            while (p < n && A[p][k].is_zero()) ++p;
            if (p == n) return BigInt(0);
            std::swap(A[k], A[p]);
            sign = -sign;
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            for (std::size_t j = k + 1; j < n; ++j) A[i][j] = div_exact(A[i][j] * A[k][k] - A[i][k] * A[k][j], prev);
            A[i][k] = BigInt(0);
        }
        prev = A[k][k];
    }
    return sign > 0 ? A[n - 1][n - 1] : -A[n - 1][n - 1];
}

LllBasisResult lll_basis(const IntMat& rows, const BigRational& delta) {
    LllBasisResult out;
    out.lll = lll_gram_integral(gram_of_rows(rows), delta);
    out.basis = int_mat_mul(out.lll.H, rows);
    return out;
}

bool lll_is_reduced(const RatMat& G, const BigRational& delta, std::string* why) {
    check_square_symmetric(G);
    const std::size_t n = G.size();
    auto fail = [&](const std::string& s) {
        if (why) *why = s;
        return false;
    };
    // Plain Gram--Schmidt from G: a[i][j] = <b_i, b*_j>, mu = a / B.
    RatMat mu(n, std::vector<BigRational>(n, BigRational(0)));
    std::vector<BigRational> B(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            BigRational s = G[i][j];
            for (std::size_t t = 0; t < j; ++t) s -= mu[j][t] * mu[i][t] * B[t];
            mu[i][j] = s / B[j];
        }
        BigRational s = G[i][i];
        for (std::size_t t = 0; t < i; ++t) s -= mu[i][t] * mu[i][t] * B[t];
        B[i] = s;
        if (s.num().sign() <= 0) return fail("B_" + std::to_string(i + 1) + " <= 0 (not positive definite)");
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (BigInt(2) * mu[i][j].num().abs() > mu[i][j].den())
                return fail("|mu_" + std::to_string(i + 1) + "," + std::to_string(j + 1) + "| > 1/2");
    for (std::size_t i = 1; i < n; ++i)
        if (rat_less(B[i], (delta - mu[i][i - 1] * mu[i][i - 1]) * B[i - 1]))
            return fail("Lovasz condition fails at i = " + std::to_string(i + 1));
    return true;
}

bool lll_is_reduced(const IntMat& G, const BigRational& delta, std::string* why) {
    RatMat Q(G.size());
    for (std::size_t i = 0; i < G.size(); ++i)
        for (const auto& x : G[i]) Q[i].push_back(BigRational(x));
    return lll_is_reduced(Q, delta, why);
}

}  // namespace nf
