// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Polred.cpp -- see Polred.h (slot-21 T1).
#include "Polred.h"

#include <algorithm>

#include "NfCheck.h"

namespace nf {

namespace {

void check_monic(const IntPoly& f) {
    NF_CHECK_MSG(f.size() >= 2 && f.back() == BigInt(1), "polred: f must be monic of degree >= 1");
}

BigInt div_exact(const BigInt& a, const BigInt& b) {
    BigInt q, r;
    BigInt::divmod(a, b, q, r);
    NF_CHECK_MSG(r.is_zero(), "polred: inexact division (element not integral?)");
    return q;
}

// Lexicographic comparison of v(Q) = (|disc|, size, |a_{n-1}|, ..., |a_0|).
int compare_v(const PolredCandidate& a, const PolredCandidate& b) {
    if (int c = compare(a.disc.abs(), b.disc.abs())) return c;
    if (int c = compare(a.size, b.size)) return c;
    const std::size_t n = a.charpoly.size() - 1;
    for (std::size_t i = n; i-- > 0;)
        if (int c = compare(a.charpoly[i].abs(), b.charpoly[i].abs())) return c;
    return 0;
}

}  // namespace

std::vector<BigInt> power_sums(const IntPoly& f, unsigned N) {
    check_monic(f);
    const std::size_t n = f.size() - 1;
    std::vector<BigInt> p(N + 1, BigInt(0));
    p[0] = BigInt(static_cast<int64_t>(n));
    // f = x^n + a_{n-1} x^{n-1} + ... + a_0:  p_k + a_{n-1} p_{k-1} + ... = -k a_{n-k} (k <= n), 0 (k > n).
    for (unsigned k = 1; k <= N; ++k) {
        BigInt s(0);
        for (std::size_t i = 1; i <= n && i < k; ++i) s += f[n - i] * p[k - i];
        if (k <= n) s += BigInt(static_cast<int64_t>(k)) * f[n - k];
        p[k] = -s;
    }
    return p;
}

IntMat trace_form_gram(const IntPoly& f) {
    check_monic(f);
    const std::size_t n = f.size() - 1;
    const std::vector<BigInt> p = power_sums(f, static_cast<unsigned>(2 * n - 2));
    IntMat G(n, std::vector<BigInt>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) G[i][j] = p[i + j];
    return G;
}

BigInt poly_discriminant(const IntPoly& f) { return det_bareiss(trace_form_gram(f)); }

IntPoly poly_mulmod_monic(const IntPoly& a, const IntPoly& b, const IntPoly& f) {
    check_monic(f);
    const std::size_t n = f.size() - 1;
    if (a.empty() || b.empty()) return IntPoly(n, BigInt(0));
    IntPoly c(a.size() + b.size() - 1, BigInt(0));
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].is_zero()) continue;
        for (std::size_t j = 0; j < b.size(); ++j)
            if (!b[j].is_zero()) c[i + j] += a[i] * b[j];
    }
    for (std::size_t k = c.size(); k-- > n;) {  // x^k = x^{k-n} (x^n) = -x^{k-n} (f - x^n)
        if (c[k].is_zero()) continue;
        const BigInt t = c[k];
        for (std::size_t i = 0; i < n; ++i)
            if (!f[i].is_zero()) c[k - n + i] -= t * f[i];
        c[k] = BigInt(0);
    }
    c.resize(n, BigInt(0));
    return c;
}

IntPoly charpoly_element(const IntPoly& f, const IntPoly& h, const BigInt& den) {
    check_monic(f);
    NF_CHECK_MSG(den.sign() > 0, "charpoly_element: den must be positive");
    const std::size_t n = f.size() - 1;
    NF_CHECK_MSG(h.size() <= n, "charpoly_element: deg h >= deg f");
    IntPoly g = h;
    g.resize(n, BigInt(0));
    const std::vector<BigInt> pf = power_sums(f, static_cast<unsigned>(n - 1));
    auto trace = [&](const IntPoly& x) {
        BigInt s(0);
        for (std::size_t i = 0; i < n; ++i)
            if (!x[i].is_zero()) s += x[i] * pf[i];
        return s;
    };
    // P_k = Tr(h^k), then e_k by Newton: k e_k = sum_{i=1..k} (-1)^{i-1} e_{k-i} P_i (e_k of h(alpha), integral).
    std::vector<BigInt> P(n + 1, BigInt(0)), e(n + 1, BigInt(0));
    IntPoly pw = g;
    for (std::size_t k = 1; k <= n; ++k) {
        if (k > 1) pw = poly_mulmod_monic(pw, g, f);
        P[k] = trace(pw);
    }
    e[0] = BigInt(1);
    for (std::size_t k = 1; k <= n; ++k) {
        BigInt s(0);
        for (std::size_t i = 1; i <= k; ++i) {
            const BigInt t = e[k - i] * P[i];
            if (i % 2 == 1) s += t;
            else s -= t;
        }
        e[k] = div_exact(s, BigInt(static_cast<int64_t>(k)));
    }
    // charpoly of h/den: y^n + sum_k (-1)^k (e_k / den^k) y^{n-k}.
    IntPoly c(n + 1, BigInt(0));
    BigInt dk(1);
    for (std::size_t k = 0; k <= n; ++k) {
        if (k > 0) dk *= den;
        const BigInt ek = div_exact(e[k], dk);
        c[n - k] = (k % 2 == 1) ? -ek : ek;
    }
    return c;
}

IntPoly polred_sign_normalise(IntPoly P) {
    NF_CHECK(!P.empty() && P.back() == BigInt(1));
    const std::size_t n = P.size() - 1;
    for (std::size_t d = n; d-- > 0;) {
        if ((n - d) % 2 == 0 || P[d].is_zero()) continue;
        if (P[d].sign() > 0)  // (-1)^n P(-X): negate the coefficients of degree d with d != n (mod 2)
            for (std::size_t j = 0; j < n; ++j)
                if ((n - j) % 2 == 1) P[j] = -P[j];
        break;
    }
    return P;
}

PolredResult polred(const IntPoly& f, const IntMat& omega_num, const std::vector<BigInt>& omega_den, const IntMat& gram,
                    const BigRational& delta) {
    check_monic(f);
    const std::size_t n = f.size() - 1;
    NF_CHECK_MSG(omega_num.size() == n && omega_den.size() == n && gram.size() == n, "polred: need n basis elements");
    BigInt D(1);  // lcm of the denominators
    for (const BigInt& d : omega_den) {
        NF_CHECK(d.sign() > 0);
        D = D / BigInt::gcd(D, d) * d;
    }
    PolredResult R;
    R.lll = lll_gram_integral(gram, delta);
    const IntMat red = transform_gram(R.lll.H, gram);
    for (std::size_t i = 0; i < n; ++i) {
        PolredCandidate c;
        c.coords = R.lll.H[i];
        c.num.assign(n, BigInt(0));
        for (std::size_t j = 0; j < n; ++j) {
            if (c.coords[j].is_zero()) continue;
            NF_CHECK(omega_num[j].size() <= n);
            const BigInt s = c.coords[j] * (D / omega_den[j]);
            for (std::size_t t = 0; t < omega_num[j].size(); ++t) c.num[t] += s * omega_num[j][t];
        }
        BigInt g = D;  // reduce num / D
        for (const BigInt& x : c.num) g = BigInt::gcd(g, x);
        c.den = D / g;
        for (BigInt& x : c.num) x /= g;
        c.size = red[i][i];
        c.charpoly = charpoly_element(f, c.num, c.den);
        c.disc = poly_discriminant(c.charpoly);
        c.generates = !c.disc.is_zero();
        if (c.generates) c.normalised = polred_sign_normalise(c.charpoly);
        R.cands.push_back(std::move(c));
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!R.cands[i].generates) continue;
        if (R.best < 0 || compare_v(R.cands[i], R.cands[static_cast<std::size_t>(R.best)]) < 0)
            R.best = static_cast<int>(i);
    }
    return R;
}

PolredResult polred_power_basis(const IntPoly& f, const IntMat& gram, const BigRational& delta) {
    check_monic(f);
    const std::size_t n = f.size() - 1;
    IntMat w(n, std::vector<BigInt>(n, BigInt(0)));
    for (std::size_t i = 0; i < n; ++i) w[i][i] = BigInt(1);
    return polred(f, w, std::vector<BigInt>(n, BigInt(1)), gram, delta);
}

}  // namespace nf
