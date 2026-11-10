// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 21, P9) the two-variable measure by interpolation: the arithmetic of
// HeckeLambdaTwoVar.h (branch elements, Lagrange weights at a point of the disc, the zero of a
// Lambda-function, 11a's Manin symbols, Tate's L-invariant).  Spec: docs/implementation/slot21/P9_spec.md.
#include "HeckeLambdaTwoVar.h"

#include <algorithm>
#include <numeric>

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "HeckeLambdaQExp.h"

namespace hecke_ext {
namespace lambda {
namespace twovar {

namespace bm = hecke_ext::lvalue::birch;

namespace {
u128 MulMod(u128 a, u128 b, u128 q) {
    // a, b < q < 2^63
    return (a * b) % q;
}
using engine::scalar::pow_wrap;
}  // namespace

uint64_t Teich(uint64_t d, uint64_t ell, uint32_t e) {
    ENGINE_CHECK_MSG(d % ell != 0 && e >= 1, "twovar::Teich: d a unit, e >= 1");
    const u128 q = pow_wrap(ell, e);
    u128 x = d % q;
    for (uint32_t i = 0; i < e + 1; ++i) {
        u128 y = 1, b = x;
        uint64_t k = ell;
        while (k) {
            if (k & 1) y = MulMod(y, b, q);
            b = MulMod(b, b, q);
            k >>= 1;
        }
        x = y;
    }
    return static_cast<uint64_t>(x);
}

std::vector<uint64_t> BranchFromBalls(const std::function<uint64_t(uint64_t)>& mass, uint64_t ell, uint32_t M, uint32_t a, uint32_t Kp) {
    ENGINE_CHECK_MSG(M >= 1, "twovar::BranchFromBalls: M >= 1");
    const bm::Ring R = bm::MakeRing(ell, Kp);
    const uint64_t MM = pow_wrap(ell, M), G = pow_wrap(ell, M - 1), u = 1 + ell;
    std::vector<uint64_t> deltas(ell, 0), wts(ell, 0);
    for (uint64_t d = 1; d < ell; ++d) {
        deltas[d] = Teich(d, ell, M) % MM;
        wts[d] = R.pow(Teich(d, ell, Kp) % R.q, a % (ell - 1));
    }
    std::vector<uint64_t> g(G, 0);
    uint64_t ui = 1 % MM;
    for (uint64_t i = 0; i < G; ++i) {
        uint64_t s = 0;
        for (uint64_t d = 1; d < ell; ++d)
            s = R.add(s, R.mul(wts[d], mass(static_cast<uint64_t>(static_cast<u128>(deltas[d]) * ui % MM)) % R.q));
        g[i] = s;
        ui = static_cast<uint64_t>(static_cast<u128>(ui) * u % MM);
    }
    return g;
}

std::vector<uint64_t> TaylorShift(const std::vector<uint64_t>& g, uint64_t q) {
    const size_t G = g.size();
    std::vector<uint64_t> c(G, 0), row(G, 0);   // row = C(i, .) mod q
    for (size_t i = 0; i < G; ++i) {
        // Pascal: C(i, t) = C(i-1, t) + C(i-1, t-1)
        for (size_t t = i; t >= 1; --t) row[t] = (row[t] + row[t - 1]) % q;
        row[0] = 1 % q;
        for (size_t t = 0; t <= i; ++t) c[t] = static_cast<uint64_t>((c[t] + static_cast<u128>(row[t]) * (g[i] % q)) % q);
    }
    return c;
}

u128 XOf(uint64_t ell, uint64_t k, uint32_t e) { return XAt(ell, k, e); }

std::vector<uint64_t> LagrangeAt(uint64_t ell, const std::vector<uint32_t>& weights, uint32_t Kp, u128 y, uint32_t ey, bool deriv) {
    const size_t N = weights.size();
    ENGINE_CHECK_MSG(N >= 2 && N <= ell, "twovar::LagrangeAt: 2 <= N <= ell");
    for (size_t j = 1; j < N; ++j)
        ENGINE_CHECK_MSG(weights[j] == weights[0] + j * (ell - 1), "twovar::LagrangeAt REFUSED: weights not consecutive in one class");
    const uint32_t W = Kp + static_cast<uint32_t>(N) - 1 + (deriv ? 1 : 0);
    const ZMod Q(ell, W), RK(ell, Kp);
    std::vector<u128> xs;
    for (uint32_t k : weights) xs.push_back(XAt(ell, k, W));
    const u128 yy = (ey >= W ? y % Q.M : y % Q.pw[ey]) % Q.M;
    ENGINE_CHECK_MSG(Q.sub(yy, xs[0]) % ell == 0, "twovar::LagrangeAt: y not in the disc of the class");
    std::vector<uint64_t> out(N, 0);
    const uint32_t Nm1 = static_cast<uint32_t>(N) - 1;
    for (size_t j = 0; j < N; ++j) {
        u128 den = 1 % Q.M;
        for (size_t b = 0; b < N; ++b)
            if (b != j) den = Q.mul(den, Q.sub(xs[j], xs[b]));
        ENGINE_CHECK_MSG(Q.val(den, W) == Nm1, "twovar::LagrangeAt: a denominator of valuation != N - 1");
        const u128 uinv = RK.inv((den / Q.pw[Nm1]) % RK.M);
        u128 num = 0;
        uint32_t shift = Nm1;
        if (!deriv) {
            num = 1 % Q.M;
            for (size_t b = 0; b < N; ++b)
                if (b != j) num = Q.mul(num, Q.sub(yy, xs[b]));
        } else {
            shift = Nm1 - 1;   // ell * lambda' = (num' / ell^{N-2}) / (den / ell^{N-1})
            for (size_t c = 0; c < N; ++c) {
                if (c == j) continue;
                u128 p = 1 % Q.M;
                for (size_t b = 0; b < N; ++b)
                    if (b != j && b != c) p = Q.mul(p, Q.sub(yy, xs[b]));
                num = Q.add(num, p);
            }
        }
        ENGINE_CHECK_MSG(num == 0 || Q.val(num, W) >= shift, "twovar::LagrangeAt: a numerator of too small a valuation");
        out[j] = static_cast<uint64_t>(RK.mul((num / Q.pw[shift]) % RK.M, uinv));
    }
    return out;
}

uint64_t Combine(const std::vector<uint64_t>& w, const std::vector<uint64_t>& v, uint64_t ell, uint32_t Kp) {
    ENGINE_CHECK_MSG(w.size() == v.size(), "twovar::Combine: shape");
    const bm::Ring R = bm::MakeRing(ell, Kp);
    uint64_t s = 0;
    for (size_t j = 0; j < w.size(); ++j) s = R.add(s, R.mul(w[j] % R.q, v[j] % R.q));
    return s;
}

u128 ZeroInDisc(uint64_t ell, const std::vector<uint32_t>& weights, const std::vector<uint64_t>& values, uint32_t P, uint64_t* residual) {
    const size_t N = weights.size();
    ENGINE_CHECK_MSG(N >= 2 && values.size() == N, "twovar::ZeroInDisc: shape");
    const uint32_t prec = std::min<uint32_t>(P, static_cast<uint32_t>(N));
    const bm::Ring R = bm::MakeRing(ell, P);
    // slope c = (v1 - v0) / (x1 - x0), a unit for a simple zero
    const u128 x0 = XAt(ell, weights[0], P + 1), x1 = XAt(ell, weights[1], P + 1);
    const ZMod Z(ell, P + 1);
    const u128 dx = Z.sub(x1, x0);
    ENGINE_CHECK_MSG(Z.val(dx, P + 1) == 1, "twovar::ZeroInDisc: v(x1 - x0) != 1");
    const uint64_t dv = R.sub(values[1] % R.q, values[0] % R.q);
    ENGINE_CHECK_MSG(dv % ell == 0, "twovar::ZeroInDisc: the values are not congruent mod ell");
    const ZMod RP1(ell, P > 1 ? P - 1 : 1);
    const u128 c = RP1.mul((static_cast<u128>(dv) / ell) % RP1.M, RP1.inv((dx / ell) % RP1.M));
    ENGINE_CHECK_MSG(c % ell != 0, "twovar::ZeroInDisc: the zero is not simple (slope not a unit)");
    const u128 cinv = R.inv(static_cast<uint64_t>(c % R.q));
    u128 y = XAt(ell, weights[0], P);
    uint64_t g = 0;
    for (uint32_t it = 0; it < 3 * P + 3; ++it) {
        g = Combine(LagrangeAt(ell, weights, P, y, P, false), values, ell, P);
        if (g % pow_wrap(ell, prec) == 0) break;
        y = static_cast<u128>(R.sub(static_cast<uint64_t>(y % R.q), R.mul(g, static_cast<uint64_t>(cinv))));
    }
    if (residual) *residual = g % pow_wrap(ell, prec);
    return y % pow_wrap(ell, prec);
}

uint64_t LogIwasawa(uint64_t x, uint64_t ell, uint32_t prec, bool div_ell) {
    ENGINE_CHECK_MSG(x % ell != 0, "twovar::LogIwasawa: a unit");
    const uint32_t T = prec + (div_ell ? 1 : 0);
    const uint32_t W = T + 6;
    const ZMod Q(ell, W), RT(ell, T);
    const u128 y = Q.pow(static_cast<u128>(x) % Q.M, ell - 1);
    const u128 z = Q.sub(y, 1 % Q.M);
    ENGINE_CHECK_MSG(z % ell == 0, "twovar::LogIwasawa: x^{ell-1} != 1 mod ell");
    u128 s = 0, zn = 1 % Q.M;
    for (uint64_t n = 1; n <= T + 6; ++n) {
        zn = Q.mul(zn, z);
        uint64_t m = n;
        uint32_t e = 0;
        while (m % ell == 0) {
            m /= ell;
            ++e;
        }
        ENGINE_CHECK_MSG(zn == 0 || Q.val(zn, W) >= e, "twovar::LogIwasawa: z^n / n not integral");
        const u128 term = RT.mul((zn / Q.pw[e]) % RT.M, RT.inv(m % RT.M));
        s = (n % 2 == 1) ? RT.add(s, term) : RT.sub(s, term);
    }
    s = RT.mul(s, RT.inv((ell - 1) % RT.M));
    if (!div_ell) return static_cast<uint64_t>(s);
    ENGINE_CHECK_MSG(s % ell == 0, "twovar::LogIwasawa: log not divisible by ell");
    return static_cast<uint64_t>((s / ell) % RT.pw[prec]);
}

// ---- Gamma_0(N), weight 2 ----------------------------------------------------------------------------------
namespace {
struct P1N {
    uint64_t N;
    uint64_t inv(uint64_t a) const {
        a %= N;
        for (uint64_t x = 1; x < N; ++x)
            if (a * x % N == 1) return x;
        ENGINE_CHECK_MSG(false, "P1N::inv: not invertible");
        return 0;
    }
    // index of (c:d), c, d integers with gcd(c, d, N) = 1
    uint32_t idx(int64_t c, int64_t d) const {
        const uint64_t cc = static_cast<uint64_t>(((c % static_cast<int64_t>(N)) + static_cast<int64_t>(N)) % static_cast<int64_t>(N));
        const uint64_t dd = static_cast<uint64_t>(((d % static_cast<int64_t>(N)) + static_cast<int64_t>(N)) % static_cast<int64_t>(N));
        if (dd != 0) return static_cast<uint32_t>(cc * inv(dd) % N);
        ENGINE_CHECK_MSG(cc != 0, "P1N::idx: (0:0)");
        return static_cast<uint32_t>(N);
    }
    std::pair<int64_t, int64_t> pt(uint32_t i) const {
        if (i == N) return {1, 0};
        return {static_cast<int64_t>(i), 1};
    }
    uint32_t act(uint32_t i, int64_t a, int64_t b, int64_t c, int64_t d) const {
        const auto p = pt(i);
        return idx(p.first * a + p.second * c, p.first * b + p.second * d);
    }
};
}  // namespace

Gamma0Symbol Gamma0Weight2Symbol(uint64_t N, int sign, int64_t ap2, uint64_t ell, uint32_t Rprec) {
    ENGINE_CHECK_MSG(N >= 5 && N % 2 == 1 && (sign == 1 || sign == -1), "twovar::Gamma0Weight2Symbol: N odd prime >= 5, sign +-1");
    Gamma0Symbol s;
    s.N = N;
    s.sign = sign;
    s.ap2 = ap2;
    s.R = bm::MakeRing(ell, Rprec);
    const bm::Ring& R = s.R;
    const P1N P{N};
    const uint32_t n = static_cast<uint32_t>(N + 1);
    std::vector<std::vector<int64_t>> rows;
    const int64_t h2[4][4] = {{2, 0, 0, 1}, {1, 0, 0, 2}, {2, 1, 0, 1}, {1, 0, 1, 2}};   // Merel's X_2
    for (uint32_t x = 0; x < n; ++x) {
        std::vector<int64_t> r(n, 0);
        r[x] += 1;
        r[P.act(x, 0, -1, 1, 0)] += 1;   // S
        rows.push_back(r);
        r.assign(n, 0);
        const uint32_t xu = P.act(x, 0, -1, 1, -1);   // U
        r[x] += 1;
        r[xu] += 1;
        r[P.act(xu, 0, -1, 1, -1)] += 1;
        rows.push_back(r);
        r.assign(n, 0);
        const auto p = P.pt(x);
        r[x] += 1;
        r[P.idx(-p.first, p.second)] -= sign;   // iota
        rows.push_back(r);
        r.assign(n, 0);
        for (const auto& h : h2) r[P.act(x, h[0], h[1], h[2], h[3])] += 1;
        r[x] -= ap2;
        rows.push_back(r);
    }
    bm::Mat A(static_cast<uint32_t>(rows.size()), n);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < n; ++j) A.at(i, j) = R.from_int(rows[i][j]);
    std::vector<bm::Vec> gens;
    std::vector<uint32_t> exps;
    bm::KernelGenerators(R, A, gens, exps);
    s.gens = static_cast<uint32_t>(gens.size());
    s.free_rank1 = gens.size() == 1 && exps[0] == R.m;
    if (!gens.empty()) s.psi = gens[0];
    return s;
}

uint64_t Gamma0Symbol::AtCusp(int64_t a, int64_t M) const {
    ENGINE_CHECK_MSG(M >= 1 && !psi.empty(), "Gamma0Symbol::AtCusp: M >= 1");
    const int64_t x = ((a % M) + M) % M;   // 0 <= x < M: the partial quotient a_0 = 0
    // convergents p_j / q_j of x / M; g_j = [[p_j, (-1)^{j-1} p_{j-1}], [q_j, (-1)^{j-1} q_{j-1}]] has bottom row
    // (q_j, (-1)^{j-1} q_{j-1}) and g_j {0, oo} = {p_{j-1}/q_{j-1}, p_j/q_j}; the sum over j = 0 .. J is {oo, x/M}.
    const P1N P{N};
    int64_t qprev = 0, qcur = 1;           // q_{-1}, q_0
    uint64_t s = psi[P.idx(qcur, -qprev)];  // j = 0
    int64_t num = M, den = x;               // the Euclidean algorithm after a_0
    int j = 0;
    while (den != 0) {
        const int64_t tq = num / den;
        const int64_t qn = tq * qcur + qprev;
        ++j;
        const int64_t sgn = ((j - 1) % 2 == 0) ? 1 : -1;
        s = R.add(s, psi[P.idx(qn, sgn * qcur)]);
        qprev = qcur;
        qcur = qn;
        const int64_t rr = num - tq * den;
        num = den;
        den = rr;
    }
    ENGINE_CHECK_MSG(qcur == M / std::gcd(x, M), "Gamma0Symbol::AtCusp: the last convergent's denominator");
    return s;
}

std::vector<std::vector<uint64_t>> MultiplicativeMeasure(const Gamma0Symbol& s, uint64_t ell, uint32_t M, uint64_t ap_ell) {
    const bm::Ring& R = s.R;
    ENGINE_CHECK_MSG(R.ell == ell && ap_ell % ell != 0, "twovar::MultiplicativeMeasure: ring / unit");
    const uint64_t api = R.inv(ap_ell % R.q);
    std::vector<std::vector<uint64_t>> mu(M + 1);
    uint64_t f = 1 % R.q;
    for (uint32_t m = 0; m <= M; ++m) {
        const uint64_t Mm = pow_wrap(ell, m);
        mu[m].assign(Mm, 0);
        for (uint64_t a = 0; a < Mm; ++a) mu[m][a] = R.mul(f, R.neg(s.AtCusp(static_cast<int64_t>(a), static_cast<int64_t>(Mm))));
        f = R.mul(f, api);
    }
    return mu;
}

uint64_t TateLInvariant11a(uint32_t prec) {
    ENGINE_CHECK_MSG(prec >= 1 && prec <= 10, "twovar::TateLInvariant11a: 1 <= prec <= 10");
    const uint64_t ell = 11;
    const uint32_t W = prec + 7;
    const ZMod Q(ell, W);
    const size_t D = 6, L = D + 3;
    auto sigma3 = [](uint64_t n) {
        uint64_t s = 0;
        for (uint64_t d = 1; d <= n; ++d)
            if (n % d == 0) s += d * d * d;
        return s;
    };
    using Ser = std::vector<u128>;
    auto mul = [&](const Ser& a, const Ser& b, size_t len) {
        Ser c(len, 0);
        for (size_t i = 0; i < a.size() && i < len; ++i)
            for (size_t j = 0; j < b.size() && i + j < len; ++j) c[i + j] = Q.add(c[i + j], Q.mul(a[i], b[j]));
        return c;
    };
    Ser E4(L, 0);
    E4[0] = 1;
    for (size_t n = 1; n < L; ++n) E4[n] = Q.mul(240 % Q.M, sigma3(n) % Q.M);
    const Ser E43 = mul(mul(E4, E4, L), E4, L);
    Ser dq(L, 0);   // prod (1 - q^n)^24
    dq[0] = 1;
    for (size_t n = 1; n < L; ++n)
        for (int r = 0; r < 24; ++r)
            for (size_t i = L - 1; i >= n; --i) dq[i] = Q.sub(dq[i], dq[i - n]);
    Ser J(L, 0);   // j = q^{-1} J(q)
    for (size_t i = 0; i < L; ++i) {
        u128 s = E43[i];
        for (size_t t = 0; t < i; ++t) s = Q.sub(s, Q.mul(J[t], dq[i - t]));
        J[i] = s;
    }
    ENGINE_CHECK_MSG(J[1] == 744 && J[2] == 196884, "twovar::TateLInvariant11a: j = 1/q + 744 + 196884 q + ...");
    // q(t) = t J(q(t)) (fixed point), as a series in t up to t^D
    Ser q(D + 1, 0);
    q[1] = 1;
    for (size_t it = 0; it < D + 2; ++it) {
        Ser res(D + 1, 0), pw(D + 1, 0);
        pw[0] = 1;
        for (size_t c = 0; c <= D; ++c) {
            for (size_t i = 0; i <= D; ++i) res[i] = Q.add(res[i], Q.mul(J[c], pw[i]));
            pw = mul(pw, q, D + 1);
        }
        Ser nq(D + 1, 0);
        for (size_t i = 1; i <= D; ++i) nq[i] = res[i - 1];
        q = nq;
    }
    ENGINE_CHECK_MSG(q[2] == 744 && q[3] == 750420, "twovar::TateLInvariant11a: q = t + 744 t^2 + 750420 t^3 + ...");
    // t_E = -11^5 / (2^12 31^3)
    const u128 tE = Q.neg(Q.mul(Q.pw[5], Q.inv((static_cast<u128>(4096) * 29791) % Q.M)));
    u128 qE = 0, tp = 1;
    for (size_t i = 1; i <= D; ++i) {
        tp = Q.mul(tp, tE);
        qE = Q.add(qE, Q.mul(q[i], tp));
    }
    ENGINE_CHECK_MSG(Q.val(qE, W) == 5, "twovar::TateLInvariant11a: ord_11(q_E) != 5");
    const uint64_t w = static_cast<uint64_t>(qE / Q.pw[5]);   // mod 11^{W - 5}
    const ZMod RP(ell, prec);
    const u128 lg = LogIwasawa(w % pow_wrap(ell, prec + 1), ell, prec, false);
    return static_cast<uint64_t>(RP.mul(lg % RP.M, RP.inv(5)));
}

}  // namespace twovar
}  // namespace lambda
}  // namespace hecke_ext
