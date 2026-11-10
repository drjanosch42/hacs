// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeLambdaQExpValidation (standard) -- the gate of P7 (hecke-extensions
// slot 21): the truncated Lambda-adic q-expansions sum_{n <= B} a_n(X) q^n
// mod (ell^K, (X - x_c)^m) of HeckeExt/HeckeLambdaQExp, per component of the
// cuspidal ordinary Hida algebra of tame level 1, and the Lambda-adic
// Eisenstein series.  Binding text: [U] docs/latex/hida_hecke_algebra.tex 0.4.0
// part 3 (sec:u-lambda: def:u-trunc, thm:u-interp, prop:u-lambdaprod(4)),
// thm:u-control (LIT, Hida: control for k >= 2), lem:u-twistvalues,
// lem:u-lambdaone.  Spec: s20/lead/owner_q3_padic_hida.md, P7.
//
// The classes: (11, 2) (the Delta family; its weight-2 member is 11a),
// (13, 0) (Delta, class 0: no Eisenstein series), (23, 2) (one component of
// residue degree 2: S_24's pair over Q(sqrt 144169)), (37, 32) and (103, 24)
// (irregular: a residually Eisenstein cusp component).  K = 3, m = 1, 2, 3,
// B = 30; the interpolation weights are k_0 + j (ell - 1), j = 3 .. 7 (the
// first K + m - 1 used); the FURTHER weights j = 0, 1, 2 are not used to build.
// Centres: the first interpolation weight, each further weight, and k = 2 at 11.
//
// The ORACLE is independent of HeckeLambdaQExp and of WP-HL's forms arm: its
// own series (Delta = q prod (1 - q^n)^24 from Euler's pentagonal series, E4,
// E6 from divisor sums), its own basis Delta^i E4^a E6^b of S_k, schoolbook
// products, T_n on q-expansions by the divisor formula for EVERY n <= B (no
// Hecke recursion), the ordinary part im T_ell^N, the component as the
// generalised kernel of g(T_{p0}) (a unit-pivot kernel, not WP-HL's image).
//   (1) oracle sanity: tau(2), tau(3), tau(5), tau(7), tau(11), tau(13)
//       (Ramanujan's values), T_2 Delta = -24 Delta; the eta-product
//       q prod (1-q^n)^2 (1-q^{11n})^2 of 11a against its a_p, p <= 13; at
//       every further weight the oracle's ordinary rank and charpoly(T_{p0} |
//       e S_k) against WP-HL's forms arm (a cross-route check);
//   (2) eps_p: the interpolated Lambda-adic weight character against the
//       closed form p^{c-1} sum_j binom(s_p, j) u^{-cj} Z^j (EpsilonBinomial);
//   (3) the Hecke relations a_m a_n = sum_{d | (m,n)} eps_d a_{mn/d^2} in
//       Lambda_{K,m}(x_c)[Y]/(F) (eps_d = 0 for ell | d), mn <= B;
//   (4) HIDA'S CONTROL at the further weights: the specialisation of every
//       a_n (n <= B), as a polynomial in T_{p0}, equals the oracle's T_n on the
//       component mod ell^{min(K, m v(x_{k'} - x_c))} (ell^K at k' = c), and
//       for rank 1 the oracle eigenform's q-expansion itself; F(x_{k'}, T_{p0})
//       = 0; the Eisenstein series against E_{k'}^{ord} with its constant term
//       -(1 - ell^{k'-1}) B_{k'} / (2 k') (exact Bernoulli numbers);
//   (5) weight 2 at 11: the Delta family at x_2 against 11a's eta-product
//       (alpha(x_2) = a_11(11a) = 1), mod 11^{min(K, m v(x_2 - x_c))};
//   (6) the Lambda-adic Eisenstein congruence at (37, 32), (103, 24):
//       a_n(f) - a_n(E) in (G~) = (X - alpha_a) for 1 <= n <= B, with
//       |Lambda_{K,m}(x_c)/(G~)| = ell^{min(K, m L_c)} (lem:u-lambdaone; L_c from
//       the exact B_c, so 103^{min(3, 2m)} at the rung centre c = 228);
//   (7) negative controls: W below K + m - 1 and fewer than K + m - 1 weights
//       REFUSED; T_5 + 37^2 at one interpolation weight refused (the
//       interpolant is not ell-integral), at every one caught at a further
//       weight; a specialisation at the wrong weight caught.
// Writes nothing, no golden object, no digest of record, no env hook.
#include "HeckeLambdaQExpValidation.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "../../ModArithmetic/EngineCheck.h"
#include "HeckeHidaLambda.h"
#include "HeckeLambdaQExp.h"
#include "../Util/ValidationCheck.h"

namespace {

namespace L = hecke_ext::lambda;
using L::Mat;
using L::QElem;
using L::u128;
using L::U128Text;
using L::Vec;
using L::ZMod;

oc_util::Checker Check("  [FAIL] ");

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

long PeakRssMiB() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024;
}

bool IsPrime(uint32_t n) {
    if (n < 2) return false;
    for (uint32_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

// ===========================================================================
// the independent forms route (the oracle)
// ===========================================================================
// schoolbook a * b mod q^Lq; residues < M < 2^40, so a u128 sum of < 2^40 products is exact
Vec SMul(const Vec& a, const Vec& b, size_t Lq, const ZMod& R) {
    ENGINE_CHECK(R.M < (static_cast<u128>(1) << 40));
    size_t va = 0, vb = 0;
    while (va < a.size() && a[va] == 0) ++va;
    while (vb < b.size() && b[vb] == 0) ++vb;
    Vec out(Lq, 0);
    for (size_t n = va + vb; n < Lq; ++n) {
        u128 acc = 0;
        const size_t hi = std::min(n - vb, a.size() - 1);
        for (size_t i = va; i <= hi; ++i)
            if (n - i < b.size()) acc += a[i] * b[n - i];
        out[n] = acc % R.M;
    }
    return out;
}

// Euler: prod_{n >= 1} (1 - q^n) = sum_j (-1)^j q^{j(3j-1)/2}, j in Z
Vec Pentagonal(size_t Lq, const ZMod& R) {
    Vec P(Lq, 0);
    for (int64_t j = -200000; j <= 200000; ++j) {
        const int64_t e = j * (3 * j - 1) / 2;
        if (e < 0 || static_cast<size_t>(e) >= Lq) continue;
        P[static_cast<size_t>(e)] = (j % 2 == 0) ? R.add(P[static_cast<size_t>(e)], 1) : R.sub(P[static_cast<size_t>(e)], 1);
    }
    return P;
}

Vec DeltaSeries(size_t Lq, const ZMod& R) {
    const Vec P = Pentagonal(Lq, R);
    const Vec P2 = SMul(P, P, Lq, R), P4 = SMul(P2, P2, Lq, R), P8 = SMul(P4, P4, Lq, R), P16 = SMul(P8, P8, Lq, R);
    const Vec P24 = SMul(P16, P8, Lq, R);
    Vec D(Lq, 0);
    for (size_t n = 1; n < Lq; ++n) D[n] = P24[n - 1];
    return D;
}

Vec EisSeries(size_t Lq, unsigned e, int64_t c, const ZMod& R) {   // 1 + c sum sigma_e(n) q^n
    Vec s(Lq, 0);
    for (size_t d = 1; d < Lq; ++d) {
        const u128 de = R.pow(d % R.M, e);
        for (size_t n = d; n < Lq; n += d) s[n] = R.add(s[n], de);
    }
    Vec E(Lq, 0);
    E[0] = 1 % R.M;
    for (size_t n = 1; n < Lq; ++n) E[n] = R.mul(R.from_i64(c), s[n]);
    return E;
}

uint32_t DimSk(uint32_t k) { return (k < 12 || k % 2) ? 0 : k / 12 - (k % 12 == 2 ? 1 : 0); }

// unit-pivot reduced row echelon over Z/ell^W of a matrix whose Smith form has
// only units and zeros; returns the kernel basis (columns), ENGINE_CHECKs the rest
Mat UnitKernel(Mat A, const ZMod& R) {
    const size_t nr = A.size(), nc = nr ? A[0].size() : 0;
    std::vector<size_t> pr, pc;
    std::vector<bool> rowp(nr, false), colp(nc, false);
    for (;;) {
        size_t bi = nr, bj = nc;
        for (size_t i = 0; i < nr && bi == nr; ++i)
            if (!rowp[i])
                for (size_t j = 0; j < nc; ++j)
                    if (!colp[j] && A[i][j] % R.ell) {
                        bi = i;
                        bj = j;
                        break;
                    }
        if (bi == nr) break;
        const u128 iv = R.inv(A[bi][bj]);
        for (auto& x : A[bi]) x = R.mul(x, iv);
        for (size_t i = 0; i < nr; ++i) {
            if (i == bi || A[i][bj] == 0) continue;
            const u128 f = A[i][bj];
            for (size_t j = 0; j < nc; ++j) A[i][j] = R.sub(A[i][j], R.mul(f, A[bi][j]));
        }
        rowp[bi] = colp[bj] = true;
        pr.push_back(bi);
        pc.push_back(bj);
    }
    for (size_t i = 0; i < nr; ++i)
        if (!rowp[i])
            for (size_t j = 0; j < nc; ++j) ENGINE_CHECK_MSG(A[i][j] == 0, "oracle UnitKernel: not a split matrix");
    std::vector<Vec> ker;
    for (size_t f = 0; f < nc; ++f) {
        if (colp[f]) continue;
        Vec v(nc, 0);
        v[f] = 1 % R.M;
        for (size_t t = 0; t < pr.size(); ++t) v[pc[t]] = R.neg(A[pr[t]][f]);
        ker.push_back(v);
    }
    Mat K(nc, Vec(ker.size()));
    for (size_t i = 0; i < nc; ++i)
        for (size_t c = 0; c < ker.size(); ++c) K[i][c] = ker[c][i];
    return K;
}

// coordinates of y in the column span of Bm (n x r, a free direct summand); ok = y in the span
Vec SolveInSpan(const Mat& Bm, const Vec& y, const ZMod& R, bool& ok) {
    const size_t n = Bm.size(), r = n ? Bm[0].size() : 0;
    Mat Bt(r, Vec(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t t = 0; t < r; ++t) Bt[t][i] = Bm[i][t];
    std::vector<size_t> rows = L::IndepColsModEll(Bt, n, R.ell);
    ENGINE_CHECK(rows.size() >= r);
    rows.resize(r);
    Mat Rm(r, Vec(r));
    for (size_t i = 0; i < r; ++i)
        for (size_t t = 0; t < r; ++t) Rm[i][t] = Bm[rows[i]][t];
    Vec yr(r);
    for (size_t i = 0; i < r; ++i) yr[i] = y[rows[i]];
    const Vec c = L::MatVec(L::InvUnitMatrix(Rm, R), yr, R);
    ok = true;
    for (size_t i = 0; i < n; ++i) {
        u128 acc = 0;
        for (size_t t = 0; t < r; ++t) acc = R.add(acc, R.mul(Bm[i][t], c[t]));
        ok = ok && acc == y[i] % R.M;
    }
    return c;
}

Mat ColBasisModEll(const Mat& P, const ZMod& R) {
    const size_t n = P.size();
    const std::vector<size_t> cols = L::IndepColsModEll(P, n, R.ell);
    Mat B(n, Vec(cols.size()));
    for (size_t i = 0; i < n; ++i)
        for (size_t t = 0; t < cols.size(); ++t) B[i][t] = P[i][cols[t]] % R.M;
    return B;
}

struct Oracle {
    uint64_t ell = 0;
    uint32_t k = 0, W = 0, d = 0;
    size_t Lq = 0;
    ZMod R;
    std::vector<Vec> b;     // b[i] = Delta^{i+1} E4^a E6^b, i < d
    Mat eS;                 // d x s, the ordinary part (S-coordinates)

    Oracle(uint64_t ell_, uint32_t k_, uint32_t W_, uint32_t B) : ell(ell_), k(k_), W(W_), R(ell_, W_) {
        d = DimSk(k);
        Lq = static_cast<size_t>(std::max<uint64_t>(ell, B)) * (d + 2) + 2;
        const Vec D = DeltaSeries(Lq, R), E4 = EisSeries(Lq, 3, 240, R), E6 = EisSeries(Lq, 5, -504, R);
        const uint32_t w = k - 12 * d;   // 0, 4, 6, 8, 10, 14
        const uint32_t bb = (w % 4 == 2) ? 1 : 0, aa = (w - 6 * bb) / 4;
        Vec C(Lq, 0);
        C[0] = 1 % R.M;
        for (uint32_t i = 0; i < aa; ++i) C = SMul(C, E4, Lq, R);
        if (bb) C = SMul(C, E6, Lq, R);
        const Vec Q = SMul(SMul(E4, E4, Lq, R), E4, Lq, R);
        std::vector<Vec> Dp(d + 1), Qp(d + 1);
        Dp[0] = Vec(Lq, 0);
        Dp[0][0] = 1 % R.M;
        Qp[0] = C;
        for (uint32_t i = 1; i <= d; ++i) {
            Dp[i] = SMul(Dp[i - 1], D, Lq, R);
            Qp[i] = SMul(Qp[i - 1], Q, Lq, R);
        }
        for (uint32_t i = 1; i <= d; ++i) b.push_back(SMul(Dp[i], Qp[d - i], Lq, R));
        for (uint32_t i = 0; i < d; ++i) {
            ENGINE_CHECK(b[i][i + 1] == 1 % R.M);
            for (uint32_t n = 0; n <= i; ++n) ENGINE_CHECK(b[i][n] == 0);
        }
        if (d) {
            Mat P = Op(ell);
            uint64_t e = 1;
            while (e < static_cast<uint64_t>(W) * d) {
                P = L::MatMul(P, P, R);
                e *= 2;
            }
            eS = ColBasisModEll(P, R);
        }
    }
    // coefficients 1 .. d+2 -> coordinates (forward substitution); ok = consistent at d+1, d+2
    Vec Coords(const Vec& c, bool& ok) const {
        Vec x(d, 0);
        for (uint32_t n = 1; n <= d; ++n) {
            u128 acc = c[n] % R.M;
            for (uint32_t i = 0; i + 1 < n; ++i) acc = R.sub(acc, R.mul(x[i], b[i][n]));
            x[n - 1] = acc;
        }
        ok = true;
        for (uint32_t n = d + 1; n <= d + 2; ++n) {
            u128 acc = 0;
            for (uint32_t i = 0; i < d; ++i) acc = R.add(acc, R.mul(x[i], b[i][n]));
            ok = ok && acc == c[n] % R.M;
        }
        return x;
    }
    Vec Series(const Vec& x) const {
        Vec g(Lq, 0);
        for (uint32_t i = 0; i < d; ++i)
            if (x[i])
                for (size_t n = 0; n < Lq; ++n) g[n] = R.add(g[n], R.mul(x[i], b[i][n]));
        return g;
    }
    // (T_n g)_j = sum_{e | (j, n)} e^{k-1} g_{jn/e^2}, j = 0 .. upto
    Vec HeckeCoeffs(uint64_t n, const Vec& g, uint32_t upto) const {
        Vec c(upto + 1, 0);
        for (uint32_t j = 0; j <= upto; ++j) {
            u128 acc = 0;
            for (uint64_t e = 1; e <= n; ++e) {
                if (n % e || (j % e)) continue;
                const uint64_t idx = static_cast<uint64_t>(j) * n / (e * e);
                ENGINE_CHECK_MSG(idx < Lq, "oracle: series too short");
                acc = R.add(acc, R.mul(R.pow(e % R.M, k - 1), g[idx]));
            }
            c[j] = acc;
        }
        return c;
    }
    Mat Op(uint64_t n) const {
        Mat T(d, Vec(d, 0));
        for (uint32_t j = 0; j < d; ++j) {
            bool ok = true;
            const Vec x = Coords(HeckeCoeffs(n, b[j], d + 2), ok);
            ENGINE_CHECK_MSG(ok, "oracle: T_n(S_k) not in S_k");
            for (uint32_t i = 0; i < d; ++i) T[i][j] = x[i];
        }
        return T;
    }
    uint32_t s() const { return eS.empty() ? 0 : static_cast<uint32_t>(eS[0].size()); }
    // T_{p0} on e S_k in the eS basis
    Mat OpOnOrd(uint64_t p) const {
        const Mat T = Op(p);
        const Mat TB = L::MatMul(T, eS, R);
        const size_t sv = s();
        Mat C(sv, Vec(sv));
        for (size_t t = 0; t < sv; ++t) {
            Vec col(d);
            for (uint32_t i = 0; i < d; ++i) col[i] = TB[i][t];
            bool ok = true;
            const Vec c = SolveInSpan(eS, col, R, ok);
            ENGINE_CHECK_MSG(ok, "oracle: e S_k not T_p-stable");
            for (size_t u = 0; u < sv; ++u) C[u][t] = c[u];
        }
        return C;
    }
};

// One component of e S_{k'} in the oracle: V (S-coordinates, d x r), T_{p0}|V, T_n|V for n <= B
struct OComp {
    Mat V;
    Mat t;
    std::vector<Mat> Tn;     // Tn[n], n = 1 .. B
    std::vector<u128> eig;   // rank 1: the normalised eigenform's a_n, n = 0 .. B
    bool ok = true;
};

OComp OracleComponent(const Oracle& O, uint64_t p0, const std::vector<uint64_t>& g, uint32_t B) {
    const ZMod& R = O.R;
    OComp oc;
    const Mat tO = O.OpOnOrd(p0);
    const size_t sv = O.s();
    Mat Gm(sv, Vec(sv, 0));   // g(tO), Horner
    for (size_t i = g.size(); i-- > 0;) {
        Gm = L::MatMul(Gm, tO, R);
        for (size_t u = 0; u < sv; ++u) Gm[u][u] = R.add(Gm[u][u], g[i] % R.M);
    }
    uint64_t e = 1;
    while (e < static_cast<uint64_t>(O.W) * std::max<size_t>(sv, 1)) {
        Gm = L::MatMul(Gm, Gm, R);
        e *= 2;
    }
    const Mat Kb = UnitKernel(Gm, R);
    oc.V = L::MatMul(O.eS, Kb, R);
    const size_t r = Kb.empty() ? 0 : Kb[0].size();
    oc.ok = r == g.size() - 1;
    if (!oc.ok) return oc;
    auto onV = [&](uint64_t n) {
        Mat M(r, Vec(r));
        for (size_t c = 0; c < r; ++c) {
            Vec x(O.d);
            for (uint32_t i = 0; i < O.d; ++i) x[i] = oc.V[i][c];
            bool ok1 = true, ok2 = true;
            const Vec y = O.Coords(O.HeckeCoeffs(n, O.Series(x), O.d + 2), ok1);
            const Vec cc = SolveInSpan(oc.V, y, R, ok2);
            oc.ok = oc.ok && ok1 && ok2;
            for (size_t u = 0; u < r; ++u) M[u][c] = cc[u];
        }
        return M;
    };
    oc.t = onV(p0);
    oc.Tn.resize(B + 1);
    for (uint32_t n = 1; n <= B; ++n) oc.Tn[n] = onV(n);
    if (r == 1) {
        Vec x(O.d);
        for (uint32_t i = 0; i < O.d; ++i) x[i] = oc.V[i][0];
        const Vec gs = O.Series(x);
        const u128 iv = R.inv(gs[1]);
        for (uint32_t n = 0; n <= B; ++n) oc.eig.push_back(R.mul(gs[n], iv));
    }
    return oc;
}

// sum_i c_i t^i mod ell^prec
Mat PolyAt(const Vec& c, const Mat& t, const ZMod& R) {
    const size_t r = t.size();
    Mat acc(r, Vec(r, 0));
    for (size_t i = c.size(); i-- > 0;) {
        acc = L::MatMul(acc, t, R);
        for (size_t u = 0; u < r; ++u) acc[u][u] = R.add(acc[u][u], c[i] % R.M);
    }
    return acc;
}

bool MatEqMod(const Mat& A, const Mat& B, const ZMod& R) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[i].size(); ++j)
            if (A[i][j] % R.M != B[i][j] % R.M) return false;
    return true;
}

// q prod (1 - q^n)^2 (1 - q^{11 n})^2, exact, n < len
std::vector<int64_t> Eta11a(size_t len) {
    std::vector<int64_t> P(len, 0);
    P[0] = 1;
    for (size_t n = 1; n < len; ++n)
        for (int rep = 0; rep < 2; ++rep) {
            for (size_t t = len - 1; t >= n; --t) P[t] -= P[t - n];
            if (11 * n < len)
                for (size_t t = len - 1; t >= 11 * n; --t) P[t] -= P[t - 11 * n];
        }
    std::vector<int64_t> f(len, 0);
    for (size_t n = 1; n < len; ++n) f[n] = P[n - 1];
    return f;
}

struct ClassSpec {
    uint64_t ell;
    uint32_t a, k0;
    bool weight2;   // the 11a point
};

// the root beta of G~ in Lambda_{K,m}(x_c) (Hensel), and min(K, m v(beta)) = log |Lambda_{K,m}/(G~)|
std::pair<u128, uint32_t> GRoot(const Vec& G, const ZMod& RK, uint32_t m) {
    const uint32_t K = RK.W;
    if (m == 1) return {0, std::min<uint32_t>(K, RK.val(G[0], K))};
    ENGINE_CHECK_MSG(G[1] % RK.ell != 0, "GRoot: lambda != 1");
    u128 beta = 0;
    for (uint32_t it = 0; it < 4 * K; ++it) {
        u128 val = 0, der = 0;
        for (size_t i = G.size(); i-- > 0;) val = RK.add(RK.mul(val, beta), G[i]);
        for (size_t i = G.size(); i-- > 1;) der = RK.add(RK.mul(der, beta), RK.mul(i % RK.M, G[i]));
        beta = RK.sub(beta, RK.mul(val, RK.inv(der)));
    }
    const uint32_t v = RK.val(beta, K);
    return {beta, static_cast<uint32_t>(std::min<uint64_t>(K, static_cast<uint64_t>(m) * v))};
}

}  // namespace

uint32_t HeckeLambdaQExpValidation() {
    const double t_start = Now();
    Check.reset();
    const uint32_t K = 3, B = 30, W = 5;
    const std::vector<uint32_t> ms{1, 2, 3};
    const std::vector<ClassSpec> specs{{11, 2, 12, true}, {13, 0, 12, false}, {23, 2, 24, false}, {37, 32, 32, false},
                                       {103, 24, 24, false}};
    std::cout << "EFFECTIVE HeckeLambdaQExpValidation arm=DEFAULT cells=" << specs.size() << " K=" << K << " m=1..3 B=" << B
              << std::endl;

    // ---- (1) oracle sanity
    {
        const ZMod R11(11, W), R13(13, W);
        const Vec D11 = DeltaSeries(40, R11), D13 = DeltaSeries(40, R13);
        const std::vector<std::pair<uint32_t, int64_t>> tau{{2, -24}, {3, 252}, {5, 4830}, {7, -16744}, {11, 534612}, {13, -577738}};
        bool ok = true;
        for (const auto& kv : tau) ok = ok && D11[kv.first] == R11.from_i64(kv.second) && D13[kv.first] == R13.from_i64(kv.second);
        Check(ok, "oracle Delta: Ramanujan's tau(p), p <= 13");
        const Oracle O12(11, 12, W, B);
        bool ok2 = O12.d == 1;
        if (ok2) {
            const Mat T2 = O12.Op(2);
            ok2 = T2[0][0] == R11.from_i64(-24);
        }
        Check(ok2, "oracle Hecke formula: T_2 Delta = -24 Delta");
        const auto f = Eta11a(40);
        const std::vector<std::pair<uint32_t, int64_t>> ap11a{{1, 1}, {2, -2}, {3, -1}, {5, 1}, {7, -2}, {11, 1}, {13, 4}};
        bool ok3 = true;
        for (const auto& kv : ap11a) ok3 = ok3 && f[kv.first] == kv.second;
        Check(ok3, "oracle 11a: eta(z)^2 eta(11z)^2 has a_p = 1,-2,-1,1,-2,1,4 (p = 1,2,3,5,7,11,13)");
    }

    for (const ClassSpec& cs : specs) {
        const double tc = Now();
        const uint64_t l = cs.ell;
        std::vector<uint64_t> primes;
        for (uint32_t p = 2; p <= B; ++p)
            if (IsPrime(p) && p != l) primes.push_back(p);
        std::vector<uint32_t> interp_k, further_k;
        for (uint32_t j = 3; j < 8; ++j) interp_k.push_back(cs.k0 + j * static_cast<uint32_t>(l - 1));
        for (uint32_t j = 0; j < 3; ++j) further_k.push_back(cs.k0 + j * static_cast<uint32_t>(l - 1));
        std::vector<L::FormsWeight> ws;
        for (uint32_t k : interp_k) ws.push_back(L::ComputeFormsWeight(l, k, W, primes));
        std::map<uint32_t, Oracle> oracles;
        for (uint32_t k : further_k) oracles.emplace(k, Oracle(l, k, W, B));
        const ZMod Rw(l, W), RK(l, K);
        const bool with_eis = cs.a % (l - 1) != 0;
        std::cout << "CLASS ell=" << l << " a=" << cs.a << " interp=" << interp_k.front() << ".." << interp_k.back()
                  << " further=" << further_k[0] << "," << further_k[1] << "," << further_k[2]
                  << (cs.weight2 ? ",2(11a)" : "") << " ordinary_rank=" << ws[0].S.s << " eisenstein="
                  << (with_eis ? "yes" : "no(class 0)") << std::endl;
        // (1) cross-route: oracle vs WP-HL forms arm at the further weights
        {
            bool ok = true;
            for (uint32_t k : further_k) {
                const L::FormsWeight fw = L::ComputeFormsWeight(l, k, W, primes);
                const Oracle& O = oracles.at(k);
                ok = ok && fw.S.s == O.s();
                if (ok && O.s())
                    for (uint64_t p : {primes[0], primes[1]})
                        ok = ok && L::Charpoly(fw.S.mats.at("T" + std::to_string(p)), Rw) == L::Charpoly(O.OpOnOrd(p), Rw);
            }
            Check(ok, "ell=" + std::to_string(l) + " oracle e S_k (rank, charpoly T_p) == WP-HL forms arm at the further weights");
        }
        std::map<std::tuple<uint32_t, uint64_t, std::vector<uint64_t>>, OComp> ocache;
        auto ocomp = [&](uint32_t k, uint64_t p0, const std::vector<uint64_t>& g) -> const OComp& {
            const auto key = std::make_tuple(k, p0, g);
            auto it = ocache.find(key);
            if (it == ocache.end()) it = ocache.emplace(key, OracleComponent(oracles.at(k), p0, g, B)).first;
            return it->second;
        };
        const auto eta = Eta11a(B + 1);
        std::map<uint32_t, uint32_t> Lcache;   // centre -> L_c
        uint32_t n_trunc = 0, n_spec = 0, n_cong = 0;
        for (uint32_t m : ms) {
            std::vector<uint32_t> centres{interp_k[0]};
            centres.insert(centres.end(), further_k.begin(), further_k.end());
            if (cs.weight2) centres.push_back(2);
            for (uint32_t c : centres) {
                const L::LambdaQExpansion LQ = L::BuildLambdaQExpansion(ws, cs.a, K, m, c, B, 0, with_eis);
                ++n_trunc;
                const std::string tag = "ell=" + std::to_string(l) + " a=" + std::to_string(cs.a) + " m=" + std::to_string(m) +
                                        " centre=" + std::to_string(c);
                // (2) eps_p
                {
                    bool ok = true;
                    for (uint64_t p : primes) ok = ok && L::EpsilonBinomial(l, p, K, m, c) == LQ.eps.at(p);
                    Check(ok, tag + " eps_p interpolated == closed form (binomial series)");
                }
                // (3) Hecke relations
                {
                    bool ok = true;
                    auto epsd = [&](uint32_t dd, uint32_t r) {
                        Vec z(m, 0);
                        z[0] = 1 % RK.M;
                        uint32_t x = dd;
                        for (uint64_t p = 2; x > 1; ++p)
                            while (x % p == 0) {
                                x /= static_cast<uint32_t>(p);
                                z = p == l ? Vec(m, 0) : L::ZMulTrunc(z, LQ.eps.at(p), RK, m);
                            }
                        return L::QScalar(z, r);
                    };
                    for (const auto& C : LQ.comps)
                        for (uint32_t m1 = 2; m1 <= B; ++m1)
                            for (uint32_t n1 = m1; m1 * n1 <= B; ++n1) {
                                uint32_t gg = m1, hh = n1;
                                while (hh) {
                                    const uint32_t t = gg % hh;
                                    gg = hh;
                                    hh = t;
                                }
                                QElem rhs(C.rank, Vec(m, 0));
                                for (uint32_t dd = 1; dd <= gg; ++dd)
                                    if (gg % dd == 0)
                                        rhs = L::QAdd(rhs, L::QMul(epsd(dd, C.rank), C.a[m1 * n1 / (dd * dd)], C.F, RK, m), RK);
                                ok = ok && L::QMul(C.a[m1], C.a[n1], C.F, RK, m) == rhs;
                            }
                    Check(ok, tag + " Hecke relations a_m a_n = sum eps_d a_{mn/d^2} (mn <= B)");
                }
                // (4) control at the further weights
                for (size_t ci = 0; ci < LQ.comps.size(); ++ci) {
                    const L::LambdaComponent& C = LQ.comps[ci];
                    for (uint32_t kf : further_k) {
                        const uint32_t e = L::SpecPrecision(l, K, m, c, kf);
                        const ZMod Re(l, e);
                        bool ok = true;
                        if (C.kind == "eisenstein") {
                            const auto BK = L::BernoulliOverKExact(l, e, {kf});
                            const u128 a0 = Re.mul(Re.neg(Re.mul(Re.sub(1 % Re.M, Re.pow(l % Re.M, kf - 1)), BK.at(kf))),
                                                   Re.inv(2 % Re.M));
                            ok = L::SpecialiseAt(C.a[0], l, c, kf, e)[0] == a0;
                            for (uint32_t n = 1; n <= B; ++n) {
                                u128 sg = 0;
                                for (uint32_t dd = 1; dd <= n; ++dd)
                                    if (n % dd == 0 && dd % l) sg = Re.add(sg, Re.pow(dd % Re.M, kf - 1));
                                ok = ok && L::SpecialiseAt(C.a[n], l, c, kf, e)[0] == sg;
                            }
                        } else {
                            const OComp& oc = ocomp(kf, LQ.p0, C.residual);
                            ok = oc.ok;
                            if (ok) {
                                // F(x_{k'}, T_{p0}) = 0
                                Vec Fv;
                                for (const Vec& fc : C.F) Fv.push_back(L::SpecialiseAt(L::QScalar(fc, 1), l, c, kf, e)[0]);
                                ok = ok && MatEqMod(PolyAt(Fv, oc.t, Re), Mat(C.rank, Vec(C.rank, 0)), Re);
                                for (uint32_t n = 1; n <= B; ++n)
                                    ok = ok && MatEqMod(PolyAt(L::SpecialiseAt(C.a[n], l, c, kf, e), oc.t, Re), oc.Tn[n], Re);
                                if (C.rank == 1)
                                    for (uint32_t n = 1; n <= B; ++n)
                                        ok = ok && L::SpecialiseAt(C.a[n], l, c, kf, e)[0] == oc.eig[n] % Re.M;
                            }
                        }
                        ++n_spec;
                        Check(ok, tag + " component " + std::to_string(ci) + " (" + C.kind + ", rank " + std::to_string(C.rank) +
                                      ") specialised at k'=" + std::to_string(kf) + " == oracle q-expansion mod ell^" +
                                      std::to_string(e));
                    }
                    // (5) weight 2 at 11: 11a
                    if (cs.weight2 && C.kind != "eisenstein") {
                        const uint32_t e = L::SpecPrecision(l, K, m, c, 2);
                        const ZMod Re(l, e);
                        bool ok = C.rank == 1;
                        for (uint32_t n = 1; ok && n <= B; ++n) ok = L::SpecialiseAt(C.a[n], l, c, 2, e)[0] == Re.from_i64(eta[n]);
                        ++n_spec;
                        Check(ok, tag + " Delta family at weight 2 == 11a (eta product) mod 11^" + std::to_string(e));
                        if (m == 3 && c == 2) {
                            std::ostringstream os;
                            os << "ALPHA_AT_2 ell=11 a=2 K=" << K << " m=3 centre=2 alpha(Z)=[";
                            for (size_t t = 0; t < C.alpha[0].size(); ++t) os << (t ? "," : "") << U128Text(C.alpha[0][t]);
                            os << "] (Z = X - x_2, x_k = 12^k - 1; alpha(x_2) = a_11(11a) = 1; for P5)";
                            std::cout << os.str() << std::endl;
                        }
                    }
                }
                // (6) the Lambda-adic Eisenstein congruence
                if (with_eis) {
                    const L::LambdaComponent* f = nullptr;
                    const L::LambdaComponent* E = nullptr;
                    for (const auto& C : LQ.comps) {
                        if (C.kind == "cusp_eis") f = &C;
                        if (C.kind == "eisenstein") E = &C;
                    }
                    if (f && E) {
                        const auto root = GRoot(LQ.G, RK, m);
                        const ZMod Re(l, std::max<uint32_t>(root.second, 1));
                        // lem:u-lambdaone: |Lambda_{K,m}(x_c)/(G~)| = ell^{min(K, m L_c)}, L_c = v(G_a(x_c)) from
                        // the exact Bernoulli number at the centre (WP-NF), not from the interpolated G~
                        if (!Lcache.count(c)) {
                            const ZMod R6(l, 6);
                            const u128 Gc = R6.neg(R6.mul(R6.sub(1 % R6.M, R6.pow(l % R6.M, c - 1)),
                                                          L::BernoulliOverKExact(l, 6, {c}).at(c)));
                            Lcache[c] = R6.val(Gc, 6);
                        }
                        const uint32_t Lc = Lcache.at(c);
                        const uint32_t pred = static_cast<uint32_t>(std::min<uint64_t>(K, static_cast<uint64_t>(m) * Lc));
                        bool ok = root.second >= 1 && root.second == pred;
                        for (uint32_t n = 1; n <= B; ++n) {
                            const Vec h = L::QSub(f->a[n], E->a[n], RK)[0];
                            u128 val = 0;
                            for (size_t i = h.size(); i-- > 0;) val = Re.add(Re.mul(val, root.first % Re.M), h[i] % Re.M);
                            ok = ok && val == 0;
                        }
                        ++n_cong;
                        Check(ok, tag + " Lambda-adic congruence a_n(f) == a_n(E) mod (G~), |Lambda_{K,m}/(G~)| = ell^" +
                                      std::to_string(root.second) + " (predicted ell^" + std::to_string(pred) + ", L_c=" +
                                      std::to_string(Lc) + ")");
                        if (Lc >= 2)
                            std::cout << "CONGRUENCE_AT_RUNG " << tag << " L_c=" << Lc << " |Lambda_{K,m}/(G~)|=ell^"
                                      << root.second << std::endl;
                    }
                }
                if (m == 3 && c == interp_k[0])
                    for (const auto& line : L::QExpRecordLines(LQ)) std::cout << line << std::endl;
            }
        }
        std::cout << "CLASS_DONE ell=" << l << " a=" << cs.a << " truncations=" << n_trunc << " specialisations=" << n_spec
                  << " congruences=" << n_cong << " wall_s=" << Now() - tc << std::endl;
        // (7) negative controls at the irregular class 37
        if (l == 37) {
            const uint32_t m = 3;
            auto refused = [&](const std::vector<L::FormsWeight>& w) {
                try {
                    L::BuildLambdaQExpansion(w, cs.a, K, m, interp_k[0], B, 0, true);
                } catch (const std::logic_error& ex) {
                    return std::string(ex.what()).find("REFUSED") != std::string::npos;
                }
                return false;
            };
            std::vector<L::FormsWeight> lowW = ws;
            for (auto& w : lowW) w.W = K + m - 2;
            Check(refused(lowW), "negative: forms data mod ell^{K+m-2} REFUSED");
            Check(refused(std::vector<L::FormsWeight>(ws.begin(), ws.begin() + (K + m - 2))),
                  "negative: K + m - 2 weights REFUSED");
            auto perturbed = [&](bool all_weights) {
                std::vector<L::FormsWeight> pert = ws;
                for (size_t j = 0; j < pert.size(); ++j) {
                    if (!all_weights && j != 1) continue;
                    Mat& T5 = pert[j].S.mats.at("T5");
                    for (size_t i = 0; i < T5.size(); ++i) T5[i][i] = Rw.add(T5[i][i], Rw.pw[2]);
                }
                return pert;
            };
            {
                // one node off by 37^2: the values are not those of one g in Lambda, and the
                // interpolant is not ell-integral (thm:u-interp(1)); refused
                bool thrown = false;
                try {
                    L::BuildLambdaQExpansion(perturbed(false), cs.a, K, m, interp_k[0], B, 0, true);
                } catch (const std::logic_error& ex) {
                    thrown = std::string(ex.what()).find("not ell-integral") != std::string::npos;
                }
                Check(thrown, "negative: T_5 + 37^2 at one interpolation weight refused (interpolant not ell-integral)");
            }
            const L::LambdaQExpansion LP = L::BuildLambdaQExpansion(perturbed(true), cs.a, K, m, interp_k[0], B, 0, true);
            bool caught = false;
            for (const auto& C : LP.comps) {
                if (C.kind == "eisenstein") continue;
                const OComp& oc = ocomp(further_k[0], LP.p0, C.residual);
                const ZMod Re(l, L::SpecPrecision(l, K, m, interp_k[0], further_k[0]));
                caught = caught || !MatEqMod(PolyAt(L::SpecialiseAt(C.a[5], l, interp_k[0], further_k[0], Re.W), oc.t, Re),
                                             oc.Tn[5], Re);
            }
            Check(caught, "negative: T_5 + 37^2 at every interpolation weight caught at k'=" + std::to_string(further_k[0]));
            const L::LambdaQExpansion LQ = L::BuildLambdaQExpansion(ws, cs.a, K, m, interp_k[0], B, 0, true);
            bool wrong = false;
            for (const auto& C : LQ.comps) {
                if (C.kind != "cusp_eis") continue;
                const OComp& oc = ocomp(further_k[0], LQ.p0, C.residual);
                const ZMod Re(l, L::SpecPrecision(l, K, m, interp_k[0], further_k[0]));
                for (uint32_t n = 2; n <= B; ++n)
                    wrong = wrong || L::SpecialiseAt(C.a[n], l, interp_k[0], further_k[0] + (l - 1), Re.W)[0] != oc.eig[n] % Re.M;
            }
            Check(wrong, "negative: the specialisation at the wrong weight k'+(ell-1) caught");
        }
    }

    std::cout << "HeckeLambdaQExpValidation: checks=" << Check.checks << " failing=" << Check.fails << " wall_s=" << Now() - t_start
              << " peak_rss_mib=" << PeakRssMiB() << std::endl;
    return Check.fails == 0 ? 0 : 1;
}
