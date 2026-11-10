// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 21, P7) Lambda-adic q-expansions, truncated: the
// component split of e S_k by the residual factors of T_{p0}, the monogenic
// presentation at every weight, thm:u-interp (through WP-HL's
// InterpolateTruncation, precision enforced), the Lambda-adic weight character
// eps_p, the Hecke recursions in Lambda_{K,m}(x_c)[Y]/(F), the Lambda-adic
// Eisenstein series, the record hecke_lambda_qexp/v1.  See HeckeLambdaQExp.h.
#include "HeckeLambdaQExp.h"

#include <algorithm>
#include <random>
#include <sstream>

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "../../ModArithmetic/Fnv.h"

namespace hecke_ext {
namespace lambda {

namespace {

// ---- F_ell[Y], u64 coefficients, low -> high ------------------------------------
using FP = std::vector<uint64_t>;

uint64_t MulL(uint64_t a, uint64_t b, uint64_t l) { return engine::scalar::mulmod(a, b, l); }

uint64_t PowL(uint64_t a, uint64_t e, uint64_t l) { return engine::scalar::powmod<uint64_t>(a, e, l); }

FP Trim(FP a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
    return a;
}

FP Monic(FP a, uint64_t l) {
    a = Trim(a);
    if (a.empty()) return a;
    const uint64_t iv = PowL(a.back(), l - 2, l);
    for (auto& x : a) x = MulL(x, iv, l);
    return a;
}

std::pair<FP, FP> DivMod(FP a, FP b, uint64_t l) {
    a = Trim(a);
    b = Trim(b);
    ENGINE_CHECK(!b.empty());
    if (a.size() < b.size()) return {FP{}, a};
    FP q(a.size() - b.size() + 1, 0);
    const uint64_t ib = PowL(b.back(), l - 2, l);
    while (a.size() >= b.size() && !a.empty()) {
        const uint64_t c = MulL(a.back(), ib, l);
        const size_t d = a.size() - b.size();
        q[d] = c;
        for (size_t i = 0; i < b.size(); ++i) a[i + d] = (a[i + d] + l - MulL(c, b[i], l)) % l;
        a = Trim(a);
    }
    return {Trim(q), a};
}

FP Gcd(FP a, FP b, uint64_t l) {
    a = Trim(a);
    b = Trim(b);
    while (!b.empty()) {
        FP r = DivMod(a, b, l).second;
        a = b;
        b = r;
    }
    return Monic(a, l);
}

FP MulP(const FP& a, const FP& b, uint64_t l) {
    if (a.empty() || b.empty()) return {};
    FP out(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) out[i + j] = (out[i + j] + MulL(a[i], b[j], l)) % l;
    return Trim(out);
}

FP MulMod(const FP& a, const FP& b, const FP& f, uint64_t l) { return DivMod(MulP(a, b, l), f, l).second; }

FP PowMod(FP a, uint64_t e, const FP& f, uint64_t l) {
    FP r{1 % l};
    a = DivMod(a, f, l).second;
    while (e) {
        if (e & 1u) r = MulMod(r, a, f, l);
        e >>= 1;
        if (e) a = MulMod(a, a, f, l);
    }
    return DivMod(r, f, l).second;
}

FP SubX(FP h, uint64_t l) {   // h - Y
    if (h.size() < 2) h.resize(2, 0);
    h[1] = (h[1] + l - 1) % l;
    return Trim(h);
}

bool SquarefreeL(const FP& f, uint64_t l) {
    FP df;
    for (size_t i = 1; i < f.size(); ++i) df.push_back(MulL(i % l, f[i], l));
    df = Trim(df);
    if (df.empty()) return f.size() <= 1;
    return Gcd(f, df, l).size() == 1;
}

// equal-degree splitting of a monic squarefree f, all factors of degree d (Cantor-Zassenhaus, ell odd)
void EqualDegree(const FP& f, uint32_t d, uint64_t l, std::mt19937_64& rng, std::vector<FP>& out) {
    const size_t n = f.size() - 1;
    if (n == d) {
        out.push_back(f);
        return;
    }
    for (int attempt = 0; attempt < 400; ++attempt) {
        FP h(n);
        for (auto& x : h) x = rng() % l;
        h = Trim(h);
        if (h.size() < 2) continue;
        // h^{(l^d - 1)/2} = (prod_{i<d} h^{l^i})^{(l-1)/2}
        FP acc{1}, fr = DivMod(h, f, l).second;
        for (uint32_t i = 0; i < d; ++i) {
            acc = MulMod(acc, fr, f, l);
            fr = PowMod(fr, l, f, l);
        }
        FP e = PowMod(acc, (l - 1) / 2, f, l);
        if (e.empty()) e.push_back(0);
        e[0] = (e[0] + l - 1) % l;
        const FP g = Gcd(f, e, l);
        if (g.size() > 1 && g.size() < f.size()) {
            EqualDegree(g, d, l, rng, out);
            EqualDegree(Monic(DivMod(f, g, l).first, l), d, l, rng, out);
            return;
        }
    }
    ENGINE_CHECK_MSG(false, "FactorSquarefreeModEll: equal-degree split did not converge");
}

// ---- dense helpers mod ell^W -----------------------------------------------------
Mat ReduceMat(const Mat& A, const ZMod& R) {
    Mat B = A;
    for (auto& row : B)
        for (auto& x : row) x %= R.M;
    return B;
}

Mat PolyOfMat(const Vec& p, const Mat& t, const ZMod& R) {   // p(t), Horner
    const size_t n = t.size();
    Mat acc(n, Vec(n, 0));
    for (size_t i = p.size(); i-- > 0;) {
        acc = MatMul(acc, t, R);
        for (size_t d = 0; d < n; ++d) acc[d][d] = R.add(acc[d][d], p[i] % R.M);
    }
    return acc;
}

struct Basis {
    Mat Bm;                    // n x r
    std::vector<size_t> rows;  // r rows with an invertible minor mod ell
    Mat Rinv;
};

Basis MakeBasis(const Mat& P, const ZMod& R) {
    const size_t n = P.size();
    Basis b;
    const std::vector<size_t> cols = IndepColsModEll(P, n, R.ell);
    const size_t r = cols.size();
    b.Bm.assign(n, Vec(r));
    for (size_t i = 0; i < n; ++i)
        for (size_t t = 0; t < r; ++t) b.Bm[i][t] = P[i][cols[t]] % R.M;
    if (r == 0) return b;
    Mat Bt(r, Vec(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t t = 0; t < r; ++t) Bt[t][i] = b.Bm[i][t];
    b.rows = IndepColsModEll(Bt, n, R.ell);
    ENGINE_CHECK(b.rows.size() >= r);
    b.rows.resize(r);
    Mat Rm(r, Vec(r));
    for (size_t i = 0; i < r; ++i)
        for (size_t t = 0; t < r; ++t) Rm[i][t] = b.Bm[b.rows[i]][t];
    b.Rinv = InvUnitMatrix(Rm, R);
    return b;
}

Mat RestrictTo(const Mat& A, const Basis& b, const ZMod& R) {
    const size_t n = A.size(), r = b.Rinv.size();
    const Mat AB = MatMul(A, b.Bm, R);
    Mat C(r, Vec(r, 0));
    for (size_t t = 0; t < r; ++t) {
        Vec yr(r);
        for (size_t i = 0; i < r; ++i) yr[i] = AB[b.rows[i]][t];
        const Vec c = MatVec(b.Rinv, yr, R);
        for (size_t i = 0; i < n; ++i) {
            u128 acc = 0;
            for (size_t u = 0; u < r; ++u) acc = R.add(acc, R.mul(b.Bm[i][u], c[u]));
            ENGINE_CHECK_MSG(acc == AB[i][t], "HeckeLambdaQExp: the component is not stable under the operator");
        }
        for (size_t u = 0; u < r; ++u) C[u][t] = c[u];
    }
    return C;
}

// coefficients c with target = sum_i c_i t^i (i < r), verified on every entry
Vec InPowers(const Mat& target, const Mat& t, const ZMod& R) {
    const size_t r = t.size();
    Mat P = Ident(r), A(r * r, Vec(r, 0));
    for (size_t i = 0; i < r; ++i) {
        for (size_t x = 0; x < r; ++x)
            for (size_t y = 0; y < r; ++y) A[x * r + y][i] = P[x][y] % R.M;
        P = MatMul(P, t, R);
    }
    Mat At(r, Vec(r * r));
    for (size_t i = 0; i < r * r; ++i)
        for (size_t u = 0; u < r; ++u) At[u][i] = A[i][u];
    std::vector<size_t> rows = IndepColsModEll(At, r * r, R.ell);
    ENGINE_CHECK_MSG(rows.size() >= r, "HeckeLambdaQExp: T_{p0} does not generate the component mod ell");
    rows.resize(r);
    Mat Ar(r);
    for (size_t i = 0; i < r; ++i) Ar[i] = A[rows[i]];
    const Mat Ainv = InvUnitMatrix(Ar, R);
    Vec tv;
    for (const auto& row : target)
        for (u128 x : row) tv.push_back(x % R.M);
    Vec sel(r);
    for (size_t i = 0; i < r; ++i) sel[i] = tv[rows[i]];
    const Vec c = MatVec(Ainv, sel, R);
    for (size_t i = 0; i < r * r; ++i) {
        u128 acc = 0;
        for (size_t u = 0; u < r; ++u) acc = R.add(acc, R.mul(A[i][u], c[u]));
        ENGINE_CHECK_MSG(acc == tv[i], "HeckeLambdaQExp: an operator is not a polynomial in T_{p0}");
    }
    return c;
}

// One weight's components: residual factor -> (charpoly of t on V_g, P_op coefficients)
std::vector<FP> FactorSquarefreeModEll(const FP& f0, uint64_t l);   // (defined below)

struct WeightComp {
    FP g;
    Vec F;                                 // r + 1 coefficients, monic
    std::map<std::string, Vec> P;          // op -> r coefficients
};

std::vector<WeightComp> ComponentsAt(const FormsWeight& w, uint64_t p0, const ZMod& R) {
    const uint64_t l = w.ell;
    const FormsSpace& S = w.S;
    std::vector<WeightComp> out;
    if (S.s == 0) return out;
    const Mat t = ReduceMat(S.mats.at("T" + std::to_string(p0)), R);
    const Vec cp1 = Charpoly(t, ZMod(l, 1));
    FP f;
    for (u128 x : cp1) f.push_back(static_cast<uint64_t>(x % l));
    const std::vector<FP> facs = FactorSquarefreeModEll(f, l);
    for (const FP& g : facs) {
        FP H{1};
        for (const FP& h : facs)
            if (h != g) H = MulP(H, h, l);
        Vec Hv(H.begin(), H.end());
        Mat P = PolyOfMat(Hv, t, R);
        uint64_t e = 1;
        while (e < static_cast<uint64_t>(R.W) * S.s) {
            P = MatMul(P, P, R);
            e *= 2;
        }
        const Basis b = MakeBasis(P, R);
        const size_t r = b.Rinv.size();
        ENGINE_CHECK_MSG(r == g.size() - 1, "HeckeLambdaQExp: component rank != residue degree (not etale)");
        WeightComp wc;
        wc.g = g;
        const Mat tg = RestrictTo(t, b, R);
        wc.F = Charpoly(tg, R);
        for (size_t i = 0; i <= r; ++i)
            ENGINE_CHECK_MSG(static_cast<uint64_t>(wc.F[i] % l) == g[i], "HeckeLambdaQExp: charpoly on V_g != g mod ell");
        for (const auto& name : S.names) wc.P[name] = InPowers(RestrictTo(ReduceMat(S.mats.at(name), R), b, R), tg, R);
        out.push_back(std::move(wc));
    }
    return out;
}

std::string PText(const Vec& p) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < p.size(); ++i) os << (i ? "," : "") << U128Text(p[i]);
    os << "]";
    return os.str();
}

std::string QText(const QElem& x) {
    std::string s = "[";
    for (size_t i = 0; i < x.size(); ++i) s += (i ? "," : "") + PText(x[i]);
    return s + "]";
}

bool IsPrime(uint32_t n) {
    if (n < 2) return false;
    for (uint32_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

}  // namespace

// ===========================================================================
// F_ell factorisation
// ===========================================================================
namespace {
std::vector<std::vector<uint64_t>> FactorSquarefreeModEll(const std::vector<uint64_t>& f0, uint64_t l) {
    ENGINE_CHECK_MSG(l >= 3 && l < (uint64_t{1} << 32), "FactorSquarefreeModEll: odd ell < 2^32");
    FP f = Monic(f0, l);
    ENGINE_CHECK_MSG(f.size() >= 1, "FactorSquarefreeModEll: zero polynomial");
    ENGINE_CHECK_MSG(SquarefreeL(f, l), "FactorSquarefreeModEll REFUSED: not squarefree mod ell");
    std::vector<FP> out;
    std::mt19937_64 rng(20261005);
    FP h{0, 1};
    for (uint32_t d = 1; f.size() > 1; ++d) {
        if (2 * d > f.size() - 1) {
            out.push_back(f);
            break;
        }
        h = PowMod(h, l, f, l);
        const FP g = Gcd(f, SubX(h, l), l);
        if (g.size() > 1) {
            EqualDegree(g, d, l, rng, out);
            f = Monic(DivMod(f, g, l).first, l);
            if (f.size() > 1) h = DivMod(h, f, l).second;
        }
    }
    std::sort(out.begin(), out.end(), [](const FP& x, const FP& y) {
        if (x.size() != y.size()) return x.size() < y.size();
        return std::lexicographical_compare(x.rbegin(), x.rend(), y.rbegin(), y.rend());
    });
    return out;
}
} // namespace

// ===========================================================================
// the ring Lambda_{K,m}(x_c)[Y]/(F)
// ===========================================================================
Vec ZMulTrunc(const Vec& x, const Vec& y, const ZMod& RK, uint32_t m) {
    Vec out(m, 0);
    for (size_t i = 0; i < m && i < x.size(); ++i) {
        if (x[i] == 0) continue;
        for (size_t j = 0; i + j < m && j < y.size(); ++j) out[i + j] = RK.add(out[i + j], RK.mul(x[i], y[j]));
    }
    return out;
}

namespace {
QElem QOne(const ZMod& RK, uint32_t m, uint32_t r) {
    QElem x(r, Vec(m, 0));
    x[0][0] = 1 % RK.M;
    return x;
}
} // namespace

QElem QScalar(const Vec& z, uint32_t r) {
    QElem x(r, Vec(z.size(), 0));
    x[0] = z;
    return x;
}

QElem QAdd(const QElem& x, const QElem& y, const ZMod& RK) {
    QElem out = x;
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t t = 0; t < out[i].size(); ++t) out[i][t] = RK.add(out[i][t], y[i][t]);
    return out;
}

QElem QSub(const QElem& x, const QElem& y, const ZMod& RK) {
    QElem out = x;
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t t = 0; t < out[i].size(); ++t) out[i][t] = RK.sub(out[i][t], y[i][t]);
    return out;
}

QElem QMul(const QElem& x, const QElem& y, const std::vector<Vec>& F, const ZMod& RK, uint32_t m) {
    const size_t r = x.size();
    ENGINE_CHECK(y.size() == r && F.size() == r + 1);
    std::vector<Vec> prod(2 * r - 1, Vec(m, 0));
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < r; ++j) {
            const Vec z = ZMulTrunc(x[i], y[j], RK, m);
            for (size_t t = 0; t < m; ++t) prod[i + j][t] = RK.add(prod[i + j][t], z[t]);
        }
    for (size_t i = prod.size(); i-- > r;) {   // Y^i = Y^{i-r} Y^r, Y^r = -sum_{t<r} F_t Y^t
        const Vec c = prod[i];
        for (size_t t = 0; t < r; ++t) {
            const Vec z = ZMulTrunc(c, F[t], RK, m);
            for (size_t u = 0; u < m; ++u) prod[i - r + t][u] = RK.sub(prod[i - r + t][u], z[u]);
        }
    }
    prod.resize(r);
    return prod;
}

// ===========================================================================
// weights, specialisation
// ===========================================================================
u128 XAt(uint64_t ell, uint64_t k, uint32_t e) {
    const ZMod R(ell, e);
    return R.sub(R.pow((ell + 1) % R.M, k), 1 % R.M);
}

namespace {
uint32_t VDiff(uint64_t ell, int64_t kprime, int64_t c) {
    if (kprime == c) return 1000;
    uint64_t d = static_cast<uint64_t>(kprime > c ? kprime - c : c - kprime);
    uint32_t v = 1;
    while (d % ell == 0) {
        d /= ell;
        ++v;
    }
    return v;
}
} // namespace

uint32_t SpecPrecision(uint64_t ell, uint32_t K, uint32_t m, uint32_t centre_k, int64_t kprime) {
    const uint64_t v = VDiff(ell, kprime, centre_k);
    return static_cast<uint32_t>(std::min<uint64_t>(K, static_cast<uint64_t>(m) * v));
}

Vec SpecialiseAt(const QElem& x, uint64_t ell, uint32_t centre_k, int64_t kprime, uint32_t prec) {
    ENGINE_CHECK(kprime >= 0 && prec >= 1);
    const ZMod R(ell, prec);
    const u128 z = R.sub(XAt(ell, static_cast<uint64_t>(kprime), prec), XAt(ell, centre_k, prec));
    Vec out;
    for (const Vec& c : x) {
        u128 acc = 0;
        for (size_t t = c.size(); t-- > 0;) acc = R.add(R.mul(acc, z), c[t] % R.M);
        out.push_back(acc);
    }
    return out;
}

Vec EpsilonBinomial(uint64_t ell, uint64_t p, uint32_t K, uint32_t m, uint32_t centre_k) {
    ENGINE_CHECK_MSG(p % ell != 0 && m < ell && K >= 1 && centre_k >= 1, "EpsilonBinomial: p != ell, m < ell");
    const ZMod R1(ell, K + 1), RK(ell, K);
    const u128 u = (ell + 1) % R1.M;
    // omega(p) = p^{ell^K} mod ell^{K+1}; <p> = p / omega(p)
    u128 om = p % R1.M;
    for (uint32_t i = 0; i < K; ++i) om = R1.pow(om, ell);
    const u128 br = R1.mul(p % R1.M, R1.inv(om));
    ENGINE_CHECK_MSG(br % ell == 1 % ell, "EpsilonBinomial: <p> != 1 mod ell");
    // s mod ell^K with u^s = <p> mod ell^{K+1}, digit by digit
    u128 s = 0;
    const u128 uinv = R1.inv(u);
    for (uint32_t i = 0; i < K; ++i) {
        const u128 y = R1.mul(br, R1.pow(uinv, static_cast<uint64_t>(s)));
        ENGINE_CHECK_MSG(R1.sub(y, 1) % R1.pw[i + 1] == 0, "EpsilonBinomial: digit invariant");
        const u128 ti = (R1.sub(y, 1) / R1.pw[i + 1]) % ell;
        const u128 base = R1.sub(R1.pow(u, static_cast<uint64_t>(R1.pw[i])), 1);
        const u128 bi = (base / R1.pw[i + 1]) % ell;
        const u128 d = (ti * PowL(static_cast<uint64_t>(bi), ell - 2, ell)) % ell;
        s += d * R1.pw[i];
    }
    ENGINE_CHECK_MSG(R1.mul(br, R1.pow(uinv, static_cast<uint64_t>(s))) == 1 % R1.M, "EpsilonBinomial: discrete log");
    // p^{c-1} binom(s, j) u^{-cj}
    const u128 pc = RK.pow(p % RK.M, centre_k - 1);
    const u128 uc_inv = RK.inv(RK.pow((ell + 1) % RK.M, centre_k));
    Vec out(m, 0);
    u128 binom = 1 % RK.M, ucj = 1 % RK.M;
    for (uint32_t j = 0; j < m; ++j) {
        out[j] = RK.mul(pc, RK.mul(binom, ucj));
        // binom(s, j+1) = binom(s, j) (s - j) / (j + 1)
        binom = RK.mul(RK.mul(binom, RK.sub(s % RK.M, j % RK.M)), RK.inv((j + 1) % RK.M));
        ucj = RK.mul(ucj, uc_inv);
    }
    return out;
}

// ===========================================================================
// the build
// ===========================================================================
LambdaQExpansion BuildLambdaQExpansion(const std::vector<FormsWeight>& ws0, uint32_t a, uint32_t K, uint32_t m,
                                       uint32_t centre_k, uint32_t B, uint64_t p0, bool with_eisenstein) {
    ENGINE_CHECK_MSG(!ws0.empty() && K >= 1 && m >= 1 && B >= 1, "BuildLambdaQExpansion: shape");
    const uint64_t l = ws0[0].ell;
    const uint32_t N = K + m - 1, Kp = K + m - 1;
    ENGINE_CHECK_MSG(ws0.size() >= N, "BuildLambdaQExpansion REFUSED: fewer than K + m - 1 weights (lem:u-wtprod)");
    std::vector<const FormsWeight*> ws;
    for (size_t j = 0; j < N; ++j) ws.push_back(&ws0[j]);
    for (const auto* w : ws) {
        ENGINE_CHECK_MSG(w->ell == l && (w->k + 1000000 * (l - 1) - a) % (l - 1) == 0,
                         "BuildLambdaQExpansion: a weight outside the class");
        ENGINE_CHECK_MSG(w->W >= Kp, "BuildLambdaQExpansion REFUSED: forms data below ell^{K+m-1} (thm:u-interp(2))");
    }
    ENGINE_CHECK_MSG((centre_k + 1000000 * (l - 1) - a) % (l - 1) == 0, "BuildLambdaQExpansion: centre outside the class");
    std::vector<uint32_t> ks;
    for (const auto* w : ws) ks.push_back(w->k);
    const ZMod RK(l, K), Rp(l, Kp);
    std::vector<uint32_t> primes;
    for (uint32_t p = 2; p <= B; ++p)
        if (IsPrime(p) && p != l) primes.push_back(p);
    for (const auto* w : ws)
        for (uint32_t p : primes)
            ENGINE_CHECK_MSG(w->S.mats.count("T" + std::to_string(p)) || w->S.s == 0,
                             ("BuildLambdaQExpansion: forms data lack T" + std::to_string(p)).c_str());

    LambdaQExpansion L;
    L.ell = l;
    L.a = a;
    L.K = K;
    L.m = m;
    L.centre_k = centre_k;
    L.B = B;
    L.N = N;
    L.W = Kp;
    L.weights = ks;
    auto interp = [&](const Vec& vals) { return InterpolateTruncation(l, ks, vals, Kp, K, m, centre_k); };

    // eps_p(X): p^{k-1} at x_k
    for (uint32_t p : primes) {
        Vec v;
        for (uint32_t k : ks) v.push_back(Rp.pow(p % Rp.M, k - 1));
        L.eps[p] = interp(v);
    }
    if (a % (l - 1) != 0) {
        const auto BK = BernoulliOverK(l, Kp, ks);
        Vec gv;
        for (uint32_t k : ks)
            gv.push_back(Rp.neg(Rp.mul(Rp.sub(1 % Rp.M, Rp.pow(l % Rp.M, k - 1)), BK.at(k))));
        L.G = interp(gv);
        L.has_G = true;
    }

    // ---- the cuspidal components
    const bool any_cusp = ws[0]->S.s > 0;
    for (const auto* w : ws) ENGINE_CHECK_MSG(w->S.s == ws[0]->S.s, "BuildLambdaQExpansion: the ordinary rank varies");
    if (any_cusp) {
        if (p0 == 0) {
            for (uint32_t p : primes) {
                bool ok = true;
                for (const auto* w : ws) {
                    const Vec cp = Charpoly(ReduceMat(w->S.mats.at("T" + std::to_string(p)), ZMod(l, 1)), ZMod(l, 1));
                    FP f;
                    for (u128 x : cp) f.push_back(static_cast<uint64_t>(x % l));
                    ok = ok && SquarefreeL(f, l);
                }
                if (ok) {
                    p0 = p;
                    break;
                }
            }
            ENGINE_CHECK_MSG(p0 != 0, "BuildLambdaQExpansion REFUSED: no T_p, p <= B, is residually squarefree (not etale)");
        }
        L.p0 = p0;
        const ZMod Rw(l, ws[0]->W);
        std::vector<std::vector<WeightComp>> per;
        for (const auto* w : ws) per.push_back(ComponentsAt(*w, p0, ZMod(l, w->W)));
        for (const auto& pc : per) {
            ENGINE_CHECK_MSG(pc.size() == per[0].size(), "BuildLambdaQExpansion: the residual systems vary across weights");
            for (size_t i = 0; i < pc.size(); ++i)
                ENGINE_CHECK_MSG(pc[i].g == per[0][i].g, "BuildLambdaQExpansion: the residual systems vary across weights");
        }
        const std::string gname = "T" + std::to_string(p0);
        for (size_t c = 0; c < per[0].size(); ++c) {
            LambdaComponent C;
            const uint32_t r = static_cast<uint32_t>(per[0][c].g.size() - 1);
            C.rank = r;
            C.gen = gname;
            C.residual = per[0][c].g;
            // residually Eisenstein: r = 1 and the root = 1 + p0^{k-1} mod ell
            const uint64_t root = (l - per[0][c].g[0]) % l;
            C.kind = (r == 1 && root == (1 + PowL(p0, ks[0] - 1, l)) % l) ? "cusp_eis" : "cusp";
            auto interp_coef = [&](const std::function<u128(size_t)>& at) {
                Vec v;
                for (size_t j = 0; j < N; ++j) v.push_back(at(j) % Rp.M);
                return interp(v);
            };
            C.F.resize(r + 1);
            for (uint32_t i = 0; i < r; ++i) C.F[i] = interp_coef([&](size_t j) { return per[j][c].F[i]; });
            C.F[r] = Vec(m, 0);
            C.F[r][0] = 1 % RK.M;
            auto op_elem = [&](const std::string& name) {
                QElem x(r);
                for (uint32_t i = 0; i < r; ++i) x[i] = interp_coef([&](size_t j) { return per[j][c].P.at(name)[i]; });
                return x;
            };
            for (uint32_t p : primes) C.ap[p] = op_elem("T" + std::to_string(p));
            C.alpha = op_elem("U");
            L.comps.push_back(std::move(C));
        }
    }
    if (with_eisenstein) {
        ENGINE_CHECK_MSG(L.has_G, "BuildLambdaQExpansion REFUSED: the Eisenstein series at a = 0 mod ell - 1 (pole)");
        LambdaComponent E;
        E.kind = "eisenstein";
        E.rank = 1;
        E.gen = "-";
        E.F = {Vec(m, 0), Vec(m, 0)};
        E.F[1][0] = 1 % RK.M;   // F = Y (the presentation Lambda[Y]/(Y) = Lambda)
        const QElem one = QOne(RK, m, 1);
        for (uint32_t p : primes) E.ap[p] = QAdd(one, QScalar(L.eps[p], 1), RK);
        E.alpha = one;
        L.comps.push_back(std::move(E));
    }

    // ---- the Hecke recursions
    for (auto& C : L.comps) {
        const uint32_t r = C.rank;
        const QElem one = QOne(RK, m, r);
        std::vector<QElem> a_n(B + 1, QElem(r, Vec(m, 0)));
        a_n[1] = one;
        // prime powers
        std::map<uint64_t, QElem> pp;   // p^e -> a_{p^e}
        for (uint32_t p = 2; p <= B; ++p) {
            if (!IsPrime(p)) continue;
            const QElem ap = p == l ? C.alpha : C.ap.at(p);
            QElem prev = one, cur = ap;
            uint64_t q = p;
            pp[q] = cur;
            while (q * p <= B) {
                QElem nxt = QMul(ap, cur, C.F, RK, m);
                if (p != l) nxt = QSub(nxt, QMul(QScalar(L.eps.at(p), r), prev, C.F, RK, m), RK);
                prev = cur;
                cur = nxt;
                q *= p;
                pp[q] = cur;
            }
        }
        for (uint32_t n = 2; n <= B; ++n) {
            uint32_t p = 2;
            while (n % p) ++p;
            uint32_t q = 1, rest = n;
            while (rest % p == 0) {
                rest /= p;
                q *= p;
            }
            a_n[n] = rest == 1 ? pp.at(q) : QMul(pp.at(q), a_n[rest], C.F, RK, m);
        }
        if (C.kind == "eisenstein") {
            const u128 half = RK.inv(2 % RK.M);
            a_n[0] = QScalar(L.G, 1);
            for (auto& x : a_n[0][0]) x = RK.mul(x, half);
        }
        C.a = std::move(a_n);
    }
    return L;
}

// ===========================================================================
// the record hecke_lambda_qexp/v1
// ===========================================================================
std::vector<std::string> QExpRecordLines(const LambdaQExpansion& L) {
    std::vector<std::string> out;
    for (size_t c = 0; c < L.comps.size(); ++c) {
        const LambdaComponent& C = L.comps[c];
        std::ostringstream os;
        os << "HECKE_LAMBDA_QEXP ell=" << L.ell << " a=" << L.a << " K=" << L.K << " m=" << L.m << " centre_k=" << L.centre_k
           << " N=" << L.N << " W=" << L.W << " B=" << L.B << " p0=" << L.p0 << " component=" << c << " kind=" << C.kind
           << " rank=" << C.rank << " gen=" << C.gen << " residual=[";
        for (size_t i = 0; i < C.residual.size(); ++i) os << (i ? "," : "") << C.residual[i];
        os << "] F=[";
        for (size_t i = 0; i < C.F.size(); ++i) os << (i ? "," : "") << PText(C.F[i]);
        os << "] alpha=" << QText(C.alpha) << " q=[";
        for (size_t n = 0; n < C.a.size(); ++n) os << (n ? ";" : "") << QText(C.a[n]);
        os << "]";
        const std::string line = os.str();
        std::ostringstream hx;
        hx << std::hex << engine::FnvBytes(engine::kFnvOffset64, line);
        out.push_back(line + " digest=0x" + hx.str());
    }
    return out;
}

}  // namespace lambda
}  // namespace hecke_ext
