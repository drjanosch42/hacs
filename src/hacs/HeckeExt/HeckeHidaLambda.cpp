// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-HL) the Lambda stage of the truncated
// Lambda-adic ordinary Hecke algebra: Bernoulli side, thm:u-interp with its
// precision bound enforced, Howell spans over Z/ell^W, the component split,
// the monogenic and general routes, the record hecke_lambda/v1.  A port of
// docs/implementation/slot19/r192/{algebra.py, lambda_trunc.py} (R19-2), whose
// JSON (r192/data/lambda_*.json) HeckeLambdaValidation reproduces exactly.
// See HeckeHidaLambda.h for the contract references.
#include "HeckeHidaLambda.h"

#include <algorithm>
#include <sstream>

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "NumberField/BigInt.h"
#include "NumberField/BigRational.h"

namespace hecke_ext {
namespace lambda {

namespace {

using nf::BigInt;
using nf::BigRational;

BigInt BigFromU128(u128 x) { return BigInt::from_string(U128Text(x)); }

u128 U128FromBig(const BigInt& b) {
    ENGINE_CHECK(!b.is_neg());
    const auto& mag = b.magnitude();
    ENGINE_CHECK_MSG(mag.size() <= 4, "U128FromBig: value >= 2^128");
    u128 x = 0;
    for (size_t i = mag.size(); i-- > 0;) x = (x << 32) | mag[i];
    return x;
}

BigInt BigPow(uint64_t ell, uint32_t e) { return BigInt::pow(BigInt::from_u64(ell), e); }

BigInt BigPowMod(const BigInt& base, uint64_t e, const BigInt& mod) {
    BigInt r = BigInt(1).floor_mod(mod), b = base.floor_mod(mod);
    while (e) {
        if (e & 1u) r = (r * b).floor_mod(mod);
        e >>= 1;
        if (e) b = (b * b).floor_mod(mod);
    }
    return r;
}

}  // namespace

// (slot-26 Z6) exported, body unchanged (HeckeHidaLambda.h)
// x_k = (1 + ell)^k - 1 mod ell^e (algebra.py x_of, k >= 0)
BigInt XOf(uint64_t ell, uint64_t k, uint32_t e) {
    const BigInt mod = BigPow(ell, e);
    return (BigPowMod(BigInt::from_u64(ell + 1), k, mod) - BigInt(1)).floor_mod(mod);
}

namespace {

u128 RatMod(const BigRational& q, const ZMod& R) {
    if (q.is_zero()) return 0;
    ENGINE_CHECK_MSG(q.den().mod_u32(static_cast<uint32_t>(R.ell)) != 0, "interpolation polynomial not ell-integral");
    const BigInt Mb = BigFromU128(R.M);
    const u128 n = U128FromBig(q.num().floor_mod(Mb));
    const u128 d = U128FromBig(q.den().floor_mod(Mb));
    return R.mul(n, R.inv(d));
}

// algebra.py newton_interp: the exact interpolant (divided differences in Q).
std::vector<BigRational> NewtonExact(const std::vector<BigInt>& xs, const std::vector<BigInt>& vals) {
    const size_t N = xs.size();
    std::vector<BigRational> D(N), coef;
    for (size_t j = 0; j < N; ++j) D[j] = BigRational(vals[j]);
    coef.push_back(D[0]);
    for (size_t i = 1; i < N; ++i) {
        std::vector<BigRational> nd(N - i);
        for (size_t j = 0; j < N - i; ++j) nd[j] = (D[j + 1] - D[j]) / BigRational(xs[j + i] - xs[j]);
        D.swap(nd);
        coef.push_back(D[0]);
    }
    std::vector<BigRational> poly(N, BigRational(0)), basis{BigRational(1)};
    for (size_t i = 0; i < N; ++i) {
        for (size_t t = 0; t < basis.size(); ++t) poly[t] += coef[i] * basis[t];
        std::vector<BigRational> nb(basis.size() + 1, BigRational(0));
        for (size_t t = 0; t < basis.size(); ++t) {
            nb[t + 1] += basis[t];
            nb[t] -= BigRational(xs[i]) * basis[t];
        }
        basis.swap(nb);
    }
    return poly;
}

}  // namespace

// (slot-26 Z6) exported, body unchanged (HeckeHidaLambda.h)
Vec NewtonMod(const std::vector<BigInt>& xs, const Vec& vals, const ZMod& R) {
    std::vector<BigInt> v;
    for (u128 x : vals) v.push_back(BigFromU128(x));
    const auto poly = NewtonExact(xs, v);
    Vec out;
    for (const auto& c : poly) out.push_back(RatMod(c, R));
    return out;
}

u128 PEval(const Vec& p, u128 x, const ZMod& R) {
    u128 r = 0;
    x %= R.M;
    for (size_t i = p.size(); i-- > 0;) r = R.add(R.mul(r, x), p[i] % R.M);
    return r;
}

namespace {

// q(Z) = p(Z + xc), mod R.M
Vec Recentre(const Vec& p, u128 xc, const ZMod& R) {
    Vec out(p.size(), 0);
    std::vector<Vec> binom(p.size() + 1);
    for (size_t i = 0; i <= p.size(); ++i) {
        binom[i].assign(i + 1, 1 % R.M);
        for (size_t t = 1; t < i; ++t) binom[i][t] = R.add(binom[i - 1][t - 1], binom[i - 1][t]);
    }
    for (size_t i = 0; i < p.size(); ++i)
        for (size_t t = 0; t <= i; ++t)
            out[t] = R.add(out[t], R.mul(R.mul(p[i] % R.M, binom[i][t]), R.pow(xc, i - t)));
    return out;
}

Vec Trunc(const Vec& p, const ZMod& RK, uint32_t m) {
    Vec q(m, 0);
    for (size_t i = 0; i < m && i < p.size(); ++i) q[i] = p[i] % RK.M;
    return q;
}

// ---------------------------------------------------------------------------
// Howell spans over Z/ell^W (algebra.py Span), with Python's insertion-ordered
// pivot dictionary replicated (the order of Z.piv.items() feeds the J generators).
// ---------------------------------------------------------------------------
class Span {
public:
    struct Entry {
        size_t col;
        uint32_t e;
        Vec row;
    };
    Span(size_t n, const ZMod& R) : n_(n), R_(R), pos_(n, -1) {}

    std::pair<Vec, long> reduce(Vec v) const {
        for (auto& x : v) x %= R_.M;
        for (size_t c = 0; c < n_; ++c) {
            if (v[c] == 0) continue;
            if (pos_[c] < 0) return {v, static_cast<long>(c)};
            const Entry& en = ent_[static_cast<size_t>(pos_[c])];
            const uint32_t e = R_.val(v[c], 1000000000u);
            if (e < en.e) return {v, static_cast<long>(c)};
            const u128 f = (v[c] / R_.pw[en.e]) % R_.M;
            for (size_t i = 0; i < n_; ++i) v[i] = R_.sub(v[i], R_.mul(f, en.row[i]));
        }
        return {v, -1};
    }
    bool insert(const Vec& v0) {
        bool grew = false;
        std::vector<Vec> stack{v0};
        while (!stack.empty()) {
            Vec top = std::move(stack.back());
            stack.pop_back();
            auto rc = reduce(std::move(top));
            if (rc.second < 0) continue;
            grew = true;
            Vec w = std::move(rc.first);
            const size_t c = static_cast<size_t>(rc.second);
            const uint32_t e = R_.val(w[c], 1000000000u);
            const u128 iu = R_.inv(w[c] / R_.pw[e]);
            for (auto& x : w) x = R_.mul(x, iu);
            if (pos_[c] >= 0) {
                Entry& en = ent_[static_cast<size_t>(pos_[c])];
                stack.push_back(en.row);
                en.e = e;
                en.row = w;
            } else {
                pos_[c] = static_cast<long>(ent_.size());
                ent_.push_back(Entry{c, e, w});
            }
            Vec cl(n_);
            for (size_t i = 0; i < n_; ++i) cl[i] = R_.mul(w[i], R_.pw[R_.W - e]);
            stack.push_back(std::move(cl));
        }
        return grew;
    }
    bool contains(const Vec& v) const { return reduce(v).second < 0; }
    uint32_t log_order() const {
        uint32_t s = 0;
        for (const auto& en : ent_) s += R_.W - en.e;
        return s;
    }
    std::vector<Vec> rows() const {
        std::vector<Vec> out;
        for (size_t c = 0; c < n_; ++c)
            if (pos_[c] >= 0) out.push_back(ent_[static_cast<size_t>(pos_[c])].row);
        return out;
    }
    const std::vector<Entry>& entries() const { return ent_; }   // insertion order

private:
    size_t n_;
    ZMod R_;
    std::vector<long> pos_;
    std::vector<Entry> ent_;
};

Span IdealSpan(const std::vector<Vec>& gens, const ZMod& RK, uint32_t m) {
    Span S(m, RK);
    for (const Vec& g0 : gens) {
        const Vec g = Trunc(g0, RK, m);
        for (uint32_t e = 0; e < m; ++e) {
            Vec v(m, 0);
            for (uint32_t i = 0; i + e < m; ++i) v[e + i] = g[i];
            S.insert(v);
        }
    }
    return S;
}

bool IdealsEqual(const Span& I, const Span& J) {
    for (const Vec& r : I.rows())
        if (!J.contains(r)) return false;
    for (const Vec& r : J.rows())
        if (!I.contains(r)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// small block helpers (lambda_trunc.py)
// ---------------------------------------------------------------------------
}  // namespace

// (slot-26 Z6) exported, body unchanged (HeckeHidaLambda.h)
Mat MAddScaledIdent(const Mat& A, u128 c, const ZMod& R) {   // A - c I
    Mat B = A;
    for (size_t i = 0; i < B.size(); ++i) {
        for (auto& x : B[i]) x %= R.M;
        B[i][i] = R.sub(B[i][i], c % R.M);
    }
    return B;
}

std::pair<std::vector<size_t>, Mat> BasisData(const Mat& Bm, const ZMod& R) {
    const size_t n = Bm.size();
    const size_t s = n ? Bm[0].size() : 0;
    if (s == 0) return {{}, {}};
    Mat Bt(s, Vec(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t t = 0; t < s; ++t) Bt[t][i] = Bm[i][t];
    std::vector<size_t> rows = IndepColsModEll(Bt, n, R.ell);
    if (rows.size() > s) rows.resize(s);
    Mat Rm(s, Vec(s));
    for (size_t i = 0; i < s; ++i)
        for (size_t t = 0; t < s; ++t) Rm[i][t] = Bm[rows[i]][t];
    return {rows, InvUnitMatrix(Rm, R)};
}

Mat Restrict(const Mat& A, const Mat& Bm, const std::vector<size_t>& rows, const Mat& Rinv, const ZMod& R) {
    const size_t s = Rinv.size();
    const Mat AB = MatMul(A, Bm, R);
    Mat C(s, Vec(s, 0));
    for (size_t t = 0; t < s; ++t) {
        Vec yr(s);
        for (size_t i = 0; i < s; ++i) yr[i] = AB[rows[i]][t];
        const Vec coord = MatVec(Rinv, yr, R);
        for (size_t i = 0; i < A.size(); ++i) {
            u128 acc = 0;
            for (size_t u = 0; u < s; ++u) acc = R.add(acc, R.mul(Bm[i][u] % R.M, coord[u]));
            ENGINE_CHECK_MSG(acc == AB[i][t], "HeckeHidaLambda: summand not stable");
        }
        for (size_t u = 0; u < s; ++u) C[u][t] = coord[u];
    }
    return C;
}

// V_c = im t^N, V_E = ker t^N (N >= W n): bases (n x r) of both.
std::pair<Mat, Mat> FittingSplit(const Mat& t, const ZMod& R) {
    const size_t n = t.size();
    uint64_t N = 1;
    Mat P = t;
    while (N < static_cast<uint64_t>(R.W) * std::max<size_t>(n, 1)) {
        P = MatMul(P, P, R);
        N *= 2;
    }
    const std::vector<size_t> cols = IndepColsModEll(P, n, R.ell);
    if (cols.empty()) return {Ident(n), Mat(n)};
    Mat Bc(n, Vec(cols.size()));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < cols.size(); ++j) Bc[i][j] = P[i][cols[j]];
    const auto bd = BasisData(Bc, R);
    const size_t rk = cols.size();
    Mat Cm(rk, Vec(n));
    for (size_t j = 0; j < n; ++j) {
        Vec pr(rk);
        for (size_t i = 0; i < rk; ++i) pr[i] = P[bd.first[i]][j];
        const Vec c = MatVec(bd.second, pr, R);
        for (size_t u = 0; u < rk; ++u) Cm[u][j] = c[u];
    }
    const std::vector<size_t> J = IndepColsModEll(Cm, n, R.ell);
    Mat CJ(rk, Vec(J.size()));
    for (size_t u = 0; u < rk; ++u)
        for (size_t jj = 0; jj < J.size(); ++jj) CJ[u][jj] = Cm[u][J[jj]];
    const Mat CJinv = InvUnitMatrix(CJ, R);
    std::vector<bool> inJ(n, false);
    for (size_t j : J) inJ[j] = true;
    std::vector<Vec> kern;
    for (size_t i = 0; i < n; ++i) {
        if (inJ[i]) continue;
        Vec v(n, 0);
        v[i] = 1 % R.M;
        Vec ci(rk);
        for (size_t u = 0; u < rk; ++u) ci[u] = Cm[u][i];
        const Vec sol = MatVec(CJinv, ci, R);
        for (size_t jj = 0; jj < J.size(); ++jj) v[J[jj]] = R.neg(sol[jj]);
        kern.push_back(v);
    }
    Mat BE(n, Vec(kern.size()));
    for (size_t i = 0; i < n; ++i)
        for (size_t c = 0; c < kern.size(); ++c) BE[i][c] = kern[c][i];
    Mat full(n, Vec());
    for (size_t i = 0; i < n; ++i) {
        full[i] = BE[i];
        full[i].insert(full[i].end(), Bc[i].begin(), Bc[i].end());
    }
    ENGINE_CHECK_MSG(IndepColsModEll(full, n, R.ell).size() == n, "V_E + V_c is not everything");
    return {BE, Bc};
}

bool NilpotentModEll(const Mat& A, uint64_t ell) {
    const size_t r = A.size();
    if (r == 0) return true;
    const ZMod F(ell, 1);
    Mat P = A;
    for (auto& row : P)
        for (auto& x : row) x %= ell;
    Mat Q = P;
    for (size_t i = 1; i < r; ++i) Q = MatMul(Q, P, F);
    for (const auto& row : Q)
        for (u128 x : row)
            if (x) return false;
    return true;
}

bool GeneratesModEll(const Mat& t, uint64_t ell) {
    const size_t r = t.size();
    if (r == 0) return true;
    const ZMod F(ell, 1);
    Mat P = Ident(r), tl = t;
    for (auto& row : tl)
        for (auto& x : row) x %= ell;
    Mat V(r * r, Vec(r, 0));   // columns: flattened powers
    for (size_t i = 0; i < r; ++i) {
        for (size_t a = 0; a < r; ++a)
            for (size_t b = 0; b < r; ++b) V[a * r + b][i] = P[a][b] % ell;
        P = MatMul(P, tl, F);
    }
    return IndepColsModEll(V, r, ell).size() == r;
}

Vec PolyCoords(const Mat& target, const Mat& t, const ZMod& R) {
    const size_t r = t.size();
    Mat P = Ident(r);
    Mat A(r * r, Vec(r, 0));
    for (size_t i = 0; i < r; ++i) {
        for (size_t a = 0; a < r; ++a)
            for (size_t b = 0; b < r; ++b) A[a * r + b][i] = P[a][b];
        P = MatMul(P, t, R);
    }
    Mat At(r, Vec(r * r));
    for (size_t i = 0; i < r * r; ++i)
        for (size_t u = 0; u < r; ++u) At[u][i] = A[i][u];
    std::vector<size_t> rows = IndepColsModEll(At, r * r, R.ell);
    if (rows.size() > r) rows.resize(r);
    ENGINE_CHECK_MSG(rows.size() == r, "PolyCoords: t does not generate mod ell");
    Mat Ar(r, Vec(r));
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
        ENGINE_CHECK_MSG(acc == tv[i], "PolyCoords: target not in Z[t]");
    }
    return c;
}

namespace {

// ---- F_ell[Y] (etale test, residue degrees) -----------------------------------
using FPoly = std::vector<uint64_t>;
FPoly FTrim(FPoly a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
    return a;
}
uint64_t FInv(uint64_t a, uint64_t l) {
    uint64_t r = 1, b = a % l, e = l - 2;
    while (e) {
        if (e & 1u) r = static_cast<uint64_t>(static_cast<u128>(r) * b % l);
        e >>= 1;
        b = static_cast<uint64_t>(static_cast<u128>(b) * b % l);
    }
    return r;
}
std::pair<FPoly, FPoly> FDivMod(FPoly a, FPoly b, uint64_t l) {
    for (auto& x : a) x %= l;
    for (auto& x : b) x %= l;
    a = FTrim(a);
    b = FTrim(b);
    ENGINE_CHECK(!b.empty());
    FPoly q(std::max<long>(static_cast<long>(a.size()) - static_cast<long>(b.size()) + 1, 1), 0);
    const uint64_t ib = FInv(b.back(), l);
    while (a.size() >= b.size() && !a.empty()) {
        const uint64_t c = static_cast<uint64_t>(static_cast<u128>(a.back()) * ib % l);
        const size_t d = a.size() - b.size();
        q[d] = c;
        for (size_t i = 0; i < b.size(); ++i)
            a[i + d] = (a[i + d] + l - static_cast<uint64_t>(static_cast<u128>(c) * b[i] % l)) % l;
        a = FTrim(a);
    }
    return {q, a};
}
FPoly FGcd(FPoly a, FPoly b, uint64_t l) {
    for (auto& x : a) x %= l;
    for (auto& x : b) x %= l;
    a = FTrim(a);
    b = FTrim(b);
    while (!b.empty()) {
        FPoly r = FDivMod(a, b, l).second;
        a = b;
        b = r;
    }
    return a;
}
FPoly FMulMod(const FPoly& a, const FPoly& b, const FPoly& f, uint64_t l) {
    FPoly out(a.size() + b.size(), 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            out[i + j] = (out[i + j] + static_cast<uint64_t>(static_cast<u128>(a[i]) * b[j] % l)) % l;
    return FDivMod(out, f, l).second;
}
FPoly FPowMod(FPoly a, uint64_t e, const FPoly& f, uint64_t l) {
    FPoly r{1};
    a = FDivMod(a, f, l).second;
    while (e) {
        if (e & 1u) r = FMulMod(r, a, f, l);
        e >>= 1;
        if (e) a = FMulMod(a, a, f, l);
    }
    return r;
}
std::vector<uint32_t> ResidueDegrees(const Vec& F, uint64_t l) {
    FPoly f;
    for (u128 x : F) f.push_back(static_cast<uint64_t>(x % l));
    f = FTrim(f);
    std::vector<uint32_t> degs;
    uint32_t d = 1;
    FPoly h{0, 1};
    while (f.size() > 1) {
        if (2 * d > f.size() - 1) {
            degs.push_back(static_cast<uint32_t>(f.size() - 1));
            break;
        }
        h = FPowMod(h, l, f, l);
        FPoly hx = h;
        if (hx.size() < 2) hx.resize(2, 0);
        hx[1] = (hx[1] + l - 1) % l;
        const FPoly g = FGcd(f, FTrim(hx), l);
        if (g.size() > 1) {
            for (size_t i = 0; i < (g.size() - 1) / d; ++i) degs.push_back(d);
            f = FDivMod(f, g, l).first;
            f = FTrim(f);
            if (f.size() > 1) h = FDivMod(h, f, l).second;
        }
        ++d;
    }
    std::sort(degs.begin(), degs.end());
    return degs;
}
bool SquarefreeModEll(const Vec& F, uint64_t l) {
    FPoly f, df;
    for (u128 x : F) f.push_back(static_cast<uint64_t>(x % l));
    for (size_t i = 1; i < f.size(); ++i) df.push_back(static_cast<uint64_t>(static_cast<u128>(i % l) * f[i] % l));
    return FGcd(f, df, l).size() == 1;
}

std::string PolyText(const Vec& p) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < p.size(); ++i) os << (i ? "," : "") << U128Text(p[i]);
    os << "]";
    return os.str();
}

// One weight's component split: per space, per factor, the restricted operators.
struct Comp {
    uint32_t r[2][2] = {{0, 0}, {0, 0}};   // [space M/S][E/c]
    std::map<std::string, Mat> ops[2][2];
};

}  // namespace

// ===========================================================================
// Bernoulli side
// ===========================================================================
std::map<uint32_t, u128> BernoulliOverK(uint64_t ell, uint32_t prec, const std::vector<uint32_t>& ks) {
    ENGINE_CHECK(!ks.empty());
    const ZMod Rp(ell, prec);
    uint32_t kmax = 0;
    std::map<uint32_t, uint32_t> vk;
    uint32_t V = 0;
    for (uint32_t k : ks) {
        ENGINE_CHECK_MSG(k >= 2 && k % 2 == 0 && k % (ell - 1) != 0, "BernoulliOverK: even k, k != 0 mod ell-1");
        kmax = std::max(kmax, k);
        const BigInt big = BigInt::pow(BigInt(4), k / 2) - BigInt(1);
        vk[k] = big.valuation(static_cast<uint32_t>(ell));
        V = std::max(V, vk[k]);
    }
    // the tangent numbers T_1..T_n mod Q = ell^{prec + V} (Brent-Harvey)
    u128 Q = 1;
    for (uint32_t i = 0; i < prec + V; ++i) {
        ENGINE_CHECK_MSG(Q < (static_cast<u128>(1) << 110) / ell, "BernoulliOverK: ell^{prec+V} too large");
        Q *= ell;
    }
    const uint32_t n = kmax / 2;
    std::vector<u128> T(n + 1, 0);
    T[1] = 1 % Q;
    for (uint32_t k = 2; k <= n; ++k) T[k] = (static_cast<u128>(k - 1) * T[k - 1]) % Q;
    for (uint32_t k = 2; k <= n; ++k)
        for (uint32_t j = k; j <= n; ++j)
            T[j] = (static_cast<u128>(j - k) * T[j - 1] + static_cast<u128>(j - k + 2) * T[j]) % Q;
    std::map<uint32_t, u128> out;
    for (uint32_t k : ks) {
        const uint32_t nn = k / 2, v = vk[k];
        u128 lv = 1;
        for (uint32_t i = 0; i < v; ++i) lv *= ell;
        ENGINE_CHECK_MSG(T[nn] % lv == 0, "BernoulliOverK: B_k/k not ell-integral");
        const u128 Tp = (T[nn] / lv) % Rp.M;
        const BigInt big = BigInt::pow(BigInt(4), nn) - BigInt(1);
        const BigInt u = big / BigPow(ell, v);
        const u128 um = U128FromBig(u.floor_mod(BigFromU128(Rp.M)));
        u128 r = Rp.mul(Tp, Rp.inv(Rp.mul(Rp.pow(4 % Rp.M, nn), um)));
        if (nn % 2 == 0) r = Rp.neg(r);   // (-1)^{n-1}
        out[k] = r;
    }
    return out;
}

std::map<uint32_t, u128> BernoulliOverKExact(uint64_t ell, uint32_t prec, const std::vector<uint32_t>& ks) {
    const ZMod Rp(ell, prec);
    uint32_t kmax = 0;
    for (uint32_t k : ks) kmax = std::max(kmax, k);
    const auto B = nf::bernoulli_numbers(kmax);
    std::map<uint32_t, u128> out;
    for (uint32_t k : ks) out[k] = RatMod(B[k] / BigRational(static_cast<int64_t>(k)), Rp);
    return out;
}

// ===========================================================================
// thm:u-interp, precision enforced
// ===========================================================================
Vec InterpolateTruncation(uint64_t ell, const std::vector<uint32_t>& weights, const Vec& values, uint32_t Kp,
                          uint32_t K, uint32_t m, uint32_t centre_k) {
    const size_t N = weights.size();
    ENGINE_CHECK_MSG(N == values.size() && N >= 1 && K >= 1 && m >= 1, "InterpolateTruncation: shape");
    for (size_t j = 1; j < N; ++j)
        ENGINE_CHECK_MSG(weights[j] == weights[0] + j * (ell - 1),
                         "InterpolateTruncation REFUSED: weights not consecutive in one class (thm:u-interp(2))");
    ENGINE_CHECK_MSG(N <= ell, "InterpolateTruncation REFUSED: N > ell (thm:u-interp(2))");
    ENGINE_CHECK_MSG(N >= K + m - 1, "InterpolateTruncation REFUSED: N < K + m - 1 weights (lem:u-wtprod)");
    ENGINE_CHECK_MSG(Kp >= K + m - 1, "InterpolateTruncation REFUSED: values known mod ell^Kp, Kp < K + m - 1 "
                                      "(thm:u-interp(2), sharp)");
    const ZMod Rv(ell, Kp), RK(ell, K);
    std::vector<BigInt> xs;
    for (uint32_t k : weights) xs.push_back(XOf(ell, k, Kp + 40));
    Vec v;
    for (u128 x : values) v.push_back(x % Rv.M);
    const Vec p = NewtonMod(xs, v, RK);
    const u128 c = U128FromBig(XOf(ell, centre_k, K + 40).floor_mod(BigFromU128(RK.M)));
    return Trunc(Recentre(p, c, RK), RK, m);
}

// ===========================================================================
// the class driver (lambda_trunc.py run_class)
// ===========================================================================
namespace {

Comp Components(const FormsWeight& w, const ZMod& R, const std::vector<uint64_t>& primes) {
    // the Fitting split of t = sum c_p (T_p - eis_p), p in primes[:4]; the split
    // is certified below, and V_E is independent of the c_p (its basis is the
    // canonical kernel basis), so the prototype's random c_p need not be copied.
    std::vector<std::pair<std::string, uint64_t>> coef;
    for (size_t i = 0; i < primes.size() && i < 4; ++i)
        coef.emplace_back("T" + std::to_string(primes[i]), 1 + (7 * i + 3) % (R.ell - 1));
    Comp out;
    for (int sp = 0; sp < 2; ++sp) {
        const FormsSpace& FS = sp == 0 ? w.M : w.S;
        const size_t s = FS.s;
        if (s == 0) continue;
        Mat t(s, Vec(s, 0));
        for (const auto& cp : coef) {
            const Mat A = MAddScaledIdent(FS.mats.at(cp.first), w.eis.at(cp.first), R);
            for (size_t i = 0; i < s; ++i)
                for (size_t j = 0; j < s; ++j) t[i][j] = R.add(t[i][j], R.mul(A[i][j], cp.second % R.M));
        }
        const auto split = FittingSplit(t, R);
        for (int f = 0; f < 2; ++f) {
            const Mat& Bm = f == 0 ? split.first : split.second;
            const size_t r = Bm.empty() || Bm[0].empty() ? 0 : Bm[0].size();
            out.r[sp][f] = static_cast<uint32_t>(r);
            if (!r) continue;
            const auto bd = BasisData(Bm, R);
            for (const auto& name : FS.names) out.ops[sp][f][name] = Restrict(FS.mats.at(name), Bm, bd.first, bd.second, R);
        }
        // certificate: on V_E every T_p - eis_p and U - 1 is nilpotent mod ell (V_E is
        // the m_Eis-part; the primes run to the Sturm bound); V_c = im t^N carries none of it.
        for (const auto& kv : out.ops[sp][0])
            ENGINE_CHECK_MSG(NilpotentModEll(MAddScaledIdent(kv.second, w.eis.at(kv.first), R), R.ell),
                             ("HeckeHidaLambda: the component split is not the m_Eis split (" + kv.first + ")").c_str());
    }
    return out;
}

}  // namespace

EisBlock EisensteinBlock(const FormsWeight& w, const std::vector<uint64_t>& primes) {
    ENGINE_CHECK_MSG(primes.size() >= 1, "EisensteinBlock: no T_p");
    const ZMod R(w.ell, w.W);
    const Comp c = Components(w, R, primes);
    EisBlock b;
    b.rank_M = c.r[0][0];
    b.rank_S = c.r[1][0];
    b.M = c.ops[0][0];
    b.S = c.ops[1][0];
    return b;
}

ClassResult RunClass(const ClassInput& in, uint32_t a, uint32_t K, const std::vector<uint32_t>& ms,
                     const std::vector<uint32_t>& extra_centres, const Report& report) {
    const uint64_t l = in.ell;
    const uint32_t W = in.W;
    const ZMod Rw(l, W);
    std::vector<const FormsWeight*> ws;
    for (const auto& w : in.weights)
        if ((w.k + (l - 1) * 1000000 - a) % (l - 1) == 0) ws.push_back(&w);
    std::sort(ws.begin(), ws.end(), [](const FormsWeight* x, const FormsWeight* y) { return x->k < y->k; });
    ENGINE_CHECK_MSG(!ws.empty(), "RunClass: no weight of the class");
    for (const auto* w : ws) ENGINE_CHECK_MSG(w->W == W && w->ell == l, "RunClass: forms data at another W");
    std::vector<uint32_t> ks;
    for (const auto* w : ws) ks.push_back(w->k);
    for (size_t j = 1; j < ks.size(); ++j)
        ENGINE_CHECK_MSG(ks[j] == ks[0] + j * (l - 1), "RunClass: the weights are not consecutive in the class");
    std::vector<Comp> comps;
    for (const auto* w : ws) comps.push_back(Components(*w, Rw, in.primes));
    for (const auto& c : comps)
        ENGINE_CHECK_MSG(c.r[0][0] == comps[0].r[0][0] && c.r[1][0] == comps[0].r[1][0] &&
                             c.r[0][1] == comps[0].r[0][1] && c.r[1][1] == comps[0].r[1][1],
                         "component ranks vary across weights");
    const uint32_t rEM = comps[0].r[0][0], rES = comps[0].r[1][0], rcM = comps[0].r[0][1], rcS = comps[0].r[1][1];
    ClassResult res;
    res.ell = l;
    res.a = a;
    res.W = W;
    res.weights = ks;
    res.d = rcS + rES;
    res.rank_E[0] = rEM;
    res.rank_E[1] = rES;
    res.rank_c[0] = rcM;
    res.rank_c[1] = rcS;
    {
        std::ostringstream os;
        os << "CLASS ell=" << l << " a=" << a << " weights=" << ks.size() << " (" << ks.front() << ".." << ks.back()
           << ") W=" << W << " rank(V_E^M,V_E^S)=(" << rEM << "," << rES << ") rank(V_c^M,V_c^S)=(" << rcM << ","
           << rcS << ")";
        report(os.str());
    }
    const std::vector<std::string>& ops = ws[0]->M.names;   // T_p in primes order, then U
    auto eis = [&](size_t j, const std::string& op) { return ws[j]->eis.at(op); };
    auto EM = [&](size_t j, const std::string& op) -> const Mat& { return comps[j].ops[0][0].at(op); };
    auto ES = [&](size_t j, const std::string& op) -> const Mat& { return comps[j].ops[1][0].at(op); };
    auto CM = [&](size_t j, const std::string& op) -> const Mat& { return comps[j].ops[0][1].at(op); };
    // ---- the Eisenstein generator Y = T_p - eis_p
    std::string yop;
    for (size_t i = 0; i + 1 < ops.size(); ++i)
        if (GeneratesModEll(MAddScaledIdent(EM(0, ops[i]), eis(0, ops[i]), Rw), l)) {
            yop = ops[i];
            break;
        }
    res.Y_E = yop;
    report("EIS_COMPONENT ell=" + std::to_string(l) + " a=" + std::to_string(a) + " rank_M=" + std::to_string(rEM) +
           " rank_S=" + std::to_string(rES) + " generator Y=" + (yop.empty() ? "None" : yop) + "-eis");
    // ---- the complement: an etale generator
    std::string cop;
    if (rcM) {
        for (const auto& op : ops) {
            bool all = true;
            for (size_t j = 0; j < ws.size() && all; ++j) all = SquarefreeModEll(Charpoly(CM(j, op), Rw), l);
            if (all) {
                cop = op;
                break;
            }
        }
        if (!cop.empty()) res.residue_degrees = ResidueDegrees(Charpoly(CM(0, cop), ZMod(l, 1)), l);
        std::ostringstream os;
        os << "COMPLEMENT ell=" << l << " a=" << a << " rank=" << rcM << " etale_generator=" << (cop.empty() ? "None" : cop)
           << " residue_degrees=[";
        for (size_t i = 0; i < res.residue_degrees.size(); ++i) os << (i ? "," : "") << res.residue_degrees[i];
        os << "] (U11: etale)";
        report(os.str());
    }
    res.Y_c = cop;
    // ---- the Bernoulli side (lem:u-twistvalues): G(x_k) = -(1 - ell^{k-1}) B_k / k
    const auto BK = BernoulliOverK(l, W, ks);
    Vec Gvals;
    for (uint32_t k : ks) {
        const u128 one_minus = Rw.sub(1 % Rw.M, Rw.pow(l % Rw.M, k - 1));
        Gvals.push_back(Rw.neg(Rw.mul(one_minus, BK.at(k))));
    }
    res.G_at_weights = Gvals;
    if (!yop.empty())
        for (size_t j = 0; j < ws.size(); ++j) {
            const Vec cp = Charpoly(MAddScaledIdent(EM(j, yop), eis(j, yop), Rw), Rw);
            res.F1_at_weights.push_back(cp.size() > 1 ? cp[1] : 0);
        }
    std::vector<BigInt> xs;
    for (uint32_t k : ks) xs.push_back(XOf(l, k, W + 40));
    // KL: G~(0) mod ell^5 from 5 weights
    if (ks.size() >= 5) {
        const ZMod R5(l, 5);
        const Vec p = NewtonMod(std::vector<BigInt>(xs.begin(), xs.begin() + 5), Vec(Gvals.begin(), Gvals.begin() + 5), R5);
        res.has_G0 = true;
        res.G0_mod_l5 = p[0];
        report("KL_ORACLE ell=" + std::to_string(l) + " a=" + std::to_string(a) + " G~(0) mod l^5 = " + U128Text(p[0]));
    }
    // alpha by Newton iteration on the min(#weights, W)-weight interpolant (thm:u-interp(3))
    const uint32_t Nw = static_cast<uint32_t>(std::min<size_t>(ks.size(), W));
    const uint32_t prec = std::min(Nw, W);
    const ZMod Rp(l, prec);
    const Vec rG = NewtonMod(std::vector<BigInt>(xs.begin(), xs.begin() + Nw), Vec(Gvals.begin(), Gvals.begin() + Nw), Rp);
    Vec drG;
    for (size_t i = 1; i < rG.size(); ++i) drG.push_back(Rp.mul(static_cast<u128>(i) % Rp.M, rG[i]));
    u128 alpha = 0;
    for (uint32_t it = 0; it < 4 * prec; ++it) {
        const u128 val = PEval(rG, alpha, Rp), der = PEval(drG, alpha, Rp);
        alpha = Rp.sub(alpha, Rp.mul(val, Rp.inv(der)));
    }
    res.alpha = alpha;
    res.alpha_prec = prec;
    res.lambda_one = Rw.val(Rw.sub(Gvals[0], Gvals.size() > 1 ? Gvals[1] : Gvals[0]), 1000000000u) == 1;
    report("LAMBDA ell=" + std::to_string(l) + " a=" + std::to_string(a) + " lambda=1,mu=0: " +
           (res.lambda_one ? "True" : "False") + "; alpha mod l^" + std::to_string(prec) + " = " + U128Text(alpha) +
           ", v(alpha)=" + std::to_string(Rp.val(alpha, prec)));
    for (uint32_t depth : {2u, 3u}) {
        if (depth > prec) continue;
        const ZMod Rq(l, depth);
        const u128 al = alpha % Rq.M;
        u128 y = Rq.pow((l + 1) % Rq.M, a);
        const u128 step = Rq.pow((l + 1) % Rq.M, l - 1);
        bool found = false;
        uint64_t kk = 0;
        for (u128 t = 0; t <= Rq.M; ++t) {
            if (Rq.sub(Rq.sub(y, 1 % Rq.M), al) == 0) {
                found = true;
                kk = a + static_cast<uint64_t>(t) * (l - 1);
                break;
            }
            y = Rq.mul(y, step);
        }
        if (depth == 2) {
            res.has_rung2 = found;
            res.rung2 = kk;
        } else {
            res.has_rung3 = found;
            res.rung3 = kk;
        }
        report("RUNG ell=" + std::to_string(l) + " a=" + std::to_string(a) + " first k in the class with v(x_k - alpha) >= " +
               std::to_string(depth) + ": k = " + (found ? std::to_string(kk) : std::string("None")));
    }
    // ---- per (m, centre)
    for (uint32_t m : ms) {
        const uint32_t N = K + m - 1;
        if (N > ks.size()) {
            report("SKIP ell=" + std::to_string(l) + " a=" + std::to_string(a) + " m=" + std::to_string(m) + ": needs N=" +
                   std::to_string(N) + " weights");
            continue;
        }
        const uint32_t Wmono = K + m - 1, Wgen = K + N - 1;
        ENGINE_CHECK_MSG(Wmono <= W, ("monogenic route REFUSED: needs forms data mod ell^{K+m-1} (thm:u-interp(2)); W = " +
                                      std::to_string(W)).c_str());
        const ZMod Rl(l, Wmono), RK(l, K);
        const std::vector<uint32_t> kN(ks.begin(), ks.begin() + N);
        std::vector<uint32_t> centres{ks[0]};
        centres.insert(centres.end(), extra_centres.begin(), extra_centres.end());
        for (uint32_t ck : centres) {
            Truncation tr;
            tr.K = K;
            tr.m = m;
            tr.centre_k = ck;
            tr.N = N;
            tr.W_mono = Wmono;
            tr.W_gen = Wgen;
            const std::string tag = "ell=" + std::to_string(l) + " a=" + std::to_string(a) + " K=" + std::to_string(K) +
                                    " m=" + std::to_string(m) + " centre=x_" + std::to_string(ck);
            const BigInt cbig = XOf(l, ck, K + 40);
            auto interp_rows = [&](const std::vector<Vec>& valrows) {   // valrows[j][i]
                std::vector<Vec> out;
                for (size_t i = 0; i < valrows[0].size(); ++i) {
                    Vec v;
                    for (size_t j = 0; j < N; ++j) v.push_back(valrows[j][i]);
                    out.push_back(InterpolateTruncation(l, kN, v, Wmono, K, m, ck));
                }
                return out;
            };
            if (!yop.empty()) {
                Component& E = tr.E;
                E.present = true;
                E.Y = yop + " - eis_" + yop.substr(1) + "(X)";
                E.op_names = ops;
                std::vector<Vec> Fvals;
                std::map<std::string, std::vector<Vec>> Pvals;
                for (size_t j = 0; j < N; ++j) {
                    const Mat t = MAddScaledIdent(EM(j, yop), eis(j, yop), Rw);
                    Mat tl = t;
                    for (auto& row : tl)
                        for (auto& x : row) x %= Rl.M;
                    Fvals.push_back(Charpoly(tl, Rl));
                    for (const auto& op : ops) {
                        Mat target = EM(j, op);
                        for (auto& row : target)
                            for (auto& x : row) x %= Rl.M;
                        Pvals[op].push_back(PolyCoords(target, tl, Rl));
                    }
                }
                E.F = interp_rows(Fvals);
                for (const auto& op : ops) E.ops[op] = interp_rows(Pvals[op]);
                {
                    std::vector<Vec> gv;
                    for (size_t j = 0; j < N; ++j) gv.push_back(Vec{Gvals[j]});
                    E.G = interp_rows(gv)[0];
                }
                E.F0_zero = std::all_of(E.F[0].begin(), E.F[0].end(), [](u128 x) { return x == 0; });
                const Span IF1 = IdealSpan({E.F[1]}, RK, m), IG = IdealSpan({E.G}, RK, m);
                const u128 cK = U128FromBig(cbig.floor_mod(BigFromU128(RK.M)));
                const u128 diffK = RK.sub(cK, alpha % RK.M);
                const Span Ialpha = m >= 2 ? IdealSpan({Vec{diffK, 1}}, RK, m) : IdealSpan({Vec{diffK}}, RK, m);
                E.cong_log = K * m - IG.log_order();
                const u128 cP = U128FromBig(cbig.floor_mod(BigFromU128(Rp.M)));
                const uint32_t L_c = Rp.val(Rp.sub(cP, alpha), prec);
                E.cong_pred = std::min(K, m * L_c);
                E.F1_eq_G = IdealsEqual(IF1, IG);
                E.G_eq_alpha = IdealsEqual(IG, Ialpha);
                report("MONO_E " + tag + ": F(Y) = Y^2 + " + PolyText(E.F[1]) + "*Y + " + PolyText(E.F[0]) +
                       "  F0=0: " + (E.F0_zero ? "True" : "False") + "  (F1) == (G~): " + (E.F1_eq_G ? "True" : "False") +
                       "  (G~) == (Z-(alpha-c)): " + (E.G_eq_alpha ? "True" : "False") + "  |Lambda/(G~)| = l^" +
                       std::to_string(E.cong_log) + " (pred " + std::to_string(E.cong_pred) + ", L=" + std::to_string(L_c) + ")");
                // ---- the general route on the Eisenstein component, precision W_gen
                ENGINE_CHECK_MSG(Wgen <= W, ("general route REFUSED: needs forms data mod ell^{K+N-1} "
                                             "(prop:u-lambdaprod(3)); W = " + std::to_string(W)).c_str());
                const ZMod Rg(l, Wgen);
                GeneralRoute& gr = E.general;
                E.has_general = true;
                const size_t bM = rEM * rEM, bS = rES * rES, blk = bM + bS + 1, nvec = N * blk;
                struct El {
                    Mat Mm, Ms;
                    u128 e;
                };
                using Tuple = std::vector<El>;
                auto vec = [&](const Tuple& el) {
                    Vec v;
                    for (size_t j = 0; j < N; ++j)
                        for (const auto& r : el[j].Mm)
                            for (u128 x : r) v.push_back(x % Rg.M);
                    for (size_t j = 0; j < N; ++j)
                        for (const auto& r : el[j].Ms)
                            for (u128 x : r) v.push_back(x % Rg.M);
                    for (size_t j = 0; j < N; ++j) v.push_back(el[j].e % Rg.M);
                    return v;
                };
                auto mul = [&](const Tuple& e1, const Tuple& e2) {
                    Tuple out(N);
                    for (size_t j = 0; j < N; ++j)
                        out[j] = El{MatMul(e1[j].Mm, e2[j].Mm, Rg), MatMul(e1[j].Ms, e2[j].Ms, Rg),
                                    Rg.mul(e1[j].e % Rg.M, e2[j].e % Rg.M)};
                    return out;
                };
                auto scal_mat = [&](size_t n, u128 x) {
                    Mat S = Ident(n);
                    for (size_t i = 0; i < n; ++i) S[i][i] = x % Rg.M;
                    return S;
                };
                std::vector<u128> xg;
                for (size_t j = 0; j < N; ++j) xg.push_back(U128FromBig(xs[j].floor_mod(BigFromU128(Rg.M))));
                const u128 cg = U128FromBig(cbig.floor_mod(BigFromU128(Rg.M)));
                Tuple one(N), X(N);
                for (size_t j = 0; j < N; ++j) {
                    one[j] = El{Ident(rEM), Ident(rES), 1 % Rg.M};
                    X[j] = El{scal_mat(rEM, xg[j]), scal_mat(rES, xg[j]), xg[j]};
                }
                std::vector<Tuple> gens{X};
                for (const auto& op : ops) {
                    Tuple g(N);
                    for (size_t j = 0; j < N; ++j) g[j] = El{EM(j, op), ES(j, op), eis(j, op)};
                    gens.push_back(g);
                }
                Span A(nvec, Rg);
                A.insert(vec(one));
                std::vector<Tuple> queue{one};
                while (!queue.empty()) {
                    const Tuple b = queue.back();
                    queue.pop_back();
                    for (const auto& g : gens) {
                        Tuple e = mul(b, g);
                        if (A.insert(vec(e))) queue.push_back(std::move(e));
                    }
                }
                auto scal_tuple = [&](const Vec& v, const std::vector<u128>& f) {
                    Vec out = v;
                    for (size_t j = 0; j < N; ++j) {
                        for (size_t t = 0; t < bM; ++t) out[j * bM + t] = Rg.mul(out[j * bM + t], f[j]);
                        for (size_t t = 0; t < bS; ++t) out[N * bM + j * bS + t] = Rg.mul(out[N * bM + j * bS + t], f[j]);
                        out[N * (bM + bS) + j] = Rg.mul(out[N * (bM + bS) + j], f[j]);
                    }
                    return out;
                };
                auto scaled = [&](const Vec& v, u128 f) {
                    Vec out(v.size());
                    for (size_t i = 0; i < v.size(); ++i) out[i] = Rg.mul(v[i], f);
                    return out;
                };
                std::vector<u128> fm, Zt;
                for (size_t j = 0; j < N; ++j) {
                    Zt.push_back(Rg.sub(xg[j], cg));
                    fm.push_back(Rg.pow(Zt.back(), m));
                }
                const u128 lK = Rg.pw[K];
                Span Ap(nvec, Rg);
                for (const Vec& r : A.rows()) {
                    Ap.insert(scaled(r, lK));
                    Ap.insert(scal_tuple(r, fm));
                }
                gr.A_log = A.log_order();
                gr.quot_log = A.log_order() - Ap.log_order();
                Span Q1(nvec, Rg);
                for (const Vec& r : A.rows()) {
                    Q1.insert(scaled(r, l % Rg.M));
                    Q1.insert(scal_tuple(r, Zt));
                }
                gr.mingens = A.log_order() - Q1.log_order();
                gr.free = gr.quot_log == K * m * rEM && gr.mingens == rEM;
                Tuple Yel(N);
                for (size_t j = 0; j < N; ++j)
                    Yel[j] = El{MAddScaledIdent(EM(j, yop), eis(j, yop) % Rg.M, Rg),
                                MAddScaledIdent(ES(j, yop), eis(j, yop) % Rg.M, Rg), 0};
                std::vector<Tuple> Ypow{one};
                for (uint32_t i = 0; i < rEM; ++i) Ypow.push_back(mul(Ypow.back(), Yel));
                Vec FY(nvec, 0);
                for (size_t i = 0; i < E.F.size(); ++i) {
                    std::vector<u128> fz;
                    for (size_t j = 0; j < N; ++j) fz.push_back(PEval(E.F[i], Zt[j], Rg));
                    const Vec add = scal_tuple(vec(Ypow[i]), fz);
                    for (size_t t = 0; t < nvec; ++t) FY[t] = Rg.add(FY[t], add[t]);
                }
                gr.FY_zero = Ap.contains(FY);
                Span Gs(nvec, Rg);
                for (const Vec& r : Ap.rows()) Gs.insert(r);
                for (uint32_t i = 0; i < rEM; ++i)
                    for (uint32_t e = 0; e < m; ++e) {
                        std::vector<u128> ze;
                        for (size_t j = 0; j < N; ++j) ze.push_back(Rg.pow(Zt[j], e));
                        Gs.insert(scal_tuple(vec(Ypow[i]), ze));
                    }
                gr.Y_generates = Gs.log_order() == A.log_order();
                // the cokernel of T -> h x Lambda: S-blocks + E-values
                const size_t off = N * bM, nSE = N * (bS + 1);
                auto scal_SE = [&](const Vec& v, const std::vector<u128>& f) {
                    Vec out = v;
                    for (size_t j = 0; j < N; ++j) {
                        for (size_t t = 0; t < bS; ++t) out[j * bS + t] = Rg.mul(out[j * bS + t], f[j]);
                        out[N * bS + j] = Rg.mul(out[N * bS + j], f[j]);
                    }
                    return out;
                };
                using HT = std::vector<Mat>;
                auto vS = [&](const HT& el) {
                    Vec v;
                    for (size_t j = 0; j < N; ++j)
                        for (const auto& r : el[j])
                            for (u128 x : r) v.push_back(x % Rg.M);
                    v.resize(v.size() + N, 0);
                    return v;
                };
                Span Hs(nSE, Rg);
                std::vector<HT> hq{HT(N, Ident(rES))};
                Hs.insert(vS(hq[0]));
                std::vector<HT> hgens;
                {
                    HT hx(N);
                    for (size_t j = 0; j < N; ++j) hx[j] = scal_mat(rES, xg[j]);
                    hgens.push_back(hx);
                    for (const auto& op : ops) {
                        HT g(N);
                        for (size_t j = 0; j < N; ++j) g[j] = ES(j, op);
                        hgens.push_back(g);
                    }
                }
                while (!hq.empty()) {
                    const HT b = hq.back();
                    hq.pop_back();
                    for (const auto& g : hgens) {
                        HT e(N);
                        for (size_t j = 0; j < N; ++j) e[j] = MatMul(b[j], g[j], Rg);
                        if (Hs.insert(vS(e))) hq.push_back(std::move(e));
                    }
                }
                Span Ys(nSE, Rg);
                for (const Vec& r : Hs.rows()) Ys.insert(r);
                for (size_t e = 0; e < N; ++e) {
                    Vec v(N * bS, 0);
                    for (size_t j = 0; j < N; ++j) v.push_back(Rg.pow(xg[j], e));
                    Ys.insert(v);
                }
                Span Yp(nSE, Rg);
                for (const Vec& r : Ys.rows()) {
                    Yp.insert(scaled(r, lK));
                    Yp.insert(scal_SE(r, fm));
                }
                Span Zs(nSE, Rg);
                for (const Vec& r : Yp.rows()) Zs.insert(r);
                for (const Vec& r : A.rows()) Zs.insert(Vec(r.begin() + static_cast<long>(off), r.end()));
                gr.coker_log = Ys.log_order() - Zs.log_order();
                std::vector<Vec> Jgens;
                for (const auto& en : Zs.entries())
                    if (en.col >= N * bS) {
                        const Vec vals(en.row.begin() + static_cast<long>(N * bS), en.row.end());
                        Jgens.push_back(InterpolateTruncation(l, kN, vals, Wgen, K, m, ck));
                    }
                const Span Jsp = IdealSpan(Jgens, RK, m);
                gr.J = Jsp.rows();
                gr.J_log_index = K * m - Jsp.log_order();
                gr.J_eq_G = IdealsEqual(Jsp, IG);
                report("GENERAL " + tag + ": W'=" + std::to_string(Wgen) + " |A|=l^" + std::to_string(gr.A_log) +
                       " |A_(K,m)|=l^" + std::to_string(gr.quot_log) + " mingens=" + std::to_string(gr.mingens) +
                       " free=" + (gr.free ? "True" : "False") + " F(Y)=0: " + (gr.FY_zero ? "True" : "False") +
                       " Y generates: " + (gr.Y_generates ? "True" : "False") + " | coker = l^" +
                       std::to_string(gr.coker_log) + " |Lambda/J| = l^" + std::to_string(gr.J_log_index) +
                       " J == (G~): " + (gr.J_eq_G ? "True" : "False"));
            }
            if (!cop.empty()) {
                Component& Cc = tr.c;
                Cc.present = true;
                Cc.Y = cop;
                Cc.op_names = ops;
                std::vector<Vec> Fv;
                std::map<std::string, std::vector<Vec>> Pv;
                for (size_t j = 0; j < N; ++j) {
                    Mat tl = CM(j, cop);
                    for (auto& row : tl)
                        for (auto& x : row) x %= Rl.M;
                    Fv.push_back(Charpoly(tl, Rl));
                    for (const auto& op : ops) {
                        Mat target = CM(j, op);
                        for (auto& row : target)
                            for (auto& x : row) x %= Rl.M;
                        Pv[op].push_back(PolyCoords(target, tl, Rl));
                    }
                }
                Cc.F = interp_rows(Fv);
                for (const auto& op : ops) Cc.ops[op] = interp_rows(Pv[op]);
                // thm:u-interp(3) at the further weights: predicted charpolys mod ell^min(K', N)
                std::string chk;
                for (size_t j = N; j < ks.size(); ++j) {
                    const uint32_t pr = std::min(Wmono, N);
                    const ZMod Rq(l, pr);
                    const std::vector<BigInt> xN(xs.begin(), xs.begin() + N);
                    const u128 xj = U128FromBig(xs[j].floor_mod(BigFromU128(Rq.M)));
                    Vec pred;
                    for (size_t i = 0; i <= rcM; ++i) {
                        Vec col;
                        for (size_t jj = 0; jj < N; ++jj) col.push_back(Fv[jj][i]);
                        pred.push_back(PEval(NewtonMod(xN, col, Rq), xj, Rq));
                    }
                    Mat tq = CM(j, cop);
                    for (auto& row : tq)
                        for (auto& x : row) x %= Rq.M;
                    const bool held = pred == Charpoly(tq, Rq);
                    Cc.further.push_back(Component::Further{ks[j], held, pr});
                    chk += "(" + std::to_string(ks[j]) + "," + (held ? "True" : "False") + "," + std::to_string(pr) + ")";
                }
                report("MONO_C " + tag + ": Y=" + cop + ", deg F=" + std::to_string(rcM) + "; U11 at further weights: " + chk);
            }
            res.truncations.push_back(std::move(tr));
        }
    }
    return res;
}

// ===========================================================================
// the record hecke_lambda/v1
// ===========================================================================
std::vector<std::string> RecordLines(const ClassResult& r, const std::string& route, const u128* kl_ref) {
    std::vector<std::string> out;
    auto polys = [](const std::vector<Vec>& ps) {
        std::string s = "[";
        for (size_t i = 0; i < ps.size(); ++i) s += (i ? "," : "") + PolyText(ps[i]);
        return s + "]";
    };
    for (const auto& tr : r.truncations) {
        for (int f = 0; f < 2; ++f) {
            const Component& C = f == 0 ? tr.E : tr.c;
            if (!C.present) continue;
            std::ostringstream os;
            os << "HECKE_LAMBDA ell=" << r.ell << " a=" << r.a << " K=" << tr.K << " m=" << tr.m
               << " centre_k=" << tr.centre_k << " N=" << tr.N << " W=" << tr.W_mono << " W_gen=" << tr.W_gen
               << " route=" << route << " component=" << (f == 0 ? "E" : "c")
               << " rank=" << (f == 0 ? r.rank_E[0] : r.rank_c[0]) << " gen="
               << (f == 0 ? r.Y_E + "-eis" : r.Y_c) << " residue_degrees=[";
            if (f == 0) {
                os << "1";
            } else {
                for (size_t i = 0; i < r.residue_degrees.size(); ++i) os << (i ? "," : "") << r.residue_degrees[i];
            }
            os << "] F=" << polys(C.F) << " ops={";
            for (size_t i = 0; i < C.op_names.size(); ++i)
                os << (i ? "," : "") << C.op_names[i] << ":" << polys(C.ops.at(C.op_names[i]));
            os << "}";
            std::vector<std::string> checks;
            if (f == 0) {
                os << " cong_log=" << C.cong_log << " J=" << polys(C.general.J);
                if (C.general.free) checks.push_back("free");
                if (C.general.FY_zero) checks.push_back("FY0");
                if (C.general.Y_generates) checks.push_back("Ygen");
                if (C.general.J_eq_G && C.cong_log == C.cong_pred && C.general.coker_log == C.cong_log)
                    checks.push_back("U12");
                if (C.F0_zero && C.F1_eq_G && C.G_eq_alpha) checks.push_back("U13");
                if (kl_ref && r.has_G0 && r.G0_mod_l5 == *kl_ref) checks.push_back("KL");
            } else {
                os << " cong_log=- J=-";
                bool all = !C.further.empty();
                for (const auto& fu : C.further) all = all && fu.held;
                if (all) checks.push_back("U11");
            }
            os << " alpha_mod=" << r.ell << "^" << r.alpha_prec << ":" << U128Text(r.alpha) << " checks=";
            for (size_t i = 0; i < checks.size(); ++i) os << (i ? "," : "") << checks[i];
            const std::string line = os.str();
            const uint64_t h = engine::FnvBytes(engine::kFnvOffset64, line);
            std::ostringstream hx;
            hx << std::hex << h;
            out.push_back(line + " digest=0x" + hx.str());
        }
    }
    return out;
}

}  // namespace lambda
}  // namespace hecke_ext
