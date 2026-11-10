// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-LIN) Birch / Manin-Stevens modular symbols -- see BirchModularSymbol.h.
#include "BirchModularSymbol.h"

#include "Bernoulli.h"   // TeichmullerPowMod, PrimitiveRootModPrime, v(B_k/k)
#include "LValueRoute.h"
#include "Gamma0NShapiro.h"   // (slot-22 P6) the level-N model

#include "../../EquivariantCohomologyOnK.h"
#include "../../EquivariantMappingCone.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../Hecke/CoefficientRing.h"
#include "../../Hecke/HeckeBoundaryDatum.h"
#include "../../Hecke/HeckeBoundaryOperator.h"
#include "../../Hecke/HeckeConeAction.h"
#include "../../Hecke/HeckeCosetTables.h"
#include "../../Hecke/HeckeInvolution.h"
#include "../../Hecke/HeckeOperatorFamily.h"   // tau_construction_mutex
#include "../../Hecke/HeckeOperatorOnK.h"
#include "../../Hecke/HeckePairHomotopy.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../Hecke/HeckeTauTable.h"
#include "../../Hecke/SL2ZBoundaryPair.h"
#include "../../Hecke/SL2ZTessellationTables.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>

namespace hecke_ext {
namespace lvalue {
namespace birch {

// ---- rho_n -----------------------------------------------------------------------------------------
SymRho::SymRho(uint32_t n, const Ring& R) : n_(n), R_(R), binom_(static_cast<size_t>(n + 1) * (n + 1), 0) {
    for (uint32_t t = 0; t <= n; ++t) {
        binom_[static_cast<size_t>(t) * (n + 1)] = 1 % R.q;
        for (uint32_t s = 1; s <= t; ++s)
            binom_[static_cast<size_t>(t) * (n + 1) + s] =
                R.add(binom_[static_cast<size_t>(t - 1) * (n + 1) + s - 1], s <= t - 1 ? binom_[static_cast<size_t>(t - 1) * (n + 1) + s] : 0);
    }
}
Vec SymRho::S(const Vec& v) const {
    Vec out(n_ + 1, 0);
    for (uint32_t t = 0; t <= n_; ++t) out[n_ - t] = (t % 2 == 0) ? v[t] : R_.neg(v[t]);
    return out;
}
Vec SymRho::diag(uint64_t a, uint64_t d, const Vec& v) const {
    Vec pa(n_ + 1, 1 % R_.q), pd(n_ + 1, 1 % R_.q), out(n_ + 1, 0);
    for (uint32_t i = 1; i <= n_; ++i) {
        pa[i] = R_.mul(pa[i - 1], a);
        pd[i] = R_.mul(pd[i - 1], d);
    }
    for (uint32_t t = 0; t <= n_; ++t) out[t] = R_.mul(R_.mul(pa[n_ - t], pd[t]), v[t]);
    return out;
}
Vec SymRho::upper(uint64_t b, const Vec& v) const {
    // x -> x, y -> b x + y:  out_s = sum_{t >= s} C(t, s) b^{t-s} v_t
    if (b % R_.q == 0) return v;
    Vec pb(n_ + 1, 1 % R_.q), out(n_ + 1, 0);
    for (uint32_t i = 1; i <= n_; ++i) pb[i] = R_.mul(pb[i - 1], b);
    for (uint32_t t = 0; t <= n_; ++t) {
        if (v[t] == 0) continue;
        for (uint32_t s = 0; s <= t; ++s) out[s] = R_.add(out[s], R_.mul(R_.mul(binom(t, s), pb[t - s]), v[t]));
    }
    return out;
}
Vec SymRho::lower(uint64_t g, const Vec& v) const {
    // x -> x + g y, y -> y:  out_s = sum_{t <= s} C(n - t, s - t) g^{s-t} v_t
    if (g % R_.q == 0) return v;
    Vec pg(n_ + 1, 1 % R_.q), out(n_ + 1, 0);
    for (uint32_t i = 1; i <= n_; ++i) pg[i] = R_.mul(pg[i - 1], g);
    for (uint32_t t = 0; t <= n_; ++t) {
        if (v[t] == 0) continue;
        for (uint32_t s = t; s <= n_; ++s) out[s] = R_.add(out[s], R_.mul(R_.mul(binom(n_ - t, s - t), pg[s - t]), v[t]));
    }
    return out;
}
Vec SymRho::apply(const Mat2& A, const Vec& v) const {
    ENGINE_CHECK_MSG(v.size() == n_ + 1, "birch::SymRho::apply: vector length != n + 1");
    const uint64_t a = R_.from_int(A.a), b = R_.from_int(A.b), c = R_.from_int(A.c);
    const __int128 det = static_cast<__int128>(A.a) * A.d - static_cast<__int128>(A.b) * A.c;
    ENGINE_CHECK_MSG(det > -(static_cast<__int128>(1) << 62) && det < (static_cast<__int128>(1) << 62), "birch::SymRho::apply: det overflow");
    const uint64_t dt = R_.from_int(static_cast<int64_t>(det));
    if (R_.unit(a)) {
        const uint64_t ia = R_.inv(a);
        Vec w = upper(R_.mul(b, ia), v);
        w = diag(a, R_.mul(dt, ia), w);
        return lower(R_.mul(c, ia), w);
    }
    const bool odd = (n_ % 2) == 1;
    if (R_.unit(c)) {
        // A = S^{-1} (S A), S A = ((-c, -d), (a, b)); rho(S^{-1}) = (-1)^n rho(S)
        Vec w = S(apply(Mat2{-A.c, -A.d, A.a, A.b}, v));
        if (odd)
            for (auto& x : w) x = R_.neg(x);
        return w;
    }
    if (R_.unit(b) || R_.unit(R_.from_int(A.d))) {
        // A = (A S) S^{-1}, A S = ((b, -a), (d, -c))
        Vec w = S(v);
        if (odd)
            for (auto& x : w) x = R_.neg(x);
        return apply(Mat2{A.b, -A.a, A.d, -A.c}, w);
    }
    throw std::invalid_argument("birch::SymRho::apply: no entry of A is an ell-unit");
}

std::vector<uint64_t> RhoDirectMatrix(const Mat2& A, uint32_t n, const Ring& R) {
    // column j: coefficients of (a + c y)^{n-j} (b + d y)^j in y (x = 1): row s <-> u_s
    const uint64_t a = R.from_int(A.a), b = R.from_int(A.b), c = R.from_int(A.c), d = R.from_int(A.d);
    std::vector<uint64_t> M(static_cast<size_t>(n + 1) * (n + 1), 0);
    for (uint32_t j = 0; j <= n; ++j) {
        Vec p(1, 1 % R.q), qv(1, 1 % R.q);
        for (uint32_t i = 0; i < n - j; ++i) {   // multiply by (a + c y)
            Vec np(p.size() + 1, 0);
            for (size_t s = 0; s < p.size(); ++s) {
                np[s] = R.add(np[s], R.mul(a, p[s]));
                np[s + 1] = R.add(np[s + 1], R.mul(c, p[s]));
            }
            p.swap(np);
        }
        for (uint32_t i = 0; i < j; ++i) {       // multiply by (b + d y)
            Vec nq(qv.size() + 1, 0);
            for (size_t s = 0; s < qv.size(); ++s) {
                nq[s] = R.add(nq[s], R.mul(b, qv[s]));
                nq[s + 1] = R.add(nq[s + 1], R.mul(d, qv[s]));
            }
            qv.swap(nq);
        }
        for (size_t s = 0; s < p.size(); ++s)
            for (size_t r = 0; r < qv.size(); ++r)
                M[(s + r) * (n + 1) + j] = R.add(M[(s + r) * (n + 1) + j], R.mul(p[s], qv[r]));
    }
    return M;
}

// ---- Manin ------------------------------------------------------------------------------------------
namespace {
int64_t FloorDiv(int64_t x, int64_t y) {
    int64_t q = x / y;
    if ((x % y != 0) && ((x < 0) != (y < 0))) --q;
    return q;
}
int64_t Gcd(int64_t a, int64_t b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}
Vec VAdd(const Ring& R, const Vec& u, const Vec& v) {
    Vec o(u.size());
    for (size_t i = 0; i < u.size(); ++i) o[i] = R.add(u[i], v[i]);
    return o;
}
Vec VSub(const Ring& R, const Vec& u, const Vec& v) {
    Vec o(u.size());
    for (size_t i = 0; i < u.size(); ++i) o[i] = R.sub(u[i], v[i]);
    return o;
}
bool IsZero(const Vec& v) {
    for (uint64_t x : v)
        if (x) return false;
    return true;
}
} // namespace

std::vector<Mat2> ManinPath(int64_t a, int64_t M) {
    ENGINE_CHECK_MSG(M > 0, "birch::ManinPath: M > 0");
    std::vector<std::pair<int64_t, int64_t>> conv;
    int64_t p_prev = 1, q_prev = 0, p_prev2 = 0, q_prev2 = 1, x = a, y = M;
    while (y != 0) {
        const int64_t qq = FloorDiv(x, y);
        const int64_t pk = qq * p_prev + p_prev2, qk = qq * q_prev + q_prev2;
        conv.emplace_back(pk, qk);
        p_prev2 = p_prev;
        q_prev2 = q_prev;
        p_prev = pk;
        q_prev = qk;
        const int64_t r = x - qq * y;
        x = y;
        y = r;
    }
    ENGINE_CHECK_MSG(static_cast<__int128>(conv.back().first) * M == static_cast<__int128>(conv.back().second) * a,
                     "birch::ManinPath: the last convergent is not a/M");
    std::vector<Mat2> mats;
    int64_t pp = 1, qq0 = 0;
    for (const auto& c : conv) {
        const int64_t det = c.first * qq0 - pp * c.second;
        if (det == 1) {
            mats.push_back(Mat2{c.first, pp, c.second, qq0});
        } else {
            ENGINE_CHECK_MSG(det == -1, "birch::ManinPath: consecutive convergents of determinant != +-1");
            mats.push_back(Mat2{-c.first, pp, -c.second, qq0});
        }
        pp = c.first;
        qq0 = c.second;
    }
    return mats;
}

Vec PhiCusp(const SymRho& rho, const Vec& P, int64_t a, int64_t M) {
    const Ring& R = rho.ring();
    Vec acc(P.size(), 0);
    for (const Mat2& g : ManinPath(a, M)) acc = VAdd(R, acc, rho.apply(g, P));
    for (auto& x : acc) x = R.neg(x);
    return acc;
}

Vec PhiPair(const SymRho& rho, const Vec& P, int64_t rn, int64_t rd, int64_t sn, int64_t sd) {
    auto to_oo = [&](int64_t num, int64_t den) -> Vec {
        if (den == 0) return Vec(P.size(), 0);
        return PhiCusp(rho, P, num, den);
    };
    return VSub(rho.ring(), to_oo(rn, rd), to_oo(sn, sd));
}

bool InW(const SymRho& rho, const Vec& P) {
    const Ring& R = rho.ring();
    if (!IsZero(VAdd(R, P, rho.S(P)))) return false;
    const Mat2 U{0, -1, 1, 1};
    const Vec u1 = rho.apply(U, P), u2 = rho.apply(U, u1);
    return IsZero(VAdd(R, VAdd(R, P, u1), u2));
}

Vec HeckeTp(const SymRho& rho, const Vec& P, uint64_t p) {
    const Ring& R = rho.ring();
    Vec out = rho.diag(1, R.from_int(static_cast<int64_t>(p)), P);
    for (uint64_t b = 0; b < p; ++b)
        out = VAdd(R, out, rho.apply(Mat2{static_cast<int64_t>(p), -static_cast<int64_t>(b), 0, 1},
                                     PhiCusp(rho, P, static_cast<int64_t>(b), static_cast<int64_t>(p))));
    return out;
}

Vec HeckeTpWrongAdjoint(const SymRho& rho, const Vec& P, uint64_t p) {
    const Ring& R = rho.ring();
    Vec out = rho.diag(R.from_int(static_cast<int64_t>(p)), 1, P);
    for (uint64_t b = 0; b < p; ++b)
        out = VAdd(R, out, rho.apply(Mat2{1, static_cast<int64_t>(b), 0, static_cast<int64_t>(p)},
                                     PhiCusp(rho, P, static_cast<int64_t>(b), static_cast<int64_t>(p))));
    return out;
}

Vec SignPart(const Ring& R, const Vec& P, int sign) {
    // (P + sign rho(iota) P) / 2 = the u_t with (-1)^t == sign, the others 0 (n even)
    Vec o(P.size(), 0);
    for (size_t t = 0; t < P.size(); ++t)
        if (((t % 2 == 0) ? 1 : -1) == sign) o[t] = P[t] % R.q;
    return o;
}

// ---- characters ------------------------------------------------------------------------------------
namespace {
int KroneckerSymbol(int64_t D, int64_t nn) {
    // (D / nn), nn > 0
    int res = 1;
    while (nn % 2 == 0) {
        nn /= 2;
        if (D % 2 == 0) return 0;
        const int64_t r8 = ((D % 8) + 8) % 8;
        res *= (r8 == 1 || r8 == 7) ? 1 : -1;
    }
    int64_t a = ((D % nn) + nn) % nn, b = nn;
    int t = 1;
    while (a) {
        while (a % 2 == 0) {
            a /= 2;
            if (b % 8 == 3 || b % 8 == 5) t = -t;
        }
        std::swap(a, b);
        if (a % 4 == 3 && b % 4 == 3) t = -t;
        a %= b;
    }
    return b == 1 ? res * t : 0;
}
bool SquarefreeOdd(int64_t x) {
    if (x < 0) x = -x;
    if (x % 2 == 0) return false;
    for (int64_t p = 3; p * p <= x; p += 2)
        if (x % (p * p) == 0) return false;
    return true;
}
bool Squarefree(int64_t x) {
    if (x < 0) x = -x;
    for (int64_t p = 2; p * p <= x; ++p)
        if (x % (p * p) == 0) return false;
    return true;
}
bool IsFundamental(int64_t D) {
    if (D == 1) return true;
    const int64_t r4 = ((D % 4) + 4) % 4;
    if (r4 == 1) return SquarefreeOdd(D);
    if (r4 == 0) {
        const int64_t m = D / 4;
        const int64_t m4 = ((m % 4) + 4) % 4;
        return (m4 == 2 || m4 == 3) && Squarefree(m);
    }
    return false;
}
} // namespace

DirichletCharacter TrivialCharacter(const Ring& R) {
    DirichletCharacter c;
    c.M = 1;
    c.chi = {1 % R.q};
    c.chibar = c.chi;
    c.parity = 1;
    c.real = true;
    c.name = "chi=1";
    return c;
}

DirichletCharacter KroneckerCharacter(int64_t D, const Ring& R) {
    if (D == 1) return TrivialCharacter(R);
    if (!IsFundamental(D)) throw std::invalid_argument("birch::KroneckerCharacter: " + std::to_string(D) + " is not a fundamental discriminant");
    DirichletCharacter c;
    c.M = static_cast<uint64_t>(D < 0 ? -D : D);
    ENGINE_CHECK_MSG(c.M % R.ell != 0, "birch::KroneckerCharacter: ell divides the conductor");
    for (uint64_t a = 0; a < c.M; ++a) {
        const int k = Gcd(static_cast<int64_t>(a), static_cast<int64_t>(c.M)) == 1 ? KroneckerSymbol(D, static_cast<int64_t>(a == 0 ? c.M : a)) : 0;
        c.chi.push_back(R.from_int(k));
    }
    c.chibar = c.chi;
    c.parity = D < 0 ? -1 : 1;
    c.real = true;
    c.name = "chi_D, D=" + std::to_string(D);
    return c;
}

DirichletCharacter QuadraticCharacterOfConductor(uint64_t M, const Ring& R) {
    if (M == 1) return TrivialCharacter(R);
    const int64_t Mi = static_cast<int64_t>(M);
    const bool p = IsFundamental(Mi), n = IsFundamental(-Mi);
    if (p == n)
        throw std::invalid_argument("birch::QuadraticCharacterOfConductor: conductor " + std::to_string(M) +
                                    (p ? " carries two quadratic characters (+-M); pass the character explicitly"
                                       : " carries no primitive quadratic character; pass the character explicitly"));
    return KroneckerCharacter(p ? Mi : -Mi, R);
}

DirichletCharacter PrimeConductorCharacter(uint64_t M, uint32_t d, uint32_t e, const Ring& R) {
    ENGINE_CHECK_MSG(M >= 3 && PrimitiveRootModPrime(M) > 0, "birch::PrimeConductorCharacter: M must be an odd prime");
    ENGINE_CHECK_MSG(d >= 2 && (M - 1) % d == 0 && (R.ell - 1) % d == 0, "birch::PrimeConductorCharacter: d | M - 1 and d | ell - 1");
    ENGINE_CHECK_MSG(M % R.ell != 0, "birch::PrimeConductorCharacter: ell divides the conductor");
    const uint64_t g = PrimitiveRootModPrime(M), r = PrimitiveRootModPrime(R.ell);
    const uint64_t zeta = TeichmullerPowMod(r, static_cast<uint32_t>((R.ell - 1) / d), R.ell, R.m) % R.q;
    ENGINE_CHECK_MSG(R.pow(zeta, d) == 1 % R.q, "birch::PrimeConductorCharacter: zeta^d != 1");
    DirichletCharacter c;
    c.M = M;
    c.chi.assign(M, 0);
    c.chibar.assign(M, 0);
    uint64_t x = 1;
    for (uint64_t i = 0; i < M - 1; ++i) {
        c.chi[x] = R.pow(zeta, (static_cast<uint64_t>(e) * i) % d);
        c.chibar[x] = R.pow(zeta, (d - (static_cast<uint64_t>(e) * i) % d) % d);
        x = (x * g) % M;
    }
    // chi(-1) = zeta^{e (M-1)/2}
    const uint64_t cm1 = c.chi[M - 1];
    c.parity = (cm1 == 1 % R.q) ? 1 : -1;
    ENGINE_CHECK_MSG(cm1 == 1 % R.q || cm1 == R.q - 1, "birch::PrimeConductorCharacter: chi(-1) != +-1");
    c.real = (d == 2);
    c.name = "chi mod " + std::to_string(M) + " of order " + std::to_string(d) + " (e=" + std::to_string(e) + ")";
    return c;
}

Vec EvAllJ(const SymRho& rho, const Vec& P, const DirichletCharacter& chi) {
    const Ring& R = rho.ring();
    const uint32_t n = rho.n();
    Vec tot(n + 1, 0);
    const int64_t M = static_cast<int64_t>(chi.M);
    for (int64_t a = 0; a < M; ++a) {
        const uint64_t cv = chi.chibar[static_cast<size_t>(a)];
        if (cv == 0) continue;
        const Vec w = rho.apply(Mat2{M, -a, 0, 1}, PhiCusp(rho, P, a, M));
        for (uint32_t t = 0; t <= n; ++t) tot[t] = R.add(tot[t], R.mul(cv, w[t]));
    }
    Vec ev(n + 1, 0);
    for (uint32_t j = 0; j <= n; ++j) ev[j] = tot[n - j];
    return ev;
}

// ---- module linear algebra -------------------------------------------------------------------------
Mat ColumnsToMat(const std::vector<Vec>& cols, uint32_t rows) {
    Mat A(rows, static_cast<uint32_t>(cols.size()));
    for (uint32_t j = 0; j < A.c; ++j) {
        ENGINE_CHECK_MSG(cols[j].size() == rows, "birch::ColumnsToMat: column length");
        for (uint32_t i = 0; i < rows; ++i) A.at(i, j) = cols[j][i];
    }
    return A;
}

Mat MatMul(const Ring& R, const Mat& A, const Mat& B) {
    ENGINE_CHECK_MSG(A.c == B.r, "birch::MatMul: shape");
    Mat C(A.r, B.c);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t k = 0; k < A.c; ++k) {
            const uint64_t a = A.at(i, k);
            if (!a) continue;
            for (uint32_t j = 0; j < B.c; ++j) C.at(i, j) = R.add(C.at(i, j), R.mul(a, B.at(k, j)));
        }
    return C;
}

Vec MatVec(const Ring& R, const Mat& A, const Vec& x) {
    ENGINE_CHECK_MSG(A.c == x.size(), "birch::MatVec: shape");
    Vec y(A.r, 0);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t k = 0; k < A.c; ++k)
            if (A.at(i, k) && x[k]) y[i] = R.add(y[i], R.mul(A.at(i, k), x[k]));
    return y;
}

SmithForm Smith(const Ring& R, Mat A, bool wantP) {
    SmithForm F;
    F.hasP = wantP;
    const uint32_t r = A.r, c = A.c;
    F.Q = Mat(c, c);
    for (uint32_t i = 0; i < c; ++i) F.Q.at(i, i) = 1 % R.q;
    if (wantP) {
        F.P = Mat(r, r);
        for (uint32_t i = 0; i < r; ++i) F.P.at(i, i) = 1 % R.q;
    }
    for (uint32_t k = 0; k < std::min(r, c); ++k) {
        uint32_t bv = R.m, bi = 0, bj = 0;
        for (uint32_t i = k; i < r && bv > 0; ++i)
            for (uint32_t j = k; j < c; ++j) {
                const uint64_t x = A.at(i, j);
                if (!x) continue;
                const uint32_t v = R.val(x);
                if (v < bv) {
                    bv = v;
                    bi = i;
                    bj = j;
                    if (v == 0) break;
                }
            }
        if (bv >= R.m) break;
        // swap rows k, bi and columns k, bj
        if (bi != k) {
            for (uint32_t j = 0; j < c; ++j) std::swap(A.at(k, j), A.at(bi, j));
            if (wantP)
                for (uint32_t j = 0; j < r; ++j) std::swap(F.P.at(k, j), F.P.at(bi, j));
        }
        if (bj != k) {
            for (uint32_t i = 0; i < r; ++i) std::swap(A.at(i, k), A.at(i, bj));
            for (uint32_t i = 0; i < c; ++i) std::swap(F.Q.at(i, k), F.Q.at(i, bj));
        }
        // scale column k so that the pivot is exactly ell^bv
        const uint64_t pv = R.ell_pow(bv);
        const uint64_t unit = A.at(k, k) / pv;
        const uint64_t iu = R.inv(unit);
        for (uint32_t i = 0; i < r; ++i) A.at(i, k) = R.mul(A.at(i, k), iu);
        for (uint32_t i = 0; i < c; ++i) F.Q.at(i, k) = R.mul(F.Q.at(i, k), iu);
        // clear row k (column operations)
        for (uint32_t j = k + 1; j < c; ++j) {
            const uint64_t y = A.at(k, j);
            if (!y) continue;
            const uint64_t f = y / pv;
            for (uint32_t i = 0; i < r; ++i)
                if (A.at(i, k)) A.at(i, j) = R.sub(A.at(i, j), R.mul(f, A.at(i, k)));
            for (uint32_t i = 0; i < c; ++i)
                if (F.Q.at(i, k)) F.Q.at(i, j) = R.sub(F.Q.at(i, j), R.mul(f, F.Q.at(i, k)));
        }
        // clear column k (row operations)
        for (uint32_t i = k + 1; i < r; ++i) {
            const uint64_t y = A.at(i, k);
            if (!y) continue;
            const uint64_t f = y / pv;
            for (uint32_t j = k; j < c; ++j)
                if (A.at(k, j)) A.at(i, j) = R.sub(A.at(i, j), R.mul(f, A.at(k, j)));
            if (wantP)
                for (uint32_t j = 0; j < r; ++j)
                    if (F.P.at(k, j)) F.P.at(i, j) = R.sub(F.P.at(i, j), R.mul(f, F.P.at(k, j)));
        }
        F.v.push_back(bv);
    }
    return F;
}

void KernelGenerators(const Ring& R, const Mat& A, std::vector<Vec>& gens, std::vector<uint32_t>& exps) {
    gens.clear();
    exps.clear();
    const SmithForm F = Smith(R, A, false);
    const uint32_t np = static_cast<uint32_t>(F.v.size());
    for (uint32_t k = 0; k < A.c; ++k) {
        uint32_t e = R.m;
        uint64_t s = 1 % R.q;
        if (k < np) {
            if (F.v[k] == 0) continue;
            e = F.v[k];
            s = R.ell_pow(R.m - F.v[k]);
        }
        Vec g(A.c);
        for (uint32_t i = 0; i < A.c; ++i) g[i] = R.mul(s, F.Q.at(i, k));
        gens.push_back(g);
        exps.push_back(e);
    }
}

uint32_t SpanLength(const Ring& R, const std::vector<Vec>& cols, uint32_t rows) {
    if (cols.empty()) return 0;
    const SmithForm F = Smith(R, ColumnsToMat(cols, rows), false);
    uint32_t L = 0;
    for (uint32_t v : F.v) L += R.m - v;
    return L;
}

std::vector<uint32_t> SpanInvariantFactors(const Ring& R, const std::vector<Vec>& cols, uint32_t rows) {
    std::vector<uint32_t> e;
    if (cols.empty()) return e;
    const SmithForm F = Smith(R, ColumnsToMat(cols, rows), false);
    for (uint32_t v : F.v) e.push_back(R.m - v);
    std::sort(e.rbegin(), e.rend());
    return e;
}

bool Solve(const Ring& R, const SmithForm& F, const Vec& y, Vec& x) {
    ENGINE_CHECK_MSG(F.hasP, "birch::Solve: the Smith form was computed without P");
    const Vec z = MatVec(R, F.P, y);
    const uint32_t np = static_cast<uint32_t>(F.v.size());
    Vec xp(F.Q.r, 0);
    for (uint32_t k = 0; k < z.size(); ++k) {
        if (k < np) {
            if (R.val(z[k]) < F.v[k]) return false;
            xp[k] = z[k] / R.ell_pow(F.v[k]);
        } else if (z[k] != 0) {
            return false;
        }
    }
    x = MatVec(R, F.Q, xp);
    return true;
}

Vec PivotBasis::coords(const Vec& w) const {
    Vec c(basis.size());
    for (size_t i = 0; i < basis.size(); ++i) c[i] = w[pivot[i]];
    return c;
}

PivotBasis SaturatedBasis(const Ring& R, std::vector<Vec> cols) {
    PivotBasis B;
    std::vector<char> used(cols.size(), 0);
    for (;;) {
        size_t gi = cols.size();
        uint32_t row = 0;
        for (size_t g = 0; g < cols.size() && gi == cols.size(); ++g) {
            if (used[g]) continue;
            for (uint32_t t = 0; t < cols[g].size(); ++t)
                if (R.unit(cols[g][t])) {
                    gi = g;
                    row = t;
                    break;
                }
        }
        if (gi == cols.size()) break;
        used[gi] = 1;
        const uint64_t iv = R.inv(cols[gi][row]);
        for (auto& x : cols[gi]) x = R.mul(x, iv);
        for (size_t h = 0; h < cols.size(); ++h) {
            if (h == gi) continue;
            const uint64_t f = cols[h][row];
            if (!f) continue;
            for (size_t t = 0; t < cols[h].size(); ++t) cols[h][t] = R.sub(cols[h][t], R.mul(f, cols[gi][t]));
        }
        for (auto& b : B.basis) {
            const uint64_t f = b[row];
            if (!f) continue;
            for (size_t t = 0; t < b.size(); ++t) b[t] = R.sub(b[t], R.mul(f, cols[gi][t]));
        }
        B.basis.push_back(cols[gi]);
        B.pivot.push_back(row);
    }
    for (size_t g = 0; g < cols.size(); ++g)
        if (!used[g] && !IsZero(cols[g])) B.free_summand = false;
    return B;
}

Vec CharPoly(const Ring& R, const Mat& A) {
    ENGINE_CHECK_MSG(A.r == A.c, "birch::CharPoly: square");
    return zk::Charpoly(R, A.r, [&](size_t i, size_t j) { return A.at(static_cast<uint32_t>(i), static_cast<uint32_t>(j)); });
}

// ---- the cone side ---------------------------------------------------------------------------------
Vec BirchConeCell::symbol(const std::vector<uint64_t>& coords) const {
    ENGINE_CHECK_MSG(coords.size() == P.size(), "BirchConeCell::symbol: coordinates on the cyclic summands of H^1_c expected");
    Vec out(n + 1, 0);
    for (size_t i = 0; i < coords.size(); ++i) {
        const uint64_t c = coords[i] % R.q;
        if (!c) continue;
        for (uint32_t t = 0; t <= n; ++t) out[t] = R.add(out[t], R.mul(c, P[i][t]));
    }
    return out;
}

namespace {
template<class Get>
Vec PsiOfFlat(const SymRho& rho, uint32_t dV, uint32_t e04, uint32_t e46, uint32_t bd_v0_row, Get get, int g_sign) {
    const Ring& R = rho.ring();
    Vec F(dV, 0);
    for (uint32_t a = 0; a < dV; ++a) {
        uint64_t f = R.add(get(e04 * dV + a) % R.q, get(e46 * dV + a) % R.q);
        const uint64_t cd = get(bd_v0_row + a) % R.q;
        f = (g_sign < 0) ? R.sub(f, cd) : R.add(f, cd);   // g := -c_d (g_sign = -1, the definition); +c_d the control
        F[a] = f;
    }
    return VSub(R, rho.S(F), F);
}
} // namespace

BirchConeCell RunBirchConeCell(uint32_t n, uint64_t ell, uint32_t m, const BirchCellOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(n >= 2 && n % 2 == 0, "BirchModularSymbol: n >= 2 even (odd n vanishes on this model)");
    ENGINE_CHECK_MSG(ell >= 5 && PrimitiveRootModPrime(ell) > 0, "BirchModularSymbol: ell >= 5 prime");
    ENGINE_CHECK_MSG(m >= 1, "BirchModularSymbol: m >= 1");
    BirchConeCell cell;
    cell.n = n;
    cell.ell = ell;
    cell.m = m;
    cell.R = MakeRing(ell, m);
    const uint32_t dV = n + 1;
    const SymRho rho(n, cell.R);

    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*K));
    }
    const int64_t e04i = K->find_simplex_index(1, {0, 4}), e46i = K->find_simplex_index(1, {4, 6}), v6i = K->find_simplex_index(0, {6});
    ENGINE_CHECK_MSG(e04i >= 0 && e46i >= 0 && v6i >= 0, "BirchModularSymbol: the model has no edge v0v4 / v4v6 or no vertex v6");
    const uint32_t e04 = static_cast<uint32_t>(e04i), e46 = static_cast<uint32_t>(e46i), v6 = static_cast<uint32_t>(v6i);

    if (m == 1) {
        // ---- F_ell: the field cone of the shipped pair ----
        ENGINE_CHECK_MSG(ell < (uint64_t(1) << 32), "BirchModularSymbol: the F_ell cone is uint32_t (ell < 2^32)");
        cell.route = "field cone (EquivariantMappingCone, SL2ZBoundaryPair)";
        const uint32_t ell32 = static_cast<uint32_t>(ell);
        LocalSystemSimplicialComplex lssc(*K, GenerateSL2ZRelationsElt<uint32_t>(*K, n, ell32), dV, ell32);
        EquivariantCohomologyOnK coh(&lssc);
        coh.compute_cohomology();
        hecke::SL2ZBoundaryPair pair(&coh, *K);
        const std::unique_ptr<EquivariantMappingCone> cone = pair.make_cone();
        const int64_t bv0 = pair.K_bd().find_simplex_index(0, {0});
        ENGINE_CHECK_MSG(bv0 >= 0, "BirchModularSymbol: the circle model has no vertex v0");
        const uint32_t off = cone->bulk_part_dim(1), bd_row = off + static_cast<uint32_t>(bv0) * dV;
        const engine::DenseBuf<uint32_t> H = cone->H_block(1).to_dense(engine::Layout::ColMajor);
        ENGINE_CHECK_MSG(H.rows == cone->flat_dim(1), "BirchModularSymbol: H^1_c basis is not in the flat cone layout");
        for (uint32_t j = 0; j < H.cols; ++j) {
            auto get = [&](uint32_t row) -> uint64_t { return H.at(row, j); };
            cell.P.push_back(PsiOfFlat(rho, dV, e04, e46, bd_row, get, -1));
            cell.P_wrong_sign.push_back(PsiOfFlat(rho, dV, e04, e46, bd_row, get, +1));
            cell.exps.push_back(1);
        }
        if (opt.coboundary_check) {
            const engine::DenseBuf<uint32_t> C0 = cone->Ceq_block(0).to_dense(engine::Layout::ColMajor);
            bool ok = true;
            for (uint32_t j = 0; j < C0.cols && ok; ++j) {
                Vec x(dV);
                for (uint32_t a = 0; a < dV; ++a) x[a] = C0.at(v6 * dV + a, j);
                ok = (rho.S(x) == x);
                ++cell.coboundaries_checked;
            }
            cell.coboundary_route = "C^0_eq at v6 in V^S";
            if (ok && n <= 80) {
                const engine::DenseBuf<uint32_t> D0 = cone->D_block(0).to_dense(engine::Layout::ColMajor);
                std::vector<uint32_t> rows;
                for (uint32_t a = 0; a < dV; ++a) {
                    rows.push_back(e04 * dV + a);
                    rows.push_back(e46 * dV + a);
                    rows.push_back(bd_row + a);
                }
                for (uint32_t j = 0; j < C0.cols && ok; ++j) {
                    std::vector<uint64_t> col(D0.rows, 0);
                    for (uint32_t r : rows) {
                        uint64_t s = 0;
                        for (uint32_t k = 0; k < D0.cols; ++k)
                            if (D0.at(r, k) && C0.at(k, j)) s = (s + static_cast<uint64_t>(D0.at(r, k)) * C0.at(k, j)) % ell;
                        col[r] = s;
                    }
                    auto get = [&](uint32_t row) -> uint64_t { return col[row]; };
                    ok = IsZero(PsiOfFlat(rho, dV, e04, e46, bd_row, get, -1));
                    ++cell.coboundaries_checked;
                }
                cell.coboundary_route += " + Psi(D_c C^0_eq) = 0";
            }
            cell.coboundaries_killed = ok;
        }
        if (opt.pi_matrix) {
            const FlatMatrix& pi = cone->pi(1);
            cell.pi = Mat(pi.get_rows(), pi.get_cols());
            for (uint32_t i = 0; i < cell.pi.r; ++i)
                for (uint32_t j = 0; j < cell.pi.c; ++j) cell.pi.at(i, j) = pi.get_entry(i, j);
            cell.have_pi = true;
            cell.bulk_exps.assign(cell.pi.r, 1);
        }
        ENGINE_CHECK_MSG(opt.hecke_primes.empty() && !opt.iota_through_pi,
                         "BirchModularSymbol: the cone Hecke action and the iota reading are read over Z/ell^m, m >= 2");
    } else {
        // ---- Z/ell^m: the module cone over uint64_t ----
        cell.route = "module cone (ModuleConeT<uint64_t>)";
        const SimplicialComplex Kb = CreateSL2ZBoundaryPathTriangulation();
        struct Tab {
            std::unique_ptr<hecke::HeckeCosetTablesT<uint64_t>> cosets;
            std::unique_ptr<hecke::HeckeTauTableT<uint64_t>> tau;
            std::unique_ptr<hecke::HeckeBoundaryTauTableT<uint64_t>> taub;
            std::unique_ptr<hecke::HeckePairHomotopyT<uint64_t>> eta;
        };
        std::vector<Tab> tabs(opt.hecke_primes.size());
        for (size_t i = 0; i < opt.hecke_primes.size(); ++i) {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            tabs[i].cosets.reset(new hecke::HeckeCosetTablesT<uint64_t>(opt.hecke_primes[i], 1, ell, n));
            tabs[i].tau.reset(new hecke::HeckeTauTableT<uint64_t>(*K, *model, *tabs[i].cosets, ell, n));
            tabs[i].taub.reset(new hecke::HeckeBoundaryTauTableT<uint64_t>(*K, *model, *tabs[i].cosets));
            tabs[i].eta.reset(new hecke::HeckePairHomotopyT<uint64_t>(*tabs[i].tau, *tabs[i].taub));
        }
        hecke::ScopedCoefficientRing ring(m);
        const engine::ModCtx<uint64_t> ctx = hecke::CoefficientRingContext<uint64_t>("BirchModularSymbol", ell, m);
        ENGINE_CHECK_MSG(ctx.p == cell.R.q, "BirchModularSymbol: ring context mismatch");
        LocalSystemSimplicialComplexT<uint64_t> lssc(*K, hecke::GenerateSL2ZRelationsRing<uint64_t>(*K, n, ctx), dV, ctx.p);
        EquivariantCohomologyOnKT<uint64_t> coh(&lssc);
        coh.compute_cohomology();
        LocalSystemSimplicialComplexT<uint64_t> lsscb(Kb, hecke::GenerateSL2ZBoundaryRelationsRing<uint64_t>(Kb, n, ctx), dV, ctx.p);
        EquivariantCohomologyOnKT<uint64_t> cohb(&lsscb);
        cohb.compute_cohomology();
        std::vector<engine::DenseBuf<uint64_t>> rf;
        for (uint32_t q = 0; q <= 1; ++q) rf.push_back(RestrictionFlatRing<uint64_t>(*K, Kb, q, dV, ctx));
        ModuleConeT<uint64_t> MC(&coh, &cohb, std::move(rf));
        MC.compute();
        const int64_t bv0 = Kb.find_simplex_index(0, {0});
        ENGINE_CHECK_MSG(bv0 >= 0, "BirchModularSymbol: the circle model has no vertex v0");
        const uint32_t off = MC.bulk_part_dim(1), bd_row = off + static_cast<uint32_t>(bv0) * dV;
        const engine::DenseBuf<uint64_t>& H = MC.H_module_basis(1);
        ENGINE_CHECK_MSG(H.rows == MC.flat_dim(1), "BirchModularSymbol: H^1_c basis is not in the flat cone layout");
        cell.exps = MC.H_invariant_factors(1);
        ENGINE_CHECK_MSG(cell.exps.size() == H.cols, "BirchModularSymbol: exponents / basis mismatch");
        for (uint32_t j = 0; j < H.cols; ++j) {
            auto get = [&](uint32_t row) -> uint64_t { return H.at(row, j); };
            cell.P.push_back(PsiOfFlat(rho, dV, e04, e46, bd_row, get, -1));
            cell.P_wrong_sign.push_back(PsiOfFlat(rho, dV, e04, e46, bd_row, get, +1));
        }
        if (opt.coboundary_check) {
            const engine::DenseBuf<uint64_t>& C0 = MC.degree(0).C;
            bool ok = true;
            for (uint32_t j = 0; j < C0.cols && ok; ++j) {
                Vec x(dV);
                for (uint32_t a = 0; a < dV; ++a) x[a] = C0.at(v6 * dV + a, j) % cell.R.q;
                ok = (rho.S(x) == x);
                ++cell.coboundaries_checked;
            }
            cell.coboundary_route = "C^0_eq at v6 in V^S";
            if (ok && n <= 80) {
                const engine::DenseBuf<uint64_t> D0 = MC.D_flat(0);
                std::vector<uint32_t> rows;
                for (uint32_t a = 0; a < dV; ++a) {
                    rows.push_back(e04 * dV + a);
                    rows.push_back(e46 * dV + a);
                    rows.push_back(bd_row + a);
                }
                for (uint32_t j = 0; j < C0.cols && ok; ++j) {
                    std::vector<uint64_t> col(D0.rows, 0);
                    for (uint32_t r : rows) {
                        uint64_t s = 0;
                        for (uint32_t k = 0; k < D0.cols; ++k)
                            if (D0.at(r, k) && C0.at(k, j)) s = cell.R.add(s, cell.R.mul(D0.at(r, k) % cell.R.q, C0.at(k, j) % cell.R.q));
                        col[r] = s;
                    }
                    auto get = [&](uint32_t row) -> uint64_t { return col[row]; };
                    ok = IsZero(PsiOfFlat(rho, dV, e04, e46, bd_row, get, -1));
                    ++cell.coboundaries_checked;
                }
                cell.coboundary_route += " + Psi(D_c C^0_eq) = 0";
            }
            cell.coboundaries_killed = ok;
        }
        const uint32_t t = static_cast<uint32_t>(cell.exps.size());
        for (size_t i = 0; i < opt.hecke_primes.size(); ++i) {
            hecke::HeckeOperatorOnKT<uint64_t> bulk(&coh, tabs[i].cosets.get(), tabs[i].tau.get());
            bulk.compute();
            hecke::HeckeBoundaryOperatorT<uint64_t> bd(&cohb, tabs[i].cosets.get(), tabs[i].taub.get());
            bd.compute();
            hecke::ModuleConeActionT<uint64_t> act(&MC, &bulk, &bd, tabs[i].eta.get());
            act.compute();
            const engine::DenseBuf<uint64_t>& A = act.A_c(1);
            ENGINE_CHECK_MSG(A.rows == t && A.cols == t, "BirchModularSymbol: A_c(1) shape");
            Mat M(t, t);
            for (uint32_t r = 0; r < t; ++r)
                for (uint32_t c = 0; c < t; ++c) M.at(r, c) = A.at(r, c) % cell.R.q;
            cell.hecke_primes.push_back(opt.hecke_primes[i]);
            cell.A_c.push_back(M);
        }
        if (opt.pi_matrix || opt.iota_through_pi) {
            const engine::DenseBuf<uint64_t>& pi = MC.pi(1);
            cell.pi = Mat(pi.rows, pi.cols);
            for (uint32_t r = 0; r < pi.rows; ++r)
                for (uint32_t c = 0; c < pi.cols; ++c) cell.pi.at(r, c) = pi.at(r, c) % cell.R.q;
            cell.have_pi = true;
            cell.bulk_exps = coh.H_invariant_factors(1);
        }
        if (opt.iota_through_pi) {
            const hecke::IotaSimplicialMap im = hecke::BuildIotaSimplicialMapOnModel(*K, *model);
            const engine::DenseBuf<uint64_t>& HB = coh.H_module_basis(1);
            ENGINE_CHECK_MSG(im.num_degrees() >= 2, "BirchModularSymbol: iota_K has no degree 1");
            const uint32_t N1 = static_cast<uint32_t>(im.image_index[1].size());
            const uint32_t tK = HB.cols;
            ENGINE_CHECK_MSG(HB.rows == N1 * dV, "BirchModularSymbol: the bulk basis is not in flat 1-cochain coordinates");
            engine::DenseBuf<uint64_t> Y(HB.rows, tK);
            for (uint32_t s = 0; s < N1; ++s) {
                const uint32_t tt = im.image_index[1][s];
                const int eps = im.image_sign[1][s];
                for (uint32_t i = 0; i < dV; ++i) {
                    const bool rho_minus = ((n - i) % 2u) == 1u;
                    const bool neg = (eps > 0) != rho_minus;   // sign = -eps * (-1)^{n-i}: [C]'s iota^* = -iota^*_geo
                    for (uint32_t j = 0; j < tK; ++j) {
                        const uint64_t x = HB.at(tt * dV + i, j) % cell.R.q;
                        Y.at(s * dV + i, j) = neg ? cell.R.neg(x) : x;
                    }
                }
            }
            const std::vector<std::vector<uint64_t>> co = coh.H_class_coordinates(1, engine::cview(Y));
            cell.iota_C = Mat(tK, tK);
            for (uint32_t j = 0; j < tK; ++j)
                for (uint32_t i = 0; i < tK; ++i) cell.iota_C.at(i, j) = co[j][i] % cell.R.q;
            cell.have_iota = true;
        }
    }
    cell.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return cell;
}

// ---- (L1) ------------------------------------------------------------------------------------------
namespace {
// the generalised e-eigenspace (Fitting block) of T on a free module with pivot basis B: generators in V
bool FittingBlock(const Ring& R, const PivotBasis& B, const Mat& T, uint64_t e, std::vector<Vec>& block) {
    const uint32_t r = static_cast<uint32_t>(B.basis.size());
    Mat t = T;
    for (uint32_t i = 0; i < r; ++i) t.at(i, i) = R.sub(t.at(i, i), e % R.q);
    Mat tN(r, r);
    for (uint32_t i = 0; i < r; ++i) tN.at(i, i) = 1 % R.q;
    for (uint32_t s = 0; s < r * R.m + 1; ++s) tN = MatMul(R, tN, t);
    std::vector<Vec> gens;
    std::vector<uint32_t> exps;
    KernelGenerators(R, tN, gens, exps);
    block.clear();
    bool free = true;
    for (size_t g = 0; g < gens.size(); ++g) {
        if (exps[g] != R.m) free = false;
        Vec v(B.basis[0].size(), 0);
        for (uint32_t i = 0; i < r; ++i)
            if (gens[g][i])
                for (size_t x = 0; x < v.size(); ++x) v[x] = R.add(v[x], R.mul(gens[g][i], B.basis[i][x]));
        block.push_back(v);
    }
    return free;
}
Mat HeckeMatrixOn(const SymRho& rho, const PivotBasis& B, uint64_t p) {
    const uint32_t r = static_cast<uint32_t>(B.basis.size());
    Mat T(r, r);
    for (uint32_t j = 0; j < r; ++j) {
        const Vec c = B.coords(HeckeTp(rho, B.basis[j], p));
        for (uint32_t i = 0; i < r; ++i) T.at(i, j) = c[i];
    }
    return T;
}

// (slot-20 WP-LV3) m_Eis is (ell, T_p - 1 - p^{k-1} : p), not (ell, T_2 - e_2) alone: when the T_2-Fitting block is larger
// than the reading needs (`want`: a cusp form whose a_2 alone is congruent to e_2, e.g. at (282, 37)), it is cut down by the
// generalised (1 + p^{k-1})-eigenspaces of T_3 and T_5 inside it.  At `want` or below nothing changes (the slot-19 values).
bool RefineToEisenstein(const Ring& R, const SymRho& rho, uint32_t k, uint32_t want, bool free, std::vector<Vec>& block) {
    for (uint64_t p : {3ull, 5ull}) {
        if (!free || block.size() <= want || p % R.ell == 0) break;
        const PivotBasis Bb = SaturatedBasis(R, block);
        if (!Bb.free_summand) return false;
        free = FittingBlock(R, Bb, HeckeMatrixOn(rho, Bb, p), R.add(1 % R.q, R.pow(p, k - 1)), block);
    }
    return free;
}
} // namespace

EisensteinSymbolReading ReadEisensteinSymbols(const BirchConeCell& cell) {
    EisensteinSymbolReading E;
    const Ring& R = cell.R;
    const uint32_t n = cell.n, k = n + 2;
    const SymRho rho(n, R);
    for (uint32_t e : cell.exps)
        if (e != R.m) {
            E.reason = "H^1_c has ell-torsion (the free reading is not implemented for this index statement)";
            return E;
        }
    std::vector<Vec> ev, od;
    for (const Vec& P : cell.P) {
        ev.push_back(SignPart(R, P, +1));
        od.push_back(SignPart(R, P, -1));
    }
    const PivotBasis Be = SaturatedBasis(R, ev), Bo = SaturatedBasis(R, od);
    if (!Be.free_summand || !Bo.free_summand || Be.basis.empty()) {
        E.reason = "the geometric sign parts of W are not free direct summands";
        return E;
    }
    E.e2 = R.add(1 % R.q, R.pow(2, k - 1));
    std::vector<Vec> block;
    const bool free = RefineToEisenstein(R, rho, k, 2, FittingBlock(R, Be, HeckeMatrixOn(rho, Be, 2), E.e2, block), block);
    E.block_rank = static_cast<uint32_t>(block.size());
    Vec p0(n + 1, 0);
    p0[0] = 1 % R.q;
    p0[n] = R.neg(1 % R.q);
    if (free && block.size() == 1) {
        // no cusp line in the block (f = 0): it is <p_0>, depth 0 (a regular class)
        const PivotBasis B1 = SaturatedBasis(R, block);
        if (B1.basis.size() == 1 && B1.pivot[0] == 0 && B1.basis[0] == p0) {
            E.applicable = true;
            E.depth = 0;
            E.depth_exact = true;
        } else {
            E.reason = "the rank-1 m_Eis block is not <p_0>";
        }
        return E;
    }
    if (!free || block.size() != 2) {
        E.reason = "the m_Eis block of the geometric-even part has rank " + std::to_string(block.size()) + (free ? "" : " (not free)") +
                   " (the reading needs rank 2: p_0 and one cusp line)";
        return E;
    }
    const PivotBasis Bb = SaturatedBasis(R, block);
    {
        const Vec c = Bb.coords(p0);
        Vec back(n + 1, 0);
        for (size_t i = 0; i < c.size(); ++i)
            for (uint32_t t = 0; t <= n; ++t) back[t] = R.add(back[t], R.mul(c[i], Bb.basis[i][t]));
        if (back != p0) {
            E.reason = "p_0 is not in the m_Eis block";
            return E;
        }
    }
    const PivotBasis Bp = SaturatedBasis(R, {p0, block[0], block[1]});
    if (Bp.basis.size() != 2 || Bp.pivot[0] != 0) {
        E.reason = "no basis {p_0, w} of the block";
        return E;
    }
    // Bp.basis[0] = p_0 normalised at u_0 (p_0 has u_0-coordinate 1), Bp.basis[1] = w
    const Vec& w = Bp.basis[1];
    const Vec cw = Bp.coords(HeckeTp(rho, w, 2));
    const uint64_t y = cw[0], a = cw[1];
    E.a2 = a;
    const uint64_t delta = R.sub(a, E.e2);
    E.v_delta = R.val(delta);
    E.v_y = R.val(y);
    E.depth_exact = E.v_delta < R.m;
    E.depth = (E.v_y <= E.v_delta) ? E.v_delta - E.v_y : 0;
    const uint32_t mn = std::min(E.v_y, E.v_delta);
    const uint64_t sc = R.ell_pow(mn);
    E.r_plus.assign(n + 1, 0);
    for (uint32_t t = 0; t <= n; ++t) {
        const uint64_t x = R.add(R.mul(y, Bp.basis[0][t]), R.mul(delta, w[t]));
        E.r_plus[t] = sc ? x / sc : 0;   // exact: every entry is divisible by ell^mn
    }
    // the geometric-odd m_Eis block
    std::vector<Vec> ob;
    if (!Bo.basis.empty() && RefineToEisenstein(R, rho, k, 1, FittingBlock(R, Bo, HeckeMatrixOn(rho, Bo, 2), E.e2, ob), ob) && ob.size() == 1) {
        E.r_minus = ob[0];
        E.have_r_minus = true;
    }
    E.applicable = true;
    return E;
}

} // namespace birch

// ===========================================================================
// BirchModularSymbolRoute ([E] rmk:lvshapes (B), kind linear)
// ===========================================================================
namespace {
uint32_t LegendreV(uint64_t N, uint64_t ell) {   // v_ell(N!)
    uint32_t v = 0;
    while (N) {
        N /= ell;
        v += static_cast<uint32_t>(N);
    }
    return v;
}
} // namespace

void BirchModularSymbolRoute::RequireSupportedDomain(const Domain& d, uint64_t twist_conductor) {
    // (slot-22 P6) the refusal of Gamma0NShapiro is lifted: level N by Shapiro on the same 7-vertex model (Gamma0NShapiro.h).
    const bool shapiro_model = d.model == ModelId::Gamma0NShapiro;
    if (d.model != ModelId::SL2ZSevenVertex && !shapiro_model)
        throw hecke::NotImplementedInThisBuild(std::string("BirchModularSymbolRoute: model ") + ToString(d.model) +
                                               " is not supported (Phi({0,oo}) = (rho(S)-1)F(i) is the 7-vertex model's formula, applied to Sym^n "
                                               "(SL2ZSevenVertex) or to Ind V (Gamma0NShapiro); a new ModelId must provide its own F(i), R19-8 Sec. 6)");
    if (d.cohomology != "H^1_c")
        throw std::invalid_argument("BirchModularSymbolRoute: the domain is H^1_c (modular symbols; the functional descends to H^1_m only "
                                    "at a residually non-Eisenstein m, isign), not '" + d.cohomology + "'");
    if (!shapiro_model && d.level != 1)
        throw hecke::NotImplementedInThisBuild("BirchModularSymbolRoute: level " + std::to_string(d.level) +
                                               " on SL2ZSevenVertex (level N is the Gamma0NShapiro model)");
    if (shapiro_model && (d.level < 1 || d.level > shapiro::P1N::kMaxLevel))
        throw std::invalid_argument("BirchModularSymbolRoute: Gamma0NShapiro needs 1 <= N <= " + std::to_string(shapiro::P1N::kMaxLevel));
    if (shapiro_model ? (d.n_sym % 2 != 0) : (d.n_sym < 2 || d.n_sym % 2 != 0))
        throw std::invalid_argument(shapiro_model ? "BirchModularSymbolRoute: n >= 0 even" : "BirchModularSymbolRoute: n >= 2 even");
    if (d.ell < 5 || PrimitiveRootModPrime(d.ell) == 0) throw std::invalid_argument("BirchModularSymbolRoute: ell >= 5 prime");
    if (d.m == 0) throw std::invalid_argument("BirchModularSymbolRoute: a finite precision R = Z/ell^m, m >= 1, is required");
    if (twist_conductor == 0 || twist_conductor % d.ell == 0)
        throw std::invalid_argument("BirchModularSymbolRoute: ell must not divide the conductor M = " + std::to_string(twist_conductor));
    if (shapiro_model && std::gcd(twist_conductor, d.level) != 1)
        throw std::invalid_argument("BirchModularSymbolRoute: gcd(M, N) = 1 is required at level N (the MTT dictionary of the twist)");
}

BirchModularSymbolRoute::BirchModularSymbolRoute(Domain d, uint64_t twist_conductor, uint32_t j)
    : d_(std::move(d)), M_(twist_conductor), j_(j) {
    RequireSupportedDomain(d_, M_);
    if (j_ > d_.n_sym) throw std::invalid_argument("BirchModularSymbolRoute: 0 <= j <= n");
    chi_ = birch::QuadraticCharacterOfConductor(M_, birch::MakeRing(d_.ell, d_.m));
}

BirchModularSymbolRoute::BirchModularSymbolRoute(std::shared_ptr<const birch::BirchConeCell> cell, birch::DirichletCharacter chi, uint32_t j)
    : M_(chi.M), j_(j), chi_(std::move(chi)), cell_(std::move(cell)) {
    ENGINE_CHECK_MSG(cell_ != nullptr, "BirchModularSymbolRoute: null cell");
    d_.model = ModelId::SL2ZSevenVertex;
    d_.cohomology = "H^1_c";
    d_.n_sym = cell_->n;
    d_.level = 1;
    d_.ell = cell_->ell;
    d_.m = cell_->m;
    d_.subquotient = "H^1_c (the whole module)";
    RequireSupportedDomain(d_, M_);
    if (j_ > d_.n_sym) throw std::invalid_argument("BirchModularSymbolRoute: 0 <= j <= n");
    ENGINE_CHECK_MSG(chi_.chi.size() == chi_.M && chi_.chibar.size() == chi_.M, "BirchModularSymbolRoute: character table size");
    for (uint64_t x : chi_.chibar) ENGINE_CHECK_MSG(x < cell_->R.q, "BirchModularSymbolRoute: the character is not over R");
}

const birch::BirchConeCell& BirchModularSymbolRoute::cell() const {
    if (d_.model != ModelId::SL2ZSevenVertex)
        throw hecke::NotImplementedInThisBuild("BirchModularSymbolRoute::cell: the cone cell is the 7-vertex model's (use shapiro_cell())");
    if (!cell_) cell_ = std::make_shared<const birch::BirchConeCell>(birch::RunBirchConeCell(d_.n_sym, d_.ell, d_.m, birch::BirchCellOptions()));
    return *cell_;
}

const shapiro::Gamma0NCell& BirchModularSymbolRoute::shapiro_cell() const {
    if (d_.model != ModelId::Gamma0NShapiro)
        throw std::logic_error("BirchModularSymbolRoute::shapiro_cell: the domain's model is not Gamma0NShapiro");
    if (!ncell_) ncell_ = std::make_shared<const shapiro::Gamma0NCell>(shapiro::RunGamma0NCell(d_.level, d_.n_sym, d_.ell, d_.m, 0));
    return *ncell_;
}

birch::Vec BirchModularSymbolRoute::symbol(const CohomologyClass& x) const {
    if (d_.model == ModelId::Gamma0NShapiro) {
        const shapiro::Gamma0NCell& nc = shapiro_cell();
        ENGINE_CHECK_MSG(x.coordinates.size() == nc.basis.size(), "BirchModularSymbolRoute: a class is given by its coordinates on the "
                                                                    "generators of W(Ind V)");
        if (!x.exponents.empty()) ENGINE_CHECK_MSG(x.exponents == nc.exps, "BirchModularSymbolRoute: the class's exponents are not the cell's");
        const birch::Vec P = nc.symbol(x.coordinates);
        ENGINE_CHECK_MSG(shapiro::InWN(birch::SymRho(nc.n, nc.R), shapiro::P1N(nc.N), P),
                               "BirchModularSymbolRoute: the level-N symbol violates Manin's relations");
        return P;
    }
    const birch::BirchConeCell& c = cell();
    ENGINE_CHECK_MSG(x.coordinates.size() == c.P.size(), "BirchModularSymbolRoute: a class is given by its coordinates on the cyclic "
                                                          "summands of H^1_c");
    if (!x.exponents.empty()) ENGINE_CHECK_MSG(x.exponents == c.exps, "BirchModularSymbolRoute: the class's exponents are not the cell's");
    const birch::Vec P = c.symbol(x.coordinates);
    const birch::SymRho rho(c.n, c.R);
    ENGINE_CHECK_MSG(birch::InW(rho, P), "BirchModularSymbolRoute: Phi({0,oo}) violates Manin's relations (G2)");
    return P;
}

std::vector<std::vector<uint64_t>> BirchModularSymbolRoute::evaluate_all_j(const std::vector<CohomologyClass>& classes) const {
    if (d_.model == ModelId::Gamma0NShapiro) {
        const shapiro::Gamma0NCell& nc = shapiro_cell();
        const birch::SymRho rho(nc.n, nc.R);
        const shapiro::P1N p1(nc.N);
        std::vector<std::vector<uint64_t>> out;
        for (const CohomologyClass& x : classes) out.push_back(shapiro::EvAllJN(rho, p1, symbol(x), chi_));
        return out;
    }
    const birch::BirchConeCell& c = cell();
    const birch::SymRho rho(c.n, c.R);
    std::vector<std::vector<uint64_t>> out;
    for (const CohomologyClass& x : classes) out.push_back(birch::EvAllJ(rho, symbol(x), chi_));
    return out;
}

std::vector<uint64_t> BirchModularSymbolRoute::evaluate_functional(const std::vector<CohomologyClass>& classes) const {
    std::vector<uint64_t> out;
    for (const auto& v : evaluate_all_j(classes)) out.push_back(v[j_]);
    return out;
}

NormalisationRecord BirchModularSymbolRoute::normalisation() const {
    NormalisationRecord N;
    const int geo = chi_.parity * ((j_ % 2 == 0) ? 1 : -1);
    N.l_function = "L(f,chi,s)";
    if (d_.model == ModelId::Gamma0NShapiro) N.l_function += ", f on Gamma_0(" + std::to_string(d_.level) + ") (Shapiro; Phi at the identity coset)";
    N.s0 = "j+1=" + std::to_string(j_ + 1);
    N.twist = chi_.name + ", conductor M=" + std::to_string(M_) + ", chi(-1)=" + std::to_string(chi_.parity) +
              "; geometric sign chi(-1)(-1)^j=" + (geo > 0 ? "+" : "-") + ", [C] sign -chi(-1)(-1)^j=" + (geo > 0 ? "-" : "+");
    N.sign_at_infinity = geo > 0 ? '-' : '+';   // [C]'s convention ('+' = the Eisenstein side of H^1)
    N.period = "Omega_f^{+-} (primitive generator of the f-eigenline of H^1_{c,+-} (x) O, geometric sign)";
    const uint64_t n = d_.n_sym, ell = d_.ell;
    const uint32_t vC = LegendreV(n, ell) - LegendreV(j_, ell) - LegendreV(n - j_, ell);
    N.constants.push_back({"MTT dictionary L/Omega = c * ev", "chi(-1)(-2 pi i)^{j+1} tau(chi) / (j! C(n,j) M^{j+1})", 0, false});
    N.constants.push_back({"C(n,j)", "C(" + std::to_string(n) + "," + std::to_string(j_) + ")", static_cast<int32_t>(vC), true});
    N.constants.push_back({"j!", std::to_string(j_) + "!", static_cast<int32_t>(LegendreV(j_, ell)), true});
    N.ambiguity = "O^x (ev lives on H^1_c; it descends to H^1_m only at a residually non-Eisenstein m: isign)";
    if (d_.model == ModelId::Gamma0NShapiro)
        N.ambiguity += "; at N > 1 the constant of [E] rmk:lvshapes (B) is derived, not tested (no complex L(f,chi,j+1) at level N was computed)";
    return N;
}

IndexStatement BirchModularSymbolRoute::index_statement() const {
    if (d_.model != ModelId::SL2ZSevenVertex)
        throw hecke::NotImplementedInThisBuild("BirchModularSymbolRoute::index_statement: (L1) is the level-one Eisenstein statement; "
                                               "level N (Gamma0NShapiro) carries no index statement in this build (P6 tail)");
    const birch::BirchConeCell& c = cell();
    const birch::EisensteinSymbolReading E = birch::ReadEisensteinSymbols(c);
    if (!E.applicable)
        throw hecke::NotImplementedInThisBuild("BirchModularSymbolRoute::index_statement: (L1) needs an m_Eis block <p_0> or <p_0, r^+> "
                                               "of the geometric-even part: " + E.reason);
    const uint32_t k = d_.n_sym + 2;
    if (k % (d_.ell - 1) == 0)
        throw hecke::NotImplementedInThisBuild("BirchModularSymbolRoute::index_statement: (ell - 1) | k (B_k/k is not ell-integral)");
    const BernoulliReading B = (k <= 4000) ? BernoulliModEllPowerExact(k, d_.ell, d_.m) : BernoulliModEllPowerVoronoi(k, d_.ell, d_.m);
    IndexStatement I;
    I.label = "R19-8 Sec. 6 (L1); [E] rmk:lvshapes (B)";
    I.status = "COMP (depth 1 at the 8 Eisenstein cells; depth 2 at (282, 37), (226, 103) since slot-20 WP-LV3)";
    I.module_name = "W^{geo-even}_{m_Eis} = <p_0, r^+>: r^+ == unit * p_0 mod ell^depth";
    I.l_value = "zeta(1-k) = -B_k/k, k=" + std::to_string(k);
    I.l_valuation = B.v;
    I.l_valuation_exact = B.v_exact;
    I.expected_lo = I.expected_hi = B.v;
    I.found = E.depth;
    I.holds = E.depth_exact && B.v_exact && E.depth == B.v;
    return I;
}

} // namespace lvalue
} // namespace hecke_ext
