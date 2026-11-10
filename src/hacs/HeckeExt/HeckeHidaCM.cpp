// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 16, WP-15) [U] part 2 in code -- see the header.
// [U] sec:u-tameN (thm:u-cmN, prop:u-cmexist, prop:u-cmeis, rmk:u-pred2);
// [C, sec:nebentypus] (def:nebmodel, rmk:nebheilbronn) through Hecke/HeckeNebentypus.
#include "HeckeHidaCM.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"

namespace hecke_ext {
namespace hidacm {

namespace {
uint64_t Gcd(uint64_t a, uint64_t b) { return engine::scalar::gcd(a, b); }
uint64_t PowModU(uint64_t b, uint64_t e, uint64_t m) { return engine::scalar::powmod<uint64_t>(b, e, m); }
uint64_t ModS(int64_t x, uint64_t m) { return engine::scalar::modi(x, m); }
bool Squarefree(uint64_t n) {
    for (uint64_t p = 2; p * p <= n; ++p)
        if (n % (p * p) == 0) return false;
    return true;
}
bool IsPrimeSmall(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p = 2; p * p <= n; ++p)
        if (n % p == 0) return false;
    return true;
}
int64_t ISqrtExact(int64_t v, bool& ok) {   // ok iff v is a perfect square >= 0
    ok = false;
    if (v < 0) return 0;
    int64_t r = static_cast<int64_t>(std::sqrt(static_cast<long double>(v)));
    while (r > 0 && r * r > v) --r;
    while ((r + 1) * (r + 1) <= v) ++r;
    ok = r * r == v;
    return r;
}
} // namespace

bool IsFundamentalDiscriminant(int64_t d) {
    if (d == 0 || d == 1) return false;
    const uint64_t a = static_cast<uint64_t>(d < 0 ? -d : d);
    const int64_t r4 = ((d % 4) + 4) % 4;
    if (r4 == 1) return Squarefree(a);
    if (r4 != 0) return false;
    const int64_t m = d / 4;
    const int64_t m4 = ((m % 4) + 4) % 4;
    return (m4 == 2 || m4 == 3) && Squarefree(static_cast<uint64_t>(m < 0 ? -m : m));
}

std::vector<int64_t> NegativeFundamentalDiscriminants(int64_t dmax) {
    std::vector<int64_t> out;
    for (int64_t a = 3; a <= dmax; ++a)
        if (IsFundamentalDiscriminant(-a)) out.push_back(-a);
    return out;
}

uint32_t ClassNumber(int64_t d) {
    ENGINE_CHECK_MSG(d < 0 && IsFundamentalDiscriminant(d), "ClassNumber: d must be a negative fundamental discriminant");
    const int64_t D = -d;
    uint32_t h = 0;
    for (int64_t a = 1; 3 * a * a <= D; ++a)
        for (int64_t b = -a + 1; b <= a; ++b) {
            if (((b - d) % 2) != 0) continue;
            const int64_t num = b * b - d;
            if (num % (4 * a) != 0) continue;
            const int64_t c = num / (4 * a);
            if (c < a) continue;
            if (a == c && b < 0) continue;
            if (Gcd(Gcd(static_cast<uint64_t>(a), static_cast<uint64_t>(b < 0 ? -b : b)), static_cast<uint64_t>(c)) != 1) continue;
            ++h;
        }
    return h;
}

uint32_t UnitCount(int64_t d) { return d == -3 ? 6u : d == -4 ? 4u : 2u; }

int Kronecker(int64_t d, uint64_t n) {
    ENGINE_CHECK_MSG(n >= 1, "Kronecker: n >= 1");
    int s = 1;
    uint64_t m = n;
    for (uint64_t p = 2; m > 1; ++p) {
        if (p * p > m) p = m;
        while (m % p == 0) {
            m /= p;
            int v;
            if (p == 2) {
                const int64_t r8 = ((d % 8) + 8) % 8;
                v = (r8 % 2 == 0) ? 0 : (r8 == 1 || r8 == 7) ? 1 : -1;
            } else {
                const uint64_t a = ModS(d, p);
                if (a == 0) v = 0;
                else v = PowModU(a, (p - 1) / 2, p) == 1 ? 1 : -1;
            }
            s *= v;
            if (s == 0) return 0;
        }
    }
    return s;
}

std::string Splitting(int64_t d, uint64_t p) {
    const int k = Kronecker(d, p);
    return k > 0 ? "split" : k < 0 ? "inert" : "ramified";
}

bool PrincipalPrimeTrace(int64_t d, uint64_t p, int64_t& trace) {
    // N(x + y w) = x^2 + d x y + c y^2, c = (d^2 - d)/4, w = (d + sqrt d)/2, Tr = 2x + d y.
    const int64_t P = static_cast<int64_t>(p);
    const int64_t Y = static_cast<int64_t>(std::sqrt(4.0L * P / static_cast<long double>(-d))) + 1;
    for (int64_t y = 0; y <= Y; ++y) {
        // x = (-d y +- sqrt(d y^2 + 4p)) / 2
        bool ok = false;
        const int64_t disc = d * y * y + 4 * P;
        const int64_t r = ISqrtExact(disc, ok);
        if (!ok) continue;
        for (int sg : {1, -1}) {
            const int64_t num = -d * y + sg * r;
            if (num % 2 != 0) continue;
            const int64_t x = num / 2;
            const int64_t c = (d * d - d) / 4;
            if (x * x + d * x * y + c * y * y != P) continue;
            if (y == 0) continue;   // a rational integer of norm p: impossible (p prime)
            trace = 2 * x + d * y;
            return true;
        }
    }
    return false;
}

namespace {
// The trace recurrence s_j = t s_{j-1} - p s_{j-2}, s_0 = 2, s_1 = t, at j = k - 1.
void CmPrecheck(int64_t d, uint32_t k, uint64_t p, int& kr, int64_t& t) {
    ENGINE_CHECK_MSG(d < 0 && IsFundamentalDiscriminant(d), "CM oracle: d must be a negative fundamental discriminant");
    ENGINE_CHECK_MSG(k >= 2 && IsPrimeSmall(p), "CM oracle: k >= 2 and p prime");
    if ((k - 1) % UnitCount(d) != 0)
        throw std::runtime_error("cm_lambda_undefined: w_K = " + std::to_string(UnitCount(d)) + " does not divide k - 1 = " +
                                 std::to_string(k - 1) + " (lambda_k((alpha)) = alpha^{k-1} is not well defined)");
    kr = Kronecker(d, p);
    if (kr == 0) throw std::runtime_error("cm_ramified: p = " + std::to_string(p) + " divides d = " + std::to_string(d));
    t = 0;
    if (kr > 0 && !PrincipalPrimeTrace(d, p, t))
        throw std::runtime_error("cm_nonprincipal: the primes above p = " + std::to_string(p) + " are not principal in Q(sqrt " +
                                 std::to_string(d) + ") (h_K = " + std::to_string(ClassNumber(d)) +
                                 "; lambda_k is defined on principal ideals only -- [U] prop:u-cmeis(2), referee U-1)");
}
} // namespace

__int128 CmCoefficientExact(int64_t d, uint32_t k, uint64_t p) {
    int kr = 0;
    int64_t t = 0;
    CmPrecheck(d, k, p, kr, t);
    if (kr < 0) return 0;
    // |s_j| <= 2 p^{j/2}: require p^{(k-1)/2} < 2^110.
    ENGINE_CHECK_MSG(static_cast<double>(k - 1) * std::log2(static_cast<double>(p)) < 220.0,
                     "CmCoefficientExact: p^{(k-1)/2} too large for __int128");
    __int128 s0 = 2, s1 = t;
    for (uint32_t j = 2; j <= k - 1; ++j) {
        const __int128 s2 = static_cast<__int128>(t) * s1 - static_cast<__int128>(p) * s0;
        s0 = s1;
        s1 = s2;
    }
    return k - 1 == 0 ? s0 : s1;
}

uint64_t CmCoefficientMod(int64_t d, uint32_t k, uint64_t p, uint64_t m) {
    ENGINE_CHECK_MSG(m >= 2 && m < (1ull << 62), "CmCoefficientMod: modulus in [2, 2^62)");
    int kr = 0;
    int64_t t = 0;
    CmPrecheck(d, k, p, kr, t);
    if (kr < 0) return 0;
    const unsigned __int128 M = m;
    const unsigned __int128 tm = ModS(t, m), pm = p % m;
    unsigned __int128 s0 = 2 % m, s1 = tm;
    for (uint32_t j = 2; j <= k - 1; ++j) {
        const unsigned __int128 s2 = (tm * s1 % M + (M - pm * s0 % M)) % M;
        s0 = s1;
        s1 = s2;
    }
    return static_cast<uint64_t>(s1);
}

hecke::neb::DirichletCharacter KroneckerCharacter(int64_t d, uint64_t N) {
    const uint64_t a = static_cast<uint64_t>(-d);
    ENGINE_CHECK_MSG(d < 0 && IsFundamentalDiscriminant(d), "KroneckerCharacter: d a negative fundamental discriminant");
    ENGINE_CHECK_MSG(N >= 1 && N % a == 0 && N < (1ull << 20), "KroneckerCharacter: |d| must divide N (< 2^20)");
    hecke::neb::DirichletCharacter c;
    c.N = N;
    c.conrey = 0;
    c.L = 2;
    c.expo.assign(N, -1);
    for (uint64_t u : hecke::neb::UnitsMod(N)) {
        const int s = Kronecker(d, u);
        ENGINE_CHECK_MSG(s != 0, "KroneckerCharacter: (d / u) = 0 at a unit u");
        c.expo[u] = s > 0 ? 0 : 1;
    }
    return c;
}

// ---- linear algebra -----------------------------------------------------------------
FqMat Kernel(const Fq& F, const FqMat& A) {
    FqMat R = A;
    const std::vector<uint32_t> piv = FqRref(F, R);
    std::vector<char> is_piv(A.cols, 0);
    for (uint32_t c : piv) is_piv[c] = 1;
    std::vector<uint32_t> fr;
    for (uint32_t c = 0; c < A.cols; ++c)
        if (!is_piv[c]) fr.push_back(c);
    FqMat K(A.cols, static_cast<uint32_t>(fr.size()));
    for (uint32_t j = 0; j < fr.size(); ++j) {
        K.at(fr[j], j) = 1;
        for (uint32_t i = 0; i < piv.size(); ++i) K.at(piv[i], j) = F.neg(R.at(i, fr[j]));
    }
    return K;
}

FqMat MatPow(const Fq& F, const FqMat& A, uint64_t e) {
    ENGINE_CHECK_MSG(A.rows == A.cols, "MatPow: square");
    FqMat R = hecke::neb::FqIdentity(A.rows), B = A;
    while (e) {
        if (e & 1) R = hecke::neb::FqMul(F, R, B);
        e >>= 1;
        if (e) B = hecke::neb::FqMul(F, B, B);
    }
    return R;
}

FqMat Shift(const Fq& F, const FqMat& A, uint32_t s) {
    ENGINE_CHECK_MSG(A.rows == A.cols, "Shift: square");
    FqMat R = A;
    for (uint32_t i = 0; i < A.rows; ++i) R.at(i, i) = F.sub(R.at(i, i), s);
    return R;
}

FqMat JointGeneralisedEigenspace(const Fq& F, const std::vector<FqMat>& ops, const std::vector<uint32_t>& scal) {
    ENGINE_CHECK_MSG(!ops.empty() && ops.size() == scal.size(), "JointGeneralisedEigenspace: one scalar per operator");
    const uint32_t h = ops[0].rows;
    if (h == 0) return FqMat(0, 0);
    FqMat S(static_cast<uint32_t>(ops.size()) * h, h);
    for (size_t i = 0; i < ops.size(); ++i) {
        ENGINE_CHECK_MSG(ops[i].rows == h && ops[i].cols == h, "JointGeneralisedEigenspace: operators of one size");
        const FqMat P = MatPow(F, Shift(F, ops[i], scal[i]), h);
        for (uint32_t r = 0; r < h; ++r)
            for (uint32_t c = 0; c < h; ++c) S.at(static_cast<uint32_t>(i) * h + r, c) = P.at(r, c);
    }
    return Kernel(F, S);
}

FqMat Restrict(const Fq& F, const FqMat& T, const FqMat& B) {
    if (B.cols == 0) return FqMat(0, 0);
    return hecke::neb::FqSolve(F, B, hecke::neb::FqMul(F, T, B));
}

uint32_t GeneralisedEigenDim(const Fq& F, const FqMat& A, uint32_t s) {
    if (A.rows == 0) return 0;
    return Kernel(F, MatPow(F, Shift(F, A, s), A.rows)).cols;
}

LoewyReading LoewyOfNilpotents(const Fq& F, const std::vector<FqMat>& nil, uint32_t w) {
    LoewyReading out;
    out.loewy = 1;
    out.dim_A = w == 0 ? 0 : 1;
    if (w == 0) return out;
    auto flatten = [&](const std::vector<FqMat>& ms) {
        FqMat V(static_cast<uint32_t>(ms.size()), w * w);
        for (uint32_t i = 0; i < ms.size(); ++i)
            for (uint32_t j = 0; j < w * w; ++j) V.at(i, j) = ms[i].a[j];
        return V;
    };
    auto basis_of = [&](const std::vector<FqMat>& ms) {
        std::vector<FqMat> b;
        if (ms.empty()) return b;
        FqMat V = flatten(ms);
        const std::vector<uint32_t> piv = FqRref(F, V);
        for (uint32_t i = 0; i < piv.size(); ++i) {
            FqMat M(w, w);
            for (uint32_t j = 0; j < w * w; ++j) M.a[j] = V.at(i, j);
            b.push_back(M);
        }
        return b;
    };
    std::vector<FqMat> all{hecke::neb::FqIdentity(w)};
    std::vector<FqMat> layer{hecke::neb::FqIdentity(w)};
    std::vector<std::vector<FqMat>> by_degree;   // by_degree[j-1]: a basis of the span of the degree-j monomials
    for (uint32_t j = 1; j <= w + 1; ++j) {
        std::vector<FqMat> prod;
        for (const FqMat& N : nil)
            for (const FqMat& M : layer) prod.push_back(hecke::neb::FqMul(F, N, M));
        layer = basis_of(prod);
        if (layer.empty()) { out.loewy = j; break; }
        out.monomial_degree_dims.push_back(static_cast<uint32_t>(layer.size()));
        by_degree.push_back(layer);
        for (const FqMat& M : layer) all.push_back(M);
        ENGINE_CHECK_MSG(j <= w, "LoewyOfNilpotents: the operators are not nilpotent");
    }
    out.dim_A = static_cast<uint32_t>(basis_of(all).size());
    // rad = the span of the monomials of degree >= 1 (the N_i commute and are nilpotent), rad^j = degree >= j.
    for (size_t j = 0; j < by_degree.size(); ++j) {
        std::vector<FqMat> tail;
        for (size_t i = j; i < by_degree.size(); ++i) tail.insert(tail.end(), by_degree[i].begin(), by_degree[i].end());
        out.radical_power_dims.push_back(static_cast<uint32_t>(basis_of(tail).size()));
    }
    uint32_t prev = out.dim_A;
    for (uint32_t r : out.radical_power_dims) { out.loewy_layers.push_back(prev - r); prev = r; }
    out.loewy_layers.push_back(prev);   // rad^{loewy-1} / rad^{loewy} = rad^{loewy-1} (= A when loewy = 1)
    return out;
}

} // namespace hidacm
} // namespace hecke_ext
