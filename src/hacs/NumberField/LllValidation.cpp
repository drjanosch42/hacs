// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// NumberFieldLllValidation (fast) -- the gate of the in-house LLL and POLRED
// (hecke-extensions slot 21, T1; plan docs/implementation/slot21/00-slot21-plan.md
// "Talk"; Cohen GTM 138 sec. 2.6 and Alg. 4.4.11 / 4.4.12, the accepted
// substitute for LLL 1982, missing_literature.md LIT-09).
//
// Every oracle is independent of Lll.cpp / Polred.cpp: the Gram--Schmidt
// reducedness test, the determinant, the matrix products, the brute-force
// shortest vector and the expected polynomials are computed here from scratch
// or are known values:
//   (1) known lattices: a scrambled Z^n (lambda_1 = 1), a scrambled E_8
//       (lambda_1 = 2), A_2 at delta = 1 (lambda_1 = 2), the integer relations
//       of 2^{1/3} (x^3 - 2) and sqrt 2 + sqrt 3 (x^4 - 10 x^2 + 1) from
//       decimal approximations;
//   (2) random lattices, n = 2..6, three deltas: H unimodular (own rational
//       determinant), the reduced Gram = H G H^T (own product), Definition
//       2.6.1 by the own Gram--Schmidt, det G preserved, the rational
//       Alg. 2.6.3 and the integral Alg. 2.6.7 give the same H (also on G/q:
//       scale invariance), brute-force lambda_1 against Theorem 2.6.2's bound
//       |b_1|^2 <= (1/(delta - 1/4))^{n-1} lambda_1, and |b_1|^2 = lambda_1 for
//       n = 2, delta = 1 (Lagrange--Gauss);
//   (3) POLRED: charpoly / disc against known values; Q(zeta_7)^+ from
//       f(x) = g(x - 5) -> x^3 - x^2 - 2x + 1 (disc 49); Q(zeta_37)^+ (degree
//       18, exact trace form) -> Psi_37(-x), disc 37^17, Psi from the Dickson
//       recursion here; Q(zeta_l), l = 7, 11, 37 (degree 36), exact T2 on
//       Z[zeta + 1] -> Phi_l(-x), disc (-1)^{(l-1)/2} l^{l-2}, lambda_1 = l - 1;
//   (4) negative controls: dependent rows, an indefinite Gram matrix, delta
//       outside (1/4, 1], a tampered H, non-reduced Gram matrices;
//   (5) LLL_DIGEST: FNV-1a-64 of every H above (a fingerprint for the
//       -DNF_NO_INT128 arm and the aarch64 build, not asserted).
// Read-only, no golden object, no env.
#include "LllValidation.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../../ModArithmetic/Fnv.h"
#include "BigInt.h"
#include "BigRational.h"
#include "Lll.h"
#include "Polred.h"

namespace {

using nf::BigInt;
using nf::BigRational;
using nf::IntMat;
using nf::IntPoly;
using nf::RatMat;

uint32_t g_checks = 0;
uint32_t g_fail = 0;
uint64_t g_digest = engine::kFnvOffset64;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  [FAIL] " << what << std::endl;
    }
}

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void Digest(const IntMat& H) {
    for (const auto& r : H)
        for (const auto& x : r) g_digest = engine::FnvBytes(g_digest, x.to_string() + ",");
    g_digest = engine::FnvBytes(g_digest, ";");
}

std::string PolyStr(const IntPoly& p) {
    std::ostringstream os;
    for (std::size_t i = p.size(); i-- > 0;) os << p[i].to_string() << (i ? " " : "");
    return os.str();
}

bool RatLess(const BigRational& a, const BigRational& b) { return a.num() * b.den() < b.num() * a.den(); }

// ---------------------------------------------------------------- own oracles
IntMat Mul(const IntMat& A, const IntMat& B) {
    IntMat C(A.size(), std::vector<BigInt>(B[0].size(), BigInt(0)));
    for (std::size_t i = 0; i < A.size(); ++i)
        for (std::size_t j = 0; j < B[0].size(); ++j)
            for (std::size_t t = 0; t < B.size(); ++t) C[i][j] += A[i][t] * B[t][j];
    return C;
}
IntMat Transpose(const IntMat& A) {
    IntMat T(A[0].size(), std::vector<BigInt>(A.size()));
    for (std::size_t i = 0; i < A.size(); ++i)
        for (std::size_t j = 0; j < A[0].size(); ++j) T[j][i] = A[i][j];
    return T;
}
RatMat ToRat(const IntMat& A) {
    RatMat R(A.size());
    for (std::size_t i = 0; i < A.size(); ++i)
        for (const auto& x : A[i]) R[i].push_back(BigRational(x));
    return R;
}

// Determinant by rational Gaussian elimination (not Bareiss).
BigRational DetRat(RatMat A) {
    const std::size_t n = A.size();
    BigRational det(1);
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t p = c;
        while (p < n && A[p][c].is_zero()) ++p;
        if (p == n) return BigRational(0);
        if (p != c) {
            std::swap(A[p], A[c]);
            det = -det;
        }
        det *= A[c][c];
        for (std::size_t i = c + 1; i < n; ++i) {
            if (A[i][c].is_zero()) continue;
            const BigRational f = A[i][c] / A[c][c];
            for (std::size_t j = c; j < n; ++j) A[i][j] -= f * A[c][j];
        }
    }
    return det;
}

RatMat InverseRat(RatMat A) {
    const std::size_t n = A.size();
    RatMat I(n, std::vector<BigRational>(n, BigRational(0)));
    for (std::size_t i = 0; i < n; ++i) I[i][i] = BigRational(1);
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t p = c;
        while (A[p][c].is_zero()) ++p;
        std::swap(A[p], A[c]);
        std::swap(I[p], I[c]);
        const BigRational inv = BigRational(1) / A[c][c];
        for (std::size_t j = 0; j < n; ++j) {
            A[c][j] *= inv;
            I[c][j] *= inv;
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (i == c || A[i][c].is_zero()) continue;
            const BigRational f = A[i][c];
            for (std::size_t j = 0; j < n; ++j) {
                A[i][j] -= f * A[c][j];
                I[i][j] -= f * I[c][j];
            }
        }
    }
    return I;
}

// Definition 2.6.1 by the textbook Gram--Schmidt on b_i b_j: B_i = det_i / det_{i-1}
// (leading minors) and mu_ij from the triangular solve -- written independently
// of Lll.cpp's recursion.
bool OracleReduced(const RatMat& G, const BigRational& delta) {
    const std::size_t n = G.size();
    std::vector<BigRational> B(n);
    RatMat mu(n, std::vector<BigRational>(n, BigRational(0)));
    // b*_i = b_i - sum_{j<i} mu_ij b*_j; coefficients of b*_i in the b basis: c[i][.]
    RatMat c(n, std::vector<BigRational>(n, BigRational(0)));
    auto inner = [&](const std::vector<BigRational>& x, const std::vector<BigRational>& y) {
        BigRational s(0);
        for (std::size_t a = 0; a < n; ++a) {
            if (x[a].is_zero()) continue;
            for (std::size_t b = 0; b < n; ++b)
                if (!y[b].is_zero()) s += x[a] * G[a][b] * y[b];
        }
        return s;
    };
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<BigRational> e(n, BigRational(0));
        e[i] = BigRational(1);
        c[i] = e;
        for (std::size_t j = 0; j < i; ++j) {
            mu[i][j] = inner(e, c[j]) / B[j];
            for (std::size_t t = 0; t < n; ++t) c[i][t] -= mu[i][j] * c[j][t];
        }
        B[i] = inner(c[i], c[i]);
        if (B[i].num().sign() <= 0) return false;
    }
    const BigRational half(BigInt(1), BigInt(2));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) {
            const BigRational m = mu[i][j].num().is_neg() ? -mu[i][j] : mu[i][j];
            if (RatLess(half, m)) return false;
        }
    for (std::size_t i = 1; i < n; ++i)
        if (RatLess(B[i], (delta - mu[i][i - 1] * mu[i][i - 1]) * B[i - 1])) return false;
    return true;
}

// lambda_1 of the positive definite integral G by enumeration of the box
// |x_i|^2 <= R (G^{-1})_{ii}, R = G_00 (an upper bound for lambda_1).
BigInt BruteLambda1(const IntMat& G) {
    const std::size_t n = G.size();
    const RatMat Gi = InverseRat(ToRat(G));
    const BigRational R(G[0][0]);
    std::vector<int64_t> bound(n);
    for (std::size_t i = 0; i < n; ++i) {
        const BigRational r = R * Gi[i][i];
        const int64_t fl = (r.num() / r.den()).to_int64();
        int64_t s = 0;
        while ((s + 1) * (s + 1) <= fl) ++s;
        bound[i] = s;
    }
    BigInt best = G[0][0];
    std::vector<int64_t> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = -bound[i];
    while (true) {
        bool nonzero = false;
        for (auto v : x) nonzero = nonzero || v != 0;
        if (nonzero) {
            BigInt q(0);
            for (std::size_t a = 0; a < n; ++a)
                for (std::size_t b = 0; b < n; ++b) q += BigInt(x[a] * x[b]) * G[a][b];
            if (q < best) best = q;
        }
        std::size_t i = 0;
        while (i < n && x[i] == bound[i]) {
            x[i] = -bound[i];
            ++i;
        }
        if (i == n) break;
        ++x[i];
    }
    return best;
}

// A random unimodular n x n matrix: a product of elementary row operations and swaps.
IntMat RandomUnimodular(std::size_t n, std::mt19937_64& rng, int steps, int maxq) {
    IntMat U(n, std::vector<BigInt>(n, BigInt(0)));
    for (std::size_t i = 0; i < n; ++i) U[i][i] = BigInt(1);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    std::uniform_int_distribution<int> q(-maxq, maxq);
    for (int s = 0; s < steps; ++s) {
        const std::size_t a = pick(rng), b = pick(rng);
        if (a == b) continue;
        const BigInt c(q(rng));
        for (std::size_t j = 0; j < n; ++j) U[a][j] += c * U[b][j];
        if (s % 7 == 3) std::swap(U[a], U[b]);
    }
    return U;
}

// The common block: H unimodular, the reduced Gram, Definition 2.6.1, det G.
void CheckResult(const IntMat& G, const nf::LllResult& R, const BigRational& delta, const std::string& tag) {
    const RatMat Hq = ToRat(R.H);
    const BigRational dH = DetRat(Hq);
    Check(dH == BigRational(1) || dH == BigRational(-1), tag + ": det H = +-1");
    const IntMat Gr = Mul(Mul(R.H, G), Transpose(R.H));
    Check(Gr == nf::transform_gram(R.H, G), tag + ": transform_gram == H G H^T");
    Check(OracleReduced(ToRat(Gr), delta), tag + ": LLL-reduced (Definition 2.6.1, own Gram-Schmidt)");
    std::string why;
    Check(nf::lll_is_reduced(Gr, delta, &why), tag + ": lll_is_reduced " + why);
    Check(DetRat(ToRat(Gr)) == DetRat(ToRat(G)), tag + ": det G preserved");
    Check(BigRational(Gr[0][0]) == R.B[0], tag + ": B_1 = |b_1|^2");
    BigRational prod(1);
    for (const auto& b : R.B) prod *= b;
    Check(prod == DetRat(ToRat(G)), tag + ": prod B_i = det G");
    Digest(R.H);
}

// ---------------------------------------------------------------- (1)
BigInt IRoot(const BigInt& a, unsigned k) {  // floor(a^{1/k}), a >= 0
    BigInt lo(0), hi = BigInt::pow(BigInt(2), a.bit_length() / k + 1);
    while (hi - lo > BigInt(1)) {
        const BigInt mid = (lo + hi) / BigInt(2);
        if (BigInt::pow(mid, k) <= a) lo = mid;
        else hi = mid;
    }
    return lo;
}

void KnownLattices() {
    std::mt19937_64 rng(0x7131u);
    const BigRational d99(BigInt(99), BigInt(100));
    // Z^n scrambled.
    for (std::size_t n : {6u, 8u}) {
        const IntMat U = RandomUnimodular(n, rng, 60, 3);
        const IntMat G = Mul(U, Transpose(U));
        const auto R = nf::lll_gram_integral(G, d99);
        const std::string tag = "Z^" + std::to_string(n);
        CheckResult(G, R, d99, tag);
        const IntMat Gr = nf::transform_gram(R.H, G);
        Check(Gr[0][0] == BigInt(1), tag + ": |b_1|^2 = lambda_1 = 1");
        std::cout << "  " << tag << " max|U_ij| bits=" << [&] { std::size_t b = 0; for (auto& r : U) for (auto& x : r) b = std::max(b, x.bit_length()); return b; }()
                  << " swaps=" << R.swaps << " |b_1|^2=" << Gr[0][0] << std::endl;
    }
    // E_8 (Cartan matrix, the even unimodular lattice; minimum 2), scrambled.
    {
        IntMat C(8, std::vector<BigInt>(8, BigInt(0)));
        for (std::size_t i = 0; i < 8; ++i) C[i][i] = BigInt(2);
        // Bourbaki labelling: chain 1-3-4-5-6-7-8, 2 attached to 4.
        auto edge = [&](std::size_t a, std::size_t b) { C[a - 1][b - 1] = C[b - 1][a - 1] = BigInt(-1); };
        edge(1, 3); edge(3, 4); edge(4, 5); edge(5, 6); edge(6, 7); edge(7, 8); edge(2, 4);
        Check(DetRat(ToRat(C)) == BigRational(1), "E8: Cartan determinant 1");
        const IntMat U = RandomUnimodular(8, rng, 80, 4);
        const IntMat G = Mul(Mul(U, C), Transpose(U));
        const auto R = nf::lll_gram_integral(G, d99);
        CheckResult(G, R, d99, "E8");
        const IntMat Gr = nf::transform_gram(R.H, G);
        Check(Gr[0][0] == BigInt(2), "E8: |b_1|^2 = lambda_1 = 2");
        bool even = true;
        for (std::size_t i = 0; i < 8; ++i) even = even && Gr[i][i].mod_u32(2) == 0;
        Check(even, "E8: reduced basis even");
        std::cout << "  E8 swaps=" << R.swaps << " G_00 before=" << G[0][0] << " after=" << Gr[0][0] << std::endl;
    }
    // A_2 at delta = 1: Lagrange-Gauss, |b_1|^2 = lambda_1 = 2.
    {
        const IntMat A2 = {{BigInt(2), BigInt(-1)}, {BigInt(-1), BigInt(2)}};
        const IntMat U = {{BigInt(17), BigInt(5)}, {BigInt(10), BigInt(3)}};  // det 1
        const IntMat G = Mul(Mul(U, A2), Transpose(U));
        const auto R = nf::lll_gram_integral(G, BigRational(1));
        CheckResult(G, R, BigRational(1), "A2");
        Check(nf::transform_gram(R.H, G)[0][0] == BigInt(2), "A2: |b_1|^2 = 2");
    }
    // Integer relations. Rows e_i (+) a_i, a_i ~ 10^N alpha^i.
    auto relation = [&](const std::vector<BigInt>& a, const std::vector<int64_t>& want, const std::string& tag) {
        const std::size_t k = a.size();
        IntMat rows(k, std::vector<BigInt>(k + 1, BigInt(0)));
        for (std::size_t i = 0; i < k; ++i) {
            rows[i][i] = BigInt(1);
            rows[i][k] = a[i];
        }
        const auto R = nf::lll_basis(rows);
        CheckResult(nf::gram_of_rows(rows), R.lll, nf::lll_default_delta(), tag);
        Check(R.basis == Mul(R.lll.H, rows), tag + ": basis = H rows");
        bool plus = true, minus = true;
        for (std::size_t i = 0; i < k; ++i) {
            plus = plus && R.basis[0][i] == BigInt(want[i]);
            minus = minus && R.basis[0][i] == BigInt(-want[i]);
        }
        Check(plus || minus, tag + ": b_1 is the minimal-polynomial relation");
        std::ostringstream os;
        for (std::size_t i = 0; i <= k; ++i) os << R.basis[0][i] << (i < k ? "," : "");
        std::cout << "  " << tag << " b_1=(" << os.str() << ")" << std::endl;
    };
    {
        const BigInt T = BigInt::pow(BigInt(10), 30);  // 2^{i/3} 10^30 = cbrt(2^i 10^90)
        std::vector<BigInt> a;
        for (unsigned i = 0; i < 4; ++i) a.push_back(IRoot(BigInt::pow(BigInt(2), i) * BigInt::pow(T, 3), 3));
        Check(a[3] == BigInt(2) * T, "cbrt: a_3 = 2 10^30 exactly");
        relation(a, {-2, 0, 0, 1}, "relation 2^(1/3)");
    }
    {
        const BigInt T = BigInt::pow(BigInt(10), 40), T2 = T * T;
        auto rt = [&](int64_t m) { return IRoot(BigInt(m) * T2, 2); };  // floor(sqrt m 10^40)
        // alpha = sqrt2 + sqrt3: alpha^2 = 5 + 2 sqrt6, alpha^3 = 11 sqrt2 + 9 sqrt3, alpha^4 = 49 + 20 sqrt6 = 49 + sqrt 2400.
        const std::vector<BigInt> a = {T, rt(2) + rt(3), BigInt(5) * T + rt(24), rt(242) + rt(243), BigInt(49) * T + rt(2400)};
        relation(a, {1, 0, -10, 0, 1}, "relation sqrt2+sqrt3");
    }
}

// ---------------------------------------------------------------- (2)
void RandomLattices() {
    std::mt19937_64 rng(0x2026100521ull);
    const std::vector<BigRational> deltas = {nf::lll_default_delta(), BigRational(BigInt(99), BigInt(100)), BigRational(1)};
    uint32_t cases = 0;
    for (std::size_t n = 2; n <= 6; ++n) {
        for (int rep = 0; rep < 8; ++rep) {
            const BigRational& delta = deltas[static_cast<std::size_t>(rep) % deltas.size()];
            // Random rows in Z^{n+1}, entries up to 2^bits, made independent by retrying.
            std::uniform_int_distribution<int64_t> e(-(int64_t(1) << (8 + 2 * rep)), int64_t(1) << (8 + 2 * rep));
            IntMat rows;
            IntMat G;
            do {
                rows.assign(n, std::vector<BigInt>(n + 1));
                for (auto& r : rows)
                    for (auto& x : r) x = BigInt(e(rng));
                G = Mul(rows, Transpose(rows));
            } while (DetRat(ToRat(G)).is_zero());
            const std::string tag = "random n=" + std::to_string(n) + " rep=" + std::to_string(rep) + " delta=" + delta.to_string();
            const auto Ri = nf::lll_gram_integral(G, delta);
            CheckResult(G, Ri, delta, tag);
            const auto Rr = nf::lll_gram_rational(ToRat(G), delta);
            Check(Rr.H == Ri.H && Rr.swaps == Ri.swaps && Rr.reductions == Ri.reductions, tag + ": Alg. 2.6.3 == Alg. 2.6.7");
            Check(Rr.B == Ri.B, tag + ": same B_i");
            // scale invariance on G / q (a rational Gram matrix)
            RatMat Gq = ToRat(G);
            for (auto& r : Gq)
                for (auto& x : r) x /= BigRational(BigInt(7 + rep));
            Check(nf::lll_gram_rational(Gq, delta).H == Ri.H, tag + ": same H on G/q");
            // brute force lambda_1 on the reduced Gram matrix
            const IntMat Gr = nf::transform_gram(Ri.H, G);
            const BigInt l1 = BruteLambda1(Gr);
            // |b_1|^2 <= alpha^{n-1} lambda_1, alpha = 1/(delta - 1/4) = 4 den / (4 num - den)
            const BigInt an = BigInt(4) * delta.den(), ad = BigInt(4) * delta.num() - delta.den();
            Check(Gr[0][0] * BigInt::pow(ad, n - 1) <= l1 * BigInt::pow(an, n - 1), tag + ": Theorem 2.6.2 bound");
            Check(l1 <= Gr[0][0], tag + ": lambda_1 <= |b_1|^2");
            if (n == 2 && delta == BigRational(1)) Check(l1 == Gr[0][0], tag + ": |b_1|^2 = lambda_1 (Gauss)");
            // the basis route agrees with the Gram route
            const auto Rb = nf::lll_basis(rows, delta);
            Check(Rb.lll.H == Ri.H && Rb.basis == Mul(Ri.H, rows), tag + ": lll_basis");
            ++cases;
        }
    }
    std::cout << "  random lattices: " << cases << " cases" << std::endl;
}

// ---------------------------------------------------------------- (3)
IntPoly Shift(const IntPoly& f, int64_t c) {  // f(x + c)
    IntPoly r(f.size(), BigInt(0));
    // Horner: r = (...((a_n)(x+c) + a_{n-1})(x+c) + ...)
    for (std::size_t i = f.size(); i-- > 0;) {
        IntPoly t(r.size(), BigInt(0));
        for (std::size_t j = 0; j + 1 < r.size(); ++j) {
            t[j + 1] += r[j];
            t[j] += BigInt(c) * r[j];
        }
        t[0] += f[i];
        r = t;
    }
    return r;
}
IntPoly Negx(IntPoly f) {  // (-1)^n f(-x)
    const std::size_t n = f.size() - 1;
    for (std::size_t j = 0; j <= n; ++j)
        if ((n - j) % 2 == 1) f[j] = -f[j];
    return f;
}
IntPoly Cyclo(unsigned l) { return IntPoly(l, BigInt(1)); }
IntPoly PsiPlus(unsigned l) {  // minpoly of zeta + zeta^{-1}: 1 + sum_{k=1}^{m} D_k, D_0 = 2, D_1 = x, D_{k+1} = x D_k - D_{k-1}
    const unsigned m = (l - 1) / 2;
    std::vector<IntPoly> D(m + 1, IntPoly(m + 1, BigInt(0)));
    D[0][0] = BigInt(2);
    D[1][1] = BigInt(1);
    for (unsigned k = 1; k < m; ++k)
        for (unsigned j = 0; j <= m; ++j) D[k + 1][j] = (j ? D[k][j - 1] : BigInt(0)) - D[k - 1][j];
    IntPoly P(m + 1, BigInt(0));
    P[0] = BigInt(1);
    for (unsigned k = 1; k <= m; ++k)
        for (unsigned j = 0; j <= m; ++j) P[j] += D[k][j];
    return P;
}

void Polred() {
    // charpoly / disc against known values
    const IntPoly x3m2 = {BigInt(-2), BigInt(0), BigInt(0), BigInt(1)};
    Check(nf::poly_discriminant(x3m2) == BigInt(-108), "disc(x^3 - 2) = -108");
    Check(nf::poly_discriminant({BigInt(1), BigInt(0), BigInt(1)}) == BigInt(-4), "disc(x^2 + 1) = -4");
    Check(nf::charpoly_element(x3m2, {BigInt(0), BigInt(1)}) == x3m2, "charpoly(alpha) = f");
    // charpoly of alpha^2 = cbrt 4: y^3 - 4; of (1 + alpha)/1: (y - 1)^3 - 2
    Check(nf::charpoly_element(x3m2, {BigInt(0), BigInt(0), BigInt(1)}) == IntPoly{BigInt(-4), BigInt(0), BigInt(0), BigInt(1)},
          "charpoly(alpha^2) = y^3 - 4");
    Check(nf::charpoly_element(x3m2, {BigInt(1), BigInt(1)}) == IntPoly{BigInt(-3), BigInt(3), BigInt(-3), BigInt(1)},
          "charpoly(1 + alpha) = (y-1)^3 - 2");
    // (1 + i)/2 is not integral: charpoly y^2 - y + 1/2 -> refused
    bool threw = false;
    try {
        nf::charpoly_element({BigInt(1), BigInt(0), BigInt(1)}, {BigInt(1), BigInt(1)}, BigInt(2));
    } catch (const std::logic_error&) {
        threw = true;
    }
    Check(threw, "charpoly of a non-integral element refused");
    Check(nf::charpoly_element({BigInt(1), BigInt(0), BigInt(1)}, {BigInt(1)}) == IntPoly{BigInt(1), BigInt(-2), BigInt(1)},
          "charpoly(1) = (y - 1)^2, disc 0");
    Check(nf::poly_discriminant(IntPoly{BigInt(1), BigInt(-2), BigInt(1)}).is_zero(), "disc((y-1)^2) = 0");
    // Totally real: Q(zeta_7)^+ and Q(zeta_37)^+ on the exact trace form of Z[alpha], alpha = (zeta + zeta^-1) + c.
    for (unsigned l : {7u, 37u}) {
        const IntPoly psi = PsiPlus(l);
        const int64_t c = l == 7 ? 5 : 3;
        const IntPoly f = Shift(psi, -c);  // root zeta + zeta^-1 + c
        const auto R = nf::polred_power_basis(f, nf::trace_form_gram(f));
        const std::string tag = "polred Q(zeta_" + std::to_string(l) + ")^+";
        CheckResult(nf::trace_form_gram(f), R.lll, nf::lll_default_delta(), tag);
        const IntPoly want = Negx(psi);
        Check(R.best >= 0 && R.best_poly() == want, tag + ": result Psi_l(-x)");
        const BigInt dK = BigInt::pow(BigInt(l), (l - 3) / 2);
        Check(R.best >= 0 && R.cands[static_cast<std::size_t>(R.best)].disc == dK, tag + ": disc = l^((l-3)/2)");
        Check(nf::poly_discriminant(f) == dK, tag + ": disc f = l^((l-3)/2) (Z[alpha] maximal)");
        std::cout << "  " << tag << " f: " << PolyStr(f).substr(0, 60) << "...  ->  " << PolyStr(R.best_poly()) << "  swaps=" << R.lll.swaps
                  << std::endl;
    }
    // CM: Q(zeta_l) on Z[zeta + 1], exact T2(x, y) = Tr(x ybar), Tr(zeta^m) = l - 1 (l | m) or -1.
    for (unsigned l : {7u, 11u, 37u}) {
        const std::size_t n = l - 1;
        const IntPoly f = Shift(Cyclo(l), -1);  // root zeta + 1
        // (zeta + 1)^i in the zeta basis: binom(i, k)
        IntMat W(n, std::vector<BigInt>(n, BigInt(0)));
        for (std::size_t i = 0; i < n; ++i) {
            W[i][0] = BigInt(1);
            for (std::size_t k = 1; k <= i; ++k) W[i][k] = (i - 1 >= k ? W[i - 1][k] : BigInt(0)) + W[i - 1][k - 1];
        }
        IntMat T(n, std::vector<BigInt>(n));
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = 0; b < n; ++b) T[a][b] = ((a + l - b) % l == 0) ? BigInt(int64_t(l) - 1) : BigInt(-1);
        const IntMat G = Mul(Mul(W, T), Transpose(W));
        const double t0 = Now();
        const auto R = nf::polred_power_basis(f, G);
        const std::string tag = "polred Q(zeta_" + std::to_string(l) + ")";
        CheckResult(G, R.lll, nf::lll_default_delta(), tag);
        Check(nf::transform_gram(R.lll.H, G)[0][0] == BigInt(int64_t(n)), tag + ": |b_1|^2 = lambda_1 = l - 1");
        Check(R.best >= 0 && R.best_poly() == Negx(Cyclo(l)), tag + ": result Phi_l(-x)");
        const BigInt dK = (((l - 1) / 2) % 2 ? BigInt(-1) : BigInt(1)) * BigInt::pow(BigInt(l), l - 2);
        Check(R.best >= 0 && R.cands[static_cast<std::size_t>(R.best)].disc == dK, tag + ": disc = (-1)^((l-1)/2) l^(l-2)");
        // the same order on a scrambled basis w = U (alpha^i): the general polred() route, with swaps
        std::mt19937_64 rng(l);
        const IntMat U = RandomUnimodular(n, rng, int(4 * n), 2);
        const IntMat GU = Mul(Mul(U, G), Transpose(U));
        const auto RU = nf::polred(f, U, std::vector<BigInt>(n, BigInt(1)), GU);
        CheckResult(GU, RU.lll, nf::lll_default_delta(), tag + " scrambled");
        Check(RU.best >= 0 && RU.best_poly() == Negx(Cyclo(l)), tag + " scrambled: result Phi_l(-x)");
        Check(nf::transform_gram(RU.lll.H, GU)[0][0] == BigInt(int64_t(n)), tag + " scrambled: |b_1|^2 = l - 1");
        // the candidate elements are integral, each candidate's size is its T2 (the Gram diagonal)
        bool sizes = true;
        const IntMat GUr = nf::transform_gram(RU.lll.H, GU);
        for (std::size_t i = 0; i < n; ++i) sizes = sizes && RU.cands[i].size == GUr[i][i] && RU.cands[i].den == BigInt(1);
        Check(sizes, tag + " scrambled: candidate sizes = reduced Gram diagonal");
        std::cout << "  " << tag << " degree " << n << " swaps=" << R.lll.swaps << " scrambled_swaps=" << RU.lll.swaps << " best=" << PolyStr(R.best_poly()).substr(0, 40)
                  << "... secs=" << Now() - t0 << std::endl;
    }
}

// ---------------------------------------------------------------- (4)
template <typename F>
bool Throws(F f) {
    try {
        f();
    } catch (const std::logic_error&) {
        return true;
    }
    return false;
}

void NegativeControls() {
    const IntMat dep = {{BigInt(1), BigInt(2), BigInt(3)}, {BigInt(2), BigInt(4), BigInt(6)}};
    Check(Throws([&] { nf::lll_basis(dep); }), "control: dependent rows refused (integral)");
    Check(Throws([&] { nf::lll_gram_rational(ToRat(nf::gram_of_rows(dep))); }), "control: dependent rows refused (rational)");
    const IntMat indef = {{BigInt(1), BigInt(2)}, {BigInt(2), BigInt(1)}};
    Check(Throws([&] { nf::lll_gram_integral(indef); }), "control: indefinite Gram refused (integral)");
    Check(Throws([&] { nf::lll_gram_rational(ToRat(indef)); }), "control: indefinite Gram refused (rational)");
    const IntMat I2 = {{BigInt(1), BigInt(0)}, {BigInt(0), BigInt(1)}};
    Check(Throws([&] { nf::lll_gram_integral(I2, BigRational(BigInt(1), BigInt(4))); }), "control: delta = 1/4 refused");
    Check(Throws([&] { nf::lll_gram_integral(I2, BigRational(BigInt(101), BigInt(100))); }), "control: delta > 1 refused");
    Check(Throws([&] { nf::lll_gram_integral({{BigInt(1), BigInt(0)}, {BigInt(1), BigInt(1)}}); }), "control: asymmetric refused");
    // non-reduced Gram matrices: Lovasz fails / size reduction fails
    const IntMat lov = {{BigInt(4), BigInt(0)}, {BigInt(0), BigInt(1)}};
    const IntMat siz = {{BigInt(1), BigInt(1)}, {BigInt(1), BigInt(5)}};
    std::string why;
    Check(!OracleReduced(ToRat(lov), nf::lll_default_delta()) && !nf::lll_is_reduced(lov, nf::lll_default_delta(), &why),
          "control: Lovasz violation detected (" + why + ")");
    Check(!OracleReduced(ToRat(siz), nf::lll_default_delta()) && !nf::lll_is_reduced(siz, nf::lll_default_delta(), &why),
          "control: size-reduction violation detected (" + why + ")");
    // and LLL repairs both
    Check(nf::transform_gram(nf::lll_gram_integral(lov).H, lov) == IntMat{{BigInt(1), BigInt(0)}, {BigInt(0), BigInt(4)}},
          "control: LLL swaps diag(4, 1) to diag(1, 4)");
    // a tampered H is caught by the oracles
    {
        std::mt19937_64 rng(99);
        const IntMat U = RandomUnimodular(4, rng, 30, 3);
        const IntMat G = Mul(U, Transpose(U));
        auto R = nf::lll_gram_integral(G);
        R.H[1][2] += BigInt(1);
        const BigRational dH = DetRat(ToRat(R.H));
        const IntMat Gr = Mul(Mul(R.H, G), Transpose(R.H));
        Check(!(dH == BigRational(1) || dH == BigRational(-1)) || !OracleReduced(ToRat(Gr), nf::lll_default_delta()),
              "control: tampered H fails det or reducedness");
    }
}

}  // namespace

uint32_t NumberFieldLllValidation() {
    g_checks = 0;
    g_fail = 0;
    g_digest = engine::kFnvOffset64;
    const double t0 = Now();
#if defined(__SIZEOF_INT128__) && !defined(NF_NO_INT128)
    const int int128 = 1;
#else
    const int int128 = 0;
#endif
    std::cout << "NumberFieldLllValidation (slot-21 T1; Cohen GTM 138 sec. 2.6, Alg. 4.4.11/4.4.12) int128_path=" << int128 << std::endl;
    double t = Now();
    KnownLattices();
    std::cout << "  (1) known lattices secs=" << Now() - t << std::endl;
    t = Now();
    RandomLattices();
    std::cout << "  (2) random lattices secs=" << Now() - t << std::endl;
    t = Now();
    Polred();
    std::cout << "  (3) polred secs=" << Now() - t << std::endl;
    t = Now();
    NegativeControls();
    std::cout << "  (4) controls secs=" << Now() - t << std::endl;
    std::ostringstream hx;
    hx << std::hex << g_digest;
    std::cout << "LLL_GATE checks=" << g_checks << " failing=" << g_fail << " digest=0x" << hx.str() << " secs=" << Now() - t0
              << std::endl;
    return g_fail;
}
