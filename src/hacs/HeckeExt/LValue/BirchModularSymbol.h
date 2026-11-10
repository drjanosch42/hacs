// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_BIRCH_MODULAR_SYMBOL_H
#define HECKE_EXT_LVALUE_BIRCH_MODULAR_SYMBOL_H

// ===========================================================================
// (hecke-extensions slot 19, WP-LIN) THE LINEAR KIND OF THE L-VALUE INTERFACE:
// Birch / Manin-Stevens modular symbols on [C]'s 7-vertex model.  [E] def:lvroute,
// rmk:lvshapes (B), sec:lvnorm row (B); spec docs/implementation/slot19/R19-8_report.md
// Sec. 6 with its Errata and docs/implementation/slot19/isign_resolution.md;
// predictions docs/implementation/slot19/wplin/predictions.md (committed first).
//
// THE DEFINITION (R19-8 Sec. 1.1, PROVED there, gated here as G1/G2).  A class of
// H^1_c = H^1(Cone(r)) of the C++ mapping cone is a cone cocycle (c, c_d): c an
// equivariant 1-cochain of K, c_d a 0-cochain of the circle model v0-v1-v2-v3,
// D(c, c_d) = (-dc, dc_d + rc) = 0 (ModuleConeT::D_flat's convention).  With
// g := -c_d, F(i) := g(v0) + c(v0v4) + c(v4v6) (the potential at v6 = i),
//
//     Psi(c, c_d) := P = Phi({0, oo}) = (rho_n(S) - 1) F(i),
//
// and Phi({a/M, oo}) = - sum_k rho_n(g_k) P along Manin's continued fraction
// (g_k in SL_2(Z); -I acts trivially, n even).  P lies in
// W(R) = ker(1 + rho(S)) cap ker(1 + rho(U) + rho(U)^2), U = [[0,-1],[1,1]]
// (Manin's two- and three-term relations), and Psi : H^1_c(R) -> W(R) is an
// isomorphism (COMP, G1).
//
// THE FUNCTIONAL (R19-8 Sec. 1.4).  For chi of conductor M (ell not dividing M) and 0 <= j <= n,
//     ev_{chi,j}(phi) = sum_{a mod M} conj(chi)(a) < rho_n([[M, -a], [0, 1]]) Phi({a/M, oo}), u_{n-j}^vee >,
// integral; analytically L(f,chi,j+1)/Omega^{+-} = chi(-1)(-2 pi i)^{j+1} tau(chi)/(j! C(n,j) M^{j+1}) ev_{chi,j},
// +- the GEOMETRIC sign chi(-1)(-1)^j ([C]'s convention: -chi(-1)(-1)^j).  ev lives on H^1_c; it descends
// to H^1_m only at a residually non-Eisenstein m (isign).
//
// HECKE ON SYMBOLS (R19-8 Sec. 1.3): T_p P = rho(diag(1,p)) P + sum_{b<p} rho([[p,-b],[0,1]]) Phi({b/p, oo}).
//
// This header is STL-only, plus the shared ../ZellK.h kit (the pipeline objects -- the cone, its Hecke action -- are
// built in BirchModularSymbol.cpp).  rho_n is applied to a vector by L D U (two
// binomial transforms and a diagonal, O(n^2)), the R19-8 prototype's route; an
// independent O(n^3) construction from [C] def:rhon (RhoDirectMatrix) is the
// G5 oracle.  All arithmetic is exact in Z/ell^m, ell^m < 2^64 (unsigned __int128
// products).  Writes nothing; no golden object; no record.
// ===========================================================================

#include "../ZellK.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {
namespace birch {

using Vec = std::vector<uint64_t>;

// Z/ell^m (m >= 1; m = 1 is F_ell), ell^m < 2^64, and the integral 2x2 matrix: the shared kit (S24 CL-G1).
using Ring = zk::Ring;
using zk::MakeRing;                                 // ell >= 2, ell^m < 2^64 (checked)
using Mat2 = zk::M22;

// rho_n on vectors of length n + 1 over a ring (binomial table cached).
class SymRho {
public:
    SymRho(uint32_t n, const Ring& R);
    uint32_t n() const { return n_; }
    const Ring& ring() const { return R_; }
    Vec apply(const Mat2& A, const Vec& v) const;   // rho_n(A) v, L D U with S-swaps
    Vec S(const Vec& v) const;                      // rho(S) u_t = (-1)^t u_{n-t}
    Vec diag(uint64_t a, uint64_t d, const Vec& v) const;
    Vec upper(uint64_t b, const Vec& v) const;      // [[1,b],[0,1]]
    Vec lower(uint64_t g, const Vec& v) const;      // [[1,0],[g,1]]
    uint64_t binom(uint32_t t, uint32_t s) const { return binom_[static_cast<size_t>(t) * (n_ + 1) + s]; }

private:
    uint32_t n_;
    Ring R_;
    std::vector<uint64_t> binom_;
};

// The independent O(n^3) oracle: rho_n(A) as a dense (n+1) x (n+1) matrix (row-major) straight from
// [C] def:rhon, column j = (a x + c y)^{n-j} (b x + d y)^j.
std::vector<uint64_t> RhoDirectMatrix(const Mat2& A, uint32_t n, const Ring& R);

// Manin: the SL_2(Z) matrices g_k with {oo, a/M} = sum_k g_k {0, oo} (M > 0).
std::vector<Mat2> ManinPath(int64_t a, int64_t M);
Vec PhiCusp(const SymRho& rho, const Vec& P, int64_t a, int64_t M);           // Phi({a/M, oo})
// Phi({r, s}), cusps as (num, den) in lowest terms with den > 0, or (1, 0) = oo.
Vec PhiPair(const SymRho& rho, const Vec& P, int64_t rn, int64_t rd, int64_t sn, int64_t sd);
bool InW(const SymRho& rho, const Vec& P);                                      // both Manin relations
Vec HeckeTp(const SymRho& rho, const Vec& P, uint64_t p);
// The NEGATIVE CONTROL of G3: rho(alpha) in place of rho(adj alpha).
Vec HeckeTpWrongAdjoint(const SymRho& rho, const Vec& P, uint64_t p);
Vec SignPart(const Ring& R, const Vec& P, int sign);                            // (P +- rho(iota) P) / 2

// A Dirichlet character with values in R (chi(a) for a in [0, M)); chibar its conjugate.
struct DirichletCharacter {
    uint64_t M = 1;
    std::vector<uint64_t> chi, chibar;
    int parity = 1;                     // chi(-1) = +-1
    bool real = true;
    std::string name;
};
DirichletCharacter TrivialCharacter(const Ring& R);
// Kronecker (D / .), D a fundamental discriminant (checked), M = |D|.
DirichletCharacter KroneckerCharacter(int64_t D, const Ring& R);
// The quadratic primitive character of conductor M when exactly one of +-M is a fundamental
// discriminant (M = 1: trivial); throws std::invalid_argument otherwise (M = 8: use KroneckerCharacter).
DirichletCharacter QuadraticCharacterOfConductor(uint64_t M, const Ring& R);
// M prime, d | M - 1, d | ell - 1: chi(g^i) = zeta_d^{e i}, g the least primitive root mod M, zeta_d the
// Teichmuller root of unity omega(r)^{(ell-1)/d} (r the least primitive root mod ell), gcd(e, d) = 1.
DirichletCharacter PrimeConductorCharacter(uint64_t M, uint32_t d, uint32_t e, const Ring& R);

// ev_{chi,j}(P) for j = 0..n.
Vec EvAllJ(const SymRho& rho, const Vec& P, const DirichletCharacter& chi);

// ---- module linear algebra over Z/ell^m (m >= 1), small dense --------------------------------------
struct Mat {
    uint32_t r = 0, c = 0;
    std::vector<uint64_t> a;
    Mat() = default;
    Mat(uint32_t r_, uint32_t c_) : r(r_), c(c_), a(static_cast<size_t>(r_) * c_, 0) {}
    uint64_t& at(uint32_t i, uint32_t j) { return a[static_cast<size_t>(i) * c + j]; }
    uint64_t at(uint32_t i, uint32_t j) const { return a[static_cast<size_t>(i) * c + j]; }
};
Mat ColumnsToMat(const std::vector<Vec>& cols, uint32_t rows);
// Smith form P A Q = diag(ell^{v_0}, ..., ell^{v_{np-1}}, 0, ...) (full valuation pivoting); P optional.
struct SmithForm {
    std::vector<uint32_t> v;            // pivot valuations (< m)
    Mat P, Q;
    bool hasP = false;
};
SmithForm Smith(const Ring& R, Mat A, bool wantP);
// ker A: generators with their exponents (orders ell^e).
void KernelGenerators(const Ring& R, const Mat& A, std::vector<Vec>& gens, std::vector<uint32_t>& exps);
uint32_t SpanLength(const Ring& R, const std::vector<Vec>& cols, uint32_t rows);     // length of the column span
std::vector<uint32_t> SpanInvariantFactors(const Ring& R, const std::vector<Vec>& cols, uint32_t rows);
// any x with A x = y; false if y is not in im A.
bool Solve(const Ring& R, const SmithForm& S, const Vec& y, Vec& x);
// A saturated free basis of span(cols) with unit pivot rows: basis[i][pivot[j]] = delta_ij.  Returns false
// (and leaves a partial result) when the span is not a free direct summand (a residual non-zero column).
struct PivotBasis {
    std::vector<Vec> basis;
    std::vector<uint32_t> pivot;
    bool free_summand = true;
    Vec coords(const Vec& w) const;     // w in the span: its coordinates
};
PivotBasis SaturatedBasis(const Ring& R, std::vector<Vec> cols);
// Berkowitz characteristic polynomial (low -> high, monic) of a square matrix over R.
Vec CharPoly(const Ring& R, const Mat& A);
Mat MatMul(const Ring& R, const Mat& A, const Mat& B);
Vec MatVec(const Ring& R, const Mat& A, const Vec& x);

// ---- the cone side (built in BirchModularSymbol.cpp from the C++ pipeline) ---------------------------
struct BirchCellOptions {
    std::vector<uint64_t> hecke_primes;     // the cone action T^c_p on H^1_c (m >= 2: ModuleConeActionT<uint64_t>)
    bool iota_through_pi = false;           // m >= 2: [C]'s iota^* on H^1 and pi : H^1_c -> H^1 (G3)
    bool pi_matrix = false;                 // the class matrix of pi (both rings)
    bool coboundary_check = true;           // Psi kills the cone coboundaries (D_flat route at n <= 80; the v6 block always)
};
struct BirchConeCell {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t m = 0;
    Ring R;
    std::vector<uint32_t> exps;             // H^1_c = (+)_i Z/ell^{e_i} (m = 1: all 1)
    std::vector<Vec> P;                     // Psi(b_i)
    std::vector<Vec> P_wrong_sign;          // the NEGATIVE CONTROL g := +c_d
    uint32_t coboundaries_checked = 0;
    bool coboundaries_killed = false;
    std::string coboundary_route;
    std::vector<uint64_t> hecke_primes;
    std::vector<Mat> A_c;                   // t x t, column i = class coordinates of T^c_p b_i
    std::vector<uint32_t> bulk_exps;        // H^1(K) cyclic summands (m >= 2) / dims (m = 1)
    Mat pi;                                 // t_K x t
    Mat iota_C;                             // t_K x t_K: [C]'s iota^* = -iota^*_geo on H^1 (m >= 2)
    bool have_pi = false, have_iota = false;
    double seconds = 0;
    std::string route;                      // "field cone (EquivariantMappingCone)" | "module cone (ModuleConeT<uint64_t>)"
    // P of a class given by its coordinates on the cyclic summands (reduced into [0, ell^{e_i})).
    Vec symbol(const std::vector<uint64_t>& coords) const;
};
// n >= 2 even, ell >= 5 prime; m = 1: ell < 2^32 (the uint32_t field cone); m >= 2: ell^m < 2^64.
BirchConeCell RunBirchConeCell(uint32_t n, uint64_t ell, uint32_t m, const BirchCellOptions& opt);

// ---- (L1) the Eisenstein index statement on H^1_c (R19-8 Sec. 6 (L1)) ------------------------------
// On the geometric-even part of W(Z/ell^m) = Psi(H^1_c): the m_Eis block of T_2 (generalised eigenvalue
// e_2 = 1 + 2^{k-1} mod ell), which must be free of rank 2 (p_0 = u_0 - u_n and one cusp line); the cusp
// eigenline r^+ (T_2 r^+ = a r^+), and its congruence depth to p_0: r^+ == unit * p_0 mod ell^depth.
struct EisensteinSymbolReading {
    bool applicable = false;                // the block is free of rank 2 (else the reason)
    std::string reason;
    uint32_t block_rank = 0;
    uint64_t e2 = 0, a2 = 0;                // T_2 eigenvalues on p_0 and on the cusp line
    uint32_t v_delta = 0, v_y = 0;          // v(a - e), v(y) (T_2 w = y p_0 + a w)
    uint32_t depth = 0;                     // v(r^+ - c p_0), r^+ primitive
    bool depth_exact = false;               // v(a - e) < m
    Vec r_plus;                             // the cusp eigenvector (primitive), known mod ell^{m - v_delta}
    Vec r_minus;                            // the geometric-odd block generator (rank-1 m_Eis block of W^-), when found
    bool have_r_minus = false;
};
EisensteinSymbolReading ReadEisensteinSymbols(const BirchConeCell& cell);

} // namespace birch
} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_BIRCH_MODULAR_SYMBOL_H
