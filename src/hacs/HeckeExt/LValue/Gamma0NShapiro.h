// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_GAMMA0N_SHAPIRO_H
#define HECKE_EXT_LVALUE_GAMMA0N_SHAPIRO_H

// ===========================================================================
// (hecke-extensions slot 22, P6) THE BIRCH ROUTE ON Gamma_0(N) BY SHAPIRO, AND THE TAME TWISTS OF THE MEASURE.
// Spec and predictions docs/implementation/slot22/P6_spec.md (committed before the first gated run); oracle
// docs/implementation/slot22/p6/p6_oracle.py (own code); report docs/implementation/slot22/P6_report.md.
//
// THE MODEL (owner requirement: level N by Ind to SL_2(Z) on the fixed model, exploiting permutation (x) rho).
// H^1_c(Gamma_0(N), V) = H^1_c(SL_2(Z), Ind V), Ind V = R[P^1(Z/N)] (x) Sym^n (Shapiro).  On [C]'s 7-vertex model the
// symbol map Psi of BirchModularSymbol.h (Phi({0,oo}) = (rho(S) - 1) F(i)) applies verbatim to the coefficient module
// Ind V, whose SL_2(Z)-action is the permutation of the cosets Gamma_0(N) g <-> x(g) = (c : d) (bottom row) tensored with
// rho_n.  A level-N symbol is therefore P = (P_x)_{x in P^1(Z/N)}, P_x in V, with
//     Phi({g 0, g oo}) = rho(g) P_{x(g)}                                    (g in SL_2(Z); the identity coset is the form's own symbol)
// and the Manin relations of W(Ind V) = ker(1 + S) cap ker(1 + U + U^2):
//     P_x + rho(S) P_{xS} = 0,   P_x + rho(U) P_{xU} + rho(U^2) P_{xU^2} = 0,   (c, d) g the row action.
// The geometric sign s: rho(iota) P_{x'} = s P_x, iota = diag(-1, 1), x' = (-c : d) (level one: SignPart).
// THE BLOCK STRUCTURE (first cut): the S-relation is solved orbit by orbit of x -> xS (P_{xS} = -rho(S) P_x; a fixed point
// keeps (1 + rho(S)) P_x = 0), so only one V-block per S-orbit is a variable; the U- and iota-relations are imposed once per
// U-orbit / iota-orbit (the relation at xU is rho(U^2) times the one at x).  The remaining system is dense Smith over
// Z/ell^m (BirchModularSymbol.h).  A sparse elimination along the U-orbits is the P6 tail (P6_report.md).
// IDENTIFICATION: Psi : H^1_c(Gamma_0(N), V) -> W(Ind V) is an isomorphism for ell >= 5 (Ash-Stevens; at N = 1 it is the
// slot-19 G1 COMP on the cone); here the domain is PRESENTED by W(Ind V), and its rank is gated against Eichler-Shimura:
// rank W^- = s_k(N), rank W^+ = s_k(N) + c(N) (k > 2; c(N) - 1 at k = 2) when every cusp is iota-fixed (N prime).
//
// HECKE (p !| N): (T_p P)_x = rho(g_x)^{-1} sum_alpha rho(adj alpha) Phi(alpha g_x {0, oo}), alpha in {diag(p,1)} u {[[1,b],[0,p]]};
// at N = 1 this is BirchModularSymbol.h's HeckeTp.  Phi of an arbitrary path by Manin's continued fraction with cosets.
// RESTRICTION / CORESTRICTION to level one: Res(P)_x = P (all x), Cores(P) = sum_x P_x; Cores Res = [SL_2(Z) : Gamma_0(N)].
// CONTRACT: Res, Cores and T_p (p !| N) are the modular-symbol shadows of [C, prop:levelcorrespondence](4)(b) (projection and trace)
// and (4)(d) (T_p^{(N)}); the presentation by Ind V is the symbol side of [C, def:levelellmodel] (the induced fibre Ind_{Gamma_0(N)}^Gamma V_R;
// there general N is future work) and needs 6 in R^x, the hypothesis of [C, prop:lhcompliant](4) / [C, obl:shapiro].
//
// THE MEASURE (level N, tame modulus T).  c(b, Mod) = < rho([[Mod, -b], [0, 1]]) Phi({b/Mod, oo}), u_n^vee > = - sum_g P_{x(g)}(g_c, g_d)
// over Manin's matrices of b/Mod (P(c, d) = sum_t P_t c^{n-t} d^t), and for m >= 1, b mod T ell^m,
//     mu(b + T ell^m) = alpha^{-m} c(b, T ell^m) - alpha^{-m-1} ell^{k-2} c(b mod T ell^{m-1}, T ell^{m-1}).
// T = 1 and N = 1: [E] def:lvmtt (BuildMeasure).  gamma = [[T ell^m, -b], [0, 1]] is the plan's [[N ell^m, -a], [0, 1]].
// Moment identity (the P6 prediction, derived in P6_spec.md sec. 2): for chi mod T (T prime to ell), psibar = chibar omega^{-i},
//     (-1)^j C(n,j) sum_{b mod T ell^M, ell !| b} chibar(b) omega^{-i}(b) b^j mu(b + T ell^M) = E ev_{psi,j}   (mod ell^{min(M,R)}),
//     E = alpha^{-1} (i != 0, conductor T ell),  E = (1 - chibar(ell) ell^j / alpha)(1 - chi(ell) ell^{n-j} / alpha) (i = 0, modulus T).
//
// STL + BirchModularSymbol.h + ModularSymbolMeasure.h.  Exact arithmetic in Z/ell^m (ell^m < 2^62).  Writes nothing; no golden
// object; no record.
// ===========================================================================

#include "BirchModularSymbol.h"
#include "ModularSymbolMeasure.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace hecke_ext {
namespace lvalue {
namespace shapiro {

using birch::Mat2;
using birch::Ring;
using birch::SymRho;
using birch::Vec;

// P^1(Z/N): canonical representative = the lexicographically least (u c mod N, u d mod N) over units u; points sorted by it.
// N = 1: one point (the coset of SL_2(Z) itself).  N <= kMaxLevel (the canonical table is N^2 entries).
class P1N {
public:
    static constexpr uint64_t kMaxLevel = 4096;
    explicit P1N(uint64_t N);
    uint64_t N() const { return N_; }
    uint32_t size() const { return static_cast<uint32_t>(pts_.size()); }
    uint32_t idx(int64_t c, int64_t d) const;                       // (c : d), gcd(c, d, N) = 1 (checked)
    std::pair<uint64_t, uint64_t> point(uint32_t i) const { return pts_[i]; }
    Mat2 lift(uint32_t i) const;                                    // g in SL_2(Z) with x(g) = point i (cached)
    uint32_t S(uint32_t i) const;                                   // x S
    uint32_t U(uint32_t i) const;                                   // x U
    uint32_t iota(uint32_t i) const;                                // (-c : d)
    uint64_t index_in_sl2z() const;                                 // [SL_2(Z) : Gamma_0(N)] = |P^1(Z/N)|

private:
    uint64_t N_;
    std::vector<std::pair<uint64_t, uint64_t>> pts_;
    std::vector<uint32_t> table_;                                   // (c mod N) * N + (d mod N) -> index, or UINT32_MAX
    std::vector<Mat2> lifts_;
};

// W_N^{sign} = the symbol module over Z/ell^m: generators (flattened, x-major then t: P_x[t] at x * (n + 1) + t) with their
// exponents (orders ell^e); sign = +1 / -1 (geometric) or 0 (the whole of W(Ind V)).
struct Gamma0NCell {
    uint64_t N = 1;
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t m = 0;
    int sign = 0;
    Ring R;
    uint32_t L = 1;                         // |P^1(Z/N)|
    std::vector<Vec> basis;                 // generators of W (flattened)
    std::vector<uint32_t> exps;             // their exponents
    bool free_module = false;               // every exponent == m
    uint32_t s_orbits = 0, s_fixed = 0, u_orbits = 0, iota_orbits = 0;
    uint32_t unknowns = 0, relation_rows = 0;   // the reduced system (after the S-orbit elimination)
    double seconds = 0;
    Vec symbol(const std::vector<uint64_t>& coords) const;          // sum_i coords_i basis_i (reduced mod ell^{e_i})
};
// n even >= 0, ell >= 5 prime, ell^m < 2^62, 1 <= N <= P1N::kMaxLevel.
Gamma0NCell RunGamma0NCell(uint64_t N, uint32_t n, uint64_t ell, uint32_t m, int sign);

// The NEGATIVE CONTROL of the coset map: the S-relation without the permutation (P_x + rho(S) P_x = 0 for every x).
Gamma0NCell RunGamma0NCellWrongCosets(uint64_t N, uint32_t n, uint64_t ell, uint32_t m, int sign);

Vec PhiCuspN(const SymRho& rho, const P1N& p1, const Vec& P, int64_t a, int64_t M);
bool InWN(const SymRho& rho, const P1N& p1, const Vec& P);
Vec HeckeTpN(const SymRho& rho, const P1N& p1, const Vec& P, uint64_t p);                // p !| N (checked)
Vec HeckeTpNWrongAdjoint(const SymRho& rho, const P1N& p1, const Vec& P, uint64_t p);    // NEGATIVE CONTROL: rho(alpha) for rho(adj alpha)
Vec ResFromLevelOne(const Vec& P1, uint32_t L);
Vec CoresToLevelOne(const Ring& R, const Vec& P, uint32_t L, uint32_t n);

// ev_{chi,j} at the identity coset, j = 0..n (BirchModularSymbol.h's EvAllJ at N = 1).
Vec EvAllJN(const SymRho& rho, const P1N& p1, const Vec& P, const birch::DirichletCharacter& chi);

// The eigenvalue of T_p on an eigensymbol (false if P is not an eigenvector, or P == 0).
bool EigenvalueN(const SymRho& rho, const P1N& p1, const Vec& P, uint64_t p, uint64_t& lambda);
// The submodule {v in span(cell.basis) : (T_p - lambda) v = 0}: generators and exponents (flattened symbols).
void EigenSubmodule(const Gamma0NCell& cell, uint64_t p, uint64_t lambda, std::vector<Vec>& gens, std::vector<uint32_t>& exps);
// The first ell-unit coordinate scaled to 1 (P6's normalisation of an eigensymbol); false if no coordinate is a unit.
bool NormaliseFirstUnit(const Ring& R, Vec& P);
// (slot 23, G-EVT23; moved verbatim from HeckeLValueShapiroValidation.cpp) The normalised generator of a rank-1 free W
// (NormaliseFirstUnit applied to its one basis vector); empty unless W is free of rank 1.
Vec RankOneGenerator(const Gamma0NCell& C);
// (slot 23, G-EVT23) The eigen-line reading of HeckeLValueShapiroValidation arm (S3) (its `kH` cells), shared with the
// golden harness's `SHAPIRO N=<N> n=<n> ell=<ell>` points: W^- = RunGamma0NCell(N, n, ell, m, -1), its rank, and the
// eigenvalues a_p mod ell^m (EigenvalueN on RankOneGenerator) at the primes ShapiroLinePrimes(N) = the five primes
// p <= 13 with p !| N, ascending (the kH table's primes).  rank_one = W^- free of rank 1; all_eigen = every a_p exists.
std::vector<uint64_t> ShapiroLinePrimes(uint64_t N);
struct ShapiroEigenLineReading {
    uint32_t rankW = 0;                                   // the number of generators of W^- (its rank when free)
    bool free_module = false, rank_one = false, all_eigen = false;
    std::vector<std::pair<uint64_t, uint64_t>> ap;        // (p, a_p mod ell^m), p ascending
};
ShapiroEigenLineReading ShapiroEigenLine(uint64_t N, uint32_t n, uint64_t ell, uint32_t m);

// The level-N measure (T = 1) as the slot-20 ModularSymbolMeasure (balls a + ell^m, m = 0 .. M; c(., 0) = c(., -1) = P_{(0:1), n}).
ModularSymbolMeasure BuildMeasureN(const SymRho& rho, const P1N& p1, const Vec& P, uint64_t a_ell, uint64_t alpha, uint32_t M,
                                   bool ell_n_factor, unsigned threads = 0);

// The tame measure: balls b + T ell^m, m = 1 .. M (c tables for m = 0 .. M).
struct TameMeasure {
    uint64_t N = 1, T = 1, ell = 0;
    uint32_t R = 0, M = 0, n = 0;
    uint64_t q = 0, a_ell = 0, alpha = 0;
    bool ell_n_factor = true;
    std::vector<std::vector<uint64_t>> c;    // c[m][b] = c(b, T ell^m), m = 0 .. M
    std::vector<std::vector<uint64_t>> mu;   // mu[m][b], m = 1 .. M (mu[0] empty)
    uint32_t three_term_failures() const;    // sum_t c(b + t T ell^m, m+1) == a_ell c(b, m) - ell^n c(b mod T ell^{m-1}, m-1), m = 1 .. M-1
    uint32_t distribution_failures() const;  // sum_t mu(b + t T ell^m, m+1) == mu(b, m), m = 1 .. M-1
    // mom[ii][j] = sum_{b mod T ell^M, ell !| b} chibar(b mod T) omega^{-i}(b) b^j mu(b + T ell^M), i = chars[ii] (mod ell^R).
    std::vector<std::vector<uint64_t>> moments(const std::vector<uint64_t>& chibar_mod_T, const std::vector<uint32_t>& chars) const;
    uint64_t fnv_units(uint32_t level) const;  // FNV-1a-64 of mu(b + T ell^level), b ascending with ell !| b and gcd(b, T) = 1
};
TameMeasure BuildTameMeasure(const SymRho& rho, const P1N& p1, const Vec& P, uint64_t T, uint64_t a_ell, uint64_t alpha, uint32_t M,
                             bool ell_n_factor, unsigned threads = 0);

// The tame moment identity (spec sec. 2): failures / total over j = 0..n and the listed i, mod ell^{min(M, R)}.  chi is a character
// mod T with values in R (chi[b], b < T; T = 1: {1}); swap_chi = true is the NEGATIVE CONTROL (chi and chibar exchanged in E).
struct MomentIdentityReading {
    uint32_t failures = 0, total = 0;
};
MomentIdentityReading TameMomentIdentity(const SymRho& rho, const P1N& p1, const Vec& P, const TameMeasure& me,
                                         const std::vector<uint64_t>& chi, const std::vector<uint32_t>& chars, bool swap_chi = false);

// A Dirichlet character mod a prime T of order d | gcd(T - 1, ell - 1) (PrimeConductorCharacter's convention); T = 2: trivial mod 2.
std::vector<uint64_t> TameCharacter(uint64_t T, uint32_t d, uint32_t e, const Ring& R);

// Eichler-Shimura: dim S_k(Gamma_0(N)) and the cusp count (Diamond-Shurman Fig. 3.4 / Thm 3.5.1, 3.6.1).
uint64_t DimCuspFormsGamma0(uint64_t N, uint32_t k);
uint64_t CuspsGamma0(uint64_t N);
// eta(z)^r eta(N z)^r, r (N + 1) = 24: the q-coefficient a_p (the weight-r newform of level N, N in {2, 3, 5, 11}).
int64_t EtaProductCoefficient(uint64_t N, uint64_t p);

} // namespace shapiro
} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_GAMMA0N_SHAPIRO_H
