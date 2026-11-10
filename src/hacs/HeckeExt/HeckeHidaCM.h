// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HIDA_CM_H
#define HECKE_EXT_HIDA_CM_H
// ===========================================================================
// (hecke-extensions slot 16, WP-15) [U] part 2 in code: the CM components on the
// character model.  Binding texts: [U] docs/latex/hida_hecke_algebra.tex
// v0.2.0-pre.2, sec:u-tameN -- def:u-hN, thm:u-controlN, thm:u-cmN (where CM can
// live: ell split in K, |d_K| | N, none at tame level 1 or 2), prop:u-cmexist (the
// theta series theta_lambda, ordinary iff ell splits), prop:u-cmeis ((1) CM meets
// Eisenstein only at a = 1 mod (ell - 1), with E_k(mu, mu chi_K); (2) under h_K = 1
// the untwisted family meets E_k(1, chi_K)), rmk:u-cmgl2, rmk:u-pred2 (U8)-(U10).
// The finite-level object read is the character model of [C, sec:nebentypus]
// (def:nebmodel, prop:neborbit, rmk:nebheilbronn) as built by slot-16 WP-10
// (Hecke/HeckeNebentypus: NebCokernel, HeilbronnChi, ReduceCharacter), under the
// convention of [GT, conv:nebentypus] (L16-2's hand-off): for the quadratic chi_K
// used here chi = chibar, so the dictionary cannot move a value.
//
// WHAT IS HERE (pipeline-free, value-free: no record key, setting, verb or golden
// object reaches it):
//  * imaginary quadratic arithmetic, exact: fundamental discriminants, the class
//    number h_K by reduced primitive forms, w_K, the Kronecker symbol (d / n), the
//    norm form of O_K = Z[(d + sqrt d)/2], principal split primes;
//  * the CM coefficient oracle of theta_{lambda_k} (prop:u-cmexist, prop:u-cmeis(2)):
//    lambda_k((alpha)) = alpha^{k-1}, defined iff w_K | k - 1; a_p = 0 at p inert,
//    a_p = Tr(alpha^{k-1}) at p = alpha alpha-bar split with (alpha) principal (always
//    when h_K = 1); exact (__int128, overflow-checked) and modulo any m;
//  * chi_K as a Dirichlet character mod N (|d_K| | N), ready for ReduceCharacter;
//  * the linear algebra of the readings over F_q (WP-10's Fq / FqMat): kernels,
//    joint generalised eigenspaces of commuting operators, restriction to an
//    invariant subspace, the Loewy length of the local algebra they generate.
// The readings themselves (which subspace is CM, which is ordinary, the
// Eisenstein coincidences) are the validation's: HeckeHidaCMValidation.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "../Hecke/HeckeNebentypus.h"

namespace hecke_ext {
namespace hidacm {

using hecke::neb::Fq;
using hecke::neb::FqMat;

// ---- imaginary quadratic fields -------------------------------------------------
// d a fundamental discriminant (d = 1 mod 4 squarefree, or d = 4m with m = 2, 3 mod 4
// squarefree); d != 1.
bool IsFundamentalDiscriminant(int64_t d);
// The negative fundamental discriminants d with |d| <= dmax, in decreasing order (-3, -4, -7, ...).
std::vector<int64_t> NegativeFundamentalDiscriminants(int64_t dmax);
// h(d) for d < 0 fundamental: the number of reduced primitive forms (a, b, c), b^2 - 4ac = d.
uint32_t ClassNumber(int64_t d);
// |O_K^x|: 6 at d = -3, 4 at d = -4, 2 otherwise.
uint32_t UnitCount(int64_t d);
// The Kronecker symbol (d / n), n >= 1 (multiplicative in n; (d / 2) by d mod 8).
int Kronecker(int64_t d, uint64_t n);
// "split" / "inert" / "ramified" for a prime p.
std::string Splitting(int64_t d, uint64_t p);
// For p split in K = Q(sqrt d): an alpha = x + y w (w = (d + sqrt d)/2) of norm p if
// one exists (the prime above p is principal); returns false otherwise.  trace = Tr alpha.
bool PrincipalPrimeTrace(int64_t d, uint64_t p, int64_t& trace);

// ---- the CM coefficient oracle of theta_{lambda_k} --------------------------------
// a_p(theta_{lambda_k}) for a prime p !| d: 0 at p inert; Tr(alpha^{k-1}) at p split with
// (alpha) principal.  Refuses (std::runtime_error) unless w_K | k - 1 (lambda_k undefined)
// or at p split non-principal (lambda_k is then defined on principal ideals only:
// [U] prop:u-cmeis(2)'s h_K = 1 hypothesis, WP-14 referee U-1).
__int128 CmCoefficientExact(int64_t d, uint32_t k, uint64_t p);      // overflow-checked
uint64_t CmCoefficientMod(int64_t d, uint32_t k, uint64_t p, uint64_t m);

// chi_K mod N (|d| divides N): chi(u) = (d / u) on the units, as a quadratic
// DirichletCharacter (conrey = 0: not read from a label; L = 2).
hecke::neb::DirichletCharacter KroneckerCharacter(int64_t d, uint64_t N);

// ---- linear algebra over F_q for the readings -------------------------------------
// A basis of ker A (A: r x c) as the columns of a c x k matrix (k = 0: a c x 0 matrix).
FqMat Kernel(const Fq& F, const FqMat& A);
FqMat MatPow(const Fq& F, const FqMat& A, uint64_t e);
// A - s I.
FqMat Shift(const Fq& F, const FqMat& A, uint32_t s);
// The joint generalised eigenspace of commuting h x h operators T_i at the scalars s_i:
// ker of the stacked (T_i - s_i)^h, as columns (h x w).
FqMat JointGeneralisedEigenspace(const Fq& F, const std::vector<FqMat>& ops, const std::vector<uint32_t>& scal);
// The matrix of T on the T-stable subspace spanned by the columns of B (full column
// rank): X with B X = T B; throws unless B is T-stable.
FqMat Restrict(const Fq& F, const FqMat& T, const FqMat& B);
// dim of the generalised s-eigenspace of a square A.
uint32_t GeneralisedEigenDim(const Fq& F, const FqMat& A, uint32_t s);
// The Loewy length of the local algebra generated by I and commuting nilpotent N_i
// (w x w): the least L with every product of L of the N_i zero (1 if all N_i = 0, i.e.
// the algebra is F); dim_A = dim of the algebra.  (S16 WP-15 repairs, audit m6: the old
// field `layers` held the dimensions of the spans of the degree-j monomials, which are
// neither dim rad^j nor the Loewy layers; it is renamed and the two true sequences added.)
//  * monomial_degree_dims[j-1] = dim span{N_{i_1} ... N_{i_j}}, j = 1 .. loewy - 1;
//  * radical_power_dims[j-1]   = dim rad^j = dim of the span of all monomials of degree >= j;
//  * loewy_layers[j]           = dim rad^j / rad^{j+1}, j = 0 .. loewy - 1 (rad^0 = A), summing to dim_A.
struct LoewyReading {
    uint32_t loewy = 1;
    uint32_t dim_A = 1;
    std::vector<uint32_t> monomial_degree_dims;
    std::vector<uint32_t> radical_power_dims;
    std::vector<uint32_t> loewy_layers;
};
LoewyReading LoewyOfNilpotents(const Fq& F, const std::vector<FqMat>& nil, uint32_t w);

} // namespace hidacm
} // namespace hecke_ext

#endif // HECKE_EXT_HIDA_CM_H
