// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_GLOVER_FACTORS_H
#define HECKE_EXT_GLOVER_FACTORS_H

// ===========================================================================
// (hecke-extensions slot 4, WP-7 / phase E6) THE GLOVER / ASH-STEVENS
// COMPOSITION FACTORS OF Sym^n(F_ell^2)
// ([E] Sec. identities item 6: "the semisimplification of H^1(Gamma; M_n) is
//  the sum over the Glover factors Sym^{n_i} (x) det^{j_i} (0 <= n_i <= ell - 1,
//  n_i + 2 j_i == n) of theta^{j_i}-twists of weight-(n_i + 2) systems, plus the
//  H^0-corrections";  [X] Sec. 5 HeckeAtlasValidation's Glover arm.)
// ===========================================================================
//
// WHAT IS COMPUTED.  For a prime ell and n >= 0, the JORDAN-HOELDER FACTORS,
// WITH MULTIPLICITY, of the F_ell[GL_2(F_ell)]-module Sym^n(F_ell^2).  The
// simple modules of GL_2(F_ell) over F_ell are exactly
//
//     L(r, j) = Sym^r(F_ell^2) (x) det^j,    0 <= r <= ell - 1,  j in Z/(ell - 1)
//
// (for ell = 2 the det-twist is trivial and j is always 0), and the central
// character forces  r + 2 j == n  (mod ell - 1)  on every factor of Sym^n.
//
// THE CONVENTION IS THE PIPELINE'S OWN.  "Sym^n" here is the module on which
// `ComputeSymmetricPowerMatrixFM` acts: for A = diag(a, d) the matrix rho_n(A)
// is diagonal with entry(j, j) = a^{n-j} d^j (FlatMatrix.cpp branch [C1]), i.e.
// the eigenvalues of rho_n(g) at a semisimple g with eigenvalues alpha, beta are
// the n + 1 products alpha^{n-t} beta^t.  This is Sym^n of the STANDARD module,
// not of its dual: the two differ by det^{-n}, which would shift every j_i.
// HeckePredictionsValidation asserts the convention against the real rho_n.
//
// THE ALGORITHM.  The contract [C] carries no label for this statement -- it is
// [E] item 6's own, and the derivation below is the unit's warrant.  Write
// theta = x^ell y - x y^ell = prod over the ell + 1 lines of P^1(F_ell); it is
// the degree-(ell + 1) SL_2(F_ell)-invariant and satisfies g . theta = det(g) theta.
// Multiplication by theta is an injective map of GL_2(F_ell)-modules
//
//     Sym^{n - ell - 1} (x) det  >---->  Sym^n ,
//
// whose cokernel has dimension (n + 1) - (n - ell) = ell + 1 and is the
// principal series Ind_B^G(chi) obtained by restricting a degree-n form to each
// of the ell + 1 lines.  The Jordan-Hoelder factors of a principal series
// Ind_B^G(chi_1^s chi_2^t) of GL_2(F_ell) in defining characteristic are the two
// modules Sym^{r} (x) det^{t} and Sym^{ell - 1 - r} (x) det^{s} with
// r = (s - t) mod (ell - 1) -- which for r = 0 is the trivial/Steinberg pair.
// Here (s, t) = (n, 0), so the recursion is
//
//     n <= ell - 1 :  Sym^n  is irreducible,      factors = { (n, 0) }
//     n >= ell     :  factors(n) = { (r, 0), (ell - 1 - r, n) }
//                                  u  ( factors(n - ell - 1) twisted by det )
//                     with r = n mod (ell - 1).
//
// The chain n, n - (ell + 1), ... contains exactly floor((n + 1) / (ell + 1))
// values >= ell, each emitting two raw factors, and terminates in a value in
// [-1, ell - 1] emitting 0 or 1.  So it takes floor((n + 1) / (ell + 1))
// non-trivial steps and emits at most 2 floor((n + 1) / (ell + 1)) + 1 factors
// before merging -- sharp at n = 0 mod (ell + 1) -- and the whole call is
// O(n / ell) plus the sort.  ([E] v1.5.1 cl:glover(4); v1.5.0's
// 2 floor(n / (ell + 1)) + 1 is FALSE -- at n = ell it gives 1 and the
// recursion emits 2 -- and was disproved by the slot-4 WP-9 referee pass,
// finding F2.)
//
// THE WARRANT IS NOT THE DERIVATION, IT IS THE CHECK.  HeckePredictionsValidation
// compares this recursion against the BRAUER CHARACTER of the real rho_n on
// every ell-regular class of GL_2(F_ell) and of SL_2(F_ell), with the eigenvalue
// multiset of rho_n(g) read off the characteristic polynomial of the matrix
// `ComputeSymmetricPowerMatrixFM` returns -- for every n <= 3 ell at
// ell in {5, 7} (the plan's range) and, beyond it, at ell in {2, 3, 11, 13}.
// Two F_ell[G]-modules with equal Brauer characters have equal composition
// factors, so that comparison is a proof, cell by cell.
//
// PLACEMENT.  HeckeExt/, like every extension-layer unit.  The TU consumes
// NOTHING but EngineCheck.h and the STL -- there is no modular arithmetic in it
// at all, only integer combinatorics -- so it is built and validated in
// isolation.  Rules 1, 2 and 4 of check_heckeext_includes.sh bind it; rule 3 is
// HeckeSturmBound's own allow-list and does not name this file.

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {

// One Jordan-Hoelder factor Sym^{sym} (x) det^{det_twist}, with multiplicity.
struct GloverFactor {
    uint32_t sym = 0;          // 0 <= sym <= ell - 1
    uint32_t det_twist = 0;    // in Z/(ell - 1); always 0 when ell = 2
    uint32_t multiplicity = 0; // >= 1

    uint32_t dimension() const { return (sym + 1u) * multiplicity; }
};

struct GloverDecomposition {
    uint32_t n = 0;
    uint32_t ell = 0;
    // Sorted by (sym, det_twist); no two entries share a (sym, det_twist).
    std::vector<GloverFactor> factors;

    // sum multiplicity * (sym + 1); equals n + 1 by construction.
    uint32_t dimension() const;
    // sum multiplicity: the Jordan-Hoelder length.
    uint32_t length() const;
    // index r in [0, ell - 1] -> total multiplicity of Sym^r over all det-twists,
    // i.e. the decomposition seen by SL_2(F_ell), where det is trivial.
    std::vector<uint32_t> sl2_multiplicities() const;

    std::string to_string() const;
    std::string to_json() const;
};

// The composition factors of Sym^n(F_ell^2) as an F_ell[GL_2(F_ell)]-module.
// Requires ell >= 2 (prime; primality is the caller's contract -- the unit does
// no arithmetic that could detect it).
GloverDecomposition GloverFactors(uint32_t n, uint32_t ell);

} // namespace hecke_ext

#endif // HECKE_EXT_GLOVER_FACTORS_H
