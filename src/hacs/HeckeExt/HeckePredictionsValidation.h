// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_PREDICTIONS_VALIDATION_H
#define HECKE_EXT_HECKE_PREDICTIONS_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 4, WP-7 (phase E6).  Returns 0 on success (invariant
// I9); registered in ValidationRegistry::RegisterAllValidations() as a `fast`
// entry.  It builds no cohomology and no coset table: the Bernoulli unit is
// pure F_ell arithmetic and the Glover unit is pure integer combinatorics; the
// only pipeline object it touches is `ComputeSymmetricPowerMatrixFM`, the
// substrate's own rho_n, on 2 x 2 inputs.
//
// ARMS.
//
//   (A) THE PIN.  HeckeExt/HeckeBoundary's pinned num(B_k/k) table (k <= 26,
//       [E] Claim 1.8's Kummer range) is RECONSTRUCTED AS AN INTEGER from
//       `bernoulli_mod` alone: num(B_k) mod ell at ten primes ell > k + 1,
//       lifted by CRT, divided by gcd(num(B_k), k), signed by the symmetric
//       representative -- and compared entrywise with
//       `pinned_bernoulli_numerator(k)`.  The denominator that turns B_k mod ell
//       into num(B_k) mod ell is von Staudt-Clausen's prod_{(p-1) | k} p,
//       computed here, never read from the unit.
//
//   (B) BEYOND THE PIN.  The same CRT reconstruction at k = 28, ..., 42 against
//       the classical numerators of B_k, i.e. sixteen values k <= 42 where the
//       shipped table stops at 26.
//
//   (C) AN INDEPENDENT FORMULA.  `bernoulli_mod(w, ell)` against the Worpitzky /
//       Stirling closed form  B_w = sum_m 1/(m+1) sum_j (-1)^j C(m, j) j^w,
//       which shares no line of code with the unit's recursion, for EVERY
//       w < ell - 1 at every prime ell <= 131, and sampled at ell = 691 and
//       ell = 3617 (so up to w = 3615).
//
//   (D) THE STRUCTURE THEOREMS.  von Staudt-Clausen (den(B_k) = prod_{(p-1)|k} p,
//       re-derived against the exact rationals of arm B) and the KUMMER
//       CONGRUENCE B_k/k == B_{k'}/k' mod ell for k == k' mod (ell - 1), which
//       is the warrant for `kummer_verdict`'s third branch.
//
//   (E) THE KUMMER / HARDER PREDICATE.  `kummer_divides(k, ell)` against plain
//       integer divisibility of the pinned numerator, for every even k <= 26 and
//       every prime ell <= 100 plus the Kummer primes -- i.e. including the
//       out-of-range k >= ell - 1 cells that exercise the von Staudt-Clausen and
//       Kummer-congruence branches; the four named divisors 691 | B_12,
//       3617 | B_16, 103 * 2294797 | B_24, 657931 | num(B_26/26); the
//       regularity of every prime ell <= 31 and the irregular pair (37, 32).
//
//   (F) THE REFUSAL.  `bernoulli_mod(w, ell)` throws BernoulliOutOfRange for
//       every w >= ell - 1 (and returns for every w < ell - 1), and
//       `kummer_divides(0, ell)` throws.  A refusal, never a wrong residue.
//
//   (G) GLOVER, STRUCTURE.  `GloverFactors(n, ell)`: dimension n + 1, every
//       sym <= ell - 1, the central character sym + 2 det == n mod (ell - 1),
//       irreducibility for n <= ell - 1, the canonical order and the
//       det-twist-collapsing SL_2 view, over ell in {2, 3, 5, 7, 11, 13, 691}
//       and n up to 4 ell + 5.
//
//   (H) GLOVER, THE BRAUER CHARACTER.  The proof arm.  For every ell-regular
//       (= semisimple) class of GL_2(F_ell), the eigenvalue multiset of the REAL
//       rho_n(g) -- read off the characteristic polynomial of the matrix
//       `ComputeSymmetricPowerMatrixFM` returns, by Hessenberg reduction over
//       F_ell and root extraction over F_{ell^2}, with the Hessenberg polynomial
//       itself cross-checked against det(cI - rho_n(g)) at every c in F_ell --
//       is compared, after Teichmueller lifting, with the multiset the factor
//       list predicts.  Equal Brauer characters on every ell-regular class is
//       equality of composition factors, so each cell is a proof.  Run for every
//       n <= 3 ell at ell in {5, 7} (the plan's range) and, beyond it, at
//       ell in {2, 3} and n <= 2 ell at ell in {11, 13}.  The SL_2(F_ell)
//       sub-family (det = 1) is counted and asserted separately, because that is
//       the arm the plan names.
uint32_t HeckePredictionsValidation();

// (hecke-extensions slot 6, WP-8) Arm [H]'s Brauer-character comparison, for
// ONE ell, callable from HeckeAtlasValidation ([X] Sec. 5's HeckeAtlas row:
// "Glover factors against the Brauer character of rho_n for n <= 3 ell at
// ell in {5, 7}") so that the atlas gate and this validation run the SAME
// oracle rather than two copies of it.  Every n <= n_max, every semisimple
// conjugacy class of GL_2(F_ell), with the det(xI - rho) cross-check at every
// x.  Returns the number of failing checks; `cells` counts (class, n) cells,
// `sl2_cells` those with det = 1, `checks` the individual comparisons.
uint32_t GloverBrauerCrossCheck(uint32_t ell, uint32_t n_max, uint32_t& cells,
                                uint32_t& sl2_cells, uint32_t& checks);

#endif // HECKE_EXT_HECKE_PREDICTIONS_VALIDATION_H
