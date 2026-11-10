// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_RELATIONS_VALIDATION_H
#define HECKE_RELATIONS_VALIDATION_H

#include <cstdint>

// Stage S9V -- INDEPENDENT Hecke-relations validation (oracles O9, O10) for
// hecke::HeckeOperatorFamily (Stage S9's post-finalization deliverable, per
// the plan's Sec 0.2 "T_{p^r} normalization for r>=2" ruling and
// HeckeOperatorFamily.h's binding block comment):
//   * T_p_pow(p, r)                    -- the FULL determinant-p^r operator,
//                                          via the [C, thm:relations] algebra
//                                          recursion (PRIMARY route).
//   * T_p_pow_direct(p, r)             -- the SAME full operator, assembled
//                                          INDEPENDENTLY from the classical
//                                          double-coset decomposition
//                                          T_{p^r}^full = sum_{2j<=r}
//                                          p^{jn} . T'_{p^{r-2j}}.
//   * T_p_pow_primitive_direct(p, r)   -- the raw PRIMITIVE double coset
//                                          Gamma diag(1,p^r) Gamma; a
//                                          DIFFERENT operator from the above
//                                          two for r >= 2.
// O9 and O10 below use ONLY the FULL operator (T_p_pow / T_p_pow_direct) on
// both sides of every asserted identity; T_p_pow_primitive_direct appears
// only in the O9 decomposition-witness identity (5), where it is explicitly
// the primitive object being related to the full one.
//
// Written by the S9V validation agent directly from
// docs/latex/hecke_action.tex ([C, thm:relations], [C, prop:diamond],
// [C, cor:ulrecursion], [C, obl:algebra]) and
// docs/implementation/hecke_action_implementation_plan.md (Sec 0.1 amendment
// A-F1, Sec 0.2's T_{p^r} full-operator ruling, and the "S9V -- Hecke-
// relations validation" stage), WITHOUT reading
// HeckeOperatorFamilySmokeValidation.cpp's expected-value logic (only
// HeckeOperatorFamily.h's public signatures/doc comments and the pipeline
// builders' headers -- LocalSystemSL2Z.h, EquivariantCohomologyOnK.h,
// SL2ZTessellationTables.h -- were consulted for how to call the API).
//
// -----------------------------------------------------------------------
// O9 -- internal-consistency identities of the abstract Hecke algebra,
// contract-backed for ALL ell >= 5 ([C, obl:algebra]; no eigenvalue/charpoly
// gating needed -- these are formal ring-homomorphism identities, not
// spectral data). Run for n in {0,2,10}, ell in {13,101}, every computed
// degree q (0,1,2):
//   1. Commutativity:            A(T_2) A(T_3) == A(T_3) A(T_2).
//   2. Recursion-vs-direct:      T_p_pow_direct(2,2) == A(T_2)^2 - 2^{n+1} I
//      (both sides are the FULL T_4; this is the corrected O9 identity that
//      only holds once T_p_pow_direct denotes the full, not primitive,
//      operator -- see the S9 finalization note above).
//   3. Multiplicativity:         T_n(6) == A(T_2) A(T_3).
//   4. Diamond:                  T_pp(p) == p^n I, for p in {2,3}.
//   5. Decomposition witness:    T_p_pow_direct(2,2) ==
//                                 T_p_pow_primitive_direct(2,2) + 2^n I.
//
// -----------------------------------------------------------------------
// O10 -- bad prime p == ell, gated per plan amendment A-F1 (the Hecke
// recursion / algebra-homomorphism property is contract-backed only at
// ell >= 5, [C, obl:algebra] hypothesis (H3)):
//   * p = ell = 5 (ell >= 5, contract-backed) -- HARD asserts, n in
//     {0,2,10}: the pipeline completes (U_5 = T_p(5) is well-defined, no
//     throw); T_p_pow(5,2) == T_p(5)^2 (recursion route: trivially true by
//     construction since the recursion's scalar 5^{n+1} mod 5 == 0, but
//     asserted anyway as a basic sanity check); T_p_pow_direct(5,2) ==
//     T_p(5)^2 (the GENUINE independent check: T_p_pow_direct(5,2) is
//     assembled from a FRESH HeckeCosetTables(5,2,...) pipeline run at
//     N = 25, entirely disjoint from the N = 5 pipeline reused for T_p(5)^2
//     -- so this equality is a non-vacuous confirmation of
//     [C, cor:ulrecursion]'s U_{25} = U_5^2); and rank(U_5^r) is
//     non-increasing in r and stationary (constant) by r = d_q, for every
//     degree q, a basic linear-algebra fact about powers of a fixed
//     endomorphism that must hold for the pipeline's own output to be a
//     well-formed matrix in the first place.
//   * p = ell = 2 (required) and p = ell = 3 (included for extra coverage;
//     both treated identically per A-F1), n in {0,2,10}: the Hecke
//     recursion is NOT contract-backed at these ell ([C, obl:algebra]
//     requires ell >= 5). HARD-asserts ONLY that the pipeline completes
//     (constructs the model, computes the base cohomology, and evaluates
//     T_p / T_p_pow / T_p_pow_direct without throwing) and that every
//     returned matrix is well-typed (square, dimension == dim H^q(K,R;L),
//     modulus == ell). The analogous U_{ell^2} == U_ell^2 identity (both
//     via the recursion route, which is tautologically true given the
//     scalar-== 0 construction, and via the independent T_p_pow_direct
//     route, which is the substantive comparison) and the rank(U_ell^r)
//     monotonicity/stabilization are RECORDED (printed to stdout) but never
//     asserted, exactly per amendment A-F1.
//
// Independence of the expected values: every identity above is derived from
// the classical abstract Hecke algebra relations cited by
// [C, thm:relations]'s proof (Shimura, Introduction to the Arithmetic Theory
// of Automorphic Functions, Thm 3.24 / Ch. 3; Ash-Stevens Sec 1):
// T(1,p)T(1,p^r) = T(1,p^{r+1}) + p T(p,p) T(1,p^{r-1}), commutativity of the
// local Hecke algebras, and T(p,p) = p^n (the trivial diamond character at
// level 1, [C, prop:diamond]) -- specialized to r = 1, 2 and to p in
// {2,3,5}. Every scalar (p^{n+1} mod ell, p^n mod ell) is recomputed here
// directly from ModArithmetic::mod_pow, not read off HeckeOperatorFamily's
// own internal recursion_scalar()/T_pp() implementation.
//
// Run: ./cohomology_app HeckeRelationsValidation (0 == success).
uint32_t HeckeRelationsValidation();

#endif // HECKE_RELATIONS_VALIDATION_H
