// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_REDUCTION_VALIDATION_H
#define HECKE_REDUCTION_VALIDATION_H

#include <cstdint>

// ===========================================================================
// HeckeReductionValidation -- E8's validation ([X] Sec. 5, the name it fixes).
//
// hecke-extensions slot 8, WP-1 (E8 item 2) ships its ELIMINATION ARMS, as
// src/ModArithmetic/ENGINE_API.md Sec. 13.9 specifies them; the
// pipeline, reduction and Bockstein arms belong to E8 items 3-4 (WP-3/WP-4).
//
//  (R) Ring: ModCtx::prime_power at (ell, k) in {(2,2), (2,3), (3,2), (5,2),
//      (5,3), (691,2)} (u32), (134217757, 2) (u64, Montgomery) and (2, 40)
//      (u64, U64_Wide -- the Sec. 13.2 even-modulus route): the declared ring
//      members, mul / add / sub / reduce against ModArithmetic on 10^6
//      fixed-seed samples, inv on units and the named error on the non-units
//      (every non-unit class exhaustively where ell^k <= 1000, sampled per
//      valuation otherwise), valuation / unit_part / divide_exact / ell_pow
//      against integer arithmetic, prime_power(ell, 1) and overflow rejected,
//      init() resetting the ring (review m5); EVERY Sec. 13.3 entry point
//      throwing logic_error on a composite context and EVERY Sec. 13.6 entry
//      point on a field context; the Sec. 13.7 replay check (a field program
//      never replays on a composite binding, nor the reverse; same-ring
//      replays still run and are exact).
//  (H) Howell: elim::howell == ref::howell entrywise over Z/4, Z/8, Z/9,
//      Z/25, Z/125, Z/691^2, Z/134217757^2 and Z/2^40 on fixed-seed random
//      and structured matrices (zero, identity, ell-multiples, closure-type
//      rows, rank-deficient, tall, wide); conditions 1-3 of Sec. 13.5 checked
//      directly; uniqueness under row permutation, left multiplication by a
//      unimodular matrix and appending rows of the span; the field path:
//      ref::howell(ell, 1, M) == ref::rref and == engine::elim::rref's
//      non-zero rows.
//  (D) Derived: kernel_module (A x = 0 row by row, rank-nullity, Howell),
//      image_module (== howell(A^T), mutual membership through
//      solve_multi_rhs_module), solve_multi_rhs_module (B X = Y exactly, X
//      reduced modulo ker B, independent of the preimage, the named throw at
//      the SMALLEST failing column), invariant_factors (== ref::
//      invariant_factors, sum e_i == the length difference), and review M1's
//      arm in Sec. 13.6's column convention: invariant_factors(Z/4,
//      [[2],[1]]) = (2) while image_module([[2],[1]])'s pivot valuations are
//      (1, 1) (and howell([[2],[1]]) has ONE pivot of valuation 0).
//
// Seeds and per-arm check counts are printed.  `standard` tier (target
// <= 10 s).  Returns the number of failing checks (0 = pass).
// ===========================================================================
uint32_t HeckeReductionValidation();

#endif // HECKE_REDUCTION_VALIDATION_H
