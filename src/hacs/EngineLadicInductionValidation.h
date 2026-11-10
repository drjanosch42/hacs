// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_LADIC_INDUCTION_VALIDATION_H
#define ENGINE_LADIC_INDUCTION_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 11, WP-18b: the (C2) ell-adic digit induction,
// engine::ladic (src/ModArithmetic/LadicInduction.h; ENGINE_API.md
// Sec. 13.11), against the owner's width oracle matrix (Sec. 13.11.7).
// Writes nothing; returns its number of failed checks.  Tier `standard`.
//
// Every comparison is entrywise on the canonical forms, through the
// width-independent base-ell serialisation (LadicMat::ell_digits): the Smith
// exponents, Howell H / pivots / pivot_v, kernel_module, image_module,
// length, invariant_factors (A alone and (G, Rl)), solve_multi_rhs_module
// (a solvable right-hand side and one whose refusal index is compared), the
// cyclic decomposition (exponents, basis, Gamma, U, position) and
// cyclic_coordinates.
//
//   (A) the digit arithmetic: Ring<32>, Ring<64>, Ring<128> against the
//       native product / sum / valuation / exact division at 2-digit moduli,
//       and against the independent base-ell oracle (engine::ref::ladic) at
//       691^20, 5^60, 3^100, 691^12, (2^31-1)^10, (2^31-1)^40 (the lazy-carry
//       folds exercised) and 2^200; unit inverse; the named throw on every
//       valuation class; the layout (m(W, ell), L, conversions, truncation);
//   (B) 2 x 32 vs NATIVE u64: at 691^6, 5^20, 3^30, 7^20, 134217757^2 and
//       2^40 (U64_Wide), two seeds each, on elimination families of the
//       HeckeReductionValidation kind (random mixed valuations, zero,
//       identity, ell-multiples, the closure example, rank-deficient
//       products, tall and wide, planted Smith forms); the same inputs also
//       at 1 x 64 and 1 x 128 (the L = 1 dispatch to Sec. 13.5 and its
//       force_induction run), level_exp = 1 and the digit exponents m' < m
//       (m' = 1: L = K levels): ALL equal to the native record; plus 1 x 32
//       vs native u32 below 2^32 (5^13, 691^3);
//   (C) cross-width beyond the words: 691^12, 5^40, 691^20, 5^60,
//       (2^31-1)^12, 3^100, 2^200: w = 32 vs w = 64 vs w = 128 (odd ell) and
//       their m' / level_exp variants mutually equal; the naive base-ell
//       oracle (Howell, kernel_module, Smith exponents read directly on A)
//       equal; planted elementary divisors recovered;
//   (D) the Smith certificate on every arm: U A V = diag(ell^s) (+) 0 exactly
//       over Z/ell^K, U and V unimodular (full rank mod ell), T's shape;
//   (E) truncation: truncate(k') of the input equals the native run at k'
//       (every canonical form), s(A mod ell^k') = {s < k'}, and the Howell /
//       image forms at K truncated and re-canonicalised at k' equal the
//       native forms at k';
//   (F) the pipeline's own d^q: EquivariantCohomologyOnK over Z/691^6 at
//       (10, 691) and Z/5^20 at (30, 5) (u64): at 2 x 32 digits,
//       kernel_module(X_q) == Zc, cyclic_decomposition(Zc, X_{q-1}) == the
//       stored decomposition, invariant_factors(X_q) == dK, rank_K == r_K(q),
//       stopping_rule == ladic_type_certified();
//   (G) the w = 128 columns (WP-18a's ModCtx<u128>): 4 x 32 and 2 x 64 vs
//       NATIVE u128 at 691^12, 5^40, 3^76, 7^40, 134217757^4; 3 x 32 vs
//       2 x 64 vs 1 x 128 vs native u128 at 691^9, 5^35.
//
// LADIC_INDUCTION_PROBE_N128=1 (read-only, not an emitter): adds an
// indicative timing of smith_lift / kernel_module / cyclic_decomposition on
// the d^1 of (128, 5) over Z/5^26 at L in {2, 4} digits (no verdict).
uint32_t EngineLadicInductionValidation();

#endif
