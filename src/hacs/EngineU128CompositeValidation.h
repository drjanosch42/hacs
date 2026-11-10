// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_U128_COMPOSITE_VALIDATION_H
#define ENGINE_U128_COMPOSITE_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 11, WP-18a: the native u128 engine layer
// (src/ModArithmetic/ENGINE_API.md Sec. 13.12, step 1 of
// Sec. 13.12.8) and the digit-width abstraction engine::DigitWord<W>
// (DigitWord.h), gated by the Sec. 13.12.4 oracle rule at the ENGINE level.
// Writes nothing; returns its number of failed checks.  Tier: standard.
//
//   (A) arithmetic -- ModCtx<u128>'s Montgomery-128 primitives (mul, the
//       mul_prep / mul_prepped pair, reduce, add / sub / neg, the Hensel inv)
//       against RefLinAlg's naive128 arithmetic (double-and-add mulmod,
//       sign-tracked extended Euclid: an independent route) on >= 10^6
//       fixed-seed samples plus boundary values (0, 1, 2, p-1, p-2, 2^64 +- j,
//       2^63, 2^127 - 1 reduced), at 4294967291^2 (2^64 - small),
//       4294967311^2 (2^64 + small), 691^12, 3^80 (~2^126.8), and below 2^64
//       at 5^27 and 691^6 also against ModCtx<uint64_t>; k_max TIGHT (a block
//       of k_max maximal products satisfies REDC's precondition T < p*2^128,
//       k_max + 1 does not; dot_delayed accepts k_max and refuses k_max + 1);
//       inv throws on every valuation class v = 1 .. k-1 and on 0; the
//       construction refuses ell = 2, k = 1, ell^k >= 2^127; the u128 kernels
//       (gemm across k-blocks, gemm_nt, gemv, dot_delayed, axpy, scal) equal
//       ref::gemm on naive128;
//   (E) the oracle rule below 2^64: u128 == u32 / u64 ENTRYWISE (Howell H,
//       pivots, pivot_v, kernel_module, image_module, length,
//       solve_multi_rhs_module incl. the refusal text, invariant_factors,
//       cyclic_decomposition's exponents / basis / Gamma / U / position,
//       cyclic_coordinates) on HeckeReductionValidation's elimination-arm
//       matrices -- its (H) and (M) case generators replayed with its seeds at
//       its odd moduli (3^2, 5^2, 5^3, 691^2 in u32; 134217757^2 and
//       4294967291^2 in u64) -- plus random cases at 5^27, 691^6 and 3^40 (u64);
//   (B) beyond 2^64 (no native oracle): elim::howell<u128> == ref::howell on
//       naive128, invariant_factors == ref::invariant_factors, the kernel /
//       solve / rank-nullity identities by ref::gemm, and PLANTED elementary
//       divisors (U diag(ell^v) V: the exponents are the planted v's) at
//       4294967311^2, 5^30, 7^40, 691^12, 3^80;
//   (K) the block model's composite surface: BlockMat<u128> through the
//       ModCtx overloads, blk::gemm / add / scale against kernels::gemm and
//       ref::gemm, the context carried (is_field() false), BlockPool's content
//       hash over both 64-bit halves (entries differing only in the high limb
//       are distinct, equal content deduplicated);
//   (W) DigitWord<W>, W in {u32, u64, u128}: digit_exponent m(W, ell) against
//       the predicted table, available(), mac_split / split against exact
//       u256 arithmetic, ctx(ell, m) the composite context of the width, and
//       the three widths' word entry points equal entrywise at 691^3 (a
//       modulus all three carry).
uint32_t EngineU128CompositeValidation();

#endif
