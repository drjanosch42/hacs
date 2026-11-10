// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_CARRIES_S11_VALIDATION_H
#define ENGINE_CARRIES_S11_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 11, WP-12: the two engine carries of the slot-10 WP-9
// report (docs/implementation/slot10/WP-9_report.md Sec. 4), both
// value-preserving by construction.  Each validation writes nothing and
// returns its number of failed checks.
//
// ModCtxMontgomeryPairValidation (fast) -- F-1: ModCtx's reused-operand pair
// mul_prep / mul_prepped (one Montgomery-64 REDC per product in the
// U64_Montgomery regime) and its use in the composite (Howell / Smith) row
// updates of DenseElim.cpp (ENGINE_API.md Sec. 13.5, 13.12.5).
//   (A) arithmetic: mul_prepped(x, mul_prep(w)) == mul(x, w) == the old
//       exact (u128)x*w % p, on fixed-seed random samples and on the boundary
//       values {0, 1, 2, p-2, p-1}, at U64_Montgomery moduli (odd prime powers
//       691^6, 5^27, 3^39, 134217757^2, the primes 2^32+15 and 2^63-25) and at
//       U64_Wide moduli (2^40, 2^63, 3^40, 2^64-59), where the pair is mul
//       itself; the regime flag of each modulus is checked; u32 moduli: the
//       pair is mul (the u32 code path is unchanged);
//   (B) the composite entry points (howell, kernel_module, image_module,
//       solve_multi_rhs_module, length, invariant_factors, cyclic_decomposition,
//       cyclic_coordinates) on random matrices with mixed valuations, ENTRYWISE
//       equal to the SAME context with the Montgomery flag cleared -- which is
//       exactly the pre-package %-path, since mul_prepped falls back to mul --
//       and the Howell forms equal ref::howell (the naive oracle, independent
//       arithmetic); at every u64 modulus of (A) that is a prime power;
//   (C) the oracle rule in miniature: at moduli < 2^32 (691^3, 5^13, 7^11)
//       the u64 instantiation (Montgomery) equals the u32 one (unchanged code)
//       entrywise on the same inputs;
//   (D) the pipeline: EquivariantCohomologyOnK over Z/ell^k at (10, 691, 3),
//       (10, 5, 3) and (22, 7, 3) -- u64 (Montgomery row updates) equal to u32
//       entrywise on every degree's C, X, Zc, H, cyclic basis / Gamma / U and
//       the invariant factors.
//
// CoefficientRingKthRootValidation (fast) -- F-2: ExactKthRoot's primality by
// deterministic Miller-Rabin (IsPrimeU64), and the ring carried as (ell, k)
// (LocalSystemSimplicialComplexT::declare_coefficient_ring, read by
// EquivariantCohomologyOnK).
//   (A) IsPrimeU64 == trial division on every n < 200000, on 20000 random
//       32-bit n, on the strong pseudoprimes to the small bases, Carmichael
//       numbers, squares of large primes, and known primes near 2^64;
//   (B) ExactKthRoot == the slot-8 trial-division implementation on every
//       perfect power ell^k < 2^64 of the primes ell < 2000 (and k up to the
//       word), on composite-root powers (625 = 25^2, 6^k, 15^k, 4^k), on
//       non-powers and the edge inputs; the ENGINE_API.md Sec. 13.12.3 worst
//       case (k = 2 at the largest prime below 2^32) returns in O(log);
//   (C) declare_coefficient_ring: refuses k < 2, ell < 2 and ell^k != the
//       modulus (logic_error); records (ell, k);
//   (D) EquivariantCohomologyOnK on a DECLARED system equals the undeclared
//       one entrywise (the module route's stored objects, invariant factors,
//       lengths) at (10, 5, 2), (10, 691, 2) and (16, 7, 3); a declaration
//       whose k differs from the setting refuses exactly as a modulus that is
//       not ell^k does.
uint32_t ModCtxMontgomeryPairValidation();
uint32_t CoefficientRingKthRootValidation();

#endif
