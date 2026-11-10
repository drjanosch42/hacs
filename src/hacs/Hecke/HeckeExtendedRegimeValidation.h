// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXTENDED_REGIME_VALIDATION_H
#define HECKE_EXTENDED_REGIME_VALIDATION_H

#include <cstdint>

// Stage S13 -- extended coefficient-regime validation (ell < 2^64).
//
// Runs the FULL integral-Hecke pipeline (relations -> EquivariantCohomologyOnK
// -> HeckeCosetTables -> HeckeTauTable -> HeckeOperatorOnK -> charpoly) over the
// 64-bit arithmetic backend ([C, thm:backend]) at two verified large primes:
//   ell_1 = 8589934627           (prime, ~2^34, exercises mod_mul64)
//   ell_2 = 2305843009213693951  (Mersenne 2^61 - 1, prime, ~2^61)
// for n in {0, 2, 10}, asserting the identities that hold at any ell >= 5:
//   * O1:   A^{(0)}(T_p) = (p+1) mod ell; dim H^0 = 1, dim H^{1,2} = 0 at n=0.
//   * dims: dim H^1 = 2*dim S_{n+2} + 1  (n=2 -> 1, n=10 -> 3); dim H^2 = 0.
//   * charpoly (integral identities reduced mod ell):
//       weight 12 (n=10): charpoly(A^{(1)}(T_2)) = (X+24)^2 (X-2049) mod ell,
//                         charpoly(A^{(1)}(T_3)) = (X-252)^2 (X-177148) mod ell.
//       weight 4  (n=2):  A^{(1)}(T_p) = Eisenstein eigenvalue 1 + p^{n+1}.
//   * The two-route bit-identity (fused vs [C, eq:heckechain]) and the tau
//     certification (obl:tau) still hold at 64-bit ell (both are asserted
//     inside HeckeOperatorOnK::compute() / the HeckeTauTable ctor -- a throw
//     there is a failure here).
// Primality of both primes is re-verified in-test via a deterministic
// Miller-Rabin. Returns 0 on success (invariant I9).
uint32_t HeckeExtendedRegimeValidation();

#endif // HECKE_EXTENDED_REGIME_VALIDATION_H
