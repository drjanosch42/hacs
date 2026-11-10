// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_OPERATOR_SMOKE_VALIDATION_H
#define HECKE_OPERATOR_SMOKE_VALIDATION_H

// Stage S7 -- Definition-of-Done smoke validation for HeckeOperatorOnK
// (the fused pipeline Theta_g and matrix levels (a), (b); see
// HeckeOperatorOnK.h and the plan's stage S7).
//
// Asserts (0 = success):
//   * (p, ell, n) = (2, 5, 0): dim H^0 = 1 and A^{(0)}(T_2) = (3)
//     (= p + 1 mod 5, oracle O1's normalization pin);
//   * (p, ell, n) = (2, 5, 2): dim H^1 = 1 and A^{(1)}(T_2) = (4)
//     (= 9 mod 5, oracle O3); empty degrees (d_q = 0) yield 0 x 0 matrices
//     without crashing;
//   * both evaluation routes (fused apply_Theta vs. the [C, eq:heckechain]
//     direct apply_route2) are BIT-IDENTICAL on full standard bases of
//     C^q(K;V) for q = 0, 1, 2 (in addition to the matrix-level O12(a)
//     assertion that compute() itself enforces);
//   * the runtime assertions (i)/(ii)/(iii) of [C, alg:assembly] were
//     actually executed inside compute() (nonzero check counters in the
//     certificate bundle) with exact zero Theta_eq solve residual (A-N3);
//   * to_json() emits parseable JSON (checked by a small embedded
//     recursive-descent parser) containing the expected values.
//
// NOTE: this is the S7 implementation stage's own smoke test; the
// INDEPENDENT primary oracle battery (O1-O8, O14) is Stage S7V's
// HeckeOracleValidation, written by a separate validation agent.

#include <cstdint>

uint32_t HeckeOperatorSmokeValidation();

#endif // HECKE_OPERATOR_SMOKE_VALIDATION_H
