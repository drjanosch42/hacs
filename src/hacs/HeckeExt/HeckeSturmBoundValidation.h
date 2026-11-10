// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_STURM_BOUND_VALIDATION_H
#define HECKE_EXT_STURM_BOUND_VALIDATION_H

#include <cstdint>

// hecke-extensions phase E1s (WP-3) -- the validation of HeckeExt/HeckeSturmBound.
//
// INDEPENDENCE.  Every oracle in this file is independent of the unit it
// checks:
//   * sigma_3 / sigma_5 are recomputed as EXACT INTEGERS by a naive divisor
//     loop and reduced afterwards, not by the unit's modular divisor sieve;
//   * Delta is pinned against the published tau(2..7) = -24, 252, -1472, 4830,
//     -6048, -16744 and, wherever 1728 is invertible, against the independent
//     identity E_4^3 - E_6^2 = 1728 Delta;
//   * dim M_k is pinned against #{(a, b) >= 0 : 4a + 6b = k}, not against the
//     floor(k/12) closed form the unit uses;
//   * the bounds B'(k, ell) themselves are pinned against the values recorded
//     in [E] Claim `cl:bprime` and its proof, [E] Claim 1.8's ten Kummer cells
//     and [X] Sec. 2.2 Stage 4(c)'s v1.2 recomputation.
//
// It also gates the resumable-state contract [X] Sec. 1.3.1 R3 -- of which the
// Sturm bound is instance 1 and WP-3 the first implementation: resume
// invariance, the refusal of a lower budget, the no-op resume on a complete
// state and the budget history.
//
// Returns 0 on success, otherwise the number of failing checks.
uint32_t HeckeSturmBoundValidation(void);

#endif // HECKE_EXT_STURM_BOUND_VALIDATION_H
