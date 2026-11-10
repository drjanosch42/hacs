// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_TAU_TABLE_SELF_CHECK_VALIDATION_H
#define HECKE_TAU_TABLE_SELF_CHECK_VALIDATION_H

#include <cstdint>

// Stage S4 -- HeckeTauTable Definition-of-Done battery: constructs the
// certified tau-table (the constructor already runs the MANDATORY exact
// certification of [C, obl:tau] and throws on failure) and explicitly
// re-asserts certify() == 0 across the DoD grid
//   (p, r) in {(2,1), (3,1), (5,1), (7,1), (2,2)}
//     x ell in {5, 7, 691, 134217757}
//     x n in {0, 2, 10}
// AND the bad cases p = ell in {2, 5} (n in {0, 2, 10}); asserts the p = 2
// table stores exactly 3 * (7 + 11 + 5) = 69 chains; checks the certified
// bound fits the int64 backend tier (2 B^2 < 2^63); exercises the FixedV6
// alternate vertex policy hook (certification only -- NO cross-tau
// assertions anywhere, in particular none for ell in {2, 3}, per the
// contract box after [C, lem:tauexists]); and writes the human-readable
// p = 2 table to docs/implementation/tau_table_p2.txt (best-effort: skipped
// with a notice if the directory is not reachable from the working
// directory; does not affect the validation result).
//
// Returns 0 on success (invariant I9). NOTE: this is the stage's own DoD
// battery; the INDEPENDENT S4V validation (HeckeTauTableValidation,
// re-implementing the checker from [C, obl:tau] alone) is written by the
// separate S4V validation agent.
uint32_t HeckeTauTableSelfCheckValidation();

#endif // HECKE_TAU_TABLE_SELF_CHECK_VALIDATION_H
