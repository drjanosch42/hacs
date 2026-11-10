// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_TESSELLATION_SMOKE_VALIDATION_H
#define HECKE_TESSELLATION_SMOKE_VALIDATION_H

#include <cstdint>

// Stage S2 smoke validation (S2 Definition of Done; the INDEPENDENT checks
// live in S2V's HeckeTessellationValidation). Returns 0 on success:
//   1. SL2ZTessellationTables closure fixed point reached and all
//      self-checks pass (validate_closure() == 0);
//   2. canonical_key is idempotent (key(key) = key) and constant on 10^4
//      randomized rewritings (gamma*g, sigma') of the same K^u-simplex
//      (fixed-seed RNG);
//   3. word_in_T_S round-trips 10^4 pseudorandom Gamma-elements (exact
//      rebuild comparison + the (P2) intermediate-entry bound);
//   4. (I3 guard; updated for FIX-negI) the first 8 relations of the
//      refactored GenerateSL2ZRelations_7VertexModel are bit-identical to
//      the pre-S2 construction for (n, ell) in {(0,5), (2,5), (10,691)};
//      the trailing 23 rows are the FIX-negI -I self-relations (one per
//      simplex of K, M == rho_n(-I), covering every simplex exactly once);
//      every relation matrix equals the independently re-derived rho_n(g_r).
uint32_t HeckeTessellationSmokeValidation();

#endif // HECKE_TESSELLATION_SMOKE_VALIDATION_H
