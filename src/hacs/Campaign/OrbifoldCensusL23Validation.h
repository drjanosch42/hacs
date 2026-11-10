// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// (hecke-extensions slot 22, L23-3) the census validations of the orbifold section at ell = 2, 3 (census_l23_plan.md v0.3
// Sec. 6 L23-3, L23_1_audit.md Sec. 7; predictions docs/implementation/slot22/L23_3_predictions.md):
//   OrbifoldCensusL23Validation (standard): the C++ q-expansion sum-rule oracle (Orbifold/L23/SumRuleOracle.h) against all 200
//     frozen Bin3SumRules.inc rows; the amalgam layers (l23::h1_layers) against the independent prototype layers_proto.py and the
//     MV oracle mv_numpy.py (L233Oracle.inc) and against the [R] closed forms; orbifold_cell/v1 cells n = 0 .. 24 at both primes
//     with the new (S) / (B/!/L) gates, the MV dimensions, the schema and the defaults; negative controls.
//   OrbifoldCensusL23SlowValidation (slow): the reader and the oracle on every even n <= 200 at both primes (every sum rule,
//     the iota halves at ell = 3); the C_I (lat) reading of ed_tp / ed_eis / ed_tell (IntegralHecke, uint32_t, K' = 31 / 19)
//     against the reader on sampled n; the layers against the MV oracle at n = 50, 100, 150, 200; the oracle at n = 240 ... 700
//     against the four L23-4 calibration records' sums.
// Both return the failure count (0 on success); last lines ORBIFOLD_CENSUS_L23 ... / ORBIFOLD_CENSUS_L23_SLOW ... verdict=OK.
#include <cstdint>
uint32_t OrbifoldCensusL23Validation();
uint32_t OrbifoldCensusL23SlowValidation();
