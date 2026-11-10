// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// (hecke-extensions slot 22, BIN-1) BinaryF2Validation: engine::binary (BitMatrix, M4RI rref /
// echelon / kernel, M4RM product) and the opt-in ENGINE_BINARY_F2 route of engine::elim are bit
// for bit the engine at p = 2 -- rank, pivots, RREF, kernel basis, product -- on 500 random and
// the real A_Cq relation matrices of the 7-vertex SL_2(Z) model mod 2 (>= 100), on every
// available row path; a whole compute_cohomology() at ell = 2 with the route on equals the route
// off; and one BINARY_F2_TIMING line per size (1024, 4096; reported, never gated).
#include <cstdint>

uint32_t BinaryF2Validation();
