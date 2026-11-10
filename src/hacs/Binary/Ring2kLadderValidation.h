// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

#include <cstdint>

// Slot-22 BIN-2 gate (standard): the Z/2^k ladder engine::ring2k (Ring2k.h) on every rung against
// the engine (ModCtx::prime_power(2, k) for k <= 63, engine::ladic beyond), the nf::BigInt reference
// rung R6 and the unblocked references; wraparound fuzz; the adaptive rung; negative controls.
// Prints RING2K_TIMING lines (reported, never gated) and one
//   RING2K_LADDER ... checks=<c> failing=<f> verdict=OK|FAIL
// line.  Returns the number of failing checks.
uint32_t Ring2kLadderValidation();
