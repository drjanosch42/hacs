// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#include <cstdint>
// (matrix-engine Phase 4) The golden grid's small / medium cells under
// ENGINE_THREADS in {1, 4, 16, 32} x ENGINE_BATCH in {0, 1}, IN PROCESS
// (ScopedConsumerPool / ScopedPassOptions), every golden object compared
// against the checked-in reference with both tiers gating, and the per-arm
// digests compared with each other.  Returns 0 iff every arm reproduces the
// reference and every digest is the same.
uint32_t EngineDeterminismValidation();
