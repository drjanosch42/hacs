// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#include <cstdint>
// (matrix-engine Phase 4) The consumers on the tape against the direct block
// path (engine::blk on the consumers' own stored objects), entrywise;
// program-cache replay against fresh recording and against a forced fresh
// compile; a changed tau pattern (another Hecke prime, and a mod-ell
// triangle-fill fallback point when the probe finds one) MISSES the cache.
uint32_t EngineTapeConsumerValidation();
