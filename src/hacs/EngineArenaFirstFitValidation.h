// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineArenaFirstFitValidation.h -- hecke-extensions slot 9, WP-7 (lever L3).
#pragma once
#include <cstdint>

// The tape scheduler's O(log L) arena first-fit (Passes.cpp, ArenaFirstFit)
// against an independent linear-scan re-derivation of the placement rule, on
// randomized programs: every temp's arena_offset and the arena size equal the
// reference's, and the arena run equals the owned-storage run entrywise.
uint32_t EngineArenaFirstFitValidation();
