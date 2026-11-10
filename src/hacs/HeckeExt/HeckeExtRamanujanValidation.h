// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_RAMANUJAN_VALIDATION_H
#define HECKE_EXT_RAMANUJAN_VALIDATION_H

#include <cstdint>

// ===========================================================================
// hecke-extensions phase E2b (slot 4, WP-2) -- [E] Claim 8.1 (`cl:ramanujan`)
// through `HeckeExt/HeckeExtDriver`, the first acceptance test of E2.
//
// Returns 0 on success, the number of failed checks otherwise.  Registered as
// `HeckeExtRamanujanValidation`, tier `standard`.
// ===========================================================================
uint32_t HeckeExtRamanujanValidation();

#endif // HECKE_EXT_RAMANUJAN_VALIDATION_H
