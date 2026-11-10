// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_DETERMINISM_VALIDATION_H
#define HECKE_EXT_DETERMINISM_VALIDATION_H

// hecke-extensions slot 5, WP-2 / E2 cont. -- the extension RECORD's
// determinism gate ([X] Sec. 5 `HeckeExtDeterminismValidation`, Sec. 2.4 risk
// R10).  Returns 0 on success, else the number of failing arms.
//
// It lives HERE and not in HeckeExt/ because installing an engine thread pool
// needs src/Engine/, which check_heckeext_includes.sh rule 2 forbids
// inside the layer; rule 1 admits this *Validation.* TU to include HeckeExt/.
// Registered in ValidationRegistry.cpp.

#include <cstdint>

uint32_t HeckeExtDeterminismValidation();

#endif // HECKE_EXT_DETERMINISM_VALIDATION_H
