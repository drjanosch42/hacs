// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_COSET_TABLES_SELF_CHECK_VALIDATION_H
#define HECKE_COSET_TABLES_SELF_CHECK_VALIDATION_H

#include <cstdint>

// Stage S3 Definition of Done: runs HeckeCosetTables::self_check() over the
// parameter grid p_hecke in {2,3,5,7,11}, r in {1,2}, ell in {5,691},
// n_sym in {0,2,10} (docs/implementation/hecke_action_implementation_plan.md,
// "### S3 -- HeckeCosetTables", Definition of Done). Returns the number of
// failing (p_hecke, r, ell, n_sym) combinations (0 == success), matching the
// ValidationFn convention (invariant I9).
uint32_t HeckeCosetTablesSelfCheckValidation();

#endif // HECKE_COSET_TABLES_SELF_CHECK_VALIDATION_H
