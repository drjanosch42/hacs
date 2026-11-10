// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_ARCHITECTURE_VALIDATION_H
#define HECKE_EXT_ARCHITECTURE_VALIDATION_H

#include <cstdint>

// hecke-extensions phase E1s (WP-3) -- the HeckeExt/ include guard as a
// validation ([X] Sec. 5, row `HeckeExtArchitectureValidation`).
//
// `src/hacs/HeckeExt/` is compiled into `cohomology_app` and
// not into a static library ([X] Sec. 1.1), so its one-way dependency has no
// link-time enforcement.  It is a property of the SOURCE TREE instead, and is
// checked by running docs/matrix-engine/tools/check_heckeext_includes.sh over
// the checkout via std::system -- exactly the pattern of
// MatrixRetirementValidation / check_matrix_includes.sh.  The script owns the
// allow-lists with a reason per entry.
//
// A checkout without the script is a SKIP, not a failure (the binary may be run
// outside the tree); a script that reports an offender is a FAILURE.
//
// Returns 0 on success, 1 on a guard violation.
uint32_t HeckeExtArchitectureValidation(void);

#endif // HECKE_EXT_ARCHITECTURE_VALIDATION_H
