// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef MATRIX_RETIREMENT_VALIDATION_H
#define MATRIX_RETIREMENT_VALIDATION_H

#include <cstdint>

// ---------------------------------------------------------------------------
// MatrixRetirementValidation (matrix-engine Phase 3)
//
// The Phase 3 exit criterion "zero `Matrix` includes outside MatrixValidation /
// HeckeModuleDecomposition" (docs/matrix-engine/00-implementation-plan.md,
// Phase 3, "Matrix retirement") is a property of the SOURCE TREE, not of any
// computed value, so it is checked by running
// docs/matrix-engine/tools/check_matrix_includes.sh over the tree via
// std::system. That script owns the allow-list and its per-entry reasons;
// removing a file from that list is the unit of progress.
//
// Returns 0 when the script reports OK, 1 when it reports an offending include.
// When the script or the source tree cannot be located (the binary was run from
// an unpacked/installed location, or `bash` is unavailable) it prints a SKIPPED
// notice and returns 0: this is a hygiene check on the checkout, and a missing
// checkout is not a failure of the code under test.
// ---------------------------------------------------------------------------
uint32_t MatrixRetirementValidation(void);

#endif // MATRIX_RETIREMENT_VALIDATION_H
