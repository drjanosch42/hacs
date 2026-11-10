// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef BLOCK_FLAT_MATRIX_VALIDATION_H
#define BLOCK_FLAT_MATRIX_VALIDATION_H

#include <cstdint> // For uint32_t
#include <vector>  // For std::vector in helper function declarations (if exposed)
#include <utility> // For std::pair in helper function declarations (if exposed)

/**
 * @brief Validates the surviving `BlockFlatMatrixT` surface -- the LEGACY ARM
 *        of the block model (matrix-engine Phase 3; see the banner in
 *        BlockFlatMatrix.h and the one at the head of the .cpp).
 *
 * Every check is ENTRYWISE against the independent `FlatMatrix` implementation
 * of the same operation, never by span (plan Sec. D5b).  The block model itself
 * is validated by BlockMatValidation / BlockEliminationValidation on
 * `engine::BlockMat`; what is left here is only what `ENGINE_BACKEND=legacy`
 * still runs.
 *
 * @param verbosity 0: summary and failures only.
 *                  1: also a line per section.
 *                  2: also a line per individual check.
 * @return 0 if all checks pass, 1 otherwise.
 */
uint32_t BlockFlatMatrixValidation(int verbosity);

#endif // BLOCK_FLAT_MATRIX_VALIDATION_H
// #pragma once // Use either #ifndef/#define/#endif guards or #pragma once
