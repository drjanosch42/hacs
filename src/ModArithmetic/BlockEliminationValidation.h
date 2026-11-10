// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// BlockEliminationValidation -- block structured elimination vs DenseElim
// vs the legacy BlockFlatMatrix (matrix-engine Phase 3, plan D5 / D5b)
// =====================================================================
//
// ENTRYWISE (never span-only) comparison of engine::blk::rref /
// kernel_basis / image_basis / factor_basis / solve_multi_rhs against
//   (a) engine::elim on the flattened dense matrix, and
//   (b) the legacy BlockFlatMatrix::flat_kernel / flat_image and
//       factor_basis_block / change_of_basis_fm (through
//       engine::elim::ScopedLegacy, i.e. the pre-Phase-2 code),
// on fixed-seed block matrices with the structural kinds the pipeline
// produces (I, alpha*I, -M, I-M, pencils, transposed windows, structural
// zeros), including grids shaped like the real A_Cq / A_Zq matrices
// (13x7, 6x7, 2x11 blocks with d_V in {17, 65, 129} at 18-29 % block
// density), rank-deficient inputs, non-uniform / zero-width partitions, both
// element widths and the regime-boundary primes.  Returns 0 on success.

#include <cstdint>

uint32_t BlockEliminationValidation();

// (S24 VAL-IMPL, VALSTREAM_audit_1030.md row 7) The slow part of the same grid: the "2x11 d=65" grid at the eight non-big
// (prime, width) passes, with the pre-split seeds.  BlockEliminationValidation + this = the pre-split case set.
uint32_t BlockEliminationSlowValidation();
