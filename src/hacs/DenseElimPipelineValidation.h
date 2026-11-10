// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// DenseElimPipelineValidation -- DenseElim on the REAL pipeline matrices
// (matrix-engine Phase 2, plan D5 / D5b)
// =====================================================================
//
// The companion of DenseElimValidation (ModArithmetic/), which cannot reach
// the SL(2,Z) model.  Here:
//
//   1. The equivariance matrices A_Cq (q = 0, 1, 2) of the 7-vertex model at
//      n_sym = 128, ell = 134217757, are re-assembled through the public API
//      exactly as EquivariantCohomologyOnK builds them (I - M resp. I / -M
//      blocks per live relation; vacuous self-relations dropped), flattened,
//      and eliminated three ways -- engine::elim, RefLinAlg, and the OLD
//      FlatMatrix / BlockFlatMatrix::flat_kernel paths under ScopedLegacy --
//      with RREF, pivots, kernel and image compared ENTRYWISE.  A_C1 is the
//      1677 x 1419 matrix of rank 258 of doc 03 Sec. 10.2, asserted as such.
//   2. A whole compute_cohomology() run at n_sym = 32 is executed once under
//      ScopedLegacy and once on the engine; C_eq / Z_eq / B_eq / H_eq bases,
//      dim H^q and the projectors P_q are compared entrywise.
//
// Returns 0 on success.

#include <cstdint>

uint32_t DenseElimPipelineValidation();
