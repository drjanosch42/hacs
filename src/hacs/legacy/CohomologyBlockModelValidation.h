// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef COHOMOLOGY_BLOCK_MODEL_VALIDATION_H
#define COHOMOLOGY_BLOCK_MODEL_VALIDATION_H

#include <cstdint>

// ===========================================================================
// CohomologyBlockModelValidation -- matrix-engine Phase 3 Stage B, the
// EquivariantCohomologyOnK migration.
//
// compute_cohomology() runs on the block model now and STORES its result as
// engine::BlockMat objects (C/Z/B/H_eq multi-vectors, the projectors P_q and
// the constraint matrices A_Cq / A_Zq); the FlatMatrix accessors are
// materialised from them.  This validation pins, for every instance of a
// small grid over the SL(2,Z) model -- n_sym in {0..6, 8, 12, 16, 32} (odd n
// included, so the -I relation and H^* = 0 are exercised), ell in {2, 3, 7,
// 134217757, 2^32-5}, both settings of the item-7.4 short-circuit on a
// subset, a relation-free local system (the identity-kernel path) and the
// uint64_t instantiation at ell = 2^61-1 --
//
//   (a) engine arm vs LEGACY arm (EquivariantCohomologyOptions::
//       ScopedLegacyCohomologyPath, i.e. BlockFlatMatrix::flat_kernel +
//       factor_basis_block + the pre-Stage-B flatten/expand loops), ENTRYWISE:
//       dim H^q, the C/Z/B/H_eq cochain lists and the projectors, including
//       the legacy shape conventions (0 x 0 projector when Z^q_eq is empty);
//   (b) on BOTH arms, the block accessors against the FlatMatrix ones: every
//       multi-vector has nbc == 1 and the (N_q, d_V) row partition, and its
//       column t in L1 order (to_flat) IS the row-major N_q x d_V cochain t --
//       the plan-7.20 contract; P_block(q) is dim H^q x dim_q and equals the
//       FlatMatrix projector;
//   (c) algebraic certificates on the block objects: A_Cq * X == 0 for X in
//       {C, Z, B, H}_eq (equivariance), P_q * H_eq == I (the projector is a
//       left inverse on its own basis) and P_q * B_eq == 0 (it kills the
//       coboundaries), A_Zq has k_q * d_V columns.
//
// Returns 0 on success; non-zero = number of failing checks.
// ===========================================================================
uint32_t CohomologyBlockModelValidation();

#endif // COHOMOLOGY_BLOCK_MODEL_VALIDATION_H
