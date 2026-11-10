// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_THETA_BLOCK_VALIDATION_H
#define HECKE_THETA_BLOCK_VALIDATION_H

#include <cstdint>

// ===========================================================================
// HeckeThetaBlockValidation -- matrix-engine Phase 3 Stage B, the
// HeckeOperatorOnK migration.
//
// HeckeOperatorOnK::compute() runs on the block model now: Theta^{(q)} is an
// N_q x N_q engine::BlockMat with the tau-table pattern, the fused route and
// route 2 are two programs compared with blk::equal, and Theta5..Theta10 are
// engine calls (gemm_dense / solve_multi_rhs / gemm).  The FlatMatrix Theta
// list is a lazily materialised shim.  This validation pins, on a grid over
// the SL(2,Z) model -- n_sym in {0, 1, 2, 3, 4, 6, 8, 12} (odd n included),
// ell in {3, 5, 7, 134217757}, p_hecke in {2, 3}, plus the uint64_t
// instantiation at ell = 2^61 - 1 --
//
//   (a) engine arm vs LEGACY arm (HeckeOperatorOptions::
//       ScopedLegacyHeckeOperatorPath, i.e. the pre-Stage-B dense per-cell
//       assembly + change_of_basis_fm_multi), ENTRYWISE: Theta_flat,
//       Theta_eq, A^{(q)} and the complete to_json() text (which carries the
//       certificate counters relation_checks / cocycle_checks /
//       theta_eq_solves / residual);
//   (b) on BOTH arms, Theta_block(q) against get_Theta_flat_matrices()[q]:
//       the grid is the uniform (N_q, d_V) partition on both axes, no stored
//       ref is Zero, every structural non-zero (sigma, sigma_k) is a pair the
//       tau-table names with a non-zero coefficient mod ell, and the block
//       matrix equals the flat one as a mathematical matrix
//       (blk::equal_dense);
//   (c) apply_Theta(q, e_b) reproduces column b of Theta_flat AND
//       apply_route2(q, e_b) on the full standard basis (O12(a) through the
//       public apply API, on the block model);
//   (d) the fused-route pool shares blocks: refs naming the same pool window
//       materialise to the same values, and the number of distinct data
//       windows behind all Theta refs is reported against the total ref
//       count.
//
// Returns 0 on success; non-zero = number of failing checks.
// ===========================================================================
uint32_t HeckeThetaBlockValidation();

#endif // HECKE_THETA_BLOCK_VALIDATION_H
