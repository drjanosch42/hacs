// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef THETA_EQ_SOLVE_VALIDATION_H
#define THETA_EQ_SOLVE_VALIDATION_H

// matrix-engine Phase 1, item 7.1 -- validation of the multi-right-hand-side
// Theta_eq solve (docs/matrix-engine/00-implementation-plan.md, "Phase 1";
// docs/matrix-engine/02-pipeline-consumer-audit.md Sec. 7.1).
//
// HeckeOperatorOnK::compute() used to determine Theta_eq[q] with one
// change_of_basis_fm() call per C^q_eq basis vector, each of which rebuilt and
// re-eliminated the augmented matrix [C_eq | w_j] although its first
// dim C^q_eq columns are identical every time. It now runs ONE augmented RREF
// of [C_eq | Theta*C_eq] via change_of_basis_fm_multi().
//
// Asserts (0 = success):
//   (A) on a grid of (n_sym, ell, p_hecke) points taken from the small/medium
//       part of the golden grid, and for every degree q with dim C^q_eq > 0:
//         * change_of_basis_fm_multi(C_eq, {Theta*c_j}) is BIT-IDENTICAL,
//           entry for entry, to the per-vector change_of_basis_fm(C_eq,
//           Theta*c_j) it replaces (the old code is the reference here);
//         * both agree with the Theta_eq[q] the shipped pipeline published;
//         * the A-N3 residual certificate (invariant C5) re-verified against
//           the UNFACTORED data vanishes identically for every column;
//   (B) on small fixed-seed synthetic systems (independent of the Hecke
//       stack): the multi-RHS solver reproduces the single-RHS coordinates on
//       every right-hand side, handles the 1-column and 0-RHS edge cases, and
//       still REJECTS a right-hand side outside the span of the basis.
//
// Grid: 12 points, n_sym in {0,1,2,4,10,16,22,24,32,48} x ell in
// {3,5,13,691,134217757} x p_hecke in {2,3}; 3122 Theta_eq columns compared.
// Tier `standard` (measured 8.1 s at RelWithDebInfo).

#include <cstdint>

uint32_t ThetaEqSolveValidation();

#endif // THETA_EQ_SOLVE_VALIDATION_H
