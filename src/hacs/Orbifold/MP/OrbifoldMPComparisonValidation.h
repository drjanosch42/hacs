// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_COMPARISON_VALIDATION_H
#define ORBIFOLD_MP_COMPARISON_VALIDATION_H

#include <cstdint>

// ===========================================================================
// (hecke-extensions slot 22, MP-2) OrbifoldMPComparisonValidation (standard).
//
// The comparison maps of the MP model with the shipped models (MP_impl_plan.md
// v0.2 Sec. 2.4 / 2.6; gate values MP_predictions.md P4.3 and "MP-2
// predictions" P4.5, committed before the first run):
//   * P4.5a Phi : Z[F'_.] -> Iso_. certified exactly over Z[Gamma] to degree 3
//     on M7 and R_Z (Comparison.h);
//   * P4.5b Phi^* a cochain map on the matrices (n <= 2, F_2, F_3, Z/4);
//   * P4.5c/d Phi^* an isomorphism H^q(C_I) -> H^q(MP), q <= 1 (ell <= 7,
//     n <= 12), q = 2 (ell <= 5, n <= 3), and over Z/ell^k (ell in {2,3},
//     k in {2,3}, n in {0,2}, q <= 2), against the frozen mv_check.out;
//   * P4.3 / P4.5f-h eps^* = Phi^* o eps_I^* from the SHIPPED Bredon complex
//     (EquivariantCohomologyOnK on the 7-vertex K): an isomorphism at
//     ell in {5,7}; at ell in {2,3} the Bredon defect (rank = Bredon dim, the
//     isotropy edge map); the "if, not iff" witness at ell = 3, odd n;
//   * P4.5i the hybrid (IsotropyModelEll::LocalModel, the periodic-column
//     instance of [MP] Thm 2.6, R2) = C_I = nerve: dims, and the strict
//     comparison's pullback an isomorphism on H^{<=2};
//   * P4.5e the negative control: a perturbed Phi_1 fails certify().
// Lines: ORACLE mv_check.out cells=<k> agree=<k>; ORBIFOLD_MP
// OrbifoldMPComparisonValidation checks=<k> failing=0.  model=mp.  Read-only:
// no env hook, no golden object, no setting, digest-neutral.
// ===========================================================================
uint32_t OrbifoldMPComparisonValidation();

#endif
