// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_REFINEMENT_VALIDATION_H
#define ORBIFOLD_MP_REFINEMENT_VALIDATION_H

#include <cstdint>

// ===========================================================================
// (hecke-extensions slot 22, MP-2) OrbifoldMPRefinementValidation (standard).
//
// The refinement map u : S(R_Z) -> S(M7) of MP sets ([MP] Sec. 7.1, Thm 7.3 (3);
// Refinement.h), gate values MP_predictions.md "MP-2 predictions" P4.6:
//   * P4.6a labels h_s~ from the subdivision record, containment decided for
//     h and h.(-I);
//   * P4.6b every string of S_<=3(R_Z) maps into S(M7);
//   * P4.6c u^* a cochain map on the matrices (n <= 2, F_2, F_3, Z/4);
//   * P4.6d u^* o Phi^*_M7 an isomorphism H^q(C_I) -> H^q(MP(R_Z)), q <= 2,
//     ell in {2,3,5}, n in {0,2,4}, equal on cohomology to Phi^*_R_Z -- so u^*
//     is an isomorphism on H^{<=2} (Phi^*_M7 is one, P4.5c);
//   * P4.6e label independence (h.(-I) everywhere / on one simplex);
//   * P4.6f NC: a label off by T fails containment and leaves S(M7).
// Line: ORBIFOLD_MP OrbifoldMPRefinementValidation checks=<k> failing=0.
// model=mp.  Read-only; no golden object, no setting, digest-neutral.
// ===========================================================================
uint32_t OrbifoldMPRefinementValidation();

#endif
