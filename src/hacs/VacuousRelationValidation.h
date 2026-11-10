// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef VACUOUS_RELATION_VALIDATION_H
#define VACUOUS_RELATION_VALIDATION_H

#include <cstdint>

// ===========================================================================
// matrix-engine Phase 1, item 7.4 -- vacuous-relation short-circuit.
//
// `EquivariantCohomologyOnK` now drops the block rows of A_Cq / A_Zq that come
// from a relation f(s) = M f(s) with M == I (the FIX-negI relations at even
// n_sym).  This validation pins that the short-circuit is EXACTLY
// value-preserving on the SL(2,Z) model, and that the odd-n semantics it must
// not disturb (H^* = 0 for odd n at ell != 2) are intact.
//
// For every (n_sym, ell) in {0..12} x {2, 3, 5, 691, 134217757} it
//   (a) runs the pipeline twice -- short-circuit ON (shipping path) and OFF
//       (the pre-7.4 assembly, reachable through
//       EquivariantCohomologyOptions::set_vacuous_relation_shortcircuit) --
//       and compares dim H^q, and the C_eq / Z_eq / B_eq / H_eq bases and the
//       H_eq projectors ENTRYWISE;
//   (b) checks the relation classification against the table of 02 Sec. 5.4:
//       at even n the model has 6/13, 2/13 and 0/5 surviving block rows in
//       degrees 0, 1, 2, so A_C1 shrinks from 13*d_V to 2*d_V rows (1313 ->
//       202 at n_sym = 100); at odd n with ell != 2 every relation survives;
//       n_sym = 0 is the trivial representation, where rho_0(g) = I for every
//       g and the model keeps exactly its 3/2/0 non-self relations;
//   (c) asserts H^* = 0 for odd n_sym at ell != 2 (Hecke oracle O2).
// ===========================================================================
uint32_t VacuousRelationValidation(void);

#endif // VACUOUS_RELATION_VALIDATION_H
