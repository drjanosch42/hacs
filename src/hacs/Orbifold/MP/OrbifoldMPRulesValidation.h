// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_RULES_VALIDATION_H
#define ORBIFOLD_MP_RULES_VALIDATION_H
// (hecke-extensions slot 22, MP-1a) OrbifoldMPRulesValidation (fast): the decidable Moerdijk--Pronk predicates
// of MP_math_plan.md v0.2 Sec. 3.1 on the five candidate triangulations M7 (shipped), R_Z (SubdivideEdgeOrbit of the
// arc rho^2 - i, found from the inventory), R_zeta (the three edges at zeta^2, slot 17), R_Z+ (R_Z + the three edge
// orbits from the horocycle to rho, iterated) and R_good (the triangle-centre table of mp_rules.py); the choice-free
// MP set S_k ([MP] Def. 1.5) to k = 3 on M7 and R_Z.  Gates exactly MP_predictions.md P1.1-P1.15:
//   |S_0..S_3| (non-degenerate) 5/127/2541/53143 (M7), 7/165/2927/56349 (R_Z); the mu-form 172/3264 = choice-free on
//   R_Z, 124/2704 vs 132/2800 on M7; (V,E,F,chi); adapted (subgroup form) full and for T^(2), T^(3); good (finite-group
//   criterion, the stronger form beside it); collar; good_fails_at; (SC); the invariant vertex order and its sign bridge
//   to IsotropyModel's ordered chains; the simplicial identities of S_<=3; NC1 (order-only "adapted" accepts a planted
//   non-central stabiliser that the subgroup form rejects); NC2 (a two-vertices-in-one-orbit order is refused).
// Oracle lines: ORACLE mp_rules.py cells=12, ORACLE adapted_form_check.py cells=6 (the frozen outputs in
// docs/implementation/slot22/mp/out_checks.txt and mp/audit/adapted_form_check.out).
// Scratch hook ORBIFOLD_MP_DUMP=<dir> (no-op unless set; <dir> must exist outside the source tree, else REFUSED):
// TSVs of the R_Z descriptor, the predicate inputs and |S_0..S_2| for the Lean cross-checks (MP_impl_plan Sec. 3.3),
// with an EFFECTIVE line.  Read-only otherwise; no golden object, record or setting is touched.
#include <cstdint>

uint32_t OrbifoldMPRulesValidation();

#endif // ORBIFOLD_MP_RULES_VALIDATION_H
