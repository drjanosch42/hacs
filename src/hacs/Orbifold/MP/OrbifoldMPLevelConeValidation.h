// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_LEVEL_CONE_VALIDATION_H
#define ORBIFOLD_MP_LEVEL_CONE_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 22, MP-3; MP_impl_plan.md v0.2.1 Sec. 2.5 / 2.6) OrbifoldMPLevelConeValidation (standard): the
// sequence of a PAIR OF GROUPS on the MP model for the five LES-0 (iv) pairs G0(4) < G0(2), G0(9) < G0(3), G1(4) < G0(4),
// Gamma^2 < Gamma, Gamma^3 < Gamma at ell in {2, 3} ([C, prop:mprescor]): the MP sets S_B of the subgroups (B-descriptors on
// M7), Shapiro on one F' (Coind in the coset-representative form = LES-1's M_{A,1}), the Shapiro comparison
// C(S_Gamma; Coind_B) -> C(S_B), res / cor (= LES-1's alpha / omega; cor o res = [A:B] on the nose), the forget-coset
// morphism S_B -> S_A with its NON-CENTRAL label factor, the cone of res and its long exact sequence, against the frozen
// mp3_oracle.out (permutation-form Shapiro + Mayer--Vietoris) and C_I with the induced coefficients.  NCs: the label factor
// dropped, a wrong coset representative, cor without its rho-blocks, a label off by T.  Gate values:
// docs/implementation/slot22/MP3_predictions.md P6 (committed before the first run).  Read-only.
uint32_t OrbifoldMPLevelConeValidation();

#endif
