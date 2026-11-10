// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_ISOTROPY_MODEL_ELL_STRICT_REFINED_H
#define HECKE_ISOTROPY_MODEL_ELL_STRICT_REFINED_H

// hecke-extensions slot 18, WP-7 (upper core): two more arms of
// HeckeIsotropyModelEllValidation (no new name), called at its tail after the
// slot-17 arms.  [C] = docs/latex/gl2_contract.tex v1.18.0-pre.1, sec:isomodelell
// code map items (v), (vi) and the checkable item IL5 (read with referee F1's
// repair M-1: the hypothesis "Iso^(ell)(Y') is an ell-local isotropy model over
// Y'" is certified on every refined model, the strict identity is a certificate).
//
//   (IL5)  the strict lift tau' on Iso^(ell)(Y') of a synthetic refined descriptor
//          (Model/RefinedStrictLift.h), ell in {3, 2}, p in {2, 3, 5}: its coset
//          transport T_p^{Y'} against the transported T'_p = Psi_Y^* T_p Phi_Y^*
//          (the comparison of the models over Y' and Xbulk) -- equal on H^q
//          (their difference maps cocycles into coboundaries), equal image
//          invariant factors of T - a, a in {0, 1, -1}, over Z/ell^e, e <= 3, and
//          equal characteristic polynomials over F_ell, all also against T_p on
//          the free model over Xbulk; a control cell with -I |-> -1; the
//          negative control (the un-averaged values).
//   (IL3Q) [C, IL3] beyond F_3: over F_9 (the F_q code path: Ind_{Gamma_0(N)}
//          Sym^n through the shipped twisted-monomial route with the trivial
//          character) and over Z/9 in lengths against the shipped level-N
//          pipeline's flat-coordinate route (coefficient_ring = Z/ell^2).
//
// Env: HECKE_ISOMODEL_ELL_FULL=1 widens both grids (the slot-17 hook).

#include "HeckeIsotropyModelEllOpenHalves.h"

IsoEllArmTally HeckeIsotropyModelEllStrictRefined(const integral_model::IsotropyModel& desc,
                                                  const integral_model::LocalModel& freeM,
                                                  const integral_model::LocalModel& iso3,
                                                  const std::vector<std::unique_ptr<integral_model::StrictLiftBundle>>& bundles);

#endif
