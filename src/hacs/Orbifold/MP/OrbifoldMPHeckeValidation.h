// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_HECKE_VALIDATION_H
#define ORBIFOLD_MP_HECKE_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 22, MP-4; MP_impl_plan.md v0.2.1 Sec. 2.5 / 2.6) OrbifoldMPHeckeValidation (standard): Hecke
// operators T_p, p in {2, 3, 5, 7}, ON THE MP MODEL ([C, rmk:mphecke]): the twisted chain datum tau on N Z[F'] built by the
// MP model's own filling (MPFill.h; no shipped tau table, no Iso_.), certified; T_p = cor o conj_g o res with the label
// factor rho_n(g) of the g-twist, a cochain map, factorised through Shapiro coordinates; on H^0, H^1 (and H^2 at
// ell in {2, 3}) over F_ell, ell in {2, 3, 5, 7}, and over Z/ell^k: T_p o Phi^* = Phi^* o Theta^_g (the shipped C_I operator,
// IntegralHecke / StrictLift); the intrinsic matrices on H^0, H^1 against the independent oracle mp4_oracle.out (amalgam
// cocycles; tau / Eichler--Shimura); the shipped Bredon pipeline's T_p transported by eps^* (an isomorphism at ell in {5, 7},
// on its image at ell in {2, 3}); on M7 and R_Z.  NCs: the label factor dropped, a label factor mismatched with the datum (gT for g), a
// perturbed tau.  Gate values: docs/implementation/slot22/MP4_predictions.md P7 (committed before the first run).  Read-only.
uint32_t OrbifoldMPHeckeValidation();

#endif
