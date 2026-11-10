// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_MP3B_LEVEL_ARMS_H
#define ORBIFOLD_MP_MP3B_LEVEL_ARMS_H

// ===========================================================================
// (hecke-extensions slot 22, MP-3b) The MP-3b arm of OrbifoldMPLevelConeValidation
// (MP3b_predictions.md P7; the F-MP3 audit's F5 coverage gaps): the sequence of a
// pair of groups ([C, prop:mprescor] (3)) on the MP model
//   * over Z/l^k: G0(4) < G0(2) over Z/4 (n = 0, 1), G0(9) < G0(3) over Z/9 and the
//     Hida-tower step G0(25) < G0(5) over Z/25 (n = 0);
//   * on R_Z: Gamma^2 < Gamma (F_2), G0(4) < G0(2) (F_3 n = 2, Z/4 n = 1);
//   * with -I not in A: G1(8) < G1(4) (F_2 n = 0, F_3 n = 1), where the forget-coset
//     label u = -I is reached (gated, >= 5) and a negative control shows it load-bearing.
// Per cell: H^q on C_I and on C(S_Gamma; Coind) (Phi^* an isomorphism), the nerve of every
// subgroup and its Shapiro comparison, res / cor (Gamma-maps, cochain maps, cor o res = [A:B]),
// the length of the image of res on H^q (MP and C_I), the cone of res (MP and C_I) and its
// LES, the forget-coset map -- all against the frozen mp3b_oracle.out (MP3bOracle.inc).
// Read-only: no env hook, setting, golden object or shipped value.
// ===========================================================================

#include <cstdint>

#include "Comparison.h"
#include "MP2Common.h"
#include "Models.h"
#include "Nerve.h"
#include "../../Integral/Model/IsotropyModel.h"
#include "../../Integral/Model/PeriodicityClass.h"

namespace orbifold_mp {

// SG / Phi: the M7 nerve S_<=3 and its certified Phi (degree 2) of the caller.  *cells / *agree: the oracle cells compared.
void RunMP3bLevelArms(MPTally& T, const MPModels& MM, const integral_model::IsotropyModel& CI, const integral_model::PeriodicityClass& P,
                      const InternedMPSet& SG, const PhiLift& Phi, uint32_t* cells, uint32_t* agree);

}  // namespace orbifold_mp

#endif
