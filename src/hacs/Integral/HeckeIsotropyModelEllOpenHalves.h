// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_ISOTROPY_MODEL_ELL_OPEN_HALVES_H
#define HECKE_ISOTROPY_MODEL_ELL_OPEN_HALVES_H

// hecke-extensions slot 17, WP-7: the open halves of [C, sec:isomodelell]
// (the slot-16 WP-16 carries), as ARMS of HeckeIsotropyModelEllValidation (no
// new registered name; Registry/Slot17/WP7.cpp stays empty):
//   (IL3N) [C, IL3]'s level-N clause: M = Ind_{Gamma_0(N)}^Gamma V_R, built from
//          the SHIPPED hecke::level::Gamma0LevelN table through MonomialOf (no
//          second coset construction), as a CoefficientModule of Iso^(3): the
//          group cohomology (Iso^(3) = Iso entry for entry in lengths), the
//          Bredon row, the closed form H^q(Z/3; M^{-I}), and the shipped
//          level-N pipeline (InducedRelationsLevel + EquivariantCohomologyOnK,
//          [C, def:levelNmodel](a), the cochain level) over F_3:
//          dim H^1(Gamma; M) = dim H^1_eq(level N) + dim H^1(Z/3; M^{-I}),
//          H^0 equal, H^2_eq = 0; odd n (-I |-> -1) gives 0 everywhere;
//   (IL4e) [C, IL4] over Z/3^e, e in {2, 3}: Phi^*, Psi^* cochain maps and
//          quasi-inverse, T'_p = Psi^* T_p Phi^* a cochain map with T' = P T' P,
//          and the invariant factors of H^q, of im(T'_p - a) and of the kernel
//          length, a in {0, 1, -1}, equal to T_p's on Iso (StrictLift), p in
//          {2, 3, 5}; the charpoly where H^q is free of rank <= 7;
//   (PC)   the pair / cone half of [C, prop:isomodelell](5): the restriction to
//          the boundary sub-model commutes with Phi^*, Psi^*, T_p (strict pair
//          datum, (B)); on Cone(res) = C^q (+) C_d^{q-1} the transported
//          T'_cone = Psi^*_cone T_cone Phi^*_cone is a cochain map, and its
//          charpolys (F_3) / image invariant factors (Z/9) on H^q(cone) and on
//          H^q(boundary) equal T_p's on Iso;
//   (REF)  [C, rmk:isomodellmp](3)'s refinement half on a SYNTHETIC refined
//          descriptor: SubdivideEdgeOrbit of the two edges at zeta^2 (the
//          Moerdijk--Pronk-conformant refinement) and, in a second descriptor,
//          also a horocycle edge -- read by the same IsotropyModel(ModelData)
//          path as the shipped model; certificates, the orbifold Euler
//          characteristic, exactly one periodic column at ell = 3 (the new
//          vertices have stabiliser {+-I}), and on COHOMOLOGY ONLY: lengths and
//          invariant factors of H^q of Iso^(3) on Y' = Iso^(3) on Xbulk = Iso
//          (Z/3^e, e <= 3), the boundary, odd n (-I |-> -1) admissible, a
//          level-N module, ell = 2 through the same code; comparison() refused.
// Env hook HECKE_ISOMODEL_ELL_FULL=1 (any other value throws): the heavier grid
// (more N, n and e), with an `EFFECTIVE HeckeIsotropyModelEllValidation
// arm=FULL cells=<k>` line.  Read-only, not an emitter.

#include <cstdint>
#include <memory>
#include <vector>

namespace integral_model {
class IsotropyModel;
class LocalModel;
class StrictLiftBundle;
}  // namespace integral_model

struct IsoEllArmTally {
    uint32_t checks = 0;
    uint32_t failures = 0;
};

// Runs (IL3N), (IL4e), (PC), (REF) on the shipped descriptor `desc`, the free
// model `freeM` (ell = 0) and Iso^(3) `iso3` built on it; `bundles` are the
// StrictLiftBundles of T_2, T_3 = U_3, T_5 (in that order).
IsoEllArmTally HeckeIsotropyModelEllOpenHalves(const integral_model::IsotropyModel& desc,
                                               const integral_model::LocalModel& freeM,
                                               const integral_model::LocalModel& iso3,
                                               const std::vector<std::unique_ptr<integral_model::StrictLiftBundle>>& bundles);

#endif
