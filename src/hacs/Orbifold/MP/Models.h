// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_MODELS_H
#define ORBIFOLD_MP_MODELS_H

// ===========================================================================
// (hecke-extensions slot 22, MP-2) The two triangulations every MP-2 validation
// uses, built exactly as MP-1a/1b build them: M7 = the shipped 7-vertex
// descriptor ([C, def:model7], integral_model::SL2Z7VertexModelData) and
// R_Z = SubdivideEdgeOrbit(M7, the arc) with the arc found from the inventory
// (the unique edge representative joining the stabiliser-order-6 and -4 vertex
// orbits, never by name; MP_impl_plan.md v0.2 Sec. 2.2).
// ===========================================================================

#include "Descriptor.h"
#include "../../Integral/Model/IsotropyModel.h"

namespace orbifold_mp {

struct MPModels {
    integral_model::ModelData base;   // M7
    integral_model::ModelData rz;     // R_Z
    Descriptor m7;
    Descriptor rzd;
};

// ENGINE_CHECK: exactly one edge representative joins the order-6 and order-4 orbits.
MPModels BuildM7AndRZ();

}  // namespace orbifold_mp

#endif
