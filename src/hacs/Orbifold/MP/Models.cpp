// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-2) M7 and R_Z -- see Models.h.
#include "Models.h"

#include <cstdint>
#include <vector>

#include "../../../ModArithmetic/EngineCheck.h"

namespace orbifold_mp {

MPModels BuildM7AndRZ() {
    MPModels M;
    M.base = integral_model::SL2Z7VertexModelData();
    uint32_t rho2 = UINT32_MAX, ivert = UINT32_MAX;
    for (uint32_t r = 0; r < M.base.reps[0].size(); ++r) {
        if (M.base.order[0][r] == 6) rho2 = M.base.reps[0][r][0];
        if (M.base.order[0][r] == 4) ivert = M.base.reps[0][r][0];
    }
    std::vector<integral_model::Simplex> arc;
    for (const auto& e : M.base.reps[1])
        if ((e[0] == rho2 && e[1] == ivert) || (e[0] == ivert && e[1] == rho2)) arc.push_back(e);
    ENGINE_CHECK_MSG(arc.size() == 1, "BuildM7AndRZ: not exactly one edge representative joins the order-6 and -4 orbits");
    M.rz = integral_model::SubdivideEdgeOrbit(M.base, arc[0]);
    M.m7 = FromModelData(M.base, "M7");
    M.rzd = FromModelData(M.rz, "R_Z");
    return M;
}

}  // namespace orbifold_mp
