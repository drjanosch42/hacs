// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// NumberFieldPRankValidation (standard; slot-22 NF-PRANK, the p-rank of Cl(Q(N^{1/p})) by
// relations: PureField.h, Ambiguous.h, ClassGroupRel.h). Compiled into cohomology_app, not
// into the STL-only NumberField library. Returns the number of failing checks (0 = pass).
#include <cstdint>

uint32_t NumberFieldPRankValidation();
