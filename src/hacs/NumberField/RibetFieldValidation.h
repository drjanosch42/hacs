// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// NumberFieldRibetFieldValidation (standard; slot-21 T2, the degree-ell Ribet field
// polynomial of RibetField.h / MpReal.h). Compiled into cohomology_app, not into the
// STL-only NumberField library. Returns the number of failing checks (0 = pass).
#include <cstdint>

uint32_t NumberFieldRibetFieldValidation();
