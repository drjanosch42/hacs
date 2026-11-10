// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// NumberFieldLllValidation (fast; slot-21 T1, the in-house LLL and POLRED of
// Lll.h / Polred.h). Compiled into cohomology_app, not into the STL-only
// NumberField library. Returns the number of failing checks (0 = pass).
#include <cstdint>

uint32_t NumberFieldLllValidation();
