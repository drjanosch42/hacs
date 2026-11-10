// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_SEPARATION_VALIDATION_H
#define HECKE_EXT_SEPARATION_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 5, WP-8 -- [E] Sec. 8.2's two further ell = 691 rows,
// the SEPARATION case: (160, 691), where no single T_q separates the systems
// and {T_2, T_3} does, and (198, 691).  Returns 0 on success, else the number
// of failing checks.
uint32_t HeckeExtSeparationValidation();

#endif // HECKE_EXT_SEPARATION_VALIDATION_H
