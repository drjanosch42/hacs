// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_KUMMER_VALIDATION_H
#define HECKE_EXT_KUMMER_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 5, WP-8 -- [E] Sec. 8.3's Kummer list on X_4, the ten
// cells of [E] `cl:harder` with n < ell - 1 and k <= 26, through
// HeckeExtDriver, with `bernoulli_mod` / `kummer_divides` (slot-4 WP-7) as the
// INDEPENDENT oracle for the list itself.  Returns 0 on success, else the
// number of failing checks.  Registered `slow` -- see the report: the ten
// cells at the DEFAULT mathematical settings cost 37 s, of which 34 s is
// U_3617 alone.
uint32_t HeckeExtKummerValidation();

#endif // HECKE_EXT_KUMMER_VALIDATION_H
