// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKEZ_UTAG_VALIDATION_H
#define HECKEZ_UTAG_VALIDATION_H

#include <cstdint>

// slot-25 HZ-UTAG (docs/implementation/slot25/00-slot25-plan.md Sec. 2.2; 06-work-packages.md "HZ-UTAG"; 02-toy-cases.md Sec. 3.2,
// 3.3, F3). `fast` tier. The library is Integral/HeckeZUTag.{h,cpp}.
//  T4 (2, 6): chi_Q(U_2) = (x - 1)(x - 128)(x + 8)^2; `upper_bound_certified` with C(f, E^(1)) | 1 (f and E^(1) do not meet),
//     C(f, E^(2^7)) | 2^3 * 17 = 136 (g_P^an = 2^7 * 17 from T_3, T_5, T_7, T_11), C(E^(1), E^(2^7)) | 127.
//  T5 (8, 10): chi_Q(U_2) = x^16 (x - 1)(x - 2048)(x^2 + 24 x + 2048)^2; with d = 13, d^an = 6 `non_reduced_certified`.
// Gate line: HECKEZ_UTAG cells=2 mismatched=0 verdict=OK
// Returns the number of failing checks (0 on success).
uint32_t HeckeZUTagValidation();

#endif
