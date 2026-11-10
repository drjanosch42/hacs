// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HIDA_CM_VALIDATION_H
#define HECKE_EXT_HIDA_CM_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 16, WP-15) HeckeHidaCMValidation (standard): [U] part 2
// (sec:u-tameN) in code -- the CM components on the character model of
// [C, sec:nebentypus] (Hecke/HeckeNebentypus, slot-16 WP-10): (U8)-(U10) of
// rmk:u-pred2 at the (NB6) cells and on the CM ladders at tame N >= 3, the h_K = 1
// hypothesis of prop:u-cmeis(2) with a negative example at h_K = 2.  Env hook
// HECKE_HIDA_CM_CELLS.  See the .cpp for the arms.
uint32_t HeckeHidaCMValidation();

#endif // HECKE_EXT_HIDA_CM_VALIDATION_H
