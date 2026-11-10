// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_RING_LOEWY_VALIDATION_H
#define HECKE_EXT_RING_LOEWY_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 14, WP-6) HeckeRingLoewyValidation (standard): the
// Z/ell^K Hecke-algebra route (HeckeExt/HeckeRingHeckeAlgebra) against [M1]'s
// Python oracle, the census and the forms side.  See the .cpp for the arms.
uint32_t HeckeRingLoewyValidation();
// (S15-O1) arm (C3) of the above, the digit route, as its own `slow` name.
uint32_t HeckeRingLoewyDigitSlowValidation();

#endif // HECKE_EXT_RING_LOEWY_VALIDATION_H
