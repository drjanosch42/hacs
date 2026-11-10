// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_HIDA_HEAVY_CELL_PROBE_H
#define HECKE_EXT_HECKE_HIDA_HEAVY_CELL_PROBE_H

#include <cstdint>

// (hecke-extensions slot 13, WP-10) HeckeHidaHeavyCellProbe -- the heavy Hida
// cells of [C] rmk:hidacheckable2, (Hi11) at (700, 691), behind the U_ell
// memory lever (slot-13 WP-4; retired S24 CL-R1: its default Auto mode).
// `slow` tier, INTERLOCKED on HECKE_HIDA_HEAVY=1 (prints [SKIP] otherwise, like
// HeckeLocalityProbe).  One HIDA_TOWER line per cell.  See the .cpp.
uint32_t HeckeHidaHeavyCellProbe();

#endif // HECKE_EXT_HECKE_HIDA_HEAVY_CELL_PROBE_H
