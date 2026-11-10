// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_SATURATION_DEFECT_VALIDATION_H
#define HECKE_EXT_SATURATION_DEFECT_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 16, WP-5) HeckeSaturationDefectValidation (standard): the
// saturation-defect Tor term I_m = Tor_1^{Z_ell}(T^sat / T_m, F_ell) on torsion-free
// blocks (HeckeExt/HeckeSaturationDefect).  See the .cpp for the arms;
// HECKE_SAT_DEFECT_CELLS="n:ell[:K],..." runs it over a cell list instead.
uint32_t HeckeSaturationDefectValidation();

#endif // HECKE_EXT_SATURATION_DEFECT_VALIDATION_H
