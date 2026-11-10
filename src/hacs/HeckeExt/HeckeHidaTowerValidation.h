// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 12, WP-5) HeckeHidaTowerValidation (standard) and
// HeckeHidaTowerSlowValidation (slow): the Hida weight towers of
// HeckeExt/HeckeHidaTower at ell in {5, 7, 11, 13}, mod ell and mod ell^2, and
// the non-ordinary cell (10, 2411) over Z/2411^m -- [C] v1.12.0-pre.2
// rmk:hidacheckable (Hi2)-(Hi6), def:hidatower, cor:ordprecision.  Write nothing;
// one HIDA_TOWER line per (cell, level) on stdout (not published).
#pragma once
#ifndef HECKE_HIDA_TOWER_VALIDATION_H
#define HECKE_HIDA_TOWER_VALIDATION_H

#include <cstdint>

uint32_t HeckeHidaTowerValidation();
uint32_t HeckeHidaTowerSlowValidation();

#endif
