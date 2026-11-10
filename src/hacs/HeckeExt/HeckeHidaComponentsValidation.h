// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HIDA_COMPONENTS_VALIDATION_H
#define HECKE_EXT_HIDA_COMPONENTS_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 15, WP-6) HeckeHidaComponentsValidation (standard): the
// bignum-free B_k/k mod ell^K and the component readings of [U] (U1)-(U3) at tame
// level 1 (HeckeExt/HeckeHidaComponents).  See the .cpp for the arms.
uint32_t HeckeHidaComponentsValidation();

#endif // HECKE_EXT_HIDA_COMPONENTS_VALIDATION_H
