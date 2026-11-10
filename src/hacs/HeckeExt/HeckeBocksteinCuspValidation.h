// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_BOCKSTEIN_CUSP_VALIDATION_H
#define HECKE_EXT_BOCKSTEIN_CUSP_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 13, WP-11 (referee finding m6 of
// docs/implementation/slot13/WP-11_referee.md) -- claim (H4') of the `mixed`
// finality proposal: r^1 : H^1(Gamma; M_n) -> H^1(bd; M_n) is injective on
// im beta^0.  A pipeline-free Sym^n(Z/ell^2) computation (a C++ port of the
// referee's bockstein_cusp.py); `fast` tier, read-only.  One BOCKSTEIN_CUSP
// summary line per ell.
uint32_t HeckeBocksteinCuspValidation();

#endif // HECKE_EXT_BOCKSTEIN_CUSP_VALIDATION_H
