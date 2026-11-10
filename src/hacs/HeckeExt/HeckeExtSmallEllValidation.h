// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_SMALL_ELL_VALIDATION_H
#define HECKE_EXT_SMALL_ELL_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 5, WP-8 -- [E] Sec. 8.3's small-ell cases: the Dickson
// loci, the two Delta-multiplicity cells, the two U_ell self-extension cells,
// the STURM COUNTEREXAMPLES (22, 5) and (26, 7) on the pipeline's own H^1, and
// (32, 131).  Returns 0 on success, else the number of failing checks.
uint32_t HeckeExtSmallEllValidation();

#endif // HECKE_EXT_SMALL_ELL_VALIDATION_H
