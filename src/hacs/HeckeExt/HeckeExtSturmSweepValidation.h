// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_STURM_SWEEP_VALIDATION_H
#define HECKE_EXT_STURM_SWEEP_VALIDATION_H

#include <cstdint>

// hecke-extensions phase E1b (slot 3, WP-2) -- the REAL-BLOCK arms of
// generation certificate (c) and of Stage 1's `u_ell` budget check.
//
// They cannot live in HeckeSturmBoundValidation, which consumes no pipeline by
// design (the HeckeExt/ guard's rule 3), nor in HeckeExtAlgebraValidation,
// whose arms are constructed truths compared against closed forms.  Here the
// blocks are the ones the SL(2,Z) pipeline itself produces.
//
// Returns the number of failing checks (0 on success).
uint32_t HeckeExtSturmSweepValidation();

#endif // HECKE_EXT_STURM_SWEEP_VALIDATION_H
