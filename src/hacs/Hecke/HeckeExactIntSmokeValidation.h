// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXACT_INT_SMOKE_VALIDATION_H
#define HECKE_EXACT_INT_SMOKE_VALIDATION_H

#include <cstdint>

// Stage S1 smoke validation (invariant I9: returns 0 on success), registered
// in ValidationRegistry::RegisterAllValidations(). Exercises the exact
// bounded-integer layer (Mat22 group identities, overflow aborts), the
// certified FP filter (random agreement of the filtered and exact paths),
// and the QuadPoint predicates.
uint32_t HeckeExactIntSmokeValidation();

#endif // HECKE_EXACT_INT_SMOKE_VALIDATION_H
