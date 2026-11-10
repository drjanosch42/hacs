// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeFamilyConcurrentBuildValidation.h -- hecke-extensions slot 9, WP-7 (lever L2).
#pragma once
#include <cstdint>

// Two HeckeOperatorFamily constructions of U_ell (plus T_p) running AT ONCE on
// two threads -- now that build_pipelines() no longer holds
// tau_construction_mutex() around its coset / tau / operator constructions --
// give entrywise the matrices of the serial builds, and each thread's
// PredicateStats deltas equal the serial run's; both on two independent cells
// and on two families sharing ONE K / model / cohomology.  Run under
// CheckThread for the race claim.
uint32_t HeckeFamilyConcurrentBuildValidation();
