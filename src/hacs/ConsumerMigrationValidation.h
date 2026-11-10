// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CONSUMER_MIGRATION_VALIDATION_H
#define CONSUMER_MIGRATION_VALIDATION_H

#include <cstdint>

// ===========================================================================
// ConsumerMigrationValidation -- matrix-engine Phase 3.
//
// The plan (docs/matrix-engine/00-implementation-plan.md, Phase 3
// "Validations") requires this validation to "assert the raw-`mod_*`-loop
// count by a build-time grep script in docs/matrix-engine/tools/", and states
// that THE GREP TARGET IS NOT ZERO: the largest concentrations of raw
// ModArithmetic calls sit in files the programme deliberately leaves alone
// (FpPoly.cpp, HeckeModuleDecomposition.cpp, HeckeTauTable.cpp's exact-Z
// solver, every *Validation.cpp, the golden harness, the bench driver). The
// assertion is therefore "zero call sites OUTSIDE an explicit allow-list", and
// REMOVING A FILE FROM THE ALLOW-LIST IS THE UNIT OF PROGRESS.
//
// This is the STAGE A form of the validation: the static gate on the migration
// tools. It runs each tool script under docs/matrix-engine/tools/ that exists
// and fails if any of them reports work outside its allow-list:
//
//   * raw_mod_calls.sh        -- raw ModArithmetic mod_* call sites
//                                (allow-list: raw_mod_calls.allow)
//   * check_matrix_includes.sh -- Matrix.h includes in pipeline TUs, the
//                                `Matrix` retirement gate. PROBED FOR
//                                EXISTENCE: it is owned by the concurrent
//                                Phase 3 retirement work and this validation
//                                must not fail on a tree where it has not
//                                landed yet, so a missing script is SKIPPED
//                                (reported, not failed).
//
// Phase 3's later stages extend this validation with the dynamic half the plan
// also names -- "-DENGINE_SHADOW_LEGACY=ON: every migrated call site compares
// against the legacy path on the golden grid". That arm is already covered
// today by GoldenRegressionValidation under GOLDEN_STRICT=1 on a shadow build
// (README.md, "Validation"); it moves in here when the BlockMat
// consumer migration gives it per-call-site granularity.
//
// Returns 0 on success; non-zero = number of failing checks.
// ===========================================================================
uint32_t ConsumerMigrationValidation();

#endif // CONSUMER_MIGRATION_VALIDATION_H
