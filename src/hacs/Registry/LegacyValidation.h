// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// LegacyValidation.h -- registering a validation (or an arm) that runs only in an
// OC_ENABLE_LEGACY=ON build (S24 CL-L0; docs/implementation/slot24/CLEANUP_plan_1030.md
// Sec. 2.3)
// =====================================================================
//
// WHOLE NAMES.  A legacy-gated validation stays registered in EVERY build, at its place
// in its original Registry/Slot<S>/ fragment and with its tier, so `--list` (and
// the cluster kits' expected roster count) never depends on the flag:
//
//     ValidationRegistry::register_validation("FooValidation", OC_LEGACY_VALIDATION(FooValidation),
//                                             ValidationTier::Standard);
//
//   ON : expands to &FooValidation;
//   OFF: expands to a stub that prints `[SKIP] FooValidation: OC_ENABLE_LEGACY=OFF` and
//        returns 0 (oc::BuildSwitchSkip, the OC_BUILD_<X> pattern).  FooValidation itself is
//        not referenced, so its definition (a legacy/ source) need not be compiled.
//
// ARMS.  An arm of a surviving validation that needs the legacy route runs under
//
//     if (oc::legacy::RunLegacyArm("FooValidation", "b")) { ... the arm ... }
//
//   ON : returns true, prints nothing;
//   OFF: prints `SKIP-ARM FooValidation arm=b: OC_ENABLE_LEGACY=OFF` on stdout and returns
//        false, so the smaller check count is explained.  The arm's body must still compile in
//        the OFF build -- or sit under `#if OC_ENABLE_LEGACY` -- when it calls legacy/ code.
// =====================================================================

#include <cstdint>
#include <iostream>
#include <string>

#include "../../ModArithmetic/LegacyConfig.h"
#include "../BuildSwitches.h"

// oc::legacy::SkipArmLine / RunLegacyArm live in LegacyConfig.h (S24 CL-L2: the ModArithmetic validations use them
// too, and BuildSwitches.h is cohomology_app-only).

#if OC_ENABLE_LEGACY
#define OC_LEGACY_VALIDATION(Fn) (&Fn)
#else
#define OC_LEGACY_VALIDATION(Fn) \
    (+[]() -> uint32_t { return static_cast<uint32_t>(oc::BuildSwitchSkip(#Fn, "OC_ENABLE_LEGACY")); })
#endif
