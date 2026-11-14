// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-27 TS-FIX / TS registration fragment (the theta search; docs/implementation/slot27/00-slot27-plan.md Sec. 2.1 "TS-D -> TS",
// docs/implementation/slot27/TS_design.md Sec. 9). Owned by package TS ONLY: register this package's validations here and nowhere else in
// the shared structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot27/HeckeZThetaSearch.cmake (Registry/Slot27/README.md).
#include "../../ValidationRegistry.h"
#include "../../Integral/HeckeZThetaSearchValidation.h"

void RegisterSlot27HeckeZThetaSearch() {
    // (TS-FIX) the g_P predicate at its five consumers; (TS) the theta_order setting, the greedy continuation at both sites.
    ValidationRegistry::register_validation("HeckeZThetaSearchValidation", &HeckeZThetaSearchValidation, ValidationTier::Standard);
}
