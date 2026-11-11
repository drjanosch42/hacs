// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-26 Z6 registration fragment (H2: the discriminant series of the ramified ordinary families at tame level 1, [U] part 7
// sec:u-H2; docs/implementation/slot26/Z6_design.md v0.2, Z6_report.md). Owned by package Z6 ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static
// initialiser); its sources are listed in cmake/slot26/HeckeHidaDiscLambda.cmake (Registry/Slot26/README.md).
#include "../../ValidationRegistry.h"
#include "../../HeckeExt/HeckeHidaDiscLambdaValidation.h"

void RegisterSlot26HeckeHidaDiscLambda() {
    // (Z6) the eleven blocks of prop:u-loewy(4): per-block Fitting split, D_F mod (ell^3, Z^3), held-out weight, oracles, controls, tampers.
    ValidationRegistry::register_validation("HeckeHidaDiscLambdaValidation", &HeckeHidaDiscLambdaValidation, ValidationTier::Standard);
}
