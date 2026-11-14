// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-27 EISUPb registration fragment (HZ-EISUP (b); docs/implementation/slot27/00-slot27-plan.md "EISUPb",
// docs/implementation/slot26/HZEISUPb_design.md v0.4 Sec. 7). Owned by package EISUPb ONLY: register this package's validations here and
// nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources are listed
// in cmake/slot27/HeckeZEisUpChar.cmake (Registry/Slot27/README.md).
#include "../../ValidationRegistry.h"
#include "../../Integral/HeckeZEisUpCharValidation.h"

void RegisterSlot27HeckeZEisUpChar() {
    // (EISUPb) the Eisenstein U_p systems with characters psi != 1: certificate, second route, record path, bit identity, tampers, schema.
    ValidationRegistry::register_validation("HeckeZEisUpCharValidation", &HeckeZEisUpCharValidation, ValidationTier::Standard);
}
