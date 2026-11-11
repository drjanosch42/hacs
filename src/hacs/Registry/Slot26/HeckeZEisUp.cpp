// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-26 HZ-EISUP registration fragment (the Eisenstein U_p systems of the full algebra at N > 1; docs/implementation/slot26/HZEISUP_design.md
// v0.2, HZEISUP_report.md). Owned by package HZ-EISUP ONLY: register this package's validations here and nowhere else in the shared
// structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot26/HeckeZEisUp.cmake (Registry/Slot26/README.md).
#include "../../ValidationRegistry.h"
#include "../../Integral/HeckeZEisUpValidation.h"

void RegisterSlot26HeckeZEisUp() {
    // (HZ-EISUP) the certificate C-E1 .. C-E5 at ten cells, the closed form as the second route, HZ-UTAG's chi_Q(U_p) as an oracle, four records
    // through the record path (C-E7), bit identity, six tampers, the HZ hash invariance and the A4 schema (tampers, edition 2 re-scanned).
    ValidationRegistry::register_validation("HeckeZEisUpValidation", &HeckeZEisUpValidation, ValidationTier::Standard);
}
