// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-27 EXW registration fragment (the at-root read and the (C3) read on the free block over Z where rule (alpha) fires;
// docs/implementation/slot27/EXW_design.md v0.2). Owned by package EXW ONLY: register this package's validations here and nowhere else in
// the shared structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot27/HeckeZExportWord.cmake (Registry/README.md).
#include "../../ValidationRegistry.h"
#include "../../Integral/HeckeZExportWordValidation.h"

void RegisterSlot27HeckeZExportWord() {
    // (EXW) nine arms: route identity, class A beyond K_max, class C / k1 at K = 1, overflow teeth, four whole records EXW on vs. off (the
    // diff rule, WriteCertFiles), route bit-identity, teeth, record-level identity by forcing, the negative case of (alpha).
    ValidationRegistry::register_validation("HeckeZExportWordValidation", &HeckeZExportWordValidation, ValidationTier::Standard);
}
