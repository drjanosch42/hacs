// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-26 Z9 HZ-LOC-PAR registration fragment (hecke-extensions slot 26). Owned by package Z9
// (docs/implementation/slot26/00-slot26-plan.md "Z9 — HZ-LOC-PAR", docs/implementation/slot26/Z9_report.md) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a
// static initialiser), after RegisterSlot25(); its sources are listed in cmake/slot26/HeckeZLocalPar.cmake (Registry/Slot26/README.md).
#include "../../ValidationRegistry.h"
#include "../../Integral/HeckeZLocalParValidation.h"

void RegisterSlot26HeckeZLocalPar() {
    // (Z9) the par route of HeckeZLocal (parallel closure products, semi-naive worklist, exact parallel reduce) against the
    // slot-25 serial route (lever HECKEZ_LOCAL_PAR=0): spans, closures and the 30 HECKEZ_LOCAL cells, threads 1 / 4 / 16.
    ValidationRegistry::register_validation("HeckeZLocalParValidation", &HeckeZLocalParValidation, ValidationTier::Standard);
}
