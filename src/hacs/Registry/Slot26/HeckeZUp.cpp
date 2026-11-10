// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-26 Z1 registration fragment (UP-Z (a): U_p, p | N, over Z_(ell), ell >= 5; docs/implementation/slot26/00-slot26-plan.md Sec. 2.1
// "Z1", docs/implementation/slot26/Z1_report.md). Owned by package Z1 ONLY: register this package's validations here and nowhere else in
// the shared structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot26/HeckeZUp.cmake (Registry/Slot26/README.md).
#include "../../ValidationRegistry.h"
#include "../../Integral/HeckeZUpValidation.h"

void RegisterSlot26HeckeZUp() {
    // (Z1) the export's U_p against HZ-UTAG's chi_Q(U_p) mod ell^K, a_p^2 = p^{k-2} on the p-new part, thread / ISA route identity, a
    // tampered control, the recompute subset (15 golden level-1 + 3 anemic records = HZ-DS ed. 1) and the full level-N records' scan.
    ValidationRegistry::register_validation("HeckeZUpValidation", &HeckeZUpValidation, ValidationTier::Standard);
}
