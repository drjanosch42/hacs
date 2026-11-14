// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-27 BIGNTT registration fragment (the NTT big-integer finder; docs/implementation/slot27/00-slot27-plan.md Sec. 2.1 "BIGNTT-D ->
// BIGNTT", docs/implementation/slot27/BIGNTT_design.md v0.2 Sec. 7). Owned by package BIGNTT ONLY. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are listed in cmake/slot27/BigIntNtt.cmake (Registry/Slot27/README.md).
#include "../../ValidationRegistry.h"
#include "../../Integral/BigIntNttValidation.h"

void RegisterSlot27BigIntNtt() {
    // (M1) nf::bigntt products and Barrett steps = BigInt's route, bit for bit, on the forced route; threads; the finder's MR.
    ValidationRegistry::register_validation("BigIntNttValidation", &BigIntNttValidation, ValidationTier::Standard);
}
