// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-17 registration (closed slot): the former per-package fragments Registry/Slot17/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot17() calls them in the old RegisterAllValidations() order.

// ===== Slot 17 WP5 ===========================================
// Slot-17 WP5 registration fragment (hecke-extensions slot 17). Owned by
// package WP5 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot17/WP5.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot17WP5() {
    // Empty at day zero (slot-17 WP-1).
}

// ===== Slot 17 WP6 ===========================================
// Slot-17 WP6 registration fragment (hecke-extensions slot 17). Owned by
// package WP6 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot17/WP6.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot17WP6() {
    // Empty at day zero (slot-17 WP-1).
}

// ===== Slot 17 WP7 ===========================================
// Slot-17 WP7 registration fragment (hecke-extensions slot 17). Owned by
// package WP7 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot17/WP7.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot17WP7() {
    // Empty at day zero (slot-17 WP-1).
}

// ===== Slot 17 WP8 ===========================================
// Slot-17 WP8 registration fragment (hecke-extensions slot 17). Owned by
// package WP8 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot17/WP8.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeSmallCarriesValidation.h"

void RegisterSlot17WP8() {
    // (WP-8) negative tests of the small carries' refusals; pure and fast.
    ValidationRegistry::register_validation("HeckeSmallCarriesValidation", &HeckeSmallCarriesValidation, ValidationTier::Fast);
}

// ===== Slot 17 L2 ===========================================
// Slot-17 L2 registration fragment (hecke-extensions slot 17). Owned by
// package L2 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot17/L2.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeSatCertDumpValidation.h"

void RegisterSlot17L2() {
    // (L17-2, owner ruling Lean Q14) the validation-only stable-certificate dump hook for the
    // Lean checker lean/GL2/Algorithms/SatCert.lean: a no-op ([SKIP] line, exit 0) unless
    // HECKE_DUMP_SAT_CERT=<dir> names an existing directory outside the source tree.
    ValidationRegistry::register_validation("HeckeSatCertDumpValidation", &HeckeSatCertDumpValidation,
                                            ValidationTier::Fast);
}

void RegisterSlot17() {
    RegisterSlot17WP5();
    RegisterSlot17WP6();
    RegisterSlot17WP7();
    RegisterSlot17WP8();
    RegisterSlot17L2();
}
