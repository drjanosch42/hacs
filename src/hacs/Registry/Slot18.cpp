// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-18 registration (closed slot): the former per-package fragments Registry/Slot18/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot18() calls them in the old RegisterAllValidations() order.

// ===== Slot 18 WP5 ===========================================
// Slot-18 WP5 registration fragment (hecke-extensions slot 18). Owned by
// package WP5 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot18/WP5.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot18WP5() {
    // Empty at slot-18 WP-1 (day zero). The package adds its validations here.
}

// ===== Slot 18 WP6 ===========================================
// Slot-18 WP6 registration fragment (hecke-extensions slot 18). Owned by
// package WP6 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot18/WP6.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot18WP6() {
    // Empty at slot-18 WP-1 (day zero) and after WP-6: WP-6 adds arms to HeckeNebentypusValidation only (no new name;
    // plan v1.0 Sec. 7), its source Hecke/HeckeNebentypusGalois.cpp through cmake/slot18/WP6.cmake.
}

// ===== Slot 18 WP7 ===========================================
// Slot-18 WP7 registration fragment (hecke-extensions slot 18). Owned by
// package WP7 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot18/WP7.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot18WP7() {
    // Empty at slot-18 WP-1 (day zero). The package adds its validations here.
}

// ===== Slot 18 WP8 ===========================================
// Slot-18 WP8 registration fragment (hecke-extensions slot 18). Owned by
// package WP8 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot18/WP8.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot18WP8() {
    // Empty at slot-18 WP-1 (day zero). The package adds its validations here.
}

// ===== Slot 18 L3 ===========================================
// Slot-18 L3 registration fragment (hecke-extensions slot 18). Owned by
// package L3 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot18/L3.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot18L3() {
    // Empty at slot-18 WP-1 (day zero). The package adds its validations here.
}

// ===== Slot 18 L5 ===========================================
// Slot-18 L5 registration fragment (hecke-extensions slot 18). Owned by
// package L5 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot18/L5.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot18L5() {
    // Empty at slot-18 WP-1 (day zero). The package adds its validations here.
}

// ===== Slot 18 WPM ===========================================
// Slot-18 WPM registration fragment (hecke-extensions slot 18, package WP-M,
// owner ruling 2026-10-03: the CLI / console user manual). Owned by package
// WP-M ONLY: register this package's validations here and nowhere else in the
// shared structure. Called explicitly from RegisterAllValidations() (never a
// static initialiser); its sources are listed in cmake/slot18/WPM.cmake.
// WP-1 created the empty stub pair and the shared one-line edits.
#include "../ValidationRegistry.h"
#include "../Console/CliManualCoverageValidation.h"

void RegisterSlot18WPM() {
    // (WP-M) docs/latex/cli_manual.tex and its generated tables against the
    // binary; read-only unless CLI_MANUAL_EMIT=<scratch-dir> is set.
    ValidationRegistry::register_validation("CliManualCoverageValidation", &CliManualCoverageValidation, ValidationTier::Fast);
}

// ===== Slot 18 WP19b ===========================================
// Slot-18 WP19b registration fragment (hecke-extensions slot 18, stretch
// package WP-19b: the p != ell digit split). Owned by package WP-19b ONLY.
// Created mid-slot with the two one-line shared edits (slot-12 WP-14
// precedent; Registry/Slot18/README.md: "Stretch packages ... do the same").
// Called explicitly from RegisterAllValidations(); its sources are listed in
// cmake/slot18/WP19b.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeLadicSplitValidation.h"

void RegisterSlot18WP19b() {
    // (WP-19b) DigitCellSession (RunCell split at the Hecke step) against fresh
    // RunCell records, bit for bit; read-only, no env.
    ValidationRegistry::register_validation("HeckeLadicSplitValidation", &HeckeLadicSplitValidation, ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 17)
}

// ===== Slot 18 WP19a ===========================================
// Slot-18 WP19a registration fragment (hecke-extensions slot 18, stretch
// package WP-19a: GPU G1, the elimination session). Owned by package WP-19a
// ONLY. Created mid-slot with the two one-line shared edits (slot-12 WP-14
// precedent; Registry/Slot18/README.md: "Stretch packages ... do the same").
// Called explicitly from RegisterAllValidations(); its sources are listed in
// cmake/slot18/WP19a.cmake.
#include "../ValidationRegistry.h"
#include "../../Engine/EngineElimSessionValidation.h"

void RegisterSlot18WP19a() {
    // (WP-19a) session-scoped device residency in the elimination driver and
    // the device row-list GATHER, entrywise against engine::blk / the CPU
    // executor under gpu / hybrid x verify x session on/off; [SKIP] without a
    // device; read-only, no env.
    ValidationRegistry::register_validation("EngineElimSessionValidation", &EngineElimSessionValidation, ValidationTier::Standard);
}

void RegisterSlot18() {
    RegisterSlot18WP5();
    RegisterSlot18WP6();
    RegisterSlot18WP7();
    RegisterSlot18WP8();
    RegisterSlot18L3();
    RegisterSlot18L5();
    RegisterSlot18WPM();
    RegisterSlot18WP19b();
    RegisterSlot18WP19a();
}
