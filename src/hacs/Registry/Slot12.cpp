// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-12 registration (closed slot): the former per-package fragments Registry/Slot12/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot12() calls them in the old RegisterAllValidations() order.

// ===== Slot 12 WP1 ===========================================
// Slot-12 WP-1 registration fragment (hecke-extensions slot 12, WP-1: day
// zero). Owned by package WP-1 ONLY: register this package's validations
// here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

#include <cstdint>

// HeckeExt/HeckeThetaCensusRingRefusalValidation.cpp (carry F-1, [E] sec:ring
// (b1)): the theta / census line's closure members each refuse a non-default
// coefficient_ring through a named guard.
uint32_t HeckeThetaCensusRingRefusalValidation();

void RegisterSlot12WP1() {
    ValidationRegistry::register_validation("HeckeThetaCensusRingRefusalValidation",
                                            &HeckeThetaCensusRingRefusalValidation, ValidationTier::Fast);
}

// ===== Slot 12 WP4 ===========================================
// Slot-12 WP-4 registration fragment (hecke-extensions slot 12). Owned by
// package WP-4 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot12/WP4.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeOrdinaryPrecisionValidation.h"   // slot-12 WP-4

void RegisterSlot12WP4() {
    // (hecke-extensions slot 12, WP-4) Hecke/HeckeOrdinaryPart: the torsion
    // pre-pass e_max ([C] prop:torsionprepass) against the certified K-mode,
    // the ordinary projector U^{N_ord} ([C] lem:ordexponent) against slot-11
    // WP-18c's Fitting idempotent -- [C] rmk:hidacheckable (Hi1), (Hi2) -- with
    // negative controls; LadicCell's opt-in precision_source = prepass.
    // Writes nothing.
    ValidationRegistry::register_validation("HeckeOrdinaryPrecisionValidation",
                                            &HeckeOrdinaryPrecisionValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 12 WP5 ===========================================
// Slot-12 WP-5 registration fragment (hecke-extensions slot 12). Owned by
// package WP-5 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot12/WP5.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeHidaTowerValidation.h"   // slot-12 WP-5

void RegisterSlot12WP5() {
    // (hecke-extensions slot 12, WP-5) HeckeExt/HeckeHidaTower: Hida weight
    // towers at ell in {5, 7, 11, 13}, mod ell and mod ell^2, and the
    // non-ordinary cell (10, 2411) -- [C] rmk:hidacheckable (Hi2)-(Hi6),
    // def:hidatower.  Write nothing; HIDA_TOWER lines on stdout only.
    ValidationRegistry::register_validation("HeckeHidaTowerValidation",
                                            &HeckeHidaTowerValidation,
                                            ValidationTier::Standard);
    // The slow companion: even n <= 200 (the ell = 13 period-156 tower) and
    // (10, 2411) over Z/2411^3.
    ValidationRegistry::register_validation("HeckeHidaTowerSlowValidation",
                                            &HeckeHidaTowerSlowValidation,
                                            ValidationTier::Slow);
}

// ===== Slot 12 WP6 ===========================================
// Slot-12 WP-6 registration fragment (hecke-extensions slot 12). Owned by
// package WP-6 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot12/WP6.cmake.
#include "../ValidationRegistry.h"

#include <cstdint>

uint32_t HeckeCensusCellListValidation();   // HeckeExt/HeckeCensusCellListValidation.cpp

void RegisterSlot12WP6() {
    // (hecke-extensions slot 12, WP-6) the census cell list: the parser behind
    // HeckeCensusSplitValidation's HECKE_CENSUS_SPLIT_CELLS=<file> option
    // (census edition 2, staged) and edition 1's grid as a function.  Pure
    // parsing; writes nothing.
    ValidationRegistry::register_validation("HeckeCensusCellListValidation",
                                            &HeckeCensusCellListValidation,
                                            ValidationTier::Fast);
}

// ===== Slot 12 WP7 ===========================================
// Slot-12 WP-7 registration fragment (hecke-extensions slot 12). Owned by
// package WP-7 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot12/WP7.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeFormsHeckeAlgebraValidation.h"   // slot-12 WP-7

void RegisterSlot12WP7() {
    // (hecke-extensions slot 12, WP-7) HeckeExt/HeckeFormsHeckeAlgebra: the
    // forms-side T_m / ell of S_k(F_ell)_m from q-expansions, against the
    // slot-11 WP-20 referee's datum at (82, 7) / (142, 7) and the ext driver's
    // A (a quotient of it).  Writes nothing; FORMS_ALGEBRA lines on stdout.
    ValidationRegistry::register_validation("HeckeFormsHeckeAlgebraValidation",
                                            &HeckeFormsHeckeAlgebraValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 12 WP8 ===========================================
// Slot-12 WP-8 registration fragment (hecke-extensions slot 12, WP-1: day
// zero). Owned by package WP-8 ONLY: register this package's validations
// here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser). Empty at WP-1 --
// see docs/implementation/slot12/00-slot12-plan.md Sec. 2 for this package's
// scope.
#include "../ValidationRegistry.h"

void RegisterSlot12WP8() {
    // WP-8: not run yet.
}

// ===== Slot 12 WP14 ===========================================
// Slot-12 WP-14 registration fragment (hecke-extensions slot 12). Owned by
// package WP-14 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot12/WP14.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeFormsHeckeAlgebraValidation.h"   // slot-12 WP-14

void RegisterSlot12WP14() {
    // (hecke-extensions slot 12, WP-14) the WP-13 referee's coverage of the
    // boundary finality proposal: Lemma B' on the boundary operators (F3), the
    // Eichler-Shimura identity (F9), the mixed negative control and, under
    // HECKE_BOUNDARY_FINALITY_ALL=1 (read-only, not an emitter), every heuristic
    // census row (F12).  The body lives in
    // HeckeExt/HeckeFormsHeckeAlgebraValidation.cpp (shared helpers); writes nothing.
    ValidationRegistry::register_validation("HeckeBoundaryFinalityValidation",
                                            &HeckeBoundaryFinalityValidation,
                                            ValidationTier::Standard);
}

void RegisterSlot12() {
    RegisterSlot12WP1();
    RegisterSlot12WP4();
    RegisterSlot12WP5();
    RegisterSlot12WP6();
    RegisterSlot12WP7();
    RegisterSlot12WP8();
    RegisterSlot12WP14();
}
