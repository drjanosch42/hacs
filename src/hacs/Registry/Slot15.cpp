// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-15 registration (closed slot): the former per-package fragments Registry/Slot15/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot15() calls them in the old RegisterAllValidations() order.

// ===== Slot 15 WP5 ===========================================
// Slot-15 WP-5 registration fragment (hecke-extensions slot 15). Owned by
// package WP-5 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot15/WP5.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeRingLoewyValidation.h"

void RegisterSlot15WP5() {
    // (hecke-extensions slot 15, WP-5) stable as a certificate; the digit route (HeckeRingHeckeAlgebra): empty at day zero.
    // (S15-O1) arm (C3) of HeckeRingLoewyValidation, moved to the slow tier by the owner.
    ValidationRegistry::register_validation("HeckeRingLoewyDigitSlowValidation",
                                            &HeckeRingLoewyDigitSlowValidation,
                                            ValidationTier::Slow);
}

// ===== Slot 15 WP6 ===========================================
// Slot-15 WP-6 registration fragment (hecke-extensions slot 15). Owned by
// package WP-6 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot15/WP6.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeHidaComponentsValidation.h"   // slot-15 WP-6

void RegisterSlot15WP6() {
    // (hecke-extensions slot 15, WP-6) WP-14c in C++: HeckeExt/HeckeHidaComponents --
    // the bignum-free B_k/k mod ell^K and the [U] (U1)-(U3) component readings at tame
    // level 1.  `standard`; HECKE_HIDA_COMPONENTS_CELLS="n:ell[:K],..." runs the
    // readings over a cell list instead (read-only, not an emitter; HIDA_COMPONENT lines).
    ValidationRegistry::register_validation("HeckeHidaComponentsValidation",
                                            &HeckeHidaComponentsValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 15 WP7 ===========================================
// Slot-15 WP-7 registration fragment (hecke-extensions slot 15). Owned by
// package WP-7 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot15/WP7.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeLevelNHeilbronnValidation.h"   // slot-15 WP-7

void RegisterSlot15WP7() {
    // (hecke-extensions slot 15, WP-7) The Heilbronn--Merel action H_{N,p} on the
    // cokernel presentation -- [C] sec:levelN (lem:merelpath, prop:heilbronn,
    // rmk:heilbronnconv) -- and (LN3) closed on both halves against the tau-route
    // of prop:levelcorrespondence (4)(d).  Writes nothing; LEVEL_N_HEILBRONN lines.
    ValidationRegistry::register_validation("HeckeLevelNHeilbronnValidation",
                                            &HeckeLevelNHeilbronnValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 15 WP8 ===========================================
// Slot-15 WP-8 registration fragment (hecke-extensions slot 15). Owned by
// package WP-8 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot15/WP8.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeLevelNSparseValidation.h"   // slot-15 WP-8

void RegisterSlot15WP8() {
    // (hecke-extensions slot 15, WP-8) The sparse cokernel at large N ([C] rmk:levelcost (4)),
    // sparse = dense entrywise; level change and degeneracy maps at composite N
    // (prop:levelcorrespondence (4)(b), (c)); the cost model.  Writes nothing; LEVEL_N_SPARSE /
    // SPARSE_CELL lines; read-only hook HECKE_LEVEL_N_SPARSE_CELLS.
    ValidationRegistry::register_validation("HeckeLevelNSparseValidation",
                                            &HeckeLevelNSparseValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 15 WP16a ===========================================
// Slot-15 WP-16a registration fragment (hecke-extensions slot 15). Owned by
// package WP-16a ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot15/WP16a.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot15WP16a() {
    // (hecke-extensions slot 15, WP-16a) GPU G1 (stretch): empty at day zero.
}

void RegisterSlot15() {
    RegisterSlot15WP5();
    RegisterSlot15WP6();
    RegisterSlot15WP7();
    RegisterSlot15WP8();
    RegisterSlot15WP16a();
}
