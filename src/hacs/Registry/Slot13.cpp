// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-13 registration (closed slot): the former per-package fragments Registry/Slot13/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot13() calls them in the old RegisterAllValidations() order.

// ===== Slot 13 WP4 ===========================================
// Slot-13 WP-4 registration fragment (hecke-extensions slot 13). Owned by
// package WP-4 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot13/WP4.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeUEllLowRankValidation.h"   // slot-13 WP-4

void RegisterSlot13WP4() {
    // (hecke-extensions slot 13, WP-4) The U_ell memory lever
    // (HeckeOperatorOptions::ScopedUEllLowRank): the opt-in rank-<= k route of
    // HeckeOperatorOnK equal to the default entrywise over F_ell and Z/ell^k.
    ValidationRegistry::register_validation("HeckeUEllLowRankValidation",
                                            &HeckeUEllLowRankValidation,
                                            ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 15)
}

// ===== Slot 13 WP5 ===========================================
// Slot-13 WP-5 registration fragment (hecke-extensions slot 13). Owned by
// package WP-5 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot13/WP5.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeHidaControlValidation.h"   // slot-13 WP-5

void RegisterSlot13WP5() {
    // (hecke-extensions slot 13, WP-5) Hida theory at level one, part 2 --
    // [C] v1.13.0-pre.2 rmk:hidacheckable2 (Hi7) mod ell^3, (Hi8) the lattice
    // datum by its decision procedure on Sym^n (a) and on Hida's lattice L_n (b),
    // D_* (c) reported, (Hi9) T_5 / T_7; the torsion pre-pass beyond 2^64.
    // Writes nothing; HIDA_TOWER lines on stdout only.  (The ell = 7 mod-ell^3
    // pair 10 <-> 304 is not in the suite: 187 s > the plan's 3-minute cap.)
    ValidationRegistry::register_validation("HeckeHidaControlValidation",
                                            &HeckeHidaControlValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 13 WP6 ===========================================
// Slot-13 WP-6 registration fragment (hecke-extensions slot 13). Owned by
// package WP-6 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot13/WP6.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeHidaLevelEllValidation.h"   // slot-13 WP-6

void RegisterSlot13WP6() {
    // (hecke-extensions slot 13, WP-6) Gamma_0(ell) on the fixed model by
    // Shapiro -- [C] v1.13.0-pre.2 sec:levelshapiro (prop:levelcorrespondence,
    // prop:levelellU / eq:Ulevell, prop:oldspace, prop:levelorbit, obl:shapiro),
    // sec:stabilisation (lem:levelelltorsion, prop:stabilisation) and (Hi10);
    // an independent group-cocycle route.  Writes nothing; HIDA_LEVEL_ELL lines
    // on stdout.
    ValidationRegistry::register_validation("HeckeHidaLevelEllValidation",
                                            &HeckeHidaLevelEllValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 13 WP7 ===========================================
// Slot-13 WP-7 registration fragment (hecke-extensions slot 13). Owned by
// package WP-7 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot13/WP7.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot13WP7() {
    // Empty on day zero (slot-13 WP-1).
}

// ===== Slot 13 WP9 ===========================================
// Slot-13 WP-9 registration fragment (hecke-extensions slot 13). Owned by
// package WP-9 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot13/WP9.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeCensusLatticeValidation.h"   // slot-13 WP-9

void RegisterSlot13WP9() {
    // WP-9: C-F9 ([PR] Sec. 7), the lattice test on census edition 2. `standard`;
    // HECKE_CENSUS_LATTICE_ALL=1 adds every non-reduced row (read-only, not an emitter).
    ValidationRegistry::register_validation("HeckeCensusLatticeValidation",
                                            &HeckeCensusLatticeValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 13 WP10 ===========================================
// Slot-13 WP-10 registration fragment (hecke-extensions slot 13). Owned by
// package WP-10 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot13/WP10.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeHidaHeavyCellProbe.h"   // slot-13 WP-10

void RegisterSlot13WP10() {
    // WP-10: the heavy Hida cells ([C] rmk:hidacheckable2, (Hi11)) behind the
    // U_ell memory lever. `slow` and INTERLOCKED on HECKE_HIDA_HEAVY=1: without
    // it the probe prints [SKIP] (even under --all-full), like HeckeLocalityProbe.
    ValidationRegistry::register_validation("HeckeHidaHeavyCellProbe",
                                            &HeckeHidaHeavyCellProbe,
                                            ValidationTier::Slow);
}

// ===== Slot 13 WP11 ===========================================
// Slot-13 WP-11 registration fragment (hecke-extensions slot 13). Owned by
// package WP-11 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot13/WP11.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeBocksteinCuspValidation.h"   // slot-13 WP-11 (referee m6)
#include "../HeckeExt/HeckeMixedFinalityValidation.h"   // slot-13 WP-11

void RegisterSlot13WP11() {
    // WP-11: the evidence arm of the `mixed` finality proposal (Proposition M,
    // docs/implementation/slot13/WP-11_mixed_finality_proposal.md). `slow` (opt-in:
    // not in --all); read-only, not an emitter; no record, golden line or digest moves.
    ValidationRegistry::register_validation("HeckeMixedFinalityValidation",
                                            &HeckeMixedFinalityValidation,
                                            ValidationTier::Slow);
    // WP-11 repairs (referee finding m6): claim (H4') -- r^1 injective on im beta^0 --
    // pipeline-free over Sym^n(Z/ell^2) on the referee's grid. `fast`; read-only.
    ValidationRegistry::register_validation("HeckeBocksteinCuspValidation",
                                            &HeckeBocksteinCuspValidation,
                                            ValidationTier::Fast);
}

// ===== Slot 13 WP12 ===========================================
// Slot-13 WP-12 registration fragment (hecke-extensions slot 13). Owned by
// package WP-12 ONLY: register this package's validations here and nowhere else
// in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser); its sources are listed in cmake/slot13/WP12.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot13WP12() {
    // Empty on day zero (slot-13 WP-1).
}

void RegisterSlot13() {
    RegisterSlot13WP4();
    RegisterSlot13WP5();
    RegisterSlot13WP6();
    RegisterSlot13WP7();
    RegisterSlot13WP9();
    RegisterSlot13WP10();
    RegisterSlot13WP11();
    RegisterSlot13WP12();
}
