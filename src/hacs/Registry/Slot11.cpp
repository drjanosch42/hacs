// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-11 registration (closed slot): the former per-package fragments Registry/Slot11/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot11() calls them in the old RegisterAllValidations() order.

// ===== Slot 11 WP5 ===========================================
// Slot-11 WP-5 registration fragment (hecke-extensions slot 11, WP-1).
// Owned by package WP-5 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

#include <cstdint>

// HeckeExt/HeckeDicksonThetaValidation.cpp
uint32_t HeckeDicksonThetaValidation();
uint32_t HeckeDicksonThetaCoverageSlowValidation();

void RegisterSlot11WP5() {
    // WP-5: the Dickson / theta line (1) -- Q_n, the long sequence, H^0.
    ValidationRegistry::register_validation("HeckeDicksonThetaValidation", &HeckeDicksonThetaValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("HeckeDicksonThetaCoverageSlowValidation",
                                            &HeckeDicksonThetaCoverageSlowValidation, ValidationTier::Slow);
}

// ===== Slot 11 WP6 ===========================================
// Slot-11 WP-6 registration fragment (hecke-extensions slot 11, WP-1).
// Owned by package WP-6 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

#include <cstdint>

// HeckeExt/HeckeDicksonThetaHeckeValidation.cpp
uint32_t HeckeDicksonThetaHeckeValidation();
uint32_t HeckeDicksonThetaHeckeCoverageSlowValidation();

void RegisterSlot11WP6() {
    // WP-6: the Dickson / theta line (2) -- the Hecke operators on the theta
    // sequence, U_ell theta = 0, H^0, the instances, the census column.
    ValidationRegistry::register_validation("HeckeDicksonThetaHeckeValidation", &HeckeDicksonThetaHeckeValidation,
                                            ValidationTier::Standard);
    ValidationRegistry::register_validation("HeckeDicksonThetaHeckeCoverageSlowValidation",
                                            &HeckeDicksonThetaHeckeCoverageSlowValidation, ValidationTier::Slow);
}

// ===== Slot 11 WP7 ===========================================
// Slot-11 WP-7 registration fragment (hecke-extensions slot 11, WP-1).
// Owned by package WP-7 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

#include <cstdint>

// HeckeExt/HeckeCensusSplitValidation.cpp
uint32_t HeckeCensusSplitValidation();

void RegisterSlot11WP7() {
    // WP-7: the congruence census's computed split ([E] sec:census (d1), (d2), (d6), (e)).
    ValidationRegistry::register_validation("HeckeCensusSplitValidation", &HeckeCensusSplitValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 11 WP9 ===========================================
// Slot-11 WP-9 registration fragment (hecke-extensions slot 11, WP-1).
// Owned by package WP-9 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

void RegisterSlot11WP9() {
    // Empty by default (WP-1); WP-9 adds its register_validation() calls here.
}

// ===== Slot 11 WP10 ===========================================
// Slot-11 WP-10 registration fragment (hecke-extensions slot 11, WP-1).
// Owned by package WP-10 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

void RegisterSlot11WP10() {
    // Empty by default (WP-1); WP-10 adds its register_validation() calls here.
}

// ===== Slot 11 WP11 ===========================================
// Slot-11 WP-11 registration fragment (hecke-extensions slot 11, WP-1).
// Owned by package WP-11 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../Integral/HeckeStrictLiftRelationsValidation.h"

void RegisterSlot11WP11() {
    // (hecke-extensions slot 11, WP-11, M-3; [C, prop:strictlift], [C, prop:diamond],
    // [C, thm:relations]) the strict lift tau_Iso for diag(1, p^2), diag(1, p^3)
    // and diag(p, p) on the free model, certified exactly over Z; T_{p^2}, T_{p^3}
    // and T(p, p) on H^q(C_I (x) R); the Hecke recursion (X7) on cohomology.
    // `standard`. Writes nothing.
    ValidationRegistry::register_validation("HeckeStrictLiftRelationsValidation",
                                            &HeckeStrictLiftRelationsValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 11 WP12 ===========================================
// Slot-11 WP-12 registration fragment (hecke-extensions slot 11, WP-1).
// Owned by package WP-12 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../EngineCarriesS11Validation.h"   // slot-11 WP-12 (F-1, F-2)

void RegisterSlot11WP12() {
    // (hecke-extensions slot 11, WP-12, F-1) ModCtx's Montgomery-64
    // reused-operand pair (mul_prep / mul_prepped) and the composite row
    // updates that use it: arithmetic against the old exact %-product at
    // U64_Montgomery and U64_Wide moduli; every composite entry point
    // entrywise equal to the %-path and to ref::howell; u64 == u32 below
    // 2^32, on the engine and on EquivariantCohomologyOnK's module route.
    // A few seconds; writes nothing.
    ValidationRegistry::register_validation("ModCtxMontgomeryPairValidation",
                                            &ModCtxMontgomeryPairValidation,
                                            ValidationTier::Fast);
    // (hecke-extensions slot 11, WP-12, F-2) ExactKthRoot's primality by
    // deterministic Miller-Rabin, equal to the slot-8 trial-division
    // implementation; the ring carried as (ell, k) by
    // LocalSystemSimplicialComplexT::declare_coefficient_ring, equal entrywise
    // to the undeclared route.  Writes nothing.
    ValidationRegistry::register_validation("CoefficientRingKthRootValidation",
                                            &CoefficientRingKthRootValidation,
                                            ValidationTier::Fast);
}

// ===== Slot 11 WP13 ===========================================
// Slot-11 WP-13 registration fragment (hecke-extensions slot 11, WP-1).
// Owned by package WP-13 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

void RegisterSlot11WP13() {
    // Empty by default (WP-1); WP-13 adds its register_validation() calls here.
}

// ===== Slot 11 WP16 ===========================================
// Slot-11 WP-16 registration fragment (hecke-extensions slot 11, WP-1).
// Owned by package WP-16 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

void RegisterSlot11WP16() {
    // Empty by default (WP-1); WP-16 adds its register_validation() calls here.
}

// ===== Slot 11 WP18 ===========================================
// Slot-11 WP-18 registration fragment (hecke-extensions slot 11, WP-1).
// Owned by package WP-18 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

void RegisterSlot11WP18a();   // Registry/Slot11/WP18a.cpp (slot-11 WP-18a)
void RegisterSlot11WP18b();   // Registry/Slot11/WP18b.cpp (slot-11 WP-18b)
void RegisterSlot11WP18c();   // Registry/Slot11/WP18c.cpp (slot-11 WP-18c)

void RegisterSlot11WP18() {
    // Empty by default (WP-1); WP-18 adds its register_validation() calls here.
    // WP-18 is split into 18a / 18b / 18c: each sub-package registers in its
    // own Registry/Slot11/WP18<x>.cpp and adds one call here, in order.
    RegisterSlot11WP18a();
    RegisterSlot11WP18b();
    RegisterSlot11WP18c();
}

// ===== Slot 11 WP18a ===========================================
// Slot-11 WP-18a registration fragment (hecke-extensions slot 11, WP-18a).
// Owned by package WP-18a ONLY.  WP-18 is split into 18a / 18b / 18c (plan
// Sec. 10, 2026-09-25); each sub-package registers in its own file, called
// from RegisterSlot11WP18() (Registry/Slot11/WP18.cpp), so concurrent
// sub-packages never edit the same fragment body.  Explicit function, never a
// static initialiser.
#include "../ValidationRegistry.h"
#include "../EngineU128CompositeValidation.h"   // slot-11 WP-18a

void RegisterSlot11WP18a() {
    // (hecke-extensions slot 11, WP-18a; ENGINE_API.md Sec. 13.12, step 1)
    // the composite-only u128 engine layer (ModCtx<u128> / U128_Montgomery,
    // u256 accumulation, Hensel inv, the u128 kernels, the Sec. 13.6 entry
    // points, ref::naive128, the block model's composite surface) and the
    // digit-width abstraction DigitWord<W>, under the Sec. 13.12.4 oracle
    // rule: u128 == u32 / u64 entrywise on HeckeReductionValidation's
    // elimination-arm matrices; against the naive oracle and planted
    // divisors beyond 2^64.  Writes nothing.
    ValidationRegistry::register_validation("EngineU128CompositeValidation",
                                            &EngineU128CompositeValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 11 WP18b ===========================================
// Slot-11 WP-18b registration fragment (hecke-extensions slot 11, WP-18b).
// Owned by package WP-18b ONLY (the C2 digit induction).  Called explicitly
// from RegisterSlot11WP18() (Registry/Slot11/WP18.cpp, after WP-18a's call),
// never a static initialiser; its sources are listed in cmake/slot11/WP18b.cmake,
// included from cmake/slot11/WP18.cmake.
#include "../ValidationRegistry.h"
#include "../EngineLadicInductionValidation.h"   // slot-11 WP-18b

void RegisterSlot11WP18b() {
    // (hecke-extensions slot 11, WP-18b) engine::ladic, the (C2) ell-adic
    // digit induction (ENGINE_API.md Sec. 13.11), against the width oracle
    // matrix: 2 x 32 digits vs native u64, cross-width agreement and the
    // naive base-ell oracle beyond 2^127, planted divisors, the Smith
    // certificate, truncation, the pipeline's own d^q.  The w = 128 columns
    // print [SKIP] until WP-18a merges.  Writes nothing.
    ValidationRegistry::register_validation("EngineLadicInductionValidation",
                                            &EngineLadicInductionValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 11 WP18c ===========================================
// Slot-11 WP-18c registration fragment (hecke-extensions slot 11, WP-18c).
// Owned by package WP-18c ONLY (the digit pipeline route).  Called explicitly
// from RegisterSlot11WP18() (Registry/Slot11/WP18.cpp, after WP-18b's call),
// never a static initialiser; its sources are listed in cmake/slot11/WP18c.cmake,
// included from cmake/slot11/WP18.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeLadicCellValidation.h"   // slot-11 WP-18c

void RegisterSlot11WP18c() {
    // (hecke-extensions slot 11, WP-18c) Hecke/LadicCell, the digit pipeline
    // route at ell^k >= 2^64 ([E] v1.12.0-pre.2 sec:ring), against the
    // pipeline-level oracle matrix: 2 x 32 vs the shipped u64 route, M-0 at
    // (10, 691) up to 691^40, UCT vs the K-mode, cross-width, truncation,
    // native u128; the per-cell rule and the certified precision bound; the
    // ordinary idempotent.  Writes nothing.
    ValidationRegistry::register_validation("HeckeLadicCellValidation",
                                            &HeckeLadicCellValidation,
                                            ValidationTier::Standard);
}

void RegisterSlot11() {
    RegisterSlot11WP5();
    RegisterSlot11WP6();
    RegisterSlot11WP7();
    RegisterSlot11WP9();
    RegisterSlot11WP10();
    RegisterSlot11WP11();
    RegisterSlot11WP12();
    RegisterSlot11WP13();
    RegisterSlot11WP16();
    RegisterSlot11WP18();
}
