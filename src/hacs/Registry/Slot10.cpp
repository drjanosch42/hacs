// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-10 registration (closed slot): the former per-package fragments Registry/Slot10/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot10() calls them in the old RegisterAllValidations() order.

// ===== Slot 10 WP3 ===========================================
// Slot-10 WP-3 registration fragment (hecke-extensions slot 10, WP-1).
// Owned by package WP-3 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
//
// WP-3: [C] obl:coneexact(vii) over Z/ell^k -- the strict-pair and relative
// cross-routes over the ring (slot-10 decision 3 (A)).
#include "../ValidationRegistry.h"

#include "../Hecke/HeckeConeCrossRouteRingValidation.h"

void RegisterSlot10WP3() {
    ValidationRegistry::register_validation("HeckeConeCrossRouteRingValidation",
                                            &HeckeConeCrossRouteRingValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 10 WP6 ===========================================
// Slot-10 WP-6 registration fragment (hecke-extensions slot 10, WP-1).
// Owned by package WP-6 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

void RegisterSlot10WP6() {
    // Empty by default (WP-1); WP-6 adds its register_validation() calls here.
}

// ===== Slot 10 WP7 ===========================================
// Slot-10 WP-7 registration fragment (hecke-extensions slot 10, WP-1).
// Owned by package WP-7 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

void RegisterSlot10WP7() {
    // Empty by default (WP-1); WP-7 adds its register_validation() calls here.
}

// ===== Slot 10 WP8 ===========================================
// Slot-10 WP-8 registration fragment (hecke-extensions slot 10, WP-1).
// Owned by package WP-8 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../EngineFingerprintDeviceBitValidation.h"   // slot-10 WP-8 (decision 10 (A), the ninth bit)

void RegisterSlot10WP8() {
    // (hecke-extensions slot 10, WP-8; decision 10 (A)) pass_options_fingerprint()
    // now distinguishes `hybrid` from `gpu` (ENGINE_API.md Sec. 10.6): (A) the
    // three from_env() device mappings give three distinct fingerprints with
    // the exact (bit 128, bit 256) pattern; (B) a tiny recorded program lands
    // in three distinct ProgramCache slots, one per policy, and replays its
    // own on a second compile.  Sub-second, needs no CUDA device.  Writes
    // nothing.
    ValidationRegistry::register_validation("EngineFingerprintDeviceBitValidation",
                                            &EngineFingerprintDeviceBitValidation,
                                            ValidationTier::Fast);
}

// ===== Slot 10 WP10 ===========================================
// Slot-10 WP-10 registration fragment (hecke-extensions slot 10, WP-1).
// Owned by package WP-10 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../Integral/HeckeStrictLiftValidation.h"

void RegisterSlot10WP10() {
    // (hecke-extensions slot 10, WP-10 / Line 2, L2-1b (1); [C, prop:strictlift],
    // [C, lem:strictlifting]) the strict lift tau_Iso of the strict pair datum to
    // the isotropy-resolution model, degrees 0..6, certified exactly over Z;
    // its coset transport Theta^_g over any ModCtx; the strict compatibility
    // eps^* Theta^s_g = Theta^_g eps^* on the rings with a shipped Theta^s_g;
    // the H^1 operator against the shipped A^(1)(T_p) at ell >= 5 and O5.
    // `standard`. Writes nothing.
    ValidationRegistry::register_validation("HeckeStrictLiftValidation",
                                            &HeckeStrictLiftValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 10 WP11 ===========================================
// Slot-10 WP-11 registration fragment (hecke-extensions slot 10, WP-1).
// Owned by package WP-11 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../Integral/HeckeIntegralActionValidation.h"

void RegisterSlot10WP11() {
    // (hecke-extensions slot 10, WP-11 / Line 2, M1(a); [IB] Sec. 6.2 M1(a))
    // the integral Hecke action, basis-free: O6 at good primes, the
    // 691-congruence module on both lines, T_p on the 2- and 3-primary torsion
    // (X6, X7, HeckeFreeModelValidation's tests (4) and (5) hosted here), the
    // local type of Lambda^1 (x) Z_ell at 2 and 3, X1.  `standard`.  Writes
    // nothing.
    ValidationRegistry::register_validation("HeckeIntegralActionValidation",
                                            &HeckeIntegralActionValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 10 WP12 ===========================================
// Slot-10 WP-12 registration fragment (hecke-extensions slot 10, WP-1).
// Owned by package WP-12 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"

void RegisterSlot10WP12() {
    // Empty by default (WP-1); WP-12 adds its register_validation() calls here.
}

// ===== Slot 10 WP13 ===========================================
// Slot-10 WP-13 registration fragment (hecke-extensions slot 10, WP-1).
// Owned by package WP-13 ONLY: register this package's validations here and
// nowhere else in the shared structure.  Called explicitly from
// RegisterAllValidations() (never a static initialiser).
#include "../ValidationRegistry.h"
#include "../Integral/HeckePeriodicityClassValidation.h"

void RegisterSlot10WP13() {
    // (hecke-extensions slot 10, WP-13 / Line 2, L2-1b (2); [C, lem:bulkperiodic],
    // [C, def:isomodel]) the class e in H^2(Gamma; Z) as a 2-cocycle on R_2, the
    // chain map E : R_{.+2} -> R_. representing e u -, and every clause of
    // [C, lem:bulkperiodic](2) on the M0 cells (n <= 20, Z/ell^K and F_ell at
    // ell in {2, 3}) and at good ell.  Tier by R-L2.  Writes nothing.
    ValidationRegistry::register_validation("HeckePeriodicityClassValidation",
                                            &HeckePeriodicityClassValidation,
                                            ValidationTier::Standard);
}

void RegisterSlot10() {
    RegisterSlot10WP3();
    RegisterSlot10WP6();
    RegisterSlot10WP7();
    RegisterSlot10WP8();
    RegisterSlot10WP10();
    RegisterSlot10WP11();
    RegisterSlot10WP12();
    RegisterSlot10WP13();
}
