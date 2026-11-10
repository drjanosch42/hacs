// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-19 registration (closed slot): the former per-package fragments Registry/Slot19/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot19() calls them in the old RegisterAllValidations() order.

// ===== Slot 19 WPNF ===========================================
// Slot-19 WPNF registration fragment (hecke-extensions slot 19). Owned by
// package WPNF ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot19/WPNF.cmake.
#include "../ValidationRegistry.h"
#include "../NumberField/NumberFieldValidation.h"

void RegisterSlot19WPNF() {
    // (WP-NF) the number-field core ([E] sec:lvnf): BigInt / NTT / Bernoulli
    // self-tests; R18-2's exact eta_32, eta_6 at 37 (FNV of the 36 coefficients,
    // v_pi = 38 / 6); layer 1 at 37 and 691 (v = 692); the equivariant Z/ell^2
    // layer: kappa = 284 / 914 / 3292 the only unramified lift at 37 / 59 / 67;
    // negative controls; Q(alpha), Newton polygon, Hensel;
    // check_numberfield_includes.sh. Read-only, no env, <= 5 s.
    ValidationRegistry::register_validation("NumberFieldValidation", &NumberFieldValidation, ValidationTier::Fast);
}

// ===== Slot 19 WPLC ===========================================
// Slot-19 WPLC registration fragment (hecke-extensions slot 19). Owned by
// package WPLC ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot19/WPLC.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeLatticeModuleValidation.h"

void RegisterSlot19WPLC() {
    // (WP-LC) [E] chk:latcmp: H^1(Gamma; Sym^258 Z/37^M)_m as a T_m-module at (258, 37), M = 1..4 (rank, mu, Smith
    // valuations of t = T_2 - 29 and U_37, per iota-sign, T_p = Q_p(t) for p < 100 from R19-1's tracked file), the
    // ell = 7 Loewy-2 / Loewy-3 controls and negative controls; read-only, no env.
    ValidationRegistry::register_validation("HeckeLatticeModuleValidation", &HeckeLatticeModuleValidation, ValidationTier::Slow);
}

// ===== Slot 19 WPLV ===========================================
// Slot-19 WPLV registration fragment (hecke-extensions slot 19). Owned by
// package WPLV ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot19/WPLV.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/LValue/HeckeLValueValidation.h"

void RegisterSlot19WPLV() {
    // (WP-LV) the L-value interface of [E] sec:lvalue and its Eisenstein / Bernoulli instance (kind congruence) on
    // rmk:lvgates's untwisted cells, m = 1..3, with negative controls; read-only, no env.
    ValidationRegistry::register_validation("HeckeLValueValidation", &HeckeLValueValidation, ValidationTier::Standard);
}

// ===== Slot 19 NB0 ===========================================
// Slot-19 NB0 registration fragment (hecke-extensions slot 19). Owned by
// package NB0 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot19/NB0.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot19NB0() {
    // Empty at slot-19 WP-1 (day zero). The package adds its validations here.
}

// ===== Slot 19 NB2 ===========================================
// Slot-19 NB2 registration fragment (hecke-extensions slot 19). Owned by
// package NB2 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot19/NB2.cmake.
#include "../ValidationRegistry.h"
#include "../Console/ConsoleReadOnlyTagValidation.h"
#include "../Console/ConsoleServeValidation.h"

void RegisterSlot19NB2() {
    // (NB-2, [P] v3.10 Sec. 18.11) The served console's conformance pair.
    // Both are registered in every build (the --list count is switch-
    // independent) and print `[SKIP] <name>: OC_BUILD_NOTEBOOK=OFF` when the
    // switch is off.
    //
    // ConsoleReadOnlyTagValidation: the per-overload read_only tag -- registry
    // <-> Console/serve/read_only_verbs.tsv, registered == explicitly tagged,
    // the source audit of the registration TUs, the manual's tag column, the
    // PrintJson type coverage and envelope rules, the JSON / env-gate units.
    // In process, read-only, no env.
    ValidationRegistry::register_validation("ConsoleReadOnlyTagValidation", &ConsoleReadOnlyTagValidation,
                                            ValidationTier::Fast);
    // ConsoleServeValidation: drives `cohomology_app --serve` as a child over
    // pipes (arms A-J): the numbered transcripts byte for byte through the
    // served `out` frames, the exempt scripts rejected, value envelopes, the
    // served determinism grid (ENGINE_THREADS 1 / 2 / 8, program cache off),
    // every tagged overload called under RLIMIT_FSIZE = 0, error mapping,
    // interrupt, protocol, environment gate, RSS cap, chatter, completion.
    // Children run with ENGINE_CUDA_DISABLE=1 in an empty scratch directory
    // under a scratch directory (removed afterwards); the
    // RSS-cap arm allocates ~0.32 GiB in one child.  Read-only.
    ValidationRegistry::register_validation("ConsoleServeValidation", &ConsoleServeValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 19 WPHL ===========================================
// Slot-19 WPHL registration fragment (hecke-extensions slot 19). Owned by
// package WPHL ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot19/WPHL.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeLambdaValidation.h"

void RegisterSlot19WPHL() {
    // (WP-HL) the truncated Lambda-adic ordinary Hecke algebra
    // T^ord(a) (x) Lambda/(ell^K, X^m) ([U] 0.3.0-pre.3 sec:u-lambda): the
    // q-expansion input arm (Q) reproduces R19-2's forms JSON, the Lambda stage
    // its lambda JSON at (37, 32) and (103, 24) exactly; (U11)-(U13), KL, the
    // rungs; arm (T) (HeckeHidaTower) as a cross-check at 37 and 103; the
    // hida_tower/v1 catalogue at ell <= 13; precision refusals and perturbed
    // inputs caught. HECKE_LAMBDA_ALL=1 (read-only) adds the 691 classes 12, 200.
    ValidationRegistry::register_validation("HeckeLambdaValidation", &HeckeLambdaValidation, ValidationTier::Standard);
}

void RegisterSlot19() {
    RegisterSlot19WPNF();
    RegisterSlot19WPLC();
    RegisterSlot19WPLV();
    RegisterSlot19NB0();
    RegisterSlot19NB2();
    RegisterSlot19WPHL();
}
