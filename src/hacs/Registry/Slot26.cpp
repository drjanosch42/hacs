// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-26 registration (closed slot): the former per-package fragments Registry/Slot26/<pkg>.cpp folded verbatim
// (S26 FOLD, the slot-26 close; include paths one level up), each package function kept; RegisterSlot26() calls them in the
// old RegisterAllValidations() order.

// ===== Slot 26 HeckeZUp ===========================================
// Slot-26 Z1 registration fragment (UP-Z (a): U_p, p | N, over Z_(ell), ell >= 5; docs/implementation/slot26/00-slot26-plan.md Sec. 2.1
// "Z1", docs/implementation/slot26/Z1_report.md). Owned by package Z1 ONLY: register this package's validations here and nowhere else in
// the shared structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot26/HeckeZUp.cmake (Registry/Slot26/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/HeckeZUpValidation.h"

void RegisterSlot26HeckeZUp() {
    // (Z1) the export's U_p against HZ-UTAG's chi_Q(U_p) mod ell^K, a_p^2 = p^{k-2} on the p-new part, thread / ISA route identity, a
    // tampered control, the recompute subset (15 golden level-1 + 3 anemic records = HZ-DS ed. 1) and the full level-N records' scan.
    ValidationRegistry::register_validation("HeckeZUpValidation", &HeckeZUpValidation, ValidationTier::Standard);
}

// ===== Slot 26 HeckeZLocalPar ===========================================
// Slot-26 Z9 HZ-LOC-PAR registration fragment (hecke-extensions slot 26). Owned by package Z9
// (docs/implementation/slot26/00-slot26-plan.md "Z9 — HZ-LOC-PAR", docs/implementation/slot26/Z9_report.md) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a
// static initialiser), after RegisterSlot25(); its sources are listed in cmake/slot26/HeckeZLocalPar.cmake (Registry/Slot26/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/HeckeZLocalParValidation.h"

void RegisterSlot26HeckeZLocalPar() {
    // (Z9) the par route of HeckeZLocal (parallel closure products, semi-naive worklist, exact parallel reduce) against the
    // slot-25 serial route (lever HECKEZ_LOCAL_PAR=0): spans, closures and the 30 HECKEZ_LOCAL cells, threads 1 / 4 / 16.
    ValidationRegistry::register_validation("HeckeZLocalParValidation", &HeckeZLocalParValidation, ValidationTier::Standard);
}

// ===== Slot 26 HeckeZEisUp ===========================================
// Slot-26 HZ-EISUP registration fragment (the Eisenstein U_p systems of the full algebra at N > 1; docs/implementation/slot26/HZEISUP_design.md
// v0.2, HZEISUP_report.md). Owned by package HZ-EISUP ONLY: register this package's validations here and nowhere else in the shared
// structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot26/HeckeZEisUp.cmake (Registry/Slot26/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/HeckeZEisUpValidation.h"

void RegisterSlot26HeckeZEisUp() {
    // (HZ-EISUP) the certificate C-E1 .. C-E5 at ten cells, the closed form as the second route, HZ-UTAG's chi_Q(U_p) as an oracle, four records
    // through the record path (C-E7), bit identity, six tampers, the HZ hash invariance and the A4 schema (tampers, edition 2 re-scanned).
    ValidationRegistry::register_validation("HeckeZEisUpValidation", &HeckeZEisUpValidation, ValidationTier::Standard);
}

// ===== Slot 26 HeckeHidaDiscLambda ===========================================
// Slot-26 Z6 registration fragment (H2: the discriminant series of the ramified ordinary families at tame level 1, [U] part 7
// sec:u-H2; docs/implementation/slot26/Z6_design.md v0.2, Z6_report.md). Owned by package Z6 ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static
// initialiser); its sources are listed in cmake/slot26/HeckeHidaDiscLambda.cmake (Registry/Slot26/README.md).
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeHidaDiscLambdaValidation.h"

void RegisterSlot26HeckeHidaDiscLambda() {
    // (Z6) the eleven blocks of prop:u-loewy(4): per-block Fitting split, D_F mod (ell^3, Z^3), held-out weight, oracles, controls, tampers.
    ValidationRegistry::register_validation("HeckeHidaDiscLambdaValidation", &HeckeHidaDiscLambdaValidation, ValidationTier::Standard);
}

// ===== Slot 26 HeckeZCrossedHomOracle ===========================================
// Slot-26 Z3b registration fragment (an M oracle at ell = 2, 3 without T~_Iso; docs/implementation/slot26/Z3b_design.md v0.2,
// Z3b_report.md). Owned by package Z3b ONLY: register this package's validations here and nowhere else in the shared structure. Called
// explicitly from RegisterAllValidations() (never a static initialiser); its sources are listed in cmake/slot26/HeckeZCrossedHomOracle.cmake
// (Registry/Slot26/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/HeckeZCrossedHomOracleValidation.h"

void RegisterSlot26HeckeZCrossedHomOracle() {
    // (Z3b) crossed homomorphisms + [C, eq:barcorrected] via Shapiro: (K) conventions, (F) the free part, (M) the M oracle at every torsion unit,
    // (Pi) the retraction, (I) RouteBasis on the ell-torsion, (W) choice-independence, (T) teeth; eight cells, 28 units (15 at ell = 2, 3).
    ValidationRegistry::register_validation("HeckeZCrossedHomOracleValidation", &HeckeZCrossedHomOracleValidation, ValidationTier::Standard);
}

void RegisterSlot26() {
    RegisterSlot26HeckeZUp();
    RegisterSlot26HeckeZLocalPar();
    RegisterSlot26HeckeZEisUp();
    RegisterSlot26HeckeHidaDiscLambda();
    RegisterSlot26HeckeZCrossedHomOracle();
}
