// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-25 registration (closed slot): the former per-package fragments Registry/Slot25/<pkg>.cpp folded verbatim
// (S25 FOLD, the slot-25 close; include paths one level up), each package function kept; RegisterSlot25() calls them in the
// old RegisterAllValidations() order.

// ===== Slot 25 HeckeZExport ===========================================
// Slot-25 HeckeZ EXPORT registration fragment (hecke-extensions slot 25). Owned by package EXPORT
// (docs/implementation/slot25/00-slot25-plan.md v0.2 Sec. 2.1, docs/implementation/slot25/EXPORT_report.md) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a
// static initialiser), after RegisterSlot24(); its sources are listed in cmake/slot25/HeckeZExport.cmake (Registry/Slot25/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/HeckeZExportValidation.h"

void RegisterSlot25HeckeZExport() {
    // (EXPORT) H^1(Gamma_0(N); Sym^n Z^2) (x) Z/ell^K with its T_p by cochain-level lifts, against the console's Z/ell^K route,
    // the descent record's groups (B2), char_Z, level_n, H^0 (the Bockstein) and the T3 certificate; basis-free invariants only.
    ValidationRegistry::register_validation("HeckeZExportValidation", &HeckeZExportValidation, ValidationTier::Standard);
}

// ===== Slot 25 HeckeZRational ===========================================
// Slot-25 HZ-RAT registration fragment (hecke-extensions slot 25). Owned by package HZ-RAT (the rational step of the Hecke structure
// over Z; docs/implementation/slot25/00-slot25-plan.md Sec. 2.2, docs/implementation/slot25/HZRAT_report.md) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static
// initialiser); its sources are listed in cmake/slot25/HeckeZRational.cmake (Registry/Slot25/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/HeckeZRationalValidation.h"

void RegisterSlot25HeckeZRational() {
    // (HZ-RAT) T1 (1,10), T2 (1,22), T5 (8,10) anemic, W44 (1,42): (C1), discriminants, (C2), g_P by residues; the untrusted finder's
    // two negative controls; the digests at 1 / 4 / 16 threads.
    ValidationRegistry::register_validation("HeckeZRationalValidation", &HeckeZRationalValidation, ValidationTier::Fast);
}

// ===== Slot 25 HeckeZUTag ===========================================
// Slot-25 HZ-UTAG registration fragment (hecke-extensions slot 25). Owned by package HZ-UTAG (chi_Q(U_p) and the two full-algebra tags
// at level N > 1; docs/implementation/slot25/00-slot25-plan.md Sec. 2.2, docs/implementation/slot25/HZRAT_report.md) ONLY: register
// this package's validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a
// static initialiser), after RegisterSlot25HeckeZRational(); its sources are listed in cmake/slot25/HeckeZUTag.cmake.
#include "../ValidationRegistry.h"
#include "../Integral/HeckeZUTagValidation.h"

void RegisterSlot25HeckeZUTag() {
    // (HZ-UTAG) T4 (2,6): chi_Q(U_2) and upper_bound_certified; T5 (8,10): non_reduced_certified.
    ValidationRegistry::register_validation("HeckeZUTagValidation", &HeckeZUTagValidation, ValidationTier::Fast);
}

// ===== Slot 25 HeckeZLocal ===========================================
// Slot-25 HZ-LOC registration fragment (hecke-extensions slot 25). Owned by package HZ-LOC (the local reads of the Hecke
// structure over Z; docs/implementation/slot25/00-slot25-plan.md v0.2 wave 2, docs/implementation/slot25/HZLOC_report.md) ONLY:
// register this package's validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser), after RegisterSlot25HeckeZUTag(); its sources are listed in
// cmake/slot25/HeckeZLocal.cmake.
#include "../ValidationRegistry.h"
#include "../Integral/HeckeZLocalValidation.h"

void RegisterSlot25HeckeZLocal() {
    // (HZ-LOC) T1, T2, the u64 item, (1,246) at 37, T3, T4 on the export, torsion routing, refusals.
    ValidationRegistry::register_validation("HeckeZLocalValidation", &HeckeZLocalValidation, ValidationTier::Standard);
    // (HZ-LOC slow sibling) the saturated census-4.0 rows of the level-1 list against the local block invariants.
    ValidationRegistry::register_validation("HeckeZLocalCensusXcheck", &HeckeZLocalCensusXcheck, ValidationTier::Slow);
}

// ===== Slot 25 HeckeMazurEis ===========================================
// Slot-25 H3 registration fragment (hecke-extensions slot 25). Owned by package H3 (Mazur's Eisenstein ideal at prime level,
// integrally; docs/implementation/slot25/H3_report.md) ONLY: register this package's validations here and nowhere else in the shared
// structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot25/HeckeMazurEis.cmake.
#include "../ValidationRegistry.h"
#include "../Integral/HeckeMazurEisValidation.h"

void RegisterSlot25HeckeMazurEis() {
    // (H3) Mazur's Eisenstein ideal: Hensel factor at 1 + q, good-prime finder, checks (a)-(d); env arms HECKE_MAZUR_EIS_ALL / _CELLS.
    ValidationRegistry::register_validation("HeckeMazurEisValidation", &HeckeMazurEisValidation, ValidationTier::Standard);
}

// ===== Slot 25 HeckeZExportZ ===========================================
// Slot-25 HeckeZ EXPORT-Z (step 1) registration fragment (hecke-extensions slot 25). Owned by package EXPORT-Z
// (docs/implementation/slot25/EXPORTZ_design.md, docs/implementation/slot25/EXPORTZ1_report.md) ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static
// initialiser), after RegisterSlot25HeckeZUTag(); its sources are listed in cmake/slot25/HeckeZExportZ.cmake (Registry/Slot25/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/HeckeZExportZValidation.h"

void RegisterSlot25HeckeZExportZ() {
    // (EXPORT-Z step 1) H^1(Gamma_0(N); Sym^n Z^2) over Z: the torsion block tau and the mixed block M (per ell, against the
    // exports' local free generators) under export_retraction/v1; F over Z is step 2.  Oracles: a cochain-level presentation
    // check by a fresh Smith form, splitting invariance, the record's invariants and torsion canon, hand-checked toys.
    ValidationRegistry::register_validation("HeckeZExportZValidation", &HeckeZExportZValidation, ValidationTier::Standard);
}

// ===== Slot 25 HeckeZTor ===========================================
// Slot-25 HZ-TOR registration fragment (hecke-extensions slot 25). Owned by package HZ-TOR (Tor terms, separation, splitting, the
// glue; docs/implementation/slot25/00-slot25-plan.md v0.2 wave 3, docs/implementation/slot25/HZTOR_report.md) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a
// static initialiser), after RegisterSlot25HeckeZExportZ(); its sources are listed in cmake/slot25/HeckeZTor.cmake.
#include "../ValidationRegistry.h"
#include "../Integral/HeckeZTorValidation.h"

void RegisterSlot25HeckeZTor() {
    // (HZ-TOR) (1,18), (1,20), (1,26), (1,30) at 5; K5 on the 54 cells of A3; K6 at (1,8)@2; B8 at 2, 3; T4, T5 on the export;
    // toy extensions; the EXPORT-Z interface; 1 vs 4 threads.
    ValidationRegistry::register_validation("HeckeZTorValidation", &HeckeZTorValidation, ValidationTier::Standard);
}

// ===== Slot 25 HeckeZRecord ===========================================
// Slot-25 HZ-REC registration fragment (hecke-extensions slot 25). Owned by package HZ-REC (+ HZ-EMIT) (the record
// integral_hecke_structure/v1 and its certificate emitter; docs/implementation/slot25/HZREC_report.md) ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static
// initialiser), after RegisterSlot25HeckeZExportZ(); its sources are listed in cmake/slot25/HeckeZRecord.cmake.
#include "../ValidationRegistry.h"
#include "../Integral/HeckeZRecordValidation.h"

void RegisterSlot25HeckeZRecord() {
    // (HZ-REC) the record on the toy cells T1-T5: the status table, the scans, threads, tamper arms, the ed.1 join.
    ValidationRegistry::register_validation("HeckeZRecordValidation", &HeckeZRecordValidation, ValidationTier::Standard);
    // (HZ-EMIT) HECKE_DUMP_HZ_CERT[_CELLS]: the scratch emitter of the certificate files (a no-op without the variable).
    ValidationRegistry::register_validation("HeckeZCertDumpValidation", &HeckeZCertDumpValidation, ValidationTier::Slow);
}

// ===== Slot 25 HeckeHidaEisLambda ===========================================
// Slot-25 H1 registration fragment (hecke-extensions slot 25). Owned by package H1 (the Eisenstein component over Lambda at irregular
// pairs, [U] part 6; docs/implementation/slot25/H1_report.md) ONLY: register this package's validations here and nowhere else in the
// shared structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot25/HeckeHidaEisLambda.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeHidaEisLambdaValidation.h"

void RegisterSlot25HeckeHidaEisLambda() {
    // (H1) prop:u-H11 on the C2 chains, (U1) on cohomology, K9's integers, G8 under heckez.G8 = hida; env arm HECKE_HIDA_EIS_ALL.
    // The 547 truncation is HeckeLambdaValidation's read-only arm HECKE_LAMBDA_CELLS.
    ValidationRegistry::register_validation("HeckeHidaEisLambdaValidation", &HeckeHidaEisLambdaValidation, ValidationTier::Standard);
}

// ===== Slot 25 HeckeZFreeBasis ===========================================
// Slot-25 EXPORT-Z step 2, packages EXZ2-P1a and EXZ2-P1b registration fragment (hecke-extensions slot 25). Owned by package EXZ2-P1a
// (docs/implementation/slot25/EXPORTZ2_design.md v2 Sec. 7, docs/implementation/slot25/EXZ2P1a_report.md) ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static
// initialiser), after RegisterSlot25HeckeZRecord(); its sources are listed in cmake/slot25/HeckeZFreeBasis.cmake (Registry/Slot25/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/HeckeZFreeBasisPairsValidation.h"
#include "../Integral/Descent/HeckeZFreeBasisSatValidation.h"

void RegisterSlot25HeckeZFreeBasis() {
    // (EXZ2-P1a) the global carrier Iso_T over Z (exact d^0, d^1; the transported T~_T in RNS lanes; IsoHeckeN::terms_Z()) and the K = 3
    // Dixon pairs (B_s, Phi_s) under iso_dixon/v3 (v2 until EXZ2-P1b); oracles: TreeModelChain(0)'s coboundary, rho_n, an exact Gauss-Jordan over Q, the cochain-map
    // identity; HECKE_EXZ2_CELLS (read-only) / HECKE_DUMP_EXZ2 (scratch) for the measurement.  No saturation, certificate or console (P1b-P3).
    ValidationRegistry::register_validation("HeckeZFreeBasisPairsValidation", &HeckeZFreeBasisPairsValidation, ValidationTier::Standard);
    // (EXZ2-P1b) the lattice sums, the saturation (local / composite / multi-word modes) and the det(Phi^T B) = +-1 certificate; oracles:
    // the audit's alt2 Delta_Sigma figures, an own Bareiss det, an in-validation integer kernel of Z^1 / B^1_sat; teeth 7, 2, 2147483659,
    // a 71-bit composite (multi-word) and SaturationCofactor; HECKE_EXZ2_CELLS (read-only) for the measurement arm.
    ValidationRegistry::register_validation("HeckeZFreeBasisSatValidation", &HeckeZFreeBasisSatValidation, ValidationTier::Standard);
}

// ===== Slot 25 HeckeZFreeBlock ===========================================
// Slot-25 EXPORT-Z step 2, package EXZ2-P2 registration fragment (hecke-extensions slot 25). Owned by package EXZ2-P2
// (docs/implementation/slot25/EXPORTZ2_design.md v2 Sec. 7.4, docs/implementation/slot25/EXZ2P2_report.md) ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static
// initialiser), after RegisterSlot25HeckeZFreeBasis(); its sources are listed in cmake/slot25/HeckeZFreeBlock.cmake (Registry/Slot25/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/HeckeZFreeBlockCrtValidation.h"

void RegisterSlot25HeckeZFreeBlock() {
    // (EXZ2-P2) the free block F_p over Z: RNS lanes (the per-lane LU route for Phi_0, tiles of B reduced on the fly, T~_Iso as GEMMs,
    // CUDA streams opt-in), the Sec. 1.6 bound (upward-rounded), the a-priori / a-posteriori proven CRT, garner_step and the BigInt lift, on
    // committed fixture bases (re-certified: det(Phi_cert^T B) = +-1); oracles: the model's exact F_p, CharpolyOverZ, the pinned charpolys;
    // bit identity across GEMM / Garner paths, threads, tiles and the GPU route; HECKE_EXZ2P2_FIXTURES (read-only) for the measurement.
    ValidationRegistry::register_validation("HeckeZFreeBlockCrtValidation", &HeckeZFreeBlockCrtValidation, ValidationTier::Standard);
}

// ===== Slot 25 HeckeZExportZ2 ===========================================
// Slot-25 HeckeZ EXPORT-Z step 2, package EXZ2-P3, registration fragment (hecke-extensions slot 25). Owned by package EXZ2-P3
// (docs/implementation/slot25/EXPORTZ2_design.md Sec. 7.5, docs/implementation/slot25/EXZ2P3_report.md) ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static
// initialiser), after RegisterSlot25HeckeZFreeBlock(); its sources are listed in cmake/slot25/HeckeZExportZ2.cmake (Registry/Slot25/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/HeckeZExportZ2Validation.h"

void RegisterSlot25HeckeZExportZ2() {
    // (EXZ2-P3) T_p = [[tau, M], [0, F]] on H^1 = T (+) Z^r with F EXACT over Z in the picked basis iso_dixon/v3 (P1b) by the proven
    // CRT (P2) and M in the same basis.  Oracles: the record's char_Z, commutation over Z and on T (+) Z^r, the export routes' own
    // T_p in B mod ell^K at the torsion primes and at primes without torsion, HZ-TOR on both bases, hand-checked toys.
    ValidationRegistry::register_validation("HeckeZExportZ2Validation", &HeckeZExportZ2Validation, ValidationTier::Standard);
    // the same on (8, 10), (1, 200), (23, 20), (311, 2) with stage timings (basis, lanes, bound, lift, exports, assembly)
    ValidationRegistry::register_validation("HeckeZExportZ2SlowValidation", &HeckeZExportZ2SlowValidation, ValidationTier::Slow);
}

// ===== Slot 25 NbConfirm ===========================================
// Slot-25 NBCONFIRM registration fragment (hecke-extensions slot 25). Owned by package NBCONFIRM (S3: the console cost
// confirmation, core + REPL/scripts; docs/console/NBCONFIRM_design.md, docs/console/NBCONFIRM_S3_report.md) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a
// static initialiser), after RegisterSlot25HeckeZExportZ2(); its sources are listed in cmake/slot25/NbConfirm.cmake.
#include "../ValidationRegistry.h"
#include "../Console/ConsoleCancelValidation.h"
#include "../Console/ConsoleCostModelValidation.h"

void RegisterSlot25NbConfirm() {
    // (NBCONFIRM S3) the C++ estimator against the frozen Python reference, the decision table, the per-mode policy, the
    // prompt-less byte-identity of the old refusals, the REPL prompt through a pty.
    ValidationRegistry::register_validation("ConsoleCostModelValidation", &ConsoleCostModelValidation, ValidationTier::Fast);
    // (NBCONFIRM S5) the cooperative cancel: every traced cancellation point of the seven long verbs, the re-run against a cold
    // child, the REPL's Ctrl-C on a pty, the served interrupt inside a lazy descent read.
    ValidationRegistry::register_validation("ConsoleCancelValidation", &ConsoleCancelValidation, ValidationTier::Standard);
}

void RegisterSlot25() {
    RegisterSlot25HeckeZExport();
    RegisterSlot25HeckeZRational();
    RegisterSlot25HeckeZUTag();
    RegisterSlot25HeckeZLocal();
    RegisterSlot25HeckeMazurEis();
    RegisterSlot25HeckeZExportZ();
    RegisterSlot25HeckeZTor();
    RegisterSlot25HeckeZRecord();
    RegisterSlot25HeckeHidaEisLambda();
    RegisterSlot25HeckeZFreeBasis();
    RegisterSlot25HeckeZFreeBlock();
    RegisterSlot25HeckeZExportZ2();
    RegisterSlot25NbConfirm();
}
