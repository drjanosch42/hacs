// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-24 registration (closed slot): the former per-package fragments Registry/Slot24/<pkg>.cpp folded verbatim
// (S24 CL-10b), each package function kept; RegisterSlot24() calls them in the old RegisterAllValidations() order.

// ===== Slot 24 ELLW ===========================================
// Slot-24 ELL-W registration fragment (hecke-extensions slot 24). Owned by package ELL-W (ell > 23170, the [C] lem:bounds
// reach; docs/implementation/slot24/00-slot24-plan.md Sec. 2 item 5, docs/implementation/slot24/ELLW_report.md) ONLY:
// register this package's validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser), after slot 23's fragments; its sources are listed in
// cmake/slot24/ELLW.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeWideTauValidation.h"

void RegisterSlot24ELLW() {
    // (ELL-W) the wide tier of the tau table: the tiered helpers (fault injection, ForceNarrow), forced-wide = narrow on
    // ell <= 23167, the boundary ell = 23173, T_30011 / U_30011 at n = 10 against the q-expansion of Delta.
    ValidationRegistry::register_validation("HeckeWideTauValidation", &HeckeWideTauValidation, ValidationTier::Standard);
    // (ELL-W) the heavier arms: forced-wide = narrow at ell = 23167 (the largest narrow prime), T_30011 mod 13 at n = 10.
    ValidationRegistry::register_validation("HeckeWideTauSlowValidation", &HeckeWideTauSlowValidation, ValidationTier::Slow);
}

// ===== Slot 24 MT1 ===========================================
// Slot-24 MT-1 registration fragment (hecke-extensions slot 24). Owned by package MT-1 (module-route multithreading for
// single extreme cells; docs/implementation/slot24/00-slot24-plan.md Sec. 2 item 2, docs/implementation/slot24/MT1_report.md)
// ONLY: register this package's validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser), after RegisterSlot24ELLW(); its sources are listed in
// cmake/slot24/MT1.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeModuleStreamValidation.h"

void RegisterSlot24MT1() {
    // (MT-1) HECKE_MODULE_MT: cyclic_decomposition's pivot-search exit and tiled search, the row splits, the read-off /
    // rhs / row-block GEMM splits, the per-column checks, the (sigma, sigma_k) assembly tasks, at
    // consumer-pool sizes 1 / 2 / 7 / 16 against the pre-MT route (unit, object and SAT_DEFECT cell level).
    ValidationRegistry::register_validation("HeckeModuleMtValidation", &HeckeModuleMtValidation, ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 2)
}

// ===== Slot 24 FNTT3 ===========================================
// Slot-24 FNTT3 registration fragment (hecke-extensions slot 24). Owned by package FNTT3 (the forms NTT's third prime;
// docs/implementation/slot24/FNTT_report.md Sec. 5 item 1, docs/implementation/slot24/FNTT3_report.md) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations()
// (never a static initialiser), after RegisterSlot24MT1(); its sources are listed in cmake/slot24/FNTT3.cmake
// (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeFormsNtt3Validation.h"

void RegisterSlot24FNTT3() {
    // (FNTT3) the three-prime chunked forms NTT (P3 = 45*2^23+1, Garner CRT) against the naive product, the two-prime
    // engine, exact dot products / closed forms / Eisenstein identities past the two-prime bound, whole spaces (HeckeSturmBound,
    // HeckeOnCusp); the FNTT refusal lifted, and restored by the rollback HECKE_FORMS_NTT3=0.
    ValidationRegistry::register_validation("HeckeFormsNtt3Validation", &HeckeFormsNtt3Validation, ValidationTier::Standard);
}

// ===== Slot 24 SATPERF ===========================================
// Slot-24 SATPERF registration fragment (hecke-extensions slot 24). Owned by package SATPERF (the serial post-module
// part of big sat cells; docs/implementation/slot24/SATPERF_report.md) ONLY: register this package's validations here and
// nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static initialiser),
// after RegisterSlot24MT1(); its sources are listed in cmake/slot24/SATPERF.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeModuleStreamValidation.h"

void RegisterSlot24SATPERF() {
    // (SATPERF) HECKE_SAT_PERF: the sat cell's per-level memo of the whole-module derivations (charpolys, RationalSystems,
    // joint idempotents, blocks, Smith exponents, the pre-pass handed to the certificate) and the End(N) product's
    // uint64_t / row-split route, at consumer-pool sizes 1 / 2 / 4 against the rollback (SAT_DEFECT lines byte for byte).
    ValidationRegistry::register_validation("HeckeSatPerfValidation", &HeckeSatPerfValidation, ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 16)
}

// ===== Slot 24 DESCENT0 ===========================================
// Slot-24 DESCENT0 registration fragment (hecke-extensions slot 24). Owned by package DESCENT D0 (the level-1 integral
// groups by the owner's descent; docs/implementation/slot24/DESCENT0_report.md) ONLY: register this package's validations
// here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() after
// RegisterSlot24SATPERF(); its sources are listed in cmake/slot24/DESCENT0.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/IntegralDescentGroupsValidation.h"

void RegisterSlot24DESCENT0() {
    // (D0) H^q(SL_2(Z); Sym^n Z^2), q <= 5, glued from the good-prime rank, the C_7 K-mode at the Dickson primes,
    // Iso^(3) on ring3k and Iso^(2) on ring2k ([C, prop:descent](2), [C, prop:certifiedeell]); oracles Appendix A and
    // descent_oracle_n128.tsv.
    ValidationRegistry::register_validation("IntegralDescentGroupsValidation", &IntegralDescentGroupsValidation,
                                            ValidationTier::Standard);
    // DESCENT_COST at n in {100, 200, 300} (HECKE_DESCENT_CELLS), the oracle on every n <= 128, the level-N RSS probe
    // (HECKE_DESCENT_LEVELN_CELLS). Read-only hooks.
    ValidationRegistry::register_validation("IntegralDescentGroupsSlowValidation", &IntegralDescentGroupsSlowValidation,
                                            ValidationTier::Slow);
}

// ===== Slot 24 DESCENT1 ===========================================
// Slot-24 DESCENT1 registration fragment (hecke-extensions slot 24). Owned by package DESCENT D1 (the level-N integral
// groups by the owner's descent; docs/implementation/slot24/DESCENT1_report.md) ONLY: register this package's validations
// here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() after
// RegisterSlot24DESCENT0(); its sources are listed in cmake/slot24/DESCENT1.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/IntegralDescentLevelNValidation.h"

void RegisterSlot24DESCENT1() {
    // (D1) H^q(Gamma_0(N); Sym^n Z^2), q <= 5: closed-form ranks, the sparse cokernel over Z/ell^K at B(n, N), the ell = 3
    // shortcut or Iso^(3), Iso^(2) ([C, prop:descentN], [C, prop:badprimesN]); oracle descentN_oracle.tsv.
    ValidationRegistry::register_validation("IntegralDescentLevelNValidation", &IntegralDescentLevelNValidation,
                                            ValidationTier::Standard);
    // the oracle on every cell, DESCENT1_L1_COST, DESCENT1_N_COST (HECKE_DESCENT_N_CELLS, HECKE_DESCENT_REPEATS; read-only).
    ValidationRegistry::register_validation("IntegralDescentLevelNSlowValidation", &IntegralDescentLevelNSlowValidation,
                                            ValidationTier::Slow);
}

// ===== Slot 24 DESCENT2 ===========================================
// Slot-24 DESCENT2 registration fragment (hecke-extensions slot 24). Owned by package DESCENT D2 (res / cor and the level-change
// long exact sequence over Z; docs/implementation/slot24/DESCENT2_report.md) ONLY: register this package's validations here and
// nowhere else in the shared structure. Called explicitly from RegisterAllValidations() after RegisterSlot24DESCENT1(); its
// sources are listed in cmake/slot24/DESCENT2.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/IntegralDescentMapsValidation.h"

void RegisterSlot24DESCENT2() {
    // (D2) res, cor, the level cone and its long exact sequence over Z for Gamma_0(MN) < Gamma_0(N) ([C, prop:descentmaps]);
    // oracle d2_maps_oracle.tsv.
    ValidationRegistry::register_validation("IntegralDescentMapsValidation", &IntegralDescentMapsValidation,
                                            ValidationTier::Standard);
    // the oracle on every cell, DESCENT2_COST (HECKE_DESCENT_PAIR_CELLS, HECKE_DESCENT_REPEATS; read-only).
    ValidationRegistry::register_validation("IntegralDescentMapsSlowValidation", &IntegralDescentMapsSlowValidation,
                                            ValidationTier::Slow);
}

// ===== Slot 24 DESCENT4 ===========================================
// Slot-24 DESCENT4 registration fragment (hecke-extensions slot 24). Owned by package DESCENT D4 (the boundary pair and the cone over
// Z, glued; docs/implementation/slot24/DESCENT4_report.md) ONLY: register this package's validations here and nowhere else in the
// shared structure. Called explicitly from RegisterAllValidations() after RegisterSlot24DESCENT2(); its sources are listed in
// cmake/slot24/DESCENT4.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/IntegralDescentPairValidation.h"

void RegisterSlot24DESCENT4() {
    // (D4) the pair sequence H^q_c -> H^q -> H^q(Gamma_d) over Z, glued; the boundary in closed form; T_p on the cone
    // ([C, sec:descentpair]); oracle d4_pair_oracle.tsv.
    ValidationRegistry::register_validation("IntegralDescentPairValidation", &IntegralDescentPairValidation, ValidationTier::Standard);
    // the cost of the pair (DESCENT4_COST): the 2-adic piece on dense Iso^(2) and Isob along a ladder, the refusal above the ceiling.
    ValidationRegistry::register_validation("IntegralDescentPairSlowValidation", &IntegralDescentPairSlowValidation,
                                            ValidationTier::Slow);
}

// ===== Slot 24 DESCENT5 ===========================================
// Slot-24 DESCENT5 registration fragment (hecke-extensions slot 24). Owned by package DESCENT D5 (the tree bulk, [C]
// prop:isomodeltree; docs/implementation/slot24/DESCENT5_report.md) ONLY: register this package's validations here and
// nowhere else in the shared structure. Called explicitly from RegisterAllValidations() after RegisterSlot24DESCENT4(); its
// sources are listed in cmake/slot24/DESCENT5.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/HeckeIsotropyModelTreeValidation.h"

void RegisterSlot24DESCENT5() {
    // (D5) the tree models: tree == dense on the D0/D1 cells, Phi / Psi of prop:isomodeltree(3), (4), the cost line.
    ValidationRegistry::register_validation("HeckeIsotropyModelTreeValidation", &HeckeIsotropyModelTreeValidation,
                                            ValidationTier::Standard);
    // every oracle cell on both routes, degree 8, DESCENT_COST route=tree (HECKE_DESCENT_TREE_CELLS, HECKE_DESCENT_REPEATS).
    ValidationRegistry::register_validation("HeckeIsotropyModelTreeSlowValidation", &HeckeIsotropyModelTreeSlowValidation,
                                            ValidationTier::Slow);
}

// ===== Slot 24 DESCENT3 ===========================================
// Slot-24 DESCENT3 registration fragment (hecke-extensions slot 24). Owned by package DESCENT D3 (Hecke T_p, p !| N, over Z as
// invariants, [C] sec:descenthecke; docs/implementation/slot24/DESCENT3_report.md) ONLY: register this package's validations here and
// nowhere else in the shared structure. Called explicitly from RegisterAllValidations() after RegisterSlot24DESCENT5(); its sources are
// listed in cmake/slot24/DESCENT3.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/IntegralDescentHeckeValidation.h"

void RegisterSlot24DESCENT3() {
    // (D3) char_Z(T_p | H^1/tors) by residues and Garner; the torsion Hecke modules (tree transport at 2, 3; cokernel at l >= 5).
    ValidationRegistry::register_validation("IntegralDescentHeckeValidation", &IntegralDescentHeckeValidation, ValidationTier::Standard);
    // larger cells and the DESCENT3_COST lines (HECKE_DESCENT_HECKE_CELLS, read-only).
    ValidationRegistry::register_validation("IntegralDescentHeckeSlowValidation", &IntegralDescentHeckeSlowValidation, ValidationTier::Slow);
}

// ===== Slot 24 DESCENT5c ===========================================
// Slot-24 DESCENT5c registration fragment (hecke-extensions slot 24). Owned by package DESCENT D5c (the tree cusp pair, [C]
// sec:treepair; docs/implementation/slot24/DESCENT5c_report.md) ONLY: register this package's validations here and nowhere else in
// the shared structure. Called explicitly from RegisterAllValidations() after RegisterSlot24DESCENT3(); its sources are listed in
// cmake/slot24/DESCENT5c.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/IntegralDescentTreePairValidation.h"

void RegisterSlot24DESCENT5c() {
    // (D5c) the tree cusp pair: groups vs the dense pair and the oracle, j_Phi / j_Psi, T_p by transport, negative controls, rollback.
    ValidationRegistry::register_validation("IntegralDescentTreePairValidation", &IntegralDescentTreePairValidation, ValidationTier::Standard);
    // the cost ladder past D4's wall (DESCENT5c_COST), the eta identity's cost (HECKE_DESCENT_TREEPAIR_CELLS).
    ValidationRegistry::register_validation("IntegralDescentTreePairSlowValidation", &IntegralDescentTreePairSlowValidation,
                                            ValidationTier::Slow);
}

// ===== Slot 24 DESCENT6 ===========================================
// Slot-24 DESCENT6 registration fragment (hecke-extensions slot 24). Owned by package DESCENT D6 (records, campaign and the
// staged golden event; docs/implementation/slot24/DESCENT6_report.md) ONLY: register this package's validations here and
// nowhere else in the shared structure. Called explicitly from RegisterAllValidations() after RegisterSlot24DESCENT5c(); its
// sources are listed in cmake/slot24/DESCENT6.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/IntegralDescentRecordValidation.h"

void RegisterSlot24DESCENT6() {
    // (D6) coefficient_ring = Z un-reserved: the setting, the refusal closure, the record against the D1 / D3 / D4 oracles, its
    // schema and hash input, the campaign's descent cells, the staged golden points against their prediction.
    ValidationRegistry::register_validation("IntegralDescentRecordValidation", &IntegralDescentRecordValidation, ValidationTier::Standard);
}

// ===== Slot 24 PARD2 ===========================================
// Slot-24 PARD2 registration fragment (hecke-extensions slot 24). Owned by package PAR-D2 (the blocked, threaded, SIMD
// maps::SmithForm with factored transforms; docs/implementation/slot24/PARD2_report.md) ONLY. Called explicitly from
// RegisterAllValidations() after RegisterSlot24DESCENT6(); its sources are listed in cmake/slot24/PARD2.cmake
// (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/IntegralDescentSmithBlockedValidation.h"

void RegisterSlot24PARD2() {
    // the blocked Smith form == D2's scalar oracle, bit for bit (random and real inputs, thread counts, SIMD paths, the callers);
    // lever HECKE_DESCENT_SMITH_BLOCKED=0 (read-only rollback)
    ValidationRegistry::register_validation("IntegralDescentSmithBlockedValidation", &IntegralDescentSmithBlockedValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 24 PAR2K ===========================================
// Slot-24 PAR2K registration fragment (hecke-extensions slot 24). Owned by package PAR-2K (the threaded ring2k / ring3k Smith of
// the integral descent; docs/implementation/slot24/PAR2K_report.md) ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from RegisterAllValidations() after RegisterSlot24DESCENT6(); its sources are
// listed in cmake/slot24/PAR2K.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/DescentSmithMtValidation.h"

void RegisterSlot24PAR2K() {
    // (PAR-2K) the threaded ring2k / ring3k Smith == serial, bit for bit, at T = 1, 2, 7, 16 (HECKE_DESCENT_SMITH_MT).
    ValidationRegistry::register_validation("DescentSmithMtValidation", &DescentSmithMtValidation, ValidationTier::Standard);
}

// ===== Slot 24 PARASM ===========================================
// Slot-24 PARASM registration fragment (hecke-extensions slot 24). Owned by package PAR-ASM (the descent's remaining serial
// assembly stages made parallel; docs/implementation/slot24/PARASM_report.md) ONLY. Called explicitly from RegisterAllValidations()
// after RegisterSlot24PAR2K(); its sources are listed in cmake/slot24/PARASM.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/IntegralDescentParAsmValidation.h"

void RegisterSlot24PARASM() {
    // the threaded tree-pair assembly, the structured cone maps and the concurrent ell >= 5 primes == the serial code at 1/2/7/16
    // workers; levers HECKE_DESCENT_PAR_ASM=0, HECKE_DESCENT_PAR_ELL=0, HECKE_DESCENT_EMBED_MAP=0 (read-only rollbacks)
    ValidationRegistry::register_validation("IntegralDescentParAsmValidation", &IntegralDescentParAsmValidation, ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 18)
}

// ===== Slot 24 PARGPU ===========================================
// Slot-24 PARGPU registration fragment (hecke-extensions slot 24). Owned by package PAR-GPU (the opt-in device-resident 2-adic
// Smith LU of the integral descent; docs/implementation/slot24/PARGPU_report.md) ONLY: register this package's validations here
// and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() after RegisterSlot24PAR2K(); its
// sources are listed in cmake/slot24/PARGPU.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/DescentSmithGpuValidation.h"

void RegisterSlot24PARGPU() {
    // (PAR-GPU) the device-resident ring2k LU (ENGINE_DEVICE=gpu, ENGINE_GPU_SMITH2K) == the host LU, entry for entry; [SKIP] without a device.
    ValidationRegistry::register_validation("DescentSmithGpuValidation", &DescentSmithGpuValidation, ValidationTier::Standard);
}

// ===== Slot 24 PARLN ===========================================
// Slot-24 PARLN registration fragment (hecke-extensions slot 24). Owned by package PAR-LN (AssembleDescentN's sparse ell >= 5
// cokernels split into elimination and residual, the eliminations concurrent under a joint RSS rule;
// docs/implementation/slot24/PARLN_report.md) ONLY. Called explicitly from RegisterAllValidations() after RegisterSlot24PARGPU();
// its sources are listed in cmake/slot24/PARLN.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Integral/Descent/IntegralDescentParLnValidation.h"

void RegisterSlot24PARLN() {
    // the concurrent level-N sparse ell >= 5 eliminations == the serial code == the dense oracle at 1/2/7/16 workers; lever
    // HECKE_DESCENT_PAR_LN=0 (read-only rollback); measurement hook HECKE_DESCENT_PARLN_CELLS (read-only)
    ValidationRegistry::register_validation("IntegralDescentParLnValidation", &IntegralDescentParLnValidation, ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 8)
}

// ===== Slot 24 AVX2K ===========================================
// Slot-24 AVX2K registration fragment (hecke-extensions slot 24). Owned by package AVX2K (the AVX2 u32 GEMM microkernel for CPUs
// without AVX-512; docs/implementation/slot24/AVX2K_report.md) ONLY: register this package's validations here and nowhere else in
// the shared structure. Called explicitly from RegisterAllValidations() after RegisterSlot24PARGPU(); its sources are listed in
// cmake/slot24/AVX2K.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../../ModArithmetic/KernelAvx2BitIdentityValidation.h"

void RegisterSlot24AVX2K() {
    // (AVX2K) the AVX2 u32 GEMM microkernel == portable == AVX-512 == RefLinAlg on adversarial moduli, lane remainders and k-blocks.
    ValidationRegistry::register_validation("KernelAvx2BitIdentityValidation", &KernelAvx2BitIdentityValidation, ValidationTier::Fast);
}

// ===== Slot 24 VALSTREAM ===========================================
// Slot-24 VAL-IMPL registration fragment (hecke-extensions slot 24; the leaner `--all` of
// docs/implementation/slot24/VALSTREAM_audit_1030.md). Owned by package VAL-IMPL ONLY: register this package's validations
// here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static
// initialiser), after RegisterSlot24AVX2K(); its sources are listed in cmake/slot24/VALSTREAM.cmake
// (Registry/Slot24/README.md).  Each of the three names below is the slow sibling of a standard validation whose grid was split: the
// standard name and its sibling together run exactly the pre-split check set (`--all-full` runs both).
#include "../ValidationRegistry.h"
#include "../../ModArithmetic/BlockEliminationValidation.h"
#include "../../ModArithmetic/DenseElimValidation.h"
#include "../Hecke/HeckeLocalityValidation.h"

void RegisterSlot24VALSTREAM() {
    // (audit row 7) BlockEliminationValidation's "2x11 d=65" grid at the eight non-big (prime, width) passes, same seeds.
    ValidationRegistry::register_validation("BlockEliminationSlowValidation", &BlockEliminationSlowValidation, ValidationTier::Slow);
    // (audit row 9) HeckeLocalityValidation's arm 7 (c): the golden-grid sweep re-run under `locality = strict`.
    ValidationRegistry::register_validation("HeckeLocalitySlowValidation", &HeckeLocalitySlowValidation, ValidationTier::Slow);
    // (audit row 13) DenseElimValidation's u64 width below 2^32 and its parameter sweeps at 2^32-5 and 2.
    ValidationRegistry::register_validation("DenseElimSlowValidation", &DenseElimSlowValidation, ValidationTier::Slow);
}

// ===== Slot 24 CL1 ===========================================
// Slot-24 CL1 registration fragment (hecke-extensions slot 24). Owned by package CL-1 (the shared helpers;
// docs/implementation/slot24/CLEANUP_plan_1030.md) ONLY. Called explicitly from RegisterAllValidations() after RegisterSlot24AVX2K();
// its sources are listed in cmake/slot24/CL1.cmake (Registry/Slot24/README.md).
#include "../ValidationRegistry.h"
#include "../Util/SharedHelpersValidation.h"

void RegisterSlot24CL1() {
    // the CL-1 helpers reproduce the copies they replace (live copies where callable, transcribed tables otherwise); read-only
    ValidationRegistry::register_validation("SharedHelpersValidation", &SharedHelpersValidation, ValidationTier::Fast);
}

void RegisterSlot24() {
    RegisterSlot24ELLW();
    RegisterSlot24MT1();
    RegisterSlot24FNTT3();
    RegisterSlot24SATPERF();
    RegisterSlot24DESCENT0();
    RegisterSlot24DESCENT1();
    RegisterSlot24DESCENT2();
    RegisterSlot24DESCENT4();
    RegisterSlot24DESCENT5();
    RegisterSlot24DESCENT3();
    RegisterSlot24DESCENT5c();
    RegisterSlot24DESCENT6();
    RegisterSlot24PARD2();
    RegisterSlot24PAR2K();
    RegisterSlot24PARASM();
    RegisterSlot24PARGPU();
    RegisterSlot24PARLN();
    RegisterSlot24AVX2K();
    RegisterSlot24VALSTREAM();
    RegisterSlot24CL1();
}
