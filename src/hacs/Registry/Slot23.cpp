// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-23 registration (closed slot): the former per-package fragments Registry/Slot23/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot23() calls them in the old RegisterAllValidations() order.

// ===== Slot 23 LES2 ===========================================
// Slot-23 LES2 registration fragment (hecke-extensions slot 23). Owned by package LES-2 (Route A: the level cone on the fixed
// model, [C] v1.22.0-pre.3 sec:nebles) ONLY: register this package's validations here and nowhere else in the shared
// structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot23/LES2.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeLevelConeValidation.h"

void RegisterSlot23LES2() {
    // (LES-2) (LES2)-(LES4), (LES6) and the Route-A half of (LES9) of rmk:lescheckable: the level cone Cone(alpha_{B/A}), its
    // six-term sequence, the strict T_p / same-level U_p squares, level-raising U_p (gated identity, recorded H^1 square),
    // the ell in {2, 3} model tags, and the negative controls of obl:levelconeexact (ix).
    ValidationRegistry::register_validation("HeckeLevelConeValidation", &HeckeLevelConeValidation, ValidationTier::Standard);
}

// ===== Slot 23 PERF ===========================================
// Slot-23 PERF registration fragment (hecke-extensions slot 23). Owned by the PERF packages (P5, P2, P4a, P7) ONLY:
// register their validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are listed in cmake/slot23/PERF.cmake.
#include "../ValidationRegistry.h"
#include "../HostElimThreadDeterminismValidation.h"
#include "../GpuLeanSessionValidation.h"
#include "../../ModArithmetic/VnniGemmBitIdentityValidation.h"
#include "../Hecke/HeckeThetaRssDietValidation.h"

void RegisterSlot23PERF() {
    // (PERF-P5) the AVX-512 VNNI u8 x s8 route of gemm<uint32_t> for p <= 128 against the u32 microkernels, the portable
    // path and the RefLinAlg oracle; elimination and FlatMatrix Off / Auto / Force; the refusals (ENGINE_API.md Sec. 6.1).
    // (slot-24 G4) extended in place with the two-limb route for 128 < p <= 2^14 (Sec. 6.2); no new name.
    ValidationRegistry::register_validation("VnniGemmBitIdentityValidation", &VnniGemmBitIdentityValidation, ValidationTier::Standard);
    // (PERF-P2) the cone / iota^* host eliminations under engine::tape::ScopedHostElimHook on consumer pools of 1, 4, 16
    // workers: every value identical, the hook taken wherever the pool has > 1 worker (EngineRuntime.h).
    ValidationRegistry::register_validation("HostElimThreadDeterminismValidation", &HostElimThreadDeterminismValidation, ValidationTier::Standard);
    // (PERF-P4a) the RSS diet of HeckeOperatorOnK::compute_block() (program 1's arena and rho2 released at their last
    // readers, the chunked A-N3 residual, W released before the Theta_eq copy) against the legacy route, entrywise.
    ValidationRegistry::register_validation("HeckeThetaRssDietValidation", &HeckeThetaRssDietValidation, ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 11)
    // (PERF-P7) the VRAM-lean device session: memory-only GPU programs on the host, the arena slack; every value identical
    // to cpu (engine programs, the Hecke operator), the rule acting and its rollback ENGINE_GPU_LEAN=0 (ENGINE_API.md Sec. 12.9).
    ValidationRegistry::register_validation("GpuLeanSessionValidation", &GpuLeanSessionValidation, ValidationTier::Standard);
}

// ===== Slot 23 TER1 ===========================================
// Slot-23 TER1 registration fragment (hecke-extensions slot 23). Owned by package TER-1 (ell = 3 arithmetic:
// the bit-sliced F_3 rung T0 and Mod3k<N>, 00-slot23-plan.md Sec. 2) ONLY: register this package's validations
// here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static
// initialiser); its sources are listed in cmake/slot23/TER1.cmake. Created before WP-0 with the two one-line
// shared edits (slot-12 WP-14 precedent; Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../Binary/TernaryF3Validation.h"

void RegisterSlot23TER1() {
    // (TER-1) engine::binary::TritMatrix (bit-sliced F_3: rref / echelon / rank / kernel / product, portable /
    // AVX2 / AVX-512 row paths) and the opt-in ENGINE_TERNARY_F3 route of engine::elim vs the engine at p = 3,
    // bit for bit on random and real A_Cq matrices mod 3 and a whole compute_cohomology() at ell = 3; the
    // degree-1 reader's F_3 route (l23::h1_mod at ell = 3, k = 1) vs its Ops3<1> route cell by cell;
    // TERNARY_F3_TIMING / TERNARY_F3_READER lines (reported, not gated).  (S24 CL-G3: the Mod3k<N> part went with Mod3k.h.)
    ValidationRegistry::register_validation("TernaryF3Validation", &TernaryF3Validation, ValidationTier::Standard);
}

// ===== Slot 23 LES3 ===========================================
// Slot-23 LES3 registration fragment (hecke-extensions slot 23). Owned by package LES-3 (Route B: the tree complex of a level pair,
// inflation-restriction and the Hida p-tower, [C] v1.23.0-pre.2 sec:nebles) ONLY: register this package's validations here and
// nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources
// are listed in cmake/slot23/LES3.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeLevelInfResValidation.h"

void RegisterSlot23LES3() {
    // (LES-3) (LES5), (LES7), (LES8) and the Route-B half of (LES9) of rmk:lescheckable: Route A = Route B, the Gamma_1-type Z/p
    // steps (Jordan / Hochschild-Serre gates), the first Hida step, the tower cells over F_p and Z/p^2, Gamma(ell) against the
    // finite-group oracle O3, Merel's squares on H^1; obl:levelconeexact (iv), (vii).
    ValidationRegistry::register_validation("HeckeLevelInfResValidation", &HeckeLevelInfResValidation, ValidationTier::Standard);
}

// ===== Slot 23 Z2K ===========================================
// Slot-23 Z2K registration fragment (hecke-extensions slot 23). Owned by package Z2K (the ell = 2 (Z) phase of the
// orbifold census on the Z/2^k ladder engine::ring2k; owner ruling 2026-10-13, GPU_scout.md target 3, CPU part) ONLY:
// register this package's validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser), after RegisterSlot23LES3(); its sources are listed in
// cmake/slot23/Z2K.cmake (Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../Campaign/OrbifoldRing2kValidation.h"

void RegisterSlot23Z2K() {
    // (Z2K) the ell = 2 Smith readings and (U) kernel modules of CampaignOrbifold on engine::ring2k vs the engine route
    // (elim::invariant_factors / elim::kernel_module) on the real coboundaries of C_I, and whole ell = 2 records route on
    // vs off (bodies byte-identical after masking the sum-rule oracle's wall time, GPU_scout.md H-1); route counters
    // (non-vacuity), ell = 3 untouched, the masked comparator's negative controls, the EFFECTIVE l23_ring2k line.
    ValidationRegistry::register_validation("OrbifoldRing2kValidation", &OrbifoldRing2kValidation, ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 4)
}

// ===== Slot 23 C3 ===========================================
// Slot-23 C3 registration fragment (hecke-extensions slot 23). Owned by package C3 (the cyclic cubic field of prime
// conductor ell and the set Lambda-bar of arXiv:2610.17865; owner-approved 2026-10-15) ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a
// static initialiser), after RegisterSlot23Z2K(); its sources are listed in cmake/slot23/C3.cmake (Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../NumberField/CyclicCubicValidation.h"

void RegisterSlot23C3() {
    // (C3) the period polynomial and d_K = ell^2 for ell < 2000; h_K = [E_K : C] by saturation of the cyclotomic units
    // (Cusick's regulator bound), cross-checked by the analytic class number formula, for the 80 primes ell = 1 mod 3
    // below 1000; Lambda-bar = {163, 277, 349, 397, 547, 607, 709, 853, 937} there; determinism; saturation controls.
    ValidationRegistry::register_validation("NumberFieldCyclicCubicValidation", &NumberFieldCyclicCubicValidation, ValidationTier::Standard);
}

// ===== Slot 23 R5V ===========================================
// Slot-23 R5V registration fragment (hecke-extensions slot 23). Owned by package R5V (the SoA / SIMD kernels of
// ring2k's R5 rung) ONLY: register this package's validations here and nowhere else in the shared structure. Called
// explicitly from RegisterAllValidations() (never a static initialiser), after RegisterSlot23C3(); its sources are
// listed in cmake/slot23/R5V.cmake (Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../Binary/Ring2kR5SimdValidation.h"

void RegisterSlot23R5V() {
    // (R5V) every R5 SIMD kernel (limb GEMM / AXPY, mask-carry row add / sub / neg) on every available kernel path
    // equal entrywise to the scalar Traits<WordN<N>> bodies, for every N with a SIMD body; smith / howell /
    // kernel_module end to end SIMD vs scalar on random, planted and amalgam (Sym^n coboundary) matrices; the BIN-3
    // reader's Taylor shift; mutation controls (dropped borrow chain, no intermediate carries, dropped top-word mask, one flipped output bit) caught.
    ValidationRegistry::register_validation("Ring2kR5SimdValidation", &Ring2kR5SimdValidation, ValidationTier::Standard);
}

// ===== Slot 23 Z3K ===========================================
// Slot-23 Z3K registration fragment (hecke-extensions slot 23). Owned by package Z3K (the ell = 3 (Z) phase of the
// orbifold census on the Z/3^k rung engine::ring3k; GPU_scout.md target 3, CPU part at ell = 3; Z2K_report.md
// Sec. 4) ONLY: register this package's validations here and nowhere else in the shared structure. Called explicitly
// from RegisterAllValidations() (never a static initialiser), after RegisterSlot23C3(); its sources are listed in
// cmake/slot23/Z3K.cmake (Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../Campaign/OrbifoldMod3kValidation.h"

void RegisterSlot23Z3K() {
    // (Z3K) the rung engine::ring3k against its unblocked references and engine::elim (random, planted Smith forms,
    // rank-deficient, every blocking boundary); the ell = 3 Smith readings and (U) kernel modules of CampaignOrbifold
    // route on vs off on the real coboundaries of C_I; whole ell = 3 records route on vs off (bodies byte-identical after
    // the H-1 mask); route counters (non-vacuity), ell = 2 untouched, the refusals, the EFFECTIVE l23_mod3k line.
    ValidationRegistry::register_validation("OrbifoldMod3kValidation", &OrbifoldMod3kValidation, ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 14)
}

// ===== Slot 23 P4R ===========================================
// Slot-23 P4R registration fragment (hecke-extensions slot 23). Owned by package P4R (route 2's host RSS: rho2 streamed
// key by key) ONLY: register this package's validations here and nowhere else in the shared structure. Called
// explicitly from RegisterAllValidations() (never a static initialiser), after RegisterSlot23C3(); its sources are
// listed in cmake/slot23/P4R.cmake (Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeRho2StreamValidation.h"

void RegisterSlot23P4R() {
    // (P4R) route 2 of HeckeOperatorOnK::compute_block() streamed key by key (no per-degree rho2 memo) against the
    // materialised memo (rollback HECKE_RHO2_STREAM=0), entrywise on small cells; non-vacuity counters, a negative
    // control and target mutations that must throw O12(a) at exactly the mutated entry, streamed and materialised.
    ValidationRegistry::register_validation("HeckeRho2StreamValidation", &HeckeRho2StreamValidation, ValidationTier::Standard);
}

// ===== Slot 23 P4I ===========================================
// Slot-23 P4I registration fragment (hecke-extensions slot 23). Owned by package P4I (the iota phase's host RSS:
// admission + streamed split) ONLY: register this package's validations here and nowhere else in the shared structure.
// Called explicitly from RegisterAllValidations() (never a static initialiser), after RegisterSlot23P4R(); its sources
// are listed in cmake/slot23/P4I.cmake (Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeIotaStreamValidation.h"

void RegisterSlot23P4I() {
    // (P4I) HeckeInvolution::compute() with its tasks admitted under a byte budget, the streamed flat split and one copy of
    // the +- bases, against the materialised route (rollback HECKE_IOTA_STREAM=0), entrywise on small cells, over admission
    // budgets that put the boundaries inside a degree; legacy elimination; mutations that must be caught; a negative
    // control; a throwing thread-context installer (no hang, the injected error surfaces).
    ValidationRegistry::register_validation("HeckeIotaStreamValidation", &HeckeIotaStreamValidation, ValidationTier::Standard);
}

// ===== Slot 23 F3L ===========================================
// Slot-23 F3L registration fragment (hecke-extensions slot 23). Owned by package F3L (the F_3 field half of an ell = 3
// orbifold census cell: rho_n by the O(d^2) column recurrence, bit-sliced F_3 elimination; Z3K_report.md Sec. 5) ONLY:
// register this package's validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser), after RegisterSlot23P4I(); its sources are listed in
// cmake/slot23/F3L.cmake (Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../Campaign/OrbifoldF3lValidation.h"

void RegisterSlot23F3L() {
    // (F3L) rho_n recurrence vs the cubic sum (fields, chain rings, u64; ascending / descending / refused); theta_hat on
    // = off; whole ell = 3 (and ell = 2) records route on vs off (bodies byte-identical after the H-1 mask); counters, the
    // switch and the EFFECTIVE l23_f3l line.
    ValidationRegistry::register_validation("OrbifoldF3lValidation", &OrbifoldF3lValidation, ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 6)
}

// ===== Slot 23 BIN3O1 ===========================================
// Slot-23 BIN3O1 registration fragment (hecke-extensions slot 23). Owned by package BIN3O1 (BIN-3 open item O1: the degree-1
// amalgam reader's O(d^2) relation matrix, parity-split A, wavefront Taylor shifts, and h1_mod at ell = 2, k = 1 on F_2 /
// BitMatrix; docs/implementation/slot23/BIN3O1_report.md) ONLY: register this package's validations here and nowhere else in
// the shared structure. Called explicitly from RegisterAllValidations() (never a static initialiser), after RegisterSlot23Z3K();
// its sources are listed in cmake/slot23/BIN3O1.cmake (Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../Orbifold/L23/L23Bin3o1Validation.h"

void RegisterSlot23BIN3O1() {
    // (BIN3O1) the new route of l23::read_lattice / h1_mod / h1_layers / oddn_row against the old one (L23_BIN3O1=0), entrywise:
    // ring2k::kernel_shift_rows against the axpy sweep at the block boundaries on every kernel path, the relation matrix and
    // B^1, the structural operators, h1_mod (F_2 route), whole lattice cells (negative controls included), route counters.
    ValidationRegistry::register_validation("L23Bin3o1Validation", &L23Bin3o1Validation, ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 3)
}

// ===== Slot 23 G1 ===========================================
// Slot-23 G1 registration fragment (hecke-extensions slot 23). Owned by package G1 (lever G1 of
// docs/implementation/slot23/ELL_scout.md Sec. 3: one module run per sat / Loewy cell, the precision levels below it
// derived; [C] v1.23.0-pre.3 sec:satprecisionsec, lem:precisionreduction, obl:precisionreuse;
// docs/implementation/slot23/G1_report.md) ONLY: register this package's validations here and nowhere else in the shared
// structure. Called explicitly from RegisterAllValidations() (never a static initialiser), after RegisterSlot23BIN3O1();
// its sources are listed in cmake/slot23/G1.cmake (Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckePrecisionReuseValidation.h"

void RegisterSlot23G1() {
    // (G1) [C, obl:precisionreuse](v): the derived levels (ringalg::LevelCache) against the per-K route (HECKE_PRECISION_REUSE=0),
    // the whole printed output byte for byte on the standard grid ((108, 7) in place of (300, 7)), the (iii) checks and
    // their two negative controls, the injected-failure arm of (i).
    ValidationRegistry::register_validation("HeckePrecisionReuseValidation", &HeckePrecisionReuseValidation, ValidationTier::Standard);
    // (G1) the same at (300, 7) (eps = [9,16,21,25,27,27]) and (300, 11) on the sat route: ~ 378 s + 243 s on the oracle alone.
    ValidationRegistry::register_validation("HeckePrecisionReuseSlowValidation", &HeckePrecisionReuseSlowValidation, ValidationTier::Slow);
}

// ===== Slot 23 G2X ===========================================
// Slot-23 G2-X registration fragment (hecke-extensions slot 23). Owned by package G2-X (the forms cross-route's fast
// q-series product, docs/implementation/slot23/G2_plan.md Sec. 3; docs/implementation/slot23/G2X_report.md) ONLY: register
// this package's validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser), after RegisterSlot23G1(); its sources are listed in
// cmake/slot23/G2X.cmake (Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeFormsFastMulValidation.h"

void RegisterSlot23G2X() {
    // (G2-X) the fast q-series product (NTT + CRT, delayed-reduction schoolbook) and the E_4 / E_6 divisor sieve against
    // the naive O(N^2) routes (HECKE_FORMS_FASTMUL=0), byte for byte: random / extreme / boundary products, an
    // independent long-length identity, the cross-route spaces and operators, whole SAT_DEFECT cells.
    ValidationRegistry::register_validation("HeckeFormsFastMulValidation", &HeckeFormsFastMulValidation, ValidationTier::Standard);
}

// ===== Slot 23 G2 ===========================================
// Slot-23 G2 registration fragment (hecke-extensions slot 23). Owned by package G2 (the sat-path streaming / RSS
// levers of docs/implementation/slot23/G2_plan.md; G2-A / G2-B first, report docs/implementation/slot23/G2AB_report.md)
// ONLY: register this package's validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser), after RegisterSlot23G1(); its sources are listed in
// cmake/slot23/G2.cmake (Registry/Slot23/README.md).
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeModuleStreamValidation.h"

void RegisterSlot23G2() {
    // (G2-V) the module-stream levers (HECKE_MODULE_STREAM: the read-off solve on the unit pivots of C, the d o d GEMM,
    // cyclic_decomposition's interchanged clear) against the legacy route, in process: unit, object (entrywise) and
    // sat-cell (byte for byte) levels, the negative controls, the non-vacuity counters.
    ValidationRegistry::register_validation("HeckeModuleStreamValidation", &HeckeModuleStreamValidation, ValidationTier::Standard);
    // (G2-V) the same at (300, 1009) and (500, 43) on the sat route, with per-run time and peak RSS (recorded, not gated).
    ValidationRegistry::register_validation("HeckeModuleStreamSlowValidation", &HeckeModuleStreamSlowValidation, ValidationTier::Slow);
}

void RegisterSlot23() {
    RegisterSlot23LES2();
    RegisterSlot23PERF();
    RegisterSlot23TER1();
    RegisterSlot23LES3();
    RegisterSlot23Z2K();
    RegisterSlot23C3();
    RegisterSlot23R5V();
    RegisterSlot23Z3K();
    RegisterSlot23P4R();
    RegisterSlot23P4I();
    RegisterSlot23F3L();
    RegisterSlot23BIN3O1();
    RegisterSlot23G1();
    RegisterSlot23G2X();
    RegisterSlot23G2();
}
