// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-22 registration (closed slot): the former per-package fragments Registry/Slot22/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot22() calls them in the old RegisterAllValidations() order.

// ===== Slot 22 LN23 ===========================================
// Slot-22 LN23 registration fragment (hecke-extensions slot 22). Owned by
// package LN23 (the weight-2 level-N Eisenstein route at p = 2, 3) ONLY:
// register this package's validations here and nowhere else in the shared
// structure. Called explicitly from RegisterAllValidations() (never a static
// initialiser); its sources are listed in cmake/slot22/LN23.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeLevelNCurveValidation.h"

void RegisterSlot22LN23() {
    // (LN23) the curve object M / (M^S + M^{ST}) at n = 0, every p (LN23_spec v0.2 section 10): the torsion gate (Z1),
    // H(Rel) in Rel at p = 2, 3, dense = sparse, Calegari--Emerton's oracles (41, 2), (2143, 3), Thm 1.1 / 1.2(i)+1.3(i)
    // grids, negative controls. Read-only; hooks HECKE_LEVEL_N_R3_SMALLP_CELLS, HECKE_LN23_*_NMAX, HECKE_LN23_HEAVY=1
    // (each with an EFFECTIVE line).
    ValidationRegistry::register_validation("HeckeLevelNCurveValidation", &HeckeLevelNCurveValidation, ValidationTier::Standard);
}

// ===== Slot 22 WAKE1 ===========================================
// Slot-22 WAKE1 registration fragment (hecke-extensions slot 22). Owned by
// package WAKE-1 (the Gamma_H(N) level and the tame-family Hecke algebra of
// Hsu--Pozzi--Wake--Wang-Erickson) ONLY: register this package's validations
// here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot22/WAKE1.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeTameFamilyValidation.h"

void RegisterSlot22WAKE1() {
    // (WAKE-1) Gamma_H(N) on the fixed model and the tame family at (p, l_0) = (5, 11), l_1 = 23, 67 (WAKE1_spec.md v0.2):
    // the table, the shipped-route and model-route comparisons, W-P1 ... W-P8 against the frozen Python oracle, the
    // reducibility ideal (V2), negative controls. Read-only; hook HECKE_TAMEFAMILY_HEAVY=1 (l_1 = 263; EFFECTIVE line).
    ValidationRegistry::register_validation("HeckeTameFamilyValidation", &HeckeTameFamilyValidation, ValidationTier::Standard);
}

// ===== Slot 22 P10 ===========================================
// Slot-22 P10 registration fragment (hecke-extensions slot 22). Owned by
// package P10 (the real Hida tower: Gamma_H(ell^r) with the tame character,
// the ordinary part over Lambda/(ell^K, Omega_r)) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called
// explicitly from RegisterAllValidations() (never a static initialiser); its
// sources are listed in cmake/slot22/P10.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeLambdaTowerValidation.h"
#include "../HeckeExt/HeckeNebentypusWildValidation.h"

void RegisterSlot22P10() {
    // (P10) the real Hida tower at ell <= 13, r <= 3 (P10_spec.md v0.3): G0 (the orientation, first; a failure stops the
    // validation), G1 - G14, G5', the negative controls. Read-only hooks HECKE_LAMBDA_TOWER_R3=1 (+ _RSS_CAP_GIB) and
    // HECKE_LAMBDA_TOWER_CELLS (EFFECTIVE lines).
    ValidationRegistry::register_validation("HeckeLambdaTowerValidation", &HeckeLambdaTowerValidation, ValidationTier::Standard);
    // (P10 (a)) the wild character models over O_{j,e} (W1 - W4: (NB7) - (NB9), (U20), the refusals). Read-only hook
    // HECKE_NEB_WILD_ALL=1 (O_{2,42} at (7, r = 3); EFFECTIVE line).
    ValidationRegistry::register_validation("HeckeNebentypusWildValidation", &HeckeNebentypusWildValidation, ValidationTier::Standard);
}

// ===== Slot 22 NFPRANK ===========================================
// Slot-22 NFPRANK registration fragment (hecke-extensions slot 22). Owned by
// package NF-PRANK (the p-rank of Cl(Q(N^{1/p})), p = 5 first) ONLY: register
// this package's validations here and nowhere else in the shared structure.
// Called explicitly from RegisterAllValidations() (never a static initialiser);
// its sources are listed in cmake/slot22/NFPRANK.cmake. Created mid-slot with the
// two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../NumberField/PRankValidation.h"

void RegisterSlot22NFPRANK() {
    // (NF-PRANK) Cl(Q(N^{1/5})) by a Minkowski-complete relation computation (NF-PRANK_spec.md sec. 8):
    // G1 field data + Round 2 at (401,5), G2 Chevalley on the 1240 frozen cells, G3 the four prototype
    // cells field by field (+ descent at (61,5), the P_1 P_2^4 path at (101,5)), G4 negative controls.
    // Read-only hooks NF_PRANK_CE=CE1|CE2|CE1C|CE2C|all, NF_PRANK_CELLS, NF_PRANK_DENSE (EFFECTIVE line).
    ValidationRegistry::register_validation("NumberFieldPRankValidation", &NumberFieldPRankValidation, ValidationTier::Standard);
}

// ===== Slot 22 P6 ===========================================
// Slot-22 P6 registration fragment (hecke-extensions slot 22). Owned by package P6
// (the Birch route on Gamma_0(N) by Shapiro; the tame twists of the measure) ONLY:
// register this package's validations here and nowhere else in the shared structure.
// Called explicitly from RegisterAllValidations() (never a static initialiser); its
// sources are listed in cmake/slot22/P6.cmake. Created mid-slot with the two one-line
// shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../HeckeExt/LValue/HeckeLValueShapiroValidation.h"

void RegisterSlot22P6() {
    // (P6) P6_spec.md: S0 the route at ModelId::Gamma0NShapiro (refusal lifted), S1 ranks of W_N^{+-} vs Eichler-Shimura,
    // S2 Res / Cores / Hecke and W_1 = the cone symbols, S3 newform lines vs eta products, S4 11a vs P9, S5 the level-N
    // measure, S6 the tame twists of Delta's measure (T in {2,3,5,7,11}), S7 negative controls; S8 the read-only scale arm HECKE_SHAPIRO_CELLS (EFFECTIVE line).
    ValidationRegistry::register_validation("HeckeLValueShapiroValidation", &HeckeLValueShapiroValidation, ValidationTier::Standard);
}

// ===== Slot 22 LES1 ===========================================
// Slot-22 LES1 registration fragment (hecke-extensions slot 22). Owned by package LES-1 (the fibre maps of the
// level-change cone, [C] v1.22.0-pre.3 sec:nebles) ONLY: register this package's validations here and nowhere else in
// the shared structure. Called explicitly from RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot22/LES1.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeLevelNormalityValidation.h"

void RegisterSlot22LES1() {
    // (LES-1) (LES1) of rmk:lescheckable: prop:gamma0normal on N <= 60, M <= 24; the Gamma_1 steps and the p-tower table of
    // prop:hidanormalsteps; Gamma^2, Gamma^3, Gamma(M) (rmk:normalN1); the model tags (prop:modeltorsion);
    // RestrictionFibreMap = LevelChangeMap; the fibre identities of obl:levelconeexact (i); fibre-level negative controls.
    ValidationRegistry::register_validation("HeckeLevelNormalityValidation", &HeckeLevelNormalityValidation, ValidationTier::Standard);
}

// ===== Slot 22 MP ===========================================
// Slot-22 MP registration fragment (hecke-extensions slot 22). Owned by the
// Moerdijk--Pronk packages (MP-1a, MP-1b, MP-T, MP-2, MP-3, MP-4; MP_impl_plan.md v0.2.1) ONLY: register
// this package's validations here and nowhere else in the shared structure.
// Called explicitly from RegisterAllValidations() (never a static initialiser);
// its sources are listed in cmake/slot22/MP.cmake.
#include "../ValidationRegistry.h"
#include "../Orbifold/MP/OrbifoldMPAllDegreesValidation.h"
#include "../Orbifold/MP/OrbifoldMPBoundaryValidation.h"
#include "../Orbifold/MP/OrbifoldMPComparisonValidation.h"
#include "../Orbifold/MP/OrbifoldMPHeckeValidation.h"
#include "../Orbifold/MP/OrbifoldMPLevelConeValidation.h"
#include "../Orbifold/MP/OrbifoldMPNerveH3Slow.h"
#include "../Orbifold/MP/OrbifoldMPNerveValidation.h"
#include "../Orbifold/MP/OrbifoldMPRefinementValidation.h"
#include "../Orbifold/MP/OrbifoldMPRulesValidation.h"
#include "../Orbifold/MP/OrbifoldMPTeardropValidation.h"

void RegisterSlot22MP() {
    // (MP-1a) The decidable MP predicates on M7, R_Z, R_zeta, R_Z+, R_good and the choice-free MP set S_<=3
    // (MP_predictions.md P1): adapted (subgroup form), good, collar, (SC), invariant order, (V,E,F,chi), |S_k|,
    // the mu-form counts, NC1 / NC2. Scratch hook ORBIFOLD_MP_DUMP=<dir> (EFFECTIVE line); otherwise read-only.
    ValidationRegistry::register_validation("OrbifoldMPRulesValidation", &OrbifoldMPRulesValidation, ValidationTier::Fast);
    // (MP-1b) The twisted normalised cochains of the choice-free MP set on M7 and R_Z (MP_predictions.md P2):
    // nerve H^0, H^1 over F_ell (ell <= 7, n <= 12), H^2 (ell <= 5, n <= 3; n in {4, 6} behind the read-only
    // ORBIFOLD_MP_CELLS, EFFECTIVE line), Z/ell^k lengths vs C_I's UCT, d o d = 0, the untwisted-d_0 NC;
    // primary oracle the frozen mv_check.out (MvCheckOracle.inc).
    ValidationRegistry::register_validation("OrbifoldMPNerveValidation", &OrbifoldMPNerveValidation, ValidationTier::Standard);
    // (MP-T) Moerdijk--Pronk's teardrop Tear-n in the G^k-form (a complex of groups, not a global quotient): the 9 cells
    // n in {2,3,4} x ell in {2,3,5} (MP_predictions.md P3, 81 values), d o d = 0 exhaustively to k = 4; NCs: swapped mu,
    // wrong cone-point order, trivial isotropy. Read-only.
    ValidationRegistry::register_validation("OrbifoldMPTeardropValidation", &OrbifoldMPTeardropValidation, ValidationTier::Fast);
    // (MP-2) Phi : Z[F'] -> Iso certified to degree 3 on M7 and R_Z; Phi^* an iso on H^{<=2} (F_ell, Z/ell^k) vs the frozen
    // mv_check.out; eps^* from the shipped Bredon complex (iso at ell in {5,7}, the defect at {2,3}, the odd-n witness); the
    // hybrid LocalModel = C_I = nerve; NC perturbed Phi_1 (MP_predictions.md P4.3, P4.5). Read-only.
    ValidationRegistry::register_validation("OrbifoldMPComparisonValidation", &OrbifoldMPComparisonValidation, ValidationTier::Standard);
    // (MP-2) The refinement map u : S(R_Z) -> S(M7): labels from the subdivision record, a cochain map, an iso on H^{<=2}
    // (= Phi^*_R_Z on cohomology), label independence, NC a label off by T (MP_predictions.md P4.6). Read-only.
    ValidationRegistry::register_validation("OrbifoldMPRefinementValidation", &OrbifoldMPRefinementValidation, ValidationTier::Standard);
    // (MP-2) All degrees: C_I and the hybrid H^0..7 vs mv_check.out, E^* = e u - iso H^q -> H^{q+2} (q = 2..5), NC ell.E^*,
    // and the MP_ALL_DEGREES certificate line (nerve_to=2 periodic_on=C_I via=thm:mpcomparison scope=bulk) (P4.1, P4.2). Read-only.
    ValidationRegistry::register_validation("OrbifoldMPAllDegreesValidation", &OrbifoldMPAllDegreesValidation, ValidationTier::Standard);
    // (MP-2) The nerve H^3 (ell in {2,3}, n in {0,2}, M7 and R_Z) on S_4, Phi^* iso in degree 3, F-MP1 F3 MPSet(D,4) = interned
    // S_4 (P4.4); minutes, <= 6 GB. EFFECTIVE line. Read-only.
    ValidationRegistry::register_validation("OrbifoldMPNerveH3Slow", &OrbifoldMPNerveH3Slow, ValidationTier::Slow);
    // (MP-3) The boundary MP set S(dX) on M7 and R_Z: H^.(S(dX)) = H^.(Gamma_d; Sym^n) (Shapiro, prop:mpboundary) vs the frozen
    // mp3_oracle.out and C_I's R_d; j^* with its label factor a cochain map, rank on H^{<=3} = C_I's restriction; the cone (the pair)
    // exact = C_I's cone, degree 3 on S_4; lem:ell2unbounded; NCs wrong boundary stabiliser, label off by T, factor dropped
    // (MP3_predictions.md P5). Read-only.
    ValidationRegistry::register_validation("OrbifoldMPBoundaryValidation", &OrbifoldMPBoundaryValidation, ValidationTier::Standard);
    // (MP-3) The sequence of a pair of groups for the five LES-0 (iv) pairs at ell in {2,3}: S_B (B-descriptors), Shapiro on one F'
    // (Coind = LES-1's M_{A,1}), res / cor (= LES-1's alpha / omega, cor o res = [A:B]), the forget-coset map with its non-central
    // label factor, the cone of res and its LES vs mp3_oracle.out and C_I(Coind); NCs factor dropped, wrong coset representative,
    // cor without rho, label off by T (MP3_predictions.md P6). Read-only.
    ValidationRegistry::register_validation("OrbifoldMPLevelConeValidation", &OrbifoldMPLevelConeValidation, ValidationTier::Standard);
    // (MP-4) Hecke operators T_p, p in {2,3,5,7}, ON THE MP MODEL (rmk:mphecke): the twisted chain datum on N Z[F'] by the MP
    // model's own filling (MPFill; no shipped tau table), certified; T_p = cor o conj_g o res with the label factor rho(g), a
    // cochain map; on H^0, H^1 (H^2 at ell in {2,3}) over F_ell and Z/ell^k: T_p Phi^* = Phi^* Theta^_g (C_I, IntegralHecke);
    // intrinsic charpolys vs the frozen mp4_oracle.out (amalgam cocycles, tau / Eichler--Shimura); the shipped Bredon T_p via
    // eps^*; M7 and R_Z; NCs label factor dropped, label factor mismatched with the datum (gT for g), perturbed tau (MP4_predictions.md P7).
    // Read-only.
    ValidationRegistry::register_validation("OrbifoldMPHeckeValidation", &OrbifoldMPHeckeValidation, ValidationTier::Standard);
}

// ===== Slot 22 G1 ===========================================
// Slot-22 G1 registration fragment (hecke-extensions slot 22). Owned by package G1
// (the int8 tensor-core IMMA GEMM in EngineCuda; P5G_prototype_report.md Sec. 7) ONLY:
// register this package's validations here and nowhere else in the shared structure.
// Called explicitly from RegisterAllValidations() (never a static initialiser); its
// sources are listed in cmake/slot22/G1.cmake. Created mid-slot with the two one-line
// shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../../Cuda/CudaImmaValidation.h"

void RegisterSlot22G1() {
    // (G1) the IMMA route vs engine::kernels::gemm by memcmp at p in {2, 3, 5, 13, 127} and Z/{4, 9, 125}: layouts, tails,
    // structured operands, K-chunking (forced intervals, the true int32 bound, the refusal above T(p)), routing (non-qualifying
    // moduli, imma = 0, no / short workspace, k = 0, uint64_t, the ENGINE_GPU_IMMA policy), GemmBatch, and the tape under
    // ENGINE_DEVICE=gpu (verify on / off, rollback ENGINE_GPU_IMMA=0). [SKIP] without a device or without an sm_80+ image.
    ValidationRegistry::register_validation("CudaImmaBitIdentityValidation", &CudaImmaBitIdentityValidation, ValidationTier::Standard);
}

// ===== Slot 22 G2 ===========================================
// Slot-22 G2 registration fragment (hecke-extensions slot 22). Owned by package G2
// (the device-resident IMMA elimination; P5G_prototype_report.md Sec. 7, G1_report.md Sec. 7) ONLY:
// register this package's validations here and nowhere else in the shared structure.
// Called explicitly from RegisterAllValidations() (never a static initialiser); its
// sources are listed in cmake/slot22/G2.cmake. Created mid-slot with the two one-line
// shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../../Cuda/CudaImmaElimValidation.h"

void RegisterSlot22G2() {
    // (G2) cuda::imma_rref vs engine::elim::rref (rank, pivots, RREF bytes, pivot_rows against the smallest-row rule and the
    // row rank profile), the gpu routes of tape::rref (+ kernel / image / factor_basis / solve) and of the DenseElim hook
    // vs the host engine, routing negatives (ENGINE_GPU_ELIM=0, Cpu, p = 131, Z/9, the size floor, uint64_t), launch counts,
    // negative controls. [SKIP] without a device or without an sm_80+ image.
    ValidationRegistry::register_validation("CudaImmaElimValidation", &CudaImmaElimValidation, ValidationTier::Standard);
}

// ===== Slot 22 BIN1 ===========================================
// Slot-22 BIN1 registration fragment (hecke-extensions slot 22). Owned by package BIN-1
// (native F_2 arithmetic, L23_master_plan.md Sec. 2 / Sec. 5) ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot22/BIN1.cmake. Created mid-slot with the two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../../ModArithmetic/Binary/BinaryF2Validation.h"

void RegisterSlot22BIN1() {
    // (BIN-1) engine::binary (BitMatrix; M4RI rref / echelon / kernel; M4RM product; portable / AVX2 /
    // AVX-512 row paths) and the opt-in ENGINE_BINARY_F2 route of engine::elim vs the engine at p = 2:
    // rank, pivots, RREF, kernel basis and product bit for bit on 500 random and >= 100 real A_Cq
    // matrices mod 2, a whole compute_cohomology() at ell = 2 route on == off, and the
    // BINARY_F2_TIMING lines at 1024 / 4096 (reported, not gated).
    ValidationRegistry::register_validation("BinaryF2Validation", &BinaryF2Validation, ValidationTier::Standard);
}

// ===== Slot 22 BIN2 ===========================================
// Slot-22 BIN2 registration fragment (hecke-extensions slot 22). Owned by package BIN-2
// (the Z/2^k ladder, L23_master_plan.md Sec. 2.2 / Sec. 5) ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot22/BIN2.cmake. Created mid-slot with the two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../Binary/Ring2kLadderValidation.h"

void RegisterSlot22BIN2() {
    // (BIN-2) engine::ring2k, the Z/2^k ladder R0-R5 (BitMatrix, u8, u16 / u32, u64, u128, N-word
    // truncated) with the nf::BigInt reference R6: Smith valuations (levels of blocked odd-pivot LU),
    // Howell form, kernel module and length on every rung vs ModCtx::prime_power(2, k) (k <= 63),
    // engine::ladic (k >= 64), R6 and the unblocked references, on random, planted and amalgam
    // (7-vertex A_Cq over Z/2^k; exact Sym^n coboundary / cocycle matrices of Z/4 *_{Z/2} Z/6)
    // matrices; wraparound fuzz of every rung's arithmetic and kernel path; the adaptive rung;
    // negative controls (planted carry / Newton bugs caught); RING2K_TIMING lines (not gated).
    ValidationRegistry::register_validation("Ring2kLadderValidation", &Ring2kLadderValidation, ValidationTier::Standard);
}

// ===== Slot 22 BIN3 ===========================================
// Slot-22 BIN3 registration fragment (hecke-extensions slot 22). Owned by package BIN-3
// (the degree-1 amalgam reader, L23_master_plan.md Sec. 2 / Sec. 5) ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot22/BIN3.cmake. Created mid-slot with the two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../Orbifold/L23/L23AmalgamReaderValidation.h"

void RegisterSlot22BIN3() {
    // (BIN-3) l23::read_lattice / h1_mod / oddn_row: Lambda^1, H^1(V/ell^k), T_p by the bar formula, iota at ell = 3,
    // ed_* and ed_*_pm on the Z/2^k ladder (engine::ring2k) and Z/3^k (engine::ladic), against the frozen outputs of
    // ladic2.py (even n <= 200, bit for bit) and oddn_l2.py (odd n <= 101), the [C] sec:oddnltwo closed forms, T_p = 1 + p,
    // the census gates, the adaptive precision rule and negative controls.  BIN3_ORACLE_ALL=1 (read-only) arms all 200 rows.
    ValidationRegistry::register_validation("L23AmalgamReaderValidation", &L23AmalgamReaderValidation, ValidationTier::Standard);
}

// ===== Slot 22 L231 ===========================================
// Slot-22 L231 registration fragment (hecke-extensions slot 22). Owned by package L23-1
// (the orbifold record and driver, census_l23_plan.md v0.3 Sec. 6 as amended by L23_master_plan.md v0.2) ONLY:
// register this package's validations here and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are listed in cmake/slot22/L231.cmake.
// Created mid-slot with the two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../Campaign/CampaignOrbifoldValidation.h"

void RegisterSlot22L231() {
    // (L23-1) `orbifold = c_i`: orbifold_cell/v1 records at ell = 2, 3 (C_I over F_ell and Z/ell^{2E}, the BIN-3 reader),
    // end to end on n <= 60, against the [C] sec:oddnltwo closed forms, the frozen BIN-3 oracles and the level-1 sum rules;
    // spec refusals, identity (fingerprint / `_s` id / hash input), resume, determinism, the fallback discipline.
    ValidationRegistry::register_validation("OrbifoldCampaignL23Validation", &OrbifoldCampaignL23Validation, ValidationTier::Standard);
}

// ===== Slot 22 L233 ===========================================
// Slot-22 L233 registration fragment (hecke-extensions slot 22). Owned by package L23-3 (the census validations of the
// orbifold section, census_l23_plan.md v0.3 Sec. 6 / L23_1_audit.md Sec. 7) ONLY: register this package's validations here
// and nowhere else in the shared structure. Called explicitly from RegisterAllValidations() (never a static initialiser);
// its sources are listed in cmake/slot22/L233.cmake. Created mid-slot with the two one-line shared edits.
#include "../ValidationRegistry.h"
#include "../Campaign/OrbifoldCensusL23Validation.h"

void RegisterSlot22L233() {
    // (L23-3) the level-1 q-expansion sum-rule oracle (C++) vs the 200 frozen rows; the amalgam layers (H^1_Br, H^1_!, T^1/ell,
    // L) vs the independent prototype and the MV oracle; orbifold_cell/v1 cells n <= 24 with the new gates; negative controls.
    ValidationRegistry::register_validation("OrbifoldCensusL23Validation", &OrbifoldCensusL23Validation, ValidationTier::Standard);
    // (L23-3) the reader vs the sum rules on every even n <= 200 (both primes, the iota halves), the C_I (lat) reading of the
    // ed_* on sampled n, the layers vs MV to n = 200, the oracle at the L23-4 calibration cells (n = 240 ... 700).
    ValidationRegistry::register_validation("OrbifoldCensusL23SlowValidation", &OrbifoldCensusL23SlowValidation, ValidationTier::Slow);
}

void RegisterSlot22() {
    RegisterSlot22LN23();
    RegisterSlot22WAKE1();
    RegisterSlot22P10();
    RegisterSlot22NFPRANK();
    RegisterSlot22P6();
    RegisterSlot22LES1();
    RegisterSlot22MP();
    RegisterSlot22G1();
    RegisterSlot22G2();
    RegisterSlot22BIN1();
    RegisterSlot22BIN2();
    RegisterSlot22BIN3();
    RegisterSlot22L231();
    RegisterSlot22L233();
}
