// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_CONE_VALIDATION_H
#define HECKE_LEVEL_CONE_VALIDATION_H
// (hecke-extensions slot 23, LES-2) HeckeLevelConeValidation (standard): Route A of [C] v1.22.0-pre.3 sec:nebles -- the
// level cone Cone(alpha_{B/A}) on the fixed 7-vertex model (Hecke/HeckeLevelCone.h), items (LES2)-(LES4), (LES6) and the
// Route-A half of (LES9) of rmk:lescheckable; obl:levelconeexact (ii), (iii), (v), (vi), (viii), (ix).  Predictions
// docs/implementation/slot22/LES_predictions.md section 2 and docs/implementation/slot23/LES2_predictions.md (both committed
// before this validation first ran).  Per cell: alpha = RestrictionFibreMap (LES-1), both character models, the cone and its
// ConeCertificate (C1-C13), the six-term sequence eq:levelsixterm (H^0(A,B) = H^3(A,B) = 0, res^0 injective, exactness at
// every spot, the two dimension identities of lem:levelconeexact (3)), d alpha_* = alpha_* d on the stored equivariant bases,
// the connecting map re-derived by the zig-zag (= +res^q), and the predicted numbers.  Arms:
//   (I)    Gamma_0(p) < Gamma, chi = 1, p in {5, 7, 11}, ell in {5, 7, 13, 101}, n in {0, 2, 4, 10} (48 cells, all split);
//   (I')   Gamma_0(19), Gamma_0(29) at ell = 5, Gamma_0(13) at ell = 7, n in {0, 2, 4} (non-split, res^1 injective);
//   (II)   Gamma_0(25) < Gamma_0(5), chi' = chi o red, chi in {1, (./5), chi_4} (14 cells);
//   (III)  Gamma_1(11) |> Gamma_0(11), ell in {5, 7}, n in {0, 2, 4};
//   (III') Gamma_1(p) |> Gamma_0(p) at ell = p in {5, 7}, n in {0, 2, 4}, with h^1(Gamma_1(p)) = sum_i h^1(Gamma_0(p), omega^i);
//   (R3)   Gamma_1(5) < Gamma_0(5), chi = chi_4 (non-real) on A, ell = 5, n in {1, 3} (LES-1 audit R3);
//   (IV)   ell in {2, 3}: Gamma_0(4) < Gamma_0(2), Gamma_0(9) < Gamma_0(3), Gamma_1(4) < Gamma_0(4), Gamma^2, Gamma^3 < Gamma,
//          n = 0..4 (50 cells): the model= tags of prop:modeltorsion and the model dimensions (COMP, never O1);
//   (H)    prop:levelconehecke (1), (2): Phi = (T^A_p, T^B_p) with the transversal of B''\B commutes with D_cone as matrices
//          on the stored equivariant cone basis (T_2, T_3; the same-level U_11 at Gamma_1(11) |> Gamma_0(11)); the three map
//          families intertwine Phi; the alternating product of the characteristic polynomials over the sequence is 1;
//   (VI)   prop:levelconehecke (3): U^B_5 alpha_* = 5 dg_g on C^q_eq(L_A) (0 failing; = 0 at ell = 5), Gamma_0(25) < Gamma_0(5),
//          chi = 1; the H^1 square res U^A_5 - U^B_5 res RECORDED at all four cells (not gated, rule A6; at (ell = 7, n = 0)
//          labelled as the (LES9) prediction "non-zero"; audit R3);
//   (NEG)  obl:levelconeexact (ix): cone_sign, res_no_block, res_wrong_scalar, incompatible_chi caught; criterion_fails (audit R2:
//          U^B_5 alpha_* != alpha_* U^A_5 on C^q_eq(L_A) at the level-raising cell (ell = 7, n = 0), prop:levelconehecke (3)(c))
//          caught; and, gated but NOT a negative control ((ix)), hecke_two_transversals (formerly tagged "NEG hecke_fixed_tables";
//          audit R1, at Gamma_0(25) < Gamma_0(5), T_2, ell = 7, n = 2, m_A = 6): T^A_2 on C_eq(L_A)
//          does not depend on the transversal and the strict square holds with A's own (0 / 0), while the two operators differ
//          on the whole flat C^q (> 0) -- LES2_predictions.md section 7, LES2_audit.md section 3.2.
// Env hook (slot 23, LES-3; read-only): HECKE_LEVEL_CONE_CELLS="A:B:ell:n[:chi],..." REPLACES the arms by the listed cells (RunCell
// without predicted numbers or Hecke operators, plus Route B = Route A at ell >= 5) and prints EFFECTIVE HeckeLevelConeValidation
// arm=CELLS cells=<k>; a malformed token is refused by position (LES3_predictions.md section 6).  No golden object, record or setting
// is touched.
#include <cstdint>

uint32_t HeckeLevelConeValidation();

#endif // HECKE_LEVEL_CONE_VALIDATION_H
