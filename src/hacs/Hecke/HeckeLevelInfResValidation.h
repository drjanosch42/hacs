// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_INF_RES_VALIDATION_H
#define HECKE_LEVEL_INF_RES_VALIDATION_H
// (hecke-extensions slot 23, LES-3) HeckeLevelInfResValidation (standard): Route B of [C] v1.23.0-pre.2 sec:nebles -- the tree
// complex of a level pair (Hecke/HeckeLevelInfRes.h), items (LES5), (LES7), (LES8) and the Route-B half of (LES9) of
// rmk:lescheckable; obl:levelconeexact (iv), (vii).  Predictions docs/implementation/slot22/LES_predictions.md section 3 and
// docs/implementation/slot23/LES3_predictions.md (both committed before this validation first ran).  Arms:
//   (AB)   Route A = Route B: LES-2's cone (LevelPairCone, its certificate) and the tree complex give the same h0A, h1A, h0B, h1B,
//          rank res^1, H^1(A,B), H^2(A,B) at 10 overlapping cells (ell >= 5), against the committed numbers;
//   (Z/p)  the Gamma_1-type Z/p steps at ell = p in {5, 7} (78 cells, LES3_predictions.md section 2): the closed forms of
//          prop:cycliccone (3)(c), the Jordan type J_p^{h1A-c} + J_1^c of R_sigma on H^1(B), Hhat^{1,2}(G; H^1(B)) = c, invariants and
//          coinvariants of dimension h1A, (R_sigma - 1) res^1 = 0, res^1 cor^1 = N_G, cor^1 res^1 = p, cor^1 (R_sigma - 1) = 0, rank cor^1
//          = h1A, the five-term identity; alpha, omega, R_sigma preserve the tree complexes (0 escaping generators);
//   (HIDA1) the first Hida step Gamma_0(N) cap Gamma_1(p) |> Gamma_0(pN) (G = (Z/p)^x, split): ker res^1 = 0, H^1(A) = H^1(B)^G, the
//          eigenspaces of R_sigma against the character models of A (prop:levelinfres (4)) with the derived eigenvalue convention;
//   (TOWER) Gamma^t_2 |> Gamma_0(p^2), n = 0, chi = omega^{a-2} (P10's character tables), over F_p and Z/p^2: LES8's numbers, Jordan
//          types (free / one non-free block), the Z/p^2 lengths (Howell), reduction mod p; Merel T_2 and same-level U_p squares;
//   (GAMMA) Gamma(ell) |> Gamma, ell in {5, 7}, n = 0..8, and the split cross cells: ker res^1 = O3 (FiniteGroupH1SL2, crossed
//          homomorphisms on SL_2(F_ell)), invariance of res, coinvariants vs cor (prop:levelinfres (5));
//   (MEREL) prop:levelconehecke (1)-(3) on H^1 by Merel's operators: T_2, T_3 at Gamma_0(25) < Gamma_0(5) and Gamma_{+-1}(11) |>
//          Gamma_0(11), same-level U_11, the diamond; level-raising U_5: U^B_5 res^1 = 0 at ell = 5 and the defect ranks = Route A's;
//   (NEG)  res_no_block (escaping generators), jordan_wrong_step; merel_odd_level recorded.
// No env hook (HECKE_LEVEL_CONE_CELLS belongs to HeckeLevelConeValidation); read-only; no golden object, record or setting touched.
#include <cstdint>

uint32_t HeckeLevelInfResValidation();

#endif // HECKE_LEVEL_INF_RES_VALIDATION_H
