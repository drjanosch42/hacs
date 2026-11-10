// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LVALUE_SHAPIRO_VALIDATION_H
#define HECKE_LVALUE_SHAPIRO_VALIDATION_H
// (hecke-extensions slot 22, P6) HeckeLValueShapiroValidation (standard): the Birch route on Gamma_0(N) by Shapiro and the tame
// twists of the measure -- Gamma0NShapiro.{h,cpp}; spec + predictions docs/implementation/slot22/P6_spec.md (committed first), oracle
// docs/implementation/slot22/p6/p6_oracle.py, its output p6/P6_oracle.out (frozen here).  Arms:
//   (S0) the route: BirchModularSymbolRoute accepts ModelId::Gamma0NShapiro (the RequireSupportedDomain refusal lifted); the
//        remaining refusals (Moerdijk-Pronk, character model, 7-vertex at N > 1, gcd(M, N) > 1, odd n, ell < 5, ell | N in measure(),
//        index_statement / cell() at level N);
//   (S1) ranks of W_N^{+-} (free) at N in {2,3,5,7,11} x n in {0..12} and N in {4,6} x n in {0,2,4}, ell = 13^3 and 7: the oracle's D table
//        and Eichler-Shimura (W^- = s_k, W^+ = s_k + c (k > 2), s_2 + c - 1; the ell = 7, k = 14 jump recorded);
//   (S2) Shapiro: Res(W_1) in W_N, Cores Res = N + 1, T_p Res = Res T_p, Cores T_p = T_p Cores; HeckeTpN at N = 1 = HeckeTp;
//        W_1 (Shapiro, N = 1) = span of the slot-19 cone symbols Psi(H^1_c) at m = 1, 3;
//   (S3) the newform lines (2,8) (3,6) (5,4) (7,4) (11,2): a_p = the oracle (= the eta products at N != 7), fnv of the symbol;
//   (S4) 11a vs P9 (Gamma0Weight2Symbol, AtCusp, MultiplicativeMeasure at ell = 11), ell in {5,7,11,13}, both signs;
//   (S5) the level-N MTT measure at seven cells (three-term, distribution, moment identity, fnv; BuildMeasureN = the tame builder at
//        T = 1; the route's measure() and evaluate_all_j at level N);
//   (S6) the tame twists of Delta's measure at ell = 13, T in {1,2,3,5,7,11} (T = 1 = BuildMeasure table by table);
//   (S7) negative controls (coset map without the permutation, rho(alpha) for rho(adj alpha), no ell^{k-2}, chi <-> chibar in E, alpha(1+ell)).
//   (S8) read-only scale arm HECKE_SHAPIRO_CELLS="N:n:ell:m,..." (EFFECTIVE HeckeLValueShapiroValidation arm=CELLS cells=<k>; malformed cells
//        refused by position): ranks of W_N^{+-} vs Eichler-Shimura (n + 1 < ell) and the cost of the reduced system.  Nothing runs unset.
// Prints SHAPIRO_* lines and P6_ARMS checks=<c> failing=<f>.  Writes nothing; no golden object.
#include <cstdint>

uint32_t HeckeLValueShapiroValidation();

#endif // HECKE_LVALUE_SHAPIRO_VALIDATION_H
