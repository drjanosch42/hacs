// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_NERVE_VALIDATION_H
#define ORBIFOLD_MP_NERVE_VALIDATION_H

#include <cstdint>

// ===========================================================================
// (hecke-extensions slot 22, MP-1b) OrbifoldMPNerveValidation (standard).
//
// The twisted normalised cochain complex of the choice-free MP set S(X; T)
// ([MP] Def. 1.5, Thm 2.1) on M7 and R_Z, with coefficients Sym^n over F_ell,
// Z/ell^k and (through C_I and the UCT) Z.  Gate values:
// docs/implementation/slot22/MP_predictions.md P2 (countersigned before the
// first run), primary oracle the frozen mv_check.out as MvCheckOracle.inc (R5):
//   * P2.1 nerve H^0, H^1 over F_ell, ell in {2,3,5,7}, n = 0..12, both
//     triangulations (104 cells);
//   * P2.2 nerve H^2, ell in {2,3,5}, n <= 3 by default (24 cells); n in
//     {4, 6} behind ORBIFOLD_MP_CELLS (read-only, EFFECTIVE line; R3);
//   * P2.3 / P2.4 lengths over Z/ell^k, ell in {2,3}, k <= 3, n in {0,1,2,4},
//     q <= 1, and n = 0, q = 2;
//   * P2.5 the integral groups H^0..H^3 on C_I (IntegralGroups, certified at
//     2 and 3) and the nerve's Z/ell^k lengths equal to their UCT prediction;
//   * P2.6 d o d = 0 (formally in Z[U] (x) Z[S], every built degree incl. the
//     k = 4 strings, and per cell on the matrices), the untwisted-d_0 negative
//     control, odd n with 2 a unit: a non-zero acyclic complex;
//   * (MP-2, F-MP1 F1 / F2(a)) NC the twist rho_n(gamma_1^{-1}) breaks d o d = 0
//     (F_2, n = 1); NC a planted Gamma_{rho^2} := <(ST)^2> on M7 moves H^2(F_2,
//     n = 0) off 1 -- the default H^2 cells are load-bearing for the central
//     gerbe ((AC) is guarded by this cohomology gate, not by a predicate);
//   * the interned S_k = MPSet's S_k string for string (k <= 3), |S_4|
//     recorded (OUTPUT), AmalgamOracle == MvCheckOracle.inc,
//     MvCheckOracle.inc == the tracked mv_check.out.
// Lines: ORACLE mv_check.out cells=<k> agree=<k>; AMALGAM_ORACLE ...;
// ORBIFOLD_MP OrbifoldMPNerveValidation checks=<k> failing=0.  model=mp.
// No golden object, no setting, digest-neutral.
// ===========================================================================
uint32_t OrbifoldMPNerveValidation();

#endif
