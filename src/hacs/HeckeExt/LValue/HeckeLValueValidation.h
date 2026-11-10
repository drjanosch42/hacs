// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LVALUE_VALIDATION_H
#define HECKE_LVALUE_VALIDATION_H
// (hecke-extensions slot 19, WP-LV) HeckeLValueValidation (standard): the L-value interface of [E] sec:lvalue and its
// Eisenstein / Bernoulli instance on Remark rmk:lvgates's untwisted cells, with negative controls.
//   (B)  Bernoulli numbers: the exact route (tangent numbers) against five printed B_k, a wrong fraction refused, the
//        exact route against the Voronoi route mod ell^m on every gate cell, and the depth-2 rungs of [U] / F1
//        (284 at 37, 228 at 103 exact and Voronoi; 342252 and 342440 at 691: B_k/k == 0 mod 691^2, v = 1 at k -+ 690).
//   (E)  cl:lveis(3) on H^-_{m_Eis} at (n, ell) = (30, 37), (10, 691), (42, 59), (56, 67) (v = 1) and the regular
//        controls (10, 37), (22, 37) (v = 0), m = 1, 2, 3: the saturation verdict (rank 1 / 0 blocks), Q cyclic of
//        length min(m, v(B_k/k)), and the inequality min_{p != ell} v(c_{k,p}(1 + p^{k-1})) >= min(m, v).
//   (N)  negative controls: the plus side (where e occurs: T/I_e = Z/ell^m, never min(m, v) for m > v); a wrong-weight
//        character 1 + p^{k+1} (Q = 0, the index statement fails); a planted non-saturated algebra Z[ell e]; the
//        Birch and Hida stubs (NotImplementedInThisBuild) and the kind refusals; an unsupported domain refused.
//   (H)  (slot-19 WP-BIL) the bilinear kind, Hida's GL2 x GL2 pairing (HidaPairingValidationArms.cpp), gates G1-G5 of
//        R19-9 Sec. 6: the C++ PoincareDualityPairing's G^1 == -<Psi, y(v4v6)> at R19-9's 9 cells; A_D(T_p) = A_c(T_p)^T, p = 2, 3, 5,
//        iota anti-isometric; G_! = -B/6 with radical Z p_0; R19-9's frozen valuations at 27 cells (k = 12..28) and at
//        (258, 37) (mixed 3 / 2, P9 3 / 3, sum 6); every value == the congruence module of its lattice (the lemma), the
//        Eisenstein values == v(B_k/k); negative controls (perturbed cocycle, wrong lattice, planted non-adjoint, refusals).
//        Env arm HECKE_LVALUE_BILINEAR_HEAVY=1: G1 at (258, 37).
// Read-only, no file; prints one LVALUE line per (cell, m).
#include <cstdint>

uint32_t HeckeLValueValidation();

#endif // HECKE_LVALUE_VALIDATION_H
