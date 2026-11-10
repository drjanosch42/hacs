// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_REDUCTION_COVERAGE_SLOW_VALIDATION_H
#define HECKE_REDUCTION_COVERAGE_SLOW_VALIDATION_H

// ===========================================================================
// (hecke-extensions slot 8, WP-4; E8 item 4a) The reduction / Bockstein arms.
// Binding: slot-8 plan WP-4 scope 1-4, [E] v1.9.0-pre.1 sec:ring (d)-(e),
// [X] Sec. 6.2 E8 item 4 (bulk half) with F3, [C] cl:bockstein.
//
// Per cell (n, ell) at levels K in {2, ...}, for every Hecke prime p of the
// cell (T_p, and U_ell at p = ell), with Hecke/HeckeReduction.h:
//   (R1) red_H^q is the class map: (b_j mod ell) - sum_i red(i,j) h_i lies in
//        B^q(F_ell), for every cyclic-summand generator b_j (q = 0, 1, 2);
//   (R2) A(T_p) o red = red o A(T_p), exactly over F_ell ([X] F3's map
//        equality; never "the matrices reduce to each other");
//   (R3) on cochains the reduction is entrywise: Theta^{(q)} over Z/ell^K
//        reduces mod ell to the F_ell Theta_flat^{(q)} (and mod ell^{K-1} to
//        the Z/ell^{K-1} Theta);
//   (R4) at K = 2, exactness at H^q(F_ell): beta^q o red_H^q = 0 and
//        rank red_H^q + rank beta^q = dim H^q(F_ell); the cokernel length
//        dim H^q(F_ell) - rank red_H^q is printed at every level (the Tor
//        cokernel, not an error);
//   (R5) between consecutive levels K+1 -> K: A o red = red o A (row i
//        compared in [0, ell^{e_i})), functoriality red_K o red_{K+1->K} =
//        red_{K+1}, and Theta entrywise;
//   (B1) beta o A(T_p) = A(T_p) o beta for beta^0, beta^1;
//   (B2) the coefficient long exact sequence of 0 -> Z/ell -> Z/ell^2 ->
//        Z/ell -> 0 with its length bookkeeping: length H^q(Z/ell^2) =
//        2 dim H^q(F_ell) - rank beta^{q-1} - rank beta^q, q = 0, 1, 2;
//   (B3) the bookkeeping identity bockstein_rank^0 + higher_torsion_count ==
//        dim H^0(F_ell) (audit B-2), the pre-stated higher_torsion_count, and
//        wherever the K-mode certifies higher_torsion_count == #{e_i >= 2}
//        and bockstein_rank^0 == #{e_i = 1} (the MEANING, [L] Sec. 2.4).
//
// HeckeReductionBulkArms (called from HeckeReductionValidation, `standard`):
// the five pre-stated cells (10, 691) at K = 2; (6, 5), (16, 5), (30, 5),
// (60, 5) at K = 2, 3, p in {2, 3, 5} (U_5 at ell = 5 validated, not frozen);
// P3's positive controls (30, 5) -> Z/25, (60, 5) -> Z/25 (+) Z/5, each with
// higher_torsion_count = 1; and the D6 arm at (10, 691): over Z/691^2
// charpoly_A(T_2)^1 = (x - 2049)(x + 24)^2 with a_E == a_Delta (mod 691),
// a_E != a_Delta (mod 691^2), and the eigenlines recorded.
//
// HeckeReductionCoverageSlowValidation (`slow`): (R1)-(R4), (B1)-(B3) at
// K = 2 on the E2 case list at ell >= 5 (the 31 ext-subgrid cells of [X]
// Sec. 1.6 and the 23 M1 cells; 52 distinct), p in {2, 3, 5}; (160, 691) and
// (198, 691) only under the read-only interlock HECKE_REDUCTION_COVERAGE_LARGE=1
// (the HECKE_DUALITY_COVERAGE_LARGE precedent; not an emitter).  Writes nothing.
// ===========================================================================

#include <cstdint>

void HeckeReductionBulkArms(uint64_t& checks, uint32_t& failures);
uint32_t HeckeReductionCoverageSlowValidation();

#endif // HECKE_REDUCTION_COVERAGE_SLOW_VALIDATION_H
