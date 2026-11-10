// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_N_CURVE_VALIDATION_H
#define HECKE_LEVEL_N_CURVE_VALIDATION_H
// (hecke-extensions slot 22, LN23) HeckeLevelNCurveValidation (standard): the weight-2 level-N Eisenstein route at
// p = 2, 3 on the curve object (docs/implementation/slot22/LN23_spec.md v0.2, section 10, and v0.3, section 11 (LN23-F);
// reports LN23_report.md, LN23F_report.md).  At n = 0 the presentation M / (M^S + M^{ST}) of M = F_p[Gamma_0(N) \ SL_2(Z)],
// the fixed spaces built as orbit sums, is Manin's presentation of H_1(X_0(N), cusps; Z) (x) F_p = H^1(Y_0(N); F_p), the
// cohomology of the open coarse curve, in every characteristic ([C] prop:levelNcurve).  Every reading is that of its
// family F of Hecke primes: W_F contains W_m, and on W_m its radical series is at most e + 1 long ([C] rmk:levelNcurvefamily, the
// F2 caveat of (LN4) / (LN6)); the CE arms therefore compare the reading of a CERTIFIED family (LN23-F): the family q <= QMAX
// widened along 13, 19, 31, 47, 71, 101, ... until it contains a Mazur good prime (generation, Mazur II (16.6) / (17.1)),
// its dim W and layers are unchanged under the next two widenings and the layers read d, d - 2, ..., 1 (separation, in
// practice; the signature clause assumes (LN6)'s faithfulness), or dim W_F = 1.  Arms:
//   (P)  the n = 0 generators are the orbit sums (every value 1) at every p; at p >= 5 BuildCurve = Build entry for
//        entry (value preservation of the orbit-sum path);
//   (Z1) the torsion gate: dim_{F_p} = 2g + c - 1 and h0 = 1 at p in {2, 3, 5, 7, 134217757}, every prime N <= 600
//        and every composite N <= 100 (HECKE_LN23_Z1_NMAX widens the primes);
//   (H)  H_{N,q} (q <= 13) maps Rel into Rel at p = 2, 3 (RelEscapesOnGenerators == 0);
//   (X)  dense = sparse Hecke matrices on the curve route at p = 2, 3; kCurve = kGroup at p >= 5;
//   (Z3) Calegari--Emerton's section-6 oracles and the prototype's cells (dim W, Loewy length), on the certified family;
//   (Z4) CE Thm 1.1 on every prime N = 9 mod 16, N <= 2000 (HECKE_LN23_THM11_NMAX), 2^m || h(-4N) by a reduced-form count,
//        on the certified family;
//   (Z5) CE Thm 1.2(i) + 1.3(i) on every prime N = 10, 19 mod 27, N <= 2500 (HECKE_LN23_THM12I_NMAX), on the certified family;
//   (FC) every reading of (Z3) / (Z4) / (Z5) / SMALLP reaches its certificate by q <= 997; (FC0) the incremental family
//        reader equals ReadEisensteinR3Scan value for value, and MazurGoodPrime on hand values;
//   (G)  generation: at (41,2), (2137,2), (2143,3), (1747,3) every single good prime q alone reads dim W_{q} and a Loewy
//        length at least the certified ones (eta_q generates m T_m);
//   (Z6) negative controls: n = 2 at p = 3 refused by name on the curve route, the averaging path at p = 3 refused,
//        the default (group) entry points at p in {2, 3} refused with their slot-15 texts, dim W = 1 off the cells.
// Read-only hooks (each prints an EFFECTIVE line): HECKE_LEVEL_N_R3_SMALLP_CELLS="N:p,..." (replaces every arm by
// readings of the listed cells; N prime, p in {2, 3}, p | num((N - 1)/12)), HECKE_LN23_Z1_NMAX, HECKE_LN23_THM11_NMAX,
// HECKE_LN23_THM12I_NMAX, HECKE_LN23_QMAX (the Hecke primes q <= QMAX of every reading, the first rung of the certificate;
// default 13: the family can fail in two ways at any N -- W too large (no prime in the family separates another maximal
// ideal) or the radical series too short (no Mazur good prime, Mazur II (16.6)/(17.1), in the family); LN23_report.md
// section 5 and the F4 report); HECKE_LN23_STURM=1 (arm=STURM, replaces every arm: the separating family q <= B =
// floor((N + 1)/6) of rmk:levelNcurvefamily (2) on the default grids plus (2137, 2), (2953, 2), or on the
// HECKE_LEVEL_N_R3_SMALLP_CELLS cells when both are set; prints the two caveats: T^0 only, Loewy needs (LN6)'s CONJ); the
// heavy interlock HECKE_LN23_HEAVY=1 adds (21929, 2) (CE e_2 = 127; predicted <= 15 min, <= 2 GiB).
#include <cstdint>

uint32_t HeckeLevelNCurveValidation();

#endif // HECKE_LEVEL_N_CURVE_VALIDATION_H
