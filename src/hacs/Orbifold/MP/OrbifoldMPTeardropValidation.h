// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_TEARDROP_VALIDATION_H
#define ORBIFOLD_MP_TEARDROP_VALIDATION_H

#include <cstdint>

// ===========================================================================
// (hecke-extensions slot 22, MP-T) OrbifoldMPTeardropValidation (fast).
//
// Moerdijk--Pronk's teardrop Tear-n ([MP] Sec. 3; MP_math_plan.md v0.2 Sec. 6) in the
// G^k-form of S(T) (a complex of groups, NOT a global quotient): the C++ port of
// docs/implementation/slot22/mp/teardrop.py.  Gates exactly MP_predictions.md P3:
// 9 cells (n in {2,3,4}) x (ell in {2,3,5}), dim H^0..H^3 of the normalised cochains with
// the trivial C_n-module F_ell (= A, 0, A, H^3(C_n;A)), and |S_0..S_4| (non-degenerate)
// = 6,39,177,705,2877 / 6,48,294,1866,13290 / 6,57,465,4323,44925 (81 values).
// d o d = 0 is checked EXHAUSTIVELY on every non-degenerate simplex of S_2, S_3, S_4
// (teardrop.py samples 200).  Negative controls: (NC1) mu_ac = mu_ca = rho (not inverse
// to each other) and (NC2) the mu-maps computed from a cone-point isotropy of order n+1
// while the labels live in C_n both break d o d = 0 (d o d = 0 iff mu_ac + mu_ca = 0 mod n);
// (NC3) trivial isotropy (n = 1) has H^3 = 0 at ell = 2, so the gate discriminates C_2 from 1.
// (NC1b) the SWAPPED pair rho <-> rho^{n-1} is an isomorphic twin (g -> g^{-1}): d o d = 0
// and the same dimensions; MP_predictions.md P3's "swapped mu breaks d o d = 0" is
// corrected in MPT_report.md.
// Lines: ORACLE teardrop.py cells=9 agree=<k>;
//        ORBIFOLD_MP OrbifoldMPTeardropValidation checks=<k> failing=0.  model=mp.
// No golden object, no setting, digest-neutral.
// ===========================================================================
uint32_t OrbifoldMPTeardropValidation();

#endif
