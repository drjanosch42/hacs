// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZTorValidation (standard) -- slot-25 HZ-TOR: Tor terms, separation, splitting, the glue (Integral/HeckeZTor.h).
// Arms (the card "HZ-TOR" of docs/implementation/slot24/heckez/06-work-packages.md; 03-bockstein-tor.md Sec. 6):
//  (A) (1,18), (1,20) at 5: split by separation (T_5), J_5 zero; the control without T_5 (a common T'-point, Hom = 5)
//  (B) (1,26), (1,30) at 5: split for T', J_5 cap T' = 0, `bounds` with |J_5| <= 25
//  (C) K5: B2(1) on the 54 cells of A3 (level 1, even n in [8, 60], ell = 2, 3) against the Lean-kernel tables
//  (D) K6 at (1,8), ell = 2: chi(T_2 | L) = x + 1, chi(T_2 | tors) = x^t, a checked Bezout pair
//  (E) B8 / Cor. B8' on the eight level-1 cells of 03 Sec. 6.2 at 2 and 3
//  (F) T4 (2,6) at 5 (and at 2, 3), T5 (8,10) at 3 (and at 2) on the export
//  (G) toy extensions: the non-split counterexample of B12(3), a mixed-exponent non-split, a separated one; tamper controls
//  (H) the EXPORT-Z interface: SplitFromModuleZ == the export read, ModuleZSplitWord
//  (I) schedule independence: the cell list on 1 and 4 threads, digest equal
//  (J) presentation invariance: three random changes of the lift and of the basis of L at five cells, the verdicts equal
//  (O5) the T4 (2,6) 2-adic at-root read at K = 20 by T_3, T_5, T_7 (a side read for the referee, Z0 referee O5)
// Gate: HECKEZ_TOR cells=<k> mismatched=0 exact_J=0 verdict=OK

#pragma once
#ifndef INTEGRAL_HECKEZ_TOR_VALIDATION_H
#define INTEGRAL_HECKEZ_TOR_VALIDATION_H

#include <cstdint>

uint32_t HeckeZTorValidation();

#endif
