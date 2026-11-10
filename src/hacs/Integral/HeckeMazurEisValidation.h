// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_MAZUR_EIS_VALIDATION_H
#define HECKE_MAZUR_EIS_VALIDATION_H

#include <cstdint>

// slot-25 H3 (docs/implementation/slot24/heckez/06-work-packages.md "H3"; 05-hida-targets.md Sec. 3). `standard` tier.
// The library is Integral/HeckeMazurEis.{h,cpp}. Mazur's Eisenstein ideal at prime level N, weight 2, integrally: per cell (N, ell)
// the Hensel factor F_q of c_q = sqrt(chi_Z(T_q) / (x - 1 - q)) at 1 + q for a recorded good q, and the checks (a) deg F = e of
// [C, LN6]'s dim W = 2e + 1 and Merel [C, LN5], (b) F(y + 1 + q) Eisenstein when ell || nu, (c) v_ell F(1 + q) = v_ell(nu), (d) the
// structure at ell^2 | nu (Newton polygon, embedding dimension, irreducibility) reported.
// Default arm: 10 cells incl. (11, 5), the e = 3 cell (181, 5), the ell^2 | nu cell (101, 5); plus the A7-M1 negative control
// (61, 5) at the forced good q = 37 (rejected by the valuation check).
// Read-only env arms (never write, change no value): HECKE_MAZUR_EIS_ALL=1 (the whole H3 list, MazurEisCellList()),
// HECKE_MAZUR_EIS_CELLS="N:ell,..." (those cells); each prints `EFFECTIVE HeckeMazurEisValidation arm=<X> cells=<k> refused=<r>`.
// Gate line: HECKE_MAZUR_EIS cells=... a_ok=... b_ok=... c_ok=... good_q_ok=... verdict=OK
// Returns the number of failing checks (0 on success).
uint32_t HeckeMazurEisValidation();

#endif
