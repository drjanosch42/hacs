// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LATTICE_MODULE_VALIDATION_H
#define HECKE_LATTICE_MODULE_VALIDATION_H
// (hecke-extensions slot 19, WP-LC) HeckeLatticeModuleValidation (standard): [E] Check chk:latcmp at (258, 37) on the
// Z/ell^K route (HeckeExt/HeckeLatticeModule), M = 1 .. 4, with the ell = 7 Loewy-2 / Loewy-3 controls and negative
// controls.
//   (P)  planted, from R19-1's P, P_1, P_2, Q_37 mod 37^6 (in source): the checker accepts T_m (+) m with exactly
//        chk:latcmp's (b)-(d) invariants, and rejects the two excluded candidates T_m (+) T~ (U_37 Smith) and T_m^2 (mu).
//   (L)  (258, 37), m = (37, T_2 - 29): (a) H_M free of rank 8 (and each iota-sign free of rank 4); (b) mu = 3;
//        (c) the Smith valuations of t = T_2 - 29 and U_37; (d) per sign, the free sign REPORTED; (e) P(t) = 0 and
//        T_p = Q_p(t) for every prime p < 100 with R19-1's tracked file docs/implementation/slot19/r191/data/
//        universal_eigenform_n1000_mod37e6.txt (read through the source root).
//   (C)  ell = 7 controls (docs/implementation/slot19/wplc/predictions.md, committed before the first run):
//        (48, 7) (1,0,0,0) Loewy 2; (90, 7) (3,3,1,0) Loewy 2 and (1,0,0,0) Loewy 3; hypothesis "free" per sign.
//   (N)  negative controls on the real block: a perturbed Q_p is rejected at every M; the swapped per-sign
//        assignment is rejected; a system that is not an eigenvalue is refused (std::logic_error).
// Read-only, no env; prints one LATCMP line per (cell, block, M).
#include <cstdint>

uint32_t HeckeLatticeModuleValidation();

#endif // HECKE_LATTICE_MODULE_VALIDATION_H
