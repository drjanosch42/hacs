// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

#include <cstdint>

// (hecke-extensions slot 23, F3L) OrbifoldF3lValidation (standard): the F_3 field half of an ell = 3 orbifold
// cell on the F3L route (integral_model::F3lRouteEnabled(); L23_F3L=0 is the rollback) against the old route.
//   (P) integral_model::rho_n_recurrence against rho_n_cubic entrywise: fields F_2 .. F_691, F_134217757, a u64
//       prime, the chain rings Z/9, Z/3^19, Z/16, Z/2^31, Z/125; random g, the coset deltas diag(1, p) / diag(p, 1),
//       a = 0 (descending), c = 0, a non-unit a with c a unit, a = c = 0 (refused: the cubic fallback); every
//       d_V in 1..40 and the boundaries 3^k - 1, 3^k, 3^k + 1, 2^k, up to 402; the public rho_n on = off.
//   (T) StrictLift::theta_hat (every T_p of the census, every degree) on = off over F_3, F_2, Z/9.
//   (R) whole ell = 3 records (and some ell = 2 records: rho_n is shared) route on vs off, bodies byte-identical
//       after GL-1's H-1 mask, logs identical after masking decimal fractions; even, odd, small and boundary n.
//   (C) counters (non-vacuity: ternary eliminations and rho_n recurrences happen on, never off), the switch and its
//       EFFECTIVE l23_f3l line.
uint32_t OrbifoldF3lValidation();
