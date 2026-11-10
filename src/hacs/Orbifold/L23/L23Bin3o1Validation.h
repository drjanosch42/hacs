// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// (hecke-extensions slot 23, BIN3O1) L23Bin3o1Validation: the BIN3O1 route of the degree-1 amalgam reader (AmalgamReader.h,
// docs/implementation/slot23/BIN3O1_report.md) against the old route, entrywise, route on vs off (l23::ScopedBin3o1):
//   (K) ring2k::kernel_shift_rows (the register-wavefront Taylor sweep) = the kernel_axpy sweep row j -= f row (j + 1), for
//       u8 / u16 / u32 / u64, f in {-1, +1, 3, random}, every kernel path, at the block boundaries (n around the 16-pass
//       block, widths around the vector lane count, strided rows);
//   (R) the relation matrix and B^1 (Pascal / U^3 = (-1)^n route against the five Taylor shifts) over Z/2^k on every rung kind
//       (R1 ... R5) and over Z/3^k, n at the block boundaries, both parities, and the parity structure of A;
//   (S) rho(T), rho(T)^-1, rho(U), words with runs (T^9, T^-5, ...), mat_add, mat_sub on random matrices over every rung kind;
//       the characteristic-2 Taylor shift against the u8 one mod 2 (and T^-1 T = id);
//   (H) h1_mod at ell = 2: k = 1 (the F_2 / BitMatrix route) and k in {2, 3, 8, 16, 40}, many n, operators T_p - c, with and
//       without the negative control mutate = 1: every field equal on / off;
//   (L) read_lattice at ell = 2 (r3-style K = 64 and adaptive full precision, n even / odd / powers of 2) and ell = 3, with
//       the negative controls mutate = 1, 2: every field of the cell (divisors, torsion, logs, failures) equal on / off;
//       h1_layers and oddn_row equal on / off;
//   (C) the route counters (non-vacuity: the even-n split accepted, the odd-n reduction, the F_p block ranks, the F_2 cells,
//       the reused T_p Z; the split fallback reported), the EFFECTIVE line, and two in-run negative controls (a flipped
//       entry is seen by the comparator; mutate = 1 changes an h1_mod image on the route).
// One BIN3O1 summary line; returns the number of failing checks.
#include <cstdint>

uint32_t L23Bin3o1Validation();
