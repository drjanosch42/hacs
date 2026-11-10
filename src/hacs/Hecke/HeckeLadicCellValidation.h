// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LADIC_CELL_VALIDATION_H
#define HECKE_LADIC_CELL_VALIDATION_H

// (hecke-extensions slot 11, WP-18c) HeckeLadicCellValidation (`standard`):
// the digit pipeline route Hecke/LadicCell against the pipeline-level oracle
// matrix of the slot-11 plan's WP-18c section ([E] v1.12.0-pre.2 sec:ring):
//   (R)  the per-cell rule and the grammar: ClassifyCell, the refusals by name
//        (ell in {2, 3} beyond 2^64, the 8192-bit cap, the memory cap, odd n),
//        ENGINE_LADIC_WIDTH's parse, the coefficient_ring parse cap 4096;
//   (i)  2 x 32 vs the shipped u64 E8 route: the digit route forced below 2^64
//        (TEST-ONLY hook) equals the shipped EquivariantCohomologyOnK64 /
//        HeckeOperatorOnK64 entrywise -- C, X, Z, the cyclic decomposition, dK,
//        Theta, A^{(q)}, the K-mode type, the free charpoly -- at the seven E8
//        points and at the u64 limit;
//   (ii) M-0: free_charpoly_H1(T_2) at (10, 691) over Z/691^k, k in {7, 12, 20,
//        40}, equals (x - 2049)(x + 24)^2 mod 691^k (line 2's integer output),
//        and every O5 row of IntegralCharpolyCrt at 691, k = 7;
//   (iii) UCT against the K-mode: every run's precision.uct_match;
//   (iv) cross-width: the whole cell record at w in {32, 64, 128};
//   (v)  truncation: C, X, Theta of a digit run truncated to a native k' equal
//        the shipped route at k';
//   (vi) the cell's own X_q and Theta^{(1)} at 2^64 <= ell^K < 2^127 through
//        native u128 kernel_module / invariant_factors;
//   (P)  the precision bound: e_max at (10, 691), (20, 5), (30, 5), (60, 5); the
//        forced-uncertified word level refused as PrecisionInsufficient;
//   (H)  the ordinary idempotent at (10, 691), U_691, certified.
// Writes nothing.
#include <cstdint>

uint32_t HeckeLadicCellValidation();

#endif
