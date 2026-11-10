// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef DESCENT_SMITH_MT_VALIDATION_H
#define DESCENT_SMITH_MT_VALIDATION_H

// (slot-24 PAR-2K) DescentSmithMtValidation (standard): the threaded ring2k / ring3k Smith (src/ModArithmetic/Binary/SmithPar.h,
// lever HECKE_DESCENT_SMITH_MT) against the serial code, bit for bit, at ENGINE thread counts 1, 2, 7, 16 (scoped; the size
// threshold lowered to 0 so that every step splits):
//   (R2) ring2k on random inputs (full rank, rank-deficient products, 2-power scaled rows), rungs u8 / u16 / u32 / u64 / u128 /
//        WordN<4>: echelon_blocked's whole matrix + EchelonInfo (both pivot rules), smith (sigma, level_ranks), howell (H, pivots);
//        on every available kernel path (portable / AVX2 / AVX-512: AVX-512 forced off included), and smith = smith_naive;
//   (R3) ring3k (k = 19 and 7: the dense blocked variant) on random inputs: smith_levels, howell_blocked (H entrywise), with the
//        engine GEMM on its AVX-512 and portable paths, and smith_levels = smith_naive;
//   (D)  real descent inputs: RunTreeLocalPieceN at ell = 2, 3 on level-N cells (every differential's r_K, nonunit, rank, the
//        torsion) and AssembleDescent's dense-rollback smith route (HECKE_DESCENT_TREE off, scoped) at one cell;
//   (L)  the lever: off (scoped) = no parallel region and the same values; outside a ScopedEnable threads() = 1; the parallel arms
//        really ran parallel regions (non-vacuous); the EFFECTIVE line.

#include <cstdint>

uint32_t DescentSmithMtValidation();

#endif
