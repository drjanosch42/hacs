// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef DESCENT_SMITH_GPU_VALIDATION_H
#define DESCENT_SMITH_GPU_VALIDATION_H

// (slot-24 PAR-GPU) DescentSmithGpuValidation (standard): the device route of the 2-adic Smith LU (ring2k::echelon_blocked
// on the u32 / u64 rungs, DescentSmithGpu.h) against the host code, entry for entry.  Prints
// "[SKIP] DescentSmithGpuValidation: no usable CUDA device (<reason>)" and passes when there is no device (or the build is
// CPU-only, OC_ENABLE_CUDA=OFF).  Arms (the device forced by a scoped override, the size floor off):
//   (K) the device GEMM C -= L U against the host kernel on edge shapes (1 .. 130 rows / columns, K = 0 .. 130), both W;
//   (E) echelon_blocked: the whole matrix + EchelonInfo, both pivot rules, k < W and k = W, on adversarial random inputs
//       (uniform, rank-deficient products, 2-power scaled rows, sparse / all-even columns, many row swaps, zero, one row / one
//       column), panel widths 64 / 16 / 128 / 1, at 1 and 16 host threads, resident and tiled routes (window capped at 128 / 16
//       columns; L store at the default budget and capped to one panel per phase);
//   (S) smith (sigma, level_ranks) and howell (H, pivots) through the device LU;
//   (D) real descent inputs: RunTreeLocalPieceN at ell = 2 on level-N cells and AssembleDescentN(11, 6) (tree and dense);
//   (V) the ENGINE_VERIFY_DEVICE twin (scoped): every device LU re-run on the host, counted;
//   (F) a simulated CudaError at session open: the host LU runs, same values, one fallback counted;
//   (P) the policy: under a thread-local cpu device and no override, no device LU runs; non-vacuity counters (device LUs,
//       tiled LUs, panel replays, lookahead panels).
// Read-only measurement hook: HECKE_DESCENT_SMITH_GPU_CELLS=<N>:<n>[,...] runs RunTreeLocalPieceN at ell = 2 on the host and
// on the device (ENGINE_THREADS host threads both), compares, prints PARGPU_CELL lines (t_piece, t_smith); with
// HECKE_DESCENT_SMITH_GPU_ARMS=gpu only the device run.  Prints one EFFECTIVE line when set.

#include <cstdint>

uint32_t DescentSmithGpuValidation();

#endif
