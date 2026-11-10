// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
#pragma once

// KernelAvx2BitIdentityValidation (slot-24 AVX2K, fast): the AVX2 u32 GEMM microkernel of
// gemm<uint32_t> -- its u64 microkernel and its vpmaddwd pair route for p <= 4096 (ENGINE_API.md
// Sec. 6.3) -- against the portable and the AVX-512 microkernels and
// the RefLinAlg oracle, entry for entry over the whole C buffer (padding and the frame outside
// the window included), on adversarial inputs:
//   (P) moduli 2, 3, 131, 211, 271, 349, 691, 997, 2048, 2053, 4093, 4096 (the pair route's
//       edge), 4099, 8191, 8209, 16381, 16411, 65521, 134217757 (the
//       project prime), 2^30.5 neighbours 1518500213 / 1518500249 (the U32_Delayed / U32_Shoup
//       switch), 2147483647, and > 2^31 (2147483659, 4294967291: k_max <= 4);
//   (L) every lane remainder: n = 1..33 (the 8-column sub-tile and the 16-column tile tails),
//       m = 1..17 (the MR = 8 / 4 / 2 / 1 sweep and the 4-row sub-groups);
//   (K) k around the 256 k-block and the k_max bound (1, 2, 255, 256, 257, 513, 1029);
//   (W) the worst case: every entry p - 1;
//   (Y) layouts: B column-major (packed), C column-major (the strided epilogue), windows of
//       larger buffers, Set / Add / Sub;
//   (A) Auto with ENGINE_KERNEL unset picks AVX-512 on an AVX-512 CPU, else AVX2, else portable;
//       a forced Avx2 path never takes the VNNI routes (vnni_route_taken == false);
//   (S) the pair route is selected for p <= 4096 on the AVX2 path and its counter moves;
//   (R) an operand entry >= p behind a Canonical tag is refused on the pair route.
// Arms: portable, avx2-u64 (pair route off), avx2-pair (forced), avx2-auto, avx512 (VNNI off).
// Arms without the ISA print a SKIP note (the portable arm and the oracle always run).
// Prints `AVX2K_GEMM avx2=<0|1> avx512=<0|1> checks=<n> failing=<f> verdict=<OK|FAIL>`.
#include <cstdint>
uint32_t KernelAvx2BitIdentityValidation();
