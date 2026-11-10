// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// VnniGemmBitIdentityValidation (slot-23 PERF-P5, tier standard; ENGINE_API.md Sec. 6.1).
//
// The small-modulus VNNI route of engine::kernels::gemm<uint32_t> (p <= 128) against the
// u32 AVX-512 microkernel, the portable microkernel and the RefLinAlg oracle, entrywise:
//   (G) gemm over moduli {2, 3, 5, 7, 13, 37, 101, 103, 127} and the composites
//       {4, 9, 121, 125, 128}, ragged shapes across the 8 x 32 tile and the 1024 k-block,
//       all layouts / transposed views / padded buffers, ops Set / Add / Sub, sub-windows;
//   (G2) (slot-24 G4, ENGINE_API.md Sec. 6.2) the two-limb route, 128 < p <= 2^14: primes
//       {131, 137, 503, 547, 691, 997, 1009, 8191, 16381}, composites {3^5, 2^8, 5^6, 127^2,
//       2^14}, across the 512 k-block and the 1024-column B block, every layout at 691 / 16381;
//   (N) n > VNNI_NC = 2048 (the second j0 block), also with m > 128 (audit R1);
//   (W) the worst case: every entry p - 1, k up to 3 k-blocks + 3 (both routes; plus the
//       maximal two-limb lane pattern at p = 16381);
//   (E) elimination: engine::elim::rref and FlatMatrix product / rank / RREF with the
//       route Off vs Auto vs Force (identical RREF, pivots, rank, product);
//   (R) the refusals: Force without VNNI throws; a "Canonical" operand entry >= 128 on a
//       routed call throws std::logic_error (two-limb: an entry >= 2^14); p > 2^14 never routes;
//   (S) the route is taken (routed-call counter moves) when VNNI is available.
// Without avx512_vnni (e.g. an AVX2-only CPU) the arms that need the route print a SKIP note and the
// fallback (Auto == the u32 route) is compared against the oracle instead.
// Prints one line `VNNI_GEMM vnni=<available|unavailable> checks=<n> failing=<f> routed=<r> routed2=<r2> verdict=<OK|FAIL>`.
#include <cstdint>
uint32_t VnniGemmBitIdentityValidation();
