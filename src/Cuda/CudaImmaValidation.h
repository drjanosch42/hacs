// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// CudaImmaValidation.h -- the slot-22 G1 gate of the IMMA GEMM route
// =====================================================================
//
// CudaImmaBitIdentityValidation (tier standard). The int8 tensor-core GEMM
// (ImmaKernels.cuh, ENGINE_API.md Sec. 11.8) against engine::kernels::gemm by
// memcmp of the whole footprint, at p in {2, 3, 5, 13, 127} and the
// composite moduli {4, 9, 125}:
//   (A) random operands, every layout / logical transposition / op, tails in
//       M, N and K (K not = 0 mod 16 / 64), gemm_nt and schur_update;
//   (B) structured operands: zero, zero rows / columns at tile boundaries,
//       duplicated columns, extreme residues (every product +-floor(p/2)^2);
//   (C) K-chunking: forced flush intervals 1, 2, 3 around their boundaries,
//       workspaces for K chunks of 64 / 128 / 320 (op on the first chunk,
//       Add / Sub after), the true int32 bound at p = 127 / 125
//       (K = 64 T(p) +- ...), and the refusal of a flush interval above T(p);
//   (D) routing: non-qualifying moduli (128, 131, 65521, 134217757, any
//       uint64_t context), imma = 0, a missing workspace or one below a
//       64-deep chunk and k = 0 take the u32 kernel (zero IMMA launches, same
//       bytes); imma = -1 follows ENGINE_GPU_IMMA (its "0" cases are legacy arms since
//       S24 CL-R2: OC_ENABLE_LEGACY=ON only);
//   (E) GemmBatch with prepare_imma (Set / Add / Sub, shared operands);
//   (F) the tape under ENGINE_DEVICE=gpu (verify on and off, graph replay):
//       hashes equal the cpu policy's, IMMA launches > 0, a block growth drops
//       and re-captures a cached program's graph, and the rollback
//       ENGINE_GPU_IMMA=0 gives the same hashes with zero IMMA launches (a legacy
//       arm since S24 CL-R2: OC_ENABLE_LEGACY=ON only).
// Every arm asserts the IMMA launch count, so a vacuous pass is a failure.
// Prints "[SKIP] ..." and returns 0 without a usable device (and with
// OC_ENABLE_CUDA=OFF), or when the device / build has no sm_80+ IMMA image.

#include <cstdint>

uint32_t CudaImmaBitIdentityValidation();
