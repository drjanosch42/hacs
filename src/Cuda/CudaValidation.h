// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// CudaValidation.h -- the Phase 5 device-layer gates
// =====================================================================
//
// CudaBitIdentityValidation (tier standard). Every device kernel of
// CudaKernels.h against its CPU twin in engine::kernels on fixed-seed
// inputs: sizes {1, 2, 7, 16, 17, 64, 65, 129, 256, 517, 1024, 2048}
// (rotated over m / n / k), all layouts and logical transpositions, both
// element widths, primes {2, 3, 134217757, 2^31-1, 2^32-5, 2^61-1} plus the
// U64_Wide fallback at 2^64-59 and u64 p = 2; k > k_max (k-blocking),
// split-K with canonicalised partials, every tile configuration, batched
// vs unbatched, graph-replayed vs direct. The comparison is memcmp of the
// WHOLE footprint after D2H (padding included -- the device never writes
// it), no tolerances.
//
// CudaTransferValidation (tier standard). DeviceBuf residency / versioning
// round trips, staged transfers larger than the pinned pool (peak pinned
// bytes <= cap, zero pageable copies), arena reuse, capture rules.
//
// Both print "[SKIP] ..." and return 0 when engine::cuda::available() is
// false, so --all stays green on a machine without a GPU.

#include <cstdint>

uint32_t CudaBitIdentityValidation();
uint32_t CudaTransferValidation();
