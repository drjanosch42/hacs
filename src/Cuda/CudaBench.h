// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// CudaBench.h -- the `--bench micro --backend gpu` arm (Phase 5)
// =====================================================================
//
// InstallMicroHooks() registers the GPU implementations of the micro ops
// (gemm / gemv / transpose, same fixed-seed inputs and checksum family as
// the engine arm, so equal checksums are a bit-identity statement) and the
// Phase-5 throughput suite (`--bench micro cuda [--repeat k]`) with
// micro::SetGpuHooks. A no-op when engine::cuda::available() is false: the
// arm then reports "not available", exit code 2, exactly like the
// pre-Stage-B engine arm did.
//
// The suite (micro::GpuHooks::suite) prints: dense u32 GEMM at n in {512, 1024, 2048, 4096}
// (ell = 134217757) with the roofline fraction (4931 G modMAC/s, 04 Sec. 1.5),
// u64 GEMM at 2^61-1 (roofline 629 G), the batched small-block program
// 2000 x (129 x 129 x 129) -- CPU single thread vs GPU unbatched vs one
// batched launch vs graph replay --, H2D / D2H bandwidth pinned vs pageable,
// and the per-launch cost stream vs graph (04 Sec. 1.4: 6.11 / 0.68 us).

#include <cstdint>

namespace engine {
namespace cuda {

void InstallMicroHooks();

} // namespace cuda
} // namespace engine
