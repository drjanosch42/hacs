// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// (slot-24 PAR-GPU) The device side of the ring2k device-resident LU (src/ModArithmetic/Binary/Ring2kDevice.h): a session
// holding the column window [0, cw) of an m x cw matrix over Z/2^32 or Z/2^64 (elem = 4 / 8 bytes), the L store and pinned
// staging; the kernels (ring2k_device.cu) are the row-swap replay, the unit-lower TRSM on the pivot rows and the trailing
// GEMM C -= L U, all in native wraparound integer arithmetic: exact, so bit-identical to the host kernels by construction.
//
// CUDA-header-free.  In a CPU-only build (OC_ENABLE_CUDA=OFF) CudaRing2kStub.cpp defines every entry point: mem_info()
// reports nothing and everything else throws CudaError (no caller gets past available()).
//
// Portability: plain SIMT integer code (no tensor cores, no warp-specialised or cluster features, 32 KiB static shared
// memory per block), so every architecture of OC_CUDA_ARCHITECTURES runs it: sm_75 (Turing), sm_80 (A100), sm_87 (Orin),
// sm_90 (H100), sm_120 (RTX 50).

#include <cstddef>
#include <cstdint>

#include "CudaCtx.h"
#include "../ModArithmetic/Binary/Ring2kDevice.h"

namespace engine {
namespace cuda {
namespace ring2k {

struct MemInfo { size_t free_bytes = 0, total_bytes = 0; };
MemInfo mem_info();                                   // cudaMemGetInfo on the engine's device (throws CudaError)

// Device bytes a session needs for an m x cw window of elem-byte entries with this panel width.
size_t session_bytes(uint32_t elem, uint32_t m, uint32_t cw, uint32_t panel);
// Pinned host bytes a session stages through (three m x panel slabs and the swap list).
size_t staging_bytes(uint32_t elem, uint32_t m, uint32_t panel);

void* open(uint32_t elem, uint32_t m, uint32_t cw, uint32_t panel);
void  put(void* s, const void* host, size_t ldh, uint32_t r0, uint32_t rows, uint32_t c0, uint32_t cols);
void  get(void* s, void* host, size_t ldh, uint32_t r0, uint32_t rows, uint32_t c0, uint32_t cols);
void  step(void* s, const engine::ring2k::device::StepArgs& a);
void  fetch(void* s, void* host, size_t ldh);
void  close(void* s) noexcept;

// Validation only: one trailing GEMM C (m x n) -= L (m x k) U (k x n) on the device, all host row-major, for the kernel's
// own entrywise check against the host kernel (any m, n, k; elem 4 or 8).
void gemm_sub_host(uint32_t elem, void* C, size_t ldc, const void* L, size_t ldl, const void* U, size_t ldu, uint32_t m,
                   uint32_t n, uint32_t k);

} // namespace ring2k
} // namespace cuda
} // namespace engine
