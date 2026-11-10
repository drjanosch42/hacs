// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (slot-24 PAR-GPU) The CPU-only twin of ring2k_device.cu (OC_ENABLE_CUDA=OFF; the CudaStub.cpp precedent): every entry
// point of CudaRing2k.h throws CudaError.  No caller gets here: the ring2k device route asks engine::cuda::available()
// (false in this build) before it plans a session, so the host LU always runs.

#include "CudaRing2k.h"

#include <string>

namespace engine {
namespace cuda {
namespace ring2k {

namespace {
[[noreturn]] void no_device(const char* what) {
    throw CudaError(std::string("cuda::ring2k::") + what + ": built without CUDA (OC_ENABLE_CUDA=OFF)");
}
} // namespace

MemInfo mem_info() { no_device("mem_info"); }
size_t session_bytes(uint32_t elem, uint32_t m, uint32_t cw, uint32_t panel) {
    return static_cast<size_t>(m) * cw * elem + static_cast<size_t>(m) * panel * elem + sizeof(uint32_t) * 2 * (panel + 1);
}
size_t staging_bytes(uint32_t elem, uint32_t m, uint32_t panel) {
    return 3 * static_cast<size_t>(m) * panel * elem + sizeof(uint32_t) * 2 * (panel + 1);
}
void* open(uint32_t, uint32_t, uint32_t, uint32_t) { no_device("open"); }
void put(void*, const void*, size_t, uint32_t, uint32_t, uint32_t, uint32_t) { no_device("put"); }
void get(void*, void*, size_t, uint32_t, uint32_t, uint32_t, uint32_t) { no_device("get"); }
void step(void*, const engine::ring2k::device::StepArgs&) { no_device("step"); }
void fetch(void*, void*, size_t) { no_device("fetch"); }
void close(void*) noexcept {}
void gemm_sub_host(uint32_t, void*, size_t, const void*, size_t, const void*, size_t, uint32_t, uint32_t, uint32_t) {
    no_device("gemm_sub_host");
}

} // namespace ring2k
} // namespace cuda
} // namespace engine
