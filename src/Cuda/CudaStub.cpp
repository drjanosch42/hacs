// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// CudaStub.cpp -- the CPU-only twin of CudaCtx.cpp + cuda_kernels.cu
// (hecke-extensions slot 19, NB-0).
//
// Compiled INSTEAD of those two files when the tree is configured with
// -DOC_ENABLE_CUDA=OFF (no CUDA toolkit, no nvcc, no libcudart). It defines
// every symbol of the CUDA-header-free public headers CudaCtx.h and
// CudaKernels.h, so MatrixEngine's DeviceBackend.cpp, CudaValidation.cpp,
// CudaBench.cpp and every other consumer compile and link unchanged:
//
//   * engine::cuda::available() is false, with the reason
//     "built without CUDA (OC_ENABLE_CUDA=OFF)"; every device validation
//     already prints "[SKIP] <name>: no usable CUDA device (<reason>)" and
//     returns 0 on that branch, and DeviceBackend::usable() is false, so the
//     executor never constructs a device arm -- ENGINE_DEVICE=cpu, the golden
//     path, is the only path.
//   * requested_device() reads ENGINE_DEVICE exactly as CudaCtx.cpp does
//     (ENGINE_DEVICE=gpu is accepted and then falls back to the CPU, as on a
//     machine without a usable device).
//   * Everything that needs a device (CudaCtx::get(), the kernels, graph
//     capture) throws CudaError -- exactly what the CUDA build does when
//     available() is false. No caller reaches them past an available() test.
//
// No value can change: the device arm is never used for a golden value (the
// GPU is never enabled by default), and on the CPU path this file is dead code.
#include "CudaCtx.h"
#include "CudaKernels.h"
#include "RetiredDeviceLevers.h"   // (S24 CL-R2) ENGINE_GPU_IMMA

#include <cstdlib>
#include <cstring>
#include <string>

namespace engine {
namespace cuda {

namespace {

const char kReason[] = "built without CUDA (OC_ENABLE_CUDA=OFF)";

[[noreturn]] void no_device(const char* what) {
    throw CudaError(std::string(what) + ": " + kReason);
}

} // namespace

bool        available() { return false; }
const char* unavailable_reason() { return kReason; }

Device requested_device() {
    static const Device d = [] {
        const char* e = std::getenv("ENGINE_DEVICE");
        if (e && std::strcmp(e, "gpu") == 0) return Device::Gpu;
        if (e && std::strcmp(e, "hybrid") == 0) return Device::Hybrid;
        return Device::Cpu;
    }();
    return d;
}
bool device_enabled() { return false; }

DeviceInfo device_info() { no_device("device_info"); }
bool capturing_now() { return false; }

// ---- CudaCtx: get() throws, so no instance ever exists; the members are
// ---- defined only so that the (unreachable) call sites link.
CudaCtx& CudaCtx::get() { no_device("CudaCtx"); }
bool context_created() { return false; }
ReleaseStats release_device_caches() { return ReleaseStats{}; }
void set_trace_tag(const char*) {}
const char* trace_tag() { return ""; }

CudaCtx::CudaCtx() : impl_(nullptr) { no_device("CudaCtx"); }
CudaCtx::~CudaCtx() {}

int    CudaCtx::num_streams() const { no_device("CudaCtx::num_streams"); }
Stream CudaCtx::stream(int) const { no_device("CudaCtx::stream"); }
Stream CudaCtx::create_stream() { no_device("CudaCtx::create_stream"); }
void   CudaCtx::destroy_stream(Stream) { no_device("CudaCtx::destroy_stream"); }
void   CudaCtx::sync(Stream) const { no_device("CudaCtx::sync"); }
void   CudaCtx::sync_all() const { no_device("CudaCtx::sync_all"); }
Event  CudaCtx::create_event() const { no_device("CudaCtx::create_event"); }
Event  CudaCtx::create_timing_event() const { no_device("CudaCtx::create_timing_event"); }
void   CudaCtx::record(Event, Stream) const { no_device("CudaCtx::record"); }
void   CudaCtx::wait(Stream, Event) const { no_device("CudaCtx::wait"); }
void   CudaCtx::sync(Event) const { no_device("CudaCtx::sync"); }
void   CudaCtx::destroy(Event) const { no_device("CudaCtx::destroy"); }
float  CudaCtx::elapsed_ms(Event, Event) const { no_device("CudaCtx::elapsed_ms"); }
void*  CudaCtx::dalloc(size_t) { no_device("CudaCtx::dalloc"); }
void   CudaCtx::dfree(void*, size_t) { no_device("CudaCtx::dfree"); }
ReleaseStats CudaCtx::release_cached(bool) { no_device("CudaCtx::release_cached"); }
ArenaStats   CudaCtx::arena_stats() const { no_device("CudaCtx::arena_stats"); }
size_t       CudaCtx::arena_idle_slack() const { no_device("CudaCtx::arena_idle_slack"); }
void  CudaCtx::upload(const void*, void*, size_t, Stream) { no_device("CudaCtx::upload"); }
void  CudaCtx::download(void*, const void*, size_t, Stream) { no_device("CudaCtx::download"); }
void  CudaCtx::copy_d2d(void*, const void*, size_t, Stream) { no_device("CudaCtx::copy_d2d"); }
void  CudaCtx::memset_d(void*, int, size_t, Stream) { no_device("CudaCtx::memset_d"); }
void  CudaCtx::upload_pageable(const void*, void*, size_t) { no_device("CudaCtx::upload_pageable"); }
void  CudaCtx::download_pageable(void*, const void*, size_t) { no_device("CudaCtx::download_pageable"); }
void* CudaCtx::pinned_alloc(size_t) { no_device("CudaCtx::pinned_alloc"); }
void  CudaCtx::pinned_free(void*) { no_device("CudaCtx::pinned_free"); }
PinnedPoolStats CudaCtx::pinned_stats() const { no_device("CudaCtx::pinned_stats"); }
TransferStats   CudaCtx::transfer_stats() const { no_device("CudaCtx::transfer_stats"); }
void            CudaCtx::reset_transfer_stats() { no_device("CudaCtx::reset_transfer_stats"); }
size_t CudaCtx::pinned_cap_bytes() const { no_device("CudaCtx::pinned_cap_bytes"); }
size_t CudaCtx::pinned_chunk_bytes() const { no_device("CudaCtx::pinned_chunk_bytes"); }
int    CudaCtx::copy_threads() const { no_device("CudaCtx::copy_threads"); }

// ---- GraphCapture: constructible (an object may be declared before the
// ---- available() test), every action throws.
GraphCapture::GraphCapture() : impl_(nullptr) {}
GraphCapture::~GraphCapture() {}
void   GraphCapture::begin(Stream) { no_device("GraphCapture::begin"); }
void   GraphCapture::end() { no_device("GraphCapture::end"); }
void   GraphCapture::launch(Stream) const { no_device("GraphCapture::launch"); }
bool   GraphCapture::instantiated() const { return false; }
size_t GraphCapture::num_nodes() const { return 0; }
Stream GraphCapture::captured_stream() const { return Stream{}; }
bool   GraphCapture::capturing() const { return false; }

// ---- kernels (CudaKernels.h) ----
size_t gemm_workspace_bytes(uint32_t, uint32_t, uint32_t, size_t) { no_device("gemm_workspace_bytes"); }
uint32_t suggest_splits(uint32_t, uint32_t, uint32_t) { no_device("suggest_splits"); }
// (slot-22 G1) no IMMA without a device; the policy reads the switch exactly as cuda_kernels.cu does.
bool imma_supported() { return false; }
bool imma_policy() {
    return engine::lever::gpu_imma_lever().enabled();   // (S24 CL-R2) the one lever object (RetiredDeviceLevers.h)
}

template<class Elt>
void gemm(const ModCtx<Elt>&, DeviceView<const Elt>, DeviceView<const Elt>, DeviceView<Elt>, GemmOp, Stream,
          const GemmOpts&) { no_device("gemm"); }
template<class Elt>
void gemm_nt(const ModCtx<Elt>&, DeviceView<const Elt>, DeviceView<const Elt>, DeviceView<Elt>, GemmOp, Stream,
             const GemmOpts&) { no_device("gemm_nt"); }
template<class Elt>
void schur_update(const ModCtx<Elt>&, DeviceView<const Elt>, DeviceView<const Elt>, DeviceView<Elt>, Stream,
                  const GemmOpts&) { no_device("schur_update"); }
template<class Elt>
void gemv(const ModCtx<Elt>&, DeviceView<const Elt>, DeviceView<const Elt>, DeviceView<Elt>, GemmOp, Stream) {
    no_device("gemv");
}
template<class Elt>
void axpy(const ModCtx<Elt>&, Elt, DeviceView<const Elt>, DeviceView<Elt>, bool, Stream) { no_device("axpy"); }
template<class Elt>
void scal(const ModCtx<Elt>&, Elt, DeviceView<Elt>, Stream) { no_device("scal"); }
template<class Elt>
void canonicalise(const ModCtx<Elt>&, DeviceView<Elt>, Stream) { no_device("canonicalise"); }
void canonicalise(const ModCtx<uint32_t>&, DeviceView<const uint64_t>, DeviceView<uint32_t>, Stream) {
    no_device("canonicalise");
}
template<class Elt>
void copy(DeviceView<const Elt>, DeviceView<Elt>, Stream) { no_device("copy"); }
template<class Elt>
void transpose(DeviceView<const Elt>, DeviceView<Elt>, Stream) { no_device("transpose"); }
template<class Elt>
void fill(DeviceView<Elt>, Elt, Stream) { no_device("fill"); }
template<class Elt>
void axpy_identity(const ModCtx<Elt>&, Elt, DeviceView<Elt>, bool, Stream) { no_device("axpy_identity"); }
template<class Elt>
void gather(const ModCtx<Elt>&, Elt, Elt, DeviceView<const Elt>, uint32_t, const uint32_t*, DeviceView<Elt>, bool,
            Stream, const uint32_t*) { no_device("gather"); }
template<class Elt>
void trsm_unit(const ModCtx<Elt>&, DeviceView<const Elt>, DeviceView<Elt>, Stream, const GemmOpts&) {
    no_device("trsm_unit");
}

template<class Elt>
GemmBatch<Elt>::~GemmBatch() {}   // d_entries_ is never allocated without a device
template<class Elt>
void GemmBatch<Elt>::set_shape(uint32_t, uint32_t, uint32_t, size_t, size_t, size_t, size_t, size_t, size_t) {
    no_device("GemmBatch::set_shape");
}
template<class Elt>
void GemmBatch<Elt>::set_shape(DeviceView<const Elt>, DeviceView<const Elt>, DeviceView<Elt>) {
    no_device("GemmBatch::set_shape");
}
template<class Elt>
void GemmBatch<Elt>::push(DeviceView<const Elt>, DeviceView<const Elt>, DeviceView<Elt>) {
    no_device("GemmBatch::push");
}
template<class Elt>
void GemmBatch<Elt>::upload(Stream) { no_device("GemmBatch::upload"); }
template<class Elt>
bool GemmBatch<Elt>::prepare_imma(const ModCtx<Elt>&) { no_device("GemmBatch::prepare_imma"); }
template<class Elt>
size_t GemmBatch<Elt>::imma_bytes_needed() const { no_device("GemmBatch::imma_bytes_needed"); }
template<class Elt>
void GemmBatch<Elt>::run(const ModCtx<Elt>&, GemmOp, Stream, int, int, void*, size_t) const { no_device("GemmBatch::run"); }

template class GemmBatch<uint32_t>;
template class GemmBatch<uint64_t>;

// (slot-22 G2) no device elimination without a device.
ImmaRrefResult imma_rref(const ModCtx<uint32_t>&, uint32_t, uint32_t, const ImmaRowSource&, const ImmaRowSink&, uint32_t,
                         uint32_t) { no_device("imma_rref"); }
ImmaElimStats imma_elim_stats() { return ImmaElimStats{}; }
void          reset_imma_elim_stats() {}
size_t        imma_rref_device_bytes(uint32_t, uint32_t, uint32_t) { return 0; }

LaunchStats launch_stats() { return LaunchStats{}; }
void        reset_launch_stats() {}
const char* kernel_layer_version() { return "engine::cuda kernels: CPU-only stub (OC_ENABLE_CUDA=OFF)"; }

// ---- explicit instantiations: exactly the set of cuda_kernels.cu ----
#define INST(Elt)                                                                                            \
    template void gemm<Elt>(const ModCtx<Elt>&, DeviceView<const Elt>, DeviceView<const Elt>, DeviceView<Elt>, GemmOp, Stream, const GemmOpts&); \
    template void gemm_nt<Elt>(const ModCtx<Elt>&, DeviceView<const Elt>, DeviceView<const Elt>, DeviceView<Elt>, GemmOp, Stream, const GemmOpts&); \
    template void schur_update<Elt>(const ModCtx<Elt>&, DeviceView<const Elt>, DeviceView<const Elt>, DeviceView<Elt>, Stream, const GemmOpts&); \
    template void gemv<Elt>(const ModCtx<Elt>&, DeviceView<const Elt>, DeviceView<const Elt>, DeviceView<Elt>, GemmOp, Stream); \
    template void axpy<Elt>(const ModCtx<Elt>&, Elt, DeviceView<const Elt>, DeviceView<Elt>, bool, Stream);      \
    template void scal<Elt>(const ModCtx<Elt>&, Elt, DeviceView<Elt>, Stream);                                    \
    template void canonicalise<Elt>(const ModCtx<Elt>&, DeviceView<Elt>, Stream);                                 \
    template void copy<Elt>(DeviceView<const Elt>, DeviceView<Elt>, Stream);                                      \
    template void transpose<Elt>(DeviceView<const Elt>, DeviceView<Elt>, Stream);                                 \
    template void fill<Elt>(DeviceView<Elt>, Elt, Stream);                                                         \
    template void axpy_identity<Elt>(const ModCtx<Elt>&, Elt, DeviceView<Elt>, bool, Stream);                      \
    template void gather<Elt>(const ModCtx<Elt>&, Elt, Elt, DeviceView<const Elt>, uint32_t, const uint32_t*,      \
                              DeviceView<Elt>, bool, Stream, const uint32_t*);                                     \
    template void trsm_unit<Elt>(const ModCtx<Elt>&, DeviceView<const Elt>, DeviceView<Elt>, Stream, const GemmOpts&);
INST(uint32_t)
INST(uint64_t)
#undef INST

} // namespace cuda
} // namespace engine
