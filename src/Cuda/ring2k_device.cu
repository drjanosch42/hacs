// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (slot-24 PAR-GPU) ring2k_device.cu -- the device side of the ring2k device-resident LU.  See CudaRing2k.h and
// src/ModArithmetic/Binary/Ring2kDevice.h.  Exact wraparound integer arithmetic only (uint32_t / uint64_t): no reduction, no
// floating point, no atomics; every output entry is written by exactly one thread.

#include "CudaRing2k.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>

#include "EngineCheck.h"

namespace engine {
namespace cuda {
namespace ring2k {

namespace {

std::string err_str(cudaError_t e, const char* what, int line) {
    std::ostringstream os;
    os << "CUDA error in " << what << " (ring2k_device.cu:" << line << "): " << cudaGetErrorName(e) << " -- " << cudaGetErrorString(e);
    return os.str();
}
#define R2_CALL(call)                                                                 \
    do {                                                                              \
        const cudaError_t e_ = (call);                                                \
        if (e_ != cudaSuccess) throw ::engine::cuda::CudaError(err_str(e_, #call, __LINE__)); \
    } while (0)

// ---- kernels -------------------------------------------------------------------------------------------------------
constexpr int kBM = 64, kBN = 64, kBK = 16, kThreads = 256;

// C (M x N, ldc) -= L (M x K, ldl) U (K x N, ldu).  64 x 64 output tile per block, 4 x 4 per thread (rows ty + 16 q,
// columns tx + 16 p: a warp writes 16 consecutive columns of two rows), K in slices of 16 through shared memory.
template<class E>
__global__ void __launch_bounds__(kThreads) gemm_sub_kernel(E* __restrict__ C, size_t ldc, const E* __restrict__ L, size_t ldl,
                                                            const E* __restrict__ U, size_t ldu, uint32_t M, uint32_t N, uint32_t K) {
    __shared__ E Ls[kBK][kBM];
    __shared__ E Us[kBK][kBN];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const uint32_t row0 = blockIdx.y * kBM, col0 = blockIdx.x * kBN;
    E acc[4][4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int p = 0; p < 4; ++p) acc[q][p] = 0;
    for (uint32_t k0 = 0; k0 < K; k0 += kBK) {
#pragma unroll
        for (int q = 0; q < (kBM * kBK) / kThreads; ++q) {
            const int e = tid + kThreads * q;
            const int i = e / kBK, kk = e % kBK;
            const uint32_t gr = row0 + i, gk = k0 + kk;
            Ls[kk][i] = (gr < M && gk < K) ? L[static_cast<size_t>(gr) * ldl + gk] : E(0);
        }
#pragma unroll
        for (int q = 0; q < (kBN * kBK) / kThreads; ++q) {
            const int e = tid + kThreads * q;
            const int kk = e / kBN, j = e % kBN;
            const uint32_t gk = k0 + kk, gc = col0 + j;
            Us[kk][j] = (gk < K && gc < N) ? U[static_cast<size_t>(gk) * ldu + gc] : E(0);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kBK; ++kk) {
            E a[4], b[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) a[q] = Ls[kk][ty + 16 * q];
#pragma unroll
            for (int p = 0; p < 4; ++p) b[p] = Us[kk][tx + 16 * p];
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
                for (int p = 0; p < 4; ++p) acc[q][p] += a[q] * b[p];
        }
        __syncthreads();
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t gr = row0 + ty + 16 * q;
        if (gr >= M) continue;
        E* crow = C + static_cast<size_t>(gr) * ldc;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t gc = col0 + tx + 16 * p;
            if (gc < N) crow[gc] -= acc[q][p];
        }
    }
}

// The unit-lower TRSM on the pivot rows: for every column j of [c0, c1), row t (t = 1 .. np-1) -= L[t][t2] row t2 for
// t2 < t, in that order (one thread per column; A points at the first pivot row, L at its L slots, ld = ldl).
template<class E>
__global__ void trsm_kernel(E* __restrict__ A, size_t lda, const E* __restrict__ L, size_t ldl, uint32_t np, uint32_t c0, uint32_t c1) {
    const uint32_t j = c0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= c1) return;
    for (uint32_t t = 1; t < np; ++t) {
        E acc = A[static_cast<size_t>(t) * lda + j];
        const E* lt = L + static_cast<size_t>(t) * ldl;
        for (uint32_t t2 = 0; t2 < t; ++t2) acc -= lt[t2] * A[static_cast<size_t>(t2) * lda + j];
        A[static_cast<size_t>(t) * lda + j] = acc;
    }
}

// The panel's row swaps (pairs (a, b), in order) on every column of [0, cw) outside [pc0, pc1).
template<class E>
__global__ void swap_kernel(E* __restrict__ A, size_t lda, uint32_t cw, uint32_t pc0, uint32_t pc1, const uint32_t* __restrict__ sw,
                            uint32_t nswaps) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cw || (j >= pc0 && j < pc1)) return;
    for (uint32_t q = 0; q < nswaps; ++q) {
        E* a = A + static_cast<size_t>(sw[2 * q]) * lda + j;
        E* b = A + static_cast<size_t>(sw[2 * q + 1]) * lda + j;
        const E x = *a;
        *a = *b;
        *b = x;
    }
}

template<class E>
void launch_gemm(E* C, size_t ldc, const E* L, size_t ldl, const E* U, size_t ldu, uint32_t M, uint32_t N, uint32_t K, cudaStream_t s) {
    if (M == 0 || N == 0 || K == 0) return;
    for (uint32_t m0 = 0; m0 < M; m0 += 65535u * kBM) {   // grid.y <= 65535
        const uint32_t mm = std::min<uint32_t>(M - m0, 65535u * kBM);
        const dim3 grid((N + kBN - 1) / kBN, (mm + kBM - 1) / kBM);
        gemm_sub_kernel<E><<<grid, kThreads, 0, s>>>(C + static_cast<size_t>(m0) * ldc, ldc, L + static_cast<size_t>(m0) * ldl, ldl, U, ldu,
                                                     mm, N, K);
    }
    R2_CALL(cudaGetLastError());
}

// ---- the session ---------------------------------------------------------------------------------------------------
struct Session {
    uint32_t elem = 0, m = 0, cw = 0, panel = 0;
    char* dA = nullptr;
    char* dL = nullptr;
    uint32_t* dSw = nullptr;
    char* hPanel = nullptr;
    char* hL = nullptr;
    char* hNext = nullptr;
    uint32_t* hSw = nullptr;
    size_t slab = 0;
    cudaStream_t s1 = nullptr, s2 = nullptr;
    cudaEvent_t ev_la = nullptr, ev_stage = nullptr;
    bool stage_pending = false;
    bool la_pending = false;
    uint32_t la_rows = 0, la_cols = 0;

    char* at(uint32_t r, uint32_t c) const { return dA + (static_cast<size_t>(r) * cw + c) * elem; }
    void free_all() noexcept {
        if (s1) (void)cudaStreamSynchronize(s1);
        if (s2) (void)cudaStreamSynchronize(s2);
        if (dA) (void)cudaFree(dA);
        if (dL) (void)cudaFree(dL);
        if (dSw) (void)cudaFree(dSw);
        if (hPanel) (void)cudaFreeHost(hPanel);
        if (hL) (void)cudaFreeHost(hL);
        if (hNext) (void)cudaFreeHost(hNext);
        if (hSw) (void)cudaFreeHost(hSw);
        if (ev_la) (void)cudaEventDestroy(ev_la);
        if (ev_stage) (void)cudaEventDestroy(ev_stage);
        if (s1) (void)cudaStreamDestroy(s1);
        if (s2) (void)cudaStreamDestroy(s2);
        (void)cudaGetLastError();
        dA = dL = hPanel = hL = hNext = nullptr;
        dSw = hSw = nullptr;
        s1 = s2 = nullptr;
        ev_la = ev_stage = nullptr;
    }
};

void select_device() {
    CudaCtx::get();   // the probe, the device choice (ENGINE_CUDA_DEVICE), the context
    R2_CALL(cudaSetDevice(device_info().index));
}

// Copies a rows x cols block between a strided host block (ld = ldh elements) and a dense staging slab.
void pack(char* dst, const char* src, size_t ldh_bytes, size_t row_bytes, uint32_t rows) {
    if (ldh_bytes == row_bytes) { std::memcpy(dst, src, row_bytes * rows); return; }
    for (uint32_t i = 0; i < rows; ++i) std::memcpy(dst + row_bytes * i, src + ldh_bytes * i, row_bytes);
}
void unpack(char* dst, size_t ldh_bytes, const char* src, size_t row_bytes, uint32_t rows) {
    if (ldh_bytes == row_bytes) { std::memcpy(dst, src, row_bytes * rows); return; }
    for (uint32_t i = 0; i < rows; ++i) std::memcpy(dst + ldh_bytes * i, src + row_bytes * i, row_bytes);
}

template<class E>
void step_t(Session& S, const engine::ring2k::device::StepArgs& a) {
    const uint32_t rows = a.m - a.r, rn = a.r + a.np, cw = S.cw;   // cw: the device row stride
    const uint32_t W = a.w ? a.w : S.cw;                             // the window columns in use
    if (S.stage_pending) {
        R2_CALL(cudaEventSynchronize(S.ev_stage));
        S.stage_pending = false;
    }
    const size_t E_ = sizeof(E);
    if (a.panel != nullptr && rows > 0) {
        const uint32_t pw = a.pc1 - a.pc0;
        pack(S.hPanel, static_cast<const char*>(a.panel), a.ldp * E_, pw * E_, rows);
        R2_CALL(cudaMemcpy2DAsync(S.at(a.r, a.pc0), cw * E_, S.hPanel, pw * E_, pw * E_, rows, cudaMemcpyHostToDevice, S.s1));
    }
    if (a.nswaps > 0) {
        std::memcpy(S.hSw, a.swaps, sizeof(uint32_t) * 2 * a.nswaps);
        R2_CALL(cudaMemcpyAsync(S.dSw, S.hSw, sizeof(uint32_t) * 2 * a.nswaps, cudaMemcpyHostToDevice, S.s1));
        swap_kernel<E><<<(W + 255) / 256, 256, 0, S.s1>>>(reinterpret_cast<E*>(S.dA), cw, W, a.pc0, a.pc1, S.dSw, a.nswaps);
        R2_CALL(cudaGetLastError());
    }
    const bool update = a.np > 0 && a.u0 < W;
    if (update) {
        pack(S.hL, static_cast<const char*>(a.L), a.ldl * E_, a.np * E_, rows);
        R2_CALL(cudaMemcpy2DAsync(S.dL, S.panel * E_, S.hL, a.np * E_, a.np * E_, rows, cudaMemcpyHostToDevice, S.s1));
    }
    R2_CALL(cudaEventRecord(S.ev_stage, S.s1));
    S.stage_pending = true;
    E* A = reinterpret_cast<E*>(S.dA);
    const E* Ld = reinterpret_cast<const E*>(S.dL);
    const bool la = a.la1 > a.la0 && rn < a.m;
    auto gemm_cols = [&](uint32_t c0, uint32_t c1) {
        if (c1 > c0 && rn < a.m)
            launch_gemm<E>(A + static_cast<size_t>(rn) * cw + c0, cw, Ld + static_cast<size_t>(a.np) * S.panel, S.panel,
                           A + static_cast<size_t>(a.r) * cw + c0, cw, a.m - rn, c1 - c0, a.np, S.s1);
    };
    if (update) {
        if (a.np >= 2) {
            trsm_kernel<E><<<(W - a.u0 + 255) / 256, 256, 0, S.s1>>>(A + static_cast<size_t>(a.r) * cw, cw, Ld, S.panel, a.np, a.u0, W);
            R2_CALL(cudaGetLastError());
        }
        if (la && a.la0 == a.u0) {
            gemm_cols(a.u0, a.la1);
        } else {
            gemm_cols(a.u0, W);
        }
    }
    if (la) {
        R2_CALL(cudaEventRecord(S.ev_la, S.s1));
        R2_CALL(cudaStreamWaitEvent(S.s2, S.ev_la, 0));
        const uint32_t lw = a.la1 - a.la0, lr = a.m - rn;
        R2_CALL(cudaMemcpy2DAsync(S.hNext, lw * E_, S.at(rn, a.la0), cw * E_, lw * E_, lr, cudaMemcpyDeviceToHost, S.s2));
        S.la_pending = true;
        S.la_rows = lr;
        S.la_cols = lw;
        if (update && a.la0 == a.u0) gemm_cols(a.la1, W);
    }
}

} // namespace

MemInfo mem_info() {
    select_device();
    MemInfo mi;
    R2_CALL(cudaMemGetInfo(&mi.free_bytes, &mi.total_bytes));
    return mi;
}

size_t session_bytes(uint32_t elem, uint32_t m, uint32_t cw, uint32_t panel) {
    return static_cast<size_t>(m) * cw * elem + static_cast<size_t>(m) * panel * elem + sizeof(uint32_t) * 2 * (panel + 1);
}
size_t staging_bytes(uint32_t elem, uint32_t m, uint32_t panel) {
    return 3 * static_cast<size_t>(m) * panel * elem + sizeof(uint32_t) * 2 * (panel + 1);
}

void* open(uint32_t elem, uint32_t m, uint32_t cw, uint32_t panel) {
    ENGINE_CHECK_MSG(elem == 4 || elem == 8, "cuda::ring2k::open: elem must be 4 or 8");
    ENGINE_CHECK_MSG(m > 0 && cw > 0 && panel > 0, "cuda::ring2k::open: empty session");
    select_device();
    Session* S = new Session();
    S->elem = elem;
    S->m = m;
    S->cw = cw;
    S->panel = panel;
    try {
        S->slab = static_cast<size_t>(m) * panel * elem;
        R2_CALL(cudaStreamCreateWithFlags(&S->s1, cudaStreamNonBlocking));
        R2_CALL(cudaStreamCreateWithFlags(&S->s2, cudaStreamNonBlocking));
        R2_CALL(cudaEventCreateWithFlags(&S->ev_la, cudaEventDisableTiming));
        R2_CALL(cudaEventCreateWithFlags(&S->ev_stage, cudaEventDisableTiming));
        R2_CALL(cudaMalloc(reinterpret_cast<void**>(&S->dA), static_cast<size_t>(m) * cw * elem));
        R2_CALL(cudaMalloc(reinterpret_cast<void**>(&S->dL), S->slab));
        R2_CALL(cudaMalloc(reinterpret_cast<void**>(&S->dSw), sizeof(uint32_t) * 2 * (panel + 1)));
        R2_CALL(cudaHostAlloc(reinterpret_cast<void**>(&S->hPanel), S->slab, cudaHostAllocDefault));
        R2_CALL(cudaHostAlloc(reinterpret_cast<void**>(&S->hL), S->slab, cudaHostAllocDefault));
        R2_CALL(cudaHostAlloc(reinterpret_cast<void**>(&S->hNext), S->slab, cudaHostAllocDefault));
        R2_CALL(cudaHostAlloc(reinterpret_cast<void**>(&S->hSw), sizeof(uint32_t) * 2 * (panel + 1), cudaHostAllocDefault));
    } catch (...) {
        S->free_all();
        delete S;
        throw;
    }
    return S;
}

void put(void* s, const void* host, size_t ldh, uint32_t r0, uint32_t rows, uint32_t c0, uint32_t cols) {
    Session& S = *static_cast<Session*>(s);
    if (rows == 0 || cols == 0) return;
    const size_t E_ = S.elem;
    R2_CALL(cudaMemcpy2DAsync(S.at(r0, c0), S.cw * E_, host, ldh * E_, cols * E_, rows, cudaMemcpyHostToDevice, S.s1));
    R2_CALL(cudaStreamSynchronize(S.s1));
}

void get(void* s, void* host, size_t ldh, uint32_t r0, uint32_t rows, uint32_t c0, uint32_t cols) {
    Session& S = *static_cast<Session*>(s);
    R2_CALL(cudaStreamSynchronize(S.s2));
    S.la_pending = false;
    if (rows == 0 || cols == 0) { R2_CALL(cudaStreamSynchronize(S.s1)); return; }
    const size_t E_ = S.elem, rb = cols * E_;
    if (rb * rows <= S.slab) {
        R2_CALL(cudaMemcpy2DAsync(S.hNext, rb, S.at(r0, c0), S.cw * E_, rb, rows, cudaMemcpyDeviceToHost, S.s1));
        R2_CALL(cudaStreamSynchronize(S.s1));
        unpack(static_cast<char*>(host), ldh * E_, S.hNext, rb, rows);
    } else {
        R2_CALL(cudaMemcpy2DAsync(host, ldh * E_, S.at(r0, c0), S.cw * E_, rb, rows, cudaMemcpyDeviceToHost, S.s1));
        R2_CALL(cudaStreamSynchronize(S.s1));
    }
}

void step(void* s, const engine::ring2k::device::StepArgs& a) {
    Session& S = *static_cast<Session*>(s);
    ENGINE_CHECK_MSG(!S.la_pending, "cuda::ring2k::step: the previous lookahead was not fetched");
    ENGINE_CHECK_MSG(a.m == S.m && a.r <= a.m && a.np <= S.panel && a.nswaps <= S.panel && a.pc0 <= a.pc1,
                     "cuda::ring2k::step: arguments outside the session");
    ENGINE_CHECK_MSG(a.pc1 - a.pc0 <= S.panel && (a.panel == nullptr || a.pc1 <= S.cw), "cuda::ring2k::step: bad panel");
    ENGINE_CHECK_MSG(a.w <= S.cw && a.la1 <= (a.w ? a.w : S.cw) && (a.la1 <= a.la0 || a.la1 - a.la0 <= S.panel),
                     "cuda::ring2k::step: bad window / lookahead");
    if (S.elem == 4) step_t<uint32_t>(S, a);
    else step_t<uint64_t>(S, a);
}

void fetch(void* s, void* host, size_t ldh) {
    Session& S = *static_cast<Session*>(s);
    ENGINE_CHECK_MSG(S.la_pending, "cuda::ring2k::fetch: no lookahead pending");
    R2_CALL(cudaStreamSynchronize(S.s2));
    S.la_pending = false;
    unpack(static_cast<char*>(host), ldh * S.elem, S.hNext, static_cast<size_t>(S.la_cols) * S.elem, S.la_rows);
}

void close(void* s) noexcept {
    Session* S = static_cast<Session*>(s);
    if (!S) return;
    S->free_all();
    delete S;
}

void gemm_sub_host(uint32_t elem, void* C, size_t ldc, const void* L, size_t ldl, const void* U, size_t ldu, uint32_t m, uint32_t n,
                   uint32_t k) {
    ENGINE_CHECK_MSG(elem == 4 || elem == 8, "cuda::ring2k::gemm_sub_host: elem must be 4 or 8");
    select_device();
    if (m == 0 || n == 0) return;
    const size_t E_ = elem;
    char *dC = nullptr, *dL = nullptr, *dU = nullptr;
    cudaStream_t s = nullptr;
    try {
        R2_CALL(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        R2_CALL(cudaMalloc(reinterpret_cast<void**>(&dC), static_cast<size_t>(m) * n * E_));
        R2_CALL(cudaMalloc(reinterpret_cast<void**>(&dL), static_cast<size_t>(m) * std::max<uint32_t>(k, 1) * E_));
        R2_CALL(cudaMalloc(reinterpret_cast<void**>(&dU), static_cast<size_t>(std::max<uint32_t>(k, 1)) * n * E_));
        R2_CALL(cudaMemcpy2DAsync(dC, n * E_, C, ldc * E_, n * E_, m, cudaMemcpyHostToDevice, s));
        if (k > 0) {
            R2_CALL(cudaMemcpy2DAsync(dL, k * E_, L, ldl * E_, k * E_, m, cudaMemcpyHostToDevice, s));
            R2_CALL(cudaMemcpy2DAsync(dU, n * E_, U, ldu * E_, n * E_, k, cudaMemcpyHostToDevice, s));
        }
        if (elem == 4)
            launch_gemm<uint32_t>(reinterpret_cast<uint32_t*>(dC), n, reinterpret_cast<const uint32_t*>(dL), k,
                                  reinterpret_cast<const uint32_t*>(dU), n, m, n, k, s);
        else
            launch_gemm<uint64_t>(reinterpret_cast<uint64_t*>(dC), n, reinterpret_cast<const uint64_t*>(dL), k,
                                  reinterpret_cast<const uint64_t*>(dU), n, m, n, k, s);
        R2_CALL(cudaMemcpy2DAsync(C, ldc * E_, dC, n * E_, n * E_, m, cudaMemcpyDeviceToHost, s));
        R2_CALL(cudaStreamSynchronize(s));
    } catch (...) {
        if (s) (void)cudaStreamSynchronize(s);
        (void)cudaFree(dC);
        (void)cudaFree(dL);
        (void)cudaFree(dU);
        if (s) (void)cudaStreamDestroy(s);
        (void)cudaGetLastError();
        throw;
    }
    (void)cudaFree(dC);
    (void)cudaFree(dL);
    (void)cudaFree(dU);
    (void)cudaStreamDestroy(s);
}

} // namespace ring2k
} // namespace cuda
} // namespace engine
