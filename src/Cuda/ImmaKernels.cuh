// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// ImmaKernels.cuh -- the int8 tensor-core (IMMA) modular GEMM (slot-22 G1)
// =====================================================================
//
// Included by cuda_kernels.cu ONLY (an nvcc TU); never by a C++ TU. The
// public surface is in CudaKernels.h (GemmOpts::imma*, imma_supported(),
// imma_policy(), imma_eligible(), imma_workspace_bytes()); ENGINE_API.md
// Sec. 11.8 is the binding description.
//
// Domain: a uint32_t context with 2 <= p <= 127 (any modulus, prime or not:
// the GEMM is modulus-only like every other device kernel). Canonical
// residues x in [0, p) are re-encoded as BALANCED residues
//     bal(x) = x - p  if x > floor(p/2),   x otherwise,      |bal(x)| <= floor(p/2) <= 63,
// which fit int8. The product A*B is accumulated EXACTLY in int32 by
// mma.sync.m16n8k32.s32.s8.s8.s32 (sm_80 .. sm_120), and the accumulator is
// reduced mod p with integer `%` -- no floating point anywhere:
//   * one k-tile (BK = 64) adds at most 64 * hb^2 in absolute value, hb = floor(p/2);
//   * without a flush, |acc| <= ktiles * 64 * hb^2, which is < 2^31 iff ktiles <= T(p);
//   * with a flush (acc %= p, |acc| <= p - 1 afterwards) every F <= T(p) k-tiles,
//     |acc| <= (p - 1) + F * 64 * hb^2 < 2^31 at every step,
//   with T(p) = floor((2^31 - 1 - (p - 1)) / (64 * hb^2)) (imma_max_flush_tiles(); 8454 at
//   p = 127, i.e. K <= 541 056 without a flush). So no int32 ever wraps, the
//   accumulator equals the exact integer dot product modulo p, and the
//   canonical result ((acc % p) + p) % p is the unique residue -- the same
//   bytes as engine::kernels::gemm by construction (CudaImmaBitIdentityValidation
//   asserts it with memcmp).
//
// Pipeline of one GEMM (C op= A*B, any strides):
//   1. imma_to_s8_kernel: A (m x k) -> A8 (m x Kp, row-major, zero-padded to
//      Kp = round_up(k, 16)); B^T (n x k) -> B8 (n x Kp). Both K-contiguous.
//   2. imma_gemm_kernel: block tile 128 x 128 x 64, 8 warps (warp tile 64 x 32),
//      3-stage cp.async pipeline, ldmatrix.x4, 80-byte smem rows, grouped
//      rasterisation -- the P5-G prototype's kernel (P5G_prototype_report.md
//      Sec. 1, docs/implementation/slot22/p5g/imma.cuh), with the canonical
//      u32 epilogue extended by the tape's op (Set / Add / Sub, the mod-p
//      add32 / sub32 of the u32 kernel) on an arbitrarily strided C.
//   Batched (GemmBatch): blockIdx.y / blockIdx.z selects the instance; each
//   instance has its own A8 / B8 slot in the workspace.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace engine {
namespace cuda {
namespace imma {

constexpr int BM = 128, BN = 128, BK = 64, STAGES = 3, PAD = 16, LDS = BK + PAD;   // 80-byte smem rows
constexpr int NTHREADS = 256;
constexpr size_t SMEM_BYTES = static_cast<size_t>(STAGES) * (BM + BN) * LDS;   // 61 440 B (dynamic)
constexpr uint32_t KALIGN = 16;   // K, lda, ldb multiple of 16 (16-byte cp.async chunks)

// The workspace layout (imma_padded_k / imma_b_offset / imma_workspace_bytes) and the
// K-chunking bound (imma_max_flush_tiles) are the CUDA-free inline functions of
// CudaKernels.h, shared with the CPU-only stub.

// ---- device helpers ---------------------------------------------------------------
__device__ __forceinline__ int8_t to_bal8(uint32_t x, uint32_t p) {
    return static_cast<int8_t>(x > (p >> 1) ? static_cast<int32_t>(x) - static_cast<int32_t>(p) : static_cast<int32_t>(x));
}
__device__ __forceinline__ uint32_t canon_of(int32_t acc, int32_t p) {
    int32_t r = acc % p;            // exact (C++ truncation), |r| < p
    if (r < 0) r += p;
    return static_cast<uint32_t>(r);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int sz = pred ? 16 : 0;   // src-size 0: the 16 bytes are zero-filled
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template<int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
__device__ __forceinline__ void mma_s8(int32_t* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- 1. u32 canonical -> balanced int8, row-major, zero-padded ----------------------
// dst(r, c) = bal(src(r, c)) for r < rows, c < cols; 0 for cols <= c < ld. 32 x 32 tiles
// through shared memory, so either source orientation (cs == 1 or rs == 1) is read
// coalesced; dst is written along its rows.
// Batched: blockIdx.z = instance (base + z); src = entries[..].A or .B (which), dst = ws + inst * inst_bytes + dst_off.
struct ConvArgs {
    const uint32_t* src; size_t rs, cs;       // the view (logical rows x cols)
    uint32_t rows, cols;
    int8_t* dst; uint32_t ld;
    uint32_t p;
    const GemmBatchEntry<uint32_t>* batch;    // non-null: batched
    uint32_t batch_base;
    int which;                                // batched: 0 = A, 1 = B
    int8_t* ws; size_t inst_bytes, dst_off;
};

__global__ void imma_to_s8_kernel(ConvArgs a) {
    __shared__ int32_t tile[32][33];
    const uint32_t r0 = blockIdx.y * 32, c0 = blockIdx.x * 32;
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    const uint32_t* src = a.src;
    int8_t* dst = a.dst;
    if (a.batch) {
        const uint32_t inst = a.batch_base + blockIdx.z;
        const GemmBatchEntry<uint32_t> e = a.batch[inst];
        src = (a.which == 0) ? e.A : e.B;
        dst = a.ws + static_cast<size_t>(inst) * a.inst_bytes + a.dst_off;
    }
    const bool col_fast = (a.cs != 1 && a.rs == 1);
    if (!col_fast) {
        for (uint32_t r = 0; r < 32; r += 8) {
            const uint32_t i = r0 + ty + r, j = c0 + tx;
            tile[ty + r][tx] = (i < a.rows && j < a.cols) ? to_bal8(src[i * a.rs + j * a.cs], a.p) : 0;
        }
    } else {
        for (uint32_t r = 0; r < 32; r += 8) {
            const uint32_t i = r0 + tx, j = c0 + ty + r;
            tile[tx][ty + r] = (i < a.rows && j < a.cols) ? to_bal8(src[i * a.rs + j * a.cs], a.p) : 0;
        }
    }
    __syncthreads();
    for (uint32_t r = 0; r < 32; r += 8) {
        const uint32_t i = r0 + ty + r, j = c0 + tx;
        if (i < a.rows && j < a.ld) dst[static_cast<size_t>(i) * a.ld + j] = static_cast<int8_t>(tile[ty + r][tx]);
    }
}

// ---- 2. the IMMA GEMM with the canonical u32 epilogue --------------------------------
struct GemmArgs {
    int M, N, K;                   // K = Kp (multiple of 16; zero-padded)
    const int8_t* A; const int8_t* Bt; int lda, ldb;
    uint32_t* C; size_t c_rs, c_cs;
    int op;                        // 0 Set, 1 Add, 2 Sub
    int32_t p;
    uint32_t flush_tiles;          // 0: never reduce inside the k loop
    const GemmBatchEntry<uint32_t>* batch;   // non-null: batched (blockIdx.y = instance - batch_base)
    uint32_t batch_base;
    const int8_t* ws; size_t inst_bytes, b_off;
    // (slot-22 G2) the int8 epilogues of the device elimination (ImmaElim.cuh): when C8 != nullptr
    // the result is written as BALANCED int8 residues into C8 (row-major, ldc8 a multiple of 16):
    // op 0 Set  c = bal(acc mod p),  op 2 Sub  c = bal((c - acc) mod p)  (in place; c is a balanced
    // residue on entry). C / c_rs / c_cs are ignored then. The tile is staged through shared memory
    // (the operand stages, free after the k loop) so global traffic is 16-byte coalesced.
    int8_t* C8; int ldc8;
};

// canonical residue -> balanced int8 (the to_bal8 rule), and bal of an exact |x| < 2^31
__device__ __forceinline__ int8_t bal_of_int(int32_t x, int32_t p) {
    int32_t r = x % p;              // exact, |r| < p
    if (r < 0) r += p;              // canonical [0, p)
    return static_cast<int8_t>(r > (p >> 1) ? r - p : r);
}

__global__ void __launch_bounds__(NTHREADS) imma_gemm_kernel(GemmArgs P) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    extern __shared__ __align__(16) int8_t smem[];
    int8_t* As = smem;                                              // [STAGES][BM][LDS]
    int8_t* Bs = smem + static_cast<size_t>(STAGES) * BM * LDS;     // [STAGES][BN][LDS]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;                        // 2 x 4 warps, warp tile 64 x 32
    const int M = P.M, N = P.N, K = P.K;
    const int8_t* A = P.A; const int8_t* Bt = P.Bt;
    uint32_t* C = P.C;
    if (P.batch) {
        const uint32_t inst = P.batch_base + blockIdx.y;
        A = P.ws + static_cast<size_t>(inst) * P.inst_bytes;
        Bt = A + P.b_off;
        C = P.batch[inst].C;
    }
    const int lda = P.lda, ldb = P.ldb;
    // grouped rasterisation (L2 reuse): blocks walk GROUP block-rows at a time, column by column
    const int nbm = (M + BM - 1) / BM, nbn = (N + BN - 1) / BN;
    constexpr int GROUP = 8;
    const int id = blockIdx.x, per = GROUP * nbn, first = (id / per) * GROUP;
    const int gs = min(nbm - first, GROUP), w = id % per;
    const int m0 = (first + w % gs) * BM, n0 = (w / gs) * BN;
    const int ktiles = (K + BK - 1) / BK;

    auto load_stage = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int c = tid + it * NTHREADS;                      // 512 chunks of 16 B per operand tile
            const int row = c >> 2, kc = (c & 3) * 16;
            const int gk = k0 + kc;
            {
                const int gr = m0 + row; const bool pr = gr < M && gk < K;
                const int8_t* src = A + static_cast<size_t>(pr ? gr : 0) * lda + (pr ? gk : 0);
                cp_async16(As + (static_cast<size_t>(stage) * BM + row) * LDS + kc, src, pr);
            }
            {
                const int gr = n0 + row; const bool pr = gr < N && gk < K;
                const int8_t* src = Bt + static_cast<size_t>(pr ? gr : 0) * ldb + (pr ? gk : 0);
                cp_async16(Bs + (static_cast<size_t>(stage) * BN + row) * LDS + kc, src, pr);
            }
        }
    };

    int32_t acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[i][j][t] = 0;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) { if (s < ktiles) load_stage(s, s); cp_async_commit(); }

    uint32_t since_flush = 0;
    for (int kt = 0; kt < ktiles; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        { const int nk = kt + STAGES - 1; if (nk < ktiles) load_stage(nk % STAGES, nk); cp_async_commit(); }
        const int st = kt % STAGES;
        const int8_t* as = As + static_cast<size_t>(st) * BM * LDS;
        const int8_t* bs = Bs + static_cast<size_t>(st) * BN * LDS;
#pragma unroll
        for (int ks = 0; ks < BK; ks += 32) {
            uint32_t af[4][4], bf[4][2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = wm * 64 + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int kof = ks + (lane >> 4) * 16;
                ldmatrix_x4(af[i][0], af[i][1], af[i][2], af[i][3], as + row * LDS + kof);
            }
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                const int row = wn * 32 + j * 8 + (lane & 7) + (lane >> 4) * 8;
                const int kof = ks + ((lane >> 3) & 1) * 16;
                ldmatrix_x4(bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1], bs + row * LDS + kof);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_s8(acc[i][j], af[i], bf[j]);
        }
        // the K-chunking rule: reduce before the int32 bound could be reached (uniform branch)
        if (P.flush_tiles != 0 && ++since_flush == P.flush_tiles) {
            since_flush = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int t = 0; t < 4; ++t) acc[i][j][t] %= P.p;
        }
    }
    cp_async_wait<0>();

    // epilogue: fragment c0,c1 -> (g, 2t..2t+1), c2,c3 -> (g+8, 2t..2t+1)
    const int g = lane >> 2, t = lane & 3;
    if (P.C8 != nullptr) {
        // (slot-22 G2) balanced int8 output, staged: T = 128 x 144 bytes over the operand stages.
        constexpr int LDT = BN + 16;
        int8_t* T = smem;
        int8_t* C8 = P.C8;
        const int ldc = P.ldc8;
        __syncthreads();   // every warp is done reading the stages
        const bool vec = (n0 + BN <= N) && ((ldc & 15) == 0) && ((reinterpret_cast<uintptr_t>(C8) & 15) == 0);
        if (P.op == 2) {
            for (int c = tid; c < BM * (BN / 16); c += NTHREADS) {
                const int row = c / (BN / 16), col = (c % (BN / 16)) * 16, gr = m0 + row;
                if (gr >= M) continue;
                if (vec) *reinterpret_cast<int4*>(T + row * LDT + col) = *reinterpret_cast<const int4*>(C8 + static_cast<size_t>(gr) * ldc + n0 + col);
                else for (int e = 0; e < 16; ++e) if (n0 + col + e < N) T[row * LDT + col + e] = C8[static_cast<size_t>(gr) * ldc + n0 + col + e];
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int row = wm * 64 + i * 16 + g + h * 8, col = wn * 32 + j * 8 + 2 * t + e;
                        const int32_t v = acc[i][j][2 * h + e] % P.p;   // |v| < p: no overflow below
                        const int32_t x = (P.op == 2) ? static_cast<int32_t>(T[row * LDT + col]) - v : v;
                        T[row * LDT + col] = bal_of_int(x, P.p);
                    }
        __syncthreads();
        for (int c = tid; c < BM * (BN / 16); c += NTHREADS) {
            const int row = c / (BN / 16), col = (c % (BN / 16)) * 16, gr = m0 + row;
            if (gr >= M) continue;
            if (vec) *reinterpret_cast<int4*>(C8 + static_cast<size_t>(gr) * ldc + n0 + col) = *reinterpret_cast<const int4*>(T + row * LDT + col);
            else for (int e = 0; e < 16; ++e) if (n0 + col + e < N) C8[static_cast<size_t>(gr) * ldc + n0 + col + e] = T[row * LDT + col + e];
        }
        return;
    }
    const uint32_t p = static_cast<uint32_t>(P.p);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = m0 + wm * 64 + i * 16 + g + h * 8;
                if (r >= M) continue;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int c = n0 + wn * 32 + j * 8 + 2 * t + e;
                    if (c >= N) continue;
                    const uint32_t v = canon_of(acc[i][j][2 * h + e], P.p);
                    uint32_t* cp = C + static_cast<size_t>(r) * P.c_rs + static_cast<size_t>(c) * P.c_cs;
                    if (P.op == 0) {
                        *cp = v;
                    } else if (P.op == 1) {
                        const uint32_t s = *cp + v;           // both < p <= 127: no wrap
                        *cp = (s >= p) ? s - p : s;
                    } else {
                        const uint32_t cur = *cp;
                        *cp = (cur >= v) ? cur - v : cur + (p - v);
                    }
                }
            }
#else
    __trap();   // never launched: imma_supported() requires an sm_80+ image of this kernel
#endif
}

} // namespace imma
} // namespace cuda
} // namespace engine
