// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// =====================================================================
// cuda_kernels.cu -- device kernels of the matrix engine (Phase 5)
// =====================================================================
// See CudaKernels.h for the contract and ENGINE_API.md Sec. 11 for the
// binding description. Everything here is exact integer arithmetic.

#include "CudaKernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <sstream>
#include <string>
#include <type_traits>

#include "EngineCheck.h"
#include "RetiredDeviceLevers.h"   // (S24 CL-R2) ENGINE_GPU_IMMA
#include "ImmaKernels.cuh"   // (slot-22 G1) the int8 tensor-core GEMM; this TU only

namespace engine {
namespace cuda {

namespace {

// ---------------------------------------------------------------------------
// error handling / counters
// ---------------------------------------------------------------------------
std::string err_str(cudaError_t e, const char* what, int line) {
    std::ostringstream os;
    os << "CUDA error in " << what << " (cuda_kernels.cu:" << line << "): "
       << cudaGetErrorName(e) << " -- " << cudaGetErrorString(e);
    return os.str();
}
#define CUDA_CALL(call)                                                                  \
    do {                                                                                 \
        const cudaError_t e_ = (call);                                                   \
        if (e_ != cudaSuccess) throw ::engine::cuda::CudaError(err_str(e_, #call, __LINE__)); \
    } while (0)

inline cudaStream_t S(Stream s) { return static_cast<cudaStream_t>(s.h); }

// (hecke-extensions slot 9, WP-6) relaxed atomics: kernels are launched from
// several threads at once (the executor's device levels of concurrently
// running programs -- HeckeExtDeterminismValidation's `gpu` arms since their
// programs reach the device), and the plain counters were a data race under
// CheckThread.  Diagnostic counters only; launch_stats() snapshots them.
struct AtomicLaunchStats {
    std::atomic<uint64_t> kernel_launches{0}, gemm_launches{0}, batched_instances{0}, imma_launches{0};
};
AtomicLaunchStats g_stats;

inline void after_launch(const char* what) {
    g_stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw CudaError(err_str(e, what, __LINE__));
}

// ---------------------------------------------------------------------------
// modular arithmetic on the device -- the twins of ModCtx's primitives
// ---------------------------------------------------------------------------
struct ModConsts {
    uint64_t p;          // modulus
    uint64_t barrett_m;  // u32: floor(2^64 / p)
    uint64_t ninv;       // u64: -p^{-1} mod 2^64 (Montgomery)
    uint64_t r2;         // u64: R^2 mod p
    uint32_t mont;       // u64: Montgomery available
};

template<class Elt> ModConsts consts_of(const ModCtx<Elt>& ctx) {
    ModConsts c;
    c.p = ctx.p; c.barrett_m = ctx.barrett_m; c.ninv = ctx.mont_ninv; c.r2 = ctx.mont_r2;
    c.mont = ctx.has_montgomery ? 1u : 0u;
    return c;
}

// Barrett: x mod p for EVERY x < 2^64 (== ModCtx<uint32_t>::reduce_u64).
__device__ __forceinline__ uint32_t red32(uint64_t x, uint32_t p, uint64_t m) {
    const uint64_t q = __umul64hi(x, m);
    uint64_t r = x - q * static_cast<uint64_t>(p);
    if (r >= p) r -= p;
    return static_cast<uint32_t>(r);
}
__device__ __forceinline__ uint32_t add32(uint32_t a, uint32_t b, uint32_t p) {
    const uint32_t r = a + b;
    return (r < a || r >= p) ? r - p : r;
}
__device__ __forceinline__ uint32_t sub32(uint32_t a, uint32_t b, uint32_t p) {
    return (a >= b) ? (a - b) : (a + (p - b));
}
__device__ __forceinline__ uint64_t add64(uint64_t a, uint64_t b, uint64_t p) {
    const uint64_t r = a + b;
    return (r < a || r >= p) ? r - p : r;
}
__device__ __forceinline__ uint64_t sub64(uint64_t a, uint64_t b, uint64_t p) {
    return (a >= b) ? (a - b) : (a + (p - b));
}
// Montgomery REDC of T = hi*2^64 + lo < p*2^64 (== ModCtx<uint64_t>::mont64_reduce).
__device__ __forceinline__ uint64_t redc64(uint64_t lo, uint64_t hi, uint64_t p, uint64_t ninv) {
    const uint64_t m = lo * ninv;
    const uint64_t mp_lo = m * p;
    const uint64_t mp_hi = __umul64hi(m, p);
    const uint64_t s_lo = lo + mp_lo;
    const uint64_t carry = (s_lo < lo) ? 1u : 0u;     // s_lo == 0 by construction
    uint64_t t = hi + mp_hi + carry;                   // < 2p, no wrap for p < 2^63
    if (t >= p) t -= p;
    return t;
}
// a*b*R^{-1} mod p
__device__ __forceinline__ uint64_t mont_mul(uint64_t a, uint64_t b, uint64_t p, uint64_t ninv) {
    return redc64(a * b, __umul64hi(a, b), p, ninv);
}
// Exact (hi*2^64 + lo) mod p for ANY p >= 2 (the U64_Wide fallback): shift-subtract.
__device__ uint64_t mod128_wide(uint64_t lo, uint64_t hi, uint64_t p) {
    if (hi == 0) return lo % p;
    uint64_t rem = 0;
    for (int b = 127; b >= 0; --b) {
        const uint64_t bit = (b >= 64) ? ((hi >> (b - 64)) & 1u) : ((lo >> b) & 1u);
        const uint64_t carry = rem >> 63;
        rem = (rem << 1) | bit;
        if (carry || rem >= p) rem -= p;
    }
    return rem;
}
// a*b mod p, canonical, any u64 p (wide path).
__device__ __forceinline__ uint64_t mulmod64_wide(uint64_t a, uint64_t b, uint64_t p) {
    return mod128_wide(a * b, __umul64hi(a, b), p);
}

// ---------------------------------------------------------------------------
// the accumulator abstraction: one output cell's running state
//   u32: exact u64 sum over <= kb products, flushed by Barrett into `part`
//   u64: Montgomery-form partial sums (already canonical), scaled once at the end
// ---------------------------------------------------------------------------
template<class Elt> struct Acc;

template<> struct Acc<uint32_t> {
    uint64_t acc = 0;
    uint32_t part = 0;
    __device__ __forceinline__ void mac(uint32_t a, uint32_t b) { acc += static_cast<uint64_t>(a) * b; }
    __device__ __forceinline__ void flush(const ModConsts& c) {
        part = add32(part, red32(acc, static_cast<uint32_t>(c.p), c.barrett_m), static_cast<uint32_t>(c.p));
        acc = 0;
    }
    __device__ __forceinline__ uint32_t result(const ModConsts& c) { flush(c); return part; }
};
template<> struct Acc<uint64_t> {
    uint64_t part = 0;
    __device__ __forceinline__ void mac(uint64_t a, uint64_t b, const ModConsts& c) {
        // mont: term = ab R^{-1}; wide: term = ab (both canonical)
        const uint64_t term = c.mont ? mont_mul(a, b, c.p, c.ninv) : mulmod64_wide(a, b, c.p);
        part = add64(part, term, c.p);
    }
    __device__ __forceinline__ void flush(const ModConsts&) {}
    __device__ __forceinline__ uint64_t result(const ModConsts& c) {
        // mont: part = S R^{-1}  ->  S = REDC(part * R^2)
        return c.mont ? mont_mul(part, c.r2, c.p, c.ninv) : part;
    }
};

template<class Elt>
__device__ __forceinline__ Elt apply_op(Elt cur, Elt v, int op, uint64_t p) {
    if (op == 0) return v;
    if (sizeof(Elt) == 4) {
        const uint32_t pp = static_cast<uint32_t>(p);
        return static_cast<Elt>(op == 1 ? add32(static_cast<uint32_t>(cur), static_cast<uint32_t>(v), pp)
                                        : sub32(static_cast<uint32_t>(cur), static_cast<uint32_t>(v), pp));
    }
    return static_cast<Elt>(op == 1 ? add64(cur, v, p) : sub64(cur, v, p));
}

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------
template<class Elt>
struct GemmParams {
    const Elt* A; const Elt* B; Elt* C;
    size_t a_rs, a_cs, b_rs, b_cs, c_rs, c_cs;
    uint32_t m, n, k;
    int op;
    uint32_t kb;            // u32: flush interval -- in k-TILES (!PERMAC) or in MACs (PERMAC)
    uint32_t k_per_split;   // split-K: k range per blockIdx.z (0 = no split)
    Elt* ws;                // split-K workspace: splits x m x n row-major
    const GemmBatchEntry<Elt>* batch;   // batched: instance table (blockIdx.z)
    uint32_t batch_base;    // first instance of this launch
    ModConsts mc;
};

// One thread block computes a BM x BN tile of C; each thread a TM x TN
// register block; k in tiles of BK through shared memory.
template<class Elt, int BM, int BN, int BK, int TM, int TN, bool A_KC, bool B_NC, bool PERMAC>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm_kernel(GemmParams<Elt> P) {
    constexpr int NT = (BM / TM) * (BN / TN);
    constexpr int TX = BN / TN;
    __shared__ Elt As[BK][BM + 1];
    __shared__ Elt Bs[BK][BN + 1];

    const int tid = threadIdx.x;
    const int tx = tid % TX, ty = tid / TX;
    const uint32_t bm0 = blockIdx.y * BM, bn0 = blockIdx.x * BN;

    const Elt* A = P.A; const Elt* B = P.B; Elt* C = P.C;
    uint32_t kbeg = 0, kend = P.k;
    if (P.batch) {
        const GemmBatchEntry<Elt> e = P.batch[P.batch_base + blockIdx.z];
        A = e.A; B = e.B; C = e.C;
    } else if (P.k_per_split) {
        kbeg = blockIdx.z * P.k_per_split;
        kend = min(P.k, kbeg + P.k_per_split);
    }

    Acc<Elt> acc[TM][TN];
    uint32_t cnt = 0;

    for (uint32_t kt = kbeg; kt < kend; kt += BK) {
        // ---- load A tile (BM x BK) ----
#pragma unroll
        for (int r = 0; r < (BM * BK + NT - 1) / NT; ++r) {
            const int idx = tid + r * NT;
            if (idx < BM * BK) {
                int i, kk;
                if (A_KC) { i = idx / BK; kk = idx % BK; } else { i = idx % BM; kk = idx / BM; }
                const uint32_t gi = bm0 + i, gk = kt + kk;
                As[kk][i] = (gi < P.m && gk < kend) ? A[gi * P.a_rs + gk * P.a_cs] : Elt(0);
            }
        }
        // ---- load B tile (BK x BN) ----
#pragma unroll
        for (int r = 0; r < (BK * BN + NT - 1) / NT; ++r) {
            const int idx = tid + r * NT;
            if (idx < BK * BN) {
                int j, kk;
                if (B_NC) { j = idx % BN; kk = idx / BN; } else { kk = idx % BK; j = idx / BK; }
                const uint32_t gj = bn0 + j, gk = kt + kk;
                Bs[kk][j] = (gj < P.n && gk < kend) ? B[gk * P.b_rs + gj * P.b_cs] : Elt(0);
            }
        }
        __syncthreads();
        // ---- multiply ----
        const int kk_end = (kend - kt) < static_cast<uint32_t>(BK) ? static_cast<int>(kend - kt) : BK;
        for (int kk = 0; kk < kk_end; ++kk) {
            Elt a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    if constexpr (sizeof(Elt) == 4) acc[i][j].mac(a[i], b[j]);
                    else                            acc[i][j].mac(a[i], b[j], P.mc);
                }
            if constexpr (PERMAC && sizeof(Elt) == 4) {
                if (++cnt == P.kb) {
                    cnt = 0;
#pragma unroll
                    for (int i = 0; i < TM; ++i)
#pragma unroll
                        for (int j = 0; j < TN; ++j) acc[i][j].flush(P.mc);
                }
            }
        }
        __syncthreads();
        if constexpr (!PERMAC && sizeof(Elt) == 4) {
            if (++cnt == P.kb) {
                cnt = 0;
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j].flush(P.mc);
            }
        }
    }

    // ---- epilogue ----
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const uint32_t gi = bm0 + ty * TM + i;
        if (gi >= P.m) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const uint32_t gj = bn0 + tx * TN + j;
            if (gj >= P.n) continue;
            const Elt v = acc[i][j].result(P.mc);
            if (P.k_per_split) {
                P.ws[(static_cast<size_t>(blockIdx.z) * P.m + gi) * P.n + gj] = v;
            } else {
                Elt* c = C + gi * P.c_rs + gj * P.c_cs;
                *c = apply_op<Elt>(*c, v, P.op, P.mc.p);
            }
        }
    }
}

// Split-K combine: C op= sum_z ws[z] (all partials canonical; fixed order).
template<class Elt>
__global__ void splitk_combine_kernel(const Elt* ws, uint32_t splits, uint32_t m, uint32_t n,
                                      Elt* C, size_t c_rs, size_t c_cs, int op, ModConsts mc) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    Elt s = ws[static_cast<size_t>(i) * n + j];
    for (uint32_t z = 1; z < splits; ++z) {
        const Elt v = ws[(static_cast<size_t>(z) * m + i) * n + j];
        if (sizeof(Elt) == 4) s = static_cast<Elt>(add32(static_cast<uint32_t>(s), static_cast<uint32_t>(v), static_cast<uint32_t>(mc.p)));
        else                  s = static_cast<Elt>(add64(s, v, mc.p));
    }
    Elt* c = C + i * c_rs + j * c_cs;
    *c = apply_op<Elt>(*c, s, op, mc.p);
}

// ---- host-side launch selection ----
template<class Elt>
uint32_t u32_flush_interval(const ModCtx<Elt>& ctx, int BK, bool& permac) {
    // kb <= min(k_max, 256) products per accumulator between reductions
    const uint64_t kmax = std::min<uint64_t>(ctx.k_max, 256);
    if (kmax >= static_cast<uint64_t>(BK)) { permac = false; return static_cast<uint32_t>(kmax / BK); }
    permac = true;
    return static_cast<uint32_t>(kmax);
}

template<class Elt, int BM, int BN, int BK, int TM, int TN, bool A_KC, bool B_NC>
void launch_gemm_cfg(GemmParams<Elt>& P, uint32_t zdim, bool permac, cudaStream_t s) {
    constexpr int NT = (BM / TM) * (BN / TN);
    const dim3 grid((P.n + BN - 1) / BN, (P.m + BM - 1) / BM, zdim);
    ENGINE_CHECK_MSG(grid.y <= 65535 && grid.z <= 65535, "gemm: grid too large");
    // The u64 path never uses PERMAC (launch_gemm passes false): instantiate that kernel for 32-bit words only (CL-4 / C-01,
    // 8 never-launched u64 kernels, about 484 KB of SASS).
    if constexpr (sizeof(Elt) == 4) {
        if (permac) gemm_kernel<Elt, BM, BN, BK, TM, TN, A_KC, B_NC, true><<<grid, NT, 0, s>>>(P);
        else        gemm_kernel<Elt, BM, BN, BK, TM, TN, A_KC, B_NC, false><<<grid, NT, 0, s>>>(P);
    } else {
        ENGINE_CHECK_MSG(!permac, "gemm: PERMAC is a 32-bit-word path");
        gemm_kernel<Elt, BM, BN, BK, TM, TN, A_KC, B_NC, false><<<grid, NT, 0, s>>>(P);
    }
    after_launch("gemm_kernel");
}

template<class Elt, int BM, int BN, int BK, int TM, int TN>
void launch_gemm_layout(GemmParams<Elt>& P, uint32_t zdim, bool permac, cudaStream_t s) {
    const bool akc = (P.a_cs == 1), bnc = (P.b_cs == 1);
    if (akc && bnc)       launch_gemm_cfg<Elt, BM, BN, BK, TM, TN, true,  true >(P, zdim, permac, s);
    else if (akc)         launch_gemm_cfg<Elt, BM, BN, BK, TM, TN, true,  false>(P, zdim, permac, s);
    else if (bnc)         launch_gemm_cfg<Elt, BM, BN, BK, TM, TN, false, true >(P, zdim, permac, s);
    else                  launch_gemm_cfg<Elt, BM, BN, BK, TM, TN, false, false>(P, zdim, permac, s);
}

int pick_tile(uint32_t m, uint32_t n, int requested) {
    if (requested == 32 || requested == 64 || requested == 128) return requested;
    if (m <= 64 || n <= 64) return 32;
    if (m >= 256 && n >= 256) return 128;
    return 64;
}

template<class Elt>
void launch_gemm(const ModCtx<Elt>& ctx, GemmParams<Elt>& P, uint32_t zdim, int tile, cudaStream_t s) {
    P.mc = consts_of(ctx);
    g_stats.gemm_launches.fetch_add(1, std::memory_order_relaxed);
    if constexpr (sizeof(Elt) == 4) {
        bool permac = false;
        if (tile == 32) {
            P.kb = u32_flush_interval(ctx, 16, permac);
            launch_gemm_layout<Elt, 32, 32, 16, 2, 2>(P, zdim, permac, s);
        } else if (tile == 128) {
            P.kb = u32_flush_interval(ctx, 16, permac);
            launch_gemm_layout<Elt, 128, 64, 16, 8, 4>(P, zdim, permac, s);
        } else {
            P.kb = u32_flush_interval(ctx, 16, permac);
            launch_gemm_layout<Elt, 64, 64, 16, 4, 4>(P, zdim, permac, s);
        }
    } else {
        P.kb = 0;
        if (tile == 32) launch_gemm_layout<Elt, 32, 32, 8, 2, 2>(P, zdim, false, s);
        else            launch_gemm_layout<Elt, 64, 64, 8, 4, 4>(P, zdim, false, s);
    }
}

template<class Elt>
void check_gemm_operands(DeviceView<const Elt> A, DeviceView<const Elt> B, DeviceView<Elt> C,
                         const char* who) {
    ENGINE_CHECK_MSG(A.cols == B.rows && C.rows == A.rows && C.cols == B.cols,
                     (std::string(who) + ": shape mismatch").c_str());
    ENGINE_CHECK_MSG(A.invariant == Invariant::Canonical && B.invariant == Invariant::Canonical,
                     (std::string(who) + ": operands must be Canonical").c_str());
    ENGINE_CHECK_MSG(C.invariant == Invariant::Canonical, (std::string(who) + ": C must be Canonical").c_str());
    ENGINE_CHECK_MSG(!windows_overlap(C, A) && !windows_overlap(C, B), (std::string(who) + ": C aliases an operand").c_str());
}

// ---------------------------------------------------------------------------
// (slot-22 G1) the IMMA route: u32 canonical -> balanced int8 operands in the
// caller's workspace, then the int8 tensor-core GEMM with the canonical u32
// epilogue (ImmaKernels.cuh; ENGINE_API.md Sec. 11.8). Exact integer arithmetic.
// ---------------------------------------------------------------------------
constexpr uint32_t kImmaMaxRows = 65535u * 32u;   // the conversion grid's y limit (32-row tiles)

// The k-loop reduction interval: 0 (never) when the whole padded K fits the int32
// bound, T(p) otherwise; an explicit request must not exceed T(p).
uint32_t imma_flush_tiles_for(uint32_t p, uint32_t kp, uint32_t requested) {
    const uint32_t T = imma_max_flush_tiles(p);
    if (requested != 0) {
        ENGINE_CHECK_MSG(requested <= T, "cuda::gemm (IMMA): imma_flush_tiles exceeds the int32 bound T(p)");
        return requested;
    }
    const uint32_t ktiles = (kp + imma::BK - 1) / imma::BK;
    return ktiles <= T ? 0u : T;
}

template<class Elt>
bool imma_route(const ModCtx<Elt>& ctx, uint32_t m, uint32_t n, uint32_t k, const GemmOpts& o) {
    if (!std::is_same<Elt, uint32_t>::value) return false;
    if (o.imma == 0 || k == 0 || m == 0 || n == 0 || !imma_eligible(ctx)) return false;
    if (o.imma != 1 && !imma_policy()) return false;
    // a workspace for at least one 64-deep K chunk (imma_k_chunk(); the full K needs no chunking)
    if (o.imma_workspace == nullptr || o.imma_workspace_bytes < imma_workspace_bytes(m, n, std::min<uint32_t>(k, 64))) return false;
    if (m > kImmaMaxRows || n > kImmaMaxRows) return false;
    return imma_supported();
}

void launch_imma_conv(const imma::ConvArgs& a, uint32_t zdim, cudaStream_t s) {
    const dim3 blk(32, 8);
    const dim3 grid((a.ld + 31) / 32, (a.rows + 31) / 32, zdim);
    ENGINE_CHECK_MSG(grid.y <= 65535 && grid.z <= 65535, "cuda::gemm (IMMA): conversion grid too large");
    imma::imma_to_s8_kernel<<<grid, blk, 0, s>>>(a);
    after_launch("imma_to_s8_kernel");
}
void launch_imma_gemm(const imma::GemmArgs& a, uint32_t zdim, cudaStream_t s) {
    const uint64_t tiles = static_cast<uint64_t>((a.M + imma::BM - 1) / imma::BM) * ((a.N + imma::BN - 1) / imma::BN);
    ENGINE_CHECK_MSG(tiles <= 0x7FFFFFFFull && zdim <= 65535, "cuda::gemm (IMMA): grid too large");
    const dim3 grid(static_cast<uint32_t>(tiles), zdim);
    imma::imma_gemm_kernel<<<grid, imma::NTHREADS, imma::SMEM_BYTES, s>>>(a);
    after_launch("imma_gemm_kernel");
    g_stats.gemm_launches.fetch_add(1, std::memory_order_relaxed);
    g_stats.imma_launches.fetch_add(1, std::memory_order_relaxed);
}

// C op= A * B on the IMMA route (imma_route() said yes). When the workspace holds less than the
// whole K, K is cut into chunks of imma_k_chunk(m, n, k, bytes) (a multiple of 64): the first chunk
// applies op, every further one Add (op Set / Add) or Sub (op Sub) -- the exact mod-p sum of the
// chunks' canonical products, so the bytes are those of the one-shot product.
void imma_gemm_u32(const ModCtx<uint32_t>& ctx, DeviceView<const uint32_t> A, DeviceView<const uint32_t> B,
                   DeviceView<uint32_t> C, int op, const GemmOpts& o, cudaStream_t s) {
    const uint32_t m = C.rows, n = C.cols, k = A.cols, p = static_cast<uint32_t>(ctx.p);
    const uint32_t kc = imma_k_chunk(m, n, k, o.imma_workspace_bytes);
    ENGINE_CHECK_MSG(kc > 0, "cuda::gemm (IMMA): workspace below one K chunk");
    for (uint32_t k0 = 0; k0 < k; k0 += kc) {
        const uint32_t kk = std::min(kc, k - k0), kp = imma_padded_k(kk);
        const DeviceView<const uint32_t> Ac = A.sub(0, m, k0, kk), Bc = B.sub(k0, kk, 0, n);
        int8_t* A8 = static_cast<int8_t*>(o.imma_workspace);
        int8_t* B8 = A8 + imma_b_offset(m, kk);
        imma::ConvArgs ca{};
        ca.src = Ac.base; ca.rs = Ac.rs; ca.cs = Ac.cs; ca.rows = m; ca.cols = kk; ca.dst = A8; ca.ld = kp; ca.p = p;
        launch_imma_conv(ca, 1, s);
        imma::ConvArgs cb{};   // B^T: n x kk
        cb.src = Bc.base; cb.rs = Bc.cs; cb.cs = Bc.rs; cb.rows = n; cb.cols = kk; cb.dst = B8; cb.ld = kp; cb.p = p;
        launch_imma_conv(cb, 1, s);
        imma::GemmArgs ga{};
        ga.M = static_cast<int>(m); ga.N = static_cast<int>(n); ga.K = static_cast<int>(kp);
        ga.A = A8; ga.Bt = B8; ga.lda = static_cast<int>(kp); ga.ldb = static_cast<int>(kp);
        ga.C = C.base; ga.c_rs = C.rs; ga.c_cs = C.cs;
        ga.op = (k0 == 0) ? op : (op == 2 ? 2 : 1);
        ga.p = static_cast<int32_t>(p);
        ga.flush_tiles = imma_flush_tiles_for(p, kp, o.imma_flush_tiles);
        launch_imma_gemm(ga, 1, s);
    }
}

template<class Elt>
void gemm_impl(const ModCtx<Elt>& ctx, DeviceView<const Elt> A, DeviceView<const Elt> B,
               DeviceView<Elt> C, GemmOp op, Stream s, const GemmOpts& opts, const char* who) {
    check_gemm_operands(A, B, C, who);
    if (C.empty()) return;
    const uint32_t m = C.rows, n = C.cols, k = A.cols;
    GemmParams<Elt> P{};
    P.A = A.base; P.B = B.base; P.C = C.base;
    P.a_rs = A.rs; P.a_cs = A.cs; P.b_rs = B.rs; P.b_cs = B.cs; P.c_rs = C.rs; P.c_cs = C.cs;
    P.m = m; P.n = n; P.k = k;
    P.op = (op == GemmOp::Set) ? 0 : (op == GemmOp::Add) ? 1 : 2;
    P.batch = nullptr; P.batch_base = 0;
    const int tile = pick_tile(m, n, opts.tile);
    uint32_t splits = std::max<uint32_t>(1, opts.splits);
    if (k == 0) {
        // A*B is the zero matrix: Set writes zeros, Add/Sub leave C
        if (op == GemmOp::Set) {
            P.k_per_split = 0; P.ws = nullptr;
            launch_gemm(ctx, P, 1, tile, S(s));
        }
        return;
    }
    if constexpr (std::is_same<Elt, uint32_t>::value) {
        if (imma_route(ctx, m, n, k, opts)) { imma_gemm_u32(ctx, A, B, C, P.op, opts, S(s)); return; }
    }
    if (splits > 1) {
        const uint32_t kps = (k + splits - 1) / splits;
        splits = (k + kps - 1) / kps;
        if (splits > 1) {
            ENGINE_CHECK_MSG(opts.workspace != nullptr &&
                             opts.workspace_bytes >= gemm_workspace_bytes(m, n, splits, sizeof(Elt)),
                             (std::string(who) + ": split-K needs a workspace of gemm_workspace_bytes()").c_str());
            P.k_per_split = kps;
            P.ws = static_cast<Elt*>(opts.workspace);
            launch_gemm(ctx, P, splits, tile, S(s));
            const dim3 blk(32, 8);
            const dim3 grid((n + 31) / 32, (m + 7) / 8);
            ENGINE_CHECK_MSG(grid.y <= 65535, "gemm: split-K combine grid too large");
            splitk_combine_kernel<Elt><<<grid, blk, 0, S(s)>>>(P.ws, splits, m, n, C.base, C.rs, C.cs,
                                                                P.op, P.mc);
            after_launch("splitk_combine_kernel");
            return;
        }
    }
    P.k_per_split = 0; P.ws = nullptr;
    launch_gemm(ctx, P, 1, tile, S(s));
}

// ---------------------------------------------------------------------------
// GEMV
// ---------------------------------------------------------------------------
template<class Elt>
struct GemvParams {
    const Elt* A; const Elt* x; Elt* y;
    size_t a_rs, a_cs, x_rs, y_rs;
    uint32_t m, k;
    int op;
    uint32_t kb;   // products per accumulator between reductions (u32)
    ModConsts mc;
};

// warp per row: lanes stride over k (A k-contiguous -> coalesced)
template<class Elt>
__global__ void gemv_warp_kernel(GemvParams<Elt> P) {
    const int lane = threadIdx.x & 31;
    const uint32_t row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= P.m) return;
    const Elt* a = P.A + row * P.a_rs;
    Acc<Elt> acc;
    uint32_t cnt = 0;
    for (uint32_t kk = lane; kk < P.k; kk += 32) {
        if constexpr (sizeof(Elt) == 4) {
            acc.mac(a[kk * P.a_cs], P.x[kk * P.x_rs]);
            if (++cnt == P.kb) { cnt = 0; acc.flush(P.mc); }
        } else {
            acc.mac(a[kk * P.a_cs], P.x[kk * P.x_rs], P.mc);
        }
    }
    Elt v = acc.result(P.mc);
    // warp reduction of canonical partials mod p (exact mod-p sum: order-free)
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        Elt o;
        if constexpr (sizeof(Elt) == 4) {
            o = __shfl_xor_sync(0xffffffffu, v, off);
            v = static_cast<Elt>(add32(static_cast<uint32_t>(v), static_cast<uint32_t>(o), static_cast<uint32_t>(P.mc.p)));
        } else {
            o = __shfl_xor_sync(0xffffffffu, v, off);
            v = static_cast<Elt>(add64(v, o, P.mc.p));
        }
    }
    if (lane == 0) {
        Elt* yp = P.y + row * P.y_rs;
        *yp = apply_op<Elt>(*yp, v, P.op, P.mc.p);
    }
}

// thread per row (A row-stride 1 -> coalesced across rows)
template<class Elt>
__global__ void gemv_thread_kernel(GemvParams<Elt> P) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= P.m) return;
    const Elt* a = P.A + row * P.a_rs;
    Acc<Elt> acc;
    uint32_t cnt = 0;
    for (uint32_t kk = 0; kk < P.k; ++kk) {
        if constexpr (sizeof(Elt) == 4) {
            acc.mac(a[kk * P.a_cs], P.x[kk * P.x_rs]);
            if (++cnt == P.kb) { cnt = 0; acc.flush(P.mc); }
        } else {
            acc.mac(a[kk * P.a_cs], P.x[kk * P.x_rs], P.mc);
        }
    }
    const Elt v = acc.result(P.mc);
    Elt* yp = P.y + row * P.y_rs;
    *yp = apply_op<Elt>(*yp, v, P.op, P.mc.p);
}

// ---------------------------------------------------------------------------
// elementwise kernels over a 2-D window (fast dim = stride-1 when possible)
// ---------------------------------------------------------------------------
struct Win {
    uint32_t nf, ns;      // extent along the fast / slow dimension
    size_t   sf, ss;      // strides
};
template<class Elt>
Win win_of(const DeviceView<Elt>& v, bool& swapped) {
    // choose the fast dimension so that consecutive threads touch consecutive addresses
    swapped = (v.cs != 1 && v.rs == 1);
    if (swapped) return Win{v.rows, v.cols, v.rs, v.cs};
    return Win{v.cols, v.rows, v.cs, v.rs};
}
inline dim3 ew_grid(const Win& w) {
    return dim3((w.nf + 31) / 32, std::min<uint32_t>((w.ns + 7) / 8, 65535));
}

template<class Elt>
__global__ void axpy_kernel(Elt alpha_form, const Elt* x, Win wx, Elt* y, Win wy, int subtract, ModConsts mc) {
    const uint32_t f = blockIdx.x * 32 + threadIdx.x;
    if (f >= wx.nf) return;
    for (uint32_t sl = blockIdx.y * 8 + threadIdx.y; sl < wx.ns; sl += gridDim.y * 8) {
        const Elt xv = x[f * wx.sf + sl * wx.ss];
        Elt* yp = y + f * wy.sf + sl * wy.ss;
        Elt t;
        if constexpr (sizeof(Elt) == 4) t = red32(static_cast<uint64_t>(alpha_form) * xv, static_cast<uint32_t>(mc.p), mc.barrett_m);
        else t = mc.mont ? mont_mul(alpha_form, xv, mc.p, mc.ninv)     // alpha_form = alpha*R
                         : mulmod64_wide(alpha_form, xv, mc.p);
        *yp = apply_op<Elt>(*yp, t, subtract ? 2 : 1, mc.p);
    }
}
template<class Elt>
__global__ void scal_kernel(Elt alpha_form, Elt* x, Win wx, ModConsts mc) {
    const uint32_t f = blockIdx.x * 32 + threadIdx.x;
    if (f >= wx.nf) return;
    for (uint32_t sl = blockIdx.y * 8 + threadIdx.y; sl < wx.ns; sl += gridDim.y * 8) {
        Elt* xp = x + f * wx.sf + sl * wx.ss;
        if constexpr (sizeof(Elt) == 4) *xp = red32(static_cast<uint64_t>(alpha_form) * *xp, static_cast<uint32_t>(mc.p), mc.barrett_m);
        else *xp = mc.mont ? mont_mul(alpha_form, *xp, mc.p, mc.ninv) : mulmod64_wide(alpha_form, *xp, mc.p);
    }
}
template<class Elt>
__global__ void canon_kernel(Elt* x, Win wx, Elt p) {
    const uint32_t f = blockIdx.x * 32 + threadIdx.x;
    if (f >= wx.nf) return;
    for (uint32_t sl = blockIdx.y * 8 + threadIdx.y; sl < wx.ns; sl += gridDim.y * 8) {
        Elt* xp = x + f * wx.sf + sl * wx.ss;
        const Elt v = *xp;
        *xp = (v >= p) ? v - p : v;
    }
}
__global__ void canon64_kernel(const uint64_t* acc, Win wa, uint32_t* dst, Win wd, uint32_t p, uint64_t m) {
    const uint32_t f = blockIdx.x * 32 + threadIdx.x;
    if (f >= wa.nf) return;
    for (uint32_t sl = blockIdx.y * 8 + threadIdx.y; sl < wa.ns; sl += gridDim.y * 8)
        dst[f * wd.sf + sl * wd.ss] = red32(acc[f * wa.sf + sl * wa.ss], p, m);
}
template<class Elt>
__global__ void copy_kernel(const Elt* src, Win ws, Elt* dst, Win wd) {
    const uint32_t f = blockIdx.x * 32 + threadIdx.x;
    if (f >= ws.nf) return;
    for (uint32_t sl = blockIdx.y * 8 + threadIdx.y; sl < ws.ns; sl += gridDim.y * 8)
        dst[f * wd.sf + sl * wd.ss] = src[f * ws.sf + sl * ws.ss];
}

// ---- tape support kernels (Phase 5 Stage B) ----
template<class Elt>
__global__ void fill_kernel(Elt* x, Win wx, Elt value) {
    const uint32_t f = blockIdx.x * 32 + threadIdx.x;
    if (f >= wx.nf) return;
    for (uint32_t sl = blockIdx.y * 8 + threadIdx.y; sl < wx.ns; sl += gridDim.y * 8)
        x[f * wx.sf + sl * wx.ss] = value;
}
template<class Elt>
__global__ void axpy_identity_kernel(Elt alpha, Elt* y, size_t rs, size_t cs, uint32_t n, int subtract, uint64_t p) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Elt* yp = y + i * rs + i * cs;
    *yp = apply_op<Elt>(*yp, alpha, subtract ? 2 : 1, p);
}
// c(i, t) [op]= alpha * [row_off + r(i) == col(t)] + beta * a(r(i), col(t)), r(i) = row_list[i] or i;
// thread per (t fast, i slow)
template<class Elt>
__global__ void gather_kernel(Elt alpha, Elt beta_form, const Elt* a, size_t a_rs, size_t a_cs, uint32_t row_off,
                              const uint32_t* cols, Elt* c, size_t c_rs, size_t c_cs, uint32_t rows, uint32_t ncols,
                              int accumulate, ModConsts mc, const uint32_t* row_list) {
    const uint32_t t = blockIdx.x * 32 + threadIdx.x;
    if (t >= ncols) return;
    const uint32_t col = cols ? cols[t] : t;
    for (uint32_t i = blockIdx.y * 8 + threadIdx.y; i < rows; i += gridDim.y * 8) {
        const uint32_t ar = row_list ? row_list[i] : i;
        Elt v = 0;
        if (a != nullptr && beta_form != 0) {
            const Elt av = a[static_cast<size_t>(ar) * a_rs + col * a_cs];
            if constexpr (sizeof(Elt) == 4) v = red32(static_cast<uint64_t>(beta_form) * av, static_cast<uint32_t>(mc.p), mc.barrett_m);
            else v = mc.mont ? mont_mul(beta_form, av, mc.p, mc.ninv) : mulmod64_wide(beta_form, av, mc.p);
        }
        if (alpha != 0 && row_off + ar == col) v = apply_op<Elt>(v, alpha, 1, mc.p);
        Elt* cp = c + i * c_rs + t * c_cs;
        *cp = accumulate ? apply_op<Elt>(*cp, v, 1, mc.p) : v;
    }
}
// dst(i, j) = src(j, i): 32 x 32 tiles through shared memory
template<class Elt>
__global__ void transpose_kernel(const Elt* src, size_t srs, size_t scs, Elt* dst, size_t drs, size_t dcs,
                                 uint32_t rows_dst, uint32_t cols_dst) {
    __shared__ Elt tile[32][33];
    const uint32_t i0 = blockIdx.y * 32, j0 = blockIdx.x * 32;   // dst tile
    // read src rows j0.., cols i0..  (src is cols_dst x rows_dst)
    for (int r = 0; r < 32; r += 8) {
        const uint32_t jj = j0 + threadIdx.y + r, ii = i0 + threadIdx.x;
        if (jj < cols_dst && ii < rows_dst) tile[threadIdx.y + r][threadIdx.x] = src[jj * srs + ii * scs];
    }
    __syncthreads();
    for (int r = 0; r < 32; r += 8) {
        const uint32_t ii = i0 + threadIdx.y + r, jj = j0 + threadIdx.x;
        if (ii < rows_dst && jj < cols_dst) dst[ii * drs + jj * dcs] = tile[threadIdx.x][threadIdx.y + r];
    }
}

// ---------------------------------------------------------------------------
// TRSM diagonal block: X = T^{-1} B for one nb x nb unit-triangular block,
// one thread per column of B; the trailing update is a gemm(Sub).
// ---------------------------------------------------------------------------
template<class Elt, bool LOWER>
__global__ void trsm_diag_kernel(const Elt* T, size_t t_rs, size_t t_cs, Elt* B, size_t b_rs, size_t b_cs,
                                 uint32_t nb, uint32_t n, uint32_t kb, ModConsts mc) {
    __shared__ Elt Ts[32][33];
    __shared__ Elt xs[32][128];
    for (int idx = threadIdx.x; idx < 32 * 32; idx += blockDim.x) {
        const int i = idx / 32, t = idx % 32;
        Ts[i][t] = (static_cast<uint32_t>(i) < nb && static_cast<uint32_t>(t) < nb) ? T[i * t_rs + t * t_cs] : Elt(0);
    }
    __syncthreads();
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    Elt* bj = B + j * b_cs;
    if (LOWER) {
        for (uint32_t i = 0; i < nb; ++i) {
            Acc<Elt> acc; uint32_t cnt = 0;
            for (uint32_t t = 0; t < i; ++t) {
                if constexpr (sizeof(Elt) == 4) {
                    acc.mac(Ts[i][t], xs[t][threadIdx.x]);
                    if (++cnt == kb) { cnt = 0; acc.flush(mc); }
                } else acc.mac(Ts[i][t], xs[t][threadIdx.x], mc);
            }
            const Elt sum = acc.result(mc);
            const Elt v = apply_op<Elt>(bj[i * b_rs], sum, 2, mc.p);
            xs[i][threadIdx.x] = v;
            bj[i * b_rs] = v;
        }
    } else {
        for (int i = static_cast<int>(nb) - 1; i >= 0; --i) {
            Acc<Elt> acc; uint32_t cnt = 0;
            for (uint32_t t = i + 1; t < nb; ++t) {
                if constexpr (sizeof(Elt) == 4) {
                    acc.mac(Ts[i][t], xs[t][threadIdx.x]);
                    if (++cnt == kb) { cnt = 0; acc.flush(mc); }
                } else acc.mac(Ts[i][t], xs[t][threadIdx.x], mc);
            }
            const Elt sum = acc.result(mc);
            const Elt v = apply_op<Elt>(bj[i * b_rs], sum, 2, mc.p);
            xs[i][threadIdx.x] = v;
            bj[i * b_rs] = v;
        }
    }
}

} // namespace

// ===========================================================================
// public API
// ===========================================================================

size_t gemm_workspace_bytes(uint32_t m, uint32_t n, uint32_t splits, size_t elt_bytes) {
    return static_cast<size_t>(std::max<uint32_t>(splits, 1)) * m * n * elt_bytes;
}

uint32_t suggest_splits(uint32_t m, uint32_t n, uint32_t k) {
    // fewer than ~2 waves of 64x64 tiles on 70 SMs and a deep k: split
    const uint64_t tiles = static_cast<uint64_t>((m + 63) / 64) * ((n + 63) / 64);
    if (tiles >= 140 || k < 512) return 1;
    uint32_t s = static_cast<uint32_t>((140 + tiles - 1) / tiles);
    s = std::min<uint32_t>(s, std::max<uint32_t>(1, k / 256));
    return std::max<uint32_t>(1, std::min<uint32_t>(s, 16));
}

bool imma_policy() {
    // re-read per call: the validations switch it per arm. Off iff the value STARTS WITH '0'
    // ("0", "00", "0x", ...); unset, empty or anything else = on (ENGINE_API.md §11.8, audit R2).
    // (S24 CL-R2) A retired lever (RetiredDeviceLevers.h, shared with the stub): ON build as
    // before; OFF build always on, `0...` refused.
    return engine::lever::gpu_imma_lever().enabled();
}

bool imma_supported() {
    // Probed once: a usable device of compute capability >= 8.0 and an sm_80+ image of the
    // kernel (SASS or PTX: both the binary and the PTX version must be >= 8.0, since a
    // JIT-compiled pre-sm_80 image would carry the __trap() body). Also raises the
    // kernel's dynamic shared-memory limit (61 440 B > the 48 KB default).
    static const bool ok = [] {
        if (!available()) return false;
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        if (major < 8) return false;
        cudaFuncAttributes fa{};
        if (cudaFuncGetAttributes(&fa, imma::imma_gemm_kernel) != cudaSuccess) { (void)cudaGetLastError(); return false; }
        if (fa.binaryVersion < 80 || fa.ptxVersion < 80) return false;
        if (cudaFuncSetAttribute(imma::imma_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(imma::SMEM_BYTES)) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        return true;
    }();
    return ok;
}

template<class Elt>
void gemm(const ModCtx<Elt>& ctx, DeviceView<const Elt> A, DeviceView<const Elt> B,
          DeviceView<Elt> C, GemmOp op, Stream s, const GemmOpts& opts) {
    gemm_impl<Elt>(ctx, A, B, C, op, s, opts, "cuda::gemm");
}
template<class Elt>
void gemm_nt(const ModCtx<Elt>& ctx, DeviceView<const Elt> A, DeviceView<const Elt> B,
             DeviceView<Elt> C, GemmOp op, Stream s, const GemmOpts& opts) {
    gemm_impl<Elt>(ctx, A, B.transposed_view(), C, op, s, opts, "cuda::gemm_nt");
}
template<class Elt>
void schur_update(const ModCtx<Elt>& ctx, DeviceView<const Elt> L, DeviceView<const Elt> U,
                  DeviceView<Elt> C, Stream s, const GemmOpts& opts) {
    gemm_impl<Elt>(ctx, L, U, C, GemmOp::Sub, s, opts, "cuda::schur_update");
}

// ---- batched ----
template<class Elt>
GemmBatch<Elt>::~GemmBatch() {
    if (d_entries_) CudaCtx::get().dfree(d_entries_, d_capacity_ * sizeof(GemmBatchEntry<Elt>));
    if (imma_ws_) CudaCtx::get().dfree(imma_ws_, imma_ws_bytes_);
}
template<class Elt>
void GemmBatch<Elt>::set_shape(uint32_t m, uint32_t n, uint32_t k, size_t a_rs, size_t a_cs,
                               size_t b_rs, size_t b_cs, size_t c_rs, size_t c_cs) {
    m_ = m; n_ = n; k_ = k;
    a_rs_ = a_rs; a_cs_ = a_cs; b_rs_ = b_rs; b_cs_ = b_cs; c_rs_ = c_rs; c_cs_ = c_cs;
    shape_set_ = true;
}
template<class Elt>
void GemmBatch<Elt>::set_shape(DeviceView<const Elt> A, DeviceView<const Elt> B, DeviceView<Elt> C) {
    check_gemm_operands(A, B, C, "GemmBatch::set_shape");
    set_shape(C.rows, C.cols, A.cols, A.rs, A.cs, B.rs, B.cs, C.rs, C.cs);
}
template<class Elt>
void GemmBatch<Elt>::push(DeviceView<const Elt> A, DeviceView<const Elt> B, DeviceView<Elt> C) {
    ENGINE_CHECK_MSG(shape_set_, "GemmBatch::push: set_shape first");
    check_gemm_operands(A, B, C, "GemmBatch::push");
    ENGINE_CHECK_MSG(C.rows == m_ && C.cols == n_ && A.cols == k_ &&
                     A.rs == a_rs_ && A.cs == a_cs_ && B.rs == b_rs_ && B.cs == b_cs_ &&
                     C.rs == c_rs_ && C.cs == c_cs_, "GemmBatch::push: instance shape/strides differ from the batch");
    entries_.push_back(GemmBatchEntry<Elt>{A.base, B.base, C.base});
}
template<class Elt>
void GemmBatch<Elt>::upload(Stream s) {
    if (entries_.empty()) { uploaded_ = 0; return; }
    CudaCtx& cx = CudaCtx::get();
    if (d_capacity_ < entries_.size()) {
        if (d_entries_) cx.dfree(d_entries_, d_capacity_ * sizeof(GemmBatchEntry<Elt>));
        d_capacity_ = entries_.size();
        d_entries_ = static_cast<GemmBatchEntry<Elt>*>(cx.dalloc(d_capacity_ * sizeof(GemmBatchEntry<Elt>)));
    }
    cx.upload(entries_.data(), d_entries_, entries_.size() * sizeof(GemmBatchEntry<Elt>), s);
    uploaded_ = entries_.size();
}
template<class Elt>
bool GemmBatch<Elt>::prepare_imma(const ModCtx<Elt>& ctx) {
    if constexpr (!std::is_same<Elt, uint32_t>::value) { (void)ctx; return false; } else {
    ENGINE_CHECK_MSG(uploaded_ == entries_.size(), "GemmBatch::prepare_imma: upload() the instance table first");
    if (entries_.empty() || !shape_set_ || m_ == 0 || n_ == 0 || k_ == 0) return false;
    if (!imma_eligible(ctx) || !imma_policy() || m_ > kImmaMaxRows || n_ > kImmaMaxRows || !imma_supported()) return false;
    const size_t need = imma_workspace_bytes(m_, n_, k_) * entries_.size();
    if (need > (static_cast<size_t>(1) << 30)) return false;   // the u32 kernel then (no VRAM spike for a huge batch)
    if (imma_ws_bytes_ < need) {
        CudaCtx& cx = CudaCtx::get();
        if (imma_ws_) cx.dfree(imma_ws_, imma_ws_bytes_);
        imma_ws_ = nullptr; imma_ws_bytes_ = 0;
        imma_ws_ = cx.dalloc(need);
        imma_ws_bytes_ = need;
    }
    return true;
    }
}

template<class Elt>
size_t GemmBatch<Elt>::imma_bytes_needed() const {
    return k_ == 0 ? 0 : imma_workspace_bytes(m_, n_, k_) * entries_.size();
}

template<class Elt>
void GemmBatch<Elt>::run(const ModCtx<Elt>& ctx, GemmOp op, Stream s, int tile, int imma,
                         void* ext_ws, size_t ext_ws_bytes) const {
    ENGINE_CHECK_MSG(uploaded_ == entries_.size(), "GemmBatch::run: upload() the instance table first");
    if (entries_.empty() || m_ == 0 || n_ == 0) return;
    if constexpr (std::is_same<Elt, uint32_t>::value) {
        const size_t inst = imma_workspace_bytes(m_, n_, k_);
        void* const ws_ptr = ext_ws != nullptr ? ext_ws : imma_ws_;
        const size_t ws_bytes = ext_ws != nullptr ? ext_ws_bytes : imma_ws_bytes_;
        if (imma != 0 && (imma == 1 || imma_policy()) && k_ > 0 && imma_eligible(ctx) && ws_ptr != nullptr &&
            ws_bytes >= inst * entries_.size() && m_ <= kImmaMaxRows && n_ <= kImmaMaxRows && imma_supported()) {
            const uint32_t p = static_cast<uint32_t>(ctx.p), kp = imma_padded_k(k_);
            const int opi = (op == GemmOp::Set) ? 0 : (op == GemmOp::Add) ? 1 : 2;
            size_t base = 0;
            while (base < entries_.size()) {
                const uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(entries_.size() - base, 65535));
                imma::ConvArgs ca{};
                ca.rs = a_rs_; ca.cs = a_cs_; ca.rows = m_; ca.cols = k_; ca.ld = kp; ca.p = p;
                ca.batch = d_entries_; ca.batch_base = static_cast<uint32_t>(base); ca.which = 0;
                ca.ws = static_cast<int8_t*>(ws_ptr); ca.inst_bytes = inst; ca.dst_off = 0;
                launch_imma_conv(ca, cnt, S(s));
                imma::ConvArgs cb = ca;   // B^T: n x k
                cb.rs = b_cs_; cb.cs = b_rs_; cb.rows = n_; cb.which = 1; cb.dst_off = imma_b_offset(m_, k_);
                launch_imma_conv(cb, cnt, S(s));
                imma::GemmArgs ga{};
                ga.M = static_cast<int>(m_); ga.N = static_cast<int>(n_); ga.K = static_cast<int>(kp);
                ga.lda = static_cast<int>(kp); ga.ldb = static_cast<int>(kp);
                ga.c_rs = c_rs_; ga.c_cs = c_cs_; ga.op = opi; ga.p = static_cast<int32_t>(p);
                ga.flush_tiles = imma_flush_tiles_for(p, kp, 0);
                ga.batch = d_entries_; ga.batch_base = static_cast<uint32_t>(base);
                ga.ws = static_cast<const int8_t*>(ws_ptr); ga.inst_bytes = inst; ga.b_off = imma_b_offset(m_, k_);
                launch_imma_gemm(ga, cnt, S(s));
                g_stats.batched_instances.fetch_add(cnt, std::memory_order_relaxed);
                base += cnt;
            }
            return;
        }
    }
    GemmParams<Elt> P{};
    P.a_rs = a_rs_; P.a_cs = a_cs_; P.b_rs = b_rs_; P.b_cs = b_cs_; P.c_rs = c_rs_; P.c_cs = c_cs_;
    P.m = m_; P.n = n_; P.k = k_;
    P.op = (op == GemmOp::Set) ? 0 : (op == GemmOp::Add) ? 1 : 2;
    P.k_per_split = 0; P.ws = nullptr;
    P.batch = d_entries_;
    const int t = pick_tile(m_, n_, tile);
    size_t base = 0;
    while (base < entries_.size()) {
        const uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(entries_.size() - base, 65535));
        P.batch_base = static_cast<uint32_t>(base);
        launch_gemm(ctx, P, cnt, t, S(s));
        g_stats.batched_instances.fetch_add(cnt, std::memory_order_relaxed);
        base += cnt;
    }
}
template class GemmBatch<uint32_t>;
template class GemmBatch<uint64_t>;

// ---- gemv ----
template<class Elt>
void gemv(const ModCtx<Elt>& ctx, DeviceView<const Elt> A, DeviceView<const Elt> x,
          DeviceView<Elt> y, GemmOp op, Stream s) {
    ENGINE_CHECK_MSG(x.cols == 1 && y.cols == 1, "cuda::gemv: x and y must be column views");
    ENGINE_CHECK_MSG(A.cols == x.rows && A.rows == y.rows, "cuda::gemv: shape mismatch");
    ENGINE_CHECK_MSG(A.invariant == Invariant::Canonical && x.invariant == Invariant::Canonical &&
                     y.invariant == Invariant::Canonical, "cuda::gemv: operands must be Canonical");
    ENGINE_CHECK_MSG(!windows_overlap(y, A) && !windows_overlap(y, x), "cuda::gemv: y aliases an operand");
    if (y.empty()) return;
    GemvParams<Elt> P{};
    P.A = A.base; P.x = x.base; P.y = y.base;
    P.a_rs = A.rs; P.a_cs = A.cs; P.x_rs = x.rs; P.y_rs = y.rs;
    P.m = A.rows; P.k = A.cols;
    P.op = (op == GemmOp::Set) ? 0 : (op == GemmOp::Add) ? 1 : 2;
    P.kb = static_cast<uint32_t>(std::min<uint64_t>(ctx.k_max, 256));
    P.mc = consts_of(ctx);
    if (A.cs == 1 || A.rs != 1) {
        const uint32_t threads = 256, rows_per_block = threads / 32;
        const uint32_t grid = (P.m + rows_per_block - 1) / rows_per_block;
        gemv_warp_kernel<Elt><<<grid, threads, 0, S(s)>>>(P);
    } else {
        const uint32_t threads = 256;
        gemv_thread_kernel<Elt><<<(P.m + threads - 1) / threads, threads, 0, S(s)>>>(P);
    }
    after_launch("gemv_kernel");
}

// ---- axpy / scal / canonicalise / copy / transpose ----
template<class Elt>
Elt alpha_form_of(const ModCtx<Elt>& ctx, Elt alpha) {
    if constexpr (sizeof(Elt) == 8) { if (ctx.has_montgomery) return ctx.mont64_to(alpha); }
    return alpha;
}

template<class Elt>
void axpy(const ModCtx<Elt>& ctx, Elt alpha, DeviceView<const Elt> x, DeviceView<Elt> y, bool subtract, Stream s) {
    ENGINE_CHECK_MSG(x.rows == y.rows && x.cols == y.cols, "cuda::axpy: shape mismatch");
    ENGINE_CHECK_MSG(x.invariant == Invariant::Canonical && y.invariant == Invariant::Canonical,
                     "cuda::axpy: operands must be Canonical");
    ENGINE_CHECK_MSG(alpha < ctx.p, "cuda::axpy: alpha must be canonical");
    if (x.empty()) return;
    bool sw = false;
    const Win wx = win_of(x, sw);
    const Win wy = sw ? Win{y.rows, y.cols, y.rs, y.cs} : Win{y.cols, y.rows, y.cs, y.rs};
    axpy_kernel<Elt><<<ew_grid(wx), dim3(32, 8), 0, S(s)>>>(alpha_form_of(ctx, alpha), x.base, wx, y.base, wy,
                                                             subtract ? 1 : 0, consts_of(ctx));
    after_launch("axpy_kernel");
}
template<class Elt>
void scal(const ModCtx<Elt>& ctx, Elt alpha, DeviceView<Elt> x, Stream s) {
    ENGINE_CHECK_MSG(x.invariant == Invariant::Canonical, "cuda::scal: operand must be Canonical");
    ENGINE_CHECK_MSG(alpha < ctx.p, "cuda::scal: alpha must be canonical");
    if (x.empty()) return;
    bool sw = false;
    const Win wx = win_of(x, sw);
    scal_kernel<Elt><<<ew_grid(wx), dim3(32, 8), 0, S(s)>>>(alpha_form_of(ctx, alpha), x.base, wx, consts_of(ctx));
    after_launch("scal_kernel");
}
template<class Elt>
void canonicalise(const ModCtx<Elt>& ctx, DeviceView<Elt> v, Stream s) {
    ENGINE_CHECK_MSG(v.invariant == Invariant::Lazy2p || v.invariant == Invariant::Canonical,
                     "cuda::canonicalise: view must be Lazy2p or Canonical");
    if (v.empty()) return;
    bool sw = false;
    const Win w = win_of(v, sw);
    canon_kernel<Elt><<<ew_grid(w), dim3(32, 8), 0, S(s)>>>(v.base, w, ctx.p);
    after_launch("canon_kernel");
}
void canonicalise(const ModCtx<uint32_t>& ctx, DeviceView<const uint64_t> acc, DeviceView<uint32_t> dst, Stream s) {
    ENGINE_CHECK_MSG(acc.rows == dst.rows && acc.cols == dst.cols, "cuda::canonicalise(acc64): shape mismatch");
    if (acc.empty()) return;
    bool sw = false;
    const Win wa = win_of(acc, sw);
    const Win wd = sw ? Win{dst.rows, dst.cols, dst.rs, dst.cs} : Win{dst.cols, dst.rows, dst.cs, dst.rs};
    canon64_kernel<<<ew_grid(wa), dim3(32, 8), 0, S(s)>>>(acc.base, wa, dst.base, wd, ctx.p, ctx.barrett_m);
    after_launch("canon64_kernel");
}
template<class Elt>
void copy(DeviceView<const Elt> src, DeviceView<Elt> dst, Stream s) {
    ENGINE_CHECK_MSG(src.rows == dst.rows && src.cols == dst.cols, "cuda::copy: shape mismatch");
    if (src.empty()) return;
    if (src.base == dst.base && src.rs == dst.rs && src.cs == dst.cs) return;
    ENGINE_CHECK_MSG(!windows_overlap(src, dst), "cuda::copy: overlapping windows");
    bool sw = false;
    const Win ws = win_of(src, sw);
    const Win wd = sw ? Win{dst.rows, dst.cols, dst.rs, dst.cs} : Win{dst.cols, dst.rows, dst.cs, dst.rs};
    copy_kernel<Elt><<<ew_grid(ws), dim3(32, 8), 0, S(s)>>>(src.base, ws, dst.base, wd);
    after_launch("copy_kernel");
}
template<class Elt>
void transpose(DeviceView<const Elt> src, DeviceView<Elt> dst, Stream s) {
    ENGINE_CHECK_MSG(src.rows == dst.cols && src.cols == dst.rows, "cuda::transpose: shape mismatch");
    if (src.empty()) return;
    ENGINE_CHECK_MSG(!windows_overlap(src, dst), "cuda::transpose: overlapping windows");
    const dim3 grid((dst.cols + 31) / 32, (dst.rows + 31) / 32);
    ENGINE_CHECK_MSG(grid.y <= 65535, "cuda::transpose: grid too large");
    transpose_kernel<Elt><<<grid, dim3(32, 8), 0, S(s)>>>(src.base, src.rs, src.cs, dst.base, dst.rs, dst.cs,
                                                          dst.rows, dst.cols);
    after_launch("transpose_kernel");
}


template<class Elt>
void fill(DeviceView<Elt> x, Elt value, Stream s) {
    if (x.empty()) return;
    bool sw = false;
    const Win wx = win_of(x, sw);
    fill_kernel<Elt><<<ew_grid(wx), dim3(32, 8), 0, S(s)>>>(x.base, wx, value);
    after_launch("fill_kernel");
}
template<class Elt>
void axpy_identity(const ModCtx<Elt>& ctx, Elt alpha, DeviceView<Elt> y, bool subtract, Stream s) {
    ENGINE_CHECK_MSG(y.invariant == Invariant::Canonical, "cuda::axpy_identity: operand must be Canonical");
    ENGINE_CHECK_MSG(alpha < ctx.p, "cuda::axpy_identity: alpha must be canonical");
    const uint32_t n = std::min(y.rows, y.cols);
    if (n == 0) return;
    axpy_identity_kernel<Elt><<<(n + 255) / 256, 256, 0, S(s)>>>(alpha, y.base, y.rs, y.cs, n, subtract ? 1 : 0, ctx.p);
    after_launch("axpy_identity_kernel");
}
template<class Elt>
void gather(const ModCtx<Elt>& ctx, Elt alpha, Elt beta, DeviceView<const Elt> a, uint32_t row_off,
            const uint32_t* cols, DeviceView<Elt> c, bool accumulate, Stream s, const uint32_t* rows) {
    ENGINE_CHECK_MSG(c.invariant == Invariant::Canonical, "cuda::gather: output must be Canonical");
    ENGINE_CHECK_MSG(alpha < ctx.p && beta < ctx.p, "cuda::gather: coefficients must be canonical");
    if (c.empty()) return;
    const bool has_a = (a.base != nullptr && beta != 0);
    if (has_a) {
        ENGINE_CHECK_MSG(rows != nullptr || a.rows == c.rows, "cuda::gather: row count mismatch");
        ENGINE_CHECK_MSG(cols != nullptr || a.cols == c.cols, "cuda::gather: column count mismatch");
        ENGINE_CHECK_MSG(!windows_overlap(a, c), "cuda::gather: output aliases the source");
    }
    const dim3 grid((c.cols + 31) / 32, std::min<uint32_t>((c.rows + 7) / 8, 65535));
    gather_kernel<Elt><<<grid, dim3(32, 8), 0, S(s)>>>(alpha, has_a ? alpha_form_of(ctx, beta) : Elt(0),
                                                        has_a ? a.base : nullptr, a.rs, a.cs, row_off, cols,
                                                        c.base, c.rs, c.cs, c.rows, c.cols, accumulate ? 1 : 0, consts_of(ctx), rows);
    after_launch("gather_kernel");
}
// ---- trsm ----
template<class Elt>
void trsm_unit(const ModCtx<Elt>& ctx, DeviceView<const Elt> T, DeviceView<Elt> B, Stream s, const GemmOpts& opts) {
    const uint32_t m = T.rows, n = B.cols;
    ENGINE_CHECK_MSG(T.cols == m && B.rows == m, "cuda::trsm_unit: shape mismatch");
    ENGINE_CHECK_MSG(T.structure == Structure::UpperUnitri || T.structure == Structure::LowerUnitri,
                     "cuda::trsm_unit: T must be tagged UpperUnitri or LowerUnitri");
    ENGINE_CHECK_MSG(T.invariant == Invariant::Canonical && B.invariant == Invariant::Canonical,
                     "cuda::trsm_unit: operands must be Canonical");
    ENGINE_CHECK_MSG(!windows_overlap(T, B), "cuda::trsm_unit: T aliases B");
    if (m == 0 || n == 0) return;
    const bool lower = (T.structure == Structure::LowerUnitri);
    const uint32_t NB = 32;
    const uint32_t kb = static_cast<uint32_t>(std::min<uint64_t>(ctx.k_max, 256));
    const ModConsts mc = consts_of(ctx);
    const uint32_t nblocks = (m + NB - 1) / NB;
    for (uint32_t bi = 0; bi < nblocks; ++bi) {
        const uint32_t i0 = lower ? bi * NB : (nblocks - 1 - bi) * NB;
        const uint32_t nb = std::min(NB, m - i0);
        const Elt* Td = T.ptr(i0, i0);
        Elt* Bd = B.ptr(i0, 0);
        const uint32_t threads = 128;
        const dim3 grid((n + threads - 1) / threads);
        if (lower) trsm_diag_kernel<Elt, true ><<<grid, threads, 0, S(s)>>>(Td, T.rs, T.cs, Bd, B.rs, B.cs, nb, n, kb, mc);
        else       trsm_diag_kernel<Elt, false><<<grid, threads, 0, S(s)>>>(Td, T.rs, T.cs, Bd, B.rs, B.cs, nb, n, kb, mc);
        after_launch("trsm_diag_kernel");
        // trailing update
        if (lower) {
            const uint32_t r0 = i0 + nb;
            if (r0 < m) {
                DeviceView<const Elt> L = T.sub(r0, m - r0, i0, nb);
                DeviceView<const Elt> X(B.ptr(i0, 0), B.rs, B.cs, nb, n);
                DeviceView<Elt> Cv = B.sub(r0, m - r0, 0, n);
                gemm_impl<Elt>(ctx, L, X, Cv, GemmOp::Sub, s, opts, "cuda::trsm_unit");
            }
        } else {
            if (i0 > 0) {
                DeviceView<const Elt> U = T.sub(0, i0, i0, nb);
                DeviceView<const Elt> X(B.ptr(i0, 0), B.rs, B.cs, nb, n);
                DeviceView<Elt> Cv = B.sub(0, i0, 0, n);
                gemm_impl<Elt>(ctx, U, X, Cv, GemmOp::Sub, s, opts, "cuda::trsm_unit");
            }
        }
    }
}

// ---- introspection ----
LaunchStats launch_stats() {
    LaunchStats o;
    o.kernel_launches = g_stats.kernel_launches.load(std::memory_order_relaxed);
    o.gemm_launches = g_stats.gemm_launches.load(std::memory_order_relaxed);
    o.batched_instances = g_stats.batched_instances.load(std::memory_order_relaxed);
    o.imma_launches = g_stats.imma_launches.load(std::memory_order_relaxed);
    return o;
}
void reset_launch_stats() {
    g_stats.kernel_launches.store(0, std::memory_order_relaxed);
    g_stats.gemm_launches.store(0, std::memory_order_relaxed);
    g_stats.batched_instances.store(0, std::memory_order_relaxed);
    g_stats.imma_launches.store(0, std::memory_order_relaxed);
}
const char* kernel_layer_version() { return "engine::cuda kernels, Phase 5 (u32 delayed-u64 / u64 Montgomery) + slot-22 G1 IMMA (int8 tensor cores, p <= 127) + slot-22 G2 device elimination (imma_rref)"; }

// ---- explicit instantiations ----
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

// (slot-22 G2) the device-resident elimination over F_p, p <= 127 (imma_rref; ENGINE_API.md Sec. 11.9).
// Included last: it uses this TU's launch_imma_gemm / after_launch / CUDA_CALL.
#include "ImmaElim.cuh"
