// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// ImmaElim.cuh -- the device-resident elimination over F_p, p <= 127 (slot-22 G2)
// =====================================================================
//
// Included ONCE, at the end of cuda_kernels.cu (outside its namespaces; it
// uses that TU's launch_imma_gemm / after_launch / CUDA_CALL). The public
// surface is imma_rref() in CudaKernels.h; ENGINE_API.md Sec. 11.9 is the
// binding description; the origin is the P5-G prototype p5g_elim.cu
// (docs/implementation/slot22/P5G_prototype_report.md Sec. 4).
//
// Storage: A8 = the matrix as balanced int8 residues (|x| <= floor(p/2) <= 63),
// row-major, ld = round_up(n, 32) (padding columns are zero and stay zero).
//
// 1. Factorisation: right-looking blocked PLU with column-order pivoting.
//    Outer panels of B = 256 columns, each factored as bases of W = 32 (or 16)
//    columns.
//    * Base (elim_base_kernel, ONE cooperative launch per base): every row
//      >= r is owned by one thread of a multi-block grid (rows r + tid + j T);
//      per column c one grid barrier. The pivot is the active row of SMALLEST
//      ORIGINAL INDEX with a non-zero in column c of the Schur complement
//      (atomicMin over orig[row]; the keys of column c + 1 are computed in the
//      same pass that eliminates column c). No row moves inside the base: the
//      pivot rows are marked (stamp), the others get multiplier and rank-1
//      update in place (packed L at the pivot column).
//    * Then the k pivot rows are swapped to positions r .. r+k-1 in pivot
//      order (elim_swaps_kernel: orig / posof follow), applied to columns
//      [J0, ld) by elim_laswp_kernel (the L entries left of J0 are never read
//      again and are cleared at the end).
//    * Inner update (rest of the outer panel) and, after the outer panel, the
//      outer update (trailing columns), each:  L11^{-1} (unit lower, k <= 256,
//      one block, X in shared memory: elim_tri_inv_kernel), U12^T = A12^T L11^{-T} (IMMA, int8 Set
//      epilogue), written back, A22 -= L21 U12 (IMMA, int8 Sub epilogue, in
//      place). K <= 256 here, far below the int32 bound T(p) (ENGINE_API Sec. 11.8).
// 2. To RREF (as DenseElim step 2, RREF = U1^{-1} U): left of each pivot and
//    every row >= rank cleared; pivot rows in blocks of 256 from the bottom:
//    R_blk = U_blk^{-1} A_blk (IMMA Set), rows above -= A[above, piv_blk] R_blk
//    (IMMA Sub).
//
// Why it is bit-identical to engine::elim::rref: RREF, pivot columns and rank
// are unique functions of A (DenseElim.h), and every step here is exact
// integer arithmetic: products of balanced residues summed in int32 (< 2^31
// by the bound), every stored value the balanced representative of the exact
// residue (bal_of_int: integer %, no floating point). pivot_rows: with the
// smallest-original-index rule the chosen set after column c is the row rank
// profile of A[:, 0..c] (a row r is chosen at c iff A[r, 0..c] is not in the
// span of the rows above it -- induction over c), a unique function of A.

#include <cooperative_groups.h>

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace engine {
namespace cuda {
namespace imma_elim {

namespace cg = cooperative_groups;

constexpr int PANEL = 256;            // outer panel (and the to-RREF block)
constexpr int BASE_THREADS = 256;
constexpr uint32_t kLd8Align = 32;    // A8 row stride: multiple of 32 (a base's 32 bytes never cross it)

__device__ __forceinline__ int32_t bal_i(int32_t x, int32_t p) {
    int32_t r = x % p;
    if (r < 0) r += p;
    return r > (p >> 1) ? r - p : r;
}

// u32 canonical slab (nr x n, row stride lds) -> A8 rows [r0, r0 + nr), columns [0, ld8) (zero beyond n)
__global__ void elim_to_s8_kernel(const uint32_t* src, size_t lds, uint32_t nr, uint32_t n, int8_t* A8, uint32_t ld8,
                                  uint32_t r0, uint32_t p) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<size_t>(nr) * ld8) return;
    const uint32_t i = static_cast<uint32_t>(idx / ld8), j = static_cast<uint32_t>(idx % ld8);
    int8_t v = 0;
    if (j < n) {
        const uint32_t x = src[static_cast<size_t>(i) * lds + j];
        v = static_cast<int8_t>(x > (p >> 1) ? static_cast<int32_t>(x) - static_cast<int32_t>(p) : static_cast<int32_t>(x));
    }
    A8[static_cast<size_t>(r0 + i) * ld8 + j] = v;
}
// A8 rows [r0, r0 + nr) x [0, n) -> canonical u32, contiguous (ld = n)
__global__ void elim_from_s8_kernel(const int8_t* A8, uint32_t ld8, uint32_t r0, uint32_t nr, uint32_t n, uint32_t* dst,
                                    int32_t p) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<size_t>(nr) * n) return;
    const uint32_t i = static_cast<uint32_t>(idx / n), j = static_cast<uint32_t>(idx % n);
    const int32_t v = A8[static_cast<size_t>(r0 + i) * ld8 + j];
    dst[idx] = static_cast<uint32_t>(v < 0 ? v + p : v);
}
__global__ void elim_iota_kernel(int* orig, int* posof, int* stamp, uint32_t m) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) { orig[i] = static_cast<int>(i); posof[i] = static_cast<int>(i); stamp[i] = -1; }
}
__global__ void elim_fill_int_kernel(int* x, int v, uint32_t cnt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cnt) x[i] = v;
}

struct BaseArgs {
    int8_t* A; uint32_t ld; int m;
    int r, j0, wd;
    int32_t p;
    const int8_t* inv;         // inv[x + 64]: the balanced inverse of the balanced residue x
    const int* orig;           // orig[pos]: the original row index of the row at position pos
    const int* posof;          // its inverse
    int* key;                  // [W]: atomicMin slots, INT_MAX on entry
    int* stamp; int stamp_id;  // stamp[pos] == stamp_id: the row became a pivot in this base (owner thread only)
    int* out_k;
    int* piv_orig;             // [W]: original index of pivot t of this base
    int* pivcols;              // global: pivcols[r + t] = pivot column of pivot t
};

template<int WB>
__global__ void __launch_bounds__(BASE_THREADS) elim_base_kernel(BaseArgs a) {
    cg::grid_group grid = cg::this_grid();
    const int T = static_cast<int>(gridDim.x * blockDim.x);
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int8_t* P = a.A + a.j0;
    const size_t ld = a.ld;
    for (int row = a.r + tid; row < a.m; row += T)
        if (P[static_cast<size_t>(row) * ld] != 0) atomicMin(&a.key[0], a.orig[row]);
    grid.sync();
    int k = 0;
    for (int c = 0; c < a.wd; ++c) {
        const int key = __ldcg(&a.key[c]);
        if (key == INT_MAX) {   // no pivot in column c (uniform): the rows are unchanged, scan column c + 1
            if (c + 1 < a.wd)
                for (int row = a.r + tid; row < a.m; row += T)
                    if (a.stamp[row] != a.stamp_id && P[static_cast<size_t>(row) * ld + c + 1] != 0)
                        atomicMin(&a.key[c + 1], a.orig[row]);
            grid.sync();
            continue;
        }
        const int pr = __ldcg(&a.posof[key]);
        __align__(16) int8_t pv[WB];
#pragma unroll
        for (int q = 0; q < WB; q += 16) *reinterpret_cast<int4*>(pv + q) = __ldcg(reinterpret_cast<const int4*>(P + static_cast<size_t>(pr) * ld + q));
        const int32_t inv = a.inv[pv[c] + 64];
        if (tid == 0) { a.piv_orig[k] = key; a.pivcols[a.r + k] = a.j0 + c; }
        for (int row = a.r + tid; row < a.m; row += T) {
            if (row == pr) { a.stamp[row] = a.stamp_id; continue; }
            if (a.stamp[row] == a.stamp_id) continue;
            int8_t* rp = P + static_cast<size_t>(row) * ld;
            __align__(16) int8_t v[WB];
#pragma unroll
            for (int q = 0; q < WB; q += 16) *reinterpret_cast<int4*>(v + q) = *reinterpret_cast<const int4*>(rp + q);
            if (v[c] != 0) {
                const int32_t mult = bal_i(static_cast<int32_t>(v[c]) * inv, a.p);
                v[c] = static_cast<int8_t>(mult);
#pragma unroll
                for (int cc = 0; cc < WB; ++cc)
                    if (cc > c && cc < a.wd) v[cc] = static_cast<int8_t>(bal_i(static_cast<int32_t>(v[cc]) - mult * pv[cc], a.p));
#pragma unroll
                for (int q = 0; q < WB; q += 16) *reinterpret_cast<int4*>(rp + q) = *reinterpret_cast<const int4*>(v + q);
            }
            if (c + 1 < a.wd && v[c + 1] != 0) atomicMin(&a.key[c + 1], a.orig[row]);
        }
        ++k;
        grid.sync();
    }
    if (tid == 0) *a.out_k = k;
}

// Move pivot t of the base (original index piv_orig[t]) to position r + t, in order; swaps[t] = the
// position it came from (LAPACK convention: apply the swaps r+t <-> swaps[t] in order).
__global__ void elim_swaps_kernel(int* orig, int* posof, const int* piv_orig, int* swaps, int r, int k) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int t = 0; t < k; ++t) {
        const int key = piv_orig[t], s = posof[key], dst = r + t;
        swaps[t] = s;
        if (s != dst) {
            const int o2 = orig[dst];
            orig[dst] = key; orig[s] = o2;
            posof[key] = dst; posof[o2] = s;
        }
    }
}
__global__ void elim_laswp_kernel(int8_t* A, uint32_t ld, int c0, int c1, int r, const int* swaps, int k) {
    const int c = c0 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= c1) return;
    for (int t = 0; t < k; ++t) {
        const int s = swaps[t];
        if (s != r + t) {
            int8_t* x = A + static_cast<size_t>(r + t) * ld + c;
            int8_t* y = A + static_cast<size_t>(s) * ld + c;
            const int8_t tmp = *x; *x = *y; *y = tmp;
        }
    }
}
// The triangular inverses in one block of k <= 256 threads (thread c owns column c of X), X in
// dynamic shared memory (k x k int8), the triangle read from a compact copy T (k x k, row-major,
// broadcast reads). The row loop is uniform across the block (zeros stand in for the other
// triangle), so every thread runs the same t-loop. Exact: |sum| <= 256 * 63^2 < 2^31.
//   lower = true : T unit lower (T[i][t], t < i), X = T^{-1}:  X[i][c] = -sum_{t<i} T[i][t] X[t][c], X[c][c] = 1
//   lower = false: T upper, non-zero diagonal,     X = T^{-1}:  X[i][c] = -inv(T[i][i]) sum_{t>i} T[i][t] X[t][c]
// Output: Xg (k x kp, row-major; the caller zeroed the padding).
__global__ void elim_tri_inv_kernel(const int8_t* T, int k, int8_t* Xg, int kp, int32_t p, const int8_t* inv, bool lower) {
    extern __shared__ int8_t X[];   // k x k
    const int c = static_cast<int>(threadIdx.x);
    if (lower) {
        for (int i = 0; i < k; ++i) {
            if (c < k) {
                int32_t v;
                if (i < c) v = 0;
                else if (i == c) v = 1;
                else {
                    int32_t s = 0;
                    const int8_t* Ti = T + static_cast<size_t>(i) * k;
                    for (int t = c; t < i; ++t) s += static_cast<int32_t>(Ti[t]) * X[t * k + c];
                    v = bal_i(-s, p);
                }
                X[i * k + c] = static_cast<int8_t>(v);
            }
            __syncthreads();
        }
    } else {
        for (int i = k - 1; i >= 0; --i) {
            if (c < k) {
                int32_t v;
                const int8_t* Ti = T + static_cast<size_t>(i) * k;
                const int32_t di = inv[Ti[i] + 64];
                if (i > c) v = 0;
                else if (i == c) v = di;
                else {
                    int32_t s = 0;
                    for (int t = i + 1; t <= c; ++t) s += static_cast<int32_t>(Ti[t]) * X[t * k + c];
                    v = bal_i(bal_i(-s, p) * di, p);
                }
                X[i * k + c] = static_cast<int8_t>(v);
            }
            __syncthreads();
        }
    }
    if (c < k)
        for (int i = 0; i < k; ++i) Xg[static_cast<size_t>(i) * kp + c] = X[i * k + c];
}
// Tc (k x k) = A[r0 + t][piv[s]]
__global__ void elim_tri_gather_kernel(const int8_t* A, uint32_t ld, int r0, const int* piv, int k, int8_t* Tc) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= k * k) return;
    const int t = idx / k, sc = idx % k;
    Tc[idx] = A[static_cast<size_t>(r0 + t) * ld + piv[sc]];
}

// Tt (ncols x kp) = A[rp .. rp+k) x [c0, c0+ncols) transposed (columns k .. kp of Tt: zero by a prior memset)
__global__ void elim_rows_T_kernel(const int8_t* A, uint32_t ld, int rp, int k, int c0, int ncols, int8_t* Tt, int kp) {
    __shared__ int8_t tile[32][33];
    const int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    for (int y = threadIdx.y; y < 32; y += 8) {
        const int t = by + y, c = bx + threadIdx.x;
        tile[y][threadIdx.x] = (t < k && c < ncols) ? A[static_cast<size_t>(rp + t) * ld + c0 + c] : 0;
    }
    __syncthreads();
    for (int y = threadIdx.y; y < 32; y += 8) {
        const int c = bx + y, t = by + threadIdx.x;
        if (c < ncols && t < k) Tt[static_cast<size_t>(c) * kp + t] = tile[threadIdx.x][y];
    }
}
// A[rp + t][c0 + c] = Tt[c][t]
__global__ void elim_rows_T_back_kernel(int8_t* A, uint32_t ld, int rp, int k, int c0, int ncols, const int8_t* Tt, int kp) {
    __shared__ int8_t tile[32][33];
    const int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    for (int y = threadIdx.y; y < 32; y += 8) {
        const int c = bx + y, t = by + threadIdx.x;
        tile[y][threadIdx.x] = (c < ncols && t < k) ? Tt[static_cast<size_t>(c) * kp + t] : 0;
    }
    __syncthreads();
    for (int y = threadIdx.y; y < 32; y += 8) {
        const int t = by + y, c = bx + threadIdx.x;
        if (t < k && c < ncols) A[static_cast<size_t>(rp + t) * ld + c0 + c] = tile[threadIdx.x][y];
    }
}
// G (rows x kp) = A[r0 + i][piv[s]] for s < k, 0 beyond
__global__ void elim_gather_cols_kernel(const int8_t* A, uint32_t ld, int r0, int rows, const int* piv, int k, int8_t* G,
                                        int kp) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<size_t>(rows) * kp) return;
    const int i = static_cast<int>(idx / kp), s = static_cast<int>(idx % kp);
    G[idx] = s < k ? A[static_cast<size_t>(r0 + i) * ld + piv[s]] : 0;
}
// clear left of the pivot in rows < rank, and every row >= rank (columns [0, n))
__global__ void elim_clean_kernel(int8_t* A, uint32_t ld, int m, int n, const int* piv, int rank) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<size_t>(m) * n) return;
    const int i = static_cast<int>(idx / n), c = static_cast<int>(idx % n);
    if (i >= rank || c < piv[i]) A[static_cast<size_t>(i) * ld + c] = 0;
}

struct AtomicElimStats {
    std::atomic<uint64_t> calls{0}, bases{0}, base_columns{0}, coop_launches{0}, gemms{0}, kernel_launches{0};
    std::atomic<uint64_t> h2d_bytes{0}, d2h_bytes{0};
    std::atomic<uint64_t> peak_device_bytes{0};
    std::atomic<uint64_t> nanos{0};
};
inline AtomicElimStats& stats() { static AtomicElimStats s; return s; }

inline uint32_t blocks_for(size_t n, uint32_t t = 256) { return static_cast<uint32_t>((n + t - 1) / t); }

// The device footprint of one call, rounded up to a size class (4 per octave) so the arena's
// exact-size free lists reuse it across the many differently shaped eliminations of a run.
struct Layout {
    uint32_t ld8 = 0;
    size_t a8 = 0, work = 0, linv = 0, ints = 0, staging = 0, total = 0;
    size_t off_work1 = 0, off_work2 = 0, off_work3 = 0, off_linv = 0, off_inv = 0, off_ints = 0, off_staging = 0;
};
inline size_t round256(size_t x) { return (x + 255) / 256 * 256; }
inline size_t size_class(size_t x) {
    size_t b = 1;
    while (b * 2 <= x) b *= 2;   // b = 2^floor(log2 x)
    if (b < 4096) return round256(x);
    const size_t q = b / 4;
    return (x + q - 1) / q * q;
}
inline uint32_t default_slab_rows(uint32_t n) {
    const size_t per = static_cast<size_t>(std::max<uint32_t>(n, 1)) * 4;
    const size_t rows = (size_t(64) << 20) / per;
    return static_cast<uint32_t>(std::max<size_t>(1, std::min<size_t>(rows, 0x7FFFFFFF)));
}
inline Layout layout_for(uint32_t m, uint32_t n, uint32_t slab_rows) {
    Layout L;
    L.ld8 = (n + kLd8Align - 1) / kLd8Align * kLd8Align;
    L.a8 = round256(static_cast<size_t>(m) * L.ld8);
    L.work = round256(static_cast<size_t>(std::max(m, n)) * PANEL);
    L.linv = static_cast<size_t>(PANEL) * PANEL;
    const size_t nint = 3 * static_cast<size_t>(m) + std::min(m, n) + 64 + 4 * 32 + 64;
    L.ints = round256(nint * 4);
    L.staging = round256(static_cast<size_t>(slab_rows) * std::max<uint32_t>(n, 1) * 4);
    L.off_work1 = L.a8;
    L.off_work2 = L.off_work1 + L.work;
    L.off_work3 = L.off_work2 + L.work;
    L.off_linv = L.off_work3 + L.work;
    L.off_inv = L.off_linv + L.linv;
    L.off_ints = L.off_inv + 256;
    L.off_staging = L.off_ints + L.ints;
    L.total = size_class(L.off_staging + L.staging);
    return L;
}

} // namespace imma_elim

size_t imma_rref_device_bytes(uint32_t m, uint32_t n, uint32_t slab_rows) {
    if (slab_rows == 0) slab_rows = imma_elim::default_slab_rows(n);
    return imma_elim::layout_for(m, n, std::min(slab_rows, std::max<uint32_t>(m, 1))).total;
}

ImmaRrefResult imma_rref(const ModCtx<uint32_t>& ctx, uint32_t m, uint32_t n, const ImmaRowSource& src,
                         const ImmaRowSink& sink, uint32_t slab_rows, uint32_t base_width) {
    using namespace imma_elim;
    ENGINE_CHECK_MSG(imma_elim_eligible(ctx), "cuda::imma_rref: needs a uint32_t field context with 2 <= p <= 127");
    ENGINE_CHECK_MSG(imma_supported(), "cuda::imma_rref: no IMMA-capable device (imma_supported() == false)");
    ENGINE_CHECK_MSG(base_width == 32 || base_width == 16, "cuda::imma_rref: base_width must be 16 or 32");
    ENGINE_CHECK_MSG(m < 0x7FFFFFF0u && n < 0x7FFFFFF0u, "cuda::imma_rref: shape too large");
    const auto t_start = std::chrono::steady_clock::now();
    static const bool trace = std::getenv("ENGINE_TRACE_IMMA_ELIM") != nullptr;   // diagnostic: phase times (adds syncs)
    double tr_up = 0, tr_base = 0, tr_upd = 0, tr_rref = 0, tr_down = 0;
    auto tr_now = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };
    AtomicElimStats& st = stats();
    st.calls.fetch_add(1, std::memory_order_relaxed);
    ImmaRrefResult res;
    if (m == 0 || n == 0) return res;
    const int32_t p = static_cast<int32_t>(ctx.p);
    if (slab_rows == 0) slab_rows = default_slab_rows(n);
    slab_rows = std::min(slab_rows, m);
    const Layout L = layout_for(m, n, slab_rows);
    const uint32_t ld8 = L.ld8;

    CudaCtx& cx = CudaCtx::get();
    const Stream stream = cx.create_stream();
    const cudaStream_t s = S(stream);
    uint8_t* mem = static_cast<uint8_t*>(cx.dalloc(L.total));
    struct Guard {
        CudaCtx& cx; Stream st; void* mem; size_t bytes; bool ok = false;
        ~Guard() {
            // on an exception the stream may still run kernels on mem: drain it first
            try { if (!ok) cx.sync(st); } catch (...) {}
            try { cx.dfree(mem, bytes); } catch (...) {}
            try { cx.destroy_stream(st); } catch (...) {}
        }
    } guard{cx, stream, mem, L.total};
    {
        uint64_t prev = st.peak_device_bytes.load(std::memory_order_relaxed);
        while (L.total > prev && !st.peak_device_bytes.compare_exchange_weak(prev, L.total)) {}
    }
    uint64_t launches = 0, coop = 0, gemms = 0, bases = 0, bcols = 0, h2d = 0, d2h = 0;

    int8_t* A8 = reinterpret_cast<int8_t*>(mem);
    int8_t* W1 = reinterpret_cast<int8_t*>(mem + L.off_work1);
    int8_t* W2 = reinterpret_cast<int8_t*>(mem + L.off_work2);
    int8_t* W3 = reinterpret_cast<int8_t*>(mem + L.off_work3);
    int8_t* Linv = reinterpret_cast<int8_t*>(mem + L.off_linv);
    int8_t* d_inv = reinterpret_cast<int8_t*>(mem + L.off_inv);
    int* ints = reinterpret_cast<int*>(mem + L.off_ints);
    int* d_orig = ints;
    int* d_posof = d_orig + m;
    int* d_stamp = d_posof + m;
    int* d_piv = d_stamp + m;                         // min(m, n) + 64
    int* d_key = d_piv + std::min(m, n) + 64;         // 32
    int* d_piv_orig = d_key + 32;                     // 32
    int* d_swaps = d_piv_orig + 32;                   // 32
    int* d_k = d_swaps + 32;                          // 1
    uint32_t* stg = reinterpret_cast<uint32_t*>(mem + L.off_staging);
    const size_t stg_elems = L.staging / 4;

    // the balanced inverse table
    {
        int8_t tab[256] = {0};
        for (int x = -64; x <= 63; ++x) {
            const int xc = ((x % p) + p) % p;
            if (xc == 0) continue;
            int inv = 0;
            for (int y = 1; y < p; ++y) if ((xc * y) % p == 1) { inv = y; break; }
            tab[x + 64] = static_cast<int8_t>(inv > p / 2 ? inv - p : inv);
        }
        cx.upload(tab, d_inv, 128, stream);
        h2d += 128;
    }
    elim_iota_kernel<<<blocks_for(m), 256, 0, s>>>(d_orig, d_posof, d_stamp, m);
    after_launch("elim_iota_kernel"); ++launches;

    // ---- upload + convert, in row slabs ----
    for (uint32_t r0 = 0; r0 < m; r0 += slab_rows) {
        const uint32_t nr = std::min(slab_rows, m - r0);
        const DenseView<const uint32_t> v = src(r0, nr);
        ENGINE_CHECK_MSG(v.rows() == nr && v.cols() == n && v.row_contiguous(), "cuda::imma_rref: the source slab must be a row-contiguous nr x n view");
        ENGINE_CHECK_MSG(v.invariant() == Invariant::Canonical, "cuda::imma_rref: the source must be Canonical");
        const size_t lds = v.row_stride();
        const size_t foot = (static_cast<size_t>(nr) - 1) * lds + n;
        if (foot <= stg_elems) {
            cx.upload(v.data(), stg, foot * 4, stream);
            h2d += foot * 4;
            elim_to_s8_kernel<<<blocks_for(static_cast<size_t>(nr) * ld8), 256, 0, s>>>(stg, lds, nr, n, A8, ld8, r0, static_cast<uint32_t>(p));
            after_launch("elim_to_s8_kernel"); ++launches;
        } else {   // a narrow window of a wide buffer: row by row
            for (uint32_t i = 0; i < nr; ++i) {
                cx.upload(v.data() + static_cast<size_t>(i) * lds, stg, static_cast<size_t>(n) * 4, stream);
                h2d += static_cast<size_t>(n) * 4;
                elim_to_s8_kernel<<<blocks_for(ld8), 256, 0, s>>>(stg, n, 1, n, A8, ld8, r0 + i, static_cast<uint32_t>(p));
                after_launch("elim_to_s8_kernel"); ++launches;
            }
        }
    }

    if (trace) { CudaCtx::get().sync(stream); tr_up = tr_now(); }
    // ---- 1. factorisation ----
    int dev = 0, sms = 0, occ = 0;
    CUDA_CALL(cudaGetDevice(&dev));
    CUDA_CALL(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const bool w32 = (base_width == 32);
    if (w32) CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, elim_base_kernel<32>, BASE_THREADS, 0));
    else     CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, elim_base_kernel<16>, BASE_THREADS, 0));
    const int max_coop = std::max(1, sms * std::max(1, std::min(occ, 1)));   // one block per SM: a cheap barrier
    static const bool tri_smem_ok = [] {
        return cudaFuncSetAttribute(elim_tri_inv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, PANEL * PANEL) == cudaSuccess;
    }();
    ENGINE_CHECK_MSG(tri_smem_ok, "cuda::imma_rref: cannot raise the triangular-inverse kernel's shared memory to 64 KiB");
    const int Wd = static_cast<int>(base_width);

    auto gemm8 = [&](int M, int N, int Kp, const int8_t* Aop, int lda, const int8_t* Bt, int ldb, int8_t* C8, int ldc, int op) {
        if (M <= 0 || N <= 0) return;
        imma::GemmArgs ga{};
        ga.M = M; ga.N = N; ga.K = Kp;
        ga.A = Aop; ga.Bt = Bt; ga.lda = lda; ga.ldb = ldb;
        ga.op = op; ga.p = p;
        ga.flush_tiles = imma_flush_tiles_for(static_cast<uint32_t>(p), static_cast<uint32_t>(Kp), 0);
        ga.C8 = C8; ga.ldc8 = ldc;
        launch_imma_gemm(ga, 1, s);
        ++gemms; ++launches;
    };
    // Linv (k x kp) = the inverse of the k x k triangle A[r0 + t][piv[s]] (unit lower, or upper); W3 holds the compact copy
    // first (W3 is free at both call sites: it is rewritten by the L21 gather that follows).
    auto tri_inv = [&](int r0, const int* piv, int k, int kp, bool lower) {
        elim_tri_gather_kernel<<<blocks_for(static_cast<size_t>(k) * k), 256, 0, s>>>(A8, ld8, r0, piv, k, W3);
        after_launch("elim_tri_gather_kernel"); ++launches;
        elim_tri_inv_kernel<<<1, std::max(32, (k + 31) / 32 * 32), static_cast<size_t>(k) * k, s>>>(W3, k, Linv, kp, p, d_inv, lower);
        after_launch("elim_tri_inv_kernel"); ++launches;
    };
    // U12 = L11^{-1} A12 on rows [rp, rp + k) x cols [ca, cb); A[rp + k .. m) x [ca, cb) -= L21 U12
    auto block_update = [&](int rp, int k, const int* piv, int ca, int cb) {
        const int ncols = cb - ca;
        if (k == 0 || ncols <= 0) return;
        const int kp = (k + 15) / 16 * 16;
        CUDA_CALL(cudaMemsetAsync(Linv, 0, static_cast<size_t>(PANEL) * PANEL, s));
        tri_inv(rp, piv, k, kp, true);
        CUDA_CALL(cudaMemsetAsync(W1, 0, static_cast<size_t>(ncols) * kp, s));
        elim_rows_T_kernel<<<dim3(blocks_for(ncols, 32), blocks_for(k, 32)), dim3(32, 8), 0, s>>>(A8, ld8, rp, k, ca, ncols, W1, kp);
        after_launch("elim_rows_T_kernel"); ++launches;
        gemm8(ncols, k, kp, W1, kp, Linv, kp, W2, kp, 0);   // U12^T = A12^T L11^{-T}  (columns k..kp of W2 stay 0: N = k)
        elim_rows_T_back_kernel<<<dim3(blocks_for(ncols, 32), blocks_for(k, 32)), dim3(32, 8), 0, s>>>(A8, ld8, rp, k, ca, ncols, W2, kp);
        after_launch("elim_rows_T_back_kernel"); ++launches;
        const int m2 = static_cast<int>(m) - (rp + k);
        if (m2 > 0) {
            elim_gather_cols_kernel<<<blocks_for(static_cast<size_t>(m2) * kp), 256, 0, s>>>(A8, ld8, rp + k, m2, piv, k, W3, kp);
            after_launch("elim_gather_cols_kernel"); ++launches;
            gemm8(m2, ncols, kp, W3, kp, W2, kp, A8 + static_cast<size_t>(rp + k) * ld8 + ca, static_cast<int>(ld8), 2);
        }
    };

    int r = 0;
    int stamp_id = 0;
    double tf[4] = {0, 0, 0, 0};   // trace: base (+ k read), swaps, inner updates, outer updates
    double tf_last = 0.0;
    auto ftick = [&](int slot) { if (trace) { cx.sync(stream); const double t = tr_now(); if (slot >= 0) tf[slot] += t - tf_last; tf_last = t; } };
    ftick(-1);
    for (int J0 = 0; J0 < static_cast<int>(n) && r < static_cast<int>(m); J0 += PANEL) {
        const int JE = std::min(J0 + PANEL, static_cast<int>(n)), rps = r;
        for (int j0 = J0; j0 < JE && r < static_cast<int>(m); j0 += Wd) {
            const int je = std::min(j0 + Wd, JE), wd = je - j0;
            const int rows = static_cast<int>(m) - r;
            elim_fill_int_kernel<<<1, 64, 0, s>>>(d_key, INT_MAX, 32);
            after_launch("elim_fill_int_kernel"); ++launches;
            BaseArgs ba{};
            ba.A = A8; ba.ld = ld8; ba.m = static_cast<int>(m);
            ba.r = r; ba.j0 = j0; ba.wd = wd; ba.p = p;
            ba.inv = d_inv; ba.orig = d_orig; ba.posof = d_posof; ba.key = d_key;
            ba.stamp = d_stamp; ba.stamp_id = stamp_id++;
            ba.out_k = d_k; ba.piv_orig = d_piv_orig; ba.pivcols = d_piv;
            const int gb = std::max(1, std::min(max_coop, static_cast<int>(blocks_for(rows, BASE_THREADS))));
            void* args[] = {&ba};
            if (w32) CUDA_CALL(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(&elim_base_kernel<32>), dim3(gb), dim3(BASE_THREADS), args, 0, s));
            else     CUDA_CALL(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(&elim_base_kernel<16>), dim3(gb), dim3(BASE_THREADS), args, 0, s));
            after_launch("elim_base_kernel"); ++launches; ++coop; ++bases; bcols += static_cast<uint64_t>(wd);
            int k = 0;
            cx.download(&k, d_k, sizeof(int), stream);
            ftick(0);
            d2h += sizeof(int);
            ENGINE_CHECK_MSG(k >= 0 && k <= wd && k <= rows, "cuda::imma_rref: base pivot count out of range");
            if (k > 0) {
                elim_swaps_kernel<<<1, 1, 0, s>>>(d_orig, d_posof, d_piv_orig, d_swaps, r, k);
                after_launch("elim_swaps_kernel"); ++launches;
                elim_laswp_kernel<<<blocks_for(ld8 - J0), 256, 0, s>>>(A8, ld8, J0, static_cast<int>(ld8), r, d_swaps, k);
                after_launch("elim_laswp_kernel"); ++launches;
                ftick(1);
                block_update(r, k, d_piv + r, je, JE);   // inner: the rest of the outer panel
                ftick(2);
            }
            r += k;
        }
        block_update(rps, r - rps, d_piv + rps, JE, static_cast<int>(n));   // outer: the trailing columns
        ftick(3);
    }
    tr_base = tf[0];
    if (trace) std::printf("        [imma_rref factor] bases %.4f swaps %.4f inner %.4f outer %.4f s\n", tf[0], tf[1], tf[2], tf[3]);
    const int rank = r;
    double tr_fact = 0;
    if (trace) { cx.sync(stream); tr_fact = tr_now(); tr_upd = tr_fact - tr_up - tr_base; }

    // ---- 2. to RREF ----
    res.rank = static_cast<uint32_t>(rank);
    std::vector<int> h_piv(rank), h_orig(rank);
    if (rank > 0) {
        cx.download(h_piv.data(), d_piv, sizeof(int) * static_cast<size_t>(rank), stream);
        cx.download(h_orig.data(), d_orig, sizeof(int) * static_cast<size_t>(rank), stream);
        d2h += 2 * sizeof(int) * static_cast<size_t>(rank);
    }
    const bool want_rref = static_cast<bool>(sink) && rank > 0;
    if (want_rref) {
        double tq[6] = {0, 0, 0, 0, 0, 0};   // trace: clean, inverse, transposes, Set GEMM, gather, Sub GEMM
        double tq_last = trace ? tr_now() : 0.0;
        auto tick = [&](int slot) { if (trace) { cx.sync(stream); const double t = tr_now(); tq[slot] += t - tq_last; tq_last = t; } };
        elim_clean_kernel<<<blocks_for(static_cast<size_t>(m) * n), 256, 0, s>>>(A8, ld8, static_cast<int>(m), static_cast<int>(n), d_piv, rank);
        after_launch("elim_clean_kernel"); ++launches;
        tick(0);
        for (int t1b = rank; t1b > 0; t1b -= PANEL) {
            const int tb = std::max(0, t1b - PANEL), k = t1b - tb, kp = (k + 15) / 16 * 16;
            const int ca = h_piv[tb] / 16 * 16, ncols = static_cast<int>(n) - ca;
            CUDA_CALL(cudaMemsetAsync(Linv, 0, static_cast<size_t>(PANEL) * PANEL, s));
            tri_inv(tb, d_piv + tb, k, kp, false);
            tick(1);
            CUDA_CALL(cudaMemsetAsync(W1, 0, static_cast<size_t>(ncols) * kp, s));
            elim_rows_T_kernel<<<dim3(blocks_for(ncols, 32), blocks_for(k, 32)), dim3(32, 8), 0, s>>>(A8, ld8, tb, k, ca, ncols, W1, kp);
            after_launch("elim_rows_T_kernel"); ++launches;
            tick(2);
            gemm8(k, ncols, kp, Linv, kp, W1, kp, A8 + static_cast<size_t>(tb) * ld8 + ca, static_cast<int>(ld8), 0);   // R_blk = U^{-1} A_blk
            tick(3);
            if (tb > 0) {
                CUDA_CALL(cudaMemsetAsync(W2, 0, static_cast<size_t>(ncols) * kp, s));
                elim_rows_T_kernel<<<dim3(blocks_for(ncols, 32), blocks_for(k, 32)), dim3(32, 8), 0, s>>>(A8, ld8, tb, k, ca, ncols, W2, kp);
                after_launch("elim_rows_T_kernel"); ++launches;
                tick(2);
                elim_gather_cols_kernel<<<blocks_for(static_cast<size_t>(tb) * kp), 256, 0, s>>>(A8, ld8, 0, tb, d_piv + tb, k, W3, kp);
                after_launch("elim_gather_cols_kernel"); ++launches;
                tick(4);
                gemm8(tb, ncols, kp, W3, kp, W2, kp, A8 + ca, static_cast<int>(ld8), 2);
                tick(5);
            }
        }
        if (trace) std::printf("        [imma_rref to_rref] clean %.4f inverse %.4f transposes %.4f gemm_set %.4f gather %.4f gemm_sub %.4f s\n",
                               tq[0], tq[1], tq[2], tq[3], tq[4], tq[5]);
        if (trace) { cx.sync(stream); tr_rref = tr_now() - tr_fact; }
        // the host side of the sink (its allocation) overlaps the to-RREF kernels queued above
        const DenseView<uint32_t> out = sink(res.rank);
        ENGINE_CHECK_MSG(out.rows() == res.rank && out.cols() == n && out.row_contiguous(), "cuda::imma_rref: the sink must be a row-contiguous rank x n view");
        // ---- download the rank x n RREF rows, in slabs ----
        const size_t ldo = out.row_stride();
        const uint32_t rows_per = static_cast<uint32_t>(std::max<size_t>(1, stg_elems / n));
        std::vector<uint32_t> tmp;
        for (uint32_t r0 = 0; r0 < res.rank; r0 += rows_per) {
            const uint32_t nr = std::min(rows_per, res.rank - r0);
            elim_from_s8_kernel<<<blocks_for(static_cast<size_t>(nr) * n), 256, 0, s>>>(A8, ld8, r0, nr, n, stg, p);
            after_launch("elim_from_s8_kernel"); ++launches;
            const size_t bytes = static_cast<size_t>(nr) * n * 4;
            uint32_t* dst = out.data() + static_cast<size_t>(r0) * ldo;
            if (ldo == n) {
                cx.download(dst, stg, bytes, stream);
            } else {
                tmp.resize(static_cast<size_t>(nr) * n);
                cx.download(tmp.data(), stg, bytes, stream);
                for (uint32_t i = 0; i < nr; ++i)
                    std::copy(tmp.begin() + static_cast<size_t>(i) * n, tmp.begin() + static_cast<size_t>(i + 1) * n,
                              dst + static_cast<size_t>(i) * ldo);
            }
            d2h += bytes;
        }
    }
    cx.sync(stream);
    guard.ok = true;
    if (trace) {
        tr_down = tr_now() - tr_fact - tr_rref;
        std::printf("        [imma_rref %ux%u p=%d rank %d] upload %.4f  bases %.4f (%llu, incl. the k read)  updates %.4f  to_rref %.4f  download %.4f  total %.4f s  dev %.1f MB\n",
                    m, n, p, rank, tr_up, tr_base, static_cast<unsigned long long>(bases), tr_upd, tr_rref, tr_down, tr_now(),
                    static_cast<double>(L.total) / 1048576.0);
    }
    res.pivots.assign(h_piv.begin(), h_piv.end());
    res.pivot_rows.assign(h_orig.begin(), h_orig.end());
    for (uint32_t t = 0; t < res.rank; ++t) {
        ENGINE_CHECK_MSG(res.pivots[t] < n && res.pivot_rows[t] < m && (t == 0 || res.pivots[t] > res.pivots[t - 1]),
                         "cuda::imma_rref: inconsistent pivot data");
    }
    st.bases.fetch_add(bases, std::memory_order_relaxed);
    st.base_columns.fetch_add(bcols, std::memory_order_relaxed);
    st.coop_launches.fetch_add(coop, std::memory_order_relaxed);
    st.gemms.fetch_add(gemms, std::memory_order_relaxed);
    st.kernel_launches.fetch_add(launches, std::memory_order_relaxed);
    st.h2d_bytes.fetch_add(h2d, std::memory_order_relaxed);
    st.d2h_bytes.fetch_add(d2h, std::memory_order_relaxed);
    st.nanos.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - t_start).count()), std::memory_order_relaxed);
    return res;
}

ImmaElimStats imma_elim_stats() {
    imma_elim::AtomicElimStats& a = imma_elim::stats();
    ImmaElimStats o;
    o.calls = a.calls.load(); o.bases = a.bases.load(); o.base_columns = a.base_columns.load();
    o.coop_launches = a.coop_launches.load(); o.gemms = a.gemms.load(); o.kernel_launches = a.kernel_launches.load();
    o.h2d_bytes = a.h2d_bytes.load(); o.d2h_bytes = a.d2h_bytes.load();
    o.peak_device_bytes = static_cast<size_t>(a.peak_device_bytes.load());
    o.seconds = static_cast<double>(a.nanos.load()) * 1e-9;
    return o;
}
void reset_imma_elim_stats() {
    imma_elim::AtomicElimStats& a = imma_elim::stats();
    a.calls = 0; a.bases = 0; a.base_columns = 0; a.coop_launches = 0; a.gemms = 0; a.kernel_launches = 0;
    a.h2d_bytes = 0; a.d2h_bytes = 0; a.peak_device_bytes = 0; a.nanos = 0;
}

} // namespace cuda
} // namespace engine
