// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// CudaKernels.h -- device kernels of the matrix engine (Phase 5)
// =====================================================================
//
// Every kernel here is the device twin of the CPU kernel of the same name in
// Kernels.h (engine::kernels) and MUST produce the identical bytes: the
// arithmetic is exact integer arithmetic, canonical residues are unique, so a
// correct device kernel is bit-identical by construction and
// CudaBitIdentityValidation asserts it with memcmp, no tolerances.
//
//   uint32_t : products accumulated exactly in uint64_t (mad.wide.u32),
//              k-blocked at kb <= min(ctx.k_max, 256): one Barrett reduction
//              (__umul64hi with ctx.barrett_m) per k-block per output cell,
//              blocks combined with mod_add. kb < 16 (p > 2^30) switches to a
//              per-product counter (same rule, finer grain).
//   uint64_t : 64-bit Montgomery (__umul64hi, ctx.mont_ninv) per product:
//              sum_k REDC(a_k b_k) = (sum a_k b_k) R^{-1}, times R^2 via one
//              final REDC -- exact, no representation change of the operands.
//              Even or >= 2^63 moduli (Regime::U64_Wide) use an exact
//              shift-subtract 128-by-64 reduction (correct, slow, unused by
//              the project).
//   uint32_t, 2 <= p <= 127 (slot-22 G1, ENGINE_API.md Sec. 11.8): with an IMMA
//              workspace (GemmOpts::imma_workspace / GemmBatch::prepare_imma) the
//              int8 tensor-core route -- balanced int8 residues, exact int32
//              accumulation (reduced mod p every T(p) k-tiles), acc % p
//              epilogue; ENGINE_GPU_IMMA=0 turns it off (retired in OFF builds by
//              S24 CL-R2: honoured only with OC_ENABLE_LEGACY=ON, refused otherwise).
//   No floating point anywhere. No atomics on results: split-K writes each
//   canonical partial to its own workspace slot and a second kernel sums
//   them mod p in a fixed order (the D4 split-K rule: partials canonicalised
//   before combination; the sum is then associative, so any order is fine).
//
// Operands are DeviceView<Elt>s: resolved strides, so every layout / logical
// transposition combination is one kernel instantiation chosen by the
// contiguity flags (A k-contiguous?, B n-contiguous?) -- both directions are
// coalesced through shared-memory tiles either way.
//
// All launches are asynchronous on `stream`; errors of the LAUNCH are thrown
// as CudaError immediately, execution errors surface at the next sync.
// Everything here is graph-capturable when no allocation is needed
// (splits == 1, batches uploaded before begin()).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "CudaCtx.h"
#include "DeviceBuf.h"
#include "Kernels.h"     // GemmOp
#include "ModCtx.h"

namespace engine {
namespace cuda {

using kernels::GemmOp;

// ---- GEMM ----------------------------------------------------------------------
struct GemmOpts {
    uint32_t splits = 1;              // split-K factor (1 = none)
    void*    workspace = nullptr;     // splits > 1: >= gemm_workspace_bytes()
    size_t   workspace_bytes = 0;
    int      tile = 0;                // 0 = auto, 32 or 64 (validation sweeps both)
    // (slot-22 G1) the int8 tensor-core (IMMA) route, ENGINE_API.md Sec. 11.8. Taken iff
    // imma_eligible(ctx) && imma_supported() && k > 0 && imma_workspace_bytes >=
    // imma_workspace_bytes(m, n, min(k, 64)) && (imma == 1 || (imma == -1 && imma_policy())).
    // Otherwise the u32 kernel runs (splits / tile apply to it only). Same bytes either way.
    int      imma = -1;               // -1: the policy (ENGINE_GPU_IMMA), 0: never, 1: whenever eligible (validation)
    void*    imma_workspace = nullptr;   // device memory; imma_workspace_bytes(m, n, k) runs K in one piece, less
                                         // (>= imma_workspace_bytes(m, n, 64)) in K chunks (imma_k_chunk); allocate before a capture
    size_t   imma_workspace_bytes = 0;
    uint32_t imma_flush_tiles = 0;    // 0: automatic (the int32 bound); else 1 .. imma_max_flush_tiles(p) (ENGINE_CHECKed)
};
size_t gemm_workspace_bytes(uint32_t m, uint32_t n, uint32_t splits, size_t elt_bytes);
uint32_t suggest_splits(uint32_t m, uint32_t n, uint32_t k);   // the cost-model hint

// ---- (slot-22 G1) the IMMA route: int8 tensor cores, balanced residues, int32 accumulation ----
// Modulus-only: a uint32_t context with 2 <= p <= 127 (field or Z/ell^k), where every balanced
// residue |bal(x)| <= floor(p/2) <= 63 fits int8.
template<class Elt>
inline bool imma_eligible(const ModCtx<Elt>& ctx) { return sizeof(Elt) == 4 && ctx.p >= 2 && ctx.p <= 127; }
// The device can run it: available(), compute capability >= 8.0 and an sm_80+ image of the kernel
// in this build (probed once). false on the CPU-only stub.
bool imma_supported();
// The run-time switch ENGINE_GPU_IMMA (re-read on every call): unset, empty or anything but "0" = on,
// "0" = off (the rollback arm; retired in OFF builds by S24 CL-R2: OC_ENABLE_LEGACY=ON only, an OFF build refuses
// "0..." at start-up and imma_policy() is always true -- RetiredDeviceLevers.h). Only consulted for device GEMMs, which exist only under ENGINE_DEVICE=gpu|hybrid
// (or a direct call of this layer): the CPU path never reaches it. A tuning switch: never changes a value.
bool imma_policy();
// Workspace: A8 (m x Kp) then B8 = (B^T as n x Kp), Kp = round_up(k, 16), each slab rounded to 256 B.
inline uint32_t imma_padded_k(uint32_t k) { return (k + 15u) / 16u * 16u; }
inline size_t   imma_b_offset(uint32_t m, uint32_t k) { return (static_cast<size_t>(m) * imma_padded_k(k) + 255) / 256 * 256; }
inline size_t   imma_workspace_bytes(uint32_t m, uint32_t n, uint32_t k) {
    return imma_b_offset(m, k) + (static_cast<size_t>(n) * imma_padded_k(k) + 255) / 256 * 256;
}
// The K chunk a workspace of `bytes` allows: k itself when imma_workspace_bytes(m, n, k) fits,
// else the largest multiple of 64 whose workspace fits (0: not even one 64-deep chunk). A
// chunked GEMM applies op on its first chunk and Add / Sub on the others -- same bytes.
inline uint32_t imma_k_chunk(uint32_t m, uint32_t n, uint32_t k, size_t bytes) {
    if (imma_workspace_bytes(m, n, k) <= bytes) return k;
    const size_t per = static_cast<size_t>(m) + n;
    uint64_t kc = per == 0 ? 0 : (bytes / per) / 64 * 64;
    if (kc >= k) kc = (k - 1) / 64 * 64;
    while (kc >= 64 && imma_workspace_bytes(m, n, static_cast<uint32_t>(kc)) > bytes) kc -= 64;
    return static_cast<uint32_t>(kc < 64 ? 0 : kc);
}
// T(p): the k-tiles (64 deep) an int32 accumulator absorbs between two reductions mod p:
// floor((2^31 - 1 - (p - 1)) / (64 floor(p/2)^2)); 8454 at p = 127 (K = 541 056).
inline uint32_t imma_max_flush_tiles(uint32_t p) {
    const uint64_t hb = p / 2, per_tile = 64ull * hb * hb;
    if (per_tile == 0) return 0xFFFFFFFFu;
    const uint64_t t = (2147483647ull - (p - 1)) / per_tile;
    return t > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(t);
}

// C op= A * B   (A: m x k, B: k x n, C: m x n; any strides). C must not overlap A or B.
template<class Elt>
void gemm(const ModCtx<Elt>& ctx, DeviceView<const Elt> A, DeviceView<const Elt> B,
          DeviceView<Elt> C, GemmOp op, Stream s, const GemmOpts& opts = GemmOpts{});

// C op= A * B^T with B given n x k (== gemm(A, B.transposed_view(), C)).
template<class Elt>
void gemm_nt(const ModCtx<Elt>& ctx, DeviceView<const Elt> A, DeviceView<const Elt> B,
             DeviceView<Elt> C, GemmOp op, Stream s, const GemmOpts& opts = GemmOpts{});

// Schur update C -= L * U (the elimination building block; == gemm Sub).
template<class Elt>
void schur_update(const ModCtx<Elt>& ctx, DeviceView<const Elt> L, DeviceView<const Elt> U,
                  DeviceView<Elt> C, Stream s, const GemmOpts& opts = GemmOpts{});

// ---- batched / grouped small-block GEMM ------------------------------------------
// One launch for many instances of the SAME shape, strides and op (the
// Theta-assembly shape: hundreds of d x d blocks over a handful of distinct
// operands). Pointer-array batching: the instance table lives in device
// memory (uploaded once by upload(); capturable afterwards). Every instance
// writes its own C window -- no two instances may share a C window.
template<class Elt>
struct GemmBatchEntry { const Elt* A; const Elt* B; Elt* C; };

template<class Elt>
class GemmBatch {
public:
    GemmBatch() = default;
    ~GemmBatch();
    GemmBatch(const GemmBatch&) = delete;
    GemmBatch& operator=(const GemmBatch&) = delete;

    // Shape and strides shared by every instance. Call once, before push().
    void set_shape(uint32_t m, uint32_t n, uint32_t k,
                   size_t a_rs, size_t a_cs, size_t b_rs, size_t b_cs, size_t c_rs, size_t c_cs);
    // Convenience: take the shape/strides from one representative triple.
    void set_shape(DeviceView<const Elt> A, DeviceView<const Elt> B, DeviceView<Elt> C);
    void push(DeviceView<const Elt> A, DeviceView<const Elt> B, DeviceView<Elt> C);   // shape-checked
    void push(const Elt* A, const Elt* B, Elt* C) { entries_.push_back(GemmBatchEntry<Elt>{A, B, C}); }
    void clear() { entries_.clear(); uploaded_ = 0; }
    size_t size() const { return entries_.size(); }

    void upload(Stream s);            // instance table -> device (NOT capturable)
    // (slot-22 G1) After upload(): allocate the IMMA operand workspace of every instance
    // (imma_workspace_bytes(m, n, k) each) when imma_eligible(ctx) && imma_supported() &&
    // imma_policy() and it stays <= 1 GiB; returns whether the batch is IMMA-prepared. NOT capturable.
    bool prepare_imma(const ModCtx<Elt>& ctx);
    // The IMMA workspace the batch needs: size() x imma_workspace_bytes(m, n, k) (0 if k == 0).
    size_t imma_bytes_needed() const;
    // Launch (capturable once uploaded). imma as GemmOpts::imma. The IMMA route needs a workspace:
    // the caller's (imma_ws, >= imma_bytes_needed() bytes) or else the one of prepare_imma().
    void run(const ModCtx<Elt>& ctx, GemmOp op, Stream s, int tile = 0, int imma = -1,
             void* imma_ws = nullptr, size_t imma_ws_bytes = 0) const;

    uint32_t m() const { return m_; } uint32_t n() const { return n_; } uint32_t k() const { return k_; }
private:
    uint32_t m_ = 0, n_ = 0, k_ = 0;
    size_t a_rs_ = 0, a_cs_ = 0, b_rs_ = 0, b_cs_ = 0, c_rs_ = 0, c_cs_ = 0;
    bool shape_set_ = false;
    std::vector<GemmBatchEntry<Elt>> entries_;
    GemmBatchEntry<Elt>* d_entries_ = nullptr;
    size_t d_capacity_ = 0;
    size_t uploaded_ = 0;
    void*  imma_ws_ = nullptr;        // (G1) prepare_imma(): instances x imma_workspace_bytes(m, n, k)
    size_t imma_ws_bytes_ = 0;
};

// ---- level-1/2 -------------------------------------------------------------------
// y op= A * x  with x a k x 1 view and y an m x 1 view (pass x.transposed_view() for a row).
template<class Elt>
void gemv(const ModCtx<Elt>& ctx, DeviceView<const Elt> A, DeviceView<const Elt> x,
          DeviceView<Elt> y, GemmOp op, Stream s);

// y = y +/- alpha * x  (equal-shape views; alpha canonical)
template<class Elt>
void axpy(const ModCtx<Elt>& ctx, Elt alpha, DeviceView<const Elt> x, DeviceView<Elt> y,
          bool subtract, Stream s);

// x = alpha * x
template<class Elt>
void scal(const ModCtx<Elt>& ctx, Elt alpha, DeviceView<Elt> x, Stream s);

// dst(i, j) = src(i, j) (physical relayout, 32 x 32 tiles, any strides)
template<class Elt>
void copy(DeviceView<const Elt> src, DeviceView<Elt> dst, Stream s);

// dst(i, j) = src(j, i)
template<class Elt>
void transpose(DeviceView<const Elt> src, DeviceView<Elt> dst, Stream s);

// ---- tape support (Phase 5 Stage B) ----------------------------------------------
// x(i, j) = value  (the SCAL alpha == 0 zero fill of the tape; any strides)
template<class Elt>
void fill(DeviceView<Elt> x, Elt value, Stream s);

// y(i, i) = y(i, i) +/- alpha for every diagonal cell of the LOGICAL window
// (the tape's AXPY with the identity source; y need not be square: only
// cells with i == j inside the window are touched).
template<class Elt>
void axpy_identity(const ModCtx<Elt>& ctx, Elt alpha, DeviceView<Elt> y, bool subtract, Stream s);

// The tape's GATHER: c(i, t) [op]= alpha * [row_off + i == col(t)] + beta * a(i, col(t)),
// col(t) = cols[t] (a device index list of c.cols entries) or t when cols == nullptr.
// `a` may be an empty view (identity source: beta == 0); accumulate == false
// is Set. Every output cell is written by exactly one thread.
// (slot-18 WP-19a) `rows` (a device list of c.rows entries, or nullptr): the
// row-list form, c(i, t) reads a(rows[i], col(t)) and the identity test is
// row_off + rows[i] == col(t) -- the CPU twin is the executor's row-list GATHER.
template<class Elt>
void gather(const ModCtx<Elt>& ctx, Elt alpha, Elt beta, DeviceView<const Elt> a, uint32_t row_off,
            const uint32_t* cols, DeviceView<Elt> c, bool accumulate, Stream s, const uint32_t* rows = nullptr);

// Lazy2p -> Canonical in place (v >= p ? v - p : v).
template<class Elt>
void canonicalise(const ModCtx<Elt>& ctx, DeviceView<Elt> v, Stream s);

// Accum64 -> Canonical: dst(i, j) = acc(i, j) mod p (u32 modulus).
void canonicalise(const ModCtx<uint32_t>& ctx, DeviceView<const uint64_t> acc,
                  DeviceView<uint32_t> dst, Stream s);

// ---- elimination building blocks -------------------------------------------------
// B <- T^{-1} * B in place, T square unit-triangular (structure UpperUnitri or
// LowerUnitri: only the strict triangle is read, the diagonal is taken as 1).
// Blocked: 32-row diagonal solves + gemm(Sub) trailing updates; every dot
// product honours k_max. T must not overlap B.
template<class Elt>
void trsm_unit(const ModCtx<Elt>& ctx, DeviceView<const Elt> T, DeviceView<Elt> B, Stream s,
               const GemmOpts& opts = GemmOpts{});

// ---- (slot-22 G2) the device-resident elimination over F_p, p <= 127 (ENGINE_API.md Sec. 11.9) ----
// The whole RREF runs on the device: the matrix lives there as balanced int8 (row-major,
// ld = round_up(n, 32)); right-looking blocked PLU with column-order pivoting -- outer panels of
// 256 columns, cooperative multi-block bases of 32 columns (one grid barrier per column) -- every
// bulk update an IMMA GEMM with the fused int8 Set / Sub epilogue; then RREF = U1^{-1} U bottom-up
// in blocks of 256 pivot rows. Exact integer arithmetic only.
// Pivot rule: the leftmost column with a non-zero Schur-complement entry and, in it, the active row
// of SMALLEST ORIGINAL INDEX (rows are swapped, the original index travels with the row). This is the
// row-rank-profile rule of PIVOT_SEARCH (tape::rref's per-panel pivot_rows), NOT the swap order of
// DenseElim's Plu (which exposes no pivot rows, only unique objects). Why the chosen set after every
// column c is the row rank profile (RRP) of A[:, 0..c]: let P be the set chosen before c (by induction
// RRP of A[:, 0..c-1]). A row r outside P has a non-zero Schur entry at c iff A[r, 0..c] is not in
// span{A[p, 0..c] : p in P}. If the rank does not grow, no such row exists. If it grows, RRP gains
// i*, the least row outside P not in the span of the rows above it; every j outside P below i* lies
// in the span of the rows above it, hence (induction on j) in span(P), so its Schur entry is zero,
// while A[i*] is outside span(P) since P + {i*} is independent: the least non-zero entry is at i*.
// So pivot_rows is a unique function of A too.
// Domain (imma_elim_eligible): a FIELD context, uint32_t, 2 <= p <= 127. Z/ell^k is not covered (its
// elimination is the Howell form, Sec. 13.5, not this RREF).
template<class Elt>
inline bool imma_elim_eligible(const ModCtx<Elt>& ctx) { return imma_eligible(ctx) && ctx.is_field(); }
struct ImmaRrefResult {
    uint32_t rank = 0;
    std::vector<uint32_t> pivots;       // pivot columns, ascending (the column rank profile)
    std::vector<uint32_t> pivot_rows;   // pivot_rows[t]: the original row that supplied pivot t (the rule above)
};
// The rows of A, in row slabs: src(r0, nr) returns a ROW-CONTIGUOUS host view of rows [r0, r0 + nr)
// of A (nr x n, canonical residues), valid until the next call. sink(rank) returns where the
// rank x n RREF rows go (a row-contiguous rank x n view; canonical u32 written); it is called once,
// for rank > 0, while the device runs the back substitution (so its allocation overlaps it). An
// EMPTY std::function (ImmaRowSink{}) asks for rank, pivots and pivot_rows only: no back
// substitution, no download of the rows.
using ImmaRowSource = std::function<DenseView<const uint32_t>(uint32_t r0, uint32_t nr)>;
using ImmaRowSink   = std::function<DenseView<uint32_t>(uint32_t rank)>;
// Requires imma_elim_eligible(ctx) && imma_supported() (ENGINE_CHECKed). Synchronous: returns when
// the result is on the host. Thread-safe (own stream and buffers per call). src's slab height is
// at most `slab_rows` (0: about 64 MB of u32 per slab). base_width: 32 (default) or 16, a tuning
// knob (validation sweeps both); values never depend on it.
ImmaRrefResult imma_rref(const ModCtx<uint32_t>& ctx, uint32_t m, uint32_t n, const ImmaRowSource& src,
                         const ImmaRowSink& sink, uint32_t slab_rows = 0, uint32_t base_width = 32);
struct ImmaElimStats {
    uint64_t calls = 0, bases = 0, base_columns = 0, coop_launches = 0, gemms = 0, kernel_launches = 0;
    uint64_t h2d_bytes = 0, d2h_bytes = 0;
    size_t   peak_device_bytes = 0;   // the largest single call's device footprint
    double   seconds = 0.0;           // wall time inside imma_rref, summed
};
ImmaElimStats imma_elim_stats();
void          reset_imma_elim_stats();
// The device bytes imma_rref(m, n) allocates (matrix + work buffers + one staging slab).
size_t imma_rref_device_bytes(uint32_t m, uint32_t n, uint32_t slab_rows = 0);

// ---- introspection --------------------------------------------------------------
struct LaunchStats {
    uint64_t kernel_launches = 0; uint64_t gemm_launches = 0; uint64_t batched_instances = 0;
    uint64_t imma_launches = 0;       // (slot-22 G1) IMMA GEMM launches (a batched launch counts once)
};
LaunchStats launch_stats();
void        reset_launch_stats();
const char* kernel_layer_version();

} // namespace cuda
} // namespace engine
