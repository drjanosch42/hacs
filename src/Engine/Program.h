// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// Program.h -- the recorded program of block ops (Phase 4, plan D4)
// =====================================================================
//
// A Program<Elt> is a straight-line list of Nodes over program-owned
// Buffers.  It is what the Recorder produces, what the passes rewrite, and
// what the Executor replays.  Nothing in this file computes anything.
//
//   Node = (op, in-views a/b, out-view c, scalars, k_bound, cost, device)
//   Buffer = external Input / Output slot (bound at run time by the caller,
//            shape and layout fixed at record time) or a Temp (arena-backed,
//            lifetime = [first_level, last_level] in wavefront levels).
//
// OP SEMANTICS (c is the output view, a / b the inputs; "gop" is Set/Add/Sub)
//   GEMM          c gop= alpha * a . b               (alpha == 1: plain kernels::gemm)
//   BATCHED_GEMM  the same for every instance in [batch_begin, batch_end)
//   SCHUR_UPDATE  c -= a . b                         (GEMM Sub; kept as its own op so
//                                                    elimination batches stay together)
//   TRSM          c = a^{-1} . b  with the pivot factor already inverted by the
//                 host driver; pivot_kind Identity -> c = b, Scalar -> c = alpha * b
//   AXPY          c = c +/- alpha * a  (a.buf == kIdentityBuf: alpha * I on the diagonal)
//   SCAL          c = alpha * c        (alpha == 0: zero fill)
//   TRANSPOSE     c(i, j) = a(i, j) into a buffer of the OTHER physical layout
//                 (the D3 physical relayout, 16x16 tiles; a logical transpose is
//                 ViewRef::transposed and costs nothing)
//   CANONICALIZE  c in place, Lazy2p -> Canonical
//   GATHER        c(i, t) [op]= alpha * [aux + i == cols[t]] + beta * a(i, cols[t]);
//                 cols = index list `index_list` (kNoList: all columns of a); aux = the
//                 row offset of a inside its block (for the alpha * I diagonal);
//                 accumulate == false: Set.  This is BlockMat::materialise_rows_of_block
//                 / gather_block_cols on the tape: block -> flat.  With a row list
//                 `row_list` (rows[i] indexes the logical rows of a), c(i, t) reads
//                 a(rows[i], cols[t]) and the diagonal test is aux + rows[i] == cols[t]:
//                 the elimination's pivot-row gathers, one node per block (device-eligible
//                 since slot-18 WP-19a: cuda::gather's row-list form).
//   SCATTER       c = a (copy, any layouts): flat -> block windows, or a plain copy.
//   PANEL_RREF    in-place elim::rref(c); result slot_out <- {rank, pivots}
//   PIVOT_SEARCH  a = the panel BEFORE its RREF, slot_in = its pivot columns Q;
//                 slot_out.pivots <- pivot ROWS (row rank profile of a[:, Q] = first
//                 non-zero rows in original order), c[0:k, 0:k] <- a[prows, Q]
//   COMPARE       slot_out.equal <- (a == b entrywise as logical matrices)
//   H2D / D2H     transfer nodes of the residency pass (Phase 5): the whole
//                 buffer c goes host -> device / device -> host at the start of
//                 the node's level.  Inserted by compile() when a node is
//                 assigned Device::GPU; never recorded by a consumer; no-ops
//                 when the program runs without a device.
//
// Every node's output is a unique exact object (plan D4 determinism guarantee):
// the executor may tile, batch, reorder or thread it freely.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/DenseView.h"
#include "../ModArithmetic/Kernels.h"
#include "../ModArithmetic/ModCtx.h"
#include "EngineCostModel.h"

namespace engine {
namespace tape {

enum class Op : uint8_t {
    GEMM, BATCHED_GEMM, AXPY, SCAL, TRANSPOSE, CANONICALIZE,
    PANEL_RREF, PIVOT_SEARCH, TRSM, SCHUR_UPDATE, GATHER, SCATTER, COMPARE,
    H2D, D2H
};
const char* op_name(Op op);
ThreadClass thread_class(Op op);
inline bool is_transfer(Op op) { return op == Op::H2D || op == Op::D2H; }
// Ops a device backend can execute (PANEL_RREF / PIVOT_SEARCH are host-serial;
// COMPARE downloads and compares on the host -- it is a CPU node whose D2H
// the residency pass inserts).
inline bool device_eligible(Op op) {
    return !(op == Op::PANEL_RREF || op == Op::PIVOT_SEARCH || op == Op::COMPARE || is_transfer(op));
}

enum class Device  : uint8_t { CPU, GPU };
// ENGINE_DEVICE: cpu = never assign the GPU (the golden path); gpu = every
// eligible node on the GPU; hybrid = the static cost model of pass 6.
enum class DevicePolicy : uint8_t { Cpu, Gpu, Hybrid };
const char* device_policy_name(DevicePolicy p);
enum class BufKind : uint8_t { Input, Output, Temp };
enum class PivotKind : uint8_t { Identity, Scalar, General };

using BufId = uint32_t;
constexpr BufId   kNoBuf       = 0xFFFFFFFFu;
constexpr BufId   kIdentityBuf = 0xFFFFFFFEu;   // AXPY source "the identity"
constexpr uint32_t kNoSlot     = 0xFFFFFFFFu;
constexpr uint32_t kNoList     = 0xFFFFFFFFu;

// Per-node device eligibility.  Until slot-18 WP-19a a GATHER with a row list
// was CPU-only; cuda::gather now has the row-list form (its CPU twin is the
// executor's GATHER, ENGINE_VERIFY_DEVICE compares them entrywise), so every
// device-eligible op is eligible as a node.
template<class NodeT>
inline bool device_eligible_node(const NodeT& n) {
    return device_eligible(n.op);
}

struct BufferDesc {
    BufKind  kind   = BufKind::Temp;
    uint32_t rows   = 0, cols = 0;
    Layout   layout = Layout::RowMajor;
    uint32_t slot   = 0;             // Input / Output: index into the bindings
    uint64_t content_key = 0;        // CSE identity (Input: caller key; Temp: producer hash); 0 = unique
    bool     layout_fixed = false;   // Input / Output, or a Temp the caller pinned
    // Filled by the schedule pass (Temp only).
    uint32_t first_level = 0xFFFFFFFFu, last_level = 0;
    size_t   arena_offset = 0;       // elements
    size_t   footprint    = 0;       // elements (major_dim * ld)
    uint32_t ld           = 0;
};

// A window onto a program buffer (buffer coordinates; `transposed` is the
// free logical transpose).
struct ViewRef {
    BufId    buf  = kNoBuf;
    uint32_t row0 = 0, rows = 0, col0 = 0, cols = 0;
    bool     transposed = false;
    uint32_t lrows() const { return transposed ? cols : rows; }   // logical shape
    uint32_t lcols() const { return transposed ? rows : cols; }
    bool     valid() const { return buf != kNoBuf; }
    ViewRef  sub(uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc) const {   // logical coordinates
        ViewRef v = *this;
        if (transposed) { v.row0 = row0 + c0; v.rows = nc; v.col0 = col0 + r0; v.cols = nr; }
        else            { v.row0 = row0 + r0; v.rows = nr; v.col0 = col0 + c0; v.cols = nc; }
        return v;
    }
    ViewRef  transposed_view() const { ViewRef v = *this; v.transposed = !transposed; return v; }
    bool overlaps(const ViewRef& o) const {
        if (buf != o.buf) return false;
        return row0 < o.row0 + o.rows && o.row0 < row0 + rows &&
               col0 < o.col0 + o.cols && o.col0 < col0 + cols;
    }
};

template<class Elt>
struct Node {
    Op       op     = Op::GEMM;
    Device   device = Device::CPU;
    ViewRef  a, b, c;
    Elt      alpha  = Elt(1);
    Elt      beta   = Elt(0);
    // Kind-carrying GEMM (recorded by Recorder::gemm_ref, lowered by pass 1):
    // c gop= (ka1 * I + kb1 * a) . (ka2 * I + kb2 * b).  `a` / `b` may be
    // invalid when the matching kb is 0.
    bool     kinds  = false;
    Elt      ka1 = Elt(0), kb1 = Elt(0), ka2 = Elt(0), kb2 = Elt(0);
    kernels::GemmOp gop = kernels::GemmOp::Set;
    bool     subtract   = false;    // AXPY
    bool     accumulate = false;    // GATHER
    PivotKind pivot_kind = PivotKind::General;   // TRSM
    uint32_t aux        = 0;        // GATHER: row offset of `a` inside its block
    uint32_t index_list = kNoList;  // GATHER: column list
    uint32_t row_list   = kNoList;  // GATHER: row list -- c(i, t) reads a(rows[i], cols[t]) and the identity test
                                    // uses aux + rows[i] (kNoList: rows 0 .. c.lrows() - 1 of `a`); CPU-only
    uint32_t slot_in    = kNoSlot;  // PIVOT_SEARCH reads the pivots of this slot
    uint32_t slot_out   = kNoSlot;  // PANEL_RREF / PIVOT_SEARCH / COMPARE
    uint32_t k_bound    = 0;        // pass 4: products accumulated before a reduction
    double   cost       = 0.0;      // modular MACs (compute ops) or elements touched (memory ops)
    uint32_t batch_begin = 0, batch_end = 0;   // BATCHED_GEMM instances
    uint32_t level      = 0;        // wavefront (schedule pass)
    bool     dead       = false;    // removed by a pass (kept so node ids stay stable)
    uint32_t origin     = 0;        // recorder node id, for diagnostics
};

struct GemmInstance { ViewRef a, b, c; };

struct ResultSlot {
    uint32_t rank = 0;
    std::vector<uint32_t> pivots;
    bool     equal = false;
    bool     written = false;
};

struct PassStats {
    uint32_t recorded = 0, folded_scalar = 0, folded_axpy = 0, cse_removed = 0;
    uint32_t transposes_inserted = 0, ksplit_nodes = 0, batches = 0, batched_instances = 0;
    uint32_t gpu_nodes = 0, levels = 0, live_nodes = 0;
    uint32_t gpu_compute_nodes = 0;   // (slot-23 P7) the GPU nodes that are compute (GEMM-class, BATCHED_GEMM, TRSM-General)
    size_t   arena_elts = 0, temp_elts = 0;
    // Phase 5: the residency pass
    uint32_t h2d_nodes = 0, d2h_nodes = 0, hoisted_h2d = 0, forced_cpu = 0, gpu_groups = 0;
    double   h2d_bytes = 0.0, d2h_bytes = 0.0;
    double   model_t_cpu = 0.0, model_t_gpu = 0.0;   // hybrid: the model's estimate of the chosen schedule
};

struct PassOptions {
    bool fold   = true;     // pass 1 (false: kind-GEMMs are materialised and multiplied densely -- the
                            // validation's unfolded arm; never faster)
    bool cse    = true;
    bool fuse   = true;     // pass 2b: an un-shared product temp is folded back into its AXPY
    bool layout = true;
    bool ksplit = false;    // split GEMMs with k > k_max into sequential k-chunks (the CPU
                            // kernels block internally at k_max anyway; ENGINE_KSPLIT=1)
    bool batch  = true;     // ENGINE_BATCH
    bool arena  = true;     // ENGINE_ARENA (0: every Temp owns its storage)
    Device device_policy = Device::CPU;   // legacy pair (Phase 4): device_model == true means Hybrid
    bool device_model = false;
    DevicePolicy policy = DevicePolicy::Cpu;   // ENGINE_DEVICE: cpu | gpu | hybrid (Phase 5); wins over the pair when != Cpu
    DevicePolicy effective_policy() const {
        if (policy != DevicePolicy::Cpu) return policy;
        return device_model ? DevicePolicy::Hybrid : DevicePolicy::Cpu;
    }
    static PassOptions from_env();
};

template<class Elt>
struct Program {
    ModCtx<Elt>               ctx;
    std::vector<BufferDesc>   bufs;
    std::vector<Node<Elt>>    nodes;
    std::vector<GemmInstance> instances;
    std::vector<std::vector<uint32_t>> index_lists;
    uint32_t                  n_slots    = 0;      // result slots
    uint32_t                  n_inputs   = 0, n_outputs = 0;
    std::vector<std::vector<uint32_t>> levels;     // node ids per wavefront (dead nodes excluded)
    size_t                    arena_elts = 0;
    uint64_t                  structural_hash = 0; // FNV-64 over the recorded structure
    uint64_t                  next_key   = 0;      // exact content-identity counter (see Passes.cpp)
    PassStats                 stats;
    PassOptions               opts;
    bool                      compiled = false;
    uint64_t                  uid = 0;            // fresh per compile(): keys the executor's device state

    const BufferDesc& desc(BufId b) const { return bufs[b]; }
    uint32_t live_nodes() const { uint32_t n = 0; for (const auto& nd : nodes) if (!nd.dead) ++n; return n; }
    std::string summary() const;
};

// FNV-1a-64 helpers shared by the signature and the validations.
inline uint64_t fnv_init() { return 1469598103934665603ull; }
inline uint64_t fnv_mix(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; ++i) { h ^= (v >> (8 * i)) & 0xFFu; h *= 1099511628211ull; }
    return h;
}

} // namespace tape
} // namespace engine
