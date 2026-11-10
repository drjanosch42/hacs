// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// BlockMat.h -- the block-sparse matrix model (matrix-engine Phase 3, D3.2)
// =====================================================================
//
//   BlockRef     one block: alpha*I + beta*M[pool_id, window]^(T?)
//   BlockMat     block grid (prefix-sum offsets) + block-CSR pattern of BlockRefs
//   BlockView    sub-grid / transposed window onto a BlockMat (metadata only)
//   BlockBuilder bulk construction, one call per block, never per cell
//
// The contract this file pins (ENGINE_API.md Sec. 9):
//
//  * GRID.  row_off[0..nbr], col_off[0..nbc] are prefix sums of the block
//    dimensions; non-uniform dimensions and ZERO-WIDTH blocks are allowed.
//    get_entry / block lookup are O(log) (kills the O(#blocks) get_entry of
//    BlockFlatMatrix, defect #8).
//  * PATTERN.  Block-CSR: row_ptr[nbr+1], and within a block row the entries
//    sorted by block column, strictly increasing.  Absent = structural zero.
//    A BlockMat never stores a Zero-kind ref.  Iteration order is therefore a
//    pure function of the pattern (02 Sec. 6.5(5): no hash containers).
//  * KINDS.  A ref means alpha*I + beta*M^(T?) where M is the dr x dc window
//    (r0, c0) of the pool buffer (dc x dr before the transpose when T is set).
//    Scalar (beta == 0) requires a square block.  Zero / I / alpha*I / M / -M
//    / I-M / beta_lhs*I - beta_rhs*M (pencils) are all one representation;
//    kernels and the elimination constant-fold on (alpha, beta) without
//    touching data.
//  * MULTI-VECTOR = BlockMat with nbc == 1.  A basis of k vectors of a space
//    partitioned into blocks (heights d_s) is one dim x k object; the scalar
//    row index of component `c` of block `s` is row_off[s] + c, which for the
//    uniform d_V partition is exactly the binding L1 layout s*d_V + c
//    (02 Sec. 6.4).  from_flat() interns the dense dim x k buffer ONCE and
//    makes every block row a zero-copy window of it; to_flat() is the
//    inverse.  These two are the single source of truth for L1 (plan 7.20).
//  * LAYOUT.  Every pool buffer has one physical layout; refs inherit it.
//    relayout(Layout) is the explicit physical relayout op (memoised in the
//    pool).  Kernels state the layout they prefer (BlockKernels.h).
//  * IMMUTABILITY.  A BlockMat is immutable after construction; new matrices
//    come from the builder or from kernels.  Data lives in the shared
//    BlockPool (refcounted; copying a BlockMat is O(nnz) metadata + refcount).
//
// Exactness: nothing here does arithmetic beyond the ModCtx primitives used to
// materialise alpha*I + beta*M; every materialisation is a fixed function of
// the ref, so a matrix's logical entries do not depend on kind choices.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "BlockPool.h"
#include "DenseBuf.h"
#include "DenseView.h"
#include "EngineCheck.h"
#include "Kernels.h"
#include "ModCtx.h"

namespace engine {

enum class BlockKind : uint8_t { Zero, Scalar, Dense, Affine };

template<class Elt>
struct BlockRef {
    using Id = typename BlockPool<Elt>::Id;
    static constexpr uint8_t kTransposed = 1;

    BlockKind kind    = BlockKind::Zero;
    Elt       alpha   = 0;                        // coefficient of I
    Elt       beta    = 0;                        // coefficient of M
    Id        pool_id = BlockPool<Elt>::kNone;    // kNone unless Dense / Affine
    uint32_t  r0      = 0;                        // window origin in the pool buffer
    uint32_t  c0      = 0;
    uint8_t   flags   = 0;

    static BlockRef zero() { return BlockRef(); }
    static BlockRef identity() { return scalar(Elt(1)); }
    static BlockRef scalar(Elt a) { BlockRef r; r.alpha = a; r.normalize(); return r; }
    static BlockRef dense(Id id, Elt beta = Elt(1), uint32_t r0 = 0, uint32_t c0 = 0,
                          bool transposed = false) {
        BlockRef r; r.beta = beta; r.pool_id = id; r.r0 = r0; r.c0 = c0;
        r.flags = transposed ? kTransposed : 0; r.normalize(); return r;
    }
    static BlockRef affine(Elt alpha, Elt beta, Id id, uint32_t r0 = 0, uint32_t c0 = 0,
                           bool transposed = false) {
        BlockRef r; r.alpha = alpha; r.beta = beta; r.pool_id = id; r.r0 = r0; r.c0 = c0;
        r.flags = transposed ? kTransposed : 0; r.normalize(); return r;
    }

    bool is_zero() const   { return kind == BlockKind::Zero; }
    bool has_data() const  { return kind == BlockKind::Dense || kind == BlockKind::Affine; }
    bool transposed() const { return (flags & kTransposed) != 0; }
    BlockRef transposed_ref() const {
        BlockRef r = *this;
        if (r.has_data()) r.flags ^= kTransposed;
        return r;
    }
    // Same mathematical block by representation (not by value).
    bool same_rep(const BlockRef& o) const {
        return kind == o.kind && alpha == o.alpha && beta == o.beta &&
               (!has_data() || (pool_id == o.pool_id && r0 == o.r0 && c0 == o.c0 && flags == o.flags));
    }
    // Canonical kind from (alpha, beta); drops the data reference when beta == 0.
    void normalize() {
        if (beta == 0) {
            pool_id = BlockPool<Elt>::kNone; r0 = c0 = 0; flags = 0;
            kind = (alpha == 0) ? BlockKind::Zero : BlockKind::Scalar;
        } else {
            kind = (alpha == 0) ? BlockKind::Dense : BlockKind::Affine;
        }
    }
};

template<class Elt> class BlockBuilder;
template<class Elt> class BlockView;

template<class Elt>
class BlockMat {
public:
    using Pool = BlockPool<Elt>;
    using Id   = typename Pool::Id;
    using Ref  = BlockRef<Elt>;

    BlockMat() = default;
    ~BlockMat() { release_all(); }
    BlockMat(const BlockMat& o)
        : pool_(o.pool_), ctx_(o.ctx_), row_off_(o.row_off_), col_off_(o.col_off_),
          row_ptr_(o.row_ptr_), col_idx_(o.col_idx_), refs_(o.refs_), layout_hint_(o.layout_hint_) { acquire_all(); }
    BlockMat& operator=(const BlockMat& o) {
        if (this != &o) { BlockMat t(o); swap(t); }
        return *this;
    }
    BlockMat(BlockMat&& o) noexcept { swap(o); }
    BlockMat& operator=(BlockMat&& o) noexcept {
        if (this != &o) { BlockMat t; t.swap(*this); swap(o); }
        return *this;
    }
    void swap(BlockMat& o) noexcept {
        pool_.swap(o.pool_);
        std::swap(ctx_, o.ctx_);
        row_off_.swap(o.row_off_); col_off_.swap(o.col_off_);
        row_ptr_.swap(o.row_ptr_); col_idx_.swap(o.col_idx_); refs_.swap(o.refs_);
        std::swap(layout_hint_, o.layout_hint_);
    }

    // Empty pattern on a grid (all structural zeros).
    static BlockMat zeros(std::shared_ptr<Pool> pool, const std::vector<uint32_t>& row_dims,
                          const std::vector<uint32_t>& col_dims, Elt p) {
        BlockMat m;
        m.init_grid(std::move(pool), row_dims, col_dims, p);
        m.row_ptr_.assign(m.nbr() + 1, 0);
        return m;
    }
    // (hecke-extensions slot 8, WP-3; ENGINE_API.md Sec. 13.7) THE RING-CARRYING
    // OVERLOADS.  Every `Elt p` entry point above and below builds ModCtx(p) and
    // therefore declares a FIELD (Sec. 13.1); a BlockMat over Z/ell^k is built
    // only through an overload that takes the ModCtx itself, which it stores
    // unchanged (is_field() == false travels with the object).  The field
    // overloads are untouched; sizeof(BlockMat) is unchanged (the member was
    // always a full ModCtx).
    static BlockMat zeros(std::shared_ptr<Pool> pool, const std::vector<uint32_t>& row_dims,
                          const std::vector<uint32_t>& col_dims, const ModCtx<Elt>& ctx) {
        BlockMat m;
        m.init_grid(std::move(pool), row_dims, col_dims, ctx);
        m.row_ptr_.assign(m.nbr() + 1, 0);
        return m;
    }

    // ---- grid --------------------------------------------------------------
    uint32_t nbr() const  { return static_cast<uint32_t>(row_off_.empty() ? 0 : row_off_.size() - 1); }
    uint32_t nbc() const  { return static_cast<uint32_t>(col_off_.empty() ? 0 : col_off_.size() - 1); }
    uint32_t rows() const { return row_off_.empty() ? 0 : row_off_.back(); }
    uint32_t cols() const { return col_off_.empty() ? 0 : col_off_.back(); }
    const std::vector<uint32_t>& row_off() const { return row_off_; }
    const std::vector<uint32_t>& col_off() const { return col_off_; }
    std::vector<uint32_t> row_dims() const { return dims_of(row_off_); }
    std::vector<uint32_t> col_dims() const { return dims_of(col_off_); }
    uint32_t row_start(uint32_t br) const { return row_off_[br]; }
    uint32_t col_start(uint32_t bc) const { return col_off_[bc]; }
    uint32_t row_dim(uint32_t br) const { return row_off_[br + 1] - row_off_[br]; }
    uint32_t col_dim(uint32_t bc) const { return col_off_[bc + 1] - col_off_[bc]; }
    // Block row containing scalar row r (the LAST block starting at or before
    // r that has positive height; zero-height blocks never own a row).
    uint32_t block_row_of(uint32_t r) const { return block_of(row_off_, r); }
    uint32_t block_col_of(uint32_t c) const { return block_of(col_off_, c); }
    bool same_grid(const BlockMat& o) const { return row_off_ == o.row_off_ && col_off_ == o.col_off_; }

    // ---- pattern (block-CSR) --------------------------------------------------
    uint32_t nnz() const { return static_cast<uint32_t>(refs_.size()); }
    uint32_t row_begin(uint32_t br) const { return row_ptr_[br]; }
    uint32_t row_end(uint32_t br) const { return row_ptr_[br + 1]; }
    uint32_t col_at(uint32_t k) const { return col_idx_[k]; }
    const Ref& ref_at(uint32_t k) const { return refs_[k]; }
    const std::vector<uint32_t>& row_ptr() const { return row_ptr_; }
    const std::vector<uint32_t>& col_idx() const { return col_idx_; }
    const std::vector<Ref>& refs() const { return refs_; }
    // nullptr when structurally zero.  O(log row length).
    const Ref* find(uint32_t br, uint32_t bc) const {
        const uint32_t k = find_index(br, bc);
        return k == kNoIndex ? nullptr : &refs_[k];
    }
    static constexpr uint32_t kNoIndex = 0xFFFFFFFFu;
    uint32_t find_index(uint32_t br, uint32_t bc) const {
        ENGINE_CHECK(br < nbr() && bc < nbc());
        const uint32_t* b = col_idx_.data() + row_ptr_[br];
        const uint32_t* e = col_idx_.data() + row_ptr_[br + 1];
        const uint32_t* it = std::lower_bound(b, e, bc);
        if (it == e || *it != bc) return kNoIndex;
        return static_cast<uint32_t>(it - col_idx_.data());
    }
    bool is_multivector() const { return nbc() == 1; }

    const std::shared_ptr<Pool>& pool() const { return pool_; }
    Pool& pool_mut() const { return *pool_; }
    const ModCtx<Elt>& ctx() const { return ctx_; }
    Elt p() const { return ctx_.p; }

    // ---- block data ------------------------------------------------------------
    // The logical dr x dc view of M^(T?) for a data-carrying ref.
    DenseView<const Elt> data_view(const Ref& r, uint32_t dr, uint32_t dc) const {
        ENGINE_CHECK(r.has_data());
        const DenseView<const Elt> whole = pool_->view(r.pool_id);
        if (r.transposed()) return whole.sub(r.r0, dc, r.c0, dr).transposed_view();
        return whole.sub(r.r0, dr, r.c0, dc);
    }
    DenseView<const Elt> data_view(uint32_t k) const {
        const uint32_t br = block_row_of_index(k);
        return data_view(refs_[k], row_dim(br), col_dim(col_idx_[k]));
    }
    // dst (dr x dc) = (or +=) alpha*I + beta*M^(T?).
    void materialise(const Ref& r, uint32_t dr, uint32_t dc, DenseView<Elt> dst,
                     bool accumulate = false) const {
        materialise_ref(ctx_, *pool_, r, dr, dc, dst, accumulate);
    }
    static void materialise_ref(const ModCtx<Elt>& ctx, const Pool& pool, const Ref& r,
                                uint32_t dr, uint32_t dc, DenseView<Elt> dst, bool accumulate) {
        ENGINE_CHECK(dst.rows() == dr && dst.cols() == dc);
        if (dr == 0 || dc == 0) return;
        if (r.kind == BlockKind::Scalar || r.kind == BlockKind::Affine)
            ENGINE_CHECK_MSG(dr == dc, "BlockRef: alpha*I on a non-square block");
        if (!accumulate) {
            if (r.has_data()) {
                DenseView<const Elt> M = r.transposed()
                    ? pool.view(r.pool_id).sub(r.r0, dc, r.c0, dr).transposed_view()
                    : pool.view(r.pool_id).sub(r.r0, dr, r.c0, dc);
                kernels::copy<Elt>(M, dst);
                if (r.beta != 1) kernels::scal<Elt>(ctx, r.beta, dst);
            } else {
                zero_view(dst);
            }
            if (r.alpha != 0)
                for (uint32_t i = 0; i < dr; ++i) dst.at(i, i) = ctx.add(dst.at(i, i), r.alpha);
            return;
        }
        if (r.has_data()) {
            DenseView<const Elt> M = r.transposed()
                ? pool.view(r.pool_id).sub(r.r0, dc, r.c0, dr).transposed_view()
                : pool.view(r.pool_id).sub(r.r0, dr, r.c0, dc);
            kernels::axpy<Elt>(ctx, r.beta, M, dst, false);
        }
        if (r.alpha != 0)
            for (uint32_t i = 0; i < dr; ++i) dst.at(i, i) = ctx.add(dst.at(i, i), r.alpha);
    }
    // dst ((hi-lo) x dc) = rows [lo, hi) of the dr x dc block alpha*I + beta*M^(T?).
    void materialise_rows_of_block(const Ref& r, uint32_t dr, uint32_t dc, uint32_t lo, uint32_t hi,
                                   DenseView<Elt> dst) const {
        ENGINE_CHECK(lo <= hi && hi <= dr && dst.rows() == hi - lo && dst.cols() == dc);
        if (lo == 0 && hi == dr) { materialise(r, dr, dc, dst, false); return; }
        if (hi == lo || dc == 0) return;
        if (r.has_data()) {
            kernels::copy<Elt>(data_view(r, dr, dc).sub(lo, hi - lo, 0, dc), dst);
            if (r.beta != 1) kernels::scal<Elt>(ctx_, r.beta, dst);
        } else {
            zero_view(dst);
        }
        if (r.alpha != 0)
            for (uint32_t i = lo; i < hi && i < dc; ++i)
                dst.at(i - lo, i) = ctx_.add(dst.at(i - lo, i), r.alpha);
    }
    // dst (dr x (hi-lo)) = columns [lo, hi) of the block.
    void materialise_cols_of_block(const Ref& r, uint32_t dr, uint32_t dc, uint32_t lo, uint32_t hi,
                                   DenseView<Elt> dst) const {
        ENGINE_CHECK(lo <= hi && hi <= dc && dst.cols() == hi - lo && dst.rows() == dr);
        if (lo == 0 && hi == dc) { materialise(r, dr, dc, dst, false); return; }
        if (hi == lo || dr == 0) return;
        if (r.has_data()) {
            kernels::copy<Elt>(data_view(r, dr, dc).sub(0, dr, lo, hi - lo), dst);
            if (r.beta != 1) kernels::scal<Elt>(ctx_, r.beta, dst);
        } else {
            zero_view(dst);
        }
        if (r.alpha != 0)
            for (uint32_t j = lo; j < hi && j < dr; ++j)
                dst.at(j, j - lo) = ctx_.add(dst.at(j, j - lo), r.alpha);
    }
    static Elt entry_of(const ModCtx<Elt>& ctx, const Pool& pool, const Ref& r,
                        uint32_t i, uint32_t j) {
        Elt v = 0;
        if (r.has_data()) {
            const DenseView<const Elt> whole = pool.view(r.pool_id);
            const Elt m = r.transposed() ? whole.at(r.r0 + j, r.c0 + i) : whole.at(r.r0 + i, r.c0 + j);
            v = (r.beta == 1) ? m : ctx.mul(r.beta, m);
        }
        if (i == j && r.alpha != 0) v = ctx.add(v, r.alpha);
        return v;
    }
    Elt get_entry(uint32_t r, uint32_t c) const {
        ENGINE_CHECK(r < rows() && c < cols());
        const uint32_t br = block_row_of(r), bc = block_col_of(c);
        const Ref* ref = find(br, bc);
        if (ref == nullptr) return 0;
        return entry_of(ctx_, *pool_, *ref, r - row_off_[br], c - col_off_[bc]);
    }
    static void zero_view(DenseView<Elt> v) {
        const auto s = v.strides();
        for (uint32_t i = 0; i < v.rows(); ++i)
            for (uint32_t j = 0; j < v.cols(); ++j) s.base[i * s.rs + j * s.cs] = 0;
    }

    // ---- conversions ----------------------------------------------------------
    // dst (rows x cols, any layout) = the whole matrix.
    void to_dense(DenseView<Elt> dst) const {
        ENGINE_CHECK(dst.rows() == rows() && dst.cols() == cols());
        zero_view(dst);
        for (uint32_t br = 0; br < nbr(); ++br) {
            const uint32_t dr = row_dim(br);
            if (dr == 0) continue;
            for (uint32_t k = row_ptr_[br]; k < row_ptr_[br + 1]; ++k) {
                const uint32_t bc = col_idx_[k];
                const uint32_t dc = col_dim(bc);
                if (dc == 0) continue;
                materialise(refs_[k], dr, dc, dst.sub(row_off_[br], dr, col_off_[bc], dc), false);
            }
        }
    }
    DenseBuf<Elt> to_dense(Layout layout = Layout::RowMajor) const {
        DenseBuf<Elt> b(rows(), cols(), layout);
        to_dense(view(b));
        return b;
    }
    // The L1 pair.  to_flat: nbc == 1; the flat row index of component c of
    // block s is row_off[s] + c.  from_flat: one interned buffer, one window
    // per block row; all-zero windows become structural zeros and (square)
    // alpha*I windows Scalar refs when `detect` is set.
    void to_flat(DenseView<Elt> dst) const {
        ENGINE_CHECK_MSG(nbc() == 1, "BlockMat::to_flat: not a multi-vector (nbc != 1)");
        to_dense(dst);
    }
    DenseBuf<Elt> to_flat(Layout layout = Layout::ColMajor) const {
        ENGINE_CHECK_MSG(nbc() == 1, "BlockMat::to_flat: not a multi-vector (nbc != 1)");
        return to_dense(layout);
    }
    static BlockMat from_flat(std::shared_ptr<Pool> pool, DenseView<const Elt> src,
                              const std::vector<uint32_t>& row_dims, Elt p,
                              Layout layout = Layout::ColMajor, bool detect = true) {
        return from_dense(std::move(pool), src, row_dims, std::vector<uint32_t>{src.cols()}, p, layout, detect);
    }
    static BlockMat from_flat(std::shared_ptr<Pool> pool, DenseBuf<Elt>&& src,
                              const std::vector<uint32_t>& row_dims, Elt p, bool detect = true) {
        const uint32_t k = src.cols;
        return from_dense(std::move(pool), std::move(src), row_dims, std::vector<uint32_t>{k}, p, detect);
    }
    // (slot-8 WP-3) the ring-carrying overload (Sec. 13.7).
    static BlockMat from_flat(std::shared_ptr<Pool> pool, DenseBuf<Elt>&& src,
                              const std::vector<uint32_t>& row_dims, const ModCtx<Elt>& ctx, bool detect = true) {
        const uint32_t k = src.cols;
        return from_dense(std::move(pool), std::move(src), row_dims, std::vector<uint32_t>{k}, ctx, detect);
    }
    // General re-blocker: one interned buffer, one window per (br, bc).
    static BlockMat from_dense(std::shared_ptr<Pool> pool, DenseView<const Elt> src,
                               const std::vector<uint32_t>& row_dims, const std::vector<uint32_t>& col_dims,
                               Elt p, Layout layout = Layout::RowMajor, bool detect = true) {
        BlockMat m;
        m.init_grid(std::move(pool), row_dims, col_dims, p);
        ENGINE_CHECK_MSG(m.rows() == src.rows() && m.cols() == src.cols(),
                         "BlockMat::from_dense: partition does not match the view");
        m.row_ptr_.assign(m.nbr() + 1, 0);
        if (m.rows() == 0 || m.cols() == 0) return m;
        const Id id = m.pool_->intern_view(src, layout);
        m.window_all(id, detect);
        return m;
    }
    // Same, interning the buffer by MOVE (no copy; keeps its layout).
    static BlockMat from_dense(std::shared_ptr<Pool> pool, DenseBuf<Elt>&& src,
                               const std::vector<uint32_t>& row_dims, const std::vector<uint32_t>& col_dims,
                               Elt p, bool detect = true) {
        return from_dense(std::move(pool), std::move(src), row_dims, col_dims, ModCtx<Elt>(p), detect);
    }
    // (slot-8 WP-3) the ring-carrying overload (Sec. 13.7); the Elt p form
    // above is this one on ModCtx(p), i.e. exactly its previous body.
    static BlockMat from_dense(std::shared_ptr<Pool> pool, DenseBuf<Elt>&& src,
                               const std::vector<uint32_t>& row_dims, const std::vector<uint32_t>& col_dims,
                               const ModCtx<Elt>& ctx, bool detect = true) {
        BlockMat m;
        m.init_grid(std::move(pool), row_dims, col_dims, ctx);
        ENGINE_CHECK_MSG(m.rows() == src.rows && m.cols() == src.cols,
                         "BlockMat::from_dense: partition does not match the buffer");
        m.row_ptr_.assign(m.nbr() + 1, 0);
        if (m.rows() == 0 || m.cols() == 0) return m;
        const Id id = m.pool_->intern(std::move(src));
        m.window_all(id, detect);
        return m;
    }
private:
    // Every (br, bc) of the grid becomes a window of pool buffer `id`.
    void window_all(Id id, bool detect) {
        BlockMat& m = *this;
        const DenseView<const Elt> whole = m.pool_->view(id);
        for (uint32_t br = 0; br < m.nbr(); ++br) {
            const uint32_t dr = m.row_dim(br);
            for (uint32_t bc = 0; bc < m.nbc(); ++bc) {
                const uint32_t dc = m.col_dim(bc);
                if (dr == 0 || dc == 0) continue;
                Ref ref = Ref::dense(id, Elt(1), m.row_off_[br], m.col_off_[bc]);
                if (detect) {
                    const DenseView<const Elt> w = whole.sub(m.row_off_[br], dr, m.col_off_[bc], dc);
                    Elt a;
                    if (is_zero_view(w)) continue;
                    if (dr == dc && is_scalar_view(w, a)) ref = Ref::scalar(a);
                }
                m.col_idx_.push_back(bc);
                m.refs_.push_back(ref);
            }
            m.row_ptr_[br + 1] = static_cast<uint32_t>(m.refs_.size());
        }
        m.acquire_all();
        m.pool_->sweep_if_unreferenced(id);
    }
public:
    // A multi-vector produced by from_flat / kernels with a single backing
    // window: the dense dim x k view of the whole object, no copy.  Returns
    // false when the blocks are not all windows of one buffer in grid order
    // (structural zeros or Scalar refs break contiguity).
    bool contiguous_view(DenseView<const Elt>& out) const {
        if (nbc() != 1 || nnz() == 0 || cols() == 0) return false;
        const Ref& f = refs_[0];
        if (!f.has_data() || f.transposed() || f.alpha != 0 || f.beta != 1) return false;
        const uint32_t base_r0 = f.r0 - row_off_[block_row_of_index(0)];
        for (uint32_t br = 0; br < nbr(); ++br) {
            if (row_dim(br) == 0) { if (row_ptr_[br + 1] != row_ptr_[br]) return false; continue; }
            if (row_ptr_[br + 1] != row_ptr_[br] + 1) return false;
            const Ref& r = refs_[row_ptr_[br]];
            if (!r.has_data() || r.transposed() || r.alpha != 0 || r.beta != 1) return false;
            if (r.pool_id != f.pool_id || r.c0 != f.c0 || r.r0 != base_r0 + row_off_[br]) return false;
        }
        out = pool_->view(f.pool_id).sub(base_r0, rows(), f.c0, cols());
        return true;
    }

    // ---- structural ops (metadata only) --------------------------------------
    // Transpose: swapped partitions, every ref's transposed flag flipped, CSR
    // rebuilt by a counting sort (O(nnz), deterministic).
    BlockMat transposed() const {
        BlockMat t;
        t.pool_ = pool_; t.ctx_ = ctx_;
        t.row_off_ = col_off_; t.col_off_ = row_off_;
        const uint32_t tn = t.nbr();
        t.row_ptr_.assign(tn + 1, 0);
        for (uint32_t c : col_idx_) ++t.row_ptr_[c + 1];
        for (uint32_t i = 0; i < tn; ++i) t.row_ptr_[i + 1] += t.row_ptr_[i];
        t.col_idx_.assign(nnz(), 0);
        t.refs_.assign(nnz(), Ref());
        std::vector<uint32_t> fill(t.row_ptr_.begin(), t.row_ptr_.end() - 1);
        for (uint32_t br = 0; br < nbr(); ++br)
            for (uint32_t k = row_ptr_[br]; k < row_ptr_[br + 1]; ++k) {
                const uint32_t pos = fill[col_idx_[k]]++;
                t.col_idx_[pos] = br;
                t.refs_[pos] = refs_[k].transposed_ref();
            }
        t.acquire_all();
        return t;
    }
    // Explicit physical relayout of every data block (memoised in the pool):
    // the returned matrix's data views all have the requested physical layout.
    // Values are unchanged by construction.
    BlockMat relayout(Layout want) const {
        BlockMat m(*this);
        // The pool twin is a distinct buffer with its own id?  No: the twin is
        // kept inside the entry, so refs are unchanged and data_view_l() picks
        // the twin.  We only force the twins to exist here.
        for (const Ref& r : m.refs_)
            if (r.has_data()) (void)m.pool_->view(r.pool_id, want);
        m.layout_hint_ = want;
        return m;
    }
    // Like data_view but in the physical layout the caller wants (pool twin).
    DenseView<const Elt> data_view_l(const Ref& r, uint32_t dr, uint32_t dc, Layout want) const {
        ENGINE_CHECK(r.has_data());
        const DenseView<const Elt> whole = pool_->view(r.pool_id, want);
        if (r.transposed()) return whole.sub(r.r0, dc, r.c0, dr).transposed_view();
        return whole.sub(r.r0, dr, r.c0, dc);
    }

    // ---- small helpers used by kernels / builder --------------------------------
    static bool is_zero_view(DenseView<const Elt> v) {
        const auto s = v.strides();
        for (uint32_t i = 0; i < v.rows(); ++i)
            for (uint32_t j = 0; j < v.cols(); ++j)
                if (s.base[i * s.rs + j * s.cs] != 0) return false;
        return true;
    }
    // true iff v == a*I (square); a returned.
    static bool is_scalar_view(DenseView<const Elt> v, Elt& a) {
        if (v.rows() != v.cols() || v.rows() == 0) return false;
        const auto s = v.strides();
        a = s.base[0];
        for (uint32_t i = 0; i < v.rows(); ++i)
            for (uint32_t j = 0; j < v.cols(); ++j)
                if (s.base[i * s.rs + j * s.cs] != (i == j ? a : Elt(0))) return false;
        return true;
    }
    uint32_t block_row_of_index(uint32_t k) const {
        // binary search in row_ptr_
        auto it = std::upper_bound(row_ptr_.begin(), row_ptr_.end(), k);
        return static_cast<uint32_t>(it - row_ptr_.begin()) - 1;
    }

private:
    friend class BlockBuilder<Elt>;
    friend class BlockView<Elt>;

    static std::vector<uint32_t> dims_of(const std::vector<uint32_t>& off) {
        std::vector<uint32_t> d;
        if (off.empty()) return d;
        d.resize(off.size() - 1);
        for (size_t i = 0; i + 1 < off.size(); ++i) d[i] = off[i + 1] - off[i];
        return d;
    }
    static uint32_t block_of(const std::vector<uint32_t>& off, uint32_t x) {
        // last index i with off[i] <= x among blocks of positive extent
        auto it = std::upper_bound(off.begin(), off.end(), x);
        uint32_t i = static_cast<uint32_t>(it - off.begin()) - 1;
        return i;
    }
    void init_grid(std::shared_ptr<Pool> pool, const std::vector<uint32_t>& row_dims,
                   const std::vector<uint32_t>& col_dims, Elt p) {
        init_grid(std::move(pool), row_dims, col_dims, ModCtx<Elt>(p));
    }
    // (slot-8 WP-3) Stores the given context unchanged -- the Elt p form is
    // this one on ModCtx(p), a value-identical copy of what it built before.
    void init_grid(std::shared_ptr<Pool> pool, const std::vector<uint32_t>& row_dims,
                   const std::vector<uint32_t>& col_dims, const ModCtx<Elt>& ctx) {
        ENGINE_CHECK_MSG(pool != nullptr, "BlockMat: null pool");
        pool_ = std::move(pool);
        ctx_ = ctx;
        row_off_.assign(row_dims.size() + 1, 0);
        for (size_t i = 0; i < row_dims.size(); ++i) row_off_[i + 1] = row_off_[i] + row_dims[i];
        col_off_.assign(col_dims.size() + 1, 0);
        for (size_t i = 0; i < col_dims.size(); ++i) col_off_[i + 1] = col_off_[i] + col_dims[i];
    }
    void acquire_all() { for (const Ref& r : refs_) if (r.has_data()) pool_->acquire(r.pool_id); }
    void release_all() {
        if (!pool_) return;
        for (const Ref& r : refs_) if (r.has_data()) pool_->release(r.pool_id);
        refs_.clear();
    }

    std::shared_ptr<Pool>  pool_;
    ModCtx<Elt>            ctx_;
    std::vector<uint32_t>  row_off_, col_off_;
    std::vector<uint32_t>  row_ptr_, col_idx_;
    std::vector<Ref>       refs_;
    Layout                 layout_hint_ = Layout::RowMajor;
};

// ---------------------------------------------------------------------------
// BlockView: a sub-grid [br0, br0+nbr) x [bc0, bc0+nbc) of a BlockMat, optionally
// transposed -- metadata only.  find()/get_entry() resolve through the parent;
// to_block_mat() materialises the view as a BlockMat (O(nnz) metadata, shares
// the pool blocks).
// ---------------------------------------------------------------------------
template<class Elt>
class BlockView {
public:
    using Ref = BlockRef<Elt>;
    BlockView(const BlockMat<Elt>& m) : m_(&m), br0_(0), nbr_(m.nbr()), bc0_(0), nbc_(m.nbc()), t_(false) {}
    BlockView(const BlockMat<Elt>& m, uint32_t br0, uint32_t nbr, uint32_t bc0, uint32_t nbc, bool transposed = false)
        : m_(&m), br0_(br0), nbr_(nbr), bc0_(bc0), nbc_(nbc), t_(transposed) {
        ENGINE_CHECK_MSG(static_cast<uint64_t>(br0) + nbr <= m.nbr() && static_cast<uint64_t>(bc0) + nbc <= m.nbc(),
                         "BlockView: window exceeds grid");
    }
    const BlockMat<Elt>& parent() const { return *m_; }
    bool transposed() const { return t_; }
    uint32_t nbr() const { return t_ ? nbc_ : nbr_; }
    uint32_t nbc() const { return t_ ? nbr_ : nbc_; }
    uint32_t row_dim(uint32_t br) const { return t_ ? m_->col_dim(bc0_ + br) : m_->row_dim(br0_ + br); }
    uint32_t col_dim(uint32_t bc) const { return t_ ? m_->row_dim(br0_ + bc) : m_->col_dim(bc0_ + bc); }
    std::vector<uint32_t> row_dims() const { std::vector<uint32_t> d(nbr()); for (uint32_t i = 0; i < nbr(); ++i) d[i] = row_dim(i); return d; }
    std::vector<uint32_t> col_dims() const { std::vector<uint32_t> d(nbc()); for (uint32_t i = 0; i < nbc(); ++i) d[i] = col_dim(i); return d; }
    uint32_t rows() const { uint32_t s = 0; for (uint32_t i = 0; i < nbr(); ++i) s += row_dim(i); return s; }
    uint32_t cols() const { uint32_t s = 0; for (uint32_t i = 0; i < nbc(); ++i) s += col_dim(i); return s; }
    // The ref at view position (br, bc), transposed flag applied.  Zero ref when absent.
    Ref at(uint32_t br, uint32_t bc) const {
        const Ref* r = t_ ? m_->find(br0_ + bc, bc0_ + br) : m_->find(br0_ + br, bc0_ + bc);
        if (r == nullptr) return Ref::zero();
        return t_ ? r->transposed_ref() : *r;
    }
    BlockView sub(uint32_t br0, uint32_t nbr, uint32_t bc0, uint32_t nbc) const {
        if (t_) return BlockView(*m_, br0_ + bc0, nbc, bc0_ + br0, nbr, true);
        return BlockView(*m_, br0_ + br0, nbr, bc0_ + bc0, nbc, false);
    }
    BlockView transposed_view() const { return BlockView(*m_, br0_, nbr_, bc0_, nbc_, !t_); }
    BlockMat<Elt> to_block_mat() const {
        if (!t_) {
            BlockMat<Elt> out = BlockMat<Elt>::zeros(m_->pool(), row_dims(), col_dims(), m_->ctx());
            for (uint32_t br = 0; br < nbr_; ++br) {
                const uint32_t pr = br0_ + br;
                for (uint32_t k = m_->row_begin(pr); k < m_->row_end(pr); ++k) {
                    const uint32_t c = m_->col_at(k);
                    if (c < bc0_ || c >= bc0_ + nbc_) continue;
                    out.col_idx_.push_back(c - bc0_);
                    out.refs_.push_back(m_->ref_at(k));
                }
                out.row_ptr_[br + 1] = static_cast<uint32_t>(out.refs_.size());
            }
            out.acquire_all();
            return out;
        }
        return BlockView(*m_, br0_, nbr_, bc0_, nbc_, false).to_block_mat().transposed();
    }
private:
    const BlockMat<Elt>* m_;
    uint32_t br0_, nbr_, bc0_, nbc_;
    bool t_;
};

// ---------------------------------------------------------------------------
// BlockBuilder: declare the grid, set / accumulate whole blocks, finalize.
// Rows are kept sorted by block column on insertion (binary search); set()
// overwrites, accumulate() folds kinds when the refs share their data window
// and otherwise materialises the sum into a new interned block.  finalize()
// moves the pattern into a BlockMat (deterministic CSR).  Never per cell.
// ---------------------------------------------------------------------------
template<class Elt>
class BlockBuilder {
public:
    using Pool = BlockPool<Elt>;
    using Id   = typename Pool::Id;
    using Ref  = BlockRef<Elt>;

    BlockBuilder(std::shared_ptr<Pool> pool, const std::vector<uint32_t>& row_dims,
                 const std::vector<uint32_t>& col_dims, Elt p)
        : BlockBuilder(std::move(pool), row_dims, col_dims, ModCtx<Elt>(p)) {}
    // (hecke-extensions slot 8, WP-3; ENGINE_API.md Sec. 13.7) The ring-
    // carrying overload: the builder -- and every BlockMat it finalizes --
    // keeps `ctx` unchanged, so a Z/ell^k context (ModCtx::prime_power) stays
    // composite and the Sec. 13.3 guards of blk::rref / tape::rref see it.
    BlockBuilder(std::shared_ptr<Pool> pool, const std::vector<uint32_t>& row_dims,
                 const std::vector<uint32_t>& col_dims, const ModCtx<Elt>& ctx)
        : pool_(std::move(pool)), ctx_(ctx) {
        ENGINE_CHECK_MSG(pool_ != nullptr, "BlockBuilder: null pool");
        row_dims_ = row_dims; col_dims_ = col_dims;
        rows_.assign(row_dims.size(), {});
    }
    ~BlockBuilder() {
        for (auto& row : rows_) for (auto& e : row) if (e.second.has_data()) pool_->release(e.second.pool_id);
    }
    BlockBuilder(const BlockBuilder&) = delete;
    BlockBuilder& operator=(const BlockBuilder&) = delete;

    uint32_t nbr() const { return static_cast<uint32_t>(row_dims_.size()); }
    uint32_t nbc() const { return static_cast<uint32_t>(col_dims_.size()); }
    uint32_t row_dim(uint32_t br) const { return row_dims_[br]; }
    uint32_t col_dim(uint32_t bc) const { return col_dims_[bc]; }
    const std::shared_ptr<Pool>& pool() const { return pool_; }
    const ModCtx<Elt>& ctx() const { return ctx_; }

    // Intern a dense block (copied into the pool in `layout`), returning a
    // ref of the right kind: Zero for an all-zero block, Scalar for a square
    // alpha*I (when detect), else Dense with coefficient 1.
    Ref intern_dense(DenseView<const Elt> M, Layout layout = Layout::RowMajor, bool detect = true) {
        if (detect) {
            Elt a;
            if (BlockMat<Elt>::is_zero_view(M)) return Ref::zero();
            if (BlockMat<Elt>::is_scalar_view(M, a)) return Ref::scalar(a);
        }
        return Ref::dense(pool_->intern_view(M, layout));
    }
    Ref intern_dense(const std::vector<Elt>& row_major, uint32_t rows, uint32_t cols,
                     Layout layout = Layout::RowMajor, bool detect = true) {
        ENGINE_CHECK(row_major.size() == static_cast<size_t>(rows) * cols);
        DenseBuf<Elt> tmp(const_cast<Elt*>(row_major.data()), row_major.size() == 0 ? 1 : row_major.size(),
                          rows, cols, Layout::RowMajor);
        return intern_dense(cview(tmp), layout, detect);
    }

    // set: overwrite the block at (br, bc).  Zero refs erase.
    void set(uint32_t br, uint32_t bc, const Ref& r) {
        check_ref(br, bc, r);
        auto& row = rows_[br];
        auto it = std::lower_bound(row.begin(), row.end(), bc,
                                   [](const std::pair<uint32_t, Ref>& e, uint32_t c) { return e.first < c; });
        // Acquire the new ref BEFORE releasing the old one: replacing a block
        // by another ref on the same pool id must not drop its count to 0.
        if (r.has_data()) pool_->acquire(r.pool_id);
        if (it != row.end() && it->first == bc) {
            if (it->second.has_data()) pool_->release(it->second.pool_id);
            if (r.is_zero()) { row.erase(it); return; }
            it->second = r;
        } else {
            if (r.is_zero()) return;
            row.insert(it, std::make_pair(bc, r));
        }
    }
    void set_scalar(uint32_t br, uint32_t bc, Elt a) { set(br, bc, Ref::scalar(a)); }
    // block = alpha*I + beta*M  (M interned; Zero when the result is zero).
    void set_dense(uint32_t br, uint32_t bc, DenseView<const Elt> M, Elt beta = Elt(1), Elt alpha = Elt(0),
                   Layout layout = Layout::RowMajor, bool detect = true) {
        Ref r = intern_dense(M, layout, detect);
        set(br, bc, compose(r, alpha, beta));
    }
    void set_dense(uint32_t br, uint32_t bc, const std::vector<Elt>& row_major, Elt beta = Elt(1), Elt alpha = Elt(0),
                   Layout layout = Layout::RowMajor, bool detect = true) {
        Ref r = intern_dense(row_major, row_dims_[br], col_dims_[bc], layout, detect);
        set(br, bc, compose(r, alpha, beta));
    }
    const Ref* get(uint32_t br, uint32_t bc) const {
        const auto& row = rows_[br];
        auto it = std::lower_bound(row.begin(), row.end(), bc,
                                   [](const std::pair<uint32_t, Ref>& e, uint32_t c) { return e.first < c; });
        if (it == row.end() || it->first != bc) return nullptr;
        return &it->second;
    }
    // accumulate: block += r.  Folds when both refs share the data window (or
    // either has none); otherwise the sum is materialised and interned.
    void accumulate(uint32_t br, uint32_t bc, const Ref& r) {
        check_ref(br, bc, r);
        if (r.is_zero()) return;
        const Ref* cur = get(br, bc);
        if (cur == nullptr) { set(br, bc, r); return; }
        Ref sum;
        if (fold_sum(*cur, r, sum)) { set(br, bc, sum); return; }
        const uint32_t dr = row_dims_[br], dc = col_dims_[bc];
        DenseBuf<Elt> tmp(dr, dc);
        BlockMat<Elt>::materialise_ref(ctx_, *pool_, *cur, dr, dc, view(tmp), false);
        BlockMat<Elt>::materialise_ref(ctx_, *pool_, r, dr, dc, view(tmp), true);
        set(br, bc, intern_dense(cview(tmp), pool_layout_of(*cur), true));
    }

    // alpha_out*I + beta_out*M for a ref r = a*I + b*M: compose(r, alpha, beta) = alpha*I + beta*r.
    Ref compose(const Ref& r, Elt alpha, Elt beta) const {
        Ref o = r;
        o.alpha = ctx_.add(alpha, ctx_.mul(beta, r.alpha));
        o.beta = ctx_.mul(beta, r.beta);
        o.normalize();
        return o;
    }
    // (a1 I + b1 M) + (a2 I + b2 M') folds iff M-windows coincide or one side has none.
    bool fold_sum(const Ref& x, const Ref& y, Ref& out) const {
        const bool same_data = x.has_data() && y.has_data() &&
            x.pool_id == y.pool_id && x.r0 == y.r0 && x.c0 == y.c0 && x.flags == y.flags;
        if (!x.has_data() || !y.has_data() || same_data) {
            out = x.has_data() ? x : y;
            out.alpha = ctx_.add(x.alpha, y.alpha);
            out.beta = ctx_.add(x.beta, y.beta);
            out.normalize();
            return true;
        }
        return false;
    }

    uint32_t nnz() const { size_t n = 0; for (const auto& r : rows_) n += r.size(); return static_cast<uint32_t>(n); }

    // Moves the pattern out; the builder is empty afterwards.
    BlockMat<Elt> finalize() {
        BlockMat<Elt> m;
        m.init_grid(pool_, row_dims_, col_dims_, ctx_);
        m.row_ptr_.assign(nbr() + 1, 0);
        m.col_idx_.reserve(nnz());
        m.refs_.reserve(nnz());
        for (uint32_t br = 0; br < nbr(); ++br) {
            for (const auto& e : rows_[br]) { m.col_idx_.push_back(e.first); m.refs_.push_back(e.second); }
            m.row_ptr_[br + 1] = static_cast<uint32_t>(m.refs_.size());
            rows_[br].clear();   // refcounts transfer to m
        }
        return m;
    }

private:
    void check_ref(uint32_t br, uint32_t bc, const Ref& r) const {
        ENGINE_CHECK_MSG(br < nbr() && bc < nbc(), "BlockBuilder: block index out of range");
        if (r.kind == BlockKind::Scalar || r.kind == BlockKind::Affine)
            ENGINE_CHECK_MSG(row_dims_[br] == col_dims_[bc], "BlockBuilder: alpha*I on a non-square block");
        if (r.has_data()) {
            ENGINE_CHECK_MSG(pool_->is_live(r.pool_id), "BlockBuilder: dead pool id");
            const DenseBuf<Elt>& b = pool_->buf(r.pool_id);
            const uint32_t wr = r.transposed() ? col_dims_[bc] : row_dims_[br];
            const uint32_t wc = r.transposed() ? row_dims_[br] : col_dims_[bc];
            ENGINE_CHECK_MSG(static_cast<uint64_t>(r.r0) + wr <= b.rows && static_cast<uint64_t>(r.c0) + wc <= b.cols,
                             "BlockBuilder: block window exceeds pool buffer");
        }
    }
    Layout pool_layout_of(const Ref& r) const { return r.has_data() ? pool_->buf(r.pool_id).layout : Layout::RowMajor; }

    std::shared_ptr<Pool> pool_;
    ModCtx<Elt> ctx_;
    std::vector<uint32_t> row_dims_, col_dims_;
    std::vector<std::vector<std::pair<uint32_t, Ref>>> rows_;
};

using BlockMat32 = BlockMat<uint32_t>;
using BlockMat64 = BlockMat<uint64_t>;

} // namespace engine
