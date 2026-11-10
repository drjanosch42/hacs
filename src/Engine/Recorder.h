// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// Recorder.h -- records a Program<Elt> (Phase 4, plan D4)
// =====================================================================
//
// The consumer declares buffers (external Input / Output slots and Temps),
// derives windows (ViewRef) and appends ops.  finish() hands back the
// Program together with the default Bindings (the views the inputs and
// outputs were declared with).  Recording is O(#ops) and does no arithmetic.
//
// Block-level helpers (gemm_ref, record_gemm_dense, record_gemm_block,
// record_gather_block) translate BlockMat / BlockRef structure into nodes;
// the kind-folding itself is pass 1, so a recorded kind-GEMM stays one node
// until the passes run.

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "../ModArithmetic/BlockMat.h"
#include "../ModArithmetic/EngineCheck.h"
#include "Program.h"

namespace engine {
namespace tape {

template<class Elt>
struct Bindings {
    std::vector<DenseView<const Elt>> inputs;
    std::vector<DenseView<Elt>>       outputs;
    // Optional modulus override for a replay under another prime of the SAME
    // regime (plan D4: "every ell at fixed (n_sym, p, r, q)"); p == 0 keeps the
    // program's own context.
    ModCtx<Elt> ctx;
    bool has_ctx = false;
    void set_ctx(const ModCtx<Elt>& c) { ctx = c; has_ctx = true; }
};

// Layout / leading dimension of a plain (ld-based) view, as the executor
// will rebind it.  Every DenseView has one unit stride.
struct ViewShape { uint32_t rows, cols; Layout layout; uint32_t ld; };
template<class E>
ViewShape view_shape(const DenseView<E>& v) {
    const auto s = v.strides();
    ViewShape sh{v.rows(), v.cols(), Layout::RowMajor, 0};
    if (s.cs == 1 && (s.rs != 1 || v.rows() <= 1 || v.cols() > 1)) {
        // (slot-14 WP-3, G2) the tape's ld is uint32: refuse, never wrap.
        ENGINE_CHECK_MSG(s.rs <= UINT32_MAX, "tape: a bound view's row stride exceeds the uint32 ld domain");
        sh.layout = Layout::RowMajor; sh.ld = static_cast<uint32_t>(s.rs);
        if (v.rows() <= 1) sh.ld = v.cols();
    } else if (s.rs == 1) {
        ENGINE_CHECK_MSG(s.cs <= UINT32_MAX, "tape: a bound view's column stride exceeds the uint32 ld domain");
        sh.layout = Layout::ColMajor; sh.ld = static_cast<uint32_t>(s.cs);
        if (v.cols() <= 1) sh.ld = v.rows();
    } else {
        ENGINE_CHECK_MSG(false, "tape: a bound view must have a unit stride");
    }
    if (sh.ld == 0) sh.ld = (sh.layout == Layout::RowMajor) ? sh.cols : sh.rows;
    if (sh.ld < (sh.layout == Layout::RowMajor ? sh.cols : sh.rows))
        sh.ld = (sh.layout == Layout::RowMajor ? sh.cols : sh.rows);
    return sh;
}

template<class Elt>
class Recorder {
public:
    using GemmOp = kernels::GemmOp;
    using Pool   = BlockPool<Elt>;
    using Ref    = BlockRef<Elt>;

    explicit Recorder(const ModCtx<Elt>& ctx) { prog_.ctx = ctx; }

    const ModCtx<Elt>& ctx() const { return prog_.ctx; }

    // ---- buffers ---------------------------------------------------------
    BufId input(uint32_t rows, uint32_t cols, Layout layout, uint64_t content_key = 0) {
        BufferDesc d; d.kind = BufKind::Input; d.rows = rows; d.cols = cols; d.layout = layout;
        d.layout_fixed = true; d.slot = prog_.n_inputs++;
        ENGINE_CHECK_MSG((content_key & 0xE000000000000000ull) == 0, "tape: caller content keys must keep bits 61-63 clear");
        d.content_key = content_key != 0 ? content_key : unique_key();
        bind_.inputs.emplace_back();
        return push(d);
    }
    BufId input(DenseView<const Elt> v, uint64_t content_key = 0) {
        const ViewShape sh = view_shape(v);
        const BufId b = input(sh.rows, sh.cols, sh.layout, content_key);
        bind_.inputs[prog_.bufs[b].slot] = v;
        return b;
    }
    BufId output(uint32_t rows, uint32_t cols, Layout layout) {
        BufferDesc d; d.kind = BufKind::Output; d.rows = rows; d.cols = cols; d.layout = layout;
        d.layout_fixed = true; d.slot = prog_.n_outputs++; d.content_key = unique_key();
        bind_.outputs.emplace_back();
        return push(d);
    }
    BufId output(DenseView<Elt> v) {
        const ViewShape sh = view_shape(v);
        const BufId b = output(sh.rows, sh.cols, sh.layout);
        bind_.outputs[prog_.bufs[b].slot] = v;
        return b;
    }
    BufId temp(uint32_t rows, uint32_t cols, Layout preferred = Layout::RowMajor, bool pin_layout = false) {
        BufferDesc d; d.kind = BufKind::Temp; d.rows = rows; d.cols = cols; d.layout = preferred;
        d.layout_fixed = pin_layout; d.content_key = unique_key();
        return push(d);
    }
    // One Input per distinct (pool, id), memoised, with an exact pool-class
    // content key (the pool deduplicates, so equal ids <=> equal contents;
    // borrowed entries are never equal to anything else).
    BufId pool_input(const Pool& pool, typename Pool::Id id) {
        const auto key = std::make_pair(static_cast<const void*>(&pool), id);
        auto it = pool_inputs_.find(key);
        if (it != pool_inputs_.end()) return it->second;
        const DenseBuf<Elt>& b = pool.buf(id);
        // exact identity: one fresh pool-class key per distinct (pool, id) --
        // the pool deduplicates, so equal ids <=> equal contents
        const BufId bid = input(cview(b), 0);
        prog_.bufs[bid].content_key = 0x2000000000000000ull | (++prog_.next_key);
        pool_inputs_.emplace(key, bid);
        return bid;
    }

    ViewRef whole(BufId b) const {
        const BufferDesc& d = prog_.bufs[b];
        return ViewRef{b, 0, d.rows, 0, d.cols, false};
    }
    ViewRef view(BufId b, uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc, bool transposed = false) const {
        const BufferDesc& d = prog_.bufs[b];
        ENGINE_CHECK_MSG(static_cast<uint64_t>(r0) + nr <= d.rows && static_cast<uint64_t>(c0) + nc <= d.cols,
                         "tape: view exceeds buffer");
        return ViewRef{b, r0, nr, c0, nc, transposed};
    }
    // The data window of a BlockRef as a ViewRef (the logical dr x dc block).
    ViewRef ref_view(const Pool& pool, const Ref& r, uint32_t dr, uint32_t dc) {
        ENGINE_CHECK(r.has_data());
        const BufId b = pool_input(pool, r.pool_id);
        if (r.transposed()) return view(b, r.r0, dc, r.c0, dr, true);
        return view(b, r.r0, dr, r.c0, dc, false);
    }

    // ---- ops ---------------------------------------------------------------
    // c gop= alpha * a . b
    void gemm(ViewRef a, ViewRef b, ViewRef c, GemmOp gop = GemmOp::Set, Elt alpha = Elt(1)) {
        ENGINE_CHECK_MSG(a.lcols() == b.lrows() && a.lrows() == c.lrows() && b.lcols() == c.lcols(),
                         "tape::gemm: shape mismatch");
        Node<Elt> n; n.op = Op::GEMM; n.a = a; n.b = b; n.c = c; n.gop = gop; n.alpha = alpha;
        push(n);
    }
    // c gop= (ka1 I + kb1 M1)(ka2 I + kb2 M2): the block product with kinds
    // (pass 1 folds it).  M1 / M2 may be invalid views when kb1 / kb2 == 0.
    void gemm_ref(Elt ka1, Elt kb1, ViewRef M1, Elt ka2, Elt kb2, ViewRef M2, ViewRef c,
                  uint32_t dr, uint32_t dk, uint32_t dc, GemmOp gop = GemmOp::Set) {
        ENGINE_CHECK(c.lrows() == dr && c.lcols() == dc);
        if (kb1 != 0) ENGINE_CHECK(M1.lrows() == dr && M1.lcols() == dk);
        if (kb2 != 0) ENGINE_CHECK(M2.lrows() == dk && M2.lcols() == dc);
        if (ka1 != 0) ENGINE_CHECK_MSG(dr == dk, "tape::gemm_ref: scalar block must be square");
        if (ka2 != 0) ENGINE_CHECK_MSG(dk == dc, "tape::gemm_ref: scalar block must be square");
        Node<Elt> n; n.op = Op::GEMM; n.kinds = true; n.a = M1; n.b = M2; n.c = c; n.gop = gop;
        n.ka1 = ka1; n.kb1 = kb1; n.ka2 = ka2; n.kb2 = kb2;
        push(n);
    }
    void gemm_ref(const Ref& ra, ViewRef M1, const Ref& rb, ViewRef M2, ViewRef c,
                  uint32_t dr, uint32_t dk, uint32_t dc, GemmOp gop = GemmOp::Set) {
        gemm_ref(ra.alpha, ra.beta, M1, rb.alpha, rb.beta, M2, c, dr, dk, dc, gop);
    }
    // c -= a . b  (the elimination's trailing update)
    void schur_update(ViewRef a, ViewRef b, ViewRef c) {
        ENGINE_CHECK_MSG(a.lcols() == b.lrows() && a.lrows() == c.lrows() && b.lcols() == c.lcols(),
                         "tape::schur_update: shape mismatch");
        Node<Elt> n; n.op = Op::SCHUR_UPDATE; n.a = a; n.b = b; n.c = c; n.gop = GemmOp::Sub;
        push(n);
    }
    // c = a^{-1} b with `a` ALREADY INVERTED by the host (General), or c = b
    // (Identity), or c = alpha * b (Scalar; alpha = the inverse scalar).
    void trsm(PivotKind kind, ViewRef a_inv, ViewRef b, ViewRef c, Elt alpha = Elt(1)) {
        ENGINE_CHECK_MSG(b.lrows() == c.lrows() && b.lcols() == c.lcols(), "tape::trsm: shape mismatch");
        if (kind == PivotKind::General)
            ENGINE_CHECK_MSG(a_inv.lrows() == a_inv.lcols() && a_inv.lcols() == b.lrows(), "tape::trsm: factor shape");
        Node<Elt> n; n.op = Op::TRSM; n.pivot_kind = kind; n.a = a_inv; n.b = b; n.c = c; n.alpha = alpha;
        push(n);
    }
    // c = c +/- alpha * a
    void axpy(Elt alpha, ViewRef a, ViewRef c, bool subtract = false) {
        ENGINE_CHECK_MSG(a.lrows() == c.lrows() && a.lcols() == c.lcols(), "tape::axpy: shape mismatch");
        Node<Elt> n; n.op = Op::AXPY; n.a = a; n.c = c; n.alpha = alpha; n.subtract = subtract;
        push(n);
    }
    // c = c +/- alpha * I
    void axpy_identity(Elt alpha, ViewRef c, bool subtract = false) {
        ENGINE_CHECK_MSG(c.lrows() == c.lcols(), "tape::axpy_identity: square view required");
        Node<Elt> n; n.op = Op::AXPY; n.a = ViewRef{kIdentityBuf, 0, c.lrows(), 0, c.lcols(), false};
        n.c = c; n.alpha = alpha; n.subtract = subtract;
        push(n);
    }
    void scal(Elt alpha, ViewRef c) { Node<Elt> n; n.op = Op::SCAL; n.c = c; n.alpha = alpha; push(n); }
    void zero(ViewRef c) { scal(Elt(0), c); }
    // c(i, j) = a(i, j), c's buffer in the other physical layout (pass 3 inserts
    // these itself; callers may also request one).
    void transpose(ViewRef a, ViewRef c) {
        ENGINE_CHECK_MSG(a.lrows() == c.lrows() && a.lcols() == c.lcols(), "tape::transpose: shape mismatch");
        Node<Elt> n; n.op = Op::TRANSPOSE; n.a = a; n.c = c; push(n);
    }
    void canonicalize(ViewRef c) { Node<Elt> n; n.op = Op::CANONICALIZE; n.c = c; push(n); }
    // c = a (copy, any layouts)
    void scatter(ViewRef a, ViewRef c) {
        ENGINE_CHECK_MSG(a.lrows() == c.lrows() && a.lcols() == c.lcols(), "tape::scatter: shape mismatch");
        Node<Elt> n; n.op = Op::SCATTER; n.a = a; n.c = c; push(n);
    }
    // c(i, t) [+]= alpha * [row_off + i == cols[t]] + beta * a(i, cols[t]);
    // `a` is the row range of the block's logical window; cols = index list
    // (kNoList: every column, t == column).
    void gather(Elt alpha, Elt beta, ViewRef a, uint32_t row_off, uint32_t index_list, ViewRef c,
                bool accumulate = false) {
        const uint32_t nc = index_list == kNoList ? a.lcols()
                                                  : static_cast<uint32_t>(prog_.index_lists[index_list].size());
        ENGINE_CHECK_MSG(c.lrows() == a.lrows() && c.lcols() == nc, "tape::gather: shape mismatch");
        Node<Elt> n; n.op = Op::GATHER; n.a = a; n.c = c; n.alpha = alpha; n.beta = beta;
        n.aux = row_off; n.index_list = index_list; n.accumulate = accumulate;
        push(n);
    }
    // c(i, t) = alpha * [row_off + rows[i] == cols[t]] + beta * a(rows[i], cols[t]) for the
    // row list `row_list` (indices into the logical rows of a).
    void gather_rows(Elt alpha, Elt beta, ViewRef a, uint32_t row_off, uint32_t row_list, uint32_t index_list,
                     ViewRef c, bool accumulate = false) {
        ENGINE_CHECK(row_list != kNoList);
        const uint32_t nr = static_cast<uint32_t>(prog_.index_lists[row_list].size());
        const uint32_t nc = index_list == kNoList ? a.lcols()
                                                  : static_cast<uint32_t>(prog_.index_lists[index_list].size());
        ENGINE_CHECK_MSG(c.lrows() == nr && c.lcols() == nc, "tape::gather_rows: shape mismatch");
        Node<Elt> n; n.op = Op::GATHER; n.a = a; n.c = c; n.alpha = alpha; n.beta = beta;
        n.aux = row_off; n.index_list = index_list; n.row_list = row_list; n.accumulate = accumulate;
        push(n);
    }
    // The rows `row_list` (block-local indices) of the block `r` (dr x dc) into c.
    void gather_block_row_list(const Pool& pool, const Ref& r, uint32_t dr, uint32_t dc, uint32_t row_list,
                               ViewRef c, bool accumulate = false, uint32_t index_list = kNoList) {
        ENGINE_CHECK(r.alpha == 0 || dr == dc);
        if (r.has_data()) {
            gather_rows(r.alpha, r.beta, ref_view(pool, r, dr, dc), 0, row_list, index_list, c, accumulate);
        } else {
            ViewRef I{kIdentityBuf, 0, dr, 0, dc, false};
            gather_rows(r.alpha, Elt(0), I, 0, row_list, index_list, c, accumulate);
        }
    }
    // Materialise rows [lo, hi) of the block `r` (dr x dc) into c: the tape
    // form of BlockMat::materialise_rows_of_block.  A data-free ref (Scalar)
    // gathers from the identity only.
    void gather_block_rows(const Pool& pool, const Ref& r, uint32_t dr, uint32_t dc, uint32_t lo, uint32_t hi,
                           ViewRef c, bool accumulate = false, uint32_t index_list = kNoList) {
        ENGINE_CHECK(r.alpha == 0 || dr == dc);
        if (r.has_data()) {
            const ViewRef M = ref_view(pool, r, dr, dc).sub(lo, hi - lo, 0, dc);
            gather(r.alpha, r.beta, M, lo, index_list, c, accumulate);
        } else {
            // identity source: alpha * I rows [lo, hi)
            ViewRef I{kIdentityBuf, lo, hi - lo, 0, dc, false};
            gather(r.alpha, Elt(0), I, lo, index_list, c, accumulate);
        }
    }
    uint32_t index_list(std::vector<uint32_t> cols) {
        prog_.index_lists.push_back(std::move(cols));
        return static_cast<uint32_t>(prog_.index_lists.size() - 1);
    }
    uint32_t panel_rref(ViewRef c) {
        Node<Elt> n; n.op = Op::PANEL_RREF; n.c = c; n.slot_out = prog_.n_slots++;
        push(n); return n.slot_out;
    }
    // a = the panel before its RREF, q_slot = the PANEL_RREF slot; c receives
    // a[prows, Q] in its top-left k x k corner (c must be at least
    // min(rows, cols)^2).  Returns the slot holding prows.
    uint32_t pivot_search(ViewRef a, uint32_t q_slot, ViewRef c) {
        const uint32_t kmax = std::min(a.lrows(), a.lcols());
        ENGINE_CHECK_MSG(c.lrows() >= kmax && c.lcols() >= kmax, "tape::pivot_search: pivot block too small");
        Node<Elt> n; n.op = Op::PIVOT_SEARCH; n.a = a; n.c = c; n.slot_in = q_slot; n.slot_out = prog_.n_slots++;
        push(n); return n.slot_out;
    }
    uint32_t compare(ViewRef a, ViewRef b) {
        Node<Elt> n; n.op = Op::COMPARE; n.a = a; n.b = b; n.slot_out = prog_.n_slots++;
        push(n); return n.slot_out;
    }

    // ---- block-level helpers ---------------------------------------------
    // Y gop= A . X for a BlockMat A and a dense multi-vector X (A.cols() x w):
    // one kind-GEMM per block entry (blk::gemm_dense on the tape).
    void record_gemm_dense(const BlockMat<Elt>& A, ViewRef X, ViewRef Y, GemmOp gop = GemmOp::Set) {
        const uint32_t w = X.lcols();
        ENGINE_CHECK_MSG(X.lrows() == A.cols() && Y.lrows() == A.rows() && Y.lcols() == w,
                         "record_gemm_dense: shape mismatch");
        if (gop == GemmOp::Set) zero(Y);
        const GemmOp acc = (gop == GemmOp::Sub) ? GemmOp::Sub : GemmOp::Add;
        const Pool& pool = *A.pool();
        for (uint32_t br = 0; br < A.nbr(); ++br) {
            const uint32_t dr = A.row_dim(br);
            if (dr == 0) continue;
            const ViewRef y = Y.sub(A.row_start(br), dr, 0, w);
            for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k) {
                const uint32_t bc = A.col_at(k), dc = A.col_dim(bc);
                if (dc == 0 || w == 0) continue;
                const Ref& r = A.ref_at(k);
                const ViewRef x = X.sub(A.col_start(bc), dc, 0, w);
                const ViewRef M = r.has_data() ? ref_view(pool, r, dr, dc) : ViewRef{};
                gemm_ref(r.alpha, r.beta, M, Elt(0), Elt(1), x, y, dr, dc, w, acc);
            }
        }
    }
    // C = A . B for two BlockMats, C a dense (A.rows() x B.cols()) output
    // (blk::gemm on the tape, dense result).
    void record_gemm_block(const BlockMat<Elt>& A, const BlockMat<Elt>& B, ViewRef C) {
        ENGINE_CHECK_MSG(A.col_off() == B.row_off(), "record_gemm_block: partitions differ");
        ENGINE_CHECK_MSG(C.lrows() == A.rows() && C.lcols() == B.cols(), "record_gemm_block: shape mismatch");
        zero(C);
        const Pool& pa = *A.pool();
        const Pool& pb = *B.pool();
        for (uint32_t br = 0; br < A.nbr(); ++br) {
            const uint32_t dr = A.row_dim(br);
            if (dr == 0) continue;
            for (uint32_t ka = A.row_begin(br); ka < A.row_end(br); ++ka) {
                const uint32_t k = A.col_at(ka), dk = A.col_dim(k);
                if (dk == 0) continue;
                const Ref& ra = A.ref_at(ka);
                const ViewRef Ma = ra.has_data() ? ref_view(pa, ra, dr, dk) : ViewRef{};
                for (uint32_t kb = B.row_begin(k); kb < B.row_end(k); ++kb) {
                    const uint32_t bc = B.col_at(kb), dc = B.col_dim(bc);
                    if (dc == 0) continue;
                    const Ref& rb = B.ref_at(kb);
                    const ViewRef Mb = rb.has_data() ? ref_view(pb, rb, dk, dc) : ViewRef{};
                    gemm_ref(ra, Ma, rb, Mb, C.sub(A.row_start(br), dr, B.col_start(bc), dc), dr, dk, dc, GemmOp::Add);
                }
            }
        }
    }
    // The whole BlockMat gathered into a dense view (block -> flat).
    void record_gather_dense(const BlockMat<Elt>& A, ViewRef D) {
        ENGINE_CHECK_MSG(D.lrows() == A.rows() && D.lcols() == A.cols(), "record_gather_dense: shape mismatch");
        zero(D);
        const Pool& pool = *A.pool();
        for (uint32_t br = 0; br < A.nbr(); ++br) {
            const uint32_t dr = A.row_dim(br);
            if (dr == 0) continue;
            for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k) {
                const uint32_t bc = A.col_at(k), dc = A.col_dim(bc);
                if (dc == 0) continue;
                gather_block_rows(pool, A.ref_at(k), dr, dc, 0, dr,
                                  D.sub(A.row_start(br), dr, A.col_start(bc), dc), false);
            }
        }
    }

    // ---- finish -------------------------------------------------------------
    Program<Elt> finish();
    const Bindings<Elt>& bindings() const { return bind_; }
    size_t num_nodes() const { return prog_.nodes.size(); }

private:
    BufId push(const BufferDesc& d) { prog_.bufs.push_back(d); return static_cast<BufId>(prog_.bufs.size() - 1); }
    void push(Node<Elt>& n) {
        n.origin = static_cast<uint32_t>(prog_.nodes.size());
        prog_.nodes.push_back(n);
    }
    uint64_t unique_key() { return 0x8000000000000000ull | (++prog_.next_key); }

    Program<Elt>   prog_;
    Bindings<Elt>  bind_;
    std::map<std::pair<const void*, uint32_t>, BufId> pool_inputs_;
};

} // namespace tape
} // namespace engine
