// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// BlockElim.cpp -- see BlockElim.h.
#include "BlockElim.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>

#include "DenseElim.h"
#include "Kernels.h"

namespace engine {
namespace blk {

namespace {

// (slot-8 WP-6) The diagnostic counters behind elim_stats().  They are bumped
// from every thread that runs block elimination -- concurrent campaign jobs,
// the parallel ext pass, the engine's own workers -- so they are RELAXED
// ATOMICS: they were plain uint64_t, a ThreadSanitizer data race under
// concurrent jobs (slot-8 WP-5 report, finding 1).  They order nothing and
// feed no computed value, so relaxed increments are exact counts with no
// ordering cost, and no value can move.
struct AtomicElimStats {
    std::atomic<uint64_t> panels{0}, panels_with_pivots{0}, dense_rrefs{0}, gemms{0};
    std::atomic<uint64_t> panel_rows{0};
    std::atomic<uint64_t> gathered_blocks{0}, scalar_blocks_gathered{0};
};
AtomicElimStats g_stats;
inline void bump(std::atomic<uint64_t>& c, uint64_t n = 1) { c.fetch_add(n, std::memory_order_relaxed); }

using kernels::GemmOp;

// A row-major (rows x n) buffer that grows by rows.  Values already stored
// are copied verbatim on growth (exact), so growth never affects a result.
template<class Elt>
class GrowRows {
public:
    explicit GrowRows(uint32_t n) : n_(n) {}
    uint32_t rows() const { return rows_; }
    uint32_t n() const { return n_; }
    DenseView<Elt> view() { return DenseView<Elt>(buf_, 0, rows_, 0, n_); }
    DenseView<const Elt> cview() const { return DenseView<const Elt>(buf_, 0, rows_, 0, n_); }
    // Appends k zero rows, returns the index of the first.
    uint32_t append(uint32_t k) {
        if (rows_ + k > cap_) {
            uint32_t nc = std::max<uint32_t>(cap_ * 2, rows_ + k);
            nc = std::max<uint32_t>(nc, 32);
            DenseBuf<Elt> nb(nc, n_);
            if (rows_ > 0 && n_ > 0)
                kernels::copy<Elt>(DenseView<const Elt>(buf_, 0, rows_, 0, n_), DenseView<Elt>(nb, 0, rows_, 0, n_));
            buf_ = std::move(nb);
            cap_ = nc;
        }
        const uint32_t first = rows_;
        rows_ += k;
        return first;
    }
    DenseBuf<Elt> take() {
        DenseBuf<Elt> out(rows_, n_);
        if (rows_ > 0 && n_ > 0) kernels::copy<Elt>(cview(), engine::view(out));
        return out;
    }
private:
    uint32_t n_, rows_ = 0, cap_ = 0;
    DenseBuf<Elt> buf_;
};

// One pivot block column: which block column, the first index of its pivots
// in the global list, and its local pivot columns.
struct PivBlock {
    uint32_t bc;
    uint32_t first;
    std::vector<uint32_t> local;
};

// Runs of selected (active) scalar rows inside one block row.
struct RowRun {
    uint32_t br;
    uint32_t lo, hi;    // block-local rows [lo, hi)
    uint32_t dst;       // first destination row in the panel buffers
};

// Column -> block rows index (CSC pattern of A), built once per rref.
std::vector<std::vector<uint32_t>> col_rows_index(uint32_t nbr, uint32_t nbc,
                                                  const std::vector<uint32_t>& row_ptr,
                                                  const std::vector<uint32_t>& col_idx) {
    std::vector<std::vector<uint32_t>> cr(nbc);
    for (uint32_t br = 0; br < nbr; ++br)
        for (uint32_t k = row_ptr[br]; k < row_ptr[br + 1]; ++k) cr[col_idx[k]].push_back(br);
    return cr;
}

// dst (nrows x ncols) = entries of the block at (rows lo..hi) x (local cols),
// accumulated (dst must be zeroed by the caller for the first write).
template<class Elt>
void gather_block_cols(const BlockMat<Elt>& A, const BlockRef<Elt>& r, uint32_t dr, uint32_t dc,
                       uint32_t lo, uint32_t hi, const std::vector<uint32_t>& local_cols,
                       DenseView<Elt> dst) {
    const ModCtx<Elt>& ctx = A.ctx();
    const uint32_t nc = static_cast<uint32_t>(local_cols.size());
    ENGINE_CHECK(dst.rows() == hi - lo && dst.cols() == nc);
    if (r.has_data()) {
        const DenseView<const Elt> M = A.data_view(r, dr, dc);
        for (uint32_t t = 0; t < nc; ++t) {
            const uint32_t c = local_cols[t];
            if (r.beta == 1) for (uint32_t i = lo; i < hi; ++i) dst.at(i - lo, t) = M.at(i, c);
            else             for (uint32_t i = lo; i < hi; ++i) dst.at(i - lo, t) = ctx.mul(r.beta, M.at(i, c));
        }
    }
    if (r.alpha != 0) {
        bump(g_stats.scalar_blocks_gathered);
        for (uint32_t t = 0; t < nc; ++t) {
            const uint32_t c = local_cols[t];
            if (c >= lo && c < hi) dst.at(c - lo, t) = ctx.add(dst.at(c - lo, t), r.alpha);
        }
    }
    bump(g_stats.gathered_blocks);
}

template<class Elt>
void gemm_sub(const ModCtx<Elt>& ctx, DenseView<const Elt> L, DenseView<const Elt> Rv, DenseView<Elt> C) {
    if (C.rows() == 0 || C.cols() == 0 || L.cols() == 0) return;
    kernels::gemm<Elt>(ctx, L, Rv, C, GemmOp::Sub);
    bump(g_stats.gemms);
}
template<class Elt>
void gemm_set(const ModCtx<Elt>& ctx, DenseView<const Elt> L, DenseView<const Elt> Rv, DenseView<Elt> C) {
    if (C.rows() == 0 || C.cols() == 0) return;
    if (L.cols() == 0) { BlockMat<Elt>::zero_view(C); return; }
    kernels::gemm<Elt>(ctx, L, Rv, C, GemmOp::Set);
    bump(g_stats.gemms);
}

} // namespace

// ---------------------------------------------------------------------------
// Invariant proof (the header's step list).  Let P = pivots so far, R the
// reduced prefix (identity at P), and for a row r define
//   S[r] = A[r, j] - A[r, P] R[:, j]                       (panel Schur complement).
// Let R_j = RREF(S) with local pivots Q, prows with B = S[prows, Q] invertible,
// and define the new rows over ALL columns
//   N = B^{-1} (A[prows, :] - A[prows, P] R)      so that  N[:, j] = R_j, N[:, P] = 0,
//   N[:, processed non-pivot columns] = 0 (they were eliminated).
// Reduce above: R' = R - R[:, Q] N.  Then R'[:, Q] = 0, R'[:, P] = I, N[:, Q] = I,
// so [R'; N] has the identity at P u Q with P < Q: reduced echelon form.
// For any row r, on processed-or-panel columns C:
//   A[r,C] - A[r,P] R'[:,C] - A[r,Q] N[:,C]
//     = (A[r,C] - A[r,P] R[:,C]) - (A[r,Q] - A[r,P] R[:,Q]) N[:,C]
//     = (A[r,C] - A[r,P] R[:,C]) - S[r, Q] N[:, C].
// On processed columns the first term is 0 (invariant) and N is 0; on the
// panel the first term is S[r] and N[:, j] = R_j, and S[r] = S[r, Q] R_j
// because S[r] lies in the row space of R_j, which has the identity at Q.
// Hence the invariant holds for every row (including prows, whose residual
// on the trailing columns is A[prows,trail] - A[prows,P] R[:,trail]
// - B N[:,trail] = 0 by the definition of N -- so they are exactly zero in
// every later panel and can be dropped from the active set).
// ---------------------------------------------------------------------------
template<class Elt>
BlockRrefResult<Elt> rref(const BlockMat<Elt>& A) {
    using Ref = BlockRef<Elt>;
    const ModCtx<Elt>& ctx = A.ctx();
    // ENGINE_API.md Sec. 13.3: field-only (it reaches elim::rref); its own entry guard.
    ENGINE_CHECK_MSG(ctx.is_field(), "blk::rref: field-only; use the Z/ell^k entry points of ENGINE_API.md Sec. 13");
    const uint32_t m = A.rows(), n = A.cols(), nbr = A.nbr(), nbc = A.nbc();
    BlockRrefResult<Elt> out;
    GrowRows<Elt> R(n);
    if (m == 0 || n == 0) { out.R = DenseBuf<Elt>(0, n); return out; }

    const auto col_rows = col_rows_index(nbr, nbc, A.row_ptr(), A.col_idx());
    std::vector<char> active(m, 1);
    std::vector<uint32_t> active_count(nbr);
    for (uint32_t br = 0; br < nbr; ++br) active_count[br] = A.row_dim(br);
    std::vector<char> touches_pivot(nbr, 0);
    std::vector<PivBlock> pivblocks;
    std::vector<uint32_t>& pivots = out.pivots;

    std::vector<uint32_t> cand_brs, crow;
    std::vector<RowRun> runs;
    std::vector<char> mark(nbr, 0);

    for (uint32_t j = 0; j < nbc; ++j) {
        const uint32_t d = A.col_dim(j);
        if (d == 0) continue;
        bump(g_stats.panels);
        const uint32_t rk = static_cast<uint32_t>(pivots.size());
        if (rk == std::min(m, n)) break;   // cannot grow further (rank bound)

        // 1. candidate block rows: structural non-zero at j, or touching a pivot block column
        cand_brs.clear();
        for (uint32_t br : col_rows[j]) if (active_count[br] > 0 && !mark[br]) { mark[br] = 1; cand_brs.push_back(br); }
        if (rk > 0)
            for (uint32_t br = 0; br < nbr; ++br)
                if (touches_pivot[br] && active_count[br] > 0 && !mark[br]) { mark[br] = 1; cand_brs.push_back(br); }
        for (uint32_t br : cand_brs) mark[br] = 0;
        std::sort(cand_brs.begin(), cand_brs.end());
        // candidate scalar rows as runs
        crow.clear(); runs.clear();
        for (uint32_t br : cand_brs) {
            const uint32_t rs = A.row_start(br), dr = A.row_dim(br);
            uint32_t i = 0;
            while (i < dr) {
                while (i < dr && !active[rs + i]) ++i;
                if (i == dr) break;
                const uint32_t lo = i;
                while (i < dr && active[rs + i]) ++i;
                runs.push_back(RowRun{br, lo, i, static_cast<uint32_t>(crow.size())});
                for (uint32_t t = lo; t < i; ++t) crow.push_back(rs + t);
            }
        }
        const uint32_t nc = static_cast<uint32_t>(crow.size());
        if (nc == 0) continue;
        bump(g_stats.panel_rows, nc);

        // 2. S = A[cand, j] - A[cand, P] * R[:, j]
        DenseBuf<Elt> S(nc, d);
        for (const RowRun& ru : runs) {
            const Ref* r = A.find(ru.br, j);
            if (r == nullptr) continue;
            A.materialise_rows_of_block(*r, A.row_dim(ru.br), d, ru.lo, ru.hi,
                                        view(S).sub(ru.dst, ru.hi - ru.lo, 0, d));
            bump(g_stats.gathered_blocks);
        }
        DenseBuf<Elt> Lg;
        if (rk > 0) {
            Lg = DenseBuf<Elt>(nc, rk);
            for (const PivBlock& pb : pivblocks) {
                const uint32_t dcp = A.col_dim(pb.bc);
                const uint32_t cnt = static_cast<uint32_t>(pb.local.size());
                for (const RowRun& ru : runs) {
                    const Ref* r = A.find(ru.br, pb.bc);
                    if (r == nullptr) continue;
                    gather_block_cols<Elt>(A, *r, A.row_dim(ru.br), dcp, ru.lo, ru.hi, pb.local,
                                           view(Lg).sub(ru.dst, ru.hi - ru.lo, pb.first, cnt));
                }
            }
            gemm_sub<Elt>(ctx, cview(Lg), R.cview().sub(0, rk, A.col_start(j), d), view(S));
        }

        // 3. panel RREF
        DenseBuf<Elt> Sc(S);
        const elim::RrefResult pr = elim::rref<Elt>(ctx, view(Sc));
        bump(g_stats.dense_rrefs);
        const uint32_t kj = pr.rank;
        if (kj == 0) continue;
        bump(g_stats.panels_with_pivots);
        const std::vector<uint32_t>& Q = pr.pivots;

        // pivot rows: row rank profile of S[:, Q]
        std::vector<uint32_t> prows;
        {
            DenseBuf<Elt> Pt(kj, nc);
            for (uint32_t t = 0; t < kj; ++t)
                for (uint32_t i = 0; i < nc; ++i) Pt.at(t, i) = S.at(i, Q[t]);
            const elim::RrefResult rp = elim::rref<Elt>(ctx, view(Pt));
            bump(g_stats.dense_rrefs);
            ENGINE_CHECK_MSG(rp.rank == kj, "BlockElim: row rank profile lost rank");
            prows = rp.pivots;
        }

        // 4. new rows
        const uint32_t first = R.append(kj);
        const uint32_t c0 = A.col_start(j);
        DenseView<Elt> Rv = R.view();
        kernels::copy<Elt>(cview(Sc).sub(0, kj, 0, d), Rv.sub(first, kj, c0, d));
        const uint32_t trail0 = c0 + d;
        const uint32_t nt = n - trail0;
        if (nt > 0) {
            DenseBuf<Elt> B(kj, kj), Binv(kj, kj);
            for (uint32_t t = 0; t < kj; ++t)
                for (uint32_t u = 0; u < kj; ++u) B.at(t, u) = S.at(prows[t], Q[u]);
            const bool ok = elim::inverse<Elt>(ctx, cview(B), view(Binv));
            ENGINE_CHECK_MSG(ok, "BlockElim: pivot block not invertible");
            bump(g_stats.dense_rrefs);
            // T = A[prows, trail] - A[prows, P] R[:, trail]
            DenseBuf<Elt> T(kj, nt);
            {
                // group prows by block row (prows ascending in crow order, so runs are grouped)
                uint32_t t = 0;
                while (t < kj) {
                    const uint32_t grow = crow[prows[t]];
                    const uint32_t br = A.block_row_of(grow);
                    const uint32_t rs = A.row_start(br), dr = A.row_dim(br);
                    uint32_t u = t;
                    while (u < kj && A.block_row_of(crow[prows[u]]) == br) ++u;
                    for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k) {
                        const uint32_t bc = A.col_at(k);
                        if (bc <= j) continue;
                        const uint32_t dc = A.col_dim(bc);
                        if (dc == 0) continue;
                        const Ref& r = A.ref_at(k);
                        for (uint32_t v = t; v < u; ++v) {
                            const uint32_t li = crow[prows[v]] - rs;
                            A.materialise_rows_of_block(r, dr, dc, li, li + 1,
                                                        view(T).sub(v, 1, A.col_start(bc) - trail0, dc));
                        }
                        bump(g_stats.gathered_blocks);
                    }
                    t = u;
                }
            }
            if (rk > 0) {
                DenseBuf<Elt> Lp(kj, rk);
                for (uint32_t t = 0; t < kj; ++t)
                    kernels::copy<Elt>(cview(Lg).sub(prows[t], 1, 0, rk), view(Lp).sub(t, 1, 0, rk));
                gemm_sub<Elt>(ctx, cview(Lp), R.cview().sub(0, rk, trail0, nt), view(T));
            }
            gemm_set<Elt>(ctx, cview(Binv), cview(T), Rv.sub(first, kj, trail0, nt));
        }

        // 5. reduce above
        if (rk > 0) {
            DenseBuf<Elt> C(rk, kj);
            for (uint32_t i = 0; i < rk; ++i)
                for (uint32_t t = 0; t < kj; ++t) C.at(i, t) = Rv.at(i, c0 + Q[t]);
            gemm_sub<Elt>(ctx, cview(C), R.cview().sub(first, kj, c0, n - c0), Rv.sub(0, rk, c0, n - c0));
        }

        // 6. bookkeeping
        PivBlock pb; pb.bc = j; pb.first = rk; pb.local = Q;
        pivblocks.push_back(std::move(pb));
        for (uint32_t t = 0; t < kj; ++t) pivots.push_back(c0 + Q[t]);
        for (uint32_t br : col_rows[j]) touches_pivot[br] = 1;
        for (uint32_t t = 0; t < kj; ++t) {
            const uint32_t grow = crow[prows[t]];
            out.pivot_rows.push_back(grow);
            active[grow] = 0;
            --active_count[A.block_row_of(grow)];
        }
    }
    out.rank = static_cast<uint32_t>(pivots.size());
    out.R = R.take();
    return out;
}

template<class Elt>
uint32_t rank(const BlockMat<Elt>& A) { return rref<Elt>(A).rank; }

template<class Elt>
BlockMat<Elt> rref_block(const BlockMat<Elt>& A, const BlockRrefResult<Elt>& res) {
    return BlockMat<Elt>::from_dense(A.pool(), cview(res.R), std::vector<uint32_t>{res.rank}, A.col_dims(),
                                     A.p(), Layout::RowMajor, true);
}

template<class Elt>
BlockMat<Elt> kernel_basis(const BlockMat<Elt>& A, const BlockRrefResult<Elt>& res, Layout layout) {
    const ModCtx<Elt>& ctx = A.ctx();
    const uint32_t n = A.cols(), k = res.rank;
    ENGINE_CHECK(k <= n);
    std::vector<char> is_piv(n, 0);
    for (uint32_t c : res.pivots) is_piv[c] = 1;
    DenseBuf<Elt> K(n, n - k, layout);
    uint32_t t = 0;
    for (uint32_t jf = 0; jf < n; ++jf) {
        if (is_piv[jf]) continue;
        K.at(jf, t) = 1;
        for (uint32_t i = 0; i < k; ++i) K.at(res.pivots[i], t) = ctx.neg(res.R.at(i, jf));
        ++t;
    }
    return BlockMat<Elt>::from_flat(A.pool(), std::move(K), A.col_dims(), A.p(), true);
}
template<class Elt>
BlockMat<Elt> kernel_basis(const BlockMat<Elt>& A, Layout layout) {
    return kernel_basis<Elt>(A, rref<Elt>(A), layout);
}

template<class Elt>
BlockMat<Elt> image_basis(const BlockMat<Elt>& A, const BlockRrefResult<Elt>& res, Layout layout) {
    const uint32_t m = A.rows(), k = res.rank;
    DenseBuf<Elt> Im(m, k, layout);
    const auto col_rows = col_rows_index(A.nbr(), A.nbc(), A.row_ptr(), A.col_idx());
    for (uint32_t t = 0; t < k; ++t) {
        const uint32_t c = res.pivots[t];
        const uint32_t bc = A.block_col_of(c), lc = c - A.col_start(bc), dc = A.col_dim(bc);
        for (uint32_t br : col_rows[bc]) {
            const uint32_t dr = A.row_dim(br);
            if (dr == 0) continue;
            const BlockRef<Elt>* r = A.find(br, bc);
            A.materialise_cols_of_block(*r, dr, dc, lc, lc + 1, view(Im).sub(A.row_start(br), dr, t, 1));
        }
    }
    return BlockMat<Elt>::from_flat(A.pool(), std::move(Im), A.row_dims(), A.p(), true);
}
template<class Elt>
BlockMat<Elt> image_basis(const BlockMat<Elt>& A, Layout layout) {
    return image_basis<Elt>(A, rref<Elt>(A), layout);
}

namespace {

template<class Elt>
uint32_t row_start_of(const BlockBuilder<Elt>& b, uint32_t br) {
    uint32_t s = 0;
    for (uint32_t i = 0; i < br; ++i) s += b.row_dim(i);
    return s;
}
// Appends `cols` columns of the dense view D (a x cols) to the builder as
// borrowed windows in chunks of `chunk`; returns the number of block columns
// added.  Column-block index offset = bc0.  All-zero windows stay structural.
template<class Elt>
uint32_t add_dense_chunks(BlockBuilder<Elt>& b, typename BlockPool<Elt>::Id id, DenseView<const Elt> D,
                          uint32_t bc0, uint32_t chunk) {
    const uint32_t cols = D.cols();
    uint32_t bc = bc0;
    for (uint32_t c = 0; c < cols; c += chunk, ++bc) {
        const uint32_t w = std::min<uint32_t>(chunk, cols - c);
        for (uint32_t br = 0; br < b.nbr(); ++br) {
            const uint32_t dr = b.row_dim(br);
            if (dr == 0) continue;
            const uint32_t rs = row_start_of(b, br);
            if (BlockMat<Elt>::is_zero_view(D.sub(rs, dr, c, w))) continue;
            b.set(br, bc, BlockRef<Elt>::dense(id, Elt(1), rs, c));
        }
    }
    return bc - bc0;
}
// A borrowed pool entry held for the duration of an elimination: acquire on
// construction, release on destruction.  The augmented BlockMat's windows
// come and go underneath; without this hold its destructor would free the
// entry (refcount 0) and RECYCLE the id, and a later sweep of the stale id
// could free another thread's freshly interned block (the pool is shared by
// the concurrently running degrees of compute_cohomology(): a 1-in-3 abort in
// CohomologyBlockModelValidation before this hold existed).
template<class Elt>
struct BorrowHold {
    std::shared_ptr<BlockPool<Elt>> pool;
    typename BlockPool<Elt>::Id id = BlockPool<Elt>::kNone;
    BorrowHold(std::shared_ptr<BlockPool<Elt>> p, DenseView<const Elt> v) : pool(std::move(p)), id(pool->borrow(v)) { pool->acquire(id); }
    ~BorrowHold() { if (id != BlockPool<Elt>::kNone) pool->release(id); }
    BorrowHold(const BorrowHold&) = delete;
    BorrowHold& operator=(const BorrowHold&) = delete;
};
std::vector<uint32_t> chunk_dims(uint32_t cols, uint32_t chunk) {
    std::vector<uint32_t> d;
    for (uint32_t c = 0; c < cols; c += chunk) d.push_back(std::min<uint32_t>(chunk, cols - c));
    return d;
}

} // namespace

template<class Elt>
FactorBasisResult<Elt> factor_basis(std::shared_ptr<BlockPool<Elt>> pool, Elt p,
                                    const std::vector<uint32_t>& row_dims,
                                    DenseView<const Elt> Rl, DenseView<const Elt> G,
                                    const ElimParams& params) {
    using Ref = BlockRef<Elt>;
    FactorBasisResult<Elt> out;
    uint32_t a = 0;
    for (uint32_t d : row_dims) a += d;
    const uint32_t nr = Rl.cols(), ng = G.cols();
    ENGINE_CHECK_MSG((nr == 0 || Rl.rows() == a) && (ng == 0 || G.rows() == a),
                     "factor_basis: operand height != sum(row_dims)");
    const uint32_t chunk = std::max<uint32_t>(1, params.chunk);
    if (ng == 0) {
        out.basis = BlockMat<Elt>::zeros(pool, row_dims, std::vector<uint32_t>{0}, p);
        out.projector = DenseBuf<Elt>(0, a);
        return out;
    }
    // Column partition: Rl chunks | G chunks | identity (row_dims)
    std::vector<uint32_t> col_dims = chunk_dims(nr, chunk);
    const uint32_t nrc = static_cast<uint32_t>(col_dims.size());
    for (uint32_t d : chunk_dims(ng, chunk)) col_dims.push_back(d);
    const uint32_t ngc = static_cast<uint32_t>(col_dims.size()) - nrc;
    for (uint32_t d : row_dims) col_dims.push_back(d);

    BlockBuilder<Elt> b(pool, row_dims, col_dims, p);
    std::unique_ptr<BorrowHold<Elt>> holdR;
    if (nr > 0) { holdR = std::make_unique<BorrowHold<Elt>>(pool, Rl); add_dense_chunks<Elt>(b, holdR->id, Rl, 0, chunk); }
    const BorrowHold<Elt> holdG(pool, G);
    add_dense_chunks<Elt>(b, holdG.id, G, nrc, chunk);
    for (uint32_t br = 0; br < row_dims.size(); ++br)
        if (row_dims[br] > 0) b.set(br, nrc + ngc + br, Ref::identity());
    BlockRrefResult<Elt> res;
    {
        const BlockMat<Elt> aug = b.finalize();
        res = rref<Elt>(aug);
    }

    for (uint32_t c : res.pivots) {
        if (c < nr) ++out.independent_relations;
        else if (c < nr + ng) out.pivots.push_back(c - nr);
    }
    const uint32_t dim_h = static_cast<uint32_t>(out.pivots.size());
    out.projector = DenseBuf<Elt>(dim_h, a);
    if (dim_h > 0 && a > 0)
        kernels::copy<Elt>(cview(res.R).sub(out.independent_relations, dim_h, nr + ng, a), view(out.projector));
    DenseBuf<Elt> basis(a, dim_h, Layout::ColMajor);
    for (uint32_t t = 0; t < dim_h; ++t)
        if (a > 0) kernels::copy<Elt>(G.sub(0, a, out.pivots[t], 1), view(basis).sub(0, a, t, 1));
    out.basis = BlockMat<Elt>::from_flat(pool, std::move(basis), row_dims, p, true);
    return out;
}

template<class Elt>
DenseView<const Elt> multivector_view(const BlockMat<Elt>& mv, DenseBuf<Elt>& scratch, Layout layout) {
    ENGINE_CHECK_MSG(mv.nbc() == 1, "multivector_view: nbc != 1");
    DenseView<const Elt> v;
    if (mv.contiguous_view(v)) return v;
    scratch = mv.to_dense(layout);
    return cview(scratch);
}

template<class Elt>
FactorBasisResult<Elt> factor_basis(const BlockMat<Elt>& Rl, const BlockMat<Elt>& G, const ElimParams& params) {
    ENGINE_CHECK_MSG(Rl.row_off() == G.row_off(), "factor_basis: row partitions differ");
    DenseBuf<Elt> sr, sg;
    const DenseView<const Elt> vr = multivector_view<Elt>(Rl, sr);
    const DenseView<const Elt> vg = multivector_view<Elt>(G, sg);
    return factor_basis<Elt>(G.pool(), G.p(), G.row_dims(), vr, vg, params);
}

template<class Elt>
DenseBuf<Elt> solve_multi_rhs(std::shared_ptr<BlockPool<Elt>> pool, Elt p,
                              const std::vector<uint32_t>& row_dims,
                              DenseView<const Elt> B, DenseView<const Elt> Y, const ElimParams& params) {
    uint32_t a = 0;
    for (uint32_t d : row_dims) a += d;
    const uint32_t nb = B.cols(), ny = Y.cols();
    ENGINE_CHECK_MSG((nb == 0 || B.rows() == a) && (ny == 0 || Y.rows() == a),
                     "solve_multi_rhs: operand height != sum(row_dims)");
    DenseBuf<Elt> result(ny, nb);
    if (ny == 0 || a == 0) return result;
    const uint32_t chunk = std::max<uint32_t>(1, params.chunk);
    std::vector<uint32_t> col_dims = chunk_dims(nb, chunk);
    const uint32_t nbc_b = static_cast<uint32_t>(col_dims.size());
    for (uint32_t d : chunk_dims(ny, chunk)) col_dims.push_back(d);
    BlockBuilder<Elt> b(pool, row_dims, col_dims, p);
    std::unique_ptr<BorrowHold<Elt>> holdB;
    if (nb > 0) { holdB = std::make_unique<BorrowHold<Elt>>(pool, B); add_dense_chunks<Elt>(b, holdB->id, B, 0, chunk); }
    const BorrowHold<Elt> holdY(pool, Y);
    add_dense_chunks<Elt>(b, holdY.id, Y, nbc_b, chunk);
    BlockRrefResult<Elt> res;
    {
        const BlockMat<Elt> aug = b.finalize();
        res = rref<Elt>(aug);
    }
    for (uint32_t i = 0; i < res.rank; ++i) {
        const uint32_t c = res.pivots[i];
        if (c >= nb)
            throw std::runtime_error("Vector is not in image of basis (right-hand side " +
                                     std::to_string(c - nb) + ").");
        for (uint32_t t = 0; t < ny; ++t) result.at(t, c) = res.R.at(i, nb + t);
    }
    return result;
}

template<class Elt>
DenseBuf<Elt> solve_multi_rhs(const BlockMat<Elt>& B, const BlockMat<Elt>& Y, const ElimParams& params) {
    ENGINE_CHECK_MSG(B.row_off() == Y.row_off(), "solve_multi_rhs: row partitions differ");
    DenseBuf<Elt> sb, sy;
    const DenseView<const Elt> vb = multivector_view<Elt>(B, sb);
    const DenseView<const Elt> vy = multivector_view<Elt>(Y, sy);
    return solve_multi_rhs<Elt>(B.pool(), B.p(), B.row_dims(), vb, vy, params);
}

// A snapshot of the counters, in a per-calling-thread slot so the public
// `const ElimStats&` signature is unchanged (slot-8 WP-6).
const ElimStats& elim_stats() {
    thread_local ElimStats snap;
    const auto ld = [](const std::atomic<uint64_t>& c) { return c.load(std::memory_order_relaxed); };
    snap.panels = ld(g_stats.panels);
    snap.panels_with_pivots = ld(g_stats.panels_with_pivots);
    snap.dense_rrefs = ld(g_stats.dense_rrefs);
    snap.gemms = ld(g_stats.gemms);
    snap.panel_rows = ld(g_stats.panel_rows);
    snap.gathered_blocks = ld(g_stats.gathered_blocks);
    snap.scalar_blocks_gathered = ld(g_stats.scalar_blocks_gathered);
    return snap;
}
void reset_elim_stats() {
    for (std::atomic<uint64_t>* c : {&g_stats.panels, &g_stats.panels_with_pivots, &g_stats.dense_rrefs,
                                     &g_stats.gemms, &g_stats.panel_rows, &g_stats.gathered_blocks,
                                     &g_stats.scalar_blocks_gathered})
        c->store(0, std::memory_order_relaxed);
}

#define BLK_ELIM_INSTANTIATE(Elt)                                                                     \
    template BlockRrefResult<Elt> rref<Elt>(const BlockMat<Elt>&);                                    \
    template uint32_t rank<Elt>(const BlockMat<Elt>&);                                                \
    template BlockMat<Elt> rref_block<Elt>(const BlockMat<Elt>&, const BlockRrefResult<Elt>&);        \
    template BlockMat<Elt> kernel_basis<Elt>(const BlockMat<Elt>&, const BlockRrefResult<Elt>&, Layout); \
    template BlockMat<Elt> kernel_basis<Elt>(const BlockMat<Elt>&, Layout);                           \
    template BlockMat<Elt> image_basis<Elt>(const BlockMat<Elt>&, const BlockRrefResult<Elt>&, Layout); \
    template BlockMat<Elt> image_basis<Elt>(const BlockMat<Elt>&, Layout);                            \
    template FactorBasisResult<Elt> factor_basis<Elt>(std::shared_ptr<BlockPool<Elt>>, Elt,           \
                                                      const std::vector<uint32_t>&, DenseView<const Elt>, \
                                                      DenseView<const Elt>, const ElimParams&);       \
    template FactorBasisResult<Elt> factor_basis<Elt>(const BlockMat<Elt>&, const BlockMat<Elt>&,     \
                                                      const ElimParams&);                             \
    template DenseBuf<Elt> solve_multi_rhs<Elt>(std::shared_ptr<BlockPool<Elt>>, Elt,                 \
                                                const std::vector<uint32_t>&, DenseView<const Elt>,   \
                                                DenseView<const Elt>, const ElimParams&);             \
    template DenseBuf<Elt> solve_multi_rhs<Elt>(const BlockMat<Elt>&, const BlockMat<Elt>&,           \
                                                const ElimParams&);                                   \
    template DenseView<const Elt> multivector_view<Elt>(const BlockMat<Elt>&, DenseBuf<Elt>&, Layout);

BLK_ELIM_INSTANTIATE(uint32_t)
BLK_ELIM_INSTANTIATE(uint64_t)
#undef BLK_ELIM_INSTANTIATE

} // namespace blk
} // namespace engine
