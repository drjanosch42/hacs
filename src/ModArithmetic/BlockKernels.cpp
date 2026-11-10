// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// BlockKernels.cpp -- see BlockKernels.h.
#include "BlockKernels.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <tuple>
#include <utility>

namespace engine {
namespace blk {

namespace {

// (slot-8 WP-6) The diagnostic counters behind gemm_stats(): relaxed atomics,
// for the reason BlockElim.cpp's counters are (a TSan data race under
// concurrent jobs, slot-8 WP-5 finding 1).  They feed no computed value.
struct AtomicGemmStats {
    std::atomic<uint64_t> dense_products{0}, cse_hits{0}, scalar_folds{0}, axpys{0};
};
AtomicGemmStats g_stats;
inline void bump(std::atomic<uint64_t>& c, uint64_t n = 1) { c.fetch_add(n, std::memory_order_relaxed); }

using kernels::GemmOp;

// Key of a dense data window (pool id, origin, transposed flag).
using WinKey = std::tuple<uint32_t, uint32_t, uint32_t, uint8_t>;
template<class Elt>
WinKey win_key(const BlockRef<Elt>& r) { return WinKey(r.pool_id, r.r0, r.c0, r.flags); }

// Accumulator for one output block: alpha*I (no data) + optional dense part.
template<class Elt>
struct Acc {
    Elt  alpha = 0;
    bool has_dense = false;
    DenseBuf<Elt> dense;
    bool touched = false;
    bool touched_dense_direct = false;
};

template<class Elt>
void ensure_dense(Acc<Elt>& a, uint32_t dr, uint32_t dc) {
    if (!a.has_dense) {
        a.dense = DenseBuf<Elt>(dr, dc);   // zeroed, row-major
        a.has_dense = true;
    }
}

// acc += beta * M  (M the logical dr x dc view)
template<class Elt>
void acc_axpy(const ModCtx<Elt>& ctx, Acc<Elt>& a, uint32_t dr, uint32_t dc, Elt beta,
              DenseView<const Elt> M) {
    if (beta == 0) return;
    ensure_dense(a, dr, dc);
    kernels::axpy<Elt>(ctx, beta, M, view(a.dense), false);
    bump(g_stats.axpys);
}

// Finalise an accumulator into a ref (interned in `pool`), Zero when all-zero.
template<class Elt>
BlockRef<Elt> finish(const ModCtx<Elt>& ctx, BlockPool<Elt>& pool, Acc<Elt>& a, uint32_t dr, uint32_t dc,
                     Layout out_layout) {
    using Ref = BlockRef<Elt>;
    if (!a.has_dense) return Ref::scalar(a.alpha);
    if (a.alpha != 0) {
        ENGINE_CHECK(dr == dc);
        for (uint32_t i = 0; i < dr; ++i) a.dense.at(i, i) = ctx.add(a.dense.at(i, i), a.alpha);
    }
    Elt s;
    if (BlockMat<Elt>::is_zero_view(cview(a.dense))) return Ref::zero();
    if (dr == dc && BlockMat<Elt>::is_scalar_view(cview(a.dense), s)) return Ref::scalar(s);
    if (out_layout == Layout::RowMajor) return Ref::dense(pool.intern(std::move(a.dense)));
    return Ref::dense(pool.intern_view(cview(a.dense), out_layout));
}

} // namespace

template<class Elt>
BlockMat<Elt> gemm(const BlockMat<Elt>& A, const BlockMat<Elt>& B, Layout out_layout) {
    using Ref = BlockRef<Elt>;
    ENGINE_CHECK_MSG(A.col_off() == B.row_off(), "blk::gemm: A's column partition != B's row partition");
    ENGINE_CHECK_MSG(A.p() == B.p(), "blk::gemm: modulus mismatch");
    ENGINE_CHECK_MSG(A.ctx().is_field() == B.ctx().is_field(), "blk::gemm: ring mismatch (ENGINE_API.md Sec. 13.7)");
    const ModCtx<Elt>& ctx = A.ctx();
    BlockPool<Elt>& pool = A.pool_mut();
    const uint32_t nbc = B.nbc();

    // (slot-8 WP-3) the result carries A's context (Sec. 13.7), not ModCtx(p).
    BlockBuilder<Elt> out(A.pool(), A.row_dims(), B.col_dims(), A.ctx());

    // Per-call CSE cache of dense products M1 * M2 (unscaled), keyed on the two
    // data windows.  std::map: ordered, no hash container (02 Sec. 6.5(5)).
    std::map<std::pair<WinKey, WinKey>, DenseBuf<Elt>> cache;

    std::vector<Acc<Elt>> acc(nbc);
    std::vector<uint32_t> touched;

    for (uint32_t br = 0; br < A.nbr(); ++br) {
        const uint32_t dr = A.row_dim(br);
        touched.clear();
        for (uint32_t ka = A.row_begin(br); ka < A.row_end(br); ++ka) {
            const uint32_t k = A.col_at(ka);
            const uint32_t dk = A.col_dim(k);
            const Ref& ra = A.ref_at(ka);
            for (uint32_t kb = B.row_begin(k); kb < B.row_end(k); ++kb) {
                const uint32_t bc = B.col_at(kb);
                const uint32_t dc = B.col_dim(bc);
                const Ref& rb = B.ref_at(kb);
                if (dr == 0 || dk == 0 || dc == 0) continue;
                Acc<Elt>& a = acc[bc];
                if (!a.touched) { a.touched = true; touched.push_back(bc); }
                // (a1 I + b1 M1)(a2 I + b2 M2) = a1a2 I + a1b2 M2 + b1a2 M1 + b1b2 M1M2
                const Elt a1 = ra.alpha, b1 = ra.beta, a2 = rb.alpha, b2 = rb.beta;
                if (a1 != 0 && a2 != 0) { a.alpha = ctx.add(a.alpha, ctx.mul(a1, a2)); bump(g_stats.scalar_folds); }
                if (a1 != 0 && b2 != 0) acc_axpy<Elt>(ctx, a, dr, dc, ctx.mul(a1, b2), B.data_view(rb, dk, dc));
                if (b1 != 0 && a2 != 0) acc_axpy<Elt>(ctx, a, dr, dc, ctx.mul(b1, a2), A.data_view(ra, dr, dk));
                if (b1 != 0 && b2 != 0) {
                    const Elt bb = ctx.mul(b1, b2);
                    const auto key = std::make_pair(win_key(ra), win_key(rb));
                    auto it = cache.find(key);
                    if (it == cache.end()) {
                        DenseBuf<Elt> prod(dr, dc);
                        kernels::gemm<Elt>(ctx, A.data_view(ra, dr, dk), B.data_view(rb, dk, dc), view(prod), GemmOp::Set);
                        bump(g_stats.dense_products);
                        it = cache.emplace(key, std::move(prod)).first;
                    } else {
                        bump(g_stats.cse_hits);
                    }
                    if (!a.has_dense && bb == 1 && !a.touched_dense_direct) {
                        // first dense term: take the product directly (no axpy)
                        a.dense = it->second;   // copy (the cache keeps its own)
                        a.has_dense = true;
                        a.touched_dense_direct = true;
                    } else {
                        acc_axpy<Elt>(ctx, a, dr, dc, bb, cview(it->second));
                    }
                }
            }
        }
        // Emit in ascending block-column order (deterministic).
        std::sort(touched.begin(), touched.end());
        for (uint32_t bc : touched) {
            Acc<Elt>& a = acc[bc];
            const Ref r = finish<Elt>(ctx, pool, a, dr, B.col_dim(bc), out_layout);
            if (!r.is_zero()) out.set(br, bc, r);
            a = Acc<Elt>();
        }
    }
    return out.finalize();
}

template<class Elt>
void gemm_dense(const BlockMat<Elt>& A, DenseView<const Elt> X, DenseView<Elt> Y, kernels::GemmOp op) {
    using Ref = BlockRef<Elt>;
    ENGINE_CHECK_MSG(X.rows() == A.cols() && Y.rows() == A.rows() && X.cols() == Y.cols(),
                     "blk::gemm_dense: shape mismatch");
    const ModCtx<Elt>& ctx = A.ctx();
    const uint32_t w = X.cols();
    if (op == GemmOp::Set) BlockMat<Elt>::zero_view(Y);
    if (w == 0) return;
    const bool sub = (op == GemmOp::Sub);
    // CSE: (window, bc) -> M * X[bc]
    std::map<std::pair<WinKey, uint32_t>, DenseBuf<Elt>> cache;
    for (uint32_t br = 0; br < A.nbr(); ++br) {
        const uint32_t dr = A.row_dim(br);
        if (dr == 0) continue;
        DenseView<Elt> y = Y.sub(A.row_start(br), dr, 0, w);
        for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k) {
            const uint32_t bc = A.col_at(k);
            const uint32_t dc = A.col_dim(bc);
            if (dc == 0) continue;
            const Ref& r = A.ref_at(k);
            const DenseView<const Elt> x = X.sub(A.col_start(bc), dc, 0, w);
            if (r.alpha != 0) {
                ENGINE_CHECK(dr == dc);
                kernels::axpy<Elt>(ctx, r.alpha, x, y, sub);
                bump(g_stats.axpys);
            }
            if (r.beta != 0) {
                const auto key = std::make_pair(win_key(r), bc);
                auto it = cache.find(key);
                if (it == cache.end()) {
                    DenseBuf<Elt> prod(dr, w);
                    kernels::gemm<Elt>(ctx, A.data_view(r, dr, dc), x, view(prod), GemmOp::Set);
                    bump(g_stats.dense_products);
                    it = cache.emplace(key, std::move(prod)).first;
                } else {
                    bump(g_stats.cse_hits);
                }
                kernels::axpy<Elt>(ctx, r.beta, cview(it->second), y, sub);
                bump(g_stats.axpys);
            }
        }
    }
}

template<class Elt>
BlockMat<Elt> add(const BlockMat<Elt>& A, const BlockMat<Elt>& B, bool subtract, Layout out_layout) {
    ENGINE_CHECK_MSG(A.same_grid(B), "blk::add: grids differ");
    ENGINE_CHECK_MSG(A.p() == B.p(), "blk::add: modulus mismatch");
    ENGINE_CHECK_MSG(A.ctx().is_field() == B.ctx().is_field(), "blk::add: ring mismatch (ENGINE_API.md Sec. 13.7)");
    const ModCtx<Elt>& ctx = A.ctx();
    const Elt s = subtract ? ctx.neg(Elt(1)) : Elt(1);
    BlockBuilder<Elt> out(A.pool(), A.row_dims(), A.col_dims(), A.ctx());
    for (uint32_t br = 0; br < A.nbr(); ++br) {
        uint32_t ka = A.row_begin(br), kb = B.row_begin(br);
        const uint32_t ea = A.row_end(br), eb = B.row_end(br);
        while (ka < ea || kb < eb) {
            const uint32_t ca = ka < ea ? A.col_at(ka) : 0xFFFFFFFFu;
            const uint32_t cb = kb < eb ? B.col_at(kb) : 0xFFFFFFFFu;
            if (ca < cb) { out.set(br, ca, A.ref_at(ka)); ++ka; }
            else if (cb < ca) { out.set(br, cb, out.compose(B.ref_at(kb), Elt(0), s)); ++kb; }
            else {
                out.set(br, ca, A.ref_at(ka));
                out.accumulate(br, ca, out.compose(B.ref_at(kb), Elt(0), s));
                ++ka; ++kb;
            }
        }
    }
    (void)out_layout;
    return out.finalize();
}

template<class Elt>
BlockMat<Elt> scale(const BlockMat<Elt>& A, Elt alpha) {
    BlockBuilder<Elt> out(A.pool(), A.row_dims(), A.col_dims(), A.ctx());
    if (alpha == 0) return out.finalize();
    for (uint32_t br = 0; br < A.nbr(); ++br)
        for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k)
            out.set(br, A.col_at(k), out.compose(A.ref_at(k), Elt(0), alpha));
    return out.finalize();
}

template<class Elt>
void materialise_rows(const BlockMat<Elt>& A, uint32_t r0, uint32_t nr, DenseView<Elt> dst) {
    ENGINE_CHECK(dst.rows() == nr && dst.cols() == A.cols() && static_cast<uint64_t>(r0) + nr <= A.rows());
    BlockMat<Elt>::zero_view(dst);
    if (nr == 0 || A.cols() == 0) return;
    // Walk the block rows intersecting [r0, r0+nr).
    for (uint32_t br = 0; br < A.nbr(); ++br) {
        const uint32_t bs = A.row_start(br), dr = A.row_dim(br);
        if (dr == 0 || bs >= r0 + nr || bs + dr <= r0) continue;
        const uint32_t lo = (r0 > bs) ? r0 - bs : 0;                   // first block-local row
        const uint32_t hi = std::min<uint32_t>(dr, r0 + nr - bs);      // one past last
        for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k) {
            const uint32_t bc = A.col_at(k), dc = A.col_dim(bc);
            if (dc == 0) continue;
            const BlockRef<Elt>& r = A.ref_at(k);
            DenseView<Elt> d = dst.sub(bs + lo - r0, hi - lo, A.col_start(bc), dc);
            if (lo == 0 && hi == dr) { A.materialise(r, dr, dc, d, false); continue; }
            // partial rows of the block
            if (r.has_data()) {
                kernels::copy<Elt>(A.data_view(r, dr, dc).sub(lo, hi - lo, 0, dc), d);
                if (r.beta != 1) kernels::scal<Elt>(A.ctx(), r.beta, d);
            }
            if (r.alpha != 0)
                for (uint32_t i = lo; i < hi && i < dc; ++i)
                    d.at(i - lo, i) = A.ctx().add(d.at(i - lo, i), r.alpha);
        }
    }
}

template<class Elt>
void materialise_cols(const BlockMat<Elt>& A, uint32_t c0, uint32_t nc, DenseView<Elt> dst) {
    ENGINE_CHECK(dst.cols() == nc && dst.rows() == A.rows() && static_cast<uint64_t>(c0) + nc <= A.cols());
    BlockMat<Elt>::zero_view(dst);
    if (nc == 0 || A.rows() == 0) return;
    for (uint32_t br = 0; br < A.nbr(); ++br) {
        const uint32_t dr = A.row_dim(br);
        if (dr == 0) continue;
        for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k) {
            const uint32_t bc = A.col_at(k), cs = A.col_start(bc), dc = A.col_dim(bc);
            if (dc == 0 || cs >= c0 + nc || cs + dc <= c0) continue;
            const uint32_t lo = (c0 > cs) ? c0 - cs : 0;
            const uint32_t hi = std::min<uint32_t>(dc, c0 + nc - cs);
            const BlockRef<Elt>& r = A.ref_at(k);
            DenseView<Elt> d = dst.sub(A.row_start(br), dr, cs + lo - c0, hi - lo);
            if (lo == 0 && hi == dc) { A.materialise(r, dr, dc, d, false); continue; }
            if (r.has_data()) {
                kernels::copy<Elt>(A.data_view(r, dr, dc).sub(0, dr, lo, hi - lo), d);
                if (r.beta != 1) kernels::scal<Elt>(A.ctx(), r.beta, d);
            }
            if (r.alpha != 0)
                for (uint32_t j = lo; j < hi && j < dr; ++j)
                    d.at(j, j - lo) = A.ctx().add(d.at(j, j - lo), r.alpha);
        }
    }
}

template<class Elt>
bool equal(const BlockMat<Elt>& A, const BlockMat<Elt>& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    if (A.p() != B.p()) return false;
    if (A.same_grid(B)) {
        for (uint32_t br = 0; br < A.nbr(); ++br) {
            const uint32_t dr = A.row_dim(br);
            uint32_t ka = A.row_begin(br), kb = B.row_begin(br);
            const uint32_t ea = A.row_end(br), eb = B.row_end(br);
            while (ka < ea || kb < eb) {
                const uint32_t ca = ka < ea ? A.col_at(ka) : 0xFFFFFFFFu;
                const uint32_t cb = kb < eb ? B.col_at(kb) : 0xFFFFFFFFu;
                const uint32_t bc = std::min(ca, cb);
                const uint32_t dc = A.col_dim(bc);
                const BlockRef<Elt>* ra = (ca == bc) ? &A.ref_at(ka) : nullptr;
                const BlockRef<Elt>* rb = (cb == bc) ? &B.ref_at(kb) : nullptr;
                if (ca == bc) ++ka;
                if (cb == bc) ++kb;
                if (dr == 0 || dc == 0) continue;
                if (ra && rb && A.pool().get() == B.pool().get() && ra->same_rep(*rb)) continue;
                DenseBuf<Elt> da(dr, dc), db(dr, dc);
                if (ra) A.materialise(*ra, dr, dc, view(da), false);
                if (rb) B.materialise(*rb, dr, dc, view(db), false);
                if (!da.same_entries(db)) return false;
            }
        }
        return true;
    }
    // Different grids: compare dense strips of A's block rows.
    for (uint32_t br = 0; br < A.nbr(); ++br) {
        const uint32_t dr = A.row_dim(br);
        if (dr == 0) continue;
        DenseBuf<Elt> sa(dr, A.cols()), sb(dr, A.cols());
        materialise_rows<Elt>(A, A.row_start(br), dr, view(sa));
        materialise_rows<Elt>(B, A.row_start(br), dr, view(sb));
        if (!sa.same_entries(sb)) return false;
    }
    return true;
}

template<class Elt>
bool equal_dense(const BlockMat<Elt>& A, DenseView<const Elt> D) {
    if (A.rows() != D.rows() || A.cols() != D.cols()) return false;
    for (uint32_t br = 0; br < A.nbr(); ++br) {
        const uint32_t dr = A.row_dim(br);
        if (dr == 0) continue;
        DenseBuf<Elt> sa(dr, A.cols());
        materialise_rows<Elt>(A, A.row_start(br), dr, view(sa));
        if (!same_entries(cview(sa), D.sub(A.row_start(br), dr, 0, A.cols()))) return false;
    }
    return true;
}

// A snapshot, in a per-calling-thread slot so the public `const GemmStats&`
// signature is unchanged (slot-8 WP-6).
const GemmStats& gemm_stats() {
    thread_local GemmStats snap;
    snap.dense_products = g_stats.dense_products.load(std::memory_order_relaxed);
    snap.cse_hits = g_stats.cse_hits.load(std::memory_order_relaxed);
    snap.scalar_folds = g_stats.scalar_folds.load(std::memory_order_relaxed);
    snap.axpys = g_stats.axpys.load(std::memory_order_relaxed);
    return snap;
}
void reset_gemm_stats() {
    for (std::atomic<uint64_t>* c : {&g_stats.dense_products, &g_stats.cse_hits, &g_stats.scalar_folds, &g_stats.axpys})
        c->store(0, std::memory_order_relaxed);
}

#define BLK_INSTANTIATE(Elt)                                                                   \
    template BlockMat<Elt> gemm<Elt>(const BlockMat<Elt>&, const BlockMat<Elt>&, Layout);     \
    template void gemm_dense<Elt>(const BlockMat<Elt>&, DenseView<const Elt>, DenseView<Elt>, \
                                  kernels::GemmOp);                                            \
    template BlockMat<Elt> add<Elt>(const BlockMat<Elt>&, const BlockMat<Elt>&, bool, Layout); \
    template BlockMat<Elt> scale<Elt>(const BlockMat<Elt>&, Elt);                              \
    template bool equal<Elt>(const BlockMat<Elt>&, const BlockMat<Elt>&);                      \
    template bool equal_dense<Elt>(const BlockMat<Elt>&, DenseView<const Elt>);                \
    template void materialise_rows<Elt>(const BlockMat<Elt>&, uint32_t, uint32_t, DenseView<Elt>); \
    template void materialise_cols<Elt>(const BlockMat<Elt>&, uint32_t, uint32_t, DenseView<Elt>);

BLK_INSTANTIATE(uint32_t)
BLK_INSTANTIATE(uint64_t)
// (hecke-extensions slot 11, WP-18a; ENGINE_API.md Sec. 13.12.3) the block
// model's composite surface for the composite-only u128 element type: the
// Sec. 13.7 ModCtx overloads (BlockMat / BlockBuilder, header-only) and these
// kernels, on A.ctx() (a u128 context is never a field).  BlockElim (field-only,
// Sec. 13.3) is not instantiated for u128.
BLK_INSTANTIATE(u128)
#undef BLK_INSTANTIATE

} // namespace blk
} // namespace engine
