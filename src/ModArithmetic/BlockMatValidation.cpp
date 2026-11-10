// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// BlockMatValidation.cpp -- see the header.
#include "BlockMatValidation.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "BlockElim.h"
#include "BlockKernels.h"
#include "BlockMat.h"
#include "BlockPool.h"
#include "DenseBuf.h"
#include "DenseView.h"
#include "Kernels.h"
#include "ModCtx.h"
#include "RefLinAlg.h"

namespace {

using engine::BlockBuilder;
using engine::BlockKind;
using engine::BlockMat;
using engine::BlockPool;
using engine::BlockRef;
using engine::BlockView;
using engine::DenseBuf;
using engine::DenseView;
using engine::Layout;
using engine::ModCtx;
using engine::cview;
using engine::view;

uint32_t g_fail = 0;
uint64_t g_checks = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
    }
}
template<class T> std::string str(T v) { return std::to_string(v); }

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    template<class Elt> Elt elt(Elt p) { return static_cast<Elt>(next() % p); }
    uint32_t below(uint32_t n) { return static_cast<uint32_t>(next() % n); }
};

template<class Elt>
DenseBuf<Elt> random_buf(Rng& rng, uint32_t r, uint32_t c, Elt p, Layout layout = Layout::RowMajor) {
    DenseBuf<Elt> b(r, c, layout);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < c; ++j) b.at(i, j) = rng.elt<Elt>(p);
    return b;
}

// Dense reference product with the legacy oracle.
template<class Elt>
DenseBuf<Elt> ref_product(Elt p, const DenseBuf<Elt>& A, const DenseBuf<Elt>& B) {
    DenseBuf<Elt> C(A.rows, B.cols);
    if (A.rows && B.cols && A.cols) engine::ref::gemm<Elt>(p, cview(A), cview(B), view(C), 0);
    return C;
}

template<class Elt>
bool same(const DenseBuf<Elt>& a, const DenseBuf<Elt>& b) { return a.same_entries(b); }

// ---------------------------------------------------------------------------
// A random block matrix with all kinds, plus its dense image built by an
// INDEPENDENT hand loop (no BlockMat code involved in the expectation).
// ---------------------------------------------------------------------------
template<class Elt>
struct Sample {
    BlockMat<Elt> M;
    DenseBuf<Elt> dense;
};

template<class Elt>
Sample<Elt> random_block_matrix(Rng& rng, std::shared_ptr<BlockPool<Elt>> pool, Elt p,
                                const std::vector<uint32_t>& rd, const std::vector<uint32_t>& cd,
                                uint32_t density_pct, Layout layout, uint32_t max_shared = 3) {
    const ModCtx<Elt> ctx(p);
    uint32_t rows = 0, cols = 0;
    std::vector<uint32_t> ro(rd.size() + 1, 0), co(cd.size() + 1, 0);
    for (size_t i = 0; i < rd.size(); ++i) { ro[i + 1] = ro[i] + rd[i]; rows += rd[i]; }
    for (size_t i = 0; i < cd.size(); ++i) { co[i + 1] = co[i] + cd[i]; cols += cd[i]; }
    Sample<Elt> out;
    out.dense = DenseBuf<Elt>(rows, cols);
    BlockBuilder<Elt> b(pool, rd, cd, p);
    // a few shared dense buffers (dedup / CSE paths)
    std::vector<typename BlockPool<Elt>::Id> shared;
    std::vector<DenseBuf<Elt>> shared_bufs;
    for (uint32_t br = 0; br < rd.size(); ++br) {
        for (uint32_t bc = 0; bc < cd.size(); ++bc) {
            if (rng.below(100) >= density_pct) continue;
            const uint32_t dr = rd[br], dc = cd[bc];
            if (dr == 0 || dc == 0) continue;
            const bool square = dr == dc;
            uint32_t kind = rng.below(8);
            if (!square && (kind == 0 || kind == 1 || kind == 4 || kind == 5)) kind = 2;
            Elt alpha = 0, beta = 0;
            bool has_data = false, transposed = false;
            switch (kind) {
                case 0: alpha = 1; break;                                   // I
                case 1: alpha = rng.elt<Elt>(p); if (alpha == 0) alpha = 1; break;  // alpha I
                case 2: beta = 1; has_data = true; break;                   // M
                case 3: beta = ctx.neg(Elt(1)); has_data = true; break;     // -M
                case 4: alpha = 1; beta = ctx.neg(Elt(1)); has_data = true; break;  // I - M
                case 5: alpha = rng.elt<Elt>(p); beta = rng.elt<Elt>(p); if (beta == 0) beta = 1; has_data = true; break; // pencil
                case 6: beta = rng.elt<Elt>(p); if (beta == 0) beta = 1; has_data = true; transposed = true; break;   // beta M^T
                default: beta = 1; has_data = true; break;
            }
            BlockRef<Elt> ref;
            DenseBuf<Elt> Mlogical;   // dr x dc logical M^(T?)
            if (has_data) {
                // choose or create a buffer of shape (dc x dr) if transposed else (dr x dc)
                const uint32_t br_ = transposed ? dc : dr, bc_ = transposed ? dr : dc;
                int use = -1;
                for (size_t s = 0; s < shared_bufs.size(); ++s)
                    if (shared_bufs[s].rows == br_ && shared_bufs[s].cols == bc_ && rng.below(2) == 0) { use = static_cast<int>(s); break; }
                typename BlockPool<Elt>::Id id;
                if (use < 0) {
                    DenseBuf<Elt> nb = random_buf<Elt>(rng, br_, bc_, p, layout);
                    if (shared_bufs.size() < max_shared) { shared_bufs.push_back(nb); id = pool->intern(nb); shared.push_back(id); }
                    else id = pool->intern(std::move(nb));
                } else {
                    id = shared[use];
                }
                ref = BlockRef<Elt>::affine(alpha, beta, id, 0, 0, transposed);
                const DenseView<const Elt> pv = pool->view(id);
                Mlogical = DenseBuf<Elt>(dr, dc);
                for (uint32_t i = 0; i < dr; ++i)
                    for (uint32_t j = 0; j < dc; ++j) Mlogical.at(i, j) = transposed ? pv.at(j, i) : pv.at(i, j);
            } else {
                ref = BlockRef<Elt>::scalar(alpha);
            }
            b.set(br, bc, ref);
            // independent expectation
            for (uint32_t i = 0; i < dr; ++i)
                for (uint32_t j = 0; j < dc; ++j) {
                    Elt v = 0;
                    if (has_data) v = ModArithmetic::mod_mul_w(beta, Mlogical.at(i, j), p);
                    if (i == j) v = ModArithmetic::mod_add_w(v, alpha, p);
                    out.dense.at(ro[br] + i, co[bc] + j) = v;
                }
        }
    }
    out.M = b.finalize();
    return out;
}

template<class Elt>
bool dense_equal(const BlockMat<Elt>& M, const DenseBuf<Elt>& D) {
    if (M.rows() != D.rows || M.cols() != D.cols) return false;
    DenseBuf<Elt> got = M.to_dense();
    return got.same_entries(D);
}

// ---------------------------------------------------------------------------
template<class Elt>
void pool_tests(Elt p, const std::string& tag) {
    Rng rng(0xB10C + static_cast<uint64_t>(p));
    // 1 bucket: every intern collides on the bucket, hash chain + compare exercised
    BlockPool<Elt> pool(1);
    DenseBuf<Elt> a = random_buf<Elt>(rng, 5, 7, p);
    DenseBuf<Elt> a_col(5, 7, Layout::ColMajor);
    engine::kernels::copy<Elt>(cview(a), view(a_col));
    DenseBuf<Elt> c = random_buf<Elt>(rng, 5, 7, p);
    if (c.same_entries(a)) c.at(0, 0) = static_cast<Elt>((c.at(0, 0) + 1) % p);
    const auto ia = pool.intern(a);
    const auto ia2 = pool.intern(a);
    const auto iacol = pool.intern(a_col);
    const auto ic = pool.intern(c);
    check(ia == ia2, tag + " pool dedup same content");
    check(ia == iacol, tag + " pool dedup is layout-independent");
    check(ic != ia, tag + " pool distinct content -> distinct id");
    check(pool.num_live() == 2, tag + " pool live count");
    check(pool.refcount(ia) == 0, tag + " fresh entry refcount 0");
    pool.acquire(ia); pool.acquire(ia);
    check(pool.refcount(ia) == 2, tag + " acquire x2");
    pool.release(ia);
    check(pool.is_live(ia) && pool.refcount(ia) == 1, tag + " release keeps live");
    pool.release(ia);
    check(!pool.is_live(ia), tag + " release to zero frees");
    check(pool.num_live() == 1, tag + " live after free");
    // re-intern after free gives a live id (recycled slot allowed)
    const auto ia3 = pool.intern(a);
    check(pool.is_live(ia3) && pool.buf(ia3).same_entries(a), tag + " re-intern after free");
    // hash of a differs from hash of c with overwhelming probability; but check the collision path works:
    check(pool.stats().hash_hits >= 2, tag + " hash hits counted");
    // memoised relayout twin
    const DenseView<const Elt> vcol = pool.view(ia3, Layout::ColMajor);
    check(vcol.buffer()->layout == Layout::ColMajor && engine::same_entries(vcol, cview(a)), tag + " relayout twin values");
    const DenseView<const Elt> vcol2 = pool.view(ia3, Layout::ColMajor);
    check(vcol.buffer() == vcol2.buffer() && pool.stats().relayouts == 1, tag + " relayout memoised (one twin)");
    check(pool.view(ia3, Layout::RowMajor).buffer() == &pool.buf(ia3), tag + " same-layout view is the entry");
    // structure tag carried
    DenseBuf<Elt> tri(4, 4, Layout::RowMajor, 0, engine::Invariant::Canonical, engine::Structure::UpperUnitri);
    const auto it = pool.intern(std::move(tri));
    check(pool.buf(it).structure == engine::Structure::UpperUnitri, tag + " structure tag carried");
    // borrowed
    DenseBuf<Elt> ext = random_buf<Elt>(rng, 6, 3, p, Layout::ColMajor);
    const auto ib = pool.borrow(cview(ext).sub(1, 4, 0, 3));
    check(pool.is_borrowed(ib) && engine::same_entries(pool.view(ib), cview(ext).sub(1, 4, 0, 3)), tag + " borrowed window");
    const auto ib2 = pool.borrow(cview(ext).sub(1, 4, 0, 3));
    check(ib2 != ib, tag + " borrowed never deduplicated");
    // sweep frees unreferenced
    pool.sweep();
    check(pool.num_live() == 0, tag + " sweep frees all unreferenced");
    // rehash path: many entries with a 1-bucket start
    BlockPool<Elt> big(1);
    std::vector<typename BlockPool<Elt>::Id> ids;
    for (uint32_t i = 0; i < 300; ++i) { DenseBuf<Elt> x(2, 2); x.at(0, 0) = static_cast<Elt>(i % p); x.at(1, 1) = static_cast<Elt>((i / p) % p); ids.push_back(big.intern(std::move(x))); }
    bool ok = true;
    for (uint32_t i = 0; i < 300; ++i) { DenseBuf<Elt> x(2, 2); x.at(0, 0) = static_cast<Elt>(i % p); x.at(1, 1) = static_cast<Elt>((i / p) % p); ok = ok && (big.intern(std::move(x)) == ids[i]); }
    check(ok, tag + " rehash keeps dedup exact");
}

// ---------------------------------------------------------------------------
template<class Elt>
void structure_tests(Elt p, uint64_t seed, const std::vector<uint32_t>& rd, const std::vector<uint32_t>& cd,
                     uint32_t density, Layout layout, const std::string& tag) {
    Rng rng(seed);
    auto pool = std::make_shared<BlockPool<Elt>>();
    const ModCtx<Elt> ctx(p);
    Sample<Elt> s = random_block_matrix<Elt>(rng, pool, p, rd, cd, density, layout);
    const BlockMat<Elt>& M = s.M;
    check(dense_equal(M, s.dense), tag + " to_dense == hand-built dense");
    // to_dense in column-major
    { DenseBuf<Elt> dc = M.to_dense(Layout::ColMajor); check(dc.same_entries(s.dense), tag + " to_dense ColMajor"); }
    // get_entry
    bool ok = true;
    for (uint32_t i = 0; i < M.rows() && ok; ++i)
        for (uint32_t j = 0; j < M.cols(); ++j) if (M.get_entry(i, j) != s.dense.at(i, j)) { ok = false; break; }
    check(ok, tag + " get_entry");
    // CSR sortedness
    ok = true;
    for (uint32_t br = 0; br < M.nbr(); ++br)
        for (uint32_t k = M.row_begin(br) + 1; k < M.row_end(br); ++k) if (M.col_at(k - 1) >= M.col_at(k)) ok = false;
    check(ok, tag + " CSR strictly increasing within rows");
    for (uint32_t k = 0; k < M.nnz(); ++k) if (M.ref_at(k).is_zero()) ok = false;
    check(ok, tag + " no Zero refs stored");
    // CSR order determinism: rebuild from the same refs in reversed order
    {
        BlockBuilder<Elt> b(pool, rd, cd, p);
        for (uint32_t br = M.nbr(); br-- > 0;)
            for (uint32_t k = M.row_end(br); k-- > M.row_begin(br);) b.set(br, M.col_at(k), M.ref_at(k));
        BlockMat<Elt> M2 = b.finalize();
        check(M2.row_ptr() == M.row_ptr() && M2.col_idx() == M.col_idx(), tag + " CSR identical for reversed insertion");
        bool same_refs = M2.nnz() == M.nnz();
        for (uint32_t k = 0; same_refs && k < M.nnz(); ++k) same_refs = M.ref_at(k).same_rep(M2.ref_at(k));
        check(same_refs, tag + " refs identical for reversed insertion");
        check(engine::blk::equal(M, M2), tag + " equal() same grid");
    }
    // copy / refcount
    {
        std::vector<uint32_t> before;
        for (uint32_t k = 0; k < M.nnz(); ++k) if (M.ref_at(k).has_data()) before.push_back(pool->refcount(M.ref_at(k).pool_id));
        {
            BlockMat<Elt> C(M);
            bool up = true; size_t t = 0;
            for (uint32_t k = 0; k < M.nnz(); ++k) if (M.ref_at(k).has_data()) { up = up && pool->refcount(M.ref_at(k).pool_id) >= before[t] + 1; ++t; }
            check(up, tag + " copy acquires");
            check(engine::blk::equal(M, C), tag + " copy equal");
        }
        bool back = true; size_t t = 0;
        for (uint32_t k = 0; k < M.nnz(); ++k) if (M.ref_at(k).has_data()) { back = back && pool->refcount(M.ref_at(k).pool_id) == before[t]; ++t; }
        check(back, tag + " copy destruction releases");
    }
    // transpose (metadata)
    {
        BlockMat<Elt> T = M.transposed();
        DenseBuf<Elt> dT(M.cols(), M.rows());
        for (uint32_t i = 0; i < M.rows(); ++i) for (uint32_t j = 0; j < M.cols(); ++j) dT.at(j, i) = s.dense.at(i, j);
        check(dense_equal(T, dT), tag + " transposed() values");
        check(T.nnz() == M.nnz(), tag + " transposed() nnz");
        BlockMat<Elt> TT = T.transposed();
        check(engine::blk::equal(TT, M) && TT.row_ptr() == M.row_ptr() && TT.col_idx() == M.col_idx(), tag + " transpose involution");
    }
    // views
    if (M.nbr() >= 2 && M.nbc() >= 2) {
        const uint32_t br0 = 1, nbr = M.nbr() - 1, bc0 = 0, nbc = M.nbc() - 1;
        BlockView<Elt> V(M, br0, nbr, bc0, nbc);
        BlockMat<Elt> S = V.to_block_mat();
        DenseBuf<Elt> dS(S.rows(), S.cols());
        const uint32_t r0 = M.row_start(br0), c0 = M.col_start(bc0);
        for (uint32_t i = 0; i < S.rows(); ++i) for (uint32_t j = 0; j < S.cols(); ++j) dS.at(i, j) = s.dense.at(r0 + i, c0 + j);
        check(dense_equal(S, dS), tag + " BlockView sub-grid");
        BlockMat<Elt> ST = V.transposed_view().to_block_mat();
        DenseBuf<Elt> dST(S.cols(), S.rows());
        for (uint32_t i = 0; i < S.rows(); ++i) for (uint32_t j = 0; j < S.cols(); ++j) dST.at(j, i) = dS.at(i, j);
        check(dense_equal(ST, dST), tag + " BlockView transposed");
        // at()
        bool okv = true;
        for (uint32_t i = 0; i < V.nbr() && okv; ++i)
            for (uint32_t j = 0; j < V.nbc(); ++j) {
                const BlockRef<Elt>* r = M.find(br0 + i, bc0 + j);
                const BlockRef<Elt> a = V.at(i, j);
                if ((r == nullptr) != a.is_zero()) { okv = false; break; }
                if (r && !r->same_rep(a)) { okv = false; break; }
            }
        check(okv, tag + " BlockView::at");
    }
    // relayout: values unchanged, twins exist in requested layout
    {
        const Layout other = (layout == Layout::RowMajor) ? Layout::ColMajor : Layout::RowMajor;
        BlockMat<Elt> R = M.relayout(other);
        check(dense_equal(R, s.dense), tag + " relayout values");
        bool lay = true;
        for (uint32_t k = 0; k < R.nnz(); ++k) {
            const BlockRef<Elt>& r = R.ref_at(k);
            if (!r.has_data()) continue;
            const uint32_t br = R.block_row_of_index(k);
            const DenseView<const Elt> v = R.data_view_l(r, R.row_dim(br), R.col_dim(R.col_at(k)), other);
            if (v.buffer()->layout != other) lay = false;
        }
        check(lay, tag + " relayout physical layout");
    }
    // add / scale
    {
        Rng rng2(seed ^ 0xADD);
        Sample<Elt> s2 = random_block_matrix<Elt>(rng2, pool, p, rd, cd, density, layout);
        BlockMat<Elt> Sum = engine::blk::add(M, s2.M, false);
        BlockMat<Elt> Dif = engine::blk::add(M, s2.M, true);
        DenseBuf<Elt> dsum(M.rows(), M.cols()), ddif(M.rows(), M.cols());
        for (uint32_t i = 0; i < M.rows(); ++i)
            for (uint32_t j = 0; j < M.cols(); ++j) {
                dsum.at(i, j) = ModArithmetic::mod_add_w(s.dense.at(i, j), s2.dense.at(i, j), p);
                ddif.at(i, j) = ModArithmetic::mod_sub_w(s.dense.at(i, j), s2.dense.at(i, j), p);
            }
        check(dense_equal(Sum, dsum), tag + " add");
        check(dense_equal(Dif, ddif), tag + " sub");
        const Elt al = static_cast<Elt>(rng2.elt<Elt>(p));
        BlockMat<Elt> Sc = engine::blk::scale(M, al);
        DenseBuf<Elt> dsc(M.rows(), M.cols());
        for (uint32_t i = 0; i < M.rows(); ++i) for (uint32_t j = 0; j < M.cols(); ++j) dsc.at(i, j) = ModArithmetic::mod_mul_w(al, s.dense.at(i, j), p);
        check(dense_equal(Sc, dsc), tag + " scale");
        // equal(): different grids, same matrix
        std::vector<uint32_t> rd1{M.rows()}, cd1{M.cols()};
        BlockMat<Elt> One = BlockMat<Elt>::from_dense(pool, cview(s.dense), rd1, cd1, p, Layout::RowMajor, true);
        check(engine::blk::equal(M, One) && engine::blk::equal(One, M), tag + " equal() across grids");
        if (M.rows() && M.cols()) {
            DenseBuf<Elt> d2 = s.dense;
            const uint32_t i = rng2.below(M.rows()), j = rng2.below(M.cols());
            d2.at(i, j) = static_cast<Elt>((d2.at(i, j) + 1) % p);
            BlockMat<Elt> Two = BlockMat<Elt>::from_dense(pool, cview(d2), rd1, cd1, p, Layout::RowMajor, true);
            check(!engine::blk::equal(M, Two), tag + " equal() detects a one-entry difference (other grid)");
            BlockMat<Elt> Three = BlockMat<Elt>::from_dense(pool, cview(d2), rd, cd, p, layout, true);
            check(!engine::blk::equal(M, Three), tag + " equal() detects a one-entry difference (same grid)");
            check(engine::blk::equal_dense(M, cview(s.dense)) && !engine::blk::equal_dense(M, cview(d2)), tag + " equal_dense");
        }
    }
    // materialise_rows / cols
    if (M.rows() >= 3 && M.cols() >= 3) {
        const uint32_t r0 = rng.below(M.rows() - 1), nr = 1 + rng.below(M.rows() - r0);
        DenseBuf<Elt> strip(nr, M.cols());
        engine::blk::materialise_rows(M, r0, nr, view(strip));
        check(engine::same_entries(cview(strip), cview(s.dense).sub(r0, nr, 0, M.cols())), tag + " materialise_rows");
        const uint32_t c0 = rng.below(M.cols() - 1), nc = 1 + rng.below(M.cols() - c0);
        DenseBuf<Elt> cstrip(M.rows(), nc);
        engine::blk::materialise_cols(M, c0, nc, view(cstrip));
        check(engine::same_entries(cview(cstrip), cview(s.dense).sub(0, M.rows(), c0, nc)), tag + " materialise_cols");
    }
}

// ---------------------------------------------------------------------------
template<class Elt>
void gemm_tests(Elt p, uint64_t seed, const std::vector<uint32_t>& rd, const std::vector<uint32_t>& kd,
                const std::vector<uint32_t>& cd, uint32_t density, Layout layout, const std::string& tag) {
    Rng rng(seed);
    auto pool = std::make_shared<BlockPool<Elt>>();
    Sample<Elt> a = random_block_matrix<Elt>(rng, pool, p, rd, kd, density, layout);
    Sample<Elt> b = random_block_matrix<Elt>(rng, pool, p, kd, cd, density, layout);
    const DenseBuf<Elt> expect = ref_product<Elt>(p, a.dense, b.dense);
    engine::blk::reset_gemm_stats();
    BlockMat<Elt> C = engine::blk::gemm(a.M, b.M, layout);
    check(dense_equal(C, expect), tag + " block gemm == dense product");
    check(C.row_off() == a.M.row_off() && C.col_off() == b.M.col_off(), tag + " gemm grid");
    // structural: no all-zero block stored, scalar blocks detected
    bool ok = true;
    for (uint32_t k = 0; k < C.nnz(); ++k) {
        const BlockRef<Elt>& r = C.ref_at(k);
        const uint32_t br = C.block_row_of_index(k), bc = C.col_at(k);
        DenseBuf<Elt> blk(C.row_dim(br), C.col_dim(bc));
        C.materialise(r, C.row_dim(br), C.col_dim(bc), view(blk), false);
        if (BlockMat<Elt>::is_zero_view(cview(blk))) ok = false;
        Elt s;
        if (r.has_data() && C.row_dim(br) == C.col_dim(bc) && BlockMat<Elt>::is_scalar_view(cview(blk), s)) ok = false;
    }
    check(ok, tag + " gemm result kinds canonical");
    // gemm_dense with X = B dense (multi-vector form), all three ops
    {
        DenseBuf<Elt> X = b.dense;
        DenseBuf<Elt> Xc(X.rows, X.cols, Layout::ColMajor);
        engine::kernels::copy<Elt>(cview(X), view(Xc));
        DenseBuf<Elt> Y(a.M.rows(), X.cols, Layout::ColMajor);
        engine::blk::gemm_dense(a.M, cview(Xc), view(Y), engine::kernels::GemmOp::Set);
        check(Y.same_entries(expect), tag + " gemm_dense Set");
        engine::blk::gemm_dense(a.M, cview(Xc), view(Y), engine::kernels::GemmOp::Add);
        DenseBuf<Elt> twice(expect.rows, expect.cols);
        for (uint32_t i = 0; i < expect.rows; ++i) for (uint32_t j = 0; j < expect.cols; ++j) twice.at(i, j) = ModArithmetic::mod_add_w(expect.at(i, j), expect.at(i, j), p);
        check(Y.same_entries(twice), tag + " gemm_dense Add");
        engine::blk::gemm_dense(a.M, cview(Xc), view(Y), engine::kernels::GemmOp::Sub);
        check(Y.same_entries(expect), tag + " gemm_dense Sub");
    }
    // multi-vector product through the block kernel: B as a from_flat multi-vector
    {
        BlockMat<Elt> Bmv = BlockMat<Elt>::from_flat(pool, cview(b.dense), kd, p, Layout::ColMajor, true);
        BlockMat<Elt> Cmv = engine::blk::gemm(a.M, Bmv, Layout::ColMajor);
        check(Cmv.nbc() == 1 && dense_equal(Cmv, expect), tag + " gemm with a multi-vector");
    }
    // transposed product: (A B)^T == B^T A^T (structural transposes)
    {
        BlockMat<Elt> Ct = engine::blk::gemm(b.M.transposed(), a.M.transposed(), layout);
        check(engine::blk::equal(Ct, C.transposed()), tag + " (AB)^T == B^T A^T");
    }
}

// ---------------------------------------------------------------------------
template<class Elt>
void flat_tests(Elt p, uint64_t seed, const std::vector<uint32_t>& rd, uint32_t k, const std::string& tag) {
    Rng rng(seed);
    auto pool = std::make_shared<BlockPool<Elt>>();
    uint32_t dim = 0;
    for (uint32_t d : rd) dim += d;
    // A multi-vector as N blocks of height d_s x k, stored per block (like the
    // legacy BlockFlatMatrix column vectors); a couple of blocks all-zero.
    std::vector<DenseBuf<Elt>> blocks;
    for (size_t s = 0; s < rd.size(); ++s) {
        DenseBuf<Elt> b = random_buf<Elt>(rng, rd[s], k, p);
        if (rng.below(4) == 0) b.zero();
        blocks.push_back(std::move(b));
    }
    // hand-written L1 flat layout: index s*d_V + c for uniform d_V, generally row_off[s] + c
    DenseBuf<Elt> flat(dim, k, Layout::ColMajor);
    {
        uint32_t off = 0;
        for (size_t s = 0; s < rd.size(); ++s) {
            for (uint32_t c = 0; c < rd[s]; ++c)
                for (uint32_t t = 0; t < k; ++t) flat.at(off + c, t) = blocks[s].at(c, t);
            off += rd[s];
        }
    }
    // build via builder (per block) and via from_flat
    BlockBuilder<Elt> b(pool, rd, std::vector<uint32_t>{k}, p);
    for (size_t s = 0; s < rd.size(); ++s) b.set_dense(static_cast<uint32_t>(s), 0, cview(blocks[s]), Elt(1), Elt(0), Layout::ColMajor, true);
    BlockMat<Elt> Mb = b.finalize();
    BlockMat<Elt> Mf = BlockMat<Elt>::from_flat(pool, cview(flat), rd, p, Layout::ColMajor, true);
    check(Mb.is_multivector() && Mf.is_multivector(), tag + " multi-vector nbc == 1");
    check(engine::blk::equal(Mb, Mf), tag + " builder == from_flat");
    DenseBuf<Elt> back = Mf.to_flat(Layout::ColMajor);
    check(back.same_entries(flat), tag + " to_flat(from_flat) round trip");
    DenseBuf<Elt> back_b = Mb.to_flat(Layout::RowMajor);
    check(back_b.same_entries(flat), tag + " builder to_flat == hand L1");
    // L1 identity entry by entry against s*d_V + c (uniform partitions only)
    bool uniform = true;
    for (uint32_t d : rd) if (d != rd[0]) uniform = false;
    if (uniform && !rd.empty()) {
        const uint32_t dV = rd[0];
        bool ok = true;
        for (uint32_t s = 0; s < rd.size() && ok; ++s)
            for (uint32_t c = 0; c < dV && ok; ++c)
                for (uint32_t t = 0; t < k; ++t)
                    if (back.at(s * dV + c, t) != blocks[s].at(c, t) || Mf.get_entry(s * dV + c, t) != blocks[s].at(c, t)) { ok = false; break; }
        check(ok, tag + " L1 identity s*d_V + c");
    }
    // zero-copy contiguous view when no block was dropped
    {
        BlockMat<Elt> Mnd = BlockMat<Elt>::from_flat(pool, cview(flat), rd, p, Layout::ColMajor, false);
        DenseView<const Elt> v;
        const bool has = Mnd.contiguous_view(v);
        check(has || dim == 0, tag + " contiguous_view available without detection");
        if (has) check(engine::same_entries(v, cview(flat)) && v.col_contiguous(), tag + " contiguous_view values, column-contiguous");
        DenseBuf<Elt> scratch;
        const DenseView<const Elt> mv = engine::blk::multivector_view(Mf, scratch);
        check(engine::same_entries(mv, cview(flat)), tag + " multivector_view");
    }
    // from_flat by move (no copy) keeps values
    {
        DenseBuf<Elt> own = flat;
        BlockMat<Elt> Mm = BlockMat<Elt>::from_flat(pool, std::move(own), rd, p, true);
        check(engine::blk::equal(Mm, Mf), tag + " from_flat(DenseBuf&&)");
    }
    // refcount: one backing buffer shared by all windows; freed when the matrix dies
    {
        const size_t live_before = pool->num_live();
        {
            BlockMat<Elt> tmp = BlockMat<Elt>::from_flat(pool, cview(flat), rd, p, Layout::RowMajor, false);
            // at most one new buffer (none when the content deduplicates
            // against Mf's backing buffer, which it does here)
            check(pool->num_live() <= live_before + 1, tag + " from_flat interns at most one buffer");
            DenseView<const Elt> cv;
            check(tmp.contiguous_view(cv) && cv.buffer() == pool->view(tmp.ref_at(0).pool_id).buffer(),
                  tag + " all windows share one backing buffer");
        }
        check(pool->num_live() == live_before, tag + " backing buffer freed with the matrix");
    }
}

// ---------------------------------------------------------------------------
template<class Elt>
void builder_tests(Elt p, uint64_t seed, const std::string& tag) {
    Rng rng(seed);
    auto pool = std::make_shared<BlockPool<Elt>>();
    const ModCtx<Elt> ctx(p);
    const std::vector<uint32_t> rd{4, 0, 3}, cd{4, 3, 0, 4};
    BlockBuilder<Elt> b(pool, rd, cd, p);
    DenseBuf<Elt> M = random_buf<Elt>(rng, 4, 4, p);
    DenseBuf<Elt> N = random_buf<Elt>(rng, 4, 4, p);
    if (BlockMat<Elt>::is_zero_view(cview(M))) M.at(1, 2) = 1;
    const auto idM = pool->intern(M);
    // set then overwrite: refcount of the overwritten block returns to 0 (freed)
    b.set(0, 0, BlockRef<Elt>::dense(idM));
    check(pool->refcount(idM) == 1, tag + " set acquires");
    b.set(0, 0, BlockRef<Elt>::identity());
    check(!pool->is_live(idM), tag + " overwrite releases (no leak, defect #15)");
    // accumulate folding: I + (-M) -> Affine(1, -1, M) without data work
    const auto idM2 = pool->intern(M);
    b.accumulate(0, 0, BlockRef<Elt>::dense(idM2, ctx.neg(Elt(1))));
    {
        const BlockRef<Elt>* r = b.get(0, 0);
        check(r && r->kind == BlockKind::Affine && r->alpha == 1 && r->beta == ctx.neg(Elt(1)) && r->pool_id == idM2, tag + " accumulate folds I + (-M)");
    }
    // accumulate same window: (I - M) + 2M -> I + M
    b.accumulate(0, 0, BlockRef<Elt>::dense(idM2, Elt(2 % p)));
    {
        const BlockRef<Elt>* r = b.get(0, 0);
        const Elt want_beta = ctx.add(ctx.neg(Elt(1)), Elt(2 % p));
        check(r && r->alpha == 1 && r->beta == want_beta && (want_beta != 0 ? r->pool_id == idM2 : r->kind == BlockKind::Scalar), tag + " accumulate folds same window");
    }
    // accumulate a different window: materialised sum
    const auto idN = pool->intern(N);
    b.accumulate(0, 0, BlockRef<Elt>::dense(idN));
    {
        DenseBuf<Elt> want(4, 4);
        const Elt bm = ctx.add(ctx.neg(Elt(1)), Elt(2 % p));
        for (uint32_t i = 0; i < 4; ++i) for (uint32_t j = 0; j < 4; ++j)
            want.at(i, j) = ctx.add(ctx.add(i == j ? Elt(1) : Elt(0), ctx.mul(bm, M.at(i, j))), N.at(i, j));
        DenseBuf<Elt> got(4, 4);
        BlockMat<Elt>::materialise_ref(ctx, *pool, *b.get(0, 0), 4, 4, view(got), false);
        check(got.same_entries(want), tag + " accumulate materialised sum");
    }
    // set_dense with alpha/beta and detection: beta_lhs*I - beta_rhs*M pencil
    const Elt bl = rng.elt<Elt>(p), brr = rng.elt<Elt>(p);
    b.set_dense(0, 3, cview(M), ctx.neg(brr), bl);
    {
        const BlockRef<Elt>* r = b.get(0, 3);
        if (brr == 0 && bl == 0) check(r == nullptr, tag + " pencil 0 -> structural zero");
        else if (brr == 0) check(r && r->kind == BlockKind::Scalar && r->alpha == bl, tag + " pencil with beta_rhs = 0 -> Scalar");
        else check(r && r->has_data() && r->beta == ctx.neg(brr) && r->alpha == bl, tag + " pencil kind");
    }
    // set_dense detection: an all-zero span is a structural zero, alpha*I a Scalar
    DenseBuf<Elt> Z(4, 4);
    b.set_dense(0, 3, cview(Z));
    check(b.get(0, 3) == nullptr, tag + " set_dense zero span -> structural zero");
    DenseBuf<Elt> aI(4, 4);
    for (uint32_t i = 0; i < 4; ++i) aI.at(i, i) = Elt(1);
    b.set_dense(0, 3, cview(aI), Elt(1), Elt(0));
    check(b.get(0, 3) && b.get(0, 3)->kind == BlockKind::Scalar && b.get(0, 3)->alpha == 1, tag + " set_dense I span -> Scalar");
    // set_dense from a row-major vector, a rectangular block
    std::vector<Elt> v(4 * 3);
    for (auto& x : v) x = rng.elt<Elt>(p);
    v[0] = 1;
    b.set_dense(0, 1, v);
    BlockMat<Elt> R = b.finalize();
    check(b.nnz() == 0, tag + " finalize empties the builder");
    bool ok = true;
    for (uint32_t i = 0; i < 4; ++i) for (uint32_t j = 0; j < 3; ++j) if (R.get_entry(i, 4 + j) != v[i * 3 + j]) ok = false;
    check(ok, tag + " set_dense(vector) values");
    check(R.rows() == 7 && R.cols() == 11 && R.nbr() == 3 && R.nbc() == 4, tag + " grid with zero-width blocks");
    check(R.row_dim(1) == 0 && R.col_dim(2) == 0, tag + " zero-width dims");
    check(R.block_row_of(4) == 2 && R.block_col_of(7) == 3, tag + " block_of skips zero-width blocks");
    // out-of-range / non-square scalar are ENGINE_CHECK failures
    {
        bool threw = false;
        try { BlockBuilder<Elt> bb(pool, rd, cd, p); bb.set(0, 1, BlockRef<Elt>::identity()); } catch (const std::logic_error&) { threw = true; }
        check(threw, tag + " alpha*I on a non-square block throws");
    }
}

template<class Elt>
void run_width(const std::vector<Elt>& primes, const std::string& w) {
    for (Elt p : primes) {
        const std::string tag = "[" + w + " p=" + str(p) + "]";
        pool_tests<Elt>(p, tag);
        builder_tests<Elt>(p, 77 + static_cast<uint64_t>(p), tag);
        const std::vector<std::vector<uint32_t>> parts{
            {5, 5, 5}, {3, 7, 2}, {3, 0, 2}, {1, 1, 1, 1}, {8}, {0, 4}, {17, 17, 17, 17}};
        uint64_t seed = 1000 + static_cast<uint64_t>(p % 1000);
        for (const auto& rd : parts)
            for (const auto& cd : parts) {
                for (Layout lay : {Layout::RowMajor, Layout::ColMajor}) {
                    structure_tests<Elt>(p, seed++, rd, cd, 55, lay, tag + " " + str(rd.size()) + "x" + str(cd.size()));
                }
            }
        // gemm: (rd x kd) * (kd x cd), kinds and shared buffers
        const std::vector<std::vector<uint32_t>> gp{{5, 5, 5}, {3, 7, 2}, {4, 0, 4}, {6, 6}, {17, 17}};
        for (const auto& rd : gp)
            for (const auto& kd : gp)
                for (const auto& cd : gp)
                    for (Layout lay : {Layout::RowMajor, Layout::ColMajor})
                        gemm_tests<Elt>(p, seed++, rd, kd, cd, 60, lay, tag + " gemm " + str(rd.size()) + "x" + str(kd.size()) + "x" + str(cd.size()));
        // multi-vectors / L1
        flat_tests<Elt>(p, seed++, {5, 5, 5, 5}, 7, tag + " flat uniform");
        flat_tests<Elt>(p, seed++, {3, 7, 0, 2}, 4, tag + " flat non-uniform");
        flat_tests<Elt>(p, seed++, {17, 17, 17}, 1, tag + " flat k=1");
        flat_tests<Elt>(p, seed++, {129, 129}, 3, tag + " flat d_V=129");
    }
}

} // namespace

uint32_t BlockMatValidation() {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    g_fail = 0; g_checks = 0;
    std::cout << "=== BlockMatValidation (block-sparse data model: pool / refs / CSR / builder / views / kernels / L1) ===" << std::endl;
    run_width<uint32_t>({2u, 3u, 7u, 23u, 134217757u, 2147483647u, 4294967291u}, "u32");
    run_width<uint64_t>({2ull, 7ull, 134217757ull, 4294967291ull, 2305843009213693951ull}, "u64");
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "BlockMatValidation: " << g_checks << " checks, " << g_fail << " failed, " << secs << " s" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
