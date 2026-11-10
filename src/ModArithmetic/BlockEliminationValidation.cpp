// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// BlockEliminationValidation.cpp -- see the header.
#include "BlockEliminationValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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
#include "DenseElim.h"
#include "DenseView.h"
#include "FlatMatrix.h"
#include "Kernels.h"
#include "ModCtx.h"
#include "LegacyConfig.h"   // (S24 CL-L2) oc::legacy::RunLegacyArm (SKIP-ARM)

namespace {

using engine::BlockBuilder;
using engine::BlockMat;
using engine::BlockPool;
using engine::BlockRef;
using engine::DenseBuf;
using engine::DenseView;
using engine::Layout;
using engine::ModCtx;
using engine::cview;
using engine::view;

uint32_t g_fail = 0;
uint64_t g_checks = 0;
void stage(const char* s) { if (std::getenv("BLOCK_ELIM_TRACE")) std::cout << "    stage " << s << std::endl; }
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
    uint32_t below(uint32_t n) { return n == 0 ? 0 : static_cast<uint32_t>(next() % n); }
    template<class Elt> Elt nonzero(Elt p) { return static_cast<Elt>(1 + next() % (p - 1)); }
};

template<class Elt>
DenseBuf<Elt> random_buf(Rng& rng, uint32_t r, uint32_t c, Elt p, Layout layout = Layout::RowMajor) {
    DenseBuf<Elt> b(r, c, layout);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < c; ++j) b.at(i, j) = rng.elt<Elt>(p);
    return b;
}

// ---------------------------------------------------------------------------
// Input generators.  Every one returns a BlockMat over `pool`.
// ---------------------------------------------------------------------------
// (matrix-engine Phase 3) `Diagonal` and `SparseGrid` are the two structural
// shapes ported over from the retired BlockFlatMatrixValidation surface: a
// block-diagonal grid with ~half its diagonal structurally empty, and a
// 100x100 block grid that is 70 % structurally zero.  Neither was reachable
// from the density-driven generators above (they produce 18-70 % density
// *scattered*, never a diagonal and never a grid this sparse at this block
// count), and both drive the parts of the elimination that the dense oracle
// does not: the pivot search across a mostly-empty block grid and the
// structural-zero short-circuits around it.  They are checked here the same way
// as everything else -- entrywise against DenseElim and against the legacy
// BlockFlatMatrix -- which is what makes deleting the legacy block_kernel /
// block_image / get_true_row_echelon_form_as_block_matrix checks safe.
enum class Shape { CqLike, ZqLike, Generic, Dependent, Diagonal, SparseGrid };

template<class Elt>
BlockMat<Elt> make_input(Rng& rng, std::shared_ptr<BlockPool<Elt>> pool, Elt p, Shape shape,
                         const std::vector<uint32_t>& rd, const std::vector<uint32_t>& cd, uint32_t density_pct) {
    const ModCtx<Elt> ctx(p);
    const uint32_t nbr = static_cast<uint32_t>(rd.size()), nbc = static_cast<uint32_t>(cd.size());
    BlockBuilder<Elt> b(pool, rd, cd, p);
    // a small pool of shared dense buffers per (dr, dc)
    std::vector<typename BlockPool<Elt>::Id> shared_ids;
    std::vector<std::pair<uint32_t, uint32_t>> shared_dims;
    auto pick = [&](uint32_t r, uint32_t c) {
        for (size_t s = 0; s < shared_ids.size(); ++s)
            if (shared_dims[s].first == r && shared_dims[s].second == c && rng.below(3) != 0) return shared_ids[s];
        typename BlockPool<Elt>::Id id = pool->intern(random_buf<Elt>(rng, r, c, p));
        shared_ids.push_back(id); shared_dims.emplace_back(r, c);
        return id;
    };
    const Elt neg1 = ctx.neg(Elt(1));
    if (shape == Shape::CqLike || shape == Shape::Dependent) {
        // relation rows: I at lhs, -M at rhs (lhs != rhs) or I - M (lhs == rhs)
        for (uint32_t br = 0; br < nbr; ++br) {
            if (rd[br] == 0) continue;
            uint32_t lhs = rng.below(nbc), rhs = rng.below(nbc);
            if (cd[lhs] != rd[br] || cd[rhs] != rd[br]) { // need square blocks: find matching
                std::vector<uint32_t> ok;
                for (uint32_t c = 0; c < nbc; ++c) if (cd[c] == rd[br]) ok.push_back(c);
                if (ok.empty()) continue;
                lhs = ok[rng.below(static_cast<uint32_t>(ok.size()))];
                rhs = ok[rng.below(static_cast<uint32_t>(ok.size()))];
            }
            const auto id = pick(rd[br], rd[br]);
            if (lhs == rhs) b.set(br, lhs, BlockRef<Elt>::affine(Elt(1), neg1, id));
            else { b.set(br, lhs, BlockRef<Elt>::identity()); b.set(br, rhs, BlockRef<Elt>::dense(id, neg1)); }
            if (shape == Shape::Dependent && br + 1 < nbr && rd[br + 1] == rd[br] && rng.below(2) == 0) {
                // duplicate the row block-wise (scaled): rank-deficient
                const Elt s = rng.elt<Elt>(p);
                if (lhs == rhs) b.set(br + 1, lhs, b.compose(BlockRef<Elt>::affine(Elt(1), neg1, id), Elt(0), s));
                else { b.set(br + 1, lhs, BlockRef<Elt>::scalar(s)); b.set(br + 1, rhs, BlockRef<Elt>::dense(id, ctx.mul(s, neg1))); }
                ++br;
            }
        }
    } else if (shape == Shape::ZqLike) {
        for (uint32_t br = 0; br < nbr; ++br)
            for (uint32_t bc = 0; bc < nbc; ++bc) {
                if (rd[br] == 0 || cd[bc] == 0 || rng.below(100) >= density_pct) continue;
                const Elt bl = rng.elt<Elt>(p), brh = rng.elt<Elt>(p);
                if (rd[br] != cd[bc]) { b.set(br, bc, BlockRef<Elt>::dense(pick(rd[br], cd[bc]), brh == 0 ? Elt(1) : brh)); continue; }
                const auto id = pick(rd[br], cd[bc]);
                b.set(br, bc, BlockRef<Elt>::affine(bl, ctx.neg(brh), id));   // beta_lhs I - beta_rhs M
            }
    } else if (shape == Shape::Diagonal) {
        // block diagonal, ~half the diagonal blocks structurally zero; the
        // non-zero ones are themselves diagonal with non-zero entries.
        for (uint32_t i = 0; i < nbr && i < nbc; ++i) {
            if (rd[i] == 0 || cd[i] == 0 || rd[i] != cd[i]) continue;
            if (rng.below(2) != 0) continue;
            DenseBuf<Elt> d(rd[i], cd[i]);
            for (uint32_t j = 0; j < rd[i]; ++j) d.at(j, j) = rng.nonzero<Elt>(p);
            b.set(i, i, BlockRef<Elt>::dense(pool->intern(std::move(d))));
        }
    } else if (shape == Shape::SparseGrid) {
        // 70 % structurally zero, 20 % diagonal blocks, 10 % dense blocks.
        for (uint32_t br = 0; br < nbr; ++br)
            for (uint32_t bc = 0; bc < nbc; ++bc) {
                if (rd[br] == 0 || cd[bc] == 0) continue;
                const uint32_t type = rng.below(100);
                if (type < 70) continue;
                if (type < 90 && rd[br] == cd[bc]) {
                    DenseBuf<Elt> d(rd[br], cd[bc]);
                    for (uint32_t j = 0; j < rd[br]; ++j) d.at(j, j) = rng.nonzero<Elt>(p);
                    b.set(br, bc, BlockRef<Elt>::dense(pool->intern(std::move(d))));
                } else {
                    b.set(br, bc, BlockRef<Elt>::dense(pick(rd[br], cd[bc])));
                }
            }
    } else {
        for (uint32_t br = 0; br < nbr; ++br)
            for (uint32_t bc = 0; bc < nbc; ++bc) {
                if (rd[br] == 0 || cd[bc] == 0 || rng.below(100) >= density_pct) continue;
                const bool sq = rd[br] == cd[bc];
                switch (rng.below(sq ? 6 : 3)) {
                    case 0: b.set(br, bc, BlockRef<Elt>::dense(pick(rd[br], cd[bc]))); break;
                    case 1: b.set(br, bc, BlockRef<Elt>::dense(pick(rd[br], cd[bc]), neg1)); break;
                    case 2: b.set(br, bc, BlockRef<Elt>::dense(pick(cd[bc], rd[br]), rng.nonzero<Elt>(p), 0, 0, true)); break;
                    case 3: b.set(br, bc, BlockRef<Elt>::identity()); break;
                    case 4: b.set(br, bc, BlockRef<Elt>::scalar(rng.nonzero<Elt>(p))); break;
                    default: b.set(br, bc, BlockRef<Elt>::affine(Elt(1), neg1, pick(rd[br], cd[bc]))); break;
                }
            }
    }
    return b.finalize();
}

template<class Elt>
std::vector<FlatMatrixT<Elt>> columns_as_fm(DenseView<const Elt> M, Elt p) {
    std::vector<FlatMatrixT<Elt>> out;
    for (uint32_t j = 0; j < M.cols(); ++j) {
        std::vector<Elt> col(M.rows());
        for (uint32_t i = 0; i < M.rows(); ++i) col[i] = M.at(i, j);
        out.emplace_back(M.rows(), 1, p, col);
    }
    return out;
}

template<class Elt>
bool same_fm_columns(const std::vector<FlatMatrixT<Elt>>& vecs, DenseView<const Elt> D) {
    if (vecs.size() != D.cols()) return false;
    for (uint32_t t = 0; t < D.cols(); ++t) {
        if (vecs[t].get_rows() != D.rows() || vecs[t].get_cols() != 1) return false;
        for (uint32_t i = 0; i < D.rows(); ++i) if (vecs[t].get_entry(i, 0) != D.at(i, t)) return false;
    }
    return true;
}
template<class Elt>
bool same_fm(const FlatMatrixT<Elt>& f, DenseView<const Elt> D) {
    if (f.get_rows() != D.rows() || f.get_cols() != D.cols()) return false;
    for (uint32_t i = 0; i < D.rows(); ++i)
        for (uint32_t j = 0; j < D.cols(); ++j) if (f.get_entry(i, j) != D.at(i, j)) return false;
    return true;
}
// rows of K (engine kernel_basis: one vector per row) == columns of D
template<class Elt>
bool rows_eq_cols(const DenseBuf<Elt>& K, DenseView<const Elt> D) {
    if (K.rows != D.cols() || K.cols != D.rows()) return false;
    for (uint32_t t = 0; t < K.rows; ++t)
        for (uint32_t i = 0; i < K.cols; ++i) if (K.at(t, i) != D.at(i, t)) return false;
    return true;
}

} // namespace   (S24 CL-L2: the legacy .inc includes its headers at file scope)
#if OC_ENABLE_LEGACY
#include "legacy/BlockEliminationLegacyArms.inc"   // (S24 CL-L2) the legacy BlockFlatMatrix / ScopedLegacy arms
#endif
namespace {

// ---------------------------------------------------------------------------
template<class Elt>
void one_case(const BlockMat<Elt>& A, bool with_legacy, const std::string& tag) {
    if (std::getenv("BLOCK_ELIM_TRACE")) std::cout << "  case " << tag << " (" << A.rows() << "x" << A.cols() << ", nnz " << A.nnz() << ")" << std::endl;
    const Elt p = A.p();
    const ModCtx<Elt>& ctx = A.ctx();
    const uint32_t m = A.rows(), n = A.cols();
    const DenseBuf<Elt> D = A.to_dense();
    auto pool = A.pool();

    // --- RREF / pivots vs DenseElim on the dense matrix
    stage("RREF / pivots vs DenseElim on the dense matrix");
    const engine::blk::BlockRrefResult<Elt> res = engine::blk::rref(A);
    DenseBuf<Elt> Dc(D);
    const engine::elim::RrefResult dres = engine::elim::rref<Elt>(ctx, view(Dc));
    check(res.rank == dres.rank && res.pivots == dres.pivots, tag + " rank/pivots vs DenseElim");
    check(res.R.rows == res.rank && res.R.cols == n, tag + " R shape");
    bool ok = res.rank == dres.rank;
    for (uint32_t i = 0; ok && i < res.rank; ++i)
        for (uint32_t j = 0; j < n; ++j) if (res.R.at(i, j) != Dc.at(i, j)) { ok = false; break; }
    check(ok, tag + " RREF entrywise vs DenseElim");
    // re-blocked RREF as a matrix
    {
        const BlockMat<Elt> Rb = engine::blk::rref_block(A, res);
        check(engine::blk::equal_dense(Rb, cview(res.R)), tag + " rref_block values");
    }

    // --- kernel basis
    stage("kernel basis");
    const BlockMat<Elt> K = engine::blk::kernel_basis(A, res, Layout::ColMajor);
    const DenseBuf<Elt> Kd = K.to_flat(Layout::ColMajor);
    check(K.nbc() == 1 && K.row_off() == A.col_off() && Kd.rows == n && Kd.cols == n - res.rank, tag + " kernel shape / partition");
    {
        const DenseBuf<Elt> Ke = engine::elim::kernel_basis<Elt>(ctx, cview(Dc).sub(0, res.rank, 0, n), res.pivots);
        check(rows_eq_cols(Ke, cview(Kd)), tag + " kernel entrywise vs DenseElim");
        // A * K == 0 through the block kernel
        if (Kd.cols > 0 && m > 0) {
            DenseBuf<Elt> Y(m, Kd.cols, Layout::ColMajor);
            engine::blk::gemm_dense(A, cview(Kd), view(Y));
            check(BlockMat<Elt>::is_zero_view(cview(Y)), tag + " A * kernel == 0");
        }
    }
    // --- image basis
    stage("image basis");
    const BlockMat<Elt> Im = engine::blk::image_basis(A, res, Layout::ColMajor);
    const DenseBuf<Elt> Imd = Im.to_flat(Layout::ColMajor);
    check(Im.nbc() == 1 && Im.row_off() == A.row_off() && Imd.rows == m && Imd.cols == res.rank, tag + " image shape / partition");
    {
        const DenseBuf<Elt> Ie = engine::elim::image_basis<Elt>(cview(D), res.pivots);
        check(rows_eq_cols(Ie, cview(Imd)), tag + " image entrywise vs DenseElim");
    }

    // --- legacy BlockFlatMatrix (pre-Phase-2 code path via ScopedLegacy, and the retrofit)
    stage("legacy BlockFlatMatrix ");
#if OC_ENABLE_LEGACY
    if (with_legacy) legacy_bfm_arm<Elt>(A, res, Kd, Imd, tag);
#endif

    // --- factor_basis: generators = kernel columns (+ a few random columns),
    stage("factor_basis");
    //     relations = combinations of them (+ a random one), ambient = A's columns
    {
        Rng rng(0xFAC7 ^ static_cast<uint64_t>(p) ^ (static_cast<uint64_t>(m) << 20) ^ n);
        const uint32_t a = n;
        const std::vector<uint32_t> row_dims = A.col_dims();
        // The legacy factor_basis_block arm is the old block elimination on a
        // dense a x (nr + ng + a) problem; at a = 1419 it costs ~30 s, so the
        // generator count is capped there (the full-size real problem is
        // covered by BlockModelPipelineValidation against the same legacy).
        const uint32_t f = std::min<uint32_t>(Kd.cols, with_legacy && a > 500 ? 96 : Kd.cols);
        const uint32_t ng = f + 2, nr = (f / 2) + 1;
        DenseBuf<Elt> G(a, ng, Layout::ColMajor), Rl(a, nr, Layout::ColMajor);
        for (uint32_t t = 0; t < f; ++t) for (uint32_t i = 0; i < a; ++i) G.at(i, t) = Kd.at(i, t);
        for (uint32_t t = f; t < ng; ++t) for (uint32_t i = 0; i < a; ++i) G.at(i, t) = rng.elt<Elt>(p);
        // duplicate one generator (dependent generators)
        if (f >= 2) for (uint32_t i = 0; i < a; ++i) G.at(i, f) = G.at(i, 0);
        for (uint32_t t = 0; t + 1 < nr; ++t) {
            // relation = random combination of the generators
            for (uint32_t s = 0; s < ng; ++s) {
                const Elt c = rng.elt<Elt>(p);
                for (uint32_t i = 0; i < a; ++i) Rl.at(i, t) = ctx.add(Rl.at(i, t), ctx.mul(c, G.at(i, s)));
            }
        }
        for (uint32_t i = 0; i < a; ++i) Rl.at(i, nr - 1) = rng.elt<Elt>(p);
        stage("fb dense call");
        const engine::blk::FactorBasisResult<Elt> fb = engine::blk::factor_basis<Elt>(pool, p, row_dims, cview(Rl), cview(G));
        stage("fb dense done");
        // vs DenseElim complement_basis
        DenseBuf<Elt> proj_e;
        const std::vector<uint32_t> gens_e = engine::elim::complement_basis<Elt>(ctx, cview(Rl), cview(G), &proj_e);
        check(fb.pivots == gens_e, tag + " factor_basis pivots vs DenseElim");
        check(fb.projector.same_entries(proj_e), tag + " factor_basis projector entrywise vs DenseElim");
        // basis = chosen G columns
        const DenseBuf<Elt> basis = fb.basis.to_flat(Layout::ColMajor);
        ok = basis.rows == a && basis.cols == fb.pivots.size() && fb.basis.row_off() == A.col_off();
        for (uint32_t t = 0; ok && t < basis.cols; ++t)
            for (uint32_t i = 0; i < a; ++i) if (basis.at(i, t) != G.at(i, fb.pivots[t])) { ok = false; break; }
        check(ok, tag + " factor_basis basis columns");
        stage("fb mv overload");
        {
            const BlockMat<Elt> Gmv = BlockMat<Elt>::from_flat(pool, cview(G), row_dims, p, Layout::ColMajor, true);
            const BlockMat<Elt> Rmv = BlockMat<Elt>::from_flat(pool, cview(Rl), row_dims, p, Layout::ColMajor, true);
            const engine::blk::FactorBasisResult<Elt> fb2 = engine::blk::factor_basis<Elt>(Rmv, Gmv);
            check(fb2.pivots == fb.pivots && fb2.projector.same_entries(fb.projector) && engine::blk::equal(fb2.basis, fb.basis),
                  tag + " factor_basis multi-vector overload");
            // chunk width must not matter (narrow chunks are slow: small problems only)
            if (a <= 300) {
                engine::blk::ElimParams pr; pr.chunk = 5;
                const engine::blk::FactorBasisResult<Elt> fb3 = engine::blk::factor_basis<Elt>(Rmv, Gmv, pr);
                check(fb3.pivots == fb.pivots && fb3.projector.same_entries(fb.projector), tag + " factor_basis chunk=5 identical");
            }
        }
        stage("fb legacy");
#if OC_ENABLE_LEGACY
        if (with_legacy) legacy_factor_basis_arm<Elt>(row_dims, Rl, G, p, fb, basis, tag);
#endif
        stage("fb empty");
        {
            const engine::blk::FactorBasisResult<Elt> fe = engine::blk::factor_basis<Elt>(pool, p, row_dims, cview(Rl), cview(G).sub(0, a, 0, 0));
            check(fe.pivots.empty() && fe.projector.rows == 0 && fe.basis.cols() == 0, tag + " factor_basis with no generators");
        }
        // --- solve_multi_rhs: Y = G_indep * X
    stage("solve_multi_rhs");
        {
            const uint32_t nb = static_cast<uint32_t>(fb.pivots.size());
            const uint32_t ny = 3;
            DenseBuf<Elt> Bm(a, nb, Layout::ColMajor);
            for (uint32_t t = 0; t < nb; ++t) for (uint32_t i = 0; i < a; ++i) Bm.at(i, t) = G.at(i, fb.pivots[t]);
            DenseBuf<Elt> X = random_buf<Elt>(rng, nb, ny, p);
            DenseBuf<Elt> Y(a, ny, Layout::ColMajor);
            if (nb > 0 && a > 0) engine::kernels::gemm<Elt>(ctx, cview(Bm), cview(X), view(Y));
            const DenseBuf<Elt> coords = engine::blk::solve_multi_rhs<Elt>(pool, p, row_dims, cview(Bm), cview(Y));
            ok = coords.rows == ny && coords.cols == nb;
            for (uint32_t t = 0; ok && t < ny; ++t) for (uint32_t j = 0; j < nb; ++j) if (coords.at(t, j) != X.at(j, t)) { ok = false; break; }
            check(ok, tag + " solve_multi_rhs recovers the coordinates");
            const std::vector<std::vector<Elt>> ce = engine::elim::solve_multi_rhs<Elt>(ctx, cview(Bm), cview(Y));
            ok = ce.size() == ny;
            for (uint32_t t = 0; ok && t < ny; ++t) for (uint32_t j = 0; j < nb; ++j) if (ce[t][j] != coords.at(t, j)) { ok = false; break; }
            check(ok, tag + " solve_multi_rhs vs DenseElim");
#if OC_ENABLE_LEGACY
            if (with_legacy && nb > 0) legacy_solve_arm<Elt>(Bm, Y, p, coords, tag);
#endif
            // out-of-span column throws the legacy message
            if (nb < a) {
                DenseBuf<Elt> Y2(a, 1, Layout::ColMajor);
                // a vector outside span(Bm): use a kernel-complement direction -- simplest: a random vector, retry until not in span
                bool threw = false; std::string msg;
                for (uint32_t attempt = 0; attempt < 8 && !threw; ++attempt) {
                    for (uint32_t i = 0; i < a; ++i) Y2.at(i, 0) = rng.elt<Elt>(p);
                    try { (void)engine::blk::solve_multi_rhs<Elt>(pool, p, row_dims, cview(Bm), cview(Y2)); }
                    catch (const std::runtime_error& e) { threw = true; msg = e.what(); }
                }
                if (threw) {
                    std::string msg_e;
                    try { (void)engine::elim::solve_multi_rhs<Elt>(ctx, cview(Bm), cview(Y2)); } catch (const std::runtime_error& e) { msg_e = e.what(); }
                    check(msg == msg_e && msg.rfind("Vector is not in image of basis", 0) == 0, tag + " out-of-span message identical to DenseElim");
                }
            }
        }
    }
}

// (S24 VAL-IMPL, VALSTREAM_audit_1030.md row 7) The grid is split between two names: at the eight non-`big` (prime, width)
// passes the "2x11 d=65" grid (four shapes each) runs in BlockEliminationSlowValidation, everything else in
// BlockEliminationValidation.  A skipped case still consumes its seed, so every case keeps its pre-split input and the two
// names together run exactly the pre-split case set.
enum class Part { Standard, Slow };
bool slow_grid(bool big, const char* name) { return !big && std::string(name) == "2x11 d=65"; }

template<class Elt>
void run_width(const std::vector<Elt>& primes, const std::string& w, bool big, Part part = Part::Standard) {
    uint64_t seed = 0x5EED;
    struct Grid { std::vector<uint32_t> rd, cd; uint32_t density; bool legacy; const char* name; };
    for (Elt p : primes) {
        const std::string tag = "[" + w + " p=" + str(p) + "]";
        std::vector<Grid> grids{
            {{4, 4, 4}, {4, 4, 4, 4}, 60, true, "3x4 d=4"},
            {{5, 5}, {5, 5, 5}, 70, true, "2x3 d=5"},
            {{3, 7, 2}, {5, 1, 6}, 70, true, "non-uniform"},
            {{3, 0, 2}, {5, 0, 6}, 70, true, "zero-width"},
            {{6}, {6}, 100, true, "single block"},
            {{1, 1, 1, 1}, {1, 1, 1}, 80, true, "scalar blocks"},
            {{17, 17, 17, 17, 17, 17}, {17, 17, 17, 17, 17, 17, 17}, 25, true, "6x7 d=17"},
            {{17, 17}, std::vector<uint32_t>(11, 17), 25, true, "2x11 d=17"},
            {std::vector<uint32_t>(13, 17), std::vector<uint32_t>(7, 17), 22, true, "13x7 d=17"},
            {{65, 65, 65, 65, 65, 65}, {65, 65, 65, 65, 65, 65, 65}, 25, big, "6x7 d=65"},
            {{65, 65}, std::vector<uint32_t>(11, 65), 25, big, "2x11 d=65"},
        };
        if (big) {
            grids.push_back({std::vector<uint32_t>(13, 65), std::vector<uint32_t>(7, 65), 20, false, "13x7 d=65"});
            grids.push_back({{129, 129}, std::vector<uint32_t>(11, 129), 25, true, "2x11 d=129"});
            grids.push_back({std::vector<uint32_t>(6, 129), std::vector<uint32_t>(7, 129), 22, false, "6x7 d=129"});
            grids.push_back({std::vector<uint32_t>(13, 129), std::vector<uint32_t>(7, 129), 18, false, "13x7 d=129"});
        }
        // (matrix-engine Phase 3) The two structural grids ported from the
        // retired BlockFlatMatrixValidation shapes.  `big` only: they are the
        // large-grid cases, and the legacy arm is kept on both because it is
        // exactly the pre-Phase-2 block elimination they used to exercise.  The
        // sparse grid is 60x60 blocks rather than 100x100: the legacy
        // factor_basis_block arm inside one_case is cubic in the ambient
        // dimension, and the 1419-wide real problem is already covered against
        // the same legacy by BlockModelPipelineValidation.
        if (big && part == Part::Standard) {
            struct SGrid { uint32_t nb, d; Shape shape; const char* name; };
            for (const SGrid& sg : { SGrid{100, 3, Shape::Diagonal, "block-diagonal 300x300"},
                                     SGrid{60, 10, Shape::SparseGrid, "sparse grid 600x600"} }) {
                const std::vector<uint32_t> dims(sg.nb, sg.d);
                auto pool = std::make_shared<BlockPool<Elt>>();
                Rng rng(seed++ ^ static_cast<uint64_t>(p));
                const BlockMat<Elt> A = make_input<Elt>(rng, pool, p, sg.shape, dims, dims, 100);
                one_case<Elt>(A, true, tag + " " + sg.name);
            }
        }
        for (const Grid& g : grids) {
            for (Shape shape : {Shape::CqLike, Shape::ZqLike, Shape::Generic, Shape::Dependent}) {
                const bool heavy = g.rd.size() * g.rd[0] >= 6 * 129;
                if (heavy && shape != Shape::CqLike && shape != Shape::ZqLike) continue;
                // the legacy arm at 258 x 1419 costs seconds per call: real shape only
                const bool legacy_here = g.legacy && !(g.cd.size() * g.cd[0] >= 1400 && shape != Shape::CqLike);
                const uint64_t case_seed = seed++;
                if (slow_grid(big, g.name) != (part == Part::Slow)) continue;   // the other name's case (its seed consumed)
                auto pool = std::make_shared<BlockPool<Elt>>();
                Rng rng(case_seed ^ static_cast<uint64_t>(p));
                const char* sname = shape == Shape::CqLike ? "Cq" : shape == Shape::ZqLike ? "Zq"
                                  : shape == Shape::Generic ? "generic" : shape == Shape::Dependent ? "dependent"
                                  : shape == Shape::Diagonal ? "diagonal" : "sparse-grid";
                const BlockMat<Elt> A = make_input<Elt>(rng, pool, p, shape, g.rd, g.cd, g.density);
                const auto tc = std::chrono::steady_clock::now();
                one_case<Elt>(A, legacy_here, tag + " " + g.name + " " + sname);
                if (std::getenv("BLOCK_ELIM_TRACE"))
                    std::cout << "  time " << std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count()
                              << " s  " << tag << " " << g.name << " " << sname << std::endl;
            }
        }
    }
}

} // namespace

uint32_t BlockEliminationValidation() {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    g_fail = 0; g_checks = 0;
    std::cout << "=== BlockEliminationValidation (block RREF / kernel / image / factor_basis / solve vs DenseElim vs legacy BlockFlatMatrix) ===" << std::endl;
    (void)oc::legacy::RunLegacyArm("BlockEliminationValidation", "legacy");   // (S24 CL-L2) OFF: the legacy BlockFlatMatrix arms are compiled out
    run_width<uint32_t>({134217757u}, "u32", true);
    run_width<uint32_t>({2u, 3u, 7u, 23u, 2147483647u, 4294967291u}, "u32", false);
    run_width<uint64_t>({134217757ull, 2305843009213693951ull}, "u64", false);
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "BlockEliminationValidation: " << g_checks << " checks, " << g_fail << " failed, " << secs << " s" << std::endl;
    return g_fail == 0 ? 0 : 1;
}

uint32_t BlockEliminationSlowValidation() {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    g_fail = 0; g_checks = 0;
    std::cout << "=== BlockEliminationSlowValidation (the \"2x11 d=65\" grid at the eight non-big (prime, width) passes) ===" << std::endl;
    (void)oc::legacy::RunLegacyArm("BlockEliminationSlowValidation", "legacy");   // (S24 CL-L2)
    run_width<uint32_t>({2u, 3u, 7u, 23u, 2147483647u, 4294967291u}, "u32", false, Part::Slow);
    run_width<uint64_t>({134217757ull, 2305843009213693951ull}, "u64", false, Part::Slow);
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "BlockEliminationSlowValidation: " << g_checks << " checks, " << g_fail << " failed, " << secs << " s" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
