// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineTapeEliminationValidation.cpp -- the panel-driven elimination on the
// tape (TapeElim.h) entrywise against engine::blk (BlockElim.h) on the grids
// BlockEliminationValidation uses: RREF (R, pivots, rank), kernel_basis,
// image_basis, factor_basis (pivots, independent_relations, basis, projector)
// and solve_multi_rhs (coefficients and the out-of-span rejection).
#include "EngineTapeEliminationValidation.h"

#include <iostream>
#include <stdexcept>

#include "../ModArithmetic/BlockElim.h"
#include "../ModArithmetic/BlockKernels.h"
#include "EngineTestUtil.h"
#include "Executor.h"
#include "TapeElim.h"

namespace {

using namespace engine;
using namespace engine::tape;
using namespace engine::tape::test;

Checker g;
TapeElimStats g_stats;

template<class Elt>
void one_case(Executor<Elt>& ex, Rng& rng, const BlockMat<Elt>& A, const std::string& tag) {
    const Elt p = A.p();
    const ModCtx<Elt> ctx(p);
    // RREF
    const blk::BlockRrefResult<Elt> ref = blk::rref<Elt>(A);
    const blk::BlockRrefResult<Elt> got = rref<Elt>(ex, A, &g_stats);
    g.check(got.rank == ref.rank, tag + " rank");
    g.check(got.pivots == ref.pivots, tag + " pivots");
    g.same(cview(got.R), cview(ref.R), tag + " R");
    // kernel / image
    for (Layout l : {Layout::ColMajor, Layout::RowMajor}) {
        const BlockMat<Elt> kr = blk::kernel_basis<Elt>(A, ref, l), kt = blk::kernel_basis<Elt>(A, got, l);
        g.check(blk::equal<Elt>(kr, kt) && kr.cols() == kt.cols(), tag + " kernel_basis");
        const BlockMat<Elt> ir = blk::image_basis<Elt>(A, ref, l), it = blk::image_basis<Elt>(A, got, l);
        g.check(blk::equal<Elt>(ir, it) && ir.cols() == it.cols(), tag + " image_basis");
    }
    const BlockMat<Elt> kt2 = kernel_basis<Elt>(ex, A);
    g.check(blk::equal<Elt>(kt2, blk::kernel_basis<Elt>(A)), tag + " kernel_basis (driver overload)");
    const BlockMat<Elt> it2 = image_basis<Elt>(ex, A);
    g.check(blk::equal<Elt>(it2, blk::image_basis<Elt>(A)), tag + " image_basis (driver overload)");

    // factor_basis / solve on the column space of A (row partition = A.col_dims())
    const std::vector<uint32_t> row_dims = A.col_dims();
    const uint32_t a = A.cols();
    if (a == 0) return;
    const uint32_t nr = 1 + rng.below(3), ng = 3 + rng.below(6);
    DenseBuf<Elt> Rl = random_buf<Elt>(rng, a, nr, p, Layout::ColMajor);
    DenseBuf<Elt> G = random_buf<Elt>(rng, a, ng, p, Layout::ColMajor);
    // dependencies: a generator equal to a relation, a generator equal to a combination
    for (uint32_t i = 0; i < a; ++i) {
        G.at(i, 1) = Rl.at(i, 0);
        if (ng > 3) G.at(i, 3) = ctx.add(G.at(i, 0), ctx.mul(Elt(2 % p), G.at(i, 2)));
    }
    for (uint32_t chunk : {256u, 5u}) {
        const blk::FactorBasisResult<Elt> fr = blk::factor_basis<Elt>(A.pool(), p, row_dims, cview(Rl), cview(G), blk::ElimParams{chunk});
        const blk::FactorBasisResult<Elt> ft = factor_basis<Elt>(ex, A.pool(), p, row_dims, cview(Rl), cview(G), chunk, &g_stats);
        g.check(fr.pivots == ft.pivots, tag + " factor_basis pivots chunk=" + std::to_string(chunk));
        g.check(fr.independent_relations == ft.independent_relations, tag + " factor_basis independent_relations");
        g.check(blk::equal<Elt>(fr.basis, ft.basis) && fr.basis.cols() == ft.basis.cols(), tag + " factor_basis basis");
        g.same(cview(fr.projector), cview(ft.projector), tag + " factor_basis projector");
        // solve: Y in the span of B = G
        const uint32_t ny = 2 + rng.below(3);
        DenseBuf<Elt> coef = random_buf<Elt>(rng, ng, ny, p);
        DenseBuf<Elt> Y(a, ny, Layout::ColMajor);
        kernels::gemm<Elt>(ctx, cview(G), cview(coef), view(Y), kernels::GemmOp::Set);
        const DenseBuf<Elt> sr = blk::solve_multi_rhs<Elt>(A.pool(), p, row_dims, cview(G), cview(Y), blk::ElimParams{chunk});
        const DenseBuf<Elt> st = solve_multi_rhs<Elt>(ex, A.pool(), p, row_dims, cview(G), cview(Y), chunk, &g_stats);
        g.same(cview(sr), cview(st), tag + " solve_multi_rhs chunk=" + std::to_string(chunk));
        // certificate: G * coords^T == Y
        DenseBuf<Elt> Y2(a, ny, Layout::ColMajor);
        kernels::gemm<Elt>(ctx, cview(G), cview(st).transposed_view(), view(Y2), kernels::GemmOp::Set);
        g.same(cview(Y2), cview(Y), tag + " solve certificate");
        // out of span: both throw the legacy message
        if (fr.pivots.size() < a) {
            DenseBuf<Elt> Yo(Y);
            bool hit = false;
            for (uint32_t tries = 0; tries < 8 && !hit; ++tries) {
                for (uint32_t i = 0; i < a; ++i) Yo.at(i, 0) = rng.elt<Elt>(p);
                std::string m1, m2;
                try { (void)blk::solve_multi_rhs<Elt>(A.pool(), p, row_dims, cview(G), cview(Yo), blk::ElimParams{chunk}); } catch (const std::runtime_error& e) { m1 = e.what(); }
                try { (void)solve_multi_rhs<Elt>(ex, A.pool(), p, row_dims, cview(G), cview(Yo), chunk, &g_stats); } catch (const std::runtime_error& e) { m2 = e.what(); }
                g.check(m1 == m2, tag + " out-of-span verdict agrees");
                hit = !m1.empty();
            }
        }
    }
}

template<class Elt>
void run_width(Executor<Elt>& ex, const std::vector<Elt>& primes, const std::string& w, bool big) {
    uint64_t seed = 0x5EED;
    struct Grid { std::vector<uint32_t> rd, cd; uint32_t density; const char* name; };
    for (Elt p : primes) {
        const std::string tag = "[" + w + " p=" + std::to_string(p) + "]";
        std::vector<Grid> grids{
            {{4, 4, 4}, {4, 4, 4, 4}, 60, "3x4 d=4"},
            {{5, 5}, {5, 5, 5}, 70, "2x3 d=5"},
            {{3, 7, 2}, {5, 1, 6}, 70, "non-uniform"},
            {{3, 0, 2}, {5, 0, 6}, 70, "zero-width"},
            {{6}, {6}, 100, "single block"},
            {{1, 1, 1, 1}, {1, 1, 1}, 80, "scalar blocks"},
            {{17, 17, 17, 17, 17, 17}, {17, 17, 17, 17, 17, 17, 17}, 25, "6x7 d=17"},
            {{17, 17}, std::vector<uint32_t>(11, 17), 25, "2x11 d=17"},
            {std::vector<uint32_t>(13, 17), std::vector<uint32_t>(7, 17), 22, "13x7 d=17"},
        };
        if (big) {
            grids.push_back({{65, 65, 65, 65, 65, 65}, {65, 65, 65, 65, 65, 65, 65}, 25, "6x7 d=65"});
            grids.push_back({{65, 65}, std::vector<uint32_t>(11, 65), 25, "2x11 d=65"});
            grids.push_back({std::vector<uint32_t>(13, 65), std::vector<uint32_t>(7, 65), 20, "13x7 d=65"});
            grids.push_back({{129, 129}, std::vector<uint32_t>(11, 129), 25, "2x11 d=129"});
            grids.push_back({std::vector<uint32_t>(6, 129), std::vector<uint32_t>(7, 129), 22, "6x7 d=129"});
            grids.push_back({std::vector<uint32_t>(13, 129), std::vector<uint32_t>(7, 129), 18, "13x7 d=129"});
        }
        for (const Grid& gr : grids) {
            for (Shape shape : {Shape::CqLike, Shape::ZqLike, Shape::Generic, Shape::Dependent}) {
                const bool heavy = gr.rd.size() * gr.rd[0] >= 6 * 129;
                if (heavy && shape != Shape::CqLike && shape != Shape::ZqLike) continue;
                auto pool = std::make_shared<BlockPool<Elt>>();
                Rng rng(seed++ ^ static_cast<uint64_t>(p));
                const BlockMat<Elt> A = make_input<Elt>(rng, pool, p, shape, gr.rd, gr.cd, gr.density);
                const auto tc = std::chrono::steady_clock::now();
                one_case<Elt>(ex, rng, A, tag + " " + gr.name + " " + shape_name(shape));
                if (std::getenv("ENGINE_TAPE_TRACE"))
                    std::cout << "  time " << seconds_since(tc) << " s  " << tag << " " << gr.name << " " << shape_name(shape) << std::endl;
            }
        }
    }
}

} // namespace

uint32_t EngineTapeEliminationValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    g = Checker(); g_stats = TapeElimStats();
    std::cout << "=== EngineTapeEliminationValidation (tape panel driver vs engine::blk: RREF / kernel / image / factor_basis / solve) ===" << std::endl;
    try {
        Executor<uint32_t> ex32;
        Executor<uint64_t> ex64;
        run_width<uint32_t>(ex32, {134217757u}, "u32", true);
        run_width<uint32_t>(ex32, {2u, 3u, 7u, 23u, 2147483647u, 4294967291u}, "u32", false);
        run_width<uint64_t>(ex64, {134217757ull, 2305843009213693951ull}, "u64", false);
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
        ++g.fails;
    }
    std::cout << "  tape elimination: " << g_stats.panels << " panels (" << g_stats.panels_with_pivots << " with pivots), "
              << g_stats.programs << " programs, " << g_stats.nodes << " nodes, " << g_stats.tasks << " tasks; pivot blocks: "
              << g_stats.identity_pivots << " I, " << g_stats.scalar_pivots << " alpha*I, " << g_stats.general_pivots << " general" << std::endl;
    std::cout << "EngineTapeEliminationValidation: " << g.checks << " checks, " << g.fails << " failed, " << seconds_since(t0) << " s" << std::endl;
    return g.fails == 0 ? 0 : 1;
}
