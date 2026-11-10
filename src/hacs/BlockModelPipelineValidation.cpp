// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "BlockModelPipelineValidation.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../ModArithmetic/BlockElim.h"
#include "../ModArithmetic/LegacyConfig.h"   // (S24 CL-L2) RunLegacyArm (SKIP-ARM)
#include "../ModArithmetic/BlockKernels.h"
#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/DenseView.h"
#include "../ModArithmetic/FlatMatrix.h"
#include "../ModArithmetic/Matrix.h"
#include "EquivariantCohomologyOnK.h"
#include "LocalSystemSL2Z.h"
#include "SimplicialCohomology.h"

// See the header.

namespace {

using engine::BlockBuilder;
using engine::BlockMat;
using engine::BlockPool;
using engine::BlockRef;
using engine::DenseBuf;
using engine::DenseView;
using engine::Layout;
using engine::cview;
using engine::view;
using Clock = std::chrono::steady_clock;

uint32_t g_fail = 0;
uint64_t g_checks = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
    }
}
template<class T> std::string str(T v) { return std::to_string(v); }
double secs_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// SimplexRelation::M is a FlatMatrix (since the Matrix retirement).
uint32_t rel_entry(const FlatMatrix& M, uint32_t r, uint32_t c) { return M.get_entry(r, c); }
uint32_t rel_rows(const FlatMatrix& M) { return M.get_rows(); }
uint32_t rel_cols(const FlatMatrix& M) { return M.get_cols(); }

template<class M>
bool is_identity(const M& m, uint32_t d) {
    if (rel_rows(m) != d || rel_cols(m) != d) return false;
    for (uint32_t r = 0; r < d; ++r)
        for (uint32_t c = 0; c < d; ++c)
            if (rel_entry(m, r, c) != (r == c ? 1u : 0u)) return false;
    return true;
}
template<class M>
DenseBuf<uint32_t> to_buf(const M& m) {
    DenseBuf<uint32_t> b(rel_rows(m), rel_cols(m));
    for (uint32_t r = 0; r < b.rows; ++r)
        for (uint32_t c = 0; c < b.cols; ++c) b.at(r, c) = rel_entry(m, r, c);
    return b;
}

std::vector<LocalSystemSimplicialComplex::SimplexRelation> BuildRelations(const SimplicialComplex& K,
                                                                          uint32_t n_sym, uint32_t ell) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) rels.push_back(datum.rel);
    return rels;
}

std::vector<const LocalSystemSimplicialComplex::SimplexRelation*> live_relations(
        const LocalSystemSimplicialComplex& lssc, uint32_t q) {
    const uint32_t d = lssc.get_representation_dim();
    const bool shortcircuit = EquivariantCohomologyOptions::vacuous_relation_shortcircuit_enabled();
    std::vector<const LocalSystemSimplicialComplex::SimplexRelation*> live;
    for (const auto& rel : lssc.get_user_defined_simplex_relations()) {
        if (rel.q_dim != q) continue;
        if (shortcircuit && rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq && is_identity(rel.M, d)) continue;
        live.push_back(&rel);
    }
    return live;
}

// Flatten a list of N_q x d_V cochains into an L1 column-major dim x k buffer.
DenseBuf<uint32_t> flatten_cochains(const std::vector<FlatMatrix>& basis, uint32_t N_q, uint32_t d) {
    DenseBuf<uint32_t> out(N_q * d, static_cast<uint32_t>(basis.size()), Layout::ColMajor);
    for (uint32_t t = 0; t < basis.size(); ++t)
        for (uint32_t s = 0; s < N_q; ++s)
            for (uint32_t k = 0; k < d; ++k) out.at(s * d + k, t) = basis[t].get_entry(s, k);
    return out;
}
bool same_fm(const FlatMatrix& f, DenseView<const uint32_t> D) {
    if (f.get_rows() != D.rows() || f.get_cols() != D.cols()) return false;
    for (uint32_t i = 0; i < D.rows(); ++i)
        for (uint32_t j = 0; j < D.cols(); ++j) if (f.get_entry(i, j) != D.at(i, j)) return false;
    return true;
}

} // namespace   (S24 CL-L2: the legacy .inc includes its headers at file scope)
#if OC_ENABLE_LEGACY
#include "legacy/BlockModelPipelineLegacyArms.inc"   // (S24 CL-L2) the legacy BlockFlatMatrix / factor_basis_block halves
#endif
namespace {

// ---------------------------------------------------------------------------
void kernel_part(const LocalSystemSimplicialComplex& lssc, uint32_t q, const std::string& tag) {
    auto pool = std::make_shared<BlockPool<uint32_t>>();
    const auto t_asm = Clock::now();
    const BlockMat<uint32_t> A = BuildACqBlockMat(pool, lssc, q);
    const double s_asm = secs_since(t_asm);
    std::cout << "  " << tag << ": A_C" << q << " " << A.rows() << " x " << A.cols() << ", " << A.nnz() << " blocks, "
              << pool->num_live() << " distinct dense blocks, assembly " << s_asm << " s" << std::endl;

    engine::blk::reset_elim_stats();
    const auto t0 = Clock::now();
    const engine::blk::BlockRrefResult<uint32_t> res = engine::blk::rref(A);
    const double s_rref = secs_since(t0);
    const auto t1 = Clock::now();
    const BlockMat<uint32_t> K = engine::blk::kernel_basis(A, res, Layout::ColMajor);
    const double s_ker = secs_since(t1);
    const engine::blk::ElimStats st = engine::blk::elim_stats();
    std::cout << "    model: rank " << res.rank << ", rref " << s_rref << " s, kernel_basis " << s_ker
              << " s  (panels " << st.panels << ", with pivots " << st.panels_with_pivots << ", gemm calls " << st.gemms
              << ", dense rrefs " << st.dense_rrefs << ", blocks gathered " << st.gathered_blocks
              << " of which scalar " << st.scalar_blocks_gathered << ")" << std::endl;
    DenseBuf<uint32_t> scratch;
    const DenseView<const uint32_t> Kv = engine::blk::multivector_view(K, scratch);
    check(K.nbc() == 1 && K.row_off() == A.col_off() && Kv.cols() == A.cols() - res.rank, tag + " kernel shape");
    {
        DenseBuf<uint32_t> Y(A.rows(), Kv.cols(), Layout::ColMajor);
        const auto t2 = Clock::now();
        engine::blk::gemm_dense(A, Kv, view(Y));
        std::cout << "    model: A * K (block gemm_dense, " << Kv.cols() << " vectors) " << secs_since(t2) << " s" << std::endl;
        check(BlockMat<uint32_t>::is_zero_view(cview(Y)), tag + " A * K == 0");
    }

    // dense DenseElim on the flattened matrix (reference)
    {
        DenseBuf<uint32_t> D = A.to_dense();
        const auto t3 = Clock::now();
        const engine::elim::RrefResult dres = engine::elim::rref<uint32_t>(A.ctx(), view(D));
        const DenseBuf<uint32_t> Ke = engine::elim::kernel_basis<uint32_t>(A.ctx(), cview(D).sub(0, dres.rank, 0, A.cols()), dres.pivots);
        std::cout << "    dense DenseElim on the flattened matrix: rref + kernel " << secs_since(t3) << " s" << std::endl;
        check(dres.rank == res.rank && dres.pivots == res.pivots, tag + " pivots vs DenseElim");
        bool ok = Ke.rows == Kv.cols();
        for (uint32_t t = 0; ok && t < Ke.rows; ++t)
            for (uint32_t i = 0; i < Ke.cols; ++i) if (Ke.at(t, i) != Kv.at(i, t)) { ok = false; break; }
        check(ok, tag + " kernel entrywise vs DenseElim");
    }

#if OC_ENABLE_LEGACY
    legacy_kernel_arm(lssc, q, Kv, tag);   // legacy/BlockModelPipelineLegacyArms.inc
#endif
}

// ---------------------------------------------------------------------------
void factor_part(const LocalSystemSimplicialComplex& lssc, const std::string& tag) {
    const SimplicialComplex* K = lssc.get_base_K_complex();
    const uint32_t d = lssc.get_representation_dim();
    const uint32_t ell = lssc.get_prime();
    EquivariantCohomologyOnK coh(&lssc);
    const auto t0 = Clock::now();
    coh.compute_cohomology();
    std::cout << "  " << tag << ": compute_cohomology (current pipeline) " << secs_since(t0) << " s, dims";
    for (uint32_t x : coh.get_H_eq_dimensions()) std::cout << " " << x;
    std::cout << std::endl;

    const uint32_t q = 1;
    const uint32_t N_q = K->get_num_q_simplices(q);
    const std::vector<uint32_t> row_dims(N_q, d);
    const DenseBuf<uint32_t> Z = flatten_cochains(coh.get_Z_eq_bases_fm()[q], N_q, d);
    const DenseBuf<uint32_t> C = flatten_cochains(coh.get_C_eq_bases_fm()[q], N_q, d);
    const DenseBuf<uint32_t> Bq = flatten_cochains(coh.get_B_eq_bases_fm()[q], N_q, d);
    std::cout << "    q=1: dim C_eq " << C.cols << ", dim Z_eq " << Z.cols << ", dim B_eq " << Bq.cols << ", ambient " << C.rows << std::endl;
    auto pool = std::make_shared<BlockPool<uint32_t>>();

    // W^1 = C^1 / Z^1 : relations Z, generators C
    engine::blk::reset_elim_stats();
    const auto t1 = Clock::now();
    const engine::blk::FactorBasisResult<uint32_t> W = engine::blk::factor_basis<uint32_t>(pool, ell, row_dims, cview(Z), cview(C));
    const double s_w = secs_since(t1);
    const engine::blk::ElimStats st = engine::blk::elim_stats();
    std::cout << "    model: factor_basis W^1 = C^1/Z^1 -> dim " << W.pivots.size() << ", independent relations "
              << W.independent_relations << ", " << s_w << " s (panels " << st.panels << ", gemm calls " << st.gemms
              << ", scalar blocks gathered " << st.scalar_blocks_gathered << ")" << std::endl;
    check(W.independent_relations == Z.cols && W.pivots.size() + Z.cols == C.cols, tag + " W^1 dimensions");
#if OC_ENABLE_LEGACY
    legacy_factor_arm(row_dims, Z, C, ell, W, tag);   // legacy/BlockModelPipelineLegacyArms.inc
#endif
    // H^1 = Z^1 / B^1 : the stored projector P_1 is factor_basis_block(B, Z)
    {
        const auto t3 = Clock::now();
        const engine::blk::FactorBasisResult<uint32_t> H = engine::blk::factor_basis<uint32_t>(pool, ell, row_dims, cview(Bq), cview(Z));
        std::cout << "    model: factor_basis H^1 = Z^1/B^1 -> dim " << H.pivots.size() << ", " << secs_since(t3) << " s" << std::endl;
        check(H.pivots.size() == coh.get_H_eq_dimensions()[q], tag + " dim H^1");
        check(same_fm(coh.get_H_eq_projectors_from_C_eq_flat_fm()[q], cview(H.projector)), tag + " P_1 entrywise vs the pipeline's stored projector");
        const DenseBuf<uint32_t> Hb = flatten_cochains(coh.get_H_eq_bases_fm()[q], N_q, d);
        DenseBuf<uint32_t> scratch;
        const DenseView<const uint32_t> bv = engine::blk::multivector_view(H.basis, scratch);
        check(engine::same_entries(bv, cview(Hb)), tag + " H^1 basis entrywise vs the pipeline");
        // Theta-style multi-RHS solve: coordinates of Z^1 in C^1
        const auto t4 = Clock::now();
        const DenseBuf<uint32_t> coords = engine::blk::solve_multi_rhs<uint32_t>(pool, ell, row_dims, cview(C), cview(Z));
        std::cout << "    model: solve_multi_rhs (Z^1 in C^1, " << Z.cols << " RHS) " << secs_since(t4) << " s" << std::endl;
        DenseBuf<uint32_t> back(C.rows, Z.cols, Layout::ColMajor);
        // back = C * coords^T
        engine::kernels::gemm<uint32_t>(engine::ModCtx<uint32_t>(ell), cview(C), cview(coords).transposed_view(), view(back));
        check(back.same_entries(Z), tag + " solve_multi_rhs residual == 0");
    }
}

} // namespace

engine::BlockMat<uint32_t> BuildACqBlockMat(std::shared_ptr<engine::BlockPool<uint32_t>> pool,
                                            const LocalSystemSimplicialComplex& lssc, uint32_t q) {
    const SimplicialComplex* K = lssc.get_base_K_complex();
    const uint32_t d = lssc.get_representation_dim();
    const uint32_t ell = lssc.get_prime();
    const uint32_t N_q = K->get_num_q_simplices(q);
    const auto live = live_relations(lssc, q);
    const std::vector<uint32_t> row_dims(live.size(), d), col_dims(N_q, d);
    BlockBuilder<uint32_t> b(pool, row_dims, col_dims, ell);
    const uint32_t neg1 = ell - 1;
    uint32_t ri = 0;
    for (const auto* rp : live) {
        const auto& rel = *rp;
        // rho_n(g_r) interned once per distinct matrix (content-addressed)
        const BlockRef<uint32_t> M = b.intern_dense(cview(to_buf(rel.M)), Layout::RowMajor, true);
        if (rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq) {
            b.set(ri, rel.s_lhs_idx_in_Kq, b.compose(M, 1u, neg1));          // I - M
        } else {
            b.set(ri, rel.s_lhs_idx_in_Kq, BlockRef<uint32_t>::identity());  // I
            b.set(ri, rel.s_rhs_idx_in_Kq, b.compose(M, 0u, neg1));          // -M
        }
        ++ri;
    }
    return b.finalize();
}

uint32_t BlockModelPipelineValidation() {
    const auto t0 = Clock::now();
    g_fail = 0; g_checks = 0;
    std::cout << "=== BlockModelPipelineValidation (block model on the real A_C1 / W^1 / H^1 at n_sym = 128, entrywise vs legacy, timed) ===" << std::endl;
    (void)oc::legacy::RunLegacyArm("BlockModelPipelineValidation", "legacy");   // (S24 CL-L2) OFF: the legacy halves are compiled out
    const uint32_t n_sym = 128, ell = 134217757u;
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const auto rels = BuildRelations(K, n_sym, ell);
    const LocalSystemSimplicialComplex lssc(K, rels, n_sym + 1, ell);
    const bool sc = EquivariantCohomologyOptions::vacuous_relation_shortcircuit_enabled();
    for (int mode = 0; mode < 2; ++mode) {
        EquivariantCohomologyOptions::set_vacuous_relation_shortcircuit(mode == 0);
        const std::string tag = std::string("n_sym=128 shortcircuit ") + (mode == 0 ? "on" : "off");
        for (uint32_t q = 0; q <= 2; ++q) kernel_part(lssc, q, tag);
    }
    EquivariantCohomologyOptions::set_vacuous_relation_shortcircuit(sc);
    factor_part(lssc, "n_sym=128");
    std::cout << "BlockModelPipelineValidation: " << g_checks << " checks, " << g_fail << " failed, " << secs_since(t0) << " s" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
