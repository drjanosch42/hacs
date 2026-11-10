// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// src/hacs/legacy/BlockModelPipelineLegacyArms.inc -- the legacy halves of
// BlockModelPipelineValidation: the block kernel of the real A_Cq against BlockFlatMatrix::flat_kernel (under
// ScopedLegacy, and on the retrofit), and W^1 = C^1/Z^1 against factor_basis_block under ScopedLegacy.  Moved verbatim
// out of BlockModelPipelineValidation.cpp by S24 CL-L2 (CLEANUP plan Sec. 2.3, row L2; owner ruling 1 of 2026-10-30):
// BuildACqLegacy and same_bfm_columns, and the two blocks as legacy_kernel_arm / legacy_factor_arm.  Included at file scope (it reopens the .cpp's anonymous namespace after its #include) under `#if OC_ENABLE_LEGACY`; an OFF build prints
// `SKIP-ARM BlockModelPipelineValidation arm=legacy`.
#if !OC_ENABLE_LEGACY
#error "BlockModelPipelineLegacyArms.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

#include "../../ModArithmetic/legacy/BlockFlatMatrix.h"

namespace {

std::vector<FlatMatrix> columns_as_fm(DenseView<const uint32_t> M, uint32_t p) {
    std::vector<FlatMatrix> out;
    for (uint32_t j = 0; j < M.cols(); ++j) {
        std::vector<uint32_t> col(M.rows());
        for (uint32_t i = 0; i < M.rows(); ++i) col[i] = M.at(i, j);
        out.emplace_back(M.rows(), 1, p, col);
    }
    return out;
}

// Legacy assembly (identical to EquivariantCohomologyOnK / DenseElimPipelineValidation).
BlockFlatMatrix BuildACqLegacy(const LocalSystemSimplicialComplex& lssc, uint32_t q) {
    const SimplicialComplex* K = lssc.get_base_K_complex();
    const uint32_t d = lssc.get_representation_dim();
    const uint32_t ell = lssc.get_prime();
    const uint32_t N_q = K->get_num_q_simplices(q);
    const auto live = live_relations(lssc, q);
    const std::vector<uint32_t> row_dims(live.size(), d), col_dims(N_q, d);
    BlockFlatMatrix A(row_dims, col_dims, ell);
    uint32_t ri = 0;
    for (const auto* rp : live) {
        const auto& rel = *rp;
        std::vector<uint32_t> buf(static_cast<size_t>(d) * d);
        if (rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq) {
            for (uint32_t r = 0; r < d; ++r)
                for (uint32_t c = 0; c < d; ++c)
                    buf[static_cast<size_t>(r) * d + c] = ModArithmetic::mod_sub(r == c ? 1u : 0u, rel_entry(rel.M, r, c), ell);
            A.set_block_entry(ri, rel.s_lhs_idx_in_Kq, FlatMatrix(d, d, ell, buf));
        } else {
            A.set_block_entry(ri, rel.s_lhs_idx_in_Kq, FlatMatrix(d, ell));
            for (uint32_t r = 0; r < d; ++r)
                for (uint32_t c = 0; c < d; ++c)
                    buf[static_cast<size_t>(r) * d + c] = ModArithmetic::mod_sub(0u, rel_entry(rel.M, r, c), ell);
            A.set_block_entry(ri, rel.s_rhs_idx_in_Kq, FlatMatrix(d, d, ell, buf));
        }
        ++ri;
    }
    return A;
}

bool same_bfm_columns(const std::vector<BlockFlatMatrix>& vecs, DenseView<const uint32_t> D) {
    if (vecs.size() != D.cols()) return false;
    for (uint32_t t = 0; t < D.cols(); ++t) {
        const FlatMatrix f = vecs[t].to_flat_matrix();
        if (f.get_rows() != D.rows() || f.get_cols() != 1) return false;
        for (uint32_t i = 0; i < D.rows(); ++i) if (f.get_entry(i, 0) != D.at(i, t)) return false;
    }
    return true;
}

void legacy_kernel_arm(const LocalSystemSimplicialComplex& lssc, uint32_t q, const DenseView<const uint32_t> Kv,
                       const std::string& tag) {
    const BlockFlatMatrix B = BuildACqLegacy(lssc, q);
    {
        engine::elim::ScopedLegacy guard;
        const BlockFlatMatrix B1 = B;
        const auto t4 = Clock::now();
        const std::vector<BlockFlatMatrix> kl = B1.flat_kernel();
        std::cout << "    LEGACY (pre-Phase-2) BlockFlatMatrix::flat_kernel " << secs_since(t4) << " s" << std::endl;
        check(same_bfm_columns(kl, Kv), tag + " kernel entrywise vs LEGACY flat_kernel");
    }
    {
        const BlockFlatMatrix B2 = B;
        const auto t5 = Clock::now();
        const std::vector<BlockFlatMatrix> kr = B2.flat_kernel();
        std::cout << "    retrofit (Phase 2) BlockFlatMatrix::flat_kernel " << secs_since(t5) << " s" << std::endl;
        check(same_bfm_columns(kr, Kv), tag + " kernel entrywise vs retrofit flat_kernel");
    }
}

void legacy_factor_arm(const std::vector<uint32_t>& row_dims, const DenseBuf<uint32_t>& Z, const DenseBuf<uint32_t>& C,
                       uint32_t ell, const engine::blk::FactorBasisResult<uint32_t>& W, const std::string& tag) {
    engine::elim::ScopedLegacy guard;
    std::vector<FlatMatrix> out;
    const auto t2 = Clock::now();
    auto [proj_l, piv_l] = factor_basis_block<uint32_t>(row_dims, columns_as_fm(cview(Z), ell), columns_as_fm(cview(C), ell), &out, ell);
    std::cout << "    LEGACY factor_basis_block W^1 " << secs_since(t2) << " s" << std::endl;
    check(piv_l == W.pivots, tag + " W^1 pivots vs LEGACY");
    check(same_fm(proj_l, cview(W.projector)), tag + " W^1 projector entrywise vs LEGACY");
    DenseBuf<uint32_t> scratch;
    const DenseView<const uint32_t> bv = engine::blk::multivector_view(W.basis, scratch);
    bool ok = out.size() == bv.cols();
    for (uint32_t t = 0; ok && t < bv.cols(); ++t)
        for (uint32_t i = 0; i < bv.rows(); ++i) if (out[t].get_entry(i, 0) != bv.at(i, t)) { ok = false; break; }
    check(ok, tag + " W^1 basis vectors vs LEGACY");
}

} // namespace
