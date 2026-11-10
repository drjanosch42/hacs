// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// src/hacs/legacy/DenseElimPipelineLegacyArms.inc -- the legacy halves of DenseElimPipelineValidation:
// BuildACq (A_Cq as a BlockFlatMatrix), the real-matrix comparisons against the legacy FlatMatrix paths and
// BlockFlatMatrix::flat_kernel under ScopedLegacy (and the retrofit flat_kernel), and Part 2, whole_pipeline (the
// whole compute_cohomology() under ScopedLegacy vs the engine).  Moved verbatim out of DenseElimPipelineValidation.cpp
// by S24 CL-L2 (CLEANUP plan Sec. 2.3, row L2; owner ruling 1 of 2026-10-30).  Included at file scope (it reopens the .cpp's anonymous namespace after its #include) under `#if OC_ENABLE_LEGACY`; an OFF build assembles A_Cq through BuildACqBlockMat, runs the engine /
// RefLinAlg / retrofit-FlatMatrix checks, and prints SKIP-ARM for the rest.
#if !OC_ENABLE_LEGACY
#error "DenseElimPipelineLegacyArms.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

#include "../../ModArithmetic/legacy/BlockFlatMatrix.h"

namespace {

bool is_identity(const FlatMatrix& M, uint32_t d) {
    if (M.get_rows() != d || M.get_cols() != d) return false;
    for (uint32_t r = 0; r < d; ++r)
        for (uint32_t c = 0; c < d; ++c)
            if (M.get_entry(r, c) != (r == c ? 1u : 0u)) return false;
    return true;
}

// A_Cq exactly as EquivariantCohomologyOnK::compute_C_eq_bases assembles it
// (I - M on the diagonal for a self-relation, I at lhs and -M at rhs
// otherwise; vacuous self-relations f(s) = I f(s) are dropped when the
// pipeline's item-7.4 short-circuit is enabled).  With the short-circuit ON
// (the default) A_C1 at n_sym = 128 has the 2 live q = 1 relations, 258 x
// 1419; with it OFF it is the pre-7.4 matrix of doc 03 Sec. 2 / 10.2, all 13
// q = 1 relations, 1677 x 1419 -- rank 258 either way.  Both are run.
BlockFlatMatrix BuildACq(const LocalSystemSimplicialComplex& lssc, uint32_t q) {
    const SimplicialComplex* K = lssc.get_base_K_complex();
    const uint32_t d = lssc.get_representation_dim();
    const uint32_t ell = lssc.get_prime();
    const uint32_t N_q = K->get_num_q_simplices(q);
    const bool shortcircuit = EquivariantCohomologyOptions::vacuous_relation_shortcircuit_enabled();
    std::vector<const LocalSystemSimplicialComplex::SimplexRelation*> live;
    for (const auto& rel : lssc.get_user_defined_simplex_relations()) {
        if (rel.q_dim != q) continue;
        if (shortcircuit && rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq && is_identity(rel.M, d)) continue;
        live.push_back(&rel);
    }
    const std::vector<uint32_t> row_dims(live.size(), d), col_dims(N_q, d);
    BlockFlatMatrix A(row_dims, col_dims, ell);
    uint32_t ri = 0;
    for (const auto* rp : live) {
        const auto& rel = *rp;
        std::vector<uint32_t> buf(static_cast<size_t>(d) * d);
        if (rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq) {
            for (uint32_t r = 0; r < d; ++r)
                for (uint32_t c = 0; c < d; ++c)
                    buf[static_cast<size_t>(r) * d + c] = ModArithmetic::mod_sub(r == c ? 1u : 0u, rel.M.get_entry(r, c), ell);
            A.set_block_entry(ri, rel.s_lhs_idx_in_Kq, FlatMatrix(d, d, ell, buf));
        } else {
            A.set_block_entry(ri, rel.s_lhs_idx_in_Kq, FlatMatrix(d, ell));
            for (uint32_t r = 0; r < d; ++r)
                for (uint32_t c = 0; c < d; ++c)
                    buf[static_cast<size_t>(r) * d + c] = ModArithmetic::mod_sub(0u, rel.M.get_entry(r, c), ell);
            A.set_block_entry(ri, rel.s_rhs_idx_in_Kq, FlatMatrix(d, d, ell, buf));
        }
        ++ri;
    }
    return A;
}

bool same_bfm_vectors(const std::vector<BlockFlatMatrix>& a, const std::vector<BlockFlatMatrix>& b) {
    if (a.size() != b.size()) return false;
    for (size_t t = 0; t < a.size(); ++t) {
        const FlatMatrix fa = a[t].to_flat_matrix(), fb = b[t].to_flat_matrix();
        if (fa.get_rows() != fb.get_rows() || fa.get_cols() != fb.get_cols()) return false;
        for (uint32_t i = 0; i < fa.get_rows(); ++i)
            for (uint32_t j = 0; j < fa.get_cols(); ++j)
                if (fa.get_entry(i, j) != fb.get_entry(i, j)) return false;
    }
    return true;
}

bool same_matrix_lists(const std::vector<FlatMatrix>& a, const std::vector<FlatMatrix>& b) {
    if (a.size() != b.size()) return false;
    for (size_t t = 0; t < a.size(); ++t) {
        if (a[t].get_rows() != b[t].get_rows() || a[t].get_cols() != b[t].get_cols()) return false;
        for (uint32_t i = 0; i < a[t].get_rows(); ++i)
            for (uint32_t j = 0; j < a[t].get_cols(); ++j)
                if (a[t].get_entry(i, j) != b[t].get_entry(i, j)) return false;
    }
    return true;
}

void legacy_flat_arm(const FlatMatrix& Afm, const engine::elim::RrefResult& res,
                     const DenseBuf<uint32_t>& R, const DenseBuf<uint32_t>& Kb, const DenseBuf<uint32_t>& Ib,
                     const std::string& tag) {
    FlatMatrix rref_leg;
    std::vector<uint32_t> piv_leg;
    std::vector<FlatMatrix> ker_leg, img_leg;
    {
        engine::elim::ScopedLegacy guard;
        const FlatMatrix M = Afm;              // fresh: no RREF cache
        rref_leg = M.get_row_echelon_form_as_flat_matrix();
        piv_leg = M.get_pivot_columns();
        ker_leg = M.kernel();
        img_leg = M.image();
    }
    check(piv_leg == res.pivots, tag + " pivots vs legacy FlatMatrix");
    check(same_buf_fm(R, rref_leg), tag + " RREF vs legacy FlatMatrix");
    check(same_rows_vs_columns(Kb, ker_leg), tag + " kernel vs legacy FlatMatrix::kernel");
    check(same_rows_vs_columns(Ib, img_leg), tag + " image vs legacy FlatMatrix::image");
}

// (the legacy flat_kernel was computed inside legacy_flat_arm's ScopedLegacy block before CL-L2; it is a pure
// function of Ablk, so computing it here, under its own guard, changes no check)
void legacy_block_kernel_arm(const BlockFlatMatrix& Ablk, const DenseBuf<uint32_t>& Kb, const std::string& tag) {
    std::vector<BlockFlatMatrix> bker_leg;
    {
        engine::elim::ScopedLegacy guard;
        const BlockFlatMatrix B = Ablk;        // fresh block RREF cache
        bker_leg = B.flat_kernel();
    }
    std::vector<FlatMatrix> bker_leg_flat, bker_eng_flat;
    for (const auto& v : bker_leg) bker_leg_flat.push_back(v.to_flat_matrix());
    check(same_rows_vs_columns(Kb, bker_leg_flat), tag + " kernel vs legacy BlockFlatMatrix::flat_kernel");
    const BlockFlatMatrix B = Ablk;
    const std::vector<BlockFlatMatrix> bker_eng = B.flat_kernel();
    for (const auto& v : bker_eng) bker_eng_flat.push_back(v.to_flat_matrix());
    check(same_rows_vs_columns(Kb, bker_eng_flat), tag + " kernel vs retrofit BlockFlatMatrix::flat_kernel");
    check(same_bfm_vectors(bker_leg, bker_eng), tag + " flat_kernel block vectors legacy == engine");
}

// Part 2: the whole compute_cohomology() run, legacy vs engine.
void whole_pipeline(uint32_t n_sym, uint32_t ell) {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const auto rels = BuildRelations(K, n_sym, ell);
    const LocalSystemSimplicialComplex lssc(K, rels, n_sym + 1, ell);
    const std::string tag = "compute_cohomology n_sym=" + str(n_sym);

    EquivariantCohomologyOnK leg(&lssc);
    {
        engine::elim::ScopedLegacy guard;
        leg.compute_cohomology();
    }
    EquivariantCohomologyOnK eng(&lssc);
    eng.compute_cohomology();

    check(leg.get_H_eq_dimensions() == eng.get_H_eq_dimensions(), tag + " dim H^q");
    const auto& lc = leg.get_C_eq_bases_fm(); const auto& ec = eng.get_C_eq_bases_fm();
    const auto& lz = leg.get_Z_eq_bases_fm(); const auto& ez = eng.get_Z_eq_bases_fm();
    const auto& lb = leg.get_B_eq_bases_fm(); const auto& eb = eng.get_B_eq_bases_fm();
    const auto& lh = leg.get_H_eq_bases_fm(); const auto& eh = eng.get_H_eq_bases_fm();
    check(lc.size() == ec.size() && lz.size() == ez.size() && lb.size() == eb.size() && lh.size() == eh.size(),
          tag + " degree counts");
    for (size_t q = 0; q < lc.size() && q < ec.size(); ++q) {
        check(same_matrix_lists(lc[q], ec[q]), tag + " C_eq basis q=" + str(q));
        if (q < lz.size() && q < ez.size()) check(same_matrix_lists(lz[q], ez[q]), tag + " Z_eq basis q=" + str(q));
        if (q < lb.size() && q < eb.size()) check(same_matrix_lists(lb[q], eb[q]), tag + " B_eq basis q=" + str(q));
        if (q < lh.size() && q < eh.size()) check(same_matrix_lists(lh[q], eh[q]), tag + " H_eq basis q=" + str(q));
    }
    check(same_matrix_lists(leg.get_H_eq_projectors_from_C_eq_flat_fm(), eng.get_H_eq_projectors_from_C_eq_flat_fm()),
          tag + " projectors P_q");
    std::cout << "  " << tag << ": dims";
    for (uint32_t d : eng.get_H_eq_dimensions()) std::cout << " " << d;
    std::cout << std::endl;
}

} // namespace
