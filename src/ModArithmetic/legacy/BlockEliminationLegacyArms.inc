// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ModArithmetic/legacy/BlockEliminationLegacyArms.inc -- the legacy halves of BlockEliminationValidation /
// BlockEliminationSlowValidation: the block results against the legacy BlockFlatMatrix (flat_kernel / flat_image /
// true RREF, under ScopedLegacy and on the retrofit), against factor_basis_block and against change_of_basis_fm under
// ScopedLegacy.  Moved verbatim out of BlockEliminationValidation.cpp by S24 CL-L2 (CLEANUP plan Sec. 2.3, row L2;
// owner ruling 1 of 2026-10-30): the three `if (with_legacy)` bodies became legacy_bfm_arm / legacy_factor_basis_arm
// / legacy_solve_arm (their locals passed in), with the two conversions to_bfm / same_bfm_columns.  Included at file scope (it reopens the .cpp's anonymous namespace after its #include) under `#if OC_ENABLE_LEGACY`; an OFF build prints one
// `SKIP-ARM BlockEliminationValidation arm=legacy` line and runs every engine check unchanged (the generator cap
// `with_legacy && a > 500` still applies, so the engine checks are the same set in both builds).
#if !OC_ENABLE_LEGACY
#error "BlockEliminationLegacyArms.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

#include "BlockFlatMatrix.h"

namespace {

// ---------------------------------------------------------------------------
// Legacy conversions.
// ---------------------------------------------------------------------------
template<class Elt>
BlockFlatMatrixT<Elt> to_bfm(const BlockMat<Elt>& A) {
    BlockFlatMatrixT<Elt> B(A.row_dims(), A.col_dims(), A.p());
    for (uint32_t br = 0; br < A.nbr(); ++br) {
        const uint32_t dr = A.row_dim(br);
        for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k) {
            const uint32_t bc = A.col_at(k), dc = A.col_dim(bc);
            if (dr == 0 || dc == 0) continue;
            DenseBuf<Elt> blk(dr, dc);
            A.materialise(A.ref_at(k), dr, dc, view(blk), false);
            std::vector<Elt> data(blk.data(), blk.data() + static_cast<size_t>(dr) * dc);
            B.set_block_entry(br, bc, FlatMatrixT<Elt>(dr, dc, A.p(), data));
        }
    }
    return B;
}

// vectors (legacy BFM column vectors) == columns of D
template<class Elt>
bool same_bfm_columns(const std::vector<BlockFlatMatrixT<Elt>>& vecs, DenseView<const Elt> D) {
    if (vecs.size() != D.cols()) return false;
    for (uint32_t t = 0; t < D.cols(); ++t) {
        const FlatMatrixT<Elt> f = vecs[t].to_flat_matrix();
        if (f.get_rows() != D.rows() || f.get_cols() != 1) return false;
        for (uint32_t i = 0; i < D.rows(); ++i) if (f.get_entry(i, 0) != D.at(i, t)) return false;
    }
    return true;
}

template<class Elt>
void legacy_bfm_arm(const BlockMat<Elt>& A, const engine::blk::BlockRrefResult<Elt>& res, const DenseBuf<Elt>& Kd,
                    const DenseBuf<Elt>& Imd, const std::string& tag) {
    const BlockFlatMatrixT<Elt> B = to_bfm(A);
    {
        engine::elim::ScopedLegacy guard;
        const BlockFlatMatrixT<Elt> B1 = B;
        check(same_bfm_columns(B1.flat_kernel(), cview(Kd)), tag + " kernel entrywise vs LEGACY flat_kernel");
        const BlockFlatMatrixT<Elt> B2 = B;
        check(same_bfm_columns(B2.flat_image(), cview(Imd)), tag + " image entrywise vs LEGACY flat_image");
        const BlockFlatMatrixT<Elt> B3 = B;
        auto [rref_fm, piv] = B3.get_true_row_echelon_form_as_flat_matrix();
        bool okp = piv.size() == res.rank;
        for (size_t i = 0; okp && i < piv.size(); ++i) okp = piv[i].first == i && piv[i].second == res.pivots[i];
        check(okp, tag + " pivots (row, col) vs LEGACY true RREF");
        check(rref_fm.get_rows() >= res.rank && same_fm(rref_fm.extract_sub_matrix_rows(0, res.rank), cview(res.R)),
              tag + " RREF rows entrywise vs LEGACY true RREF");
    }
    {
        const BlockFlatMatrixT<Elt> B1 = B;
        check(same_bfm_columns(B1.flat_kernel(), cview(Kd)), tag + " kernel entrywise vs retrofit flat_kernel");
    }
}

template<class Elt>
void legacy_factor_basis_arm(const std::vector<uint32_t>& row_dims, const DenseBuf<Elt>& Rl, const DenseBuf<Elt>& G,
                             Elt p, const engine::blk::FactorBasisResult<Elt>& fb, const DenseBuf<Elt>& basis,
                             const std::string& tag) {
    engine::elim::ScopedLegacy guard;
    std::vector<FlatMatrixT<Elt>> out;
    auto [proj_l, piv_l] = factor_basis_block<Elt>(row_dims, columns_as_fm<Elt>(cview(Rl), p), columns_as_fm<Elt>(cview(G), p), &out, p);
    check(piv_l == fb.pivots, tag + " factor_basis pivots vs LEGACY factor_basis_block");
    check(same_fm(proj_l, cview(fb.projector)), tag + " factor_basis projector entrywise vs LEGACY");
    check(same_fm_columns(out, cview(basis)), tag + " factor_basis vectors vs LEGACY");
}

template<class Elt>
void legacy_solve_arm(const DenseBuf<Elt>& Bm, const DenseBuf<Elt>& Y, Elt p, const DenseBuf<Elt>& coords,
                      const std::string& tag) {
    const uint32_t ny = Y.cols, nb = Bm.cols;
    bool ok;
    engine::elim::ScopedLegacy guard;
    const std::vector<std::vector<Elt>> cl = change_of_basis_fm<Elt>(columns_as_fm<Elt>(cview(Bm), p), columns_as_fm<Elt>(cview(Y), p));
    ok = cl.size() == ny;
    for (uint32_t t = 0; ok && t < ny; ++t) for (uint32_t j = 0; j < nb; ++j) if (cl[t][j] != coords.at(t, j)) { ok = false; break; }
    check(ok, tag + " solve_multi_rhs vs LEGACY change_of_basis_fm");
}

} // namespace
