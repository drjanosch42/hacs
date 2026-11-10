// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ModArithmetic/legacy/DenseElimLegacyArms.inc -- the legacy halves of DenseElimValidation / DenseElimSlowValidation:
// the engine::elim results against the pre-Phase-2 FlatMatrix paths under engine::elim::ScopedLegacy (RREF / pivots /
// kernel / image, get_rref_and_transform_matrix, inverse, change_of_basis_fm_multi, compute_elimination_factor_and_apply)
// and against factor_basis_block (legacy BlockFlatMatrix.cpp), each the verbatim ScopedLegacy block of run_case with its
// checks.  Moved out of DenseElimValidation.cpp by S24 CL-L2 (CLEANUP plan Sec. 2.3, row L2; owner ruling 1 of
// 2026-10-30).  The "retrofit" checks (the same FlatMatrix calls on the engine route) stay in run_case and run in both
// builds, except the retrofit factor_basis_block pair (BlockFlatMatrix.cpp is ON-only).  Included at file scope (it reopens the .cpp's anonymous namespace after its #include) under `#if OC_ENABLE_LEGACY`; an OFF build prints one `SKIP-ARM <name> arm=legacy` line.
#if !OC_ENABLE_LEGACY
#error "DenseElimLegacyArms.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

#include "BlockFlatMatrix.h"   // factor_basis_block

namespace {

template<class Elt>
void legacy_flat_paths_arm(uint32_t rows, uint32_t cols, Elt p, const std::vector<Elt>& data,
                           const engine::elim::RrefResult& res, const DenseBuf<Elt>& R, const DenseBuf<Elt>& K,
                           const DenseBuf<Elt>& I, const std::string& tag) {
    std::vector<uint32_t> piv_legacy;
    std::vector<FlatMatrixT<Elt>> ker_legacy, img_legacy;
    FlatMatrixT<Elt> rref_legacy;
    {
        engine::elim::ScopedLegacy guard;
        const FlatMatrixT<Elt> M(rows, cols, p, data);
        rref_legacy = M.get_row_echelon_form_as_flat_matrix();
        piv_legacy = M.get_pivot_columns();
        ker_legacy = M.kernel();
        img_legacy = M.image();
    }
    check(piv_legacy == res.pivots, tag + " pivots vs legacy FlatMatrix");
    check(same_buf_fm(R, rref_legacy), tag + " RREF vs legacy FlatMatrix");
    check(same_rows_vs_columns(K, ker_legacy), tag + " kernel vs legacy FlatMatrix::kernel");
    check(same_rows_vs_columns(I, img_legacy), tag + " image vs legacy FlatMatrix::image");
}

template<class Elt>
void legacy_transform_arm(uint32_t rows, uint32_t cols, Elt p, const std::vector<Elt>& data, const DenseBuf<Elt>& Rt,
                          const DenseBuf<Elt>& U, const std::string& tag) {
    std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> leg;
    {
        engine::elim::ScopedLegacy guard;
        const FlatMatrixT<Elt> M(rows, cols, p, data);
        leg = M.get_rref_and_transform_matrix();
    }
    check(same_buf_fm(Rt, leg.first) && same_buf_fm(U, leg.second), tag + " transform vs legacy get_rref_and_transform_matrix");
}

// The pre-CL-L2 check sequence: existence (legacy and retrofit together), then legacy, then retrofit entrywise.
template<class Elt>
void legacy_inverse_arm(uint32_t rows, uint32_t cols, Elt p, const std::vector<Elt>& data, bool ok, bool eng_ok,
                        const DenseBuf<Elt>& Ainv, const FlatMatrixT<Elt>& eng, const std::string& tag) {
    bool leg_ok = false;
    FlatMatrixT<Elt> leg;
    try {
        engine::elim::ScopedLegacy guard;
        const FlatMatrixT<Elt> M(rows, cols, p, data);
        leg = M.inverse(); leg_ok = true;
    } catch (const std::runtime_error&) {}
    check(leg_ok == ok && eng_ok == ok, tag + " inverse existence vs legacy/retrofit");
    if (ok && leg_ok && eng_ok) {
        check(same_buf_fm(Ainv, leg), tag + " inverse vs legacy FlatMatrix::inverse");
        check(same_buf_fm(Ainv, eng), tag + " inverse vs retrofit FlatMatrix::inverse");
    }
}

template<class Elt>
void legacy_solve_arm(const std::vector<FlatMatrixT<Elt>>& basis, const std::vector<FlatMatrixT<Elt>>& vecs,
                      const std::vector<std::vector<Elt>>& coords, const std::string& tag) {
    std::vector<std::vector<Elt>> leg;
    { engine::elim::ScopedLegacy guard; leg = change_of_basis_fm_multi<Elt>(basis, vecs); }
    check(leg == coords, tag + " solve_multi_rhs vs legacy change_of_basis_fm_multi");
}

template<class Elt>
void legacy_complement_arm(uint32_t a, Elt p, const DenseBuf<Elt>& Rl, const DenseBuf<Elt>& G,
                           const std::vector<uint32_t>& gens, const DenseBuf<Elt>& proj, const std::string& tag) {
    // legacy: single block row of dimension a
    const std::vector<uint32_t> block_row_dims{ a };
    const auto rels = columns_of<Elt>(Rl, p);
    const auto gvs = columns_of<Elt>(G, p);
    std::pair<FlatMatrixT<Elt>, std::vector<uint32_t>> leg, eng;
    { engine::elim::ScopedLegacy guard; leg = factor_basis_block<Elt>(block_row_dims, rels, gvs, nullptr, p); }
    eng = factor_basis_block<Elt>(block_row_dims, rels, gvs, nullptr, p);
    check(leg.second == gens, tag + " complement pivots vs legacy factor_basis_block");
    check(eng.second == gens, tag + " complement pivots vs retrofit factor_basis_block");
    check(same_buf_fm(proj, leg.first), tag + " projector vs legacy factor_basis_block");
    check(same_buf_fm(proj, eng.first), tag + " projector vs retrofit factor_basis_block");
}

template<class Elt>
void legacy_eliminate_arm(uint32_t rows, uint32_t cols, uint32_t kR, Elt p, const std::vector<Elt>& bdata,
                          const std::vector<Elt>& rref_flat, const DenseBuf<Elt>& Bb, const DenseBuf<Elt>& L,
                          const std::string& tag) {
    std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> leg;
    {
        engine::elim::ScopedLegacy guard;
        const FlatMatrixT<Elt> B(rows, cols, p, bdata), Rf(kR, cols, p, rref_flat);
        leg = B.compute_elimination_factor_and_apply(Rf);
    }
    check(same_buf_fm(Bb, leg.first) && same_buf_fm(L, leg.second), tag + " eliminate_against vs legacy compute_elimination_factor_and_apply");
}

} // namespace
