// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// src/hacs/legacy/SimplicialCohomologyLegacy.inc -- SimplicialComplex::coboundary_operator_block_elt,
// the block-diagonal D_1^q as a BlockFlatMatrixT, consumed only by the legacy arm of compute_cohomology()
// (legacy/EquivariantCohomologyOnKLegacy.inc) and the legacy-gated validations.  Moved verbatim out of
// SimplicialCohomology.cpp by S24 CL-L2 (CLEANUP plan Sec. 2.2, C-12 step 2; owner ruling 1 of 2026-10-30), with its
// two explicit instantiations.  Included by SimplicialCohomology.cpp under `#if OC_ENABLE_LEGACY`; declared in
// SimplicialCohomology.h under the same guard.  Also legacy_trivial_factor_basis: CochainComplex::compute_cohomology_bases'
// former factor_basis_block call (unit block rows), run under engine::elim::use_legacy() (ENGINE_BACKEND=legacy /
// ScopedLegacy); the default route is engine::elim::complement_basis.
#if !OC_ENABLE_LEGACY
#error "SimplicialCohomologyLegacy.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

#include "../../ModArithmetic/legacy/BlockFlatMatrix.h"

template<class Elt>
BlockFlatMatrixT<Elt> SimplicialComplex::coboundary_operator_block_elt(uint32_t q, Elt p, uint32_t d_V) const {
    FlatMatrixT<Elt> D1 = coboundary_operator_elt<Elt>(q, p);
    uint32_t N_source = D1.get_cols();   // N_q
    uint32_t N_target = D1.get_rows();   // N_{q+1}

    if (N_source == 0 || N_target == 0 || d_V == 0) {
        std::vector<uint32_t> row_dims(N_target, d_V);
        std::vector<uint32_t> col_dims(N_source, d_V);
        return BlockFlatMatrixT<Elt>(row_dims, col_dims, p);
    }

    std::vector<uint32_t> row_dims(N_target, d_V);
    std::vector<uint32_t> col_dims(N_source, d_V);
    BlockFlatMatrixT<Elt> D1_block(row_dims, col_dims, p);

    for (uint32_t i = 0; i < N_target; ++i) {
        for (uint32_t j = 0; j < N_source; ++j) {
            Elt scalar = D1.get_entry(i, j);
            if (scalar != 0) {
                D1_block.set_block_entry(i, j, FlatMatrixT<Elt>::CreateScalar(d_V, scalar, p));
            }
        }
    }
    return D1_block;
}

template BlockFlatMatrixT<uint32_t> SimplicialComplex::coboundary_operator_block_elt<uint32_t>(uint32_t, uint32_t, uint32_t) const;
template BlockFlatMatrixT<uint64_t> SimplicialComplex::coboundary_operator_block_elt<uint64_t>(uint32_t, uint64_t, uint32_t) const;

// (S24 CL-L2) CochainComplex::compute_cohomology_bases' factor_basis_block call, verbatim: one block row per ambient
// coordinate.  Must sit before compute_cohomology_bases (this .inc is included above it).
static std::pair<FlatMatrix, std::vector<uint32_t>> legacy_trivial_factor_basis(
        uint32_t dim_Z_q, const std::vector<FlatMatrix>& B_q_in_Z_q_basis,
        const std::vector<FlatMatrix>& Z_q_standard_basis, std::vector<FlatMatrix>* H_q_coeff_basis, uint32_t p) {
    const std::vector<uint32_t> unit_block_rows(dim_Z_q, 1u);
    return factor_basis_block<uint32_t>(unit_block_rows, B_q_in_Z_q_basis, Z_q_standard_basis, H_q_coeff_basis, p);
}
