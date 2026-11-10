// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ModArithmetic/legacy/SubstrateDifferentialLegacyArms.inc -- the THIRD substrate of SubstrateDifferentialValidation:
// BlockFlatMatrixT (the deprecated block container, legacy BlockFlatMatrix.cpp) beside MatrixT and FlatMatrixT.  Moved
// out of SubstrateDifferentialValidation.cpp by S24 CL-L2 (CLEANUP plan Sec. 2.3, row L2; owner ruling 1 of
// 2026-10-30): FromBlock, ColsToRows_B and MakeBlock verbatim; the block halves of CompareOneShape (now one call,
// legacy_block_shape, whose results feed the unchanged three-way checks), of CacheCoherence ((d), (e)) and of
// AlgebraicIdentities (the product and inverse cross-checks), verbatim as functions.  Included at file scope (it reopens the .cpp's anonymous namespace after its #include) under `#if OC_ENABLE_LEGACY`; an OFF build prints
// `SKIP-ARM SubstrateDifferentialValidation arm=BlockFlatMatrix` and runs the Matrix-vs-FlatMatrix checks alone.
#if !OC_ENABLE_LEGACY
#error "SubstrateDifferentialLegacyArms.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

#include "BlockFlatMatrix.h"

namespace {

template <class Elt>
Dense<Elt> FromBlock(const BlockFlatMatrixT<Elt>& b) {
    Dense<Elt> d(b.get_total_rows(), std::vector<Elt>(b.get_total_cols(), Elt(0)));
    for (uint32_t i = 0; i < b.get_total_rows(); ++i)
        for (uint32_t j = 0; j < b.get_total_cols(); ++j) d[i][j] = b.get_entry(i, j);
    return d;
}

template <class Elt>
std::vector<std::vector<Elt>> ColsToRows_B(const std::vector<BlockFlatMatrixT<Elt>>& basis) {
    std::vector<std::vector<Elt>> out;
    for (const auto& v : basis) {
        std::vector<Elt> row(v.get_total_rows());
        for (uint32_t i = 0; i < v.get_total_rows(); ++i) row[i] = v.get_entry(i, 0);
        out.push_back(std::move(row));
    }
    return out;
}

// Slices the dense matrix along the given block partition. Zero-width /
// zero-height blocks are supported (and deliberately exercised).
template <class Elt>
BlockFlatMatrixT<Elt> MakeBlock(const Dense<Elt>& d,
                                const std::vector<uint32_t>& row_dims,
                                const std::vector<uint32_t>& col_dims, Elt p) {
    std::vector<std::vector<FlatMatrixT<Elt>>> blocks(row_dims.size());
    uint32_t r0 = 0;
    for (size_t br = 0; br < row_dims.size(); ++br) {
        uint32_t c0 = 0;
        for (size_t bc = 0; bc < col_dims.size(); ++bc) {
            const uint32_t nr = row_dims[br], nc = col_dims[bc];
            std::vector<Elt> flat;
            flat.reserve(static_cast<size_t>(nr) * nc);
            for (uint32_t i = 0; i < nr; ++i)
                for (uint32_t j = 0; j < nc; ++j) flat.push_back(d[r0 + i][c0 + j]);
            if (flat.empty()) blocks[br].push_back(FlatMatrixT<Elt>(nr, nc, p));
            else blocks[br].push_back(FlatMatrixT<Elt>(nr, nc, p, flat));
            c0 += nc;
        }
        r0 += row_dims[br];
    }
    return BlockFlatMatrixT<Elt>(row_dims, col_dims, blocks, p);
}

template <class Elt>
BlockShape<Elt> legacy_block_shape(const Dense<Elt>& d, const std::vector<uint32_t>& row_dims,
                                   const std::vector<uint32_t>& col_dims, Elt p) {
    BlockFlatMatrixT<Elt> B = MakeBlock(d, row_dims, col_dims, p);
    BlockShape<Elt> out;
    out.have = true;
    out.construct = FromBlock(B);
    out.rank = B.flat_rank();
    out.rref = FromFlat(B.get_true_row_echelon_form_as_flat_matrix().first);
    out.ker = ColsToRows_B(B.flat_kernel());
    out.img = ColsToRows_B(B.flat_image());
    return out;
}

template <class Elt>
void legacy_block_cache_coherence(const Dense<Elt>& d, Elt p, const std::string& label) {
    // (d) BlockFlatMatrix: flat_rank; set_block_entry; flat_rank
    {
        const std::vector<uint32_t> rd = {2, 4}, cd = {3, 3};
        BlockFlatMatrixT<Elt> B = MakeBlock(d, rd, cd, p);
        (void)B.flat_rank();
        FlatMatrixT<Elt> newblk = FlatMatrixT<Elt>::CreateScalar(0, Elt(0), p); // placeholder
        // Replace block (1,0) (4 x 3) by the identity-like scalar block.
        newblk = FlatMatrixT<Elt>(4, 3, p, std::vector<Elt>(12, Elt(0)));
        newblk.set_entry(0, 0, Elt(1));
        B.set_block_entry(1, 0, newblk);
        const uint32_t after = B.flat_rank();

        Dense<Elt> d2 = d;
        for (uint32_t i = 0; i < 4; ++i)
            for (uint32_t j = 0; j < 3; ++j) d2[2 + i][j] = (i == 0 && j == 0) ? Elt(1) : Elt(0);
        BlockFlatMatrixT<Elt> Fresh = MakeBlock(d2, rd, cd, p);
        Check(after == Fresh.flat_rank(), Tag(label, "cache/BlockFlatMatrix rank;set_block;rank"));
    }

    // (e) BlockFlatMatrix: flat_rank; is_zero; flat_rank
    {
        const std::vector<uint32_t> rd = {2, 4}, cd = {3, 3};
        BlockFlatMatrixT<Elt> B = MakeBlock(d, rd, cd, p);
        const uint32_t r1 = B.flat_rank();
        const bool z = B.is_zero();
        const uint32_t r2 = B.flat_rank();
        Check(r1 == r2 && !z, Tag(label, "cache/BlockFlatMatrix rank;is_zero;rank"));
    }
}

template <class Elt>
void legacy_block_product(const Dense<Elt>& da, const Dense<Elt>& db, const Dense<Elt>& ab_matrix, Elt p,
                          const std::string& label) {
    const std::vector<uint32_t> ard = {2, 3}, acd = {1, 3}, bcd = {2, 4};
    BlockFlatMatrixT<Elt> Ab = MakeBlock(da, ard, acd, p);
    BlockFlatMatrixT<Elt> Bb = MakeBlock(db, acd, bcd, p);
    Check(ExactlyEqual(ab_matrix, FromBlock(Ab * Bb)),
          Tag(label, "product/Matrix-vs-BlockFlatMatrix"));
}

template <class Elt>
void legacy_block_inverse(const Dense<Elt>& dm, const Dense<Elt>& ainv_matrix, Elt p, const std::string& label) {
    const std::vector<uint32_t> bd = {2, 3};
    BlockFlatMatrixT<Elt> Ab = MakeBlock(dm, bd, bd, p);
    Check(ExactlyEqual(ainv_matrix, FromFlat(Ab.flat_inverse())),
          Tag(label, "inverse/Matrix-vs-BlockFlatMatrix"));
}

} // namespace
