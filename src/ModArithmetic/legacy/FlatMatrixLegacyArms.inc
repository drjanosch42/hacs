// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ModArithmetic/legacy/FlatMatrixLegacyArms.inc -- FlatMatrixValidation's Q11 third class: BlockFlatMatrixT::flat_image()
// of alpha*I as a single d x d block (legacy BlockFlatMatrix.cpp).  Moved out of FlatMatrixValidation.cpp by S24 CL-L2
// (CLEANUP plan Sec. 2.3, row L2; owner ruling 1 of 2026-10-30): the construction verbatim, returned as the image
// columns' entries (img[j][i] = entry i of column j).  Included at namespace scope under `#if OC_ENABLE_LEGACY`; an OFF
// build prints `SKIP-ARM FlatMatrixValidation arm=Q11-BlockFlatMatrix` and compares Matrix with FlatMatrix only.
#if !OC_ENABLE_LEGACY
#error "FlatMatrixLegacyArms.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

#include "BlockFlatMatrix.h"

static std::vector<std::vector<uint32_t>> LegacyQ11BlockImage(const FlatMatrix& aI_scaled, uint32_t d, uint32_t p) {
    std::vector<uint32_t> dims = { d };
    BlockFlatMatrix aI_bfm(dims, dims, p);
    aI_bfm.set_block_entry(0, 0, aI_scaled);
    std::vector<BlockFlatMatrix> img_bfm = aI_bfm.flat_image();
    std::vector<std::vector<uint32_t>> out(img_bfm.size(), std::vector<uint32_t>(d, 0u));
    for (size_t j = 0; j < img_bfm.size(); ++j)
        for (uint32_t i = 0; i < d; ++i) out[j][i] = img_bfm[j].get_entry(i, 0);
    return out;
}
