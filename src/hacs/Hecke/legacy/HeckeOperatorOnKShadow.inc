// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Hecke/legacy/HeckeOperatorOnKShadow.inc -- the comparison half of the -DENGINE_SHADOW_LEGACY=ON build mode for
// HeckeOperatorOnKT::compute() (matrix-engine Phase 3): after the block arm, the legacy dense arm (compute_legacy) runs
// as well and Theta_flat, Theta_eq, A^{(q)} and the certificate counters are compared entrywise; the first difference
// throws std::logic_error.  Moved out of HeckeOperatorOnK.cpp by S24 CL-L1 (CLEANUP plan Sec. 2, C-12 step 1; owner
// ruling 1 of 2026-10-30): verbatim, the inline block having become the member shadow_compare_with_legacy().  Included
// at the end of HeckeOperatorOnK.cpp under `#ifdef ENGINE_SHADOW_LEGACY` (admitted by CMakeLists.txt only with
// OC_ENABLE_LEGACY=ON), before the explicit instantiations.  The legacy arm itself (compute_legacy and the Legacy*
// types) stays in the .cpp: it is also the ENGINE_BACKEND=legacy route, which CL-L2 moves.  The one other shadow line,
// `needs_omega = true` in compute(), stays inline (the legacy arm borrows the Omega table).
//
// (S24 CL-L1 repair) Theta_eq and its two solve counters are compared only when the block arm ran the Theta8/Theta9
// solve (theta_eq_certified_).  Under ScopedThetaEqCertification(false) -- the HeckeExt cells -- the block arm
// publishes no Theta_eq (an empty vector, the documented contract of the switch) and theta_eq_solves = 0, while the
// legacy arm always solves; the shadow therefore threw "Theta_eq differs at q=0" on every such cell, in every
// shadow build since HeckeExt turned the solve off (slot-4 E2b, 47677b6b, 2026-09-21; reproduced on the unmodified
// 3ed47b59: HeckeExtSmallEllValidation).  Theta_flat, A^{(q)} and the relation / cocycle checks are still compared.

#if !OC_ENABLE_LEGACY
#error "HeckeOperatorOnKShadow.inc is an OC_ENABLE_LEGACY=ON source (ENGINE_SHADOW_LEGACY needs OC_ENABLE_LEGACY=ON)"
#endif

template<class Elt>
void HeckeOperatorOnKT<Elt>::shadow_compare_with_legacy() {
    // Both arms; every published object entrywise.
    LegacyResult L;
    compute_legacy(L);
    ensure_theta_flat();
    auto same_flat = [](const FlatMatrixT<Elt>& a, const FlatMatrixT<Elt>& b) {
        if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) return false;
        for (uint32_t i = 0; i < a.get_rows(); ++i)
            for (uint32_t j = 0; j < a.get_cols(); ++j)
                if (a.get_entry(i, j) != b.get_entry(i, j)) return false;
        return true;
    };
    const bool cmp_theta_eq = theta_eq_certified_;
    if (L.theta_flat.size() != Theta_flat_.size() || (cmp_theta_eq && L.theta_eq.size() != Theta_eq_.size()) ||
        L.A.size() != A_.size())
        throw std::logic_error("ENGINE_SHADOW_LEGACY: HeckeOperatorOnK: degree counts differ");
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        if (!same_flat(Theta_flat_[q], L.theta_flat[q]))
            throw std::logic_error("ENGINE_SHADOW_LEGACY: HeckeOperatorOnK: Theta_flat differs at q=" +
                                   std::to_string(q));
        if (cmp_theta_eq && !same_flat(Theta_eq_[q], L.theta_eq[q]))
            throw std::logic_error("ENGINE_SHADOW_LEGACY: HeckeOperatorOnK: Theta_eq differs at q=" +
                                   std::to_string(q));
        if (A_[q].rows != L.A[q].rows || A_[q].cols != L.A[q].cols)
            throw std::logic_error("ENGINE_SHADOW_LEGACY: HeckeOperatorOnK: A^{(q)} shape differs at q=" +
                                   std::to_string(q));
        for (uint32_t i = 0; i < A_[q].rows; ++i)
            for (uint32_t j = 0; j < A_[q].cols; ++j)
                if (A_[q].data[i][j] != L.A[q].data[i][j])
                    throw std::logic_error("ENGINE_SHADOW_LEGACY: HeckeOperatorOnK: A^{(q)} differs at q=" +
                                           std::to_string(q));
    }
    if (L.relation_checks != relation_checks_ || L.cocycle_checks != cocycle_checks_ ||
        (cmp_theta_eq && (L.theta_eq_solves != theta_eq_solves_ || L.residual_max != theta_eq_residual_max_)))
        throw std::logic_error("ENGINE_SHADOW_LEGACY: HeckeOperatorOnK: certificate counters differ");
}
