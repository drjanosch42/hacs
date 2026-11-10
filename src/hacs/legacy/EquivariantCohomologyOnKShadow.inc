// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// legacy/EquivariantCohomologyOnKShadow.inc -- the comparison half of the -DENGINE_SHADOW_LEGACY=ON build mode for
// EquivariantCohomologyOnKT::compute_cohomology() (matrix-engine Phase 3, Stage B): after the block arm, the legacy
// arm (compute_cohomology_legacy) runs as well and every stored object -- the H^q dimensions, the five basis families
// C/Z/B/H and the projectors P -- is compared entrywise; the first difference throws std::logic_error.  Moved out of
// EquivariantCohomologyOnK.cpp by S24 CL-L1 (CLEANUP plan Sec. 2, C-12 step 1; owner ruling 1 of 2026-10-30):
// verbatim, the inline block having become the member shadow_compare_with_legacy().  Included at the end of
// EquivariantCohomologyOnK.cpp under `#ifdef ENGINE_SHADOW_LEGACY` (admitted by CMakeLists.txt only with
// OC_ENABLE_LEGACY=ON), before the explicit instantiations.  The legacy arm itself stays in the .cpp: it is also the
// ENGINE_BACKEND=legacy route, which CL-L2 moves.

#if !OC_ENABLE_LEGACY
#error "EquivariantCohomologyOnKShadow.inc is an OC_ENABLE_LEGACY=ON source (ENGINE_SHADOW_LEGACY needs OC_ENABLE_LEGACY=ON)"
#endif

namespace {

template<class Elt>
bool same_flat(const FlatMatrixT<Elt>& a, const FlatMatrixT<Elt>& b) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) return false;
    for (uint32_t i = 0; i < a.get_rows(); ++i)
        for (uint32_t j = 0; j < a.get_cols(); ++j)
            if (a.get_entry(i, j) != b.get_entry(i, j)) return false;
    return true;
}
template<class Elt>
void shadow_compare_lists(const std::vector<std::vector<FlatMatrixT<Elt>>>& e,
                          const std::vector<std::vector<FlatMatrixT<Elt>>>& l, const char* what) {
    if (e.size() != l.size())
        throw std::logic_error(std::string("ENGINE_SHADOW_LEGACY: compute_cohomology ") + what +
                               ": degree count differs (engine " + std::to_string(e.size()) +
                               ", legacy " + std::to_string(l.size()) + ")");
    for (size_t q = 0; q < e.size(); ++q) {
        if (e[q].size() != l[q].size())
            throw std::logic_error(std::string("ENGINE_SHADOW_LEGACY: compute_cohomology ") + what + " q=" +
                                   std::to_string(q) + ": basis size differs (engine " + std::to_string(e[q].size()) +
                                   ", legacy " + std::to_string(l[q].size()) + ")");
        for (size_t t = 0; t < e[q].size(); ++t)
            if (!same_flat<Elt>(e[q][t], l[q][t]))
                throw std::logic_error(std::string("ENGINE_SHADOW_LEGACY: compute_cohomology ") + what + " q=" +
                                       std::to_string(q) + " vector " + std::to_string(t) + " differs");
    }
}
} // namespace

template<class Elt>
void EquivariantCohomologyOnKT<Elt>::shadow_compare_with_legacy() {
    // Both arms, every stored object entrywise.
    FlatResult L;
    compute_cohomology_legacy(L);
    cohomology_computed = true;   // the accessors below require it
    ensure_fm_cache();
    if (fm_.dims != L.dims)
        throw std::logic_error("ENGINE_SHADOW_LEGACY: compute_cohomology: H^q dimensions differ");
    shadow_compare_lists<Elt>(fm_.C, L.C, "C_eq");
    shadow_compare_lists<Elt>(fm_.Z, L.Z, "Z_eq");
    shadow_compare_lists<Elt>(fm_.B, L.B, "B_eq");
    shadow_compare_lists<Elt>(fm_.H, L.H, "H_eq");
    if (fm_.P.size() != L.P.size())
        throw std::logic_error("ENGINE_SHADOW_LEGACY: compute_cohomology: projector count differs");
    for (size_t q = 0; q < fm_.P.size(); ++q)
        if (!same_flat<Elt>(fm_.P[q], L.P[q]))
            throw std::logic_error("ENGINE_SHADOW_LEGACY: compute_cohomology: projector P_" +
                                   std::to_string(q) + " differs");
}
