// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ModArithmetic/legacy/ShadowLegacy.inc -- the comparison half of the -DENGINE_SHADOW_LEGACY=ON build mode for
// FlatMatrix.cpp and BlockFlatMatrix.cpp (matrix-engine Phase 2): every retrofitted operation runs the engine route
// AND the pre-Phase-2 legacy route and throws std::logic_error on the first differing entry.  Moved out of the two
// .cpp files by S24 CL-L1 (CLEANUP plan Sec. 2, C-12 step 1; owner ruling 1 of 2026-10-30): the helper bodies are
// verbatim, and the inline compare blocks became the shadow_* calls below (same comparisons, same messages).
// Included inside each file's anonymous namespace under `#ifdef ENGINE_SHADOW_LEGACY`, which CMakeLists.txt admits
// only together with OC_ENABLE_LEGACY=ON (FATAL_ERROR otherwise).  The legacy route itself moved beside it in S24 CL-L2
// (legacy/FlatMatrixLegacy.inc: _gaussian_elimination_core and the former legacy_route lambdas, now the
// _legacy_* members; legacy/BlockFlatMatrix.cpp); the entrywise run_legacy loops stay in FlatMatrix.cpp, where they
// are also the p < 2 / empty-operand fallback.

#if !OC_ENABLE_LEGACY
#error "ShadowLegacy.inc is an OC_ENABLE_LEGACY=ON source (ENGINE_SHADOW_LEGACY needs OC_ENABLE_LEGACY=ON)"
#endif

template<class Elt>
void shadow_check(const char* what, const std::vector<Elt>& engine_out,
                  const std::vector<Elt>& legacy_out) {
    if (engine_out.size() != legacy_out.size() ||
        (!engine_out.empty() &&
         std::memcmp(engine_out.data(), legacy_out.data(),
                     engine_out.size() * sizeof(Elt)) != 0)) {
        throw std::logic_error(std::string("ENGINE_SHADOW_LEGACY: engine and legacy "
                                           "results differ in ") + what);
    }
}

[[noreturn]] inline void shadow_mismatch(const char* where, const std::string& what) {
    throw std::logic_error(std::string("ENGINE_SHADOW_LEGACY: legacy/engine mismatch in ") + where +
                           ": " + what);
}

// Entrywise comparison of two FlatMatrix values.
template<class Elt>
void shadow_check_equal(const FlatMatrixT<Elt>& engine_v, const FlatMatrixT<Elt>& legacy_v, const char* where) {
    if (engine_v.get_rows() != legacy_v.get_rows() || engine_v.get_cols() != legacy_v.get_cols())
        shadow_mismatch(where, "shape");
    for (uint32_t i = 0; i < engine_v.get_rows(); ++i)
        for (uint32_t j = 0; j < engine_v.get_cols(); ++j)
            if (engine_v.get_entry(i, j) != legacy_v.get_entry(i, j))
                shadow_mismatch(where, "entry (" + std::to_string(i) + "," + std::to_string(j) + ")");
}

// The entrywise operators (transpose, +, -, *, *scalar): park the engine result and hand the legacy loop a zeroed
// buffer of n entries (shadow_park), then compare the two and put the engine result back (shadow_restore).
template<class Elt>
std::vector<Elt> shadow_park(bool use_engine, std::vector<Elt>& buf, size_t n) {
    std::vector<Elt> shadow_engine;
    if (use_engine) {
        shadow_engine.swap(buf);
        buf.assign(n, Elt(0));
    }
    return shadow_engine;
}

template<class Elt>
void shadow_restore(bool use_engine, const char* what, std::vector<Elt>& shadow_engine, std::vector<Elt>& buf) {
    if (use_engine) {
        shadow_check<Elt>(what, shadow_engine, buf);
        buf.swap(shadow_engine);
    }
}

// compute_row_echelon_form: the legacy row echelon form (legacy_2d, legacy_pivots) against the cached engine one.
template<class Elt>
void shadow_check_ref(const std::vector<std::vector<Elt>>& legacy_2d, const std::vector<uint32_t>& legacy_pivots,
                      const std::vector<Elt>& engine_ref, const std::vector<uint32_t>& engine_pivots,
                      uint32_t rows, uint32_t cols) {
    if (legacy_pivots != engine_pivots) shadow_mismatch("compute_row_echelon_form", "pivot columns");
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j)
            if (legacy_2d[i][j] != engine_ref[static_cast<size_t>(i) * cols + j])
                shadow_mismatch("compute_row_echelon_form",
                                "entry (" + std::to_string(i) + "," + std::to_string(j) + ")");
}

// get_rref_and_transform_matrix / compute_elimination_factor_and_apply: both members of the returned pair.
template<class Elt>
void shadow_check_pair(const std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>>& engine_v,
                       const std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>>& legacy_v,
                       const char* where_first, const char* where_second) {
    shadow_check_equal<Elt>(engine_v.first, legacy_v.first, where_first);
    shadow_check_equal<Elt>(engine_v.second, legacy_v.second, where_second);
}
