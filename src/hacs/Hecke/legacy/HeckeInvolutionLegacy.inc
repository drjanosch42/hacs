// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// src/hacs/Hecke/legacy/HeckeInvolutionLegacy.inc -- the two FlatMatrix pivot routes of
// HeckeInvolution.cpp that engine::elim::use_legacy() (ENGINE_BACKEND=legacy / ScopedLegacy) selects instead of
// engine::elim::column_rank_profile: the (iii-c) carry check of check_obl_iii (PERF1 audit R2, counted by
// iota_carry_legacy_calls(), which HeckeInvolutionValidation's IOTA_CARRY_ROUTE arm reads) and pivots_of in
// build_split.  Moved out of HeckeInvolution.cpp by S24 CL-L2 (CLEANUP plan Sec. 2.2, B8 / A8; owner ruling 1 of
// 2026-10-30): the two branch bodies verbatim as functions, the call sites keep a guarded one-line call.  Included by
// HeckeInvolution.cpp inside namespace hecke under `#if OC_ENABLE_LEGACY`; iota_carry_legacy_calls() is declared in
// HeckeInvolution.h under the same guard.
#if !OC_ENABLE_LEGACY
#error "HeckeInvolutionLegacy.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

namespace {
std::atomic<uint64_t> g_iota_carry_legacy_calls{0};

// (PERF1 audit R2) ENGINE_BACKEND=legacy / ENGINE_SHADOW_LEGACY: the previous
// FlatMatrix route, so the rollback arm (and the shadow twin inside FlatMatrix)
// keeps covering this check.
std::vector<uint32_t> iota_carry_pivots_legacy(uint32_t flat, uint32_t w, uint32_t ell,
                                               const std::vector<uint32_t>& aug) {
    const FlatMatrix legacy_aug(flat, w, ell, aug);
    std::vector<uint32_t> piv = legacy_aug.get_pivot_columns();
    g_iota_carry_legacy_calls.fetch_add(1u, std::memory_order_relaxed);
    return piv;
}

// build_split's pivots_of: the FlatMatrix route (buf is released once the matrix holds its copy).
std::vector<uint32_t> iota_split_pivots_legacy(uint32_t flat, uint32_t k, uint32_t ell, std::vector<uint32_t>& buf) {
    const FlatMatrix M(flat, k, ell, buf);
    std::vector<uint32_t>().swap(buf);
    return M.get_pivot_columns();
}
} // namespace

uint64_t iota_carry_legacy_calls() { return g_iota_carry_legacy_calls.load(std::memory_order_relaxed); }
