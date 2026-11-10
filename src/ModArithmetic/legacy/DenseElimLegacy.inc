// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ModArithmetic/legacy/DenseElimLegacy.inc -- the thread-local half of the ENGINE_BACKEND=legacy backend switch
// (matrix-engine Phase 2): engine::elim::ScopedLegacy and legacy_forced().  Moved verbatim out of DenseElim.cpp by
// S24 CL-L2 (CLEANUP plan Sec. 2.2, C-12 step 2; owner ruling 1 of 2026-10-30).  Included by DenseElim.cpp inside
// namespace engine::elim under `#if OC_ENABLE_LEGACY`; the declarations sit in DenseElim.h under the same guard, and
// the pool hand-over (a task thread under a forced-legacy caller forces legacy too) in EngineRuntime.cpp.
#if !OC_ENABLE_LEGACY
#error "DenseElimLegacy.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

namespace {
thread_local int g_legacy_depth = 0;
} // namespace

bool legacy_forced() { return g_legacy_depth > 0; }

ScopedLegacy::ScopedLegacy()  { ++g_legacy_depth; }
ScopedLegacy::~ScopedLegacy() { --g_legacy_depth; }
