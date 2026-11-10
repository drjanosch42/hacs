// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineConfig.h -- runtime backend switch for the matrix engine (Phase 2).
#pragma once
#include <cstdlib>
#include <cstring>
#include "LegacyConfig.h"
namespace engine {
// ENGINE_BACKEND=legacy forces every retrofitted path back to the pre-Phase-2 code.
// (S24 CL-L0, CLEANUP plan Sec. 2.4) Only in an OC_ENABLE_LEGACY=ON build: in an OFF build
// the request is REFUSED at the first read (stderr `REFUSED ENGINE_BACKEND=legacy: compiled
// out (OC_ENABLE_LEGACY=OFF)`, exit 1) -- main() reads it once at start-up -- never ignored.
// Every other value (unset, "engine", ...) reads false exactly as before.
inline bool backend_is_legacy() {
    static const bool legacy = [] {
        const char* e = std::getenv("ENGINE_BACKEND");
        const bool want = e != nullptr && std::strcmp(e, "legacy") == 0;
        if (want && !oc::legacy::kLegacyBuild) oc::legacy::refuse_compiled_out("ENGINE_BACKEND=legacy");
        return want;
    }();
    return legacy;
}
}  // namespace engine
