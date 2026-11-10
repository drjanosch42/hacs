// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// BuildSwitches.h -- the configure-time build switches as seen by the code
// (hecke-extensions slot 19, NB-0; owner decisions 2026-10-03,
// docs/implementation/slot18/WP-19d_notebook_design_note.md Sec. 10).
//
// CMakeLists.txt defines OC_ENABLE_CUDA, OC_BUILD_DOCS and OC_BUILD_NOTEBOOK
// to 0 or 1 on cohomology_app (always, so test them with #if, never #ifdef).
// A validation whose body depends on a switched-off component keeps its
// registered name (the --list count does not depend on the switches) and
// returns BuildSwitchSkip(...) -- an explicit `[SKIP] <name>: <switch>=OFF`
// line and exit 0, never a silent pass:
//
//     int ConsoleServeValidation() {
//         if (!oc::kBuildNotebook) return oc::BuildSwitchSkip("ConsoleServeValidation", "OC_BUILD_NOTEBOOK");
//         ...
//     }
//
// cohomology_app only: the STL-only libraries (ConsoleCore, HeckeCore,
// NumberField) never include this header.
#pragma once

#include <iostream>

#if !defined(OC_ENABLE_CUDA) || !defined(OC_BUILD_DOCS) || !defined(OC_BUILD_NOTEBOOK)
#error "BuildSwitches.h: OC_ENABLE_CUDA / OC_BUILD_DOCS / OC_BUILD_NOTEBOOK must be defined by CMakeLists.txt (cohomology_app only)"
#endif

namespace oc {

constexpr bool kEnableCuda    = (OC_ENABLE_CUDA != 0);
constexpr bool kBuildDocs     = (OC_BUILD_DOCS != 0);
constexpr bool kBuildNotebook = (OC_BUILD_NOTEBOOK != 0);

// Prints `[SKIP] <name>: <switch_name>=OFF` and returns 0 (the validation's
// exit code). `switch_name` is the CMake option, e.g. "OC_BUILD_NOTEBOOK".
inline int BuildSwitchSkip(const char* name, const char* switch_name) {
    std::cout << "[SKIP] " << name << ": " << switch_name << "=OFF" << std::endl;
    return 0;
}

} // namespace oc
