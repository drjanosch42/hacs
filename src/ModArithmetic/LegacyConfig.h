// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef OC_LEGACY_CONFIG_H
#define OC_LEGACY_CONFIG_H
// =====================================================================
// LegacyConfig.h -- the OC_ENABLE_LEGACY compile flag as the code sees it
// (S24 CL-L0; docs/implementation/slot24/CLEANUP_plan_1030.md Sec. 2.1, 2.4;
// owner ruling 1 of 2026-10-30)
// =====================================================================
//
// Header-only, STL-only (usable from every target, HeckeCore included).
//
// CMakeLists.txt defines OC_ENABLE_LEGACY to 0 or 1 -- PUBLIC on ModArithmetic,
// HeckeCore, MatrixEngine and EngineCuda, PRIVATE on cohomology_app -- because the
// macro will guard class members (ODR: every TU must agree).  So:
//   * test it with `#if OC_ENABLE_LEGACY` or `if constexpr (oc::legacy::kLegacyBuild)`,
//     NEVER with `#ifdef` -- a TU that lacks the definition fails to compile here
//     instead of silently selecting OFF;
//   * retired code lives in `legacy/` directories (Sec. 2.2) and is compiled only ON.
//
// A request for a compiled-out route in an OFF build is REFUSED, never ignored
// (Sec. 2.4, the vacuous-arm lesson): refuse_compiled_out("<request>") prints
//     REFUSED <request>: compiled out (OC_ENABLE_LEGACY=OFF)
// on stderr and exits 1.  The console (exit 5, hecke::NotImplementedInThisBuild) and
// the bench (usage error, exit 1) build their own refusals around compiled_out_text().
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#ifndef OC_ENABLE_LEGACY
#error "LegacyConfig.h: OC_ENABLE_LEGACY must be defined (0 or 1) by CMakeLists.txt on every target that includes it"
#endif

namespace oc {
namespace legacy {

inline constexpr bool kLegacyBuild = (OC_ENABLE_LEGACY != 0);

// "<request>: compiled out (OC_ENABLE_LEGACY=OFF)" -- the one refusal text (no prefix,
// no newline), e.g. request = "ENGINE_BACKEND=legacy" or "HECKE_COSET_STREAM=0".
inline std::string compiled_out_text(const std::string& request) {
    return request + ": compiled out (OC_ENABLE_LEGACY=OFF)";
}

// Prints "REFUSED " + compiled_out_text(request) + "\n" on stderr and terminates the
// process with exit code 1.  stdout / stderr are flushed first; std::_Exit (no static
// destructors) because a refusal may fire inside a function-local static's initialiser
// or on a pool worker thread.  Never call it in an ON build (it says OFF).
[[noreturn]] inline void refuse_compiled_out(const std::string& request) {
    std::fflush(stdout);
    std::fputs(("REFUSED " + compiled_out_text(request) + "\n").c_str(), stderr);
    std::fflush(stderr);
    std::_Exit(1);
}

// ARMS (S24 CL-L0; moved here from Registry/LegacyValidation.h by S24 CL-L2 so that every target's validations can
// use them).  An arm of a surviving validation that needs the legacy route runs under
//     if (oc::legacy::RunLegacyArm("FooValidation", "b")) { ... the arm ... }
// ON: returns true, prints nothing; OFF: prints `SKIP-ARM FooValidation arm=b: OC_ENABLE_LEGACY=OFF` on stdout and
// returns false, so the smaller check count is explained.

// The SKIP-ARM line, without its newline (pure).
inline std::string SkipArmLine(const char* name, const char* arm) {
    return std::string("SKIP-ARM ") + name + " arm=" + arm + ": OC_ENABLE_LEGACY=OFF";
}

// true in an ON build; in an OFF build prints SkipArmLine(name, arm) and returns false.
inline bool RunLegacyArm(const char* name, const char* arm) {
    if (kLegacyBuild) return true;
    std::cout << SkipArmLine(name, arm) << std::endl;
    return false;
}

} // namespace legacy
} // namespace oc

#endif // OC_LEGACY_CONFIG_H
