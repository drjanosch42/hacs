// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// EngineCheck.h -- the project's NDEBUG policy (matrix-engine Phase 0)
// =====================================================================
//
// Rationale. Before Phase 0 the tree was built -O0/Debug, so bare `assert`
// was always live. The default configuration is now RelWithDebInfo, which
// defines NDEBUG and turns every `assert` into a no-op -- silently deleting
// invariants that were part of the code's contract (e.g. the block-RREF
// pivot-count bound in BlockFlatMatrix.cpp). Two macros replace `assert`:
//
//   ENGINE_CHECK(cond)        Always compiled in, in every configuration.
//                             For O(1) invariants: dimension agreement,
//                             index/bound checks, precondition guards.
//                             On failure it throws std::logic_error carrying
//                             "file:line: ENGINE_CHECK failed: <condition>".
//                             It throws (rather than abort()ing) so that the
//                             validation harness reports a failure instead of
//                             killing the process mid-suite.
//
//   ENGINE_DEBUG_CHECK(cond)  Compiled out under NDEBUG, exactly like assert.
//                             For checks whose cost scales with the problem
//                             size (O(n), O(n^2), re-deriving a result).
//
// ENGINE_CHECK_MSG / ENGINE_DEBUG_CHECK_MSG take an extra literal message
// (the `assert(cond && "text")` idiom, without the string-in-boolean-context
// wart).
//
// ALL NEW CODE USES THESE, NEVER A BARE `assert`. Existing `assert`s in the
// *Validation.cpp files are deliberately left alone: those TUs are the test
// harness, and are always built with asserts live in the Check configuration.

#include <stdexcept>
#include <string>

namespace engine {

// Out-of-line-ish (inline, but never on the hot path) failure reporter, so the
// check itself compiles down to a predictable-branch test.
[[noreturn]] inline void engine_check_failed(const char* condition,
                                             const char* file,
                                             int line,
                                             const char* message) {
    std::string what;
    what.reserve(160);
    what += file;
    what += ':';
    what += std::to_string(line);
    what += ": ENGINE_CHECK failed: ";
    what += condition;
    if (message != nullptr && message[0] != '\0') {
        what += " -- ";
        what += message;
    }
    throw std::logic_error(what);
}

} // namespace engine

#define ENGINE_CHECK_MSG(cond, msg)                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            ::engine::engine_check_failed(#cond, __FILE__, __LINE__, (msg));   \
        }                                                                      \
    } while (false)

#define ENGINE_CHECK(cond) ENGINE_CHECK_MSG(cond, "")

#ifdef NDEBUG
#  define ENGINE_DEBUG_CHECK_MSG(cond, msg) ((void)0)
#  define ENGINE_DEBUG_CHECK(cond)          ((void)0)
#else
#  define ENGINE_DEBUG_CHECK_MSG(cond, msg) ENGINE_CHECK_MSG(cond, msg)
#  define ENGINE_DEBUG_CHECK(cond)          ENGINE_CHECK(cond)
#endif
