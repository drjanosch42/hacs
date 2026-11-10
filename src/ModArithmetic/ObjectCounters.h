// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef ENGINE_OBJECT_COUNTERS_H
#define ENGINE_OBJECT_COUNTERS_H

// ===========================================================================
// ObjectCounters.h -- (matrix-engine Phase 3) opt-in construction counters for
// FlatMatrixT / BlockFlatMatrixT.
//
// WHY. Phase 3's exit criteria are stated against a *measured* allocation
// baseline (00-implementation-plan.md Phase 3: "malloc count <= 5 % of today's
// 24 M"), and `--bench --count-allocs` supplies the malloc side. The dominant
// producer of those allocations is matrix construction -- 02 Sec. 2.2's seven
// conversions, the per-cell assembly loops, and the `to_matrix()` copies -- so
// "how many matrix objects were built" is the number that says whether a
// BlockBuilder rewrite actually removed work, independently of allocator
// behaviour. See 03-profiling-baseline.md Sec. 11.
//
// COST. OFF by default and then literally zero code: the counting hooks are
// empty inline functions and the accessors are `constexpr`-false branches that
// the optimiser folds away. Turn it on with
//     cmake -DENGINE_COUNT_OBJECTS=ON .
//
// Header-only (C++17 inline variables) so enabling it needs no change to the
// library's source list. Plain `uint64_t`, no atomics: the pipeline is
// single-threaded by project convention, and these are diagnostics, not a
// result path.
// ===========================================================================

#include <cstdint>

namespace engine {
namespace counters {

#ifdef ENGINE_COUNT_OBJECTS

inline constexpr bool kObjectCountersEnabled = true;

inline uint64_t g_flat_matrix_ctors = 0;
// (S24 CL-L2) BlockFlatMatrixT is compiled only with OC_ENABLE_LEGACY=ON (legacy/BlockFlatMatrix.h); in an OFF build
// nothing increments this counter, so the BENCH line keeps its `block_ctors=` field and prints 0.
inline uint64_t g_block_flat_matrix_ctors = 0;

inline void CountFlatMatrix() { ++g_flat_matrix_ctors; }
inline void CountBlockFlatMatrix() { ++g_block_flat_matrix_ctors; }
inline uint64_t FlatMatrixCtors() { return g_flat_matrix_ctors; }
inline uint64_t BlockFlatMatrixCtors() { return g_block_flat_matrix_ctors; }
inline void ResetObjectCounters() { g_flat_matrix_ctors = 0; g_block_flat_matrix_ctors = 0; }

// Counting is done by an empty member sub-object rather than by a statement in
// every constructor body: an unlisted member is default-initialised by EVERY
// constructor, copy and move included, so one declaration per class covers all
// of them and no constructor can be added later that silently escapes the
// census. Under -UENGINE_COUNT_OBJECTS these types do not exist and the member
// declarations are preprocessed away, so sizeof() and layout are untouched.
struct FlatMatrixCtorTick { FlatMatrixCtorTick() noexcept { ++g_flat_matrix_ctors; } };
struct BlockFlatMatrixCtorTick { BlockFlatMatrixCtorTick() noexcept { ++g_block_flat_matrix_ctors; } };

#else

inline constexpr bool kObjectCountersEnabled = false;

inline void CountFlatMatrix() {}
inline void CountBlockFlatMatrix() {}
inline uint64_t FlatMatrixCtors() { return 0; }
inline uint64_t BlockFlatMatrixCtors() { return 0; }
inline void ResetObjectCounters() {}

#endif

} // namespace counters
} // namespace engine

#endif // ENGINE_OBJECT_COUNTERS_H
