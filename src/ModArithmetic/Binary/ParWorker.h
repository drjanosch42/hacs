// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
#pragma once
// (slot-24 PAR-ASM audit F4) One thread-local flag: "this thread is a worker of an outer parallel region" (the descent's
// PAR-ASM ForEachIndex: the concurrent ell >= 5 primes / pair pieces, the tree pair's tasks).  Every inner threaded layer
// consults it and runs serially on such a thread -- the descent Smith form and maps::MulMod / ProductsEqual
// (Integral/Descent/DescentSmith.cpp, default_threads()), the ring2k / ring3k pool (SmithPar.cpp, threads()) -- so an outer
// region of w workers never becomes w x 16 OS threads.  Header-only (a C++17 inline thread_local), so any pool may consult it
// without a link dependency.  It never changes a value: every inner layer's serial route computes the same entries.
#include <cstdint>

namespace engine {
namespace parworker {

inline thread_local uint32_t tl_depth = 0;

// true on a worker (or the participating caller) of an outer parallel region
inline bool active() { return tl_depth > 0; }

// marks the calling thread as a region worker for the scope's lifetime (nests)
struct Scope {
    Scope() { ++tl_depth; }
    ~Scope() { --tl_depth; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

}  // namespace parworker
}  // namespace engine
