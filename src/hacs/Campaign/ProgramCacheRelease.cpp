// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Campaign/ProgramCacheRelease.cpp -- see the header (slot-8 WP-5, [KI] Sec. 4c).
#include "ProgramCacheRelease.h"

#include <cstdint>

#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim
#endif

#include "../../Engine/ProgramCache.h"

namespace campaign {

std::size_t EngineProgramCacheSize() {
    return engine::tape::ProgramCache<uint32_t>::global().size() +
           engine::tape::ProgramCache<uint64_t>::global().size();
}

std::size_t ReleaseEngineProgramCaches(std::size_t threshold) {
    if (EngineProgramCacheSize() <= threshold) return 0;
    // The size is read per instantiation just before its own clear, so the
    // count returned is what this call dropped as far as either mutex saw it;
    // a program inserted by another thread between the read and the clear is
    // dropped too and is simply not counted (it is recompiled on its next
    // miss, like every other).
    engine::tape::ProgramCache<uint32_t>& c32 = engine::tape::ProgramCache<uint32_t>::global();
    engine::tape::ProgramCache<uint64_t>& c64 = engine::tape::ProgramCache<uint64_t>::global();
    std::size_t released = c32.size();
    c32.clear();
    released += c64.size();
    c64.clear();
#if defined(__GLIBC__)
    // MEASURED (docs/implementation/slot8/WP-5_report.md Sec. 2): dropping
    // the programs returns their memory to the allocator but not to the
    // kernel -- VmRSS after clear() equalled VmRSS before it on every cell of
    // the ten-cell ext_first sample -- because the freed chunks sit below live
    // allocations and in the per-thread arenas.  malloc_trim(0) returns every
    // free page of every arena, which is what makes the pass's RSS flat.  It
    // frees only memory no live object owns, so no value can move, and it is
    // thread-safe (it takes each arena's lock in turn).
    malloc_trim(0);
#endif
    return released;
}

} // namespace campaign
