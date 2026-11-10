// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Campaign/ProgramCacheRelease.h -- the between-cells release of the engine's
// process-wide program caches (hecke-extensions slot 8, WP-5; [KI] Sec. 4c).
//
// THE DEFECT.  engine::tape::run_cached() is record-always / compile-on-miss
// into ProgramCache<Elt>::global(), which is process-wide and has no eviction.
// A process that computes many cells -- the campaign's ext pass, the
// `--atlas` build (Sigma(691): ~16 GB of growth, peak 19.5 GB) -- therefore
// never releases the compiled programs of cells it has finished.
//
// THE HOOK.  ReleaseEngineProgramCaches() clears BOTH explicit
// instantiations, ProgramCache<uint32_t>::global() and
// ProgramCache<uint64_t>::global() (ProgramCache.cpp), when their combined
// size exceeds `threshold`, and then (glibc only, `#if defined(__GLIBC__)`; a no-op on bionic / musl) malloc_trim(0): the freed
// programs' pages are otherwise kept by the allocator and VmRSS does not move
// (measured; WP-5 report).  The shipped policies are the pinned size
// thresholds kCellLoopReleaseThreshold (cell loops) and
// kJobLoopReleaseThreshold (the job loop).
//
// WHY NO VALUE CAN CHANGE.  A cleared program is recompiled on the next miss,
// and compile() is a pure function of (program, options) (ENGINE_API.md
// Sec. 10): a replayed program and a freshly compiled one run the same nodes
// on the same bindings.  The cache is a memo, never an input.
//
// CONCURRENCY.  ProgramCache::clear() takes the cache's mutex, and a lookup
// hands out a std::shared_ptr<const Program>: a thread that is replaying a
// program while another thread clears keeps its own reference alive until
// its run returns.  A clear concurrent with run_cached() is therefore
// memory-safe; its only cost is a recompile on the next miss.
// ProgramCacheReleaseValidation proves it on a two-thread clear-while-running
// arm (run it under CheckThread for the TSan statement).
//
// WHERE IT IS CALLED (outside HeckeExt/: check_heckeext_includes.sh rule 2
// forbids Engine/ inside the layer):
//   * Campaign/CampaignExt.cpp, RunExtPass: after every cell the engine ran,
//     at kCellLoopReleaseThreshold;
//   * Campaign/CampaignDriver.cpp, RunJobList: after every job (both the
//     sequential and the concurrent path -- the job loop holds the same
//     growth, ~28 programs per job), at kJobLoopReleaseThreshold;
//   * main.cpp's `--atlas` branch, as the `between_cells` callback injected
//     into hecke_ext::RunAtlasCli -> AtlasOptions::between_cells, at
//     kCellLoopReleaseThreshold;
// and it is the entry point a parallel ext pass (slot-8 WP-6) calls from its
// workers.
//
// This header is engine-free on purpose: main.cpp (an nvcc TU) includes it
// without pulling the engine's templates through nvcc.  The definition lives
// in ProgramCacheRelease.cpp.
// ===========================================================================
#pragma once
#ifndef OC_CAMPAIGN_PROGRAM_CACHE_RELEASE_H
#define OC_CAMPAIGN_PROGRAM_CACHE_RELEASE_H

#include <cstddef>

namespace campaign {

// Release whenever anything is cached (the plan's first policy, "clear after
// every cell"; kept as the named value the validation and callers that want
// it use).
constexpr std::size_t kReleaseEveryCell = 0;

// The shipped policy of the CELL loops (the ext pass, `--atlas`): the plan's
// FALLBACK -- release once the caches hold more than this many programs.
// Releasing after every cell cost 6-7 % of the ten-cell ext_first sample's
// ext-pass wall in the quietest indicative rounds (median NEW/REF 1.067 >
// 1.04, WP-5 report Sec. 4): the trim hands the pages back and the next cell
// faults them in again.  A cell at ell = 691, n <= 200 compiles ~100-120
// programs, so 512 releases every ~4-5 cells there: the retained programs
// are bounded by the threshold, and the re-fault cost is amortised over the
// cells in between.
constexpr std::size_t kCellLoopReleaseThreshold = 512;

// The shipped policy of the campaign JOB loop: release once the caches hold
// more than this many programs (~28 per job at n <= 200, so every ~70 jobs).
// Per-job release cost 4-8 % of the wall on a 729-job, 18 ms-per-job sweep
// (WP-5 report Sec. 3) -- the recompiles of concurrent jobs and the trims --
// and the threshold costs nothing measurable there.
constexpr std::size_t kJobLoopReleaseThreshold = 2048;

// The combined number of programs currently cached, over both instantiations.
std::size_t EngineProgramCacheSize();

// Clears ProgramCache<uint32_t>::global() and ProgramCache<uint64_t>::global()
// iff EngineProgramCacheSize() > threshold; returns the number of programs
// the call released (0 when it did not clear).  Thread-safe (each clear is
// under that cache's own mutex); never changes a value.
std::size_t ReleaseEngineProgramCaches(std::size_t threshold = kReleaseEveryCell);

} // namespace campaign

#endif // OC_CAMPAIGN_PROGRAM_CACHE_RELEASE_H
