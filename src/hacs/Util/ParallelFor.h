// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Util/ParallelFor.h -- the std::thread chunk loops outside the tape pool, ONE
// definition (S24 CL-1; CODEAUDIT A3; docs/implementation/slot24/CLEANUP_plan_1030.md)
// ===========================================================================
//
// Four file-local pools bypass engine::tape::ThreadPool.  Each one is
// index-disjoint with no cross-worker reduction, so its RESULT does not depend
// on the split; the split itself is still reproduced exactly (so a migration
// moves no timing, no RSS and no counter):
//
//  ParallelChunks(total, threads, plan, f(lo, hi))  -- a static split into T
//  contiguous ranges of ceil(total / T), one std::thread each:
//      t = threads; if plan.cap: t = max(1, min(t, cap))
//      serial f(0, total) when t <= 1 or total < plan.serial_below
//      T = plan.per_task ? min(t, total / per_task + 1) : t
//    kShapiroChunks     {512, 128, 0}   LValue/Gamma0NShapiro.cpp ParallelChunks
//    kMeasureChunks     {256,  64, 0}   LValue/ModularSymbolMeasure.cpp ParallelChunks
//    kEisensteinChunks  {4096,  0, 16}  LValue/EisensteinMeasure.cpp ParallelFor
//
//  ParallelForBlocks(n, threads, fn(i), block, cap) -- dynamic claiming of
//  `block` consecutive indices (HeckeAtlas.cpp ParallelFor: block 256, cap 16):
//      workers = max(1, min(threads, cap, n / block + 1)); one worker = serial.
//
// `threads` is the caller's count; Threads.h's presets give the copies' own
// ENGINE_THREADS readings (HeckeAtlas: engine::threads::configured(kCap16),
// re-read per call, as its copy does).  Placing the chunks on the global pool
// instead is a later performance question (the placement lesson).
// Header-only, STL + Threads.h.
// ===========================================================================
#pragma once
#ifndef OC_UTIL_PARALLEL_FOR_H
#define OC_UTIL_PARALLEL_FOR_H

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "../../Engine/Threads.h"

namespace oc_util {

struct ChunkPlan {
    uint64_t serial_below;   // total below this runs serially
    uint64_t per_task;       // at most total / per_task + 1 threads; 0 = no such limit
    uint32_t cap;            // the thread count is clamped to [1, cap] first; 0 = no clamp
};

constexpr ChunkPlan kShapiroChunks{512, 128, 0};
constexpr ChunkPlan kMeasureChunks{256, 64, 0};
constexpr ChunkPlan kEisensteinChunks{4096, 0, 16};

// The ranges ParallelChunks hands out, in thread order; one range {0, total} when it runs serially.
inline std::vector<std::pair<uint64_t, uint64_t>> ChunkRanges(uint64_t total, unsigned threads, const ChunkPlan& plan) {
    unsigned t = threads;
    if (plan.cap != 0) t = std::max(1u, std::min<unsigned>(t, plan.cap));
    if (t <= 1 || total < plan.serial_below) return {{0, total}};
    const unsigned T = plan.per_task != 0 ? static_cast<unsigned>(std::min<uint64_t>(t, total / plan.per_task + 1)) : t;
    std::vector<std::pair<uint64_t, uint64_t>> out;
    const uint64_t chunk = (total + T - 1) / T;
    for (unsigned i = 0; i < T; ++i) {
        const uint64_t s = static_cast<uint64_t>(i) * chunk, e = std::min<uint64_t>(total, s + chunk);
        if (s >= e) break;
        out.push_back({s, e});
    }
    return out;
}

template <class F>
void ParallelChunks(uint64_t total, unsigned threads, const ChunkPlan& plan, F f) {
    const auto ranges = ChunkRanges(total, threads, plan);
    unsigned t = threads;
    if (plan.cap != 0) t = std::max(1u, std::min<unsigned>(t, plan.cap));
    if (t <= 1 || total < plan.serial_below) {
        f(uint64_t(0), total);
        return;
    }
    std::vector<std::thread> ts;
    ts.reserve(ranges.size());
    for (const auto& r : ranges) {
        const uint64_t s = r.first, e = r.second;
        ts.emplace_back([&f, s, e]() { f(s, e); });
    }
    for (auto& th : ts) th.join();
}

// The worker count of ParallelForBlocks.
inline size_t BlockWorkers(size_t n, unsigned threads, size_t block, size_t cap) {
    return std::max<size_t>(1, std::min<size_t>({static_cast<size_t>(threads), cap, n / block + 1u}));
}

template <class Fn>
void ParallelForBlocks(size_t n, unsigned threads, Fn fn, size_t block = 256, size_t cap = 16) {
    const size_t workers = BlockWorkers(n, threads, block, cap);
    if (workers == 1) {
        for (size_t i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (size_t t = 0; t < workers; ++t)
        pool.emplace_back([&] {
            for (;;) {
                const size_t b = next.fetch_add(block);
                if (b >= n) return;
                const size_t e = std::min(n, b + block);
                for (size_t i = b; i < e; ++i) fn(i);
            }
        });
    for (std::thread& th : pool) th.join();
}

} // namespace oc_util

#endif // OC_UTIL_PARALLEL_FOR_H
