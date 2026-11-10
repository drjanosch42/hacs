// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// ThreadPool.h -- the persistent worker pool of the executor (Phase 4)
// =====================================================================
//
// A fixed set of workers (ENGINE_THREADS, default = hardware concurrency,
// capped at kMaxWorkers) that run parallel_for() regions: `ntasks` tasks
// distributed over at most `max_threads` threads (the per-kernel thread
// count of EngineCostModel.h) by an atomic task counter.  The counter is
// WORK DISTRIBUTION only -- which task a thread takes is not observable in
// any result, because every task writes disjoint memory.  The calling
// thread participates as a worker.  Exceptions thrown by tasks are captured
// and rethrown on the caller after the region completes.
//
// (Phase 4, performance closure) Regions of DIFFERENT callers run
// CONCURRENTLY: an active region is registered in a small registry, every
// worker claims tasks from the oldest registered region that still has
// tasks and room under its max_threads, and a worker that finishes one
// region looks for another before going idle -- so the per-degree chains,
// the concurrent Hecke degrees and the elimination's small hook regions
// share the workers instead of queueing on one region mutex, and a large
// region that started while the workers were busy picks them up as they
// free.  A caller wakes idle workers explicitly (per-worker generation
// counters and condition variables) and idle workers also notice new
// regions while they spin; callers and workers spin with `pause` (which
// yields the SMT sibling's resources) before blocking.  Nothing about which
// task runs where changed: every task still runs exactly once.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace engine {
namespace tape {

class ThreadPool {
public:
    using Fn = std::function<void(uint32_t task, uint32_t worker)>;

    explicit ThreadPool(uint32_t workers);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    uint32_t size() const { return size_; }

    void parallel_for(uint32_t ntasks, uint32_t max_threads, const Fn& fn);

    // The process-wide pool sized by ENGINE_THREADS (default: hardware
    // concurrency, at most kMaxWorkers).
    static ThreadPool& global();
    static uint32_t default_threads();

private:
    struct Region;
    struct Slot {
        std::mutex              m;
        std::condition_variable cv;
        std::atomic<uint64_t>   generation{0};   // bumped by a caller to wake THIS worker
        std::atomic<bool>       idle{false};
    };

    void worker_loop(uint32_t idx);
    Region* find_work(uint32_t idx);            // joins the oldest region with claimable tasks (or nullptr)
    void run_region(Region& r, uint32_t idx);   // claim / run loop; leaves the region on exit
    void unregister(Region& r);

    // (hecke-extensions slot 10, WP-6) The cache-line layout of the members
    // below is PINNED with alignas(64), so it no longer depends on where the
    // linker happens to place the pool object (ThreadPool::global()'s static
    // lives in .bss, whose offsets move with every unrelated global added
    // anywhere in the binary).  Line 1: the read-mostly slots_ header and the
    // WHOLE registry mutex (a mutex straddling two lines costs every lock /
    // unlock two lines); line 2: the registry state a caller writes on every
    // region and the idle workers poll (regions_, n_regions_, reg_epoch_,
    // stop_).  This is exactly the grouping the slot-8 close binary (bd6d1bb)
    // had by accident (pool at .bss offset = 32 mod 64); slot 9's binaries
    // landed it at 0 mod 64, where reg_m_ straddled lines 0/1 and
    // (10, 691)'s --bench ext pipeline_s read ~4 % slower at 16 threads with
    // identical work (same regions, nodes, runs, compiles).  Layout only: no
    // member, order of operation or memory order changed, so no value can
    // move.  sizeof(ThreadPool) 144 -> 192; the pool is created once.
    uint32_t                 size_;
    std::vector<std::thread> threads_;
    alignas(64) std::vector<std::unique_ptr<Slot>> slots_;   // one per worker (index 0 unused)
    std::mutex               reg_m_;             // the registry of active regions
    alignas(64) std::vector<Region*> regions_;
    std::atomic<uint32_t>    n_regions_{0};      // registered regions
    std::atomic<uint64_t>    reg_epoch_{0};      // bumped on register / unregister / member exit: idle workers rescan only then
    std::atomic<bool>        stop_{false};
};

} // namespace tape
} // namespace engine
