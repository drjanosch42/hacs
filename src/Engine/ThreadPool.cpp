// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ThreadPool.cpp -- see ThreadPool.h.
#include "ThreadPool.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define ENGINE_CPU_PAUSE() _mm_pause()
#else
#define ENGINE_CPU_PAUSE() std::this_thread::yield()
#endif

#include "EngineCostModel.h"
#include "EngineTrace.h"
#include "Threads.h"

namespace engine {
namespace tape {

// One parallel_for call.  Lives on the caller's stack for the duration of
// the call: it is unregistered before the caller returns and no worker holds
// a pointer to it afterwards (see parallel_for).
struct ThreadPool::Region {
    const Fn*              fn;
    uint32_t               ntasks;
    uint32_t               max_threads;
    std::atomic<uint32_t>  next{0};       // next task to claim
    std::atomic<uint32_t>  done{0};       // tasks completed
    std::atomic<uint32_t>  members{0};    // threads inside the region (caller included)
    std::mutex             err_m;
    std::exception_ptr     first_error;
    std::mutex             done_m;
    std::condition_variable done_cv;
    // ENGINE_TRACE_NODES: busy seconds per worker (index 0 = the caller)
    std::vector<double>    busy;
    bool                   trace = false;
};

// (S24 CL-5, C-05) the one ENGINE_THREADS reader, Threads.h's kThreadPool preset (strtoul; 0 or unset = the hardware; capped at 32)
static_assert(threads::kThreadPool.env_cap == kMaxWorkers && threads::kThreadPool.default_cap == kMaxWorkers,
              "Threads.h kThreadPool caps == kMaxWorkers");
uint32_t ThreadPool::default_threads() { return threads::configured(threads::kThreadPool); }

// (slot-10 WP-6) the pinned cache-line layout of ThreadPool.h: the object is
// 64-byte aligned wherever it lives (.bss, stack, aligned new).
static_assert(alignof(ThreadPool) == 64, "ThreadPool's cache-line layout is pinned by alignas(64)");

ThreadPool& ThreadPool::global() {
    static ThreadPool pool(default_threads());
    return pool;
}

ThreadPool::ThreadPool(uint32_t workers) : size_(std::max<uint32_t>(1, std::min<uint32_t>(workers, kMaxWorkers))) {
    slots_.reserve(size_);
    for (uint32_t i = 0; i < size_; ++i) slots_.emplace_back(new Slot());
    regions_.reserve(64);
    for (uint32_t i = 1; i < size_; ++i) threads_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool() {
    stop_.store(true, std::memory_order_release);
    for (uint32_t i = 1; i < size_; ++i) {
        Slot& s = *slots_[i];
        { std::lock_guard<std::mutex> lk(s.m); s.generation.fetch_add(1, std::memory_order_release); }
        s.cv.notify_one();
    }
    for (std::thread& t : threads_) t.join();
}

namespace {
// Spin budget before blocking (idle workers and waiting callers).  Bounds
// the wake-up latency of the small regions; ENGINE_SPIN overrides it
// (measurement knob; cannot change a value).
uint32_t spin_iterations() {
    static const uint32_t n = [] {
        const char* e = std::getenv("ENGINE_SPIN");
        return e != nullptr ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 20000u;
    }();
    return n;
}
}

// Joins the oldest registered region that still has unclaimed tasks and room
// under its max_threads.  Membership is taken UNDER the registry lock, so a
// region that has been unregistered can never gain a member afterwards.
ThreadPool::Region* ThreadPool::find_work(uint32_t) {
    std::lock_guard<std::mutex> lk(reg_m_);
    for (Region* r : regions_) {
        if (r->next.load(std::memory_order_relaxed) >= r->ntasks) continue;
        const uint32_t m = r->members.load(std::memory_order_relaxed);
        if (m >= r->max_threads) continue;
        r->members.fetch_add(1, std::memory_order_acq_rel);
        return r;
    }
    return nullptr;
}

void ThreadPool::run_region(Region& r, uint32_t idx) {
    const Fn& fn = *r.fn;
    const bool tr = r.trace;
    try {
        for (;;) {
            const uint32_t t = r.next.fetch_add(1, std::memory_order_relaxed);
            if (t >= r.ntasks) break;
            if (tr) {
                const auto b0 = std::chrono::steady_clock::now();
                fn(t, idx);
                r.busy[idx] += std::chrono::duration<double>(std::chrono::steady_clock::now() - b0).count();
            } else fn(t, idx);
            if (r.done.fetch_add(1, std::memory_order_acq_rel) + 1 == r.ntasks) {
                std::lock_guard<std::mutex> lk(r.done_m);
                r.done_cv.notify_all();
            }
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lk(r.err_m);
            if (!r.first_error) r.first_error = std::current_exception();
        }
        // The failed task still counts as done; the remaining tasks are
        // abandoned by claiming them (the caller rethrows after the region).
        for (;;) {
            const uint32_t t = r.next.fetch_add(1, std::memory_order_relaxed);
            if (t >= r.ntasks) break;
            if (r.done.fetch_add(1, std::memory_order_acq_rel) + 1 == r.ntasks) {
                std::lock_guard<std::mutex> lk(r.done_m);
                r.done_cv.notify_all();
            }
        }
        if (r.done.fetch_add(1, std::memory_order_acq_rel) + 1 == r.ntasks) {
            std::lock_guard<std::mutex> lk(r.done_m);
            r.done_cv.notify_all();
        }
    }
    // Room for another member -- worth a rescan only while tasks remain.
    // Read BEFORE leaving: once members drops the caller may free the Region.
    const bool tasks_left = r.next.load(std::memory_order_relaxed) < r.ntasks;
    r.members.fetch_sub(1, std::memory_order_acq_rel);
    if (tasks_left) reg_epoch_.fetch_add(1, std::memory_order_release);
}

void ThreadPool::worker_loop(uint32_t idx) {
    Slot& slot = *slots_[idx];
    uint64_t seen = 0;
    uint64_t epoch_seen = 0;
    for (;;) {
        // Run every region that has work for this worker.
        for (;;) {
            epoch_seen = reg_epoch_.load(std::memory_order_acquire);
            Region* r = find_work(idx);
            if (r == nullptr) break;
            run_region(*r, idx);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        // Idle: spin on (a wake-up for this worker) or (a change of the
        // registry -- a new region, or a member leaving one that was full),
        // then block until a caller bumps this worker's generation.
        const uint32_t spins = spin_iterations();
        uint64_t g = slot.generation.load(std::memory_order_acquire);
        bool work = false;
        for (uint32_t spin = 0; spin < spins; ++spin) {
            if (g != seen) break;
            if (reg_epoch_.load(std::memory_order_acquire) != epoch_seen) { work = true; break; }
            ENGINE_CPU_PAUSE();
            g = slot.generation.load(std::memory_order_acquire);
        }
        if (work) continue;
        if (g == seen) {
            slot.idle.store(true, std::memory_order_release);
            {
                std::unique_lock<std::mutex> lk(slot.m);
                slot.cv.wait(lk, [&] { return slot.generation.load(std::memory_order_acquire) != seen; });
            }
            slot.idle.store(false, std::memory_order_release);
            g = slot.generation.load(std::memory_order_acquire);
        }
        seen = g;
    }
}

void ThreadPool::unregister(Region& r) {
    std::lock_guard<std::mutex> lk(reg_m_);
    for (size_t i = 0; i < regions_.size(); ++i)
        if (regions_[i] == &r) { regions_.erase(regions_.begin() + static_cast<std::ptrdiff_t>(i)); break; }
    n_regions_.fetch_sub(1, std::memory_order_acq_rel);
    reg_epoch_.fetch_add(1, std::memory_order_release);
}

void ThreadPool::parallel_for(uint32_t ntasks, uint32_t max_threads, const Fn& fn) {
    if (ntasks == 0) return;
    using Clock = std::chrono::steady_clock;
    const bool tr = trace_nodes_enabled();
    const Clock::time_point tw0 = tr ? Clock::now() : Clock::time_point();
    const uint32_t nthreads = std::max<uint32_t>(1, std::min<uint32_t>({max_threads, size_, ntasks}));
    if (nthreads == 1 || size_ == 1) {
        for (uint32_t t = 0; t < ntasks; ++t) fn(t, 0);
        if (tr) {
            const double w = std::chrono::duration<double>(Clock::now() - tw0).count();
            trace_add_pool_region(w, w, 1, 0.0);
        }
        return;
    }
    Region r;
    r.fn = &fn; r.ntasks = ntasks; r.max_threads = nthreads;
    r.members.store(1, std::memory_order_relaxed);   // the caller
    r.trace = tr;
    if (tr) r.busy.assign(size_, 0.0);
    {
        std::lock_guard<std::mutex> lk(reg_m_);
        regions_.push_back(&r);
        n_regions_.fetch_add(1, std::memory_order_acq_rel);
        reg_epoch_.fetch_add(1, std::memory_order_release);
    }
    // Wake up to nthreads - 1 workers, idle ones first (a busy worker ignores
    // the bump and rescans the registry when its current region ends).
    {
        uint32_t woken = 0;
        for (int pass = 0; pass < 2 && woken + 1 < nthreads; ++pass)
            for (uint32_t i = 1; i < size_ && woken + 1 < nthreads; ++i) {
                Slot& s = *slots_[i];
                if (pass == 0 && !s.idle.load(std::memory_order_acquire)) continue;
                { std::lock_guard<std::mutex> lk(s.m); s.generation.fetch_add(1, std::memory_order_release); }
                s.cv.notify_one();
                ++woken;
            }
    }
    const Clock::time_point tw1 = tr ? Clock::now() : Clock::time_point();
    run_region(r, 0);   // the caller participates (and leaves: members -= 1)
    // Wait for every task to complete: spin, then block on the region's cv.
    {
        const uint32_t spins = spin_iterations();
        for (uint32_t spin = 0; spin < spins && r.done.load(std::memory_order_acquire) < ntasks; ++spin) ENGINE_CPU_PAUSE();
        if (r.done.load(std::memory_order_acquire) < ntasks) {
            std::unique_lock<std::mutex> lk(r.done_m);
            r.done_cv.wait(lk, [&] { return r.done.load(std::memory_order_acquire) >= ntasks; });
        }
    }
    // No new member can join after unregistration; the members still inside
    // are between their last claim and their exit -- wait them out.
    unregister(r);
    while (r.members.load(std::memory_order_acquire) != 0) ENGINE_CPU_PAUSE();
    if (tr) {
        double busy = 0.0;
        for (double b : r.busy) busy += b;
        trace_add_pool_region(std::chrono::duration<double>(Clock::now() - tw1).count(), busy, nthreads,
                              std::chrono::duration<double>(tw1 - tw0).count());
    }
    if (r.first_error) std::rethrow_exception(r.first_error);
}

} // namespace tape
} // namespace engine
