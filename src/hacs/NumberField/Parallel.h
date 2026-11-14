// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Parallel.h -- a minimal std::thread parallel-for of the NumberField library
// (slot-19 WP-NF2). STL only. Every work item is a contiguous index range and
// writes only its own outputs, so the result never depends on the thread count.
//
// Slot-27 BIGNTT M1 (design v0.2, audit B2): the ranges run on a persistent
// process-wide worker pool instead of T freshly created std::threads per call
// (thread creation dominated a radix-5 transform at T = 16). The ranges, the
// worker indices and the exception propagation (the first exception, by
// exception_ptr, rethrown on the caller) are exactly as before. A call made
// from inside a pool task (nesting), or while another thread's call holds the
// pool, falls back to the former per-call std::threads, so the pool never
// deadlocks and never changes which range computes what.
//
// Slot-27 BIGNTT M1b (code audit M1): a pass invites at most n - 1 workers,
// each through its own condition variable, taken last-idle-first from a stack
// (the warmest threads), not the whole pool; an invited worker attaches only
// while a range is still unclaimed; the caller withdraws the invitations not
// yet taken up once every range is claimed and waits only for the workers that
// attached. So a pass costs at most n - 1 wake-ups, independent of how large
// the pool once grew. A worker that finishes watches for its next invitation
// for 200 us (yielding) before it sleeps, so back-to-back passes skip the
// futex wake-up (measured in design Sec. 13). Code audit m5: the pool is never destroyed, so std::exit from
// inside a range no longer ends in std::terminate (nor hangs on a destroyed
// condition variable). Code audit m6 (deferred to the theta card): the pool serves
// one caller at a time; a concurrent caller still spawns per call.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace nf {

namespace detail {

inline thread_local bool t_in_range_pool = false;

class RangePool {
public:
    // Never destroyed (code audit m5): no static destructor joins (or destroys the synchronisation of) workers at
    // process exit, so std::exit, also from inside a range, ends the process as it did before the pool; the idle
    // workers end with it. Reachable from a static pointer, so not a leak to LeakSanitizer.
    static RangePool& instance() {
        static RangePool* pool = new RangePool;
        return *pool;
    }

    // Runs task(i) for every i in [0, n) on the caller and up to n - 1 pool workers; false (nothing run) if
    // the caller is itself inside a pool task or the pool is busy with another caller's job.
    bool run(std::size_t n, const std::function<void(std::size_t)>& task) {
        if (t_in_range_pool || n < 2) return false;
        std::unique_lock<std::mutex> job_lock(job_mu_, std::try_to_lock);
        if (!job_lock.owns_lock()) return false;
        Job job;
        job.task = &task;
        job.n = n;
        job.pending = n;
        const std::size_t want = std::min<std::size_t>(n - 1, kMaxWorkers);
        std::vector<Slot*> invited;
        invited.reserve(want);
        {
            std::unique_lock<std::mutex> lk(mu_);
            // grow before inviting: n ranges that wait on each other all find a thread
            if (slots_.size() < want) {
                slots_.reserve(want);   // no reallocation below once a thread runs (a throw leaves no orphan)
                idle_.reserve(want);
            }
            while (slots_.size() < want) {
                std::unique_ptr<Slot> sl(new Slot);
                Slot* raw = sl.get();
                raw->th = std::thread([this, raw] { worker_loop(*raw); });
                slots_.push_back(std::move(sl));
                idle_.push_back(raw);
            }
            // every worker is idle here (the previous caller waited for its attached workers and withdrew the rest)
            while (invited.size() < want && !idle_.empty()) {
                Slot* sl = idle_.back();
                idle_.pop_back();
                sl->job.store(&job, std::memory_order_release);
                invited.push_back(sl);
            }
        }
        for (Slot* sl : invited) sl->cv.notify_one();
        t_in_range_pool = true;
        drain(job, false);
        t_in_range_pool = false;
        std::unique_lock<std::mutex> lk(mu_);
        for (Slot* sl : invited) {   // every range is claimed: withdraw the invitations not taken up
            if (sl->job.load(std::memory_order_relaxed) == &job) {
                sl->job.store(nullptr, std::memory_order_relaxed);
                idle_.push_back(sl);
            }
        }
        done_cv_.wait(lk, [&] { return job.pending.load() == 0 && attached_ == 0; });
        return true;
    }

    // Ranges run by a pool worker (not by the caller) since process start (relaxed; diagnostics only).
    std::uint64_t worker_ranges() const { return worker_ranges_.load(std::memory_order_relaxed); }

private:
    static constexpr std::size_t kMaxWorkers = 255;
    static constexpr long kSpinMicros = 200;   // M1b measurement: 0 / 50 / 200 us, see design Sec. 13

    struct Job {
        const std::function<void(std::size_t)>* task = nullptr;
        std::size_t n = 0;
        std::atomic<std::size_t> next{0};
        std::atomic<std::size_t> pending{0};
    };

    struct Slot {                 // one worker: its thread, its wake-up, its invitation (written under mu_ only)
        std::thread th;
        std::condition_variable cv;
        std::atomic<Job*> job{nullptr};
    };

    RangePool() = default;

    void drain(Job& job, bool worker) {
        for (;;) {
            const std::size_t i = job.next.fetch_add(1);
            if (i >= job.n) return;
            if (worker) worker_ranges_.fetch_add(1, std::memory_order_relaxed);
            (*job.task)(i);   // never throws: parallel_ranges wraps every task
            if (job.pending.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lk(mu_);
                done_cv_.notify_all();
            }
        }
    }

    void worker_loop(Slot& sl) {
        t_in_range_pool = true;
        for (;;) {
            // Back-to-back passes (a Barrett step makes ~ 25): watch the invitation for kSpinMicros before sleeping,
            // so the next pass finds this worker awake instead of paying a futex wake-up.
            {
                const auto t0 = std::chrono::steady_clock::now();
                for (unsigned it = 1; sl.job.load(std::memory_order_acquire) == nullptr; ++it) {
                    if ((it & 63u) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(kSpinMicros)) break;
                    std::this_thread::yield();
                }
            }
            Job* job = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu_);
                sl.cv.wait(lk, [&] { return sl.job.load(std::memory_order_relaxed) != nullptr; });
                job = sl.job.load(std::memory_order_relaxed);
                sl.job.store(nullptr, std::memory_order_relaxed);
                if (job->next.load() >= job->n) {   // nothing left to claim: do not attach
                    idle_.push_back(&sl);
                    continue;
                }
                ++attached_;
            }
            drain(*job, true);
            bool last = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                idle_.push_back(&sl);
                last = --attached_ == 0;
            }
            if (last) done_cv_.notify_all();
        }
    }

    std::mutex job_mu_;                 // held by the one caller whose job runs
    std::mutex mu_;
    std::condition_variable done_cv_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<Slot*> idle_;           // idle workers, the most recently idle on top
    std::size_t attached_ = 0;
    std::atomic<std::uint64_t> worker_ranges_{0};
};

}  // namespace detail

// f(begin, end, worker) over [0, n) in at most `threads` contiguous ranges.
template <class F>
void parallel_ranges(std::size_t n, unsigned threads, F&& f) {
    if (n == 0) return;
    const std::size_t T = std::max<std::size_t>(1, std::min<std::size_t>(threads ? threads : 1, n));
    if (T == 1) {
        f(std::size_t(0), n, 0u);
        return;
    }
    std::exception_ptr err;
    std::mutex mu;
    const std::function<void(std::size_t)> task = [&](std::size_t t) {
        const std::size_t b = n * t / T, e = n * (t + 1) / T;
        try {
            f(b, e, static_cast<unsigned>(t));
        } catch (...) {
            std::lock_guard<std::mutex> lk(mu);
            if (!err) err = std::current_exception();
        }
    };
    if (!detail::RangePool::instance().run(T, task)) {
        std::vector<std::thread> pool;
        pool.reserve(T);
        for (std::size_t t = 0; t < T; ++t) pool.emplace_back([&, t] { task(t); });
        for (auto& th : pool) th.join();
    }
    if (err) std::rethrow_exception(err);
}

// Ranges of parallel_ranges run by a pool worker rather than the caller, since process start (diagnostics: a
// validation's evidence that its threaded passes really ran on more than one thread; never a value).
inline std::uint64_t parallel_pool_worker_ranges() { return detail::RangePool::instance().worker_ranges(); }

}  // namespace nf
