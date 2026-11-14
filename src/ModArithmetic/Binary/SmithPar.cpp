// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (slot-24 PAR-2K) The ring2k / ring3k Smith thread pool and the HECKE_DESCENT_SMITH_MT lever.  See SmithPar.h.

#include "SmithPar.h"
#include "ParWorker.h"
#include "../Engine/Threads.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define SMITHPAR_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#  define SMITHPAR_PAUSE() __asm__ __volatile__("yield")
#else
#  define SMITHPAR_PAUSE() do {} while (0)
#endif

namespace engine {
namespace smithpar {

namespace {

constexpr uint64_t kDefaultMinWork = uint64_t(1) << 20;   // ~1 M multiply-adds: below this a region costs more than it saves
constexpr uint32_t kSpin = 20000;                          // pause iterations before a worker sleeps (~0.1-1 ms)

thread_local uint32_t tl_enable_depth = 0;
thread_local bool tl_in_task = false;

std::atomic<uint32_t> g_threads_override{0};   // 0 = default
std::atomic<uint64_t> g_min_work{kDefaultMinWork};
std::atomic<uint32_t> g_panel_override{0};     // 0 = default / env
std::atomic<uint64_t> g_regions_par{0}, g_regions_ser{0}, g_tasks{0}, g_panel_splits{0}, g_regions_scoped{0}, g_regions_n{0};
std::atomic<uint32_t> g_max_threads{1};
// (slot-24 PAR-F2) the per-phase wall clocks (nanoseconds; diagnostic only)
std::atomic<uint64_t> g_phase_ns[kPhaseCount] = {};
std::atomic<uint64_t> g_f2_lazy_calls{0}, g_f2_par_searches{0};

// (S24 CL-5, C-08) the two on/off levers on EnvLever: "" = unset; exactly 0 = off; 1 = on; anything else warns once and stays on
// (source=env).  Read once per process.
constexpr lever::Policy lever_policy(const char* warning) {
    return lever::Policy{true, lever::Empty::Unset, lever::Match::Exact01, lever::Cache::Once, warning, "default", "env", "env"};
}
// (S26 LEVERS-26) The env arms HECKE_DESCENT_SMITH_MT=0 / HECKE_DESCENT_SMITH_F2_MT=0 are RETIRED in OC_ENABLE_LEGACY=OFF builds
// (refused, exit 1: EnvLever.h EnvArmRetiredLever); the serial code stays (ENGINE_THREADS=1, ScopedLever / ScopedF2Lever).
// HECKE_DESCENT_SMITH_PANEL is a tuning knob, not a lever (reclassified by LEVERS-26), and is unchanged.
lever::EnvArmRetiredLever g_mt_lever("HECKE_DESCENT_SMITH_MT",
                           lever_policy("WARNING: HECKE_DESCENT_SMITH_MT=%s is neither 0 nor 1; the threaded Smith stays on (default)\n"));
lever::EnvArmRetiredLever g_f2_lever(
    "HECKE_DESCENT_SMITH_F2_MT",
    lever_policy("WARNING: HECKE_DESCENT_SMITH_F2_MT=%s is neither 0 nor 1; the threaded F_2 / F_3 echelon stays on (default)\n"));
bool env_set(const lever::EnvLever& l) { return l.env_state() != lever::State::Unset; }   // set and non-empty
bool scoped(const lever::EnvLever& l) { return std::strcmp(l.source(), "scoped") == 0; }

// (PAR-UNIFY) HECKE_DESCENT_SMITH_PANEL: the threaded route's LU panel (read-only knob).  (slot-24 PANEL-IMPL) Unset = auto
// (descent_panel_2k: kLargePanel for a ring2k LU of >= kLargePanelRows rows, else kDefaultPanel; ring3k kDefaultPanel); a valid
// value wins for every size and both rings.  env_panel() == 0 means auto.
constexpr uint32_t kMaxPanel = 4096;
const char* panel_env() {
    static const char* v = [] {
        const char* e = std::getenv("HECKE_DESCENT_SMITH_PANEL");
        return (e && *e) ? e : static_cast<const char*>(nullptr);
    }();
    return v;
}
uint32_t env_panel() {
    static const uint32_t w = [] {
        const char* e = panel_env();
        if (!e) return 0u;
        char* end = nullptr;
        errno = 0;
        const long long v = std::strtoll(e, &end, 10);
        if (end == e || *end != '\0' || errno != 0 || v < 1 || v > static_cast<long long>(kMaxPanel)) {
            std::fprintf(stderr,
                         "WARNING: HECKE_DESCENT_SMITH_PANEL=%s is not an integer in [1, %u]; the panel stays auto (default: %u, %u from %u rows)\n",
                         e, kMaxPanel, kDefaultPanel, kLargePanel, kLargePanelRows);
            return 0u;
        }
        return static_cast<uint32_t>(v);
    }();
    return w;
}
// (PANEL-IMPL) the LUs the threaded ring2k route ran, by panel (diagnostic: the EFFECTIVE exit line's lu_panels=)
std::atomic<uint64_t> g_lu_small{0}, g_lu_large{0}, g_lu_other{0};

// (PAR-UNIFY) The thread budget: ENGINE_THREADS (strict), else the CPU affinity count (capped).  (S24 CL-5, C-05) Threads.h's
// kSmithPar preset is this reader verbatim, its WARNING texts included; read once per process.
static_assert(threads::kSmithPar.default_cap == kDefaultThreadCap && threads::kSmithPar.env_cap == kMaxThreads,
              "Threads.h kSmithPar caps == SmithPar.h");
const threads::Result& thread_budget() {
    static const threads::Result r = threads::configured_result(threads::kSmithPar);
    return r;
}

void print_effective_at_exit() { std::fprintf(stderr, "%s\n", effective_text("exit").c_str()); }

// Read at start-up: a set HECKE_DESCENT_SMITH_MT / HECKE_DESCENT_SMITH_PANEL prints its EFFECTIVE line now (phase=start, so
// a run killed by a wall-clock limit still shows the mode) and again at exit (phase=exit, with the region counts).
const bool g_env_read = [] {
    g_mt_lever.env_state();
    env_panel();
    g_f2_lever.env_state();
    if (env_set(g_mt_lever) || panel_env() || env_set(g_f2_lever)) {
        std::fprintf(stderr, "%s\n", effective_text("start").c_str());
        std::atexit(print_effective_at_exit);
    }
    return true;
}();

// The pool.  One region at a time (region_m).  The control word ctl = (generation << 32) | open << 31 | nparts: the
// caller writes the job fields, publishes ctl OPEN (seq_cst), runs tasks itself, waits until every task has finished
// (tasks_done == ntasks, or an error), CLOSES ctl and then waits until no worker is inside the region (active == 0).
// A worker that sees an open generation g with idx < nparts increments `active`, re-reads ctl, and backs out unless
// it is still g open (a seq_cst Dekker pair with the caller's close-then-read-active), so the job fields are never read
// after the caller may rewrite them.  The caller never waits for a worker that did not join: a descheduled worker costs
// nothing.  Idle workers spin on ctl for a while, then sleep on cv_work (counted in `sleepers`; the caller notifies only
// when someone sleeps: publish-then-read-sleepers against increment-then-re-read-ctl under the mutex).
struct Pool {
    static constexpr uint64_t kOpen = uint64_t(1) << 31;
    std::mutex region_m;
    std::mutex m;
    std::condition_variable cv_work;
    std::atomic<uint64_t> ctl{0};
    std::atomic<uint32_t> sleepers{0};
    std::atomic<uint32_t> active{0};
    std::atomic<const std::function<void(uint32_t)>*> fn{nullptr};
    std::atomic<uint32_t> ntasks{0};
    std::atomic<uint32_t> next{0};
    std::atomic<uint32_t> tasks_done{0};
    std::atomic<bool> failed{false};
    std::mutex err_m;
    std::exception_ptr err;                             // guarded by err_m
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;                   // grown under region_m
    uint32_t gen = 0;                                   // the caller's counter (under region_m)

    static bool is_open(uint64_t c) { return (c & kOpen) != 0; }
    static uint32_t gen_of(uint64_t c) { return static_cast<uint32_t>(c >> 32); }
    static uint32_t parts_of(uint64_t c) { return static_cast<uint32_t>(c & (kOpen - 1)); }

    void claim(const std::function<void(uint32_t)>& f, uint32_t nt) {
        for (;;) {
            if (failed.load(std::memory_order_relaxed)) break;
            const uint32_t t = next.fetch_add(1, std::memory_order_relaxed);
            if (t >= nt) break;
            try {
                f(t);
            } catch (...) {
                std::lock_guard<std::mutex> g(err_m);
                if (!err) err = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
            }
            tasks_done.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    // A worker waits for an open generation other than `seen` in which it takes part.
    void worker_loop(uint32_t idx) {
        uint32_t seen = 0;
        for (;;) {
            auto wanted = [&](uint64_t c) { return is_open(c) && gen_of(c) != seen && idx < parts_of(c); };
            uint64_t c = ctl.load(std::memory_order_seq_cst);
            for (uint32_t s = 0; s < kSpin && !wanted(c); ++s) {
                SMITHPAR_PAUSE();
                c = ctl.load(std::memory_order_seq_cst);
            }
            if (!wanted(c)) {
                std::unique_lock<std::mutex> lk(m);
                sleepers.fetch_add(1, std::memory_order_seq_cst);
                cv_work.wait(lk, [&] {
                    c = ctl.load(std::memory_order_seq_cst);
                    return stop.load() || wanted(c);
                });
                sleepers.fetch_sub(1, std::memory_order_seq_cst);
                if (stop.load()) return;
            }
            seen = gen_of(c);
            active.fetch_add(1, std::memory_order_seq_cst);
            if (ctl.load(std::memory_order_seq_cst) == c) {
                const std::function<void(uint32_t)>* f = fn.load(std::memory_order_acquire);
                const uint32_t nt = ntasks.load(std::memory_order_acquire);
                tl_in_task = true;
                claim(*f, nt);
                tl_in_task = false;
            }
            active.fetch_sub(1, std::memory_order_seq_cst);
        }
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop.store(true);
        }
        cv_work.notify_all();
        for (auto& t : workers) t.join();
    }
};

Pool& pool() {
    // Leaked on purpose: no worker may be joined during static destruction while a late region could still run.
    static Pool* p = new Pool();
    return *p;
}

void run_serial(uint32_t ntasks, const std::function<void(uint32_t)>& fn) {
    g_regions_ser.fetch_add(1, std::memory_order_relaxed);
    for (uint32_t t = 0; t < ntasks; ++t) fn(t);
}

bool scoped_override_active() {
    return scoped(g_mt_lever) || g_threads_override.load(std::memory_order_relaxed) != 0;
}

void run_region(uint32_t T, uint32_t ntasks, const std::function<void(uint32_t)>& fn, bool from_n);

} // namespace

uint32_t default_threads() { return thread_budget().threads; }

const char* default_threads_source() { return thread_budget().source; }

uint32_t explicit_panel() {
    const uint32_t o = g_panel_override.load(std::memory_order_relaxed);
    return o ? o : env_panel();
}

uint32_t descent_panel() {
    const uint32_t x = explicit_panel();
    return x ? x : kDefaultPanel;
}

uint32_t descent_panel_2k(uint32_t rows) {
    const uint32_t x = explicit_panel();
    if (x) return x;
    return rows >= kLargePanelRows ? kLargePanel : kDefaultPanel;
}

uint32_t route_panel_2k(uint32_t serial_panel, uint32_t rows) {
    if (threads() <= 1) return serial_panel;
    const uint32_t w = descent_panel_2k(rows);
    (w == kDefaultPanel ? g_lu_small : w == kLargePanel ? g_lu_large : g_lu_other).fetch_add(1, std::memory_order_relaxed);
    return w;
}

const lever::EnvLever& mt_lever() { return g_mt_lever; }
const lever::EnvLever& f2_lever() { return g_f2_lever; }
bool lever_enabled() { return g_mt_lever.enabled(); }
bool f2_lever_enabled() { return g_f2_lever.enabled(); }

bool f2_route() { return f2_lever_enabled() && threads() > 1; }

void add_phase_ns(Phase p, uint64_t ns) {
    if (static_cast<uint32_t>(p) < kPhaseCount) g_phase_ns[static_cast<uint32_t>(p)].fetch_add(ns, std::memory_order_relaxed);
}
double phase_seconds(Phase p) {
    return static_cast<uint32_t>(p) < kPhaseCount ? 1e-9 * static_cast<double>(g_phase_ns[static_cast<uint32_t>(p)].load()) : 0.0;
}
static uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
PhaseTimer::PhaseTimer(Phase p) : p_(p), t0_(now_ns()) {}
PhaseTimer::~PhaseTimer() { add_phase_ns(p_, now_ns() - t0_); }

void count_f2_lazy_call() { g_f2_lazy_calls.fetch_add(1, std::memory_order_relaxed); }
void count_f2_par_search() { g_f2_par_searches.fetch_add(1, std::memory_order_relaxed); }

uint32_t threads() {
    // (PAR-ASM audit F4) a worker of an outer descent region (ParWorker.h) runs its Smith forms serially: no nested pool region
    if (tl_enable_depth == 0 || tl_in_task || engine::parworker::active() || !lever_enabled()) return 1;
    const uint32_t o = g_threads_override.load(std::memory_order_relaxed);
    return o ? std::min(o, kMaxThreads) : default_threads();
}

uint64_t min_work() { return g_min_work.load(std::memory_order_relaxed); }

void parallel_for(uint32_t ntasks, const std::function<void(uint32_t)>& fn) {
    if (ntasks == 0) return;
    run_region(threads(), ntasks, fn, false);
}

void parallel_for_n(uint32_t nthreads, uint32_t ntasks, const std::function<void(uint32_t)>& fn) {
    if (ntasks == 0) return;
    run_region(tl_in_task || engine::parworker::active() ? 1u : std::min(nthreads, kMaxThreads), ntasks, fn, true);
}

namespace {
void run_region(uint32_t T, uint32_t ntasks, const std::function<void(uint32_t)>& fn, bool from_n) {
    if (T <= 1 || ntasks == 1) { run_serial(ntasks, fn); return; }
    Pool& P = pool();
    std::unique_lock<std::mutex> rl(P.region_m, std::try_to_lock);
    if (!rl.owns_lock()) { run_serial(ntasks, fn); return; }
    uint32_t parts = std::min(T, ntasks) - 1;   // workers that may take part (the caller is the extra one)
    // (PAR-ASM audit F4, integration) a failed thread creation degrades the region to the workers already started (at worst
    // the caller alone): it never throws out of the shared pool
    try {
        while (P.workers.size() < parts) {
            const uint32_t idx = static_cast<uint32_t>(P.workers.size());
            P.workers.emplace_back([&P, idx] { P.worker_loop(idx); });
        }
    } catch (const std::system_error&) {
    }
    parts = std::min<uint32_t>(parts, static_cast<uint32_t>(P.workers.size()));
    P.fn.store(&fn, std::memory_order_relaxed);
    P.ntasks.store(ntasks, std::memory_order_relaxed);
    P.next.store(0, std::memory_order_relaxed);
    P.tasks_done.store(0, std::memory_order_relaxed);
    P.failed.store(false, std::memory_order_relaxed);
    ++P.gen;
    if (P.gen == 0) ++P.gen;   // generation 0 is the workers' initial `seen`
    const uint64_t open = (static_cast<uint64_t>(P.gen) << 32) | Pool::kOpen | parts;
    P.ctl.store(open, std::memory_order_seq_cst);
    if (P.sleepers.load(std::memory_order_seq_cst) > 0) {
        std::lock_guard<std::mutex> lk(P.m);
        P.cv_work.notify_all();
    }
    tl_in_task = true;
    P.claim(fn, ntasks);
    tl_in_task = false;
    for (uint32_t s = 0; P.tasks_done.load(std::memory_order_acquire) != ntasks && !P.failed.load(std::memory_order_acquire); ++s) {
        if (s < kSpin) SMITHPAR_PAUSE();
        else std::this_thread::yield();
    }
    P.ctl.store(static_cast<uint64_t>(P.gen) << 32, std::memory_order_seq_cst);   // closed
    for (uint32_t s = 0; P.active.load(std::memory_order_seq_cst) != 0; ++s) {
        if (s < kSpin) SMITHPAR_PAUSE();
        else std::this_thread::yield();
    }
    std::exception_ptr e;
    {
        std::lock_guard<std::mutex> g(P.err_m);
        e = P.err;
        P.err = nullptr;
    }
    g_regions_par.fetch_add(1, std::memory_order_relaxed);
    if (scoped_override_active()) g_regions_scoped.fetch_add(1, std::memory_order_relaxed);
    if (from_n) g_regions_n.fetch_add(1, std::memory_order_relaxed);
    g_tasks.fetch_add(ntasks, std::memory_order_relaxed);
    uint32_t mx = g_max_threads.load(std::memory_order_relaxed);
    const uint32_t used = parts + 1;
    while (used > mx && !g_max_threads.compare_exchange_weak(mx, used, std::memory_order_relaxed)) {}
    if (e) std::rethrow_exception(e);
}
} // namespace

ScopedEnable::ScopedEnable() { ++tl_enable_depth; }
ScopedEnable::~ScopedEnable() { --tl_enable_depth; }

ScopedThreads::ScopedThreads(uint32_t n) : saved_(g_threads_override.exchange(n)) {}
ScopedThreads::~ScopedThreads() { g_threads_override.store(saved_); }

ScopedPanel::ScopedPanel(uint32_t w) : saved_(g_panel_override.exchange(std::min(w, kMaxPanel))) {}
ScopedPanel::~ScopedPanel() { g_panel_override.store(saved_); }

ScopedMinWork::ScopedMinWork(uint64_t w) : saved_(g_min_work.exchange(w)) {}
ScopedMinWork::~ScopedMinWork() { g_min_work.store(saved_); }

Stats stats() {
    Stats s;
    s.regions_parallel = g_regions_par.load();
    s.regions_serial = g_regions_ser.load();
    s.tasks = g_tasks.load();
    s.max_threads = g_max_threads.load();
    s.panel_splits = g_panel_splits.load();
    s.regions_scoped = g_regions_scoped.load();
    s.regions_parallel_n = g_regions_n.load();
    s.f2_lazy_calls = g_f2_lazy_calls.load();
    s.f2_par_searches = g_f2_par_searches.load();
    return s;
}

void count_panel_split() { g_panel_splits.fetch_add(1, std::memory_order_relaxed); }

// The EFFECTIVE line's panel= field (PANEL-IMPL): the explicit width when HECKE_DESCENT_SMITH_PANEL is valid, else
// "auto(<small>|<large>) panel_rows=<threshold>"; 64 (Params::panel) on the serial route.
static std::string panel_text(bool threaded) {
    char b[96];
    const uint32_t x = env_panel();
    if (!threaded) std::snprintf(b, sizeof b, "64");
    else if (x) std::snprintf(b, sizeof b, "%u", x);
    else std::snprintf(b, sizeof b, "auto(%u|%u) panel_rows=%u", kDefaultPanel, kLargePanel, kLargePanelRows);
    return b;
}

std::string effective_text(const char* phase) {
    // mode / lever / threads are what the descent's unscoped calls run (the env and the defaults), not a validation's
    // scoped override; regions run under such an override are counted apart (scoped_regions).
    const Stats s = stats();
    const bool lever = g_mt_lever.env_enabled();
    const uint32_t T = default_threads();
    const bool threaded = lever && T > 1;
    char buf[1024];
    const int n = std::snprintf(buf, sizeof buf,
                                "EFFECTIVE HECKE_DESCENT_SMITH_MT phase=%s mode=%s lever=%s source=%s threads=%u threads_source=%s panel=%s f2=%s f2_source=%s",
                                phase, threaded ? "threaded" : "serial", lever ? "on" : "off", env_set(g_mt_lever) ? "env" : "default", T,
                                default_threads_source(), panel_text(threaded).c_str(), (threaded && g_f2_lever.env_enabled()) ? "threaded" : "serial",
                                env_set(g_f2_lever) ? "env" : "default");
    if (std::strcmp(phase, "start") != 0 && n > 0 && static_cast<size_t>(n) < sizeof buf)
        std::snprintf(buf + n, sizeof buf - static_cast<size_t>(n),
                      " max_used=%u regions=parallel:%llu,serial:%llu d2_regions=%llu tasks=%llu panel_splits=%llu scoped_regions=%llu"
                      " lu_panels=%u:%llu,%u:%llu,other:%llu",
                      s.max_threads, static_cast<unsigned long long>(s.regions_parallel - s.regions_parallel_n),
                      static_cast<unsigned long long>(s.regions_serial), static_cast<unsigned long long>(s.regions_parallel_n),
                      static_cast<unsigned long long>(s.tasks), static_cast<unsigned long long>(s.panel_splits),
                      static_cast<unsigned long long>(s.regions_scoped), kDefaultPanel,
                      static_cast<unsigned long long>(g_lu_small.load()), kLargePanel, static_cast<unsigned long long>(g_lu_large.load()),
                      static_cast<unsigned long long>(g_lu_other.load()));
    if (std::strcmp(phase, "start") != 0) {
        const size_t used = std::strlen(buf);
        std::snprintf(buf + used, sizeof buf - used,
                      " f2_lazy=%llu f2_par_searches=%llu t_s=f2_pack:%.2f,f2_ech:%.2f,f3_pack:%.2f,f3_ech:%.2f,level:%.2f,lu:%.2f,ech_search:%.2f,ech_flush:%.2f,ech_table:%.2f,perm:%.2f,schur:%.2f",
                      static_cast<unsigned long long>(s.f2_lazy_calls), static_cast<unsigned long long>(s.f2_par_searches),
                      phase_seconds(Phase::F2Pack), phase_seconds(Phase::F2Echelon), phase_seconds(Phase::F3Pack),
                      phase_seconds(Phase::F3Echelon), phase_seconds(Phase::LevelPass), phase_seconds(Phase::LU),
                      phase_seconds(Phase::EchSearch), phase_seconds(Phase::EchFlush), phase_seconds(Phase::EchTable),
                      phase_seconds(Phase::Perm), phase_seconds(Phase::Schur));
    }
    return buf;
}

} // namespace smithpar
} // namespace engine
