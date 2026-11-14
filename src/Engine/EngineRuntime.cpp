// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineRuntime.cpp -- see EngineRuntime.h.
#include "EngineRuntime.h"

#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/EnvLever.h"
#include "EngineCostModel.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>

namespace engine {
namespace tape {

namespace {
// (matrix-engine Phase 6) THREAD-LOCAL, not process-wide: ThreadPool::
// parallel_for() serialises concurrent callers on one region mutex, so
// independent pipelines running on different threads (the campaign driver's
// shape-group batch) must each be able to select their OWN pool.  A capture is
// registered below so run_concurrently()'s task threads inherit the launching
// thread's choice, exactly as they inherit elim::ScopedLegacy.
thread_local ThreadPool* g_pool_override = nullptr;
// (slot-8 WP-6b) the thread-local device override and its tally (header).
thread_local ThreadDevice g_thread_device = ThreadDevice::Inherit;
thread_local DeviceTally* g_thread_tally = nullptr;
std::mutex g_opts_mutex;
bool g_opts_overridden = false;
PassOptions g_opts_override;
std::atomic<bool> g_cache_bypass{false};
std::mutex& ctx_mutex() { static std::mutex m; return m; }
std::vector<ThreadContextCapture>& ctx_captures() { static std::vector<ThreadContextCapture> v; return v; }
}

// ---- thread context ----------------------------------------------------------------
void register_thread_context(ThreadContextCapture capture) {
    std::lock_guard<std::mutex> lk(ctx_mutex());
    ctx_captures().push_back(std::move(capture));
}
std::vector<ThreadContextInstaller> capture_thread_context() {
    std::vector<ThreadContextCapture> caps;
    { std::lock_guard<std::mutex> lk(ctx_mutex()); caps = ctx_captures(); }
    std::vector<ThreadContextInstaller> out;
    out.reserve(caps.size());
    for (const ThreadContextCapture& c : caps) out.push_back(c());
    return out;
}

// ---- pool ----------------------------------------------------------------------
ThreadPool& consumer_pool() {
    ThreadPool* p = g_pool_override;
    return p != nullptr ? *p : ThreadPool::global();
}
ThreadPool* consumer_pool_override() { return g_pool_override; }
void set_consumer_pool_override(ThreadPool* pool) { g_pool_override = pool; }

// ---- pass options ----------------------------------------------------------------
PassOptions consumer_pass_options() {
    {
        std::lock_guard<std::mutex> lk(g_opts_mutex);
        if (g_opts_overridden) return apply_thread_device(g_opts_override);
    }
    return apply_thread_device(PassOptions::from_env());
}

// ---- the thread-local device override (WP-6b) ---------------------------------------
ThreadDevice thread_device_override() { return g_thread_device; }
DeviceTally* thread_device_tally() { return g_thread_tally; }
PassOptions apply_thread_device(PassOptions o) {
    switch (g_thread_device) {
        case ThreadDevice::Inherit: break;
        case ThreadDevice::Cpu:     // from_env()'s `cpu` mapping
            o.device_policy = Device::CPU; o.device_model = false; o.policy = DevicePolicy::Cpu;
            break;
        case ThreadDevice::Gpu:     // from_env()'s `gpu` mapping (never `hybrid`: see the header)
            o.device_policy = Device::GPU; o.device_model = true; o.policy = DevicePolicy::Gpu;
            break;
    }
    return o;
}
void note_lean_host_run() {
    DeviceTally* t = g_thread_tally;
    if (t != nullptr) t->lean_host_programs.fetch_add(1, std::memory_order_relaxed);
}
void note_program_run(uint32_t gpu_nodes) {
    DeviceTally* t = g_thread_tally;
    if (t == nullptr) return;
    t->programs.fetch_add(1, std::memory_order_relaxed);
    if (gpu_nodes != 0) {
        t->gpu_programs.fetch_add(1, std::memory_order_relaxed);
        t->gpu_nodes.fetch_add(gpu_nodes, std::memory_order_relaxed);
    }
}
ScopedThreadDevice::ScopedThreadDevice(ThreadDevice d, DeviceTally* tally)
    : saved_dev_(g_thread_device), saved_tally_(g_thread_tally) {
    g_thread_device = d;
    if (tally != nullptr) g_thread_tally = tally;
}
ScopedThreadDevice::~ScopedThreadDevice() {
    g_thread_device = saved_dev_;
    g_thread_tally = saved_tally_;
}
// (hecke-extensions slot 9, WP-6; slot-8 WP-6b finding 2) the elimination
// driver compiles under exactly the consumers' options: the process-wide
// ScopedPassOptions override (or from_env() without one), with the thread's
// device override applied -- which consumer_pass_options() already does.
// With no override installed this is from_env(), as before.
PassOptions elim_pass_options() { return consumer_pass_options(); }
bool pass_options_overridden() { std::lock_guard<std::mutex> lk(g_opts_mutex); return g_opts_overridden; }
void set_pass_options_override(const PassOptions* opts) {
    std::lock_guard<std::mutex> lk(g_opts_mutex);
    g_opts_overridden = (opts != nullptr);
    if (opts != nullptr) g_opts_override = *opts;
}
ScopedPassOptions::ScopedPassOptions(const PassOptions& o) : opts_(o) {
    {
        std::lock_guard<std::mutex> lk(g_opts_mutex);
        had_override_ = g_opts_overridden;
        saved_ = g_opts_override;
    }
    set_pass_options_override(&opts_);
}
ScopedPassOptions::~ScopedPassOptions() { set_pass_options_override(had_override_ ? &saved_ : nullptr); }

uint64_t pass_options_fingerprint(const PassOptions& o) {
    uint64_t f = 0;
    f |= o.fold ? 1u : 0u;
    f |= o.cse ? 2u : 0u;
    f |= o.fuse ? 4u : 0u;
    f |= o.layout ? 8u : 0u;
    f |= o.ksplit ? 16u : 0u;
    f |= o.batch ? 32u : 0u;
    f |= o.arena ? 64u : 0u;
    // (hecke-extensions slot 9, WP-6; slot-8 WP-6b finding 3) the device bit is
    // the EFFECTIVE policy's -- the one pass 6 compiles under
    // (Passes.cpp pass_device) -- so options that select the device through
    // `policy` alone (not the legacy pair) key their programs apart from the
    // CPU's.  Every from_env() / ScopedThreadDevice / console mapping sets the
    // pair and the policy together, so their fingerprints are unchanged.
    const DevicePolicy eff = o.effective_policy();
    f |= (eff != DevicePolicy::Cpu) ? 128u : 0u;
    // (hecke-extensions slot 10, WP-8; decision 10 (A), plan Sec. 10.6) the
    // ninth bit: bit 128 alone does not distinguish `hybrid` from `gpu`, so a
    // hybrid-compiled and a gpu-compiled program shared a cache slot.  Bit 256
    // is set iff the effective policy is Gpu specifically, so the pair
    // (bit 128, bit 256) reads (0, 0) Cpu, (1, 0) Hybrid, (1, 1) Gpu -- three
    // distinct fingerprints -- while every from_env() / ScopedThreadDevice /
    // console mapping still keys exactly as before on bit 128 alone. Value-free
    // (Sec. 11): a device result is the same bytes as the CPU result by
    // construction: this only stops two DIFFERENT compiled programs from
    // sharing one cache slot.
    f |= (eff == DevicePolicy::Gpu) ? 256u : 0u;
    return f;
}

namespace {
// The consumer-pool override is thread-local (above): a task thread must run on
// the pool its launching thread selected, not on ThreadPool::global().
const bool g_register_pool_choice = [] {
    register_thread_context([]() -> ThreadContextInstaller {
        ThreadPool* pool = consumer_pool_override();
        return [pool]() -> ThreadContextGuard {
            if (pool == nullptr) return nullptr;
            return std::make_shared<ScopedConsumerPool>(pool);
        };
    });
    return true;
}();

// (slot-8 WP-6b) The device override and its tally are thread-local: a task
// thread of a device-overridden caller compiles for the same device and
// counts into the same tally.
const bool g_register_thread_device = [] {
    register_thread_context([]() -> ThreadContextInstaller {
        const ThreadDevice d = thread_device_override();
        DeviceTally* tally = thread_device_tally();
        return [d, tally]() -> ThreadContextGuard {
            if (d == ThreadDevice::Inherit && tally == nullptr) return nullptr;
            return std::make_shared<ScopedThreadDevice>(d, tally);
        };
    });
    return true;
}();

#if OC_ENABLE_LEGACY
// engine::elim::ScopedLegacy is thread-local (legacy/DenseElimLegacy.inc): a task thread
// under a forced-legacy caller must force legacy too.  (S24 CL-L2: ON builds only.)
const bool g_register_elim_legacy = [] {
    register_thread_context([]() -> ThreadContextInstaller {
        const bool forced = engine::elim::legacy_forced();
        return [forced]() -> ThreadContextGuard {
            if (!forced) return nullptr;
            return std::make_shared<engine::elim::ScopedLegacy>();
        };
    });
    return true;
}();
#endif
}

// ---- cache bypass ------------------------------------------------------------------
bool program_cache_bypassed() { return g_cache_bypass.load(std::memory_order_acquire); }
void set_program_cache_bypassed(bool on) { g_cache_bypass.store(on, std::memory_order_release); }

// ---- host fan-out ------------------------------------------------------------------
// (S24 CL-5, C-08) the two silent on/off switches: exactly "0" = off, anything else (unset, "" included) = on; read once.
// ENGINE_HOST_ELIM_HOOK is a rollback lever (register row); ENGINE_HOST_THREADS is not (a 2026-09-06 tape switch and the
// console setting host_threads, a thread-count choice; named in the register's "Not levers" sentence, S26 LEVERS-26 F2).
constexpr lever::Policy kSilentOnPolicy{true, lever::Empty::Unset, lever::Match::Exact01, lever::Cache::Once, nullptr};
static lever::EnvLever g_host_threads_lever("ENGINE_HOST_THREADS", kSilentOnPolicy);
static lever::EnvLever g_host_elim_hook_lever("ENGINE_HOST_ELIM_HOOK", kSilentOnPolicy);

bool host_fanout_enabled() { return g_host_threads_lever.env_enabled(); }

void run_concurrently(const std::vector<std::function<void()>>& tasks) {
    const size_t n = tasks.size();
    if (n == 0) return;
    const bool serial = (n == 1) || consumer_pool().size() <= 1 || !host_fanout_enabled();
    if (serial) {
        for (const auto& t : tasks) t();
        return;
    }
    std::vector<std::exception_ptr> errors(n);
    std::vector<std::thread> threads;
    threads.reserve(n - 1);
    const std::vector<ThreadContextInstaller> context = capture_thread_context();
    auto body = [&](size_t i) {
        try { tasks[i](); } catch (...) { errors[i] = std::current_exception(); }
    };
    auto thread_body = [&](size_t i) {
        std::vector<ThreadContextGuard> guards;
        guards.reserve(context.size());
        try {
            for (const ThreadContextInstaller& install : context) guards.push_back(install());
        } catch (...) { errors[i] = std::current_exception(); return; }
        body(i);
        guards.clear();
    };
    for (size_t i = 1; i < n; ++i) threads.emplace_back(thread_body, i);
    body(0);   // the calling thread already carries its own context
    for (std::thread& t : threads) t.join();
    for (size_t i = 0; i < n; ++i)
        if (errors[i]) std::rethrow_exception(errors[i]);
}

// ---- the host eliminations' inner parallelism (slot-23 PERF-P2; EngineRuntime.h) ------------
namespace {
std::atomic<uint64_t> g_host_elim_regions{0};
std::atomic<uint64_t> g_host_elim_binds{0};   // (S27 LEV-K) installs, for the EFFECTIVE line only
thread_local uint32_t g_host_elim_depth = 0;   // installed ScopedHostElimHooks on this thread (0 or 1)
}

bool host_elim_hook_enabled() { return g_host_elim_hook_lever.env_enabled(); }
bool host_elim_hook_active() { return g_host_elim_depth > 0; }
uint64_t host_elim_hook_regions() { return g_host_elim_regions.load(std::memory_order_relaxed); }
uint64_t host_elim_hook_binds() { return g_host_elim_binds.load(std::memory_order_relaxed); }
std::string host_elim_hook_effective_line() {
    return std::string("EFFECTIVE ENGINE_HOST_ELIM_HOOK mode=") + (host_elim_hook_enabled() ? "on" : "off") +
           " source=" + g_host_elim_hook_lever.source() + " binds=" + std::to_string(host_elim_hook_binds()) +
           " regions=" + std::to_string(host_elim_hook_regions());
}

ScopedHostElimHook::ScopedHostElimHook() {
    // (audit L4) Precedence: an elimination hook already in effect on this thread -- ours (depth > 0) or
    // anybody else's, e.g. an executor's serial-node hook bound to another pool -- is never replaced.
    if (!host_elim_hook_enabled() || g_host_elim_depth > 0 || elim::parallel_hook() != nullptr ||
        consumer_pool().size() <= 1)
        return;
    // The pool is looked up when the hook RUNS, on the thread that owns the hook (parallel_hook() is
    // thread-local), so it is that thread's consumer_pool(): a ScopedConsumerPool is honoured.  The
    // executor's own hooks use the same compute-bound thread cap (Executor.cpp, TapeElim.cpp).
    hook_ = [](uint32_t nt, const std::function<void(uint32_t)>& fn) {
        g_host_elim_regions.fetch_add(1, std::memory_order_relaxed);
        consumer_pool().parallel_for(nt, kThreadsComputeBound, [&](uint32_t t, uint32_t) { fn(t); });
    };
    scoped_ = std::make_unique<elim::ScopedParallelHook>(&hook_);
    ++g_host_elim_depth;
    g_host_elim_binds.fetch_add(1, std::memory_order_relaxed);
}
ScopedHostElimHook::~ScopedHostElimHook() {
    if (scoped_ == nullptr) return;
    scoped_.reset();
    --g_host_elim_depth;
}

namespace {
// A run_concurrently() task thread of a hooked caller installs its own hook, bound to its own
// consumer_pool() -- which the pool-choice capture above has installed first (same TU, earlier
// definition, so earlier registration and earlier installation).
const bool g_register_host_elim_hook = [] {
    register_thread_context([]() -> ThreadContextInstaller {
        const bool active = host_elim_hook_active();
        return [active]() -> ThreadContextGuard {
            if (!active) return nullptr;
            return std::make_shared<ScopedHostElimHook>();
        };
    });
    return true;
}();
}

} // namespace tape
} // namespace engine
