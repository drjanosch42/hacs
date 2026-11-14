// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// EngineRuntime.h -- the consumer-side runtime glue of the tape engine
// (matrix-engine Phase 4, Stage B: consumer wiring)
// =====================================================================
//
// What the pipeline consumers (EquivariantCohomologyOnK, HeckeOperatorOnK,
// HeckeOperatorFamily) need from the engine beyond Sec. 10 of ENGINE_API.md,
// in one place so that the validations can steer every knob IN-PROCESS:
//
//   consumer_pool()          the ThreadPool the consumers execute on --
//                            ThreadPool::global() (ENGINE_THREADS) unless a
//                            validation installed a ScopedConsumerPool, which
//                            is how EngineDeterminismValidation runs the same
//                            pipeline under 1 / 4 / 16 / 32 workers without
//                            re-launching the process.
//   consumer_pass_options()  PassOptions::from_env() unless a ScopedPassOptions
//                            override is active (the ENGINE_BATCH arms); the
//                            device fields replaced by a THREAD-LOCAL
//                            ScopedThreadDevice when one is installed (WP-6b).
//   run_cached()             the record-ALWAYS / compile-on-miss protocol of the
//                            program cache (plan D4, ENGINE_API Sec. 10.6):
//                            recording is O(#ops) and cheap, so a consumer
//                            records its program every time and hands the
//                            fresh Program plus its Bindings here together with
//                            the ShapeKey; a cache hit replays the CACHED
//                            compiled program on the FRESH bindings (whose slot
//                            order is the same by construction -- the same
//                            code recorded both), a miss compiles the fresh
//                            program, runs it and inserts it.  A hit whose
//                            recorded structural hash differs from the fresh
//                            recording's is a key-design error and throws:
//                            a stale program is never replayed (D4 caveat).
//   run_concurrently()       host-side fan-out for the consumer-level parallel
//                            axes ((c) the degrees q, (d) the (p, r) pipelines):
//                            one std::thread per task -- never pool workers,
//                            which would deadlock the pool's own regions --
//                            joined before return; the FIRST failing task's
//                            exception, in task order, is rethrown, so the
//                            failure surfaced does not depend on timing.
//                            Serial when the consumer pool has one worker, so
//                            ENGINE_THREADS=1 is a genuinely single-threaded
//                            arm (the exit-criterion baseline and the TSan
//                            reference).
//
// None of these can change a computed value: every program's output is an
// exact object (D4), a replayed program computes exactly what a fresh one
// does, and the fan-out only changes WHEN independent results are produced.

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <string>

#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/EngineCheck.h"
#include "Executor.h"
#include "Passes.h"
#include "Program.h"
#include "ProgramCache.h"
#include "Recorder.h"
#include "ThreadPool.h"

namespace engine {
namespace cuda { void set_trace_tag(const char* tag); const char* trace_tag(); }   // CudaCtx.h (slot-23 P7, diagnostic)
namespace tape {

// ---- the consumers' thread pool ----------------------------------------------
ThreadPool& consumer_pool();
ThreadPool* consumer_pool_override();
void set_consumer_pool_override(ThreadPool* pool);   // nullptr: ThreadPool::global()

class ScopedConsumerPool {
    ThreadPool* saved_;
public:
    explicit ScopedConsumerPool(ThreadPool* pool) : saved_(consumer_pool_override()) { set_consumer_pool_override(pool); }
    ~ScopedConsumerPool() { set_consumer_pool_override(saved_); }
    ScopedConsumerPool(const ScopedConsumerPool&) = delete;
    ScopedConsumerPool& operator=(const ScopedConsumerPool&) = delete;
};

// ---- the consumers' pass options ---------------------------------------------
PassOptions consumer_pass_options();
bool pass_options_overridden();
void set_pass_options_override(const PassOptions* opts);   // nullptr: from_env()

class ScopedPassOptions {
    PassOptions opts_;
    bool had_override_;
    PassOptions saved_;
public:
    explicit ScopedPassOptions(const PassOptions& o);
    ~ScopedPassOptions();
    ScopedPassOptions(const ScopedPassOptions&) = delete;
    ScopedPassOptions& operator=(const ScopedPassOptions&) = delete;
};

// ---- the THREAD-LOCAL device override (hecke-extensions slot 8, WP-6b) --------
// The device policy above is process-wide: PassOptions::from_env() reads
// ENGINE_DEVICE, and the ScopedPassOptions override is one mutex-guarded
// global.  The campaign's ext pass (Campaign/CampaignExt.cpp, `ext_gpu = on`)
// runs ONE worker whose programs are compiled for the device beside CPU
// workers whose programs must not be, so the device choice needs a thread
// scope.  A ScopedThreadDevice installs, on the calling thread only, a device
// policy that REPLACES the device fields (device_policy, device_model, policy)
// of whatever the thread would otherwise use -- the process-wide override or
// from_env() -- and leaves every other pass option to that source.  It is
// captured by register_thread_context(), so run_concurrently()'s task threads
// inherit it exactly as they inherit the consumer pool.
//
//   * Inherit (the default, no override installed): consumer_pass_options()
//     and elim_pass_options() are exactly what they were before WP-6b
//     (since slot-9 WP-6 they are one function; see elim_pass_options()).
//   * Cpu / Gpu: the three device fields of from_env()'s `cpu` / `gpu`
//     mapping.  `hybrid` is not offered here: this override exists for the
//     campaign ext pass's optional device worker (Campaign/CampaignExt.cpp),
//     which is a CPU/GPU choice only -- since slot-10 WP-8
//     pass_options_fingerprint() gives `hybrid` and `gpu` distinct bits (the
//     ninth bit, decision 10 (A)), so a `Hybrid` case here would no longer
//     share a cache slot with `Gpu` either, and adding it is just outside
//     this class's scope.
//
// No value can move: a device result is the same bytes as the CPU result by
// construction (ENGINE_API.md Sec. 11), and the fingerprint keeps CPU- and
// GPU-compiled programs in different cache slots.
//
// An optional DeviceTally rides with the override (and is captured with it):
// every program the thread -- or a task thread it launched -- RUNS through
// run_cached() or the tape elimination driver is counted there, with its
// GPU-assigned node count, which is how a caller proves from the compiled
// programs themselves that a CPU worker ran zero GPU nodes (WP-6b (c)).
enum class ThreadDevice : uint8_t { Inherit = 0, Cpu = 1, Gpu = 2 };

struct DeviceTally {
    std::atomic<uint64_t> programs{0};        // programs run
    std::atomic<uint64_t> gpu_programs{0};    // ... of which with >= 1 GPU-assigned node
    std::atomic<uint64_t> gpu_nodes{0};       // GPU-assigned nodes over all of them
    // (slot-23 P7, audit R4) programs, counted in gpu_programs above (compiled placement), that the executor
    // nevertheless ran on the host under the VRAM-lean rule (Executor.h): device programs = gpu_programs - lean_host_programs.
    std::atomic<uint64_t> lean_host_programs{0};
};

ThreadDevice thread_device_override();
DeviceTally* thread_device_tally();
// Applies the calling thread's override (if any) to `o`'s device fields.
PassOptions apply_thread_device(PassOptions o);
// Records one run program in the calling thread's tally (no-op without one).
void note_program_run(uint32_t gpu_nodes);
// (slot-23 P7) records one lean host run in the calling thread's tally (no-op without one); Executor::run calls it.
void note_lean_host_run();

class ScopedThreadDevice {
    ThreadDevice saved_dev_;
    DeviceTally* saved_tally_;
public:
    // `tally` may be null; the previous tally is kept when it is.
    explicit ScopedThreadDevice(ThreadDevice d, DeviceTally* tally = nullptr);
    ~ScopedThreadDevice();
    ScopedThreadDevice(const ScopedThreadDevice&) = delete;
    ScopedThreadDevice& operator=(const ScopedThreadDevice&) = delete;
};

// The options the tape elimination driver (TapeElim.cpp) compiles under.
// Since hecke-extensions slot 9 (WP-6) this IS consumer_pass_options(): the
// process-wide ScopedPassOptions override when one is installed, from_env()
// otherwise, with the thread's device override applied.  (Slot-8 WP-6b's
// version read from_env() alone, so a ScopedPassOptions caller -- the
// determinism validations' batch / device arms, the console's `set batch` /
// `set device` -- never reached the per-panel programs.)  With no override of
// either kind installed it is exactly from_env(), as it always was.  The
// elimination programs are not cached, so no key moves; compile is a pure
// function of (program, options), so no value can.
PassOptions elim_pass_options();

// 9-bit fingerprint of the pass options that shape a compiled program; folded
// into every consumer ShapeKey so that programs compiled under different
// options never share a cache slot.  Bits 1-64 are `fold`, `cse`, `fuse`,
// `layout`, `ksplit`, `batch`, `arena`; bit 128 is set iff
// o.effective_policy() != Cpu (slot-9 WP-6: before, iff device_model &&
// device_policy == GPU, which missed an options value selecting the device by
// `policy` alone).  Bit 256 (slot-10 WP-8; decision 10 (A)) is set iff the
// effective policy is Gpu specifically, so `hybrid` (128 only) and `gpu`
// (128 and 256) no longer share a slot -- the ninth bit `run_cached()` folds
// in below.
uint64_t pass_options_fingerprint(const PassOptions& o);

// ---- program-cache bypass (validation: force a fresh compile every time) ------
bool program_cache_bypassed();
void set_program_cache_bypassed(bool on);
class ScopedProgramCacheBypass {
    bool saved_;
public:
    explicit ScopedProgramCacheBypass(bool on) : saved_(program_cache_bypassed()) { set_program_cache_bypassed(on); }
    ~ScopedProgramCacheBypass() { set_program_cache_bypassed(saved_); }
    ScopedProgramCacheBypass(const ScopedProgramCacheBypass&) = delete;
    ScopedProgramCacheBypass& operator=(const ScopedProgramCacheBypass&) = delete;
};

// ---- thread context (scoped consumer options under the fan-out) --------------
// The consumers' scoped options (EquivariantCohomologyOptions::Scoped*,
// HeckeOperatorOptions::Scoped*, engine::elim::ScopedLegacy) are THREAD-LOCAL,
// so two pipelines running concurrently with different settings cannot see
// each other's -- and a task thread started by run_concurrently() would see
// the defaults.  run_concurrently() therefore snapshots the calling thread's
// context and installs it on every task thread before the task runs.  A TU
// owning thread-local options registers ONE capture function at static-init
// time: called on the launching thread, it returns the installer that
// reproduces that state on another thread; the installer's return value (a
// guard, e.g. a ScopedLegacy) is kept alive for the task's duration.
using ThreadContextGuard = std::shared_ptr<void>;
using ThreadContextInstaller = std::function<ThreadContextGuard()>;
using ThreadContextCapture = std::function<ThreadContextInstaller()>;
void register_thread_context(ThreadContextCapture capture);
std::vector<ThreadContextInstaller> capture_thread_context();

// ---- host-side fan-out -------------------------------------------------------
// Runs every task; concurrently on std::threads when the consumer pool has
// more than one worker (and the host fan-out is not disabled by
// ENGINE_HOST_THREADS=0), serially in task order otherwise.  Every task
// thread first receives the calling thread's context (above).  Rethrows the
// first failing task's exception in TASK ORDER after every task finished.
void run_concurrently(const std::vector<std::function<void()>>& tasks);
bool host_fanout_enabled();

// ---- the host eliminations' inner parallelism (hecke-extensions slot 23, PERF-P2) ----
// engine::elim's blocked PLU (rref, column_rank_profile, rref_with_transform, inverse, ... and so
// FlatMatrix::rank / image) splits its trailing updates, triangular solves and Schur steps over
// the THREAD-LOCAL elim::ParallelFor hook when one is installed; only the tape installed it, so
// the large host eliminations of the cone (EquivariantMappingCone::compute, ConeCertificate::
// verify) and of iota^* (HeckeInvolution::compute, InvolutionCertificate::verify) ran on one
// thread (slot-22 PERF audit, item 4).  A ScopedHostElimHook binds the hook, on the calling
// thread, to consumer_pool() (looked up when the hook runs, so a ScopedConsumerPool is honoured;
// the global pool object is never touched).  It installs NOTHING when
//   * the consumer pool has one worker (ENGINE_THREADS=1: the serial path exactly as before),
//   * the calling thread already carries one (nesting is a no-op),
//   * ANY elimination hook is already in effect on the calling thread (precedence, audit L4: an
//     executor's serial-node hook or a caller's own hook is never replaced; inside a hook task
//     elim::parallel_hook() is null and nested calls are serial anyway), or
//   * ENGINE_HOST_ELIM_HOOK=0 (read once per process; the rollback arm).
// It is captured by register_thread_context(), so run_concurrently()'s task threads of a caller
// whose ScopedHostElimHook is INSTALLED get their own; a foreign hook is not propagated (as before).  No value can move: every split writes disjoint rows / columns whose
// entries are exact residues computed by the same operation sequence whatever the tiling
// (DenseElim.cpp run_hook; ENGINE_API Sec. 3 / D4), and the pivots and the RREF are unique.
// HostElimThreadDeterminismValidation gates it on pools of 1 / 4 / 16 workers.
bool host_elim_hook_enabled();
bool host_elim_hook_active();          // a ScopedHostElimHook is installed on the calling thread
uint64_t host_elim_hook_regions();     // pool regions run through such hooks, process total (diagnostic)
uint64_t host_elim_hook_binds();       // (S27 LEV-K) ScopedHostElimHooks that installed the hook, process total (diagnostic)
// (S27 LEV-K) "EFFECTIVE ENGINE_HOST_ELIM_HOOK mode=on|off source=default|env binds=<b> regions=<r>": the rollback lever's
// check line (the rollback-lever register needs an EFFECTIVE line read on a default-route run).  Printed on stderr by the
// `--campaign` CLI branch of main() only, after a campaign that bound the hook at least once or with the variable set;
// never by a validation, --help or --dry-run.  Reads counters only: it cannot move a value.
std::string host_elim_hook_effective_line();

class ScopedHostElimHook {
    elim::ParallelFor hook_;
    std::unique_ptr<elim::ScopedParallelHook> scoped_;
public:
    ScopedHostElimHook();
    ~ScopedHostElimHook();
    bool installed() const { return scoped_ != nullptr; }
    ScopedHostElimHook(const ScopedHostElimHook&) = delete;
    ScopedHostElimHook& operator=(const ScopedHostElimHook&) = delete;
};

// ---- the program cache protocol --------------------------------------------
struct CachedRun {
    RunResult result;
    bool hit = false;        // replayed a cached compiled program
    bool inserted = false;   // compiled fresh and inserted
};

template<class Elt>
CachedRun run_cached(Executor<Elt>& ex, ShapeKey key, Program<Elt>&& fresh, const Bindings<Elt>& b) {
    CachedRun out;
    // Programs compiled under different pass options never share a slot.
    // Shifted by 55, not 56 (slot-10 WP-8): the fingerprint is now 9 bits
    // (0..511), so 56 would truncate bit 256 off the top of a 64-bit `extra`.
    // Every consumer's own `extra` value is a tiny integer (q, route: see
    // make_key()'s call sites), so bits 0..54 stay free for it.
    key.extra ^= pass_options_fingerprint(consumer_pass_options()) << 55;
    // The PRIME is part of the key: pass 1 folds coefficient products
    // (ka1*ka2, kb1*kb2, ...) at compile time under the recording modulus, so
    // a program compiled at ell is not a program for ell' even when every
    // recorded coefficient coincides.  (Plan D4's "every ell at fixed shape"
    // is therefore served by one compile per ell -- milliseconds -- not by a
    // cross-prime replay.)
    key.pattern_hash = fnv_mix(key.pattern_hash, static_cast<uint64_t>(fresh.ctx.p));
    struct TagScope {   // (slot-23 P7) ENGINE_TRACE_VRAM attribution; diagnostic only
        const char* prev;
        explicit TagScope(const char* t) : prev(cuda::trace_tag()) { cuda::set_trace_tag(t); }
        ~TagScope() { cuda::set_trace_tag(prev); }
    } tag_scope(key.tag.c_str());
    ProgramCache<Elt>& cache = ProgramCache<Elt>::global();
    const bool bypass = program_cache_bypassed();
    if (!bypass) {
        if (std::shared_ptr<const Program<Elt>> hit = cache.find(key)) {
            ENGINE_CHECK_MSG(hit->structural_hash == fresh.structural_hash,
                             "run_cached: the cached program's recorded structure differs from the fresh "
                             "recording under the same ShapeKey (stale-program hazard, plan D4 caveat)");
            ENGINE_CHECK_MSG(hit->n_inputs == b.inputs.size() && hit->n_outputs == b.outputs.size(),
                             "run_cached: binding count differs from the cached program's slots");
            // The key deliberately omits the prime ("every ell at fixed shape",
            // D4): the replay runs under the FRESH program's modulus -- the
            // executor checks the regime and the k_bound <= k_max contract.
            Bindings<Elt> bb = b;
            bb.set_ctx(fresh.ctx);
            note_program_run(hit->stats.gpu_nodes);
            out.result = ex.run(*hit, bb);
            out.hit = true;
            return out;
        }
    }
    compile(fresh, consumer_pass_options());
    note_program_run(fresh.stats.gpu_nodes);
    out.result = ex.run(static_cast<const Program<Elt>&>(fresh), b);
    if (!bypass) { cache.insert(key, std::move(fresh)); out.inserted = true; }
    return out;
}

// A ShapeKey pre-filled with the parts every consumer key shares.
template<class Elt>
ShapeKey make_key(const char* tag, const ModCtx<Elt>& ctx, uint32_t d_V, uint64_t extra) {
    ShapeKey k;
    k.tag = tag;
    k.d_V = d_V;
    k.regime = static_cast<uint8_t>(ctx.regime);
    k.elt_width = static_cast<uint8_t>(sizeof(Elt));
    k.extra = extra;
    return k;
}

} // namespace tape
} // namespace engine
