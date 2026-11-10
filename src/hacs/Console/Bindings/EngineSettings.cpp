// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// EngineSettings.cpp -- console phase C2a (hecke-extensions slot 6, WP-13):
// the ENGINE KNOBS as `set` keys.
//
// Plan docs/console/00-console-interface-plan.md v3.4 Sec. 2 C4, table "Engine
// knobs", and Sec. 10 C2 (f).  None of these keys can change a computed value
// -- they change how fast it is produced -- so every one is registered with
// SettingFamily::Engine and none enters a hash input or the fingerprint.
//
// THE MECHANISM IS THE ENGINE'S OWN, NEVER A SECOND ONE ([P] Sec. 2 C4, "Why
// not setenv + re-read?"): every key is applied through exactly the in-process
// setter the validation arms already use --
//   threads          engine::tape::set_consumer_pool_override (thread-local)
//   backend          engine::elim::ScopedLegacy (depth-counted RAII, held while
//                    `legacy`) + EquivariantCohomologyOptions::
//                    set_legacy_cohomology_path_forced + HeckeOperatorOptions::
//                    set_legacy_hecke_operator_path_forced (thread-local)
//   device, batch, ksplit, arena, cse, fuse, layout
//                    engine::tape::set_pass_options_override (process-global)
//   lean             EquivariantCohomologyOptions::set_retain_intermediates(!lean)
//                    + HeckeOperatorOptions::set_theta_eq_certification(!lean)
//                    -- exactly what `--bench --lean` scopes (BenchDriver.cpp)
//   program_cache    engine::tape::set_program_cache_bypassed(!on)
// and the thread-local ones reach run_concurrently() task threads through the
// captures their owners register (EngineRuntime.cpp, EquivariantCohomologyOnK
// .cpp, HeckeOperatorOnK.cpp), so a handle's snapshot follows its computation
// onto the host fan-out.
//
// DEFAULTS ARE WHAT IS IN FORCE WHEN THE SESSION STARTS -- the environment's
// values (ENGINE_THREADS, ENGINE_BACKEND, ENGINE_DEVICE, ENGINE_BATCH, ...), read
// through the engine's own accessors, never re-parsed here.  `reset_settings()`
// returns every getter to them.
//
// DIRTY TRACKING, AND WHY.  Settings::install() re-applies EVERY value around
// every handle's compute() (no RAII stack, audit amendment A2).  A knob the
// session never moved must therefore be a strict no-op on install: it may not
// clobber an override the HOST installed (a validation running the console
// under its own ScopedConsumerPool, say).  So each family captures the
// ambient state the first time the session moves it off its default, and
// applying the default again restores exactly that ambient state.
//
// ENV-ONLY knobs ([P] Sec. 2 C4, last table row) are registered with
// SettingSource::EnvOnly: `settings()` shows their values (`source: env`) and
// `set` on one is an error naming the variable.  They are read once into
// statics by the engine, so there is no in-process override to bind.
// ===========================================================================

#include "Bindings.h"

#include "../Core/Confirm.h"   // (NBCONFIRM S3) RegisterConfirmSettings
#include "../Core/ConsoleError.h"   // (S24 CL-L2) NotImplementedError: `backend = "legacy"` in an OFF build
#include "../Core/Settings.h"

#include "../../EquivariantCohomologyOnK.h"
#include "../../Hecke/HeckeOperatorOnK.h"
#include "../../../Engine/EngineRuntime.h"
#include "../../../Engine/Program.h"
#include "../../../Engine/ThreadPool.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/EngineConfig.h"
#include "../../../ModArithmetic/LegacyConfig.h"

#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace console {
namespace bind {

namespace {

// One pool per worker count, created on first use and kept for the process:
// the pool override is a raw pointer, and a handle computed under `threads = 3`
// may still be referenced by a program-cache entry after the session moves on.
engine::tape::ThreadPool* PoolFor(uint32_t workers) {
    static std::mutex m;
    static std::map<uint32_t, std::unique_ptr<engine::tape::ThreadPool>> pools;
    std::lock_guard<std::mutex> lk(m);
    std::unique_ptr<engine::tape::ThreadPool>& slot = pools[workers];
    if (!slot) slot.reset(new engine::tape::ThreadPool(workers));
    return slot.get();
}

bool SamePassOptions(const engine::tape::PassOptions& a, const engine::tape::PassOptions& b) {
    return a.fold == b.fold && a.cse == b.cse && a.fuse == b.fuse && a.layout == b.layout &&
           a.ksplit == b.ksplit && a.batch == b.batch && a.arena == b.arena &&
           a.device_policy == b.device_policy && a.device_model == b.device_model &&
           a.policy == b.policy;
}

const char* DeviceText(engine::tape::DevicePolicy p) {
    switch (p) {
        case engine::tape::DevicePolicy::Gpu:    return "gpu";
        case engine::tape::DevicePolicy::Hybrid: return "hybrid";
        case engine::tape::DevicePolicy::Cpu:    return "cpu";
    }
    return "cpu";
}

void SetDevice(engine::tape::PassOptions& o, const std::string& d) {
    // The PassOptions::from_env() mapping (Passes.cpp), applied to a value
    // instead of to ENGINE_DEVICE.
    if (d == "gpu") {
        o.device_policy = engine::tape::Device::GPU;
        o.device_model = true;
        o.policy = engine::tape::DevicePolicy::Gpu;
    } else if (d == "hybrid") {
        o.device_policy = engine::tape::Device::GPU;
        o.device_model = true;
        o.policy = engine::tape::DevicePolicy::Hybrid;
    } else {
        o.device_policy = engine::tape::Device::CPU;
        o.device_model = false;
        o.policy = engine::tape::DevicePolicy::Cpu;
    }
}

// The per-session state of every settable knob family (see the header note on
// dirty tracking).  Shared by the apply lambdas of one Settings store.
struct EngineKnobState {
    // threads
    uint32_t threads_default = 1;
    bool pool_dirty = false;
    engine::tape::ThreadPool* pool_ambient = nullptr;

    // the pass options: `base` is what was in force at session start, `want`
    // what the session's values ask for.
    engine::tape::PassOptions opts_base;
    engine::tape::PassOptions opts_want;
    bool opts_dirty = false;
    bool opts_ambient_overridden = false;
    engine::tape::PassOptions opts_ambient;

    // backend
#if OC_ENABLE_LEGACY
    bool backend_env_legacy = false;
    std::unique_ptr<engine::elim::ScopedLegacy> legacy_guard;
    bool legacy_dirty = false;
    bool legacy_coh_ambient = false;
    bool legacy_hecke_ambient = false;
#endif

    // lean
    bool lean_default = false;
    bool lean_dirty = false;
    bool retain_ambient = true;
    bool theta_ambient = true;

    // program cache
    bool cache_default = true;       // true = the cache is in use
    bool cache_dirty = false;
    bool cache_ambient_bypass = false;

    void apply_pass_options() {
        if (SamePassOptions(opts_want, opts_base)) {
            if (opts_dirty) {
                engine::tape::set_pass_options_override(opts_ambient_overridden ? &opts_ambient
                                                                                : nullptr);
                opts_dirty = false;
            }
            return;
        }
        if (!opts_dirty) {
            opts_ambient_overridden = engine::tape::pass_options_overridden();
            opts_ambient = engine::tape::consumer_pass_options();
            opts_dirty = true;
        }
        engine::tape::set_pass_options_override(&opts_want);
    }
};

SettingSpec EngineSpec(const std::string& name, SettingKind kind, Value def,
                       const std::string& env_var, const std::string& doc) {
    SettingSpec s;
    s.name = name;
    s.kind = kind;
    s.default_value = std::move(def);
    s.source = SettingSource::Session;
    s.family = SettingFamily::Engine;
    s.env_var = env_var;
    s.doc = doc;
    return s;
}

void InstallEngineSettings(Settings& settings) {
    auto st = std::make_shared<EngineKnobState>();

    // --- threads -------------------------------------------------------------
    st->threads_default = engine::tape::consumer_pool().size();
    {
        SettingSpec s = EngineSpec(
            "threads", SettingKind::Int,
            Value::from_int(static_cast<std::int64_t>(st->threads_default)), "ENGINE_THREADS",
            "the engine consumer pool's worker count (engine::tape::consumer_pool().size()); "
            "thread-local, propagated onto run_concurrently tasks");
        s.validate = [](const Value& v) -> std::string {
            if (v.int_value() < 1 || v.int_value() > 256) return "must be in 1..256";
            return std::string();
        };
        s.apply = [st](const Value& v) {
            if (!v.is_int()) return;
            const uint32_t n = static_cast<uint32_t>(v.int_value());
            if (n == st->threads_default) {
                if (st->pool_dirty) {
                    engine::tape::set_consumer_pool_override(st->pool_ambient);
                    st->pool_dirty = false;
                }
                return;
            }
            if (!st->pool_dirty) {
                st->pool_ambient = engine::tape::consumer_pool_override();
                st->pool_dirty = true;
            }
            engine::tape::set_consumer_pool_override(PoolFor(n));
        };
        settings.register_setting(std::move(s));
    }

    // --- backend -------------------------------------------------------------
    // (S24 CL-L2) One help text in both builds (console.md and the CLI manual are generated from it).  An
    // OC_ENABLE_LEGACY=OFF build keeps the setting and its two legal values, but has no legacy route: `legacy`
    // is REFUSED as not implemented in this build (exit 5; CLEANUP plan Sec. 2.4), never ignored.
    static const char* const kBackendHelp =
        "engine | legacy -- the Phase 2/3 rollback; `legacy` holds engine::elim::ScopedLegacy "
        "and forces the legacy cohomology and Hecke-operator arms (thread-local; OC_ENABLE_LEGACY=ON "
        "builds only -- an OFF build refuses `legacy`, exit 5)";
#if OC_ENABLE_LEGACY
    st->backend_env_legacy = engine::backend_is_legacy();
    {
        const bool legacy_now = engine::elim::use_legacy();
        SettingSpec s = EngineSpec(
            "backend", SettingKind::String, Value::from_string(legacy_now ? "legacy" : "engine"),
            "ENGINE_BACKEND",
            kBackendHelp);
        s.allowed = {"engine", "legacy"};
        s.validate = [st](const Value& v) -> std::string {
            if (st->backend_env_legacy && v.string_value() == "engine") {
                return "ENGINE_BACKEND=legacy is read once, process-wide "
                       "(engine::backend_is_legacy) and cannot be lifted in-process";
            }
            return std::string();
        };
        s.apply = [st](const Value& v) {
            if (!v.is_string()) return;
            const bool want_legacy = (v.string_value() == "legacy");
            if (!want_legacy || st->backend_env_legacy) {
                // `engine`, or `legacy` under ENGINE_BACKEND=legacy (where the
                // static already forces it): release whatever this session holds.
                if (st->legacy_dirty) {
                    st->legacy_guard.reset();
                    EquivariantCohomologyOptions::set_legacy_cohomology_path_forced(
                        st->legacy_coh_ambient);
                    hecke::HeckeOperatorOptions::set_legacy_hecke_operator_path_forced(
                        st->legacy_hecke_ambient);
                    st->legacy_dirty = false;
                }
                return;
            }
            if (st->legacy_dirty) return;   // already held
            st->legacy_coh_ambient = EquivariantCohomologyOptions::legacy_cohomology_path_forced();
            st->legacy_hecke_ambient =
                hecke::HeckeOperatorOptions::legacy_hecke_operator_path_forced();
            st->legacy_guard.reset(new engine::elim::ScopedLegacy());
            EquivariantCohomologyOptions::set_legacy_cohomology_path_forced(true);
            hecke::HeckeOperatorOptions::set_legacy_hecke_operator_path_forced(true);
            st->legacy_dirty = true;
        };
        settings.register_setting(std::move(s));
    }

#else
    {
        SettingSpec s = EngineSpec("backend", SettingKind::String, Value::from_string("engine"), "ENGINE_BACKEND",
                                   kBackendHelp);
        s.allowed = {"engine", "legacy"};
        s.validate = [](const Value& v) -> std::string {
            if (v.is_string() && v.string_value() == "legacy") {
                throw NotImplementedError("setting 'backend': " + oc::legacy::compiled_out_text("backend = \"legacy\""),
                                          SourcePos());
            }
            return std::string();
        };
        settings.register_setting(std::move(s));
    }
#endif

    // --- the pass options ------------------------------------------------------
    st->opts_base = engine::tape::consumer_pass_options();
    st->opts_want = st->opts_base;
    {
        SettingSpec s = EngineSpec(
            "device", SettingKind::String,
            Value::from_string(DeviceText(st->opts_base.effective_policy())), "ENGINE_DEVICE",
            "cpu | gpu | hybrid -- pass 6's device policy (PassOptions); process-global.  "
            "`cpu` is the golden path; the GPU is never the default");
        s.allowed = {"cpu", "gpu", "hybrid"};
        s.apply = [st](const Value& v) {
            if (!v.is_string()) return;
            SetDevice(st->opts_want, v.string_value());
            st->apply_pass_options();
        };
        settings.register_setting(std::move(s));
    }
    struct BoolPass {
        const char* name;
        const char* env;
        bool engine::tape::PassOptions::*field;
        const char* doc;
    };
    const BoolPass bool_passes[] = {
        {"batch", "ENGINE_BATCH", &engine::tape::PassOptions::batch, "pass 5, batching"},
        {"ksplit", "ENGINE_KSPLIT", &engine::tape::PassOptions::ksplit,
         "pass 4, k-splitting of GEMMs with k > k_max"},
        {"arena", "ENGINE_ARENA", &engine::tape::PassOptions::arena,
         "the temp arena (false: every Temp owns its storage)"},
        {"cse", "ENGINE_CSE", &engine::tape::PassOptions::cse, "pass 2, common subexpressions"},
        {"fuse", "ENGINE_FUSE", &engine::tape::PassOptions::fuse,
         "pass 2b, product-into-AXPY fusion"},
        {"layout", "ENGINE_LAYOUT", &engine::tape::PassOptions::layout, "pass 3, layout"},
    };
    for (const BoolPass& bp : bool_passes) {
        SettingSpec s = EngineSpec(bp.name, SettingKind::Bool,
                                   Value::from_bool(st->opts_base.*(bp.field)), bp.env,
                                   std::string(bp.doc) + " (PassOptions; process-global)");
        bool engine::tape::PassOptions::*field = bp.field;
        s.apply = [st, field](const Value& v) {
            if (!v.is_bool()) return;
            st->opts_want.*field = v.bool_value();
            st->apply_pass_options();
        };
        settings.register_setting(std::move(s));
    }

    // --- lean ------------------------------------------------------------------
    st->lean_default = !EquivariantCohomologyOptions::retain_intermediates() &&
                       !hecke::HeckeOperatorOptions::theta_eq_certification_enabled();
    {
        SettingSpec s = EngineSpec(
            "lean", SettingKind::Bool, Value::from_bool(st->lean_default), "",
            "the memory-lean configuration of `--bench --lean`: drop the cohomology objects no "
            "production consumer reads and skip the Theta_eq solve (A^(q) unchanged); "
            "thread-local, captured at a handle's CREATION like every setting");
        s.apply = [st](const Value& v) {
            if (!v.is_bool()) return;
            if (v.bool_value() == st->lean_default) {
                if (st->lean_dirty) {
                    EquivariantCohomologyOptions::set_retain_intermediates(st->retain_ambient);
                    hecke::HeckeOperatorOptions::set_theta_eq_certification(st->theta_ambient);
                    st->lean_dirty = false;
                }
                return;
            }
            if (!st->lean_dirty) {
                st->retain_ambient = EquivariantCohomologyOptions::retain_intermediates();
                st->theta_ambient = hecke::HeckeOperatorOptions::theta_eq_certification_enabled();
                st->lean_dirty = true;
            }
            EquivariantCohomologyOptions::set_retain_intermediates(!v.bool_value());
            hecke::HeckeOperatorOptions::set_theta_eq_certification(!v.bool_value());
        };
        settings.register_setting(std::move(s));
    }

    // --- program cache -----------------------------------------------------------
    st->cache_default = !engine::tape::program_cache_bypassed();
    {
        SettingSpec s = EngineSpec(
            "program_cache", SettingKind::Bool, Value::from_bool(st->cache_default), "",
            "whether compiled tape programs are served from the program cache (false: a fresh "
            "compile every time); process-global");
        s.apply = [st](const Value& v) {
            if (!v.is_bool()) return;
            if (v.bool_value() == st->cache_default) {
                if (st->cache_dirty) {
                    engine::tape::set_program_cache_bypassed(st->cache_ambient_bypass);
                    st->cache_dirty = false;
                }
                return;
            }
            if (!st->cache_dirty) {
                st->cache_ambient_bypass = engine::tape::program_cache_bypassed();
                st->cache_dirty = true;
            }
            engine::tape::set_program_cache_bypassed(!v.bool_value());
        };
        settings.register_setting(std::move(s));
    }

    // --- (NBCONFIRM S3) confirm, confirm_wall_s ----------------------------------
    // Non-mathematical presentation / policy keys of the cost confirmation (Console/Core/Confirm.h): never in a record, a hash
    // input or the fingerprint, and no engine state behind them (no apply hook).  ConsoleSettingsValidation (g)'s count of the
    // eleven MATHEMATICAL keys does not move.
    RegisterConfirmSettings(settings);

    // --- env-only ------------------------------------------------------------------
    struct EnvOnly {
        const char* name;
        const char* env;
    };
    const EnvOnly env_only[] = {
        {"host_threads", "ENGINE_HOST_THREADS"},
        {"tile_mult", "ENGINE_TILE_MULT"},
        {"tile_cols", "ENGINE_TILE_COLS"},
        {"spin", "ENGINE_SPIN"},
        {"elim_base", "ENGINE_ELIM_BASE"},
        {"cuda_disable", "ENGINE_CUDA_DISABLE"},
        {"cuda_device", "ENGINE_CUDA_DEVICE"},
        {"cuda_program_cache", "ENGINE_CUDA_PROGRAM_CACHE"},
        {"cuda_graph", "ENGINE_CUDA_GRAPH"},
        {"cuda_merge_levels", "ENGINE_CUDA_MERGE_LEVELS"},
        {"cuda_pinned_mb", "ENGINE_CUDA_PINNED_MB"},
        {"cuda_copy_threads", "ENGINE_CUDA_COPY_THREADS"},
        {"trace_nodes", "ENGINE_TRACE_NODES"},
        {"trace_rss", "ENGINE_TRACE_RSS"},
        {"trace_elim", "ENGINE_TRACE_ELIM"},
        {"trace_levels", "ENGINE_TRACE_LEVELS"},
        {"trace_passes", "ENGINE_TRACE_PASSES"},
    };
    for (const EnvOnly& e : env_only) {
        const char* raw = std::getenv(e.env);
        SettingSpec s = EngineSpec(
            e.name, SettingKind::String, Value::from_string(raw != nullptr ? raw : ""), e.env,
            std::string("environment-only: read once by the engine from ") + e.env +
                " (empty = unset); there is no in-process override");
        s.source = SettingSource::EnvOnly;
        settings.register_setting(std::move(s));
    }
}

} // namespace

void RegisterEngineSettingsBindings(Registry& registry) {
    registry.add_settings_installer(&InstallEngineSettings);
}

} // namespace bind
} // namespace console
