// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleSettingsValidation.cpp -- see ConsoleSettingsValidation.h.
//
// Compiled into cohomology_app (like ConsoleValidation.cpp), never into
// ConsoleCore: it reads engine getters, builds matrices and names hecke types.
//
// HOW A GETTER IS OBSERVED.  The console's own `settings()` only says what the
// SESSION stored; the claim of (f) is that the ENGINE saw it.  So the fixture
// registers test-only verbs on its OWN registry that read the engine's
// accessors directly (`__engine()`), from run_concurrently task threads
// (`__concurrent()`), and from inside a lazy handle's compute (`__lean_probe()`).
// None of them is registered on the shipped registry.
// ===========================================================================

#include "ConsoleSettingsValidation.h"
#include "../../ModArithmetic/LegacyConfig.h"   // (S24 CL-L2) RunLegacyArm, kLegacyBuild

#include "Bindings/Bindings.h"
#include "Bindings/RegisterPipelineVerbs.h"
#include "Core/ConsoleError.h"
#include "Core/Object.h"
#include "Core/Registry.h"
#include "Core/ScriptRunner.h"
#include "Core/Session.h"
#include "Core/Settings.h"
#include "Core/Value.h"

#include "../EquivariantCohomologyOnK.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeSettings.h"
#include "../HeckeExt/HeckeCanonicalField.h"
#include "../HeckeExt/RegisterHeckeExtensionVerbs.h"
#include "../../Engine/EngineRuntime.h"
#include "../../Engine/ThreadPool.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/EngineConfig.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/TextIo.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace console;

// --- check harness (the ConsoleValidation shape) ---------------------------

std::uint32_t g_checks = 0;
std::uint32_t g_failures = 0;

void Check(bool ok, const std::string& label) {
    ++g_checks;
    if (ok) return;
    ++g_failures;
    if (g_failures <= 60) {
        std::cout << "  [FAIL] " << label << std::endl;
    } else if (g_failures == 61) {
        std::cout << "  [FAIL] (further failures suppressed)" << std::endl;
    }
}

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

using oc_util::HexU64;   // S24 CL-7 (F11): Util/TextIo.h
// --- the test-only probes ----------------------------------------------------

const char* DeviceText(engine::tape::DevicePolicy p) {
    switch (p) {
        case engine::tape::DevicePolicy::Gpu:    return "gpu";
        case engine::tape::DevicePolicy::Hybrid: return "hybrid";
        case engine::tape::DevicePolicy::Cpu:    return "cpu";
    }
    return "cpu";
}

// Every engine getter a knob documents, read on THIS thread.
Record EngineGetters() {
    const engine::tape::PassOptions o = engine::tape::consumer_pass_options();
    Record r;
    r.set("threads", Value::from_int(engine::tape::consumer_pool().size()));
#if OC_ENABLE_LEGACY
    r.set("legacy", Value::from_bool(engine::elim::use_legacy()));
    r.set("legacy_cohomology",
          Value::from_bool(EquivariantCohomologyOptions::legacy_cohomology_path_forced()));
    r.set("legacy_hecke",
          Value::from_bool(hecke::HeckeOperatorOptions::legacy_hecke_operator_path_forced()));
#else
    // (S24 CL-L2) an OFF build has no legacy route: the three getters are constant false (same record shape).
    r.set("legacy", Value::from_bool(false));
    r.set("legacy_cohomology", Value::from_bool(false));
    r.set("legacy_hecke", Value::from_bool(false));
#endif
    r.set("device", Value::from_string(DeviceText(o.effective_policy())));
    r.set("batch", Value::from_bool(o.batch));
    r.set("ksplit", Value::from_bool(o.ksplit));
    r.set("arena", Value::from_bool(o.arena));
    r.set("cse", Value::from_bool(o.cse));
    r.set("fuse", Value::from_bool(o.fuse));
    r.set("layout", Value::from_bool(o.layout));
    r.set("retain", Value::from_bool(EquivariantCohomologyOptions::retain_intermediates()));
    r.set("theta_eq",
          Value::from_bool(hecke::HeckeOperatorOptions::theta_eq_certification_enabled()));
    r.set("program_cache", Value::from_bool(!engine::tape::program_cache_bypassed()));
    r.set("locality", Value::from_string(hecke::math_settings().locality.to_text()));
    r.set("fingerprint", Value::from_string(HexU64(hecke::math_settings().fingerprint())));
    return r;
}

struct LeanProbe {
    bool retained = true;
    bool theta_eq = true;
};

const TypeInfo* g_lean_type = nullptr;

void RegisterProbes(Registry& reg) {
    reg.register_verb(
        "__engine", bind::Sig({}),
        [](CallContext&, const std::vector<Value>&) { return Value::from_record(EngineGetters()); },
        "test-only: the engine getters on this thread", "test");

    reg.register_verb(
        "__concurrent", bind::Sig({}),
        [](CallContext&, const std::vector<Value>&) {
            // Four tasks through the SHIPPED fan-out; each reads the getters on
            // its own thread.  Task 0 runs on the caller (EngineRuntime.cpp).
            const std::size_t n = 4;
            std::vector<Record> seen(n);
            std::vector<char> other_thread(n, 0);   // not vector<bool>: tasks write concurrently
            const std::thread::id caller = std::this_thread::get_id();
            std::vector<std::function<void()>> tasks;
            for (std::size_t i = 0; i < n; ++i) {
                tasks.push_back([i, caller, &seen, &other_thread]() {
                    seen[i] = EngineGetters();
                    other_thread[i] = (std::this_thread::get_id() != caller);
                });
            }
            engine::tape::run_concurrently(tasks);
            Record out;
            bool fanned = false;
            for (char b : other_thread) fanned = fanned || (b != 0);
            out.set("fanned_out", Value::from_bool(fanned));
            // A key is reported as its value if EVERY task saw the same one,
            // and as "<disagree>" otherwise -- which no check accepts.
            for (const char* key : {"threads", "legacy", "legacy_cohomology", "legacy_hecke",
                                    "retain", "theta_eq", "locality", "fingerprint"}) {
                const Value* first = seen[0].find(key);
                bool same = first != nullptr;
                for (std::size_t i = 1; i < n && same; ++i) {
                    const Value* v = seen[i].find(key);
                    same = v != nullptr && v->equals(*first);
                }
                out.set(key, same ? *first : Value::from_string("<disagree>"));
            }
            return Value::from_record(std::move(out));
        },
        "test-only: the getters as run_concurrently task threads see them", "test");

    TypeInfo t;
    t.name = "LeanProbe";
    t.members.push_back({"retained", true, "retain_intermediates() inside compute()",
                         [](const ObjectPtr& o) {
                             return Value::from_bool(o->payload_as<LeanProbe>()->retained);
                         }});
    t.members.push_back({"theta_eq", true, "theta_eq_certification_enabled() inside compute()",
                         [](const ObjectPtr& o) {
                             return Value::from_bool(o->payload_as<LeanProbe>()->theta_eq);
                         }});
    g_lean_type = reg.register_type(std::move(t));
    reg.register_verb(
        "__lean_probe", bind::Sig({}),
        [](CallContext& ctx, const std::vector<Value>&) {
            auto p = std::make_shared<LeanProbe>();
            ObjectPtr o = Object::create(
                g_lean_type, ctx.pos, ctx.call_text, ctx.session->settings().snapshot(), p,
                [p](Object&) {
                    p->retained = EquivariantCohomologyOptions::retain_intermediates();
                    p->theta_eq = hecke::HeckeOperatorOptions::theta_eq_certification_enabled();
                });
            return Value::from_object(o);
        },
        "test-only: a lazy handle recording the lean getters at COMPUTE time", "test");

    // [E] Claim 3.1's non-local family over F_5, f = 2, EIGHT times over, as a
    // REAL Decomposition handle (Bindings.h's MakeMatrixDecomposition).
    //   C  = companion of pi = X^2 - 2 (irreducible mod 5: 2 is a non-residue),
    //   4C = multiplication by alpha^5 (alpha^5 = alpha * (alpha^2)^2 = 4 alpha),
    //   t1 = diag(C, 4C) -- (alpha, alpha^ell),  t2 = diag(C, C) -- (alpha, alpha).
    // Every operator has the ONE irreducible min-poly pi, and T_S/rad is
    // F_25 x F_25: two maximal ideals.  The witness sequence of
    // HeckeModuleDecomposition.h opens with the m = 8 TRIVIAL witnesses, and a
    // single t_i never certifies (no one P_p works on both summands), so a
    // budget of 8 exhausts them and leaves the block UNDECIDED -- and candidate
    // [8], the all-ones combination 4(t1 + t2) = (3 alpha, 0), splits it.  So
    // `locality = budget=8` is incomplete and `budget=64` completes: the shape
    // [P] Sec. 10 C2 (h) names.
    reg.register_verb(
        "__nonlocal", bind::Sig({}),
        [](CallContext& ctx, const std::vector<Value>&) {
            const uint32_t ell = 5;
            MatrixT<uint32_t> t1(4, 4, ell), t2(4, 4, ell);
            // C = [[0, 2], [1, 0]]  (C^2 = 2 I).
            auto put = [](MatrixT<uint32_t>& M, uint32_t off, uint32_t scale) {
                M.data[off + 0][off + 1] = (2u * scale) % 5u;
                M.data[off + 1][off + 0] = (1u * scale) % 5u;
            };
            put(t1, 0, 1); put(t1, 2, 4);
            put(t2, 0, 1); put(t2, 2, 1);
            std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops;
            const uint64_t primes[] = {2, 3, 5, 7, 11, 13, 17, 19};
            for (std::size_t i = 0; i < 8; ++i) ops.emplace_back(primes[i], (i % 2 == 0) ? t1 : t2);
            return Value::from_object(bind::MakeMatrixDecomposition(ctx, std::move(ops), ell));
        },
        "test-only: [E] Claim 3.1's non-local family, as a real Decomposition", "test");
}

// --- the fixture -------------------------------------------------------------

struct Fixture {
    Registry registry;
    std::ostringstream out;
    Session session;

    Fixture() : session(registry, /*verbose_default=*/false) {
        RegisterPipelineVerbs(registry);
        hecke_ext::RegisterHeckeExtensionVerbs(registry);
        RegisterProbes(registry);
        // The Session is a MEMBER, constructed before the bindings registered
        // their installers (ConsoleValidation.cpp's Fixture says why), so the
        // installers are re-run here and the baseline re-taken.
        registry.install_settings(session.settings());
        session.settings().set_baseline(session.settings().snapshot());
        session.set_output(out);
    }
};

struct Outcome {
    int code = 0;
    std::string text;
};

Outcome Run(Fixture& fx, const std::string& src) {
    Outcome o;
    o.code = RunSource(fx.session, src, "<settings>");
    o.text = fx.out.str();
    fx.out.str(std::string());
    fx.out.clear();
    return o;
}

// A script whose every claim is an `assert`: exit 0 iff all hold.
void Expect0(Fixture& fx, const std::string& src, const std::string& label) {
    const Outcome o = Run(fx, src);
    Check(o.code == 0, label + (o.code == 0 ? std::string() : "  [exit " +
                                std::to_string(o.code) + "]\n" + o.text));
}

void ExpectError(Fixture& fx, const std::string& src, int code, const std::string& needle,
                 const std::string& label) {
    const Outcome o = Run(fx, src);
    const bool ok = o.code == code && Contains(o.text, needle);
    Check(ok, label + (ok ? std::string() : "  [exit " + std::to_string(o.code) + ", want " +
                                                std::to_string(code) + " with '" + needle +
                                                "']\n" + o.text));
}

std::string Quote(const std::string& s) { return "\"" + s + "\""; }

// ===========================================================================
// (f) the engine knobs
// ===========================================================================

void ArmEngineKnobs() {
    std::cout << "-- (f) the engine knobs: getters, env-only, reset_settings(), "
                 "run_concurrently, snapshot-at-creation, session end\n";
    const Record ambient = EngineGetters();
    const bool env_legacy = engine::backend_is_legacy();
    // (S24 CL-L2) the `backend = "legacy"` knob arms need the legacy route (OC_ENABLE_LEGACY=ON); an OFF build
    // checks the refusal instead (exit 5, NotImplementedInThisBuild).
    const bool legacy_knob = !env_legacy && oc::legacy::RunLegacyArm("ConsoleSettingsValidation", "backend-legacy");
    {
        Fixture fx;
        const Settings& s = fx.session.settings();

        // Every ENGINE key is registered with its family and the env default.
        std::uint32_t engine_keys = 0, env_only = 0;
        for (const std::string& name : s.names()) {
            const SettingSpec* spec = s.find_spec(name);
            if (spec == nullptr || spec->family != SettingFamily::Engine) continue;
            if (name == kSettingPrintMaxEntries || name == kSettingVerbose) continue;
            ++engine_keys;
            if (spec->source == SettingSource::EnvOnly) {
                ++env_only;
                // `set` on an env-only key errors, naming the variable.
                ExpectError(fx, "set " + name + " = \"1\"", 3, spec->env_var,
                            "(f) `set " + name + "` is refused, naming " + spec->env_var);
            }
        }
        std::cout << "   engine keys registered: " << engine_keys << " (" << env_only
                  << " env-only)\n";
        Check(engine_keys > env_only && env_only > 0,
              "(f) settable and env-only engine keys are both registered");

    }
    {
        Fixture fx;
        Expect0(fx, "ENGINE0 = __engine();", "(f) the ambient getters are readable");
        // The settable keys, each against its documented getter.  `alt` is a
        // legal NON-default value, `getter` the __engine() field it must move.
        const std::int64_t d_threads = ambient.find("threads")->int_value();
        const std::int64_t alt_threads = (d_threads == 3) ? 2 : 3;
        struct Knob {
            std::string key, alt, getter, want;
        };
        std::vector<Knob> knobs = {
            {"threads", std::to_string(alt_threads), "threads", std::to_string(alt_threads)},
            {"device", Quote("hybrid"), "device", Quote("hybrid")},
            {"lean", "true", "retain", "false"},
            {"lean", "true", "theta_eq", "false"},
            {"program_cache", "false", "program_cache", "false"},
        };
        if (legacy_knob) {
            knobs.push_back({"backend", Quote("legacy"), "legacy", "true"});
            knobs.push_back({"backend", Quote("legacy"), "legacy_cohomology", "true"});
            knobs.push_back({"backend", Quote("legacy"), "legacy_hecke", "true"});
        }
        for (const char* b : {"batch", "ksplit", "arena", "cse", "fuse", "layout"}) {
            const bool def = ambient.find(b)->bool_value();
            knobs.push_back({b, def ? "false" : "true", b, def ? "false" : "true"});
        }
        for (const Knob& k : knobs) {
            Expect0(fx,
                    "set " + k.key + " = " + k.alt + ";\n"
                    "assert(__engine()." + k.getter + " == " + k.want + ");\n"
                    "assert(settings()." + k.key + ".family == \"engine\");\n"
                    "reset_settings();\n"
                    "assert(__engine() == ENGINE0);\n",
                    "(f) `set " + k.key + " = " + k.alt + "` moves " + k.getter +
                        ", and reset_settings() restores every getter");
        }
        // A kind or a range error is the setting's own.
        ExpectError(fx, "set threads = 0", 3, "must be in 1..256",
                    "(f) threads = 0 is refused");
        ExpectError(fx, "set device = \"tpu\"", 3, "accepts only cpu | gpu | hybrid",
                    "(f) an unknown device is refused");
        ExpectError(fx, "set lean = 1", 3, "expects Bool", "(f) lean takes a Bool");
        if (!oc::legacy::kLegacyBuild) {
            // (S24 CL-L2, CLEANUP plan Sec. 2.4) the compiled-out route is refused, never ignored.
            ExpectError(fx, "set backend = \"legacy\"", 5, "compiled out (OC_ENABLE_LEGACY=OFF)",
                        "(f) backend = \"legacy\" is refused in an OFF build (exit 5)");
            ExpectError(fx, "micro(\"gemm\", 64, backend = \"legacy\")", 5, "compiled out (OC_ENABLE_LEGACY=OFF)",
                        "(f) micro(..., backend = \"legacy\") is refused in an OFF build (exit 5)");
        } else if (legacy_knob) {
            // (S24 CL-L2) moved from 022_harness.oc: the micro checksum is a bit-identity statement between backends.
            Expect0(fx,
                    "ME = micro(\"gemm\", 64, backend = \"engine\");\n"
                    "ML = micro(\"gemm\", 64, backend = \"legacy\");\n"
                    "assert(ME.checksum == ML.checksum, \"micro gemm 64: the engine and legacy checksums agree\");\n",
                    "(f) micro gemm 64: the engine and legacy checksums agree (moved from 022_harness.oc)");
        }

        // Interleave three keys, then reset_settings(): every getter returns
        // to its env default.
        Expect0(fx,
                "set threads = " + std::to_string(alt_threads) + ";\n"
                "set lean = true;\n"
                "set program_cache = false;\n"
                "set threads = " + std::to_string(alt_threads == 3 ? 4 : 3) + ";\n"
                "set lean = false;\n"
                "set lean = true;\n"
                "assert(__engine() != ENGINE0);\n"
                "reset_settings();\n"
                "assert(__engine() == ENGINE0);\n",
                "(f) three keys interleaved, then reset_settings(): every getter is back at "
                "its env default");

        // run_concurrently: the session's engine AND mathematical settings are
        // what every task thread sees.
        std::string conc =
            "set threads = 4;\n"
            "set lean = true;\n"
            "set locality = \"budget=8\";\n";
        if (legacy_knob) conc += "set backend = \"legacy\";\n";
        conc +=
            "C = __concurrent();\n"
            "assert(C.threads == 4);\n"
            "assert(C.retain == false);\n"
            "assert(C.theta_eq == false);\n"
            "assert(C.locality == \"budget=8\");\n"
            "assert(C.fingerprint != \"0x0000000000000000\");\n";
        if (legacy_knob) {
            conc += "assert(C.legacy == true);\nassert(C.legacy_cohomology == true);\n"
                    "assert(C.legacy_hecke == true);\n";
        }
        conc += "FANNED = C.fanned_out;\nreset_settings();\nassert(__engine() == ENGINE0);\n";
        Expect0(fx, conc,
                "(f)(g) threads / lean / backend / locality set in the session are what every "
                "run_concurrently task thread sees");
        {
            const Outcome o = Run(fx, "print(FANNED);");
            const char* host = std::getenv("ENGINE_HOST_THREADS");
            const bool fanout_off = host != nullptr && std::string(host) == "0";
            std::cout << "   run_concurrently fanned out onto task threads: " << o.text;
            Check(fanout_off || Contains(o.text, "true"),
                  "(f) the probe really ran on task threads (unless ENGINE_HOST_THREADS=0)");
        }

        // Snapshot-at-creation: a handle created under lean = true computes
        // lean after `set lean = false`, and one created after computes full.
        Expect0(fx,
                "set lean = true;\n"
                "P = __lean_probe();\n"
                "set lean = false;\n"
                "Q = __lean_probe();\n"
                "assert(P.retained == false);\n"
                "assert(P.theta_eq == false);\n"
                "assert(Q.retained == true);\n"
                "assert(state(P).settings.lean == true);\n"
                "assert(__engine() == ENGINE0);\n",
                "(f) a handle created under lean = true computes lean after set lean = false "
                "(and the session getter is unchanged afterwards)");

        // Session end: leave three knobs moved, and let the fixture die.
        Expect0(fx,
                "set threads = " + std::to_string(alt_threads) + ";\n"
                "set lean = true;\n"
                "set program_cache = false;\n" +
                    std::string(legacy_knob ? "set backend = \"legacy\";\n" : "") +
                    "set locality = \"strict\";\n",
                "(f) knobs moved before the session ends");
    }
    const Record after = EngineGetters();
#if OC_ENABLE_LEGACY
    const bool legacy_released = !engine::elim::legacy_forced();
#else
    const bool legacy_released = true;   // (S24 CL-L2) no legacy guard exists in an OFF build
#endif
    Check(Value::from_record(after).equals(Value::from_record(ambient)) && engine::tape::consumer_pool_override() == nullptr &&
              legacy_released,
          "(f) a session that ends restores every knob it moved (engine and mathematical)");
}

// ===========================================================================
// (g) the mathematical settings
// ===========================================================================

// [X] Sec. 1.5's table, row by row, with each row's consuming phase -- the
// RECONCILIATION the plan asks for.  It is read, never counted: a key the
// registry has and this table lacks is a failure; a row not yet registered is
// printed with its phase.
struct PlanRow {
    const char* key;
    const char* phase;
};
const PlanRow kPlanRows[] = {
    {"final_mode", "E2 (is_final)"},
    {"sturm_unavailable", "E1b (SturmCertificate)"},
    {"sturm", "E1s (HeckeSturmBound)"},
    {"commutant_budget", "E1a / E2a (CommutantRoute)"},
    {"locality", "E0b (the refinement loop)"},
    {"field_polynomial", "E6 (HeckeCanonicalField)"},
    {"split_field_report", "E6 (the report layer)"},
    {"coefficient_ring", "E8 (the whole pipeline)"},
    {"predictions", "E6 (HeckePredictions)"},
    {"presentation", "E2 (HeckePresentation)"},
    {"u_ell", "E1b (Stage 1)"},
    {"integral_fallback", "E5 (the fills of tau / eta)"},
    {"q_max", "E9"},
};

// One legal NON-default value per registered key.  Test data, not a rule: the
// rule is HeckeSettings', and the arm below asserts that HeckeSettings accepts
// each value and renders it back canonically.  A key registered later without
// a row here FAILS, so the round trip cannot silently skip it.
struct Sample {
    const char* key;
    const char* value;
};
const Sample kSamples[] = {
    {"locality", "budget=8"},
    {"u_ell", "off"},
    {"sturm", "off"},
    {"sturm_unavailable", "heuristic"},
    {"commutant_budget", "1000"},
    {"final_mode", "sturm_heuristic"},
    {"field_polynomial", "conway"},
    {"split_field_report", "own_field"},
    {"presentation", "on"},        // slot-7 WP-2
    {"predictions", "assert"},     // slot-7 WP-2
    {"coefficient_ring", "Z/ell^2"},   // slot-8 WP-3
};

const char* SampleFor(const std::string& key) {
    for (const Sample& s : kSamples) {
        if (key == s.key) return s.value;
    }
    return nullptr;
}

std::string SourceInJson(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\":{\"value\":\"";
    const std::size_t at = json.find(needle);
    if (at == std::string::npos) return std::string();
    const std::size_t src = json.find("\"source\":\"", at);
    if (src == std::string::npos) return std::string();
    const std::size_t b = src + 10;
    const std::size_t e = json.find('"', b);
    return e == std::string::npos ? std::string() : json.substr(b, e - b);
}

void ArmMathSettings() {
    std::cout << "-- (g) the mathematical settings, through hecke::MathSettings only\n";
    const std::vector<std::string>& keys = hecke::MathSettings::keys();
    std::cout << "   hecke::MathSettings::keys() registers " << keys.size() << ":";
    for (const std::string& k : keys) std::cout << " " << k;
    // (slot-7 WP-2, acceptance (d)) The count, from keys(): stage 1 (2)
    // registered `presentation` and `predictions` after the eight of slots 2-5.
    // (slot-8 WP-3) `coefficient_ring` appended at the tail: eleven keys.
    Check(keys.size() == 11, "(g) hecke::MathSettings::keys() registers 11 keys (got " +
                                 std::to_string(keys.size()) + ")");
    std::cout << "\n   reconciled against [X] Sec. 1.5:\n";
    for (const PlanRow& row : kPlanRows) {
        const bool reg = std::find(keys.begin(), keys.end(), row.key) != keys.end();
        std::cout << "     " << row.key << ": "
                  << (reg ? "registered" : std::string("not yet registered -- arrives with ") +
                                               row.phase)
                  << "\n";
    }
    for (const std::string& k : keys) {
        bool in_plan = false;
        for (const PlanRow& row : kPlanRows) in_plan = in_plan || (k == row.key);
        Check(in_plan, "(g) registered key '" + k + "' is a row of [X] Sec. 1.5");
    }

    Fixture fx;
    Check(hecke::math_settings().is_default(), "(g) a fresh session starts at the defaults");
    Expect0(fx, "assert(fingerprint() == \"0x0000000000000000\");\n",
            "(g) the fingerprint is 0 at the defaults");
    const std::string json0 = hecke::MathSettings::defaults().to_json();
    const std::uint64_t json0_fnv = engine::FnvBytes(engine::kFnvOffset64, json0);

    std::vector<std::uint64_t> fps;
    for (const std::string& key : keys) {
        const std::string def = hecke::MathSettings::defaults().text(key);
        const char* alt = SampleFor(key);
        Check(alt != nullptr, "(g) key '" + key + "' has a sample non-default value in this "
                                                  "validation");
        if (alt == nullptr) continue;
        // The console registered the key, as a MATHEMATICAL setting, with the
        // engine's default text.
        const SettingSpec* spec = fx.session.settings().find_spec(key);
        Check(spec != nullptr && spec->family == SettingFamily::Mathematical &&
                  spec->default_value.is_string() && spec->default_value.string_value() == def,
              "(g) '" + key + "' is a mathematical `set` key with the engine's default '" + def +
                  "'");

        Expect0(fx,
                "set " + key + " = " + Quote(alt) + ";\n"
                "assert(settings()." + key + ".value == " + Quote(alt) + ");\n"
                "assert(settings()." + key + ".family == \"mathematical\");\n"
                "assert(fingerprint() != \"0x0000000000000000\");\n",
                "(g) `set " + key + " = " + alt + "` round-trips through settings()");
        const hecke::MathSettings& now = hecke::math_settings();
        Check(now.text(key) == alt,
              "(g) HeckeSettings renders '" + key + "' back as '" + alt + "' (got '" +
                  now.text(key) + "')");
        Check(SourceInJson(now.to_json(), key) == "console",
              "(g) '" + key + "' carries source = console after `set`");
        const std::uint64_t fp = now.fingerprint();
        Check(fp != 0, "(g) a non-default '" + key + "' moves the fingerprint");
        Check(engine::FnvBytes(engine::kFnvOffset64, now.to_json()) != json0_fnv,
              "(g) a non-default '" + key + "' moves the record's settings block -- the "
              "ext-layer hash input");
        fps.push_back(fp);
        Expect0(fx,
                "set " + key + " = " + Quote(def) + ";\n"
                "assert(fingerprint() == \"0x0000000000000000\");\n"
                "assert(settings()." + key + ".value == " + Quote(def) + ");\n",
                "(g) `set " + key + " = " + def + "` returns to fingerprint 0");
        Check(SourceInJson(hecke::math_settings().to_json(), key) == "default",
              "(g) coming back to the default restores source = default for '" + key + "'");

        // A bad value: the error text IS HeckeSettings' own.
        const std::string bad = "no_such_value_zz";
        std::string engine_text;
        {
            hecke::MathSettings probe = hecke::math_settings();
            try {
                probe.set(key, bad, hecke::SettingSource::Console);
            } catch (const std::invalid_argument& e) {
                engine_text = e.what();
            }
        }
        Check(!engine_text.empty(), "(g) HeckeSettings refuses '" + key + " = " + bad + "'");
        if (!engine_text.empty()) {
            ExpectError(fx, "set " + key + " = " + Quote(bad), 3, engine_text,
                        "(g) the console's refusal of '" + key + " = " + bad +
                            "' carries HeckeSettings' own text");
        }
        Check(hecke::math_settings().is_default(),
              "(g) a refused value leaves the settings at the defaults");
    }
    std::vector<std::uint64_t> sorted = fps;
    std::sort(sorted.begin(), sorted.end());
    Check(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
          "(g) every key's non-default value gives a DIFFERENT fingerprint");

    // An unknown key: HeckeSettings does not know it, and the console refuses.
    Check(!hecke::MathSettings::known_key("no_such_math_key"),
          "(g) HeckeSettings does not know 'no_such_math_key'");
    ExpectError(fx, "set no_such_math_key = \"x\"", 3, "unknown setting 'no_such_math_key'",
                "(g) an unknown key is refused");

    // THE EXIT-5 ARM ([P] Sec. 10 C2 (g), v3.4).
    Expect0(fx,
            "set field_polynomial = \"conway\";\n"
            "assert(settings().field_polynomial.value == \"conway\");\n"
            "assert(fingerprint() != \"0x0000000000000000\");\n",
            "(g) exit-5 arm: `set field_polynomial = \"conway\"` is ACCEPTED -- it parses, "
            "validates and moves the fingerprint");
    ExpectError(fx, "canonical_field_polynomial(5, 2)", 5, "not_implemented_in_this_build",
                "(g) exit-5 arm: a use that reaches HeckeCanonicalField exits 5 with the named "
                "error");
    {
        const Outcome o = Run(fx, "canonical_field_polynomial(5, 2)");
        Check(!Contains(o.text, "unknown field_polynomial") && Contains(o.text, "conway"),
              "(g) exit-5 arm: never a bad-value error, and the message names conway");
    }
    Expect0(fx, "set field_polynomial = \"least\";\n", "(g) back to least");
    {
        const Outcome o = Run(fx, "print(canonical_field_polynomial(5, 2));");
        Check(o.code == 0 && hecke_ext::CanonicalFieldPolynomial(5, 2).degree() == 2,
              "(g) under `least` the consumer answers (exit 0)");
        Check(hecke::math_settings().is_default(), "(g) and the defaults are restored");
        std::cout << "   canonical_field_polynomial(5, 2) under least: " << o.text;
    }
}

// ===========================================================================
// (h) resume-invariance
// ===========================================================================

void ArmResume() {
    std::cout << "-- (h) resume-invariance on the shipped instances (Decomposition, Sturm)\n";
    Fixture fx;

    // --- Decomposition: the locality loop, budget=8 -> 64 --------------------
    Expect0(fx,
            "set locality = \"budget=8\";\n"
            "D8 = __nonlocal();\n"
            "assert(D8.locality.complete == false);\n"
            "assert(state(D8).resumable.resumable == true);\n"
            "assert(state(D8).resumable.budget_setting == \"locality\");\n"
            "assert(state(D8).resumable.progress.budget_history == [8]);\n"
            "set locality = \"budget=64\";\n"
            "F = __nonlocal();\n"
            "assert(F.locality.complete == true);\n"
            "R = resume(D8, 64);\n"
            "assert(R.locality.complete == true);\n"
            "assert(hash(R) == hash(F));\n"
            "assert(hash(R.U) == hash(F.U));\n"
            "assert(R.witness_hash == F.witness_hash);\n"
            "assert(R.num_blocks == F.num_blocks);\n"
            "assert(R.certify() == 0);\n"
            "assert(state(R).settings.locality == \"budget=64\");\n"
            "assert(state(R).resumable.progress.budget_history == [8, 64]);\n"
            "assert(state(F).resumable.progress.budget_history == [64]);\n"
            "assert(state(R).resumable.progress.budget_history != "
            "state(F).resumable.progress.budget_history);\n",
            "(h) Decomposition: created under budget=8 and resumed to 64 hashes equal to a "
            "fresh handle under 64; budget_history [8, 64] vs [64]");
    {
        // Entrywise, modulo budget_history -- and NOT equal without the modulo.
        const Value* R = fx.session.find_var("R");
        const Value* F = fx.session.find_var("F");
        const hecke::HeckeModuleDecomposition* dr =
            (R != nullptr && R->is_object()) ? bind::DecompositionOf(R->object_value()) : nullptr;
        const hecke::HeckeModuleDecomposition* df =
            (F != nullptr && F->is_object()) ? bind::DecompositionOf(F->object_value()) : nullptr;
        Check(dr != nullptr && df != nullptr, "(h) both decompositions are computed");
        if (dr != nullptr && df != nullptr) {
            Check(dr->locality_state().equals(df->locality_state(), true),
                  "(h) resumed state == fresh state, entrywise, modulo budget_history");
            Check(!dr->locality_state().equals(df->locality_state(), false),
                  "(h) and they DIFFER in budget_history (the one member that must)");
            // Engine level: a resume on a COMPLETE state is a strict no-op.
            hecke::HeckeModuleDecomposition copy(*dr);
            copy.resume_locality(100);
            Check(copy.locality_state().equals(dr->locality_state(), false) &&
                      copy.change_of_basis() == dr->change_of_basis(),
                  "(h) HeckeModuleDecomposition::resume_locality on a complete state is a "
                  "strict no-op (history included)");
        }
    }
    Expect0(fx, "R2 = resume(D8, \"budget=64\");\nassert(hash(R2) == hash(F));\n",
            "(h) the budget may be given as the setting's own text \"budget=<N>\"");
    ExpectError(fx, "resume(D8, 4)", 3, "budget below reached state",
                "(h) Decomposition: a resume to a LOWER budget is refused with the named error");
    ExpectError(fx, "resume(D8, \"strict\")", 3, "must be budget=<N>",
                "(h) Decomposition: a resume continues the budget loop, never strict mode");
    Expect0(fx, "HR = state(R).resumable.progress.budget_history;", "(h) record R's history");
    ExpectError(fx, "resume(R, 100)", 3, "is complete",
                "(h) Decomposition: a resume on a COMPLETE handle is refused");
    Expect0(fx, "assert(state(R).resumable.progress.budget_history == HR);\n",
            "(h) ... and the refused resume moved nothing");
    Expect0(fx,
            "set locality = \"budget=8\";\n"
            "D8b = __nonlocal();\n"
            "assert(D8b.locality.complete == false);\n"
            "set locality = \"budget=64\";\n"
            "set u_ell = \"off\";\n",
            "(h) a second budget=8 handle, then a NON-budget key moved");
    ExpectError(fx, "resume(D8b, 64)", 3, "resume under a different setting",
                "(h) Decomposition: a resume under a changed non-budget setting is refused");
    Expect0(fx,
            "set u_ell = \"auto\";\n"
            "R3 = resume(D8b, 64);\n"
            "assert(hash(R3) == hash(F));\n",
            "(h) ... and allowed again once the setting is back");
    {
        const Outcome o = Run(fx, "print(save_state(D8));");
        Check(o.code == 0 && Contains(o.text, "\"kind\": \"locality\"") &&
                  Contains(o.text, "\"complete\": false") &&
                  Contains(o.text, "\"budget_history\": [8]"),
              "(h) save_state(D8) writes the R3 object: kind locality, incomplete, history [8]" +
                  (o.code == 0 ? std::string() : "\n" + o.text));
    }

    // --- Sturm: the sweep, cap 3 -> 36 -----------------------------------------
    Expect0(fx,
            "CUT = sturm_bound(24, 5, 3);\n"
            "assert(CUT.found == false);\n"
            "CONT = resume(CUT, 36);\n"
            "FRESH = sturm_bound(24, 5, 36);\n"
            "assert(CONT.found == FRESH.found);\n"
            "assert(CONT.B_prime == FRESH.B_prime);\n"
            "assert(CONT.closure_dim == FRESH.closure_dim);\n"
            "assert(CONT.primes_done == FRESH.primes_done);\n"
            "assert(CONT.certified == true);\n"
            "assert(state(CONT).resumable.progress.budget_history == [3, 36]);\n"
            "assert(state(FRESH).resumable.progress.budget_history == [36]);\n",
            "(h) Sturm: cut at cap 3 and resumed to 36 equals a fresh run at 36, modulo "
            "budget_history [3, 36] vs [36]");
    ExpectError(fx, "resume(CUT, 2)", 3, "budget below reached state",
                "(h) Sturm: a resume to a LOWER cap is refused with the named error, eagerly");
    ExpectError(fx, "resume(CONT, 100)", 3, "is complete",
                "(h) Sturm: a resume on a COMPLETE handle is refused");
    Expect0(fx, "set final_mode = \"sturm_heuristic\";\n", "(h) a non-budget key moved");
    ExpectError(fx, "resume(CUT, 36)", 3, "resume under a different setting",
                "(h) Sturm: a resume under a changed non-budget setting is refused");
    Expect0(fx,
            "set final_mode = \"closure\";\n"
            "set sturm = \"off\";\n"
            "C2 = resume(CUT, 36);\n"
            "assert(C2.B_prime == FRESH.B_prime);\n"
            "set sturm = \"auto\";\n"
            "assert(fingerprint() == \"0x0000000000000000\");\n",
            "(h) Sturm: the budget key's own root (`sturm`) is exempt from the comparison");
}

} // namespace

uint32_t ConsoleSettingsValidation() {
    g_checks = 0;
    g_failures = 0;
    std::cout << "=== ConsoleSettingsValidation (console C2a: [P] Sec. 10 C2 (f), (g), (h)) ===\n";
    const hecke::MathSettings before = hecke::math_settings();
    try {
        ArmEngineKnobs();
        ArmMathSettings();
        ArmResume();
    } catch (const std::exception& e) {
        Check(false, std::string("unexpected exception: ") + e.what());
    }
    Check(hecke::math_settings() == before,
          "the validation leaves the mathematical settings as it found them");
    std::cout << "ConsoleSettingsValidation: " << g_checks << " checks, " << g_failures
              << " failed\n";
    return g_failures;
}
