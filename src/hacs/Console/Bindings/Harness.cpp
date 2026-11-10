// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Harness.cpp -- console phase C2b (hecke-extensions slot 7, WP-8): the
// harness verbs.
//
// Plan docs/console/00-console-interface-plan.md v3.4 Sec. 5 (rows
// "Validations", "Benchmarks", "Micro-benchmarks", "Campaigns", "Golden
// harness", "Timing / memory"), Sec. 2 C4 (the session model: harness verbs
// run with the session overrides LIFTED), Sec. 2 C5 ("No timings on stdout by
// default"), Sec. 9 (this TU) and Sec. 10 C2b, acceptance (a)-(e).
//
//   validations([tier])            the registered roster, as [name, tier] pairs
//   run_validation(name [, inherit_settings])
//                                  ValidationRegistry::run, in process -> Int
//   bench(mode, n, ell, p = .., r = .., family = [..], repeat = .., lean = ..,
//         cross_checks = .., count_allocs = .., inherit_settings = ..)
//                                  bench::RunMode (gap G4) -> a BenchRun Record
//   micro(op, a, b, c, d, backend = .., repeat = .., inherit_settings = ..)
//                                  the `micro` mode of the same runner
//   campaign_spec(path) / campaign_spec_text(text)
//                                  a parsed CampaignSpec handle
//   campaign_plan(spec)            plan_campaign(): the shape groups and job ids
//   run_campaign(spec [, output, inherit_settings])
//                                  RunCampaignSpec -> a CampaignRun Record
//   golden_subset(cells [, inherit_settings])
//                                  GoldenSubset32 -> {mismatches, digest, records}
//   timed(x)                       forces a declared handle and times it
//   bench_line(record [, mode])    the record's `BENCH ...` line, as a `#~` line
//   rss()                          the PROCESS resident-set figures (gap G4b)
//
// THE SESSION'S OVERRIDES ARE LIFTED ([P] Sec. 2 C4, risk R6).  A session's
// `set threads = 1` or `set backend = "legacy"` is applied through
// process-global and thread-local setters; a validation, a benchmark or a
// campaign started from the session would inherit it silently and its result
// would be misread.  So every verb that runs one of those installs the
// session's BASELINE (the environment's values, and the mathematical
// defaults) for its duration through the core's SettingsScope, and the
// session's own values are re-installed afterwards.  `inherit_settings = true`
// opts out.
//
// THE CHATTER IS CAPTURED ([P] Sec. 2 C5, acceptance (c)).  A validation, a
// bench runner and a campaign print freely; around each call both std::cout
// and std::cerr are swapped to a sink (the core's ChatterCapture), so the
// transcript sees only what the script prints -- unless `verbose` is on, when
// the text goes to the terminal as it always did.
//
// NO TIMING REACHES A TRANSCRIPT BY ACCIDENT.  The console has no floating
// type, so every time and every MiB figure is a String (`"0.0421"`) with an
// Int twin in micro-units (`total_us`, `peak_rss_kb`) for arithmetic.  A script
// that prints a whole BenchRun record prints timings -- which is its choice;
// `bench_line` prints the `BENCH ...` line as a `#~` annotation, which every
// `.expected` comparison ignores.
// ===========================================================================

#include "Bindings.h"
#include "RegisterPipelineVerbs.h"

#include "../Core/Cancel.h"
#include "../Core/Evaluator.h"
#include "../Core/Printer.h"
#include "../Core/Session.h"
#include "../Core/Settings.h"

#include "../../Bench/BenchDriver.h"
#include "../../../ModArithmetic/LegacyConfig.h"   // (S24 CL-L2) kLegacyBuild, compiled_out_text
#include "../../Campaign/CampaignDescent.h"
#include "../../Campaign/CampaignDriver.h"
#include "../../Campaign/CampaignSpec.h"
#include "../../Golden/GoldenHarness.h"
#include "../../Util/Rss.h"
#include "../../ValidationRegistry.h"
#include "../../Util/TextIo.h"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace console {
namespace bind {

namespace {

// --- the test seam (ConsoleDeterminismValidation, acceptance (c)) ----------
std::function<void()>& LiftProbe() {
    static std::function<void()> probe;
    return probe;
}

// The validations a `run_validation` is currently inside, so that a script
// cannot make a validation run itself (ConsoleDeterminismValidation runs a
// script that calls run_validation; that script must not be able to call
// ConsoleDeterminismValidation back).
std::vector<std::string>& RunningValidations() {
    static thread_local std::vector<std::string> running;
    return running;
}

std::string Fixed(double v, int prec) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

using oc_util::HexU64;   // S24 CL-7 (F11): Util/TextIo.h
std::int64_t Micro(double seconds) {
    return static_cast<std::int64_t>(seconds * 1e6 + 0.5);
}

// The RSS figure the BENCH line prints ("n/a" below zero).
std::string MiBText(double mib) {
    return mib < 0.0 ? std::string("n/a") : Fixed(mib, 1);
}

std::int64_t KiB(double mib) {
    return mib < 0.0 ? -1 : static_cast<std::int64_t>(mib * 1024.0 + 0.5);
}

bool BoolArg(const Value& v, const SourcePos& pos, const char* who) {
    if (v.is_nil()) return false;
    if (!v.is_bool()) throw TypeError(std::string(who) + ": expected a Bool", pos);
    return v.bool_value();
}

// The two scopes every harness verb runs inside; see the header comment.
// Declared in this order so the destructors run in the reverse: the session's
// values come back first, and only then does std::cout come back.
class HarnessScope {
public:
    HarnessScope(CallContext& ctx, bool inherit_settings) {
        Session& s = *ctx.session;
        if (!s.verbose()) capture_.reset(new ChatterCapture(s));
        if (!inherit_settings) {
            lift_.reset(new SettingsScope(s.settings(), s.settings().baseline()));
        }
        if (LiftProbe()) LiftProbe()();
    }

private:
    std::unique_ptr<ChatterCapture> capture_;
    std::unique_ptr<SettingsScope> lift_;
};

// --- validations -----------------------------------------------------------

Value VerbValidations(CallContext& ctx, const std::vector<Value>& a) {
    std::string want;
    if (!a.empty() && !a[0].is_nil()) {
        want = a[0].string_value();
        if (want != "fast" && want != "standard" && want != "slow") {
            throw TypeError("validations(tier): tier is \"fast\", \"standard\" or \"slow\", got \"" +
                                want + "\"",
                            ctx.pos);
        }
    }
    std::vector<Value> out;
    for (const auto& e : ValidationRegistry::list_with_tiers()) {
        const std::string tier = ValidationTierName(e.second);
        if (!want.empty() && tier != want) continue;
        out.push_back(Value::from_list({Value::from_string(e.first), Value::from_string(tier)}));
    }
    return Value::from_list(std::move(out));
}

// The owner-gated emitters (CONTRIBUTING.md: GOLDEN_EMIT, CONSOLE_EMIT_EXPECTED,
// the dumps): each is a no-op without its variable, but a console verb is
// not where an owner gate should be exercised, so they are refused by name.
bool IsOwnerGatedEmitter(const std::string& name) {
    return name == "GoldenEmit" || name == "GoldenDump" || name == "ConsoleScriptEmit";
}

Value VerbRunValidation(CallContext& ctx, const std::vector<Value>& a) {
    const std::string& name = a[0].string_value();
    const bool inherit = a.size() > 1 && BoolArg(a[1], ctx.pos, "run_validation(name, inherit_settings)");
    bool known = false;
    for (const std::string& n : ValidationRegistry::list()) {
        if (n == name) { known = true; break; }
    }
    if (!known) {
        throw RuntimeError("run_validation: no validation is registered as '" + name +
                               "' (validations() lists them)",
                           ctx.pos);
    }
    if (IsOwnerGatedEmitter(name)) {
        throw RuntimeError("run_validation: '" + name +
                               "' is an owner-gated emitter; it runs from the command line only",
                           ctx.pos);
    }
    for (const std::string& n : RunningValidations()) {
        if (n == name) {
            throw RuntimeError("run_validation: '" + name +
                                   "' is already running in this call chain (it would run itself)",
                               ctx.pos);
        }
    }
    uint32_t rc = 0;
    {
        HarnessScope scope(ctx, inherit);
        RunningValidations().push_back(name);
        try {
            rc = ValidationRegistry::run(name);
        } catch (...) {
            RunningValidations().pop_back();
            throw;
        }
        RunningValidations().pop_back();
    }
    return Value::from_int(static_cast<std::int64_t>(rc));
}

// --- bench / micro ----------------------------------------------------------

Value BenchRecordValue(const bench::BenchRecord& r) {
    Record rec;
    rec.set("mode", Value::from_string(r.mode));
    rec.set("params", Value::from_string(r.params));
    std::vector<Value> dims;
    for (uint32_t d : r.dims) dims.push_back(Value::from_int(d));
    rec.set("dims", Value::from_list(std::move(dims)));
    rec.set("total_s", Value::from_string(Fixed(r.total_s, 4)));
    rec.set("total_us", Value::from_int(Micro(r.total_s)));
    rec.set("median_s", Value::from_string(Fixed(r.median_s, 4)));
    rec.set("repeats", Value::from_int(static_cast<std::int64_t>(r.totals.size())));
    rec.set("rss_mb", Value::from_string(MiBText(r.rss_mb)));
    std::vector<Value> phases;
    for (const auto& ph : r.phases) {
        phases.push_back(Value::from_list(
            {Value::from_string(ph.first), Value::from_string(Fixed(ph.second, 4))}));
    }
    rec.set("phases", Value::from_list(std::move(phases)));
    Record extra;
    for (const auto& kv : r.extra) extra.set(kv.first, Value::from_string(kv.second));
    rec.set("extra", Value::from_record(std::move(extra)));
    if (r.mode == "micro") rec.set("checksum", Value::from_string(HexU64(r.checksum)));
    rec.set("bench_line", Value::from_string(r.bench_line));
    return Value::from_record(std::move(rec));
}

bench::BenchRecord RunBenchChecked(CallContext& ctx, const bench::ModeArgs& args, bool inherit,
                                   const char* who) {
    bench::BenchRecord r;
    {
        HarnessScope scope(ctx, inherit);
        r = bench::RunMode(args);
    }
    if (r.exit_code != 0) {
        std::string msg = r.error;
        while (!msg.empty() && msg[0] == ' ') msg.erase(0, 1);
        throw RuntimeError(std::string(who) + ": " + (msg.empty() ? "failed" : msg) +
                               " (exit code " + std::to_string(r.exit_code) + ")",
                           ctx.pos);
    }
    return r;
}

// bench(mode, n, ell, p, r, family, repeat, lean, cross_checks, count_allocs,
//       inherit_settings)
Value VerbBench(CallContext& ctx, const std::vector<Value>& a) {
    const char* who = "bench(mode, n, ell, ...)";
    bench::ModeArgs args;
    args.mode = a[0].string_value();
    if (args.mode == "micro") {
        throw TypeError("bench: the micro-benchmarks are micro(op, ...)", ctx.pos);
    }
    if (args.mode != "cohom" && args.mode != "hecke" && args.mode != "family" &&
        args.mode != "decomp" && args.mode != "ext") {
        throw TypeError("bench: mode is \"cohom\", \"hecke\", \"family\", \"decomp\" or \"ext\", got \"" +
                            args.mode + "\"",
                        ctx.pos);
    }
    args.n_sym = U32Arg(a[1], ctx.pos, who);
    args.ell = U32Arg(a[2], ctx.pos, who);
    args.p = U64Arg(a[3], ctx.pos, who);
    args.r = U32Arg(a[4], ctx.pos, who);
    for (const Value& q : a[5].list_value()) args.family.push_back(U64Arg(q, ctx.pos, who));
    args.repeats = U32Arg(a[6], ctx.pos, who);
    args.lean = BoolArg(a[7], ctx.pos, who);
    args.cross_checks = BoolArg(a[8], ctx.pos, who);
    args.count_allocs = BoolArg(a[9], ctx.pos, who);
    const bool inherit = BoolArg(a[10], ctx.pos, who);
    if ((args.mode == "hecke" || args.mode == "family" || args.mode == "decomp") && args.p < 2) {
        throw TypeError(std::string("bench: mode \"") + args.mode + "\" needs p = <a prime >= 2>",
                        ctx.pos);
    }
    return BenchRecordValue(RunBenchChecked(ctx, args, inherit, "bench"));
}

// micro(op, a, b, c, d, backend, repeat, inherit_settings)
Value VerbMicro(CallContext& ctx, const std::vector<Value>& a) {
    const char* who = "micro(op, ...)";
    bench::ModeArgs args;
    args.mode = "micro";
    args.op = a[0].string_value();
    if (args.op == "cuda") {
        throw TypeError("micro: the GPU throughput table (`cuda`) is a command-line report "
                        "(--bench micro cuda)",
                        ctx.pos);
    }
    for (std::size_t i = 1; i <= 4; ++i) {
        if (a[i].is_nil()) break;
        args.op_args.push_back(U32Arg(a[i], ctx.pos, who));
    }
    args.backend = a[5].string_value();
    if (args.backend == "legacy" && !oc::legacy::kLegacyBuild) {
        // (S24 CL-L2, CLEANUP plan Sec. 2.4) a legal argument this build cannot execute: exit 5, never the engine.
        throw NotImplementedError(std::string("micro: ") + oc::legacy::compiled_out_text("backend = \"legacy\""), ctx.pos);
    }
    args.repeats = U32Arg(a[6], ctx.pos, who);
    const bool inherit = BoolArg(a[7], ctx.pos, who);
    return BenchRecordValue(RunBenchChecked(ctx, args, inherit, "micro"));
}

// --- campaigns --------------------------------------------------------------

struct CampaignSpecVal {
    campaign::CampaignSpec spec;
};

const CampaignSpecVal& AsCampaignSpec(const Value& v, const SourcePos& pos, const char* who) {
    const ObjectPtr& o = AsObjectOf(v, kTypeCampaignSpec, pos, who);
    return *o->payload_as<CampaignSpecVal>();
}

Value CampaignSpecValue(CallContext& ctx, campaign::CampaignSpec spec) {
    std::shared_ptr<CampaignSpecVal> p = std::make_shared<CampaignSpecVal>();
    p->spec = std::move(spec);
    return Value::from_object(MakeValueObject(ctx, kTypeCampaignSpec, p));
}

Value VerbCampaignSpec(CallContext& ctx, const std::vector<Value>& a) {
    const std::string path = ResolvePath(ctx.session->current_dir(), a[0].string_value());
    try {
        return CampaignSpecValue(ctx, campaign::ParseCampaignSpecFile(path));
    } catch (const ConsoleError&) {
        throw;
    } catch (const hecke::NotImplementedInThisBuild& e) {
        // (slot-9 WP-8c; [P] Sec. 8's own exit code, 5) A spec REFUSED FOR A
        // RING REASON (`ext = true` under a non-default `coefficient_ring`, or
        // a grid cell with ell^k >= 2^64: CampaignSpec.cpp's two
        // `hecke::RefuseCoefficientRing` sites) is "this build cannot do it",
        // never "you asked for something wrong" (exit 1) or "the computation
        // failed" (exit 3).  Every OTHER spec error stays a std::runtime_error
        // below, unchanged, so it keeps exit 3.  Caught ahead of the generic
        // std::exception handler so the kind is not lost.
        throw NotImplementedError(std::string("campaign_spec: ") + e.what(), ctx.pos);
    } catch (const std::exception& e) {
        throw RuntimeError(std::string("campaign_spec: ") + e.what(), ctx.pos);
    }
}

Value VerbCampaignSpecText(CallContext& ctx, const std::vector<Value>& a) {
    std::istringstream is(a[0].string_value());
    try {
        return CampaignSpecValue(ctx, campaign::ParseCampaignSpec(is, "<campaign_spec_text>"));
    } catch (const ConsoleError&) {
        throw;
    } catch (const hecke::NotImplementedInThisBuild& e) {
        // See VerbCampaignSpec above: the same ring refusal, reached from spec
        // text instead of a file.
        throw NotImplementedError(std::string("campaign_spec_text: ") + e.what(), ctx.pos);
    } catch (const std::exception& e) {
        throw RuntimeError(std::string("campaign_spec_text: ") + e.what(), ctx.pos);
    }
}

// (C5 leftovers, slot-10 WP-12; [S10] decision 6 (iii) (A), [P] v3.8 Sec. 8)
// The WP-8c handler of campaign_spec / campaign_spec_text, for the two verbs
// that use a parsed spec: a ConsoleError is re-thrown unchanged; a
// hecke::NotImplementedInThisBuild ("this build cannot do it") becomes exit 5,
// caught AHEAD of std::exception so its kind is not lost; every other
// exception stays a runtime error, exit 3.
template <class F>
Value CampaignVerbHandler(CallContext& ctx, const char* verb, F&& body) {
    try {
        return body();
    } catch (const ConsoleError&) {
        throw;
    } catch (const hecke::NotImplementedInThisBuild& e) {
        throw NotImplementedError(std::string(verb) + ": " + e.what(), ctx.pos);
    } catch (const std::exception& e) {
        throw RuntimeError(std::string(verb) + ": " + e.what(), ctx.pos);
    }
}

// Reachability (slot-10 WP-12, recorded in its report): at WP-12's base no
// input reaches a hecke::NotImplementedInThisBuild here -- a spec's ring
// refusals are raised when it is parsed (campaign_spec / campaign_spec_text),
// and plan_campaign() is a pure function of the parsed spec.  The handler
// makes explicit what the core's kind classifier already did for this verb.
Value VerbCampaignPlan(CallContext& ctx, const std::vector<Value>& a) {
    return CampaignVerbHandler(ctx, "campaign_plan", [&]() -> Value {
        const CampaignSpecVal& s = AsCampaignSpec(a[0], ctx.pos, "campaign_plan(spec)");
        std::vector<Value> groups;
        // (slot-24 D6 audit N1) a coefficient_ring = Z spec plans descent cells (CampaignDescent.h), one group
        if (s.spec.math.coefficient_ring.is_integral()) {
            Record r;
            r.set("signature", Value::from_string("integral_descent_cell/v1"));
            std::vector<Value> ids;
            for (const campaign::DescentCellPlan& c : campaign::PlanDescentCells(s.spec)) ids.push_back(Value::from_string(c.id));
            r.set("jobs", Value::from_list(std::move(ids)));
            groups.push_back(Value::from_record(std::move(r)));
            return Value::from_list(std::move(groups));
        }
        for (const campaign::JobGroup& g : campaign::plan_campaign(s.spec)) {
            Record r;
            r.set("signature", Value::from_string(g.sig.to_string()));
            std::vector<Value> ids;
            for (const campaign::Job& j : g.jobs) ids.push_back(Value::from_string(j.id()));
            r.set("jobs", Value::from_list(std::move(ids)));
            groups.push_back(Value::from_record(std::move(r)));
        }
        return Value::from_list(std::move(groups));
    });
}

// Termux has no writable /tmp: an explicit "/tmp/..." output goes under $TMPDIR instead (only when /tmp is not writable).
static std::string RedirectTmpPath(const std::string& p) {
    if (p.compare(0, 5, "/tmp/") != 0 || ::access("/tmp", W_OK) == 0) return p;
    const char* t = std::getenv("TMPDIR");
    return (t && *t) ? std::string(t) + p.substr(4) : p;
}

// run_campaign(spec, output = "", inherit_settings = false)
Value VerbRunCampaign(CallContext& ctx, const std::vector<Value>& a) {
    const char* who = "run_campaign(spec, output, inherit_settings)";
    campaign::CampaignSpec spec = AsCampaignSpec(a[0], ctx.pos, who).spec;
    if (!a[1].string_value().empty()) spec.output_dir = RedirectTmpPath(a[1].string_value());
    const bool inherit = BoolArg(a[2], ctx.pos, who);
    // (slot-24 D6 audit N1) a coefficient_ring = Z spec runs its descent cells (CampaignDescent.h) under the SPEC's settings,
    // as the CLI's --campaign does; the record has the same keys plus `refused` (skipped = the resumed cells)
    if (spec.math.coefficient_ring.is_integral()) {
        campaign::DescentCampaignResult dres;
        const auto t0 = std::chrono::steady_clock::now();
        {
            HarnessScope scope(ctx, inherit);
            CampaignVerbHandler(ctx, "run_campaign", [&]() -> Value {
                dres = campaign::RunDescentCampaign(spec, /*verbose=*/ctx.session->verbose());
                return Value::nil();
            });
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        Record r;
        r.set("output_dir", Value::from_string(spec.output_dir));
        r.set("ran", Value::from_int(dres.ran));
        r.set("skipped", Value::from_int(dres.resumed));
        r.set("failed", Value::from_int(dres.failed));
        r.set("refused", Value::from_int(dres.refused));
        std::vector<Value> ids;
        for (const campaign::DescentCellPlan& c : campaign::PlanDescentCells(spec)) ids.push_back(Value::from_string(c.id));
        r.set("jobs", Value::from_list(std::move(ids)));
        r.set("summary_tsv", Value::from_string(spec.output_dir + "/summary.tsv"));
        r.set("index_json", Value::from_string(spec.output_dir + "/index.json"));
        r.set("wall_s", Value::from_string(Fixed(wall, 4)));
        return Value::from_record(std::move(r));
    }
    campaign::CampaignResult res;
    {
        HarnessScope scope(ctx, inherit);
        // (slot-10 WP-12) The campaign handler above: a ring refusal escaping
        // the driver is exit 5 (it was wrapped as exit 3); everything else
        // keeps exit 3.  Reachability at WP-12's base: none found -- every
        // job, ext-cell and ring-campaign refusal is caught and recorded
        // inside RunCampaignSpec (the job loop, RunCellWork, CampaignRing's
        // ExpectRefusal), which runs under the SPEC's mathematical settings,
        // so `inherit_settings = true` under a session ring reaches none
        // either (WP-12 report).
        CampaignVerbHandler(ctx, "run_campaign", [&]() -> Value {
            res = campaign::RunCampaignSpec(spec, /*verbose=*/ctx.session->verbose());
            return Value::nil();
        });
    }
    Record r;
    r.set("output_dir", Value::from_string(res.output_dir));
    r.set("ran", Value::from_int(static_cast<std::int64_t>(res.num_ran)));
    r.set("skipped", Value::from_int(static_cast<std::int64_t>(res.num_skipped)));
    r.set("failed", Value::from_int(static_cast<std::int64_t>(res.num_failed)));
    std::vector<Value> ids;
    for (const campaign::JobResult& j : res.jobs) ids.push_back(Value::from_string(j.job.id()));
    r.set("jobs", Value::from_list(std::move(ids)));
    r.set("summary_tsv", Value::from_string(res.output_dir + "/summary.tsv"));
    r.set("index_json", Value::from_string(res.output_dir + "/index.json"));
    r.set("wall_s", Value::from_string(Fixed(res.wall_s, 4)));
    return Value::from_record(std::move(r));
}

// --- the golden harness -----------------------------------------------------

Value VerbGoldenSubset(CallContext& ctx, const std::vector<Value>& a) {
    const char* who = "golden_subset(cells, inherit_settings)";
    std::vector<std::pair<uint32_t, uint32_t>> cells;
    for (const Value& c : a[0].list_value()) {
        if (!c.is_list() || c.list_value().size() != 2) {
            throw TypeError("golden_subset: every cell is a two-element list [n, ell]", ctx.pos);
        }
        cells.emplace_back(U32Arg(c.list_value()[0], ctx.pos, who),
                           U32Arg(c.list_value()[1], ctx.pos, who));
    }
    if (cells.empty()) throw TypeError("golden_subset: the cell list is empty", ctx.pos);
    const bool inherit = BoolArg(a[1], ctx.pos, who);
    uint64_t digest = 0;
    size_t records = 0;
    uint32_t mismatches = 0;
    {
        HarnessScope scope(ctx, inherit);
        mismatches = GoldenSubset32(cells, digest, records, /*quiet=*/true);
    }
    Record r;
    r.set("mismatches", Value::from_int(mismatches));
    r.set("digest", Value::from_string(HexU64(digest)));
    r.set("records", Value::from_int(static_cast<std::int64_t>(records)));
    return Value::from_record(std::move(r));
}

// --- timing and memory -------------------------------------------------------

Value VerbTimed(CallContext& ctx, const std::vector<Value>& a) {
    const Value& x = a[0];
    const std::size_t n0 = bench::ProcessAllocCount();
    const std::size_t b0 = bench::ProcessAllocBytes();
    const auto t0 = std::chrono::steady_clock::now();
    if (x.is_object() && x.object_value() && !x.object_value()->computed()) {
        x.object_value()->ensure_computed(ctx.session->settings(), ctx.pos);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const std::size_t n1 = bench::ProcessAllocCount();
    const std::size_t b1 = bench::ProcessAllocBytes();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    const double peak = oc_util::PeakRssMiB();
    Record r;
    r.set("value", x);
    r.set("wall_s", Value::from_string(Fixed(wall, 4)));
    r.set("wall_us", Value::from_int(Micro(wall)));
    r.set("peak_rss_mb", Value::from_string(MiBText(peak)));
    r.set("peak_rss_kb", Value::from_int(KiB(peak)));
    r.set("allocs", Value::from_int(static_cast<std::int64_t>(n1 - n0)));
    r.set("alloc_bytes", Value::from_int(static_cast<std::int64_t>(b1 - b0)));
    return Value::from_record(std::move(r));
}

Value VerbBenchLine(CallContext& ctx, const std::vector<Value>& a) {
    const Record& r = a[0].record_value();
    std::string line;
    const Value* bl = r.find("bench_line");
    if (bl != nullptr && bl->is_string() && !bl->string_value().empty()) {
        line = bl->string_value();
    } else {
        // A timed() record: the tree's BENCH format over its fields, formatted
        // by the one formatter every `--bench` line goes through.
        const Value* us = r.find("wall_us");
        if (us == nullptr || !us->is_int()) {
            throw TypeError("bench_line(record): the record carries neither `bench_line` (bench, "
                            "micro) nor `wall_us` (timed)",
                            ctx.pos);
        }
        const std::string mode =
            (a.size() > 1 && !a[1].is_nil()) ? a[1].string_value() : std::string("timed");
        const Value* kb = r.find("peak_rss_kb");
        const double rss = (kb != nullptr && kb->is_int() && kb->int_value() >= 0)
                               ? static_cast<double>(kb->int_value()) / 1024.0
                               : -1.0;
        std::string extra;
        const Value* al = r.find("allocs");
        const Value* ab = r.find("alloc_bytes");
        if (al != nullptr && al->is_int() && ab != nullptr && ab->is_int()) {
            extra = "allocs=" + std::to_string(al->int_value()) +
                    " alloc_bytes=" + std::to_string(ab->int_value());
        }
        line = bench::FormatBenchLine(mode, std::string(),
                                      static_cast<double>(us->int_value()) / 1e6, rss, extra);
    }
    ctx.session->print_annotation(line);
    return Value::nil();
}

Value VerbRss(CallContext&, const std::vector<Value>&) {
    // Current first, peak second: VmHWM only grows, so the later read of the peak is
    // always >= the earlier read of the current set (KNOWN_ISSUES Sec. 6).
    const double cur = oc_util::CurrentRssMiB();
    const double peak = oc_util::PeakRssMiB();
    Record r;
    r.set("peak_rss_mb", Value::from_string(MiBText(peak)));
    r.set("rss_mb", Value::from_string(MiBText(cur)));
    r.set("peak_rss_kb", Value::from_int(KiB(peak)));
    r.set("rss_kb", Value::from_int(KiB(cur)));
    r.set("scope", Value::from_string("process"));
    return Value::from_record(std::move(r));
}

void RegisterCampaignSpecType(Registry& registry) {
    TypeInfo t;
    t.name = kTypeCampaignSpec;
    t.doc = "a parsed campaign spec (docs/matrix-engine/campaign.md) -- campaign_spec(path), "
            "campaign_spec_text(text)";
    t.members.push_back({"name", false, "the spec's `name`", [](const ObjectPtr& o) {
                             return Value::from_string(o->payload_as<CampaignSpecVal>()->spec.name);
                         }});
    t.members.push_back({"output_dir", false, "where run_campaign writes by default",
                         [](const ObjectPtr& o) {
                             return Value::from_string(
                                 o->payload_as<CampaignSpecVal>()->spec.output_dir);
                         }});
    t.members.push_back({"jobs", false, "the number of jobs the plan has",
                         [](const ObjectPtr& o) {
                             const campaign::CampaignSpec& s =
                                 o->payload_as<CampaignSpecVal>()->spec;
                             return Value::from_int(static_cast<std::int64_t>(
                                 campaign::flatten(campaign::plan_campaign(s)).size()));
                         }});
    t.print = [](const ObjectPtr& o, std::int64_t) {
        const campaign::CampaignSpec& s = o->payload_as<CampaignSpecVal>()->spec;
        return "CampaignSpec(" + s.name + ", " + s.origin + ")";
    };
    t.show = [](const ObjectPtr& o) {
        const campaign::CampaignSpec& s = o->payload_as<CampaignSpecVal>()->spec;
        return "CampaignSpec(" + s.name + ", " + s.origin + ", " +
               std::to_string(s.sections.size()) + " section(s))";
    };
    registry.register_type(std::move(t));
}

} // namespace

void SetHarnessLiftProbe(std::function<void()> probe) { LiftProbe() = std::move(probe); }

const std::vector<std::string>& HarnessVerbNames() {
    static const std::vector<std::string> names = {
        "validations", "run_validation", "bench", "micro", "campaign_spec", "campaign_spec_text",
        "campaign_plan", "run_campaign", "golden_subset", "timed", "bench_line", "rss",
    };
    return names;
}

void RegisterHarnessBindings(Registry& registry) {
    RegisterCampaignSpecType(registry);
    const char* g = "harness";
    auto add = [&registry, g](const char* name, Signature sig, VerbImpl impl, const char* doc) {
        // (NBCONFIRM S5) the drivers (validations, benchmarks, campaigns, golden subsets) report their own exceptions as
        // failures or records (catch (...) sites), so no cancellation point inside them may raise: a Ctrl-C / interrupt takes
        // effect at the next point after the driver returns (a second Ctrl-C ends the REPL; the page's Kill session).
        const std::string n(name);
        if (n == "run_validation" || n == "bench" || n == "micro" || n == "run_campaign" || n == "golden_subset") {
            impl = [inner = std::move(impl)](CallContext& ctx, const std::vector<Value>& a) -> Value {
                const NoCancelScope no_cancel;
                return inner(ctx, a);
            };
        }
        registry.register_verb(name, std::move(sig), std::move(impl), doc, g);
    };
    const Value no = Value::from_bool(false);

    add("validations", Sig({}), &VerbValidations,
        "every registered validation as [name, tier], in registration order");
    add("validations", Sig({Param::of("tier", ValueKind::String)}), &VerbValidations,
        "the registered validations of one tier (\"fast\" / \"standard\" / \"slow\")");
    add("run_validation",
        Sig({Param::of("name", ValueKind::String),
             Param::of("inherit_settings", ValueKind::Bool).optional(no)}),
        &VerbRunValidation,
        "runs one registered validation in process and returns its code (0 = pass); the "
        "session's settings are lifted to its baseline for the run unless inherit_settings, and "
        "the validation's own output is captured unless `verbose`");
    add("bench",
        Sig({Param::of("mode", ValueKind::String), Param::of("n", ValueKind::Int),
             Param::of("ell", ValueKind::Int),
             Param::of("p", ValueKind::Int).optional(Value::from_int(0)),
             Param::of("r", ValueKind::Int).optional(Value::from_int(1)),
             Param::of("family", ValueKind::List).optional(Value::from_list({})),
             Param::of("repeat", ValueKind::Int).optional(Value::from_int(1)),
             Param::of("lean", ValueKind::Bool).optional(no),
             Param::of("cross_checks", ValueKind::Bool).optional(no),
             Param::of("count_allocs", ValueKind::Bool).optional(no),
             Param::of("inherit_settings", ValueKind::Bool).optional(no)}),
        &VerbBench,
        "one `--bench` run (cohom / hecke / family / decomp / ext) through bench::RunMode: a "
        "record {mode, params, dims, total_s, total_us, median_s, repeats, rss_mb, phases, "
        "extra, bench_line}; times are Strings");
    add("micro",
        Sig({Param::of("op", ValueKind::String), Param::any("a").optional(Value()),
             Param::any("b").optional(Value()), Param::any("c").optional(Value()),
             Param::any("d").optional(Value()),
             Param::of("backend", ValueKind::String).optional(Value::from_string("")),
             Param::of("repeat", ValueKind::Int).optional(Value::from_int(1)),
             Param::of("inherit_settings", ValueKind::Bool).optional(no)}),
        &VerbMicro,
        "one `--bench micro` op (gemm / gemv / rref / kernel / transpose / suite) with up to "
        "four integer arguments; backend \"\" = the ENGINE_BACKEND default");
    add("campaign_spec", Sig({Param::of("path", ValueKind::String)}), &VerbCampaignSpec,
        "parses a campaign spec file (relative to the running script)");
    add("campaign_spec_text", Sig({Param::of("text", ValueKind::String)}), &VerbCampaignSpecText,
        "parses campaign spec text");
    add("campaign_plan", Sig({Param::object("spec", kTypeCampaignSpec)}), &VerbCampaignPlan,
        "the plan: one {signature, jobs} record per shape group, in execution order");
    add("run_campaign",
        Sig({Param::object("spec", kTypeCampaignSpec),
             Param::of("output", ValueKind::String).optional(Value::from_string("")),
             Param::of("inherit_settings", ValueKind::Bool).optional(no)}),
        &VerbRunCampaign,
        "runs the campaign (writes one JSON per job, index.json and summary.tsv under `output`, "
        "default the spec's output_dir): {output_dir, ran, skipped, failed, jobs, summary_tsv, "
        "index_json, wall_s}");
    add("golden_subset",
        Sig({Param::of("cells", ValueKind::List),
             Param::of("inherit_settings", ValueKind::Bool).optional(no)}),
        &VerbGoldenSubset,
        "recomputes every golden object of the [n, ell] cells and compares both tiers: "
        "{mismatches, digest, records}");
    add("timed", Sig({Param::any("x")}), &VerbTimed,
        "forces a declared handle and times it: {value, wall_s, wall_us, peak_rss_mb, "
        "peak_rss_kb, allocs, alloc_bytes} (peak RSS is the PROCESS high-water mark)");
    add("bench_line", Sig({Param::of("record", ValueKind::Record)}), &VerbBenchLine,
        "prints the record's `BENCH ...` line as a `#~` annotation");
    add("bench_line",
        Sig({Param::of("record", ValueKind::Record), Param::of("mode", ValueKind::String)}),
        &VerbBenchLine, "the same, naming the mode of a timed() record");
    add("rss", Sig({}), &VerbRss,
        "the process's resident set: {peak_rss_mb, rss_mb, peak_rss_kb, rss_kb, scope} -- the "
        "high-water mark of the PROCESS, not of any one verb");
}

} // namespace bind
} // namespace console
