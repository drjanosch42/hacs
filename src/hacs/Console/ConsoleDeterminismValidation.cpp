// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleDeterminismValidation.cpp -- see ConsoleDeterminismValidation.h.
//
// Compiled into cohomology_app (like ConsoleValidation.cpp), never into
// ConsoleCore: it walks directories through POSIX <dirent.h>, reads the engine
// runtime's state and runs campaigns -- none of which the STL-only rule allows
// inside the library.
// ===========================================================================

#include "ConsoleDeterminismValidation.h"
#include "../../ModArithmetic/LegacyConfig.h"   // (S24 CL-L2) RunLegacyArm (SKIP-ARM)

#include "Bindings/Bindings.h"
#include "Bindings/RegisterPipelineVerbs.h"
#include "Core/ConsoleError.h"
#include "Core/Registry.h"
#include "Core/ScriptRunner.h"
#include "Core/Session.h"
#include "Core/Settings.h"
#include "Core/Value.h"

#include "../Campaign/CampaignDriver.h"
#include "../Campaign/CampaignSpec.h"
#include "../Campaign/CampaignValidation.h"
#include "../Hecke/DicksonCount.h"
#include "../Hecke/HeckeSettings.h"
#include "../HeckeExt/RegisterHeckeExtensionVerbs.h"
#include "../../Engine/DeviceBackend.h"
#include "../../Engine/EngineRuntime.h"
#include "../../Engine/TapeElim.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/TextIo.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace console;

std::uint32_t g_checks = 0;
std::uint32_t g_failures = 0;

void Check(bool ok, const std::string& label) {
    ++g_checks;
    if (ok) return;
    ++g_failures;
    if (g_failures <= 40) {
        std::cout << "  [FAIL] " << label << std::endl;
    } else if (g_failures == 41) {
        std::cout << "  [FAIL] (further failures suppressed)" << std::endl;
    }
}

// --- the source tree (ConsoleValidation.cpp's two routes) -------------------

bool PathExists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

bool LooksLikeRepoRoot(const std::string& dir) {
    return PathExists(dir + "/docs/matrix-engine/tools") &&
           PathExists(dir + "/src/hacs");
}

std::string FindRepoRoot() {
#ifdef OC_SOURCE_DIR
    {
        const std::string cmake_root(OC_SOURCE_DIR);
        if (!cmake_root.empty() && LooksLikeRepoRoot(cmake_root)) return cmake_root;
    }
#endif
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            const std::size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir = dir.substr(0, slash);
            if (LooksLikeRepoRoot(dir)) return dir;
        }
    }
    return std::string();
}

using oc_util::HexU64;
using oc_util::ReadFile;   // S24 CL-7 (F11): Util/TextIo.h
std::vector<std::string> SplitLines(const std::string& text) {
    std::vector<std::string> out;
    std::string line;
    for (char c : text) {
        if (c == '\n') { out.push_back(line); line.clear(); }
        else line += c;
    }
    if (!line.empty()) out.push_back(line);
    return out;
}

// The transcript ConsoleScriptValidation compares: `#~` lines dropped,
// trailing whitespace stripped.  (The same rule, re-stated here rather than
// shared, because ConsoleValidation.cpp keeps it file-local.)
std::string TranscriptOf(const std::string& raw) {
    std::string out;
    for (const std::string& line : SplitLines(raw)) {
        if (line.rfind("#~", 0) == 0) continue;
        std::size_t end = line.size();
        while (end > 0 && (line[end - 1] == ' ' || line[end - 1] == '\r' || line[end - 1] == '\t')) {
            --end;
        }
        out += line.substr(0, end);
        out += '\n';
    }
    return out;
}

void CollectScripts(const std::string& dir, std::vector<std::string>& out) {
    DIR* d = ::opendir(dir.c_str());
    if (d == nullptr) return;
    while (struct dirent* e = ::readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() < 4 || name.substr(name.size() - 3) != ".oc") continue;
        if (name[0] < '0' || name[0] > '9') continue;   // numbered scripts only
        out.push_back(dir + "/" + name);
    }
    ::closedir(d);
    std::sort(out.begin(), out.end());
}

// A session over its OWN registry, exactly as ConsoleScriptValidation builds
// one (the extension verbs included), writing to an in-memory stream.
struct Fixture {
    Registry registry;
    std::ostringstream out;
    Session session;

    Fixture() : session(registry, /*verbose_default=*/false) {
        RegisterPipelineVerbs(registry);
        hecke_ext::RegisterHeckeExtensionVerbs(registry);
        registry.install_settings(session.settings());
        session.settings().set_baseline(session.settings().snapshot());
        session.set_output(out);
    }
};

struct ScriptRun {
    int code = -1;
    std::string transcript;
};

// Runs one script on a fresh session whose STARTING value of `key` is `value`
// -- i.e. `set key = value` made the session's baseline, as the environment
// variable would make it, so a script that calls reset_settings() returns to
// the arm's value and not to the host's.  The host's baseline is put back
// before the session ends (Session's destructor re-applies the baseline, and
// the arm's value must not leak into the next validation of an `--all` run).
// (C4) The same over several keys at once -- one arm of the threads x
// program_cache x device product.
using Arm = std::vector<std::pair<std::string, Value>>;

ScriptRun RunScriptUnderArm(const std::string& path, const Arm& arm) {
    const hecke::ScopedMathSettings restore(hecke::math_settings());
    Fixture fx;
    const SettingsSnapshot host = fx.session.settings().baseline();
    if (!arm.empty()) {
        for (const auto& kv : arm) {
            fx.session.settings().set(kv.first, kv.second,
                                      SourcePos("<ConsoleDeterminismValidation>", 0, 0));
        }
        fx.session.settings().set_baseline(fx.session.settings().snapshot());
    }
    ScriptRun r;
    {
        // The pipeline's own chatter (the legacy arm's [TIMING] lines, ...)
        // is not the transcript -- which goes to fx.out -- so it is swallowed
        // rather than left to flood the validation's log.
        ChatterCapture quiet(fx.session);
        r.code = RunFile(fx.session, path);
    }
    r.transcript = TranscriptOf(fx.out.str());
    fx.session.settings().set_baseline(host);
    return r;
}

ScriptRun RunScriptUnder(const std::string& path, const std::string& key, const Value& value) {
    Arm arm;
    if (!key.empty()) arm.emplace_back(key, value);
    return RunScriptUnderArm(path, arm);
}

std::string BaseName(const std::string& path) {
    return path.substr(path.find_last_of('/') + 1);
}

// --- (a) threads, and (b) backend ------------------------------------------

void TestThreadsAndBackend(const std::string& root) {
    const std::string dir = root + "/src/hacs/Console/scripts";
    std::vector<std::string> scripts;
    CollectScripts(dir, scripts);
    Check(!scripts.empty(), "the numbered script set is present");

    uint64_t d1 = engine::kFnvOffset64, d16 = engine::kFnvOffset64;
    uint64_t dlegacy = engine::kFnvOffset64, dengine = engine::kFnvOffset64;
    uint32_t c1_scripts = 0;
    // (S24 CL-L2) arm (b) needs `backend = "legacy"`, which an OC_ENABLE_LEGACY=OFF build refuses (exit 5).
    const bool arm_b = oc::legacy::RunLegacyArm("ConsoleDeterminismValidation", "b");
    for (const std::string& path : scripts) {
        const std::string name = BaseName(path);
        const ScriptRun t1 = RunScriptUnder(path, "threads", Value::from_int(1));
        const ScriptRun t16 = RunScriptUnder(path, "threads", Value::from_int(16));
        Check(t1.code == 0, "(a) " + name + " exits 0 under threads = 1");
        Check(t16.code == 0, "(a) " + name + " exits 0 under threads = 16");
        const bool same = (t1.transcript == t16.transcript);
        Check(same, "(a) " + name + ": the threads = 1 and threads = 16 transcripts are identical");
        if (!same) {
            std::cout << "    --- threads = 1 ---\n" << t1.transcript
                      << "    --- threads = 16 ---\n" << t16.transcript << std::flush;
        }
        d1 = engine::FnvBytes(d1, t1.transcript);
        d16 = engine::FnvBytes(d16, t16.transcript);

        // (b) The C1 script set: 001-016 (C1a + C1b).  017+ are C5's early
        // scripts, C2a's and C2b's.
        const int number = std::atoi(name.substr(0, 3).c_str());
        if (number < 1 || number > 16) continue;
        ++c1_scripts;
        if (!arm_b) continue;
        const ScriptRun leg = RunScriptUnder(path, "backend", Value::from_string("legacy"));
        Check(leg.code == 0, "(b) " + name + " exits 0 under backend = legacy");
        uint32_t hash_lines = 0;
        for (const std::string& line : SplitLines(t16.transcript)) {
            if (line.find("0x") != std::string::npos) ++hash_lines;
        }
        const bool same_b = (leg.transcript == t16.transcript);
        Check(same_b, "(b) " + name + ": the backend = legacy transcript -- every hash() line "
                      "included (" + std::to_string(hash_lines) + ") -- equals the engine arm's");
        if (!same_b) {
            std::cout << "    --- legacy ---\n" << leg.transcript
                      << "    --- engine ---\n" << t16.transcript << std::flush;
        }
        dlegacy = engine::FnvBytes(dlegacy, leg.transcript);
        dengine = engine::FnvBytes(dengine, t16.transcript);
    }
    std::cout << "  (a) threads: " << scripts.size() << " scripts, arm threads=1 digest "
              << HexU64(d1) << ", arm threads=16 digest " << HexU64(d16)
              << (d1 == d16 ? " (identical)" : " (DIFFER)") << std::endl;
    if (arm_b) {
        std::cout << "  (b) backend: " << c1_scripts << " C1 scripts, legacy digest "
                  << HexU64(dlegacy) << ", engine digest " << HexU64(dengine)
                  << (dlegacy == dengine ? " (identical)" : " (DIFFER)") << std::endl;
    }
    Check(d1 == d16, "(a) the two arms' transcript digests are equal");
    Check(c1_scripts == 16, "(b) the C1 script set is 001-016");
    if (arm_b) Check(dlegacy == dengine, "(b) the two arms' transcript digests are equal");
}

// --- (C4 a) the full product: threads x program_cache x device ---------------
//
// [P] Sec. 10 C4 acceptance (a): "the C1 script set under threads in {1, 4, 16,
// 32} x program_cache in {on, off} x device in {cpu, hybrid, gpu} (device arms
// [SKIP] without a device, like every Phase 5 validation) produces one
// identical transcript digest."  24 arms (8 without a device), each the 16 C1
// scripts on fresh sessions whose BASELINE carries the arm's three values (so
// a script's reset_settings() returns to the arm, not to the host).  Every
// script must exit 0 and every arm's digest -- FNV-1a-64 over the 16
// transcripts in order -- must equal the first arm's; a differing script is
// named and both transcripts printed.
//
// CONSOLE_DETERMINISM_ARMS=threads (read-only, not an emitter) restricts the
// product to its thread arms (program_cache = on, device = cpu: 4 arms) --
// the subset the CheckThread acceptance item names, which under TSan would
// otherwise cost the whole product.  Unset, the full product runs.

bool IsC1Script(const std::string& path) {
    const int number = std::atoi(BaseName(path).substr(0, 3).c_str());
    return number >= 1 && number <= 16;
}

void TestArmProduct(const std::string& root) {
    const std::string dir = root + "/src/hacs/Console/scripts";
    std::vector<std::string> all, c1;
    CollectScripts(dir, all);
    for (const std::string& p : all) {
        if (IsC1Script(p)) c1.push_back(p);
    }
    Check(c1.size() == 16, "(C4 a) the C1 script set is 001-016");

    const char* only = std::getenv("CONSOLE_DETERMINISM_ARMS");
    const bool threads_only = (only != nullptr && std::string(only) == "threads");
    const bool device = engine::tape::DeviceBackend<uint32_t>::usable();
    const std::vector<std::int64_t> threads = {1, 4, 16, 32};
    const std::vector<bool> caches = threads_only ? std::vector<bool>{true}
                                                  : std::vector<bool>{true, false};
    const std::vector<std::string> devices =
        threads_only ? std::vector<std::string>{"cpu"}
                     : std::vector<std::string>{"cpu", "hybrid", "gpu"};

    bool have_ref = false;
    uint64_t ref_digest = 0;
    std::string ref_label;
    std::vector<std::string> ref_transcripts;
    uint32_t arms_run = 0, arms_skipped = 0, arms_equal = 0;
    for (const std::string& dev : devices) {
        if (dev != "cpu" && !device) {
            arms_skipped += static_cast<uint32_t>(threads.size() * caches.size());
            continue;
        }
        for (const bool cache : caches) {
            for (const std::int64_t t : threads) {
                const std::string label = "threads=" + std::to_string(t) + " program_cache=" +
                                          (cache ? "on" : "off") + " device=" + dev;
                Arm arm;
                arm.emplace_back("threads", Value::from_int(t));
                arm.emplace_back("program_cache", Value::from_bool(cache));
                arm.emplace_back("device", Value::from_string(dev));
                const auto a0 = std::chrono::steady_clock::now();
                const engine::tape::TapeElimStats elim0 = engine::tape::tape_elim_totals();
                uint64_t digest = engine::kFnvOffset64;
                std::vector<std::string> transcripts;
                for (std::size_t i = 0; i < c1.size(); ++i) {
                    const ScriptRun r = RunScriptUnderArm(c1[i], arm);
                    Check(r.code == 0, "(C4 a) " + BaseName(c1[i]) + " exits 0 under " + label);
                    digest = engine::FnvBytes(digest, r.transcript);
                    if (have_ref && r.transcript != ref_transcripts[i]) {
                        Check(false, "(C4 a) " + BaseName(c1[i]) + ": the " + label +
                                         " transcript equals the " + ref_label + " one");
                        std::cout << "    --- " << ref_label << " ---\n" << ref_transcripts[i]
                                  << "    --- " << label << " ---\n" << r.transcript
                                  << std::flush;
                    }
                    transcripts.push_back(r.transcript);
                }
                const double secs =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count();
                ++arms_run;
                if (!have_ref) {
                    have_ref = true;
                    ref_digest = digest;
                    ref_label = label;
                    ref_transcripts = std::move(transcripts);
                }
                const bool same = (digest == ref_digest);
                if (same) ++arms_equal;
                Check(same, "(C4 a) arm " + label + ": digest " + HexU64(digest) +
                                " equals the reference arm's " + HexU64(ref_digest));
                std::cout << "  (C4 a) " << std::left << std::setw(46) << label << " digest "
                          << HexU64(digest) << (same ? "" : " (DIFFERS)") << "  " << std::fixed
                          << std::setprecision(2) << secs << " s" << std::endl;
                // (hecke-extensions slot 9, WP-6) non-vacuity: the elimination
                // driver's per-panel programs under this arm's device setting.
                const engine::tape::TapeElimStats elim1 = engine::tape::tape_elim_totals();
                std::cout << "           elim panel programs +" << (elim1.programs - elim0.programs)
                          << ", compiled under a device policy +"
                          << (elim1.device_policy_programs - elim0.device_policy_programs)
                          << ", with GPU nodes +" << (elim1.gpu_programs - elim0.gpu_programs)
                          << " (" << (elim1.gpu_nodes - elim0.gpu_nodes) << " nodes)" << std::endl;
            }
        }
    }
    if (arms_skipped > 0) {
        std::cout << "  [SKIP] (C4 a) " << arms_skipped
                  << " device arms (hybrid / gpu): no usable CUDA device" << std::endl;
    }
    std::cout << "  (C4 a) product: " << arms_run << " arms run"
              << (threads_only ? " (CONSOLE_DETERMINISM_ARMS=threads)" : "") << ", "
              << arms_skipped << " skipped, " << arms_equal << " / " << arms_run
              << " with the ONE digest " << HexU64(ref_digest) << " over " << c1.size()
              << " C1 scripts" << std::endl;
}

// --- (c) run_validation from inside a script --------------------------------

void TestRunValidation() {
    const hecke::ScopedMathSettings restore(hecke::math_settings());
    // The HOST's engine state -- what a lifted scope must show.
    engine::tape::ThreadPool* const host_pool = engine::tape::consumer_pool_override();
    const bool host_opts = engine::tape::pass_options_overridden();

    Fixture fx;
    bool probed = false;
    bool probe_pool_is_host = false;
    bool probe_opts_is_host = false;
    bind::SetHarnessLiftProbe([&]() {
        probed = true;
        probe_pool_is_host = (engine::tape::consumer_pool_override() == host_pool);
        probe_opts_is_host = (engine::tape::pass_options_overridden() == host_opts);
    });
    // (slot-14 WP-2, [C4] Sec.2 (c)) The script must `set device` to a value
    // that installs a REAL override -- i.e. one that differs from the
    // environment's own effective policy. A hard-coded "hybrid" is a no-op
    // override (and so `pass_options_overridden()` reads false afterwards,
    // failing the very check below) whenever the process runs under
    // ENGINE_DEVICE=hybrid to begin with (found on the cluster: WP-C4 Sec.2,
    // reproduced cross-architecture by WP-C7). Picking the value at runtime
    // from `from_env()`'s effective policy makes the script always install a
    // genuine override, in every environment.
    const engine::tape::DevicePolicy env_eff =
        engine::tape::PassOptions::from_env().effective_policy();
    const std::string device_val =
        (env_eff == engine::tape::DevicePolicy::Hybrid) ? "cpu" : "hybrid";
    const std::string src =
        "set threads = 3;\n"
        "set device = \"" + device_val + "\";\n"
        "rc = run_validation(\"GoldenRegressionValidation\");\n"
        "print(\"rc =\", rc);\n"
        "assert(rc == 0, \"GoldenRegressionValidation passes from inside a script\");\n"
        "assert(settings().threads.value == 3, \"the session's threads are back\");\n"
        "assert(settings().device.value == \"" + device_val + "\", \"the session's device is back\");\n";
    const int code = RunSource(fx.session, src, "<acceptance (c)>");
    // Read the engine state BEFORE the session ends (its destructor resets).
    engine::tape::ThreadPool* const after_pool = engine::tape::consumer_pool_override();
    const bool after_opts = engine::tape::pass_options_overridden();
    bind::SetHarnessLiftProbe(std::function<void()>());
    const std::string text = fx.out.str();

    Check(code == 0, "(c) the script exits 0 (run_validation(\"GoldenRegressionValidation\") == 0)");
    Check(text == "rc = 0\n",
          "(c) the transcript is exactly the script's own line -- the validation's chatter "
          "is captured");
    if (text != "rc = 0\n") std::cout << "    --- transcript ---\n" << text << std::flush;
    Check(probed, "(c) the lift probe ran inside run_validation");
    Check(probe_pool_is_host, "(c) inside the run, the consumer pool is the HOST's, not threads = 3");
    Check(probe_opts_is_host,
          "(c) inside the run, the pass options are the HOST's, not device = hybrid");
    Check(after_pool != host_pool && after_pool != nullptr && after_pool->size() == 3,
          "(c) after the run, the session's threads = 3 pool is in force again");
    Check(after_opts, "(c) after the run, the session's device override is in force again");
    std::cout << "  (c) run_validation(GoldenRegressionValidation) = 0 from a script; lifted: pool "
              << (probe_pool_is_host ? "host" : "SESSION") << ", pass options "
              << (probe_opts_is_host ? "host" : "SESSION") << std::endl;
}

// --- (d) run_campaign ------------------------------------------------------

// summary.tsv with its two volatile columns (time_s, rss_mb) blanked -- the
// CampaignSmokeValidation rule (columns 18 and 19 of the header).
std::string StripTimingColumns(const std::string& tsv) {
    std::ostringstream out;
    std::istringstream in(tsv);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string field;
        uint32_t col = 0;
        while (std::getline(ls, field, '\t')) {
            if (col) out << "\t";
            out << ((col == 18 || col == 19) ? std::string("-") : field);
            ++col;
        }
        out << "\n";
    }
    return out.str();
}

void TestRunCampaign() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    const std::string root =
        ((ec ? fs::path(".") : base) / ("oc_console_c2b_" + std::to_string(::getpid()))).string();
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    const std::string dir_driver = root + "/driver";
    const std::string dir_console = root + "/console";

    std::string tsv_driver, tsv_console;
    {
        std::istringstream is(CampaignSmokeSpecText());
        campaign::CampaignSpec spec = campaign::ParseCampaignSpec(is, "<smoke>");
        spec.output_dir = dir_driver;
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        Check(res.num_failed == 0, "(d) the driver runs the smoke spec with no failed job");
        ReadFile(dir_driver + "/summary.tsv", tsv_driver);
    }
    {
        const hecke::ScopedMathSettings restore(hecke::math_settings());
        Fixture fx;
        fx.session.set_arg("spec", Value::from_string(CampaignSmokeSpecText()));
        fx.session.set_arg("out", Value::from_string(dir_console));
        const std::string src =
            "S = campaign_spec_text(args.spec);\n"
            "R = run_campaign(S, output = args.out);\n"
            "assert(R.failed == 0, \"no job failed\");\n"
            "assert(R.ran == S.jobs, \"every planned job ran\");\n"
            "print(\"run_campaign:\", R.ran, \"jobs\");\n";
        const int code = RunSource(fx.session, src, "<acceptance (d)>");
        Check(code == 0, "(d) run_campaign on the smoke spec exits 0 from a script");
        Check(fx.out.str() == "run_campaign: 20 jobs\n",
              "(d) the script's transcript is its own line (the driver's chatter captured)");
        if (code != 0) std::cout << fx.out.str() << std::flush;
        ReadFile(dir_console + "/summary.tsv", tsv_console);
    }
    Check(!tsv_driver.empty() && !tsv_console.empty(), "(d) both runs wrote a summary.tsv");
    const bool same = StripTimingColumns(tsv_driver) == StripTimingColumns(tsv_console);
    Check(same, "(d) run_campaign's summary.tsv equals the driver's outside the timing columns");
    std::cout << "  (d) run_campaign(smoke spec): summary.tsv "
              << SplitLines(tsv_console).size() << " lines, "
              << (same ? "identical" : "DIFFERENT") << " to RunCampaignSpec's outside time_s/rss_mb"
              << std::endl;
    fs::remove_all(root, ec);
}

// --- (e) bench ---------------------------------------------------------------

void TestBench() {
    const hecke::ScopedMathSettings restore(hecke::math_settings());
    Fixture fx;
    const std::string src =
        "B = bench(\"hecke\", n = 48, ell = 134217757, p = 2);\n"
        "H = cohomology(symmetric_power(sl2z_model(), 48, 134217757));\n"
        "assert(B.dims == H.dims, \"bench dims == cohomology dims\");\n"
        "print(B.dims);\n"
        "bench_line(B);\n";
    const int code = RunSource(fx.session, src, "<acceptance (e)>");
    Check(code == 0, "(e) bench(\"hecke\", n = 48, ell = 134217757, p = 2).dims == "
                     "cohomology(...).dims");
    const std::vector<std::string> lines = SplitLines(fx.out.str());
    Check(lines.size() == 2 && lines[1].rfind("#~ BENCH mode=hecke n=48 ell=134217757 p=2 ", 0) == 0,
          "(e) bench_line prints the BENCH line as a `#~` annotation");
    if (code != 0 || lines.size() != 2) std::cout << fx.out.str() << std::flush;
    std::cout << "  (e) " << (lines.empty() ? std::string("?") : lines[0])
              << " == cohomology(...).dims" << std::endl;
}

// --- `--log` in script mode ---------------------------------------------------

void TestLog() {
    const hecke::ScopedMathSettings restore(hecke::math_settings());
    Fixture fx;
    std::ostringstream log;
    fx.session.set_log(&log);
    const std::string src =
        "# a comment before the first statement\n"
        "x = 2;\n"
        "print(\"x =\", x);\n"
        "\n"
        "# a comment between statements\n"
        "for i in [1, 2] {\n"
        "    print(i * x);\n"
        "}\n"
        "x + 1\n";
    const int code = RunSource(fx.session, src, "<log>");
    fx.session.set_log(nullptr);
    Check(code == 0, "--log: the script runs");
    std::string inputs, outputs;
    for (const std::string& line : SplitLines(log.str())) {
        if (line.rfind("> ", 0) == 0) inputs += line.substr(2) + "\n";
        else outputs += line + "\n";
    }
    Check(inputs == src, "--log: the `> ` lines, prefix stripped, are the script, line for line");
    Check(outputs == fx.out.str(), "--log: the other lines are the transcript");
    Check(log.str() ==
              "> # a comment before the first statement\n> x = 2;\n> print(\"x =\", x);\nx = 2\n"
              "> \n> # a comment between statements\n> for i in [1, 2] {\n>     print(i * x);\n"
              "> }\n2\n4\n> x + 1\n3\n",
          "--log: each input chunk is logged just before its output");
}

// --- G9 ----------------------------------------------------------------------

void TestDickson() {
    Fixture fx;
    std::vector<Value> want;
    for (uint32_t ell : {2u, 3u, 5u, 7u, 11u, 13u}) {
        for (uint32_t n = 0; n <= 60; ++n) {
            want.push_back(Value::from_bool(hecke::IsDicksonExceptional(n, ell)));
        }
    }
    fx.session.set_arg("want", Value::from_list(want));
    const std::string src =
        "i = 0;\n"
        "bad = 0;\n"
        "for ell in [2, 3, 5, 7, 11, 13] {\n"
        "    for n in range(0, 61) {\n"
        "        if dickson_exceptional(n, ell) != args.want[i] { bad = bad + 1; }\n"
        "        i = i + 1;\n"
        "    }\n"
        "}\n"
        "print(i, bad);\n";
    const int code = RunSource(fx.session, src, "<G9>");
    Check(code == 0, "G9: the dickson_exceptional loop runs");
    Check(fx.out.str() == std::to_string(want.size()) + " 0\n",
          "G9: dickson_exceptional(n, ell) == hecke::IsDicksonExceptional(n, ell) on " +
              std::to_string(want.size()) + " cells");
    if (fx.out.str() != std::to_string(want.size()) + " 0\n") std::cout << fx.out.str();
}

} // namespace

uint32_t ConsoleDeterminismValidation() {
    g_checks = 0;
    g_failures = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "--- ConsoleDeterminismValidation (console C2b + C4, [P] Sec. 10 C2 (a)-(e), C4 (a)) ---"
              << std::endl;
    const std::string root = FindRepoRoot();
    if (root.empty()) {
        std::cout << "  [SKIP] the script arms: could not locate the source tree" << std::endl;
    } else {
        TestThreadsAndBackend(root);
        TestArmProduct(root);
    }
    TestRunValidation();
    TestRunCampaign();
    TestBench();
    TestLog();
    TestDickson();
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  ConsoleDeterminismValidation " << (g_failures == 0 ? "PASSED" : "FAILED")
              << " (" << g_checks << " checks, " << g_failures << " failing, " << std::fixed
              << std::setprecision(1) << wall << " s)" << std::endl;
    return g_failures;
}
