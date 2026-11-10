// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef BENCH_DRIVER_H
#define BENCH_DRIVER_H

// ===========================================================================
// Phase 0 -- in-repo benchmark CLI (`./cohomology_app --bench <mode> <args>`).
//
// Productisation of the scratch `bench_driver.cpp` of
// docs/matrix-engine/03-profiling-baseline.md Sec. 1.3, whose reason for
// existing is recorded there verbatim: "there is no supported way to sweep a
// parameter from the CLI". Every exit criterion of the matrix-engine plan's
// Phase 0..P6 is a `--bench` command, so the driver is part of the tree
// rather than a scratch artefact.
//
// PLACEMENT (per the plan's Phase-0 scope note, mechanic (ii)): this is a
// plain C++ translation unit, NOT part of main.cpp. `main.cpp` is an nvcc
// TU that inherits neither CMAKE_CXX_FLAGS_<CONFIG> nor -march=native, so
// keeping the driver in a .cpp file is what makes the benchmark numbers
// reflect the C++ flags actually under study. main.cpp only dispatches.
//
// The driver calls the PUBLIC API only, using the same construction recipe as
// every V-stage validation and as doc 03 Sec. 1.3:
//   CreateSL2ZFundamentalDomainTriangulation()
//     -> GenerateSL2ZRelationsWithGroupData()
//     -> LocalSystemSimplicialComplex
//     -> EquivariantCohomologyOnK::compute_cohomology()
//     -> HeckeCosetTables -> HeckeTauTable -> HeckeOperatorOnK::compute()
//     -> HeckeOperatorFamily / HeckeModuleDecomposition / HeckeCrossChecks
// It never touches arithmetic or pipeline internals, so `--bench` numbers and
// `--all` results come from exactly the same code paths.
// ===========================================================================

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace bench {

// ---------------------------------------------------------------------------
// (hecke-extensions slot 7, WP-8 / console C2b, gap G4 of
// docs/console/00-console-interface-plan.md Sec. 14.2) The per-mode runner,
// PUBLIC.  Until slot 7 only the argv entry RunBench() was public and the
// runners sat in an anonymous namespace, so the console's `bench(...)` /
// `micro(...)` verbs would have had to fake an argv.  The CLI is now a thin
// parser in front of RunMode(): RunBench() parses exactly as before, fills a
// ModeArgs and returns RunMode(args).exit_code.
//
// BYTE IDENTITY.  RunMode() prints to std::cout exactly what `--bench` printed
// before the extraction -- the `=== bench ... ===` header, every phase line,
// the mode-specific lines and the closing `BENCH ...` summary line -- so the
// CLI output is unchanged (WP-8's byte-identity gate, every mode, timing
// columns normalised; ConsoleCliPreservationValidation re-runs the frozen
// `--bench` forms).  The console verbs call it under the session's chatter
// capture, so that text never reaches a transcript; what they use is the
// returned BenchRecord.
// ---------------------------------------------------------------------------
struct ModeArgs {
    // "cohom" | "hecke" | "family" | "decomp" | "ext" | "micro"
    std::string mode;
    uint32_t n_sym = 0;
    uint32_t ell = 0;
    uint64_t p = 0;                   // hecke / family / decomp
    uint32_t r = 1;                   // family
    std::vector<uint64_t> family;     // ext (the CLI default is 2,3,5)
    // micro only: the op name and its positional integer arguments, exactly
    // as `--bench micro <op> <args...>` takes them.
    std::string op;
    std::vector<uint32_t> op_args;
    // micro only: "legacy" | "engine" | "gpu"; empty = the ENGINE_BACKEND
    // default the CLI uses (micro::BackendFromEnv()).
    std::string backend;
    uint32_t repeats = 1;
    bool cross_checks = false;        // hecke
    bool lean = false;
    bool count_allocs = false;

    // (hecke-extensions slot 7, WP-6 / E2 (e')) ext only, OUTSIDE the frozen
    // `--bench --help` text (`cli_frozen/bench_help.stdout` must not move):
    // `--campaign-cell` turns both on and flips the CLI's own default family
    // fallback from 2,3,5 to 2,3,5,7, so `--bench ext <n> <ell> --campaign-cell`
    // (no explicit family) runs the campaign cell's driver options -- family
    // {2,3,5,7}, with_cone, with_sign_split -- exactly as CampaignExt.cpp's
    // opts.with_cone / opts.with_sign_split (both true at the default spec,
    // CampaignSpec.h ExtSpec::with_cone()/with_sign_split()).  The record's
    // canonical split and rendering need no flag: since slot-7 WP-3 item 6 it
    // runs unconditionally inside HeckeExtDriver::run() (RunExt's
    // canonical_field_s_out).  See docs/performance-reference.md.
    bool with_cone = false;
    bool with_sign_split = false;
    // "auto" | "off" | "budget_mb=<M>" (Hecke/HeckeSettings.h ParseUEll);
    // empty = leave the process-wide u_ell setting exactly where it is (the
    // default is itself "auto", which is also ext_first.spec's value -- [S7]
    // Sec 6 confirmation 9 -- so an indicative or W3 run against that spec
    // needs no override).  Scoped for the duration of the run only, stamped
    // SettingSource::Spec, exactly as CampaignSpec.cpp stamps a spec's own
    // `u_ell = ...` line.
    std::string u_ell;
};

struct BenchRecord {
    // RunBench()'s process exit code for the same arguments: 0 success, 1 a
    // usage error, 2 an exception or a micro-op whose backend is unavailable.
    int exit_code = 0;
    std::string error;                // the message printed with exit_code != 0
    std::string mode;
    std::string params;               // "n=48 ell=134217757 p=2" / "op=gemm n=512 ..."
    std::vector<uint32_t> dims;       // dim H^q_eq, q = 0.. (pipeline modes)
    // The phase split of the LAST repeat, in execution order (pipeline modes).
    std::vector<std::pair<std::string, double>> phases;
    std::vector<double> totals;       // one per repeat
    double total_s = 0.0;             // the BENCH line's total_s (min over repeats)
    double median_s = 0.0;
    double rss_mb = -1.0;             // the BENCH line's rss_mb; -1 = n/a
    // The BENCH line's mode-specific key=value pairs after rss_mb, in order.
    std::vector<std::pair<std::string, std::string>> extra;
    std::string bench_line;           // the whole `BENCH ...` line, no newline
    uint64_t checksum = 0;            // micro: the FNV-1a-64 result checksum
};

// Runs one benchmark.  See the block comment above.  Never throws for a
// pipeline exception: that is exit_code 2 with `error` set, as on the CLI.
BenchRecord RunMode(const ModeArgs& args);

// The `BENCH ...` summary line, as a function of its fields:
//   "BENCH mode=<mode>[ <params>] total_s=<%.4f> rss_mb=<%.1f | n/a>[ <extra>]"
// (rss_mb < 0 prints "n/a").  Every `--bench` line is formatted by it, and so
// is the console's `bench_line(timed(...))`, so the two cannot drift.
std::string FormatBenchLine(const std::string& mode, const std::string& params, double total_s,
                            double rss_mb, const std::string& extra);

// The process-wide totals of the counting global operator new that backs
// `--count-allocs` (every allocation of the process, relaxed atomics).  A
// caller snapshots them around a region and reports the delta, as `--bench`
// does per repeat and the console's `timed(...)` does per call.
std::size_t ProcessAllocCount();
std::size_t ProcessAllocBytes();

// Runs the benchmark CLI. `argc`/`argv` are the arguments AFTER the `--bench`
// token, i.e. argv[0] is the mode name ("cohom", "hecke", "family",
// "decomp", "micro"). Returns a process exit code: 0 on success, 1 on a usage
// error, 2 on an exception escaping the measured pipeline or on a micro-op
// whose selected backend is not implemented.
//
// Phase 2 adds the `micro` mode -- arithmetic micro-benchmarks
// (gemm/gemv/rref/kernel/transpose) plus the `suite` that prints the plan's
// Phase-2 exit criteria next to the measurement. Its measurement core is
// src/ModArithmetic/PerformanceSmokeValidation.{h,cpp}, NOT this
// file: a validation registered inside the ModArithmetic static library must
// be able to call it, and that library cannot depend on cohomology_app. See
// that header for the rationale and for the five-line recipe that wires the
// `--backend engine` path once the new kernels exist.
int RunBench(int argc, char** argv);

// Prints the `--bench` usage block (also reachable via `--bench --help`).
void PrintBenchUsage(const char* argv0);

} // namespace bench

#endif // BENCH_DRIVER_H
