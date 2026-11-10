// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CAMPAIGN_DRIVER_H
#define CAMPAIGN_DRIVER_H

// ===========================================================================
// matrix-engine Phase 6 -- the campaign driver (`./cohomology_app --campaign
// <spec-file>`). Documentation: docs/matrix-engine/campaign.md.
//
// WHAT IT IS. `--bench hecke <n> <ell> <p>` runs ONE point and prints timings.
// The campaign driver runs a SWEEP of points and writes down the mathematics:
// one streamed JSON per job with the canonical objects and their FNV-1a-64
// hashes, the full HeckeReport levels (a)/(b)/(c), the certificates that were
// executed, timings and peak RSS -- plus a campaign-level index.json and a
// summary.tsv, which is the "published table of new char-polys" of the plan's
// Phase 6 exit criterion.
//
// PLACEMENT. Like Bench/BenchDriver.cpp and Golden/GoldenHarness.cpp this is a
// plain C++ translation unit driving the PUBLIC API only -- the same recipe
// every V-stage validation uses:
//   CreateSL2ZFundamentalDomainTriangulation() -> GenerateSL2ZRelationsWithGroupData()
//     -> LocalSystemSimplicialComplex -> EquivariantCohomologyOnK::compute_cohomology()
//     -> HeckeCosetTables -> HeckeTauTable -> HeckeOperatorOnK::compute()
//     -> HeckeOperatorFamily / HeckeModuleDecomposition / HeckeCrossChecks
// so campaign numbers and `--all` results come from the same code paths.
// main.cpp only dispatches into RunCampaign().
//
// HASHING. Byte-for-byte the FNV-1a-64 of Golden/GoldenHarness.cpp (offset
// 0xcbf29ce484222325, prime 0x100000001b3; rows, cols, then every entry
// widened to uint64 in row-major order), and the object / point NAMES are the
// golden ones too -- so a campaign point that also lies on the golden grid
// must reproduce the frozen GoldenReference.inc hash, and
// CampaignSmokeValidation asserts exactly that.
//
// CERTIFY SEMANTICS (binding, plan Phase 6 "Validations"):
//   * The IN-PROCESS certificates C3-C6 are NEVER sampled. They run on every
//     job in every mode, because they are not optional extras but part of what
//     compute() means:
//       C3  tau certification  -- HeckeTauTable's constructor proves
//                                 d.tau = tau.d exactly, or throws;
//       C4  O12(a)             -- the two Theta routes are compared over every
//                                 entry of every degree inside compute();
//       C5  A-N3 residual      -- the exact Theta_eq solve residual, recomputed
//                                 independently of the solver, asserted zero;
//       C6  decomposition      -- HeckeModuleDecomposition::certify() == 0 on
//                                 every reported degree.
//     Every job's JSON records their executed counts, so "did they run?" is a
//     machine-checkable property of the output, not a claim.
//   * The EXTERNAL cross-checks are the sampled ones -- they are validation
//     cost, not result (03 Sec. 4.2): HeckeCrossChecks (the classical
//     alpha/beta/omega three-map factorisation) and, for r >= 2,
//     HeckeOperatorFamily::T_p_pow_direct (the independent double-coset route
//     compared against the algebra recursion). Under `certify = full` they run
//     for every job; under `certify = sampled(...)` for the selected subset.
//     The mode AND the per-job decision are recorded in the JSON.
//
// MEMORY-LEAN MODE (`lean = true`). Two changes, both documented in the JSON:
//   (1) the O12(a) route-2 comparison runs one block row at a time
//       (HeckeOperatorOptions::ScopedMemoryLean) instead of holding a second
//       complete Theta;
//   (2) level (a) -- both its hash and its JSON -- is read block-row-wise off
//       HeckeOperatorOnK::Theta_block(q) through a hecke::LevelARowSource, so
//       the (N_q*d_V)^2 FlatMatrix of get_Theta_flat_matrices() is never
//       materialised (audit item 7.19).
// The COMPUTED VALUES are identical either way; only the peak footprint moves.
//
// THE EXTENSION RECORD (hecke-extensions slot 5, WP-2 / E2 cont.).  With
// `ext = true` on the spec the driver additionally runs the extension engine
// ONCE PER (n_sym, ell) CELL ([XA m14]: a job is (n_sym, ell, p, r), so
// running it inside RunOneJob would rebuild the same record once per p),
// SEQUENTIALLY ON THE CAMPAIGN'S MAIN THREAD and BEFORE the job loop, and
// attaches the record to the cell's FIRST job in plan order.  Campaign/
// CampaignExt.h owns that seam and states why the pass may not run on a job
// thread; here it is enough that `ext = false` -- the default -- leaves every
// byte of this driver's output unchanged, including summary.tsv's column set
// and every shape signature.
//
// THE BATCHING SEAM (Phase 6 Part A, done). The driver executes jobs in
// campaign::plan_campaign() order, i.e. grouped by shape signature
// (n_sym, p, r, degrees, model) with only `ell` varying inside a group.
// RunJobGroup() runs `spec.jobs` of a group's jobs CONCURRENTLY, each on its
// own engine ThreadPool of `spec.threads` workers (the 2-level split), with
//   * the launching thread's thread_local consumer options installed on every
//     job thread through engine::tape::capture_thread_context(), and
//   * every HeckeCosetTables / HeckeTauTable construction under the SAME
//     process-wide hecke::tau_construction_mutex() HeckeOperatorFamily uses
//     (tau is first-match-wins and drives the non-atomic PredicateStats --
//     audit 02 Sec. 6.5 hazards 1-3).
// Results land in a pre-sized slot vector and are appended in GROUP ORDER, so
// nothing observable depends on the schedule. `spec.jobs == 1` takes a
// separate branch that is the pre-Phase-6 sequential loop unchanged.
// ===========================================================================

#include "CampaignSpec.h"
#include "CampaignExt.h"

#include <cstdint>
#include <string>
#include <vector>

namespace campaign {

// One canonical object and its hash, in the Golden/GoldenHarness.cpp
// vocabulary (`point` / `object` / `rows` / `cols` / `hash`; for a scalar
// invariant `rows` IS the value and `cols` is 0).
struct CanonicalObject {
    std::string point;
    std::string object;
    char tier = 'A';       // 'A' basis-independent, 'B' basis-dependent
    uint64_t rows = 0;
    uint64_t cols = 0;
    uint64_t hash = 0;
};

// What one job produced. Returned in memory as well as written to JSON, so a
// validation can compare two runs without parsing anything.
struct JobResult {
    Job job;
    bool ran = false;          // false when `resume` skipped it
    bool ok = false;
    std::string error;         // non-empty iff the job threw
    std::string output_path;

    std::vector<uint32_t> dim_H;
    std::vector<CanonicalObject> objects;
    // Characteristic polynomial of the job's cohomology-level operator per
    // degree q -- A^{(q)}(T_p) for r = 1, the recursion's T_{p^r} for r >= 2 --
    // as coefficients LOW -> HIGH degree (amendment A-N4), semicolon
    // separated. This is what summary.tsv publishes.
    std::vector<std::string> charpoly;

    // --- in-process certificates C3-C6 (never sampled) ---
    bool tau_certified = false;          // C3: the tau constructor returned
    bool routes_bit_identical = false;   // C4
    uint64_t relation_checks = 0;        // runtime assertion (i)
    uint64_t cocycle_checks = 0;         // runtime assertion (ii)
    uint64_t theta_eq_solves = 0;        // runtime assertion (iii)
    uint64_t theta_eq_residual_max = 0;  // C5: always 0
    uint64_t decomp_certified = 0;       // C6: degrees whose certify() == 0
    uint64_t decomp_certify_failures = 0;

    // --- external cross-checks (sampled) ---
    bool external_selected = false;
    bool external_ran = false;
    uint32_t external_rc = 0;
    std::string external_note;

    // --- bookkeeping ---
    double t_setup = 0, t_cohomology = 0, t_cosets = 0, t_tau = 0, t_theta = 0;
    double t_family = 0, t_decomp = 0, t_external = 0, t_report = 0, t_total = 0;
    double peak_rss_mb = -1.0;
    std::string peak_rss_scope = "process";
    bool lean = false;
    std::string level_a_state = "omitted";

    // --- (slot-5 WP-2 / E2 cont.) the extension record ------------------
    // True only for the job that OWNS its (n_sym, ell) cell's ext record
    // ([XA m14]); `ext` is then that cell's summary.  Both are untouched
    // when the spec does not set `ext = true`.
    bool           ext_owner = false;
    ExtCellSummary ext;

    // --- (slot-9 WP-8, C5c) a job under a non-default coefficient_ring ---
    // (Campaign/CampaignRing.h; [E] sec:ring (g)).  All untouched -- and
    // unread -- in a fingerprint-0 campaign.
    bool ring_job = false;
    std::string ring_status;                  // (g14): "ok" | "ok_partial" | "failed"
    std::vector<std::string> ring_refusals;   // (g13)'s refused parts, in order
    std::vector<std::string> ring_columns;    // summary.tsv's 15 appended values
};

struct CampaignResult {
    std::string output_dir;
    std::vector<JobGroup> groups;
    std::vector<JobResult> jobs;     // in execution order
    uint64_t num_ran = 0, num_skipped = 0, num_failed = 0;
    double wall_s = 0.0;

    // --- Phase 6 Part A: what the 2-level split actually was ---
    uint32_t jobs_concurrency = 1;   // concurrent jobs per shape group
    uint32_t threads_per_job = 0;    // 0 = the process pool was shared
    bool per_job_pools = false;      // false: the override is not thread-local
    double peak_rss_mb = -1.0;       // the whole campaign's process VmHWM
    double jobs_per_minute = 0.0;    // ran / wall_s * 60

    // --- (slot-5 WP-2 / E2 cont.) what the ext pass did ------------------
    bool     ext_enabled = false;
    uint64_t ext_cells = 0, ext_ran = 0, ext_reused = 0, ext_resumed = 0, ext_failed = 0;
    double   ext_wall_s = 0.0;
};

// Runs a parsed spec: creates the output directory, executes every job group
// in plan order, streams one JSON per job, then writes index.json and
// summary.tsv. Never throws for a per-job failure (it is recorded in the
// JobResult and the job's JSON); throws only for a spec/IO error that makes
// the campaign impossible.
CampaignResult RunCampaignSpec(const CampaignSpec& spec, bool verbose = true);

// (slot-8 WP-6) Whether engine::tape::set_consumer_pool_override() is
// thread-local, probed once at run time: the condition under which a
// concurrent job -- or a parallel ext-pass worker -- gets its own engine pool
// of `threads` workers instead of sharing the process pool.
bool PerJobPoolsSupported();

// A one-job spec with the same campaign-wide settings (used by
// CampaignSmokeValidation to re-run a single point in isolation, and handy for
// re-running one failed point of a big campaign).
CampaignSpec SingleJobSpec(const CampaignSpec& spec, const Job& job,
                           const std::string& output_dir);

// True iff `path` exists and carries the completion trailer -- the `resume`
// predicate. A half-written file (killed mid-job) has no trailer and is re-run.
bool JobOutputComplete(const std::string& path);

// The CLI. argv[0] is the first argument AFTER `--campaign`.
// Returns 0 on success, 1 on a usage/spec error, 2 if any job failed.
int RunCampaign(int argc, char** argv);
void PrintCampaignUsage(const char* argv0);

} // namespace campaign

#endif // CAMPAIGN_DRIVER_H
