// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CAMPAIGN_EXT_H
#define CAMPAIGN_EXT_H

// ===========================================================================
// hecke-extensions phase E2 cont. (slot 5, WP-2) -- THE CAMPAIGN'S EXTENSION
// SEAM: the ext record on the campaign, per (n_sym, ell) CELL.
//
// [X] = docs/implementation/hecke_extensions_implementation_plan.md v1.8
// Sec. 1.4 (the writer and its standalone path), Sec. 1.5 (the spec keys, the
// per-cell grouping, the summary.tsv columns), Sec. 1.3.1 R3.3 item 4 (the
// `u_ell` resume); [E] = docs/latex/hecke_extensions.tex v1.5.2 Sec. 7;
// [C] = docs/latex/gl2_contract.tex v1.6.2, cited by label.
//
// WHY A SEPARATE TU.  Three reasons, in order of weight:
//
//  (1) THE EXT ENGINE IS SINGLE-THREADED AND MUST NOT RUN ON A JOB THREAD.
//      HeckeExtDriver's own header says so ("the concurrency in a campaign is
//      one driver per job"), and it is not a style preference: `Impl`'s
//      constructor builds a SimplicialComplex and an SL2ZTessellationTables
//      OUTSIDE hecke::tau_construction_mutex(), and that build drives the
//      NON-ATOMIC global PredicateStats (audit 02 Sec. 6.5 hazards 1-3).  A
//      campaign job thread holds that lock around its own table constructions,
//      so an unlocked ext build on a second thread would race it.  Holding the
//      lock around HeckeExtDriver::run() is NOT an option either: the driver
//      re-enters it (HeckeSturmSweep's streamed route, HeckeBoundary), and
//      std::mutex is not recursive -- it would deadlock.
//
//      So the ext pass runs on the campaign's MAIN THREAD, BEFORE the job
//      loop, and never on a job thread.  That makes the ext half of the
//      output independent of the `jobs x threads` split, which is the
//      determinism contract the campaign seam owes (CampaignDriver.h, "THE
//      BATCHING SEAM").
//
//      (slot-8 WP-6) What made the DRIVER single-threaded is gone: the
//      PredicateStats counters are thread_local (Hecke/Predicates.h, "PER
//      THREAD"), and the engine's elim / gemm diagnostic counters are
//      relaxed atomics.  Every other piece of state a cell touches is its
//      own (one driver = one K, one model, one cohomology, one family), and
//      the table CONSTRUCTIONS stay under hecke::tau_construction_mutex(),
//      held by the code that builds them, exactly as before.  So several
//      drivers -- one per cell -- may now run AT ONCE, each on its own
//      thread; see "THE PARALLEL PASS" below.  The pass as a whole still runs
//      before the job loop, never interleaved with it.
//
//  (2) IT KEEPS THE HeckeExt/ INCLUDE IN ONE PLACE.
//      check_heckeext_includes.sh rule 1 admits Campaign/; this is the only
//      Campaign/ TU that uses the admission.
//
//  (3) THE RECORD IS PER CELL, THE JOB IS PER (n, ell, p, r) ([XA m14]).
//      Running the engine inside RunOneJob would rebuild the same record once
//      per Hecke prime.
//
// WHAT IT WRITES.  Both halves [X] asks for, from ONE HeckeExtRecord:
//
//   * the STANDALONE record, `<output>/ext/hecke_ext_ell<ell>_n<n>.json`,
//     through hecke_ext::WriteHeckeExtReport -- which is the only form that
//     HAS a stem, hence the only one that can write the R3.1 item-6 state
//     SIDECARS beside it ([X] Sec. 1.4).  This file is also what the campaign
//     `resume` path reads back.
//   * the OWNER JOB's `"extensions"`, streamed by copying that file's bytes
//     verbatim ([X] Sec. 1.5, "the per-job JSON gains `extensions`"), followed
//     by `"extensions_file"` naming where the sidecars' relative paths are
//     taken from.  One record, two appearances, zero re-serialisation.
//
// THE RECORD-LEVEL `u_ell` RESUME ([X] Sec. 1.3.1 R3.3 item 4; slot-5 plan
// Sec. 2.3 item 8: "what is left is the CAMPAIGN-level path").  With
// `ext = true` and `resume = true` the pass reads the persisted record of each
// cell, verifies every `state_ref` sidecar it names (present, and FNV-1a-64 of
// the bytes as recorded), and then:
//
//     persisted verdict INCLUDED, or no `resumable`   -> reuse, a strict no-op
//     the spec's budget  <  the persisted budget      -> REFUSED BY NAME
//     the spec's budget  == the persisted budget      -> reuse
//     the spec's budget  >  the persisted budget      -> RESUME: the driver is
//         run under the PERSISTED budget and then resume_u_ell(new budget) --
//         i.e. slot-4 WP-2's own semantics, unchanged, so the resumed record
//         is the one a fresh run at the new budget produces (byte for byte
//         when the new budget is the shipped default, [WP4-2] finding 4)
//
// and a resumed cell forces its owner job to re-run, so the job file that
// carries `"extensions"` is rewritten with the continued record.  The record
// is read with a small depth-aware scanner: `"u_ell"` at depth 1 is the
// CELL's object, and the `"u_ell"` inside `"settings"` is at depth 2, so the
// two cannot be confused.
//
// PLACEMENT.  Campaign/, plain C++, uint32_t cells only (HeckeExtDriver is the
// 32-bit instantiation); a cell with ell >= 2^32 records
// `{"status":"unavailable"}` with its reason, under [C] `sec:recordschema`
// item 6(c).
// ===========================================================================

#include "CampaignSpec.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace hecke_ext { struct HeckeExtRecord; }

namespace campaign {

// THE PARALLEL PASS (hecke-extensions slot 8, WP-6; [S8] = the slot-8 plan,
// WP-6).  `ext_jobs` (spec key, default 1) cells run at once, each on its own
// worker thread with its own engine pool of `threads` workers (the job loop's
// model, CampaignDriver.cpp "PER-JOB ENGINE POOLS").
//
//   * THREAD CONTEXT.  Every worker installs the launching thread's context
//     through engine::tape::capture_thread_context() -- the MATHEMATICAL
//     SETTINGS among it ([S8] Sec. 2.3 item 10: a worker started any other
//     way would silently run at the defaults, which the default-settings
//     byte identity cannot see; HeckeExtCampaignValidation arm 10 (b) runs a
//     non-default setting for exactly that reason).
//   * ADMISSION.  A cell is admitted while the sum of the running cells'
//     predicted memory stays <= `ext_memory_mb` (default kExtMemoryMbDefault).
//     A cell's prediction is kExtCellFloorMb plus, when U_ell is admitted,
//     the frozen U_ell RSS model (HeckeUEllBudget.h).  A cell that exceeds
//     the cap alone runs when nothing else does, so the pass cannot stall.
//   * ORDER.  Longest processing time first (ExtLptOrder): the U_ell cells by
//     their predicted MB (linear in the coset count ell + 1), then the rest by
//     n, then the plan index -- so the pass wall approaches max(cell).
//   * DETERMINISM OF EVERY OUTPUT.  A cell is a pure function of (n_sym, ell,
//     family, spec, settings): the record FILE is written per cell by its
//     worker; `pass.cells` is an ordered std::map; the counters,
//     `force_rerun_job_ids` and the verbose lines are assembled in PLAN ORDER
//     after the join; the resume decision is DecideCell's, unchanged per cell.
//     `ext_jobs` and `ext_memory_mb` are SCHEDULING KNOBS: they enter neither
//     ExtSpec::signature_tag() (the `_x<16 hex>` file tag), nor any Job::id(),
//     nor index.json / summary.tsv (arm 10 (d) asserts it).
//   * `ext_jobs = 1` is the pre-WP-6 loop, cell by cell on the calling thread
//     and the process-wide pool, down to where a refusal throws; `ext_jobs >
//     1` decides every cell first, so a refusal throws before any engine runs.
//
// THE SEAM FOR WP-6b (the optional CPU + GPU dispatch).  Two points only: the
// worker's scheduling class (CampaignExt.cpp, ExtWorkerClass, installed at the
// top of the worker: a device-backed class would install a THREAD-LOCAL
// pass-options / device override there, after the captured context), and
// ExtCellScheduler::take() (a device worker would take cells by its own rule
// -- the cheapest non-U_ell cells, only while every CPU slot is busy -- from
// the same pending list, under the same memory cap).  Neither the decision,
// the cell run, nor the plan-order assembly changes.
//
// THE DEVICE WORKER (slot 8, WP-6b, optional (a/s); spec knobs `ext_gpu`,
// `ext_gpu_cells`, CampaignSpec.h).  Under `ext_gpu = on` the parallel pass
// runs ONE extra worker beside the `ext_jobs` CPU workers, on its own engine
// pool, with a THREAD-LOCAL device override (engine::tape::ScopedThreadDevice
// (Gpu), EngineRuntime.h) installed after the captured context -- so its
// programs, and those of every task thread it launches, are compiled for the
// GPU, while each CPU worker installs ScopedThreadDevice(Cpu) and compiles for
// the CPU whatever ENGINE_DEVICE says.  It is a scheduling mode, not an
// arithmetic path:
//   * VALUES.  A device result is the same bytes as the CPU result by
//     construction (ENGINE_API.md Sec. 11); pass_options_fingerprint() keeps
//     GPU- and CPU-compiled programs in different cache slots; which worker
//     ran a cell is written into NO record, index.json or summary.tsv -- only
//     into the EXT_PASS line's `ext_gpu` fields and the verbose EXT_GPU_CELLS
//     line.  `ext_gpu = off` (the default) installs nothing: WP-6's pass.
//   * ITS RULE (ExtCellScheduler::take(ExtWorkerClass::Gpu)).  With an
//     explicit `ext_gpu_cells` list it takes exactly the listed engine cells
//     (LPT order among them) and no CPU worker takes one.  Otherwise it takes
//     the CHEAPEST pending cell -- the tail of the LPT order, so the cells
//     without U_ell first, then the cheapest U_ell cells --
//     only while every CPU slot is busy, and only while its modelled GPU cost
//     (kExtGpuCostFactor x the CPU prediction: the pessimistic end of the
//     measured 1.4-3.3x) does not
//     exceed the pending CPU work per CPU slot -- so the slower device never
//     becomes the pass's tail by the model.  Its cell counts against
//     `ext_memory_mb` like any other, and must fit a device-memory cap read
//     from the device at the start of the pass (kExtGpuDeviceCapFraction of
//     its free memory).
//   * NO DEVICE, NO MODE.  Without a usable device (or under
//     ENGINE_CUDA_DISABLE=1) `on` prints one line and runs WP-6's CPU-only
//     pass; `force` runs the worker anyway (its GPU-assigned nodes fall back
//     to the CPU in the executor) -- the plumbing test.
//
// ---------------------------------------------------------------------------
// One cell's ext outcome, as the campaign records it.
// ---------------------------------------------------------------------------
struct ExtCellSummary {
    uint32_t n_sym = 0;
    uint64_t ell   = 0;

    // False => the cell has no record (ell >= 2^32, or the run threw); the
    // owner job emits the item-6(c) placeholder and `reason` says why.
    bool        available = false;
    std::string reason;                 // "" when available

    // Where the standalone record lives, relative to the campaign output dir.
    std::string record_file;

    // How this cell was obtained.
    bool     ran      = false;          // the engine ran
    bool     reused   = false;          // `resume` reused the persisted record
    bool     resumed  = false;          // the engine CONTINUED a persisted refusal
    uint32_t resumed_from_mb = 0;       // the persisted budget it continued from
    uint32_t budget_mb       = 0;       // the budget this run used

    // The family S_0 was asked for (before the u_ell admission).
    std::vector<uint64_t> family;

    // --- the summary.tsv columns of [X] Sec. 1.5 -------------------------
    // False when the engine did not run for this cell (`resume` reused the
    // persisted record), so the eight columns print `-` rather than zeros --
    // exactly as the rest of a skipped job's row does.
    bool     summary_known   = false;
    uint32_t ideals          = 0;
    uint32_t max_embdim      = 0;
    uint32_t max_loewy       = 0;
    uint32_t nonsplit_ideals = 0;
    uint32_t nonlocal_ideals = 0;
    bool     has_sturm_B_prime = false;
    uint32_t sturm_B_prime     = 0;
    std::string certificate;            // distinct per-ideal values, joined by ","
    std::string final_verdict;          // "true" | "false" | "heuristic"

    // Diagnostics, recorded and never gated.
    uint32_t    resumable_items = 0;
    bool        u_ell_included  = false;
    std::string u_ell_reason;
    uint32_t    verify_code = 0;
    double      seconds     = 0.0;
};

// The per-cell key, in plan order.
struct ExtCellKey {
    uint32_t n_sym = 0;
    uint64_t ell   = 0;
    bool operator<(const ExtCellKey& o) const {
        if (n_sym != o.n_sym) return n_sym < o.n_sym;
        return ell < o.ell;
    }
};

// The whole pass's result: one summary per cell, plus the owner job ids whose
// files must be rewritten even under `resume`.
struct ExtPassResult {
    bool enabled = false;
    std::map<ExtCellKey, ExtCellSummary> cells;
    std::vector<std::string> force_rerun_job_ids;
    uint32_t num_ran = 0, num_reused = 0, num_resumed = 0, num_failed = 0;
    double wall_s = 0.0;

    // (slot-8 WP-6) The scheduling record of the pass -- printed on the
    // EXT_PASS line (ExtPassLine), NEVER written into index.json, summary.tsv
    // or a record: none of it may differ between the serial and a parallel
    // pass except by timing.
    uint32_t jobs = 1;                  // the ext_jobs the pass ran with
    uint64_t memory_cap_mb = 0;         // the ext_memory_mb it ran under
    uint32_t num_cells_run = 0;         // cells that needed the engine
    double   peak_rss_mb = -1.0;        // process VmHWM at the end of the pass
    double   peak_admitted_mb = 0.0;    // max sum of admitted predictions
    uint32_t peak_running = 0;          // max cells in flight
    uint32_t over_cap_alone = 0;        // cells admitted alone above the cap
    bool     per_worker_pools = false;  // each worker had its own engine pool

    // (slot-8 WP-6b) The device worker's scheduling record -- EXT_PASS line
    // and verbose log only, like the rest of this block.
    ExtGpuMode gpu_mode = ExtGpuMode::Off;  // the spec's ext_gpu
    bool     gpu_worker = false;        // the device worker ran
    std::string gpu_fallback;           // why `on` fell back to CPU-only ("" otherwise)
    double   gpu_device_cap_mb = 0.0;   // 0: not enforced (no device under `force`)
    std::vector<ExtCellKey> gpu_cells;  // cells the device worker ran, plan order
    uint64_t gpu_programs = 0, gpu_program_gpu_nodes = 0;  // its programs / their GPU nodes
    uint64_t cpu_programs = 0, cpu_program_gpu_nodes = 0;  // the CPU workers' (must be 0 nodes)

    const ExtCellSummary* find(uint32_t n_sym, uint64_t ell) const;
};

// The standalone record's path, relative to the campaign output directory
// ([X] Sec. 1.4's default name, under an `ext/` subdirectory).
//
// `ext_tag` -- (hecke-extensions slot 6, WP-6 / E2 close-out; [WP5-2]
// finding 3: "ext, family, sign, ext_groups do not fingerprint, so two
// campaigns differing only in family share the ext record file and its
// resume key"; owner confirmation 9, the record-file-name fix) -- is
// `ExtSpec::signature_tag()` (empty when `ext = false`; `"S0=..."` when
// `ext = true`, naming the good-prime set `family` asks for). When non-empty
// it is hashed into an `_x<16 lowercase hex digits of FNV-1a-64>` suffix, so
// a cell run under a different `family` writes -- and, under `resume`,
// looks for -- a DIFFERENT file, and can no longer silently reuse the wrong
// one. `sign` and `ext_groups` are NOT part of the tag: `sign` selects which
// record the SAME (n_sym, ell, family) cell writes into that one file
// (RunExtPass), and a resume asking for the cone over a record built
// without it is caught by the separate `pair_available` check, not by the
// file name. Pass the SAME string on every read that a write used, or the
// caller (not this function) will silently read the wrong file if one with
// the OLD tag happens to exist.
std::string ExtRecordRelPath(uint32_t n_sym, uint64_t ell, const std::string& ext_tag);

// Runs the pass.  `jobs` is the campaign's plan in execution order; the cell's
// FIRST job in that order owns the attachment.  The mathematical settings of
// `spec.math` must already be installed by the caller (the pass reads
// hecke::math_settings() for the `u_ell` budget).  Throws only for a refusal
// that makes the campaign impossible -- a lowered `u_ell` budget on a resume,
// or a corrupt state sidecar; a cell whose ENGINE threw is recorded as
// unavailable with the message and the campaign continues.
//
// `out_records`, when non-null (default null: production callers pass
// nothing and get the unchanged pass) -- (hecke-extensions slot 6, WP-6 /
// E2 close-out) receives the IN-MEMORY hecke_ext::HeckeExtRecord of every
// cell the engine actually RAN this call (not a `resume`d reuse), keyed the
// same way as `pass.cells`.  This is what lets HeckeExtCampaignValidation's
// golden arm hash `ext_ideals` / `ext_certify` from the campaign's own
// record object rather than re-parsing the JSON it wrote -- the record is
// never re-serialised or re-derived for this, only captured before it goes
// out of scope.
ExtPassResult RunExtPass(const CampaignSpec& spec, const std::vector<Job>& jobs,
                         const std::string& output_dir, bool verbose,
                         std::map<ExtCellKey, hecke_ext::HeckeExtRecord>* out_records = nullptr);

// (slot-8 WP-6) The machine-readable timing line of the pass, which the T-P
// measurement reads (docs/matrix-engine/tools/ext_pass_tp.sh):
//   EXT_PASS wall_s=<f> jobs=<J> cells=<n> engine_cells=<n> ran=<n> reused=<n>
//            resumed=<n> failed=<n> peak_rss_mb=<f> memory_cap_mb=<n>
//            peak_admitted_mb=<f> peak_running=<n> per_worker_pools=<0|1>
// and, ONLY when `ext_gpu != off` (so an `off` line is WP-6's byte for byte):
//            ext_gpu=<on|force> gpu_worker=<0|1> gpu_cells=<n> gpu_programs=<n>
//            gpu_nodes=<n> cpu_programs=<n> cpu_gpu_nodes=<n> device_cap_mb=<f>
//            [gpu_fallback="<reason>"]
std::string ExtPassLine(const ExtPassResult& pass);

// ---------------------------------------------------------------------------
// (slot-8 WP-6) The cost model and the admission rule of the parallel pass.
// Public so HeckeExtCampaignValidation can pin them on synthetic input.
// ---------------------------------------------------------------------------

// The per-cell memory floor of the admission model: what a cell WITHOUT
// U_ell may hold at n <= 200 (the ten-cell ext_first sample's non-U_ell
// cells stay far below it; WP-5 report Sec. 5 (b)), and what a U_ell cell
// holds beside the U_ell model's own prediction.
constexpr uint32_t kExtCellFloorMb = 512;
// (slot-8 WP-6b) The device worker's cost model: GPU cost = this factor x the
// CPU prediction (the pessimistic end of the measured 1.4-3.3x slowdown of
// ENGINE_DEVICE=gpu on the pipeline's shapes), and the share of the device's
// free memory (read at the start of the pass) its cell may be predicted at.
constexpr double kExtGpuCostFactor = 3.3;
constexpr double kExtGpuDeviceCapFraction = 0.8;
// The default `ext_memory_mb` is kExtMemoryMbDefault (CampaignSpec.h).

struct ExtCellCost {
    uint32_t n_sym = 0;
    uint64_t ell = 0;
    bool     u_ell_admitted = false;
    double   u_ell_predicted_mb = 0.0;  // 0 unless admitted
    double   memory_mb = 0.0;           // kExtCellFloorMb + u_ell_predicted_mb
};

// The prediction under the `u_ell` setting the cell runs to the end with.
ExtCellCost PredictExtCellCost(uint32_t n_sym, uint64_t ell, const hecke::UEllSetting& u_ell);

// Longest processing time first; ties by n (descending), then by index.
std::vector<size_t> ExtLptOrder(const std::vector<ExtCellCost>& cells);

// The admission: take() hands out the first pending cell (LPT order) whose
// prediction fits beside the running ones under the cap -- or any cell when
// nothing runs -- blocking while none fits; kNone when every cell is taken.
//
// (slot-8 WP-6b) With enable_device_worker() called, take(Gpu) is the device
// worker's rule (CampaignExt.h, "THE DEVICE WORKER") and take(Cpu) skips the
// cells an explicit list reserves for it.  take() == take(Cpu).
enum class ExtWorkerClass { Cpu, Gpu };

class ExtCellScheduler {
public:
    static constexpr size_t kNone = static_cast<size_t>(-1);
    ExtCellScheduler(const std::vector<ExtCellCost>& costs, double cap_mb);
    // `gpu_listed` (size = costs.size(), or empty for the worker's own rule)
    // marks the cells an explicit ext_gpu_cells list gives the device worker;
    // `device_cap_mb` <= 0 disables the device-memory check.
    void enable_device_worker(uint32_t cpu_slots, const std::vector<char>& gpu_listed,
                              double device_cap_mb);
    size_t take() { return take(ExtWorkerClass::Cpu); }
    size_t take(ExtWorkerClass cls);
    void done(size_t i);
    double peak_admitted_mb() const;
    uint32_t peak_running() const;
    uint32_t over_cap_alone() const;
private:
    std::vector<ExtCellCost> costs_;
    double cap_mb_;
    std::vector<size_t> pending_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    double running_mb_ = 0.0, peak_admitted_mb_ = 0.0;
    uint32_t running_ = 0, peak_running_ = 0, over_cap_alone_ = 0;
    // (slot-8 WP-6b) the device worker
    bool device_ = false;
    uint32_t cpu_slots_ = 0, cpu_running_ = 0;
    std::vector<char> gpu_listed_;      // empty: the worker's own rule
    std::vector<char> on_gpu_;          // per cell: taken by the device worker
    double device_cap_mb_ = 0.0;
    bool admit_locked(size_t i);        // bookkeeping of one admission (mu_ held)
};

// Streams the owner job's `"extensions": {...},"extensions_file":"..."` pair.
// Emits the item-6(c) placeholder when the cell has no record.  Always writes
// a trailing comma, like every other streamed member of a job file.
void StreamJobExtensions(std::ostream& os, const std::string& output_dir,
                         const ExtCellSummary& cell);

// The summary.tsv columns of [X] Sec. 1.5, appended ONLY when `ext = true`.
// `cell` is null for a job that does not own its cell's record.
std::string ExtSummaryHeader();
std::string ExtSummaryRow(const ExtCellSummary* cell);

// index.json's `"ext"` block, emitted only when the pass ran.
std::string ExtIndexJson(const CampaignSpec& spec, const ExtPassResult& pass);

// ---------------------------------------------------------------------------
// The persisted `u_ell` object of a cell's record file -- the resume input.
// Exposed for HeckeExtCampaignValidation, which pins the reader itself.
// ---------------------------------------------------------------------------
struct PersistedUEll {
    bool     file_found   = false;
    bool     object_found = false;
    bool     included     = false;
    std::string reason;
    uint32_t budget_mb    = 0;          // the verdict's own budget
    bool     resumable    = false;      // an INCOMPLETE `resumable` is present
    uint32_t resumable_budget_mb = 0;   // its `budget.budget_mb`
    std::vector<uint32_t> budget_history;
    // Every `state_ref` the record names, with its recorded hash.
    uint32_t sidecars_named = 0, sidecars_verified = 0;
    // Whether `complexes.pair` is a computed object rather than the item-6(c)
    // placeholder.  `ext_groups` is not a mathematical setting and therefore
    // does not fingerprint, so a `resume` that now asks for the cone over a
    // record built without it would otherwise silently reuse a record that
    // cannot answer -- the pass re-runs the cell instead.
    bool pair_available = false;
};

// Reads `<output_dir>/<ExtRecordRelPath(n, ell, ext_tag)>`.  Verifies every
// `state_ref` sidecar the record names (present and FNV-1a-64 as recorded)
// and throws std::runtime_error naming the file when one is missing or
// corrupt.  `ext_tag` must be the SAME `ExtSpec::signature_tag()` the run
// whose record is being read used (see `ExtRecordRelPath`'s comment).
PersistedUEll ReadPersistedUEll(const std::string& output_dir, uint32_t n_sym, uint64_t ell,
                                const std::string& ext_tag);

} // namespace campaign

#endif // CAMPAIGN_EXT_H
