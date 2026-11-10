// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 5, WP-2 / E2 cont. -- HeckeExtCampaignValidation.
//
// [X] = docs/implementation/hecke_extensions_implementation_plan.md v1.8
// Sec. 1.5 (the spec keys, the per-cell grouping, the summary.tsv columns),
// Sec. 1.3.1 R3.3 item 4 (the `u_ell` resume), Sec. 5 (this validation);
// [C] = docs/latex/gl2_contract.tex v1.6.2 by label.
//
// WHAT IT GATES, in the plan's own order:
//
//   (a) `ext = false` BYTE-IDENTITY.  The default campaign must be unchanged
//       down to summary.tsv's column set, every shape signature and every
//       Job::id() -- the acceptance criterion the whole package is written
//       around.  Arm 2 pins the 21-column header and the shipped id
//       `n16_ell5_p2_r1` as literals, and asserts that no `"extensions"` key
//       and no ` ext=` component exists anywhere in the output.
//
//   (b) THE PER-CELL GROUPING ([XA m14]).  One record per (n_sym, ell) cell,
//       attached to that cell's FIRST job in plan order, with the family
//       `spec.family u {p of the cell's jobs}` -- a property of
//       plan_campaign() alone, so arm 1 checks it without running anything.
//
//   (c) DETERMINISM.  The same spec at four `jobs x threads` splits must
//       produce the same jobs in the same ORDER with the same objects, the
//       same ext record FILES byte for byte, the same `u_ell` verdicts and a
//       summary.tsv byte-identical outside its two timing columns --
//       CampaignSmokeValidation's contract, extended to the ext half verbatim.
//
//   (d) THE RECORD-LEVEL `u_ell` RESUME.  A refused U_691 at (10, 691) under
//       `u_ell = budget_mb=64`, then a resume at the shipped default that
//       reproduces a FRESH `u_ell = auto` campaign's record BYTE FOR BYTE; a
//       lowered budget refused by NAME; a resume over an included verdict a
//       strict no-op; and the sidecar contract, on a record forced oversize in
//       a scratch directory (the shipped `u_ell` state is small, so the only
//       honest way to exercise the sidecar is to construct one).
//
// NO GOLDEN OBJECT IS DEFINED OR EMITTED HERE.  The slot's one emission is
// WP-3's; this file compares campaign output against campaign output and
// against literals -- EXCEPT arm 8 below, which is a LOOKUP, never a
// definition (docs/implementation/hecke_extensions_slot6_plan.md WP-6 /
// E2 close-out):
//
//   (e) THE GOLDEN ARM (owner ruling, slot-6 Sec. 6 decision 5; audit B-2).
//       `ext_ideals`, `ext_certify` and `sturm_bound` -- the
//       SETTINGS-INDEPENDENT ext objects only -- recomputed from the
//       campaign's own in-memory `HeckeExtRecord` through the golden
//       harness's own recipes (Golden/GoldenHarness.h), looked up in
//       `GoldenReference.inc` through `GoldenLookup`, on two of the golden
//       ext-subgrid's 31 cells, run at `u_ell = off` / `family = 2,3,5` --
//       the SAME install the harness's own sweep uses. SINCE SLOT 7 (WP-2,
//       stage 1 (2)) E2 (a) holds IN FULL: `ext_json_canonical` and
//       `ext_json_full` are ASSERTED equal to the reference under the
//       harness's path-scoped settings recipe, and the campaign's full
//       settings block equals the harness's install (the harness now stamps
//       `spec`). Slot 6 recorded, and did not assert, a one-substring
//       `"source"` difference here.
//
//   (g) THE PARALLEL EXT PASS (slot-8 WP-6, arm 10): serial == parallel at
//       `ext_jobs` in {1, 2, 4, 16}, the non-default-setting arm, the
//       admission, the knob invariance, the scheduler, a parallel resume.
//
//   (h) THE DEVICE WORKER (slot-8 WP-6b, arm 11): `ext_gpu = force` (every
//       box) and `on` (with a device) against the serial CPU-only pass, byte
//       for byte; the listed cells ran on the device worker, whose programs
//       carry GPU nodes while every CPU worker's carry none; the fallback
//       without a device; the non-default-setting arm; the knobs' invariance;
//       the worker's rule on synthetic input.
//
//   (f) THE EXTENSION-IDENTITY FIX ([WP5-2] finding 3, owner confirmation
//       9): two campaigns into ONE output directory, differing only in
//       `family`, the second `resume = true` -- it does NOT reuse the
//       first's record: a different `family` writes, and looks for, a
//       DIFFERENT record file (`ExtRecordRelPath`'s `ext_tag` suffix).
#include "CampaignExtValidation.h"

#include "CampaignDriver.h"
#include "CampaignExt.h"
#include "CampaignSpec.h"

#include "../Golden/GoldenHarness.h"
#include "../HeckeExt/HeckeExtRecord.h"
#include "../HeckeExt/HeckeExtReport.h"
#include "../HeckeExt/HeckeSturmBound.h"
#include "../Hecke/HeckeSettings.h"

#include "../../ModArithmetic/Fnv.h"
#include "../../Cuda/CudaCtx.h"   // slot-8 WP-6b: available()
#include "../Util/TextIo.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>   // getpid, for a unique scratch directory

namespace {

uint32_t g_failures = 0;
uint64_t g_checks = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        std::cout << "  FAIL: " << what << std::endl;
        ++g_failures;
    }
}

template<class A, class B>
void CheckEq(const A& got, const B& want, const std::string& what) {
    ++g_checks;
    if (!(got == want)) {
        std::cout << "  FAIL: " << what << " -- got '" << got << "', want '" << want << "'"
                  << std::endl;
        ++g_failures;
    }
}

std::string TempDir(const std::string& tag) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    const fs::path dir = (ec ? fs::path(".") : base) /
                         ("oc_extcamp_" + tag + "_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    return dir.string();
}

void CleanUp(const std::string& dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

using oc_util::ReadFile;   // S24 CL-7 (F11): Util/TextIo.h
campaign::CampaignSpec SpecFromString(const std::string& text, const std::string& out_dir) {
    std::istringstream is(text);
    campaign::CampaignSpec spec = campaign::ParseCampaignSpec(is, "<ext-validation-spec>");
    spec.output_dir = out_dir;
    return spec;
}

// summary.tsv with its two volatile columns blanked -- the same 18 / 19 the
// pre-WP-2 CampaignValidation.cpp blanks, which is itself an assertion that
// the ext columns were APPENDED and did not move them.
std::string StripVolatileTsvColumns(const std::string& tsv) {
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

// The 21-column header frozen before this package ([X] Sec. 1.5's "the pinned
// default form is untouched"; the `q=` / `_q` precedent).
const char* kFrozenSummaryHeader =
    "index\tn_sym\tell\tp\tr\tmodel\tstatus\tdim_H0\tdim_H1\tdim_H2"
    "\tcharpoly_A0\tcharpoly_A1\tcharpoly_A2\ttrace_A1\trank_A1"
    "\ttheta_flat_hash_q1\tcertify\texternal\ttime_s\trss_mb\tfile";

std::string FirstLine(const std::string& s) {
    const size_t nl = s.find('\n');
    return nl == std::string::npos ? s : s.substr(0, nl);
}

// A cheap ext grid: even n, ell = 5, and the cells the extension layer has
// something to say about.  (10, 691) is the resume arm's and is kept out of
// the repeated split arms, where it would be run five times.
const char* kExtSpecText =
    "name    = ext_campaign\n"
    "certify = none\n"
    "ext     = true\n"
    "family  = 2,3\n"
    "n_sym   = 2,4\n"
    "ell     = 5\n"
    "p       = 2,3\n"
    "r       = 1\n";

// ===========================================================================
// Arm 1 -- the plan alone: the per-cell grouping of [X] Sec. 1.5 / [XA m14].
// ===========================================================================
void Arm1Plan(const std::string& root) {
    std::cout << "-- arm 1: the per-cell grouping (plan only)" << std::endl;
    campaign::CampaignSpec spec = SpecFromString(kExtSpecText, root + "/plan");
    const std::vector<campaign::JobGroup> groups = campaign::plan_campaign(spec);
    const std::vector<campaign::Job> jobs = campaign::flatten(groups);
    CheckEq(jobs.size(), size_t(4), "the 2x1x2x1 ext spec plans 4 jobs");

    std::map<std::pair<uint32_t, uint64_t>, uint32_t> owners;
    std::map<std::pair<uint32_t, uint64_t>, std::string> first_id;
    for (const campaign::Job& j : jobs) {
        const std::pair<uint32_t, uint64_t> cell{j.n_sym, j.ell};
        if (first_id.find(cell) == first_id.end()) first_id[cell] = j.id();
        if (j.ext_owner) ++owners[cell];
        // The family is the cell's, not the job's: every job of a cell agrees.
        const std::vector<uint64_t> want = {2, 3};
        CheckEq(j.ext_family.size(), want.size(),
                "job " + j.id() + " carries the cell family");
        Check(j.ext_family == want, "job " + j.id() + " family = {2,3} (spec u {p})");
    }
    CheckEq(owners.size(), size_t(2), "two (n, ell) cells");
    for (const auto& kv : owners)
        CheckEq(kv.second, 1u, "exactly one owner for cell n=" + std::to_string(kv.first.first));
    for (const campaign::Job& j : jobs)
        if (j.ext_owner)
            CheckEq(j.id(), first_id[{j.n_sym, j.ell}],
                    "the owner is the cell's FIRST job in plan order");

    // The shape signature gains ` ext=`; the job id does not.
    Check(jobs[0].shape().to_string().find(" ext=S0=2,3") != std::string::npos,
          "the shape signature carries the family's prime set");
    Check(jobs[0].id().find("ext") == std::string::npos,
          "Job::id() does NOT carry the ext tag");

    // `family = 2,3,5,7,ell`: the token is recorded and never resolved.
    {
        campaign::CampaignSpec s2 = SpecFromString(
            std::string(kExtSpecText) + "family = 2,3,5,7,ell\n", root + "/plan2");
        Check(s2.ext.names_ell, "the literal token `ell` is recorded");
        const std::vector<uint64_t> want = {2, 3, 5, 7};
        Check(s2.ext.good_primes == want, "`ell` is not resolved into the good-prime list");
        Check(s2.ext.signature_tag().find(",ell") != std::string::npos,
              "the signature names it");
    }
    // The three enumerated keys reject an unknown value by name.
    for (const char* bad : {"sign = sideways\n", "ext_groups = everything\n",
                            "family = 1\n"}) {
        bool threw = false;
        try {
            SpecFromString(std::string(kExtSpecText) + bad, root + "/plan3");
        } catch (const std::exception&) { threw = true; }
        Check(threw, std::string("a bad value for '") + bad + "' is a spec error");
    }
}

// ===========================================================================
// Arm 2 -- acceptance (a): `ext = false` is byte-identical.
// ===========================================================================
void Arm2Default(const std::string& root) {
    std::cout << "-- arm 2: `ext = false` leaves the campaign byte-identical" << std::endl;
    static const char* kPlain =
        "name    = ext_off\n"
        "certify = none\n"
        "n_sym   = 0,2\n"
        "ell     = 5\n"
        "p       = 2\n"
        "r       = 1\n";
    campaign::CampaignSpec spec = SpecFromString(kPlain, root + "/plain");
    Check(!spec.ext.ext, "`ext` defaults to false");
    Check(spec.ext.signature_tag().empty(), "and its signature tag is empty");

    const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
    CheckEq(res.num_failed, uint64_t(0), "the default campaign completes");
    Check(!res.ext_enabled, "the ext pass did not run");

    const std::string tsv = ReadFile(spec.output_dir + "/summary.tsv");
    CheckEq(FirstLine(tsv), std::string(kFrozenSummaryHeader),
            "summary.tsv's column set is the frozen 21-column one");

    const std::string index = ReadFile(spec.output_dir + "/index.json");
    Check(index.find("\"extensions\"") == std::string::npos,
          "index.json carries no `extensions` block");
    for (const campaign::JobResult& r : res.jobs) {
        const std::string js = ReadFile(r.output_path);
        Check(js.find("\"extensions\"") == std::string::npos,
              "job " + r.job.id() + " carries no `extensions` member");
        Check(js.find(" ext=") == std::string::npos,
              "job " + r.job.id() + "'s shape signature carries no ext tag");
        Check(!r.ext_owner, "no job owns an ext record");
    }
    // The shipped id and shape signature, as literals.
    {
        campaign::Job j;
        j.n_sym = 16; j.ell = 5; j.p = 2; j.r = 1;
        j.degrees = campaign::ModelDefaultDegrees("sl2z_7vertex");
        CheckEq(j.id(), std::string("n16_ell5_p2_r1"), "the default-degree id is unchanged");
        CheckEq(j.shape().to_string(),
                std::string("n_sym=16 p=2 r=1 q=0,1,2 model=sl2z_7vertex"),
                "the default shape signature is unchanged");
    }
    CleanUp(spec.output_dir);
}

// ===========================================================================
// Arm 3 -- the attachment: one record per cell, on the owner, in both files.
// ===========================================================================
std::vector<std::string> Arm3Attach(const std::string& root, std::string& summary_out) {
    std::cout << "-- arm 3: the attachment" << std::endl;
    campaign::CampaignSpec spec = SpecFromString(kExtSpecText, root + "/run");
    const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
    CheckEq(res.num_failed, uint64_t(0), "the ext campaign completes");
    Check(res.ext_enabled, "the ext pass ran");
    CheckEq(res.ext_cells, uint64_t(2), "two cells");
    CheckEq(res.ext_ran, uint64_t(2), "both ran");
    CheckEq(res.ext_failed, uint64_t(0), "none unavailable");

    std::vector<std::string> record_texts;
    uint32_t owners = 0;
    for (const campaign::JobResult& r : res.jobs) {
        const std::string js = ReadFile(r.output_path);
        const bool has = js.find("\"extensions\":") != std::string::npos;
        if (r.job.ext_owner) {
            ++owners;
            Check(has, "the owner job " + r.job.id() + " carries `extensions`");
            Check(js.find("\"hecke_ext/v5\"") != std::string::npos,
                  "and the record is `hecke_ext/v5`");
            Check(js.find("\"extensions_file\":\"ext/hecke_ext_ell") != std::string::npos,
                  "and names the standalone record file");
            Check(r.ext_owner && r.ext.available && r.ext.summary_known,
                  "the JobResult carries the cell summary");
            const std::string rec =
                ReadFile(spec.output_dir + "/" + r.ext.record_file);
            Check(!rec.empty(), "the standalone record file exists");
            Check(js.find(rec.substr(0, rec.size() > 200 ? 200 : rec.size())) !=
                      std::string::npos,
                  "the embedded copy is the standalone record verbatim");
            record_texts.push_back(rec);
        } else {
            Check(!has, "the non-owner job " + r.job.id() + " carries no `extensions`");
        }
    }
    CheckEq(owners, 2u, "exactly two owner jobs");

    // summary.tsv: the eight columns are APPENDED and carry the owner's values.
    const std::string tsv = ReadFile(spec.output_dir + "/summary.tsv");
    summary_out = tsv;
    const std::string header = FirstLine(tsv);
    Check(header.rfind(kFrozenSummaryHeader, 0) == 0,
          "the 21 frozen columns still come first");
    Check(header.find("\tideals\tmax_embdim\tmax_loewy\tnonsplit_ideals\tnonlocal_ideals"
                      "\tsturm_B_prime\text_certificate\text_final") != std::string::npos,
          "and [X] Sec. 1.5's eight ext columns follow");
    {
        uint32_t cols = 1;
        for (char c : header) if (c == '\t') ++cols;
        CheckEq(cols, 29u, "29 columns with `ext = true`");
    }
    uint32_t rows_with_values = 0, rows_with_dashes = 0;
    {
        std::istringstream in(tsv);
        std::string line;
        std::getline(in, line);                    // header
        while (std::getline(in, line)) {
            std::vector<std::string> f;
            std::istringstream ls(line);
            std::string field;
            while (std::getline(ls, field, '\t')) f.push_back(field);
            CheckEq(f.size(), size_t(29), "every row has 29 fields");
            if (f.size() == 29) {
                if (f[21] == "-") ++rows_with_dashes; else ++rows_with_values;
            }
        }
    }
    CheckEq(rows_with_values, 2u, "the two owner rows carry the ext columns");
    CheckEq(rows_with_dashes, 2u, "the two non-owner rows print dashes");

    const std::string index = ReadFile(spec.output_dir + "/index.json");
    Check(index.find("\"extensions\":{\"ext\":true") != std::string::npos,
          "index.json carries the ext block");
    Check(index.find("\"family\":\"2,3\"") != std::string::npos,
          "which echoes the family verbatim");
    return record_texts;
}

// ===========================================================================
// Arm 4 -- determinism: four `jobs x threads` splits.
// ===========================================================================
void Arm4Splits(const std::string& root, const std::vector<std::string>& seq_records,
                const std::string& seq_summary) {
    std::cout << "-- arm 4: the same spec at four jobs x threads splits" << std::endl;
    static const struct { uint32_t jobs, threads; } kSplits[] = {{2, 1}, {4, 2}, {3, 4}, {8, 1}};
    for (const auto& split : kSplits) {
        campaign::CampaignSpec spec = SpecFromString(
            kExtSpecText, root + "/split_" + std::to_string(split.jobs) + "x" +
                              std::to_string(split.threads));
        spec.jobs = split.jobs;
        spec.threads = split.threads;
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        const std::string tag = std::to_string(split.jobs) + "x" + std::to_string(split.threads);
        CheckEq(res.num_failed, uint64_t(0), tag + ": the campaign completes");
        CheckEq(res.ext_cells, uint64_t(2), tag + ": two cells");
        CheckEq(res.ext_ran, uint64_t(2), tag + ": both ran");

        std::vector<std::string> recs;
        for (const campaign::JobResult& r : res.jobs)
            if (r.job.ext_owner)
                recs.push_back(ReadFile(spec.output_dir + "/" + r.ext.record_file));
        CheckEq(recs.size(), seq_records.size(), tag + ": the same number of records");
        for (size_t i = 0; i < recs.size() && i < seq_records.size(); ++i)
            Check(recs[i] == seq_records[i],
                  tag + ": ext record #" + std::to_string(i) +
                      " is byte-identical to the sequential arm's");
        Check(StripVolatileTsvColumns(ReadFile(spec.output_dir + "/summary.tsv")) ==
                  StripVolatileTsvColumns(seq_summary),
              tag + ": summary.tsv is byte-identical outside its timing columns");
        CleanUp(spec.output_dir);
    }
}

// ===========================================================================
// Arm 5 -- the record-level `u_ell` resume ([X] Sec. 1.3.1 R3.3 item 4).
// ===========================================================================
void Arm5Resume(const std::string& root) {
    std::cout << "-- arm 5: the record-level u_ell resume at (10, 691)" << std::endl;
    static const char* kCell =
        "name    = ext_resume\n"
        "certify = none\n"
        "ext     = true\n"
        "ext_groups = bulk\n"          // the cone is not what this arm is about
        "family  = 2,3\n"
        "n_sym   = 10\n"
        "ell     = 691\n"
        "p       = 2\n"
        "r       = 1\n";

    // `family` (hence the ext-identity tag, slot-6 WP-6 / [WP5-2] finding 3)
    // is "2,3" throughout every sub-step of this arm -- only `u_ell` and
    // `resume` vary below -- so one tag serves the whole arm.
    const std::string kTag = SpecFromString(kCell, std::string()).ext.signature_tag();

    // (1) The reference: a fresh campaign at the DEFAULT `u_ell = auto`.
    const std::string fresh_dir = root + "/fresh";
    {
        campaign::CampaignSpec spec = SpecFromString(kCell, fresh_dir);
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.num_failed, uint64_t(0), "the fresh `auto` campaign completes");
    }
    const std::string fresh_record =
        ReadFile(fresh_dir + "/" + campaign::ExtRecordRelPath(10, 691, kTag));
    Check(!fresh_record.empty(), "the fresh record exists");

    // (2) The cut run: `u_ell = budget_mb=64` refuses U_691 (~104 MB predicted).
    const std::string cut_dir = root + "/cut";
    {
        campaign::CampaignSpec spec =
            SpecFromString(std::string(kCell) + "u_ell = budget_mb=64\n", cut_dir);
        Check(spec.math.fingerprint() != 0, "a non-default u_ell fingerprints");
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.num_failed, uint64_t(0), "the budgeted campaign completes");
        const campaign::PersistedUEll p = campaign::ReadPersistedUEll(cut_dir, 10, 691, kTag);
        Check(p.file_found && p.object_found, "the record carries a cell-level u_ell object");
        Check(!p.included, "U_691 is refused");
        CheckEq(p.reason, std::string("u_ell_over_budget"), "the refusal names the budget");
        Check(p.resumable, "and carries an INCOMPLETE resumable");
        CheckEq(p.resumable_budget_mb, 64u, "whose budget is 64 MB");
        Check(p.budget_history == std::vector<uint32_t>({64}), "budget_history = [64]");
        const campaign::ExtCellSummary* c = res.jobs[0].ext_owner ? &res.jobs[0].ext : nullptr;
        Check(c != nullptr && c->resumable_items == 1,
              "one refusal counts ONCE, whatever the ideal count (hecke_ext/v2)");
        const std::string cut_record =
            ReadFile(cut_dir + "/" + campaign::ExtRecordRelPath(10, 691, kTag));
        Check(!cut_record.empty() && cut_record != fresh_record,
              "and the cut record is genuinely different from the fresh one");
    }

    // (3) A LOWER budget on a resume is refused BY NAME.
    {
        campaign::CampaignSpec spec =
            SpecFromString(std::string(kCell) + "u_ell = budget_mb=32\nresume = true\n", cut_dir);
        bool named = false;
        try {
            campaign::RunCampaignSpec(spec, /*verbose=*/false);
        } catch (const std::exception& e) {
            named = std::string(e.what()).find("u_ell budget below reached state") !=
                    std::string::npos;
            if (!named) std::cout << "        (message was: " << e.what() << ")" << std::endl;
        }
        Check(named, "a resume to 32 MB over a state that reached 64 is refused BY NAME");
    }

    // (4) THE RESUME to the shipped default: the record must equal the fresh
    //     `u_ell = auto` campaign's BYTE FOR BYTE ([WP4-2] finding 4 -- the
    //     resume installs `auto` at the default budget, so even the settings
    //     block and its zero fingerprint agree).
    //
    //     The job file of the earlier (fingerprinted) run is copied onto the
    //     default run's id first, so the OWNER job is complete before the
    //     resume: that is the case `ext_force_rerun` exists for, and without
    //     it the resumed record would never reach `"extensions"`.
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        campaign::Job probe;
        probe.n_sym = 10; probe.ell = 691; probe.p = 2; probe.r = 1;
        probe.degrees = campaign::ModelDefaultDegrees("sl2z_7vertex");
        const std::string default_id = probe.id();
        std::string cut_job;
        for (const auto& e : fs::directory_iterator(cut_dir + "/jobs", ec))
            cut_job = e.path().string();
        Check(!cut_job.empty(), "the budgeted run wrote a job file");
        fs::copy_file(cut_job, cut_dir + "/jobs/" + default_id + ".json",
                      fs::copy_options::overwrite_existing, ec);
        Check(campaign::JobOutputComplete(cut_dir + "/jobs/" + default_id + ".json"),
              "and the default-id job file now reports complete");

        campaign::CampaignSpec spec = SpecFromString(std::string(kCell) + "resume = true\n",
                                                     cut_dir);
        CheckEq(spec.math.fingerprint(), uint64_t(0), "the resume runs at the defaults");
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.num_failed, uint64_t(0), "the resuming campaign completes");
        CheckEq(res.ext_resumed, uint64_t(1), "exactly one cell was RESUMED");
        CheckEq(res.ext_reused, uint64_t(0), "and none reused");
        CheckEq(res.num_ran, uint64_t(1),
                "the owner job was re-run although its file was complete");

        const campaign::PersistedUEll p = campaign::ReadPersistedUEll(cut_dir, 10, 691, kTag);
        Check(p.included, "after the resume U_691 is included");
        Check(!p.resumable, "and the record carries no incomplete resumable");
        const std::string resumed =
            ReadFile(cut_dir + "/" + campaign::ExtRecordRelPath(10, 691, kTag));
        Check(resumed == fresh_record,
              "the resumed record equals a fresh `u_ell = auto` campaign's, BYTE FOR BYTE");
        const std::string js = ReadFile(cut_dir + "/jobs/" + default_id + ".json");
        Check(js.find("\"extensions\":{\"schema\":\"hecke_ext/v5\"") != std::string::npos,
              "and the owner job's `extensions` carries the resumed record");
    }

    // (5) A resume over an INCLUDED verdict is a strict no-op: the cell is
    //     reused, the owner job is skipped and nothing is rewritten.
    {
        campaign::CampaignSpec spec = SpecFromString(std::string(kCell) + "resume = true\n",
                                                     cut_dir);
        const std::string before =
            ReadFile(cut_dir + "/" + campaign::ExtRecordRelPath(10, 691, kTag));
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.ext_reused, uint64_t(1), "the included cell is REUSED");
        CheckEq(res.ext_ran, uint64_t(0), "the engine did not run");
        CheckEq(res.num_skipped, uint64_t(1), "and the owner job is skipped");
        Check(ReadFile(cut_dir + "/" + campaign::ExtRecordRelPath(10, 691, kTag)) == before,
              "the record is untouched");
    }
    CleanUp(fresh_dir);
    CleanUp(cut_dir);

    // (6) (slot-8 WP-11b, [KI] Sec. 4d, owner ruling 9 (A)) THE `source`
    //     STAMP ON A RESUME TO A NON-DEFAULT BUDGET.  Step (4)'s resume lands
    //     on the shipped default (`budget_mb=2048`), which IS the default
    //     value, so `ScopedUEll` stamps its source `default` whatever the
    //     caller asks for and the fix is invisible there.  Resume instead to
    //     a budget that stays non-default (4096 MB, still comfortably above
    //     the ~104 MB (10, 691) prediction) and the campaign's setter -- the
    //     spec, never an interactive console -- must read `spec`, not the
    //     `ScopedUEll` default of `console`.
    const std::string src_dir = root + "/source_stamp";
    {
        campaign::CampaignSpec spec =
            SpecFromString(std::string(kCell) + "u_ell = budget_mb=64\n", src_dir);
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.num_failed, uint64_t(0), "(6) the budgeted campaign completes");
        const campaign::PersistedUEll p = campaign::ReadPersistedUEll(src_dir, 10, 691, kTag);
        Check(p.file_found && p.object_found && !p.included && p.resumable,
              "(6) the cut record is resumable");
    }
    {
        campaign::CampaignSpec spec = SpecFromString(
            std::string(kCell) + "u_ell = budget_mb=4096\nresume = true\n", src_dir);
        Check(spec.math.fingerprint() != 0, "(6) budget_mb=4096 is not the default");
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.num_failed, uint64_t(0), "(6) the resuming campaign completes");
        CheckEq(res.ext_resumed, uint64_t(1), "(6) exactly one cell was RESUMED");
        const campaign::PersistedUEll p = campaign::ReadPersistedUEll(src_dir, 10, 691, kTag);
        Check(p.included, "(6) after the resume U_691 is included");
        const std::string resumed =
            ReadFile(src_dir + "/" + campaign::ExtRecordRelPath(10, 691, kTag));
        Check(resumed.find("\"u_ell\":{\"value\":\"budget_mb=4096\",\"source\":\"spec\"}") !=
                  std::string::npos,
              "(6) the resumed record's settings.u_ell stamps source = spec, not console");
        Check(resumed.find("\"source\":\"console\"") == std::string::npos,
              "(6) and no key in the resumed record stamps source = console");

        // A FRESH campaign at the same non-default budget (no resume in its
        // history) must produce the identical record: the source stamp does
        // not depend on the resume path having been taken.
        const std::string fresh_dir2 = root + "/source_stamp_fresh";
        campaign::CampaignSpec fresh_spec =
            SpecFromString(std::string(kCell) + "u_ell = budget_mb=4096\n", fresh_dir2);
        const campaign::CampaignResult fresh_res =
            campaign::RunCampaignSpec(fresh_spec, /*verbose=*/false);
        CheckEq(fresh_res.num_failed, uint64_t(0), "(6) the fresh budget_mb=4096 campaign completes");
        const std::string fresh_at_4096 =
            ReadFile(fresh_dir2 + "/" + campaign::ExtRecordRelPath(10, 691, kTag));
        Check(resumed == fresh_at_4096,
              "(6) the resumed record equals a fresh `u_ell = budget_mb=4096` campaign's, "
              "BYTE FOR BYTE");
        CleanUp(fresh_dir2);
    }
    CleanUp(src_dir);
}

// ===========================================================================
// Arm 6 -- the sidecar contract of [X] R3.1 item 6, on a constructed record.
//
// The shipped `u_ell` resumable carries no state JSON, so no campaign of a
// realistic size ever writes a sidecar.  The reader must still verify one --
// a resume reads the record as PERSISTED STATE -- so the arm constructs a
// record whose commutant state is over the 64 KiB limit, writes it where a
// campaign would, and checks that the reader finds it, verifies its FNV-1a-64,
// and refuses BY NAME when it is deleted or altered.
// ===========================================================================
void Arm6Sidecar(const std::string& root) {
    std::cout << "-- arm 6: the state sidecar" << std::endl;
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string out = root + "/sidecar";
    fs::create_directories(out + "/ext", ec);

    hecke_ext::HeckeExtRecord rec;
    rec.n_sym = 10;
    rec.ell = 691;
    rec.weight = 12;
    rec.bulk.available = true;
    rec.bulk.num_degrees = 3;
    rec.u_ell.present = true;
    rec.u_ell.included = false;
    rec.u_ell.reason = "u_ell_over_budget";
    rec.u_ell.predicted_mb = 104.06;
    rec.u_ell.budget_mb = 64;
    rec.u_ell.model_version = "u_ell_rss/v2-2026-09-17";
    rec.u_ell.resumable.present = true;
    rec.u_ell.resumable.kind = hecke_ext::kKindUEll;
    rec.u_ell.resumable.complete = false;
    rec.u_ell.resumable.budget_json = "{\"budget_mb\":64}";
    rec.u_ell.resumable.budget_history_json = "[64]";
    // Forced oversize: > 64 KiB, so the writer must emit a `state_ref`.
    rec.u_ell.resumable.state_json = "[" + std::string(70000, '0') + "]";
    CheckEq(rec.verify(), 0u, "the constructed record verifies");

    // No CampaignSpec here (the record is constructed directly), so no
    // family and hence the empty, untagged identity (ExtRecordRelPath's
    // comment): consistent with every read below, which is all this arm
    // needs.
    const std::string rel = campaign::ExtRecordRelPath(10, 691, std::string());
    hecke_ext::WriteHeckeExtReport(rec, out + "/" + rel);
    const std::string text = ReadFile(out + "/" + rel);
    Check(text.find("\"state_ref\":") != std::string::npos,
          "an oversize state is written as a sidecar reference");

    const campaign::PersistedUEll p = campaign::ReadPersistedUEll(out, 10, 691, std::string());
    CheckEq(p.sidecars_named, 1u, "the reader finds the one state_ref");
    CheckEq(p.sidecars_verified, 1u, "and verifies its fnv64");
    Check(p.resumable && p.resumable_budget_mb == 64,
          "and reads the cell-level resumable through it");

    // Find the sidecar and break it, two ways.
    std::string sidecar;
    for (const auto& e : fs::directory_iterator(out + "/ext", ec))
        if (e.path().string().find(".state.") != std::string::npos) sidecar = e.path().string();
    Check(!sidecar.empty(), "the sidecar file exists beside the record");
    {
        const std::string body = ReadFile(sidecar);
        std::ofstream os(sidecar, std::ios::out | std::ios::trunc);
        os << body.substr(0, body.size() - 1);
        os.close();
        bool named = false;
        try { campaign::ReadPersistedUEll(out, 10, 691, std::string()); }
        catch (const std::exception& e) {
            named = std::string(e.what()).find("does not match its recorded fnv64") !=
                    std::string::npos;
        }
        Check(named, "an ALTERED sidecar is refused by name");
    }
    fs::remove(sidecar, ec);
    {
        bool named = false;
        try { campaign::ReadPersistedUEll(out, 10, 691, std::string()); }
        catch (const std::exception& e) {
            named = std::string(e.what()).find("is missing") != std::string::npos;
        }
        Check(named, "a MISSING sidecar is refused by name");
    }
    CleanUp(out);
}

// ===========================================================================
// Arm 7 -- the two cases a `resume` must NOT reuse.
//
//  (a) `ext_groups` is not a mathematical setting, so it does not fingerprint
//      and does not change a job id: a record built WITHOUT the cone must
//      therefore not be reused by a later run that asks for it, or the cell
//      would silently keep a record that cannot answer.
//  (b) `ell >= 2^32`: the extension layer is the uint32_t instantiation
//      ([XA M3]), so the cell has NO record and the owner job emits the
//      [C] `sec:recordschema` item 6(c) placeholder with its reason.
// ===========================================================================
void Arm7ReuseGuardAndWidth(const std::string& root) {
    std::cout << "-- arm 7: what a resume must not reuse, and the 64-bit edge" << std::endl;
    const std::string dir = root + "/guard";
    static const char* kBulk =
        "name    = ext_guard\n"
        "certify = none\n"
        "ext     = true\n"
        "family  = 2\n"
        "n_sym   = 2\n"
        "ell     = 5\n"
        "p       = 2\n"
        "r       = 1\n";
    // `family` (hence the ext-identity tag) is "2" throughout this arm --
    // only `ext_groups` varies -- so one tag serves all three sub-blocks.
    const std::string kTag = SpecFromString(kBulk, std::string()).ext.signature_tag();
    {
        campaign::CampaignSpec spec =
            SpecFromString(std::string(kBulk) + "ext_groups = bulk\n", dir);
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.ext_ran, uint64_t(1), "the `bulk` campaign ran the cell");
        const campaign::PersistedUEll p = campaign::ReadPersistedUEll(dir, 2, 5, kTag);
        Check(!p.pair_available, "and its record's `pair` is the item-6(c) placeholder");
    }
    {
        campaign::CampaignSpec spec =
            SpecFromString(std::string(kBulk) + "ext_groups = all\nresume = true\n", dir);
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.ext_ran, uint64_t(1), "a resume asking for the cone RE-RUNS the cell");
        CheckEq(res.ext_reused, uint64_t(0), "and does not reuse it");
        const campaign::PersistedUEll p = campaign::ReadPersistedUEll(dir, 2, 5, kTag);
        Check(p.pair_available, "the rebuilt record carries a computed `pair`");
    }
    {
        campaign::CampaignSpec spec =
            SpecFromString(std::string(kBulk) + "ext_groups = all\nresume = true\n", dir);
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.ext_reused, uint64_t(1), "the second resume DOES reuse it");
        CheckEq(res.ext_ran, uint64_t(0), "and the engine does not run");
    }
    CleanUp(dir);

    // (b) a 64-bit cell.
    {
        const std::string wide = root + "/wide";
        campaign::CampaignSpec spec = SpecFromString(
            "name    = ext_wide\n"
            "certify = none\n"
            "ext     = true\n"
            "n_sym   = 0\n"
            "ell     = 4294967311\n"
            "p       = 2\n"
            "r       = 1\n", wide);
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.num_failed, uint64_t(0), "the 64-bit job itself still completes");
        CheckEq(res.ext_ran, uint64_t(0), "no ext record is built at ell >= 2^32");
        CheckEq(res.ext_failed, uint64_t(1), "the cell is recorded unavailable");
        const std::string js = ReadFile(res.jobs[0].output_path);
        Check(js.find("\"extensions\":{\"status\":\"unavailable\"") != std::string::npos,
              "and the owner job carries the item-6(c) placeholder");
        Check(js.find("the extension layer is 32-bit") != std::string::npos,
              "with the reason that made it one");
        CleanUp(wide);
    }
}

// ===========================================================================
// Arm 8 -- (hecke-extensions slot 6, WP-6 / E2 close-out) THE GOLDEN ARM,
// [X] Sec. 1.6; owner ruling, slot-6 Sec. 6 decision 5 -- audit B-2.
//
// E2 (a)'s golden arm was unsatisfiable against the FROZEN `hecke_ext/v2`
// text (the campaign installs `u_ell = off` through `ScopedMathSettings`,
// which stamps `source = "spec"`; the harness installs the SAME value
// through `ScopedUEll`, which stamps `source = "console"` -- Sec. 2.3 item
// 2). So this arm compares only the SETTINGS-INDEPENDENT ext objects --
// `ext_ideals`, `ext_certify`, `sturm_bound` -- through the harness's own
// recipes (Golden/GoldenHarness.h's exposed forwarders), recomputed from
// the campaign's own IN-MEMORY `hecke_ext::HeckeExtRecord` (RunExtPass's
// `out_records`, never a re-derivation), against `GoldenReference.inc`
// through `GoldenLookup` -- never a copied constant. (Slot 6 RECORDED the
// `ext_json_canonical` gap -- exactly the one `"source"` substring above --
// and booked the equality to slot 7. Slot-7 WP-2 closes it: stage 1 (1)'s
// path-scoped recipe and stage 1 (2)'s `spec` stamp make both `ext_json_*`
// objects an ASSERTED equality; see sub-arm (c) below.)
//
// `u_ell = off` AND `family = 2,3,5` (`good_primes`) are what the harness's
// OWN sweep installs for this subgrid (GoldenHarness.cpp `RunExtCell32`);
// without both, this arm would compare two different computations. The
// cells are two of the golden ext-subgrid's 31 ([X] Sec. 1.6: n > 0 even,
// 5 <= ell < 2^32): (10, 691), the Ramanujan cell, and (4, 13).
// ===========================================================================
void Arm8GoldenSubgrid(const std::string& root) {
    std::cout << "-- arm 8: E2 (a) -- the golden arm, settings-independent ext objects"
              << std::endl;
    struct Cell { uint32_t n; uint64_t ell; };
    static const Cell kCells[] = {{10, 691}, {4, 13}};

    static const char* kSpec =
        "name    = ext_golden\n"
        "certify = none\n"
        "ext     = true\n"
        "family  = 2,3,5\n"
        "u_ell   = off\n"
        "n_sym   = 10\n"
        "ell     = 691\n"
        "p       = 2\n"
        "r       = 1\n"
        "---\n"
        "n_sym   = 4\n"
        "ell     = 13\n"
        "p       = 2\n"
        "r       = 1\n";
    const std::string dir = root + "/golden";
    campaign::CampaignSpec spec = SpecFromString(kSpec, dir);
    Check(spec.ext.good_primes == std::vector<uint64_t>({2, 3, 5}),
          "the spec's family = {2,3,5} -- the harness's own good_primes");

    // (b) THE DEFAULT FAMILY WOULD NOT MATCH -- the two installs are
    // load-bearing, not incidental.  Checked on the plan alone (the default
    // `good_primes` is empty, meaning the engine's own {2,3,5,7}; `u_ell`
    // stays `auto` unless overridden), before the (expensive) recipe run.
    {
        const campaign::CampaignSpec def = SpecFromString(
            "name = ext_default\ncertify = none\next = true\nn_sym = 10\nell = 691\np = 2\nr = 1\n",
            dir + "_probe");
        Check(def.ext.good_primes.empty(),
              "the default spec leaves `family` unset (engine default)");
        Check(def.math.u_ell.mode != hecke::UEllMode::Off,
              "and `u_ell` stays at its own default (not `off`)");
    }

    std::map<campaign::ExtCellKey, hecke_ext::HeckeExtRecord> records;
    {
        const hecke::ScopedMathSettings math_guard(spec.math);
        Check(hecke::math_settings().u_ell.mode == hecke::UEllMode::Off,
              "u_ell = off is installed -- the harness never builds U_ell");
        const std::vector<campaign::JobGroup> groups = campaign::plan_campaign(spec);
        const std::vector<campaign::Job> jobs = campaign::flatten(groups);
        const campaign::ExtPassResult pass =
            campaign::RunExtPass(spec, jobs, spec.output_dir, /*verbose=*/false, &records);
        CheckEq(pass.num_failed, uint32_t(0), "the ext pass completes on both cells");
    }
    CheckEq(records.size(), size_t(2), "both cells produced an in-memory record");

    for (const Cell& c : kCells) {
        campaign::ExtCellKey key;
        key.n_sym = c.n;
        key.ell = c.ell;
        const auto it = records.find(key);
        Check(it != records.end(), "the campaign's in-memory record for n=" +
                                        std::to_string(c.n) + " ell=" + std::to_string(c.ell) +
                                        " was captured");
        if (it == records.end()) continue;
        const hecke_ext::HeckeExtRecord& rec = it->second;

        const std::string pt = "n=" + std::to_string(c.n) + " ell=" + std::to_string(c.ell);
        const std::string pe = pt + " ext";

        // --- (a) ext_ideals / ext_certify -- SETTINGS-INDEPENDENT (ruling 5). ---
        {
            const GoldenEntry* e = GoldenLookup(pe, "ext_ideals");
            Check(e != nullptr, pe + " / ext_ideals is in the reference");
            if (e != nullptr)
                CheckEq(GoldenHashExtIdeals(rec), e->hash,
                        pe + " / ext_ideals matches the campaign's in-memory record");
        }
        {
            const GoldenEntry* e = GoldenLookup(pe, "ext_certify");
            Check(e != nullptr, pe + " / ext_certify is in the reference");
            if (e != nullptr)
                CheckEq(GoldenHashExtCertify(rec), e->hash,
                        pe + " / ext_certify matches the campaign's in-memory record");
        }
        // --- sturm_bound -- pipeline-free, exactly as the harness forms it
        // (GoldenHarness.cpp: `SturmBound B(n + 2, ell); B.run();`). ---
        {
            hecke_ext::SturmBound B(c.n + 2, c.ell);
            B.run();
            const GoldenEntry* e = GoldenLookup(pe, "sturm_bound");
            Check(e != nullptr, pe + " / sturm_bound is in the reference");
            if (e != nullptr)
                CheckEq(GoldenHashSturmBound(B), e->hash, pe + " / sturm_bound matches");
        }
        // --- the per-ideal E3 / E4b / E5 collectors: the campaign's ext
        // pass exposes only the HeckeExtRecord, not the HeckeExtDriver that
        // owns `sign_split(q)` / `cone()` -- so, per the pinned scope, they
        // are NOT compared here.
        std::cout << "     n=" << c.n << " ell=" << c.ell
                  << ": E3/E4b/E5 collector objects not compared (the campaign exposes"
                     " HeckeExtRecord, not the driver)\n";

        // --- (c) E2 (a) IN FULL (slot 7, WP-2, stage 1 (2)): the
        // settings-DEPENDENT objects `ext_json_canonical` (tier A) and
        // `ext_json_full` (tier B), recomputed from the campaign's record
        // through the harness's OWN recipe (the top-level `settings` reduced
        // to its fingerprint -- stage 1 (1) -- then the key redaction), are
        // ASSERTED EQUAL to GoldenReference.inc.  Slot 6 recorded a
        // `"source"` difference here; stage 1 (1) took `source` out of the
        // golden text and stage 1 (2) makes the harness stamp `spec`, so the
        // campaign's FULL settings block now equals the harness's too --
        // asserted below against the block the harness installs. ---
        {
            const std::string j =
                GoldenNormalizeJsonWhitespace(hecke_ext::BuildHeckeExtReportJson(rec));
            const std::string r = GoldenReduceSettingsToFingerprint(j);
            const struct { const char* object; std::vector<std::string> keys; } kTexts[] = {
                {"ext_json_canonical", {"commit", "t_p"}},
                {"ext_json_full", {"commit"}},
            };
            for (const auto& t : kTexts) {
                const std::string text = GoldenRedactJsonKeys(r, t.keys);
                const uint64_t campaign_hash = engine::FnvBytes(engine::kFnvOffset64, text);
                const GoldenEntry* e = GoldenLookup(pe, t.object);
                Check(e != nullptr, pe + " / " + t.object + " is in the reference");
                if (e == nullptr) continue;
                std::cout << "     " << pe << " / " << t.object << ": campaign hash 0x"
                          << std::hex << campaign_hash << " vs reference 0x" << e->hash
                          << std::dec << (campaign_hash == e->hash ? " (EQUAL)" : " (DIFFERS)")
                          << "\n";
                CheckEq(campaign_hash, e->hash,
                        pe + " / " + t.object + " -- the campaign's record equals the "
                             "reference (E2 (a), [X] Sec. 1.6)");
                CheckEq(uint64_t(text.size()), e->rows,
                        pe + " / " + t.object + " -- and its text length");
            }
            // The harness's install, stamped as the harness stamps it
            // (GoldenHarness.cpp RunExtCell32: ScopedUEll(u_off, Spec)).
            hecke::MathSettings harness;
            harness.u_ell.mode = hecke::UEllMode::Off;
            harness.u_ell_source = hecke::SettingSource::Spec;
            CheckEq(rec.settings.to_json(), harness.to_json(),
                    pe + " -- the campaign's FULL settings block (values AND sources) equals "
                         "the golden harness's install");
        }
    }
    CleanUp(dir);
}

// ===========================================================================
// Arm 9 -- (hecke-extensions slot 6, WP-6 / E2 close-out) THE EXTENSION
// IDENTITY GAP, [WP5-2] finding 3, owner confirmation 9.
//
// "ext, family, sign, ext_groups do not fingerprint, so two campaigns
// differing only in family share the ext record file and its resume key."
// The fix (the ext tag in the ext record FILE's name, not `Job::id()`):
// two runs into ONE output directory, differing only in `family`, the
// second under `resume = true` -- it must NOT reuse the first's record.
// ===========================================================================
void Arm9ExtIdentityFamily(const std::string& root) {
    std::cout << "-- arm 9: the ext-identity fix -- `family` changes the record file"
              << std::endl;
    const std::string dir = root + "/identity";
    static const char* kBase =
        "name    = ext_identity\n"
        "certify = none\n"
        "ext     = true\n"
        "n_sym   = 2\n"
        "ell     = 5\n"
        "p       = 2\n"
        "r       = 1\n";

    {
        campaign::CampaignSpec spec = SpecFromString(std::string(kBase) + "family = 2,3\n", dir);
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.num_failed, uint64_t(0), "the family={2,3} campaign completes");
        CheckEq(res.ext_ran, uint64_t(1), "and the cell ran");
    }
    {
        campaign::CampaignSpec spec =
            SpecFromString(std::string(kBase) + "family = 2,3,5\nresume = true\n", dir);
        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        CheckEq(res.num_failed, uint64_t(0), "the family={2,3,5} resume completes");
        CheckEq(res.ext_ran, uint64_t(1),
                "a resume under a DIFFERENT family finds no record and RE-RUNS the cell");
        CheckEq(res.ext_reused, uint64_t(0), "and does not reuse the first family's record");
    }
    // Both families' records exist, on DIFFERENT files -- neither overwrote
    // the other, and `Job::id()` did not move (acceptance (a)'s byte-identity
    // arm covers that already: `ext_tag` is not part of `Job::id()`).
    const std::string tag23 =
        SpecFromString(std::string(kBase) + "family = 2,3\n", std::string()).ext.signature_tag();
    const std::string tag235 =
        SpecFromString(std::string(kBase) + "family = 2,3,5\n", std::string()).ext.signature_tag();
    Check(tag23 != tag235, "the two families produce different ext-identity tags");
    Check(campaign::ExtRecordRelPath(2, 5, tag23) != campaign::ExtRecordRelPath(2, 5, tag235),
          "and hence different record file names");
    namespace fs = std::filesystem;
    std::error_code ec;
    Check(fs::exists(dir + "/" + campaign::ExtRecordRelPath(2, 5, tag23), ec),
          "the family={2,3} record file still exists");
    Check(fs::exists(dir + "/" + campaign::ExtRecordRelPath(2, 5, tag235), ec),
          "and the family={2,3,5} record file exists alongside it");
    CleanUp(dir);
}

} // namespace

// ===========================================================================
// Arm 10 -- (hecke-extensions slot 8, WP-6) THE PARALLEL EXT PASS.
//
//   (a) serial == parallel: `ext_jobs` in {1, 2, 4, 16} on ten cells -- every
//       record file byte for byte, summary.tsv outside its timing columns,
//       index.json outside its timing / RSS numbers, and every job file
//       (which carries the record verbatim) outside its timing / RSS numbers;
//   (b) THE NON-DEFAULT-SETTING ARM ([S8] Sec. 2.3 item 10): under `u_ell =
//       off` serial == parallel AND both differ from the default run -- the
//       clause that makes the arm non-vacuous: a worker that silently ran at
//       the DEFAULTS would reproduce the default records and fail here;
//   (c) the admission, observed on the pass: a cap below every cell's
//       prediction runs the cells one at a time (each "over the cap alone"),
//       with the same records; under the default cap `ext_jobs = 4` never
//       has more than 4 cells in flight or more than the cap admitted;
//   (d) the two knobs are scheduling knobs: neither moves a Job::id(), a
//       shape signature, ExtSpec::signature_tag(), the record file name or
//       ExtIndexJson;
//   (e) the cost model and the scheduler on synthetic input (LPT order,
//       first fit under the cap, a cell above the cap never stalls);
//   (f) a parallel `resume` over a complete run reuses every cell and runs
//       no engine.
// ===========================================================================
const char* kParSpecText =
    "name    = ext_parallel\n"
    "certify = none\n"
    "lean    = true\n"
    "level_a = never\n"
    "jobs    = 2\n"
    "threads = 1\n"
    "ext     = true\n"
    "family  = 2,3\n"
    "n_sym   = 2..10 step 2\n"
    "ell     = 5,7\n"
    "p       = 2\n"
    "r       = 1\n";

// Blanks the value of every numeric `"<key>":<number>` whose key is a timing
// or RSS figure, so two runs compare equal iff everything else does.
std::string NormaliseVolatileJson(const std::string& text) {
    static const char* kKeys[] = {"wall_s", "peak_rss_mb", "jobs_per_minute", "rss_mb",
                                  "time_s", "total", "setup", "cohomology", "cosets", "tau",
                                  "theta", "family", "decomp", "external", "report"};
    std::string out = text;
    for (const char* k : kKeys) {
        const std::string needle = std::string("\"") + k + "\":";
        size_t pos = 0;
        while ((pos = out.find(needle, pos)) != std::string::npos) {
            const size_t a = pos + needle.size();
            size_t b = a;
            while (b < out.size() && (std::isdigit(static_cast<unsigned char>(out[b])) ||
                                      out[b] == '.' || out[b] == '-' || out[b] == 'e' ||
                                      out[b] == '+'))
                ++b;
            if (b > a) out.replace(a, b - a, "X");
            pos = a;
        }
    }
    return out;
}

struct ParRun {
    campaign::CampaignResult res;
    std::map<std::string, std::string> records;    // ext/<file> -> bytes
    std::map<std::string, std::string> jobs;       // job id -> normalised JSON
    std::string tsv, index;
};

std::map<std::string, std::string> ReadExtDir(const std::string& dir) {
    std::map<std::string, std::string> out;
    std::error_code ec;
    for (const std::filesystem::directory_entry& e :
         std::filesystem::directory_iterator(dir + "/ext", ec))
        out[e.path().filename().string()] = ReadFile(e.path().string());
    return out;
}

ParRun RunParallelArm(const std::string& dir, const std::string& extra, uint32_t ext_jobs,
                      uint64_t ext_memory_mb, bool resume = false) {
    campaign::CampaignSpec spec = SpecFromString(std::string(kParSpecText) + extra, dir);
    spec.ext_jobs = ext_jobs;
    spec.ext_memory_mb = ext_memory_mb;
    spec.resume = resume;
    ParRun out;
    out.res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
    out.records = ReadExtDir(dir);
    for (const campaign::JobResult& r : out.res.jobs)
        out.jobs[r.job.id()] = NormaliseVolatileJson(ReadFile(r.output_path));
    out.tsv = StripVolatileTsvColumns(ReadFile(dir + "/summary.tsv"));
    out.index = NormaliseVolatileJson(ReadFile(dir + "/index.json"));
    return out;
}

void Arm10ParallelPass(const std::string& root) {
    std::cout << "-- arm 10: the parallel ext pass (slot-8 WP-6)" << std::endl;
    const uint64_t kCap = campaign::kExtMemoryMbDefault;

    // (a) serial against parallel, at the defaults.
    const ParRun ser = RunParallelArm(root + "/par_j1", "", 1, kCap);
    CheckEq(ser.res.num_failed, uint64_t(0), "(a) the serial pass completes");
    CheckEq(ser.res.ext_cells, uint64_t(10), "(a) ten cells");
    CheckEq(ser.res.ext_ran, uint64_t(10), "(a) all ten ran");
    CheckEq(ser.records.size(), size_t(10), "(a) ten record files");
    for (uint32_t J : {2u, 4u, 16u}) {
        const std::string tag = "(a) ext_jobs=" + std::to_string(J);
        const std::string dir = root + "/par_j" + std::to_string(J);
        const ParRun par = RunParallelArm(dir, "", J, kCap);
        CheckEq(par.res.num_failed, uint64_t(0), tag + ": the campaign completes");
        CheckEq(par.res.ext_ran, uint64_t(10), tag + ": all ten ran");
        Check(par.records == ser.records, tag + ": every record file is byte-identical");
        Check(par.tsv == ser.tsv, tag + ": summary.tsv is identical outside its timing columns");
        Check(par.index == ser.index, tag + ": index.json is identical outside timing / RSS");
        Check(par.jobs == ser.jobs, tag + ": every job file is identical outside timing / RSS");
        CleanUp(dir);
    }

    // (b) the non-default-setting arm.
    {
        const ParRun off1 = RunParallelArm(root + "/par_off_j1", "u_ell   = off\n", 1, kCap);
        const ParRun off4 = RunParallelArm(root + "/par_off_j4", "u_ell   = off\n", 4, kCap);
        CheckEq(off4.res.ext_ran, uint64_t(10), "(b) u_ell = off, ext_jobs=4: all ten ran");
        Check(off1.records.size() == 10 && off4.records == off1.records,
              "(b) u_ell = off: serial and parallel records are byte-identical");
        Check(off4.tsv == off1.tsv && off4.jobs == off1.jobs,
              "(b) u_ell = off: summary.tsv and the job files agree");
        uint32_t differ = 0;
        for (const auto& kv : off4.records) {
            const auto it = ser.records.find(kv.first);
            if (it == ser.records.end() || it->second != kv.second) ++differ;
        }
        CheckEq(differ, 10u, "(b) every u_ell = off record differs from the default run's "
                             "(a worker at the defaults would not)");
        for (const auto& kv : off4.records)
            Check(kv.second.find("\"reason\":\"off\"") != std::string::npos,
                  "(b) the parallel record " + kv.first + " carries the u_ell = off verdict");
        CleanUp(root + "/par_off_j1");
        CleanUp(root + "/par_off_j4");
    }

    // (c) the admission, observed on the pass itself.
    {
        campaign::CampaignSpec spec = SpecFromString(kParSpecText, root + "/par_cap");
        const std::vector<campaign::Job> jobs = campaign::flatten(campaign::plan_campaign(spec));
        const hecke::ScopedMathSettings math_guard(spec.math);
        spec.ext_jobs = 4;
        spec.ext_memory_mb = 1;     // below every cell's floor
        const campaign::ExtPassResult tight =
            campaign::RunExtPass(spec, jobs, spec.output_dir, /*verbose=*/false);
        CheckEq(tight.peak_running, uint32_t(1), "(c) a cap below every cell runs one at a time");
        CheckEq(tight.over_cap_alone, uint32_t(10), "(c) and each cell runs alone over the cap");
        CheckEq(tight.num_ran, uint32_t(10), "(c) all ten ran");
        Check(ReadExtDir(spec.output_dir) == ser.records,
              "(c) and the records are the serial pass's, byte for byte");

        spec.ext_memory_mb = kCap;
        const campaign::ExtPassResult wide =
            campaign::RunExtPass(spec, jobs, spec.output_dir + "_wide", /*verbose=*/false);
        Check(wide.peak_running >= 1 && wide.peak_running <= 4,
              "(c) ext_jobs = 4 never has more than 4 cells in flight");
        Check(wide.peak_admitted_mb <= static_cast<double>(kCap),
              "(c) and never admits more than the cap");
        CheckEq(wide.over_cap_alone, uint32_t(0), "(c) no cell is over the default cap");
        CheckEq(wide.jobs, uint32_t(4), "(c) the pass records the ext_jobs it ran with");
        const std::string line = campaign::ExtPassLine(wide);
        Check(line.rfind("EXT_PASS wall_s=", 0) == 0 && line.find(" jobs=4 ") != std::string::npos &&
                  line.find(" peak_rss_mb=") != std::string::npos,
              "(c) the EXT_PASS line carries wall_s, jobs and peak_rss_mb");
        CleanUp(spec.output_dir);
        CleanUp(spec.output_dir + "_wide");
    }

    // (d) the knobs are scheduling knobs.
    {
        const campaign::CampaignSpec a = SpecFromString(kParSpecText, root + "/knob");
        const campaign::CampaignSpec b = SpecFromString(
            std::string(kParSpecText) + "ext_jobs = 7\next_memory_mb = 123\n", root + "/knob");
        CheckEq(b.ext_jobs, uint32_t(7), "(d) `ext_jobs` parses");
        CheckEq(b.ext_memory_mb, uint64_t(123), "(d) `ext_memory_mb` parses");
        CheckEq(a.ext_jobs, uint32_t(1), "(d) `ext_jobs` defaults to 1 (the serial pass)");
        CheckEq(a.ext_memory_mb, campaign::kExtMemoryMbDefault, "(d) `ext_memory_mb` default");
        CheckEq(b.ext.signature_tag(), a.ext.signature_tag(),
                "(d) neither knob enters ExtSpec::signature_tag()");
        const std::vector<campaign::JobGroup> ga = campaign::plan_campaign(a);
        const std::vector<campaign::JobGroup> gb = campaign::plan_campaign(b);
        const std::vector<campaign::Job> ja = campaign::flatten(ga);
        const std::vector<campaign::Job> jb = campaign::flatten(gb);
        bool same = ja.size() == jb.size();
        for (size_t i = 0; same && i < ja.size(); ++i)
            same = ja[i].id() == jb[i].id() && ja[i].ext_tag == jb[i].ext_tag;
        Check(same, "(d) neither knob moves a Job::id() or a job's ext tag");
        bool sigs = ga.size() == gb.size();
        for (size_t i = 0; sigs && i < ga.size(); ++i)
            sigs = ga[i].sig.to_string() == gb[i].sig.to_string();
        Check(sigs, "(d) neither knob moves a shape signature");
        CheckEq(campaign::ExtRecordRelPath(4, 5, b.ext.signature_tag()),
                campaign::ExtRecordRelPath(4, 5, a.ext.signature_tag()),
                "(d) neither knob moves the record file name");
        const campaign::ExtPassResult empty;
        Check(campaign::ExtIndexJson(a, empty) == campaign::ExtIndexJson(b, empty),
              "(d) neither knob appears in index.json's ext block");
    }

    // (e) the cost model and the scheduler, on synthetic input.
    {
        const hecke::UEllSetting auto_u;     // the shipped default (auto, 2048 MB)
        const campaign::ExtCellCost u10007 = campaign::PredictExtCellCost(18, 10007, auto_u);
        const campaign::ExtCellCost over = campaign::PredictExtCellCost(20, 10007, auto_u);
        const campaign::ExtCellCost small = campaign::PredictExtCellCost(40, 5, auto_u);
        Check(u10007.u_ell_admitted && u10007.u_ell_predicted_mb > 2000.0 &&
                  u10007.u_ell_predicted_mb <= 2048.0,
              "(e) U_10007 at n = 18 is admitted by the frozen model, ~2 GB");
        Check(!over.u_ell_admitted && over.memory_mb == double(campaign::kExtCellFloorMb),
              "(e) at n = 20 it is over budget: the cell costs the floor only");
        Check(small.u_ell_admitted && small.memory_mb > double(campaign::kExtCellFloorMb),
              "(e) a small U_5 cell is admitted and costs the floor plus its model");
        hecke::UEllSetting off_u;
        off_u.mode = hecke::UEllMode::Off;
        Check(!campaign::PredictExtCellCost(18, 10007, off_u).u_ell_admitted,
              "(e) under u_ell = off nothing is admitted");

        auto cost = [](uint32_t n, double mb) {
            campaign::ExtCellCost c;
            c.n_sym = n;
            c.u_ell_predicted_mb = mb;
            c.u_ell_admitted = mb > 0.0;
            c.memory_mb = campaign::kExtCellFloorMb + mb;
            return c;
        };
        // Plan order: two small cells, a big and a mid U_ell cell, a small n = 40.
        const std::vector<campaign::ExtCellCost> cs = {cost(2, 0), cost(10, 0), cost(0, 3000),
                                                       cost(4, 1000), cost(40, 0)};
        const std::vector<size_t> want = {2, 3, 4, 1, 0};
        Check(campaign::ExtLptOrder(cs) == want,
              "(e) LPT order: U_ell by predicted MB, then n descending, then plan order");
        // Cap 4100: the big cell (3512) is taken; the mid one (1512) does not
        // fit beside it, so the next in order that fits (n = 40, 512) is; the
        // two hold 4024, so nothing else fits until a done().
        campaign::ExtCellScheduler sched(cs, 4100.0);
        CheckEq(sched.take(), size_t(2), "(e) the scheduler starts the biggest cell");
        CheckEq(sched.take(), size_t(4), "(e) then the first cell in LPT order that fits");
        sched.done(2);
        CheckEq(sched.take(), size_t(3), "(e) a finished cell frees room for the next in order");
        CheckEq(sched.take(), size_t(1), "(e) and the order continues");
        sched.done(4);
        sched.done(3);
        CheckEq(sched.take(), size_t(0), "(e) until the last cell");
        sched.done(1);
        sched.done(0);
        CheckEq(sched.take(), campaign::ExtCellScheduler::kNone, "(e) then none");
        Check(sched.peak_admitted_mb() <= 4100.0, "(e) the cap was never exceeded");
        CheckEq(sched.peak_running(), uint32_t(3), "(e) at most three cells were in flight");
        campaign::ExtCellScheduler alone({cost(0, 9000)}, 4100.0);
        CheckEq(alone.take(), size_t(0), "(e) a cell above the whole cap runs when nothing else does");
        CheckEq(alone.over_cap_alone(), uint32_t(1), "(e) and is counted as over the cap");
        alone.done(0);
    }

    // (f) a parallel resume over the complete serial run reuses every cell.
    {
        const ParRun again = RunParallelArm(root + "/par_j1", "", 4, kCap, /*resume=*/true);
        CheckEq(again.res.ext_reused, uint64_t(10), "(f) a parallel resume reuses all ten cells");
        CheckEq(again.res.ext_ran, uint64_t(0), "(f) and runs no engine");
        Check(again.records == ser.records, "(f) and leaves every record file as it was");
    }
    CleanUp(root + "/par_j1");
}


// (slot-8 WP-6b) The pass's own record of one ext pass, under `extra`.
campaign::ExtPassResult RunPassOnly(const std::string& dir, const std::string& extra, uint32_t ext_jobs) {
    campaign::CampaignSpec spec = SpecFromString(std::string(kParSpecText) + extra, dir);
    spec.ext_jobs = ext_jobs;
    const std::vector<campaign::Job> jobs = campaign::flatten(campaign::plan_campaign(spec));
    const hecke::ScopedMathSettings math_guard(spec.math);
    return campaign::RunExtPass(spec, jobs, spec.output_dir, /*verbose=*/false);
}

std::string GpuCellsText(const campaign::ExtPassResult& p) {
    std::string t;
    for (const campaign::ExtCellKey& k : p.gpu_cells)
        t += (t.empty() ? "" : " ") + std::to_string(k.n_sym) + ":" + std::to_string(k.ell);
    return t;
}

// ---------------------------------------------------------------------------
// Arm 11 (slot-8 WP-6b, optional): THE DEVICE WORKER.  `ext_gpu = force` runs
// the worker whether or not a device exists (without one its GPU-assigned
// nodes fall back to the CPU in the executor), so the plumbing is exercised
// on every box; `ext_gpu = on` needs a usable device and is [SKIP]ped (its
// fallback asserted) without one.  The oracle is the serial CPU-only pass,
// byte for byte; the non-vacuity is the pass's own count of the cells the
// device worker ran and of the GPU-assigned nodes in the programs each class
// of worker ran (zero on every CPU worker).
// ---------------------------------------------------------------------------
void Arm11DeviceWorker(const std::string& root) {
    std::cout << "-- arm 11: the device worker (slot-8 WP-6b)" << std::endl;
    const uint64_t kCap = campaign::kExtMemoryMbDefault;
    const bool device = engine::cuda::available();
    const ParRun ser = RunParallelArm(root + "/gpu_j1", "", 1, kCap);
    CheckEq(ser.res.ext_ran, uint64_t(10), "(a) the serial CPU-only pass ran all ten cells");
    const std::string listed = "ext_gpu_cells = 2:5, 6:7, 10:7\n";

    std::vector<std::string> modes = {"force"};
    if (device) modes.push_back("on");
    else std::cout << "  [SKIP] no usable CUDA device (" << engine::cuda::unavailable_reason()
                   << "): the `ext_gpu = on` arms run the fallback check only" << std::endl;
    for (const std::string& mode : modes) {
        const std::string extra = "ext_gpu = " + mode + "\n";
        for (uint32_t J : {1u, 2u}) {
            const std::string tag = "(a) ext_gpu=" + mode + " ext_jobs=" + std::to_string(J) + " (listed)";
            const std::string dir = root + "/gpu_" + mode + "_j" + std::to_string(J);
            const ParRun par = RunParallelArm(dir, extra + listed, J, kCap);
            CheckEq(par.res.num_failed, uint64_t(0), tag + ": the campaign completes");
            Check(par.records == ser.records, tag + ": every record file is byte-identical to the CPU-only pass");
            Check(par.tsv == ser.tsv, tag + ": summary.tsv is identical outside its timing columns");
            Check(par.index == ser.index, tag + ": index.json is identical outside timing / RSS");
            Check(par.jobs == ser.jobs, tag + ": every job file is identical outside timing / RSS");
            CleanUp(dir);
            const campaign::ExtPassResult pass = RunPassOnly(dir + "_pass", extra + listed, J);
            Check(pass.gpu_worker, tag + ": the device worker ran");
            CheckEq(GpuCellsText(pass), std::string("2:5 6:7 10:7"),
                    tag + ": it ran exactly the listed cells (non-vacuity)");
            Check(pass.gpu_programs > 0 && pass.gpu_program_gpu_nodes > 0,
                  tag + ": its programs carry GPU-assigned nodes");
            Check(pass.cpu_programs > 0 && pass.cpu_program_gpu_nodes == 0,
                  tag + ": every CPU worker's programs carry ZERO GPU-assigned nodes");
            Check(ReadExtDir(dir + "_pass") == ser.records, tag + ": the pass-only records agree too");
            const std::string line = campaign::ExtPassLine(pass);
            Check(line.find(" ext_gpu=" + mode + " gpu_worker=1 gpu_cells=3 ") != std::string::npos &&
                      line.find(" cpu_gpu_nodes=0 ") != std::string::npos,
                  tag + ": the EXT_PASS line reports the device worker");
            CleanUp(dir + "_pass");
        }
        {
            // The worker's own rule (no list): values identical whatever it takes.
            const std::string tag = "(a) ext_gpu=" + mode + " ext_jobs=2 (own rule)";
            const campaign::ExtPassResult pass = RunPassOnly(root + "/gpu_rule", extra, 2);
            Check(pass.gpu_worker && pass.cpu_program_gpu_nodes == 0,
                  tag + ": the device worker ran; the CPU workers ran zero GPU nodes");
            Check(ReadExtDir(root + "/gpu_rule") == ser.records, tag + ": every record file is byte-identical");
            std::cout << "  " << tag << ": the device worker took " << pass.gpu_cells.size()
                      << " cell(s) [" << GpuCellsText(pass) << "] (timing-dependent; reported)" << std::endl;
            CleanUp(root + "/gpu_rule");
        }
    }
    if (!device) {
        const campaign::ExtPassResult pass = RunPassOnly(root + "/gpu_fallback", "ext_gpu = on\n", 2);
        Check(!pass.gpu_worker && !pass.gpu_fallback.empty(),
              "(no device) ext_gpu = on falls back to the CPU-only pass, naming why");
        Check(pass.gpu_cells.empty() && pass.gpu_programs == 0, "(no device) and runs no device worker");
        Check(ReadExtDir(root + "/gpu_fallback") == ser.records, "(no device) with identical records");
        CleanUp(root + "/gpu_fallback");
    }

    // (d) the non-default-setting arm with the device worker.
    {
        const std::string off = "u_ell   = off\n";
        const ParRun off1 = RunParallelArm(root + "/gpu_off_j1", off, 1, kCap);
        const std::string mode = device ? "on" : "force";
        const ParRun offg = RunParallelArm(root + "/gpu_off_g", off + "ext_gpu = " + mode + "\n" + listed, 2, kCap);
        Check(off1.records.size() == 10 && offg.records == off1.records,
              "(d) u_ell = off, ext_gpu = " + mode + ": records identical to the CPU-only run under the same setting");
        Check(offg.tsv == off1.tsv && offg.jobs == off1.jobs && offg.index == off1.index,
              "(d) u_ell = off: summary.tsv, index.json and the job files agree");
        uint32_t differ = 0;
        for (const auto& kv : offg.records) {
            const auto it = ser.records.find(kv.first);
            if (it == ser.records.end() || it->second != kv.second) ++differ;
        }
        CheckEq(differ, 10u, "(d) and every record differs from the default run's");
        CleanUp(root + "/gpu_off_j1");
        CleanUp(root + "/gpu_off_g");
    }

    // (f) the knobs are scheduling knobs; `off` is WP-6's pass.
    {
        const campaign::CampaignSpec a = SpecFromString(kParSpecText, root + "/gknob");
        const campaign::CampaignSpec b = SpecFromString(
            std::string(kParSpecText) + "ext_gpu = on\next_gpu_cells = 2:5 4:7\n", root + "/gknob");
        Check(a.ext_gpu == campaign::ExtGpuMode::Off && a.ext_gpu_cells.empty(), "(f) `ext_gpu` defaults to off");
        Check(b.ext_gpu == campaign::ExtGpuMode::On && b.ext_gpu_cells.size() == 2 &&
                  b.ext_gpu_cells[1].first == 4 && b.ext_gpu_cells[1].second == 7,
              "(f) `ext_gpu` / `ext_gpu_cells` parse");
        CheckEq(b.ext.signature_tag(), a.ext.signature_tag(), "(f) neither knob enters ExtSpec::signature_tag()");
        const std::vector<campaign::JobGroup> ga = campaign::plan_campaign(a);
        const std::vector<campaign::JobGroup> gb = campaign::plan_campaign(b);
        const std::vector<campaign::Job> ja = campaign::flatten(ga);
        const std::vector<campaign::Job> jb = campaign::flatten(gb);
        bool same = ja.size() == jb.size() && ga.size() == gb.size();
        for (size_t i = 0; same && i < ja.size(); ++i)
            same = ja[i].id() == jb[i].id() && ja[i].ext_tag == jb[i].ext_tag;
        for (size_t i = 0; same && i < ga.size(); ++i) same = ga[i].sig.to_string() == gb[i].sig.to_string();
        Check(same, "(f) neither knob moves a Job::id(), an ext tag or a shape signature");
        CheckEq(campaign::ExtRecordRelPath(4, 5, b.ext.signature_tag()),
                campaign::ExtRecordRelPath(4, 5, a.ext.signature_tag()),
                "(f) neither knob moves the record file name");
        const campaign::ExtPassResult empty;
        Check(campaign::ExtIndexJson(a, empty) == campaign::ExtIndexJson(b, empty),
              "(f) neither knob appears in index.json's ext block");
        Check(campaign::ExtPassLine(empty).find("ext_gpu") == std::string::npos,
              "(f) an `ext_gpu = off` EXT_PASS line is WP-6's (no device fields)");
        bool threw = false;
        try { SpecFromString(std::string(kParSpecText) + "ext_gpu_cells = 2:5\n", root + "/gknob"); }
        catch (const std::exception&) { threw = true; }
        Check(threw, "(f) `ext_gpu_cells` without `ext_gpu` is refused at parse time");
        threw = false;
        try { SpecFromString(std::string(kParSpecText) + "ext_gpu = hybrid\n", root + "/gknob"); }
        catch (const std::exception&) { threw = true; }
        Check(threw, "(f) `ext_gpu = hybrid` is refused (gpu only: hybrid shares the cache-key bit)");
        const campaign::ExtPassResult off = RunPassOnly(root + "/gpu_offmode", "", 2);
        Check(!off.gpu_worker && off.gpu_programs == 0 && off.cpu_programs == 0,
              "(f) `ext_gpu = off` installs no device override and no tally");
        Check(ReadExtDir(root + "/gpu_offmode") == ser.records, "(f) and its records are the serial pass's");
        CleanUp(root + "/gpu_offmode");
    }

    // (e) the device worker's rule, on synthetic input (never blocks: every
    // take(Gpu) below either returns or finds its condition met).
    {
        auto cost = [](uint32_t n, double mb) {
            campaign::ExtCellCost c;
            c.n_sym = n;
            c.u_ell_predicted_mb = mb;
            c.u_ell_admitted = mb > 0.0;
            c.memory_mb = campaign::kExtCellFloorMb + mb;
            return c;
        };
        // Plan index 0: a U_ell cell (3000 MB); 1..11: cells without U_ell,
        // n = 40, 30, 20, 10, 8, 6, 4, 2, 1, 0, 3.  LPT: 0, then n descending,
        // so the tail (the cheapest) is n = 0 (index 10), then n = 1 (index 9).
        std::vector<campaign::ExtCellCost> cs = {cost(0, 3000)};
        for (uint32_t n : {40u, 30u, 20u, 10u, 8u, 6u, 4u, 2u, 1u, 0u, 3u}) cs.push_back(cost(n, 0));
        {
            campaign::ExtCellScheduler sch(cs, 1.0e9);
            sch.enable_device_worker(1, {}, 0.0);
            CheckEq(sch.take(campaign::ExtWorkerClass::Cpu), size_t(0), "(e) the CPU worker starts the U_ell cell");
            CheckEq(sch.take(campaign::ExtWorkerClass::Gpu), size_t(10),
                    "(e) with every CPU slot busy the device takes the cheapest cell (the LPT tail)");
            CheckEq(sch.take(campaign::ExtWorkerClass::Gpu), size_t(9), "(e) and then the next cheapest");
        }
        {
            // The modelled cost guard: 3.3 x 512 MB > the rest per CPU slot.
            campaign::ExtCellScheduler sch({cost(0, 3000), cost(2, 0), cost(4, 0)}, 1.0e9);
            sch.enable_device_worker(1, {}, 0.0);
            CheckEq(sch.take(campaign::ExtWorkerClass::Cpu), size_t(0), "(e) guard: the CPU takes the big cell");
            CheckEq(sch.take(campaign::ExtWorkerClass::Gpu), campaign::ExtCellScheduler::kNone,
                    "(e) guard: the device never takes a cell the model says would be the tail");
        }
        {
            // The device-memory cap.
            campaign::ExtCellScheduler sch(cs, 1.0e9);
            sch.enable_device_worker(1, {}, 100.0);
            CheckEq(sch.take(campaign::ExtWorkerClass::Cpu), size_t(0), "(e) cap: the CPU takes the big cell");
            CheckEq(sch.take(campaign::ExtWorkerClass::Gpu), campaign::ExtCellScheduler::kNone,
                    "(e) cap: no cell fits a 100 MB device cap");
        }
        {
            // An explicit list: the device takes exactly it; no CPU worker does.
            std::vector<char> l(cs.size(), 0);
            l[3] = l[5] = 1;
            campaign::ExtCellScheduler sch(cs, 1.0e9);
            sch.enable_device_worker(2, l, 0.0);
            CheckEq(sch.take(campaign::ExtWorkerClass::Gpu), size_t(3), "(e) list: the device takes the first listed (LPT)");
            CheckEq(sch.take(campaign::ExtWorkerClass::Gpu), size_t(5), "(e) list: then the second");
            CheckEq(sch.take(campaign::ExtWorkerClass::Gpu), campaign::ExtCellScheduler::kNone, "(e) list: then none");
            std::set<size_t> cpu;
            for (;;) {
                const size_t k = sch.take(campaign::ExtWorkerClass::Cpu);
                if (k == campaign::ExtCellScheduler::kNone) break;
                cpu.insert(k);
            }
            Check(cpu.size() == cs.size() - 2 && !cpu.count(3) && !cpu.count(5),
                  "(e) list: the CPU workers take every other cell and never a listed one");
        }
    }
    CleanUp(root + "/gpu_j1");
}

// ===========================================================================
uint32_t HeckeExtCampaignValidation() {
    std::cout << "--- HeckeExtCampaignValidation (hecke-extensions slot 5, WP-2 / E2) ---"
              << std::endl;
    g_failures = 0;
    g_checks = 0;
    const std::string root = TempDir("main");
    try {
        Arm1Plan(root);
        Arm2Default(root);
        std::string seq_summary;
        const std::vector<std::string> seq_records = Arm3Attach(root, seq_summary);
        Arm4Splits(root, seq_records, seq_summary);
        Arm5Resume(root);
        Arm6Sidecar(root);
        Arm7ReuseGuardAndWidth(root);
        Arm8GoldenSubgrid(root);
        Arm9ExtIdentityFamily(root);
        Arm10ParallelPass(root);
        Arm11DeviceWorker(root);
    } catch (const std::exception& e) {
        std::cout << "  FAIL: exception: " << e.what() << std::endl;
        ++g_failures;
    }
    CleanUp(root);
    std::cout << "EXT_CAMPAIGN checks=" << g_checks << " failures=" << g_failures << std::endl;
    std::cout << (g_failures == 0 ? "  HeckeExtCampaignValidation PASSED"
                                  : "  HeckeExtCampaignValidation FAILED")
              << " (" << g_checks << " checks, " << g_failures << " failing)" << std::endl;
    return g_failures;
}
