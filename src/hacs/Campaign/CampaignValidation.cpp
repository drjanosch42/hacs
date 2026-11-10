// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// matrix-engine Phase 6 -- campaign validations. See CampaignValidation.h.

#include "CampaignValidation.h"

#include "CampaignDriver.h"
#include "../Golden/GoldenHarness.h"
#include "CampaignSpec.h"
#include "../Util/TextIo.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>   // getpid, for a unique scratch directory

namespace {

// The frozen golden reference: the single compiled copy, kGoldenReference of Golden/GoldenHarness.h.

std::string TempDir(const std::string& tag) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    const fs::path dir = (ec ? fs::path(".") : base) /
                         ("oc_campaign_" + tag + "_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    return dir.string();
}

void CleanUp(const std::string& dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

campaign::CampaignSpec SpecFromString(const std::string& text, const std::string& out_dir) {
    std::istringstream is(text);
    campaign::CampaignSpec spec = campaign::ParseCampaignSpec(is, "<validation-spec>");
    spec.output_dir = out_dir;
    return spec;
}

using oc_util::HexU64;
using oc_util::ReadFile;   // S24 CL-7 (F11): Util/TextIo.h

// summary.tsv with its two volatile columns (time_s, rss_mb) blanked, so two
// runs of the same spec can be compared byte for byte on everything that IS a
// computed value.  Columns are the header's, 0-based: 18 = time_s, 19 = rss_mb.
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

bool SameObjects(const std::vector<campaign::CanonicalObject>& a,
                 const std::vector<campaign::CanonicalObject>& b, std::string& why) {
    if (a.size() != b.size()) {
        why = "object count " + std::to_string(a.size()) + " vs " + std::to_string(b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].point != b[i].point || a[i].object != b[i].object || a[i].tier != b[i].tier ||
            a[i].rows != b[i].rows || a[i].cols != b[i].cols || a[i].hash != b[i].hash) {
            why = "object #" + std::to_string(i) + " '" + a[i].point + "' / '" + a[i].object +
                  "': " + HexU64(a[i].hash) + " vs " + HexU64(b[i].hash);
            return false;
        }
    }
    return true;
}

} // namespace

// (slot-7 WP-8) See CampaignValidation.h.  The spec CampaignSmokeValidation
// runs, verbatim.
const char* CampaignSmokeSpecText() {
    return
        "name    = campaign_smoke\n"
        "model   = sl2z_7vertex\n"
        "certify = full\n"
        "lean    = false\n"
        "n_sym   = 0,2,4,10,16\n"
        "ell     = 5,134217757\n"
        "p       = 2\n"
        "r       = 1\n"
        "---\n"
        "n_sym   = 0,2\n"
        "ell     = 2\n"
        "p       = 2,3\n"
        "r       = 1,2\n"
        "---\n"
        "n_sym   = 4\n"
        "ell     = 134217757\n"
        "p       = 3\n"
        "r       = 1,2\n";
}

// ===========================================================================
// CampaignSmokeValidation
// ===========================================================================
uint32_t CampaignSmokeValidation() {
    std::cout << "--- CampaignSmokeValidation (matrix-engine Phase 6) ---" << std::endl;
    uint32_t failures = 0;
    uint64_t checks = 0;

    // 20 jobs, all of them ALSO points of the Golden/ grid, so every hash can
    // be cross-checked against the frozen reference:
    //   10 = {0,2,4,10,16} x {5, 134217757} x {2} x {1}
    //    8 = {0,2}         x {2}            x {2,3} x {1,2}
    //    2 = {4}           x {134217757}    x {3}   x {1,2}
    // (slot-7 WP-8) The text is CampaignSmokeSpecText(), file scope, so the
    // console's acceptance (d) can run the very same spec.
    const char* const kSpec = CampaignSmokeSpecText();

    const std::string root = TempDir("smoke");
    try {
        campaign::CampaignSpec spec = SpecFromString(kSpec, root + "/campaign");

        // --- the plan: shape-signature groups, deterministic order ----------
        const std::vector<campaign::JobGroup> groups = campaign::plan_campaign(spec);
        const std::vector<campaign::Job> jobs = campaign::flatten(groups);
        ++checks;
        if (jobs.size() != 20) {
            std::cout << "  FAIL: expected 20 jobs, planned " << jobs.size() << std::endl;
            ++failures;
        }
        // Every group must be internally shape-uniform and ell-sorted -- the
        // contract the Phase 4 batching hook relies on.
        for (const campaign::JobGroup& g : groups) {
            uint64_t prev_ell = 0;
            for (const campaign::Job& j : g.jobs) {
                ++checks;
                if (j.n_sym != g.sig.n_sym || j.p != g.sig.p || j.r != g.sig.r ||
                    j.model != g.sig.model || j.degrees != g.sig.degrees) {
                    std::cout << "  FAIL: job " << j.to_string() << " does not match its group "
                              << g.sig.to_string() << std::endl;
                    ++failures;
                }
                ++checks;
                if (j.ell <= prev_ell && prev_ell != 0) {
                    std::cout << "  FAIL: group " << g.sig.to_string()
                              << " is not sorted by ell" << std::endl;
                    ++failures;
                }
                prev_ell = j.ell;
            }
        }
        // plan_campaign() must be a pure function of the spec.
        {
            const std::vector<campaign::Job> again = campaign::flatten(campaign::plan_campaign(spec));
            ++checks;
            if (again.size() != jobs.size()) { std::cout << "  FAIL: plan is not deterministic\n"; ++failures; }
            else for (size_t i = 0; i < jobs.size(); ++i) {
                if (again[i].id() != jobs[i].id() || again[i].index != jobs[i].index) {
                    std::cout << "  FAIL: plan is not deterministic at #" << i << std::endl;
                    ++failures;
                    break;
                }
            }
        }

        // --- REGRESSION (2026-09-16): the degree list is part of the id -----
        // `plan_campaign` de-duplicates by (n_sym, ell, p, r, degrees), but
        // Job::id() used to be "n<N>_ell<L>_p<P>_r<R>" only, and the id IS the
        // output file's stem and the resume key -- so two jobs differing only
        // in the reported degrees wrote to the same file and mis-resumed.
        // The per-job "shape_signature" string had the same defect (it dropped
        // the `q=` component that the group signature carries).
        // See docs/implementation/oc1_decision_brief.md Sec. 4.4.
        {
            const std::vector<uint32_t> def = campaign::ModelDefaultDegrees("sl2z_7vertex");
            ++checks;
            if (def != std::vector<uint32_t>({0, 1, 2})) {
                std::cout << "  FAIL: the sl2z_7vertex default degree list is not {0,1,2}"
                          << std::endl;
                ++failures;
            }

            // (1) Two jobs equal but for the degree list.
            campaign::Job a;
            a.n_sym = 16; a.ell = 5; a.p = 2; a.r = 1; a.degrees = def;
            campaign::Job b = a;
            b.degrees = {1};
            ++checks;
            if (a.id() == b.id()) {
                std::cout << "  FAIL: jobs differing only in degrees share the id '" << a.id()
                          << "'" << std::endl;
                ++failures;
            }
            ++checks;
            if (a.shape().to_string() == b.shape().to_string()) {
                std::cout << "  FAIL: jobs differing only in degrees share the shape signature '"
                          << a.shape().to_string() << "'" << std::endl;
                ++failures;
            }
            ++checks;
            if (a.shape() == b.shape() || !(a.shape() < b.shape() || b.shape() < a.shape())) {
                std::cout << "  FAIL: ShapeSignature does not compare/order on the degree list"
                          << std::endl;
                ++failures;
            }
            // The RESUME path is derived from the id, so it must separate too.
            ++checks;
            if (("jobs/" + a.id() + ".json") == ("jobs/" + b.id() + ".json")) {
                std::cout << "  FAIL: the two jobs resolve to the same resume path" << std::endl;
                ++failures;
            }
            // (2) BACKWARD COMPATIBILITY: the default degree list keeps the
            // pre-fix id verbatim, so every published output path, index.json
            // entry and resume state under docs/matrix-engine/results/ stands.
            ++checks;
            if (a.id() != "n16_ell5_p2_r1") {
                std::cout << "  FAIL: the default-degree id changed: '" << a.id()
                          << "' != 'n16_ell5_p2_r1'" << std::endl;
                ++failures;
            }
            ++checks;
            if (b.id() != "n16_ell5_p2_r1_q1") {
                std::cout << "  FAIL: unexpected non-default-degree id '" << b.id() << "'"
                          << std::endl;
                ++failures;
            }

            // (3) INJECTIVITY on a small grid that includes the degree list:
            // 2 x 2 x 2 x 2 x 7 = 112 distinct points, 112 distinct ids.
            {
                std::map<std::string, std::string> by_id;
                std::map<std::string, std::string> by_sig;
                uint64_t points = 0, id_collisions = 0, sig_collisions = 0;
                const std::vector<std::vector<uint32_t>> kDegreeSets = {
                    {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}
                };
                for (uint32_t n : {uint32_t(0), uint32_t(2)})
                    for (uint64_t e : {uint64_t(2), uint64_t(134217757)})
                        for (uint64_t pp : {uint64_t(2), uint64_t(3)})
                            for (uint32_t rr : {uint32_t(1), uint32_t(2)})
                                for (const std::vector<uint32_t>& dg : kDegreeSets) {
                                    campaign::Job j;
                                    j.n_sym = n; j.ell = e; j.p = pp; j.r = rr; j.degrees = dg;
                                    std::ostringstream key;
                                    key << n << "|" << e << "|" << pp << "|" << rr << "|";
                                    for (uint32_t q : dg) key << q << ",";
                                    ++points;
                                    auto ins = by_id.emplace(j.id(), key.str());
                                    if (!ins.second && ins.first->second != key.str()) {
                                        if (++id_collisions <= 5)
                                            std::cout << "  FAIL: id '" << j.id()
                                                      << "' is shared by points " << key.str()
                                                      << " and " << ins.first->second << std::endl;
                                    }
                                    // The shape signature drops `ell` by design,
                                    // so key it on everything but ell.
                                    std::ostringstream skey;
                                    skey << n << "|" << pp << "|" << rr << "|";
                                    for (uint32_t q : dg) skey << q << ",";
                                    auto sins = by_sig.emplace(j.shape().to_string(), skey.str());
                                    if (!sins.second && sins.first->second != skey.str()) {
                                        if (++sig_collisions <= 5)
                                            std::cout << "  FAIL: shape signature '"
                                                      << j.shape().to_string()
                                                      << "' is shared by " << skey.str() << " and "
                                                      << sins.first->second << std::endl;
                                    }
                                }
                ++checks;
                if (points != 112 || id_collisions != 0) {
                    std::cout << "  FAIL: Job::id() is not injective on the " << points
                              << "-point degree grid (" << id_collisions << " collisions)"
                              << std::endl;
                    ++failures;
                }
                ++checks;
                if (sig_collisions != 0) {
                    std::cout << "  FAIL: ShapeSignature::to_string() is not injective ("
                              << sig_collisions << " collisions)" << std::endl;
                    ++failures;
                }
                std::cout << "  job-id injectivity: " << points
                          << " points (degree list included), " << by_id.size()
                          << " distinct ids, " << by_sig.size() << " distinct shape signatures"
                          << std::endl;
            }

            // (4) END TO END: the same (n, ell, p, r) at two different degree
            // lists in ONE campaign must produce two files and two shape groups,
            // and `resume` must then skip all four jobs.
            {
                static const char* kDegSpec =
                    "name    = campaign_degree_id\n"
                    "certify = none\n"
                    "n_sym   = 0,2\n"
                    "ell     = 5\n"
                    "p       = 2\n"
                    "r       = 1\n"
                    "---\n"
                    "n_sym   = 0,2\n"
                    "ell     = 5\n"
                    "p       = 2\n"
                    "r       = 1\n"
                    "q       = 1\n";
                campaign::CampaignSpec dspec = SpecFromString(kDegSpec, root + "/degid");
                const std::vector<campaign::JobGroup> dgroups = campaign::plan_campaign(dspec);
                const std::vector<campaign::Job> djobs = campaign::flatten(dgroups);
                ++checks;
                if (djobs.size() != 4 || dgroups.size() != 4) {
                    std::cout << "  FAIL: the two-degree-list spec planned " << djobs.size()
                              << " jobs in " << dgroups.size() << " groups, expected 4 and 4"
                              << std::endl;
                    ++failures;
                }
                const campaign::CampaignResult dr =
                    campaign::RunCampaignSpec(dspec, /*verbose=*/false);
                ++checks;
                if (dr.jobs.size() != 4 || dr.num_failed != 0) {
                    std::cout << "  FAIL: the two-degree-list campaign did not complete ("
                              << dr.jobs.size() << " jobs, " << dr.num_failed << " failed)"
                              << std::endl;
                    ++failures;
                } else {
                    std::map<std::string, uint32_t> paths;
                    for (const auto& r : dr.jobs) ++paths[r.output_path];
                    ++checks;
                    if (paths.size() != 4) {
                        std::cout << "  FAIL: 4 jobs wrote only " << paths.size()
                                  << " distinct files -- the degree list is not in the id"
                                  << std::endl;
                        ++failures;
                    }
                    for (const auto& r : dr.jobs) {
                        ++checks;
                        if (!campaign::JobOutputComplete(r.output_path)) {
                            std::cout << "  FAIL: no trailer in " << r.output_path << std::endl;
                            ++failures;
                        }
                        // The per-job JSON's shape_signature must carry `q=`.
                        const std::string want =
                            "\"shape_signature\":\"" + r.job.shape().to_string() + "\"";
                        ++checks;
                        if (ReadFile(r.output_path).find(want) == std::string::npos) {
                            std::cout << "  FAIL: " << r.output_path << " does not carry " << want
                                      << std::endl;
                            ++failures;
                        }
                    }
                }
                // resume: all four are complete, so all four must be skipped.
                campaign::CampaignSpec dspec2 = dspec;
                dspec2.resume = true;
                const campaign::CampaignResult rr2 =
                    campaign::RunCampaignSpec(dspec2, /*verbose=*/false);
                ++checks;
                if (rr2.num_skipped != 4 || rr2.num_ran != 0) {
                    std::cout << "  FAIL: resume over two degree lists skipped "
                              << rr2.num_skipped << " and ran " << rr2.num_ran
                              << ", expected 4 and 0" << std::endl;
                    ++failures;
                }
                CleanUp(dspec.output_dir);
            }
        }

        // --- the campaign run ----------------------------------------------
        const campaign::CampaignResult batch = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        ++checks;
        if (batch.num_failed != 0) {
            std::cout << "  FAIL: " << batch.num_failed << " campaign job(s) failed" << std::endl;
            for (const auto& r : batch.jobs)
                if (!r.ok) std::cout << "        " << r.job.to_string() << ": " << r.error << std::endl;
            ++failures;
        }
        ++checks;
        if (batch.jobs.size() != jobs.size()) {
            std::cout << "  FAIL: campaign ran " << batch.jobs.size() << " jobs, planned "
                      << jobs.size() << std::endl;
            ++failures;
        }

        // Every job file must exist and carry the completion trailer.
        for (const auto& r : batch.jobs) {
            ++checks;
            if (!campaign::JobOutputComplete(r.output_path)) {
                std::cout << "  FAIL: no completion trailer in " << r.output_path << std::endl;
                ++failures;
            }
        }
        ++checks;
        if (ReadFile(spec.output_dir + "/index.json").find("\"schema\":\"campaign_index/v1\"")
                == std::string::npos) {
            std::cout << "  FAIL: index.json missing or malformed" << std::endl;
            ++failures;
        }
        ++checks;
        {
            const std::string tsv = ReadFile(spec.output_dir + "/summary.tsv");
            size_t rows = 0;
            for (char c : tsv) if (c == '\n') ++rows;
            if (rows != jobs.size() + 1) {   // header + one row per job
                std::cout << "  FAIL: summary.tsv has " << rows << " lines, expected "
                          << (jobs.size() + 1) << std::endl;
                ++failures;
            }
        }

        // --- 20 individual runs, hash for hash ------------------------------
        uint64_t objects_compared = 0;
        for (size_t i = 0; i < batch.jobs.size(); ++i) {
            const campaign::JobResult& b = batch.jobs[i];
            const campaign::CampaignSpec one =
                campaign::SingleJobSpec(spec, b.job, root + "/single_" + std::to_string(i));
            const campaign::CampaignResult solo = campaign::RunCampaignSpec(one, /*verbose=*/false);
            ++checks;
            if (solo.jobs.size() != 1 || !solo.jobs[0].ok) {
                std::cout << "  FAIL: individual run of " << b.job.to_string() << " failed"
                          << std::endl;
                ++failures;
                continue;
            }
            std::string why;
            ++checks;
            if (!SameObjects(b.objects, solo.jobs[0].objects, why)) {
                std::cout << "  FAIL: batched vs individual differ for " << b.job.to_string()
                          << ": " << why << std::endl;
                ++failures;
            }
            ++checks;
            if (b.dim_H != solo.jobs[0].dim_H) {
                std::cout << "  FAIL: dim_H differs for " << b.job.to_string() << std::endl;
                ++failures;
            }
            objects_compared += b.objects.size();
            CleanUp(one.output_dir);
        }
        std::cout << "  batched == individual on " << objects_compared << " objects across "
                  << batch.jobs.size() << " jobs" << std::endl;

        // --- against the frozen golden reference ----------------------------
        std::map<std::string, const GoldenEntry*> golden;
        for (const GoldenEntry* e = kGoldenReference; e->point != nullptr; ++e)
            golden[std::string(e->point) + "\x1f" + e->object] = e;
        uint64_t golden_hits = 0, golden_mismatch = 0;
        for (const auto& r : batch.jobs) {
            for (const campaign::CanonicalObject& o : r.objects) {
                auto it = golden.find(o.point + std::string("\x1f") + o.object);
                if (it == golden.end()) continue;
                ++golden_hits;
                ++checks;
                if (it->second->hash != o.hash || it->second->rows != o.rows ||
                    it->second->cols != o.cols) {
                    ++golden_mismatch;
                    if (golden_mismatch <= 10) {
                        std::cout << "  FAIL: golden mismatch '" << o.point << "' / '" << o.object
                                  << "': campaign " << HexU64(o.hash) << " (" << o.rows << "x"
                                  << o.cols << ") vs reference " << HexU64(it->second->hash) << " ("
                                  << it->second->rows << "x" << it->second->cols << ")"
                                  << std::endl;
                    }
                }
            }
        }
        if (golden_mismatch != 0) ++failures;
        ++checks;
        if (golden_hits < 500) {
            std::cout << "  FAIL: only " << golden_hits
                      << " campaign objects matched a GoldenReference.inc key -- the campaign's "
                         "point/object vocabulary has drifted from the golden harness's"
                      << std::endl;
            ++failures;
        }
        std::cout << "  golden cross-check: " << golden_hits << " shared objects, "
                  << golden_mismatch << " mismatching" << std::endl;

        // --- memory-lean mode is value-identical -----------------------------
        {
            campaign::CampaignSpec lean = spec;
            lean.lean = true;
            lean.output_dir = root + "/lean";
            const campaign::CampaignResult lr = campaign::RunCampaignSpec(lean, /*verbose=*/false);
            ++checks;
            if (lr.jobs.size() != batch.jobs.size() || lr.num_failed != 0) {
                std::cout << "  FAIL: the lean-mode campaign did not complete" << std::endl;
                ++failures;
            } else {
                for (size_t i = 0; i < lr.jobs.size(); ++i) {
                    std::string why;
                    ++checks;
                    if (!SameObjects(batch.jobs[i].objects, lr.jobs[i].objects, why)) {
                        std::cout << "  FAIL: lean vs resident differ for "
                                  << batch.jobs[i].job.to_string() << ": " << why << std::endl;
                        ++failures;
                    }
                }
                std::cout << "  memory-lean mode == resident mode on every object" << std::endl;
            }
            CleanUp(lean.output_dir);
        }

        // --- the BATCHING SEAM: concurrent == sequential ---------------------
        // Phase 6 Part A. The same spec run with `jobs` of each shape group's
        // jobs in flight at once, each on its own engine pool of `threads`
        // workers, must produce the same jobs in the same ORDER carrying the
        // same objects with the same hashes as the sequential arm above (which
        // is itself pinned against 20 individual campaigns and against
        // GoldenReference.inc).  Several splits, including one where a group
        // has fewer jobs than the concurrency and one where it has more, so
        // both the "some threads idle" and the "work-steal a second job" paths
        // are exercised; `threads > 1` additionally puts each job's own
        // parallel_for regions on a private pool.
        {
            static const struct { uint32_t jobs, threads; } kSplits[] = {
                {2, 1}, {4, 2}, {8, 1}, {3, 4},
            };
            uint64_t concurrent_objects = 0;
            for (const auto& split : kSplits) {
                campaign::CampaignSpec par = spec;
                par.jobs = split.jobs;
                par.threads = split.threads;
                par.output_dir = root + "/par_" + std::to_string(split.jobs) + "x" +
                                 std::to_string(split.threads);
                const campaign::CampaignResult pr = campaign::RunCampaignSpec(par, /*verbose=*/false);
                ++checks;
                if (pr.jobs.size() != batch.jobs.size() || pr.num_failed != 0) {
                    std::cout << "  FAIL: the " << split.jobs << "x" << split.threads
                              << " concurrent campaign did not complete (" << pr.jobs.size()
                              << " jobs, " << pr.num_failed << " failed)" << std::endl;
                    ++failures;
                    CleanUp(par.output_dir);
                    continue;
                }
                for (size_t i = 0; i < pr.jobs.size(); ++i) {
                    // Output ORDER: slot i must be the same point as sequentially.
                    ++checks;
                    if (pr.jobs[i].job.id() != batch.jobs[i].job.id()) {
                        std::cout << "  FAIL: " << split.jobs << "x" << split.threads
                                  << " concurrent run reordered the jobs at #" << i << ": '"
                                  << pr.jobs[i].job.id() << "' vs '" << batch.jobs[i].job.id()
                                  << "'" << std::endl;
                        ++failures;
                        continue;
                    }
                    std::string why;
                    ++checks;
                    if (!SameObjects(batch.jobs[i].objects, pr.jobs[i].objects, why)) {
                        std::cout << "  FAIL: " << split.jobs << "x" << split.threads
                                  << " concurrent vs sequential differ for "
                                  << batch.jobs[i].job.to_string() << ": " << why << std::endl;
                        ++failures;
                    }
                    ++checks;
                    if (pr.jobs[i].dim_H != batch.jobs[i].dim_H ||
                        pr.jobs[i].charpoly != batch.jobs[i].charpoly) {
                        std::cout << "  FAIL: " << split.jobs << "x" << split.threads
                                  << " concurrent dim_H / charpoly differ for "
                                  << batch.jobs[i].job.to_string() << std::endl;
                        ++failures;
                    }
                    // The in-process certificates C3-C6 are not sampled and do
                    // not care about the schedule either.
                    ++checks;
                    if (!pr.jobs[i].tau_certified || !pr.jobs[i].routes_bit_identical ||
                        pr.jobs[i].theta_eq_residual_max != 0 ||
                        pr.jobs[i].decomp_certify_failures != 0) {
                        std::cout << "  FAIL: " << split.jobs << "x" << split.threads
                                  << " concurrent run lost a certificate on "
                                  << batch.jobs[i].job.to_string() << std::endl;
                        ++failures;
                    }
                    concurrent_objects += pr.jobs[i].objects.size();
                }
                // summary.tsv must be byte-identical to the sequential one:
                // everything in it but the timing columns is a computed value.
                ++checks;
                if (StripVolatileTsvColumns(ReadFile(par.output_dir + "/summary.tsv")) !=
                    StripVolatileTsvColumns(ReadFile(spec.output_dir + "/summary.tsv"))) {
                    std::cout << "  FAIL: " << split.jobs << "x" << split.threads
                              << " concurrent summary.tsv differs from the sequential one "
                                 "outside its timing columns" << std::endl;
                    ++failures;
                }
                CleanUp(par.output_dir);
            }
            std::cout << "  concurrent == sequential on " << concurrent_objects
                      << " objects over 4 jobs x threads splits" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "  FAIL: exception: " << e.what() << std::endl;
        ++failures;
    }
    CleanUp(root);

    std::cout << (failures == 0 ? "  CampaignSmokeValidation PASSED"
                                : "  CampaignSmokeValidation FAILED")
              << " (" << checks << " checks, " << failures << " failing)" << std::endl;
    return failures;
}

// ===========================================================================
// CampaignCertificateValidation
// ===========================================================================
uint32_t CampaignCertificateValidation() {
    std::cout << "--- CampaignCertificateValidation (matrix-engine Phase 6) ---" << std::endl;
    uint32_t failures = 0;
    uint64_t checks = 0;

    // 6 jobs; sampled(every 3) selects the external cross-checks on the jobs
    // with campaign index 0 and 3 only.
    static const char* kSpec =
        "name    = campaign_certify\n"
        "certify = sampled(every 3)\n"
        "n_sym   = 0,2,4\n"
        "ell     = 5\n"
        "p       = 2,3\n"
        "r       = 1\n";

    const std::string root = TempDir("certify");
    try {
        campaign::CampaignSpec spec = SpecFromString(kSpec, root + "/campaign");
        ++checks;
        if (spec.certify.mode != campaign::CertifyMode::Sampled || spec.certify.every_k != 3) {
            std::cout << "  FAIL: 'sampled(every 3)' parsed as mode="
                      << spec.certify.mode_name() << " every_k=" << spec.certify.every_k
                      << std::endl;
            ++failures;
        }

        const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
        ++checks;
        if (res.jobs.size() != 6 || res.num_failed != 0) {
            std::cout << "  FAIL: expected 6 successful jobs, got " << res.jobs.size()
                      << " with " << res.num_failed << " failures" << std::endl;
            ++failures;
        }

        // --- the in-process certificates C3-C6 ran on EVERY job -------------
        uint64_t tau_certified = 0, routes_ok = 0, residual_zero = 0, with_solves = 0,
                 decomp_ok = 0, decomp_bad = 0;
        for (const auto& r : res.jobs) {
            if (r.tau_certified) ++tau_certified;                    // C3
            if (r.routes_bit_identical) ++routes_ok;                 // C4
            if (r.theta_eq_residual_max == 0) ++residual_zero;       // C5
            if (r.theta_eq_solves > 0) ++with_solves;
            decomp_ok += r.decomp_certified;                         // C6
            decomp_bad += r.decomp_certify_failures;
        }
        ++checks;
        if (tau_certified != res.jobs.size() || routes_ok != res.jobs.size() ||
            residual_zero != res.jobs.size() || with_solves != res.jobs.size()) {
            std::cout << "  FAIL: in-process certificates did NOT run on every job "
                      << "(tau " << tau_certified << ", routes " << routes_ok
                      << ", residual0 " << residual_zero << ", solves " << with_solves
                      << " of " << res.jobs.size() << ")" << std::endl;
            ++failures;
        }
        ++checks;
        if (decomp_bad != 0 || decomp_ok == 0) {
            std::cout << "  FAIL: decomposition certificates: " << decomp_ok << " ok, "
                      << decomp_bad << " failing" << std::endl;
            ++failures;
        }
        std::cout << "  C3-C6 ran on all " << res.jobs.size() << " jobs under `sampled` ("
                  << decomp_ok << " decomposition certificates)" << std::endl;

        // --- the external cross-checks were sampled -------------------------
        uint64_t external_ran = 0;
        for (const auto& r : res.jobs) {
            const bool want = (r.job.index % 3) == 0;
            ++checks;
            if (r.external_selected != want || r.external_ran != want) {
                std::cout << "  FAIL: job #" << r.job.index << " external selected="
                          << r.external_selected << " ran=" << r.external_ran
                          << ", expected " << want << std::endl;
                ++failures;
            }
            ++checks;
            if (r.external_rc != 0) {
                std::cout << "  FAIL: job #" << r.job.index << " external rc=" << r.external_rc
                          << std::endl;
                ++failures;
            }
            if (r.external_ran) ++external_ran;

            // ... and the JSON says so.
            const std::string js = ReadFile(r.output_path);
            ++checks;
            if (js.find("\"mode\":\"sampled\"") == std::string::npos ||
                js.find("\"every_k\":3") == std::string::npos ||
                js.find("\"C3-C6 are never sampled\"") == std::string::npos) {
                std::cout << "  FAIL: " << r.output_path << " does not record the certify mode"
                          << std::endl;
                ++failures;
            }
            ++checks;
            const std::string want_sel = want ? "\"selected\":true" : "\"selected\":false";
            if (js.find(want_sel) == std::string::npos) {
                std::cout << "  FAIL: " << r.output_path << " does not record "
                          << want_sel << std::endl;
                ++failures;
            }
            ++checks;
            if (js.find("\"routes_bit_identical\":true") == std::string::npos) {
                std::cout << "  FAIL: " << r.output_path
                          << " does not record the O12(a) certificate" << std::endl;
                ++failures;
            }
        }
        ++checks;
        if (external_ran != 2) {
            std::cout << "  FAIL: expected 2 externally certified jobs, got " << external_ran
                      << std::endl;
            ++failures;
        }

        // --- resume ---------------------------------------------------------
        campaign::CampaignSpec again = spec;
        again.resume = true;
        const campaign::CampaignResult r2 = campaign::RunCampaignSpec(again, /*verbose=*/false);
        ++checks;
        if (r2.num_skipped != res.jobs.size() || r2.num_ran != 0) {
            std::cout << "  FAIL: resume ran " << r2.num_ran << " and skipped " << r2.num_skipped
                      << ", expected 0 / " << res.jobs.size() << std::endl;
            ++failures;
        }

        // Break one job's trailer: it (and only it) must be re-run.
        {
            const std::string victim = res.jobs[2].output_path;
            const std::string body = ReadFile(victim);
            std::ofstream out(victim, std::ios::out | std::ios::trunc);
            out << body.substr(0, body.size() > 64 ? body.size() - 64 : 0);
            out.close();
            ++checks;
            if (campaign::JobOutputComplete(victim)) {
                std::cout << "  FAIL: a truncated job file still reports complete" << std::endl;
                ++failures;
            }
            const campaign::CampaignResult r3 = campaign::RunCampaignSpec(again, /*verbose=*/false);
            ++checks;
            if (r3.num_ran != 1 || r3.num_skipped != res.jobs.size() - 1) {
                std::cout << "  FAIL: after truncation resume ran " << r3.num_ran
                          << " and skipped " << r3.num_skipped << ", expected 1 / "
                          << (res.jobs.size() - 1) << std::endl;
                ++failures;
            }
        }
        std::cout << "  resume: 6 skipped, then exactly the truncated job re-run" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  FAIL: exception: " << e.what() << std::endl;
        ++failures;
    }
    CleanUp(root);

    std::cout << (failures == 0 ? "  CampaignCertificateValidation PASSED"
                                : "  CampaignCertificateValidation FAILED")
              << " (" << checks << " checks, " << failures << " failing)" << std::endl;
    return failures;
}
