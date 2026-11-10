// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "CampaignRingValidation.h"

// hecke-extensions slot 9, WP-8 -- see the header.

#include "CampaignDriver.h"
#include "CampaignRing.h"
#include "CampaignSpec.h"
#include "../Golden/GoldenHarness.h"

#include "../Hecke/HeckeSettings.h"
#include "../Util/TextIo.h"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The frozen reference, walked in full for the set equality of arm (A): the single compiled copy,
// kGoldenReference of Golden/GoldenHarness.h.

uint32_t g_checks = 0, g_failures = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::cout << "  [FAIL] " << what << "\n";
    }
}

std::string TempDir(const std::string& tag) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    const fs::path dir = (ec ? fs::path(".") : base) / ("oc_campaign_ring_" + tag + "_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    return dir.string();
}

campaign::CampaignSpec SpecFromString(const std::string& text, const std::string& out_dir) {
    std::istringstream is(text);
    campaign::CampaignSpec spec = campaign::ParseCampaignSpec(is, "<wp8-spec>");
    spec.output_dir = out_dir;
    return spec;
}

using oc_util::ReadFile;   // S24 CL-7 (F11): Util/TextIo.h
std::vector<std::string> Split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

std::string E8Point(uint32_t n, uint64_t ell, uint32_t k) {
    return "E8 ring=Z/" + std::to_string(ell) + "^" + std::to_string(k) + " n=" + std::to_string(n) +
           " ell=" + std::to_string(ell);
}

// An E8 item-4a object (the record carries (d)'s objects, not (f)'s 4b ones).
bool Is4aObject(const std::string& name) {
    return name.compare(0, 5, "bdry_") != 0 && name.compare(0, 5, "cone_") != 0 &&
           name.compare(0, 9, "coneexact") != 0;
}

bool Throws(const std::string& text, const std::string& needle, std::string* what = nullptr) {
    try {
        std::istringstream is(text);
        (void)campaign::ParseCampaignSpec(is, "<wp8-spec>");
    } catch (const hecke::NotImplementedInThisBuild& e) {
        if (what) *what = e.what();
        return std::string(e.what()).find(needle) != std::string::npos &&
               std::string(e.what()).find("not_implemented_in_this_build:") == 0;
    } catch (...) {
        return false;
    }
    return false;
}

const campaign::JobResult* FindJob(const campaign::CampaignResult& r, uint32_t n, uint64_t ell, uint64_t p,
                                   uint32_t rr = 1) {
    for (const campaign::JobResult& j : r.jobs)
        if (j.job.n_sym == n && j.job.ell == ell && j.job.p == p && j.job.r == rr) return &j;
    return nullptr;
}

// (e)'s table, the pre-stated values.
struct E8Pred {
    uint32_t n, ell, k;
    bool certified;
    std::vector<uint32_t> H0, H1;
    uint32_t beta, htc;
};
const std::vector<E8Pred>& E8Table() {
    static const std::vector<E8Pred> t = {
        {10, 691, 2, true, {}, {2, 2, 2}, 0, 0},
        {16, 5, 2, true, {}, {2, 2, 2}, 0, 0},
        {6, 5, 2, true, {1}, {2, 1}, 1, 0},
        {30, 5, 2, false, {2}, {2, 2, 2, 2, 2, 2}, 0, 1},
        {30, 5, 3, true, {2}, {3, 3, 3, 3, 3, 2}, 0, 1},
        {60, 5, 2, false, {2, 1}, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1}, 1, 1},
        {60, 5, 3, true, {2, 1}, {3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1}, 1, 1},
    };
    return t;
}

std::string ListText(const std::vector<uint32_t>& v) {
    if (v.empty()) return "()";
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ";" : "") + std::to_string(v[i]);
    return s;
}

// The summary row of job file `file`, as columns.
std::vector<std::string> SummaryRow(const std::string& tsv, const std::string& file) {
    for (const std::string& line : Split(tsv, '\n')) {
        const std::vector<std::string> c = Split(line, '\t');
        if (c.size() > 20 && c[20] == file) return c;
    }
    return {};
}

// Arm (A) for one ring campaign.
void ArmE8(const campaign::CampaignSpec& spec, const campaign::CampaignResult& res, uint32_t k) {
    for (const E8Pred& e : E8Table()) {
        if (e.k != k) continue;
        const std::vector<uint64_t> primes = (e.ell == 691) ? std::vector<uint64_t>{2, 3, 5} : std::vector<uint64_t>{2, 3};
        const std::string pt = E8Point(e.n, e.ell, k);
        for (uint64_t p : primes) {
            const campaign::JobResult* j = FindJob(res, e.n, e.ell, p);
            const std::string tag = "(A) " + pt + " p=" + std::to_string(p);
            check(j != nullptr && j->ran && j->ok && j->ring_status == "ok", tag + ": the job ran, status ok");
            if (j == nullptr || !j->ok) continue;
            // Every carried object equals the frozen golden entry ...
            std::set<std::string> carried;
            for (const campaign::CanonicalObject& o : j->objects) {
                carried.insert(o.object);
                bool found = false, same = false;
                for (const GoldenEntry* g = kGoldenReference; g->point != nullptr; ++g) {
                    if (pt != g->point || o.object != g->object) continue;
                    found = true;
                    same = g->tier == o.tier && g->rows == o.rows && g->cols == o.cols && g->hash == o.hash;
                }
                check(found && same, tag + ": object " + o.object + " == golden");
            }
            // ... and every frozen 4a entry of the point is carried (charpoly for this p only).
            uint32_t expected = 0;
            for (const GoldenEntry* g = kGoldenReference; g->point != nullptr; ++g) {
                if (pt != g->point || !Is4aObject(g->object)) continue;
                const std::string name = g->object;
                if (name.compare(0, 8, "charpoly") == 0 && name != "charpoly_A(T_p)^1_free_p" + std::to_string(p)) continue;
                ++expected;
                check(carried.count(name) == 1, tag + ": golden 4a object " + name + " is carried");
            }
            check(expected > 0 && expected == j->objects.size(),
                  tag + ": the record's object set == the golden 4a set (" + std::to_string(j->objects.size()) +
                      " vs " + std::to_string(expected) + ")");
            // The (e) table, literally, through summary.tsv's columns.
            const std::string tsv = ReadFile(spec.output_dir + "/summary.tsv");
            const std::vector<std::string> row = SummaryRow(tsv, "jobs/" + j->job.id() + ".json");
            check(row.size() == 36, tag + ": summary row has 21 + 15 columns");
            if (row.size() == 36) {
                check(row[23] == (e.certified ? "certified" : "uncertified"), tag + ": kmode (e)");
                check(row[27] == ListText(e.H0) && row[28] == ListText(e.H1) && row[29] == "()",
                      tag + ": invariant factors H^0 / H^1 / H^2 == (e)'s table");
                check(row[32] == std::to_string(e.beta) && row[33] == std::to_string(e.htc),
                      tag + ": Bockstein rank / higher torsion count == (e)'s table");
                check((row[34] != "-") == e.certified, tag + ": charpoly_free present iff certified");
                check(row[35] == "-", tag + ": no refusal");
            }
        }
    }
}

} // namespace

uint32_t CampaignRingValidation() {
    g_checks = g_failures = 0;
    std::cout << "=== CampaignRingValidation (slot-9 WP-8, C5c: hecke_ring_report/v1) ===\n";
    namespace fs = std::filesystem;
    const std::string root = TempDir("wp8");

    // ---------------------------------------------------------------- (X)
    {
        std::string what;
        check(Throws("coefficient_ring = Z/ell^2\next = true\nn_sym = 10\nell = 691\np = 2\nr = 1\n", "ext pass", &what),
              "(X) ext = true under Z/ell^2 refused at spec validation, by name");
        check(what.find("campaign spec <wp8-spec>") != std::string::npos, "(X) the refusal names the spec");
        check(Throws("coefficient_ring = Z/ell^3\next = true\nn_sym = 10\nell = 5\np = 2\nr = 1\n", "ext pass"),
              "(X) ext = true under Z/ell^3 refused");
        check(Throws("coefficient_ring = Z/ell^2\nn_sym = 10\nell = 5,4294967311\np = 2\nr = 1\n",
                     "ell = 4294967311", &what) && what.find("2^64") != std::string::npos,
              "(X) a cell with ell^2 >= 2^64 refused, naming the cell");
        check(Throws("coefficient_ring = Z/ell^41\nn_sym = 4\nell = 3\np = 2\nr = 1\n", "ell = 3"),
              "(X) 3^41 >= 2^64 refused");
        bool admitted = true;
        try {
            std::istringstream is("coefficient_ring = Z/ell^63\nn_sym = 4\nell = 2\np = 2\nr = 1\n");
            (void)campaign::ParseCampaignSpec(is, "<wp8-spec>");
        } catch (...) { admitted = false; }
        check(admitted, "(X) 2^63 < 2^64 admitted (job loop)");
        // Through the CLI: exit 1, and no output directory is ever created.
        const std::string spec_path = root + "/ext_ring.spec";
        const std::string out = root + "/ext_ring_out";
        {
            std::ofstream f(spec_path);
            f << "coefficient_ring = Z/ell^2\next = true\nn_sym = 10\nell = 691\np = 2\nr = 1\n";
        }
        std::string a0 = spec_path, a1 = "--output", a2 = out, a3 = "--quiet";
        char* av[] = {&a0[0], &a1[0], &a2[0], &a3[0]};
        const int rc = campaign::RunCampaign(4, av);
        check(rc == 1 && !fs::exists(out), "(X) --campaign with ext = true under the ring: exit 1, no output dir");
        // A default spec: no ring, no suffix.
        std::istringstream is("n_sym = 10\nell = 691\np = 2\nr = 1\n");
        const campaign::CampaignSpec def = campaign::ParseCampaignSpec(is, "<wp8-spec>");
        const std::vector<campaign::Job> jobs = campaign::flatten(campaign::plan_campaign(def));
        check(jobs.size() == 1 && jobs[0].id() == "n10_ell691_p2_r1", "(X) the default id is unchanged");
    }

    // ------------------------------------------------ (A) (T) (B) (R) (S), k = 2
    const std::string spec2_text =
        "name = wp8_ring_k2\ncoefficient_ring = Z/ell^2\njobs = 2\nthreads = 2\n"
        "n_sym = 6,16,30,60\nell = 5\np = 2,3\nr = 1\n---\n"
        "n_sym = 10\nell = 691\np = 2,3,5\nr = 1\n---\n"
        "n_sym = 10\nell = 134217757\np = 2\nr = 1\n---\n"
        "n_sym = 4,7\nell = 2,3\np = 2\nr = 1\n---\n"
        "n_sym = 10\nell = 5\np = 2\nr = 2\n";
    const campaign::CampaignSpec spec2 = SpecFromString(spec2_text, root + "/k2");
    const campaign::CampaignResult res2 = campaign::RunCampaignSpec(spec2, /*verbose=*/false);
    check(res2.num_failed == 0 && res2.num_ran == res2.jobs.size(), "(S) the k = 2 campaign ran every job, none failed");
    ArmE8(spec2, res2, 2);

    // (T) weight 12: (x - 2049)(x + 24)^2 over Z/ell^2, at 691 and at the uint64_t cell.
    for (uint64_t ell : {uint64_t(691), uint64_t(134217757)}) {
        const unsigned __int128 m = static_cast<unsigned __int128>(ell) * ell;
        auto red = [&](__int128 v) { __int128 r = v % static_cast<__int128>(m); if (r < 0) r += m; return static_cast<uint64_t>(r); };
        const std::string want = std::to_string(red(-1180224)) + ";" + std::to_string(red(-97776)) + ";" +
                                 std::to_string(red(-2001)) + ";1@2";
        const campaign::JobResult* j = FindJob(res2, 10, ell, 2);
        const std::string tsv = ReadFile(spec2.output_dir + "/summary.tsv");
        const std::vector<std::string> row = j ? SummaryRow(tsv, "jobs/" + j->job.id() + ".json") : std::vector<std::string>{};
        check(row.size() == 36 && row[34] == want,
              "(T) (10, " + std::to_string(ell) + ") charpoly_free_A1 == (x-(1+2^11))(x-tau(2))^2 mod ell^2: " +
                  (row.size() == 36 ? row[34] : std::string("?")));
        const std::string rec = j ? ReadFile(j->output_path) : std::string();
        check(rec.find("\"modulus\":\"" + std::to_string(static_cast<uint64_t>(m)) + "\"") != std::string::npos,
              "(T) the modulus is ell^2 as a decimal string");
        // (g11) the coefficients are decimal STRINGS (they may exceed 2^53).
        std::string coeffs = "\"charpoly_free\":{\"precision\":2,\"coeffs\":[";
        for (const std::string& c : Split(want.substr(0, want.find('@')), ';')) {
            if (coeffs.back() != '[') coeffs += ",";
            coeffs += "\"" + c + "\"";
        }
        coeffs += "]}";
        check(rec.find(coeffs) != std::string::npos,
              "(T) (10, " + std::to_string(ell) + ") charpoly_free.coeffs as decimal strings: " + coeffs);
    }

    // (B) ell in {2, 3}: the O7 rows (HeckeBredonTreeRingValidation's frozen table).
    struct O7 { uint32_t n; uint64_t ell; std::vector<uint32_t> len; std::vector<std::vector<uint32_t>> inv; };
    const std::vector<O7> o7 = {
        {4, 2, {2, 4, 0}, {{2}, {2, 2}, {}}},
        {7, 2, {1, 3, 0}, {{1}, {1, 1, 1}, {}}},
        {4, 3, {1, 2, 0}, {{1}, {2}, {}}},
        {7, 3, {0, 0, 0}, {{}, {}, {}}},
    };
    for (const O7& c : o7) {
        const campaign::JobResult* j = FindJob(res2, c.n, c.ell, 2);
        const std::string tag = "(B) (" + std::to_string(c.n) + ", " + std::to_string(c.ell) + ") over Z/ell^2";
        check(j != nullptr && j->ok && j->ring_status == "ok_partial", tag + ": status ok_partial");
        if (j == nullptr || !j->ok) continue;
        const std::vector<std::string> want_refused = {"hecke", "bockstein", "reduction"};
        check(j->ring_refusals == want_refused, tag + ": refused exactly hecke, bockstein, reduction");
        check(j->ring_columns.size() == 15, tag + ": 15 summary columns");
        if (j->ring_columns.size() == 15) {
            const auto& c15 = j->ring_columns;
            check(c15[2] == "not_applicable", tag + ": kmode not_applicable");
            bool lens = true, invs = true;
            for (uint32_t q = 0; q < 3; ++q) {
                lens = lens && c15[3 + q] == std::to_string(c.len[q]);
                invs = invs && c15[6 + q] == ListText(c.inv[q]);
            }
            check(lens && invs, tag + ": lengths and invariant factors == O7");
            check(c15[9] == "-" && c15[10] == "-" && c15[11] == "-" && c15[12] == "-" && c15[13] == "-" &&
                      c15[14] == "hecke;bockstein;reduction",
                  tag + ": no K-mode / Bockstein / charpoly column; refusals column");
        }
        const std::string rec = ReadFile(j->output_path);
        check(rec.find("\"ladic\":") == std::string::npos && rec.find("\"bockstein\":") == std::string::npos &&
                  rec.find("\"hecke\":{") == std::string::npos,
              tag + ": no ladic / bockstein / hecke member");
        check(rec.find("{\"part\":\"hecke\",\"who\":\"HeckeOperatorOnK\",\"message\":\"not_implemented_in_this_build: ") !=
                      std::string::npos &&
                  rec.find("{\"part\":\"bockstein\",\"who\":\"HeckeReduction (BocksteinCounts)\",\"message\":"
                           "\"not_implemented_in_this_build: ") != std::string::npos &&
                  rec.find("{\"part\":\"reduction\",\"who\":\"HeckeReduction (ReduceHToField)\",\"message\":"
                           "\"not_implemented_in_this_build: ") != std::string::npos,
              tag + ": each refusal names its member and carries the tag");
        check(j->objects.size() == 6, tag + ": the objects are the lengths and invariant factors only");
    }

    // (R) r = 2 at ell = 5.
    {
        const campaign::JobResult* j = FindJob(res2, 10, 5, 2, 2);
        check(j != nullptr && j->ok && j->ring_status == "ok_partial" &&
                  j->ring_refusals == std::vector<std::string>{"hecke"},
              "(R) r = 2: status ok_partial, the Hecke part refused");
        const std::string rec = j ? ReadFile(j->output_path) : std::string();
        check(rec.find("{\"part\":\"hecke\",\"who\":\"campaign job loop\",\"message\":\"not_implemented_in_this_build: ") !=
                      std::string::npos,
              "(R) refused at the job loop's own dispatch, with the tag");
        check(rec.find("\"hecke\":{") == std::string::npos && rec.find("\"ladic\":{") != std::string::npos &&
                  rec.find("\"bockstein\":{") != std::string::npos,
              "(R) no hecke member; the K-mode and the Bockstein computed");
    }

    // (S) the record's shape, summary.tsv, index.json.
    {
        const campaign::JobResult* j = FindJob(res2, 6, 5, 2);
        const std::string rec = j ? ReadFile(j->output_path) : std::string();
        const std::vector<std::string> order = {
            "{\"schema\":\"hecke_ring_report/v1\"", "\"job_id\":\"n6_ell5_p2_r1_s", "\"point\":", "\"model_descriptor\":",
            "\"shape_signature\":", "\"coefficient_ring\":\"Z/ell^2\"", "\"modulus\":\"25\"", "\"ladic_level\":2",
            "\"settings\":", "\"certify\":{\"mode\":\"full\",\"applicable\":false}", "\"degree_fields\":{\"q_max\":2,"
            "\"truncated\":false,\"q_max_source\":\"derived\",\"degrees\":[0,1,2]}",
            "\"kmode\":\"certified\"", "\"cohomology\":[", "\"ladic\":", "\"bockstein\":", "\"hecke\":", "\"objects\":[",
            "\"refusals\":[]", "\"status\":\"ok\"", "\"timings_s\":", "\"peak_rss_mb\":", "\"peak_rss_scope\":",
            "\"campaign_job_complete\":true}"};
        size_t at = 0;
        bool ordered = !rec.empty();
        for (const std::string& m : order) {
            const size_t f = rec.find(m, at);
            if (f == std::string::npos) { ordered = false; std::cout << "  missing/out of order: " << m << "\n"; break; }
            at = f + m.size();
        }
        check(ordered, "(S) the record's members in (g1)-(g16) order");
        check(rec.find("\"dim_H\"") == std::string::npos && rec.find("\"reports\"") == std::string::npos &&
                  rec.find("\"schema\":\"campaign_job/v1\"") == std::string::npos,
              "(S) no dim_H, no reports, not campaign_job/v1");
        check(rec.find("\"fingerprint\":\"0000000000000000\"") == std::string::npos, "(S) the fingerprint is non-zero");

        const std::string tsv = ReadFile(spec2.output_dir + "/summary.tsv");
        const std::vector<std::string> lines = Split(tsv, '\n');
        const std::vector<std::string> head = Split(lines.empty() ? std::string() : lines[0], '\t');
        const std::string want_head =
            "index\tn_sym\tell\tp\tr\tmodel\tstatus\tdim_H0\tdim_H1\tdim_H2\tcharpoly_A0\tcharpoly_A1\tcharpoly_A2"
            "\ttrace_A1\trank_A1\ttheta_flat_hash_q1\tcertify\texternal\ttime_s\trss_mb\tfile" +
            campaign::RingSummaryHeader();
        check(!lines.empty() && lines[0] == want_head && head.size() == 36, "(S) summary.tsv header: 21 + 15 columns");
        bool dashes = true, widths = true;
        uint32_t rows = 0;
        for (size_t i = 1; i < lines.size(); ++i) {
            if (lines[i].empty()) continue;
            ++rows;
            const std::vector<std::string> c = Split(lines[i], '\t');
            widths = widths && c.size() == 36;
            if (c.size() != 36) continue;
            for (size_t col : {7, 8, 9, 10, 11, 12, 13, 14, 15, 17}) dashes = dashes && c[col] == "-";
        }
        check(widths && rows == res2.jobs.size(), "(S) one 36-column row per job");
        check(dashes, "(S) the ten field-valued columns hold -");
        const std::string idx = ReadFile(spec2.output_dir + "/index.json");
        check(idx.find("\"spec_origin\":\"<wp8-spec>\",\"coefficient_ring\":\"Z/ell^2\",\"settings\":") != std::string::npos,
              "(S) index.json gains coefficient_ring, one top-level key");
    }

    // (S) resume skips every ring job and still writes the ring columns' ring / level.
    {
        campaign::CampaignSpec again = spec2;
        again.resume = true;
        const campaign::CampaignResult r = campaign::RunCampaignSpec(again, false);
        check(r.num_skipped == res2.jobs.size() && r.num_ran == 0, "(S) resume skips every completed ring job");
        const std::string tsv = ReadFile(again.output_dir + "/summary.tsv");
        const std::vector<std::string> row = SummaryRow(tsv, "jobs/" + res2.jobs[0].job.id() + ".json");
        check(row.size() == 36 && row[6] == "skipped" && row[21] == "Z/ell^2" && row[22] == "2" && row[23] == "-",
              "(S) a skipped ring row: status skipped, ring and level, the rest -");
    }

    // (S) jobs = 1 == jobs = 2 (the concurrent loop moves nothing).
    {
        campaign::CampaignSpec seq = spec2;
        seq.jobs = 1;
        seq.threads = 0;
        seq.output_dir = root + "/k2_seq";
        const campaign::CampaignResult rs = campaign::RunCampaignSpec(seq, false);
        bool same = rs.jobs.size() == res2.jobs.size();
        for (size_t i = 0; same && i < rs.jobs.size(); ++i) {
            const auto& a = rs.jobs[i];
            const auto& b = res2.jobs[i];
            same = a.job.id() == b.job.id() && a.ring_status == b.ring_status && a.ring_columns == b.ring_columns &&
                   a.objects.size() == b.objects.size();
            for (size_t o = 0; same && o < a.objects.size(); ++o)
                same = a.objects[o].object == b.objects[o].object && a.objects[o].hash == b.objects[o].hash &&
                       a.objects[o].rows == b.objects[o].rows && a.objects[o].cols == b.objects[o].cols;
        }
        check(same, "(S) jobs = 1 and jobs = 2 give the same records (objects, status, summary columns)");
    }

    // ------------------------------------------------------------- (A), k = 3
    {
        const std::string spec3_text =
            "name = wp8_ring_k3\ncoefficient_ring = Z/ell^3\njobs = 2\nthreads = 2\n"
            "n_sym = 30,60\nell = 5\np = 2,3\nr = 1\n";
        const campaign::CampaignSpec spec3 = SpecFromString(spec3_text, root + "/k3");
        const campaign::CampaignResult res3 = campaign::RunCampaignSpec(spec3, false);
        check(res3.num_failed == 0 && res3.num_ran == 4, "(A) the k = 3 campaign ran its 4 jobs");
        ArmE8(spec3, res3, 3);
    }

    std::error_code ec;
    fs::remove_all(root, ec);
    std::cout << "CampaignRingValidation: " << g_checks << " checks, " << g_failures << " failing\n";
    return g_failures == 0 ? 0 : 1;
}
