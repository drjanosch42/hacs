// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// CampaignHeckeZ -- slot-25 HZ-REC: the `structure = hecke` campaign (see the header).

#include "CampaignHeckeZ.h"

#include "../Hecke/HeckeSettings.h"
#include "../HeckeExt/HeckeZRecordSource.h"
#include "../Integral/HeckeZRecord.h"
#include "../Integral/HeckeZUp.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace campaign {

namespace {

namespace hzr = heckez::record;

[[noreturn]] void SpecFail(const CampaignSpec& spec, uint32_t line, const std::string& msg) {
    throw std::runtime_error(spec.origin + ":" + std::to_string(line) + ": " + msg);
}

std::string JsonEsc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
            o += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            o += ' ';
        } else {
            o += c;
        }
    }
    return o;
}

bool FileComplete(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string trailer = "\"complete\":true}";
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s.size() >= trailer.size() && s.compare(s.size() - trailer.size(), trailer.size(), trailer) == 0;
}

std::string Ed1RowsPath() {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/docs/integral_census/results/ed1/rows.tsv";
#else
    return "docs/integral_census/results/ed1/rows.tsv";
#endif
}

std::string FingerprintOf(const CampaignSpec& spec) {
    const hecke::ScopedMathSettings scope(spec.math, hecke::SettingSource::Spec);
    hzr::RecordSettings rs;
    rs.g8 = spec.hz_g8;   // (slot-25 H1) hz_g8 = hida: a non-default settings value, its own fingerprint
    const std::string text = hecke::math_settings().canonical_text() + rs.caps_text();
    std::ostringstream o;
    o << "0x" << std::hex << std::setw(16) << std::setfill('0') << engine::FnvBytes(engine::kFnvOffset64, text);
    return o.str();
}

}  // namespace

void ValidateHeckeZSpec(const CampaignSpec& spec) {
    if (spec.orbifold.enabled) SpecFail(spec, spec.orbifold.line, "orbifold: not a key of a structure = hecke campaign");
    if (spec.jobs > 1) SpecFail(spec, 1, "jobs: must be 1 under coefficient_ring = Z (one cell at a time)");
    if (spec.descent_parts_line != 0)
        SpecFail(spec, spec.descent_parts_line, "parts: the record's descent_ref is the groups,hecke descent (not a key of structure = hecke)");
    for (const SpecSection& sec : spec.sections) {
        const uint32_t ln = sec.first_line;
        if (sec.n_sym.empty()) SpecFail(spec, ln, "section is missing 'n_sym'");
        if (!sec.ell.empty()) SpecFail(spec, ln, "ell: a hecke-structure cell derives its primes (S_cand)");
        if (!sec.r.empty()) SpecFail(spec, ln, "r: not a key of structure = hecke");
        if (!sec.q.empty()) SpecFail(spec, ln, "q: not a key of structure = hecke");
        if (!sec.p.empty()) SpecFail(spec, ln, "p: the operator set is the record's P_rule (a cap of the settings text)");
        for (uint32_t n : sec.n_sym)
            if (n % 2) SpecFail(spec, ln, "n_sym: " + std::to_string(n) + " is odd (the record is defined for even n)");
        for (uint32_t N : sec.level)
            if (N == 0) SpecFail(spec, ln, "level: N must be >= 1");
        for (uint32_t N : sec.level.empty() ? std::vector<uint32_t>{1} : sec.level)
            if (N == 1 && std::find(sec.algebra.begin(), sec.algebra.end(), "anemic") != sec.algebra.end())
                SpecFail(spec, sec.algebra_line, "algebra: `anemic` at level 1 (at N = 1 only `full`, the anemic algebra)");
    }
}

std::vector<HeckeZCellPlan> PlanHeckeZCells(const CampaignSpec& spec) {
    const std::string fp = FingerprintOf(spec);
    std::vector<HeckeZCellPlan> out;
    std::set<std::tuple<uint32_t, uint32_t, std::string>> seen;
    for (const SpecSection& sec : spec.sections) {
        const std::vector<uint32_t> levels = sec.level.empty() ? std::vector<uint32_t>{1} : sec.level;
        for (uint32_t N : levels)
            for (uint32_t n : sec.n_sym) {
                std::vector<std::string> algs = sec.algebra;
                if (algs.empty()) algs = {"full"};   // (schema audit F15) one default with the console verb: `full` at every N
                for (const std::string& a : algs) {
                    if (!seen.insert({N, n, a}).second) continue;
                    out.push_back(HeckeZCellPlan{N, n, a, hzr::RecordId(N, n, a, fp)});
                }
            }
    }
    std::sort(out.begin(), out.end(), [](const HeckeZCellPlan& a, const HeckeZCellPlan& b) {
        return std::tie(a.N, a.n, a.algebra) < std::tie(b.N, b.n, b.algebra);
    });
    return out;
}

int RunHeckeZCampaignCli(const CampaignSpec& spec, bool dry_run, bool verbose) {
    namespace fs = std::filesystem;
    const std::vector<HeckeZCellPlan> plan = PlanHeckeZCells(spec);
    if (dry_run) {
        std::cout << "hecke-structure campaign '" << spec.name << "': " << plan.size() << " cells (" << hzr::kSchema
                  << ", coefficient_ring = Z, structure = hecke, fingerprint " << FingerprintOf(spec) << ")\n";
        for (const HeckeZCellPlan& c : plan) std::cout << "    " << c.id << "  N=" << c.N << " n=" << c.n << " algebra=" << c.algebra << "\n";
        std::cout << "  " << plan.size() << " cells total\n";
        return 0;
    }
    std::error_code ec;
    fs::create_directories(spec.output_dir + "/cells", ec);
    if (ec) throw std::runtime_error("campaign: cannot create '" + spec.output_dir + "/cells': " + ec.message());
    if (spec.hz_cert) {
        fs::create_directories(spec.output_dir + "/hzcert", ec);
        if (ec) throw std::runtime_error("campaign: cannot create '" + spec.output_dir + "/hzcert': " + ec.message());
    }
    const hecke::ScopedMathSettings scope(spec.math, hecke::SettingSource::Spec);
    const hzr::RecordSources src = hzr::DefaultRecordSources();
    hzr::RecordSettings rs;
    rs.threads = spec.threads ? spec.threads : 1;
    rs.g8 = spec.hz_g8;
    // (S26 Z1, plan Z1 card / audit S2) the twelve descent levers as this process reads them, once per process (stdout only)
    heckez::up::PrintDescentEffectiveOnce();
    uint32_t ran = 0, resumed = 0, refused = 0, pending = 0, fin = 0, failed = 0, outside = 0, under = 0, ed1_mismatch = 0, rejected = 0;
    std::ostringstream sum, idx;
    sum << "id\tN\tn\talgebra\tstatus\trecord_fnv1a64\tcert_sha256\tdescent_record_fnv1a64\tdescent_cell_sha256\tseconds\trss_mib\n";
    idx << "{\"schema\":\"integral_hecke_structure_campaign/v1\",\"name\":\"" << JsonEsc(spec.name) << "\",\"settings_fingerprint\":\""
        << FingerprintOf(spec) << "\",\"cells\":[";
    bool first = true;
    auto count = [&](const std::string& st) {
        if (st == "refused") ++refused;
        else if (st == "pending") ++pending;
        else ++fin;
    };
    for (const HeckeZCellPlan& c : plan) {
        const std::string path = spec.output_dir + "/cells/" + c.id + ".json";
        std::string status = "failed", fnv = "-", cert = "-", dfnv = "-", dsha = "-";
        double secs = 0;
        uint64_t rss = 0;
        bool have = false;
        // (schema audit F4) a resumed cell file is re-read: id, fingerprint, both hashes and the status must recompute and the scan
        // must be clean; a file that fails is rejected (counted) and the cell is recomputed
        if (spec.resume && FileComplete(path)) {
            std::ifstream in(path, std::ios::binary);
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            hzr::HeckeStructureRecord R;
            std::string why;
            bool ok = hzr::VerifyCellText(text, c.id, &R, &why);
            if (ok) {
                // (re-audit R5 (a)) the ed.1 join is redone, not read: the row's record hash must be the hashed descent_ref, and the
                // file's cell_sha256 the row's (or the compute-time `ed1_mismatch`, which a resume cannot re-derive: kept and counted)
                const hzr::Ed1Row row = hzr::LookupEd1(c.N, c.n, Ed1RowsPath());
                const std::string want = !row.found ? std::string("not_in_ed1")
                                                    : (row.record_fnv1a64 == R.descent.record_fnv1a64 ? row.cell_sha256 : std::string("ed1_mismatch"));
                if (R.descent.cell_sha256 != want && R.descent.cell_sha256 != "ed1_mismatch") {
                    ok = false;
                    why = "descent_ref.cell_sha256 '" + R.descent.cell_sha256 + "' is not the ed.1 join '" + want + "'";
                }
            }
            if (ok) {
                ++resumed;
                have = true;
                status = R.status;
                fnv = R.record_fnv1a64;
                cert = R.cert_sha256;
                dfnv = R.descent.record_fnv1a64;
                dsha = R.descent.cell_sha256;
                if (dsha == "ed1_mismatch") ++ed1_mismatch;
                secs = R.secs;
                rss = R.rss_peak;
                count(status);
            } else {
                ++rejected;
                std::cerr << "campaign: hecke-structure cell " << c.id << ": the resumed file is REJECTED (" << why << "); recomputing" << std::endl;
            }
        }
        if (!have) {
            try {
                hzr::HeckeStructureRecord R = hzr::BuildHeckeStructureRecord(c.N, c.n, c.algebra, rs, src);
                ed1_mismatch += hzr::JoinEd1(R, hzr::LookupEd1(c.N, c.n, Ed1RowsPath())) ? 1 : 0;
                ENGINE_CHECK_MSG(R.id == c.id, "campaign: the record id differs from the planned id");
                // (F4) every written record is scanned; a value outside the closed lists or an exact value under a bound fails the cell
                const hzr::ScanResult sc = hzr::ScanRecord(R);
                outside += sc.values_outside_lists;
                under += sc.exact_under_bound;
                if (sc.values_outside_lists || sc.exact_under_bound || !sc.overall_rule_ok) {
                    std::string p1 = sc.problems.empty() ? std::string("overall status rule") : sc.problems.front();
                    throw std::runtime_error("the record fails its scan (outside=" + std::to_string(sc.values_outside_lists) + " under=" +
                                             std::to_string(sc.exact_under_bound) + "): " + p1);
                }
                status = R.status;
                fnv = R.record_fnv1a64;
                cert = R.cert_sha256;
                dfnv = R.descent.record_fnv1a64;
                dsha = R.descent.cell_sha256;
                secs = R.secs;
                rss = R.rss_peak;
                count(R.status);
                std::ofstream out(path, std::ios::binary | std::ios::trunc);
                if (!out) throw std::runtime_error("cannot write '" + path + "'");
                out << hzr::CellFileText(R, spec.name);
                if (!out) throw std::runtime_error("short write on '" + path + "'");
                if (spec.hz_cert) hzr::WriteCertFiles(R, src, rs, spec.output_dir + "/hzcert");
                ++ran;
            } catch (const std::exception& e) {
                ++failed;
                status = "failed";
                std::cerr << "campaign: hecke-structure cell " << c.id << " FAILED: " << e.what() << std::endl;
            }
        }
        std::ostringstream row;
        row << c.id << "\t" << c.N << "\t" << c.n << "\t" << c.algebra << "\t" << status << "\t" << fnv << "\t" << cert << "\t" << dfnv << "\t"
            << dsha << "\t" << std::fixed << std::setprecision(3) << secs << "\t" << rss;
        sum << row.str() << "\n";
        idx << (first ? "" : ",") << "{\"id\":\"" << c.id << "\",\"status\":\"" << status << "\",\"record_fnv1a64\":\"" << fnv << "\"}";
        first = false;
        if (verbose) std::cout << "  [hecke_structure] " << row.str() << std::endl;
    }
    idx << "]}\n";
    std::ofstream(spec.output_dir + "/summary.tsv") << sum.str();
    std::ofstream(spec.output_dir + "/index.json") << idx.str();
    std::cout << "EFFECTIVE campaign arm=hecke_structure cells=" << plan.size() << " ran=" << ran << " resumed=" << resumed
              << " rejected=" << rejected << " refused=" << refused << " pending=" << pending << " final=" << fin << " failed=" << failed
              << " outside=" << outside << " under=" << under << " ed1_mismatch=" << ed1_mismatch << std::endl;
    return failed == 0 && ed1_mismatch == 0 ? 0 : 2;
}

}  // namespace campaign
