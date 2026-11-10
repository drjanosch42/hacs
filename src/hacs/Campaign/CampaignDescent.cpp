// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (slot-24 D6) The campaign under coefficient_ring = Z -- see CampaignDescent.h.

#include "CampaignDescent.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../Hecke/HeckeSettings.h"
#include "../Integral/Descent/DescentRecord.h"

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

namespace rec = integral_descent::record;

std::string hex16(uint64_t v) {
    std::ostringstream o;
    o << std::hex << std::setw(16) << std::setfill('0') << v;
    return o.str();
}

bool IsPrime32(uint64_t p) {
    if (p < 2 || p >= (1ull << 16)) return false;
    for (uint64_t d = 2; d * d <= p; ++d)
        if (p % d == 0) return false;
    return true;
}

[[noreturn]] void SpecFail(const CampaignSpec& spec, uint32_t line, const std::string& msg) {
    throw std::runtime_error(spec.origin + ":" + std::to_string(line) + ": " + msg);
}

// (D6 audit N2) A p dividing a section's N is refused at spec validation (ValidateDescentSpec), never dropped or replaced by the
// defaults here: the cell's P is the section's p list, or the record's default when the section has none.
std::vector<uint32_t> CellPrimes(const std::vector<uint64_t>& section_p, uint32_t N) {
    std::vector<uint32_t> ps;
    for (uint64_t p : section_p) {
        ENGINE_CHECK_MSG(N % p != 0, "campaign: a Hecke prime dividing N reached the descent plan (ValidateDescentSpec refuses it)");
        if (std::find(ps.begin(), ps.end(), uint32_t(p)) == ps.end()) ps.push_back(uint32_t(p));
    }
    std::sort(ps.begin(), ps.end());
    if (ps.empty()) ps = rec::DefaultRecordPrimes(N);
    return ps;
}

bool FileComplete(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string trailer = "\"complete\":true}";
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s.size() >= trailer.size() && s.compare(s.size() - trailer.size(), trailer.size(), trailer) == 0;
}

std::string JsonEsc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

}  // namespace

void ValidateDescentSpec(const CampaignSpec& spec) {
    if (spec.orbifold.enabled)
        SpecFail(spec, spec.orbifold.line, "orbifold: an orbifold campaign is F_ell / Bredon-free group cohomology at ell in {2, 3}; "
                                           "under coefficient_ring = Z the campaign runs descent cells");
    if (spec.jobs > 1) SpecFail(spec, 1, "jobs: must be 1 under coefficient_ring = Z (one descent cell at a time)");
    for (const SpecSection& sec : spec.sections) {
        const uint32_t ln = sec.first_line;
        if (sec.n_sym.empty()) SpecFail(spec, ln, "section is missing 'n_sym'");
        if (!sec.ell.empty())
            SpecFail(spec, ln, "ell: a descent cell has no coefficient prime (coefficient_ring = Z); its torsion primes are derived");
        if (!sec.r.empty()) SpecFail(spec, ln, "r: a descent cell records T_p (r = 1) only");
        if (!sec.q.empty()) SpecFail(spec, ln, "q: a descent cell's degrees are derived (q_max = 3, [C] thm:descentcert(6))");
        for (uint32_t N : sec.level)
            if (N == 0) SpecFail(spec, ln, "level: N must be >= 1");
        for (uint64_t p : sec.p)
            if (!IsPrime32(p)) SpecFail(spec, ln, "p: " + std::to_string(p) + " is not a prime < 2^16");
        // (D6 audit N2) as the record and the console verb: a Hecke prime must not divide the cell's level (U_p is deferred, [R-Q4])
        for (uint32_t N : sec.level.empty() ? std::vector<uint32_t>{1} : sec.level)
            for (uint64_t p : sec.p)
                if (N % p == 0)
                    SpecFail(spec, ln, "p: " + std::to_string(p) + " divides level " + std::to_string(N) +
                                           " (U_p is deferred; put that level in a section of its own with other primes)");
    }
}

uint64_t DescentFingerprint(const CampaignSpec& spec) { return spec.math.fingerprint(); }

std::string DescentCellId(const CampaignSpec& spec, uint32_t N, uint32_t n, const std::vector<uint32_t>& ps) {
    std::string id = "desc_N" + std::to_string(N) + "_n" + std::to_string(n);
    if (ps != rec::DefaultRecordPrimes(N)) {
        id += "_p";
        for (size_t i = 0; i < ps.size(); ++i) id += (i ? "-" : "") + std::to_string(ps[i]);
    }
    const uint64_t fp = DescentFingerprint(spec);
    if (fp != 0) id += "_s" + hex16(fp);
    // (ICP2) a non-default `parts` is part of the id (a groups-only cell never collides with a full one); the default adds nothing
    if (spec.descent_parts == DescentParts::GroupsHecke) id += "_partsGH";
    else if (spec.descent_parts == DescentParts::Groups) id += "_partsG";
    return id;
}

std::vector<DescentCellPlan> PlanDescentCells(const CampaignSpec& spec) {
    std::vector<DescentCellPlan> out;
    std::set<std::tuple<uint32_t, uint32_t, std::vector<uint32_t>>> seen;
    for (const SpecSection& sec : spec.sections) {
        const std::vector<uint32_t> levels = sec.level.empty() ? std::vector<uint32_t>{1} : sec.level;
        for (uint32_t N : levels)
            for (uint32_t n : sec.n_sym) {
                DescentCellPlan c;
                c.N = N;
                c.n = n;
                c.ps = CellPrimes(sec.p, N);
                if (!seen.insert({c.N, c.n, c.ps}).second) continue;
                c.id = DescentCellId(spec, c.N, c.n, c.ps);
                out.push_back(std::move(c));
            }
    }
    std::sort(out.begin(), out.end(), [](const DescentCellPlan& a, const DescentCellPlan& b) {
        return std::tie(a.N, a.n, a.ps) < std::tie(b.N, b.n, b.ps);
    });
    return out;
}

DescentCampaignResult RunDescentCampaign(const CampaignSpec& spec, bool verbose) {
    namespace fs = std::filesystem;
    DescentCampaignResult res;
    std::error_code ec;
    fs::create_directories(spec.output_dir + "/cells", ec);
    if (ec) throw std::runtime_error("campaign: cannot create '" + spec.output_dir + "/cells': " + ec.message());
    const std::vector<DescentCellPlan> plan = PlanDescentCells(spec);
    res.cells = static_cast<uint32_t>(plan.size());
    // the settings of the spec are the records' (coefficient_ring = Z), installed for the whole loop
    const hecke::ScopedMathSettings scope(spec.math, hecke::SettingSource::Spec);
    std::ostringstream sum, idx;
    sum << "id\tN\tn\tprimes\tstatus\tH0\tH1\tH2\tH3\trecord_fnv1a64\tseconds\n";
    idx << "{\"schema\":\"integral_descent_campaign/v1\",\"name\":\"" << JsonEsc(spec.name) << "\",\"settings_fingerprint\":\"0x"
        << hex16(DescentFingerprint(spec)) << "\"";
    if (spec.descent_parts != DescentParts::All) idx << ",\"parts\":\"" << DescentPartsText(spec.descent_parts) << "\"";   // (ICP2) default: absent
    idx << ",\"cells\":[";
    bool first = true;
    for (const DescentCellPlan& c : plan) {
        const std::string path = spec.output_dir + "/cells/" + c.id + ".json";
        std::string status, fnv = "-", groups[4] = {"-", "-", "-", "-"};
        double secs = 0;
        if (spec.resume && FileComplete(path)) {
            ++res.resumed;
            status = "resumed";
        } else {
            const auto t0 = std::chrono::steady_clock::now();
            try {
                rec::RecordOptions o;
                o.ps = c.ps;
                o.hecke = spec.descent_parts != DescentParts::Groups;
                o.pair = spec.descent_parts == DescentParts::All;
                const rec::DescentRecordZ R = rec::BuildDescentRecord(c.N, c.n, o);
                status = R.status;
                fnv = "0x" + hex16(R.digest());
                for (size_t q = 0; q < R.H.size() && q < 4; ++q) groups[q] = R.H[q].to_string();
                if (!R.ok()) ++res.refused;
                std::ofstream out(path, std::ios::binary | std::ios::trunc);
                if (!out) throw std::runtime_error("cannot write '" + path + "'");
                out << "{\"schema\":\"integral_descent_cell/v1\",\"id\":\"" << c.id << "\",\"campaign\":\"" << JsonEsc(spec.name)
                    << "\",\"record\":" << R.to_json() << ",\"complete\":true}\n";
                if (!out) throw std::runtime_error("short write on '" + path + "'");
                ++res.ran;
            } catch (const std::exception& e) {
                ++res.failed;
                status = "failed";
                std::cerr << "campaign: descent cell " << c.id << " FAILED: " << e.what() << std::endl;
            }
            secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        std::ostringstream row;
        row << c.id << "\t" << c.N << "\t" << c.n << "\t";
        for (size_t i = 0; i < c.ps.size(); ++i) row << (i ? "," : "") << c.ps[i];
        row << "\t" << status << "\t" << groups[0] << "\t" << groups[1] << "\t" << groups[2] << "\t" << groups[3] << "\t" << fnv << "\t"
            << std::fixed << std::setprecision(3) << secs;
        sum << row.str() << "\n";
        res.lines.push_back(row.str());
        idx << (first ? "" : ",") << "{\"id\":\"" << c.id << "\",\"N\":" << c.N << ",\"n\":" << c.n << ",\"status\":\"" << status
            << "\",\"record_fnv1a64\":\"" << fnv << "\"}";
        first = false;
        if (verbose) std::cout << "  [descent] " << row.str() << std::endl;
    }
    idx << "]}\n";
    std::ofstream(spec.output_dir + "/summary.tsv") << sum.str();
    std::ofstream(spec.output_dir + "/index.json") << idx.str();
    std::cout << "EFFECTIVE campaign arm=descent cells=" << res.cells << " ran=" << res.ran << " resumed=" << res.resumed
              << " refused=" << res.refused << " failed=" << res.failed << std::endl;
    return res;
}

int RunDescentCampaignCli(const CampaignSpec& spec, bool dry_run, bool verbose) {
    if (dry_run) {
        const std::vector<DescentCellPlan> plan = PlanDescentCells(spec);
        std::cout << "descent campaign '" << spec.name << "': " << plan.size()
                  << " cells (integral_descent_record/v1, coefficient_ring = Z, fingerprint 0x" << hex16(DescentFingerprint(spec)) << ")\n";
        for (const DescentCellPlan& c : plan) {
            std::cout << "    " << c.id << "  N=" << c.N << " n=" << c.n << " primes=";
            for (size_t i = 0; i < c.ps.size(); ++i) std::cout << (i ? "," : "") << c.ps[i];
            std::cout << " D=" << rec::RecordModuleRank(c.N, c.n) << "\n";
        }
        std::cout << "  " << plan.size() << " cells total\n";
        return 0;
    }
    const DescentCampaignResult r = RunDescentCampaign(spec, verbose);
    return r.failed == 0 ? 0 : 2;
}

}  // namespace campaign
