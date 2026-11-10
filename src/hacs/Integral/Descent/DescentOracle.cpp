// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// The frozen O1 table ([IB] Appendix A, integral_line_probe.py group rows, n <= 20), MOVED VERBATIM from
// Integral/HeckeFreeModelValidation.cpp (slot-9 WP-11) by DESCENT D0 so that HeckeFreeModelValidation and
// IntegralDescentGroupsValidation read one copy; plus the reader of the D0 oracle table of record
// (docs/implementation/slot24/descent_oracle_n128.tsv).  See DescentOracle.h.
#include "DescentOracle.h"

#include <fstream>
#include <sstream>

#include "../../../ModArithmetic/EngineCheck.h"

namespace integral_descent {

namespace {

// ---------------------------------------------------------------------------
// O1, FROZEN: H^q(SL_2(Z); Sym^n Z^2), q = 0..6, n = 0..20 -- the "group" rows
// of docs/implementation/integral_line_probe.py (unchanged since a500d67,
// the pinned O1), run at this package's base d6bc839 as
// `integral_line_probe.py 0,1,...,20 7`, copied VERBATIM.  On every row and
// degree Appendix A of [IB] prints (n <= 8 and even n <= 20, q <= 5 through
// H^4 = H^2, H^5 = H^3) these lines ARE Appendix A (checked against it by
// script, WP-11 report Sec. 2); the odd rows 9..19 are the probe's own output
// (O1 extended), as in WP-10.  Parsed below; never adjusted.
const char* const kProbeGroupRows[] = {
    "group n=0: H^0=Z^1 | H^1=Z^0 | H^2=Z^0+Z/2^2+Z/3 | H^3=Z^0 | H^4=Z^0+Z/2^2+Z/3 | H^5=Z^0 | H^6=Z^0+Z/2^2+Z/3",
    "group n=1: H^0=Z^0 | H^1=Z^0 | H^2=Z^0+Z/2 | H^3=Z^0 | H^4=Z^0+Z/2 | H^5=Z^0 | H^6=Z^0+Z/2",
    "group n=2: H^0=Z^0 | H^1=Z^1+Z/2 | H^2=Z^0 | H^3=Z^0+Z/2+Z/2 | H^4=Z^0 | H^5=Z^0+Z/2+Z/2 | H^6=Z^0",
    "group n=3: H^0=Z^0 | H^1=Z^0+Z/2 | H^2=Z^0+Z/2 | H^3=Z^0+Z/2 | H^4=Z^0+Z/2 | H^5=Z^0+Z/2 | H^6=Z^0+Z/2",
    "group n=4: H^0=Z^0 | H^1=Z^1+Z/2^2+Z/3 | H^2=Z^0+Z/2^2 | H^3=Z^0+Z/2+Z/2+Z/3 | H^4=Z^0+Z/2^2 | H^5=Z^0+Z/2+Z/2+Z/3 | H^6=Z^0+Z/2^2",
    "group n=5: H^0=Z^0 | H^1=Z^0+Z/2 | H^2=Z^0+Z/2+Z/2 | H^3=Z^0+Z/2 | H^4=Z^0+Z/2+Z/2 | H^5=Z^0+Z/2 | H^6=Z^0+Z/2+Z/2",
    "group n=6: H^0=Z^0 | H^1=Z^1+Z/2+Z/2^2+Z/3+Z/5 | H^2=Z^0+Z/2+Z/3 | H^3=Z^0+Z/2+Z/2+Z/2 | H^4=Z^0+Z/2+Z/3 | H^5=Z^0+Z/2+Z/2+Z/2 | H^6=Z^0+Z/2+Z/3",
    "group n=7: H^0=Z^0 | H^1=Z^0+Z/2 | H^2=Z^0+Z/2+Z/2+Z/2 | H^3=Z^0+Z/2 | H^4=Z^0+Z/2+Z/2+Z/2 | H^5=Z^0+Z/2 | H^6=Z^0+Z/2+Z/2+Z/2",
    "group n=8: H^0=Z^0 | H^1=Z^1+Z/2+Z/2^3+Z/3+Z/7 | H^2=Z^0+Z/2+Z/2^2 | H^3=Z^0+Z/2+Z/2+Z/2 | H^4=Z^0+Z/2+Z/2^2 | H^5=Z^0+Z/2+Z/2+Z/2 | H^6=Z^0+Z/2+Z/2^2",
    "group n=9: H^0=Z^0 | H^1=Z^0+Z/2+Z/2 | H^2=Z^0+Z/2+Z/2+Z/2 | H^3=Z^0+Z/2+Z/2 | H^4=Z^0+Z/2+Z/2+Z/2 | H^5=Z^0+Z/2+Z/2 | H^6=Z^0+Z/2+Z/2+Z/2",
    "group n=10: H^0=Z^0 | H^1=Z^3+Z/2+Z/2^2+Z/3 | H^2=Z^0+Z/2 | H^3=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/3 | H^4=Z^0+Z/2 | H^5=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/3 | H^6=Z^0+Z/2",
    "group n=11: H^0=Z^0 | H^1=Z^0+Z/2+Z/2 | H^2=Z^0+Z/2+Z/2+Z/2+Z/2 | H^3=Z^0+Z/2+Z/2 | H^4=Z^0+Z/2+Z/2+Z/2+Z/2 | H^5=Z^0+Z/2+Z/2 | H^6=Z^0+Z/2+Z/2+Z/2+Z/2",
    "group n=12: H^0=Z^0 | H^1=Z^1+Z/2+Z/2^2+Z/2^3+Z/3+Z/3^2+Z/5+Z/11 | H^2=Z^0+Z/2+Z/2+Z/2^2+Z/3 | H^3=Z^0+Z/2+Z/2+Z/2+Z/2 | H^4=Z^0+Z/2+Z/2+Z/2^2+Z/3 | H^5=Z^0+Z/2+Z/2+Z/2+Z/2 | H^6=Z^0+Z/2+Z/2+Z/2^2+Z/3",
    "group n=13: H^0=Z^0 | H^1=Z^0+Z/2+Z/2 | H^2=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2 | H^3=Z^0+Z/2+Z/2 | H^4=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2 | H^5=Z^0+Z/2+Z/2 | H^6=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2",
    "group n=14: H^0=Z^0 | H^1=Z^3+Z/2+Z/2+Z/2^2+Z/3+Z/13 | H^2=Z^0+Z/2+Z/2 | H^3=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2 | H^4=Z^0+Z/2+Z/2 | H^5=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2 | H^6=Z^0+Z/2+Z/2",
    "group n=15: H^0=Z^0 | H^1=Z^0+Z/2+Z/2+Z/2 | H^2=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2 | H^3=Z^0+Z/2+Z/2+Z/2 | H^4=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2 | H^5=Z^0+Z/2+Z/2+Z/2 | H^6=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2",
    "group n=16: H^0=Z^0 | H^1=Z^3+Z/2+Z/2^2+Z/2^4+Z/3+Z/3+Z/7 | H^2=Z^0+Z/2+Z/2+Z/2^2 | H^3=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2+Z/3 | H^4=Z^0+Z/2+Z/2+Z/2^2 | H^5=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2+Z/3 | H^6=Z^0+Z/2+Z/2+Z/2^2",
    "group n=17: H^0=Z^0 | H^1=Z^0+Z/2+Z/2+Z/2 | H^2=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2 | H^3=Z^0+Z/2+Z/2+Z/2 | H^4=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2 | H^5=Z^0+Z/2+Z/2+Z/2 | H^6=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2",
    "group n=18: H^0=Z^0 | H^1=Z^3+Z/2+Z/2+Z/2^2+Z/2^2+Z/3+Z/3^2+Z/5+Z/17 | H^2=Z^0+Z/2+Z/2+Z/2+Z/3 | H^3=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2 | H^4=Z^0+Z/2+Z/2+Z/2+Z/3 | H^5=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2 | H^6=Z^0+Z/2+Z/2+Z/2+Z/3",
    "group n=19: H^0=Z^0 | H^1=Z^0+Z/2+Z/2+Z/2 | H^2=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2 | H^3=Z^0+Z/2+Z/2+Z/2 | H^4=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2 | H^5=Z^0+Z/2+Z/2+Z/2 | H^6=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2",
    "group n=20: H^0=Z^0 | H^1=Z^3+Z/2+Z/2+Z/2^2+Z/2^3+Z/3+Z/3+Z/5+Z/19 | H^2=Z^0+Z/2+Z/2+Z/2+Z/2^2 | H^3=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2 | H^4=Z^0+Z/2+Z/2+Z/2+Z/2^2 | H^5=Z^0+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2+Z/2 | H^6=Z^0+Z/2+Z/2+Z/2+Z/2^2",
};

}  // namespace

// Parses one "H^q=Z^r+Z/p^a+..." term into a group type.
AbelianGroupType ParseProbeGroup(const std::string& term) {
    AbelianGroupType g;
    std::stringstream ss(term);
    std::string part;
    bool first = true;
    while (std::getline(ss, part, '+')) {
        if (first) {   // "Z^r"
            ENGINE_CHECK_MSG(part.rfind("Z^", 0) == 0, "O1 table: bad free part");
            g.free_rank = static_cast<uint32_t>(std::stoul(part.substr(2)));
            first = false;
            continue;
        }
        ENGINE_CHECK_MSG(part.rfind("Z/", 0) == 0, "O1 table: bad torsion part");
        const std::string pp = part.substr(2);
        const size_t caret = pp.find('^');
        const uint32_t ell = static_cast<uint32_t>(std::stoul(pp.substr(0, caret)));
        const uint32_t a = (caret == std::string::npos) ? 1u : static_cast<uint32_t>(std::stoul(pp.substr(caret + 1)));
        g.add_torsion(ell, a);
    }
    return g;
}

// table[n][q], q = 0..6.
const std::vector<std::vector<AbelianGroupType>>& AppendixAGroups() {
    static const std::vector<std::vector<AbelianGroupType>> t = [] {
        std::vector<std::vector<AbelianGroupType>> out;
        for (const char* row : kProbeGroupRows) {
            const std::string s(row);
            const size_t colon = s.find(": ");
            ENGINE_CHECK(s.rfind("group n=", 0) == 0 && colon != std::string::npos);
            const uint32_t n = static_cast<uint32_t>(std::stoul(s.substr(8, colon - 8)));
            ENGINE_CHECK_MSG(n == out.size(), "O1 table: rows out of order");
            std::vector<AbelianGroupType> degs;
            std::stringstream ss(s.substr(colon + 2));
            std::string cell;
            while (std::getline(ss, cell, '|')) {
                // trim, then strip "H^q="
                const size_t b = cell.find_first_not_of(' '), e = cell.find_last_not_of(' ');
                cell = cell.substr(b, e - b + 1);
                const size_t eq = cell.find('=');
                ENGINE_CHECK(eq != std::string::npos);
                ENGINE_CHECK_MSG(std::stoul(cell.substr(2, eq - 2)) == degs.size(), "O1 table: degrees out of order");
                degs.push_back(ParseProbeGroup(cell.substr(eq + 1)));
            }
            ENGINE_CHECK(degs.size() == 7);
            out.push_back(std::move(degs));
        }
        return out;
    }();
    return t;
}

// The table's own group form: "0", or "Z^r" (omitted if r = 0; "Z" read as r = 1) then "Z/p" / "Z/p^k" joined by '+'.
AbelianGroupType ParseOracleGroup(const std::string& term) {
    AbelianGroupType g;
    if (term == "0") return g;
    std::stringstream ss(term);
    std::string part;
    while (std::getline(ss, part, '+')) {
        if (part == "Z") { g.free_rank += 1; continue; }
        if (part.rfind("Z^", 0) == 0) { g.free_rank += static_cast<uint32_t>(std::stoul(part.substr(2))); continue; }
        ENGINE_CHECK_MSG(part.rfind("Z/", 0) == 0, ("oracle table: bad group term '" + part + "'").c_str());
        const std::string pp = part.substr(2);
        const size_t caret = pp.find('^');
        const uint32_t ell = static_cast<uint32_t>(std::stoul(pp.substr(0, caret)));
        const uint32_t a = (caret == std::string::npos) ? 1u : static_cast<uint32_t>(std::stoul(pp.substr(caret + 1)));
        g.add_torsion(ell, a);
    }
    return g;
}

// ---- the D0 oracle table of record -----------------------------------------------------
// Accepted lines (comments '#', blank lines skipped):
//   (a) the probe's own row form   "group n=<n>: H^0=... | H^1=... | ..."
//   (b) a tab-separated row         "<n>\t<q>\t<group>"   (an optional header row starting with "n\t" is skipped)
OracleTable LoadOracleTable(const std::string& path) {
    OracleTable T;
    T.path = path;
    std::ifstream f(path);
    if (!f) {
        T.error = "cannot open " + path;
        return T;
    }
    T.found = true;
    std::string line;
    uint32_t lineno = 0;
    try {
        while (std::getline(f, line)) {
            ++lineno;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            if (line.rfind("group n=", 0) == 0) {
                const size_t colon = line.find(": ");
                ENGINE_CHECK_MSG(colon != std::string::npos, "oracle table: bad group row");
                const uint32_t n = static_cast<uint32_t>(std::stoul(line.substr(8, colon - 8)));
                std::stringstream ss(line.substr(colon + 2));
                std::string cell;
                uint32_t q = 0;
                while (std::getline(ss, cell, '|')) {
                    const size_t b = cell.find_first_not_of(' '), e = cell.find_last_not_of(' ');
                    cell = cell.substr(b, e - b + 1);
                    const size_t eq = cell.find('=');
                    ENGINE_CHECK_MSG(eq != std::string::npos && std::stoul(cell.substr(2, eq - 2)) == q,
                                     "oracle table: degrees out of order");
                    T.groups[n].push_back(ParseProbeGroup(cell.substr(eq + 1)));
                    ++q;
                }
                continue;
            }
            if (line.rfind("n\t", 0) == 0) continue;   // header
            std::stringstream ss(line);
            std::string a, b, g;
            ENGINE_CHECK_MSG(std::getline(ss, a, '\t') && std::getline(ss, b, '\t') && std::getline(ss, g, '\t'),
                             "oracle table: expected n<TAB>q<TAB>group");
            const uint32_t n = static_cast<uint32_t>(std::stoul(a)), q = static_cast<uint32_t>(std::stoul(b));
            auto& row = T.groups[n];
            ENGINE_CHECK_MSG(row.size() == q, "oracle table: degrees out of order");
            row.push_back(ParseOracleGroup(g));
        }
    } catch (const std::exception& e) {
        T.error = "line " + std::to_string(lineno) + ": " + e.what();
        T.groups.clear();
    }
    return T;
}

}  // namespace integral_descent
