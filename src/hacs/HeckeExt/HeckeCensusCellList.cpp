// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 12, WP-6 -- the census cell list (see the header).
#include "HeckeCensusCellList.h"

#include <cctype>
#include <fstream>
#include <set>
#include <sstream>
#include <utility>

namespace hecke_ext {

namespace {

bool IsPrime32(uint32_t v) {
    if (v < 2) return false;
    for (uint32_t d = 2; static_cast<uint64_t>(d) * d <= v; ++d)
        if (v % d == 0) return false;
    return true;
}

// A decimal field of at most 9 digits (fits uint32_t).
bool ParseField(const std::string& tok, uint32_t& v) {
    if (tok.empty() || tok.size() > 9) return false;
    for (char c : tok)
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    v = static_cast<uint32_t>(std::stoul(tok));
    return true;
}

} // namespace

bool ParseCensusCellList(const std::string& text, std::vector<CensusCell>& out, std::string& error) {
    out.clear();
    error.clear();
    std::set<std::pair<uint32_t, uint32_t>> seen;
    std::istringstream in(text);
    std::string line;
    uint32_t k = 0;
    auto fail = [&](const std::string& why) {
        error = "line " + std::to_string(k) + ": " + why;
        out.clear();
        return false;
    };
    while (std::getline(in, line)) {
        ++k;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        std::istringstream ls(line);
        std::vector<std::string> tok;
        std::string t;
        while (ls >> t) tok.push_back(t);
        if (tok.empty()) continue;
        if (tok.size() < 2 || tok.size() > 3)
            return fail("expected `<n> <ell> [<u_ell_budget_mb>]`, got " + std::to_string(tok.size()) + " field(s)");
        CensusCell c;
        if (!ParseField(tok[0], c.n)) return fail("n `" + tok[0] + "` is not a decimal integer");
        if (!ParseField(tok[1], c.ell)) return fail("ell `" + tok[1] + "` is not a decimal integer");
        if (tok.size() == 3 && !ParseField(tok[2], c.u_ell_budget_mb))
            return fail("u_ell_budget_mb `" + tok[2] + "` is not a decimal integer");
        if (c.n % 2 != 0) return fail("n = " + std::to_string(c.n) + " is odd (odd n vanishes on the model)");
        if (c.ell < 5 || !IsPrime32(c.ell)) return fail("ell = " + std::to_string(c.ell) + " is not a prime >= 5");
        if (!seen.insert({ c.n, c.ell }).second)
            return fail("cell (" + std::to_string(c.n) + ", " + std::to_string(c.ell) + ") listed twice");
        out.push_back(c);
    }
    if (out.empty()) {
        error = "the cell list holds no cell";
        return false;
    }
    return true;
}

bool ReadCensusCellList(const std::string& path, std::vector<CensusCell>& out, std::string& error) {
    std::ifstream f(path);
    if (!f) {
        out.clear();
        error = "cannot read the cell list `" + path + "`";
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ParseCensusCellList(ss.str(), out, error);
}

std::vector<CensusCell> CensusEdition1Grid() {
    std::vector<CensusCell> g;
    for (uint32_t ell : { 5u, 7u, 11u, 13u })
        for (uint32_t n = 0; n <= 200; n += 2) g.push_back({ n, ell, 0 });
    g.push_back({ 10, 691, 0 });
    g.push_back({ 198, 691, 8192 });
    return g;
}

} // namespace hecke_ext
