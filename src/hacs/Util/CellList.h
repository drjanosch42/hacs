// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Util/CellList.h -- the "a:b:c,a:b:c,..." cell-list grammars of the *_CELLS
// hooks, ONE definition (S24 CL-1; CODEAUDIT A4; CLEANUP_plan_1030.md)
// ===========================================================================
//
// 16+ bespoke *_CELLS readers exist.  Three grammars are pinned by tests, and
// this header reproduces each one BYTE FOR BYTE (accepted cells and error
// texts); SharedHelpersValidation compares all three with the originals:
//
//  ParseStrict(value, grammar, out, why) -- the strict k-field grammar,
//  first error wins, 1-based position.  Two message styles:
//   Style::Hook     = Hecke/HeckeLevelNDegenValidation.cpp ParseCells (HookParseCells;
//                     HECKE_LEVEL_N_{DEGEN,R3K,SPARSE}_CELLS), pinned by
//                     HeckeSmallCarriesValidation (1).  Its copy THROWS ENGINE_CHECK with
//                     `why` as the message; a migrated site keeps that:
//                     ENGINE_CHECK_MSG(ParseStrict(...), why.c_str()).
//   Style::SatCert  = HeckeExt/HeckeSatCertDumpValidation.cpp ParseCells
//                     (HECKE_DUMP_SAT_CERT_CELLS), pinned by HeckeSmallCarriesValidation (3).
//  ParseLax(value) -- census::ParseCellList (HeckeCensusJoin.h; the
//  HECKE_CENSUS_LATTICE / HECKE_FORMS_CENSUS cell filters): exactly two
//  fields, EVERY bad token collected as "token <k> '<text>'", nothing thrown.
//
// The drift between them (kept, not repaired -- each is its hook's contract):
//   * digits per field: lax 9, SatCert 12, Hook 18;
//   * a trailing ',': refused by position in Hook / SatCert, but SWALLOWED by the
//     lax grammar (std::getline drops it: "654:37," reads one cell, no error);
//   * an empty value: Hook refuses ("lists no cell"), SatCert refuses ("empty cell
//     list"), lax reports set = false;
//   * an empty token or field: Hook "an empty cell or field" / "field '' is not
//     a decimal number"; SatCert the grammar's own `want` text; lax a bad token;
//   * per-field ranges and a cell predicate exist only in SatCert (the Grammar
//     carries them; Hook has none, the lax grammar none).
//
// (HeckeCensusCellList.h's FILE grammar -- whitespace fields, '#' comments,
// "line <k>:" errors, HeckeCensusCellListValidation -- is a different grammar,
// already one definition, and is not touched.)
//
// (S24 CL-6) The SatCert and lax originals now route through this header
// (HeckeSatCertDumpValidation ParseCells, census::CellList = LaxList); their texts
// stay pinned by HeckeSmallCarriesValidation (3) and HeckeCensusLatticeValidation (J2).
// Tokenize(value) is the split under the other HeckeExt readers' own grammars.
//
// EffectiveLine(name, arm, cells) = "EFFECTIVE <name> arm=<arm> cells=<k>",
// the EFFECTIVE convention's line (every env-armed _ALL / _CELLS arm prints one).
// Header-only, STL only.
// ===========================================================================
#pragma once
#ifndef OC_UTIL_CELL_LIST_H
#define OC_UTIL_CELL_LIST_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace oc_util {
namespace cells {

using Cell = std::vector<uint64_t>;

enum class Style : unsigned char { Hook, SatCert };

struct Range {
    uint64_t lo = 0, hi = UINT64_MAX;
};

struct Grammar {
    Style style = Style::Hook;
    std::string name;                 // Hook: the variable's name, the message prefix
    size_t fields = 2;                // exact number of ':'-separated fields
    size_t max_digits = 18;           // Hook 18, SatCert 12
    std::vector<Range> ranges;        // SatCert: per field, inclusive (empty = none)
    std::function<bool(const Cell&)> accept;   // SatCert: the cell predicate (empty = none)
    std::string want;                 // SatCert: the parenthesised reason of a refused cell
};

inline bool IsDigits(const std::string& x, size_t max_digits) {
    if (x.empty() || x.size() > max_digits) return false;
    for (char c : x)
        if (c < '0' || c > '9') return false;
    return true;
}

// True on success (`out` appended in order); false with `why` set on the first refusal.
inline bool ParseStrict(const std::string& value, const Grammar& g, std::vector<Cell>& out, std::string& why) {
    if (g.style == Style::Hook) {
        const std::string none = g.name + ": the value lists no cell (refused: the arm would run vacuously)";
        if (value.empty()) { why = none; return false; }
        if (value.back() == ',') { why = g.name + ": an empty trailing cell"; return false; }
        std::stringstream ss(value);
        std::string item;
        size_t idx = 0;
        std::vector<Cell> got;
        while (std::getline(ss, item, ',')) {
            ++idx;
            const std::string where = g.name + ": cell " + std::to_string(idx) + " '" + item + "': ";
            if (item.empty() || item.back() == ':') { why = where + "an empty cell or field"; return false; }
            Cell v;
            std::stringstream is(item);
            std::string f;
            while (std::getline(is, f, ':')) {
                if (!IsDigits(f, g.max_digits)) { why = where + "field '" + f + "' is not a decimal number"; return false; }
                v.push_back(std::strtoull(f.c_str(), nullptr, 10));
            }
            if (v.size() != g.fields) {
                why = where + "expected " + std::to_string(g.fields) + " colon-separated fields";
                return false;
            }
            got.push_back(std::move(v));
        }
        if (got.empty()) { why = none; return false; }
        out.insert(out.end(), got.begin(), got.end());
        return true;
    }
    // Style::SatCert
    if (!value.empty() && value.back() == ',') {
        why = "cell " + std::to_string(std::count(value.begin(), value.end(), ',') + 1) + " '' (an empty trailing cell)";
        return false;
    }
    std::stringstream ss(value);
    std::string tok;
    size_t pos = 0;
    std::vector<Cell> got;
    while (std::getline(ss, tok, ',')) {
        ++pos;
        if (!tok.empty() && tok.back() == ':') {
            why = "cell " + std::to_string(pos) + " '" + tok + "' (an empty trailing field)";
            return false;
        }
        std::vector<std::string> f;
        std::stringstream ts(tok);
        std::string x;
        while (std::getline(ts, x, ':')) f.push_back(x);
        bool ok = f.size() == g.fields;
        Cell v;
        for (size_t i = 0; ok && i < f.size(); ++i) {
            if (!IsDigits(f[i], g.max_digits)) { ok = false; break; }
            const uint64_t y = std::stoull(f[i]);
            if (i < g.ranges.size() && (y < g.ranges[i].lo || y > g.ranges[i].hi)) { ok = false; break; }
            v.push_back(y);
        }
        if (ok && g.accept) ok = g.accept(v);
        if (!ok) {
            why = "cell " + std::to_string(pos) + " '" + tok + "' (" + g.want + ")";
            return false;
        }
        got.push_back(std::move(v));
    }
    if (got.empty()) { why = "empty cell list"; return false; }
    out.insert(out.end(), got.begin(), got.end());
    return true;
}

// census::ParseCellList's result, field for field.
struct LaxList {
    std::vector<std::pair<uint32_t, uint32_t>> cells;
    std::vector<std::string> bad;     // "token <k> '<text>'"
    bool set = false;                 // the value was non-null and non-empty
};

inline LaxList ParseLax(const char* value) {
    LaxList out;
    if (!value || !*value) return out;
    out.set = true;
    std::stringstream s(value);
    std::string tok;
    uint32_t k = 0;
    while (std::getline(s, tok, ',')) {
        ++k;
        const size_t c = tok.find(':');
        if (c == std::string::npos || !IsDigits(tok.substr(0, c), 9) || !IsDigits(tok.substr(c + 1), 9)) {
            out.bad.push_back("token " + std::to_string(k) + " '" + tok + "'");
            continue;
        }
        out.cells.push_back({static_cast<uint32_t>(std::stoul(tok.substr(0, c))), static_cast<uint32_t>(std::stoul(tok.substr(c + 1)))});
    }
    return out;
}

// (S24 CL-6) The tokenizer under the per-hook grammars that carry their own messages and
// predicates (HECKE_LAMBDA_TOWER_CELLS, HECKE_SHAPIRO_CELLS, ...): one Token per ','-separated
// item, std::getline semantics (a trailing ',' is swallowed, an inner empty item is a token).
// `fields` is filled only when the item is numeric: every ':'-field a decimal number of at
// most `max_digits` digits, no empty item, no empty field (a trailing ':' included).
struct Token {
    size_t pos = 0;                   // 1-based
    std::string text;
    std::vector<std::string> raw;     // the ':'-fields as written (std::getline split)
    Cell fields;                      // the numbers, when `numeric`
    bool numeric = false;
    bool Fits(size_t min_fields, size_t max_fields) const {
        return numeric && fields.size() >= min_fields && fields.size() <= max_fields;
    }
};

inline std::vector<Token> Tokenize(const std::string& value, size_t max_digits = 18) {
    std::vector<Token> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        Token t;
        t.pos = out.size() + 1;
        t.text = item;
        std::stringstream is(item);
        std::string f;
        while (std::getline(is, f, ':')) t.raw.push_back(f);
        t.numeric = !item.empty() && item.back() != ':';
        for (const std::string& x : t.raw) {
            if (!IsDigits(x, max_digits)) { t.numeric = false; break; }
            t.fields.push_back(std::strtoull(x.c_str(), nullptr, 10));
        }
        if (!t.numeric) t.fields.clear();
        out.push_back(std::move(t));
    }
    return out;
}

inline std::string EffectiveLine(const std::string& name, const std::string& arm, size_t cells) {
    return "EFFECTIVE " + name + " arm=" + arm + " cells=" + std::to_string(cells);
}

} // namespace cells
} // namespace oc_util

#endif // OC_UTIL_CELL_LIST_H
