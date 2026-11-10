// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_CENSUS_JOIN_H
#define HECKE_EXT_CENSUS_JOIN_H

// ===========================================================================
// hecke-extensions slot 16, WP-3 -- THE CENSUS RE-RUN JOIN
//
// A census row (hecke_census/v1, hecke_census_dataset/v1-v2) is re-run by
// HeckeFormsHeckeAlgebraValidation arm (T) and HeckeCensusLatticeValidation
// arm (ALL).  Both re-run the ext driver on the row's cell.  The row's ideal ids
// mean something only under the SAME operator set as the record the row was
// built from.  The driver's operator set depends on the `u_ell` setting: U_ell
// is in it iff the frozen RSS model's prediction fits the budget.
// Slot 15 found both re-runs at the default `u_ell` (auto, 2048 MB) on rows
// whose records ran at `budget_mb=16384`.  WP-15: arm (T) rc = 1 on 230 cells
// at ell 61 ... 103, n >= 302.  WP-12 Sec. 8.1: the lattice ideals 17 / 18 of
// (654, 37) merged, and every later id shifted by one.
//
// (1) CensusUEllBudgetMb: the `u_ell` budget of a row's SOURCE SET, read off
//     the records' own `settings.u_ell` (scanned 2026-10-01 over every record of
//     every set): ext_first / ext_census_l7l11 `auto` (0 here: the setting in
//     force, the default), ext_u691 `budget_mb=8192`, the np / npx cluster sets
//     `budget_mb=16384` (RA / RS included: the October kit's spec).  An unknown
//     set is NOT run at the default: the caller refuses the row by its TSV line
//     number.
// (2) ReadRecordSystems: a row's eigen-system (split.primes, split.eigenvalues
//     at r = 1) and witness_poly, read from its record (`record_path` under one
//     of the roots of HECKE_CENSUS_LATTICE_RECORDS, a ':' list, default the
//     tree's docs/matrix-engine/results).  A minimal read-only JSON reader, with
//     no dependency.  Used by the lattice join, which joins by system, not by id.
// (3) (slot-16 WP-3 audit repairs) CompareSystem / CheckUEllOperator, shared by
//     the lattice join and arm (T)'s join verification; RecordRootsFrom (the
//     roots reported, an explicit empty / missing root detectable);
//     ParseCellList (a malformed cell token refused by position);
//     ParseRecordSystems (malformed operators / duplicate ids refused).
//
// Read-only: nothing here writes, and nothing reaches a record, golden or digest.
// Header-only (inline), so it adds no source to the build.
// ===========================================================================

#include "../Util/CellList.h"

#include <sys/stat.h>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace hecke_ext {
namespace census {

// The source set's `u_ell` budget in MB; 0 = `auto` at the default budget (the
// setting in force).  *known = false for a set not in the table.
inline uint32_t CensusUEllBudgetMb(const std::string& source_set, bool* known) {
    static const std::map<std::string, uint32_t> table = {
        {"ext_first", 0},    {"ext_census_l7l11", 0}, {"ext_u691", 8192},
        {"np_ext_a", 16384}, {"np_ext_b", 16384},     {"np_local", 16384}, {"np_local_d2", 16384},
        {"npx_rb1", 16384},  {"npx_rb2", 16384},      {"npx_rb3", 16384},  {"npx_rb4", 16384},
        {"npx_ra1", 16384},  {"npx_ra2", 16384},
        {"npx_rs1", 16384},  {"npx_rs2", 16384},      {"npx_rs3", 16384},
    };
    const auto it = table.find(source_set);
    if (known) *known = it != table.end();
    return it == table.end() ? 0u : it->second;
}

// "auto" for 0, else "budget_mb=<mb>" -- the record's own spelling.
inline std::string UEllText(uint32_t budget_mb) {
    return budget_mb == 0 ? std::string("auto") : "budget_mb=" + std::to_string(budget_mb);
}

// Parses an override variable's value: a positive integer number of MB.  An
// unset or empty variable is "no override" (true, *present = false); anything
// else that is not a positive integer returns false with *err.
inline bool ParseUEllOverride(const char* value, uint32_t* budget_mb, bool* present, std::string* err) {
    *present = value != nullptr && *value != '\0';
    if (!*present) return true;
    const std::string v(value);
    bool digits = !v.empty() && v.size() <= 9;
    for (char c : v) digits = digits && std::isdigit(static_cast<unsigned char>(c));
    if (!digits || std::stoul(v) == 0) {
        if (err) *err = "not a positive integer number of MB: '" + v + "'";
        return false;
    }
    *budget_mb = static_cast<uint32_t>(std::stoul(v));
    return true;
}

struct RecordIdeal {
    uint32_t id = 0;
    bool split_ok = false;                // split.ok and, at r = 1, the eigenvalues read
    uint32_t r = 0;                       // residue degree of the split
    std::vector<uint64_t> primes;         // split.primes
    std::vector<uint32_t> a;              // split.eigenvalues at r = 1, aligned with primes
    std::vector<uint32_t> witness_poly;
};

struct RecordSystems {
    bool ok = false;
    std::string path;                     // the file read
    std::string why;                      // why not ok
    uint32_t n = 0, ell = 0;
    std::vector<uint64_t> operators;      // operators_computed
    bool u_ell_included = false;
    std::map<uint32_t, RecordIdeal> ideals;
};

// The record roots: <env> (':'-separated; default HECKE_CENSUS_LATTICE_RECORDS) if set, else
// <OC_SOURCE_DIR>/docs/matrix-engine/results.
//
// (slot-16 WP-3 audit repair, finding 2) The root list is reported, never silently empty:
// RecordRootSet says whether the variable was set, whether a set variable named no root at all
// (e.g. ":"), and which roots are not directories.  The callers print it on their EFFECTIVE
// line and refuse an explicit list that is empty or names a missing directory.
struct RecordRootSet {
    std::vector<std::string> roots;
    bool explicit_ = false;               // the variable was set (non-empty)
    bool explicit_empty = false;          // set, but naming no root (e.g. ":")
    std::vector<std::string> missing;     // roots that are not directories
    std::string text() const {
        std::string s = explicit_ ? "env[" : "default[";
        for (size_t i = 0; i < roots.size(); ++i) s += (i ? ":" : "") + roots[i];
        s += "]";
        if (roots.empty()) s += "EMPTY";
        if (!missing.empty()) s += ",missing:" + std::to_string(missing.size());
        return s;
    }
};

inline bool IsDirectory(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline RecordRootSet RecordRootsFrom(const char* env_name) {
    RecordRootSet out;
    const char* e = std::getenv(env_name);
    if (e && *e) {
        out.explicit_ = true;
        std::stringstream s(e);
        std::string tok;
        while (std::getline(s, tok, ':'))
            if (!tok.empty()) out.roots.push_back(tok);
        out.explicit_empty = out.roots.empty();
    } else {
#ifdef OC_SOURCE_DIR
        out.roots.push_back(std::string(OC_SOURCE_DIR) + "/docs/matrix-engine/results");
#endif
    }
    for (const std::string& r : out.roots)
        if (!IsDirectory(r)) out.missing.push_back(r);
    return out;
}

inline std::vector<std::string> RecordRoots(const char* env_name = "HECKE_CENSUS_LATTICE_RECORDS") {
    return RecordRootsFrom(env_name).roots;
}

// "[3,1]" / "3,1" -> {3, 1}: every maximal digit run.  A census list column (witness_poly).
inline std::vector<uint32_t> ParseUIntList(const std::string& s) {
    std::vector<uint32_t> out;
    std::string cur;
    for (char ch : s) {
        if (ch >= '0' && ch <= '9') {
            cur.push_back(ch);
        } else if (!cur.empty()) {
            out.push_back(static_cast<uint32_t>(std::stoul(cur)));
            cur.clear();
        }
    }
    if (!cur.empty()) out.push_back(static_cast<uint32_t>(std::stoul(cur)));
    return out;
}

inline std::string ListText(const std::vector<uint64_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

// (slot-16 WP-3 audit repair, finding 4) A cell-list variable "n:ell,n:ell,...": every token must be
// two digit runs around one ':' (at most 9 digits each); a bad token is reported by its 1-based
// position, never dropped.  (S24 CL-6) The grammar is Util/CellList.h's lax one, byte for byte.
using CellList = oc_util::cells::LaxList;

inline CellList ParseCellList(const char* env_name) { return oc_util::cells::ParseLax(std::getenv(env_name)); }

namespace detail {

// A minimal read-only JSON value.
struct Json {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    std::string text;                     // a number's text, or a string
    std::vector<Json> arr;
    std::map<std::string, Json> obj;
    const Json* get(const std::string& k) const {
        if (kind != Obj) return nullptr;
        const auto it = obj.find(k);
        return it == obj.end() ? nullptr : &it->second;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s) {}
    bool parse(Json& out) {
        ok_ = true;
        value(out, 0);
        ws();
        return ok_ && i_ == s_.size();
    }

private:
    const std::string& s_;
    size_t i_ = 0;
    bool ok_ = true;

    void ws() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }
    bool lit(const char* w) {
        const std::string t(w);
        if (s_.compare(i_, t.size(), t) != 0) return false;
        i_ += t.size();
        return true;
    }
    void str(std::string& out) {
        ++i_;   // the opening quote
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\' && i_ + 1 < s_.size()) {
                const char c = s_[i_ + 1];
                if (c == 'u') {   // no reader here needs a \u escape's value
                    out.push_back('?');
                    i_ += 6;
                    continue;
                }
                out.push_back(c == 'n' ? '\n' : c == 't' ? '\t' : c);
                i_ += 2;
                continue;
            }
            out.push_back(s_[i_++]);
        }
        if (i_ >= s_.size()) {
            ok_ = false;
            return;
        }
        ++i_;
    }
    void value(Json& v, int depth) {
        ws();
        if (!ok_ || i_ >= s_.size() || depth > 200) {
            ok_ = false;
            return;
        }
        const char c = s_[i_];
        if (c == '{') {
            v.kind = Json::Obj;
            ++i_;
            ws();
            if (i_ < s_.size() && s_[i_] == '}') {
                ++i_;
                return;
            }
            while (ok_) {
                ws();
                if (i_ >= s_.size() || s_[i_] != '"') {
                    ok_ = false;
                    return;
                }
                std::string k;
                str(k);
                ws();
                if (i_ >= s_.size() || s_[i_] != ':') {
                    ok_ = false;
                    return;
                }
                ++i_;
                value(v.obj[k], depth + 1);
                ws();
                if (i_ < s_.size() && s_[i_] == ',') { ++i_; continue; }
                if (i_ < s_.size() && s_[i_] == '}') { ++i_; return; }
                ok_ = false;
            }
        } else if (c == '[') {
            v.kind = Json::Arr;
            ++i_;
            ws();
            if (i_ < s_.size() && s_[i_] == ']') {
                ++i_;
                return;
            }
            while (ok_) {
                v.arr.emplace_back();
                value(v.arr.back(), depth + 1);
                ws();
                if (i_ < s_.size() && s_[i_] == ',') { ++i_; continue; }
                if (i_ < s_.size() && s_[i_] == ']') { ++i_; return; }
                ok_ = false;
            }
        } else if (c == '"') {
            v.kind = Json::Str;
            str(v.text);
        } else if (lit("true")) {
            v.kind = Json::Bool;
            v.b = true;
        } else if (lit("false")) {
            v.kind = Json::Bool;
        } else if (lit("null")) {
            v.kind = Json::Null;
        } else {
            v.kind = Json::Num;
            const size_t a = i_;
            while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '-' ||
                                      s_[i_] == '+' || s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E'))
                ++i_;
            if (a == i_) {
                ok_ = false;
                return;
            }
            v.text = s_.substr(a, i_ - a);
        }
    }
};

inline bool AsU64(const Json* j, uint64_t* out) {
    if (!j || j->kind != Json::Num || j->text.empty() || j->text.size() > 19) return false;
    for (char c : j->text)
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    *out = std::stoull(j->text);
    return true;
}

} // namespace detail

// Parses one record's text (the part of ReadRecordSystems after the file read; public so that the
// default arm (J) of HeckeCensusLatticeValidation can feed it synthetic records).
// (slot-16 WP-3 audit repair, finding 6) Malformed data is not swallowed: a non-integer entry of
// operators_computed and a duplicate ideal id return ok = false with a reason.
inline RecordSystems ParseRecordSystems(const std::string& text, const std::string& path) {
    using detail::AsU64;
    using detail::Json;
    RecordSystems out;
    out.path = path;
    Json j;
    if (!detail::JsonParser(text).parse(j) || j.kind != Json::Obj) {
        out.why = "record_unparsable";
        return out;
    }
    uint64_t v = 0;
    if (!AsU64(j.get("n_sym"), &v)) {
        out.why = "record_no_n_sym";
        return out;
    }
    out.n = static_cast<uint32_t>(v);
    if (!AsU64(j.get("ell"), &v)) {
        out.why = "record_no_ell";
        return out;
    }
    out.ell = static_cast<uint32_t>(v);
    if (const Json* oc = j.get("operators_computed")) {
        for (const Json& x : oc->arr) {
            if (!AsU64(&x, &v)) {
                out.why = "record_operator_not_an_integer";
                return out;
            }
            out.operators.push_back(v);
        }
    }
    if (const Json* u = j.get("u_ell"))
        if (const Json* inc = u->get("included")) out.u_ell_included = inc->kind == Json::Bool && inc->b;
    const Json* ideals = j.get("ideals");
    if (!ideals || ideals->kind != Json::Arr) {
        out.why = "record_no_ideals";
        return out;
    }
    for (const Json& id : ideals->arr) {
        RecordIdeal r;
        if (!AsU64(id.get("id"), &v)) {
            out.why = "record_ideal_without_id";
            return out;
        }
        r.id = static_cast<uint32_t>(v);
        if (out.ideals.count(r.id)) {
            out.why = "record_duplicate_ideal_" + std::to_string(r.id);
            return out;
        }
        if (const Json* w = id.get("witness_poly"))
            for (const Json& x : w->arr)
                if (AsU64(&x, &v)) r.witness_poly.push_back(static_cast<uint32_t>(v));
        if (const Json* sp = id.get("split")) {
            const Json* okj = sp->get("ok");
            r.split_ok = okj && okj->kind == Json::Bool && okj->b;
            if (AsU64(sp->get("r"), &v)) r.r = static_cast<uint32_t>(v);
            if (const Json* pr = sp->get("primes"))
                for (const Json& x : pr->arr)
                    if (AsU64(&x, &v)) r.primes.push_back(v);
            const Json* ev = sp->get("eigenvalues");
            if (r.split_ok && r.r == 1 && ev && ev->arr.size() == r.primes.size()) {
                for (const Json& x : ev->arr) {
                    if (x.arr.size() != 1 || !AsU64(&x.arr[0], &v)) {
                        r.a.clear();
                        r.split_ok = false;
                        break;
                    }
                    r.a.push_back(static_cast<uint32_t>(v));
                }
            } else {
                r.split_ok = r.split_ok && r.r > 1;   // r > 1: no F_ell system; r = 1 unreadable: not ok
            }
        }
        out.ideals[r.id] = r;
    }
    out.ok = true;
    return out;
}

// Reads <root>/<record_path> for the first root holding it.
inline RecordSystems ReadRecordSystems(const std::string& record_path, const std::vector<std::string>& roots) {
    for (const std::string& root : roots) {
        std::ifstream in(root + "/" + record_path, std::ios::binary);
        if (!in.good()) continue;
        std::ostringstream ss;
        ss << in.rdbuf();
        return ParseRecordSystems(ss.str(), root + "/" + record_path);
    }
    RecordSystems out;
    out.why = "record_missing";
    return out;
}

// (slot-16 WP-3 audit repair, findings 1 and 2) Does a re-run ideal's system (primes, a, index-aligned)
// carry the record ideal's split system?  "" when it does; "operator_set:record=[..],rerun=[..]" when a
// split prime is not a re-run operator prime (the slot-15 U_ell drop); "system_differs:at_<q>" when an
// eigenvalue differs.  Meaningful only for split_ok && r == 1.
inline std::string CompareSystem(const RecordIdeal& ri, const std::vector<uint64_t>& primes, const std::vector<uint32_t>& a) {
    for (size_t k = 0; k < ri.primes.size(); ++k) {
        size_t j = 0;
        while (j < primes.size() && primes[j] != ri.primes[k]) ++j;
        if (j == primes.size()) return "operator_set:record=" + ListText(ri.primes) + ",rerun=" + ListText(primes);
        if (j >= a.size() || k >= ri.a.size() || a[j] != ri.a[k]) return "system_differs:at_" + std::to_string(ri.primes[k]);
    }
    return std::string();
}

// (slot-16 WP-3 audit repair, finding 2) The U_ell operator of a re-run against what the row expects.
// With a readable record: U_ell (ell in operators_computed) must be in the re-run iff it is in the record.
// Without one: a row whose u_ell budget was set (budget_mb != 0: the np / npx sets, ext_u691, or an
// override) expects U_ell, and a re-run without it is refused -- so the fallback join cannot pair an id
// under a different operator set.  At budget 0 (`auto`) there is no expectation.  "" when consistent.
inline std::string CheckUEllOperator(const RecordSystems& rs, uint64_t ell, bool rerun_has_u, uint32_t budget_mb) {
    if (rs.ok) {
        bool rec_u = false;
        for (uint64_t q : rs.operators) rec_u = rec_u || q == ell;
        if (rec_u != rerun_has_u)
            return std::string("operator_set:u_ell_record=") + (rec_u ? "present" : "absent") + ",rerun=" +
                   (rerun_has_u ? "present" : "absent");
        return std::string();
    }
    if (budget_mb != 0 && !rerun_has_u)
        return "operator_set:u_ell_absent_under_budget_mb=" + std::to_string(budget_mb) + ",no_record";
    return std::string();
}

} // namespace census
} // namespace hecke_ext

#endif // HECKE_EXT_CENSUS_JOIN_H
