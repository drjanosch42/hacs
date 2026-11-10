// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// matrix-engine Phase 6 -- the campaign spec parser and planner.
// Grammar and semantics: CampaignSpec.h + docs/matrix-engine/campaign.md.

#include "CampaignSpec.h"

#include "../Hecke/LadicCell.h"
#include "../Hecke/HeckeModelDescriptor.h"

#include <algorithm>
#include <fstream>
#include <istream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace campaign {

namespace {

std::string Trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
    return s.substr(a, b - a);
}

// Strips a `#` / `//` comment, respecting neither quotes nor escapes (there
// are none in this grammar).
std::string StripComment(const std::string& s) {
    const size_t h = s.find('#');
    size_t c = s.find("//");
    size_t cut = std::string::npos;
    if (h != std::string::npos) cut = h;
    if (c != std::string::npos && (cut == std::string::npos || c < cut)) cut = c;
    return cut == std::string::npos ? s : s.substr(0, cut);
}

[[noreturn]] void Fail(const std::string& origin, uint32_t line, const std::string& msg) {
    throw std::runtime_error(origin + ":" + std::to_string(line) + ": " + msg);
}

std::vector<std::string> SplitCommas(const std::string& v) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : v) {
        if (ch == ',') { out.push_back(Trim(cur)); cur.clear(); }
        else cur.push_back(ch);
    }
    out.push_back(Trim(cur));
    // Drop trailing empties from "a, b," -- but keep a genuinely empty value.
    while (out.size() > 1 && out.back().empty()) out.pop_back();
    return out;
}

bool ParseU64(const std::string& tok, uint64_t& out) {
    if (tok.empty()) return false;
    uint64_t v = 0;
    for (char ch : tok) {
        if (ch < '0' || ch > '9') return false;
        const uint64_t d = static_cast<uint64_t>(ch - '0');
        if (v > (~0ull - d) / 10ull) return false;      // overflow
        v = v * 10ull + d;
    }
    out = v;
    return true;
}

// One list element: a literal `n` or a range `a..b [step c]`.
void ExpandElement(const std::string& origin, uint32_t line, const std::string& elem,
                   std::vector<uint64_t>& out) {
    const size_t dots = elem.find("..");
    if (dots == std::string::npos) {
        uint64_t v = 0;
        if (!ParseU64(elem, v)) Fail(origin, line, "not a non-negative integer: '" + elem + "'");
        out.push_back(v);
        return;
    }
    const std::string lo_s = Trim(elem.substr(0, dots));
    std::string rest = Trim(elem.substr(dots + 2));
    uint64_t step = 1;
    const size_t sp = rest.find("step");
    if (sp != std::string::npos) {
        const std::string step_s = Trim(rest.substr(sp + 4));
        if (!ParseU64(step_s, step) || step == 0)
            Fail(origin, line, "bad step in range '" + elem + "'");
        rest = Trim(rest.substr(0, sp));
    }
    uint64_t lo = 0, hi = 0;
    if (!ParseU64(lo_s, lo) || !ParseU64(rest, hi))
        Fail(origin, line, "bad range '" + elem + "' (expected a..b [step c])");
    if (hi < lo) Fail(origin, line, "empty range '" + elem + "' (b < a)");
    for (uint64_t v = lo; v <= hi; v += step) {
        out.push_back(v);
        if (v > ~0ull - step) break;   // guard the increment
    }
}

std::vector<uint64_t> ExpandList(const std::string& origin, uint32_t line, const std::string& value) {
    std::vector<uint64_t> out;
    for (const std::string& elem : SplitCommas(value)) {
        if (elem.empty()) Fail(origin, line, "empty list element");
        ExpandElement(origin, line, elem, out);
    }
    if (out.empty()) Fail(origin, line, "empty list");
    return out;
}

std::vector<uint32_t> To32(const std::string& origin, uint32_t line, const std::string& key,
                          const std::vector<uint64_t>& v) {
    std::vector<uint32_t> out;
    out.reserve(v.size());
    for (uint64_t x : v) {
        if (x > 0xffffffffull) Fail(origin, line, key + ": value out of 32-bit range");
        out.push_back(static_cast<uint32_t>(x));
    }
    return out;
}

bool ParseBool(const std::string& origin, uint32_t line, const std::string& v) {
    if (v == "true" || v == "1" || v == "yes" || v == "on") return true;
    if (v == "false" || v == "0" || v == "no" || v == "off") return false;
    Fail(origin, line, "expected true/false, got '" + v + "'");
}

// certify = full | sampled(<fraction>) | sampled(every <k>) | sampled(<k>)
CertifySpec ParseCertify(const std::string& origin, uint32_t line, const std::string& v) {
    CertifySpec c;
    c.text = v;
    if (v == "full") { c.mode = CertifyMode::Full; c.every_k = 1; return c; }
    // `none` == sampled with an EMPTY sample: the external cross-checks are
    // skipped on every job. (The in-process certificates C3-C6 still run --
    // they are not sampleable, see CampaignDriver.h.) This exists because
    // `sampled(every k)` always selects job 0, and a 1000-point campaign at
    // n_sym ~ 1000 cannot afford even one HeckeCrossChecks run.
    if (v == "none") { c.mode = CertifyMode::Sampled; c.every_k = 0; return c; }
    if (v.rfind("sampled", 0) != 0) Fail(origin, line, "certify: expected full|sampled(...)");
    const size_t lp = v.find('(');
    const size_t rp = v.rfind(')');
    if (lp == std::string::npos || rp == std::string::npos || rp < lp)
        Fail(origin, line, "certify: sampled needs a parenthesised argument");
    std::string arg = Trim(v.substr(lp + 1, rp - lp - 1));
    c.mode = CertifyMode::Sampled;
    if (arg.rfind("every", 0) == 0) arg = Trim(arg.substr(5));
    if (arg.find('.') != std::string::npos || arg.find('/') != std::string::npos) {
        // A fraction: "0.25" or "1/4". every_k = round(1 / fraction), >= 1.
        double frac = 0.0;
        const size_t slash = arg.find('/');
        if (slash != std::string::npos) {
            uint64_t a = 0, b = 0;
            if (!ParseU64(Trim(arg.substr(0, slash)), a) ||
                !ParseU64(Trim(arg.substr(slash + 1)), b) || b == 0)
                Fail(origin, line, "certify: bad fraction '" + arg + "'");
            frac = static_cast<double>(a) / static_cast<double>(b);
        } else {
            std::istringstream is(arg);
            if (!(is >> frac)) Fail(origin, line, "certify: bad fraction '" + arg + "'");
        }
        if (!(frac > 0.0) || frac > 1.0) Fail(origin, line, "certify: fraction must be in (0, 1]");
        const double k = 1.0 / frac;
        c.every_k = static_cast<uint32_t>(k + 0.5);
        if (c.every_k == 0) c.every_k = 1;
        return c;
    }
    uint64_t k = 0;
    if (!ParseU64(arg, k) || k == 0) Fail(origin, line, "certify: bad sample stride '" + arg + "'");
    if (k > 0xffffffffull) Fail(origin, line, "certify: sample stride out of range");
    c.every_k = static_cast<uint32_t>(k);
    return c;
}

// (slot-5 WP-2) `family = 2,3,5,7,ell` -- the good primes of S_0, with `ell`
// as a literal token.  The token is RECORDED, never resolved to a number: the
// admission of U_ell is the `u_ell` budget's decision ([X] Sec. 1.5) and a
// campaign key may not override a mathematical setting ([OD] R1).
void ParseExtFamily(const std::string& origin, uint32_t line, const std::string& value,
                    ExtSpec& ext) {
    ext.family_text = value;
    ext.good_primes.clear();
    ext.names_ell = false;
    for (const std::string& tok : SplitCommas(value)) {
        if (tok.empty()) Fail(origin, line, "family: empty list element");
        if (tok == "ell") { ext.names_ell = true; continue; }
        uint64_t v = 0;
        if (!ParseU64(tok, v) || v < 2)
            Fail(origin, line, "family: expected a prime >= 2 or the token 'ell', got '" +
                               tok + "'");
        bool seen = false;
        for (uint64_t x : ext.good_primes) if (x == v) seen = true;
        if (!seen) ext.good_primes.push_back(v);
    }
    std::sort(ext.good_primes.begin(), ext.good_primes.end());
    if (ext.good_primes.empty() && !ext.names_ell)
        Fail(origin, line, "family: empty operator set");
}

bool SectionEmpty(const SpecSection& s) {
    return s.n_sym.empty() && s.ell.empty() && s.p.empty() && s.r.empty() && s.q.empty() && s.level.empty() && s.algebra.empty();
}

} // namespace

// ---------------------------------------------------------------------------
// Job / ShapeSignature
// ---------------------------------------------------------------------------

// (slot-5 WP-2) [X] Sec. 1.5: the shape signature is extended by the family's
// prime set -- and by nothing else, because `sign` and `ext_groups` select
// which objects the CELL's record carries and never a matrix shape.  Empty
// when `ext` is false, which is what makes acceptance (a) a property of this
// one function.
std::string ExtSpec::signature_tag() const {
    if (!ext) return std::string();
    std::string t = "S0=";
    for (size_t i = 0; i < good_primes.size(); ++i) {
        if (i) t += ",";
        t += std::to_string(good_primes[i]);
    }
    if (good_primes.empty()) t += "default";
    if (names_ell) t += ",ell";
    return t;
}

std::vector<uint32_t> ModelDefaultDegrees(const std::string& model) {
    std::vector<uint32_t> out;
    const hecke::SimplicialModelDescriptor* d = hecke::FindModelDescriptor(model);
    if (d == nullptr) return out;
    for (uint32_t q = 0; q < d->num_degrees(); ++q) out.push_back(q);
    return out;
}

std::string Job::degree_tag() const {
    if (degrees.empty()) return "none";
    std::string t;
    for (size_t i = 0; i < degrees.size(); ++i) {
        if (i) t += "-";
        t += std::to_string(degrees[i]);
    }
    return t;
}

bool Job::degrees_are_model_default() const {
    const std::vector<uint32_t> def = ModelDefaultDegrees(model);
    return !def.empty() && def == degrees;
}

// The id must be INJECTIVE on (n_sym, ell, p, r, degrees) -- the very tuple
// plan_campaign() de-duplicates by -- because it is the output file's stem AND
// the resume key.  It is BACKWARD COMPATIBLE: the suffix is emitted only for a
// degree list that is not the model's default, so every id ever written by a
// spec without a `q` key is unchanged.
//
// Injectivity of the encoding: the prefix fields are tagged ("n", "_ell", "_p",
// "_r") and `_r<digits>` is the last of them, so the optional "_q..." tail can
// never be confused with any prefix field; `degrees` is sorted and duplicate-free
// by construction (plan_campaign), so its "-"-joined tag is a canonical form of
// the set; and the empty tail denotes exactly one degree list, the model default.
// (slot-4 WP-1 / E2a) The FULL 64-bit settings fingerprint, 16 lower-case hex
// digits -- never a truncation, because the id must stay injective on
// everything that can change a job's value.
static std::string SettingsTag(uint64_t fingerprint) {
    static const char* kHex = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 0; i < 16; ++i) out[15 - i] = kHex[(fingerprint >> (4 * i)) & 0xfu];
    return out;
}

std::string Job::id() const {
    std::string s = "n" + std::to_string(n_sym) + "_ell" + std::to_string(ell) + "_p" +
                    std::to_string(p) + "_r" + std::to_string(r);
    if (!degrees_are_model_default()) s += "_q" + degree_tag();
    if (settings_fingerprint != 0) s += "_s" + SettingsTag(settings_fingerprint);
    return s;
}

std::string Job::to_string() const {
    std::string s = "n_sym=" + std::to_string(n_sym) + " ell=" + std::to_string(ell) + " p=" +
                    std::to_string(p) + " r=" + std::to_string(r);
    if (!degrees_are_model_default()) s += " q=" + degree_tag();
    if (settings_fingerprint != 0) s += " s=" + SettingsTag(settings_fingerprint);
    return s;
}

ShapeSignature Job::shape() const {
    ShapeSignature sig;
    sig.n_sym = n_sym;
    sig.p = p;
    sig.r = r;
    sig.degrees = degrees;
    sig.model = model;
    sig.settings_fingerprint = settings_fingerprint;
    sig.ext_tag = ext_tag;
    return sig;
}

std::string Job::golden_point_ne() const {
    return "n=" + std::to_string(n_sym) + " ell=" + std::to_string(ell);
}

std::string Job::golden_point_nelpr() const {
    return golden_point_ne() + " p=" + std::to_string(p) + " r=" + std::to_string(r);
}

bool ShapeSignature::operator==(const ShapeSignature& o) const {
    return n_sym == o.n_sym && p == o.p && r == o.r && degrees == o.degrees &&
           model == o.model && settings_fingerprint == o.settings_fingerprint &&
           ext_tag == o.ext_tag;
}

bool ShapeSignature::operator<(const ShapeSignature& o) const {
    if (n_sym != o.n_sym) return n_sym < o.n_sym;
    if (p != o.p) return p < o.p;
    if (r != o.r) return r < o.r;
    if (degrees != o.degrees) return degrees < o.degrees;
    if (model != o.model) return model < o.model;
    // Appended LAST, so a campaign at the defaults (fingerprint 0 everywhere)
    // orders exactly as before.
    if (settings_fingerprint != o.settings_fingerprint)
        return settings_fingerprint < o.settings_fingerprint;
    // (slot-5 WP-2) A campaign-wide constant, empty when `ext` is false.
    return ext_tag < o.ext_tag;
}

std::string ShapeSignature::to_string() const {
    std::ostringstream oss;
    oss << "n_sym=" << n_sym << " p=" << p << " r=" << r << " q=";
    for (size_t i = 0; i < degrees.size(); ++i) { if (i) oss << ","; oss << degrees[i]; }
    oss << " model=" << model;
    // The `_q` / ` q=` pattern: emitted only when it is not the default, so
    // every signature string a default campaign writes is byte-unchanged.
    if (settings_fingerprint != 0) oss << " s=" << SettingsTag(settings_fingerprint);
    // (slot-5 WP-2) [X] Sec. 1.5, emitted only when the ext engine runs.
    if (!ext_tag.empty()) oss << " ext=" << ext_tag;
    return oss.str();
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

bool ParseDescentParts(const std::string& text, DescentParts& out) {
    // one of three words: blanks are tolerated around the value and around the comma, never inside a word ("gro ups" is refused)
    const auto trim = [](const std::string& x) {
        size_t b = x.find_first_not_of(" \t"), e = x.find_last_not_of(" \t");
        return b == std::string::npos ? std::string() : x.substr(b, e - b + 1);
    };
    std::string t;
    const size_t comma = text.find(',');
    if (comma == std::string::npos) t = trim(text);
    else t = trim(text.substr(0, comma)) + "," + trim(text.substr(comma + 1));
    if (t == "all") out = DescentParts::All;
    else if (t == "groups,hecke") out = DescentParts::GroupsHecke;
    else if (t == "groups") out = DescentParts::Groups;
    else return false;
    return true;
}

const char* DescentPartsText(DescentParts p) {
    return p == DescentParts::All ? "all" : p == DescentParts::GroupsHecke ? "groups,hecke" : "groups";
}

CampaignSpec ParseCampaignSpec(std::istream& in, const std::string& origin) {
    CampaignSpec spec;
    spec.origin = origin;
    SpecSection cur;
    cur.first_line = 1;
    std::string raw;
    uint32_t line = 0;
    bool saw_any_grid = false;
    uint32_t gpu_cells_line = 0;   // (slot-8 WP-6b) for the ext_gpu_cells check

    auto flush_section = [&]() {
        if (!SectionEmpty(cur)) spec.sections.push_back(cur);
        cur = SpecSection();
        cur.first_line = line + 1;
    };

    while (std::getline(in, raw)) {
        ++line;
        const std::string s = Trim(StripComment(raw));
        if (s.empty()) continue;
        if (s == "---") { flush_section(); continue; }

        const size_t eq = s.find('=');
        if (eq == std::string::npos)
            Fail(origin, line, "expected 'key = value', '---' or a comment, got '" + s + "'");
        const std::string key = Trim(s.substr(0, eq));
        const std::string value = Trim(s.substr(eq + 1));
        if (key.empty()) Fail(origin, line, "empty key");
        if (value.empty()) Fail(origin, line, "empty value for key '" + key + "'");

        if (key == "n_sym" || key == "n") {
            cur.n_sym = To32(origin, line, key, ExpandList(origin, line, value));
            saw_any_grid = true;
        } else if (key == "ell") {
            cur.ell = ExpandList(origin, line, value);
            saw_any_grid = true;
        } else if (key == "p" || key == "p_hecke") {
            cur.p = ExpandList(origin, line, value);
            saw_any_grid = true;
        } else if (key == "r") {
            cur.r = To32(origin, line, key, ExpandList(origin, line, value));
            saw_any_grid = true;
        } else if (key == "q" || key == "degrees") {
            cur.q = To32(origin, line, key, ExpandList(origin, line, value));
            saw_any_grid = true;
        } else if (key == "level") {
            // (slot-24 D6) the descent grid key; admitted only under coefficient_ring = Z (checked below).
            cur.level = To32(origin, line, key, ExpandList(origin, line, value));
            saw_any_grid = true;
        } else if (key == "name") {
            spec.name = value;
        } else if (key == "output" || key == "output_dir") {
            spec.output_dir = value;
        } else if (key == "model") {
            if (hecke::FindModelDescriptor(value) == nullptr)
                Fail(origin, line, "unknown model '" + value + "' (known: sl2z_7vertex)");
            spec.model = value;
        } else if (key == "resume") {
            spec.resume = ParseBool(origin, line, value);
        } else if (key == "lean" || key == "memory_lean") {
            spec.lean = ParseBool(origin, line, value);
        } else if (key == "threads") {
            uint64_t t = 0;
            if (!ParseU64(value, t) || t > 4096) Fail(origin, line, "threads: bad value");
            spec.threads = static_cast<uint32_t>(t);   // 0 = inherit the process pool
        } else if (key == "jobs" || key == "concurrent_jobs") {
            uint64_t t = 0;
            if (!ParseU64(value, t) || t == 0 || t > 4096) Fail(origin, line, "jobs: bad value");
            spec.jobs = static_cast<uint32_t>(t);
        } else if (key == "structure") {
            // (slot-25 HZ-REC) the record schema of a coefficient_ring = Z campaign (checked below)
            if (value == "hecke") spec.structure_hecke = true;
            else if (value != "none") Fail(origin, line, "structure: unknown value '" + value + "' (known: none, hecke)");
            spec.structure_line = line;
        } else if (key == "hz_cert") {
            spec.hz_cert = ParseBool(origin, line, value);
            spec.hz_cert_line = line;
        } else if (key == "hz_g8") {
            if (value != "off" && value != "hida") Fail(origin, line, "hz_g8: unknown value '" + value + "' (known: off, hida)");
            spec.hz_g8 = value;
            spec.hz_g8_line = line;
        } else if (key == "algebra") {
            // (slot-25 HZ-REC) the grid key of a structure = hecke campaign (checked below)
            cur.algebra.clear();
            std::stringstream ss(value);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                tok = Trim(tok);
                if (tok != "full" && tok != "anemic") Fail(origin, line, "algebra: unknown value '" + tok + "' (known: full, anemic)");
                if (std::find(cur.algebra.begin(), cur.algebra.end(), tok) == cur.algebra.end()) cur.algebra.push_back(tok);
            }
            cur.algebra_line = line;
            saw_any_grid = true;
        } else if (key == "parts") {
            // (slot-24 ICP2) the descent campaign's record parts; admitted only under coefficient_ring = Z (checked below)
            if (!ParseDescentParts(value, spec.descent_parts))
                Fail(origin, line, "parts: unknown value '" + value + "' (known: groups, groups,hecke, all)");
            spec.descent_parts_line = line;
        } else if (key == "ext_jobs") {
            // (slot-8 WP-6) a scheduling knob: cells of the ext pass at once.
            uint64_t t = 0;
            if (!ParseU64(value, t) || t == 0 || t > 4096) Fail(origin, line, "ext_jobs: bad value");
            spec.ext_jobs = static_cast<uint32_t>(t);
        } else if (key == "ext_memory_mb") {
            // (slot-8 WP-6) a scheduling knob: the admission cap, in MB.
            uint64_t t = 0;
            if (!ParseU64(value, t) || t == 0) Fail(origin, line, "ext_memory_mb: bad value");
            spec.ext_memory_mb = t;
        } else if (key == "ext_gpu") {
            // (slot-8 WP-6b) a scheduling knob: the optional device worker.
            if (value == "off") spec.ext_gpu = ExtGpuMode::Off;
            else if (value == "on") spec.ext_gpu = ExtGpuMode::On;
            else if (value == "force") spec.ext_gpu = ExtGpuMode::Force;
            else Fail(origin, line, "ext_gpu: expected off|on|force");
        } else if (key == "ext_gpu_cells") {
            // (slot-8 WP-6b) a scheduling knob: `n:ell` pairs, separated by
            // commas and/or blanks.
            spec.ext_gpu_cells.clear();
            std::string tok;
            auto flush = [&]() {
                if (tok.empty()) return;
                const size_t c = tok.find(':');
                uint64_t n = 0, e = 0;
                if (c == std::string::npos || !ParseU64(tok.substr(0, c), n) ||
                    !ParseU64(tok.substr(c + 1), e) || n > 0xffffffffull || e < 2)
                    Fail(origin, line, "ext_gpu_cells: expected 'n_sym:ell' pairs, got '" + tok + "'");
                spec.ext_gpu_cells.emplace_back(static_cast<uint32_t>(n), e);
                tok.clear();
            };
            for (char ch : value) {
                if (ch == ',' || ch == ' ' || ch == '\t') flush();
                else tok.push_back(ch);
            }
            flush();
            if (spec.ext_gpu_cells.empty()) Fail(origin, line, "ext_gpu_cells: empty list");
            gpu_cells_line = line;
        } else if (key == "certify") {
            spec.certify = ParseCertify(origin, line, value);
        } else if (key == "level_a") {
            if (value == "auto") spec.level_a = LevelAMode::Auto;
            else if (value == "always") spec.level_a = LevelAMode::Always;
            else if (value == "never" || value == "off") spec.level_a = LevelAMode::Never;
            else Fail(origin, line, "level_a: expected auto|always|never");
        } else if (key == "level_a_max_entries") {
            if (!ParseU64(value, spec.level_a_max_entries))
                Fail(origin, line, "level_a_max_entries: bad value");
        } else if (key == "theta_rank_max_dim") {
            uint64_t v = 0;
            if (!ParseU64(value, v) || v > 0xffffffffull)
                Fail(origin, line, "theta_rank_max_dim: bad value");
            spec.theta_rank_max_dim = static_cast<uint32_t>(v);
        } else if (key == "ext") {
            // (slot-5 WP-2 / E2 cont.) [X] Sec. 1.5.  Default false, and with
            // it false NOTHING below is consulted.
            spec.ext.ext = ParseBool(origin, line, value);
        } else if (key == "family") {
            ParseExtFamily(origin, line, value, spec.ext);
        } else if (key == "sign") {
            if (value == "all") spec.ext.sign = ExtSign::All;
            else if (value == "plus") spec.ext.sign = ExtSign::Plus;
            else if (value == "minus") spec.ext.sign = ExtSign::Minus;
            else Fail(origin, line, "sign: expected all|plus|minus");
            spec.ext.sign_text = value;
        } else if (key == "ext_groups") {
            if (value == "bulk") spec.ext.groups = ExtGroups::Bulk;
            else if (value == "bulk+bd") spec.ext.groups = ExtGroups::BulkBd;
            else if (value == "all") spec.ext.groups = ExtGroups::All;
            else Fail(origin, line, "ext_groups: expected bulk|bulk+bd|all");
            spec.ext.groups_text = value;
        } else if (key == "orbifold") {
            // (slot-22 L23-1) the orbifold section (CampaignSpec.h OrbifoldSpec,
            // Campaign/CampaignOrbifold.h): `c_i` or `off`; the model is checked here,
            // everything else after the sections are complete (ValidateOrbifoldSpec).
            if (value == "c_i") spec.orbifold.enabled = true;
            else if (value == "off" || value == "false") spec.orbifold.enabled = false;
            else Fail(origin, line, "orbifold: expected c_i|off (the only model of this build is c_i)");
            spec.orbifold.mode_text = value;
            spec.orbifold.line = line;
        } else if (key == "orbifold_primes") {
            // (slot-22 L23-1) the T_p of an orbifold cell: primes < 2^16, ascending, duplicate-free.
            std::vector<uint32_t> ps = To32(origin, line, key, ExpandList(origin, line, value));
            for (uint32_t x : ps) {
                bool prime = x >= 2 && x < 65536;
                for (uint32_t d = 2; prime && d * d <= x; ++d) prime = (x % d) != 0;
                if (!prime) Fail(origin, line, "orbifold_primes: " + std::to_string(x) + " is not a prime < 2^16");
            }
            std::sort(ps.begin(), ps.end());
            ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
            spec.orbifold.primes = ps;
            spec.orbifold.primes_default = (ps == std::vector<uint32_t>{2, 3, 5, 7});
        } else if (key == "orbifold_m_target") {
            // (slot-22 L23-1) `auto` (default) or the certified ell-adic precision every cell must reach.
            uint64_t t = 0;
            if (value == "auto") spec.orbifold.m_target = 0;
            else if (ParseU64(value, t) && t >= 1 && t <= 1024) spec.orbifold.m_target = static_cast<uint32_t>(t);
            else Fail(origin, line, "orbifold_m_target: expected auto or an integer in [1, 1024]");
        } else if (key == "orbifold_k_cap") {
            // (slot-22 L23-1) the largest working precision K the reader may climb to (0 = its instantiated range).
            uint64_t t = 0;
            if (!ParseU64(value, t) || t > 1024) Fail(origin, line, "orbifold_k_cap: expected an integer in [0, 1024]");
            spec.orbifold.k_cap = static_cast<uint32_t>(t);
        } else if (hecke::MathSettings::known_key(key)) {
            // (slot-4 WP-1 / E2a) A registered MATHEMATICAL key ([X] Sec. 1.5).
            // The engine owns the defaults and the validation; the spec only
            // sets, and a bad value becomes an ordinary spec error with the
            // engine's own message ([OD] R1: validated once, never twice).
            try {
                spec.math.set(key, value, hecke::SettingSource::Spec);
            } catch (const std::invalid_argument& e) {
                Fail(origin, line, std::string(e.what()));
            }
        } else {
            Fail(origin, line, "unknown key '" + key + "'");
        }
    }
    flush_section();

    // (slot-8 WP-3 -> slot-9 WP-8, C5c) [E] sec:ring (b4), (g): the JOB LOOP
    // honours `coefficient_ring = Z/ell^<k>` (hecke_ring_report/v1,
    // Campaign/CampaignRing.h); the EXT PASS stays F_ell-only and is refused
    // HERE, at spec validation, by name and before any job runs -- never after
    // a job has run (`Z` / `Z/<m>` already raised in MathSettings::set()).  The
    // grid is checked below, once the sections are complete.
    if (spec.math.coefficient_ring.k != 1 && spec.ext.ext) {
        hecke::RefuseCoefficientRing("campaign spec " + origin, spec.math.coefficient_ring.k,
                                     " for the ext pass (`ext = true`: the extension record is F_ell-only, [E] "
                                     "sec:ring (b4); the job loop honours the ring without `ext`)");
    }

    if (!spec.ext_gpu_cells.empty() && spec.ext_gpu == ExtGpuMode::Off)
        Fail(origin, gpu_cells_line, "ext_gpu_cells: needs ext_gpu = on (or force)");
    if (spec.sections.empty()) {
        if (saw_any_grid) Fail(origin, line, "internal: grid keys seen but no section produced");
        throw std::runtime_error(origin + ": no jobs -- the spec declares no grid keys");
    }
    // (slot-24 D6, DESCENT_plan.md Sec. 4 D6) `coefficient_ring = Z`: the campaign is a campaign of descent cells
    // (integral_descent_record/v1, Campaign/CampaignDescent.h), one per (level, n_sym); its own grid rules and refusals,
    // before any cell runs.  `level` anywhere else is a spec error, so no other campaign is affected.
    if (spec.math.coefficient_ring.is_integral()) {
        // (slot-25 HZ-REC) `structure = hecke`: integral_hecke_structure/v1 cells (Campaign/CampaignHeckeZ.h), their own rules
        if (spec.structure_hecke) {
            ValidateHeckeZSpec(spec);
            return spec;
        }
        if (spec.hz_cert_line != 0) Fail(origin, spec.hz_cert_line, "hz_cert: a structure = hecke key");
        if (spec.hz_g8_line != 0) Fail(origin, spec.hz_g8_line, "hz_g8: a structure = hecke key");
        for (const SpecSection& sec : spec.sections)
            if (!sec.algebra.empty()) Fail(origin, sec.algebra_line, "algebra: a structure = hecke grid key");
        ValidateDescentSpec(spec);
        return spec;
    }
    if (spec.structure_line != 0) Fail(origin, spec.structure_line, "structure: a coefficient_ring = Z key");
    if (spec.hz_cert_line != 0) Fail(origin, spec.hz_cert_line, "hz_cert: a structure = hecke key");
    if (spec.hz_g8_line != 0) Fail(origin, spec.hz_g8_line, "hz_g8: a structure = hecke key");
    for (const SpecSection& sec : spec.sections)
        if (!sec.algebra.empty()) Fail(origin, sec.algebra_line, "algebra: a structure = hecke grid key, admitted only under coefficient_ring = Z");
    if (spec.descent_parts_line != 0)
        Fail(origin, spec.descent_parts_line, "parts: a descent key, admitted only under coefficient_ring = Z");
    for (const SpecSection& sec : spec.sections)
        if (!sec.level.empty())
            Fail(origin, sec.first_line, "level: a descent grid key, admitted only under coefficient_ring = Z");
    // Every section must pin all four grid axes; `q` may be omitted.
    for (const SpecSection& sec : spec.sections) {
        const uint32_t ln = sec.first_line;
        if (sec.n_sym.empty()) Fail(origin, ln, "section is missing 'n_sym'");
        if (sec.ell.empty())   Fail(origin, ln, "section is missing 'ell'");
        // (slot-22 L23-1) an orbifold campaign is per (n_sym, ell) cell: no p / r / q axis
        // (its T_p are `orbifold_primes`); ValidateOrbifoldSpec refuses them.
        if (sec.p.empty() && !spec.orbifold.enabled) Fail(origin, ln, "section is missing 'p'");
        if (sec.r.empty() && !spec.orbifold.enabled) Fail(origin, ln, "section is missing 'r'");
        for (uint64_t e : sec.ell) if (e < 2) Fail(origin, ln, "ell must be a prime >= 2");
        for (uint64_t x : sec.p) if (x < 2) Fail(origin, ln, "p must be a prime >= 2");
        for (uint32_t x : sec.r) if (x == 0) Fail(origin, ln, "r must be >= 1");
    }
    // (slot-22 L23-1) The orbifold section's own refusals, by name and before any cell runs
    // (Campaign/CampaignOrbifold.cpp): ell outside {2, 3}, a p / r / q axis, `ext`, a
    // non-default coefficient_ring, jobs > 1, a cell beyond the reader's precision range.
    if (spec.orbifold.enabled) ValidateOrbifoldSpec(spec);
    // (slot-9 WP-8) [E] sec:ring (b4), (g): under Z/ell^k a grid cell outside
    // what the build executes is refused here, naming the cell, so no job of
    // the campaign has run when it is.  (slot-11 WP-18c; [E] v1.12.0-pre.2
    // sec:ring, the per-cell rule) ell^k < 2^64 is the native route as
    // before; ell^k >= 2^64 at ell >= 5 is the DIGIT route (Hecke/LadicCell),
    // which needs n > 0 even; ell in {2, 3} beyond 2^64, the 8192-bit cap and
    // the memory cap are refused by name.
    if (spec.math.coefficient_ring.k >= 2) {
        const uint32_t k = spec.math.coefficient_ring.k;
        for (const SpecSection& sec : spec.sections)
            for (uint64_t e : sec.ell) {
                if (hecke::ladic_cell::ClassifyCell(e, k) == hecke::ladic_cell::CellRoute::Native) continue;
                for (uint32_t n : sec.n_sym) {
                    try {
                        // (WP-14 pass 2 m-3) at the largest working precision the run can
                        // reach, K = k + e_max <= k + K_w - 1, so no job refuses inside the run.
                        hecke::ladic_cell::RequireDigitCell("campaign spec " + origin, n, e,
                                                            k + hecke::ladic_cell::WordLevelMax(e) - 1,
                                                            hecke::ladic_cell::EngineLadicWidth());
                    } catch (const hecke::NotImplementedInThisBuild& ex) {
                        // The route's own refusal (it names the spec, the ring and the
                        // cell), with the section's line appended.
                        std::string m = ex.what();
                        const std::string prefix = "not_implemented_in_this_build: ";
                        if (m.compare(0, prefix.size(), prefix) == 0) m = m.substr(prefix.size());
                        throw hecke::NotImplementedInThisBuild(m + " -- the grid cell of the section at line " +
                                                               std::to_string(sec.first_line));
                    }
                }
            }
    }
    return spec;
}

CampaignSpec ParseCampaignSpecFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("campaign: cannot open spec file '" + path + "'");
    return ParseCampaignSpec(in, path);
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

std::vector<JobGroup> plan_campaign(const CampaignSpec& spec) {
    const hecke::SimplicialModelDescriptor* desc = hecke::FindModelDescriptor(spec.model);
    if (desc == nullptr) throw std::runtime_error("campaign: unknown model '" + spec.model + "'");

    std::vector<uint32_t> all_degrees;
    for (uint32_t q = 0; q < desc->num_degrees(); ++q) all_degrees.push_back(q);

    // One fingerprint for the whole campaign: the mathematical settings are
    // campaign-wide keys ([X] Sec. 1.5).
    const uint64_t fingerprint = spec.math.fingerprint();

    // (n_sym, ell, p, r, degrees, model) -> first occurrence wins.
    using Key = std::tuple<uint32_t, uint64_t, uint64_t, uint32_t, std::vector<uint32_t>>;
    std::set<Key> seen;
    std::map<ShapeSignature, std::vector<Job>> by_sig;

    for (const SpecSection& sec : spec.sections) {
        std::vector<uint32_t> degrees = sec.q.empty() ? all_degrees : sec.q;
        std::sort(degrees.begin(), degrees.end());
        degrees.erase(std::unique(degrees.begin(), degrees.end()), degrees.end());
        for (uint32_t q : degrees) {
            if (q >= desc->num_degrees())
                throw std::runtime_error("campaign: degree q=" + std::to_string(q) +
                                         " is out of range for model " + desc->name);
        }
        for (uint32_t n : sec.n_sym)
            for (uint64_t e : sec.ell)
                for (uint64_t pp : sec.p)
                    for (uint32_t rr : sec.r) {
                        const Key k{n, e, pp, rr, degrees};
                        if (!seen.insert(k).second) continue;
                        Job j;
                        j.n_sym = n; j.ell = e; j.p = pp; j.r = rr;
                        j.degrees = degrees;
                        j.model = spec.model;
                        // (slot-4 WP-1 / E2a) 0 at the defaults, so every id,
                        // shape signature and output path of a campaign that
                        // sets no mathematical key is byte-unchanged.
                        j.settings_fingerprint = fingerprint;
                        // (slot-5 WP-2) A campaign-wide constant, empty when
                        // `ext` is false, so a default campaign's shape
                        // signatures and grouping are byte-unchanged.
                        j.ext_tag = spec.ext.signature_tag();
                        by_sig[j.shape()].push_back(j);
                    }
    }

    std::vector<JobGroup> groups;
    groups.reserve(by_sig.size());
    uint64_t idx = 0;
    for (auto& kv : by_sig) {
        JobGroup g;
        g.sig = kv.first;
        g.jobs = std::move(kv.second);
        std::sort(g.jobs.begin(), g.jobs.end(),
                  [](const Job& a, const Job& b) { return a.ell < b.ell; });
        for (Job& j : g.jobs) {
            j.index = idx++;
            j.external_certify = spec.certify.selects(j.index);
        }
        groups.push_back(std::move(g));
    }

    // (slot-5 WP-2 / E2 cont.) THE PER-CELL GROUPING of [X] Sec. 1.5 / [XA
    // m14].  The ext record is per (n_sym, ell) CELL; a job is per
    // (n_sym, ell, p, r).  Two passes over the plan's EXECUTION order (group
    // order, then ascending ell inside a group -- exactly what flatten()
    // yields and what `index` counts), so the assignment is a pure function of
    // the spec and cannot depend on the `jobs x threads` split:
    //
    //   pass 1  collect each cell's Hecke primes, in first-seen order;
    //   pass 2  mark the cell's FIRST job as the owner and give every job of
    //           the cell the same family, `spec.family` u {those primes},
    //           ascending and duplicate-free.
    //
    // With `ext = false` both passes still run but write only the defaults
    // (empty family, no owner), so nothing observable moves.
    if (spec.ext.ext) {
        using Cell = std::pair<uint32_t, uint64_t>;
        std::map<Cell, std::vector<uint64_t>> primes_of;
        std::vector<Cell> cell_order;
        for (const JobGroup& g : groups)
            for (const Job& j : g.jobs) {
                const Cell c{j.n_sym, j.ell};
                auto it = primes_of.find(c);
                if (it == primes_of.end()) {
                    primes_of.emplace(c, std::vector<uint64_t>{j.p});
                    cell_order.push_back(c);
                } else {
                    bool seen = false;
                    for (uint64_t x : it->second) if (x == j.p) seen = true;
                    if (!seen) it->second.push_back(j.p);
                }
            }
        for (Cell& c : cell_order) {
            std::vector<uint64_t>& fam = primes_of[c];
            for (uint64_t x : spec.ext.good_primes) {
                bool seen = false;
                for (uint64_t y : fam) if (y == x) seen = true;
                if (!seen) fam.push_back(x);
            }
            std::sort(fam.begin(), fam.end());
        }
        std::set<Cell> owned;
        for (JobGroup& g : groups)
            for (Job& j : g.jobs) {
                const Cell c{j.n_sym, j.ell};
                j.ext_family = primes_of[c];
                j.ext_owner = owned.insert(c).second;
            }
    }
    return groups;
}

std::vector<Job> flatten(const std::vector<JobGroup>& groups) {
    std::vector<Job> out;
    for (const JobGroup& g : groups)
        out.insert(out.end(), g.jobs.begin(), g.jobs.end());
    return out;
}

} // namespace campaign
