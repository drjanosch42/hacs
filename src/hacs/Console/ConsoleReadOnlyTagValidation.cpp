// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleReadOnlyTagValidation.cpp -- see ConsoleReadOnlyTagValidation.h.
// ===========================================================================

#include "ConsoleReadOnlyTagValidation.h"

#include "../BuildSwitches.h"

#if OC_BUILD_NOTEBOOK
#include "Bindings/PrintJson.h"
#include "Bindings/ReadOnlyTags.h"
#include "Bindings/RegisterPipelineVerbs.h"
#include "Bindings/Serve.h"
#include "Bindings/ServeJson.h"
#include "Core/Object.h"
#include "Core/Registry.h"
#include "Core/ScriptRunner.h"
#include "Core/Session.h"
#include "Core/Settings.h"
#include "Core/Value.h"

#include "../Hecke/HeckeSettings.h"
#include "../HeckeExt/RegisterHeckeExtensionVerbs.h"
#include "../Util/TextIo.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>
#endif

#if OC_BUILD_NOTEBOOK
namespace {

using namespace console;
using serve_json::Json;

uint32_t g_checks = 0;
uint32_t g_fail = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 60) std::cout << "  [FAIL] " << what << "\n";
    }
}

using oc_util::ReadFile;   // S24 CL-7 (F11): Util/TextIo.h
std::vector<std::string> Lines(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream in(s);
    std::string line;
    while (std::getline(in, line)) out.push_back(line);
    return out;
}

bool PathExists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

std::string FindRepoRoot() {
    const char* marker = "/src/hacs/Console/serve/read_only_verbs.tsv";
#ifdef OC_SOURCE_DIR
    {
        const std::string r(OC_SOURCE_DIR);
        if (!r.empty() && PathExists(r + marker)) return r;
    }
#endif
    char buf[PATH_MAX];
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (PathExists(dir + marker)) return dir;
            const std::size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir = dir.substr(0, slash);
        }
    }
    return std::string();
}

struct FullRegistry {
    Registry registry;
    std::ostringstream out;
    Session session;
    FullRegistry() : session(registry, /*verbose_default=*/false) {
        RegisterPipelineVerbs(registry);
        hecke_ext::RegisterHeckeExtensionVerbs(registry);
        registry.install_settings(session.settings());
        session.settings().set_baseline(session.settings().snapshot());
        session.set_output(out);
    }
};

// ---------------------------------------------------------------------------
// (A) registry <-> fixture
// ---------------------------------------------------------------------------
void ArmA(const Registry& reg, const std::string& root, std::size_t& rows_out) {
    std::string text;
    const std::string rel = "src/hacs/Console/serve/read_only_verbs.tsv";
    if (!ReadFile(root + "/" + rel, text)) {
        Check(false, "(A) " + rel + " is readable");
        return;
    }
    std::map<std::pair<std::string, std::string>, bool> fixture;
    std::vector<std::pair<std::string, std::string>> order;
    for (const std::string& line : Lines(text)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> f;
        std::string cur;
        for (char c : line) {
            if (c == '\t') { f.push_back(cur); cur.clear(); }
            else cur += c;
        }
        f.push_back(cur);
        if (f.size() != 4 || (f[2] != "true" && f[2] != "false")) {
            Check(false, "(A) malformed fixture row: '" + line.substr(0, 120) + "'");
            continue;
        }
        if (f[2] == "true") Check(!f[3].empty(), "(A) the true row " + f[0] + " | " + f[1] + " carries an audit text");
        const auto key = std::make_pair(f[0], f[1]);
        Check(!fixture.count(key), "(A) the fixture lists " + f[0] + " | " + f[1] + " once");
        fixture[key] = (f[2] == "true");
        order.push_back(key);
    }
    rows_out = order.size();
    std::vector<std::pair<std::string, std::string>> reg_order;
    for (const std::string& name : reg.verb_names()) {
        const VerbEntry* v = reg.find_verb(name);
        if (v == nullptr) continue;
        for (const Overload& o : v->overloads) {
            const std::string sig = o.sig.to_string(name);
            reg_order.push_back({name, sig});
            const auto it = fixture.find({name, sig});
            if (it == fixture.end()) {
                Check(false, "(A) the registered overload " + name + " | " + sig +
                                 " has no row in Console/serve/read_only_verbs.tsv (a new verb is excluded until audited)");
                continue;
            }
            const serve_tags::TagState s = serve_tags::LookupTag(*v, o);
            const bool reg_ro = (s == serve_tags::TagState::ReadOnly);
            Check(reg_ro == it->second, "(A) the tag of " + name + " | " + sig + " is " +
                                            (reg_ro ? "true" : "false") + " in the registry and " +
                                            (it->second ? "true" : "false") + " in the fixture");
        }
    }
    std::set<std::pair<std::string, std::string>> reg_set(reg_order.begin(), reg_order.end());
    for (const auto& k : order)
        Check(reg_set.count(k) != 0, "(A) the fixture row " + k.first + " | " + k.second + " matches a registered overload");
    Check(order == reg_order, "(A) the fixture rows are in registration order");
}

// ---------------------------------------------------------------------------
// (C1) the source audit
// ---------------------------------------------------------------------------
std::vector<std::string> RegistrationTUs(const std::string& root) {
    std::vector<std::string> out = {"src/hacs/Console/Core/CoreVerbs.cpp"};
    // (S24 CL-8) HeckeExt/RegisterHeckeExtensionVerbs.cpp is split by area: every RegisterHeckeExtensionVerbs*.cpp.
    const std::string ext = "src/hacs/HeckeExt";
    const std::string stem = "RegisterHeckeExtensionVerbs";
    std::vector<std::string> x;
    if (DIR* d = ::opendir((root + "/" + ext).c_str())) {
        while (struct dirent* e = ::readdir(d)) {
            const std::string n = e->d_name;
            if (n.compare(0, stem.size(), stem) == 0 && n.size() > 4 && n.compare(n.size() - 4, 4, ".cpp") == 0)
                x.push_back(ext + "/" + n);
        }
        ::closedir(d);
    }
    std::sort(x.begin(), x.end());
    out.insert(out.end(), x.begin(), x.end());
    const std::string dir = "src/hacs/Console/Bindings";
    std::vector<std::string> b;
    if (DIR* d = ::opendir((root + "/" + dir).c_str())) {
        while (struct dirent* e = ::readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() > 4 && n.compare(n.size() - 4, 4, ".cpp") == 0) b.push_back(dir + "/" + n);
        }
        ::closedir(d);
    }
    std::sort(b.begin(), b.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

void ArmC1(const std::string& root, std::size_t& hits_out, std::size_t& files_out) {
    const std::regex writer(
        R"re(ofstream|\bfopen\(|(^|[^.\w])open\(|\.open\(|filesystem::(create_director|remove|rename|copy|resize_file|permissions)|\bfs::(create_director|remove|rename|copy|resize_file|permissions)|\bgetenv\(|(^|[^.\w])system\(|\bpopen\(|\bfreopen\(|\bmkdir\(|\bunlink\(|\bexecv|\bfork\()re");
    std::string allow_text;
    const std::string allow_rel = "src/hacs/Console/serve/read_only_sources.allow";
    if (!ReadFile(root + "/" + allow_rel, allow_text)) {
        Check(false, "(C1) " + allow_rel + " is readable");
        return;
    }
    struct Allow {
        std::string file, fragment;
        std::size_t used = 0;
    };
    std::vector<Allow> allow;
    for (const std::string& line : Lines(allow_text)) {
        if (line.empty() || line[0] == '#') continue;
        const std::size_t bar = line.find('|');
        const std::size_t bar2 = (bar == std::string::npos) ? bar : line.find('|', bar + 1);
        if (bar == std::string::npos || bar2 == std::string::npos) {
            Check(false, "(C1) malformed allow entry '" + line + "' (relpath|fragment|reason)");
            continue;
        }
        allow.push_back({line.substr(0, bar), line.substr(bar + 1, bar2 - bar - 1), 0});
    }
    const std::vector<std::string> tus = RegistrationTUs(root);
    files_out = tus.size();
    for (const std::string& rel : tus) {
        std::string src;
        if (!ReadFile(root + "/" + rel, src)) {
            Check(false, "(C1) cannot read " + rel);
            continue;
        }
        const std::vector<std::string> lines = Lines(src);
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const std::string& l = lines[i];
            std::size_t b = l.find_first_not_of(" \t");
            if (b != std::string::npos && l.compare(b, 2, "//") == 0) continue;   // a comment line
            const std::string code = l.substr(0, l.find("//"));
            if (!std::regex_search(code, writer)) continue;
            ++hits_out;
            bool ok = false;
            for (Allow& a : allow) {
                if (a.file == rel && code.find(a.fragment) != std::string::npos) {
                    ++a.used;
                    ok = true;
                }
            }
            Check(ok, "(C1) " + rel + ":" + std::to_string(i + 1) + ": a writer / env / process entry point in a "
                      "registration TU of the console verbs is not on " + allow_rel + ": '" + code.substr(0, 120) + "'");
        }
    }
    for (const Allow& a : allow)
        Check(a.used > 0, "(C1) the allow entry " + a.file + "|" + a.fragment + " matches nothing (STALE)");
}

// ---------------------------------------------------------------------------
// (D) the manual
// ---------------------------------------------------------------------------
void ArmD(const Registry& reg, const std::string& root) {
    std::string verbs, counts;
    if (!ReadFile(root + "/docs/latex/cli_manual/verbs.tex", verbs) ||
        !ReadFile(root + "/docs/latex/cli_manual/counts.tex", counts)) {
        Check(false, "(D) docs/latex/cli_manual/{verbs,counts}.tex are readable");
        return;
    }
    std::vector<std::string> marks;
    for (const std::string& line : Lines(verbs)) {
        if (line.rfind("\\ocverb{", 0) != 0) continue;
        const std::size_t open = line.rfind('{');
        const std::size_t close = line.rfind('}');
        marks.push_back(open != std::string::npos && close > open ? line.substr(open + 1, close - open - 1)
                                                                   : std::string());
    }
    std::vector<std::string> want;
    std::size_t ro = 0;
    for (const std::string& name : reg.verb_names()) {
        const VerbEntry* v = reg.find_verb(name);
        if (v == nullptr) continue;
        for (const Overload& o : v->overloads) {
            const serve_tags::TagState s = serve_tags::LookupTag(*v, o);
            want.push_back(s == serve_tags::TagState::ReadOnly ? "ro" : s == serve_tags::TagState::NotReadOnly ? "rw" : "untagged");
            if (s == serve_tags::TagState::ReadOnly) ++ro;
        }
    }
    Check(marks.size() == want.size(), "(D) verbs.tex has one \\ocverb per registered overload (" +
                                           std::to_string(marks.size()) + " vs " + std::to_string(want.size()) + ")");
    for (std::size_t i = 0; i < marks.size() && i < want.size(); ++i)
        Check(marks[i] == want[i], "(D) verbs.tex overload #" + std::to_string(i + 1) + " is marked '" + marks[i] +
                                       "', the registry says '" + want[i] + "'");
    Check(counts.find("\\newcommand{\\CMNumReadOnly}{" + std::to_string(ro) + "}") != std::string::npos,
          "(D) counts.tex carries \\CMNumReadOnly{" + std::to_string(ro) + "}");
}

// ---------------------------------------------------------------------------
// (E) PrintJson type coverage and the envelope rules
// ---------------------------------------------------------------------------
bool NumbersOk(const Json& j, std::string& bad) {
    switch (j.kind) {
        case Json::Kind::Number: {
            std::int64_t v = 0;
            if (!j.as_int64(v)) { bad = j.text; return false; }
            const std::int64_t lim = 9007199254740991LL;
            if (v > lim || v < -lim) { bad = j.text; return false; }
            return true;
        }
        case Json::Kind::Array:
            for (const Json& x : j.items)
                if (!NumbersOk(x, bad)) return false;
            return true;
        case Json::Kind::Object:
            for (const auto& m : j.members)
                if (!NumbersOk(m.second, bad)) return false;
            return true;
        default:
            return true;
    }
}

bool EnvelopeKeysOk(const Json& e) {
    static const std::vector<std::string> head = {"t", "canonical", "hash", "v"};
    static const std::vector<std::string> tail = {"truncated", "length", "generic", "state", "hint"};
    if (!e.is_object() || e.members.size() < 4) return false;
    for (std::size_t i = 0; i < 4; ++i)
        if (e.members[i].first != head[i]) return false;
    std::size_t t = 0;
    for (std::size_t i = 4; i < e.members.size(); ++i) {
        while (t < tail.size() && tail[t] != e.members[i].first) ++t;
        if (t == tail.size()) return false;
        ++t;
    }
    return true;
}

void ArmE(std::size_t& rendered) {
    // The dedicated set equals [P] Sec. 18.5's table.
    const std::vector<std::string> spec = {"Nil", "Bool", "Int", "String", "List", "Record", "Fp", "Poly", "Matrix",
                                           "Model", "Complex", "LocalSystem", "Cohomology", "GF", "GFElt", "Block",
                                           "Decomposition", "Report", "ExtRecord", "Settings",
                                           // (slot-24 NB-RENDER R1) the member form D-m
                                           "HeckeOperator", "Pairing", "MaximalIdeal", "HeckeAlgebra", "LoewyProfile",
                                           "ExtensionClass", "BoundaryCohomology", "ConeCohomology", "DicksonLES",
                                           "DescentRecord"};
    Check(serve::DedicatedSchemaTypes() == spec, "(E) the dedicated (D) schema set equals [P] Sec. 18.5's table");
    FullRegistry fx;
    for (const std::string& t : spec) {
        if (t == "Nil" || t == "Bool" || t == "Int" || t == "String" || t == "List" || t == "Record") continue;
        Check(fx.registry.find_type(t) != nullptr, "(E) the D type " + t + " is registered");
    }
    std::size_t generic = 0;
    for (const std::string& t : fx.registry.type_names())
        if (!serve::HasDedicatedSchema(t)) ++generic;
    Check(generic + (spec.size() - 6) == fx.registry.type_names().size(),
          "(E) every registered type has a dedicated schema or the generic form");

    const hecke::ScopedMathSettings restore(hecke::math_settings());
    const char* script =
        "A = matrix(7, [[1, 2], [3, 4]]);\n"
        "f = poly(7, [1, 2, 3]);\n"
        "x = Fp(7, 3);\n"
        "big = 9007199254740993;\n"
        "safe = 9007199254740991;\n"
        "M = sl2z_model();\n"
        "L = symmetric_power(M, 4, 7);\n"
        "H = cohomology(L);\n"
        "lst = [1, \"a\", true];\n"
        "F = GF(7, 2);\n"
        "y = F.gen;\n"
        "S = settings_snapshot();\n"
        "R = sturm_bound(12, 691);\n"
        "Z = zero(7, 30, 30);\n";
    const int code = RunSource(fx.session, script, "<print_json>");
    Check(code == 0, "(E) the rendering fixture runs (exit " + std::to_string(code) + "): " + fx.out.str().substr(0, 200));
    if (code != 0) return;
    serve::PrintJsonContext ctx;
    ctx.registry = &fx.registry;
    ctx.session = &fx.session;
    ctx.max_entries = fx.session.print_max_entries();
    std::map<std::string, Json> env;
    for (const auto& kv : fx.session.vars()) {
        const std::string text = serve::PrintJson(kv.second, ctx);
        Json j;
        std::string err;
        const bool ok = serve_json::ParseJson(text, j, err);
        Check(ok, "(E) the envelope of " + kv.first + " parses: " + err);
        if (!ok) continue;
        ++rendered;
        Check(EnvelopeKeysOk(j), "(E) the envelope of " + kv.first + " has the key order t, canonical, hash, v[, ...]");
        std::string bad;
        Check(NumbersOk(j, bad), "(E) every number in the envelope of " + kv.first +
                                     " is an integer with |x| <= 2^53 - 1 (got '" + bad + "')");
        const Json* t = j.get("t");
        Check(t != nullptr && t->is_string() && t->text == kv.second.type_name(),
              "(E) t of " + kv.first + " is type(x) = " + kv.second.type_name());
        Check(text.find("\\\\(") == std::string::npos && text.find("\\\\frac") == std::string::npos &&
                  text.find("$") == std::string::npos,
              "(E) no TeX in the envelope of " + kv.first);
        env[kv.first] = j;
    }
    auto v_of = [&](const std::string& n) -> const Json* {
        const auto it = env.find(n);
        return it == env.end() ? nullptr : it->second.get("v");
    };
    if (const Json* v = v_of("A")) {
        Check(serve_json::WriteJson(*v) == "{\"rows\":2,\"cols\":2,\"p\":7,\"entries\":[[1,2],[3,4]]}",
              "(E) Matrix v = {rows, cols, p, entries} row-major: " + serve_json::WriteJson(*v));
        const Json* h = env["A"].get("hash");
        Check(h != nullptr && h->is_string() && h->text.rfind("0x", 0) == 0, "(E) a Matrix carries its golden hash");
    }
    if (const Json* v = v_of("f")) {
        Check(serve_json::WriteJson(*v) == "{\"ell\":7,\"deg\":2,\"order\":\"low_to_high\",\"coeffs\":[1,2,3]}",
              "(E) Poly v is labelled low_to_high: " + serve_json::WriteJson(*v));
    }
    if (const Json* v = v_of("x"))
        Check(serve_json::WriteJson(*v) == "{\"p\":7,\"value\":3}", "(E) Fp v: " + serve_json::WriteJson(*v));
    if (const Json* v = v_of("big"))
        Check(v->is_string() && v->text == "9007199254740993", "(E) an Int above 2^53 - 1 is a decimal String");
    if (const Json* v = v_of("safe"))
        Check(v->is_number() && v->text == "9007199254740991", "(E) an Int of 2^53 - 1 is a JSON number");
    if (const Json* v = v_of("lst")) Check(v->is_array() && v->items.size() == 3, "(E) a List is an array of envelopes");
    if (const Json* v = v_of("H")) {
        const Value* H = fx.session.find_var("H");
        Check(H != nullptr && H->is_object() && !H->object_value()->computed(),
              "(E) rendering a declared Cohomology did not force it");
        Check(v->get("dims") == nullptr && v->get("coefficient_ring") != nullptr,
              "(E) a declared Cohomology shows its structural members only (ell / d_V are registered computed)");
        const Json* h = env["H"].get("hash");
        Check(h != nullptr && h->is_null(), "(E) a declared handle's hash is null (never forced)");
        const Json* st = env["H"].get("state");
        Check(st != nullptr && st->text == "declared", "(E) a declared handle carries state \"declared\"");
    }
    if (const Json* v = v_of("y")) {
        Check(v->get("order") != nullptr && v->get("ell") != nullptr && v->get("r") != nullptr,
              "(E) GFElt v carries ell, r, coeffs and the order label");
    }
    if (const Json* v = v_of("F")) {
        const Json* p = v->get("polynomial");
        Check(p != nullptr && p->get("order") != nullptr, "(E) GF v carries its polynomial as a Poly tree");
    }
    if (const Json* v = v_of("S")) Check(v->get("entries") != nullptr && v->get("fingerprint") != nullptr,
                                         "(E) Settings v = {entries, fingerprint}");
    if (env.count("R")) {
        const Json* g = env["R"].get("generic");
        Check(g != nullptr && g->is_bool() && g->b, "(E) a G type (Sturm) uses the generic form");
    }
    if (env.count("Z")) {
        const Json* tr = env["Z"].get("truncated");
        Check(tr != nullptr && tr->b && env["Z"].get("v")->get("entries") == nullptr,
              "(E) a Matrix above print_max_entries is shape-only and truncated");
    }
    // A Record is an array of pairs, in the record's order.
    {
        Record r;
        r.set("zeta", Value::from_int(1));
        r.set("alpha", Value::from_int(2));
        Json j;
        std::string err;
        serve_json::ParseJson(serve::PrintJson(Value::from_record(r), ctx), j, err);
        const Json* v = j.get("v");
        Check(v != nullptr && v->is_array() && v->items.size() == 2 && v->items[0].items.size() == 2 &&
                  v->items[0].items[0].text == "zeta",
              "(E) a Record is an array of [key, envelope] pairs in the record's order");
    }
}

// ---------------------------------------------------------------------------
// (H) render hints (slot-24 NB-RENDER R1): Console/serve/render_hints.tsv ==
// PrintJson's compiled table, row for row; every type-named row names a
// registered type; a hint reaches the envelope at its path with its ring
// resolved; the member form carries no `generic` flag and makes no call on a
// declared handle.
// ---------------------------------------------------------------------------
void ArmH(const Registry& reg, const std::string& root, std::size_t& hint_rows) {
    const std::vector<serve::RenderHintRow>& rows = serve::RenderHintRows();
    hint_rows = rows.size();
    static const std::vector<std::string> kinds = {"poly", "qexp", "abgroup", "hilbert", "layers", "partition",
                                                   "cyclic", "eigen", "lesmap", "factored"};
    for (const serve::RenderHintRow& r : rows) {
        Check(std::find(kinds.begin(), kinds.end(), std::string(r.kind)) != kinds.end(),
              std::string("(H) the hint kind '") + r.kind + "' is one of the [P] Sec. 18.5 kinds");
        const std::string n = r.name;
        if (n.find('/') == std::string::npos)
            Check(reg.find_type(n) != nullptr, "(H) the hint row's type " + n + " is registered");
    }
    if (!root.empty()) {
        std::string text;
        const std::string rel = "src/hacs/Console/serve/render_hints.tsv";
        if (!ReadFile(root + "/" + rel, text)) {
            Check(false, "(H) " + rel + " is readable");
        } else {
            std::vector<std::vector<std::string>> fx;
            for (const std::string& line : Lines(text)) {
                if (line.empty() || line[0] == '#') continue;
                std::vector<std::string> f;
                std::string cur;
                for (char c : line) {
                    if (c == '\t') { f.push_back(cur); cur.clear(); }
                    else cur += c;
                }
                f.push_back(cur);
                fx.push_back(f);
            }
            Check(fx.size() == rows.size(), "(H) render_hints.tsv has " + std::to_string(fx.size()) + " rows, PrintJson " +
                                                std::to_string(rows.size()));
            for (std::size_t i = 0; i < fx.size() && i < rows.size(); ++i) {
                const std::string attrs = *rows[i].attrs ? rows[i].attrs : "-";
                const bool same = fx[i].size() == 4 && fx[i][0] == rows[i].name && fx[i][1] == rows[i].path &&
                                  fx[i][2] == rows[i].kind && fx[i][3] == attrs;
                Check(same, "(H) render_hints.tsv row " + std::to_string(i + 1) + " equals PrintJson's (" + rows[i].name + " " +
                                rows[i].path + ")");
            }
        }
    }
    // A schema'd Record: the hint lands on ops[].charpoly with ell resolved from the record.
    FullRegistry fx;
    serve::PrintJsonContext ctx;
    ctx.registry = &fx.registry;
    ctx.session = &fx.session;
    ctx.max_entries = fx.session.print_max_entries();
    Record op;
    op.set("p", Value::from_int(2));
    op.set("charpoly", Value::from_list({Value::from_int(4), Value::from_int(1)}));
    Record r;
    r.set("schema", Value::from_string("level_n/v1"));
    r.set("ell", Value::from_int(691));
    r.set("ops", Value::from_list({Value::from_record(op)}));
    Json j;
    std::string err;
    const std::string text = serve::PrintJson(Value::from_record(r), ctx);
    Check(serve_json::ParseJson(text, j, err), "(H) the hinted Record parses: " + err);
    Check(text.find("\"hint\":{\"kind\":\"poly\",\"order\":\"low_to_high\",\"var\":\"X\",\"ring\":\"F_ell\",\"ell\":691}") !=
              std::string::npos,
          "(H) level_n/v1 ops[].charpoly carries the poly hint with ell = 691: " + text.substr(0, 400));
    Check(text.find("\"hint\"") == text.rfind("\"hint\""), "(H) exactly one hint in the fixture (no hint off its path)");
    // Without the schema the same Record carries no hint.
    Record bare = r;
    bare.set("schema", Value::from_string("not_a_schema/v1"));
    Check(serve::PrintJson(Value::from_record(bare), ctx).find("\"hint\"") == std::string::npos,
          "(H) an unknown schema carries no hint");
    // The member form on a declared DescentRecord: no generic flag, no call, not forced.
    const hecke::ScopedMathSettings restore(hecke::math_settings());
    const int code = RunSource(fx.session, "set coefficient_ring = \"Z\";\nD = descent(1, 2);\n", "<print_json_h>");
    Check(code == 0, "(H) the descent fixture runs (exit " + std::to_string(code) + ")");
    if (code == 0) {
        const Value* D = fx.session.find_var("D");
        if (D != nullptr) {
            Json e;
            const std::string dt = serve::PrintJson(*D, ctx);
            Check(serve_json::ParseJson(dt, e, err) && EnvelopeKeysOk(e), "(H) the DescentRecord envelope parses, keys in order");
            Check(e.get("generic") == nullptr && e.get("v") != nullptr && e.get("v")->get("members") != nullptr &&
                      e.get("v")->get("calls") != nullptr,
                  "(H) DescentRecord uses the member form (members + calls, no generic flag)");
            Check(D->is_object() && !D->object_value()->computed(), "(H) rendering a declared DescentRecord did not force it");
        }
    }
}

// ---------------------------------------------------------------------------
// (R) render coverage (slot-24 NB-RENDER R4): Console/serve/render_coverage.tsv
// decides, for every value that can reach --serve, how the page renders it --
// tex (a toTeXRaw case of notebook/page/render.js), hint (a row of
// render_hints.tsv), table (the generic Record / List structure) or allow (its
// canonical text, by an explicit decision with a reason).  Both directions: every
// core kind, registered type and Record schema of the verb sources has exactly one
// row, and every row names one; every toTeXRaw case of a registered type is marked
// tex; every hint name is marked hint; every hint kind has a page renderer.
// ---------------------------------------------------------------------------
std::vector<std::vector<std::string>> TsvRows(const std::string& text) {
    std::vector<std::vector<std::string>> rows;
    for (const std::string& line : Lines(text)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> f;
        std::string cur;
        for (char c : line) {
            if (c == '\t') { f.push_back(cur); cur.clear(); }
            else cur += c;
        }
        f.push_back(cur);
        rows.push_back(f);
    }
    return rows;
}

// The `case "<name>":` labels of the body of `function <fn>(` in render.js, up to the next top-level `function `.
std::set<std::string> JsCases(const std::string& js, const std::string& fn) {
    std::set<std::string> out;
    const std::size_t a = js.find("function " + fn + "(");
    if (a == std::string::npos) return out;
    std::size_t b = js.find("\n  function ", a + 1);
    if (b == std::string::npos) b = js.size();
    const std::string body = js.substr(a, b - a);
    static const std::regex re("case \"([A-Za-z_][A-Za-z0-9_]*)\":");
    for (std::sregex_iterator it(body.begin(), body.end(), re), end; it != end; ++it) out.insert((*it)[1].str());
    return out;
}

void ScanSchemas(const std::string& dir, std::set<std::string>& literal, std::size_t& indirect, std::size_t& files) {
    DIR* d = ::opendir(dir.c_str());
    if (d == nullptr) return;
    std::vector<std::string> names;
    while (dirent* e = ::readdir(d)) names.push_back(e->d_name);
    ::closedir(d);
    std::sort(names.begin(), names.end());
    static const std::regex lit("set\\(\"schema\",\\s*Value::from_string\\(\"([^\"]+)\"\\)");
    static const std::regex any("set\\(\"schema\",\\s*Value::from_string\\(");
    for (const std::string& n : names) {
        if (n == "." || n == "..") continue;
        const std::string path = dir + "/" + n;
        struct stat st;
        if (::stat(path.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            ScanSchemas(path, literal, indirect, files);
            continue;
        }
        if (n.size() < 4 || n.substr(n.size() - 4) != ".cpp" || n.find("Validation") != std::string::npos) continue;
        std::string text;
        if (!ReadFile(path, text)) continue;
        ++files;
        std::size_t all = 0, lits = 0;
        for (std::sregex_iterator it(text.begin(), text.end(), any), end; it != end; ++it) ++all;
        for (std::sregex_iterator it(text.begin(), text.end(), lit), end; it != end; ++it) {
            literal.insert((*it)[1].str());
            ++lits;
        }
        indirect += all - lits;
    }
}

void ArmR(const Registry& reg, const std::string& root, std::size_t& coverage_rows) {
    if (root.empty()) {
        std::cout << "  [SKIP] (R): the source tree was not found\n";
        return;
    }
    const std::string rel = "src/hacs/Console/serve/render_coverage.tsv";
    std::string text, hints_text, js;
    Check(ReadFile(root + "/" + rel, text), "(R) " + rel + " is readable");
    Check(ReadFile(root + "/src/hacs/Console/serve/render_hints.tsv", hints_text),
          "(R) render_hints.tsv is readable");
    Check(ReadFile(root + "/notebook/page/render.js", js), "(R) notebook/page/render.js is readable");
    const std::vector<std::vector<std::string>> rows = TsvRows(text);
    coverage_rows = rows.size();
    std::set<std::string> hint_names, hint_kinds;
    for (const std::vector<std::string>& h : TsvRows(hints_text))
        if (h.size() == 4) {
            hint_names.insert(h[0]);
            hint_kinds.insert(h[2]);
        }
    const std::set<std::string> tex_cases = JsCases(js, "toTeXRaw");
    const std::set<std::string> page_kinds = JsCases(js, "hintTeX");

    // The universe: the core kinds, the registered types, the Record schemas of the verb sources.
    const std::vector<std::string> kinds = {"Nil", "Bool", "Int", "String", "List", "Record"};
    std::set<std::string> types;
    for (const std::string& t : reg.type_names()) types.insert(t);
    std::set<std::string> schemas;
    std::size_t indirect = 0, files = 0;
    for (const char* d : {"src/hacs/HeckeExt", "src/hacs/Console/Bindings",
                          "src/hacs/Integral"})
        ScanSchemas(root + "/" + d, schemas, indirect, files);
    Check(files > 0 && !schemas.empty(), "(R) the verb sources were scanned for Record schemas (" + std::to_string(files) +
                                             " files, " + std::to_string(schemas.size()) + " literal schemas)");

    std::map<std::string, std::vector<std::string>> by_name;
    std::size_t schema_rows = 0;
    static const std::set<std::string> renders = {"tex", "hint", "table", "allow"};
    for (const std::vector<std::string>& r : rows) {
        if (r.size() != 4) {
            Check(false, "(R) a render_coverage.tsv row has 4 TAB-separated fields: " + (r.empty() ? std::string() : r[0]));
            continue;
        }
        const std::string& name = r[0];
        const std::string& cls = r[1];
        Check(by_name.count(name) == 0, "(R) " + name + " has one row only");
        by_name[name] = r;
        // the class names a real thing
        if (cls == "kind") Check(std::find(kinds.begin(), kinds.end(), name) != kinds.end(), "(R) " + name + " is a core kind");
        else if (cls == "type") Check(types.count(name) == 1, "(R) " + name + " is a registered type (stale row?)");
        else if (cls == "schema") {
            ++schema_rows;
            Check(schemas.count(name) == 1 || hint_names.count(name) == 1,
                  "(R) the schema " + name + " is set by a verb source or named by a render hint (stale row?)");
        } else {
            Check(false, "(R) " + name + ": the class '" + cls + "' is kind, type or schema");
        }
        // the render decision is well-formed and true
        std::vector<std::string> parts;
        {
            std::string cur;
            for (char c : r[2] + "+") {
                if (c == '+') { if (!cur.empty()) parts.push_back(cur); cur.clear(); }
                else cur += c;
            }
        }
        bool tex = false, hint = false;
        for (const std::string& p : parts) {
            Check(renders.count(p) == 1, "(R) " + name + ": the render '" + p + "' is tex, hint, table or allow");
            tex = tex || p == "tex";
            hint = hint || p == "hint";
            if (p == "table") Check(name == "List" || name == "Record", "(R) " + name + ": `table` is the List / Record structure only");
        }
        Check(!parts.empty(), "(R) " + name + " has a render decision");
        if (std::find(parts.begin(), parts.end(), "allow") != parts.end())
            Check(parts.size() == 1, "(R) " + name + ": `allow` combines with nothing");
        const std::string reason = r[3];
        Check(reason.size() >= 8 && reason != "-", "(R) " + name + " gives a reason (" + reason + ")");
        if (tex) Check(tex_cases.count(name) == 1, "(R) " + name + " is marked tex: render.js toTeXRaw has case \"" + name + "\"");
        if (hint) Check(hint_names.count(name) == 1, "(R) " + name + " is marked hint: render_hints.tsv has a row for it");
    }
    // the other direction
    for (const std::string& k : kinds) Check(by_name.count(k) == 1 && by_name[k][1] == "kind", "(R) the core kind " + k + " has a row");
    for (const std::string& t : types)
        Check(by_name.count(t) == 1 && by_name[t][1] == "type",
              "(R) the registered type " + t + " has a render_coverage.tsv row (decide how the page renders it)");
    for (const std::string& sc : schemas)
        Check(by_name.count(sc) == 1 && by_name[sc][1] == "schema",
              "(R) the Record schema " + sc + " (set in a verb source) has a render_coverage.tsv row");
    Check(schema_rows >= schemas.size() + indirect,
          "(R) every schema site has a row: " + std::to_string(schema_rows) + " schema rows for " + std::to_string(schemas.size()) +
              " literal + " + std::to_string(indirect) + " indirect sites");
    for (const std::string& c : tex_cases)
        if (types.count(c) == 1 || std::find(kinds.begin(), kinds.end(), c) != kinds.end())
            Check(by_name.count(c) == 1 && by_name[c][2].find("tex") != std::string::npos,
                  "(R) render.js renders " + c + " as TeX, and render_coverage.tsv says tex");
    for (const std::string& h : hint_names)
        Check(by_name.count(h) == 1 && by_name[h][2].find("hint") != std::string::npos,
              "(R) the hinted name " + h + " is marked hint in render_coverage.tsv");
    for (const std::string& k : hint_kinds)
        Check(page_kinds.count(k) == 1, "(R) the hint kind " + k + " has a page renderer (render.js hintTeX case)");
}

// ---------------------------------------------------------------------------
// (U) units
// ---------------------------------------------------------------------------
void ArmU() {
    Json j;
    std::string err;
    Check(serve_json::ParseJson("{\"a\":[1,-2,\"x\\u00e9\\n\"],\"b\":null,\"c\":true}", j, err),
          "(U) the reader parses a well-formed request: " + err);
    Check(j.get("a") != nullptr && j.get("a")->items.size() == 3 && j.get("a")->items[2].text == "x\xc3\xa9\n",
          "(U) \\u escapes decode to UTF-8");
    Check(!serve_json::ParseJson("{\"a\":", j, err), "(U) a truncated object is refused");
    Check(!serve_json::ParseJson("{\"a\":1} x", j, err), "(U) trailing characters are refused");
    Check(!serve_json::ParseJson("[01]", j, err), "(U) a leading zero is refused");
    Check(serve_json::ParseJson("{\"n\":12345678901234567890}", j, err) &&
              serve_json::WriteJson(j) == "{\"n\":\"12345678901234567890\"}",
          "(U) the writer re-emits an integer above 2^53 - 1 as a String");
    Check(serve_json::IntJson(9007199254740991LL) == "9007199254740991", "(U) IntJson(2^53 - 1) is a number");
    Check(serve_json::IntJson(9007199254740992LL) == "\"9007199254740992\"", "(U) IntJson(2^53) is a String");
    Check(serve_json::IntJson(-9007199254740992LL) == "\"-9007199254740992\"", "(U) IntJson(-2^53) is a String");
    const std::vector<std::pair<std::string, bool>> env = {
        {"GOLDEN_EMIT", true},       {"GOLDEN_STRICT", true},   {"CONSOLE_EMIT_EXPECTED", true},
        {"CLI_MANUAL_EMIT", true},   {"HECKE_DUMP_TAU", true},  {"HECKE_NEBENTYPUS_NB3_FULL", true},
        {"HECKE_SOMETHING_NEW", true}, {"ENGINE_DEVICE", true}, {"ENGINE_BACKEND", true},
        {"ENGINE_LADIC_WIDTH", true}, {"OC_ANYTHING", true},     {"ASAN_OPTIONS", true},
        {"UBSAN_OPTIONS", true},     {"TSAN_OPTIONS", true},    {"ENGINE_THREADS", false},
        {"ENGINE_CUDA_DISABLE", false}, {"ENGINE_CUDA_DEVICE", false}, {"ENGINE_HOST_THREADS", false},
        {"PATH", false},             {"HOME", false},           {"LANG", false},
        {"TMPDIR", false},           {"USER", false},           {"CUDA_VISIBLE_DEVICES", false}};
    for (const auto& e : env)
        Check(serve::EnvNameRefused(e.first) == e.second,
              std::string("(U) the environment gate ") + (e.second ? "refuses " : "admits ") + e.first);
}

} // namespace
#endif

uint32_t ConsoleReadOnlyTagValidation() {
    if (!oc::kBuildNotebook) return oc::BuildSwitchSkip("ConsoleReadOnlyTagValidation", "OC_BUILD_NOTEBOOK");
#if OC_BUILD_NOTEBOOK
    std::cout << "=== ConsoleReadOnlyTagValidation ([P] v3.10 Sec. 18.8 / 18.11; slot-19 NB-2) ===\n";
    g_checks = 0;
    g_fail = 0;
    const std::string root = FindRepoRoot();
    FullRegistry fx;
    const serve_tags::TagCensus census = serve_tags::CensusOf(fx.registry);

    // (B) completeness of the default.
    Check(census.untagged.empty(), "(B) every registered overload is explicitly tagged (" +
                                       std::to_string(census.untagged.size()) + " untagged)");
    for (const std::string& u : census.untagged) Check(false, "(B) untagged overload: " + u);
    Check(census.stale.empty(), "(B) every row of the tag table matches a registered overload");
    for (const std::string& s : census.stale) Check(false, "(B) stale tag row: " + s);
    Check(census.overloads == census.tagged, "(B) registered == explicitly tagged (" + std::to_string(census.overloads) +
                                                 " / " + std::to_string(census.tagged) + ")");

    std::size_t rows = 0, hits = 0, files = 0, rendered = 0;
    if (root.empty()) {
        std::cout << "  [SKIP] (A) (C1) (D): the source tree was not found (OC_SOURCE_DIR / working directory)\n";
    } else {
        ArmA(fx.registry, root, rows);
        ArmC1(root, hits, files);
        ArmD(fx.registry, root);
    }
    ArmE(rendered);
    std::size_t hint_rows = 0;
    ArmH(fx.registry, root, hint_rows);
    std::size_t coverage_rows = 0;
    ArmR(fx.registry, root, coverage_rows);
    ArmU();

    std::cout << "EFFECTIVE ConsoleReadOnlyTagValidation arm=ALL cells=" << census.overloads
              << " tagged=" << census.tagged << " read_only=" << census.read_only
              << " not_read_only=" << (census.tagged - census.read_only)
              << " read_only_verbs=" << census.read_only_verbs << " fixture_rows=" << rows
              << " source_files=" << files << " source_hits=" << hits << " envelopes=" << rendered
              << " hint_rows=" << hint_rows << " coverage_rows=" << coverage_rows << "\n";
    std::cout << "ConsoleReadOnlyTagValidation: " << g_checks << " checks, " << g_fail << " failing\n";
    std::cout << "ConsoleReadOnlyTagValidation: " << (g_fail == 0 ? "PASSED" : "FAILED") << "\n";
    return g_fail;
#else
    return 0;
#endif
}
