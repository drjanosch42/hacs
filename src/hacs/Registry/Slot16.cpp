// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-16 registration (closed slot): the former per-package fragments Registry/Slot16/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot16() calls them in the old RegisterAllValidations() order.

// ===== Slot 16 WP2 ===========================================
// Slot-16 WP2 registration fragment (hecke-extensions slot 16). Owned by
// package WP2 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot16/WP2.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot16WP2() {
    // Empty at day zero (slot-16 WP-1).
}

// ===== Slot 16 WP3 ===========================================
// Slot-16 WP3 registration fragment (hecke-extensions slot 16). Owned by
// package WP3 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot16/WP3.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot16WP3() {
    // Empty at day zero (slot-16 WP-1).
}

// ===== Slot 16 WP5 ===========================================
// Slot-16 WP5 registration fragment (hecke-extensions slot 16). Owned by
// package WP5 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot16/WP5.cmake.
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeSaturationDefectValidation.h"   // slot-16 WP-5

void RegisterSlot16WP5() {
    // (hecke-extensions slot 16, WP-5) the saturation-defect Tor term
    // I_m = Tor_1^{Z_ell}(T^sat / T_m, F_ell) on torsion-free blocks
    // (HeckeExt/HeckeSaturationDefect).  `standard`; HECKE_SAT_DEFECT_CELLS=
    // "n:ell[:KMAX],..." runs it over a cell list instead (read-only, not an
    // emitter; SAT_DEFECT lines, one EFFECTIVE line).
    ValidationRegistry::register_validation("HeckeSaturationDefectValidation",
                                            &HeckeSaturationDefectValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 16 WP10 ===========================================
// Slot-16 WP10 registration fragment (hecke-extensions slot 16). Owned by
// package WP10 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot16/WP10.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeNebentypusValidation.h"   // slot-16 WP-10

void RegisterSlot16WP10() {
    // (hecke-extensions slot 16, WP-10) Nebentypus on the fixed model by Shapiro --
    // [C, sec:nebentypus] (eq:MHchi, eq:nebcorr, prop:neborbit, rmk:nebheilbronn),
    // rmk:nebcheckable (NB1)-(NB3), (NB5), (NB6), the chi / chibar dictionary of
    // [GT, bridge:nebentypus-dictionary] read against WP-9's table.  Writes nothing;
    // NEB_ROUTES / NEB_GAMMA1 / NEB_DICT / NEB_CELL lines.  Env hook HECKE_NEBENTYPUS_CELLS.
    ValidationRegistry::register_validation("HeckeNebentypusValidation",
                                            &HeckeNebentypusValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 16 WP13 ===========================================
// Slot-16 WP13 registration fragment (hecke-extensions slot 16). Owned by
// package WP13 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot16/WP13.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot16WP13() {
    // Empty at day zero (slot-16 WP-1).
}

// ===== Slot 16 WP16 ===========================================
// Slot-16 WP16 registration fragment (hecke-extensions slot 16, stretch package
// WP-16). Owned by package WP16 ONLY: register this package's validations here
// and nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot16/WP16.cmake.
#include "../ValidationRegistry.h"
#include "../Integral/HeckeIsotropyModelEllValidation.h"   // slot-16 WP-16

void RegisterSlot16WP16() {
    // (hecke-extensions slot 16, WP-16) The ell-local isotropy model at ell = 3 in code --
    // [C, def:isomodelell], prop:isomodelell (every lift averaged by e_sigma, C-1),
    // prop:ellthreeclosed, rmk:isomodellcheckable (IL1)-(IL4), against the free model and
    // the shipped Bredon route; the un-averaged comparison map as the negative control.
    // Writes nothing; ISOMODEL_ELL_* lines.  No env hook.
    ValidationRegistry::register_validation("HeckeIsotropyModelEllValidation",
                                            &HeckeIsotropyModelEllValidation,
                                            ValidationTier::Standard);
}

// ===== Slot 16 L0 ===========================================
// Slot-16 L16-0 registration fragment (hecke-extensions slot 16). Owned by package L16-0 ONLY.
// Called explicitly from RegisterAllValidations() (never a static initialiser); its source is
// listed in cmake/slot16/L0.cmake.
//
// LeanCensusFileValidation (`fast`): checks the COMMITTED lean/census.tsv against the \lean{} /
// \label{} / \leanok content of [GT] (lean/blueprint/src/gl2_groundtruth.tex), and the Lake pin
// files.  It NEVER invokes lake or lean: the build-and-axioms side is tools/lean_census.sh.
#include "../ValidationRegistry.h"

#include <algorithm>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

bool PathExists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }

bool LooksLikeRepoRoot(const std::string& d) {
    return PathExists(d + "/lean/census.tsv") && PathExists(d + "/docs/latex/gl2_contract.tex");
}

std::string FindRepoRoot() {
#ifdef OC_SOURCE_DIR
    { const std::string r(OC_SOURCE_DIR); if (!r.empty() && LooksLikeRepoRoot(r)) return r; }
#endif
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            const size_t s = dir.find_last_of('/');
            if (s == std::string::npos || s == 0) break;
            dir = dir.substr(0, s);
            if (LooksLikeRepoRoot(dir)) return dir;
        }
    }
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (LooksLikeRepoRoot(dir)) return dir;
            const size_t s = dir.find_last_of('/');
            if (s == std::string::npos || s == 0) break;
            dir = dir.substr(0, s);
        }
    }
    return std::string();
}

std::string Slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

// Remove unescaped TeX comments (% to end of line).
std::string StripTexComments(const std::string& t) {
    std::string out; out.reserve(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
        if (t[i] == '%' && (i == 0 || t[i - 1] != '\\')) {
            while (i < t.size() && t[i] != '\n') ++i;
            if (i < t.size()) out.push_back('\n');
        } else out.push_back(t[i]);
    }
    return out;
}

std::vector<std::string> Split(const std::string& s, char c) {
    std::vector<std::string> v; std::string cur;
    for (char x : s) { if (x == c) { v.push_back(cur); cur.clear(); } else cur.push_back(x); }
    v.push_back(cur);
    return v;
}

std::string Trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

// (label, decl) -> lean_status, computed from [GT] exactly as lean_census.py does.
std::map<std::pair<std::string, std::string>, std::string> GtTable(const std::string& tex_raw) {
    std::map<std::pair<std::string, std::string>, std::string> out;
    const std::string tex = StripTexComments(tex_raw);
    static const char* envs[] = {"theorem", "lemma", "proposition", "corollary", "definition",
                                 "remark", "conjecture", "example", "axiom", "hypothesis"};
    size_t pos = 0;
    while (true) {
        size_t best = std::string::npos; std::string env; bool star = false;
        for (const char* e : envs)
            for (int st = 0; st < 2; ++st) {
                const std::string tag = std::string("\\begin{") + e + (st ? "*}" : "}");
                const size_t p = tex.find(tag, pos);
                if (p != std::string::npos && p < best) { best = p; env = e; star = st; }
            }
        if (best == std::string::npos) break;
        const std::string open = std::string("\\begin{") + env + (star ? "*}" : "}");
        const std::string close = std::string("\\end{") + env + (star ? "*}" : "}");
        const size_t bs = best + open.size();
        const size_t be = tex.find(close, bs);
        if (be == std::string::npos) break;
        const std::string body = tex.substr(bs, be - bs);
        pos = be + close.size();
        std::string proof;
        {
            size_t q = pos;
            while (q < tex.size() && (tex[q] == ' ' || tex[q] == '\n' || tex[q] == '\t' || tex[q] == '\r')) ++q;
            if (tex.compare(q, 13, "\\begin{proof}") == 0) {
                const size_t pe = tex.find("\\end{proof}", q);
                if (pe != std::string::npos) { proof = tex.substr(q, pe - q); pos = pe + 11; }
            }
        }
        std::vector<std::string> decls;
        for (size_t p = 0; (p = body.find("\\lean{", p)) != std::string::npos;) {
            const size_t e = body.find('}', p);
            if (e == std::string::npos) break;
            for (const std::string& d : Split(body.substr(p + 6, e - p - 6), ',')) if (!Trim(d).empty()) decls.push_back(Trim(d));
            p = e;
        }
        if (decls.empty()) continue;
        std::string label = "?";
        const size_t l = body.find("\\label{");
        if (l != std::string::npos) { const size_t e = body.find('}', l); if (e != std::string::npos) label = body.substr(l + 7, e - l - 7); }
        const bool sok = body.find("\\leanok") != std::string::npos;
        const bool pok = proof.find("\\leanok") != std::string::npos;
        const bool mok = body.find("\\mathlibok") != std::string::npos;
        const bool lok = body.find("\\gtmodlit") != std::string::npos;  // L16-2: proved modulo named literature
        // L16-5 repair: a `hypothesis` environment (the literature register) is lit_hyp, never stated.
        const std::string status = mok ? "lit_mathlib"
                                 : (env == "hypothesis") ? (sok ? "lit_hyp" : "planned")
                                 : (sok && pok) ? (lok ? "proved_mod_lit" : "proved") : sok ? "stated" : "planned";
        for (const std::string& d : decls) out[{label, d}] = status;
    }
    return out;
}

// (S18 L18-5, F3 13) The census.tsv comment line `# sources_sha256=<64 lowercase hex>`.  Value-free: the format only;
// the VALUE is checked against the committed sources by lean_census.py / lean_census_committed.py (no sha256 in C++).
bool IsSourcesHashLine(const std::string& line) {
    static const std::string pre = "# sources_sha256=";
    if (line.compare(0, pre.size(), pre) != 0) return false;
    const std::string v = line.substr(pre.size());
    if (v.size() != 64) return false;
    for (char ch : v) if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
    return true;
}

unsigned LeanCensusFileValidation() {
    int failures = 0;
    long checks = 0;
    auto check = [&](bool ok, const std::string& what) {
        ++checks;
        if (!ok) { ++failures; std::cout << "  [FAIL] " << what << "\n"; }
    };

    // negative control of the comment-line format check (planted malformed lines must all be rejected)
    check(IsSourcesHashLine("# sources_sha256=" + std::string(64, 'a')), "sources_sha256 line: well-formed line accepted");
    check(!IsSourcesHashLine("# sources_sha256=" + std::string(63, 'a')), "sources_sha256 control: 63 hex digits rejected");
    check(!IsSourcesHashLine("# sources_sha256=" + std::string(64, 'A')), "sources_sha256 control: uppercase rejected");
    check(!IsSourcesHashLine("# sources_sha256=" + std::string(63, 'a') + "g"), "sources_sha256 control: non-hex rejected");
    check(!IsSourcesHashLine("# sources_sha256="), "sources_sha256 control: empty value rejected");

    const std::string root = FindRepoRoot();
    check(!root.empty(), "repository root (lean/census.tsv + docs/latex/gl2_contract.tex) located");
    if (root.empty()) { std::cout << "LeanCensusFileValidation: " << checks << " checks, " << failures << " failing\n"; return 1; }
    const std::string lean = root + "/lean";

    // Lake pin files.
    check(Trim(Slurp(lean + "/lean-toolchain")).rfind("leanprover/lean4:v4.33.", 0) == 0, "lean-toolchain is a v4.33.x pin");
    const std::string lake = Slurp(lean + "/lakefile.toml");
    check(lake.find("db584cd6d46c92f209a44c0f1c829460d327499d") != std::string::npos, "lakefile.toml pins mathlib db584cd");
    check(lake.find("packagesDir") == std::string::npos, "lakefile.toml sets no packagesDir (the shared tree comes from the .lake/packages symlink)");
    check(PathExists(lean + "/GL2.lean") && PathExists(lean + "/GL2/Census.lean"), "GL2.lean and GL2/Census.lean exist");
    check(PathExists(lean + "/tools/lean_census.sh") && PathExists(lean + "/tools/lean_census.py"), "tools/lean_census.{sh,py} exist");

    // census.tsv versus [GT].
    const std::string tex = Slurp(lean + "/blueprint/src/gl2_groundtruth.tex");
    check(!tex.empty(), "[GT] source blueprint/src/gl2_groundtruth.tex is readable");
    auto gt = GtTable(tex);
    {   // [GT]'s parts: blueprint/src/gt_part_*.tex, each parsed on its own, in name order (as lean_census.py does).
        std::vector<std::string> parts;
        if (DIR* d = ::opendir((lean + "/blueprint/src").c_str())) {
            while (const dirent* e = ::readdir(d)) {
                const std::string n(e->d_name);
                if (n.rfind("gt_part_", 0) == 0 && n.size() > 12 && n.compare(n.size() - 4, 4, ".tex") == 0) parts.push_back(n);
            }
            ::closedir(d);
        }
        std::sort(parts.begin(), parts.end());
        for (const std::string& n : parts)
            for (const auto& kv : GtTable(Slurp(lean + "/blueprint/src/" + n))) gt[kv.first] = kv.second;
    }

    std::map<std::pair<std::string, std::string>, std::string> tsv;
    {
        std::istringstream in(Slurp(lean + "/census.tsv"));
        std::string line; bool header = true; int hash_lines = 0;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            // S18 L18-5: `#` lines are comments; exactly one must be the well-formed sources_sha256 line.
            if (line[0] == '#') {
                if (line.rfind("# sources_sha256=", 0) == 0) { ++hash_lines; check(IsSourcesHashLine(line), "census.tsv sources_sha256 line is well-formed"); }
                continue;
            }
            // L16-5 repair: a fifth column, lit_hyps (the literature hypotheses of a proved_mod_lit row).
            if (header) { check(line == "label\tdecl\tlean_status\taxioms\tlit_hyps", "census.tsv header"); header = false; continue; }
            const auto c = Split(line, '\t');
            check(c.size() == 5, "census.tsv row has 5 columns: " + line);
            if (c.size() == 5) {
                check(tsv.find({c[0], c[1]}) == tsv.end(), "no duplicate census row " + c[0] + " " + c[1]);
                tsv[{c[0], c[1]}] = c[2];
                // The committed axioms column of a proved row must be a standard-axiom list.
                // lit_hyps is recorded only on proved_mod_lit rows.
                if (c[2] != "proved_mod_lit") check(c[4] == "-", "non-proved_mod_lit row " + c[1] + " has lit_hyps '-'");
                if (c[2] == "proved" || c[2] == "proved_mod_lit" || c[2] == "lit_hyp") {
                    bool ok = true;
                    for (const std::string& a : Split(c[3], ','))
                        if (a != "none" && a != "propext" && a != "Classical.choice" && a != "Quot.sound") ok = false;
                    check(ok, "proved / lit_hyp row " + c[1] + " depends only on the standard axioms (or none)");
                }
            }
        }
        check(!header, "census.tsv is non-empty");
        check(hash_lines == 1, "census.tsv carries exactly one # sources_sha256= line");
    }
    for (const auto& kv : gt) {
        auto it = tsv.find(kv.first);
        check(it != tsv.end(), "[GT] statement " + kv.first.first + " / " + kv.first.second + " has a census row");
        if (it != tsv.end()) check(it->second == kv.second, "status of " + kv.first.first + ": census " + it->second + " vs [GT] " + kv.second);
    }
    for (const auto& kv : tsv)
        check(gt.find(kv.first) != gt.end(), "census row " + kv.first.first + " / " + kv.first.second + " is in [GT] (no stale row)");

    std::cout << "LeanCensusFileValidation: " << checks << " checks, " << failures << " failing\n";
    return failures == 0 ? 0 : 1;
}

}  // namespace

void RegisterSlot16L0() {
    // (hecke-extensions slot 16, L16-0) the committed Lean census against [GT]; never runs lake.
    ValidationRegistry::register_validation("LeanCensusFileValidation", &LeanCensusFileValidation,
                                            ValidationTier::Fast);
}

// ===== Slot 16 L4 ===========================================
// Slot-16 L4 registration fragment (hecke-extensions slot 16). Owned by
// package L4 ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot16/L4.cmake.
#include "../ValidationRegistry.h"
#include "../Hecke/HeckeTauDumpValidation.h"

void RegisterSlot16L4() {
    // (L16-4, owner ruling Lean Q14) the validation-only table-dump hook for the Lean
    // checkers: a no-op ([SKIP] line, exit 0) unless HECKE_DUMP_TAU=<dir> names an
    // existing directory outside the source tree.
    ValidationRegistry::register_validation("HeckeTauDumpValidation", &HeckeTauDumpValidation,
                                            ValidationTier::Fast);
}

// ===== Slot 16 WP15 ===========================================
// Slot-16 WP15 registration fragment (hecke-extensions slot 16, stretch package
// WP-15). Owned by package WP15 ONLY: register this package's validations here and
// nowhere else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are listed in
// cmake/slot16/WP15.cmake.  Created by the package itself with the two one-line
// shared edits (slot-12 WP-14 precedent; Registry/Slot16/README.md).
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeHidaCMValidation.h"   // slot-16 WP-15

void RegisterSlot16WP15() {
    // (hecke-extensions slot 16, WP-15) [U] part 2 in code: the CM components on the
    // character model of [C, sec:nebentypus] -- (U8)-(U10) of [U] rmk:u-pred2 at the
    // (NB6) cells and on the CM ladders at tame N >= 3, (U9) at tame level 1 and 2,
    // the h_K = 1 hypothesis of [U] prop:u-cmeis(2) with an h_K = 2 negative example.
    // Writes nothing; HIDA_CM_* lines.  Env hook HECKE_HIDA_CM_CELLS.
    ValidationRegistry::register_validation("HeckeHidaCMValidation",
                                            &HeckeHidaCMValidation,
                                            ValidationTier::Standard);
}

void RegisterSlot16() {
    RegisterSlot16WP2();
    RegisterSlot16WP3();
    RegisterSlot16WP5();
    RegisterSlot16WP10();
    RegisterSlot16WP13();
    RegisterSlot16WP16();
    RegisterSlot16L0();
    RegisterSlot16L4();
    RegisterSlot16WP15();
}
