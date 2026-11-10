// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ConsoleCostModelValidation -- see the header.

#include "ConsoleCostModelValidation.h"

#include "Bindings/Calibrate.h"
#include "Bindings/HostProbe.h"
#include "Bindings/RegisterPipelineVerbs.h"
#include "Core/Confirm.h"
#include "Core/ConsoleCli.h"
#include "Core/CostModel.h"
#include "Core/Registry.h"
#include "Core/ScriptRunner.h"
#include "Core/Session.h"
#include "Core/Settings.h"
#include "Core/Value.h"
#include "../HeckeExt/RegisterHeckeExtensionVerbs.h"
#include "../Util/ValidationCheck.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using console::HostProbe;
using console::Registry;
using console::Session;
namespace cost = console::cost;

oc_util::Checker Check("  [FAIL] ConsoleCostModelValidation: ", true, 40);

std::string SourceRoot() {
#ifdef OC_SOURCE_DIR
    return OC_SOURCE_DIR;
#else
    return ".";
#endif
}

bool ReadFile(const std::string& path, std::string& out) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

std::vector<std::string> SplitTab(const std::string& s) {
    std::vector<std::string> f;
    std::size_t a = 0;
    for (;;) {
        const std::size_t b = s.find('\t', a);
        if (b == std::string::npos) {
            f.push_back(s.substr(a));
            return f;
        }
        f.push_back(s.substr(a, b - a));
        a = b + 1;
    }
}

std::string Join(const std::vector<std::string>& v, const char* sep, const char* empty) {
    if (v.empty()) return empty;
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + v[i];
    return s;
}

double MaxRel = 0.0;
bool Close(const std::string& want, double got) {
    const double w = std::strtod(want.c_str(), nullptr);
    if (std::isnan(w) && std::isnan(got)) return true;
    const double rel = std::fabs(got - w) / std::max(std::fabs(w), 1e-300);
    if (w == got) return true;
    MaxRel = std::max(MaxRel, rel);
    return rel <= 1e-9;
}

std::vector<std::string> SplitComma(const std::string& s) {
    std::vector<std::string> out;
    std::size_t a = 0;
    for (;;) {
        const std::size_t b = s.find(',', a);
        if (b == std::string::npos) {
            out.push_back(s.substr(a));
            return out;
        }
        out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
}

// An environment variable pinned for a scope.
struct ScopedEnv {
    std::string name;
    bool had = false;
    std::string old;
    ScopedEnv(const char* n, const std::string& v) : name(n) {
        const char* o = std::getenv(n);
        had = o != nullptr;
        if (had) old = o;
        ::setenv(n, v.c_str(), 1);
    }
    ~ScopedEnv() {
        if (had) ::setenv(name.c_str(), old.c_str(), 1);
        else ::unsetenv(name.c_str());
    }
};

Registry& Reg() {
    static Registry* reg = [] {
        Registry* r = new Registry();
        console::RegisterPipelineVerbs(*r);
        hecke_ext::RegisterHeckeExtensionVerbs(*r);
        return r;
    }();
    return *reg;
}

struct Outcome {
    int rc = -1;
    std::string out;
};
Outcome RunC(const std::vector<std::string>& args, const std::string& stdin_text = std::string()) {
    std::istringstream in(stdin_text);
    std::ostringstream out;
    Outcome o;
    o.rc = console::RunConsole(Reg(), args, &in, &out);
    o.out = out.str();
    return o;
}

bool Contains(const std::string& h, const std::string& n) { return h.find(n) != std::string::npos; }

// --- (A) (B) ------------------------------------------------------------------
struct ExtraCounts {
    int x = 0, xd = 0, n = 0, s = 0;   // (S3 Fix round 1) X cells, their D rows, N (no model) rows, S (unusable estimate) rows
};
ExtraCounts Extra;

void ArmEstimates(const cost::CostModels& M, int& nE, int& nP, int& nD) {
    std::string text;
    const std::string path = SourceRoot() + "/src/hacs/Console/cost/expected_estimates.tsv";
    if (!Check(ReadFile(path, text), "(A) cannot read " + path)) return;
    const cost::DeviceProfile prof =
        cost::LoadDeviceProfile(SourceRoot() + "/src/hacs/Console/cost/test_device_profile.tsv");
    Check(prof.loaded && prof.factor.size() == 12, "(A) the fixture profile loads with 12 factor rows");
    std::map<std::string, cost::CostEstimate> byE;   // label|threads -> estimate, for (B)
    std::istringstream in(text);
    std::string line;
    struct Env {
        const char* name;
        double T, mem, cap, cf;
    };
    const Env envs[] = {{"big", 60, 77000, -1, 255}, {"small", 60, 4000, 2048, 255}, {"cwarn", 60, 77000, -1, 15},
                        {"crefuse", 60, 77000, -1, 8}, {"t0", 0, -1, -1, -1}};
    std::string last_label;
    cost::CostEstimate last;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> f = SplitTab(line);
        if ((f[0] == "E" || f[0] == "P" || f[0] == "X") && f.size() == 22) {
            const unsigned threads = static_cast<unsigned>(std::atoi(f[4].c_str()));
            const unsigned hw = static_cast<unsigned>(std::atoi(f[5].c_str()));
            const cost::CostEstimate e =
                cost::Estimate(M, f[2], SplitComma(f[3]), threads, f[0] == "P" ? &prof : nullptr, hw);
            const std::string id = "(" + std::string(f[0] == "E" ? "A" : f[0] == "X" ? "A/X" : "A/P") + ") " + f[1] + " " + f[3] +
                                   " T=" + f[4] + " hw=" + f[5];
            if (!Check(e.ok, id + ": no estimate: " + e.error)) continue;
            bool ok = std::to_string(e.T_eff) == f[6] && Close(f[7], e.wall_s) && Close(f[8], e.rss_mib) && Close(f[9], e.wall_band) &&
                      Close(f[10], e.rss_band) && Close(f[11], e.wall_hi) && Close(f[12], e.rss_hi) && Close(f[13], e.x_wall) &&
                      Close(f[14], e.x_rss) && Close(f[15], e.x_factor) && f[16] == (e.extrapolated ? "1" : "0") &&
                      f[17] == Join(e.x_features, ",", "-") && f[18] == Join(e.low_confidence, ",", "-") &&
                      Close(f[19], e.device_factor) && f[20] == e.calibration && f[21] == e.model;
            Check(ok, id + ": C++ differs from the Python reference (wall " + std::to_string(e.wall_s) + " / " + f[7] + ", wall_hi " +
                          std::to_string(e.wall_hi) + " / " + f[11] + ", rss_hi " + std::to_string(e.rss_hi) + " / " + f[12] +
                          ", x " + std::to_string(e.x_factor) + " / " + f[15] + ", flags " + Join(e.x_features, ",", "-") + " | " +
                          Join(e.low_confidence, ",", "-") + ", cal " + e.calibration + ")");
            if (f[0] == "E" || f[0] == "X") {
                ++(f[0] == "E" ? nE : Extra.x);
                last_label = f[1];
                last = e;
            } else {
                ++nP;
            }
        } else if (f[0] == "D" && f.size() == 6) {
            // D rows follow their E row
            if (!Check(f[1] == last_label, "(B) a D row without its E row: " + line)) continue;
            const Env* env = nullptr;
            for (const Env& v : envs)
                if (f[2] == v.name) env = &v;
            if (!Check(env != nullptr, "(B) unknown env " + f[2])) continue;
            const cost::CostDecision d = cost::Decide(M, last, env->T, env->mem, env->cap, env->cf);
            const bool ok = f[3] == cost::CostLevelName(d.level) && (f[4] == "-" ? d.kill_s < 0 : Close(f[4], d.kill_s)) &&
                            f[5] == Join(d.reasons, "; ", "-");
            Check(ok, "(B) " + f[1] + " env " + f[2] + ": " + cost::CostLevelName(d.level) + " [" + Join(d.reasons, "; ", "-") +
                          "] differs from " + f[3] + " [" + f[5] + "]");
            ++(f[1].rfind("x_", 0) == 0 ? Extra.xd : nD);
        } else if ((f[0] == "N" || f[0] == "S") && f.size() == 6) {
            // N: decide() without a model; S: decide() of the last X estimate with one field replaced (S3 audit M1)
            const Env* env = nullptr;
            for (const Env& v : envs)
                if (f[2] == v.name) env = &v;
            if (!Check(env != nullptr, "(B) unknown env " + f[2])) continue;
            cost::CostEstimate e = last;
            if (f[0] == "N") {
                e = cost::CostEstimate();
                e.error = "no model (test)";
            } else {
                const std::size_t eq = f[1].find('=');
                const std::string fld = f[1].substr(0, eq);
                const double val = std::strtod(f[1].c_str() + eq + 1, nullptr);
                (fld == "wall" ? e.wall_s : fld == "rss" ? e.rss_mib : fld == "wall_hi" ? e.wall_hi : e.rss_hi) = val;
            }
            const cost::CostDecision d = cost::Decide(M, e, env->T, env->mem, env->cap, env->cf);
            const bool ok = f[3] == cost::CostLevelName(d.level) && (f[4] == "-" ? d.kill_s < 0 : Close(f[4], d.kill_s)) &&
                            f[5] == Join(d.reasons, "; ", "-");
            Check(ok, "(B) " + f[0] + " " + f[1] + " env " + f[2] + ": " + cost::CostLevelName(d.level) + " [" +
                          Join(d.reasons, "; ", "-") + "] differs from " + f[3] + " [" + f[5] + "]");
            Check(f[0] == "S" ? d.level == cost::CostLevel::Refuse
                              : (d.level == cost::CostLevel::Refuse) == (env->mem > 0 || env->cap > 0 || (env->cf >= 0 && env->cf <= 10)),
                  "(B) " + f[0] + " " + f[1] + " env " + f[2] + ": rule 1 is not bypassed (audit M1)");
            ++(f[0] == "N" ? Extra.n : Extra.s);
        } else {
            Check(false, "(A) malformed row: " + line.substr(0, 80));
        }
    }
    Check(nE == 213 && nP == 72 && nD == 1065,
          "(A)(B) row counts E/P/D = " + std::to_string(nE) + "/" + std::to_string(nP) + "/" + std::to_string(nD) + " (want 213/72/1065)");
    Check(Extra.x == 3 && Extra.xd == 15 && Extra.n == 5 && Extra.s == 100,
          "(A)(B) row counts X/XD/N/S = " + std::to_string(Extra.x) + "/" + std::to_string(Extra.xd) + "/" + std::to_string(Extra.n) +
              "/" + std::to_string(Extra.s) + " (want 3/15/5/100)");
}

// --- (C) --------------------------------------------------------------------
void ArmModelFile(const cost::CostModels& M) {
    std::string disk;
    const std::string path = SourceRoot() + "/src/hacs/Console/cost/cost_models.tsv";
    if (Check(ReadFile(path, disk), "(C) cannot read " + path))
        Check(disk == cost::EmbeddedCostModelsText(),
              "(C) the embedded model text differs from the tracked cost_models.tsv (re-run the CMake configure step)");
    struct G {
        const char *verb, *var, *reg;
    };
    const G groups[] = {{"descent", "-", "N1"},          {"descent", "-", "n0"},          {"descent", "-", "gen"},
                        {"hecke_structure", "full", "N1"}, {"hecke_structure", "full", "n0"}, {"hecke_structure", "full", "gen"},
                        {"hecke_structure", "anemic", "n0"}, {"hecke_structure", "anemic", "gen"}, {"level_n", "-", "all"},
                        {"nebentypus", "-", "all"},      {"forms_algebra", "-", "all"},   {"hida_lambda", "-", "all"},
                        {"hida_tower", "-", "all"}};
    for (const G& g : groups)
        for (const char* q : {"wall", "rss"}) {
            const std::string k = std::string(g.verb) + "|" + g.var + "|" + g.reg + "|" + q;
            Check(M.slots.count(k) == 1, "(C) no " + std::string(q) + " model for " + g.verb + " " + g.var + " " + g.reg);
        }
    for (const char* v : {"descent", "hecke_structure", "level_n", "nebentypus", "forms_algebra", "hida_lambda", "hida_tower"})
        Check(M.klass.count(v) == 1, std::string("(C) no class row for ") + v);
    Check(M.safety.count("wall") == 1 && M.safety.count("rss") == 1, "(C) the two safety rows");
    Check(M.safety.count("wall") && M.safety.at("wall").x_exp == 2.0 && M.safety.at("wall").strong_x == 2.0 &&
              M.safety.count("rss") && M.safety.at("rss").x_exp == 1.0,
          "(C) safety: wall x^2, rss x^1, strong at x > 2 (S2 FR.2)");
    Check(!M.calib_set.empty() && M.calib.size() == 20, "(C) the calib rows: one set, 10 classes x 2 slots (" + M.calib_set + ", " +
                                                             std::to_string(M.calib.size()) + ")");
    Check(!M.fitted_on.empty() && M.fitted_on.front() == "fb7f58b9", "(C) fitted_on names the S2 probe code fb7f58b9");
}

// --- (D) --------------------------------------------------------------------
void ArmSettings() {
    Session s(Reg(), false);
    const console::SettingSpec* c = s.settings().find_spec(console::kSettingConfirm);
    const console::SettingSpec* w = s.settings().find_spec(console::kSettingConfirmWallS);
    if (!Check(c != nullptr && w != nullptr, "(D) confirm and confirm_wall_s are registered")) return;
    Check(c->family == console::SettingFamily::Engine && w->family == console::SettingFamily::Engine,
          "(D) both are engine-family (never mathematical)");
    Check(c->default_value.is_string() && c->default_value.string_value() == "ask", "(D) confirm defaults to ask");
    Check(w->default_value.is_int() && w->default_value.int_value() == 60, "(D) confirm_wall_s defaults to 60 (owner Q1)");
    const console::SettingsSnapshot m0 = s.settings().snapshot().mathematical();
    const Outcome bad1 = RunC({"-c", "set confirm = \"maybe\""});
    Check(bad1.rc == 3 && Contains(bad1.out, "confirm"), "(D) confirm = \"maybe\" is refused (exit 3): " + bad1.out);
    const Outcome bad2 = RunC({"-c", "set confirm_wall_s = 0 - 1"});
    Check(bad2.rc == 3 && Contains(bad2.out, "must be >= 0"), "(D) confirm_wall_s = -1 is refused: " + bad2.out);
    const Outcome bad3 = RunC({"-c", "print(1)", "--set", "confirm=sometimes"});
    Check(bad3.rc == 1, "(D) --set confirm=sometimes is a usage error (exit 1)");
    s.settings().set(console::kSettingConfirm, console::Value::from_string("yes"), console::SourcePos());
    s.settings().set(console::kSettingConfirmWallS, console::Value::from_int(5), console::SourcePos());
    Check(s.settings().snapshot().mathematical().equals(m0), "(D) setting them leaves the mathematical snapshot unchanged");
    const Outcome get = RunC({"-c", "print(settings().confirm.value, settings().confirm_wall_s.value, settings().confirm.family)"});
    Check(get.rc == 0 && Contains(get.out, "ask 60 engine"), "(D) settings() shows them: " + get.out);
}

// --- (E) --------------------------------------------------------------------
std::string Unescape(const std::string& s) {
    std::string o;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == 'n') {
            o += '\n';
            ++i;
        } else {
            o += s[i];
        }
    }
    return o;
}

HostProbe Hostile() {   // a terminal, almost no memory, C: nearly full, 1 CPU: must change nothing prompt-less
    HostProbe h;
    h.prompt_tty = [] { return true; };
    h.write_prompt = [](const std::string&) {};
    h.mem_available_mib = [] { return 1.0; };
    h.c_free_gb = [] { return 5.0; };
    h.cpu_count = [] { return 1u; };
    return h;
}

int ArmByteIdentity() {
    std::string text;
    const std::string path = SourceRoot() + "/src/hacs/Console/cost/refusals_pre_s3.tsv";
    if (!Check(ReadFile(path, text), "(E) cannot read " + path)) return 0;
    std::istringstream in(text);
    std::string line;
    int n = 0;
    const std::string fixture = SourceRoot() + "/src/hacs/Console/cost/test_device_profile.tsv";
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> f = SplitTab(line);
        if (!Check(f.size() == 3, "(E) malformed row")) continue;
        const std::string want = Unescape(f[2]);
        const int rc = std::atoi(f[1].c_str());
        ++n;
        const Outcome a = RunC({"-c", f[0]});
        Check(a.rc == rc && a.out == want, "(E) -c '" + f[0] + "': exit " + std::to_string(a.rc) + ", got\n" + a.out + "want\n" + want);
        const Outcome b = RunC({"-c", f[0], "--set", "confirm=no"});
        Check(b.rc == rc && b.out == want, "(E) confirm=no -c '" + f[0] + "' differs from the pre-S3 text");
        {
            console::ScopedHostProbe hp(Hostile());
            ScopedEnv pe("HACS_DEVICE_PROFILE", fixture);
            const Outcome c = RunC({"-c", f[0]});
            Check(c.rc == rc && c.out == want, "(E) hostile host -c '" + f[0] + "' differs from the pre-S3 text: " + c.out);
            // the script form: the same text with the file name of the script
            const Outcome d = RunC({"-"}, f[0] + "\n");
            std::string want_d = want;
            for (std::size_t p = want_d.find("<string>"); p != std::string::npos; p = want_d.find("<string>"))
                want_d.replace(p, 8, "<stdin>");
            Check(d.rc == rc && d.out == want_d, "(E) hostile host, script on stdin '" + f[0] + "' differs: " + d.out);
            // a piped REPL (its input is not std::cin): refuses with the same message, the REPL survives
            const Outcome r = RunC({"--console"}, f[0] + "\nprint(\"after\")\n");
            const std::string first = want_d.substr(0, want_d.find('\n'));
            Check(r.rc == 0 && Contains(r.out, first) && Contains(r.out, "after") && !Contains(r.out, "not run"),
                  "(E) piped REPL '" + f[0] + "' does not refuse as before: " + r.out);
        }
    }
    Check(n == 41, "(E) 41 frozen pre-S3 refusals (" + std::to_string(n) + ")");
    return n;
}

// --- (F) --------------------------------------------------------------------
void ArmConfirmYes() {
    const char* runs[] = {"x = forms_algebra(402, 7); print(x.k)", "x = hida_lambda(103, 12, 2); print(x.ell)",
                          "x = level_n(101, 2, 7); print(x.N)", "x = nebentypus(31, 1, 2, 7, [17]); print(x.N)"};
    const char* want[] = {"402", "103", "101", "31"};
    for (int i = 0; i < 4; ++i) {
        const Outcome o = RunC({"-c", runs[i], "--set", "confirm=yes"});
        Check(o.rc == 0 && Contains(o.out, std::string("\n") + want[i] + "\n") && Contains(o.out, "#~ confirm = yes:"),
              std::string("(F) confirm=yes runs '") + runs[i] + "': exit " + std::to_string(o.rc) + ": " + o.out);
        const Outcome s = RunC({"-c", std::string("set confirm = \"yes\"; ") + runs[i]});
        Check(s.rc == 0, std::string("(F) `set confirm = \"yes\"` in the script runs '") + runs[i] + "'");
    }
    {
        HostProbe h;
        h.mem_available_mib = [] { return 10.0; };
        console::ScopedHostProbe hp(h);
        const Outcome o = RunC({"-c", "x = forms_algebra(402, 7)", "--set", "confirm=yes"});
        Check(o.rc == 3 && Contains(o.out, "refused before the run") && Contains(o.out, "MemAvailable"),
              "(F) hard memory refusal under confirm=yes (MemAvailable 10 MiB): " + o.out);
        const Outcome in = RunC({"-c", "x = forms_algebra(400, 7); print(x.k)", "--set", "confirm=yes"});
        Check(in.rc == 0, "(F) inside the envelope nothing is estimated, even at 10 MiB MemAvailable");
    }
    {
        HostProbe h;
        h.c_free_gb = [] { return 9.0; };
        console::ScopedHostProbe hp(h);
        const Outcome o = RunC({"-c", "x = forms_algebra(402, 7)", "--set", "confirm=yes"});
        Check(o.rc == 3 && Contains(o.out, "free space on C:"), "(F) hard disk refusal (C: 9 GB) under confirm=yes: " + o.out);
    }
    {   // (S3 Fix round 1, audit M1) descent(10^6, 0) under confirm = yes: refused at T = 32 (the ln T line went negative) and 16
        HostProbe h;
        h.mem_available_mib = [] { return 77000.0; };
        h.c_free_gb = [] { return 255.0; };
        h.cpu_count = [] { return 32u; };
        console::ScopedHostProbe hp(h);
        for (const char* t : {"32", "16"}) {
            const Outcome o = RunC({"-c", std::string("set coefficient_ring = \"Z\"; set threads = ") + t + "; r = descent(1000000, 0); print(1)",
                                    "--set", "confirm=yes"});
            Check(o.rc == 3 && Contains(o.out, "refused before the run") && Contains(o.out, "estimated peak RSS up to") &&
                      !Contains(o.out, "~ ?") && !Contains(o.out, "\n1\n"),
                  std::string("(F) descent(1000000, 0) at threads = ") + t + " is hard-refused under confirm=yes: " + o.out);
        }
    }
    {   // (S3 Fix round 1, audit m1) N >= 2^31: a TypeError by name in the yes mode, the pre-S3 bound text prompt-less
        const char* calls[] = {"level_n(2147483648, 2, 7)", "nebentypus(2147483648, 1, 2, 7)"};
        for (const char* c : calls) {
            const Outcome y = RunC({"-c", c, "--set", "confirm=yes"});
            Check(y.rc == 3 && Contains(y.out, "type error: ") && Contains(y.out, "N must be < 2^31 (word size of the level code)") &&
                      !Contains(y.out, "ENGINE_CHECK failed"),
                  std::string("(F) confirm=yes ") + c + " is a TypeError by name: " + y.out);
            const Outcome n = RunC({"-c", c});
            Check(n.rc == 3 && Contains(n.out, "N must satisfy 1 <= N <= 100") && !Contains(n.out, "2^31"),
                  std::string("(F) prompt-less ") + c + " keeps the pre-S3 text: " + n.out);
        }
    }
    {   // the descent memory hint: fewer threads lower the peak RSS
        HostProbe h;
        h.mem_available_mib = [] { return 3000.0; };
        h.cpu_count = [] { return 16u; };
        console::ScopedHostProbe hp(h);
        const Outcome o = RunC({"-c", "set coefficient_ring = \"Z\"; set threads = 16; r = descent(1, 200)", "--set", "confirm=yes"});
        Check(o.rc == 3 && Contains(o.out, "fewer threads lower the peak RSS"), "(F) the descent refusal names threads = 1: " + o.out);
    }
}

// --- (G) --------------------------------------------------------------------
struct TtyRun {
    int rc = -1;
    std::string out, prompts;
    bool bound = false;
};
TtyRun RunTty(const std::string& code, const std::string& answers, HostProbe h = HostProbe()) {
    TtyRun t;
    auto prompts = std::make_shared<std::string>();
    h.write_prompt = [prompts](const std::string& s) { *prompts += s; };
    console::ScopedHostProbe hp(h);
    Session s(Reg(), false);
    std::ostringstream os;
    s.set_output(os);
    std::istringstream ans(answers);
    s.set_confirm_input(&ans);
    t.rc = console::RunSource(s, code, "<string>");
    t.out = os.str();
    t.prompts = *prompts;
    t.bound = s.find_var("x") != nullptr;
    return t;
}

void ArmTty() {
    {
        const TtyRun t = RunTty("x = forms_algebra(402, 7)", "n\n");
        Check(t.rc == 0 && t.bound && t.prompts.empty(), "(G) rule 4: a cheap just-over cell runs silently on the terminal");
    }
    {
        const TtyRun t = RunTty("set confirm_wall_s = 0; x = forms_algebra(402, 7)", "n\n");
        Check(t.rc == 3 && !t.bound && Contains(t.out, "not run") && Contains(t.prompts, "run it? [y/N]") &&
                  Contains(t.prompts, "outside its silent envelope") && Contains(t.prompts, "model forms_algebra_cost/v1") &&
                  Contains(t.prompts, "uncalibrated") && !Contains(t.prompts, "STRONG"),
              "(G) normal prompt, declined: nothing bound: " + t.out + " | " + t.prompts);
    }
    {
        const TtyRun t = RunTty("set confirm_wall_s = 0; x = forms_algebra(402, 7)", "Y\n");
        Check(t.rc == 0 && t.bound, "(G) normal prompt, accepted: runs");
    }
    {
        const TtyRun t = RunTty("set confirm_wall_s = 0; x = forms_algebra(402, 7)", "");
        Check(t.rc == 3 && !t.bound, "(G) end of input at the prompt = declined");
    }
    HostProbe cwarn;
    cwarn.c_free_gb = [] { return 15.0; };
    {
        const TtyRun t = RunTty("x = forms_algebra(402, 7)", "y\nno\n", cwarn);
        Check(t.rc == 3 && !t.bound && Contains(t.prompts, "STRONG WARNING") && Contains(t.prompts, "C: <= 20 GB free") &&
                  Contains(t.prompts, "type `yes`"),
              "(G) strong (C: 15 GB): asked twice, the second answer must be `yes`: " + t.prompts);
    }
    {
        const TtyRun t = RunTty("x = forms_algebra(402, 7)", "y\nyes\n", cwarn);
        Check(t.rc == 0 && t.bound, "(G) strong, `y` then `yes`: runs");
    }
    {
        const TtyRun t = RunTty("x = forms_algebra(20000, 7)", "y\ny\n");
        Check(t.rc == 3 && !t.bound && Contains(t.prompts, "wall_hi > 1 h") && Contains(t.prompts, "extrapolated"),
              "(G) strong (k = 20000: > 1 h, extrapolated), a second `y` is not `yes`: " + t.prompts);
    }
    {
        const TtyRun t = RunTty("set confirm = \"no\"; x = forms_algebra(402, 7)", "y\n");
        Check(t.rc == 3 && !t.bound && t.prompts.empty() &&
                  Contains(t.out, "type error: forms_algebra(k, ell, primes, terms): k must be even with 12 <= k <= 400"),
              "(G) confirm = no on the terminal refuses as before: " + t.out);
    }
    {
        HostProbe h;
        h.c_free_gb = [] { return 8.0; };
        const TtyRun t = RunTty("set confirm_wall_s = 0; x = forms_algebra(402, 7)", "y\nyes\n", h);
        Check(t.rc == 3 && !t.bound && t.prompts.empty() && Contains(t.out, "refused before the run"),
              "(G) a hard refusal never asks: " + t.out);
    }
    {   // a math refusal after a crossed bound reports the math reason on the terminal (it would refuse anyway)
        const TtyRun t = RunTty("x = forms_algebra(402, 4)", "y\n");
        Check(t.rc == 3 && Contains(t.out, "ell must be a prime") && t.prompts.empty(), "(G) a refusal is never asked about");
        const TtyRun u = RunTty("x = level_n(101, 3, 7)", "y\n");
        Check(u.rc == 3 && Contains(u.out, "n must be even") && u.prompts.empty(),
              "(G) on the terminal the precondition, not the crossed bound, is reported: " + u.out);
    }
    {   // the fixture profile names another binary: it is stale, not applied (factor 1), and the prompt says so
        ScopedEnv pe("HACS_DEVICE_PROFILE", SourceRoot() + "/src/hacs/Console/cost/test_device_profile.tsv");
        const TtyRun t = RunTty("set confirm_wall_s = 0; x = forms_algebra(402, 7)", "n\n");
        Check(Contains(t.prompts, "stale calibration (") && Contains(t.prompts, "re-measures it"),
              "(G) a stale profile is named in the prompt and not applied: " + t.prompts);
    }
}

// --- (H) --------------------------------------------------------------------
std::string SelfExe() { return console::host::SelfExe(); }

// Runs `exe --console` on a pseudo-terminal, feeding `input`; returns everything the terminal showed.
bool RunPty(const std::string& input, std::string& shown, int& rc) {
    const int m = ::posix_openpt(O_RDWR | O_NOCTTY);
    if (m < 0 || ::grantpt(m) != 0 || ::unlockpt(m) != 0) return false;
    const char* sn = ::ptsname(m);
    if (sn == nullptr) return false;
    const std::string slave = sn;
    const std::string exe = SelfExe();
    const pid_t pid = ::fork();
    if (pid < 0) return false;
    if (pid == 0) {
        ::setsid();
        const int s = ::open(slave.c_str(), O_RDWR);
        if (s < 0) ::_exit(126);
        ::dup2(s, 0);
        ::dup2(s, 1);
        ::dup2(s, 2);
        if (s > 2) ::close(s);
        ::close(m);
        ::setenv("HACS_DEVICE_PROFILE", "none", 1);
        ::setenv("HACS_HOST_PROBE_FIXTURE", "1", 1);   // (S3 Fix round 1, audit m2) MemAvailable 4096 MiB, C: 255 GB
        ::setenv("ENGINE_CUDA_DISABLE", "1", 1);
        ::execl(exe.c_str(), exe.c_str(), "--console", static_cast<char*>(nullptr));
        ::_exit(127);
    }
    if (::write(m, input.data(), input.size()) != static_cast<ssize_t>(input.size())) {
        ::kill(pid, SIGKILL);
    }
    const auto t0 = std::chrono::steady_clock::now();
    char buf[4096];
    for (;;) {
        struct pollfd p{m, POLLIN, 0};
        const int r = ::poll(&p, 1, 1000);
        if (r > 0) {
            const ssize_t n = ::read(m, buf, sizeof buf);
            if (n <= 0) break;   // EIO once the child has exited and closed the slave
            shown.append(buf, static_cast<std::size_t>(n));
        }
        int st = 0;
        if (r == 0 && ::waitpid(pid, &st, WNOHANG) == pid) {
            rc = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
            ::close(m);
            return true;
        }
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(90)) {
            ::kill(pid, SIGKILL);
            break;
        }
    }
    int st = 0;
    ::waitpid(pid, &st, 0);
    rc = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    ::close(m);
    return true;
}

void ArmPty() {
    std::string shown;
    int rc = -1;
    const std::string input =
        "set confirm_wall_s = 0\n"
        "x = forms_algebra(402, 7)\n"
        "n\n"
        "print(x)\n"
        "y2 = forms_algebra(402, 7)\n"
        "y\n"
        "print(\"k=\", y2.k)\n"
        "exit(0)\n";
    if (!Check(RunPty(input, shown, rc), "(H) cannot open a pseudo-terminal")) return;
    Check(rc == 0, "(H) the terminal REPL exits 0 (exit " + std::to_string(rc) + ")");
    Check(Contains(shown, "run it? [y/N]"), "(H) the prompt appears on the terminal");
    Check(Contains(shown, "not run"), "(H) a declined prompt prints `not run`");
    Check(Contains(shown, "x") && (Contains(shown, "undefined") || Contains(shown, "unknown") || Contains(shown, "not defined")),
          "(H) the declined binding stays unset");
    Check(Contains(shown, "k= 402"), "(H) an accepted prompt runs the call");
    if (Check.fails) std::cout << "  (H) terminal transcript:\n" << shown << "\n";
}

// --- (I) --------------------------------------------------------------------
void ArmTypedRefusals() {
    const Outcome a = RunC({"-c", "set coefficient_ring = \"Z\"; hecke_structure(1, 0)"});
    Check(a.rc == 3 && Contains(a.out, "type error: hecke_structure(N, n, algebra): at N = 1, n = 0 the cusp space is 0") &&
              !Contains(a.out, "ENGINE_CHECK failed"),
          "(I) hecke_structure(1, 0) is a TypeError: " + a.out);
    const Outcome b = RunC({"-c", "hida_tower(100, 23, 1)"});
    Check(b.rc == 3 && Contains(b.out, "type error: hida_tower(n, ell, m): m + e_max = K = 1 < 2") && Contains(b.out, "def:hidatower") &&
              !Contains(b.out, "ENGINE_CHECK failed"),
          "(I) hida_tower(100, 23, 1) is a TypeError after the pre-pass: " + b.out);
    const Outcome c = RunC({"-c", "x = hida_tower(10, 5, 2); print(x.K)"});
    Check(c.rc == 0, "(I) a K >= 2 cell still runs: " + c.out);
}

// --- (J) --------------------------------------------------------------------
void ArmProfile() {
    {
        ScopedEnv a("HACS_DEVICE_PROFILE", "/x/p.tsv");
        Check(cost::DeviceProfilePath() == "/x/p.tsv", "(J) HACS_DEVICE_PROFILE wins");
    }
    {
        ScopedEnv a("HACS_DEVICE_PROFILE", "");
        ScopedEnv b("XDG_CONFIG_HOME", "/cfg");
        Check(cost::DeviceProfilePath() == "/cfg/hacs/device_profile.tsv", "(J) then XDG_CONFIG_HOME");
    }
    Check(!cost::LoadDeviceProfile("none").loaded, "(J) `none` loads nothing");
    Check(!cost::LoadDeviceProfile("/nonexistent/profile.tsv").loaded, "(J) a missing file loads nothing");
    cost::DeviceProfile p;
    p.loaded = true;
    p.kv = {{"calib_set", "calib/v2"}, {"binary_build_id", "abcd"}, {"binary_size", "10"}, {"binary_mtime", "20"}};
    Check(cost::ProfileStaleReason(p, "calib/v2", "abcd", 99, 99, true).empty(), "(J) same build id: fresh (a copy stays fresh)");
    Check(!cost::ProfileStaleReason(p, "calib/v2", "abce", 10, 20, true).empty(), "(J) other build id: stale (an in-place rebuild)");
    Check(!cost::ProfileStaleReason(p, "calib/v1", "abcd", 10, 20, true).empty(), "(J) other calib set: stale");
    Check(cost::ProfileStaleReason(p, "calib/v2", "", 10, 20, true).empty(), "(J) no build id: size + mtime decide (equal)");
    Check(!cost::ProfileStaleReason(p, "calib/v2", "", 10, 21, true).empty(), "(J) no build id: size + mtime decide (differ)");
    const std::string bid = console::host::ElfBuildId(SelfExe());
    bool hex = !bid.empty();
    for (char ch : bid) hex = hex && std::isxdigit(static_cast<unsigned char>(ch));
    Check(hex && bid.size() >= 16, "(J) the running binary carries an ELF build id (" + bid + ")");
    Check(console::host::AffinityCpuCount() >= 1, "(J) the affinity CPU count");
    {   // (S3 Fix round 1, audit m5) --calibrate --profile refuses a path inside the source tree, before any probe runs
        for (const std::string& pth : {SourceRoot() + "/device_profile.tsv", SourceRoot() + "/no/such/dir/p.tsv"}) {
            std::vector<std::string> args{"--calibrate", "--profile", pth};
            std::vector<char*> argv;
            for (std::string& x : args) argv.push_back(&x[0]);
            std::ostringstream cap;
            std::streambuf* old = std::cout.rdbuf(cap.rdbuf());
            const int rc = console::calibrate::RunCalibrate(static_cast<int>(argv.size()), argv.data());
            std::cout.rdbuf(old);
            Check(rc == 1 && Contains(cap.str(), "REFUSED") && Contains(cap.str(), "inside the source tree"),
                  "(J) --calibrate --profile " + pth + " is REFUSED (exit 1): exit " + std::to_string(rc) + ": " + cap.str());
        }
    }
}

} // namespace

uint32_t ConsoleCostModelValidation() {
    std::cout << "--- ConsoleCostModelValidation (NBCONFIRM S3: estimator = Python reference, policy per mode) ---" << std::endl;
    Check.reset();
    MaxRel = 0.0;
    Extra = ExtraCounts();
    ScopedEnv pin("HACS_DEVICE_PROFILE", "none");   // no user profile can change a verdict (S2 audit F5)
    console::ScopedHostProbe no_host(HostProbe{});
    const cost::CostModels& M = cost::EmbeddedCostModels();
    int nE = 0, nP = 0, nD = 0;
    ArmEstimates(M, nE, nP, nD);
    const uint32_t after_ab = Check.fails;
    ArmModelFile(M);
    ArmSettings();
    const int nref = ArmByteIdentity();
    ArmConfirmYes();
    ArmTty();
    const uint32_t before_h = Check.fails;
    ArmPty();
    const bool pty_ok = Check.fails == before_h;
    ArmTypedRefusals();
    ArmProfile();
    char rel[32];
    std::snprintf(rel, sizeof rel, "%.2e", MaxRel);
    std::cout << "  CONSOLE_COST_MODEL estimates=" << nE << "+" << nP << " decisions=" << nD << " max_rel=" << rel
              << " ab_fails=" << after_ab << " refusals_pre_s3=" << nref << "x5 pty=" << (pty_ok ? "OK" : "FAIL")
              << " beyond=" << Extra.x << "+" << Extra.xd << " no_model=" << Extra.n << " unusable=" << Extra.s
              << " calib_set=" << M.calib_set << " verdict=" << (Check.fails ? "FAIL" : "OK") << std::endl;
    std::cout << "  " << Check.summary("ConsoleCostModelValidation") << std::endl;
    return Check.exit_code();
}
