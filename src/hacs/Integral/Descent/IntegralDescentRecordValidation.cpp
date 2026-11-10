// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D6: IntegralDescentRecordValidation -- see IntegralDescentRecordValidation.h.  The oracles are the D1 / D3 /
// D4 tables outside the build (docs/implementation/slot24/{descentN_oracle,descent_oracle_n128,d3_hecke_oracle,d4_pair_oracle}.tsv),
// each produced by a Python script that shares no code with cohomology_app.

#include "IntegralDescentRecordValidation.h"

#include "DescentHecke.h"
#include "DescentRecord.h"
#include "DescentTree.h"
#include "DescentTreePair.h"

#include "../../Campaign/CampaignDescent.h"
#include "../../Campaign/CampaignDriver.h"
#include "../../Campaign/CampaignSpec.h"
#include "../../EquivariantCohomologyOnK.h"
#include "../../Golden/GoldenHarness.h"
#include "../../Hecke/HeckeCosetTables.h"
#include "../../Hecke/HeckeOperatorOnK.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../Hecke/HeckeTauTable.h"
#include "../../Hecke/SL2ZTessellationTables.h"
#include "../../LocalSystemSL2Z.h"
// (NB-TOURFIX) the harness only: arm (Z) checks the console's factored char_Z against the NumberField factoriser; the
// Integral library itself stays bignum-free.
#include "../../NumberField/IntPolyFactor.h"
#include "../../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace {

namespace rec = integral_descent::record;
using integral_model::AbelianGroupType;
using clk = std::chrono::steady_clock;

uint32_t g_checks = 0, g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  FAIL " << what << std::endl;
    }
    return ok;
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

std::string src_path(const std::string& rel) {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/" + rel;
#else
    return rel;
#endif
}

template <class F>
bool throws_niitb(F&& f, const std::string& must_contain) {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild& e) {
        return std::string(e.what()).find(must_contain) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}
template <class F>
bool throws_invalid(F&& f, const std::string& must_contain = "") {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find(must_contain) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}
template <class F>
bool throws_runtime(F&& f, const std::string& must_contain = "") {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild&) {
        return false;
    } catch (const std::runtime_error& e) {
        return std::string(e.what()).find(must_contain) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}

// The oracles' group form: "Z^r" (omitted if r = 0), then "Z/p^k" (k omitted if 1) ascending in (p, k), '+'-joined; "0".
std::string OracleForm(const AbelianGroupType& G) {
    std::vector<std::string> parts;
    if (G.free_rank > 0) parts.push_back("Z^" + std::to_string(G.free_rank));
    for (const auto& [p, exps] : G.torsion) {
        std::vector<uint32_t> a(exps.begin(), exps.end());
        std::sort(a.begin(), a.end());
        for (uint32_t k : a) parts.push_back("Z/" + std::to_string(p) + (k > 1 ? "^" + std::to_string(k) : ""));
    }
    if (parts.empty()) return "0";
    std::string s;
    for (size_t i = 0; i < parts.size(); ++i) s += (i ? "+" : "") + parts[i];
    return s;
}

std::vector<std::string> SplitTab(const std::string& line) {
    std::vector<std::string> f;
    std::string cur;
    for (char c : line) {
        if (c == '\t') {
            f.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur += c;
        }
    }
    f.push_back(cur);
    return f;
}

struct Oracles {
    bool ok = false;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::string> groups;                       // (N, n, q)
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::string> chi;                          // (N, n, p) -> "1,-9"
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>, std::string> tors;     // (N, n, q, ell, p)
    std::map<std::tuple<uint32_t, uint32_t, std::string, uint32_t>, std::string> pair;            // (N, n, obj, q)
};
const Oracles& TheOracles() {
    static const Oracles O = [] {
        Oracles o;
        auto U = [](const std::string& s) { return static_cast<uint32_t>(std::stoul(s)); };
        size_t files = 0;
        {
            std::ifstream in(src_path("docs/implementation/slot24/descentN_oracle.tsv"));
            std::string line;
            files += in ? 1 : 0;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#' || line[0] == 'N') continue;
                const auto f = SplitTab(line);
                if (f.size() >= 4) o.groups[{U(f[0]), U(f[1]), U(f[2])}] = f[3];
            }
        }
        {
            std::ifstream in(src_path("docs/implementation/slot24/descent_oracle_n128.tsv"));
            std::string line;
            files += in ? 1 : 0;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#' || line[0] == 'n') continue;
                const auto f = SplitTab(line);
                if (f.size() >= 3) o.groups.emplace(std::make_tuple(1u, U(f[0]), U(f[1])), f[2]);
            }
        }
        {
            std::ifstream in(src_path("docs/implementation/slot24/d3_hecke_oracle.tsv"));
            std::string line;
            files += in ? 1 : 0;
            while (std::getline(in, line)) {
                const auto f = SplitTab(line);
                if (f.size() >= 5 && f[0] == "CHI") o.chi[{U(f[1]), U(f[2]), U(f[3])}] = f[4];
                if (f.size() >= 7 && f[0] == "TORS") o.tors[{U(f[1]), U(f[2]), U(f[3]), U(f[4]), U(f[5])}] = f[6];
            }
        }
        {
            std::ifstream in(src_path("docs/implementation/slot24/d4_pair_oracle.tsv"));
            std::string line;
            files += in ? 1 : 0;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#' || line[0] == 'N' || line.rfind("HECKE", 0) == 0) continue;
                const auto f = SplitTab(line);
                if (f.size() >= 5) o.pair[{U(f[0]), U(f[1]), f[2], U(f[3])}] = f[4];
            }
        }
        o.ok = files == 4 && !o.groups.empty() && !o.chi.empty() && !o.tors.empty() && !o.pair.empty();
        return o;
    }();
    return O;
}

int run_console(const std::string& code) {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return -1;
    buf[n] = 0;
    const std::string cmd = std::string(buf) + " -c '" + code + "' >/dev/null 2>&1";
    const int st = std::system(cmd.c_str());
    if (st == -1 || !WIFEXITED(st)) return -1;
    return WEXITSTATUS(st);
}
// The same with an environment prefix ("VAR=0 "), returning the child's stdout ("" when it fails).
std::string run_console_out(const std::string& env, const std::string& code) {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return "";
    buf[n] = 0;
    const std::string cmd = env + std::string(buf) + " -c '" + code + "' 2>/dev/null";
    FILE* f = ::popen(cmd.c_str(), "r");
    if (f == nullptr) return "";
    std::string out;
    char line[512];
    while (std::fgets(line, sizeof line, f) != nullptr) out += line;
    const int st = ::pclose(f);
    if (st == -1 || !WIFEXITED(st) || WEXITSTATUS(st) != 0) return "";
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out;
}

// (D6 audit m1) The console verbs that compute over F_ell from scalar arguments and never read the ring: each is wrapped by
// console::bind::GateVerbsOnField and must refuse Z and Z/ell^k at its statement (exit 5).  One cheap call per verb.
const std::vector<std::string>& FieldVerbCalls() {
    static const std::vector<std::string> calls = {
        "dickson_exceptional(10, 691)", "dickson_count(10, 691)", "s1_scalar(2, 7)", "s1_unipotent(7)", "cylinder_scalar(2, 7)",
        "cylinder_unipotent(7)", "moebius_scalar(2, 7)", "moebius_unipotent(7)", "sturm_bound(12, 691)", "sturm_bound(12, 691, 0)",
        "u_ell_decision(10, 691)", "u_ell_decision(10, 691, 100)", "bernoulli_mod(12, 691)", "glover_factors(10, 691)",
        "lfunction(10, 691, 199)", "kummer_unit(37, 32)", "hida_tower(10, 5, 2)", "level_n(11, 2, 7)", "nebentypus(5, 1, 2, 11)",
        "forms_algebra(12, 691)", "hida_lambda(5, 12, 2)",
        // the verbs that refused already (their own RequireField / CoefficientRingK(who)), probed so the closure is one list
        "symmetric_power(sl2z_model(), 10, 691)", "ext_record(10, 691)", "atlas(5)", "dickson_les(10, 691)", "hecke_ring_algebra(10, 691)",
    };
    return calls;
}

// ---- (S) the setting ------------------------------------------------------------------------------------------------
void ArmSetting() {
    hecke::MathSettings s;
    bool fine = false;
    try {
        fine = s.set("coefficient_ring", "Z", hecke::SettingSource::Spec);
    } catch (...) {
        fine = false;
    }
    check(fine && s.coefficient_ring.is_integral() && s.coefficient_ring.k == 0, "(S) `coefficient_ring = Z` parses (k = 0): un-reserved");
    check(s.text("coefficient_ring") == "Z" && s.canonical_text() == "coefficient_ring = Z\n", "(S) its canonical text is `coefficient_ring = Z`");
    check(s.fingerprint() == engine::FnvBytes(engine::kFnvOffset64, "coefficient_ring = Z\n") && s.fingerprint() != 0,
          "(S) the fingerprint is FNV-1a-64 of the canonical text, non-zero");
    check(s.to_json().find("\"Z\"") != std::string::npos, "(S) the settings block records the value Z");
    for (const std::string& res : {std::string("Z/25"), std::string("Z/7")}) {
        hecke::MathSettings t;
        check(throws_niitb([&] { t.set("coefficient_ring", res, hecke::SettingSource::Spec); }, "coefficient_ring=" + res) && t.is_default(),
              "(S) the general `" + res + "` stays reserved (refused inside set(), never stored)");
    }
    {
        const hecke::ScopedCoefficientRing z(0);
        check(hecke::CoefficientRingIsZ(), "(S) ScopedCoefficientRing(0) installs Z");
        check(throws_niitb([] { (void)hecke::CoefficientRingK(); }, "coefficient_ring=Z"), "(S) CoefficientRingK() refuses under Z");
        check(throws_niitb([] { (void)hecke::CoefficientRingK("probe-site"); }, "probe-site: coefficient_ring=Z"),
              "(S) CoefficientRingK(who) refuses naming who");
    }
    check(!hecke::CoefficientRingIsZ() && hecke::CoefficientRingK() == 1, "(S) and the scope restores F_ell");
}

// ---- (F) the refusal closure under Z -------------------------------------------------------------------------------
void ArmClosure() {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    hecke::SL2ZTessellationTables model(K);
    auto rels = GenerateSL2ZRelationsElt<uint32_t>(K, 10, 691u);
    LocalSystemSimplicialComplex lssc(K, rels, 11, 691u);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    hecke::HeckeCosetTables cosets(2, 1, 691u, 10);
    hecke::HeckeTauTable tau(K, model, cosets, 691u, 10);
    {
        const hecke::ScopedCoefficientRing z(0);
        check(throws_niitb([&] { EquivariantCohomologyOnK x(&lssc); }, "EquivariantCohomologyOnK: coefficient_ring=Z"),
              "(F) EquivariantCohomologyOnK refuses Z by name");
        check(throws_niitb([&] { hecke::HeckeOperatorOnK x(&coh, &cosets, &tau); }, "HeckeOperatorOnK: coefficient_ring=Z"),
              "(F) HeckeOperatorOnK refuses Z by name");
        check(throws_niitb([] { hecke::RequireFieldCoefficientRing("FieldOnlySite"); }, "FieldOnlySite: coefficient_ring=Z"),
              "(F) a field-only site (RequireFieldCoefficientRing) refuses Z by name");
    }
    {
        // and the descent record is refused under any ring but Z
        check(throws_invalid([] { (void)rec::BuildDescentRecord(1, 4); }, "set coefficient_ring = Z"),
              "(F) BuildDescentRecord under F_ell is refused (the record is over Z)");
        const hecke::ScopedCoefficientRing r2(2);
        check(throws_invalid([] { (void)rec::BuildDescentRecord(1, 4); }, "Z/ell^2"), "(F) and under Z/ell^2");
    }
    const int e_field = run_console("set coefficient_ring = \"Z\"; H = cohomology(symmetric_power(sl2z_model(), 10, 691)); print(1);");
    if (e_field == -1) {
        std::cout << "  (F) console arms SKIPPED: /proc/self/exe unavailable" << std::endl;
        return;
    }
    const int e_desc = run_console("set coefficient_ring = \"Z\"; D = descent(1, 4, [2]); assert(D.status == \"ok\"); print(D.groups);");
    const int e_desc_f = run_console("D = descent(1, 4, [2]); print(D.groups);");
    const int e_bad = run_console("set coefficient_ring = \"Z\"; D = descent(2, 4, [2]); print(D.groups);");
    const int e_big = run_console("set coefficient_ring = \"Z\"; D = descent(1, 5000); print(D.groups);");
    const int e_hecke = run_console("H = cohomology(symmetric_power(sl2z_model(), 10, 691)); set coefficient_ring = \"Z\"; T = hecke(H, 2); print(1);");
    check(e_field == 5, "(F) console: a field construction under coefficient_ring = Z exits 5 (got " + std::to_string(e_field) + ")");
    check(e_hecke == 5, "(F) console: `hecke` after `set coefficient_ring = \"Z\"` exits 5 (got " + std::to_string(e_hecke) + ")");
    check(e_desc == 0, "(F) console: `descent` under coefficient_ring = Z exits 0 (got " + std::to_string(e_desc) + ")");
    check(e_desc_f == 3, "(F) console: `descent` under F_ell is a runtime error, exit 3 (got " + std::to_string(e_desc_f) + ")");
    check(e_bad == 3, "(F) console: a Hecke prime dividing N is a type error, exit 3 (got " + std::to_string(e_bad) + ")");
    check(e_big == 3, "(F) console: a cell beyond the verb's cost bound is a type error, exit 3 (got " + std::to_string(e_big) + ")");
    // (D6 audit m2) the bound accounts for the Hecke primes: sum p <= 100
    const int e_psum = run_console("set coefficient_ring = \"Z\"; D = descent(1, 4, [97, 5]); print(D.groups);");
    check(e_psum == 3, "(F) console: Hecke primes summing above 100 are beyond the cost bound, exit 3 (got " + std::to_string(e_psum) + ")");
    // (D6 audit m1) every field verb refuses Z and Z/ell^2 at its statement, exit 5 -- never an F_ell result under another ring
    uint32_t refused = 0;
    for (const std::string& call : FieldVerbCalls())
        for (const std::string& ring : {std::string("Z"), std::string("Z/ell^2")}) {
            if (ring != "Z" && call.rfind("symmetric_power", 0) == 0) continue;   // honours Z/ell^k ([E] sec:ring)
            const int e = run_console("set coefficient_ring = \"" + ring + "\"; x = " + call + "; print(x);");
            refused += e == 5 ? 1 : 0;
            check(e == 5, "(F) console: `" + call + "` under coefficient_ring = " + ring + " exits 5 (got " + std::to_string(e) + ")");
        }
    // (D6 audit N1) the console plans a Z spec's descent cells
    const int e_plan = run_console("set coefficient_ring = \"Z\"; S = campaign_spec_text(\"coefficient_ring = Z\\nn_sym = 2, 4\\n\"); "
                                   "P = campaign_plan(S); assert(len(P) == 1); assert(len(P[0].jobs) == 2); print(P);");
    check(e_plan == 0, "(F) console: campaign_plan of a coefficient_ring = Z spec lists its descent cells, exit 0 (got " + std::to_string(e_plan) + ")");
    std::cout << "  (F) " << refused << " field-verb calls refused under Z / Z/ell^2 (exit 5)" << std::endl;
}

// ---- (R) the record against the oracles ----------------------------------------------------------------------------
void ArmOracles() {
    const Oracles& O = TheOracles();
    if (!check(O.ok, "(R) the four oracle tables are found under " + src_path("docs/implementation/slot24/"))) return;
    const std::vector<std::pair<uint32_t, uint32_t>> cells = {{1, 2},  {1, 3}, {1, 4}, {1, 6}, {1, 10}, {1, 12}, {2, 2},
                                                              {2, 3}, {2, 6}, {3, 2}, {5, 2}, {7, 2},  {11, 0}};
    const hecke::ScopedCoefficientRing z(0);
    uint32_t n_groups = 0, n_chi = 0, n_tors = 0, n_empty = 0, n_pair = 0;
    for (const auto& [N, n] : cells) {
        const std::string at = "(N, n) = (" + std::to_string(N) + ", " + std::to_string(n) + ")";
        const rec::DescentRecordZ R = rec::BuildDescentRecord(N, n);
        if (!check(R.ok(), "(R) " + at + ": the record is ok (" + R.refusal + ")")) continue;
        for (uint32_t q = 0; q <= R.q_max; ++q) {
            const auto it = O.groups.find({N, n, q});
            if (it == O.groups.end()) continue;
            check(OracleForm(R.H[q]) == it->second, "(R) " + at + " H^" + std::to_string(q) + ": " + OracleForm(R.H[q]) + " == oracle " + it->second);
            ++n_groups;
        }
        uint32_t chi_here = 0;
        for (const rec::CharpolyEntry& c : R.charpoly) {
            const auto it = O.chi.find({N, n, c.p});
            if (it == O.chi.end()) continue;
            std::string got;
            for (size_t i = 0; i < c.coefficients.size(); ++i) got += (i ? "," : "") + c.coefficients[i];
            check(c.determined && got == it->second, "(R) " + at + " char_Z(T_" + std::to_string(c.p) + ") = " + got + " == oracle " + it->second);
            ++n_chi;
            ++chi_here;
        }
        check(chi_here > 0 || R.charpoly.empty(), "(R) " + at + ": at least one char_Z is compared");
        for (const rec::TorsionModule& m : R.torsion) {
            for (const auto& [p, canon] : m.invariants) {
                const auto it = O.tors.find({N, n, m.q, m.ell, p});
                if (it == O.tors.end()) continue;
                check(canon == it->second, "(R) " + at + " tors H^" + std::to_string(m.q) + "_(" + std::to_string(m.ell) + ") with T_" +
                                               std::to_string(p) + ": " + canon + " == oracle " + it->second);
                ++n_tors;
            }
            // the oracle reads H^1 and H^2 at every torsion prime (for every p it lists) and writes no row for an empty group
            bool oracle_row = false, oracle_cell = false;
            for (uint32_t p : R.ps) {
                oracle_row = oracle_row || O.tors.count({N, n, m.q, m.ell, p}) != 0;
                oracle_cell = oracle_cell || O.chi.count({N, n, p}) != 0;
            }
            if (m.q <= 2 && oracle_cell && !oracle_row) {
                check(m.exponents.empty() && !m.refused, "(R) " + at + " tors H^" + std::to_string(m.q) + "_(" + std::to_string(m.ell) +
                                                             ") is empty where the oracle has no row");
                ++n_empty;
            }
        }
        for (uint32_t q = 0; q <= R.pair.q_max && q < R.pair.H.size(); ++q) {
            const std::pair<const char*, const std::vector<AbelianGroupType>*> objs[] = {{"H", &R.pair.H}, {"Hd", &R.pair.Hd}, {"Hc", &R.pair.Hc}};
            for (const auto& [obj, G] : objs) {
                const auto it = O.pair.find({N, n, obj, q});
                if (it == O.pair.end()) continue;
                check(OracleForm((*G)[q]) == it->second, "(R) " + at + " pair " + obj + "^" + std::to_string(q) + ": " + OracleForm((*G)[q]) +
                                                             " == oracle " + it->second);
                ++n_pair;
            }
        }
    }
    check(n_groups >= 40 && n_chi >= 30 && n_tors >= 40 && n_pair >= 100, "(R) the comparisons are not vacuous");
    std::cout << "  (R) " << cells.size() << " cells: " << n_groups << " groups, " << n_chi << " char_Z, " << n_tors << " torsion modules, "
              << n_empty << " empty torsion groups, " << n_pair << " pair groups equal to the oracles" << std::endl;
}

// ---- (J) the record's schema and hash input ---------------------------------------------------------------------------
void ArmSchema() {
    rec::DescentRecordZ A, B, C;
    {
        const hecke::ScopedCoefficientRing z(0);
        A = rec::BuildDescentRecord(1, 12);
        B = rec::BuildDescentRecord(1, 12);
    }
    {
        hecke::MathSettings s = hecke::math_settings();
        s.set("coefficient_ring", "Z", hecke::SettingSource::Spec);
        s.set("locality", "strict", hecke::SettingSource::Spec);
        const hecke::ScopedMathSettings scope(s, hecke::SettingSource::Spec);
        C = rec::BuildDescentRecord(1, 12);
    }
    check(A.ok() && B.ok() && C.ok(), "(J) the three records of (1, 12) are ok");
    check(A.digest() == B.digest() && A.body() == B.body() && A.to_json().size() > 0, "(J) the record is deterministic (two runs, one digest)");
    // (D6 audit N5) the timing / RSS fields are t_groups, t_hecke, t_pair, t_total and rss_peak (the JSON's timings_s, rss_peak_mib)
    check(A.body().find("timings") == std::string::npos && A.body().find("t_total") == std::string::npos &&
              A.body().find("rss") == std::string::npos && A.body().find("route") == std::string::npos &&
              A.body().find("K=") == std::string::npos && A.body().find("derived_from") == std::string::npos,
          "(J) the hash input carries no timing, RSS, route, precision or derived-from list (provenance only)");
    check(A.to_json().find("\"timings_s\":{\"groups\":") != std::string::npos && A.to_json().find("\"rss_peak_mib\":") != std::string::npos,
          "(J) (and the JSON does carry the timings and the RSS, so the check above is not vacuous)");
    check(C.digest() != A.digest() && C.settings_fingerprint != A.settings_fingerprint && C.H == A.H,
          "(J) a second non-default setting changes the hash input and the fingerprint, not the invariants");
    check(A.settings_fingerprint == engine::FnvBytes(engine::kFnvOffset64, "coefficient_ring = Z\n"), "(J) under Z alone the fingerprint is FNV(\"coefficient_ring = Z\\n\")");
    const std::string j = A.to_json();
    for (const char* key : {"\"schema\":\"integral_descent_record/v1\"", "\"coefficient_ring\":\"Z\"", "\"object\":\"group\"", "\"q_max\":3",
                            "\"truncated\":true", "\"q_max_source\":\"derived\"", "\"degrees\":[0,1,2,3]", "\"periodic_from\":2", "\"groups\":[",
                            "\"bad_primes\":[{\"ell\":5,\"count\":1},{\"ell\":11,\"count\":1}]", "\"torsion_primes\":[2,3,5,11]", "\"K\":[",
                            "\"charpoly\":[", "\"torsion\":[", "\"pair\":{\"ran\":true,\"bulk\":{\"q_max\":4,\"truncated\":true", "\"boundary\":{",
                            "\"compact\":{\"q_max\":4,\"truncated\":true,\"q_max_source\":\"derived\",\"degrees\":[0,1,2,3,4],\"derived_from\":[\"bulk\",\"boundary\"]",
                            "\"status\":\"ok\"", "\"settings\":{", "\"settings_fingerprint\":\"0x", "\"record_fnv1a64\":\"0x",
                            "\"timings_s\":{"})
        check(j.find(key) != std::string::npos, std::string("(J) the JSON carries ") + key);
    bool kcert = !A.K.empty();
    for (const rec::PrimeK& k : A.K) kcert = kcert && k.certified && k.K > 0;
    check(kcert, "(J) every prime of the groups' run has a certified K");
    std::set<uint32_t> kells;
    for (const rec::PrimeK& k : A.K) kells.insert(k.ell);
    check(kells == std::set<uint32_t>{2, 3, 5, 11}, "(J) K is recorded at 2, 3 and every bad prime");
    check(A.torsion.size() == 2 * 3 + 2, "(J) (1, 12): torsion modules at (q, 2), (q, 3), q = 1..3, and (1, 5), (1, 11)");
    // refusals by name
    {
        const hecke::ScopedCoefficientRing z(0);
        check(throws_invalid([] { (void)rec::BuildDescentRecord(2, 4, rec::RecordOptions{{2}}); }, "divides N"), "(J) p | N is refused");
        check(throws_invalid([] { (void)rec::BuildDescentRecord(1, 4, rec::RecordOptions{{4}}); }, "not a prime"), "(J) a non-prime p is refused");
        check(throws_invalid([] { (void)rec::BuildDescentRecord(0, 4); }, "N must be"), "(J) N = 0 is refused");
        // (D6 audit N3) P is a set: the record (its primes, charpoly order and hash) does not depend on the caller's order
        const rec::DescentRecordZ P23 = rec::BuildDescentRecord(1, 4, rec::RecordOptions{{2, 3}});
        const rec::DescentRecordZ P32 = rec::BuildDescentRecord(1, 4, rec::RecordOptions{{3, 2}});
        check(P23.ok() && P32.ok() && P32.ps == std::vector<uint32_t>{2, 3} && P32.body() == P23.body() && P32.digest() == P23.digest(),
              "(J) primes [3, 2] give the record of [2, 3]: P ascending, one body, one record_fnv1a64");
        rec::RecordOptions tiny;
        tiny.rss_cap_bytes = 1ull << 20;
        const rec::DescentRecordZ T = rec::BuildDescentRecord(1, 12, tiny);
        check(!T.ok() && T.status == "refused" && T.H.empty() && !T.refusal.empty(),
              "(J) a part above the RSS ceiling refuses the record: status refused, the reason, the groups EMPTY (never partial)");
    }
    std::cout << "  (J) record_fnv1a64 (1, 12) under Z = 0x" << std::hex << A.digest() << ", under Z + locality=strict = 0x" << C.digest() << std::dec
              << "; json " << j.size() << " bytes" << std::endl;
}

// ---- (K) the campaign under Z -----------------------------------------------------------------------------------------
campaign::CampaignSpec SpecOf(const std::string& text) {
    std::istringstream is(text);
    return campaign::ParseCampaignSpec(is, "<d6-spec>");
}
void ArmCampaign() {
    namespace fs = std::filesystem;
    const std::string base = "coefficient_ring = Z\nname = d6val\n";
    check(throws_niitb([&] { (void)SpecOf(base + "ext = true\nn_sym = 4\n"); }, "coefficient_ring=Z"), "(K) `ext = true` under Z is refused (exit 5)");
    check(throws_runtime([&] { (void)SpecOf(base + "n_sym = 4\nell = 5\n"); }, "ell:"), "(K) `ell` under Z is a spec error");
    check(throws_runtime([&] { (void)SpecOf(base + "n_sym = 4\nr = 2\n"); }, "r:"), "(K) `r` under Z is a spec error");
    check(throws_runtime([&] { (void)SpecOf(base + "n_sym = 4\nq = 1\n"); }, "q:"), "(K) `q` under Z is a spec error");
    check(throws_runtime([&] { (void)SpecOf(base + "n_sym = 4\np = 4\n"); }, "not a prime"), "(K) a non-prime p is a spec error");
    check(throws_runtime([&] { (void)SpecOf(base + "level = 2\n"); }, "missing 'n_sym'"), "(K) a section without n_sym is a spec error");
    check(throws_runtime([&] { (void)SpecOf(base + "n_sym = 4\nlevel = 1, 2\np = 2, 3\n"); }, "p: 2 divides level 2"),
          "(K) a Hecke prime dividing a section's level is a spec error (never dropped, never replaced by the defaults; audit N2)");
    check(throws_runtime([&] { (void)SpecOf("n_sym = 10\nell = 691\np = 2\nr = 1\nlevel = 2\n"); }, "level:"),
          "(K) `level` outside coefficient_ring = Z is a spec error");
    campaign::CampaignSpec spec = SpecOf(base + "n_sym = 2, 4\nlevel = 1, 11\np = 2, 3\n---\nn_sym = 0\nlevel = 11\n");
    const uint64_t fp = spec.math.fingerprint();
    check(fp == engine::FnvBytes(engine::kFnvOffset64, "coefficient_ring = Z\n"), "(K) the spec's fingerprint is Z's");
    const std::vector<campaign::DescentCellPlan> plan = campaign::PlanDescentCells(spec);
    std::vector<std::string> ids;
    for (const auto& c : plan) ids.push_back(c.id);
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(fp));
    const std::string s = std::string("_s") + hex;
    const std::vector<std::string> want = {"desc_N1_n2_p2-3" + s, "desc_N1_n4_p2-3" + s, "desc_N11_n0" + s, "desc_N11_n2_p2-3" + s,
                                           "desc_N11_n4_p2-3" + s};
    check(ids == want, "(K) the plan: five cells, ids desc_N<N>_n<n>[_p..]_s<fp> in (N, n) order");
    check(throws_runtime([&] { (void)campaign::RunCampaignSpec(spec, false); }, "RunDescentCampaignCli"),
          "(K) the field job loop refuses a coefficient_ring = Z spec");
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / ("oc_d6_descent_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    spec = SpecOf(base + "n_sym = 2\nlevel = 1, 3\n");
    spec.output_dir = dir.string();
    const campaign::DescentCampaignResult r1 = campaign::RunDescentCampaign(spec, false);
    check(r1.cells == 2 && r1.ran == 2 && r1.failed == 0 && r1.refused == 0, "(K) a two-cell run: both cells ran, none failed or refused");
    const std::vector<campaign::DescentCellPlan> p2 = campaign::PlanDescentCells(spec);
    bool files = p2.size() == 2;
    for (const auto& c : p2) {
        std::ifstream in(dir / "cells" / (c.id + ".json"));
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        files = files && body.find("\"schema\":\"integral_descent_cell/v1\"") != std::string::npos &&
                body.find("\"schema\":\"integral_descent_record/v1\"") != std::string::npos &&
                body.find("\"complete\":true}") != std::string::npos;
    }
    check(files, "(K) every cell file carries the cell schema, the record and the completion trailer");
    check(fs::exists(dir / "summary.tsv") && fs::exists(dir / "index.json"), "(K) summary.tsv and index.json are written");
    spec.resume = true;
    const campaign::DescentCampaignResult r2 = campaign::RunDescentCampaign(spec, false);
    check(r2.resumed == 2 && r2.ran == 0, "(K) resume skips both complete cells");
    fs::remove_all(dir, ec);
}

// ---- (P) the `parts` key (slot-24 ICP2) ---------------------------------------------------------------------------------
// (scratch under temp_directory_path() as arm (K) does, a pre-existing pattern)
// default `all` = today's campaign (ids, bytes); `groups` / `groups,hecke` compute a prefix of the record's parts, and a skipped
// part is marked not run, never missing or zero; a non-default value is part of the cell id; an unknown value is refused.
void ArmParts() {
    namespace fs = std::filesystem;
    const std::string base = "coefficient_ring = Z\nname = icp2val\nn_sym = 2, 4\nlevel = 1, 3\n";
    const campaign::CampaignSpec sAll = SpecOf(base), sAllX = SpecOf(base + "parts = all\n"), sGH = SpecOf(base + "parts = groups,hecke\n"),
                                 sGH2 = SpecOf(base + "parts = groups, hecke\n"), sG = SpecOf(base + "parts = groups\n");
    // (a) the default and the explicit `all` plan the same ids; the default id has no parts suffix
    const auto idsOf = [](const campaign::CampaignSpec& sp) {
        std::vector<std::string> v;
        for (const auto& c : campaign::PlanDescentCells(sp)) v.push_back(c.id);
        return v;
    };
    const auto iA = idsOf(sAll), iAx = idsOf(sAllX), iGH = idsOf(sGH), iG = idsOf(sG);
    check(iA == iAx && iA.size() == 4 && iA[0].find("_parts") == std::string::npos, "(P) parts = all is the default: the same four ids, no suffix");
    check(iGH == idsOf(sGH2), "(P) `groups,hecke` and `groups, hecke` are one value");
    // (c) distinct ids across the three values
    std::set<std::string> all3(iA.begin(), iA.end());
    all3.insert(iGH.begin(), iGH.end());
    all3.insert(iG.begin(), iG.end());
    check(all3.size() == 12, "(P) the three parts values give 12 distinct ids over four cells");
    // (d) unknown values are refused, with the key's name and the known values
    check(throws_runtime([&] { (void)SpecOf(base + "parts = pair\n"); }, "parts: unknown value"), "(P) an unknown parts value is a spec error");
    check(throws_runtime([&] { (void)SpecOf(base + "parts = hecke\n"); }, "known: groups"), "(P) `hecke` alone is not a value");
    check(throws_runtime([&] { (void)SpecOf("n_sym = 10\nell = 691\np = 2\nr = 1\nparts = groups\n"); }, "parts:"),
          "(P) `parts` outside coefficient_ring = Z is a spec error");
    // (b) the records: the groups (and the Hecke data) of every parts value equal the full record's; the skipped parts are marked
    const hecke::ScopedCoefficientRing z(0);
    const std::vector<std::pair<uint32_t, uint32_t>> cells = {{1, 4}, {3, 2}, {2, 6}};
    for (const auto& [N, n] : cells) {
        const std::string at = "(N, n) = (" + std::to_string(N) + ", " + std::to_string(n) + ")";
        rec::RecordOptions oG, oGH;
        oG.hecke = false;
        oG.pair = false;
        oGH.pair = false;
        const rec::DescentRecordZ F = rec::BuildDescentRecord(N, n), GH = rec::BuildDescentRecord(N, n, oGH), G = rec::BuildDescentRecord(N, n, oG);
        if (!check(F.ok() && GH.ok() && G.ok(), "(P) " + at + ": the three records are ok")) continue;
        check(F.hecke_ran && F.pair.ran && F.to_json().find("\"hecke\":{\"ran\"") == std::string::npos,
              "(P) " + at + ": the full record carries no hecke key (its bytes are the pre-ICP2 bytes)");
        check(GH.hecke_ran && !GH.pair.ran && GH.to_json().find("\"pair\":{\"ran\":false") != std::string::npos,
              "(P) " + at + ": groups,hecke: hecke ran, pair marked ran:false");
        check(!G.hecke_ran && !G.pair.ran && G.to_json().find("\"hecke\":{\"ran\":false}") != std::string::npos &&
                  G.to_json().find("\"pair\":{\"ran\":false") != std::string::npos,
              "(P) " + at + ": groups: hecke and pair both marked ran:false");
        bool same = G.H.size() == F.H.size() && GH.H.size() == F.H.size();
        for (size_t q = 0; same && q < F.H.size(); ++q) same = G.H[q].to_string() == F.H[q].to_string() && GH.H[q].to_string() == F.H[q].to_string();
        check(same, "(P) " + at + ": the groups are identical in all three");
        bool hsame = GH.charpoly.size() == F.charpoly.size() && GH.torsion.size() == F.torsion.size() && G.charpoly.empty() && G.torsion.empty();
        for (size_t i = 0; hsame && i < F.charpoly.size(); ++i) hsame = GH.charpoly[i].digest == F.charpoly[i].digest;
        check(hsame && !F.charpoly.empty(), "(P) " + at + ": char_Z / torsion of groups,hecke equal the full record's; groups has none, by absence");
        check(G.digest() != F.digest() && GH.digest() != F.digest() && G.body().find("hecke=not_run") != std::string::npos,
              "(P) " + at + ": a partial record's hash differs from the full one (the body says hecke=not_run)");
    }
    // the campaign runs: parts = groups writes cells under its own ids, summary status ok
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / ("oc_icp2_parts_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    campaign::CampaignSpec sp = SpecOf("coefficient_ring = Z\nname = icp2run\nn_sym = 2\nlevel = 1\nparts = groups\n");
    sp.output_dir = dir.string();
    const campaign::DescentCampaignResult r = campaign::RunDescentCampaign(sp, false);
    check(r.cells == 1 && r.ran == 1 && r.failed == 0 && r.refused == 0, "(P) a parts = groups campaign cell runs");
    std::ifstream in(dir / "index.json");
    const std::string idx((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    check(idx.find("\"parts\":\"groups\"") != std::string::npos, "(P) index.json records the non-default parts");
    // resume across parts values: a groups cell file is never resumed for an `all` cell, nor the reverse (the id carries the tail)
    campaign::CampaignSpec spAll = SpecOf("coefficient_ring = Z\nname = icp2run\nn_sym = 2\nlevel = 1\n");
    spAll.output_dir = dir.string();
    spAll.resume = true;
    const campaign::DescentCampaignResult rA = campaign::RunDescentCampaign(spAll, false);
    check(rA.cells == 1 && rA.ran == 1 && rA.resumed == 0, "(P) groups run, then all with resume into the same dir: ran, not resumed");
    sp.resume = true;
    const campaign::DescentCampaignResult rB = campaign::RunDescentCampaign(sp, false);
    check(rB.resumed == 1 && rB.ran == 0, "(P) and the reverse: the groups cell resumes from its own file");
    fs::remove_all(dir, ec);
    sp.resume = false;
    check(throws_runtime([&] { (void)SpecOf(base + "parts = gro ups\n"); }, "parts: unknown value"), "(P) blanks inside a word are refused");
    check(idsOf(SpecOf(base + "parts = groups , hecke\n")) == iGH, "(P) blanks around the comma are tolerated");
}

// ---- (B) the rollbacks are value-preserving on the hash ([C] sec:recordschema item 7 (h)) ------------------------------
// HECKE_DESCENT_TREE=0 (the dense route of the groups and the pair) and HECKE_DESCENT_TREE_PAIR=0 (the dense cusp pair) change
// routes, never record_fnv1a64: in process through the scoped overrides, and once through the real env variables in a child.
void ArmRollback() {
    const std::vector<std::pair<uint32_t, uint32_t>> cells = {{1, 12}, {1, 3}, {2, 8}, {3, 4}, {11, 2}};
    const hecke::ScopedCoefficientRing z(0);
    uint32_t compared = 0;
    for (const auto& [N, n] : cells) {
        const std::string at = "(N, n) = (" + std::to_string(N) + ", " + std::to_string(n) + ")";
        const rec::DescentRecordZ D = rec::BuildDescentRecord(N, n);
        rec::DescentRecordZ T0, P0;
        {
            const integral_descent::ScopedDescentTree off(false);
            T0 = rec::BuildDescentRecord(N, n);
        }
        {
            const integral_descent::ScopedDescentTreePair off(false);
            P0 = rec::BuildDescentRecord(N, n);
        }
        if (!check(D.ok() && T0.ok() && P0.ok(), "(B) " + at + ": the default and both rollback records are ok")) continue;
        check(D.pair.route_two == "tree" && T0.pair.route_two != "tree" && P0.pair.route_two != "tree",
              "(B) " + at + ": the rollbacks took effect (pair route_two " + D.pair.route_two + " / " + T0.pair.route_two + " / " +
                  P0.pair.route_two + ")");
        check(T0.digest() == D.digest() && T0.body() == D.body(), "(B) " + at + ": HECKE_DESCENT_TREE=0 keeps record_fnv1a64");
        check(P0.digest() == D.digest() && P0.body() == D.body(), "(B) " + at + ": HECKE_DESCENT_TREE_PAIR=0 keeps record_fnv1a64");
        ++compared;
    }
    const std::string code = "set coefficient_ring = \"Z\"; D = descent(1, 12); print(D.digest);";
    const std::string d0 = run_console_out("", code);
    if (d0.empty()) {
        std::cout << "  (B) env arms SKIPPED: /proc/self/exe unavailable" << std::endl;
    } else {
        check(run_console_out("HECKE_DESCENT_TREE=0 ", code) == d0, "(B) console (1, 12): HECKE_DESCENT_TREE=0 prints the default digest " + d0);
        check(run_console_out("HECKE_DESCENT_TREE_PAIR=0 ", code) == d0,
              "(B) console (1, 12): HECKE_DESCENT_TREE_PAIR=0 prints the default digest " + d0);
    }
    std::cout << "  (B) " << compared << " cells: record_fnv1a64 equal under the default, HECKE_DESCENT_TREE=0 and HECKE_DESCENT_TREE_PAIR=0"
              << std::endl;
}

// ---- (G) the staged golden points ---------------------------------------------------------------------------------------
void ArmGolden() {
    std::set<std::tuple<std::string, std::string, char>> want;
    {
        std::ifstream in(src_path("docs/implementation/slot24/d6golden/expect.tsv"));
        if (!check(static_cast<bool>(in), "(G) the prediction d6golden/expect.tsv is found")) return;
        std::string line;
        bool only_adds = true;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            const auto f = SplitTab(line);
            if (f.size() != 4 || f[0] != "+" || f[3].size() != 1) {
                only_adds = false;
                continue;
            }
            want.insert({f[1], f[2], f[3][0]});
        }
        check(only_adds, "(G) the prediction lists additions only (no `~` / `-` line)");
    }
    const auto t0 = clk::now();
    const std::vector<GoldenComputedRecord> a = GoldenDescentRecords();
    const double t_golden = since(t0);
    const std::vector<GoldenComputedRecord> b = GoldenDescentRecords();
    std::set<std::tuple<std::string, std::string, char>> got;
    // (s24-d6golden, the landing branch; D6 audit M1) the event has landed: every staged key is PRESENT in the reference with
    // equal tier, rows, cols and hash (on s24-d6impl, before the landing, this check read "absent")
    bool landed = true, tier_a = true;
    for (const GoldenComputedRecord& r : a) {
        got.insert({r.point, r.object, r.tier});
        const GoldenEntry* e = GoldenLookup(r.point, r.object);
        if (e == nullptr || e->tier != r.tier || e->rows != r.rows || e->cols != r.cols || e->hash != r.hash) landed = false;
        if (r.tier != 'A') tier_a = false;
    }
    check(got.size() == a.size(), "(G) the staged keys are distinct");
    check(got == want, "(G) the staged keys equal the prediction key for key (" + std::to_string(got.size()) + " computed, " +
                           std::to_string(want.size()) + " predicted)");
    check(tier_a, "(G) every staged object is tier A");
    check(landed, "(G) every staged key is in the reference with equal tier, rows, cols and hash (the event has landed)");
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i)
        same = a[i].point == b[i].point && a[i].object == b[i].object && a[i].rows == b[i].rows && a[i].cols == b[i].cols && a[i].hash == b[i].hash;
    check(same, "(G) two computations agree hash for hash");
    // (D6 audit N5) the VALUES are pinned: d6golden/values.tsv (the audit's scratch emission, ten byte-identical variants)
    {
        std::ifstream in(src_path("docs/implementation/slot24/d6golden/values.tsv"));
        std::vector<GoldenComputedRecord> pin;
        std::string line;
        while (in && std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            const auto f = SplitTab(line);
            if (f.size() != 6 || f[2].size() != 1) continue;
            pin.push_back({f[0], f[1], f[2][0], std::stoull(f[3]), std::stoull(f[4]), std::stoull(f[5], nullptr, 16)});
        }
        bool equal = pin.size() == a.size();
        for (size_t i = 0; equal && i < a.size(); ++i)
            equal = a[i].point == pin[i].point && a[i].object == pin[i].object && a[i].tier == pin[i].tier && a[i].rows == pin[i].rows &&
                    a[i].cols == pin[i].cols && a[i].hash == pin[i].hash;
        check(equal, "(G) the staged records equal the pinned values d6golden/values.tsv line for line (" + std::to_string(pin.size()) +
                         " pinned)");
    }
    std::cout << "  (G) " << a.size() << " staged DESCENT records (tier A), " << want.size() << " predicted; " << t_golden << " s per computation"
              << std::endl;
}

}  // namespace

// (Z) (slot-24 NB-TOURFIX) the factored char_Z(T_p) of the console member `charpoly_factored`: the factoriser
// (NumberField/IntPolyFactor.h, Yun + Zassenhaus) on constructed products with known factorisations (X^4 + 1,
// X^4 - 10X^2 + 1 and X^8 + 1 are irreducible over Z and split modulo every prime: the recombination must join them),
// and on every char_Z of the D3 oracle (each product verified); the console member at (1, 10) and (11, 0).
void ArmFactor() {
    using nf::BigInt;
    using P = std::vector<BigInt>;   // constant term first
    auto mk = [](std::initializer_list<int64_t> hi_to_lo) {
        P f;
        for (int64_t c : hi_to_lo) f.insert(f.begin(), BigInt(c));
        return f;
    };
    auto pw = [](const P& a, unsigned m) {
        P r = {BigInt(1)};
        for (unsigned i = 0; i < m; ++i) r = nf::zpoly_mul(r, a);
        return r;
    };
    struct Case {
        P f;
        std::string text;
    };
    const P x4p1 = mk({1, 0, 0, 0, 1}), sw = mk({1, 0, -10, 0, 1}), gold = mk({1, -1, -1}), cub = mk({1, 0, 0, -2});
    const std::vector<Case> cases = {
        {nf::zpoly_mul(pw(mk({1, 24}), 2), mk({1, -2049})), "(X + 24)^2 (X - 2049)"},
        {nf::zpoly_mul(mk({1, -3}), pw(mk({1, 2}), 2)), "(X + 2)^2 (X - 3)"},
        {x4p1, "X^4 + 1"},
        {sw, "X^4 - 10X^2 + 1"},
        {nf::zpoly_mul(nf::zpoly_mul(pw(gold, 2), cub), nf::zpoly_mul(x4p1, mk({1, -5}))), "(X - 5) (X^2 - X - 1)^2 (X^3 - 2) (X^4 + 1)"},
        {nf::zpoly_mul(nf::zpoly_mul(sw, x4p1), pw(mk({1, 0}), 3)), "X^3 (X^4 + 1) (X^4 - 10X^2 + 1)"},
        {pw(nf::zpoly_mul(mk({1, -1}), mk({1, 1})), 4), "(X + 1)^4 (X - 1)^4"},
        {nf::zpoly_mul(pw(sw, 2), nf::zpoly_mul(nf::zpoly_mul(x4p1, x4p1), mk({1, 0, 0, 0, 0, 0, 0, 0, 1}))),
         "(X^4 + 1)^2 (X^4 - 10X^2 + 1)^2 (X^8 + 1)"},
    };
    for (const Case& c : cases) {
        const nf::ZFactorisation F = nf::factor_monic_over_z(c.f);
        check(F.verified && F.complete && F.text() == c.text, "(Z) factor " + nf::zpoly_text(c.f) + " = " + F.text() + " (want " + c.text + ")");
    }
    // every char_Z of the D3 oracle
    std::ifstream in(src_path("docs/implementation/slot24/d3_hecke_oracle.tsv"));
    std::string line;
    uint32_t n = 0, nontriv = 0;
    bool all = true;
    while (std::getline(in, line)) {
        const auto f = SplitTab(line);
        if (f.size() < 5 || f[0] != "CHI") continue;
        P g;
        std::stringstream ss(f[4]);
        std::string tok;
        while (std::getline(ss, tok, ',')) g.insert(g.begin(), BigInt::from_string(tok));
        if (g.size() < 2) continue;
        const nf::ZFactorisation F = nf::factor_monic_over_z(g);   // NF_CHECKs the product
        ++n;
        if (F.factors.size() > 1 || F.factors[0].multiplicity > 1) ++nontriv;
        if (!F.verified) {
            all = false;
            check(false, "(Z) oracle char_Z at (" + f[1] + ", " + f[2] + ") p = " + f[3] + ": product not verified");
        }
    }
    check(n > 0 && all, "(Z) " + std::to_string(n) + " oracle char_Z factored, every product verified (" + std::to_string(nontriv) +
                            " with more than one factor)");
    // the console member
    const int e1 = run_console("set coefficient_ring = \"Z\"; D = descent(1, 10, [2]); c = D.charpoly_factored[0]; "
                               "assert(c.factored == \"(X + 24)^2 (X - 2049)\" and c.verified and c.complete and c.status == \"ok\"); "
                               "assert(c.expanded == \"X^3 - 2001X^2 - 97776X - 1180224\"); print(c.factored);");
    const int e2 = run_console("set coefficient_ring = \"Z\"; D = descent(11, 0, [2]); "
                               "assert(D.charpoly_factored[0].factored == \"(X + 2)^2 (X - 3)\"); "
                               "assert(D.charpoly_factored[0].factors[1].multiplicity == 1); print(1);");
    if (e1 >= 0) {
        check(e1 == 0, "(Z) console: descent(1, 10).charpoly_factored[0].factored = (X + 24)^2 (X - 2049) (exit " + std::to_string(e1) + ")");
        check(e2 == 0, "(Z) console: descent(11, 0).charpoly_factored[0].factored = (X + 2)^2 (X - 3) (exit " + std::to_string(e2) + ")");
    }
    std::cout << "  (Z) " << cases.size() << " constructed products and " << n << " oracle char_Z factored over Z (Yun + Zassenhaus), "
              << "products verified; the console member at (1, 10) and (11, 0)" << std::endl;
}

// ---- (X) the faithful value of char_Z beyond 2^63 (slot-24 FMTMP; [C] sec:recordschema item 7 (l)) -----------------------
// Every coefficient beyond 2^63 is in the record as canonical signed hex (CharpolyEntry::exact, JSON `big`, one hashed body line).
// (X1) ExactHex against an nf::BigInt mixed-radix rebuild (constructed and seeded random digits, 1 .. 40 primes near 2^30) and
//      ExactHexModQ against BigInt's remainder; the canonical form; a perturbed text is caught.
// (X2) on cells with coefficients beyond 2^63: the exact value decodes (an independent path) to D3's c mod 2^64 and mod 3^40, to the
//      shipped residue charpoly (ShippedResidueCharpoly, one F_ell' pipeline per prime) at EVERY residue prime of the list, the
//      held-out prime and two primes outside the list; the digest of output rule (5) recomputed from the exact integers is the
//      entry's digest; the integers equal the D3 Python oracle (d3_hecke_oracle.tsv) in decimal; body and JSON carry exactly one
//      line / `big` item per such coefficient.
// (X3) a record with no coefficient beyond 2^63 has no `big` key and no exact line (its bytes and record_fnv1a64 are unchanged).
// (X4) round trip: the cell's JSON decoded by the independent stdlib decoder docs/integral_census/icensus_exact.py gives the same
//      decimal integers and its own digest check passes (SKIPPED without python3); the console's `exact` member at (1, 22).
namespace {
constexpr uint64_t kThree40 = 12157665459056928801ull;

nf::BigInt BigOfDigits(const std::vector<uint32_t>& d, const std::vector<uint32_t>& primes, bool negative) {
    nf::BigInt acc, P(1);
    for (size_t k = d.size(); k-- > 0;) acc = acc * nf::BigInt(static_cast<int64_t>(primes[k])) + nf::BigInt(static_cast<int64_t>(d[k]));
    for (size_t k = 0; k < d.size(); ++k) P *= nf::BigInt(static_cast<int64_t>(primes[k]));
    return negative ? acc - P : acc;
}
nf::BigInt BigOfHex(const std::string& hex) {
    const bool neg = hex[0] == '-';
    nf::BigInt acc, sixteen(16);
    for (size_t j = neg ? 3 : 2; j < hex.size(); ++j) acc = acc * sixteen + nf::BigInt(static_cast<int64_t>(hex[j] <= '9' ? hex[j] - '0' : hex[j] - 'a' + 10));
    return neg ? -acc : acc;
}
// canonical residue of a BigInt mod Q (Q = 0: 2^64)
uint64_t BigMod(const nf::BigInt& x, uint64_t Q) {
    const nf::BigInt M = Q == 0 ? nf::BigInt::pow(nf::BigInt(2), 64) : nf::BigInt::from_string(std::to_string(Q));
    nf::BigInt r = x % M;
    if (r < nf::BigInt(0)) r += M;
    return std::stoull(r.to_string());
}
bool PlainIntegerText(const std::string& s) {
    size_t i = (!s.empty() && s[0] == '-') ? 1 : 0;
    if (i >= s.size()) return false;
    for (; i < s.size(); ++i)
        if (s[i] < '0' || s[i] > '9') return false;
    return true;
}
bool IsPrimeU32(uint32_t p) {
    if (p < 2) return false;
    for (uint32_t d = 2; uint64_t(d) * d <= p; ++d)
        if (p % d == 0) return false;
    return true;
}
}  // namespace

void ArmExact() {
    namespace h3 = integral_descent::hecke3;
    // (X1) the encoding itself
    {
        check(rec::ExactHex({1, 2, 1}, {7, 5, 3}, false) == "0x32" && rec::ExactHex({1, 2, 1}, {7, 5, 3}, true) == "-0x37",
              "(X1) ExactHex: digits (1, 2, 1) over (7, 5, 3) are 50 = 0x32, and -55 = -0x37 when negative");
        check(rec::ExactHexCanonical("0x1f") && rec::ExactHexCanonical("-0x1f") && rec::ExactHexCanonical("0x0") && !rec::ExactHexCanonical("-0x0") &&
                  !rec::ExactHexCanonical("0x01f") && !rec::ExactHexCanonical("0x1F") && !rec::ExactHexCanonical("1f") && !rec::ExactHexCanonical("0x") &&
                  !rec::ExactHexCanonical("+0x1f"),
              "(X1) the canonical form: sign, 0x, lowercase, no leading zero, no -0x0");
        std::mt19937_64 g(0x666d746d70ull);   // "fmtmp"
        std::vector<uint32_t> primes;
        for (uint32_t q = (1u << 30) - 1; primes.size() < 40; q -= 2)
            if (IsPrimeU32(q)) primes.push_back(q);
        uint32_t ok = 0, total = 0;
        for (uint32_t t = 0; t < 200; ++t) {
            const size_t D = 1 + g() % primes.size();
            std::vector<uint32_t> d(D);
            for (size_t k = 0; k < D; ++k) d[k] = static_cast<uint32_t>(g() % primes[k]);
            if (t % 7 == 0) d[D - 1] = primes[D - 1] / 2;   // the sign boundary
            const bool neg = h3::MixedRadixNegative(d, primes);
            const std::string hex = rec::ExactHex(d, primes, neg);
            const nf::BigInt want = BigOfDigits(d, primes, neg);
            bool good = rec::ExactHexCanonical(hex) && BigOfHex(hex) == want;
            for (uint64_t Q : {uint64_t(0), kThree40, uint64_t(primes[0]), uint64_t(1000003), uint64_t(4294967291u), uint64_t(18446744073709551557ull)}) {
                uint64_t m = 0;
                good = good && rec::ExactHexModQ(hex, Q, &m) && m == BigMod(want, Q);
                if (Q == 0) good = good && m == h3::MixedRadixModQ(d, primes, neg, 0);
            }
            ++total;
            ok += good ? 1 : 0;
        }
        check(ok == total, "(X1) " + std::to_string(ok) + " / " + std::to_string(total) +
                               " seeded mixed-radix numbers: ExactHex = the BigInt rebuild, ExactHexModQ = BigInt's remainder (6 moduli)");
        uint64_t m = 0;
        check(!rec::ExactHexModQ("0x01", 0, &m) && !rec::ExactHexModQ("12", 7, &m), "(X1) ExactHexModQ refuses a non-canonical text");
    }
    // (X2), (X3)
    const Oracles& O = TheOracles();
    if (!check(O.ok, "(X2) the oracle tables are found")) return;
    // (X2) cells: the default four, or (read-only env arm, the `_CELLS` convention) INTEGRAL_DESCENT_EXACT_CELLS="N:n,N:n,..." in their
    // place.  An armed cell skips the D3 rerun (its c mod 2^64, mod 3^40 are then read through the recomputed digest) and probes a sample
    // of the residue primes (the first two, the middle, the last two of the used prefix) besides the two primes outside the list.
    std::vector<std::pair<uint32_t, uint32_t>> cells = {{1, 22}, {1, 26}, {2, 10}, {5, 6}};
    const char* env_cells = std::getenv("INTEGRAL_DESCENT_EXACT_CELLS");
    const bool armed = env_cells != nullptr && *env_cells != 0;
    if (armed) {
        cells.clear();
        std::stringstream ss(env_cells);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            const size_t c = tok.find(':');
            if (c == std::string::npos) continue;
            cells.push_back({static_cast<uint32_t>(std::stoul(tok.substr(0, c))), static_cast<uint32_t>(std::stoul(tok.substr(c + 1)))});
        }
        std::cout << "EFFECTIVE IntegralDescentRecordValidation arm=X cells=" << cells.size() << std::endl;
    }
    uint32_t n_big = 0, n_res = 0, n_oracle = 0;
    std::string json_1_22;
    std::map<std::pair<uint32_t, uint32_t>, std::string> dec_1_22;   // (p, i) -> decimal
    for (const auto& [N, n] : cells) {
        const std::string at = "(N, n) = (" + std::to_string(N) + ", " + std::to_string(n) + ")";
        rec::DescentRecordZ R;
        {
            const hecke::ScopedCoefficientRing z(0);
            // groups + Hecke only (the campaign's parts = groups,hecke): the arm reads char_Z, and the boundary pair at an armed cell such
            // as (90, 20) would cost ~27 GB (measured, FMTMP_acceptance.md Sec. 7) against the cell's 2.3 GB
            rec::RecordOptions o;
            o.pair = false;
            R = rec::BuildDescentRecord(N, n, o);
        }
        if (!check(R.ok(), "(X2) " + at + ": the record is ok")) continue;
        const std::string body = R.body(), json = R.to_json();
        if (N == 1 && n == 22) json_1_22 = json;
        uint32_t big_here = 0, res_here = 0;
        bool cell_ok = true;
        for (const rec::CharpolyEntry& c : R.charpoly) {
            const std::string ap = at + " T_" + std::to_string(c.p);
            h3::CharpolyZ C;
            if (!armed) {
                C = h3::CharpolyOverZ(N, n, c.p);   // F_ell in force: D3 directly, for its residues
                if (!check(C.determined && C.c.size() == c.coefficients.size() && c.exact.size() == c.coefficients.size() && C.digest() == c.digest,
                           "(X2) " + ap + ": D3 rerun is determined, same length, same digest")) {
                    cell_ok = false;
                    continue;
                }
            }
            if (!check(c.determined && c.exact.size() == c.coefficients.size(), "(X2) " + ap + ": determined, one exact slot per coefficient")) {
                cell_ok = false;
                continue;
            }
            // the residue primes to probe: every prime of the list a coefficient used (armed: a sample), the held-out prime, two outside
            uint32_t used = 0;
            for (const std::string& t : c.coefficients)
                if (t.rfind("[", 0) == 0) used = std::max<uint32_t>(used, static_cast<uint32_t>(std::stoul(t.substr(3))));
            used = std::max<uint32_t>(used, 1);
            const std::vector<uint32_t>& L = armed ? c.primes : C.primes;
            std::vector<uint32_t> probe;
            const size_t U = std::min<size_t>(used, L.size());
            if (!armed || U <= 5) probe.assign(L.begin(), L.begin() + U);
            else probe = {L[0], L[1], L[U / 2], L[U - 2], L[U - 1]};
            if (!armed && C.held_out_prime) probe.push_back(C.held_out_prime);
            const size_t n_list = probe.size() - ((!armed && C.held_out_prime) ? 1 : 0);
            for (uint32_t q : {1000003u, 999983u})
                if (std::find(L.begin(), L.end(), q) == L.end()) probe.push_back(q);
            std::map<uint32_t, std::vector<uint32_t>> shipped;
            for (uint32_t q : probe) shipped[q] = h3::ShippedResidueCharpoly(N, n, c.p, q);
            engine::FnvHasher h;
            h.feed(c.r);
            std::vector<nf::BigInt> vals;
            bool all = true, lines = true;
            for (size_t i = 0; i < c.coefficients.size(); ++i) {
                nf::BigInt v;
                const bool small = c.exact[i].empty();
                if (small) {
                    all = all && PlainIntegerText(c.coefficients[i]) && (armed || (C.c[i].fits && c.coefficients[i] == std::to_string(C.c[i].value)));
                    if (!all) break;
                    v = nf::BigInt::from_string(c.coefficients[i]);
                } else {
                    const std::string& hex = c.exact[i];
                    uint64_t m64 = 0, m3 = 0;
                    const bool neg = !hex.empty() && hex[0] == '-';
                    all = all && rec::ExactHexCanonical(hex) && rec::ExactHexModQ(hex, 0, &m64) && rec::ExactHexModQ(hex, kThree40, &m3) &&
                          c.coefficients[i].rfind(std::string("[") + (neg ? "-" : "+") + ",", 0) == 0;
                    if (!armed) {
                        const h3::CrtCoefficient& x = C.c[i];
                        all = all && !x.fits && m64 == x.mod_2_64 && m3 == x.mod_3_40 && neg == x.negative &&
                              c.coefficients[i] == std::string("[") + (x.negative ? "-" : "+") + "," + std::to_string(x.digits.size()) + " digits]";
                    }
                    if (!all) break;
                    v = BigOfHex(hex);
                    const nf::BigInt two63 = nf::BigInt::pow(nf::BigInt(2), 63);
                    all = all && (two63 <= v || v <= -two63);   // elided means |c| >= 2^63
                    const std::string ln = "charpoly p=" + std::to_string(c.p) + " i=" + std::to_string(i) + " exact=" + hex + "\n";
                    const std::string js = "{\"i\":" + std::to_string(i) + ",\"hex\":\"" + hex + "\"}";
                    lines = lines && body.find(ln) != std::string::npos && body.find(ln) == body.rfind(ln) && json.find(js) != std::string::npos;
                    ++n_big;
                    ++big_here;
                }
                // residues at every probed prime, from the decoded text (hex for the big ones, the integer otherwise)
                for (const auto& [q, res] : shipped) {
                    uint64_t mq = 0;
                    if (small) mq = BigMod(v, q);
                    else rec::ExactHexModQ(c.exact[i], q, &mq);
                    all = all && res.size() == c.coefficients.size() && mq == res[i];
                    ++n_res;
                    ++res_here;
                }
                h.feed(i);
                h.feed(1u + (v < nf::BigInt(0) ? 1u : 0u));
                h.feed(BigMod(v, 0));
                h.feed(BigMod(v, kThree40));
                vals.push_back(v);
                if (N == 1 && n == 22 && !small) dec_1_22[{c.p, uint32_t(i)}] = v.to_string();   // the big ones (what the decoder prints)
            }
            check(all, "(X2) " + ap + ": every coefficient beyond 2^63 decodes canonically" + std::string(armed ? "" : " to D3's c mod 2^64, mod 3^40 and sign") +
                           ", and every coefficient to the shipped residue charpoly at " + std::to_string(shipped.size()) + " primes (" +
                           std::to_string(n_list) + " of the list" + std::string(armed ? ", sampled" : ", the held-out prime") + ", 2 outside)");
            check(lines, "(X2) " + ap + ": body and JSON carry one exact line / big item per coefficient beyond 2^63");
            check(h.value() == c.digest, "(X2) " + ap + ": output rule (5)'s digest recomputed from the exact integers is the entry's digest");
            cell_ok = cell_ok && all && lines && h.value() == c.digest;
            const auto it = O.chi.find({N, n, c.p});
            if (it != O.chi.end()) {
                std::string got;
                for (size_t i = 0; i < vals.size(); ++i) got += (i ? "," : "") + vals[i].to_string();
                check(got == it->second, "(X2) " + ap + ": the exact char_Z equals the D3 Python oracle in decimal");
                cell_ok = cell_ok && got == it->second;
                ++n_oracle;
            }
        }
        if (!armed) check(big_here > 0, "(X2) " + at + ": the cell has a coefficient beyond 2^63 (the arm is not vacuous)");
        else
            std::cout << "EXACT_CELL N=" << N << " n=" << n << " record_fnv1a64=0x" << std::hex << R.digest() << std::dec << " big=" << big_here
                      << " residues_checked=" << res_here << " json_bytes=" << json.size() << " verdict=" << (cell_ok ? "OK" : "FAIL") << std::endl;
    }
    if (!armed)
        check(n_big >= 20 && n_oracle >= 10, "(X2) the comparisons are not vacuous (" + std::to_string(n_big) + " coefficients beyond 2^63, " +
                                                 std::to_string(n_oracle) + " char_Z against the oracle, " + std::to_string(n_res) + " residues)");
    {
        const hecke::ScopedCoefficientRing z(0);
        const rec::DescentRecordZ S = rec::BuildDescentRecord(1, 12);
        bool none = S.ok();
        for (const rec::CharpolyEntry& c : S.charpoly)
            for (const std::string& e : c.exact) none = none && e.empty();
        check(none && S.body().find("exact=") == std::string::npos && S.to_json().find("\"big\"") == std::string::npos,
              "(X3) (1, 12) has no coefficient beyond 2^63: no exact line, no big key (bytes and record_fnv1a64 unchanged)");
        // a perturbed exact text moves record_fnv1a64 (the body covers the field)
        rec::DescentRecordZ T;
        T = rec::BuildDescentRecord(1, 22);
        rec::DescentRecordZ U = T;
        bool done = false;
        for (rec::CharpolyEntry& c : U.charpoly)
            for (std::string& e : c.exact)
                if (!e.empty() && !done) {
                    e.back() = e.back() == '1' ? '2' : '1';
                    done = true;
                }
        check(T.ok() && done && U.digest() != T.digest(), "(X3) one perturbed hex digit of an exact value moves record_fnv1a64");
    }
    // (X4) the independent Python decoder, then the console member
    const std::string py = src_path("docs/integral_census/icensus_exact.py");
    if (armed) {
        std::cout << "  (X4) SKIPPED: INTEGRAL_DESCENT_EXACT_CELLS replaces the cells" << std::endl;
    } else if (std::system("command -v python3 > /dev/null 2>&1") != 0 || json_1_22.empty()) {
        std::cout << "  (X4) python round trip SKIPPED: python3 unavailable" << std::endl;
    } else {
        namespace fs = std::filesystem;
        std::error_code ec;
        const fs::path dir = fs::temp_directory_path(ec) / ("oc_fmtmp_exact_" + std::to_string(::getpid()));
        fs::create_directories(dir, ec);
        const fs::path f = dir / "desc_N1_n22.json";
        std::ofstream(f) << "{\"schema\":\"integral_descent_cell/v1\",\"id\":\"desc_N1_n22\",\"campaign\":\"fmtmp\",\"record\":" << json_1_22
                         << ",\"complete\":true}\n";
        const std::string cmd = "python3 -I \"" + py + "\" decode --check \"" + f.string() + "\" 2>&1";
        FILE* pf = ::popen(cmd.c_str(), "r");
        std::string out;
        char line[1 << 16];
        while (pf != nullptr && std::fgets(line, sizeof line, pf) != nullptr) out += line;
        const int st = pf ? ::pclose(pf) : -1;
        uint32_t matched = 0, mism = 0;
        bool verdict = false;
        std::istringstream in(out);
        std::string l;
        while (std::getline(in, l)) {
            uint32_t p = 0, i = 0;
            char dec[1 << 15];
            if (std::sscanf(l.c_str(), "EXACT N=%*u n=%*u p=%u i=%u value=%32767s", &p, &i, dec) == 3) {
                const auto it = dec_1_22.find({p, i});
                if (it != dec_1_22.end() && it->second == dec) ++matched;
                else ++mism;
            }
            if (l.rfind("ICENSUS_EXACT ", 0) == 0 && l.find("verdict=OK") != std::string::npos) verdict = true;
        }
        check(st == 0 && verdict && mism == 0 && matched == dec_1_22.size() && matched > 0,
              "(X4) icensus_exact.py decodes (1, 22)'s JSON to the same " + std::to_string(matched) + " integers, its digest check passes (" +
                  std::to_string(mism) + " mismatching)");
        fs::remove_all(dir, ec);
    }
    if (!armed) {
        const auto it = dec_1_22.find({5u, 5u});
        const std::string want = it != dec_1_22.end() ? it->second : "?";
        const int e = run_console("set coefficient_ring = \"Z\"; D = descent(1, 22); c = D.charpoly[2]; assert(c.p == 5); "
                                  "assert(c.exact[5] == \"" + want + "\"); assert(c.exact[0] == \"1\"); assert(D.charpoly_factored[2].verified); print(1);");
        if (e >= 0) check(e == 0 && want != "?", "(X4) console: descent(1, 22).charpoly[2].exact[5] = " + want + " (exit " + std::to_string(e) + ")");
    }
    std::cout << "  (X) " << n_big << " coefficients beyond 2^63 decoded and checked at " << n_res << " residues; " << n_oracle
              << " char_Z equal the D3 Python oracle exactly" << std::endl;
}

uint32_t IntegralDescentRecordValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = clk::now();
    std::cout << "IntegralDescentRecordValidation: coefficient_ring = Z, the record " << rec::kSchema
              << ", its campaign cells, console verb and staged golden points (DESCENT_plan.md Sec. 4 D6)" << std::endl;
    const std::vector<std::pair<const char*, std::function<void()>>> arms = {{"S", ArmSetting}, {"F", ArmClosure},  {"R", ArmOracles},
                                                                              {"J", ArmSchema},  {"K", ArmCampaign}, {"P", ArmParts}, {"B", ArmRollback},
                                                                              {"G", ArmGolden},  {"Z", ArmFactor}, {"X", ArmExact}};
    for (const auto& [name, arm] : arms) {
        try {
            arm();
        } catch (const std::exception& e) {
            check(false, std::string("(") + name + ") exception: " + e.what());
        }
    }
    std::cout << "IntegralDescentRecordValidation: " << g_checks << " checks, " << g_fail << " failing, " << since(t0) << " s" << std::endl;
    return g_fail;
}
