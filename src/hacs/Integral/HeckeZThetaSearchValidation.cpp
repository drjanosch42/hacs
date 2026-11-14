// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZThetaSearchValidation -- slot-27 TS-FIX / TS; see the header and docs/implementation/slot27/TS_design.md Sec. 9.

#include "HeckeZThetaSearchValidation.h"

#include "HeckeZRational.h"
#include "HeckeZRecord.h"
#include "HeckeZUp.h"
#include "../Campaign/CampaignHeckeZ.h"
#include "../Campaign/CampaignSpec.h"
#include "../HeckeExt/HeckeZRecordSource.h"
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using heckez::RationalResult;
using heckez::RationalSettings;
using heckez::ThetaOrder;
using heckez::record::HeckeStructureRecord;
using heckez::record::RecordSettings;

struct Tally {
    uint32_t checks = 0, failing = 0;
    bool check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failing;
            std::cout << "  [MISMATCH] " << what << "\n";
        }
        return ok;
    }
};

std::string Coeffs(const std::vector<uint32_t>& c) {
    std::ostringstream o;
    for (size_t i = 0; i < c.size(); ++i) o << (i ? "," : "") << c[i];
    return o.str();
}

std::string ThetaOf(const RationalResult& R) {
    return R.theta_index >= 0 ? Coeffs(R.thetas[static_cast<size_t>(R.theta_index)].coeffs) : std::string("-");
}

bool HasPrefix(const HeckeStructureRecord& R, const std::string& prefix) {
    for (const auto& f : R.fields)
        if (f.path.compare(0, prefix.size(), prefix) == 0) return true;
    return false;
}

std::string FieldOr(const HeckeStructureRecord& R, const std::string& path, const std::string& dflt = "(absent)") {
    const heckez::record::Field* f = R.find(path);
    return f ? f->value : dflt;
}

// the golden default record (1, 10) full (HZREC_report.md Sec. 10.2; settings fingerprint 0x8ffc29730c4f9672)
constexpr const char* kGold110 = "0x5e84b4c2540ff22a";
constexpr const char* kEd3Fingerprint = "0x8ffc29730c4f9672";

// ---- (c') TS-FIX: the g_P predicate ---------------------------------------------------------------------------------------------------
uint32_t ArmGP(Tally& T) {
    const uint32_t before = T.failing;
    // the predicate on its whole domain (RationalStepImpl writes exactly these five words)
    const std::vector<std::pair<std::string, bool>> dom = {{"upper_bound_certified", true},
                                                           {"partial(unfactored)", true},
                                                           {"open(no primitive T_p)", false},
                                                           {"open(C1)", false},
                                                           {"general_case", false}};
    for (const auto& [w, b] : dom) {
        RationalResult R;
        R.gP_status = w;
        T.check(heckez::GPIsBound(R) == b, "(c') GPIsBound(" + w + ") = " + (b ? "true" : "false"));
    }

    // the rational step (1, 10) on P = {2, 3, 5}: the control certifies T_2 (a primitive unit, g_P a bound); under the validation-only
    // hook the units are non-primitive and lex certifies the combination T_3 + T_5, with no g_P
    const std::vector<uint32_t> P = {2, 3, 5};
    RationalSettings rs;
    rs.theta_order = ThetaOrder::Lex;
    const RationalResult C = heckez::RationalStep(1, 10, P, rs);
    T.check(C.c1_status == "certified" && C.eis_rational && heckez::GPIsBound(C) && C.gP_status == "upper_bound_certified",
            "(c') control RationalStep(1, 10): certified, Eisenstein-rational, g_P upper_bound_certified (got " + C.c1_status + ", " +
                C.gP_status + ")");
    RationalSettings rsk;
    rsk.validation_skip_units = true;
    T.check(rsk.canonical_text() == rs.canonical_text(), "(c') the hook is not in HZ-RAT's settings text");
    const RationalResult Q = heckez::RationalStep(1, 10, P, rsk);
    std::cout << "  (c') RationalStep(1, 10, {2,3,5}) skip_units: C1=" << Q.c1_status << " theta=" << ThetaOf(Q)
              << " gP_status=" << Q.gP_status << " gP_ops=" << Q.gP_ops.size() << "\n";
    T.check(Q.c1_status == "certified" && ThetaOf(Q) == "0,1,1", "(c') skip_units: C1 certified by the combination T_3 + T_5 (got " +
                                                                    Q.c1_status + ", theta " + ThetaOf(Q) + ")");
    T.check(Q.gP_status == "open(no primitive T_p)" && !heckez::GPIsBound(Q) && Q.gP_ops.empty() && Q.gP.factors.empty(),
            "(c') skip_units: g_P open(no primitive T_p), not a bound, no factors");

    // the record (1, 10) full under the hook: no exception, G3 not_run, no bound word, no eis_single, no K3
    const hecke::ScopedCoefficientRing ring(0);   // coefficient_ring = Z
    const heckez::record::RecordSources src = heckez::record::DefaultRecordSources();
    RecordSettings Sk;
    Sk.rat.theta_order = ThetaOrder::Lex;   // explicit: the arm survives the default's flip
    Sk.rat.validation_skip_units = true;
    try {
        const HeckeStructureRecord R = heckez::record::BuildHeckeStructureRecord(1, 10, "full", Sk, src);
        std::cout << "  (c') record (1, 10) full skip_units: status=" << R.status << " theta=" << FieldOr(R, "G1.theta")
                  << " G3.partition=" << FieldOr(R, "G3.partition") << " G3.C_T.status=" << FieldOr(R, "G3.C_T.status")
                  << " G3.fibre_product=" << FieldOr(R, "G3.fibre_product") << "\n";
        T.check(R.status == "final", "(c') skip_units record is final (got " + R.status + ")");
        T.check(FieldOr(R, "G1.theta") == "0,1,1", "(c') skip_units record: G1.theta = 0,1,1");
        T.check(FieldOr(R, "G3.partition") == "eisenstein" && FieldOr(R, "G3.S") == "0" && FieldOr(R, "G3.S_prime") == "1",
                "(c') G3.partition = eisenstein, S = 0, S' = 1");
        T.check(FieldOr(R, "G3.C_T.status") == "not_run", "(c') G3.C_T.status = not_run");
        T.check(FieldOr(R, "G3.fibre_product") == "not_certified", "(c') G3.fibre_product = not_certified");
        for (const char* w : {"G3.C_T.upper", "G3.C_T.exact", "G3.C_T.cofactor", "G3.C_T.ell=", "G3.eta_L", "G3.second_route", "G9.K3"})
            T.check(!HasPrefix(R, w), std::string("(c') no ") + w + " word without a g_P bound");
        uint32_t routes = 0;
        for (const auto& f : R.fields)
            if (f.path.size() > 17 && f.path.compare(f.path.size() - 17, 17, ".generation.route") == 0) {
                ++routes;
                std::cout << "  (c') " << f.path << " = " << f.value << "\n";
                T.check(f.value != "eis_single", "(c') " + f.path + " is not eis_single");
            }
        // the root word at the congruence prime 691: read from the run (the read's own rule), printed, not expected
        for (const auto& f : R.fields)
            if (f.path.compare(0, 11, "G2.ell=691.") == 0 && f.path.find(".m[") == std::string::npos)
                std::cout << "  (c') " << f.path << " = " << f.value << "\n";
        T.check(routes > 0, "(c') the skip_units record has generation routes");
    } catch (const std::exception& e) {
        T.check(false, std::string("(c') the skip_units record (1, 10) threw: ") + e.what());
    }
    // the control: the hook off reproduces the golden (1, 10) record
    RecordSettings S0;
    S0.rat.theta_order = ThetaOrder::Lex;
    const HeckeStructureRecord R0 = heckez::record::BuildHeckeStructureRecord(1, 10, "full", S0, src);
    std::cout << "  (c') control record (1, 10) full: record_fnv1a64=" << R0.record_fnv1a64 << " fp=" << R0.settings_fingerprint
              << " G3.C_T.status=" << FieldOr(R0, "G3.C_T.status") << "\n";
    T.check(R0.record_fnv1a64 == kGold110 && R0.settings_fingerprint == kEd3Fingerprint,
            "(c') control (1, 10) full = the golden record " + std::string(kGold110));
    return T.failing - before;
}

// ---- shared ----------------------------------------------------------------------------------------------------------------------------
std::string SourcePath(const std::string& rel) {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/" + rel;
#else
    return rel;
#endif
}

// HZ-DS edition 3: key (N<N>_n<n>_<alg>) -> record_fnv1a64 (docs/integral_census/results/hecke_ed3/rows.tsv)
std::map<std::string, std::string> Ed3Hashes() {
    std::map<std::string, std::string> m;
    std::ifstream in(SourcePath("docs/integral_census/results/hecke_ed3/rows.tsv"));
    std::string line;
    int col = -1;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> f;
        std::string cell;
        std::istringstream ls(line);
        while (std::getline(ls, cell, '\t')) f.push_back(cell);
        if (col < 0) {
            for (size_t i = 0; i < f.size(); ++i)
                if (f[i] == "record_fnv1a64") col = static_cast<int>(i);
            continue;
        }
        if (col >= 0 && static_cast<size_t>(col) < f.size()) m[f[0]] = f[static_cast<size_t>(col)];
    }
    return m;
}

std::string Hex64(uint64_t v) {
    std::ostringstream o;
    o << "0x" << std::hex;
    o.width(16);
    o.fill('0');
    o << v;
    return o.str();
}

// the edition-3 settings text (the member hz_N1_n2_full_s8ffc29730c4f9672.json's `settings`, byte for byte; FNV-1a-64 0x8ffc29730c4f9672)
const char* const kEd3SettingsText =
    "coefficient_ring = Z\n"
    "heckez.schema = integral_hecke_structure/v1\n"
    "heckez.P_rule = descent primes (the first three of 2, 3, 5, 7, 11, 13 not dividing N), enlarged by the next primes not dividing N until "
    "(C1) is certified, |P| <= 6; read set at ell = P u {T_ell}, ell !| N, ell <= 23170\n"
    "heckez.theta_search+factor_budget = hz_rat/v1;trial_division_B=1000000;theta_cap=64;theta_coeff_max=binom(d,2);c1_probe_primes=3;"
    "rho=nf::factor_u64(untrusted;seeds=39;r<=2^26);pratt_nodes=256;pratt_witness_cap=64;big_mr_rounds=12(untrusted;probable);pool=fixed_point;"
    "multilimb_rho=none;ecm=none\n"
    "heckez.S_rule = every certified prime of gcd_j D_j (v_disc = 1 included: no LC-3b drop rule in v1), the torsion primes of the descent, "
    "the certified primes of g_P (or of the U_p bounds)\n"
    "heckez.K_read = start max(2, v_ell c(lambda) + 1), escalation steps <= 3; (C3) at K = 1 when the K0 read is not served\n"
    "heckez.maximal_ideals = decomposed iff generation is certified by saturation or rational and theta mod ell generates A'_1 (Krylov rank = "
    "dim A'_1 = d), theta = the read's own generating theta, else the first of theta_search's order over P, else (theta_search = P u {T_ell}) "
    "the first of theta_search's order over the read set P u {T_ell} when t_ell = added; else not_run (eis_single: not_run)\n"
    "heckez.irreducible = c_theta irreducible mod q for some prime q >= 3, the first 256 primes tried in order\n"
    "heckez.second_route = v_ell(B_k/k) mod ell^(e+2), e = v_ell(g_P), level 1, ell >= 5, (ell-1) !| k, BernoulliOverK; served iff "
    "ell^(e+5) < 2^128, ell^(e+2) + ell <= 2^84, ell^(e+2+V) + ell <= 2^110 (V = v_ell(4^(k/2) - 1)) and v < e+2; else v_Bk = not_run, "
    "agrees = not_read\n"
    "heckez.G8 = off (default off: no G8 field; another value is H1's Hida block per maximal ideal)\n"
    "heckez.bezout_record_cap = d <= 12, 64 KiB (v1 writes by_generator)\n";
// HZ-RAT's canonical text of editions 1--3 (the theta_search line above + its erased word_primes key)
const char* const kEd3RatText =
    "hz_rat/v1;trial_division_B=1000000;theta_cap=64;theta_coeff_max=binom(d,2);c1_probe_primes=3;rho=nf::factor_u64(untrusted;seeds=39;"
    "r<=2^26);pratt_nodes=256;pratt_witness_cap=64;big_mr_rounds=12(untrusted;probable);word_primes=CrtPrimes(<2^30,sized_from_bound,cap=2^21)";
// the default fingerprint of THIS build (the flip commit changes it with RationalSettings' default; before the flip = the lex text's)
constexpr const char* kDefaultFingerprint = "0x8ffc29730c4f9672";
// the lex_then_greedy text's fingerprint (TS_design.md Sec. 6.1 draft spelling, s27/tsfix/doc/settings_new_v02.txt)
constexpr const char* kLexThenGreedyFingerprint = "0x325b510edad11c8d";
const char* const kThetaKey = ";theta_order=lex_then_greedy(t=2654435761;steps_without_U_p:|P|>=p_rule_cap)";
const char* const kMaxIdealsClause =
    "; else (theta_order = lex_then_greedy) the greedy continuation over P, then over the read set (score: Krylov rank, target t mod ell, "
    "c <= min(binom(d,2), ell - 1), at most theta_cap tests per list)";

std::string SettingsTextOf(const RecordSettings& S) {
    return hecke::math_settings().canonical_text() + S.caps_text();
}

// ---- (a) the text rule --------------------------------------------------------------------------------------------------------------------
uint32_t ArmText(Tally& T) {
    const uint32_t before = T.failing;
    const hecke::ScopedCoefficientRing ring(0);   // coefficient_ring = Z
    RationalSettings lex;
    lex.theta_order = ThetaOrder::Lex;   // explicit (after the flip RationalSettings{} is no longer lex; audit M3)
    T.check(lex.canonical_text() == kEd3RatText, "(a) canonical_text() at explicit lex = the edition-1--3 literal, byte for byte");
    RecordSettings Sl;
    Sl.rat.theta_order = ThetaOrder::Lex;
    const std::string tl = SettingsTextOf(Sl);
    T.check(tl == kEd3SettingsText, "(a) caps text at explicit lex = the edition-3 member's settings string, byte for byte");
    T.check(Hex64(engine::FnvBytes(engine::kFnvOffset64, tl)) == "0x8ffc29730c4f9672", "(a) its fingerprint 0x8ffc29730c4f9672");
    // lex_then_greedy: exactly the two insertions
    RationalSettings ltg;
    ltg.theta_order = ThetaOrder::LexThenGreedy;
    std::string exp_rat = kEd3RatText;
    const std::string after = ";theta_coeff_max=binom(d,2)";
    exp_rat.insert(exp_rat.find(after) + after.size(), kThetaKey);
    T.check(ltg.canonical_text() == exp_rat, "(a) canonical_text() at lex_then_greedy = the lex text + the theta_order key after theta_coeff_max");
    RecordSettings Sg;
    Sg.rat.theta_order = ThetaOrder::LexThenGreedy;
    const std::string tg = SettingsTextOf(Sg);
    std::string exp = kEd3SettingsText;
    exp.insert(exp.find(after) + after.size(), kThetaKey);
    const std::string mi_tail = "; else not_run (eis_single: not_run)\n";
    exp.insert(exp.find(mi_tail), kMaxIdealsClause);
    T.check(tg == exp, "(a) caps text at lex_then_greedy = the lex text + exactly the two insertions (theta_order key, maximal_ideals clause)");
    const std::string fg = Hex64(engine::FnvBytes(engine::kFnvOffset64, tg));
    std::cout << "  (a) fingerprints: lex 0x8ffc29730c4f9672, lex_then_greedy " << fg << "\n";
    T.check(fg == kLexThenGreedyFingerprint, "(a) lex_then_greedy fingerprint = " + std::string(kLexThenGreedyFingerprint) + " (TS_design.md Sec. 6.1 draft)");
    // the default of this build
    const std::string fd = Hex64(engine::FnvBytes(engine::kFnvOffset64, SettingsTextOf(RecordSettings{})));
    T.check(fd == kDefaultFingerprint, "(a) the default settings fingerprint = " + std::string(kDefaultFingerprint) + " (got " + fd + ")");
    // the unprinted fields: p_rule_cap's copy, threads, the TS-FIX hook
    RationalSettings u = lex;
    u.p_rule_cap = 9;
    u.threads = 7;
    u.validation_skip_units = true;
    T.check(u.canonical_text() == lex.canonical_text(), "(a) p_rule_cap's copy, threads and the TS-FIX hook are not printed");
    // names
    ThetaOrder o = ThetaOrder::Lex;
    T.check(heckez::ParseThetaOrder("lex_then_greedy", &o) && o == ThetaOrder::LexThenGreedy && heckez::ParseThetaOrder("lex", &o) &&
                o == ThetaOrder::Lex && !heckez::ParseThetaOrder("greedy", &o) && !heckez::ParseThetaOrder("", &o),
            "(a) ParseThetaOrder: lex, lex_then_greedy; nothing else");
    // the campaign spec: absent key = the default; explicit lex; explicit lex_then_greedy; refused outside structure = hecke
    auto plan_id = [&](const std::string& extra) -> std::string {
        std::istringstream in("name = ts\noutput = /nonexistent/ts\ncoefficient_ring = Z\nstructure = hecke\n" + extra + "n_sym = 10\n");
        const campaign::CampaignSpec spec = campaign::ParseCampaignSpec(in, "ts.spec");
        const auto plan = campaign::PlanHeckeZCells(spec);
        return plan.empty() ? std::string("(none)") : plan[0].id;
    };
    const std::string id_absent = plan_id(""), id_lex = plan_id("hz_theta_order = lex\n"), id_ltg = plan_id("hz_theta_order = lex_then_greedy\n");
    std::cout << "  (a) spec ids: absent " << id_absent << ", lex " << id_lex << ", lex_then_greedy " << id_ltg << "\n";
    T.check(id_lex == "hz_N1_n10_full_s8ffc29730c4f9672", "(a) spec hz_theta_order = lex -> the edition-3 id hz_N1_n10_full_s8ffc29730c4f9672");
    T.check(id_absent == "hz_N1_n10_full_s" + std::string(kDefaultFingerprint).substr(2), "(a) spec without the key -> the default's id");
    T.check(id_ltg == "hz_N1_n10_full_s" + std::string(kLexThenGreedyFingerprint).substr(2), "(a) spec hz_theta_order = lex_then_greedy -> its own id");
    for (const std::string& bad : {std::string("coefficient_ring = Z\nhz_theta_order = lex\nn_sym = 10\n"),
                                   std::string("hz_theta_order = lex\nn_sym = 10\n"),
                                   std::string("coefficient_ring = Z\nstructure = hecke\nhz_theta_order = greedy\nn_sym = 10\n")}) {
        bool refused = false;
        try {
            std::istringstream in("name = ts\noutput = /nonexistent/ts\n" + bad);
            (void)campaign::ParseCampaignSpec(in, "ts.spec");
        } catch (const std::exception& e) {
            refused = std::string(e.what()).find("hz_theta_order") != std::string::npos;
        }
        T.check(refused, "(a) spec refused: " + bad.substr(0, bad.find('\n', bad.find("hz_theta")) ));
    }
    return T.failing - before;
}

// ---- (b), (c) the orders on the probe cells (site 1) --------------------------------------------------------------------------------------
struct Probe {
    uint32_t N, n;
    std::vector<uint32_t> P;          // the T_p of the full step (G1.P)
    std::string lex_theta;            // "-" = open(cap)
    std::string ltg_theta;
};
const std::vector<Probe>& Probes() {
    static const std::vector<Probe> v = {
        {9, 0, {2, 5, 7}, "0,1,0,1", "0,1,0,1"},          {18, 0, {5, 7, 11}, "0,0,0,1,2", "0,0,0,1,2"},
        {25, 0, {2, 3, 7}, "0,0,1,1", "0,0,1,1"},         {49, 0, {2, 3, 5}, "0,0,1,1", "0,0,1,1"},
        {25, 2, {2, 3, 7}, "-", "1,0,0,2"},               {45, 0, {2, 7, 11}, "-", "1,0,0,1,2"},
    };
    return v;
}

uint64_t Binom2(int64_t d) { return d >= 2 ? uint64_t(d) * uint64_t(d - 1) / 2 : 1; }

uint32_t ArmOrders(Tally& T, uint32_t threads) {
    const uint32_t before = T.failing;
    for (const Probe& c : Probes()) {
        const std::vector<uint32_t> U = heckez::up::PrimesDividing(c.N);
        RationalSettings sl, sg;
        sl.theta_order = ThetaOrder::Lex;
        sg.theta_order = ThetaOrder::LexThenGreedy;
        sl.threads = sg.threads = threads;
        const RationalResult A = heckez::RationalStepFull(c.N, c.n, c.P, U, sl);
        const RationalResult G = heckez::RationalStepFull(c.N, c.n, c.P, U, sg);
        const std::string cell = "(" + std::to_string(c.N) + ", " + std::to_string(c.n) + ")";
        std::cout << "  (b/c) " << cell << " d=" << A.d_target << " m=" << A.P.size() << " lex: " << A.c1_status << " theta=" << ThetaOf(A)
                  << " cands=" << A.thetas.size() << "; lex_then_greedy: " << G.c1_status << " theta=" << ThetaOf(G)
                  << " cands=" << G.thetas.size() << "\n";
        T.check(ThetaOf(A) == c.lex_theta, "(b) " + cell + " lex theta = " + c.lex_theta + " (got " + ThetaOf(A) + ")");
        if (c.lex_theta == "-")
            T.check(A.c1_status == "open(cap)" && A.thetas.size() == 64, "(b) " + cell + " lex ends open(cap) after 64 candidates");
        T.check(G.c1_status == "certified" && ThetaOf(G) == c.ltg_theta,
                "(c) " + cell + " lex_then_greedy certifies theta = " + c.ltg_theta + " (got " + G.c1_status + " " + ThetaOf(G) + ")");
        // phase (a) unchanged: the continuation's candidates follow lex's, which it never re-lists
        bool prefix = G.thetas.size() >= A.thetas.size();
        for (size_t i = 0; prefix && i < A.thetas.size(); ++i)
            prefix = G.thetas[i].coeffs == A.thetas[i].coeffs && G.thetas[i].primitive == A.thetas[i].primitive;
        T.check(prefix, "(c) " + cell + " lex_then_greedy's candidate list begins with lex's, unchanged");
        std::set<std::vector<uint32_t>> seen;
        bool dup = false;
        for (const auto& t : G.thetas) dup = dup || !seen.insert(t.coeffs).second;
        T.check(!dup, "(c) " + cell + " no candidate is listed twice");
        // (d) on the record cells: the bound theta_cap + m (B + 1), every coefficient <= binom(d, 2)
        const uint64_t B = Binom2(G.d_target), m = G.P.size();
        bool within = true;
        for (const auto& t : G.thetas)
            for (uint32_t x : t.coeffs) within = within && x <= B;
        T.check(G.thetas.size() <= sl.theta_cap + m * (B + 1) && within,
                "(d) " + cell + " candidates <= theta_cap + m (B + 1) and every c_i <= binom(d, 2)");
        T.check(A.text().find("hz_rat/v1") != std::string::npos && A.settings == sl.canonical_text() && G.settings == sg.canonical_text(),
                "(b/c) " + cell + " the step's settings text is its RationalSettings' canonical text");
    }
    return T.failing - before;
}

// P_rule as BuildHeckeStructureRecord runs it (descent primes, enlarged until (C1) is certified or |P| = p_rule_cap)
RationalResult PRule(uint32_t N, uint32_t n, const RationalSettings& rs, std::vector<uint32_t>* Pout) {
    std::vector<uint32_t> P;
    for (uint32_t p : {2u, 3u, 5u, 7u, 11u, 13u})
        if (N % p != 0 && P.size() < 3) P.push_back(p);
    RationalResult Q;
    for (;;) {
        Q = heckez::RationalStep(N, n, P, rs);
        if (Q.c1_status == "certified" || P.size() >= rs.p_rule_cap) break;
        uint32_t p = P.back() + 1;
        for (;; ++p) {
            bool prime = p >= 2;
            for (uint32_t q = 2; q * q <= p && prime; ++q) prime = p % q != 0;
            if (prime && N % p != 0) break;
        }
        P.push_back(p);
    }
    *Pout = P;
    return Q;
}

std::string Join(const std::vector<uint32_t>& v) { return Coeffs(v); }

uint32_t ArmLastRound(Tally& T, uint32_t threads) {
    const uint32_t before = T.failing;
    RationalSettings sg;
    sg.theta_order = ThetaOrder::LexThenGreedy;
    sg.threads = threads;
    // (341, 0) anemic: P_rule keeps edition 3's P = {2, 3, 5, 7} and theta = T_7 (the continuation does not run at |P| < p_rule_cap)
    std::vector<uint32_t> P;
    const RationalResult A = PRule(341, 0, sg, &P);
    std::cout << "  (c) (341, 0) anemic lex_then_greedy: P=" << Join(P) << " " << A.c1_status << " theta=" << ThetaOf(A) << "\n";
    T.check(Join(P) == "2,3,5,7" && ThetaOf(A) == "0,0,0,1", "(c) last-round rule: (341, 0) anemic keeps P = {2,3,5,7}, theta = T_7");
    // (147, 0) anemic: lex fails at every round, the continuation certifies at the 6-prime P
    const RationalResult B = PRule(147, 0, sg, &P);
    std::cout << "  (c) (147, 0) anemic lex_then_greedy: P=" << Join(P) << " " << B.c1_status << " theta=" << ThetaOf(B) << "\n";
    T.check(Join(P) == "2,5,11,13,17,19" && B.c1_status == "certified" && ThetaOf(B) == "1,2,0,0,0,0",
            "(c) (147, 0) anemic certifies at P = {2,5,11,13,17,19} with theta = T_2 + 2T_5");
    // a direct call with |P| < p_rule_cap is lex alone
    const RationalResult C = heckez::RationalStep(147, 0, {2, 5, 11}, sg);
    T.check(C.c1_status == "open(cap)" && C.thetas.size() == sg.theta_cap,
            "(c) a direct RationalStep(147, 0, {2,5,11}) at |P| < p_rule_cap is lex alone: open(cap) after theta_cap candidates");
    return T.failing - before;
}

// ---- (d) the continuations as pure functions ---------------------------------------------------------------------------------------------
// r(c) = the number of distinct values of sum_i c_i X_i(sigma) over the systems sigma (exact, over Z)
struct Table {
    std::vector<std::vector<int64_t>> X;   // X[i][sigma]
    int d() const { return X.empty() ? 0 : static_cast<int>(X[0].size()); }
    int score(const std::vector<uint32_t>& c) const {
        std::set<__int128> vals;
        for (int s = 0; s < d(); ++s) {
            __int128 v = 0;
            for (size_t i = 0; i < X.size(); ++i) v += static_cast<__int128>(c[i]) * X[i][static_cast<size_t>(s)];
            vals.insert(v);
        }
        return static_cast<int>(vals.size());
    }
    int classes(size_t upto) const {   // joint classes of (X_1 .. X_upto)
        std::set<std::vector<int64_t>> cl;
        for (int s = 0; s < d(); ++s) {
            std::vector<int64_t> k;
            for (size_t i = 0; i < upto; ++i) k.push_back(X[i][static_cast<size_t>(s)]);
            cl.insert(k);
        }
        return static_cast<int>(cl.size());
    }
};

// the reference least-argmax construction of TS_design.md Sec. 2.3 with the exact targets M_i (no target probe)
heckez::ContinuationResult Reference(const Table& tb, uint32_t B) {
    heckez::ContinuationResult out;
    const uint32_t m = static_cast<uint32_t>(tb.X.size());
    std::vector<uint32_t> th(m, 0);
    int r = 1;
    for (uint32_t i = 0; i < m; ++i) {
        const int M = tb.classes(i + 1);
        if (M <= r) continue;
        for (uint32_t c = 1; c <= B; ++c) {
            std::vector<uint32_t> v = th;
            v[i] = c;
            const int x = tb.score(v);
            if (x == tb.d()) { out.found = true; out.theta = v; return out; }
            if (x >= M) { th = v; r = x; break; }
        }
    }
    return out;
}

uint32_t ArmPure(Tally& T) {
    const uint32_t before = T.failing;
    std::mt19937_64 rng(0x7453u);   // fixed seed: deterministic
    uint32_t sep_ok = 0, sep_n = 0;
    for (int trial = 0; trial < 60; ++trial) {
        const int d = 3 + static_cast<int>(rng() % 10);
        const uint32_t m = 2 + static_cast<uint32_t>(rng() % 4);
        Table tb;
        tb.X.assign(m, std::vector<int64_t>(static_cast<size_t>(d)));
        for (auto& row : tb.X)
            for (auto& x : row) x = static_cast<int64_t>(rng() % 7) - 3;   // small values: many coincidences per operator
        if (tb.classes(m) != d) continue;                                   // separating tables only
        ++sep_n;
        const uint32_t B = static_cast<uint32_t>(Binom2(d));
        std::map<std::vector<uint32_t>, int> tested;
        for (uint32_t i = 0; i < m; ++i) {   // phase (a)'s units
            std::vector<uint32_t> e(m, 0);
            e[i] = 1;
            tested[e] = tb.score(e);
        }
        bool retested = false;
        const heckez::ContinuationResult G =
            heckez::GreedyContinuation(m, d, B, heckez::kThetaGreedyTarget, [&](const std::vector<uint32_t>& c) {
                if (tested.count(c)) retested = true;
                return tb.score(c);
            }, tested);
        const heckez::ContinuationResult R = Reference(tb, B);
        bool ok = G.found && tb.score(G.theta) == d && R.found && G.theta == R.theta && G.tests <= uint64_t(m) * (B + 1) && !retested;
        for (uint32_t x : G.theta) ok = ok && x <= B;
        if (!ok)
            std::cout << "  (d) separating table " << trial << " d=" << d << " m=" << m << " found=" << G.found << " theta=" << Coeffs(G.theta)
                      << " ref=" << Coeffs(R.theta) << " tests=" << G.tests << "\n";
        sep_ok += ok;
    }
    std::cout << "  (d) separating tables: " << sep_ok << "/" << sep_n << " primitive, = the least-argmax construction, <= m (B + 1) tests\n";
    T.check(sep_n >= 20 && sep_ok == sep_n, "(d) every random separating table: a primitive theta in the grid, the reference construction's, "
                                            "within m (B + 1) tests, no vector of `tested` re-tested");
    // negative controls (audit m4): a duplicated operator, two systems agreeing on every operator
    {
        Table tb;
        tb.X = {{0, 1, 2, 2}, {0, 1, 2, 2}, {5, 1, 4, 4}};   // systems 3 and 4 agree on every operator
        const uint32_t B = static_cast<uint32_t>(Binom2(4));
        const heckez::ContinuationResult G = heckez::GreedyContinuation(3, 4, B, heckez::kThetaGreedyTarget,
                                                                          [&](const std::vector<uint32_t>& c) { return tb.score(c); }, {});
        T.check(!G.found && G.tests <= 3 * (B + 1), "(d) negative control (duplicated operator, two systems never separated): none, <= m (B + 1) tests");
    }
    // the caveat: t an excluded value (theta_1 + t X_2 merges a pair theta_1 separates), so step 2 sees no gain and the pass ends without a
    // primitive theta on a separating list -- documented behaviour of the one pass (TS_design.md Sec. 2.3), not an error
    {
        const int64_t t = heckez::kThetaGreedyTarget;
        Table tb;
        tb.X = {{0, t, 0}, {0, -1, 1}};
        const heckez::ContinuationResult G = heckez::GreedyContinuation(2, 3, 3, heckez::kThetaGreedyTarget,
                                                                          [&](const std::vector<uint32_t>& c) { return tb.score(c); }, {});
        std::cout << "  (d) caveat table (t excluded at step 2): found=" << G.found << " tests=" << G.tests << " (the list separates: classes="
                  << tb.classes(2) << ")\n";
        T.check(tb.classes(2) == 3 && !G.found, "(d) caveat: t excluded -> the one pass ends without a primitive theta (documented)");
    }
    // the site-2 continuation: the cap counts every test (target probes included) and ends the list
    {
        uint32_t calls = 0;
        const heckez::ContinuationResult K =
            heckez::KrylovContinuation(4, 10, 100, 1000003, 7, 64, [&](const std::vector<uint32_t>&) { ++calls; return 1u; });
        T.check(!K.found && K.tests == 4 && calls == 4, "(e) Krylov continuation without any gain: one target probe per operator, then none");
        calls = 0;
        // a score that rises by one at every scan step but never reaches D1: the scan runs into the cap
        uint32_t level = 1;
        const heckez::ContinuationResult C = heckez::KrylovContinuation(
            3, 1000, 5000, 1000003, 7, 64, [&](const std::vector<uint32_t>&) -> uint32_t {
                ++calls;
                return calls == 1 ? 999u : ++level;   // the first target probe promises 999; every scan step gains one
            });
        T.check(!C.found && C.tests == 64 && calls == 64, "(e) a list whose continuation reaches the cap stops with exactly theta_cap = 64 tests");
        // tbar and the scan bound min(B2, ell - 1): at ell = 3 only c = 1, 2 are scanned
        std::vector<uint32_t> cs;
        const heckez::ContinuationResult S = heckez::KrylovContinuation(1, 5, 100, 3, 1, 64, [&](const std::vector<uint32_t>& c) -> uint32_t {
            cs.push_back(c[0]);
            return c[0] == 1 && cs.size() == 1 ? 4u : 2u;   // target 4, never reached
        });
        T.check(!S.found && cs.size() == 3 && cs[1] == 1 && cs[2] == 2, "(e) the scan stops at min(B2, ell - 1) = 2 at ell = 3");
    }
    return T.failing - before;
}

// ---- (e) site 2 on records; (b) / (f) record byte identity under lex and route identity ---------------------------------------------------
uint32_t ArmSite2(Tally& T, uint32_t threads) {
    const uint32_t before = T.failing;
    const hecke::ScopedCoefficientRing ring(0);   // coefficient_ring = Z
    const heckez::record::RecordSources src = heckez::record::DefaultRecordSources();
    const std::map<std::string, std::string> ed3 = Ed3Hashes();
    struct Cell {
        uint32_t N, n;
        const char* alg;
        uint64_t ell;
        const char* lex_mi;        // maximal_ideals at ell under lex
        const char* ltg_mi;        // ... under lex_then_greedy
    };
    // (2, 6) full: the full-N site; (77, 0) anemic: the level-1 / anemic site (TS_design.md Sec. 3: both decompose at the named ell only
    // under the continuation); (5, 8) full at 107: decomposed by lex, kept (the last-resort placement)
    const std::vector<Cell> cells = {{2, 6, "full", 3, "not_run", "certified"},
                                     {77, 0, "anemic", 7, "not_run", "certified"},
                                     {5, 8, "full", 107, "certified", "certified"}};
    for (const Cell& c : cells) {
        RecordSettings Sl, Sg;
        Sl.rat.theta_order = ThetaOrder::Lex;
        Sg.rat.theta_order = ThetaOrder::LexThenGreedy;
        Sl.threads = Sg.threads = threads;
        const HeckeStructureRecord L = heckez::record::BuildHeckeStructureRecord(c.N, c.n, c.alg, Sl, src);
        const HeckeStructureRecord G = heckez::record::BuildHeckeStructureRecord(c.N, c.n, c.alg, Sg, src);
        const std::string key = "N" + std::to_string(c.N) + "_n" + std::to_string(c.n) + "_" + c.alg;
        const std::string pre = "G2.ell=" + std::to_string(c.ell) + ".";
        std::cout << "  (e) " << key << " ell=" << c.ell << ": lex " << FieldOr(L, pre + "maximal_ideals") << " " << FieldOr(L, pre + "decomposed_by")
                  << " record=" << L.record_fnv1a64 << "; lex_then_greedy " << FieldOr(G, pre + "maximal_ideals") << " "
                  << FieldOr(G, pre + "decomposed_by") << " record=" << G.record_fnv1a64 << " fp=" << G.settings_fingerprint << "\n";
        const auto it = ed3.find(key);
        T.check(it != ed3.end() && L.record_fnv1a64 == it->second, "(b) " + key + " under explicit lex = edition 3's record_fnv1a64");
        T.check(FieldOr(L, pre + "maximal_ideals") == c.lex_mi, "(e) " + key + " lex: maximal_ideals at " + std::to_string(c.ell) + " = " + c.lex_mi);
        T.check(FieldOr(G, pre + "maximal_ideals") == c.ltg_mi,
                "(e) " + key + " lex_then_greedy: maximal_ideals at " + std::to_string(c.ell) + " = " + c.ltg_mi);
        if (std::string(c.lex_mi) == "certified")
            T.check(FieldOr(L, pre + "decomposed_by") == FieldOr(G, pre + "decomposed_by"),
                    "(e) " + key + " keeps decomposed_by at " + std::to_string(c.ell) + " (the continuation only turns not_run into certified)");
        T.check(G.settings_fingerprint == kLexThenGreedyFingerprint, "(e) " + key + " lex_then_greedy record carries its own fingerprint");
        // every word moved by the continuation is a maximal_ideals word of an ideal lex left not_run (R2), or the settings
        std::map<std::string, std::string> lf, gf;
        for (const auto& f : L.fields) lf[f.path] = f.value;
        for (const auto& f : G.fields) gf[f.path] = f.value;
        std::set<std::string> moved_ells;
        bool other = false;
        for (const auto& [p, v] : gf) {
            const auto jt = lf.find(p);
            if (jt != lf.end() && jt->second == v) continue;
            if (p.compare(0, 7, "G2.ell=") == 0) {
                const std::string ells = p.substr(7, p.find('.', 7) - 7);
                if (FieldOr(L, "G2.ell=" + ells + ".maximal_ideals") == "not_run") { moved_ells.insert(ells); continue; }
            }
            if (p.compare(0, 3, "G0.") == 0 || p.compare(0, 4, "G10.") == 0 || p.compare(0, 3, "G9.") == 0) continue;   // status / provenance
            other = true;
            std::cout << "  (e) " << key << " moved outside R2: " << p << " lex=" << (jt == lf.end() ? "(absent)" : jt->second) << " ltg=" << v << "\n";
        }
        for (const auto& [p, v] : lf)
            if (!gf.count(p) && p.compare(0, 4, "G10.") != 0) { other = true; std::cout << "  (e) " << key << " dropped: " << p << "\n"; }
        T.check(!other, "(e) " + key + ": the continuation moves only maximal_ideals words of ideals lex left not_run (rule R2)");
    }
    return T.failing - before;
}

uint32_t ArmRoute(Tally& T) {
    const uint32_t before = T.failing;
    // (f) the search and the record at threads 1 vs 16: equal candidate lists and record hashes (the search itself is sequential: one
    // candidate at a time, no batch knob; the threads are the word-prime parallelism)
    RationalSettings s1, s16;
    s1.theta_order = s16.theta_order = ThetaOrder::LexThenGreedy;
    s1.threads = 1;
    s16.threads = 16;
    const RationalResult A = heckez::RationalStepFull(25, 2, {2, 3, 7}, {5}, s1);
    const RationalResult B = heckez::RationalStepFull(25, 2, {2, 3, 7}, {5}, s16);
    bool same = A.thetas.size() == B.thetas.size();
    for (size_t i = 0; same && i < A.thetas.size(); ++i) same = A.thetas[i].coeffs == B.thetas[i].coeffs && A.thetas[i].primitive == B.thetas[i].primitive;
    T.check(same && A.digest() == B.digest(), "(f) (25, 2) full step under lex_then_greedy: threads 1 vs 16 give equal R.thetas and digest");
    const hecke::ScopedCoefficientRing ring(0);
    const heckez::record::RecordSources src = heckez::record::DefaultRecordSources();
    RecordSettings S1, S16;
    S1.rat.theta_order = S16.rat.theta_order = ThetaOrder::LexThenGreedy;
    S1.threads = 1;
    S16.threads = 16;
    const HeckeStructureRecord R1 = heckez::record::BuildHeckeStructureRecord(2, 6, "full", S1, src);
    const HeckeStructureRecord R16 = heckez::record::BuildHeckeStructureRecord(2, 6, "full", S16, src);
    T.check(R1.record_fnv1a64 == R16.record_fnv1a64 && R1.cert_sha256 == R16.cert_sha256,
            "(f) (2, 6) full record under lex_then_greedy: threads 1 vs 16 hash-equal");
    return T.failing - before;
}

}  // namespace

uint32_t HeckeZThetaSearchValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeZThetaSearchValidation: the theta search of HZ-RAT and heckez.maximal_ideals (slot-27 TS-FIX / TS; "
                 "TS_design.md Sec. 9; [C] item 8 (j))\n";
    Tally T;
    const uint32_t threads = 4;
    auto word = [](uint32_t f) { return f == 0 ? "OK" : "FAIL"; };
    const uint32_t f_text = ArmText(T);
    const uint32_t f_ord = ArmOrders(T, threads);
    const uint32_t f_last = ArmLastRound(T, threads);
    const uint32_t f_gp = ArmGP(T);
    const uint32_t f_pure = ArmPure(T);
    const uint32_t f_site2 = ArmSite2(T, threads);
    const uint32_t f_route = ArmRoute(T);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool ok = T.failing == 0;
    std::cout << "HECKEZ_THETA_SEARCH text=" << word(f_text) << " orders=" << word(f_ord) << " last_round=" << word(f_last)
              << " gp=" << word(f_gp) << " pure=" << word(f_pure) << " site2=" << word(f_site2) << " route=" << word(f_route)
              << " default_fp=" << kDefaultFingerprint << " checks=" << T.checks << " failing=" << T.failing << " secs=" << secs
              << " verdict=" << (ok ? "OK" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
