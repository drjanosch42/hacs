// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, L23-1) OrbifoldCampaignL23Validation -- see CampaignOrbifoldValidation.h.
//
// Parts:
//   (A) spec surface: a spec without `orbifold` (and one with `orbifold = off`) plans the same jobs with the same ids
//       ("n16_ell5_p2_r1"), `orbifold` disabled; the refusals at spec validation, before any cell runs: ell = 5
//       (not_implemented_in_this_build), a `p` axis, `ext = true`, a non-default coefficient_ring, jobs = 2, a cell
//       beyond the reader's precision range, an unknown model; RunCampaignSpec refuses an orbifold spec.
//   (B) identity: the default fingerprint is 0 and the id "orb_ell2_n10"; a non-default orbifold_primes, a non-default
//       mathematical setting (locality = budget=32) and orbifold_m_target each give a distinct `_s<16 hex>` id and a
//       distinct record hash input (the settings block), while the computed dims are unchanged.
//   (C) the end-to-end campaign at ell = 2 (n = 0 .. 60) and ell = 3 (n = 0 .. 60): every cell `ok` (no G gate failing,
//       no fallback), refused = 0, fallback = 0; every file complete and its record_fnv1a64 the FNV of its body; then
//         * even n >= 2: the reader's columns against the frozen ladic2.py rows (Bin3Oracle.inc) -- rank, torsion,
//           ed_tp, ed_eis (finite part and the one Eisenstein `inf`), ed_tell (the cusp part);
//         * even n >= 2: the three level-1 modular-forms sum rules (Bin3SumRules.inc): sum ed_tell = 2 v det T_ell|S_k,
//           sum ed_tp = 2 v det(T_p - (1+p))|S_k + v(p^n - 1), sum of the finite ed_eis = 2 v det(T_p - (1+p^{n+1}))|S_k;
//         * odd n at ell = 2: the frozen oddn_l2.py rows -- h0, h1, dim H^1(V), dim H^2(V), rank T_2 on H^1(M), T_2 a
//           rank-one idempotent there, rank T_2 on red H^1(V);
//         * the closed forms of [C] sec:oddnltwo and [P3] Sec. 2 are the record's own G gates (counted here);
//       G-conj findings are printed and counted, never failed.
//       L23-1 audit repairs: every gate carries a source class (R2; G-conj = CONJ, G never CONJ); the amalgam oracle
//       ran on the 90 cells n >= 1 (ell = 2) / even n >= 2 (ell = 3) with its five cross-model G gates (R3); the R5
//       gate e_cup_h0_h2_injective on every ell = 2 cell with h0 > 0; the hashed body carries no precision path and
//       is unchanged when rung / K / m / e_A / e_1 / reruns change (R4); every file carries reader_path and the
//       provenance block (commit, binary sha256; R6).
//   (D) resume (a second run skips every cell), determinism (two cells recomputed: identical bodies, the sum-rule
//       oracle's wall time masked -- GPU_scout.md H-1, KNOWN_ISSUES Sec. 7); resume over a planted `failed` record STOPS there (R1, [P3] R7: stopped, failed = 1, nothing run past it, exit 2, the
//       record's status carried into index.json / summary.tsv), while a planted complete `fallback` record is skipped.
//   (E) negative controls: orbifold_m_target = 3 at (2, 10) is a `fallback` cell with an `_s` id; a tampered oracle row
//       is caught by the comparison.
// Last line: ORBIFOLD_L23 ... checks= failing= verdict=OK.
#include "CampaignOrbifoldValidation.h"

#include "CampaignDriver.h"
#include "CampaignOrbifold.h"
#include "CampaignSpec.h"
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/TextIo.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct Ladic2Row { uint32_t ell, n, K, comm_v; const char* line; };
const Ladic2Row kLadic2[] = {
#define BIN3_LADIC2_ROWS
#include "../Orbifold/L23/Bin3Oracle.inc"
#undef BIN3_LADIC2_ROWS
};
struct SumRuleRow { uint32_t ell, n; int vTl, vTl2, hodge; int vP[4], vE[4], vN[4]; };
const SumRuleRow kSumRules[] = {
#include "../Orbifold/L23/Bin3SumRules.inc"
};
struct OddnOracleRow { uint32_t n; uint32_t v[22]; };
const OddnOracleRow kOddn[] = {
#define BIN3_ODDN_ROWS
#include "../Orbifold/L23/Bin3Oracle.inc"
#undef BIN3_ODDN_ROWS
};

uint32_t g_checks = 0, g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
std::string str(uint64_t x) { return std::to_string(x); }

std::string TempDir(const std::string& tag) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    const fs::path dir = (ec ? fs::path(".") : base) / ("oc_campaign_orbifold_" + tag + "_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    return dir.string();
}
campaign::CampaignSpec Spec(const std::string& text, const std::string& out = "unused") {
    std::istringstream is(text);
    campaign::CampaignSpec s = campaign::ParseCampaignSpec(is, "<l231-spec>");
    s.output_dir = out;
    return s;
}
// 0 = parsed, 1 = runtime_error (spec error), 2 = not_implemented_in_this_build
int ParseOutcome(const std::string& text, std::string* msg = nullptr) {
    try {
        (void)Spec(text);
        return 0;
    } catch (const hecke::NotImplementedInThisBuild& e) {
        if (msg) *msg = e.what();
        return 2;
    } catch (const std::exception& e) {
        if (msg) *msg = e.what();
        return 1;
    }
}
using oc_util::ReadFile;   // S24 CL-7 (F11): Util/TextIo.h

// A record divisor string ("0,3,3,inf,>=40" / "-") -> (positive finite ascending, count of non-finite, units).
struct Parsed { std::vector<uint32_t> pos; uint32_t nonfinite = 0, units = 0; };
Parsed ParseDivs(const std::string& s) {
    Parsed P;
    if (s == "-" || s.empty()) return P;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) {
        if (t == "inf" || t.rfind(">=", 0) == 0) ++P.nonfinite;
        else if (t == "0") ++P.units;
        else P.pos.push_back(static_cast<uint32_t>(std::stoul(t)));
    }
    std::sort(P.pos.begin(), P.pos.end());
    return P;
}
// A ladic2.py field ("3,3,3", "-", "8,8+[>=240]^1", "-+[>=240]^1") -> (positive finite ascending, ">=" count).
Parsed ParseLadic2(const std::string& f) {
    Parsed P;
    std::string a = f;
    const size_t plus = f.find("+[");
    if (plus != std::string::npos) {
        a = f.substr(0, plus);
        const size_t hat = f.find("]^", plus);
        P.nonfinite = static_cast<uint32_t>(std::stoul(f.substr(hat + 2)));
    }
    if (a != "-" && !a.empty()) {
        std::stringstream ss(a);
        std::string t;
        while (std::getline(ss, t, ',')) P.pos.push_back(static_cast<uint32_t>(std::stoul(t)));
    }
    std::sort(P.pos.begin(), P.pos.end());
    return P;
}
std::string Field(const std::string& line, const std::string& key) {   // " key:VALUE" up to the next blank
    const std::string k = " " + key + ":";
    const size_t a = line.find(k);
    if (a == std::string::npos) return "?";
    const size_t b = line.find(' ', a + k.size());
    return line.substr(a + k.size(), b == std::string::npos ? std::string::npos : b - a - k.size());
}
std::vector<uint32_t> TorsionOf(const std::string& line) {   // "T1=[2, 1]"
    std::vector<uint32_t> v;
    const size_t a = line.find("T1=[");
    const size_t b = line.find(']', a);
    std::stringstream ss(line.substr(a + 4, b - a - 4));
    std::string t;
    while (std::getline(ss, t, ',')) {
        if (t.find_first_not_of(' ') == std::string::npos) continue;
        v.push_back(static_cast<uint32_t>(std::stoul(t)));
    }
    return v;
}
uint32_t RankOf(const std::string& line) {
    const size_t a = line.find("rankL1=");
    return static_cast<uint32_t>(std::stoul(line.substr(a + 7)));
}
uint32_t Sum(const std::vector<uint32_t>& v) {
    uint32_t s = 0;
    for (uint32_t x : v) s += x;
    return s;
}

// The reader columns of a record against one ladic2.py row; returns "" on agreement, else the first difference.
std::string CompareLadic2(const campaign::OrbifoldCell& c, const std::string& line) {
    if (c.rank_lambda != RankOf(line)) return "rankL1 " + str(c.rank_lambda) + " vs " + str(RankOf(line));
    if (c.torsion1 != TorsionOf(line)) return "T1";
    const std::vector<uint32_t> ps = c.ell == 2 ? std::vector<uint32_t>{3, 5, 7} : std::vector<uint32_t>{2, 5, 7};
    for (uint32_t p : ps) {
        const Parsed a = ParseDivs(c.ed_tp.count(p) ? c.ed_tp.at(p) : "?"), b = ParseLadic2(Field(line, "p" + str(p)));
        if (a.pos != b.pos || a.nonfinite != b.nonfinite || a.units != 0) return "p" + str(p);
        const Parsed e = ParseDivs(c.ed_eis.count(p) ? c.ed_eis.at(p) : "?"), f = ParseLadic2(Field(line, "Eis" + str(p)));
        if (e.pos != f.pos || e.nonfinite != f.nonfinite || e.units != 0) return "Eis" + str(p);
    }
    const Parsed t = ParseDivs(c.ed_tell), u = ParseLadic2(Field(line, "Tl"));
    if (t.pos != u.pos || t.nonfinite != u.nonfinite) return "Tl";
    return "";
}

}  // namespace

uint32_t OrbifoldCampaignL23Validation() {
    g_checks = g_fail = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "OrbifoldCampaignL23Validation: the orbifold section of the campaign (orbifold_cell/v2, slot-22 L23-1, S24 L23-H1)" << std::endl;

    // ---------------- (A) spec surface -------------------------------------------------------------------------
    {
        const std::string base = "n_sym = 16\nell = 5\np = 2\nr = 1\n";
        const campaign::CampaignSpec a = Spec(base), b = Spec(base + "orbifold = off\n");
        const std::vector<campaign::Job> ja = campaign::flatten(campaign::plan_campaign(a)), jb = campaign::flatten(campaign::plan_campaign(b));
        check(!a.orbifold.enabled && !b.orbifold.enabled, "(A) orbifold disabled without the key and under `orbifold = off`");
        check(ja.size() == 1 && jb.size() == 1 && ja[0].id() == "n16_ell5_p2_r1" && jb[0].id() == ja[0].id() &&
                  ja[0].shape().to_string() == jb[0].shape().to_string(),
              "(A) a spec without `orbifold` plans the unchanged job id n16_ell5_p2_r1 and shape signature");
        const std::string ok = "orbifold = c_i\nn_sym = 0..10\nell = 2,3\n";
        check(ParseOutcome(ok) == 0, "(A) an orbifold spec without p / r parses");
        std::string m;
        check(ParseOutcome("orbifold = c_i\nn_sym = 4\nell = 5\n", &m) == 2 && m.find("not_implemented_in_this_build") == 0,
              "(A) ell = 5 refused (not_implemented_in_this_build): " + m);
        check(ParseOutcome(ok + "p = 2\n", &m) == 1, "(A) a `p` axis refused: " + m);
        check(ParseOutcome(ok + "ext = true\n", &m) == 1, "(A) ext = true refused: " + m);
        check(ParseOutcome(ok + "coefficient_ring = Z/ell^2\n", &m) == 2, "(A) a non-default coefficient_ring refused: " + m);
        check(ParseOutcome(ok + "jobs = 2\n", &m) == 1, "(A) jobs = 2 refused: " + m);
        check(ParseOutcome("orbifold = c_i\nn_sym = 5000\nell = 3\n", &m) == 2, "(A) a cell beyond the reader's range refused: " + m);
        check(ParseOutcome("orbifold = mp\nn_sym = 4\nell = 2\n", &m) == 1, "(A) an unknown model refused: " + m);
        check(ParseOutcome("orbifold = c_i\nn_sym = 4\nell = 2\norbifold_primes = 2,4\n", &m) == 1, "(A) a non-prime in orbifold_primes refused");
        bool threw = false;
        try {
            (void)campaign::RunCampaignSpec(Spec(ok), false);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "(A) RunCampaignSpec refuses an orbifold spec (never runs its Hecke grid)");
    }

    // ---------------- (B) identity ---------------------------------------------------------------------------
    {
        const std::string ok = "orbifold = c_i\nn_sym = 10\nell = 2\n";
        const campaign::CampaignSpec d = Spec(ok), pr = Spec(ok + "orbifold_primes = 2,3,5\n"), lo = Spec(ok + "locality = budget=32\n"),
                                     mt = Spec(ok + "orbifold_m_target = 60\n");
        check(campaign::OrbifoldFingerprint(d) == 0 && campaign::OrbifoldCellId(d, 2, 10) == "orb_ell2_n10", "(B) default: fingerprint 0, id orb_ell2_n10");
        const std::string ip = campaign::OrbifoldCellId(pr, 2, 10), il = campaign::OrbifoldCellId(lo, 2, 10), im = campaign::OrbifoldCellId(mt, 2, 10);
        check(ip.size() == 12 + 18 && ip.rfind("orb_ell2_n10_s", 0) == 0 && il.rfind("orb_ell2_n10_s", 0) == 0 && im.rfind("orb_ell2_n10_s", 0) == 0 &&
                  ip != il && il != im && ip != im,
              "(B) orbifold_primes / a mathematical setting / orbifold_m_target: three distinct _s<16 hex> ids (" + ip + ", " + il + ", " + im + ")");
        const campaign::OrbifoldCell c0 = campaign::ComputeOrbifoldCell(d, 2, 10);
        const campaign::OrbifoldCell cl = campaign::ComputeOrbifoldCell(lo, 2, 10);
        const std::string b0 = campaign::OrbifoldCellBody(d, c0), bl = campaign::OrbifoldCellBody(lo, cl);
        check(c0.status == "ok" && cl.status == "ok" && c0.h[1] == cl.h[1] && c0.ed_tell == cl.ed_tell,
              "(B) the cell's values do not depend on a setting the cell does not consult");
        check(engine::FnvBytes(engine::kFnvOffset64, b0) != engine::FnvBytes(engine::kFnvOffset64, bl) &&
                  bl.find("\"math_fingerprint\":\"0000000000000000\"") == std::string::npos,
              "(B) a non-default mathematical setting changes the record hash input (settings block)");
    }

    // ---------------- (C) the end-to-end campaign --------------------------------------------------------------
    const std::string root = TempDir("e2e");
    uint32_t cells = 0, l2 = 0, l3 = 0, gates = 0, conj = 0, lad_ok = 0, lad_n = 0, sum_ok = 0, sum_n = 0, odd_ok = 0, odd_n = 0;
    uint32_t amal_n = 0, amal_gates = 0, ecup_n = 0, src_bad = 0;
    {
        const std::string text = "name = l231_e2e\norbifold = c_i\nn_sym = 0..60\nell = 2\n---\nn_sym = 0..60\nell = 3\n";
        campaign::CampaignSpec spec = Spec(text, root + "/run");
        const campaign::OrbifoldCampaignResult res = campaign::RunOrbifoldCampaign(spec, /*verbose=*/false);
        check(res.plan.size() == 122 && res.ran == 122 && res.failed == 0 && res.fallback == 0 && res.refused == 0 && !res.stopped,
              "(C) 122 cells ran: failed " + str(res.failed) + ", fallback " + str(res.fallback) + ", refused " + str(res.refused) +
                  (res.stopped ? ", STOPPED: " + res.stop_reason : std::string()));
        for (const campaign::OrbifoldCell& c : res.cells) {
            ++cells;
            (c.ell == 2 ? l2 : l3)++;
            gates += static_cast<uint32_t>(c.gates.size());
            conj += c.conj_findings();
            const std::string at = " (ell=" + str(c.ell) + " n=" + str(c.n) + ")";
            check(c.status == "ok", "(C) status ok" + at + ": " + c.status + (c.failures.empty() ? std::string() : " -- " + c.failures.front()));
            for (const std::string& f : c.findings) std::cout << "  (C) G-conj finding" << at << ": " << f << std::endl;
            // R2: the source of every gate
            for (const campaign::OrbifoldGate& g : c.gates) {
                const std::string& k = g.source_class;
                const bool known = k == "C" || k == "R" || k == "LIT" || k == "UCT" || k == "cross-model" || k == "internal" || k == "CONJ";
                if (!known || g.source_ref.empty() || (g.kind == "G-conj") != (k == "CONJ")) {
                    ++src_bad;
                    std::cout << "  [src] " << g.name << at << ": class '" << k << "' ref '" << g.source_ref << "'" << std::endl;
                }
                if (g.name.rfind("amalgam_", 0) == 0) ++amal_gates;
                if (g.name == "e_cup_h0_h2_injective") ++ecup_n;
            }
            if (c.amalgam_ran) ++amal_n;
            check(c.amalgam_ran == (c.n > 0 && (c.ell == 2 || c.n % 2 == 0)), "(C) R3: the amalgam oracle ran exactly where specified" + at);
            const std::string path = root + "/run/cells/" + c.id + ".json";
            const std::string file = ReadFile(path);
            const std::string body = campaign::OrbifoldCellBody(spec, c);
            char hx[24];
            std::snprintf(hx, sizeof(hx), "%016llx", static_cast<unsigned long long>(engine::FnvBytes(engine::kFnvOffset64, body)));
            check(file.find("\"orbifold_cell_complete\":true") != std::string::npos && file.find(std::string("\"record_fnv1a64\":\"") + hx + "\"") != std::string::npos &&
                      file.compare(1, body.size(), body) == 0,
                  "(C) the record file is complete and its record_fnv1a64 is the FNV of its body" + at);
            check(body.find("\"rung\"") == std::string::npos && body.find("\"reruns\"") == std::string::npos &&
                      body.find("\"e_A\"") == std::string::npos && body.find("binary_sha256") == std::string::npos &&
                      file.find("\"reader_path\":{\"rung\":") != std::string::npos && file.find("\"binary_sha256\":\"") != std::string::npos &&
                      file.find("\"commit\":\"") != std::string::npos,
                  "(C) R4 / R6: the precision path and the provenance are outside the hashed body, present in the file" + at);
            const bool even = c.n % 2 == 0;
            if (even && c.n >= 2) {
                for (const Ladic2Row& r : kLadic2)
                    if (r.ell == c.ell && r.n == c.n) {
                        ++lad_n;
                        const std::string d = CompareLadic2(c, r.line);
                        lad_ok += check(d.empty(), "(C) record = ladic2.py row" + at + ": " + d) ? 1 : 0;
                    }
                for (const SumRuleRow& r : kSumRules)
                    if (r.ell == c.ell && r.n == c.n) {
                        ++sum_n;
                        bool ok = Sum(ParseDivs(c.ed_tell).pos) == static_cast<uint32_t>(2 * r.vTl);
                        const uint32_t ps[4] = {2, 3, 5, 7};
                        for (int i = 0; i < 4; ++i) {
                            if (ps[i] == c.ell) continue;
                            ok = ok && Sum(ParseDivs(c.ed_tp.at(ps[i])).pos) == static_cast<uint32_t>(2 * r.vP[i] + r.vN[i]);
                            ok = ok && Sum(ParseDivs(c.ed_eis.at(ps[i])).pos) == static_cast<uint32_t>(2 * r.vE[i]);
                        }
                        sum_ok += check(ok, "(C) the level-1 modular-forms sum rules" + at) ? 1 : 0;
                    }
            }
            if (!even && c.ell == 2) {
                for (const OddnOracleRow& r : kOddn)
                    if (r.n == c.n) {
                        ++odd_n;
                        const bool ok = c.h[0] == r.v[0] && c.h[1] == r.v[2] && c.sub_dim[1] == r.v[4] && c.t2 == r.v[5] && c.t_ell_rank[1] == r.v[14] &&
                                        (c.layers[1].jordan.rank_one_idempotent() ? 1u : 0u) == r.v[17] && c.layers[4].jordan.rank == r.v[18];
                        odd_ok += check(ok, "(C) record = oddn_l2.py row" + at) ? 1 : 0;
                    }
            }
        }
        uint32_t ecup_want = 0;
        for (const campaign::OrbifoldCell& c : res.cells) ecup_want += (c.ell == 2 && c.h[0] > 0);
        check(src_bad == 0, "(C) R2: every gate names a source class and reference (G-conj <=> CONJ): " + str(src_bad) + " bad");
        check(amal_n == 90 && amal_gates == 5 * 90, "(C) R3: the amalgam oracle on 90 cells, 450 cross-model gates: " + str(amal_n) + " cells, " + str(amal_gates) + " gates");
        check(ecup_n == ecup_want && ecup_want > 0, "(C) R5: e_cup_h0_h2_injective on every ell = 2 cell with h0 > 0: " + str(ecup_n) + "/" + str(ecup_want));
        if (!res.cells.empty()) {
            // R4: the hashed body does not move with the reader's working-precision path
            campaign::OrbifoldCell x = res.cells.back();
            const std::string b1 = campaign::OrbifoldCellBody(spec, x);
            x.rung = "R9(other)";
            x.K += 7;
            x.m += 5;
            x.e_A += 1;
            x.e_1 += 2;
            x.reruns += 3;
            check(campaign::OrbifoldCellBody(spec, x) == b1 && campaign::OrbifoldCellJson(spec, x) != campaign::OrbifoldCellJson(spec, res.cells.back()),
                  "(C) R4: a different precision path (rung, K, m, e_A, e_1, reruns) leaves the hashed body unchanged, the file not");
        }
        check(campaign::OrbifoldGateTriage("R").find("review's proof") != std::string::npos &&
                  campaign::OrbifoldGateTriage("C").find("misquoted") != std::string::npos &&
                  campaign::OrbifoldGateTriage("cross-model").find("bug") != std::string::npos,
              "(C) R2: the stop message's triage reading per source class");
        check(lad_n == 60 && sum_n == 60 && odd_n == 30, "(C) oracle coverage: ladic2 " + str(lad_n) + "/60, sum rules " + str(sum_n) + "/60, oddn " + str(odd_n) + "/30");
        std::cout << "  (C) cells=" << cells << " (ell=2: " << l2 << ", ell=3: " << l3 << ") gates=" << gates << " conj_findings=" << conj
                  << " ladic2=" << lad_ok << "/" << lad_n << " sumrules=" << sum_ok << "/" << sum_n << " oddn=" << odd_ok << "/" << odd_n << std::endl;
        if (!res.cells.empty()) {
            const campaign::OrbifoldCell* show = nullptr;
            for (const auto& c : res.cells) if (c.ell == 3 && c.n == 22) show = &c;
            if (show)
                std::cout << "  (C) e.g. ell=3 n=22: h=(" << show->h[0] << "," << show->h[1] << "," << show->h[2] << "," << show->h[3] << ") T1="
                          << show->torsion1.size() << " ed_tell=" << show->ed_tell << " +:" << show->ed_tell_plus << " -:" << show->ed_tell_minus
                          << std::endl;
        }

        // ---------------- (D) resume and determinism ----------------
        campaign::CampaignSpec again = spec;
        again.resume = true;
        const campaign::OrbifoldCampaignResult r2 = campaign::RunOrbifoldCampaign(again, false);
        check(r2.ran == 0 && r2.skipped == 122 && r2.resumed_not_v2 == 0,
              "(D) resume skips every complete cell: ran " + str(r2.ran) + ", skipped " + str(r2.skipped) + ", not v2 " + str(r2.resumed_not_v2));
        for (const auto& [e, n] : std::vector<std::pair<uint32_t, uint32_t>>{{2, 22}, {3, 34}}) {
            const campaign::OrbifoldCell x = campaign::ComputeOrbifoldCell(spec, e, n);
            std::string first_raw;
            for (const auto& c : res.cells) if (c.ell == e && c.n == n) first_raw = campaign::OrbifoldCellBody(spec, c);
            const std::string again_raw = campaign::OrbifoldCellBody(spec, x);
            // (S24 L23-H1, KNOWN_ISSUES Sec. 7 / H-1, closed) orbifold_cell/v2: the oracle's wall time is not in the hashed body, so the
            // two bodies are compared RAW, no mask.  The gate detail ends "-> resolved" (once per body; both cells have a lattice).
            const std::string first = first_raw, again_body = again_raw;
            auto gate_sites = [](const std::string& body) {
                uint32_t k = 0;
                const std::string key = "\"name\":\"sumrule_oracle_resolved\"";
                for (size_t q = body.find(key); q != std::string::npos; q = body.find(key, q + 1)) ++k;
                return k;
            };
            const bool timeless = first_raw.find("-> resolved\"}") != std::string::npos && first_raw.find(" s)\"") == std::string::npos &&
                                  first_raw.find("\"schema\":\"orbifold_cell/v2\"") != std::string::npos;
            check(gate_sites(first_raw) == 1 && gate_sites(again_raw) == 1 && timeless && !x.sumrule_oracle_s.empty(),
                  "(D) v2 body at (" + str(e) + ", " + str(n) + "): one oracle gate, detail '-> resolved' with no wall time, time kept apart ('" + x.sumrule_oracle_s + "')");
            // (S23 G2-A/B merge; KNOWN_ISSUES) on a mismatch, name the first differing field of the two bodies
            std::string where;
            if (!first.empty() && again_body != first) {
                size_t k = 0;
                while (k < first.size() && k < again_body.size() && first[k] == again_body[k]) ++k;
                const size_t key = first.rfind("\":", k);
                const size_t q0 = key == std::string::npos ? std::string::npos : first.rfind('"', key == 0 ? 0 : key - 1);
                const size_t from = q0 == std::string::npos ? (k > 60 ? k - 60 : 0) : q0;
                where = " -- first difference at byte " + str(static_cast<uint32_t>(k)) + ": run `" + first.substr(from, k - from + 60) +
                        "` vs recompute `" + again_body.substr(from, k - from + 60) + "`";
            }
            check(!first.empty() && again_body == first, "(D) deterministic body at (" + str(e) + ", " + str(n) + ")" + where);
        }
        {   // (S24 L23-H1) the wall time lives outside the hash: a different time moves the JSON record, not the body or record_fnv1a64
            campaign::OrbifoldCell x = campaign::ComputeOrbifoldCell(spec, 3, 34);
            const std::string j1 = campaign::OrbifoldCellJson(spec, x), b1 = campaign::OrbifoldCellBody(spec, x);
            x.sumrule_oracle_s = "98.765";
            const std::string j2 = campaign::OrbifoldCellJson(spec, x), b2 = campaign::OrbifoldCellBody(spec, x);
            auto fnv_of = [](const std::string& j) {
                const std::string key = "\"record_fnv1a64\":\"";
                const size_t i = j.find(key);
                return i == std::string::npos ? std::string() : j.substr(i + key.size(), 16);
            };
            check(b1 == b2 && j1 != j2 && fnv_of(j1) == fnv_of(j2) && fnv_of(j1).size() == 16 && j2.find("\"sumrule_oracle\":98.765") != std::string::npos &&
                      j2.find("\"orbifold_cell_complete\":true}") != std::string::npos,
                  "(D) the oracle's wall time is in timings_s.sumrule_oracle only: body and record_fnv1a64 do not move with it");
        }
        // R1 ([P3] R7 across restarts): plant a complete `fallback` record at n = 2 and a complete `failed` one at n = 3
        // (the shape of docs/implementation/slot22/fl231/resume_probe.sh), resume over n = 0..6.
        {
            const std::string rd = root + "/resume_failed";
            std::filesystem::create_directories(rd + "/cells");
            std::ofstream(rd + "/cells/orb_ell2_n2.json")
                << "{\"schema\":\"orbifold_cell/v1\",\"cell_id\":\"orb_ell2_n2\",\"ell\":2,\"n\":2,\"status\":\"fallback\",\"gates\":[],"
                   "\"failures\":[],\"orbifold_cell_complete\":true}\n";
            std::ofstream(rd + "/cells/orb_ell2_n3.json")
                << "{\"schema\":\"orbifold_cell/v1\",\"cell_id\":\"orb_ell2_n3\",\"ell\":2,\"n\":3,\"status\":\"failed\",\"gates\":[{\"name\":"
                   "\"fake_G\",\"kind\":\"G\",\"ok\":false,\"detail\":\"planted failure\"}],\"gate_failures\":1,\"failures\":[\"fake_G: planted "
                   "failure\"],\"record_fnv1a64\":\"0\",\"log\":[],\"orbifold_cell_complete\":true}\n";
            campaign::CampaignSpec rs = Spec("name = l231_resume_failed\norbifold = c_i\nresume = true\nn_sym = 0..6\nell = 2\n", rd);
            const campaign::OrbifoldCampaignResult r3 = campaign::RunOrbifoldCampaign(rs, false);
            const std::string ix = ReadFile(rd + "/index.json"), sm = ReadFile(rd + "/summary.tsv");
            // (L23-H1 audit F2) the planted fallback record is orbifold_cell/v1: skipped as before, and counted as a mixed-format cell
            check(r3.stopped && r3.failed == 1 && r3.ran == 2 && r3.skipped == 1 && r3.resumed_not_v2 == 1 && r3.stop_reason.find("planted failure") != std::string::npos &&
                      r3.stop_reason.find("resume on a failed cell") != std::string::npos,
                  "(D) R1: resume over a planted failed record stops there (stopped " + str(r3.stopped) + ", failed " + str(r3.failed) + ", ran " +
                      str(r3.ran) + ", skipped " + str(r3.skipped) + ", not v2 " + str(r3.resumed_not_v2) + "): " + r3.stop_reason);
            check(!std::filesystem::exists(rd + "/cells/orb_ell2_n4.json") && ix.find("\"id\":\"orb_ell2_n3\",\"ell\":2,\"n\":3,\"status\":\"failed\"") != std::string::npos &&
                      ix.find("\"stopped\":true") != std::string::npos && sm.find("orb_ell2_n3\t2\t3\t5\tc_i\t-\tfailed") != std::string::npos &&
                      sm.find("orb_ell2_n2\t2\t2\t4\tc_i\t-\tresumed") != std::string::npos,
                  "(D) R1: nothing ran past the failed cell; index.json / summary.tsv carry `failed`, the fallback record `resumed`");
            check(campaign::RunOrbifoldCampaignCli(rs, false, false) == 2, "(D) R1: the CLI exits 2 on a resume over a failed record");
        }
    }

    // ---------------- (E) negative controls ----------------
    {
        const campaign::CampaignSpec s = Spec("orbifold = c_i\nn_sym = 10\nell = 2\norbifold_m_target = 3\n");
        const campaign::OrbifoldCell c = campaign::ComputeOrbifoldCell(s, 2, 10);
        check(c.status == "fallback" && c.fallback_divisors > 0 && c.id.rfind("orb_ell2_n10_s", 0) == 0,
              "(E) orbifold_m_target = 3 at (2, 10): a fallback cell (" + c.status + ", " + str(c.fallback_divisors) + " uncertified divisors), id " + c.id);
        const campaign::OrbifoldCell g = campaign::ComputeOrbifoldCell(Spec("orbifold = c_i\nn_sym = 10\nell = 2\n"), 2, 10);
        std::string tampered;
        for (const Ladic2Row& r : kLadic2)
            if (r.ell == 2 && r.n == 10) tampered = r.line;
        const size_t at = tampered.find("Eis3:8,8");
        if (at != std::string::npos) tampered.replace(at, 8, "Eis3:8,9");
        check(at != std::string::npos && !CompareLadic2(g, tampered).empty(), "(E) a tampered ladic2.py row is caught");
    }
    std::error_code ec;
    std::filesystem::remove_all(root, ec);

    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "ORBIFOLD_L23 cells=" << cells << " ell2=" << l2 << " ell3=" << l3 << " gates=" << gates << " conj_findings=" << conj
              << " ladic2=" << lad_ok << "/" << lad_n << " sumrules=" << sum_ok << "/" << sum_n << " oddn=" << odd_ok << "/" << odd_n
              << " amalgam=" << amal_n << "/" << amal_gates << " checks=" << g_checks << " failing=" << g_fail << " s=" << s << " verdict=" << (g_fail ? "FAIL" : "OK") << std::endl;
    return g_fail;
}
