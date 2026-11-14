// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZEisUpCharValidation -- slot-27 EISUPb (HZ-EISUP (b)); see the header and HZEISUPb_design.md v0.4 Sec. 7.

#include "HeckeZEisUpCharValidation.h"

#include "HeckeZEisUp.h"
#include "HeckeZEisUpChar.h"
#include "HeckeZRational.h"
#include "HeckeZRecord.h"
#include "HeckeZUTag.h"
#include "HeckeZUp.h"
#include "Descent/DescentRecord.h"
#include "../HeckeExt/HeckeZRecordSource.h"
#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace hzr = heckez::record;
namespace eu = heckez::eisup;

std::string SourcePath(const std::string& rel) {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/" + rel;
#else
    return rel;
#endif
}

std::string Hex64(uint64_t v) {
    std::ostringstream o;
    o << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
    return o.str();
}

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

struct Cell {
    uint32_t N, n;
};
std::string Key(uint32_t N, uint32_t n) { return "(" + std::to_string(N) + "," + std::to_string(n) + ")"; }

bool Caught(const std::function<void()>& fn, std::string* what) {
    try {
        fn();
    } catch (const heckez::WordPrimeCapRefusal& e) {
        *what = std::string("refused: ") + e.what();
        return true;
    } catch (const std::exception& e) {
        *what = e.what();
        return true;
    }
    *what = "not caught";
    return false;
}

// the inputs of the step for a cell exactly as the record builds them (P by P_rule, then the full step, its retained runs kept)
struct CellInput {
    heckez::RationalResult QF;
    std::vector<heckez::PrimeRun> runs;
    eu::EisUpCharInput in;
};
CellInput MakeInput(uint32_t N, uint32_t n, uint32_t threads) {
    CellInput C;
    heckez::RationalSettings rs;
    rs.threads = threads;
    std::vector<uint32_t> P = integral_descent::record::DefaultRecordPrimes(N);
    for (;;) {
        const heckez::RationalResult Q = heckez::RationalStep(N, n, P, rs);
        if (Q.c1_status == "certified" || P.size() >= 6) break;
        uint32_t p = P.back() + 1;
        for (;; ++p) {
            bool prime = p > 1;
            for (uint32_t d = 2; d * d <= p; ++d) prime = prime && p % d != 0;
            if (prime && N % p != 0) break;
        }
        P.push_back(p);
    }
    C.QF = heckez::RationalStepFull(N, n, P, heckez::up::PrimesDividing(N), rs, nullptr, &C.runs);
    ENGINE_CHECK_MSG(C.QF.c1_status == "certified" && C.QF.theta_index >= 0, "HeckeZEisUpCharValidation: the full (C1) is not certified");
    const heckez::ThetaEntry& T = C.QF.thetas[static_cast<size_t>(C.QF.theta_index)];
    C.in.N = N;
    C.in.n = n;
    C.in.ops = C.QF.P;
    C.in.nU = C.QF.nU;
    C.in.theta = T.coeffs;
    C.in.f = T.f;
    C.in.e_k = C.QF.e_k;
    C.in.d = C.QF.d;
    C.in.b1 = C.QF.b1;
    C.in.rs = rs;
    C.in.runs = nullptr;
    return C;
}
const eu::EisUpCharInput& Bound(CellInput& C) {
    C.in.runs = &C.runs;
    return C.in;
}

// arm (a): the digests of EisUpCharResult::text() (first computed by this package, 2026-11-13; cross-checked by (b), (c), (d))
const std::map<std::string, std::string>& DigestPins() {
    static const std::map<std::string, std::string> m = {
        {"(9,0)", "0xa474f5e28f4bdce5"}, {"(18,0)", "0x99415ef4e97e1bbc"}, {"(25,0)", "0xc9cacf2ea875cec1"}, {"(49,0)", "0x69ba2ef1b9eff601"},
        // arm (a'), step level under lex_then_greedy
        {"(9,2)", "0x36cfdff9acae7df6"}, {"(49,2)", "0x9afe658d2cc3307e"}, {"(121,0)", "0x8d3986cf6dd7dc0e"}, {"(225,0)", "0x2539f6c7f15d8b9a"},
    };
    return m;
}
// arm (e): record_fnv1a64 of the four (b) records (settings fingerprint unchanged)
const std::map<std::string, std::string>& RecordPins() {
    static const std::map<std::string, std::string> m = {
        {"(9,0)", "0x0783cd2151bc2605"}, {"(18,0)", "0xb1f4c3b3e57bd255"}, {"(25,0)", "0x5611bfec0f352eac"}, {"(49,0)", "0x27ac24bd701543b3"},
    };
    return m;
}
// arm (j): (a)'s record pins (HeckeZEisUpValidation arm (d)); the branch is not taken there
const std::map<std::string, std::string>& APins() {
    static const std::map<std::string, std::string> m = {
        {"(11,0)", "0x3409215438a036a6"}, {"(2,4)", "0x940f29550a7aa409"}, {"(22,0)", "0x8fd36b1a8b3405c9"}, {"(5,4)", "0x671876317a45370d"},
    };
    return m;
}
// arm (c): v_ell(disc f_theta) predicted at design time (HZEISUPb_design.md Sec. 3, eisupb_predict.py)
const std::map<std::string, std::map<uint64_t, uint32_t>>& DiscPins() {
    static const std::map<std::string, std::map<uint64_t, uint32_t>> m = {
        {"(9,0)", {{2, 4}, {3, 2}, {13, 2}}},
        {"(18,0)", {{2, 28}, {3, 16}, {5, 6}}},
        {"(25,0)", {{2, 20}, {3, 6}, {5, 8}, {13, 2}, {17, 2}}},
        {"(49,0)", {{2, 36}, {3, 26}, {7, 16}, {13, 2}, {31, 4}}},
    };
    return m;
}

// edition 3 (docs/integral_census/results/hecke_ed3): key -> record_fnv1a64, and the 226 cell texts
std::map<std::string, std::string> Ed3Hashes() {
    std::map<std::string, std::string> m;
    std::ifstream in(SourcePath("docs/integral_census/results/hecke_ed3/rows.tsv"));
    std::string line;
    int col = -1;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string t;
        while (std::getline(ss, t, '\t')) f.push_back(t);
        if (col < 0) {
            for (size_t i = 0; i < f.size(); ++i)
                if (f[i] == "record_fnv1a64") col = static_cast<int>(i);
            continue;
        }
        if (static_cast<size_t>(col) < f.size()) m[f[0]] = f[static_cast<size_t>(col)];
    }
    return m;
}
std::vector<std::string> Ed3CellTexts() {
    std::vector<std::string> out;
    const std::string cmd = "tar -xzOf '" + SourcePath("docs/integral_census/results/hecke_ed3/records_bundle.tar.gz") + "' 2>/dev/null";
    FILE* f = ::popen(cmd.c_str(), "r");
    if (!f) return out;
    std::string cur;
    char buf[1 << 16];
    size_t r;
    while ((r = std::fread(buf, 1, sizeof buf, f)) > 0) cur.append(buf, r);
    ::pclose(f);
    std::stringstream ss(cur);
    std::string line;
    while (std::getline(ss, line))
        if (!line.empty() && line[0] == '{') out.push_back(line);
    return out;
}
std::string IdOfCell(const std::string& text) {
    const std::string k = "\"id\":\"";
    const size_t a = text.find(k);
    if (a == std::string::npos) return std::string();
    const size_t b = text.find('"', a + k.size());
    return text.substr(a + k.size(), b - a - k.size());
}
std::string Summary(const eu::EisUpCharResult& R) {
    std::ostringstream o;
    for (size_t j = 0; j < R.comps.size(); ++j) {
        const auto& c = R.comps[j];
        o << (j ? ";" : "") << "d" << c.degree << "/c" << c.psi_cond << "/o" << c.psi_order;
        if (c.degree == 1) {
            o << "(";
            for (size_t i = 0; i < c.U.size(); ++i) o << (i ? "," : "") << c.U[i];
            o << ")";
        }
    }
    return o.str();
}
// the closed-form component strings of a result (degree, cond, order, type), sorted
std::vector<std::string> CompStrings(const eu::EisUpCharResult& R) {
    std::vector<std::string> v;
    for (const auto& c : R.comps) {
        std::ostringstream o;
        o << "deg=" << c.degree << " cond=" << c.psi_cond << " ord=" << c.psi_order << " y=";
        for (int32_t t : c.y) o << (t == 0 ? "0" : t == 1 ? "a" : "b");
        v.push_back(o.str());
    }
    std::sort(v.begin(), v.end());
    return v;
}

}  // namespace

uint32_t HeckeZEisUpCharValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeZEisUpCharValidation: the Eisenstein U_p systems with characters psi != 1 (slot-27 EISUPb; [C] 1.28.0-pre.1 "
                 "prop:heckeZeisupchar, item 8 (d')(ii); HZEISUPb_design.md v0.4)\n";
    Tally T;
    const hecke::ScopedCoefficientRing ring(0);   // coefficient_ring = Z
    const std::vector<Cell> cells = {{9, 0}, {18, 0}, {25, 0}, {49, 0}};
    const std::string kContract = "gl2_contract 1.28.0-pre.1, item 8 (d')";
    T.check(std::string(heckez::up::kFullContract) == kContract, "(h) the binary's G10.contract value is " + kContract);

    // ---- (a) certificate, (b) second route, (c) disc, (d) HZ-UTAG oracle ---------------------------------------------------------
    uint32_t comps_total = 0, deg2 = 0, second_ok = 0, disc_ok = 0, disc_all = 0, utag_ok = 0, utag_all = 0;
    std::map<std::string, CellInput> inputs;
    std::map<std::string, eu::EisUpCharResult> results;
    for (const Cell& c : cells) {
        const std::string key = Key(c.N, c.n);
        T.check(eu::InScopeChar(c.N, c.n) && !eu::InScope(c.N, c.n), "(a) " + key + " is in (b)'s scope (e^an_k(N) != 1)");
        std::string why;
        eu::EisUpCharResult R;
        const bool threw = Caught(
            [&] {
                inputs.emplace(key, MakeInput(c.N, c.n, 4));
                R = eu::EisUpCharStep(Bound(inputs.at(key)));
            },
            &why);
        if (!T.check(!threw, "(a) " + key + ": EisUpCharStep (C-B0 .. C-B8) ran: " + why)) continue;
        results[key] = R;
        comps_total += static_cast<uint32_t>(R.comps.size());
        int64_t sdeg = 0;
        for (const auto& x : R.comps) {
            sdeg += x.degree;
            deg2 += x.degree >= 2 ? 1 : 0;
        }
        T.check(sdeg == R.e_k, "(a) " + key + ": sum of the degrees = e_k");
        const std::string dg = Hex64(R.digest());
        const auto pin = DigestPins().find(key);
        T.check(pin != DigestPins().end() && pin->second == dg, "(a) " + key + " digest " + dg + " = pin " + (pin == DigestPins().end() ? "-" : pin->second));
        std::cout << "  EISUPB " << key << " e_k=" << R.e_k << " d=" << R.d << " s_N=" << R.s_N << " o_N=" << R.o_N << " theta=(";
        for (size_t i = 0; i < R.theta.size(); ++i) std::cout << (i ? " " : "") << R.theta[i];
        std::cout << ") comps=" << Summary(R) << " primes=" << R.primes_used << " sep=" << R.sep_used << " skipped=" << R.skipped.size()
                  << " q*=" << R.id_prime << " S=" << R.S.size() << " extra=" << R.id_prime_extra << " reused=" << R.reused << " computed=" << R.computed
                  << " digest=" << dg << "\n";
        // (b) the closed form: the components, their count = c(N) - [k = 2] = e_k
        const eu::ClosedFormSummary cf = eu::ClosedForm(c.N, c.n + 2);
        const bool sec = cf.comps == CompStrings(R) && cf.systems == R.e_k && cf.cusps_minus == R.e_k;
        second_ok += sec ? 1 : 0;
        T.check(sec, "(b) " + key + ": the components equal the closed form, sum deg = e_k = c(N) - [k = 2]");
        // (c) v_ell(disc f_theta) = the design-time prediction = the record's v_disc (the latter in arm (e))
        {
            const std::vector<std::string>& f = inputs.at(key).in.f;
            const auto pin2 = DiscPins().find(key);
            if (pin2 != DiscPins().end())
                for (const auto& [ell, v] : pin2->second) {
                    ++disc_all;
                    const uint32_t got = eu::DiscValuation(f, ell);
                    disc_ok += got == v ? 1 : 0;
                    T.check(got == v, "(c) " + key + ": v_" + std::to_string(ell) + "(disc f_theta) = " + std::to_string(got) + " = " + std::to_string(v));
                }
        }
        // (d) every g_{U_p,j} divides HZ-UTAG's chi_Q(U_p) exactly over Z
        const std::vector<uint32_t> Up = R.Uprimes();
        const std::vector<uint32_t> Tp(inputs.at(key).QF.P.begin(), inputs.at(key).QF.P.end() - R.nU);
        for (size_t i = 0; i < Up.size(); ++i) {
            const heckez::UTagResult tag = heckez::UTagStep(c.N, c.n, Up[i], Tp, inputs.at(key).in.rs);
            for (const auto& x : R.comps) {
                ++utag_all;
                std::vector<std::string> g = x.Ucharpoly[i];
                std::string w;
                const bool div = !Caught([&] { eu::PolyDivExact(tag.chi, g); }, &w);
                utag_ok += div ? 1 : 0;
                T.check(div, "(d) " + key + ": g_{U_" + std::to_string(Up[i]) + ",j} divides chi_Q(U_p): " + w);
            }
        }
    }

    // ---- (a') step level, a non-default theta (design Sec. 7 "step level, explicit theta"): the full step under TS's lex_then_greedy
    // order, never a record (no settings question).  Every cell is pinned, so (C1) not reached is a FAILURE (audit M1: (121,0) is the only
    // degree-4 / o_N = 10 cell, (225,0) the only non-cyclic (Z/s_N)^x; a silently dropped cell would be the vacuous-arm pattern); the
    // EFFECTIVE line gates step_cells = 4
    const std::vector<Cell> step_cells = {{9, 2}, {49, 2}, {121, 0}, {225, 0}};
    uint32_t step_ok = 0, step_all = 0;
    std::string step_list;
    std::map<std::string, eu::EisUpCharResult> step_results;
    std::map<std::string, CellInput> step_inputs;   // kept for the tampers of arm (g) at (225,0)
    for (const Cell& c : step_cells) {
        const std::string key = Key(c.N, c.n);
        std::string why;
        bool reached = false;
        eu::EisUpCharResult R;
        const bool threw = Caught(
            [&] {
                CellInput C;
                heckez::RationalSettings rs;
                rs.threads = 4;
                rs.theta_order = heckez::ThetaOrder::LexThenGreedy;
                std::vector<uint32_t> P = integral_descent::record::DefaultRecordPrimes(c.N);
                C.QF = heckez::RationalStepFull(c.N, c.n, P, heckez::up::PrimesDividing(c.N), rs, nullptr, &C.runs);
                if (C.QF.c1_status != "certified" || C.QF.theta_index < 0) return;
                reached = true;
                const heckez::ThetaEntry& Th = C.QF.thetas[static_cast<size_t>(C.QF.theta_index)];
                C.in.N = c.N;
                C.in.n = c.n;
                C.in.ops = C.QF.P;
                C.in.nU = C.QF.nU;
                C.in.theta = Th.coeffs;
                C.in.f = Th.f;
                C.in.e_k = C.QF.e_k;
                C.in.d = C.QF.d;
                C.in.b1 = C.QF.b1;
                C.in.rs = rs;
                R = eu::EisUpCharStep(Bound(C));
                step_inputs[key] = std::move(C);
            },
            &why);
        if (!reached && !threw) {
            T.check(false, "(a') " + key + ": (C1) not reached by lex_then_greedy on P_rule's primes (the cell is pinned: a failure, audit M1)");
            step_list += std::string(step_list.empty() ? "" : ",") + key + ":NOT_REACHED";
            continue;
        }
        ++step_all;
        if (!T.check(!threw, "(a') " + key + ": EisUpCharStep at an explicit theta: " + why)) continue;
        const eu::ClosedFormSummary cf = eu::ClosedForm(c.N, c.n + 2);
        const bool sec = cf.comps == CompStrings(R) && cf.systems == R.e_k && cf.cusps_minus == R.e_k;
        step_ok += sec ? 1 : 0;
        T.check(sec, "(a') " + key + ": the components equal the closed form");
        {
            const auto pin = DigestPins().find(key);
            T.check(pin != DigestPins().end() && pin->second == Hex64(R.digest()), "(a') " + key + " digest " + Hex64(R.digest()) + " pinned");
        }
        step_results[key] = R;
        step_list += std::string(step_list.empty() ? "" : ",") + key + ":" + (sec ? "ok" : "FAIL");
        std::cout << "  STEP " << key << " theta=(";
        for (size_t i = 0; i < R.theta.size(); ++i) std::cout << (i ? " " : "") << R.theta[i];
        std::cout << ") ops=" << R.ops.size() << " comps=" << Summary(R) << " q*=" << R.id_prime << " S=" << R.S.size() << " digest=" << Hex64(R.digest())
                  << "\n";
    }
    std::cout << "EFFECTIVE HeckeZEisUpCharValidation step_cells=" << step_ok << "\n";
    T.check(step_ok == step_cells.size() && step_all == step_cells.size(),
            "(a') EFFECTIVE step_cells=" + std::to_string(step_ok) + " = " + std::to_string(step_cells.size()) + " (every pinned step cell reached and equal)");

    // ---- (e) the record path ----------------------------------------------------------------------------------------------------------
    const hzr::RecordSources src = hzr::DefaultRecordSources();
    hzr::RecordSettings S4;
    S4.threads = 4;
    uint32_t rec_ok = 0, rec_all = 0, meetings = 0, vdisc_ok = 0;
    std::map<std::string, hzr::HeckeStructureRecord> recs;
    for (const Cell& c : cells) {
        ++rec_all;
        const std::string key = Key(c.N, c.n);
        std::string why;
        hzr::HeckeStructureRecord R;
        if (!T.check(!Caught([&] { R = hzr::BuildHeckeStructureRecord(c.N, c.n, "full", S4, src); }, &why), "(e) " + key + " full: built: " + why))
            continue;
        recs[key] = R;
        const hzr::ScanResult sc = hzr::ScanRecord(R);
        for (const auto& pr : sc.problems) std::cout << "  [SCAN] " << key << " " << pr << "\n";
        auto val = [&](const std::string& p) { const hzr::Field* f = R.find(p); return f ? f->value : std::string("<absent>"); };
        bool blocked = false;
        uint32_t decomposed = 0;
        for (const auto& f : R.fields) {
            if (f.value == "blocked(U_p)" && (f.path.compare(0, 13, "G1.components") == 0 || f.path.find(".maximal_ideals") != std::string::npos))
                blocked = true;
            if (f.path.size() > 15 && f.path.compare(f.path.size() - 15, 15, ".maximal_ideals") == 0 && f.value == "certified") ++decomposed;
        }
        // the rows equal the step's (arm (a))
        bool rows = val("G1.components") == "certified" && results.count(key);
        if (rows) {
            const auto& S = results.at(key);
            for (size_t j = 0; j < S.comps.size(); ++j) {
                const std::string cp = "G1.components[" + std::to_string(j) + "]";
                rows = rows && val(cp + ".type") == "eis" && val(cp + ".degree") == std::to_string(S.comps[j].degree) &&
                       val(cp + ".psi_cond") == std::to_string(S.comps[j].psi_cond) && val(cp + ".psi_order") == std::to_string(S.comps[j].psi_order);
            }
        }
        // the Eisenstein--Eisenstein meetings (design Sec. 2.4): some m of a decomposed ell carrying two Eisenstein components (on these
        // records the ell | o_N are 2, 3, not decomposed by the shipped binary; the meetings there are observed on theta below)
        uint32_t meet = 0;
        {
            std::map<std::string, uint32_t> eis_through;
            const size_t ne = results.count(key) ? results.at(key).comps.size() : 0;
            for (const auto& f : R.fields) {
                const size_t a = f.path.rfind(".components_through.");
                if (a == std::string::npos || f.value != "certified") continue;
                const size_t j = std::stoul(f.path.substr(a + 20));
                if (j < ne) ++eis_through[f.path.substr(0, a)];
            }
            for (const auto& [m, cnt] : eis_through) {
                if (cnt >= 2) ++meet;
                if (cnt >= 2) std::cout << "  MEETING " << key << " " << m << " eis_components=" << cnt << "\n";
            }
        }
        meetings += meet;
        // (c) the record's v_disc at the design's primes
        const auto pin2 = DiscPins().find(key);
        bool vd = pin2 != DiscPins().end();
        if (vd)
            for (const auto& [ell, v] : pin2->second) vd = vd && val("G1.S_cand.ell=" + std::to_string(ell) + ".v_disc") == std::to_string(v);
        vdisc_ok += vd ? 1 : 0;
        T.check(vd, "(c) " + key + ": the record's v_disc = the design-time v_ell(disc f_theta)");
        const auto pin = RecordPins().find(key);
        const bool ok = sc.values_outside_lists == 0 && sc.exact_under_bound == 0 && sc.overall_rule_ok && !blocked && rows &&
                        val("G10.contract") == kContract && R.settings_fingerprint == "0x8ffc29730c4f9672";
        T.check(ok, "(e) " + key + " full: scan clean, the rows written, no blocked(U_p) in G1.components / maximal_ideals, G10.contract");
        T.check(pin != RecordPins().end() && pin->second == R.record_fnv1a64,
                "(e) " + key + " full record_fnv1a64 " + R.record_fnv1a64 + " = pin " + (pin == RecordPins().end() ? "-" : pin->second));
        rec_ok += ok ? 1 : 0;
        std::cout << "  RECORD " << hzr::StatusSignature(R) << "\n";
        std::cout << "  HASH " << R.id << " record_fnv1a64=" << R.record_fnv1a64 << " status=" << R.status << " decomposed_ells=" << decomposed
                  << " meetings=" << meet << " secs=" << R.secs << "\n";
        for (const auto& f : R.fields)
            if (f.path.compare(0, 14, "G1.components[") == 0 || f.path.find(".components_through.") != std::string::npos ||
                f.path.find(".pair_kinds[") != std::string::npos || f.path.find(".maximal_ideals") != std::string::npos ||
                f.path.compare(0, 10, "G9.eisupb.") == 0)
                std::cout << "    " << f.path << " = " << f.value << "\n";
    }
    {   // (design Sec. 2.4) the forced congruences, observed on theta: at ell = 2 (and every ell | o_N) two Eisenstein factors e_{theta,j}
        // share an irreducible factor mod ell, in every cell (a necessary sign of the meeting; components_through needs generation there)
        for (const auto& [key, R] : results)
            for (uint64_t ell : {2ull, 3ull, 5ull}) {
                if (R.o_N % ell != 0) continue;
                std::map<std::vector<uint64_t>, uint32_t> cnt;
                for (const auto& c : R.comps) {
                    std::set<std::vector<uint64_t>> fs;
                    for (const auto& fa : hecke::FactorPoly(hecke::FpPoly64(ell, eu::PolyModEll(c.theta_poly, ell))).factors) fs.insert(fa.pi.coeffs);
                    for (const auto& pi : fs) ++cnt[pi];
                }
                uint32_t mx = 0;
                for (const auto& [pi, k] : cnt) mx = std::max(mx, k);
                std::cout << "  MEETING_THETA " << key << " ell=" << ell << " max_components_per_residue_factor=" << mx << "\n";
                if (mx >= 2) ++meetings;
                T.check(mx >= 2, "(e) " + key + ": Eisenstein--Eisenstein meeting at ell = " + std::to_string(ell) + " | o_N observed on theta");
            }
    }

    // ---- (f) bit identity: threads, reuse vs. recompute, ISA, the embedding at q*; the direction --------------------------------------
    bool threads_eq = true, reuse_eq = true, isa_eq = true, embed_eq = true;
    std::string isa_list;
    for (const Cell& c : std::vector<Cell>{{25, 0}, {49, 0}}) {
        const std::string key = Key(c.N, c.n);
        if (!results.count(key)) {
            threads_eq = reuse_eq = isa_eq = embed_eq = false;
            continue;
        }
        const std::string ref = results.at(key).text();
        for (uint32_t th : {1u, 16u}) {
            CellInput C = MakeInput(c.N, c.n, th);
            const eu::EisUpCharResult Rt = eu::EisUpCharStep(Bound(C));
            threads_eq = threads_eq && Rt.text() == ref;
            std::cout << "  THREADS " << key << " threads=" << th << " reused=" << Rt.reused << " computed=" << Rt.computed << "\n";
        }
        {
            eu::EisUpCharInput in = inputs.at(key).in;
            in.force_recompute = true;
            const eu::EisUpCharResult R = eu::EisUpCharStep(in);
            reuse_eq = reuse_eq && R.text() == ref && R.reused == 0;
        }
        {
            eu::EisUpCharInput in = inputs.at(key).in;
            in.embed_next_root = true;
            embed_eq = embed_eq && eu::EisUpCharStep(in).text() == ref;
        }
        const struct {
            ::RhoFillPath path;
            const char* name;
        } paths[] = {{::RhoFillPath::Scalar, "scalar"}, {::RhoFillPath::Avx2, "avx2"}, {::RhoFillPath::Avx512, "avx512"}};
        for (const auto& ph : paths) {
            if (!::rho_fill_path_available(ph.path)) continue;
            if (c.N == 25) isa_list += std::string(isa_list.empty() ? "" : ",") + ph.name;
            const ::ScopedRhoFillPath sp(ph.path);
            eu::EisUpCharInput in = inputs.at(key).in;
            in.force_recompute = true;
            isa_eq = isa_eq && eu::EisUpCharStep(in).text() == ref;
        }
    }
    {   // the record at threads 1 and 16 (25, 0): record_fnv1a64 and cert_sha256 equal
        bool rec_eq = recs.count("(25,0)") > 0;
        for (uint32_t th : {1u, 16u}) {
            if (!rec_eq) break;
            hzr::RecordSettings St = S4;
            St.threads = th;
            const hzr::HeckeStructureRecord R = hzr::BuildHeckeStructureRecord(25, 0, "full", St, src);
            rec_eq = rec_eq && R.record_fnv1a64 == recs.at("(25,0)").record_fnv1a64 && R.cert_sha256 == recs.at("(25,0)").cert_sha256;
        }
        threads_eq = threads_eq && rec_eq;
    }
    T.check(threads_eq, "(f) text() at threads 1 / 4 / 16 equal; the (25,0) record's hashes at threads 1 / 4 / 16 equal");
    T.check(reuse_eq, "(f) reuse of the retained runs vs. forced recompute: text() equal");
    T.check(isa_eq, "(f) under every available rho-fill path (" + isa_list + "): text() equal");
    T.check(embed_eq, "(f) two embeddings at q* (the least and the next primitive root): text() equal (invariance only)");
    bool direction = false;
    {   // (referee M1) the direction at o_N = 10 (N = 121): psi(2) = zeta read at sigma_3 Q is zeta^3 (sigma_3 o lambda), not zeta^7
        uint32_t q = 1000000;
        while (!((q - 1) % 10 == 0 && [&] {
            for (uint32_t d = 2; d * d <= q; ++d)
                if (q % d == 0) return false;
            return true;
        }()))
            ++q;
        const uint32_t got = eu::DirectionRead(10, 1, 3, q);
        direction = got == 3;
        T.check(direction, "(f) direction at o_N = 10: psi(2) = zeta read at sigma_3 Q as zeta^" + std::to_string(got) + " (expected zeta^3)");
    }

    // ---- (f') the set S, the q* chooser, the cap refusal ------------------------------------------------------------------------------
    bool s_read = true, s_forced = false, s_appended = false, qstar_cap = false;
    {
        const std::vector<uint32_t> S225 = eu::ExtraPrimesS(225, {2, 17});
        s_read = s_read && T.check(S225 == std::vector<uint32_t>{7}, "(f') S at (225,0) with T_2, T_17 (index-2 subgroup {1,2,4,8}) is {7}");
        s_read = s_read && T.check(eu::ExtraPrimesS(225, {2, 7}).empty(), "(f') S at (225,0) with T_2, T_7 (generating) is empty");
        s_read = s_read && T.check(eu::ExtraPrimesS(49, {2, 5}).empty(), "(f') S at (49,0) with T_2, T_5 is empty");
        // (design M2; audit M1) S produced by ExtraPrimesS on a real (C1), end to end.  The design's (225,0) on T_2, T_17, U_3, U_5 can never
        // reach (C1): with the T_p in ker chi (chi = chi_3 chi_5, 2 = 17 mod 15 in {1,2,4,8}) and no p || N, the systems (1, (0,0)) and
        // (chi, (0,0)) agree on every operator of the list.  A cell with S != {} needs a U_q, q || N, to separate what the T_p do not: (18,0)
        // on T_7, T_13 (both = 1 mod 3: S = {5}), U_2, U_3 (U_2 reads psi(2), d = e_k, no cusp form).  Reached and equal, or a failure.
        {
            std::string why;
            bool reached = false, eq = false;
            const bool threw = Caught(
                [&] {
                    CellInput C;
                    heckez::RationalSettings rs;
                    rs.threads = 4;
                    rs.theta_order = heckez::ThetaOrder::LexThenGreedy;
                    C.QF = heckez::RationalStepFull(18, 0, {7, 13}, {2, 3}, rs, nullptr, &C.runs);
                    if (C.QF.c1_status != "certified" || C.QF.theta_index < 0) return;
                    reached = true;
                    const heckez::ThetaEntry& Th = C.QF.thetas[static_cast<size_t>(C.QF.theta_index)];
                    C.in.N = 18;
                    C.in.n = 0;
                    C.in.ops = C.QF.P;
                    C.in.nU = C.QF.nU;
                    C.in.theta = Th.coeffs;
                    C.in.f = Th.f;
                    C.in.e_k = C.QF.e_k;
                    C.in.d = C.QF.d;
                    C.in.b1 = C.QF.b1;
                    C.in.rs = rs;
                    const eu::EisUpCharResult R = eu::EisUpCharStep(Bound(C));
                    eq = R.S == std::vector<uint32_t>{5} && R.id_prime_extra == 1 && results.count("(18,0)") &&
                         CompStrings(R) == CompStrings(results.at("(18,0)"));
                    std::cout << "  S_FORCED (18,0) ops=T_7,T_13,U_2,U_3 theta=(";
                    for (size_t i = 0; i < R.theta.size(); ++i) std::cout << (i ? " " : "") << R.theta[i];
                    std::cout << ") S=" << (R.S.empty() ? 0 : R.S[0]) << " q*=" << R.id_prime << " extra=" << R.id_prime_extra << " comps=" << Summary(R) << "\n";
                },
                &why);
            s_read = s_read && T.check(reached && !threw && eq, std::string("(f') S forced by ExtraPrimesS at (18,0) on T_7, T_13, U_2, U_3: ") +
                                                                     (reached ? "" : "(C1) NOT reached; ") + "S = {5}, T_5 read at q*, labels equal the "
                                                                     "default list's: " + why);
            s_forced = reached && !threw && eq;
        }
        // the (6') read itself, forced: S = {13} appended at (25, 0) (13 = 3 mod 5 generates): T_13 read on K_q* at an extra run, psi(13)
        // from its residue; the components and labels equal the run without it
        if (inputs.count("(25,0)") && results.count("(25,0)")) {
            eu::EisUpCharInput in = inputs.at("(25,0)").in;
            in.validation_extra_S = {13};
            std::string why;
            eu::EisUpCharResult R;
            const bool threw = Caught([&] { R = eu::EisUpCharStep(in); }, &why);
            const bool eq = !threw && R.S == std::vector<uint32_t>{13} && R.id_prime_extra == 1 && CompStrings(R) == CompStrings(results.at("(25,0)")) &&
                            R.e_theta == results.at("(25,0)").e_theta;
            std::cout << "  S_READ (25,0) S=13 q*=" << R.id_prime << " extra=" << R.id_prime_extra << " comps=" << Summary(R) << "\n";
            s_read = s_read && T.check(eq, "(f') the (6') read of T_13 at q* (S = {13} appended at (25,0)): labels unchanged: " + why);
            s_appended = eq;
        }
        // a q* violating p~^{(k-1) o_N} != 1 is refused: q = 1 mod 4 with 7^4 = 1 mod q, i.e. q | 7^4 - 1 = 2400 = 2^5 3 5^2: q = 5
        s_read = s_read && T.check(!eu::IdPrimeArithmeticOk(5, 4, 2, {7}) && eu::IdPrimeArithmeticOk(13, 4, 2, {7}) &&
                                       !eu::IdPrimeArithmeticOk(7, 2, 2, {7}) && !eu::IdPrimeArithmeticOk(11, 4, 2, {}),
                                   "(f') the q* chooser: 7^{o_N} = 1 mod 5 refused, 13 admitted, q in S refused, q != 1 mod o_N refused");
        // a capped list without an admissible q*: refused(word_prime_cap)
        if (inputs.count("(25,0)")) {
            eu::EisUpCharInput in = inputs.at("(25,0)").in;
            in.runs = nullptr;
            in.rs.word_prime_cap = 2;
            std::string why;
            const bool caught = Caught([&] { eu::EisUpCharStep(in); }, &why);
            s_read = s_read && T.check(caught && why.find("refused(word_prime_cap)") != std::string::npos, "(f') a capped list: " + why);
        }
        // (audit m5) the q* chooser's OWN cap refusal: a cap that admits every read of (25,0) (shown by the untampered run at that cap),
        // with the chooser admitting no candidate, must refuse at the chooser
        if (inputs.count("(25,0)") && results.count("(25,0)")) {
            eu::EisUpCharInput in = inputs.at("(25,0)").in;
            const uint32_t cap = results.at("(25,0)").primes_used + static_cast<uint32_t>(results.at("(25,0)").skipped.size());
            in.rs.word_prime_cap = cap;
            std::string w1, w2;
            eu::EisUpCharResult R;
            const bool plain = !Caught([&] { R = eu::EisUpCharStep(in); }, &w1) && R.text() == results.at("(25,0)").text();
            eu::EisUpCharTamper tz;
            tz.qstar_none = true;
            in.tamper = &tz;
            const bool caught = Caught([&] { eu::EisUpCharStep(in); }, &w2);
            qstar_cap = plain && caught && w2.find("refused(word_prime_cap)") != std::string::npos;
            std::cout << "  QSTAR_CAP (25,0) cap=" << cap << " untampered=" << (plain ? "ok" : "FAIL:" + w1) << " no_admissible_qstar=" << w2 << "\n";
            s_read = s_read && T.check(qstar_cap, "(f') the q* chooser's own cap refusal at cap " + std::to_string(cap) +
                                                      " (the untampered run passes at that cap): " + w2);
        }
    }

    // ---- (g) tampers, each caught ------------------------------------------------------------------------------------------------------
    uint32_t tamper_ok = 0, tamper_all = 0;
    auto tamper = [&](const std::string& key, const eu::EisUpCharTamper& tz, uint32_t cap, const std::string& expect, const std::string& what) {
        ++tamper_all;
        CellInput* src_in = inputs.count(key) ? &inputs.at(key) : step_inputs.count(key) ? &step_inputs.at(key) : nullptr;
        if (!src_in) {
            T.check(false, "(g) " + what + ": no input");
            return;
        }
        eu::EisUpCharInput in = Bound(*src_in);
        in.tamper = &tz;
        if (cap) in.rs.word_prime_cap = cap;
        std::string why;
        const bool caught = Caught([&] { eu::EisUpCharStep(in); }, &why);
        const bool ok = caught && why.find(expect) != std::string::npos;
        tamper_ok += ok ? 1 : 0;
        T.check(ok, "(g) " + what + " caught by " + expect + ": " + why);
    };
    {
        eu::EisUpCharTamper a;
        a.factor_plus_one = 0;
        tamper("(25,0)", a, 0, "C-B4", "a factor e_{theta,j} with its constant coefficient + 1");
        eu::EisUpCharTamper b;
        b.swap_labels = true;
        tamper("(25,0)", b, 0, "C-B6", "the order-4 and quadratic labels swapped at (25,0)");
        eu::EisUpCharTamper c;
        c.psi_squared = true;
        tamper("(49,0)", c, 0, "C-B4", "psi replaced by psi^2 in the proposal at (49,0)");
        eu::EisUpCharTamper d;
        d.y_alpha_at = 3;
        // (audit M2) a VALID wrong type: v_3 = 2, 3 !| cond psi = 5, y_3 = 0 -> alpha on the order-4 orbit (5, (0)) at (225,0) (step input)
        tamper("(225,0)", d, 0, "C-B4", "y_3 = alpha on the order-4 orbit with y_3 = 0 at (225,0) (a valid wrong type)");
        eu::EisUpCharTamper e;
        e.ek_delta = 1;
        tamper("(9,0)", e, 16, "refused(word_prime_cap)", "e_k + 1");
        eu::EisUpCharTamper f;
        f.norm_full = true;
        tamper("(49,0)", f, 0, "C-B4", "the norm from Q(zeta_{o_N}) instead of the orbit product at (49,0)");
        eu::EisUpCharTamper g;
        g.forge_U_deg2 = true;
        tamper("(25,0)", g, 0, "C-B5", "a forged g_{U_5,j} on the degree-2 component at (25,0)");
        // (audit M2) the decisive comparisons with cohomology, each with C-B4 intact
        eu::EisUpCharTamper h;
        // (25,0): the last operator is U_5; the proposal g_{U_5,0} + 1 before the CRT comparison
        h.gprop_plus_one_op = inputs.count("(25,0)") ? static_cast<int32_t>(inputs.at("(25,0)").in.ops.size()) - 1 : 0;
        tamper("(25,0)", h, 0, "C-B5: g_{X,j} = charpoly", "the proposal g_{U_5,0} + 1 before C-B5's comparison (e_{theta,j} untouched)");
        eu::EisUpCharTamper i;
        i.misalign_unweighted = true;   // (25,0): theta = (0 0 1 1) does not weigh T_2; the order-4 orbit (omega, omega^3) read as omega^3 on T_2
        tamper("(25,0)", i, 0, "C-B6: the character read", "the order-4 orbit's closed form misaligned on T_2 (theta-weight 0; C-B4, C-B5 intact)");
        // C-B7 on the record path: f_0(theta') + 1 at ell = 13 of (25, 0) (a decomposed ell)
        ++tamper_all;
        hzr::RecordSettings St = S4;
        St.eisup_tamper_ell = 13;
        std::string why;
        const bool caught = Caught([&] { hzr::BuildHeckeStructureRecord(25, 0, "full", St, src); }, &why);
        const bool ok = caught && why.find("C-B7") != std::string::npos;
        tamper_ok += ok ? 1 : 0;
        T.check(ok, "(g) (25,0) ell = 13: charpoly(theta' | E_0) + 1 before the attribution caught by C-B7: " + why);
    }

    // ---- (i) amendment A5: schema tampers on a (b) record; edition 3 re-scanned ---------------------------------------------------------
    uint32_t schema_ok = 0, schema_all = 0, rescan_ok = 0, rescan_all = 0;
    {
        auto caught = [&](hzr::HeckeStructureRecord R, const std::string& what, const std::string& expect) {
            ++schema_all;
            const hzr::ScanResult sc = hzr::ScanRecord(R);
            bool hit = false;
            for (const auto& pr : sc.problems) hit = hit || pr.find(expect) != std::string::npos;
            const bool ok = sc.values_outside_lists > 0 && hit;
            schema_ok += ok ? 1 : 0;
            T.check(ok, "(i) " + what + " caught by the scan (" + expect + ")");
        };
        if (recs.count("(25,0)") && results.count("(25,0)")) {
            const hzr::HeckeStructureRecord base = recs.at("(25,0)");
            size_t j2 = 0;
            while (j2 < results.at("(25,0)").comps.size() && results.at("(25,0)").comps[j2].degree < 2) ++j2;
            const std::string cp = "G1.components[" + std::to_string(j2) + "]";
            {
                hzr::HeckeStructureRecord R = base;
                R.fields.push_back(hzr::Field{"G1", cp + ".U_5", "0x0", "zbig", ""});
                caught(R, "a forged U_5 zbig on the degree-2 row " + cp, "of degree >= 2 does not carry");
            }
            {
                hzr::HeckeStructureRecord R = base;
                for (auto& f : R.fields)
                    if (f.path == cp + ".degree") f.value = "3";
                caught(R, "a degree 3 on the degree-2 row (sum of the degrees != e_k)", "sum to");
            }
            {
                hzr::HeckeStructureRecord R = base;
                for (auto& f : R.fields)
                    if (f.path == cp + ".psi_order") f.value = "2";
                caught(R, "psi_order 2 on the degree-2 row (phi(order) != degree)", "phi(psi_order) = degree");
            }
            {
                hzr::HeckeStructureRecord R = base;
                for (auto& f : R.fields)
                    if (f.path == cp + ".psi_cond") f.value = "3";
                caught(R, "psi_cond 3 (not dividing s_N = 5)", "psi_cond | s_N");
            }
        } else {
            T.check(false, "(i) no (25,0) record from arm (e)");
        }
        const std::vector<std::string> ed3 = Ed3CellTexts();
        for (const std::string& t : ed3) {
            ++rescan_all;
            hzr::HeckeStructureRecord R;
            std::string why;
            if (hzr::VerifyCellText(t, IdOfCell(t), &R, &why)) {
                const hzr::ScanResult sc = hzr::ScanRecord(R);
                if (sc.values_outside_lists == 0) ++rescan_ok;
                else std::cout << "  [ED3] " << R.id << " values_outside_lists=" << sc.values_outside_lists << "\n";
            } else {
                std::cout << "  [ED3] " << IdOfCell(t) << " not verified: " << why << "\n";
            }
        }
        T.check(rescan_all == 226 && rescan_ok == 226, "(i) edition 3's 226 records re-scanned with values_outside_lists = 0 (A5 neutral): " +
                                                           std::to_string(rescan_ok) + "/" + std::to_string(rescan_all));
    }

    // ---- (j) the (a)-regression; (h) G10.contract per binary on every full N > 1 record written --------------------------------------
    uint32_t a_ok = 0, contract_ok = 0, contract_all = 0;
    for (const Cell& c : std::vector<Cell>{{11, 0}, {22, 0}, {2, 4}, {5, 4}}) {
        const std::string key = Key(c.N, c.n);
        const hzr::HeckeStructureRecord R = hzr::BuildHeckeStructureRecord(c.N, c.n, "full", S4, src);
        const auto pin = APins().find(key);
        const bool eq = pin != APins().end() && pin->second == R.record_fnv1a64 && !R.find("G9.eisupb.digest");
        a_ok += eq ? 1 : 0;
        T.check(eq, "(j) " + key + " full record_fnv1a64 " + R.record_fnv1a64 + " = (a)'s pin, no (b) key");
        ++contract_all;
        const hzr::Field* ct = R.find("G10.contract");
        contract_ok += ct && ct->value == kContract ? 1 : 0;
    }
    for (const auto& [key, R] : recs) {
        ++contract_all;
        const hzr::Field* ct = R.find("G10.contract");
        contract_ok += ct && ct->value == kContract ? 1 : 0;
    }
    bool a_step = false;
    {   // (design Sec. 2.10) the (b) step forced on (22, 0) reproduces (a)'s rows, e_theta and c_theta
        std::string why;
        Caught(
            [&] {
                CellInput C = MakeInput(22, 0, 4);
                eu::EisUpCharInput in = Bound(C);
                in.force_scope = true;
                const eu::EisUpCharResult B = eu::EisUpCharStep(in);
                eu::EisUpInput ain;
                ain.N = 22;
                ain.n = 0;
                ain.ops = in.ops;
                ain.nU = in.nU;
                ain.theta = in.theta;
                ain.f = in.f;
                ain.e_k = in.e_k;
                ain.d = in.d;
                ain.b1 = in.b1;
                ain.rs = in.rs;
                ain.runs = &C.runs;
                const eu::EisUpResult A = eu::EisUpStep(ain);
                bool eq = A.e_theta == B.e_theta && A.c_theta == B.c_theta && A.lines.size() == B.comps.size();
                for (size_t j = 0; eq && j < A.lines.size(); ++j)
                    eq = B.comps[j].degree == 1 && B.comps[j].U == A.lines[j].U && B.comps[j].lambda_theta == A.lines[j].lambda_theta &&
                         B.comps[j].psi_cond == 1 && B.comps[j].psi_order == 1;
                a_step = eq;
            },
            &why);
        T.check(a_step, "(j) the (b) step forced on (22,0) reproduces (a)'s rows, e_theta, c_theta (" + why + ")");
    }
    T.check(contract_ok == contract_all, "(h) G10.contract = " + kContract + " on every full N > 1 record written (" + std::to_string(contract_ok) + "/" +
                                             std::to_string(contract_all) + ")");

    // ---- (k) the seven refused(non_reduced) records of edition 3 (e^an != 1, p^3 | N) unchanged --------------------------------------
    uint32_t nr_ok = 0;
    {
        const auto ed3 = Ed3Hashes();
        for (const Cell& c : std::vector<Cell>{{16, 2}, {16, 4}, {27, 2}, {27, 4}, {32, 0}, {64, 0}, {125, 0}}) {
            const std::string key = "N" + std::to_string(c.N) + "_n" + std::to_string(c.n) + "_full";
            const hzr::HeckeStructureRecord R = hzr::BuildHeckeStructureRecord(c.N, c.n, "full", S4, src);
            const auto it = ed3.find(key);
            const hzr::Field* c1 = R.find("G1.C1.status");
            const bool eq = it != ed3.end() && it->second == R.record_fnv1a64 && c1 && c1->value == "refused(non_reduced)";
            nr_ok += eq ? 1 : 0;
            T.check(eq, "(k) " + key + " record_fnv1a64 " + R.record_fnv1a64 + " = edition 3, refused(non_reduced)");
        }
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool ok = T.failing == 0;
    std::cout << "HECKEZ_EISUPB cells=" << cells.size() << " step_cells=" << step_ok << "/" << step_all << "[" << step_list << "] records=" << rec_ok << "/" << rec_all << " components=" << comps_total << " deg_ge2=" << deg2
              << " meetings=" << meetings << " second_route=" << second_ok << "/" << cells.size() << " utag_div=" << utag_ok << "/" << utag_all
              << " disc_match=" << disc_ok << "/" << disc_all << " vdisc_record=" << vdisc_ok << "/" << cells.size()
              << " threads=1,4,16:" << (threads_eq ? "equal" : "DIFFER") << " reuse=" << (reuse_eq ? "equal" : "DIFFER") << " isa=" << isa_list << ":"
              << (isa_eq ? "equal" : "DIFFER") << " embed=" << (embed_eq ? "equal" : "DIFFER") << " direction=" << (direction ? "ok" : "FAIL")
              << " s_read=" << (s_read ? "ok" : "FAIL") << " s_forced=" << (s_forced ? "ok" : "FAIL") << " s_appended=" << (s_appended ? "ok" : "FAIL")
              << " qstar_cap=" << (qstar_cap ? "ok" : "FAIL") << " tamper=" << tamper_ok << "/" << tamper_all << " schema_tamper=" << schema_ok << "/"
              << schema_all << " ed3_rescan=" << rescan_ok << "/" << rescan_all << " a_regress=" << a_ok << "/4 a_step=" << (a_step ? "equal" : "DIFFER")
              << " contract=" << contract_ok << "/" << contract_all << " nonreduced_unchanged=" << nr_ok << "/7 checks=" << T.checks
              << " failing=" << T.failing << " secs=" << secs << " verdict=" << (ok ? "OK" : "FAIL") << "\n";
    return T.failing;
}
