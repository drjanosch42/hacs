// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZEisUpValidation -- slot-26 HZ-EISUP; see the header and HZEISUP_design.md v0.2 Sec. 5.2.

#include "HeckeZEisUpValidation.h"

#include "HeckeZEisUp.h"
#include "HeckeZRational.h"
#include "HeckeZRecord.h"
#include "HeckeZTor.h"
#include "HeckeZUTag.h"
#include "HeckeZUp.h"
#include "Descent/DescentRecord.h"
#include "../HeckeExt/HeckeZRecordSource.h"
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

// The inputs of EisUpStep for a cell exactly as the record builds them: P by P_rule (the anemic (C1) on the descent primes, enlarged),
// then the full step on P's T_p and the U_p (p | N), its retained runs kept.
struct CellInput {
    heckez::RationalResult QF;
    std::vector<heckez::PrimeRun> runs;
    eu::EisUpInput in;
};
CellInput MakeInput(uint32_t N, uint32_t n, uint32_t threads);
// the input with its retained runs bound (the pointer is taken at the object's final address)
const eu::EisUpInput& Bound(CellInput& C) {
    C.in.runs = &C.runs;
    return C.in;
}
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
    ENGINE_CHECK_MSG(C.QF.c1_status == "certified" && C.QF.theta_index >= 0, "HeckeZEisUpValidation: the full (C1) is not certified");
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
    C.in.runs = nullptr;   // bound by the caller once the object has its final address (Bound)
    return C;
}

// the digests of EisUpResult::text() at the ten cells (arm (a); first computed by this package, 2026-11-10, and cross-checked by (b), (c))
const std::map<std::string, std::string>& DigestPins() {
    static const std::map<std::string, std::string> m = {
        {"(11,0)", "0x68912e5b981f9b33"},
        {"(23,0)", "0x90a44ae0de5370b5"},
        {"(2,4)", "0xc662cd1e782a2dae"},
        {"(2,12)", "0x79fa566ed7b8d558"},
        {"(3,6)", "0xd771c919455a90b7"},
        {"(5,8)", "0xda860be8b7772ed3"},
        {"(7,4)", "0xdf1cac6f6feedac5"},
        {"(22,0)", "0x75be7149fe3e83de"},
        {"(33,0)", "0xd019c8ef058c3da0"},
        {"(38,0)", "0x5cae8c9bfe0d1871"},
    };
    return m;
}
// arm (d): record_fnv1a64 of the four records (HZ-EISUP's words; settings fingerprint unchanged)
const std::map<std::string, std::string>& RecordPins() {
    static const std::map<std::string, std::string> m = {
        {"(11,0)", "0x3409215438a036a6"}, {"(2,4)", "0x940f29550a7aa409"}, {"(22,0)", "0x8fd36b1a8b3405c9"}, {"(5,4)", "0x671876317a45370d"},
    };
    return m;
}

// HZ-DS edition 1: key -> record_fnv1a64 (docs/integral_census/results/hecke_ed1/rows.tsv)
std::map<std::string, std::string> Ed1Hashes() {
    std::map<std::string, std::string> m;
    std::ifstream in(SourcePath("docs/integral_census/results/hecke_ed1/rows.tsv"));
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
        if (col >= 0 && static_cast<size_t>(col) < f.size()) m[f[0]] = f[col];
    }
    return m;
}

// edition 2's 226 cell files (docs/integral_census/results/hecke_ed2/records_bundle.tar.gz), one JSON line each
std::vector<std::string> Ed2CellTexts() {
    std::vector<std::string> out;
    const std::string cmd = "tar -xzOf '" + SourcePath("docs/integral_census/results/hecke_ed2/records_bundle.tar.gz") + "' 2>/dev/null";
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
        if (!line.empty()) out.push_back(line);
    return out;
}
std::string IdOfCell(const std::string& text) {
    const std::string k = "\"id\":\"";
    const size_t a = text.find(k);
    if (a == std::string::npos) return std::string();
    const size_t b = text.find('"', a + k.size());
    return text.substr(a + k.size(), b - a - k.size());
}

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

std::string Tuples(const eu::EisUpResult& R) {
    std::ostringstream o;
    for (size_t j = 0; j < R.lines.size(); ++j) {
        o << (j ? ";" : "") << "(";
        for (size_t i = 0; i < R.lines[j].U.size(); ++i) o << (i ? "," : "") << R.lines[j].U[i];
        o << ")";
    }
    return o.str();
}

}  // namespace

uint32_t HeckeZEisUpValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeZEisUpValidation: the Eisenstein U_p systems of the full algebra at N > 1 (slot-26 HZ-EISUP; [C] 1.27.0-pre.1 "
                 "prop:heckeZeisup, item 8 (d')(ii); HZEISUP_design.md v0.2)\n";
    Tally T;
    const hecke::ScopedCoefficientRing ring(0);   // coefficient_ring = Z
    const std::vector<Cell> cells = {{11, 0}, {23, 0}, {2, 4}, {2, 12}, {3, 6}, {5, 8}, {7, 4}, {22, 0}, {33, 0}, {38, 0}};

    // ---- (a) certificate, (b) second route, (c) independent oracle --------------------------------------------------------------------
    uint32_t systems = 0, mismatched = 0, tag_ok = 0, tag_all = 0;
    std::map<std::string, CellInput> inputs;
    std::map<std::string, eu::EisUpResult> results;
    for (const Cell& c : cells) {
        const std::string key = Key(c.N, c.n);
        T.check(eu::InScope(c.N, c.n), "(a) " + key + " is in HZ-EISUP's scope (e^an_k(N) = 1)");
        std::string why;
        eu::EisUpResult R;
        const bool threw = Caught(
            [&] {
                inputs.emplace(key, MakeInput(c.N, c.n, 4));
                R = eu::EisUpStep(Bound(inputs.at(key)));
            },
            &why);
        if (!T.check(!threw, "(a) " + key + ": EisUpStep (C-E1 .. C-E5) ran: " + why)) continue;
        results[key] = R;
        systems += static_cast<uint32_t>(R.lines.size());
        T.check(R.lines.size() == static_cast<size_t>(R.e_k), "(a) " + key + ": e_k systems");
        const std::string dg = Hex64(R.digest());
        const auto pin = DigestPins().find(key);
        T.check(pin != DigestPins().end() && pin->second == dg, "(a) " + key + " digest " + dg + " = pin " + (pin == DigestPins().end() ? "-" : pin->second));
        std::cout << "  EISUP " << key << " e_k=" << R.e_k << " d=" << R.d << " theta=(";
        for (size_t i = 0; i < R.theta.size(); ++i) std::cout << (i ? " " : "") << R.theta[i];
        std::cout << ") tuples=" << Tuples(R) << " primes=" << R.primes_used << " skipped=" << R.skipped.size() << " q*=" << R.id_prime
                  << " reused=" << R.reused << " computed=" << R.computed << " digest=" << dg << "\n";
        // (b) the closed form (prop:heckeZeisup (5)): the second route, never a record field
        bool predicted = false;
        const auto pred = eu::PredictedEisUp(c.N, c.n + 2, &predicted);
        std::vector<std::vector<std::string>> got;
        for (const auto& L : R.lines) got.push_back(L.U);
        const bool agree = predicted && pred == got;
        if (!agree) ++mismatched;
        T.check(agree, "(b) " + key + ": the tuples equal the closed form E-hat(N, k)");
        // (c) every lambda_j(U_p) is a root of HZ-UTAG's chi_Q(U_p); at the tag-certified cells the eis(.) roots are the tuple set
        const std::vector<uint32_t> Up = R.Uprimes();
        for (size_t i = 0; i < Up.size(); ++i) {
            const heckez::UTagResult tag = heckez::UTagStep(c.N, c.n, Up[i], inputs.at(key).QF.P.size() > R.nU
                                                                                  ? std::vector<uint32_t>(inputs.at(key).QF.P.begin(), inputs.at(key).QF.P.end() - R.nU)
                                                                                  : std::vector<uint32_t>{},
                                                            inputs.at(key).in.rs);
            std::set<std::string> roots, eis;
            if (tag.m0 > 0) roots.insert("0x0");
            for (const auto& r : tag.roots) {
                const std::string h = eu::HexPlus("0x0", r.r);
                roots.insert(h);
                if (r.type == "eis") eis.insert(h);
            }
            std::set<std::string> ours;
            for (const auto& L : R.lines) ours.insert(L.U[i]);
            bool all_roots = true;
            for (const auto& v : ours) all_roots = all_roots && roots.count(v);
            T.check(all_roots, "(c) " + key + ": every lambda_j(U_" + std::to_string(Up[i]) + ") is a root of chi_Q(U_p)");
            if ((c.N == 2 && (c.n == 4 || c.n == 12)) || (c.N == 3 && c.n == 6)) {   // the cells (N, n) = (2, 4), (2, 12), (3, 6)
                ++tag_all;
                const bool eq = tag.upper_bound_certified && eis == ours;
                tag_ok += eq ? 1 : 0;
                T.check(eq, "(c) " + key + ": the tag-certified eis(.) roots equal the tuple set");
            }
        }
    }

    // ---- (d) the record path ----------------------------------------------------------------------------------------------------------
    const hzr::RecordSources src = hzr::DefaultRecordSources();
    hzr::RecordSettings S4;
    S4.threads = 4;
    uint32_t rec_ok = 0, rec_all = 0, ce7_ells = 0;
    std::map<std::string, hzr::HeckeStructureRecord> recs;
    for (const Cell& c : std::vector<Cell>{{11, 0}, {2, 4}, {22, 0}, {5, 4}}) {
        ++rec_all;
        const std::string key = Key(c.N, c.n);
        std::string why;
        hzr::HeckeStructureRecord R;
        if (!T.check(!Caught([&] { R = hzr::BuildHeckeStructureRecord(c.N, c.n, "full", S4, src); }, &why), "(d) " + key + " full: built: " + why))
            continue;
        recs[key] = R;
        const hzr::ScanResult sc = hzr::ScanRecord(R);
        for (const auto& pr : sc.problems) std::cout << "  [SCAN] " << key << " " << pr << "\n";
        auto val = [&](const std::string& p) { const hzr::Field* f = R.find(p); return f ? f->value : std::string("<absent>"); };
        bool blocked = false, rows = val("G1.components") == "certified";
        uint32_t decomposed = 0;
        for (const auto& f : R.fields) {
            if (f.value == "blocked(U_p)" && (f.path.compare(0, 13, "G1.components") == 0 || f.path.find(".maximal_ideals") != std::string::npos))
                blocked = true;
            if (f.path.size() > 15 && f.path.compare(f.path.size() - 15, 15, ".maximal_ideals") == 0 && f.value == "certified") ++decomposed;
        }
        const int64_t ek = std::stoll(val("G1.e_k"));
        for (int64_t j = 0; j < ek; ++j) rows = rows && val("G1.components[" + std::to_string(j) + "].type") == "eis";
        ce7_ells += decomposed;
        const auto pin = RecordPins().find(key);
        const bool ok = sc.values_outside_lists == 0 && sc.exact_under_bound == 0 && sc.overall_rule_ok && !blocked && rows &&
                        R.status == "final" && val("G10.contract") == heckez::up::kFullContract && R.settings_fingerprint == "0x8ffc29730c4f9672";
        T.check(ok, "(d) " + key + " full: scan clean, the rows written, no blocked(U_p) in G1.components / maximal_ideals, final, G10.contract");
        T.check(pin != RecordPins().end() && pin->second == R.record_fnv1a64,
                "(d) " + key + " full record_fnv1a64 " + R.record_fnv1a64 + " = pin " + (pin == RecordPins().end() ? "-" : pin->second));
        if (c.N == 5) T.check(val("G2.ell=5.maximal_ideals") == "certified", "(d) (5,4) full: decomposed at ell = 5 (ell | N, C-E7 the only meeting)");
        rec_ok += ok ? 1 : 0;
        std::cout << "  RECORD " << hzr::StatusSignature(R) << "\n";
        std::cout << "  HASH " << R.id << " record_fnv1a64=" << R.record_fnv1a64 << " status=" << R.status << " decomposed_ells=" << decomposed
                  << " secs=" << R.secs << "\n";
        for (const auto& f : R.fields)
            if (f.path.compare(0, 14, "G1.components[") == 0 || f.path.find(".components_through.") != std::string::npos ||
                f.path.find(".pair_kinds[") != std::string::npos)
                std::cout << "    " << f.path << " = " << f.value << "\n";
    }

    // ---- (e) bit identity: threads, reuse vs. recompute, ISA ---------------------------------------------------------------------------
    bool threads_eq = true, reuse_eq = true, isa_eq = true;
    std::string isa_list;
    for (const Cell& c : std::vector<Cell>{{22, 0}, {2, 12}}) {
        const std::string key = Key(c.N, c.n);
        if (!results.count(key)) {
            threads_eq = reuse_eq = isa_eq = false;
            continue;
        }
        const std::string ref = results.at(key).text();
        for (uint32_t th : {1u, 16u}) {
            CellInput C = MakeInput(c.N, c.n, th);
            const eu::EisUpResult Rt = eu::EisUpStep(Bound(C));
            threads_eq = threads_eq && Rt.text() == ref;
            std::cout << "  THREADS " << key << " threads=" << th << " reused=" << Rt.reused << " computed=" << Rt.computed << "\n";
        }
        {
            eu::EisUpInput in = inputs.at(key).in;
            in.force_recompute = true;
            const eu::EisUpResult R = eu::EisUpStep(in);
            reuse_eq = reuse_eq && R.text() == ref && R.reused == 0;
            std::cout << "  REUSE " << key << " reused=" << results.at(key).reused << " vs forced recompute computed=" << R.computed << "\n";
        }
        const struct {
            ::RhoFillPath path;
            const char* name;
        } paths[] = {{::RhoFillPath::Scalar, "scalar"}, {::RhoFillPath::Avx2, "avx2"}, {::RhoFillPath::Avx512, "avx512"}};
        for (const auto& ph : paths) {
            if (!::rho_fill_path_available(ph.path)) continue;
            if (c.N == 22) isa_list += std::string(isa_list.empty() ? "" : ",") + ph.name;
            const ::ScopedRhoFillPath sp(ph.path);
            eu::EisUpInput in = inputs.at(key).in;
            in.force_recompute = true;
            isa_eq = isa_eq && eu::EisUpStep(in).text() == ref;
        }
    }
    T.check(threads_eq, "(e) EisUpResult::text() at threads 1 / 4 / 16 equal");
    T.check(reuse_eq, "(e) reuse of the retained runs vs. forced recompute: text() equal");
    T.check(isa_eq, "(e) under every available rho-fill path (" + isa_list + "): text() equal");

    // ---- (f) tampered inputs, each caught ---------------------------------------------------------------------------------------------
    uint32_t tamper_ok = 0;
    const uint32_t tamper_all = 6;
    auto tamper = [&](const std::string& key, const eu::EisUpTamper& tz, uint32_t cap, const std::string& expect, const std::string& what) {
        if (!inputs.count(key)) {
            T.check(false, "(f) " + what + ": no input");
            return;
        }
        eu::EisUpInput in = inputs.at(key).in;
        in.tamper = &tz;
        if (cap) in.rs.word_prime_cap = cap;
        std::string why;
        const bool caught = Caught([&] { eu::EisUpStep(in); }, &why);
        const bool ok = caught && why.find(expect) != std::string::npos;
        tamper_ok += ok ? 1 : 0;
        T.check(ok, "(f) " + what + " caught by " + expect + ": " + why);
    };
    {
        eu::EisUpTamper a;
        a.f_plus_one = true;
        tamper("(2,12)", a, 0, "C-E2", "f_theta + 1");
        eu::EisUpTamper b;
        b.ek_delta = 1;
        tamper("(22,0)", b, 16, "refused(word_prime_cap)", "e_k + 1");
        eu::EisUpTamper c;
        c.theta_after = true;
        tamper("(22,0)", c, 0, "C-E5", "theta altered after e_theta");
        eu::EisUpTamper d;
        d.drop_top_candidate = true;
        tamper("(3,6)", d, 0, "C-E4", "C_p without p^{k-1}");
        eu::EisUpTamper e;   // (audit m7) U_2 swapped between (1, 11) and (2, 1): (2, 11) and (1, 1)
        e.swap_p = 2;
        e.swap_a = {"1", "11"};
        e.swap_b = {"2", "1"};
        tamper("(22,0)", e, 0, "C-E5", "(22,0) U_2 swapped between (1,11) and (2,1)");
        // (audit M4) at (5, 4), ell = 5: lambda_0(U_5) + 1 before the attribution
        hzr::RecordSettings St = S4;
        St.eisup_tamper_ell = 5;
        std::string why;
        const bool caught = Caught([&] { hzr::BuildHeckeStructureRecord(5, 4, "full", St, src); }, &why);
        const bool ok = caught && why.find("C-E7") != std::string::npos;
        tamper_ok += ok ? 1 : 0;
        T.check(ok, "(f) (5,4) ell = 5: lambda_0(U_5) + 1 before the attribution caught by C-E7: " + why);
    }

    // ---- (g) HZ hash invariance: the settings fingerprint, the 15 HZ_N1_n<n>_full records = HZ-DS ed. 1 --------------------------------
    uint32_t hz_ok = 0;
    {
        const auto ed1 = Ed1Hashes();
        hzr::RecordSettings S1;
        S1.threads = 4;
        for (uint32_t n = 2; n <= 30; n += 2) {
            const hzr::HeckeStructureRecord R = hzr::BuildHeckeStructureRecord(1, n, "full", S1, src);
            const auto it = ed1.find("N1_n" + std::to_string(n) + "_full");
            const bool eq = it != ed1.end() && it->second == R.record_fnv1a64 && R.settings_fingerprint == "0x8ffc29730c4f9672";
            hz_ok += eq ? 1 : 0;
            T.check(eq, "(g) HZ_N1_n" + std::to_string(n) + "_full record_fnv1a64 " + R.record_fnv1a64 + " = ed. 1");
        }
    }

    // ---- (h) the A4 schema: forged layouts caught; edition 2's records re-scanned (neutrality) ----------------------------------------
    uint32_t schema_ok = 0;
    const uint32_t schema_all = 10;
    uint32_t rescan_ok = 0, rescan_all = 0;
    {
        const std::vector<std::string> ed2 = Ed2CellTexts();
        hzr::HeckeStructureRecord anemic, level1;
        bool have_an = false, have_l1 = false;
        for (const std::string& t : ed2) {
            ++rescan_all;
            hzr::HeckeStructureRecord R;
            std::string why;
            const bool ok = hzr::VerifyCellText(t, IdOfCell(t), &R, &why);
            if (ok) {
                const hzr::ScanResult sc = hzr::ScanRecord(R);
                if (sc.values_outside_lists == 0) ++rescan_ok;
                else std::cout << "  [ED2] " << R.id << " values_outside_lists=" << sc.values_outside_lists << "\n";
                if (!have_an && R.algebra == "anemic" && R.find("G1.components[0].type")) { anemic = R; have_an = true; }
                if (!have_l1 && R.N == 1 && R.find("G1.components[0].type")) { level1 = R; have_l1 = true; }
            } else {
                std::cout << "  [ED2] " << IdOfCell(t) << " not verified: " << why << "\n";
            }
        }
        T.check(rescan_all == 226 && rescan_ok == 226, "(h) edition 2's 226 records re-scanned with values_outside_lists = 0 (A4 neutral): " +
                                                           std::to_string(rescan_ok) + "/" + std::to_string(rescan_all));
        // expect: "" = any problem; else (code audit m4) a problem text containing it, so the case is caught by the intended check
        auto caught = [&](hzr::HeckeStructureRecord R, const std::string& what, const std::string& expect = std::string()) {
            const hzr::ScanResult sc = hzr::ScanRecord(R);
            bool hit = expect.empty();
            for (const auto& pr : sc.problems) hit = hit || pr.find(expect) != std::string::npos;
            const bool ok = sc.values_outside_lists > 0 && hit;
            schema_ok += ok ? 1 : 0;
            T.check(ok, "(h) " + what + " caught by the scan");
        };
        if (have_an) {
            hzr::HeckeStructureRecord R = anemic;
            R.fields.push_back(hzr::Field{"G1", "G1.components[2].type", "eis", "component_type", ""});
            caught(R, "a forged components[2] on an anemic record (" + anemic.id + ")");
        } else {
            T.check(false, "(h) no anemic record with components in edition 2");
        }
        if (have_l1) {
            hzr::HeckeStructureRecord R = level1;
            R.fields.push_back(hzr::Field{"G1", "G1.components[2].type", "eis", "component_type", ""});
            caught(R, "a forged components[2] on a level-1 record (" + level1.id + ")");
        } else {
            T.check(false, "(h) no level-1 record with components in edition 2");
        }
        if (recs.count("(22,0)")) {
            const hzr::HeckeStructureRecord& base = recs.at("(22,0)");
            {
                hzr::HeckeStructureRecord R = base;
                R.fields.push_back(hzr::Field{"G1", "G1.components[0].U_3", "0x1", "zbig", ""});
                caught(R, "a forged G1.components[0].U_3 on (22,0) full (3 !| 22)");
            }
            {
                hzr::HeckeStructureRecord R = base;
                R.fields.erase(std::remove_if(R.fields.begin(), R.fields.end(), [](const hzr::Field& f) { return f.path == "G1.components[0].U_11"; }),
                               R.fields.end());
                caught(R, "a missing G1.components[0].U_11 on (22,0) full");
            }
            {
                hzr::HeckeStructureRecord R = base;
                int64_t last = -1;
                for (const auto& f : R.fields)
                    if (f.path.compare(0, 14, "G1.components[") == 0) last = std::max<int64_t>(last, std::stoll(f.path.substr(14)));
                const std::string cp = "G1.components[" + std::to_string(last + 1) + "]";
                for (const auto& kv : std::vector<std::pair<std::string, std::string>>{{"type", "eis"}, {"U_2", "0x1"}, {"U_11", "0x1"}})
                    R.fields.push_back(hzr::Field{"G1", cp + "." + kv.first, kv.second, kv.first == "type" ? "component_type" : "zbig", ""});
                caught(R, "an Eisenstein count e_k + 1 on (22,0) full");
            }
            // (code audit m4) the scope and the per-m completeness of the A4 layout
            {
                hzr::HeckeStructureRecord R = base;
                for (auto& f : R.fields)
                    if (f.path == "G1.C1.status") f.value = "open(cap)";
                caught(R, "the layout beside G1.C1.status = open(cap) on (22,0) full", "without G1.C1.status = certified");
            }
            {
                hzr::HeckeStructureRecord R = base;
                uint32_t out_N = 0;   // e^an_k(22) = 1 at every weight: move the level to the first N with e^an_k(N) != 1 at n = 0
                for (uint32_t N = 2; N <= 1000 && !out_N; ++N)
                    if (!eu::InScope(N, R.n)) out_N = N;
                R.N = out_N;
                if (out_N) caught(R, "the (22,0) full layout relabelled to N = " + std::to_string(out_N) + " (e^an_k(N) != 1)", "outside HZ-EISUP's scope");
                else T.check(false, "(h) no N <= 1000 with e^an_k(N) != 1 at n = 0");
            }
            std::string ell_c;   // an ell of (22,0) full with maximal_ideals = certified
            for (const auto& f : base.fields)
                if (ell_c.empty() && f.value == "certified" && f.path.size() > 15 && f.path.compare(f.path.size() - 15, 15, ".maximal_ideals") == 0)
                    ell_c = f.path.substr(7, f.path.size() - 15 - 7);
            if (!ell_c.empty()) {
                const std::string m0 = "G2.ell=" + ell_c + ".m[0]";
                {
                    hzr::HeckeStructureRecord R = base;
                    R.fields.erase(std::remove_if(R.fields.begin(), R.fields.end(),
                                                  [&](const hzr::Field& f) { return f.path == m0 + ".components_through.1"; }),
                                   R.fields.end());
                    caught(R, "a missing " + m0 + ".components_through.1 on (22,0) full", "lacks components_through.1");
                }
                {
                    hzr::HeckeStructureRecord R = base;
                    std::string par;
                    std::vector<size_t> pass, nopass;
                    for (const auto& f : R.fields) {
                        const std::string t = m0 + ".components_through.";
                        if (f.path.compare(0, t.size(), t) == 0) {
                            (f.value == "certified" ? pass : nopass).push_back(std::stoul(f.path.substr(t.size())));
                            par = f.parent;
                        }
                    }
                    if (!pass.empty() && !nopass.empty()) {
                        const size_t a = std::min(pass[0], nopass[0]), b = std::max(pass[0], nopass[0]);
                        const std::string pk = m0 + ".pair_kinds[" + std::to_string(a) + "," + std::to_string(b) + "]";
                        R.fields.push_back(hzr::Field{"G2", pk + ".depth", "not_run", "not_run", par});
                        R.fields.push_back(hzr::Field{"G2", pk + ".status", "not_run", "status", par});
                        R.fields.push_back(hzr::Field{"G2", pk + ".kind", "not_run", "pair_kind", par});
                        caught(R, "a forged " + pk + " (one component not passing) on (22,0) full", "not both passing");
                    } else {
                        T.check(false, "(h) (22,0) full " + m0 + ": no passing and non-passing component to forge a pair");
                    }
                }
                {
                    hzr::HeckeStructureRecord R = base;
                    std::string par;
                    for (const auto& f : R.fields)
                        if (f.path == m0 + ".components_through.0") par = f.parent;
                    R.fields.push_back(hzr::Field{"G2", m0 + ".components_through.eis", "certified", "through", par});
                    caught(R, "a level-1 key " + m0 + ".components_through.eis on (22,0) full", "level-1 / anemic attribution key");
                }
            } else {
                T.check(false, "(h) (22,0) full: no ell with maximal_ideals = certified");
            }
        } else {
            T.check(false, "(h) no (22,0) full record from arm (d)");
        }
    }

    // ---- (i) (code audit m2) the support word: not_run exactly where theta' involves an operator the torsion read does not carry --------
    // A forged read at ell = 5 carrying T_3, T_5, T_7 (no U_2), one common block with degree-1 residues x - 1, x - 2, x - 3 at
    // them; theta' on thP = (T_3, U_2, T_5, T_7) (U_2 listed before T_5, T_7, as the full rows list the U_p before an added T_ell).
    uint32_t sup_ok = 0, sup_all = 0;
    {
        heckez::tor::TorRead rd;
        rd.ell = 5;
        rd.status = "ok";
        rd.P = {3, 5, 7};
        heckez::tor::BlockRead b;
        b.kind = "common";
        b.residue = {{4, 1}, {3, 1}, {2, 1}};   // x - 1, x - 2, x - 3 mod 5
        rd.blocks.push_back(b);
        const std::vector<uint32_t> thP = {3, 2, 5, 7};
        auto sup_case = [&](const std::vector<uint32_t>& c, const std::vector<uint64_t>& pi, bool involved_only, const std::string& expect,
                            const std::string& what) {
            ++sup_all;
            const std::string got = hzr::SupportWordOfRead(&rd, 5, c, thP, pi, involved_only);
            const bool ok = got == expect;
            sup_ok += ok ? 1 : 0;
            T.check(ok, "(i) " + what + ": support = " + got + ", expected " + expect);
        };
        // the full rows (involved_only): theta' without U_2 is determined
        sup_case({0, 0, 1, 0}, {3, 1}, true, "both", "theta' = T_5 (a unit listed after U_2), full rows");
        sup_case({0, 0, 1, 0}, {4, 1}, true, "free", "theta' = T_5 at another m, full rows");
        sup_case({1, 0, 0, 1}, {1, 1}, true, "both", "theta' = T_3 + T_7 (U_2 coefficient 0; 1 + 3 = 4), full rows");
        // theta' involving U_2 (absent from the read): not_run, final ([C] (d')(ii), M1)
        sup_case({0, 1, 0, 0}, {3, 1}, true, "not_run", "theta' = U_2, full rows");
        sup_case({1, 1, 0, 0}, {3, 1}, true, "not_run", "theta' = T_3 + U_2, full rows");
        // the anemic / level-1 rule (involved_only = false) is unchanged: every operator of thP, the old over-strict word here
        sup_case({0, 0, 1, 0}, {3, 1}, false, "not_run", "theta' = T_5 after U_2, the anemic rule unchanged");
        // an involved operator of residue degree 2: not determined in both rules
        heckez::tor::TorRead rd2 = rd;
        rd2.blocks[0].residue[2] = {2, 0, 1};   // x^2 + 2 at T_7
        ++sup_all;
        const std::string g2 = hzr::SupportWordOfRead(&rd2, 5, {1, 0, 0, 1}, thP, {1, 1}, true);
        sup_ok += g2 == "not_run" ? 1 : 0;
        T.check(g2 == "not_run", "(i) theta' = T_3 + T_7 with a degree-2 residue at T_7: support = " + g2 + ", expected not_run");
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool ok = T.failing == 0;
    std::cout << "HECKEZ_EISUP cells=" << cells.size() << " systems=" << systems << " mismatched=" << mismatched << " tag_xcheck=" << tag_ok << "/"
              << tag_all << " records=" << rec_ok << "/" << rec_all << " ce7_ells=" << ce7_ells << " threads=1,4,16:" << (threads_eq ? "equal" : "DIFFER")
              << " reuse=" << (reuse_eq ? "equal" : "DIFFER") << " isa=" << isa_list << ":" << (isa_eq ? "equal" : "DIFFER") << " tamper=" << tamper_ok
              << "/" << tamper_all << " schema_tamper=" << schema_ok << "/" << schema_all << " support_cases=" << sup_ok << "/" << sup_all << " ed2_rescan=" << rescan_ok << "/" << rescan_all
              << " hz_level1=" << hz_ok << "/15 checks=" << T.checks << " failing=" << T.failing << " secs=" << secs << " verdict=" << (ok ? "OK" : "FAIL")
              << "\n";
    return T.failing;
}
