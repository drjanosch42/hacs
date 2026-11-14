// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZExportWordValidation -- slot-27 EXW; see the header and docs/implementation/slot27/EXW_design.md v0.2 Sec. 6.

#include "HeckeZExportWordValidation.h"

#include "HeckeZLocal.h"
#include "HeckeZRecord.h"
#include "Descent/DescentModuleExportZ.h"
#include "../HeckeExt/HeckeZLocalSource.h"
#include "../HeckeExt/HeckeZRecordSource.h"
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace hzr = heckez::record;
namespace loc = heckez::local;
namespace mx = integral_descent::modexport;

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

bool Caught(const std::function<void()>& fn, std::string* what) {
    try {
        fn();
    } catch (const std::exception& e) {
        *what = e.what();
        return true;
    }
    *what = "not caught";
    return false;
}

double Since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// a cell of the module-level arms (edition 3, rows.tsv: G1.P, b1, e_k, d resp. d^an)
struct Cell {
    uint32_t N, n;
    bool full;            // N > 1: the full algebra (U_p in the read set); N = 1: always full
    std::vector<uint32_t> P;
    uint32_t b1, e_k, d;
    std::string key() const {
        return "(" + std::to_string(N) + "," + std::to_string(n) + (N > 1 ? (full ? ",full" : ",anemic") : "") + ")";
    }
};
loc::ReadOptions Opts(const Cell& c) {
    loc::ReadOptions o;
    o.d_target = c.d;
    o.r_eis = c.e_k;
    o.k_weight = c.n + 2;
    o.full_algebra = c.N == 1 || c.full;
    o.eis_rational = !(c.N > 1 && c.full);
    o.allow_bad_primes = c.N > 1 && c.full;
    return o;
}
// the read set of a full N > 1 cell: P then U_p (p | N)
std::vector<uint32_t> Ops(const Cell& c) {
    std::vector<uint32_t> ops = c.P;
    if (c.N > 1 && c.full)
        for (uint32_t p = 2; p <= c.N; ++p) {
            bool prime = true;
            for (uint32_t q = 2; q * q <= p; ++q) prime = prime && p % q != 0;
            if (prime && c.N % p == 0) ops.push_back(p);
        }
    return ops;
}
// the read's words without the route's name (source, word): what Lemma 1 says is route-independent
std::string Words(loc::LocalRead L) {
    L.source = "-";
    L.word = "-";
    return L.text();
}
uint64_t ModuleFnv(const loc::LatticeModule& M) {
    std::ostringstream o;
    o << M.ell << "|" << M.K << "|" << M.D << "|";
    for (size_t i = 0; i < M.T.size(); ++i) {
        o << "T_" << M.P[i] << ":";
        for (uint64_t x : M.T[i]) o << x << ",";
        o << ";";
    }
    return engine::FnvBytes(engine::kFnvOffset64, o.str());
}

std::map<std::string, std::string> Hashed(const hzr::HeckeStructureRecord& R) {
    std::map<std::string, std::string> m;
    for (const auto& f : R.fields)
        if (f.group != "G10") m[f.path] = f.value;
    return m;
}
// G10 with every wall-clock figure ("secs=<x>" inside a value, a path ending in .secs) blanked: provenance compared without timings
std::map<std::string, std::string> Prov(const hzr::HeckeStructureRecord& R) {
    std::map<std::string, std::string> m;
    for (const auto& f : R.fields) {
        if (f.group != "G10") continue;
        if (f.path.size() > 5 && f.path.compare(f.path.size() - 5, 5, ".secs") == 0) continue;
        std::string v = f.value;
        for (size_t at = v.find("secs="); at != std::string::npos; at = v.find("secs=", at + 6)) {
            size_t e = at + 5;
            while (e < v.size() && v[e] != ' ' && v[e] != ';' && v[e] != ',') ++e;
            v.replace(at + 5, e - at - 5, "*");
        }
        m[f.path] = v;
    }
    return m;
}
std::string Val(const hzr::HeckeStructureRecord& R, const std::string& p) {
    const hzr::Field* f = R.find(p);
    return f ? f->value : std::string("-");
}
bool HasPendingExportWord(const hzr::HeckeStructureRecord& R) {
    for (const auto& f : R.fields)
        if (f.group != "G10" && (f.value == "refused(export_word)" || f.value == "refused(k1_route)")) return true;
    return false;
}
// the triggered ell of a record: G10 ... .exw lines that start with free_block_z (root vs. (C3) by the text)
struct Triggers {
    std::set<std::string> root, c3, refused;
};
Triggers TriggersOf(const hzr::HeckeStructureRecord& R) {
    Triggers t;
    for (const auto& f : R.fields) {
        if (f.group != "G10") continue;
        const size_t e = f.path.find(".exw");
        if (e == std::string::npos || e + 4 != f.path.size()) continue;
        const size_t a = f.path.find(".ell=");
        const std::string ell = f.path.substr(a + 5, e - a - 5);
        if (f.value.compare(0, 12, "free_block_z") == 0 && f.value.find("c3_fallback_K1") == std::string::npos) t.root.insert(ell);
        if (f.value.find("free_block_z; export: c3_fallback_K1") != std::string::npos) t.c3.insert(ell);
        if (f.value.find("refused(") != std::string::npos && f.value.find("free_block_z") == std::string::npos) t.refused.insert(ell);
    }
    return t;
}
bool StartsWith(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
// EXW_design.md Sec. 5, the diff rule: the paths that may move at a triggered ell (root / (C3)) and the record aggregates
bool AllowedMove(const std::string& path, const Triggers& t) {
    for (const std::string& l : t.root) {
        const std::string c = "G3.C_T.ell=" + l + ".";
        if (path == c + "read" || path == c + "status" || path == c + "upper_v" || path == c + "lower_v") return true;
        if (StartsWith(path, "G3.eta_L.ell=" + l + ".")) return true;
        if (path == "G3.second_route.ell=" + l + ".agrees") return true;
        if (path == "G9.K3.ell=" + l || StartsWith(path, "G9.K3.ell=" + l + ".")) return true;
    }
    for (const std::string& l : t.c3) {
        if (StartsWith(path, "G2.ell=" + l + ".")) return true;
        if (StartsWith(path, "G9.K1.ell=" + l + ".")) return true;
        if (StartsWith(path, "G8.ell=" + l + ".")) return true;
    }
    static const std::set<std::string> agg = {"G0.operators", "G3.C_T.upper", "G3.C_T.status", "G3.C_T.exact", "G3.fibre_product",
                                              "G0.status", "G9.trust.tb_mat"};
    return agg.count(path) > 0;
}
// the basis-dependent values (Lemma 1): may differ between two routes that read the same module up to conjugation
bool BasisDependent(const std::string& path) {
    if (path == "G9.trust.tb_mat") return true;
    const size_t s = path.rfind('.');
    const std::string leaf = s == std::string::npos ? path : path.substr(s + 1);
    return (StartsWith(path, "G9.K1.") && leaf == "satcert_sha256") ||
           (StartsWith(path, "G9.K3.") && (leaf == "dump_sha256" || leaf == "witness"));
}

std::string ReadFileText(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream o;
    o << in.rdbuf();
    return o.str();
}

// the 22 pending records of edition 3 (EXW_design.md Sec. 1) for the env arm
struct PendingRec {
    uint32_t N, n;
    const char* alg;
};
const std::vector<PendingRec>& Pending22() {
    static const std::vector<PendingRec> v = {
        {1, 32, "full"},     {1, 36, "full"},     {7, 12, "anemic"},  {167, 0, "anemic"}, {167, 0, "full"},   {173, 0, "anemic"},
        {173, 0, "full"},    {197, 0, "anemic"},  {197, 0, "full"},   {8, 10, "anemic"},  {26, 12, "anemic"}, {341, 0, "anemic"},
        {503, 0, "anemic"},  {1, 166, "full"},    {1, 198, "full"},   {1, 210, "full"},   {1, 216, "full"},   {1, 226, "full"},
        {1, 246, "full"},    {1, 254, "full"},    {1, 256, "full"},   {1, 268, "full"}};
    return v;
}

uint32_t EnvArmAll() {
    std::vector<PendingRec> cells = Pending22();
    if (const char* sel = std::getenv("HECKEZ_EXW_CELLS")) {
        cells.clear();
        std::stringstream ss(sel);
        std::string t;
        while (std::getline(ss, t, ',')) {
            const size_t a = t.find(':'), b = t.find(':', a + 1);
            if (a == std::string::npos || b == std::string::npos) continue;
            const uint32_t N = static_cast<uint32_t>(std::stoul(t.substr(0, a))), n = static_cast<uint32_t>(std::stoul(t.substr(a + 1, b - a - 1)));
            const std::string alg = t.substr(b + 1);
            for (const auto& p : Pending22())
                if (p.N == N && p.n == n && alg == p.alg) cells.push_back(p);
        }
    }
    std::cout << "EFFECTIVE HeckeZExportWordValidation arm=all cells=" << cells.size() << "\n";
    const hzr::RecordSources src = hzr::DefaultRecordSources();
    hzr::RecordSettings S;
    if (const char* th = std::getenv("ENGINE_THREADS")) S.threads = static_cast<uint32_t>(std::max(1, std::atoi(th)));
    uint32_t fail = 0, pairs = 0, finals = 0;
    for (const auto& c : cells) {
        const auto t0 = std::chrono::steady_clock::now();
        hzr::HeckeStructureRecord R;
        std::string why;
        if (Caught([&] { R = hzr::BuildHeckeStructureRecord(c.N, c.n, c.alg, S, src); }, &why)) {
            std::cout << "  EXW_ALL (" << c.N << "," << c.n << "," << c.alg << ") EXCEPTION " << why << "\n";
            ++fail;
            continue;
        }
        const Triggers t = TriggersOf(R);
        pairs += static_cast<uint32_t>(t.root.size() + t.c3.size());
        finals += R.status == "final";
        const hzr::ScanResult sc = hzr::ScanRecord(R);
        const bool ok = !HasPendingExportWord(R) && t.refused.empty() && sc.values_outside_lists == 0 && sc.exact_under_bound == 0 && sc.overall_rule_ok;
        fail += !ok;
        std::cout << "  EXW_ALL (" << c.N << "," << c.n << "," << c.alg << ") status=" << R.status << " root=" << t.root.size()
                  << " c3=" << t.c3.size() << " refused=" << t.refused.size() << " pending_export_word=" << HasPendingExportWord(R)
                  << " record_fnv1a64=" << R.record_fnv1a64 << " secs=" << Since(t0) << " sig=" << hzr::StatusSignature(R) << "\n";
        for (const auto& f : R.fields)
            if (f.group == "G10" && f.path.find(".exw") != std::string::npos) std::cout << "    " << f.path << " = " << f.value << "\n";
    }
    std::cout << "HECKEZ_EXW_ALL cells=" << cells.size() << " pairs=" << pairs << " final=" << finals << " failing=" << fail
              << " verdict=" << (fail == 0 ? "OK" : "FAIL") << "\n";
    return fail;
}

}  // namespace

uint32_t HeckeZExportWordValidation() {
    const hecke::ScopedCoefficientRing ring(0);   // coefficient_ring = Z
    if (const char* a = std::getenv("HECKEZ_EXW_ALL"); a && std::string(a) == "1") return EnvArmAll();
    const auto t_all = std::chrono::steady_clock::now();
    Tally T;
    const hzr::RecordSources src = hzr::DefaultRecordSources();
    hzr::RecordSources off = src;
    off.local_read_free = nullptr;
    const hzr::RecordSettings S;   // the defaults (threads 1)
    uint32_t identity = 0, identity_n = 0, pairs = 0, moved = 0, negative = 0, negative_n = 0;

    // ---- (A) route identity (Lemma 1): every word of the read equal on the EXPORT / level-1 route and on the free block -------------
    {
        const auto t0 = std::chrono::steady_clock::now();
        struct ACase {
            Cell c;
            uint64_t ell;
            std::vector<uint32_t> Ks;
        };
        const std::vector<ACase> cases = {
            {{11, 0, false, {2, 3, 5}, 3, 1, 2}, 5, {1, 2}},
            {{2, 6, false, {3, 5, 7}, 4, 2, 2}, 17, {1, 2}},
            {{2, 6, false, {3, 5, 7}, 4, 2, 2}, 2, {1, 9}},
            {{1, 40, true, {2, 3, 5}, 7, 1, 4}, 7, {1, 2, 3}},   // a level-1 torsion prime: T_7 added (ReadPrimes), compared (m3)
            {{8, 10, false, {3, 5, 7}, 22, 4, 6}, 2, {1, 25}},
            {{11, 0, false, {2, 3, 5}, 3, 1, 2}, 65521, {1}},
            {{11, 0, true, {2, 3, 5}, 3, 1, 2}, 5, {1, 2}},      // U_11 in the read set (cokernel route, m4 (i))
            {{11, 0, true, {2, 3, 5}, 3, 1, 2}, 2, {1, 2}},      // U_11, the tree route at ell = 2
            {{1, 22, true, {2, 3, 5}, 5, 1, 3}, 103, {2}},       // the level-1 H^1 route (m4 (ii))
        };
        for (const auto& a : cases) {
            const loc::ReadOptions o = Opts(a.c);
            uint32_t served = 0;
            for (uint32_t K : a.Ks) {
                const hzr::LocalReadOut ex = src.local_read(a.c.N, a.c.n, a.ell, K, Ops(a.c), a.c.b1, o, 0);
                const hzr::LocalReadOut fr = src.local_read_free(a.c.N, a.c.n, a.ell, K, Ops(a.c), a.c.b1, o, 0);
                T.check(fr.read && fr.M.source == "free_block_z", "(A) " + a.c.key() + " @" + std::to_string(a.ell) + " K=" + std::to_string(K) +
                                                                   ": the free read is served (" + fr.M.status + " " + fr.M.detail + ")");
                if (!ex.read) {
                    std::cout << "  (A) " << a.c.key() << " @" << a.ell << " K=" << K << ": the export route does not serve (" << ex.M.status
                              << "): no comparison\n";
                    continue;
                }
                ++served;
                ++identity_n;
                const bool eq = fr.read && Words(ex.L) == Words(fr.L);
                identity += eq;
                if (!T.check(eq, "(A) " + a.c.key() + " @" + std::to_string(a.ell) + " K=" + std::to_string(K) + ": words equal"))
                    std::cout << "    export: " << ex.L.text() << "\n    free:   " << fr.L.text() << "\n";
                if (a.ell == 7 && a.c.N == 1)
                    T.check(fr.L.t_ell == "added" && std::find(fr.M.P.begin(), fr.M.P.end(), 7u) != fr.M.P.end(),
                            "(A) (1,40) @7: T_7 is appended (t_ell = added) and read on the free block");
            }
            T.check(served >= 1, "(A) " + a.c.key() + " @" + std::to_string(a.ell) + ": at least one K served by both routes");
        }
        std::cout << "  (A) route identity " << identity << "/" << identity_n << " secs=" << Since(t0) << "\n";
    }

    // ---- (C) class C / k1: (C3) at K = 1 with ell > 2^32 (the export's word ring needs ell < 2^31; the level-1 route runs at K >= 2) --
    {
        const auto t0 = std::chrono::steady_clock::now();
        struct CCase {
            Cell c;
            uint64_t ell;
            const char* export_status;
        };
        const std::vector<CCase> cases = {
            {{7, 12, false, {2, 3, 5}, 16, 2, 8}, 5322111401ull, "refused(export_word)"},
            {{173, 0, false, {2, 3, 5}, 29, 1, 15}, 5608385124289ull, "refused(export_word)"},
            {{1, 32, true, {2, 3, 5}, 5, 1, 3}, 151628697551ull, "refused(word_modulus)"},
        };
        for (const auto& a : cases) {
            const loc::ReadOptions o = Opts(a.c);
            const hzr::LocalReadOut ex = src.local_read(a.c.N, a.c.n, a.ell, 1, Ops(a.c), a.c.b1, o, 0);
            const hzr::LocalReadOut fr = src.local_read_free(a.c.N, a.c.n, a.ell, 1, Ops(a.c), a.c.b1, o, 0);
            std::cout << "  (C) " << a.c.key() << " @" << a.ell << " export=" << ex.M.status << " free: c3=" << fr.L.c3 << " dim_A1=" << fr.L.dim_A1
                      << " mono=" << fr.L.mono << " regime=" << fr.L.regime << "\n";
            T.check(!ex.read && ex.M.status == a.export_status, "(C) " + a.c.key() + ": the old route refuses (" + ex.M.status + ")");
            T.check(fr.read && fr.L.c3 == "certified" && fr.L.dim_A1 == a.c.d && fr.L.mono != "none" && fr.L.mono != "",
                    "(C) " + a.c.key() + " @" + std::to_string(a.ell) + ": (C3) certified, dim A'_1 = d = " + std::to_string(a.c.d) +
                        " (c3probe.py), monogenic");
        }
        std::cout << "  (C) secs=" << Since(t0) << "\n";
    }

    // ---- (D) the overflow teeth ------------------------------------------------------------------------------------------------------
    {
        // the reduction FreeBlockModQ against an independent u128 Horner residue of the decimal text
        const std::vector<std::string> ints = {"1267650600228229401496703205376", "-1237940039285380274899124224", "0", "9223372036854775807",
                                               "-1", "340282366920938463463374607431768211457", "-5103536431379172", "5103536431379173"};
        mx::FreeBlockZ G;
        G.F.push_back({});
        for (size_t i = 0; i < 4; ++i) G.F[0].push_back(nf::BigInt::from_string(ints[i]));
        G.F.push_back({});
        for (size_t i = 4; i < 8; ++i) G.F[1].push_back(nf::BigInt::from_string(ints[i]));
        auto horner = [](const std::string& t, uint64_t q) {
            const bool neg = t[0] == '-';
            unsigned __int128 acc = 0;
            for (size_t i = neg ? 1 : 0; i < t.size(); ++i) acc = (acc * 10 + static_cast<unsigned>(t[i] - '0')) % q;
            const uint64_t v = static_cast<uint64_t>(acc);
            return neg && v ? q - v : v;
        };
        uint32_t bad = 0;
        for (uint64_t q : {uint64_t(2147483647ull), uint64_t(4294967311ull), uint64_t(5103536431379173ull), uint64_t(1ull << 62),
                           uint64_t(7ull * 7 * 7 * 7 * 7 * 7 * 7 * 7 * 7 * 7 * 7 * 7 * 7 * 7 * 7 * 7 * 7)}) {
            for (size_t o = 0; o < 2; ++o) {
                const std::vector<uint64_t> r = mx::FreeBlockModQ(G, o, q);
                for (size_t i = 0; i < 4; ++i) bad += r[i] != horner(ints[o * 4 + i], q);
            }
        }
        T.check(bad == 0, "(D) FreeBlockModQ = the u128 residue of the decimal text at q in {2^31 - 1, 2^32 + 15, 5103536431379173, 2^62, 7^17}");
        std::string why;
        T.check(Caught([&] { mx::FreeBlockModQ(G, 0, uint64_t(1) << 63); }, &why), "(D) FreeBlockModQ refuses q >= 2^63: " + why);
        // the word modulus first (Zlk::Fits): 3037000493^2 < 2^63 <= 3037000507^2 (the primes either side of 2^31.5; EXW_design.md Sec. 6 (D)
        // names 3037000499, which is neither prime nor above the bound -- 3037000507 is the first prime above)
        T.check(loc::Zlk::Fits(3037000493ull, 2) && !loc::Zlk::Fits(3037000507ull, 2), "(D) Fits(3037000493, 2) and not Fits(3037000507, 2)");
        G.F[0].resize(4);
        const loc::LatticeModule A = loc::LatticeFromFreeZ(11, 0, 3037000493ull, 2, {2}, G, {2, 3});
        const loc::LatticeModule B = loc::LatticeFromFreeZ(11, 0, 3037000507ull, 2, {2}, G, {2, 3});
        T.check(A.ok && A.D == 2 && A.source == "free_block_z" && !B.ok && B.status == "refused(word_modulus)",
                "(D) LatticeFromFreeZ: K = 2 served at 3037000493, refused(word_modulus) at 3037000507 (" + B.status + ")");
        T.check(Caught([&] { loc::LatticeFromFreeZ(11, 0, 5, 1, {7}, G, {2, 3}); }, &why), "(D) an operator outside the block is a defect: " + why);
    }

    // ---- (E) whole records: EXW on vs. the seam empty; (B) inside (8, 10) anemic -----------------------------------------------------
    std::map<std::string, hzr::HeckeStructureRecord> on_cache;
    {
        const auto t0 = std::chrono::steady_clock::now();
        struct ERec {
            uint32_t N, n;
            const char* alg;
            size_t root, c3;   // predicted triggers (EXW_design.md Sec. 1)
        };
        const std::vector<ERec> recs = {{173, 0, "anemic", 0, 1}, {197, 0, "anemic", 0, 1}, {1, 36, "full", 0, 1}, {8, 10, "anemic", 1, 0}};
        for (const auto& e : recs) {
            const std::string key = "(" + std::to_string(e.N) + "," + std::to_string(e.n) + "," + e.alg + ")";
            hzr::HeckeStructureRecord Ron, Roff;
            std::string why;
            if (!T.check(!Caught([&] { Ron = hzr::BuildHeckeStructureRecord(e.N, e.n, e.alg, S, src); }, &why), "(E) " + key + " EXW on: " + why))
                continue;
            if (!T.check(!Caught([&] { Roff = hzr::BuildHeckeStructureRecord(e.N, e.n, e.alg, S, off); }, &why), "(E) " + key + " EXW off: " + why))
                continue;
            on_cache[key] = Ron;
            const Triggers t = TriggersOf(Ron);
            pairs += static_cast<uint32_t>(t.root.size() + t.c3.size());
            T.check(t.root.size() == e.root && t.c3.size() == e.c3 && t.refused.empty(),
                    "(E) " + key + ": triggers root=" + std::to_string(t.root.size()) + " c3=" + std::to_string(t.c3.size()) + " as predicted");
            T.check(HasPendingExportWord(Roff) && Roff.status == "pending", "(E) " + key + ": the seam-empty record is edition 3's (pending, export word)");
            T.check(!HasPendingExportWord(Ron) && Ron.status == "final", "(E) " + key + ": EXW on: no export_word / k1_route left, final");
            const Triggers toff = TriggersOf(Roff);
            T.check(toff.root.empty() && toff.c3.empty() && toff.refused.empty(), "(E) " + key + ": no exw line with the seam empty");
            const hzr::ScanResult sc = hzr::ScanRecord(Ron);
            T.check(sc.values_outside_lists == 0 && sc.exact_under_bound == 0 && sc.overall_rule_ok, "(E) " + key + ": EXW-on record scans clean");
            // the diff rule (Sec. 5)
            const auto a = Hashed(Roff), b = Hashed(Ron);
            std::set<std::string> paths;
            for (const auto& [p, v] : a) paths.insert(p);
            for (const auto& [p, v] : b) paths.insert(p);
            uint32_t mv = 0, outside = 0;
            for (const auto& p : paths) {
                const auto ia = a.find(p), ib = b.find(p);
                const std::string va = ia == a.end() ? "<absent>" : ia->second, vb = ib == b.end() ? "<absent>" : ib->second;
                if (va == vb) continue;
                ++mv;
                const bool ok = AllowedMove(p, t);
                outside += !ok;
                std::cout << "    " << key << (ok ? " moved " : " OUTSIDE ") << p << ": " << va << " -> " << vb << "\n";
            }
            moved += mv;
            T.check(mv > 0 && outside == 0, "(E) " + key + ": " + std::to_string(mv) + " fields moved, all inside the diff rule (Sec. 5)");
            // G10: only the triggered ell's read lines, the exw keys and secs move
            const auto pa = Prov(Roff), pb = Prov(Ron);
            uint32_t prov_out = 0;
            for (const auto& [p, v] : pb) {
                const auto it = pa.find(p);
                if (it != pa.end() && it->second == v) continue;
                bool trig = false;
                for (const auto& l : t.root) trig = trig || p.find(".ell=" + l + ".") != std::string::npos;
                for (const auto& l : t.c3) trig = trig || p.find(".ell=" + l + ".") != std::string::npos;
                if (!trig) {
                    ++prov_out;
                    std::cout << "    " << key << " G10 OUTSIDE " << p << " = " << v << "\n";
                }
            }
            T.check(prov_out == 0, "(E) " + key + ": G10 moves only at the triggered ell (and secs)");
            if (e.N == 173) {
                const std::string mi = Val(Ron, "G2.ell=5608385124289.maximal_ideals");
                std::cout << "    (E) (173,0,anemic) G2.ell=5608385124289.maximal_ideals = " << mi << " decomposed_by = "
                          << Val(Ron, "G2.ell=5608385124289.decomposed_by") << "\n";
                T.check(Val(Ron, "G2.ell=5608385124289.generation.status") == "certified" &&
                            Val(Ron, "G2.ell=5608385124289.generation.route") == "saturation" && mi != "-" && mi != "not_run",
                        "(E) (173,0,anemic): ell = 5608385124289 >= 2^31: generation certified (saturation), maximal ideals by FpPoly64 (" + mi + ")");
            }
            if (e.N == 8) {
                // (B) class A beyond K_max: eta_oracle.py: eta exponents [0, 0, 8, 18] (the non-zero ones written), g_P 2^32
                std::cout << "    (B) (8,10,anemic) @2: read=" << Val(Ron, "G3.C_T.ell=2.read") << " lower_v=" << Val(Ron, "G3.C_T.ell=2.lower_v")
                          << " eta=" << Val(Ron, "G3.eta_L.ell=2.exponents") << " upper_v=" << Val(Ron, "G3.C_T.ell=2.upper_v")
                          << " K=" << Val(Ron, "G10.read.ell=2.K") << "\n";
                T.check(Val(Ron, "G3.C_T.ell=2.read") == "upper_bound_certified" && Val(Ron, "G3.C_T.ell=2.lower_v") == "18" &&
                            Val(Ron, "G3.C_T.ell=2.upper_v") == "32" && Val(Ron, "G10.read.ell=2.source") == "free_block_z",
                        "(B) (8,10,anemic) @2 at K0 = 33 > K_max = 25: upper_bound_certified, lower_v 18, upper_v 32 (the oracle)");
                std::multiset<std::string> eta;
                std::stringstream ss(Val(Ron, "G3.eta_L.ell=2.exponents"));
                for (std::string x; std::getline(ss, x, ',');) eta.insert(x);
                T.check(eta == std::multiset<std::string>{"18", "8"}, "(B) (8,10,anemic) @2: eta_L's non-trivial exponents {8, 18}");
            }
            // WriteCertFiles twice (M2): byte-stable, dump_sha256_recomputed = G9's
            if (e.N == 8 || e.N == 173) {
                namespace fs = std::filesystem;
                std::error_code ec;
                const fs::path base = fs::temp_directory_path(ec);
                const fs::path dir = (ec ? fs::path(".") : base) / ("oc_exw_cert_" + std::to_string(static_cast<long>(::getpid())));
                fs::remove_all(dir, ec);
                fs::create_directories(dir, ec);
                std::vector<std::string> fa, fb;
                std::map<std::string, std::string> first;
                const bool wrote = !Caught(
                    [&] {
                        fa = hzr::WriteCertFiles(Ron, src, S, dir.string());
                        for (const auto& f : fa) first[f] = ReadFileText((dir / f).string());
                        fb = hzr::WriteCertFiles(Ron, src, S, dir.string());
                    },
                    &why);
                uint32_t unstable = 0, shabad = 0, moved_files = 0;
                for (const auto& f : fb) {
                    const std::string txt = ReadFileText((dir / f).string());
                    unstable += !first.count(f) || first[f] != txt;
                    const size_t p = txt.find("dump_sha256_recomputed\t");
                    const size_t q1 = txt.find("satcert_sha256\t"), q3 = txt.find("\ndump_sha256\t");
                    if (p != std::string::npos) {
                        const size_t q = q1 != std::string::npos ? q1 + 15 : (q3 != std::string::npos ? q3 + 13 : std::string::npos);
                        shabad += q == std::string::npos || txt.substr(q, 64) != txt.substr(p + 23, 64);
                        // the triggered ell's file: its module is the free block's (the export route refuses it)
                        for (const auto& l : t.root) moved_files += f.find("_K3_l" + l + ".txt") != std::string::npos;
                        for (const auto& l : t.c3) moved_files += f.find("_K1_l" + l + ".txt") != std::string::npos;
                    }
                }
                fs::remove_all(dir, ec);
                std::cout << "    (E) " << key << " WriteCertFiles files=" << fb.size() << " unstable=" << unstable << " sha_mismatch=" << shabad
                          << " triggered_files=" << moved_files << "\n";
                T.check(wrote && fa == fb && unstable == 0 && shabad == 0 && moved_files == t.root.size() + t.c3.size(),
                        "(E) " + key + ": WriteCertFiles twice: byte-stable, sha256 = G9's, the triggered ell's file written (" + why + ")");
            }
        }
        std::cout << "  (E) secs=" << Since(t0) << "\n";
    }

    // ---- (F) route bit-identity: free-block threads 1 / 16, HZ-LOC-PAR par / serial ---------------------------------------------------
    {
        const auto t0 = std::chrono::steady_clock::now();
        struct FCase {
            Cell c;
            uint64_t ell;
            uint32_t K;
        };
        const std::vector<FCase> cases = {{{7, 12, false, {2, 3, 5}, 16, 2, 8}, 5322111401ull, 1}, {{8, 10, false, {3, 5, 7}, 22, 4, 6}, 2, 33}};
        for (const auto& a : cases) {
            std::string ref;
            uint64_t ref_mod = 0;
            std::vector<uint64_t> ref_fnv;
            bool all_eq = true;
            hzr::FreeSourceOptions fo16;
            fo16.threads = 16;
            fo16.stats = std::make_shared<hzr::FreeSourceStats>();
            const auto fn16 = hzr::FreeLocalRead(fo16);
            for (int v = 0; v < 3; ++v) {
                // v = 0: threads 1, par; v = 1: threads 16, par; v = 2: the same block (cached), serial
                hzr::FreeSourceOptions fo;
                fo.threads = 1;
                fo.stats = std::make_shared<hzr::FreeSourceStats>();
                const auto fn = v == 0 ? hzr::FreeLocalRead(fo) : fn16;
                if (v > 0) fo.stats = fo16.stats;
                loc::ParRoute pr;
                pr.par = v != 2;
                const loc::ScopedParRoute scope(pr);
                const hzr::LocalReadOut r = fn(a.c.N, a.c.n, a.ell, a.K, Ops(a.c), a.c.b1, Opts(a.c), 0);
                const std::string w = r.read ? r.L.text() : r.M.status;
                const uint64_t mf = r.M.ok ? ModuleFnv(r.M) : 0;
                if (v == 0) {
                    ref = w;
                    ref_mod = mf;
                    ref_fnv = fo.stats->F_fnv;
                } else {
                    all_eq = all_eq && w == ref && mf == ref_mod && fo.stats->F_fnv == ref_fnv;
                }
                T.check(r.read, "(F) " + a.c.key() + " @" + std::to_string(a.ell) + ": served");
            }
            T.check(all_eq, "(F) " + a.c.key() + " @" + std::to_string(a.ell) + ": words, module and F FNVs equal at threads 1 / 16 and par / serial");
        }
        std::cout << "  (F) secs=" << Since(t0) << "\n";
    }

    // ---- (G) teeth -----------------------------------------------------------------------------------------------------------------------
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::string why;
        // (i) the chi_Z check: F_3 + ell^{K-1} E_11 on the free read of (8, 10) anemic -> the record's ENGINE_CHECK
        hzr::RecordSources tam = src;
        const auto inner = src.local_read_free;
        tam.local_read_free = [inner](uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1,
                                      const loc::ReadOptions& o, uint32_t st) {
            hzr::LocalReadOut x = inner(N, n, ell, K, P, b1, o, st);
            if (x.M.ok && !x.M.T.empty() && x.M.D > 0) {
                const loc::Zlk R = loc::Zlk::Make(x.M.ell, x.M.K);
                x.M.T[0][0] = R.add(x.M.T[0][0], R.pw(x.M.K - 1));
            }
            return x;
        };
        const bool caught = Caught([&] { hzr::BuildHeckeStructureRecord(8, 10, "anemic", S, tam); }, &why);
        T.check(caught && why.find("chi_Z") != std::string::npos, "(G) a tampered F_3 (+ ell^(K-1) E_11) is caught by the chi_Z check: " + why);
        // (ii) a free-block refusal (rss cap 1 MiB): the pending word kept, G10 exw = refused(RssCeiling), the hashed record = the seam-empty one
        hzr::RecordSources rs = src;
        hzr::FreeSourceOptions fo;
        fo.rss_cap_bytes = 1ull << 20;
        rs.local_read_free = hzr::FreeLocalRead(fo);
        hzr::HeckeStructureRecord X;
        const bool built = !Caught([&] { X = hzr::BuildHeckeStructureRecord(8, 10, "anemic", S, rs); }, &why);
        const auto it = on_cache.find("(8,10,anemic)");
        hzr::HeckeStructureRecord Off;
        Caught([&] { Off = hzr::BuildHeckeStructureRecord(8, 10, "anemic", S, off); }, &why);
        const std::string exw = Val(X, "G10.read.ell=2.exw");
        std::cout << "  (G) rss cap 1 MiB: status=" << X.status << " exw=" << exw << "\n";
        T.check(built && X.status == "pending" && Val(X, "G3.C_T.ell=2.read") == "refused(export_word)" && exw == "refused(RssCeiling)" &&
                    X.record_fnv1a64 == Off.record_fnv1a64 && X.cert_sha256 == Off.cert_sha256 && it != on_cache.end(),
                "(G) a RssCeiling refusal of the free block keeps the pending word, names it in G10, and the hashed record is the seam-empty one");
        // (iii) a forced [Terms] of T_7 at (1, 40): the pair at 7 (T_7 appended) is refused by name; a pair without T_ell proceeds
        hzr::FreeSourceOptions ft;
        ft.force_terms = {7};
        ft.stats = std::make_shared<hzr::FreeSourceStats>();
        const auto fn = hzr::FreeLocalRead(ft);
        const Cell c40{1, 40, true, {2, 3, 5}, 7, 1, 4};
        const hzr::LocalReadOut r7 = fn(1, 40, 7, 2, c40.P, c40.b1, Opts(c40), 0);
        const hzr::LocalReadOut rb = fn(1, 40, 65537, 1, c40.P, c40.b1, Opts(c40), 0);
        std::cout << "  (G) forced [Terms] T_7: @7 " << r7.M.status << " (" << r7.M.detail << "); @65537 read=" << rb.read << " runs=" << ft.stats->runs
                  << "\n";
        T.check(!r7.read && r7.M.status == "blocked(free_block)" && r7.M.detail.compare(0, 15, "REFUSED [Terms]") == 0 && rb.read &&
                    ft.stats->ops == std::vector<uint32_t>({2, 3, 5}),
                "(G) a [Terms] refusal of the appended T_7: retried without it; the pair needing T_7 refused by name, ell = 65537 served");
        std::cout << "  (G) secs=" << Since(t0) << "\n";
    }

    // ---- (J) (audit M-EXW1) refusals per requested set; (K) (audit M-EXW2) the RssCeiling decision at a fixed lane count -------------
    // The cell (8, 10) anemic, P = {3, 5, 7}: its one trigger (@2, 2 | N) needs P only; @11 appends T_11 (a union with one more operator,
    // predicted above P's own).  pred = P's free block predicted at kFreeReadRssLanes lanes; the ceiling set to pred serves P exactly.
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::string why;
        const Cell c{8, 10, false, {3, 5, 7}, 22, 4, 6};
        const loc::ReadOptions o = Opts(c);
        auto source = [](uint64_t cap, uint32_t threads, std::shared_ptr<hzr::FreeSourceStats> st) {
            hzr::FreeSourceOptions fo;
            fo.rss_cap_bytes = cap;
            fo.threads = threads;
            fo.stats = st;
            return hzr::FreeLocalRead(fo);
        };
        const auto rss_refused = [](const hzr::LocalReadOut& r) {
            return !r.read && r.M.status == "blocked(free_block)" && r.M.detail.compare(0, 20, "REFUSED [RssCeiling]") == 0;
        };
        auto probe_st = std::make_shared<hzr::FreeSourceStats>();
        const hzr::LocalReadOut ref = source(3ull << 30, 0, probe_st)(8, 10, 2, 33, c.P, c.b1, o, 0);
        const uint64_t pred = probe_st->rss_pred;
        auto union_st = std::make_shared<hzr::FreeSourceStats>();
        source(3ull << 30, 0, union_st)(8, 10, 11, 1, c.P, c.b1, o, 0);
        const uint64_t pred_u = union_st->rss_pred;
        const std::string ref_w = ref.read ? ref.L.text() : ref.M.status;
        std::cout << "  (J) (8,10,anemic) pred(P)=" << pred << " pred(P+T_11)=" << pred_u << " lanes=" << hzr::kFreeReadRssLanes << "\n";
        T.check(ref.read && pred > 0 && pred_u > pred, "(J) (8,10) @2 served at the default ceiling; P + T_11 predicted above P");
        // (J)(i) refused-then-servable: @11 refused [RssCeiling] at the ceiling pred, then @2 (P only) is served, then @11 is not re-run
        {
            auto st = std::make_shared<hzr::FreeSourceStats>();
            const auto fn = source(pred, 0, st);
            const hzr::LocalReadOut a = fn(8, 10, 11, 1, c.P, c.b1, o, 0);
            const hzr::LocalReadOut b = fn(8, 10, 2, 33, c.P, c.b1, o, 0);
            const uint64_t runs_b = st->runs;
            const hzr::LocalReadOut a2 = fn(8, 10, 11, 1, c.P, c.b1, o, 0);
            std::cout << "  (J)(i) @11 " << a.M.status << "; @2 read=" << b.read << "; @11 again " << a2.M.status << " runs=" << st->runs << "\n";
            T.check(rss_refused(a) && b.read && b.L.text() == ref_w && rss_refused(a2) && runs_b == 2 && st->runs == 2,
                    "(J)(i) a RssCeiling refusal of P + T_11 does not block the later P-only read @2 (served, words = the reference); the "
                    "refused set is not re-run");
        }
        // (J)(ii) served-then-refused: @2 served, @11 refused, @2 again served from the cached block (no run)
        {
            auto st = std::make_shared<hzr::FreeSourceStats>();
            const auto fn = source(pred, 0, st);
            const hzr::LocalReadOut b = fn(8, 10, 2, 33, c.P, c.b1, o, 0);
            const hzr::LocalReadOut a = fn(8, 10, 11, 1, c.P, c.b1, o, 0);
            const uint64_t runs_a = st->runs;
            const hzr::LocalReadOut b2 = fn(8, 10, 2, 33, c.P, c.b1, o, 0);
            std::cout << "  (J)(ii) @2 read=" << b.read << "; @11 " << a.M.status << "; @2 again read=" << b2.read << " runs=" << st->runs << "\n";
            T.check(b.read && rss_refused(a) && b2.read && b2.L.text() == ref_w && st->runs == runs_a && st->ops == c.P,
                    "(J)(ii) after a served P and a refused P + T_11 the P-only read @2 is served again from the cached block (words = the reference)");
        }
        // (J)(iii) the record at the ceiling pred (final, = the default record), the same source then poisoned by a refused @11,
        // then WriteCertFiles with that source: the re-read @2 is served (pre-fix: "re-reads to G9's sha256 on neither route")
        {
            hzr::RecordSources s2 = src;
            s2.local_read_free = source(pred, 0, nullptr);
            hzr::HeckeStructureRecord R2;
            const bool built = !Caught([&] { R2 = hzr::BuildHeckeStructureRecord(8, 10, "anemic", S, s2); }, &why);
            const auto it = on_cache.find("(8,10,anemic)");
            const bool same = built && it != on_cache.end() && R2.record_fnv1a64 == it->second.record_fnv1a64 &&
                              R2.cert_sha256 == it->second.cert_sha256 && R2.status == "final";
            const hzr::LocalReadOut a = s2.local_read_free(8, 10, 11, 1, c.P, c.b1, o, 0);
            namespace fs = std::filesystem;
            std::error_code ec;
            const fs::path base = fs::temp_directory_path(ec);
            const fs::path dir = (ec ? fs::path(".") : base) / ("oc_exw_certj_" + std::to_string(static_cast<long>(::getpid())));
            fs::remove_all(dir, ec);
            fs::create_directories(dir, ec);
            std::vector<std::string> files;
            std::string wwhy;
            const bool wrote = !Caught([&] { files = hzr::WriteCertFiles(R2, s2, S, dir.string()); }, &wwhy);
            fs::remove_all(dir, ec);
            std::cout << "  (J)(iii) record status=" << R2.status << " same_as_default=" << same << "; @11 " << a.M.status
                      << "; WriteCertFiles wrote=" << wrote << " files=" << files.size() << (wrote ? "" : " (" + wwhy + ")") << "\n";
            T.check(same && rss_refused(a) && wrote && !files.empty(),
                    "(J)(iii) (8,10,anemic) at the ceiling pred is the default record; a later refused @11 on the same source does not make "
                    "WriteCertFiles throw");
        }
        // (K) the decision at threads 1 / 4 / 16: served at the ceiling pred, refused [RssCeiling] at pred - 1, the same prediction
        // (pre-fix: predicted at the run's lanes, so threads 1 and 4 were served at pred - 1)
        {
            uint32_t equal = 0;
            for (uint32_t th : {1u, 4u, 16u}) {
                auto s_at = std::make_shared<hzr::FreeSourceStats>();
                auto s_below = std::make_shared<hzr::FreeSourceStats>();
                const hzr::LocalReadOut at = source(pred, th, s_at)(8, 10, 2, 33, c.P, c.b1, o, 0);
                const hzr::LocalReadOut below = source(pred - 1, th, s_below)(8, 10, 2, 33, c.P, c.b1, o, 0);
                const bool ok = at.read && at.L.text() == ref_w && rss_refused(below) && s_at->rss_pred == pred && s_below->rss_pred == pred;
                equal += ok;
                std::cout << "  (K) threads=" << th << ": @pred read=" << at.read << " @pred-1 " << below.M.status << " rss_pred=" << s_at->rss_pred
                          << "/" << s_below->rss_pred << "\n";
                T.check(ok, "(K) threads " + std::to_string(th) + ": served at the ceiling pred, refused [RssCeiling] at pred - 1, predicted at " +
                                std::to_string(hzr::kFreeReadRssLanes) + " lanes");
            }
            T.check(equal == 3, "(K) the RssCeiling decision is the same at threads 1, 4, 16");
        }
        std::cout << "  (J)(K) secs=" << Since(t0) << "\n";
    }

    // ---- (H) record-level identity by forcing: the EXPORT read refused, the free read in its place = the unforged record -------------
    // from_K >= 2: the export's K_max refusal at K >= from_K (origin (a)); from_K == 1: [ModulePrecision] at every K; from_K == 0: the read is
    // served but its escalation is stopped by the export's K_max (origin (b): RootWord -> refused(export_word) on a served read)
    auto forged = [&](uint64_t at, uint32_t from_K, std::shared_ptr<hzr::FreeSourceStats> stats, bool seam) {
        hzr::RecordSources f = src;
        if (stats) {
            hzr::FreeSourceOptions fo;
            fo.stats = stats;
            f.local_read_free = hzr::FreeLocalRead(fo);
        }
        if (!seam) f.local_read_free = nullptr;
        f.local_read = [&src, at, from_K](uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1,
                                          const loc::ReadOptions& o, uint32_t steps) {
            if (ell != at || K < from_K) return src.local_read(N, n, ell, K, P, b1, o, steps);
            if (from_K == 0) {
                hzr::LocalReadOut x = src.local_read(N, n, ell, K, P, b1, o, 0);
                if (steps > 0 && x.read) {
                    x.L.root = "open(precision)";
                    x.L.C_T = "open(precision)";
                    x.escalation = "open(precision) at K=" + std::to_string(K + 1) + " (K = " + std::to_string(K + 1) + " > the export's K_max = " +
                                   std::to_string(K) + ")";
                }
                return x;
            }
            hzr::LocalReadOut x;
            x.M.N = N;
            x.M.n = n;
            x.M.ell = ell;
            x.M.K = K;
            x.M.P = P;
            x.M.source = "export";
            if (from_K >= 2) {   // mode 1 of HeckeZRecordValidation: the export's own K_max (pending export_word)
                x.M.status = "open(precision)";
                x.M.detail = "K = " + std::to_string(K) + " > the export's K_max = 1";
            } else {             // mode 6: [ModulePrecision] at every K
                x.M.status = "blocked(export)";
                x.M.detail = "REFUSED [ModulePrecision] K = " + std::to_string(K) + " above the read's precision";
            }
            return x;
        };
        return f;
    };
    auto same_but_basis = [](const hzr::HeckeStructureRecord& a, const hzr::HeckeStructureRecord& b, const std::string& key) {
        const auto ma = Hashed(a), mb = Hashed(b);
        std::set<std::string> paths;
        for (const auto& [p, v] : ma) paths.insert(p);
        for (const auto& [p, v] : mb) paths.insert(p);
        uint32_t body = 0, cert = 0;
        for (const auto& p : paths) {
            const auto ia = ma.find(p), ib = mb.find(p);
            const std::string va = ia == ma.end() ? "<absent>" : ia->second, vb = ib == mb.end() ? "<absent>" : ib->second;
            if (va == vb) continue;
            const bool basis = BasisDependent(p);
            if (StartsWith(p, "G9")) cert += !basis;
            else ++body;
            std::cout << "    " << key << (basis ? " basis-dependent " : " DIFF ") << p << ": " << va << " -> " << vb << "\n";
        }
        return std::make_pair(body, cert);
    };
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::string why;
        // (i) the root trigger: (1, 52) @7, final, C_T certified at K0 = 3, K3 present
        hzr::HeckeStructureRecord U, F;
        const bool b1 = !Caught([&] { U = hzr::BuildHeckeStructureRecord(1, 52, "full", S, src); }, &why);
        const bool b2 = !Caught([&] { F = hzr::BuildHeckeStructureRecord(1, 52, "full", S, forged(7, 2, nullptr, true)); }, &why);
        const Triggers t = TriggersOf(F);
        const auto d = same_but_basis(U, F, "(H)(i) (1,52)@7");
        std::cout << "  (H)(i) (1,52) @7: forged exw=" << Val(F, "G10.read.ell=7.exw") << " C_T.ell=7=" << Val(F, "G3.C_T.ell=7.status")
                  << " unforged record_fnv1a64=" << U.record_fnv1a64 << " forged=" << F.record_fnv1a64 << "\n";
        T.check(b1 && b2 && t.root.count("7") && U.record_fnv1a64 == F.record_fnv1a64 && d.first == 0 && d.second == 0 &&
                    Val(F, "G3.C_T.ell=7.status") == "certified" && Val(F, "G9.K3.ell=7.K") == Val(U, "G9.K3.ell=7.K"),
                "(H)(i) (1,52) @7: the root trigger on a forged export refusal gives the unforged record's hashed body; the cert differs only "
                "in the basis-dependent values (" + why + ")");
        // (iii) origin (b): (1, 52) @7 served at K0 but its escalation forged to stop at the export's K_max -> the root trigger; the (C3) path keeps
        // the served module (G2, G9.K1, tb_mat byte for byte), so the cert differs from the unforged one only in G9.K3's basis-dependent values
        {
            hzr::HeckeStructureRecord F3;
            const bool b3 = !Caught([&] { F3 = hzr::BuildHeckeStructureRecord(1, 52, "full", S, forged(7, 0, nullptr, true)); }, &why);
            const auto d3 = same_but_basis(U, F3, "(H)(iii) (1,52)@7");
            std::cout << "  (H)(iii) (1,52) @7 origin (b): exw=" << Val(F3, "G10.read.ell=7.exw") << " tb_mat equal="
                      << (Val(F3, "G9.trust.tb_mat") == Val(U, "G9.trust.tb_mat")) << "\n";
            T.check(b3 && TriggersOf(F3).root.count("7") && U.record_fnv1a64 == F3.record_fnv1a64 && d3.first == 0 && d3.second == 0 &&
                        Val(F3, "G9.trust.tb_mat") == Val(U, "G9.trust.tb_mat"),
                    "(H)(iii) (1,52) @7: a served read stopped at the export's K_max -> the root trigger; hashed body equal, the (C3) module kept "
                    "(tb_mat equal), cert differing only in G9.K3's basis-dependent values (" + why + ")");
        }
        // (ii) the (C3) trigger on the full loop at N > 1: (5, 4) full, at an ell of S with generation by saturation and no rational theta
        hzr::HeckeStructureRecord U2;
        Caught([&] { U2 = hzr::BuildHeckeStructureRecord(5, 4, "full", S, src); }, &why);
        uint32_t found = 0;
        for (const auto& f : U2.fields) {
            const std::string suf = ".generation.route";
            if (f.group != "G2" || f.value != "saturation" || f.path.size() < suf.size() ||
                f.path.compare(f.path.size() - suf.size(), suf.size(), suf) != 0)
                continue;
            const uint64_t ell = std::stoull(f.path.substr(7, f.path.size() - 7 - suf.size()));
            if (found >= 2) break;
            if (ell != 5 && ell != 11) continue;   // (5, 4, full): 5 | N (U_5, in P's read set), 11 (T_11 appended by the K0 read)
            hzr::HeckeStructureRecord F2;
            if (Caught([&] { F2 = hzr::BuildHeckeStructureRecord(5, 4, "full", S, forged(ell, 1, nullptr, true)); }, &why)) {
                T.check(false, "(H)(ii) (5,4,full) @" + std::to_string(ell) + ": built: " + why);
                continue;
            }
            const Triggers t2 = TriggersOf(F2);
            if (!t2.c3.count(std::to_string(ell))) {
                std::cout << "  (H)(ii) (5,4,full) @" << ell << ": no (C3) trigger (generation certified by another route once forged: "
                          << Val(F2, "G2.ell=" + std::to_string(ell) + ".generation.status") << ")\n";
                continue;
            }
            ++found;
            // the forging refuses the K0 read as well (else (C3) comes from it and nothing triggers), so G0.operators loses the T_ell a served K0
            // read appends (t_ell = added) -- a consequence of the forging, not of EXW; every other field must be the unforged record's
            std::string ops_u = Val(U2, "G0.operators") + ",", ops_f = Val(F2, "G0.operators") + ",";
            const std::string tl = "T_" + std::to_string(ell) + ",";
            if (ops_u.find(tl) != std::string::npos && ops_f.find(tl) == std::string::npos) ops_u.erase(ops_u.find(tl), tl.size());
            hzr::HeckeStructureRecord F2n = F2;
            for (auto& f : F2n.fields)
                if (f.path == "G0.operators") f.value = Val(U2, "G0.operators");
            const auto d2 = same_but_basis(U2, F2n, "(H)(ii) (5,4,full)@" + std::to_string(ell));
            std::cout << "  (H)(ii) (5,4,full) @" << ell << ": forged exw=" << Val(F2, "G10.read_full.ell=" + std::to_string(ell) + ".exw")
                      << " operators " << Val(U2, "G0.operators") << " -> " << Val(F2, "G0.operators") << "\n";
            T.check(ops_u == ops_f && d2.first == 0 && d2.second == 0 &&
                        Val(F2, "G2.ell=" + std::to_string(ell) + ".generation.route") == "saturation",
                    "(H)(ii) (5,4,full) @" + std::to_string(ell) + ": the (C3) trigger on the full loop gives the unforged record (every hashed "
                    "field, G0.operators up to the forged K0 read's T_ell)");
        }
        T.check(found >= 1, "(H)(ii) (5,4,full): at least one ell with a (C3) trigger (generation by saturation, no rational theta)");
        std::cout << "  (H) secs=" << Since(t0) << "\n";
    }

    // ---- (I) the negative case of (alpha): a refusal that stands in no field triggers nothing --------------------------------------------
    {
        const auto t0 = std::chrono::steady_clock::now();
        struct ICase {
            uint32_t N, n;
            uint64_t ell;
            uint32_t from_K;
            const char* what;
        };
        const std::vector<ICase> cases = {{1, 22, 3, 1, "(1,22) @3: generation eis_single, a forged (C3) refusal at every K"},
                                          {1, 40, 7, 2, "(1,40) @7: 7 !| g_P (C_T not listed), a forged export K_max refusal at K >= 2"}};
        for (const auto& c : cases) {
            ++negative_n;
            std::string why;
            auto stats = std::make_shared<hzr::FreeSourceStats>();
            hzr::HeckeStructureRecord A, B;
            const bool ba = !Caught([&] { A = hzr::BuildHeckeStructureRecord(c.N, c.n, "full", S, forged(c.ell, c.from_K, stats, true)); }, &why);
            const bool bb = !Caught([&] { B = hzr::BuildHeckeStructureRecord(c.N, c.n, "full", S, forged(c.ell, c.from_K, nullptr, false)); }, &why);
            // provenance equal except the timings
            const auto pa = Prov(A), pb = Prov(B);
            // the forged refusal is in the record (G10), standing in no hashed field
            const bool forged_word = Val(A, "G10.read.ell=" + std::to_string(c.ell) + ".module_status").find(c.from_K >= 2 ? "K_max" : "ModulePrecision") !=
                                         std::string::npos ||
                                     Val(A, "G10.read.ell=" + std::to_string(c.ell) + ".c3_fallback_K1").find("ModulePrecision") != std::string::npos;
            const bool ok = ba && bb && stats->calls == 0 && A.record_fnv1a64 == B.record_fnv1a64 && A.cert_sha256 == B.cert_sha256 &&
                            Hashed(A) == Hashed(B) && pa == pb && forged_word;
            std::cout << "  (I) " << c.what << ": calls=" << stats->calls << " status=" << A.status << " record_fnv1a64 " << A.record_fnv1a64
                      << " / " << B.record_fnv1a64 << "\n";
            negative += ok;
            T.check(ok, std::string("(I) ") + c.what + ": no call of the free source, the two records byte-identical (" + why + ")");
        }
        std::cout << "  (I) secs=" << Since(t0) << "\n";
    }

    const bool ok = T.failing == 0 && identity == identity_n && negative == negative_n;
    std::cout << "HECKEZ_EXW arms=11 failing=" << T.failing << " identity=" << identity << "/" << identity_n << " pairs=" << pairs << " moved=" << moved
              << " negative=" << negative << "/" << negative_n << " checks=" << T.checks << " secs=" << Since(t_all)
              << " verdict=" << (ok ? "OK" : "FAIL") << "\n";
    return ok ? 0u : 1u;
}
