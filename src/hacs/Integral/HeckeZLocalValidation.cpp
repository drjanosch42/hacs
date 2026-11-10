// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZLocalValidation, HeckeZLocalCensusXcheck -- slot-25 HZ-LOC (see the header).

#include "HeckeZLocalValidation.h"

#include "HeckeZLocal.h"
#include "HeckeZRational.h"
#include "../HeckeExt/HeckeZLocalSource.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/Rss.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace heckez::local;

struct Tally {
    uint32_t checks = 0, failing = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failing;
            std::cout << "  [MISMATCH] " << what << "\n";
        }
    }
};

double Since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int32_t ValIn(const heckez::Factored& F, uint64_t ell) {
    for (const auto& pp : F.factors)
        if (pp.q == ell && pp.proof != "probable") return static_cast<int32_t>(pp.e);
    return F.complete() ? 0 : -1;
}

RationalInput RatFor(const heckez::RationalResult& R, uint64_t ell) {
    RationalInput ri;
    if (R.theta_index < 0 || !R.eis_rational) return ri;
    const auto& t = R.thetas[static_cast<size_t>(R.theta_index)];
    ri.present = true;
    ri.theta = t.coeffs;
    ri.lambda_hex = t.lambda_hex;
    ri.c_hex = R.cusp_part;
    ri.v_c_lambda = ValIn(t.eis_res, ell);
    ri.v_gP = ValIn(R.gP, ell);
    ri.gP_complete = R.gP.complete();
    return ri;
}

struct CellRun {
    LatticeModule M;
    LocalRead L;
    bool read = false;
    std::string escalation;
    double secs = 0;
};

uint32_t g_torsion_certified = 0;   // torsion_prime_certified_from_H1
uint32_t g_cells = 0;
uint64_t g_digest = engine::kFnvOffset64;

// one (N, n, ell) read; K raised while the at-root read is open(precision) (at most 3 steps, within the word)
CellRun ReadCell(uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P,
                 const heckez::RationalResult* rat, LatticeModule* h1_side = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t k = n + 2;
    const int64_t dM = heckez::DimMk(k, N), dS = heckez::DimSk(k, N);
    const int64_t e_k = dM - dS;
    const uint32_t b1 = static_cast<uint32_t>(2 * dM - e_k);
    ReadOptions o;
    o.d_target = N == 1 ? dM : heckez::DimAnemic(k, N);
    o.r_eis = static_cast<uint32_t>(e_k);
    o.eis_rational = heckez::EisAnemicCount(k, N) == 1;
    o.k_weight = k;
    o.full_algebra = N == 1;
    if (rat) o.rat = RatFor(*rat, ell);
    CellRun C;
    {
        const CellLocalRead R = ReadCellLocal(N, n, ell, K, P, b1, o, 3, h1_side);
        C.M = R.M;
        C.L = R.L;
        C.read = R.read;
        C.escalation = R.escalation;
    }
    C.secs = Since(t0);
    ++g_cells;
    if (C.read) {
        if (!C.L.is_lattice && C.L.certified_any()) ++g_torsion_certified;
        std::cout << "  READ " << C.L.text() << (C.escalation.empty() ? "" : " escalation=" + C.escalation) << "    secs=" << C.secs
                  << "\n";
        g_digest = engine::FnvBytes(g_digest, C.L.text());
    } else {
        std::cout << "  READ (" << N << "," << n << ") ell=" << ell << " K=" << K << " " << C.M.status << " source=" << C.M.source
                  << " (" << C.M.detail << ")    secs=" << C.secs << "\n";
        g_digest = engine::FnvBytes(g_digest, C.M.status);
    }
    return C;
}

heckez::RationalResult Rat(uint32_t N, uint32_t n, const std::vector<uint32_t>& P) {
    heckez::RationalSettings s;
    s.threads = 4;
    return heckez::RationalStep(N, n, P, s);
}

// ---- (I) the Z/ell^K layer on random spans --------------------------------------------------------------------------------------
uint32_t SelfTest(Tally& T) {
    uint64_t seed = 0x9e3779b97f4a7c15ull;
    auto rnd = [&]() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    uint32_t tests = 0;
    const std::vector<std::pair<uint64_t, uint32_t>> rings = {{7, 3}, {2, 5}, {691, 2}, {2294797, 2}, {1798482437, 2}};
    for (const auto& rk : rings) {
        const Zlk R = Zlk::Make(rk.first, rk.second);
        for (int t = 0; t < 6; ++t) {
            const size_t m = 5 + rnd() % 4, g = 2 + rnd() % 5;
            Span S(R, m);
            std::vector<std::vector<uint64_t>> gens;
            for (size_t i = 0; i < g; ++i) {
                std::vector<uint64_t> v(m);
                // entries with valuations spread over [0, K]
                for (size_t j = 0; j < m; ++j) v[j] = R.mul(rnd() % R.q, R.pw(static_cast<uint32_t>(rnd() % (R.K + 1))));
                gens.push_back(v);
                S.insert(v);
            }
            bool member = true;
            for (const auto& v : gens) member = member && S.contains(v);
            // a sum of generators is a member; a random vector of full support usually is not (no check on it)
            std::vector<uint64_t> sum(m, 0);
            for (const auto& v : gens)
                for (size_t j = 0; j < m; ++j) sum[j] = R.add(sum[j], R.mul(v[j], 3));
            member = member && S.contains(sum);
            std::vector<uint64_t> A;
            for (const auto& v : gens) A.insert(A.end(), v.begin(), v.end());
            uint32_t Ls = 0;
            for (uint32_t s : SmithValuations(R, A, g, m)) Ls += R.K - s;
            T.check(member && S.length() == Ls && S.length_by_smith() == Ls,
                    "self-test Z/" + std::to_string(rk.first) + "^" + std::to_string(rk.second) + " #" + std::to_string(t) +
                        ": membership and Howell length " + std::to_string(S.length()) + " = Smith length " + std::to_string(Ls));
            ++tests;
        }
    }
    // the polynomial layer: Charpoly / DivideByRoot / MonicSqrt on diag(l, a, a, b, b) mod 691^2
    {
        const Zlk R = Zlk::Make(691, 2);
        const size_t D = 5;
        Mat A(D * D, 0);
        const uint64_t l = 2049, a = R.neg(24), b = 7;
        A[0] = l;
        A[6] = a;
        A[12] = a;
        A[18] = b;
        A[24] = b;
        A[1] = 5;   // an off-diagonal entry (upper triangular: the characteristic polynomial unchanged)
        Poly chi = Charpoly(R, A, D), q, c;
        const bool ok = DivideByRoot(R, chi, l, 1, &q) && MonicSqrt(R, q, &c) && c.size() == 3 &&
                        c[0] == R.mul(a, b) && c[1] == R.neg(R.add(a, b)) && c[2] == 1;
        T.check(ok, "self-test: Charpoly, DivideByRoot, MonicSqrt on diag(2049, -24, -24, 7, 7) mod 691^2");
        T.check(PolyEval(R, c, l) == R.mul(R.sub(l, a), R.sub(l, b)), "self-test: c(lambda) = (lambda - a)(lambda - b)");
        T.check(HexResidue(R, "-0x18") == a && HexResidue(R, "0x801") == l, "self-test: HexResidue");
        ++tests;
    }
    return tests;
}

}  // namespace

uint32_t HeckeZLocalValidation() {
    std::cout << "============ HeckeZLocalValidation (slot-25 HZ-LOC) ============\n";
    Tally T;
    g_torsion_certified = 0;
    g_cells = 0;
    g_digest = engine::kFnvOffset64;
    const auto t_all = std::chrono::steady_clock::now();
    uint32_t u64_named = 0, u64_closed = 0;
    try {
        // ---- (I)
        {
            const uint32_t n = SelfTest(T);
            std::cout << "  SELFTEST spans=" << n << "\n";
        }
        // ---- (A) T1 (1, 10) at 691
        uint64_t t1_digest = 0;
        {
            const auto R = Rat(1, 10, {2, 3, 5, 7});
            const CellRun C = ReadCell(1, 10, 691, 2, {2, 3, 5, 7}, &R);
            const LocalRead& L = C.L;
            t1_digest = L.digest();
            T.check(C.read && L.source == "h1_level1" && L.is_lattice && L.D == 3 && L.d == 2, "T1: the H^1 route is L/691^2, D = 3, d = 2");
            T.check(L.dim_A1 == 2 && L.c3 == "certified", "T1: dim A'_1(L/691) = 2 = d, (C3) certified");
            T.check(L.mono == "theta" && L.e == 0, "T1: theta-bar = T_2-bar generates, e = 0 (stopping rule at K = 1)");
            T.check(L.p26 == "eis+cusp", "T1: both components pass through the Eisenstein block (P 2.6)");
            T.check(L.root == "closes" && L.v_eta == 1 && L.v_gP == 1 && L.v_gP_local == 1 && L.eta == std::vector<uint32_t>{1},
                    "T1: the at-root read closes: v_691(exp eta_L) = 1 = v_691(g_P), eta_L = Z/691");
            bool all_a0 = !L.roots.empty();
            for (const auto& r : L.roots) all_a0 = all_a0 && r.status == "ok" && r.a == std::vector<uint32_t>{0} && r.v_c_lambda == 1;
            T.check(all_a0, "T1: Smith of c(T) mod 691^2 has exponent [0] for theta, T_2, T_3, T_5, T_7 (independent of p)");
            T.check(L.C_T == "691^1_exact", "T1: the 691-part of C_T exact = 691");
            // determinism
            const CellRun C2 = ReadCell(1, 10, 691, 2, {2, 3, 5, 7}, &R);
            T.check(C2.L.digest() == t1_digest, "T1: the read is deterministic (digest equal on a re-run)");
        }
        // ---- (B) T2 (1, 22) at 103 and 2294797
        {
            const auto R = Rat(1, 22, {2, 3, 5, 7});
            for (uint64_t ell : {uint64_t(103), uint64_t(2294797)}) {
                const CellRun C = ReadCell(1, 22, ell, 2, {2, 3, 5, 7}, &R);
                const LocalRead& L = C.L;
                const std::string tag = "T2 at " + std::to_string(ell) + ": ";
                T.check(C.read && L.is_lattice && L.D == 5 && L.dim_A1 == 3 && L.c3 == "certified", tag + "dim A'_1 = 3 = d, certified");
                T.check(L.mono == "theta" && L.e == 0, tag + "theta-bar generates, e = 0");
                T.check(L.root == "closes" && L.v_eta == 1 && L.v_gP == 1 && L.v_gP_local == 1, tag + "the at-root read closes, v = 1");
                T.check(L.t_ell == (ell <= kTEllCap ? "added" : "omitted(cap:ell>23170)"), tag + "T_ell in the read set iff ell <= 23170 (F1)");
                if (ell == 2294797) {
                    // (audit F3) the word RunModule ran on, not the ModCtx regime name (Zlk is ModCtx<uint64_t> at every ell)
                    T.check(L.word == "u64", tag + "the module route ran RunModuleT<uint64_t> (2294797^2 > 2^32)");
                    ++u64_named;
                    if (L.root == "closes" && L.word == "u64") ++u64_closed;
                }
            }
        }
        // ---- (C) the u64 item: every certified prime >= 5 of g_P at (1, 24), (1, 26), (1, 28), (1, 30), (1, 42).
        // (audit F3) the named primes are pinned by value (card 06 HZ-LOC: T2's 2294797 in (B) and these six); every other
        // prime with ell^2 > 2^31 is a mismatch, and each named read must run RunModuleT<uint64_t> (word u64).
        const std::map<uint32_t, std::vector<uint64_t>> kNamed = {
            {24, {657931}}, {26, {362903}}, {28, {1001259881}}, {30, {305065927}}, {42, {2947939, 1798482437}}};
        for (uint32_t n : {24u, 26u, 28u, 30u, 42u}) {
            const auto R = Rat(1, n, {2, 3, 5, 7});
            T.check(R.gP_status == "upper_bound_certified", "u64 (1," + std::to_string(n) + "): g_P certified by HZ-RAT");
            std::cout << "  GP (1," << n << ") g_P=" << heckez::FactoredText(R.gP) << " status=" << R.gP_status << "\n";
            const std::vector<uint64_t>& want = kNamed.at(n);
            uint32_t seen = 0;
            for (const auto& pp : R.gP.factors) {
                if (pp.proof == "probable" || pp.q < 5) continue;
                const uint32_t K = pp.e + 1;
                const bool big = static_cast<unsigned __int128>(pp.q) * pp.q > (static_cast<unsigned __int128>(1) << 31);
                const bool named = std::find(want.begin(), want.end(), pp.q) != want.end();
                const CellRun C = ReadCell(1, n, pp.q, K, {2, 3, 5, 7}, &R);
                const std::string tag = "u64 (1," + std::to_string(n) + ") at " + std::to_string(pp.q) + ": ";
                T.check(C.read && C.L.root != "inconsistent", tag + "read, consistent across operators");
                T.check(!big || named, tag + "ell^2 > 2^31 but not one of the seven named primes (F3)");
                if (named) {
                    ++u64_named;
                    ++seen;
                    T.check(C.read && C.L.word == "u64", tag + "the module route ran RunModuleT<uint64_t>");
                    T.check(C.read && C.L.c3 == "certified", tag + "(C3) certified");
                    T.check(C.read && C.L.root == "closes" && C.L.v_gP == 1 && C.L.v_eta == 1, tag + "the at-root read closes, v = 1");
                    if (C.read && C.L.root == "closes" && C.L.word == "u64") ++u64_closed;
                }
                if (n == 42 && pp.q == 11) {
                    // (audit F1, F4) with T_11 in the read set: dim A'_1 = 4 = d, v_11 c_11(lambda_11) = 0, so v_11(g_{P u {11}}) = 0;
                    // eta_L 11-part trivial: the 11-part of C_T is exactly 1
                    T.check(C.read && C.L.t_ell == "added" && C.L.dim_A1 == 4 && C.L.c3 == "certified",
                            tag + "T_11 in the read set: dim A'_1(L/11) = 4 = d, (C3) certified");
                    T.check(C.read && C.L.v_gP == 0 && C.L.v_gP_source == "hz_rat+T_ell" && C.L.v_eta == 0 && C.L.eta.empty(),
                            tag + "v_11(g_{P u {11}}) = 0 (HZ-RAT's 1 lowered by T_11), eta_L 11-part trivial");
                    T.check(C.read && C.L.root == "closes" && C.L.C_T == "11^0_exact", tag + "closes: the 11-part of C_T is exactly 1");
                }
            }
            T.check(seen == want.size(), "u64 (1," + std::to_string(n) + "): every named prime is a certified factor of g_P");
        }
        // ---- (D) (1, 246) at 37, local only (no HZ-RAT at k = 248)
        {
            const CellRun C = ReadCell(1, 246, 37, 2, {2, 3, 5, 7}, nullptr);
            const LocalRead& L = C.L;
            T.check(C.read && L.is_lattice && L.d == 21 && L.D == 41, "(1,246) at 37: the lattice L/37^K, D = 41, d = 21");
            uint32_t ok_reads = 0;
            for (const auto& r : L.roots) ok_reads += r.status == "ok" ? 1 : 0;
            T.check(L.root != "inconsistent" && ok_reads >= 2 && L.v_gP_local >= 1,
                    "(1,246) at 37: >= 2 at-root reads agree, 37 | g_P (the Kummer pair (37, 32): k = 248 = 32 + 6 * 36)");
            // (audit F4) pinned: closes with v_37(g_P) = 1 (auditor, exact over Z)
            T.check(L.c3 == "certified" && L.root == "closes" && L.v_gP == 1 && L.v_eta == 1 && L.C_T == "37^1_exact",
                    "(1,246) at 37: (C3) certified, closes, v_37(g_P) = 1, C_T 37-part exact");
            T.check(C.secs < 60.0, "(1,246) at 37: within the card's budget scale (< 60 s; the card's 20 s is for the matrices)");
        }
        // ---- (E) T3 (11, 0) at 5 on the export
        {
            const auto R = Rat(11, 0, {2, 3, 5, 7});
            const CellRun C = ReadCell(11, 0, 5, 2, {2, 3, 7}, nullptr);
            // the HZ-RAT theta lives on P = {2, 3, 5, 7}; at ell = 5 the export takes T_p, p != 5: read locally, compare g_P
            const LocalRead& L = C.L;
            T.check(C.read && L.source == "export" && L.D == 3 && L.d == 2, "T3: the export's L/5^2, D = 3, d^an = 2");
            T.check(L.dim_A1 == 2 && L.c3 == "certified", "T3: dim A'_1(L/5) = 2 = d^an, (C3) certified at 5");
            T.check(L.root == "closes" && L.v_eta == 1 && L.v_gP_local == 1 && ValIn(R.gP, 5) == 1,
                    "T3: eta_L = Z/5, the read closes; v_5(g_P) = 1 (HZ-RAT and local)");
        }
        // ---- (F) T4 (2, 6), anemic, rank L_S = 2: at 17 and at 2 (K >= 8)
        {
            const auto R = Rat(2, 6, {3, 5, 7, 11});
            for (const auto& ek : std::vector<std::pair<uint64_t, uint32_t>>{{17, 2}, {2, 9}}) {
                const CellRun C = ReadCell(2, 6, ek.first, ek.second, {3, 5, 7, 11}, &R);
                const LocalRead& L = C.L;
                const std::string tag = "T4 at " + std::to_string(ek.first) + ": ";
                T.check(C.read && L.source == "export" && L.D == 4 && L.d == 2, tag + "the export's L, D = 4, d^an = 2");
                uint32_t ok_reads = 0;
                bool rank2 = true;
                for (const auto& r : L.roots)
                    if (r.status == "ok") {
                        ++ok_reads;
                        rank2 = rank2 && r.a.size() == 2;
                    }
                T.check(ok_reads >= 1 && rank2 && L.root != "inconsistent", tag + "rank-2 at-root reads (r = 2), consistent");
                T.check(L.root == "closes" || L.root == "upper_bound_certified", tag + "a final certificate status");
                T.check(ek.first != 2 || L.K >= 8, tag + "the 2-part read at K >= 8");
                // (audit F4) pinned values (auditor, exact over Z on both integral forms)
                if (ek.first == 17) {
                    bool a01 = ok_reads >= 1;
                    for (const auto& r : L.roots)
                        if (r.status == "ok") a01 = a01 && r.a == std::vector<uint32_t>{0, 1};
                    T.check(a01 && L.eta == std::vector<uint32_t>{1} && L.root == "closes" && L.v_gP == 1,
                            tag + "a = [0, 1] for every read operator, eta_L = Z/17, closes, v_17(g_P) = 1");
                    T.check(L.t_ell == "added", tag + "T_17 in the read set (F1)");
                } else {
                    bool th = false;
                    for (const auto& r : L.roots)
                        if (r.op == "theta") th = r.status == "ok" && r.a == std::vector<uint32_t>{7, 7} && r.v_c_lambda == 7;
                    T.check(th && L.eta.empty() && L.v_eta == 0 && L.v_gP == 7 && L.root == "upper_bound_certified",
                            tag + "theta: a = [7, 7], v_2 c_theta(lambda) = 7; eta_L 2-part trivial; v_2(g_P) = 7; upper_bound_certified");
                    T.check(L.t_ell == "omitted(ell|N:U_ell_not_exported)", tag + "2 | N: no U_2 in the read set (F1, stated)");
                }
            }
        }
        // ---- (G) the torsion-prime routing at level 1: (1, 40) at 7 (H^1 has a Z/7, descent groups_inv)
        {
            LatticeModule h1;
            const CellRun C = ReadCell(1, 40, 7, 2, {2, 3, 5}, nullptr, &h1);
            const int64_t d34 = heckez::DimMk(42, 1);
            T.check(h1.ok && !h1.is_lattice && h1.h1_rank > 2 * d34 - 1, "(1,40) at 7: the H^1 module has 7-torsion (rank > b1): not L/7^K");
            if (h1.ok) {
                ReadOptions o;
                o.d_target = d34;
                o.k_weight = 42;
                const LocalRead H = ReadLocal(h1, o);
                std::cout << "  READ " << H.text() << "\n";
                if (!H.is_lattice && H.certified_any()) ++g_torsion_certified;
                T.check(H.c3 == "A1_of_computed" && H.root == "A1_of_computed" && !H.certified_any(),
                        "(1,40) at 7: the H^1 read is A1_of_computed only, nothing certified");
            }
            T.check(C.read && C.L.source == "export" && C.L.is_lattice && C.L.D == 2 * d34 - 1,
                    "(1,40) at 7: the export's free block read instead (L/7^K, D = b1 = 7)");
        }
        // ---- (G') (audit F5) the torsion-prime routing at a second cell: (1, 24) at 7, 11, 23 (H^1 torsion 2^4 3^2 5 7 11 23)
        {
            const int64_t d26 = heckez::DimMk(26, 1);
            for (uint64_t ell : {uint64_t(7), uint64_t(11), uint64_t(23)}) {
                LatticeModule h1;
                const CellRun C = ReadCell(1, 24, ell, 2, {2, 3, 5, 7}, nullptr, &h1);
                const std::string tag = "(1,24) at " + std::to_string(ell) + ": ";
                T.check(h1.ok && !h1.is_lattice && h1.h1_rank > 2 * d26 - 1, tag + "the H^1 module has ell-torsion (rank > b1): not L/ell^K");
                if (h1.ok) {
                    ReadOptions o;
                    o.d_target = d26;
                    o.k_weight = 26;
                    const LocalRead H = ReadLocal(h1, o);
                    std::cout << "  READ " << H.text() << "\n";
                    if (!H.is_lattice && H.certified_any()) ++g_torsion_certified;
                    T.check(H.c3 == "A1_of_computed" && !H.certified_any(), tag + "the H^1 read is A1_of_computed only");
                }
                T.check(C.read && C.L.source == "export" && C.L.is_lattice && C.L.D == 2 * d26 - 1,
                        tag + "the export's free block read instead (L/ell^K, D = b1)");
            }
        }
        // ---- (F2) (audit F2) the refusal word survives the K-escalation: (2, 22) at 2294797 (>= 46341, export K_max = 1), from
        // K = 1, where every at-root read is open(precision) (2294797 | g_P: the Eisenstein congruence of E_24 at level 2)
        {
            const CellRun C = ReadCell(2, 22, 2294797, 1, {3, 5, 7, 11}, nullptr);
            T.check(C.read && C.L.K == 1 && C.L.root == "refused(export_word)" && C.L.C_T == "refused(export_word)",
                    "escalation: (2,22) at 2294797 from K = 1: the K = 2 step is refused(export_word), and the read says so (not "
                    "open(precision))");
            T.check(C.read && C.L.c3 == "certified" && C.L.t_ell == "omitted(cap:ell>23170)",
                    "escalation: the K = 1 fields survive ((C3) certified); T_ell beyond the cap, stated");
        }
        // ---- (H) refusals and the scope limit
        {
            const CellRun a = ReadCell(1, 10, 4294967311ull, 2, {2, 3, 5, 7}, nullptr);
            T.check(!a.read && a.M.status == "refused(word_modulus)", "refusal: ell = 4294967311, K = 2: ell^2 >= 2^63, refused(word_modulus)");
            const CellRun b = ReadCell(11, 0, 65521, 2, {2, 3, 5, 7}, nullptr);
            T.check(!b.read && b.M.status == "refused(export_word)", "scope: level 11, ell = 65521 >= 46341, K = 2: refused(export_word)");
            const CellRun c = ReadCell(11, 0, 65521, 1, {2, 3, 5, 7}, nullptr);
            T.check(c.read && c.L.c3 == "certified" && c.L.root == "closes" && c.L.v_gP_local == 0,
                    "scope: the (C3) read at K = 1 still runs there (ell does not divide g_P: trivially closed)");
            const CellRun d = ReadCell(1, 40, 3, 2, {2, 5, 7}, nullptr);
            T.check(d.read && d.L.c3 == "refused(by_theory)", "refusal: (C3) at level 1, ell = 3, d = 4: refused(by_theory) (03 B9)");
            // (audit F1) with T_3 in the read set (the export's tree route lifts it): v_3 c_3(lambda_3) = 0, so 3 does not divide
            // g_{P u {3}}; eta_L trivial: the 3-part of C_T is exactly 1 (independently over Z: hz_indep.py 1 40 [2,5,7,3] [3] 1)
            T.check(d.read && d.L.t_ell == "added" && d.L.dim_A1 == 2 && d.L.root == "closes" && d.L.v_gP == 0 && d.L.C_T == "3^0_exact",
                    "(1,40) at 3: T_3 in the read set, v_3(g_{P u {3}}) = 0, closes, C_T 3-part exactly 1");
        }
    } catch (const std::exception& e) {
        std::cout << "HeckeZLocalValidation: EXCEPTION: " << e.what() << std::endl;
        ++T.failing;
    }
    T.check(g_torsion_certified == 0, "no certified status from the H^1 route at a torsion prime");
    std::cout << "  U64_ITEM named=" << u64_named << " closed=" << u64_closed
              << " verdict=" << (u64_named > 0 && u64_named == u64_closed ? "OK" : "FAIL") << "\n";
    std::cout << "  DIGEST 0x" << std::hex << g_digest << std::dec << " secs=" << Since(t_all)
              << " rss_mib=" << oc_util::CurrentRssMiB() << "\n";
    std::cout << "HECKEZ_LOCAL cells=" << g_cells << " mismatched=" << T.failing
              << " torsion_prime_certified_from_H1=" << g_torsion_certified << " verdict=" << (T.failing == 0 ? "OK" : "FAIL") << "\n";
    std::cout << "HeckeZLocalValidation: " << T.checks << " checks, " << T.failing << " failing\n";
    return T.failing;
}

// ---- the slow sibling ---------------------------------------------------------------------------------------------------------
namespace {

const char* kRowsRel = "docs/implementation/slot25/hzloc/census40_level1_saturated.tsv";

bool Exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

std::string FindRoot() {
#ifdef OC_SOURCE_DIR
    {
        const std::string r(OC_SOURCE_DIR);
        if (!r.empty() && Exists(r + "/" + kRowsRel)) return r;
    }
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
            if (Exists(dir + "/" + kRowsRel)) return dir;
        }
    }
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (Exists(dir + "/" + kRowsRel)) return dir;
            const size_t s = dir.find_last_of('/');
            if (s == std::string::npos || s == 0) break;
            dir = dir.substr(0, s);
        }
    }
    return std::string();
}

std::vector<std::string> Split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == c) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    out.push_back(cur);
    return out;
}

std::string Hilb(const std::vector<uint32_t>& h) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < h.size(); ++i) o << (i ? "," : "") << h[i];
    o << "]";
    return o.str();
}

}  // namespace

uint32_t HeckeZLocalCensusXcheck() {
    std::cout << "============ HeckeZLocalCensusXcheck (slot-25 HZ-LOC, slow) ============\n";
    const std::string root = FindRoot();
    if (root.empty()) {
        std::cout << "  [FAIL] " << kRowsRel << " not found (OC_SOURCE_DIR, /proc/self/exe, cwd)\n";
        std::cout << "HECKEZ_LOCAL_XCHECK rows=0 disagree=1 verdict=FAIL\n";
        return 1;
    }
    std::ifstream in(root + "/" + kRowsRel);
    std::string line;
    std::vector<std::string> header;
    struct Row { std::vector<std::string> f; };
    std::map<std::pair<uint32_t, uint64_t>, std::vector<std::vector<std::string>>> groups;
    std::map<std::string, size_t> ix;
    size_t declared = 0, rows = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const size_t p = line.find(" rows=");
            if (p != std::string::npos) declared = std::stoul(line.substr(p + 6));
            continue;
        }
        const auto f = Split(line, '\t');
        if (header.empty()) {
            header = f;
            for (size_t i = 0; i < f.size(); ++i) ix[f[i]] = i;
            continue;
        }
        groups[{static_cast<uint32_t>(std::stoul(f[ix["n"]])), std::stoull(f[ix["ell"]])}].push_back(f);
        ++rows;
    }
    uint32_t disagree = 0, dimE_unchecked = 0, witness_unchecked = 0, D_unchecked = 0, blocks_without_row = 0;
    for (const char* col : {"witness_poly", "module_D", "label_system"})
        if (!ix.count(col)) {
            std::cout << "  [MISMATCH] column " << col << " missing (regenerate with mk_census_rows.py)\n";
            ++disagree;
        }
    // the census witness: residue_witness T_2 (the record's split eigenvalue at p = 2 = P[0]), witness_poly [c, 1] = x + c,
    // so a_2 = -c mod ell; -1 when the row's witness_poly is not of degree 1 (then unchecked, counted)
    auto witness_root = [](const std::string& w, uint64_t ell) -> int64_t {
        if (w.size() < 2 || w.front() != '[' || w.back() != ']') return -1;
        const auto parts = Split(w.substr(1, w.size() - 2), ',');
        if (parts.size() != 2 || parts[1] != "1") return -1;
        const uint64_t c = std::stoull(parts[0]) % ell;
        return static_cast<int64_t>((ell - c) % ell);
    };
    if (rows == 0 || rows != declared) {
        std::cout << "  [MISMATCH] rows read " << rows << " != declared " << declared << "\n";
        ++disagree;
    }
    const auto t_all = std::chrono::steady_clock::now();
    for (const auto& g : groups) {
        const uint32_t n = g.first.first;
        const uint64_t ell = g.first.second;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<uint32_t> P = {2, 3, 5, 7};
        if (std::find(P.begin(), P.end(), ell) == P.end()) P.push_back(static_cast<uint32_t>(ell));
        uint32_t bad = 0;
        try {
            const Level1Blocks B = BlockInvariantsLevel1(n, ell, P);
            std::vector<bool> used(B.blocks.size(), false);
            for (const auto& f : g.second) {
                if (f[ix["dim_E"]] == "not_computed") ++dimE_unchecked;
                const int64_t w = witness_root(f[ix["witness_poly"]], ell);
                if (w < 0) ++witness_unchecked;
                const bool has_D = f[ix["module_D"]] != "not_computed" && !f[ix["module_D"]].empty();
                if (!has_D) ++D_unchecked;
                bool found = false;
                for (size_t i = 0; i < B.blocks.size() && !found; ++i) {
                    if (used[i]) continue;
                    const BlockInvariants& b = B.blocks[i];
                    const bool dimE_ok = f[ix["dim_E"]] == "not_computed" || std::to_string(b.dim_E) == f[ix["dim_E"]];
                    // (audit F6) the system (a_2 against the witness root) and the block rank D = module_D
                    const bool sys_ok = w < 0 || (!b.system.empty() && static_cast<int64_t>(b.system[0] % ell) == w);
                    const bool D_ok = !has_D || std::to_string(b.dim_E) == f[ix["module_D"]];
                    if (f[ix["residue_degree"]] == "1" && dimE_ok && sys_ok && D_ok &&
                        std::to_string(b.length) == f[ix["length_A"]] && Hilb(b.hilbert) == f[ix["hilbert"]] &&
                        std::to_string(b.embdim) == f[ix["embdim"]] && (b.gorenstein ? "true" : "false") == f[ix["gorenstein"]] &&
                        std::to_string(b.loewy()) == f[ix["loewy_A"]]) {
                        used[i] = true;
                        found = true;
                    }
                }
                if (!found) {
                    ++bad;
                    std::cout << "  [DISAGREE] (" << n << "," << ell << ") ideal " << f[ix["ideal_id"]] << ": census dim_E="
                              << f[ix["dim_E"]] << " length_A=" << f[ix["length_A"]] << " hilbert=" << f[ix["hilbert"]]
                              << " embdim=" << f[ix["embdim"]] << " gorenstein=" << f[ix["gorenstein"]] << " module_D=" << f[ix["module_D"]]
                              << " witness_poly=" << f[ix["witness_poly"]] << " has no block of HZ-LOC\n";
                }
            }
            // the other direction, informational: non-reduced HZ-LOC blocks no saturated census row took (the TSV holds the
            // saturated rows only, so an unsaturated census block legitimately has no row here)
            for (size_t i = 0; i < B.blocks.size(); ++i)
                if (!used[i] && B.blocks[i].length >= 2) ++blocks_without_row;
            std::ostringstream bl;
            for (const auto& b : B.blocks)
                if (b.length >= 2) bl << " [E" << b.dim_E << " L" << b.length << " h" << Hilb(b.hilbert) << " soc" << b.socle << "]";
            std::cout << "  XCHECK (" << n << "," << ell << ") census_rows=" << g.second.size() << " h1_rank=" << B.h1_rank
                      << " blocks=" << B.blocks.size() << " nonreduced:" << bl.str() << " disagree=" << bad
                      << " secs=" << Since(t0) << "\n";
        } catch (const std::exception& e) {
            std::cout << "  [EXCEPTION] (" << n << "," << ell << "): " << e.what() << "\n";
            bad += static_cast<uint32_t>(g.second.size());
        }
        disagree += bad;
    }
    std::cout << "  XCHECK secs=" << Since(t_all) << " rss_mib=" << oc_util::CurrentRssMiB() << " cells=" << groups.size() << "\n";
    std::cout << "  XCHECK nonreduced_blocks_without_saturated_row=" << blocks_without_row << " (informational)\n";
    std::cout << "HECKEZ_LOCAL_XCHECK rows=" << rows << " disagree=" << disagree << " unchecked_witness=" << witness_unchecked
              << " unchecked_module_D=" << D_unchecked << " dim_E_not_computed=" << dimE_unchecked
              << " verdict=" << (disagree == 0 ? "OK" : "FAIL") << "\n";
    return disagree;
}
