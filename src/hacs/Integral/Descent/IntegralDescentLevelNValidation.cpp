// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT D1: IntegralDescentLevelNValidation / IntegralDescentLevelNSlowValidation.  See the header for the arms.
// [C] = docs/latex/gl2_contract.tex (v1.23.0-pre.17), cited by label (prop:descentN, prop:descentNthree, prop:badprimesN, prop:levelorbit,
// prop:modeltorsion, prop:certifiedeell).
#include "IntegralDescentLevelNValidation.h"
#include "DescentParAsm.h"

#include "DescentAssembly.h"
#include "DescentLevelN.h"
#include "DescentOracle.h"
#include "DescentTree.h"

#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeLevelNSparse.h"
#include "../../../Engine/EngineRuntime.h"
#include "../../../Engine/ThreadPool.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <tuple>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using integral_descent::AbelianGroupType;
using integral_descent::LevelNResult;
using U32Ctx = engine::ModCtx<uint32_t>;
using clk = std::chrono::steady_clock;

uint64_t g_checks = 0;
uint32_t g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 80) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
std::string gib(uint64_t b) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << double(b) / double(1ull << 30);
    return o.str();
}
std::string fx(double x, int p = 4) {
    std::ostringstream o;
    o << std::setprecision(p) << x;
    return o.str();
}
std::string cellname(uint32_t N, uint32_t n) { return "(N,n)=(" + std::to_string(N) + "," + std::to_string(n) + ")"; }
uint32_t summands(const AbelianGroupType& g, uint32_t ell) {
    const auto it = g.torsion.find(ell);
    return it == g.torsion.end() ? 0u : static_cast<uint32_t>(it->second.size());
}
std::string groups_text(const std::vector<AbelianGroupType>& H) {
    std::ostringstream o;
    for (size_t q = 0; q < H.size(); ++q) o << (q ? " | " : "") << "H^" << q << "=" << H[q].to_string();
    return o.str();
}

// ---- the oracle table -----------------------------------------------------------------
struct OracleN {
    bool found = false;
    std::string error;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<AbelianGroupType>> groups;   // (N, n) -> H^0..H^5
};
std::string oracle_path() {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/docs/implementation/slot24/descentN_oracle.tsv";
#else
    return "docs/implementation/slot24/descentN_oracle.tsv";
#endif
}
OracleN LoadOracleN() {
    OracleN T;
    std::ifstream f(oracle_path());
    if (!f) {
        T.error = "cannot open " + oracle_path();
        return T;
    }
    T.found = true;
    std::string line;
    bool header = false;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!header) {
            header = true;
            if (line.rfind("N\t", 0) == 0) continue;
        }
        std::istringstream is(line);
        uint32_t N = 0, n = 0, q = 0;
        std::string g;
        if (!(is >> N >> n >> q >> g)) {
            T.error = "malformed row: " + line;
            return T;
        }
        auto& v = T.groups[{N, n}];
        if (v.size() != q) {
            T.error = "rows out of order at " + line;
            return T;
        }
        v.push_back(integral_descent::ParseOracleGroup(g));
    }
    return T;
}

uint32_t g_R_cells = 0;
const LevelNResult& descentN(uint32_t N, uint32_t n) {
    // keyed by the route as well (D5 audit m4: D5's (RB) arm reruns this name under HECKE_DESCENT_TREE=0 in the same process)
    static std::map<std::tuple<bool, uint32_t, uint32_t>, LevelNResult> memo;
    const auto key = std::make_tuple(integral_descent::DescentTreeEnabled(), N, n);
    auto it = memo.find(key);
    if (it == memo.end()) it = memo.emplace(key, integral_descent::AssembleDescentN(N, n)).first;
    return it->second;
}

uint32_t oracle_compare(const OracleN& T, uint32_t N, uint32_t n) {
    const auto it = T.groups.find({N, n});
    if (!check(it != T.groups.end() && it->second.size() == 6, "(T) the oracle has " + cellname(N, n) + ", q = 0..5"))
        return 0;
    const LevelNResult& R = descentN(N, n);
    if (!check(!R.refused, "(T) the descent is not refused at " + cellname(N, n) + (R.refused ? " -- " + R.refusal : "")))
        return 0;
    if (n % 2 == 0) {
        ++g_R_cells;
        check(R.good_h0 == R.c.b0 && R.good_h1 == R.c.b1, "(R) closed-form b_0, b_1 == the good prime at " + cellname(N, n));
    }
    uint32_t cells = 0;
    for (uint32_t q = 0; q <= 5; ++q) {
        check(R.H[q] == it->second[q], "(T) " + cellname(N, n) + " q=" + std::to_string(q) + ": descent " + R.H[q].to_string() +
                                           " == oracle " + it->second[q].to_string());
        ++cells;
    }
    return cells;
}

// (B) + (Z) at one cell
void badprime_arm(uint32_t N, uint32_t n) {
    const auto c = integral_descent::LevelNClosedForm(N, n);
    const hecke::level::Gamma0LevelN A(N);
    std::vector<uint32_t> primes;
    for (uint32_t ell = 5; ell <= std::max(n + 2, N); ++ell)
        if (integral_descent::IsPrime(ell) && (ell <= n + 2 || N % ell == 0)) primes.push_back(ell);
    std::ostringstream line;
    line << "  (B,Z) " << cellname(N, n) << ":";
    for (uint32_t ell : primes) {
        const uint32_t pred = integral_descent::PredictedSummands(N, n, ell);
        const auto Q = hecke::level::SparseCokernel::Build(A, n, U32Ctx(ell), false);
        check(Q.h0() == c.b0 + pred, "(B) shipped F_" + std::to_string(ell) + " h^0 - b_0 == prop:badprimesN count " +
                                         std::to_string(pred) + " at " + cellname(N, n) + " (h^0 = " + std::to_string(Q.h0()) + ")");
        const auto r = integral_descent::RunSparseCokernelSmith(N, n, ell);
        check(r.certified && summands(r.tors1, ell) == pred,
              std::string(pred ? "(B)" : "(Z)") + " the Z/ell^K sparse Smith finds " + std::to_string(pred) + " summands at ell=" +
                  std::to_string(ell) + ", " + cellname(N, n));
        line << " " << ell << (N % ell == 0 ? "|N" : "") << ":" << pred << "=" << r.tors1.to_string();
    }
    std::cout << line.str() << std::endl;
}

}  // namespace

uint32_t IntegralDescentLevelNValidation() {
    g_checks = 0;
    g_fail = 0;
    g_R_cells = 0;
    const auto t0 = clk::now();
    std::cout << "=== IntegralDescentLevelNValidation (DESCENT D1: H^q(Gamma_0(N); Sym^n Z^2) by the descent) ===" << std::endl;
    std::cout << "EFFECTIVE HeckeDescentTree route=" << (integral_descent::DescentTreeEnabled() ? "tree" : "dense")
              << " source=" << integral_descent::DescentTreeSource() << " (IntegralDescentLevelNValidation)" << std::endl;
    engine::tape::ThreadPool serial_pool(1);
    engine::tape::ScopedConsumerPool pin_all(&serial_pool);
    try {
        // (T)
        const OracleN T = LoadOracleN();
        if (check(T.found && T.error.empty(), "(T) the oracle table loads: " + (T.error.empty() ? oracle_path() : T.error))) {
            uint32_t cells = 0, ncell = 0;
            for (const auto& kv : T.groups) {
                const auto c = integral_descent::LevelNClosedForm(kv.first.first, kv.first.second);
                if (c.D > 100) continue;
                cells += oracle_compare(T, kv.first.first, kv.first.second);
                ++ncell;
            }
            std::cout << "  (T) " << ncell << " oracle cells (D <= 100; " << T.groups.size() << " in the slow name), " << cells
                      << " groups compared; (R) " << g_R_cells << " good-prime cross-checks" << std::endl;
            const LevelNResult& R = descentN(5, 4);
            if (!R.refused) std::cout << "  (T) " << cellname(5, 4) << ": " << groups_text(R.H) << std::endl;
        }
        // (B), (Z)
        for (const auto& c : std::vector<std::pair<uint32_t, uint32_t>>{{5, 4}, {7, 6}, {11, 10}, {5, 20}, {25, 4}, {1, 30}, {35, 12}})
            badprime_arm(c.first, c.second);
        // (D) three routes at ell >= 5, and SmithLevelsEll == invariant_factors
        for (const auto& c : std::vector<std::array<uint32_t, 3>>{{5, 20, 5}, {11, 10, 11}, {7, 16, 7}, {1, 30, 5}, {1, 56, 7}, {13, 12, 11}}) {
            const uint32_t N = c[0], n = c[1], ell = c[2];
            const uint32_t K = integral_model::DefaultKBudget(ell);
            const auto sp = integral_descent::RunSparseCokernelSmith(N, n, ell, K);
            std::vector<uint32_t> dn = integral_descent::DenseCokernelSmith(N, n, ell, K);
            std::vector<uint32_t> sps;
            for (const auto& [p, ex] : sp.tors1.torsion) sps.insert(sps.end(), ex.begin(), ex.end());
            std::vector<uint32_t> dnz;
            for (uint32_t x : dn)
                if (x) dnz.push_back(x);
            std::sort(sps.begin(), sps.end());
            std::sort(dnz.begin(), dnz.end());
            const U32Ctx ctx = U32Ctx::prime_power(ell, K);
            const auto br = integral_descent::RunBredonDirectSmith(integral_descent::InducedSymModule(N, n, ctx), ell, K,
                                                                   n == 0 ? 1u : 0u, true);
            const auto c2 = integral_descent::LevelNClosedForm(N, n);
            check(sp.certified && dn.size() == sp.r_K && dnz == sps, "(D) sparse == dense-stacked Smith at ell=" + std::to_string(ell) +
                                                                       " " + cellname(N, n));
            check(br.certified && br.d1_onto && br.tors1 == sp.tors1 && br.free_rank1 == c2.b1,
                  "(D) direct Smith of C_eq (x) Z/ell^K == sparse, d^1 onto, rank H^1 == b_1 at ell=" + std::to_string(ell) + " " +
                      cellname(N, n));
            std::cout << "  (D) " << cellname(N, n) << " ell=" << ell << ": " << sp.tors1.to_string() << " (sparse " << fx(sp.seconds)
                      << " s, units " << sp.units << ", residual " << sp.residual_rows << "x" << sp.residual_cols << "; C_eq "
                      << br.c1 << "x" << br.c0 << " " << fx(br.seconds) << " s)" << std::endl;
        }
        {
            std::mt19937 rng(20261025);
            uint32_t agree = 0, tests = 0;
            for (uint32_t ell : {5u, 7u, 3u})
                for (uint32_t t = 0; t < 6; ++t) {
                    const uint32_t K = ell == 3 ? 6 : 4;
                    const U32Ctx C = U32Ctx::prime_power(ell, K);
                    const uint32_t r = 20 + rng() % 30, cc = 15 + rng() % 30;
                    engine::DenseBuf<uint32_t> A(r, cc);
                    // a planted product B1 diag(ell^v) B2 plus noise of high valuation
                    for (uint32_t i = 0; i < r; ++i)
                        for (uint32_t j = 0; j < cc; ++j) {
                            const uint32_t v = (i + j + t) % (K + 1);
                            uint64_t x = rng() % C.p;
                            for (uint32_t s = 0; s < v; ++s) x = x * ell % C.p;
                            A.at(i, j) = static_cast<uint32_t>(x);
                        }
                    std::vector<uint32_t> s = integral_descent::SmithLevelsEll(ell, K, A);
                    std::vector<uint32_t> e = engine::elim::invariant_factors<uint32_t>(C, engine::cview(A));
                    std::vector<uint32_t> s2;
                    for (uint32_t x : e) s2.push_back(K - x);
                    std::sort(s.begin(), s.end());
                    std::sort(s2.begin(), s2.end());
                    ++tests;
                    agree += s == s2;
                }
            check(agree == tests, "(D) SmithLevelsEll == elim::invariant_factors on " + std::to_string(tests) + " test matrices");
            std::cout << "  (D) SmithLevelsEll == elim::invariant_factors: " << agree << "/" << tests << std::endl;
        }
        // (S) the ell = 3 shortcut, [C, prop:descentNthree]: its hypothesis nu_3(N) = 0 <=> 9 | N or a prime p = 2 mod 3 divides N
        {
            uint32_t bad = 0, zero = 0;
            for (uint32_t N = 1; N <= 5000; ++N) {
                bool cond = N % 9 == 0;
                uint32_t m = N;
                for (uint32_t p = 2; p * p <= m; ++p)
                    if (m % p == 0) {
                        cond = cond || p % 3 == 2;
                        while (m % p == 0) m /= p;
                    }
                if (m > 1) cond = cond || m % 3 == 2;
                const bool z = hecke::level::Nu3(N) == 0;
                zero += z;
                bad += z != cond;
            }
            check(bad == 0, "(S) the gate Nu3(N) == 0 is exactly prop:descentNthree's hypothesis (9 | N or p = 2 mod 3 | N), N <= 5000");
            std::cout << "  (S) gate == prop:descentNthree's hypothesis on N <= 5000: " << zero << " levels with nu_3 = 0, " << bad
                      << " disagreeing" << std::endl;
        }
        for (const auto& c : std::vector<std::pair<uint32_t, uint32_t>>{{5, 4}, {11, 6}, {2, 12}, {35, 4}, {8, 6}}) {
            const auto cc = integral_descent::LevelNClosedForm(c.first, c.second);
            if (!check(cc.nu3 == 0, "(S) nu_3 = 0 at " + cellname(c.first, c.second))) continue;
            const LevelNResult& R = descentN(c.first, c.second);
            integral_descent::LevelNOptions o;
            o.force_dense_three = true;
            o.skip_two = true;
            const LevelNResult R3 = integral_descent::AssembleDescentN(c.first, c.second, o);
            // a refused result has an empty R.H (and a piece that did not run an empty tors): read nothing from it (D5 audit m1)
            const bool have = !R.refused && R.H.size() == 6 && R3.ran_three && R3.three.tors.size() > 5;
            bool ok = have && R.three_shortcut && R3.three.all_certified();
            if (ok) {
                ok = R3.three.tors[1].primary_part(3) == R.H[1].primary_part(3);
                for (uint32_t q = 2; q <= 5; ++q) ok = ok && R3.three.tors[q].primary_part(3).is_zero();
            }
            check(ok, "(S) the nu_3 = 0 shortcut == dense Iso^(3) (3-part of H^1; nothing in H^{>=2}) at " + cellname(c.first, c.second) +
                          (R.refused ? " -- " + R.refusal : "") + (R3.refused ? " -- " + R3.refusal : ""));
            if (have)
                std::cout << "  (S) " << cellname(c.first, c.second) << ": 3-part " << R.H[1].primary_part(3).to_string() << " shortcut "
                          << fx(R.t_three) << " s, dense Iso^(3) " << fx(R3.t_three) << " s" << std::endl;
        }
        // (Q) the rank sources
        for (const auto& c : std::vector<std::pair<uint32_t, uint32_t>>{{3, 4}, {7, 2}, {4, 3}}) {
            integral_descent::LevelNOptions o;
            o.rank_from_betti = false;
            o.force_dense_three = true;
            const LevelNResult Ri = integral_descent::AssembleDescentN(c.first, c.second, o);
            o.rank_from_betti = true;
            const LevelNResult Rii = integral_descent::AssembleDescentN(c.first, c.second, o);
            check(!Ri.refused && !Rii.refused && Ri.H == Rii.H, "(Q) Iso^(ell) rank sources (4)(c)(i) == (ii) at " + cellname(c.first, c.second));
        }
        // (L) level 1
        for (uint32_t n : {12u, 30u, 56u}) {
            const LevelNResult& R = descentN(1, n);
            const auto D0 = integral_descent::AssembleDescent(n);
            check(!R.refused && !D0.refused && R.H == D0.H, "(L) AssembleDescentN(1, n) == AssembleDescent(n) (D0) at n=" + std::to_string(n));
            integral_descent::DescentOptions ok;
            ok.bad_route = integral_descent::BadPrimeRoute::KMode;
            const auto DK = integral_descent::AssembleDescent(n, ok);
            check(!DK.refused && DK.H == D0.H, "(L) direct Smith (D1 default) == C_7 K-mode (D0) at n=" + std::to_string(n));
            double tk = 0, td = 0;
            for (const auto& b : DK.bad) tk += b.seconds;
            for (const auto& b : D0.bad) td += b.seconds;
            std::cout << "  (L) n=" << n << ": " << groups_text(D0.H) << "; ell>=5 piece K-mode " << fx(tk) << " s, direct " << fx(td) << " s"
                      << std::endl;
        }
        // (N) negative controls
        {
            const auto r = integral_descent::RunSparseCokernelSmith(5, 20, 5, 2);
            check(!r.certified && r.r_K < r.rank_Q && r.tors1.is_zero(), "(N) (5,20) at Z/5^2 (H^1 has Z/25): not certified, nothing read off");
            integral_descent::LevelNOptions o;
            o.K_ell = 2;
            const LevelNResult R = integral_descent::AssembleDescentN(5, 20, o);
            check(R.refused && R.H.empty() && R.refusal.find("ell=5") != std::string::npos, "(N) the descent at K = 2 is REFUSED, H empty");
            std::cout << "  (N) " << R.refusal << std::endl;
            integral_descent::LevelNOptions o2;
            o2.rss_cap_bytes = 1ull << 20;
            const LevelNResult R2 = integral_descent::AssembleDescentN(7, 4, o2);
            check(R2.refused && R2.H.empty() && !R2.ran_two && R2.refusal.find("RSS") != std::string::npos,
                  "(N) an RSS ceiling below the prediction REFUSES before Iso^(2) starts");
            std::cout << "  (N) " << R2.refusal << std::endl;
            const auto br = integral_descent::RunBredonDirectSmith(
                integral_descent::InducedSymModule(5, 20, U32Ctx::prime_power(5, 2)), 5, 2, 0);
            check(!br.certified && br.tors1.is_zero(), "(N) the direct Smith at Z/5^2 is not certified at (5,20)");
            // (D1 audit m4) the three refusal paths not covered above, and the ell = 3 gate
            integral_descent::LevelNOptions o3;
            o3.test_count_offset = 1;
            const LevelNResult R3 = integral_descent::AssembleDescentN(7, 6, o3);
            check(R3.refused && R3.H.empty() && R3.refusal.find("summands != prop:badprimesN") != std::string::npos,
                  "(N) a summand count off prop:badprimesN REFUSES, H empty");
            std::cout << "  (N) " << R3.refusal << std::endl;
            integral_descent::LevelNOptions o4;
            o4.test_b1_offset = 1;
            const LevelNResult R4 = integral_descent::AssembleDescentN(11, 2, o4);
            check(R4.refused && R4.H.empty() && R4.refusal.find("good prime") != std::string::npos,
                  "(N) a closed form off the good prime REFUSES, H empty");
            std::cout << "  (N) " << R4.refusal << std::endl;
            integral_descent::LevelNOptions o5;
            o5.rss_cap_bytes = 300ull << 20;
            o5.ell_route = integral_descent::EllRoute::DirectBredon;
            o5.skip_two = true;
            const LevelNResult R5 = integral_descent::AssembleDescentN(35, 12, o5);
            check(R5.refused && R5.H.empty() && R5.bredon.empty() && R5.refusal.find("direct Smith RSS") != std::string::npos,
                  "(N) the dense direct Smith above the ceiling is never started: REFUSED");
            const auto rr = integral_descent::RunSparseCokernelSmith(5, 20, 5, 0, 16);
            check(rr.residual_refused && !rr.certified, "(N) a dense residual above its ceiling is never started: not certified");
            bool threw = false;
            try {
                (void)integral_descent::RunSparseCokernelSmith(7, 4, 3);
            } catch (const std::logic_error&) {
                threw = true;
            }
            check(threw, "(N) the ell = 3 sparse route with nu_3(7) = 2 > 0 is refused by its gate (ENGINE_CHECK)");
            std::cout << "  (N) count offset, b_1 offset, direct-Smith ceiling, residual ceiling and the ell = 3 gate: all refuse" << std::endl;
        }
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    std::cout << "IntegralDescentLevelNValidation: checks=" << g_checks << " failing=" << g_fail << " wall=" << fx(since(t0)) << " s"
              << std::endl;
    return g_fail == 0 ? 0u : 1u;
}

// ===========================================================================
// the slow companion
// ===========================================================================
namespace {

struct Stat {
    std::vector<double> v;
    double median() const {
        std::vector<double> w = v;
        std::sort(w.begin(), w.end());
        if (w.empty()) return 0;
        return w.size() % 2 ? w[w.size() / 2] : 0.5 * (w[w.size() / 2 - 1] + w[w.size() / 2]);
    }
    double min() const { return v.empty() ? 0 : *std::min_element(v.begin(), v.end()); }
    double max() const { return v.empty() ? 0 : *std::max_element(v.begin(), v.end()); }
    std::string text() const { return fx(median()) + "(" + fx(min()) + "-" + fx(max()) + ")"; }
};

void l1_cost(uint32_t n, uint32_t repeats) {
    Stat tk, td, ts, tall;
    std::vector<AbelianGroupType> Hk, Hd, Hs;
    const auto dick = integral_model::DicksonPrimes(n);
    for (uint32_t rep = 0; rep < repeats; ++rep)
        for (uint32_t k = 0; k < 4; ++k) {
            const uint32_t route = (k + rep) % 4;
            const auto t = clk::now();
            AbelianGroupType G;
            if (route == 0) {
                for (const auto& [ell, cnt] : dick) {
                    const auto b = integral_descent::RunShippedKMode(n, ell, cnt);
                    for (uint32_t a : b.tors1.torsion.count(ell) ? b.tors1.torsion.at(ell) : std::vector<uint32_t>{}) G.add_torsion(ell, a);
                }
                tk.v.push_back(since(t));
                Hk = {G};
            } else if (route == 1) {
                for (const auto& [ell, cnt] : dick) {
                    const auto b = integral_descent::RunDirectSmithLevel1(n, ell, cnt);
                    for (uint32_t a : b.tors1.torsion.count(ell) ? b.tors1.torsion.at(ell) : std::vector<uint32_t>{}) G.add_torsion(ell, a);
                }
                td.v.push_back(since(t));
                Hd = {G};
            } else if (route == 2) {
                for (const auto& [ell, cnt] : dick) {
                    const auto b = integral_descent::RunSparseCokernelSmith(1, n, ell);
                    for (uint32_t a : b.tors1.torsion.count(ell) ? b.tors1.torsion.at(ell) : std::vector<uint32_t>{}) G.add_torsion(ell, a);
                }
                ts.v.push_back(since(t));
                Hs = {G};
            } else {
                const auto R = integral_descent::AssembleDescent(n);
                tall.v.push_back(R.t_total);
            }
        }
    check(Hk == Hd && Hk == Hs, "(L-cost) the three ell >= 5 routes agree at n=" + std::to_string(n));
    std::cout << "DESCENT1_L1_COST n=" << n << " dickson=" << dick.size() << " tors=" << (Hk.empty() ? "?" : Hk[0].to_string())
              << " t_kmode=" << tk.text() << " t_direct=" << td.text() << " t_sparse=" << ts.text() << " t_descent_d1=" << tall.text()
              << " speedup_direct=" << fx(tk.median() / std::max(1e-9, td.median()))
              << " speedup_sparse=" << fx(tk.median() / std::max(1e-9, ts.median())) << " repeats=" << repeats << std::endl;
}

struct NCell {
    uint32_t N = 0, n = 0;
    std::string pieces = "z32t";
};

void n_cost(const NCell& cell, uint32_t repeats, uint64_t cap) {
    // (D5) DESCENT1_N_COST measures D1's dense design of record: Iso^(2) / Iso^(3) pinned (the default is now the tree route)
    integral_descent::ScopedDescentTree dense_route(false);
    const auto c = integral_descent::LevelNClosedForm(cell.N, cell.n);
    const bool z = cell.pieces.find('z') != std::string::npos, three = cell.pieces.find('3') != std::string::npos,
               two = cell.pieces.find('2') != std::string::npos, tree = cell.pieces.find('t') != std::string::npos;
    // the direct Smith of C_eq(M_N) (x) Z/ell^K is dense (the full coboundary 14D x 5D and the reduced matrix, uint32)
    const uint64_t pbr = 424ull * c.D * c.D + (200ull << 20), ptree = integral_descent::PredictTreeRss(c.D);
    const uint64_t p2 = integral_descent::PredictLocalRss(2, c.D, 4), p3 = integral_descent::PredictLocalRss(3, c.D, 6);
    std::cout << "  DESCENT1_N_PRED N=" << cell.N << " n=" << cell.n << " D=" << c.D << " nu3=" << c.nu3 << " rss_pred_two=" << gib(p2)
              << " GiB rss_pred_three=" << gib(p3) << " GiB rss_pred_direct=" << gib(pbr) << " GiB rss_pred_tree2=" << gib(ptree) << " GiB cap=" << gib(cap)
              << " GiB pieces=" << cell.pieces << std::endl;
    Stat t_sp, t_br, t_good, t_three, t_two, t_tot, t_tree;
    uint64_t rss_two = 0, rss_three = 0, rss_ell = 0, rss_tree = 0;
    std::vector<AbelianGroupType> two_tors, tree_tors;
    std::string refusal, groups;
    bool all_cert = true;
    for (uint32_t rep = 0; rep < repeats; ++rep)
        for (uint32_t k = 0; k < 3; ++k) {
            const uint32_t route = (k + rep) % 3;
            if (route == 0) {
                integral_descent::LevelNOptions o;
                o.rss_cap_bytes = cap;
                o.skip_two = !two;
                o.skip_three = !three;
                if (!z) {
                    // the ell >= 5 piece is cheap; it always runs (its time is reported)
                }
                const LevelNResult R = integral_descent::AssembleDescentN(cell.N, cell.n, o);
                t_sp.v.push_back(R.t_ell);
                t_good.v.push_back(R.t_good);
                t_three.v.push_back(R.t_three);
                t_two.v.push_back(R.t_two);
                t_tot.v.push_back(R.t_total);
                rss_two = std::max(rss_two, R.rss_meas_two);
                rss_three = std::max(rss_three, R.rss_meas_three);
                rss_ell = std::max(rss_ell, R.rss_meas_ell);
                refusal = R.refusal;
                if (!R.refused) groups = groups_text(R.H);
                for (const auto& s : R.sparse) all_cert = all_cert && s.certified;
                if (R.ran_two) {
                    all_cert = all_cert && R.two.all_certified();
                    two_tors = R.two.tors;
                }
                if (R.ran_three) all_cert = all_cert && R.three.all_certified();
                if (R.three_shortcut) all_cert = all_cert && R.three_sparse.certified;
            } else if (route == 2) {
                if (!tree) continue;
                const auto T2 = integral_descent::RunTreeTwoN(cell.N, cell.n, cap);
                if (!T2.ran) continue;
                all_cert = all_cert && T2.certified;
                t_tree.v.push_back(T2.seconds);
                rss_tree = std::max(rss_tree, T2.rss_meas);
                tree_tors = T2.tors;
            } else if (z && c.n % 2 == 0 && pbr <= cap) {
                // the measured alternative at ell >= 5: the direct Smith of C_eq(M_N) (x) Z/ell^K (dense)
                const auto t = clk::now();
                for (const auto& [ell, s] : integral_descent::BadPrimesN(cell.N, cell.n)) {
                    (void)s;
                    const uint32_t K = integral_model::DefaultKBudget(ell);
                    const auto br = integral_descent::RunBredonDirectSmith(
                        integral_descent::InducedSymModule(cell.N, cell.n, U32Ctx::prime_power(ell, K)), ell, K, c.b0);
                    all_cert = all_cert && br.certified;
                }
                t_br.v.push_back(since(t));
            }
        }
    const bool expect_refusal = !two || !three || (two && p2 > cap) || (three && c.nu3 > 0 && p3 > cap);
    check(all_cert, "(N-cost) every piece that ran is certified at " + cellname(cell.N, cell.n));
    if (!two_tors.empty() && !tree_tors.empty()) {
        bool same = true;
        for (uint32_t q = 0; q <= 4; ++q) same = same && two_tors[q].primary_part(2) == tree_tors[q].primary_part(2);
        check(same, "(N-cost) the tree at 2 == Iso^(2) (2-part of H^0..H^4) at " + cellname(cell.N, cell.n));
    }
    if (!expect_refusal) check(refusal.empty(), "(N-cost) the descent is not refused at " + cellname(cell.N, cell.n) + " -- " + refusal);
    std::cout << "DESCENT1_N_COST N=" << cell.N << " n=" << cell.n << " D=" << c.D << " psi=" << c.m << " nu2=" << c.nu2 << " nu3=" << c.nu3
              << " bad=" << integral_descent::BadPrimesN(cell.N, cell.n).size() << " t_good=" << t_good.text() << " t_ell5_sparse=" << t_sp.text()
              << " t_ell5_direct=" << (t_br.v.empty() ? std::string("-") : t_br.text()) << " t_three=" << t_three.text()
              << (c.nu3 == 0 ? "(shortcut)" : "(Iso3)") << " t_two=" << t_two.text() << " t_total=" << t_tot.text()
              << " rss_two=" << gib(rss_two) << "/" << gib(p2) << " rss_three=" << gib(rss_three) << "/" << gib(p3)
              << " rss_ell=" << gib(rss_ell) << " t_tree2=" << (t_tree.v.empty() ? std::string("-") : t_tree.text())
              << " rss_tree2=" << gib(rss_tree) << "/" << gib(ptree) << " repeats=" << repeats
              // (PAR-LN audit A6) with HECKE_DESCENT_PAR_LN on, t_ell5_sparse / rss_ell cover the dense residuals only (the eliminations
              // run in the concurrent stage, LevelNResult::t_ell_elim): not comparable with pre-PAR-LN logs under the same names
              << " ell5_timing=" << (integral_descent::par::Enabled(integral_descent::par::Lever::EllN) ? "residual-only" : "full") << std::endl;
    if (!groups.empty()) std::cout << "  groups " << cellname(cell.N, cell.n) << ": " << groups << std::endl;
    if (!refusal.empty()) std::cout << "  " << refusal << std::endl;
}

}  // namespace

uint32_t IntegralDescentLevelNSlowValidation() {
    g_checks = 0;
    g_fail = 0;
    g_R_cells = 0;
    const auto t0 = clk::now();
    std::cout << "=== IntegralDescentLevelNSlowValidation (DESCENT D1: the oracle on every cell, controlled costs) ===" << std::endl;
    engine::tape::ThreadPool serial_pool(1);   // every route at width 1 (D0 audit M2)
    engine::tape::ScopedConsumerPool pin_all(&serial_pool);
    try {
        uint32_t repeats = 5;
        if (const char* e = std::getenv("HECKE_DESCENT_REPEATS")) repeats = std::max(1, std::atoi(e));
        std::vector<NCell> cells = {{11, 20}, {16, 20}, {35, 10}, {50, 6}};
        std::string arm = "default";
        if (const char* e = std::getenv("HECKE_DESCENT_N_CELLS")) {
            cells.clear();
            arm = "env";
            std::stringstream ss(e);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                NCell c;
                const size_t a = tok.find(':'), b = tok.find(':', a + 1);
                ENGINE_CHECK_MSG(a != std::string::npos, "HECKE_DESCENT_N_CELLS: tokens N:n[:pieces]");
                c.N = static_cast<uint32_t>(std::stoul(tok.substr(0, a)));
                c.n = static_cast<uint32_t>(std::stoul(tok.substr(a + 1, b == std::string::npos ? std::string::npos : b - a - 1)));
                if (b != std::string::npos) c.pieces = tok.substr(b + 1);
                cells.push_back(c);
            }
        }
        uint64_t cap = 40ull << 30;
        std::cout << "EFFECTIVE IntegralDescentLevelNSlowValidation arm=" << arm << " cells=" << cells.size() << " repeats=" << repeats
                  << " rss_cap_gib=" << (cap >> 30) << std::endl;
        // (T) every oracle cell
        const OracleN T = LoadOracleN();
        if (check(T.found && T.error.empty(), "(T) the oracle table loads")) {
            uint32_t cells_T = 0;
            for (const auto& kv : T.groups) cells_T += oracle_compare(T, kv.first.first, kv.first.second);
            std::cout << "  (T) " << T.groups.size() << " oracle cells, " << cells_T << " groups compared; (R) " << g_R_cells
                      << " good-prime cross-checks" << std::endl;
        }
        if (arm == "default")
            for (uint32_t n : {100u, 200u, 300u}) l1_cost(n, repeats);
        for (const auto& c : cells) n_cost(c, repeats, cap);
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    std::cout << "IntegralDescentLevelNSlowValidation: checks=" << g_checks << " failing=" << g_fail << " wall=" << fx(since(t0)) << " s"
              << std::endl;
    return g_fail == 0 ? 0u : 1u;
}
