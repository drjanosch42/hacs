// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT D0: IntegralDescentGroupsValidation / IntegralDescentGroupsSlowValidation.  See the header for the arms.
// [C] = docs/latex/gl2_contract.tex, cited by label (prop:descent, prop:cohglue, prop:certifiede, prop:certifiedeell,
// prop:badprimes, prop:ellthreeclosed, lem:bulkperiodic, rmk:treeconeres).
#include "IntegralDescentGroupsValidation.h"

#include "DescentAssembly.h"
#include "DescentOracle.h"
#include "DescentTree.h"
#include "../../Hecke/HeckeLevelNSparse.h"

#include "../../../Engine/EngineRuntime.h"
#include "../../../Engine/ThreadPool.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <functional>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using integral_descent::AbelianGroupType;
using integral_descent::DescentResult;
using U32Ctx = engine::ModCtx<uint32_t>;
using clk = std::chrono::steady_clock;

uint64_t g_checks = 0;
uint32_t g_fail = 0;
const char* g_name = "IntegralDescentGroupsValidation";
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
std::string vs(const std::vector<uint32_t>& v) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << "]";
    return o.str();
}
AbelianGroupType tors_only(AbelianGroupType g) {
    g.free_rank = 0;
    return g;
}
uint32_t summands(const AbelianGroupType& g, uint32_t ell) {
    const auto it = g.torsion.find(ell);
    return it == g.torsion.end() ? 0u : static_cast<uint32_t>(it->second.size());
}
std::string groups_text(const std::vector<AbelianGroupType>& H) {
    std::ostringstream o;
    for (size_t q = 0; q < H.size(); ++q) o << (q ? " | " : "") << "H^" << q << "=" << H[q].to_string();
    return o.str();
}
std::string oracle_path() {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/docs/implementation/slot24/descent_oracle_n128.tsv";
#else
    return "docs/implementation/slot24/descent_oracle_n128.tsv";
#endif
}

// The descent at n, memoised for the run (several arms read the same n).
const DescentResult& descent(uint32_t n) {
    // keyed by the route as well (D5 audit m4: D5's (RB) arm reruns this name under HECKE_DESCENT_TREE=0 in the same process)
    static std::map<std::pair<bool, uint32_t>, DescentResult> memo;
    const std::pair<bool, uint32_t> key{integral_descent::DescentTreeEnabled(), n};
    auto it = memo.find(key);
    if (it == memo.end()) {
        // (D1 audit m2) the default ell >= 5 route is the direct Smith of C_eq (x) Z/ell^K; verify_d1 tests d^1 onto on the
        // object, so the (cert) arm's "torsion only in H^1" is not vacuous.
        integral_descent::DescentOptions o;
        o.verify_d1 = true;
        it = memo.emplace(key, integral_descent::AssembleDescent(n, o)).first;
    }
    return it->second;
}

uint32_t g_b4_cells = 0, g_b4_ok = 0;
void certified_arm(const DescentResult& R) {
    const std::string at = " at n=" + std::to_string(R.n);
    check(!R.refused, "(cert) the descent is not refused" + at + (R.refused ? " -- " + R.refusal : std::string()));
    if (R.refused) return;
    // (4)(b), counted (D0 audit m1): rank P_0 - rank delta^0 over F_ell' == b_0 on both local pieces
    ++g_b4_cells;
    if (R.two.dimH0_rank_prime == R.rational_dims[0] && R.three.dimH0_rank_prime == R.rational_dims[0]) ++g_b4_ok;
    check(R.two.all_certified(), "(cert) Iso^(2) every d^q certified (ring2k, K=30 or W=61)" + at);
    check(R.three.all_certified(), "(cert) Iso^(3) every d^q certified (ring3k, K=19; else a refusal)" + at);
    for (const auto& b : R.bad) {
        check(b.certified && b.torsion_only_in_H1, "(cert) C_eq direct Smith certified, d^1 onto (torsion only in H^1) at ell=" +
                                                       std::to_string(b.ell) + at);
        check(b.free_rank1 == R.rational_dims[1], "(cert) rank H^1(C_eq(Z)_(ell)) == good-prime rank at ell=" +
                                                      std::to_string(b.ell) + at);
    }
}

// ---- (T) one n against the oracle table ----------------------------------------------
uint32_t oracle_compare(const integral_descent::OracleTable& T, uint32_t n) {
    const auto it = T.groups.find(n);
    if (!check(it != T.groups.end() && it->second.size() >= 6, "(T) the oracle table has n=" + std::to_string(n) +
                                                                   ", q = 0..5"))
        return 0;
    const DescentResult& R = descent(n);
    certified_arm(R);
    if (R.refused) return 0;
    uint32_t cells = 0;
    for (uint32_t q = 0; q <= 5; ++q) {
        check(R.H[q] == it->second[q], "(T) n=" + std::to_string(n) + " q=" + std::to_string(q) + ": descent " +
                                           R.H[q].to_string() + " == oracle " + it->second[q].to_string());
        ++cells;
    }
    return cells;
}

// ---- the cost line (D0 audit M2: controlled timings) ---------------------------------------
// Every route runs on ONE serial consumer pool (engine::tape::ScopedConsumerPool over a 1-worker ThreadPool: the shipped
// C_7 pipeline's executor, the eliminations and run_concurrently all read consumer_pool()), so every route is measured
// at the same width, 1; process CPU seconds are printed beside wall seconds as the check (cpu/wall ~ 1).  `repeats`
// rounds, the route order rotated each round (descent, a, c, c', field); median and min per route.
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
};
std::string fx(double x, int p = 4) {
    std::ostringstream o;
    o << std::setprecision(p) << x;
    return o.str();
}

uint32_t g_repeats = 5;

void cost_cell(uint32_t n) {
    // (D5) DESCENT_COST measures D0's dense design of record: Iso^(2) / Iso^(3) pinned (the default is now the tree route)
    integral_descent::ScopedDescentTree dense_route(false);
    engine::tape::ThreadPool serial(1);
    engine::tape::ScopedConsumerPool pin(&serial);
    const uint32_t good = integral_descent::DefaultGoodPrime(n);
    Stat t_d, t_a, t_c, t_cp, t_f, c_d, c_a, c_c, c_cp, t_good, t_bad, t_two, t_three, t_rank2, t_rank3, t_smith2,
        t_kmode_run;
    uint64_t rss_d = 0, rss_a = 0, rss_c = 0, rss_cp = 0;
    DescentResult R;
    integral_descent::ComparatorRun A, C, Cp;
    uint32_t runs_doubling = 0;
    std::vector<uint32_t> dick;
    for (const auto& pr : integral_model::DicksonPrimes(n)) dick.push_back(pr.first);
    for (uint32_t rep = 0; rep < g_repeats; ++rep) {
        for (uint32_t k = 0; k < 5; ++k) {
            const uint32_t route = (k + rep) % 5;
            if (route == 0) {
                integral_descent::RssSampler rs;
                // (D1) the D0 measurement of record: the shipped C_7 K-mode route, pinned (the default is now the direct Smith)
                integral_descent::DescentOptions d0opt;
                d0opt.bad_route = integral_descent::BadPrimeRoute::KMode;
                R = integral_descent::AssembleDescent(n, d0opt);
                rss_d = std::max(rss_d, rs.stop());
                t_d.v.push_back(R.t_total);
                c_d.v.push_back(R.cpu_total);
                t_good.v.push_back(R.t_good);
                t_bad.v.push_back(R.t_bad);
                t_two.v.push_back(R.t_two);
                t_three.v.push_back(R.t_three);
                t_rank2.v.push_back(R.two.rank_seconds);
                t_rank3.v.push_back(R.three.rank_seconds);
                t_smith2.v.push_back(R.two.smith_seconds);
                runs_doubling = 0;
                for (const auto& b : R.bad) {
                    runs_doubling += static_cast<uint32_t>(b.K_tried.size());
                    for (double s : b.run_seconds) t_kmode_run.v.push_back(s);
                }
            } else if (route == 1) {
                integral_descent::RssSampler rs;
                A = integral_descent::RunComparatorA(n, good);
                rss_a = std::max(rss_a, rs.stop());
                t_a.v.push_back(A.seconds);
                c_a.v.push_back(A.cpu_seconds);
            } else if (route == 2) {
                integral_descent::RssSampler rs;
                C = integral_descent::RunComparatorTree(n, good);
                rss_c = std::max(rss_c, rs.stop());
                t_c.v.push_back(C.seconds);
                c_c.v.push_back(C.cpu_seconds);
            } else if (route == 3) {
                integral_descent::RssSampler rs;
                Cp = integral_descent::RunComparatorTreeBredon(n, good);
                rss_cp = std::max(rss_cp, rs.stop());
                t_cp.v.push_back(Cp.seconds);
                c_cp.v.push_back(Cp.cpu_seconds);
            } else {
                // a plain field run of the shipped pipeline at each Dickson prime (the K-mode overhead's denominator)
                const auto tf = clk::now();
                for (uint32_t ell : dick) (void)integral_descent::ShippedFieldDims(n, ell);
                t_f.v.push_back(dick.empty() ? 0.0 : since(tf) / double(dick.size()));
            }
        }
    }
    certified_arm(R);
    if (R.refused) return;
    // (B) at large n
    for (const auto& b : R.bad)
        check(summands(b.tors1, b.ell) == b.count, "(B) n=" + std::to_string(n) + " ell=" + std::to_string(b.ell) +
                                                     ": cyclic summands == [C, prop:badprimes](1) count");
    // (Z') every prime 5 <= ell <= n + 1 (D0 audit m5): the tree Bredon Betti numbers over F_ell == the good-prime
    // ranks, + the Dickson count at the Dickson primes -- independent of prop:badprimes' list.
    uint32_t zp = 0;
    if (n % 2 == 0)
        for (uint32_t ell = 5; ell <= n + 1; ++ell) {
            if (!integral_descent::IsPrime(ell)) continue;
            uint32_t cnt = 0;
            for (const auto& b : R.bad)
                if (b.ell == ell) cnt = b.count;
            const auto dims = integral_descent::TreeBredonFieldDims(n, ell);
            check(dims[0] == R.rational_dims[0] + cnt && dims[1] == R.rational_dims[1] + cnt,
                  "(Z') n=" + std::to_string(n) + " ell=" + std::to_string(ell) + ": tree-Bredon F_ell dims " + vs(dims) +
                      " == good-prime dims + count " + std::to_string(cnt));
            ++zp;
        }
    // the comparators reproduce the descent (measured only; they never feed the result)
    bool same_a = A.certified, same_c = C.certified;
    for (uint32_t q = 0; q <= 3; ++q) {
        same_a = same_a && A.H[q] == R.H[q];
        same_c = same_c && C.H[q] == R.H[q];
    }
    same_a = same_a && A.H[4] == tors_only(R.H[4]);
    same_c = same_c && tors_only(C.H[4]) == tors_only(R.H[4]);
    check(same_a, "(cost) n=" + std::to_string(n) + ": comparator (a) reproduces the descent");
    check(same_c, "(cost) n=" + std::to_string(n) + ": comparator (c) reproduces the descent");
    // the linear K search's run count, for the report (one extra pass, not timed)
    uint32_t runs_linear = 0;
    for (const auto& pr : R.dickson)
        runs_linear += static_cast<uint32_t>(
            integral_descent::RunShippedKMode(n, pr.first, pr.second, integral_descent::KSearch::Linear).K_tried.size());
    uint32_t a2 = 0, a3 = 0;
    for (const auto& g : R.H) {
        if (g.torsion.count(2)) a2 = std::max(a2, g.torsion.at(2).front());
        if (g.torsion.count(3)) a3 = std::max(a3, g.torsion.at(3).front());
    }
    std::cout << "  (cost) n=" << n << " (certified): " << groups_text(R.H) << std::endl;
    std::cout << "    (4b) n=" << n << ": rank P_0 - rank delta^0 over F_" << R.good_prime << " = " << R.two.dimH0_rank_prime
              << " (Iso^(2)), " << R.three.dimH0_rank_prime << " (Iso^(3)) vs b_0 = " << R.rational_dims[0]
              << "; (Z') " << zp << " primes; max exponents 2^" << a2 << " 3^" << a3 << "; Dickson " << R.dickson.size()
              << "; K runs doubling " << runs_doubling << " vs linear " << runs_linear << std::endl;
    const double md = t_d.median(), mz = t_good.median() + t_bad.median();
    // the decomposition (D0 audit m6): the descent with (c') in place of the C_7 K-mode, and with (4)(c)(ii) in place of
    // the full-layout F_ell' rank -- the structural descent-vs-tree gap
    const double s1 = t_good.median() + t_cp.median() + t_three.median() + t_two.median();
    const double s2 = s1 - t_rank2.median() - t_rank3.median();
    const double kr = t_kmode_run.median(), fr = t_f.median();
    std::cout << "DESCENT_COST n=" << n << " threads=1 repeats=" << g_repeats << " t_descent=" << fx(md) << " (min "
              << fx(t_d.min()) << ", max " << fx(t_d.max()) << ", cpu " << fx(c_d.median()) << ") t_a=" << fx(t_a.median())
              << " (min " << fx(t_a.min()) << ", max " << fx(t_a.max()) << ", cpu " << fx(c_a.median()) << ") t_c="
              << fx(t_c.median()) << " (min " << fx(t_c.min()) << ", cpu " << fx(c_c.median()) << ") t_c'="
              << fx(t_cp.median()) << " (min " << fx(t_cp.min()) << ") ratio_a=" << fx(t_a.median() / md) << " (min/min "
              << fx(t_a.min() / t_d.min()) << ") ratio_c=" << fx(t_c.median() / md) << " ratio_c'=" << fx(t_cp.median() / mz)
              << " share2=" << fx(t_two.median() / md) << " share3=" << fx(t_three.median() / md)
              << " shareZ16=" << fx(mz / md) << " t_good=" << fx(t_good.median()) << " t_kmode=" << fx(t_bad.median())
              << " t_iso3=" << fx(t_three.median()) << " t_iso2=" << fx(t_two.median()) << " (rank " << fx(t_rank2.median())
              << ", smith " << fx(t_smith2.median()) << ") kmode_run=" << fx(kr) << " field_run=" << fx(fr)
              << " kmode_overhead=" << fx(fr > 0 ? kr / fr : 0.0) << " struct_vs_c=" << fx(s1 / t_c.median())
              << " struct_ii_vs_c=" << fx(s2 / t_c.median()) << " rss_descent=" << gib(rss_d) << "GiB rss_a=" << gib(rss_a)
              << "GiB rss_c=" << gib(rss_c) << "GiB rss_c'=" << gib(rss_cp) << "GiB" << std::endl;
}

std::vector<uint32_t> parse_list(const char* env, const char* name) {
    std::vector<uint32_t> out;
    std::stringstream ss(env);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        if (item.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument(std::string(name) + ": '" + item + "' is not a non-negative integer");
        out.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    return out;
}

void levelN_cell(uint32_t N, uint32_t n) {
    const uint64_t cap = 40ull << 30;   // rule 19: 48 GiB summed ceiling, one cell at a time, 8 GiB margin
    const uint32_t D = integral_descent::PsiIndex(N) * (n + 1);
    const uint64_t p2 = integral_descent::PredictLocalRss(2, D, 4), p3 = integral_descent::PredictLocalRss(3, D, 4),
                   pt = integral_descent::PredictTreeRss(D);
    std::cout << "  (LN) N=" << N << " n=" << n << " D=" << D << ": predicted RSS Iso^(2) " << gib(p2) << " GiB, Iso^(3) "
              << gib(p3) << " GiB, tree " << gib(pt) << " GiB (cap " << gib(cap) << " GiB)" << std::endl;
    if (std::max(p3, pt) > cap) {
        std::cout << "  (LN) N=" << N << " n=" << n << ": NOT STARTED (prediction above the cap)" << std::endl;
        return;
    }
    const auto P = integral_descent::RunLevelNProbe(N, n, cap);
    check(P.three.all_certified(), "(LN) N=" + std::to_string(N) + " n=" + std::to_string(n) + ": Iso^(3) certified");
    if (P.ran_two)
        check(P.two.all_certified(), "(LN) N=" + std::to_string(N) + " n=" + std::to_string(n) + ": Iso^(2) certified");
    check(P.agree, "(LN) N=" + std::to_string(N) + " n=" + std::to_string(n) +
                       ": Iso^(ell) torsion == the tree comparator's at 2 and 3");
    std::ostringstream t2, t3;
    for (uint32_t q = 1; q <= 4; ++q) {
        t3 << (q > 1 ? " | " : "") << "H^" << q << ":" << P.three.tors[q].to_string();
        if (P.ran_two) t2 << (q > 1 ? " | " : "") << "H^" << q << ":" << P.two.tors[q].to_string();
    }
    std::cout << "    rational dims " << vs(P.rational_dims) << " (good prime " << P.good_prime << "); 3-part " << t3.str()
              << "; 2-part " << (P.ran_two ? t2.str() : std::string("not run")) << std::endl;
    std::cout << std::setprecision(4) << "DESCENT_LEVELN N=" << N << " n=" << n << " D=" << D
              << " t_shipped=" << P.t_shipped << " t_iso3=" << P.t_three << " t_iso2=" << (P.ran_two ? P.t_two : -1.0)
              << " t_tree2=" << P.t_tree2 << " t_tree3=" << P.t_tree3 << " rss_pred_iso2=" << gib(P.rss_pred_two)
              << " rss_meas_iso2=" << (P.ran_two ? gib(P.rss_meas_two) : std::string("-"))
              << " rss_pred_iso3=" << gib(P.rss_pred_three) << " rss_meas_iso3=" << gib(P.rss_meas_three)
              << " rss_pred_tree=" << gib(P.rss_pred_tree) << " rss_meas_tree=" << gib(P.rss_meas_tree)
              << " rss_meas_shipped=" << gib(P.rss_meas_shipped) << " agree=" << (P.agree ? 1 : 0) << std::endl;
}

uint32_t finish(clk::time_point t0) {
    std::cout << g_name << ": " << (g_fail ? "FAIL" : "PASS") << " (" << g_checks << " checks, " << g_fail << " failing, "
              << since(t0) << " s)" << std::endl;
    return g_fail;
}

}  // namespace

uint32_t IntegralDescentGroupsValidation() {
    g_checks = 0;
    g_fail = 0;
    g_name = "IntegralDescentGroupsValidation";
    g_b4_cells = g_b4_ok = 0;
    engine::tape::ThreadPool serial_pool(1);   // every route at width 1 (D0 audit M2)
    engine::tape::ScopedConsumerPool pin_all(&serial_pool);
    const auto t0 = clk::now();
    std::cout << "IntegralDescentGroupsValidation (slot-24 DESCENT D0; [C] prop:descent(2), prop:certifiedeell, "
                 "prop:badprimes; oracles [IB] Appendix A and descent_oracle_n128.tsv)" << std::endl;
    std::cout << "EFFECTIVE HeckeDescentTree route=" << (integral_descent::DescentTreeEnabled() ? "tree" : "dense")
              << " source=" << integral_descent::DescentTreeSource() << " (IntegralDescentGroupsValidation)" << std::endl;
    std::cout << "EFFECTIVE HeckeDescentGoodSparse route=" << (integral_descent::DescentGoodSparseEnabled() ? "sparse(even n; odd n: shipped)" : "shipped")
              << " source=" << integral_descent::DescentGoodSparseSource() << " (IntegralDescentGroupsValidation)" << std::endl;
    std::cout << "EFFECTIVE HeckeDescentGoodDense route=" << integral_descent::GoodPrimeRouteName(integral_descent::GoodPrimeRouteFor(2))
              << "(even n; odd n: shipped) source=" << integral_descent::DescentGoodDenseSource() << " (IntegralDescentGroupsValidation)"
              << std::endl;

    // (Q) build-time denominators
    {
        const uint64_t c2 = integral_descent::CheckDenominators(integral_descent::Iso2(), integral_descent::QEll(2));
        const uint64_t c3 = integral_descent::CheckDenominators(integral_descent::Iso3(), integral_descent::QEll(3));
        check(c2 > 0 && c3 > 0, "(Q) denominators checked");
        std::cout << "  (Q) every coefficient of d and every e_sigma: denominators divide q_2 = 3 (" << c2
                  << " elements) and q_3 = 4 (" << c3 << " elements)" << std::endl;
    }

    // (A) n = 0..20 against [IB] Appendix A
    const auto& O1 = integral_descent::AppendixAGroups();
    uint32_t a_cells = 0;
    for (uint32_t n = 0; n <= 20; ++n) {
        const DescentResult& R = descent(n);
        certified_arm(R);
        if (R.refused) continue;
        for (uint32_t q = 0; q <= 5; ++q) {
            check(R.H[q] == O1[n][q], "(A) n=" + std::to_string(n) + " q=" + std::to_string(q) + ": descent " +
                                          R.H[q].to_string() + " == Appendix A " + O1[n][q].to_string());
            ++a_cells;
        }
        for (const auto& b : R.bad)
            check(summands(b.tors1, b.ell) == b.count, "(B) n=" + std::to_string(n) + " ell=" + std::to_string(b.ell) +
                                                         ": cyclic summands == [C, prop:badprimes](1) count");
    }
    std::cout << "  (A) " << a_cells << " cells n = 0..20, q = 0..5 against [IB] Appendix A" << std::endl;
    if (!descent(16).refused) std::cout << "  (A) n=16: " << groups_text(descent(16).H) << std::endl;

    // (T) the oracle table of record
    const integral_descent::OracleTable T = integral_descent::LoadOracleTable(oracle_path());
    check(T.found && T.error.empty(), "(T) the oracle table is readable: " + T.path + " " + T.error);
    {
        // A/T agreement on n <= 20 (the oracle author checked 90 of these cells against [IB]; this covers all 126)
        uint32_t at_cells = 0, at_dis = 0;
        for (uint32_t n = 0; n <= 20; ++n) {
            const auto it = T.groups.find(n);
            if (it == T.groups.end() || it->second.size() < 6) continue;
            for (uint32_t q = 0; q <= 5; ++q) {
                ++at_cells;
                if (!check(it->second[q] == O1[n][q], "(T) A/T: oracle n=" + std::to_string(n) + " q=" + std::to_string(q) +
                                                          " " + it->second[q].to_string() + " == Appendix A " +
                                                          O1[n][q].to_string()))
                    ++at_dis;
            }
        }
        check(at_cells == 126, "(T) A/T covers 126 cells");
        std::cout << "  (T) A/T: oracle vs Appendix A on " << at_cells << " cells, " << at_dis << " disagreeing" << std::endl;
        uint32_t cells = 0;
        const std::vector<uint32_t> sample = {22, 24, 30, 32, 36, 48, 64, 128};
        for (uint32_t n = 0; n <= 20; ++n) cells += oracle_compare(T, n);
        for (uint32_t n : sample) cells += oracle_compare(T, n);
        std::cout << "  (T) " << cells << " cells against the oracle (n <= 20 and n in " << vs(sample)
                  << "; every n <= 128 in the slow name)" << std::endl;
        for (uint32_t j : {5u, 6u, 7u}) {
            const uint32_t n = 1u << j;
            if (descent(n).refused) continue;
            const auto& H1 = descent(n).H[1];
            const uint32_t top = H1.torsion.count(2) ? H1.torsion.at(2).front() : 0;
            check(top == j, "(T) [IB] F-4: Z/2^" + std::to_string(j) + " is the top 2-power of H^1 at n=" + std::to_string(n));
            std::cout << "  (T) [IB] F-4 at n=" << n << ": top 2-exponent of H^1 = " << top << std::endl;
        }
    }

    // (D), (P), (W), (C), (Z), (c) at sample n
    for (uint32_t n : {22u, 32u, 44u, 60u}) {
        const DescentResult& R = descent(n);
        const std::string at = " at n=" + std::to_string(n);
        if (!check(!R.refused, "(D) descent certified" + at)) continue;
        // (D)
        const auto A = integral_descent::RunComparatorA(n, R.good_prime);
        bool d_ok = A.certified;
        for (uint32_t q = 0; q <= 3; ++q) d_ok = d_ok && A.H[q] == R.H[q];
        d_ok = d_ok && A.H[4] == tors_only(R.H[4]);
        check(d_ok, "(D) descent == comparator (a) (C_I to degree 4; independent route, shared model family)" + at);
        // (P)
        integral_descent::DescentOptions o6;
        o6.q_top_two = 6;
        const DescentResult R6 = integral_descent::AssembleDescent(n, o6);
        if (!check(!R6.refused && !R6.H5_from_periodicity && R6.two.all_certified(), "(P) Iso^(2) to degree 6 certified" + at))
            continue;
        check(tors_only(R6.H[4]) == tors_only(R6.H[2]) && tors_only(R6.H[5]) == tors_only(R6.H[3]),
              "(P) direct H^4 == H^2, H^5 == H^3 ([C, lem:bulkperiodic] tested, not assumed)" + at);
        check(R6.H == R.H, "(P) degree-6 groups == production groups" + at);
        // (c), (c')
        const auto C = integral_descent::RunComparatorTree(n, R.good_prime);
        bool c_ok = C.certified;
        for (uint32_t q = 0; q <= 3; ++q) c_ok = c_ok && C.H[q] == R.H[q];
        check(c_ok, "(c) the tree comparator reproduces H^0..H^3" + at);
        const auto Cp = integral_descent::RunComparatorTreeBredon(n, R.good_prime);
        AbelianGroupType big1;
        for (const auto& b : R.bad) big1.torsion[b.ell] = b.tors1.torsion.count(b.ell) ? b.tors1.torsion.at(b.ell)
                                                                                      : std::vector<uint32_t>{};
        for (auto it = big1.torsion.begin(); it != big1.torsion.end();)
            it = it->second.empty() ? big1.torsion.erase(it) : std::next(it);
        big1.free_rank = R.H[1].free_rank;
        check(Cp.certified && Cp.H[1] == big1 && Cp.H[0].free_rank == R.H[0].free_rank,
              "(c') the tree Bredon comparator reproduces the Z[1/6] piece" + at);
        // (C) at the Dickson primes: dim H^q(C_7 (x) F_ell) = rank + #summands(H^q) + #summands(H^{q+1})
        for (const auto& b : R.bad) {
            const auto dims = integral_descent::ShippedFieldDims(n, b.ell);
            const uint32_t c = summands(b.tors1, b.ell);
            check(dims[0] == R.rational_dims[0] + c && dims[1] == R.rational_dims[1] + c && dims[2] == 0,
                  "(C) UCT at the Dickson prime " + std::to_string(b.ell) + at + ": shipped F_ell dims " + vs(dims));
        }
        // (C) at 2 and 3 against Iso^(ell) (x) F_ell
        for (uint32_t ell : {2u, 3u}) {
            const auto& L = ell == 2 ? integral_descent::Iso2() : integral_descent::Iso3();
            const auto len = L.cohomology_lengths(U32Ctx(ell), n);
            std::vector<uint32_t> pred;
            for (uint32_t q = 0; q <= 4; ++q) pred.push_back(R.H[q].tensor_length(ell, 1) + R.H[q + 1].tor_length(ell, 1));
            check(std::vector<uint32_t>(len.begin(), len.begin() + 5) == pred,
                  "(C) UCT at " + std::to_string(ell) + at + ": Iso^(ell) (x) F_ell " + vs(len) + " vs " + vs(pred));
            if (ell == 3) {
                const auto b = integral_descent::ShippedFieldDims(n, 3);
                check(b[0] == len[0] && b[1] + len[3] == len[1],
                      "(C) [C, prop:ellthreeclosed](3): Bredon F_3 H^0 == group H^0, H^1(Bredon) + H^3(group) == H^1(group)" + at);
            }
        }
        // (Z) zero-count primes 5 <= ell <= n + 1
        uint32_t z = 0;
        for (uint32_t ell = 5; ell <= n + 1; ++ell) {
            if (!integral_descent::IsPrime(ell)) continue;
            bool dick = false;
            for (const auto& pr : R.dickson) dick = dick || pr.first == ell;
            if (dick) continue;
            const auto dims = integral_descent::ShippedFieldDims(n, ell);
            check(dims == R.rational_dims, "(Z) zero-count prime " + std::to_string(ell) + at +
                                               ": shipped F_ell dims == good-prime dims " + vs(dims));
            ++z;
        }
        std::cout << "  (D,P,c,c',C,Z) n=" << n << ": (a) " << A.seconds << " s; (P) degree 6 ok; " << z
                  << " zero-count primes; Dickson " << R.dickson.size() << "; descent " << R.t_total << " s" << std::endl;
    }

    // (GS) (slot-24 PAR-L1) the good-prime routes agree: the sparse cokernel of Gamma_0(1) (default at even n) == the shipped C_7
    // over F_good, every degree; the rollback HECKE_DESCENT_GOOD_SPARSE=0 (scoped here) returns the shipped dims verbatim.
    // (PAR-GP audit F2) the dense route is the default since PAR-GP, so this arm pins the sparse route (HECKE_DESCENT_GOOD_DENSE=0
    // scoped) and keeps testing what it says; the dense route is (GD)'s.
    {
        integral_descent::ScopedDescentGoodDense pin_sparse(false);
        uint32_t gs = 0;
        for (uint32_t n : {0u, 2u, 10u, 12u, 22u, 40u, 64u, 100u, 128u, 13u}) {
            const uint32_t g = integral_descent::DefaultGoodPrime(n);
            const auto shipped = integral_descent::ShippedFieldDims(n, g);
            bool sp = false, sp0 = true;
            const auto viaSparse = integral_descent::GoodPrimeFieldDims(n, g, &sp);
            std::vector<uint32_t> viaRollback;
            {
                integral_descent::ScopedDescentGoodSparse off(false);
                viaRollback = integral_descent::GoodPrimeFieldDims(n, g, &sp0);
            }
            check(sp == (n % 2 == 0) && !sp0, "(GS) the route taken at n=" + std::to_string(n));
            check(viaSparse == shipped && viaRollback == shipped,
                  "(GS) good-prime dims: sparse " + vs(viaSparse) + " == shipped C_7 " + vs(shipped) + " at n=" + std::to_string(n));
            ++gs;
        }
        std::cout << "  (GS) good prime: sparse cokernel (pinned: HECKE_DESCENT_GOOD_DENSE=0 scoped) == shipped C_7 at " << gs << " n (odd n: shipped); rollback scoped" << std::endl;
    }

    // (GD) (slot-24 PAR-GP) the dense good prime: rank e_S, rank e_ST, rank [e_S | e_ST] at 1 / 2 / 7 / 16 threads == the sparse
    // cokernel's dim_S, dim_ST, dim Rel (PAR-L1's route) entry for entry; the dims by the dense route (default), the scoped rollback
    // HECKE_DESCENT_GOOD_DENSE=0 (the sparse cokernel) and the shipped C_7 agree; the threaded runs opened hook regions
    {
        uint32_t gd = 0;
        const uint64_t reg0 = integral_descent::GoodPrimeDenseHookRegions();
        for (uint32_t n : {0u, 2u, 10u, 40u, 100u, 200u, 400u, 13u}) {
            const uint32_t g = integral_descent::DefaultGoodPrime(n);
            const std::string at = " at n=" + std::to_string(n);
            if (n % 2 == 0) {
                const auto Q = hecke::level::SparseCokernel::Build(hecke::level::Gamma0LevelN(1), n, engine::ModCtx<uint32_t>(g), false);
                for (uint32_t t : {1u, 2u, 7u, 16u}) {
                    const auto r = integral_descent::GoodPrimeDenseRanks(n, g, t);
                    check(r.threads == t && r.dim_S == Q.stats().dim_S && r.dim_ST == Q.stats().dim_ST && r.dim_rel == Q.dim_rel(),
                          "(GD) dense ranks (" + std::to_string(r.dim_S) + ", " + std::to_string(r.dim_ST) + ", " + std::to_string(r.dim_rel) +
                              ") == sparse (" + std::to_string(Q.stats().dim_S) + ", " + std::to_string(Q.stats().dim_ST) + ", " +
                              std::to_string(Q.dim_rel()) + ") at " + std::to_string(t) + " threads" + at);
                }
            }
            integral_descent::GoodPrimeRoute rd = integral_descent::GoodPrimeRoute::Shipped, rs = rd;
            std::vector<uint32_t> viaDense, viaSparse;
            {
                integral_descent::ScopedDescentGoodDense on(true);
                viaDense = integral_descent::GoodPrimeFieldDimsRoute(n, g, &rd);
            }
            {
                integral_descent::ScopedDescentGoodDense off(false);
                viaSparse = integral_descent::GoodPrimeFieldDimsRoute(n, g, &rs);
            }
            const auto shipped = integral_descent::ShippedFieldDims(n, g);
            const bool even = n % 2 == 0;
            check(rd == (even ? integral_descent::GoodPrimeRoute::Dense : integral_descent::GoodPrimeRoute::Shipped) &&
                      rs == (even ? integral_descent::GoodPrimeRoute::Sparse : integral_descent::GoodPrimeRoute::Shipped),
                  "(GD) the routes taken" + at);
            check(viaDense == shipped && viaSparse == shipped,
                  "(GD) good-prime dims: dense " + vs(viaDense) + " == sparse " + vs(viaSparse) + " == shipped C_7 " + vs(shipped) + at);
            ++gd;
        }
        const uint64_t regions = integral_descent::GoodPrimeDenseHookRegions() - reg0;
        check(regions > 0, "(GD) the threaded dense runs split over the descent pool (hook regions " + std::to_string(regions) + ")");
        std::cout << "  (GD) good prime: dense ranks == sparse cokernel at 1/2/7/16 threads, dims == shipped C_7 at " << gd
                  << " n; hook regions " << regions << std::endl;
    }

    // (W) the W = 61 route
    for (uint32_t n : {16u, 32u, 40u}) {
        const DescentResult& R = descent(n);
        std::vector<uint32_t> h(8, 0);
        h[0] = R.rational_dims[0];
        h[1] = R.rational_dims[1];
        const auto W = integral_descent::RunLocalPieceSym(integral_descent::Iso2(), n, 4, h, R.good_prime, 0, true);
        bool ok = W.all_certified();
        for (uint32_t q = 0; q <= 4; ++q) ok = ok && W.tors[q] == R.two.tors[q];
        for (const auto& d : W.d) ok = ok && d.W == 61;
        check(ok, "(W) the O_2 -> Z/2^61 route (ModCtx<uint64_t>) == the Z/2^30 route at n=" + std::to_string(n));
    }
    std::cout << "  (W) Z/2^61 (ModCtx<uint64_t>, ring2k uint64_t) == Z/2^30 on Iso^(2) at n = 16, 32, 40" << std::endl;

    // (N) negative control
    {
        std::vector<uint32_t> h(8, 0);
        const DescentResult& R16 = descent(16);
        h[0] = R16.rational_dims[0];
        h[1] = R16.rational_dims[1];
        const U32Ctx c4 = U32Ctx::prime_power(2, 4);
        const auto L4 = integral_descent::RunLocalPiece(integral_descent::Iso2(), integral_model::SymModule(c4, 16), c4, 4,
                                                        h, R16.good_prime, integral_model::SymModule(U32Ctx(R16.good_prime), 16),
                                                        -1, false);
        check(!L4.d[0].certified && L4.d[0].r_K < L4.d[0].rank_Q && L4.tors[1].is_zero(),
              "(N) n=16 at Z/2^4 (H^1 has Z/2^4): d^0 refused, nothing read off");
        const DescentResult& R12 = descent(12);
        h[0] = R12.rational_dims[0];
        h[1] = R12.rational_dims[1];
        const auto L2 = integral_descent::RunLocalPieceSym(integral_descent::Iso3(), 12, 6, h, R12.good_prime, 2);
        check(!L2.d[0].certified && L2.d[0].r_K < L2.d[0].rank_Q, "(N) n=12 at Z/3^2 (H^1 has Z/9): d^0 refused");
        check(L4.d[0].W == 4 && !L4.d[0].escalated, "(N) the negative control ran without the W = 61 escalation");
        std::cout << "  (N) n=16 Z/2^4: r_K(d^0)=" << L4.d[0].r_K << " < rank_Q=" << L4.d[0].rank_Q
                  << " refused; n=12 Z/3^2: r_K(d^0)=" << L2.d[0].r_K << " < " << L2.d[0].rank_Q << " refused" << std::endl;
    }

    // (R) the descent REFUSES as a whole (D0 audit M3): Iso^(3) at K = 2, n = 12 (true 3-part of H^1: Z/3 + Z/3^2)
    {
        integral_descent::DescentOptions o;
        o.K_three = 2;
        const DescentResult R = integral_descent::AssembleDescent(12, o);
        check(R.refused && R.H.empty() && !R.certified() && R.refusal.find("Iso^(3)") != std::string::npos,
              "(R) n=12 with Iso^(3) at Z/3^2: the descent is refused and reports no group");
        std::cout << "  (R) " << R.refusal << " (H has " << R.H.size() << " entries)" << std::endl;
    }

    // (G) the guards and the natural escalation (D0 audit m2), each an expected refusal (std::logic_error)
    {
        auto refuses = [](const std::function<void()>& f) {
            try { f(); } catch (const std::logic_error&) { return true; }
            return false;
        };
        const DescentResult& R16 = descent(16);
        std::vector<uint32_t> h(8, 0);
        h[1] = R16.rational_dims.size() > 1 ? R16.rational_dims[1] : 0u;
        const U32Ctx F(10007);
        uint32_t guards = 0;
        auto g = [&](bool ok, const std::string& what) { check(ok, "(G) " + what); ++guards; };
        g(refuses([&] {
              const U32Ctx c = U32Ctx::prime_power(3, 19);
              (void)integral_descent::RunLocalPiece(integral_descent::Iso2(), integral_model::SymModule(c, 16), c, 4, h, 10007,
                                                    integral_model::SymModule(F, 16), -1, false);
          }), "Iso^(2) reduced into Z/3^19 is refused (the ell() guard)");
        g(refuses([&] {
              const U32Ctx c(3);
              (void)integral_descent::RunLocalPiece(integral_descent::Iso3(), integral_model::SymModule(c, 16), c, 6, h, 10007,
                                                    integral_model::SymModule(F, 16), -1, false);
          }), "Iso^(3) reduced into the field F_3 is refused");
        g(refuses([&] {
              const U32Ctx c = U32Ctx::prime_power(2, 30);
              (void)integral_descent::RunLocalPiece(integral_descent::Iso2(), integral_model::SymModule(c, 16), c, 4, h, 10007,
                                                    integral_model::SymModule(c, 16), -1, false);
          }), "a rank module over Z/2^30 read at F_10007 is refused (the (4)(c)(ii) cross-check)");
        g(refuses([&] {
              const U32Ctx c = U32Ctx::prime_power(2, 30);
              (void)integral_descent::RunLocalPiece(integral_descent::Iso2(), integral_model::SymModule(c, 16), c, 4, h, 10007,
                                                    integral_model::SymModule(F, 18), -1, false);
          }), "a rank module of another dimension (Sym^18 for Sym^16) is refused");
        g(refuses([&] { (void)integral_descent::RunLocalPieceSym(integral_descent::Iso2(), 16, 4, h, 7); }),
          "rank prime 7 at n = 16 (7 divides tors H^1: does not qualify for d^0) is refused at 2");
        g(refuses([&] { (void)integral_descent::RunLocalPieceSym(integral_descent::Iso3(), 16, 6, h, 7); }),
          "rank prime 7 at n = 16 is refused at 3");
        {
            std::vector<uint32_t> hb = h;
            hb[1] += 1;
            g(refuses([&] { (void)integral_descent::RunLocalPieceSym(integral_descent::Iso2(), 16, 4, hb, 10007); }),
              "a wrong b_1 is refused");
        }
        // the natural escalation: K = 4 at 2, n = 16 (H^1 has Z/2^4) with sym_n set: d^0 reruns at W = 61 and certifies
        const auto E = integral_descent::RunLocalPieceSym(integral_descent::Iso2(), 16, 4, h, 10007, 4);
        bool esc = E.all_certified() && E.d[0].escalated && E.d[0].W == 61;
        for (uint32_t q = 1; q < E.d.size(); ++q) esc = esc && !E.d[q].escalated && E.d[q].W == 4;
        for (uint32_t q = 0; q <= 4; ++q) esc = esc && E.tors[q] == R16.two.tors[q];
        g(esc, "K = 4 at n = 16: d^0 escalates to W = 61 and certifies, d^1..d^3 certify at K = 4 (a_t <= 3), groups == K = 30");
        std::cout << "  (G) " << guards << " guard / escalation cases" << std::endl;
    }
    check(g_b4_cells > 0 && g_b4_ok == g_b4_cells, "(4b) every cell: rank P_0 - rank delta^0 over F_ell' == b_0");
    std::cout << "  (4b) cells=" << g_b4_cells << " ok=" << g_b4_ok
              << " (rank P_0 - rank delta^0 over F_10007 == b_0 on Iso^(2) and Iso^(3))" << std::endl;
    return finish(t0);
}

uint32_t IntegralDescentGroupsSlowValidation() {
    g_checks = 0;
    g_fail = 0;
    g_name = "IntegralDescentGroupsSlowValidation";
    g_b4_cells = g_b4_ok = 0;
    // (D0 audit M2) the whole name runs on one serial consumer pool: every route at width 1 (cost_cell pins it again).
    engine::tape::ThreadPool serial_pool(1);
    engine::tape::ScopedConsumerPool pin_all(&serial_pool);
    const auto t0 = clk::now();
    std::cout << "IntegralDescentGroupsSlowValidation (slot-24 DESCENT D0: DESCENT_COST, oracle (T) n <= 128, level-N "
                 "probe)" << std::endl;
    // (T) every cell of the oracle table
    {
        const integral_descent::OracleTable T = integral_descent::LoadOracleTable(oracle_path());
        check(T.found && T.error.empty(), "(T) the oracle table is readable: " + T.path + " " + T.error);
        uint32_t cells = 0;
        const auto tt = clk::now();
        for (const auto& kv : T.groups) cells += oracle_compare(T, kv.first);
        check(cells == 129 * 6, "(T) 774 oracle cells (n = 0..128, q = 0..5)");
        std::cout << "  (T) " << cells << " cells n = 0..128 against the oracle table (" << since(tt) << " s)" << std::endl;
        std::cout << "  (4b) cells=" << g_b4_cells << " ok=" << g_b4_ok << std::endl;
        check(g_b4_ok == g_b4_cells, "(4b) every cell of (T)");
    }
    // DESCENT_COST
    {
        std::vector<uint32_t> cells = {100, 200, 300};
        std::string arm = "default";
        if (const char* env = std::getenv("HECKE_DESCENT_CELLS")) {
            cells = parse_list(env, "HECKE_DESCENT_CELLS");
            arm = "HECKE_DESCENT_CELLS";
            check(!cells.empty(), "HECKE_DESCENT_CELLS is set but names no cell: 0 cells is not a pass");
        }
        g_repeats = 5;
        if (const char* env = std::getenv("HECKE_DESCENT_REPEATS")) {
            const auto r = parse_list(env, "HECKE_DESCENT_REPEATS");
            if (r.size() != 1 || r[0] < 1 || r[0] > 99) throw std::invalid_argument("HECKE_DESCENT_REPEATS must be one integer in 1..99");
            g_repeats = r[0];
        }
        std::cout << "EFFECTIVE IntegralDescentGroupsSlowValidation arm=" << arm << " cells=" << cells.size()
                  << " repeats=" << g_repeats << " threads=1" << std::endl;
        for (uint32_t n : cells) cost_cell(n);
    }
    // the level-N probe
    {
        std::vector<std::pair<uint32_t, uint32_t>> cells = {{11, 6}, {11, 20}, {16, 6}, {16, 20}, {50, 6}};
        std::string arm = "default";
        if (const char* env = std::getenv("HECKE_DESCENT_LEVELN_CELLS")) {
            cells.clear();
            arm = "HECKE_DESCENT_LEVELN_CELLS";
            std::stringstream ss(env);
            std::string item;
            while (std::getline(ss, item, ',')) {
                const size_t c = item.find(':');
                if (c == std::string::npos) throw std::invalid_argument("HECKE_DESCENT_LEVELN_CELLS: '" + item + "' is not N:n");
                const auto N = parse_list(item.substr(0, c).c_str(), "HECKE_DESCENT_LEVELN_CELLS");
                const auto n = parse_list(item.substr(c + 1).c_str(), "HECKE_DESCENT_LEVELN_CELLS");
                if (N.size() != 1 || n.size() != 1 || N[0] < 1)
                    throw std::invalid_argument("HECKE_DESCENT_LEVELN_CELLS: '" + item + "' is not N:n");
                cells.push_back({N[0], n[0]});
            }
            check(!cells.empty(), "HECKE_DESCENT_LEVELN_CELLS is set but names no cell: 0 cells is not a pass");
        }
        std::cout << "EFFECTIVE IntegralDescentGroupsSlowValidation levelN_arm=" << arm << " cells=" << cells.size()
                  << std::endl;
        for (const auto& [N, n] : cells) levelN_cell(N, n);
    }
    return finish(t0);
}
