// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT D5: HeckeIsotropyModelTreeValidation / HeckeIsotropyModelTreeSlowValidation.  See the header for the arms.
// [C] = docs/latex/gl2_contract.tex, cited by label (def:isomodeltree, prop:isomodeltree, prop:certifiedeell, prop:descentN).
#include "HeckeIsotropyModelTreeValidation.h"
#include "DescentParAsm.h"

#include "DescentAssembly.h"
#include "DescentLevelN.h"
#include "DescentMaps.h"
#include "DescentOracle.h"
#include "DescentTree.h"
#include "IntegralDescentGroupsValidation.h"
#include "IntegralDescentLevelNValidation.h"
#include "IntegralDescentMapsValidation.h"
#include "TreeComparisonArms.h"

#include "../Model/TreeCochains.h"
#include "../../../Engine/EngineRuntime.h"
#include "../../../Engine/ThreadPool.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <array>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using integral_descent::AbelianGroupType;
using integral_descent::DescentResult;
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

std::string src_path(const std::string& rel) {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/" + rel;
#else
    return rel;
#endif
}

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
    std::string text() const {
        if (v.empty()) return "-";
        return fx(median()) + "(" + fx(min()) + "-" + fx(max()) + ")";
    }
};

// ---- the level-N oracle (descentN_oracle.tsv) ---------------------------------------------------
std::map<std::pair<uint32_t, uint32_t>, std::vector<AbelianGroupType>> load_oracle_n(bool& ok) {
    std::map<std::pair<uint32_t, uint32_t>, std::vector<AbelianGroupType>> T;
    std::ifstream f(src_path("docs/implementation/slot24/descentN_oracle.tsv"));
    ok = static_cast<bool>(f);
    std::string line;
    bool header = false;
    while (ok && std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!header) {
            header = true;
            if (line.rfind("N\t", 0) == 0) continue;
        }
        std::istringstream is(line);
        uint32_t N = 0, n = 0, q = 0;
        std::string g;
        if (!(is >> N >> n >> q >> g)) {
            ok = false;
            break;
        }
        auto& v = T[{N, n}];
        if (v.size() != q) {
            ok = false;
            break;
        }
        v.push_back(integral_descent::ParseOracleGroup(g));
    }
    return T;
}

// ---- (L1) / (LN): the two routes -----------------------------------------------------------------
DescentResult descent1(uint32_t n, bool tree) {
    integral_descent::ScopedDescentTree route(tree);
    return integral_descent::AssembleDescent(n);
}
LevelNResult descentN(uint32_t N, uint32_t n, bool tree) {
    integral_descent::ScopedDescentTree route(tree);
    return integral_descent::AssembleDescentN(N, n);
}
bool same_pieces(const integral_descent::LocalPiece& a, const integral_descent::LocalPiece& b, uint32_t ell, uint32_t qmax) {
    if (a.tors.size() <= qmax || b.tors.size() <= qmax) return false;
    for (uint32_t q = 1; q <= qmax; ++q)
        if (a.tors[q].primary_part(ell) != b.tors[q].primary_part(ell)) return false;
    return true;
}

uint32_t arm_L1(const std::vector<uint32_t>& ns, const integral_descent::OracleTable* T) {
    uint32_t cells = 0;
    for (uint32_t n : ns) {
        const DescentResult A = descent1(n, true), B = descent1(n, false);
        const std::string at = " at n=" + std::to_string(n);
        if (!check(!A.refused && !B.refused, "(L1) neither route refuses" + at + " -- " + A.refusal + B.refusal)) continue;
        check(A.two.route == "tree" && A.three.route == "tree" && B.two.route == "dense" && B.three.route == "dense",
              "(L1) the switch selects the routes" + at);
        bool eq = A.H.size() == 6 && B.H.size() == 6;
        for (uint32_t q = 0; eq && q <= 5; ++q) eq = A.H[q] == B.H[q];
        check(eq, "(L1) tree == dense, every H^q" + at);
        check(same_pieces(A.two, B.two, 2, 4) && same_pieces(A.three, B.three, 3, 5),
              "(L1) the 2- and 3-primary pieces agree (q = 1..4 / 1..5)" + at);
        // (D5 audit m3) the ranks computed here on the production form over F_ell' (A.good_prime), against the shipped C_7's Betti
        // numbers -- not the piece's own rank_Q (which the piece ENGINE_CHECKs against the same h, so a comparison with it is vacuous)
        for (uint32_t ell : {2u, 3u}) {
            std::string why;
            check(integral_descent::TreeFieldDimsEqual(ell, n, A.good_prime, A.rational_dims, 3, why),
                  "(L1) the tree form over F_ell' at ell=" + std::to_string(ell) + ": dim H^q (ranks computed here) == the shipped C_7 Betti "
                  "numbers, q = 0..3" + at + " -- " + why);
        }
        if (T) {
            const auto it = T->groups.find(n);
            if (check(it != T->groups.end() && it->second.size() >= 6, "(L1) the oracle has n=" + std::to_string(n))) {
                bool ok = true;
                for (uint32_t q = 0; q <= 5; ++q) ok = ok && A.H[q] == it->second[q];
                check(ok, "(L1) tree == descent_oracle_n128.tsv" + at);
            }
        }
        ++cells;
    }
    return cells;
}

uint32_t arm_LN(const std::map<std::pair<uint32_t, uint32_t>, std::vector<AbelianGroupType>>& T, uint32_t D_max) {
    uint32_t cells = 0;
    for (const auto& kv : T) {
        const uint32_t N = kv.first.first, n = kv.first.second;
        if (integral_descent::PsiIndex(N) * (n + 1) > D_max) continue;
        const LevelNResult A = descentN(N, n, true), B = descentN(N, n, false);
        const std::string at = " at " + cellname(N, n);
        if (!check(!A.refused && !B.refused, "(LN) neither route refuses" + at + " -- " + A.refusal + B.refusal)) continue;
        bool eq = A.H.size() == 6 && B.H.size() == 6 && kv.second.size() == 6;
        for (uint32_t q = 0; eq && q <= 5; ++q) eq = A.H[q] == B.H[q] && A.H[q] == kv.second[q];
        check(eq, "(LN) tree == dense == descentN_oracle.tsv, every H^q" + at);
        if (A.ran_two && B.ran_two) check(A.two.route == "tree" && same_pieces(A.two, B.two, 2, 4), "(LN) 2-primary pieces agree" + at);
        if (A.ran_three && B.ran_three)
            check(A.three.route == "tree" && same_pieces(A.three, B.three, 3, 5), "(LN) 3-primary pieces agree" + at);
        ++cells;
    }
    return cells;
}

// ---- (D2): the level pair on both routes -----------------------------------------------------------------
bool same_maps(const std::vector<integral_descent::maps::MapTypes>& a, const std::vector<integral_descent::maps::MapTypes>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].ker != b[i].ker || a[i].im != b[i].im || a[i].coker != b[i].coker) return false;
    return true;
}
std::vector<std::array<uint32_t, 3>> load_d2_cells(bool& ok) {
    std::vector<std::array<uint32_t, 3>> cells;
    std::ifstream f(src_path("docs/implementation/slot24/d2_maps_oracle.tsv"));
    ok = static_cast<bool>(f);
    std::string line;
    std::set<std::array<uint32_t, 3>> seen;
    while (ok && std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("N\t", 0) == 0) continue;
        std::istringstream is(line);
        std::array<uint32_t, 3> c{};
        if (!(is >> c[0] >> c[1] >> c[2])) continue;
        if (seen.insert(c).second) cells.push_back(c);
    }
    return cells;
}
uint32_t arm_D2(uint32_t S_max) {
    bool ok = false;
    const auto cells = load_d2_cells(ok);
    check(ok && !cells.empty(), "(D2) d2_maps_oracle.tsv loads");
    uint32_t done = 0;
    double tt = 0, td = 0;
    for (const auto& c : cells) {
        const uint32_t N = c[0], MN = c[1], n = c[2];
        if (integral_descent::PsiIndex(MN) * (n + 1) > S_max) continue;
        integral_descent::maps::PairResult A, B;
        {
            integral_descent::ScopedDescentTree r(true);
            A = integral_descent::maps::AssemblePairMaps(N, MN / N, n);
        }
        {
            integral_descent::ScopedDescentTree r(false);
            B = integral_descent::maps::AssemblePairMaps(N, MN / N, n);
        }
        tt += A.t_total;
        td += B.t_total;
        const std::string at = " at (N,MN,n)=(" + std::to_string(N) + "," + std::to_string(MN) + "," + std::to_string(n) + ")";
        if (!check(!A.refused && !B.refused, "(D2) neither route refuses" + at + " -- " + A.refusal + B.refusal)) continue;
        bool route = false;
        for (const auto& pr : A.pieces) route = route || pr.route == "Iso2T";
        check(route, "(D2) the 2-adic pair piece ran on the tree" + at);
        check(A.HA == B.HA && A.HB == B.HB && A.HAB == B.HAB, "(D2) tree == dense: H^q(A), H^q(B), H^q(A, B)" + at);
        check(same_maps(A.res, B.res) && same_maps(A.cor, B.cor) && same_maps(A.j, B.j) && same_maps(A.dl, B.dl),
              "(D2) tree == dense: ker / im / coker of res, cor, j, dl" + at);
        check(A.exact_spots == B.exact_spots && A.identity_checks == B.identity_checks && A.five_term == B.five_term,
              "(D2) the same exactness spots, identities and five-term statement on both routes" + at);
        ++done;
    }
    std::cout << "  (D2) " << done << " level pairs (psi(MN)(n+1) <= " << S_max << "): AssemblePairMaps tree == dense (groups, res, cor, "
              << "j, dl, exactness); total " << fx(tt) << " s tree vs " << fx(td) << " s dense" << std::endl;
    return done;
}

// ---- (E), (W), (I), (B) -------------------------------------------------------------------------------
// (E)'s sweep: TreeLocalForm against TreeModelChain on every (cell, ell, F_p) (p = ell itself and two primes prime to q_ell; the
// Bredon tree at 5, 7).  report: count each run as a check (the arm); else only return whether all passed (the (NT) arm).
bool e_sweep(const std::vector<std::pair<uint32_t, uint32_t>>& cells, uint32_t q_max, bool report, uint32_t* n_checks = nullptr,
             uint32_t* n_runs = nullptr) {
    bool all = true;
    for (const auto& [N, n] : cells)
        for (uint32_t ell : {2u, 3u, 5u, 7u}) {
            const std::vector<uint32_t> primes = ell == 2 ? std::vector<uint32_t>{2, 5, 7}
                                                 : ell == 3 ? std::vector<uint32_t>{3, 5, 7}
                                                            : std::vector<uint32_t>{5, 7, 11};
            for (uint32_t p : primes) {
                integral_descent::TreeFormCheck r;
                try {
                    r = integral_descent::CheckTreeFormAgainstChain(ell, N, n, p, q_max);
                } catch (const std::logic_error& e) {
                    r.dims = false;
                    r.first = std::string("exception: ") + e.what();
                }
                all = all && r.ok();
                if (report) check(r.ok(), "(E) TreeLocalForm == TreeModelChain (d o d = 0, c_q, dim H^q) -- " + r.first);
                if (n_checks) *n_checks += r.checks;
                if (n_runs) ++*n_runs;
            }
        }
    return all;
}

void arm_EWIB() {
    // (E) (D5 audit m3) the production form TreeLocalForm against the independent chain model TreeModelChain, q <= 7 (2, 3) and
    // q <= 1 (the Bredon tree at 5, 7): d o d = 0, c_q = rank of the chain model's projector, dim H^q equal (TreeComparisonArms.h).
    // The old arm compared an "explicit" d^q with its (2)(d) copy, i.e. the same closed-form branch twice.
    uint32_t e_checks = 0, e_runs = 0;
    e_sweep({{1, 4}, {1, 5}, {1, 12}, {1, 13}, {1, 22}, {2, 2}, {3, 3}, {3, 4}, {5, 2}}, 7, true, &e_checks, &e_runs);
    std::cout << "  (E) TreeLocalForm against TreeModelChain on " << e_runs << " (cell, ell, F_p): d o d = 0, c_q, dim H^q to degree 7 ("
              << e_checks << " statements)" << std::endl;
    for (uint32_t n : {6u, 12u, 30u, 64u}) {
        const std::vector<uint32_t> h = [n] {
            std::vector<uint32_t> v(10, 0);
            const auto dims = integral_descent::ShippedFieldDims(n, 10007);
            v[0] = dims[0];
            v[1] = dims[1];
            return v;
        }();
        // (W) the W = 61 route on the tree == the K = 30 route
        const auto K30 = integral_descent::RunTreeLocalPieceSym(2, n, 4, h, 10007);
        const auto W61 = integral_descent::RunTreeLocalPieceSym(2, n, 4, h, 10007, 0, true);
        bool w = K30.all_certified() && W61.all_certified();
        for (uint32_t q = 1; w && q <= 4; ++q) w = K30.tors[q] == W61.tors[q];
        for (const auto& d : W61.d) w = w && (d.copy_of >= 0 || d.W == 61);
        check(w, "(W) n=" + std::to_string(n) + ": the tree at Z/2^61 (ModCtx<uint64_t>) == at Z/2^30");
    }
    std::cout << "  (W) Z/2^61 == Z/2^30 on the tree, n = 6, 12, 30, 64" << std::endl;
    // (I) the two rank sources on the tree at level N (source (i) at the good prime vs the Betti recursion)
    for (const auto& [N, n] : std::vector<std::pair<uint32_t, uint32_t>>{{11, 6}, {7, 4}, {13, 10}, {25, 4}, {9, 8}, {16, 6}}) {
        const auto c = integral_descent::LevelNClosedForm(N, n);
        std::vector<uint32_t> h(8, 0);
        h[0] = c.b0;
        h[1] = c.b1;
        const uint32_t g = integral_descent::GoodPrimeN(N, n);
        for (uint32_t ell : {2u, 3u}) {
            const auto Bt = integral_descent::RunTreeLocalPieceN(ell, N, n, 6, h, 0);
            const auto It = integral_descent::RunTreeLocalPieceN(ell, N, n, 6, h, g);   // ENGINE_CHECKs (i) == (ii) inside
            bool ok = Bt.all_certified() && It.all_certified();
            for (uint32_t q = 1; ok && q <= 6; ++q) ok = Bt.tors[q] == It.tors[q];
            for (uint32_t q = 0; ok && q < 6; ++q) ok = Bt.d[q].rank_Q == It.d[q].rank_Q;
            check(ok, "(I) ell=" + std::to_string(ell) + " " + cellname(N, n) + ": rank source (i) on THIS complex == the Betti recursion");
        }
    }
    std::cout << "  (I) source (i) (F_ell' reduction of the tree complex) == source (ii) (Betti) at 2 and 3 on six level-N cells" << std::endl;
    // (B) the Bredon tree at ell >= 5 == the direct Smith of C_eq (level 1)
    for (uint32_t n : {12u, 30u, 56u, 100u}) {
        for (const auto& [ell, count] : integral_model::DicksonPrimes(n)) {
            const auto ds = integral_descent::RunDirectSmithLevel1(n, ell, count);
            const auto tb = integral_descent::RunTreeBredonSym(n, ell, 0, n == 0 ? 1u : 0u);
            check(ds.certified && tb.certified && tb.tors1 == ds.tors1 && tb.free_rank1 == ds.free_rank1,
                  "(B) n=" + std::to_string(n) + " ell=" + std::to_string(ell) + ": the Bredon tree == the direct Smith of C_eq (" +
                      tb.tors1.to_string() + ")");
        }
    }
    std::cout << "  (B) the Bredon tree at ell >= 5 == the direct Smith of C_eq at the Dickson primes, n = 12, 30, 56, 100" << std::endl;
}

// ---- the ell >= 5 measurement (is the Bredon tree useful at level 1?) ----------------------------------------
void ell5_cost(uint32_t n, uint32_t repeats) {
    engine::tape::ThreadPool serial(1);
    engine::tape::ScopedConsumerPool pin(&serial);
    Stat td, tt, ts;
    const auto dick = integral_model::DicksonPrimes(n);
    for (uint32_t rep = 0; rep < repeats; ++rep)
        for (uint32_t k = 0; k < 3; ++k) {
            const uint32_t route = (k + rep) % 3;
            const auto t = clk::now();
            bool ok = true;
            for (const auto& [ell, count] : dick) {
                if (route == 0) ok = integral_descent::RunDirectSmithLevel1(n, ell, count).certified && ok;
                if (route == 1) ok = integral_descent::RunTreeBredonSym(n, ell, 0, 0).certified && ok;
                if (route == 2) ok = integral_descent::RunSparseCokernelSmith(1, n, ell).certified && ok;
            }
            check(ok, "(B-cost) every route certified at n=" + std::to_string(n));
            (route == 0 ? td : route == 1 ? tt : ts).v.push_back(since(t));
        }
    std::cout << "DESCENT_ELL5_COST n=" << n << " dickson=" << dick.size() << " repeats=" << repeats << " t_direct_Ceq=" << td.text()
              << " t_tree_bredon=" << tt.text() << " t_sparse=" << ts.text() << " gain_tree_vs_direct=" << fx(td.median() / std::max(1e-9, tt.median()))
              << "x gain_tree_vs_sparse=" << fx(ts.median() / std::max(1e-9, tt.median())) << "x" << std::endl;
}

// ---- the cost line ----------------------------------------------------------------------------------
struct CostCell {
    uint32_t N = 1, n = 0;
    bool both = true;
    uint32_t MN = 0;   // > 0: a level pair (N, MN, n) of D2 (AssemblePairMaps)
};

// The pair cost line (D2's AssemblePairMaps on the tree; the dense route beside it when its prediction is under the ceiling).
void pair_cost_cell(const CostCell& cell, uint32_t repeats, uint64_t cap) {
    engine::tape::ThreadPool serial(1);
    engine::tape::ScopedConsumerPool pin(&serial);
    const uint32_t DA = integral_descent::PsiIndex(cell.N) * (cell.n + 1), DB = integral_descent::PsiIndex(cell.MN) * (cell.n + 1);
    const uint64_t pt = integral_descent::maps::PredictPairRssTree(2, DA, DB, 4), pd = integral_descent::maps::PredictPairRss(2, DA, DB, 4);
    std::cout << "  DESCENT_PAIR_PRED N=" << cell.N << " MN=" << cell.MN << " n=" << cell.n << " S=" << DA + DB << " rss_pred_tree2=" << gib(pt)
              << " GiB rss_pred_dense2=" << gib(pd) << " GiB cap=" << gib(cap) << " GiB" << std::endl;
    Stat tt[2];
    uint64_t rss2[2] = {0, 0};
    std::string refusal[2];
    bool ran[2] = {false, false};
    integral_descent::maps::PairResult R[2];
    const bool dense_ok = cell.both && pd <= cap / 10 * 9;
    for (uint32_t rep = 0; rep < repeats; ++rep)
        for (uint32_t k = 0; k < 2; ++k) {
            const uint32_t route = (k + rep) % 2;
            if (route == 1 && !dense_ok) continue;
            integral_descent::ScopedDescentTree sc(route == 0);
            integral_descent::maps::PairOptions o;
            o.rss_cap_bytes = cap;
            R[route] = integral_descent::maps::AssemblePairMaps(cell.N, cell.MN / cell.N, cell.n, o);
            tt[route].v.push_back(R[route].t_total);
            for (const auto& pc : R[route].pieces)
                if (pc.ell == 2) rss2[route] = std::max<uint64_t>(rss2[route], pc.rss_meas > pc.rss_base ? pc.rss_meas - pc.rss_base : uint64_t(0));
            refusal[route] = R[route].refusal;
            ran[route] = true;
        }
    const std::string at = " at (N,MN,n)=(" + std::to_string(cell.N) + "," + std::to_string(cell.MN) + "," + std::to_string(cell.n) + ")";
    check(refusal[0].empty(), "(pair cost) the tree route is not refused" + at + " -- " + refusal[0]);
    if (ran[1] && refusal[1].empty())
        check(R[0].HA == R[1].HA && R[0].HB == R[1].HB && R[0].HAB == R[1].HAB && same_maps(R[0].res, R[1].res) &&
                  same_maps(R[0].cor, R[1].cor) && same_maps(R[0].j, R[1].j) && same_maps(R[0].dl, R[1].dl),
              "(pair cost) tree == dense" + at);
    check(rss2[0] <= pt, "(R) tree pair piece growth " + gib(rss2[0]) + " GiB <= predicted " + gib(pt) + " GiB" + at);
    for (uint32_t route = 0; route < 2; ++route) {
        if (!ran[route]) continue;
        std::cout << "DESCENT_PAIR_COST N=" << cell.N << " MN=" << cell.MN << " n=" << cell.n << " S=" << DA + DB
                  << " route=" << (route == 0 ? "tree" : "dense") << " threads=1 repeats=" << tt[route].v.size() << " t_total=" << tt[route].text()
                  << " rss_growth_two=" << gib(rss2[route]) << " rss_pred_two=" << gib(route == 0 ? pt : pd) << std::endl;
    }
    if (!R[0].refused) {
        std::ostringstream o;
        for (size_t q = 0; q < R[0].HAB.size(); ++q) o << (q ? " " : "") << "H^" << q << "(A,B)=" << R[0].HAB[q].to_string();
        std::string t = o.str();
        if (t.size() > 500) t = t.substr(0, 500) + " ...";
        std::cout << "  pair groups" << at << ": " << t << " exact_spots=" << R[0].exact_spots << std::endl;
    }
    if (!refusal[0].empty()) std::cout << "  " << refusal[0] << std::endl;
}

void cost_cell(const CostCell& cell, uint32_t repeats, uint64_t cap) {
    engine::tape::ThreadPool serial(1);
    engine::tape::ScopedConsumerPool pin(&serial);
    const bool level1 = cell.N == 1;
    const uint32_t D = level1 ? cell.n + 1 : integral_descent::PsiIndex(cell.N) * (cell.n + 1);
    const uint32_t dblk = level1 ? D : cell.n + 1;
    const uint64_t pred2 = integral_descent::PredictTreeLocalRss(2, D, dblk), pred3 = integral_descent::PredictTreeLocalRss(3, D, dblk);
    const uint64_t pred_dense2 = integral_descent::PredictLocalRss(2, D, 4);
    std::cout << "  DESCENT_PRED N=" << cell.N << " n=" << cell.n << " D=" << D << " rss_pred_tree2=" << gib(pred2)
              << " GiB rss_pred_tree3=" << gib(pred3) << " GiB rss_pred_dense2=" << gib(pred_dense2) << " GiB cap=" << gib(cap) << " GiB"
              << std::endl;
    if (pred2 > cap) {
        check(false, "(cost) the tree prediction is above the ceiling at " + cellname(cell.N, cell.n) + ": not started");
        return;
    }
    Stat tt[2], t2[2], t3[2], t5[2], tg[2];
    uint64_t rss[2] = {0, 0}, rss2[2] = {0, 0}, rss_whole[2] = {0, 0};
    std::vector<AbelianGroupType> H[2];
    std::string refusal[2];
    bool ran[2] = {false, false};
    const bool dense_ok = cell.both && pred_dense2 <= cap;
    for (uint32_t rep = 0; rep < repeats; ++rep)
        for (uint32_t k = 0; k < 2; ++k) {
            const uint32_t route = (k + rep) % 2;   // 0 tree, 1 dense
            if (route == 1 && !dense_ok) continue;
#if defined(__GLIBC__)
            if (level1) malloc_trim(0);
#endif
            const uint64_t base_whole = level1 ? integral_descent::CurrentRssBytes() : 0;   // level 1: the whole run's growth over it
            integral_descent::RssSampler rs;
            integral_descent::ScopedDescentTree sc(route == 0);
            if (level1) {
                const DescentResult R = integral_descent::AssembleDescent(cell.n);
                tt[route].v.push_back(R.t_total);
                t2[route].v.push_back(R.t_two);
                t3[route].v.push_back(R.t_three);
                t5[route].v.push_back(R.t_bad);
                tg[route].v.push_back(R.t_good);
                refusal[route] = R.refusal;
                H[route] = R.H;
            } else {
                integral_descent::LevelNOptions o;
                o.rss_cap_bytes = cap;
#if defined(__GLIBC__)
                malloc_trim(0);
#endif
                const uint64_t base = integral_descent::CurrentRssBytes();   // the retained heap of earlier work (e.g. under --all)
                const LevelNResult R = integral_descent::AssembleDescentN(cell.N, cell.n, o);
                tt[route].v.push_back(R.t_total);
                t2[route].v.push_back(R.t_two);
                t3[route].v.push_back(R.t_three);
                t5[route].v.push_back(R.t_ell);   // (PAR-LN) the sparse ell >= 5 stage
                tg[route].v.push_back(R.t_good);
                // the 2-adic piece's sampled peak as GROWTH over the base (the prediction's 200 MiB term absorbs allocator noise)
                const uint64_t grow = R.rss_meas_two > base ? R.rss_meas_two - base : 0;
                rss2[route] = std::max<uint64_t>(rss2[route], grow);
                refusal[route] = R.refusal;
                H[route] = R.H;
            }
            const uint64_t peak = rs.stop();
            rss[route] = std::max(rss[route], peak);
            if (level1) rss_whole[route] = std::max<uint64_t>(rss_whole[route], peak > base_whole ? peak - base_whole : 0);
            ran[route] = true;
        }
    check(refusal[0].empty(), "(cost) the tree route is not refused at " + cellname(cell.N, cell.n) + " -- " + refusal[0]);
    if (ran[1] && refusal[1].empty()) check(H[0] == H[1], "(cost) tree == dense, every H^q, at " + cellname(cell.N, cell.n));
    // (R) the measured RSS below the prediction.  Level N: the 2-adic piece's own sampler inside AssembleDescentN, as growth over
    // the process base before the call.  Level 1: AssembleDescent samples no piece, so the 2-adic tree piece is
    // run once more alone (after malloc_trim) and its GROWTH over the process base is compared with the prediction.
    uint64_t meas = rss2[0];
    if (level1) {
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        std::vector<uint32_t> h(8, 0);   // the Betti numbers first: the good-prime run is NOT part of the piece (PAR-L1: AssembleDescent's route)
        const auto dims = integral_descent::GoodPrimeFieldDims(cell.n, integral_descent::DefaultGoodPrime(cell.n));
        h[0] = dims[0];
        h[1] = dims[1];
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        const uint64_t base = integral_descent::CurrentRssBytes();
        integral_descent::RssSampler rs;
        // the production arguments (D5 audit m2): rank source (i) at the shipped good prime, as AssembleDescent passes it
        const auto P2 = integral_descent::RunTreeLocalPieceSym(2, cell.n, 4, h, integral_descent::DefaultGoodPrime(cell.n));
        const uint64_t peak = rs.stop();
        meas = peak > base ? peak - base : 0;
        check(P2.all_certified(), "(R) the 2-adic tree piece alone is certified at " + cellname(cell.N, cell.n));
        std::cout << "  DESCENT_RSS_TWO N=1 n=" << cell.n << " growth=" << gib(meas) << " GiB base=" << gib(base) << " GiB pred=" << gib(pred2)
                  << " GiB t=" << fx(P2.seconds) << " s" << std::endl;
    }
    check(meas <= pred2, "(R) tree RSS " + gib(meas) + " GiB <= predicted " + gib(pred2) + " GiB at " + cellname(cell.N, cell.n));
    if (level1 && ran[0]) {
        // (R) the WHOLE level-1 run on the tree route (D5 audit m2): the good prime included, against PredictDescentLevel1Rss
        const uint64_t pw = integral_descent::PredictDescentLevel1Rss(cell.n);
        std::cout << "  DESCENT_RSS_WHOLE N=1 n=" << cell.n << " growth=" << gib(rss_whole[0]) << " GiB pred=" << gib(pw) << " GiB (good prime + "
                  << "tree pieces)" << std::endl;
        check(rss_whole[0] <= pw, "(R) the whole level-1 tree descent's RSS growth " + gib(rss_whole[0]) + " GiB <= predicted " + gib(pw) +
                                      " GiB at " + cellname(cell.N, cell.n));
    }
    for (uint32_t route = 0; route < 2; ++route) {
        if (!ran[route]) continue;
        std::cout << "DESCENT_COST N=" << cell.N << " n=" << cell.n << " D=" << D << " route=" << (route == 0 ? "tree" : "dense")
                  << " threads=1 repeats=" << tt[route].v.size() << " t_descent=" << tt[route].text() << " t_two=" << t2[route].text()
                  << " t_three=" << t3[route].text()
                  << " t_ell5=" << t5[route].text() << " t_good=" << tg[route].text() << " rss_peak=" << gib(rss[route])
                  << (level1 ? std::string() : " rss_two=" + gib(rss2[route])) << " rss_pred_two="
                  << gib(route == 0 ? pred2 : pred_dense2)
                  // (PAR-LN audit A6) level N: t_ell5 is residual-only while HECKE_DESCENT_PAR_LN is on
                  << (level1 ? std::string() : std::string(" ell5_timing=") +
                                                   (integral_descent::par::Enabled(integral_descent::par::Lever::EllN) ? "residual-only" : "full"))
                  << std::endl;
    }
    if (ran[0] && ran[1])
        std::cout << "  DESCENT_GAIN N=" << cell.N << " n=" << cell.n << " total=" << fx(tt[1].median() / std::max(1e-9, tt[0].median()))
                  << "x two=" << fx(t2[1].median() / std::max(1e-9, t2[0].median()))
                  << "x three=" << fx(t3[1].median() / std::max(1e-9, t3[0].median())) << "x" << std::endl;
    if (!H[0].empty()) {
        std::ostringstream o;
        for (size_t q = 0; q < H[0].size(); ++q) o << (q ? " | " : "") << "H^" << q << "=" << H[0][q].to_string();
        std::string s = o.str();
        if (s.size() > 600) s = s.substr(0, 600) + " ...";
        std::cout << "  groups " << cellname(cell.N, cell.n) << ": " << s << std::endl;
    }
    if (!refusal[0].empty()) std::cout << "  " << refusal[0] << std::endl;
}

// ---- (NT) planted faults through the production tree route (D5 audit m4) -------------------------------------------------
// Each fault of integral_model::TreeFault (TreeCochains.h; validation-only, set through ScopedTreeFault) is run through the
// production calls -- AssembleDescent (level 1), AssembleDescentN (level N), AssemblePairMaps (D2) on the tree default -- and through
// the arms (E) and (L1).  A production call "catches" a fault when it throws (an ENGINE_CHECK of the piece), REFUSES, or returns
// other groups / maps than the unfaulted tree run.  The expected pattern per fault is asserted exactly (DESCENT5_report.md Sec. 8).
struct FaultPattern {
    integral_model::TreeFault f;
    const char* what;
    bool level1, levelN, d2, armE, armL1;
};
bool same_pair(const integral_descent::maps::PairResult& a, const integral_descent::maps::PairResult& b) {
    return a.HA == b.HA && a.HB == b.HB && a.HAB == b.HAB && same_maps(a.res, b.res) && same_maps(a.cor, b.cor) && same_maps(a.j, b.j) &&
           same_maps(a.dl, b.dl);
}
void arm_NT() {
    using integral_model::TreeFault;
    integral_descent::ScopedDescentTree route(true);
    const std::vector<uint32_t> n1 = {12, 30};
    const std::vector<std::pair<uint32_t, uint32_t>> cN = {{11, 6}, {7, 4}};
    const std::vector<std::array<uint32_t, 3>> cP = {{2, 4, 2}, {1, 3, 4}};
    std::vector<DescentResult> ref1;
    std::vector<LevelNResult> refN;
    std::vector<integral_descent::maps::PairResult> refP;
    for (uint32_t n : n1) ref1.push_back(integral_descent::AssembleDescent(n));
    for (const auto& c : cN) refN.push_back(integral_descent::AssembleDescentN(c.first, c.second));
    for (const auto& c : cP) refP.push_back(integral_descent::maps::AssemblePairMaps(c[0], c[1] / c[0], c[2]));
    bool refs = true;
    for (const auto& R : ref1) refs = refs && !R.refused && R.two.route == "tree" && R.three.route == "tree";
    for (const auto& R : refN) refs = refs && !R.refused && R.ran_two && R.two.route == "tree";
    for (const auto& R : refP) refs = refs && !R.refused;
    refs = refs && refN[1].ran_three && refN[1].three.route == "tree";
    for (const auto& R : refP) {
        bool t = false;
        for (const auto& pr : R.pieces) t = t || pr.route == "Iso2T";
        refs = refs && t;
    }
    if (!check(refs, "(NT) the unfaulted references run on the tree route and are not refused")) return;
    for (const auto& R : ref1) {
        bool b5 = false;
        for (const auto& b : R.bad) b5 = b5 || b.certified;
        refs = refs && b5;
    }
    check(refs, "(NT) the level-1 references have an ell >= 5 Bredon-tree piece (n = 12, 30)");
    // (expected) 1-5, 7: a group-changing term at 2 / 3 / >= 5 (5 only at level 1: the level-N ell >= 5 route is the sparse one, and
    // D2 has no Bredon tree); 6: a unit column scaling, invisible to every group, caught by (E)'s d o d = 0 only; 8: a unit change of a
    // block, invisible to the groups, caught by D2's presentations and (E); 9, 10: D2's reduction only (TreeLocalForm untouched, (E)
    // and (L1) clean).  (L1) works at level 1 over F_ell' on Sym^n (n even, c = 1): faults 5-8 change nothing there.
    const FaultPattern pats[] = {
        {TreeFault::Ell2OddEdgeSign, "ell=2 odd d^q: (1+S)x -> (1-S)x", true, true, true, true, true},
        {TreeFault::Ell2EvenZ, "ell=2 even d^q: -(1+c)z -> -(1-c)z", true, true, true, true, true},
        {TreeFault::Ell3OddNorm, "ell=3 odd d^q: 1+U+U^2 -> 1-U", true, true, true, true, true},
        {TreeFault::Ell2D0DropY, "ell=2 d^0: -y dropped", true, true, true, true, true},
        {TreeFault::BredonDropY, "ell>=5 d^0: -y dropped", true, false, false, true, false},
        {TreeFault::Ell2OddZSign, "ell=2 odd d^q: sign of the z column", false, false, false, true, false},
        {TreeFault::Ell3D0EdgeIdem, "ell=3 d^0: e_c for N_S/4", true, true, true, true, false},
        {TreeFault::Ell2OddNoEU, "ell=2 odd d^q: (1+c)y without e_U", false, false, true, true, false},
        {TreeFault::D2FibreNoGinv, "D2 ReduceFibreTree: no G_Y^{-1}", false, false, true, false, false},
        {TreeFault::D2DiffNoGinv, "D2 BuildRedTree: no G^{-1}", false, false, true, false, false},
    };
    uint32_t pinned = 0;
    for (const FaultPattern& fp : pats) {
        integral_model::ScopedTreeFault fault(fp.f);
        bool c1 = false, cN_ = false, cD = false, cE = false, cL = false;
        for (size_t i = 0; i < n1.size(); ++i) {
            try {
                const DescentResult R = integral_descent::AssembleDescent(n1[i]);
                c1 = c1 || R.refused || R.H != ref1[i].H;
            } catch (const std::logic_error&) {
                c1 = true;
            }
        }
        for (size_t i = 0; i < cN.size(); ++i) {
            try {
                const LevelNResult R = integral_descent::AssembleDescentN(cN[i].first, cN[i].second);
                cN_ = cN_ || R.refused || R.H != refN[i].H;
            } catch (const std::logic_error&) {
                cN_ = true;
            }
        }
        for (size_t i = 0; i < cP.size(); ++i) {
            try {
                const auto R = integral_descent::maps::AssemblePairMaps(cP[i][0], cP[i][1] / cP[i][0], cP[i][2]);
                cD = cD || R.refused || !same_pair(R, refP[i]);
            } catch (const std::logic_error&) {
                cD = true;
            }
        }
        cE = !e_sweep({{1, 12}, {1, 13}, {3, 3}}, 5, false);
        for (uint32_t i = 0; i < n1.size(); ++i)
            for (uint32_t ell : {2u, 3u}) {
                std::string why;
                try {
                    cL = cL || !integral_descent::TreeFieldDimsEqual(ell, n1[i], ref1[i].good_prime, ref1[i].rational_dims, 3, why);
                } catch (const std::logic_error&) {
                    cL = true;
                }
            }
        const auto yn = [](bool b) { return b ? "caught" : "clean"; };
        std::cout << "  (NT) fault " << static_cast<uint32_t>(fp.f) << " (" << fp.what << "): level1=" << yn(c1) << " levelN=" << yn(cN_)
                  << " D2=" << yn(cD) << " (E)=" << yn(cE) << " (L1)=" << yn(cL) << std::endl;
        check(c1 == fp.level1 && cN_ == fp.levelN && cD == fp.d2 && cE == fp.armE && cL == fp.armL1,
              std::string("(NT) the planted fault '") + fp.what + "' gives exactly the expected pattern (level1/levelN/D2/(E)/(L1) = " +
                  yn(fp.level1) + "/" + yn(fp.levelN) + "/" + yn(fp.d2) + "/" + yn(fp.armE) + "/" + yn(fp.armL1) + ")");
        // fault 6 is a unit scaling of a column: no group can move, so no production route can see it; (E)'s d o d = 0 does
        check(c1 || cN_ || cD || (fp.f == TreeFault::Ell2OddZSign && cE),
              std::string("(NT) the planted fault '") + fp.what + "' is caught on a production route (fault 6: by (E))");
        ++pinned;
    }
    // after the scopes: the production route is clean again
    bool clean = true;
    for (size_t i = 0; i < n1.size(); ++i) clean = clean && integral_descent::AssembleDescent(n1[i]).H == ref1[i].H;
    check(clean && integral_model::ActiveTreeFault() == integral_model::TreeFault::None, "(NT) no fault survives its scope");
    std::cout << "  (NT) " << pinned << " planted faults through AssembleDescent / AssembleDescentN / AssemblePairMaps on the tree, (E) and (L1)"
              << std::endl;
}

// ---- (G) D0's guards on the tree route (D5 audit m4), and the tree's refusal paths ----------------------------------------
void arm_G() {
    auto refuses = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const std::logic_error&) {
            return true;
        }
        return false;
    };
    const auto dims = integral_descent::ShippedFieldDims(16, 10007);
    std::vector<uint32_t> h(8, 0);
    h[0] = dims[0];
    h[1] = dims[1];
    const U32Ctx F(10007);
    uint32_t guards = 0;
    auto g = [&](bool ok, const std::string& what) {
        check(ok, "(G) " + what);
        ++guards;
    };
    using integral_model::TreeSymModule;
    g(refuses([&] {
          const U32Ctx c = U32Ctx::prime_power(3, 19);
          const auto M = TreeSymModule(c, 16), MF = TreeSymModule(F, 16);
          (void)integral_descent::RunTreeLocalPiece(2, M, c, 4, h, 10007, &MF, -1);
      }), "the 2-adic tree reduced into Z/3^19 is refused (the context's ell guard)");
    g(refuses([&] {
          const U32Ctx c(3);
          const auto M = TreeSymModule(c, 16), MF = TreeSymModule(F, 16);
          (void)integral_descent::RunTreeLocalPiece(3, M, c, 6, h, 10007, &MF, -1);
      }), "the 3-adic tree reduced into the field F_3 is refused");
    g(refuses([&] {
          const U32Ctx c = U32Ctx::prime_power(2, 30);
          const auto M = TreeSymModule(c, 16), MF = TreeSymModule(F, 18);
          (void)integral_descent::RunTreeLocalPiece(2, M, c, 4, h, 10007, &MF, -1);
      }), "a rank module of another dimension (Sym^18 for Sym^16) is refused");
    g(refuses([&] { (void)integral_descent::RunTreeLocalPieceSym(2, 16, 4, h, 7); }),
      "rank prime 7 at n = 16 (7 divides tors H^1: does not qualify for d^0) is refused at 2");
    g(refuses([&] { (void)integral_descent::RunTreeLocalPieceSym(3, 16, 6, h, 7); }), "rank prime 7 at n = 16 is refused at 3");
    g(refuses([&] { (void)integral_descent::RunTreeLocalPieceSym(2, 16, 4, h, 3); }),
      "rank prime 3 (divides q_2 = 3) is refused at 2 ([C, prop:isomodeltree](1)(d))");
    {
        std::vector<uint32_t> hb = h;
        hb[1] += 1;
        g(refuses([&] { (void)integral_descent::RunTreeLocalPieceSym(2, 16, 4, hb, 10007); }), "a wrong b_1 is refused (source (i) at F_10007)");
        g(refuses([&] { (void)integral_descent::RunTreeLocalPieceSym(2, 16, 4, hb, 0); }), "a wrong b_1 is refused (Betti recursion alone)");
        g(refuses([&] { (void)integral_descent::RunTreeLocalPieceSym(3, 16, 6, hb, 10007); }), "a wrong b_1 is refused at 3");
    }
    g(refuses([&] { (void)integral_descent::RunTreeBredonSym(16, 4, 0, 0); }), "the Bredon tree at a non-prime 4 is refused");
    // the stopping rule on the tree: K = 4 at 2, n = 16 (H^1 has Z/2^4).  Without the W = 61 route (sym_n = -1) d^0 is uncertified and
    // nothing is read off; with it, d^0 escalates and certifies, d^1.. stay at K = 4, groups == K = 30.
    const auto R30 = integral_descent::RunTreeLocalPieceSym(2, 16, 4, h, 10007);
    {
        const U32Ctx c4 = U32Ctx::prime_power(2, 4);
        const auto M = TreeSymModule(c4, 16), MF = TreeSymModule(F, 16);
        const auto L4 = integral_descent::RunTreeLocalPiece(2, M, c4, 4, h, 10007, &MF, -1);
        g(!L4.d[0].certified && L4.d[0].r_K < L4.d[0].rank_Q && L4.tors[1].is_zero() && !L4.d[0].escalated && !L4.all_certified(),
          "n=16 on the tree at Z/2^4 without escalation: d^0 uncertified, nothing read off");
        const auto L2 = integral_descent::RunTreeLocalPieceSym(3, 12, 6, [] {
            std::vector<uint32_t> v(8, 0);
            const auto d = integral_descent::ShippedFieldDims(12, 10007);
            v[0] = d[0];
            v[1] = d[1];
            return v;
        }(), 10007, 2);
        g(!L2.d[0].certified && L2.d[0].r_K < L2.d[0].rank_Q, "n=12 on the 3-adic tree at Z/3^2 (H^1 has Z/9): d^0 uncertified");
    }
    {
        const auto E = integral_descent::RunTreeLocalPieceSym(2, 16, 4, h, 10007, 4);
        bool esc = E.all_certified() && E.d[0].escalated && E.d[0].W == 61;
        for (uint32_t q = 1; q < E.d.size(); ++q) esc = esc && !E.d[q].escalated && (E.d[q].copy_of >= 0 || E.d[q].W == 4);
        for (uint32_t q = 0; q <= 4; ++q) esc = esc && E.tors[q] == R30.tors[q];
        g(esc, "K = 4 on the tree at n = 16: d^0 escalates to W = 61 and certifies, d^1.. at K = 4, groups == K = 30");
    }
    // the whole descent on the tree REFUSES (D0's (R) on the tree): Iso^(3)_T at K = 2, n = 12
    {
        integral_descent::ScopedDescentTree route(true);
        integral_descent::DescentOptions o;
        o.K_three = 2;
        const DescentResult R = integral_descent::AssembleDescent(12, o);
        g(R.refused && R.H.empty() && R.refusal.find("Iso^(3)_T (tree)") != std::string::npos,
          "n=12 with the 3-adic tree at Z/3^2: AssembleDescent is REFUSED on the tree route and reports no group");
    }
    std::cout << "  (G) " << guards << " guard / refusal / escalation cases on the tree route" << std::endl;
}

// ---- (RB) the rollback HECKE_DESCENT_TREE=0 inside --all (D5 audit m4) --------------------------------------------------------
// D0, D1 and D2's standard names test the dense route by design; with the tree default they run on the tree.  Here each runs again in
// this process under ScopedDescentTree(false) (their memos are keyed by the route), so the dense route keeps its value-level arms
// (oracles, Appendix A, refusals, periodicity, D2's exactness) in --all.
void arm_RB() {
    integral_descent::ScopedDescentTree route(false);
    std::cout << "  (RB) --- D0 / D1 / D2 standard names on the dense route (route=" << (integral_descent::DescentTreeEnabled() ? "tree" : "dense")
              << ") ---" << std::endl;
    const uint32_t r0 = IntegralDescentGroupsValidation();
    const uint32_t r1 = IntegralDescentLevelNValidation();
    const uint32_t r2 = IntegralDescentMapsValidation();
    check(!integral_descent::DescentTreeEnabled(), "(RB) the scope selects the dense route");
    check(r0 == 0, "(RB) IntegralDescentGroupsValidation passes on the dense route (HECKE_DESCENT_TREE=0)");
    check(r1 == 0, "(RB) IntegralDescentLevelNValidation passes on the dense route");
    check(r2 == 0, "(RB) IntegralDescentMapsValidation passes on the dense route");
    std::cout << "  (RB) D0 / D1 / D2 standard on the dense route: " << (r0 + r1 + r2 == 0 ? "all PASS" : "FAILED") << std::endl;
}

void print_effective(const char* name) {
    std::cout << "EFFECTIVE HeckeDescentTree route=" << (integral_descent::DescentTreeEnabled() ? "tree" : "dense")
              << " source=" << integral_descent::DescentTreeSource() << " (" << name << ")" << std::endl;
    std::cout << "EFFECTIVE HeckeDescentGoodSparse route=" << (integral_descent::DescentGoodSparseEnabled() ? "sparse(even n; odd n: shipped)" : "shipped")
              << " source=" << integral_descent::DescentGoodSparseSource() << " (" << name << ")" << std::endl;
    std::cout << "EFFECTIVE HeckeDescentGoodDense route=" << integral_descent::GoodPrimeRouteName(integral_descent::GoodPrimeRouteFor(2))
              << "(even n; odd n: shipped) source=" << integral_descent::DescentGoodDenseSource() << " (" << name << ")" << std::endl;
}

}  // namespace

uint32_t HeckeIsotropyModelTreeValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = clk::now();
    std::cout << "=== HeckeIsotropyModelTreeValidation (DESCENT D5: the tree bulk, [C] prop:isomodeltree) ===" << std::endl;
    print_effective("HeckeIsotropyModelTreeValidation");
    engine::tape::ThreadPool serial_pool(1);
    engine::tape::ScopedConsumerPool pin_all(&serial_pool);
    try {
        // (L1)
        const integral_descent::OracleTable T = integral_descent::LoadOracleTable(src_path("docs/implementation/slot24/descent_oracle_n128.tsv"));
        check(T.found && T.error.empty(), "(L1) descent_oracle_n128.tsv loads");
        std::vector<uint32_t> ns;
        for (uint32_t n = 0; n <= 40; ++n) ns.push_back(n);
        for (uint32_t n : {48u, 56u, 64u, 96u, 128u}) ns.push_back(n);
        const uint32_t c1 = arm_L1(ns, T.found ? &T : nullptr);
        std::cout << "  (L1) " << c1 << " level-1 cells: tree == dense == oracle (H^0..H^5)" << std::endl;
        // (LN)
        bool ok = false;
        const auto TN = load_oracle_n(ok);
        check(ok && !TN.empty(), "(LN) descentN_oracle.tsv loads");
        const uint32_t cN = arm_LN(TN, 100);
        std::cout << "  (LN) " << cN << " level-N cells (D <= 100): tree == dense == oracle (H^0..H^5)" << std::endl;
        arm_D2(40);
        arm_EWIB();
        integral_descent::RunTreeComparisonArms(6, false, check);
        arm_NT();
        arm_G();
        arm_RB();
        // cost (one repeat, small cells; the controlled cells are the slow name's)
        cost_cell({1, 100, true}, 1, 40ull << 30);
        cost_cell({16, 20, true}, 1, 40ull << 30);
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    std::cout << "HeckeIsotropyModelTreeValidation: checks=" << g_checks << " failing=" << g_fail << " wall=" << fx(since(t0)) << " s"
              << std::endl;
    return g_fail == 0 ? 0u : 1u;
}

uint32_t HeckeIsotropyModelTreeSlowValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = clk::now();
    std::cout << "=== HeckeIsotropyModelTreeSlowValidation (DESCENT D5: every oracle cell on both routes, degree 8, cost / reach) ==="
              << std::endl;
    print_effective("HeckeIsotropyModelTreeSlowValidation");
    engine::tape::ThreadPool serial_pool(1);
    engine::tape::ScopedConsumerPool pin_all(&serial_pool);
    try {
        uint32_t repeats = 5;
        if (const char* e = std::getenv("HECKE_DESCENT_REPEATS")) repeats = std::max(1, std::atoi(e));
        std::vector<CostCell> cells = {{1, 100, true}, {1, 200, true}, {1, 300, true}, {11, 20, true},
                                       {16, 20, true}, {35, 10, true}, {50, 6, true}, {4, 6, true, 16}, {10, 6, true, 20}};
        std::string arm = "default";
        if (const char* e = std::getenv("HECKE_DESCENT_TREE_CELLS")) {
            cells.clear();
            arm = "env";
            std::stringstream ss(e);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                CostCell c;
                if (!tok.empty() && tok[0] == 'p') {   // p<N>:<MN>:<n>[:both|tree]
                    std::stringstream ts(tok.substr(1));
                    std::string f;
                    std::vector<std::string> v;
                    while (std::getline(ts, f, ':')) v.push_back(f);
                    ENGINE_CHECK_MSG(v.size() >= 3, "HECKE_DESCENT_TREE_CELLS: pair tokens pN:MN:n[:both|tree]");
                    c.N = static_cast<uint32_t>(std::stoul(v[0]));
                    c.MN = static_cast<uint32_t>(std::stoul(v[1]));
                    c.n = static_cast<uint32_t>(std::stoul(v[2]));
                    c.both = v.size() < 4 || v[3] == "both";
                    ENGINE_CHECK_MSG(c.MN % c.N == 0 && c.MN > c.N, "HECKE_DESCENT_TREE_CELLS: N | MN, MN > N");
                    cells.push_back(c);
                    continue;
                }
                const size_t a = tok.find(':'), b = tok.find(':', a + 1);
                ENGINE_CHECK_MSG(a != std::string::npos, "HECKE_DESCENT_TREE_CELLS: tokens N:n[:both|tree]");
                c.N = static_cast<uint32_t>(std::stoul(tok.substr(0, a)));
                c.n = static_cast<uint32_t>(std::stoul(tok.substr(a + 1, b == std::string::npos ? std::string::npos : b - a - 1)));
                c.both = b == std::string::npos || tok.substr(b + 1) == "both";
                cells.push_back(c);
            }
        }
        const uint64_t cap = 40ull << 30;
        std::cout << "EFFECTIVE HeckeIsotropyModelTreeSlowValidation arm=" << arm << " cells=" << cells.size() << " repeats=" << repeats
                  << " rss_cap_gib=" << (cap >> 30) << std::endl;
        if (arm == "default") {
            const integral_descent::OracleTable T =
                integral_descent::LoadOracleTable(src_path("docs/implementation/slot24/descent_oracle_n128.tsv"));
            if (check(T.found && T.error.empty(), "(L1) descent_oracle_n128.tsv loads")) {
                std::vector<uint32_t> ns;
                for (const auto& kv : T.groups) ns.push_back(kv.first);
                const uint32_t c1 = arm_L1(ns, &T);
                std::cout << "  (L1) " << c1 << " level-1 cells: tree == dense == oracle" << std::endl;
            }
            bool ok = false;
            const auto TN = load_oracle_n(ok);
            if (check(ok && !TN.empty(), "(LN) descentN_oracle.tsv loads")) {
                const uint32_t cN = arm_LN(TN, 1u << 30);
                std::cout << "  (LN) " << cN << " level-N cells: tree == dense == oracle" << std::endl;
            }
            arm_D2(1u << 30);
            integral_descent::RunTreeComparisonArms(8, true, check);
            for (uint32_t n : {100u, 200u, 300u}) ell5_cost(n, repeats);
        }
        for (const auto& c : cells) {
            if (c.MN) pair_cost_cell(c, repeats, cap);
            else cost_cell(c, repeats, cap);
        }
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    std::cout << "HeckeIsotropyModelTreeSlowValidation: checks=" << g_checks << " failing=" << g_fail << " wall=" << fx(since(t0)) << " s"
              << std::endl;
    return g_fail == 0 ? 0u : 1u;
}
