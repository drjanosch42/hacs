// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT D2: IntegralDescentMapsValidation / IntegralDescentMapsSlowValidation.  See the header for the arms.
// [C] = docs/latex/gl2_contract.tex (v1.23.0-pre.22), cited by label (prop:descentmaps incl. (2)(g), prop:descentN, prop:descentNthree).
// The (T) table d2_maps_oracle.tsv (87 cells) has a second, independent oracle: the D2 code audit's d2audit/d2audit_oracle.py (no code
// shared with the tree or with d2_maps_oracle.py / cd3_descentmaps_check.py), which agrees on every row (DESCENT2_report.md, "Audit fixes").
#include "IntegralDescentMapsValidation.h"

#include "DescentAssembly.h"
#include "DescentLevelN.h"
#include "DescentMaps.h"
#include "DescentOracle.h"
#include "DescentTree.h"

#include "../../../Engine/EngineRuntime.h"
#include "../../../Engine/ThreadPool.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

using integral_descent::AbelianGroupType;
using integral_descent::maps::PairOptions;
using integral_descent::maps::PairResult;
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
std::string fx(double x, int p = 4) {
    std::ostringstream o;
    o << std::setprecision(p) << x;
    return o.str();
}
std::string gib(uint64_t b) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << double(b) / double(1ull << 30);
    return o.str();
}
using Cell = std::array<uint32_t, 3>;   // N, MN, n
std::string cellname(const Cell& c) {
    return "(N,MN,n)=(" + std::to_string(c[0]) + "," + std::to_string(c[1]) + "," + std::to_string(c[2]) + ")";
}

// ---- the oracle table -----------------------------------------------------------------
struct Oracle {
    bool found = false;
    std::string error;
    std::map<Cell, std::map<std::pair<std::string, uint32_t>, AbelianGroupType>> rows;
};
std::string oracle_path() {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/docs/implementation/slot24/d2_maps_oracle.tsv";
#else
    return "docs/implementation/slot24/d2_maps_oracle.tsv";
#endif
}
Oracle LoadOracle() {
    Oracle T;
    std::ifstream f(oracle_path());
    if (!f) {
        T.error = "cannot open " + oracle_path();
        return T;
    }
    T.found = true;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("N\t", 0) == 0) continue;
        std::istringstream is(line);
        Cell c{};
        std::string obj, g;
        uint32_t q = 0;
        if (!(is >> c[0] >> c[1] >> c[2] >> obj >> q >> g)) {
            T.error = "malformed row: " + line;
            return T;
        }
        T.rows[c][{obj, q}] = integral_descent::ParseOracleGroup(g);
    }
    return T;
}

const AbelianGroupType* lookup(const PairResult& R, const std::string& obj, uint32_t q) {
    auto mt = [&](const std::vector<integral_descent::maps::MapTypes>& v, const std::string& part) -> const AbelianGroupType* {
        if (q >= v.size()) return nullptr;
        if (part == "ker") return &v[q].ker;
        if (part == "im") return &v[q].im;
        return &v[q].coker;
    };
    if (obj == "HA") return q < R.HA.size() ? &R.HA[q] : nullptr;
    if (obj == "HB") return q < R.HB.size() ? &R.HB[q] : nullptr;
    if (obj == "HAB") return q < R.HAB.size() ? &R.HAB[q] : nullptr;
    const size_t u = obj.find('_');
    if (u == std::string::npos) return nullptr;
    const std::string m = obj.substr(0, u), part = obj.substr(u + 1);
    if (m == "res") return mt(R.res, part);
    if (m == "cor") return mt(R.cor, part);
    if (m == "j") return mt(R.j, part);
    if (m == "dl") return mt(R.dl, part);
    return nullptr;
}

uint32_t g_exact = 0, g_ident = 0, g_cells = 0, g_rows = 0, g_x3 = 0;

PairResult run(const Cell& c, const PairOptions& o = {}) {
    return integral_descent::maps::AssemblePairMaps(c[0], c[1] / c[0], c[2], o);
}

void print_cell(const PairResult& R) {
    std::ostringstream o;
    o << "  " << cellname({R.N, R.MN, R.n}) << " [A:B]=" << R.index << (R.normal ? " normal" : "") << " S={";
    for (size_t i = 0; i < R.primes.size(); ++i) o << (i ? "," : "") << R.primes[i];
    o << "} t=" << fx(R.t_total, 3) << " s:";
    for (uint32_t q = 0; q <= 4; ++q) o << " H^" << q << "(A,B)=" << R.HAB[q].to_string();
    std::cout << o.str() << std::endl;
}

// (T) + (E) + (I) + (X: 24 H^{>=3}(A,B) = 0) on one cell
void compare_cell(const Oracle& T, const Cell& c, bool verify_d1, bool print) {
    PairOptions o;
    o.verify_ends_with_d1 = verify_d1;
    const PairResult R = run(c, o);
    if (!check(!R.refused, "(T) not refused at " + cellname(c) + (R.refused ? ": " + R.refusal : ""))) return;
    ++g_cells;
    g_exact += R.exact_spots;
    g_ident += R.identity_checks;
    check(R.exact_spots > 0, "(E) exactness spots checked at " + cellname(c));
    for (const auto& [key, g] : T.rows.at(c)) {
        const AbelianGroupType* x = lookup(R, key.first, key.second);
        ++g_rows;
        check(x && *x == g, "(T) " + cellname(c) + " " + key.first + "^" + std::to_string(key.second) + ": descent " +
                                (x ? x->to_string() : std::string("(missing)")) + " == oracle " + g.to_string());
    }
    for (uint32_t q = 3; q <= 5; ++q) {
        bool killed = R.HAB[q].free_rank == 0;
        for (const auto& [p, ex] : R.HAB[q].torsion)
            for (uint32_t a : ex)
                if (!((p == 2 && a <= 3) || (p == 3 && a <= 1))) killed = false;
        check(killed, "(X) 24 H^" + std::to_string(q) + "(A,B) = 0 at " + cellname(c));
        ++g_x3;
    }
    if (R.normal) check(R.five_term_checked, "(F) the five-term statement was checked at the normal pair " + cellname(c));
    if (print) print_cell(R);
}

std::vector<Cell> parse_cells(const char* e) {
    std::vector<Cell> cells;
    std::stringstream ss(e);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const size_t a = tok.find(':'), b = tok.find(':', a + 1);
        ENGINE_CHECK_MSG(a != std::string::npos && b != std::string::npos, "HECKE_DESCENT_PAIR_CELLS: tokens N:M:n");
        const uint32_t N = static_cast<uint32_t>(std::stoul(tok.substr(0, a)));
        const uint32_t M = static_cast<uint32_t>(std::stoul(tok.substr(a + 1, b - a - 1)));
        const uint32_t n = static_cast<uint32_t>(std::stoul(tok.substr(b + 1)));
        cells.push_back({N, N * M, n});
    }
    return cells;
}

uint32_t DAof(const Cell& c) { return integral_descent::PsiIndex(c[0]) * (c[2] + 1); }
uint32_t DBof(const Cell& c) { return integral_descent::PsiIndex(c[1]) * (c[2] + 1); }

// DESCENT2_PREDICT for every cell first (rule 19); a cell predicted above the ceiling runs once and must be refused before
// anything is built; the others run in `repeats` ROTATED rounds (round r starts at cell r mod k), serial pinned pool.
void cost_cells(const std::vector<Cell>& cells, uint32_t repeats, uint64_t cap) {
    // (D5) DESCENT2_COST measures D2's dense design of record: the dense Iso^(2) pair piece pinned (the default is now the tree)
    integral_descent::ScopedDescentTree dense_route(false);
    PairOptions o;
    o.rss_cap_bytes = cap;
    std::vector<uint64_t> pred(cells.size());
    std::vector<size_t> live;
    for (size_t i = 0; i < cells.size(); ++i) {
        const Cell& c = cells[i];
        const uint32_t DA = integral_descent::PsiIndex(c[0]) * (c[2] + 1), DB = integral_descent::PsiIndex(c[1]) * (c[2] + 1);
        pred[i] = integral_descent::maps::PredictPairRss(2, DA, DB, 4);
        std::cout << "  DESCENT2_PREDICT " << cellname(c) << " D_A=" << DA << " D_B=" << DB << " rss_pred_gib=" << gib(pred[i])
                  << " cap_gib=" << (cap >> 30) << std::endl;
        if (pred[i] > cap / 10 * 9) {
            const PairResult R = run(c, o);
            check(R.refused && R.pieces.empty(), "(cost) a cell predicted above the ceiling is REFUSED before any piece: " + cellname(c));
            std::cout << "  DESCENT2_COST " << cellname(c) << " " << R.refusal << std::endl;
        } else {
            live.push_back(i);
        }
    }
    std::vector<std::vector<double>> t(cells.size()), t2(cells.size());
    std::vector<PairResult> last(cells.size());
    for (uint32_t r = 0; r < repeats && !live.empty(); ++r)
        for (size_t k = 0; k < live.size(); ++k) {
            const size_t i = live[(k + r) % live.size()];
            last[i] = run(cells[i], o);
            if (!check(!last[i].refused, "(cost) not refused at " + cellname(cells[i]) + ": " + last[i].refusal)) return;
            t[i].push_back(last[i].t_total);
            t2[i].push_back(last[i].pieces.empty() ? 0.0 : last[i].pieces[0].seconds);
        }
    for (size_t i : live) {
        auto& v = t[i];
        auto& v2 = t2[i];
        std::sort(v.begin(), v.end());
        std::sort(v2.begin(), v2.end());
        const PairResult& L = last[i];
        std::ostringstream o2;
        o2 << "  DESCENT2_COST " << cellname(cells[i]) << " repeats=" << v.size() << " median_s=" << fx(v[v.size() / 2]) << " min_s=" << fx(v.front())
           << " max_s=" << fx(v.back()) << " iso2_median_s=" << fx(v2[v2.size() / 2]) << " exact_spots=" << L.exact_spots;
        for (const auto& p : L.pieces)
            o2 << " | ell=" << p.ell << " " << p.route << " t=" << fx(p.seconds, 3) << " rss=" << gib(p.rss_meas) << " base=" << gib(p.rss_base)
               << (p.rss_pred ? " pred=" + gib(p.rss_pred) + " struct=" +
                                    gib(integral_descent::maps::PairRssStructural(p.ell, DAof(cells[i]), DBof(cells[i]), p.q_top))
                              : std::string())
               << " smith=" << p.smith_forms << " prec>=" << p.min_prec;
        std::cout << o2.str() << std::endl;
        // the model predicts the piece's own peak (incl. a 256 MiB allowance), so the comparison is the growth over the
        // process RSS at the piece's start (the retained heap of the oracle arm and earlier cells excluded)
        for (const auto& p : L.pieces)
            if (p.rss_pred)
                check(p.rss_meas - std::min(p.rss_meas, p.rss_base) <= p.rss_pred,
                      "(cost) piece RSS growth <= prediction, ell=" + std::to_string(p.ell) + ", " + cellname(cells[i]));
    }
}

}  // namespace

uint32_t IntegralDescentMapsValidation() {
    g_checks = 0;
    g_fail = 0;
    g_exact = g_ident = g_cells = g_rows = g_x3 = 0;
    const auto t0 = clk::now();
    std::cout << "=== IntegralDescentMapsValidation (DESCENT D2: res, cor, the level cone and its long exact sequence over Z) ==="
              << std::endl;
    std::cout << "EFFECTIVE HeckeDescentTree route=" << (integral_descent::DescentTreeEnabled() ? "tree" : "dense")
              << " source=" << integral_descent::DescentTreeSource() << " (IntegralDescentMapsValidation)" << std::endl;
    engine::tape::ThreadPool serial_pool(1);
    engine::tape::ScopedConsumerPool pin_all(&serial_pool);
    try {
        const Oracle T = LoadOracle();
        if (check(T.found && T.error.empty() && !T.rows.empty(),
                  "(T) the oracle table loads: " + (T.error.empty() ? oracle_path() : T.error))) {
            uint32_t ncells = 0;
            for (const auto& kv : T.rows) {
                const Cell& c = kv.first;
                if (integral_descent::PsiIndex(c[1]) * (c[2] + 1) > 40) continue;
                compare_cell(T, c, false, false);
                ++ncells;
            }
            std::cout << "  (T) " << ncells << " oracle cells (psi(MN)(n+1) <= 40; " << T.rows.size() << " in the slow name), " << g_rows
                      << " groups compared; (E) " << g_exact << " exact spots; (I, F) " << g_ident << " identities; (X) " << g_x3
                      << " relative groups killed by 24" << std::endl;
        }
        // (X) the examples of (2)(b)
        {
            const PairResult R = run({4, 8, 0});
            if (check(!R.refused, "(X) (4, 8, 0) not refused")) {
                AbelianGroupType z2;
                z2.add_torsion(2, 1);
                check(R.HAB[3] == z2, "(X) H^3(Gamma_0(4), Gamma_0(8)) = Z/2 at n = 0: " + R.HAB[3].to_string());
                print_cell(R);
            }
            const PairResult R2 = run({1, 11, 4});
            if (check(!R2.refused, "(X) (1, 11, 4) not refused")) {
                AbelianGroupType z3;
                z3.add_torsion(3, 1);
                check(R2.HAB[3] == z3, "(X) H^3(Gamma, Gamma_0(11)) = Z/3 at n = 4: " + R2.HAB[3].to_string());
                print_cell(R2);
            }
        }
        // (F) normal pairs with their five-term line
        for (const Cell& c : std::vector<Cell>{{2, 4, 0}, {2, 4, 2}, {3, 9, 0}, {1, 2, 2}}) {
            const PairResult R = run(c);
            if (check(!R.refused, "(F) not refused at " + cellname(c) + (R.refused ? ": " + R.refusal : "")) && R.normal) {
                check(R.five_term_checked, "(F) five-term checked at " + cellname(c));
                std::cout << "  (F) " << cellname(c) << ": " << R.five_term << std::endl;
            }
        }
        // (P) periodicity: Iso^(2) to degree 6
        for (const Cell& c : std::vector<Cell>{{1, 2, 0}, {2, 4, 2}}) {
            PairOptions o;
            o.q_top_two = 6;
            const PairResult P = run(c, o), R = run(c);
            if (check(!P.refused && !R.refused, "(P) not refused at " + cellname(c))) {
                check(P.HA[5] == P.HA[3] && P.HB[5] == P.HB[3] && P.HAB[5] == P.HAB[3],
                      "(P) H^5 = H^3 (A, B, (A,B)) computed at degree 6, " + cellname(c));
                check(P.res[5].ker == P.res[3].ker && P.res[5].coker == P.res[3].coker && P.cor[5].ker == P.cor[3].ker &&
                          P.cor[5].coker == P.cor[3].coker,
                      "(P) res^5, cor^5 = res^3, cor^3 (kernel, cokernel), " + cellname(c));
                bool same = true;
                for (uint32_t q = 0; q <= 5; ++q)
                    same = same && P.HAB[q] == R.HAB[q] && P.res[q].coker == R.res[q].coker && P.cor[q].ker == R.cor[q].ker;
                check(same, "(P) degree 6 == degree 4 + periodicity on every q <= 5, " + cellname(c));
                std::cout << "  (P) " << cellname(c) << ": exact spots " << P.exact_spots << " (degree 6) vs " << R.exact_spots << std::endl;
            }
        }
        // (D) the end groups against D1
        for (const Cell& c : std::vector<Cell>{{1, 11, 4}, {2, 6, 2}, {3, 9, 2}}) {
            PairOptions o;
            o.verify_ends_with_d1 = true;
            const PairResult R = run(c, o);
            check(!R.refused, "(D) the end groups == D1's AssembleDescentN at " + cellname(c) + (R.refused ? ": " + R.refusal : ""));
        }
        // (N) negative controls
        {
            PairOptions o;
            o.test_break_alpha = true;
            const PairResult R = run({2, 4, 2}, o);
            check(R.refused && R.HAB.empty() && R.res.empty(), "(N) alpha perturbed: REFUSED, results empty");
            std::cout << "  (N) alpha perturbed: " << R.refusal.substr(0, 200) << std::endl;
            PairOptions o2;
            o2.test_rank_offset = 1;
            const PairResult R2 = run({2, 4, 2}, o2);
            check(R2.refused && R2.HAB.empty(), "(N) a wrong rational rank of res^1: REFUSED");
            std::cout << "  (N) rank + 1: " << R2.refusal.substr(0, 200) << std::endl;
            PairOptions o2m;
            o2m.test_rank_offset = -1;
            const PairResult R2m = run({2, 4, 2}, o2m);
            check(R2m.refused && R2m.HAB.empty() && R2m.refusal.find("not certified") != std::string::npos,
                  "(N) rank_Q(res^1) - 1: the Smith certification REFUSES (r_K > the claimed rank)");
            std::cout << "  (N) rank - 1: " << R2m.refusal.substr(0, 200) << std::endl;
            // (D2 audit F5) an admissible but false claim (rank_Q + 1 on the invariants map, below the free-rank bound) must reach
            // the Smith certification, not the trivial guard
            PairOptions o2i;
            o2i.test_inv_rank_offset = 1;
            const PairResult R2i = run({2, 4, 2}, o2i);
            check(R2i.refused && R2i.HAB.empty() && R2i.refusal.find("invariants map") != std::string::npos &&
                      R2i.refusal.find("not certified") != std::string::npos &&
                      R2i.refusal.find("rank input above") == std::string::npos,
                  "(N) rank_Q(invariants) + 1 (admissible, false): the Smith certification REFUSES (r_K < the claimed rank)");
            std::cout << "  (N) invariants rank + 1: " << R2i.refusal.substr(0, 200) << std::endl;
            PairOptions o3;
            o3.rss_cap_bytes = 1ull << 20;
            const PairResult R3 = run({2, 4, 2}, o3);
            check(R3.refused && R3.pieces.empty() && R3.HAB.empty(), "(N) a 1 MiB ceiling: REFUSED before any piece starts");
            std::cout << "  (N) ceiling: " << R3.refusal.substr(0, 200) << std::endl;
        }
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    std::cout << "IntegralDescentMapsValidation: checks=" << g_checks << " failing=" << g_fail << " wall=" << fx(since(t0)) << " s"
              << std::endl;
    return g_fail == 0 ? 0u : 1u;
}

uint32_t IntegralDescentMapsSlowValidation() {
    g_checks = 0;
    g_fail = 0;
    g_exact = g_ident = g_cells = g_rows = g_x3 = 0;
    const auto t0 = clk::now();
    std::cout << "=== IntegralDescentMapsSlowValidation (DESCENT D2: the oracle on every cell, controlled costs) ===" << std::endl;
    engine::tape::ThreadPool serial_pool(1);
    engine::tape::ScopedConsumerPool pin_all(&serial_pool);
    try {
        uint32_t repeats = 5;
        if (const char* e = std::getenv("HECKE_DESCENT_REPEATS")) repeats = std::max(1, std::atoi(e));
        std::vector<Cell> cells = {{1, 11, 10}, {5, 25, 4}, {6, 12, 6}, {4, 16, 6}};
        std::string arm = "default";
        if (const char* e = std::getenv("HECKE_DESCENT_PAIR_CELLS")) {
            cells = parse_cells(e);
            arm = "env";
        }
        const uint64_t cap = 40ull << 30;
        std::cout << "EFFECTIVE IntegralDescentMapsSlowValidation arm=" << arm << " cells=" << cells.size() << " repeats=" << repeats
                  << " rss_cap_gib=" << (cap >> 30) << std::endl;
        if (arm == "default") {
            const Oracle T = LoadOracle();
            if (check(T.found && T.error.empty(), "(T) the oracle table loads")) {
                for (const auto& kv : T.rows) compare_cell(T, kv.first, true, true);
                std::cout << "  (T) " << g_cells << " oracle cells, " << g_rows << " groups compared (end groups also against D1); (E) "
                          << g_exact << " exact spots; (I, F) " << g_ident << " identities" << std::endl;
            }
        }
        cost_cells(cells, repeats, cap);
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    std::cout << "IntegralDescentMapsSlowValidation: checks=" << g_checks << " failing=" << g_fail << " wall=" << fx(since(t0)) << " s"
              << std::endl;
    return g_fail == 0 ? 0u : 1u;
}
