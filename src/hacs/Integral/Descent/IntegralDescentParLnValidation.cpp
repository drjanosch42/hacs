// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// PAR-LN: AssembleDescentN's concurrent sparse ell >= 5 eliminations against the serial code and the dense oracle.  See the header.
#include "IntegralDescentParLnValidation.h"

#include "DescentAssembly.h"
#include "DescentLevelN.h"
#include "DescentParAsm.h"
#include "DescentRecord.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace {

namespace ID = integral_descent;
namespace par = integral_descent::par;
using clk = std::chrono::steady_clock;

uint64_t g_checks = 0;
uint32_t g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
std::string fx(double x, int p = 4) {
    std::ostringstream o;
    o << std::setprecision(p) << x;
    return o.str();
}
std::string gib(uint64_t b) { return fx(double(b) / double(1ull << 30), 3); }
std::string cell(uint32_t N, uint32_t n) { return "(" + std::to_string(N) + ", " + std::to_string(n) + ")"; }

const std::vector<uint32_t> kThreads = {1, 2, 7, 16};

// level-N cells with two to four bad primes >= 5 (Dickson and Borel counts, ell | N at (5, 24), (11, 24), (35, 30)), even n; nu_3 = 0 or not, so the
// ell = 3 shortcut and the Iso^(3) piece both occur
struct Cell {
    uint32_t N, n;
};
const std::vector<Cell> kCells = {{11, 12}, {7, 20}, {5, 24}, {11, 24}, {35, 30}};

// every field of a sparse run but the timings
bool SameRun(const ID::SparseSmithRun& a, const ID::SparseSmithRun& b) {
    return a.ell == b.ell && a.K == b.K && a.D == b.D && a.generators == b.generators && a.generator_nnz == b.generator_nnz &&
           a.dependent == b.dependent && a.deferred == b.deferred && a.final_nnz == b.final_nnz && a.peak_nnz == b.peak_nnz &&
           a.merge_ops == b.merge_ops && a.units == b.units && a.residual_rows == b.residual_rows && a.residual_cols == b.residual_cols &&
           a.rank_Q == b.rank_Q && a.r_K == b.r_K && a.certified == b.certified && a.tors1 == b.tors1 &&
           a.residual_refused == b.residual_refused;
}
bool SameRuns(const std::vector<ID::SparseSmithRun>& a, const std::vector<ID::SparseSmithRun>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!SameRun(a[i], b[i])) return false;
    return true;
}
bool SameResult(const ID::LevelNResult& a, const ID::LevelNResult& b) {
    return a.refused == b.refused && a.refusal == b.refusal && a.H == b.H && a.bad == b.bad && a.good_prime == b.good_prime &&
           a.good_h0 == b.good_h0 && a.good_h1 == b.good_h1 && SameRuns(a.sparse, b.sparse) && a.three_shortcut == b.three_shortcut &&
           (!a.three_shortcut || SameRun(a.three_sparse, b.three_sparse)) && a.ran_two == b.ran_two && a.ran_three == b.ran_three &&
           (!a.ran_two || a.two.tors == b.two.tors) && (!a.ran_three || a.three.tors == b.three.tors);
}

// ---- (S) sparse == dense ----------------------------------------------------------------------------------------------------------
void ArmSparseDense() {
    for (const Cell& c : kCells) {
        const auto bad = ID::BadPrimesN(c.N, c.n);
        check(bad.size() >= 2, "(S) " + cell(c.N, c.n) + " has at least two bad primes >= 5 (got " + std::to_string(bad.size()) + ")");
        std::vector<ID::SparseSmithRun> ref;
        for (const auto& [ell, s] : bad) {
            (void)s;
            const ID::SparseSmithRun one = ID::RunSparseCokernelSmith(c.N, c.n, ell);
            ID::SparseElimination e = ID::EliminateSparseCokernel(c.N, c.n, ell);
            check(e.run.residual_rows == e.residual.size(), "(S) " + cell(c.N, c.n) + " ell=" + std::to_string(ell) + ": residual rows held");
            const ID::SparseSmithRun two = ID::FinishSparseCokernelSmith(std::move(e));
            check(SameRun(one, two), "(S) " + cell(c.N, c.n) + " ell=" + std::to_string(ell) + ": split == RunSparseCokernelSmith");
            // the dense oracle: the same generators stacked, one Smith over Z/ell^K
            std::vector<uint32_t> dv = ID::DenseCokernelSmith(c.N, c.n, ell, one.K);
            std::vector<uint32_t> nonunit;
            for (uint32_t x : dv)
                if (x >= 1) nonunit.push_back(x);
            std::sort(nonunit.rbegin(), nonunit.rend());
            const auto it = one.tors1.torsion.find(ell);
            const std::vector<uint32_t> got = it == one.tors1.torsion.end() ? std::vector<uint32_t>{} : it->second;
            check(one.certified && dv.size() == one.r_K && nonunit == got,
                  "(S) " + cell(c.N, c.n) + " ell=" + std::to_string(ell) + ": sparse rank / valuations == dense (rank " +
                      std::to_string(one.r_K) + " vs " + std::to_string(dv.size()) + ")");
            ref.push_back(one);
        }
        for (uint32_t t : kThreads) {
            par::ScopedLever on(par::Lever::EllN, true);
            par::ScopedThreads st(t);
            ID::LevelNOptions o;
            o.skip_two = o.skip_three = true;   // the ell >= 5 stage only (the result is then refused: measurement)
            const ID::LevelNResult R = ID::AssembleDescentN(c.N, c.n, o);
            check(SameRuns(R.sparse, ref), "(S) " + cell(c.N, c.n) + " the threaded assembly's sparse runs == the one-call runs at " +
                                               std::to_string(t) + " workers");
        }
    }
}

// ---- (T) threaded == serial -------------------------------------------------------------------------------------------------------
void ArmThreaded() {
    for (const Cell& c : kCells) {
        ID::LevelNResult ref;
        {
            par::ScopedLever off(par::Lever::EllN, false);
            ref = ID::AssembleDescentN(c.N, c.n);
            check(!ref.refused, "(T) " + cell(c.N, c.n) + " serial: not refused -- " + ref.refusal);
            check(ref.ell_workers == 1 && !ref.ell_concurrent, "(T) " + cell(c.N, c.n) + " serial: one prime at a time");
        }
        // the concurrent items: the good prime, the bad primes, the ell = 3 shortcut (nu_3 = 0)
        const uint32_t m = 1u + static_cast<uint32_t>(ref.bad.size()) + (ref.three_shortcut ? 1u : 0u);
        for (uint32_t t : kThreads) {
            par::ScopedLever on(par::Lever::EllN, true);
            par::ScopedThreads st(t);
            const uint64_t before = par::GetCounters().ln_concurrent;
            const ID::LevelNResult R = ID::AssembleDescentN(c.N, c.n);
            const std::string at = "(T) " + cell(c.N, c.n) + " at " + std::to_string(t) + " workers";
            check(SameResult(R, ref), at + ": every group, refusal, good prime, sparse run and shortcut == serial");
            check(R.ell_items == m, at + ": " + std::to_string(R.ell_items) + " concurrent items (good + bad + shortcut)");
            check(R.ell_workers == std::min(t, m), at + ": " + std::to_string(R.ell_workers) + " items at a time == min(workers, #items)");
            check(R.ell_concurrent == (std::min(t, m) > 1), at + ": the concurrency flag");
            check((par::GetCounters().ln_concurrent > before) == (std::min(t, m) > 1), at + ": the ln_concurrent counter");
            check(R.ell_rss_pred_one == ID::PredictSparseElimRss(c.N, c.n) && R.ell_rss_pred_joint == R.ell_rss_pred_one * R.ell_workers,
                  at + ": the joint prediction = workers x one");
        }
    }
}

// ---- (P) the joint RSS rule ---------------------------------------------------------------------------------------------------------
void ArmRss() {
    const Cell c{35, 30};
    const uint64_t one = ID::PredictSparseElimRss(c.N, c.n);
    const uint32_t m = 1u + static_cast<uint32_t>(ID::BadPrimesN(c.N, c.n).size());   // + the good prime (skip_three: no shortcut)
    for (const auto& [mult, want] : std::vector<std::pair<double, uint32_t>>{{2.5, 2u}, {1.0, 1u}}) {
        ID::LevelNOptions o;
        o.rss_cap_bytes = static_cast<uint64_t>(double(one) * mult);
        o.skip_two = o.skip_three = true;
        ID::LevelNResult ref;
        {
            par::ScopedLever off(par::Lever::EllN, false);
            ref = ID::AssembleDescentN(c.N, c.n, o);
        }
        par::ScopedLever on(par::Lever::EllN, true);
        par::ScopedThreads st(16);
        const uint64_t lim0 = par::GetCounters().ln_serial_rss;
        const ID::LevelNResult R = ID::AssembleDescentN(c.N, c.n, o);
        const std::string at = "(P) " + cell(c.N, c.n) + " ceiling " + fx(mult) + " x the prediction";
        check(R.ell_workers == std::min(want, m), at + ": " + std::to_string(R.ell_workers) + " primes at a time");
        check(par::GetCounters().ln_serial_rss > lim0, at + ": counted as RSS-limited (ln_rss_limited)");
        check(SameResult(R, ref), at + ": == serial");
    }
    {
        // the measured growth of the elimination stage within the joint prediction (16 workers, default ceiling)
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        par::ScopedLever on(par::Lever::EllN, true);
        par::ScopedThreads st(16);
        ID::LevelNOptions o;
        o.skip_two = o.skip_three = true;
        const ID::LevelNResult R = ID::AssembleDescentN(c.N, c.n, o);
        check(R.ell_rss_growth <= R.ell_rss_pred_joint, "(P) " + cell(c.N, c.n) + " elimination growth " + gib(R.ell_rss_growth) +
                                                            " GiB <= joint prediction " + gib(R.ell_rss_pred_joint) + " GiB");
        uint32_t max_rows = 0, max_cols = 0;
        for (const auto& s : R.sparse) {
            max_rows = std::max(max_rows, s.residual_rows);
            max_cols = std::max(max_cols, s.residual_cols);
        }
        std::cout << "  (P) " << cell(c.N, c.n) << " workers=" << R.ell_workers << " growth=" << gib(R.ell_rss_growth) << " GiB pred_joint="
                  << gib(R.ell_rss_pred_joint) << " GiB residual_max=" << max_rows << "x" << max_cols << std::endl;
    }
    {
        // a ceiling below a residual: the same primes refused with the same text (the residual ceiling rule of record)
        const Cell r{13, 16};
        uint64_t smallest = ~0ull;
        {
            par::ScopedLever off(par::Lever::EllN, false);
            const ID::LevelNResult R0 = ID::AssembleDescentN(r.N, r.n);
            for (const auto& s : R0.sparse)
                if (s.residual_rows) smallest = std::min<uint64_t>(smallest, 4ull * s.residual_rows * s.residual_cols);
        }
        check(smallest != ~0ull && smallest > 1, "(P) " + cell(r.N, r.n) + " has a non-empty residual");
        if (smallest == ~0ull || smallest <= 1) return;
        ID::LevelNOptions o;
        o.rss_cap_bytes = smallest - 1;
        o.skip_two = o.skip_three = true;
        ID::LevelNResult ref;
        {
            par::ScopedLever off(par::Lever::EllN, false);
            ref = ID::AssembleDescentN(r.N, r.n, o);
        }
        bool any = false;
        for (const auto& s : ref.sparse) any = any || s.residual_refused;
        check(any && ref.refusal.find("dense residual is above the RSS ceiling") != std::string::npos,
              "(P) " + cell(r.N, r.n) + " a ceiling below the smallest residual refuses it (serial)");
        for (uint32_t t : {2u, 16u}) {
            par::ScopedLever on(par::Lever::EllN, true);
            par::ScopedThreads st(t);
            const ID::LevelNResult R = ID::AssembleDescentN(r.N, r.n, o);
            check(SameResult(R, ref), "(P) " + cell(r.N, r.n) + " the residual refusal == serial at " + std::to_string(t) + " workers");
        }
    }
}

// ---- (E) a failing elimination ---------------------------------------------------------------------------------------------------
void ArmException() {
    const Cell c{11, 12};
    ID::LevelNOptions o;
    o.K_ell = 1;   // EliminateSparseCokernel: ENGINE_CHECK K >= 2 at every prime
    std::string ref;
    {
        par::ScopedLever off(par::Lever::EllN, false);
        try {
            ID::AssembleDescentN(c.N, c.n, o);
        } catch (const std::exception& e) {
            ref = e.what();
        }
    }
    check(!ref.empty(), "(E) K = 1 throws (serial)");
    for (uint32_t t : kThreads) {
        par::ScopedLever on(par::Lever::EllN, true);
        par::ScopedThreads st(t);
        std::string got;
        try {
            ID::AssembleDescentN(c.N, c.n, o);
        } catch (const std::exception& e) {
            got = e.what();
        }
        check(got == ref, "(E) the same exception at " + std::to_string(t) + " workers");
    }
}

// ---- (R) the record ---------------------------------------------------------------------------------------------------------------
void ArmRecord() {
    const hecke::ScopedCoefficientRing z(0);
    for (const Cell& c : std::vector<Cell>{{11, 12}, {7, 20}}) {
        uint64_t ref;
        std::string body;
        {
            par::ScopedLever off(par::Lever::EllN, false);
            const auto R = ID::record::BuildDescentRecord(c.N, c.n);
            ref = R.digest();
            body = R.body();
            check(R.ok(), "(R) " + cell(c.N, c.n) + " serial record ok");
        }
        for (uint32_t t : {1u, 7u, 16u}) {
            par::ScopedLever on(par::Lever::EllN, true);
            par::ScopedThreads st(t);
            const auto R = ID::record::BuildDescentRecord(c.N, c.n);
            check(R.digest() == ref && R.body() == body,
                  "(R) " + cell(c.N, c.n) + " the record body and digest at " + std::to_string(t) + " workers");
        }
    }
}

// ---- the measurement hook --------------------------------------------------------------------------------------------------------
struct CostCell {
    uint32_t N = 0, n = 0;
    bool full = false, no2 = false;   // :full the whole call; :no2 everything but the 2-adic piece (the ell = 3 shortcut / Iso^(3) run)
};
std::vector<CostCell> ParseCells(const char* e) {
    std::vector<CostCell> out;
    std::stringstream ss(e);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        CostCell c;
        const size_t a = tok.find(':');
        if (a == std::string::npos) continue;
        c.N = static_cast<uint32_t>(std::strtoul(tok.substr(0, a).c_str(), nullptr, 10));
        const size_t b = tok.find(':', a + 1);
        c.n = static_cast<uint32_t>(std::strtoul(tok.substr(a + 1, b == std::string::npos ? std::string::npos : b - a - 1).c_str(), nullptr, 10));
        c.full = b != std::string::npos && tok.substr(b + 1) == "full";
        c.no2 = b != std::string::npos && tok.substr(b + 1) == "no2";
        if (c.N >= 1) out.push_back(c);
    }
    return out;
}

void CostRun(const CostCell& c) {
    const ID::LevelNCounts k = ID::LevelNClosedForm(c.N, c.n);
    const uint32_t T = par::Threads();
    auto run = [&](bool on, uint32_t threads, bool full, bool no2) {
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        par::ScopedLever lv(par::Lever::EllN, on);
        par::ScopedThreads st(threads);
        ID::LevelNOptions o;
        o.skip_two = !full;
        o.skip_three = !full && !no2;
        return ID::AssembleDescentN(c.N, c.n, o);
    };
    const ID::LevelNResult off = run(false, T, c.full, c.no2);
    const ID::LevelNResult one = run(true, 1, false, c.no2);   // one item at a time: the single-item growth (calibration)
    const ID::LevelNResult on = run(true, T, c.full, c.no2);
    check(SameResult(on, off), "(cost) " + cell(c.N, c.n) + " threaded == serial");
    double fill = 0;
    uint32_t rr = 0, rc = 0;
    double t_res_max = 0;
    for (const auto& s : on.sparse) {
        fill = std::max(fill, double(s.peak_nnz) / (double(k.D) * k.d));
        rr = std::max(rr, s.residual_rows);
        rc = std::max(rc, s.residual_cols);
        t_res_max = std::max(t_res_max, s.residual_seconds);
    }
    check(one.ell_rss_growth <= one.ell_rss_pred_one, "(cost) " + cell(c.N, c.n) + " one elimination's growth " + gib(one.ell_rss_growth) +
                                                           " GiB <= " + gib(one.ell_rss_pred_one) + " GiB");
    check(on.ell_rss_growth <= on.ell_rss_pred_joint, "(cost) " + cell(c.N, c.n) + " joint growth " + gib(on.ell_rss_growth) + " GiB <= " +
                                                          gib(on.ell_rss_pred_joint) + " GiB");
    std::ostringstream o;
    o << "PARLN_COST N=" << c.N << " n=" << c.n << " D=" << k.D << " d=" << k.d << " bad=" << on.bad.size() << " threads=" << T
      << " items=" << on.ell_items << " workers=" << on.ell_workers << " t_good=" << fx(on.t_good) << " t_ell_serial=" << fx(off.t_ell)
      << " t_sparse_serial=" << fx(off.t_good + off.t_ell + (off.three_shortcut ? off.t_three : 0.0)) << " t_stage_par=" << fx(on.t_ell_elim)
      << " t_residual=" << fx(on.t_ell_residual) << " t_residual_max=" << fx(t_res_max)
      << " gain=" << fx((off.t_good + off.t_ell + (off.three_shortcut ? off.t_three : 0.0)) /
                        std::max(1e-9, on.t_ell_elim + on.t_ell + (on.three_shortcut ? on.t_three : 0.0))) << "x fill_max=" << fx(fill) << " residual_max=" << rr << "x" << rc
      << " growth_one=" << gib(one.ell_rss_growth) << " pred_one=" << gib(on.ell_rss_pred_one) << " growth_joint=" << gib(on.ell_rss_growth)
      << " pred_joint=" << gib(on.ell_rss_pred_joint) << " GiB";
    if (c.full) {
        const double before = off.t_total, after = on.t_total;
        o << " t_total_serial=" << fx(before) << " t_total_par=" << fx(after) << " t_two=" << fx(on.t_two) << " t_three=" << fx(on.t_three)
          << " share_ell_serial=" << fx(100.0 * off.t_ell / std::max(1e-9, before), 3) << "% share_sparse_serial="
          << fx(100.0 * (off.t_good + off.t_ell + (off.three_shortcut ? off.t_three : 0.0)) / std::max(1e-9, before), 3)
          << "% share_sparse_par=" << fx(100.0 * (on.t_ell_elim + on.t_ell + (on.three_shortcut ? on.t_three : 0.0)) / std::max(1e-9, after), 3)
          << "%";
    }
    std::cout << o.str() << std::endl;
}

}  // namespace

uint32_t IntegralDescentParLnValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = clk::now();
    std::cout << "=== IntegralDescentParLnValidation (PAR-LN: AssembleDescentN's concurrent sparse ell >= 5 eliminations == serial == dense) ==="
              << std::endl;
    if (const char* e = std::getenv("HECKE_DESCENT_PARLN_CELLS")) {
        const std::vector<CostCell> cells = ParseCells(e);
        std::cout << "EFFECTIVE IntegralDescentParLnValidation arm=cells cells=" << cells.size() << std::endl;
        for (const CostCell& c : cells) {
            try {
                CostRun(c);
            } catch (const std::exception& x) {
                check(false, "(cost) " + cell(c.N, c.n) + " threw: " + x.what());
            }
        }
        std::cout << par::EffectiveLine() << std::endl;
        std::cout << "IntegralDescentParLnValidation: " << g_checks << " checks, " << g_fail << " failing, " << fx(since(t0)) << " s" << std::endl;
        return g_fail == 0 ? 0u : 1u;
    }
    const char* arms[] = {"S", "T", "P", "E", "R"};
    void (*fns[])() = {ArmSparseDense, ArmThreaded, ArmRss, ArmException, ArmRecord};
    for (size_t i = 0; i < 5; ++i) {
        const auto ta = clk::now();
        const uint64_t c0 = g_checks;
        const uint32_t f0 = g_fail;
        try {
            fns[i]();
        } catch (const std::exception& e) {
            check(false, std::string("(") + arms[i] + ") threw: " + e.what());
        }
        std::cout << "  (" << arms[i] << ") checks=" << (g_checks - c0) << " failing=" << (g_fail - f0) << " " << fx(since(ta)) << " s" << std::endl;
    }
    std::cout << par::EffectiveLine() << std::endl;
    std::cout << "IntegralDescentParLnValidation: " << g_checks << " checks, " << g_fail << " failing, " << fx(since(t0)) << " s" << std::endl;
    return g_fail == 0 ? 0u : 1u;
}
