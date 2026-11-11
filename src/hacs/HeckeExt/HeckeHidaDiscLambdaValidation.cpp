// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeHidaDiscLambdaValidation -- slot-26 package Z6 (= H2): [U] part 7 (sec:u-H2, docs/latex/hida_hecke_algebra.tex 0.7.4-pre.1;
// def:u-DF, lem:u-DF, lem:u-DFread, pr:u-H2 (U25)) on the eleven ramified ordinary non-Eisenstein blocks of prop:u-loewy(4) at the
// ten cells (n, ell), (68, 157) carrying two.  Design docs/implementation/slot26/Z6_design.md v0.2 Sec. 4.3; report Z6_report.md.
//
// Per cell: the forms side (HeckeHidaLambda ComputeFormsWeight) mod ell^5 at k_j = k0 + j(ell - 1), j = 0..4 (and the held-out
// j = 5 at ell <= 227; under ALL at every cell), one prime list P = SturmPrimes(ell, k_5) for every weight of the family; the
// non-reduced residual systems at k0 (FindNonReducedSystems: one per cell, two at (68, 157) with rbar2 = 96, 111 -- oracle); per
// block and weight the certified block (ResidualBlock: rank 2, plus a nilpotency consistency check that holds by construction),
// T2 generates, det U a unit, rank e S_k constant; per block D_F mod (ell^3, Z^3) through T2 (lem:u-DFread(R)): v(e0) = 1, ring
// consistency (a code check), the (2, 2) subset self-check from j = 2, 3, 4 (a code check, NOT an error detector: once the five-node
// interpolant is ell-integral it agrees for any values, Z6_code_audit.md F1), q-independence through a second generator t'
// (unit-square ratio, same case and s_F), the held-out weight (thm:
// u-interp(3), the full Newton polynomial, mod ell^5), the case word (A | B: findings, not conditions), s_F mod ell^2, the rungs,
// the internal v-reading at the nodes (a code check).  Oracles: (30, 67) chi(T2 | block at 32) = X^2 - 39960 X - 2235350016 mod 67^5
// and det U = 56 mod 67 (prop:u-loewy(4)); (68, 157) rbar2 in {96, 111}; the eleven rows (rbar2, D, case, s_F mod ell^2, rung2,
// rung3, lambda) pinned as oracle constants (Z6_report.md Sec. 3, [U] rmk:u-H2data; audit F2), incl. (36, 181) case B, lambda = 2:
// any change of a row fails ("pinned=11/11").  Must-refuse controls (54, 67) at k = 56 and (46, 383) at
// k = 48 (a T2 double root that T3 separates into a conjugate pair: no non-reduced system).  Tampers: (i) +ell^4 on Delta at j = 2 of
// (30, 67), detector the held-out check; (ii) +ell^2 at j = 3, detector the integrality refusal of the five-node interpolation (a
// single-node error of valuation < 4 makes the degree-4 Newton coefficient non-integral; thm:u-interp(2)); (iii) rbar2 + 1 at
// (30, 67), ResidualBlock refuses (rank 0).
//
// HECKE_HIDA_H2_ALL=1 (read-only): the held-out weight at 349 and 479 too; the known answer (547, 486) of pr:u-H2(c) (the cuspidal
// block of the Eisenstein system: chi(T2 | block) = H1's f_k mod 547^5 at the five weights 486 .. 2670, case A, rung2 = 55086); the
// rung check (pr:u-H2(b)) at ell in {67, 71, 73} in case A, the block at k^(2)_F cut with P' = SturmPrimes(ell, k0) (fallback: the
// family's P, forms with SturmPrimes(ell, k)).  HECKE_HIDA_H2_PROBE=1 (read-only; replaces the arms): the two-job cost probe of
// design Sec. 6.3, serial -- the (32, 479) family and (38, 73) at k = 4360 with P' -- per-job walls and peak RSS, no gate.
// Prints "EFFECTIVE HeckeHidaDiscLambdaValidation arm=<DEFAULT|ALL|PROBE> cells=.. blocks=.. heldout=.. rung=..", one HIDA_H2_BLOCK
// line per block and the gate "HIDA_H2 blocks=11 cells=10 ... pinned=11/11 ... verdict=OK".  Writes nothing, no golden object, no
// digest of record (the gate's fnv= is not pinned: audit Q8; the rows themselves are, above).
#include "HeckeHidaDiscLambdaValidation.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "HeckeHidaDiscLambda.h"
#include "HeckeHidaEisLambda.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/ParallelFor.h"
#include "../Util/ValidationCheck.h"

namespace {

namespace D = hecke_ext::disclambda;
namespace L = hecke_ext::lambda;
namespace E = hecke_ext::eislambda;
using L::u128;
using L::Vec;
using L::ZMod;

oc_util::Checker Check("  [FAIL] ");

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
long PeakRssMiB() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024;
}
bool EnvOne(const char* n) {
    const char* v = std::getenv(n);
    return v && std::string(v) == "1";
}

struct Cell {
    uint32_t n;
    uint64_t ell;
    uint32_t blocks;
};
// prop:u-loewy(4): the ten cells, table order of design Sec. 1.1
const std::vector<Cell>& Cells() {
    static const std::vector<Cell> v = {{30, 67, 1},  {26, 131, 1}, {26, 139, 1}, {32, 479, 1}, {36, 181, 1},
                                        {36, 349, 1}, {38, 73, 1},  {44, 227, 1}, {52, 71, 1},  {68, 157, 2}};
    return v;
}
// H1's f_k = chi(T2 | the cuspidal Eisenstein block) mod 547^7 at k = 486 + 546 j (slot25/H1_report.md Sec. 2), low -> high
const std::vector<std::pair<uint64_t, uint64_t>>& H1fk() {
    static const std::vector<std::pair<uint64_t, uint64_t>> v = {
        {12956597795357322337ull, 4308378410345699387ull}, {9310097133968604166ull, 885804927614767839ull},
        {2780669233491663746ull, 10887889745225521499ull}, {11102034818952670343ull, 9419618947729934553ull},
        {3132602299545580460ull, 12326472980649907912ull}};
    return v;
}

// The eleven blocks (Z6_report.md Sec. 3; [U] 0.7.4-pre.2 rmk:u-H2data): D = [e0, e1, e2] = D_{T(2)} mod (ell^3, Z^3), Z = X - x_{k0};
// case, s_F mod ell^2, rung2, rung3 (case A; 0 = "-"), lambda.  Pinned (Z6_code_audit.md F2): a change of any field fails.
struct PinnedRow {
    uint32_t n;
    uint64_t ell, rbar2;
    const char* D;
    char cas;
    uint64_t sF, rung2, rung3;
    const char* lambda;
};
const std::vector<PinnedRow>& Pinned() {
    static const std::vector<PinnedRow> v = {
        {30, 67, 14, "[67670,81595,32906]", 'A', 2079, 2012, 293864, "1"},
        {26, 131, 52, "[2028797,1769733,956238]", 'A', 8181, 13028, 643138, "1"},
        {26, 139, 30, "[903500,1799153,2635989]", 'A', 18627, 3754, 617578, "1"},
        {32, 479, 472, "[61589820,24027696,6939087]", 'A', 163322, 24412, 43298230, "1"},
        {36, 181, 178, "[4850076,877126,5105143]", 'B', 0, 0, 0, "2"},
        {36, 349, 171, "[21863454,40091683,4242504]", 'A', 553, 63722, 22046534, "1"},
        {38, 73, 61, "[248054,218975,21318]", 'A', 27, 976, 69304, "1"},
        {44, 227, 5, "[5546291,7722249,7001584]", 'A', 17357, 38014, 4706496, "1"},
        {52, 71, 1, "[97341,43135,167397]", 'A', 3084, 1664, 255134, "1"},
        {68, 157, 96, "[2470552,253158,98053]", 'A', 22529, 23314, 145774, "1"},
        {68, 157, 111, "[880770,2234752,1764920]", 'A', 5411, 24094, 2938642, "1"}};
    return v;
}

uint32_t Wt(uint32_t k0, uint64_t ell, uint32_t j) { return k0 + j * static_cast<uint32_t>(ell - 1); }

std::string PolyText(const Vec& p) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < p.size(); ++i) os << (i ? "," : "") << L::U128Text(p[i]);
    os << "]";
    return os.str();
}
std::string Hex(uint64_t h) {
    std::ostringstream os;
    os << "0x" << std::hex << h;
    return os.str();
}

struct Family {
    uint32_t n = 0, k0 = 0, blocks = 0, nw = 0;
    uint64_t ell = 0;
    std::vector<uint64_t> P;
    bool known = false;   // (547, 486)
};

struct BlockOut {
    const Family* fam = nullptr;
    D::ResidualSystem sys;
    bool rank2 = false, gen = false, gen_t2 = false, ordinary = false, s_const = false;
    std::string t2;
    Vec disc, disc2, tau, nu;
    Vec chi2_k0;           // chi(T2 | block at k0) mod ell^5
    uint64_t detU_k0 = 0;  // mod ell
    std::vector<Vec> chi2; // per weight, mod ell^5
    D::DiscTruncation d, d2;
    bool ring = false, qind = false, stage = false;
    std::string line;
};

// One block through the weights of its family (steps 3, 4 and the Lambda stage).
BlockOut RunBlock(const Family& f, const std::vector<const L::FormsWeight*>& fws, const D::ResidualSystem& sys) {
    BlockOut b;
    b.fam = &f;
    b.sys = sys;
    const ZMod R5(f.ell, 5);
    std::vector<D::BlockAt> blocks;
    try {
        for (const auto* fw : fws) blocks.push_back(D::ResidualBlock(*fw, f.P, sys));
    } catch (const std::exception& e) {
        Check(false, "ResidualBlock at (" + std::to_string(f.n) + ", " + std::to_string(f.ell) + ") rbar2=" +
                         std::to_string(sys.res[0]) + " threw: " + e.what());
        return b;
    }
    b.rank2 = true;
    b.gen = b.ordinary = b.s_const = b.gen_t2 = true;
    b.t2 = D::SecondGenerator(blocks[0], f.P, f.ell);
    for (size_t j = 0; j < blocks.size(); ++j) {
        const D::BlockAt& bk = blocks[j];
        b.rank2 = b.rank2 && bk.rank == 2;
        b.s_const = b.s_const && fws[j]->S.s == fws[0]->S.s;
        const D::Quad q = D::ReadQuad(bk.ops.at("T2"), R5);
        const D::Quad q2 = D::ReadQuad(D::GeneratorMatrix(bk, b.t2, R5), R5);
        b.gen = b.gen && q.gen;
        b.gen_t2 = b.gen_t2 && q2.gen;
        b.disc.push_back(q.disc);
        b.disc2.push_back(q2.disc);
        b.tau.push_back(q.tau);
        b.nu.push_back(q.nu);
        b.chi2.push_back(L::Charpoly(bk.ops.at("T2"), R5));
        const ZMod F(f.ell, 1);
        const Vec cu = L::Charpoly(bk.ops.at("U"), F);
        const uint64_t detU = static_cast<uint64_t>(cu[0] % f.ell);   // det = chi(0) for rank 2
        b.ordinary = b.ordinary && detU != 0;
        if (j == 0) {
            b.chi2_k0 = b.chi2.back();
            b.detU_k0 = detU;
        }
    }
    try {
        b.d = D::ReadDisc(f.ell, f.k0, b.disc);
        b.d2 = D::ReadDisc(f.ell, f.k0, b.disc2);
        b.ring = D::RingConsistent(f.ell, D::Trunc33(f.ell, f.k0, b.tau), D::Trunc33(f.ell, f.k0, b.nu), b.d.e);
        b.qind = D::UnitSquareRatio(f.ell, b.d.e[0], b.d2.e[0]) && b.d.cas == b.d2.cas &&
                 (b.d.cas != 'A' || b.d.sF_mod_l2 == b.d2.sF_mod_l2);
        b.stage = true;
    } catch (const std::exception& e) {
        Check(false, "the Lambda stage at (" + std::to_string(f.n) + ", " + std::to_string(f.ell) + ") threw: " + e.what());
        return b;
    }
    std::ostringstream os;
    os << "HIDA_H2_BLOCK ell=" << f.ell << " cell=(" << f.n << "," << f.ell << ") k0=" << f.k0 << " a=" << f.k0 % (f.ell - 1)
       << " rbar2=" << sys.res[0] << " rank=" << (b.rank2 ? 2 : 0) << " gen=" << (b.gen ? "T2" : "none") << " t2=" << b.t2
       << " K=3 m=3 N=5 W=5 centre=x_" << f.k0 << " D=" << PolyText(b.d.e) << " v_e0=" << b.d.v_e0 << " case=" << b.d.cas
       << " s_F_mod_l2=" << (b.d.cas == 'A' ? std::to_string(b.d.sF_mod_l2) : "-")
       << " rung2=" << (b.d.cas == 'A' ? std::to_string(b.d.rung2) : "-") << " rung3=" << (b.d.cas == 'A' ? std::to_string(b.d.rung3) : "-")
       << " lambda=" << b.d.lambda_word << " subset_selfcheck=" << (b.d.subset_ok ? "ok" : "FAIL") << " q_indep=" << (b.qind ? "ok" : "FAIL")
       << " heldout=" << (b.d.heldout < 0 ? "skip" : b.d.heldout ? "ok" : "FAIL");
    const std::string body = os.str();
    b.line = body + " digest=" + Hex(engine::FnvBytes(engine::kFnvOffset64, body));
    return b;
}

// The two-job probe (design Sec. 6.3), serial.
uint32_t RunProbe() {
    std::cout << "EFFECTIVE HeckeHidaDiscLambdaValidation arm=PROBE cells=2 blocks=2 heldout=0 rung=1" << std::endl;
    const double t0 = Now();
    // job 1: the (32, 479) family, five weights, the family's P
    const uint64_t l1 = 479;
    const uint32_t k01 = 34;
    const std::vector<uint64_t> P1 = E::SturmPrimes(l1, Wt(k01, l1, 5));
    std::vector<L::FormsWeight> f1;
    double s1 = 0;
    for (uint32_t j = 0; j < 5; ++j) {
        const double a = Now();
        f1.push_back(L::ComputeFormsWeight(l1, Wt(k01, l1, j), 5, P1));
        const double w = Now() - a;
        s1 += w;
        std::cout << "HIDA_H2_PROBE job=1 ell=479 k=" << Wt(k01, l1, j) << " d=" << f1.back().dim_S << " L=" << f1.back().L
                  << " s=" << f1.back().S.s << " wall_s=" << w << std::endl;
    }
    {
        const D::SystemSearch ss = D::FindNonReducedSystems(f1[0], P1);
        Check(ss.systems.size() == 1, "probe: one non-reduced system at (32, 479)");
        if (ss.systems.size() == 1) {
            Family fam;
            fam.n = 32, fam.k0 = k01, fam.ell = l1, fam.blocks = 1, fam.nw = 5, fam.P = P1;
            std::vector<const L::FormsWeight*> ws;
            for (const auto& w : f1) ws.push_back(&w);
            const double a = Now();
            const BlockOut b = RunBlock(fam, ws, ss.systems[0]);
            std::cout << b.line << std::endl;
            std::cout << "HIDA_H2_PROBE job=1 lambda_stage_s=" << Now() - a << std::endl;
        }
    }
    std::cout << "HIDA_H2_PROBE job=1 total_wall_s=" << s1 << " peak_rss_mib=" << PeakRssMiB() << std::endl;
    // job 2: (38, 73) at k = 40 + 72 * 60 = 4360, P' = SturmPrimes(73, 40)
    const uint64_t l2 = 73;
    const uint32_t k02 = 40, k2 = 40 + 72 * 60;
    const std::vector<uint64_t> P2 = E::SturmPrimes(l2, k02), Pfam = E::SturmPrimes(l2, Wt(k02, l2, 5));
    const L::FormsWeight w0 = L::ComputeFormsWeight(l2, k02, 5, Pfam);
    const D::SystemSearch ss = D::FindNonReducedSystems(w0, Pfam);
    Check(ss.systems.size() == 1, "probe: one non-reduced system at (38, 73)");
    const double a = Now();
    const L::FormsWeight w2 = L::ComputeFormsWeight(l2, k2, 5, P2);
    const double w = Now() - a;
    std::cout << "HIDA_H2_PROBE job=2 ell=73 k=" << k2 << " d=" << w2.dim_S << " L=" << w2.L << " s=" << w2.S.s << " primes=" << P2.size()
              << " wall_s=" << w << " peak_rss_mib=" << PeakRssMiB() << std::endl;
    if (ss.systems.size() == 1) {
        try {
            const D::BlockAt bk = D::ResidualBlock(w2, P2, ss.systems[0]);
            const D::Quad q = D::ReadQuad(bk.ops.at("T2"), ZMod(l2, 5));
            std::cout << "HIDA_H2_PROBE job=2 rank=" << bk.rank << " v_disc=" << q.v_disc << std::endl;
        } catch (const std::exception& e) {
            std::cout << "HIDA_H2_PROBE job=2 ResidualBlock refused: " << e.what() << std::endl;
        }
    }
    std::cout << "HIDA_H2_PROBE done wall_s=" << Now() - t0 << " peak_rss_mib=" << PeakRssMiB() << " checks=" << Check.checks
              << " failing=" << Check.fails << std::endl;
    return Check.exit_code();
}

}  // namespace

uint32_t HeckeHidaDiscLambdaValidation() {
    Check.reset();
    const double t0 = Now();
    if (EnvOne("HECKE_HIDA_H2_PROBE")) return RunProbe();
    const bool all = EnvOne("HECKE_HIDA_H2_ALL");
    const unsigned threads = engine::threads::configured(engine::threads::kCap16);

    // ---- the plan: families, controls, (ALL) the known answer
    std::vector<Family> fams;
    for (const Cell& c : Cells()) {
        Family f;
        f.n = c.n, f.ell = c.ell, f.k0 = c.n + 2, f.blocks = c.blocks;
        f.nw = (all || c.ell <= 227) ? 6 : 5;
        f.P = E::SturmPrimes(c.ell, Wt(f.k0, c.ell, 5));
        fams.push_back(f);
    }
    if (all) {
        Family f;
        f.n = 484, f.ell = 547, f.k0 = 486, f.blocks = 1, f.nw = 5, f.known = true;
        f.P = E::SturmPrimes(547, Wt(486, 547, 4));
        fams.push_back(f);
    }
    const std::vector<std::pair<uint32_t, uint64_t>> controls = {{54, 67}, {46, 383}};
    uint32_t n_held = 0, n_blocks_plan = 0;
    for (const auto& f : fams) {
        n_blocks_plan += f.blocks;
        if (!f.known && f.nw == 6) n_held += f.blocks;
    }
    std::cout << "EFFECTIVE HeckeHidaDiscLambdaValidation arm=" << (all ? "ALL" : "DEFAULT") << " cells=" << fams.size()
              << " blocks=" << n_blocks_plan << " heldout=" << n_held << " rung=" << (all ? 3 : 0) << " threads=" << threads << std::endl;

    // ---- phase 1: every forms job, in parallel (index-addressed; cost-descending order for the balance only)
    std::vector<D::FormsJob> jobs;
    for (const auto& f : fams)
        for (uint32_t j = 0; j < f.nw; ++j) jobs.push_back(D::FormsJob{f.ell, Wt(f.k0, f.ell, j), 5, f.P});
    for (const auto& c : controls) jobs.push_back(D::FormsJob{c.second, c.first + 2, 5, E::SturmPrimes(c.second, c.first + 2)});
    std::stable_sort(jobs.begin(), jobs.end(), [](const D::FormsJob& a, const D::FormsJob& b) {
        return static_cast<double>(a.ell) * a.k * a.k > static_cast<double>(b.ell) * b.k * b.k;
    });
    std::map<std::pair<uint64_t, uint32_t>, size_t> at;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const bool fresh = at.emplace(std::make_pair(jobs[i].ell, jobs[i].k), i).second;
        if (!fresh) throw std::logic_error("HeckeHidaDiscLambdaValidation: duplicate (ell, k) job");
    }
    const double tf = Now();
    std::vector<L::FormsWeight> fw;
    try {
        fw = D::ComputeFormsParallel(jobs, threads);
    } catch (const std::exception& e) {
        Check(false, std::string("ComputeFormsParallel threw: ") + e.what());
        std::cout << "HIDA_H2 blocks=0 cells=0 verdict=FAIL" << std::endl;
        return 1;
    }
    const double forms_wall = Now() - tf;
    double forms_cpu = 0, max_job = 0;
    std::string max_job_at;
    for (size_t i = 0; i < fw.size(); ++i) {
        forms_cpu += fw[i].wall_s;
        if (fw[i].wall_s > max_job) {
            max_job = fw[i].wall_s;
            max_job_at = "(" + std::to_string(jobs[i].ell) + "," + std::to_string(jobs[i].k) + ")";
        }
        Check(fw[i].stable_ok && fw[i].S.comm_ok && fw[i].M.comm_ok,
              "forms data stable / commuting at (" + std::to_string(jobs[i].ell) + ", " + std::to_string(jobs[i].k) + ")");
    }
    auto FW = [&](uint64_t ell, uint32_t k) -> const L::FormsWeight& { return fw.at(at.at({ell, k})); };

    // ---- per family: the residual systems at k0, then every block
    std::vector<BlockOut> outs;   // the eleven (+ the known answer under ALL, flagged by fam->known)
    uint32_t oracle_ok = 0, oracle_n = 0;
    for (const auto& f : fams) {
        const L::FormsWeight& w0 = FW(f.ell, f.k0);
        D::SystemSearch ss;
        try {
            ss = D::FindNonReducedSystems(w0, f.P);
        } catch (const std::exception& e) {
            Check(false, "FindNonReducedSystems at (" + std::to_string(f.n) + ", " + std::to_string(f.ell) + ") threw: " + e.what());
            continue;
        }
        Check(ss.systems.size() == f.blocks, "(" + std::to_string(f.n) + ", " + std::to_string(f.ell) + "): " +
                                                 std::to_string(ss.systems.size()) + " non-reduced systems, expected " +
                                                 std::to_string(f.blocks) + " (prop:u-loewy(4))");
        for (const auto& s : ss.systems) Check(s.rank == 2, "residual system of rank 2 at (" + std::to_string(f.n) + ", " + std::to_string(f.ell) + ")");
        if (f.ell == 157) {   // oracle: the two double roots 96, 111 (prop:u-loewy(4))
            ++oracle_n;
            const bool ok = ss.systems.size() == 2 && ss.systems[0].res[0] == 96 && ss.systems[1].res[0] == 111;
            oracle_ok += Check(ok, "(68, 157): the residual T2-values are {96, 111}");
        }
        std::vector<const L::FormsWeight*> ws;
        for (uint32_t j = 0; j < f.nw; ++j) ws.push_back(&FW(f.ell, Wt(f.k0, f.ell, j)));
        for (const auto& s : ss.systems) outs.push_back(RunBlock(f, ws, s));
    }

    // ---- per-block checks (the case words are findings, not conditions)
    uint32_t nb = 0, n_rank = 0, n_gen = 0, n_v1 = 0, n_ring = 0, n_sub = 0, n_q = 0, n_held_ok = 0, n_held_run = 0, cA = 0, cB = 0, cU = 0;
    uint64_t fnv = engine::kFnvOffset64;
    const BlockOut* b3067 = nullptr;
    const L::FormsWeight* w3067 = nullptr;
    for (const auto& b : outs) {
        const Family& f = *b.fam;
        const std::string at_s = "(" + std::to_string(f.n) + ", " + std::to_string(f.ell) + ") rbar2=" + std::to_string(b.sys.res[0]);
        if (!b.line.empty()) std::cout << b.line << std::endl;
        if (f.known) continue;
        ++nb;
        if (!b.stage) continue;
        fnv = engine::FnvBytes(fnv, b.line);
        n_rank += Check(b.rank2, "rank 2 at every weight " + at_s);
        Check(b.s_const, "rank e S_k constant along the class " + at_s);
        Check(b.ordinary, "det U a unit on the block " + at_s);
        n_gen += Check(b.gen, "T2 generates at every weight " + at_s + " (lem:u-DF(2))");
        Check(b.gen_t2, "the second generator " + b.t2 + " generates at every weight " + at_s);
        n_v1 += Check(b.d.v_e0 == 1, "v(e0) = 1 " + at_s + " (prop:u-loewy(4): a defect, not a finding)");
        n_ring += Check(b.ring, "ring consistency tau~^2 - 4 nu~ = D~ " + at_s);
        n_sub += Check(b.d.subset_ok, "the (2, 2) subset self-check (a code check, not an error detector) " + at_s);
        n_q += Check(b.qind, "q-independence (" + b.t2 + ") " + at_s);
        Check(b.d.reading_ok, "the internal v-reading at the nodes " + at_s);
        if (b.d.heldout >= 0) {
            ++n_held_run;
            n_held_ok += Check(b.d.heldout == 1, "held-out weight k5 = " + std::to_string(Wt(f.k0, f.ell, 5)) + " " + at_s + " (thm:u-interp(3))");
        }
        if (b.d.cas == 'A') ++cA;
        if (b.d.cas == 'B') ++cB;
        if (b.d.lambda_word == "undecided") ++cU;
        if (f.ell == 67 && f.n == 30) {
            b3067 = &b;
            w3067 = &FW(67, 32);
        }
    }
    Check(nb == 11, "eleven blocks (" + std::to_string(nb) + ")");

    // ---- the eleven pinned rows (audit F2): every field of every row, incl. (36, 181) case B, lambda = 2
    uint32_t n_pinned = 0;
    for (const PinnedRow& r : Pinned()) {
        const BlockOut* hit = nullptr;
        for (const auto& b : outs)
            if (!b.fam->known && b.fam->n == r.n && b.fam->ell == r.ell && b.sys.res[0] == r.rbar2) hit = &b;
        const std::string at_s = "(" + std::to_string(r.n) + ", " + std::to_string(r.ell) + ") rbar2=" + std::to_string(r.rbar2);
        bool ok = hit && hit->stage && PolyText(hit->d.e) == r.D && hit->d.cas == r.cas && hit->d.lambda_word == r.lambda;
        if (ok && r.cas == 'A') ok = hit->d.sF_mod_l2 == r.sF && hit->d.rung2 == r.rung2 && hit->d.rung3 == r.rung3;
        n_pinned += Check(ok, "pinned row " + at_s + ": D=" + r.D + " case=" + std::string(1, r.cas) + " lambda=" + r.lambda +
                                  (hit && hit->stage ? std::string(" (got D=") + PolyText(hit->d.e) + " case=" + std::string(1, hit->d.cas) +
                                                           " lambda=" + hit->d.lambda_word + ")"
                                                     : std::string(" (block missing)")));
    }

    // ---- oracles at (30, 67): chi(T2 | block at 32) = X^2 - 39960 X - 2235350016 mod 67^5, det U = 56 mod 67
    {
        const ZMod R5(67, 5);
        oracle_n += 2;
        const Vec want{R5.from_i64(-2235350016LL), R5.from_i64(-39960), 1};
        oracle_ok += Check(b3067 && b3067->rank2 && b3067->chi2_k0 == want, "(30, 67): chi(T2 | block at 32) = X^2 - 39960X - 2235350016 mod 67^5");
        oracle_ok += Check(b3067 && b3067->rank2 && b3067->detU_k0 == 56, "(30, 67): det U = 56 mod 67");
    }

    // ---- the known answer (547, 486) (ALL; pr:u-H2(c))
    if (all) {
        ++oracle_n;
        bool ok = false;
        for (const auto& b : outs) {
            if (!b.fam->known) continue;
            const ZMod R5(547, 5);
            bool fk = b.rank2 && b.chi2.size() == 5;
            for (size_t j = 0; fk && j < 5; ++j)
                fk = b.chi2[j] == Vec{static_cast<u128>(H1fk()[j].first) % R5.M, static_cast<u128>(H1fk()[j].second) % R5.M, 1};
            Check(fk, "(547, 486): chi(T2 | block) = H1's f_k mod 547^5 at the five weights");
            Check(b.stage && b.d.cas == 'A', "(547, 486): case A (pr:u-H2(c))");
            Check(b.stage && b.d.rung2 == 55086, "(547, 486): rung2 = 55086 (pr:u-H2(c), the rung of prop:u-H12(5))");
            ok = fk && b.stage && b.d.cas == 'A' && b.d.rung2 == 55086;
            std::cout << "HIDA_H2_KNOWN ell=547 a=486 f_k_mod_l5=" << (fk ? "agree" : "DIFFER") << " case=" << b.d.cas
                      << " rung2=" << b.d.rung2 << " (expected 55086) verdict=" << (ok ? "ok" : "FAIL") << std::endl;
        }
        oracle_ok += ok;
    }

    // ---- must-refuse controls
    uint32_t ctl_ok = 0;
    for (const auto& c : controls) {
        const uint32_t k = c.first + 2;
        const L::FormsWeight& w = FW(c.second, k);
        bool ok = false;
        try {
            const D::SystemSearch ss = D::FindNonReducedSystems(w, E::SturmPrimes(c.second, k));
            ok = ss.systems.empty() && ss.candidates >= 1;
            std::cout << "HIDA_H2_CONTROL cell=(" << c.first << "," << c.second << ") k=" << k << " dim_S=" << w.dim_S << " s=" << w.S.s
                      << " candidates=" << ss.candidates << " refused=" << ss.refused << " systems=" << ss.systems.size()
                      << " note=\"" << (ss.notes.empty() ? "" : ss.notes[0]) << "\" verdict=" << (ok ? "refused" : "FAIL") << std::endl;
        } catch (const std::exception& e) {
            Check(false, std::string("control threw: ") + e.what());
        }
        ctl_ok += Check(ok, "control (" + std::to_string(c.first) + ", " + std::to_string(c.second) + "): a T2 coincidence, no non-reduced system");
    }

    // ---- tampers at (30, 67)
    uint32_t tampered = 0;
    if (b3067 && b3067->stage && b3067->disc.size() == 6) {
        const ZMod R5(67, 5);
        const u128 l2 = R5.pw[2], l4 = R5.pw[4];
        {   // (i) +ell^4 at j = 2: the held-out check (the interpolation stays integral; the subset self-check sees nothing)
            Vec v = b3067->disc;
            v[2] = R5.add(v[2], l4);
            const D::DiscTruncation d = D::ReadDisc(67, 32, v);
            const bool det = d.heldout == 0;
            tampered += Check(det, "tamper (i) +ell^4 at j = 2 caught by the held-out check");
            std::cout << "HIDA_H2_TAMPER i=1 heldout=" << (d.heldout == 1 ? "ok" : "FAIL") << " detector=heldout"
                      << " detected=" << (det ? "yes" : "no") << std::endl;
        }
        {   // (ii) +ell^2 at j = 3: detector the integrality refusal.  A single-node error of valuation v < 4 makes the degree-4
            // Newton coefficient ell^{v-4}-non-integral, and InterpolateTruncation's RatMod refuses (the precondition of
            // thm:u-interp(2)).  The (2, 2) subset self-check is not a detector (Z6_code_audit.md F1) and is not consulted.
            Vec v = b3067->disc;
            v[3] = R5.add(v[3], l2);
            bool refused = false;
            try {
                (void)D::ReadDisc(67, 32, v);
            } catch (const std::exception&) {
                refused = true;
            }
            tampered += Check(refused, "tamper (ii) +ell^2 at j = 3 caught by the integrality refusal of the five-node interpolation");
            std::cout << "HIDA_H2_TAMPER i=2 interp=" << (refused ? "REFUSED" : "accepted") << " detector=integrality"
                      << " detected=" << (refused ? "yes" : "no") << std::endl;
        }
        {   // (iii) a wrong residual system: rbar2 + 1
            D::ResidualSystem s = b3067->sys;
            s.res[0] = (s.res[0] + 1) % 67;
            bool det = false;
            std::string why;
            try {
                D::ResidualBlock(*w3067, b3067->fam->P, s);
            } catch (const std::exception& e) {
                det = true;
                why = e.what();
            }
            tampered += Check(det, "tamper (iii) rbar2 + 1 refused by ResidualBlock");
            std::cout << "HIDA_H2_TAMPER i=3 refused=" << (det ? "yes" : "no") << " detector=rank" << " detected=" << (det ? "yes" : "no") << std::endl;
        }
    } else {
        Check(false, "tampers: the (30, 67) block is not available");
    }

    // ---- (ALL) the rung check (pr:u-H2(b)) at ell in {67, 71, 73}, case A
    uint32_t rung_ok = 0, rung_n = 0;
    if (all) {
        struct RJ {
            const BlockOut* b;
            uint32_t k;
        };
        std::vector<RJ> rj;
        std::vector<D::FormsJob> rjobs;
        for (const auto& b : outs) {
            const Family& f = *b.fam;
            if (f.known || !b.stage || b.d.cas != 'A' || (f.ell != 67 && f.ell != 71 && f.ell != 73)) continue;
            rj.push_back(RJ{&b, static_cast<uint32_t>(b.d.rung2)});
            rjobs.push_back(D::FormsJob{f.ell, static_cast<uint32_t>(b.d.rung2), 5, E::SturmPrimes(f.ell, f.k0)});
        }
        std::vector<L::FormsWeight> rw;
        try {
            rw = D::ComputeFormsParallel(rjobs, threads);
        } catch (const std::exception& e) {
            Check(false, std::string("rung jobs threw: ") + e.what());
        }
        for (size_t i = 0; i < rj.size() && i < rw.size(); ++i) {
            ++rung_n;
            const BlockOut& b = *rj[i].b;
            const Family& f = *b.fam;
            const ZMod R5(f.ell, 5);
            forms_cpu += rw[i].wall_s;
            max_job = std::max(max_job, rw[i].wall_s);
            std::string route = "Pprime";
            D::Quad q;
            bool have = false;
            try {
                q = D::ReadQuad(D::ResidualBlock(rw[i], rjobs[i].primes, b.sys).ops.at("T2"), R5);
                have = true;
            } catch (const std::exception& e) {
                // a collision on P' (rank > 2) or another refusal: the costed fallback, forms with SturmPrimes(ell, k), cut by the family's P
                std::cout << "HIDA_H2_RUNG fallback at (" << f.ell << ", " << rj[i].k << "): " << e.what() << std::endl;
                route = "fallback";
                try {
                    const std::vector<uint64_t> Pk = E::SturmPrimes(f.ell, std::max(rj[i].k, Wt(f.k0, f.ell, 5)));
                    const L::FormsWeight wf = L::ComputeFormsWeight(f.ell, rj[i].k, 5, Pk);
                    forms_cpu += wf.wall_s;
                    q = D::ReadQuad(D::ResidualBlock(wf, f.P, b.sys).ops.at("T2"), R5);
                    have = true;
                } catch (const std::exception& e2) {
                    Check(false, "rung block at (" + std::to_string(f.ell) + ", " + std::to_string(rj[i].k) + ") refused: " + e2.what());
                }
            }
            const uint32_t pred = D::PredictedV(b.d, f.ell, rj[i].k);
            const bool ok = have && (pred < 3 ? q.v_disc == pred : q.v_disc >= 3) && q.v_disc >= 2;
            rung_ok += Check(ok, "rung (" + std::to_string(f.n) + ", " + std::to_string(f.ell) + ") at k = " + std::to_string(rj[i].k) +
                                     ": v(disc) = 1 + v(k - s_F) >= 2 (pr:u-H2(b))");
            std::cout << "HIDA_H2_RUNG ell=" << f.ell << " cell=(" << f.n << "," << f.ell << ") k=" << rj[i].k << " d=" << rw[i].dim_S
                      << " s=" << rw[i].S.s << " route=" << route << " v_disc=" << q.v_disc << " predicted=" << (pred < 3 ? std::to_string(pred) : ">=3")
                      << " nonmaximal=" << (q.v_disc >= 2 ? "yes" : "no") << " wall_s=" << rw[i].wall_s << " verdict=" << (ok ? "ok" : "FAIL") << std::endl;
        }
    }

    const bool ok = Check.fails == 0;
    std::cout << "HIDA_H2 blocks=" << nb << " cells=" << Cells().size() << " rank2=" << n_rank << "/" << nb << " gen=" << n_gen << "/" << nb
              << " v_e0_one=" << n_v1 << "/" << nb << " oracle=" << oracle_ok << "/" << oracle_n << " controls_refused=" << ctl_ok << "/"
              << controls.size() << " ring=" << n_ring << "/" << nb << " subset_selfcheck=" << n_sub << "/" << nb << " q_indep=" << n_q << "/" << nb
              << " heldout=" << n_held_ok << "/" << n_held_run << " caseA=" << cA << " caseB=" << cB << " undecided=" << cU
              << " pinned=" << n_pinned << "/" << Pinned().size() << " rung_checked=" << rung_ok << "/" << rung_n << " tampered_detected=" << tampered << " fnv=" << Hex(fnv)
              << " verdict=" << (ok ? "OK" : "FAIL") << std::endl;
    std::cout << "HIDA_H2_COST forms_wall_s=" << forms_wall << " forms_cpu_s=" << forms_cpu << " max_job_s=" << max_job << " at " << max_job_at
              << " jobs=" << jobs.size() << " threads=" << threads << std::endl;
    std::cout << "HeckeHidaDiscLambdaValidation: checks=" << Check.checks << " failing=" << Check.fails << " wall_s=" << Now() - t0
              << " peak_rss_mib=" << PeakRssMiB() << std::endl;
    return ok ? 0 : 1;
}
