// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, P10) HeckeLambdaTowerValidation -- see the header and docs/implementation/slot22/P10_spec.md
// (v0.3; section 7 = the referee-checked predictions, section 15 = the implementation predictions this validation gates).
#include "HeckeLambdaTowerValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "HeckeHidaLambda.h"
#include "HeckeLambdaQExp.h"
#include "HeckeLambdaTower.h"
#include "../Util/Rss.h"
#include "../Util/CellList.h"
#include "../../ModArithmetic/EngineCheck.h"

namespace {

using namespace hecke_ext::tower;
namespace lam = hecke_ext::lambda;

struct Checker {
    uint32_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (fails <= 80) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

double Since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
uint64_t IPow(uint64_t b, uint32_t e) { uint64_t r = 1; while (e--) r *= b; return r; }
uint32_t VEll(int64_t x, uint32_t ell) {
    if (x == 0) return 1000;
    uint32_t v = 0;
    while (x % static_cast<int64_t>(ell) == 0) { x /= static_cast<int64_t>(ell); ++v; }
    return v;
}
std::string Vec(const std::vector<uint32_t>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << "]";
    return os.str();
}
std::string YN(bool b) { return b ? "yes" : "no"; }

// d(a) of [U] thm:u-ell13 at ell <= 13
// (slot 23, G-EVT23) moved to HeckeLambdaTower.h (TowerDofA, TowerRankPred, DefaultTowerOptions): one definition, shared
// with the golden harness's TOWER points
uint32_t DofA(uint32_t ell, uint32_t a) { return TowerDofA(ell, a); }
uint32_t RPred(uint32_t ell, uint32_t a, uint32_t r) { return TowerRankPred(ell, a, r); }
uint64_t M0(uint64_t N) {   // [Gamma : Gamma_0(N)]
    uint64_t r = N, m = N;
    for (uint64_t p = 2; p * p <= m; ++p)
        if (m % p == 0) { while (m % p == 0) m /= p; r += r / p; }
    if (m > 1) r += r / m;
    return r;
}
// h = rank H^1(Gamma^t_r; omega^{a-2}) of section 7.1 (out_predict section 7), keyed (ell, a, r)
uint32_t HPred(uint32_t ell, uint32_t a, uint32_t r) {
    static const std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> T = {
        {{5, 0, 1}, 2}, {{5, 0, 2}, 30}, {{5, 0, 3}, 650}, {{5, 2, 1}, 1}, {{5, 2, 2}, 21}, {{5, 2, 3}, 601},
        {{7, 0, 1}, 2}, {{7, 0, 2}, 70}, {{7, 0, 3}, 3234}, {{7, 4, 1}, 2}, {{7, 4, 2}, 70}, {{7, 4, 3}, 3234},
        {{7, 2, 1}, 1}, {{7, 2, 2}, 57}, {{7, 2, 3}, 3137},
        {{11, 2, 1}, 3}, {{11, 2, 2}, 243}, {{11, 2, 3}, 29283}, {{11, 4, 1}, 2}, {{11, 4, 2}, 242}, {{11, 4, 3}, 29282},
        {{13, 0, 1}, 4}, {{13, 0, 2}, 416}, {{13, 0, 3}, 66924}, {{13, 4, 1}, 4}, {{13, 4, 2}, 416}, {{13, 4, 3}, 66924}};
    const auto it = T.find({ell, a, r});
    return it == T.end() ? 0 : it->second;
}

// the spec's section 7.3 tables (mod ell^3)
const std::map<uint64_t, uint32_t> kTau11 = {{2, 1307}, {3, 252}, {5, 837}, {7, 559}, {13, 1247}, {17, 625}, {19, 110}, {23, 1286}, {29, 1067}};
const std::map<uint64_t, uint32_t> kDE10_11 = {{2, 1043}, {3, 263}, {5, 221}, {7, 31}, {13, 70}, {17, 1131}, {19, 704}, {23, 637}, {29, 924}};
const std::map<uint64_t, int64_t> k11a = {{2, -2}, {3, -1}, {5, 1}, {7, -2}, {13, 4}, {17, -2}, {19, 0}, {23, -1}, {29, 0}};
const std::map<uint64_t, uint32_t> kTau13 = {{2, 2173}, {3, 252}, {5, 436}, {7, 832}, {11, 741}, {17, 1434}, {19, 1576}, {23, 1727}, {29, 768}};
const std::map<uint64_t, uint32_t> kS24_13 = {{2, 1159}, {3, 1617}, {5, 1723}, {7, 1898}, {11, 312}, {17, 199}, {19, 1017}, {23, 37}, {29, 326}};
// spec v0.4 section 17 (erratum E1): the ordinary line of S_132 mod 13^3 (p10c/p10_k132_predict.py), U = a_13
const std::map<uint64_t, uint32_t> kS132_13 = {{2, 1159}, {3, 382}, {5, 1983}, {7, 2028}, {11, 845}, {13, 944}, {17, 1083}, {19, 42}, {23, 375}, {29, 1249}};
const std::map<uint64_t, uint32_t> kD11 = {{2, 2}, {3, 10}, {5, 1}, {7, 4}, {13, 8}, {17, 9}, {19, 1}, {23, 4}, {29, 2}};
const std::map<uint64_t, uint32_t> kD13 = {{2, 0}, {3, 12}, {5, 5}, {7, 9}, {11, 7}, {17, 4}, {19, 4}, {23, 0}, {29, 8}};

// group ring product in (Z/ell^K)[G_r]
GR GRMul(const GR& x, const GR& y, const ZK& R) {
    const size_t L = x.size();
    GR z(L, 0);
    for (size_t i = 0; i < L; ++i) {
        if (!x[i]) continue;
        for (size_t j = 0; j < L; ++j) z[(i + j) % L] = R.add(z[(i + j) % L], R.mul(x[i], y[j]));
    }
    return z;
}
// the matrix of multiplication by c on the basis g^0 .. g^{L-1}, repeated `copies` times block-diagonally
DM Circulant(const GR& c, uint32_t copies) {
    const uint32_t L = static_cast<uint32_t>(c.size());
    DM M(L * copies, L * copies);
    for (uint32_t b = 0; b < copies; ++b)
        for (uint32_t i = 0; i < L; ++i)
            for (uint32_t j = 0; j < L; ++j) M.at(b * L + i, b * L + j) = c[(i + L - j) % L];
    return M;
}

// A(Z + x_c) mod Z^m (A a polynomial in X mod ell^K)
std::vector<uint32_t> ShiftTrunc(const std::vector<uint32_t>& A, uint32_t xc, uint32_t m, const ZK& R) {
    std::vector<uint32_t> out(m, 0);
    // Horner: P(Z) = (...((A_n)(Z + xc) + A_{n-1})(Z + xc) + ...)
    std::vector<uint32_t> P(m, 0);
    for (size_t i = A.size(); i-- > 0;) {
        std::vector<uint32_t> Q(m, 0);
        for (uint32_t j = 0; j < m; ++j) {
            Q[j] = R.add(Q[j], R.mul(P[j], xc));
            if (j + 1 < m) Q[j + 1] = R.add(Q[j + 1], P[j]);
        }
        Q[0] = R.add(Q[0], A[i]);
        P = Q;
    }
    out = P;
    return out;
}
// Omega_r(X) = (1 + X)^{L} - u^{2L}, coefficients mod ell^K
std::vector<uint32_t> OmegaPoly(uint32_t L, const ZK& R) {
    std::vector<uint32_t> c(L + 1, 0);
    std::vector<uint32_t> row{static_cast<uint32_t>(1 % R.Q)};
    for (uint32_t i = 1; i <= L; ++i) {
        std::vector<uint32_t> nr(i + 1, 0);
        for (uint32_t j = 0; j <= i; ++j) nr[j] = R.add(j < i ? row[j] : 0, j > 0 ? row[j - 1] : 0);
        row = nr;
    }
    for (uint32_t j = 0; j <= L; ++j) c[j] = row[j];
    const uint32_t u = (1 + R.ell) % static_cast<uint32_t>(R.Q);
    c[0] = R.sub(c[0], R.pow(u, 2ull * L));
    return c;
}

struct CellResult {
    TowerReading t;
    bool ok = false;
};

// ---- G0: the orientation, on the project code (r = 1) -----------------------------------------------------------
struct OrientCell {
    uint32_t ell, i, h, ordrank;
    std::vector<uint32_t> t2, t3;
};
const std::vector<OrientCell> kOrient = {
    {5, 2, 2, 1, {4}, {3}},        {7, 2, 2, 1, {2}, {0}},        {7, 4, 2, 1, {5}, {6}},
    {11, 0, 3, 3, {3, 9, 9}, {4, 10, 10}}, {11, 2, 2, 1, {9}, {6}}, {13, 10, 4, 3, {2, 2, 8}, {5, 5, 10}},
    {13, 2, 4, 1, {9}, {2}},       {13, 4, 2, 1, {7}, {10}}};

bool RunG0(Checker& g) {
    bool all = true;
    for (const OrientCell& oc : kOrient) {
        TowerCell c{oc.ell, (oc.i + 2) % (oc.ell - 1), 1, 3};
        TowerOptions o;
        o.R_pred = oc.ordrank;
        o.deterministic = true;
        const TowerReading t = BuildTower(c, o);
        auto t2 = t.eigsE.count("T2") ? t.eigsE.at("T2") : std::vector<uint32_t>{};
        auto t3 = t.eigsE.count("T3") ? t.eigsE.at("T3") : std::vector<uint32_t>{};
        std::sort(t2.begin(), t2.end());
        std::sort(t3.begin(), t3.end());
        const bool ok = t.h == oc.h && t.rank == oc.ordrank && t2 == oc.t2 && t3 == oc.t3 && t.E_free && t.E_Ustable && t.U_invertible &&
                        t.g5prime_ok;
        std::cout << "  ORIENT ell=" << oc.ell << " chi=omega^" << oc.i << " (class a=" << c.a << "): dimH1=" << t.h << " (pred " << oc.h
                  << ") ordinary_rank=" << t.rank << " (pred " << oc.ordrank << ") T2|ord=" << Vec(t2) << " (pred " << Vec(oc.t2)
                  << ") T3|ord=" << Vec(t3) << " (pred " << Vec(oc.t3) << ") G5'=" << YN(t.g5prime_ok) << " agree=" << YN(ok) << "\n";
        g.check(ok, "G0 orientation at (" + std::to_string(oc.ell) + ", omega^" + std::to_string(oc.i) + ")");
        all = all && ok;
    }
    return all;
}

// ---- the cell gates G1 - G3, G5 - G9 --------------------------------------------------------------------------
void CellGates(Checker& g, const TowerReading& t, const TowerCell& c) {
    const std::string cs = c.text();
    const uint32_t d = DofA(c.ell, c.a);
    const uint32_t Rp = RPred(c.ell, c.a, c.r);
    const uint32_t hp = HPred(c.ell, c.a, c.r);
    const ZK R(c.ell, c.K);
    // G1
    const uint64_t mp = M0(IPow(c.ell, c.r)) * IPow(c.ell, c.r - 1);
    g.check(t.m == mp, "G1 " + cs + " m = m_N ell^{r-1}");
    if (c.r == 1) g.check(t.table_r1_ok, "G1 " + cs + " the r = 1 table = def:cosetrepsN entry for entry");
    // G2
    if (hp) g.check(t.h == hp, "G2 " + cs + " h = " + std::to_string(hp));
    g.check(t.fl_pivots_ok && t.fl_tails_ok, "G2 " + cs + " the chain-ring pivots and tails = the F_ell RREF");
    // G3
    uint32_t esc = 0;
    for (const auto& kv : t.rel_escapes) esc += kv.second;
    g.check(esc == 0 && !t.rel_escapes.empty(), "G3 " + cs + " Rel preserved by every operator");
    g.check(t.commute_ok, "G3 " + cs + " U, T_2, T_3, <g> commute");
    g.check(t.diamond_p_ok && t.diamond_H_ok, "G3 " + cs + " <p> = <g>^{s(p)}, <h> = 1 on H_r");
    // G5
    g.check(t.krylov_stable && t.E_free && t.E_Ustable && t.U_invertible, "G5 " + cs + " Krylov certificate");
    g.check(t.rank == Rp, "G5 " + cs + " rank e H^1 = " + std::to_string(Rp));
    if (c.r <= 2) g.check(t.g5prime_run && t.g5prime_ok, "G5' " + cs + " deterministic e = U^T equals the Krylov span");
    g.check(t.split_ok, "split " + cs + " ker / im of (T_2 - lambda_Eis)^N is a direct sum");
    // G6
    uint32_t lr = 0;
    bool bases = true;
    for (const TowerPart& P : t.parts) { lr += P.lambda_rank; bases = bases && P.basis_ok; }
    g.check(lr == 2 * d + 1 && bases, "G6 " + cs + " dim M/(ell, g - 1) = 2d + 1 and the g-orbit bases are bases");
    const TowerPart* E = t.part("eis");
    const TowerPart* C = t.part("cusp");
    g.check(E && E->rank == t.L && E->lambda_rank == 1, "G9 " + cs + " M^Eis free of rank 1 over Lambda_r");
    g.check(d == 0 ? C == nullptr : (C && C->rank == 2 * d * t.L && C->lambda_rank == 2 * d), "G6 " + cs + " the cusp part has Lambda-rank 2d");
    // G7
    if (C) g.check(C->scalar_ok, "G7 " + cs + " T_p, U act by scalars of Lambda_r on M^cusp");
    // G8
    if (C && C->residual.size() == 4) {
        const uint64_t e2 = (c.a + c.ell - 1 - 2) % (c.ell - 1);
        std::vector<uint32_t> coinc;
        for (uint32_t b = 0; b + 1 < c.ell; ++b) {
            bool all = true;
            const uint64_t ps[4] = {2, 3, 5, 7};
            for (int k = 0; k < 4; ++k) {
                const uint64_t p = ps[k];
                if (p == c.ell) continue;
                uint64_t wb = 1, wc = 1;
                for (uint32_t i = 0; i < b; ++i) wb = wb * p % c.ell;
                const uint64_t ec = (e2 + (c.ell - 1) - b) % (c.ell - 1);
                for (uint64_t i = 0; i < ec; ++i) wc = wc * p % c.ell;
                if ((wb + wc * p) % c.ell != C->residual[k]) { all = false; break; }
            }
            if (all) coinc.push_back(b);
        }
        std::cout << "  BF' " << cs << " cusp residual system (T2,T3,T5,T7)=" << Vec(C->residual) << " boundary-coincident b=" << Vec(coinc) << "\n";
        g.check(coinc.empty(), "G8 " + cs + " (BF'): the cusp system is not a boundary system");
    }
    // G9
    if (E) {
        g.check(E->scalar_ok, "G9 " + cs + " T_p, U act by scalars on M^Eis");
        bool eps = true;
        for (uint64_t p : t.primes) {
            const auto it = E->scalars.find("T" + std::to_string(p));
            if (it == E->scalars.end() || it->second != EisensteinScalar(c.ell, c.a, c.r, p, R)) eps = false;
        }
        GR one(t.L, 0);
        one[0] = 1;
        const auto iu = E->scalars.find("U");
        g.check(eps, "G9 " + cs + " T_p = eps_p = 1 + p omega^{a-2}(p) g^{s(p)}, p <= 29");
        g.check(iu != E->scalars.end() && iu->second == one, "G9 " + cs + " U = 1 on M^Eis");
    }
}

void PrintCell(const TowerReading& t) {
    std::cout << "  CELL " << t.cell.text() << " K=" << t.cell.K << " N=" << t.N << " m=" << t.m << " h=" << t.h << " rank=" << t.rank
              << " krylov_T=" << t.krylov_T << " krylov_s=" << t.krylov_s << " ranks=" << Vec(t.krylov_ranks) << " G5'=" << (t.g5prime_run ? YN(t.g5prime_ok) : "n/a")
              << " elim_final_nnz=" << t.elim.final_nnz << " elim_peak_bytes=" << t.elim.peak_bytes << " residual_nonunit=" << t.elim.residual_nonunit
              << " nnz(U)=" << (t.nnz.count("U") ? t.nnz.at("U") : 0) << " parts=";
    for (const TowerPart& P : t.parts) std::cout << P.kind << ":" << P.rank << "/" << P.lambda_rank << (P.scalar_ok ? "" : "(!scalar)") << " ";
    std::cout << "wall=" << t.seconds << "s (elim " << t.elim_s << ", ops " << t.ops_s << ", krylov " << t.krylov_s_wall << ") peak_rss_mib=" << oc_util::PeakRssMiB() << "\n";
}

// charpoly of the scalar c acting on `copies` free generators (the reduced-scalar side of G10)
std::vector<uint32_t> ScalarCharpoly(const GR& c, uint32_t copies, const ZK& R) { return Charpoly(Circulant(c, copies), R); }

std::vector<TowerCell> ParseCells(const std::string& s) {
    std::vector<TowerCell> out;
    for (const oc_util::cells::Token& t : oc_util::cells::Tokenize(s, 9)) {
        const size_t pos = t.pos;
        if (!t.Fits(3, 4))
            throw std::logic_error("HECKE_LAMBDA_TOWER_CELLS: malformed token at position " + std::to_string(pos) + ": '" + t.text + "'");
        const std::string& tok = t.text;
        std::vector<uint32_t> f;
        for (uint64_t x : t.fields) f.push_back(static_cast<uint32_t>(x));
        TowerCell c{f[0], f[1], f[2], f.size() == 4 ? f[3] : 3u};
        if (c.ell > 13)
            throw std::logic_error("tower_edge_not_built: ell = " + std::to_string(c.ell) + " at position " + std::to_string(pos) +
                                   " (the edge cell needs the component split and the residue-degree refusal, not built in this commit)");
        if (c.ell < 5 || c.a % 2 || c.a >= c.ell - 1 || c.r < 1 || c.r > 3 || c.K < 1 || c.K > 3)
            throw std::logic_error("HECKE_LAMBDA_TOWER_CELLS: out-of-range token at position " + std::to_string(pos) + ": '" + tok + "'");
        out.push_back(c);
    }
    if (out.empty()) throw std::logic_error("HECKE_LAMBDA_TOWER_CELLS: the list selects no cell");
    return out;
}

} // namespace

uint32_t HeckeLambdaTowerValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    std::cout << "  P10 tower: spec docs/implementation/slot22/P10_spec.md v0.3; [C] v1.22.0-pre.1 sec:nebwildsec, [U] 0.6.0-pre.1 sec:u-tower\n";
    // ---------------- G0 first ----------------
    if (!RunG0(g)) {
        std::cout << "  G0 FAILED: the orientation of prop:u-towerorient is not reproduced on the project code -- STOP (no Lambda-adic reading)\n";
        std::cout << "  HECKE_LAMBDA_TOWER checks=" << g.checks << " failing=" << g.fails << " verdict=FAIL\n";
        return g.fails ? g.fails : 1;
    }
    // ---------------- the cell list ----------------
    std::vector<TowerCell> cells;
    for (uint32_t ell : {5u, 7u})
        for (uint32_t a = 0; a + 1 < ell; a += 2)
            for (uint32_t r = 1; r <= 3; ++r) cells.push_back({ell, a, r, 3});
    for (auto la : std::vector<std::pair<uint32_t, uint32_t>>{{11, 2}, {11, 4}, {13, 0}, {13, 4}})
        for (uint32_t r = 1; r <= 2; ++r) cells.push_back({la.first, la.second, r, 3});
    std::string arm = "DEFAULT";
    if (const char* ce = std::getenv("HECKE_LAMBDA_TOWER_CELLS")) {
        cells = ParseCells(ce);
        arm = "CELLS";
    }
    bool r3 = false;
    if (const char* e3 = std::getenv("HECKE_LAMBDA_TOWER_R3")) {
        ENGINE_CHECK_MSG(std::string(e3) == "1", "HECKE_LAMBDA_TOWER_R3 must be 1 when set");
        r3 = true;
    }
    double cap_gib = 4.0;
    if (const char* cg = std::getenv("HECKE_LAMBDA_TOWER_RSS_CAP_GIB")) cap_gib = std::stod(cg);
    std::cout << "  " << oc_util::cells::EffectiveLine("HeckeLambdaTowerValidation", arm, cells.size()) << "\n";
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, TowerReading> rd;
    for (const TowerCell& c : cells) {
        const TowerOptions o = DefaultTowerOptions(c);
        const TowerReading t = BuildTower(c, o);
        PrintCell(t);
        CellGates(g, t, c);
        rd[{c.ell, c.a, c.r}] = t;
    }
    if (r3) {
        std::vector<TowerCell> c3 = {{11, 2, 3, 3}, {13, 0, 3, 3}};
        std::cout << "  EFFECTIVE HeckeLambdaTowerValidation arm=R3 cells=" << c3.size() << " rss_cap_gib=" << cap_gib << "\n";
        for (const TowerCell& c : c3) {
            // rule 19: bytes and work predicted from the measured r = 2 cell before allocating (fill scales with D h)
            uint64_t pred = 0;
            double ops = 0;
            const auto it2 = rd.find({c.ell, c.a, 2});
            if (it2 != rd.end()) {
                const TowerReading& t2 = it2->second;
                const double D3 = static_cast<double>(M0(IPow(c.ell, 3)) * IPow(c.ell, 2)), h3 = HPred(c.ell, c.a, 3);
                const double scale = (D3 / t2.D) * (h3 / t2.h);
                uint64_t nnz2 = 0;
                for (const auto& kv : t2.nnz) nnz2 += kv.second;
                pred = static_cast<uint64_t>(static_cast<double>(t2.elim.peak_bytes) * scale + 8.0 * static_cast<double>(nnz2) * scale);
                const double T0 = 3.0 * (IPow(c.ell, 2) - IPow(c.ell, 1)) + 6.0;
                ops = 4.0 * T0 * (RPred(c.ell, c.a, 3) + 16.0) * static_cast<double>(t2.nnz.at("U")) * scale;
            }
            double ops_cap = 1e12;
            if (const char* oc = std::getenv("HECKE_LAMBDA_TOWER_OPS_CAP")) ops_cap = std::stod(oc);
            const uint64_t cap = static_cast<uint64_t>(cap_gib * 1073741824.0);
            std::cout << "  EFFECTIVE HeckeLambdaTowerValidation arm=R3 cell=" << c.text() << " predicted_bytes=" << pred << " cap_bytes=" << cap
                      << " predicted_krylov_ops=" << ops << " ops_cap=" << ops_cap << "\n";
            if (pred == 0 || pred > cap) {
                std::cout << "  REFUSED tower_rss_cap " << c.text() << " (predicted " << pred << " bytes > " << cap << "; counted, not a failure)\n";
                continue;
            }
            if (ops > ops_cap) {
                std::cout << "  REFUSED tower_cost_cap " << c.text() << " (predicted " << ops << " Krylov operations > " << ops_cap << "; counted, not a failure)\n";
                continue;
            }
            TowerOptions o;
            o.R_pred = RPred(c.ell, c.a, c.r);
            o.deterministic = false;
            o.fl_crosscheck = false;
            o.byte_cap = cap;
            const TowerReading t = BuildTower(c, o);
            PrintCell(t);
            CellGates(g, t, c);
            rd[{c.ell, c.a, c.r}] = t;
        }
    }
    // ---------------- G10: the tower ----------------
    for (const auto& kv : rd) {
        const TowerReading& t = kv.second;
        const TowerCell& c = t.cell;
        const ZK R(c.ell, c.K);
        for (uint32_t rp = 1; rp < c.r; ++rp) {
            const auto it = rd.find({c.ell, c.a, rp});
            if (it == rd.end()) continue;
            const TowerReading& s = it->second;
            const uint32_t Lp = s.L;
            bool ok = true, cp = true;
            for (const TowerPart& P : t.parts) {
                const TowerPart* Q = s.part(P.kind);
                if (!Q) { ok = false; continue; }
                for (const auto& sc : P.scalars) {
                    const auto jt = Q->scalars.find(sc.first);
                    if (jt == Q->scalars.end() || ReduceTo(sc.second, Lp, R) != jt->second) ok = false;
                }
                for (const char* n : {"T2", "T3", "U"}) {
                    const auto jt = Q->charpoly.find(n);
                    if (jt == Q->charpoly.end() || !P.scalars.count(n)) { cp = false; continue; }
                    if (ScalarCharpoly(ReduceTo(P.scalars.at(n), Lp, R), P.lambda_rank, R) != jt->second) cp = false;
                }
            }
            std::cout << "  TOWER " << c.text() << " -> r'=" << rp << ": scalars reduce=" << YN(ok) << " charpolys(T2,T3,U)=" << YN(cp) << "\n";
            g.check(ok, "G10 " + c.text() + " -> r' = " + std::to_string(rp) + ": a_p, alpha, eps reduce exactly");
            g.check(cp, "G10 " + c.text() + " -> r' = " + std::to_string(rp) + ": charpolys of the reduced module = the lower cell's");
        }
    }
    // ---------------- G11, G12, G14 (the Delta family cells) ----------------
    struct FamilySpec { uint32_t ell, a; std::vector<uint32_t> ks; };
    const std::vector<FamilySpec> fams = {{11, 2, {12, 22, 112}}, {13, 0, {12, 24, 132}}};   // 132: 158 until spec v0.3, not in class 0 (erratum E1, spec section 17)
    for (const FamilySpec& F : fams) {
        std::vector<uint64_t> primes;
        for (uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull}) if (p != F.ell) primes.push_back(p);
        std::map<uint32_t, lam::FormsWeight> fw;
        bool any = false;
        for (uint32_t r = 1; r <= 3; ++r) any = any || rd.count({F.ell, F.a, r});
        if (!any) continue;
        for (uint32_t k : F.ks) {
            fw[k] = lam::ComputeFormsWeight(F.ell, k, 3, primes);
            const auto& S = fw[k].S;
            std::cout << "  FORMS ell=" << F.ell << " k=" << k << " ordinary_cusp_rank=" << S.s << "\n";
            g.check(S.s == 1, "G11 forms side ell=" + std::to_string(F.ell) + " k=" + std::to_string(k) + ": e S_k of rank 1");
        }
        // the oracle values against section 7.3
        for (uint64_t p : primes) {
            const lam::ZMod Z(F.ell, 3);
            auto val = [&](uint32_t k, const std::string& n) { return static_cast<uint32_t>(fw[k].S.mats.at(n)[0][0] % Z.M); };
            if (F.ell == 11) {
                if (kTau11.count(p)) {
                    g.check(val(12, "T" + std::to_string(p)) == kTau11.at(p), "ORACLE tau(" + std::to_string(p) + ") mod 11^3");
                    g.check(val(22, "T" + std::to_string(p)) == kDE10_11.at(p), "ORACLE Delta E10 a_" + std::to_string(p) + " mod 11^3");
                }
            } else {
                if (kTau13.count(p)) {
                    g.check(val(12, "T" + std::to_string(p)) == kTau13.at(p), "ORACLE tau(" + std::to_string(p) + ") mod 13^3");
                    g.check(val(24, "T" + std::to_string(p)) == kS24_13.at(p), "ORACLE S_24 ord a_" + std::to_string(p) + " mod 13^3");
                    if (fw[132].S.s == 1)
                        g.check(val(132, "T" + std::to_string(p)) == kS132_13.at(p), "ORACLE S_132 ord a_" + std::to_string(p) + " mod 13^3 (erratum E1)");
                }
            }
        }
        if (F.ell == 13 && fw[132].S.s == 1)
            g.check(static_cast<uint32_t>(fw[132].S.mats.at("U")[0][0] % 2197) == kS132_13.at(13), "ORACLE S_132 ord U mod 13^3 (erratum E1)");
        for (uint32_t r = 1; r <= 3; ++r) {
            const auto it = rd.find({F.ell, F.a, r});
            if (it == rd.end()) continue;
            const TowerReading& t = it->second;
            const TowerPart* C = t.part("cusp");
            if (!C || !C->scalar_ok) { g.check(false, "G11 " + t.cell.text() + " no cusp scalars"); continue; }
            const ZK R(F.ell, 3);
            uint32_t bad = 0, tot = 0;
            // k >= 12
            for (uint32_t k : F.ks) {
                if (fw[k].S.s != 1) continue;   // the forms gate above already failed (no rank-1 e S_k to compare with)
                const uint32_t prec = std::min<uint32_t>(3, r + VEll(static_cast<int64_t>(k) - 2, F.ell));
                const uint64_t M = IPow(F.ell, prec);
                for (uint64_t p : primes) {
                    const uint32_t mine = FibreAt(C->scalars.at("T" + std::to_string(p)), F.ell, k, prec);
                    const uint32_t theirs = static_cast<uint32_t>(fw[k].S.mats.at("T" + std::to_string(p))[0][0] % M);
                    ++tot; bad += mine != theirs;
                }
                const uint32_t ma = FibreAt(C->scalars.at("U"), F.ell, k, prec);
                const uint32_t ta = static_cast<uint32_t>(fw[k].S.mats.at("U")[0][0] % M);
                ++tot; bad += ma != ta;
                std::cout << "  FIBRE " << t.cell.text() << " k=" << k << " mod " << F.ell << "^" << prec << ": a_2=" << FibreAt(C->scalars.at("T2"), F.ell, k, prec)
                          << " forms=" << static_cast<uint64_t>(fw[k].S.mats.at("T2")[0][0] % M) << " alpha=" << ma << " forms_U=" << ta << "\n";
            }
            // k = 2
            for (uint64_t p : primes) {
                const uint32_t mine = FibreAt(C->scalars.at("T" + std::to_string(p)), F.ell, 2, 3);
                uint32_t want;
                if (F.ell == 11) want = R.from_i64(k11a.at(p));
                else {
                    const auto i1 = rd.find({F.ell, F.a, 1});
                    if (i1 == rd.end() || !i1->second.part("cusp")) continue;
                    want = i1->second.part("cusp")->scalars.at("T" + std::to_string(p))[0];
                }
                ++tot; bad += mine != want;
            }
            if (F.ell == 11) { ++tot; bad += FibreAt(C->scalars.at("U"), F.ell, 2, 3) != 1; }
            std::cout << "  FIBRES " << t.cell.text() << " compared=" << tot << " mismatching=" << bad << "\n";
            g.check(bad == 0 && tot > 0, "G11 " + t.cell.text() + " weight fibres a_p(x_k), alpha(x_k) = the forms side mod ell^{K_r(k)}");
            // G12 (r >= 2)
            if (r >= 2) {
                uint32_t b12 = 0;
                std::ostringstream os;
                const ZK R1(F.ell, 1);
                const uint32_t x12 = static_cast<uint32_t>((R1.pow(static_cast<uint32_t>((1 + F.ell) % F.ell), 12) + F.ell - 1) % F.ell);
                const auto& D = F.ell == 11 ? kD11 : kD13;
                for (uint64_t p : primes) {
                    const auto& A = C->weierstrass.at("T" + std::to_string(p));
                    uint64_t der = 0, w = 1;
                    for (size_t j = 1; j < A.size(); ++j) { der = (der + j % F.ell * (A[j] % F.ell) % F.ell * w) % F.ell; w = w * x12 % F.ell; }
                    os << p << ":" << der << " ";
                    if (D.count(p) && D.at(p) != der) ++b12;
                }
                std::cout << "  FIRST_ORDER " << t.cell.text() << " a_p'(x_12) mod " << F.ell << ": " << os.str() << "\n";
                g.check(b12 == 0, "G12 " + t.cell.text() + " first-order coefficients = section 7.3");
                // G14
                std::vector<uint64_t> gens;
                for (uint64_t p : primes) {
                    const GR& a = C->scalars.at("T" + std::to_string(p));
                    DM M(t.L, t.L);
                    GR pw(t.L, 0);
                    pw[0] = 1;
                    for (uint32_t j = 0; j < t.L; ++j) {
                        for (uint32_t i = 0; i < t.L; ++i) M.at(i, j) = pw[i];
                        pw = GRMul(pw, a, R);
                    }
                    if (RankModEll(M, F.ell) == t.L) gens.push_back(p);
                }
                std::cout << "  GENERATE " << t.cell.text() << " T_p generating Lambda_r: ";
                for (uint64_t p : gens) std::cout << p << " ";
                std::cout << "\n";
                const uint64_t p0 = F.ell == 11 ? 2 : 3;
                const bool p2_13 = F.ell == 13 && std::find(gens.begin(), gens.end(), 2ull) != gens.end();
                g.check(std::find(gens.begin(), gens.end(), p0) != gens.end() && !p2_13,
                        "G14 " + t.cell.text() + " (U21): T_" + std::to_string(p0) + " generates Lambda_r (T_2 at 13 does not)");
                // NEG: the inverted orientation 1 + X -> u^2 <u>^{-1}, at k = 12 mod ell^2
                if (r == 2) {
                    auto inv = [&](const GR& c) { GR o(c.size()); for (size_t i = 0; i < c.size(); ++i) o[(c.size() - i) % c.size()] = c[i]; return o; };
                    const uint64_t pc = F.ell == 11 ? 2 : 3;
                    const uint32_t M2 = F.ell * F.ell;
                    const bool caught = FibreAt(inv(C->scalars.at("T" + std::to_string(pc))), F.ell, 12, 2) != fw[12].S.mats.at("T" + std::to_string(pc))[0][0] % M2;
                    std::cout << "  NEG inverted_orientation " << t.cell.text() << " p=" << pc << " " << (caught ? "caught" : "NOT caught") << "\n";
                    g.check(caught, "NEG inverted orientation caught at " + t.cell.text() + " p = " + std::to_string(pc));
                    if (F.ell == 13) {
                        const bool c2 = FibreAt(inv(C->scalars.at("T2")), F.ell, 12, 2) != fw[12].S.mats.at("T2")[0][0] % M2;
                        std::cout << "  NEG inverted_orientation " << t.cell.text() << " p=2 " << (c2 ? "caught" : "not caught (predicted: a'_2(x_12) = 0 mod 13)") << "\n";
                        g.check(!c2, "NEG inverted orientation at (13, 0, 2), p = 2: not caught, as predicted");
                    }
                    // NEG: one coefficient of a_2 perturbed -> the weight-2 fibre / the tower
                    GR pert = C->scalars.at("T2");
                    pert[0] = R.add(pert[0], 1);
                    const auto i1 = rd.find({F.ell, F.a, 1});
                    const bool cpert = i1 != rd.end() && i1->second.part("cusp") && ReduceTo(pert, 1, R) != i1->second.part("cusp")->scalars.at("T2");
                    std::cout << "  NEG perturbed_coefficient " << t.cell.text() << " " << (cpert ? "caught" : "NOT caught") << " (G10 / the weight-2 fibre)\n";
                    g.check(cpert, "NEG perturbed coefficient caught at " + t.cell.text());
                }
            }
        }
        // ---------------- G13: P7's truncations in Q ----------------
        for (uint32_t m = 1; m <= 3; ++m) {
            const uint32_t K = 3;
            std::vector<lam::FormsWeight> ws;
            for (uint32_t j = 0; j < K + m - 1; ++j)
                ws.push_back(lam::ComputeFormsWeight(F.ell, 12 + j * (F.ell - 1), K + m - 1, primes));
            const lam::LambdaQExpansion Lq = lam::BuildLambdaQExpansion(ws, F.a, K, m, 12, 30, 0, false);
            const lam::LambdaComponent* comp = nullptr;
            for (const auto& cc : Lq.comps) if (cc.kind == "cusp" && cc.rank == 1) comp = &cc;
            g.check(comp != nullptr, "G13 P7 truncation at ell=" + std::to_string(F.ell) + " m=" + std::to_string(m) + " has one rank-1 cusp component");
            if (!comp) continue;
            for (uint32_t r = 1; r <= 3; ++r) {
                const auto it = rd.find({F.ell, F.a, r});
                if (it == rd.end()) continue;
                const TowerReading& t = it->second;
                const TowerPart* C = t.part("cusp");
                if (!C || !C->scalar_ok) continue;
                const ZK R(F.ell, K);
                const uint32_t xc = static_cast<uint32_t>((R.pow(static_cast<uint32_t>((1 + F.ell) % R.Q), 12) + R.Q - 1) % R.Q);
                // the ideal (Omega_r(Z + x_c)) in (Z/ell^K)[Z]/(Z^m)
                const std::vector<uint32_t> om = ShiftTrunc(OmegaPoly(t.L, R), xc, m, R);
                std::vector<std::vector<uint32_t>> gensI;
                for (uint32_t j = 0; j < m; ++j) {
                    std::vector<uint32_t> row(m, 0);
                    for (uint32_t i = 0; i + j < m; ++i) row[i + j] = om[i];
                    gensI.push_back(row);
                }
                const hecke::level::tower::HowellZK H(R, gensI, m);
                const uint32_t qlen = K * m - H.length();
                static const uint32_t kQ[3][3] = {{1, 2, 3}, {2, 4, 5}, {3, 5, 7}};
                g.check(qlen == kQ[t.cell.r - 1][m - 1], "G13 " + t.cell.text() + " length of Q at (K, m) = (3, " + std::to_string(m) + ")");
                uint32_t bad = 0, tot = 0;
                for (uint64_t p : primes) {
                    const auto ip = comp->ap.find(p);
                    if (ip == comp->ap.end()) continue;
                    const std::vector<uint32_t> mine = ShiftTrunc(C->weierstrass.at("T" + std::to_string(p)), xc, m, R);
                    std::vector<uint32_t> diff(m);
                    for (uint32_t i = 0; i < m; ++i) diff[i] = R.sub(mine[i], static_cast<uint32_t>(ip->second[0][i] % R.Q));
                    ++tot;
                    bad += H.contains(diff) ? 0 : 1;
                }
                {
                    const std::vector<uint32_t> mine = ShiftTrunc(C->weierstrass.at("U"), xc, m, R);
                    std::vector<uint32_t> diff(m);
                    for (uint32_t i = 0; i < m; ++i) diff[i] = R.sub(mine[i], static_cast<uint32_t>(comp->alpha[0][i] % R.Q));
                    ++tot;
                    bad += H.contains(diff) ? 0 : 1;
                }
                std::cout << "  P7 " << t.cell.text() << " (K,m)=(3," << m << ") len(Q)=" << qlen << " compared=" << tot << " mismatching=" << bad << "\n";
                g.check(bad == 0 && tot > 0, "G13 " + t.cell.text() + " P7 truncation (3, " + std::to_string(m) + ") = A_p reduced in Q");
            }
            if (m == 3) {   // NEG: N = K + m - 2 weights: P7's own refusal
                bool refused = false;
                try {
                    std::vector<lam::FormsWeight> ws2(ws.begin(), ws.begin() + (K + m - 2));
                    (void)lam::BuildLambdaQExpansion(ws2, F.a, K, m, 12, 30, 0, false);
                } catch (const std::exception& ex) {
                    refused = true;
                }
                std::cout << "  NEG p7_too_few_weights ell=" << F.ell << " " << (refused ? "caught (refused by P7)" : "NOT caught") << "\n";
                g.check(refused, "NEG P7 with K + m - 2 weights refused");
            }
        }
    }
    // ---------------- G4: the Gamma_1 oracle (NB10) ----------------
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> g1cells;
    for (const auto& kv : rd)
        if (kv.second.cell.r <= 2) g1cells[{kv.second.cell.ell, kv.second.cell.r}].push_back(kv.second.cell.a);
    for (const auto& kv : g1cells) {
        const uint32_t ell = kv.first.first, r = kv.first.second;
        std::map<uint32_t, uint32_t> rp;
        for (uint32_t a : kv.second) rp[a] = RPred(ell, a, r);
        const Gamma1Oracle O = BuildGamma1Oracle(ell, r, 3, kv.second, rp);
        std::cout << "  GAMMA1 ell=" << ell << " r=" << r << " m=" << O.m << " h=" << O.h << " wall=" << O.seconds << "s peak_rss_mib=" << oc_util::PeakRssMiB() << "\n";
        g.check(O.m == M0(IPow(ell, r)) * (IPow(ell, r) - IPow(ell, r - 1)), "G4 Gamma_1(" + std::to_string(IPow(ell, r)) + ") index");
        for (const Gamma1OracleClass& C : O.classes) {
            const TowerReading& t = rd.at({ell, C.a, r});
            const std::string cs = t.cell.text();
            bool cpok = true;
            for (const char* n : {"T2", "T3", "U"}) {
                const auto i1 = C.charpoly.find(n);
                const auto i2 = t.charpolyE.find(n);
                if (i1 == C.charpoly.end() || i2 == t.charpolyE.end() || i1->second != i2->second) cpok = false;
            }
            std::cout << "  NB10 " << cs << " tr(e_chi)=" << C.trace_e << " h(Gamma^t)=" << t.h << " ordinary_rank=" << C.rank << " (Gamma^t " << t.rank
                      << ") certified=" << YN(C.certified) << " charpolys(T2,T3,U)=" << YN(cpok) << "\n";
            g.check(C.trace_e == t.h % IPow(ell, 3), "G4 " + cs + " tr(e_chi | H^1(Gamma_1)) = h(Gamma^t_r)");
            g.check(C.rank == t.rank && C.certified, "G4 " + cs + " the ordinary rank of e_chi H^1(Gamma_1), certified, in the chi-eigenspace");
            g.check(cpok, "G4 " + cs + " charpolys of T_2, T_3, U on the ordinary parts equal");
            const bool neg = O.h != t.h;
            std::cout << "  NEG gamma1_without_projector " << cs << " " << (neg ? "caught" : "NOT caught") << " (h(Gamma_1)=" << O.h << ")\n";
            g.check(neg, "NEG Gamma_1 oracle without the tame projector caught at " + cs);
        }
    }
    // ---------------- the remaining negative controls ----------------
    {
        // untwisted 1 + X -> <u>: G9 at every r >= 2 cell
        uint32_t caught = 0, tot = 0;
        for (const auto& kv : rd) {
            const TowerReading& t = kv.second;
            if (t.cell.r < 2) continue;
            const TowerPart* E = t.part("eis");
            if (!E) continue;
            const ZK R(t.cell.ell, t.cell.K);
            ++tot;
            caught += E->scalars.at("T2") != EisensteinScalar(t.cell.ell, t.cell.a, t.cell.r, 2, R, true) ? 1 : 0;
        }
        std::cout << "  NEG untwisted_lambda (1+X -> <u>) caught at " << caught << " / " << tot << " cells with r >= 2 (G9)\n";
        g.check(tot > 0 && caught == tot, "NEG untwisted Lambda structure caught at every r >= 2 cell");
    }
    for (uint32_t r = 1; r <= 2; ++r) {
        // the tame character omega^a instead of omega^{a-2}, and the conjugate omega^{2-a}, at (13, 0)
        for (int which = 0; which < 2; ++which) {
            TowerOptions o;
            o.R_pred = RPred(13, 0, r);
            o.have_tame_override = true;
            o.tame_override = which == 0 ? 0 : 2;
            o.fl_crosscheck = false;
            o.rel_escapes = false;
            const TowerReading t = BuildTower({13, 0, r, 3}, o);
            const bool caught = t.rank != RPred(13, 0, r);
            const char* nm = which == 0 ? "tame_omega^a" : "conjugate_omega^{2-a}";
            std::cout << "  NEG " << nm << " (13,0," << r << "): ordinary rank " << t.rank << " (pred " << IPow(13, r - 1) << " for the wrong class, "
                      << RPred(13, 0, r) << " for the right one) " << (caught ? "caught" : "NOT caught") << "\n";
            g.check(caught && t.rank == IPow(13, r - 1), std::string("NEG ") + nm + " caught by the G5 rank at r = " + std::to_string(r));
        }
    }
    {
        // e skipped: rank h instead of the ordinary rank
        const auto it = rd.find({13, 0, 2});
        if (it != rd.end()) {
            const bool caught = it->second.h != RPred(13, 0, 2);
            std::cout << "  NEG e_skipped (13,0,2): rank " << it->second.h << " vs " << RPred(13, 0, 2) << " " << (caught ? "caught" : "NOT caught") << "\n";
            g.check(caught, "NEG e skipped caught (G5 rank)");
        }
        // Krylov T = K at (7, 0, 3)
        TowerOptions o;
        o.R_pred = RPred(7, 0, 3);
        o.krylov_fixed_T = 3;
        o.fl_crosscheck = false;
        o.rel_escapes = false;
        const TowerReading t = BuildTower({7, 0, 3, 3}, o);
        const bool caught = !(t.E_free && t.E_Ustable && t.U_invertible && t.rank == RPred(7, 0, 3));
        std::cout << "  NEG krylov_T=K (7,0,3): rank=" << t.rank << " free=" << YN(t.E_free) << " U-stable=" << YN(t.E_Ustable) << " U-invertible="
                  << YN(t.U_invertible) << " " << (caught ? "caught (certificate fails: krylov_unstable)" : "NOT caught") << "\n";
        g.check(caught, "NEG Krylov T = K caught by the G5 certificate");
    }
    // ---------------- the record ----------------
    for (const auto& kv : rd)
        for (const std::string& line : TowerRecordLines(kv.second, {})) std::cout << "  " << line << "\n";
    std::cout << "  HECKE_LAMBDA_TOWER checks=" << g.checks << " failing=" << g.fails << " wall_s=" << Since(t0) << " peak_rss_mib=" << oc_util::PeakRssMiB()
              << " verdict=" << (g.fails ? "FAIL" : "OK") << "\n";
    return g.fails;
}
