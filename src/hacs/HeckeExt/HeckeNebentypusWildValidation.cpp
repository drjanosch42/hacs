// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, P10 (a)) HeckeNebentypusWildValidation -- see the header and
// docs/implementation/slot22/P10_spec.md (section 7.5 = the predicted lengths).
#include "HeckeNebentypusWildValidation.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "HeckeLambdaTower.h"
#include "../Hecke/HeckeNebentypus.h"
#include "../Hecke/HeckeNebentypusWild.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/EngineCheck.h"

namespace {

using namespace hecke::level;
using hecke_ext::tower::DM;
using hecke_ext::tower::GR;
using hecke_ext::tower::MatMul;
using hecke_ext::tower::MatPow;
using hecke_ext::tower::MatSub;

struct Checker {
    uint32_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (fails <= 60) std::cout << "  FAIL: " << what << "\n";
        }
    }
};
uint64_t IPow(uint64_t b, uint32_t e) { uint64_t r = 1; while (e--) r *= b; return r; }
std::string YN(bool b) { return b ? "yes" : "no"; }

struct WildCell {
    uint32_t ell, r, j, e, a;
    uint32_t EC, lenH1, lenH0, lenE;   // section 7.5
};
const std::vector<WildCell> kCells = {
    {5, 2, 1, 12, 0, 6, 72, 0, 12},      {5, 2, 1, 12, 2, 4, 49, 1, 12},
    {7, 2, 1, 18, 0, 10, 180, 0, 18},    {7, 2, 1, 18, 2, 8, 145, 1, 18},    {7, 2, 1, 18, 4, 10, 180, 0, 18},
    {11, 2, 1, 30, 2, 22, 661, 1, 90},   {11, 2, 1, 30, 4, 22, 660, 0, 30},
    {13, 2, 1, 36, 0, 32, 1152, 0, 108}, {13, 2, 1, 36, 4, 32, 1152, 0, 36},
    {5, 3, 2, 40, 0, 26, 1040, 0, 40},   {5, 3, 2, 40, 2, 24, 961, 1, 40}};
const std::vector<WildCell> kEnvCells = {
    {7, 3, 2, 42, 0, 66, 2772, 0, 42}, {7, 3, 2, 42, 2, 64, 2689, 1, 42}, {7, 3, 2, 42, 4, 66, 2772, 0, 42}};

uint32_t DofA(uint32_t ell, uint32_t a) { return (ell == 11 && a == 2) || (ell == 13 && a == 0) ? 1 : 0; }

// the quotient operator as a dense h x h matrix
DM Dense(const tower::QuotientOperator& X) {
    DM M(X.h, X.h);
    for (uint32_t f = 0; f < X.h; ++f)
        for (uint64_t k = X.start[f]; k < X.start[f + 1]; ++k) M.at(X.ent[k].first, f) = X.ent[k].second;
    return M;
}
std::vector<std::vector<uint32_t>> TorsionRows(const tower::ChainCokernel& Q) {
    std::vector<std::vector<uint32_t>> rows;
    for (const SparseVec& v : Q.torsion_relations()) {
        std::vector<uint32_t> row(Q.h(), 0);
        for (size_t i = 0; i < v.col.size(); ++i) row[v.col[i]] = v.val[i];
        rows.push_back(std::move(row));
    }
    return rows;
}
std::vector<std::vector<uint32_t>> ColsAsRows(const DM& A) {
    std::vector<std::vector<uint32_t>> rows(A.c, std::vector<uint32_t>(A.r));
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.c; ++j) rows[j][i] = A.at(i, j);
    return rows;
}
// every column of A lies in the torsion span T
bool ColsInSpan(const DM& A, const tower::HowellZK* T) {
    for (const auto& c : ColsAsRows(A)) {
        bool zero = true;
        for (uint32_t v : c) zero = zero && v == 0;
        if (zero) continue;
        if (!T || !T->contains(c)) return false;
    }
    return true;
}
// the group-ring element c at g -> zeta (t = 1), reduced into O_{j,e}
wild::OElem AtZeta(const GR& c, const wild::WildRing& O) {
    wild::OElem s = O.zero();
    for (size_t i = 0; i < c.size(); ++i) s = O.add(s, O.mul(O.scalar(static_cast<uint32_t>(c[i] % O.R.Q)), O.zpow(i)));
    return s;
}

void RunCell(Checker& g, const WildCell& c) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t i = (c.a + c.ell - 1 - 2) % (c.ell - 1);
    const std::string cs = "(ell=" + std::to_string(c.ell) + ", r=" + std::to_string(c.r) + ", O_{" + std::to_string(c.j) + "," +
                           std::to_string(c.e) + "}, a=" + std::to_string(c.a) + ", i=" + std::to_string(i) + ")";
    const uint64_t N = IPow(c.ell, c.r);
    const wild::WildRing O = wild::MakeWildRing(c.ell, c.j, c.e);
    const wild::WildCharacter chi = wild::MakeWildCharacter(O, N, i, 1);
    // W1 (NB7): chi_{i,t} = omega^i mod pi at every unit; the F_ell omega^i-model has dimension EC + [i = 0] (= length at e = 1)
    bool w1 = true;
    for (uint64_t d = 1; d < N; ++d) {
        if (d % c.ell == 0) continue;
        uint64_t w = 1;
        for (uint32_t k = 0; k < i; ++k) w = w * (d % c.ell) % c.ell;
        if (O.mod_pi(chi.table[d]) != w) w1 = false;
    }
    const tower::ZK F(c.ell, 1);
    std::vector<uint64_t> units;
    for (uint64_t u = 1; u < N; ++u) if (u % c.ell) units.push_back(u);
    const GammaHLevelN G0(N, units);
    const std::vector<uint32_t> om = tower::TameCharacterTable(N, F, i);
    std::vector<SparseVec> gS, gU;
    tower::RelGeneratorsChi(tower::MonomialChiOf(G0, hecke::Mat22::S(), om), tower::MonomialChiOf(G0, hecke::Mat22::ST(), om), F, 2, 3, gS, gU);
    gS.insert(gS.end(), gU.begin(), gU.end());
    const tower::ChainCokernel QF(G0.m(), gS, F);
    const uint32_t hF = QF.h();
    g.check(w1 && hF == c.EC + (i == 0 ? 1u : 0u), "W1 (NB7) " + cs + ": chi_{i,t} = omega^i mod pi; dim H^1 over F_ell = EC + [i = 0]");
    // W2 (NB8)
    const wild::WildModel W(O, chi);
    const auto& tp = W.torsion_pivot_valuations();
    bool nonunit = !tp.empty();
    for (uint32_t v : tp) nonunit = nonunit && v >= 1;
    g.check(W.len_H1() == c.lenH1, "W2 (NB8) " + cs + ": length H^1 = " + std::to_string(c.lenH1));
    g.check(W.len_H0() == c.lenH0, "W2 (NB8) " + cs + ": length H^0 = " + std::to_string(c.lenH0));
    g.check(i == 0 ? nonunit : tp.empty(), "W2 " + cs + ": a non-unit Howell pivot (torsion O/pi) exactly at i = 0");
    // W3 (NB9 / U20): the ordinary part
    const tower::ChainCokernel& Q = W.cokernel();
    const uint32_t h = Q.h();
    const auto trows = TorsionRows(Q);
    std::unique_ptr<tower::HowellZK> T;
    if (!trows.empty()) T = std::make_unique<tower::HowellZK>(O.R, trows, h);
    const uint32_t lenT = T ? T->length() : 0;
    const DM Ud = Dense(tower::OnQuotient(Q, *W.heilbronn(c.ell)));
    uint64_t Te = 1;
    while (Te < static_cast<uint64_t>(O.K) * h) Te *= 2;
    const DM E = MatPow(Ud, Te, O.R);
    auto len_mod_T = [&](const DM& A) {
        auto rows = ColsAsRows(A);
        rows.insert(rows.end(), trows.begin(), trows.end());
        return tower::HowellZK(O.R, rows, h).length() - lenT;
    };
    const uint32_t lenE = len_mod_T(E);
    const uint32_t d = DofA(c.ell, c.a);
    bool freeE = true;
    // free iff length = K * rank(E mod ell): the lifts of rank-many columns independent mod ell span a free direct summand of
    // that length inside the span (a Howell pivot of positive valuation alone does not mean torsion: span(ell, 1) is free)
    if (!T) freeE = lenE == O.K * hecke_ext::tower::RankModEll(E, O.ell);
    g.check(lenE == c.lenE && lenE == c.e * (2 * d + 1) && freeE,
            "W3 (NB9) " + cs + ": e H^1 of length e(2d + 1) = " + std::to_string(c.lenE) + (T ? "" : ", free"));
    // the tower's scalars at g -> zeta
    hecke_ext::tower::TowerOptions to;
    to.R_pred = static_cast<uint32_t>(IPow(c.ell, c.r - 1)) * (2 * d + 1);
    to.deterministic = false;
    to.fl_crosscheck = false;
    to.rel_escapes = false;
    const hecke_ext::tower::TowerReading tw = hecke_ext::tower::BuildTower({c.ell, c.a, c.r, 3}, to);
    const hecke_ext::tower::TowerPart* Ep = tw.part("eis");
    const hecke_ext::tower::TowerPart* Cp = tw.part("cusp");
    bool w3 = Ep != nullptr && (d == 0 || Cp != nullptr);
    uint32_t cusp_len_ok = 1;
    if (w3) {
        auto Sc = [&](const wild::OElem& s) { return Dense(tower::OnQuotient(Q, *W.scalar(s))); };
        std::vector<uint64_t> ps = {2, 3, 5, 7};
        for (uint64_t p : ps) {
            if (p == c.ell) continue;
            const wild::OElem eps_tower = AtZeta(Ep->scalars.at("T" + std::to_string(p)), O);
            const wild::OElem eps_formula = O.add(O.one(), O.mul(O.scalar(static_cast<uint32_t>(p % O.R.Q)), chi.table[p % N]));
            if (eps_tower != eps_formula) w3 = false;
            const DM Tp = Dense(tower::OnQuotient(Q, *W.heilbronn(p)));
            DM A = MatSub(Tp, Sc(eps_formula), O.R);
            if (d == 1) {
                const DM Bc = MatSub(Tp, Sc(AtZeta(Cp->scalars.at("T" + std::to_string(p)), O)), O.R);
                if (p == 2) cusp_len_ok = len_mod_T(MatMul(A, E, O.R)) == 2 * c.e;
                A = MatMul(Bc, A, O.R);
            }
            if (!ColsInSpan(MatMul(A, E, O.R), T.get())) w3 = false;
        }
        DM Au = MatSub(Ud, Sc(O.one()), O.R);
        if (d == 1) Au = MatMul(MatSub(Ud, Sc(AtZeta(Cp->scalars.at("U"), O)), O.R), Au, O.R);
        if (!ColsInSpan(MatMul(Au, E, O.R), T.get())) w3 = false;
    }
    g.check(w3 && cusp_len_ok, "W3 (U20) " + cs + ": T_p (p = 2, 3, 5, 7), U on e H^1 = the tower's a_p, alpha, eps at g -> zeta (Eis 1 + p chi_{i,t}(p))");
    std::cout << "  WILD " << cs << " D=" << W.D() << " K=" << O.K << " lenH1=" << W.len_H1() << " (pred " << c.lenH1 << ") lenH0=" << W.len_H0()
              << " (pred " << c.lenH0 << ") torsion_pivots=" << tp.size() << " len_eH1=" << lenE << " (pred " << c.lenE << ") dimF_ell=" << hF
              << " (EC " << c.EC << ") W3=" << YN(w3 && cusp_len_ok) << " wall="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "s peak_rss_mib=" << oc_util::PeakRssMiB() << "\n";
}

template <class F>
bool Refuses(F f, const std::string& name) {
    try { f(); } catch (const std::exception& ex) { return std::string(ex.what()).find(name) != std::string::npos; }
    return false;
}

} // namespace

uint32_t HeckeNebentypusWildValidation() {
    Checker g;
    std::vector<WildCell> cells = kCells;
    std::cout << "  EFFECTIVE HeckeNebentypusWildValidation arm=DEFAULT cells=" << cells.size() << "\n";
    if (const char* e = std::getenv("HECKE_NEB_WILD_ALL")) {
        ENGINE_CHECK_MSG(std::string(e) == "1", "HECKE_NEB_WILD_ALL must be 1 when set");
        cells.insert(cells.end(), kEnvCells.begin(), kEnvCells.end());
        std::cout << "  EFFECTIVE HeckeNebentypusWildValidation arm=ALL cells=" << kEnvCells.size()
                  << " (O_{2,42} at (7, r = 3); (11, 3), (13, 3) not built: W3 needs the r = 3 tower)\n";
    }
    for (const WildCell& c : cells) RunCell(g, c);
    // W4: the refusals
    {
        bool fq = false;
        for (const auto& chi : hecke::neb::AllCharacters(25))
            if (chi.order() % 5 == 0) {
                const hecke::neb::Fq F = hecke::neb::Fq::Prime(5);
                fq = Refuses([&]() { (void)hecke::neb::ReduceCharacter(chi, F); }, "character_wild");
                break;
            }
        g.check(fq, "W4 character_wild over F_q stays (ReduceCharacter, an order-5 character mod 25)");
        const wild::WildRing O2 = wild::MakeWildRing(5, 2, 20);
        const bool ro = Refuses([&]() { (void)wild::MakeWildCharacter(O2, 25, 0, 1); }, "wild_ring_order");
        const bool pr = Refuses([&]() { (void)wild::MakeWildRing(5, 1, 5); }, "wild_precision");
        const wild::WildRing O1 = wild::MakeWildRing(5, 1, 4);
        const bool tl = Refuses([&]() { (void)wild::MakeWildCharacter(O1, 50, 0, 1); }, "wild_tame_level");
        std::cout << "  NEG character_wild_over_Fq " << (fq ? "caught" : "NOT caught") << "; NEG wild_ring_order (t of order ell^{j+1} into O_j) "
                  << (ro ? "caught" : "NOT caught") << "; wild_precision " << (pr ? "fires" : "does NOT fire") << "; wild_tame_level "
                  << (tl ? "fires" : "does NOT fire") << "\n";
        g.check(ro, "W4 wild_ring_order fires (chi_{0,1} mod 25 into O_2)");
        g.check(pr, "W4 wild_precision fires (e = 5 at phi_1 = 4)");
        g.check(tl, "W4 wild_tame_level fires (N = 50)");
    }
    std::cout << "  HECKE_NEB_WILD checks=" << g.checks << " failing=" << g.fails << " peak_rss_mib=" << oc_util::PeakRssMiB()
              << " verdict=" << (g.fails ? "FAIL" : "OK") << "\n";
    return g.fails;
}
