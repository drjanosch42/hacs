// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-LV) see HeckeLValueValidation.h.
#include "HeckeLValueValidation.h"

#include "Bernoulli.h"
#include "EisensteinBernoulliRoute.h"
#include "LValueRoute.h"
#include "HeckeLValueBirchArms.h"
#include "MBlockReader.h"

#include "../HeckeHidaComponents.h"   // the slot-15 Kummer-Mahler B_k/k mod ell^K: a third route

#include "../../Hecke/HeckeSettings.h"
#include "../../Util/ValidationCheck.h"
#include "../../Util/CellList.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hecke_ext::lvalue;

// (slot-19 WP-BIL) HidaPairingValidationArms.cpp
void RunHidaPairingArms(const std::function<void(bool, const std::string&)>& Check);
// (slot-20 WP-LV3) HeckeLValueLV3Arms.cpp
void RunLV3Arms(const std::function<void(bool, const std::string&)>& Check);
void RunLV3HeavyArms(const std::function<void(bool, const std::string&)>& Check);

namespace {
oc_util::Checker Check("  [FAIL] ");
template <class F> bool ThrowsNotImplemented(F&& f) {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild& e) {
        return std::string(e.what()).rfind("not_implemented_in_this_build:", 0) == 0;
    } catch (...) {
        return false;
    }
    return false;
}
template <class F> bool ThrowsLogic(F&& f) {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild&) {
        return false;
    } catch (const std::logic_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}
std::string Cell(uint32_t n, uint64_t ell) { return "(" + std::to_string(n) + "," + std::to_string(ell) + ")"; }

FreeMat Diag(uint64_t ell, uint32_t K, const std::vector<uint64_t>& d) {
    FreeMat M;
    M.ell = ell;
    M.K = K;
    M.r = static_cast<uint32_t>(d.size());
    M.a.assign(static_cast<size_t>(M.r) * M.r, 0);
    for (uint32_t i = 0; i < M.r; ++i) M.a[static_cast<size_t>(i) * M.r + i] = d[i];
    return M;
}
uint64_t PowU(uint64_t b, uint32_t e) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < e; ++i) r *= b;
    return r;
}

// ---- (slot-19 WP-LV2) ------------------------------------------------------------------------------------------
// The cells and their expected values are those of docs/implementation/slot19/WP-LV2_predictions.md (committed
// before any run); v is re-derived here by the two Bernoulli routes, never copied from a run.
struct TwistCell { uint32_t n; uint64_t ell; uint32_t j; uint32_t v; uint32_t f_minus; };
struct RungCell { uint32_t n; uint64_t ell; uint32_t v; uint32_t f_minus; };
// {2, 37, 34}: n + j == 0 mod ell - 1, so H^0(Gamma_0(37); V_{F_37}(chi)) != 0 and H^1 over Z/37^K has torsion -- the
// twisted free-part reading (a v = 0 control, B_{4,omega^34}/4 == B_2/2 mod 37).
std::vector<TwistCell> DefaultTwistedCells() { return {{2, 37, 28, 1, 1}, {4, 37, 26, 1, 1}, {2, 37, 8, 0, 0}, {2, 37, 34, 0, 0}}; }
std::vector<TwistCell> HeavyTwistedCells() { return {{2, 59, 40, 1, 1}, {2, 67, 54, 1, 1}}; }
// The twisted depth-2 rung (k = 62 == 25 mod 37, chi = omega^6, class 32): its Bernoulli reading is gated; its module run
// (D = 38 * 61 = 2318) peaked above 7.3 GiB under a 6 GiB watchdog (2026-10-04) and is not run.
std::vector<TwistCell> BernoulliOnlyTwistedCells() { return {{60, 37, 6, 2, 1}}; }
// The deep rungs cost ~ 250 s each (n = 282 at 37): env arm only.
std::vector<RungCell> HeavyRungCells() { return {{282, 37, 2, 1}, {226, 103, 2, 1}}; }

// matrix helpers for the precision cross-check: tr(T^i) mod ell^K, i = 1..r (basis-free)
std::vector<uint64_t> PowerTraces(const FreeMat& T) {
    uint64_t q = 1;
    for (uint32_t i = 0; i < T.K; ++i) q *= T.ell;
    const uint32_t r = T.r;
    std::vector<uint64_t> P = T.a, out;
    for (uint32_t it = 1; it <= r; ++it) {
        uint64_t tr = 0;
        for (uint32_t i = 0; i < r; ++i) tr = (tr + P[static_cast<size_t>(i) * r + i]) % q;
        out.push_back(tr);
        std::vector<uint64_t> N(P.size(), 0);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t k = 0; k < r; ++k) {
                const uint64_t a = P[static_cast<size_t>(i) * r + k];
                if (!a) continue;
                for (uint32_t j = 0; j < r; ++j)
                    N[static_cast<size_t>(i) * r + j] = static_cast<uint64_t>(
                        (N[static_cast<size_t>(i) * r + j] + static_cast<unsigned __int128>(a) * T.a[static_cast<size_t>(k) * r + j]) % q);
            }
        P = N;
    }
    return out;
}

// (EF) the free part at finite precision on cheap torsion cells (n = ell + 1: H^1 has ell-torsion exactly there in the probe n <= 38; regular classes, v(B_k/k) = 0): WP-LV's refusal
// without the opt-in; with it, rank H = 2 d_k - 1 on the free part (sign ranks d_k, d_k - 1 are ENGINE_CHECKed), Q = 0,
// and the free part does not depend on the working precision (tr T_p^i agree at K_run and K_run + 2).
void ArmFreePart() {
    struct FC { uint32_t n; uint64_t ell; };
    for (const FC& c : std::vector<FC>{{8, 7}, {24, 23}, {38, 37}}) {
        const std::string tc = Cell(c.n, c.ell);
        EisensteinOptions opt;
        opt.m_max = 3;
        Check(ThrowsNotImplemented([&] { (void)RunEisensteinCell(c.n, c.ell, opt); }),
              "(EF) " + tc + ": without torsion_aware the route refuses the torsion cell (WP-LV's behaviour kept)");
        opt.torsion_aware = true;
        try {
            const EisensteinCell a = RunEisensteinCell(c.n, c.ell, opt);
            opt.min_extra_precision = a.K_run - a.K + 2;
            const EisensteinCell b = RunEisensteinCell(c.n, c.ell, opt);
            std::cout << "  LVALUE_FREE_PART n=" << c.n << " ell=" << c.ell << " K=" << a.K << " K_run=" << a.K_run << " torsion_t=" << a.torsion_t
                      << " torsion_summands=" << a.torsion_summands << " rank=" << a.rank << " cross_K_run=" << b.K_run << std::endl;
            Check(a.torsion_summands > 0 && a.torsion_t >= 1 && a.K_run >= a.K + a.torsion_t,
                  "(EF) " + tc + ": H^1 has ell-torsion; the free part is read at K_run >= K + t");
            bool same = a.T.size() == b.T.size() && PowerTraces(a.iota) == PowerTraces(b.iota);
            for (size_t i = 0; same && i < a.T.size(); ++i) same = PowerTraces(a.T[i]) == PowerTraces(b.T[i]);
            Check(same, "(EF) " + tc + ": tr(T_p^i), tr(iota^i) on (H/tors)/ell^K agree at K_run = " + std::to_string(a.K_run) + " and " +
                            std::to_string(b.K_run));
            const auto pa = std::make_shared<const EisensteinCell>(a);
            for (uint32_t m = 1; m <= 3; ++m) {
                const EisensteinBernoulliRoute route(pa, m);
                const IndexStatement I = route.index_statement();
                Check(I.holds && I.found == 0 && route.bernoulli().v == 0 && a.minus.f == 0,
                      "(EF) " + tc + " m=" + std::to_string(m) + ": regular control on the free part: Q = 0; " + I.to_string());
            }
        } catch (const std::exception& e) {
            Check(false, "(EF) " + tc + " threw: " + e.what());
        }
    }
}

std::string TCell(const TwistCell& c) {
    return "(n=" + std::to_string(c.n) + ",ell=" + std::to_string(c.ell) + ",chi=omega^" + std::to_string(c.j) + ")";
}

// (BT) B_{k,omega^j}/k mod ell^m: the definition route (XT) against Kummer's congruence (KV), and the third route
// (the slot-15 Kummer-Mahler B_{k'}/k') on the k' of (KV); the valuations of the cells; a wrong twist differs.
void ArmTwistedBernoulli() {
    std::vector<TwistCell> all = DefaultTwistedCells();
    for (const TwistCell& c : HeavyTwistedCells()) all.push_back(c);
    for (const TwistCell& c : BernoulliOnlyTwistedCells()) all.push_back(c);
    for (const TwistCell& c : all) {
        const uint32_t k = c.n + 2;
        for (uint32_t m = 1; m <= 3; ++m) {
            const BernoulliReading X = TwistedBernoulliOverK(k, c.ell, c.j, m);
            uint64_t kp = 0;
            const BernoulliReading V = TwistedBernoulliKummer(k, c.ell, c.j, m, &kp);
            const uint64_t q = PowU(c.ell, m);
            const uint64_t euler = (1 + q - PowModU64(c.ell, kp - 1, q)) % q;
            const uint64_t km = hecke_ext::hidacomp::BernoulliOverKModPower(kp, c.ell, m);
            const std::string tag = "(BT) B_{" + std::to_string(k) + ",omega^" + std::to_string(c.j) + "}/" + std::to_string(k) + " mod " +
                                    std::to_string(c.ell) + "^" + std::to_string(m);
            Check(X.residue == V.residue, tag + ": definition == Kummer congruence (k'=" + std::to_string(kp) + ")");
            Check(V.residue == static_cast<uint64_t>((static_cast<unsigned __int128>(km) * euler) % q),
                  tag + ": Kummer congruence on the slot-15 Kummer-Mahler B_{k'}/k'");
            Check(X.v_exact && X.v == c.v, tag + ": v = " + std::to_string(c.v) + " (got " + std::to_string(X.v) + ")");
        }
        const BernoulliReading W = TwistedBernoulliOverK(k, c.ell, c.j + 2 == c.ell - 1 ? 2 : c.j + 2, 2);
        const BernoulliReading R = TwistedBernoulliOverK(k, c.ell, c.j, 2);
        Check(W.residue != R.residue, "(BT) negative: B_{k,omega^{j+2}} != B_{k,omega^j} mod ell^2 at " + TCell(c));
        std::cout << "  BERNOULLI_TWIST k=" << k << " ell=" << c.ell << " j=" << c.j << " a=" << (k + c.j) % (c.ell - 1)
                  << " v(B_{k,chi}/k)=" << R.v << " residue_mod_ell^2=" << R.residue << std::endl;
    }
    // Kummer mod ell: B_{k,omega^j}/k == B_a/a (mod ell), a = k + j (the untwisted gate classes)
    Check(TwistedBernoulliOverK(4, 37, 28, 1).residue == BernoulliModEllPowerExact(32, 37, 1).residue &&
              TwistedBernoulliOverK(4, 37, 8, 1).residue == BernoulliModEllPowerExact(12, 37, 1).residue,
          "(BT) B_{4,omega^28}/4 == B_32/32 and B_{4,omega^8}/4 == B_12/12 (mod 37)");
}

// (ET) [E] cl:lvtwist on the character model, (N6) a wrong twist, (N7) the untwisted character, (N1) the plus side.
void ArmTwisted(const std::vector<TwistCell>& cells, const char* arm) {
    EisensteinOptions opt;
    opt.m_max = 3;
    opt.torsion_aware = true;
    for (const TwistCell& c : cells) {
        std::shared_ptr<const EisensteinCell> cell;
        try {
            cell = std::make_shared<const EisensteinCell>(RunTwistedEisensteinCell(c.n, c.ell, c.j, opt));
        } catch (const std::exception& e) {
            Check(false, std::string("(ET) ") + TCell(c) + " module run threw: " + e.what());
            continue;
        }
        const uint32_t k = c.n + 2;
        std::cout << "  LVALUE_TWIST_CELL arm=" << arm << " n=" << c.n << " ell=" << c.ell << " j=" << c.j << " k=" << k << " K=" << cell->K
                  << " K_run=" << cell->K_run << " torsion_summands=" << cell->torsion_summands << " rank=" << cell->rank
                  << " plus=" << cell->side_plus << " minus=" << cell->side_minus << " f_plus=" << cell->plus.f << " f_minus=" << cell->minus.f
                  << " seconds=" << cell->seconds << std::endl;
        Check(cell->side_plus == cell->side_minus + 2,
              "(ET) " + TCell(c) + ": rank H^+ = rank H^- + 2 (the two Eisenstein classes of Gamma_0(ell) lie on +)");
        Check(cell->plus.f == cell->minus.f + 1, "(ET) " + TCell(c) + ": the m_Eis block of H^+ is f^- + 1 (E_k(1, chi))");
        Check(cell->minus.f == c.f_minus, "(ET) " + TCell(c) + ": rank of the m_Eis block of H^- = " + std::to_string(c.f_minus));
        const uint64_t qK = PowU(c.ell, cell->K);
        for (uint32_t m = 1; m <= 3; ++m) {
            const EisensteinBernoulliRoute route(cell, m);
            std::cout << "  " << LValueLine(route) << std::endl;
            const IndexStatement I = route.index_statement();
            const CongruenceModule Q = route.congruence_module();
            const std::string tm = "(ET) " + TCell(c) + " m=" + std::to_string(m);
            Check(route.kind() == RouteKind::Congruence && route.domain().model == ModelId::CharacterModel && route.domain().level == c.ell,
                  tm + ": a congruence route on the character model at level ell");
            Check(Q.saturated, tm + ": the m_Eis block is saturated");
            Check(I.holds && I.found == std::min(m, c.v), tm + ": #Q = ell^min(m, v(B_{k,chi}/k)) (cl:lvtwist); " + I.to_string());
            Check(route.bernoulli().v == c.v, tm + ": v(B_{k,chi}/k) = " + std::to_string(c.v));
            for (size_t i = 0; i < route.reading().charpoly_valuations.size(); ++i)
                Check(route.reading().charpoly_valuations[i] >= std::min(m, c.v),
                      tm + ": v(c_{k,p}(e_p)) >= min(m, v) at p=" + std::to_string(route.reading().charpoly_primes[i]));
            const SignReading P = ReadSign(*cell, Sign::Plus, m);
            Check(P.Q.length == m, tm + ": (N1) plus side: E_k(1, chi) occurs, T^+/I_e has length m");
            if (c.v >= 1) {
                std::vector<uint64_t> wrong, untw;
                for (uint64_t p : cell->primes) {
                    if (p == c.ell) {
                        wrong.push_back(1);
                        untw.push_back((1 + PowModU64(p, k - 1, qK)) % qK);
                        continue;
                    }
                    const uint64_t cb = TeichmullerPowMod(p, static_cast<uint32_t>(c.ell - 1 - c.j), c.ell, cell->K);   // chibar(p)
                    wrong.push_back((1 + static_cast<uint64_t>(static_cast<unsigned __int128>(cb) * PowModU64(p, k - 1, qK) % qK)) % qK);
                    untw.push_back((1 + PowModU64(p, k - 1, qK)) % qK);
                }
                Check(ReadSign(*cell, Sign::Minus, m, wrong).Q.length == 0,
                      tm + ": (N6) the conjugate character 1 + chibar(p) p^{k-1} gives Q = 0 (the index statement fails)");
                Check(ReadSign(*cell, Sign::Minus, m, untw).Q.length == 0,
                      tm + ": (N7) the untwisted character 1 + p^{k-1} gives Q = 0");
            }
        }
    }
}

// (N8) the two-sided bound of cl:lveis(3) on planted non-saturated algebras: T = Z[ell e] (c_sat = 1) and
// T = Z[X], X = ell + ell^2 e (c_sat = 2, T/(X) = Z/ell^2): min(m - c_sat, 2) <= length(Q_m) <= min(m, 2).
void ArmPlantedTwoSided() {
    const uint64_t ell = 37, e = 5;
    const uint32_t K = 4;
    auto Fake = [&](const FreeMat& op) {
        EisensteinCell c;
        c.n = 10;
        c.ell = ell;
        c.K = K;
        c.primes = {2};
        c.e_values = {e};
        c.minus.sign = Sign::Minus;
        c.minus.f = op.r;
        c.minus.side_rank = op.r;
        c.minus.ops = {op};
        return c;
    };
    {
        const EisensteinCell c = Fake(Diag(ell, K, {e, e + ell}));
        const SignReading R = ReadSign(c, Sign::Minus, 2);
        Check(!R.Q.saturated && R.Q.c_sat == 1, "(N8) Z[ell e]: not saturated, c_sat = 1 (got " + std::to_string(R.Q.c_sat) + ")");
    }
    {
        const EisensteinCell c = Fake(Diag(ell, K, {e + ell, e + ell + ell * ell}));
        bool all = true;
        std::string got;
        for (uint32_t m = 1; m <= K; ++m) {
            const SignReading R = ReadSign(c, Sign::Minus, m);
            const uint32_t lo = m > 2 ? std::min<uint32_t>(m - 2, 2) : 0, hi = std::min<uint32_t>(m, 2);
            all = all && !R.Q.saturated && R.Q.c_sat == 2 && lo <= R.Q.length && R.Q.length <= hi;
            got += (m > 1 ? "," : "") + std::to_string(R.Q.length);
            if (m == K) Check(R.Q.length == 2, "(N8) Z[ell + ell^2 e] at m = K = 4: the bound is sharp, length = 2");
        }
        Check(all, "(N8) Z[ell + ell^2 e]: c_sat = 2 and min(m-2, 2) <= length(Q_m) <= min(m, 2) for m = 1..4 (lengths " + got + ")");
    }
}

// (ED) the deep rungs v(B_k/k) = 2 at n >= ell - 1, where H^1(Gamma; Sym^n Z_ell) has ell-torsion: WP-LV's refusal
// without the opt-in, then the free part (H/tors)/ell^K read at K_run = K + t.
void ArmDeepRungs(const std::vector<RungCell>& cells, const char* arm) {
    for (const RungCell& c : cells) {
        const std::string tc = Cell(c.n, c.ell);
        EisensteinOptions opt;
        opt.m_max = 3;
        opt.torsion_aware = true;   // read H/tors whether or not torsion occurs (it does not at (282, 37): run of 2026-10-04)
        std::shared_ptr<const EisensteinCell> cell;
        try {
            cell = std::make_shared<const EisensteinCell>(RunEisensteinCell(c.n, c.ell, opt));
        } catch (const std::exception& e) {
            Check(false, "(ED) " + tc + " module run threw: " + e.what());
            continue;
        }
        const ModuleFormReading MF = ReadModuleForm(*cell);
        std::cout << "  LVALUE_RUNG_CELL arm=" << arm << " n=" << c.n << " ell=" << c.ell << " k=" << c.n + 2 << " K=" << cell->K
                  << " K_run=" << cell->K_run << " torsion_t=" << cell->torsion_t << " torsion_summands=" << cell->torsion_summands
                  << " exps=" << cell->exps.size() << " rank=" << cell->rank << " plus=" << cell->side_plus << " minus=" << cell->side_minus
                  << " f_plus=" << cell->plus.f << " f_minus=" << cell->minus.f << " module_form_c=" << (MF.applicable ? std::to_string(MF.c) : "n/a")
                  << " (CONJ, not gated) seconds=" << cell->seconds << std::endl;
        Check(cell->K_run >= cell->K + cell->torsion_t, "(ED) " + tc + ": the free part is read at K_run >= K + t");
        Check(cell->minus.f == c.f_minus,
              "(ED) " + tc + ": rank of the m_Eis block of H^- = " + std::to_string(c.f_minus) + " ([E] rmk:lvgates, (U13) PROVED at pre.3)");
        Check(cell->plus.f == cell->minus.f + 1, "(ED) " + tc + ": the m_Eis block of H^+ is f^- + 1");
        for (uint32_t m = 1; m <= 3; ++m) {
            const EisensteinBernoulliRoute route(cell, m);
            std::cout << "  " << LValueLine(route) << std::endl;
            const IndexStatement I = route.index_statement();
            const CongruenceModule Q = route.congruence_module();
            const std::string tm = "(ED) " + tc + " m=" + std::to_string(m);
            Check(route.bernoulli().v == c.v, tm + ": v(B_k/k) = " + std::to_string(c.v));
            Check(I.holds, tm + ": cl:lveis(3) holds (" + I.status + "); " + I.to_string());
            if (Q.saturated) Check(I.found == std::min(m, c.v), tm + ": saturated: #Q = ell^min(m, 2)");
            for (size_t i = 0; i < route.reading().charpoly_valuations.size(); ++i)
                Check(route.reading().charpoly_valuations[i] >= std::min(m, c.v),
                      tm + ": v(c_{k,p}(e_p)) >= min(m, v) at p=" + std::to_string(route.reading().charpoly_primes[i]));
        }
        // (slot-20 WP-LV3) the depth-2 cross-kind against [C]'s own H^1: #Q_3 of the congruence route (on H^- of the C++ module
        // cohomology) == the bilinear (H2) value == the linear (L1) depth of the C++ cone over Z/ell^3 (the default arm reads
        // the linear kind on the group-cocycle W; here on the cone itself)
        try {
            const EisensteinBernoulliRoute route3(cell, 3);
            const uint32_t q3 = route3.index_statement().found;
            const hidapair::HidaPairingCell C(c.n, c.ell, 6, {2, 3, 5});
            mblock::MBlockOptions mo;
            mo.hd_torsion = false;
            mo.joint = mblock::EisensteinJoint(C.R(), c.n);
            const mblock::MBlockReading B = mblock::ReadMBlockCell(C, mblock::EisensteinT2(C.R(), c.n), mo);
            uint32_t bil = 0;
            for (const auto& f : B.sign[0].forms)
                if (!f.eisenstein) bil = f.mixed;
            const BirchModularSymbolRoute lin(std::make_shared<const birch::BirchConeCell>(
                                                  birch::RunBirchConeCell(c.n, c.ell, 3, birch::BirchCellOptions())),
                                              birch::TrivialCharacter(birch::MakeRing(c.ell, 3)), 0);
            const IndexStatement L = lin.index_statement();
            std::cout << "  LV3_CROSS_KIND_HEAVY n=" << c.n << " ell=" << c.ell << " congruence(#Q_3 on [C]'s H^-)=" << q3
                      << " bilinear(H2)=" << bil << " linear(L1 on the C++ cone, Z/ell^3)=" << L.found << " v(B_k/k)=" << c.v << std::endl;
            Check(q3 == c.v && bil == c.v && L.found == c.v && L.holds,
                  "(LV3-D2 heavy) " + tc + ": CROSS-KIND on [C]'s H^1 and the C++ cone: congruence = bilinear = linear = " + std::to_string(c.v));
        } catch (const std::exception& e) {
            Check(false, "(LV3-D2 heavy) " + tc + " threw: " + e.what());
        }
    }
}
} // namespace

uint32_t HeckeLValueValidation() {
    Check.reset();
    const auto t0 = std::chrono::steady_clock::now();

    // ---- (B) Bernoulli numbers ------------------------------------------------------
    {
        Check(TangentNumbers(4).back() == nf::BigInt(272), "(B) T_7 = 272");
        Check(BernoulliEquals(12, true, "691", "2730"), "(B) B_12 = -691/2730");
        Check(BernoulliEquals(24, true, "236364091", "2730"), "(B) B_24 = -236364091/2730");
        Check(BernoulliEquals(32, true, "7709321041217", "510"), "(B) B_32 = -7709321041217/510");
        Check(BernoulliEquals(44, true, "27833269579301024235023", "690"), "(B) B_44 = -27833269579301024235023/690");
        Check(BernoulliEquals(58, false, "84483613348880041862046775994036021", "354"),
              "(B) B_58 = 84483613348880041862046775994036021/354");
        Check(!BernoulliEquals(12, true, "691", "2731") && !BernoulliEquals(12, false, "691", "2730"),
              "(B) negative: a wrong fraction / sign is refused");
        Check(nf::BigInt::from_string("123456789012345678901234567890").to_string() == "123456789012345678901234567890",
              "(B) BigInt decimal round trip");
        // exact vs Voronoi on the gate classes, m = 1..3
        const struct { uint32_t k; uint64_t ell; uint32_t v; } classes[] = {
            {32, 37, 1}, {12, 691, 1}, {44, 59, 1}, {58, 67, 1}, {12, 37, 0}, {24, 37, 0}, {74, 37, 0}};
        for (const auto& c : classes) {
            for (uint32_t m = 1; m <= 3; ++m) {
                if (c.ell == 691 && m == 3) continue;   // 691^3 Voronoi terms: m <= 2 at 691
                const BernoulliReading X = BernoulliModEllPowerExact(c.k, c.ell, m);
                const BernoulliReading V = BernoulliModEllPowerVoronoi(c.k, c.ell, m);
                Check(X.v == c.v && X.v_exact, "(B) v_" + std::to_string(c.ell) + "(B_" + std::to_string(c.k) + "/" + std::to_string(c.k) + ") = " + std::to_string(c.v));
                Check(X.residue == V.residue, "(B) exact == Voronoi: B_" + std::to_string(c.k) + "/k mod " + std::to_string(c.ell) + "^" + std::to_string(m));
                Check(X.residue == hecke_ext::hidacomp::BernoulliOverKModPower(c.k, c.ell, m),
                      "(B) exact == the slot-15 Kummer-Mahler route (HeckeHidaComponents): B_" + std::to_string(c.k) + "/k mod " +
                          std::to_string(c.ell) + "^" + std::to_string(m));
            }
        }
        // the depth-2 rungs ([U] prop:u-eistrunc; [E] rmk:lvgates, referee F1)
        {
            const BernoulliReading X = BernoulliModEllPowerExact(284, 37, 3), V = BernoulliModEllPowerVoronoi(284, 37, 3);
            Check(X.v == 2 && X.residue == V.residue && V.v == 2 && V.v_exact, "(B) rung k=284 at 37: v = 2 (exact and Voronoi)");
            const BernoulliReading X2 = BernoulliModEllPowerExact(228, 103, 3), V2 = BernoulliModEllPowerVoronoi(228, 103, 3);
            Check(X2.v == 2 && X2.residue == V2.residue && V2.v == 2, "(B) rung k=228 at 103: v = 2 (exact and Voronoi)");
            for (uint64_t k : {342252ull, 342440ull}) {
                const BernoulliReading R = BernoulliModEllPowerVoronoi(k, 691, 2);
                Check(R.residue == 0, "(B) rung k=" + std::to_string(k) + " at 691: B_k/k == 0 mod 691^2");
                const uint64_t km = hecke_ext::hidacomp::BernoulliOverKModPower(k, 691, 3);
                Check(km != 0 && ValuationU64(km, 691) == 2,
                      "(B) rung k=" + std::to_string(k) + " at 691: v = 2 exactly (Kummer-Mahler mod 691^3)");
                for (uint64_t kk : {k - 690, k + 690}) {
                    const BernoulliReading S = BernoulliModEllPowerVoronoi(kk, 691, 2);
                    Check(S.v == 1 && S.v_exact, "(B) k=" + std::to_string(kk) + " at 691: v = 1");
                }
            }
            std::cout << "  BERNOULLI exact_vs_voronoi=agree rungs 284@37:v=2 228@103:v=2 342252@691:v=2 342440@691:v=2" << std::endl;
        }
    }

    // ---- (E) + (N1, N2) the Eisenstein instance on the gate cells ----------------------
    {
        const struct { uint32_t n; uint64_t ell; uint32_t v; uint32_t f_minus; } cells[] = {
            {30, 37, 1, 1}, {10, 691, 1, 1}, {42, 59, 1, 1}, {56, 67, 1, 1}, {10, 37, 0, 0}, {22, 37, 0, 0}};
        EisensteinOptions opt;
        opt.m_max = 3;
        for (const auto& c : cells) {
            std::shared_ptr<const EisensteinCell> cell;
            try {
                cell = std::make_shared<const EisensteinCell>(RunEisensteinCell(c.n, c.ell, opt));
            } catch (const std::exception& e) {
                Check(false, "(E) " + Cell(c.n, c.ell) + " module run threw: " + e.what());
                continue;
            }
            const uint32_t k = c.n + 2;
            std::cout << "  LVALUE_CELL n=" << c.n << " ell=" << c.ell << " k=" << k << " K=" << cell->K << " rank=" << cell->rank
                      << " plus=" << cell->plus.side_rank << " minus=" << cell->minus.side_rank << " f_plus=" << cell->plus.f
                      << " f_minus=" << cell->minus.f << " seconds=" << cell->seconds << std::endl;
            Check(cell->minus.f == c.f_minus, "(E) " + Cell(c.n, c.ell) + " rank of the m_Eis block of H^- = " + std::to_string(c.f_minus));
            Check(cell->plus.f == c.f_minus + 1, "(E) " + Cell(c.n, c.ell) + " rank of the m_Eis block of H^+ = f^- + 1 (the Eisenstein line)");
            {
                const ModuleFormReading MF = ReadModuleForm(*cell);
                std::cout << "  LVALUE_MODULE_FORM n=" << c.n << " ell=" << c.ell << " f_plus=" << MF.f_plus
                          << " applicable=" << (MF.applicable ? "yes" : "no") << " c=" << MF.c << " primes_read=" << MF.primes.size()
                          << " v(B_k/k)=" << c.v << std::endl;
                Check(MF.applicable && MF.c == c.v,
                      "(E) " + Cell(c.n, c.ell) + ": module form cl:lveis(2), [H^+ : H^+_! (+) Z_ell E] = ell^v(B_k/k)");
            }
            for (uint32_t m = 1; m <= 3; ++m) {
                const EisensteinBernoulliRoute route(cell, m);
                std::cout << "  " << LValueLine(route) << std::endl;
                const IndexStatement I = route.index_statement();
                const CongruenceModule Q = route.congruence_module();
                Check(route.kind() == RouteKind::Congruence, "(E) the instance is of kind congruence");
                Check(Q.saturated, "(E) " + Cell(c.n, c.ell) + " m=" + std::to_string(m) + ": the m_Eis block is saturated (stable at K=1)");
                Check(I.holds && I.found == std::min(m, c.v),
                      "(E) " + Cell(c.n, c.ell) + " m=" + std::to_string(m) + ": #Q = ell^min(m, v(B_k/k)) (cl:lveis(3)); " + I.to_string());
                Check(route.bernoulli().v == c.v, "(E) " + Cell(c.n, c.ell) + ": v(B_k/k) = " + std::to_string(c.v));
                for (size_t i = 0; i < route.reading().charpoly_valuations.size(); ++i)
                    Check(route.reading().charpoly_valuations[i] >= std::min(m, c.v),
                          "(E) " + Cell(c.n, c.ell) + " m=" + std::to_string(m) + ": v(c_{k,p}(e_p)) >= min(m, v) at p=" +
                              std::to_string(route.reading().charpoly_primes[i]));
                // (N1) the plus side: e occurs there, T^+/I_e = Z_ell, so the naive quotient has length m
                const SignReading P = ReadSign(*cell, Sign::Plus, m);
                Check(P.Q.saturated && P.Q.length == m,
                      "(N1) " + Cell(c.n, c.ell) + " m=" + std::to_string(m) + ": plus side T^+/I_e has length m (got " + std::to_string(P.Q.length) + ")");
                if (m > c.v)
                    Check(P.Q.length != std::min(m, c.v), "(N1) the plus side does not satisfy the index statement (the gate discriminates)");
                // (N2) a wrong-weight character 1 + p^{k+1} on a v >= 1 cell: Q = 0
                if (c.v >= 1) {
                    std::vector<uint64_t> wrong;
                    uint64_t q = 1;
                    for (uint32_t i = 0; i < cell->K; ++i) q *= c.ell;
                    for (uint64_t p : cell->primes) wrong.push_back((1 + PowModU64(p, k + 1, q)) % q);
                    const SignReading W = ReadSign(*cell, Sign::Minus, m, wrong);
                    Check(W.Q.length == 0 && W.Q.length != std::min(m, c.v),
                          "(N2) " + Cell(c.n, c.ell) + " m=" + std::to_string(m) + ": the wrong-weight character gives Q = 0 (index statement fails)");
                }
                // the kind refusals of the interface
                Check(ThrowsLogic([&] { (void)route.evaluate_functional({}); }) &&
                          ThrowsLogic([&] { (void)route.evaluate_pairing({}, {}); }),
                      "(N4) a congruence route refuses the linear and bilinear evaluations");
            }
        }
    }

    // ---- (N3) a planted non-saturated algebra: T = Z_ell[ell e] in Z_ell x Z_ell ----------
    {
        const uint64_t ell = 37;
        const uint32_t K = 4;
        Check(PlantedDimA1({Diag(ell, K, {0, ell})}) == 1, "(N3) Z[ell e]: dim A_1 = 1 < rank 2 (not saturated)");
        Check(PlantedDimA1({Diag(ell, K, {0, 1})}) == 2, "(N3) Z[e] = Z x Z: dim A_1 = 2 (saturated)");
    }

    // ---- (N4, N5) the stubs and the domain refusal -------------------------------------------
    {
        Domain d;
        d.model = ModelId::SL2ZSevenVertex;
        d.cohomology = "H^1_c";
        d.n_sym = 10;
        d.ell = 691;
        d.m = 1;
        d.subquotient = "H^1_c";
        const BirchModularSymbolRoute birchr(d, 1, 0);
        Check(birchr.kind() == RouteKind::Linear, "(N4) Birch is linear");
        // (slot-19 WP-LIN) the linear instance is implemented: the (BS-*) arms below gate it
        Check(birchr.evaluate_functional({}).empty() && !ThrowsNotImplemented([&] { (void)birchr.evaluate_functional({}); }),
              "(N4) Birch is implemented (slot-19 WP-LIN): no NotImplementedInThisBuild");
        Check(ThrowsLogic([&] { (void)birchr.congruence_module(); }), "(N4) Birch refuses the congruence evaluation (kind)");
        d.cohomology = "H^1_!";
        const HidaPairingRoute hida(d);
        Check(hida.kind() == RouteKind::Bilinear && !hida.adjunction().empty(), "(N4) Hida is bilinear with a named adjunction");
        // (slot-19 WP-BIL) Hida is implemented: a wrong-shaped class is a wrong call (logic_error), a foreign model is refused
        Check(ThrowsLogic([&] { (void)hida.evaluate_pairing({}, {}); }), "(N4) Hida refuses an empty class (logic_error)");
        {
            Domain hm = d;
            hm.model = ModelId::MoerdijkPronk;
            const HidaPairingRoute hida_mp(hm);
            Check(ThrowsNotImplemented([&] { (void)hida_mp.index_statement(); }), "(N4) Hida refuses the Moerdijk-Pronk model by name");
        }
        Check(ThrowsLogic([&] { (void)hida.evaluate_functional({}); }), "(N4) Hida refuses the linear evaluation (kind)");
        Domain tw = d;
        tw.model = ModelId::CharacterModel;
        tw.cohomology = "H^1(Gamma_0(N);V(chi))";
        tw.level = 37;
        Check(ThrowsNotImplemented([&] { EisensteinBernoulliRoute::RequireSupportedDomain(tw); }),
              "(N5) the Eisenstein route refuses the character model at a level other than ell (37 != 691)");
        // (slot-19 WP-LV2) the character model at level ell is supported (cl:lvtwist); other models are refused by name
        Domain tw2 = tw;
        tw2.ell = 37;
        Check(!ThrowsNotImplemented([&] { EisensteinBernoulliRoute::RequireSupportedDomain(tw2); }),
              "(N5) the Eisenstein route accepts H^1(Gamma_0(ell); V(chi)) at level ell (cl:lvtwist)");
        Domain mp = d;
        mp.model = ModelId::MoerdijkPronk;
        mp.cohomology = "H^1";
        Domain g0 = tw2;
        g0.model = ModelId::Gamma0NShapiro;
        Check(ThrowsNotImplemented([&] { EisensteinBernoulliRoute::RequireSupportedDomain(mp); }) &&
                  ThrowsNotImplemented([&] { EisensteinBernoulliRoute::RequireSupportedDomain(g0); }),
              "(N5) the Eisenstein route refuses the Moerdijk-Pronk and the Gamma_0(N)-Shapiro models by name");
        std::cout << "  NORMALISATION (B) " << birchr.normalisation().to_string() << std::endl;
        std::cout << "  NORMALISATION (H) " << hida.normalisation().to_string() << std::endl;
    }

    // ---- (slot-19 WP-LV2) (BT) twisted Bernoulli numbers, (ET) cl:lvtwist, (ED) the deep rungs, (N6)-(N8) --------
    // ---- (slot-19 WP-BIL) the bilinear kind: Hida's GL2 x GL2 pairing, gates G1-G5 + negative controls -----------------
    RunHidaPairingArms([](bool ok, const std::string& what) { Check(ok, what); });

    ArmTwistedBernoulli();
    ArmTwisted(DefaultTwistedCells(), "default");
    ArmPlantedTwoSided();
    ArmFreePart();
    {
        // (slot-19 WP-LIN) the linear instance: BirchModularSymbolRoute, R19-8 Sec. 6's G1-G5 and the isign facts at (258, 37)
        const char* henv = std::getenv("HECKE_LVALUE_HEAVY");
        const bool heavy = henv && std::string(henv) == "1";
        if (heavy) std::cout << "  EFFECTIVE HeckeLValueValidation arm=HEAVY_BIRCH cells=1" << std::endl;
        RunBirchArms([](bool ok, const std::string& what) { Check(ok, what); }, heavy);
    }
    // (slot-20 WP-LV3) the generic m-block reader, (H2)/(L1) at depth 2 and the cross-kind gate, R20-3's change of lattice,
    // iota^* over Z/ell^k
    RunLV3Arms([](bool ok, const std::string& what) { Check(ok, what); });
    {
        // read-only env arm: heavier cells (documented in docs/validation-catalogue.md's env table)
        const char* env = std::getenv("HECKE_LVALUE_HEAVY");
        if (env && std::string(env) == "1") {
            const std::vector<TwistCell> tw = HeavyTwistedCells();
            const std::vector<RungCell> ru = HeavyRungCells();
            std::cout << "  EFFECTIVE HeckeLValueValidation arm=HEAVY cells=" << tw.size() + ru.size() << std::endl;
            ArmTwisted(tw, "heavy");
            ArmDeepRungs(ru, "heavy");
            RunLV3HeavyArms([](bool ok, const std::string& what) { Check(ok, what); });
        }
        // (slot-20 WP-LV3) the twisted depth-2 rung (37; omega^6, 62) on the character model: its own arm value (it is NOT in
        // HECKE_LVALUE_HEAVY=1; RSS predicted from the measured curve at n = 20, 40 -- docs/implementation/slot20/WP-LV3_report.md)
        if (env && std::string(env) == "twist62") {
            const std::vector<TwistCell> tw = BernoulliOnlyTwistedCells();
            std::cout << "  EFFECTIVE HeckeLValueValidation arm=HEAVY_TWIST62 cells=" << tw.size() << std::endl;
            ArmTwisted(tw, "twist62");
        }
    }
    // read-only cell hook (R19-7's request): HECKE_LVALUE_CELLS="n:ell[:j],..." -- level one (no j) or the character model
    // chi = omega^j; torsion-aware, m = 1..3; gates only what holds on every cell (cl:lveis(3) / cl:lvtwist with the
    // Bernoulli valuation computed in-house, I.holds), no expected value of its own.  A malformed entry is a failure.
    if (const char* env = std::getenv("HECKE_LVALUE_CELLS")) {
        std::vector<std::vector<uint64_t>> cells;
        const std::string s(env);
        bool bad = !s.empty() && s.back() == ',';   // an empty trailing cell is malformed (getline would swallow it)
        for (const oc_util::cells::Token& t : oc_util::cells::Tokenize(s)) {
            if (!t.Fits(2, 3)) bad = true;
            else cells.push_back(t.fields);
        }
        Check(!bad, "(EC) HECKE_LVALUE_CELLS parses as n:ell[:j],...");
        if (!bad) {
            std::cout << "  " << oc_util::cells::EffectiveLine("HeckeLValueValidation", "CELLS", cells.size()) << std::endl;
            for (const auto& f : cells) {
                EisensteinOptions opt;
                opt.m_max = 3;
                opt.torsion_aware = true;
                try {
                    const auto cell = std::make_shared<const EisensteinCell>(
                        f.size() == 2 ? RunEisensteinCell(static_cast<uint32_t>(f[0]), f[1], opt)
                                      : RunTwistedEisensteinCell(static_cast<uint32_t>(f[0]), f[1], static_cast<uint32_t>(f[2]), opt));
                    std::cout << "  LVALUE_HOOK_CELL n=" << f[0] << " ell=" << f[1] << (f.size() == 3 ? " j=" + std::to_string(f[2]) : "")
                              << " K_run=" << cell->K_run << " torsion_summands=" << cell->torsion_summands << " rank=" << cell->rank
                              << " plus=" << cell->side_plus << " minus=" << cell->side_minus << " f_plus=" << cell->plus.f
                              << " f_minus=" << cell->minus.f << " seconds=" << cell->seconds << std::endl;
                    for (uint32_t m = 1; m <= 3; ++m) {
                        const EisensteinBernoulliRoute route(cell, m);
                        std::cout << "  " << LValueLine(route) << std::endl;
                        Check(route.index_statement().holds, "(EC) " + Cell(static_cast<uint32_t>(f[0]), f[1]) + " m=" + std::to_string(m) +
                                                                 ": the index statement holds; " + route.index_statement().to_string());
                    }
                } catch (const std::exception& e) {
                    Check(false, "(EC) " + Cell(static_cast<uint32_t>(f[0]), f[1]) + " threw: " + e.what());
                }
            }
        }
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  HeckeLValueValidation: checks=" << Check.checks << " failing=" << Check.fails << " seconds=" << secs << std::endl;
    return Check.fails ? 1 : 0;
}
