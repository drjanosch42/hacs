// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 20, WP-MEAS) HeckeLValueMeasureValidation -- see HeckeLValueMeasureValidation.h.
// Gate [E] 1.20.0-pre.1 chk:lvmeas; spec docs/implementation/slot20/WP-MEAS_spec.md Sec. 2; predictions
// docs/implementation/slot20/wpmeas/predictions.md (Q1-Q9, committed before the first run); the numbers of record are
// R20-1's (MeasureFrozenR201.inc), never this code's output.
#include "HeckeLValueMeasureValidation.h"

#include "Bernoulli.h"
#include "BirchModularSymbol.h"
#include "EisensteinBernoulliRoute.h"
#include "LFunctionRecord.h"
#include "LValueRoute.h"
#include "ModularSymbolMeasure.h"

#include "../../Hecke/HeckeSettings.h"   // hecke::NotImplementedInThisBuild
#include "../../../ModArithmetic/EngineCheck.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {

namespace {
#include "MeasureFrozenR201.inc"
#include "MeasureFrozenS2.inc"

using namespace birch;

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
double PeakRssGb() {
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
}
uint64_t PowU(uint64_t b, uint32_t e) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < e; ++i) r *= b;
    return r;
}

struct Log {
    uint32_t checks = 0, failing = 0;
    std::map<std::string, uint32_t> per_arm;
    void check(const std::string& arm, bool ok, const std::string& what) {
        ++checks;
        ++per_arm[arm];
        if (!ok) {
            ++failing;
            std::cout << "  FAIL (" << arm << ") " << what << std::endl;
        }
    }
};

const FrozenMeasureCell* FindFrozen(uint32_t n, uint64_t ell) {
    for (const auto& f : kFrozenMeasure)
        if (f.n == n && f.ell == ell) return &f;
    return nullptr;
}
const FrozenTotals* FindTotals(uint32_t n, uint64_t ell) {
    for (const auto& f : kFrozenTotals)
        if (f.n == n && f.ell == ell) return &f;
    for (const auto& f : kFrozenTotalsS2)   // (slot-20 S2) the 691 arm's predictions
        if (f.n == n && f.ell == ell) return &f;
    return nullptr;
}
const FrozenInvariants* FindInv(uint32_t n, uint64_t ell, int sign) {
    for (const auto& f : FrozenInvariantTables())
        if (f.n == n && f.ell == ell && f.sign == sign) return &f;
    for (const auto& f : FrozenInvariantTablesS2())   // (slot-20 S2)
        if (f.n == n && f.ell == ell && f.sign == sign) return &f;
    return nullptr;
}

// (mu, lambda) on the branches of parity `parity` (0 even, 1 odd), against a frozen table; returns the mismatch text ("" = equal)
// and the non-(0,0) list for the MEAS_INV line.
std::string CheckInvariants(const ModularSymbolMeasure& me, uint32_t parity, const FrozenInvariants* fr, std::string& listing) {
    std::ostringstream bad, lst;
    for (uint32_t a = 0; a + 1 < me.ell; ++a) {
        if (a % 2 != parity) continue;
        const IwasawaInvariants I = me.invariants(a);
        if (I.mu != 0 || I.lambda != 0) lst << a << ":(" << I.mu << "," << I.lambda << ") ";
        if (!fr) continue;
        uint32_t emu = fr->def_mu, elam = fr->def_lambda;
        for (const auto& e : fr->except)
            if (e.branch == a) {
                emu = e.mu;
                elam = e.lambda;
            }
        if (I.mu != emu || I.lambda != elam)
            bad << " branch " << a << " (" << I.mu << "," << I.lambda << ") != (" << emu << "," << elam << ")";
    }
    listing = lst.str();
    if (!fr) return "no frozen table";
    return bad.str();
}

// ---- (M4a) the Kubota-Leopoldt group rings at one ell --------------------------------------------------------------
struct KlRoutes {
    uint64_t ell = 0;
    uint32_t M = 0, R = 0;
    std::map<uint32_t, std::vector<uint64_t>> bern, layer;
};
KlRoutes RunKl(uint64_t ell, uint32_t M, uint32_t R, Log& log, bool gate) {
    const double t0 = Now();
    KlRoutes K;
    K.ell = ell;
    K.M = M;
    K.R = R;
    std::vector<uint32_t> classes;
    for (uint32_t b = 2; b + 1 < ell - 1 + 1; b += 2)
        if (b % (ell - 1) != 0) classes.push_back(b);
    for (uint32_t b : classes) K.bern[b] = KlGroupRingBernoulli(b, ell, M, R);
    const double t1 = Now();
    K.layer = KlGroupRingLayer(classes, ell, M, 0);
    const double t2 = Now();
    if (!gate) return K;
    const uint64_t pm = PowU(ell, std::min(M, R));
    uint32_t agree = 0;
    for (uint32_t b : classes) {
        const auto& x = K.bern.at(b);
        const auto it = K.layer.find(b);
        bool ok = it != K.layer.end() && it->second.size() == x.size();
        for (size_t i = 0; ok && i < x.size(); ++i) ok = (x[i] % pm) == (it->second[i] % pm);
        agree += ok;
    }
    log.check("M4", agree == classes.size(),
              "(M4a) ell=" + std::to_string(ell) + ": E_{1,c} == CwLayer on " + std::to_string(agree) + " / " + std::to_string(classes.size()));
    // both evaluate to -(1 - ell^{k-1}) B_k/k at k = b + t(ell - 1), t = 0..3
    const BernoulliOverKTable BT(ell, std::min(M, R), classes.back() + 3 * static_cast<uint32_t>(ell - 1));
    uint32_t ev_ok = 0, ev_tot = 0;
    for (uint32_t b : classes)
        for (uint32_t t = 0; t < 4; ++t) {
            const uint32_t k = b + t * static_cast<uint32_t>(ell - 1);
            const uint64_t euler = (1 + pm - PowModU64(ell % pm, k - 1, pm)) % pm;
            const uint64_t ref = (pm - static_cast<uint64_t>(static_cast<unsigned __int128>(euler) * BT.residue(k) % pm)) % pm;
            ++ev_tot;
            ev_ok += (GrEval(K.bern.at(b), k, ell, pm) == ref) && (GrEval(K.layer.at(b), k, ell, pm) == ref);
        }
    log.check("M4", ev_ok == ev_tot && ev_tot >= 4 * classes.size(),
              "(M4a) ell=" + std::to_string(ell) + ": both routes == -(1-ell^{k-1})B_k/k at " + std::to_string(ev_ok) + " / " + std::to_string(ev_tot));
    std::cout << "MEAS_KL ell=" << ell << " M=" << M << " classes=" << classes.size() << " routes_agree=" << agree << "/" << classes.size()
              << " bernoulli_values=" << ev_ok << "/" << ev_tot << " e1c_secs=" << (t1 - t0) << " layer_secs=" << (t2 - t1) << std::endl;
    return K;
}

// cross-checks of this TU's Bernoulli tables against the slot-19 library routes (exact / twisted (XT))
void RunBernoulliCrossChecks(Log& log) {
    uint32_t ok = 0, tot = 0;
    for (uint64_t ell : std::vector<uint64_t>{37, 103}) {
        const BernoulliOverKTable BT(ell, 3, 300);
        for (uint32_t k : {12u, 24u, 32u, 228u, 284u}) {
            if (k % (ell - 1) == 0) continue;
            ++tot;
            ok += BT.residue(k) == BernoulliModEllPowerExact(k, ell, 3).residue;
        }
    }
    log.check("M4", ok == tot && tot >= 8, "(M4a) BernoulliOverKTable == BernoulliModEllPowerExact at " + std::to_string(ok) + " / " + std::to_string(tot));
    ok = tot = 0;
    for (uint64_t ell : std::vector<uint64_t>{37, 103}) {
        const TwistedBernoulliTable TB(ell, 3, 40);
        for (uint32_t r = 1; r <= 30; ++r)
            for (uint32_t i : {1u, 2u, 5u, 8u}) {
                if ((r + i) % 2 != 0 || (r + i) % (ell - 1) == 0) continue;
                const LadicReading L = TB(r, i);
                const BernoulliReading lib = TwistedBernoulliOverK(r, ell, i, 3);
                const uint64_t q3 = PowU(ell, 3);
                const uint64_t mine = (L.zero || L.v >= 3) ? 0 : static_cast<uint64_t>(static_cast<unsigned __int128>(L.unit) * PowU(ell, static_cast<uint32_t>(L.v)) % q3);
                ++tot;
                ok += (!L.zero && L.v >= 0) && mine == lib.residue;
            }
    }
    log.check("M4", ok == tot && tot >= 100, "(M4a) TwistedBernoulliTable == TwistedBernoulliOverK at " + std::to_string(ok) + " / " + std::to_string(tot));
    std::cout << "MEAS_KL bernoulli_cross_checks twisted=" << ok << "/" << tot << std::endl;
}

// ---- one cell ------------------------------------------------------------------------------------------------------
struct CellSpec {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t d = 0, R = 4, M = 2;
    std::map<uint64_t, uint64_t> sep;
    bool control = false;
    bool all_chars = false;
    bool heavy = false;
};
struct CellOut {
    SymbolCell C;
    uint64_t alpha = 0;
    ModularSymbolMeasure mm, mp;
    bool ok = false;
    uint32_t pole_points = 0, pole_unit_diff = 0;   // (slot-20 S2) pole points, and those with v(lhs - rhs) = 0
    uint64_t s = 0;                                 // r^- == s P^Eis (Eisenstein cells)
};

// ---- (M7) lfunction/v1 (slot-20 S1; predictions docs/implementation/slot20/s1s2/predictions.md P-S1-1..4) -----------------
// Per sign: the record of every branch of the right parity; (1) the T-polynomial evaluated at u^j - 1 == the group-ring branch
// value, every j <= n (exact mod ell^R); (2) at level M >= 3: the level-M element folded mod Gamma^{ell^{M-2}} == the level-(M-1)
// element from the level-(M-1) balls, and the level-(M-1) coefficients == the level-M ones mod ell^{prec_{M-1}(t)}, t < ell
// (lem:lvlfunprec against a deeper truncation); (3) (mu, lambda) recomputed from the record's coefficients == invariants(a), and
// lambda_mod_ell == lambda where mu = 0; (4) the canonical text is deterministic.
void RunLFunctionArm(const CellSpec& sp, const CellOut& out, Log& log) {
    const SymbolCell& C = out.C;
    const uint32_t n = sp.n;
    const uint64_t ell = sp.ell;
    const std::string tag = "(" + std::to_string(n) + "," + std::to_string(ell) + ") R=" + std::to_string(sp.R) + " M=" + std::to_string(sp.M);
    for (int sign : {-1, +1}) {
        const double t0 = Now();
        const ModularSymbolMeasure& me = sign < 0 ? out.mm : out.mp;
        const uint64_t q = me.q;
        LFunctionForm form;
        std::string sepw;
        for (const auto& pc : sp.sep) sepw += (sepw.empty() ? "" : "+") + std::to_string(pc.second) + "T_" + std::to_string(pc.first);
        form.separator = sepw;
        form.target_root = C.target;
        form.eisenstein_congruent = C.eisenstein_congruent;
        form.a_p = C.a_p;
        form.sign = sign < 0 ? "minus" : "plus";
        form.symbol_normalisation = sign < 0 ? "r^- first unit coordinate 1" : "r^+_0 = 1";
        std::vector<uint32_t> bs;
        for (uint32_t a = (sign < 0 ? 1u : 0u); a + 1 < ell; a += 2) bs.push_back(a);
        const LFunctionRecord rec = MakeLFunctionRecord(me, form, bs);
        const std::string st = tag + (sign < 0 ? " r^-" : " r^+");
        // (1) values
        uint32_t vbad = 0, vtot = 0;
        std::vector<uint64_t> xj(n + 1);
        {
            uint64_t uj = 1 % q;
            for (uint32_t j = 0; j <= n; ++j) {
                xj[j] = (uj + q - 1 % q) % q;
                uj = MulModFast(uj, (1 + ell) % q, q);
            }
        }
        for (const LFunctionBranch& b : rec.branches) {
            const std::vector<uint64_t> g = me.branch(b.a);
            for (uint32_t j = 0; j <= n; ++j) {
                ++vtot;
                vbad += EvalTPolynomial(b.coefficients, xj[j], q) != GrEval(g, j, ell, q);
            }
        }
        log.check("M7", vbad == 0 && vtot == bs.size() * (n + 1),
                  st + ": (P-S1-1) L_a(u^j-1) != branch value at " + std::to_string(vbad) + " / " + std::to_string(vtot));
        // (2) truncation, level M-1 against level M
        uint32_t fold_bad = 0, prec_bad = 0, prec_tot = 0, prec_sharp = 0;
        if (sp.M >= 3) {
            const uint64_t Gs = PowU(ell, sp.M - 2);
            const std::vector<uint32_t> prs = CertifiedTPrecision(ell, sp.R, sp.M - 1);
            for (const LFunctionBranch& b : rec.branches) {
                const std::vector<uint64_t> gs = BranchElementAtLevel(me, b.a, sp.M - 1);
                fold_bad += FoldBranchElement(me.branch(b.a), Gs, q) != gs;
                const std::vector<uint64_t> cs = TCoefficients(gs, q);
                for (uint64_t t = 0; t < std::min<uint64_t>(Gs, ell); ++t) {
                    ++prec_tot;
                    const uint64_t pe = PowU(ell, prs[t]);
                    const uint64_t df = (cs[t] + q - b.coefficients[t]) % q;
                    prec_bad += df % pe != 0;
                    prec_sharp += (prs[t] < sp.R) && (df % (pe * ell) != 0);
                }
            }
            log.check("M7", fold_bad == 0, st + ": (P-S1-2) fold(level M) != level M-1 at " + std::to_string(fold_bad) + " branches");
            log.check("M7", prec_bad == 0 && prec_tot > 0,
                      st + ": (P-S1-2) level-(M-1) coefficients off the level-M ones beyond the certified precision at " + std::to_string(prec_bad) +
                          " / " + std::to_string(prec_tot));
        }
        // (3) invariants recomputed from the coefficients
        uint32_t ibad = 0;
        for (const LFunctionBranch& b : rec.branches) {
            const uint64_t tmax = std::min<uint64_t>(b.coefficients.size(), ell);
            uint32_t best = me.R, lam = 0;
            for (uint64_t t = 0; t < tmax; ++t) {
                uint64_t x = b.coefficients[t];
                uint32_t v = 0;
                if (x == 0) {
                    v = me.R;
                } else {
                    while (x % ell == 0 && v < me.R) {
                        x /= ell;
                        ++v;
                    }
                }
                if (v < best) {
                    best = v;
                    lam = static_cast<uint32_t>(t);
                }
            }
            bool ok = best == b.mu && lam == b.lambda;
            if (b.mu == 0 && b.readable) ok = ok && b.lambda_mod_ell == static_cast<int64_t>(b.lambda);
            ibad += !ok;
        }
        log.check("M7", ibad == 0, st + ": (P-S1-3) (mu, lambda) from the record's coefficients != invariants() at " + std::to_string(ibad) + " branches");
        // (4) determinism
        const LFunctionRecord rec2 = MakeLFunctionRecord(me, form, bs);
        log.check("M7", rec2.canonical() == rec.canonical(), st + ": (P-S1-4) the canonical text is not deterministic");
        std::ostringstream h;
        h << std::hex << rec.hash();
        std::cout << "MEAS_LFUN cell=" << tag << " sign=" << (sign < 0 ? "-" : "+") << " schema=" << LFunctionRecord::kSchema << " branches=" << bs.size()
                  << " G=" << rec.G << " values_bad=" << vbad << "/" << vtot << " fold_bad=" << fold_bad << " prec_bad=" << prec_bad << "/" << prec_tot
                  << " prec_sharp=" << prec_sharp << " inv_bad=" << ibad << " hash=0x" << h.str() << " secs=" << (Now() - t0) << std::endl;
    }
}

CellOut RunCell(const CellSpec& sp, Log& log, const KlRoutes* kl) {
    CellOut out;
    const double t0 = Now();
    const uint32_t n = sp.n, k = n + 2;
    const uint64_t ell = sp.ell;
    const std::string tag = "(" + std::to_string(n) + "," + std::to_string(ell) + ") R=" + std::to_string(sp.R) + " M=" + std::to_string(sp.M);
    out.C = RunSymbolCell(n, ell, sp.R, sp.sep, sp.control);
    const SymbolCell& C = out.C;
    if (!C.simple || !C.have_r_plus) {
        log.check("M1", false, tag + ": the symbol cell did not run: " + C.refusal + (C.have_r_plus ? "" : " (no r^+)"));
        return out;
    }
    const Ring& Rg = C.ring;
    const uint64_t q = Rg.q;
    const SymRho rho(n, Rg);
    const uint64_t a_ell = C.a_p.at(ell);
    out.alpha = UnitRoot(Rg, a_ell, k);
    const uint64_t alpha = out.alpha;
    const double t1 = Now();
    out.mm = BuildMeasure(rho, C.r_minus, a_ell, alpha, sp.M, true);
    out.mp = BuildMeasure(rho, C.r_plus, a_ell, alpha, sp.M, true);
    const double t2 = Now();
    const FrozenMeasureCell* fr = FindFrozen(n, ell);
    const FrozenTotals* tots = FindTotals(n, ell);
    std::cout << "MEAS_CELL n=" << n << " ell=" << ell << " k=" << k << " d=" << sp.d << " R=" << sp.R << " M=" << sp.M
              << " dimW-=" << C.Wminus.size() << " dimW+=" << C.Wplus.size() << " a_2=" << C.a_p.at(2) << " a_3=" << C.a_p.at(3)
              << " a_ell=" << a_ell << " alpha=" << alpha << " eisenstein=" << (C.eisenstein_congruent ? 1 : 0)
              << " symbol_secs=" << (t1 - t0) << " table_secs=" << (t2 - t1) << std::endl;
    const uint64_t prec = PowU(ell, std::min(sp.M, sp.R));
    const uint64_t ia = Rg.inv(alpha);
    std::vector<uint32_t> chars;
    if (sp.all_chars) {
        for (uint32_t i = 0; i + 1 < ell; ++i) chars.push_back(i);
    } else {
        chars = {0, 1, 2, 3, static_cast<uint32_t>((ell - 1) / 2), static_cast<uint32_t>(ell - 2)};
    }
    std::map<int, std::vector<std::vector<uint64_t>>> branches;   // sign -> a -> g

    // ---- (M1) ----
    for (int sign : {-1, +1}) {
        const ModularSymbolMeasure& me = sign < 0 ? out.mm : out.mp;
        const Vec& P = sign < 0 ? C.r_minus : C.r_plus;
        const std::string st = tag + (sign < 0 ? " r^-" : " r^+");
        uint32_t tt = 0, df = 0;
        for (uint32_t m = 0; m < sp.M; ++m) {
            tt += ThreeTermFailures(me, m);
            df += DistributionFailures(me, m);
        }
        log.check("M1", tt == 0, st + ": three-term failures " + std::to_string(tt));
        log.check("M1", df == 0, st + ": distribution failures " + std::to_string(df));
        auto& br = branches[sign];
        br.resize(ell - 1);
        uint32_t wrong = 0;
        for (uint32_t a = 0; a + 1 < ell; ++a) {
            br[a] = me.branch(a);
            const int par = (a % 2 == 0) ? 1 : -1;
            if (par != me.geometric_sign && std::any_of(br[a].begin(), br[a].end(), [](uint64_t x) { return x != 0; })) ++wrong;
        }
        log.check("M1", wrong == 0 && me.geometric_sign == (sign < 0 ? -1 : 1),
                  st + ": wrong-parity branches non-zero " + std::to_string(wrong) + ", geometric sign " + std::to_string(me.geometric_sign));
        const double tm0 = Now();
        const std::vector<std::vector<uint64_t>> mom = me.moments(chars);
        uint32_t bad = 0, tot = 0;
        for (size_t ii = 0; ii < chars.size(); ++ii) {
            const uint32_t i = chars[ii];
            Vec ev(n + 1, 0);
            if (i == 0) {
                for (uint32_t j = 0; j <= n; ++j) ev[j] = P[n - j];
            } else {
                ev = EvAllJ(rho, P, TeichmullerCharacter(i, Rg));
            }
            for (uint32_t j = 0; j <= n; ++j) {
                const uint64_t E = (i == 0) ? Rg.mul(Rg.sub(1 % q, Rg.mul(Rg.pow(ell, j), ia)), Rg.sub(1 % q, Rg.mul(Rg.pow(ell, n - j), ia))) : ia;
                uint64_t lhs = Rg.mul(rho.binom(n, j), mom[ii][j]);
                if (j % 2 == 1) lhs = Rg.neg(lhs);
                ++tot;
                if ((Rg.sub(lhs, Rg.mul(E, ev[j]))) % prec != 0) ++bad;
            }
        }
        const bool tot_ok = sp.all_chars || (tots && tot == tots->moments_per_sign);
        log.check("M1", bad == 0 && tot_ok, st + ": moment identity failures " + std::to_string(bad) + " / " + std::to_string(tot));
        std::cout << "MEAS_ID cell=" << tag << " sign=" << (sign < 0 ? "-" : "+") << " three_term=" << tt << " distribution=" << df
                  << " wrong_parity=" << wrong << " moments=" << bad << "/" << tot << " chars=" << chars.size() << " moment_secs=" << (Now() - tm0)
                  << std::endl;
        if (sign < 0) {
            // Riemann-sum moment == group-ring branch value at three j (character omega^1: branch a = j - 1)
            uint32_t cb = 0;
            const size_t i1 = static_cast<size_t>(std::find(chars.begin(), chars.end(), 1u) - chars.begin());
            for (uint32_t j : {0u, n / 2u + 1u, n}) {
                const uint32_t a = static_cast<uint32_t>((j + (ell - 1) - 1) % (ell - 1));
                const uint64_t bv = GrEval(br[a], j, ell, q);
                if ((bv + q - mom[i1][j]) % prec != 0) ++cb;
            }
            log.check("M1", cb == 0, st + ": moment(j,1) != branch_value(j-1, j) at " + std::to_string(cb) + " of 3 j");
        }
    }

    // ---- (M2) the frozen values (mod ell^4) ----
    if (fr) {
        const uint64_t q4 = PowU(ell, 4);
        const bool r4 = sp.R >= 4;
        if (r4) {
            log.check("M2", C.a_p.at(2) % q4 == fr->a2, tag + ": a_2 = " + std::to_string(C.a_p.at(2) % q4) + " (frozen " + std::to_string(fr->a2) + ")");
            log.check("M2", C.a_p.at(3) % q4 == fr->a3, tag + ": a_3 = " + std::to_string(C.a_p.at(3) % q4) + " (frozen " + std::to_string(fr->a3) + ")");
            log.check("M2", a_ell % q4 == fr->a_ell, tag + ": a_ell = " + std::to_string(a_ell % q4) + " (frozen " + std::to_string(fr->a_ell) + ")");
            log.check("M2", alpha % q4 == fr->alpha, tag + ": alpha = " + std::to_string(alpha % q4) + " (frozen " + std::to_string(fr->alpha) + ")");
            const uint64_t mum = out.mm.ball(2, 2) % q4, mup = out.mp.ball(2, 2) % q4;
            log.check("M2", mum == fr->mu_minus_2, tag + ": mu^-(2+ell^2) = " + std::to_string(mum));
            log.check("M2", mup == fr->mu_plus_2, tag + ": mu^+(2+ell^2) = " + std::to_string(mup));
            if (sp.R == 4) {
                log.check("M2", out.mm.fnv_units(2) == fr->fnv_minus, tag + ": fnv mu^- differs");
                log.check("M2", out.mp.fnv_units(2) == fr->fnv_plus, tag + ": fnv mu^+ differs");
            }
        }
        log.check("M2", out.mm.unit_total(2) == 0, tag + ": mu^-(Z_ell^x) != 0");
        std::ostringstream os;
        os << std::hex << "0x" << out.mm.fnv_units(2) << " 0x" << out.mp.fnv_units(2);
        std::cout << "MEAS_CELL frozen cell=" << tag << " mu-(2+l^2)=" << out.mm.ball(2, 2) << " mu+(2+l^2)=" << out.mp.ball(2, 2) << " fnv=" << os.str()
                  << std::endl;
    }

    // ---- (M5) the minus / plus invariants ----
    for (int sign : {-1, +1}) {
        const ModularSymbolMeasure& me = sign < 0 ? out.mm : out.mp;
        std::string listing;
        const std::string bad = CheckInvariants(me, sign < 0 ? 1u : 0u, FindInv(n, ell, sign), listing);
        log.check("M5", bad.empty(), tag + (sign < 0 ? " minus" : " plus") + " invariants:" + bad);
        std::cout << "MEAS_INV cell=" << tag << " sign=" << (sign < 0 ? "-" : "+") << " (mu,lambda)!=(0,0): " << (listing.empty() ? "none" : listing)
                  << std::endl;
    }

    // ---- (M7) the record lfunction/v1 (slot-20 stretch S1; [E] sec:lvlfunction) ----
    if (!sp.heavy) RunLFunctionArm(sp, out, log);

    // ---- (M6) negative controls (default arm) ----
    if (!sp.heavy) {
        const uint32_t Mn = std::min<uint32_t>(sp.M, 2);
        for (int sign : {-1, +1}) {
            const Vec& P = sign < 0 ? C.r_minus : C.r_plus;
            const ModularSymbolMeasure neg = BuildMeasure(rho, P, a_ell, alpha, Mn, false);
            const uint32_t f1 = DistributionFailures(neg, 1);
            log.check("M6", f1 > 0, tag + (sign < 0 ? " r^-" : " r^+") + ": NEG no ell^{k-2} distributes at m=1");
            std::cout << "MEAS_NEG cell=" << tag << " sign=" << (sign < 0 ? "-" : "+") << " no_ell_k-2: fails m=0,1 = " << DistributionFailures(neg, 0)
                      << "," << f1;
            if (sign < 0) {
                const ModularSymbolMeasure wr = BuildMeasure(rho, P, a_ell, Rg.mul(alpha, (1 + ell) % q), Mn, true);
                const uint32_t w1 = DistributionFailures(wr, 1);
                log.check("M6", w1 > 0, tag + " r^-: NEG wrong root alpha(1+ell) distributes at m=1");
                std::cout << "; wrong_root: fails m=1 = " << w1;
            }
            std::cout << std::endl;
        }
    }

    if (sp.control) {
        out.ok = true;
        std::cout << "MEAS_CELL done cell=" << tag << " secs=" << (Now() - t0) << std::endl;
        return out;
    }

    // ---- (M3) the Eisenstein side ----
    const uint32_t d = sp.d;
    {
        uint32_t badp = 0;
        std::ostringstream vs;
        for (uint64_t p : std::vector<uint64_t>{2, 3, 5, 7, ell}) {
            const uint64_t e = Rg.add(1 % q, Rg.pow(p % q, k - 1));
            const uint32_t v = Rg.val(Rg.sub(C.a_p.at(p), e));
            vs << p << ":" << v << " ";
            if (v != d) ++badp;
        }
        log.check("M3", badp == 0, tag + ": v(a_p - e_p) != d at " + std::to_string(badp) + " primes (" + vs.str() + ")");
        Vec p0(n + 1, 0);
        p0[0] = 1 % q;
        p0[n] = Rg.neg(1 % q);
        Vec diff(n + 1);
        for (uint32_t t = 0; t <= n; ++t) diff[t] = Rg.sub(C.r_plus[t], Rg.mul(C.r_plus[0], p0[t]));
        uint32_t dep = Rg.m;
        for (uint64_t x : diff) dep = std::min(dep, Rg.val(x));
        log.check("M3", dep == d, tag + ": (L1) r^+ == u p_0 to depth " + std::to_string(dep));
        uint32_t minv = Rg.m;
        const uint64_t MMl = PowU(ell, sp.M);
        for (uint64_t a = 1; a < MMl; ++a)
            if (a % ell) minv = std::min(minv, Rg.val(out.mp.mu[sp.M][a]));
        log.check("M3", minv == d, tag + ": min v(mu^+) on level-M unit balls = " + std::to_string(minv));
        // the reduced plus measure ell^{-d} mu^+
        const Ring Rw = MakeRing(ell, sp.R - dep);
        const SymRho rhow(n, Rw);
        Vec w(n + 1);
        const uint64_t pd = Rg.ell_pow(dep);
        for (uint32_t t = 0; t <= n; ++t) w[t] = (diff[t] / pd) % Rw.q;
        const ModularSymbolMeasure mw = BuildMeasure(rhow, w, a_ell % Rw.q, alpha % Rw.q, std::min<uint32_t>(sp.M, 2), true);
        std::string listing;
        const std::string badw = CheckInvariants(mw, 0, FindInv(n, ell, 0), listing);
        bool mu0 = false;
        for (uint32_t a = 0; a + 1 < ell; a += 2) mu0 = mu0 || mw.invariants(a).mu == 0;
        log.check("M3", mu0, tag + ": the reduced plus measure has mu > 0 on every even branch");
        log.check("M5", badw.empty(), tag + " reduced plus invariants:" + badw);
        std::cout << "MEAS_INV cell=" << tag << " sign=reduced+ (mu,lambda)!=(0,0): " << (listing.empty() ? "none" : listing) << std::endl;
        // P^Eis
        int32_t vmin = 0;
        const Vec PE = EisensteinSymbol(n, Rg, &vmin);
        log.check("M3", vmin == 0, tag + ": P^Eis min valuation " + std::to_string(vmin) + " (integral and primitive iff 0)");
        const uint32_t wres = WResidualValuation(rho, PE);
        log.check("M3", wres == d, tag + ": P^Eis in W(Z/ell^e) for e <= " + std::to_string(wres));
        uint32_t t0u = 0;
        while (t0u <= n && !Rg.unit(C.r_minus[t0u])) ++t0u;
        ENGINE_CHECK_MSG(t0u <= n && Rg.unit(PE[t0u]), "HeckeLValueMeasureValidation: P^Eis is not a unit at r^-'s first unit coordinate");
        const uint64_t s = Rg.mul(C.r_minus[t0u], Rg.inv(PE[t0u]));
        uint32_t dm = Rg.m;
        for (uint32_t t = 0; t <= n; ++t) dm = std::min(dm, Rg.val(Rg.sub(C.r_minus[t], Rg.mul(s, PE[t]))));
        log.check("M3", dm == d, tag + ": r^- == s P^Eis to depth " + std::to_string(dm));
        if (fr && fr->s >= 0) log.check("M2", s % PowU(ell, 4) == static_cast<uint64_t>(fr->s), tag + ": s = " + std::to_string(s % PowU(ell, 4)));
        out.s = s;
        std::cout << "MEAS_EIS cell=" << tag << " d=" << d << " v(a_p-e_p)=" << vs.str() << "L1_depth=" << dep << " min_v_mu+=" << minv
                  << " PEis_vmin=" << vmin << " PEis_W_residual=" << wres << " s=" << s << " r-_sPEis_depth=" << dm << std::endl;

        // ---- (M4b, d) the critical moments against twisted Bernoulli numbers ----
        const TwistedBernoulliTable TB(ell, sp.R, n + 2);
        const BernoulliOverKTable BT(ell, sp.R, n + 2);
        auto Fval = [&](uint32_t r, uint32_t i, uint32_t eexp, bool& skip) -> LadicReading {
            if (i % (ell - 1) != 0) {
                const LadicReading L = TB(r, i);
                if (L.zero) skip = true;
                return L;
            }
            LadicReading L = BT(r);
            const uint64_t eu = Rg.sub(1 % q, Rg.pow(ell, eexp));
            if (eu == 0) {
                L.zero = true;
                return L;
            }
            L.unit = Rg.mul(L.unit, eu);   // (1 - ell^e), e >= 1: a unit
            return L;
        };
        const uint64_t pdq = PowU(ell, d), pd1 = PowU(ell, d >= 1 ? d - 1 : 0);
        const uint32_t pole1 = static_cast<uint32_t>(ell - 2), pole2 = static_cast<uint32_t>((k - 1) % (ell - 1));
        uint32_t off_bad = 0, off_tot = 0, pole_bad = 0, pole_tot = 0;
        std::vector<FrozenPolePoint> located;
        const auto& brm = branches[-1];
        for (uint32_t a = 1; a + 1 < ell; a += 2) {
            const bool pole = (a == pole1 || a == pole2);
            for (uint32_t j = 0; j <= n; ++j) {
                const uint64_t lhs = GrEval(brm[a], j, ell, q);
                bool skip = false;
                const uint32_t i1 = static_cast<uint32_t>(((static_cast<int64_t>(a) - j) % static_cast<int64_t>(ell - 1) + (ell - 1)) % (ell - 1));
                const uint32_t i2 = static_cast<uint32_t>(((static_cast<int64_t>(j) - a) % static_cast<int64_t>(ell - 1) + (ell - 1)) % (ell - 1));
                const LadicReading F1 = Fval(j + 1, i1, j, skip);
                const LadicReading F2 = Fval(n - j + 1, i2, n - j, skip);
                if (skip) continue;
                uint64_t rhs = 0;
                if (!F1.zero && !F2.zero) {
                    const int32_t vv = F1.v + F2.v;
                    if (vv < 0) continue;
                    if (static_cast<uint32_t>(vv) < sp.R) rhs = Rg.neg(Rg.mul(Rg.mul(s, Rg.mul(F1.unit, F2.unit)), Rg.ell_pow(static_cast<uint32_t>(vv))));
                }
                const uint64_t df = Rg.sub(lhs, rhs);
                if (pole) {
                    ++pole_tot;
                    pole_bad += (df % pdq) != 0;
                    out.pole_unit_diff += (df % ell) != 0;
                    if (d >= 2 && df % pd1 != 0) located.push_back({a, j});
                } else {
                    ++off_tot;
                    off_bad += (df % pdq) != 0;
                }
            }
        }
        out.pole_points = pole_tot;
        const bool tot_ok = sp.heavy || (tots && off_tot == tots->off_pole && pole_tot == tots->pole);
        log.check("M4", off_bad == 0 && tot_ok, tag + ": (M4b) off-pole critical moments fail " + std::to_string(off_bad) + " / " + std::to_string(off_tot));
        log.check("M4", pole_bad == pole_tot && pole_tot > 0,
                  tag + ": (M4d) pole branches fail mod ell^d at " + std::to_string(pole_bad) + " / " + std::to_string(pole_tot));
        std::ostringstream lp;
        for (const auto& p : located) lp << "(" << p.a << "," << p.j << ")";
        if (d >= 2) {
            const auto& exp = FrozenPolePoints(n, ell);
            bool same = exp.size() == located.size();
            for (size_t i = 0; same && i < exp.size(); ++i) same = exp[i].a == located[i].a && exp[i].j == located[i].j;
            log.check("M4", same, tag + ": (M4d) located pole points mod ell^{d-1}: " + lp.str());
        }
        std::cout << "MEAS_EIS cell=" << tag << " M4b off_pole_fail=" << off_bad << "/" << off_tot << " pole_fail_mod_ell^d=" << pole_bad << "/"
                  << pole_tot << " pole_fail_mod_ell^{d-1}=" << located.size() << " at " << (lp.str().empty() ? "-" : lp.str()) << std::endl;

        // ---- (M4c) group-ring identity, (M6) wrong twist / wrong sign ----
        if (kl && PowU(ell, sp.M - 1) <= 1369 && kl->M == sp.M) {
            const uint64_t pg = PowU(ell, std::min(d, sp.M));
            for (int route = 0; route < 2; ++route) {
                const auto& Gm = route == 0 ? kl->bern : kl->layer;
                auto G = [&](int64_t b) -> const std::vector<uint64_t>& {
                    const uint32_t bb = static_cast<uint32_t>(((b % static_cast<int64_t>(ell - 1)) + (ell - 1)) % (ell - 1));
                    return Gm.at(bb);
                };
                uint32_t ok = 0, tot = 0, neg_ok = 0, neg_tot = 0, sgn_ok = 0;
                for (uint32_t a = 1; a + 1 < ell; a += 2) {
                    if (a == pole1 || a == pole2) continue;
                    const auto& g = brm[a];
                    const std::vector<uint64_t> e = GrMul(GrTw1(G(a + 1), ell, q), GrTwInv(G(static_cast<int64_t>(k) - 1 - a), ell, q, k), q);
                    ++tot;
                    bool m1 = true, m2 = true;
                    for (size_t i = 0; i < g.size(); ++i) {
                        const uint64_t se = Rg.mul(s, e[i]);
                        m1 = m1 && (Rg.add(g[i], se) % pg == 0);    // g == -s e
                        m2 = m2 && (Rg.sub(g[i], se) % pg == 0);    // NEG: g == +s e
                    }
                    ok += m1;
                    sgn_ok += m2;
                    if (route == 0 && (a + 3) % (ell - 1) != 0) {
                        const std::vector<uint64_t> e2 = GrMul(GrTw1(G(a + 3), ell, q), GrTwInv(G(static_cast<int64_t>(k) - 1 - a), ell, q, k), q);
                        bool m3 = true;
                        for (size_t i = 0; i < g.size() && m3; ++i) m3 = Rg.add(g[i], Rg.mul(s, e2[i])) % pg == 0;
                        ++neg_tot;
                        neg_ok += m3;
                    }
                }
                const bool tot_ok2 = sp.heavy || (tots && tot == tots->branches_off_pole);
                log.check("M4", ok == tot && tot_ok2,
                          tag + ": (M4c) " + (route == 0 ? "E_{1,c}" : "CwLayer") + " group ring matches " + std::to_string(ok) + " / " + std::to_string(tot));
                if (route == 0 && !sp.heavy) {
                    log.check("M6", neg_ok == 0 && tots && neg_tot == tots->branches_wrong_twist,
                              tag + ": NEG wrong twist G_{a+3} matches " + std::to_string(neg_ok) + " / " + std::to_string(neg_tot));
                    log.check("M6", sgn_ok == 0, tag + ": NEG wrong sign +s matches " + std::to_string(sgn_ok) + " / " + std::to_string(tot));
                }
                std::cout << "MEAS_KL cell=" << tag << " route=" << (route == 0 ? "E1c" : "CwLayer") << " M4c=" << ok << "/" << tot;
                if (route == 0) std::cout << " NEG_wrong_twist=" << neg_ok << "/" << neg_tot << " NEG_wrong_sign=" << sgn_ok << "/" << tot;
                std::cout << std::endl;
            }
        }
    }
    out.ok = true;
    std::cout << "MEAS_CELL done cell=" << tag << " secs=" << (Now() - t0) << std::endl;
    return out;
}

// ---- (M0) the route on the C++ module cone ------------------------------------------------------------------------
bool SameMeasure(const ModularSymbolMeasure& a, const ModularSymbolMeasure& b) { return a.mu == b.mu && a.q == b.q; }

template<class Fn>
std::string ThrowsWhat(Fn f, int& kind) {   // kind: 0 none, 1 logic_error, 2 invalid_argument, 3 NotImplementedInThisBuild, 4 other
    kind = 0;
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild& e) {
        kind = 3;
        return e.what();
    } catch (const std::invalid_argument& e) {
        kind = 2;
        return e.what();
    } catch (const std::logic_error& e) {
        kind = 1;
        return e.what();
    } catch (const std::exception& e) {
        kind = 4;
        return e.what();
    }
    return "";
}

void RunRouteArm(const CellOut& c30, Log& log) {
    const double t0 = Now();
    const SymbolCell& C = c30.C;
    const uint32_t n = C.n;
    BirchCellOptions opt;
    opt.coboundary_check = false;
    const auto cell = std::make_shared<const BirchConeCell>(RunBirchConeCell(n, C.ell, C.R, opt));
    const double t1 = Now();
    const Ring& R = cell->R;
    ENGINE_CHECK_MSG(R.q == C.ring.q, "HeckeLValueMeasureValidation: cone ring differs from the symbol ring");
    const BirchModularSymbolRoute route(cell, TrivialCharacter(R), 0);
    const SmithForm F = Smith(R, ColumnsToMat(cell->P, n + 1), true);
    auto solve = [&](const Vec& P, CohomologyClass& x) -> bool {
        Vec y;
        if (!Solve(R, F, P, y)) return false;
        x.exponents = cell->exps;
        x.coordinates = y;
        return route.symbol(x) == P;
    };
    MeasureRequest req;
    req.level = c30.mm.M;
    CohomologyClass xm, xp;
    const bool sm = solve(C.r_minus, xm), spl = solve(C.r_plus, xp);
    const FrozenMeasureCell* fr = FindFrozen(n, C.ell);
    bool eqm = false, eqp = false;
    ModularSymbolMeasure rm, rp;
    if (sm) {
        rm = route.measure(xm, req);
        eqm = SameMeasure(rm, c30.mm) && fr && rm.fnv_units(2) == fr->fnv_minus;
    }
    if (spl) {
        rp = route.measure(xp, req);
        eqp = SameMeasure(rp, c30.mp) && fr && rp.fnv_units(2) == fr->fnv_plus;
    }
    log.check("M0", sm && eqm, "(30,37) cone class x^- with Psi(x^-) = r^-: measure() == BuildMeasure(r^-), frozen fnv");
    log.check("M0", spl && eqp, "(30,37) cone class x^+ with Psi(x^+) = r^+: measure() == BuildMeasure(r^+), frozen fnv");
    // linearity
    bool lin_sum = false, lin_scal = false;
    if (sm && spl) {
        CohomologyClass xs = xm;
        for (size_t i = 0; i < xs.coordinates.size(); ++i) xs.coordinates[i] = R.add(xm.coordinates[i], xp.coordinates[i]);
        const ModularSymbolMeasure ms = route.measure(xs, req);
        lin_sum = true;
        for (uint32_t m = 0; m <= ms.M && lin_sum; ++m)
            for (size_t a = 0; a < ms.mu[m].size() && lin_sum; ++a) lin_sum = ms.mu[m][a] == R.add(rm.mu[m][a], rp.mu[m][a]);
        CohomologyClass x5 = xm;
        for (auto& v : x5.coordinates) v = R.mul(v, 5);
        const ModularSymbolMeasure m5 = route.measure(x5, req);
        lin_scal = true;
        for (uint32_t m = 0; m <= m5.M && lin_scal; ++m)
            for (size_t a = 0; a < m5.mu[m].size() && lin_scal; ++a) lin_scal = m5.mu[m][a] == R.mul(5, rm.mu[m][a]);
    }
    log.check("M0", lin_sum, "(30,37) linearity: mu_{x^- + x^+} == mu_{x^-} + mu_{x^+}");
    log.check("M0", lin_scal, "(30,37) linearity: mu_{5 x^-} == 5 mu_{x^-}");
    // the other kinds refuse
    int kind = 0;
    {
        Domain dh;
        dh.model = ModelId::SL2ZSevenVertex;
        dh.cohomology = "H^1_c(M_n) x H^1(D_n)";
        dh.n_sym = n;
        dh.level = 1;
        dh.ell = C.ell;
        dh.m = C.R;
        const HidaPairingRoute hida(dh);
        const std::string w = ThrowsWhat([&]() { (void)hida.measure(xm, req); }, kind);
        log.check("M0", kind == 1, "HidaPairingRoute::measure must refuse (std::logic_error): " + w);
    }
    {
        auto ec = std::make_shared<EisensteinCell>();
        ec->n = n;
        ec->ell = C.ell;
        ec->K = 2;
        ec->level = 1;
        const EisensteinBernoulliRoute eb(ec, 1);
        const std::string w = ThrowsWhat([&]() { (void)eb.measure(xm, req); }, kind);
        log.check("M0", kind == 1, "EisensteinBernoulliRoute::measure must refuse (std::logic_error): " + w);
    }
    // (10, 5): Delta is not ordinary at 5
    {
        const auto cell5 = std::make_shared<const BirchConeCell>(RunBirchConeCell(10, 5, 2, opt));
        const Ring& R5 = cell5->R;
        const BirchModularSymbolRoute route5(cell5, TrivialCharacter(R5), 0);
        std::vector<Vec> od;
        for (const Vec& P : cell5->P) od.push_back(SignPart(R5, P, -1));
        const PivotBasis Bo = SaturatedBasis(R5, od);
        std::string w = "W^- of (10,5) is not of rank 1";
        kind = 0;
        if (Bo.basis.size() == 1) {
            const SmithForm F5 = Smith(R5, ColumnsToMat(cell5->P, 11), true);
            Vec y;
            if (Solve(R5, F5, Bo.basis[0], y)) {
                CohomologyClass x5;
                x5.exponents = cell5->exps;
                x5.coordinates = y;
                w = ThrowsWhat([&]() { (void)route5.measure(x5, MeasureRequest()); }, kind);
            }
        }
        log.check("M0", kind == 2 && w.find("not ordinary") != std::string::npos, "(10,5) must refuse \"not ordinary\": " + w);
    }
    // a class off the eigenline
    {
        std::string w = "no W^- vector off the eigenline";
        kind = 0;
        for (const Vec& u : C.Wminus) {
            Vec P(n + 1);
            for (uint32_t t = 0; t <= n; ++t) P[t] = R.add(R.add(C.r_minus[t], C.r_plus[t]), u[t]);
            CohomologyClass x;
            if (!solve(P, x)) continue;
            w = ThrowsWhat([&]() { (void)route.measure(x, req); }, kind);
            if (kind != 0) break;
        }
        log.check("M0", kind == 2 && w.find("not a T_ell eigenclass") != std::string::npos, "non-eigenclass must refuse: " + w);
    }
    std::cout << "MEAS_ROUTE cell=(30,37) R=Z/37^" << C.R << " cone_route=\"" << cell->route << "\" exps=" << cell->exps.size()
              << " solve-=" << sm << " solve+=" << spl << " equal-=" << eqm << " equal+=" << eqp << " lin=" << lin_sum << lin_scal
              << " cone_secs=" << (t1 - t0) << " secs=" << (Now() - t0) << std::endl;
}

// ---- (S2) the 691 arm: the MTT measure of Delta at (10, 691) against Kubota-Leopoldt (env HECKE_LVALUE_MEASURE_691=1) ----
// Predictions docs/implementation/slot20/s1s2/predictions.md (Q-691-1..11, committed first); oracle values MeasureFrozenS2.inc.
// RunCell does (M1), (M3), (M4b-d), (M5), (M6), (M7) on the cell exactly as at the other Eisenstein cells (d = 1); this adds
// Ramanujan on the symbol side (a_p == tau(p) mod 691^2 from the q-expansion), the oracle's values, and the sharper pole reading.
void Run691Arm(Log& log, const std::map<uint64_t, uint64_t>& sep) {
    const double t0 = Now();
    const KlRoutes kl = RunKl(691, 2, 2, log, true);
    const double t1 = Now();
    const CellSpec sp{10, 691, 1, 2, 2, sep, false, false, false};
    const CellOut o = RunCell(sp, log, &kl);
    if (!o.ok) {
        log.check("S2", false, "(10,691): the cell did not complete");
        return;
    }
    const uint64_t q = o.C.ring.q;
    std::ostringstream aps;
    uint32_t bad = 0;
    for (const FrozenTau691& f : kFrozenTau691) {
        const uint64_t ap = o.C.a_p.at(f.p) % q;
        aps << f.p << ":" << ap << " ";
        bad += ap != f.tau_mod;
    }
    log.check("S2", bad == 0 && o.C.eisenstein_congruent && o.C.simple && o.C.Wminus.size() == 1,
              "(10,691) Q-691-1: a_p == tau(p) mod 691^2 fails at " + std::to_string(bad) + " primes (" + aps.str() + "), rank W^- " +
                  std::to_string(o.C.Wminus.size()));
    log.check("S2", o.alpha == kFrozenTau691[4].tau_mod && o.C.a_p.at(691) % 691 == 1,
              "(10,691) Q-691-1: alpha == tau(691) mod 691^2 and tau(691) == 1 mod 691 (ordinary): alpha = " + std::to_string(o.alpha));
    log.check("S2", o.s == kFrozenOracle691.s, "(10,691) oracle: s = " + std::to_string(o.s) + " (oracle " + std::to_string(kFrozenOracle691.s) + ")");
    log.check("S2", o.mm.ball(2, 2) == kFrozenOracle691.mu_minus_2 && o.mm.fnv_units(2) == kFrozenOracle691.fnv_minus,
              "(10,691) oracle: mu^-(2+ell^2) / fnv mu^- differ");
    log.check("S2", o.mp.ball(2, 2) == kFrozenOracle691.mu_plus_2 && o.mp.fnv_units(2) == kFrozenOracle691.fnv_plus,
              "(10,691) oracle: mu^+(2+ell^2) / fnv mu^+ differ");
    log.check("S2", o.mm.unit_total(2) == 0, "(10,691): mu^-(Z_ell^x) != 0");
    log.check("S2", o.pole_points == 22 && o.pole_unit_diff == 22,
              "(10,691) Q-691-7: v(lhs - rhs) = 0 on the pole branches at " + std::to_string(o.pole_unit_diff) + " / " + std::to_string(o.pole_points));
    std::ostringstream h;
    h << std::hex << "0x" << o.mm.fnv_units(2) << " 0x" << o.mp.fnv_units(2);
    std::cout << "MEAS_691 cell=(10,691) R=2 M=2 a_p=" << aps.str() << "alpha=" << o.alpha << " s=" << o.s << " fnv=" << h.str()
              << " pole_points=" << o.pole_points << " pole_v0=" << o.pole_unit_diff << " kl_secs=" << (t1 - t0) << " cell_secs=" << (Now() - t1)
              << " peak_rss_gb=" << PeakRssGb() << std::endl;
}

} // namespace
} // namespace lvalue
} // namespace hecke_ext

uint32_t HeckeLValueMeasureValidation() {
    using namespace hecke_ext::lvalue;
    const double T0 = Now();
    Log log;
    const char* hv = std::getenv("HECKE_LVALUE_MEASURE_HEAVY");
    const bool heavy = hv && std::string(hv) == "1";
    std::cout << "EFFECTIVE HeckeLValueMeasureValidation arm=" << (heavy ? "HEAVY cells=3" : "DEFAULT cells=5") << std::endl;
    // (slot-20 S2) the 691 arm, read-only, env-armed (its own EFFECTIVE line; independent of HECKE_LVALUE_MEASURE_HEAVY)
    const char* e691 = std::getenv("HECKE_LVALUE_MEASURE_691");
    const bool arm691 = e691 && std::string(e691) == "1";
    if (arm691) std::cout << "EFFECTIVE HeckeLValueMeasureValidation arm=ELL691 cells=1" << std::endl;
    try {
        const std::map<uint64_t, uint64_t> sep235 = {{2, 1}, {3, 3}, {5, 7}}, sep3 = {{3, 1}};
        // (M4a) the two Kubota-Leopoldt routes
        const KlRoutes kl37 = RunKl(37, 3, 4, log, true);
        const KlRoutes kl103 = RunKl(103, 2, 4, log, true);
        RunBernoulliCrossChecks(log);
        // the five cells
        std::vector<CellSpec> cells = {
            {10, 37, 0, 4, 2, sep235, true, false, false},  {16, 37, 0, 4, 3, sep235, true, false, false},
            {30, 37, 1, 4, 3, sep235, false, false, false}, {282, 37, 2, 4, 3, sep3, false, false, false},
            {226, 103, 2, 4, 2, sep3, false, false, false},
        };
        CellOut c30;
        for (const CellSpec& sp : cells) {
            const KlRoutes* kl = sp.ell == 37 ? &kl37 : &kl103;
            CellOut o = RunCell(sp, log, kl);
            if (sp.n == 30 && sp.ell == 37) c30 = std::move(o);
        }
        // (M0) the route
        if (c30.ok) {
            RunRouteArm(c30, log);
        } else {
            log.check("M0", false, "(30,37) symbol cell missing: the route arm did not run");
        }
        // (M6) the separating operator T_2 + 3T_3 + 7T_5 collides at the depth-2 cells
        for (const auto& nl : std::vector<std::pair<uint32_t, uint64_t>>{{282, 37}, {226, 103}}) {
            const SymbolCell C = RunSymbolCell(nl.first, nl.second, 4, sep235, false);
            log.check("M6", !C.simple && C.target_multiplicity == 2,
                      "(" + std::to_string(nl.first) + "," + std::to_string(nl.second) + "): T_2+3T_3+7T_5 must be refused with multiplicity 2 (got " +
                          std::to_string(C.target_multiplicity) + ")");
            std::cout << "MEAS_NEG cell=(" << nl.first << "," << nl.second << ") separator=T_2+3T_3+7T_5 " << (C.simple ? "ACCEPTED" : C.refusal) << std::endl;
        }
        if (heavy) {
            const KlRoutes kl37r5 = RunKl(37, 3, 5, log, false);
            KlRoutes kl37h = kl37r5;
            kl37h.layer = kl37.layer;
            std::vector<CellSpec> hc = {
                {226, 103, 2, 4, 3, sep3, false, false, true},
                {282, 37, 2, 5, 3, sep3, false, true, true},
                {30, 37, 1, 4, 3, sep235, false, true, true},
            };
            for (const CellSpec& sp : hc) RunCell(sp, log, sp.ell == 37 ? (sp.R == 5 ? &kl37h : &kl37) : nullptr);
        }
        if (arm691) Run691Arm(log, sep235);
    } catch (const std::exception& e) {
        log.check("EXC", false, std::string("exception: ") + e.what());
    }
    std::ostringstream arms;
    for (const auto& kv : log.per_arm) arms << kv.first << "=" << kv.second << " ";
    std::cout << "MEAS_ARMS checks=" << log.checks << " failing=" << log.failing << " per_arm: " << arms.str() << "secs=" << (Now() - T0)
              << " peak_rss_gb=" << PeakRssGb() << std::endl;
    return log.failing == 0 ? 0u : 1u;
}
