// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeLambdaTwoVarValidation (standard) -- the gate of P9 (hecke-extensions slot 21): the two-variable
// measure by interpolation.  Spec: docs/implementation/slot21/P9_spec.md v0.2 (gates V1-V12, the negative
// controls of its Sec. 4).  Inputs: P8's normalised unit-ball masses (HeckeLambdaModSym, [U] v0.5.0-pre.2
// part 4) at N = 5 consecutive weights + F = 3 further weights of each class, K = 3, K' = 5, m = 1, 2, 3.
//
// What it builds: per cell, component, [C]-sign, the branch elements g^{(a)}_i(X) of the normalised
// family of measures at ball level M (3 at ell = 11, 13; 2 at 23, 37), their truncations
// mod (ell^K, (X - x_c)^m) and the Taylor-shifted coefficients of L_a(X, T) mod omega_{M-1}(T)
// (record hecke_lambda_twovar/v1), and predictions at further points of the disc: the further weights,
// the Eisenstein point alpha_a at (37, 32), and the weight-2 point x_2 at (11, 2) (11a).
//
// ORACLES (never this code's own output): P1's lfunction/v1 pipeline (RunSymbolCell's own eigensymbol,
// BuildMeasure, MakeLFunctionRecord) for V3 / V4; WP-MEAS's Kubota-Leopoldt group ring (the regularised
// Bernoulli measure) for V5 / V6; the exact Bernoulli numbers and [U]'s alpha_32 mod 37^4 (the Coates-Wiles
// layer, R19-7) for V7; 11a's Manin symbols on Gamma_0(11) (HeckeLambdaTwoVar, no level-one routine) for
// V8 / V9; Tate's period of 11a1 from j (V12) and the Python oracle's value committed with the spec.
//
// Gate kinds: PROVED gates count as [FAIL] (a defect); CONJ / LIT gates as [FINDING] (counted, printed,
// never a failure): V2, V4, V5 off k_0 and L_k = 1, V6, V9, V10, V11, V12.
//
// Env (read-only; prints EFFECTIVE HeckeLambdaTwoVarValidation arm=<X> cells=<k>):
//   HECKE_LAMBDA_TWOVAR_ALL=1     adds (103, 24) (Eisenstein arm at M_E = 2) and M = 3 at 37;
//   HECKE_LAMBDA_TWOVAR_CELLS="ell:a:k0,..."  replaces the cell list (a malformed token is refused by
//                                 position; a list selecting no cell fails).
// Writes nothing, no golden object, no digest of record.
#include "HeckeLambdaTwoVarValidation.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "HeckeHidaLambda.h"
#include "HeckeLambdaModSym.h"
#include "HeckeLambdaQExp.h"
#include "HeckeLambdaTwoVar.h"
#include "LValue/LFunctionRecord.h"
#include "LValue/ModularSymbolMeasure.h"
#include "../Util/ValidationCheck.h"
#include "../Util/CellList.h"

namespace {

namespace L = hecke_ext::lambda;
namespace MS = hecke_ext::lambda::msym;
namespace TV = hecke_ext::lambda::twovar;
namespace bm = hecke_ext::lvalue::birch;
namespace lv = hecke_ext::lvalue;
using MS::SMat;
using MS::SVec;
using L::u128;
using GR = std::vector<uint64_t>;   // a group-ring element / branch element

oc_util::Checker g_check("  [FAIL] "), g_finding("  [FINDING] ");
std::map<std::string, uint32_t> g_by_gate;

void Check(bool ok, const std::string& what) {
    ++g_by_gate[what.substr(0, what.find(' '))];
    g_check(ok, what);
}
void Finding(bool ok, const std::string& what) {
    ++g_by_gate[what.substr(0, what.find(' '))];
    g_finding(ok, what);
}

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
long PeakRssMiB() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024;
}
bool IsPrime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}
uint64_t PowU(uint64_t b, uint32_t e) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < e; ++i) r *= b;
    return r;
}
uint32_t ValU(uint64_t x, uint64_t ell, uint32_t cap) {
    if (x == 0) return cap;
    uint32_t v = 0;
    while (x % ell == 0 && v < cap) {
        x /= ell;
        ++v;
    }
    return v;
}
std::string CellTag(uint64_t ell, uint32_t a) { return "(" + std::to_string(ell) + "," + std::to_string(a) + ")"; }

constexpr uint32_t kK = 3, kKp = 5, kB = 30, kN = 5;

struct CellSpec {
    uint64_t ell;
    uint32_t a, k0;
    uint32_t M;    // ball level of the two-variable measure
    uint32_t ME;   // > 0: the Eisenstein arm with the KL side at level ME
};

// ---- one (component, sign) of a cell ------------------------------------------------------------------------
struct Side {
    size_t ci = 0;
    uint32_t r = 0;
    bool eis = false, p0side = false;
    int sign = 0;
    std::string tag;
    std::vector<MS::Reading> rd;
    int b0 = -1;
    MS::Ball B0{0, 0};
    std::vector<std::vector<SVec>> nu;                 // per weight, per ball: r coordinates
    std::vector<std::vector<std::vector<GR>>> br;      // per weight, per branch a, per Y-power: g_i, i < ell^{M-1}
    std::vector<uint64_t> full;                        // r = 1, not the p_0 sign: nu_{Z_ell}(x_k)
    bool usable = false;
};

// ---- P1's pipeline at one weight (the oracle of V3 / V4) ------------------------------------------------------
struct P1Oracle {
    bool ok = false;
    std::string why, sep;
    lv::LFunctionRecord rec;
    uint64_t muB0 = 0;
};
P1Oracle RunP1(uint64_t ell, uint32_t k, uint64_t p0, const MS::Reading& rd, int sign, uint32_t M, const MS::Ball& B0,
               const std::vector<uint64_t>& primes, bool other_sign_control, bool eis_elsewhere) {
    P1Oracle o;
    if (eis_elsewhere) {
        // RunSymbolCell takes the Eisenstein root sum c_p (1 + p^{k-1}) whenever it is a root of charpoly(sep | W^-): at a class
        // with a cusp form through m_Eis that is always so, whatever the separator -- P1's pipeline cannot select another component
        o.why = "P1 refused: RunSymbolCell prefers the Eisenstein root at a class through m_Eis";
        return o;
    }
    const uint32_t n = k - 2;
    const uint64_t ap0 = rd.P.at(p0)[0] % ell;
    std::vector<uint64_t> qs;
    for (uint64_t q : primes)
        if (q != p0 && q != ell && rd.P.count(q) && rd.P.at(q)[0] % ell != 0) qs.push_back(q);
    if (ap0 == 0) qs.insert(qs.begin(), 0);   // T_{p0} alone has the root 0
    for (uint64_t q : qs) {
        std::map<uint64_t, uint64_t> sep{{p0, 1}};
        if (q) {
            const bm::Ring R1 = bm::MakeRing(ell, 1);
            sep[q] = R1.mul((ell - ap0) % ell, R1.inv(rd.P.at(q)[0] % ell));
        }
        bm::SymbolCell C;
        try {
            C = bm::RunSymbolCell(n, ell, kKp, sep, true);
        } catch (const std::logic_error& ex) {
            // P1's own refusal (e.g. W(Z/ell^R) not free where H^2_c has ell-torsion): a limit of the oracle, not of P9
            o.why = std::string("P1 refused: ") + ex.what();
            return o;
        }
        if (!C.refusal.empty() || !C.simple || C.lam % ell != 0) continue;
        bool same = true;
        for (const auto& kv : C.a_p) {
            if (kv.first == ell) {
                same = same && kv.second % C.ring.q == rd.P.at(ell)[0] % C.ring.q;
            } else if (rd.P.count(kv.first)) {
                same = same && kv.second % C.ring.q == rd.P.at(kv.first)[0] % C.ring.q;
            }
        }
        if (!same) continue;
        const bool want_plus = (sign < 0) != other_sign_control;   // [C]-minus = geometric even = P1 "plus"
        if (want_plus && !C.have_r_plus) {
            o.why = "no r^+";
            return o;
        }
        const bm::Vec& sym = want_plus ? C.r_plus : C.r_minus;
        const uint64_t a_ell = C.a_p.at(ell);
        const uint64_t alpha = bm::UnitRoot(C.ring, a_ell, k);
        const bm::SymRho rho(n, C.ring);
        const lv::ModularSymbolMeasure me = bm::BuildMeasure(rho, sym, a_ell, alpha, M, true);
        lv::LFunctionForm form;
        form.sign = want_plus ? "plus" : "minus";
        o.rec = lv::MakeLFunctionRecord(me, form, {});
        o.muB0 = me.ball(B0.b, B0.M);
        std::ostringstream ss;
        ss << "T" << p0;
        if (q) ss << "+" << sep[q] << "T" << q;
        o.sep = ss.str();
        o.ok = true;
        return o;
    }
    o.why = "no separator selects the component";
    return o;
}

// ---- the record hecke_lambda_twovar/v1 -----------------------------------------------------------------------
std::string RecordLine(uint64_t ell, uint32_t a, uint32_t m, uint32_t k0, uint32_t M, const Side& s, uint64_t p0, uint32_t branch,
                       const std::vector<std::vector<L::Vec>>& ct /* t -> Y-power -> Z-coeffs */, const std::vector<std::string>& further) {
    std::ostringstream head, body, full;
    head << "HECKE_LAMBDA_TWOVAR ell=" << ell << " a=" << a << " K=" << kK << " m=" << m << " centre_k=" << k0 << " N=" << kN << " M=" << M
         << " component=" << s.ci << " sign=" << (s.sign > 0 ? "+" : "-") << " rank=" << s.r << " p0=" << p0 << " B0=" << s.B0.b << "+ell^"
         << s.B0.M << " branch=" << branch;
    const std::vector<uint32_t> tp = lv::CertifiedTPrecision(ell, kK, M);
    auto tline = [&](size_t lim) {
        std::ostringstream os;
        os << " tprec=[";
        for (size_t t = 0; t < tp.size() && t < lim; ++t) os << (t ? "," : "") << tp[t];
        if (tp.size() > lim) os << ",..";
        os << "] L=[";
        for (size_t t = 0; t < ct.size() && t < lim; ++t) {
            os << (t ? ";" : "") << t << ":[";
            for (size_t y = 0; y < ct[t].size(); ++y) {
                os << (y ? "," : "") << "[";
                for (size_t z = 0; z < ct[t][y].size(); ++z) os << (z ? "," : "") << L::U128Text(ct[t][y][z]);
                os << "]";
            }
            os << "]";
        }
        if (ct.size() > lim) os << ";..";
        os << "] further=[";
        for (size_t i = 0; i < further.size(); ++i) os << (i ? " " : "") << further[i];
        os << "]";
        return os.str();
    };
    const std::string all = head.str() + tline(ct.size() + 1);
    const uint64_t h = engine::FnvBytes(engine::kFnvOffset64, all);
    std::ostringstream out;
    out << head.str() << tline(4) << " digest=0x" << std::hex << h << std::dec;
    return out.str();
}

// ---- one cell ------------------------------------------------------------------------------------------------
void RunCell(const CellSpec& cs, uint32_t Fw) {
    const double tc = Now();
    const uint64_t l = cs.ell;
    const uint32_t M = cs.M;
    const uint64_t G = PowU(l, M - 1);
    std::vector<uint64_t> primes;
    for (uint64_t p = 2; p <= kB; ++p)
        if (IsPrime(p) && p != l) primes.push_back(p);
    std::vector<uint32_t> ks, interp, further;
    for (uint32_t j = 0; j < kN + Fw; ++j) {
        const uint32_t k = cs.k0 + j * static_cast<uint32_t>(l - 1);
        ks.push_back(k);
        (j < kN ? interp : further).push_back(k);
    }
    const std::string ctag = CellTag(l, cs.a);
    const bm::Ring Rk = bm::MakeRing(l, kKp);
    const uint32_t prec_pred = std::min(kKp, kN);
    const uint64_t qpred = PowU(l, prec_pred);
    const uint32_t eyw = kKp + kN - 1;   // y = x_k mod ell^{K'+N-1}: exact weights

    // ---- inputs: P7 (p0, components), P8 (readings)
    std::vector<L::FormsWeight> ws;
    for (uint32_t k : ks) ws.push_back(L::ComputeFormsWeight(l, k, kKp, primes));
    const std::vector<L::FormsWeight> wsI(ws.begin(), ws.begin() + kN);
    const bool with_eis = cs.a % (l - 1) != 0;
    const L::LambdaQExpansion LQ = L::BuildLambdaQExpansion(wsI, cs.a, kK, 3, cs.k0, kB, 0, with_eis);
    const uint64_t p0 = LQ.p0;
    std::vector<size_t> compIdx;
    for (size_t i = 0; i < LQ.comps.size(); ++i)
        if (LQ.comps[i].kind != "eisenstein") compIdx.push_back(i);
    bool eis_elsewhere = false;
    for (size_t i : compIdx) eis_elsewhere = eis_elsewhere || LQ.comps[i].kind == "cusp_eis";
    std::cout << "CELL ell=" << l << " a=" << cs.a << " weights=" << ks.front() << ".." << ks.back() << " (N=" << kN << ", F=" << Fw << ") M=" << M
              << " p0=" << p0 << " components=" << compIdx.size() << std::endl;

    uint32_t maxprec = 0;
    for (unsigned __int128 x = l; x < (static_cast<unsigned __int128>(1) << 62); x *= l) ++maxprec;
    const uint32_t D = std::min<uint32_t>(3, maxprec - kKp - 2);
    std::map<std::tuple<uint32_t, int, uint32_t>, std::shared_ptr<MS::SignSpace>> spaces;
    auto space = [&](uint32_t k, int sign, uint32_t R) {
        const auto key = std::make_tuple(k, sign, R);
        auto it = spaces.find(key);
        if (it == spaces.end()) it = spaces.emplace(key, std::make_shared<MS::SignSpace>(MS::BuildSignSpace(l, k - 2, sign, R, 2))).first;
        return it->second;
    };

    std::vector<Side> sides;
    for (size_t ci : compIdx) {
        const L::LambdaComponent& C3 = LQ.comps[ci];
        for (int sign : {-1, 1}) {
            Side s;
            s.ci = ci;
            s.r = C3.rank;
            s.eis = C3.kind == "cusp_eis";
            s.sign = sign;
            s.p0side = s.eis && sign < 0;
            s.tag = ctag + " comp=" + std::to_string(ci) + "(" + C3.kind + ",r=" + std::to_string(s.r) + ") sign=" + (sign > 0 ? "+" : "-");
            for (uint32_t k : ks) {
                MS::ReadingRequest rq;
                rq.ell = l;
                rq.k = k;
                rq.Kp = kKp;
                rq.c_sign = sign;
                rq.p0 = p0;
                rq.g = C3.residual;
                rq.eis = s.eis;
                rq.D = D;
                rq.Emax = 2;
                rq.M = M;
                rq.primes = primes;
                rq.space = space(k, sign, kKp + (s.p0side ? D : 0));
                s.rd.push_back(MS::ReadComponent(rq));
            }
            bool refused = false;
            for (const auto& rd : s.rd) refused = refused || !rd.refusal.empty();
            if (refused) {
                std::cout << "  SIDE " << s.tag << ": a weight refused by P8's reading (" << s.rd[0].refusal << ") -- skipped" << std::endl;
                sides.push_back(s);
                continue;
            }
            s.b0 = MS::FirstUnitBall(s.rd[0]);
            if (s.b0 < 0) {
                std::cout << "  SIDE " << s.tag << ": no unit ball at the centre -- skipped" << std::endl;
                sides.push_back(s);
                continue;
            }
            s.B0 = s.rd[0].balls[static_cast<size_t>(s.b0)];
            bool allunit = true;
            s.nu.resize(ks.size());
            for (size_t wi = 0; wi < ks.size(); ++wi) allunit = MS::Normalised(s.rd[wi], s.b0, s.nu[wi]) && allunit;
            if (!allunit) {
                Finding(false, "V2 " + s.tag + ": B_0 not a unit at some weight (P8's G8) -- side skipped");
                sides.push_back(s);
                continue;
            }
            s.usable = true;
            sides.push_back(s);
        }
    }

    // ball index of the level-M unit balls
    std::map<uint64_t, size_t> bidx;
    for (const Side& s : sides)
        if (s.usable) {
            for (size_t i = 0; i < s.rd[0].balls.size(); ++i)
                if (s.rd[0].balls[i].M == M) bidx[s.rd[0].balls[i].b] = i;
            break;
        }

    for (Side& s : sides) {
        if (!s.usable) continue;
        const double ts = Now();
        const int eps_geo = s.sign > 0 ? -1 : 1;
        // ---- branch elements per weight (own assembly)
        s.br.assign(ks.size(), std::vector<std::vector<GR>>(l - 1, std::vector<GR>(s.r)));
        for (size_t wi = 0; wi < ks.size(); ++wi)
            for (uint32_t a = 0; a + 1 < l; ++a)
                for (uint32_t y = 0; y < s.r; ++y)
                    s.br[wi][a][y] = TV::BranchFromBalls([&](uint64_t b) { return s.nu[wi][bidx.at(b)][y]; }, l, M, a, kKp);
        // full mass nu_{Z_ell}(x_k) (r = 1, not the p_0 sign)
        if (s.r == 1 && !s.p0side)
            for (size_t wi = 0; wi < ks.size(); ++wi) {
                const MS::Reading& rd = s.rd[wi];
                const MS::OperatorMeasure om(Rk, *rd.tables, MS::Reduce(rd.block->U, Rk.q));
                const uint64_t inv0 = Rk.inv(rd.muh[static_cast<size_t>(s.b0)][0] % Rk.q);
                s.full.push_back(Rk.mul(om.ball(rd.phi, 0, 0), inv0));
            }
        // ---- V1: parity, integrality, specialisation at the interpolation weights, Lagrange == P8's NewtonWeights
        {
            bool par = true;
            for (size_t wi = 0; wi < ks.size(); ++wi)
                for (uint32_t a = 0; a + 1 < l; ++a)
                    if (((a % 2 == 0) ? 1 : -1) != eps_geo)
                        for (const GR& g : s.br[wi][a])
                            for (uint64_t x : g) par = par && x == 0;
            Check(par, "V1 " + s.tag + ": the wrong-parity branches ((-1)^a != eps_geo = " + std::to_string(eps_geo) +
                           ") vanish at every weight (so identically in X)");
            bool nw = true;
            for (uint32_t kp : further) {
                const auto lw = TV::LagrangeAt(l, interp, kKp, TV::XOf(l, kp, eyw), eyw, false);
                const auto nwt = MS::NewtonWeights(l, interp, kKp, kp);
                nw = nw && lw == nwt;
            }
            Check(nw, "V1 " + s.tag + ": the Lagrange weights at every further weight == P8's NewtonWeights mod ell^5");
        }
        // ---- truncations (record) and V1's specialisation check, V2's truncation half
        uint32_t trunc_bad = 0, spec_bad = 0, spec_tot = 0;
        std::vector<std::string> further_txt;
        std::map<uint32_t, uint32_t> furtherTruncBad;   // m -> failures at further weights
        for (uint32_t m = 1; m <= 3; ++m) {
            const MS::TruncationMap TM = MS::MakeTruncationMap(l, interp, kKp, kK, m, cs.k0);
            for (uint32_t a = 0; a + 1 < l; ++a) {
                if (((a % 2 == 0) ? 1 : -1) != eps_geo) continue;
                // truncation of every g_i, per Y-power
                std::vector<std::vector<L::Vec>> tr(G, std::vector<L::Vec>(s.r));
                for (uint64_t i = 0; i < G; ++i)
                    for (uint32_t y = 0; y < s.r; ++y) {
                        std::vector<uint64_t> vals;
                        for (size_t j = 0; j < kN; ++j) vals.push_back(s.br[j][a][y][i]);
                        if (!TM.Apply(vals, tr[i][y])) ++trunc_bad;
                        for (size_t wi = 0; wi < ks.size(); ++wi) {
                            const uint32_t e = L::SpecPrecision(l, kK, m, cs.k0, ks[wi]);
                            const uint64_t got = static_cast<uint64_t>(L::SpecialiseAt(L::QScalar(tr[i][y], 1), l, cs.k0, ks[wi], e)[0] % PowU(l, e));
                            const bool okv = got == s.br[wi][a][y][i] % PowU(l, e);
                            if (wi < kN) {
                                ++spec_tot;
                                spec_bad += !okv;
                            } else if (!okv) {
                                ++furtherTruncBad[m];
                            }
                        }
                    }
                // Taylor shift per Z-coefficient: c_t = sum_i C(i, t) g_i (mod ell^K)
                const uint64_t qK = PowU(l, kK);
                std::vector<std::vector<L::Vec>> ct(G, std::vector<L::Vec>(s.r, L::Vec(m, 0)));
                for (uint32_t y = 0; y < s.r; ++y)
                    for (uint32_t z = 0; z < m; ++z) {
                        GR col(G);
                        for (uint64_t i = 0; i < G; ++i) col[i] = static_cast<uint64_t>(tr[i][y][z] % qK);
                        const GR sh = TV::TaylorShift(col, qK);
                        for (uint64_t t = 0; t < G; ++t) ct[t][y][z] = sh[t];
                    }
                std::vector<std::string> fur;
                for (uint32_t kp : further) fur.push_back(std::to_string(kp) + ":" + (furtherTruncBad[m] ? "fail" : "ok"));
                std::cout << RecordLine(l, cs.a, m, cs.k0, M, s, p0, a, ct, fur) << std::endl;
            }
        }
        Check(trunc_bad == 0 && spec_bad == 0,
              "V1 " + s.tag + ": every branch coordinate's truncation (m = 1, 2, 3) is ell-integral and specialises to the weight's branch element at the " +
                  std::to_string(kN) + " interpolation weights (" + std::to_string(spec_tot) + " comparisons)");
        {
            uint32_t tb = 0;
            for (const auto& kv : furtherTruncBad) tb += kv.second;
            Finding(tb == 0, "V2 " + s.tag + ": the truncations (m = 1, 2, 3) specialise to the further weights' branch elements mod ell^{SpecPrecision} (failures=" +
                                 std::to_string(tb) + ")");
        }
        // ---- V2: Lagrange prediction of every branch coordinate (and nu_{Z_ell}) at the further weights
        {
            uint32_t mism = 0, tot = 0, vmin = kKp;
            for (size_t fi = 0; fi < further.size(); ++fi) {
                const auto lw = TV::LagrangeAt(l, interp, kKp, TV::XOf(l, further[fi], eyw), eyw, false);
                auto cmp = [&](const std::vector<uint64_t>& vals, uint64_t got) {
                    ++tot;
                    const uint64_t pr = TV::Combine(lw, vals, l, kKp);
                    if (pr % qpred != got % qpred) {
                        ++mism;
                        vmin = std::min(vmin, ValU(Rk.sub(pr, got), l, kKp));
                    }
                };
                for (uint32_t a = 0; a + 1 < l; ++a)
                    for (uint32_t y = 0; y < s.r; ++y)
                        for (uint64_t i = 0; i < G; ++i) {
                            std::vector<uint64_t> vals;
                            for (size_t j = 0; j < kN; ++j) vals.push_back(s.br[j][a][y][i]);
                            cmp(vals, s.br[kN + fi][a][y][i]);
                        }
                if (!s.full.empty()) cmp(std::vector<uint64_t>(s.full.begin(), s.full.begin() + kN), s.full[kN + fi]);
            }
            Finding(mism == 0, "V2 " + s.tag + ": numerical control -- the interpolant predicts every branch coordinate" +
                                   std::string(s.full.empty() ? "" : " and nu_{Z_ell}") + " at the " + std::to_string(further.size()) +
                                   " further weights mod ell^" + std::to_string(prec_pred) + " (" + std::to_string(tot) + " values, mismatches=" +
                                   std::to_string(mism) + (mism ? ", min v=" + std::to_string(vmin) : "") + ")");
        }
        // ---- V3 / V4: P1's lfunction/v1 pipeline (r = 1)
        if (s.r == 1) {
            uint32_t v3bad = 0, v3tot = 0, v4bad = 0, v4tot = 0, notsel = 0, noplus = 0, p1ref = 0;
            size_t w_ok = ks.size();
            std::string seps;
            for (size_t wi = 0; wi < ks.size(); ++wi) {
                const P1Oracle o = RunP1(l, ks[wi], p0, s.rd[wi], s.sign, M, s.B0, primes, false, eis_elsewhere && !s.eis);
                if (!o.ok) {
                    if (o.why == "no r^+")
                        ++noplus;
                    else if (o.why.rfind("P1 refused", 0) == 0)
                        ++p1ref;
                    else
                        ++notsel;
                    std::cout << "  P1 " << s.tag << " k=" << ks[wi] << ": oracle not run (" << o.why << ")" << std::endl;
                    continue;
                }
                if (w_ok == ks.size() && wi < kN) {
                    w_ok = wi;
                    seps = o.sep;
                }
                for (const lv::LFunctionBranch& b : o.rec.branches) {
                    GR mine;
                    if (wi < kN) {
                        mine = TV::TaylorShift(s.br[wi][b.a][0], Rk.q);
                    } else {
                        GR pg(G);
                        const auto lw = TV::LagrangeAt(l, interp, kKp, TV::XOf(l, ks[wi], eyw), eyw, false);
                        for (uint64_t i = 0; i < G; ++i) {
                            std::vector<uint64_t> vals;
                            for (size_t j = 0; j < kN; ++j) vals.push_back(s.br[j][b.a][0][i]);
                            pg[i] = TV::Combine(lw, vals, l, kKp);
                        }
                        mine = TV::TaylorShift(pg, Rk.q);
                    }
                    const uint64_t qq = wi < kN ? Rk.q : qpred;
                    bool ok = b.coefficients.size() == mine.size();
                    for (size_t t = 0; ok && t < mine.size(); ++t) ok = b.coefficients[t] % qq == Rk.mul(o.muB0 % Rk.q, mine[t]) % qq;
                    if (wi < kN) {
                        ++v3tot;
                        v3bad += !ok;
                    } else {
                        ++v4tot;
                        v4bad += !ok;
                    }
                }
            }
            const bool none = noplus + p1ref == ks.size();
            Check(notsel == 0, "V3 " + s.tag + ": P1's RunSymbolCell selects this component wherever P1 runs (separator " + seps + "; weights without "
                                   "P1's r^+: " + std::to_string(noplus) + ", refused by P1 itself: " + std::to_string(p1ref) + ")");
            Check(v3bad == 0 && (v3tot > 0 || none), "V3 " + s.tag + ": P1's lfunction/v1 == mu^{P1}(B_0) * L_a(x_k) mod ell^5 at the interpolation weights (" +
                                               std::to_string(v3tot) + " branches, every t)");
            if (v4tot > 0)
                Finding(v4bad == 0, "V4 " + s.tag + ": P1's lfunction/v1 at the further weights == mu^{P1}(B_0) * the PREDICTED L_a mod ell^" +
                                        std::to_string(prec_pred) + " (" + std::to_string(v4tot) + " branches, mismatches=" + std::to_string(v4bad) + ")");
            std::cout << "  P1_COVERAGE " << s.tag << " ran_interp_branches=" << v3tot << " ran_further_branches=" << v4tot << " no_r_plus=" << noplus
                      << " refused_by_P1=" << p1ref << " of " << ks.size() << " weights" << std::endl;
            // negative control: the other sign's symbol (V3 must fail)
            if (!s.p0side && w_ok < ks.size()) {
                const size_t w0 = w_ok;
                const P1Oracle o = RunP1(l, ks[w0], p0, s.rd[w0], s.sign, M, s.B0, primes, true, false);
                bool caught = !o.ok;
                if (o.ok) {
                    for (const lv::LFunctionBranch& b : o.rec.branches) {
                        const GR mine = TV::TaylorShift(s.br[w0][b.a][0], Rk.q);
                        for (size_t t = 0; t < mine.size(); ++t)
                            if (b.coefficients[t] % Rk.q != Rk.mul(o.muB0 % Rk.q, mine[t])) caught = true;
                    }
                }
                Check(caught, "NEG " + s.tag + ": P1's oracle with the other sign's symbol fails V3");
            }
        }
        std::cout << "  SIDE " << s.tag << " B0=" << s.B0.b << "+ell^" << s.B0.M << " branches=" << (l - 1) << " G=" << G << " wall_s=" << Now() - ts
                  << std::endl;
    }

    // ---- negative control: the branch element with omega^{a+1} (V1's parity vanishing fails)
    for (const Side& s : sides) {
        if (!s.usable) continue;
        const int eps_geo = s.sign > 0 ? -1 : 1;
        bool caught = false;
        for (uint32_t a = 0; a + 1 < l && !caught; ++a) {
            if (((a % 2 == 0) ? 1 : -1) == eps_geo) continue;
            const GR g = TV::BranchFromBalls([&](uint64_t b) { return s.nu[0][bidx.at(b)][0]; }, l, M, a + 1, kKp);
            for (uint64_t x : g) caught = caught || x != 0;
        }
        Check(caught, "NEG " + s.tag + ": omega^{a+1} in place of omega^a breaks the parity vanishing");
        break;
    }

    // ---- the Eisenstein arm (V5, V6, V7)
    if (cs.ME > 0) {
        const double te = Now();
        const Side* sp = nullptr;
        for (const Side& s : sides)
            if (s.eis && s.sign > 0 && s.usable) sp = &s;
        Check(sp != nullptr, "V5 " + ctag + ": the [C]-plus (geometric-odd) side of the component through m_Eis is read");
        if (sp) {
            const Side& s = *sp;
            const uint32_t ME = std::max(cs.ME, M);
            // alpha_a from the exact Bernoulli numbers
            const uint32_t NG = l > 50 ? 6 : 7, PG = NG;
            std::vector<uint32_t> kg;
            for (uint32_t j = 0; j < NG; ++j) kg.push_back(cs.k0 + j * static_cast<uint32_t>(l - 1));
            const auto bk = L::BernoulliOverKExact(l, PG, kg);
            const bm::Ring RG = bm::MakeRing(l, PG);
            std::vector<uint64_t> gv;
            for (uint32_t k : kg) gv.push_back(RG.neg(RG.mul(RG.sub(1, RG.pow(l % RG.q, k - 1)), static_cast<uint64_t>(bk.at(k) % RG.q))));
            uint64_t resid = 1;
            const u128 alpha = TV::ZeroInDisc(l, kg, gv, PG, &resid);
            const uint32_t pa = std::min(PG, NG);
            {
                const uint64_t exp = l == 37 ? 281570 : (l == 103 ? 2266 : 0);
                const uint32_t ee = l == 37 ? 4 : 2;
                Check(resid == 0 && (exp == 0 || static_cast<uint64_t>(alpha % PowU(l, ee)) == exp),
                      "V7 " + ctag + ": alpha_a = the zero of G_a (7 exact Bernoulli values) = " + L::U128Text(alpha) + " mod ell^" + std::to_string(pa) +
                          (exp ? "; == [U]'s Coates-Wiles value " + std::to_string(exp) + " mod ell^" + std::to_string(ee) : std::string("")));
            }
            // the Kubota-Leopoldt side at level ME, folded to M
            const uint64_t q = Rk.q;
            std::map<uint32_t, GR> KL;
            for (uint32_t b = 2; b + 1 < l; b += 2) KL[b] = bm::KlGroupRingBernoulli(b, l, ME, kKp);
            auto Gb = [&](int64_t b) -> const GR& {
                return KL.at(static_cast<uint32_t>(((b % static_cast<int64_t>(l - 1)) + static_cast<int64_t>(l - 1)) % static_cast<int64_t>(l - 1)));
            };
            const uint32_t pole1 = static_cast<uint32_t>(l - 2), pole2 = (cs.k0 - 1) % static_cast<uint32_t>(l - 1);
            std::vector<uint32_t> odd;
            for (uint32_t a = 1; a + 1 < l; a += 2)
                if (a != pole1 && a != pole2) odd.push_back(a);
            auto Eis = [&](uint32_t a, uint32_t k, int twist) {
                return lv::FoldBranchElement(bm::GrMul(bm::GrTw1(Gb(static_cast<int64_t>(a) + 1 + twist), l, q),
                                                       bm::GrTwInv(Gb(static_cast<int64_t>(k) - 1 - a), l, q, k), q),
                                             G, q);
            };
            // per weight: slots (a, i) -> (P, E)
            std::vector<std::vector<uint64_t>> P(ks.size()), E(ks.size()), E3(ks.size());
            for (size_t wi = 0; wi < ks.size(); ++wi)
                for (uint32_t a : odd) {
                    const int tw = (a + 3) % (l - 1) != 0 ? 2 : 4;   // the wrong twist G_{a+3} (G_{a+5} where a + 3 == 0 mod ell - 1)
                    const GR e = Eis(a, ks[wi], 0), e3 = Eis(a, ks[wi], tw);
                    for (uint64_t i = 0; i < G; ++i) {
                        P[wi].push_back(s.br[wi][a][0][i]);
                        E[wi].push_back(e[i]);
                        E3[wi].push_back(e3[i]);
                    }
                }
            const size_t S = P[0].size();
            // cross-product valuation of (P, E) at one point
            auto crossV = [&](const std::vector<uint64_t>& Pp, const std::vector<uint64_t>& Ep, uint32_t cap) {
                size_t star = S;
                for (size_t i = 0; i < S && star == S; ++i)
                    if (Ep[i] % l != 0) star = i;
                if (star == S) return std::make_pair(static_cast<uint32_t>(0), star);
                uint32_t vmin = cap;
                for (size_t i = 0; i < S; ++i) {
                    const uint64_t c = Rk.sub(Rk.mul(Pp[i], Ep[star]), Rk.mul(Pp[star], Ep[i]));
                    vmin = std::min(vmin, ValU(c, l, cap));
                }
                return std::make_pair(vmin, star);
            };
            // V5: weight by weight, mod ell^{min(L_k, ME)}
            {
                const auto bk6 = L::BernoulliOverKExact(l, 6, ks);
                const L::ZMod R6(l, 6);
                std::ostringstream os;
                for (size_t wi = 0; wi < ks.size(); ++wi) {
                    const uint32_t Lk = R6.val(bk6.at(ks[wi]), 6);
                    const uint32_t need = std::min<uint32_t>(Lk, std::min(ME, kKp));
                    const auto cv = crossV(P[wi], E[wi], kKp);
                    os << " " << ks[wi] << ":L=" << Lk << ",v=" << cv.first;
                    const std::string what = "V5 " + s.tag + " k=" + std::to_string(ks[wi]) + ": L_a(x_k) == c_k * G_{a+1} G_{k-1-a} (cross products of " +
                                             std::to_string(S) + " slots, " + std::to_string(odd.size()) + " off-pole odd branches) mod ell^" +
                                             std::to_string(need) + " (v=" + std::to_string(cv.first) + ")";
                    if (ks[wi] == cs.k0 || Lk >= 2)
                        Check(cv.first >= need, what);
                    else
                        Finding(cv.first >= need, what);
                }
                std::cout << "  P9_EIS2V weightwise" << os.str() << std::endl;
            }
            // V6: at X = alpha_a
            {
                const uint32_t need = std::min(std::min(kKp, kN), ME);
                auto at = [&](const std::vector<std::vector<uint64_t>>& V, u128 y, uint32_t ey) {
                    const auto lw = TV::LagrangeAt(l, interp, kKp, y, ey, false);
                    std::vector<uint64_t> out(S);
                    for (size_t i = 0; i < S; ++i) {
                        std::vector<uint64_t> vals;
                        for (size_t j = 0; j < kN; ++j) vals.push_back(V[j][i]);
                        out[i] = TV::Combine(lw, vals, l, kKp);
                    }
                    return out;
                };
                const auto Pa = at(P, alpha, pa), Ea = at(E, alpha, pa);
                const auto cv = crossV(Pa, Ea, kKp);
                Finding(cv.first >= need, "V6 " + s.tag + ": the TWO-VARIABLE Eisenstein congruence -- at X = alpha_a the cross products of L_a(X, T) and "
                                                          "G_{a+1} G_{k-1-a}(X, T) vanish mod ell^" +
                                              std::to_string(need) + " (v_min=" + std::to_string(cv.first) + ", " + std::to_string(S) + " slots)");
                // negative controls: the wrong twist G_{a+3}; alpha_a + ell
                const auto E3a = at(E3, alpha, pa);
                const auto cv3 = crossV(Pa, E3a, kKp);
                Check(cv3.first < need, "NEG " + s.tag + ": the wrong twist G_{a+3} fails V6 (v_min=" + std::to_string(cv3.first) + ")");
                const u128 alpha2 = (alpha + l) % PowU(l, pa);
                const auto Pb = at(P, alpha2, pa), Eb = at(E, alpha2, pa);
                const auto cvb = crossV(Pb, Eb, kKp);
                Check(cvb.first < need, "NEG " + s.tag + ": alpha_a + ell in place of alpha_a fails V6 (v_min=" + std::to_string(cvb.first) + ")");
                // the reading at the integer points nearest alpha (for the record): v at x_{k0}
                const auto cvc = crossV(P[0], E[0], kKp);
                std::cout << "  P9_EIS2V cell=" << ctag << " M=" << M << " M_E=" << ME << " alpha=" << L::U128Text(alpha) << " mod ell^" << pa
                          << " slots=" << S << " v_at_alpha=" << cv.first << " need=" << need << " v_at_centre=" << cvc.first
                          << " NEG_twist_v=" << cv3.first << " NEG_alpha+ell_v=" << cvb.first << " wall_s=" << Now() - te << std::endl;
            }
        }
    }

    // ---- the weight-2 arm at (11, 2): 11a (V8 - V12)
    if (l == 11 && cs.a == 2) {
        const double t2 = Now();
        const u128 y2 = TV::XOf(11, 2, eyw);
        for (const Side& s : sides) {
            if (!s.usable || s.r != 1) continue;
            const int sign11 = s.sign < 0 ? 1 : -1;   // geometric even = [C]-minus <-> 11a plus
            const TV::Gamma0Symbol sy = TV::Gamma0Weight2Symbol(11, sign11, -2, 11, kKp + 1);
            const uint32_t Ma = 5;
            const auto mu11 = TV::MultiplicativeMeasure(sy, 11, Ma, 1);
            const bm::Ring R6 = sy.R;
            // V8
            uint32_t distbad = 0;
            for (uint32_t m = 0; m < Ma; ++m)
                for (uint64_t a = 0; a < mu11[m].size(); ++a) {
                    uint64_t sm = 0;
                    for (uint64_t b = 0; b < 11; ++b) sm = R6.add(sm, mu11[m + 1][a + b * mu11[m].size()]);
                    distbad += sm != mu11[m][a];
                }
            uint64_t units = 0;
            for (uint64_t b = 1; b < 11; ++b) units = R6.add(units, mu11[1][b]);
            Check(sy.free_rank1 && distbad == 0 && units == 0,
                  "V8 " + s.tag + ": 11a's " + (sign11 > 0 ? "plus" : "minus") + " symbol on Gamma_0(11) is free of rank 1 (psi o T_2 = -2 psi), its MTT measure is a "
                  "distribution to level 5 (U_11 = 1) and mu(Z_11^x) = 0 (the trivial zero)");
            // V9: control at k = 2, every unit ball to level M, cross-multiplied
            const auto lw2 = TV::LagrangeAt(11, interp, kKp, y2, eyw, false);
            const uint64_t m11B0 = mu11[s.B0.M][s.B0.b] % Rk.q;
            uint32_t bad = 0, tot = 0, vmin = kKp;
            std::map<uint64_t, uint64_t> nu2lvl1;
            for (size_t bi = 0; bi < s.rd[0].balls.size(); ++bi) {
                const MS::Ball& B = s.rd[0].balls[bi];
                std::vector<uint64_t> vals;
                for (size_t j = 0; j < kN; ++j) vals.push_back(s.nu[j][bi][0]);
                const uint64_t pr = TV::Combine(lw2, vals, 11, kKp);
                if (B.M == 1) nu2lvl1[B.b] = pr;
                const uint64_t lhs = Rk.mul(pr, m11B0), rhs = mu11[B.M][B.b] % Rk.q;
                ++tot;
                if (lhs % qpred != rhs % qpred) {
                    ++bad;
                    vmin = std::min(vmin, ValU(Rk.sub(lhs, rhs), 11, kKp));
                }
            }
            Finding(bad == 0, "V9 " + s.tag + ": control at k = 2 -- the family's predicted nu_B(x_2) * mu_11a(B_0) == mu_11a(B) mod 11^" +
                                  std::to_string(prec_pred) + " on all " + std::to_string(tot) + " unit balls to level " + std::to_string(M) +
                                  " (B_0 = " + std::to_string(s.B0.b) + "+11^" + std::to_string(s.B0.M) + ", mu_11a(B_0) = " + std::to_string(m11B0) +
                                  "; mismatches=" + std::to_string(bad) + (bad ? ", min v=" + std::to_string(vmin) : "") + ")");
            Finding(m11B0 % 11 != 0, "V9 " + s.tag + ": (U16)(a) at k = 2 -- mu_11a(B_0) is a unit (B_0 the family's normaliser of record)");
            std::ostringstream os;
            os << "  P9_11A " << s.tag << " B0=" << s.B0.b << "+11^" << s.B0.M << " nu(x_2) level-1 = [";
            for (const auto& kv : nu2lvl1) os << (kv.first > 1 ? "," : "") << kv.second;
            os << "] mu_11a level-1 = [";
            for (uint64_t b = 1; b < 11; ++b) os << (b > 1 ? "," : "") << mu11[1][b] % Rk.q;
            os << "]";
            if (s.full.empty()) {
                std::cout << os.str() << std::endl;
                continue;
            }
            // V9 (full mass), V10, V11
            const std::vector<uint64_t> fullI(s.full.begin(), s.full.begin() + kN);
            const uint64_t full2 = TV::Combine(lw2, fullI, 11, kKp);
            Finding(Rk.mul(full2, m11B0) % qpred == mu11[0][0] % qpred,
                    "V9 " + s.tag + ": the improved value -- nu_{Z_11}(x_2) * mu_11a(B_0) == mu_11a(Z_11) = Phi_11a({0, oo}) mod 11^" + std::to_string(prec_pred));
            std::vector<uint64_t> unitI(kN, 0), alphaI(kN, 0);
            for (size_t j = 0; j < kN; ++j) {
                for (size_t bi = 0; bi < s.rd[0].balls.size(); ++bi)
                    if (s.rd[0].balls[bi].M == 1) unitI[j] = Rk.add(unitI[j], s.nu[j][bi][0]);
                alphaI[j] = s.rd[j].P.at(11)[0] % Rk.q;
            }
            const uint64_t unit2 = TV::Combine(lw2, unitI, 11, kKp);
            if (s.sign < 0) {
                Finding(unit2 % qpred == 0, "V10 " + s.tag + ": the trivial zero seen from k >= 12 -- nu_{Z_11^x}(x_2) == 0 mod 11^" + std::to_string(prec_pred) +
                                                " (got " + std::to_string(unit2) + ")");
                const auto dw = TV::LagrangeAt(11, interp, kKp, y2, eyw, true);
                const uint64_t dunit = TV::Combine(dw, unitI, 11, kKp), dalpha = TV::Combine(dw, alphaI, 11, kKp);
                Finding(dunit % qpred == Rk.mul(dalpha, full2) % qpred,
                        "V11 " + s.tag + ": Greenberg-Stevens' derivative -- ell d/dX nu_{Z_11^x}(x_2) == ell alpha'(x_2) * nu_{Z_11}(x_2) mod 11^" +
                            std::to_string(prec_pred) + " (" + std::to_string(dunit) + " vs " + std::to_string(Rk.mul(dalpha, full2)) + ")");
                os << " nu_Z11(x_2)=" << full2 << " nu_Z11x(x_2)=" << unit2 << " ell*dnu_Z11x=" << dunit << " ell*alpha'(x_2)=" << dalpha;
            }
            std::cout << os.str() << std::endl;
        }
        // V12: the L-invariant three ways
        {
            const uint32_t Wa = 8, Na = 8;
            std::vector<uint32_t> ka;
            for (uint32_t j = 0; j < Na; ++j) ka.push_back(12 + 10 * j);
            std::vector<uint64_t> al;
            bool rank1 = true;
            for (uint32_t k : ka) {
                const L::FormsWeight fw = L::ComputeFormsWeight(11, k, Wa, {2, 3});
                rank1 = rank1 && fw.S.s == 1;
                al.push_back(fw.S.s == 1 ? static_cast<uint64_t>(fw.S.mats.at("U")[0][0] % PowU(11, Wa)) : 0);
            }
            bool agree = rank1;
            for (const Side& s : sides)
                if (s.usable && s.r == 1)
                    for (size_t j = 0; j < kN; ++j) agree = agree && al[j] % Rk.q == s.rd[j].P.at(11)[0] % Rk.q;
            Check(agree, "V12 " + ctag + ": the forms side at W = 8 has e S_k of rank 1 at k = 12..82 and its alpha == P8's U mod 11^5");
            const uint64_t q8 = PowU(11, Wa);
            const bm::Ring R8 = bm::MakeRing(11, Wa);
            const auto dw = TV::LagrangeAt(11, ka, Wa, TV::XOf(11, 2, Wa + Na), Wa + Na, true);
            const uint64_t ldalpha = TV::Combine(dw, al, 11, Wa);   // ell alpha'(x_2) mod 11^8
            const uint64_t logu_over_ell = TV::LogIwasawa(12, 11, Wa, true);
            const uint64_t dadk = R8.mul(R8.mul(ldalpha, 144 % q8), logu_over_ell);   // alpha'(x_2) u^2 log(u)
            const uint64_t LGS = R8.neg(R8.mul(2, dadk));
            const uint64_t LT = TV::TateLInvariant11a(Wa);
            Check(LT == 14882352232ull % q8, "V12 ORACLE Tate's L(11a) mod 11^8 == the Python oracle's 14882352232 (committed with the spec)");
            Finding(LGS == LT, "V12 " + ctag + ": L_GS = -2 dalpha/dk(2) (family, forms side, 8 weights at W = 8) == L_Tate mod 11^8 (" + std::to_string(LGS) +
                                   " vs " + std::to_string(LT) + ")");
            Check(R8.neg(dadk) != LT, "NEG " + ctag + ": L_GS without the factor -2 does not match L_Tate");
            // the cyclotomic derivative of 11a's plus measure
            const TV::Gamma0Symbol sy = TV::Gamma0Weight2Symbol(11, 1, -2, 11, 6);
            const uint32_t Ma = 5;
            const auto mu = TV::MultiplicativeMeasure(sy, 11, Ma, 1);
            const GR g0 = TV::BranchFromBalls([&](uint64_t b) { return mu[Ma][b]; }, 11, Ma, 0, 6);
            const GR c = TV::TaylorShift(g0, PowU(11, 6));
            const bm::Ring R5 = bm::MakeRing(11, Ma);
            const uint64_t LM = R5.mul(R5.mul(TV::LogIwasawa(12, 11, Ma, false), c[1] % R5.q), R5.inv(mu[0][0] % R5.q));
            Finding(LM == LT % R5.q, "V12 " + ctag + ": L_MTT = log(u) c_1 / mu(Z_11) of 11a's plus measure (level 5) == L_Tate mod 11^5 (" + std::to_string(LM) +
                                         " vs " + std::to_string(LT % R5.q) + ")");
            // negative control: the Eisenstein eigenvalue +3 for T_2 (the plus module is not the cusp form's)
            const TV::Gamma0Symbol se = TV::Gamma0Weight2Symbol(11, 1, 3, 11, 6);
            bool caught = !se.free_rank1;
            if (!caught) {
                const auto me = TV::MultiplicativeMeasure(se, 11, 1, 1);
                uint64_t u = 0;
                for (uint64_t b = 1; b < 11; ++b) u = R5.add(u, me[1][b] % R5.q);
                const uint64_t ratio = me[0][0] % 11 ? R5.mul(u, R5.inv(me[0][0] % R5.q)) : 1;
                caught = u != 0 || ratio != 0 || LM == 0;
                // the Eisenstein functional is the boundary: its Z_11^x mass is not 0 or the space differs
                caught = caught || me[1][1] % 11 != mu[1][1] % 11 || me[1][2] % 11 != mu[1][2] % 11;
            }
            Check(caught, "NEG " + ctag + ": 11a with psi o T_2 = +3 psi (the Eisenstein eigenvalue) is not 11a's symbol");
            std::cout << "  P9_11A L_Tate=" << LT << " L_GS=" << LGS << " (mod 11^8) L_MTT=" << LM << " (mod 11^5) ell*alpha'(x_2)=" << ldalpha
                      << " wall_s=" << Now() - t2 << std::endl;
        }
    }
    std::cout << "CELL " << ctag << " wall_s=" << Now() - tc << std::endl;
}

}  // namespace

uint32_t HeckeLambdaTwoVarValidation() {
    const double t_start = Now();
    g_check.reset();
    g_finding.reset();
    g_by_gate.clear();
    std::vector<CellSpec> cells{{11, 2, 12, 3, 0}, {13, 0, 12, 3, 0}, {23, 2, 24, 2, 0}, {37, 32, 32, 2, 3}};
    const uint32_t Fw = 3;
    std::string arm = "DEFAULT";
    if (const char* e = std::getenv("HECKE_LAMBDA_TWOVAR_ALL"))
        if (std::string(e) == "1") {
            arm = "ALL";
            cells[3].M = 3;
            cells.push_back({103, 24, 24, 2, 2});
        }
    if (const char* e = std::getenv("HECKE_LAMBDA_TWOVAR_CELLS")) {
        arm = "CELLS";
        cells.clear();
        bool bad = false;
        for (const oc_util::cells::Token& t : oc_util::cells::Tokenize(e)) {
            const size_t pos = t.pos;
            const std::string& tok = t.text;
            const bool fits = t.Fits(3, 3);
            const uint64_t a = fits ? t.fields[0] : 0, b = fits ? t.fields[1] : 0, c = fits ? t.fields[2] : 0;
            if (!fits || !IsPrime(a) || a < 5 || b % 2 || c < 12 ||
                (c + 1000 * (a - 1) - b) % (a - 1) != 0) {
                std::cout << "  HECKE_LAMBDA_TWOVAR_CELLS: token " << pos << " '" << tok
                          << "' REFUSED (want ell:a:k0, ell >= 5 prime, a even, k0 >= 12 in the class)" << std::endl;
                bad = true;
                continue;
            }
            const bool eisc = (a == 37 && b == 32) || (a == 103 && b == 24);
            cells.push_back({a, static_cast<uint32_t>(b), static_cast<uint32_t>(c), a <= 13 ? 3u : 2u, eisc ? (a == 37 ? 3u : 2u) : 0u});
        }
        Check(!bad && !cells.empty(), "ENV HECKE_LAMBDA_TWOVAR_CELLS selects at least one cell and every token is well formed");
    }
    std::cout << "EFFECTIVE HeckeLambdaTwoVarValidation arm=" << arm << " cells=" << cells.size() << " K=" << kK << " Kp=" << kKp << " m=1..3 N=" << kN
              << " F=" << Fw << " B=" << kB << std::endl;

    // arithmetic self-checks of HeckeLambdaTwoVar (code checks)
    {
        // Lagrange weights reproduce a polynomial of degree < N exactly; derivative weights its derivative
        const uint64_t l = 13;
        const std::vector<uint32_t> w{12, 24, 36, 48, 60};
        const L::ZMod Q(l, 9);
        const bm::Ring R = bm::MakeRing(l, kKp);
        auto f = [&](u128 x) { return Q.add(Q.add(Q.mul(7, Q.pow(x, 4)), Q.mul(3, x)), 11); };
        auto fp = [&](u128 x) { return Q.add(Q.mul(28, Q.pow(x, 3)), 3); };
        std::vector<uint64_t> v;
        for (uint32_t k : w) v.push_back(static_cast<uint64_t>(f(TV::XOf(l, k, 9)) % R.q));
        const u128 y = TV::XOf(l, 2, 9);
        const uint64_t pv = TV::Combine(TV::LagrangeAt(l, w, kKp, y, 9, false), v, l, kKp);
        const uint64_t pd = TV::Combine(TV::LagrangeAt(l, w, kKp, y, 9, true), v, l, kKp);
        Check(pv == static_cast<uint64_t>(f(y) % R.q) && pd == R.mul(13, static_cast<uint64_t>(fp(y) % R.q)),
              "ORACLE Lagrange weights at y = x_2 reproduce a quartic and (times ell) its derivative mod 13^5");
        // Taylor shift: sum_t c_t T^t at T = u^s - 1 equals sum_i g_i u^{is}
        const GR g{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9};
        const GR c = TV::TaylorShift(g, R.q);
        const uint64_t s = 7;
        const uint64_t T = R.sub(R.pow(14, s), 1);
        uint64_t lhs = 0, tp = 1;
        for (uint64_t x : c) {
            lhs = R.add(lhs, R.mul(x, tp));
            tp = R.mul(tp, T);
        }
        Check(lhs == bm::GrEval(g, s, l, R.q), "ORACLE TaylorShift: sum c_t (u^s - 1)^t == WP-MEAS's GrEval");
        // Teichmueller: omega(d)^{ell-1} = 1, omega(d) == d mod ell
        bool tok = true;
        for (uint64_t d = 1; d < l; ++d) {
            const uint64_t t = TV::Teich(d, l, kKp);
            tok = tok && t % l == d && R.pow(t, l - 1) == 1;
        }
        Check(tok, "ORACLE Teichmueller lifts mod 13^5");
    }

    for (const CellSpec& cs : cells) {
        try {
            RunCell(cs, Fw);
        } catch (const std::exception& ex) {
            Check(false, "CELL " + CellTag(cs.ell, cs.a) + " threw: " + ex.what());
        }
    }
    // negative control: N = K + m - 2 weights refused by the interpolation (as in P8)
    {
        bool refused = false;
        try {
            MS::MakeTruncationMap(11, {12, 22, 32, 42}, kKp, kK, 3, 12);
        } catch (const std::logic_error& ex) {
            refused = std::string(ex.what()).find("REFUSED") != std::string::npos;
        }
        Check(refused, "NEG four weights for m = 3 (N = K + m - 2) are REFUSED by the truncation map");
    }
    std::cout << "HeckeLambdaTwoVarValidation: checks=" << g_check.checks << " failing=" << g_check.fails << " findings=" << g_finding.checks
              << " finding_failures=" << g_finding.fails << " wall_s=" << Now() - t_start << " peak_rss_mib=" << PeakRssMiB() << std::endl;
    for (const auto& kv : g_by_gate) std::cout << "  gate " << kv.first << ": " << kv.second << std::endl;
    return g_check.fails == 0 ? 0 : 1;
}
