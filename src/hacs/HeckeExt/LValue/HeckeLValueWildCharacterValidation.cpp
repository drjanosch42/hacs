// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 21, P4) HeckeLValueWildCharacterValidation -- see HeckeLValueWildCharacterValidation.h.
// Spec docs/implementation/slot21/P4_spec.md (v0.1, predictions committed before this code ran); the Python calibration of record is
// docs/implementation/slot21/p4/out_cal.txt.  Every expected value below is the spec's prediction, never this code's output.
#include "HeckeLValueWildCharacterValidation.h"

#include "Bernoulli.h"
#include "BirchModularSymbol.h"
#include "EisensteinMeasure.h"
#include "ModularSymbolMeasure.h"
#include "WildCharacter.h"

#include "NumberField/Cyclotomic.h"
#include "NumberField/MpReal.h"

#include "../../../ModArithmetic/EngineCheck.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hecke_ext {
namespace lvalue {
namespace {

using namespace birch;
using namespace wild;
using u128 = unsigned __int128;
using Elt = OArith::Elt;

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
double PeakRssGb() {
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
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

uint64_t Pow(uint64_t b, uint32_t e) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < e; ++i) r *= b;
    return r;
}
uint64_t PowM(uint64_t b, uint64_t e, uint64_t q) {
    u128 r = 1 % q, x = b % q;
    while (e) {
        if (e & 1) r = r * x % q;
        x = x * x % q;
        e >>= 1;
    }
    return static_cast<uint64_t>(r);
}
uint32_t ValU(uint64_t x, uint64_t ell) {
    uint32_t v = 0;
    while (x && x % ell == 0) {
        x /= ell;
        ++v;
    }
    return v;
}
uint32_t MaxK(uint64_t ell) { return static_cast<uint32_t>(std::floor(61.9 / std::log2(static_cast<double>(ell)))); }
std::string Tag(uint32_t n, uint64_t ell) { return "(" + std::to_string(n) + "," + std::to_string(ell) + ")"; }
uint64_t ModNeg(uint64_t x, uint64_t q) { return (q - x % q) % q; }

// a deterministic stream
struct Lcg {
    uint64_t s;
    uint64_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return s >> 11;
    }
};

// omega^x(rho) mod q for rho = 1 .. ell-1, every x mod ell-1: om[x][rho]
std::vector<std::vector<uint64_t>> OmegaTable(uint64_t ell, uint32_t P) {
    std::vector<std::vector<uint64_t>> om(ell - 1, std::vector<uint64_t>(ell, 0));
    for (uint32_t x = 0; x + 1 < ell; ++x)
        for (uint64_t r = 1; r < ell; ++r) om[x][r] = TeichmullerPowMod(r, x, ell, P);
    return om;
}
uint32_t ModL1(int64_t x, uint64_t ell) {
    const int64_t L1 = static_cast<int64_t>(ell - 1);
    return static_cast<uint32_t>(((x % L1) + L1) % L1);
}

// ======================================================================================================== W0
void RunW0(uint64_t ell, uint32_t m, Log& log) {
    const std::string tag = "(ell=" + std::to_string(ell) + ",m=" + std::to_string(m) + ")";
    const uint32_t P = std::min<uint32_t>(6, static_cast<uint32_t>(std::floor(31.9 / std::log2(static_cast<double>(ell)))));
    const OArith O = MakeOArith(ell, m - 1, P);
    const std::vector<uint32_t> logs = LogTable(ell, m);
    const uint64_t F = Pow(ell, m), G = Pow(ell, m - 1);
    Lcg rng{0x51a4ull * ell + m};
    uint32_t multBad = 0, primBad = 0, uBad = 0, negPrim = 0, nchars = 0;
    auto primitive = [&](const WildChar& c) {
        for (uint64_t a = 1; a < G; ++a) {
            if (a % ell == 0) continue;
            Elt s = O.zero();
            for (uint64_t b = 0; b < ell; ++b) s = O.add(s, c.value(O, a + b * G));
            if (!O.is_zero(s)) return false;
        }
        return true;
    };
    for (int64_t i : {int64_t(0), int64_t(1), static_cast<int64_t>(ell - 2)}) {
        for (int64_t t : {int64_t(1), int64_t(2)}) {
            const WildChar c = MakeWildChar(ell, m, i, t, P, logs);
            ++nchars;
            for (int r = 0; r < 40; ++r) {
                uint64_t a = rng.next() % F, b = rng.next() % F;
                if (a % ell == 0) a = 1 + ell;
                if (b % ell == 0) b = F - 1;
                multBad += !(c.value(O, static_cast<uint64_t>(static_cast<u128>(a) * b % F)) == O.mul(c.value(O, a), c.value(O, b)));
            }
            primBad += !primitive(c);
            uBad += !(c.value(O, 1 + ell) == O.zeta_pow(static_cast<uint64_t>(t)));
        }
        // NEG: t = ell (psi of order ell^{m-2}): conductor ell^{m-1} at most
        negPrim += primitive(MakeWildChar(ell, m, i, static_cast<int64_t>(ell), P, logs));
    }
    log.check("W0", multBad == 0 && primBad == 0 && uBad == 0, tag + ": characters: multiplicativity " + std::to_string(multBad) + ", primitivity " +
                                                                   std::to_string(primBad) + ", chi(u) " + std::to_string(uBad) + " failures");
    log.check("W0", negPrim == 0, tag + ": NEG t = ell passes the primitivity test at " + std::to_string(negPrim) + " / 3");
    // P4's O against NumberField's CycloModRing (uint32 coefficients, M = ell^P < 2^32)
    const nf::CycloModRing CR(static_cast<uint32_t>(ell), m - 1, static_cast<uint32_t>(O.q));
    uint32_t mulBad = 0, vpiBad = 0;
    for (int r = 0; r < 20; ++r) {
        Elt a(O.e), b(O.e);
        nf::CycloModRing::Elt A(O.e), B(O.e);
        for (uint32_t x = 0; x < O.e; ++x) {
            a[x] = rng.next() % O.q;
            b[x] = (r % 4 == 0) ? (a[x] * ell) % O.q : rng.next() % O.q;   // some elements of positive valuation
            A[x] = static_cast<uint32_t>(a[x]);
            B[x] = static_cast<uint32_t>(b[x]);
        }
        if (r % 5 == 1) {   // pi^3 * unit
            Elt pi = O.sub(O.one(), O.zeta_pow(1));
            b = O.mul(O.mul(pi, O.mul(pi, pi)), a);
            for (uint32_t x = 0; x < O.e; ++x) B[x] = static_cast<uint32_t>(b[x]);
        }
        const Elt ab = O.mul(a, b);
        const nf::CycloModRing::Elt AB = CR.mul(A, B);
        for (uint32_t x = 0; x < O.e; ++x) mulBad += ab[x] != AB[x];
        vpiBad += O.vpi(b) != CR.vpi(B);
    }
    log.check("W0", mulBad == 0 && vpiBad == 0, tag + ": O vs nf::CycloModRing: product mismatches " + std::to_string(mulBad) + ", v_pi mismatches " + std::to_string(vpiBad));
    std::cout << "WILD_W0 " << tag << " e=" << O.e << " P=" << P << " chars=" << nchars << " mult=" << multBad << " prim=" << primBad << " chi(u)=" << uBad
              << " neg_t=ell_primitive=" << negPrim << " cyclo_mul=" << mulBad << " cyclo_vpi=" << vpiBad << std::endl;
}

// ======================================================================================================== W1
void RunW1(uint64_t ell, uint32_t m, uint32_t M, uint32_t R, Log& log) {
    const double t0 = Now();
    const std::string tag = "(" + std::to_string(ell) + "," + std::to_string(m) + "," + std::to_string(M) + "," + std::to_string(R) + ")";
    const uint32_t rmax = 10;
    const uint32_t Pw = std::min<uint32_t>(MaxK(ell), R + m + 2);
    const OArith O = MakeOArith(ell, m - 1, Pw);
    const std::vector<uint32_t> logs = LogTable(ell, m);
    const std::vector<uint64_t> ellB = eis::EllBernoulliModEllPower(rmax, ell, Pw);
    const uint64_t F = Pow(ell, m), lim = Pow(ell, std::min(M, R));
    std::map<std::pair<uint32_t, int64_t>, std::vector<Elt>> cache;
    auto FB = [&](int64_t i, int64_t t) -> const std::vector<Elt>& {
        const auto key = std::make_pair(ModL1(i, ell), t);
        auto it = cache.find(key);
        if (it == cache.end()) it = cache.emplace(key, WildBernoulliFB(O, MakeWildChar(ell, m, i, t, Pw, logs), rmax, ellB)).first;
        return it->second;
    };
    std::vector<uint32_t> classes;
    for (uint32_t b = 2; b + 1 < ell; b += 2) classes.push_back(b);
    std::map<uint32_t, std::vector<uint64_t>> layer;
    const bool useLayer = ell <= 103;
    if (useLayer) layer = KlGroupRingLayer(classes, ell, M, MeasureThreads());
    uint32_t tot = 0, bad = 0, badLayer = 0, negMatch = 0, negTot = 0;
    for (uint32_t b : classes) {
        const std::vector<uint64_t> gB = KlGroupRingBernoulli(b, ell, M, R);
        for (int64_t t : {int64_t(1), int64_t(2)}) {
            for (uint32_t r = 1; r <= rmax; ++r) {
                const Elt& X = FB(static_cast<int64_t>(b) - r, t)[r];
                const uint64_t modc = lim * F * Pow(ell, ValU(r, ell));
                ENGINE_CHECK_MSG(O.q % modc == 0, "P4 W1: working precision too small");
                const uint64_t fr = static_cast<uint64_t>(static_cast<u128>(F) * r % O.q);
                const Elt Y = O.scal(ModNeg(fr, O.q), GroupRingAtWildPoint(O, gB, t, r));
                ++tot;
                bad += !O.eq_mod(X, Y, modc);
                if (useLayer) {
                    const Elt Yl = O.scal(ModNeg(fr, O.q), GroupRingAtWildPoint(O, layer.at(b), t, r));
                    const uint64_t modl = Pow(ell, std::min(M, R)) * F * Pow(ell, ValU(r, ell));
                    badLayer += !O.eq_mod(X, Yl, modl);
                }
                // NEG: zeta^{-t}
                bool nz = false;
                for (uint64_t x : X) nz = nz || (x % modc) != 0;
                if (nz) {
                    ++negTot;
                    negMatch += O.eq_mod(X, O.scal(ModNeg(fr, O.q), GroupRingAtWildPoint(O, gB, -t, r)), modc);
                }
            }
        }
    }
    // pole branch b = 0: v_pi(B_{r, omega^{-r} psi_t}/r) = -1, i.e. v_pi(F B) = e (m + v(r)) - 1
    uint32_t poleTot = 0, poleBad = 0;
    std::set<int32_t> poleV;
    for (int64_t t : {int64_t(1), int64_t(2)})
        for (uint32_t r = 1; r <= rmax; ++r) {
            const Elt& X = FB(-static_cast<int64_t>(r), t)[r];
            const int32_t v = static_cast<int32_t>(O.vpi(X)) - static_cast<int32_t>(O.e * (m + ValU(r, ell)));
            poleV.insert(v);
            ++poleTot;
            poleBad += v != -1;
        }
    log.check("W1", bad == 0 && tot > 0, tag + ": definition != -F r G_b(zeta^t u^r - 1) (E_1c route) at " + std::to_string(bad) + " / " + std::to_string(tot));
    if (useLayer) log.check("W1", badLayer == 0, tag + ": definition != the Coates-Wiles layer route at " + std::to_string(badLayer) + " / " + std::to_string(tot));
    log.check("W1", poleBad == 0, tag + ": pole branch v_pi(B/r) != -1 at " + std::to_string(poleBad) + " / " + std::to_string(poleTot));
    log.check("W1", negTot > 0 && negMatch == 0, tag + ": NEG zeta^{-t} matches at " + std::to_string(negMatch) + " / " + std::to_string(negTot) + " non-zero values");
    std::ostringstream pv;
    for (int32_t v : poleV) pv << v << ";";
    std::cout << "WILD_W1 cell=" << tag << " e=" << O.e << " points=" << tot << " fail_E1c=" << bad << " fail_layer=" << (useLayer ? std::to_string(badLayer) : "-")
              << " pole_vpi=" << pv.str() << " neg_conj=" << negMatch << "/" << negTot << " secs=" << (Now() - t0) << std::endl;
}

// ======================================================================================================== W2 / W3
struct CuspSpec {
    uint32_t n;
    uint64_t ell;
    uint32_t R, M, m;
    std::map<uint64_t, uint64_t> sep;
    bool control;
};

void RunCusp(const CuspSpec& cs, Log& log) {
    const double t0 = Now();
    const uint32_t n = cs.n, k = n + 2, M = cs.M, m = cs.m;
    const uint64_t ell = cs.ell;
    const std::string tag = Tag(n, ell) + " R=" + std::to_string(cs.R) + " M=" + std::to_string(M) + " m=" + std::to_string(m);
    const SymbolCell C = RunSymbolCell(n, ell, cs.R, cs.sep, cs.control);
    if (!C.simple) {
        log.check("W2", false, tag + ": the symbol cell did not run: " + C.refusal);
        return;
    }
    const Ring& Rg = C.ring;
    const uint64_t q = Rg.q;
    const SymRho rho(n, Rg);
    const uint64_t a_ell = C.a_p.at(ell);
    const uint64_t alpha = UnitRoot(Rg, a_ell, k);
    const uint64_t ia = Rg.inv(alpha);
    const OArith O = MakeOArith(ell, m - 1, cs.R);
    const std::vector<uint32_t> logs = LogTable(ell, m);
    const auto om = OmegaTable(ell, cs.R);
    const uint64_t F = Pow(ell, m), MM = Pow(ell, M);
    const uint32_t vAlpha = Rg.val(Rg.sub(alpha, 1 % q));
    // C(n, j) mod q
    std::vector<uint64_t> binom(n + 1, 1 % q);
    for (uint32_t j = 1; j <= n; ++j) binom[j] = rho.binom(n, j) % q;
    std::ostringstream lines;
    struct SideSpec {
        const Vec* P;
        int eps;
        uint32_t prec;   // R' (r^+ is known mod ell^{R - d'})
        const char* name;
    };
    std::vector<SideSpec> sides;
    sides.push_back({&C.r_minus, -1, cs.R, "r-"});
    if (C.have_r_plus) {
        uint32_t dp = 0;
        for (uint32_t p : C.wplus_pivots)
            if (p < cs.R) dp = std::max(dp, p);
        sides.push_back({&C.r_plus, 1, cs.R - dp, "r+"});
    }
    for (const SideSpec& sd : sides) {
        const Vec& P = *sd.P;
        const ModularSymbolMeasure mm = BuildMeasure(rho, P, a_ell, alpha, M, true);
        const uint64_t lim = Pow(ell, std::min(M, sd.prec)), limR = Pow(ell, sd.prec);
        // bins: Bm[j][rho][L] = sum_{a unit < ell^M, a = rho mod ell, L(a mod F) = L} a^j mu(a + ell^M);
        //       Cv[j][rho][L] = sum_{a unit < F} < rho(gamma_{a,m}) Phi({a/F, oo}), u_{n-j}^vee >
        const uint32_t G1 = static_cast<uint32_t>(Pow(ell, m - 1));
        auto idx = [&](uint32_t j, uint64_t r, uint32_t L) { return (static_cast<size_t>(j) * ell + r) * G1 + L; };
        std::vector<uint64_t> Bm(static_cast<size_t>(n + 1) * ell * G1, 0), Cv(static_cast<size_t>(n + 1) * ell * G1, 0);
        for (uint64_t a = 1; a < MM; ++a) {
            if (a % ell == 0) continue;
            const uint64_t mu = mm.mu[M][a] % q;
            if (mu == 0) continue;
            const uint32_t L = logs[a % F];
            uint64_t p = mu;
            for (uint32_t j = 0; j <= n; ++j) {
                uint64_t& y = Bm[idx(j, a % ell, L)];
                y = (y + p) % q;
                p = static_cast<uint64_t>(static_cast<u128>(p) * (a % q) % q);
            }
        }
        for (uint64_t a = 1; a < F; ++a) {
            if (a % ell == 0) continue;
            Mat2 g;
            g.a = static_cast<int64_t>(F);
            g.b = -static_cast<int64_t>(a);
            g.c = 0;
            g.d = 1;
            const Vec v = rho.apply(g, PhiCusp(rho, P, static_cast<int64_t>(a), static_cast<int64_t>(F)));
            const uint32_t L = logs[a];
            for (uint32_t j = 0; j <= n; ++j) {
                uint64_t& y = Cv[idx(j, a % ell, L)];
                y = (y + v[n - j]) % q;
            }
        }
        // the O-valued sum over bins with the character omega^{xi}(rho) zeta^{xt L}
        auto binSum = [&](const std::vector<uint64_t>& bins, uint32_t j, int64_t xi, int64_t xt) {
            std::vector<u128> acc(O.order, 0);
            const uint32_t oi = ModL1(xi, ell);
            const int64_t G = G1;
            const int64_t tt = ((xt % G) + G) % G;
            for (uint64_t r = 1; r < ell; ++r)
                for (uint32_t L = 0; L < G1; ++L) {
                    const uint64_t b = bins[idx(j, r, L)];
                    if (b == 0) continue;
                    const uint32_t x = static_cast<uint32_t>((static_cast<u128>(tt) * L) % static_cast<u128>(G));
                    acc[x] = (acc[x] + static_cast<u128>(om[oi][r]) * b) % q;
                }
            std::vector<uint64_t> d(O.order);
            for (uint32_t x = 0; x < O.order; ++x) d[x] = static_cast<uint64_t>(acc[x]);
            return O.reduce(std::move(d));
        };
        const uint64_t iam = PowM(ia, m, q), iam1 = PowM(ia, m - 1, q);
        uint32_t tot = 0, bad = 0, parityBad = 0, zeroMatched = 0, feBad = 0, feTot = 0, negChi = 0, negChiOff = 0, negAlpha = 0, nz = 0;
        std::set<uint32_t> negChiJ;
        for (uint32_t i = 0; i + 1 < ell; ++i) {
            for (int64_t t : {int64_t(1), int64_t(2)}) {
                const int cm1 = (i % 2 == 0) ? 1 : -1;
                for (uint32_t j = 0; j <= n; ++j) {
                    const Elt rB = binSum(Bm, j, -static_cast<int64_t>(i), -t);           // Riemann sum of chibar x^j
                    const uint64_t sc = (j % 2 == 0) ? binom[j] : ModNeg(binom[j], q);
                    const Elt lhs = O.scal(sc, rB);
                    const Elt lam = binSum(Cv, j, -static_cast<int64_t>(i), -t);          // lambda_{chi,j}
                    const Elt rhs = O.scal(iam, lam);
                    ++tot;
                    const bool ok = O.eq_mod(lhs, rhs, lim);
                    bad += !ok;
                    const bool parityOn = cm1 * ((j % 2 == 0) ? 1 : -1) == sd.eps;
                    bool lz = true;
                    for (uint64_t x : lhs) lz = lz && x % lim == 0;
                    if (!parityOn) {
                        bool rz = true;
                        for (uint64_t x : lam) rz = rz && x % limR == 0;
                        parityBad += !(lz && rz);
                    } else if (lz) {
                        ++zeroMatched;
                    }
                    // FE: lambda_{chi,j} = -eps lambda_{chibar,n-j}
                    const Elt lamBar = binSum(Cv, n - j, static_cast<int64_t>(i), t);
                    ++feTot;
                    feBad += !O.eq_mod(lam, O.scal(sd.eps > 0 ? ModNeg(1, q) : 1 % q, lamBar), limR);
                    if (!lz) {
                        ++nz;
                        // NEG: chi for chibar on the left; alpha^{-(m-1)} on the right
                        const Elt lw = O.scal(sc, binSum(Bm, j, static_cast<int64_t>(i), t));
                        if (O.eq_mod(lw, rhs, lim)) {
                            ++negChi;
                            negChiJ.insert(j);
                            negChiOff += j != n / 2;
                        }
                        negAlpha += O.eq_mod(lhs, O.scal(iam1, lam), lim);
                    }
                }
            }
        }
        log.check("W2", bad == 0, tag + " " + sd.name + ": (-1)^j C(n,j) int chibar x^j dmu != alpha^{-m} lambda_{chi,j} at " + std::to_string(bad) + " / " + std::to_string(tot));
        log.check("W2", parityBad == 0, tag + " " + sd.name + ": parity-mismatched values not zero at " + std::to_string(parityBad));
        log.check("W2", feBad == 0, tag + " " + sd.name + ": lambda_{chi,j} != -eps lambda_{chibar,n-j} at " + std::to_string(feBad) + " / " + std::to_string(feTot));
        log.check("W2", negChiOff == 0, tag + " " + sd.name + ": NEG chi for chibar matches off the central point j = n/2 at " + std::to_string(negChiOff));
        log.check("W2", negAlpha == 0 && nz > 0, tag + " " + sd.name + ": NEG alpha^{-(m-1)} matches at " + std::to_string(negAlpha) + " / " + std::to_string(nz));
        // ---- W3
        const uint32_t e = O.e;
        uint32_t brTot = 0, brBad = 0, otherNZ = 0, lamTot = 0, lamBad = 0, unreadable = 0;
        std::ostringstream lamRows;
        for (uint32_t b = 0; b + 1 < ell; ++b) {
            const std::vector<uint64_t> g = mm.branch(b);
            const bool parityOn = ((b % 2 == 0) ? 1 : -1) == sd.eps;
            if (!parityOn) {
                for (uint64_t x : g) otherNZ += (x % limR) != 0;
                continue;
            }
            for (int64_t t : {int64_t(1), int64_t(2)})
                for (uint32_t j = 0; j <= 2 && j <= n; ++j) {
                    const Elt V = GroupRingAtWildPoint(O, g, t, j);
                    const Elt W = binSum(Bm, j, static_cast<int64_t>(b) - static_cast<int64_t>(j), t);
                    ++brTot;
                    brBad += !O.eq_mod(V, W, lim);   // the Riemann sum of x^j is correct mod ell^{min(M,R')} only
                }
            const IwasawaInvariants inv = mm.invariants(b);
            const Elt V0 = GroupRingAtWildPoint(O, g, 1, 0);
            Elt V0r(e);
            for (uint32_t x = 0; x < e; ++x) V0r[x] = V0[x] % limR;
            const uint32_t vp = O.vpi(V0r);
            uint32_t vg = sd.prec;
            for (uint64_t x : g)
                if (x % limR) vg = std::min(vg, ValU(x % limR, ell));
            if (inv.readable && inv.lambda < e && vg >= inv.mu) {
                ++lamTot;
                lamBad += vp != e * inv.mu + inv.lambda;
                lamRows << "(" << b << ":" << inv.mu << "," << inv.lambda << "," << vp << ")";
            } else {
                ++unreadable;
                lamRows << "(" << b << ":mu" << inv.mu << (inv.readable ? "" : "?") << ",v" << vp << ")";
            }
        }
        log.check("W3", brBad == 0 && brTot > 0, tag + " " + sd.name + ": branch value at zeta^t u^j != Riemann sum at " + std::to_string(brBad) + " / " + std::to_string(brTot));
        log.check("W3", otherNZ == 0, tag + " " + sd.name + ": branches of the other parity non-zero at " + std::to_string(otherNZ) + " entries");
        log.check("W3", lamBad == 0 && lamTot > 0, tag + " " + sd.name + ": v_pi(L_b(zeta - 1)) != e mu_b + lambda_b at " + std::to_string(lamBad) + " / " + std::to_string(lamTot) + " readable branches");
        std::ostringstream nj;
        for (uint32_t j : negChiJ) nj << j << ";";
        lines << "WILD_CUSP cell=" << Tag(n, ell) << " side=" << sd.name << " R'=" << sd.prec << " M=" << M << " m=" << m << " e=" << e
              << " eis=" << C.eisenstein_congruent << " v(alpha-1)=" << vAlpha << " W2=" << bad << "/" << tot << " parity=" << parityBad
              << " zero_on_parity=" << zeroMatched << " FE=" << feBad << "/" << feTot << " neg_chi=" << negChi << "(j:" << nj.str() << ") neg_alpha=" << negAlpha << "/" << nz
              << " W3=" << brBad << "/" << brTot << " lambda=" << lamBad << "/" << lamTot << " unreadable=" << unreadable << " " << lamRows.str() << "\n";
    }
    std::cout << lines.str() << "WILD_CUSP cell=" << Tag(n, ell) << " secs=" << (Now() - t0) << std::endl;
}

// ======================================================================================================== W4
void RunEis(uint32_t n, uint64_t ell, uint32_t M, const std::vector<uint32_t>& ms, Log& log) {
    const double t0 = Now();
    const uint32_t k = n + 2;
    const std::string tag = Tag(n, ell) + " M=" + std::to_string(M);
    const BernoulliOverKTable BT(ell, 2, k);
    const int32_t d = BT(k).v;
    const uint32_t E0 = 2 * M + 1 + ValU(k - 1, ell);
    const uint32_t R = std::min<uint32_t>(8, MaxK(ell) - E0);
    const eis::EisLadic L = eis::BuildEisLadic(n, ell, M, R, MeasureThreads());
    const std::vector<uint64_t> ellB = eis::EllBernoulliModEllPower(k, ell, L.K);
    const uint32_t pole1 = static_cast<uint32_t>(ell - 2), pole2 = static_cast<uint32_t>((k - 1) % (ell - 1));
    for (uint32_t m : ms) {
        const OArith O = MakeOArith(ell, m - 1, L.K);
        const std::vector<uint32_t> logs = LogTable(ell, m);
        const uint64_t F = Pow(ell, m), q = O.q;
        const uint64_t f2k = static_cast<uint64_t>(static_cast<u128>(F % q) * (F % q) % q * ((k - 1) % q) % q);
        const uint64_t lE = Pow(ell, L.E);
        uint32_t oddTot = 0, oddBad = 0, evenBad = 0;
        for (uint32_t i = 0; i + 1 < ell; ++i)
            for (int64_t t : {int64_t(1), int64_t(2)}) {
                const WildChar chi = MakeWildChar(ell, m, i, t, L.K, logs);
                const Elt V = CharSum(O, chi, L.mu);
                if (i % 2 == 0) {
                    evenBad += !O.is_zero(V);
                    continue;
                }
                const Elt X1 = WildBernoulliFB(O, chi, 1, ellB)[1];
                const Elt X2 = WildBernoulliFB(O, MakeWildChar(ell, m, -static_cast<int64_t>(i), -t, L.K, logs), k - 1, ellB)[k - 1];
                const Elt lhs = O.scal(f2k, V), rhs = O.scal(ModNeg(lE, q), O.mul(X1, X2));
                ++oddTot;
                oddBad += !O.eq_mod(lhs, rhs, q);
            }
        log.check("W4", oddBad == 0 && oddTot > 0, tag + " m=" + std::to_string(m) + ": sum chi mu^Eis != -B_{1,chi} B_{k-1,chibar}/(k-1) at " + std::to_string(oddBad) + " / " +
                                                      std::to_string(oddTot));
        log.check("W4", evenBad == 0, tag + " m=" + std::to_string(m) + ": even wild masses non-zero at " + std::to_string(evenBad));
        std::ostringstream pr;
        const uint32_t e = O.e;
        for (uint32_t a : {pole1, pole2}) {
            const std::vector<uint64_t> g = L.branch(a), pol = eis::PolarElement(L, a);
            for (int64_t t : {int64_t(1), int64_t(2)}) {
                const Elt Vb = GroupRingAtWildPoint(O, g, t, 0), Pb = GroupRingAtWildPoint(O, pol, t, 0);
                const int32_t vP = static_cast<int32_t>(O.vpi(Pb)) - static_cast<int32_t>(e * L.E);
                const int32_t vD = static_cast<int32_t>(O.vpi(O.sub(Vb, Pb))) - static_cast<int32_t>(e * L.E);
                const int32_t vV = static_cast<int32_t>(O.vpi(Vb)) - static_cast<int32_t>(e * L.E);
                log.check("W4", vP == static_cast<int32_t>(e) * d - 1, tag + " m=" + std::to_string(m) + ": pole branch " + std::to_string(a) + " polar v_pi " +
                                                                          std::to_string(vP) + " != e d - 1 = " + std::to_string(static_cast<int32_t>(e) * d - 1));
                log.check("W4", vD >= 0, tag + " m=" + std::to_string(m) + ": pole branch " + std::to_string(a) + " value - polar has v_pi " + std::to_string(vD));
                if (t == 1) pr << a << ":(" << vV << "," << vP << "," << vD << ")";
            }
        }
        std::cout << "WILD_EIS cell=" << Tag(n, ell) << " M=" << M << " m=" << m << " d=" << d << " e=" << e << " K=" << L.K << " E=" << L.E << " odd=" << oddBad << "/" << oddTot
                  << " even_nonzero=" << evenBad << " pole(v_value,v_polar,v_diff)=" << pr.str() << " secs=" << (Now() - t0) << std::endl;
    }
}


// ======================================================================================================== W5 (complex)
constexpr unsigned kPrec = 256;

nf::MpComplex CZero() { return nf::MpComplex{nf::mp_from_i64(0), nf::mp_from_i64(0)}; }
nf::MpComplex CReal(const nf::MpReal& x) { return nf::MpComplex{x, nf::mp_from_i64(0)}; }
nf::MpComplex CConj(const nf::MpComplex& a) { return nf::MpComplex{a.re, nf::mp_neg(a.im)}; }
nf::MpComplex CMul(const nf::MpComplex& a, const nf::MpComplex& b) { return nf::mp_cmul(a, b, kPrec); }
nf::MpComplex CAdd(const nf::MpComplex& a, const nf::MpComplex& b) { return nf::mp_cadd(a, b, kPrec); }
nf::MpComplex CSub(const nf::MpComplex& a, const nf::MpComplex& b) { return nf::mp_csub(a, b, kPrec); }
nf::MpComplex CDiv(const nf::MpComplex& a, const nf::MpComplex& b) {
    const nf::MpReal nb = nf::mp_cnorm(b, kPrec);
    const nf::MpComplex x = CMul(a, CConj(b));
    return nf::MpComplex{nf::mp_div(x.re, nb, kPrec), nf::mp_div(x.im, nb, kPrec)};
}
// log2 |x - y| / |y|
double RelDevLog2(const nf::MpComplex& x, const nf::MpComplex& y) {
    const nf::MpReal a = nf::mp_cnorm(CSub(x, y), kPrec), b = nf::mp_cnorm(y, kPrec);
    if (nf::mp_is_zero(a)) return -1e9;
    return 0.5 * (nf::mp_log2_abs(a) - nf::mp_log2_abs(b));
}
double AbsLog2(const nf::MpComplex& x) {
    const nf::MpReal a = nf::mp_cnorm(x, kPrec);
    return nf::mp_is_zero(a) ? -1e9 : 0.5 * nf::mp_log2_abs(a);
}

// (-i)^{j+1}
nf::MpComplex MinusIPow(uint32_t j) {
    switch (j % 4) {
    case 0: return nf::MpComplex{nf::mp_from_i64(0), nf::mp_from_i64(-1)};
    case 1: return nf::MpComplex{nf::mp_from_i64(-1), nf::mp_from_i64(0)};
    case 2: return nf::MpComplex{nf::mp_from_i64(0), nf::mp_from_i64(1)};
    default: return nf::MpComplex{nf::mp_from_i64(1), nf::mp_from_i64(0)};
    }
}

// Residue-class sums of the two halves of I(s) at the splits c = cn / (cd M):
//   A[ci][s][r] = sum_{m == r} a_m Gamma(s, 2 pi m c) / (2 pi m)^s,  B[ci][s][r] = sum_{m == r} a_m Gamma(k-s, 2 pi m / (N c)) / (2 pi m)^{k-s}.
struct SeriesSums {
    std::vector<std::vector<std::vector<nf::MpReal>>> A, B;
};
SeriesSums BuildSums(const std::vector<nf::BigInt>& f, uint32_t k, uint64_t M, uint64_t ellSkip, uint32_t mmax, const nf::MpContext& C) {
    const int64_t splits[2][2] = {{1, 1}, {137, 100}};
    SeriesSums S;
    S.A.assign(2, std::vector<std::vector<nf::MpReal>>(k, std::vector<nf::MpReal>(M, nf::mp_from_i64(0))));
    S.B = S.A;
    const nf::MpReal twopi = nf::mp_ldexp(C.pi(), 1);
    std::vector<nf::MpReal> fact(k + 1, nf::mp_from_i64(1));
    for (uint32_t s = 1; s <= k; ++s) fact[s] = nf::mp_mul_i64(fact[s - 1], s, kPrec);
    auto gammas = [&](const nf::MpReal& x) {
        // G[s] = Gamma(s, x) = (s-1)! e^{-x} sum_{r < s} x^r / r!, s = 1 .. k-1
        std::vector<nf::MpReal> G(k, nf::mp_from_i64(0));
        const nf::MpReal ex = C.exp(nf::mp_neg(x));
        nf::MpReal term = nf::mp_from_i64(1), ps = nf::mp_from_i64(0);
        for (uint32_t s = 1; s < k; ++s) {
            ps = nf::mp_add(ps, term, kPrec);
            G[s] = nf::mp_mul(nf::mp_mul(fact[s - 1], ex, kPrec), ps, kPrec);
            term = nf::mp_div_u32(nf::mp_mul(term, x, kPrec), s, kPrec);
        }
        return G;
    };
    for (uint32_t m = 1; m <= mmax; ++m) {
        if (f[m].is_zero() || (ellSkip && m % ellSkip == 0)) continue;
        const nf::MpReal am = nf::mp_from_int(f[m]);
        const nf::MpReal tw = nf::mp_mul_i64(twopi, m, kPrec);
        const nf::MpReal itw = nf::mp_div(nf::mp_from_i64(1), tw, kPrec);
        std::vector<nf::MpReal> ip(k, nf::mp_from_i64(1));
        for (uint32_t s = 1; s < k; ++s) ip[s] = nf::mp_mul(ip[s - 1], itw, kPrec);
        const uint64_t r = m % M;
        for (int ci = 0; ci < 2; ++ci) {
            const int64_t cn = splits[ci][0], cd = splits[ci][1];
            const nf::MpReal x1 = nf::mp_div_u32(nf::mp_mul_i64(tw, cn, kPrec), static_cast<uint32_t>(cd * static_cast<int64_t>(M)), kPrec);
            const nf::MpReal x2 = nf::mp_div_u32(nf::mp_mul_i64(tw, cd, kPrec), static_cast<uint32_t>(cn * static_cast<int64_t>(M)), kPrec);
            const std::vector<nf::MpReal> G1 = gammas(x1), G2 = gammas(x2);
            for (uint32_t s = 1; s < k; ++s) {
                S.A[ci][s][r] = nf::mp_add(S.A[ci][s][r], nf::mp_mul(am, nf::mp_mul(G1[s], ip[s], kPrec), kPrec), kPrec);
                S.B[ci][s][r] = nf::mp_add(S.B[ci][s][r], nf::mp_mul(am, nf::mp_mul(G2[k - s], ip[k - s], kPrec), kPrec), kPrec);
            }
        }
    }
    return S;
}

uint32_t TermCount(uint32_t k, uint64_t M) {
    // e^{-2 pi m / (1.37 M)} (2 pi m)^k below 2^{-(kPrec + 40)}
    const double need = (kPrec + 40) * 0.693147 + 3.0 * k;
    return static_cast<uint32_t>(std::ceil(need * 1.37 * static_cast<double>(M) / (2.0 * M_PI))) + 20;
}

struct ComplexCell {
    uint32_t k;
    uint64_t ell;
    uint32_t m;
};

void RunComplex(const ComplexCell& cc, const nf::MpContext& C, Log& log) {
    const double t0 = Now();
    const uint32_t k = cc.k, n = k - 2;
    const uint64_t ell = cc.ell, M = Pow(ell, cc.m);
    const std::string tag = "k=" + std::to_string(k) + " M=" + std::to_string(M);
    const uint32_t mmax = TermCount(k, M);
    const std::vector<nf::BigInt> f = CuspQExp(k, mmax);
    const ExactEigensymbols ES = ComputeExactEigensymbols(k, f[2]);
    const bool esOk = ES.kernel_dim == 2 && !ES.Pplus.empty() && !ES.Pminus.empty();
    // sanity mod p = 2^61 - 1: both in W and T_3-eigen with a_3
    bool modpOk = esOk;
    if (esOk) {
        const uint64_t p = (uint64_t(1) << 61) - 1;
        const Ring Rp = MakeRing(p, 1);
        const SymRho rp(n, Rp);
        const nf::BigInt bp = nf::BigInt::from_u64(p);
        const uint64_t a3 = static_cast<uint64_t>(f[3].floor_mod(bp).to_int64());
        for (const auto* P : {&ES.Pplus, &ES.Pminus}) {
            Vec v(n + 1);
            for (uint32_t t = 0; t <= n; ++t) v[t] = static_cast<uint64_t>((*P)[t].floor_mod(bp).to_int64());
            const Vec T3 = HeckeTp(rp, v, 3);
            bool eig = InW(rp, v);
            for (uint32_t t = 0; t <= n; ++t) eig = eig && T3[t] == Rp.mul(a3, v[t]);
            modpOk = modpOk && eig;
        }
    }
    log.check("W5", esOk && modpOk, tag + ": exact eigensymbols (kernel dim " + std::to_string(ES.kernel_dim) + ", in W and T_3-eigen mod 2^61-1: " + (modpOk ? "yes" : "no") + ")");
    if (!esOk) return;
    const double minus30 = -30.0 * std::log2(10.0), neg01 = std::log2(0.1);
    const nf::MpReal twopi = nf::mp_ldexp(C.pi(), 1);
    const double ik = (k / 2) % 2 == 0 ? 1.0 : -1.0;
    // ---- conductor 1: Omega^{+-}_ref
    std::map<int, nf::MpComplex> ref;
    double refDev = -1e9, refSplit = -1e9;
    {
        const uint32_t m1 = TermCount(k, 1);
        const SeriesSums S1 = BuildSums(f, k, 1, 0, m1, C);
        for (uint32_t j = 0; j <= n; ++j) {
            const uint32_t s = j + 1;
            nf::MpReal I[2];
            for (int ci = 0; ci < 2; ++ci) I[ci] = nf::mp_add(S1.A[ci][s][0], nf::mp_mul_i64(S1.B[ci][s][0], static_cast<int64_t>(ik), kPrec), kPrec);
            refSplit = std::max(refSplit, RelDevLog2(CReal(I[0]), CReal(I[1])));
            const int sg = (j % 2 == 0) ? 1 : -1;
            const nf::BigInt& lam = (sg > 0 ? ES.Pplus : ES.Pminus)[n - j];
            if (lam.is_zero()) continue;
            const nf::MpReal& num = I[0];
            nf::BigInt bin(1);
            for (uint32_t x = 0; x < j; ++x) bin = bin * nf::BigInt(static_cast<int64_t>(n - x)) / nf::BigInt(static_cast<int64_t>(x + 1));
            const nf::MpComplex Om = CDiv(CReal(nf::mp_mul(num, nf::mp_from_int(bin), kPrec)), CMul(MinusIPow(j), CReal(nf::mp_from_int(lam))));
            auto it = ref.find(sg);
            if (it == ref.end()) ref[sg] = Om;
            else refDev = std::max(refDev, RelDevLog2(Om, it->second));
        }
    }
    log.check("W5", ref.size() == 2 && refDev <= minus30 && refSplit <= minus30,
              tag + ": conductor-1 Omega^{+-} not constant (log2 dev " + std::to_string(refDev) + ", split " + std::to_string(refSplit) + ")");
    // ---- conductor M
    const SeriesSums S = BuildSums(f, k, M, ell, mmax, C);
    const uint64_t D = (ell - 1) * Pow(ell, cc.m - 1), G1 = Pow(ell, cc.m - 1);
    std::vector<nf::MpComplex> z(D), ea(M);
    for (uint64_t r = 0; r < D; ++r) z[r] = C.expi(nf::mp_div_u32(nf::mp_mul_i64(twopi, static_cast<int64_t>(r), kPrec), static_cast<uint32_t>(D), kPrec));
    for (uint64_t a = 0; a < M; ++a) ea[a] = C.expi(nf::mp_div_u32(nf::mp_mul_i64(twopi, static_cast<int64_t>(a), kPrec), static_cast<uint32_t>(M), kPrec));
    const std::vector<uint32_t> logs = LogTable(ell, cc.m);
    const uint64_t g = PrimitiveRootModPrime(ell);
    std::vector<uint32_t> ind(ell, 0);
    {
        uint64_t y = 1;
        for (uint32_t e = 0; e + 1 < ell; ++e) {
            ind[y] = e;
            y = y * g % ell;
        }
    }
    // exact twist vectors
    std::vector<std::vector<nf::BigInt>> Cp(M), Cm(M);
    for (uint64_t a = 1; a < M; ++a) {
        if (a % ell == 0) continue;
        Cp[a] = ExactTwistVector(ES.Pplus, n, static_cast<int64_t>(a), static_cast<int64_t>(M));
        Cm[a] = ExactTwistVector(ES.Pminus, n, static_cast<int64_t>(a), static_cast<int64_t>(M));
    }
    // N^{k/2 - s} i^k = ik M^{k - 2s}
    std::vector<nf::MpReal> nfac(k);
    for (uint32_t s = 1; s < k; ++s) {
        const int32_t ex = static_cast<int32_t>(k) - 2 * static_cast<int32_t>(s);
        const nf::MpReal pw = nf::mp_from_int(nf::BigInt::pow(nf::BigInt::from_u64(M), static_cast<uint64_t>(std::abs(ex))));
        nfac[s] = ex >= 0 ? pw : nf::mp_div(nf::mp_from_i64(1), pw, kPrec);
        if (ik < 0) nfac[s] = nf::mp_neg(nfac[s]);
    }
    std::vector<nf::BigInt> binom(n + 1, nf::BigInt(1));
    for (uint32_t j = 1; j <= n; ++j) binom[j] = binom[j - 1] * nf::BigInt(static_cast<int64_t>(n - j + 1)) / nf::BigInt(static_cast<int64_t>(j));
    uint32_t nchars = 0, nvals = 0, zeroBoth = 0, zeroBad = 0;
    double devMax[2] = {-1e9, -1e9}, splitMax = -1e9, wDev = -1e9, wAbs = -1e9, otherMax = -1e9, negTau = -1e9;
    std::vector<std::pair<uint32_t, int64_t>> negCondChars;
    std::map<std::pair<uint32_t, int64_t>, std::pair<std::vector<nf::MpComplex>, nf::MpComplex>> keep;   // (I1 per j, tau) for the conductor control
    for (uint32_t i = 0; i + 1 < ell; ++i) {
        for (int64_t t = 1; t < static_cast<int64_t>(G1); ++t) {
            if (t % static_cast<int64_t>(ell) == 0) continue;
            ++nchars;
            std::vector<uint32_t> ex(M, 0);
            for (uint64_t a = 1; a < M; ++a) {
                if (a % ell == 0) continue;
                ex[a] = static_cast<uint32_t>((static_cast<u128>(i) * ind[a % ell] * G1 + static_cast<u128>(t) * logs[a] * (ell - 1)) % D);
            }
            const int cm1 = (i % 2 == 0) ? 1 : -1;
            nf::MpComplex tau = CZero(), taub = CZero();
            for (uint64_t a = 1; a < M; ++a) {
                if (a % ell == 0) continue;
                tau = CAdd(tau, CMul(z[ex[a]], ea[a]));
                taub = CAdd(taub, CMul(CConj(z[ex[a]]), ea[a]));
            }
            auto AB = [&](int ci, uint32_t s, nf::MpComplex& A, nf::MpComplex& B) {
                A = CZero();
                B = CZero();
                for (uint64_t r = 1; r < M; ++r) {
                    if (r % ell == 0) continue;
                    if (!nf::mp_is_zero(S.A[ci][s][r])) A = CAdd(A, nf::mp_cmul_real(z[ex[r]], S.A[ci][s][r], kPrec));
                    if (!nf::mp_is_zero(S.B[ci][s][r])) B = CAdd(B, nf::mp_cmul_real(CConj(z[ex[r]]), S.B[ci][s][r], kPrec));
                }
                B = nf::mp_cmul_real(B, nfac[s], kPrec);
            };
            nf::MpComplex A1, B1, A2, B2;
            AB(0, 1, A1, B1);
            AB(1, 1, A2, B2);
            const nf::MpComplex w = CDiv(CSub(A1, A2), CSub(B2, B1));
            const nf::MpComplex wPred = CDiv(CMul(tau, tau), CReal(nf::mp_from_i64(static_cast<int64_t>(M))));   // P4_spec v0.2, deviation 1
            wDev = std::max(wDev, RelDevLog2(w, wPred));
            wAbs = std::max(wAbs, std::fabs(nf::mp_log2_abs(nf::mp_cnorm(w, kPrec))));
            std::vector<nf::MpComplex> I1s(n + 1);
            for (uint32_t j = 0; j <= n; ++j) {
                const uint32_t s = j + 1;
                AB(0, s, A1, B1);
                AB(1, s, A2, B2);
                const nf::MpComplex I1 = CAdd(A1, CMul(w, B1)), I2 = CAdd(A2, CMul(w, B2));
                I1s[j] = I1;
                splitMax = std::max(splitMax, RelDevLog2(I2, I1));
                const int sg = cm1 * ((j % 2 == 0) ? 1 : -1);
                // lambda on both signs: bins per exponent, exactly
                nf::MpComplex lam[2];
                double sumLog[2];
                bool exactZero[2];
                for (int side = 0; side < 2; ++side) {
                    const std::vector<std::vector<nf::BigInt>>& Cv = side == 0 ? Cp : Cm;
                    std::vector<nf::BigInt> bins(D, nf::BigInt(0));
                    for (uint64_t a = 1; a < M; ++a)
                        if (a % ell) bins[ex[a]] += Cv[a][n - j];
                    nf::MpComplex acc = CZero();
                    nf::MpReal sa = nf::mp_from_i64(0);
                    exactZero[side] = true;
                    for (uint64_t r = 0; r < D; ++r) {
                        if (bins[r].is_zero()) continue;
                        exactZero[side] = false;
                        const nf::MpReal br = nf::mp_from_int(bins[r]);
                        acc = CAdd(acc, nf::mp_cmul_real(CConj(z[r]), br, kPrec));
                        sa = nf::mp_add(sa, nf::mp_abs(br), kPrec);
                    }
                    lam[side] = acc;
                    sumLog[side] = exactZero[side] ? -1e9 : nf::mp_log2_abs(sa);
                }
                const int on = sg > 0 ? 0 : 1, off = 1 - on;
                if (!exactZero[off]) otherMax = std::max(otherMax, AbsLog2(lam[off]) - sumLog[off]);
                const bool lamZero = exactZero[on] || AbsLog2(lam[on]) - sumLog[on] < -150.0;
                if (lamZero) {
                    ++zeroBoth;
                    zeroBad += AbsLog2(I1) > -100.0;   // lambda = 0 must mean L = 0
                    continue;
                }
                ++nvals;
                const nf::MpReal scale = nf::mp_from_int(binom[j] * nf::BigInt::pow(nf::BigInt::from_u64(M), j + 1));
                const nf::MpComplex den = nf::mp_cmul_real(CMul(MinusIPow(j), CMul(tau, lam[on])), nf::mp_from_i64(cm1), kPrec);
                const nf::MpComplex Om = CDiv(nf::mp_cmul_real(I1, scale, kPrec), den);
                const int si = sg > 0 ? 0 : 1;
                devMax[si] = std::max(devMax[si], RelDevLog2(Om, ref[sg]));
                negTau = std::max(negTau, RelDevLog2(CDiv(CMul(Om, tau), taub), ref[sg]));
            }
            if ((i == 0 || i == 1) && t == 1) keep[{i, t}] = {I1s, tau};
        }
    }
    // NEG conductor ell M: lambda at modulus ell^{m+1} (the induced character), M' = ell M in the formula
    double negCond = -1e9;
    {
        const uint64_t M2 = M * ell;
        std::vector<std::vector<nf::BigInt>> Cp2(M2), Cm2(M2);
        for (uint64_t a = 1; a < M2; ++a) {
            if (a % ell == 0) continue;
            Cp2[a] = ExactTwistVector(ES.Pplus, n, static_cast<int64_t>(a), static_cast<int64_t>(M2));
            Cm2[a] = ExactTwistVector(ES.Pminus, n, static_cast<int64_t>(a), static_cast<int64_t>(M2));
        }
        for (const auto& kv : keep) {
            const uint32_t i = kv.first.first;
            const int64_t t = kv.first.second;
            const int cm1 = (i % 2 == 0) ? 1 : -1;
            for (uint32_t j = 0; j <= n; ++j) {
                const int sg = cm1 * ((j % 2 == 0) ? 1 : -1);
                const std::vector<std::vector<nf::BigInt>>& Cv = sg > 0 ? Cp2 : Cm2;
                std::vector<nf::BigInt> bins(D, nf::BigInt(0));
                for (uint64_t a = 1; a < M2; ++a) {
                    if (a % ell == 0) continue;
                    const uint64_t r = a % M;
                    const uint32_t e = static_cast<uint32_t>((static_cast<u128>(i) * ind[r % ell] * G1 + static_cast<u128>(t) * logs[r] * (ell - 1)) % D);
                    bins[e] += Cv[a][n - j];
                }
                nf::MpComplex lam = CZero();
                for (uint64_t r = 0; r < D; ++r)
                    if (!bins[r].is_zero()) lam = CAdd(lam, nf::mp_cmul_real(CConj(z[r]), nf::mp_from_int(bins[r]), kPrec));
                if (AbsLog2(lam) < -100.0) continue;
                const nf::MpReal scale = nf::mp_from_int(binom[j] * nf::BigInt::pow(nf::BigInt::from_u64(M2), j + 1));
                const nf::MpComplex den = nf::mp_cmul_real(CMul(MinusIPow(j), CMul(kv.second.second, lam)), nf::mp_from_i64(cm1), kPrec);
                negCond = std::max(negCond, RelDevLog2(CDiv(nf::mp_cmul_real(kv.second.first[j], scale, kPrec), den), ref[sg]));
            }
        }
    }
    const double dev = std::max(devMax[0], devMax[1]);
    log.check("W5", nvals > 0 && dev <= minus30, tag + ": Omega_{chi,j} differs from the conductor-1 Omega^{+-} (log2 rel. dev " + std::to_string(dev) + ")");
    log.check("W5", splitMax <= minus30, tag + ": split dependence log2 " + std::to_string(splitMax));
    log.check("W5", wDev <= minus30 && wAbs <= 1e-20, tag + ": root number != tau(chi)^2 / M (log2 dev " + std::to_string(wDev) + ")");
    log.check("W5", otherMax <= -150.0 && zeroBad == 0, tag + ": lambda of the wrong sign not zero (log2 " + std::to_string(otherMax) + "), lambda = 0 with L != 0 at " + std::to_string(zeroBad));
    log.check("W5", negTau >= neg01, tag + ": NEG tau(chibar) agrees (log2 rel. dev " + std::to_string(negTau) + ")");
    log.check("W5", negCond >= neg01, tag + ": NEG conductor ell M agrees (log2 rel. dev " + std::to_string(negCond) + ")");
    auto d10 = [](double l2) { return l2 < -1e8 ? std::string("-inf") : std::to_string(l2 * 0.30103).substr(0, 7); };
    std::cout << "WILD_COMPLEX " << tag << " terms=" << mmax << " chars=" << nchars << " values=" << nvals << " zero=" << zeroBoth << " log10dev+=" << d10(devMax[0])
              << " log10dev-=" << d10(devMax[1]) << " log10split=" << d10(splitMax) << " log10|w-tau^2/M|=" << d10(wDev) << " log10other=" << d10(otherMax)
              << " neg_tau(chibar)_log10dev=" << d10(negTau) << " neg_cond_log10dev=" << d10(negCond) << " ref_log10dev=" << d10(refDev) << " secs=" << (Now() - t0) << std::endl;
}

// ======================================================================================================== W6
struct PoleSpec {
    uint32_t n;
    uint64_t ell;
    uint32_t R, M, m;
    int64_t t;
    std::map<uint64_t, uint64_t> sep;
    bool cross = false;   // (P4x) also run the 128-bit route and require route agreement (P4_spec Sec. 10)
};

// (P4x) one point's elements of the 64-bit route, kept for the route agreement
struct PolePoint {
    Elt X1, X2, prod, pol;
};

void RunPole(const PoleSpec& ps, Log& log) {
    const double t0 = Now();
    const uint32_t n = ps.n, k = n + 2, m = ps.m;
    const uint64_t ell = ps.ell;
    const std::string tag = Tag(n, ell) + " R=" + std::to_string(ps.R) + " M=" + std::to_string(ps.M) + " m=" + std::to_string(m) + " t=" + std::to_string(ps.t);
    const SymbolCell C = RunSymbolCell(n, ell, ps.R, ps.sep, false);
    if (!C.simple || !C.eisenstein_congruent) {
        log.check("W6", false, tag + ": the symbol cell did not run: " + C.refusal);
        return;
    }
    const Ring& Rg = C.ring;
    const SymRho rho(n, Rg);
    const uint64_t a_ell = C.a_p.at(ell);
    const uint64_t alpha = UnitRoot(Rg, a_ell, k);
    const ModularSymbolMeasure mm = BuildMeasure(rho, C.r_minus, a_ell, alpha, ps.M, true);
    int32_t vmin = 0;
    const Vec PE = EisensteinSymbol(n, Rg, &vmin);
    uint32_t t0u = 0;
    while (t0u <= n && !Rg.unit(C.r_minus[t0u])) ++t0u;
    ENGINE_CHECK_MSG(t0u <= n && Rg.unit(PE[t0u]), "P4 W6: P^Eis is not a unit at r^-'s first unit coordinate");
    const uint64_t s = Rg.mul(C.r_minus[t0u], Rg.inv(PE[t0u]));
    const BernoulliOverKTable BTd(ell, 4, k);
    const int32_t d = BTd(k).v;
    ENGINE_CHECK_MSG(d >= 1, "P4 W6: an Eisenstein-congruent cell has d >= 1");
    const uint64_t F = Pow(ell, m), L = Pow(ell, m - 1), u = 1 + ell;
    const uint32_t pole1 = static_cast<uint32_t>(ell - 2), pole2 = static_cast<uint32_t>((k - 1) % (ell - 1));
    const uint64_t pd = Pow(ell, static_cast<uint32_t>(d));
    const OArith OR = MakeOArith(ell, m - 1, ps.R);
    // (P4x) the precision the scaled comparison needs: v(D) + d, D = F^2 (j+1)(n-j+1)(u^{rL} - 1), v(u^{rL} - 1) = m + v(r)
    uint32_t Pneed = 0;
    for (uint32_t a : {pole2, pole1})
        for (uint32_t j = 0; j <= n; ++j) {
            const uint64_t r = a == pole1 ? j + 1 : k - 1 - j;
            Pneed = std::max(Pneed, ValU(F * F * static_cast<uint64_t>(j + 1) * static_cast<uint64_t>(n - j + 1), ell) + m + ValU(r, ell) + static_cast<uint32_t>(d));
        }
    const uint32_t P64 = MaxK(ell);
    const bool run64 = Pneed <= P64, run128 = !run64 || ps.cross;
    const uint32_t expect = 2 * (n + 1);
    std::vector<PolePoint> rec;   // the 64-bit route's points, for the route agreement
    uint32_t cnt64[5] = {0, 0, 0, 0, 0};
    if (run64) {
        const uint32_t P = P64;
        const OArith O = MakeOArith(ell, m - 1, P);
        const uint64_t q = O.q;
        const std::vector<uint32_t> logs = LogTable(ell, m);
        const std::vector<uint64_t> ellB = eis::EllBernoulliModEllPower(n + 1, ell, P);
        const eis::Lad Res = eis::Residue(k, ell, P);
        ENGINE_CHECK_MSG(!Res.zero && Res.v == d, "P4 W6: v(Res) != d");
        const uint64_t resVal = static_cast<uint64_t>(static_cast<u128>(Res.unit % q) * Pow(ell, static_cast<uint32_t>(Res.v)) % q);
        std::map<std::pair<uint32_t, int64_t>, std::vector<Elt>> cache;
        auto FB = [&](int64_t i, int64_t t) -> const std::vector<Elt>& {
            const auto key = std::make_pair(ModL1(i, ell), t);
            auto it = cache.find(key);
            if (it == cache.end()) it = cache.emplace(key, WildBernoulliFB(O, MakeWildChar(ell, m, i, t, P, logs), n + 1, ellB)).first;
            return it->second;
        };
        uint32_t tot = 0, bad = 0, nonint = 0, negNone = 0, negSign = 0, negScale = 0;
        const std::vector<uint64_t> ones(L, 1);
        for (uint32_t a : {pole2, pole1}) {
            const std::vector<uint64_t> g = mm.branch(a);
            for (uint32_t j = 0; j <= n; ++j) {
                const uint64_t r = a == pole1 ? j + 1 : k - 1 - j;
                const int64_t sgn = a == pole1 ? ps.t : -ps.t;
                const uint64_t scalInt = F * F * static_cast<uint64_t>(j + 1) * static_cast<uint64_t>(n - j + 1);
                const uint32_t vs = ValU(scalInt, ell);
                const uint64_t c = (PowM(u, r * L, q) + q - 1) % q;
                const uint32_t vc = ValU(c, ell);
                const uint32_t vD = vs + vc;
                ENGINE_CHECK_MSG(vc == m + ValU(r, ell) && vD + static_cast<uint32_t>(d) <= P, "P4 W6: precision or v(c) out of range");
                const Elt& X1 = FB(static_cast<int64_t>(a) - static_cast<int64_t>(j), ps.t)[j + 1];
                const Elt& X2 = FB(static_cast<int64_t>(j) - static_cast<int64_t>(a), -ps.t)[n - j + 1];
                const Elt prod = O.scal(c, O.mul(X1, X2));
                const Elt Sy = GroupRingAtWildPoint(O, ones, sgn, r);
                const Elt pol = O.scal(static_cast<uint64_t>(static_cast<u128>(resVal) * (scalInt % q) % q), Sy);
                if (ps.cross) rec.push_back({X1, X2, prod, pol});
                const uint64_t lv = Pow(ell, vD);
                const uint64_t us = (scalInt / Pow(ell, vs)) % pd, uc = (c / Pow(ell, vc)) % pd;
                const uint64_t iD = InvModU64(static_cast<uint64_t>(static_cast<u128>(us) * uc % pd), pd);
                const Elt V = GroupRingAtWildPoint(OR, g, ps.t, j);
                auto rhsMatches = [&](int64_t scale, bool withPolar, bool& integral) {
                    const uint64_t sc = scale >= 0 ? static_cast<uint64_t>(scale) % q : ModNeg(static_cast<uint64_t>(-scale), q);
                    integral = true;
                    bool same = true;
                    for (uint32_t x = 0; x < O.e; ++x) {
                        const uint64_t zx = withPolar ? (prod[x] + O.mulq(sc, pol[x])) % q : prod[x];
                        if (zx % lv) {
                            integral = false;
                            continue;
                        }
                        const uint64_t y = static_cast<uint64_t>(static_cast<u128>((zx / lv) % pd) * iD % pd * (s % pd) % pd);
                        same = same && ModNeg(y, pd) == V[x] % pd;
                    }
                    return integral && same;
                };
                bool in1 = true, in2 = true, in3 = true, in4 = true;
                ++tot;
                const bool ok = rhsMatches(1, true, in1);
                bad += !ok;
                nonint += !in1;
                negNone += !rhsMatches(1, false, in2);
                negSign += !rhsMatches(-1, true, in3);
                negScale += !rhsMatches(static_cast<int64_t>(1 + ell), true, in4);
            }
        }
        log.check("W6", tot == expect, tag + ": " + std::to_string(tot) + " wild critical points on the pole branches (predicted " + std::to_string(expect) + ")");
        log.check("W6", bad == 0, tag + ": -s (F1 F2 + Res/(zeta u^r - 1)) fails mod ell^d at " + std::to_string(bad) + " / " + std::to_string(tot) + " (non-integral " +
                                      std::to_string(nonint) + ")");
        log.check("W6", negNone == tot, tag + ": NEG no polar term fails at " + std::to_string(negNone) + " / " + std::to_string(tot) + " (all predicted)");
        log.check("W6", negSign == tot, tag + ": NEG -Res fails at " + std::to_string(negSign) + " / " + std::to_string(tot) + " (all predicted)");
        log.check("W6", negScale == 0, tag + ": Res (1+ell) fails at " + std::to_string(negScale) + " (predicted 0: invisible at wild points)");
        std::cout << "WILD_POLE cell=" << Tag(n, ell) << " d=" << d << " M=" << ps.M << " m=" << m << " t=" << ps.t << " s=" << s << " points=" << tot << " fail=" << bad
                  << " neg_nopolar=" << negNone << " neg_minusRes=" << negSign << " Res(1+ell)=" << negScale << " secs=" << (Now() - t0) << std::endl;
        cnt64[0] = tot;
        cnt64[1] = bad;
        cnt64[2] = negNone;
        cnt64[3] = negSign;
        cnt64[4] = negScale;
    }
    if (!run128) return;
    // ---- (P4x) the 128-bit route: the same identity in O / ell^P, P = max(Pneed, MaxK) + 2 (above 2^62 whenever it runs)
    const double t1 = Now();
    const uint32_t P = std::max(Pneed, P64) + 2;
    const OArith128 O = MakeOArith128(ell, m - 1, P);
    const u128 q = O.q;
    const std::vector<uint32_t> logs = LogTable(ell, m);
    WildBernoulliFB128 FB(O, n + 1, logs);
    const Lad128 Res = Residue128(k, O);
    ENGINE_CHECK_MSG(Res.v == d, "P4 W6 (u128): v(Res) != d");
    const u128 resVal = O.ctx.mul(Res.unit, O.ctx.ell_pow(static_cast<uint32_t>(Res.v)));
    // route agreement, inputs: omega^x, ell B_s, Res modulo ell^{P64} (P < P128 always)
    uint32_t inBad = 0, ptBad = 0, ptCmp = 0;
    const u128 q64 = static_cast<u128>(Pow(ell, P64));
    if (ps.cross) {
        const std::vector<std::vector<u128>> om = OmegaTable128(O);
        for (uint32_t x = 0; x + 1 < ell; ++x)
            for (uint64_t r = 1; r < ell; ++r) inBad += om[x][r] % q64 != TeichmullerPowMod(r, x, ell, P64);
        const std::vector<uint64_t> ellB64 = eis::EllBernoulliModEllPower(n + 1, ell, P64);
        for (uint32_t sI = 0; sI < FB.ell_bernoulli().size(); ++sI) inBad += FB.ell_bernoulli()[sI] % q64 != ellB64[sI];
        const eis::Lad Res64 = eis::Residue(k, ell, P64);
        inBad += Res64.v != Res.v || Res.unit % q64 != Res64.unit % static_cast<uint64_t>(q64);
    }
    auto modq64 = [&](const OArith128::Elt& a, const Elt& b) {
        for (uint32_t x = 0; x < O.e; ++x)
            if (a[x] % q64 != static_cast<u128>(b[x] % static_cast<uint64_t>(q64))) return false;
        return true;
    };
    uint32_t tot = 0, bad = 0, nonint = 0, negNone = 0, negSign = 0, negScale = 0;
    const std::vector<u128> ones(L, 1);
    for (uint32_t a : {pole2, pole1}) {
        const std::vector<uint64_t> g = mm.branch(a);
        for (uint32_t j = 0; j <= n; ++j) {
            const uint64_t r = a == pole1 ? j + 1 : k - 1 - j;
            const int64_t sgn = a == pole1 ? ps.t : -ps.t;
            const uint64_t scalInt = F * F * static_cast<uint64_t>(j + 1) * static_cast<uint64_t>(n - j + 1);
            const uint32_t vs = ValU(scalInt, ell);
            const u128 c = O.ctx.sub(O.pow(u, static_cast<u128>(r) * L), 1);
            const uint32_t vc = O.ctx.valuation(c);
            const uint32_t vD = vs + vc;
            ENGINE_CHECK_MSG(vc == m + ValU(r, ell) && vD + static_cast<uint32_t>(d) <= P, "P4 W6 (u128): precision or v(c) out of range");
            const OArith128::Elt X1 = FB.at(static_cast<int64_t>(a) - static_cast<int64_t>(j), ps.t, j + 1);
            const OArith128::Elt X2 = FB.at(static_cast<int64_t>(j) - static_cast<int64_t>(a), -ps.t, n - j + 1);
            const OArith128::Elt prod = O.scal(c, O.mul(X1, X2));
            const OArith128::Elt Sy = GroupRingAtWildPoint128(O, ones, sgn, r);
            const OArith128::Elt pol = O.scal(O.ctx.mul(resVal, static_cast<u128>(scalInt) % q), Sy);
            if (ps.cross && tot < rec.size()) {
                const PolePoint& R64 = rec[tot];
                ++ptCmp;
                ptBad += !(modq64(X1, R64.X1) && modq64(X2, R64.X2) && modq64(prod, R64.prod) && modq64(pol, R64.pol));
            }
            const u128 lv = O.ctx.ell_pow(vD);
            const uint64_t us = (scalInt / Pow(ell, vs)) % pd, uc = static_cast<uint64_t>((c / O.ctx.ell_pow(vc)) % pd);
            const uint64_t iD = InvModU64(static_cast<uint64_t>(static_cast<u128>(us) * uc % pd), pd);
            const Elt V = GroupRingAtWildPoint(OR, g, ps.t, j);
            auto rhsMatches = [&](int64_t scale, bool withPolar, bool& integral) {
                const u128 sc = scale >= 0 ? static_cast<u128>(scale) % q : O.ctx.neg(static_cast<u128>(-scale) % q);
                integral = true;
                bool same = true;
                for (uint32_t x = 0; x < O.e; ++x) {
                    const u128 zx = withPolar ? O.ctx.add(prod[x], O.ctx.mul(sc, pol[x])) : prod[x];
                    if (zx % lv) {
                        integral = false;
                        continue;
                    }
                    const uint64_t y = static_cast<uint64_t>(static_cast<u128>(static_cast<uint64_t>((zx / lv) % pd)) * iD % pd * (s % pd) % pd);
                    same = same && ModNeg(y, pd) == V[x] % pd;
                }
                return integral && same;
            };
            bool in1 = true, in2 = true, in3 = true, in4 = true;
            ++tot;
            const bool ok = rhsMatches(1, true, in1);
            bad += !ok;
            nonint += !in1;
            negNone += !rhsMatches(1, false, in2);
            negSign += !rhsMatches(-1, true, in3);
            negScale += !rhsMatches(static_cast<int64_t>(1 + ell), true, in4);
        }
    }
    const std::string tag128 = tag + " [u128, P=" + std::to_string(P) + "]";
    log.check("W6", tot == expect, tag128 + ": " + std::to_string(tot) + " wild critical points on the pole branches (predicted " + std::to_string(expect) + ")");
    log.check("W6", bad == 0, tag128 + ": -s (F1 F2 + Res/(zeta u^r - 1)) fails mod ell^d at " + std::to_string(bad) + " / " + std::to_string(tot) + " (non-integral " +
                                     std::to_string(nonint) + ")");
    log.check("W6", negNone == tot, tag128 + ": NEG no polar term fails at " + std::to_string(negNone) + " / " + std::to_string(tot) + " (all predicted)");
    log.check("W6", negSign == tot, tag128 + ": NEG -Res fails at " + std::to_string(negSign) + " / " + std::to_string(tot) + " (all predicted)");
    log.check("W6", negScale == 0, tag128 + ": Res (1+ell) fails at " + std::to_string(negScale) + " (predicted 0: invisible at wild points)");
    std::cout << "WILD_POLE cell=" << Tag(n, ell) << " route=u128 P=" << P << " Pneed=" << Pneed << " d=" << d << " M=" << ps.M << " m=" << m << " t=" << ps.t << " s=" << s
              << " points=" << tot << " fail=" << bad << " neg_nopolar=" << negNone << " neg_minusRes=" << negSign << " Res(1+ell)=" << negScale << " secs=" << (Now() - t1)
              << std::endl;
    if (ps.cross) {
        const bool countsEq = run64 && cnt64[0] == tot && cnt64[1] == bad && cnt64[2] == negNone && cnt64[3] == negSign && cnt64[4] == negScale;
        log.check("W6", run64 && inBad == 0, tag128 + ": route agreement: omega^x, ell B_s, Res differ from the 64-bit route mod ell^" + std::to_string(P64) + " at " +
                                                std::to_string(inBad));
        log.check("W6", run64 && ptCmp == tot && ptBad == 0, tag128 + ": route agreement: F B_{r,chi} / product / polar elements differ mod ell^" + std::to_string(P64) +
                                                                  " at " + std::to_string(ptBad) + " / " + std::to_string(ptCmp) + " points");
        log.check("W6", countsEq, tag128 + ": route agreement: the five W6 counts differ between the two routes");
        std::cout << "WILD_ROUTE cell=" << Tag(n, ell) << " P64=" << P64 << " P128=" << P << " inputs_bad=" << inBad << " points_compared=" << ptCmp << " points_bad=" << ptBad
                  << " counts_equal=" << countsEq << std::endl;
    }
}

// ======================================================================================================== W7
void RunNonOrdinary(uint64_t ell, Log& log) {
    const std::string tag = Tag(10, ell);
    std::string what;
    bool refused = false;
    uint64_t a_ell = 0;
    try {
        const SymbolCell C = RunSymbolCell(10, ell, 2, {{2, 1}}, true);
        if (!C.simple) {
            what = "symbol cell: " + C.refusal;
        } else {
            a_ell = C.a_p.at(ell);
            (void)UnitRoot(C.ring, a_ell, 12);
            what = "UnitRoot accepted a_ell = " + std::to_string(a_ell);
        }
    } catch (const std::logic_error& e) {
        what = e.what();
        refused = what.find("ordinary") != std::string::npos;
    }
    log.check("W7", refused && a_ell % ell == 0, tag + ": the non-ordinary cell was not refused by UnitRoot (" + what + ")");
    std::cout << "WILD_W7 cell=" << tag << " a_ell=" << a_ell << " refused=" << refused << std::endl;
}

} // namespace
} // namespace lvalue
} // namespace hecke_ext

uint32_t HeckeLValueWildCharacterValidation() {
    using namespace hecke_ext::lvalue;
    const double T0 = Now();
    Log log;
    const char* hv = std::getenv("HECKE_LVALUE_WILD_HEAVY");
    const bool heavy = hv && std::string(hv) == "1";
    const std::map<uint64_t, uint64_t> sep2{{2, 1}}, sep235{{2, 1}, {3, 3}, {5, 7}}, sep3{{3, 1}};
    std::vector<std::pair<uint64_t, uint32_t>> w0{{5, 2}, {5, 3}, {7, 2}, {11, 2}, {37, 2}};
    struct W1Cell {
        uint64_t ell;
        uint32_t m, M, R;
    };
    std::vector<W1Cell> w1{{5, 2, 2, 6}, {5, 3, 3, 6}, {7, 2, 3, 5}, {11, 2, 2, 5}, {37, 2, 2, 4}};
    std::vector<CuspSpec> cusp{{10, 11, 4, 3, 2, sep2, true}, {10, 13, 4, 3, 2, sep2, true}, {10, 37, 4, 3, 2, sep235, true}, {30, 37, 4, 3, 2, sep235, false}};
    struct W4Cell {
        uint32_t n;
        uint64_t ell;
        uint32_t M;
        std::vector<uint32_t> ms;
    };
    std::vector<W4Cell> w4{{12, 5, 3, {2, 3}}, {14, 7, 3, {2}}, {16, 11, 2, {2}}, {30, 37, 2, {2}}};
    std::vector<ComplexCell> cx{{12, 5, 2}, {12, 7, 2}, {12, 11, 2}, {16, 5, 2}, {16, 7, 2}};
    std::vector<PoleSpec> pole{{30, 37, 4, 2, 2, 1, sep235, true}, {282, 37, 4, 2, 2, 1, sep3, true}, {226, 103, 4, 2, 2, 1, sep3, false}};
    if (heavy) {
        cx.push_back({12, 5, 3});
        cx.push_back({12, 13, 2});
        cusp.push_back({10, 11, 4, 3, 3, sep2, true});
        pole.push_back({30, 37, 4, 2, 2, 2, sep235});
    }
    const size_t cells = w0.size() + w1.size() + cusp.size() + w4.size() + cx.size() + pole.size() + 2;
    std::cout << "EFFECTIVE HeckeLValueWildCharacterValidation arm=" << (heavy ? "HEAVY" : "DEFAULT") << " cells=" << cells << std::endl;
    auto guarded = [&](const std::string& arm, const std::string& what, const std::function<void()>& body) {
        try {
            body();
        } catch (const std::exception& e) {
            log.check(arm, false, what + ": exception: " + e.what());
        }
    };
    for (const auto& c : w0) guarded("W0", "W0", [&] { RunW0(c.first, c.second, log); });
    for (const auto& c : w1) guarded("W1", "W1", [&] { RunW1(c.ell, c.m, c.M, c.R, log); });
    for (const auto& c : cusp) guarded("W2", "W2/W3 " + Tag(c.n, c.ell), [&] { RunCusp(c, log); });
    for (const auto& c : w4) guarded("W4", "W4 " + Tag(c.n, c.ell), [&] { RunEis(c.n, c.ell, c.M, c.ms, log); });
    {
        const nf::MpContext C(kPrec);
        for (const auto& c : cx) guarded("W5", "W5 k=" + std::to_string(c.k), [&] { RunComplex(c, C, log); });
    }
    for (const auto& c : pole) guarded("W6", "W6 " + Tag(c.n, c.ell), [&] { RunPole(c, log); });
    for (uint64_t ell : {uint64_t(5), uint64_t(7)}) guarded("W7", "W7", [&] { RunNonOrdinary(ell, log); });
    std::ostringstream arms;
    for (const auto& kv : log.per_arm) arms << kv.first << "=" << kv.second << " ";
    std::cout << "P4_ARMS checks=" << log.checks << " failing=" << log.failing << " per_arm: " << arms.str() << "secs=" << (Now() - T0)
              << " peak_rss_gb=" << PeakRssGb() << std::endl;
    return log.failing == 0 ? 0u : 1u;
}
