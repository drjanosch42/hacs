// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 21, P2) HeckeLValueEisensteinMeasureValidation -- see HeckeLValueEisensteinMeasureValidation.h.
// Spec docs/implementation/slot21/P2_spec.md (v0.1, predictions committed before this code ran); the Python oracle's run of record is
// docs/implementation/slot21/p2/out_run_p2.txt.  Every expected value below is the spec's prediction, never this code's output.
#include "HeckeLValueEisensteinMeasureValidation.h"

#include "Bernoulli.h"
#include "BirchModularSymbol.h"
#include "EisensteinMeasure.h"
#include "ModularSymbolMeasure.h"

#include "../../../ModArithmetic/EngineCheck.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace hecke_ext {
namespace lvalue {
namespace {

using namespace birch;
using namespace eis;
using u128 = unsigned __int128;

constexpr uint64_t kQ1 = (uint64_t(1) << 61) - 1;   // primes for the identities over Q
constexpr uint64_t kQ2 = (uint64_t(1) << 62) - 57;

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
uint32_t ValU(uint64_t x, uint64_t ell) {
    uint32_t v = 0;
    while (x && x % ell == 0) {
        x /= ell;
        ++v;
    }
    return v;
}
std::string Tag(uint32_t n, uint64_t ell) { return "(" + std::to_string(n) + "," + std::to_string(ell) + ")"; }
uint32_t MaxK(uint64_t ell) { return static_cast<uint32_t>(std::floor(61.9 / std::log2(static_cast<double>(ell)))); }

struct EisCell {
    uint32_t n;
    uint64_t ell;
    uint32_t M;
};

// ---- one Eisenstein cell: E0 - E7, E9(iv) ----
void RunEisCell(const EisCell& ce, Log& log) {
    const double t0 = Now();
    const uint32_t n = ce.n, k = n + 2, M = ce.M;
    const uint64_t ell = ce.ell;
    const std::string tag = Tag(n, ell) + " M=" + std::to_string(M);
    ENGINE_CHECK_MSG(n % 2 == 0 && (k % (ell - 1)) != 0 && ell >= 5, "P2: cell outside the spec's range");
    const unsigned th = MeasureThreads();
    const BernoulliOverKTable BT(ell, 2, k);   // only the exact valuation d is read
    const int32_t d = BT(k).v;
    std::ostringstream line;
    line << "EISMEAS cell=" << Tag(n, ell) << " k=" << k << " M=" << M << " d=" << d;

    // ---- E0: Zagier's relations (mod q1) ----
    {
        const uint64_t p = kQ1;
        const std::vector<uint64_t> B = BernoulliModP(n + 3, p);
        const std::vector<uint64_t> Pt = ZagierLaurentModP(n, p, B);
        uint64_t seed = 0x9e3779b97f4a7c15ull ^ (n * 1000003ull + ell);
        uint32_t badS = 0, badU = 0, negS = 0, negU = 0, pts = 0;
        while (pts < 6) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            const uint64_t x = (seed >> 3) % p;
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            const uint64_t y = (seed >> 3) % p;
            // the transformed points (x,y), (y,-x), (y, y-x), (-x, -x+... ) must have non-zero coordinates
            const uint64_t mx = (p - x) % p, ymx = (y + p - x) % p;
            if (x == 0 || y == 0 || ymx == 0 || mx == 0) continue;
            ++pts;
            for (int interior = 0; interior < 2; ++interior) {
                const bool io = interior == 1;
                const uint64_t P0 = EvalLaurentAt(Pt, n, 1, 0, 0, 1, x, y, p, io);
                const uint64_t PS = EvalLaurentAt(Pt, n, 0, -1, 1, 0, x, y, p, io);
                const uint64_t PU = EvalLaurentAt(Pt, n, 0, -1, 1, 1, x, y, p, io);
                const uint64_t PU2 = EvalLaurentAt(Pt, n, -1, -1, 1, 0, x, y, p, io);
                const bool okS = (P0 + PS) % p == 0, okU = ((P0 + PU) % p + PU2) % p == 0;
                if (!io) {
                    badS += !okS;
                    badU += !okU;
                } else {
                    negS += !okS;
                    negU += !okU;
                }
            }
        }
        log.check("E0", badS == 0 && badU == 0, tag + ": Zagier's n! p_k^- fails 1+S at " + std::to_string(badS) + ", 1+U+U^2 at " + std::to_string(badU) + " / 6 points");
        log.check("E0", negS + negU > 0, tag + ": NEG the interior P^Eis alone satisfies both relations at all 6 points");
        line << " E0=" << badS + badU << "/neg" << negS + negU;
    }

    // ---- E1, E2, E3 over Q (mod q1, q2) ----
    std::vector<uint64_t> primes{kQ1};
    if (ell < 100) primes.push_back(kQ2);
    uint32_t e1bad = 0, e1tot = 0, e2bad = 0, e2par = 0, negCusp = 0, negPoly = 0, e3bad = 0, e3tot = 0;
    for (uint64_t p : primes) {
        const EisModP Ep = BuildEisModP(n, ell, M, p, th);
        const std::vector<uint64_t> B = BernoulliModP(n + 3, p);
        const std::vector<uint64_t> Pt = ZagierLaurentModP(n, p, B);
        const uint64_t lk2 = PowModP(ell, k - 2, p);
        std::vector<std::vector<uint64_t>> cpoly(M + 1);
        for (uint32_t m = 0; m <= M; ++m) {
            const uint64_t Mm = Pow(ell, m), iM = InvModP(Mm % p, p);
            cpoly[m].resize(Mm);
            for (uint64_t a = 0; a < Mm; ++a) {
                cpoly[m][a] = CuspSymbolModP(Pt, n, a, Mm, p, true);
                if (m == 0) continue;
                const uint64_t lhs = CuspSymbolModP(Pt, n, a, Mm, p, false);
                const uint64_t rhs = (p - Ep.D[m][a] + MulModP(Ep.Bk_over_k, MulModP(a % p, iM, p), p)) % p;
                ++e1tot;
                e1bad += lhs != rhs;
            }
        }
        // E2: distribution m = 1 .. M-1, parity m = 1 .. M; NEG cusp term kept; NEG mu^poly
        bool pCusp = false, pPoly = false;
        for (uint32_t m = 1; m < M; ++m) {
            const uint64_t Mm = Pow(ell, m), iM1 = InvModP(Pow(ell, m + 1) % p, p), iM = InvModP(Mm % p, p);
            const uint64_t iMm1 = InvModP(Pow(ell, m - 1) % p, p);
            bool cuspFail = false, polyFail = false;
            for (uint64_t a = 0; a < Mm; ++a) {
                uint64_t s = 0, sc = 0, sp = 0;
                for (uint64_t b = 0; b < ell; ++b) {
                    const uint64_t ab = a + b * Mm;
                    s = (s + Ep.mu(ab, m + 1)) % p;
                    // mu~ = mu^Eis + (B_k/k)(a/ell^{m+1} - ell^{k-2} (a mod ell^m)/ell^m)
                    const uint64_t cusp = (MulModP(ab % p, iM1, p) + p - MulModP(lk2, MulModP((ab % Mm) % p, iM, p), p)) % p;
                    sc = (sc + Ep.mu(ab, m + 1) + MulModP(Ep.Bk_over_k, cusp, p)) % p;
                    sp = (sp + cpoly[m + 1][ab] + p - MulModP(lk2, cpoly[m][ab % Mm], p)) % p;
                }
                e2bad += s != Ep.mu(a, m);
                const uint64_t cuspA = (MulModP(a % p, iM, p) + p - MulModP(lk2, MulModP((a % Pow(ell, m - 1)) % p, iMm1, p), p)) % p;
                cuspFail = cuspFail || sc != (Ep.mu(a, m) + MulModP(Ep.Bk_over_k, cuspA, p)) % p;
                const uint64_t mpa = (cpoly[m][a] + p - MulModP(lk2, cpoly[m - 1][a % Pow(ell, m - 1)], p)) % p;
                polyFail = polyFail || sp != mpa;
            }
            pCusp = pCusp || cuspFail;
            pPoly = pPoly || polyFail;
        }
        negCusp += pCusp;
        negPoly += pPoly;
        for (uint32_t m = 1; m <= M; ++m) {
            const uint64_t Mm = Pow(ell, m);
            for (uint64_t a = 0; a < Mm; ++a) e2par += (Ep.mu((Mm - a) % Mm, m) + Ep.mu(a, m)) % p != 0;
        }
        if (p == kQ1) {
            const std::vector<uint64_t> cv = ConvolutionModP(n, ell, M, p, th);
            const uint64_t MM = Pow(ell, M);
            for (uint64_t a = 1; a < MM; ++a) {
                if (a % ell == 0) continue;
                ++e3tot;
                e3bad += cv[a] != Ep.mu(a, M);
            }
        }
    }
    log.check("E1", e1bad == 0, tag + ": symbol != -D + (B_k/k) a/ell^m at " + std::to_string(e1bad) + " / " + std::to_string(e1tot));
    log.check("E2", e2bad == 0 && e2par == 0, tag + ": mu^Eis distribution failures " + std::to_string(e2bad) + ", parity " + std::to_string(e2par));
    if (M >= 2) {
        log.check("E2", negCusp == primes.size(), tag + ": NEG the cusp term kept is a distribution at some level");
        log.check("E2", negPoly == primes.size(), tag + ": NEG mu^poly is a distribution over Q at some level");
    }
    log.check("E3", e3bad == 0 && e3tot > 0, tag + ": the convolution form fails at " + std::to_string(e3bad) + " / " + std::to_string(e3tot) + " units");
    line << " E1=" << e1bad << "/" << e1tot << " E2=" << e2bad << "+par" << e2par << " negcusp=" << negCusp << " negpoly=" << negPoly << " E3=" << e3bad << "/" << e3tot;

    // ---- ell-adic arms ----
    const uint32_t E = 2 * M + 1 + ValU(k - 1, ell);
    const uint32_t R = std::min<uint32_t>(8, MaxK(ell) - E);
    ENGINE_CHECK_MSG(R >= 3 && static_cast<int32_t>(R) > d, "P2: ell-adic precision too small for this cell");
    const EisLadic L = BuildEisLadic(n, ell, M, R, th);
    const uint64_t NK = L.NK, MM = Pow(ell, M), qR = Pow(ell, R), lim = Pow(ell, std::min(M, R));
    const uint32_t pole1 = static_cast<uint32_t>(ell - 2), pole2 = static_cast<uint32_t>((k - 1) % (ell - 1));

    // E4: twisted masses, exact
    {
        const uint32_t Rtb = std::min<uint32_t>(R, MaxK(ell) >= 4 ? MaxK(ell) - 4 : 0);
        const TwistedBernoulliTable TB(ell, Rtb, k + 1);
        uint32_t bad = 0, tot = 0;
        for (uint32_t i = 1; i + 1 < ell; i += 2) {
            u128 lhs = 0;
            for (uint64_t a = 1; a < MM; ++a)
                if (a % ell) lhs = (lhs + static_cast<u128>(TeichmullerPowMod(a % ell, i, ell, L.K) % NK) * L.mu[a]) % NK;
            const LadicReading F1 = TB(1, i), F2 = TB(k - 1, static_cast<uint32_t>((ell - 1 - i) % (ell - 1)));
            Lad x;
            if (F1.zero || F2.zero) {
                x.zero = true;
            } else {
                x.v = F1.v + F2.v;
                x.unit = static_cast<uint64_t>(static_cast<u128>(F1.unit) * F2.unit % Pow(ell, Rtb));
            }
            const int32_t Rc = std::min<int32_t>(static_cast<int32_t>(R), static_cast<int32_t>(Rtb) + (x.zero ? 0 : x.v));
            if (Rc <= 0) continue;
            ++tot;
            // compare -x * ell^E with lhs modulo ell^{E + Rc}
            const uint64_t Nc = Pow(ell, E + static_cast<uint32_t>(Rc));
            const uint64_t xs = LadScaled(x, ell, E, static_cast<uint32_t>(Rc));
            bad += static_cast<uint64_t>(lhs % Nc) != (Nc - xs % Nc) % Nc;
        }
        log.check("E4", bad == 0 && tot > 0, tag + ": twisted masses differ from -B_{1,w^i} B_{k-1,w^-i}/(k-1) at " + std::to_string(bad) + " / " + std::to_string(tot));
        line << " E4=" << bad << "/" << tot;
    }

    // E5, E6, E7, E9(iv)
    std::vector<uint32_t> classes;
    for (uint32_t b = 2; b < ell - 1; b += 2) classes.push_back(b);
    std::map<uint32_t, std::vector<uint64_t>> layer;
    const bool useLayer = ell <= 103 && M >= 1;
    if (useLayer) layer = KlGroupRingLayer(classes, ell, M, th);
    std::vector<uint64_t> mup;
    const bool link = d >= 1;
    if (link) {
        const Ring Rg = MakeRing(ell, R);   // values mod ell^R, scaled by ell^E below (EisensteinSymbol needs ell^{R+3} < 2^62)
        const SymRho rho(n, Rg);
        int32_t vmin = 0;
        const Vec PE = EisensteinSymbol(n, Rg, &vmin);
        ENGINE_CHECK_MSG(vmin == 0, "P2: P^Eis not integral and primitive at an irregular cell");
        const uint64_t a_ell = Rg.add(1 % Rg.q, Rg.pow(ell, k - 1));
        const ModularSymbolMeasure mp = BuildMeasure(rho, PE, a_ell, 1, M, true);
        mup.assign(MM, 0);
        const uint64_t lE = Pow(ell, L.E);
        for (uint64_t a = 1; a < MM; ++a)
            if (a % ell) mup[a] = static_cast<uint64_t>(static_cast<u128>(mp.mu[M][a]) * lE % NK);
    }
    uint32_t e5bad = 0, e5tot = 0, evenNZ = 0, wrongMatch = 0, wrongTot = 0;
    std::set<int32_t> e7off, e7pole;
    std::ostringstream e6s;
    for (uint32_t a = 0; a + 1 < ell; ++a) {
        const std::vector<uint64_t> g = L.branch(a);
        if (a % 2 == 0) {
            for (uint64_t x : g) evenNZ += (x % NK) != 0;
            continue;
        }
        std::vector<uint64_t> gp;
        if (link) gp = L.branch(a, &mup);
        if (a == pole1 || a == pole2) {
            const std::vector<uint64_t> pol = PolarElement(L, a);
            std::vector<uint64_t> reg(g.size());
            for (size_t i = 0; i < g.size(); ++i) reg[i] = L.sub(g[i], pol[i]);
            const int32_t vg = L.vmin(g), vr = L.vmin(reg);
            const bool vok = d < static_cast<int32_t>(M) ? vg == d - static_cast<int32_t>(M) : vg >= 0;
            log.check("E6", vok, tag + ": pole branch " + std::to_string(a) + " valuation " + std::to_string(vg) + " (d - M = " + std::to_string(d - static_cast<int32_t>(M)) + ")");
            log.check("E6", vr >= 0, tag + ": pole branch " + std::to_string(a) + " minus the polar term has valuation " + std::to_string(vr));
            e6s << a << ":(" << vg << "," << vr << ")";
            if (link) {
                std::vector<uint64_t> df(g.size());
                for (size_t i = 0; i < g.size(); ++i) df[i] = L.sub(gp[i], reg[i]);
                e7pole.insert(L.vmin(df));
            }
            continue;
        }
        // regular odd branch: integral, = -G_{a+1}(u(1+T)-1) G_{k-1-a}(u^{k-1}(1+T)^{-1}-1)
        const uint64_t lE = Pow(ell, L.E);
        bool integral = true;
        std::vector<uint64_t> gi(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            integral = integral && g[i] % lE == 0;
            gi[i] = (g[i] / lE) % qR;
        }
        const uint32_t b1 = (a + 1) % (ell - 1), b2 = static_cast<uint32_t>(((static_cast<int64_t>(k) - 1 - a) % static_cast<int64_t>(ell - 1) + (ell - 1)) % (ell - 1));
        const std::vector<uint64_t> G1 = GrTw1(KlGroupRingBernoulli(b1, ell, M, R), ell, qR);
        const std::vector<uint64_t> G2 = GrTwInv(KlGroupRingBernoulli(b2, ell, M, R), ell, qR, k);
        std::vector<uint64_t> pr = GrMul(G1, G2, qR);
        bool ok = integral;
        for (size_t i = 0; i < pr.size(); ++i) ok = ok && (gi[i] + pr[i]) % lim == 0;
        if (useLayer) {
            const uint64_t qm = Pow(ell, M);
            const std::vector<uint64_t> H = GrMul(GrTw1(layer.at(b1), ell, qm), GrTwInv(layer.at(b2), ell, qm, k), qm);
            for (size_t i = 0; i < H.size(); ++i) ok = ok && (gi[i] + H[i]) % std::min(lim, qm) == 0;
        }
        ++e5tot;
        e5bad += !ok;
        // NEG (E9 iv): the wrong twist G_{a+3}
        const uint32_t bw = (a + 3) % (ell - 1);
        // (at level 1 the comparison is one residue mod ell: a coincidence has probability 1/ell per branch, so the control is run at M >= 2 only)
        if (M >= 2 && bw != 0 && bw != b1) {
            const std::vector<uint64_t> W1 = GrTw1(KlGroupRingBernoulli(bw, ell, M, R), ell, qR);
            const std::vector<uint64_t> pw = GrMul(W1, G2, qR);
            bool same = true;
            for (size_t i = 0; i < pw.size(); ++i) same = same && (gi[i] + pw[i]) % lim == 0;
            ++wrongTot;
            wrongMatch += same;
        }
        if (link) {
            std::vector<uint64_t> df(g.size());
            for (size_t i = 0; i < g.size(); ++i) df[i] = L.sub(gp[i], g[i]);
            e7off.insert(L.vmin(df));
        }
    }
    // (ell = 5: both odd branches are pole branches, so there is no regular branch to compare)
    log.check("E5", e5bad == 0 && (e5tot > 0 || ell == 5), tag + ": regular odd branches != -G G at " + std::to_string(e5bad) + " / " + std::to_string(e5tot));
    log.check("E5", evenNZ == 0, tag + ": even branches non-zero at " + std::to_string(evenNZ) + " entries");
    if (wrongTot > 0) log.check("E9", wrongMatch == 0, tag + ": NEG the wrong twist G_{a+3} matches " + std::to_string(wrongMatch) + " / " + std::to_string(wrongTot));
    line << " E5=" << e5bad << "/" << e5tot << (useLayer ? "(2 routes)" : "(E_1c route)") << " E6=" << e6s.str() << " wrongtwist=" << wrongMatch << "/" << wrongTot;
    if (link) {
        const bool okOff = e7off.size() == 1 && *e7off.begin() == d, okPole = e7pole.size() == 1 && *e7pole.begin() == d;
        log.check("E7", okOff, tag + ": v(mu^poly - mu^Eis) off the pole branches is not exactly d");
        log.check("E7", okPole, tag + ": v(mu^poly - (mu^Eis - polar)) on the pole branches is not exactly d");
        line << " E7=off" << (e7off.size() == 1 ? std::to_string(*e7off.begin()) : "mixed") << ",pole" << (e7pole.size() == 1 ? std::to_string(*e7pole.begin()) : "mixed");
    }
    line << " R=" << R << " E=" << L.E << " secs=" << (Now() - t0);
    std::cout << line.str() << std::endl;
}

// ---- E8 / E9 (i)-(iii): the cusp form's pole branches at the critical points ----
struct CuspCell {
    uint32_t n;
    uint64_t ell;
    uint32_t R, M;
    std::map<uint64_t, uint64_t> sep;
    uint32_t total;                                     // predicted number of critical points on both pole branches
    std::vector<std::pair<uint32_t, uint32_t>> located; // [E] cl:lvmeascong(4)
    // P2x (spec Sec. 10): every odd branch at the critical points (not only the pole branches), plus the group-ring arm (b)
    bool all_odd = false;
    std::string arm = "E8", neg_arm = "E9";
    uint64_t s_expect = 0;                              // 0: not predicted
};

// branch elements of mm for the listed branches, threaded over branches (ModularSymbolMeasure::branch is const)
template <class Fn>
void ForBranches(const std::vector<uint32_t>& bs, Fn fn) {
    const unsigned th = std::max(1u, std::min<unsigned>(MeasureThreads(), static_cast<unsigned>(bs.size())));
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < th; ++t)
        pool.emplace_back([&, t]() {
            for (size_t i = t; i < bs.size(); i += th) fn(i, bs[i]);
        });
    for (std::thread& x : pool) x.join();
}

// ---- E8-L2 (b): Delta's branch elements against s (g^Eis - polar) in Z/ell^d[C_G], d = 1 (spec Sec. 10.2) ----
// g^Eis = -branch(E_1) branch(iota_* E_{k-1}) (the E3 convolution through the ring homomorphism branch_a), scaled by ell^E mod ell^{E+1}.
void RunGroupRingL2(const CuspCell& cc, const std::vector<uint32_t>& odd, const std::vector<std::vector<uint64_t>>& gD, uint64_t s, int32_t d,
                    Log& log) {
    const double t0 = Now();
    const uint32_t n = cc.n, M = cc.M;
    const uint64_t ell = cc.ell;
    const std::string tag = Tag(n, ell) + " R=" + std::to_string(cc.R) + " M=" + std::to_string(M);
    ENGINE_CHECK_MSG(d == 1, "P2x: the group-ring arm reads one ell-adic digit (d = 1)");
    const EisLadic L = BuildEisLadic(n, ell, M, 1, MeasureThreads(), false);
    const uint64_t MM = Pow(ell, M), NK = L.NK, lE = Pow(ell, L.E), phi = MM / ell * (ell - 1);
    std::vector<uint64_t> iek(MM, 0);
    for (uint64_t x = 1; x < MM; ++x)
        if (x % ell) iek[x] = L.ek[PowModP(x, phi - 1, MM)];
    const uint32_t pole1 = static_cast<uint32_t>(ell - 2), pole2 = static_cast<uint32_t>((n + 1) % (ell - 1));
    struct Out {
        bool pole = false, ok = false, negNone = false, negMinus = false, negScale = false, negS1 = false;
        int32_t v = 0, vreg = 0;
    };
    std::vector<Out> out(odd.size());
    const uint64_t s1 = s % ell;
    ForBranches(odd, [&](size_t ib, uint32_t a) {
        const std::vector<uint64_t> bA = L.branch(a, &L.e1), bB = L.branch(a, &iek);
        std::vector<uint64_t> g = GrMul(bA, bB, NK);
        for (uint64_t& x : g) x = (NK - x) % NK;   // mu^Eis = -E_1 * iota E_{k-1}
        Out& o = out[ib];
        o.pole = a == pole1 || a == pole2;
        // matches(res_scale): X = g - polar(res_scale) integral and s (X / ell^E) = g^Delta mod ell, coefficientwise
        auto matches = [&](int64_t res_scale, uint64_t sc) -> bool {
            const std::vector<uint64_t> pol = PolarElement(L, a, res_scale);
            for (size_t i = 0; i < g.size(); ++i) {
                const uint64_t X = L.sub(g[i], pol[i]);
                if (X % lE) return false;
                if ((sc * ((X / lE) % ell) + ell - gD[ib][i] % ell) % ell) return false;
            }
            return true;
        };
        o.ok = matches(1, s1);
        o.negS1 = !matches(1, (s1 + 1) % ell);
        if (o.pole) {
            o.v = L.vmin(g);
            std::vector<uint64_t> reg(g.size());
            const std::vector<uint64_t> pol = PolarElement(L, a, 1);
            for (size_t i = 0; i < g.size(); ++i) reg[i] = L.sub(g[i], pol[i]);
            o.vreg = L.vmin(reg);
            o.negNone = !matches(0, s1);
            o.negMinus = !matches(-1, s1);
            o.negScale = !matches(static_cast<int64_t>(1 + ell), s1);
        }
    });
    uint32_t regTot = 0, regBad = 0, poleTot = 0, poleBad = 0, e6bad = 0, nNone = 0, nMinus = 0, nScale = 0, nS1 = 0;
    std::ostringstream e6s;
    for (size_t ib = 0; ib < odd.size(); ++ib) {
        const Out& o = out[ib];
        nS1 += o.negS1;
        if (!o.pole) {
            ++regTot;
            regBad += !o.ok;
            continue;
        }
        ++poleTot;
        poleBad += !o.ok;
        e6bad += !(o.v == d - static_cast<int32_t>(M) && o.vreg >= 0);
        e6s << odd[ib] << ":(" << o.v << "," << o.vreg << ")";
        nNone += o.negNone;
        nMinus += o.negMinus;
        nScale += o.negScale;
    }
    const uint32_t nOdd = static_cast<uint32_t>(odd.size());
    log.check(cc.arm, regTot == nOdd - 2 && regBad == 0, tag + ": (b) regular odd branches g^Delta != s g^Eis mod ell at " + std::to_string(regBad) + " / " + std::to_string(regTot));
    log.check(cc.arm, poleTot == 2 && e6bad == 0, tag + ": (b) E6 at level 2: min v(g^Eis) != d - M or g^Eis - polar not integral: " + e6s.str());
    log.check(cc.arm, poleBad == 0, tag + ": (b) pole branches g^Delta != s (g^Eis - polar) mod ell at " + std::to_string(poleBad) + " / " + std::to_string(poleTot));
    log.check(cc.neg_arm, nNone == poleTot, tag + ": (b) NEG no polar term fails at " + std::to_string(nNone) + " / " + std::to_string(poleTot) + " (all predicted)");
    log.check(cc.neg_arm, nMinus == poleTot, tag + ": (b) NEG -Res fails at " + std::to_string(nMinus) + " / " + std::to_string(poleTot) + " (all predicted)");
    log.check(cc.neg_arm, nScale == poleTot, tag + ": (b) NEG Res (1+ell) fails at " + std::to_string(nScale) + " / " + std::to_string(poleTot) + " (all predicted: ell polar is a unit times N_u)");
    log.check(cc.neg_arm, nS1 == nOdd, tag + ": (b) NEG s+1 fails at " + std::to_string(nS1) + " / " + std::to_string(nOdd) + " (all predicted)");
    std::ostringstream fp;
    for (size_t ib = 0; ib < odd.size(); ++ib)
        if (out[ib].pole) fp << odd[ib] << ":(" << gD[ib][0] << "," << gD[ib][1] << ")";
    std::cout << "EISMEAS_DELTA2 cell=" << Tag(n, ell) << " M=" << M << " R=" << cc.R << " d=" << d << " s=" << s << " gr_regular=" << regBad << "/" << regTot
              << " gr_pole=" << poleBad << "/" << poleTot << " e6=" << e6s.str() << " neg_nopolar=" << nNone << " neg_minusRes=" << nMinus
              << " neg_Res(1+ell)=" << nScale << " neg_s+1=" << nS1 << "/" << nOdd << " fp_gDelta=" << fp.str() << " E=" << L.E << " K=" << L.K
              << " secs=" << (Now() - t0) << std::endl;
}

void RunCuspCell(const CuspCell& cc, Log& log) {
    const double t0 = Now();
    const uint32_t n = cc.n, k = n + 2;
    const uint64_t ell = cc.ell;
    const std::string tag = Tag(n, ell) + " R=" + std::to_string(cc.R) + " M=" + std::to_string(cc.M);
    const SymbolCell C = RunSymbolCell(n, ell, cc.R, cc.sep, false);
    if (!C.simple || !C.eisenstein_congruent) {
        log.check("E8", false, tag + ": the symbol cell did not run: " + C.refusal);
        return;
    }
    const Ring& Rg = C.ring;
    const uint64_t q = Rg.q;
    const SymRho rho(n, Rg);
    const uint64_t a_ell = C.a_p.at(ell);
    const uint64_t alpha = UnitRoot(Rg, a_ell, k);
    const ModularSymbolMeasure mm = BuildMeasure(rho, C.r_minus, a_ell, alpha, cc.M, true);
    int32_t vmin = 0;
    const Vec PE = EisensteinSymbol(n, Rg, &vmin);
    uint32_t t0u = 0;
    while (t0u <= n && !Rg.unit(C.r_minus[t0u])) ++t0u;
    ENGINE_CHECK_MSG(t0u <= n && Rg.unit(PE[t0u]), "P2: P^Eis is not a unit at r^-'s first unit coordinate");
    const uint64_t s = Rg.mul(C.r_minus[t0u], Rg.inv(PE[t0u]));
    const BernoulliOverKTable BTd(ell, std::min<uint32_t>(4, MaxK(ell) - 4), k);   // 4 at ell <= 103; 2 at 691 (only v is read)
    const int32_t d = BTd(k).v;
    const uint32_t P = std::min<uint32_t>(6, MaxK(ell) - 4);
    const TwistedBernoulliTable TB(ell, P, n + 2);
    const BernoulliOverKTable BT(ell, P, n + 2);
    const Lad Res = Residue(k, ell, P);
    const uint32_t S = std::min<uint32_t>(6, MaxK(ell) - static_cast<uint32_t>(d));   // ell^{S+d} < 2^62 (6 at ell <= 103; 5 at 691)
    const uint64_t pd = Pow(ell, static_cast<uint32_t>(d)), lS = Pow(ell, S), NSd = Pow(ell, S + static_cast<uint32_t>(d));
    auto Fval = [&](uint32_t r, uint32_t i, uint32_t eexp) -> Lad {
        Lad x;
        if (i % (ell - 1) != 0) {
            const LadicReading L = TB(r, i);
            x.zero = L.zero;
            x.v = L.v;
            x.unit = L.unit;
            return x;
        }
        const LadicReading L = BT(r);
        x.zero = L.zero;
        x.v = L.v;
        const uint64_t NP = Pow(ell, P);
        x.unit = static_cast<uint64_t>(static_cast<u128>(L.unit) * ((1 + NP - PowModP(ell, eexp, NP)) % NP) % NP);
        return x;
    };
    const uint32_t pole1 = static_cast<uint32_t>(ell - 2), pole2 = static_cast<uint32_t>((k - 1) % (ell - 1));
    std::vector<uint32_t> branches{pole2, pole1};
    if (cc.all_odd) {
        branches.clear();
        for (uint32_t a = 1; a < ell - 1; a += 2) branches.push_back(a);
    }
    std::vector<std::vector<uint64_t>> brs(branches.size());
    ForBranches(branches, [&](size_t ib, uint32_t a) { brs[ib] = mm.branch(a); });
    uint32_t tot = 0, bad = 0, negNone = 0, negScale = 0, negSign = 0, nonint = 0, poleTot = 0;
    std::vector<std::pair<uint32_t, uint32_t>> located, scaleFails;
    for (size_t ib = 0; ib < branches.size(); ++ib) {
        const uint32_t a = branches[ib];
        const std::vector<uint64_t>& br = brs[ib];
        const bool isPole = a == pole1 || a == pole2;
        for (uint32_t j = 0; j <= n; ++j) {
            const uint32_t i1 = static_cast<uint32_t>(((static_cast<int64_t>(a) - j) % static_cast<int64_t>(ell - 1) + (ell - 1)) % (ell - 1));
            const uint32_t i2 = static_cast<uint32_t>(((static_cast<int64_t>(j) - a) % static_cast<int64_t>(ell - 1) + (ell - 1)) % (ell - 1));
            const Lad F1 = Fval(j + 1, i1, j), F2 = Fval(n - j + 1, i2, n - j);
            if (F1.zero || F2.zero) continue;
            const Lad FF = LadMul(F1, F2, ell, P);
            ++tot;
            const uint64_t lhs = GrEval(br, j, ell, q) % pd;
            if (!isPole) {   // off the pole branches (P2x): -s F1 F2, no polar term
                bool in0 = true;
                u128 x = LadScaled(FF, ell, S, static_cast<uint32_t>(d));
                in0 = x % lS == 0;
                const uint64_t v = static_cast<uint64_t>(x / lS) % pd;
                const uint64_t rhs = (pd - static_cast<uint64_t>(static_cast<u128>(s % pd) * v % pd)) % pd;
                nonint += !in0;
                bad += !in0 || rhs != lhs;
                continue;
            }
            ++poleTot;
            const uint64_t r = a == pole1 ? j + 1 : k - 1 - j;
            const Lad pol = PolarValue(Res, ell, r, P);
            if (pol.v < d - 1) located.emplace_back(a, j);
            auto rhsOf = [&](int64_t scale, bool withPolar, bool& integral) -> uint64_t {
                u128 x = LadScaled(FF, ell, S, static_cast<uint32_t>(d));
                if (withPolar) {
                    const uint64_t ps = LadScaled(pol, ell, S, static_cast<uint32_t>(d));
                    const uint64_t sc = scale >= 0 ? static_cast<uint64_t>(scale) % NSd : (NSd - static_cast<uint64_t>(-scale) % NSd) % NSd;
                    x = (x + static_cast<u128>(ps) * sc) % NSd;
                }
                integral = x % lS == 0;
                const uint64_t v = static_cast<uint64_t>(x / lS) % pd;
                // -s (F1 F2 + polar)
                return (pd - static_cast<uint64_t>(static_cast<u128>(s % pd) * v % pd)) % pd;
            };
            bool in1 = true, in2 = true, in3 = true, in4 = true;
            const uint64_t rhs = rhsOf(1, true, in1);
            nonint += !in1;
            bad += !in1 || rhs != lhs;
            const uint64_t r0 = rhsOf(1, false, in2);
            negNone += !in2 || r0 != lhs;
            const uint64_t r1 = rhsOf(static_cast<int64_t>(1 + ell), true, in3);
            if (!in3 || r1 != lhs) scaleFails.emplace_back(a, j);
            const uint64_t r2 = rhsOf(-1, true, in4);
            negSign += !in4 || r2 != lhs;
        }
    }
    negScale = static_cast<uint32_t>(scaleFails.size());
    std::sort(located.begin(), located.end());
    std::sort(scaleFails.begin(), scaleFails.end());
    std::vector<std::pair<uint32_t, uint32_t>> expect = cc.located;
    std::sort(expect.begin(), expect.end());
    log.check(cc.arm, tot == cc.total, tag + ": " + std::to_string(tot) + " critical points on the " + (cc.all_odd ? "odd" : "pole") + " branches (predicted " + std::to_string(cc.total) + ")");
    log.check(cc.arm, bad == 0 && nonint == 0, tag + ": -s (F1 F2 + Res/(u^r-1)) fails mod ell^d at " + std::to_string(bad) + " / " + std::to_string(tot));
    log.check(cc.arm, located == expect, tag + ": the points with v(Res/(u^r-1)) < d-1 differ from [E] cl:lvmeascong(4)'s located set");
    log.check(cc.neg_arm, negNone == poleTot, tag + ": NEG no polar term fails at " + std::to_string(negNone) + " / " + std::to_string(poleTot) + " (all predicted)");
    log.check(cc.neg_arm, negSign == poleTot, tag + ": NEG -Res fails at " + std::to_string(negSign) + " / " + std::to_string(poleTot) + " (all predicted)");
    std::vector<std::pair<uint32_t, uint32_t>> ell_div;   // Res (1+ell) changes the polar value by ell Res/(u^r-1): fails iff ell | r
    for (uint32_t a : {pole2, pole1})
        for (uint32_t j = 0; j <= n; ++j) {
            const uint64_t r = a == pole1 ? j + 1 : k - 1 - j;
            if (r % ell == 0) ell_div.emplace_back(a, j);
        }
    std::sort(ell_div.begin(), ell_div.end());
    log.check(cc.neg_arm, scaleFails == ell_div, tag + ": NEG Res (1+ell) fails at " + std::to_string(negScale) + " points, not exactly at the ell | r points (" + std::to_string(ell_div.size()) + ")");
    std::ostringstream lp;
    for (const auto& pj : located) lp << "(" << pj.first << "," << pj.second << ")";
    if (cc.s_expect) log.check(cc.arm, s == cc.s_expect, tag + ": s = " + std::to_string(s) + " (predicted " + std::to_string(cc.s_expect) + ")");
    std::cout << "EISMEAS_POLE cell=" << Tag(n, ell) << " d=" << d << " M=" << cc.M << " s=" << s << " points=" << tot
              << (cc.all_odd ? " pole_points=" + std::to_string(poleTot) : std::string()) << " fail=" << bad
              << " located=" << (lp.str().empty() ? "-" : lp.str()) << " neg_nopolar=" << negNone << " neg_minusRes=" << negSign << " neg_Res(1+ell)=" << negScale
              << " secs=" << (Now() - t0) << std::endl;
    if (cc.all_odd) RunGroupRingL2(cc, branches, brs, s, d, log);
}

} // namespace
} // namespace lvalue
} // namespace hecke_ext

uint32_t HeckeLValueEisensteinMeasureValidation() {
    using namespace hecke_ext::lvalue;
    const double T0 = Now();
    Log log;
    const char* hv = std::getenv("HECKE_LVALUE_EISMEAS_HEAVY");
    const bool heavy = hv && std::string(hv) == "1";
    std::vector<EisCell> cells{{12, 5, 4},   {14, 7, 3},   {16, 11, 2},  {20, 13, 2},  {10, 37, 2},  {30, 37, 2},   {66, 37, 2},
                               {282, 37, 2}, {42, 59, 2},  {56, 67, 2},  {22, 103, 2}, {226, 103, 2}, {10, 691, 1}};
    if (heavy) {
        cells.push_back({30, 37, 3});
        cells.push_back({20, 131, 2});
    }
    std::cout << "EFFECTIVE HeckeLValueEisensteinMeasureValidation arm=" << (heavy ? "HEAVY" : "DEFAULT") << " cells=" << cells.size() << std::endl;
    try {
        for (const EisCell& c : cells) RunEisCell(c, log);
        const std::map<uint64_t, uint64_t> sep235{{2, 1}, {3, 3}, {5, 7}}, sep3{{3, 1}};
        const std::vector<CuspCell> cusp{
            {30, 37, 4, 3, sep235, 62, {}},
            {282, 37, 4, 3, sep3, 566,
             {{31, 24}, {31, 61}, {31, 98}, {31, 135}, {31, 172}, {31, 209}, {31, 246}, {35, 36}, {35, 73}, {35, 110}, {35, 147}, {35, 184}, {35, 221}, {35, 258}}},
            {226, 103, 4, 2, sep3, 454, {{23, 21}, {23, 124}, {101, 102}, {101, 205}}}};
        for (const CuspCell& c : cusp) RunCuspCell(c, log);
        {   // P2x (spec Sec. 10): Delta's own measure at (10,691), level 2, R = 2 (S2's symbol cell); in the default arm (0.8 s < 5 s, Sec. 10.4)
            CuspCell c2{10, 691, 2, 2, sep235, 3795, {}};
            c2.all_odd = true;
            c2.arm = "E8L2";
            c2.neg_arm = "E9L2";
            c2.s_expect = 286647;
            RunCuspCell(c2, log);
        }
    } catch (const std::exception& e) {
        log.check("EXC", false, std::string("exception: ") + e.what());
    }
    std::ostringstream arms;
    for (const auto& kv : log.per_arm) arms << kv.first << "=" << kv.second << " ";
    std::cout << "P2_ARMS checks=" << log.checks << " failing=" << log.failing << " per_arm: " << arms.str() << "secs=" << (Now() - T0)
              << " peak_rss_gb=" << PeakRssGb() << std::endl;
    return log.failing == 0 ? 0u : 1u;
}
