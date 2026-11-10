// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// PAR-D2: the blocked Smith form against D2's oracle (see the header).
#include "IntegralDescentSmithBlockedValidation.h"

#include "DescentMaps.h"
#include "DescentPair.h"
#include "DescentSmith.h"
#include "DescentTree.h"
#include "DescentTreePair.h"

#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Binary/SmithPar.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace M = integral_descent::maps;
namespace P = integral_descent::pair;
using M::Mat;
using M::ZMod;
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

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t below(uint64_t n) { return static_cast<uint32_t>(next() % n); }
};

Mat MulModRef(const ZMod& Z, const Mat& A, const Mat& B) {
    Mat C(A.r, B.c);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < B.c; ++j) {
            uint64_t acc = 0;
            for (uint32_t k = 0; k < A.c; ++k) acc = (acc + uint64_t(A.at(i, k)) * B.at(k, j)) % Z.m;
            C.at(i, j) = static_cast<uint32_t>(acc);
        }
    return C;
}
Mat Uniform(const ZMod& Z, uint32_t r, uint32_t c, Rng& g) {
    Mat A(r, c);
    for (auto& x : A.a) x = static_cast<uint32_t>(g.next() % Z.m);
    return A;
}
// kind: 0 dense; 1 low rank with torsion; 2 sparse with ell-divisible entries; 3 structured rows; 4 all divisible by ell;
// 5 sparse {-1, 0, 1} (a differential); 6 wide range of valuations (dense)
Mat Gen(const ZMod& Z, uint32_t kind, uint32_t r, uint32_t c, Rng& g) {
    if (r == 0 || c == 0) return Mat(r, c);
    switch (kind) {
        case 0: return Uniform(Z, r, c, g);
        case 1:
        case 3:
        case 4: {
            const uint32_t q = 1 + g.below(std::min(r, c));
            Mat X = Uniform(Z, r, q, g), Y = Uniform(Z, q, c, g);
            for (uint32_t i = 0; i < q; ++i) {
                const uint32_t a = g.below(3) ? 0u : g.below(std::min<uint32_t>(Z.k, 4));
                const uint32_t d = Z.mul(Z.pw(a), 1 + static_cast<uint32_t>(g.next() % (Z.m - 1)));
                for (uint32_t j = 0; j < c; ++j) Y.at(i, j) = Z.mul(Y.at(i, j), d);
            }
            Mat A = MulModRef(Z, X, Y);
            if (kind == 3) {
                for (uint32_t i = 0; i < r; ++i) {
                    const uint32_t w = g.below(6);
                    if (w == 0) std::fill(A.row(i), A.row(i) + c, 0u);
                    if (w == 1 && i > 0) std::copy(A.row(i - 1), A.row(i - 1) + c, A.row(i));
                    if (w == 2)
                        for (uint32_t j = 0; j < c; ++j) A.at(i, j) = Z.mul(A.at(i, j), Z.ell);
                }
            }
            if (kind == 4)
                for (auto& x : A.a) x = Z.mul(x, Z.ell);
            return A;
        }
        case 2: {
            Mat A(r, c);
            for (auto& x : A.a)
                if (g.below(12) == 0) x = Z.mul(Z.pw(g.below(3) ? 0u : g.below(std::min<uint32_t>(Z.k, 3))), static_cast<uint32_t>(g.next() % Z.m));
            return A;
        }
        case 5: {
            Mat A(r, c);
            for (auto& x : A.a) {
                const uint32_t w = g.below(16);
                x = w == 0 ? 1u : (w == 1 ? Z.sub(0, 1) : (w == 2 ? Z.red(2) : 0u));
            }
            return A;
        }
        default: {
            Mat A(r, c);
            for (auto& x : A.a) x = Z.mul(Z.pw(g.below(Z.k + 1) % (Z.k + 1)), static_cast<uint32_t>(g.next() % Z.m));
            return A;
        }
    }
}

bool SameMat(const Mat& a, const Mat& b) { return a.r == b.r && a.c == b.c && a.a == b.a; }
std::string SmithDiff(const M::Smith& a, const M::Smith& b) {
    if (a.sigma != b.sigma) return "sigma";
    if (!SameMat(a.L, b.L)) return "L";
    if (!SameMat(a.LinvT, b.LinvT)) return "LinvT";
    if (!SameMat(a.RT, b.RT)) return "RT";
    if (!SameMat(a.Rinv, b.Rinv)) return "Rinv";
    return "";
}
Mat Rows(const Mat& A, const std::vector<uint32_t>& rows) {
    Mat B(static_cast<uint32_t>(rows.size()), A.c);
    for (uint32_t q = 0; q < rows.size(); ++q) std::copy(A.row(rows[q]), A.row(rows[q]) + A.c, B.row(q));
    return B;
}

std::vector<int> SimdPaths() {
    std::vector<int> p = {0};
    for (int s : {1, 2})
        if (M::DescentSmithClampSimd(s) == s) p.push_back(s);
    return p;
}

// ===========================================================================
// (R) random matrices
// ===========================================================================
void ArmR() {
    struct ZK {
        uint32_t ell, k;
    };
    const std::vector<ZK> rings = {{2, 30}, {2, 1}, {2, 5}, {3, 19}, {3, 2}, {5, 13}, {7, 11}, {691, 3}, {46337, 2}};
    const std::vector<std::pair<uint32_t, uint32_t>> shapes = {{0, 5}, {5, 0}, {1, 1}, {1, 9}, {9, 1}, {3, 7}, {7, 3}, {17, 17},
                                                               {40, 40}, {61, 23}, {23, 61}, {96, 130}, {150, 88}};
    const std::vector<int> simds = SimdPaths();
    Rng g(20261027);
    uint32_t mats = 0, runs = 0, acc = 0;
    for (const ZK& zk : rings) {
        const ZMod Z(zk.ell, zk.k);
        for (const auto& sh : shapes)
            for (uint32_t kind = 0; kind <= 6; ++kind) {
                const Mat A = Gen(Z, kind, sh.first, sh.second, g);
                const M::Smith O = M::SmithFormOracle(Z, A, true, true);
                ++mats;
                // the configurations: thread counts x panels x SIMD paths (all on the large shapes, a rotation on the small ones)
                const bool big = uint64_t(sh.first) * sh.second >= 2000;
                for (uint32_t th : {1u, 2u, 7u, 16u})
                    for (uint32_t panel : {1u, 5u, 32u})
                        for (int sp : simds) {
                            if (!big && ((th + panel + static_cast<uint32_t>(sp) + kind) % 3 != 0)) continue;
                            M::SmithEngineOptions o;
                            o.blocked = 1;
                            o.threads = th;
                            o.panel = panel;
                            o.simd = sp;
                            const M::Smith B = M::SmithFormBlocked(Z, A, true, true, o);
                            ++runs;
                            const std::string d = SmithDiff(O, B);
                            check(d.empty(), "(R) " + d + " differs: ell=" + std::to_string(zk.ell) + " k=" + std::to_string(zk.k) + " " +
                                                 std::to_string(sh.first) + "x" + std::to_string(sh.second) + " kind=" + std::to_string(kind) +
                                                 " threads=" + std::to_string(th) + " panel=" + std::to_string(panel) + " simd=" + std::to_string(sp));
                        }
                // the accessors (rows of L, L^{-1}T, R^T, R^{-1}; L B) against the dense transforms
                if (sh.first && sh.second) {
                    M::SmithEngineOptions o;
                    o.threads = 3;
                    const M::SmithFactors F = M::SmithFactor(Z, A, true, o);
                    std::vector<uint32_t> rr, cc;
                    for (uint32_t i = 0; i < A.r; ++i)
                        if (g.below(3) == 0 || i + 1 == A.r) rr.push_back(i);
                    for (uint32_t j = 0; j < A.c; ++j)
                        if (g.below(3) == 0 || j == 0) cc.push_back(j);
                    std::reverse(cc.begin(), cc.end());
                    check(SameMat(M::FactorsLRows(F, rr), Rows(O.L, rr)), "(R) L rows");
                    check(SameMat(M::FactorsLinvTRows(F, rr), Rows(O.LinvT, rr)), "(R) LinvT rows");
                    check(SameMat(M::FactorsRTRows(F, cc), Rows(O.RT, cc)), "(R) RT rows");
                    check(SameMat(M::FactorsRinvRows(F, cc), Rows(O.Rinv, cc)), "(R) Rinv rows");
                    const Mat Bm = Uniform(Z, A.r, 1 + g.below(5), g);
                    check(SameMat(M::FactorsApplyL(F, Bm), MulModRef(Z, O.L, Bm)), "(R) L B");
                    acc += 5;
                }
            }
    }
    std::cout << "  (R) " << mats << " random matrices (9 rings, 13 shapes, 7 kinds), " << runs
              << " blocked runs (threads 1/2/7/16, panels 1/5/32, SIMD paths " << simds.size() << ") == the oracle in sigma, L, L^-1, R, R^-1; "
              << acc << " accessor checks" << std::endl;
}

// ===========================================================================
// (B) PAR-UNIFY (audit PARD2 M1): adversarial matrices >= 300 x 300, large enough that the shared descent pool's regions really
// run (the panel updates, the G normalisation, the input reduction, the accessors, MulMod), at threads 16 and 64, under the
// default work threshold and under threshold 0; rings incl. the Montgomery limit (2^31 - 1, 1) and the near-bound 46337^2
// ===========================================================================
Mat GenTies(const ZMod& Z, uint32_t r, uint32_t c, Rng& g) {   // every entry of valuation exactly 1 (ell times a unit): pure ties
    Mat A(r, c);
    for (auto& x : A.a) {
        uint32_t u = 1 + static_cast<uint32_t>(g.next() % (Z.m - 1));
        if (u % Z.ell == 0) u = Z.add(u, 1);
        x = Z.mul(Z.red(Z.ell), u);
    }
    return A;
}
void ArmB() {
    namespace sp = engine::smithpar;
    struct Case {
        uint32_t ell, k, kind, r, c;   // kind 7 = pure ties (GenTies)
    };
    const std::vector<Case> cases = {{2, 30, 7, 300, 320},         {2, 30, 6, 320, 300},         {2, 30, 1, 310, 310},
                                     {2147483647, 1, 0, 300, 330}, {2147483647, 1, 1, 330, 300}, {46337, 2, 6, 300, 310},
                                     {46337, 2, 7, 310, 300},      {3, 19, 6, 300, 300},         {3, 19, 3, 320, 300}};
    Rng g(20261027 + 300);
    uint32_t runs = 0;
    const sp::Stats s0 = sp::stats();
    for (const Case& cs : cases) {
        const ZMod Z(cs.ell, cs.k);
        const Mat A = cs.kind == 7 ? GenTies(Z, cs.r, cs.c, g) : Gen(Z, cs.kind, cs.r, cs.c, g);
        const M::Smith O = M::SmithFormOracle(Z, A, true, true);
        const Mat C = Uniform(Z, A.c, 3, g);
        const Mat AC = MulModRef(Z, A, C);
        const Mat Bm = Uniform(Z, A.r, 3, g);
        const Mat LB = MulModRef(Z, O.L, Bm);
        std::vector<uint32_t> rr, cc;
        for (uint32_t i = 0; i < A.r; i += 3) rr.push_back(i);
        for (uint32_t j = A.c; j-- > 0;)
            if (j % 4 == 1) cc.push_back(j);
        const Mat Lr = Rows(O.L, rr), LinvTr = Rows(O.LinvT, rr), RTr = Rows(O.RT, cc), Rinvr = Rows(O.Rinv, cc);
        for (uint32_t th : {16u, 64u})
            for (bool zero : {false, true}) {
                const sp::ScopedMinWork mw(zero ? 0 : sp::min_work());
                M::SmithEngineOptions o;
                o.blocked = 1;
                o.threads = th;
                const M::Smith B = M::SmithFormBlocked(Z, A, true, true, o);
                const M::SmithFactors F = M::SmithFactor(Z, A, true, o);
                ++runs;
                const std::string tag = " ell=" + std::to_string(cs.ell) + " k=" + std::to_string(cs.k) + " kind=" + std::to_string(cs.kind) +
                                        " " + std::to_string(cs.r) + "x" + std::to_string(cs.c) + " threads=" + std::to_string(th) +
                                        (zero ? " threshold=0" : " threshold=default");
                const std::string d = SmithDiff(O, B);
                check(d.empty(), "(B) " + d + " differs:" + tag);
                check(SameMat(M::FactorsLRows(F, rr), Lr) && SameMat(M::FactorsLinvTRows(F, rr), LinvTr) &&
                          SameMat(M::FactorsRTRows(F, cc), RTr) && SameMat(M::FactorsRinvRows(F, cc), Rinvr),
                      "(B) the row accessors:" + tag);
                check(SameMat(M::FactorsApplyL(F, Bm), LB), "(B) L B:" + tag);
                {
                    const sp::ScopedMinWork mw2(zero ? 0 : sp::min_work());
                    check(SameMat(M::MulMod(Z, A, C), AC), "(B) MulMod:" + tag);
                }
            }
    }
    const sp::Stats s1 = sp::stats();
    check(s1.regions_parallel > s0.regions_parallel, "(B) non-vacuous: the shared descent pool ran parallel regions");
    std::cout << "  (B) " << cases.size() << " adversarial matrices >= 300 x 300 (rings 2^30, 2^31-1, 46337^2, 3^19; pure ties, valuation "
              << "ladders, low rank with torsion, structured rows), " << runs
              << " blocked runs at threads 16 / 64, threshold default / 0: sigma, L, L^-1, R, R^-1, the row accessors, L B and MulMod == the "
              << "oracle; parallel regions " << (s1.regions_parallel - s0.regions_parallel) << std::endl;
}

// ===========================================================================
// (D) real D2 inputs
// ===========================================================================
M::SmithCapture CaptureRealInputs(M::CallerCapture& callers) {
    M::SmithCapture cap;
    cap.min_entries = 400;
    cap.max_items = 2000;
    M::SetSmithCapture(&cap);
    M::SetCallerCapture(&callers);
    try {
        M::ScopedDescentSmithBlocked oracle(false);
        (void)M::AssemblePairMaps(4, 4, 6);    // tree route (default), ell = 2, 3
        (void)M::AssemblePairMaps(1, 11, 10);  // the ell >= 5 cokernel route
        (void)M::AssemblePairMaps(6, 2, 6);
        {
            integral_descent::ScopedDescentTree dense(false);
            (void)M::AssemblePairMaps(2, 2, 2);   // the dense Iso^(2) route
        }
        (void)P::AssembleDescentPair(23, 2);   // D5c: the tree cusp pair
        {
            integral_descent::ScopedDescentTreePair dense(false);
            (void)P::AssembleDescentPair(11, 2);   // D4: the dense pair with the boundary cone
        }
    } catch (...) {
        M::SetSmithCapture(nullptr);
        M::SetCallerCapture(nullptr);
        throw;
    }
    M::SetSmithCapture(nullptr);
    M::SetCallerCapture(nullptr);
    return cap;
}

// (C) the callers on their captured real inputs, field by field, lever on / off
bool SamePres(const M::Pres& a, const M::Pres& b) {
    return a.ok == b.ok && a.why == b.why && a.Z.ell == b.Z.ell && a.Z.k == b.Z.k && a.c == b.c && a.tors == b.tors && a.free == b.free &&
           SameMat(a.gensT, b.gensT) && SameMat(a.coord, b.coord);
}
bool SameMapRes(const M::MapRes& a, const M::MapRes& b) {
    return a.ok == b.ok && a.why == b.why && a.Z.k == b.Z.k && SameMat(a.F, b.F) && a.rankF == b.rankF && a.ker == b.ker && a.im == b.im &&
           a.coker == b.coker && SameMat(a.Lf, b.Lf) && a.rankL == b.rankL && a.ZL.k == b.ZL.k;
}
void ArmCallers(const M::CallerCapture& cl) {
    uint32_t nb = 0, ni = 0, nref = 0;
    for (const auto& x : cl.bp) {
        M::Pres a, b;
        {
            M::ScopedDescentSmithBlocked on(true);
            a = M::BuildPres(x.Z, x.P, x.rankP, x.Nx, x.c, x.b);
        }
        {
            M::ScopedDescentSmithBlocked off(false);
            b = M::BuildPres(x.Z, x.P, x.rankP, x.Nx, x.c, x.b);
        }
        check(SamePres(a, b), "(C) BuildPres blocked == oracle on a captured input (c=" + std::to_string(x.c) + ", b=" + std::to_string(x.b) +
                                  ", ell=" + std::to_string(x.Z.ell) + ")");
        // the refusal paths: a wrong rank claim on the same input
        if (x.rankP > 0) {
            M::Pres c1, d1;
            {
                M::ScopedDescentSmithBlocked on(true);
                c1 = M::BuildPres(x.Z, x.P, x.rankP - 1, x.Nx, x.c, std::min(x.b + 1, x.c - (x.rankP - 1)));
            }
            {
                M::ScopedDescentSmithBlocked off(false);
                d1 = M::BuildPres(x.Z, x.P, x.rankP - 1, x.Nx, x.c, std::min(x.b + 1, x.c - (x.rankP - 1)));
            }
            check(SamePres(c1, d1) && !c1.ok, "(C) BuildPres refuses identically on a false rank claim");
            ++nref;
        }
        ++nb;
    }
    for (const auto& x : cl.im) {
        M::MapRes a, b;
        {
            M::ScopedDescentSmithBlocked on(true);
            a = M::InducedMap(x.X, x.Y, x.f, x.rankF);
        }
        {
            M::ScopedDescentSmithBlocked off(false);
            b = M::InducedMap(x.X, x.Y, x.f, x.rankF);
        }
        check(SameMapRes(a, b), "(C) InducedMap (incl. LatticeQuotient) blocked == oracle on a captured input (gX=" + std::to_string(x.X.g()) +
                                    ", gY=" + std::to_string(x.Y.g()) + ")");
        ++ni;
    }
    check(nb >= 20 && ni >= 20, "(C) callers captured (" + std::to_string(nb) + " BuildPres, " + std::to_string(ni) + " InducedMap)");
    std::cout << "  (C) " << nb << " BuildPres (+ " << nref << " refusal probes) and " << ni
              << " InducedMap / LatticeQuotient calls on captured real inputs: every precision, exponent, generator, coordinate, F, Lf, "
              << "kernel / image / cokernel and refusal text equal (lever on / off)" << std::endl;
}

void ArmD(const M::SmithCapture& cap) {
    uint32_t n = 0;
    uint64_t maxe = 0;
    for (const auto& it : cap.items) {
        const M::Smith O = M::SmithFormOracle(it.first, it.second, true, true);
        struct Cfg {
            uint32_t th;
            int simd;
        };
        std::vector<Cfg> cfgs = {{1, -1}, {16, -1}};
        for (int sp : SimdPaths()) cfgs.push_back({7, sp});
        for (const Cfg& g : cfgs) {
            M::SmithEngineOptions o;
            o.threads = g.th;
            o.simd = g.simd;
            const M::Smith B = M::SmithFormBlocked(it.first, it.second, true, true, o);
            const std::string d = SmithDiff(O, B);
            check(d.empty(), "(D) " + d + " differs on a captured input " + std::to_string(it.second.r) + "x" + std::to_string(it.second.c) +
                                 " ell=" + std::to_string(it.first.ell) + " threads=" + std::to_string(g.th) + " simd=" + std::to_string(g.simd));
        }
        ++n;
        maxe = std::max<uint64_t>(maxe, uint64_t(it.second.r) * it.second.c);
    }
    check(n >= 50, "(D) at least 50 real inputs captured (" + std::to_string(n) + ")");
    std::cout << "  (D) " << n << " real Smith inputs of D2 / D4 / D5c (largest " << maxe << " entries), full transforms equal at threads 1, 16 and at 7 on every SIMD path"
              << std::endl;
}

// ===========================================================================
// (C) the callers and the pipeline, lever on / off
// ===========================================================================
bool SameTypes(const M::MapTypes& a, const M::MapTypes& b) { return a.ker == b.ker && a.im == b.im && a.coker == b.coker; }
template <class V>
bool SameTypeVec(const V& a, const V& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!SameTypes(a[i], b[i])) return false;
    return true;
}
bool SameMaps(const M::PairResult& a, const M::PairResult& b, std::string& why) {
    if (a.refused != b.refused || a.refusal != b.refusal) return why = "refusal", false;
    if (a.HA != b.HA || a.HB != b.HB || a.HAB != b.HAB) return why = "groups", false;
    if (!SameTypeVec(a.res, b.res) || !SameTypeVec(a.cor, b.cor) || !SameTypeVec(a.j, b.j) || !SameTypeVec(a.dl, b.dl)) return why = "maps", false;
    if (a.exact_spots != b.exact_spots || a.identity_checks != b.identity_checks || a.five_term != b.five_term) return why = "counts", false;
    if (a.pieces.size() != b.pieces.size()) return why = "pieces", false;
    for (size_t i = 0; i < a.pieces.size(); ++i) {
        const auto &x = a.pieces[i], &y = b.pieces[i];
        if (x.ell != y.ell || x.route != y.route || x.K != y.K || x.exact_spots != y.exact_spots || x.identity_checks != y.identity_checks ||
            x.smith_forms != y.smith_forms || x.min_prec != y.min_prec || x.ok != y.ok || x.why != y.why)
            return why = "piece " + std::to_string(x.ell), false;
    }
    return true;
}
bool SamePair(const P::PairResult& a, const P::PairResult& b, std::string& why) {
    if (a.refused != b.refused || a.refusal != b.refusal) return why = "refusal", false;
    if (a.H != b.H || a.Hd != b.Hd || a.Hc != b.Hc) return why = "groups", false;
    if (!SameTypeVec(a.res, b.res) || !SameTypeVec(a.j, b.j) || !SameTypeVec(a.proj, b.proj)) return why = "maps", false;
    if (a.exact_spots != b.exact_spots || a.checks != b.checks || a.closed_form != b.closed_form) return why = "counts", false;
    if (a.pieces.size() != b.pieces.size()) return why = "pieces", false;
    for (size_t i = 0; i < a.pieces.size(); ++i) {
        const auto &x = a.pieces[i], &y = b.pieces[i];
        if (x.ell != y.ell || x.route != y.route || x.exact_spots != y.exact_spots || x.checks != y.checks || x.min_prec != y.min_prec ||
            x.ok != y.ok || x.why != y.why)
            return why = "piece " + std::to_string(x.ell), false;
    }
    return true;
}

void ArmC() {
    struct C3 {
        uint32_t N, M, n;
        bool dense;
    };
    uint32_t cells = 0;
    for (const C3& c : std::vector<C3>{{4, 4, 6, false}, {1, 11, 10, false}, {6, 2, 6, false}, {5, 5, 4, false}, {1, 2, 3, false}, {2, 2, 2, true}}) {
        M::PairResult a, b;
        {
            integral_descent::ScopedDescentTree route(!c.dense);
            {
                M::ScopedDescentSmithBlocked on(true);
                a = M::AssemblePairMaps(c.N, c.M, c.n);
            }
            {
                M::ScopedDescentSmithBlocked off(false);
                b = M::AssemblePairMaps(c.N, c.M, c.n);
            }
        }
        std::string why;
        check(SameMaps(a, b, why), "(C) AssemblePairMaps(" + std::to_string(c.N) + ", " + std::to_string(c.M) + ", " + std::to_string(c.n) +
                                       ") blocked == oracle: " + why);
        check(!a.refused, "(C) the cell is computed, not refused");
        ++cells;
    }
    // the negative controls refuse identically (a perturbed alpha, a false rank claim)
    {
        M::PairOptions o;
        o.test_break_alpha = true;
        M::PairResult a, b;
        {
            M::ScopedDescentSmithBlocked on(true);
            a = M::AssemblePairMaps(4, 2, 2, o);
        }
        {
            M::ScopedDescentSmithBlocked off(false);
            b = M::AssemblePairMaps(4, 2, 2, o);
        }
        std::string why;
        check(SameMaps(a, b, why) && a.refused, "(C) the perturbed alpha refuses identically: " + why);
        o = M::PairOptions();
        o.test_inv_rank_offset = 1;
        {
            M::ScopedDescentSmithBlocked on(true);
            a = M::AssemblePairMaps(1, 4, 2, o);
        }
        {
            M::ScopedDescentSmithBlocked off(false);
            b = M::AssemblePairMaps(1, 4, 2, o);
        }
        check(SameMaps(a, b, why), "(C) the false invariants rank refuses identically: " + why);
        cells += 2;
    }
    for (const auto& c : std::vector<std::pair<uint32_t, uint32_t>>{{23, 2}, {11, 4}, {1, 12}}) {
        for (bool tree : {true, false}) {
            if (!tree && c.first > 11) continue;
            integral_descent::ScopedDescentTreePair route(tree);
            P::PairResult a, b;
            {
                M::ScopedDescentSmithBlocked on(true);
                a = P::AssembleDescentPair(c.first, c.second);
            }
            {
                M::ScopedDescentSmithBlocked off(false);
                b = P::AssembleDescentPair(c.first, c.second);
            }
            std::string why;
            check(SamePair(a, b, why), "(C) AssembleDescentPair(" + std::to_string(c.first) + ", " + std::to_string(c.second) + ", " +
                                           (tree ? "tree" : "dense") + ") blocked == oracle: " + why);
            ++cells;
        }
    }
    std::cout << "  (C) " << cells << " pipeline runs (D2 level pairs on the tree / dense routes incl. ell >= 5, two negative controls, "
              << "D4 / D5c cusp pairs): every group, map type, count and refusal equal with the lever on and off" << std::endl;
}

// ===========================================================================
// (P) one timing line
// ===========================================================================
void ArmP(const M::SmithCapture& cap) {
    const std::pair<ZMod, Mat>* big = nullptr;
    for (const auto& it : cap.items)
        if (!big || uint64_t(it.second.r) * it.second.c > uint64_t(big->second.r) * big->second.c) big = &it;
    if (!big) return;
    const auto t0 = clk::now();
    const M::Smith O = M::SmithFormOracle(big->first, big->second, true, true);
    const double to = since(t0);
    const auto t1 = clk::now();
    const M::Smith B = M::SmithFormBlocked(big->first, big->second, true, true);
    const double tb = since(t1);
    check(SmithDiff(O, B).empty(), "(P) the timed input agrees");
    std::cout << "SMITH_BLOCKED_COST r=" << big->second.r << " c=" << big->second.c << " ell=" << big->first.ell << " rank=" << O.rank()
              << " oracle_s=" << fx(to) << " blocked_s=" << fx(tb) << " threads=" << M::DescentSmithDefaultThreads()
              << " simd=" << M::DescentSmithSimdPath() << std::endl;
}

}  // namespace

uint32_t IntegralDescentSmithBlockedValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = clk::now();
    std::cout << "=== IntegralDescentSmithBlockedValidation (PAR-D2: the blocked Smith form == D2's oracle, bit for bit) ===" << std::endl;
    std::cout << "  lever HECKE_DESCENT_SMITH_BLOCKED: " << (M::DescentSmithBlockedEnabled() ? "on" : "off (rollback)")
              << "; threads=" << M::DescentSmithDefaultThreads() << " simd=" << M::DescentSmithSimdPath() << std::endl;
    try {
        ArmR();
        ArmB();
        M::CallerCapture callers;
        callers.max_entries = uint64_t(1) << 18;
        const M::SmithCapture cap = CaptureRealInputs(callers);
        ArmD(cap);
        ArmCallers(callers);
        ArmC();
        ArmP(cap);
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    std::cout << "IntegralDescentSmithBlockedValidation: checks=" << g_checks << " failing=" << g_fail << " wall=" << fx(since(t0)) << " s"
              << std::endl;
    return g_fail == 0 ? 0u : 1u;
}
