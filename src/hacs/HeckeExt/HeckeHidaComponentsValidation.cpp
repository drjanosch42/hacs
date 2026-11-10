// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 15, WP-6 -- HeckeHidaComponentsValidation (standard).
//
// HeckeExt/HeckeHidaComponents (slot-14 WP-14c in C++) against its oracles.
// The out-of-tree oracles are docs/implementation/slot14/wpc14/
// {bernoulli_depth, eis_depth_join, hida_types}.py (self-tests 5199 / 399 / 150).
//
//   (A) the bignum-free B_k/k mod ell^K:
//       (A1) both routes against the frozen exact-rational oracle
//            (HeckeHidaComponentsOracle.inc, sympy): 611 values, K <= 3 on 12
//            primes and near the 2^64 limit (5^27, 7^22, 691^6, 1009^6 -- the
//            128-bit-modulus path of the direct route);
//       (A2) Kummer-Mahler = direct on every even k <= 1200 at eight primes, K = 4;
//       (A3) the irregular indices below 160 = the classical table; 691 -> {12, 200};
//       (A4) k*: lambda = 1 on every classical pair and at 691; k* refines with prec;
//       (A5) the depth law L_k = v(B_k/k) = 1 + v(k - k*) on every k <= 6000 of the
//            classes at 37, 59, 67, 103 (prec 4); the first depth-2 weights
//            (37, 284), (103, 228), (59, 914), (67, 3292) and every weight before
//            them at depth 1; the first depth-3 weight at 37, k = 37580, on both
//            routes; bernoulli_depth.py's predict: 37 -> n = 282, 103 -> n = 226,
//            59 -> n = 912 below 1000;
//       (A6) the named refusals (work budget, ell^K >= 2^64, (ell - 1) | k).
//   (B) synthetic: the synthetic division, the square root, the g = 1 reading and
//       the g = 2 kinds (split / inert / ramified) on planted polynomials
//       (eis_depth_join.py (A), hida_types.py intra_block cases).
//   (C) (10, 691) over Z/691^2: D = 3, L_coh = 1 = v(B_12/12), law agree; the
//       cusp root c_p = tau(p) mod 691^2 at p = 2, 3, 5, 7 (Ramanujan).
//   (D) (U1) at the small irregular cells over Z/ell^3: (30, 37), (66, 37),
//       (22, 103), (124, 103), (42, 59), (56, 67) -> L_coh = v(B_k/k) = L_pred = 1,
//       law agree; the regular controls (10, 37) (class 12 regular at 37) and
//       (30, 41) -> D = 1, L = 0.
//   (E) the g = 2 ramified cells of [U] prop:u-loewy(4): (30, 67), (26, 131) over
//       Z/ell^3: one ordinary non-Eisenstein D = 4 block, verdict
//       one_component_ramified, v(disc c_2) = 1, level 1/2; at (30, 67) the cusp
//       factor of T_2 is c_{32,2} = X^2 - 39960 X - 2235350016 mod 67^3 ([U]'s
//       example; S_32 has dimension 2).
//   (F) (U3) on (30, 67) by the Z/ell^K Hecke-algebra route (slot-14 WP-6) in
//       process, K <= 4: the ramified block is stable, eps = 0, A_1 = A_K/67 =
//       (1,1).  PROVED there ([U] prop:u-maxorder, v(disc) = 1): a failure is a
//       DEFECT (WP-14c Sec. 5 item 5).
//   (W) the module runner: Native and Hida (hida::RunModuleCell) give identical
//       HIDA_COMPONENT lines at (30, 37).
//   (ENV) HECKE_HIDA_COMPONENTS_CELLS="n:ell[:K],..." (read-only, not an emitter;
//       K defaults to 3): the readings over the listed cells, one HIDA_COMPONENT
//       (+ HIDA_INTRA) line each, one EFFECTIVE line; gated on eis_root and on
//       law not in {LAW_FAIL, COH_FAIL, BOUND_FAIL}.  The L = 2 cells (282, 37)
//       and (226, 103) of [U] sec:u-test run here (minutes each).
// ===========================================================================

#include "HeckeHidaComponentsValidation.h"

#include "HeckeHidaComponents.h"
#include "HeckeRingHeckeAlgebra.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../Util/ValidationCheck.h"
#include "../Util/CellList.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace hecke_ext::hidacomp;

oc_util::Checker Check("  FAIL: ", false);

uint64_t P64(uint64_t b, uint32_t e) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < e; ++i) r *= b;
    return r;
}

uint64_t Mod(int64_t x, uint64_t q) {
    const int64_t r = x % static_cast<int64_t>(q);
    return static_cast<uint64_t>(r < 0 ? r + static_cast<int64_t>(q) : r);
}

std::string S(uint64_t a, uint64_t b) { return "(" + std::to_string(a) + ", " + std::to_string(b) + ")"; }

struct OracleRow {
    uint64_t ell;
    uint32_t K;
    uint64_t k;
    uint64_t r;
};
const OracleRow kOracle[] = {
#include "HeckeHidaComponentsOracle.inc"
};

// ---- (A) -------------------------------------------------------------------------

void ArmA1() {
    std::cout << "(A1) both routes against the frozen exact-rational oracle\n";
    std::map<std::pair<uint64_t, uint32_t>, uint64_t> wmax;
    for (const OracleRow& o : kOracle) {
        auto& w = wmax[{o.ell, o.K}];
        w = std::max(w, o.k);
    }
    uint32_t n = 0, bad = 0;
    for (const auto& kv : wmax) {
        const uint64_t ell = kv.first.first;
        const uint32_t K = kv.first.second;
        // one direct table per (ell, K): Q = K + 1 + max v(k)
        uint32_t vmax = 0;
        for (const OracleRow& o : kOracle)
            if (o.ell == ell && o.K == K) vmax = std::max(vmax, ValU64(o.k, ell));
        const BernoulliTable T(ell, K + 1 + vmax, kv.second);
        std::map<uint64_t, ClassSeries> series;
        for (const OracleRow& o : kOracle) {
            if (o.ell != ell || o.K != K) continue;
            const uint64_t k0 = o.k % (ell - 1);
            if (!series.count(k0)) series.emplace(k0, MakeClassSeries(ell, k0, K));
            const uint64_t m = BernoulliOverKFromSeries(series.at(k0), o.k);
            const uint64_t d = T.BOverK(o.k, K);
            ++n;
            if (m != o.r || d != o.r) {
                ++bad;
                Check(false, "(A1) " + S(ell, o.k) + " K=" + std::to_string(K) + ": oracle " + std::to_string(o.r) +
                                 " mahler " + std::to_string(m) + " direct " + std::to_string(d));
            }
        }
    }
    Check(bad == 0 && n == sizeof(kOracle) / sizeof(kOracle[0]), "(A1) all " + std::to_string(n) + " oracle values on both routes");
    // the one-call entry points agree with the table
    Check(BernoulliOverKModPower(284, 37, 3) == BernoulliOverKDirect(284, 37, 3), "(A1) (37, 284): entry points agree");
    std::cout << "  " << n << " oracle values, " << bad << " mismatches\n";
}

void ArmA2() {
    std::cout << "(A2) Kummer-Mahler = direct on every even k <= 1200, K = 4\n";
    for (uint64_t ell : {5, 7, 11, 13, 37, 59, 67, 103}) {
        const uint32_t K = 4;
        uint32_t vmax = 0;
        for (uint64_t k = 2; k <= 1200; k += 2) vmax = std::max(vmax, ValU64(k, ell));
        const BernoulliTable T(ell, K + 1 + vmax, 1200);
        std::map<uint64_t, ClassSeries> series;
        uint32_t n = 0, bad = 0;
        for (uint64_t k = 2; k <= 1200; k += 2) {
            const uint64_t k0 = k % (ell - 1);
            if (k0 == 0) continue;
            if (!series.count(k0)) series.emplace(k0, MakeClassSeries(ell, k0, K));
            ++n;
            if (BernoulliOverKFromSeries(series.at(k0), k) != T.BOverK(k, K)) ++bad;
        }
        Check(bad == 0, "(A2) ell = " + std::to_string(ell) + ": " + std::to_string(bad) + " of " + std::to_string(n) + " differ");
    }
}

void ArmA3() {
    std::cout << "(A3) irregular indices\n";
    const std::map<uint64_t, std::vector<uint64_t>> classical = {
        {37, {32}}, {59, {44}}, {67, {58}}, {101, {68}}, {103, {24}}, {131, {22}}, {149, {130}}, {157, {62, 110}}};
    for (uint64_t ell = 5; ell < 160; ++ell) {
        bool prime = ell % 2 == 1;
        for (uint64_t d = 3; d * d <= ell && prime; d += 2) prime = ell % d != 0;
        if (!prime) continue;
        const auto it = classical.find(ell);
        const std::vector<uint64_t> want = it == classical.end() ? std::vector<uint64_t>{} : it->second;
        Check(IrregularIndices(ell) == want, "(A3) irregular indices at " + std::to_string(ell));
    }
    Check(IrregularIndices(691) == std::vector<uint64_t>({12, 200}), "(A3) 691 -> {12, 200}");
}

void ArmA4() {
    std::cout << "(A4) k*: lambda = 1, refinement\n";
    const std::vector<std::pair<uint64_t, uint64_t>> pairs = {{37, 32}, {59, 44}, {67, 58}, {101, 68}, {103, 24}, {131, 22},
                                                              {149, 130}, {157, 62}, {157, 110}, {691, 12}, {691, 200}};
    for (const auto& pr : pairs) {
        const KStar a = ComputeKStar(pr.first, pr.second, 4);
        Check(a.irregular && a.lambda_one, "(A4) " + S(pr.first, pr.second) + ": irregular, lambda = 1");
        Check(a.k_star % pr.first != 1, "(A4) " + S(pr.first, pr.second) + ": k* != 1 mod ell");
        const KStar b = ComputeKStar(pr.first, pr.second, 5);
        Check(b.lambda_one && b.k_star % a.modulus == a.k_star, "(A4) " + S(pr.first, pr.second) + ": k* at prec 5 refines prec 4");
        std::cout << "  k*" << S(pr.first, pr.second) << " = " << a.k_star << " mod " << pr.first << "^3\n";
    }
    Check(!ComputeKStar(37, 12, 4).irregular, "(A4) (37, 12) is a regular class");
}

void ArmA5() {
    std::cout << "(A5) the depth law at 37, 59, 67, 103\n";
    const std::vector<std::pair<uint64_t, uint64_t>> cls = {{37, 32}, {59, 44}, {67, 58}, {103, 24}};
    const std::map<uint64_t, uint64_t> first2 = {{37, 284}, {103, 228}, {59, 914}, {67, 3292}};
    for (const auto& pr : cls) {
        const uint64_t ell = pr.first, k0 = pr.second;
        const ClassSeries s = MakeClassSeries(ell, k0, 4);
        uint32_t n = 0, bad = 0;
        uint64_t first_depth2 = 0;
        for (uint64_t k = k0; k <= 6000; k += ell - 1) {
            const uint32_t v = ValCapped(BernoulliOverKFromSeries(s, k), ell, 4);
            const DepthPred d = PredictedDepth(k, ell, 4);
            ++n;
            const bool ok = d.lower_bound ? v >= d.L : v == d.L;
            if (!ok) {
                ++bad;
                std::cout << "  LAW_FAIL " << S(ell, k) << " v=" << v << " L_pred=" << d.L << "\n";
            }
            if (!first_depth2 && v >= 2) first_depth2 = k;
        }
        Check(bad == 0, "(A5) ell = " + std::to_string(ell) + ": the law on " + std::to_string(n) + " weights");
        Check(first_depth2 == first2.at(ell), "(A5) ell = " + std::to_string(ell) + ": first depth-2 weight " +
                                                 std::to_string(first_depth2) + " == " + std::to_string(first2.at(ell)));
        Check(ValBernoulliOverK(first2.at(ell), ell, 4).v == 2 && PredictedDepth(first2.at(ell), ell, 4).L == 2,
              "(A5) " + S(ell, first2.at(ell)) + ": v = L_pred = 2");
    }
    // the first depth-3 weight at 37 (bernoulli_depth.py: k = 37580, exact and Voronoi mod 37^4), both routes
    const KStar ks = ComputeKStar(37, 32, 4);
    const uint64_t first3 = 32 + 36 * (ks.x_star % (37 * 37));
    Check(first3 == 37580, "(A5) the first depth-3 weight at 37 is k = 37580, got " + std::to_string(first3));
    Check(ValBernoulliOverK(37580, 37, 4).v == 3 && PredictedDepth(37580, 37, 4).L == 3, "(A5) (37, 37580): v = L_pred = 3 (Mahler)");
    Check(ValCapped(BernoulliOverKDirect(37580, 37, 4), 37, 4) == 3, "(A5) (37, 37580): v = 3 (direct, W = 37580)");
    // bernoulli_depth.py predict ELL 1000: the cells n <= 1000 of depth >= 2
    for (const auto& pr : std::vector<std::pair<uint64_t, uint64_t>>{{37, 282}, {103, 226}, {59, 912}}) {
        std::vector<uint64_t> got;
        for (uint64_t k0 : IrregularIndices(pr.first))
            for (uint64_t k = k0; k <= 1002; k += pr.first - 1)
                if (PredictedDepth(k, pr.first, 4).L >= 2) got.push_back(k - 2);
        Check(got == std::vector<uint64_t>({pr.second}), "(A5) predict " + std::to_string(pr.first) + " 1000 -> n = " +
                                                             std::to_string(pr.second));
    }
}

template <class F>
bool Throws(F f, const std::string& needle) {
    try {
        f();
    } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

void ArmA6() {
    std::cout << "(A6) named refusals\n";
    Check(Throws([] { BernoulliOverKModPower(4, 1000003, 2, uint64_t(1) << 20); }, "refused"),
          "(A6) a work above the budget is refused by name");
    Check(Throws([] { BernoulliOverKModPower(6, 5, 28); }, "ell^K >= 2^64"), "(A6) 5^28 >= 2^64 is refused by name");
    Check(Throws([] { BernoulliOverKModPower(36, 37, 2); }, "(ell - 1) | k"), "(A6) (ell - 1) | k is refused");
    Check(Throws([] { BernoulliOverKModPower(12, 9, 2); }, "prime"), "(A6) a composite ell is refused");
    Check(Throws([] { BernoulliOverKModPower(12, 1000001, 2); }, "prime"), "(A6) a composite ell > 2^20 is refused (Miller-Rabin)");
    // a direct call beyond the 2 GiB beta_ cap is refused by name, before any allocation, whatever the budget
    Check(Throws([] { BernoulliTable t(5, 3, BernoulliTable::kMaxBetaBytes / 16, ~uint64_t(0)); (void)t; },
                 "table memory"),
          "(A6) a beta_ table above the memory cap is refused by name");
    Check(BernoulliTable::PredictedWork(5, 30, 100) == 128 * BernoulliTable::PredictedWork(5, 2, 100) ||
              BernoulliTable::PredictedWork(5, 30, 100) > BernoulliTable::PredictedWork(5, 2, 100),
          "(A6) a 128-bit-modulus table is predicted dearer than a 64-bit one");
}

// ---- (B) --------------------------------------------------------------------------

Poly FromRoots(const std::vector<uint64_t>& roots, uint64_t q) {
    Poly f{1};
    for (uint64_t r : roots) {
        Poly g(f.size() + 1, 0);
        for (size_t i = 0; i < f.size(); ++i) {
            g[i + 1] = (g[i + 1] + f[i]) % q;
            g[i] = static_cast<uint64_t>((g[i] + (unsigned __int128)(q - r % q) * f[i]) % q);
        }
        f = g;
    }
    return f;
}

void ArmB() {
    std::cout << "(B) synthetic\n";
    for (const auto& t : std::vector<std::vector<uint64_t>>{{37, 10, 2}, {103, 9, 2}, {691, 6, 1}, {37, 10, 0}, {7, 3, 3}}) {
        const uint64_t ell = t[0], q = P64(ell, static_cast<uint32_t>(t[1]));
        const uint64_t e = 1 + 1234567, c = (e + P64(ell, static_cast<uint32_t>(t[2])) * 5) % q;
        const Poly cp = FromRoots({e, c, c}, q);
        const auto dv = SyntheticDivide(cp, e % q, q);
        Check(dv.second == 0 && dv.first == FromRoots({c, c}, q), "(B) synthetic division at " + std::to_string(ell));
        const auto sr = MonicSqrt(dv.first, q);
        Check(sr.second && sr.first == FromRoots({c}, q), "(B) the square root at " + std::to_string(ell));
        Check(ValCapped((e + q - c) % q, ell, static_cast<uint32_t>(t[1])) == std::min<uint64_t>(t[2], t[1]),
              "(B) the g = 1 reading v(e - c) at " + std::to_string(ell));
    }
    Check(SyntheticDivide(FromRoots({5}, P64(37, 4)), 7, P64(37, 4)).second != 0, "(B) a non-root is flagged");
    Check(!MonicSqrt(Poly{3, 0, 1}, P64(7, 3)).second, "(B) x^2 + 3 is not a monic square mod 7^3");
    const Poly sq = FromRoots({11, 11, 29, 29}, P64(37, 5));
    const auto r = MonicSqrt(sq, P64(37, 5));
    Check(r.second && r.first == FromRoots({11, 29}, P64(37, 5)), "(B) a degree-4 square root");
}

// ---- (C) - (F), (W) ------------------------------------------------------------------

void ArmC() {
    std::cout << "(C) (10, 691) over Z/691^2\n";
    const ComponentCell c = ComputeComponentCell(10, 691, 2);
    std::cout << "  " << ComponentLine(c) << "  (" << c.seconds << " s)\n";
    Check(c.eis.present && c.eis.free && c.eis.D == 3 && c.eis.g == 1, "(C) (10, 691): the Eisenstein block, free, D = 3");
    Check(c.eis.eis_root && c.eis.square_ok && c.eis.kind == "exact", "(C) (10, 691): e_p a root, h_p = (X - c_p)^2");
    Check(c.eis.L == 1 && !c.eis.capped && c.v_bk.v == 1 && c.pred.L == 1 && c.law == "agree",
          "(C) (10, 691): L_coh = v(B_12/12) = L_pred = 1, law agree");
    const uint64_t q = 691ull * 691;
    const std::map<uint64_t, int64_t> tau = {{2, -24}, {3, 252}, {5, 4830}, {7, -16744}};
    for (size_t i = 0; i < c.eis.primes.size(); ++i) {
        const uint64_t p = c.eis.primes[i];
        const Poly& f = c.eis.cusp_factor[i];
        Check(f.size() == 2 && (q - f[0]) % q == Mod(tau.at(p), q), "(C) (10, 691): c_" + std::to_string(p) + " = tau(" +
                                                                       std::to_string(p) + ") mod 691^2");
    }
}

void ArmD() {
    std::cout << "(D) (U1) at the small irregular cells over Z/ell^3, and regular controls\n";
    for (const auto& cell : std::vector<std::pair<uint32_t, uint64_t>>{{30, 37}, {66, 37}, {22, 103}, {124, 103}, {42, 59}, {56, 67}}) {
        const ComponentCell c = ComputeComponentCell(cell.first, cell.second, 3);
        std::cout << "  " << ComponentLine(c) << "  (" << c.seconds << " s)\n";
        const std::string tag = "(D) " + S(cell.first, cell.second);
        Check(c.eis.free && c.eis.D == 3 && c.eis.eis_root && c.eis.square_ok, tag + ": a free D = 3 Eisenstein block, e_p a root");
        Check(c.eis.L == 1 && c.v_bk.v == 1 && !c.v_bk.lower_bound && c.pred.L == 1 && c.law == "agree",
              tag + ": L_coh = v(B_k/k) = L_pred = 1, law agree");
    }
    for (const auto& cell : std::vector<std::pair<uint32_t, uint64_t>>{{10, 37}, {30, 41}}) {
        const ComponentCell c = ComputeComponentCell(cell.first, cell.second, 3);
        std::cout << "  " << ComponentLine(c) << "  (" << c.seconds << " s)\n";
        Check(c.eis.D == 1 && c.eis.L == 0 && c.v_bk.v == 0 && c.pred.L == 0 && c.law == "agree" && c.eis.eis_root,
              "(D) " + S(cell.first, cell.second) + ": regular class, D = 1, L = 0, law agree");
    }
}

void ArmE() {
    std::cout << "(E) the g = 2 ramified cells of prop:u-loewy(4)\n";
    for (const auto& cell : std::vector<std::pair<uint32_t, uint64_t>>{{30, 67}, {26, 131}}) {
        const ComponentCell c = ComputeComponentCell(cell.first, cell.second, 3);
        std::cout << "  " << ComponentLine(c) << "  (" << c.seconds << " s)\n";
        const std::string tag = "(E) " + S(cell.first, cell.second);
        Check(c.intra.size() == 1, tag + ": one ordinary non-Eisenstein D = 4 block");
        Check(c.eis.D == 1 && c.law == "agree", tag + ": the Eisenstein block is D = 1 (L = 0)");
        if (c.intra.size() != 1) continue;
        const IntraBlock& b = c.intra[0];
        std::cout << "  " << IntraLine(c, b) << "\n";
        Check(b.square_ok && b.verdict == "one_component_ramified" && b.level2 == 1,
              tag + ": one_component_ramified, level 1/2");
        Check(!b.disc_v.empty() && b.disc_v[0] == 1, tag + ": v(disc c_2) = 1");
        if (cell.second == 67) {
            const uint64_t q = P64(67, 3);
            const Poly want = {Mod(-2235350016LL, q), Mod(-39960, q), 1};
            Check(b.cusp_factor[0] == want, tag + ": c_2 = X^2 - 39960 X - 2235350016 mod 67^3");
            Check(b.system[0] == 14, tag + ": the double root of c-bar_{32,2} is 14");
        }
    }
}

void ArmF() {
    std::cout << "(F) (U3) on (30, 67) by the Z/ell^K Hecke-algebra route, K <= 4\n";
    using namespace hecke_ext::ringalg;
    LoewyOptions o;
    o.kmax = 4;
    const LoewyCell c = ComputeRingLoewyCell(30, 67, o);
    const LoewyBlock* rb = nullptr;
    for (const LoewyBlock& b : c.blocks)
        if (!b.system.empty() && b.system[0] == 14 && b.D == 4) rb = &b;
    Check(rb != nullptr, "(F) (30, 67): the ramified block (a_2 = 14, D = 4) exists");
    if (!rb) return;
    std::cout << "  " << LoewyRingLine(c, *rb) << "\n";
    const AlgebraReading* T = rb->last_T();
    const bool held = rb->verdict == "stable" && rb->eps == 0 && rb->A1.hilbert == std::vector<uint32_t>({1, 1}) && T &&
                      T->hilbert == std::vector<uint32_t>({1, 1});
    if (!held) std::cout << "  DEFECT: (U3) fails on (30, 67), where prop:u-maxorder proves it\n";
    Check(held, "(F) (30, 67): (U3) held -- stable, eps = 0, A_1 = A_K/67 = (1,1)");
}

void ArmW() {
    std::cout << "(W) Native and Hida module runners\n";
    ComponentOptions o;
    const ComponentCell a = ComputeComponentCell(30, 37, 3, o);
    o.route = hecke_ext::ringalg::ModuleRoute::Hida;
    const ComponentCell b = ComputeComponentCell(30, 37, 3, o);
    Check(ComponentLine(a) == ComponentLine(b) && a.exps == b.exps, "(W) (30, 37): identical lines on both runners");
}

// ---- (ENV) ------------------------------------------------------------------------

void ArmEnv(const char* spec) {
    std::vector<std::vector<uint64_t>> cells;
    for (const oc_util::cells::Token& t : oc_util::cells::Tokenize(spec)) {
        if (t.text.empty()) continue;
        ENGINE_CHECK_MSG(t.Fits(2, 3), ("HECKE_HIDA_COMPONENTS_CELLS: cell '" + t.text + "' is not n:ell[:K]").c_str());
        std::vector<uint64_t> f = t.fields;
        if (f.size() == 2) f.push_back(3);
        cells.push_back(f);
    }
    std::cout << oc_util::cells::EffectiveLine("HeckeHidaComponentsValidation", "CELLS", cells.size()) << "\n";
    for (const auto& f : cells) {
        const ComponentCell c = ComputeComponentCell(static_cast<uint32_t>(f[0]), f[1], static_cast<uint32_t>(f[2]));
        std::cout << ComponentLine(c) << " seconds=" << c.seconds << "\n";
        for (const IntraBlock& b : c.intra) std::cout << IntraLine(c, b) << "\n";
        Check(c.eis.eis_root, "(ENV) " + S(f[0], f[1]) + ": e_p is a root at every p");
        Check(c.law != "LAW_FAIL" && c.law != "COH_FAIL" && c.law != "BOUND_FAIL", "(ENV) " + S(f[0], f[1]) + ": law " + c.law);
    }
}

} // namespace

uint32_t HeckeHidaComponentsValidation() {
    Check.reset();
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeHidaComponentsValidation (slot-15 WP-6: [U] (U1)-(U3) readings, bignum-free B_k/k mod ell^K)\n";
    if (const char* env = std::getenv("HECKE_HIDA_COMPONENTS_CELLS")) {
        ArmEnv(env);
    } else {
        ArmA1();
        ArmA2();
        ArmA3();
        ArmA4();
        ArmA5();
        ArmA6();
        ArmB();
        ArmC();
        ArmD();
        ArmE();
        ArmF();
        ArmW();
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeHidaComponentsValidation: " << Check.checks << " checks, " << Check.fails << " failed, " << s << " s\n";
    return Check.fails ? 1u : 0u;
}
