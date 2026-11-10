// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, LES-3) HeckeLevelInfResValidation -- see the header, [C] v1.23.0-pre.2 sec:nebles and
// docs/implementation/slot23/LES3_predictions.md (committed before the first run).
#include "HeckeLevelInfResValidation.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "HeckeGammaH.h"
#include "HeckeGammaHTower.h"
#include "HeckeLevelCone.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelInfRes.h"
#include "HeckeLevelN.h"
#include "HeckeLevelPair.h"
#include "../LocalSystemSL2Z.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Util/Timing.h"
#include "../../ModArithmetic/ScalarMod.h"

namespace {

using oc_util::Seconds;
using engine::scalar::gcd;
using engine::scalar::powmod;

using hecke::Mat22;
using namespace hecke::level;
using namespace hecke::level::les;
namespace tower = hecke::level::tower;

struct Checker {
    uint32_t checks = 0, fails = 0;
    bool quiet = false;   // a scratch checker (negative controls): failures are expected and not printed
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (!quiet && fails <= 80) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

struct Totals {
    uint32_t ab = 0, zp = 0, hida1 = 0, tower = 0, gamma = 0, merel = 0, squares = 0, square_bad = 0, escapes = 0;
    double max_cell_s = 0;
    std::string max_cell;
    void timed(const std::string& tag, double s) { if (s > max_cell_s) { max_cell_s = s; max_cell = tag; } }
};

// x with x = r1 mod m1, x = r2 mod m2 (coprime moduli), 0 <= x < m1 m2.
uint64_t Crt(uint64_t r1, uint64_t m1, uint64_t r2, uint64_t m2) {
    for (uint64_t x = r1 % m1; x < m1 * m2; x += m1)
        if (x % m2 == r2 % m2) return x;
    ENGINE_CHECK_MSG(false, "Crt: no solution");
    return 0;
}

std::unique_ptr<GammaHLevelN> GH(uint64_t N, const std::function<bool(uint64_t)>& pred) {
    std::vector<uint64_t> elems;
    for (uint64_t u = 1; u < N; ++u)
        if (gcd<uint64_t>(u, N) == 1 && pred(u)) elems.push_back(u);
    if (elems.empty()) elems.push_back(1);
    return std::make_unique<GammaHLevelN>(N, elems);
}
std::unique_ptr<GammaHLevelN> G0H(uint64_t N) { return GH(N, [](uint64_t) { return true; }); }

CharacterDatum Chi(uint64_t N, uint32_t Q, const std::function<uint32_t(uint64_t)>& f, const std::string& label) {
    CharacterDatum c;
    c.N = N;
    c.table.assign(N, 0);
    for (uint64_t u = 0; u < N; ++u)
        if (N == 1 || gcd<uint64_t>(u, N) == 1) c.table[u] = f(u) % Q;
    c.label = label;
    return c;
}
CharacterDatum Triv(uint64_t N, uint32_t Q) { return Chi(N, Q, [](uint64_t) { return 1u; }, "1"); }
CharacterDatum Chi4mod5(uint64_t N) { return Chi(N, 5, [](uint64_t u) { return static_cast<uint32_t>(u % 5); }, "chi_4"); }
CharacterDatum OmegaPow(uint64_t N, uint32_t p, uint32_t i) {   // u -> (u mod p)^i in F_p, p | N
    return Chi(N, p, [p, i](uint64_t u) { return static_cast<uint32_t>(powmod<uint64_t>(u % p, i, p)); }, "omega^" + std::to_string(i));
}
CharacterDatum TameChi(uint64_t N, const tower::ZK& R, int64_t e) {
    CharacterDatum c;
    c.N = N;
    c.table = tower::TameCharacterTable(N, R, e);
    c.label = "omega^" + std::to_string(e);
    return c;
}

const SimplicialComplex& TheK() {
    static std::unique_ptr<SimplicialComplex> K;
    static std::once_flag once;
    std::call_once(once, [] { K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation())); });
    return *K;
}

std::string Six(uint32_t h0A, uint32_t h1A, uint32_t h0B, uint32_t h1B, uint32_t rk, uint32_t H1, uint32_t H2) {
    std::ostringstream os;
    os << h0A << " " << h1A << " " << h0B << " " << h1B << " " << rk << " " << H1 << " " << H2;
    return os.str();
}

// ---- the cyclic-step gates (prop:cycliccone (3), prop:levelinfres (1), (2), (5)) ----------------------------------------
// G = <sigma> = Z/p acting on H^1(B), H^0(B) = H^0(A) of dimension c0 with trivial action (c0 = c on the Gamma_1 steps, 0 / 1 on
// the tower).  Matrices over F_p: res (h1B x h1A), cor (h1A x h1B), Rs (h1B x h1B).
struct CyclicOut {
    std::vector<uint32_t> jt;
    uint32_t rank_res = 0, inv = 0, coinv = 0, hhat1 = 0, hhat2 = 0, rank_cor = 0;
    bool ok = true;
};
CyclicOut CyclicGates(Checker& g, const std::string& tag, uint32_t p, uint32_t h1A, uint32_t h1B, uint32_t c0, const DMat& res,
                      const DMat& cor, const DMat& Rs) {
    CyclicOut o;
    const tower::ZK F(p, 1);
    const uint32_t h = h1B;
    std::vector<uint32_t> ranks;
    o.jt = JordanTypeUnipotent(Rs, p, &ranks);
    uint32_t blocks = 0, total = 0, other = 0;
    for (size_t k = 1; k < o.jt.size(); ++k) {
        blocks += o.jt[k];
        total += static_cast<uint32_t>(k) * o.jt[k];
        if (k != 1 && k != p) other += o.jt[k];
    }
    const uint32_t jp = o.jt.size() > p ? o.jt[p] : 0, j1 = o.jt.size() > 1 ? o.jt[1] : 0;
    const bool jok = (h1A >= c0) && jp == h1A - c0 && j1 == c0 && other == 0 && total == h;
    o.ok &= jok;
    g.check(jok, tag + ": Jordan type of R_sigma on H^1(B) = " + JordanText(o.jt) + ", predicted J_" + std::to_string(p) + "^" +
                     std::to_string(h1A - c0) + "+J_1^" + std::to_string(c0) + " (prop:cycliccone (3))");
    // the norm N = sum_{i < p} R^i and R^p = 1
    DMat N(h, h), P = Identity(h);
    for (uint32_t i = 0; i < p; ++i) {
        N = MatAdd(F, N, P);
        P = MatMul(F, P, Rs);
    }
    g.check(Differ(P, Identity(h)) == 0, tag + ": R_sigma^p = 1 on H^1(B)");
    const DMat t = MatSub(F, Rs, Identity(h));
    const uint32_t rank_t = ranks.size() > 1 ? ranks[1] : 0, rank_N = RankModEll(N, p);
    o.inv = h - rank_t;
    o.coinv = h - rank_t;   // dim M_G = dim M / (sigma - 1) M
    o.hhat1 = (h - rank_N) - rank_t;   // ker N / im(sigma - 1)
    o.hhat2 = (h - rank_t) - rank_N;   // ker(sigma - 1) / im N
    g.check(o.inv == h1A && o.coinv == h1A, tag + ": dim H^1(B)^G = dim H^1(B)_G = h1A = " + std::to_string(h1A) + " (got " +
                                                 std::to_string(o.inv) + ")");
    g.check(o.hhat1 == c0 && o.hhat2 == c0, tag + ": dim Hhat^1 = dim Hhat^2 (G; H^1(B)) = " + std::to_string(o.hhat1) + ", " +
                                                std::to_string(o.hhat2) + ", predicted " + std::to_string(c0) + " (prop:levelinfres (2))");
    o.rank_res = RankModEll(res, p);
    g.check(o.rank_res + c0 == h1A, tag + ": rank res^1 = h1A - c = " + std::to_string(o.rank_res));
    g.check(IsZero(MatMul(F, t, res)), tag + ": (R_sigma - 1) res^1 = 0 (res lands in the invariants)");
    g.check(Differ(MatMul(F, res, cor), N) == 0, tag + ": res^1 cor^1 = N_G on H^1(B) (prop:levelinfres (1), (5))");
    g.check(IsZero(MatMul(F, cor, res)), tag + ": cor^1 res^1 = [A:B] = p = 0");
    g.check(IsZero(MatMul(F, cor, t)), tag + ": cor^1 (R_sigma - 1) = 0");
    o.rank_cor = RankModEll(cor, p);
    g.check(o.rank_cor == h1A, tag + ": rank cor^1 = h1A (cor^1 : H^1(B)_G -> H^1(A) an isomorphism)");
    g.check(o.inv - o.rank_res == c0, tag + ": dim H^1(B)^G - rank res^1 = dim H^2(G; H^0(B)) = " + std::to_string(c0) +
                                          " (eq:levelfiveterm)");
    return o;
}

// ---- (AB) Route A = Route B -------------------------------------------------------------------------------------------
void ArmAB(Checker& g, Totals& tot) {
    struct Cell {
        std::string tag;
        const CosetLevel* A;
        const CosetLevel* B;
        CharacterDatum cA, cB;
        uint32_t ell, n;
        uint32_t v[7];
    };
    const FullLevel Gam;
    const auto L5 = G0H(5), L13 = G0H(13), L25 = G0H(25), L11 = G0H(11);
    const auto G111 = GH(11, [](uint64_t u) { return u % 11 == 1; });
    const auto G15 = GH(5, [](uint64_t u) { return u % 5 == 1; });
    const auto G15_25 = GH(25, [](uint64_t u) { return u % 5 == 1; });
    const CongruenceLevel P5 = CongruenceLevel::Principal(5);
    const GammaHLevelN Tt2 = GammaHLevelN::PrimeToPart(25, 5);
    const tower::ZK R5(5, 1);
    std::vector<Cell> cells;
    cells.push_back({"G0(5) < G", &Gam, L5.get(), Triv(1, 5), Triv(5, 5), 5, 4, {0, 1, 1, 5, 1, 1, 4}});
    cells.push_back({"G0(13) < G", &Gam, L13.get(), Triv(1, 7), Triv(13, 7), 7, 2, {0, 1, 0, 8, 1, 0, 7}});
    cells.push_back({"G0(25) < G0(5) chi=1", L5.get(), L25.get(), Triv(5, 5), Triv(25, 5), 5, 2, {0, 4, 0, 16, 3, 1, 13}});
    cells.push_back({"G0(25) < G0(5) chi=chi_4", L5.get(), L25.get(), Chi4mod5(5), Chi4mod5(25), 5, 1, {0, 2, 0, 10, 1, 1, 9}});
    cells.push_back({"G1(11) < G0(11)", L11.get(), G111.get(), Triv(11, 5), Triv(11, 5), 5, 0, {1, 3, 1, 11, 2, 1, 9}});
    cells.push_back({"G1(5) < G0(5) chi_4 on A", L5.get(), G15.get(), Chi4mod5(5), Triv(5, 5), 5, 1, {0, 2, 1, 5, 2, 1, 3}});
    cells.push_back({"G1(5)^G0(25) < G1(5) n=0", G15.get(), G15_25.get(), Triv(5, 5), Triv(25, 5), 5, 0, {1, 3, 1, 11, 2, 1, 9}});
    cells.push_back({"G1(5)^G0(25) < G1(5) n=3", G15.get(), G15_25.get(), Triv(5, 5), Triv(25, 5), 5, 3, {1, 9, 1, 41, 8, 1, 33}});
    cells.push_back({"G(5) < G", &Gam, &P5, Triv(1, 5), Triv(1, 5), 5, 2, {0, 1, 3, 33, 0, 4, 33}});
    cells.push_back({"G^t_2 < G0(25) a=0", L25.get(), &Tt2, TameChi(25, R5, -2), TameChi(25, R5, -2), 5, 0, {0, 6, 0, 30, 6, 0, 24}});
    const SimplicialComplex& K = TheK();
    for (const Cell& c : cells) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::string tag = "(AB) " + c.tag + " ell=" + std::to_string(c.ell) + " n=" + std::to_string(c.n);
        const std::string pred = Six(c.v[0], c.v[1], c.v[2], c.v[3], c.v[4], c.v[5], c.v[6]);
        try {
            const engine::ModCtx<uint32_t> ctx(c.ell);
            const RestrictionFibreMap R(*c.A, c.cA, *c.B, c.cB, c.n, ctx);
            const auto MA = BuildLevelModel(K, *c.A, c.cA, c.n, ctx);
            const auto MB = BuildLevelModel(K, *c.B, c.cB, c.n, ctx);
            LevelPairCone C(*MA, *MB, R.alpha(), tag);
            C.compute();
            std::vector<std::string> failures;
            const uint32_t cert_bad = C.cone().certificate().verify(failures);
            const LevelPairCone::SixTerm a = C.six_term();
            const RouteBSix b = RouteBSixTerm(*c.A, c.cA, *c.B, c.cB, c.n, c.ell);
            const std::string sa = Six(a.h0A, a.h1A, a.h0B, a.h1B, a.rank_res1, a.H1c, a.H2c);
            const std::string sb = Six(b.h0A, b.h1A, b.h0B, b.h1B, b.rank_res1, b.H1c, b.H2c);
            std::cout << "  " << tag << ": Route A " << sa << " | Route B " << sb << " | predicted " << pred << "\n";
            g.check(cert_bad == 0 && a.spots_exact, tag + ": Route A certificate and exactness");
            g.check(sa == sb, tag + ": Route A = Route B (obl:levelconeexact (iv), (LES5))");
            g.check(sb == pred, tag + ": Route B = the committed numbers");
            g.check(b.alpha_escapes == 0, tag + ": alpha preserves the tree complexes (" + std::to_string(b.alpha_escapes) + " escaping)");
        } catch (const std::exception& e) {
            g.check(false, tag + ": threw: " + e.what());
        }
        ++tot.ab;
        tot.timed(tag, Seconds(t0));
    }
}

// ---- (Z/p) the Gamma_1-type Z/p steps ---------------------------------------------------------------------------------
uint64_t Psi(uint64_t N) {
    uint64_t r = N, nn = N;
    for (uint64_t q = 2; nn > 1; ++q)
        if (nn % q == 0) { r = r / q * (q + 1); while (nn % q == 0) nn /= q; }
    return r;
}

void ArmZp(Checker& g, Totals& tot) {
    struct Plan { uint32_t p, N, step; std::vector<uint32_t> ns; };
    std::vector<Plan> plans;
    auto range = [](uint32_t lo, uint32_t hi) { std::vector<uint32_t> v; for (uint32_t i = lo; i <= hi; ++i) v.push_back(i); return v; };
    for (uint32_t N : {1u, 2u, 3u}) {
        plans.push_back({5, N, 1, range(0, 12)});
        plans.push_back({5, N, 2, N == 1 ? range(0, 12) : range(0, 2)});
    }
    plans.push_back({7, 1, 1, range(0, 12)});
    plans.push_back({7, 1, 2, range(0, 2)});
    plans.push_back({7, 2, 1, {0, 4}});
    plans.push_back({7, 3, 1, {0, 4}});
    uint32_t cells = 0;
    for (const Plan& pl : plans) {
        const uint32_t p = pl.p, N = pl.N;
        const uint64_t pN = static_cast<uint64_t>(p) * N, p2N = pN * p, p2 = static_cast<uint64_t>(p) * p;
        std::unique_ptr<GammaHLevelN> A, B;
        Mat22 sigma;
        if (pl.step == 1) {
            A = GH(pN, [p](uint64_t u) { return u % p == 1; });
            B = GH(p2N, [p](uint64_t u) { return u % p == 1; });
            sigma = Mat22{1, 0, static_cast<int64_t>(pN), 1};
        } else {
            A = GH(p2N, [p](uint64_t u) { return u % p == 1; });
            B = GH(p2N, [p2](uint64_t u) { return u % p2 == 1; });
            sigma = B->sigma(N == 1 ? 1 + p : Crt(1 + p, p2, 1, N));
        }
        const uint64_t mA_pred = (p2 - 1) * Psi(N) * (pl.step == 1 ? 1 : p), mB_pred = p * mA_pred;
        const CosetPermutation pB = PermutationOf(*B);
        for (uint32_t n : pl.ns) {
            const auto t0 = std::chrono::steady_clock::now();
            std::ostringstream ts;
            ts << "(Z/p) p=" << p << " N=" << N << " step=" << pl.step << " n=" << n;
            const std::string tag = ts.str();
            const uint32_t c = n / p + 1;
            const uint32_t h1A = c + static_cast<uint32_t>(mA_pred * (n + 1) / 12), h1B = p * h1A - (p - 1) * c;
            try {
                const engine::ModCtx<uint32_t> ctx(p);
                const CharacterDatum cA = Triv(A->N(), p), cB = Triv(B->N(), p);
                const RestrictionFibreMap R(*A, cA, *B, cB, n, ctx);
                const ConjugationFibreMap Cs(*A, cA, *B, pB, sigma, n, ctx);
                const TreeCokernel QA(*A, cA, n, p, 1), QB(*B, cB, n, p, 1);
                g.check(A->m() == mA_pred && B->m() == mB_pred && R.index() == p, tag + ": m_A, m_B, [A:B] as predicted");
                g.check(QA.h0() == c && QB.h0() == c, tag + ": h0A = h0B = c = " + std::to_string(c) + " (got " + std::to_string(QA.h0()) +
                                                           ", " + std::to_string(QB.h0()) + ")");
                g.check(QA.h1() == h1A && QB.h1() == h1B, tag + ": h1A = " + std::to_string(QA.h1()) + ", h1B = " + std::to_string(QB.h1()) +
                                                              ", predicted " + std::to_string(h1A) + ", " + std::to_string(h1B));
                if (QA.h1() != h1A || QB.h1() != h1B) { ++cells; continue; }
                const DMat res = OnH1(R.alpha(), QA, QB), cor = OnH1(R.trace(), QB, QA), Rs = OnH1(Cs.matrix(), QB, QB);
                const uint64_t esc = Escapes(R.alpha(), QA, QB) + Escapes(R.trace(), QB, QA) + Escapes(Cs.matrix(), QB, QB);
                tot.escapes += static_cast<uint32_t>(esc);
                g.check(esc == 0, tag + ": alpha, omega, R_sigma preserve the tree complexes (" + std::to_string(esc) + " escaping)");
                const CyclicOut o = CyclicGates(g, tag, p, h1A, h1B, c, res, cor, Rs);
                const uint32_t H1c = QB.h0() - QA.h0() + (h1A - o.rank_res), H2c = h1B - o.rank_res;
                g.check(H1c == c && H2c == (p - 1) * h1A - (p - 2) * c, tag + ": H1(A,B) = c, H2(A,B) = (p-1) h1A - (p-2) c");
                if (n == 0 || n == 12 || (pl.step == 2 && n == 2))
                    std::cout << "  " << tag << ": m_A=" << A->m() << " m_B=" << B->m() << " c=" << c << " h1A=" << QA.h1() << " h1B=" << QB.h1()
                              << " Jordan " << JordanText(o.jt) << " rank res1=" << o.rank_res << " H1(A,B)=" << H1c << " H2(A,B)=" << H2c
                              << " (" << Seconds(t0) << " s)\n";
            } catch (const std::exception& e) {
                g.check(false, tag + ": threw: " + e.what());
            }
            ++cells;
            tot.timed(tag, Seconds(t0));
        }
    }
    tot.zp = cells;
    g.check(cells == 78, "(Z/p) 78 cells");
}

// ---- (HIDA1) the first Hida step ----------------------------------------------------------------------------------------
void ArmHida1(Checker& g, Totals& tot) {
    for (uint32_t p : {5u, 7u}) {
        const uint32_t gen = p == 5 ? 2 : 3;   // a primitive root mod p
        for (uint32_t N : {1u, 2u, 3u}) {
            const uint64_t pN = static_cast<uint64_t>(p) * N;
            const auto A = G0H(pN);
            const auto B = GH(pN, [p](uint64_t u) { return u % p == 1; });
            const uint64_t u = N == 1 ? gen : Crt(gen, p, 1, N);
            const Mat22 sigma = B->sigma(u);
            const CosetPermutation pB = PermutationOf(*B);
            for (uint32_t n : {0u, 2u, 4u}) {
                const auto t0 = std::chrono::steady_clock::now();
                std::ostringstream ts;
                ts << "(HIDA1) Gamma_0(" << N << ")^Gamma_1(" << p << ") < Gamma_0(" << pN << ") ell=" << p << " n=" << n;
                const std::string tag = ts.str();
                try {
                    const engine::ModCtx<uint32_t> ctx(p);
                    const tower::ZK F(p, 1);
                    const CharacterDatum cA = Triv(pN, p), cB = Triv(pN, p);
                    const RestrictionFibreMap R(*A, cA, *B, cB, n, ctx);
                    const ConjugationFibreMap Cs(*A, cA, *B, pB, sigma, n, ctx);
                    const TreeCokernel QA(*A, cA, n, p, 1), QB(*B, cB, n, p, 1);
                    const DMat res = OnH1(R.alpha(), QA, QB), Rs = OnH1(Cs.matrix(), QB, QB);
                    const uint64_t esc = Escapes(R.alpha(), QA, QB) + Escapes(Cs.matrix(), QB, QB);
                    g.check(esc == 0, tag + ": alpha, R_sigma preserve the tree complexes");
                    const uint32_t h = QB.h1(), rk = RankModEll(res, p);
                    g.check(rk == QA.h1(), tag + ": ker res^1 = 0 (prop:levelsplit)");
                    DMat P = Identity(h);
                    for (uint32_t i = 0; i + 1 < p; ++i) P = MatMul(F, P, Rs);
                    g.check(Differ(P, Identity(h)) == 0, tag + ": R_sigma^(p-1) = 1 on H^1(B)");
                    const uint32_t inv = h - RankModEll(MatSub(F, Rs, Identity(h)), p);
                    g.check(inv == QA.h1(), tag + ": H^1(A) = H^1(B)^G (dim " + std::to_string(inv) + ")");
                    std::vector<uint32_t> eig, chi_h1;
                    uint32_t sum = 0, sumchi = 0;
                    for (uint32_t i = 0; i + 1 < p; ++i) {
                        const uint32_t lam = static_cast<uint32_t>(powmod<uint64_t>(u % p, i, p));   // omega(a_sigma)^{-i}, a_sigma = u^{-1}
                        DMat L = Identity(h);
                        for (uint32_t& x : L.a) x = static_cast<uint32_t>(static_cast<uint64_t>(x) * lam % p);
                        eig.push_back(h - RankModEll(MatSub(F, Rs, L), p));
                        sum += eig.back();
                        const TreeCokernel Qi(*A, OmegaPow(pN, p, i), n, p, 1, false);
                        chi_h1.push_back(Qi.h1());
                        sumchi += Qi.h1();
                    }
                    std::ostringstream os;
                    os << tag << ": eigenspaces of R_sigma (lambda = omega(a_sigma)^-i, i = 0.." << p - 2 << ") {";
                    for (size_t i = 0; i < eig.size(); ++i) os << (i ? "," : "") << eig[i];
                    os << "} vs h1(Gamma_0(" << pN << "); omega^i) {";
                    for (size_t i = 0; i < chi_h1.size(); ++i) os << (i ? "," : "") << chi_h1[i];
                    os << "}, h1B = " << h;
                    std::cout << "  " << os.str() << "\n";
                    g.check(eig == chi_h1 && sum == h && sumchi == h, os.str() + " (prop:levelinfres (4))");
                    if (N == 1) {
                        static const std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> committed = {
                            {{5, 0}, {1, 0, 2, 0}}, {{5, 2}, {4, 0, 3, 0}}, {{5, 4}, {5, 0, 6, 0}},
                            {{7, 0}, {1, 0, 2, 0, 2, 0}}, {{7, 2}, {4, 0, 4, 0, 5, 0}}, {{7, 4}, {8, 0, 7, 0, 6, 0}}};
                        g.check(chi_h1 == committed.at({p, n}), tag + ": the summands = LES_predictions.md section 2 (iii')");
                    }
                } catch (const std::exception& e) {
                    g.check(false, tag + ": threw: " + e.what());
                }
                ++tot.hida1;
                tot.timed(tag, Seconds(t0));
            }
        }
    }
}

// ---- Merel squares on one pair ----------------------------------------------------------------------------------------
template <class OpA, class OpB>
void Square(Checker& g, Totals& tot, const std::string& tag, const tower::ZK& F, const DMat& res, const TreeCokernel& QA, const TreeCokernel& QB,
            const OpA& XA, const OpB& XB, const DMat* Rs) {
    const DMat TA = OnH1(XA, QA), TB = OnH1(XB, QB);
    const uint64_t esc = Escapes(XA, QA) + Escapes(XB, QB);
    tot.escapes += static_cast<uint32_t>(esc);
    g.check(esc == 0, tag + ": Merel's operator preserves Rel' on both levels (" + std::to_string(esc) + " escaping)");
    const uint64_t bad = Differ(MatMul(F, res, TA), MatMul(F, TB, res));
    ++tot.squares;
    tot.square_bad += static_cast<uint32_t>(std::min<uint64_t>(bad, 1u << 30));
    g.check(bad == 0, tag + ": res^1 T^A = T^B res^1 on H^1 (" + std::to_string(bad) + " differing entries)");
    // (audit R1) the operators are recorded as non-trivial: rank T^A, rank T^B on H^1 (the gated witnesses are in ArmMerel / ArmTower).
    std::cout << "  MEREL_SQUARE " << tag << ": h1A=" << QA.h1() << " h1B=" << QB.h1() << " rank T^A=" << RankModEll(TA, QA.ell())
              << " rank T^B=" << RankModEll(TB, QB.ell()) << " differing=" << bad << "\n";
    if (Rs) g.check(Differ(MatMul(F, *Rs, TB), MatMul(F, TB, *Rs)) == 0, tag + ": R_sigma T^B = T^B R_sigma");
}

// ---- (TOWER) ----------------------------------------------------------------------------------------------------------
void ArmTower(Checker& g, Totals& tot) {
    struct Row { uint32_t p, a; uint32_t v[8]; };   // h0A h1A h0B h1B rank ker H1 H2
    const Row rows[] = {{5, 0, {0, 6, 0, 30, 6, 0, 0, 24}}, {5, 2, {1, 5, 1, 21, 4, 1, 1, 17}}, {7, 0, {0, 10, 0, 70, 10, 0, 0, 60}},
                        {7, 2, {1, 9, 1, 57, 8, 1, 1, 49}}, {7, 4, {0, 10, 0, 70, 10, 0, 0, 60}}};
    for (const Row& r : rows) {
        const uint32_t p = r.p;
        const uint64_t p2 = static_cast<uint64_t>(p) * p;
        const auto A = G0H(p2);
        const GammaHLevelN B = GammaHLevelN::PrimeToPart(p2, p);
        const CosetPermutation pB = PermutationOf(B);
        const Mat22 sigma = B.sigma(1 + p);
        const bool a2 = (r.a % (p - 1)) == 2 % (p - 1);
        const uint32_t c0 = a2 ? 1 : 0;
        DMat res1, Rs1;
        std::vector<uint32_t> freeA1, freeB1;
        for (uint32_t K : {1u, 2u}) {
            const auto t0 = std::chrono::steady_clock::now();
            std::ostringstream ts;
            ts << "(TOWER) Gamma^t_2 < Gamma_0(" << p2 << ") a=" << r.a << " over Z/" << p << "^" << K;
            const std::string tag = ts.str();
            try {
                const tower::ZK R(p, K);
                const engine::ModCtx<uint32_t> ctx = RingCtx(p, K);
                const CharacterDatum chi = TameChi(p2, R, static_cast<int64_t>(r.a) - 2);
                const RestrictionFibreMap Rf(*A, chi, B, chi, 0, ctx);
                const ConjugationFibreMap Cs(*A, chi, B, pB, sigma, 0, ctx);
                const TreeCokernel QA(*A, chi, 0, p, K), QB(B, chi, 0, p, K);
                const DMat res = OnH1(Rf.alpha(), QA, QB), cor = OnH1(Rf.trace(), QB, QA), Rs = OnH1(Cs.matrix(), QB, QB);
                const uint64_t esc = Escapes(Rf.alpha(), QA, QB) + Escapes(Rf.trace(), QB, QA) + Escapes(Cs.matrix(), QB, QB);
                g.check(esc == 0, tag + ": alpha, omega, R_sigma preserve the tree complexes");
                g.check(QA.h1() == r.v[1] && QB.h1() == r.v[3] && QA.h0() == r.v[0] && QB.h0() == r.v[2] && A->m() == p * (p + 1) &&
                            B.m() == p * p * (p + 1),
                        tag + ": h0A h1A h0B h1B = " + std::to_string(QA.h0()) + " " + std::to_string(QA.h1()) + " " + std::to_string(QB.h0()) +
                            " " + std::to_string(QB.h1()) + " (free ranks), m_A, m_B as predicted");
                if (K == 1) {
                    const CyclicOut o = CyclicGates(g, tag, p, QA.h1(), QB.h1(), c0, res, cor, Rs);
                    const uint32_t ker = QA.h1() - o.rank_res, H1c = QB.h0() - QA.h0() + ker, H2c = QB.h1() - o.rank_res;
                    g.check(o.rank_res == r.v[4] && ker == r.v[5] && H1c == r.v[6] && H2c == r.v[7],
                            tag + ": rank res1, ker res1, H1(A,B), H2(A,B) = " + std::to_string(o.rank_res) + " " + std::to_string(ker) + " " +
                                std::to_string(H1c) + " " + std::to_string(H2c) + " as committed (LES8)");
                    std::cout << "  " << tag << ": h1A=" << QA.h1() << " h1B=" << QB.h1() << " Jordan " << JordanText(o.jt) << " rank res1="
                              << o.rank_res << " H1(A,B)=" << H1c << " H2(A,B)=" << H2c << "\n";
                    res1 = res;
                    Rs1 = Rs;
                    freeA1 = QA.Q().free_columns();
                    freeB1 = QB.Q().free_columns();
                } else {
                    g.check(QA.Q().free_columns() == freeA1 && QB.Q().free_columns() == freeB1,
                            tag + ": the Z/p^2 pivots are the F_p pivots (lem:nebchainrref)");
                    g.check(Differ(ReduceMod(res, p), res1) == 0 && Differ(ReduceMod(Rs, p), Rs1) == 0,
                            tag + ": res^1 and R_sigma reduce mod p to the F_p matrices");
                    const uint32_t h1A = QA.h1(), h1B = QB.h1();
                    DMat N(h1B, h1B), P = Identity(h1B);
                    for (uint32_t i = 0; i < p; ++i) { N = MatAdd(R, N, P); P = MatMul(R, P, Rs); }
                    g.check(Differ(P, Identity(h1B)) == 0, tag + ": R_sigma^p = 1 over Z/p^2");
                    const DMat t = MatSub(R, Rs, Identity(h1B));
                    g.check(IsZero(MatMul(R, t, res)), tag + ": (R_sigma - 1) res^1 = 0 over Z/p^2");
                    const uint32_t Lres = SpanLength(R, res), Lt = SpanLength(R, t), LN = SpanLength(R, N);
                    const uint32_t Lker_t = 2 * h1B - Lt;
                    std::cout << "  " << tag << ": lengths im res1=" << Lres << " im(R-1)=" << Lt << " im N=" << LN << " ker(R-1)=" << Lker_t << "\n";
                    if (!a2) {
                        g.check(Lres == 2 * h1A && Lt == 2 * (h1B - h1A) && LN == 2 * h1A,
                                tag + ": res H^1(A) = H^1(B)^G = N H^1(B), free over (Z/p^2)[G] (lengths 2 h1A, 2 (h1B - h1A), 2 h1A)");
                    } else {
                        g.check(Lres == 2 * h1A - 1 && Lker_t - Lres == 1,
                                tag + ": length ker res^1 = 1 = length H^2(G; H^0(B)) (prop:levelinfres (3), H^0 = Z/p^2)");
                    }
                }
                // Merel: T_2 and the same-level U_p (A, B both of Gamma_0-level p^2)
                const tower::ChiHeilbronn T2A(*A, 2, chi.table, R), T2B(B, 2, chi.table, R);
                const tower::ChiHeilbronn UpA(*A, p, chi.table, R), UpB(B, p, chi.table, R);
                Square(g, tot, tag + " T_2", R, res, QA, QB, T2A, T2B, &Rs);
                if (a2 && K == 1) {
                    // (audit R1, LES3_predictions.md section 9) the Eisenstein eigenvalue 1 + 2 = 3 of T_2 on H^1(Gamma_0(p^2); F_p), chi = 1.
                    const DMat T = OnH1(T2A, QA);
                    DMat L = Identity(QA.h1());
                    for (uint32_t& x : L.a) x = (3 * x) % p;
                    const uint32_t rk = RankModEll(MatSub(R, T, L), p);
                    g.check(rk < QA.h1(), tag + " T_2: the Eisenstein eigenvalue 3 occurs on H^1(A) (rank(T_2 - 3) = " + std::to_string(rk) + " < " +
                                              std::to_string(QA.h1()) + ")");
                }
                Square(g, tot, tag + " U_" + std::to_string(p), R, res, QA, QB, UpA, UpB, &Rs);
            } catch (const std::exception& e) {
                g.check(false, tag + ": threw: " + e.what());
            }
            ++tot.tower;
            tot.timed(tag, Seconds(t0));
        }
    }
}

// ---- (GAMMA) Gamma(ell) |> Gamma --------------------------------------------------------------------------------------
void ArmGamma(Checker& g, Totals& tot) {
    struct Row { uint32_t M, ell, n; uint32_t v[8]; int o3; };   // h0A h1A h0B h1B ker rank H1 H2; o3 = -1 at the cross cells
    const Row rows[] = {
        {5, 5, 0, {1, 0, 1, 11, 0, 0, 0, 11}, 0},   {5, 5, 1, {0, 0, 2, 22, 0, 0, 2, 22}, 0},   {5, 5, 2, {0, 1, 3, 33, 1, 0, 4, 33}, 1},
        {5, 5, 3, {0, 0, 4, 44, 0, 0, 4, 44}, 0},   {5, 5, 4, {0, 1, 5, 55, 0, 1, 5, 54}, 0},   {5, 5, 5, {0, 0, 6, 66, 0, 0, 6, 66}, 0},
        {5, 5, 6, {1, 2, 7, 77, 1, 1, 7, 76}, 1},   {5, 5, 7, {0, 0, 8, 88, 0, 0, 8, 88}, 0},   {5, 5, 8, {0, 1, 9, 99, 0, 1, 9, 98}, 0},
        {7, 7, 0, {1, 0, 1, 29, 0, 0, 0, 29}, 0},   {7, 7, 1, {0, 0, 2, 58, 0, 0, 2, 58}, 0},   {7, 7, 2, {0, 1, 3, 87, 0, 1, 3, 86}, 0},
        {7, 7, 3, {0, 0, 4, 116, 0, 0, 4, 116}, 0}, {7, 7, 4, {0, 1, 5, 145, 1, 0, 6, 145}, 1}, {7, 7, 5, {0, 0, 6, 174, 0, 0, 6, 174}, 0},
        {7, 7, 6, {0, 1, 7, 203, 0, 1, 7, 202}, 0}, {7, 7, 7, {0, 0, 8, 232, 0, 0, 8, 232}, 0}, {7, 7, 8, {1, 2, 9, 261, 0, 2, 8, 259}, 0},
        {5, 7, 0, {1, 0, 1, 11, 0, 0, 0, 11}, -1},  {5, 7, 2, {0, 1, 0, 30, 0, 1, 0, 29}, -1},  {5, 7, 4, {0, 1, 0, 50, 0, 1, 0, 49}, -1},
        {7, 5, 0, {1, 0, 1, 29, 0, 0, 0, 29}, -1},  {7, 5, 2, {0, 1, 0, 84, 0, 1, 0, 83}, -1},  {7, 5, 4, {0, 1, 0, 140, 0, 1, 0, 139}, -1}};
    const FullLevel Gam;
    std::map<uint32_t, std::unique_ptr<CongruenceLevel>> Bs;
    for (const Row& r : rows) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!Bs.count(r.M)) Bs[r.M].reset(new CongruenceLevel(CongruenceLevel::Principal(r.M)));
        const CongruenceLevel& B = *Bs[r.M];
        std::ostringstream ts;
        ts << "(GAMMA) Gamma(" << r.M << ") < Gamma ell=" << r.ell << " n=" << r.n;
        const std::string tag = ts.str();
        try {
            const uint32_t ell = r.ell;
            const engine::ModCtx<uint32_t> ctx(ell);
            const tower::ZK F(ell, 1);
            const CharacterDatum cA = Triv(1, ell), cB = Triv(1, ell);
            const CosetPermutation pB = PermutationOf(B);
            const RestrictionFibreMap R(Gam, cA, B, cB, r.n, ctx);
            const ConjugationFibreMap CS(Gam, cA, B, pB, Mat22::S(), r.n, ctx), CT(Gam, cA, B, pB, Mat22::T(), r.n, ctx);
            const TreeCokernel QA(Gam, cA, r.n, ell, 1), QB(B, cB, r.n, ell, 1);
            const DMat res = OnH1(R.alpha(), QA, QB), cor = OnH1(R.trace(), QB, QA);
            const DMat RS = OnH1(CS.matrix(), QB, QB), RT = OnH1(CT.matrix(), QB, QB);
            const uint64_t esc = Escapes(R.alpha(), QA, QB) + Escapes(R.trace(), QB, QA) + Escapes(CS.matrix(), QB, QB) +
                                 Escapes(CT.matrix(), QB, QB);
            g.check(esc == 0, tag + ": alpha, omega, R_S, R_T preserve the tree complexes");
            const uint32_t h = QB.h1(), rk = RankModEll(res, ell), ker = QA.h1() - rk;
            const uint32_t H1c = QB.h0() - QA.h0() + ker, H2c = h - rk;
            const DMat tS = MatSub(F, RS, Identity(h)), tT = MatSub(F, RT, Identity(h));
            DMat side(h, 2 * h), stack(2 * h, h);
            for (uint32_t i = 0; i < h; ++i)
                for (uint32_t j = 0; j < h; ++j) {
                    side.at(i, j) = tS.at(i, j); side.at(i, h + j) = tT.at(i, j);
                    stack.at(i, j) = tS.at(i, j); stack.at(h + i, j) = tT.at(i, j);
                }
            const uint32_t coinv = h - RankModEll(side, ell), inv = h - RankModEll(stack, ell);
            const uint32_t o3 = r.o3 >= 0 ? FiniteGroupH1SL2(ell, r.n) : 0;
            std::ostringstream os;
            os << tag << ": h0A=" << QA.h0() << " h1A=" << QA.h1() << " h0B=" << QB.h0() << " h1B=" << h << " ker res1=" << ker << " rank res1=" << rk
               << " H1(A,B)=" << H1c << " H2(A,B)=" << H2c;
            if (r.o3 >= 0) os << " O3=" << o3;
            os << " | dim H^1(B)^G=" << inv << " (REC: dim H^1(B)^G - rank res1 = " << (inv >= rk ? inv - rk : 0) << ") dim H^1(B)_G=" << coinv;
            std::cout << "  " << os.str() << "\n";
            g.check(QA.h0() == r.v[0] && QA.h1() == r.v[1] && QB.h0() == r.v[2] && h == r.v[3] && ker == r.v[4] && rk == r.v[5] &&
                        H1c == r.v[6] && H2c == r.v[7],
                    tag + ": the six-term numbers as committed (LES_predictions.md section 3)");
            if (r.o3 >= 0) g.check(o3 == static_cast<uint32_t>(r.o3) && ker == o3, tag + ": dim ker res^1 = O3 = dim H^1(SL_2(F_ell); Sym^n)");
            g.check(IsZero(MatMul(F, tS, res)) && IsZero(MatMul(F, tT, res)), tag + ": res^1 lands in H^1(B)^G (R_S, R_T fix it)");
            g.check(IsZero(MatMul(F, cor, tS)) && IsZero(MatMul(F, cor, tT)) && RankModEll(cor, ell) == QA.h1() && coinv == QA.h1(),
                    tag + ": cor^1 : H^1(B)_G -> H^1(A) is an isomorphism (prop:levelinfres (5))");
            if (r.o3 < 0) g.check(inv == QA.h1(), tag + ": |G| a unit: H^1(A) = H^1(B)^G (prop:levelinfres (4))");
        } catch (const std::exception& e) {
            g.check(false, tag + ": threw: " + e.what());
        }
        ++tot.gamma;
        tot.timed(tag, Seconds(t0));
    }
}

// ---- (MEREL) ----------------------------------------------------------------------------------------------------------
void ArmMerel(Checker& g, Totals& tot) {
    // Gamma_0(25) < Gamma_0(5), chi = 1: T_2, T_3; level-raising U_5.
    {
        const auto A = G0H(5), B = G0H(25);
        struct Row { uint32_t ell, n; uint32_t v[8]; uint32_t defect; };
        const Row rows[] = {{5, 0, {1, 1, 1, 5, 1, 0, 0, 4}, 1}, {5, 2, {0, 4, 0, 16, 3, 1, 1, 13}, 0},
                            {7, 0, {1, 1, 1, 5, 1, 0, 0, 4}, 1}, {7, 2, {0, 4, 0, 16, 4, 0, 0, 12}, 3}};
        for (const Row& r : rows) {
            const auto t0 = std::chrono::steady_clock::now();
            const std::string tag = "(MEREL) Gamma_0(25) < Gamma_0(5) ell=" + std::to_string(r.ell) + " n=" + std::to_string(r.n);
            try {
                const engine::ModCtx<uint32_t> ctx(r.ell);
                const tower::ZK F(r.ell, 1);
                const CharacterDatum cA = Triv(5, r.ell), cB = Triv(25, r.ell);
                const RestrictionFibreMap R(*A, cA, *B, cB, r.n, ctx);
                const TreeCokernel QA(*A, cA, r.n, r.ell, 1), QB(*B, cB, r.n, r.ell, 1);
                const DMat res = OnH1(R.alpha(), QA, QB);
                const uint32_t rk = RankModEll(res, r.ell), ker = QA.h1() - rk;
                g.check(QA.h0() == r.v[0] && QA.h1() == r.v[1] && QB.h0() == r.v[2] && QB.h1() == r.v[3] && rk == r.v[4] && ker == r.v[5] &&
                            QB.h0() - QA.h0() + ker == r.v[6] && QB.h1() - rk == r.v[7],
                        tag + ": the six-term numbers (LES_predictions.md section 2 (ii))");
                for (uint64_t q : {2ull, 3ull}) {
                    const GammaHHeilbronn XA(*A, q, r.n, ctx), XB(*B, q, r.n, ctx);
                    Square(g, tot, tag + " T_" + std::to_string(q), F, res, QA, QB, XA, XB, nullptr);
                    if (r.n == 0) {
                        // (audit R1, LES3_predictions.md section 9) H^1(Gamma_0(5); F_ell) is one-dimensional and Eisenstein: T_q = 1 + q.
                        const DMat TA = OnH1(XA, QA);
                        const uint32_t want = static_cast<uint32_t>((1 + q) % r.ell);
                        g.check(TA.r == 1 && TA.c == 1 && TA.at(0, 0) == want,
                                tag + " T_" + std::to_string(q) + ": the witness T_q = 1 + q = " + std::to_string(want) + " on H^1(Gamma_0(5)) (got " +
                                    (TA.r == 1 && TA.c == 1 ? std::to_string(TA.at(0, 0)) : std::string("shape ") + std::to_string(TA.r)) + ")");
                    }
                }
                const GammaHHeilbronn UA(*A, 5, r.n, ctx), UB(*B, 5, r.n, ctx);
                const DMat TA = OnH1(UA, QA), TB = OnH1(UB, QB);
                const uint64_t esc = Escapes(UA, QA) + Escapes(UB, QB);
                g.check(esc == 0, tag + " U_5: Merel's U_5 preserves Rel' on both levels");
                const DMat UBres = MatMul(F, TB, res);
                if (r.ell == 5) g.check(IsZero(UBres), tag + ": U^B_5 res^1 = 0 on H^1 at ell = 5 (prop:levelconehecke (3)(a))");
                const uint32_t defect = RankModEll(MatSub(F, MatMul(F, res, TA), UBres), r.ell);
                std::cout << "  LEVEL_RAISE_SQUARE_B " << tag << " U_5 on H^1: rank(res U^A - U^B res) = " << defect << " (Route A recorded "
                          << r.defect << ")\n";
                g.check(defect == r.defect, tag + ": the level-raising defect rank = Route A's (ADOPT, prop:heilbronn (2))");
                if (r.ell == 7 && r.n == 0) g.check(defect > 0, tag + ": the (LES9) prediction: the U_5 square fails on H^1 (prop:levelconehecke (3)(c))");
            } catch (const std::exception& e) {
                g.check(false, tag + ": threw: " + e.what());
            }
            ++tot.merel;
            tot.timed(tag, Seconds(t0));
        }
    }
    // Gamma_{+-1}(11) |> Gamma_0(11), ell = 5: T_2, T_3, same-level U_11, the diamond.
    {
        const auto A = G0H(11);
        const auto B = GH(11, [](uint64_t u) { return u == 1 || u == 10; });
        const CosetPermutation pB = PermutationOf(*B);
        const Mat22 sigma = B->sigma(2);
        struct Row { uint32_t n; uint32_t v[8]; };
        const Row rows[] = {{0, {1, 3, 1, 11, 2, 1, 1, 9}}, {2, {0, 6, 0, 30, 6, 0, 0, 24}}};
        for (const Row& r : rows) {
            const auto t0 = std::chrono::steady_clock::now();
            const std::string tag = "(MEREL) Gamma_{+-1}(11) < Gamma_0(11) ell=5 n=" + std::to_string(r.n);
            try {
                const uint32_t ell = 5;
                const engine::ModCtx<uint32_t> ctx(ell);
                const tower::ZK F(ell, 1);
                const CharacterDatum cA = Triv(11, ell), cB = Triv(11, ell);
                const RestrictionFibreMap R(*A, cA, *B, cB, r.n, ctx);
                const ConjugationFibreMap Cs(*A, cA, *B, pB, sigma, r.n, ctx);
                const TreeCokernel QA(*A, cA, r.n, ell, 1), QB(*B, cB, r.n, ell, 1);
                const DMat res = OnH1(R.alpha(), QA, QB), Rs = OnH1(Cs.matrix(), QB, QB);
                const uint32_t rk = RankModEll(res, ell), ker = QA.h1() - rk;
                g.check(QA.h0() == r.v[0] && QA.h1() == r.v[1] && QB.h0() == r.v[2] && QB.h1() == r.v[3] && rk == r.v[4] && ker == r.v[5] &&
                            QB.h0() - QA.h0() + ker == r.v[6] && QB.h1() - rk == r.v[7],
                        tag + ": the six-term numbers = Gamma_1(11)'s (n even, -I acts trivially; LES_predictions.md section 2 (iii))");
                for (uint64_t q : {2ull, 3ull, 11ull}) {
                    const GammaHHeilbronn XA(*A, q, r.n, ctx), XB(*B, q, r.n, ctx);
                    Square(g, tot, tag + (q == 11 ? " U_11" : " T_" + std::to_string(q)), F, res, QA, QB, XA, XB, &Rs);
                }
            } catch (const std::exception& e) {
                g.check(false, tag + ": threw: " + e.what());
            }
            ++tot.merel;
            tot.timed(tag, Seconds(t0));
        }
    }
}

// ---- (NEG) ------------------------------------------------------------------------------------------------------------
void ArmNeg(Checker& g) {
    const uint32_t p = 5, n = 2;
    const auto A = GH(5, [](uint64_t u) { return u % 5 == 1; });
    const auto B = GH(25, [](uint64_t u) { return u % 5 == 1; });
    const engine::ModCtx<uint32_t> ctx(p);
    const CharacterDatum cA = Triv(5, p), cB = Triv(25, p);
    const TreeCokernel QA(*A, cA, n, p, 1), QB(*B, cB, n, p, 1);
    // res_no_block: rho(h_i) replaced by I in alpha -- not a Gamma-map, so it does not preserve the tree complex.
    {
        uint64_t esc = 0;
        std::string why;
        try {
            RestrictionOptions o;
            o.neg_identity_block = true;
            const RestrictionFibreMap R(*A, cA, *B, cB, n, ctx, o);
            esc = Escapes(R.alpha(), QA, QB);
            why = std::to_string(esc) + " escaping generators";
        } catch (const std::exception& e) {
            why = std::string("setup threw: ") + e.what();
        }
        std::cout << "  NEG res_no_block " << (esc > 0 ? "caught" : "NOT caught") << " (" << why << ")\n";
        g.check(esc > 0, "(NEG) res_no_block caught (alpha with rho(h_i) = I leaves the tree complex)");
    }
    // jordan_wrong_step (audit R2): CyclicGates fed the conjugation by b = [[1,0],[25,1]] in B (trivial on H^1(B)) in place of R_sigma, at
    // Gamma_1(5) cap Gamma_0(25) |> Gamma_1(5), ell = 5, n = 3 (h1A = 9, h1B = 41, c = 1): the Jordan, invariants and five-term checks must fail.
    {
        const uint32_t n3 = 3;
        const TreeCokernel QA3(*A, cA, n3, p, 1), QB3(*B, cB, n3, p, 1);
        const RestrictionFibreMap R3(*A, cA, *B, cB, n3, ctx);
        const CosetPermutation pB = PermutationOf(*B);
        const ConjugationFibreMap Cb(*A, cA, *B, pB, Mat22{1, 0, 25, 1}, n3, ctx);
        const DMat res = OnH1(R3.alpha(), QA3, QB3), cor = OnH1(R3.trace(), QB3, QA3), Rb = OnH1(Cb.matrix(), QB3, QB3);
        Checker scratch;
        scratch.quiet = true;
        const CyclicOut o = CyclicGates(scratch, "(NEG jordan_wrong_step, scratch)", p, QA3.h1(), QB3.h1(), 1, res, cor, Rb);
        const bool jordan_fails = !(o.jt.size() > 5 && o.jt[5] == QA3.h1() - 1 && o.jt[1] == 1);
        const bool inv_fails = o.inv != QA3.h1();
        const bool five_fails = o.inv - o.rank_res != 1;
        const bool caught = jordan_fails && inv_fails && five_fails && scratch.fails >= 3;
        std::cout << "  NEG jordan_wrong_step " << (caught ? "caught" : "NOT caught") << " (R_b, b in B, on H^1(B): " << JordanText(o.jt) << ", invariants "
                  << o.inv << " vs " << QA3.h1() << ", five-term " << (o.inv - o.rank_res) << " vs 1; CyclicGates failing " << scratch.fails << " of "
                  << scratch.checks << ")\n";
        g.check(caught, "(NEG) jordan_wrong_step caught by CyclicGates (Jordan, invariants and five-term checks fail)");
    }
    // merel_odd_level: Merel's T_2 on Gamma_1(5) cap Gamma_0(25) (-1 not in H, outside prop:nebgammaHheilbronn), n = 0: RECORDED.
    {
        std::string what;
        try {
            const TreeCokernel Q0(*B, cB, 0, p, 1);
            const GammaHHeilbronn X(*B, 2, 0, ctx);
            what = std::to_string(Escapes(X, Q0)) + " of " + std::to_string(Q0.generators().size()) + " generators escape";
        } catch (const std::exception& e) {
            what = std::string("refused: ") + e.what();
        }
        std::cout << "  MEREL_ODD_LEVEL (recorded, no prediction) T_2 on " << B->name() << ", n = 0, ell = 5: " << what << "\n";
    }
}

} // namespace

uint32_t HeckeLevelInfResValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeLevelInfResValidation: Route B, the tree complex of a level pair; inflation-restriction and the Hida p-tower "
                 "([C] sec:nebles, obl:levelconeexact (iv), (vii); LES3_predictions.md)\n";
    Checker g;
    Totals tot;
    ArmAB(g, tot);
    ArmZp(g, tot);
    ArmHida1(g, tot);
    ArmTower(g, tot);
    ArmGamma(g, tot);
    ArmMerel(g, tot);
    ArmNeg(g);
    const double wall = Seconds(t0);
    std::cout << "  INF_RES cells ab=" << tot.ab << " zp=" << tot.zp << " hida1=" << tot.hida1 << " tower=" << tot.tower << " gamma=" << tot.gamma
              << " merel=" << tot.merel << " merel_squares=" << tot.squares << " square_bad_entries=" << tot.square_bad
              << " escapes=" << tot.escapes << "\n";
    std::cout << "  slowest cell " << tot.max_cell << " " << tot.max_cell_s << " s; peak_rss=" << oc_util::PeakRssMiB() << " MiB wall=" << wall
              << " s\n";
    std::cout << "INF_RES checks=" << g.checks << " failing=" << g.fails << " verdict=" << (g.fails == 0 ? "OK" : "FAIL") << "\n";
    return g.fails == 0 ? 0u : 1u;
}
