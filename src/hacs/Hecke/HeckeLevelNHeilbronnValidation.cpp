// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 15, WP-7) HeckeLevelNHeilbronnValidation -- see the header.
// [C] sec:levelN: lem:merelpath, prop:heilbronn, rmk:heilbronnconv, rmk:levelNcheckable (LN3);
// sec:levelshapiro: prop:levelcorrespondence (the tau-route), prop:levelorbit (the presentation).
#include "HeckeLevelNHeilbronnValidation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "CoefficientRing.h"
#include "HeckeCosetTables.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNHeilbronn.h"
#include "HeckeOperatorFamily.h"  // hecke::tau_construction_mutex()
#include "HeckeTauTable.h"
#include "InducedLocalSystem.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "ValidationSupport.h"

namespace {

using engine::scalar::gcd;

using hecke::Mat22;
using namespace hecke::level;
using u128 = unsigned __int128;

constexpr uint32_t kP = 1000003;   // the characteristic-0 surrogate field F_P (as HeckeLevelNValidation)

std::map<std::string, double>& PhaseTimes() { static std::map<std::string, double> t; return t; }
struct PhaseTimer {
    std::string name;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit PhaseTimer(std::string n) : name(std::move(n)) {}
    ~PhaseTimer() { PhaseTimes()[name] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

struct Checker {
    uint32_t checks = 0, fails = 0;
    std::map<std::string, uint32_t> by_arm;
    void check(bool ok, const std::string& what) {
        ++checks;
        by_arm[what.substr(0, what.find(' '))] += 1;
        if (!ok) {
            ++fails;
            if (fails <= 80) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

int64_t SGcd(int64_t a, int64_t b) { return static_cast<int64_t>(gcd<uint64_t>(static_cast<uint64_t>(a < 0 ? -a : a), static_cast<uint64_t>(b < 0 ? -b : b))); }
uint64_t Mod(int64_t x, uint64_t m) { const int64_t r = x % static_cast<int64_t>(m); return static_cast<uint64_t>(r < 0 ? r + static_cast<int64_t>(m) : r); }

std::string RingText(const engine::ModCtx<uint32_t>& ctx) {
    if (ctx.is_field()) return "F_" + std::to_string(ctx.p);
    return "Z/" + std::to_string(ctx.ell) + "^" + std::to_string(ctx.k);
}

using Poly = std::vector<uint64_t>;   // low -> high
Poly PolyMul(const Poly& a, const Poly& b, uint64_t p) {
    Poly c(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) c[i + j] = (c[i + j] + static_cast<uint64_t>(static_cast<u128>(a[i]) * b[j] % p)) % p;
    return c;
}
Poly FromFactors(const std::vector<std::pair<std::vector<int64_t>, uint32_t>>& fs, uint64_t p) {
    Poly out{1};
    for (const auto& f : fs) {
        Poly q;
        for (int64_t c : f.first) q.push_back(Mod(c, p));
        for (uint32_t k = 0; k < f.second; ++k) out = PolyMul(out, q, p);
    }
    return out;
}
std::string PolyText(const Poly& f) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < f.size(); ++i) os << (i ? "," : "") << f[i];
    os << "]";
    return os.str();
}

// ---------------------------------------------------------------------------
// The tau-route side (the model, level-1 tau tables, eq:levelcorr) -- the same
// construction as HeckeLevelNValidation's, repeated here (that file's helpers
// are TU-local).
// ---------------------------------------------------------------------------
struct Model {
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> tess;
};
Model& TheModel() {
    static Model m;
    static std::once_flag once;
    std::call_once(once, [] {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        m.K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        m.tess.reset(new hecke::SL2ZTessellationTables(*m.K));
    });
    return m;
}

using hecke::vsupport::HeckeDatum;
HeckeDatum MakeDatum(uint64_t p, uint32_t coef, uint32_t n) {
    Model& M = TheModel();
    return hecke::vsupport::MakeHeckeDatum(*M.K, *M.tess, p, 1, coef, n);
}

struct NCell {
    uint64_t N = 1;
    uint32_t n = 0;
    engine::ModCtx<uint32_t> ctx;
    uint32_t coef = 0;
    std::unique_ptr<Gamma0LevelN> lvl;
    std::map<uint32_t, HeckeDatum> Tp;
    std::map<uint32_t, std::unique_ptr<Gamma0LevelN>> lvl_Np;
    std::unique_ptr<LocalSystemSimplicialComplex> L;
    std::unique_ptr<EquivariantCohomologyOnK> H;
    std::string tag() const { return "(N=" + std::to_string(N) + ", n=" + std::to_string(n) + ") over " + RingText(ctx); }
};

std::unique_ptr<NCell> BuildNCell(uint64_t N, uint32_t n, const engine::ModCtx<uint32_t>& ctx, const std::vector<uint32_t>& primes) {
    PhaseTimer pt("tau-route: model + tables");
    Model& M = TheModel();
    auto c = std::make_unique<NCell>();
    c->N = N;
    c->n = n;
    c->ctx = ctx;
    c->coef = ctx.p;
    c->lvl.reset(new Gamma0LevelN(N));
    for (uint32_t p : primes) {
        c->Tp[p] = MakeDatum(p, c->coef, n);
        c->lvl_Np[p].reset(new Gamma0LevelN(N * p));
    }
    const uint32_t D = c->lvl->m() * (n + 1);
    c->L.reset(new LocalSystemSimplicialComplex(*M.K, InducedRelationsLevel(*M.K, *c->lvl, n, ctx), D, ctx.p));
    c->H.reset(new EquivariantCohomologyOnK(c->L.get()));
    c->H->compute_cohomology();
    return c;
}

// T_p^{(N)} (p !| N) / U_p^{(N)} (p | N), the pinned r_t of def:cosetrepsN (6): prop:levelcorrespondence (4)(d).
std::map<uint32_t, HMat> TauRouteOps(const NCell& c, const HFrame& f) {
    PhaseTimer pt("tau-route: operators on H^1");
    std::map<uint32_t, HMat> ops;
    for (const auto& kv : c.Tp) {
        const uint32_t p = kv.first;
        const bool bad = c.N % p == 0;
        CorrespondenceSpec s{c.lvl.get(), c.lvl.get(), kv.second.cosets.get(), kv.second.tau.get(),
                             Gamma0SubTransversalN(*c.lvl, *c.lvl_Np.at(p)),
                             std::string(bad ? "U_" : "T_") + std::to_string(p) + "^(N)"};
        VerifyTransversal(s, bad ? p : p + 1);
        ops[p] = InducedOnH(f, f, BuildLevelCorrespondence(c.ctx, s, *TheModel().K, c.n, 1), s.label);
    }
    return ops;
}

HMat Combination(const std::map<uint32_t, HMat>& ops) {
    HMat A = HScale(ops.begin()->second, 0);
    uint64_t coef = 1;
    for (const auto& kv : ops) {
        A = HAdd(A, HScale(kv.second, coef));
        coef = coef * 3 + 1;
    }
    return A;
}

// ---------------------------------------------------------------------------
// (M) Merel's set, lem:merelpath.
// ---------------------------------------------------------------------------
// The cusp of a nonzero integral vector (x, y): primitive, sign-normalised (y > 0, or y = 0 and x > 0).
std::pair<int64_t, int64_t> Cusp(int64_t x, int64_t y) {
    const int64_t g = SGcd(x, y);
    x /= g;
    y /= g;
    if (y < 0 || (y == 0 && x < 0)) { x = -x; y = -y; }
    return {x, y};
}

void ArmMerel(Checker& g) {
    PhaseTimer pt("(M) Merel's set");
    const std::map<uint64_t, size_t> counts{{2, 4}, {3, 7}, {5, 15}, {7, 25}, {11, 49}, {13, 63}, {101, 1097}};
    for (const auto& kv : counts) {
        const uint64_t p = kv.first;
        const std::vector<Mat22> mer = MerelSet(p);
        const std::string tag = "(M) p=" + std::to_string(p);
        g.check(mer.size() == kv.second, tag + ": |Mer_p| = " + std::to_string(mer.size()) + " (rmk:heilbronncost (1): " +
                                             std::to_string(kv.second) + ")");
        g.check(mer.size() <= p * p + 1, tag + ": |Mer_p| <= p^2 + 1 (rmk:heilbronncost (1))");
        // Group by the column lattice L = eps Z^2 (the left coset eps Gamma): key (x_0, b mod x_0), x_0 y_1 = p.
        std::map<std::pair<int64_t, int64_t>, std::vector<Mat22>> by_lattice;
        for (const Mat22& e : mer) {
            const int64_t y1 = SGcd(e.c, e.d);
            const int64_t x0 = static_cast<int64_t>(p) / y1;
            // u c + v d = y1 by extended Euclid on (c, d).
            int64_t r0 = e.c, r1 = e.d, s0 = 1, s1 = 0, t0 = 0, t1 = 1;
            while (r1 != 0) {
                const int64_t q = r0 / r1;
                int64_t tmp = r0 - q * r1; r0 = r1; r1 = tmp;
                tmp = s0 - q * s1; s0 = s1; s1 = tmp;
                tmp = t0 - q * t1; t0 = t1; t1 = tmp;
            }
            if (r0 < 0) { s0 = -s0; t0 = -t0; }
            const int64_t first = s0 * e.a + t0 * e.b;
            by_lattice[{x0, static_cast<int64_t>(Mod(first, static_cast<uint64_t>(x0)))}].push_back(e);
        }
        g.check(by_lattice.size() == p + 1, tag + ": Mer_p meets all p + 1 lattices of index p (" +
                                                std::to_string(by_lattice.size()) + ")");
        bool path = true, bound = true;
        for (const auto& lv : by_lattice) {
            std::map<std::pair<int64_t, int64_t>, int64_t> div;
            for (const Mat22& e : lv.second) {
                div[Cusp(e.a, e.c)] += 1;   // tail [v_1]
                div[Cusp(e.b, e.d)] -= 1;   // head [v_2]
            }
            for (auto it = div.begin(); it != div.end();) it = it->second == 0 ? div.erase(it) : std::next(it);
            const std::map<std::pair<int64_t, int64_t>, int64_t> D0{{{1, 0}, 1}, {{0, 1}, -1}};
            path = path && div == D0;
            bound = bound && static_cast<int64_t>(lv.second.size()) <= lv.first.first;
        }
        g.check(path, tag + ": on every lattice sum_{eps} eps . D_0 = [inf] - [0] (lem:merelpath)");
        g.check(bound, tag + ": |Mer_p cap eta Gamma| <= x_0(eta Z^2) (lem:merelpath)");
    }
    // (slot-25 HZ-REC re-audit R12) MerelSet's divisor-range enumeration against the direct one (every b < a), element by element
    // and in order, for every n <= 200 (composite n included: Gamma_H and the degeneracy maps call MerelSet(n))
    bool same = true;
    uint64_t first_bad = 0;
    for (int64_t n = 1; n <= 200 && same; ++n) {
        std::vector<Mat22> ref;
        for (int64_t a = 1; a <= n; ++a)   // the slot-12 enumeration: (a, d, b) ascending, c determined (all c < d when t = b = 0)
            for (int64_t d = 1; a + d - 1 <= n; ++d) {
                const int64_t t = a * d - n;
                if (t < 0) continue;
                for (int64_t b = 0; b < a; ++b) {
                    if (t == 0) {
                        if (b == 0)
                            for (int64_t c = 0; c < d; ++c) ref.push_back(Mat22{a, 0, c, d});
                        else
                            ref.push_back(Mat22{a, b, 0, d});
                    } else if (b > 0 && t % b == 0 && t / b < d) {
                        ref.push_back(Mat22{a, b, t / b, d});
                    }
                }
            }
        const std::vector<Mat22> got = MerelSet(static_cast<uint64_t>(n));
        bool eq = got.size() == ref.size();
        for (size_t i = 0; eq && i < got.size(); ++i)
            eq = got[i].a == ref[i].a && got[i].b == ref[i].b && got[i].c == ref[i].c && got[i].d == ref[i].d;
        if (!eq) {
            same = false;
            first_bad = static_cast<uint64_t>(n);
        }
    }
    g.check(same, "(M) MerelSet(n) = the direct enumeration, in order, for n <= 200" +
                      (same ? std::string() : " (first difference at n = " + std::to_string(first_bad) + ")"));
}

// ---------------------------------------------------------------------------
// (L) (LN3), both halves, at one cell; (K) classical charpolys.
// ---------------------------------------------------------------------------
struct LCell {
    uint64_t N;
    uint32_t n;
    uint32_t field;                 // the coefficient field F_field
    std::vector<uint32_t> primes;
};

void ArmClassical(Checker& g, const LCell& lc, const std::map<uint32_t, HMat>& hops, const std::string& tag) {
    using F = std::vector<std::pair<std::vector<int64_t>, uint32_t>>;
    std::vector<std::pair<uint32_t, F>> exp;
    if (lc.N == 1 && lc.n == 10) exp = {{2, {{{24, 1}, 2}, {{-2049, 1}, 1}}}};
    if (lc.N == 11 && lc.n == 0) exp = {{2, {{{2, 1}, 2}, {{-3, 1}, 1}}}, {3, {{{1, 1}, 2}, {{-4, 1}, 1}}}, {11, {{{-1, 1}, 3}}}};
    if (lc.N == 5 && lc.n == 2) exp = {{2, {{{4, 1}, 2}, {{-9, 1}, 2}}}, {5, {{{5, 1}, 2}, {{-1, 1}, 1}, {{-125, 1}, 1}}}};
    if (lc.N == 6 && lc.n == 2) exp = {{2, {{{2, 1}, 2}, {{-1, 1}, 2}, {{-8, 1}, 2}}}};
    if (lc.N == 35 && lc.n == 0) exp = {{2, {{{0, 1}, 2}, {{-3, 1}, 3}, {{-4, 1, 1}, 2}}}};
    for (const auto& e : exp) {
        if (!hops.count(e.first)) continue;
        const Poly want = FromFactors(e.second, lc.field), got = HCharpoly(hops.at(e.first));
        g.check(want == got, "(K) " + tag + ": charpoly of [H_{N," + std::to_string(e.first) + "}] is the classical " +
                                 PolyText(want) + " (got " + PolyText(got) + ")");
    }
}

void ArmCell(Checker& g, const LCell& lc, uint32_t& n_bad_ops, uint32_t& n_good_ops, uint32_t& n_mixed_cells) {
    const engine::ModCtx<uint32_t> ctx(lc.field);
    const auto c = BuildNCell(lc.N, lc.n, ctx, lc.primes);
    const std::string tag = c->tag();
    const HFrame f(c->H.get(), 1);
    const std::map<uint32_t, HMat> tops = TauRouteOps(*c, f);

    std::map<uint32_t, HMat> hops;
    std::ostringstream line;
    uint32_t mixed = 0;
    {
        PhaseTimer pt("Heilbronn: presentation + operators");
        const CokernelPresentation pres = CokernelPresentation::Build(*c->lvl, lc.n, ctx);
        const auto& dims = c->H->get_H_eq_dimensions();
        g.check(pres.h1() == dims[1], "(L) " + tag + ": dim M_N / Rel_N = " + std::to_string(pres.h1()) +
                                          " = dim H^1 of the model (" + std::to_string(dims[1]) + ") (prop:levelorbit (2))");
        line << "LEVEL_N_HEILBRONN N=" << lc.N << " n=" << lc.n << " ring=" << RingText(ctx) << " m=" << c->lvl->m()
             << " h1=" << pres.h1() << " rel=" << pres.dim_rel();
        for (uint32_t p : lc.primes) {
            const HeilbronnOperator H(*c->lvl, p, lc.n, ctx);
            const std::string op = "[H_{N," + std::to_string(p) + "}] (" + H.label() + ")";
            const uint32_t esc = RelEscapes(pres, H);
            g.check(esc == 0, "(L) " + tag + ": " + op + " -- H(Rel_N) in Rel_N entrywise on all " +
                                  std::to_string(pres.dim_rel()) + " RREF rows (prop:heilbronn (1), (LN3))");
            const uint32_t admissible = static_cast<uint32_t>(H.terms().size());
            const bool counts = H.pairs_total() == static_cast<uint32_t>(H.merel().size()) * c->lvl->m() &&
                                admissible + H.pairs_dropped() == H.pairs_total() &&
                                (H.bad() ? H.pairs_dropped() > 0 : H.pairs_dropped() == 0);
            g.check(counts, "(L) " + tag + ": " + op + " -- " + std::to_string(admissible) + " admissible pairs of " +
                                std::to_string(H.pairs_total()) + " (p | N drops exactly the (c_j, d_j) eps = 0 mod p)");
            mixed += H.mixed_terms();
            (H.bad() ? n_bad_ops : n_good_ops) += 1;
            if (esc != 0) continue;
            hops[p] = OnCokernel(pres, H);
            const Poly ph = HCharpoly(hops[p]), pt_ = HCharpoly(tops.at(p));
            g.check(ph == pt_, "(L) " + tag + ": (LN3) charpoly of " + op + " on M_N / Rel_N == the tau-route's on H^1 " +
                                   PolyText(pt_));
            line << " chi(" << (H.bad() ? "U" : "T") << p << ")=" << PolyText(ph);
        }
    }
    if (hops.size() == tops.size() && hops.size() >= 2) {
        const HMat hc = Combination(hops), tc = Combination(tops);
        g.check(HCharpoly(hc) == HCharpoly(tc), "(L) " + tag + ": charpoly of sum_p c_p [H_{N,p}] == the tau-route's");
        const uint32_t p0 = hops.begin()->first, p1 = std::next(hops.begin())->first;
        g.check(HCharpoly(HAdd(HMul(hops[p0], hops[p1]), hops[p0])) ==
                    HCharpoly(HAdd(HMul(tops.at(p0), tops.at(p1)), tops.at(p0))),
                "(L) " + tag + ": charpoly of H_" + std::to_string(p0) + " H_" + std::to_string(p1) + " + H_" +
                    std::to_string(p0) + " == the tau-route's");
        bool comm = true;
        for (auto a = hops.begin(); a != hops.end(); ++a)
            for (auto b = std::next(a); b != hops.end(); ++b)
                comm = comm && HEqual(HMul(a->second, b->second), HMul(b->second, a->second));
        g.check(comm, "(L) " + tag + ": the [H_{N,p}] commute on M_N / Rel_N");
    }
    ArmClassical(g, lc, hops, tag);
    if (mixed > 0) ++n_mixed_cells;
    line << " mixed_terms=" << mixed;
    std::cout << "  " << line.str() << std::endl;
}

// ---------------------------------------------------------------------------
// (N1) prop:heilbronn (3): at N = 1, H_{1,p} x = sum_{eps} rho_n(adj eps) x.
// ---------------------------------------------------------------------------
void ArmLevelOne(Checker& g) {
    PhaseTimer pt("(N1) N = 1");
    const engine::ModCtx<uint32_t> ctx(kP);
    const Gamma0LevelN one(1);
    for (uint32_t n : {10u, 22u})
        for (uint64_t p : {2ull, 3ull, 5ull}) {
            const HeilbronnOperator H(one, p, n, ctx);
            const uint32_t d = n + 1;
            std::vector<uint64_t> want(static_cast<size_t>(d) * d, 0);
            RhoCache rc(ctx, n);
            std::vector<Block> store;
            for (const Mat22& e : MerelSet(p)) {
                const Block& B = store[rc.block_of(e.adj(), store)];
                for (size_t x = 0; x < want.size(); ++x) want[x] = (want[x] + B.dense[x]) % kP;
            }
            const std::vector<uint32_t> got = H.dense();
            bool eq = got.size() == want.size();
            for (size_t x = 0; eq && x < want.size(); ++x) eq = got[x] == want[x];
            g.check(eq, "(N1) (n=" + std::to_string(n) + ", p=" + std::to_string(p) +
                            "): H_{1,p} = sum_{eps in Mer_p} rho_n(adj eps) entrywise (prop:heilbronn (3))");
        }
}

// ---------------------------------------------------------------------------
// (E) the N = ell plug point: Heilbronn on Gamma0LevelN(ell) against the tau-route
// on the SHIPPED level-ell table (Gamma0Level over HeckeCosetTables, slot-13 WP-6).
// ---------------------------------------------------------------------------
void ArmPlugPoint(Checker& g, uint32_t n, uint32_t ell, uint32_t field) {
    PhaseTimer pt("(E) N = ell plug point");
    Model& M = TheModel();
    const engine::ModCtx<uint32_t> ctx(field);
    const std::string tag = "(E) (n=" + std::to_string(n) + ", ell=" + std::to_string(ell) + ") over " + RingText(ctx);
    HeckeDatum U = MakeDatum(ell, field, n), T2 = MakeDatum(2, field, n);
    const Gamma0Level shipped(*U.cosets);
    const uint32_t D = (ell + 1) * (n + 1);
    LocalSystemSimplicialComplex Lref(*M.K, hecke::MakeInducedLocalSystemGamma0(*M.K, *U.cosets, field, n).get_user_defined_simplex_relations(),
                                      D, field);
    EquivariantCohomologyOnK Href(&Lref);
    Href.compute_cohomology();
    const HFrame fr(&Href, 1);
    const Gamma0LevelN mine(ell);
    const CokernelPresentation pres = CokernelPresentation::Build(mine, n, ctx);
    g.check(pres.h1() == Href.get_H_eq_dimensions()[1], tag + ": dim M_ell / Rel_ell = dim H^1 on the shipped level-ell model");
    for (const auto& dp : std::vector<std::pair<HeckeDatum*, uint32_t>>{{&U, ell}, {&T2, 2}}) {
        const uint32_t p = dp.second;
        CorrespondenceSpec s{&shipped, &shipped, dp.first->cosets.get(), dp.first->tau.get(), Gamma0SubTransversal(ell, p), "ref"};
        VerifyTransversal(s, p == ell ? p : p + 1);
        const HMat A = InducedOnH(fr, fr, BuildLevelCorrespondence(ctx, s, *M.K, n, 1), "ref");
        const HeilbronnOperator H(mine, p, n, ctx);
        const std::string op = std::string(p == ell ? "U_" : "T_") + std::to_string(p) + "^(ell)";
        g.check(RelEscapes(pres, H) == 0, tag + ": " + op + " Heilbronn preserves Rel_ell");
        g.check(HCharpoly(OnCokernel(pres, H)) == HCharpoly(A),
                tag + ": " + op + " Heilbronn on Gamma0LevelN(ell) == tau-route on the shipped level-ell table (charpoly " +
                    PolyText(HCharpoly(A)) + ")");
    }
}

// ---------------------------------------------------------------------------
// (Z) rmk:heilbronnconv (2): the two mis-transcriptions, negative controls.
// ---------------------------------------------------------------------------
void ArmTraps(Checker& g) {
    PhaseTimer pt("(Z) traps");
    const engine::ModCtx<uint32_t> ctx(kP);
    for (const auto& cell : std::vector<std::array<uint32_t, 3>>{{1, 10, 2}, {5, 2, 2}, {5, 2, 5}, {6, 2, 5}}) {
        const Gamma0LevelN L(cell[0]);
        const CokernelPresentation pres = CokernelPresentation::Build(L, cell[1], ctx);
        const std::string tag = "(Z) (N=" + std::to_string(cell[0]) + ", n=" + std::to_string(cell[1]) + ", p=" +
                                std::to_string(cell[2]) + ")";
        const uint32_t ok = RelEscapes(pres, HeilbronnOperator(L, cell[2], cell[1], ctx));
        const uint32_t ea = RelEscapes(pres, HeilbronnOperator(L, cell[2], cell[1], ctx, HeilbronnVariant::kTrapA));
        const uint32_t eb = RelEscapes(pres, HeilbronnOperator(L, cell[2], cell[1], ctx, HeilbronnVariant::kTrapB));
        g.check(ok == 0, tag + ": eq:heilbronn preserves Rel_N");
        g.check(ea > 0, tag + ": trap (a) rho(h_j(eps)) leaves Rel_N (" + std::to_string(ea) + " rows) -- negative control");
        g.check(eb > 0, tag + ": trap (b) Mer_p as the right-coset set leaves Rel_N (" + std::to_string(eb) +
                            " rows) -- negative control");
    }
    // At weight 2 rho_n is trivial: trap (a) IS eq:heilbronn (rmk:heilbronnconv (2)).
    for (const auto& cell : std::vector<std::array<uint32_t, 2>>{{11, 2}, {35, 5}, {30, 7}}) {
        const Gamma0LevelN L(cell[0]);
        const bool eq = HeilbronnOperator(L, cell[1], 0, ctx).dense() ==
                        HeilbronnOperator(L, cell[1], 0, ctx, HeilbronnVariant::kTrapA).dense();
        g.check(eq, "(Z) (N=" + std::to_string(cell[0]) + ", n=0, p=" + std::to_string(cell[1]) +
                        "): at weight 2 trap (a) coincides with eq:heilbronn entrywise");
    }
}

} // namespace

// No catch-all: the harness reports an escaping exception as [EXCEPTION] / [FAIL] (S13 WP-16).
uint32_t HeckeLevelNHeilbronnValidation() {
    std::cout << "=== HeckeLevelNHeilbronnValidation (slot-15 WP-7: the Heilbronn--Merel action H_{N,p} on the cokernel "
                 "presentation; [C] sec:levelN lem:merelpath, prop:heilbronn, (LN3) both halves) ===\n";
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    Checker g;

    ArmMerel(g);
    ArmLevelOne(g);
    ArmTraps(g);
    std::cout << "  [(M), (N1), (Z) done at " << secs() << " s]" << std::endl;

    // (L): WP-12's twenty cocycle-comparison cells (HeckeLevelNValidation's (G) list) and its weight-6 cell, every
    // operator p in {2, 3, 5, 7}; level 1 at weights 12, 24; (LN3)'s referee cells; F_ell with ell | N and ell !| N.
    const std::vector<uint32_t> P4{2, 3, 5, 7};
    std::vector<LCell> cells{
        {4, 0, kP, P4},  {4, 2, kP, P4},  {6, 0, kP, P4},  {6, 2, kP, P4},  {6, 4, kP, P4},  {8, 2, kP, P4},
        {9, 0, kP, P4},  {9, 2, kP, P4},  {10, 0, kP, P4}, {10, 2, kP, P4}, {12, 0, kP, P4}, {12, 2, kP, P4},
        {15, 0, kP, P4}, {15, 2, kP, P4}, {21, 0, kP, P4}, {21, 2, kP, P4}, {5, 2, kP, P4},  {11, 0, kP, {2, 3, 5, 11}},
        {35, 0, kP, P4}, {30, 0, kP, P4}, {12, 4, kP, {2, 3, 5}},
        {1, 10, kP, {2, 3, 5}}, {1, 22, kP, {2, 3}},
        {13, 2, kP, {2, 13}}, {4, 4, kP, {2, 3}}, {20, 2, kP, {2, 5}}, {22, 2, kP, {2, 11}},
        // F_ell, ell | N and ell !| N (and p = ell on both sides).
        {10, 2, 5, {2, 3, 5}}, {21, 2, 7, {2, 3, 7}}, {12, 2, 5, {2, 3, 5}}, {15, 4, 5, {2, 3, 5}},
        {14, 0, 7, {2, 3, 7}}, {8, 2, 7, {2, 7}}, {13, 2, 7, {2, 7}}, {15, 2, 7, {5, 7}},
        // larger: two R3 cells (prime N, F_p, p | N - 1, weight 2) and composite N with many mixed classes.
        {181, 0, 5, P4}, {131, 0, 13, P4}, {60, 0, kP, P4}, {42, 2, kP, P4}};
    uint32_t n_bad = 0, n_good = 0, n_mixed = 0;
    for (const LCell& lc : cells) ArmCell(g, lc, n_bad, n_good, n_mixed);
    g.check(n_bad > 0 && n_good > 0, "(L) both halves tested: " + std::to_string(n_good) + " T_p (p !| N) and " +
                                         std::to_string(n_bad) + " U_p (p | N) operators");
    g.check(n_mixed > 0, "(L) mixed classes (case (iii)) reached by H_{N,p} at " + std::to_string(n_mixed) + " cells");
    std::cout << "  [(L) done at " << secs() << " s: " << cells.size() << " cells, " << n_good << " T_p, " << n_bad
              << " U_p]" << std::endl;

    ArmPlugPoint(g, 2, 5, kP);
    ArmPlugPoint(g, 4, 7, kP);
    ArmPlugPoint(g, 2, 5, 5);
    std::cout << "  [(E) done at " << secs() << " s]" << std::endl;

    for (const auto& kv : g.by_arm) std::cout << "  arm " << kv.first << ": " << kv.second << " checks\n";
    for (const auto& kv : PhaseTimes()) std::cout << "  phase " << kv.first << ": " << kv.second << " s\n";
    std::cout << "HeckeLevelNHeilbronnValidation: " << g.checks << " checks, " << g.fails << " failing, " << secs() << " s\n";
    return g.fails == 0 ? 0 : 1;
}
