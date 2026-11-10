// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 16, WP-13) The WP-13 arms of HeckeLevelNSparseValidation -- see HeckeLevelNDegenValidation.h.
// [C] v1.16.0-pre.3 sec:levelNdegsec: prop:kappamodel ((LN7)), prop:degenpres ((LN8)), rmk:degenpresconv (the traps),
// rmk:levelNscale (the ladder's RSS model), rmk:levelNcheckabletwo (LN7)-(LN9).
//
//   (K)   (LN7) model = presentation, entrywise: kappa^mod_N(f) = -f(e_46) on a basis of Z^1 mod B^1 spans
//         M_N / Rel_N and is independent there; kappa(B^1) in Rel_N; kappa^{mod,0}(H^0) = M^S cap M^ST;
//         (T~ f)(e_46) - H_{N,q} f(e_46) in Rel_N for q !| N and q | N; alpha^0, omega commute with f -> f(e_46)
//         EXACTLY (and their e_46 blocks are the LevelChangeMap blocks entry for entry).
//   (A)   (LN8) alpha_d on the presentation: the two enumerations give the same term multiset; (1) D(Rel_M) in
//         Rel_N; (2) the model's alpha_d (prop:levelcorrespondence (4)(c), through Gamma_0(M d_1) at composite d)
//         at e_46 minus D f(e_46) in Rel_N; (3) d = 1 is alpha^0 entrywise; (4) composite d; Hecke equivariance
//         T_q alpha_d = alpha_d T_q (q !| N); the traps of rmk:degenpresconv (2) fail (negative controls).
//   (W)   prop:degenpres (5), the trace, on the presentation: omega D_{M,dM,d} = T_p (d = p !| M), = U_p
//         (d = p | M: proved, untested before WP-13 -- WP-14 referee C-4), = T_p^2 - p^n (p + 1) (M = 1, d = p^2);
//         the old-space relations U_p alpha_1 = p alpha_p, U_p alpha_p = alpha_p T_p - p^n alpha_1 (STD, not
//         verified: REPORTED, never gated).
//   (L9)  (LN9) R3_k at weight k >= 4: REPORTED, never gated (a CONJ); two sanity gates on the reading itself.
//   (G)   alpha_d at scale on the sparse route (composite N, k > 2): (1) on every generator, (5) and Hecke
//         equivariance on sampled free vectors; DEGEN_CELL lines with the RSS model.  Env hook
//         HECKE_LEVEL_N_DEGEN_CELLS="M:N:d:n:p,..." replaces the default list (EFFECTIVE line); env hook
//         HECKE_LEVEL_N_R3K_CELLS="N:p:k,..." replaces (L9)'s list (EFFECTIVE line).  Read-only, not emitters.  Both hooks
//         parse strictly, refuse an empty value / cell / field by position and refuse a cell whose predicted peak RSS
//         exceeds the cap before the run (slot-16 WP-17, audit WP-13 F3; since slot-17 WP-17 the calibrated hook_rss/v1
//         model and the HECKE_LEVEL_N_HOOK_RSS_CAP_GIB override, HeckeLevelNDegenValidation.h).
//   (F1)  (slot-16 WP-17, audit WP-13 F1) HeilbronnOperator's __int128 fallback forced on every term = the int64 route,
//         term for term and entry for entry, at composite N with case-(iii) classes.
#include "HeckeLevelNDegenValidation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "CoefficientRing.h"
#include "HeckeCosetTables.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNDegen.h"
#include "HeckeLevelNHeilbronn.h"
#include "HeckeLevelNSparse.h"
#include "HeckeOperatorFamily.h"  // hecke::tau_construction_mutex()
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../HeckeExt/HeckeHidaComponents.h"   // B_k / k mod p^K for (LN9)'s report (a *Validation.* TU)
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Util/Timing.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "ValidationSupport.h"

namespace {

using oc_util::Seconds;
using engine::scalar::powmod;

using hecke::Mat22;
using namespace hecke::level;
using u128 = unsigned __int128;

constexpr uint32_t kP = 1000003;   // the characteristic-0 surrogate field (as every level-N validation)

// Non-vacuity: every "difference in Rel" test also counts the columns whose two sides are NOT in Rel on their own
// (so the comparison is not 0 = 0 mod Rel); each arm gates that its total is positive.
struct Nontrivial { uint64_t K = 0, A = 0, W = 0, G = 0; };
Nontrivial& NT() { static Nontrivial n; return n; }

struct Checker {
    uint32_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (fails <= 80) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

// ---- the model side (the construction of the other level-N validations; their helpers are TU-local) ----
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
HeckeDatum MakeDatum(uint64_t p, uint32_t nu, uint32_t coef, uint32_t n) {
    Model& M = TheModel();
    return hecke::vsupport::MakeHeckeDatum(*M.K, *M.tess, p, nu, coef, n);
}
struct LevelModel {
    std::unique_ptr<Gamma0LevelN> lvl;
    std::unique_ptr<LocalSystemSimplicialComplex> L;
    std::unique_ptr<EquivariantCohomologyOnK> H;
    std::unique_ptr<HFrame> f1, f0;
    uint32_t D = 0;
};
std::unique_ptr<LevelModel> BuildLevelModel(uint64_t N, uint32_t n, const engine::ModCtx<uint32_t>& ctx) {
    Model& M = TheModel();
    auto c = std::make_unique<LevelModel>();
    c->lvl.reset(new Gamma0LevelN(N));
    c->D = c->lvl->m() * (n + 1);
    c->L.reset(new LocalSystemSimplicialComplex(*M.K, InducedRelationsLevel(*M.K, *c->lvl, n, ctx), c->D, ctx.p));
    c->H.reset(new EquivariantCohomologyOnK(c->L.get()));
    c->H->compute_cohomology();
    c->f1.reset(new HFrame(c->H.get(), 1));
    c->f0.reset(new HFrame(c->H.get(), 0));
    return c;
}

std::vector<uint32_t> ToDense(const SparseVec& s, uint32_t D) {
    std::vector<uint32_t> x(D, 0);
    for (size_t k = 0; k < s.col.size(); ++k) x[s.col[k]] = s.val[k];
    return x;
}
std::vector<uint32_t> Sub(const engine::ModCtx<uint32_t>& ctx, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    std::vector<uint32_t> o(a.size());
    for (size_t i = 0; i < a.size(); ++i) o[i] = ctx.sub(a[i], b[i]);
    return o;
}
std::vector<uint32_t> Scale(const engine::ModCtx<uint32_t>& ctx, const std::vector<uint32_t>& a, uint64_t s) {
    std::vector<uint32_t> o(a.size());
    const uint32_t ss = static_cast<uint32_t>(s % ctx.p);
    for (size_t i = 0; i < a.size(); ++i) o[i] = ctx.mul(a[i], ss);
    return o;
}
bool InRel(const SparseCokernel& Q, const std::vector<uint32_t>& x) {
    const std::vector<uint32_t> c = Q.coords_dense(x);
    return std::all_of(c.begin(), c.end(), [](uint32_t v) { return v == 0; });
}
std::string Tag4(const char* arm, uint64_t M, uint64_t N, uint64_t d, uint32_t n, uint32_t p) {
    return std::string(arm) + " (M=" + std::to_string(M) + ", N=" + std::to_string(N) + ", d=" + std::to_string(d) +
           ", n=" + std::to_string(n) + ") over F_" + std::to_string(p);
}

// The model operator T_q^{(N)} / U_q^{(N)} as a cochain map (prop:levelcorrespondence (4)(d)).
BlockCochainMap ModelHecke(const LevelModel& lm, uint32_t q, const HeckeDatum& D, uint32_t n, const engine::ModCtx<uint32_t>& ctx) {
    const bool bad = lm.lvl->N() % q == 0;
    const Gamma0LevelN lNq(lm.lvl->N() * q);
    CorrespondenceSpec s{lm.lvl.get(), lm.lvl.get(), D.cosets.get(), D.tau.get(), Gamma0SubTransversalN(*lm.lvl, lNq),
                         std::string(bad ? "U_" : "T_") + std::to_string(q) + "^(N)"};
    VerifyTransversal(s, bad ? q : q + 1);
    return BuildLevelCorrespondence(ctx, s, *TheModel().K, n, 1);
}

// The model's alpha_d (M -> N) applied to flat level-M cochains X (ncols columns): a chain of prime-power
// DegeneracySpecs through Gamma_0(M d_1) (prop:levelcorrespondence (4)(c)); d = 1 is the restriction.
std::vector<uint32_t> ModelAlpha(uint64_t M, uint64_t N, uint64_t d, uint32_t n, const engine::ModCtx<uint32_t>& ctx,
                                 const std::vector<uint32_t>& X, uint32_t ncols) {
    const SimplicialComplex& K = *TheModel().K;
    if (d == 1) {
        const Gamma0LevelN a(M), b(N);
        return BuildLevelCorrespondence(ctx, RestrictionSpec(a, b), K, n, 1).apply(X, ncols);
    }
    const auto fac = DegeneracyFactors(d);
    std::vector<uint32_t> Y = X;
    uint64_t L = M;
    for (size_t i = 0; i < fac.size(); ++i) {
        uint64_t q = 1;
        for (uint32_t e = 0; e < fac[i].second; ++e) q *= fac[i].first;
        const uint64_t Lp = (i + 1 == fac.size()) ? N : L * q;
        const Gamma0LevelN a(L), b(Lp);
        const HeckeDatum D = MakeDatum(fac[i].first, fac[i].second, ctx.p, n);
        Y = BuildLevelCorrespondence(ctx, DegeneracySpec(a, b, *D.cosets, *D.tau), K, n, 1).apply(Y, ncols);
        L = Lp;
    }
    return Y;
}

// ---------------------------------------------------------------------------
// (K) (LN7): prop:kappamodel on the model.
// ---------------------------------------------------------------------------
struct KCell { uint64_t N; uint32_t n; uint32_t p; std::vector<uint64_t> sub; };

void ArmKappa(Checker& g, const KCell& c, std::vector<std::string>& lines) {
    const engine::ModCtx<uint32_t> ctx(c.p);
    const std::string tag = "(K) (N=" + std::to_string(c.N) + ", n=" + std::to_string(c.n) + ") over F_" + std::to_string(c.p);
    const KappaPoints pts = KappaModelPoints(*TheModel().K);
    auto lm = BuildLevelModel(c.N, c.n, ctx);
    const SparseCokernel Q = SparseCokernel::Build(*lm->lvl, c.n, ctx, true);
    const uint32_t D = lm->D, t1 = lm->f1->t(), t0 = lm->f0->t();
    g.check(t1 == Q.h1() && t0 == Q.h0(), tag + ": dim H^1, H^0 of the model = h1, h0 of the presentation (" +
                                               std::to_string(t1) + ", " + std::to_string(t0) + ")");
    // (1) kappa^mod induces an isomorphism H^1 -> M / Rel: the -f(e_46) of the basis, in coordinates, have rank h1.
    std::vector<uint32_t> co(static_cast<size_t>(Q.h1()) * t1, 0);   // h1 x t1
    for (uint32_t j = 0; j < t1; ++j) {
        const std::vector<uint32_t> cj = Q.coords_dense(KappaModel(ctx, lm->f1->basis(), t1, j, pts, D));
        for (uint32_t i = 0; i < Q.h1(); ++i) co[static_cast<size_t>(i) * t1 + j] = cj[i];
    }
    const uint32_t rk = t1 ? FpRank(co, Q.h1(), t1, c.p) : 0;
    g.check(rk == Q.h1() && rk == t1, tag + ": (LN7) the classes -f(e_46) of a basis of H^1 span M_N / Rel_N and are independent (rank " +
                                          std::to_string(rk) + ")");
    // kappa^mod(B^1) in Rel: (d g)(e_46) = g(v_6) - g(v_4) for g in C^0_eq.
    uint32_t c0 = 0;
    const std::vector<uint32_t> C0 = lm->f0->ceq_basis(c0);
    bool b1 = true;
    for (uint32_t j = 0; j < c0 && b1; ++j)
        b1 = InRel(Q, Sub(ctx, CochainAt(C0, c0, j, pts.v4, D), CochainAt(C0, c0, j, pts.v6, D)));
    g.check(b1, tag + ": kappa^mod(B^1) in Rel_N on all " + std::to_string(c0) + " C^0_eq generators");
    // (slot-16 WP-17, audit WP-13 F5) The orientation, EXACTLY, in the real equivariant complex (not modulo Rel and not
    // only on the trivial-coefficient incidence matrix): (i) the model's d^0 = (D_0 (x) I_D) applied to the C^0_eq
    // generators spans the model's B^1_eq (the stored B^1_eq basis, which compute_cohomology() built), and
    // (ii) (d g)(e_46) = g(v_6) - g(v_4) entry for entry, so kappa(d g) = g(v_4) - g(v_6) carries the sign of
    // prop:kappamodel; (iii) the sign is observed: some generator has g(v_6) != g(v_4) (else (ii) is 0 = 0).
    {
        const SimplicialComplex& K = *TheModel().K;
        const FlatMatrixT<uint32_t> D0 = K.coboundary_operator_elt<uint32_t>(0, c.p);
        const uint32_t n0 = K.get_num_q_simplices(0), n1 = K.get_num_q_simplices(1);
        std::vector<std::vector<uint32_t>> dg, bstored, both;
        bool exact = true;
        uint32_t observed = 0;
        for (uint32_t j = 0; j < c0; ++j) {
            std::vector<uint32_t> y(static_cast<size_t>(n1) * D, 0);
            for (uint32_t s1 = 0; s1 < n1; ++s1)
                for (uint32_t s0 = 0; s0 < n0; ++s0) {
                    const uint32_t inc = D0.get_entry(s1, s0);
                    if (!inc) continue;
                    for (uint32_t r = 0; r < D; ++r)
                        y[static_cast<size_t>(s1) * D + r] =
                            ctx.add(y[static_cast<size_t>(s1) * D + r], ctx.mul(inc, C0[(static_cast<size_t>(s0) * D + r) * c0 + j]));
                }
            const std::vector<uint32_t> want = Sub(ctx, CochainAt(C0, c0, j, pts.v6, D), CochainAt(C0, c0, j, pts.v4, D));
            const std::vector<uint32_t> at(y.begin() + static_cast<size_t>(pts.e46) * D, y.begin() + static_cast<size_t>(pts.e46 + 1) * D);
            exact = exact && at == want;
            observed += std::any_of(want.begin(), want.end(), [](uint32_t v) { return v != 0; }) ? 1 : 0;
            dg.push_back(y);
        }
        for (const auto& b : lm->H->get_B_eq_bases_fm()[1]) {
            std::vector<uint32_t> y(static_cast<size_t>(n1) * D, 0);
            for (uint32_t s1 = 0; s1 < n1; ++s1)
                for (uint32_t r = 0; r < D; ++r) y[static_cast<size_t>(s1) * D + r] = b.get_entry(s1, r);
            bstored.push_back(y);
        }
        both = dg;
        both.insert(both.end(), bstored.begin(), bstored.end());
        const uint32_t rd = FieldRank(dg, c.p), rb = FieldRank(bstored, c.p), ru = FieldRank(both, c.p);
        g.check(rd == rb && rb == ru, tag + ": (LN7) d^0 = D_0 (x) I on C^0_eq spans the model's stored B^1_eq (ranks " +
                                          std::to_string(rd) + ", " + std::to_string(rb) + ", " + std::to_string(ru) + ")");
        g.check(exact && (c0 == 0 || observed > 0),
                tag + ": (LN7) orientation EXACT in the real complex: (d g)(e_46) = g(v_6) - g(v_4) on all " + std::to_string(c0) +
                    " C^0_eq generators (" + std::to_string(observed) + " with g(v_6) != g(v_4))");
    }
    // kappa^{mod,0}: H^0 -> M^S cap M^ST, injective.
    const MonomialAction mS = MonomialOf(*lm->lvl, Mat22::S(), c.n, ctx), mU = MonomialOf(*lm->lvl, Mat22::ST(), c.n, ctx);
    bool fixed = true;
    std::vector<std::vector<uint32_t>> h0v;
    for (uint32_t j = 0; j < t0; ++j) {
        const std::vector<uint32_t> x = KappaModel0(lm->f0->basis(), t0, j, pts, D);
        fixed = fixed && mS.apply(ctx, x) == x && mU.apply(ctx, x) == x;
        h0v.push_back(x);
    }
    g.check(fixed && FieldRank(h0v, c.p) == t0, tag + ": kappa^{mod,0} maps H^0 injectively into M^S cap M^ST (dim " +
                                                   std::to_string(t0) + ")");
    // (4) / (LN7): (T~ f)(e_46) - H_{N,q} f(e_46) in Rel_N, q !| N and q | N.
    std::vector<uint32_t> qs{2u, 3u, 5u, 7u};
    for (const auto& pf : FactorLevel(c.N))
        if (pf.q > 7 && pf.q < 30) qs.push_back(static_cast<uint32_t>(pf.q));
    uint32_t ops = 0, bad_ops = 0;
    for (uint32_t q : qs) {
        const HeckeDatum Dq = MakeDatum(q, 1, c.p, c.n);
        const BlockCochainMap F = ModelHecke(*lm, q, Dq, c.n, ctx);
        const std::vector<uint32_t> Y = F.apply(lm->f1->basis(), t1);
        lm->f1->require_equivariant_cocycles(Y, t1, "(K) model operator");
        const HeilbronnOperator H(*lm->lvl, q, c.n, ctx);
        bool ok = true;
        for (uint32_t j = 0; j < t1 && ok; ++j) {
            const std::vector<uint32_t> rhs = H.apply(CochainAt(lm->f1->basis(), t1, j, pts.e46, D));
            ok = InRel(Q, Sub(ctx, CochainAt(Y, t1, j, pts.e46, D), rhs));
            NT().K += InRel(Q, rhs) ? 0 : 1;
        }
        g.check(ok, tag + ": (LN7) (" + H.label() + "~ f)(e_46) - H_{N," + std::to_string(q) + "} f(e_46) in Rel_N for every basis cocycle");
        ++ops;
        bad_ops += c.N % q == 0 ? 1 : 0;
    }
    // (3) / (LN7): alpha^0, omega commute with f -> f(e_46) exactly; the e_46 blocks are the fibre maps' blocks.
    uint32_t fibre_pairs = 0;
    for (uint64_t M : c.sub) {
        auto lM = BuildLevelModel(M, c.n, ctx);
        const LevelChangeMap res(*lM->lvl, *lm->lvl, LevelChangeKind::kRestriction, c.n, ctx);
        const LevelChangeMap tr(*lM->lvl, *lm->lvl, LevelChangeKind::kTrace, c.n, ctx);
        const BlockCochainMap Fr = BuildLevelCorrespondence(ctx, RestrictionSpec(*lM->lvl, *lm->lvl), *TheModel().K, c.n, 1);
        const BlockCochainMap Ft = BuildLevelCorrespondence(ctx, TraceSpec(*lm->lvl, *lM->lvl), *TheModel().K, c.n, 1);
        const uint32_t tM = lM->f1->t();
        const std::vector<uint32_t> Yr = Fr.apply(lM->f1->basis(), tM), Yt = Ft.apply(lm->f1->basis(), t1);
        bool er = true, et = true;
        for (uint32_t j = 0; j < tM; ++j)
            er = er && CochainAt(Yr, tM, j, pts.e46, D) == res.apply(CochainAt(lM->f1->basis(), tM, j, pts.e46, lM->D));
        for (uint32_t j = 0; j < t1; ++j)
            et = et && CochainAt(Yt, t1, j, pts.e46, lM->D) == tr.apply(CochainAt(lm->f1->basis(), t1, j, pts.e46, D));
        const std::string pt = tag + " M=" + std::to_string(M);
        g.check(er, pt + ": (LN7) alpha^0 commutes with f -> f(e_46) EXACTLY (" + std::to_string(tM) + " cocycles)");
        g.check(et, pt + ": (LN7) omega commutes with f -> f(e_46) EXACTLY (" + std::to_string(t1) + " cocycles)");
        // the e_46 x e_46 blocks of the model's cochain maps = the fibre maps' dense matrices, entry for entry
        auto e46_block = [&](const BlockCochainMap& F, uint32_t rows, uint32_t cols) {
            std::vector<uint32_t> out(static_cast<size_t>(rows) * cols, 0);
            bool clean = true;
            for (const auto& kv : F.aggregate()) {
                const auto& key = kv.first;
                if (key[0] != pts.e46) continue;
                if (key[2] != pts.e46) { clean = false; continue; }
                for (uint32_t r = 0; r < F.d; ++r)
                    for (uint32_t s = 0; s < F.d; ++s)
                        out[static_cast<size_t>((key[1] - 1) * F.d + r) * cols + (key[3] - 1) * F.d + s] =
                            ctx.add(out[static_cast<size_t>((key[1] - 1) * F.d + r) * cols + (key[3] - 1) * F.d + s],
                                    kv.second[static_cast<size_t>(r) * F.d + s]);
            }
            return std::make_pair(clean, out);
        };
        const auto br = e46_block(Fr, res.dst_dim(), res.src_dim()), bt = e46_block(Ft, tr.dst_dim(), tr.src_dim());
        g.check(br.first && br.second == res.dense() && bt.first && bt.second == tr.dense(),
                pt + ": (LN7) the e_46 blocks of the model's alpha^0 and omega are LevelChangeMap's, entry for entry");
        ++fibre_pairs;
    }
    std::ostringstream os;
    os << "LEVEL_N_KAPPA N=" << c.N << " n=" << c.n << " p=" << c.p << " D=" << D << " h1=" << t1 << " h0=" << t0
       << " operators=" << ops << " bad_prime_operators=" << bad_ops << " fibre_pairs=" << fibre_pairs;
    lines.push_back(os.str());
}

// ---------------------------------------------------------------------------
// (A) (LN8): prop:degenpres, the presentation against the model.
// ---------------------------------------------------------------------------
struct ACell { uint64_t M, N, d; uint32_t n; uint32_t p; };

void ArmAlpha(Checker& g, const ACell& c, std::vector<std::string>& lines, uint32_t& model_cells) {
    const engine::ModCtx<uint32_t> ctx(c.p);
    const std::string tag = Tag4("(A)", c.M, c.N, c.d, c.n, c.p);
    const KappaPoints pts = KappaModelPoints(*TheModel().K);
    const Gamma0LevelN lM(c.M), lN(c.N);
    const SparseCokernel QM = SparseCokernel::Build(lM, c.n, ctx, true), QN = SparseCokernel::Build(lN, c.n, ctx, true);
    const DegeneracyOperator D(lM, lN, c.d, c.n, ctx);
    const DegeneracyOperator Dscan(lM, lN, c.d, c.n, ctx, DegeneracyEnumeration::kScan);
    g.check(D.canonical_terms() == Dscan.canonical_terms(),
            tag + ": per-eps enumeration = scan, the same " + std::to_string(D.terms().size()) + " terms (prop:degenpres (7))");
    // (1)
    const uint32_t esc = DegeneracyRelEscapes(QM, QN, D);
    g.check(esc == 0, tag + ": (LN8)(1) D(Rel_M) in Rel_N on every generator (" + std::to_string(QM.generators().size()) + ")");
    // (3) d = 1 is alpha^0 entrywise
    if (c.d == 1)
        g.check(D.dense() == LevelChangeMap(lM, lN, LevelChangeKind::kRestriction, c.n, ctx).dense(),
                tag + ": (LN8)(3) D_{M,N,1} = alpha^0 entry for entry");
    // (2) against the model: (alpha_d~ f)(e_46) - D f(e_46) in Rel_N
    bool model_ok = true;
    std::string model_note = "skipped";
    if (static_cast<uint64_t>(lN.m()) * (c.n + 1) <= 260) {
        auto mM = BuildLevelModel(c.M, c.n, ctx);
        auto mN = BuildLevelModel(c.N, c.n, ctx);
        const uint32_t tM = mM->f1->t();
        const std::vector<uint32_t> Y = ModelAlpha(c.M, c.N, c.d, c.n, ctx, mM->f1->basis(), tM);
        mN->f1->require_equivariant_cocycles(Y, tM, "(A) model alpha_d");
        for (uint32_t j = 0; j < tM && model_ok; ++j) {
            const std::vector<uint32_t> rhs = D.apply(CochainAt(mM->f1->basis(), tM, j, pts.e46, mM->D));
            model_ok = InRel(QN, Sub(ctx, CochainAt(Y, tM, j, pts.e46, mN->D), rhs));
            NT().A += InRel(QN, rhs) ? 0 : 1;
        }
        g.check(model_ok, tag + ": (LN8)(2) (alpha_d~ f)(e_46) - D_{M,N,d} f(e_46) in Rel_N for every basis cocycle (" +
                              std::to_string(tM) + ")");
        model_note = model_ok ? "held" : "FAILED";
        ++model_cells;
    }
    // Hecke equivariance T_q alpha_d = alpha_d T_q, q !| N
    uint32_t heq = 0;
    for (uint32_t q : {5u, 7u, 11u}) {
        if (c.N % q == 0 || q == c.p) continue;
        const HeilbronnOperator HM(lM, q, c.n, ctx), HN(lN, q, c.n, ctx);
        bool ok = true;
        for (uint32_t j = 0; j < QM.h1() && ok; ++j) {
            std::vector<uint32_t> e(QM.D(), 0);
            e[QM.free_columns()[j]] = 1;
            ok = InRel(QN, Sub(ctx, HN.apply(D.apply(e)), D.apply(HM.apply(e))));
        }
        g.check(ok, tag + ": (LN8) Hecke equivariance H_{N," + std::to_string(q) + "} D = D H_{M," + std::to_string(q) + "} mod Rel_N");
        ++heq;
    }
    // (4) composite d: every factorisation d = d1 d2, d1, d2 > 1
    uint32_t comps = 0;
    for (uint64_t d1 = 2; d1 < c.d; ++d1) {
        if (c.d % d1) continue;
        const uint64_t d2 = c.d / d1;
        const Gamma0LevelN lC(c.M * d1);
        const DegeneracyOperator D1(lM, lC, d1, c.n, ctx), D2(lC, lN, d2, c.n, ctx);
        bool ok = true;
        for (uint32_t j = 0; j < QM.h1() && ok; ++j) {
            std::vector<uint32_t> e(QM.D(), 0);
            e[QM.free_columns()[j]] = 1;
            ok = InRel(QN, Sub(ctx, D.apply(e), D2.apply(D1.apply(e))));
        }
        g.check(ok, tag + ": (LN8)(4) [D_{M,N,d}] = [D_{Md1,N,d2}] [D_{M,Md1,d1}] at d1=" + std::to_string(d1));
        ++comps;
    }
    std::ostringstream os;
    os << "LEVEL_N_DEGEN (A) M=" << c.M << " N=" << c.N << " d=" << c.d << " n=" << c.n << " p=" << c.p << " h1_M=" << QM.h1()
       << " h1_N=" << QN.h1() << " merel=" << D.merel_size() << " terms=" << D.terms().size() << " pair_tests=" << D.pair_tests()
       << " scan_tests=" << Dscan.pair_tests() << " rel_escapes=" << esc << " model=" << model_note << " hecke_eq=" << heq
       << " composites=" << comps;
    lines.push_back(os.str());
}

// The traps of rmk:degenpresconv (2): each must fail (1) or disagree with the model.
void ArmTraps(Checker& g, uint64_t M, uint64_t N, uint64_t d, uint32_t n, std::vector<std::string>& lines) {
    const engine::ModCtx<uint32_t> ctx(kP);
    const KappaPoints pts = KappaModelPoints(*TheModel().K);
    const Gamma0LevelN lM(M), lN(N);
    const SparseCokernel QM = SparseCokernel::Build(lM, n, ctx, true), QN = SparseCokernel::Build(lN, n, ctx, true);
    auto mM = BuildLevelModel(M, n, ctx);
    const uint32_t tM = mM->f1->t();
    const std::vector<uint32_t> Y = ModelAlpha(M, N, d, n, ctx, mM->f1->basis(), tM);
    for (DegeneracyVariant v : {DegeneracyVariant::kTrapA, DegeneracyVariant::kTrapB}) {
        const DegeneracyOperator T(lM, lN, d, n, ctx, DegeneracyEnumeration::kScan, v);
        const uint32_t esc = DegeneracyRelEscapes(QM, QN, T);
        bool agree = true;
        for (uint32_t j = 0; j < tM && agree; ++j)
            agree = InRel(QN, Sub(ctx, CochainAt(Y, tM, j, pts.e46, QN.D()), T.apply(CochainAt(mM->f1->basis(), tM, j, pts.e46, QM.D()))));
        // (slot-16 WP-17, audit WP-13 F6) BOTH signals: the trap leaves Rel_N AND disagrees with the model.
        g.check(esc > 0 && !agree, Tag4("(A) negative control", M, N, d, n, kP) + ": " + T.label() +
                                       " fails both ways (rel_escapes=" + std::to_string(esc) + ", model " +
                                       (agree ? "agrees" : "disagrees") + ")");
        lines.push_back("LEVEL_N_DEGEN (A) negctl M=" + std::to_string(M) + " N=" + std::to_string(N) + " d=" + std::to_string(d) +
                        " n=" + std::to_string(n) + " variant=" + T.label() + " rel_escapes=" + std::to_string(esc) +
                        " model=" + (agree ? "agrees" : "disagrees"));
    }
}

// ---------------------------------------------------------------------------
// (W) prop:degenpres (5), the trace, on the presentation; the old-space relations (reported).
// ---------------------------------------------------------------------------
enum class TraceKind { kTp, kUp, kTp2 };
struct WCell { uint64_t M; uint64_t p; uint32_t nu; uint32_t n; TraceKind kind; };

void ArmTrace(Checker& g, const WCell& c, std::vector<std::string>& lines) {
    const engine::ModCtx<uint32_t> ctx(kP);
    uint64_t d = 1;
    for (uint32_t e = 0; e < c.nu; ++e) d *= c.p;
    const uint64_t N = d * c.M;
    const char* kind = c.kind == TraceKind::kTp ? "T_p" : (c.kind == TraceKind::kUp ? "U_p" : "T_p^2-p^n(p+1)");
    const std::string tag = Tag4("(W)", c.M, N, d, c.n, kP) + " " + kind;
    if (c.kind == TraceKind::kUp) ENGINE_CHECK_MSG(c.M % c.p == 0 && c.nu == 1, "(W): U_p needs d = p | M");
    if (c.kind == TraceKind::kTp) ENGINE_CHECK_MSG(c.M % c.p != 0 && c.nu == 1, "(W): T_p needs d = p !| M");
    if (c.kind == TraceKind::kTp2) ENGINE_CHECK_MSG(c.M == 1 && c.nu == 2, "(W): T(1, p^2) needs M = 1, d = p^2");
    const Gamma0LevelN lM(c.M), lN(N);
    const SparseCokernel QM = SparseCokernel::Build(lM, c.n, ctx, true);
    const DegeneracyOperator D(lM, lN, d, c.n, ctx);
    const LevelChangeMap om(lM, lN, LevelChangeKind::kTrace, c.n, ctx);
    const HeilbronnOperator H(lM, c.p, c.n, ctx);
    const uint64_t pn = powmod<uint64_t>(c.p, c.n, kP);
    bool ok = true;
    for (uint32_t j = 0; j < QM.h1() && ok; ++j) {
        std::vector<uint32_t> e(QM.D(), 0);
        e[QM.free_columns()[j]] = 1;
        const std::vector<uint32_t> lhs = om.apply(D.apply(e));
        std::vector<uint32_t> rhs = H.apply(e);
        if (c.kind == TraceKind::kTp2) rhs = Sub(ctx, H.apply(rhs), Scale(ctx, e, pn * (c.p + 1) % kP));
        ok = InRel(QM, Sub(ctx, lhs, rhs));
        NT().W += InRel(QM, rhs) ? 0 : 1;
    }
    g.check(ok, tag + ": (LN8) prop:degenpres (5) [omega] [D_{M,dM,d}] = [Gamma_0(M) diag(1,d) Gamma_0(M)] on M_M / Rel_M (" +
                    std::to_string(QM.h1()) + " basis vectors)");
    lines.push_back("LEVEL_N_DEGEN (W) M=" + std::to_string(c.M) + " d=" + std::to_string(d) + " n=" + std::to_string(c.n) +
                    " operator=" + kind + " h1_M=" + std::to_string(QM.h1()) + " result=" + (ok ? "held" : "FAILED"));
}

// Old-space relations at M -> Mp, p !| M (STD atkin_lehner, not verified: reported).
void ArmOldSpace(Checker& g, uint64_t M, uint64_t p, uint32_t n, std::vector<std::string>& lines) {
    (void)g;
    const engine::ModCtx<uint32_t> ctx(kP);
    const uint64_t N = M * p;
    const Gamma0LevelN lM(M), lN(N);
    const SparseCokernel QM = SparseCokernel::Build(lM, n, ctx, false), QN = SparseCokernel::Build(lN, n, ctx, false);
    const DegeneracyOperator A1(lM, lN, 1, n, ctx), Ap(lM, lN, p, n, ctx);
    const HeilbronnOperator U(lN, p, n, ctx), T(lM, p, n, ctx);
    const uint64_t pn = powmod<uint64_t>(p, n, kP);
    bool o1 = true, o3 = true;
    for (uint32_t j = 0; j < QM.h1(); ++j) {
        std::vector<uint32_t> e(QM.D(), 0);
        e[QM.free_columns()[j]] = 1;
        const std::vector<uint32_t> a1 = A1.apply(e), ap = Ap.apply(e);
        o1 = o1 && InRel(QN, Sub(ctx, U.apply(a1), Scale(ctx, ap, p)));
        o3 = o3 && InRel(QN, Sub(ctx, U.apply(ap), Sub(ctx, Ap.apply(T.apply(e)), Scale(ctx, a1, pn))));
    }
    std::ostringstream os;
    os << "LEVEL_N_DEGEN (W) oldspace M=" << M << " p=" << p << " n=" << n << " U_p_alpha1=p_alpha_p:" << (o1 ? "held" : "FINDING")
       << " U_p_alpha_p=alpha_p_T_p-p^n_alpha1:" << (o3 ? "held" : "FINDING") << " (STD atkin_lehner, reported, not gated)";
    lines.push_back(os.str());
    std::cout << "  " << os.str() << std::endl;
}

// ---------------------------------------------------------------------------
// (L9) (LN9): R3_k, reported.
// ---------------------------------------------------------------------------
uint32_t ValP(uint64_t x_mod_p3, uint64_t p) {   // v_p of a residue mod p^3 (3 = "at least 3")
    if (x_mod_p3 == 0) return 3;
    uint32_t v = 0;
    while (x_mod_p3 % p == 0) { x_mod_p3 /= p; ++v; }
    return v;
}

void ArmR3k(Checker& g, uint64_t N, uint64_t p, uint32_t k, std::vector<std::string>& lines, uint32_t& findings) {
    (void)g;
    // (slot-16 WP-17, audit WP-13 F4) p^3 is formed in uint64_t below: p < 2^21 (the hook refuses larger p by name).
    ENGINE_CHECK_MSG(p < (1ull << 21), "(L9) R3_k: p < 2^21 (p^3 must fit uint64_t)");
    const uint32_t n = k - 2;
    oc_util::ResetPeakRss();
    const auto t0 = std::chrono::steady_clock::now();
    const R3Reading R = ReadEisensteinR3k(N, p, n, 7);
    const double wall = Seconds(t0);
    const uint64_t p3 = p * p * p;
    const uint32_t vNk = ValP((powmod<uint64_t>(N, k, p3) + p3 - 1) % p3, p), vNk2 = ValP((powmod<uint64_t>(N, k - 2, p3) + p3 - 1) % p3, p);
    uint32_t vB = 0;
    std::string bnote;
    if (p > k + 1) {
        const uint64_t b = hecke_ext::hidacomp::BernoulliOverKDirect(k, p, 3);
        vB = ValP(b, p);
    } else {
        bnote = "(p<=k+1)";
    }
    const bool hyp = hecke::IsPrimeU64(N) && p > k + 1 && k >= 4 && N % p != 0;
    const bool divides = vNk > 0 || vNk2 > 0 || vB > 0;
    std::string verdict;
    if (!hyp) verdict = "OUT_OF_HYPOTHESIS";
    else if (R.dimW < 2) verdict = "FINDING(dimW<2)";
    else if (R.dimW >= 3 && !divides) verdict = "FINDING(dimW>=3_without_divisibility)";
    else verdict = "HELD";
    if (verdict.rfind("FINDING", 0) == 0) ++findings;
    std::ostringstream os, ls;
    for (size_t i = 0; i < R.layers.size(); ++i) ls << (i ? "," : "") << R.layers[i];
    os << "LEVEL_N_R3K N=" << N << " p=" << p << " k=" << k << " dimH1=" << R.dimH1 << " dimW=" << R.dimW << " loewy=" << R.loewy
       << " layers=" << ls.str() << " v_p(N^k-1)=" << vNk << " v_p(N^(k-2)-1)=" << vNk2 << " v_p(B_k/k)=" << vB << bnote
       << " prediction=" << verdict << " wall_s=" << wall << " peak_rss_mib=" << oc_util::PeakRssMiB() << " (CONJ, reported)";
    lines.push_back(os.str());
}

// ---------------------------------------------------------------------------
// (G) alpha_d at scale on the sparse route.
// ---------------------------------------------------------------------------
struct GCell { uint64_t M, N, d; uint32_t n; uint32_t p; };

void ArmScale(Checker& g, const GCell& c, std::vector<std::string>& lines) {
    const engine::ModCtx<uint32_t> ctx(c.p);
    const std::string tag = Tag4("(G)", c.M, c.N, c.d, c.n, c.p);
    const CokernelCost cmM = CokernelCostModel(c.M, c.n), cmN = CokernelCostModel(c.N, c.n);
    oc_util::ResetPeakRss();
    ResetLevelNWideProducts();   // (slot-17 WP-5) the __int128 escalations of this cell's coset products
    const auto t0 = std::chrono::steady_clock::now();
    const Gamma0LevelN lM(c.M), lN(c.N);
    const SparseCokernel QM = SparseCokernel::Build(lM, c.n, ctx, true);
    const SparseCokernel QN = SparseCokernel::Build(lN, c.n, ctx, false);
    const double t_pres = Seconds(t0);
    const auto t1 = std::chrono::steady_clock::now();
    const DegeneracyOperator D(lM, lN, c.d, c.n, ctx);
    const double t_op = Seconds(t1);
    const auto t2 = std::chrono::steady_clock::now();
    const uint32_t esc = DegeneracyRelEscapes(QM, QN, D);
    g.check(esc == 0, tag + ": (LN8)(1) D(Rel_M) in Rel_N on every generator (" + std::to_string(QM.generators().size()) + ")");
    const double t_rel = Seconds(t2);
    // sampled free vectors of M_M / Rel_M
    const uint32_t S = std::min<uint32_t>(32, QM.h1());
    auto sample = [&](uint32_t s) {
        SparseVec e;
        e.col.push_back(QM.free_columns()[static_cast<size_t>(s) * QM.h1() / std::max<uint32_t>(1, S)]);
        e.val.push_back(1);
        return e;
    };
    // Hecke equivariance at the smallest q in {5, ..., 23} with q !| N p.
    uint32_t qh = 0;
    for (uint32_t q : {5u, 7u, 11u, 13u, 17u, 19u, 23u})
        if (c.N % q && q != c.p) { qh = q; break; }
    bool heq = true;
    uint64_t wide = 0;
    if (qh) {
        const HeilbronnOperator HM(lM, qh, c.n, ctx), HN(lN, qh, c.n, ctx);
        wide = HM.wide_terms() + HN.wide_terms();
        const HeilbronnSparseApply aM(HM), aN(HN);
        for (uint32_t s = 0; s < S && heq; ++s) {
            const SparseVec e = sample(s);
            heq = InRel(QN, Sub(ctx, ToDense(aN.apply(D.apply_sparse(e)), QN.D()), ToDense(D.apply_sparse(aM.apply(e)), QN.D())));
        }
        g.check(heq, tag + ": (LN8) H_{N," + std::to_string(qh) + "} D = D H_{M," + std::to_string(qh) + "} mod Rel_N on " +
                         std::to_string(S) + " sampled basis vectors");
    }
    // the trace (5) when N = d M and d is prime
    std::string trace = "n/a";
    if (c.N == c.d * c.M && hecke::IsPrimeU64(c.d)) {
        const LevelChangeMap om(lM, lN, LevelChangeKind::kTrace, c.n, ctx);
        const HeilbronnOperator H(lM, c.d, c.n, ctx);
        const HeilbronnSparseApply aH(H);
        bool ok = true;
        for (uint32_t s = 0; s < S && ok; ++s) {
            const SparseVec e = sample(s);
            const std::vector<uint32_t> rhs = ToDense(aH.apply(e), QM.D());
            ok = InRel(QM, Sub(ctx, ToDense(om.apply_sparse(D.apply_sparse(e)), QM.D()), rhs));
            NT().G += InRel(QM, rhs) ? 0 : 1;
        }
        trace = std::string(c.M % c.d == 0 ? "U_p:" : "T_p:") + (ok ? "held" : "FAILED");
        g.check(ok, tag + ": (LN8) prop:degenpres (5) omega D = " + (c.M % c.d == 0 ? "U_" : "T_") + std::to_string(c.d) +
                        " mod Rel_M on " + std::to_string(S) + " sampled basis vectors");
    }
    const double wall = Seconds(t0);
    std::ostringstream os;
    os << "DEGEN_CELL M=" << c.M << " N=" << c.N << " d=" << c.d << " n=" << c.n << " p=" << c.p << " m_M=" << lM.m()
       << " m_N=" << lN.m() << " D_N=" << QN.D() << " h1_M=" << QM.h1() << " h1_N=" << QN.h1()
       << " final_nnz_M=" << QM.stats().final_nnz << " final_nnz_N=" << QN.stats().final_nnz << " merel=" << D.merel_size()
       << " terms=" << D.terms().size() << " pair_tests=" << D.pair_tests() << " max_adm_per_eps=" << D.max_admissible_per_eps()
       << " rel_escapes=" << esc << " hecke_q=" << qh << " heilbronn_wide_terms=" << wide << " trace=" << trace
       << " sparse_bytes_pred_M=" << cmM.sparse_bytes << " sparse_bytes_pred_N=" << cmN.sparse_bytes << " presentation_s=" << t_pres
       << " operator_s=" << t_op << " rel_s=" << t_rel << " wall_s=" << wall << " peak_rss_mib=" << oc_util::PeakRssMiB()
       << " level_wide_products=" << LevelNWideProducts();
    lines.push_back(os.str());
    std::cout << "  " << os.str() << std::endl;
}

// (slot-16 WP-17, audit WP-13 F3) Strict: every field is a non-empty decimal digit string (no sign, no blank, at most 18
// digits), an empty cell is refused by its position, and a value that yields no cell at all is refused -- the hook can
// never run an arm vacuously.
std::vector<std::vector<uint64_t>> ParseCells(const char* env, const char* name, size_t fields) {
    std::vector<std::vector<uint64_t>> out;
    const std::string all(env);
    ENGINE_CHECK_MSG(!all.empty(), (std::string(name) + ": the value lists no cell (refused: the arm would run vacuously)").c_str());
    ENGINE_CHECK_MSG(all.back() != ',', (std::string(name) + ": an empty trailing cell").c_str());
    std::stringstream ss(all);
    std::string item;
    size_t idx = 0;
    while (std::getline(ss, item, ',')) {
        ++idx;
        const std::string where = std::string(name) + ": cell " + std::to_string(idx) + " '" + item + "': ";
        ENGINE_CHECK_MSG(!item.empty() && item.back() != ':', (where + "an empty cell or field").c_str());
        std::vector<uint64_t> v;
        std::stringstream is(item);
        std::string f;
        while (std::getline(is, f, ':')) {
            ENGINE_CHECK_MSG(!f.empty() && f.size() <= 18 &&
                                 std::all_of(f.begin(), f.end(), [](char ch) { return ch >= '0' && ch <= '9'; }),
                             (where + "field '" + f + "' is not a decimal number").c_str());
            v.push_back(std::strtoull(f.c_str(), nullptr, 10));
        }
        ENGINE_CHECK_MSG(v.size() == fields, (where + "expected " + std::to_string(fields) + " colon-separated fields").c_str());
        out.push_back(v);
    }
    ENGINE_CHECK_MSG(!out.empty(), (std::string(name) + ": the value lists no cell (refused: the arm would run vacuously)").c_str());
    return out;
}

} // namespace

// slot-17 WP-8: the shared hook helpers, public (HeckeLevelNDegenValidation.h) so HeckeLevelNSparseValidation's
// HECKE_LEVEL_N_SPARSE_CELLS parses and refuses exactly like DEGEN / R3K.
std::vector<std::vector<uint64_t>> HookParseCells(const char* env, const char* name, size_t fields) { return ParseCells(env, name, fields); }

double HookRssFactor(HookKind kind, uint64_t N, uint32_t n) {
    // hook_rss/v1 -- see HeckeLevelNDegenValidation.h for the measured ratios behind each constant.
    const auto f = FactorLevel(N);
    if (f.size() >= 2 && f.back().q > 13) return 64.0;   // the W5-F1 regime (99330): uncalibrated, > 50 measured
    if (n == 0) return 16.0;
    return kind == HookKind::kSparse ? 4.0 : 6.0;
}

double HookPredictedRssBytes(HookKind kind, uint64_t N, uint32_t n) {
    return HookRssFactor(kind, N, n) * CokernelCostModel(N, n).sparse_bytes;
}

HookRssCap HookRssCapFrom(const char* value) {
    HookRssCap c;
    if (value == nullptr) return c;
    const std::string v(value);
    ENGINE_CHECK_MSG(!v.empty() && v.size() <= 4 && std::all_of(v.begin(), v.end(), [](char ch) { return ch >= '0' && ch <= '9'; }),
                     ("HECKE_LEVEL_N_HOOK_RSS_CAP_GIB='" + v + "': whole GiB in [1, 1024] expected").c_str());
    const uint64_t g = std::strtoull(v.c_str(), nullptr, 10);
    ENGINE_CHECK_MSG(g >= 1 && g <= 1024, ("HECKE_LEVEL_N_HOOK_RSS_CAP_GIB='" + v + "': whole GiB in [1, 1024] expected").c_str());
    c.gib = g;
    c.overridden = true;
    return c;
}

HookRssCap HookRssCapFromEnv() { return HookRssCapFrom(std::getenv("HECKE_LEVEL_N_HOOK_RSS_CAP_GIB")); }

std::string HookRssCapText(const HookRssCap& cap) {
    return " rss_cap_gib=" + std::to_string(cap.gib) +
           " rss_cap_source=" + (cap.overridden ? "HECKE_LEVEL_N_HOOK_RSS_CAP_GIB" : "default") + " rss_model=hook_rss/v1";
}

void HookRefuseOverCap(const char* name, size_t idx, const std::string& what, double predicted_bytes, const HookRssCap& cap) {
    const double cap_bytes = static_cast<double>(cap.gib) * 1024.0 * 1024.0 * 1024.0;
    const uint64_t mib = static_cast<uint64_t>(predicted_bytes / (1024.0 * 1024.0));
    ENGINE_CHECK_MSG(predicted_bytes <= cap_bytes,
                     (std::string(name) + ": cell " + std::to_string(idx) + ": " + what + ": predicted peak RSS " + std::to_string(mib) +
                      " MiB (hook_rss/v1) > the " + std::to_string(cap.gib) + " GiB cap" +
                      (cap.overridden ? " (HECKE_LEVEL_N_HOOK_RSS_CAP_GIB)" : "") +
                      " (rule 19; refused before the run; HECKE_LEVEL_N_HOOK_RSS_CAP_GIB=<G> raises the cap for a deliberate run)")
                         .c_str());
    std::cout << "  HOOK_RSS " << name << " cell=" << idx << " " << what << " predicted_mib=" << mib << " cap_gib=" << cap.gib << std::endl;
}

uint32_t HeckeLevelNDegenArms(uint32_t& checks) {
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    std::vector<std::string> lines;
    std::cout << "  -- WP-13 arms ([C] sec:levelNdegsec: (K) (LN7), (A) / (W) (LN8), (L9) (LN9) reported, (G) alpha_d at scale)"
              << std::endl;

    // ---- (K) (LN7): composite N, k > 2, fields F_P and small p, bad primes q | N.
    const std::vector<KCell> kcells{
        {1, 10, kP, {}},        {2, 4, kP, {1}},       {6, 2, kP, {1, 2, 3}}, {6, 4, kP, {2}},    {10, 2, kP, {2, 5}},
        {12, 2, kP, {4, 6}},    {15, 0, kP, {3, 5}},   {11, 2, kP, {1}},      {30, 0, kP, {6, 10, 15}}, {9, 2, kP, {3}},
        {35, 0, kP, {5, 7}},    {8, 4, kP, {4}},       {10, 2, 7, {5}},       {14, 2, 5, {7}},    {12, 4, 11, {6}},
        {13, 2, 13, {1}},       {22, 2, kP, {11}}};
    for (const KCell& c : kcells) ArmKappa(g, c, lines);
    std::cout << "  [(K) " << kcells.size() << " cells done at " << Seconds(t0) << " s]" << std::endl;

    // ---- (A) (LN8): WP-12's 28 COMP^a cells (degen.py) plus d = p | M, k > 2 at composite N, small fields.
    const std::vector<ACell> acells{
        {1, 2, 1, 10, kP}, {1, 2, 2, 10, kP}, {1, 3, 3, 10, kP}, {1, 4, 2, 6, kP},  {1, 4, 4, 6, kP},  {1, 5, 5, 4, kP},
        {1, 6, 2, 4, kP},  {1, 6, 3, 4, kP},  {1, 6, 6, 4, kP},  {2, 4, 2, 4, kP},  {2, 6, 3, 4, kP},  {3, 6, 2, 4, kP},
        {2, 12, 2, 2, kP}, {2, 12, 3, 2, kP}, {2, 12, 6, 2, kP}, {3, 12, 4, 2, kP}, {5, 10, 2, 2, kP}, {5, 15, 3, 2, kP},
        {6, 30, 5, 0, kP}, {11, 22, 2, 0, kP}, {1, 9, 9, 2, kP}, {3, 9, 3, 2, kP},  {1, 8, 4, 2, kP},  {2, 8, 4, 2, kP},
        {7, 35, 5, 2, kP}, {5, 35, 7, 2, kP}, {1, 7, 7, 4, kP},  {1, 1, 1, 10, kP},
        // beyond WP-12: d = p | M (the trace cells' presentation-vs-model), M > 1 at k > 2, small fields, d = 1 at composite N
        {6, 12, 2, 2, kP}, {5, 25, 5, 0, kP}, {4, 8, 2, 2, kP},  {3, 18, 6, 0, kP}, {2, 20, 10, 0, kP}, {6, 36, 6, 0, kP},
        {10, 20, 2, 2, 7}, {6, 18, 3, 2, 5}, {5, 30, 6, 0, 11}, {6, 30, 1, 2, kP}, {15, 30, 2, 0, 7}, {1, 12, 12, 0, kP},
        // at scale (presentation only: the model is skipped above D_N = 260)
        {30, 210, 7, 2, kP}, {105, 210, 2, 4, kP}, {70, 420, 6, 0, 13}};
    uint32_t model_cells = 0;
    for (const ACell& c : acells) ArmAlpha(g, c, lines, model_cells);
    // (slot-16 WP-17, audit WP-13 F2) the model comparison may not erode silently: the cells with D_N <= 260 are
    // exactly the ones compared against the model, and there are at least 40 of them.
    {
        uint32_t want = 0;
        for (const ACell& c : acells)
            if (static_cast<uint64_t>(Gamma0LevelN(c.N).m()) * (c.n + 1) <= 260) ++want;
        g.check(model_cells == want && want >= 40, "(A) the model comparison ran on every cell with D_N <= 260 (" +
                                                       std::to_string(model_cells) + " of " + std::to_string(want) +
                                                       " expected, >= 40 required; " + std::to_string(acells.size()) + " cells)");
    }
    for (const auto& t : std::vector<std::array<uint64_t, 4>>{{1, 2, 2, 10}, {1, 6, 3, 4}, {2, 12, 3, 2}, {1, 4, 4, 6}})
        ArmTraps(g, t[0], t[1], t[2], static_cast<uint32_t>(t[3]), lines);
    // ---- (F1) (slot-16 WP-17, audit WP-13 F1): the __int128 fallback of HeilbronnOperator, FORCED on every term by
    // ScopedForceWideHeilbronn, against the int64 route: the same term list (dst, src, block entries) and the same
    // dense matrix, at composite N with case-(iii) classes, q !| N and q | N, prop. and trap variants.
    {
        const engine::ModCtx<uint32_t> ctx(kP);
        auto canon = [](const HeilbronnOperator& H) {
            std::vector<std::vector<uint32_t>> out;
            for (const HeilbronnTerm& t : H.terms()) {
                std::vector<uint32_t> k{t.dst, t.src};
                const std::vector<uint32_t>& B = H.blocks()[t.block].dense;
                k.insert(k.end(), B.begin(), B.end());
                out.push_back(std::move(k));
            }
            return out;   // in construction order: the two routes must agree term by term, not only as multisets
        };
        uint32_t f1_cells = 0;
        uint64_t f1_terms = 0, f1_mixed = 0;
        for (const auto& c : std::vector<std::array<uint64_t, 4>>{{30, 5, 2, 0}, {30, 7, 2, 0}, {30, 13, 2, 0}, {210, 5, 2, 0},
                                                                  {210, 13, 0, 0}, {2310, 7, 0, 0}, {2310, 13, 0, 0},
                                                                  {30, 7, 2, 1}, {210, 7, 0, 1}}) {
            const Gamma0LevelN lvl(c[0]);
            const uint32_t n = static_cast<uint32_t>(c[2]);
            const HeilbronnVariant var = c[3] ? HeilbronnVariant::kTrapA : HeilbronnVariant::kProposition;
            const HeilbronnOperator H64(lvl, c[1], n, ctx, var);
            ScopedForceWideHeilbronn force;
            const HeilbronnOperator HW(lvl, c[1], n, ctx, var);
            const std::string tag = "(F1) N=" + std::to_string(c[0]) + " q=" + std::to_string(c[1]) + " n=" + std::to_string(n) +
                                    (c[3] ? " [trapA]" : "");
            g.check(H64.wide_terms() == 0 && HW.wide_terms() == HW.terms().size() && !HW.terms().empty(),
                    tag + ": the int64 route took no wide term; the forced route took every term wide (" +
                        std::to_string(HW.wide_terms()) + " of " + std::to_string(HW.terms().size()) + ")");
            g.check(canon(H64) == canon(HW) && H64.pairs_dropped() == HW.pairs_dropped() && H64.dense() == HW.dense(),
                    tag + ": the __int128 route = the int64 route, term for term and entry for entry (" +
                        std::to_string(H64.terms().size()) + " terms, " + std::to_string(H64.mixed_terms()) + " at case (iii))");
            ++f1_cells;
            f1_terms += HW.terms().size();
            f1_mixed += HW.mixed_terms();
        }
        g.check(f1_mixed > 0, "(F1) non-vacuity: the forced-wide comparison covers case-(iii) representatives");
        lines.push_back("LEVEL_N_HEILBRONN_WIDE cells=" + std::to_string(f1_cells) + " terms=" + std::to_string(f1_terms) +
                        " case_iii_terms=" + std::to_string(f1_mixed) + " result=" + (g.fails ? "see FAIL lines" : "held"));
    }
    std::cout << "  [(A) " << acells.size() << " cells (" << model_cells << " against the model) + 4 negative-control cells done at "
              << Seconds(t0) << " s]" << std::endl;

    // ---- (W) the trace (5), incl. d = p | M; the old-space relations (reported).
    const std::vector<WCell> wcells{
        {1, 2, 1, 10, TraceKind::kTp}, {1, 3, 1, 10, TraceKind::kTp}, {1, 5, 1, 4, TraceKind::kTp}, {2, 3, 1, 4, TraceKind::kTp},
        {3, 2, 1, 4, TraceKind::kTp},  {5, 2, 1, 2, TraceKind::kTp},  {1, 2, 2, 6, TraceKind::kTp2}, {1, 3, 2, 4, TraceKind::kTp2},
        // d = p | M: omega D_{M,pM,p} = U_p (proved, untested before WP-13)
        {2, 2, 1, 4, TraceKind::kUp},  {3, 3, 1, 2, TraceKind::kUp},  {5, 5, 1, 2, TraceKind::kUp},  {6, 2, 1, 2, TraceKind::kUp},
        {6, 3, 1, 2, TraceKind::kUp},  {4, 2, 1, 2, TraceKind::kUp},  {10, 5, 1, 0, TraceKind::kUp}, {7, 7, 1, 2, TraceKind::kUp},
        {12, 2, 1, 4, TraceKind::kUp}, {15, 5, 1, 2, TraceKind::kUp}, {11, 11, 1, 0, TraceKind::kUp}, {30, 3, 1, 0, TraceKind::kUp}};
    for (const WCell& c : wcells) ArmTrace(g, c, lines);
    for (const auto& o : std::vector<std::array<uint64_t, 3>>{{1, 2, 10}, {1, 3, 6}, {1, 5, 4}, {2, 3, 4}, {3, 2, 4}, {5, 2, 2},
                                                              {1, 11, 0}, {6, 5, 2}, {7, 2, 4}})
        ArmOldSpace(g, o[0], o[1], static_cast<uint32_t>(o[2]), lines);
    std::cout << "  [(W) " << wcells.size() << " trace cells done at " << Seconds(t0) << " s]" << std::endl;

    // ---- (L9) (LN9) R3_k: reported.  Two gates on the READING (not on the prediction):
    //  (i) the weight normalisation: charpoly(T_2 | H^1(SL_2(Z); Sym^10)) = (X + 24)^2 (X - 2049) (tau(2) = -24, 1 + 2^11);
    //  (ii) at n = 0 the reading equals the frozen weight-2 (R3) readings where q = p is not in the family anyway (p > 7).
    {
        const engine::ModCtx<uint32_t> ctx(kP);
        const Gamma0LevelN l1(1);
        const SparseCokernel Q1 = SparseCokernel::Build(l1, 10, ctx, true);
        const HMat T2 = OnCokernelSparse(Q1, HeilbronnOperator(l1, 2, 10, ctx));
        // (X + 24)^2 (X - 2049) = (X^2 + 48 X + 576)(X + b), b = -2049: low -> high
        const uint64_t b = (kP - 2049) % kP;
        const uint64_t c0 = 576ull * b % kP, c1 = (576 + 48ull * b) % kP, c2 = (48 + b) % kP;
        g.check(HCharpoly(T2) == std::vector<uint64_t>{c0, c1, c2, 1}, "(L9) weight normalisation: charpoly(H_{1,2} | n = 10) = (X + 24)^2 (X - 2049)");
        for (const auto& r : std::vector<std::array<uint64_t, 5>>{{23, 11, 5, 3, 2}, {53, 13, 9, 3, 2}, {67, 11, 11, 3, 2},
                                                                  {79, 13, 13, 3, 2}, {89, 11, 15, 3, 2}, {131, 13, 23, 3, 2}}) {
            const R3Reading R = ReadEisensteinR3k(r[0], r[1], 0, 7);
            g.check(R.dimH1 == r[2] && R.dimW == r[3] && R.loewy == r[4],
                    "(L9) ReadEisensteinR3k at n = 0 = the frozen (R3) reading at (N=" + std::to_string(r[0]) + ", p=" + std::to_string(r[1]) + ")");
        }
    }
    std::vector<std::array<uint64_t, 3>> r3k;   // (N, p, k)
    // default: N prime <= 113; k in {4, 6}; p in {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43} with p > k + 1, p !| N, and
    // (to keep the list short) only the cells where p divides (N^k - 1)(N^{k-2} - 1) -- the "only if" side -- plus one
    // control per (N, k) and the level-1-old cells at k = 12 (691), 16 (3617).
    {
        const std::vector<uint64_t> Ns{11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113};
        const std::vector<uint64_t> ps{7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43};
        for (uint64_t N : Ns)
            for (uint32_t k : {4u, 6u}) {
                bool control = false;
                for (uint64_t p : ps) {
                    if (p <= k + 1 || N == p) continue;
                    const bool div = powmod<uint64_t>(N, k, p) == 1 || powmod<uint64_t>(N, k - 2, p) == 1;
                    if (div) r3k.push_back({N, p, k});
                    else if (!control) { r3k.push_back({N, p, k}); control = true; }
                }
            }
        r3k.push_back({11, 691, 12});
        r3k.push_back({13, 691, 12});
        r3k.push_back({5, 3617, 16});
    }
    if (const char* env = std::getenv("HECKE_LEVEL_N_R3K_CELLS")) {
        r3k.clear();
        const HookRssCap cap = HookRssCapFromEnv();
        size_t idx = 0;
        for (const auto& v : ParseCells(env, "HECKE_LEVEL_N_R3K_CELLS", 3)) {
            ++idx;
            ENGINE_CHECK_MSG(v[0] >= 1 && v[0] < (1ull << 20), "HECKE_LEVEL_N_R3K_CELLS: N in [1, 2^20)");
            ENGINE_CHECK_MSG(v[1] >= 5 && v[1] < (1ull << 21) && hecke::IsPrimeU64(v[1]) && v[0] % v[1] != 0,
                             "HECKE_LEVEL_N_R3K_CELLS: p >= 5 prime, p < 2^21 (p^3 in uint64_t), p !| N");
            ENGINE_CHECK_MSG(v[2] >= 2 && v[2] % 2 == 0 && v[2] < 200, "HECKE_LEVEL_N_R3K_CELLS: k even in [2, 200)");
            {
                const uint32_t nn = static_cast<uint32_t>(v[2] - 2);
                const double rss = kHookRssBaselineMiB * 1024.0 * 1024.0 + HookPredictedRssBytes(HookKind::kR3k, v[0], nn);
                HookRefuseOverCap("HECKE_LEVEL_N_R3K_CELLS", idx,
                                  "N=" + std::to_string(v[0]) + " n=" + std::to_string(nn) + " factor=" +
                                      std::to_string(static_cast<int>(HookRssFactor(HookKind::kR3k, v[0], nn))),
                                  rss, cap);
            }
            r3k.push_back({v[0], v[1], v[2]});
        }
        std::cout << "  EFFECTIVE HeckeLevelNSparseValidation arm=R3K cells=" << r3k.size() << " (HECKE_LEVEL_N_R3K_CELLS)" << HookRssCapText(cap) << std::endl;
    }
    uint32_t findings = 0;
    for (const auto& r : r3k) ArmR3k(g, r[0], r[1], static_cast<uint32_t>(r[2]), lines, findings);
    std::cout << "  [(L9) " << r3k.size() << " R3_k readings done at " << Seconds(t0) << " s; " << findings
              << " findings against the (LN9) prediction (CONJ; reported, never gated)]" << std::endl;

    // ---- (G) alpha_d at scale (presentation only).
    std::vector<GCell> gcells{{1155, 2310, 2, 2, kP}, {770, 2310, 3, 4, kP}, {1001, 2002, 2, 4, 13}, {2002, 4004, 2, 2, kP}};
    if (const char* env = std::getenv("HECKE_LEVEL_N_DEGEN_CELLS")) {
        gcells.clear();
        const HookRssCap cap = HookRssCapFromEnv();
        size_t idx = 0;
        for (const auto& v : ParseCells(env, "HECKE_LEVEL_N_DEGEN_CELLS", 5)) {
            ++idx;
            ENGINE_CHECK_MSG(v[1] >= 1 && v[1] < (1ull << 20), "HECKE_LEVEL_N_DEGEN_CELLS: N in [1, 2^20)");
            ENGINE_CHECK_MSG(v[0] >= 1 && v[0] < (1ull << 20) && v[2] >= 1 && v[2] < (1ull << 20) && v[1] % (v[0] * v[2]) == 0,
                             "HECKE_LEVEL_N_DEGEN_CELLS: d M | N");
            ENGINE_CHECK_MSG(v[3] % 2 == 0 && v[3] < 200, "HECKE_LEVEL_N_DEGEN_CELLS: n even, < 200");
            ENGINE_CHECK_MSG(v[4] >= 5 && v[4] < (1ull << 31) && hecke::IsPrimeU64(v[4]), "HECKE_LEVEL_N_DEGEN_CELLS: p >= 5 prime");
            {
                const uint32_t nn = static_cast<uint32_t>(v[3]);
                const double rss = kHookRssBaselineMiB * 1024.0 * 1024.0 + HookPredictedRssBytes(HookKind::kDegen, v[0], nn) +
                                   HookPredictedRssBytes(HookKind::kDegen, v[1], nn);
                HookRefuseOverCap("HECKE_LEVEL_N_DEGEN_CELLS", idx,
                                  "M=" + std::to_string(v[0]) + " N=" + std::to_string(v[1]) + " n=" + std::to_string(nn) + " factor=" +
                                      std::to_string(static_cast<int>(HookRssFactor(HookKind::kDegen, v[1], nn))),
                                  rss, cap);
            }
            gcells.push_back({v[0], v[1], v[2], static_cast<uint32_t>(v[3]), static_cast<uint32_t>(v[4])});
        }
        std::cout << "  EFFECTIVE HeckeLevelNSparseValidation arm=DEGEN cells=" << gcells.size() << " (HECKE_LEVEL_N_DEGEN_CELLS)"
                  << HookRssCapText(cap) << std::endl;
    }
    for (const GCell& c : gcells) ArmScale(g, c, lines);
    std::cout << "  [(G) " << gcells.size() << " cells done at " << Seconds(t0) << " s]" << std::endl;
    std::cout << "  WP13_NONTRIVIAL K=" << NT().K << " A=" << NT().A << " W=" << NT().W << " G=" << NT().G
              << " (columns whose presentation side is not in Rel on its own)" << std::endl;
    g.check(NT().K > 0 && NT().A > 0 && NT().W > 0, "non-vacuity: (K), (A), (W) each compare at least one class not in Rel");
    if (!gcells.empty()) g.check(NT().G > 0 || std::none_of(gcells.begin(), gcells.end(), [](const GCell& c) {
                                     return c.N == c.d * c.M && hecke::IsPrimeU64(c.d); }),
                                 "non-vacuity: (G)'s trace comparisons include a class not in Rel");

    for (const std::string& l : lines)
        if (l.rfind("DEGEN_CELL", 0) != 0 && l.rfind("LEVEL_N_DEGEN (W) oldspace", 0) != 0) std::cout << "  " << l << "\n";
    std::cout << "  WP-13 arms: " << g.checks << " checks, " << g.fails << " failing, " << Seconds(t0) << " s" << std::endl;
    checks += g.checks;
    return g.fails;
}
