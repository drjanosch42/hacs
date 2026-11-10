// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 15, WP-8) HeckeLevelNSparseValidation -- see the header.
// [C] sec:levelN (rmk:levelcost (4), prop:levelorbit, prop:heilbronn), sec:levelshapiro
// (prop:levelcorrespondence (4)(b), (c)).
#include "HeckeLevelNSparseValidation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CoefficientRing.h"
#include "HeckeCosetTables.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNDegenValidation.h"   // slot-16 WP-13 arms
#include "HeckeLevelNHeilbronn.h"
#include "HeckeLevelNSparse.h"
#include "HeckeLevelNWideValidation.h"   // slot-17 WP-5 arms
#include "HeckeLevelNOrderValidation.h"  // slot-18 WP-5 arms
#include "HeckeOperatorFamily.h"  // hecke::tau_construction_mutex()
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Util/Timing.h"
#include "ValidationSupport.h"

namespace {

using oc_util::Seconds;

using hecke::Mat22;
using namespace hecke::level;

constexpr uint32_t kP = 1000003;   // the characteristic-0 surrogate field F_P (as HeckeLevelNValidation)

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

std::string CellText(uint64_t N, uint32_t n, uint32_t p) {
    return "(N=" + std::to_string(N) + ", n=" + std::to_string(n) + ") over F_" + std::to_string(p);
}
std::vector<uint64_t> Divisors(uint64_t N) {
    std::vector<uint64_t> d;
    for (uint64_t t = 1; t <= N; ++t)
        if (N % t == 0) d.push_back(t);
    return d;
}
// Arm (S)'s operator primes (WP-17, WP-8 audit LOW 4: beyond {2, 3, 5, 7}); 11 and 13 on the cells with
// D <= 300 (40 of the 47), which keeps the added runtime within the audit's budget.
std::vector<uint32_t> PrimesUpTo13(uint64_t D) {
    if (D <= 300) return {2u, 3u, 5u, 7u, 11u, 13u};
    return {2u, 3u, 5u, 7u};
}
std::vector<uint32_t> PrimesUpTo7Except(uint64_t N, bool keep_bad) {
    std::vector<uint32_t> q;
    for (uint32_t x : {2u, 3u, 5u, 7u})
        if (keep_bad || N % x != 0) q.push_back(x);
    return q;
}

// ---------------------------------------------------------------------------
// The model side (the same construction as HeckeLevelNValidation's / WP-7's validation;
// those helpers are TU-local).
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
// The level-1 datum of the primitive double coset Gamma diag(1, p^nu) Gamma.
HeckeDatum MakeDatum(uint64_t p, uint32_t nu, uint32_t coef, uint32_t n) {
    Model& M = TheModel();
    return hecke::vsupport::MakeHeckeDatum(*M.K, *M.tess, p, nu, coef, n);
}
struct LevelModel {
    std::unique_ptr<Gamma0LevelN> lvl;
    std::unique_ptr<LocalSystemSimplicialComplex> L;
    std::unique_ptr<EquivariantCohomologyOnK> H;
    std::unique_ptr<HFrame> f;
};
std::unique_ptr<LevelModel> BuildLevelModel(uint64_t N, uint32_t n, const engine::ModCtx<uint32_t>& ctx) {
    Model& M = TheModel();
    auto c = std::make_unique<LevelModel>();
    c->lvl.reset(new Gamma0LevelN(N));
    const uint32_t D = c->lvl->m() * (n + 1);
    c->L.reset(new LocalSystemSimplicialComplex(*M.K, InducedRelationsLevel(*M.K, *c->lvl, n, ctx), D, ctx.p));
    c->H.reset(new EquivariantCohomologyOnK(c->L.get()));
    c->H->compute_cohomology();
    c->f.reset(new HFrame(c->H.get(), 1));
    return c;
}
HMat OnH1(const LevelModel& src, const LevelModel& dst, const CorrespondenceSpec& s, uint32_t n,
          const engine::ModCtx<uint32_t>& ctx) {
    return InducedOnH(*src.f, *dst.f, BuildLevelCorrespondence(ctx, s, *TheModel().K, n, 1), s.label);
}

// ---------------------------------------------------------------------------
// (S) sparse = dense entrywise.
// ---------------------------------------------------------------------------
struct SCell { uint64_t N; uint32_t n; uint32_t p; };

void ArmSparseDense(Checker& g, const SCell& c, std::vector<std::string>& lines) {
    const engine::ModCtx<uint32_t> ctx(c.p);
    const std::string tag = "(S) " + CellText(c.N, c.n, c.p);
    const Gamma0LevelN lvl(c.N);
    auto t0 = std::chrono::steady_clock::now();
    const CokernelPresentation dense = CokernelPresentation::Build(lvl, c.n, ctx);
    const double t_dense = Seconds(t0);
    t0 = std::chrono::steady_clock::now();
    const SparseCokernel sp = SparseCokernel::Build(lvl, c.n, ctx);
    const double t_sparse = Seconds(t0);
    g.check(sp.D() == dense.D() && sp.h1() == dense.h1() && sp.dim_rel() == dense.dim_rel(),
            tag + ": D, dim Rel, h1 equal (" + std::to_string(sp.h1()) + " / " + std::to_string(dense.h1()) + ")");
    g.check(sp.free_columns() == dense.free_columns(), tag + ": the free columns (the quotient basis) equal");
    // every RREF row entrywise
    bool rows_eq = sp.dim_rel() == dense.dim_rel();
    for (uint32_t i = 0; rows_eq && i < dense.dim_rel(); ++i) {
        const std::vector<uint32_t>& R = dense.rel_basis()[i];
        uint32_t piv = 0;
        while (piv < R.size() && R[piv] == 0) ++piv;
        if (piv >= R.size() || piv != sp.pivots()[i] || R[piv] != 1) { rows_eq = false; break; }
        std::vector<uint32_t> S(sp.D(), 0);
        S[piv] = 1;
        const SparseVec& t = sp.tail_of(piv);
        for (size_t k = 0; k < t.col.size(); ++k) S[t.col[k]] = t.val[k];
        rows_eq = S == R;
    }
    g.check(rows_eq, tag + ": every RREF row of Rel_N entrywise (" + std::to_string(dense.dim_rel()) + " rows)");
    // h0 / h1 against the dense ranks of prop:levelorbit (2)
    const FixedSpace fS = OrbitWiseFixedSpace(lvl, Mat22::S(), 2, c.n, ctx);
    const FixedSpace fU = OrbitWiseFixedSpace(lvl, Mat22::ST(), 3, c.n, ctx);
    const CokernelDims cd = CokernelFromFixedSpaces(fS, fU, c.p);
    g.check(sp.stats().dim_S == cd.dim_S && sp.stats().dim_ST == cd.dim_ST && sp.h0() == cd.h0 && sp.h1() == cd.h1,
            tag + ": dim M^S, dim M^ST, h0, h1 = CokernelFromFixedSpaces");
    // the operators, entrywise
    uint32_t ops = 0;
    for (uint32_t q : PrimesUpTo13(static_cast<uint64_t>(lvl.m()) * (c.n + 1))) {
        if (q == c.p) continue;
        const HeilbronnOperator H(lvl, q, c.n, ctx);
        const uint32_t esc_d = RelEscapes(dense, H), esc_s = RelEscapesOnGenerators(sp, H);
        g.check(esc_d == 0 && esc_s == 0, tag + ": " + H.label() + "(Rel) in Rel on both routes");
        if (esc_d || esc_s) continue;
        g.check(HEqual(OnCokernel(dense, H), OnCokernelSparse(sp, H)), tag + ": [" + H.label() + "] sparse = dense entrywise");
        ++ops;
    }
    // the unified entry: dense, sparse and the cost model's choice give the same matrices entrywise.
    {
        std::vector<uint32_t> qs;
        for (uint32_t q : PrimesUpTo13(static_cast<uint64_t>(lvl.m()) * (c.n + 1)))
            if (q != c.p) qs.push_back(q);
        CokernelRoute used = CokernelRoute::kAuto;
        const auto Md = CokernelHeckeMatrices(lvl, c.n, ctx, qs, CokernelRoute::kDense);
        const auto Ms = CokernelHeckeMatrices(lvl, c.n, ctx, qs, CokernelRoute::kSparse);
        const auto Ma = CokernelHeckeMatrices(lvl, c.n, ctx, qs, CokernelRoute::kAuto, &used);
        bool same = Md.size() == Ms.size() && Md.size() == Ma.size();
        for (const auto& kv : Md) same = same && HEqual(kv.second, Ms.at(kv.first)) && HEqual(kv.second, Ma.at(kv.first));
        g.check(same && used == ChooseCokernelRoute(c.N, c.n),
                tag + ": CokernelHeckeMatrices dense = sparse = auto (" + (used == CokernelRoute::kSparse ? "sparse" : "dense") + ") entrywise");
    }
    std::ostringstream os;
    os << "LEVEL_N_SPARSE (S) N=" << c.N << " n=" << c.n << " p=" << c.p << " D=" << sp.D() << " h1=" << sp.h1()
       << " h0=" << sp.h0() << " nnz=" << sp.stats().final_nnz << " ops=" << ops << " dense_s=" << t_dense
       << " sparse_s=" << t_sparse;
    lines.push_back(os.str());
}

// ---------------------------------------------------------------------------
// (F) level change on the presentation.
// ---------------------------------------------------------------------------
void ArmLevelChange(Checker& g, uint64_t M, uint64_t N, uint32_t n, std::vector<std::string>& lines) {
    const engine::ModCtx<uint32_t> ctx(kP);
    const std::string tag = "(F) M=" + std::to_string(M) + " -> N=" + std::to_string(N) + " n=" + std::to_string(n);
    const Gamma0LevelN big(M), small(N);
    const LevelChangeMap res(big, small, LevelChangeKind::kRestriction, n, ctx);
    const LevelChangeMap tr(big, small, LevelChangeKind::kTrace, n, ctx);
    // Gamma-equivariance entrywise, gamma in {S, ST, T}
    bool equi = true;
    for (const Mat22& gam : {Mat22::S(), Mat22::ST(), Mat22::T()}) {
        const MonomialAction aM = MonomialOf(big, gam, n, ctx), aN = MonomialOf(small, gam, n, ctx);
        for (uint32_t c = 0; equi && c < res.src_dim(); ++c) {
            std::vector<uint32_t> e(res.src_dim(), 0);
            e[c] = 1;
            equi = res.apply(aM.apply(ctx, e)) == aN.apply(ctx, res.apply(e));
        }
        for (uint32_t c = 0; equi && c < tr.src_dim(); ++c) {
            std::vector<uint32_t> e(tr.src_dim(), 0);
            e[c] = 1;
            equi = tr.apply(aN.apply(ctx, e)) == aM.apply(ctx, tr.apply(e));
        }
    }
    g.check(equi, tag + ": alpha^0 and omega are Gamma-module maps (entrywise on S, ST, T)");
    const SparseCokernel QM = SparseCokernel::Build(big, n, ctx), QN = SparseCokernel::Build(small, n, ctx);
    const HMat A = OnCokernelLevelChange(QM, QN, res);   // throws unless Rel is preserved
    const HMat W = OnCokernelLevelChange(QN, QM, tr);
    g.check(true, tag + ": alpha^0 and omega preserve Rel (every generator)");
    const uint64_t idx = small.m() / big.m();
    g.check(HEqual(HMul(W, A), HScale(HIdentity(HMul(W, A)), idx)),
            tag + ": omega o alpha^0 = [Gamma_0(M):Gamma_0(N)] = " + std::to_string(idx) + " on H^1 (entrywise)");
    // omega o H_{N,q} o alpha^0 = idx * H_{M,q} for q !| N (T_q commutes with res).
    for (uint32_t q : PrimesUpTo7Except(N, false)) {
        const HMat HN = OnCokernelSparse(QN, HeilbronnOperator(small, q, n, ctx));
        const HMat HM = OnCokernelSparse(QM, HeilbronnOperator(big, q, n, ctx));
        g.check(HEqual(HMul(W, HMul(HN, A)), HScale(HM, idx)), tag + ": omega T_" + std::to_string(q) + "^(N) alpha^0 = " +
                                                                 std::to_string(idx) + " T_" + std::to_string(q) + "^(M) entrywise");
        // H_{N,q} o alpha^0 = alpha^0 o H_{M,q}
        g.check(HEqual(HMul(HN, A), HMul(A, HM)), tag + ": T_" + std::to_string(q) + " alpha^0 = alpha^0 T_" + std::to_string(q));
    }
    std::ostringstream os;
    os << "LEVEL_N_SPARSE (F) M=" << M << " N=" << N << " n=" << n << " h1(M)=" << QM.h1() << " h1(N)=" << QN.h1()
       << " rank(alpha0)=" << FpRank(std::vector<uint32_t>(A.a.begin(), A.a.end()), A.rows, A.cols, kP)
       << " rank(omega)=" << FpRank(std::vector<uint32_t>(W.a.begin(), W.a.end()), W.rows, W.cols, kP);
    lines.push_back(os.str());
}

// ---------------------------------------------------------------------------
// (D) degeneracy maps on the model, old / new.
// ---------------------------------------------------------------------------
uint64_t Sigma0(uint64_t n) { return Divisors(n).size(); }
// primitive Dirichlet characters mod f
uint64_t PhiPrim(uint64_t f) {
    uint64_t r = 1;
    for (const PrimePowerFactor& pf : FactorLevel(f)) {
        if (pf.nu == 1) r *= pf.q - 2;
        else r *= pf.Q - 2 * (pf.Q / pf.q) + pf.Q / (pf.q * pf.q);
    }
    return r;
}
int64_t CuspNewDim(uint64_t L, uint32_t k, std::map<uint64_t, int64_t>& memo) {
    auto it = memo.find(L);
    if (it != memo.end()) return it->second;
    int64_t s = DimCuspForms(L, k);
    for (uint64_t Lp : Divisors(L))
        if (Lp < L) s -= static_cast<int64_t>(Sigma0(L / Lp)) * CuspNewDim(Lp, k, memo);
    memo[L] = s;
    return s;
}
// dim Eis_k(Gamma_0(N)) and its old part (see WP-8_report.md section 3: the E^{chi, chi^-1}(t z)
// basis; at k = 2 the chi = 1 family is { sum c_t E_2(t z) : sum c_t / t = 0 }).
void EisDims(uint64_t N, uint32_t k, int64_t& eis, int64_t& eis_old) {
    eis = 0;
    eis_old = 0;
    for (uint64_t f = 1; f * f <= N; ++f) {
        if (N % (f * f)) continue;
        const int64_t term = static_cast<int64_t>(PhiPrim(f) * Sigma0(N / (f * f)));
        eis += term;
        if (f * f < N) eis_old += term;
    }
    if (k == 2) {
        eis -= 1;                                                    // the chi = 1 family: sigma_0(N) - 1
        const bool prime = N > 1 && hecke::IsPrimeU64(N);
        eis_old -= (N > 1) ? 1 : 0;                                  // f = 1 counted with sigma_0(N): now sigma_0(N) - 1 ...
        if (prime) eis_old -= static_cast<int64_t>(Sigma0(N)) - 1;   // ... and none of it old at N prime
    }
}

struct DegCell { uint64_t N; uint32_t n; };

void ArmDegeneracy(Checker& g, const DegCell& c, std::vector<std::string>& lines) {
    const engine::ModCtx<uint32_t> ctx(kP);
    const uint32_t k = c.n + 2;
    const std::string tag = "(D) N=" + std::to_string(c.N) + " k=" + std::to_string(k);
    std::map<uint64_t, std::unique_ptr<LevelModel>> lm;
    for (uint64_t M : Divisors(c.N)) lm[M] = BuildLevelModel(M, c.n, ctx);
    std::map<std::pair<uint64_t, uint32_t>, HeckeDatum> datum;
    auto dat = [&](uint64_t p, uint32_t nu) -> HeckeDatum& {
        auto key = std::make_pair(p, nu);
        auto it = datum.find(key);
        if (it == datum.end()) it = datum.emplace(key, MakeDatum(p, nu, kP, c.n)).first;
        return it->second;
    };
    // alpha_{p^nu}: level L -> level L' (p^nu L | L')
    auto alpha_pp = [&](uint64_t L, uint64_t Lp, uint64_t p, uint32_t nu) -> HMat {
        HeckeDatum& D = dat(p, nu);
        return OnH1(*lm.at(L), *lm.at(Lp), DegeneracySpec(*lm.at(L)->lvl, *lm.at(Lp)->lvl, *D.cosets, *D.tau), c.n, ctx);
    };
    // alpha_d, M -> N, through the chain of prime-power factors in the given order ((4)(c)).
    auto alpha = [&](uint64_t M, uint64_t N, const std::vector<std::pair<uint64_t, uint32_t>>& fac) -> HMat {
        if (fac.empty()) return OnH1(*lm.at(M), *lm.at(N), RestrictionSpec(*lm.at(M)->lvl, *lm.at(N)->lvl), c.n, ctx);
        HMat acc;
        uint64_t L = M;
        for (size_t i = 0; i < fac.size(); ++i) {
            uint64_t q = 1;
            for (uint32_t e = 0; e < fac[i].second; ++e) q *= fac[i].first;
            const uint64_t Lp = (i + 1 == fac.size()) ? N : L * q;
            const HMat step = alpha_pp(L, Lp, fac[i].first, fac[i].second);
            acc = (i == 0) ? step : HMul(step, acc);
            L = Lp;
        }
        return acc;
    };
    const LevelModel& top = *lm.at(c.N);
    const uint32_t hN = top.f->t();
    // Eichler--Shimura over F_P: dim H^1 = 2 dim S_k + dim Eis_k.
    int64_t eis = 0, eis_old = 0;
    EisDims(c.N, k, eis, eis_old);
    const int64_t sN = DimCuspForms(c.N, k);
    g.check(static_cast<int64_t>(hN) == 2 * sN + eis, tag + ": dim H^1 = 2 dim S_k + dim Eis_k = " + std::to_string(2 * sN + eis) +
                                                          " (model: " + std::to_string(hN) + ")");
    // the old space: the images of every alpha_d, M | N, M < N, d | N / M
    std::vector<uint32_t> stack;   // hN x total
    uint32_t total = 0;
    std::vector<std::vector<uint32_t>> cols;
    uint32_t maps = 0;
    for (uint64_t M : Divisors(c.N)) {
        if (M == c.N) continue;
        for (uint64_t d : Divisors(c.N / M)) {
            const std::vector<std::pair<uint64_t, uint32_t>> fac = DegeneracyFactors(d);
            const HMat A = alpha(M, c.N, fac);
            ++maps;
            for (uint32_t j = 0; j < A.cols; ++j) {
                std::vector<uint32_t> col(hN);
                for (uint32_t i = 0; i < hN; ++i) col[i] = static_cast<uint32_t>(A.at(i, j));
                cols.push_back(std::move(col));
            }
            // composite d: the other factor order gives the same map entrywise
            if (fac.size() >= 2) {
                std::vector<std::pair<uint64_t, uint32_t>> rev(fac.rbegin(), fac.rend());
                g.check(HEqual(A, alpha(M, c.N, rev)), tag + ": alpha_" + std::to_string(d) + " (M=" + std::to_string(M) +
                                                            ") independent of the factor order (prop:levelcorrespondence (4)(c))");
            }
            // omega o alpha_d = [Gamma_0(dM):Gamma_0(N)] [Gamma_0(M) g_d Gamma_0(M)] for d a prime power
            if (fac.size() == 1) {
                const uint64_t p = fac[0].first;
                const uint32_t nu = fac[0].second;
                const HMat W = OnH1(top, *lm.at(M), TraceSpec(*top.lvl, *lm.at(M)->lvl), c.n, ctx);
                HeckeDatum& D = dat(p, nu);
                const Gamma0LevelN lvl_dM(d * M);
                CorrespondenceSpec s{lm.at(M)->lvl.get(), lm.at(M)->lvl.get(), D.cosets.get(), D.tau.get(),
                                     Gamma0SubTransversalN(*lm.at(M)->lvl, lvl_dM), "[Gamma_0(M) g_d Gamma_0(M)]"};
                VerifyTransversal(s, lvl_dM.m() / lm.at(M)->lvl->m());
                const HMat T = OnH1(*lm.at(M), *lm.at(M), s, c.n, ctx);
                const uint64_t idx = IndexGamma0(c.N) / IndexGamma0(d * M);
                g.check(HEqual(HMul(W, A), HScale(T, idx)),
                        tag + ": omega o alpha_" + std::to_string(d) + " = " + std::to_string(idx) + " [Gamma_0(" +
                            std::to_string(M) + ") diag(1," + std::to_string(d) + ") Gamma_0(" + std::to_string(M) + ")] (entrywise)");
            }
        }
    }
    total = static_cast<uint32_t>(cols.size());
    stack.assign(static_cast<size_t>(hN) * total, 0);
    for (uint32_t j = 0; j < total; ++j)
        for (uint32_t i = 0; i < hN; ++i) stack[static_cast<size_t>(i) * total + j] = cols[j][i];
    const uint32_t old_rank = total ? FpRank(stack, hN, total, kP) : 0;
    std::map<uint64_t, int64_t> memo;
    int64_t cusp_old = 0;
    for (uint64_t M : Divisors(c.N))
        if (M < c.N) cusp_old += static_cast<int64_t>(Sigma0(c.N / M)) * CuspNewDim(M, k, memo);
    const int64_t expect = 2 * cusp_old + eis_old;
    g.check(static_cast<int64_t>(old_rank) == expect,
            tag + ": rank of the old space (all alpha_d images) = " + std::to_string(old_rank) + ", the formula 2 * " +
                std::to_string(cusp_old) + " + " + std::to_string(eis_old) + " = " + std::to_string(expect));
    std::ostringstream os;
    os << "LEVEL_N_SPARSE (D) N=" << c.N << " k=" << k << " h1=" << hN << " maps=" << maps << " old=" << old_rank
       << " new=" << (hN - old_rank) << " cusp_new=" << CuspNewDim(c.N, k, memo) << " expect_old=" << expect;
    lines.push_back(os.str());
}

// the model-side cross check of (F): charpoly(omega T_q^{(N)} alpha^0) on the model = on the presentation.
void ArmLevelChangeCrossRoute(Checker& g, uint64_t M, uint64_t N, uint32_t n) {
    const engine::ModCtx<uint32_t> ctx(kP);
    const std::string tag = "(F) cross-route M=" + std::to_string(M) + " -> N=" + std::to_string(N) + " n=" + std::to_string(n);
    auto mM = BuildLevelModel(M, n, ctx), mN = BuildLevelModel(N, n, ctx);
    const HMat Am = OnH1(*mM, *mN, RestrictionSpec(*mM->lvl, *mN->lvl), n, ctx);
    const HMat Wm = OnH1(*mN, *mM, TraceSpec(*mN->lvl, *mM->lvl), n, ctx);
    const SparseCokernel QM = SparseCokernel::Build(*mM->lvl, n, ctx), QN = SparseCokernel::Build(*mN->lvl, n, ctx);
    const HMat Ap = OnCokernelLevelChange(QM, QN, LevelChangeMap(*mM->lvl, *mN->lvl, LevelChangeKind::kRestriction, n, ctx));
    const HMat Wp = OnCokernelLevelChange(QN, QM, LevelChangeMap(*mM->lvl, *mN->lvl, LevelChangeKind::kTrace, n, ctx));
    g.check(mM->f->t() == QM.h1() && mN->f->t() == QN.h1(), tag + ": h1 equal on both routes at both levels");
    for (uint32_t q : {2u, 3u}) {
        const bool bad = N % q == 0;
        HeckeDatum D = MakeDatum(q, 1, kP, n);
        const Gamma0LevelN lNq(N * q);
        CorrespondenceSpec s{mN->lvl.get(), mN->lvl.get(), D.cosets.get(), D.tau.get(), Gamma0SubTransversalN(*mN->lvl, lNq),
                             std::string(bad ? "U_" : "T_") + std::to_string(q)};
        VerifyTransversal(s, bad ? q : q + 1);
        const HMat Tm = OnH1(*mN, *mN, s, n, ctx);
        const HMat Tp = OnCokernelSparse(QN, HeilbronnOperator(*mN->lvl, q, n, ctx));
        const HMat lhs = HMul(Wm, HMul(Tm, Am)), rhs = HMul(Wp, HMul(Tp, Ap));
        g.check(HCharpoly(lhs) == HCharpoly(rhs), tag + ": charpoly(omega " + s.label + "^(N) alpha^0), model = presentation");
        g.check(HCharpoly(HMul(Am, Wm)) == HCharpoly(HMul(Ap, Wp)), tag + ": charpoly(alpha^0 omega) on H^1(Gamma_0(N)), model = presentation");
        // WP-17 (WP-8 audit LOW 1): beyond the charpolys.  The two routes carry different bases of H^1, related by
        // one unknown isomorphism P at each level, so the model pair (T, alpha^0 omega) on H^1(Gamma_0(N)) is the
        // presentation pair conjugated by the SAME P_N.  Joint-similarity invariants the charpoly alone misses:
        // the ranks (the nilpotent part), the charpoly of T itself, and the charpolys of a sum and a product of
        // the pair (which a separate conjugation of each operator would not preserve).
        const HMat Em = HMul(Am, Wm), Ep = HMul(Ap, Wp);
        g.check(HCharpoly(Tm) == HCharpoly(Tp), tag + ": charpoly(" + s.label + "^(N)), model = presentation");
        g.check(HRankModEll(lhs) == HRankModEll(rhs) && HRankModEll(Em) == HRankModEll(Ep) &&
                    HRankModEll(Tm) == HRankModEll(Tp),
                tag + ": rank of omega " + s.label + " alpha^0, of alpha^0 omega and of " + s.label + ", model = presentation");
        g.check(HCharpoly(HAdd(Tm, HScale(Em, 2))) == HCharpoly(HAdd(Tp, HScale(Ep, 2))) &&
                    HCharpoly(HMul(Tm, Em)) == HCharpoly(HMul(Tp, Ep)),
                tag + ": charpoly(" + s.label + " + 2 alpha^0 omega) and charpoly(" + s.label +
                    " alpha^0 omega), model = presentation (joint invariants)");
    }
    g.check(HRankModEll(Am) == HRankModEll(Ap) && HRankModEll(Wm) == HRankModEll(Wp),
            tag + ": rank alpha^0 = " + std::to_string(HRankModEll(Ap)) + ", rank omega = " + std::to_string(HRankModEll(Wp)) +
                ", model = presentation");
}

// ---------------------------------------------------------------------------
// (R) (R3) on the sparse route, (C) the cost model / ladder.
// ---------------------------------------------------------------------------
struct R3Cell { uint64_t N, p; uint32_t dimH1, dimW, loewy; };

void ArmR3(Checker& g, const R3Cell& r, std::vector<std::string>& lines) {
    oc_util::ResetPeakRss();
    const auto t0 = std::chrono::steady_clock::now();
    const R3Reading R = ReadEisensteinR3(r.N, r.p);
    const double wall = Seconds(t0);
    const double rss = oc_util::PeakRssMiB();
    const std::string tag = "(R) (N=" + std::to_string(r.N) + ", p=" + std::to_string(r.p) + ")";
    g.check(R.dimH1 == r.dimH1 && R.dimW == r.dimW && R.loewy == r.loewy,
            tag + ": sparse route dimH1 / dimW / loewy = " + std::to_string(R.dimH1) + " / " + std::to_string(R.dimW) + " / " +
                std::to_string(R.loewy) + " (frozen: " + std::to_string(r.dimH1) + " / " + std::to_string(r.dimW) + " / " +
                std::to_string(r.loewy) + ")");
    g.check(R.dimW % 2 == 1 && R.loewy == (R.dimW + 1) / 2, tag + ": (LN6) Loewy length = (dim W + 1) / 2");
    std::ostringstream os, ls;
    for (size_t i = 0; i < R.layers.size(); ++i) ls << (i ? "," : "") << R.layers[i];
    os << "LEVEL_N_SPARSE (R) N=" << r.N << " p=" << r.p << " dimH1=" << R.dimH1 << " dimW=" << R.dimW << " loewy=" << R.loewy
       << " layers=" << ls.str() << " nnz=" << R.final_nnz << " presentation_s=" << R.presentation_s
       << " operators_s=" << R.operators_s << " eisenstein_s=" << R.eisenstein_s << " wall_s=" << wall
       << " peak_rss_mib=" << rss;
    lines.push_back(os.str());
    std::cout << "  " << os.str() << std::endl;
}

// The ladder cell: presentation + T_q (q in {2, 3, 5, 7} \ {N}) on H^1, the Euler characteristic,
// the cost model against the measurement.
void ArmLadder(Checker& g, uint64_t N, uint32_t n, uint32_t p, bool ops, std::vector<std::string>& lines) {
    const engine::ModCtx<uint32_t> ctx(p);
    const std::string tag = "(C) " + CellText(N, n, p);
    oc_util::ResetPeakRss();
    ResetLevelNWideProducts();   // (slot-17 WP-5) the __int128 escalations of this cell's coset products
    const auto t0 = std::chrono::steady_clock::now();
    const Gamma0LevelN lvl(N);
    const double t_table = Seconds(t0);
    const SparseCokernel Q = SparseCokernel::Build(lvl, n, ctx, ops);
    const double t_pres = Seconds(t0) - t_table;
    // prop:levelorbit (3): h1 - h0 = D - dim M^S - dim M^ST, with the orbit counts nu_2, nu_3 and d_S, d_ST.
    const uint32_t d = n + 1;
    auto fixed_dim = [&](const Mat22& gam, uint32_t order) {
        RhoCache rc(ctx, n);
        std::vector<Block> st;
        std::vector<uint64_t> e(static_cast<size_t>(d) * d, 0);
        Mat22 x = Mat22::I();
        for (uint32_t i = 0; i < order; ++i) {
            const std::vector<uint32_t>& R = st[rc.block_of(x, st)].dense;
            for (size_t t = 0; t < e.size(); ++t) e[t] = (e[t] + R[t]) % p;
            x = x * gam;
        }
        return FpRank(std::vector<uint32_t>(e.begin(), e.end()), d, d, p);
    };
    const uint64_t m = lvl.m();
    const int64_t nu2 = Nu2(N), nu3 = Nu3(N);
    const int64_t dimS = (static_cast<int64_t>(m) - nu2) / 2 * d + nu2 * fixed_dim(Mat22::S(), 2);
    const int64_t dimU = (static_cast<int64_t>(m) - nu3) / 3 * d + nu3 * fixed_dim(Mat22::ST(), 3);
    g.check(Q.stats().dim_S == dimS && Q.stats().dim_ST == dimU,
            tag + ": dim M^S, dim M^ST = the orbit counts of prop:levelorbit (3)");
    g.check(static_cast<int64_t>(Q.h1()) - Q.h0() == static_cast<int64_t>(Q.D()) - dimS - dimU,
            tag + ": h1 - h0 = D - dim M^S - dim M^ST (prop:levelorbit (3))");
    double t_ops = 0;
    uint32_t nops = 0;
    if (ops) {
        const auto t1 = std::chrono::steady_clock::now();
        // Full matrices on H^1 up to h1 <= 2500 (h1^2 words each); beyond, the operators are applied sparsely
        // and the commutation is checked on 64 free basis vectors modulo Rel (O(D) memory).
        const bool full = Q.h1() <= 2500;
        std::vector<HMat> Ms;
        std::vector<std::unique_ptr<HeilbronnOperator>> Hs;
        std::vector<std::unique_ptr<HeilbronnSparseApply>> As;
        for (uint32_t q : PrimesUpTo7Except(N, true)) {
            if (q == p) continue;
            Hs.emplace_back(new HeilbronnOperator(lvl, q, n, ctx));
            const HeilbronnOperator& H = *Hs.back();
            g.check(RelEscapesOnGenerators(Q, H) == 0, tag + ": " + H.label() + "(Rel) in Rel (every generator)");
            if (full) Ms.push_back(OnCokernelSparse(Q, H));
            else As.emplace_back(new HeilbronnSparseApply(H));
            ++nops;
        }
        if (full) {
            for (size_t i = 0; i + 1 < Ms.size(); ++i)
                g.check(HEqual(HMul(Ms[i], Ms[i + 1]), HMul(Ms[i + 1], Ms[i])), tag + ": the Hecke family commutes (pair " + std::to_string(i) + ")");
        } else {
            for (size_t i = 0; i + 1 < As.size(); ++i) {
                bool comm = true;
                for (uint32_t j = 0; comm && j < std::min<uint32_t>(64, Q.h1()); ++j) {
                    SparseVec e;
                    e.col.push_back(Q.free_columns()[static_cast<size_t>(j) * Q.h1() / std::min<uint32_t>(64, Q.h1())]);
                    e.val.push_back(1);
                    comm = Q.coords(As[i]->apply(As[i + 1]->apply(e))) == Q.coords(As[i + 1]->apply(As[i]->apply(e)));
                }
                g.check(comm, tag + ": the Hecke family commutes mod Rel on 64 basis vectors (pair " + std::to_string(i) + ")");
            }
        }
        t_ops = Seconds(t1);
    }
    const double wall = Seconds(t0);
    const double rss = oc_util::PeakRssMiB();
    const CokernelCost cm = CokernelCostModel(N, n);
    const double fill_ratio = static_cast<double>(Q.stats().final_nnz) / std::max(1.0, cm.sparse_nnz);
    const double ops_ratio = static_cast<double>(Q.stats().merge_ops) / std::max(1.0, cm.sparse_ops - 3.0 * d * cm.gen_rows);
    std::ostringstream os;
    os << "SPARSE_CELL N=" << N << " n=" << n << " p=" << p << " m=" << m << " D=" << Q.D() << " h1=" << Q.h1() << " h0=" << Q.h0()
       << " gens=" << Q.stats().generators << " gen_nnz=" << Q.stats().generator_nnz << " final_nnz=" << Q.stats().final_nnz
       << " peak_nnz=" << Q.stats().peak_nnz << " merge_ops=" << Q.stats().merge_ops << " model_fill_ratio=" << fill_ratio
       << " model_ops_ratio=" << ops_ratio << " route=" << (cm.route == CokernelRoute::kSparse ? "sparse" : "dense")
       << " sparse_bytes_pred=" << cm.sparse_bytes << " dense_ops_pred=" << cm.dense_ops << " dense_bytes_pred=" << cm.dense_bytes << " table_s=" << t_table
       << " presentation_s=" << t_pres << " ops=" << nops << " ops_s=" << t_ops << " wall_s=" << wall << " peak_rss_mib=" << rss
       << " level_wide_products=" << LevelNWideProducts()
       // (slot-18 WP-5 (b)) the generator order, the occurrence-list entries, and the order-independent presentation digest
       << " gen_order=" << GeneratorOrderInForceName() << " occ_entries=" << Q.stats().occ_entries << " presentation_fnv=0x"
       << std::hex << PresentationDigest(Q) << std::dec;
    lines.push_back(os.str());
    std::cout << "  " << os.str() << std::endl;
}

} // namespace

uint32_t HeckeLevelNSparseValidation() {
    std::cout << "=== HeckeLevelNSparseValidation (slot-15 WP-8: the sparse cokernel at large N, level change and "
                 "degeneracy maps at composite N, the cost model; [C] sec:levelN rmk:levelcost (4), prop:levelorbit, "
                 "prop:heilbronn, prop:levelcorrespondence (4)(b),(c)) ===\n";
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    std::vector<std::string> lines;

    // ---- (S) sparse = dense entrywise: WP-7's (L) cells plus larger dense-reachable ones.
    const std::vector<SCell> scells{
        {4, 0, kP}, {4, 2, kP}, {6, 0, kP}, {6, 2, kP}, {6, 4, kP}, {8, 2, kP}, {9, 0, kP}, {9, 2, kP}, {10, 0, kP},
        {10, 2, kP}, {12, 0, kP}, {12, 2, kP}, {15, 0, kP}, {15, 2, kP}, {21, 0, kP}, {21, 2, kP}, {5, 2, kP},
        {11, 0, kP}, {35, 0, kP}, {30, 0, kP}, {12, 4, kP}, {1, 10, kP}, {1, 22, kP}, {13, 2, kP}, {4, 4, kP},
        {20, 2, kP}, {22, 2, kP}, {60, 0, kP}, {42, 2, kP},
        {10, 2, 5}, {21, 2, 7}, {15, 4, 5}, {14, 0, 7}, {15, 2, 7}, {12, 2, 5}, {8, 2, 7}, {13, 2, 7},
        {181, 0, 5}, {131, 0, 13},
        // larger, still dense-reachable
        {211, 0, 5}, {499, 0, kP}, {210, 0, kP}, {97, 4, kP}, {50, 6, kP}, {256, 0, kP}, {225, 2, 7}, {143, 2, 11}};
    for (const SCell& c : scells) ArmSparseDense(g, c, lines);
    // WP-17 (WP-8 audit LOW 3): p in {2, 3} is refused by name on every route, kAuto included.
    for (uint32_t p : {2u, 3u})
        for (CokernelRoute r : {CokernelRoute::kAuto, CokernelRoute::kDense, CokernelRoute::kSparse}) {
            bool refused = false;
            try {
                (void)CokernelHeckeMatrices(Gamma0LevelN(11), 0, engine::ModCtx<uint32_t>(p), {7u}, r);
            } catch (const std::logic_error& e) {
                refused = std::string(e.what()).find("p in {2, 3} refused") != std::string::npos;
            }
            g.check(refused, "(S) CokernelHeckeMatrices at p = " + std::to_string(p) + ", route " +
                                 std::to_string(static_cast<int>(r)) + " refused by name");
        }
    std::cout << "  [(S) " << scells.size() << " cells done at " << Seconds(t0) << " s]" << std::endl;

    // ---- (F) level change on the presentation.
    const std::vector<std::array<uint64_t, 3>> fcells{{1, 6, 0}, {2, 6, 0}, {3, 6, 2}, {5, 10, 0}, {2, 10, 2}, {4, 12, 0},
                                                      {6, 12, 2}, {3, 15, 0}, {5, 35, 0}, {7, 35, 2}, {11, 22, 0},
                                                      {1, 11, 10}, {4, 16, 0}, {9, 27, 0}, {6, 30, 0}};
    for (const auto& f : fcells) ArmLevelChange(g, f[0], f[1], static_cast<uint32_t>(f[2]), lines);
    ArmLevelChangeCrossRoute(g, 2, 6, 0);
    ArmLevelChangeCrossRoute(g, 5, 10, 2);
    ArmLevelChangeCrossRoute(g, 3, 15, 0);
    std::cout << "  [(F) done at " << Seconds(t0) << " s]" << std::endl;

    // ---- (D) degeneracy maps on the model, old / new.
    const std::vector<DegCell> dcells{{11, 0}, {22, 0}, {33, 0}, {35, 0}, {30, 0}, {26, 0}, {12, 0}, {16, 0}, {18, 0},
                                      {27, 0}, {25, 0}, {36, 0}, {6, 2}, {10, 2}, {12, 2}, {9, 2}, {15, 2}, {8, 2},
                                      {16, 2}, {20, 2}, {18, 4}, {25, 2}};
    for (const DegCell& c : dcells) ArmDegeneracy(g, c, lines);
    std::cout << "  [(D) done at " << Seconds(t0) << " s]" << std::endl;

    // ---- (R) (R3) on the sparse route: the frozen scan (HeckeLevelNValidation (R3), slot-14 WP-4) and the named cells.
    const std::vector<R3Cell> r3{
        {11, 5, 3, 3, 2}, {23, 11, 5, 3, 2}, {29, 7, 5, 3, 2}, {31, 5, 5, 5, 3}, {41, 5, 7, 3, 2}, {43, 7, 7, 3, 2},
        {53, 13, 9, 3, 2}, {61, 5, 9, 3, 2}, {67, 11, 11, 3, 2}, {71, 5, 13, 3, 2}, {71, 7, 13, 3, 2}, {79, 13, 13, 3, 2},
        {89, 11, 15, 3, 2}, {101, 5, 17, 3, 2}, {113, 7, 19, 3, 2}, {127, 7, 21, 5, 3}, {131, 5, 23, 5, 3},
        {131, 13, 23, 3, 2}, {151, 5, 25, 3, 2}, {157, 13, 25, 3, 2}, {181, 5, 29, 7, 4}, {191, 5, 33, 3, 2},
        {197, 7, 33, 3, 2}, {199, 11, 33, 3, 2}, {211, 5, 35, 5, 3}, {211, 7, 35, 3, 2},
        {1321, 11, 219, 7, 4}, {1571, 5, 263, 7, 4}};
    for (const R3Cell& r : r3) ArmR3(g, r, lines);
    std::cout << "  [(R) done at " << Seconds(t0) << " s]" << std::endl;

    // ---- (C) the cost model and the ladder (default: modest cells; the env hook runs the heavy ladder).
    std::vector<std::array<uint64_t, 3>> ladder{{1571, 0, 5}, {2003, 0, kP}, {1000, 0, kP}, {997, 2, kP}, {210, 4, kP}};
    if (const char* env = std::getenv("HECKE_LEVEL_N_SPARSE_CELLS")) {
        ladder.clear();
        // slot-17 WP-8 (WP-5 audit MINOR-2): strict parsing + the rule-19 pre-run RSS refusal, as DEGEN / R3K. slot-17 WP-17
        // (WP-8 audit MAJOR-1): the factor is per cell, hook_rss/v1 (HeckeLevelNDegenValidation.h) -- the measured peak RSS /
        // sparse_bytes is ~10 at n = 0 and 1.9 at 60060:2, and > 50 at 99330:0 (W5-F1), so one constant (the former 6x, whose
        // "4-5x" rationale was wrong) could not refuse 99330:0:13 and admit 60060:2:17. HECKE_LEVEL_N_HOOK_RSS_CAP_GIB overrides
        // the 8 GiB cap for a deliberate run.
        const HookRssCap cap = HookRssCapFromEnv();
        size_t idx = 0;
        for (const auto& v : HookParseCells(env, "HECKE_LEVEL_N_SPARSE_CELLS", 3)) {
            ++idx;
            const std::string where = "HECKE_LEVEL_N_SPARSE_CELLS: cell " + std::to_string(idx) + ": ";
            ENGINE_CHECK_MSG(v[0] >= 1 && v[0] < (1ull << 20), (where + "N in [1, 2^20)").c_str());
            ENGINE_CHECK_MSG(v[1] % 2 == 0, (where + "n even").c_str());
            ENGINE_CHECK_MSG(v[2] >= 5 && v[2] < (1ull << 31) && hecke::IsPrimeU64(v[2]), (where + "p >= 5 prime").c_str());
            {
                const uint32_t nn = static_cast<uint32_t>(v[1]);
                const double rss = kHookRssBaselineMiB * 1024.0 * 1024.0 + HookPredictedRssBytes(HookKind::kSparse, v[0], nn);
                HookRefuseOverCap("HECKE_LEVEL_N_SPARSE_CELLS", idx,
                                  "N=" + std::to_string(v[0]) + " n=" + std::to_string(nn) + " factor=" +
                                      std::to_string(static_cast<int>(HookRssFactor(HookKind::kSparse, v[0], nn))),
                                  rss, cap);
            }
            ladder.push_back({v[0], v[1], v[2]});
        }
        std::cout << "  EFFECTIVE HeckeLevelNSparseValidation arm=CELLS cells=" << ladder.size() << " (HECKE_LEVEL_N_SPARSE_CELLS)"
                  << HookRssCapText(cap) << " gen_order=" << GeneratorOrderInForceName() << std::endl;
    }
    for (const auto& c : ladder) ArmLadder(g, c[0], static_cast<uint32_t>(c[1]), static_cast<uint32_t>(c[2]), true, lines);
    // the route choice: sparse beyond the crossover, dense below it (value-identical either way, arm (S)).
    g.check(ChooseCokernelRoute(10007, 0) == CokernelRoute::kSparse && ChooseCokernelRoute(1571, 0) == CokernelRoute::kSparse &&
                ChooseCokernelRoute(11, 0) == CokernelRoute::kDense && ChooseCokernelRoute(181, 0) == CokernelRoute::kDense,
            "(C) the cost model chooses dense below the crossover (N = 11, 181) and sparse at N = 1571, 10007");
    std::cout << "  [(C) done at " << Seconds(t0) << " s]" << std::endl;

    // ---- slot-16 WP-13: (K) (LN7), (A) / (W) (LN8), (L9) (LN9) reported, (G) alpha_d at scale ([C] sec:levelNdegsec).
    g.fails += HeckeLevelNDegenArms(g.checks);
    // ---- slot-17 WP-5: (W5b) / (W5n), the width of the coset products (F-1).
    g.fails += HeckeLevelNWideArms(g.checks);
    // ---- slot-18 WP-5: (W18o) the generator order is value-free; (W18g) Gamma1LevelN::contains_wide, forced wide = int64.
    g.fails += HeckeLevelNOrderArms(g.checks);

    for (const std::string& l : lines)
        if (l.rfind("LEVEL_N_SPARSE (R)", 0) != 0 && l.rfind("SPARSE_CELL", 0) != 0) std::cout << "  " << l << "\n";
    std::cout << "HeckeLevelNSparseValidation: " << g.checks << " checks, " << g.fails << " failing, " << Seconds(t0) << " s"
              << std::endl;
    return g.fails ? 1 : 0;
}
