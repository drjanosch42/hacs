// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D0: the level-1 integral groups by the owner's descent, and the measured comparators.  See the
// header; [C] = docs/latex/gl2_contract.tex, cited by label.
#include "DescentAssembly.h"
#include "DescentParAsm.h"
#include "DescentTree.h"
#include "DescentLevelN.h"
#include "DescentSmithGpu.h"
#include "DescentSmith.h"
#include "../../Util/Rss.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <sys/resource.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>

#include "../../EquivariantCohomologyOnK.h"
#include "../../Hecke/CoefficientRing.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeLevelNSparse.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../SimplicialCohomology.h"
#include "../../../ModArithmetic/Binary/Ring2k.h"
#include "../../../ModArithmetic/Binary/Ring3k.h"
#include "../../../ModArithmetic/Binary/ParWorker.h"
#include "../../../ModArithmetic/Binary/SmithPar.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/FlatMatrix.h"
#include "../../../ModArithmetic/Kernels.h"

namespace integral_descent {

using engine::DenseBuf;
using U32Ctx = engine::ModCtx<uint32_t>;
using clk = std::chrono::steady_clock;

namespace {
double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

}  // namespace

bool IsPrime(uint32_t v) {
    if (v < 2) return false;
    for (uint64_t d = 2; d * d <= v; ++d)
        if (v % d == 0) return false;
    return true;
}

// VmRSS in bytes via Util/Rss.h (kB / 1024.0 * 2^20 is exact in double); 0 if /proc is unreadable.
uint64_t CurrentRssBytes() {
    const double mib = oc_util::CurrentRssMiB();
    return mib < 0.0 ? 0 : static_cast<uint64_t>(mib * 1048576.0);
}
struct RssSampler::Impl {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> peak{0};
    std::thread th;
};
RssSampler::RssSampler() : impl_(new Impl) {
    impl_->peak.store(CurrentRssBytes());
    impl_->th = std::thread([this] {
        while (!impl_->stop.load(std::memory_order_acquire)) {
            const uint64_t v = CurrentRssBytes();
            uint64_t cur = impl_->peak.load();
            while (v > cur && !impl_->peak.compare_exchange_weak(cur, v)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
}
uint64_t RssSampler::stop() {
    if (impl_->th.joinable()) {
        impl_->stop.store(true, std::memory_order_release);
        impl_->th.join();
        const uint64_t v = CurrentRssBytes();
        if (v > impl_->peak.load()) impl_->peak.store(v);
    }
    return impl_->peak.load();
}
RssSampler::~RssSampler() {
    stop();
    delete impl_;
}
double ProcessCpuSeconds() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return double(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) + 1e-6 * double(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
}

const integral_model::IsotropyModel& Descriptor5() {
    static const integral_model::IsotropyModel m(5);
    return m;
}
const integral_model::IsotropyModel& Descriptor3() {
    static const integral_model::IsotropyModel m(3);
    return m;
}
const LocalModel& Iso2() {
    static const LocalModel m(Descriptor5(), 2);
    return m;
}
const LocalModel& Iso3() {
    static const LocalModel m(Descriptor5(), 3);
    return m;
}

uint32_t LocalWidth(uint32_t ell) {
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "LocalWidth: ell in {2, 3}");
    return ell == 2 ? 30u : engine::ring3k::kMaxK;   // 2^30: the ModCtx<uint32_t> bound; 3^19 < 2^31
}

uint32_t DefaultGoodPrime(uint32_t n) {
    ENGINE_CHECK_MSG(n + 1 < 10007, "DefaultGoodPrime: n too large for the default good prime 10007");
    return 10007;
}

// (slot-24 PAR-L1) the good-prime route lever (DescentAssembly.h)
// (S24 CL-5) on EnvLever: "" = unset; an invalid value warns once and reads as the default (source=default).
namespace {
// (S26 LEVERS-26) The env arm HECKE_DESCENT_GOOD_SPARSE=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1: EnvLever.h
// EnvArmRetiredLever); the shipped C_7 route stays (odd n; ScopedDescentGoodSparse).
engine::lever::EnvArmRetiredLever g_good_sparse("HECKE_DESCENT_GOOD_SPARSE",
                                      engine::lever::Policy{true, engine::lever::Empty::Unset, engine::lever::Match::Exact01, engine::lever::Cache::Once, "WARNING: HECKE_DESCENT_GOOD_SPARSE=%s is neither 0 nor 1; the sparse good prime stays on (default)\n", "default", "env", "default"});
}  // namespace
const engine::lever::EnvLever& DescentGoodSparseLever() { return g_good_sparse; }
bool DescentGoodSparseEnabled() { return g_good_sparse.enabled(); }
std::string DescentGoodSparseSource() { return g_good_sparse.source(); }

// ---------------------------------------------------------------------------
// (1) the shipped pipeline over a field, (2) its K-mode at ell >= 5
// ---------------------------------------------------------------------------
std::vector<uint32_t> ShippedFieldDims(uint32_t n, uint32_t ell) {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n, ell)) rels.push_back(datum.rel);
    LocalSystemSimplicialComplex lssc(K, rels, n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    std::vector<uint32_t> dims = coh.get_H_eq_dimensions();
    ENGINE_CHECK_MSG(dims.size() <= 3, "ShippedFieldDims: the 7-vertex complex has three degrees");
    dims.resize(3, 0);
    return dims;
}

// (slot-24 PAR-GP) the dense good-prime lever (DescentAssembly.h)
// (S24 CL-5) on EnvLever: "" is invalid; (PAR-GP audit F6) an invalid value warns once, takes the default route and is reported
// as source=env-invalid(default kept).
namespace {
// (S26 LEVERS-26) The env arm HECKE_DESCENT_GOOD_DENSE=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1: EnvLever.h
// EnvArmRetiredLever); the sparse elimination stays via ScopedDescentGoodDense.
engine::lever::EnvArmRetiredLever g_good_dense("HECKE_DESCENT_GOOD_DENSE",
                                     engine::lever::Policy{true, engine::lever::Empty::Invalid, engine::lever::Match::Exact01, engine::lever::Cache::Once, "WARNING: HECKE_DESCENT_GOOD_DENSE=%s is neither 0 nor 1; the dense good prime stays on (default)\n", "default", "env", "env-invalid(default kept)"});
}  // namespace
const engine::lever::EnvLever& DescentGoodDenseLever() { return g_good_dense; }
bool DescentGoodDenseEnabled() { return g_good_dense.enabled(); }
std::string DescentGoodDenseSource() { return g_good_dense.source(); }

namespace {
std::atomic<uint64_t> g_good_dense_regions{0};
}  // namespace
uint64_t GoodPrimeDenseHookRegions() { return g_good_dense_regions.load(); }

const char* GoodPrimeRouteName(GoodPrimeRoute r) {
    return r == GoodPrimeRoute::Dense ? "dense" : r == GoodPrimeRoute::Sparse ? "sparse" : "shipped";
}
GoodPrimeRoute GoodPrimeRouteFor(uint32_t n) {
    if (n % 2 != 0 || !DescentGoodSparseEnabled()) return GoodPrimeRoute::Shipped;
    return DescentGoodDenseEnabled() ? GoodPrimeRoute::Dense : GoodPrimeRoute::Sparse;
}

GoodPrimeRanks GoodPrimeDenseRanks(uint32_t n, uint32_t good, uint32_t threads) {
    ENGINE_CHECK_MSG(n % 2 == 0, "GoodPrimeDenseRanks: n even (-I acts trivially; prop:levelorbit)");
    ENGINE_CHECK_MSG(good >= 5 && IsPrime(good), "GoodPrimeDenseRanks: a prime good >= 5 (6 a unit)");
    // (PAR-GP audit F3) the ranks are dim H^q(Gamma; Sym^n F_good) = the rational dims only for good > n + 1
    ENGINE_CHECK_MSG(good > n + 1, "GoodPrimeDenseRanks: a good prime > n + 1");
    const auto t0 = clk::now();
    GoodPrimeRanks out;
    out.threads = threads == 0 ? maps::DescentSmithDefaultThreads() : threads;
    if (engine::parworker::active()) out.threads = 1;   // serial on a PAR-ASM worker (no nested pool)
    const uint32_t T = out.threads, D = n + 1;
    const U32Ctx ctx(good);
    // rho_n(g) over F_good: the shipped symmetric-power matrix (as RhoCache::block_of builds the sparse route's blocks)
    auto rho = [&](const hecke::Mat22& g) {
        const auto k = g.reduced_mod(good);
        const FlatMatrixT<uint32_t> A(2, 2, good, std::vector<uint32_t>{k[0], k[1], k[2], k[3]});
        return ComputeSymmetricPowerMatrixFM(A, n);
    };
    // G = e^T, row c = the generator "column c of e" (the sparse route's generators up to the unit 1/|gamma|)
    auto idempotent_t = [&](const hecke::Mat22& g, uint32_t order) {
        DenseBuf<uint32_t> G(D, D);
        for (uint32_t i = 0; i < D; ++i) G.at(i, i) = 1;
        hecke::Mat22 gi = g;
        for (uint32_t i = 1; i < order; ++i) {
            const FlatMatrixT<uint32_t> R = rho(gi);
            for (uint32_t a = 0; a < D; ++a)
                for (uint32_t c = 0; c < D; ++c) G.at(c, a) = ctx.add(G.at(c, a), R.get_entry(a, c));
            gi = gi * g;
        }
        return G;
    };
    // the elimination's ParallelFor hook on the descent pool (none when serial, or when a hook is already in force)
    const engine::elim::ParallelFor hook = [T](uint32_t ntasks, const std::function<void(uint32_t)>& fn) {
        g_good_dense_regions.fetch_add(1, std::memory_order_relaxed);
        engine::smithpar::parallel_for_n(T, ntasks, fn);
    };
    std::unique_ptr<engine::elim::ScopedParallelHook> scoped;
    out.caller_hook = engine::elim::parallel_hook() != nullptr;   // (PAR-GP audit F4) rref then runs on the caller's hook
    if (T > 1 && !out.caller_hook) scoped = std::make_unique<engine::elim::ScopedParallelHook>(&hook);

    DenseBuf<uint32_t> GS = idempotent_t(hecke::Mat22::S(), 2);
    const engine::elim::RrefResult rS = engine::elim::rref(ctx, engine::view(GS));
    DenseBuf<uint32_t> GU = idempotent_t(hecke::Mat22::ST(), 3);
    const engine::elim::RrefResult rU = engine::elim::rref(ctx, engine::view(GU));
    out.dim_S = rS.rank;
    out.dim_ST = rU.rank;
    // rank [e_S | e_ST] = dim_S + rank of e_ST's RREF rows reduced against e_S's RREF rows
    uint32_t extra = 0;
    if (rU.rank > 0 && rS.rank > 0) {
        const engine::DenseView<const uint32_t> RS = engine::cview(GS).sub(0, rS.rank, 0, D);
        engine::DenseView<uint32_t> BU = engine::view(GU).sub(0, rU.rank, 0, D);
        const uint32_t rows = rU.rank;
        const uint32_t tiles = T > 1 ? std::min<uint32_t>(rows, 4 * T) : 1;
        engine::smithpar::parallel_for_n(T, tiles, [&](uint32_t t) {
            const uint32_t a = static_cast<uint32_t>(uint64_t(rows) * t / tiles);
            const uint32_t b = static_cast<uint32_t>(uint64_t(rows) * (t + 1) / tiles);
            if (b > a) (void)engine::elim::eliminate_against(ctx, BU.sub(a, b - a, 0, D), RS, rS.pivots, rS.rank);
        });
        DenseBuf<uint32_t> Res(rows, D);
        for (uint32_t i = 0; i < rows; ++i)
            for (uint32_t c = 0; c < D; ++c) Res.at(i, c) = BU.at(i, c);
        GU = DenseBuf<uint32_t>();
        extra = engine::elim::column_rank_profile(ctx, engine::view(Res)).rank;
    } else {
        extra = rU.rank;
    }
    out.dim_rel = out.dim_S + extra;
    out.seconds = since(t0);
    return out;
}

std::vector<uint32_t> GoodPrimeFieldDims(uint32_t n, uint32_t good, bool* sparse) {
    GoodPrimeRoute r = GoodPrimeRoute::Shipped;
    std::vector<uint32_t> dims = GoodPrimeFieldDimsRoute(n, good, &r);
    if (sparse) *sparse = r != GoodPrimeRoute::Shipped;
    return dims;
}

std::vector<uint32_t> GoodPrimeFieldDimsRoute(uint32_t n, uint32_t good, GoodPrimeRoute* route) {
    const GoodPrimeRoute r = GoodPrimeRouteFor(n);   // (PAR-L1 / PAR-GP) read once per call
    if (route) *route = r;
    if (r == GoodPrimeRoute::Shipped) return ShippedFieldDims(n, good);
    ENGINE_CHECK_MSG(good >= 5 && good > n + 1 && IsPrime(good), "GoodPrimeFieldDims: a good prime > n + 1");
    const LevelNCounts cf = LevelNClosedForm(1, n);
    if (r == GoodPrimeRoute::Dense) {
        const GoodPrimeRanks g = GoodPrimeDenseRanks(n, good);
        const uint32_t h1 = n + 1 - g.dim_rel, h0 = g.dim_S + g.dim_ST - g.dim_rel;
        ENGINE_CHECK_MSG(h0 == cf.b0 && h1 == cf.b1,
                         "GoodPrimeFieldDims: the dense ranks at the good prime disagree with the closed form b_0, b_1");
        return {h0, h1, 0u};
    }
    const hecke::level::Gamma0LevelN A1(1);
    const auto Q = hecke::level::SparseCokernel::Build(A1, n, U32Ctx(good), false);
    ENGINE_CHECK_MSG(Q.h0() == cf.b0 && Q.h1() == cf.b1,
                     "GoodPrimeFieldDims: the sparse cokernel at the good prime disagrees with the closed form b_0, b_1");
    return {Q.h0(), Q.h1(), 0u};
}

BadPrimeRun RunShippedKMode(uint32_t n, uint32_t ell, uint32_t count, KSearch search) {
    ENGINE_CHECK_MSG(ell >= 5 && IsPrime(ell), "RunShippedKMode: ell >= 5 prime ([C, prop:certifiede](5) on C_eq(Z))");
    const auto t0 = clk::now();
    BadPrimeRun out;
    out.ell = ell;
    out.count = count;
    const uint32_t budget = integral_model::DefaultKBudget(ell);
    std::vector<uint32_t> ladder;
    if (search == KSearch::Linear) {
        for (uint32_t K = 2; K <= budget; ++K) ladder.push_back(K);
    } else {
        for (uint32_t K = 2; K < budget; K *= 2) ladder.push_back(K);
        ladder.push_back(budget);
    }
    for (uint32_t K : ladder) {
        out.K_tried.push_back(K);
        const auto tk = clk::now();
        struct RunTimer {
            std::vector<double>& v;
            clk::time_point t;
            ~RunTimer() { v.push_back(std::chrono::duration<double>(clk::now() - t).count()); }
        } timer{out.run_seconds, tk};
        hecke::ScopedCoefficientRing ring(K);
        const U32Ctx ctx = U32Ctx::prime_power(ell, K);
        SimplicialComplex K7 = CreateSL2ZFundamentalDomainTriangulation();
        auto rels = hecke::GenerateSL2ZRelationsRing<uint32_t>(K7, n, ctx);
        LocalSystemSimplicialComplex lssc(K7, std::move(rels), n + 1, ctx.p);
        EquivariantCohomologyOnK coh(&lssc);
        coh.compute_cohomology();
        if (!coh.ladic_type_certified()) continue;
        out.K = K;
        out.certified = true;
        bool only1 = true;
        for (uint32_t q = 0; q < 3; ++q) {
            const auto t = coh.ladic_type(q);
            if (q == 1) {
                out.free_rank1 = t.free_rank;
                for (uint32_t a : t.torsion) out.tors1.add_torsion(ell, a);
            } else if (!t.torsion.empty()) {
                only1 = false;
            }
        }
        out.torsion_only_in_H1 = only1;
        break;
    }
    out.seconds = since(t0);
    return out;
}

// ---------------------------------------------------------------------------
// (3), (4) the ell-local pieces on the reduced complex
// ---------------------------------------------------------------------------
bool LocalPiece::all_certified() const {
    for (const auto& x : d)
        if (!x.certified) return false;
    return true;
}

namespace {

struct BlockSelection {
    std::vector<uint32_t> J;   // basis columns of im P_cell (indices into 0..D-1)
    std::vector<uint32_t> R;   // rows on which the coordinate projection is an isomorphism on im P_cell
};

// The selection for one column spec: identity when Q = 1; else F_ell rank profiles of rho(e) and of rho(e)[:, J]^T.
BlockSelection select_block(const U32Ctx& ctx, const integral_model::ColumnSpec& col, const CoefficientModule<uint32_t>& M,
                            std::map<integral_model::GKey, DenseBuf<uint32_t>>& memo) {
    const uint32_t D = M.dim;
    BlockSelection s;
    if (col.q_order == 1) {
        s.J.resize(D);
        for (uint32_t i = 0; i < D; ++i) s.J[i] = i;
        s.R = s.J;
        return s;
    }
    const uint32_t ell = ctx.ell;
    const U32Ctx F(ell);
    const DenseBuf<uint32_t> P = integral_model::RhoQ(ctx, col.e, M, memo);
    DenseBuf<uint32_t> Pf(D, D);
    for (uint32_t i = 0; i < D; ++i)
        for (uint32_t j = 0; j < D; ++j) Pf.at(i, j) = P.at(i, j) % ell;
    s.J = engine::elim::rref(F, engine::view(Pf)).pivots;
    if (s.J.empty()) return s;
    DenseBuf<uint32_t> Bt(static_cast<uint32_t>(s.J.size()), D);
    for (uint32_t a = 0; a < s.J.size(); ++a)
        for (uint32_t i = 0; i < D; ++i) Bt.at(a, i) = P.at(i, s.J[a]) % ell;
    s.R = engine::elim::rref(F, engine::view(Bt)).pivots;
    ENGINE_CHECK_MSG(s.R.size() == s.J.size(), "select_block: rho(e)[:, J] is not of full column rank mod ell");
    return s;
}

// Smith valuations sigma (ascending, each < K) of a dense matrix over Z/ell^K on the native ring engines.
std::vector<uint32_t> smith_valuations(const U32Ctx& ctx, const DenseBuf<uint32_t>& A) {
    if (A.rows == 0 || A.cols == 0) return {};
    const engine::smithpar::ScopedEnable mt;   // (S24 PAR-2K) the threaded ring2k / ring3k Smith (HECKE_DESCENT_SMITH_MT)
    const uint32_t ell = ctx.ell, K = ctx.k;
    if (ell == 2 && K >= 2) {
        engine::ring2k::Mat<uint32_t> W(A.rows, A.cols);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) W.at(i, j) = A.at(i, j);
        return engine::ring2k::smith<engine::ring2k::Traits<uint32_t>>(std::move(W), K).sigma;
    }
    if (ell == 3 && K >= 2 && K <= engine::ring3k::kMaxK) {
        DenseBuf<uint32_t> W(A.rows, A.cols);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) W.at(i, j) = A.at(i, j);
        return engine::ring3k::smith(std::move(W), K).sigma;
    }
    // general Z/ell^K (or F_ell): the engine's invariant factors e_i (colspan ~ (+) Z/ell^{e_i}), sigma = K - e_i.
    if (ctx.is_field()) {
        DenseBuf<uint32_t> W = A;
        const uint32_t r = engine::elim::rref(ctx, engine::view(W)).rank;
        return std::vector<uint32_t>(r, 0u);
    }
    std::vector<uint32_t> e = engine::elim::invariant_factors<uint32_t>(ctx, engine::cview(A));
    std::vector<uint32_t> sigma;
    for (uint32_t x : e) {
        ENGINE_CHECK(x >= 1 && x <= K);
        sigma.push_back(K - x);
    }
    std::sort(sigma.begin(), sigma.end());
    return sigma;
}

}  // namespace

uint32_t QEll(uint32_t ell) {
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "QEll: ell in {2, 3}");
    return ell == 2 ? 3u : 4u;   // [C, prop:certifiedeell]: q_2 = lcm|Q_sigma| = 3, q_3 = 4 (Q_{v6} = <S>)
}

uint64_t CheckDenominators(const LocalModel& L, uint32_t q_ell) {
    // (C-D0 gap (2); referee F5) every denominator of a coefficient a_{yx} of d is a product of powers of |Q_x| and
    // |Q_y| (a_{yx} = e_y a_{yx} e_x), every denominator of e_sigma a power of |Q_sigma|, and all divide a power of q_ell.
    uint64_t checked = 0;
    auto powers_of = [](int64_t den, uint32_t qa, uint32_t qb) {
        if (den <= 0) return false;
        uint64_t v = static_cast<uint64_t>(den);
        bool moved = true;
        while (v > 1 && moved) {
            moved = false;
            if (qa > 1 && v % qa == 0) { v /= qa; moved = true; }
            else if (qb > 1 && v % qb == 0) { v /= qb; moved = true; }
        }
        return v == 1;
    };
    auto divides_power = [q_ell](int64_t den) {
        uint64_t v = static_cast<uint64_t>(den);
        for (int i = 0; i < 64 && v > 1; ++i) {
            const uint64_t g = std::gcd<uint64_t>(v, q_ell);
            if (g == 1) return false;
            v /= g;
        }
        return v == 1;
    };
    auto qord = [&L](uint32_t deg, uint32_t cell) {
        const auto& c = L.cells(deg)[cell];
        return L.column(c.s, c.rep).q_order;
    };
    for (uint32_t q = 1; q <= L.q_top(); ++q)
        for (const auto& e : L.differential(q)) {
            const uint32_t qx = qord(q - 1, e.row), qy = qord(q, e.col);
            for (const auto& kv : e.coeff.terms)
                ENGINE_CHECK_MSG(powers_of(kv.second.den, qx, qy) && divides_power(kv.second.den),
                                 "CheckDenominators: a coefficient of d has a denominator that is not a product of powers "
                                 "of |Q_x|, |Q_y| dividing a power of q_ell ([C, prop:certifiedeell](1))");
            ++checked;
        }
    for (uint32_t s = 0; s < 3; ++s)
        for (uint32_t r = 0; r < L.num_reps(s); ++r) {
            const auto& col = L.column(s, r);
            for (const auto& kv : col.e.terms)
                ENGINE_CHECK_MSG(powers_of(kv.second.den, col.q_order, 1) && divides_power(kv.second.den),
                                 "CheckDenominators: an idempotent e_sigma has a denominator not a power of |Q_sigma|");
            ++checked;
        }
    return checked;
}

namespace {

// The clause's delta^q restricted to the cochains, in coordinates ([C, prop:certifiedeell](5)): with B_q = blockdiag
// P_x[:, J_x] (a basis of im P_q over Z_(ell)) and pi_R the coordinate projection onto the rows R_y of each block (an
// isomorphism on im P_y), A = pi_R P_{q+1} delta_code B_q.  delta_code is LocalModel::coboundary (blocks rho(a) of the
// stored coefficients); the outer projectors make A the matrix of P_{q+1} delta P_q whatever normalisation the stored
// coefficients carry (a_{yx} = e_y a_{yx} e_x in the clause).  The DenseBuf -> ring hand-off of [C, prop:certifiedeell]'s
// code map.
template<class Elt>
DenseBuf<Elt> clause_matrix(const LocalModel& L, const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M,
                            uint32_t q, const std::vector<std::vector<uint32_t>>& Jq, const std::vector<std::vector<uint32_t>>& Rq1,
                            const std::vector<integral_model::ColumnSpec const*>& colq,
                            const std::vector<integral_model::ColumnSpec const*>& colq1, uint64_t* peak_bytes) {
    const uint32_t D = M.dim;
    const DenseBuf<Elt> full = L.coboundary(ctx, M, q);
    const uint32_t nx = L.rank(q), ny = L.rank(q + 1);
    ENGINE_CHECK(full.rows == ny * D && full.cols == nx * D);
    std::map<integral_model::GKey, DenseBuf<Elt>> memo;
    // T = delta_code B_q  (full rows x reduced cols)
    uint32_t ncols = 0, nrows = 0;
    for (const auto& j : Jq) ncols += static_cast<uint32_t>(j.size());
    for (const auto& r : Rq1) nrows += static_cast<uint32_t>(r.size());
    DenseBuf<Elt> T(full.rows, ncols);
    uint32_t c0 = 0;
    for (uint32_t x = 0; x < nx; ++x) {
        const auto& J = Jq[x];
        if (J.empty()) continue;
        if (colq[x]->q_order == 1) {
            for (uint32_t i = 0; i < full.rows; ++i)
                for (uint32_t a = 0; a < J.size(); ++a) T.at(i, c0 + a) = full.at(i, x * D + J[a]);
        } else {
            const DenseBuf<Elt> P = integral_model::RhoQ(ctx, colq[x]->e, M, memo);
            DenseBuf<Elt> PJ(D, static_cast<uint32_t>(J.size())), Fx(full.rows, D), TJ(full.rows, static_cast<uint32_t>(J.size()));
            for (uint32_t i = 0; i < D; ++i)
                for (uint32_t a = 0; a < J.size(); ++a) PJ.at(i, a) = P.at(i, J[a]);
            for (uint32_t i = 0; i < full.rows; ++i)
                for (uint32_t b = 0; b < D; ++b) Fx.at(i, b) = full.at(i, x * D + b);
            engine::kernels::gemm<Elt>(ctx, engine::cview(Fx), engine::cview(PJ), engine::view(TJ));
            for (uint32_t i = 0; i < full.rows; ++i)
                for (uint32_t a = 0; a < J.size(); ++a) T.at(i, c0 + a) = TJ.at(i, a);
        }
        c0 += static_cast<uint32_t>(J.size());
    }
    // A = pi_R P_{q+1} T
    DenseBuf<Elt> A(nrows, ncols);
    uint32_t r0 = 0;
    for (uint32_t y = 0; y < ny; ++y) {
        const auto& R = Rq1[y];
        if (R.empty()) continue;
        if (colq1[y]->q_order == 1) {
            for (uint32_t a = 0; a < R.size(); ++a)
                for (uint32_t j = 0; j < ncols; ++j) A.at(r0 + a, j) = T.at(y * D + R[a], j);
        } else {
            const DenseBuf<Elt> P = integral_model::RhoQ(ctx, colq1[y]->e, M, memo);
            DenseBuf<Elt> PR(static_cast<uint32_t>(R.size()), D), Ty(D, ncols), AR(static_cast<uint32_t>(R.size()), ncols);
            for (uint32_t a = 0; a < R.size(); ++a)
                for (uint32_t b = 0; b < D; ++b) PR.at(a, b) = P.at(R[a], b);
            for (uint32_t b = 0; b < D; ++b)
                for (uint32_t j = 0; j < ncols; ++j) Ty.at(b, j) = T.at(y * D + b, j);
            engine::kernels::gemm<Elt>(ctx, engine::cview(PR), engine::cview(Ty), engine::view(AR));
            for (uint32_t a = 0; a < R.size(); ++a)
                for (uint32_t j = 0; j < ncols; ++j) A.at(r0 + a, j) = AR.at(a, j);
        }
        r0 += static_cast<uint32_t>(R.size());
    }
    if (peak_bytes)
        *peak_bytes = std::max<uint64_t>(*peak_bytes, sizeof(Elt) * (uint64_t(full.rows) * full.cols +
                                                                     uint64_t(T.rows) * T.cols + 2ull * A.rows * A.cols));
    return A;
}

// P_{q+1} delta_code P_q on the full layout over a field (the rank source (4)(c)(i): this model mod ell').
uint32_t clause_rank_full(const LocalModel& L, const U32Ctx& F, const CoefficientModule<uint32_t>& M, uint32_t q) {
    const uint32_t D = M.dim;
    DenseBuf<uint32_t> full = L.coboundary(F, M, q);
    if (full.rows == 0 || full.cols == 0) return 0;
    std::map<integral_model::GKey, DenseBuf<uint32_t>> memo;
    auto blockP = [&](uint32_t deg, uint32_t cell) -> const integral_model::ColumnSpec& {
        const auto& c = L.cells(deg)[cell];
        return L.column(c.s, c.rep);
    };
    for (uint32_t x = 0; x < L.rank(q); ++x) {   // right: delta P_q
        const auto& col = blockP(q, x);
        if (col.q_order == 1) continue;
        const DenseBuf<uint32_t> P = integral_model::RhoQ(F, col.e, M, memo);
        DenseBuf<uint32_t> Fx(full.rows, D), O(full.rows, D);
        for (uint32_t i = 0; i < full.rows; ++i)
            for (uint32_t b = 0; b < D; ++b) Fx.at(i, b) = full.at(i, x * D + b);
        engine::kernels::gemm<uint32_t>(F, engine::cview(Fx), engine::cview(P), engine::view(O));
        for (uint32_t i = 0; i < full.rows; ++i)
            for (uint32_t b = 0; b < D; ++b) full.at(i, x * D + b) = O.at(i, b);
    }
    for (uint32_t y = 0; y < L.rank(q + 1); ++y) {   // left: P_{q+1} delta
        const auto& col = blockP(q + 1, y);
        if (col.q_order == 1) continue;
        const DenseBuf<uint32_t> P = integral_model::RhoQ(F, col.e, M, memo);
        DenseBuf<uint32_t> Fy(D, full.cols), O(D, full.cols);
        for (uint32_t b = 0; b < D; ++b)
            for (uint32_t j = 0; j < full.cols; ++j) Fy.at(b, j) = full.at(y * D + b, j);
        engine::kernels::gemm<uint32_t>(F, engine::cview(P), engine::cview(Fy), engine::view(O));
        for (uint32_t b = 0; b < D; ++b)
            for (uint32_t j = 0; j < full.cols; ++j) full.at(y * D + b, j) = O.at(b, j);
    }
    return engine::elim::rref(F, engine::view(full)).rank;
}

}  // namespace

// (D2) the reduced-complex building blocks of RunLocalPiece, exposed unchanged for Integral/Descent/DescentMaps.
void SelectBlockJR(const U32Ctx& ctx, const integral_model::ColumnSpec& col, const CoefficientModule<uint32_t>& M,
                   std::map<integral_model::GKey, DenseBuf<uint32_t>>& memo, std::vector<uint32_t>& J, std::vector<uint32_t>& R) {
    BlockSelection s = select_block(ctx, col, M, memo);
    J = std::move(s.J);
    R = std::move(s.R);
}
DenseBuf<uint32_t> ClauseMatrixReduced(const LocalModel& L, const U32Ctx& ctx, const CoefficientModule<uint32_t>& M, uint32_t q,
                                       const std::vector<std::vector<uint32_t>>& Jq, const std::vector<std::vector<uint32_t>>& Rq1,
                                       const std::vector<const integral_model::ColumnSpec*>& colq,
                                       const std::vector<const integral_model::ColumnSpec*>& colq1) {
    return clause_matrix<uint32_t>(L, ctx, M, q, Jq, Rq1, colq, colq1, nullptr);
}

LocalPiece RunLocalPiece(const LocalModel& L, const CoefficientModule<uint32_t>& M, const U32Ctx& ctx, uint32_t q_top,
                         const std::vector<uint32_t>& h, uint32_t rank_prime, const CoefficientModule<uint32_t>& M_rank,
                         int sym_n, bool force_wide, bool rank_from_betti) {
    const auto t0 = clk::now();
    const uint32_t ell = L.ell();
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "RunLocalPiece: Iso^(ell) at ell in {2, 3}");
    // ([C, prop:certifiedeell] code map; audit F-5) the model's ell is tied to the ring it is reduced into.
    ENGINE_CHECK_MSG(ctx.ell == ell && !ctx.is_field(), "RunLocalPiece: the context must be Z/ell^K, K >= 2, at the model's ell");
    ENGINE_CHECK_MSG(q_top >= 1 && q_top <= L.q_top(), "RunLocalPiece: q_top out of the model's range");
    ENGINE_CHECK_MSG(h.size() >= q_top + 1u, "RunLocalPiece: rational dims must cover q = 0 .. q_top");
    ENGINE_CHECK_MSG(rank_prime >= 5 && IsPrime(rank_prime) && rank_prime != ell && rank_prime % QEll(ell) != 0,
                     "RunLocalPiece: the rank prime ell' must be >= 5 and prime to q_ell ([C, prop:certifiedeell](4))");
    ENGINE_CHECK_MSG(!force_wide || (ell == 2 && sym_n >= 0), "RunLocalPiece: the W = 61 route is Sym^n at ell = 2 only");
    ENGINE_CHECK_MSG(M_rank.dim == M.dim, "RunLocalPiece: the rank module must be the same module (same dimension) over F_ell'");
    const uint32_t D = M.dim;
    LocalPiece out;
    out.ell = ell;
    out.K = ctx.k;
    out.q_top = q_top;
    out.rank_prime = rank_prime;
    // ([C, prop:certifiedeell](1), code map) build-time: every denominator divides q_ell, as powers of |Q_sigma|.
    out.denominators_checked = CheckDenominators(L, QEll(ell));

    // per (s, rep) selections (mod ell, from the uint32_t context); per degree the per-cell lists
    std::map<integral_model::GKey, DenseBuf<uint32_t>> memo;
    std::map<std::pair<uint32_t, uint32_t>, BlockSelection> sel;
    std::vector<std::vector<std::vector<uint32_t>>> J(q_top + 1), R(q_top + 1);
    std::vector<std::vector<const integral_model::ColumnSpec*>> col(q_top + 1);
    for (uint32_t q = 0; q <= q_top; ++q) {
        const auto& cells = L.cells(q);
        uint32_t c = 0;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            const auto key = std::make_pair(cells[i].s, cells[i].rep);
            const auto& cs = L.column(cells[i].s, cells[i].rep);
            auto it = sel.find(key);
            if (it == sel.end()) it = sel.emplace(key, select_block(ctx, cs, M, memo)).first;
            J[q].push_back(it->second.J);
            R[q].push_back(it->second.R);
            col[q].push_back(&cs);
            c += static_cast<uint32_t>(it->second.J.size());
        }
        out.full_rank.push_back(static_cast<uint32_t>(cells.size()) * D);
        out.reduced_rank.push_back(c);
    }
    memo.clear();
    out.tors.assign(q_top + 1, AbelianGroupType{});

    // source (4)(c)(ii), a cross-check: rank_Q d^q = c_q - b_q - rank_Q d^{q-1}
    std::vector<uint32_t> derived(q_top, 0);
    {
        uint32_t prev = 0;
        for (uint32_t q = 0; q < q_top; ++q) {
            const int64_t r = int64_t(out.reduced_rank[q]) - int64_t(prev) - int64_t(h[q]);
            ENGINE_CHECK_MSG(r >= 0, "RunLocalPiece: negative derived rank_Q d^q (rational dims inconsistent)");
            derived[q] = static_cast<uint32_t>(r);
            prev = derived[q];
        }
    }

    const U32Ctx RP(rank_prime);
    for (uint32_t q = 0; q < q_top; ++q) {
        LocalDifferential ld;
        ld.q = q;
        // [C, prop:certifiedeell](4)(c)(i): rank_{F_ell'} of THIS LocalModel(desc, ell) reduced mod ell' (the clause's
        // delta^q = P delta P on the full layout) -- never a census, shipped-C_7 or Iso^(ell') rank.  ell' qualifies for
        // q >= 1 by (4)(a); for q = 0 by (4)(b) iff rank P_0 - rank delta^0 = b_0, which is the equality with the derived
        // rank below (b_0 = h[0], the closed form for Sym^n / Ind).
        const auto tr = clk::now();
        // (D1) rank_from_betti: source (4)(c)(ii) alone -- the derived rank from certified Betti numbers h.
        ld.rank_Q = rank_from_betti ? derived[q] : clause_rank_full(L, RP, M_rank, q);
        out.rank_seconds += since(tr);
        ld.rank_derived = derived[q];
        if (ld.rank_Q != ld.rank_derived)
            std::fprintf(stderr, "RunLocalPiece: rank mismatch ell=%u D=%u q=%u rank_F=%u derived=%u c_q=%u c_q1=%u h=%u,%u full=%u\n", ell, D, q,
                         ld.rank_Q, ld.rank_derived, out.reduced_rank[q], out.reduced_rank[q + 1], h[0], h[1], out.full_rank[q]);
        if (q == 0) out.dimH0_rank_prime = out.reduced_rank[0] - std::min(out.reduced_rank[0], ld.rank_Q);
        ENGINE_CHECK_MSG(ld.rank_Q == ld.rank_derived,
                         "RunLocalPiece: rank_{F_ell'} d^q != c_q - b_q - rank d^{q-1} (ell' does not qualify, or wrong b_q)");

        const auto tb = clk::now();
        const DenseBuf<uint32_t> A = clause_matrix<uint32_t>(L, ctx, M, q, J[q], R[q + 1], col[q], col[q + 1],
                                                            &out.peak_matrix_bytes);
        out.build_seconds += since(tb);
        ld.rows = A.rows;
        ld.cols = A.cols;
        std::vector<uint32_t> sigma;
        if (!force_wide) {
            const auto ts = clk::now();
            sigma = smith_valuations(ctx, A);
            out.smith_seconds += since(ts);
            ld.W = ctx.k;
            // [C, prop:certifiede](1) via [C, prop:certifiedeell](3),(6): r_K <= rank_Q, always.
            ENGINE_CHECK_MSG(sigma.size() <= ld.rank_Q, "RunLocalPiece: r_K(d^q) > rank_Q(d^q)");
        }
        if ((force_wide || sigma.size() < ld.rank_Q) && ell == 2 && sym_n >= 0) {
            // The stopping rule of [C, prop:certifiedeell](6) at 2: rerun at W = 61 through ModCtx<uint64_t> (the code map's
            // O_2 -> Z/2^W, W <= 61), on ring2k's uint64_t rung read mod 2^61.
            const auto tw = clk::now();
            const auto c64 = engine::ModCtx<uint64_t>::prime_power(2, 61);
            const auto M64 = integral_model::SymModule(c64, static_cast<uint32_t>(sym_n));
            const DenseBuf<uint64_t> A64 = clause_matrix<uint64_t>(L, c64, M64, q, J[q], R[q + 1], col[q], col[q + 1],
                                                                  &out.peak_matrix_bytes);
            // the W = 61 matrix read mod 2^K is the K-route's matrix (O_2 -> Z/2^61 -> Z/2^K)
            const uint64_t mask = (uint64_t(1) << ctx.k) - 1;
            for (uint32_t i = 0; i < A.rows; ++i)
                for (uint32_t j = 0; j < A.cols; ++j)
                    ENGINE_CHECK_MSG((A64.at(i, j) & mask) == A.at(i, j), "RunLocalPiece: the W = 61 matrix mod 2^K differs");
            engine::ring2k::Mat<uint64_t> W(A64.rows, A64.cols);
            for (uint32_t i = 0; i < A64.rows; ++i)
                for (uint32_t j = 0; j < A64.cols; ++j) W.at(i, j) = A64.at(i, j);
            out.build_seconds += since(tw);
            const auto ts = clk::now();
            {
                const engine::smithpar::ScopedEnable mt;   // (S24 PAR-2K)
                sigma = engine::ring2k::smith<engine::ring2k::Traits<uint64_t>>(std::move(W), 61).sigma;
            }
            out.smith_seconds += since(ts);
            ld.W = 61;
            ld.escalated = !force_wide;
            ENGINE_CHECK_MSG(sigma.size() <= ld.rank_Q, "RunLocalPiece: r_K(d^q) > rank_Q(d^q) at W = 61");
        }
        ld.r_K = static_cast<uint32_t>(sigma.size());
        // [C, prop:certifiedeell](6): certified iff s = rank_Q; else (at 3, K = 19 = ring3k's cap) a REFUSAL.
        ld.certified = ld.r_K == ld.rank_Q;
        if (ld.certified) {
            for (uint32_t s : sigma)
                if (s >= 1) ld.nonunit.push_back(s);
            std::sort(ld.nonunit.begin(), ld.nonunit.end(), std::greater<uint32_t>());
            // a-priori exponents for q >= 1: a_t <= v_ell(24) (3 at 2, 1 at 3) -- a consistency check of the read-out
            if (q >= 1)
                for (uint32_t a : ld.nonunit)
                    ENGINE_CHECK_MSG(a <= (ell == 2 ? 3u : 1u),
                                     "RunLocalPiece: an exponent above v_ell(24) at q >= 1 ([C, prop:certifiedeell](6))");
            for (uint32_t a : ld.nonunit) out.tors[q + 1].add_torsion(ell, a);
        }
        out.d.push_back(std::move(ld));
    }
    out.seconds = since(t0);
    return out;
}

LocalPiece RunLocalPieceSym(const LocalModel& L, uint32_t n, uint32_t q_top, const std::vector<uint32_t>& h,
                            uint32_t rank_prime, uint32_t K, bool force_wide) {
    if (K == 0) K = LocalWidth(L.ell());
    const U32Ctx ctx = U32Ctx::prime_power(L.ell(), K);
    return RunLocalPiece(L, integral_model::SymModule(ctx, n), ctx, q_top, h, rank_prime,
                         integral_model::SymModule(U32Ctx(rank_prime), n), static_cast<int>(n), force_wide);
}

// ---------------------------------------------------------------------------
// (D1) the direct Smith form at ell >= 5 ([C, prop:descentN](7)(a))
// ---------------------------------------------------------------------------
std::vector<uint32_t> SmithLevelsEll(uint32_t ell, uint32_t K, DenseBuf<uint32_t> A) {
    ENGINE_CHECK_MSG(ell >= 3 && IsPrime(ell) && K >= 1, "SmithLevelsEll: ell an odd prime, K >= 1");
    uint64_t pk = 1;
    for (uint32_t i = 0; i < K; ++i) pk *= ell;
    ENGINE_CHECK_MSG(pk < (uint64_t(1) << 31), "SmithLevelsEll: ell^K must stay below 2^31 (uint32_t ModCtx)");
    std::vector<uint32_t> sigma;
    const U32Ctx F(ell);
    for (uint32_t s = 0; s < K && A.rows > 0 && A.cols > 0; ++s) {
        const uint32_t kk = K - s;
        const U32Ctx C = kk >= 2 ? U32Ctx::prime_power(ell, kk) : U32Ctx(ell);
        bool any = false;
        for (uint32_t i = 0; i < A.rows && !any; ++i)
            for (uint32_t j = 0; j < A.cols; ++j)
                if (A.at(i, j)) { any = true; break; }
        if (!any) break;
        // the F_ell column rank profile Jc and, on those columns, the row rank profile Ir
        DenseBuf<uint32_t> Af(A.rows, A.cols);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) Af.at(i, j) = A.at(i, j) % ell;
        const std::vector<uint32_t> Jc = engine::elim::rref(F, engine::view(Af)).pivots;
        DenseBuf<uint32_t>().swap(Af);
        const uint32_t r0 = static_cast<uint32_t>(Jc.size());
        for (uint32_t t = 0; t < r0; ++t) sigma.push_back(s);
        std::vector<uint32_t> Ir;
        if (r0 > 0) {
            DenseBuf<uint32_t> Bt(r0, A.rows);
            for (uint32_t a = 0; a < r0; ++a)
                for (uint32_t i = 0; i < A.rows; ++i) Bt.at(a, i) = A.at(i, Jc[a]) % ell;
            Ir = engine::elim::rref(F, engine::view(Bt)).pivots;
            ENGINE_CHECK_MSG(Ir.size() == r0, "SmithLevelsEll: the row rank profile has the wrong size");
        }
        std::vector<uint8_t> inJ(A.cols, 0), inI(A.rows, 0);
        for (uint32_t j : Jc) inJ[j] = 1;
        for (uint32_t i : Ir) inI[i] = 1;
        std::vector<uint32_t> Jr, Irr;
        for (uint32_t j = 0; j < A.cols; ++j)
            if (!inJ[j]) Jr.push_back(j);
        for (uint32_t i = 0; i < A.rows; ++i)
            if (!inI[i]) Irr.push_back(i);
        const uint32_t sr = static_cast<uint32_t>(Irr.size()), sc = static_cast<uint32_t>(Jr.size());
        DenseBuf<uint32_t> Sc(sr, sc);
        for (uint32_t a = 0; a < sr; ++a)
            for (uint32_t b = 0; b < sc; ++b) Sc.at(a, b) = A.at(Irr[a], Jr[b]) % C.p;
        if (r0 > 0 && sr > 0 && sc > 0) {
            DenseBuf<uint32_t> A11(r0, r0), A12(r0, sc), A21(sr, r0);
            for (uint32_t a = 0; a < r0; ++a) {
                for (uint32_t b = 0; b < r0; ++b) A11.at(a, b) = A.at(Ir[a], Jc[b]) % C.p;
                for (uint32_t b = 0; b < sc; ++b) A12.at(a, b) = A.at(Ir[a], Jr[b]) % C.p;
            }
            for (uint32_t a = 0; a < sr; ++a)
                for (uint32_t b = 0; b < r0; ++b) A21.at(a, b) = A.at(Irr[a], Jc[b]) % C.p;
            DenseBuf<uint32_t>().swap(A);
            // X0 = A11^{-1} mod ell: rref of [A11 | I] over F_ell
            DenseBuf<uint32_t> W(r0, 2 * r0);
            for (uint32_t a = 0; a < r0; ++a) {
                for (uint32_t b = 0; b < r0; ++b) W.at(a, b) = A11.at(a, b) % ell;
                W.at(a, r0 + a) = 1;
            }
            const auto rw = engine::elim::rref(F, engine::view(W));
            ENGINE_CHECK_MSG(rw.rank == r0 && rw.pivots[r0 - 1] == r0 - 1, "SmithLevelsEll: A11 is not invertible mod ell");
            DenseBuf<uint32_t> X0(r0, r0);
            for (uint32_t a = 0; a < r0; ++a)
                for (uint32_t b = 0; b < r0; ++b) X0.at(a, b) = W.at(a, r0 + b);
            DenseBuf<uint32_t>().swap(W);
            // Y = A11^{-1} A12 mod ell^kk by ell-adic lifting: R_0 = A12, y_i = X0 R_i mod ell,
            // R_{i+1} = (R_i - A11 y_i) / ell (exact), Y = sum ell^i y_i.
            DenseBuf<uint32_t> Y(r0, sc), R = A12, yi(r0, sc), Rm(r0, sc), Rf(r0, sc);
            uint64_t lp = 1;
            for (uint32_t i = 0; i < kk; ++i) {
                for (uint32_t a = 0; a < r0; ++a)
                    for (uint32_t b = 0; b < sc; ++b) Rf.at(a, b) = R.at(a, b) % ell;
                engine::kernels::gemm<uint32_t>(F, engine::cview(X0), engine::cview(Rf), engine::view(yi));
                for (uint32_t a = 0; a < r0; ++a)
                    for (uint32_t b = 0; b < sc; ++b)
                        Y.at(a, b) = static_cast<uint32_t>((Y.at(a, b) + lp * yi.at(a, b)) % C.p);
                if (i + 1 == kk) break;
                engine::kernels::gemm<uint32_t>(C, engine::cview(A11), engine::cview(yi), engine::view(Rm));
                for (uint32_t a = 0; a < r0; ++a)
                    for (uint32_t b = 0; b < sc; ++b) {
                        const uint32_t v = C.sub(R.at(a, b) % C.p, Rm.at(a, b));
                        ENGINE_DEBUG_CHECK_MSG(v % ell == 0, "SmithLevelsEll: the lifting residual is not 0 mod ell");
                        R.at(a, b) = v / ell;
                    }
                lp *= ell;
            }
            // Schur complement S = A22 - A21 Y mod ell^kk
            engine::kernels::gemm<uint32_t>(C, engine::cview(A21), engine::cview(Y), engine::view(Sc),
                                            engine::kernels::GemmOp::Sub);
        } else {
            DenseBuf<uint32_t>().swap(A);
        }
        // S is 0 mod ell (the F_ell rank is r0); divide by ell exactly
        for (uint32_t a = 0; a < sr; ++a)
            for (uint32_t b = 0; b < sc; ++b) {
                const uint32_t x = Sc.at(a, b);
                ENGINE_CHECK_MSG(x % ell == 0, "SmithLevelsEll: the Schur complement is not 0 mod ell");
                Sc.at(a, b) = x / ell;
            }
        A = std::move(Sc);
    }
    return sigma;
}

const LocalModel& BredonModel(uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 5 && IsPrime(ell), "BredonModel: ell >= 5 prime");
    static std::map<uint32_t, std::unique_ptr<LocalModel>> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(ell);
    if (it == cache.end()) {
        auto L = std::make_unique<LocalModel>(Descriptor5(), ell);
        for (uint32_t q = 0; q < 3; ++q)
            for (const auto& c : L->cells(q))
                ENGINE_CHECK_MSG(c.t == 0 && !L->column(c.s, c.rep).periodic(),
                                 "BredonModel: a periodic column at ell >= 5 (a stabiliser order divisible by ell)");
        ENGINE_CHECK_MSG(L->rank(3) == 0, "BredonModel: cells above degree 2 at ell >= 5");
        it = cache.emplace(ell, std::move(L)).first;
    }
    return *it->second;
}

BredonSmithRun RunBredonDirectSmith(const CoefficientModule<uint32_t>& M, uint32_t ell, uint32_t K, uint32_t b0,
                                    bool verify_d1) {
    const auto t0 = clk::now();
    if (K == 0) K = integral_model::DefaultKBudget(ell);
    const LocalModel& L = BredonModel(ell);
    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
    BredonSmithRun out;
    out.ell = ell;
    out.K = K;
    std::map<integral_model::GKey, DenseBuf<uint32_t>> memo;
    std::map<std::pair<uint32_t, uint32_t>, BlockSelection> sel;
    std::vector<std::vector<std::vector<uint32_t>>> J(3), R(3);
    std::vector<std::vector<const integral_model::ColumnSpec*>> col(3);
    uint32_t c[3] = {0, 0, 0};
    for (uint32_t q = 0; q < 3; ++q)
        for (const auto& cell : L.cells(q)) {
            const auto key = std::make_pair(cell.s, cell.rep);
            const auto& cs = L.column(cell.s, cell.rep);
            auto it = sel.find(key);
            if (it == sel.end()) it = sel.emplace(key, select_block(ctx, cs, M, memo)).first;
            J[q].push_back(it->second.J);
            R[q].push_back(it->second.R);
            col[q].push_back(&cs);
            c[q] += static_cast<uint32_t>(it->second.J.size());
        }
    memo.clear();
    out.c0 = c[0];
    out.c1 = c[1];
    out.c2 = c[2];
    ENGINE_CHECK_MSG(b0 <= c[0], "RunBredonDirectSmith: b_0 > c_0");
    out.rank_Q = c[0] - b0;   // [C, prop:descentN](7)(a): rank_Q d^0 = rank C^0 - b_0, the closed-form b_0
    auto tb = clk::now();
    DenseBuf<uint32_t> A = clause_matrix<uint32_t>(L, ctx, M, 0, J[0], R[1], col[0], col[1], &out.peak_matrix_bytes);
    out.build_seconds += since(tb);
    auto ts = clk::now();
    const std::vector<uint32_t> sigma = SmithLevelsEll(ell, K, std::move(A));
    out.smith_seconds += since(ts);
    out.r_K = static_cast<uint32_t>(sigma.size());
    ENGINE_CHECK_MSG(out.r_K <= out.rank_Q, "RunBredonDirectSmith: r_K(d^0) > rank_Q(d^0)");
    out.certified = out.r_K == out.rank_Q;
    if (out.certified)
        for (uint32_t x : sigma)
            if (x >= 1) out.tors1.add_torsion(ell, x);
    if (verify_d1) {
        tb = clk::now();
        DenseBuf<uint32_t> A1 = clause_matrix<uint32_t>(L, ctx, M, 1, J[1], R[2], col[1], col[2], &out.peak_matrix_bytes);
        out.build_seconds += since(tb);
        ts = clk::now();
        const std::vector<uint32_t> s1 = SmithLevelsEll(ell, K, std::move(A1));
        out.smith_seconds += since(ts);
        out.d1_checked = true;
        out.d1_onto = s1.size() == c[2] && std::all_of(s1.begin(), s1.end(), [](uint32_t v) { return v == 0; });
    }
    // rank H^1 = c_1 - rank d^0 - rank d^1, rank d^1 = c_2 (d^1 onto, [C, prop:bredon](2))
    ENGINE_CHECK_MSG(c[1] >= out.rank_Q + c[2], "RunBredonDirectSmith: c_1 < rank d^0 + c_2");
    out.free_rank1 = c[1] - out.rank_Q - c[2];
    out.seconds = since(t0);
    return out;
}

uint64_t PredictTreeBredonRss(uint32_t D) {
    // (PAR-ASM (b)) one RunTreeBredonLevel1 / RunTreeBredonSym at D = n + 1: the tree module's rho memo (dense d x d blocks, d = D at
    // level 1, at most a dozen group elements), d^0 (c_1 x c_0 <= D x D), SmithLevelsEll's working set (the F_ell copy, the
    // blocks A11, A12, A21, [A11 | I], X0, the lifting buffers and the Schur complement: < 6 D^2 words): 24 D^2 words, plus a base.
    // Calibrated before the concurrent runs (PARASM_report.md Sec. 4: measured <= 0.5 x this at n = 500 .. 2000).
    return 96ull * uint64_t(D) * D + (32ull << 20);
}

BadPrimeRun RunDirectSmithLevel1(uint32_t n, uint32_t ell, uint32_t count, bool verify_d1) {
    const auto t0 = clk::now();
    BadPrimeRun out;
    out.ell = ell;
    out.count = count;
    const uint32_t K = integral_model::DefaultKBudget(ell);
    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
    const BredonSmithRun b = RunBredonDirectSmith(integral_model::SymModule(ctx, n), ell, K, n == 0 ? 1u : 0u, verify_d1);
    out.K_tried.push_back(K);
    out.run_seconds.push_back(b.seconds);
    out.certified = b.certified && (!verify_d1 || b.d1_onto);
    if (out.certified) out.K = K;
    out.tors1 = b.tors1;
    out.free_rank1 = b.free_rank1;
    // (7)(a): d^1 is onto, so ell >= 5 torsion lies in H^1 only (H^2 of C_eq(M_O) is 0, [C, prop:bredon](2)); verify_d1
    // tests it on the object.
    out.torsion_only_in_H1 = !verify_d1 || b.d1_onto;
    out.seconds = since(t0);
    return out;
}

// ---------------------------------------------------------------------------
// the glue ([C, prop:descent](2))
// ---------------------------------------------------------------------------
bool DescentResult::certified() const {
    if (refused) return false;
    if (!two.all_certified() || !three.all_certified()) return false;
    for (const auto& b : bad) {
        if (!b.certified || !b.torsion_only_in_H1) return false;
        const auto it = b.tors1.torsion.find(b.ell);
        if ((it == b.tors1.torsion.end() ? 0u : static_cast<uint32_t>(it->second.size())) != b.count) return false;
    }
    return true;
}

DescentResult AssembleDescent(uint32_t n, const DescentOptions& opt) {
    const auto t0 = clk::now();
    DescentResult R;
    R.n = n;
    R.good_prime = opt.good_prime ? opt.good_prime : DefaultGoodPrime(n);
    // (D5 audit N6) the route is read ONCE per call: a concurrent ScopedDescentTree cannot mix routes inside one descent
    const bool tree = DescentTreeEnabled();
    const double c0 = ProcessCpuSeconds();
    double cp = c0;
    auto cpu_lap = [&cp]() {
        const double c = ProcessCpuSeconds();
        const double d = c - cp;
        cp = c;
        return d;
    };
    ENGINE_CHECK_MSG(R.good_prime >= 5 && R.good_prime > n + 1 && IsPrime(R.good_prime),
                     "AssembleDescent: the good prime must be >= 5, > n + 1 ([C, prop:certifiede](4))");
    ENGINE_CHECK_MSG(opt.q_top_two >= 4 && opt.q_top_two <= Iso2().q_top(), "AssembleDescent: q_top_two in 4 .. 6");
    ENGINE_CHECK_MSG(opt.q_top_three >= 6 && opt.q_top_three <= Iso3().q_top(), "AssembleDescent: q_top_three = 6");
    R.H.assign(6, AbelianGroupType{});

    // (1) the good prime
    auto t = clk::now();
    // (PAR-L1) even n: the shipped sparse cokernel of Gamma_0(1) (the record's good-prime step), checked against the closed form
    R.rational_dims = GoodPrimeFieldDimsRoute(n, R.good_prime, &R.good_route);
    R.good_sparse = R.good_route != GoodPrimeRoute::Shipped;
    ENGINE_CHECK_MSG(R.rational_dims[2] == 0, "AssembleDescent: H^2(Gamma; Sym^n Q) != 0 at the good prime");
    R.t_good = since(t);
    R.cpu_good = cpu_lap();
    // H^0(Gamma; Sym^n Q) = 0 for n > 0, Q for n = 0 (the closed form; referee F1: d^0's rank prime qualifies iff the
    // complex's dim H^0 over F_ell' equals it, which RunLocalPiece's rank cross-check enforces).
    ENGINE_CHECK_MSG(R.rational_dims[0] == (n == 0 ? 1u : 0u), "AssembleDescent: H^0 at the good prime != the closed form");
    std::vector<uint32_t> h(8, 0);
    h[0] = R.rational_dims[0];
    h[1] = R.rational_dims[1];
    R.H[0].free_rank = h[0];
    R.H[1].free_rank = h[1];

    // (2) the bad primes ell >= 5
    t = clk::now();
    R.dickson = integral_model::DicksonPrimes(n);
    // (D5) at ell >= 5 the Bredon tree ([C, prop:isomodeltree](2)(c); measured 18-26x faster than the direct Smith of C_eq at
    // n = 100 .. 300, DESCENT5_report.md) unless HECKE_DESCENT_TREE=0; verify_d1 (the d^1-onto test on C_eq) keeps the direct Smith.
    const bool tree5 = opt.bad_route == BadPrimeRoute::DirectSmith && !opt.verify_d1 && tree;
    auto run_bad = [&](uint32_t ell, uint32_t count) {
        return opt.bad_route == BadPrimeRoute::KMode ? RunShippedKMode(n, ell, count, opt.k_search)
               : tree5                              ? RunTreeBredonLevel1(n, ell, count)
                                                     : RunDirectSmithLevel1(n, ell, count, opt.verify_d1);
    };
    // (PAR-ASM (b)) the Bredon-tree primes are independent: run them concurrently, at most JointWorkers at a time under the joint
    // RSS prediction (rule 19), each into its own slot; consumed below in the serial order.  Lever HECKE_DESCENT_PAR_ELL=0: one after
    // another (verbatim).  The KMode and direct routes (the engine tape; the measurement arms) stay serial.
    std::vector<BadPrimeRun> runs(R.dickson.size());
    R.ell_rss_pred_one = PredictTreeBredonRss(n + 1);
    R.ell_workers = 1;
    if (tree5 && par::Enabled(par::Lever::Ell) && !par::DeviceRequested() && R.dickson.size() > 1) {
        const uint32_t T = par::Threads();
        R.ell_workers = par::JointWorkers(static_cast<uint32_t>(R.dickson.size()), R.ell_rss_pred_one, opt.rss_cap_bytes, T);
        par::NoteEll(R.ell_workers > 1, R.ell_workers < std::min<uint32_t>(T, static_cast<uint32_t>(R.dickson.size())));
    }
    R.ell_rss_pred_joint = R.ell_rss_pred_one * R.ell_workers;
    {
        RssSampler rs;
        const uint64_t base = CurrentRssBytes();
        par::ForEachIndex(R.dickson.size(), R.ell_workers, [&](size_t i) { runs[i] = run_bad(R.dickson[i].first, R.dickson[i].second); });
        const uint64_t peak = rs.stop();
        R.ell_rss_growth = peak > base ? peak - base : 0;
    }
    for (BadPrimeRun& b : runs) {
        if (b.certified)
            for (const auto& [p, exps] : b.tors1.torsion)
                for (uint32_t a : exps) R.H[1].add_torsion(p, a);
        R.bad.push_back(std::move(b));
    }
    R.t_bad = since(t);
    R.cpu_bad = cpu_lap();

    // (3) Iso^(3) on ring3k
    t = clk::now();
    // (D5) the tree model Iso^(3)_T ([C, prop:isomodeltree](2)(b), (5)) unless HECKE_DESCENT_TREE=0 (the dense rollback)
    R.three = tree ? RunTreeLocalPieceSym(3, n, opt.q_top_three, h, R.good_prime, opt.K_three)
                   : RunLocalPieceSym(Iso3(), n, opt.q_top_three, h, R.good_prime, opt.K_three);
    for (uint32_t q = 1; q <= 5; ++q)
        for (const auto& [p, exps] : R.three.tors[q].torsion)
            for (uint32_t a : exps) R.H[q].add_torsion(p, a);
    R.t_three = since(t);
    R.cpu_three = cpu_lap();

    // (4) Iso^(2) on ring2k
    t = clk::now();
    // (D5) Iso^(2)_T ([C, prop:isomodeltree](2)(a)); q_top_two > 4 (arm (P)) builds every d^q explicitly, not by (2)(d)
    R.two = tree ? RunTreeLocalPieceSym(2, n, opt.q_top_two, h, R.good_prime, 0, opt.force_wide_two, opt.q_top_two > 4)
                 : RunLocalPieceSym(Iso2(), n, opt.q_top_two, h, R.good_prime, 0, opt.force_wide_two);
    const uint32_t direct_top = std::min<uint32_t>(opt.q_top_two, 5);
    for (uint32_t q = 1; q <= direct_top; ++q)
        for (const auto& [p, exps] : R.two.tors[q].torsion)
            for (uint32_t a : exps) R.H[q].add_torsion(p, a);
    R.H5_from_periodicity = opt.q_top_two < 5;
    if (R.H5_from_periodicity) {
        // [C, lem:bulkperiodic]: H^q ~= H^{q+2} for q >= 2 (cup with e), so H^5_(2) := H^3_(2).
        for (const auto& [p, exps] : R.two.tors[3].torsion)
            for (uint32_t a : exps) R.H[5].add_torsion(p, a);
    }
    R.t_two = since(t);
    R.cpu_two = cpu_lap();
    R.t_total = since(t0);
    R.cpu_total = ProcessCpuSeconds() - c0;
    // (D0 audit M3) the refusal: any uncertified piece, or a Dickson count off prop:badprimes(1), and H is cleared.
    if (!R.certified()) {
        std::ostringstream why;
        if (!R.two.all_certified()) why << (tree ? " Iso^(2)_T (tree) not certified;" : " Iso^(2) not certified;");
        if (!R.three.all_certified())
            why << (tree ? " Iso^(3)_T (tree) not certified (K = " : " Iso^(3) not certified (K = ") << R.three.K << ");";
        for (const auto& b : R.bad) {
            const auto it = b.tors1.torsion.find(b.ell);
            const uint32_t s = it == b.tors1.torsion.end() ? 0u : static_cast<uint32_t>(it->second.size());
            if (!b.certified) why << (opt.bad_route == BadPrimeRoute::KMode ? " C_7 K-mode at " : (tree && !opt.verify_d1 ? " Bredon tree at " : " C_eq direct Smith at "))
                                  << b.ell << " not certified within the budget;";
            else if (!b.torsion_only_in_H1) why << (opt.bad_route == BadPrimeRoute::KMode ? " C_7 K-mode at " : (tree && !opt.verify_d1 ? " Bredon tree at " : " C_eq direct Smith at "))
                                                << b.ell << " found torsion outside H^1 (d^1 not onto);";
            else if (s != b.count) why << (opt.bad_route == BadPrimeRoute::KMode ? " C_7 K-mode at " : (tree && !opt.verify_d1 ? " Bredon tree at " : " C_eq direct Smith at ")) << b.ell
                                       << ": " << s << " summands != prop:badprimes(1) count " << b.count << ";";
        }
        R.refused = true;
        R.refusal = "REFUSED n=" + std::to_string(n) + ":" + why.str();
        R.H.clear();
    }
    return R;
}

// ---------------------------------------------------------------------------
// the comparators
// ---------------------------------------------------------------------------
ComparatorRun RunComparatorA(uint32_t n, uint32_t good_prime) {
    const auto& desc = Descriptor3();
    ComparatorRun out;
    out.route = "a";
    const auto t0 = clk::now();
    const double c0 = ProcessCpuSeconds();
    std::vector<uint32_t> primes = {2, 3};
    for (const auto& pr : integral_model::DicksonPrimes(n)) primes.push_back(pr.first);
    const integral_model::ModelIntegralCohomology G = integral_model::AssembleFromModel(desc, n, primes, good_prime);
    out.seconds = since(t0);
    out.cpu_seconds = ProcessCpuSeconds() - c0;
    out.H = G.H;
    out.H.push_back(G.tors_H_top);   // tors H^4
    out.certified = G.all_certified();
    return out;
}

namespace {
DenseBuf<uint32_t> mat_mul(const U32Ctx& ctx, const DenseBuf<uint32_t>& A, const DenseBuf<uint32_t>& B) {
    DenseBuf<uint32_t> C(A.rows, B.cols);
    engine::kernels::gemm<uint32_t>(ctx, engine::cview(A), engine::cview(B), engine::view(C));
    return C;
}
DenseBuf<uint32_t> identity(uint32_t D) {
    DenseBuf<uint32_t> I(D, D);
    for (uint32_t i = 0; i < D; ++i) I.at(i, i) = 1;
    return I;
}
// sum_{i<k} G^i
DenseBuf<uint32_t> psum(const U32Ctx& ctx, const DenseBuf<uint32_t>& G, uint32_t k) {
    const uint32_t D = G.rows;
    DenseBuf<uint32_t> S(D, D), P = identity(D);
    for (uint32_t i = 0; i < k; ++i) {
        for (uint32_t a = 0; a < D; ++a)
            for (uint32_t b = 0; b < D; ++b) S.at(a, b) = ctx.add(S.at(a, b), P.at(a, b));
        if (i + 1 < k) P = mat_mul(ctx, P, G);
    }
    return S;
}
DenseBuf<uint32_t> minus_one(const U32Ctx& ctx, const DenseBuf<uint32_t>& G) {
    DenseBuf<uint32_t> out = G;
    for (uint32_t i = 0; i < G.rows; ++i) out.at(i, i) = ctx.sub(out.at(i, i), 1u);
    return out;
}
void put(const U32Ctx& ctx, DenseBuf<uint32_t>& T, uint32_t r0, uint32_t c0, const DenseBuf<uint32_t>& B, bool negate) {
    for (uint32_t i = 0; i < B.rows; ++i)
        for (uint32_t j = 0; j < B.cols; ++j) T.at(r0 + i, c0 + j) = negate ? ctx.neg(B.at(i, j)) : B.at(i, j);
}
}  // namespace

std::vector<DenseBuf<uint32_t>> TreeDifferentials(const U32Ctx& ctx, const CoefficientModule<uint32_t>& M,
                                                  uint32_t q_top) {
    const uint32_t D = M.dim;
    const DenseBuf<uint32_t> rS = M.rho(hecke::Mat22::S()), rU = M.rho(hecke::Mat22::ST()),
                             rN = M.rho(hecke::Mat22::I().neg());
    const DenseBuf<uint32_t> I = identity(D);
    // dcyc(G, order, q): G - 1 at even q, the norm at odd q; res(G, j, q): 1 at even q, sum_{i<j} G^i at odd q.
    const DenseBuf<uint32_t> dS0 = minus_one(ctx, rS), dS1 = psum(ctx, rS, 4), dU0 = minus_one(ctx, rU),
                             dU1 = psum(ctx, rU, 6), dN0 = minus_one(ctx, rN), dN1 = psum(ctx, rN, 2),
                             resS1 = psum(ctx, rS, 2), resU1 = psum(ctx, rU, 3);
    std::vector<DenseBuf<uint32_t>> out;
    for (uint32_t q = 0; q < q_top; ++q) {
        const bool ev = q % 2 == 0;
        const uint32_t cols = q == 0 ? 2 * D : 3 * D;
        DenseBuf<uint32_t> T(3 * D, cols);
        put(ctx, T, 0, 0, ev ? dS0 : dS1, false);
        put(ctx, T, D, D, ev ? dU0 : dU1, false);
        put(ctx, T, 2 * D, 0, ev ? I : resS1, false);
        put(ctx, T, 2 * D, D, ev ? I : resU1, true);
        if (q >= 1) put(ctx, T, 2 * D, 2 * D, ((q - 1) % 2 == 0) ? dN0 : dN1, true);
        out.push_back(std::move(T));
    }
    return out;
}

bool TorsionFromDifferentials(const std::vector<DenseBuf<uint32_t>>& Dl, const U32Ctx& ctx,
                              const std::vector<uint32_t>& rank_Q, std::vector<AbelianGroupType>& tors_out) {
    ENGINE_CHECK(rank_Q.size() == Dl.size());
    tors_out.assign(Dl.size() + 1, AbelianGroupType{});
    bool all = true;
    for (uint32_t q = 0; q < Dl.size(); ++q) {
        const std::vector<uint32_t> sigma = smith_valuations(ctx, Dl[q]);
        ENGINE_CHECK_MSG(sigma.size() <= rank_Q[q], "TorsionFromDifferentials: r_K > rank_Q");
        if (sigma.size() != rank_Q[q]) {
            all = false;
            continue;
        }
        for (uint32_t s : sigma)
            if (s >= 1) tors_out[q + 1].add_torsion(ctx.ell, s);
    }
    return all;
}

ComparatorRun RunComparatorTree(uint32_t n, uint32_t good_prime, uint32_t q_top) {
    ComparatorRun out;
    out.route = "c";
    const auto t0 = clk::now();
    const double c0 = ProcessCpuSeconds();
    const U32Ctx G(good_prime);
    const auto DG = TreeDifferentials(G, integral_model::SymModule(G, n), q_top);
    std::vector<uint32_t> rank_Q;
    for (const auto& Dq : DG) {
        DenseBuf<uint32_t> W = Dq;
        rank_Q.push_back(engine::elim::rref(G, engine::view(W)).rank);
    }
    const uint32_t d = n + 1;
    out.H.assign(q_top + 1, AbelianGroupType{});
    for (uint32_t q = 0; q < q_top; ++q) {
        const uint32_t c = (q == 0 ? 2 : 3) * d;
        out.H[q].free_rank = c - rank_Q[q] - (q ? rank_Q[q - 1] : 0u);
    }
    std::vector<uint32_t> primes = {2, 3};
    for (const auto& pr : integral_model::DicksonPrimes(n)) primes.push_back(pr.first);
    bool all = true;
    for (uint32_t ell : primes) {
        std::vector<AbelianGroupType> tors;
        bool ok = false;
        if (ell <= 3) {
            const U32Ctx C = U32Ctx::prime_power(ell, LocalWidth(ell));
            ok = TorsionFromDifferentials(TreeDifferentials(C, integral_model::SymModule(C, n), q_top), C, rank_Q, tors);
        } else {
            for (uint32_t K = 2; K <= integral_model::DefaultKBudget(ell) && !ok; ++K) {
                const U32Ctx C = U32Ctx::prime_power(ell, K);
                ok = TorsionFromDifferentials(TreeDifferentials(C, integral_model::SymModule(C, n), q_top), C, rank_Q,
                                              tors);
            }
        }
        all = all && ok;
        for (uint32_t q = 1; q <= q_top; ++q)
            for (const auto& [p, exps] : tors[q].torsion)
                for (uint32_t a : exps) out.H[q].add_torsion(p, a);
    }
    out.certified = all;
    out.seconds = since(t0);
    out.cpu_seconds = ProcessCpuSeconds() - c0;
    return out;
}

ComparatorRun RunComparatorTreeBredon(uint32_t n, uint32_t good_prime) {
    ComparatorRun out;
    out.route = "c'";
    out.H.assign(2, AbelianGroupType{});
    out.certified = true;
    if (n % 2 == 1) return out;   // M^{-I} = 0: the Z[1/6] piece is zero
    const auto t0 = clk::now();
    const double cpu0 = ProcessCpuSeconds();
    // delta^0 : M^S (+) M^U -> M^{-I} = M, (a, b) |-> a - b, columns = bases of the images of the averaging idempotents.
    auto delta0 = [n](const U32Ctx& ctx) {
        const auto M = integral_model::SymModule(ctx, n);
        const uint32_t D = M.dim;
        const DenseBuf<uint32_t> rS = M.rho(hecke::Mat22::S()), rU = M.rho(hecke::Mat22::ST());
        const uint32_t inv4 = ctx.inv(4u % ctx.p), inv6 = ctx.inv(6u % ctx.p);
        DenseBuf<uint32_t> PS = psum(ctx, rS, 4), PU = psum(ctx, rU, 6);
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) {
                PS.at(i, j) = ctx.mul(PS.at(i, j), inv4);
                PU.at(i, j) = ctx.mul(PU.at(i, j), inv6);
            }
        const U32Ctx F(ctx.ell);
        auto basis_cols = [&](const DenseBuf<uint32_t>& P) {
            DenseBuf<uint32_t> Pf(D, D);
            for (uint32_t i = 0; i < D; ++i)
                for (uint32_t j = 0; j < D; ++j) Pf.at(i, j) = P.at(i, j) % ctx.ell;
            return engine::elim::rref(F, engine::view(Pf)).pivots;
        };
        const auto JS = basis_cols(PS), JU = basis_cols(PU);
        DenseBuf<uint32_t> A(D, static_cast<uint32_t>(JS.size() + JU.size()));
        for (uint32_t i = 0; i < D; ++i) {
            for (uint32_t a = 0; a < JS.size(); ++a) A.at(i, a) = PS.at(i, JS[a]);
            for (uint32_t b = 0; b < JU.size(); ++b) A.at(i, JS.size() + b) = ctx.neg(PU.at(i, JU[b]));
        }
        return std::make_pair(A, static_cast<uint32_t>(JS.size() + JU.size()));
    };
    const U32Ctx G(good_prime);
    auto [AG, c0] = delta0(G);
    DenseBuf<uint32_t> W = AG;
    const uint32_t r = engine::elim::rref(G, engine::view(W)).rank;
    out.H[0].free_rank = c0 - r;
    out.H[1].free_rank = (n + 1) - r;
    for (const auto& [ell, count] : integral_model::DicksonPrimes(n)) {
        (void)count;
        bool ok = false;
        for (uint32_t K = 2; K <= integral_model::DefaultKBudget(ell) && !ok; ++K) {
            const U32Ctx C = U32Ctx::prime_power(ell, K);
            std::vector<AbelianGroupType> tors;
            ok = TorsionFromDifferentials({delta0(C).first}, C, {r}, tors);
            if (ok)
                for (const auto& [p, exps] : tors[1].torsion)
                    for (uint32_t a : exps) out.H[1].add_torsion(p, a);
        }
        out.certified = out.certified && ok;
    }
    out.seconds = since(t0);
    out.cpu_seconds = ProcessCpuSeconds() - cpu0;
    return out;
}

std::vector<uint32_t> TreeBredonFieldDims(uint32_t n, uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 5 && IsPrime(ell) && n % 2 == 0, "TreeBredonFieldDims: ell >= 5 prime, n even");
    const U32Ctx F(ell);
    const auto M = integral_model::SymModule(F, n);
    const uint32_t D = M.dim;
    const DenseBuf<uint32_t> rS = M.rho(hecke::Mat22::S()), rU = M.rho(hecke::Mat22::ST());
    DenseBuf<uint32_t> PS = psum(F, rS, 4), PU = psum(F, rU, 6);
    const uint32_t inv4 = F.inv(4u % ell), inv6 = F.inv(6u % ell);
    for (uint32_t i = 0; i < D; ++i)
        for (uint32_t j = 0; j < D; ++j) {
            PS.at(i, j) = F.mul(PS.at(i, j), inv4);
            PU.at(i, j) = F.mul(PU.at(i, j), inv6);
        }
    DenseBuf<uint32_t> WS = PS, WU = PU;
    const auto JS = engine::elim::rref(F, engine::view(WS)).pivots, JU = engine::elim::rref(F, engine::view(WU)).pivots;
    DenseBuf<uint32_t> A(D, static_cast<uint32_t>(JS.size() + JU.size()));
    for (uint32_t i = 0; i < D; ++i) {
        for (uint32_t a = 0; a < JS.size(); ++a) A.at(i, a) = PS.at(i, JS[a]);
        for (uint32_t b = 0; b < JU.size(); ++b) A.at(i, JS.size() + b) = F.neg(PU.at(i, JU[b]));
    }
    const uint32_t c0 = A.cols;
    const uint32_t r = c0 ? engine::elim::rref(F, engine::view(A)).rank : 0u;
    return {c0 - r, D - r, 0u};
}

// ---------------------------------------------------------------------------
// the level-N probe
// ---------------------------------------------------------------------------
CoefficientModule<uint32_t> InducedSymModule(uint32_t N, uint32_t n, const U32Ctx& ctx) {
    auto A = std::make_shared<const hecke::level::Gamma0LevelN>(N);
    CoefficientModule<uint32_t> M;
    M.name = "Ind_{Gamma_0(" + std::to_string(N) + ")}^Gamma Sym^" + std::to_string(n);
    M.dim = A->m() * (n + 1);
    auto cache = std::make_shared<std::map<integral_model::GKey, DenseBuf<uint32_t>>>();
    const uint32_t D = M.dim;
    M.rho = [A, n, ctx, cache, D](const hecke::Mat22& g) {
        const integral_model::GKey k = integral_model::key_of(g);
        auto it = cache->find(k);
        if (it != cache->end()) return it->second;
        const auto mono = hecke::level::MonomialOf(*A, g, n, ctx);
        const std::vector<uint32_t> dense = mono.dense();
        DenseBuf<uint32_t> R(D, D);
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) R.at(i, j) = dense[static_cast<size_t>(i) * D + j];
        cache->emplace(k, R);
        return R;
    };
    return M;
}

uint32_t PsiIndex(uint32_t N) {
    uint64_t psi = N;
    uint32_t m = N;
    for (uint32_t p = 2; p * p <= m; ++p)
        if (m % p == 0) {
            psi = psi / p * (p + 1);
            while (m % p == 0) m /= p;
        }
    if (m > 1) psi = psi / m * (m + 1);
    return static_cast<uint32_t>(psi);
}

uint64_t PredictLocalRss(uint32_t ell, uint32_t D, uint32_t q_top) {
    // cells per degree: 5, 14, 19, 19, ... (Iso^(2) keeps every column; Iso^(3)'s full layout has the same cells in
    // degrees <= 2 and one cell above).  Peak = the largest full coboundary (uint32_t) + the reduced working copy and
    // the engine's working copy (each <= the full size at 2, 1/2 of it at 3) + the rho memo (~48 blocks) + 0.2 GiB.
    const uint64_t w[] = {5, 14, 19, 19, 19, 19, 19};
    const uint64_t w3[] = {5, 14, 19, 1, 1, 1, 1};
    if (ell == 3) {
        // (D1 audit m6) recalibrated to measurement: D0's structural model (2 * 4 * 19 * 14 D^2 + 192 D^2 = 2320 D^2) over-predicted
        // Iso^(3) 2.4-2.5x (D0 Sec. 5; D1: 3.18 GiB at D = 1904 against 8.03).  The fitted leading term is <= 912 D^2 on every
        // measured cell (D = 84 .. 1904); 1100 D^2 + 200 MiB keeps a >= 12 % margin on all of them.
        (void)q_top;
        return 1100ull * D * D + (200ull << 20);
    }
    uint64_t peak = 0, dev = 0;
    for (uint32_t q = 0; q < q_top; ++q) {
        const uint64_t r = (ell == 2 ? w[q + 1] : w3[q + 1]) * D, c = (ell == 2 ? w[q] : w3[q]) * D;
        const uint64_t full = 4 * r * c;
        peak = std::max(peak, full + (ell == 2 ? 2 * full : full));
        // (PAR-GPU audit L2, L3) the device route's host memory for the largest 2-adic LU (0 unless ENGINE_DEVICE=gpu would take
        // it; u64 assumed: the level-1 W = 61 route is the wider rung)
        if (ell == 2) dev = std::max(dev, smith_gpu::PredictHostRssBytes(r, c, 8u));
    }
    return peak + 48ull * 4 * D * D + (200ull << 20) + dev;
}
uint64_t PredictTreeRss(uint32_t D) {
    // q_top = 4 differentials of 3D x 3D held at once (uint32_t), the ring2k copy, ~12 D x D blocks, + 0.2 GiB.
    return 4ull * (4 * 9ull * D * D + 9ull * D * D + 12ull * D * D) + (200ull << 20);
}

LevelNProbe RunLevelNProbe(uint32_t N, uint32_t n, uint64_t rss_cap_bytes) {
    LevelNProbe P;
    P.N = N;
    P.n = n;
    P.D = PsiIndex(N) * (n + 1);
    // the good prime: > n + 1, prime to 6N
    uint32_t g = 10007;
    while (!IsPrime(g) || N % g == 0) ++g;
    P.good_prime = g;
    // the shipped level-N route at the good prime (the Z[1/6] piece's rank)
    {
        RssSampler rs;
        const auto t0 = clk::now();
        const U32Ctx F(g);
        auto A = std::make_shared<const hecke::level::Gamma0LevelN>(N);
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        LocalSystemSimplicialComplex L(K, hecke::level::InducedRelationsLevel(K, *A, n, F), P.D, F.p);
        EquivariantCohomologyOnK coh(&L);
        coh.compute_cohomology();
        P.rational_dims = coh.get_H_eq_dimensions();
        P.rational_dims.resize(3, 0);
        P.t_shipped = since(t0);
        P.rss_meas_shipped = rs.stop();
    }
    // (D0 audit m3) b_0 is stated per M ([C, prop:certifiedeell](4)(b)): 0 for n > 0, 1 for n = 0 -- not read at the
    // rank prime, so that a non-qualifying prime cannot move both sides together.
    ENGINE_CHECK_MSG(P.rational_dims[0] == (n == 0 ? 1u : 0u), "RunLevelNProbe: H^0 at the good prime != the closed form b_0");
    std::vector<uint32_t> h(8, 0);
    h[0] = n == 0 ? 1u : 0u;
    h[1] = P.rational_dims[1];
    const uint32_t q_top = 4;
    // Iso^(3)
    {
        P.rss_pred_three = PredictLocalRss(3, P.D, q_top);
        RssSampler rs;
        const auto t0 = clk::now();
        const U32Ctx C = U32Ctx::prime_power(3, LocalWidth(3));
        P.three = RunLocalPiece(Iso3(), InducedSymModule(N, n, C), C, q_top, h, g, InducedSymModule(N, n, U32Ctx(g)), -1, false);
        P.t_three = since(t0);
        P.rss_meas_three = rs.stop();
    }
    // Iso^(2), only below the cap
    P.rss_pred_two = PredictLocalRss(2, P.D, q_top);
    if (P.rss_pred_two <= rss_cap_bytes) {
        RssSampler rs;
        const auto t0 = clk::now();
        const U32Ctx C = U32Ctx::prime_power(2, LocalWidth(2));
        P.two = RunLocalPiece(Iso2(), InducedSymModule(N, n, C), C, q_top, h, g, InducedSymModule(N, n, U32Ctx(g)), -1, false);
        P.t_two = since(t0);
        P.rss_meas_two = rs.stop();
        P.ran_two = true;
    }
    // the closed-form tree comparator with the Ind module, at 2 and 3
    {
        P.rss_pred_tree = PredictTreeRss(P.D);
        RssSampler rs;
        const U32Ctx G(g);
        const auto DG = TreeDifferentials(G, InducedSymModule(N, n, G), q_top);
        std::vector<uint32_t> rank_Q;
        for (const auto& Dq : DG) {
            DenseBuf<uint32_t> W = Dq;
            rank_Q.push_back(engine::elim::rref(G, engine::view(W)).rank);
        }
        bool ok = true;
        for (uint32_t ell : {2u, 3u}) {
            const auto t0 = clk::now();
            const U32Ctx C = U32Ctx::prime_power(ell, LocalWidth(ell));
            std::vector<AbelianGroupType> tors;
            ok = TorsionFromDifferentials(TreeDifferentials(C, InducedSymModule(N, n, C), q_top), C, rank_Q, tors) && ok;
            for (uint32_t q = 0; q <= q_top && q < 5; ++q) (ell == 2 ? P.tree_tors2 : P.tree_tors3)[q] = tors[q];
            (ell == 2 ? P.t_tree2 : P.t_tree3) = since(t0);
        }
        P.tree_certified = ok;
        P.rss_meas_tree = rs.stop();
    }
    bool agree = P.tree_certified && P.three.all_certified();
    for (uint32_t q = 0; q <= q_top; ++q) agree = agree && P.three.tors[q] == P.tree_tors3[q];
    if (P.ran_two) {
        agree = agree && P.two.all_certified();
        for (uint32_t q = 0; q <= q_top; ++q) agree = agree && P.two.tors[q] == P.tree_tors2[q];
    }
    P.agree = agree;
    return P;
}

}  // namespace integral_descent
