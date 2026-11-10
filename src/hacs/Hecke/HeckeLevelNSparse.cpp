// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 15, WP-8) The sparse cokernel at large N, level change and degeneracy
// maps at composite N, the cost model -- see HeckeLevelNSparse.h ([C] sec:levelN, rmk:levelcost,
// prop:levelorbit, prop:heilbronn, prop:levelcorrespondence (4)(b), (c)).
#include "HeckeLevelNSparse.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <map>

#include "CoefficientRing.h"   // hecke::IsPrimeU64
#include "HeckeCosetTables.h"
#include "DenseRref.h"
#include "HeckeTauTable.h"
#include "../../Engine/EngineRuntime.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

namespace hecke {
namespace level {

namespace {

using u128 = unsigned __int128;

uint64_t PowModU(uint64_t a, uint64_t e, uint64_t p) {
    uint64_t r = 1 % p, b = a % p;
    while (e) {
        if (e & 1) r = static_cast<uint64_t>(static_cast<u128>(r) * b % p);
        b = static_cast<uint64_t>(static_cast<u128>(b) * b % p);
        e >>= 1;
    }
    return r;
}
uint32_t InvP(uint32_t a, uint32_t p) {
    ENGINE_CHECK_MSG(a % p != 0, "HeckeLevelNSparse: inverse of zero");
    return static_cast<uint32_t>(PowModU(a, p - 2, p));
}
uint64_t Gcd(uint64_t a, uint64_t b) { while (b) { const uint64_t t = a % b; a = b; b = t; } return a; }

double Seconds(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// (S24 CL-G2, B1) F_p as the field policy of the shared sparse kernel (SparseRref.h); every operand is a reduced code
// except reduce()'s.
struct FpOps {
    uint32_t p;
    uint32_t add(uint32_t a, uint32_t b) const {
        const uint64_t s = static_cast<uint64_t>(a) + b;
        return static_cast<uint32_t>(s >= p ? s - p : s);
    }
    uint32_t mul(uint32_t a, uint32_t b) const { return static_cast<uint32_t>(static_cast<uint64_t>(a) * b % p); }
    uint32_t muladd(uint32_t a, uint32_t b, uint32_t c) const {
        return static_cast<uint32_t>((a + static_cast<uint64_t>(b) * c) % p);
    }
    uint32_t neg(uint32_t a) const { return a ? p - a : 0; }
    uint32_t inv(uint32_t a) const { return InvP(a, p); }
    uint32_t reduce(uint32_t a) const { return a % p; }
    uint32_t one() const { return 1; }
};

// A dense accumulator over [0, D) with a touched list (values mod p).
class Acc : public sparse::Acc<FpOps> {
public:
    Acc(uint32_t D, uint32_t p) : sparse::Acc<FpOps>(FpOps{p}, D) {}
};

} // namespace

// ---- prop:levelorbit (1), sparse --------------------------------------------------------------
std::vector<SparseVec> OrbitWiseFixedGenerators(const CosetLevel& A, const Mat22& gamma, uint32_t order, uint32_t n_sym,
                                                const engine::ModCtx<uint32_t>& ctx) {
    ENGINE_CHECK_MSG(n_sym % 2 == 0, "OrbitWiseFixedGenerators: n even (-I must act trivially)");
    ENGINE_CHECK_MSG(order == 2 || order == 3, "OrbitWiseFixedGenerators: gamma of order 2 or 3 in PSL_2(Z)");
    // (slot-22 LN23) n = 0: the orbit sums, every p (see OrbitWiseFixedSpace); n > 0: the averaging path, order a unit.
    const bool orbit_sums = n_sym == 0;
    ENGINE_CHECK_MSG(ctx.is_field() && (orbit_sums || Gcd(order, ctx.p) == 1), "OrbitWiseFixedGenerators: a prime field, order a unit");
    const uint32_t p = ctx.p, inv_o = orbit_sums ? 1u : InvP(order, p);
    // WP-17 (WP-8 audit LOW 2): every column index (orb - 1) * d + a below is < m * d, formed in uint32.
    ENGINE_CHECK_MSG(static_cast<uint64_t>(A.m()) * (static_cast<uint64_t>(n_sym) + 1) <= UINT32_MAX,
                     "OrbitWiseFixedGenerators: D = m (n + 1) exceeds the uint32 column-index domain");
    const uint32_t m = A.m(), d = n_sym + 1;
    RhoCache rc(ctx, n_sym);
    std::vector<Block> store;
    std::vector<uint8_t> seen(m + 1, 0);
    std::vector<Mat22> gpow{Mat22::I()};
    for (uint32_t i = 1; i < order; ++i) gpow.push_back(gpow.back() * gamma);
    std::vector<SparseVec> out;
    for (uint32_t j = 1; j <= m; ++j) {
        if (seen[j]) continue;
        std::vector<uint32_t> orb{j};
        seen[j] = 1;
        for (uint32_t i = 1; i < order; ++i) {
            const uint32_t ji = A.class_of(A.rep(j) * gpow[i]);
            if (ji == j) break;
            ENGINE_CHECK_MSG(!seen[ji], "OrbitWiseFixedGenerators: orbits overlap");
            seen[ji] = 1;
            orb.push_back(ji);
        }
        if (orb.size() == order) {
            // free orbit: f_{j.gamma^i} = rho(h_j(gamma^i))^{-1} f_j (the OrbitWiseFixedSpace vectors).
            std::vector<uint32_t> blk(order);
            for (uint32_t i = 0; i < order; ++i) {
                // (slot-17 WP-5, F-1) int64 first, __int128 on an overflow (the same RhoCache key).
                const Mat22 hinv = SubgroupProductForRho(A, A.rep(orb[i]), gpow[i].inv_unimodular(), A.rep(j).inv_unimodular(), p,
                                                         false, "OrbitWiseFixedGenerators: h_j(gamma^i) not in the subgroup");
                blk[i] = rc.block_of(hinv, store);
            }
            // blocks sorted by position so that the columns come out ascending
            std::vector<uint32_t> ord(order);
            for (uint32_t i = 0; i < order; ++i) ord[i] = i;
            std::sort(ord.begin(), ord.end(), [&](uint32_t x, uint32_t y) { return orb[x] < orb[y]; });
            for (uint32_t r = 0; r < d; ++r) {
                SparseVec f;
                for (uint32_t oi : ord) {
                    const std::vector<uint32_t>& B = store[blk[oi]].dense;
                    for (uint32_t a = 0; a < d; ++a) {
                        const uint32_t v = B[static_cast<size_t>(a) * d + r];
                        if (v) { f.col.push_back((orb[oi] - 1) * d + a); f.val.push_back(v); }
                    }
                }
                if (f.nnz()) out.push_back(std::move(f));
            }
        } else {
            ENGINE_CHECK_MSG(orb.size() == 1, "OrbitWiseFixedGenerators: an orbit neither free nor a fixed point");
            std::vector<uint64_t> e(static_cast<size_t>(d) * d, 0);
            for (uint32_t i = 0; i < order; ++i) {
                const Mat22 h = SubgroupProductForRho(A, A.rep(j), gpow[i], A.rep(j).inv_unimodular(), p, false,
                                                      "OrbitWiseFixedGenerators: stabiliser element not in the subgroup");
                const std::vector<uint32_t>& R = store[rc.block_of(h, store)].dense;
                ENGINE_CHECK_MSG(!orbit_sums || (R.size() == 1 && R[0] == 1),
                                 "OrbitWiseFixedGenerators: rho_0 of a stabiliser element is not 1");
                for (size_t x = 0; x < e.size(); ++x) e[x] = (e[x] + R[x]) % p;
            }
            if (orbit_sums) e.assign(1, 1);   // the orbit sum e_j (n = 0, every p)
            for (uint32_t c = 0; c < d; ++c) {
                SparseVec f;
                for (uint32_t a = 0; a < d; ++a) {
                    const uint32_t v = static_cast<uint32_t>(e[static_cast<size_t>(a) * d + c] * inv_o % p);
                    if (v) { f.col.push_back((j - 1) * d + a); f.val.push_back(v); }
                }
                if (f.nnz()) out.push_back(std::move(f));
            }
        }
    }
    return out;
}

// ---- (slot-18 WP-5 (b)) the generator order ------------------------------------------------------------
#if OC_ENABLE_LEGACY
// (S24 CL-G2) the retired as-built order: GeneratorOrder, CurrentGeneratorOrder, ScopedGeneratorOrder, the env arm.
#include "legacy/HeckeLevelNGenOrderLegacy.inc"
#else
namespace {
// (S24 CL-G2) HECKE_LEVEL_N_GEN_ORDER in an OFF build: the as-built order is compiled out, so `as_built` is refused
// (never silently ignored -- the vacuous-arm lesson); unset or `sorted` is the one route; anything else throws, as before.
void CheckEnvGeneratorOrder() {
    static const std::string v = [] {
        const char* e = std::getenv("HECKE_LEVEL_N_GEN_ORDER");
        return e == nullptr ? std::string("sorted") : std::string(e);
    }();
    if (v == "sorted") return;
    if (v == "as_built") oc::legacy::refuse_compiled_out("HECKE_LEVEL_N_GEN_ORDER=as_built");
    ENGINE_CHECK_MSG(false, ("HECKE_LEVEL_N_GEN_ORDER='" + v + "': expected 'sorted' (the default; 'as_built' is retired, "
                             "OC_ENABLE_LEGACY=ON only)").c_str());
}
} // namespace

const char* GeneratorOrderInForceName() {
    CheckEnvGeneratorOrder();
    return "sorted";
}
#endif

uint64_t PresentationDigest(const SparseCokernel& Q) {
    uint64_t h = engine::kFnvOffset64;
    h = engine::FnvU64(h, Q.D());
    h = engine::FnvU64(h, Q.p());
    for (uint32_t c : Q.pivots()) {
        h = engine::FnvU64(h, c);
        const SparseVec& t = Q.tail_of(c);
        h = engine::FnvU64(h, t.nnz());
        for (size_t k = 0; k < t.nnz(); ++k) h = engine::FnvU64(engine::FnvU64(h, t.col[k]), t.val[k]);
    }
    h = engine::FnvU64(h, 0xffffffffull);
    for (uint32_t c : Q.free_columns()) h = engine::FnvU64(h, c);
    return h;
}

// ---- the sparse RREF -------------------------------------------------------------------------------
SparseCokernel::SparseCokernel(uint32_t D, const std::vector<SparseVec>& gen_S, const std::vector<SparseVec>& gen_ST,
                               uint32_t p)
    : D_(D), p_(p) {
    ENGINE_CHECK_MSG(p >= 5 && hecke::IsPrimeU64(p), "SparseCokernel: a prime field with 6 a unit (prop:levelorbit (2))");
    init(gen_S, gen_ST);
}

SparseCokernel::SparseCokernel(uint32_t D, const std::vector<SparseVec>& gen_S, const std::vector<SparseVec>& gen_ST,
                               uint32_t p, CurveTag)
    : D_(D), p_(p) {
    ENGINE_CHECK_MSG(p >= 2 && hecke::IsPrimeU64(p), "SparseCokernel (object=curve): a prime field");
    init(gen_S, gen_ST);
}

void SparseCokernel::init(const std::vector<SparseVec>& gen_S, const std::vector<SparseVec>& gen_ST) {
    const uint32_t D = D_, p = p_;
    const auto t0 = std::chrono::steady_clock::now();
    stats_.dim_S = SparseRank(D, gen_S, p);
    stats_.dim_ST = SparseRank(D, gen_ST, p);
    std::vector<SparseVec> all;
    all.reserve(gen_S.size() + gen_ST.size());
    all.insert(all.end(), gen_S.begin(), gen_S.end());
    all.insert(all.end(), gen_ST.begin(), gen_ST.end());
    eliminate(all);
    stats_.eliminate_s = Seconds(t0);
}

void SparseCokernel::eliminate(const std::vector<SparseVec>& gens) {
    // (slot-18 WP-5 (b)) the consumption order: the generators by their largest column, descending (stable); the result
    // is the unique RREF in any order (see the header).
    std::vector<uint32_t> seq(gens.size());
    for (uint32_t i = 0; i < seq.size(); ++i) seq[i] = i;
#if OC_ENABLE_LEGACY
    stats_.order = CurrentGeneratorOrder();
    const bool sorted = stats_.order == GeneratorOrder::kLastColumnDescending;
#else
    CheckEnvGeneratorOrder();
    const bool sorted = true;
#endif
    if (sorted) {
        auto last = [&](uint32_t i) -> int64_t { return gens[i].col.empty() ? -1 : static_cast<int64_t>(gens[i].col.back()); };
        std::stable_sort(seq.begin(), seq.end(), [&](uint32_t a, uint32_t b) { return last(a) > last(b); });
    }
    sparse::Counters st;
    sparse::Eliminate(FpOps{p_}, D_, gens, seq, "SparseCokernel", R_, st);
    stats_.generators += st.generators;
    stats_.generator_nnz += st.generator_nnz;
    stats_.dependent += st.dependent;
    stats_.final_nnz += st.final_nnz;
    stats_.peak_nnz = std::max(stats_.peak_nnz, st.peak_nnz);
    stats_.merge_ops += st.merge_ops;
    stats_.occ_entries += st.occ_entries;
}

SparseCokernel SparseCokernel::Build(const CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                                     bool keep_generators) {
    ENGINE_CHECK_MSG(ctx.is_field(), "SparseCokernel::Build: a prime field");
    // WP-17 (WP-8 audit LOW 2): D = m (n + 1) is formed in uint32 below.
    ENGINE_CHECK_MSG(static_cast<uint64_t>(A.m()) * (static_cast<uint64_t>(n_sym) + 1) <= UINT32_MAX,
                     "SparseCokernel::Build: D = m (n + 1) exceeds the uint32 column-index domain");
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<SparseVec> gS = OrbitWiseFixedGenerators(A, Mat22::S(), 2, n_sym, ctx);
    const std::vector<SparseVec> gU = OrbitWiseFixedGenerators(A, Mat22::ST(), 3, n_sym, ctx);
    const double tg = Seconds(t0);
    SparseCokernel Q(A.m() * (n_sym + 1), gS, gU, ctx.p);
    Q.stats_.generators_s = tg;
    if (keep_generators) {
        Q.gens_ = gS;
        Q.gens_.insert(Q.gens_.end(), gU.begin(), gU.end());
    }
    return Q;
}

SparseCokernel SparseCokernel::BuildCurve(const CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                                          bool keep_generators) {
    ENGINE_CHECK_MSG(n_sym == 0, "SparseCokernel::BuildCurve: object=curve is the weight-2 (n = 0) presentation only (LN23_spec L2); "
                                 "n > 0 refused");
    ENGINE_CHECK_MSG(ctx.is_field(), "SparseCokernel::BuildCurve: a prime field");
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<SparseVec> gS = OrbitWiseFixedGenerators(A, Mat22::S(), 2, 0, ctx);
    const std::vector<SparseVec> gU = OrbitWiseFixedGenerators(A, Mat22::ST(), 3, 0, ctx);
    const double tg = Seconds(t0);
    SparseCokernel Q(A.m(), gS, gU, ctx.p, CurveTag{});
    Q.stats_.generators_s = tg;
    if (keep_generators) {
        Q.gens_ = gS;
        Q.gens_.insert(Q.gens_.end(), gU.begin(), gU.end());
    }
    return Q;
}

std::vector<uint32_t> SparseCokernel::coords(const SparseVec& x) const {
    return sparse::Coords(FpOps{p_}, R_, x, "SparseCokernel");
}

std::vector<uint32_t> SparseCokernel::coords_dense(const std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(x.size() == D_, "SparseCokernel::coords_dense: operand length");
    return coords(sparse::FromDense(FpOps{p_}, x));
}

bool SparseCokernel::in_rel(const SparseVec& x) const {
    const std::vector<uint32_t> c = coords(x);
    return std::all_of(c.begin(), c.end(), [](uint32_t v) { return v == 0; });
}

uint32_t SparseRank(uint32_t D, const std::vector<SparseVec>& vecs, uint32_t p) {
    return sparse::EchelonRank(FpOps{p}, D, vecs);
}

// ---- the Heilbronn action on the sparse presentation ----------------------------------------------
namespace {
// term indices grouped by source block (1-based blocks).
std::vector<std::vector<uint32_t>> TermsBySource(const HeilbronnOperator& H) {
    std::vector<std::vector<uint32_t>> by(H.m() + 1);
    for (size_t t = 0; t < H.terms().size(); ++t) by[H.terms()[t].src].push_back(static_cast<uint32_t>(t));
    return by;
}
SparseVec ApplyWith(const HeilbronnOperator& H, const std::vector<std::vector<uint32_t>>& by, const SparseVec& x, Acc& acc) {
    const uint32_t d = H.d(), p = H.ctx().p;
    for (size_t k = 0; k < x.col.size(); ++k) {
        const uint32_t blk = x.col[k] / d + 1, r = x.col[k] % d;
        const uint64_t a = x.val[k];
        for (uint32_t ti : by[blk]) {
            const HeilbronnTerm& t = H.terms()[ti];
            const std::vector<uint32_t>& B = H.blocks()[t.block].dense;
            for (uint32_t row = 0; row < d; ++row) {
                const uint32_t b = B[static_cast<size_t>(row) * d + r];
                if (b) acc.add((t.dst - 1) * d + row, a * b % p);
            }
        }
    }
    return acc.take();
}
} // namespace

SparseVec ApplySparse(const HeilbronnOperator& H, const SparseVec& x) {
    ENGINE_CHECK_MSG(H.ctx().is_field(), "ApplySparse: a prime field");
    const auto by = TermsBySource(H);
    Acc acc(H.D(), H.ctx().p);
    return ApplyWith(H, by, x, acc);
}

HeilbronnSparseApply::HeilbronnSparseApply(const HeilbronnOperator& H) : H_(&H), by_src_(TermsBySource(H)) {
    ENGINE_CHECK_MSG(H.ctx().is_field(), "HeilbronnSparseApply: a prime field");
}

SparseVec HeilbronnSparseApply::apply(const SparseVec& x) const {
    Acc acc(H_->D(), H_->ctx().p);
    return ApplyWith(*H_, by_src_, x, acc);
}

uint32_t RelEscapesOnGenerators(const SparseCokernel& Q, const HeilbronnOperator& H) {
    ENGINE_CHECK_MSG(H.ctx().is_field() && H.ctx().p == Q.p() && H.D() == Q.D(),
                     "RelEscapesOnGenerators: operator and presentation over the same field and fibre");
    ENGINE_CHECK_MSG(!Q.generators().empty() || Q.D() == 0, "RelEscapesOnGenerators: the generators were not kept");
    const auto by = TermsBySource(H);
    Acc acc(H.D(), H.ctx().p);
    uint32_t bad = 0;
    for (const SparseVec& g : Q.generators()) bad += Q.in_rel(ApplyWith(H, by, g, acc)) ? 0 : 1;
    return bad;
}

HMat OnCokernelSparse(const SparseCokernel& Q, const HeilbronnOperator& H) {
    ENGINE_CHECK_MSG(H.ctx().is_field() && H.ctx().p == Q.p() && H.D() == Q.D(),
                     "OnCokernelSparse: operator and presentation over the same field and fibre");
    const auto by = TermsBySource(H);
    Acc acc(H.D(), H.ctx().p);
    const uint32_t h = Q.h1();
    HMat M;
    M.rows = h;
    M.cols = h;
    M.modulus = Q.p();
    M.ell = Q.p();
    M.field = true;
    M.row_exp.assign(h, 1);
    M.a.assign(static_cast<size_t>(h) * h, 0);
    for (uint32_t j = 0; j < h; ++j) {
        SparseVec e;
        e.col.push_back(Q.free_columns()[j]);
        e.val.push_back(1);
        const std::vector<uint32_t> c = Q.coords(ApplyWith(H, by, e, acc));
        for (uint32_t i = 0; i < h; ++i) M.a[static_cast<size_t>(i) * h + j] = c[i];
    }
    return M;
}

// ---- level change -----------------------------------------------------------------------------------
LevelChangeMap::LevelChangeMap(const Gamma0LevelN& big, const Gamma0LevelN& small, LevelChangeKind kind, uint32_t n_sym,
                               const engine::ModCtx<uint32_t>& ctx)
    : kind_(kind), d_(n_sym + 1), ctx_(ctx) {
    ENGINE_CHECK_MSG(small.N() % big.N() == 0, "LevelChangeMap: M must divide N (Gamma_0(N) <= Gamma_0(M))");
    const uint32_t mN = small.m();
    src_m_ = kind == LevelChangeKind::kRestriction ? big.m() : mN;
    dst_m_ = kind == LevelChangeKind::kRestriction ? mN : big.m();
    fibre_.assign(mN + 1, 0);
    blk_.assign(mN + 1, 0);
    RhoCache rc(ctx, n_sym);
    std::vector<uint32_t> count(big.m() + 1, 0);
    for (uint32_t i = 1; i <= mN; ++i) {
        const Mat22 v = small.rep(i);
        const uint32_t a = big.class_of(v);
        // v_i w_a^{-1} in Gamma_0(M) (its inverse for the trace); (slot-17 WP-5) int64 first, __int128 on an overflow.
        const Mat22 x = SubgroupProductForRho(big, v, big.rep(a).inv_unimodular(), Mat22::I(), ctx.p,
                                              kind != LevelChangeKind::kRestriction,
                                              "LevelChangeMap: v_i w_{a(i)}^{-1} not in Gamma_0(M)");
        fibre_[i] = a;
        ++count[a];
        blk_[i] = rc.block_of(x, blocks_);
    }
    const uint64_t index = mN / big.m();
    for (uint32_t a = 1; a <= big.m(); ++a)
        ENGINE_CHECK_MSG(count[a] == index, "LevelChangeMap: the fibre map P^1(Z/N) -> P^1(Z/M) is not [Gamma_0(M):Gamma_0(N)]-to-one");
}

std::vector<uint32_t> LevelChangeMap::apply(const std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(x.size() == src_dim(), "LevelChangeMap::apply: operand length");
    std::vector<uint64_t> y(dst_dim(), 0);
    const uint64_t p = ctx_.p;
    for (uint32_t i = 1; i < fibre_.size(); ++i) {
        const uint32_t a = fibre_[i];
        const std::vector<uint32_t>& B = blocks_[blk_[i]].dense;
        const uint32_t sblk = kind_ == LevelChangeKind::kRestriction ? a : i;
        const uint32_t dblk = kind_ == LevelChangeKind::kRestriction ? i : a;
        for (uint32_t r = 0; r < d_; ++r) {
            uint64_t s = 0;
            for (uint32_t c = 0; c < d_; ++c) s = (s + static_cast<uint64_t>(B[static_cast<size_t>(r) * d_ + c]) * x[(sblk - 1) * d_ + c]) % p;
            uint64_t& o = y[(dblk - 1) * d_ + r];
            o = (o + s) % p;
        }
    }
    return std::vector<uint32_t>(y.begin(), y.end());
}

SparseVec LevelChangeMap::apply_sparse(const SparseVec& x) const {
    std::vector<uint32_t> dense(src_dim(), 0);
    for (size_t k = 0; k < x.col.size(); ++k) dense[x.col[k]] = x.val[k];
    const std::vector<uint32_t> y = apply(dense);
    SparseVec out;
    for (uint32_t c = 0; c < y.size(); ++c)
        if (y[c]) { out.col.push_back(c); out.val.push_back(y[c]); }
    return out;
}

std::vector<uint32_t> LevelChangeMap::dense() const {
    const uint32_t R = dst_dim(), C = src_dim();
    std::vector<uint32_t> out(static_cast<size_t>(R) * C, 0);
    std::vector<uint32_t> e(C, 0);
    for (uint32_t c = 0; c < C; ++c) {
        e[c] = 1;
        const std::vector<uint32_t> y = apply(e);
        e[c] = 0;
        for (uint32_t r = 0; r < R; ++r) out[static_cast<size_t>(r) * C + c] = y[r];
    }
    return out;
}

HMat OnCokernelLevelChange(const SparseCokernel& src, const SparseCokernel& dst, const LevelChangeMap& F) {
    ENGINE_CHECK_MSG(src.D() == F.src_dim() && dst.D() == F.dst_dim() && src.p() == dst.p(),
                     "OnCokernelLevelChange: presentations and map disagree");
    ENGINE_CHECK_MSG(!src.generators().empty(), "OnCokernelLevelChange: the source generators were not kept");
    for (const SparseVec& g : src.generators())
        ENGINE_CHECK_MSG(dst.in_rel(F.apply_sparse(g)),
                         "OnCokernelLevelChange: the fibre map sends a relation out of Rel (not a Gamma-module map)");
    HMat M;
    M.rows = dst.h1();
    M.cols = src.h1();
    M.modulus = dst.p();
    M.ell = dst.p();
    M.field = true;
    M.row_exp.assign(M.rows, 1);
    M.a.assign(static_cast<size_t>(M.rows) * M.cols, 0);
    for (uint32_t j = 0; j < src.h1(); ++j) {
        SparseVec e;
        e.col.push_back(src.free_columns()[j]);
        e.val.push_back(1);
        const std::vector<uint32_t> c = dst.coords(F.apply_sparse(e));
        for (uint32_t i = 0; i < M.rows; ++i) M.a[static_cast<size_t>(i) * M.cols + j] = c[i];
    }
    return M;
}

// ---- degeneracy specs --------------------------------------------------------------------------------
std::vector<std::pair<uint64_t, uint32_t>> DegeneracyFactors(uint64_t d) {
    ENGINE_CHECK_MSG(d >= 1, "DegeneracyFactors: d >= 1");
    std::vector<std::pair<uint64_t, uint32_t>> f;
    for (const PrimePowerFactor& pf : FactorLevel(d)) f.emplace_back(pf.q, pf.nu);
    return f;
}

CorrespondenceSpec DegeneracySpec(const Gamma0LevelN& A, const Gamma0LevelN& B, const HeckeCosetTables& g_cosets,
                                  const HeckeTauTable& tau) {
    const uint64_t d = g_cosets.N();
    ENGINE_CHECK_MSG(B.N() % (d * A.N()) == 0, "DegeneracySpec: d M must divide N (prop:levelcorrespondence (4)(c))");
    CorrespondenceSpec s{&A, &B, &g_cosets, &tau, std::vector<Mat22>{Mat22::I()},
                         "alpha_" + std::to_string(d) + ": " + A.name() + " -> " + B.name()};
    VerifyTransversal(s, 1);
    return s;
}

CorrespondenceSpec RestrictionSpec(const Gamma0LevelN& A, const Gamma0LevelN& B) {
    ENGINE_CHECK_MSG(B.N() % A.N() == 0, "RestrictionSpec: M | N");
    CorrespondenceSpec s{&A, &B, nullptr, nullptr, std::vector<Mat22>{Mat22::I()}, "res: " + A.name() + " -> " + B.name()};
    VerifyTransversal(s, 1);
    return s;
}

CorrespondenceSpec TraceSpec(const Gamma0LevelN& A, const Gamma0LevelN& B) {
    ENGINE_CHECK_MSG(A.N() % B.N() == 0, "TraceSpec: A = Gamma_0(N), B = Gamma_0(M), M | N");
    CorrespondenceSpec s{&A, &B, nullptr, nullptr, Gamma0SubTransversalN(B, A), "cor: " + A.name() + " -> " + B.name()};
    VerifyTransversal(s, A.m() / B.m());
    return s;
}

// ---- the cost model ----------------------------------------------------------------------------------
namespace {
// Fitted on HeckeLevelNSparseValidation's ladder (docs/implementation/slot15/WP-8_report.md section 4): at weight 2
// and prime N (D = m = N + 1 from 1572 to 100004) the final off-pivot fill is 0.645 m^1.335 and the merge
// operations 0.034 m^2.14 (both within 25 % on the ladder).  At d > 1 the fill scales with d^2 (a further
// factor ~1.7) and the operations with d^3; composite N with many prime factors fills up to ~7x more and
// merges up to ~12x less.  The model is for the route choice and the memory prediction only.
constexpr double kFillC = 0.645, kFillE = 1.335, kFillD = 1.7;
constexpr double kOpsC = 0.034, kOpsE = 2.14;
constexpr double kCrossoverD = 256;
} // namespace

CokernelCost CokernelCostModel(uint64_t N, uint32_t n_sym) {
    CokernelCost c;
    c.m = IndexGamma0(N);
    c.d = n_sym + 1;
    c.D = c.m * c.d;
    const double m = static_cast<double>(c.m), d = c.d, D = static_cast<double>(c.D);
    const double nu2 = Nu2(N), nu3 = Nu3(N);
    c.gen_rows = (m - nu2) / 2 * d + nu2 * d + (m - nu3) / 3 * d + nu3 * d;
    const double rank = D - (D / 6.0);                 // dim Rel ~ 5 D / 6 (rmk:levelcost (4))
    // rmk:levelcost (1)-style dense RREF of the stacked spans: every pivot touches every row, D columns.
    c.dense_ops = c.gen_rows * rank * D;
    c.dense_bytes = c.gen_rows * D * 4.0;
    c.sparse_nnz = kFillC * std::pow(m, kFillE) * d * d * (c.d > 1 ? kFillD : 1.0);
    c.sparse_ops = kOpsC * std::pow(m, kOpsE) * d * d * d + 3.0 * d * c.gen_rows;
    // rows (8 bytes per entry) + the occurrence lists (4) + the generators (8 bytes, 3 d per row) + the
    // accumulator (13 bytes per column)
    c.sparse_bytes = 12.0 * c.sparse_nnz + 24.0 * d * c.gen_rows + 13.0 * D;
    // Below D = kCrossoverD the dense RREF is faster in wall time (both are microseconds there): the sparse route
    // pays O(D) set-up (the accumulator, the occurrence lists, two rank eliminations).  Measured on arm (S)
    // 2026-09-30: dense faster at every cell with D <= 212, sparse faster at every cell with D >= 256.
    c.route = (D < kCrossoverD || c.dense_ops <= c.sparse_ops) ? CokernelRoute::kDense : CokernelRoute::kSparse;
    return c;
}

CokernelRoute ChooseCokernelRoute(uint64_t N, uint32_t n_sym) { return CokernelCostModel(N, n_sym).route; }

std::map<uint32_t, HMat> CokernelHeckeMatrices(const Gamma0LevelN& lvl, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                                               const std::vector<uint32_t>& primes, CokernelRoute route, CokernelRoute* used,
                                               CokernelObject object) {
    ENGINE_CHECK_MSG(ctx.is_field(), "CokernelHeckeMatrices: a prime field");
    if (object == CokernelObject::kCurve) {
        // (slot-22 LN23) the curve object: n = 0 only, every p, the fixed spaces as orbit sums on either route.
        ENGINE_CHECK_MSG(n_sym == 0, "CokernelHeckeMatrices: object=curve is the weight-2 (n = 0) presentation only (LN23_spec L2); "
                                     "n > 0 refused");
        const CokernelRoute r = route == CokernelRoute::kAuto ? ChooseCokernelRoute(lvl.N(), 0) : route;
        if (used) *used = r;
        std::map<uint32_t, HMat> out;
        if (r == CokernelRoute::kDense) {
            const CokernelPresentation P = CokernelPresentation::BuildCurve(lvl, 0, ctx);
            for (uint32_t q : primes) out[q] = OnCokernel(P, HeilbronnOperator(lvl, q, 0, ctx));
        } else {
            const SparseCokernel Q = SparseCokernel::BuildCurve(lvl, 0, ctx, true);
            for (uint32_t q : primes) {
                const HeilbronnOperator H(lvl, q, 0, ctx);
                ENGINE_CHECK_MSG(RelEscapesOnGenerators(Q, H) == 0, "CokernelHeckeMatrices: H(Rel) not in Rel (prop:heilbronn (1))");
                out[q] = OnCokernelSparse(Q, H);
            }
        }
        return out;
    }
    // WP-17 (WP-8 audit LOW 3): neither route exists at p in {2, 3} -- both build the fixed spaces of S (order 2)
    // and ST (order 3) by averaging, which needs 6 a unit (prop:levelorbit (2)); the dense route's OrderInverse and
    // the sparse route's SparseCokernel would each refuse deeper down.  Refuse here, by name, for every route
    // (kAuto included), so the refusal does not depend on which route the cost model picks.
    ENGINE_CHECK_MSG(ctx.p >= 5, "CokernelHeckeMatrices: p in {2, 3} refused -- the cokernel presentation needs 6 a unit "
                                 "in F_p (prop:levelorbit (2)); no route (dense, sparse, auto) serves p < 5");
    const CokernelRoute r = route == CokernelRoute::kAuto ? ChooseCokernelRoute(lvl.N(), n_sym) : route;
    if (used) *used = r;
    std::map<uint32_t, HMat> out;
    if (r == CokernelRoute::kDense) {
        const CokernelPresentation P = CokernelPresentation::Build(lvl, n_sym, ctx);
        for (uint32_t q : primes) out[q] = OnCokernel(P, HeilbronnOperator(lvl, q, n_sym, ctx));
    } else {
        const SparseCokernel Q = SparseCokernel::Build(lvl, n_sym, ctx, true);
        for (uint32_t q : primes) {
            const HeilbronnOperator H(lvl, q, n_sym, ctx);
            ENGINE_CHECK_MSG(RelEscapesOnGenerators(Q, H) == 0, "CokernelHeckeMatrices: H(Rel) not in Rel (prop:heilbronn (1))");
            out[q] = OnCokernelSparse(Q, H);
        }
    }
    return out;
}

// ---- dense F_p helpers ----------------------------------------------------------------------------------
namespace {
// In-place RREF of a rows x cols matrix; returns the pivot columns.  (S24 CL-G2, B3 (b)) the engine's RREF (DenseRref.h).
std::vector<uint32_t> Rref(std::vector<uint32_t>& A, uint32_t rows, uint32_t cols, uint32_t p) {
    return RrefRowMajor<uint32_t>(A, rows, cols, p);
}
} // namespace

uint32_t FpRank(std::vector<uint32_t> A, uint32_t rows, uint32_t cols, uint32_t p) {
    return static_cast<uint32_t>(Rref(A, rows, cols, p).size());
}

std::vector<uint32_t> FpKernelBasis(const std::vector<uint32_t>& A0, uint32_t rows, uint32_t cols, uint32_t p, uint32_t& dim) {
    std::vector<uint32_t> A = A0;
    const std::vector<uint32_t> piv = Rref(A, rows, cols, p);
    std::vector<int32_t> is_piv(cols, -1);
    for (size_t i = 0; i < piv.size(); ++i) is_piv[piv[i]] = static_cast<int32_t>(i);
    std::vector<uint32_t> freec;
    for (uint32_t c = 0; c < cols; ++c)
        if (is_piv[c] < 0) freec.push_back(c);
    dim = static_cast<uint32_t>(freec.size());
    std::vector<uint32_t> K(static_cast<size_t>(cols) * dim, 0);
    for (uint32_t j = 0; j < dim; ++j) {
        const uint32_t f = freec[j];
        K[static_cast<size_t>(f) * dim + j] = 1;
        for (size_t i = 0; i < piv.size(); ++i) {
            const uint32_t v = A[i * cols + f];
            K[static_cast<size_t>(piv[i]) * dim + j] = v ? p - v : 0;
        }
    }
    return K;
}

// ---- (R3) on the sparse presentation ---------------------------------------------------------------------
namespace {
// C = A (r x k) * B (k x c) over F_p.
std::vector<uint32_t> MatMul(const std::vector<uint32_t>& A, const std::vector<uint32_t>& B, uint32_t r, uint32_t k, uint32_t c,
                             uint32_t p) {
    std::vector<uint64_t> acc(static_cast<size_t>(r) * c, 0);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t t = 0; t < k; ++t) {
            const uint64_t a = A[static_cast<size_t>(i) * k + t];
            if (!a) continue;
            const uint32_t* Bt = &B[static_cast<size_t>(t) * c];
            uint64_t* Ci = &acc[static_cast<size_t>(i) * c];
            for (uint32_t j = 0; j < c; ++j) Ci[j] = (Ci[j] + a * Bt[j]) % p;
        }
    return std::vector<uint32_t>(acc.begin(), acc.end());
}
// Coordinates Z (w x c) of the columns of Y (n x c) in the column basis W (n x w), W of full rank;
// ENGINE_CHECKs that Y lies in span W.
std::vector<uint32_t> CoordsIn(const std::vector<uint32_t>& W, const std::vector<uint32_t>& Y, uint32_t n, uint32_t w, uint32_t c,
                               uint32_t p) {
    // RREF of [W | Y] (n x (w + c)).
    std::vector<uint32_t> A(static_cast<size_t>(n) * (w + c));
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < w; ++j) A[static_cast<size_t>(i) * (w + c) + j] = W[static_cast<size_t>(i) * w + j];
        for (uint32_t j = 0; j < c; ++j) A[static_cast<size_t>(i) * (w + c) + w + j] = Y[static_cast<size_t>(i) * c + j];
    }
    const std::vector<uint32_t> piv = Rref(A, n, w + c, p);
    ENGINE_CHECK_MSG(piv.size() == w && (w == 0 || piv.back() == w - 1), "ReadEisensteinR3: an image leaves W (W not stable)");
    std::vector<uint32_t> Z(static_cast<size_t>(w) * c);
    for (uint32_t i = 0; i < w; ++i)
        for (uint32_t j = 0; j < c; ++j) Z[static_cast<size_t>(i) * c + j] = A[static_cast<size_t>(i) * (w + c) + w + j];
    return Z;
}
// Generalised kernel of X (w x w, over F_p): the stable kernel chain; returns a column basis (w x k).
std::vector<uint32_t> GeneralisedKernel(const std::vector<uint32_t>& X, uint32_t w, uint32_t p, uint32_t& k) {
    // ker X^{i+1} = { x : X x in ker X^i }: kernel of [X | -Y_i], projected to x.
    uint32_t kd = 0;
    std::vector<uint32_t> Y = FpKernelBasis(X, w, w, p, kd);   // w x kd
    for (uint32_t iter = 0; iter <= w; ++iter) {
        std::vector<uint32_t> A(static_cast<size_t>(w) * (w + kd), 0);
        for (uint32_t i = 0; i < w; ++i) {
            for (uint32_t j = 0; j < w; ++j) A[static_cast<size_t>(i) * (w + kd) + j] = X[static_cast<size_t>(i) * w + j];
            for (uint32_t j = 0; j < kd; ++j) {
                const uint32_t v = Y[static_cast<size_t>(i) * kd + j];
                A[static_cast<size_t>(i) * (w + kd) + w + j] = v ? p - v : 0;
            }
        }
        uint32_t kk = 0;
        const std::vector<uint32_t> Kfull = FpKernelBasis(A, w, w + kd, p, kk);
        // project to the x part and extract a column basis
        std::vector<uint32_t> P(static_cast<size_t>(w) * kk);
        for (uint32_t i = 0; i < w; ++i)
            for (uint32_t j = 0; j < kk; ++j) P[static_cast<size_t>(i) * kk + j] = Kfull[static_cast<size_t>(i) * kk + j];
        // column basis of P: RREF of P^T, rows = basis vectors
        std::vector<uint32_t> PT(static_cast<size_t>(kk) * w);
        for (uint32_t i = 0; i < w; ++i)
            for (uint32_t j = 0; j < kk; ++j) PT[static_cast<size_t>(j) * w + i] = P[static_cast<size_t>(i) * kk + j];
        const uint32_t rk = static_cast<uint32_t>(Rref(PT, kk, w, p).size());
        std::vector<uint32_t> Ynew(static_cast<size_t>(w) * rk);
        for (uint32_t j = 0; j < rk; ++j)
            for (uint32_t i = 0; i < w; ++i) Ynew[static_cast<size_t>(i) * rk + j] = PT[static_cast<size_t>(j) * w + i];
        if (rk == kd) { k = kd; return Y; }
        Y.swap(Ynew);
        kd = rk;
    }
    k = kd;
    return Y;
}
} // namespace

R3Reading ReadEisensteinR3(uint64_t N, uint64_t p) {
    ENGINE_CHECK_MSG(p >= 5 && hecke::IsPrimeU64(p) && N % p != 0, "ReadEisensteinR3: p >= 5 prime, p !| N");
    R3Reading R;
    R.N = N;
    R.p = p;
    const engine::ModCtx<uint32_t> ctx(static_cast<uint32_t>(p));
    auto t0 = std::chrono::steady_clock::now();
    const Gamma0LevelN lvl(N);
    const SparseCokernel Q = SparseCokernel::Build(lvl, 0, ctx, false);
    R.presentation_s = Seconds(t0);
    R.final_nnz = Q.stats().final_nnz;
    R.dimH1 = Q.h1();
    const uint32_t h = Q.h1(), P = static_cast<uint32_t>(p);
    t0 = std::chrono::steady_clock::now();
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> nil;   // (q, T_q - (1 + q)) on H^1
    for (uint32_t q : {2u, 3u, 5u, 7u}) {
        if (q == N) continue;
        const HeilbronnOperator H(lvl, q, 0, ctx);
        HMat M = OnCokernelSparse(Q, H);
        std::vector<uint32_t> A(M.a.begin(), M.a.end());
        const uint32_t lam = static_cast<uint32_t>((1 + q) % p);
        for (uint32_t i = 0; i < h; ++i) {
            uint32_t& x = A[static_cast<size_t>(i) * h + i];
            x = (x + P - lam) % P;
        }
        nil.emplace_back(q, std::move(A));
    }
    R.operators_s = Seconds(t0);
    t0 = std::chrono::steady_clock::now();
    // W: the joint generalised eigenspace, one operator at a time, each restricted to the previous W.
    uint32_t w = h;
    std::vector<uint32_t> Wb(static_cast<size_t>(h) * h, 0);            // h x w, the basis in H^1 coordinates
    for (uint32_t i = 0; i < h; ++i) Wb[static_cast<size_t>(i) * h + i] = 1;
    std::vector<std::vector<uint32_t>> Z;                                // the restrictions to W (w x w)
    for (auto& qa : nil) {
        const std::vector<uint32_t> img = MatMul(qa.second, Wb, h, h, w, P);                // h x w
        const std::vector<uint32_t> X = CoordsIn(Wb, img, h, w, w, P);                     // w x w
        uint32_t k = 0;
        const std::vector<uint32_t> G = GeneralisedKernel(X, w, P, k);                      // w x k
        Wb = MatMul(Wb, G, h, w, k, P);
        w = k;
    }
    for (auto& qa : nil) {
        const std::vector<uint32_t> img = MatMul(qa.second, Wb, h, h, w, P);
        Z.push_back(CoordsIn(Wb, img, h, w, w, P));                                          // W stable (checked)
    }
    R.dimW = w;
    // Loewy layers m^i W: spans of the images under every Z_q.
    R.layers.push_back(w);
    std::vector<uint32_t> cur(static_cast<size_t>(w) * w, 0);            // w x c, columns spanning the layer
    for (uint32_t i = 0; i < w; ++i) cur[static_cast<size_t>(i) * w + i] = 1;
    uint32_t c = w;
    for (int guard = 0; guard < 64 && c > 0; ++guard) {
        std::vector<uint32_t> imgs(static_cast<size_t>(w) * c * Z.size(), 0);  // w x (c * |Z|)
        const uint32_t cc = c * static_cast<uint32_t>(Z.size());
        for (size_t z = 0; z < Z.size(); ++z) {
            const std::vector<uint32_t> Im = MatMul(Z[z], cur, w, w, c, P);
            for (uint32_t i = 0; i < w; ++i)
                for (uint32_t j = 0; j < c; ++j) imgs[static_cast<size_t>(i) * cc + z * c + j] = Im[static_cast<size_t>(i) * c + j];
        }
        // column basis: RREF of the transpose
        std::vector<uint32_t> T(static_cast<size_t>(cc) * w);
        for (uint32_t i = 0; i < w; ++i)
            for (uint32_t j = 0; j < cc; ++j) T[static_cast<size_t>(j) * w + i] = imgs[static_cast<size_t>(i) * cc + j];
        const uint32_t rk = static_cast<uint32_t>(Rref(T, cc, w, P).size());
        if (rk == 0) break;
        R.layers.push_back(rk);
        std::vector<uint32_t> nxt(static_cast<size_t>(w) * rk);
        for (uint32_t j = 0; j < rk; ++j)
            for (uint32_t i = 0; i < w; ++i) nxt[static_cast<size_t>(i) * rk + j] = T[static_cast<size_t>(j) * w + i];
        cur.swap(nxt);
        c = rk;
    }
    R.loewy = static_cast<uint32_t>(R.layers.size());
    R.eisenstein_s = Seconds(t0);
    return R;
}

R3Reading ReadEisensteinR3k(uint64_t N, uint64_t p, uint32_t n_sym, uint32_t qmax) {
    ENGINE_CHECK_MSG(p >= 5 && p < (1ull << 31) && hecke::IsPrimeU64(p) && N % p != 0,
                     "ReadEisensteinR3k: p >= 5 prime (< 2^31), p !| N");
    ENGINE_CHECK_MSG(n_sym % 2 == 0, "ReadEisensteinR3k: n even (def:kappaN)");
    ENGINE_CHECK_MSG(qmax >= 2 && qmax < 1000, "ReadEisensteinR3k: qmax in [2, 1000)");
    R3Reading R;
    R.N = N;
    R.p = p;
    const engine::ModCtx<uint32_t> ctx(static_cast<uint32_t>(p));
    auto t0 = std::chrono::steady_clock::now();
    const Gamma0LevelN lvl(N);
    const SparseCokernel Q = SparseCokernel::Build(lvl, n_sym, ctx, false);
    R.presentation_s = Seconds(t0);
    R.final_nnz = Q.stats().final_nnz;
    R.dimH1 = Q.h1();
    const uint32_t h = Q.h1(), P = static_cast<uint32_t>(p);
    t0 = std::chrono::steady_clock::now();
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> nil;   // (q, T_q - (1 + q^{k-1})) on H^1
    for (uint32_t q = 2; q <= qmax; ++q) {
        if (!hecke::IsPrimeU64(q) || N % q == 0 || p % q == 0) continue;
        const HeilbronnOperator H(lvl, q, n_sym, ctx);
        HMat M = OnCokernelSparse(Q, H);
        std::vector<uint32_t> A(M.a.begin(), M.a.end());
        uint64_t qk = 1;   // q^{n+1} = q^{k-1} mod p
        for (uint32_t e = 0; e <= n_sym; ++e) qk = qk * q % p;
        const uint32_t lam = static_cast<uint32_t>((1 + qk) % p);
        for (uint32_t i = 0; i < h; ++i) {
            uint32_t& x = A[static_cast<size_t>(i) * h + i];
            x = (x + P - lam) % P;
        }
        nil.emplace_back(q, std::move(A));
    }
    ENGINE_CHECK_MSG(!nil.empty(), "ReadEisensteinR3k: no prime q <= qmax with q !| N p");
    R.operators_s = Seconds(t0);
    t0 = std::chrono::steady_clock::now();
    uint32_t w = h;
    std::vector<uint32_t> Wb(static_cast<size_t>(h) * h, 0);
    for (uint32_t i = 0; i < h; ++i) Wb[static_cast<size_t>(i) * h + i] = 1;
    std::vector<std::vector<uint32_t>> Z;
    for (auto& qa : nil) {
        const std::vector<uint32_t> img = MatMul(qa.second, Wb, h, h, w, P);
        const std::vector<uint32_t> X = CoordsIn(Wb, img, h, w, w, P);
        uint32_t k = 0;
        const std::vector<uint32_t> G = GeneralisedKernel(X, w, P, k);
        Wb = MatMul(Wb, G, h, w, k, P);
        w = k;
    }
    for (auto& qa : nil) {
        const std::vector<uint32_t> img = MatMul(qa.second, Wb, h, h, w, P);
        Z.push_back(CoordsIn(Wb, img, h, w, w, P));
    }
    R.dimW = w;
    R.layers.push_back(w);
    if (w == 0) {
        // (slot-16 WP-17, audit WP-13 F9) no generalised eigenspace: the Loewy length of the zero module is 0
        // (layers = {0}); the reading is then (L9)'s FINDING(dimW<2), never a Loewy datum.
        R.loewy = 0;
        R.eisenstein_s = Seconds(t0);
        return R;
    }
    std::vector<uint32_t> cur(static_cast<size_t>(w) * w, 0);
    for (uint32_t i = 0; i < w; ++i) cur[static_cast<size_t>(i) * w + i] = 1;
    uint32_t c = w;
    for (int guard = 0; guard < 64 && c > 0; ++guard) {
        std::vector<uint32_t> imgs(static_cast<size_t>(w) * c * Z.size(), 0);
        const uint32_t cc = c * static_cast<uint32_t>(Z.size());
        for (size_t z = 0; z < Z.size(); ++z) {
            const std::vector<uint32_t> Im = MatMul(Z[z], cur, w, w, c, P);
            for (uint32_t i = 0; i < w; ++i)
                for (uint32_t j = 0; j < c; ++j) imgs[static_cast<size_t>(i) * cc + z * c + j] = Im[static_cast<size_t>(i) * c + j];
        }
        std::vector<uint32_t> T(static_cast<size_t>(cc) * w);
        for (uint32_t i = 0; i < w; ++i)
            for (uint32_t j = 0; j < cc; ++j) T[static_cast<size_t>(j) * w + i] = imgs[static_cast<size_t>(i) * cc + j];
        const uint32_t rk = static_cast<uint32_t>(Rref(T, cc, w, P).size());
        if (rk == 0) break;
        R.layers.push_back(rk);
        std::vector<uint32_t> nxt(static_cast<size_t>(w) * rk);
        for (uint32_t j = 0; j < rk; ++j)
            for (uint32_t i = 0; i < w; ++i) nxt[static_cast<size_t>(i) * rk + j] = T[static_cast<size_t>(j) * w + i];
        cur.swap(nxt);
        c = rk;
    }
    R.loewy = static_cast<uint32_t>(R.layers.size());
    R.eisenstein_s = Seconds(t0);
    return R;
}

} // namespace level
} // namespace hecke
