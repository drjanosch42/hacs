// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 14, WP-12) Level N on the fixed model -- see
// HeckeLevelN.h for the contract references ([C] v1.14.0 sec:levelN).
#include "HeckeLevelN.h"

#include <algorithm>
#include <map>

#include "../LocalSystemSL2Z.h" // GenerateSL2ZRelationsWithGroupData
#include "../../Engine/EngineRuntime.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "FpRank.h"

namespace hecke {
namespace level {

namespace {

using engine::scalar::gcd;
using engine::scalar::modi;
using engine::scalar::mulmod;

using u128 = unsigned __int128;

// Inverse of a modulo m (gcd(a, m) = 1 required), in [0, m).
uint64_t InvMod(uint64_t a, uint64_t m) {
    ENGINE_CHECK_MSG(m >= 1, "InvMod: modulus 0");
    uint64_t x = 0;
    ENGINE_CHECK_MSG(engine::scalar::try_invmod<uint64_t>(a, m, x), "InvMod: not a unit");
    return x;
}

// Dense row-major D x D product over ctx.
std::vector<uint32_t> DenseMul(const engine::ModCtx<uint32_t>& ctx, const std::vector<uint32_t>& A,
                               const std::vector<uint32_t>& B, uint32_t D) {
    std::vector<uint32_t> C(static_cast<size_t>(D) * D, 0);
    std::vector<uint64_t> acc(D);
    for (uint32_t i = 0; i < D; ++i) {
        std::fill(acc.begin(), acc.end(), 0);
        for (uint32_t k = 0; k < D; ++k) {
            const uint32_t a = A[static_cast<size_t>(i) * D + k];
            if (a == 0) continue;
            const uint32_t* brow = &B[static_cast<size_t>(k) * D];
            for (uint32_t j = 0; j < D; ++j)
                if (brow[j]) acc[j] = (acc[j] + static_cast<uint64_t>(a) * brow[j]) % ctx.p;
        }
        for (uint32_t j = 0; j < D; ++j) C[static_cast<size_t>(i) * D + j] = static_cast<uint32_t>(acc[j]);
    }
    return C;
}

// Keep the vectors of `in` that raise the rank over F_p (a basis of their span).
std::vector<std::vector<uint32_t>> FieldBasisOf(const std::vector<std::vector<uint32_t>>& in, uint32_t p) {
    std::vector<std::vector<uint32_t>> out, ech;
    std::vector<size_t> piv;
    for (const auto& v : in) {
        std::vector<uint32_t> w = v;
        for (size_t b = 0; b < ech.size(); ++b) {
            const uint32_t f = w[piv[b]];
            if (f == 0) continue;
            for (size_t i = 0; i < w.size(); ++i)
                if (ech[b][i]) w[i] = static_cast<uint32_t>((w[i] + p - mulmod(uint64_t{f}, uint64_t{ech[b][i]}, uint64_t{p})) % p);
        }
        size_t pr = w.size();
        for (size_t i = 0; i < w.size(); ++i) if (w[i]) { pr = i; break; }
        if (pr == w.size()) continue;
        const uint64_t iv = InvMod(w[pr], p);
        for (uint32_t& x : w) x = static_cast<uint32_t>(mulmod(uint64_t{x}, iv, uint64_t{p}));
        ech.push_back(std::move(w));
        piv.push_back(pr);
        out.push_back(v);
    }
    return out;
}

} // namespace

// ---- factorisation and counts ----------------------------------------------------
std::vector<PrimePowerFactor> FactorLevel(uint64_t N) {
    ENGINE_CHECK_MSG(N >= 1 && N < (1ull << 31), "FactorLevel: N in [1, 2^31)");
    std::vector<PrimePowerFactor> f;
    uint64_t x = N;
    for (uint64_t q = 2; q * q <= x; ++q) {
        if (x % q) continue;
        PrimePowerFactor pf;
        pf.q = q;
        pf.Q = 1;
        while (x % q == 0) { x /= q; ++pf.nu; pf.Q *= q; }
        pf.m = static_cast<uint32_t>(pf.Q + pf.Q / q);
        f.push_back(pf);
    }
    if (x > 1) f.push_back(PrimePowerFactor{x, 1, x, static_cast<uint32_t>(x + 1)});
    return f;
}

uint64_t IndexGamma0(uint64_t N) {
    uint64_t m = 1;
    for (const PrimePowerFactor& f : FactorLevel(N)) m *= f.m;
    return m;
}

uint32_t Nu2(uint64_t N) {
    uint32_t v = 1;
    for (const PrimePowerFactor& f : FactorLevel(N)) {
        if (f.q == 2) { if (f.nu >= 2) return 0; continue; }       // 1 root mod 2, none mod 4
        v *= (f.q % 4 == 1) ? 2u : 0u;
    }
    return v;
}

uint32_t Nu3(uint64_t N) {
    uint32_t v = 1;
    for (const PrimePowerFactor& f : FactorLevel(N)) {
        if (f.q == 3) { if (f.nu >= 2) return 0; continue; }       // 1 root mod 3, none mod 9
        v *= (f.q % 3 == 1) ? 2u : 0u;
    }
    return v;
}

uint64_t CuspCount(uint64_t N) {
    uint64_t c = 0;
    for (uint64_t t = 1; t <= N; ++t) {
        if (N % t) continue;
        const uint64_t g = gcd<uint64_t>(t, N / t);
        uint64_t phi = 0;
        for (uint64_t u = 1; u <= g; ++u) phi += (gcd<uint64_t>(u, g) == 1) ? 1 : 0;
        c += phi;
    }
    return c;
}

int64_t DimCuspForms(uint64_t N, uint32_t k) {
    ENGINE_CHECK_MSG(k >= 2 && k % 2 == 0, "DimCuspForms: even k >= 2");
    const int64_t m = static_cast<int64_t>(IndexGamma0(N)), e2 = Nu2(N), e3 = Nu3(N),
                  c = static_cast<int64_t>(CuspCount(N));
    const int64_t twelve_g = 12 + m - 3 * e2 - 4 * e3 - 6 * c;
    ENGINE_CHECK_MSG(twelve_g % 12 == 0 && twelve_g >= 0, "DimCuspForms: the genus formula is not an integer");
    const int64_t g = twelve_g / 12;
    if (k == 2) return g;
    const int64_t kk = k;
    return (kk - 1) * (g - 1) + (kk / 4) * e2 + (kk / 3) * e3 + (kk / 2 - 1) * c;
}

// ---- def:cosetrepsN ------------------------------------------------------------------
Gamma0LevelN::Gamma0LevelN(uint64_t N) : N_(N), fac_(FactorLevel(N)) {
    uint64_t m = 1;
    for (const PrimePowerFactor& f : fac_) m *= f.m;
    ENGINE_CHECK_MSG(m < (1ull << 31), "Gamma0LevelN: index too large");
    m_ = static_cast<uint32_t>(m);
    pair_.assign(m_ + 1, {0, 0});
    w_.assign(m_ + 1, Mat22::I());
    w_inv_.assign(m_ + 1, Mat22::I());
    case_.assign(m_ + 1, 0);
    for (uint32_t j = 1; j <= m_; ++j) {
        // (3) the canonical pair: CRT of the component pairs of the digits.
        uint64_t c = 0, d = N_ == 1 ? 1 : 0, mod = 1;
        const std::vector<uint32_t> dig = digits(j);
        for (size_t i = 0; i < fac_.size(); ++i) {
            const PrimePowerFactor& f = fac_[i];
            const uint64_t ji = dig[i];
            const uint64_t ci = ji <= f.Q ? ji - 1 : 1;
            const uint64_t di = ji <= f.Q ? 1 : f.q * (ji - f.Q - 1);
            // x = c + mod * s, s = (ci - c) mod^{-1} mod Q
            const uint64_t inv = InvMod(mod % f.Q, f.Q);
            const uint64_t sc = mulmod((ci + f.Q - c % f.Q) % f.Q, inv, f.Q);
            const uint64_t sd = mulmod((di + f.Q - d % f.Q) % f.Q, inv, f.Q);
            c += mod * sc;
            d += mod * sd;
            mod *= f.Q;
        }
        pair_[j] = {c, d};
        // (4) the representative, first applicable case.
        Mat22 w;
        if (gcd<uint64_t>(d, N_) == 1) {
            const int64_t k = static_cast<int64_t>(mulmod(c, InvMod(d, N_), N_));
            w = Mat22{1, 0, k, 1};
            case_[j] = 1;
        } else if (gcd<uint64_t>(c, N_) == 1) {
            const int64_t k = static_cast<int64_t>(mulmod(d, InvMod(c, N_), N_));
            w = Mat22{0, -1, 1, k};
            case_[j] = 2;
        } else {
            ENGINE_CHECK_MSG(fac_.size() >= 2 && c >= 2, "Gamma0LevelN: case (iii) needs a mixed class");
            uint64_t t = 0;
            while (gcd<uint64_t>(c, d + t * N_) != 1) {
                ++t;
                ENGINE_CHECK_MSG(t < c, "Gamma0LevelN: case (iii) -- no t < c_j (lem:cosetrepsN (2))");
            }
            const uint64_t dp = d + t * N_;
            const uint64_t a = InvMod(dp % c, c);
            const u128 ad = static_cast<u128>(a) * dp;
            ENGINE_CHECK_MSG((ad - 1) % c == 0, "Gamma0LevelN: case (iii) -- a d' != 1 mod c");
            const int64_t b = static_cast<int64_t>((ad - 1) / c);
            w = Mat22{static_cast<int64_t>(a), b, static_cast<int64_t>(c), static_cast<int64_t>(dp)};
            case_[j] = 3;
        }
        ENGINE_CHECK_MSG(w.det() == 1, "Gamma0LevelN: a representative is not in SL_2(Z)");
        w_[j] = w;
        w_inv_[j] = w.inv_unimodular();
        ENGINE_CHECK_MSG(index_of_pair(w.c, w.d) == j, "Gamma0LevelN: w_j's bottom row is not of class j (lem:cosetrepsN (2))");
    }
    ENGINE_CHECK_MSG(w_[1] == Mat22::I(), "Gamma0LevelN: w_1 != I");
}

std::vector<uint32_t> Gamma0LevelN::digits(uint32_t j) const {
    ENGINE_CHECK_MSG(j >= 1 && j <= m_, "Gamma0LevelN::digits: index out of range");
    std::vector<uint32_t> dg(fac_.size());
    uint32_t x = j - 1;
    for (size_t i = 0; i < fac_.size(); ++i) {
        dg[i] = 1 + x % fac_[i].m;
        x /= fac_[i].m;
    }
    return dg;
}

bool Gamma0LevelN::primitive(int64_t c, int64_t d) const {
    for (const PrimePowerFactor& f : fac_)
        if (modi(c, f.q) == 0 && modi(d, f.q) == 0) return false;
    return true;
}

uint32_t Gamma0LevelN::component_index(size_t i, int64_t c, int64_t d) const {
    const PrimePowerFactor& f = fac_[i];
    const uint64_t cq = modi(c, f.Q), dq = modi(d, f.Q);
    if (dq % f.q != 0) return static_cast<uint32_t>(1 + mulmod(cq, InvMod(dq, f.Q), f.Q));
    ENGINE_CHECK_MSG(cq % f.q != 0, "Gamma0LevelN: pair not primitive modulo a prime of N");
    const uint64_t x = mulmod(dq, InvMod(cq, f.Q), f.Q);
    ENGINE_CHECK_MSG(x % f.q == 0, "Gamma0LevelN: component index");
    return static_cast<uint32_t>(f.Q + 1 + x / f.q);
}

uint32_t Gamma0LevelN::index_of_pair(int64_t c, int64_t d) const {
    ENGINE_CHECK_MSG(primitive(c, d), "Gamma0LevelN::index_of_pair: (c, d) is not primitive modulo N");
    uint64_t j = 0, radix = 1;
    for (size_t i = 0; i < fac_.size(); ++i) {
        j += (component_index(i, c, d) - 1) * radix;
        radix *= fac_[i].m;
    }
    return static_cast<uint32_t>(j + 1);
}

std::pair<uint64_t, uint64_t> Gamma0LevelN::canonical_pair(uint32_t j) const {
    ENGINE_CHECK_MSG(j >= 1 && j <= m_, "Gamma0LevelN::canonical_pair: index out of range");
    return pair_[j];
}

uint32_t Gamma0LevelN::rep_case(uint32_t j) const {
    ENGINE_CHECK_MSG(j >= 1 && j <= m_, "Gamma0LevelN::rep_case: index out of range");
    return case_[j];
}

Mat22 Gamma0LevelN::rep(uint32_t j) const {
    ENGINE_CHECK_MSG(j >= 1 && j <= m_, "Gamma0LevelN::rep: index out of range");
    return w_[j];
}

uint32_t Gamma0LevelN::class_of(const Mat22& gamma) const {
    ENGINE_CHECK_MSG(gamma.det() == 1, "Gamma0LevelN::class_of: not in SL_2(Z)");
    return index_of_pair(gamma.c, gamma.d);
}

bool Gamma0LevelN::contains(const Mat22& gamma) const {
    return gamma.det() == 1 && modi(gamma.c, N_) == 0;
}

bool Gamma0LevelN::act_defined(uint32_t j, const Mat22& delta) const {
    const auto pr = canonical_pair(j);
    const uint64_t a = modi(delta.a, N_), b = modi(delta.b, N_), c = modi(delta.c, N_),
                   d = modi(delta.d, N_);
    const uint64_t c2 = (mulmod(pr.first, a, N_) + mulmod(pr.second, c, N_)) % N_;
    const uint64_t d2 = (mulmod(pr.first, b, N_) + mulmod(pr.second, d, N_)) % N_;
    return primitive(static_cast<int64_t>(c2), static_cast<int64_t>(d2));
}

uint32_t Gamma0LevelN::act(uint32_t j, const Mat22& delta) const {
    const auto pr = canonical_pair(j);
    const uint64_t a = modi(delta.a, N_), b = modi(delta.b, N_), c = modi(delta.c, N_),
                   d = modi(delta.d, N_);
    const uint64_t c2 = (mulmod(pr.first, a, N_) + mulmod(pr.second, c, N_)) % N_;
    const uint64_t d2 = (mulmod(pr.first, b, N_) + mulmod(pr.second, d, N_)) % N_;
    return index_of_pair(static_cast<int64_t>(c2), static_cast<int64_t>(d2));
}

Mat22 Gamma0LevelN::cocycle(uint32_t j, const Mat22& delta) const {
    return w_[j] * delta * w_inv_[act(j, delta)];
}

Mat22W Gamma0LevelN::cocycle_wide(uint32_t j, const Mat22& delta) const {
    return Mat22W::from(w_[j]) * Mat22W::from(delta) * Mat22W::from(w_inv_[act(j, delta)]);
}

bool Gamma0LevelN::contains_wide(const Mat22W& gamma) const {
    return gamma.det() == 1 && (gamma.c % static_cast<__int128>(N_)) == 0;
}

// ---- (slot-17 WP-5) the width of the coset products ----------------------------------------
namespace {
thread_local LevelNWidth t_width = LevelNWidth::kAuto;
thread_local uint64_t t_wide_products = 0;
// (slot-24 PAR-LN) the width is a scoped thread-local option: a worker of engine::tape::run_concurrently() or of the descent's
// par::ForEachIndex receives the launching thread's value (the wide-product counter stays per thread)
const bool g_width_registered = [] {
    engine::tape::register_thread_context([]() -> engine::tape::ThreadContextInstaller {
        const LevelNWidth w = t_width;
        return [w]() -> engine::tape::ThreadContextGuard {
            const LevelNWidth prev = t_width;
            t_width = w;
            return engine::tape::ThreadContextGuard(nullptr, [prev](void*) { t_width = prev; });
        };
    });
    return true;
}();
} // namespace

ScopedLevelNWidth::ScopedLevelNWidth(LevelNWidth w) : prev_(t_width) { t_width = w; }
ScopedLevelNWidth::~ScopedLevelNWidth() { t_width = prev_; }
LevelNWidth ScopedLevelNWidth::current() { return t_width; }
uint64_t LevelNWideProducts() { return t_wide_products; }
void ResetLevelNWideProducts() { t_wide_products = 0; }

Mat22 SubgroupProductForRho(const CosetLevel& A, const Mat22& X, const Mat22& Y, const Mat22& Z, uint32_t mod, bool invert,
                            const char* what) {
    if (t_width != LevelNWidth::kForceWide) {
        try {
            const Mat22 h = X * Y * Z;
            ENGINE_CHECK_MSG(A.contains(h), what);
            return invert ? h.inv_unimodular() : h;
        } catch (const HeckeOverflow&) {
            if (t_width == LevelNWidth::kForceNarrow) throw;
        }
    }
    ++t_wide_products;
    const Mat22W h = Mat22W::from(X) * Mat22W::from(Y) * Mat22W::from(Z);
    ENGINE_CHECK_MSG(A.contains_wide(h), what);
    const Mat22W v = invert ? Mat22W{h.d, -h.b, -h.c, h.a} : h;   // det h = 1 (checked): adj h = h^{-1}
    const std::array<uint32_t, 4> r = v.reduced_mod(mod);
    return Mat22{r[0], r[1], r[2], r[3]};
}

uint32_t Gamma0LevelN::digit_act(size_t i, uint32_t ji, const Mat22& delta) const {
    ENGINE_CHECK_MSG(i < fac_.size(), "Gamma0LevelN::digit_act: factor out of range");
    const PrimePowerFactor& f = fac_[i];
    const uint64_t ci = ji <= f.Q ? ji - 1 : 1;
    const uint64_t di = ji <= f.Q ? 1 : f.q * (ji - f.Q - 1);
    const uint64_t a = modi(delta.a, f.Q), b = modi(delta.b, f.Q), c = modi(delta.c, f.Q),
                   d = modi(delta.d, f.Q);
    const uint64_t c2 = (mulmod(ci, a, f.Q) + mulmod(di, c, f.Q)) % f.Q;
    const uint64_t d2 = (mulmod(ci, b, f.Q) + mulmod(di, d, f.Q)) % f.Q;
    return component_index(i, static_cast<int64_t>(c2), static_cast<int64_t>(d2));
}

std::vector<Mat22> Gamma0SubTransversalN(const Gamma0LevelN& L, const Gamma0LevelN& Lp) {
    ENGINE_CHECK_MSG(Lp.N() % L.N() == 0, "Gamma0SubTransversalN: N must divide N'");
    std::vector<Mat22> r;
    for (uint32_t j = 1; j <= Lp.m(); ++j)
        if (Lp.canonical_pair(j).first % L.N() == 0) r.push_back(Lp.rep(j));
    ENGINE_CHECK_MSG(!r.empty() && r[0] == Mat22::I(), "Gamma0SubTransversalN: the first representative is not I");
    // lem:cosetrepsN (5): every w^{(N')}_j with N | c_j lies in Gamma_0(N) (self-check at construction, S14 WP-12 review S-1).
    for (const Mat22& x : r)
        ENGINE_CHECK_MSG(L.contains(x), "Gamma0SubTransversalN: a representative is not in Gamma_0(N) (lem:cosetrepsN (5))");
    ENGINE_CHECK_MSG(static_cast<uint64_t>(r.size()) * L.m() == Lp.m(),
                     "Gamma0SubTransversalN: |r| != m_{N'} / m_N (lem:cosetrepsN (5))");
    return r;
}

// ---- the monomial action --------------------------------------------------------------
MonomialAction MonomialOf(const CosetLevel& A, const Mat22& gamma, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx) {
    ENGINE_CHECK_MSG(gamma.det() == 1, "MonomialOf: gamma must lie in SL_2(Z)");
    MonomialAction M;
    M.m = A.m();
    M.d = n_sym + 1;
    M.perm.assign(M.m + 1, 0);
    M.blk.assign(M.m + 1, 0);
    RhoCache rho(ctx, n_sym);
    for (uint32_t j = 1; j <= M.m; ++j) {
        const Mat22 x = A.rep(j) * gamma;
        const uint32_t jj = A.class_of(x);
        // (slot-17 WP-5) w_j gamma w_jj^{-1}, int64 first, __int128 on an overflow (the same block).
        const Mat22 h = SubgroupProductForRho(A, A.rep(j), gamma, A.rep(jj).inv_unimodular(), ctx.p, false,
                                              "MonomialOf: h_j(gamma) is not in the subgroup (prop:blockmatrices)");
        M.perm[j] = jj;
        M.blk[j] = rho.block_of(h, M.blocks);
    }
    return M;
}

std::vector<uint32_t> MonomialAction::apply(const engine::ModCtx<uint32_t>& ctx, const std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(x.size() == static_cast<size_t>(m) * d, "MonomialAction::apply: operand length");
    std::vector<uint32_t> y(x.size(), 0);
    for (uint32_t j = 1; j <= m; ++j) {
        const Block& B = blocks[blk[j]];
        const uint32_t* xs = &x[static_cast<size_t>(perm[j] - 1) * d];
        for (uint32_t r = 0; r < d; ++r) {
            uint64_t acc = 0;
            for (uint32_t s = 0; s < d; ++s) acc = (acc + static_cast<uint64_t>(B.dense[static_cast<size_t>(r) * d + s]) * xs[s]) % ctx.p;
            y[static_cast<size_t>(j - 1) * d + r] = static_cast<uint32_t>(acc);
        }
    }
    return y;
}

std::vector<uint32_t> MonomialAction::dense() const {
    const uint32_t D = m * d;
    std::vector<uint32_t> out(static_cast<size_t>(D) * D, 0);
    for (uint32_t j = 1; j <= m; ++j) {
        const Block& B = blocks[blk[j]];
        for (uint32_t r = 0; r < d; ++r)
            for (uint32_t s = 0; s < d; ++s)
                out[static_cast<size_t>((j - 1) * d + r) * D + (perm[j] - 1) * d + s] = B.dense[static_cast<size_t>(r) * d + s];
    }
    return out;
}

// ---- the level-A relation list -----------------------------------------------------------
std::vector<LocalSystemSimplicialComplex::SimplexRelation> InducedRelationsLevel(
    const SimplicialComplex& K, const CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx) {
    using Rel = LocalSystemSimplicialComplex::SimplexRelation;
    const uint32_t D = A.m() * (n_sym + 1);
    const uint32_t base_prime = ctx.is_field() ? ctx.p : static_cast<uint32_t>(ctx.ell);
    const std::vector<SL2ZRelationDatum> base = GenerateSL2ZRelationsWithGroupData(K, n_sym, base_prime);
    std::map<std::array<int64_t, 4>, FlatMatrixT<uint32_t>> by_gen;
    std::vector<Rel> out;
    for (const SL2ZRelationDatum& dd : base) {
        const std::array<int64_t, 4> key{dd.g_r.a, dd.g_r.b, dd.g_r.c, dd.g_r.d};
        auto it = by_gen.find(key);
        if (it == by_gen.end())
            it = by_gen.emplace(key, FlatMatrixT<uint32_t>(D, D, ctx.p, MonomialOf(A, dd.g_r, n_sym, ctx).dense())).first;
        out.push_back(Rel(dd.rel.q_dim, dd.rel.s_lhs_idx_in_Kq, dd.rel.s_rhs_idx_in_Kq, it->second, dd.rel.tag));
    }
    return out;
}

// ---- prop:levelorbit (1) ------------------------------------------------------------------
namespace {

// rho(g) over ctx as a dense d x d row-major vector (via RhoCache).
std::vector<uint32_t> RhoDense(RhoCache& rc, std::vector<Block>& store, const Mat22& g) {
    return store[rc.block_of(g, store)].dense;
}

uint32_t OrderInverse(uint32_t order, const engine::ModCtx<uint32_t>& ctx) {
    ENGINE_CHECK_MSG(gcd<uint64_t>(order, ctx.p) == 1, "OrbitWiseFixedSpace: the order of gamma is not a unit in the ring");
    return static_cast<uint32_t>(InvMod(order, ctx.p));
}

} // namespace

FixedSpace OrbitWiseFixedSpace(const CosetLevel& A, const Mat22& gamma, uint32_t order, uint32_t n_sym,
                               const engine::ModCtx<uint32_t>& ctx) {
    // The free-orbit construction closes only because gamma^o = -I acts trivially, i.e. n even. DenseFixedSpace needs no
    // such check: at odd n its averaging idempotent is simply 0 (review M-7: the asymmetry is deliberate).
    ENGINE_CHECK_MSG(n_sym % 2 == 0, "OrbitWiseFixedSpace: n even (-I must act trivially)");
    ENGINE_CHECK_MSG(order == 2 || order == 3, "OrbitWiseFixedSpace: gamma of order 2 or 3 in PSL_2(Z)");
    // (slot-22 LN23, LN23_spec L2) n = 0: M_A is a permutation module, so its fixed space is spanned by the orbit sums in every
    // characteristic -- no 1/order.  At a fixed point the averaging idempotent's column was order * order^{-1} = 1, the orbit sum:
    // the same vector, so p >= 5 is value-identical.  n > 0 keeps the averaging path and its "order a unit" refusal.
    const bool orbit_sums = n_sym == 0;
    const uint32_t inv_o = orbit_sums ? 1u : OrderInverse(order, ctx);
    const uint32_t m = A.m(), d = n_sym + 1, D = m * d;
    FixedSpace F;
    F.D = D;
    RhoCache rc(ctx, n_sym);
    std::vector<Block> store;
    std::vector<uint8_t> seen(m + 1, 0);
    std::vector<Mat22> gpow{Mat22::I()};
    for (uint32_t i = 1; i < order; ++i) gpow.push_back(gpow.back() * gamma);
    for (uint32_t j = 1; j <= m; ++j) {
        if (seen[j]) continue;
        // the orbit j, j.gamma, j.gamma^2, ...
        std::vector<uint32_t> orb{j};
        seen[j] = 1;
        for (uint32_t i = 1; i < order; ++i) {
            const uint32_t ji = A.class_of(A.rep(j) * gpow[i]);
            if (ji == j) break;
            ENGINE_CHECK_MSG(!seen[ji], "OrbitWiseFixedSpace: orbits overlap");
            seen[ji] = 1;
            orb.push_back(ji);
        }
        ++F.orbits;
        if (orb.size() == order) {
            // free orbit (mod +-I): f_{j.gamma^i} = rho(h_j(gamma^i))^{-1} f_j, no solve.
            ++F.free_orbits;
            std::vector<std::vector<uint32_t>> inv_blocks;
            for (uint32_t i = 0; i < order; ++i) {
                // h_j(gamma^i)^{-1} = w_{j.gamma^i} gamma^{-i} w_j^{-1}
                const Mat22 hinv = SubgroupProductForRho(A, A.rep(orb[i]), gpow[i].inv_unimodular(), A.rep(j).inv_unimodular(), ctx.p,
                                                         false, "OrbitWiseFixedSpace: h_j(gamma^i) not in the subgroup");
                inv_blocks.push_back(RhoDense(rc, store, hinv));
            }
            for (uint32_t r = 0; r < d; ++r) {
                std::vector<uint32_t> f(D, 0);
                for (uint32_t i = 0; i < order; ++i)
                    for (uint32_t a = 0; a < d; ++a) f[static_cast<size_t>(orb[i] - 1) * d + a] = inv_blocks[i][static_cast<size_t>(a) * d + r];
                F.span.push_back(std::move(f));
            }
        } else {
            ENGINE_CHECK_MSG(orb.size() == 1, "OrbitWiseFixedSpace: an orbit neither free nor a fixed point");
            ++F.fixed_points;
            ++F.solves;
            // V^{F_j}: the image of (1/o) sum_i rho(w_j gamma^i w_j^{-1}).
            std::vector<uint64_t> e(static_cast<size_t>(d) * d, 0);
            for (uint32_t i = 0; i < order; ++i) {
                const Mat22 h = SubgroupProductForRho(A, A.rep(j), gpow[i], A.rep(j).inv_unimodular(), ctx.p, false,
                                                      "OrbitWiseFixedSpace: stabiliser element not in the subgroup");
                const std::vector<uint32_t> R = RhoDense(rc, store, h);
                ENGINE_CHECK_MSG(!orbit_sums || (R.size() == 1 && R[0] == 1), "OrbitWiseFixedSpace: rho_0 of a stabiliser element is not 1");
                for (size_t x = 0; x < e.size(); ++x) e[x] = (e[x] + R[x]) % ctx.p;
            }
            if (orbit_sums) e.assign(1, 1);   // the orbit sum e_j (n = 0, every p)
            std::vector<std::vector<uint32_t>> cols;
            for (uint32_t c = 0; c < d; ++c) {
                std::vector<uint32_t> f(D, 0);
                bool nz = false;
                for (uint32_t a = 0; a < d; ++a) {
                    const uint32_t v = static_cast<uint32_t>(mulmod(uint64_t{e[static_cast<size_t>(a) * d + c]}, uint64_t{inv_o}, uint64_t{ctx.p}));
                    f[static_cast<size_t>(j - 1) * d + a] = v;
                    nz = nz || v != 0;
                }
                if (nz) cols.push_back(std::move(f));
            }
            if (ctx.is_field()) cols = FieldBasisOf(cols, ctx.p);
            for (auto& f : cols) F.span.push_back(std::move(f));
        }
    }
    return F;
}

FixedSpace DenseFixedSpace(const CosetLevel& A, const Mat22& gamma, uint32_t order, uint32_t n_sym,
                           const engine::ModCtx<uint32_t>& ctx) {
    ENGINE_CHECK_MSG(order == 2 || order == 3, "DenseFixedSpace: gamma of order 2 or 3 in PSL_2(Z)");
    const uint32_t inv_o = OrderInverse(order, ctx);
    const MonomialAction M = MonomialOf(A, gamma, n_sym, ctx);
    const uint32_t D = M.m * M.d;
    const std::vector<uint32_t> Md = M.dense();
    std::vector<uint32_t> P(static_cast<size_t>(D) * D, 0), E(static_cast<size_t>(D) * D, 0);
    for (uint32_t i = 0; i < D; ++i) P[static_cast<size_t>(i) * D + i] = 1;
    for (uint32_t i = 0; i < order; ++i) {
        for (size_t x = 0; x < E.size(); ++x) E[x] = ctx.add(E[x], P[x]);
        if (i + 1 < order) P = DenseMul(ctx, P, Md, D);
    }
    FixedSpace F;
    F.D = D;
    std::vector<std::vector<uint32_t>> cols;
    for (uint32_t c = 0; c < D; ++c) {
        std::vector<uint32_t> f(D);
        bool nz = false;
        for (uint32_t r = 0; r < D; ++r) {
            f[r] = static_cast<uint32_t>(mulmod(uint64_t{E[static_cast<size_t>(r) * D + c]}, uint64_t{inv_o}, uint64_t{ctx.p}));
            nz = nz || f[r] != 0;
        }
        if (nz) cols.push_back(std::move(f));
    }
    F.span = ctx.is_field() ? FieldBasisOf(cols, ctx.p) : cols;
    return F;
}

uint32_t FieldRank(const std::vector<std::vector<uint32_t>>& vecs, uint32_t p) {
    if (vecs.empty()) return 0;
    std::vector<uint32_t> flat;
    for (const auto& v : vecs) flat.insert(flat.end(), v.begin(), v.end());
    return RankModPrime(flat.data(), static_cast<uint32_t>(vecs.size()), static_cast<uint32_t>(vecs[0].size()), p);
}

CokernelDims CokernelFromFixedSpaces(const FixedSpace& fS, const FixedSpace& fST, uint32_t p) {
    ENGINE_CHECK_MSG(fS.D == fST.D, "CokernelFromFixedSpaces: fibre ranks differ");
    CokernelDims c;
    c.D = fS.D;
    c.dim_S = FieldRank(fS.span, p);
    c.dim_ST = FieldRank(fST.span, p);
    std::vector<std::vector<uint32_t>> all = fS.span;
    all.insert(all.end(), fST.span.begin(), fST.span.end());
    c.dim_rel = FieldRank(all, p);
    c.h1 = c.D - c.dim_rel;
    c.h0 = c.dim_S + c.dim_ST - c.dim_rel;
    return c;
}

} // namespace level
} // namespace hecke
