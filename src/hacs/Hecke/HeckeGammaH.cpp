// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, WAKE-1) The level Gamma_H(N) -- see HeckeGammaH.h and
// docs/implementation/slot22/WAKE1_spec.md.
#include "HeckeGammaH.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "CoefficientRing.h"   // hecke::IsPrimeU64
#include "HeckeLevelNHeilbronn.h"
#include "HeckeNebentypus.h"   // neb::Gamma1SubTransversal (read-only use)
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"

namespace hecke {
namespace level {

namespace {

using engine::scalar::gcd;
using engine::scalar::modi;
using engine::scalar::mulmod;
using engine::scalar::powmod;

using u128 = unsigned __int128;

// a^{-1} mod m (gcd(a, m) = 1, m >= 1); throws on a non-unit.
uint64_t InvMod(uint64_t a, uint64_t m) {
    uint64_t x = 0;
    ENGINE_CHECK_MSG(engine::scalar::try_invmod<uint64_t>(a, m, x), "GammaHLevelN: an inverse of a non-unit");
    return x;
}
uint64_t ModW(__int128 x, uint64_t m) {
    const __int128 mm = static_cast<__int128>(m);
    __int128 r = x % mm;
    if (r < 0) r += mm;
    return static_cast<uint64_t>(r);
}
uint64_t EulerPhi(uint64_t N) {
    uint64_t phi = N, x = N;
    for (uint64_t q = 2; q * q <= x; ++q)
        if (x % q == 0) {
            while (x % q == 0) x /= q;
            phi = phi / q * (q - 1);
        }
    if (x > 1) phi = phi / x * (x - 1);
    return phi;
}

} // namespace

// ============================================================================
// GammaHLevelN
// ============================================================================
GammaHLevelN::GammaHLevelN(uint64_t N, const std::vector<uint64_t>& generators) : N_(N), lvl0_(N) { build(N, generators); }

void GammaHLevelN::build(uint64_t N, const std::vector<uint64_t>& generators) {
    ENGINE_CHECK_MSG(N >= 1 && N < (1ull << 20), "GammaHLevelN: N in [1, 2^20)");
    inH_.assign(N, 0);
    rank_.assign(N, -1);
    // H by BFS from 1 under multiplication by the generators.
    for (uint64_t g : generators) {
        ENGINE_CHECK_MSG(gcd<uint64_t>(g % N, N) == 1 || N == 1, "GammaHLevelN: a generator is not a unit mod N");
        gens_.push_back(g % N);
    }
    const uint64_t one = 1 % N;
    std::vector<uint64_t> queue{one};
    inH_[one] = 1;
    for (size_t i = 0; i < queue.size(); ++i)
        for (uint64_t g : gens_) {
            const uint64_t y = mulmod(queue[i], g, N);
            if (!inH_[y]) { inH_[y] = 1; queue.push_back(y); }
        }
    H_ = queue;
    std::sort(H_.begin(), H_.end());
    // U: the least representative of every coset u H, ascending (u ascending over the units).
    for (uint64_t u = 0; u < N; ++u) {
        if (N > 1 && gcd<uint64_t>(u, N) != 1) continue;
        if (N == 1 && u != 0) continue;
        if (rank_[u] >= 0) continue;
        const int32_t r = static_cast<int32_t>(U_.size());
        U_.push_back(u == 0 ? 1 : u);   // N = 1: the single class, named 1
        for (uint64_t h : H_) rank_[mulmod(u, h, N)] = r;
    }
    ENGINE_CHECK_MSG(!U_.empty() && U_[0] == 1, "GammaHLevelN: U[0] != 1");
    ENGINE_CHECK_MSG(static_cast<uint64_t>(lvl0_.m()) * U_.size() < (1ull << 32), "GammaHLevelN: m_0(N) [G_N : H] >= 2^32");
    m_ = static_cast<uint32_t>(lvl0_.m() * U_.size());
    for (uint64_t u : U_) sigma_.push_back(sigma(u));
    // bottom rows of w_j, and the class_of_row data.
    const auto& fac = lvl0_.factors();
    const size_t F = fac.size();
    row0_.assign(lvl0_.m() + 1, {0, 0});
    unit_coord_.assign(static_cast<size_t>(lvl0_.m()) * F, 0);
    unit_inv_.assign(static_cast<size_t>(lvl0_.m()) * F, 0);
    idem_.assign(F, 0);
    for (size_t i = 0; i < F; ++i) {
        const uint64_t M = N / fac[i].Q;
        idem_[i] = mulmod(M, InvMod(M % fac[i].Q, fac[i].Q), N);   // = 1 mod Q_i, 0 mod the others
    }
    for (uint32_t j = 1; j <= lvl0_.m(); ++j) {
        const Mat22 w = lvl0_.rep(j);
        row0_[j] = {modi(w.c, N), modi(w.d, N)};
        for (size_t i = 0; i < F; ++i) {
            const uint64_t Q = fac[i].Q, q = fac[i].q;
            const uint64_t c = row0_[j].first % Q, d = row0_[j].second % Q;
            const size_t k = static_cast<size_t>(j - 1) * F + i;
            if (c % q != 0) { unit_coord_[k] = 0; unit_inv_[k] = InvMod(c, Q); }
            else {
                ENGINE_CHECK_MSG(d % q != 0, "GammaHLevelN: a bottom row of w_j is not primitive");
                unit_coord_[k] = 1; unit_inv_[k] = InvMod(d, Q);
            }
        }
    }
    // Self-check (I1): w_1 = I, class_of(w_J) = J, class_of_row(bottom_row(J)) = J, bottom row = u (c_j, d_j).
    ENGINE_CHECK_MSG(rep(1) == Mat22::I(), "GammaHLevelN: w_1 != I");
    for (uint32_t J = 1; J <= m_; ++J) {
        const Mat22 w = rep(J);
        const auto br = bottom_row(J);
        ENGINE_CHECK_MSG(w.det() == 1 && modi(w.c, N) == br.first && modi(w.d, N) == br.second,
                         "GammaHLevelN: the bottom row of w_J is not u (c_j, d_j) mod N");
        ENGINE_CHECK_MSG(class_of(w) == J, "GammaHLevelN: class_of(w_J) != J");
        ENGINE_CHECK_MSG(class_of_row(static_cast<int64_t>(br.first), static_cast<int64_t>(br.second)) == J,
                         "GammaHLevelN: class_of_row(bottom_row(J)) != J");
    }
}

GammaHLevelN GammaHLevelN::PrimeToPart(uint64_t N, uint64_t p) {
    ENGINE_CHECK_MSG(N >= 1 && N < (1ull << 20) && p >= 2 && hecke::IsPrimeU64(p), "GammaHLevelN::PrimeToPart: N in [1, 2^20), p prime");
    // u has order prime to p iff u^{e'} = 1, e' the prime-to-p part of phi(N) (the order divides phi(N)).
    uint64_t eprime = EulerPhi(N);
    while (eprime % p == 0) eprime /= p;
    auto prime_to_p = [&](uint64_t u) { return gcd<uint64_t>(u, N) == 1 && powmod<uint64_t>(u, eprime, N) == 1 % N; };
    std::vector<uint8_t> cur(N, 0);
    std::vector<uint64_t> curv{1 % N};
    cur[1 % N] = 1;
    std::vector<uint64_t> gens;
    for (uint64_t u = 1; u < N; ++u) {
        if (!prime_to_p(u)) continue;
        if (cur[u]) continue;
        // adjoin u: cur <- cur <u> (u^k for k up to the first power already in cur)
        gens.push_back(u);
        std::vector<uint64_t> add;
        uint64_t x = u;
        while (!cur[x]) {
            for (uint64_t h : curv) {
                const uint64_t y = mulmod(h, x, N);
                if (!cur[y]) { cur[y] = 2; add.push_back(y); }
            }
            x = mulmod(x, u, N);
            if (cur[x] == 2) break;
        }
        for (uint64_t y : add) { cur[y] = 1; curv.push_back(y); }
    }
    GammaHLevelN G(N, gens);
    for (uint64_t u = 1; u < N; ++u)
        ENGINE_CHECK_MSG(G.in_H(u) == prime_to_p(u), "GammaHLevelN::PrimeToPart: the generated H is not the prime-to-p part");
    return G;
}

std::string GammaHLevelN::name() const {
    if (index_H() == 1) return "Gamma_H(" + std::to_string(N_) + "), H = G_N";
    return "Gamma_H(" + std::to_string(N_) + "), H = " + H_text();
}

std::string GammaHLevelN::H_text() const {
    std::ostringstream os;
    os << "<";
    for (size_t i = 0; i < gens_.size(); ++i) os << (i ? "," : "") << gens_[i];
    os << "> (|H| = " << H_.size() << ", [G_N : H] = " << index_H() << ")";
    return os.str();
}

bool GammaHLevelN::in_H(uint64_t u) const { return inH_[u % N_] != 0; }

uint32_t GammaHLevelN::coset_rank(uint64_t u) const {
    const int32_t r = rank_[u % N_];
    ENGINE_CHECK_MSG(r >= 0, "GammaHLevelN::coset_rank: not a unit mod N");
    return static_cast<uint32_t>(r);
}

Mat22 GammaHLevelN::sigma(uint64_t u) const {
    u %= N_;
    if (N_ <= 2 || u == 1) return Mat22::I();
    ENGINE_CHECK_MSG(gcd<uint64_t>(u, N_) == 1, "GammaHLevelN::sigma: u not a unit");
    const int64_t up = static_cast<int64_t>(u);
    const int64_t alpha = static_cast<int64_t>(InvMod(u, N_));
    const int64_t beta = (alpha * up - 1) / static_cast<int64_t>(N_);
    const Mat22 s{alpha, beta, static_cast<int64_t>(N_), up};
    ENGINE_CHECK_MSG(s.det() == 1, "GammaHLevelN::sigma: det != 1");
    return s;
}

uint32_t GammaHLevelN::index(uint32_t j, uint32_t r) const {
    ENGINE_CHECK_MSG(j >= 1 && j <= lvl0_.m() && r < index_H(), "GammaHLevelN::index: out of range");
    return (j - 1) * index_H() + r + 1;
}

Mat22 GammaHLevelN::rep(uint32_t J) const {
    ENGINE_CHECK_MSG(J >= 1 && J <= m_, "GammaHLevelN::rep: out of range");
    return sigma_[r_of(J)] * lvl0_.rep(j_of(J));
}

std::pair<uint64_t, uint64_t> GammaHLevelN::bottom_row(uint32_t J) const {
    ENGINE_CHECK_MSG(J >= 1 && J <= m_, "GammaHLevelN::bottom_row: out of range");
    const uint64_t u = U_[r_of(J)];
    const auto& r0 = row0_[j_of(J)];
    return {mulmod(u, r0.first, N_), mulmod(u, r0.second, N_)};
}

uint32_t GammaHLevelN::class_of(const Mat22& gamma) const {
    const uint32_t j = lvl0_.class_of(gamma);
    // h = gamma w_j^{-1} in Gamma_0(N); int64 first, __int128 on a HeckeOverflow (composite N: w_j has entries ~ N^2).
    try {
        const Mat22 h = gamma * lvl0_.rep(j).inv_unimodular();
        ENGINE_CHECK_MSG(lvl0_.contains(h), "GammaHLevelN::class_of: gamma w_j^{-1} not in Gamma_0(N)");
        return index(j, coset_rank(modi(h.d, N_)));
    } catch (const HeckeOverflow&) {
        const Mat22W h = Mat22W::from(gamma) * Mat22W::from(lvl0_.rep(j).inv_unimodular());
        ENGINE_CHECK_MSG(h.det() == 1 && ModW(h.c, N_) == 0, "GammaHLevelN::class_of: gamma w_j^{-1} not in Gamma_0(N) (int128)");
        return index(j, coset_rank(ModW(h.d, N_)));
    }
}

bool GammaHLevelN::contains(const Mat22& g) const {
    return g.det() == 1 && modi(g.c, N_) == 0 && in_H(modi(g.d, N_));
}

bool GammaHLevelN::contains_wide(const Mat22W& g) const {
    return g.det() == 1 && ModW(g.c, N_) == 0 && in_H(ModW(g.d, N_));
}

uint32_t GammaHLevelN::class_of_row(int64_t c, int64_t d) const {
    const uint64_t cN = modi(c, N_), dN = modi(d, N_);
    const uint32_t j = lvl0_.index_of_pair(static_cast<int64_t>(cN), static_cast<int64_t>(dN));   // checks primitivity
    const auto& fac = lvl0_.factors();
    const size_t F = fac.size();
    uint64_t lambda = 0;
    for (size_t i = 0; i < F; ++i) {
        const uint64_t Q = fac[i].Q;
        const size_t k = static_cast<size_t>(j - 1) * F + i;
        const uint64_t x = unit_coord_[k] == 0 ? cN % Q : dN % Q;
        const uint64_t li = mulmod(x, unit_inv_[k], Q);
        lambda = (lambda + mulmod(li, idem_[i], N_)) % N_;
    }
    if (N_ == 1) lambda = 0;
    ENGINE_DEBUG_CHECK_MSG(mulmod(lambda, row0_[j].first, N_) == cN && mulmod(lambda, row0_[j].second, N_) == dN,
                           "GammaHLevelN::class_of_row: (c, d) != lambda (c_j, d_j) mod N");
    return index(j, coset_rank(lambda));
}

bool GammaHLevelN::act_defined(uint32_t J, const Mat22& delta) const {
    const auto r = bottom_row(J);
    const uint64_t c2 = (mulmod(r.first, modi(delta.a, N_), N_) + mulmod(r.second, modi(delta.c, N_), N_)) % N_;
    const uint64_t d2 = (mulmod(r.first, modi(delta.b, N_), N_) + mulmod(r.second, modi(delta.d, N_), N_)) % N_;
    return lvl0_.primitive(static_cast<int64_t>(c2), static_cast<int64_t>(d2));
}

uint32_t GammaHLevelN::act(uint32_t J, const Mat22& delta) const {
    const auto r = bottom_row(J);
    const uint64_t c2 = (mulmod(r.first, modi(delta.a, N_), N_) + mulmod(r.second, modi(delta.c, N_), N_)) % N_;
    const uint64_t d2 = (mulmod(r.first, modi(delta.b, N_), N_) + mulmod(r.second, modi(delta.d, N_), N_)) % N_;
    return class_of_row(static_cast<int64_t>(c2), static_cast<int64_t>(d2));
}

std::vector<Mat22> GammaHSubTransversal(const GammaHLevelN& G, uint64_t p) {
    std::vector<Mat22> r = neb::Gamma1SubTransversal(G.N(), p);
    for (const Mat22& x : r) ENGINE_CHECK_MSG(G.contains(x), "GammaHSubTransversal: an r_t is not in Gamma_H(N)");
    return r;
}

// ============================================================================
// diamonds
// ============================================================================
namespace {

// The source class u' of <d> at (j, u) and the matrix sigma_u sigma_{u'}^{-1}: (R f)_{(j,u)} = rho(R) f_{(j,u')}.
struct DiamondData {
    std::vector<uint32_t> src;   // per J (1-based)
    std::vector<Mat22> R;
};

DiamondData MakeDiamond(const GammaHLevelN& G, uint64_t dd, bool inverse) {
    const uint64_t N = G.N();
    ENGINE_CHECK_MSG(N == 1 || gcd<uint64_t>(dd % N, N) == 1, "GammaH diamond: d must be a unit mod N");
    const uint64_t d = inverse ? InvMod(dd % N, N) : dd % N;
    const uint64_t dinv = N == 1 ? 0 : InvMod(d, N);
    DiamondData D;
    D.src.assign(G.m() + 1, 0);
    D.R.assign(G.m() + 1, Mat22::I());
    for (uint32_t J = 1; J <= G.m(); ++J) {
        const uint32_t j = G.j_of(J);
        const uint64_t u = G.coset_reps()[G.r_of(J)];
        const uint32_t r2 = N == 1 ? 0 : G.coset_rank(mulmod(dinv, u, N));
        D.src[J] = G.index(j, r2);
        D.R[J] = G.sigma(u) * G.sigma(G.coset_reps()[r2]).inv_unimodular();
        ENGINE_CHECK_MSG(G.level0().contains(D.R[J]) && (N == 1 || G.in_H(mulmod(modi(D.R[J].d, N), InvMod(d, N), N))),
                         "GammaH diamond: sigma_u sigma_{u'}^{-1} is not in Gamma_0(N) with d_R in d H");
    }
    // R is a Gamma-module map: h_J(gamma) R_{J.gamma} = R_J h_{src J}(gamma) with the same target (S and T).
    // (formed in __int128: at composite N the triple products leave int64)
    for (const Mat22& gam : {Mat22::S(), Mat22::T()}) {
        for (uint32_t J = 1; J <= G.m(); ++J) {
            const Mat22 x = G.rep(J) * gam;
            const uint32_t Jg = G.class_of(x);
            const Mat22W hJ = Mat22W::from(x) * Mat22W::from(G.rep(Jg).inv_unimodular());
            const Mat22W lhs = hJ * Mat22W::from(D.R[Jg]);
            const uint32_t Js = D.src[J];
            const Mat22 y = G.rep(Js) * gam;
            const uint32_t Jsg = G.class_of(y);
            const Mat22W rhs = Mat22W::from(D.R[J]) * (Mat22W::from(y) * Mat22W::from(G.rep(Jsg).inv_unimodular()));
            ENGINE_CHECK_MSG(D.src[Jg] == Jsg && lhs == rhs, "GammaH diamond: <d> does not commute with the Gamma-action");
        }
    }
    return D;
}

std::vector<uint32_t> RhoDense(const Mat22& g, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx) {
    RhoCache rc(ctx, n_sym);
    std::vector<Block> store;
    const uint32_t b = rc.block_of(g, store);
    return store[b].dense;
}

} // namespace

BlockCochainMap GammaHDiamondOnModel(const GammaHLevelN& G, uint64_t d, const SimplicialComplex& K, uint32_t n_sym,
                                     const engine::ModCtx<uint32_t>& ctx, uint32_t q, bool inverse) {
    const DiamondData D = MakeDiamond(G, d, inverse);
    BlockCochainMap F;
    F.ctx = ctx;
    F.q = q;
    F.N_q = K.get_num_q_simplices(q);
    F.d = n_sym + 1;
    F.m_src = F.m_dst = G.m();
    RhoCache rho(ctx, n_sym);
    std::vector<uint32_t> blk(G.m() + 1);
    for (uint32_t J = 1; J <= G.m(); ++J) blk[J] = rho.block_of(D.R[J], F.blocks);
    for (uint32_t s = 0; s < F.N_q; ++s)
        for (uint32_t J = 1; J <= G.m(); ++J) F.terms.push_back(BlockTerm{s, J, s, D.src[J], 1, blk[J]});
    return F;
}

GammaHDiamond::GammaHDiamond(const GammaHLevelN& G, uint64_t d, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx, bool inverse)
    : G_(&G), dd_(d), inverse_(inverse), d_(n_sym + 1), ctx_(ctx) {
    ENGINE_CHECK_MSG(ctx.is_field(), "GammaHDiamond: a prime field");
    const DiamondData D = MakeDiamond(G, d, inverse);
    // (R f)_J = rho(R_J) f_{src J}: the image of e_{(src J, a)} is sum_b rho(R_J)[b][a] e_{(J, b)}.
    dst_.assign(G.m() + 1, 0);
    blk_.assign(G.m() + 1, {});
    for (uint32_t J = 1; J <= G.m(); ++J) {
        ENGINE_CHECK_MSG(dst_[D.src[J]] == 0, "GammaHDiamond: <d> is not a permutation of the classes");
        dst_[D.src[J]] = J;
        blk_[D.src[J]] = RhoDense(D.R[J], n_sym, ctx);
    }
}

std::string GammaHDiamond::label() const {
    return std::string("<") + std::to_string(dd_) + (inverse_ ? ">^inv" : ">");
}

void GammaHDiamond::image(uint32_t col, std::vector<std::pair<uint32_t, uint32_t>>& out) const {
    const uint32_t J = col / d_ + 1, a = col % d_;
    const uint32_t T = dst_[J];
    const std::vector<uint32_t>& B = blk_[J];
    for (uint32_t b = 0; b < d_; ++b) {
        const uint32_t v = B[static_cast<size_t>(b) * d_ + a];
        if (v) out.push_back({(T - 1) * d_ + b, v});
    }
}

// ============================================================================
// Heilbronn--Merel on M
// ============================================================================
GammaHHeilbronn::GammaHHeilbronn(const GammaHLevelN& G, uint64_t n, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx)
    : G_(&G), n_(n), d_(n_sym + 1), ctx_(ctx) {
    ENGINE_CHECK_MSG(ctx.is_field(), "GammaHHeilbronn: a prime field");
    ENGINE_CHECK_MSG(n >= 1 && n < (1ull << 20), "GammaHHeilbronn: n in [1, 2^20)");
    mer_ = MerelSet(n);
}

std::string GammaHHeilbronn::label() const {
    return std::string(gcd<uint64_t>(n_, G_->N()) == 1 ? "T_" : "U_") + std::to_string(n_);
}

std::vector<uint32_t> GammaHHeilbronn::rho_block(const Mat22& adj_h) const { return RhoDense(adj_h, d_ - 1, ctx_); }

void GammaHHeilbronn::image(uint32_t col, std::vector<std::pair<uint32_t, uint32_t>>& out) const {
    const uint32_t J = col / d_ + 1, a = col % d_;
    const uint64_t N = G_->N();
    const bool coprime = gcd<uint64_t>(n_, N) == 1;
    for (const Mat22& eps : mer_) {
        if (!G_->act_defined(J, eps)) {
            ENGINE_CHECK_MSG(!coprime, "GammaHHeilbronn: an inadmissible pair at (n, N) = 1");
            continue;
        }
        const uint32_t i = G_->act(J, eps);
        if (d_ == 1) {
            out.push_back({i - 1, 1u % ctx_.p});
            continue;
        }
        // h = w_J eps w_i^{-1} (det n, lower left 0, lower right in H mod N); block rho_n(adj h).
        std::array<uint32_t, 4> red;
        try {
            const Mat22 h = G_->rep(J) * eps * G_->rep(i).inv_unimodular();
            ENGINE_CHECK_MSG(h.det() == static_cast<int64_t>(n_) && modi(h.c, N) == 0 && G_->in_H(modi(h.d, N)),
                             "GammaHHeilbronn: h_J(eps) not of det n with lower row (0, H) mod N");
            red = h.adj().reduced_mod(ctx_.p);
        } catch (const HeckeOverflow&) {
            const Mat22W h = Mat22W::from(G_->rep(J)) * Mat22W::from(eps) * Mat22W::from(G_->rep(i).inv_unimodular());
            ENGINE_CHECK_MSG(h.det() == static_cast<__int128>(n_) && ModW(h.c, N) == 0 && G_->in_H(ModW(h.d, N)),
                             "GammaHHeilbronn: h_J(eps) not of det n with lower row (0, H) mod N (int128)");
            red = Mat22W{h.d, -h.b, -h.c, h.a}.reduced_mod(ctx_.p);
        }
        const std::vector<uint32_t> B = rho_block(Mat22{red[0], red[1], red[2], red[3]});
        for (uint32_t b = 0; b < d_; ++b) {
            const uint32_t v = B[static_cast<size_t>(b) * d_ + a];
            if (v) out.push_back({(i - 1) * d_ + b, v});
        }
    }
}

SparseVec ApplySym(const SymbolOperator& X, const SparseVec& x) {
    std::vector<std::pair<uint32_t, uint32_t>> img, acc;
    const uint64_t p = X.p();
    for (size_t k = 0; k < x.col.size(); ++k) {
        img.clear();
        X.image(x.col[k], img);
        for (const auto& e : img) acc.push_back({e.first, static_cast<uint32_t>(static_cast<uint64_t>(e.second) * x.val[k] % p)});
    }
    std::sort(acc.begin(), acc.end());
    SparseVec y;
    for (size_t i = 0; i < acc.size();) {
        uint64_t s = 0;
        size_t j = i;
        for (; j < acc.size() && acc[j].first == acc[i].first; ++j) s += acc[j].second;
        if (s % p) { y.col.push_back(acc[i].first); y.val.push_back(static_cast<uint32_t>(s % p)); }
        i = j;
    }
    return y;
}

uint32_t RelEscapesSym(const SparseCokernel& Q, const SymbolOperator& X) {
    ENGINE_CHECK_MSG(!Q.generators().empty() && Q.p() == X.p() && Q.D() == X.D(), "RelEscapesSym: kept generators, same field and fibre");
    uint32_t bad = 0;
    for (const SparseVec& g : Q.generators()) bad += Q.in_rel(ApplySym(X, g)) ? 0 : 1;
    return bad;
}

// ============================================================================
// QuotientCoords
// ============================================================================
QuotientCoords::QuotientCoords(const SparseCokernel& Q) : h_(Q.h1()), p_(Q.p()), free_(Q.free_columns()) {
    start_.assign(static_cast<size_t>(Q.D()) + 1, 0);
    for (uint32_t c = 0; c < Q.D(); ++c) {
        start_[c] = ent_.size();
        if (Q.is_pivot(c)) {
            const SparseVec& t = Q.tail_of(c);
            for (size_t i = 0; i < t.col.size(); ++i) {
                const int32_t f = Q.free_index(t.col[i]);
                ENGINE_CHECK_MSG(f >= 0, "QuotientCoords: a tail entry at a pivot column");
                ent_.push_back({static_cast<uint32_t>(f), (p_ - t.val[i] % p_) % p_});
            }
        } else {
            ent_.push_back({static_cast<uint32_t>(Q.free_index(c)), 1u});
        }
    }
    start_[Q.D()] = ent_.size();
    ent_.push_back({0, 0});   // sentinel: begin(D - 1) / end(D - 1) stay dereferenceable
}

// ============================================================================
// cusps
// ============================================================================
CuspData CuspsOf(const GammaHLevelN& G) {
    if (!G.contains_minus_one())
        throw std::logic_error("gammaH_minus_one: the cusp / boundary routine needs -1 in H (-I in Gamma_H(N)); H = " + G.H_text());
    CuspData C;
    const uint32_t m = G.m();
    C.cusp.assign(m + 1, UINT32_MAX);
    std::vector<uint32_t> tnext(m + 1), snext(m + 1);
    for (uint32_t J = 1; J <= m; ++J) {
        const Mat22 w = G.rep(J);
        tnext[J] = G.class_of(w * Mat22::T());
        snext[J] = G.class_of(w * Mat22::S());
        if (snext[J] == J) ++C.nu2;
        if (G.class_of(w * Mat22::ST()) == J) ++C.nu3;
    }
    for (uint32_t J = 1; J <= m; ++J) {
        if (C.cusp[J] != UINT32_MAX) continue;
        uint32_t len = 0, K = J;
        do { C.cusp[K] = C.c; ++len; K = tnext[K]; } while (K != J);
        C.width.push_back(len);
        ++C.c;
    }
    C.cusp_of_S.assign(m + 1, 0);
    for (uint32_t J = 1; J <= m; ++J) C.cusp_of_S[J] = C.cusp[snext[J]];
    C.twelve_g = 12 + static_cast<int64_t>(m) - 3 * static_cast<int64_t>(C.nu2) - 4 * static_cast<int64_t>(C.nu3) -
                 6 * static_cast<int64_t>(C.c);
    return C;
}

} // namespace level
} // namespace hecke
