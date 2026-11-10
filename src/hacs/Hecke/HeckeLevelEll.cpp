// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 13, WP-6) Level structure by Shapiro on the fixed
// model -- see HeckeLevelEll.h for the contract references and the plug points.
#include "HeckeLevelEll.h"

#include <algorithm>
#include <stdexcept>

#include "CoefficientRing.h"   // RhoOfExactRing
#include "../LocalSystemSL2Z.h" // GenerateSL2ZRelationsWithGroupData
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "FpRank.h"

namespace hecke {
namespace level {

namespace {

using u128 = unsigned __int128;

uint32_t CoeffMod(int64_t c, uint32_t p) {
    const int64_t r = c % static_cast<int64_t>(p);
    return static_cast<uint32_t>(r < 0 ? r + static_cast<int64_t>(p) : r);
}

// Extended gcd: returns g, x, y with a x + b y = g.
int64_t ExtGcd(int64_t a, int64_t b, int64_t& x, int64_t& y) {
    if (b == 0) { x = 1; y = 0; return a; }
    int64_t x1, y1;
    const int64_t g = ExtGcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return g;
}

// The modulus a row with exponent e is meaningful in.
uint64_t RowModulus(const HMat& A, uint32_t i) {
    return A.field ? A.modulus : engine::scalar::pow_wrap(A.ell, A.row_exp[i]);
}

} // namespace

// ---- level structures -------------------------------------------------------
Mat22 FullLevel::rep(uint32_t a) const {
    ENGINE_CHECK_MSG(a == 1, "FullLevel: Gamma has one coset");
    return Mat22::I();
}
uint32_t FullLevel::class_of(const Mat22& gamma) const {
    ENGINE_CHECK_MSG(gamma.det() == 1, "FullLevel::class_of: not in SL_2(Z)");
    return 1;
}
bool FullLevel::contains(const Mat22& gamma) const { return gamma.det() == 1; }
bool FullLevel::contains_wide(const Mat22W& gamma) const { return gamma.det() == 1; }

// (slot-17 WP-5) the default wide membership test: narrow when it fits, else a loud refusal.
bool CosetLevel::contains_wide(const Mat22W& g) const {
    const __int128 lo = static_cast<__int128>(INT64_MIN), hi = static_cast<__int128>(INT64_MAX);
    for (const __int128 x : {g.a, g.b, g.c, g.d})
        if (x < lo || x > hi)
            throw HeckeOverflow(("CosetLevel::contains_wide: an entry beyond int64 and no wide membership test for " + name()).c_str());
    return contains(Mat22{static_cast<int64_t>(g.a), static_cast<int64_t>(g.b), static_cast<int64_t>(g.c),
                          static_cast<int64_t>(g.d)});
}

bool Gamma0Level::contains(const Mat22& gamma) const {
    return gamma.det() == 1 && (gamma.c % static_cast<int64_t>(c_->N())) == 0;
}

bool ConjugateByInverse(const Mat22& g, const Mat22& x, Mat22& out) {
    const int64_t D = g.det();
    ENGINE_CHECK_MSG(D != 0, "ConjugateByInverse: g is singular");
    const Mat22 M = g.adj() * x * g;
    if (M.a % D != 0 || M.b % D != 0 || M.c % D != 0 || M.d % D != 0) return false;
    out = Mat22{M.a / D, M.b / D, M.c / D, M.d / D};
    return true;
}

std::vector<Mat22> Gamma0SubTransversal(uint64_t N, uint64_t p) {
    ENGINE_CHECK_MSG(N >= 1 && p >= 2, "Gamma0SubTransversal: N >= 1, p prime");
    std::vector<Mat22> r;
    const int64_t Ni = static_cast<int64_t>(N), pi = static_cast<int64_t>(p);
    for (int64_t t = 0; t < pi; ++t) r.push_back(Mat22{1, 0, Ni * t, 1});
    if (N % p != 0) {
        // [[x, y], [N, p]], x p - y N = 1: bottom row (N : p) = (1 : 0) mod p.
        int64_t x, y;
        const int64_t g = ExtGcd(pi, Ni, x, y);   // p x + N y = 1
        ENGINE_CHECK_MSG(g == 1 || g == -1, "Gamma0SubTransversal: gcd(N, p) != 1");
        if (g == -1) { x = -x; y = -y; }
        const Mat22 m{x, -y, Ni, pi};
        ENGINE_CHECK_MSG(m.det() == 1, "Gamma0SubTransversal: extra representative is not unimodular");
        r.push_back(m);
    }
    return r;
}

// ---- correspondences ----------------------------------------------------------
Mat22 CorrespondenceSpec::g() const {
    return g_cosets ? g_cosets->g() : Mat22::I();
}

bool InDoublePrime(const CorrespondenceSpec& s, const Mat22& x) {
    if (!s.B->contains(x)) return false;
    Mat22 k;
    if (!ConjugateByInverse(s.g(), x, k)) return false;
    return s.A->contains(k);
}

void VerifyTransversal(const CorrespondenceSpec& s, uint32_t expected_index) {
    ENGINE_CHECK_MSG(s.A && s.B, "VerifyTransversal: A and B must be set");
    ENGINE_CHECK_MSG((s.g_cosets == nullptr) == (s.tau == nullptr),
                     "VerifyTransversal: the level-1 datum is (cosets, tau) or the identity");
    if (s.tau) ENGINE_CHECK_MSG(&s.tau->cosets() == s.g_cosets, "VerifyTransversal: tau is not built on g_cosets");
    ENGINE_CHECK_MSG(s.r.size() == expected_index,
                     ("VerifyTransversal: " + s.label + ": |r| = " + std::to_string(s.r.size()) +
                      " != [B : B''] = " + std::to_string(expected_index)).c_str());
    for (const Mat22& x : s.r) ENGINE_CHECK_MSG(s.B->contains(x), ("VerifyTransversal: " + s.label + ": r_t not in B").c_str());
    for (size_t t = 0; t < s.r.size(); ++t)
        for (size_t u = t + 1; u < s.r.size(); ++u)
            ENGINE_CHECK_MSG(!InDoublePrime(s, s.r[t] * s.r[u].inv_unimodular()),
                             ("VerifyTransversal: " + s.label + ": two r_t in the same B''-coset").c_str());
}

uint32_t RhoCache::block_of(const Mat22& g, std::vector<Block>& store) {
    const std::array<uint32_t, 4> key = g.reduced_mod(ctx_.p);
    auto it = memo_.find(key);
    if (it != memo_.end()) return it->second;
    const uint32_t d = n_ + 1;
    FlatMatrixT<uint32_t> R;
    if (ctx_.is_field()) {
        const FlatMatrixT<uint32_t> A(2, 2, ctx_.p, std::vector<uint32_t>{key[0], key[1], key[2], key[3]});
        R = ComputeSymmetricPowerMatrixFM(A, n_);
    } else {
        R = ComputeSymmetricPowerMatrixRing<uint32_t>(ctx_, key[0], key[1], key[2], key[3], n_);
    }
    Block b;
    b.dense.assign(static_cast<size_t>(d) * d, 0);
    for (uint32_t r = 0; r < d; ++r)
        for (uint32_t c = 0; c < d; ++c) b.dense[static_cast<size_t>(r) * d + c] = R.get_entry(r, c);
    // rank <= 1 detection (field only: a pivot must be a unit).
    if (ctx_.is_field()) {
        uint32_t r0 = d, c0 = d;
        for (uint32_t r = 0; r < d && r0 == d; ++r)
            for (uint32_t c = 0; c < d; ++c)
                if (b.dense[static_cast<size_t>(r) * d + c] != 0) { r0 = r; c0 = c; break; }
        b.u.assign(d, 0);
        b.v.assign(d, 0);
        bool ok = true;
        if (r0 < d) {
            for (uint32_t c = 0; c < d; ++c) b.v[c] = b.dense[static_cast<size_t>(r0) * d + c];
            const uint32_t iv = ctx_.inv(b.v[c0]);
            for (uint32_t r = 0; r < d && ok; ++r) {
                b.u[r] = ctx_.mul(b.dense[static_cast<size_t>(r) * d + c0], iv);
                for (uint32_t c = 0; c < d; ++c)
                    if (ctx_.mul(b.u[r], b.v[c]) != b.dense[static_cast<size_t>(r) * d + c]) { ok = false; break; }
            }
        }
        b.rank1 = ok;
        if (!ok) { b.u.clear(); b.v.clear(); }
    }
    store.push_back(std::move(b));
    const uint32_t idx = static_cast<uint32_t>(store.size() - 1);
    memo_.emplace(key, idx);
    return idx;
}

std::vector<uint32_t> BlockCochainMap::apply(const std::vector<uint32_t>& X, uint32_t ncols) const {
    ENGINE_CHECK_MSG(X.size() == static_cast<size_t>(src_dim()) * ncols, "BlockCochainMap::apply: operand shape");
    const uint32_t p = ctx.p;
    const bool lazy = p < (1u << 21);   // a sum of <= 2^22 products < 2^42 each fits u64
    std::vector<uint64_t> Y(static_cast<size_t>(dst_dim()) * ncols, 0);
    std::vector<uint64_t> w(ncols);
    uint64_t pending = 0;               // products accumulated into Y since the last reduction
    auto reduce_all = [&]() { for (uint64_t& y : Y) y %= p; pending = 0; };
    for (const BlockTerm& t : terms) {
        if (t.coeff == 0) continue;
        const Block& B = blocks[t.block];
        const size_t bs = static_cast<size_t>(t.src_sigma) * m_src * d + static_cast<size_t>(t.src_blk - 1) * d;
        const size_t bd = static_cast<size_t>(t.dst_sigma) * m_dst * d + static_cast<size_t>(t.dst_blk - 1) * d;
        if (lazy && pending > (1ull << 21)) reduce_all();
        if (B.rank1) {
            for (uint32_t c = 0; c < ncols; ++c) {
                uint64_t acc = 0;
                for (uint32_t s = 0; s < d; ++s) {
                    const uint32_t vs = B.v[s];
                    if (vs == 0) continue;
                    acc += static_cast<uint64_t>(vs) * X[(bs + s) * ncols + c];
                    if (!lazy) acc %= p;
                }
                w[c] = ctx.mul(static_cast<uint32_t>(acc % p), t.coeff);
            }
            for (uint32_t r = 0; r < d; ++r) {
                const uint32_t ur = B.u[r];
                if (ur == 0) continue;
                uint64_t* yrow = &Y[(bd + r) * ncols];
                for (uint32_t c = 0; c < ncols; ++c) {
                    yrow[c] += static_cast<uint64_t>(ur) * w[c];
                    if (!lazy) yrow[c] %= p;
                }
            }
            pending += 1;
        } else {
            for (uint32_t r = 0; r < d; ++r) {
                uint64_t* yrow = &Y[(bd + r) * ncols];
                for (uint32_t s = 0; s < d; ++s) {
                    const uint32_t b = B.dense[static_cast<size_t>(r) * d + s];
                    if (b == 0) continue;
                    const uint32_t cb = ctx.mul(b, t.coeff);
                    const uint32_t* xrow = &X[(bs + s) * ncols];
                    for (uint32_t c = 0; c < ncols; ++c) {
                        yrow[c] += static_cast<uint64_t>(cb) * xrow[c];
                        if (!lazy) yrow[c] %= p;
                    }
                }
            }
            pending += d;
        }
    }
    std::vector<uint32_t> out(Y.size());
    for (size_t i = 0; i < Y.size(); ++i) out[i] = static_cast<uint32_t>(Y[i] % p);
    return out;
}

std::map<std::array<uint32_t, 4>, std::vector<uint32_t>> BlockCochainMap::aggregate() const {
    std::map<std::array<uint32_t, 4>, std::vector<uint32_t>> out;
    for (const BlockTerm& t : terms) {
        const std::array<uint32_t, 4> key{t.dst_sigma, t.dst_blk, t.src_sigma, t.src_blk};
        std::vector<uint32_t>& acc = out[key];
        if (acc.empty()) acc.assign(static_cast<size_t>(d) * d, 0);
        const Block& B = blocks[t.block];
        for (size_t e = 0; e < acc.size(); ++e) acc[e] = ctx.add(acc[e], ctx.mul(t.coeff, B.dense[e]));
    }
    // drop blocks that summed to zero, so that equal maps have equal aggregates
    for (auto it = out.begin(); it != out.end();) {
        if (std::all_of(it->second.begin(), it->second.end(), [](uint32_t x) { return x == 0; })) it = out.erase(it);
        else ++it;
    }
    return out;
}

BlockCochainMap BuildLevelCorrespondence(const engine::ModCtx<uint32_t>& ctx, const CorrespondenceSpec& spec,
                                         const SimplicialComplex& K, uint32_t n_sym, uint32_t q) {
    ENGINE_CHECK_MSG(spec.A && spec.B, "BuildLevelCorrespondence: A and B must be set");
    ENGINE_CHECK_MSG((spec.g_cosets == nullptr) == (spec.tau == nullptr),
                     "BuildLevelCorrespondence: the level-1 datum is (cosets, tau) or the identity");
    const bool identity = spec.tau == nullptr;
    if (!identity) {
        ENGINE_CHECK_MSG(spec.tau->n_sym() == n_sym, "BuildLevelCorrespondence: tau's n_sym differs");
        ENGINE_CHECK_MSG(&spec.tau->complex() == &K || spec.tau->complex().get_num_q_simplices(q) == K.get_num_q_simplices(q),
                         "BuildLevelCorrespondence: tau is built on another model");
    }
    ENGINE_CHECK_MSG(q < 3, "BuildLevelCorrespondence: q in {0, 1, 2} on the 7-vertex model");
    BlockCochainMap F;
    F.ctx = ctx;
    F.q = q;
    F.N_q = K.get_num_q_simplices(q);
    F.d = n_sym + 1;
    F.m_src = spec.A->m();
    F.m_dst = spec.B->m();
    RhoCache rho(ctx, n_sym);
    const Mat22 g = spec.g();
    const int64_t detg = g.det();
    const std::vector<ChainTerm> id_chain_proto{ChainTerm{1, Mat22::I(), 0}};
    std::vector<Mat22> w_inv(F.m_src + 1);
    for (uint32_t a = 1; a <= F.m_src; ++a) w_inv[a] = spec.A->rep(a).inv_unimodular();
    for (uint32_t i = 1; i <= F.m_dst; ++i) {
        const Mat22 v_i = spec.B->rep(i);
        for (const Mat22& r_t : spec.r) {
            const Mat22 x = r_t * v_i;
            uint32_t j = 1;
            Mat22 h = x, delta1 = Mat22::I();
            if (!identity) {
                j = spec.g_cosets->class_index_of(x);
                h = x * spec.g_cosets->s_inv(j);
                ENGINE_CHECK_MSG(h.det() == 1 && (h.c % static_cast<int64_t>(spec.g_cosets->N())) == 0,
                                 "BuildLevelCorrespondence: r_t v_i s_j^{-1} is not in Gamma''_g");
                delta1 = spec.g_cosets->delta(j);   // s_j^{-1} g, integral
            }
            Mat22 k;
            ENGINE_CHECK_MSG(ConjugateByInverse(g, h, k) && k.det() == 1,
                             "BuildLevelCorrespondence: g^{-1} h g is not in Gamma (prop:levelcorrespondence (1))");
            const uint32_t a = spec.A->class_of(k);
            ENGINE_CHECK_MSG(spec.A->contains(k * w_inv[a]), "BuildLevelCorrespondence: k w_a^{-1} not in A");
            const Mat22 vd = v_i * delta1;          // v_i s_j^{-1} g
            for (uint32_t sigma = 0; sigma < F.N_q; ++sigma) {
                std::vector<ChainTerm> id_chain;
                if (identity) { id_chain = id_chain_proto; id_chain[0].sigma_idx = sigma; }
                const std::vector<ChainTerm>& chain = identity ? id_chain : spec.tau->tau(j, q, sigma);
                const Mat22 wa = spec.A->rep(a);
                for (const ChainTerm& ct : chain) {
                    ENGINE_CHECK_MSG(ct.sigma_idx < F.N_q, "BuildLevelCorrespondence: tau term out of range");
                    const uint32_t c = CoeffMod(ct.coeff, ctx.p);
                    if (c == 0) continue;
                    const uint32_t a2 = spec.A->class_of(wa * ct.gamma);   // a . gamma
                    const Mat22 E = vd * ct.gamma * w_inv[a2];            // v_i s_j^{-1} g gamma w_{a.gamma}^{-1}
                    ENGINE_CHECK_MSG(E.det() == detg, "BuildLevelCorrespondence: a block matrix has det != det g");
                    F.terms.push_back(BlockTerm{sigma, i, ct.sigma_idx, a2, c, rho.block_of(E, F.blocks)});
                }
            }
        }
    }
    return F;
}

BlockCochainMap BuildUellLevelEllFused(const engine::ModCtx<uint32_t>& ctx, const Gamma0Level& lvl,
                                       const HeckeTauTable& tau, const SimplicialComplex& K, uint32_t n_sym,
                                       uint32_t q) {
    const HeckeCosetTables& C = lvl.cosets();
    ENGINE_CHECK_MSG(&tau.cosets() == &C, "BuildUellLevelEllFused: tau must be built on the level-ell coset table");
    ENGINE_CHECK_MSG(C.r() == 1, "BuildUellLevelEllFused: N = ell (a prime) only");
    ENGINE_CHECK_MSG(tau.n_sym() == n_sym, "BuildUellLevelEllFused: tau's n_sym differs");
    BlockCochainMap F;
    F.ctx = ctx;
    F.q = q;
    F.N_q = K.get_num_q_simplices(q);
    F.d = n_sym + 1;
    F.m_src = F.m_dst = C.num_cosets();
    const uint32_t m = F.m_src;
    RhoCache rho(ctx, n_sym);
    const Mat22 g = C.g();
    const Mat22 w_last = C.s(m);                     // s_{ell+1} = S (the class (1:0))
    std::vector<Mat22> s_inv(m + 1);
    for (uint32_t j = 1; j <= m; ++j) s_inv[j] = C.s_inv(j);
    for (uint32_t sigma = 0; sigma < F.N_q; ++sigma)
        for (uint32_t i = 1; i <= m; ++i)
            for (const ChainTerm& ct : tau.tau(i, q, sigma)) {
                const uint32_t c = CoeffMod(ct.coeff, ctx.p);
                if (c == 0) continue;
                const uint32_t j0 = C.class_index_of(w_last * ct.gamma);   // (ell+1) . gamma
                const Mat22 gg = g * ct.gamma;
                for (uint32_t j = 1; j <= m; ++j) {
                    if (j == j0) continue;
                    const Mat22 E = gg * s_inv[j];                      // g gamma s_j^{-1}
                    ENGINE_CHECK_MSG(E.det() == g.det(), "BuildUellLevelEllFused: det != ell");
                    F.terms.push_back(BlockTerm{sigma, i, ct.sigma_idx, j, c, rho.block_of(E, F.blocks)});
                }
            }
    return F;
}

// ---- the level-N model over a ring ---------------------------------------------
std::vector<LocalSystemSimplicialComplex::SimplexRelation> InducedRelationsRing(
    const SimplicialComplex& K, const HeckeCosetTables& cosets, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx) {
    using Rel = LocalSystemSimplicialComplex::SimplexRelation;
    ENGINE_CHECK_MSG(!ctx.is_field(), "InducedRelationsRing: composite-only (the field version is MakeInducedLocalSystemGamma0)");
    const uint32_t m = cosets.num_cosets(), d = n_sym + 1, D = m * d;
    const std::vector<SL2ZRelationDatum> base = GenerateSL2ZRelationsWithGroupData(K, n_sym, static_cast<uint32_t>(ctx.ell));
    const Mat22 neg_I = Mat22::I().neg();
    auto gen_index = [](const Mat22& g) -> uint32_t {
        if (g == Mat22::T()) return 0;
        if (g == Mat22::S()) return 1;
        if (g == Mat22::ST()) return 2;
        if (g == Mat22::TS()) return 3;
        ENGINE_CHECK_MSG(false, "InducedRelationsRing: g_r is not one of {T, S, ST, TS, -I}");
        return 0;
    };
    std::map<uint32_t, FlatMatrixT<uint32_t>> by_gen;
    std::vector<Rel> out;
    for (const SL2ZRelationDatum& dd : base) {
        uint32_t key = 4;
        if (!(dd.g_r == neg_I)) key = gen_index(dd.g_r);
        auto it = by_gen.find(key);
        if (it == by_gen.end()) {
            std::vector<uint32_t> buf(static_cast<size_t>(D) * D, 0);
            if (key == 4) {
                const uint32_t s = (n_sym % 2 == 0) ? 1u : ctx.p - 1;
                for (uint32_t r = 0; r < D; ++r) buf[static_cast<size_t>(r) * D + r] = s;
            } else {
                for (uint32_t i = 1; i <= m; ++i) {
                    const uint32_t j = cosets.perm(key, i);
                    const FlatMatrixT<uint32_t> R = RhoOfExactRing<uint32_t>(ctx, cosets.cocycle(key, i), n_sym);
                    for (uint32_t r = 0; r < d; ++r)
                        for (uint32_t c = 0; c < d; ++c)
                            buf[static_cast<size_t>((i - 1) * d + r) * D + (j - 1) * d + c] = R.get_entry(r, c);
                }
            }
            it = by_gen.emplace(key, FlatMatrixT<uint32_t>(D, D, ctx.p, buf)).first;
        }
        out.push_back(Rel(dd.rel.q_dim, dd.rel.s_lhs_idx_in_Kq, dd.rel.s_rhs_idx_in_Kq, it->second, dd.rel.tag));
    }
    return out;
}

// ---- reading on cohomology ----------------------------------------------------------
HFrame::HFrame(const EquivariantCohomologyOnK* coh, uint32_t q) : coh_(coh), q_(q) {
    ENGINE_CHECK_MSG(coh && coh->is_cohomology_computed(), "HFrame: the cohomology must be computed");
    const SimplicialComplex* K = coh->get_lssc()->get_base_K_complex();
    N_q_ = K->get_num_q_simplices(q);
    D_ = coh->get_representation_dim();
    dim_ = N_q_ * D_;
    if (coh->is_composite()) {
        ctx_ = coh->ring_context();
        const engine::DenseBuf<uint32_t>& Hb = coh->H_module_basis(q);
        t_ = Hb.cols;
        ENGINE_CHECK_MSG(Hb.rows == dim_, "HFrame: module basis shape");
        exps_ = coh->module_degree(q).dec.exponents;
        ENGINE_CHECK_MSG(exps_.size() == t_, "HFrame: exponents / basis size mismatch");
        basis_.assign(static_cast<size_t>(dim_) * t_, 0);
        for (uint32_t r = 0; r < dim_; ++r)
            for (uint32_t c = 0; c < t_; ++c) basis_[static_cast<size_t>(r) * t_ + c] = Hb.at(r, c);
    } else {
        ctx_ = engine::ModCtx<uint32_t>(coh->get_prime());
        t_ = coh->get_H_eq_dimensions()[q];
        exps_.assign(t_, 1);
        const auto& Hb = coh->get_H_eq_bases_fm()[q];
        ENGINE_CHECK_MSG(Hb.size() == t_, "HFrame: basis size mismatch");
        basis_.assign(static_cast<size_t>(dim_) * t_, 0);
        for (uint32_t c = 0; c < t_; ++c)
            for (uint32_t s = 0; s < N_q_; ++s)
                for (uint32_t r = 0; r < D_; ++r) basis_[static_cast<size_t>(s * D_ + r) * t_ + c] = Hb[c].get_entry(s, r);
    }
}

void HFrame::require_equivariant_cocycles(const std::vector<uint32_t>& Y, uint32_t ncols, const std::string& what) const {
    ENGINE_CHECK_MSG(Y.size() == static_cast<size_t>(dim_) * ncols, "HFrame: operand shape");
    const engine::ModCtx<uint32_t>& ctx = ctx_;
    for (const auto& rel : coh_->get_lssc()->get_user_defined_simplex_relations()) {
        if (rel.q_dim != q_) continue;
        // sparse rows of M (monomial: m d nonzeros per ... row block)
        for (uint32_t a = 0; a < D_; ++a) {
            std::vector<std::pair<uint32_t, uint32_t>> row;
            for (uint32_t b = 0; b < D_; ++b) {
                const uint32_t mv = rel.M.get_entry(a, b);
                if (mv != 0) row.emplace_back(b, mv);
            }
            for (uint32_t c = 0; c < ncols; ++c) {
                uint32_t acc = Y[static_cast<size_t>(rel.s_lhs_idx_in_Kq * D_ + a) * ncols + c];
                for (const auto& e : row)
                    acc = ctx.sub(acc, ctx.mul(e.second, Y[static_cast<size_t>(rel.s_rhs_idx_in_Kq * D_ + e.first) * ncols + c]));
                if (acc != 0)
                    throw std::runtime_error("hecke::level: " + what + ": an image cochain violates relation '" + rel.tag +
                                             "' at q=" + std::to_string(q_));
            }
        }
    }
    if (q_ + 1 < 3) {
        const SimplicialComplex* K = coh_->get_lssc()->get_base_K_complex();
        const FlatMatrixT<uint32_t> D1 = K->coboundary_operator_elt<uint32_t>(q_, ctx.p);
        const uint32_t N1 = K->get_num_q_simplices(q_ + 1);
        for (uint32_t s1 = 0; s1 < N1; ++s1)
            for (uint32_t a = 0; a < D_; ++a)
                for (uint32_t c = 0; c < ncols; ++c) {
                    uint32_t acc = 0;
                    for (uint32_t s0 = 0; s0 < N_q_; ++s0) {
                        const uint32_t inc = D1.get_entry(s1, s0);
                        if (inc != 0) acc = ctx.add(acc, ctx.mul(inc, Y[static_cast<size_t>(s0 * D_ + a) * ncols + c]));
                    }
                    if (acc != 0)
                        throw std::runtime_error("hecke::level: " + what + ": an image cochain is not a cocycle at q=" +
                                                 std::to_string(q_));
                }
    }
}

HMat HFrame::coords(const std::vector<uint32_t>& Y, uint32_t ncols) const {
    HMat out;
    out.rows = t_;
    out.cols = ncols;
    out.modulus = ctx_.p;
    out.ell = ctx_.ell;
    out.field = ctx_.is_field();
    out.row_exp = exps_;
    out.a.assign(static_cast<size_t>(t_) * ncols, 0);
    if (t_ == 0 || ncols == 0) return out;
    if (coh_->is_composite()) {
        engine::DenseBuf<uint32_t> Yb(dim_, ncols);
        for (uint32_t r = 0; r < dim_; ++r)
            for (uint32_t c = 0; c < ncols; ++c) Yb.at(r, c) = Y[static_cast<size_t>(r) * ncols + c];
        const std::vector<std::vector<uint32_t>> co = coh_->H_class_coordinates(q_, engine::cview(Yb));
        for (uint32_t c = 0; c < ncols; ++c)
            for (uint32_t i = 0; i < t_; ++i) out.a[static_cast<size_t>(i) * ncols + c] = co[c][i];
    } else {
        const FlatMatrixT<uint32_t>& P = coh_->get_H_eq_projectors_from_C_eq_flat_fm()[q_];
        ENGINE_CHECK_MSG(P.get_rows() == t_ && P.get_cols() == dim_, "HFrame: projector shape");
        const uint32_t p = ctx_.p;
        for (uint32_t i = 0; i < t_; ++i) {
            std::vector<uint64_t> acc(ncols, 0);
            for (uint32_t r = 0; r < dim_; ++r) {
                const uint32_t pv = P.get_entry(i, r);
                if (pv == 0) continue;
                const uint32_t* yrow = &Y[static_cast<size_t>(r) * ncols];
                for (uint32_t c = 0; c < ncols; ++c) acc[c] = (acc[c] + static_cast<uint64_t>(pv) * yrow[c]) % p;
            }
            for (uint32_t c = 0; c < ncols; ++c) out.a[static_cast<size_t>(i) * ncols + c] = acc[c];
        }
    }
    return HCanonical(out);
}

std::vector<uint32_t> HFrame::ceq_basis(uint32_t& c) const {
    std::vector<uint32_t> out;
    if (coh_->is_composite()) {
        const engine::DenseBuf<uint32_t>& C = coh_->module_degree(q_).C;
        c = C.cols;
        out.assign(static_cast<size_t>(dim_) * c, 0);
        for (uint32_t r = 0; r < dim_; ++r)
            for (uint32_t j = 0; j < c; ++j) out[static_cast<size_t>(r) * c + j] = C.at(r, j);
    } else {
        const auto& Cb = coh_->get_C_eq_bases_fm()[q_];
        c = static_cast<uint32_t>(Cb.size());
        out.assign(static_cast<size_t>(dim_) * c, 0);
        for (uint32_t j = 0; j < c; ++j)
            for (uint32_t s = 0; s < N_q_; ++s)
                for (uint32_t r = 0; r < D_; ++r) out[static_cast<size_t>(s * D_ + r) * c + j] = Cb[j].get_entry(s, r);
    }
    return out;
}

HMat InducedOnH(const HFrame& src, const HFrame& dst, const BlockCochainMap& F, const std::string& what) {
    ENGINE_CHECK_MSG(F.src_dim() == src.flat_dim() && F.dst_dim() == dst.flat_dim(),
                     ("InducedOnH: " + what + ": the cochain map does not fit the frames").c_str());
    ENGINE_CHECK_MSG(src.ctx().p == dst.ctx().p && F.ctx.p == src.ctx().p, ("InducedOnH: " + what + ": rings differ").c_str());
    const std::vector<uint32_t> Y = F.apply(src.basis(), src.t());
    dst.require_equivariant_cocycles(Y, src.t(), what);
    return dst.coords(Y, src.t());
}

// ---- HMat arithmetic --------------------------------------------------------------
HMat HCanonical(HMat A) {
    for (uint32_t i = 0; i < A.rows; ++i) {
        const uint64_t q = RowModulus(A, i);
        for (uint32_t j = 0; j < A.cols; ++j) A.a[static_cast<size_t>(i) * A.cols + j] %= q;
    }
    return A;
}

HMat HMul(const HMat& A, const HMat& B) {
    ENGINE_CHECK_MSG(A.cols == B.rows && A.modulus == B.modulus, "HMul: shapes / rings differ");
    HMat C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.modulus = A.modulus;
    C.ell = A.ell;
    C.field = A.field;
    C.row_exp = A.row_exp;
    C.a.assign(static_cast<size_t>(C.rows) * C.cols, 0);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t k = 0; k < A.cols; ++k) {
            const uint64_t aik = A.at(i, k);
            if (aik == 0) continue;
            for (uint32_t j = 0; j < B.cols; ++j) {
                uint64_t& c = C.a[static_cast<size_t>(i) * C.cols + j];
                c = static_cast<uint64_t>((static_cast<u128>(aik) * B.at(k, j) + c) % A.modulus);
            }
        }
    return HCanonical(C);
}

HMat HAdd(const HMat& A, const HMat& B) {
    ENGINE_CHECK_MSG(A.rows == B.rows && A.cols == B.cols && A.modulus == B.modulus, "HAdd: shapes differ");
    HMat C = A;
    for (size_t e = 0; e < C.a.size(); ++e) C.a[e] = (A.a[e] + B.a[e]) % A.modulus;
    return HCanonical(C);
}

HMat HSub(const HMat& A, const HMat& B) {
    ENGINE_CHECK_MSG(A.rows == B.rows && A.cols == B.cols && A.modulus == B.modulus, "HSub: shapes differ");
    HMat C = A;
    for (size_t e = 0; e < C.a.size(); ++e) C.a[e] = (A.a[e] + A.modulus - B.a[e] % A.modulus) % A.modulus;
    return HCanonical(C);
}

HMat HScale(const HMat& A, uint64_t s) {
    HMat C = A;
    for (uint64_t& x : C.a) x = static_cast<uint64_t>((static_cast<u128>(x) * (s % A.modulus)) % A.modulus);
    return HCanonical(C);
}

HMat HIdentity(const HMat& like) {
    ENGINE_CHECK_MSG(like.rows == like.cols, "HIdentity: square only");
    HMat C = like;
    std::fill(C.a.begin(), C.a.end(), 0);
    for (uint32_t i = 0; i < C.rows; ++i) C.a[static_cast<size_t>(i) * C.cols + i] = 1;
    return HCanonical(C);
}

bool HEqual(const HMat& A, const HMat& B) {
    if (A.rows != B.rows || A.cols != B.cols || A.modulus != B.modulus) return false;
    const HMat a = HCanonical(A), b = HCanonical(B);
    // compare row-wise mod the smaller row exponent convention: both carry the target's exps
    for (uint32_t i = 0; i < a.rows; ++i) {
        const uint64_t q = std::min(RowModulus(a, i), RowModulus(b, i));
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(i, j) % q != b.at(i, j) % q) return false;
    }
    return true;
}

bool HIsZero(const HMat& A) {
    const HMat a = HCanonical(A);
    return std::all_of(a.a.begin(), a.a.end(), [](uint64_t x) { return x == 0; });
}

uint32_t HRankModEll(const HMat& A) {
    std::vector<uint64_t> M(static_cast<size_t>(A.rows) * A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) M[static_cast<size_t>(i) * A.cols + j] = A.at(i, j);
    return RankModPrime(M.data(), A.rows, A.cols, A.ell);
}

std::vector<uint64_t> HCharpoly(const HMat& A) {
    ENGINE_CHECK_MSG(A.field && A.rows == A.cols, "HCharpoly: a square matrix over a field");
    const uint64_t p = A.modulus;
    const uint32_t n = A.rows;
    auto mul = [&](uint64_t x, uint64_t y) { return static_cast<uint64_t>(static_cast<u128>(x) * y % p); };
    // Berkowitz, division-free: high -> low, then reversed.
    std::vector<uint64_t> pc{1};
    for (uint32_t r = 1; r <= n; ++r) {
        const uint32_t m = r - 1;
        std::vector<uint64_t> t(r + 1, 0);
        t[0] = 1;
        t[1] = (p - A.at(m, m) % p) % p;
        std::vector<uint64_t> v(m);
        for (uint32_t i = 0; i < m; ++i) v[i] = A.at(i, m);
        for (uint32_t kk = 2; kk <= r; ++kk) {
            uint64_t acc = 0;
            for (uint32_t i = 0; i < m; ++i) acc = (acc + mul(A.at(m, i), v[i])) % p;
            t[kk] = (p - acc) % p;
            std::vector<uint64_t> nv(m, 0);
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 0; j < m; ++j) nv[i] = (nv[i] + mul(A.at(i, j), v[j])) % p;
            v.swap(nv);
        }
        std::vector<uint64_t> np(r + 1, 0);
        for (uint32_t i = 0; i <= r; ++i)
            for (uint32_t j = 0; j < r && j <= i; ++j) np[i] = (np[i] + mul(t[i - j], pc[j])) % p;
        pc.swap(np);
    }
    std::reverse(pc.begin(), pc.end());
    return pc;
}

} // namespace level
} // namespace hecke
