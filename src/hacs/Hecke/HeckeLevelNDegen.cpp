// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 16, WP-13) kappa_N on the model and alpha_d on the cokernel presentation --
// see HeckeLevelNDegen.h ([C] v1.16.0-pre.3 sec:levelNdegsec: prop:kappamodel, prop:degenpres,
// rmk:degenpresconv, rmk:levelNscale, rmk:levelNcheckabletwo).
#include "HeckeLevelNDegen.h"

#include <algorithm>
#include <array>

#include "../../ModArithmetic/EngineCheck.h"

namespace hecke {
namespace level {

namespace {
int64_t ModS(int64_t x, uint64_t m) {
    const int64_t r = x % static_cast<int64_t>(m);
    return r < 0 ? r + static_cast<int64_t>(m) : r;
}
// (slot-16 WP-17, audit WP-13 F7) x a + y b with overflow-checked int64 products (a HeckeOverflow, never a wrap).
int64_t RowDot(int64_t x, int64_t a, int64_t y, int64_t b) {
    return checked_add<int64_t>(checked_mul<int64_t>(x, a), checked_mul<int64_t>(y, b));
}
} // namespace

// ---- prop:kappamodel ----------------------------------------------------------------------
KappaPoints KappaModelPoints(const SimplicialComplex& K) {
    KappaPoints pts;
    const int64_t e = K.find_simplex_index(1, {4u, 6u});
    const int64_t a = K.find_simplex_index(0, {4u});
    const int64_t b = K.find_simplex_index(0, {6u});
    ENGINE_CHECK_MSG(e >= 0 && a >= 0 && b >= 0, "KappaModelPoints: the model has no edge {v_4, v_6} (def:model7)");
    pts.e46 = static_cast<uint32_t>(e);
    pts.v4 = static_cast<uint32_t>(a);
    pts.v6 = static_cast<uint32_t>(b);
    // The stored orientation: (d g)(e_46) = g(v_6) - g(v_4), i.e. d e_46 = v_6 - v_4 (prop:kappamodel).
    const uint32_t p = 1000003;
    const FlatMatrixT<uint32_t> D0 = K.coboundary_operator_elt<uint32_t>(0, p);
    const uint32_t nv = K.get_num_q_simplices(0);
    for (uint32_t v = 0; v < nv; ++v) {
        const uint32_t want = v == pts.v6 ? 1u : (v == pts.v4 ? p - 1 : 0u);
        ENGINE_CHECK_MSG(D0.get_entry(pts.e46, v) == want,
                         "KappaModelPoints: the stored edge e_46 is not oriented v_4 -> v_6 (prop:kappamodel)");
    }
    return pts;
}

std::vector<uint32_t> CochainAt(const std::vector<uint32_t>& Y, uint32_t ncols, uint32_t col, uint32_t sigma, uint32_t D) {
    ENGINE_CHECK_MSG(col < ncols && Y.size() >= static_cast<size_t>(sigma + 1) * D * ncols, "CochainAt: out of range");
    std::vector<uint32_t> out(D);
    for (uint32_t r = 0; r < D; ++r) out[r] = Y[(static_cast<size_t>(sigma) * D + r) * ncols + col];
    return out;
}

std::vector<uint32_t> KappaModel(const engine::ModCtx<uint32_t>& ctx, const std::vector<uint32_t>& Y, uint32_t ncols,
                                 uint32_t col, const KappaPoints& pts, uint32_t D) {
    std::vector<uint32_t> v = CochainAt(Y, ncols, col, pts.e46, D);
    for (uint32_t& x : v) x = ctx.sub(0u, x);
    return v;
}

std::vector<uint32_t> KappaModel0(const std::vector<uint32_t>& Y, uint32_t ncols, uint32_t col, const KappaPoints& pts,
                                  uint32_t D) {
    return CochainAt(Y, ncols, col, pts.v4, D);
}

// ---- prop:degenpres ------------------------------------------------------------------------
DegeneracyOperator::DegeneracyOperator(const Gamma0LevelN& lvl_M, const Gamma0LevelN& lvl_N, uint64_t d, uint32_t n_sym,
                                       const engine::ModCtx<uint32_t>& ctx, DegeneracyEnumeration enumeration,
                                       DegeneracyVariant variant)
    : M_(lvl_M.N()), N_(lvl_N.N()), dd_(d), dv_(n_sym + 1), mM_(lvl_M.m()), mN_(lvl_N.m()), enum_(enumeration),
      variant_(variant), ctx_(ctx) {
    ENGINE_CHECK_MSG(d >= 1 && d < (1ull << 20), "DegeneracyOperator: d in [1, 2^20)");
    ENGINE_CHECK_MSG(N_ % (d * M_) == 0, "DegeneracyOperator: d M must divide N (prop:degenpres)");
    ENGINE_CHECK_MSG(N_ < (1ull << 20), "DegeneracyOperator: N < 2^20 (Mat22 triple products, HeckeLevelN.h)");
    std::vector<Mat22> mer = MerelSet(d);
    if (d == 1)
        ENGINE_CHECK_MSG(mer.size() == 1 && mer[0] == Mat22::I(), "DegeneracyOperator: Mer_1 = {I} (prop:degenpres (3))");
    mer_size_ = mer.size();
    std::vector<Mat22> winv_M(mM_ + 1);
    for (uint32_t j = 1; j <= mM_; ++j) winv_M[j] = lvl_M.rep(j).inv_unimodular();
    std::vector<Mat22> w_N(mN_ + 1);
    for (uint32_t i = 1; i <= mN_; ++i) w_N[i] = lvl_N.rep(i);

    // The per-eps enumeration: every class i of P^1(Z/N) bucketed ONCE by its class in P^1(Z/d)
    // (the bottom row of w_i is primitive mod N, hence mod d; admissibility is a linear condition
    // on that row mod d, invariant under units, so it depends on the class mod d only).
    const Gamma0LevelN lvl_d(d);
    const uint32_t md = lvl_d.m();
    std::vector<std::vector<uint32_t>> bucket;
    if (enum_ == DegeneracyEnumeration::kPerEpsilon) {
        bucket.assign(md + 1, {});
        for (uint32_t i = 1; i <= mN_; ++i)
            bucket[lvl_d.index_of_pair(ModS(w_N[i].c, d), ModS(w_N[i].d, d))].push_back(i);
    }

    RhoCache rho(ctx_, n_sym);
    const int64_t sd = static_cast<int64_t>(d);
    std::vector<uint32_t> adm;
    for (const Mat22& eps : mer) {
        const Mat22 ae = variant_ == DegeneracyVariant::kTrapA ? eps : eps.adj();
        adm.clear();
        if (enum_ == DegeneracyEnumeration::kPerEpsilon) {
            for (uint32_t t = 1; t <= md; ++t) {
                ++pair_tests_;
                const auto pr = lvl_d.canonical_pair(t);
                const int64_t c = static_cast<int64_t>(pr.first), e = static_cast<int64_t>(pr.second);
                if (ModS(RowDot(c, ModS(ae.a, d), e, ModS(ae.c, d)), d) != 0 || ModS(RowDot(c, ModS(ae.b, d), e, ModS(ae.d, d)), d) != 0)
                    continue;
                adm.insert(adm.end(), bucket[t].begin(), bucket[t].end());
            }
            std::sort(adm.begin(), adm.end());
        } else {
            for (uint32_t i = 1; i <= mN_; ++i) {
                ++pair_tests_;
                const Mat22& w = w_N[i];
                if (ModS(RowDot(w.c, ae.a, w.d, ae.c), d) == 0 && ModS(RowDot(w.c, ae.b, w.d, ae.d), d) == 0) adm.push_back(i);
            }
        }
        max_adm_ = std::max<uint32_t>(max_adm_, static_cast<uint32_t>(adm.size()));
        for (uint32_t i : adm) {
            const Mat22& w = w_N[i];
            const int64_t r0 = RowDot(w.c, ae.a, w.d, ae.c), r1 = RowDot(w.c, ae.b, w.d, ae.d);
            ENGINE_CHECK_MSG(r0 % sd == 0 && r1 % sd == 0, "DegeneracyOperator: an enumerated pair is not admissible");
            const uint32_t j = lvl_M.index_of_pair(r0 / sd, r1 / sd);   // primitive (prop:degenpres), checked there
            // Exact in __int128: w_i has entries O(t N) at a case-(iii) class, so the int64 triple product can
            // overflow at composite N (the HeilbronnOperator finding of WP-13); only the residues reach rho_n.
            Mat22W B = Mat22W::from(w) * Mat22W::from(ae) * Mat22W::from(winv_M[j]);
            const __int128 wd = static_cast<__int128>(sd);
            ENGINE_CHECK_MSG(B.det() == wd, "DegeneracyOperator: w_i adj(eps) w'_j^{-1} is not of determinant d");
            if (variant_ == DegeneracyVariant::kProposition) {
                // = diag(1, d) a, a in Gamma_0(M): bottom row divisible by d, (c / d) = 0 mod M.
                ENGINE_CHECK_MSG(B.c % wd == 0 && B.d % wd == 0 && (B.c / wd) % static_cast<__int128>(M_) == 0,
                                 "DegeneracyOperator: w_i adj(eps) w'_j^{-1} is not diag(1, d) Gamma_0(M) (prop:degenpres)");
            }
            if (variant_ == DegeneracyVariant::kTrapB) B = Mat22W{B.d, -B.b, -B.c, B.a};
            const std::array<uint32_t, 4> rB = B.reduced_mod(ctx_.p);
            DegeneracyTerm term;
            term.dst = i;
            term.src = j;
            term.block = rho.block_of(Mat22{rB[0], rB[1], rB[2], rB[3]}, blocks_);
            terms_.push_back(term);
        }
    }
    by_src_.assign(mM_ + 1, {});
    for (size_t t = 0; t < terms_.size(); ++t) by_src_[terms_[t].src].push_back(static_cast<uint32_t>(t));
    if (d == 1 && variant_ == DegeneracyVariant::kProposition)
        ENGINE_CHECK_MSG(terms_.size() == mN_, "DegeneracyOperator: at d = 1 every class is admissible exactly once");
}

std::string DegeneracyOperator::label() const {
    std::string v = variant_ == DegeneracyVariant::kProposition ? "" : (variant_ == DegeneracyVariant::kTrapA ? "[trapA]" : "[trapB]");
    return "D_{" + std::to_string(M_) + "," + std::to_string(N_) + "," + std::to_string(dd_) + "}" + v;
}

std::vector<uint32_t> DegeneracyOperator::apply(const std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(x.size() == static_cast<size_t>(src_dim()), "DegeneracyOperator::apply: operand length");
    std::vector<uint32_t> y(dst_dim(), 0);
    for (const DegeneracyTerm& t : terms_) {
        const std::vector<uint32_t>& B = blocks_[t.block].dense;
        const uint32_t* xs = &x[static_cast<size_t>(t.src - 1) * dv_];
        uint32_t* ys = &y[static_cast<size_t>(t.dst - 1) * dv_];
        for (uint32_t r = 0; r < dv_; ++r) {
            uint32_t s = 0;
            for (uint32_t c = 0; c < dv_; ++c) s = ctx_.add(s, ctx_.mul(B[static_cast<size_t>(r) * dv_ + c], xs[c]));
            ys[r] = ctx_.add(ys[r], s);
        }
    }
    return y;
}

SparseVec DegeneracyOperator::apply_sparse(const SparseVec& x) const {
    ENGINE_CHECK_MSG(ctx_.is_field(), "DegeneracyOperator::apply_sparse: a prime field");
    const uint64_t p = ctx_.p;
    // accumulate into touched dst blocks
    std::vector<std::pair<uint32_t, uint64_t>> acc;   // (dst column, value)
    for (size_t k = 0; k < x.col.size(); ++k) {
        const uint32_t blk = x.col[k] / dv_ + 1, r = x.col[k] % dv_;
        const uint64_t a = x.val[k];
        for (uint32_t ti : by_src_[blk]) {
            const DegeneracyTerm& t = terms_[ti];
            const std::vector<uint32_t>& B = blocks_[t.block].dense;
            for (uint32_t row = 0; row < dv_; ++row) {
                const uint32_t b = B[static_cast<size_t>(row) * dv_ + r];
                if (b) acc.emplace_back((t.dst - 1) * dv_ + row, a * b % p);
            }
        }
    }
    std::sort(acc.begin(), acc.end(), [](const std::pair<uint32_t, uint64_t>& u, const std::pair<uint32_t, uint64_t>& v) {
        return u.first < v.first;
    });
    SparseVec out;
    for (size_t k = 0; k < acc.size();) {
        const uint32_t c = acc[k].first;
        uint64_t s = 0;
        for (; k < acc.size() && acc[k].first == c; ++k) s = (s + acc[k].second) % p;
        if (s) { out.col.push_back(c); out.val.push_back(static_cast<uint32_t>(s)); }
    }
    return out;
}

std::vector<uint32_t> DegeneracyOperator::dense() const {
    const uint32_t R = dst_dim(), C = src_dim();
    std::vector<uint32_t> out(static_cast<size_t>(R) * C, 0);
    for (const DegeneracyTerm& t : terms_) {
        const std::vector<uint32_t>& B = blocks_[t.block].dense;
        for (uint32_t r = 0; r < dv_; ++r)
            for (uint32_t c = 0; c < dv_; ++c) {
                uint32_t& o = out[static_cast<size_t>((t.dst - 1) * dv_ + r) * C + (t.src - 1) * dv_ + c];
                o = ctx_.add(o, B[static_cast<size_t>(r) * dv_ + c]);
            }
    }
    return out;
}

std::vector<std::vector<uint32_t>> DegeneracyOperator::canonical_terms() const {
    std::vector<std::vector<uint32_t>> out;
    out.reserve(terms_.size());
    for (const DegeneracyTerm& t : terms_) {
        std::vector<uint32_t> k{t.dst, t.src};
        const std::vector<uint32_t>& B = blocks_[t.block].dense;
        k.insert(k.end(), B.begin(), B.end());
        out.push_back(std::move(k));
    }
    std::sort(out.begin(), out.end());
    return out;
}

uint32_t DegeneracyRelEscapes(const SparseCokernel& src, const SparseCokernel& dst, const DegeneracyOperator& D) {
    ENGINE_CHECK_MSG(src.D() == D.src_dim() && dst.D() == D.dst_dim() && src.p() == dst.p() && D.ctx().p == src.p(),
                     "DegeneracyRelEscapes: presentations and operator disagree");
    ENGINE_CHECK_MSG(!src.generators().empty() || src.D() == 0, "DegeneracyRelEscapes: the source generators were not kept");
    uint32_t bad = 0;
    for (const SparseVec& g : src.generators()) bad += dst.in_rel(D.apply_sparse(g)) ? 0 : 1;
    return bad;
}

HMat OnCokernelDegeneracy(const SparseCokernel& src, const SparseCokernel& dst, const DegeneracyOperator& D) {
    ENGINE_CHECK_MSG(DegeneracyRelEscapes(src, dst, D) == 0,
                     "OnCokernelDegeneracy: D_{M,N,d}(Rel_M) is not contained in Rel_N (prop:degenpres (1))");
    HMat Mx;
    Mx.rows = dst.h1();
    Mx.cols = src.h1();
    Mx.modulus = dst.p();
    Mx.ell = dst.p();
    Mx.field = true;
    Mx.row_exp.assign(Mx.rows, 1);
    Mx.a.assign(static_cast<size_t>(Mx.rows) * Mx.cols, 0);
    for (uint32_t j = 0; j < src.h1(); ++j) {
        SparseVec e;
        e.col.push_back(src.free_columns()[j]);
        e.val.push_back(1);
        const std::vector<uint32_t> c = dst.coords(D.apply_sparse(e));
        for (uint32_t i = 0; i < Mx.rows; ++i) Mx.a[static_cast<size_t>(i) * Mx.cols + j] = c[i];
    }
    return Mx;
}

} // namespace level
} // namespace hecke
