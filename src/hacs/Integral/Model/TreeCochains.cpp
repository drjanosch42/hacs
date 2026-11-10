// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT D5: the ell-local tree cochains of [C, prop:isomodeltree](2) in closed form.  See TreeCochains.h.
#include "TreeCochains.h"

#include <algorithm>
#include <atomic>
#include <numeric>

#include "../../Hecke/HeckeLevelEll.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"

namespace integral_model {

namespace {
std::atomic<uint32_t> g_tree_fault{0};
}  // namespace
TreeFault ActiveTreeFault() { return static_cast<TreeFault>(g_tree_fault.load(std::memory_order_relaxed)); }
ScopedTreeFault::ScopedTreeFault(TreeFault f) : saved_(g_tree_fault.exchange(static_cast<uint32_t>(f))) {}
ScopedTreeFault::~ScopedTreeFault() { g_tree_fault.store(saved_); }

using hecke::Mat22;

namespace {

uint64_t inv_mod_u64(uint64_t a, uint64_t m) {   // (S24 CL-G3, C-07) ScalarMod.h's extended Euclid (any m, no overflow)
    uint64_t x = 0;
    ENGINE_CHECK_MSG(engine::scalar::try_invmod<uint64_t>(a, m, x), "TreeCochains: a denominator is not a unit modulo the context's modulus");
    return x;
}

template<class Elt>
Elt reduce(const engine::ModCtx<Elt>& ctx, const Rat& r) {
    const uint64_t m = static_cast<uint64_t>(ctx.p);
    const int64_t pm = static_cast<int64_t>(m);
    int64_t a = r.num % pm;
    if (a < 0) a += pm;
    const uint64_t iv = inv_mod_u64(static_cast<uint64_t>(r.den), m);
    return ctx.mul(static_cast<Elt>(a), static_cast<Elt>(iv));
}

uint32_t uf_find(std::vector<uint32_t>& p, uint32_t x) {
    while (p[x] != x) {
        p[x] = p[p[x]];
        x = p[x];
    }
    return x;
}

}  // namespace

// ---------------------------------------------------------------------------
// modules
// ---------------------------------------------------------------------------
template<class Elt>
TreeModule<Elt> TreeSymModule(const engine::ModCtx<Elt>& ctx, uint32_t n) {
    TreeModule<Elt> M;
    M.name = "Sym^" + std::to_string(n);
    M.m = 1;
    M.d = n + 1;
    M.make = [ctx, n](const Mat22& g) {
        BlockMonomial<Elt> B;
        B.m = 1;
        B.d = n + 1;
        B.col = {0};
        B.blk = {0};
        const engine::DenseBuf<Elt> R = rho_n(ctx, g, n);
        std::vector<Elt> v(static_cast<size_t>(B.d) * B.d);
        for (uint32_t i = 0; i < B.d; ++i)
            for (uint32_t j = 0; j < B.d; ++j) v[static_cast<size_t>(i) * B.d + j] = R.at(i, j);
        B.blocks.push_back(std::move(v));
        return B;
    };
    return M;
}

TreeModule<uint32_t> TreeInducedSymModule(uint32_t N, uint32_t n, const engine::ModCtx<uint32_t>& ctx) {
    auto A = std::make_shared<const hecke::level::Gamma0LevelN>(N);
    TreeModule<uint32_t> M;
    M.name = "Ind_{Gamma_0(" + std::to_string(N) + ")}^Gamma Sym^" + std::to_string(n);
    M.m = A->m();
    M.d = n + 1;
    M.make = [A, n, ctx](const Mat22& g) {
        const hecke::level::MonomialAction mono = hecke::level::MonomialOf(*A, g, n, ctx);
        BlockMonomial<uint32_t> B;
        B.m = mono.m;
        B.d = mono.d;
        B.col.resize(mono.m);
        B.blk.resize(mono.m);
        for (uint32_t j = 1; j <= mono.m; ++j) {
            B.col[j - 1] = mono.perm[j] - 1;
            B.blk[j - 1] = mono.blk[j];
        }
        B.blocks.reserve(mono.blocks.size());
        for (const auto& b : mono.blocks) B.blocks.push_back(b.dense);
        return B;
    };
    return M;
}

template<class Elt>
TreeModule<Elt> TreeModuleFromDense(const CoefficientModule<Elt>& Md) {
    TreeModule<Elt> M;
    M.name = Md.name;
    M.m = 1;
    M.d = Md.dim;
    const auto rho = Md.rho;
    const uint32_t D = Md.dim;
    M.make = [rho, D](const Mat22& g) {
        BlockMonomial<Elt> B;
        B.m = 1;
        B.d = D;
        B.col = {0};
        B.blk = {0};
        const engine::DenseBuf<Elt> R = rho(g);
        std::vector<Elt> v(static_cast<size_t>(D) * D);
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) v[static_cast<size_t>(i) * D + j] = R.at(i, j);
        B.blocks.push_back(std::move(v));
        return B;
    };
    return M;
}

// ---------------------------------------------------------------------------
// image selection and block assembly
// ---------------------------------------------------------------------------
template<class Elt>
ImageSelection SelectImage(const engine::ModCtx<Elt>& ctx, const TreeModule<Elt>& M, const QGroupRingElt& e) {
    const uint32_t D = M.dim(), m = M.m, d = M.d;
    ImageSelection s;
    s.jpos.assign(D, -1);
    s.rpos.assign(D, -1);
    if (e == tree_elt::one()) {
        s.all = true;
        s.J.resize(D);
        std::iota(s.J.begin(), s.J.end(), 0u);
        s.R = s.J;
        for (uint32_t i = 0; i < D; ++i) s.jpos[i] = s.rpos[i] = static_cast<int32_t>(i);
        return s;
    }
    const uint32_t res = static_cast<uint32_t>(ctx.ell);   // the residue prime
    ENGINE_CHECK_MSG(res >= 2, "SelectImage: residue prime");
    // components of the block pattern of rho(e)
    std::vector<std::pair<Elt, const BlockMonomial<Elt>*>> terms;
    for (const auto& [g, r] : e.terms) terms.push_back({reduce(ctx, r), &M.rho(mat_of(g))});
    std::vector<uint32_t> par(m);
    std::iota(par.begin(), par.end(), 0u);
    for (const auto& t : terms)
        for (uint32_t j = 0; j < m; ++j) {
            const uint32_t a = uf_find(par, j), b = uf_find(par, t.second->col[j]);
            if (a != b) par[std::max(a, b)] = std::min(a, b);
        }
    std::map<uint32_t, std::vector<uint32_t>> comps;
    for (uint32_t j = 0; j < m; ++j) comps[uf_find(par, j)].push_back(j);
    const engine::ModCtx<uint32_t> F(res);
    for (const auto& kv : comps) {
        const std::vector<uint32_t>& blks = kv.second;   // ascending
        const uint32_t L = static_cast<uint32_t>(blks.size()) * d;
        std::map<uint32_t, uint32_t> where;
        for (uint32_t a = 0; a < blks.size(); ++a) where[blks[a]] = a;
        engine::DenseBuf<uint32_t> P(L, L);
        for (const auto& t : terms) {
            const BlockMonomial<Elt>& B = *t.second;
            for (uint32_t a = 0; a < blks.size(); ++a) {
                const uint32_t j = blks[a];
                const uint32_t b = where.at(B.col[j]);
                const std::vector<Elt>& blk = B.blocks[B.blk[j]];
                for (uint32_t r = 0; r < d; ++r)
                    for (uint32_t c = 0; c < d; ++c) {
                        const Elt v = blk[static_cast<size_t>(r) * d + c];
                        if (v == 0) continue;
                        const uint32_t x = static_cast<uint32_t>(ctx.mul(t.first, v) % static_cast<Elt>(res));
                        uint32_t& dst = P.at(a * d + r, b * d + c);
                        dst = (dst + x) % res;
                    }
            }
        }
        engine::DenseBuf<uint32_t> W = P;
        const std::vector<uint32_t> piv = engine::elim::rref(F, engine::view(W)).pivots;
        if (piv.empty()) continue;
        engine::DenseBuf<uint32_t> Bt(static_cast<uint32_t>(piv.size()), L);
        for (uint32_t a = 0; a < piv.size(); ++a)
            for (uint32_t i = 0; i < L; ++i) Bt.at(a, i) = P.at(i, piv[a]);
        const std::vector<uint32_t> rp = engine::elim::rref(F, engine::view(Bt)).pivots;
        ENGINE_CHECK_MSG(rp.size() == piv.size(), "SelectImage: rho(e)[:, J] is not of full column rank mod ell");
        for (uint32_t x : piv) s.J.push_back(blks[x / d] * d + x % d);
        for (uint32_t x : rp) s.R.push_back(blks[x / d] * d + x % d);
    }
    std::sort(s.J.begin(), s.J.end());
    std::sort(s.R.begin(), s.R.end());
    for (uint32_t i = 0; i < s.J.size(); ++i) s.jpos[s.J[i]] = static_cast<int32_t>(i);
    for (uint32_t i = 0; i < s.R.size(); ++i) s.rpos[s.R[i]] = static_cast<int32_t>(i);
    return s;
}

template<class Elt>
void AddRhoBlock(const engine::ModCtx<Elt>& ctx, std::vector<Elt>& out, uint32_t ld, uint32_t r0, uint32_t c0,
                 const TreeModule<Elt>& M, const QGroupRingElt& a, const ImageSelection& rows, const ImageSelection& cols) {
    const uint32_t d = M.d;
    for (const auto& [g, r] : a.terms) {
        const Elt coef = reduce(ctx, r);
        if (coef == 0) continue;
        const BlockMonomial<Elt>& B = M.rho(mat_of(g));
        for (uint32_t j = 0; j < B.m; ++j) {
            const uint32_t cb = B.col[j];
            const std::vector<Elt>& blk = B.blocks[B.blk[j]];
            for (uint32_t i = 0; i < d; ++i) {
                const int32_t pr = rows.rpos[j * d + i];
                if (pr < 0) continue;
                Elt* orow = out.data() + static_cast<size_t>(r0 + static_cast<uint32_t>(pr)) * ld + c0;
                const Elt* brow = blk.data() + static_cast<size_t>(i) * d;
                const int32_t* cp = cols.jpos.data() + static_cast<size_t>(cb) * d;
                for (uint32_t s = 0; s < d; ++s) {
                    const int32_t pc = cp[s];
                    if (pc < 0 || brow[s] == 0) continue;
                    orow[pc] = ctx.add(orow[pc], ctx.mul(coef, brow[s]));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the group-ring elements
// ---------------------------------------------------------------------------
namespace tree_elt {
QGroupRingElt one() { return QGroupRingElt::unit(Mat22::I()); }
QGroupRingElt g(const Mat22& m, int64_t num, int64_t den) { return QGroupRingElt::unit(m, Rat::of(num, den)); }
QGroupRingElt S() { return g(Mat22::S()); }
QGroupRingElt U() { return g(Mat22::ST()); }
QGroupRingElt c() { return g(Mat22::I().neg()); }
QGroupRingElt sum(const QGroupRingElt& a, const QGroupRingElt& b, int64_t sb) {
    QGroupRingElt x = a;
    x.add(b, Rat::of(sb));
    return x;
}
QGroupRingElt eU2() {
    const Mat22 U2 = Mat22::ST() * Mat22::ST();
    QGroupRingElt x = g(Mat22::I(), 1, 3);
    x.add(g(U2, 1, 3));
    x.add(g(U2 * U2, 1, 3));
    return x;
}
QGroupRingElt NS() {
    QGroupRingElt x;
    Mat22 p = Mat22::I();
    for (int j = 0; j < 4; ++j) {
        x.add(g(p));
        p = p * Mat22::S();
    }
    return x;
}
QGroupRingElt eS4() {
    QGroupRingElt x;
    x.add(NS(), Rat::of(1, 4));
    return x;
}
QGroupRingElt eU6() {
    QGroupRingElt x;
    Mat22 p = Mat22::I();
    for (int j = 0; j < 6; ++j) {
        x.add(g(p, 1, 6));
        p = p * Mat22::ST();
    }
    return x;
}
QGroupRingElt ec() {
    QGroupRingElt x = g(Mat22::I(), 1, 2);
    x.add(g(Mat22::I().neg(), 1, 2));
    return x;
}
}  // namespace tree_elt

uint32_t TreeCochainRank(uint32_t ell, uint32_t q, uint32_t r1, uint32_t rS, uint32_t rU, uint32_t rE) {
    if (ell == 2) return q == 0 ? r1 + rU : 2 * r1 + rU;
    if (ell == 3) return q == 0 ? rS + rE : (q == 1 ? 2 * rE : rE);
    return q == 0 ? rS + rU : (q == 1 ? rE : 0u);
}

// ---------------------------------------------------------------------------
// the closed forms of [C, prop:isomodeltree](2)
// ---------------------------------------------------------------------------
template<class Elt>
uint32_t TreeLocalForm<Elt>::cochain_rank(uint32_t q) const {
    return TreeCochainRank(ell, q, D, sel_S.rank(), sel_U.rank(), sel_E.rank());
}

template<class Elt>
uint32_t TreeLocalForm<Elt>::representative(uint32_t q) const {
    if (ell == 2) return q <= 2 ? q : (q % 2 == 1 ? 1u : 2u);    // (2)(d): d^{q+2} = d^q, q >= 1
    if (ell == 3) return q <= 3 ? q : (q % 2 == 0 ? 2u : 3u);    // (2)(d): d^{q+2} = d^q, q >= 2
    return q;
}

template<class Elt>
TreeLocalForm<Elt> MakeTreeLocalForm(uint32_t local_ell, const engine::ModCtx<Elt>& ctx, const TreeModule<Elt>& M,
                                     uint32_t q_top) {
    using namespace tree_elt;
    ENGINE_CHECK_MSG(local_ell == 2 || local_ell == 3 || local_ell >= 5, "MakeTreeLocalForm: ell = 2, 3 or >= 5");
    // [C, def:isomodeltree]: every |Q_x| must be a unit of the ring (an O_ell-algebra): q_ell = 3, 4, 12.
    const uint32_t q_ell = local_ell == 2 ? 3u : (local_ell == 3 ? 4u : 12u);
    const uint32_t res = static_cast<uint32_t>(ctx.ell);
    ENGINE_CHECK_MSG(std::gcd(res, q_ell) == 1, "MakeTreeLocalForm: the ring is not an O_ell-algebra (q_ell not a unit)");
    ENGINE_CHECK_MSG(local_ell >= 5 || q_top <= 64, "MakeTreeLocalForm: q_top");
    TreeLocalForm<Elt> T;
    T.ell = local_ell;
    T.D = M.dim();
    T.q_top = local_ell >= 5 ? 1u : q_top;
    if (local_ell == 2) {
        T.sel_S = SelectImage(ctx, M, one());
        T.sel_U = SelectImage(ctx, M, eU2());
        T.sel_E = T.sel_S;
    } else if (local_ell == 3) {
        T.sel_S = SelectImage(ctx, M, eS4());
        T.sel_E = SelectImage(ctx, M, ec());
        T.sel_U = T.sel_E;
    } else {
        T.sel_S = SelectImage(ctx, M, eS4());
        T.sel_U = SelectImage(ctx, M, eU6());
        T.sel_E = SelectImage(ctx, M, ec());
    }
    return T;
}

template<class Elt>
std::vector<Elt> TreeLocalForm<Elt>::differential(const engine::ModCtx<Elt>& ctx, const TreeModule<Elt>& M,
                                                   uint32_t q) const {
    using namespace tree_elt;
    ENGINE_CHECK_MSG(q < q_top, "TreeLocalForm::differential: q >= q_top");
    const uint32_t rows = cochain_rank(q + 1), cols = cochain_rank(q);
    std::vector<Elt> out(static_cast<size_t>(rows) * cols, Elt(0));
    if (rows == 0 || cols == 0) return out;
    const QGroupRingElt I = one(), Sg = S(), Ug = U(), Cg = c();
    const TreeFault fault = ActiveTreeFault();   // None outside the validation's fault arms
    auto add = [&](uint32_t r0, uint32_t c0, const QGroupRingElt& a, const ImageSelection& rsel, const ImageSelection& csel) {
        AddRhoBlock(ctx, out, cols, r0, c0, M, a, rsel, csel);
    };
    if (ell == 2) {
        // [C, prop:isomodeltree](2)(a); components (x on F^S, y on F^U, z on F^e of degree q - 1).
        const uint32_t rU = sel_U.rank();
        const QGroupRingElt eU = eU2();
        const QGroupRingElt one_minus_c = sum(I, Cg, -1), one_plus_c = sum(I, Cg, 1);
        const uint32_t rowS = 0, rowU = D, rowE = D + rU;
        const uint32_t colX = 0, colY = D, colZ = D + rU;
        const QGroupRingElt minus_eU = sum(QGroupRingElt{}, eU, -1);
        if (q == 0) {
            add(rowS, colX, sum(I, Sg, -1), sel_S, sel_S);              // (1 - S) x
            add(rowU, colY, one_minus_c * eU, sel_U, sel_U);            // (1 - c) y
            add(rowE, colX, I, sel_E, sel_S);                           // x
            if (fault != TreeFault::Ell2D0DropY) add(rowE, colY, minus_eU, sel_E, sel_U);   // - y
        } else if (q % 2 == 1) {
            add(rowS, colX, NS(), sel_S, sel_S);                        // N_S x
            add(rowU, colY, fault == TreeFault::Ell2OddNoEU ? one_plus_c : one_plus_c * eU, sel_U, sel_U);   // (1 + c) y
            add(rowE, colX, sum(I, Sg, fault == TreeFault::Ell2OddEdgeSign ? -1 : 1), sel_E, sel_S);   // (1 + S) x
            add(rowE, colY, minus_eU, sel_E, sel_U);                    // - y
            add(rowE, colZ, sum(QGroupRingElt{}, one_minus_c, fault == TreeFault::Ell2OddZSign ? 1 : -1), sel_E, sel_E);   // - (1 - c) z
        } else {
            add(rowS, colX, sum(I, Sg, -1), sel_S, sel_S);              // (1 - S) x
            add(rowU, colY, one_minus_c * eU, sel_U, sel_U);            // (1 - c) y
            add(rowE, colX, I, sel_E, sel_S);                           // x
            add(rowE, colY, minus_eU, sel_E, sel_U);                    // - y
            add(rowE, colZ, sum(QGroupRingElt{}, fault == TreeFault::Ell2EvenZ ? one_minus_c : one_plus_c, -1), sel_E, sel_E);   // - (1 + c) z
        }
    } else if (ell == 3) {
        // [C, prop:isomodeltree](2)(b): C^0 = M^<S> (+) M^{-I}, C^1 = M^{-I} (+) M^{-I} (U column of degree 1, edge),
        // C^q = M^{-I} (q >= 2).
        const QGroupRingElt e = ec();
        const uint32_t rS = sel_S.rank(), rE = sel_E.rank();
        const QGroupRingElt one_minus_U = sum(I, Ug, -1) * e;
        QGroupRingElt nU = sum(I, Ug, 1);
        nU.add(Ug * Ug);
        nU = nU * e;
        if (q == 0) {
            add(0, rS, one_minus_U, sel_U, sel_E);                       // U column: (1 - U) y
            add(rE, 0, fault == TreeFault::Ell3D0EdgeIdem ? ec() : eS4(), sel_E, sel_S);   // edge: x
            add(rE, rS, sum(QGroupRingElt{}, e, -1), sel_E, sel_E);      //       - y
        } else if (q == 1) {
            add(0, 0, nU, sel_U, sel_U);                                 // (1 + U + U^2) y; the edge component maps to 0
        } else if (q % 2 == 0) {
            add(0, 0, one_minus_U, sel_U, sel_U);
        } else {
            add(0, 0, fault == TreeFault::Ell3OddNorm ? one_minus_U : nU, sel_U, sel_U);
        }
    } else {
        // [C, prop:isomodeltree](2)(c): d^0(x, y) = x - y on M^<S> (+) M^<U> -> M^{-I}.
        ENGINE_CHECK(q == 0);
        add(0, 0, eS4(), sel_E, sel_S);
        if (fault != TreeFault::BredonDropY) add(0, sel_S.rank(), sum(QGroupRingElt{}, eU6(), -1), sel_E, sel_U);
    }
    return out;
}

#define TREE_COCHAINS_INSTANTIATE(Elt)                                                                                     \
    template TreeModule<Elt> TreeSymModule<Elt>(const engine::ModCtx<Elt>&, uint32_t);                                     \
    template TreeModule<Elt> TreeModuleFromDense<Elt>(const CoefficientModule<Elt>&);                                      \
    template ImageSelection SelectImage<Elt>(const engine::ModCtx<Elt>&, const TreeModule<Elt>&, const QGroupRingElt&);     \
    template void AddRhoBlock<Elt>(const engine::ModCtx<Elt>&, std::vector<Elt>&, uint32_t, uint32_t, uint32_t,              \
                                   const TreeModule<Elt>&, const QGroupRingElt&, const ImageSelection&, const ImageSelection&); \
    template struct TreeLocalForm<Elt>;                                                                                    \
    template TreeLocalForm<Elt> MakeTreeLocalForm<Elt>(uint32_t, const engine::ModCtx<Elt>&, const TreeModule<Elt>&, uint32_t);

TREE_COCHAINS_INSTANTIATE(uint32_t)
TREE_COCHAINS_INSTANTIATE(uint64_t)

}  // namespace integral_model
