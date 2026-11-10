// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT D5: the chain-level tree models and the comparisons of [C, prop:isomodeltree](3), (4).  See IsotropyModelTree.h.
#include "IsotropyModelTree.h"

#include <algorithm>
#include <sstream>

#include "PeriodicColumn.h"
#include "TreeCochains.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Kernels.h"
#include "../../../ModArithmetic/DenseView.h"

namespace integral_model {

using hecke::Mat22;
namespace te = tree_elt;

namespace {

void chain_add(LChain& x, uint32_t cell, const QGroupRingElt& v, const Rat& scale = Rat::of(1)) {
    if (v.is_zero() || scale.is_zero()) return;
    auto& slot = x[cell];
    slot.add(v, scale);
    if (slot.is_zero()) x.erase(cell);
}
void chain_add(LChain& x, const LChain& y, const Rat& scale = Rat::of(1)) {
    for (const auto& [c, v] : y) chain_add(x, c, v, scale);
}
bool chain_zero(const LChain& x) {
    for (const auto& [c, v] : x)
        if (!v.is_zero()) return false;
    return true;
}
LChain lmul(const QGroupRingElt& a, const LChain& x) {
    LChain out;
    for (const auto& [c, v] : x) chain_add(out, c, a * v);
    return out;
}
QGroupRingElt scaled(const QGroupRingElt& a, const Rat& r) {
    QGroupRingElt x;
    x.add(a, r);
    return x;
}
Rat rpow(int64_t b, uint32_t e) {
    Rat r = Rat::of(1);
    for (uint32_t i = 0; i < e; ++i) r = r * Rat::of(b);
    return r;
}
Rat rpow_inv(int64_t b, uint32_t e) {
    Rat r = Rat::of(1);
    for (uint32_t i = 0; i < e; ++i) r = r * Rat::of(1, b);
    return r;
}
std::vector<Mat22> powers(const Mat22& g, uint32_t m) {
    std::vector<Mat22> p;
    Mat22 x = Mat22::I();
    for (uint32_t i = 0; i < m; ++i) {
        p.push_back(x);
        x = x * g;
    }
    return p;
}
// The column contraction of a tree column (differentials r_{1-g} odd, r_N even): (4)(b), the code's h negated from even
// degrees.  m = |<g>| (the free columns only).
QGroupRingElt tree_contr(const QGroupRingElt& a, const std::vector<Mat22>& pw, uint32_t t) {
    const uint32_t m = static_cast<uint32_t>(pw.size());
    QGroupRingElt out;
    for (const auto& [g, r] : a.terms) {
        const auto [c0, j] = decompose(g, pw);
        const Mat22 C = mat_of(c0);
        if (t % 2 == 0) {
            for (uint32_t i = 0; i < j; ++i) out.add(key_of(C * pw[i]), -r);
        } else if (j == m - 1) {
            out.add(c0, r);
        }
    }
    return out;
}
bool integral(const QGroupRingElt& a) {
    for (const auto& kv : a.terms)
        if (kv.second.den != 1) return false;
    return true;
}
}  // namespace

// ---------------------------------------------------------------------------
// TreeModelChain
// ---------------------------------------------------------------------------
TreeModelChain::TreeModelChain(uint32_t ell, uint32_t q_top) : ell_(ell), q_top_(q_top) {
    ENGINE_CHECK_MSG(ell == 0 || ell == 2 || ell == 3 || ell >= 5, "TreeModelChain: ell = 0 (free), 2, 3 or >= 5");
    const Mat22 S = Mat22::S(), U = Mat22::ST(), C = Mat22::I().neg();
    pow_['S'] = powers(S, 4);
    pow_['U'] = powers(U, 6);
    pow_['E'] = powers(C, 2);
    if (ell == 0) {
        porder_ = {{'S', 4}, {'U', 6}, {'E', 2}};
        e_ = {{'S', te::one()}, {'U', te::one()}, {'E', te::one()}};
    } else if (ell == 2) {
        porder_ = {{'S', 4}, {'U', 2}, {'E', 2}};
        e_ = {{'S', te::one()}, {'U', te::eU2()}, {'E', te::one()}};
    } else if (ell == 3) {
        porder_ = {{'S', 1}, {'U', 3}, {'E', 1}};
        e_ = {{'S', te::eS4()}, {'U', te::ec()}, {'E', te::ec()}};
    } else {
        porder_ = {{'S', 1}, {'U', 1}, {'E', 1}};
        e_ = {{'S', te::eS4()}, {'U', te::eU6()}, {'E', te::ec()}};
    }
    cells_.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        if (col_exists('S', q)) cells_[q].push_back({'S', q});
        if (col_exists('U', q)) cells_[q].push_back({'U', q});
        if (q >= 1 && col_exists('E', q - 1)) cells_[q].push_back({'E', q - 1});
    }
    // the lift phi'_t: F^E_t -> F^S_t (+) F^U_t (def:isomodeltree(3), (4))
    auto lift = [&](uint32_t t) -> std::pair<QGroupRingElt, QGroupRingElt> {
        const QGroupRingElt I = te::one(), Sg = te::S(), Ug = te::U();
        if (ell == 0) {
            if (t % 2 == 0) return {I, scaled(I, Rat::of(-1))};
            QGroupRingElt b = te::sum(I, Ug);
            b.add(Ug * Ug);
            return {te::sum(I, Sg), scaled(b, Rat::of(-1))};
        }
        if (ell == 2) return {t % 2 == 0 ? I : te::sum(I, Sg), scaled(te::eU2(), Rat::of(-1))};
        ENGINE_CHECK(t == 0);
        if (ell == 3) return {te::eS4(), scaled(te::ec(), Rat::of(-1))};
        return {te::eS4(), scaled(te::eU6(), Rat::of(-1))};
    };
    diff_.assign(q_top + 1, {});
    for (uint32_t q = 1; q <= q_top; ++q) {
        for (uint32_t ci = 0; ci < cells_[q].size(); ++ci) {
            const TreeCell& y = cells_[q][ci];
            if (y.x != 'E') {
                const int64_t r = index(q - 1, y.x, y.t - 1);
                if (y.t >= 1 && r >= 0) diff_[q].push_back({static_cast<uint32_t>(r), ci, col_diff(y.x, y.t)});
            } else {
                const auto [a, b] = lift(y.t);
                const int64_t rs = index(q - 1, 'S', y.t), ru = index(q - 1, 'U', y.t);
                if (rs >= 0) diff_[q].push_back({static_cast<uint32_t>(rs), ci, a});
                if (ru >= 0) diff_[q].push_back({static_cast<uint32_t>(ru), ci, b});
                if (y.t >= 1) {
                    const int64_t re = index(q - 1, 'E', y.t - 1);
                    if (re >= 0) diff_[q].push_back({static_cast<uint32_t>(re), ci, scaled(col_diff('E', y.t), Rat::of(-1))});
                }
            }
        }
    }
    by_col_.assign(q_top + 1, {});
    for (uint32_t q = 1; q <= q_top; ++q) {
        by_col_[q].assign(cells_[q].size(), {});
        for (const auto& e : diff_[q]) by_col_[q][e.col].push_back(&e);
    }
}

bool TreeModelChain::col_exists(char x, uint32_t t) const { return t == 0 || porder_.at(x) > 1; }

QGroupRingElt TreeModelChain::col_diff(char x, uint32_t t) const {
    const QGroupRingElt& e = e_.at(x);
    const auto& pw = pow_.at(x);
    if (t % 2 == 1) return te::sum(te::one(), QGroupRingElt::unit(pw[1]), -1) * e;   // (1 - g) e
    QGroupRingElt N;
    for (uint32_t j = 0; j < porder_.at(x); ++j) N.add(QGroupRingElt::unit(pw[j]));
    return N * e;   // N'_x
}

int64_t TreeModelChain::index(uint32_t q, char x, uint32_t t) const {
    if (q > q_top_) return -1;
    for (uint32_t i = 0; i < cells_[q].size(); ++i)
        if (cells_[q][i].x == x && cells_[q][i].t == t) return i;
    return -1;
}
const QGroupRingElt& TreeModelChain::idem(uint32_t q, uint32_t cell) const { return e_.at(cells_.at(q).at(cell).x); }
const std::vector<Mat22>& TreeModelChain::column_pow(char x) const { return pow_.at(x); }
const QGroupRingElt& TreeModelChain::column_e(char x) const { return e_.at(x); }

LChain TreeModelChain::apply_d(uint32_t q, const LChain& x) const {
    ENGINE_CHECK_MSG(q >= 1 && q <= q_top_, "TreeModelChain::apply_d: degree out of range");
    LChain out;
    for (const auto& [col, v] : x) {
        if (v.is_zero()) continue;
        for (const LEntry* e : by_col_[q].at(col)) chain_add(out, e->row, v * e->coeff);
    }
    return out;
}

LChain TreeModelChain::contract(uint32_t d, const LChain& z0) const {
    ENGINE_CHECK_MSG(ell_ == 0, "TreeModelChain::contract: the solver of (4)(b) is on the free tree model");
    ENGINE_CHECK_MSG(d + 1 <= q_top_, "TreeModelChain::contract: degree beyond the built range");
    LChain z = z0, k;
    // the edge column (column index s = 1, global sign -1) first
    if (d >= 1) {
        const int64_t ie = index(d, 'E', d - 1);
        auto it = ie >= 0 ? z.find(static_cast<uint32_t>(ie)) : z.end();
        if (it != z.end() && !it->second.is_zero()) {
            const QGroupRingElt h = scaled(tree_contr(it->second, pow_.at('E'), d - 1), Rat::of(-1));
            LChain kk;
            chain_add(kk, static_cast<uint32_t>(index(d + 1, 'E', d)), h);
            chain_add(k, kk);
            chain_add(z, apply_d(d + 1, kk), Rat::of(-1));
        }
    }
    LChain kk;
    for (char x : {'S', 'U'}) {
        const int64_t ix = index(d, x, d);
        auto it = ix >= 0 ? z.find(static_cast<uint32_t>(ix)) : z.end();
        if (it == z.end() || it->second.is_zero()) continue;
        const QGroupRingElt h = tree_contr(it->second, pow_.at(x), d);
        chain_add(kk, static_cast<uint32_t>(index(d + 1, x, d + 1)), h);
    }
    chain_add(k, kk);
    if (!kk.empty()) chain_add(z, apply_d(d + 1, kk), Rat::of(-1));
    ENGINE_CHECK_MSG(chain_zero(z), "TreeModelChain::contract: the descent left a remainder (z not a cycle of ker eps_T)");
    return k;
}

std::map<std::pair<char, GKey>, Rat> TreeModelChain::augment(uint32_t q, const LChain& x) const {
    std::map<std::pair<char, GKey>, Rat> out;
    for (const auto& [ci, v] : x) {
        const TreeCell& c = cells_.at(q).at(ci);
        if (c.t != 0) continue;
        for (const auto& [g, r] : v.terms) {
            auto& slot = out[{c.x, decompose(g, pow_.at(c.x)).first}];
            slot = slot + r;
        }
    }
    for (auto it = out.begin(); it != out.end();) it = it->second.is_zero() ? out.erase(it) : std::next(it);
    return out;
}

TreeCertificates TreeModelChain::certify() const {
    TreeCertificates cert;
    cert.d_squared_zero = cert.well_defined = cert.lift_over_boundary = true;
    auto fail = [&](const std::string& w) {
        if (cert.first_failure.empty()) cert.first_failure = w;
    };
    auto norm = [&](uint32_t q, const LChain& x) {
        LChain o;
        for (const auto& [c, v] : x) chain_add(o, c, v * idem(q, c));
        return o;
    };
    for (uint32_t q = 2; q <= q_top_; ++q)
        for (uint32_t c = 0; c < cells_[q].size(); ++c) {
            const LChain dd = norm(q - 2, apply_d(q - 1, apply_d(q, LChain{{c, idem(q, c)}})));
            ++cert.checked;
            if (!chain_zero(dd)) {
                cert.d_squared_zero = false;
                fail("d o d != 0 at q=" + std::to_string(q));
            }
        }
    for (uint32_t q = 1; q <= q_top_; ++q)
        for (const auto& e : diff_[q]) {
            ++cert.checked;
            if (idem(q, e.col) * e.coeff * idem(q - 1, e.row) != e.coeff) {
                cert.well_defined = false;
                fail("an entry is not in e_y R[Gamma] e_x at q=" + std::to_string(q));
            }
        }
    // the lift augments to <S> - <U> in degree 0 (aug a_0 = 1, aug b_0 = -1)
    if (q_top_ >= 1) {
        const int64_t ie = index(1, 'E', 0);
        Rat aS = Rat::of(0), aU = Rat::of(0);
        for (const auto& e : diff_[1])
            if (static_cast<int64_t>(e.col) == ie)
                for (const auto& kv : e.coeff.terms) (cells_[0][e.row].x == 'S' ? aS : aU) = (cells_[0][e.row].x == 'S' ? aS : aU) + kv.second;
        ++cert.checked;
        if (!(aS == Rat::of(1) && aU == Rat::of(-1))) {
            cert.lift_over_boundary = false;
            fail("the lift does not lift the tree boundary");
        }
    }
    return cert;
}

template<class Elt>
engine::DenseBuf<Elt> TreeModelChain::coboundary(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, uint32_t q) const {
    ENGINE_CHECK_MSG(q + 1 <= q_top_, "TreeModelChain::coboundary: q + 1 > q_top");
    const uint32_t D = M.dim;
    engine::DenseBuf<Elt> out(rank(q + 1) * D, rank(q) * D);
    std::map<GKey, engine::DenseBuf<Elt>> memo;
    for (const auto& e : diff_[q + 1]) {
        const engine::DenseBuf<Elt> R = RhoQ(ctx, e.coeff, M, memo);
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) {
                Elt& d = out.at(e.col * D + i, e.row * D + j);
                d = ctx.add(d, R.at(i, j));
            }
    }
    return out;
}

template<class Elt>
engine::DenseBuf<Elt> TreeModelChain::projector(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, uint32_t q) const {
    const uint32_t D = M.dim;
    engine::DenseBuf<Elt> out(rank(q) * D, rank(q) * D);
    std::map<GKey, engine::DenseBuf<Elt>> memo;
    for (uint32_t c = 0; c < rank(q); ++c) {
        const engine::DenseBuf<Elt> R = RhoQ(ctx, idem(q, c), M, memo);
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) out.at(c * D + i, c * D + j) = R.at(i, j);
    }
    return out;
}

// ---------------------------------------------------------------------------
// maps
// ---------------------------------------------------------------------------
LChain ApplyMap(const TreeMap& f, uint32_t q, const LChain& x) {
    LChain out;
    for (const auto& [y, a] : x) {
        if (a.is_zero()) continue;
        chain_add(out, lmul(a, f.value.at(q).at(y)));
    }
    return out;
}

TreeMap ComposeMaps(const TreeMap& first, const TreeMap& second) {
    TreeMap out;
    out.degree_shift = first.degree_shift + second.degree_shift;
    const size_t Q = first.value.size();
    for (size_t q = 0; q < Q; ++q) {
        if (q + first.degree_shift >= second.value.size()) break;
        std::vector<LChain> row;
        for (const auto& img : first.value[q]) row.push_back(ApplyMap(second, static_cast<uint32_t>(q + first.degree_shift), img));
        out.value.push_back(std::move(row));
    }
    return out;
}

namespace {
uint32_t q_order(uint32_t ell, char x) {
    // |Q_x| (def:isomodeltree's table)
    if (ell == 2) return x == 'U' ? 3u : 1u;
    if (ell == 3) return x == 'S' ? 4u : 2u;
    return x == 'S' ? 4u : (x == 'U' ? 6u : 2u);
}
int64_t kj(uint32_t j) {   // (3^j - 1) / 2
    int64_t p = 1;
    for (uint32_t i = 0; i < j; ++i) p *= 3;
    return (p - 1) / 2;
}
QGroupRingElt W2() {   // w = (2 + 2U + U^2 + U^3)/3 - (1 + c) e_U   ((3)(a))
    const QGroupRingElt U = te::U();
    QGroupRingElt a = scaled(te::one(), Rat::of(2, 3));
    a.add(U, Rat::of(2, 3));
    a.add(U * U, Rat::of(1, 3));
    a.add(U * U * U, Rat::of(1, 3));
    a.add(te::sum(te::one(), te::c()) * te::eU2(), Rat::of(-1));
    return a;
}
QGroupRingElt V5() {   // (2 + 2U + U^2 + U^3)/3
    const QGroupRingElt U = te::U();
    QGroupRingElt a = scaled(te::one(), Rat::of(2, 3));
    a.add(U, Rat::of(2, 3));
    a.add(U * U, Rat::of(1, 3));
    a.add(U * U * U, Rat::of(1, 3));
    return a;
}
}  // namespace

TreeMap PhiTreeEll(uint32_t ell, uint32_t q_top) {
    const TreeModelChain F(0, q_top), L(ell, q_top);
    TreeMap out;
    out.value.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        out.value[q].assign(F.rank(q), {});
        for (uint32_t y = 0; y < F.rank(q); ++y) {
            const TreeCell c = F.cells(q)[y];
            LChain img;
            if (c.x != 'E') {
                const int64_t t = L.index(q, c.x, q);
                if (t >= 0) chain_add(img, static_cast<uint32_t>(t), scaled(L.column_e(c.x), rpow(q_order(ell, c.x), q / 2)));
            } else {
                const uint32_t tt = c.t;
                const int64_t t = L.index(q, 'E', tt);
                if (t >= 0) chain_add(img, static_cast<uint32_t>(t), scaled(L.column_e('E'), rpow(q_order(ell, 'E'), tt / 2)));
                const int64_t u = L.index(q, 'U', q);
                if (ell == 2 && tt >= 1 && u >= 0)
                    chain_add(img, static_cast<uint32_t>(u), scaled(te::eU2(), Rat::of(-kj((tt + 1) / 2))));
                if (ell == 3 && tt % 2 == 1 && u >= 0)
                    chain_add(img, static_cast<uint32_t>(u), scaled(te::ec(), Rat::of(-(int64_t(1) << ((tt - 1) / 2)))));
            }
            out.value[q][y] = std::move(img);
        }
    }
    return out;
}

TreeMap PsiTreeEll(uint32_t ell, uint32_t q_top) {
    const TreeModelChain F(0, q_top), L(ell, q_top);
    TreeMap out;
    out.value.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        out.value[q].assign(L.rank(q), {});
        for (uint32_t y = 0; y < L.rank(q); ++y) {
            const TreeCell c = L.cells(q)[y];
            LChain img;
            if (c.x != 'E') {
                chain_add(img, static_cast<uint32_t>(F.index(q, c.x, q)), scaled(L.column_e(c.x), rpow_inv(q_order(ell, c.x), q / 2)));
            } else {
                const uint32_t t = c.t;
                chain_add(img, static_cast<uint32_t>(F.index(q, 'E', t)), scaled(L.column_e('E'), rpow_inv(q_order(ell, 'E'), t / 2)));
                if (ell == 2) {
                    const uint32_t m = (t + 1) / 2;
                    QGroupRingElt kap = scaled(te::eU2(), (Rat::of(1) + (-rpow_inv(3, m))) * Rat::of(1, 2));
                    if (t % 2 == 0) kap.add(W2());
                    chain_add(img, static_cast<uint32_t>(F.index(q, 'U', q)), kap);
                }
                if (ell == 3 && t == 0) chain_add(img, static_cast<uint32_t>(F.index(1, 'S', 1)), scaled(te::ec(), Rat::of(-1, 2)));
                if (ell >= 5 && t == 0) {
                    chain_add(img, static_cast<uint32_t>(F.index(1, 'S', 1)), scaled(te::ec(), Rat::of(-1, 2)));
                    chain_add(img, static_cast<uint32_t>(F.index(1, 'U', 1)), te::ec() * V5());
                }
            }
            out.value[q][y] = std::move(img);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// (4)(a): the tables
// ---------------------------------------------------------------------------
RetractionTables DefaultRetractionTables() {
    RetractionTables T;
    const QGroupRingElt I = te::one(), Sg = te::S();
    const QGroupRingElt mI = scaled(I, Rat::of(-1)), mS = scaled(Sg, Rat::of(-1));
    T.r0[{0}] = {'U', I};
    T.r0[{1}] = {'S', I};
    T.r0[{2}] = {'S', I};
    T.r0[{4}] = {'U', I};
    T.r0[{6}] = {'S', I};
    T.r1[{0, 1}] = I;
    T.r1[{0, 4}] = QGroupRingElt{};
    T.r1[{1, 2}] = QGroupRingElt{};
    T.r1[{1, 4}] = mI;
    T.r1[{1, 6}] = QGroupRingElt{};
    T.r1[{2, 3}] = mS;
    T.r1[{2, 5}] = mS;
    T.r1[{2, 6}] = QGroupRingElt{};
    T.r1[{4, 6}] = I;
    T.G[{0}] = {{{0, 4}, mI}};
    T.G[{1}] = {{{1, 6}, mI}};
    T.G[{2}] = {{{2, 6}, mI}};
    T.G[{4}] = {};
    T.G[{6}] = {};
    T.G[{0, 1}] = {{{0, 1, 4}, I}, {{1, 4, 6}, mI}};
    T.G[{0, 4}] = {};
    T.G[{1, 2}] = {{{1, 2, 6}, I}};
    T.G[{1, 4}] = {{{1, 4, 6}, I}};
    T.G[{1, 6}] = {};
    T.G[{2, 3}] = {{{2, 3, 5}, I}, {{2, 5, 6}, I}};
    T.G[{2, 5}] = {{{2, 5, 6}, I}};
    T.G[{2, 6}] = {};
    T.G[{4, 6}] = {};
    for (const Simplex& f : std::vector<Simplex>{{0, 1, 4}, {1, 2, 6}, {1, 4, 6}, {2, 3, 5}, {2, 5, 6}}) T.G[f] = {};
    T.iota['S'] = {6};
    T.iota['U'] = {4};
    T.iota['E'] = {4, 6};
    return T;
}

// ---------------------------------------------------------------------------
// IntegralTreeComparison
// ---------------------------------------------------------------------------
namespace {
// s-chains of Xbulk on the representatives: Simplex -> coefficient; normalised by the stabiliser cosets.
using XChain = std::map<Simplex, QGroupRingElt>;
using TChain = std::map<char, QGroupRingElt>;
void xadd(XChain& x, const Simplex& s, const QGroupRingElt& v) {
    if (v.is_zero()) return;
    auto& slot = x[s];
    slot.add(v);
    if (slot.is_zero()) x.erase(s);
}
void tadd(TChain& x, char c, const QGroupRingElt& v) {
    if (v.is_zero()) return;
    auto& slot = x[c];
    slot.add(v);
    if (slot.is_zero()) x.erase(c);
}
}  // namespace

IntegralTreeComparison::IntegralTreeComparison(const IsotropyModel& desc, uint32_t q_top, const RetractionTables& tables)
    : q_top_(q_top), desc_(&desc), iso_(desc, 0), tree_(0, q_top + 1), tab_(tables) {
    ENGINE_CHECK_MSG(desc.is_shipped_model(), "IntegralTreeComparison: (4)(a)'s tables are on the def:model7 descriptor");
    ENGINE_CHECK_MSG(iso_.q_top() >= q_top + 1, "IntegralTreeComparison: the descriptor must be built to degree q_top + 1");
    check_tables();
    auto rep_index = [&](uint32_t s, const Simplex& sim) -> uint32_t {
        for (uint32_t r = 0; r < desc.reps(s).size(); ++r)
            if (desc.reps(s)[r] == sim) return r;
        ENGINE_CHECK_MSG(false, "IntegralTreeComparison: a table simplex is not a representative");
        return 0;
    };
    auto iso_cell = [&](uint32_t s, uint32_t rep) { return static_cast<uint32_t>(iso_.cell_index(s, rep, 0)); };
    // Phi^Z_T := (L) for f = r_T o cmpI along eps_T
    phi_.value.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        phi_.value[q].assign(iso_.rank(q), {});
        for (uint32_t y = 0; y < iso_.rank(q); ++y) {
            const LCell& c = iso_.cells(q)[y];
            LChain ys;
            if (c.t == 0 && c.s == 0) {
                const auto& v = tab_.r0.at(desc.reps(0)[c.rep]);
                chain_add(ys, static_cast<uint32_t>(tree_.index(0, v.first, 0)), v.second);
            } else if (c.t == 0 && c.s == 1) {
                chain_add(ys, static_cast<uint32_t>(tree_.index(1, 'E', 0)), tab_.r1.at(desc.reps(1)[c.rep]));
            }
            LChain val = ys;
            if (q >= 1) {
                LChain z = ApplyMap(phi_, q - 1, iso_.apply_d(q, LChain{{y, te::one()}}));
                if (!ys.empty()) chain_add(z, tree_.apply_d(q, ys), Rat::of(-1));
                chain_add(val, tree_.contract(q - 1, z));
            }
            phi_.value[q][y] = std::move(val);
        }
    }
    // Psi^Z_T := (L) for f = i_T o eps_T along cmpI
    psi_.value.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        psi_.value[q].assign(tree_.rank(q), {});
        for (uint32_t y = 0; y < tree_.rank(q); ++y) {
            const TreeCell& c = tree_.cells(q)[y];
            LChain ys;
            if (c.t == 0) {
                const Simplex& sim = tab_.iota.at(c.x);
                const uint32_t s = static_cast<uint32_t>(sim.size()) - 1;
                chain_add(ys, iso_cell(s, rep_index(s, sim)), te::one());
            }
            LChain val = ys;
            if (q >= 1) {
                LChain z = ApplyMap(psi_, q - 1, tree_.apply_d(q, LChain{{y, te::one()}}));
                if (!ys.empty()) chain_add(z, iso_.apply_d(q, ys), Rat::of(-1));
                chain_add(val, iso_.contract(q - 1, z));
            }
            psi_.value[q][y] = std::move(val);
        }
    }
    // H_T: d H + H d = Phi Psi - id on Iso_T (g = 0)
    const TreeMap PP = ComposeMaps(psi_, phi_);   // Phi o Psi : tree -> tree
    HT_.degree_shift = 1;
    HT_.value.assign(q_top, {});
    for (uint32_t q = 0; q < q_top; ++q) {
        HT_.value[q].assign(tree_.rank(q), {});
        for (uint32_t y = 0; y < tree_.rank(q); ++y) {
            LChain w = PP.value[q][y];
            chain_add(w, y, te::one(), Rat::of(-1));
            if (q >= 1) chain_add(w, ApplyMap(HT_, q - 1, tree_.apply_d(q, LChain{{y, te::one()}})), Rat::of(-1));
            HT_.value[q][y] = tree_.contract(q, w);
        }
    }
    // H_X: d H + H d = id - Psi Phi on Iso (over G_T)
    const TreeMap QQ = ComposeMaps(phi_, psi_);   // Psi o Phi : iso -> iso
    HX_.degree_shift = 1;
    HX_.value.assign(q_top, {});
    for (uint32_t q = 0; q < q_top; ++q) {
        HX_.value[q].assign(iso_.rank(q), {});
        for (uint32_t y = 0; y < iso_.rank(q); ++y) {
            const LCell& c = iso_.cells(q)[y];
            LChain ys;
            if (c.t == 0 && c.s <= 1)
                for (const auto& [sim, a] : tab_.G.at(desc.reps(c.s)[c.rep])) chain_add(ys, iso_cell(c.s + 1, rep_index(c.s + 1, sim)), a);
            LChain w;
            chain_add(w, y, te::one());
            chain_add(w, QQ.value[q][y], Rat::of(-1));
            if (q >= 1) chain_add(w, ApplyMap(HX_, q - 1, iso_.apply_d(q, LChain{{y, te::one()}})), Rat::of(-1));
            LChain z = w;
            if (!ys.empty()) chain_add(z, iso_.apply_d(q + 1, ys), Rat::of(-1));
            LChain val = ys;
            chain_add(val, iso_.contract(q, z));
            HX_.value[q][y] = std::move(val);
        }
    }
}

void IntegralTreeComparison::check_tables() {
    const IsotropyModel& desc = *desc_;
    tables_ok_ = true;
    auto fail = [&](const std::string& w) {
        tables_ok_ = false;
        if (tables_failure_.empty()) tables_failure_ = w;
    };
    auto dim_of = [](const Simplex& s) { return static_cast<uint32_t>(s.size()) - 1; };
    auto rep_index = [&](uint32_t s, const Simplex& sim) -> int64_t {
        for (uint32_t r = 0; r < desc.reps(s).size(); ++r)
            if (desc.reps(s)[r] == sim) return r;
        return -1;
    };
    // normal form of an Xbulk chain: (rep, coset) -> multiplicity (the stabiliser acts trivially on orientations)
    auto xnorm = [&](const XChain& x) {
        std::map<std::pair<Simplex, GKey>, Rat> out;
        for (const auto& [sim, a] : x) {
            const uint32_t s = dim_of(sim);
            const int64_t r = rep_index(s, sim);
            ENGINE_CHECK(r >= 0);
            const auto& pw = iso_.column(s, static_cast<uint32_t>(r)).pow;
            for (const auto& [g, c] : a.terms) {
                auto& slot = out[{sim, decompose(g, pw).first}];
                slot = slot + c;
            }
        }
        for (auto it = out.begin(); it != out.end();) it = it->second.is_zero() ? out.erase(it) : std::next(it);
        return out;
    };
    auto tnorm = [&](const TChain& x) {
        std::map<std::pair<char, GKey>, Rat> out;
        for (const auto& [c, a] : x)
            for (const auto& [g, r] : a.terms) {
                auto& slot = out[{c, decompose(g, tree_.column_pow(c)).first}];
                slot = slot + r;
            }
        for (auto it = out.begin(); it != out.end();) it = it->second.is_zero() ? out.erase(it) : std::next(it);
        return out;
    };
    // boundary of a representative in C(Xbulk), on representatives
    auto bdry = [&](const Simplex& sim) {
        XChain out;
        const uint32_t s = dim_of(sim);
        const int64_t r = rep_index(s, sim);
        if (s == 0) return out;
        for (const auto& [sign, ft] : desc.faces(s, static_cast<uint32_t>(r)))
            xadd(out, desc.reps(s - 1)[ft.rep], QGroupRingElt::unit(ft.g, Rat::of(sign)));
        return out;
    };
    auto bdry_chain = [&](const XChain& x) {
        XChain out;
        for (const auto& [sim, a] : x)
            for (const auto& [f, b] : bdry(sim)) xadd(out, f, a * b);
        return out;
    };
    auto r_chain = [&](const XChain& x) {
        TChain out;
        for (const auto& [sim, a] : x) {
            if (dim_of(sim) == 0) {
                const auto& v = tab_.r0.at(sim);
                tadd(out, v.first, a * v.second);
            } else if (dim_of(sim) == 1) {
                tadd(out, 'E', a * tab_.r1.at(sim));
            }
        }
        return out;
    };
    auto G_chain = [&](const XChain& x) {
        XChain out;
        for (const auto& [sim, a] : x)
            for (const auto& [t, b] : tab_.G.at(sim)) xadd(out, t, a * b);
        return out;
    };
    auto iota_chain = [&](const TChain& x) {
        XChain out;
        for (const auto& [c, a] : x) xadd(out, tab_.iota.at(c), a);
        return out;
    };
    // i_T a chain map: d iota(E) = iota(v_S - v_U)
    {
        ++tables_checked_;
        const XChain lhs = bdry_chain(XChain{{tab_.iota.at('E'), te::one()}});
        XChain rhs;
        xadd(rhs, tab_.iota.at('S'), te::one());
        xadd(rhs, tab_.iota.at('U'), scaled(te::one(), Rat::of(-1)));
        if (xnorm(lhs) != xnorm(rhs)) fail("(4)(a) i_T is not a chain map");
    }
    for (uint32_t s = 0; s < 3; ++s)
        for (const Simplex& sim : desc.reps(s)) {
            const XChain x{{sim, te::one()}};
            // r_T a chain map (s >= 1): r(d sigma) = d r(sigma)
            if (s >= 1) {
                ++tables_checked_;
                const TChain lhs = r_chain(bdry(sim));
                TChain rhs;
                if (s == 1) {
                    const QGroupRingElt& b = tab_.r1.at(sim);
                    tadd(rhs, 'S', b);
                    tadd(rhs, 'U', scaled(b, Rat::of(-1)));
                }
                if (tnorm(lhs) != tnorm(rhs)) fail("(4)(a) r_T is not a chain map at a representative of dimension " + std::to_string(s));
            } else {
                ++tables_checked_;
                Rat a = Rat::of(0);
                for (const auto& kv : tab_.r0.at(sim).second.terms) a = a + kv.second;
                if (a != Rat::of(1)) fail("(4)(a) r_T does not preserve the augmentation");
            }
            // id - i_T r_T = d G + G d
            ++tables_checked_;
            XChain lhs = x;
            if (s <= 1)
                for (const auto& [t, a] : iota_chain(r_chain(x)))
                    if (dim_of(t) == s) xadd(lhs, t, scaled(a, Rat::of(-1)));
            XChain rhs;
            if (s <= 1)
                for (const auto& [t, a] : bdry_chain(G_chain(x))) xadd(rhs, t, a);
            if (s >= 1)
                for (const auto& [t, a] : G_chain(bdry(sim))) xadd(rhs, t, a);
            if (xnorm(lhs) != xnorm(rhs)) fail("(4)(a) id - i_T r_T != dG + Gd at a representative of dimension " + std::to_string(s));
        }
    // r_T i_T = id
    for (char c : {'S', 'U', 'E'}) {
        ++tables_checked_;
        const TChain v = r_chain(XChain{{tab_.iota.at(c), te::one()}});
        if (tnorm(v) != tnorm(TChain{{c, te::one()}})) fail("(4)(a) r_T i_T != id");
    }
}

IsoTreeCertificates IntegralTreeComparison::certify() const {
    IsoTreeCertificates cert;
    cert.tables = tables_ok_;
    cert.checked = tables_checked_;
    cert.first_failure = tables_failure_;
    auto fail = [&](bool& flag, const std::string& w) {
        flag = false;
        if (cert.first_failure.empty()) cert.first_failure = w;
    };
    // the Iso of (4) is the code's IsotropyModel
    cert.iso_is_code_model = true;
    for (uint32_t q = 1; q <= iso_.q_top() && q <= desc_->q_top(); ++q) {
        std::map<std::pair<uint32_t, uint32_t>, QGroupRingElt> a, b;
        for (const auto& e : desc_->differential(q)) a[{e.row, e.col}].add(QGroupRingElt::from(e.coeff));
        for (const auto& e : iso_.differential(q)) b[{e.row, e.col}].add(e.coeff);
        ++cert.checked;
        if (a != b || iso_.rank(q) != desc_->rank(q)) fail(cert.iso_is_code_model, "LocalModel(desc, 0) != IsotropyModel at q=" + std::to_string(q));
    }
    cert.phi_chain = cert.psi_chain = cert.phi_over_r = cert.psi_over_iota = cert.integral = cert.H_T = cert.H_X = true;
    for (uint32_t q = 1; q <= q_top_; ++q) {
        for (uint32_t y = 0; y < iso_.rank(q); ++y) {
            ++cert.checked;
            LChain a = tree_.apply_d(q, phi_.value[q][y]);
            chain_add(a, ApplyMap(phi_, q - 1, iso_.apply_d(q, LChain{{y, te::one()}})), Rat::of(-1));
            if (!chain_zero(a)) fail(cert.phi_chain, "Phi^Z_T is not a chain map at q=" + std::to_string(q));
        }
        for (uint32_t y = 0; y < tree_.rank(q); ++y) {
            ++cert.checked;
            LChain a = iso_.apply_d(q, psi_.value[q][y]);
            chain_add(a, ApplyMap(psi_, q - 1, tree_.apply_d(q, LChain{{y, te::one()}})), Rat::of(-1));
            if (!chain_zero(a)) fail(cert.psi_chain, "Psi^Z_T is not a chain map at q=" + std::to_string(q));
        }
    }
    // over r_T, resp. i_T, in degrees 0 and 1
    for (uint32_t q = 0; q <= std::min<uint32_t>(1, q_top_); ++q) {
        for (uint32_t y = 0; y < iso_.rank(q); ++y) {
            const LCell& c = iso_.cells(q)[y];
            LChain f;
            if (c.t == 0 && c.s == 0) {
                const auto& v = tab_.r0.at(desc_->reps(0)[c.rep]);
                chain_add(f, static_cast<uint32_t>(tree_.index(0, v.first, 0)), v.second);
            } else if (c.t == 0 && c.s == 1) {
                chain_add(f, static_cast<uint32_t>(tree_.index(1, 'E', 0)), tab_.r1.at(desc_->reps(1)[c.rep]));
            }
            ++cert.checked;
            if (tree_.augment(q, phi_.value[q][y]) != tree_.augment(q, f)) fail(cert.phi_over_r, "Phi^Z_T is not over r_T o cmpI");
        }
        for (uint32_t y = 0; y < tree_.rank(q); ++y) {
            const TreeCell& c = tree_.cells(q)[y];
            LChain f;
            if (c.t == 0) {
                const Simplex& sim = tab_.iota.at(c.x);
                const uint32_t s = static_cast<uint32_t>(sim.size()) - 1;
                for (uint32_t r = 0; r < desc_->reps(s).size(); ++r)
                    if (desc_->reps(s)[r] == sim) chain_add(f, static_cast<uint32_t>(iso_.cell_index(s, r, 0)), te::one());
            }
            ++cert.checked;
            if (iso_.augment(q, psi_.value[q][y]) != iso_.augment(q, f)) fail(cert.psi_over_iota, "Psi^Z_T is not over i_T o eps_T");
        }
    }
    for (const TreeMap* m : {&phi_, &psi_, &HT_, &HX_})
        for (const auto& row : m->value)
            for (const auto& ch : row)
                for (const auto& [c, v] : ch) {
                    ++cert.checked;
                    if (!integral(v)) fail(cert.integral, "a coefficient of Phi^Z, Psi^Z, H_T or H_X is not integral");
                }
    const TreeMap PP = ComposeMaps(psi_, phi_), QQ = ComposeMaps(phi_, psi_);
    for (uint32_t q = 0; q < q_top_; ++q) {
        for (uint32_t y = 0; y < tree_.rank(q); ++y) {
            ++cert.checked;
            LChain lhs = tree_.apply_d(q + 1, HT_.value[q][y]);
            if (q >= 1) chain_add(lhs, ApplyMap(HT_, q - 1, tree_.apply_d(q, LChain{{y, te::one()}})));
            chain_add(lhs, PP.value[q][y], Rat::of(-1));
            chain_add(lhs, y, te::one());
            if (!chain_zero(lhs)) fail(cert.H_T, "dH_T + H_T d != Phi Psi - id at q=" + std::to_string(q));
        }
        for (uint32_t y = 0; y < iso_.rank(q); ++y) {
            ++cert.checked;
            LChain lhs = iso_.apply_d(q + 1, HX_.value[q][y]);
            if (q >= 1) chain_add(lhs, ApplyMap(HX_, q - 1, iso_.apply_d(q, LChain{{y, te::one()}})));
            chain_add(lhs, y, te::one(), Rat::of(-1));
            chain_add(lhs, QQ.value[q][y]);
            if (!chain_zero(lhs)) fail(cert.H_X, "dH_X + H_X d != id - Psi Phi at q=" + std::to_string(q));
        }
    }
    return cert;
}

TreeMap IntegralTreeComparison::phi_T(uint32_t ell) const { return ComposeMaps(phi_, PhiTreeEll(ell, q_top_ + 1)); }
TreeMap IntegralTreeComparison::psi_T(uint32_t ell) const { return ComposeMaps(PsiTreeEll(ell, q_top_ + 1), psi_); }

IntegralTreeComparison::EllCheck IntegralTreeComparison::check_ell(uint32_t ell) const {
    EllCheck r;
    r.phi_T_chain = r.psi_T_chain = r.H_ell = r.K = r.K_prime = r.phi_psi_strict = true;
    auto fail = [&](bool& flag, const std::string& w) {
        flag = false;
        if (r.first_failure.empty()) r.first_failure = w;
    };
    const TreeModelChain L(ell, q_top_ + 1);
    const TreeMap Phl = PhiTreeEll(ell, q_top_ + 1), Psl = PsiTreeEll(ell, q_top_ + 1);
    auto lnorm = [&](uint32_t q, const LChain& x) {
        LChain o;
        for (const auto& [c, v] : x) chain_add(o, c, v * L.idem(q, c));
        return o;
    };
    // (3) Phi^T_ell, Psi^T_ell chain maps, Phi Psi = id strictly (on the ell-local model)
    for (uint32_t q = 0; q <= q_top_; ++q)
        for (uint32_t y = 0; y < L.rank(q); ++y) {
            ++r.checked;
            LChain a = lnorm(q, ApplyMap(Phl, q, Psl.value[q][y]));
            chain_add(a, y, L.idem(q, y), Rat::of(-1));
            if (!chain_zero(lnorm(q, a))) fail(r.phi_psi_strict, "Phi^T Psi^T != id strictly at q=" + std::to_string(q));
        }
    const TreeMap PhT = phi_T(ell), PsT = psi_T(ell);
    for (uint32_t q = 1; q <= q_top_; ++q) {
        for (uint32_t y = 0; y < iso_.rank(q); ++y) {
            ++r.checked;
            LChain a = L.apply_d(q, PhT.value[q][y]);
            chain_add(a, ApplyMap(PhT, q - 1, iso_.apply_d(q, LChain{{y, te::one()}})), Rat::of(-1));
            if (!chain_zero(lnorm(q - 1, a))) fail(r.phi_T_chain, "Phi_T is not a chain map at q=" + std::to_string(q));
        }
        for (uint32_t y = 0; y < L.rank(q); ++y) {
            ++r.checked;
            LChain a = iso_.apply_d(q, PsT.value[q][y]);
            chain_add(a, ApplyMap(PsT, q - 1, L.apply_d(q, LChain{{y, L.idem(q, y)}})), Rat::of(-1));
            if (!chain_zero(a)) fail(r.psi_T_chain, "Psi_T is not a chain map at q=" + std::to_string(q));
        }
    }
    // H_ell on the free tree: d H + H d = Psi^T Phi^T - id
    const TreeMap RR = ComposeMaps(Phl, Psl);
    TreeMap Hl;
    Hl.degree_shift = 1;
    Hl.value.assign(q_top_, {});
    for (uint32_t q = 0; q < q_top_; ++q) {
        Hl.value[q].assign(tree_.rank(q), {});
        for (uint32_t y = 0; y < tree_.rank(q); ++y) {
            LChain w = RR.value[q][y];
            chain_add(w, y, te::one(), Rat::of(-1));
            if (q >= 1) chain_add(w, ApplyMap(Hl, q - 1, tree_.apply_d(q, LChain{{y, te::one()}})), Rat::of(-1));
            Hl.value[q][y] = tree_.contract(q, w);
        }
    }
    for (uint32_t q = 0; q < q_top_; ++q)
        for (uint32_t y = 0; y < tree_.rank(q); ++y) {
            ++r.checked;
            LChain lhs = tree_.apply_d(q + 1, Hl.value[q][y]);
            if (q >= 1) chain_add(lhs, ApplyMap(Hl, q - 1, tree_.apply_d(q, LChain{{y, te::one()}})));
            chain_add(lhs, RR.value[q][y], Rat::of(-1));
            chain_add(lhs, y, te::one());
            if (!chain_zero(lhs)) fail(r.H_ell, "dH_ell + H_ell d != Psi^T Phi^T - id at q=" + std::to_string(q));
        }
    // K = Phi^T H_T Psi^T on the ell-local model: d K + K d = Phi_T Psi_T - id
    const TreeMap K = ComposeMaps(ComposeMaps(Psl, HT_), Phl);
    const TreeMap A = ComposeMaps(PsT, PhT);   // Phi_T o Psi_T : local -> local
    for (uint32_t q = 0; q + 1 < q_top_; ++q)
        for (uint32_t y = 0; y < L.rank(q); ++y) {
            ++r.checked;
            LChain lhs = L.apply_d(q + 1, K.value[q][y]);
            if (q >= 1) chain_add(lhs, ApplyMap(K, q - 1, L.apply_d(q, LChain{{y, L.idem(q, y)}})));
            chain_add(lhs, A.value[q][y], Rat::of(-1));
            chain_add(lhs, y, L.idem(q, y));
            if (!chain_zero(lnorm(q, lhs))) fail(r.K, "dK + Kd != Phi_T Psi_T - id at q=" + std::to_string(q));
        }
    // K' = H_X - Psi^Z H_ell Phi^Z on Iso: d K' + K' d = id - Psi_T Phi_T
    const TreeMap X = ComposeMaps(ComposeMaps(phi_, Hl), psi_);
    const TreeMap B = ComposeMaps(PhT, PsT);   // Psi_T o Phi_T : iso -> iso
    for (uint32_t q = 0; q + 1 < q_top_; ++q)
        for (uint32_t y = 0; y < iso_.rank(q); ++y) {
            ++r.checked;
            auto Kp = [&](uint32_t qq, const LChain& x) {
                LChain o = ApplyMap(HX_, qq, x);
                chain_add(o, ApplyMap(X, qq, x), Rat::of(-1));
                return o;
            };
            LChain lhs = iso_.apply_d(q + 1, Kp(q, LChain{{y, te::one()}}));
            if (q >= 1) chain_add(lhs, Kp(q - 1, iso_.apply_d(q, LChain{{y, te::one()}})));
            chain_add(lhs, y, te::one(), Rat::of(-1));
            chain_add(lhs, B.value[q][y]);
            if (!chain_zero(lhs)) fail(r.K_prime, "dK' + K'd != id - Psi_T Phi_T at q=" + std::to_string(q));
        }
    return r;
}

// ---------------------------------------------------------------------------
// cochains
// ---------------------------------------------------------------------------
template<class Elt>
engine::DenseBuf<Elt> PullbackMap(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, const TreeMap& f, uint32_t q,
                                  uint32_t src_cells, uint32_t dst_cells) {
    const uint32_t D = M.dim;
    engine::DenseBuf<Elt> out(src_cells * D, dst_cells * D);
    std::map<GKey, engine::DenseBuf<Elt>> memo;
    ENGINE_CHECK_MSG(q < f.value.size() && f.value[q].size() == src_cells, "PullbackMap: the map's source cells");
    for (uint32_t y = 0; y < src_cells; ++y)
        for (const auto& [x, b] : f.value[q][y]) {
            ENGINE_CHECK_MSG(x < dst_cells, "PullbackMap: a target cell beyond dst_cells");
            const engine::DenseBuf<Elt> R = RhoQ(ctx, b, M, memo);
            for (uint32_t i = 0; i < D; ++i)
                for (uint32_t j = 0; j < D; ++j) {
                    Elt& d = out.at(y * D + i, x * D + j);
                    d = ctx.add(d, R.at(i, j));
                }
        }
    return out;
}

template<class Elt>
engine::DenseBuf<Elt> TransportToTree(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, const IntegralTreeComparison& cmp,
                                      uint32_t ell, uint32_t q, const engine::DenseBuf<Elt>& T_on_Iso) {
    const TreeModelChain L(ell, cmp.q_top() + 1);
    const TreeMap PhT = cmp.phi_T(ell), PsT = cmp.psi_T(ell);
    const engine::DenseBuf<Elt> Ph = PullbackMap(ctx, M, PhT, q, cmp.iso().rank(q), L.rank(q));   // C_T -> C_I
    const engine::DenseBuf<Elt> Ps = PullbackMap(ctx, M, PsT, q, L.rank(q), cmp.iso().rank(q));   // C_I -> C_T
    ENGINE_CHECK_MSG(T_on_Iso.rows == Ph.rows && T_on_Iso.cols == Ph.rows, "TransportToTree: T must be an endomorphism of C^q_I");
    engine::DenseBuf<Elt> TP(T_on_Iso.rows, Ph.cols), out(Ps.rows, Ph.cols);
    engine::kernels::gemm<Elt>(ctx, engine::cview(T_on_Iso), engine::cview(Ph), engine::view(TP));
    engine::kernels::gemm<Elt>(ctx, engine::cview(Ps), engine::cview(TP), engine::view(out));
    return out;
}

#define ISOMODELTREE_INSTANTIATE(Elt)                                                                                        \
    template engine::DenseBuf<Elt> TreeModelChain::coboundary<Elt>(const engine::ModCtx<Elt>&, const CoefficientModule<Elt>&, \
                                                                   uint32_t) const;                                         \
    template engine::DenseBuf<Elt> TreeModelChain::projector<Elt>(const engine::ModCtx<Elt>&, const CoefficientModule<Elt>&,  \
                                                                  uint32_t) const;                                          \
    template engine::DenseBuf<Elt> PullbackMap<Elt>(const engine::ModCtx<Elt>&, const CoefficientModule<Elt>&, const TreeMap&, \
                                                    uint32_t, uint32_t, uint32_t);                                          \
    template engine::DenseBuf<Elt> TransportToTree<Elt>(const engine::ModCtx<Elt>&, const CoefficientModule<Elt>&,             \
                                                        const IntegralTreeComparison&, uint32_t, uint32_t, const engine::DenseBuf<Elt>&);

ISOMODELTREE_INSTANTIATE(uint32_t)

}  // namespace integral_model
