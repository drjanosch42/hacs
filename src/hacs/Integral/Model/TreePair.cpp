// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT D5c: the chain-level tree cusp pair of [C, sec:treepair] (def:treepair, prop:treepair(1)(b), (3)).  See TreePair.h.
#include "TreePair.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>

#include "PeriodicColumn.h"
#include "TreeCochains.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace integral_model {

using hecke::Mat22;
namespace te = tree_elt;

namespace {
std::atomic<uint32_t> g_pair_fault{0};
}  // namespace
TreePairFault ActiveTreePairFault() { return static_cast<TreePairFault>(g_pair_fault.load(std::memory_order_relaxed)); }
ScopedTreePairFault::ScopedTreePairFault(TreePairFault f) : saved_(g_pair_fault.exchange(static_cast<uint32_t>(f))) {}
ScopedTreePairFault::~ScopedTreePairFault() { g_pair_fault.store(saved_); }

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
QGroupRingElt scaled(const QGroupRingElt& a, const Rat& r) {
    QGroupRingElt x;
    x.add(a, r);
    return x;
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
// The column contraction of a periodic column with differentials r_{1-g} (odd t) and r_N (even t): k with k d_{t+1} = a for a column
// cycle a of degree t (t = 0: of augmentation zero per coset); the convention of IsotropyModelTree's tree_contr.
QGroupRingElt column_contr(const QGroupRingElt& a, const std::vector<Mat22>& pw, uint32_t t) {
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
// right multiplication of every coefficient by e (the ell-local normal form)
LChain norm_by(const LChain& x, const std::function<const QGroupRingElt&(uint32_t)>& idem) {
    LChain o;
    for (const auto& [c, v] : x) chain_add(o, c, v * idem(c));
    return o;
}
bool integral(const QGroupRingElt& a) {
    for (const auto& kv : a.terms)
        if (kv.second.den != 1) return false;
    return true;
}
Mat22 Sinv() { return Mat22::S().inv_unimodular(); }

}  // namespace

// ---------------------------------------------------------------------------
// CuspModelChain
// ---------------------------------------------------------------------------
CuspModelChain::CuspModelChain(uint32_t ell, uint32_t q_top) : ell_(ell), q_top_(q_top) {
    ENGINE_CHECK_MSG(ell == 0 || ell == 2 || ell == 3 || ell >= 5, "CuspModelChain: ell = 0 (free), 2, 3 or >= 5");
    per_ = ell == 0 || ell == 2;
    e_ = per_ ? te::one() : te::ec();
    pw_ = powers(Mat22::I().neg(), 2);
    cells_.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        if (q == 0 || per_) cells_[q].push_back({'A', q});
        if (q >= 1 && (q - 1 == 0 || per_)) cells_[q].push_back({'Z', q - 1});
    }
    const bool t2 = ActiveTreePairFault() == TreePairFault::CuspOneMinusT2;   // VALIDATION ONLY
    QGroupRingElt one_minus_T = te::one();
    one_minus_T.add(te::g(t2 ? Mat22::T() * Mat22::T() : Mat22::T()), Rat::of(-1));
    one_minus_T = one_minus_T * e_;
    diff_.assign(q_top + 1, {});
    for (uint32_t q = 1; q <= q_top; ++q) {
        for (uint32_t ci = 0; ci < cells_[q].size(); ++ci) {
            const CuspCell& y = cells_[q][ci];
            if (y.k == 'A') {
                const int64_t r = index(q - 1, 'A', y.t - 1);
                if (r >= 0) diff_[q].push_back({static_cast<uint32_t>(r), ci, eps(y.t)});
            } else {
                const int64_t ra = index(q - 1, 'A', y.t);
                if (ra >= 0) diff_[q].push_back({static_cast<uint32_t>(ra), ci, one_minus_T});
                if (y.t >= 1) {
                    const int64_t rz = index(q - 1, 'Z', y.t - 1);
                    if (rz >= 0) diff_[q].push_back({static_cast<uint32_t>(rz), ci, scaled(eps(y.t), Rat::of(-1))});
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

QGroupRingElt CuspModelChain::eps(uint32_t t) const {
    const QGroupRingElt c = te::c();
    if (t % 2 == 1) return te::sum(te::one(), c, -1) * e_;   // (1 - c) e
    return te::sum(te::one(), c, 1) * e_;                    // (1 + c) e
}

int64_t CuspModelChain::index(uint32_t q, char k, uint32_t t) const {
    if (q > q_top_) return -1;
    for (uint32_t i = 0; i < cells_[q].size(); ++i)
        if (cells_[q][i].k == k && cells_[q][i].t == t) return i;
    return -1;
}

LChain CuspModelChain::apply_d(uint32_t q, const LChain& x) const {
    ENGINE_CHECK_MSG(q >= 1 && q <= q_top_, "CuspModelChain::apply_d: degree out of range");
    LChain out;
    for (const auto& [col, v] : x) {
        if (v.is_zero()) continue;
        for (const LEntry* e : by_col_[q].at(col)) chain_add(out, e->row, v * e->coeff);
    }
    return out;
}

LChain CuspModelChain::contract(uint32_t d, const LChain& z0) const {
    ENGINE_CHECK_MSG(ell_ == 0, "CuspModelChain::contract: the solver is on the free cusp model");
    ENGINE_CHECK_MSG(d + 1 <= q_top_, "CuspModelChain::contract: degree beyond the built range");
    LChain z = z0, k;
    // the z column (on top, sign -1) first
    if (d >= 1) {
        const int64_t iz = index(d, 'Z', d - 1);
        auto it = iz >= 0 ? z.find(static_cast<uint32_t>(iz)) : z.end();
        if (it != z.end() && !it->second.is_zero()) {
            const QGroupRingElt h = scaled(column_contr(it->second, pw_, d - 1), Rat::of(-1));
            LChain kk;
            chain_add(kk, static_cast<uint32_t>(index(d + 1, 'Z', d)), h);
            chain_add(k, kk);
            chain_add(z, apply_d(d + 1, kk), Rat::of(-1));
        }
    }
    {
        const int64_t ia = index(d, 'A', d);
        auto it = ia >= 0 ? z.find(static_cast<uint32_t>(ia)) : z.end();
        if (it != z.end() && !it->second.is_zero()) {
            const QGroupRingElt h = column_contr(it->second, pw_, d);
            LChain kk;
            chain_add(kk, static_cast<uint32_t>(index(d + 1, 'A', d + 1)), h);
            chain_add(k, kk);
            chain_add(z, apply_d(d + 1, kk), Rat::of(-1));
        }
    }
    ENGINE_CHECK_MSG(chain_zero(z), "CuspModelChain::contract: the descent left a remainder (z not a cycle of ker eps_B)");
    return k;
}

bool CuspModelChain::d_squared_zero(uint64_t& checked, std::string& why) const {
    for (uint32_t q = 2; q <= q_top_; ++q)
        for (uint32_t c = 0; c < cells_[q].size(); ++c) {
            ++checked;
            LChain dd = apply_d(q - 1, apply_d(q, LChain{{c, e_}}));
            LChain o;
            for (const auto& [x, v] : dd) chain_add(o, x, v * e_);
            if (!chain_zero(o)) {
                why = "d o d != 0 on the cusp model (ell = " + std::to_string(ell_) + ") at q = " + std::to_string(q);
                return false;
            }
        }
    return true;
}

// ---------------------------------------------------------------------------
// the axis map and the cusp comparisons
// ---------------------------------------------------------------------------
TreeMap AxisMapChain(uint32_t ell, const CuspModelChain& B, const TreeModelChain& L) {
    const TreePairFault f = ActiveTreePairFault();
    const Mat22 si = f == TreePairFault::AxisSForSinv ? Mat22::S() : Sinv();   // VALIDATION ONLY
    const QGroupRingElt eS = L.column_e('S'), eU = L.column_e('U'), eE = L.column_e('E');
    const QGroupRingElt SI = te::g(si);
    QGroupRingElt gam = SI;
    gam.add(te::g(Mat22::T()), Rat::of(-1));   // S^{-1} - T
    auto a_t = [&](uint32_t t) -> QGroupRingElt {
        // the S-component of the canonical lift (def:isomodeltree(3), (4))
        if (ell == 0 || ell == 2) return t % 2 == 0 ? te::one() : te::sum(te::one(), te::S());
        return te::eS4();
    };
    TreeMap out;
    const uint32_t Q = std::min(B.q_top(), L.q_top());
    out.value.assign(Q + 1, {});
    for (uint32_t q = 0; q <= Q; ++q) {
        out.value[q].assign(B.rank(q), {});
        for (uint32_t u = 0; u < B.rank(q); ++u) {
            const CuspCell c = B.cells(q)[u];
            LChain img;
            if (c.k == 'A') {
                const int64_t s = L.index(q, 'S', c.t);
                if (s >= 0) chain_add(img, static_cast<uint32_t>(s), a_t(c.t) * eS);
            } else {
                const uint32_t t = c.t;
                if (t % 2 == 0) {
                    const int64_t s = L.index(q, 'S', t + 1);
                    if (s >= 0) chain_add(img, static_cast<uint32_t>(s), SI * eS, Rat::of(-1));
                    const int64_t uu = L.index(q, 'U', t + 1);
                    if (uu >= 0 && f != TreePairFault::AxisDropBeta) chain_add(img, static_cast<uint32_t>(uu), SI * eU);
                }
                const int64_t e = L.index(q, 'E', t);
                if (e >= 0) chain_add(img, static_cast<uint32_t>(e), gam * eE);
            }
            out.value[q][u] = std::move(img);
        }
    }
    return out;
}

namespace {
uint32_t q_order_e(uint32_t ell) { return (ell == 0 || ell == 2) ? 1u : 2u; }   // |Q_e|
}  // namespace

TreeMap PhiCuspEll(const CuspModelChain& F, const CuspModelChain& Lc) {
    ENGINE_CHECK(F.ell() == 0);
    TreeMap out;
    const uint32_t Q = std::min(F.q_top(), Lc.q_top());
    out.value.assign(Q + 1, {});
    const uint32_t qe = q_order_e(Lc.ell());
    for (uint32_t q = 0; q <= Q; ++q) {
        out.value[q].assign(F.rank(q), {});
        for (uint32_t y = 0; y < F.rank(q); ++y) {
            const CuspCell c = F.cells(q)[y];
            const int64_t t = Lc.index(q, c.k, c.t);
            if (t < 0) continue;
            Rat s = Rat::of(1);
            for (uint32_t i = 0; i < c.t / 2; ++i) s = s * Rat::of(qe);
            chain_add(out.value[q][y], static_cast<uint32_t>(t), scaled(Lc.idem(), s));
        }
    }
    return out;
}

TreeMap PsiCuspEll(const CuspModelChain& Lc, const CuspModelChain& F) {
    ENGINE_CHECK(F.ell() == 0);
    TreeMap out;
    const uint32_t Q = std::min(F.q_top(), Lc.q_top());
    out.value.assign(Q + 1, {});
    const uint32_t qe = q_order_e(Lc.ell());
    for (uint32_t q = 0; q <= Q; ++q) {
        out.value[q].assign(Lc.rank(q), {});
        for (uint32_t y = 0; y < Lc.rank(q); ++y) {
            const CuspCell c = Lc.cells(q)[y];
            const int64_t t = F.index(q, c.k, c.t);
            ENGINE_CHECK(t >= 0);
            Rat s = Rat::of(1);
            for (uint32_t i = 0; i < c.t / 2; ++i) s = s * Rat::of(1, qe);
            chain_add(out.value[q][y], static_cast<uint32_t>(t), scaled(Lc.idem(), s));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// TreePairComparison
// ---------------------------------------------------------------------------
namespace {
const Simplex kV0{0}, kV1{1}, kV2{2}, kE01{0, 1}, kE12{1, 2}, kE23{2, 3};
}  // namespace

uint32_t TreePairComparison::cell_of(const Simplex& sim) const {
    const IsotropyModel& desc = iso().descriptor();
    const uint32_t s = static_cast<uint32_t>(sim.size()) - 1;
    for (uint32_t r = 0; r < desc.reps(s).size(); ++r)
        if (desc.reps(s)[r] == sim) {
            const int64_t c = iso().cell_index(s, r, 0);
            ENGINE_CHECK_MSG(c >= 0, "TreePairComparison: a table simplex has no t = 0 cell");
            return static_cast<uint32_t>(c);
        }
    ENGINE_CHECK_MSG(false, "TreePairComparison: a table simplex is not a representative of the descriptor");
    return 0;
}

bool TreePairComparison::is_boundary(uint32_t q, uint32_t c) const { return q < isb_.size() && c < isb_[q].size() && isb_[q][c]; }

TreePairComparison::TreePairComparison(const IntegralTreeComparison& cmp, uint32_t ell, uint32_t q_top)
    : cmp_(&cmp), ell_(ell), q_top_(q_top), L_(ell, q_top + 1), B0_(0, q_top + 1), Bl_(ell, q_top + 1) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(ell == 2 || ell == 3 || ell >= 5, "TreePairComparison: ell in {2, 3} (or >= 5)");
    ENGINE_CHECK_MSG(cmp.q_top() >= q_top && cmp.tree().q_top() >= q_top + 1 && iso().q_top() >= q_top + 1,
                     "TreePairComparison: the comparison of prop:isomodeltree(4) is not built to degree q_top");
    const LocalModel& I = iso();
    const TreeModelChain& T0 = cmp.tree();
    isob_.assign(q_top + 2, {});
    isb_.assign(q_top + 2, {});
    for (uint32_t q = 0; q <= q_top + 1; ++q) {
        isb_[q].assign(I.rank(q), 0);
        for (uint32_t c = 0; c < I.rank(q); ++c)
            if (I.cells(q)[c].boundary) {
                isob_[q].push_back(c);
                isb_[q][c] = 1;
            }
    }
    auto unit = [](uint32_t c) { return LChain{{c, te::one()}}; };
    auto in_isob = [&](uint32_t q, const LChain& x) {
        for (const auto& [c, v] : x)
            if (!v.is_zero() && !is_boundary(q, c)) return false;
        return true;
    };
    axis_ = AxisMapChain(ell, Bl_, L_);
    axis0_ = AxisMapChain(0, B0_, T0);
    phiDl_ = PhiCuspEll(B0_, Bl_);
    psiDl_ = PsiCuspEll(Bl_, B0_);
    phiT_ = cmp.phi_T(ell);
    psiT_ = cmp.psi_T(ell);
    const TreeMap phiTl = PhiTreeEll(ell, q_top + 1), psiTl = PsiTreeEll(ell, q_top + 1);
    const IsotropyModel& desc = I.descriptor();
    // Phi^Z_d := (L) for f = r_d o cmpI along eps_B (Isob -> B_T)
    phiZd_.value.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        phiZd_.value[q].assign(I.rank(q), {});
        for (uint32_t y : isob_[q]) {
            const LCell& c = I.cells(q)[y];
            LChain ys;
            if (c.t == 0) {
                const Simplex& sim = desc.reps(c.s)[c.rep];
                if (c.s == 0) {
                    ENGINE_CHECK_MSG(sim == kV0 || sim == kV1 || sim == kV2, "TreePairComparison: an unexpected boundary vertex");
                    chain_add(ys, static_cast<uint32_t>(B0_.index(0, 'A', 0)), te::one());
                } else {
                    ENGINE_CHECK_MSG(sim == kE01 || sim == kE12 || sim == kE23, "TreePairComparison: an unexpected boundary edge");
                    if (sim == kE23) chain_add(ys, static_cast<uint32_t>(B0_.index(1, 'Z', 0)), te::one(), Rat::of(-1));
                }
            }
            LChain val = ys;
            if (q >= 1) {
                const LChain dy = I.apply_d(q, unit(y));
                ENGINE_CHECK_MSG(in_isob(q - 1, dy), "TreePairComparison: Isob is not a subcomplex of Iso (IM2)");
                LChain z = ApplyMap(phiZd_, q - 1, dy);
                if (!ys.empty()) chain_add(z, B0_.apply_d(q, ys), Rat::of(-1));
                chain_add(val, B0_.contract(q - 1, z));
            }
            phiZd_.value[q][y] = std::move(val);
        }
    }
    // Psi^Z_d := (L) for f = i_d o eps_B along cmpI|Isob (B_T -> Isob)
    psiZd_.value.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        psiZd_.value[q].assign(B0_.rank(q), {});
        for (uint32_t u = 0; u < B0_.rank(q); ++u) {
            const CuspCell c = B0_.cells(q)[u];
            LChain ys;
            if (c.k == 'A' && c.t == 0) chain_add(ys, cell_of(kV0), te::one());
            if (c.k == 'Z' && c.t == 0)
                for (const Simplex* s : {&kE01, &kE12, &kE23}) chain_add(ys, cell_of(*s), te::one(), Rat::of(-1));
            LChain val = ys;
            if (q >= 1) {
                LChain z = ApplyMap(psiZd_, q - 1, B0_.apply_d(q, unit(u)));
                if (!ys.empty()) chain_add(z, I.apply_d(q, ys), Rat::of(-1));
                chain_add(val, I.contract(q - 1, z));
            }
            psiZd_.value[q][u] = std::move(val);
        }
    }
    // Phi_d = Phi^d_ell Phi^Z_d, Psi_d = Psi^Z_d Psi^d_ell
    phid_.value.assign(q_top + 1, {});
    psid_.value.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        phid_.value[q].assign(I.rank(q), {});
        for (uint32_t y : isob_[q]) phid_.value[q][y] = ApplyMap(phiDl_, q, phiZd_.value[q][y]);
        psid_.value[q].assign(Bl_.rank(q), {});
        for (uint32_t u = 0; u < Bl_.rank(q); ++u) psid_.value[q][u] = ApplyMap(psiZd_, q, psiDl_.value[q][u]);
    }
    // y_ell := (L') along eps_T on Isob: A = Phi^Z_T j, B = Psi^T_ell a Phi_d, g = g_sq (v_0 -> -e)
    const uint32_t tE0 = static_cast<uint32_t>(T0.index(1, 'E', 0));
    yl_.degree_shift = 1;
    yl_.value.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        yl_.value[q].assign(I.rank(q), {});
        for (uint32_t y : isob_[q]) {
            const LCell& c = I.cells(q)[y];
            LChain ys;
            if (q == 0 && c.t == 0 && desc.reps(0)[c.rep] == kV0) chain_add(ys, tE0, te::one(), Rat::of(-1));
            LChain w = cmp.phi().value[q][y];
            chain_add(w, ApplyMap(psiTl, q, ApplyMap(axis_, q, phid_.value[q][y])), Rat::of(-1));
            if (q >= 1) chain_add(w, ApplyMap(yl_, q - 1, I.apply_d(q, unit(y))), Rat::of(-1));
            LChain z = w;
            if (!ys.empty()) chain_add(z, T0.apply_d(q + 1, ys), Rat::of(-1));
            LChain val = ys;
            chain_add(val, T0.contract(q, z));
            yl_.value[q][y] = std::move(val);
        }
    }
    Yphi_.degree_shift = 1;
    Yphi_.value.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        Yphi_.value[q].assign(I.rank(q), {});
        for (uint32_t y : isob_[q]) Yphi_.value[q][y] = ApplyMap(phiTl, q + 1, yl_.value[q][y]);
    }
    // y'_ell := (L') along cmpI on B_T: A = Psi_T a Phi^d_ell, B = j Psi^Z_d, g = g' (v -> v0v4 + v4v6, e -> ...)
    ypl_.degree_shift = 1;
    ypl_.value.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        ypl_.value[q].assign(B0_.rank(q), {});
        for (uint32_t u = 0; u < B0_.rank(q); ++u) {
            const CuspCell c = B0_.cells(q)[u];
            LChain ys;
            if (c.k == 'A' && c.t == 0) {
                chain_add(ys, cell_of({0, 4}), te::one());
                chain_add(ys, cell_of({4, 6}), te::one());
            }
            if (c.k == 'Z' && c.t == 0) {
                chain_add(ys, cell_of({0, 1, 4}), te::one());
                chain_add(ys, cell_of({1, 4, 6}), te::one(), Rat::of(-1));
                chain_add(ys, cell_of({1, 2, 6}), te::one());
                chain_add(ys, cell_of({2, 3, 5}), te::one());
                chain_add(ys, cell_of({2, 5, 6}), te::one());
            }
            LChain w = ApplyMap(psiT_, q, ApplyMap(axis_, q, phiDl_.value[q][u]));
            chain_add(w, psiZd_.value[q][u], Rat::of(-1));
            if (q >= 1) chain_add(w, ApplyMap(ypl_, q - 1, B0_.apply_d(q, unit(u))), Rat::of(-1));
            LChain z = w;
            if (!ys.empty()) chain_add(z, I.apply_d(q + 1, ys), Rat::of(-1));
            LChain val = ys;
            chain_add(val, I.contract(q, z));
            ypl_.value[q][u] = std::move(val);
        }
    }
    Ypsi_.degree_shift = 1;
    Ypsi_.value.assign(q_top + 1, {});
    for (uint32_t q = 0; q <= q_top; ++q) {
        Ypsi_.value[q].assign(Bl_.rank(q), {});
        for (uint32_t u = 0; u < Bl_.rank(q); ++u) Ypsi_.value[q][u] = ApplyMap(ypl_, q, psiDl_.value[q][u]);
    }
    build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

TreePairCertificates TreePairComparison::certify() const {
    TreePairCertificates cert;
    cert.tables = cert.cusp_complex = cert.axis_chain = cert.axis_over_fa = cert.phiZd_chain = cert.psiZd_chain = cert.phiZd_over_r =
        cert.psiZd_over_i = cert.psiZd_in_isob = cert.phid_psid_strict = cert.cusp_maps_chain = cert.square_phi = cert.square_psi =
            cert.integral = true;
    auto fail = [&](bool& flag, const std::string& w) {
        flag = false;
        if (cert.first_failure.empty()) cert.first_failure = w;
    };
    const LocalModel& I = iso();
    const IsotropyModel& desc = I.descriptor();
    const TreeModelChain& T0 = cmp_->tree();
    auto unit = [](uint32_t c) { return LChain{{c, te::one()}}; };
    const std::function<const QGroupRingElt&(uint32_t)> eB = [&](uint32_t) -> const QGroupRingElt& { return Bl_.idem(); };
    auto lnorm = [&](uint32_t q, const LChain& x) {
        return norm_by(x, [&](uint32_t c) -> const QGroupRingElt& { return L_.idem(q, c); });
    };
    // ---- (3)(a) the tables, on representatives -----------------------------------------------------------------------------
    {
        using XChain = std::map<Simplex, QGroupRingElt>;
        using CChain = std::map<char, QGroupRingElt>;   // C(T) ('S', 'U', 'E') or C(L) ('v', 'e')
        auto xadd = [](XChain& x, const Simplex& s, const QGroupRingElt& v, const Rat& sc = Rat::of(1)) {
            if (v.is_zero()) return;
            x[s].add(v, sc);
            if (x[s].is_zero()) x.erase(s);
        };
        auto cadd = [](CChain& x, char c, const QGroupRingElt& v, const Rat& sc = Rat::of(1)) {
            if (v.is_zero()) return;
            x[c].add(v, sc);
            if (x[c].is_zero()) x.erase(c);
        };
        auto dim_of = [](const Simplex& s) { return static_cast<uint32_t>(s.size()) - 1; };
        auto rep_index = [&](uint32_t s, const Simplex& sim) -> int64_t {
            for (uint32_t r = 0; r < desc.reps(s).size(); ++r)
                if (desc.reps(s)[r] == sim) return r;
            return -1;
        };
        auto xnorm = [&](const XChain& x) {
            std::map<std::pair<Simplex, GKey>, Rat> out;
            for (const auto& [sim, a] : x) {
                const uint32_t s = dim_of(sim);
                const int64_t r = rep_index(s, sim);
                ENGINE_CHECK(r >= 0);
                const auto& pw = I.column(s, static_cast<uint32_t>(r)).pow;
                for (const auto& [g, cf] : a.terms) out[{sim, decompose(g, pw).first}] = out[{sim, decompose(g, pw).first}] + cf;
            }
            for (auto it = out.begin(); it != out.end();) it = it->second.is_zero() ? out.erase(it) : std::next(it);
            return out;
        };
        const std::vector<Mat22> pwc = powers(Mat22::I().neg(), 2);
        auto cnorm = [&](const CChain& x, bool tree) {
            std::map<std::pair<char, GKey>, Rat> out;
            for (const auto& [c, a] : x)
                for (const auto& [g, r] : a.terms) {
                    const GKey k = decompose(g, tree ? T0.column_pow(c) : pwc).first;
                    out[{c, k}] = out[{c, k}] + r;
                }
            for (auto it = out.begin(); it != out.end();) it = it->second.is_zero() ? out.erase(it) : std::next(it);
            return out;
        };
        auto bdry = [&](const Simplex& sim) {
            XChain out;
            const uint32_t s = dim_of(sim);
            const int64_t r = rep_index(s, sim);
            ENGINE_CHECK(r >= 0);
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
        const RetractionTables tab = DefaultRetractionTables();
        const QGroupRingElt one = te::one(), mone = scaled(te::one(), Rat::of(-1));
        // i_d, r_d, G_d, f_a, g_sq, g' (def:treepair(4))
        auto i_d = [&](const CChain& x) {
            XChain o;
            for (const auto& [c, a] : x) {
                if (c == 'v') xadd(o, kV0, a);
                else
                    for (const Simplex* s : {&kE01, &kE12, &kE23}) xadd(o, *s, a, Rat::of(-1));
            }
            return o;
        };
        auto r_d = [&](const XChain& x) {
            CChain o;
            for (const auto& [sim, a] : x) {
                if (sim == kV0 || sim == kV1 || sim == kV2) cadd(o, 'v', a);
                else if (sim == kE23) cadd(o, 'e', a, Rat::of(-1));
                else ENGINE_CHECK(sim == kE01 || sim == kE12);
            }
            return o;
        };
        auto G_d = [&](const XChain& x) {
            XChain o;
            for (const auto& [sim, a] : x) {
                if (sim == kV1) xadd(o, kE01, a);
                if (sim == kV2) {
                    xadd(o, kE01, a);
                    xadd(o, kE12, a);
                }
            }
            return o;
        };
        QGroupRingElt gam = te::g(Sinv());
        gam.add(te::g(Mat22::T()), Rat::of(-1));
        auto f_a = [&](const CChain& x) {
            CChain o;
            for (const auto& [c, a] : x) {
                if (c == 'v') cadd(o, 'S', a);
                else cadd(o, 'E', a * gam);
            }
            return o;
        };
        auto g_sq = [&](const XChain& x) {
            CChain o;
            for (const auto& [sim, a] : x)
                if (sim == kV0) cadd(o, 'E', a, Rat::of(-1));
            return o;
        };
        auto gp = [&](char c) {
            XChain o;
            if (c == 'v') {
                xadd(o, {0, 4}, one);
                xadd(o, {4, 6}, one);
            } else {
                xadd(o, {0, 1, 4}, one);
                xadd(o, {1, 4, 6}, mone);
                xadd(o, {1, 2, 6}, one);
                xadd(o, {2, 3, 5}, one);
                xadd(o, {2, 5, 6}, one);
            }
            return o;
        };
        auto dL = [&](const CChain& x) {   // d e = (1 - T) v
            CChain o;
            auto it = x.find('e');
            if (it != x.end()) cadd(o, 'v', it->second * te::sum(one, te::g(Mat22::T()), -1));
            return o;
        };
        auto dT = [&](const CChain& x) {   // d E = v_S - v_U
            CChain o;
            auto it = x.find('E');
            if (it != x.end()) {
                cadd(o, 'S', it->second);
                cadd(o, 'U', it->second, Rat::of(-1));
            }
            return o;
        };
        auto r_T = [&](const XChain& x) {
            CChain o;
            for (const auto& [sim, a] : x) {
                if (dim_of(sim) == 0) {
                    const auto& v = tab.r0.at(sim);
                    cadd(o, v.first, a * v.second);
                } else if (dim_of(sim) == 1) {
                    cadd(o, 'E', a * tab.r1.at(sim));
                }
            }
            return o;
        };
        auto G_T = [&](const XChain& x) {
            XChain o;
            for (const auto& [sim, a] : x)
                for (const auto& [t, b] : tab.G.at(sim)) xadd(o, t, a * b);
            return o;
        };
        auto i_T = [&](const CChain& x) {
            XChain o;
            for (const auto& [c, a] : x) xadd(o, tab.iota.at(c), a);
            return o;
        };
        auto lmulC = [&](const QGroupRingElt& a, const XChain& x) {
            XChain o;
            for (const auto& [s, v] : x) xadd(o, s, a * v);
            return o;
        };
        auto xsub = [&](XChain a, const XChain& b) {
            for (const auto& [s, v] : b) xadd(a, s, v, Rat::of(-1));
            return a;
        };
        auto csub = [&](CChain a, const CChain& b) {
            for (const auto& [s, v] : b) cadd(a, s, v, Rat::of(-1));
            return a;
        };
        auto tick = [&](bool ok, const std::string& w) {
            ++cert.checked;
            if (!ok) fail(cert.tables, "(3)(a) " + w);
        };
        // i_d a chain map
        tick(xnorm(bdry_chain(i_d({{'e', one}}))) == xnorm(i_d(dL({{'e', one}}))), "i_d is not a chain map");
        for (const Simplex* e : {&kE01, &kE12, &kE23}) {
            tick(cnorm(dL(r_d({{*e, one}})), false) == cnorm(r_d(bdry(*e)), false), "r_d is not a chain map");
            // id - i_d r_d = G_d d on edges (G_d = 0 on edges)
            tick(xnorm(xsub({{*e, one}}, i_d(r_d({{*e, one}})))) == xnorm(G_d(bdry(*e))), "id - i_d r_d != G_d d on an edge");
            // r_T j_X - f_a r_d = g_sq d on edges
            tick(cnorm(csub(r_T({{*e, one}}), f_a(r_d({{*e, one}}))), true) == cnorm(g_sq(bdry(*e)), true), "r_T - f_a r_d != g_sq d on an edge");
        }
        for (char x : {'v', 'e'}) tick(cnorm(r_d(i_d({{x, one}})), false) == cnorm({{x, one}}, false), "r_d i_d != id");
        for (const Simplex* v : {&kV0, &kV1, &kV2}) {
            tick(xnorm(xsub({{*v, one}}, i_d(r_d({{*v, one}})))) == xnorm(bdry_chain(G_d({{*v, one}}))), "id - i_d r_d != d G_d on a vertex");
            tick(cnorm(csub(r_T({{*v, one}}), f_a(r_d({{*v, one}}))), true) == cnorm(dT(g_sq({{*v, one}})), true), "r_T - f_a r_d != d g_sq on a vertex");
        }
        // i_T f_a - j_X i_d = d g' + g' d on C(L); and g' = -(G_T j_X i_d + i_T g_sq i_d) as displayed
        tick(xnorm(xsub(i_T(f_a({{'v', one}})), i_d({{'v', one}}))) == xnorm(bdry_chain(gp('v'))), "i_T f_a - i_d != d g' on v");
        {
            XChain rhs = bdry_chain(gp('e'));
            for (const auto& [s, v] : lmulC(te::sum(one, te::g(Mat22::T()), -1), gp('v'))) xadd(rhs, s, v);
            tick(xnorm(xsub(i_T(f_a({{'e', one}})), i_d({{'e', one}}))) == xnorm(rhs), "i_T f_a - i_d != d g' + g' d on e");
        }
        for (char x : {'v', 'e'}) {
            XChain g = G_T(i_d({{x, one}}));
            for (const auto& [s, v] : i_T(g_sq(i_d({{x, one}})))) xadd(g, s, v);
            XChain neg;
            for (const auto& [s, v] : g) xadd(neg, s, v, Rat::of(-1));
            tick(xnorm(neg) == xnorm(gp(x)), "g' differs from -(G_T j_X i_d + i_T g_sq i_d)");
        }
        // f_a a chain map
        tick(cnorm(dT(f_a({{'e', one}})), true) == cnorm(f_a(dL({{'e', one}})), true), "f_a is not a chain map");
        // eps_B a chain map onto C(L) (free model, degrees 1, 2)
        auto epsB = [&](uint32_t q, const LChain& x) {
            CChain o;
            for (const auto& [c, a] : x) {
                const CuspCell cc = B0_.cells(q)[c];
                if (cc.t != 0) continue;
                if (cc.k == 'A' && q == 0) cadd(o, 'v', a);
                if (cc.k == 'Z' && q == 1) cadd(o, 'e', a);
            }
            return o;
        };
        for (uint32_t q = 1; q <= 2 && q <= B0_.q_top(); ++q)
            for (uint32_t u = 0; u < B0_.rank(q); ++u) {
                CChain lhs = q == 1 ? dL(epsB(1, unit(u))) : CChain{};
                tick(cnorm(lhs, false) == cnorm(epsB(q - 1, B0_.apply_d(q, unit(u))), false), "eps_B is not a chain map");
            }
    }
    // ---- the cusp models ----------------------------------------------------------------------------------------------------
    {
        std::string w;
        if (!B0_.d_squared_zero(cert.checked, w)) fail(cert.cusp_complex, w);
        if (!Bl_.d_squared_zero(cert.checked, w)) fail(cert.cusp_complex, w);
    }
    // ---- (1)(b) the axis map: a chain map, well defined, over f_a ----------------------------------------------------------
    auto axis_checks = [&](const TreeMap& A, const CuspModelChain& B, const TreeModelChain& L, const std::string& tag) {
        auto ln = [&](uint32_t q, const LChain& x) { return norm_by(x, [&](uint32_t c) -> const QGroupRingElt& { return L.idem(q, c); }); };
        for (uint32_t q = 0; q < A.value.size(); ++q)
            for (uint32_t u = 0; u < B.rank(q); ++u) {
                for (const auto& [x, a] : A.value[q][u]) {
                    ++cert.checked;
                    if (B.idem() * a * L.idem(q, x) != a) fail(cert.axis_chain, "a^(" + tag + ") is not well defined (e_e a e_x != a) at q=" + std::to_string(q));
                }
                if (q == 0) continue;
                ++cert.checked;
                LChain lhs = L.apply_d(q, ApplyMap(A, q, LChain{{u, B.idem()}}));
                chain_add(lhs, ApplyMap(A, q - 1, B.apply_d(q, LChain{{u, B.idem()}})), Rat::of(-1));
                if (!chain_zero(ln(q - 1, lhs))) fail(cert.axis_chain, "a^(" + tag + ") is not a chain map at q=" + std::to_string(q));
            }
        // over f_a: degree 0 e_0 -> v_S; degree 1 z_0 -> (S^{-1} - T) e
        if (!A.value.empty()) {
            ++cert.checked;
            const auto lhs = L.augment(0, ApplyMap(A, 0, LChain{{static_cast<uint32_t>(B.index(0, 'A', 0)), B.idem()}}));
            const auto rhs = L.augment(0, LChain{{static_cast<uint32_t>(L.index(0, 'S', 0)), te::one()}});
            if (lhs != rhs) fail(cert.axis_over_fa, "a^(" + tag + ") is not over f_a in degree 0");
        }
        if (A.value.size() > 1) {
            ++cert.checked;
            QGroupRingElt gam = te::g(Sinv());
            gam.add(te::g(Mat22::T()), Rat::of(-1));
            const auto lhs = L.augment(1, ApplyMap(A, 1, LChain{{static_cast<uint32_t>(B.index(1, 'Z', 0)), B.idem()}}));
            const auto rhs = L.augment(1, LChain{{static_cast<uint32_t>(L.index(1, 'E', 0)), gam}});
            if (lhs != rhs) fail(cert.axis_over_fa, "a^(" + tag + ") is not over f_a in degree 1");
        }
    };
    axis_checks(axis_, Bl_, L_, std::to_string(ell_));
    axis_checks(axis0_, B0_, T0, "0");
    // ---- (3)(b) Phi^Z_d, Psi^Z_d ----------------------------------------------------------------------------------------------
    for (uint32_t q = 0; q <= q_top_; ++q) {
        for (uint32_t y : isob_[q]) {
            for (const auto& [c, v] : phiZd_.value[q][y]) {
                ++cert.checked;
                if (!integral(v)) fail(cert.integral, "Phi^Z_d has a non-integral coefficient");
            }
            if (q == 0) continue;
            ++cert.checked;
            LChain a = B0_.apply_d(q, phiZd_.value[q][y]);
            chain_add(a, ApplyMap(phiZd_, q - 1, I.apply_d(q, unit(y))), Rat::of(-1));
            if (!chain_zero(a)) fail(cert.phiZd_chain, "Phi^Z_d is not a chain map at q=" + std::to_string(q));
        }
        for (uint32_t u = 0; u < B0_.rank(q); ++u) {
            for (const auto& [c, v] : psiZd_.value[q][u]) {
                ++cert.checked;
                if (!integral(v)) fail(cert.integral, "Psi^Z_d has a non-integral coefficient");
                if (!v.is_zero() && !is_boundary(q, c)) fail(cert.psiZd_in_isob, "Psi^Z_d leaves Isob at q=" + std::to_string(q));
            }
            if (q == 0) continue;
            ++cert.checked;
            LChain a = I.apply_d(q, psiZd_.value[q][u]);
            chain_add(a, ApplyMap(psiZd_, q - 1, B0_.apply_d(q, unit(u))), Rat::of(-1));
            if (!chain_zero(a)) fail(cert.psiZd_chain, "Psi^Z_d is not a chain map at q=" + std::to_string(q));
        }
    }
    {
        // over r_d o cmpI, resp. i_d o eps_B, in degrees 0 and 1 (eps_B of the free cusp model as (column, <c>-coset) multiplicities)
        const std::vector<Mat22> pwc = powers(Mat22::I().neg(), 2);
        auto epsB = [&](uint32_t q, const LChain& x) {
            std::map<std::pair<char, GKey>, Rat> o;
            for (const auto& [c, a] : x) {
                const CuspCell cc = B0_.cells(q)[c];
                if (cc.t != 0 || (q == 0) != (cc.k == 'A')) continue;
                for (const auto& [g, r] : a.terms) {
                    const GKey k = decompose(g, pwc).first;
                    o[{cc.k, k}] = o[{cc.k, k}] + r;
                }
            }
            for (auto it = o.begin(); it != o.end();) it = it->second.is_zero() ? o.erase(it) : std::next(it);
            return o;
        };
        for (uint32_t q = 0; q <= std::min<uint32_t>(1, q_top_); ++q) {
            for (uint32_t y : isob_[q]) {
                const LCell& c = I.cells(q)[y];
                LChain want;
                if (c.t == 0) {
                    const Simplex& sim = desc.reps(c.s)[c.rep];
                    if (c.s == 0) chain_add(want, static_cast<uint32_t>(B0_.index(0, 'A', 0)), te::one());
                    else if (sim == kE23) chain_add(want, static_cast<uint32_t>(B0_.index(1, 'Z', 0)), te::one(), Rat::of(-1));
                }
                ++cert.checked;
                if (epsB(q, phiZd_.value[q][y]) != epsB(q, want)) fail(cert.phiZd_over_r, "Phi^Z_d is not over r_d o cmpI");
            }
            for (uint32_t u = 0; u < B0_.rank(q); ++u) {
                const CuspCell c = B0_.cells(q)[u];
                LChain want;
                if (c.k == 'A' && c.t == 0 && q == 0) chain_add(want, cell_of(kV0), te::one());
                if (c.k == 'Z' && c.t == 0 && q == 1)
                    for (const Simplex* s : {&kE01, &kE12, &kE23}) chain_add(want, cell_of(*s), te::one(), Rat::of(-1));
                ++cert.checked;
                if (I.augment(q, psiZd_.value[q][u]) != I.augment(q, want)) fail(cert.psiZd_over_i, "Psi^Z_d is not over i_d o eps_B");
            }
        }
    }
    // ---- Phi^d_ell, Psi^d_ell ------------------------------------------------------------------------------------------------
    for (uint32_t q = 0; q <= q_top_; ++q) {
        for (uint32_t u = 0; u < Bl_.rank(q); ++u) {
            ++cert.checked;
            LChain a = norm_by(ApplyMap(phiDl_, q, psiDl_.value[q][u]), eB);
            chain_add(a, u, Bl_.idem(), Rat::of(-1));
            if (!chain_zero(norm_by(a, eB))) fail(cert.phid_psid_strict, "Phi^d_ell Psi^d_ell != id strictly at q=" + std::to_string(q));
            if (q >= 1) {
                ++cert.checked;
                LChain b = B0_.apply_d(q, psiDl_.value[q][u]);
                chain_add(b, ApplyMap(psiDl_, q - 1, Bl_.apply_d(q, LChain{{u, Bl_.idem()}})), Rat::of(-1));
                if (!chain_zero(b)) fail(cert.cusp_maps_chain, "Psi^d_ell is not a chain map at q=" + std::to_string(q));
            }
        }
        if (q >= 1)
            for (uint32_t y = 0; y < B0_.rank(q); ++y) {
                ++cert.checked;
                LChain b = Bl_.apply_d(q, phiDl_.value[q][y]);
                chain_add(b, ApplyMap(phiDl_, q - 1, B0_.apply_d(q, unit(y))), Rat::of(-1));
                if (!chain_zero(norm_by(b, eB))) fail(cert.cusp_maps_chain, "Phi^d_ell is not a chain map at q=" + std::to_string(q));
            }
    }
    // ---- (3)(c) the two squares ----------------------------------------------------------------------------------------------
    for (uint32_t q = 0; q <= q_top_; ++q) {
        for (uint32_t y : isob_[q]) {
            ++cert.checked;
            LChain lhs = phiT_.value[q][y];
            chain_add(lhs, ApplyMap(axis_, q, phid_.value[q][y]), Rat::of(-1));
            LChain rhs = L_.apply_d(q + 1, Yphi_.value[q][y]);
            if (q >= 1) chain_add(rhs, ApplyMap(Yphi_, q - 1, I.apply_d(q, unit(y))));
            chain_add(lhs, rhs, Rat::of(-1));
            if (!chain_zero(lnorm(q, lhs))) fail(cert.square_phi, "Phi_T j - a Phi_d != d Y_Phi + Y_Phi d at q=" + std::to_string(q));
        }
        for (uint32_t u = 0; u < Bl_.rank(q); ++u) {
            ++cert.checked;
            const LChain x{{u, Bl_.idem()}};
            LChain lhs = ApplyMap(psiT_, q, ApplyMap(axis_, q, x));
            chain_add(lhs, ApplyMap(psid_, q, x), Rat::of(-1));
            LChain rhs = I.apply_d(q + 1, ApplyMap(Ypsi_, q, x));
            if (q >= 1) chain_add(rhs, ApplyMap(Ypsi_, q - 1, Bl_.apply_d(q, x)));
            chain_add(lhs, rhs, Rat::of(-1));
            if (!chain_zero(lhs)) fail(cert.square_psi, "Psi_T a - j Psi_d != d Y_Psi + Y_Psi d at q=" + std::to_string(q));
        }
    }
    return cert;
}

}  // namespace integral_model
