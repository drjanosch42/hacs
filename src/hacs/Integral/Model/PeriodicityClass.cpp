// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Line 2, L2-1b (2) (hecke-extensions slot 10, WP-13) -- the class e of
// [C, lem:bulkperiodic](1) and the periodicity map E_q = e u - on the
// isotropy-resolution model.  See PeriodicityClass.h for the contract.
#include "PeriodicityClass.h"

#include <algorithm>
#include <deque>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "PeriodicColumn.h"

namespace integral_model {

using hecke::Mat22;

namespace {

using XKey = std::pair<uint32_t, GKey>;        // (representative, canonical coset key)
using XChain = std::map<XKey, int64_t>;

// The prime of the F_p solve inside fill_x1 (the result is certified exactly
// over Z afterwards, so the prime only has to exceed twice the largest
// coefficient of the unique solution; a failure to certify grows the
// candidate set, and a persistent failure is an ENGINE_CHECK).
constexpr uint32_t kFillPrime = 134217757u;
constexpr uint32_t kFillMaxRounds = 24;
constexpr size_t kFillMaxTriangles = 20000;
constexpr uint32_t kTreeMaxDepth = 64;

int64_t aug_of(const GroupRingElt& x) {
    int64_t s = 0;
    for (const auto& [g, n] : x.terms) s = hecke::checked_add<int64_t>(s, n);
    return s;
}

// The canonical key of gamma modulo {+-I} (every edge and triangle
// representative has stabiliser <-I>).
GKey canon_pm(const Mat22& g) {
    const GKey k1 = key_of(g), k2 = key_of(g.neg());
    return std::min(k1, k2);
}

// decompose / dF_coeff / column_homotopy: shared with StrictLift.cpp and
// IsotropyModel.cpp since slot-14 WP-14b (review finding slot-10 m-2) -- see
// PeriodicColumn.h (found by ordinary enclosing-namespace lookup, this file
// being inside `namespace integral_model` throughout). Previously
// reimplemented here because IsotropyModel keeps its copy private and WP-13
// used the public API only; PeriodicColumn.h is that shared public surface.

Chain left_mul(const GroupRingElt& a, const Chain& x) {
    Chain out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        if (!x[i].is_zero()) out[i] = a * x[i];
    return out;
}

void chain_add(Chain& a, const Chain& b, int64_t scale = 1) {
    ENGINE_CHECK(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        if (!b[i].is_zero()) a[i].add(b[i], scale);
}

bool chain_is_zero(const Chain& x) {
    for (const auto& c : x)
        if (!c.is_zero()) return false;
    return true;
}

// The generators of the words: S, T, T^{-1}, -I, with chi in twelfths.
struct WordGen {
    Mat22 g;
    uint32_t chi;
};

// A word of g in S, T, T^{-1}, -I (the Euclidean algorithm on the first
// column): g = T^{k_1} S T^{k_2} S ... (+-I) T^b.
std::vector<Mat22> word_of(const Mat22& g0) {
    const Mat22 T = Mat22::T(), Ti = T.inv_unimodular(), S = Mat22::S(), Si = S.inv_unimodular();
    ENGINE_CHECK_MSG(g0.det() == 1, "PeriodicityClass: word_of needs an element of SL_2(Z)");
    std::vector<Mat22> f;
    Mat22 m = g0;
    uint32_t guard = 0;
    while (m.c != 0) {
        const int64_t a = m.a, c = m.c;
        int64_t k = a / c;
        int64_t best = k;
        uint64_t bestv = hecke::uabs64(hecke::checked_sub<int64_t>(a, hecke::checked_mul<int64_t>(k, c)));
        for (int64_t dk : {int64_t(-1), int64_t(1)}) {
            const int64_t kk = k + dk;
            const uint64_t v = hecke::uabs64(hecke::checked_sub<int64_t>(a, hecke::checked_mul<int64_t>(kk, c)));
            if (v < bestv) { best = kk; bestv = v; }
        }
        k = best;
        for (int64_t i = 0; i < (k >= 0 ? k : -k); ++i) {
            f.push_back(k > 0 ? T : Ti);
            m = (k > 0 ? Ti : T) * m;
        }
        f.push_back(S);
        m = Si * m;
        ENGINE_CHECK_MSG(++guard < 100000, "PeriodicityClass: word_of did not terminate");
    }
    if (m.a == -1) {
        f.push_back(Mat22::I().neg());
        m = m.neg();
    }
    ENGINE_CHECK(m.a == 1 && m.d == 1 && m.c == 0);
    for (int64_t i = 0; i < (m.b >= 0 ? m.b : -m.b); ++i) f.push_back(m.b > 0 ? T : Ti);
    Mat22 p = Mat22::I();
    for (const auto& x : f) p = p * x;
    ENGINE_CHECK_MSG(p == g0, "PeriodicityClass: word_of product mismatch");
    return f;
}

uint32_t chi_of_gen(const Mat22& x) {
    if (x == Mat22::S()) return 3;
    if (x == Mat22::T()) return 11;
    if (x == Mat22::T().inv_unimodular()) return 1;
    if (x == Mat22::I().neg()) return 6;
    ENGINE_CHECK_MSG(false, "PeriodicityClass: not a word generator");
    return 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// chi and zeta_+ ([C, lem:bulkperiodic](1))
// ---------------------------------------------------------------------------
uint32_t Chi12(const Mat22& g) {
    uint32_t s = 0;
    for (const auto& x : word_of(g)) s = (s + chi_of_gen(x)) % 12;
    return s;
}

uint32_t ZetaPlusTwelfths(const Mat22& h) {
    if (h == Mat22::I()) return 0;
    if (h == Mat22::I().neg()) return 6;
    ENGINE_CHECK(h.det() == 1);
    const int64_t t = hecke::checked_add<int64_t>(h.a, h.d);
    ENGINE_CHECK_MSG(t >= -1 && t <= 1 && h.c != 0,
                     "ZetaPlusTwelfths: not an element of finite order (|tr| < 2 and c != 0 required)");
    const bool pos = h.c > 0;
    // zeta = t/2 + i sgn(c) sqrt(4 - t^2)/2 = exp(2 pi i k/12):
    //   t = 0: +-i (k = 3, 9); t = 1: exp(+-i pi/3) (k = 2, 10); t = -1: exp(+-2 pi i/3) (k = 4, 8).
    if (t == 0) return pos ? 3 : 9;
    if (t == 1) return pos ? 2 : 10;
    return pos ? 4 : 8;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
PeriodicityClass::PeriodicityClass(const IsotropyModel& M) : M_(&M) {
    ENGINE_CHECK_MSG(M.q_top() >= 4, "PeriodicityClass: the model must be built to degree >= 4");
    pow0_.resize(M.reps(0).size());
    for (uint32_t r = 0; r < M.reps(0).size(); ++r) {
        Mat22 p = Mat22::I();
        for (uint32_t i = 0; i < M.stabiliser_order(0, r); ++i) {
            pow0_[r].push_back(p);
            p = p * M.stabiliser_generator(0, r);
        }
    }
    for (uint32_t s = 1; s <= 2; ++s)
        for (uint32_t r = 0; r < M.reps(s).size(); ++r)
            ENGINE_CHECK_MSG(M.stabiliser_order(s, r) == 2 && M.stabiliser_generator(s, r) == Mat22::I().neg(),
                             "PeriodicityClass: edge / triangle stabilisers must be <-I>");
    build_paths();
    build_class();
    build_E();
    build_tree();
}

Chain PeriodicityClass::zero_chain(uint32_t q) const { return Chain(M_->rank(q)); }

Chain PeriodicityClass::apply_d(uint32_t q, const Chain& x) const {
    ENGINE_CHECK(q >= 1 && q <= M_->q_top() && x.size() == M_->rank(q));
    Chain out = zero_chain(q - 1);
    for (const auto& e : M_->differential(q))
        if (!x[e.col].is_zero()) out[e.row].add(x[e.col] * e.coeff);
    return out;
}

int64_t PeriodicityClass::augmentation(const Chain& x0) const {
    ENGINE_CHECK(x0.size() == M_->rank(0));
    int64_t s = 0;
    for (const auto& c : x0) s = hecke::checked_add<int64_t>(s, aug_of(c));
    return s;
}

void PeriodicityClass::build_paths() {
    const IsotropyModel& M = *M_;
    const uint32_t n0 = static_cast<uint32_t>(M.reps(0).size());
    ENGINE_CHECK(M.reps(0)[0] == Simplex({0}) && M.stabiliser_generator(0, 0) == Mat22::I().neg());
    path_.assign(n0, zero_chain(1));
    std::vector<bool> known(n0, false);
    known[0] = true;
    bool changed = true;
    while (changed) {
        changed = false;
        for (uint32_t j = 0; j < M.reps(1).size(); ++j) {
            const auto& f = M.faces(1, j);
            ENGINE_CHECK(f.size() == 2 && f[0].first == 1 && f[1].first == -1);
            const FaceTransport& fb = f[0].second;   // + end
            const FaceTransport& fa = f[1].second;   // - end
            if (!(fa.g == Mat22::I()) || !(fb.g == Mat22::I())) continue;
            const uint32_t ci = M.cell_index(1, j, 0);
            if (known[fa.rep] && !known[fb.rep]) {
                path_[fb.rep] = path_[fa.rep];
                path_[fb.rep][ci].add(key_of(Mat22::I()), 1);
                known[fb.rep] = changed = true;
            } else if (known[fb.rep] && !known[fa.rep]) {
                path_[fa.rep] = path_[fb.rep];
                path_[fa.rep][ci].add(key_of(Mat22::I()), -1);
                known[fa.rep] = changed = true;
            }
        }
    }
    for (uint32_t s = 0; s < n0; ++s) {
        ENGINE_CHECK_MSG(known[s], "PeriodicityClass: K's representative edges do not connect Sigma_0");
        Chain want = zero_chain(0);
        want[s].add(key_of(Mat22::I()), 1);
        want[0].add(key_of(Mat22::I()), -1);
        ENGINE_CHECK_MSG(apply_d(1, path_[s]) == want, "PeriodicityClass: d p_sigma != e_sigma - e_v0");
    }

    // c_gamma for the word generators: d c_gamma = (gamma - 1) e_{0,v0,0}.
    auto unit_chain = [&](uint32_t q, uint32_t cell, const Mat22& g) {
        Chain x = zero_chain(q);
        x[cell].add(key_of(g), 1);
        return x;
    };
    int v4 = -1, v6 = -1;
    for (uint32_t r = 0; r < n0; ++r) {
        if (M.stabiliser_generator(0, r) == Mat22::ST()) v4 = static_cast<int>(r);
        if (M.stabiliser_generator(0, r) == Mat22::S()) v6 = static_cast<int>(r);
    }
    ENGINE_CHECK_MSG(v4 >= 0 && v6 >= 0, "PeriodicityClass: no vertex with stabiliser <ST> / <S>");
    const Mat22 S = Mat22::S(), ST = Mat22::ST(), T = Mat22::T();
    const Mat22 Si = S.inv_unimodular(), Ti = T.inv_unimodular();
    auto loop_at = [&](int v, const Mat22& g) {
        Chain c = unit_chain(1, M.cell_index(0, static_cast<uint32_t>(v), 1), Mat22::I());
        chain_add(c, path_[v]);
        chain_add(c, left_mul(GroupRingElt::unit(g), path_[v]), -1);
        return c;
    };
    const Chain cNegI = unit_chain(1, M.cell_index(0, 0, 1), Mat22::I());
    const Chain cS = loop_at(v6, S);
    const Chain cST = loop_at(v4, ST);
    Chain cSi = left_mul(GroupRingElt::unit(Si, -1), cS);
    Chain cT = cSi;
    chain_add(cT, left_mul(GroupRingElt::unit(Si), cST));
    const Chain cTi = left_mul(GroupRingElt::unit(Ti, -1), cT);
    cgen_[key_of(Mat22::I().neg())] = cNegI;
    cgen_[key_of(S)] = cS;
    cgen_[key_of(T)] = cT;
    cgen_[key_of(Ti)] = cTi;
    for (const auto& [k, c] : cgen_) {
        Chain want = zero_chain(0);
        want[0].add(k, 1);
        want[0].add(key_of(Mat22::I()), -1);
        ENGINE_CHECK_MSG(apply_d(1, c) == want, "PeriodicityClass: d c_gamma != (gamma - 1) e_v0");
    }
}

Chain PeriodicityClass::c_gamma(const Mat22& g) const {
    Chain acc = zero_chain(1);
    Mat22 prefix = Mat22::I();
    for (const auto& x : word_of(g)) {
        const auto it = cgen_.find(key_of(x));
        ENGINE_CHECK(it != cgen_.end());
        chain_add(acc, left_mul(GroupRingElt::unit(prefix), it->second));
        prefix = prefix * x;
    }
    return acc;
}

void PeriodicityClass::build_class() {
    const IsotropyModel& M = *M_;
    // Phi on R_1.
    phi12_.assign(M.rank(1), 0);
    for (uint32_t ci = 0; ci < M.rank(1); ++ci) {
        const Cell& c = M.cells(1)[ci];
        if (c.s == 0) {
            phi12_[ci] = Chi12(M.stabiliser_generator(0, c.rep));
        } else {
            int64_t v = 0;
            for (const auto& [sign, ft] : M.faces(1, c.rep))
                v = hecke::checked_add<int64_t>(v, sign * static_cast<int64_t>(Chi12(ft.g)));
            phi12_[ci] = v;
        }
    }
    // e = (Phi o d_2) / 12.
    e_.assign(M.rank(2), 0);
    std::vector<int64_t> acc(M.rank(2), 0);
    for (const auto& en : M.differential(2))
        acc[en.col] = hecke::checked_add<int64_t>(acc[en.col], hecke::checked_mul<int64_t>(aug_of(en.coeff), phi12_[en.row]));
    for (uint32_t c = 0; c < M.rank(2); ++c) {
        ENGINE_CHECK_MSG(acc[c] % 12 == 0, "PeriodicityClass: Phi o d_2 not divisible by 12 (chi not a cocycle mod Z)");
        e_[c] = acc[c] / 12;
    }
}

void PeriodicityClass::build_E() {
    const IsotropyModel& M = *M_;
    const uint32_t qtop = M.q_top();
    E_.assign(qtop - 1, {});
    // E_0.
    E_[0].assign(M.rank(2), zero_chain(0));
    for (uint32_t c = 0; c < M.rank(2); ++c)
        if (e_[c] != 0) E_[0][c][0].add(key_of(Mat22::I()), e_[c]);
    // E_q, q >= 1, generator by generator.
    for (uint32_t q = 1; q + 2 <= qtop; ++q) {
        std::vector<std::vector<const DiffEntry*>> bycol(M.rank(q + 2));
        for (const auto& en : M.differential(q + 2)) bycol[en.col].push_back(&en);
        E_[q].assign(M.rank(q + 2), zero_chain(q));
        for (uint32_t c = 0; c < M.rank(q + 2); ++c) {
            Chain z = zero_chain(q - 1);
            for (const DiffEntry* en : bycol[c]) chain_add(z, left_mul(en->coeff, E_[q - 1][en->row]));
            E_[q][c] = fill(q - 1, z);
        }
    }
}

// ---------------------------------------------------------------------------
// Fillings
// ---------------------------------------------------------------------------
Chain PeriodicityClass::fill0(const Chain& z) const {
    ENGINE_CHECK_MSG(augmentation(z) == 0, "PeriodicityClass::fill: degree-0 input has non-zero augmentation");
    Chain y = zero_chain(1);
    for (uint32_t s = 0; s < z.size(); ++s)
        for (const auto& [g, n] : z[s].terms) {
            const Mat22 G = mat_of(g);
            chain_add(y, left_mul(GroupRingElt::unit(G), path_[s]), n);
            chain_add(y, c_gamma(G), n);
        }
    ENGINE_CHECK_MSG(apply_d(1, y) == z, "PeriodicityClass::fill: degree-0 filling failed");
    return y;
}

Chain PeriodicityClass::fill(uint32_t q, const Chain& z) const {
    const IsotropyModel& M = *M_;
    ENGINE_CHECK(q + 1 <= M.q_top() && z.size() == M.rank(q));
    if (q == 0) return fill0(z);
    ENGINE_CHECK_MSG(chain_is_zero(apply_d(q, z)), "PeriodicityClass::fill: input is not a cycle");

    // Per-column powers of the stabiliser generators.
    auto powers = [&](uint32_t s, uint32_t r) {
        std::vector<Mat22> pw;
        Mat22 p = Mat22::I();
        for (uint32_t i = 0; i < M.stabiliser_order(s, r); ++i) { pw.push_back(p); p = p * M.stabiliser_generator(s, r); }
        return pw;
    };

    Chain r = z;
    Chain y = zero_chain(q + 1);
    for (int si = static_cast<int>(std::min<uint32_t>(q, 2)); si >= 0; --si) {
        const uint32_t s = static_cast<uint32_t>(si);
        const uint32_t t = q - s;
        const uint32_t nrep = static_cast<uint32_t>(M.reps(s).size());
        std::vector<GroupRingElt> rs(nrep);
        auto extract = [&]() {
            bool nz = false;
            for (uint32_t rep = 0; rep < nrep; ++rep) {
                rs[rep] = r[M.cell_index(s, rep, t)];
                nz = nz || !rs[rep].is_zero();
            }
            return nz;
        };
        if (!extract()) continue;

        if (s == 1 && t == 0) {
            // eps(r) is a 1-cycle of Xbulk: fill it there, lift by the section.
            XChain x;
            for (uint32_t rep = 0; rep < nrep; ++rep)
                for (const auto& [g, n] : rs[rep].terms) {
                    auto& v = x[{rep, canon_pm(mat_of(g))}];
                    v = hecke::checked_add<int64_t>(v, n);
                    if (v == 0) x.erase({rep, canon_pm(mat_of(g))});
                }
            if (!x.empty()) {
                const XChain w = fill_x1(x);
                Chain wh = zero_chain(2);
                for (const auto& [k, n] : w) wh[M.cell_index(2, k.first, 0)].add(k.second, n);
                chain_add(y, wh);
                chain_add(r, apply_d(2, wh), -1);
                if (!extract()) continue;
            }
        }

        // The column step: rs is a cycle of the augmented column.
        std::vector<GroupRingElt> u(nrep);
        for (uint32_t rep = 0; rep < nrep; ++rep) {
            if (rs[rep].is_zero()) continue;
            const auto pw = powers(s, rep);
            // Input condition.
            if (t == 0) {
                std::map<GKey, int64_t> cosets;
                for (const auto& [g, n] : rs[rep].terms) {
                    auto& v = cosets[decompose(g, pw).first];
                    v = hecke::checked_add<int64_t>(v, n);
                }
                for (const auto& [c, n] : cosets)
                    ENGINE_CHECK_MSG(n == 0, "PeriodicityClass::fill: t = 0 column residue has non-zero augmentation");
            } else {
                ENGINE_CHECK_MSG((rs[rep] * dF_coeff(pw, t)).is_zero(),
                                 "PeriodicityClass::fill: column residue is not a d^F-cycle");
            }
            // (slot-14 WP-14b, m-2) The column homotopy, folded from an
            // inline copy of column_homotopy's body into a call to it.
            u[rep] = column_homotopy(rs[rep], pw, t);
        }
        Chain yc = zero_chain(q + 1);
        const int64_t sign = (s % 2 == 1) ? -1 : 1;
        for (uint32_t rep = 0; rep < nrep; ++rep)
            if (!u[rep].is_zero()) yc[M.cell_index(s, rep, t + 1)].add(u[rep], sign);
        chain_add(y, yc);
        chain_add(r, apply_d(q + 1, yc), -1);
        ENGINE_CHECK_MSG(!extract(), "PeriodicityClass::fill: column step left a residue");
    }
    ENGINE_CHECK_MSG(chain_is_zero(r), "PeriodicityClass::fill: staircase left a residue");
    ENGINE_CHECK_MSG(apply_d(q + 1, y) == z, "PeriodicityClass::fill: d y != z");
    return y;
}

std::map<std::pair<uint32_t, GKey>, int64_t> PeriodicityClass::fill_x1(const XChain& x) const {
    const IsotropyModel& M = *M_;
    const uint32_t ntri = static_cast<uint32_t>(M.reps(2).size());
    auto tri_boundary = [&](const XKey& tk, int64_t mult, XChain& out) {
        const Mat22 G = mat_of(tk.second);
        for (const auto& [sign, ft] : M.faces(2, tk.first)) {
            const XKey ek{ft.rep, canon_pm(G * ft.g)};
            auto& v = out[ek];
            v = hecke::checked_add<int64_t>(v, hecke::checked_mul<int64_t>(sign, mult));
            if (v == 0) out.erase(ek);
        }
    };
    auto adjacent = [&](const XKey& ek, std::set<XKey>& out) {
        const Mat22 G = mat_of(ek.second);
        for (uint32_t r = 0; r < ntri; ++r)
            for (const auto& [sign, ft] : M.faces(2, r))
                if (ft.rep == ek.first) out.insert({r, canon_pm(G * ft.g.inv_unimodular())});
    };

    std::set<XKey> cand;
    for (const auto& [ek, n] : x) adjacent(ek, cand);
    const engine::ModCtx<uint32_t> F(kFillPrime);
    for (uint32_t round = 0; round < kFillMaxRounds; ++round) {
        std::map<XKey, uint32_t> eidx;
        for (const auto& [ek, n] : x) eidx.emplace(ek, 0);
        for (const auto& tk : cand) {
            XChain b;
            tri_boundary(tk, 1, b);
            for (const auto& [ek, n] : b) eidx.emplace(ek, 0);
        }
        uint32_t k = 0;
        for (auto& [ek, i] : eidx) i = k++;
        const std::vector<XKey> tris(cand.begin(), cand.end());
        engine::DenseBuf<uint32_t> B(static_cast<uint32_t>(eidx.size()), static_cast<uint32_t>(tris.size()));
        engine::DenseBuf<uint32_t> Y(static_cast<uint32_t>(eidx.size()), 1);
        auto red = [&](int64_t v) {
            int64_t r = v % static_cast<int64_t>(kFillPrime);
            if (r < 0) r += kFillPrime;
            return static_cast<uint32_t>(r);
        };
        for (uint32_t j = 0; j < tris.size(); ++j) {
            XChain b;
            tri_boundary(tris[j], 1, b);
            for (const auto& [ek, n] : b) B.at(eidx.at(ek), j) = F.add(B.at(eidx.at(ek), j), red(n));
        }
        for (const auto& [ek, n] : x) Y.at(eidx.at(ek), 0) = red(n);
        bool solved = false;
        std::vector<std::vector<uint32_t>> sol;
        if (!tris.empty()) {
            try {
                sol = engine::elim::solve_multi_rhs(F, engine::cview(B), engine::cview(Y));
                solved = true;
            } catch (const std::runtime_error&) {
                solved = false;
            }
        }
        if (solved) {
            XChain w;
            for (uint32_t j = 0; j < tris.size(); ++j) {
                const uint32_t v = sol[0][j];
                if (v == 0) continue;
                const int64_t lifted = (v > kFillPrime / 2) ? static_cast<int64_t>(v) - kFillPrime : static_cast<int64_t>(v);
                w[tris[j]] = lifted;
            }
            XChain dw;
            for (const auto& [tk, n] : w) tri_boundary(tk, n, dw);
            if (dw == x) {
                ++x_fillings_;
                x_max_rounds_ = std::max(x_max_rounds_, round + 1);
                x_max_triangles_ = std::max(x_max_triangles_, static_cast<uint32_t>(tris.size()));
                return w;
            }
        }
        std::set<XKey> next = cand;
        for (const auto& [ek, i] : eidx) adjacent(ek, next);
        ENGINE_CHECK_MSG(next.size() <= kFillMaxTriangles, "PeriodicityClass::fill_x1: candidate set too large");
        if (next.size() == cand.size()) break;
        cand.swap(next);
    }
    ENGINE_CHECK_MSG(false, "PeriodicityClass::fill_x1: no 2-chain of Xbulk bounds the 1-cycle");
    return {};
}

// ---------------------------------------------------------------------------
// The Bass-Serre tree and the connecting map
// ---------------------------------------------------------------------------
namespace {

using TVert = std::pair<int, GKey>;   // (0 = gamma<S>, 1 = gamma<U>), canonical coset key
using TChain = std::map<GKey, int64_t>;   // edge gamma<-I> (canonical mod +-I) -> multiplicity

GKey canon_coset(const Mat22& g, const Mat22& h, uint32_t m) {
    GKey best = key_of(g);
    Mat22 p = g;
    for (uint32_t i = 1; i < m; ++i) {
        p = p * h;
        best = std::min(best, key_of(p));
    }
    return best;
}
TVert tvert(int type, const Mat22& g) {
    return type == 0 ? TVert{0, canon_coset(g, Mat22::S(), 4)} : TVert{1, canon_coset(g, Mat22::ST(), 6)};
}

// Neighbours of a tree vertex: (edge key, orientation sign when moving from v
// to the neighbour, neighbour).  d[gamma<-I>] = gamma<U> - gamma<S>.
std::vector<std::tuple<GKey, int, TVert>> tneighbours(const TVert& v) {
    std::vector<std::tuple<GKey, int, TVert>> out;
    const Mat22 G = mat_of(v.second);
    if (v.first == 0) {
        Mat22 p = G;
        for (int i = 0; i < 2; ++i) {
            out.emplace_back(canon_pm(p), +1, tvert(1, p));
            p = p * Mat22::S();
        }
    } else {
        Mat22 p = G;
        for (int j = 0; j < 3; ++j) {
            out.emplace_back(canon_pm(p), -1, tvert(0, p));
            p = p * Mat22::ST();
        }
    }
    return out;
}

bool tgeodesic(const TVert& a, const TVert& b, TChain& out) {
    out.clear();
    if (a == b) return true;
    std::map<TVert, std::tuple<TVert, GKey, int>> parent;
    std::map<TVert, uint32_t> depth;
    std::deque<TVert> qu;
    qu.push_back(a);
    depth[a] = 0;
    while (!qu.empty()) {
        const TVert v = qu.front();
        qu.pop_front();
        if (v == b) break;
        if (depth[v] >= kTreeMaxDepth) continue;
        for (const auto& [ek, sg, w] : tneighbours(v)) {
            if (depth.count(w)) continue;
            depth[w] = depth[v] + 1;
            parent[w] = {v, ek, sg};
            qu.push_back(w);
        }
    }
    if (!depth.count(b)) return false;
    for (TVert v = b; !(v == a);) {
        const auto& [pv, ek, sg] = parent.at(v);
        auto& m = out[ek];
        m += sg;
        if (m == 0) out.erase(ek);
        v = pv;
    }
    return true;
}

std::map<TVert, int64_t> tboundary(const TChain& c) {
    std::map<TVert, int64_t> out;
    for (const auto& [ek, m] : c) {
        const Mat22 G = mat_of(ek);
        out[tvert(1, G)] += m;
        out[tvert(0, G)] -= m;
    }
    for (auto it = out.begin(); it != out.end();) it = (it->second == 0) ? out.erase(it) : std::next(it);
    return out;
}

}  // namespace

void PeriodicityClass::build_tree() {
    const IsotropyModel& M = *M_;
    tree_ok_ = true;
    auto fail = [&](const std::string& why) {
        tree_ok_ = false;
        if (tree_failure_.empty()) tree_failure_ = why;
    };
    // phi_0: v6 (stabiliser <S>) -> <S>, every other representative -> <U>.
    auto phi0 = [&](uint32_t rep, const Mat22& g) {
        return tvert(M.stabiliser_generator(0, rep) == Mat22::S() ? 0 : 1, g);
    };
    tree_phi1_.assign(M.reps(1).size(), {});
    for (uint32_t j = 0; j < M.reps(1).size(); ++j) {
        const auto& f = M.faces(1, j);
        const TVert b = phi0(f[0].second.rep, f[0].second.g);   // + end
        const TVert a = phi0(f[1].second.rep, f[1].second.g);   // - end
        if (!tgeodesic(a, b, tree_phi1_[j])) { fail("tree geodesic not found for edge " + std::to_string(j)); continue; }
        std::map<TVert, int64_t> want;
        if (!(a == b)) { want[b] += 1; want[a] -= 1; }
        if (tboundary(tree_phi1_[j]) != want) fail("d phi_1 != phi_0 d on edge " + std::to_string(j));
    }
    for (uint32_t r = 0; r < M.reps(2).size(); ++r) {
        TChain acc;
        for (const auto& [sign, ft] : M.faces(2, r))
            for (const auto& [ek, m] : tree_phi1_[ft.rep]) {
                const GKey k = canon_pm(ft.g * mat_of(ek));
                auto& v = acc[k];
                v += sign * m;
                if (v == 0) acc.erase(k);
            }
        if (!acc.empty()) fail("phi_1 d != 0 on triangle " + std::to_string(r));
    }
}

// ---------------------------------------------------------------------------
// Certificate
// ---------------------------------------------------------------------------
PeriodicityCertificate PeriodicityClass::certify() const {
    const IsotropyModel& M = *M_;
    PeriodicityCertificate cert;
    auto fail = [&](const std::string& why) {
        if (cert.first_failure.empty()) cert.first_failure = why;
    };

    // e a cocycle; 12 e = Phi o d_2.
    {
        std::vector<int64_t> de(M.rank(3), 0), dphi(M.rank(2), 0);
        for (const auto& en : M.differential(3))
            de[en.col] = hecke::checked_add<int64_t>(de[en.col], hecke::checked_mul<int64_t>(aug_of(en.coeff), e_[en.row]));
        for (const auto& en : M.differential(2))
            dphi[en.col] = hecke::checked_add<int64_t>(dphi[en.col], hecke::checked_mul<int64_t>(aug_of(en.coeff), phi12_[en.row]));
        cert.e_cocycle = std::all_of(de.begin(), de.end(), [](int64_t v) { return v == 0; });
        if (!cert.e_cocycle) fail("e o d_3 != 0");
        cert.e_bockstein = true;
        for (uint32_t c = 0; c < M.rank(2); ++c)
            if (dphi[c] != hecke::checked_mul<int64_t>(12, e_[c])) cert.e_bockstein = false;
        if (!cert.e_bockstein) fail("12 e != Phi o d_2");
    }
    // Restrictions to the vertex stabilisers, and chi == zeta_+.
    {
        cert.e_restrictions = true;
        cert.chi_is_zeta = true;
        // (S13 WP-12, slot-10 m-10) The 64-element cap is arbitrary, and chi and zeta_+ are both
        // conjugation-invariant, so the conjugates prove nothing beyond the stabiliser powers.
        std::vector<Mat22> conj;
        for (const auto& k : M.group_elements()) {
            if (conj.size() >= 64) break;
            conj.push_back(mat_of(k));
        }
        conj.push_back(Mat22::T());
        conj.push_back(Mat22::T().inv_unimodular());
        for (uint32_t r = 0; r < M.reps(0).size(); ++r) {
            const Mat22 g = M.stabiliser_generator(0, r);
            const int64_t m = M.stabiliser_order(0, r);
            const int64_t z = ZetaPlusTwelfths(g);
            ENGINE_CHECK((z * m) % 12 == 0);
            const int64_t want = (z * m) / 12;
            const int64_t got = e_[M.cell_index(0, r, 2)];
            if (((got - want) % m + m) % m != 0) {
                cert.e_restrictions = false;
                fail("res_C e != chi_C at vertex representative " + std::to_string(r));
            }
            Mat22 h = Mat22::I();
            for (int64_t j = 0; j < m; ++j) {
                for (size_t ci = 0; ci <= conj.size(); ++ci) {
                    const Mat22 hh = (ci == conj.size()) ? h : conj[ci] * h * conj[ci].inv_unimodular();
                    ++cert.finite_order_elements_checked;
                    if (Chi12(hh) != ZetaPlusTwelfths(hh)) {
                        cert.chi_is_zeta = false;
                        fail("chi != zeta_+ on a finite-order element");
                    }
                }
                h = h * g;
            }
        }
    }
    // aug o E_0 = e.
    cert.aug_E0 = true;
    for (uint32_t c = 0; c < M.rank(2); ++c)
        if (augmentation(E_[0][c]) != e_[c]) cert.aug_E0 = false;
    if (!cert.aug_E0) fail("aug o E_0 != e");
    // d_q E_q = E_{q-1} d_{q+2}.
    cert.chain_map = true;
    for (uint32_t q = 1; q <= top(); ++q) {
        std::vector<std::vector<const DiffEntry*>> bycol(M.rank(q + 2));
        for (const auto& en : M.differential(q + 2)) bycol[en.col].push_back(&en);
        for (uint32_t c = 0; c < M.rank(q + 2); ++c) {
            Chain rhs = zero_chain(q - 1);
            for (const DiffEntry* en : bycol[c]) chain_add(rhs, left_mul(en->coeff, E_[q - 1][en->row]));
            const Chain lhs = apply_d(q, E_[q][c]);
            ++cert.chain_terms_checked;
            if (!(lhs == rhs)) {
                cert.chain_map = false;
                fail("d E_" + std::to_string(q) + " != E_" + std::to_string(q - 1) + " d at generator " + std::to_string(c));
            }
        }
    }
    cert.tree_chain_map = tree_ok_;
    if (!tree_ok_) fail(tree_failure_);
    // Integer sizes.
    std::set<GKey> seen;
    for (const auto& Eq : E_)
        for (const auto& ch : Eq)
            for (const auto& x : ch) {
                cert.max_abs_coefficient = std::max(cert.max_abs_coefficient, x.max_abs_coefficient());
                cert.max_abs_entry = std::max(cert.max_abs_entry, x.max_abs_entry());
                cert.total_terms += x.terms.size();
                for (const auto& [g, n] : x.terms) seen.insert(g);
            }
    cert.distinct_group_elements = seen.size();
    return cert;
}

void PeriodicityClass::perturb_E_for_negative_control(uint32_t q, uint32_t c, uint32_t b) {
    E_.at(q).at(c).at(b).add(key_of(Mat22::I()), 1);
}

void PeriodicityClass::perturb_e_for_negative_control(uint32_t c) { e_.at(c) += 1; }

// ---------------------------------------------------------------------------
// Matrices over a ModCtx
// ---------------------------------------------------------------------------
template<class Elt>
engine::DenseBuf<Elt> PeriodicityClass::matrix(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const {
    const IsotropyModel& M = *M_;
    ENGINE_CHECK(q <= top());
    const uint32_t D = n + 1;
    engine::DenseBuf<Elt> out(M.rank(q + 2) * D, M.rank(q) * D);
    std::map<GKey, engine::DenseBuf<Elt>> memo;
    const int64_t pm = static_cast<int64_t>(ctx.p);
    for (uint32_t c = 0; c < M.rank(q + 2); ++c)
        for (uint32_t b = 0; b < M.rank(q); ++b)
            for (const auto& [g, nm] : E_[q][c][b].terms) {
                int64_t rr = nm % pm;
                if (rr < 0) rr += pm;
                const Elt coef = static_cast<Elt>(rr);
                if (coef == 0) continue;
                auto it = memo.find(g);
                if (it == memo.end()) it = memo.emplace(g, rho_n(ctx, mat_of(g), n)).first;
                const auto& R = it->second;
                for (uint32_t i = 0; i < D; ++i)
                    for (uint32_t j = 0; j < D; ++j) {
                        Elt& dst = out.at(c * D + i, b * D + j);
                        dst = ctx.add(dst, ctx.mul(coef, R.at(i, j)));
                    }
            }
    return out;
}

template<class Elt>
engine::DenseBuf<Elt> PeriodicityClass::connecting_cochains(const engine::ModCtx<Elt>& ctx, uint32_t n) const {
    const IsotropyModel& M = *M_;
    const uint32_t D = n + 1;
    engine::DenseBuf<Elt> out(M.rank(1) * D, D);
    // N = I + rho_n(-I).
    engine::DenseBuf<Elt> N = rho_n(ctx, Mat22::I().neg(), n);
    for (uint32_t i = 0; i < D; ++i) N.at(i, i) = ctx.add(N.at(i, i), static_cast<Elt>(1 % ctx.p));
    const int64_t pm = static_cast<int64_t>(ctx.p);
    for (uint32_t j = 0; j < M.reps(1).size(); ++j) {
        engine::DenseBuf<Elt> blk(D, D);
        for (const auto& [ek, m] : tree_phi1_[j]) {
            const auto R = rho_n(ctx, mat_of(ek), n);
            int64_t rr = m % pm;
            if (rr < 0) rr += pm;
            const Elt coef = static_cast<Elt>(rr);
            for (uint32_t a = 0; a < D; ++a)
                for (uint32_t b = 0; b < D; ++b) blk.at(a, b) = ctx.add(blk.at(a, b), ctx.mul(coef, R.at(a, b)));
        }
        const uint32_t ci = M.cell_index(1, j, 0);
        for (uint32_t a = 0; a < D; ++a)
            for (uint32_t b = 0; b < D; ++b) {
                Elt acc = 0;
                for (uint32_t k = 0; k < D; ++k) acc = ctx.add(acc, ctx.mul(blk.at(a, k), N.at(k, b)));
                out.at(ci * D + a, b) = acc;
            }
    }
    return out;
}

#define PERIODICITY_INSTANTIATE(Elt)                                                                          \
    template engine::DenseBuf<Elt> PeriodicityClass::matrix<Elt>(const engine::ModCtx<Elt>&, uint32_t,        \
                                                                 uint32_t) const;                           \
    template engine::DenseBuf<Elt> PeriodicityClass::connecting_cochains<Elt>(const engine::ModCtx<Elt>&,     \
                                                                              uint32_t) const;

PERIODICITY_INSTANTIATE(uint32_t)
PERIODICITY_INSTANTIATE(uint64_t)
#undef PERIODICITY_INSTANTIATE

}  // namespace integral_model
