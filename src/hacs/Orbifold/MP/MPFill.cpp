// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-4) The explicit filling of the MP resolution -- see MPFill.h.
#include "MPFill.h"

#include <algorithm>
#include <deque>
#include <iterator>
#include <stdexcept>

#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace orbifold_mp {

using integral_model::key_of;
using integral_model::mat_of;

namespace {

// The F_p of the 2-chain solve (the result is certified exactly over Z), as PeriodicityClass::fill_x1.
constexpr uint32_t kSolvePrime = 134217757u;
constexpr uint32_t kMaxRounds = 30;
constexpr size_t kMaxTriangles = 20000;
constexpr size_t kMaxBfs = 2000000;

int64_t Sgn(size_t i) { return (i % 2 == 0) ? 1 : -1; }

}  // namespace

MPFill::MPFill(const Descriptor& D) : D_(&D) {}

void MPFill::add(TotChain& c, const TotKey& k, int64_t n) {
    if (n == 0) return;
    auto it = c.find(k);
    if (it == c.end()) {
        c.emplace(k, n);
        return;
    }
    it->second = hecke::checked_add<int64_t>(it->second, n);
    if (it->second == 0) c.erase(it);
}

void MPFill::add(XChain& c, const XS& k, int64_t n) {
    if (n == 0) return;
    auto it = c.find(k);
    if (it == c.end()) {
        c.emplace(k, n);
        return;
    }
    it->second = hecke::checked_add<int64_t>(it->second, n);
    if (it->second == 0) c.erase(it);
}

void MPFill::add(StrChain& c, const Str& k, int64_t n) {
    if (n == 0) return;
    auto it = c.find(k);
    if (it == c.end()) {
        c.emplace(k, n);
        return;
    }
    it->second = hecke::checked_add<int64_t>(it->second, n);
    if (it->second == 0) c.erase(it);
}

const XS& MPFill::points(const Tr& u) const {
    auto it = points_.find(u);
    if (it != points_.end()) return it->second;
    const Mat22 h = mat_of(u.h);
    XS v;
    for (const Point& p : D_->lifted(u.t)) v.push_back(D_->key(Point{p.orbit, h * p.g}));
    std::sort(v.begin(), v.end());
    for (size_t i = 1; i < v.size(); ++i)
        ENGINE_CHECK_MSG(v[i - 1].first != v[i].first, "MPFill: a simplex with two vertices in one orbit ((IO) fails)");
    return points_.emplace(u, std::move(v)).first->second;
}

XS MPFill::common(const Str& s) const {
    ENGINE_CHECK(!s.empty());
    XS cur = points(s[0]);
    for (size_t i = 1; i < s.size() && !cur.empty(); ++i) {
        const XS& b = points(s[i]);
        XS out;
        std::set_intersection(cur.begin(), cur.end(), b.begin(), b.end(), std::back_inserter(out));
        cur.swap(out);
    }
    return cur;
}

bool MPFill::degenerate(const Str& s) {
    for (size_t i = 1; i < s.size(); ++i)
        if (s[i] == s[i - 1]) return true;
    return false;
}

int64_t MPFill::augmentation(const StrChain& c) {
    int64_t a = 0;
    for (const auto& [s, n] : c) {
        ENGINE_CHECK(s.size() == 1);
        a = hecke::checked_add<int64_t>(a, n);
    }
    return a;
}

StrChain MPFill::boundary(const StrChain& c) const {
    StrChain out;
    for (const auto& [s, n] : c) {
        ENGINE_CHECK(s.size() >= 2);
        for (size_t j = 0; j < s.size(); ++j) {
            Str f;
            f.reserve(s.size() - 1);
            for (size_t i = 0; i < s.size(); ++i)
                if (i != j) f.push_back(s[i]);
            if (degenerate(f)) continue;
            add(out, f, Sgn(j) * n);
        }
    }
    return out;
}

const std::vector<Tr>& MPFill::containing(const PointKey& P) const {
    auto it = containing_.find(P);
    if (it != containing_.end()) return it->second;
    const Mat22 g = mat_of(P.second);
    std::set<Tr> out;
    for (uint32_t t = 0; t < D_->top.size(); ++t)
        for (const Point& q : D_->lifted(t)) {
            if (q.orbit != P.first) continue;
            const Mat22 qinv = q.g.inv_unimodular();
            for (const Mat22& u : Elements(D_->orbit_stab.at(P.first))) out.insert(Tr{t, key_of(g * u * qinv)});
        }
    std::vector<Tr> v(out.begin(), out.end());
    for (const Tr& u : v) {
        const XS& pts = points(u);
        ENGINE_CHECK_MSG(std::binary_search(pts.begin(), pts.end(), P), "MPFill::containing: a translate misses its vertex");
    }
    return containing_.emplace(P, std::move(v)).first->second;
}

const Tr& MPFill::apex(const XS& x) const {
    auto it = apex_.find(x);
    if (it != apex_.end()) return it->second;
    ENGINE_CHECK(!x.empty());
    for (const Tr& u : containing(x[0])) {
        const XS& pts = points(u);
        if (std::includes(pts.begin(), pts.end(), x.begin(), x.end())) return apex_.emplace(x, u).first->second;
    }
    ENGINE_CHECK_MSG(false, "MPFill::apex: no translate contains the simplex");
    return apex_.begin()->second;
}

std::vector<PointKey> MPFill::neighbours(const PointKey& P) const {
    std::set<PointKey> out;
    for (const Tr& u : containing(P))
        for (const PointKey& q : points(u))
            if (q != P) out.insert(q);
    return std::vector<PointKey>(out.begin(), out.end());
}

std::vector<XS> MPFill::triangles_at_edge(const XS& e) const {
    ENGINE_CHECK(e.size() == 2);
    std::set<XS> out;
    for (const Tr& u : containing(e[0])) {
        const XS& pts = points(u);
        if (std::binary_search(pts.begin(), pts.end(), e[1])) out.insert(pts);
    }
    return std::vector<XS>(out.begin(), out.end());
}

MPFill::TotChain MPFill::dtot(const TotChain& c) const {
    TotChain out;
    for (const auto& [k, n] : c) {
        const XS& x = k.first;
        const Str& s = k.second;
        const size_t a = x.size() - 1, b = s.size() - 1;
        if (a >= 1)
            for (size_t i = 0; i <= a; ++i) {
                XS f;
                for (size_t r = 0; r <= a; ++r)
                    if (r != i) f.push_back(x[r]);
                add(out, TotKey{f, s}, Sgn(i) * n);
            }
        if (b >= 1) {
            const int64_t sg = Sgn(a);
            for (size_t j = 0; j <= b; ++j) {
                Str f;
                for (size_t r = 0; r <= b; ++r)
                    if (r != j) f.push_back(s[r]);
                if (degenerate(f)) continue;
                add(out, TotKey{x, f}, sg * Sgn(j) * n);
            }
        }
    }
    return out;
}

MPFill::XChain MPFill::fill_x0(const XChain& beta) const {
    // Breadth-first search from the hub (the least vertex of the support) to every other vertex.
    int64_t aug = 0;
    for (const auto& [x, n] : beta) {
        ENGINE_CHECK(x.size() == 1);
        aug = hecke::checked_add<int64_t>(aug, n);
    }
    ENGINE_CHECK_MSG(aug == 0, "MPFill::fill_x0: augmentation != 0");
    const PointKey hub = beta.begin()->first[0];
    std::set<PointKey> targets;
    for (const auto& [x, n] : beta)
        if (x[0] != hub) targets.insert(x[0]);
    std::map<PointKey, PointKey> parent;
    parent.emplace(hub, hub);
    std::deque<PointKey> queue{hub};
    size_t found = 0;
    while (!queue.empty() && found < targets.size()) {
        const PointKey P = queue.front();
        queue.pop_front();
        for (const PointKey& Q : neighbours(P)) {
            if (parent.count(Q)) continue;
            parent.emplace(Q, P);
            if (targets.count(Q)) ++found;
            queue.push_back(Q);
        }
        ENGINE_CHECK_MSG(parent.size() <= kMaxBfs, "MPFill::fill_x0: breadth-first search too large");
    }
    ENGINE_CHECK_MSG(found == targets.size(), "MPFill::fill_x0: a vertex not reached");
    stats_.bfs_max_visited = std::max<uint64_t>(stats_.bfs_max_visited, parent.size());
    XChain w;
    for (const auto& [x, n] : beta) {
        PointKey C = x[0];
        while (C != hub) {
            const PointKey Par = parent.at(C);
            // the step Par -> C of the path hub -> x: + [Par, C] in orbit order
            if (Par < C) add(w, XS{Par, C}, n);
            else add(w, XS{C, Par}, -n);
            C = Par;
        }
    }
    ++stats_.x0_fills;
    return w;
}

MPFill::XChain MPFill::fill_x1(const XChain& beta) const {
    auto tri_boundary = [&](const XS& t, int64_t mult, XChain& out) {
        ENGINE_CHECK(t.size() == 3);
        add(out, XS{t[1], t[2]}, mult);
        add(out, XS{t[0], t[2]}, -mult);
        add(out, XS{t[0], t[1]}, mult);
    };
    std::set<XS> cand;
    for (const auto& [e, n] : beta) {
        ENGINE_CHECK(e.size() == 2);
        for (const XS& t : triangles_at_edge(e)) cand.insert(t);
    }
    const engine::ModCtx<uint32_t> F(kSolvePrime);
    auto red = [&](int64_t v) {
        int64_t r = v % static_cast<int64_t>(kSolvePrime);
        if (r < 0) r += kSolvePrime;
        return static_cast<uint32_t>(r);
    };
    for (uint32_t round = 0; round < kMaxRounds; ++round) {
        std::map<XS, uint32_t> eidx;
        for (const auto& [e, n] : beta) eidx.emplace(e, 0);
        for (const XS& t : cand) {
            XChain b;
            tri_boundary(t, 1, b);
            for (const auto& [e, n] : b) eidx.emplace(e, 0);
        }
        uint32_t k = 0;
        for (auto& [e, i] : eidx) i = k++;
        const std::vector<XS> tris(cand.begin(), cand.end());
        engine::DenseBuf<uint32_t> B(static_cast<uint32_t>(eidx.size()), static_cast<uint32_t>(tris.size()));
        engine::DenseBuf<uint32_t> Y(static_cast<uint32_t>(eidx.size()), 1);
        for (uint32_t j = 0; j < tris.size(); ++j) {
            XChain b;
            tri_boundary(tris[j], 1, b);
            for (const auto& [e, n] : b) B.at(eidx.at(e), j) = F.add(B.at(eidx.at(e), j), red(n));
        }
        for (const auto& [e, n] : beta) Y.at(eidx.at(e), 0) = red(n);
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
                const int64_t lifted =
                    (v > kSolvePrime / 2) ? static_cast<int64_t>(v) - static_cast<int64_t>(kSolvePrime) : static_cast<int64_t>(v);
                w[tris[j]] = lifted;
            }
            XChain dw;
            for (const auto& [t, n] : w) tri_boundary(t, n, dw);
            if (dw == beta) {
                ++stats_.x1_fills;
                stats_.x1_max_rounds = std::max(stats_.x1_max_rounds, round + 1);
                stats_.x1_max_triangles = std::max<uint64_t>(stats_.x1_max_triangles, tris.size());
                return w;
            }
        }
        std::set<XS> next = cand;
        for (const auto& [e, i] : eidx)
            for (const XS& t : triangles_at_edge(e)) next.insert(t);
        ENGINE_CHECK_MSG(next.size() <= kMaxTriangles, "MPFill::fill_x1: candidate set too large");
        if (next.size() == cand.size()) break;
        cand.swap(next);
    }
    ENGINE_CHECK_MSG(false, "MPFill::fill_x1: no 2-chain of X bounds the 1-cycle");
    return {};
}

StrChain MPFill::fill(uint32_t k, const StrChain& z) const {
    ++stats_.fills;
    if (z.empty()) return {};
    for (const auto& [s, n] : z) ENGINE_CHECK_MSG(s.size() == k + 1 && !degenerate(s), "MPFill::fill: input not of degree k");
    if (k == 0) {
        ENGINE_CHECK_MSG(augmentation(z) == 0, "MPFill::fill: degree-0 input has non-zero augmentation");
    } else {
        ENGINE_CHECK_MSG(boundary(z).empty(), "MPFill::fill: input is not a cycle");
    }
    // (1) The lift and its correction to a Tot cycle (column cones, descending b).
    TotChain c;
    for (const auto& [s, n] : z) {
        const XS cm = common(s);
        ENGINE_CHECK_MSG(!cm.empty(), "MPFill::fill: a string with empty intersection");
        add(c, TotKey{XS{cm[0]}, s}, n);
    }
    if (k >= 1) {
        TotChain r = dtot(c);
        TotChain bacc;
        while (!r.empty()) {
            size_t bmax = 0;
            for (const auto& [key, n] : r) bmax = std::max(bmax, key.second.size() - 1);
            TotChain u;
            for (const auto& [key, n] : r) {
                if (key.second.size() - 1 != bmax) continue;
                const PointKey v = common(key.second).at(0);
                const XS& x = key.first;
                if (std::binary_search(x.begin(), x.end(), v)) continue;
                XS y = x;
                const auto pos = std::lower_bound(y.begin(), y.end(), v);
                const size_t ip = static_cast<size_t>(pos - y.begin());
                y.insert(pos, v);
                add(u, TotKey{y, key.second}, Sgn(ip) * n);
            }
            const TotChain du = dtot(u);
            for (const auto& [key, n] : u) add(bacc, key, n);
            for (const auto& [key, n] : du) add(r, key, -n);
            for (const auto& [key, n] : r)
                ENGINE_CHECK_MSG(key.second.size() - 1 < bmax, "MPFill::fill: the column step left a residue");
            stats_.max_tot_terms = std::max<uint64_t>(stats_.max_tot_terms, r.size() + bacc.size());
        }
        for (const auto& [key, n] : bacc) add(c, key, -n);
    }
    // (2) Fill the Tot cycle (row cones, descending a; the X-filling at b = 0).
    TotChain y, cur = c;
    auto collect = [&](size_t a) {
        TotChain comp;
        for (const auto& [key, n] : cur)
            if (key.first.size() == a + 1) comp.emplace(key, n);
        return comp;
    };
    for (int ai = static_cast<int>(std::min<uint32_t>(k, 2)); ai >= 0; --ai) {
        const size_t a = static_cast<size_t>(ai);
        TotChain comp = collect(a);
        if (comp.empty()) continue;
        if (a == k) {
            XChain beta;
            for (const auto& [key, n] : comp) add(beta, key.first, n);
            if (!beta.empty()) {
                ENGINE_CHECK_MSG(a <= 1, "MPFill::fill: a non-zero 2-cycle of X");
                const XChain w = (a == 0) ? fill_x0(beta) : fill_x1(beta);
                TotChain sw;
                for (const auto& [e, n] : w) add(sw, TotKey{e, Str{apex(e)}}, n);
                const TotChain dsw = dtot(sw);
                for (const auto& [key, n] : sw) add(y, key, n);
                for (const auto& [key, n] : dsw) add(cur, key, -n);
                comp = collect(a);
            }
        }
        TotChain u;
        const int64_t sg = Sgn(a);
        for (const auto& [key, n] : comp) {
            const Tr& ap = apex(key.first);
            if (key.second[0] == ap) continue;
            Str s2;
            s2.reserve(key.second.size() + 1);
            s2.push_back(ap);
            s2.insert(s2.end(), key.second.begin(), key.second.end());
            add(u, TotKey{key.first, s2}, sg * n);
        }
        const TotChain du = dtot(u);
        for (const auto& [key, n] : u) add(y, key, n);
        for (const auto& [key, n] : du) add(cur, key, -n);
        ENGINE_CHECK_MSG(collect(a).empty(), "MPFill::fill: the row step left a residue");
        stats_.max_tot_terms = std::max<uint64_t>(stats_.max_tot_terms, cur.size() + y.size());
    }
    ENGINE_CHECK_MSG(cur.empty(), "MPFill::fill: the staircase left a residue");
    // (3) alpha.
    StrChain out;
    for (const auto& [key, n] : y)
        if (key.first.size() == 1) add(out, key.second, n);
    ENGINE_CHECK_MSG(boundary(out) == z, "MPFill::fill: d y != z");
    return out;
}

Str ConcreteString(const InternedMPSet& S, uint32_t k, uint64_t nd_j, const Mat22& g0) {
    const uint64_t idx = S.nd(k).at(nd_j);
    const uint32_t* sg = S.sigma(k, idx);
    const uint32_t* gm = S.gamma(k, idx);
    Str s;
    s.reserve(k + 1);
    Mat22 h = g0;
    s.push_back(Tr{sg[0], key_of(h)});
    for (uint32_t i = 1; i <= k; ++i) {
        h = h * S.elements()[gm[i - 1]];
        s.push_back(Tr{sg[i], key_of(h)});
    }
    return s;
}

std::pair<GKey, uint64_t> FreeForm(const InternedMPSet& S, const Str& s) {
    const uint32_t k = static_cast<uint32_t>(s.size()) - 1;
    std::vector<uint32_t> sig(k + 1), gam(k);
    const auto& U = S.elements();
    for (uint32_t i = 0; i <= k; ++i) sig[i] = s[i].t;
    for (uint32_t i = 1; i <= k; ++i) {
        const Mat22 g = mat_of(s[i - 1].h).inv_unimodular() * mat_of(s[i].h);
        const GKey gk = key_of(g);
        const auto it = std::lower_bound(U.begin(), U.end(), gk,
                                         [](const Mat22& a, const GKey& b) { return key_of(a) < b; });
        ENGINE_CHECK_MSG(it != U.end() && key_of(*it) == gk, "orbifold_mp::FreeForm: an arrow outside the interned set");
        gam[i - 1] = static_cast<uint32_t>(it - U.begin());
    }
    const uint64_t full = S.find(k, sig.data(), k ? gam.data() : nullptr);
    ENGINE_CHECK_MSG(full != InternedMPSet::kNoIndex, "orbifold_mp::FreeForm: the string is not in S_k");
    const uint64_t nd = S.nd_index(k, full);
    ENGINE_CHECK_MSG(nd != InternedMPSet::kNoIndex, "orbifold_mp::FreeForm: a degenerate string");
    return {s[0].h, nd};
}

}  // namespace orbifold_mp
