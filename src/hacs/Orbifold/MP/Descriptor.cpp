// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-1a) Orbifold triangulations as finite data and
// the decidable MP predicates -- see Descriptor.h for the contract.
#include "Descriptor.h"

#include <algorithm>
#include <sstream>

#include "../../../ModArithmetic/EngineCheck.h"

namespace orbifold_mp {

using integral_model::key_of;

// ---------------------------------------------------------------------------
// Cyclic groups
// ---------------------------------------------------------------------------
std::vector<Mat22> Elements(const CyclicGroup& H) {
    std::vector<Mat22> out;
    Mat22 x = Mat22::I();
    for (uint32_t j = 0; j < H.order; ++j) {
        out.push_back(x);
        x = x * H.gen;
    }
    ENGINE_CHECK_MSG(x == Mat22::I(), "orbifold_mp: a cyclic group's generator has the wrong order");
    return out;
}

bool Contains(const CyclicGroup& H, const Mat22& x) {
    Mat22 y = Mat22::I();
    for (uint32_t j = 0; j < H.order; ++j) {
        if (y == x) return true;
        y = y * H.gen;
    }
    return false;
}

bool IsSubgroup(const CyclicGroup& A, const CyclicGroup& B) { return Contains(B, A.gen); }

CyclicGroup Conjugate(const Mat22& g, const CyclicGroup& H) {
    return CyclicGroup{g * H.gen * g.inv_unimodular(), H.order};
}

CyclicGroup Sylow(const CyclicGroup& H, uint32_t ell) {
    if (ell == 0) return H;
    ENGINE_CHECK_MSG(ell >= 2, "orbifold_mp::Sylow: ell must be 0 (full) or a prime");
    uint32_t part = 1, m = H.order;
    while (m % ell == 0) {
        m /= ell;
        part *= ell;
    }
    Mat22 g = Mat22::I();
    for (uint32_t j = 0; j < m; ++j) g = g * H.gen;   // gen^{order / part}, of order exactly `part`
    return CyclicGroup{g, part};
}

// ---------------------------------------------------------------------------
// The descriptor
// ---------------------------------------------------------------------------
PointKey Descriptor::key(const Point& p) const {
    // Least GKey of g * Gamma_orbit (allocation-free: this is the enumeration's inner loop).
    const CyclicGroup& H = orbit_stab.at(p.orbit);
    Mat22 x = p.g;
    GKey best = key_of(x);
    for (uint32_t j = 1; j < H.order; ++j) {
        x = x * H.gen;
        const GKey k = key_of(x);
        if (k < best) best = k;
    }
    return {p.orbit, best};
}

CyclicGroup Descriptor::stabiliser(const Point& p) const { return Conjugate(p.g, orbit_stab.at(p.orbit)); }

std::vector<Point> Descriptor::lifted(uint32_t t) const {
    std::vector<Point> out;
    for (uint32_t v : top.at(t)) out.push_back(point.at(v));
    return out;
}

Descriptor FromModelData(const integral_model::ModelData& d, const std::string& name) {
    Descriptor D;
    D.name = name;
    for (uint32_t r = 0; r < d.reps[0].size(); ++r) {
        ENGINE_CHECK(d.reps[0][r].size() == 1);
        const uint32_t label = d.reps[0][r][0];
        D.orbit_label.push_back(label);
        D.orbit_stab.push_back(CyclicGroup{d.gen[0][r], d.order[0][r]});
        ENGINE_CHECK_MSG(D.point.emplace(label, Point{r, Mat22::I()}).second, "orbifold_mp: a vertex label listed twice");
    }
    for (const auto& nr : d.nonreps) {
        if (nr.face.size() == 1)
            ENGINE_CHECK_MSG(D.point.emplace(nr.face[0], Point{nr.rep, nr.g}).second, "orbifold_mp: a vertex label listed twice");
        ENGINE_CHECK_MSG(nr.face.size() <= 2, "orbifold_mp: a non-representative triangle (Sigma_top must be the representatives)");
    }
    for (const auto& t : d.reps[2]) D.top.push_back(t);
    for (uint32_t v : d.horocycle) D.boundary.insert(v);
    for (const auto& t : D.top)
        for (uint32_t v : t) ENGINE_CHECK_MSG(D.point.count(v) == 1, "orbifold_mp: a triangle vertex is not a point of the descriptor");
    return D;
}

Descriptor RGoodDescriptor() {
    // mp_rules.py make("R_good"), label for label.  Orbits: 0, 1, 2 (horocycle), 4 = rho^2 (<ST>), 6 = i (<S>),
    // 13, 14 (the arc rho^2 - i cut twice), 17 p, 18 q, 19 c, 20 p', 21 q', 22 c' -- all central except 4, 6.
    const Mat22 I = Mat22::I(), T = Mat22::T(), S = Mat22::S();
    Descriptor D;
    D.name = "R_good";
    const std::vector<uint32_t> reps = {0, 1, 2, 4, 6, 13, 14, 17, 18, 19, 20, 21, 22};
    for (uint32_t i = 0; i < reps.size(); ++i) {
        D.orbit_label.push_back(reps[i]);
        CyclicGroup H{I.neg(), 2};
        if (reps[i] == 4) H = CyclicGroup{Mat22::ST(), 6};
        if (reps[i] == 6) H = CyclicGroup{S, 4};
        D.orbit_stab.push_back(H);
        D.point[reps[i]] = Point{i, I};
    }
    auto orbit_of = [&](uint32_t label) {
        return static_cast<uint32_t>(std::find(reps.begin(), reps.end(), label) - reps.begin());
    };
    D.point[3] = Point{orbit_of(0), T};
    D.point[5] = Point{orbit_of(4), T};
    D.point[15] = Point{orbit_of(13), S};
    D.point[16] = Point{orbit_of(14), S};
    D.top = {{0, 1, 17}, {0, 17, 4}, {2, 3, 20}, {3, 20, 5}, {1, 2, 18}, {2, 21, 18}, {18, 21, 6},
             {4, 13, 17}, {13, 19, 17}, {13, 14, 19}, {14, 18, 19}, {14, 6, 18}, {17, 19, 18}, {1, 17, 18},
             {5, 15, 20}, {15, 22, 20}, {15, 16, 22}, {16, 21, 22}, {16, 6, 21}, {20, 22, 21}, {2, 20, 21}};
    D.boundary = {0, 1, 2, 3};
    return D;
}

std::vector<Mat22> Arrows(const Descriptor& D, uint32_t s0, uint32_t s1) {
    std::map<GKey, Mat22> gs;
    for (const Point& p0 : D.lifted(s0))
        for (const Point& p1 : D.lifted(s1)) {
            if (p0.orbit != p1.orbit) continue;
            const Mat22 inv1 = p1.g.inv_unimodular();
            for (const Mat22& u : Elements(D.orbit_stab.at(p0.orbit))) {
                const Mat22 g = p0.g * u * inv1;   // g * p1 = p0
                gs.emplace(key_of(g), g);
            }
        }
    std::vector<Mat22> out;
    for (const auto& kv : gs) out.push_back(kv.second);
    return out;
}

std::vector<PointKey> SimplexOrbitKey(const Descriptor& D, const std::vector<Point>& simplex) {
    std::vector<PointKey> best;
    bool first = true;
    for (const Point& x : simplex) {
        const Mat22 xinv = x.g.inv_unimodular();
        for (const Mat22& u : Elements(D.orbit_stab.at(x.orbit))) {
            const Mat22 h = u * xinv;   // h * x = u * iota(orbit) = iota(orbit)
            std::vector<PointKey> ks;
            for (const Point& y : simplex) ks.push_back(D.key(Point{y.orbit, h * y.g}));
            std::sort(ks.begin(), ks.end());
            if (first || ks < best) best = ks;
            first = false;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Predicates
// ---------------------------------------------------------------------------
bool AdaptedFace(const std::vector<CyclicGroup>& vs, bool subgroup_form) {
    if (subgroup_form) {
        for (const CyclicGroup& v : vs) {
            bool dominates = true;
            for (const CyclicGroup& w : vs) dominates = dominates && IsSubgroup(w, v);
            if (dominates) return true;
        }
        return false;
    }
    // mp_rules.py: the largest order is divisible by every order.
    uint32_t mx = 0;
    for (const CyclicGroup& v : vs) mx = std::max(mx, v.order);
    for (const CyclicGroup& v : vs)
        if (mx % v.order != 0) return false;
    return true;
}

bool Adapted(const Descriptor& D, uint32_t ell, bool subgroup_form) {
    for (uint32_t t = 0; t < D.top.size(); ++t) {
        const std::vector<Point> pts = D.lifted(t);
        const uint32_t k = static_cast<uint32_t>(pts.size());
        for (uint32_t mask = 1; mask < (1u << k); ++mask) {
            if (__builtin_popcount(mask) < 2) continue;
            std::vector<CyclicGroup> vs;
            for (uint32_t i = 0; i < k; ++i)
                if (mask & (1u << i)) vs.push_back(Sylow(D.stabiliser(pts[i]), ell));
            if (!AdaptedFace(vs, subgroup_form)) return false;
        }
    }
    return true;
}

namespace {

// The closed star of iota(orbit): every point of every translate of a top simplex containing it.
std::vector<Point> ClosedStar(const Descriptor& D, uint32_t orbit) {
    std::map<PointKey, Point> star;
    const Point v{orbit, Mat22::I()};
    star.emplace(D.key(v), v);
    for (uint32_t t = 0; t < D.top.size(); ++t) {
        const std::vector<Point> pts = D.lifted(t);
        for (const Point& a : pts) {
            if (a.orbit != orbit) continue;
            const Mat22 ainv = a.g.inv_unimodular();
            for (const Mat22& u : Elements(D.orbit_stab.at(orbit))) {
                const Mat22 h = u * ainv;   // h * a = v
                for (const Point& b : pts) {
                    const Point hb{b.orbit, h * b.g};
                    star.emplace(D.key(hb), hb);
                }
            }
        }
    }
    std::vector<Point> out;
    for (const auto& kv : star) out.push_back(kv.second);
    return out;
}

}  // namespace

GoodResult Good(const Descriptor& D) {
    GoodResult R;
    for (uint32_t r = 0; r < D.orbit_label.size(); ++r) {
        const std::vector<Point> star = ClosedStar(D, r);
        std::map<GKey, Mat22> T;   // gamma with gamma a = b, a, b in St(v)
        for (const Point& a : star)
            for (const Point& b : star) {
                if (a.orbit != b.orbit) continue;
                const Mat22 ainv = a.g.inv_unimodular();
                for (const Mat22& u : Elements(D.orbit_stab.at(a.orbit))) {
                    const Mat22 g = b.g * u * ainv;
                    T.emplace(key_of(g), g);
                }
            }
        // The finite-group criterion: close <T(v)>; more than 6 elements => infinite.
        std::map<GKey, Mat22> grp;
        grp.emplace(key_of(Mat22::I()), Mat22::I());
        std::vector<Mat22> frontier = {Mat22::I()};
        bool finite = true;
        while (!frontier.empty() && finite) {
            std::vector<Mat22> next;
            for (const Mat22& x : frontier)
                for (const auto& kv : T) {
                    const Mat22 y = x * kv.second;
                    if (grp.emplace(key_of(y), y).second) next.push_back(y);
                    if (grp.size() > 6) {
                        finite = false;
                        break;
                    }
                }
            frontier = std::move(next);
        }
        // The stronger form: T(v) inside the stabiliser of one point of St(v).
        bool strong = false;
        for (const Point& p : star) {
            const CyclicGroup Hp = D.stabiliser(p);
            bool all = true;
            for (const auto& kv : T) all = all && Contains(Hp, kv.second);
            if (all) {
                strong = true;
                break;
            }
        }
        if (!finite) {
            R.good = false;
            R.fails_at.push_back(D.orbit_label[r]);
        }
        if (!strong) {
            R.good_strong = false;
            R.fails_at_strong.push_back(D.orbit_label[r]);
        }
    }
    std::sort(R.fails_at.begin(), R.fails_at.end());
    std::sort(R.fails_at_strong.begin(), R.fails_at_strong.end());
    return R;
}

bool Collar(const Descriptor& D) {
    for (const auto& t : D.top) {
        bool bd = false, ell = false;
        for (uint32_t v : t) {
            bd = bd || D.boundary.count(v) == 1;
            ell = ell || D.elliptic_order(v) > 2;
        }
        if (bd && ell) return false;
    }
    return true;
}

SCResult SimplicialComplexProperty(const Descriptor& D) {
    SCResult R;
    const uint32_t F = static_cast<uint32_t>(D.top.size());
    std::vector<std::set<PointKey>> keys(F);
    for (uint32_t t = 0; t < F; ++t) {
        for (const Point& p : D.lifted(t)) keys[t].insert(D.key(p));
        if (keys[t].size() != D.top[t].size() && R.ok) {
            R.ok = false;
            R.first_failure = "top simplex " + std::to_string(t) + " has a repeated vertex";
        }
    }
    for (uint32_t a = 0; a < F; ++a)
        for (uint32_t b = 0; b < F; ++b)
            for (const Mat22& g : Arrows(D, a, b)) {
                ++R.pairs_checked;
                std::set<PointKey> img;
                for (const Point& p : D.lifted(b)) img.insert(D.key(Point{p.orbit, g * p.g}));
                uint32_t common = 0;
                for (const PointKey& k : img) common += keys[a].count(k);
                if (common == 0 && R.ok) {
                    R.ok = false;
                    R.first_failure = "an arrow whose translate does not meet the simplex";
                }
                if (common == img.size()) {
                    bool fixes = (a == b);
                    if (fixes)
                        for (const Point& p : D.lifted(a)) fixes = fixes && Contains(D.stabiliser(p), g);
                    if (!fixes && R.ok) {
                        std::ostringstream os;
                        os << "equal vertex sets without gamma in S(sigma, sigma'): top " << a << " vs " << b;
                        R.ok = false;
                        R.first_failure = os.str();
                    }
                }
            }
    return R;
}

bool InvariantOrder(const Descriptor& D) {
    for (const auto& t : D.top) {
        std::set<uint32_t> orbits;
        for (uint32_t v : t) orbits.insert(D.point.at(v).orbit);
        if (orbits.size() != t.size()) return false;
    }
    return true;
}

std::vector<uint32_t> OrbitOrdered(const Descriptor& D, const std::vector<uint32_t>& simplex) {
    std::vector<uint32_t> out = simplex;
    std::sort(out.begin(), out.end(), [&](uint32_t x, uint32_t y) { return D.point.at(x).orbit < D.point.at(y).orbit; });
    for (size_t i = 1; i < out.size(); ++i)
        ENGINE_CHECK_MSG(D.point.at(out[i - 1]).orbit != D.point.at(out[i]).orbit,
                         "orbifold_mp::OrbitOrdered: two vertices in one orbit (no Gamma-invariant vertex order)");
    return out;
}

QuotientCounts Quotient(const Descriptor& D) {
    QuotientCounts Q;
    std::set<uint32_t> orbits;
    std::set<std::vector<PointKey>> edges, tris;
    for (uint32_t t = 0; t < D.top.size(); ++t) {
        const std::vector<Point> pts = D.lifted(t);
        for (const Point& p : pts) orbits.insert(p.orbit);
        tris.insert(SimplexOrbitKey(D, pts));
        for (size_t i = 0; i < pts.size(); ++i)
            for (size_t j = i + 1; j < pts.size(); ++j) edges.insert(SimplexOrbitKey(D, {pts[i], pts[j]}));
    }
    ENGINE_CHECK_MSG(tris.size() == D.top.size(), "orbifold_mp::Quotient: two top simplices in one orbit (not a transversal)");
    Q.V = static_cast<uint32_t>(orbits.size());
    Q.E = static_cast<uint32_t>(edges.size());
    Q.F = static_cast<uint32_t>(tris.size());
    Q.chi = static_cast<int64_t>(Q.V) - static_cast<int64_t>(Q.E) + static_cast<int64_t>(Q.F);
    return Q;
}

namespace {

int Parity(const std::vector<uint32_t>& from, const std::vector<uint32_t>& to) {
    ENGINE_CHECK(from.size() == to.size());
    std::vector<size_t> pos;
    for (uint32_t x : to) {
        const auto it = std::find(from.begin(), from.end(), x);
        ENGINE_CHECK(it != from.end());
        pos.push_back(static_cast<size_t>(it - from.begin()));
    }
    int s = 1;
    for (size_t i = 0; i < pos.size(); ++i)
        for (size_t j = i + 1; j < pos.size(); ++j)
            if (pos[i] > pos[j]) s = -s;
    return s;
}

}  // namespace

uint64_t OrderedChainSignBridge(const integral_model::ModelData& d, bool* ok) {
    *ok = true;
    const Descriptor D = FromModelData(d, d.header.name);
    const integral_model::IsotropyModel M(d, 1);
    uint64_t checked = 0;
    for (uint32_t s = 1; s < 3; ++s)
        for (uint32_t r = 0; r < M.reps(s).size(); ++r) {
            const std::vector<uint32_t> sig = M.reps(s)[r];
            const std::vector<uint32_t> sig_ord = OrbitOrdered(D, sig);
            const int eps_sig = Parity(sig, sig_ord);
            for (const auto& [sign_im, ft] : M.faces(s, r)) {
                // The omitted vertex and its position in the orbit-ordered simplex.
                uint32_t omitted = UINT32_MAX;
                for (uint32_t v : sig)
                    if (std::find(ft.face.begin(), ft.face.end(), v) == ft.face.end()) omitted = v;
                ENGINE_CHECK(omitted != UINT32_MAX);
                const size_t j = static_cast<size_t>(std::find(sig_ord.begin(), sig_ord.end(), omitted) - sig_ord.begin());
                const int eps_face = Parity(ft.face, OrbitOrdered(D, ft.face));
                const int expect = eps_sig * ((j % 2 == 0) ? 1 : -1) * eps_face;
                ++checked;
                if (expect != sign_im) *ok = false;
            }
        }
    return checked;
}

}  // namespace orbifold_mp
