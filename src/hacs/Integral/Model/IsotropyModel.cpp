// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Line 2, L2-1a (hecke-extensions slot 9, WP-10) -- the isotropy-resolution
// model of [C, def:isomodel].  See IsotropyModel.h for the contract.
#include "IsotropyModel.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <utility>

#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "PeriodicColumn.h"

namespace integral_model {

using hecke::Mat22;

// ---------------------------------------------------------------------------
// Group elements and the group ring
// ---------------------------------------------------------------------------
GKey key_of(const Mat22& g) { return GKey{g.a, g.b, g.c, g.d}; }
Mat22 mat_of(const GKey& k) { return Mat22{k[0], k[1], k[2], k[3]}; }

GroupRingElt GroupRingElt::unit(const Mat22& g, int64_t n) {
    GroupRingElt r;
    r.add(key_of(g), n);
    return r;
}

void GroupRingElt::add(const GKey& g, int64_t n) {
    if (n == 0) return;
    auto it = terms.find(g);
    if (it == terms.end()) {
        terms.emplace(g, n);
        return;
    }
    it->second = hecke::checked_add<int64_t>(it->second, n);
    if (it->second == 0) terms.erase(it);
}

void GroupRingElt::add(const GroupRingElt& o, int64_t scale) {
    if (scale == 0) return;
    for (const auto& [g, n] : o.terms) add(g, hecke::checked_mul<int64_t>(n, scale));
}

GroupRingElt GroupRingElt::operator*(const GroupRingElt& o) const {
    GroupRingElt r;
    for (const auto& [g1, n1] : terms) {
        const Mat22 m1 = mat_of(g1);
        for (const auto& [g2, n2] : o.terms) {
            r.add(key_of(m1 * mat_of(g2)), hecke::checked_mul<int64_t>(n1, n2));
        }
    }
    return r;
}

int64_t GroupRingElt::max_abs_coefficient() const {
    int64_t m = 0;
    for (const auto& [g, n] : terms) m = std::max<int64_t>(m, static_cast<int64_t>(hecke::uabs64(n)));
    return m;
}

int64_t GroupRingElt::max_abs_entry() const {
    int64_t m = 0;
    for (const auto& [g, n] : terms) m = std::max(m, mat_of(g).max_abs());
    return m;
}

std::string GroupRingElt::to_string() const {
    if (terms.empty()) return "0";
    std::ostringstream os;
    bool first = true;
    for (const auto& [g, n] : terms) {
        if (!first) os << " + ";
        first = false;
        os << n << "*[" << g[0] << "," << g[1] << ";" << g[2] << "," << g[3] << "]";
    }
    return os.str();
}

namespace {

// A chain in one column G_{s, t}: one Z[Gamma]-coefficient per representative.
using ColumnChain = std::vector<GroupRingElt>;

bool column_is_zero(const ColumnChain& c) {
    for (const auto& e : c)
        if (!e.is_zero()) return false;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// The model descriptor: [C, def:model7], [C, lem:model7coincidences],
// [C, def:isomodel]
// ---------------------------------------------------------------------------
const std::vector<std::vector<Simplex>>& IsotropyModel::K_simplices() {
    // Stored order: increasing vertex index, lexicographic within a dimension
    // ([C, def:model7]).  HeckeIsotropyModelValidation checks this list against
    // CreateSL2ZFundamentalDomainTriangulation().
    static const std::vector<std::vector<Simplex>> K = {
        {{0}, {1}, {2}, {3}, {4}, {5}, {6}},
        {{0, 1}, {0, 4}, {1, 2}, {1, 4}, {1, 6}, {2, 3}, {2, 5}, {2, 6}, {3, 5}, {4, 6}, {5, 6}},
        {{0, 1, 4}, {1, 2, 6}, {1, 4, 6}, {2, 3, 5}, {2, 5, 6}},
    };
    return K;
}

// [C, def:model7] / [C, lem:model7coincidences] (2) / [C, def:isomodel] as DATA
// (slot-17 WP-7: formerly IsotropyModel::build_descriptor(), value for value).
ModelData SL2Z7VertexModelData() {
    const Mat22 I = Mat22::I();
    const Mat22 negI = I.neg();
    const Mat22 T = Mat22::T();
    const Mat22 S = Mat22::S();
    const Mat22 ST = Mat22::ST();

    ModelData d;
    d.header = hecke::SL2Z7VertexModel();
    // Sigma_s ([C, def:isomodel]): the simplices of K minus the four
    // non-representatives of [C, lem:model7coincidences] (2): v3 = T v0,
    // v5 = T v4, {v3,v5} = T {v0,v4}, {v5,v6} = S^{-1} {v4,v6}.
    d.nonreps.push_back({{3}, 0, T});
    d.nonreps.push_back({{5}, 3, T});             // Sigma_0 = {v0, v1, v2, v4, v6}: v4 at index 3
    d.nonreps.push_back({{3, 5}, 1, T});          // Sigma_1 index 1 = {v0, v4}
    d.nonreps.push_back({{5, 6}, 8, S.inv_unimodular()});   // Sigma_1 index 8 = {v4, v6}

    const auto& K = IsotropyModel::K_simplices();
    for (uint32_t s = 0; s < 3; ++s) {
        for (const auto& sim : K[s]) {
            bool nonrep = false;
            for (const auto& nr : d.nonreps)
                if (nr.face == sim) nonrep = true;
            if (!nonrep) d.reps[s].push_back(sim);
        }
    }
    ENGINE_CHECK_MSG(d.reps[0].size() == 5 && d.reps[1].size() == 9 && d.reps[2].size() == 5,
                     "IsotropyModel: |Sigma_s| must be 5, 9, 5 ([C, lem:model7coincidences] (2))");
    ENGINE_CHECK(d.reps[0][3] == Simplex({4}) && d.reps[1][1] == Simplex({0, 4}) && d.reps[1][8] == Simplex({4, 6}));

    // Stabiliser generators ([C, def:isomodel]): ST at v4, S at v6, -I elsewhere.
    for (uint32_t s = 0; s < 3; ++s) {
        d.gen[s].assign(d.reps[s].size(), negI);
        d.order[s].assign(d.reps[s].size(), 2);
        d.bdry[s].assign(d.reps[s].size(), false);
        for (uint32_t r = 0; r < d.reps[s].size(); ++r) {
            const Simplex& sim = d.reps[s][r];
            if (s == 0 && sim[0] == 4) { d.gen[s][r] = ST; d.order[s][r] = 6; }
            if (s == 0 && sim[0] == 6) { d.gen[s][r] = S;  d.order[s][r] = 4; }
            // Sigma^d_0 = {v0, v1, v2}, Sigma^d_1 = {v0v1, v1v2, v2v3}.
            bool all_on_horocycle = s <= 1;
            for (uint32_t v : sim) all_on_horocycle = all_on_horocycle && v <= 3;
            d.bdry[s][r] = all_on_horocycle;
        }
    }
    ENGINE_CHECK(std::count(d.bdry[0].begin(), d.bdry[0].end(), true) == 3);
    ENGINE_CHECK(std::count(d.bdry[1].begin(), d.bdry[1].end(), true) == 3);
    ENGINE_CHECK(std::count(d.bdry[2].begin(), d.bdry[2].end(), true) == 0);
    d.horocycle = {0, 1, 2, 3};
    return d;
}

std::vector<uint32_t> ModelData::simplex_counts() const {
    std::vector<uint32_t> n(3, 0);
    for (uint32_t s = 0; s < 3; ++s) n[s] = static_cast<uint32_t>(reps[s].size());
    for (const auto& nr : nonreps) {
        ENGINE_CHECK(!nr.face.empty() && nr.face.size() <= 3);
        ++n[nr.face.size() - 1];
    }
    return n;
}

namespace {

// The parity (+1 / -1) of the permutation taking `from` to `to` (same vertex set,
// no repeats; ENGINE_CHECKed).
int permutation_sign(const Simplex& from, const Simplex& to) {
    ENGINE_CHECK(from.size() == to.size());
    std::vector<uint32_t> pos(from.size());
    for (size_t i = 0; i < to.size(); ++i) {
        const auto it = std::find(from.begin(), from.end(), to[i]);
        ENGINE_CHECK_MSG(it != from.end(), "IsotropyModel: a face is not a permutation of its stored simplex");
        pos[i] = static_cast<uint32_t>(it - from.begin());
    }
    int sign = 1;
    for (size_t i = 0; i < pos.size(); ++i)
        for (size_t j = i + 1; j < pos.size(); ++j)
            if (pos[i] > pos[j]) sign = -sign;
    return sign;
}

Simplex sorted_of(Simplex s) {
    std::sort(s.begin(), s.end());
    return s;
}

}  // namespace

void IsotropyModel::load_descriptor(const ModelData& d) {
    const Mat22 I = Mat22::I();
    data_ = d;
    nonreps_ = d.nonreps;
    for (uint32_t s = 0; s < 3; ++s) {
        reps_[s] = d.reps[s];
        gen_[s] = d.gen[s];
        order_[s] = d.order[s];
        bdry_[s] = d.bdry[s];
        ENGINE_CHECK_MSG(!reps_[s].empty() && gen_[s].size() == reps_[s].size() && order_[s].size() == reps_[s].size() &&
                             bdry_[s].size() == reps_[s].size(),
                         "IsotropyModel: descriptor arrays of unequal length");
        for (const auto& sim : reps_[s]) ENGINE_CHECK_MSG(sim.size() == s + 1, "IsotropyModel: a representative of the wrong dimension");
    }
    // The header states the model (Hecke/HeckeModelDescriptor): dimension 2 and N_q of K'.
    ENGINE_CHECK_MSG(d.header.dimension == 2 && d.header.simplex_counts == d.simplex_counts(),
                     "IsotropyModel: the descriptor header's N_q disagree with the descriptor data");
    for (const auto& nr : nonreps_) {
        const uint32_t dim = static_cast<uint32_t>(nr.face.size()) - 1;
        ENGINE_CHECK_MSG(nr.rep < reps_[dim].size(), "IsotropyModel: a transport to a missing representative");
    }
    // The order is exact: g^m = I and g^j != I for 0 < j < m; -I lies in every stabiliser ([C, H4]).
    for (uint32_t s = 0; s < 3; ++s)
        for (uint32_t r = 0; r < reps_[s].size(); ++r) {
            Mat22 p = I;
            bool has_negI = false;
            for (uint32_t j = 1; j <= order_[s][r]; ++j) {
                p = p * gen_[s][r];
                has_negI = has_negI || p == I.neg();
                ENGINE_CHECK_MSG((j == order_[s][r]) == (p == I), "IsotropyModel: stabiliser generator has the wrong order");
            }
            ENGINE_CHECK_MSG(has_negI, "IsotropyModel: -I must lie in every stabiliser (the centre is not trivialised)");
        }

    // The face transports of every representative: face_i = sigma minus its i-th
    // vertex, sign (-1)^i times the parity of the face's stored orientation.  A face
    // is located by its vertex SET among the representatives and the transports.
    std::map<Simplex, std::pair<Simplex, FaceTransport>> by_set;   // sorted set -> (stored list, transport)
    for (uint32_t s = 0; s < 3; ++s)
        for (uint32_t r = 0; r < reps_[s].size(); ++r) {
            const bool fresh = by_set.emplace(sorted_of(reps_[s][r]), std::make_pair(reps_[s][r], FaceTransport{reps_[s][r], r, I})).second;
            ENGINE_CHECK_MSG(fresh, "IsotropyModel: a simplex listed twice");
        }
    for (const auto& nr : nonreps_) {
        const bool fresh = by_set.emplace(sorted_of(nr.face), std::make_pair(nr.face, nr)).second;
        ENGINE_CHECK_MSG(fresh, "IsotropyModel: a simplex listed twice");
    }
    for (uint32_t s = 0; s < 3; ++s) {
        faces_[s].assign(reps_[s].size(), {});
        if (s == 0) continue;
        for (uint32_t r = 0; r < reps_[s].size(); ++r) {
            const Simplex& sim = reps_[s][r];
            for (uint32_t i = 0; i < sim.size(); ++i) {
                Simplex f;
                for (uint32_t k = 0; k < sim.size(); ++k)
                    if (k != i) f.push_back(sim[k]);
                const auto it = by_set.find(sorted_of(f));
                ENGINE_CHECK_MSG(it != by_set.end(), "IsotropyModel: a face is not a simplex of K");
                const int sign = ((i % 2 == 0) ? 1 : -1) * permutation_sign(it->second.first, f);
                faces_[s][r].push_back({sign, it->second.second});
            }
        }
    }
}

ModelData SubdivideEdgeOrbit(const ModelData& d, const Simplex& edge) {
    const Mat22 I = Mat22::I();
    struct KS {
        Simplex s;      // the simplex of K' (stored orientation)
        Simplex rep;    // its representative (stored orientation); s = g * rep positionally
        Mat22 g;
    };
    struct Attr {
        Mat22 gen;
        uint32_t order = 1;
        bool bdry = false;
    };
    std::array<std::vector<KS>, 3> k;
    std::array<std::map<Simplex, Attr>, 3> attr;   // representative -> its stabiliser data
    for (uint32_t s = 0; s < 3; ++s)
        for (uint32_t r = 0; r < d.reps[s].size(); ++r) {
            k[s].push_back({d.reps[s][r], d.reps[s][r], I});
            attr[s][d.reps[s][r]] = Attr{d.gen[s][r], d.order[s][r], d.bdry[s][r]};
        }
    for (const auto& nr : d.nonreps) {
        const uint32_t dim = static_cast<uint32_t>(nr.face.size()) - 1;
        ENGINE_CHECK_MSG(dim < 2, "SubdivideEdgeOrbit: non-representative triangles are not supported");
        k[dim].push_back({nr.face, d.reps[dim].at(nr.rep), nr.g});
    }
    ENGINE_CHECK_MSG(edge.size() == 2 && attr[1].count(edge) == 1, "SubdivideEdgeOrbit: not an edge representative");
    const Attr ea = attr[1][edge];
    // slot-17 WP-17 (WP-7 audit MINOR-2): the midpoint and the two halves inherit the edge's stabiliser, which is right
    // only when that stabiliser fixes the edge POINTWISE, i.e. is the centre {+-I} (-I lies in every stabiliser, [C, H4]).
    // An edge flipped by its stabiliser (order 4, say) would need a different midpoint orbit; refuse it.
    ENGINE_CHECK_MSG(ea.order == 2 && ea.gen == I.neg(),
                     "SubdivideEdgeOrbit: the edge's stabiliser is not the centre {+-I} (only pointwise-central edge "
                     "stabilisers are supported)");
    uint32_t next = 0;
    for (const auto& v : k[0]) next = std::max(next, v.s[0] + 1);

    std::vector<uint32_t> horocycle = d.horocycle;
    // The new orbits: the midpoint m, the two halves e' = [e0, m], e'' = [m, e1].
    const uint32_t m = next++;
    const Simplex rm = {m}, re1 = {edge[0], m}, re2 = {m, edge[1]};
    attr[0][rm] = ea;
    attr[1][re1] = ea;
    attr[1][re2] = ea;
    attr[1].erase(edge);
    std::map<Simplex, std::pair<Simplex, uint32_t>> midpoint;   // sorted K'-edge -> (stored edge, midpoint label)
    // (slot-18 WP-7) the subdivision record: every split K' simplex (stored) -> its two pieces (as chains,
    // sum of the pieces = the simplex), and every new midpoint label -> the first vertex of its stored edge.
    std::map<Simplex, std::vector<std::pair<int, Simplex>>> pieces;
    std::map<uint32_t, uint32_t> new_collapse = d.collapse;
    auto root_of = [&](uint32_t v) {
        const auto it = d.collapse.find(v);
        return it == d.collapse.end() ? v : it->second;
    };
    std::vector<KS> edges;
    for (const KS& E : k[1]) {
        if (E.rep != edge) {
            edges.push_back(E);
            continue;
        }
        const bool is_rep = (E.g == I) && (E.s == edge);
        const uint32_t mid = is_rep ? m : next++;
        k[0].push_back({{mid}, rm, E.g});
        midpoint[sorted_of(E.s)] = {E.s, mid};
        edges.push_back({{E.s[0], mid}, re1, E.g});
        edges.push_back({{mid, E.s[1]}, re2, E.g});
        pieces[E.s] = {{1, Simplex{E.s[0], mid}}, {1, Simplex{mid, E.s[1]}}};
        new_collapse[mid] = root_of(E.s[0]);
        if (ea.bdry) horocycle.push_back(mid);
    }
    // Triangles: split every one with a face in the orbit, through the midpoint and the opposite vertex.
    std::vector<KS> work = k[2], tris;
    while (!work.empty()) {
        const KS tau = work.back();
        work.pop_back();
        ENGINE_CHECK(tau.g == I && tau.s == tau.rep);
        int split_at = -1;
        for (uint32_t i = 0; i < 3 && split_at < 0; ++i) {
            Simplex f;
            for (uint32_t j = 0; j < 3; ++j)
                if (j != i) f.push_back(tau.s[j]);
            if (midpoint.count(sorted_of(f))) split_at = static_cast<int>(i);
        }
        if (split_at < 0) {
            tris.push_back(tau);
            continue;
        }
        const uint32_t z = tau.s[static_cast<uint32_t>(split_at)];
        Simplex f;
        for (uint32_t j = 0; j < 3; ++j)
            if (j != static_cast<uint32_t>(split_at)) f.push_back(tau.s[j]);
        const auto& [stored, mid] = midpoint.at(sorted_of(f));
        const uint32_t x = stored[0], y = stored[1];
        // tau = eps [x, y, z]; [x, y, z] = [x, mid, z] + [mid, y, z] as chains.
        const int eps = permutation_sign(tau.s, Simplex{x, y, z});
        const Simplex t1 = (eps > 0) ? Simplex{x, mid, z} : Simplex{mid, x, z};
        const Simplex t2 = (eps > 0) ? Simplex{mid, y, z} : Simplex{y, mid, z};
        const Simplex fe = {mid, z};
        const Attr ta = attr[2].at(tau.rep);
        // (same restriction) the new interior edge and both halves inherit the triangle's stabiliser: pointwise only if central.
        ENGINE_CHECK_MSG(ta.order == 2 && ta.gen == I.neg(),
                         "SubdivideEdgeOrbit: a split triangle's stabiliser is not the centre {+-I}");
        attr[2].erase(tau.rep);
        attr[2][t1] = ta;
        attr[2][t2] = ta;
        attr[1][fe] = Attr{ta.gen, ta.order, false};
        pieces[tau.s] = {{1, t1}, {1, t2}};   // tau = t1 + t2 as chains (the comment above)
        edges.push_back({fe, fe, I});
        work.push_back({t1, t1, I});
        work.push_back({t2, t2, I});
    }
    k[1] = std::move(edges);
    k[2] = std::move(tris);

    // Re-index: representatives in K' order (old ones first, in their order), transports to them.
    ModelData out;
    out.horocycle = horocycle;
    std::array<std::map<Simplex, uint32_t>, 3> index;
    for (uint32_t s = 0; s < 3; ++s)
        for (const KS& x : k[s])
            if (x.g == I && x.s == x.rep) {
                const Attr& a = attr[s].at(x.rep);
                index[s][x.rep] = static_cast<uint32_t>(out.reps[s].size());
                out.reps[s].push_back(x.rep);
                out.gen[s].push_back(a.gen);
                out.order[s].push_back(a.order);
                out.bdry[s].push_back(a.bdry);
            }
    for (uint32_t s = 0; s < 3; ++s)
        for (const KS& x : k[s])
            if (!(x.g == I && x.s == x.rep)) out.nonreps.push_back({x.s, index[s].at(x.rep), x.g});
    // (slot-18 WP-7) the cumulative record relative to the ROOT descriptor (d's record composed with this step).
    std::function<std::vector<std::pair<int, Simplex>>(const Simplex&)> expand = [&](const Simplex& x) {
        const auto it = pieces.find(x);
        if (it == pieces.end()) return std::vector<std::pair<int, Simplex>>{{1, x}};
        std::vector<std::pair<int, Simplex>> outv;
        for (const auto& [sg, child] : it->second)
            for (const auto& [sg2, leaf] : expand(child)) outv.push_back({sg * sg2, leaf});
        return outv;
    };
    out.collapse = new_collapse;
    for (const auto& [root, chain] : d.refines) {
        std::vector<std::pair<int, Simplex>> v;
        for (const auto& [sg, x] : chain)
            for (const auto& [sg2, leaf] : expand(x)) v.push_back({sg * sg2, leaf});
        out.refines[root] = std::move(v);
    }
    for (const auto& [x, pc] : pieces) {
        (void)pc;
        // x is a simplex of d's K'.  It is a ROOT simplex (absent from d.refines: it maps to itself) iff no vertex
        // of it was created by an earlier subdivision; every other simplex of d's K' contains a midpoint and was
        // handled above (a piece of a root simplex) or is an interior edge of a split triangle (not in the image of
        // the subdivision chain map; nothing to record).
        bool root = true;
        for (uint32_t v : x) root = root && d.collapse.count(v) == 0;
        if (root) {
            ENGINE_CHECK_MSG(d.refines.count(x) == 0, "SubdivideEdgeOrbit: a recorded root simplex is still a simplex of K'");
            out.refines[x] = expand(x);
        }
    }
    out.header = d.header;
    out.header.name = d.header.name + "+sub(" + std::to_string(edge[0]) + "," + std::to_string(edge[1]) + ")";
    out.header.description = "a Gamma-equivariant subdivision of " + d.header.name + " (synthetic, slot-17 WP-7)";
    out.header.simplex_counts = out.simplex_counts();
    return out;
}


// ---------------------------------------------------------------------------
// The construction ([C, prop:isomodel] (1)): d^(1) and d^(2) by the Z-linear
// contracting homotopies of the periodic resolutions, boundary summands first.
// ---------------------------------------------------------------------------
namespace {

struct ColumnData {
    // Per representative of Sigma_s: g, its powers g^0..g^{m-1}, and m.
    std::vector<std::vector<Mat22>> pow;
    std::vector<uint32_t> m;
};

// decompose / dF_coeff: shared with StrictLift.cpp and PeriodicityClass.cpp
// since slot-14 WP-14b (review finding slot-10 m-2) -- see PeriodicColumn.h
// (found by ordinary enclosing-namespace lookup, this file being inside
// `namespace integral_model` throughout; this TU also DEFINES key_of / mat_of,
// which the header's inline bodies call).

// Apply d^F : G_{s, t} -> G_{s, t-1} (right multiplication of each coefficient).
ColumnChain apply_dF(const ColumnData& col, const ColumnChain& z, uint32_t t) {
    ColumnChain out(z.size());
    for (size_t r = 0; r < z.size(); ++r) out[r] = z[r] * dF_coeff(col.pow[r], t);
    return out;
}

// The augmentation G_{s,0} -> (+) Z[Gamma / Gamma_rho]: zero iff, per rep, the
// multiplicities of every coset sum to zero.
bool augmentation_is_zero(const ColumnData& col, const ColumnChain& z) {
    for (size_t r = 0; r < z.size(); ++r) {
        std::map<GKey, int64_t> cosets;
        for (const auto& [g, n] : z[r].terms) {
            const GKey c = decompose(g, col.pow[r]).first;
            cosets[c] = hecke::checked_add<int64_t>(cosets[c], n);
        }
        for (const auto& [c, n] : cosets)
            if (n != 0) return false;
    }
    return true;
}

// The Z-linear contracting homotopy h_t : G_{s,t} -> G_{s,t+1} of the periodic
// resolution, coset by coset: on c * g^j,
//   t even:  h(c g^j) = sum_{i<j} c g^i;
//   t odd:   h(c g^j) = c   if j = m - 1, else 0.
// It satisfies d^F h + h d^F = id (with h_{-1}(c) = c on the augmentation).
// (slot-14 WP-14b, m-2) Per representative this is exactly column_homotopy of
// PeriodicColumn.h; folded to call it (col.m[r] == col.pow[r].size() always,
// by construction in BuildResolution() below, so column_homotopy's own
// powg.size() reading of m is the same value col.m[r] gave here).
ColumnChain homotopy(const ColumnData& col, const ColumnChain& z, uint32_t t) {
    ColumnChain out(z.size());
    for (size_t r = 0; r < z.size(); ++r) out[r] = column_homotopy(z[r], col.pow[r], t);
    return out;
}

// Certifies z in G_{s, t} is a cycle of the augmented column (d^F z = 0 for
// t >= 1; augmentation zero for t = 0) -- the input condition of the homotopy.
void check_cycle(const ColumnData& col, const ColumnChain& z, uint32_t t, const char* what) {
    const bool ok = (t == 0) ? augmentation_is_zero(col, z) : column_is_zero(apply_dF(col, z, t));
    if (!ok) {
        const std::string msg = std::string("IsotropyModel: homotopy input is not a cycle (") + what + ")";
        ENGINE_CHECK_MSG(ok, msg.c_str());
    }
}

}  // namespace

uint32_t IsotropyModel::cell_index(uint32_t s, uint32_t rep, uint32_t t) const {
    // The offset depends on s alone (cells of degree s + t are ordered by s,
    // then representative); t is checked against the built range.
    ENGINE_CHECK(s <= 2 && s + t < cells_.size() && rep < reps_[s].size());
    uint32_t base = 0;
    for (uint32_t s2 = 0; s2 < s; ++s2) base += static_cast<uint32_t>(reps_[s2].size());
    return base + rep;
}

uint32_t IsotropyModel::boundary_rank(uint32_t q) const {
    uint32_t w = 0;
    for (const auto& c : cells_.at(q)) w += c.boundary ? 1 : 0;
    return w;
}

void IsotropyModel::build_differentials() {
    const uint32_t Q = q_top();
    // Cells: degree q = s + t, ordered by s ascending, then representative.
    cells_.assign(Q + 1, {});
    for (uint32_t q = 0; q <= Q; ++q)
        for (uint32_t s = 0; s <= std::min<uint32_t>(q, 2); ++s)
            for (uint32_t r = 0; r < reps_[s].size(); ++r)
                cells_[q].push_back(Cell{s, r, q - s, bdry_[s][r]});

    std::array<ColumnData, 3> col;
    for (uint32_t s = 0; s < 3; ++s) {
        col[s].m = order_[s];
        col[s].pow.resize(reps_[s].size());
        for (uint32_t r = 0; r < reps_[s].size(); ++r) {
            Mat22 p = Mat22::I();
            for (uint32_t i = 0; i < order_[s][r]; ++i) {
                col[s].pow[r].push_back(p);
                p = p * gen_[s][r];
            }
        }
    }

    // D1[s][t][rep] in G_{s-1, t};  D2[t][rep] in G_{0, t+1}.
    std::array<std::vector<std::vector<ColumnChain>>, 3> D1;
    std::vector<std::vector<ColumnChain>> D2;
    for (uint32_t s = 1; s <= 2; ++s) {
        if (Q < s) continue;
        const uint32_t tmax = Q - s;
        D1[s].assign(tmax + 1, std::vector<ColumnChain>(reps_[s].size()));
        // Boundary summands first ([C, prop:isomodel] (1)): the order of the
        // representatives does not change any value here (each lift uses only
        // its own summand's data), but boundary edges are processed first and
        // their lifts are checked to stay in boundary vertex summands.
        std::vector<uint32_t> order;
        for (uint32_t r = 0; r < reps_[s].size(); ++r) if (bdry_[s][r]) order.push_back(r);
        for (uint32_t r = 0; r < reps_[s].size(); ++r) if (!bdry_[s][r]) order.push_back(r);
        for (uint32_t t = 0; t <= tmax; ++t) {
            for (uint32_t r : order) {
                ColumnChain& out = D1[s][t][r];
                if (t == 0) {
                    out.assign(reps_[s - 1].size(), {});
                    for (const auto& [sign, ft] : faces_[s][r]) out[ft.rep].add(key_of(ft.g), sign);
                } else {
                    // z = x_{sigma,t} * d1(e_{sigma,t-1}) = d1(d^F e_{sigma,t}), a cycle.
                    const GroupRingElt x = dF_coeff(col[s].pow[r], t);
                    ColumnChain z(reps_[s - 1].size());
                    for (size_t k = 0; k < z.size(); ++k) z[k] = x * D1[s][t - 1][r][k];
                    check_cycle(col[s - 1], z, t - 1, "d^(1)");
                    out = homotopy(col[s - 1], z, t - 1);
                }
                if (bdry_[s][r]) {
                    for (size_t k = 0; k < out.size(); ++k)
                        ENGINE_CHECK_MSG(out[k].is_zero() || bdry_[s - 1][k],
                                         "IsotropyModel: d^(1) of a boundary summand left the boundary");
                }
            }
        }
    }
    if (Q >= 2) {
        const uint32_t tmax = Q - 2;
        D2.assign(tmax + 1, std::vector<ColumnChain>(reps_[2].size()));
        for (uint32_t t = 0; t <= tmax; ++t) {
            for (uint32_t r = 0; r < reps_[2].size(); ++r) {
                // w = -d1 d1 e_{sigma,t} - x_{sigma,t} d2(e_{sigma,t-1}), a cycle in G_{0,t}.
                ColumnChain w(reps_[0].size());
                const ColumnChain& c1 = D1[2][t][r];          // in G_{1,t}
                for (size_t e = 0; e < c1.size(); ++e) {
                    if (c1[e].is_zero()) continue;
                    const ColumnChain& c0 = D1[1][t][e];      // d1(e_{edge,t}) in G_{0,t}
                    for (size_t v = 0; v < c0.size(); ++v) w[v].add(c1[e] * c0[v], -1);
                }
                if (t >= 1) {
                    const GroupRingElt x = dF_coeff(col[2].pow[r], t);
                    for (size_t v = 0; v < w.size(); ++v) w[v].add(x * D2[t - 1][r][v], -1);
                }
                check_cycle(col[0], w, t, "d^(2)");
                D2[t][r] = homotopy(col[0], w, t);
            }
        }
    }

    // Assemble d_q : Iso_q -> Iso_{q-1}.
    diff_.assign(Q + 1, {});
    for (uint32_t q = 1; q <= Q; ++q) {
        auto& dq = diff_[q];
        for (uint32_t ci = 0; ci < cells_[q].size(); ++ci) {
            const Cell& c = cells_[q][ci];
            std::vector<DiffEntry> col_entries;
            if (c.t >= 1) {
                GroupRingElt x = dF_coeff(col[c.s].pow[c.rep], c.t);
                if (c.s % 2 == 1) { GroupRingElt nx; nx.add(x, -1); x = nx; }
                col_entries.push_back({cell_index(c.s, c.rep, c.t - 1), ci, x});
            }
            if (c.s >= 1) {
                const ColumnChain& z = D1[c.s][c.t][c.rep];
                for (uint32_t k = 0; k < z.size(); ++k)
                    if (!z[k].is_zero()) col_entries.push_back({cell_index(c.s - 1, k, c.t), ci, z[k]});
            }
            if (c.s == 2) {
                const ColumnChain& z = D2[c.t][c.rep];
                for (uint32_t k = 0; k < z.size(); ++k)
                    if (!z[k].is_zero()) col_entries.push_back({cell_index(0, k, c.t + 1), ci, z[k]});
            }
            std::sort(col_entries.begin(), col_entries.end(),
                      [](const DiffEntry& a, const DiffEntry& b) { return a.row < b.row; });
            for (auto& e : col_entries) dq.push_back(std::move(e));
        }
    }
}

IsotropyModel::IsotropyModel(uint32_t q_max) : IsotropyModel(SL2Z7VertexModelData(), q_max) {}

IsotropyModel::IsotropyModel(const ModelData& data, uint32_t q_max) : q_max_(q_max) {
    ENGINE_CHECK_MSG(q_max >= 1 && q_max <= 64, "IsotropyModel: q_max out of range [1, 64]");
    load_descriptor(data);
    // The shipped model is recognised by its header AND its data, never by a count alone.
    const ModelData ref = SL2Z7VertexModelData();
    shipped_ = data.header.name == ref.header.name && data.reps == ref.reps && data.horocycle == ref.horocycle &&
               data.nonreps.size() == ref.nonreps.size();
    build_differentials();
}

std::vector<GKey> IsotropyModel::group_elements() const {
    std::map<GKey, bool> seen;
    for (const auto& dq : diff_)
        for (const auto& e : dq)
            for (const auto& [g, n] : e.coeff.terms) seen[g] = true;
    std::vector<GKey> out;
    for (const auto& [g, b] : seen) out.push_back(g);
    return out;
}

// ---------------------------------------------------------------------------
// Certificates ([C, def:isomodel] IM1, IM2), exact over Z[Gamma] / Z
// ---------------------------------------------------------------------------
Certificates IsotropyModel::certify() const {
    Certificates cert;
    cert.d_squared_zero = true;
    cert.boundary_subcomplex = true;
    cert.eps_chain_map = true;
    cert.eps_boundary = true;
    auto fail = [&](const std::string& why) {
        if (cert.first_failure.empty()) cert.first_failure = why;
    };

    for (uint32_t q = 1; q <= q_top(); ++q)
        for (const auto& e : diff_[q]) {
            cert.max_abs_coefficient = std::max(cert.max_abs_coefficient, e.coeff.max_abs_coefficient());
            cert.max_abs_entry = std::max(cert.max_abs_entry, e.coeff.max_abs_entry());
        }
    cert.distinct_group_elements = group_elements().size();

    // Per-degree column index of d_q.
    auto by_col = [&](uint32_t q) {
        std::vector<std::vector<const DiffEntry*>> idx(cells_[q].size());
        for (const auto& e : diff_[q]) idx[e.col].push_back(&e);
        return idx;
    };

    // (IM1) d o d = 0.
    for (uint32_t q = 2; q <= q_top(); ++q) {
        const auto cq = by_col(q);
        const auto cq1 = by_col(q - 1);
        for (uint32_t b = 0; b < cells_[q].size(); ++b) {
            std::map<uint32_t, GroupRingElt> acc;
            for (const DiffEntry* e1 : cq[b])
                for (const DiffEntry* e2 : cq1[e1->row]) {
                    acc[e2->row].add(e1->coeff * e2->coeff);
                    ++cert.d_squared_terms_checked;
                }
            for (const auto& [row, v] : acc)
                if (!v.is_zero()) {
                    cert.d_squared_zero = false;
                    fail("d_" + std::to_string(q - 1) + " d_" + std::to_string(q) + " != 0 at cell " +
                         std::to_string(b) + " -> " + std::to_string(row) + ": " + v.to_string());
                }
        }
    }

    // (IM2) R_d is a subcomplex.
    for (uint32_t q = 1; q <= q_top(); ++q)
        for (const auto& e : diff_[q])
            if (cells_[q][e.col].boundary && !cells_[q - 1][e.row].boundary) {
                cert.boundary_subcomplex = false;
                fail("d of boundary cell " + std::to_string(e.col) + " in degree " + std::to_string(q) +
                     " has a non-boundary component");
            }

    // (IM1) eps is a chain map, eps d = d eps, on C_.(Xbulk) written as
    // (+)_{rho in Sigma_s} Z[Gamma / Gamma_rho] (canonical coset representatives).
    std::array<std::vector<std::vector<Mat22>>, 3> pw;
    for (uint32_t s = 0; s < 3; ++s) {
        pw[s].resize(reps_[s].size());
        for (uint32_t r = 0; r < reps_[s].size(); ++r) {
            Mat22 p = Mat22::I();
            for (uint32_t i = 0; i < order_[s][r]; ++i) { pw[s][r].push_back(p); p = p * gen_[s][r]; }
        }
    }
    using XChain = std::map<std::pair<uint32_t, GKey>, int64_t>;   // (rep, coset) -> multiplicity
    auto xadd = [](XChain& x, uint32_t rep, const GKey& coset, int64_t n) {
        auto& v = x[{rep, coset}];
        v = hecke::checked_add<int64_t>(v, n);
        if (v == 0) x.erase({rep, coset});
    };
    for (uint32_t q = 1; q <= q_top(); ++q) {
        const auto cq = by_col(q);
        for (uint32_t b = 0; b < cells_[q].size(); ++b) {
            const Cell& c = cells_[q][b];
            XChain lhs, rhs;
            for (const DiffEntry* e : cq[b]) {
                const Cell& r = cells_[q - 1][e->row];
                if (r.t != 0) continue;
                for (const auto& [g, n] : e->coeff.terms)
                    xadd(lhs, r.rep, decompose(g, pw[r.s][r.rep]).first, n);
            }
            if (c.t == 0 && c.s >= 1) {
                for (const auto& [sign, ft] : faces_[c.s][c.rep])
                    xadd(rhs, ft.rep, decompose(key_of(ft.g), pw[c.s - 1][ft.rep]).first, sign);
            }
            ++cert.eps_cells_checked;
            if (lhs != rhs) {
                cert.eps_chain_map = false;
                fail("eps d != d eps at degree " + std::to_string(q) + " cell " + std::to_string(b));
            }
        }
    }
    // eps(R_d) lies in C_.(Xbdry): a boundary representative lies on the horocycle.
    for (uint32_t s = 0; s < 3; ++s)
        for (uint32_t r = 0; r < reps_[s].size(); ++r)
            if (bdry_[s][r])
                for (uint32_t v : reps_[s][r])
                    if (std::find(data_.horocycle.begin(), data_.horocycle.end(), v) == data_.horocycle.end()) {
                        cert.eps_boundary = false;
                        fail("boundary representative off the horocycle");
                    }
    return cert;
}

// ---------------------------------------------------------------------------
// rho_n over a ModCtx ([C, def:rhon])
// ---------------------------------------------------------------------------
template<class Elt>
engine::DenseBuf<Elt> rho_n_cubic(const engine::ModCtx<Elt>& ctx, const Mat22& g, uint32_t n) {
    const Elt p = ctx.p;
    auto red = [&](int64_t v) -> Elt {
        const int64_t pm = static_cast<int64_t>(p);   // p < 2^63 for every context used here
        int64_t r = (pm > 0) ? v % pm : v;
        if (r < 0) r += pm;
        return static_cast<Elt>(r);
    };
    ENGINE_CHECK_MSG(static_cast<uint64_t>(p) < (uint64_t(1) << 62), "rho_n: modulus too large for the reduction");
    const Elt a = red(g.a), b = red(g.b), c = red(g.c), d = red(g.d);
    const uint32_t D = n + 1;
    std::vector<std::vector<Elt>> binom(D, std::vector<Elt>(D, 0));
    for (uint32_t i = 0; i < D; ++i) {
        binom[i][0] = ctx.canonicalise(static_cast<Elt>(1 % p));
        for (uint32_t k = 1; k <= i; ++k)
            binom[i][k] = ctx.add(binom[i - 1][k - 1], (k <= i - 1) ? binom[i - 1][k] : Elt(0));
    }
    auto powers = [&](Elt x) {
        std::vector<Elt> pv(D, 0);
        pv[0] = static_cast<Elt>(1 % p);
        for (uint32_t i = 1; i < D; ++i) pv[i] = ctx.mul(pv[i - 1], x);
        return pv;
    };
    const auto pa = powers(a), pb = powers(b), pc = powers(c), pd = powers(d);
    engine::DenseBuf<Elt> M(D, D);
    for (uint32_t j = 0; j <= n; ++j) {
        const uint32_t nj = n - j;
        for (uint32_t t = 0; t <= n; ++t) {
            Elt acc = 0;
            const uint32_t s_lo = (t > j) ? t - j : 0;
            const uint32_t s_hi = std::min(nj, t);
            for (uint32_t s = s_lo; s <= s_hi; ++s) {
                const uint32_t u = t - s;   // 0 <= u <= j
                const Elt left = ctx.mul(ctx.mul(binom[nj][s], pa[nj - s]), pc[s]);
                const Elt right = ctx.mul(ctx.mul(binom[j][u], pb[j - u]), pd[u]);
                acc = ctx.add(acc, ctx.mul(left, right));
            }
            M.at(t, j) = acc;
        }
    }
    return M;
}

// ---------------------------------------------------------------------------
// (slot-23 F3L) rho_n by the column recurrence, O(d_V^2)
// ---------------------------------------------------------------------------
// f_j = (a x + c y)^{n-j} (b x + d y)^j satisfies (a x + c y) f_{j+1} = (b x + d y) f_j.
// Reading the coefficient of x^{n+1-t} y^t:
//     a f_{j+1}[t] + c f_{j+1}[t-1] = b f_j[t] + d f_j[t-1]      (f[-1] = f[n+1] = 0).
// If a is a unit of Z/p, the equations t = 0..n determine f_{j+1} ascending; if c is a unit,
// the equations t = n+1..1 determine it descending.  Either system is triangular with unit
// diagonal, so its unique solution IS the column of rho_n -- the same residues as the closed
// double sum (rho_n_cubic), not an approximation.  If neither a nor c is a unit (over a field:
// a = c = 0, a singular g) the cubic sum is used.  Unit-ness is gcd(x, p) == 1, exact for every
// modulus.
namespace {

// (S24 CL-5) L23_F3L on EnvLever: "" is invalid; an invalid value warns once and stays on (source=env).
engine::lever::EnvLever g_rho_lever("L23_F3L", engine::lever::Policy{true, engine::lever::Empty::Invalid, engine::lever::Match::Exact01, engine::lever::Cache::Once, "WARNING: L23_F3L='%s' is neither 0 nor 1; the F3L route stays on\n", "default", "env", "env"});
std::atomic<uint64_t> g_rho_rec{0}, g_rho_cubic{0}, g_rho_fallback{0};

template<class Elt>
bool is_unit_mod(Elt x, Elt p) {
    uint64_t a = static_cast<uint64_t>(x), b = static_cast<uint64_t>(p);
    while (b != 0) { const uint64_t t = a % b; a = b; b = t; }
    return a == 1;
}

}  // namespace

const engine::lever::EnvLever& F3lRouteLever() { return g_rho_lever; }
bool F3lRouteEnabled() { return g_rho_lever.enabled(); }
std::string F3lRouteSource() { return g_rho_lever.source(); }
RhoNTotals rho_n_totals() {
    RhoNTotals t;
    t.recurrence = g_rho_rec.load(std::memory_order_relaxed);
    t.cubic = g_rho_cubic.load(std::memory_order_relaxed);
    t.fallback = g_rho_fallback.load(std::memory_order_relaxed);
    return t;
}

template<class Elt>
bool rho_n_recurrence(const engine::ModCtx<Elt>& ctx, const Mat22& g, uint32_t n, engine::DenseBuf<Elt>& M) {
    const Elt p = ctx.p;
    ENGINE_CHECK_MSG(static_cast<uint64_t>(p) < (uint64_t(1) << 62), "rho_n: modulus too large for the reduction");
    auto red = [&](int64_t v) -> Elt {
        const int64_t pm = static_cast<int64_t>(p);
        int64_t r = v % pm;
        if (r < 0) r += pm;
        return static_cast<Elt>(r);
    };
    const Elt a = red(g.a), b = red(g.b), c = red(g.c), d = red(g.d);
    const bool up = is_unit_mod<Elt>(a, p);
    if (!up && !is_unit_mod<Elt>(c, p)) return false;
    const uint32_t D = n + 1;
    M = engine::DenseBuf<Elt>(D, D);
    const Elt one = static_cast<Elt>(1 % p);
    // f_0 = (a x + c y)^n: binom(n, t) a^{n-t} c^t, the binomial row by Pascal additions.
    std::vector<Elt> f(D + 1, 0), h(D + 1, 0);   // index t in [0, n]; slot D is the zero f[n+1]
    {
        std::vector<Elt> row(D, 0);
        row[0] = one;
        for (uint32_t i = 1; i <= n; ++i)
            for (uint32_t k = i; k >= 1; --k) row[k] = ctx.add(row[k], row[k - 1]);
        std::vector<Elt> pa(D, 0), pc(D, 0);
        pa[0] = pc[0] = one;
        for (uint32_t i = 1; i < D; ++i) {
            pa[i] = ctx.mul(pa[i - 1], a);
            pc[i] = ctx.mul(pc[i - 1], c);
        }
        for (uint32_t t = 0; t <= n; ++t) f[t] = ctx.mul(row[t], ctx.mul(pa[n - t], pc[t]));
    }
    for (uint32_t t = 0; t <= n; ++t) M.at(t, 0) = f[t];
    const Elt ainv = up ? ctx.inv(a) : Elt(0);
    const Elt cinv = up ? Elt(0) : ctx.inv(c);
    for (uint32_t j = 0; j < n; ++j) {
        if (up) {
            // h[t] = a^{-1} (b f[t] + d f[t-1] - c h[t-1]), t = 0..n
            Elt prevf = 0, prevh = 0;
            for (uint32_t t = 0; t <= n; ++t) {
                const Elt rhs = ctx.sub(ctx.add(ctx.mul(b, f[t]), ctx.mul(d, prevf)), ctx.mul(c, prevh));
                h[t] = ctx.mul(ainv, rhs);
                prevf = f[t];
                prevh = h[t];
            }
        } else {
            // the equation at index t + 1: a h[t+1] + c h[t] = b f[t+1] + d f[t]; h[n+1] = f[n+1] = 0
            Elt nextf = 0, nexth = 0;
            for (uint32_t t = n + 1; t-- > 0;) {
                const Elt rhs = ctx.sub(ctx.add(ctx.mul(b, nextf), ctx.mul(d, f[t])), ctx.mul(a, nexth));
                h[t] = ctx.mul(cinv, rhs);
                nextf = f[t];
                nexth = h[t];
            }
        }
        std::swap(f, h);
        for (uint32_t t = 0; t <= n; ++t) M.at(t, j + 1) = f[t];
    }
    return true;
}

template<class Elt>
engine::DenseBuf<Elt> rho_n(const engine::ModCtx<Elt>& ctx, const Mat22& g, uint32_t n) {
    if (F3lRouteEnabled()) {
        engine::DenseBuf<Elt> M;
        if (rho_n_recurrence<Elt>(ctx, g, n, M)) {
            g_rho_rec.fetch_add(1, std::memory_order_relaxed);
            return M;
        }
        g_rho_fallback.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_rho_cubic.fetch_add(1, std::memory_order_relaxed);
    }
    return rho_n_cubic<Elt>(ctx, g, n);
}

// ---------------------------------------------------------------------------
// The cochain complex over a ModCtx
// ---------------------------------------------------------------------------
template<class Elt>
engine::DenseBuf<Elt> IsotropyModel::coboundary_impl(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q,
                                                     bool boundary_only) const {
    ENGINE_CHECK_MSG(q <= q_max_, "IsotropyModel::coboundary: q > q_max (the complex is built to q_max + 1)");
    const uint32_t D = n + 1;
    auto local_index = [&](uint32_t deg) {
        std::vector<int64_t> idx(cells_[deg].size(), -1);
        int64_t k = 0;
        for (uint32_t i = 0; i < cells_[deg].size(); ++i)
            if (!boundary_only || cells_[deg][i].boundary) idx[i] = k++;
        return std::make_pair(idx, static_cast<uint32_t>(k));
    };
    const auto [src, ncols] = local_index(q);
    const auto [dst, nrows] = local_index(q + 1);
    engine::DenseBuf<Elt> M(nrows * D, ncols * D);
    std::map<GKey, engine::DenseBuf<Elt>> memo;
    const int64_t pm = static_cast<int64_t>(ctx.p);
    for (const auto& e : diff_[q + 1]) {
        // (delta^q f)(e_col) = sum rho(coeff) f(e_row): block (col, row).
        const int64_t bi = dst[e.col], bj = src[e.row];
        if (bi < 0 || bj < 0) {
            ENGINE_CHECK_MSG(bi < 0, "IsotropyModel: boundary cochain differential leaves R_d");
            continue;
        }
        for (const auto& [g, nmul] : e.coeff.terms) {
            auto it = memo.find(g);
            if (it == memo.end()) it = memo.emplace(g, rho_n(ctx, mat_of(g), n)).first;
            int64_t r = nmul % pm;
            if (r < 0) r += pm;
            const Elt coef = static_cast<Elt>(r);
            if (coef == 0) continue;
            const auto& R = it->second;
            for (uint32_t i = 0; i < D; ++i)
                for (uint32_t j = 0; j < D; ++j) {
                    Elt& dstv = M.at(static_cast<uint32_t>(bi) * D + i, static_cast<uint32_t>(bj) * D + j);
                    dstv = ctx.add(dstv, ctx.mul(coef, R.at(i, j)));
                }
        }
    }
    return M;
}

template<class Elt>
engine::DenseBuf<Elt> IsotropyModel::coboundary(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const {
    return coboundary_impl(ctx, n, q, false);
}

template<class Elt>
engine::DenseBuf<Elt> IsotropyModel::boundary_coboundary(const engine::ModCtx<Elt>& ctx, uint32_t n,
                                                         uint32_t q) const {
    return coboundary_impl(ctx, n, q, true);
}

template<class Elt>
engine::DenseBuf<Elt> IsotropyModel::comparison(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const {
    ENGINE_CHECK_MSG(q <= 2, "IsotropyModel::comparison: C_eq lives in degrees 0..2");
    const uint32_t D = n + 1;
    ENGINE_CHECK_MSG(shipped_, "IsotropyModel::comparison: the shipped pipeline's K is the def:model7 complex only");
    const auto& Kq = K_simplices()[q];
    engine::DenseBuf<Elt> M(rank(q) * D, static_cast<uint32_t>(Kq.size()) * D);
    const Elt one = static_cast<Elt>(1 % ctx.p);
    for (uint32_t r = 0; r < reps_[q].size(); ++r) {
        const uint32_t k = static_cast<uint32_t>(std::find(Kq.begin(), Kq.end(), reps_[q][r]) - Kq.begin());
        ENGINE_CHECK(k < Kq.size());
        const uint32_t ci = cell_index(q, r, 0);
        for (uint32_t i = 0; i < D; ++i) M.at(ci * D + i, k * D + i) = one;
    }
    return M;
}

template<class Elt>
std::vector<uint32_t> IsotropyModel::cohomology_lengths(const engine::ModCtx<Elt>& ctx, uint32_t n,
                                                        bool boundary) const {
    const uint32_t D = n + 1;
    // len_im[q] = length of im delta^q, q = 0 .. q_max.
    std::vector<uint32_t> len_im(q_max_ + 1, 0);
    for (uint32_t q = 0; q <= q_max_; ++q) {
        engine::DenseBuf<Elt> M = boundary ? boundary_coboundary(ctx, n, q) : coboundary(ctx, n, q);
        if (M.rows == 0 || M.cols == 0) continue;
        if (ctx.is_field()) {
            len_im[q] = engine::elim::rref(ctx, engine::view(M)).rank;
        } else {
            len_im[q] = engine::elim::length(ctx, engine::cview(M));
        }
    }
    std::vector<uint32_t> H(q_max_ + 1, 0);
    for (uint32_t q = 0; q <= q_max_; ++q) {
        const uint32_t cells = boundary ? boundary_rank(q) : rank(q);
        const uint32_t total = cells * D * ctx.k;
        const uint32_t sub = len_im[q] + (q > 0 ? len_im[q - 1] : 0);
        ENGINE_CHECK_MSG(sub <= total, "IsotropyModel::cohomology_lengths: negative length");
        H[q] = total - sub;
    }
    return H;
}

#define INTEGRAL_MODEL_INSTANTIATE(Elt)                                                                  \
    template engine::DenseBuf<Elt> rho_n<Elt>(const engine::ModCtx<Elt>&, const Mat22&, uint32_t);       \
    template engine::DenseBuf<Elt> rho_n_cubic<Elt>(const engine::ModCtx<Elt>&, const Mat22&, uint32_t); \
    template bool rho_n_recurrence<Elt>(const engine::ModCtx<Elt>&, const Mat22&, uint32_t,               \
                                        engine::DenseBuf<Elt>&);                                          \
    template engine::DenseBuf<Elt> IsotropyModel::coboundary<Elt>(const engine::ModCtx<Elt>&, uint32_t,   \
                                                                  uint32_t) const;                      \
    template engine::DenseBuf<Elt> IsotropyModel::boundary_coboundary<Elt>(const engine::ModCtx<Elt>&,   \
                                                                           uint32_t, uint32_t) const;   \
    template engine::DenseBuf<Elt> IsotropyModel::comparison<Elt>(const engine::ModCtx<Elt>&, uint32_t,   \
                                                                  uint32_t) const;                      \
    template std::vector<uint32_t> IsotropyModel::cohomology_lengths<Elt>(const engine::ModCtx<Elt>&,     \
                                                                          uint32_t, bool) const;

INTEGRAL_MODEL_INSTANTIATE(uint32_t)
INTEGRAL_MODEL_INSTANTIATE(uint64_t)
#undef INTEGRAL_MODEL_INSTANTIATE

}  // namespace integral_model
