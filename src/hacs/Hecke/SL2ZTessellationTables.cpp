// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S2 -- Tessellation bookkeeping for the 7-vertex model of SL(2,Z)\H.
// See SL2ZTessellationTables.h for the contract references.

#include "SL2ZTessellationTables.h"

#include "HeckeModelDescriptor.h"
#include "../LocalSystemSL2Z.h"       // SL2ZRelationSeeds_7VertexModel (single source of truth, I3)
#include "../SimplicialCohomology.h"  // SimplicialComplex

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <string>

namespace hecke {

namespace {

// Lexicographic (a,b,c,d) order on exact 2x2 matrices (used for the
// canonical-key minimum and for set comparisons).
bool lex_less(const Mat22& x, const Mat22& y) {
    if (x.a != y.a) return x.a < y.a;
    if (x.b != y.b) return x.b < y.b;
    if (x.c != y.c) return x.c < y.c;
    return x.d < y.d;
}

const std::array<uint8_t, 3> kIdentityPerm = { 0, 1, 2 };

// Parity (+1/-1) of the slot permutation restricted to slots 0..q.
int8_t perm_sign(const std::array<uint8_t, 3>& perm, uint32_t q) {
    int inversions = 0;
    for (uint32_t i = 0; i + 1 <= q; ++i) {
        for (uint32_t j = i + 1; j <= q; ++j) {
            if (perm[i] > perm[j]) ++inversions;
        }
    }
    return (inversions % 2 == 0) ? int8_t(1) : int8_t(-1);
}

std::array<uint8_t, 3> compose_perms(const std::array<uint8_t, 3>& outer,
                                     const std::array<uint8_t, 3>& inner) {
    // outer: slots of the middle simplex -> slots of the target,
    // inner: slots of the source simplex -> slots of the middle simplex.
    std::array<uint8_t, 3> r = kIdentityPerm;
    for (uint32_t j = 0; j < 3; ++j) r[j] = outer[inner[j]];
    return r;
}

std::array<uint8_t, 3> invert_perm(const std::array<uint8_t, 3>& perm) {
    std::array<uint8_t, 3> r = kIdentityPerm;
    for (uint32_t j = 0; j < 3; ++j) r[perm[j]] = static_cast<uint8_t>(j);
    return r;
}

// Local simplex lookup on a copied simplex table (sorted vertex lists).
uint32_t find_simplex_index_local(
    const std::vector<std::vector<std::vector<uint32_t>>>& simplices,
    uint32_t q, const std::vector<uint32_t>& verts) {
    const auto& list = simplices.at(q);
    for (uint32_t i = 0; i < list.size(); ++i) {
        if (list[i] == verts) return i;
    }
    throw std::runtime_error("SL2ZTessellationTables: simplex not found in the model complex.");
}

// Finite-stabilizer bound of the S2 spec: every S(a,b) is a coset of a
// stabilizer of order <= 12; exceeding it means the closure diverged (bug).
constexpr size_t kMaxTransporterSetSize = 12;

} // namespace

// --- KuSimplexKey ------------------------------------------------------------

bool KuSimplexKey::operator<(const KuSimplexKey& rhs) const {
    if (q != rhs.q) return q < rhs.q;
    if (sigma_hat != rhs.sigma_hat) return sigma_hat < rhs.sigma_hat;
    if (!(gamma_canon == rhs.gamma_canon)) return lex_less(gamma_canon, rhs.gamma_canon);
    return sign < rhs.sign;
}

bool KuSimplexKey::operator==(const KuSimplexKey& rhs) const {
    return q == rhs.q && sigma_hat == rhs.sigma_hat &&
           gamma_canon == rhs.gamma_canon && sign == rhs.sign;
}

// --- Construction ------------------------------------------------------------

SL2ZTessellationTables::SL2ZTessellationTables(const SimplicialComplex& K)
    : simplices_(K.get_simplices()) {
    // Structural checks: this class is specific to the fixed 7-vertex model
    // (matrix-engine Phase 6, audit item 7.18) -- asserted against the MODEL
    // DESCRIPTOR, not the literals 7/11/5; the predicate is unchanged.
    RequireModel(SL2Z7VertexModel(), &K, "SL2ZTessellationTables");
    for (uint32_t v = 0; v < 7; ++v) {
        if (simplices_[0][v] != std::vector<uint32_t>{ v }) {
            throw std::runtime_error("SL2ZTessellationTables: vertex list is not 0..6 in order.");
        }
    }
    for (uint32_t e = 0; e < simplices_[1].size(); ++e) {
        const auto& verts = simplices_[1][e];
        if (verts.size() != 2 || verts[0] >= verts[1]) {
            throw std::runtime_error("SL2ZTessellationTables: edge list not sorted/canonical.");
        }
        edge_index_[{ verts[0], verts[1] }] = e;
    }

    // Exact fixed vertex geometry (S2 spec; cusp line at height Y = 2, FIXED
    // and documented -- positions only steer tau choices, invariant I8):
    //   v0 = (-1 + 4i)/2 = -1/2 + 2i          (u,v,w,D) = (-1, 4, 2, 1)
    //   v1 = (-1 + 12i)/6 = -1/6 + 2i                    (-1, 12, 6, 1)
    //   v2 = (1 + 12i)/6 = 1/6 + 2i                      ( 1, 12, 6, 1)
    //   v3 = (1 + 4i)/2 = 1/2 + 2i                       ( 1, 4, 2, 1)
    //   v4 = rho^2 = (-1 + sqrt(3) i)/2                  (-1, 1, 2, 3)
    //   v5 = rho = (1 + sqrt(3) i)/2                     ( 1, 1, 2, 3)
    //   v6 = i                                           ( 0, 1, 1, 1)
    vertex_positions_ = {
        QuadPoint(-1, 4, 2, 1), QuadPoint(-1, 12, 6, 1), QuadPoint(1, 12, 6, 1),
        QuadPoint(1, 4, 2, 1),  QuadPoint(-1, 1, 2, 3),  QuadPoint(1, 1, 2, 3),
        QuadPoint(0, 1, 1, 1),
    };

    sets_ = build_closed_sets(simplices_);
    compute_orbit_representatives();
    build_path_templates(); // certifies every template exactly (throws on failure)
    build_boundary_subcomplex();
    {
        const uint32_t code = validate_boundary(); // (B1)-(B3), certified (I8)
        if (code != 0) {
            throw std::runtime_error(
                "SL2ZTessellationTables: boundary subcomplex failed certification (code " +
                std::to_string(code) + ") -- [C, sec:boundarydatum] (B1)-(B3).");
        }
    }
}

// --- Transporter closure ((P4)) ----------------------------------------------

bool SL2ZTessellationTables::insert_elem(SetMap& sets, const SetKey& key,
                                         const Mat22& g,
                                         const std::array<uint8_t, 3>& perm,
                                         uint32_t q) {
    TransporterSet& set = sets[key];
    for (size_t i = 0; i < set.elems.size(); ++i) {
        if (set.elems[i] == g) {
            // The vertex correspondence is determined by g (Gamma acts on H
            // with -I as the only kernel element, and -g carries the same
            // correspondence); an inconsistent duplicate is a hard bug.
            for (uint32_t j = 0; j <= q; ++j) {
                if (set.perms[i][j] != perm[j]) {
                    throw std::runtime_error(
                        "SL2ZTessellationTables: inconsistent vertex correspondence "
                        "for a duplicate transporter element (closure bug).");
                }
            }
            return false;
        }
    }
    set.elems.push_back(g);
    set.perms.push_back(perm);
    if (set.elems.size() > kMaxTransporterSetSize) {
        throw std::runtime_error(
            "SL2ZTessellationTables: transporter set exceeded the finite-stabilizer "
            "bound (closure termination violated).");
    }
    return true;
}

SL2ZTessellationTables::SetMap SL2ZTessellationTables::build_closed_sets(
    const std::vector<std::vector<std::vector<uint32_t>>>& simplices) {
    SetMap sets;

    // Seed 1: S(sigma, sigma) contains {+I, -I} for every simplex (the -I
    // quotient of the S2 risk note: (gamma, sigma) and (-gamma, sigma) are
    // the SAME simplex of K^u).
    for (uint32_t q = 0; q < simplices.size(); ++q) {
        for (uint32_t s = 0; s < simplices[q].size(); ++s) {
            insert_elem(sets, { q, s, s }, Mat22::I(), kIdentityPerm, q);
            insert_elem(sets, { q, s, s }, Mat22::I().neg(), kIdentityPerm, q);
        }
    }

    // Seed 2: the elliptic vertex stabilizers, per the S2 spec:
    // S(v4,v4) = <ST> (order 6), S(v5,v5) = <TS> (order 6),
    // S(v6,v6) = <S> (order 4). ((ST)^3 = (TS)^3 = S^2 = -I, so the -I
    // element is contained in each cyclic group.)
    {
        Mat22 g = Mat22::I();
        for (int k = 0; k < 6; ++k) { insert_elem(sets, { 0, 4, 4 }, g, kIdentityPerm, 0); g = g * Mat22::ST(); }
        g = Mat22::I();
        for (int k = 0; k < 6; ++k) { insert_elem(sets, { 0, 5, 5 }, g, kIdentityPerm, 0); g = g * Mat22::TS(); }
        g = Mat22::I();
        for (int k = 0; k < 4; ++k) { insert_elem(sets, { 0, 6, 6 }, g, kIdentityPerm, 0); g = g * Mat22::S(); }
    }

    // Seed 3: one transporter element per model relation, from the single
    // source of truth (invariant I3): i(s_lhs) = g_r * i(s_rhs), i.e.
    // g_r in S(s_lhs, s_rhs); the vertex correspondence comes from the
    // relation's image_of_rhs data (combinatorial, invariant I8).
    for (const auto& seed : SL2ZRelationSeeds_7VertexModel()) {
        const uint32_t q = seed.q_dim;
        const uint32_t lhs = find_simplex_index_local(simplices, q, seed.lhs_vertices);
        const uint32_t rhs = find_simplex_index_local(simplices, q, seed.rhs_vertices);
        std::array<uint8_t, 3> perm = kIdentityPerm;
        for (uint32_t j = 0; j <= q; ++j) {
            const uint32_t image_vertex = seed.image_of_rhs.at(j);
            const auto& lhs_verts = seed.lhs_vertices;
            const auto it = std::find(lhs_verts.begin(), lhs_verts.end(), image_vertex);
            if (it == lhs_verts.end()) {
                throw std::runtime_error(
                    "SL2ZTessellationTables: relation seed image vertex not in lhs simplex.");
            }
            perm[j] = static_cast<uint8_t>(std::distance(lhs_verts.begin(), it));
        }
        insert_elem(sets, { q, lhs, rhs }, seed.g_r, perm, q);
    }

    // Closure under composition S(a,b)*S(b,c) <= S(a,c), inverse
    // S(a,b)^{-1} = S(b,a) and negation (-I central). Terminates: every
    // S(a,b) is a left coset of the finite stabilizer S(b,b)
    // (order <= 12, guarded in insert_elem).
    while (closure_pass(sets)) {
    }
    return sets;
}

bool SL2ZTessellationTables::closure_pass(SetMap& sets) {
    bool changed = false;

    // Negation.
    for (auto& kv : sets) {
        const uint32_t q = kv.first[0];
        const size_t n = kv.second.elems.size(); // snapshot; new elements next pass
        for (size_t i = 0; i < n; ++i) {
            const Mat22 neg = kv.second.elems[i].neg();
            const std::array<uint8_t, 3> perm = kv.second.perms[i];
            changed |= insert_elem(sets, kv.first, neg, perm, q);
        }
    }

    // Inverse.
    {
        std::vector<std::pair<SetKey, size_t>> snapshot;
        for (const auto& kv : sets) snapshot.emplace_back(kv.first, kv.second.elems.size());
        for (const auto& snap : snapshot) {
            const SetKey& key = snap.first;
            const uint32_t q = key[0];
            const SetKey inv_key = { q, key[2], key[1] };
            for (size_t i = 0; i < snap.second; ++i) {
                const Mat22 g_inv = sets[key].elems[i].inv_unimodular();
                const std::array<uint8_t, 3> p_inv = invert_perm(sets[key].perms[i]);
                changed |= insert_elem(sets, inv_key, g_inv, p_inv, q);
            }
        }
    }

    // Composition.
    {
        std::vector<std::pair<SetKey, size_t>> snapshot;
        for (const auto& kv : sets) snapshot.emplace_back(kv.first, kv.second.elems.size());
        for (const auto& ab : snapshot) {
            for (const auto& bc : snapshot) {
                if (ab.first[0] != bc.first[0]) continue; // same dimension
                if (ab.first[2] != bc.first[1]) continue; // middle simplex matches
                const uint32_t q = ab.first[0];
                const SetKey ac_key = { q, ab.first[1], bc.first[2] };
                for (size_t i = 0; i < ab.second; ++i) {
                    for (size_t j = 0; j < bc.second; ++j) {
                        const Mat22 prod = sets[ab.first].elems[i] * sets[bc.first].elems[j];
                        const std::array<uint8_t, 3> perm = compose_perms(
                            sets[ab.first].perms[i], sets[bc.first].perms[j]);
                        changed |= insert_elem(sets, ac_key, prod, perm, q);
                    }
                }
            }
        }
    }

    return changed;
}

// --- Accessors ----------------------------------------------------------------

const std::vector<Mat22>& SL2ZTessellationTables::transporter(
    uint32_t q, uint32_t s1, uint32_t s2) const {
    static const std::vector<Mat22> empty;
    const auto it = sets_.find({ q, s1, s2 });
    return (it == sets_.end()) ? empty : it->second.elems;
}

const std::vector<std::array<uint8_t, 3>>& SL2ZTessellationTables::transporter_vertex_perms(
    uint32_t q, uint32_t s1, uint32_t s2) const {
    static const std::vector<std::array<uint8_t, 3>> empty;
    const auto it = sets_.find({ q, s1, s2 });
    return (it == sets_.end()) ? empty : it->second.perms;
}

bool SL2ZTessellationTables::same_groupoid_orbit(uint32_t q, uint32_t s1, uint32_t s2) const {
    return !transporter(q, s1, s2).empty();
}

void SL2ZTessellationTables::compute_orbit_representatives() {
    orbit_rep_.assign(simplices_.size(), {});
    for (uint32_t q = 0; q < simplices_.size(); ++q) {
        const uint32_t n = static_cast<uint32_t>(simplices_[q].size());
        orbit_rep_[q].assign(n, 0);
        for (uint32_t s = 0; s < n; ++s) {
            uint32_t rep = s;
            for (uint32_t t = 0; t < s; ++t) {
                if (same_groupoid_orbit(q, s, t)) { rep = t; break; } // smallest first
            }
            // Representative of an already-seen orbit member is transitive-
            // consistent because transporter sets are closed under
            // composition (S(s,t) nonempty and S(t,rep(t)) nonempty imply
            // S(s,rep(t)) nonempty).
            orbit_rep_[q][s] = (rep == s) ? s : orbit_rep_[q][rep];
        }
    }
}

uint32_t SL2ZTessellationTables::orbit_representative(uint32_t q, uint32_t sigma) const {
    return orbit_rep_.at(q).at(sigma);
}

// --- Canonical key -------------------------------------------------------------

KuSimplexKey SL2ZTessellationTables::canonical_key(const Mat22& gamma, uint32_t q,
                                                   uint32_t sigma) const {
    if (q >= simplices_.size() || sigma >= simplices_[q].size()) {
        throw std::runtime_error("SL2ZTessellationTables::canonical_key: simplex out of range.");
    }
    const uint32_t sigma_hat = orbit_rep_[q][sigma];
    const auto it = sets_.find({ q, sigma, sigma_hat });
    if (it == sets_.end() || it->second.elems.empty()) {
        throw std::runtime_error("SL2ZTessellationTables::canonical_key: missing transporter set.");
    }
    // Candidates gamma' = gamma * g over g in S(sigma, sigma_hat):
    // gamma * i(sigma) = gamma * g * i(sigma_hat). The set is closed under
    // -I, so the lexicographic minimum quotients by -I automatically.
    const TransporterSet& set = it->second;
    bool have = false;
    Mat22 best = Mat22::I();
    int8_t best_sign = 1;
    for (size_t i = 0; i < set.elems.size(); ++i) {
        const Mat22 cand = gamma * set.elems[i];
        if (!have || lex_less(cand, best)) {
            have = true;
            best = cand;
            // Orientation: sigma's increasing vertex order pushed to
            // sigma_hat's slots through g's vertex correspondence
            // (combinatorial parity, never geometric -- invariant I8).
            best_sign = perm_sign(set.perms[i], q);
        }
    }
    return KuSimplexKey{ q, sigma_hat, best, best_sign };
}

// --- Word reduction ((P2)) -------------------------------------------------------

std::vector<std::pair<char, int64_t>> SL2ZTessellationTables::word_impl(
    const Mat22& gamma, int64_t& max_intermediate_out) const {
    if (det_tiered(gamma) != 1) {   // (slot-24 ELL-W) the wide tier on the determinant (ExactInt.h)
        throw HeckeInternalError("SL2ZTessellationTables::word_in_T_S: det(gamma) != 1.");
    }
    // Certified no-growth bound of [C, lem:bounds](4): intermediate entries
    // never exceed twice the maximum of the input entries.
    const int64_t input_norm = gamma.max_abs() > 0 ? gamma.max_abs() : 1;
    const int64_t bound = checked_mul<int64_t>(2, input_norm);

    Mat22 M = gamma;
    int64_t max_seen = gamma.max_abs();
    std::vector<std::pair<char, int64_t>> body;

    const auto note = [&](const Mat22& m) {
        const int64_t norm = m.max_abs();
        if (norm > max_seen) max_seen = norm;
        if (norm > bound) {
            throw HeckeOverflow(
                "SL2ZTessellationTables::word_in_T_S: intermediate entry exceeded "
                "the certified (P2) bound.");
        }
    };

    while (M.c != 0) {
        // Nearest-integer quotient q of a/c (|a - q*c| <= |c|/2), with the
        // deterministic tie rule "round the residue up".
        int64_t quot = M.a / M.c;
        int64_t rem = checked_sub<int64_t>(M.a, checked_mul<int64_t>(quot, M.c));
        if (rem != 0 && ((rem < 0) != (M.c < 0))) { // put rem into [0, |c|)
            quot = checked_sub<int64_t>(quot, 1);
            rem = checked_add<int64_t>(rem, M.c);
        }
        if (rem < 0) rem = checked_neg<int64_t>(rem); // rem = |rem| in [0, |c|)
        const int64_t abs_c = (M.c < 0) ? checked_neg<int64_t>(M.c) : M.c;
        if (checked_mul<int64_t>(2, rem) >= abs_c) quot = checked_add<int64_t>(quot, 1);

        // M <- T^{-quot} * M  (subtract quot * row2 from row1)
        const Mat22 M1{ checked_sub<int64_t>(M.a, checked_mul<int64_t>(quot, M.c)),
                        checked_sub<int64_t>(M.b, checked_mul<int64_t>(quot, M.d)),
                        M.c, M.d };
        note(M1);
        // M <- S^{-1} * M1, S^{-1} = [[0,1],[-1,0]]
        const Mat22 M2{ M1.c, M1.d,
                        checked_neg<int64_t>(M1.a), checked_neg<int64_t>(M1.b) };
        note(M2);
        // |new c| = |a - quot*c| <= |c|/2 < |c|: strictly decreasing, terminates.

        if (quot != 0) body.emplace_back('T', quot);
        body.emplace_back('S', int64_t(1));
        M = M2;
    }

    // c == 0 and det == 1 force a = d = eps in {+1, -1}; M = eps * T^{eps*b}.
    if (!((M.a == 1 && M.d == 1) || (M.a == -1 && M.d == -1))) {
        throw HeckeInternalError("SL2ZTessellationTables::word_in_T_S: internal reduction error.");
    }
    const bool negative = (M.a == -1);
    const int64_t tail_exp = negative ? checked_neg<int64_t>(M.b) : M.b;

    std::vector<std::pair<char, int64_t>> word;
    if (negative) word.emplace_back('-', int64_t(1));
    word.insert(word.end(), body.begin(), body.end());
    if (tail_exp != 0) word.emplace_back('T', tail_exp);

    max_intermediate_out = max_seen;
    return word;
}

std::vector<std::pair<char, int64_t>> SL2ZTessellationTables::word_in_T_S(
    const Mat22& gamma) const {
    int64_t max_intermediate = 0;
    return word_impl(gamma, max_intermediate);
}

std::vector<std::pair<char, int64_t>> SL2ZTessellationTables::word_in_T_S_with_bound(
    const Mat22& gamma, int64_t& max_intermediate_out) const {
    return word_impl(gamma, max_intermediate_out);
}

Mat22 SL2ZTessellationTables::evaluate_word(
    const std::vector<std::pair<char, int64_t>>& word) {
    Mat22 M = Mat22::I();
    for (const auto& letter : word) {
        switch (letter.first) {
        case '-': M = M.neg(); break;
        case 'T': M = M * Mat22{ 1, letter.second, 0, 1 }; break;
        case 'S': M = M * Mat22::S(); break;
        default:
            throw HeckeInternalError("SL2ZTessellationTables::evaluate_word: unknown letter.");
        }
    }
    return M;
}

// --- Geometry -------------------------------------------------------------------

QuadPoint SL2ZTessellationTables::vertex_position(uint32_t v) const {
    if (v >= vertex_positions_.size()) {
        throw std::runtime_error("SL2ZTessellationTables::vertex_position: vertex out of range.");
    }
    return vertex_positions_[v];
}

// --- Path templates ([C, alg:tau] step 2) ----------------------------------------

std::vector<uint32_t> SL2ZTessellationTables::bfs_vertex_path(uint32_t a, uint32_t b) const {
    const uint32_t n = static_cast<uint32_t>(simplices_[0].size());
    // Deterministic adjacency: ascending neighbour order.
    std::vector<std::vector<uint32_t>> adj(n);
    for (const auto& kv : edge_index_) {
        adj[kv.first.first].push_back(kv.first.second);
        adj[kv.first.second].push_back(kv.first.first);
    }
    for (auto& list : adj) std::sort(list.begin(), list.end());

    std::vector<int32_t> parent(n, -1);
    std::vector<bool> seen(n, false);
    std::queue<uint32_t> queue;
    queue.push(a);
    seen[a] = true;
    while (!queue.empty()) {
        const uint32_t x = queue.front();
        queue.pop();
        if (x == b) break;
        for (uint32_t y : adj[x]) {
            if (!seen[y]) {
                seen[y] = true;
                parent[y] = static_cast<int32_t>(x);
                queue.push(y);
            }
        }
    }
    if (!seen[b]) {
        throw std::runtime_error("SL2ZTessellationTables: model complex is not connected.");
    }
    std::vector<uint32_t> path;
    for (int32_t x = static_cast<int32_t>(b); x != -1; x = parent[x]) {
        path.push_back(static_cast<uint32_t>(x));
    }
    std::reverse(path.begin(), path.end()); // now a -> b
    return path;
}

void SL2ZTessellationTables::append_path_chain(std::vector<ChainTerm1>& out,
                                               const std::vector<uint32_t>& vertex_path,
                                               const Mat22& gamma) const {
    for (size_t i = 0; i + 1 < vertex_path.size(); ++i) {
        const uint32_t x = vertex_path[i];
        const uint32_t y = vertex_path[i + 1];
        const uint32_t u = std::min(x, y);
        const uint32_t v = std::max(x, y);
        const uint32_t e = edge_index_.at({ u, v });
        // Base edge {u,v} (u < v) is oriented with boundary (v) - (u); the
        // traversal x -> y therefore carries coefficient +1 iff x < y.
        out.push_back(ChainTerm1{ (x < y) ? int32_t(1) : int32_t(-1), gamma, e });
    }
}

uint32_t SL2ZTessellationTables::tree_attachment(uint32_t w) const {
    // t(v0)=v4, t(v1)=v6, t(v2)=v6, t(v3)=v5; tree vertices are fixed. The
    // table commutes with iota_K (v0<->v3, v1<->v2, v4<->v5, v6 fixed) and
    // with the relation i(v3) = T i(v0) (t(v3) = v5 = T v4 = T t(v0)), which
    // is what makes the resulting edge chains Gamma- and iota-equivariant.
    // Each (w, t(w)) with w != t(w) must be an edge of K (asserted).
    static const uint32_t table[7] = { 4, 6, 6, 5, 4, 5, 6 };
    if (w >= 7) throw std::runtime_error("SL2ZTessellationTables::tree_attachment: bad vertex.");
    return table[w];
}

std::vector<ChainTerm1> SL2ZTessellationTables::canonical_template(
    char letter, const Mat22& g, uint32_t w, uint32_t w2) const {
    // Vertices/edges of the finite subgraph of the tree spanned by the tree
    // edges of i(K) and g . i(K), addressed by canonical K^u keys.
    const uint32_t e46 = edge_index_.at({ 4, 6 });
    const uint32_t e56 = edge_index_.at({ 5, 6 });
    struct TreeEdge { Mat22 gamma; uint32_t edge_idx; KuSimplexKey ka, kb; }; // ka = lower vertex
    std::vector<TreeEdge> edges;
    const auto add_edge = [&](const Mat22& gamma, uint32_t e) {
        const auto& verts = simplices_[1][e];
        TreeEdge te{ gamma, e, canonical_key(gamma, 0, verts[0]), canonical_key(gamma, 0, verts[1]) };
        const KuSimplexKey kn = canonical_key(gamma, 1, e);
        for (const auto& x : edges) { // dedup by canonical simplex (S . e46 == e56), sign ignored
            const KuSimplexKey kx = canonical_key(x.gamma, 1, x.edge_idx);
            if (kx.q == kn.q && kx.sigma_hat == kn.sigma_hat && kx.gamma_canon == kn.gamma_canon) return;
        }
        edges.push_back(te);
    };
    add_edge(Mat22::I(), e46);
    add_edge(Mat22::I(), e56);
    add_edge(g, e46);
    add_edge(g, e56);
    std::vector<KuSimplexKey> nodes;
    const auto node_id = [&](const KuSimplexKey& k) -> uint32_t {
        for (uint32_t i = 0; i < nodes.size(); ++i) if (nodes[i] == k) return i;
        nodes.push_back(k);
        return static_cast<uint32_t>(nodes.size() - 1);
    };
    for (const auto& te : edges) { node_id(te.ka); node_id(te.kb); }
    // A tree: |V| = |E| + 1 (connectedness is established by the BFS below).
    if (nodes.size() != edges.size() + 1) {
        throw std::runtime_error("SL2ZTessellationTables: tree-edge subgraph is not a tree.");
    }
    const uint32_t src = node_id(canonical_key(Mat22::I(), 0, tree_attachment(w)));
    const uint32_t dst = node_id(canonical_key(g, 0, tree_attachment(w2)));
    // BFS on the forest; the path (as a chain) is unique, so the traversal
    // order is immaterial to the output.
    std::vector<int32_t> parent_node(nodes.size(), -1), parent_edge(nodes.size(), -1);
    std::vector<bool> seen(nodes.size(), false);
    std::queue<uint32_t> queue;
    queue.push(src);
    seen[src] = true;
    while (!queue.empty()) {
        const uint32_t x = queue.front();
        queue.pop();
        if (x == dst) break;
        for (uint32_t ei = 0; ei < edges.size(); ++ei) {
            const uint32_t a = node_id(edges[ei].ka), b = node_id(edges[ei].kb);
            const uint32_t y = (a == x) ? b : (b == x) ? a : x;
            if (y == x || seen[y]) continue;
            seen[y] = true;
            parent_node[y] = static_cast<int32_t>(x);
            parent_edge[y] = static_cast<int32_t>(ei);
            queue.push(y);
        }
    }
    if (!seen[dst]) {
        throw std::runtime_error("SL2ZTessellationTables: tree-edge subgraph is not connected.");
    }
    std::vector<ChainTerm1> tree_chain;
    for (uint32_t y = dst; y != src; y = static_cast<uint32_t>(parent_node[y])) {
        const TreeEdge& te = edges[static_cast<size_t>(parent_edge[y])];
        // Traversal parent -> y along gamma.{u<v}: +1 iff it runs u -> v,
        // i.e. iff y is the upper endpoint kb.
        const int32_t coeff = (node_id(te.kb) == y) ? int32_t(1) : int32_t(-1);
        tree_chain.push_back(ChainTerm1{ coeff, te.gamma, te.edge_idx });
    }
    std::reverse(tree_chain.begin(), tree_chain.end());

    std::vector<ChainTerm1> chain;
    if (tree_attachment(w) != w) { // spoke(w): w -> t(w), w < t(w) so coeff +1
        chain.push_back(ChainTerm1{ 1, Mat22::I(), edge_index_.at({ w, tree_attachment(w) }) });
    }
    chain.insert(chain.end(), tree_chain.begin(), tree_chain.end());
    if (tree_attachment(w2) != w2) { // - g . spoke(w2)
        chain.push_back(ChainTerm1{ -1, g, edge_index_.at({ w2, tree_attachment(w2) }) });
    }
    (void)letter;
    return chain;
}

void SL2ZTessellationTables::build_path_templates() {
    // Crossing data per letter: i(cross_lhs) = letter * i(cross_rhs), taken
    // from the model's relation table ([C, sec:model]):
    //   'T': i(v3) = T * i(v0);  't' (= T^{-1}): i(v0) = T^{-1} * i(v3);
    //   'S': i(v6) = S * i(v6).
    struct Crossing { char letter; Mat22 g; uint32_t lhs; uint32_t rhs; };
    const Crossing crossings[3] = {
        { 'T', Mat22::T(), 3, 0 },
        { 't', Mat22::T().inv_unimodular(), 0, 3 },
        { 'S', Mat22::S(), 6, 6 },
    };
    const uint32_t num_vertices = static_cast<uint32_t>(simplices_[0].size());
    for (const auto& crossing : crossings) {
        for (uint32_t w = 0; w < num_vertices; ++w) {
            for (uint32_t w2 = 0; w2 < num_vertices; ++w2) {
                for (const PathTemplateFamily family :
                     { PathTemplateFamily::Canonical, PathTemplateFamily::LegacyBfs }) {
                    std::vector<ChainTerm1> chain;
                    if (family == PathTemplateFamily::Canonical) {
                        chain = canonical_template(crossing.letter, crossing.g, w, w2);
                    } else {
                        // Legacy: path i(w) -> i(cross_lhs) inside i(K), then
                        // letter * (path i(cross_rhs) -> i(w2)) inside
                        // letter * i(K); they share the boundary simplex
                        // i(cross_lhs) = letter * i(cross_rhs).
                        append_path_chain(chain, bfs_vertex_path(w, crossing.lhs), Mat22::I());
                        append_path_chain(chain, bfs_vertex_path(crossing.rhs, w2), crossing.g);
                    }
                    const uint32_t code =
                        certify_path_template(crossing.letter, w, w2, chain);
                    if (code != 0) {
                        throw std::runtime_error(
                            "SL2ZTessellationTables: path template failed exact boundary "
                            "certification (letter " + std::string(1, crossing.letter) +
                            ", w=" + std::to_string(w) + ", w2=" + std::to_string(w2) +
                            ", family " + std::to_string(static_cast<int>(family)) + ").");
                    }
                    path_templates_[{ static_cast<uint32_t>(family),
                                      static_cast<uint32_t>(crossing.letter), w, w2 }] =
                        std::move(chain);
                }
            }
        }
    }
}

uint32_t SL2ZTessellationTables::certify_path_template(
    char letter, uint32_t w, uint32_t w2, const std::vector<ChainTerm1>& chain) const {
    // Exact certification over Z (invariant I8): the boundary of the chain,
    // reduced to canonical K^u vertex keys, must equal
    // (letter * i(w2)) - (i(w)). Vertex keys always carry sign +1 (a single
    // slot has no nontrivial permutation), so signed accumulation over
    // KuSimplexKey is exact integer bookkeeping.
    std::map<KuSimplexKey, int64_t> boundary;
    const auto accumulate = [&](const Mat22& gamma, uint32_t vertex, int64_t coeff) {
        const KuSimplexKey key = canonical_key(gamma, 0, vertex);
        boundary[key] += coeff * key.sign;
    };
    for (const auto& term : chain) {
        const auto& verts = simplices_[1][term.edge_idx]; // {u, v}, u < v
        accumulate(term.gamma, verts[1], term.coeff);
        accumulate(term.gamma, verts[0], -static_cast<int64_t>(term.coeff));
    }
    Mat22 g = Mat22::I();
    switch (letter) {
    case 'T': g = Mat22::T(); break;
    case 't': g = Mat22::T().inv_unimodular(); break;
    case 'S': g = Mat22::S(); break;
    default: return 100;
    }
    accumulate(g, w2, -1);       // subtract the target 0-chain ...
    accumulate(Mat22::I(), w, 1); // ... and add back the source
    for (const auto& kv : boundary) {
        if (kv.second != 0) return 101; // nonzero residual: reject (I8)
    }
    return 0;
}

const std::vector<ChainTerm1>& SL2ZTessellationTables::path_template(
    char letter, uint32_t w, uint32_t w2, PathTemplateFamily family) const {
    const auto it = path_templates_.find({ static_cast<uint32_t>(family),
                                           static_cast<uint32_t>(letter), w, w2 });
    if (it == path_templates_.end()) {
        throw std::runtime_error(
            "SL2ZTessellationTables::path_template: unknown template (letter must be "
            "'T', 't' or 'S'; vertices 0..6).");
    }
    return it->second;
}

// --- CuspKey (P^1(Q) labels of the boundary components) -------------------------

CuspKey CuspKey::normalise(int64_t a, int64_t c) {
    if (a == 0 && c == 0) throw HeckeInternalError("CuspKey: (0 : 0) is not a point of P^1(Q).");
    uint64_t g = uabs64(a);
    uint64_t h = uabs64(c);
    while (h != 0) { const uint64_t t = g % h; g = h; h = t; }
    if (g > static_cast<uint64_t>(INT64_MAX)) throw HeckeOverflow("CuspKey: gcd exceeds int64.");
    const int64_t gi = static_cast<int64_t>(g);
    a /= gi;
    c /= gi;
    // Sign: c > 0, or c == 0 and a == 1.
    if (c < 0 || (c == 0 && a < 0)) {
        a = checked_neg<int64_t>(a);
        c = checked_neg<int64_t>(c);
    }
    return CuspKey{ a, c };
}

CuspKey CuspKey::of(const Mat22& gamma) {
    if (gamma.det() != 1) throw HeckeInternalError("CuspKey::of: det(gamma) != 1.");
    return normalise(gamma.a, gamma.c);
}

CuspKey CuspKey::apply(const Mat22& M, const CuspKey& c) {
    if (M.det() == 0) throw HeckeInternalError("CuspKey::apply: singular matrix.");
    const int64_t a2 = checked_add<int64_t>(checked_mul<int64_t>(M.a, c.a), checked_mul<int64_t>(M.b, c.c));
    const int64_t c2 = checked_add<int64_t>(checked_mul<int64_t>(M.c, c.a), checked_mul<int64_t>(M.d, c.c));
    return normalise(a2, c2);
}

Mat22 CuspKey::reduce_to_infinity(const CuspKey& c) {
    // Left-multiply the column (a, c) by T^{-q} then S^{-1} until c == 0,
    // exactly as the (P2) word reduction does on a matrix's first column
    // (word_impl above): q = nearest integer to a / c, ties "round the
    // residue up". gamma_red is the product of the left factors; the column
    // ends as (+-1, 0) because it is primitive.
    int64_t a = c.a;
    int64_t cc = c.c;
    Mat22 gred = Mat22::I();
    uint32_t guard = 0;
    while (cc != 0) {
        if (++guard > 100000u) {
            throw HeckeInternalError("CuspKey::reduce_to_infinity: reduction failed to terminate.");
        }
        int64_t quot = a / cc;
        int64_t rem = checked_sub<int64_t>(a, checked_mul<int64_t>(quot, cc));
        if (rem != 0 && ((rem < 0) != (cc < 0))) { // rem into [0, |cc|)
            quot = checked_sub<int64_t>(quot, 1);
            rem = checked_add<int64_t>(rem, cc);
        }
        if (rem < 0) rem = checked_neg<int64_t>(rem);
        const int64_t abs_c = (cc < 0) ? checked_neg<int64_t>(cc) : cc;
        if (checked_mul<int64_t>(2, rem) >= abs_c) quot = checked_add<int64_t>(quot, 1);
        // (a, c) <- T^{-quot} (a, c) = (a - quot c, c); then S^{-1} (a, c) = (c, -a).
        const int64_t a1 = checked_sub<int64_t>(a, checked_mul<int64_t>(quot, cc));
        const Mat22 Tshift{ 1, checked_neg<int64_t>(quot), 0, 1 };
        const Mat22 Sinv{ 0, 1, -1, 0 };
        gred = Sinv * (Tshift * gred);
        a = cc;
        cc = checked_neg<int64_t>(a1);
    }
    if (!(a == 1 || a == -1)) {
        throw HeckeInternalError("CuspKey::reduce_to_infinity: column is not primitive.");
    }
    if (gred.det() != 1) throw HeckeInternalError("CuspKey::reduce_to_infinity: internal error.");
    return gred;
}

// --- The boundary subcomplex ([C, sec:boundarydatum] (B1)-(B3)) -------------------

void SL2ZTessellationTables::build_boundary_subcomplex() {
    // The horocycle path of [C, rmk:bdrymodel]: v0 - v1 - v2 - v3 with
    // i(v3) = T . i(v0), read off the model's own data: the horocycle
    // vertices are those whose exact position has the truncation height
    // Y = 2 (Im = v/w with D = 1), in the order of their real parts; the
    // T-relation i(v3) = T i(v0) is seed row 1 of
    // SL2ZRelationSeeds_7VertexModel (so S(v3, v0) contains T), which
    // validate_boundary() asserts.
    boundary_translation_ = Mat22::T();
    std::vector<std::pair<std::pair<int64_t, int64_t>, uint32_t>> horo; // ((re num, re den), v)
    for (uint32_t v = 0; v < vertex_positions_.size(); ++v) {
        const QuadPoint& P = vertex_positions_[v];
        // Im = (v/w) sqrt(D): on the horocycle iff D == 1 and v/w == 2/1.
        if (P.D == 1 && P.v == checked_mul<int64_t>(2, P.w)) {
            horo.push_back({ { P.u, P.w }, v });
        }
    }
    std::sort(horo.begin(), horo.end(), [](const auto& x, const auto& y) {
        // x.u / x.w < y.u / y.w  (w > 0)
        return checked_mul<int64_t>(x.first.first, y.first.second) <
               checked_mul<int64_t>(y.first.first, x.first.second);
    });
    boundary_line_vertices_.clear();
    for (const auto& h : horo) boundary_line_vertices_.push_back(h.second);
    if (boundary_line_vertices_.size() < 2) {
        throw std::runtime_error("SL2ZTessellationTables: no horocycle path found.");
    }
    boundary_period_ = static_cast<uint32_t>(boundary_line_vertices_.size() - 1);
    boundary_line_edges_.clear();
    for (uint32_t i = 0; i + 1 < boundary_line_vertices_.size(); ++i) {
        const uint32_t a = boundary_line_vertices_[i];
        const uint32_t b = boundary_line_vertices_[i + 1];
        const auto it = edge_index_.find({ std::min(a, b), std::max(a, b) });
        if (it == edge_index_.end()) {
            throw std::runtime_error("SL2ZTessellationTables: horocycle vertices are not joined by an edge.");
        }
        boundary_line_edges_.push_back(it->second);
    }
    boundary_dim_ = 1;
    boundary_simplices_.assign(simplices_.size(), {});
    boundary_simplices_[0] = boundary_line_vertices_;
    std::sort(boundary_simplices_[0].begin(), boundary_simplices_[0].end());
    boundary_simplices_[1] = boundary_line_edges_;
    std::sort(boundary_simplices_[1].begin(), boundary_simplices_[1].end());
}

const std::vector<uint32_t>& SL2ZTessellationTables::boundary_simplices(uint32_t q) const {
    static const std::vector<uint32_t> empty;
    return (q < boundary_simplices_.size()) ? boundary_simplices_[q] : empty;
}

bool SL2ZTessellationTables::is_boundary_simplex(uint32_t q, uint32_t sigma) const {
    const auto& list = boundary_simplices(q);
    return std::find(list.begin(), list.end(), sigma) != list.end();
}

uint32_t SL2ZTessellationTables::boundary_line_index(uint32_t vertex) const {
    for (uint32_t i = 0; i < boundary_line_vertices_.size(); ++i) {
        if (boundary_line_vertices_[i] == vertex) return i;
    }
    throw std::runtime_error("SL2ZTessellationTables::boundary_line_index: not a boundary vertex.");
}

int64_t SL2ZTessellationTables::boundary_line_position(const Mat22& gamma_inf, uint32_t vertex) const {
    // gamma_inf = +-T^k = [[e, e k],[0, e]], e = +-1: k = a * b.
    if (!(gamma_inf.c == 0 && gamma_inf.a == gamma_inf.d && (gamma_inf.a == 1 || gamma_inf.a == -1))) {
        throw std::runtime_error("SL2ZTessellationTables::boundary_line_position: gamma is not in Gamma_inf.");
    }
    const int64_t k = checked_mul<int64_t>(gamma_inf.a, gamma_inf.b);
    return checked_add<int64_t>(checked_mul<int64_t>(k, static_cast<int64_t>(boundary_period_)),
                                static_cast<int64_t>(boundary_line_index(vertex)));
}

namespace {
// floor(P / m) for m > 0.
int64_t floor_div(int64_t P, int64_t m) {
    int64_t q = P / m;
    if ((P % m != 0) && ((P < 0) != (m < 0))) --q;
    return q;
}
} // namespace

std::pair<Mat22, uint32_t> SL2ZTessellationTables::boundary_line_vertex(int64_t P) const {
    const int64_t per = static_cast<int64_t>(boundary_period_);
    const int64_t k = floor_div(P, per);
    const int64_t i = checked_sub<int64_t>(P, checked_mul<int64_t>(k, per));
    const Mat22 Tk{ 1, k, 0, 1 };
    return { Tk, boundary_line_vertices_[static_cast<size_t>(i)] };
}

SL2ZTessellationTables::LineEdge SL2ZTessellationTables::boundary_line_edge(int64_t P) const {
    const int64_t per = static_cast<int64_t>(boundary_period_);
    const int64_t k = floor_div(P, per);
    const int64_t i = checked_sub<int64_t>(P, checked_mul<int64_t>(k, per));
    const uint32_t va = boundary_line_vertices_[static_cast<size_t>(i)];
    const uint32_t vb = boundary_line_vertices_[static_cast<size_t>(i + 1)];
    // K's base edge {min, max} has boundary (max) - (min); the traversal
    // va -> vb (position P -> P+1) carries +1 iff va < vb.
    const Mat22 Tk{ 1, k, 0, 1 };
    return LineEdge{ Tk, boundary_line_edges_[static_cast<size_t>(i)], (va < vb) ? int32_t(1) : int32_t(-1) };
}

void SL2ZTessellationTables::boundary_line_real_part(int64_t P, int64_t& num_out, int64_t& den_out) const {
    const int64_t per = static_cast<int64_t>(boundary_period_);
    const int64_t k = floor_div(P, per);
    const int64_t i = checked_sub<int64_t>(P, checked_mul<int64_t>(k, per));
    const QuadPoint& pos = vertex_positions_[boundary_line_vertices_[static_cast<size_t>(i)]];
    // Re(T^k . pos) = u/w + k * shift, shift = T.b (T = [[1, shift],[0, 1]]).
    const int64_t shift = boundary_translation_.b;
    num_out = checked_add<int64_t>(pos.u, checked_mul<int64_t>(checked_mul<int64_t>(k, shift), pos.w));
    den_out = pos.w;
}

int64_t SL2ZTessellationTables::boundary_line_nearest(int64_t x_num, int64_t x_den) const {
    if (x_den <= 0) throw HeckeInternalError("SL2ZTessellationTables::boundary_line_nearest: den <= 0.");
    using I = __int128;
    const int64_t shift = boundary_translation_.b; // T = [[1, shift],[0, 1]], shift > 0 (asserted)
    // For each vertex i of the first period, with real part r_i = u_i / w_i:
    //   t = x - r_i = (x_num w_i - u_i x_den) / (x_den w_i) =: tn / td,
    //   k_i = nearest integer to t / shift, ties rounded UP,
    //   dist_i = | t - k_i shift |.
    // Choose the i with minimal dist_i; ties go to the larger real part
    // k_i shift + r_i. Everything is an exact comparison of fractions.
    int64_t best_P = 0;
    I best_dn = 0, best_dd = 1;     // best distance as dn/dd
    I best_rn = 0, best_rd = 1;     // best real part as rn/rd
    bool have = false;
    for (uint32_t i = 0; i < boundary_period_; ++i) {
        const QuadPoint& pos = vertex_positions_[boundary_line_vertices_[i]];
        const I tn = checked_sub<I>(checked_mul<I>(static_cast<I>(x_num), static_cast<I>(pos.w)),
                                    checked_mul<I>(static_cast<I>(pos.u), static_cast<I>(x_den)));
        const I td = checked_mul<I>(static_cast<I>(x_den), static_cast<I>(pos.w)); // > 0
        // k = floor((t/shift) + 1/2) = floor((2 tn + td shift) / (2 td shift)).
        const I sn = checked_add<I>(checked_mul<I>(static_cast<I>(2), tn), checked_mul<I>(td, static_cast<I>(shift)));
        const I sd = checked_mul<I>(checked_mul<I>(static_cast<I>(2), td), static_cast<I>(shift));
        I k = sn / sd;
        if ((sn % sd != 0) && (sn < 0)) --k; // floor for negative numerators (sd > 0)
        // dist = |tn - k shift td| / td
        I dn = checked_sub<I>(tn, checked_mul<I>(checked_mul<I>(k, static_cast<I>(shift)), td));
        if (dn < 0) dn = -dn;
        const I dd = td;
        // real part = (u_i + k shift w_i) / w_i
        const I rn = checked_add<I>(static_cast<I>(pos.u),
                                    checked_mul<I>(checked_mul<I>(k, static_cast<I>(shift)), static_cast<I>(pos.w)));
        const I rd = static_cast<I>(pos.w);
        bool take = false;
        if (!have) {
            take = true;
        } else {
            const I lhs = checked_mul<I>(dn, best_dd);
            const I rhs = checked_mul<I>(best_dn, dd);
            if (lhs < rhs) take = true;
            else if (lhs == rhs) {
                // tie: larger real part wins
                take = checked_mul<I>(rn, best_rd) > checked_mul<I>(best_rn, rd);
            }
        }
        if (take) {
            have = true;
            best_dn = dn; best_dd = dd; best_rn = rn; best_rd = rd;
            if (k > static_cast<I>(INT64_MAX / 8) || k < static_cast<I>(INT64_MIN / 8)) {
                throw HeckeOverflow("SL2ZTessellationTables::boundary_line_nearest: position overflow.");
            }
            best_P = checked_add<int64_t>(checked_mul<int64_t>(static_cast<int64_t>(k),
                                                               static_cast<int64_t>(boundary_period_)),
                                          static_cast<int64_t>(i));
        }
    }
    return best_P;
}

uint32_t SL2ZTessellationTables::validate_boundary() const {
    // (B1) the line data: a path, closing up under T.
    const size_t nv = boundary_line_vertices_.size();
    if (nv < 2 || boundary_line_edges_.size() + 1 != nv) return 1;
    if (boundary_period_ != nv - 1) return 1;
    if (!(boundary_translation_ == Mat22::T())) return 2;
    for (size_t i = 0; i + 1 < nv; ++i) {
        const uint32_t a = boundary_line_vertices_[i];
        const uint32_t b = boundary_line_vertices_[i + 1];
        const auto it = edge_index_.find({ std::min(a, b), std::max(a, b) });
        if (it == edge_index_.end() || it->second != boundary_line_edges_[i]) return 3;
    }
    {   // i(v_last) = T . i(v_first): T in S(v_last, v_first).
        const auto& ts = transporter(0, boundary_line_vertices_[nv - 1], boundary_line_vertices_[0]);
        bool found = false;
        for (const auto& t : ts) if (t == boundary_translation_) found = true;
        if (!found) return 4;
    }
    // The boundary simplex lists: exactly the line vertices / edges, face-closed,
    // and no 2-simplex of K lies in the boundary (a 1-dimensional subcomplex).
    {
        std::vector<uint32_t> vs = boundary_line_vertices_;
        std::sort(vs.begin(), vs.end());
        if (boundary_simplices(0) != vs) return 5;
        std::vector<uint32_t> es = boundary_line_edges_;
        std::sort(es.begin(), es.end());
        if (boundary_simplices(1) != es) return 5;
        for (uint32_t q = 2; q < simplices_.size(); ++q) {
            if (!boundary_simplices(q).empty()) return 5;
        }
        for (uint32_t e : boundary_simplices(1)) {
            for (uint32_t v : simplices_[1][e]) if (!is_boundary_simplex(0, v)) return 6;
        }
        for (uint32_t f = 0; f < simplices_[2].size(); ++f) {
            bool all = true;
            for (uint32_t v : simplices_[2][f]) all = all && is_boundary_simplex(0, v);
            if (all) return 7;
        }
    }
    // (B2) orbits on the line: the vertex orbits are {v_first, v_last} and
    // singletons; the edges are pairwise inequivalent ([C, rmk:bdrymodel]:
    // three Gamma_inf-orbits of vertices and three of edges).
    for (size_t i = 0; i < nv; ++i) {
        for (size_t j = i + 1; j < nv; ++j) {
            const bool same = same_groupoid_orbit(0, boundary_line_vertices_[i], boundary_line_vertices_[j]);
            const bool expected = (i == 0 && j == nv - 1);
            if (same != expected) return 8;
        }
    }
    for (size_t i = 0; i < boundary_line_edges_.size(); ++i) {
        for (size_t j = i + 1; j < boundary_line_edges_.size(); ++j) {
            if (same_groupoid_orbit(1, boundary_line_edges_[i], boundary_line_edges_[j])) return 9;
        }
    }
    // (B3) stabilisers are exactly {+-I}.
    for (uint32_t q = 0; q <= boundary_dim_; ++q) {
        for (uint32_t s : boundary_simplices(q)) {
            const auto& st = transporter(q, s, s);
            if (st.size() != 2) return 10;
            const bool has_pos = (st[0] == Mat22::I()) || (st[1] == Mat22::I());
            const bool has_neg = (st[0] == Mat22::I().neg()) || (st[1] == Mat22::I().neg());
            if (!has_pos || !has_neg) return 10;
        }
    }
    // Geometry: one horocycle (equal imaginary parts, D = 1), increasing real
    // parts along the line, pos(v_last) = T . pos(v_first).
    {
        const QuadPoint& P0 = vertex_positions_[boundary_line_vertices_[0]];
        for (size_t i = 0; i < nv; ++i) {
            const QuadPoint& P = vertex_positions_[boundary_line_vertices_[i]];
            if (P.D != 1) return 11;
            if (checked_mul<int64_t>(P.v, P0.w) != checked_mul<int64_t>(P0.v, P.w)) return 11;
            if (i + 1 < nv) {
                const QuadPoint& Q = vertex_positions_[boundary_line_vertices_[i + 1]];
                if (!(checked_mul<int64_t>(P.u, Q.w) < checked_mul<int64_t>(Q.u, P.w))) return 12;
            }
        }
        if (!(vertex_positions_[boundary_line_vertices_[nv - 1]] == P0.moebius(boundary_translation_))) return 13;
    }
    // Line coordinates round-trip, and nearest() fixes every vertex and is
    // T-equivariant; the edge P -> P+1 has boundary (P+1) - (P) through
    // canonical keys.
    for (int64_t P = -7; P <= 7; ++P) {
        const auto vt = boundary_line_vertex(P);
        if (boundary_line_position(vt.first, vt.second) != P) return 14;
        if (boundary_line_position(vt.first.neg(), vt.second) != P) return 14;
        int64_t num = 0, den = 1;
        boundary_line_real_part(P, num, den);
        if (boundary_line_nearest(num, den) != P) return 15;
        if (boundary_line_nearest(checked_add<int64_t>(num, den), den) !=
            checked_add<int64_t>(P, static_cast<int64_t>(boundary_period_))) return 16;
        const LineEdge le = boundary_line_edge(P);
        const auto& ev = simplices_[1][le.edge_idx];
        std::map<KuSimplexKey, int64_t> bd;
        bd[canonical_key(le.gamma, 0, ev[1])] += le.coeff;
        bd[canonical_key(le.gamma, 0, ev[0])] -= le.coeff;
        const auto v1 = boundary_line_vertex(P + 1);
        bd[canonical_key(v1.first, 0, v1.second)] -= 1;
        bd[canonical_key(vt.first, 0, vt.second)] += 1;
        for (const auto& kv : bd) if (kv.second != 0) return 17;
    }
    if (boundary_line_index(boundary_line_vertices_[nv - 1]) != boundary_period_) return 18;
    if (CuspKey::of(boundary_translation_) != CuspKey::infinity()) return 19;
    return 0;
}

// --- Self-validation --------------------------------------------------------------

uint32_t SL2ZTessellationTables::validate_closure() const {
    // (1) Fixed point: one more closure pass on a copy adds nothing, and an
    // independent rebuild from the seeds reproduces the stored tables.
    {
        SetMap copy = sets_;
        if (closure_pass(copy)) return 1;
        SetMap rebuilt = build_closed_sets(simplices_);
        if (rebuilt.size() != sets_.size()) return 2;
        for (const auto& kv : sets_) {
            const auto it = rebuilt.find(kv.first);
            if (it == rebuilt.end()) return 2;
            if (it->second.elems.size() != kv.second.elems.size()) return 2;
            for (size_t i = 0; i < kv.second.elems.size(); ++i) {
                if (!(it->second.elems[i] == kv.second.elems[i])) return 2;
                if (it->second.perms[i] != kv.second.perms[i]) return 2;
            }
        }
    }

    // (2) Every stored transporter is in SL(2,Z) with a valid vertex
    // correspondence; for this model all correspondences are slot-identities
    // (the identifications are vertex-order-compatible, S2 risk note), so
    // every canonical-key sign is +1.
    for (const auto& kv : sets_) {
        const uint32_t q = kv.first[0];
        for (size_t i = 0; i < kv.second.elems.size(); ++i) {
            if (kv.second.elems[i].det() != 1) return 3;
            const auto& perm = kv.second.perms[i];
            bool used[3] = { false, false, false };
            for (uint32_t j = 0; j <= q; ++j) {
                if (perm[j] > q || used[perm[j]]) return 4;
                used[perm[j]] = true;
            }
            for (uint32_t j = 0; j <= q; ++j) {
                if (perm[j] != j) return 5; // model is vertex-order-compatible
            }
        }
    }

    // (3) Expected stabilizer orders: vertices (2,2,2,2,6,6,4); every edge
    // and triangle stabilizer has order 2 (= {+-I}).
    const size_t expected_vertex_stab[7] = { 2, 2, 2, 2, 6, 6, 4 };
    for (uint32_t v = 0; v < 7; ++v) {
        if (transporter(0, v, v).size() != expected_vertex_stab[v]) return 6;
    }
    for (uint32_t q = 1; q <= 2; ++q) {
        for (uint32_t s = 0; s < simplices_[q].size(); ++s) {
            if (transporter(q, s, s).size() != 2) return 7;
        }
    }

    // (4) Coset property: nonempty S(a,b) has |S(a,b)| = |S(b,b)| (it is a
    // left coset of the stabilizer of b).
    for (const auto& kv : sets_) {
        const uint32_t q = kv.first[0];
        const uint32_t b = kv.first[2];
        if (kv.second.elems.size() != transporter(q, b, b).size()) return 8;
    }

    // (5) Expected orbits (vertices {0,3}, {1}, {2}, {4,5}, {6}; edges
    // {0,4}~{3,5} and {4,6}~{5,6}, the rest singletons; triangles all
    // singletons) and the specific cross-transporter contents
    // S(v3,v0) = {T,-T}, S({3,5},{0,4}) = {T,-T}, S({4,6},{5,6}) = {S,-S}.
    {
        const uint32_t expected_vertex_rep[7] = { 0, 1, 2, 0, 4, 4, 6 };
        for (uint32_t v = 0; v < 7; ++v) {
            if (orbit_representative(0, v) != expected_vertex_rep[v]) return 9;
        }
        const uint32_t e04 = edge_index_.at({ 0, 4 });
        const uint32_t e35 = edge_index_.at({ 3, 5 });
        const uint32_t e46 = edge_index_.at({ 4, 6 });
        const uint32_t e56 = edge_index_.at({ 5, 6 });
        for (uint32_t e = 0; e < simplices_[1].size(); ++e) {
            uint32_t expected = e;
            if (e == e04 || e == e35) expected = std::min(e04, e35);
            if (e == e46 || e == e56) expected = std::min(e46, e56);
            if (orbit_representative(1, e) != expected) return 10;
        }
        for (uint32_t f = 0; f < simplices_[2].size(); ++f) {
            if (orbit_representative(2, f) != f) return 11;
        }

        const auto set_is_plus_minus = [&](const std::vector<Mat22>& set, const Mat22& g) {
            if (set.size() != 2) return false;
            const bool has_pos = (set[0] == g) || (set[1] == g);
            const bool has_neg = (set[0] == g.neg()) || (set[1] == g.neg());
            return has_pos && has_neg;
        };
        if (!set_is_plus_minus(transporter(0, 3, 0), Mat22::T())) return 12;
        if (!set_is_plus_minus(transporter(1, e35, e04), Mat22::T())) return 13;
        if (!set_is_plus_minus(transporter(1, e46, e56), Mat22::S())) return 14;
        // The elliptic transporters S(v5,v4) (contains T) and S(v4,v5)
        // (contains S) are full stabilizer cosets of order 6.
        const auto contains = [&](const std::vector<Mat22>& set, const Mat22& g) {
            for (const auto& x : set) if (x == g) return true;
            return false;
        };
        if (transporter(0, 5, 4).size() != 6 || !contains(transporter(0, 5, 4), Mat22::T())) return 15;
        if (transporter(0, 4, 5).size() != 6 || !contains(transporter(0, 4, 5), Mat22::S())) return 16;
    }

    // (6) Re-certify every path template exactly.
    for (const auto& kv : path_templates_) {
        const char letter = static_cast<char>(kv.first[1]);
        if (certify_path_template(letter, kv.first[2], kv.first[3], kv.second) != 0) {
            return 17;
        }
    }

    return 0;
}

} // namespace hecke
