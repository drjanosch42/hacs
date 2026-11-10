// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_SL2Z_TESSELLATION_TABLES_H
#define HECKE_SL2Z_TESSELLATION_TABLES_H

// Stage S2 -- Tessellation bookkeeping for the 7-vertex model of SL(2,Z)\H.
//
// Contract: docs/latex/hecke_action.tex ([C]), [C, def:primitives] (P2)
// (word reduction in Gamma via Euclid on the first column, with the
// no-growth intermediate-entry bound of [C, lem:bounds](4)) and (P4)
// (translate bookkeeping: finite transporter sets closed under
// composition/inverse); [C, sec:model] (the relation/generator dictionary);
// path templates of [C, alg:tau] step 2.
//
// Binding invariants: I3 (relations carry group elements; the seed data here
// comes from SL2ZRelationSeeds_7VertexModel(), the single source of truth in
// LocalSystemSL2Z.cpp), I8 (certification over geometry: orientation signs
// are carried through composition of the stored vertex correspondences,
// NEVER recomputed geometrically; the exact vertex positions below only
// steer tau choices in S4). All arithmetic is exact over Z through the
// overflow-checked hecke::Mat22 of Stage S1.

#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "ExactInt.h"
#include "QuadPoint.h"

class SimplicialComplex; // 7-vertex model complex (src/hacs)

namespace hecke {

// Canonical form of an ORIENTED simplex of the tessellation K^u = Gamma * i(K).
// A simplex of K^u is a pair (gamma, sigma) representing gamma * i(sigma),
// sigma taken in increasing vertex order ((P4) of [C, def:primitives]).
// The key quotients by the FULL identification groupoid, in particular by
// -I (so (gamma, sigma) and (-gamma, sigma) have identical keys: -I acts
// trivially on H, S2 risk note).
struct KuSimplexKey {
    uint32_t q;          // dimension
    uint32_t sigma_hat;  // orbit representative in K (canonical simplex list index)
    Mat22 gamma_canon;   // lexicographically minimal (a,b,c,d) representative
    int8_t sign;         // orientation vs. increasing-vertex-order convention
                         // (parity of the vertex-slot permutation of the
                         // chosen transporter; combinatorial, never geometric)

    bool operator<(const KuSimplexKey& rhs) const;
    bool operator==(const KuSimplexKey& rhs) const;
    bool operator!=(const KuSimplexKey& rhs) const { return !(*this == rhs); }
};

// One term of an integer 1-chain in K^u: coeff * (gamma * i(edge_idx)),
// where edge_idx indexes K's 1-simplex list and the base edge {u,v} (u < v)
// is oriented so that its boundary is (v) - (u).
struct ChainTerm1 {
    int32_t coeff;
    Mat22 gamma;
    uint32_t edge_idx;
};

// A point of P^1(Q) -- a cusp -- as a PRIMITIVE integer column (a : c),
// normalised so that c > 0, or c == 0 and a == 1 (the cusp infinity).
// [C, sec:boundarydatum] (B2): pi_0(X_bd) -> P^1(Q) is a Gamma-equivariant
// bijection, and the component of the boundary simplex gamma . i(sigma) is
// the cusp gamma . infinity = (gamma.a : gamma.c). Exact integer only.
struct CuspKey {
    int64_t a;
    int64_t c;

    static CuspKey infinity() { return CuspKey{ 1, 0 }; }

    // gcd- and sign-normalised (a : c); throws HeckeOverflow on (0 : 0).
    static CuspKey normalise(int64_t a, int64_t c);

    // gamma . infinity for gamma in Gamma (its first column).
    static CuspKey of(const Mat22& gamma);

    // The Moebius action M . (a : c) = (M.a a + M.b c : M.c a + M.d c) of an
    // integral M with det M != 0 on P^1(Q). Projective: adj(g) = N g^{-1}
    // acts exactly as g^{-1} does ([C, lem:bdrycomponent]: c -> g^{-1} c),
    // so no rational number is ever formed.
    static CuspKey apply(const Mat22& M, const CuspKey& c);

    // gamma_red in Gamma with gamma_red . c = infinity, found by the
    // nearest-integer Euclidean algorithm on the column (the (P2) device of
    // [C, def:primitives] applied to a column; deterministic tie rule "round
    // the residue up"). Overflow-checked.
    static Mat22 reduce_to_infinity(const CuspKey& c);

    bool operator==(const CuspKey& r) const { return a == r.a && c == r.c; }
    bool operator!=(const CuspKey& r) const { return !(*this == r); }
    bool operator<(const CuspKey& r) const { return (a != r.a) ? a < r.a : c < r.c; }
};

// Exact bookkeeping tables for the fixed 7-vertex model:
//   * finite transporter sets S(sigma1, sigma2) = { g in Gamma :
//     i(sigma1) = g * i(sigma2) }, seeded from the relation/generator table
//     (+ the +-I and elliptic stabilizers) and closed under composition,
//     inverse and -I ((P4)); each element carries its vertex correspondence;
//   * signed canonical forms of oriented K^u-simplices (canonical_key);
//   * word reduction in Gamma ((P2));
//   * the fixed exact vertex geometry (cusp line at height Y = 2, fixed and
//     documented; correctness never depends on it -- invariant I8);
//   * the T^{+-1}/S edge-path templates of [C, alg:tau] step 2, certified
//     exactly (boundary identity over Z) at construction -- two families,
//     the canonical tree-routed one and the legacy BFS one (see
//     path_template()).
class SL2ZTessellationTables {
public:
    // K must be the 7-vertex model (CreateSL2ZFundamentalDomainTriangulation).
    // Builds and certifies all tables; throws on any internal inconsistency.
    explicit SL2ZTessellationTables(const SimplicialComplex& K);

    // (P4): finite transporter set S(s1, s2) (elements g with
    // i(s1) = g * i(s2)); s1, s2 index K's q-simplex list. Empty if the two
    // simplices are not in the same groupoid orbit. S(s, s) is the
    // stabilizer (contains +-I). Deduplicated exact matrices; each S(a,b) is
    // a left coset of the finite stabilizer S(b,b) (order <= 12), which is
    // what terminates the closure.
    const std::vector<Mat22>& transporter(uint32_t q, uint32_t s1, uint32_t s2) const;

    // Vertex correspondences carried alongside transporter(q, s1, s2):
    // perm[j] is the slot (in s1's increasing vertex order) of the image of
    // s2's j-th vertex under the transporter element (slots j > q are padded
    // with the identity). Carried by composition from the relation seed data
    // (invariant I8: combinatorial, never geometric).
    const std::vector<std::array<uint8_t, 3>>& transporter_vertex_perms(
        uint32_t q, uint32_t s1, uint32_t s2) const;

    bool same_groupoid_orbit(uint32_t q, uint32_t s1, uint32_t s2) const;

    // Fixed orbit representative (minimal simplex index) of sigma's orbit.
    uint32_t orbit_representative(uint32_t q, uint32_t sigma) const;

    // Canonical form of the oriented K^u-simplex gamma * i(sigma):
    // candidates gamma * g over g in S(sigma, sigma_hat); gamma_canon is the
    // lexicographically minimal (a,b,c,d) candidate; sign is the parity of
    // the chosen candidate's vertex-slot permutation. Quotients by -I
    // (S(sigma, sigma_hat) is closed under negation).
    KuSimplexKey canonical_key(const Mat22& gamma, uint32_t q, uint32_t sigma) const;

    // (P2): word reduction gamma = +-T^{a1} S T^{a2} S ... T^{a_k} via the
    // Euclidean algorithm (nearest-integer continued fractions) on the first
    // column. Requires det gamma == +1 (throws HeckeOverflow otherwise).
    // Output encoding: an optional leading ('-', 1) denotes the -I factor;
    // then pairs ('T', a) (a != 0; exact exponent) and ('S', 1), in product
    // order. The empty word is the identity. Every intermediate matrix is
    // checked against the certified no-growth bound
    // max_abs <= 2 * max(max_abs(input), 1) of [C, lem:bounds](4)
    // (throws HeckeOverflow on violation).
    std::vector<std::pair<char, int64_t>> word_in_T_S(const Mat22& gamma) const;

    // As word_in_T_S, additionally reporting the maximal intermediate entry
    // magnitude actually attained (for validation of the (P2) bound).
    std::vector<std::pair<char, int64_t>> word_in_T_S_with_bound(
        const Mat22& gamma, int64_t& max_intermediate_out) const;

    // Rebuilds the exact matrix from a word in the encoding above.
    static Mat22 evaluate_word(const std::vector<std::pair<char, int64_t>>& word);

    // Exact fixed vertex geometry (positions only steer tau choices in S4;
    // correctness is certification-based, invariant I8). Cusp line at height
    // Y = 2 (fixed here and documented):
    //   v0 = -1/2 + 2i, v1 = -1/6 + 2i, v2 = 1/6 + 2i, v3 = 1/2 + 2i,
    //   v4 = rho^2 = (-1 + sqrt(3) i)/2, v5 = rho = (1 + sqrt(3) i)/2, v6 = i.
    QuadPoint vertex_position(uint32_t v) const;

    // Path templates ([C, alg:tau] step 2): a 1-chain in K^u from i(w) to
    // letter * i(w2), supported in i(K) u letter * i(K), where letter is
    // 'T' (T), 't' (T^{-1}) or 'S' (S). Certified exactly at construction:
    // the boundary of the returned chain equals
    // canonical_key(letter, 0, w2) - canonical_key(I, 0, w) over Z.
    //
    // Two families are built and certified (both are admissible; only the
    // Canonical one is iota-symmetric):
    //   * Canonical (binding, 2026-09-16): spoke(w) + tree(t(w), letter.t(w2))
    //     - letter.spoke(w2), where the TREE is the Gamma-orbit of the two
    //     edges {v4,v6}, {v5,v6} -- the Bass-Serre tree of PSL_2(Z) with the
    //     elliptic points as vertices -- in which a 1-chain with prescribed
    //     boundary is UNIQUE (a tree has no 1-cycles); t(w) attaches each
    //     horocycle vertex to the tree along one edge of K (the spoke):
    //     t(v0)=v4, t(v1)=v6, t(v2)=v6, t(v3)=v5, t(v)=v for v in {v4,v5,v6}
    //     -- a table that commutes with iota_K (v0<->v3, v1<->v2, v4<->v5)
    //     and with T (t(v3) = T t(v0)). Concatenating such templates along
    //     any word for gamma_a^{-1} gamma_b telescopes to spoke(A) +
    //     tree(t(A), t(B)) - spoke(B), a chain that depends only on the two
    //     K^u endpoints A, B: Gamma-equivariant, iota_X-equivariant and
    //     antisymmetric under A <-> B. That is what makes the canonical tau
    //     of [C, alg:tau] iota-symmetric ([C, prop:iotacommutes](3b)).
    //   * LegacyBfs (the pre-2026-09-16 templates, kept for the record): a
    //     BFS vertex path w -> cross_lhs inside i(K) followed by letter *
    //     (BFS path cross_rhs -> w2), ascending neighbour order -- neither
    //     iota_K-equivariant nor reversal-symmetric.
    enum class PathTemplateFamily { Canonical, LegacyBfs };
    const std::vector<ChainTerm1>& path_template(
        char letter, uint32_t w, uint32_t w2,
        PathTemplateFamily family = PathTemplateFamily::Canonical) const;

    // The spoke attachment t(w) of the Canonical family (see above).
    uint32_t tree_attachment(uint32_t w) const;

    // --- The boundary subcomplex ([C, sec:boundarydatum], (B1)-(B3);
    // hecke-extensions WP-4a, additive) --------------------------------------
    //
    // K_bd is the horocycle path v0 - v1 - v2 - v3 of K, with i(v3) = T i(v0)
    // ([C, rmk:bdrymodel]). Its Gamma-orbit X_bd = Gamma . i(K_bd) is the
    // boundary subcomplex of K^u; its infinity-component X_bd,inf =
    // Gamma_inf . i(K_bd) is the whole horocycle at height Y, a LINE, with
    // Gamma_inf = <T> x <-I> ([C, def:bdrygroup]) acting through T. Every
    // other component is gamma . X_bd,inf over the cusp gamma . infinity
    // ([C, prop:circlemodel](1), (B2)). The data below are what the boundary
    // datum tau_bd (Hecke/HeckeBoundaryDatum) needs: which K-simplices are
    // boundary simplices (degree-generic lists), and integer coordinates on
    // the line so that "the unique 1-chain with a given boundary" is a walk
    // between two positions. All of it is certified at construction by
    // validate_boundary() (throws on failure) -- invariant I8.
    //
    // Line coordinates. The vertices of X_bd,inf are T^k . i(v_i), k in Z,
    // i in {0, 1, 2} (i(v3) = T i(v0) is NOT a fourth orbit); their POSITION
    // is period * k + i with period = 3, the number of Gamma_inf-orbits of
    // vertices. The edge from position P to P + 1 is T^k . i(e_i), i = P mod
    // period, and its ORIENTED boundary is (vertex P+1) - (vertex P) up to
    // the sign carried by boundary_line_edge().coeff (K's base edge {u,v},
    // u < v, has boundary (v) - (u)).

    uint32_t boundary_dim() const { return boundary_dim_; } // 1 on this model
    // Indices (into K's q-simplex list) of the boundary q-simplices; empty
    // for q > boundary_dim().
    const std::vector<uint32_t>& boundary_simplices(uint32_t q) const;
    bool is_boundary_simplex(uint32_t q, uint32_t sigma) const;
    // Line index of a boundary vertex: 0..period for v0..v3 (v3 = T v0 sits
    // at index period). Throws for a non-boundary vertex.
    uint32_t boundary_line_index(uint32_t vertex) const;
    // The translation T with i(v_period) = T . i(v_0) (generator of
    // Gamma_inf modulo -I).
    const Mat22& boundary_translation() const { return boundary_translation_; }
    uint32_t boundary_period() const { return boundary_period_; }
    // Position of gamma . i(vertex) on the line, for gamma = +-T^k in
    // Gamma_inf (throws if gamma is not of that form) and a boundary vertex.
    int64_t boundary_line_position(const Mat22& gamma_inf, uint32_t vertex) const;
    // The vertex at position P, as (T^k, K vertex id), the id taken in the
    // first period (never v3).
    std::pair<Mat22, uint32_t> boundary_line_vertex(int64_t P) const;
    // The edge from position P to P + 1 as coeff * (gamma . i(edge_idx)),
    // whose boundary is exactly (vertex at P+1) - (vertex at P).
    struct LineEdge {
        Mat22 gamma;
        uint32_t edge_idx;
        int32_t coeff;
    };
    LineEdge boundary_line_edge(int64_t P) const;
    // The position on the line whose vertex is nearest (by real part, the
    // vertical projection onto the horocycle) to the real number
    // x = x_num / x_den (x_den > 0); a tie between two equidistant vertices
    // goes to the LARGER real part. T-equivariant: nearest(x + 1) =
    // nearest(x) + period. Exact (__int128, overflow-checked).
    int64_t boundary_line_nearest(int64_t x_num, int64_t x_den) const;
    // Real part of the vertex at position P, as num / den (den > 0).
    void boundary_line_real_part(int64_t P, int64_t& num_out, int64_t& den_out) const;

    // 0 on success: (B1) the line data are a path closing up under T, the
    // boundary simplex lists are face-closed and contain no 2-simplex; (B2)
    // the vertex orbits on the line are exactly {v0,v3}, {v1}, {v2} and the
    // three edges are pairwise inequivalent; (B3) every boundary simplex has
    // stabiliser exactly {+-I}; the exact positions lie on one horocycle and
    // pos(v3) = T . pos(v0); the line coordinates round-trip; nearest() is
    // T-equivariant and fixes every vertex. Called by the constructor.
    uint32_t validate_boundary() const;

    // 0 on success: re-runs the transporter closure and confirms the fixed
    // point is already reached; checks the expected stabilizer orders
    // (|S(v4,v4)| = 6, |S(v5,v5)| = 6, |S(v6,v6)| = 4, order 2 elsewhere),
    // det g == 1 and perm consistency for every stored transporter, the
    // expected orbits, the coset property |S(a,b)| == |S(b,b)|, and
    // re-certifies every path template boundary exactly.
    uint32_t validate_closure() const;

private:
    struct TransporterSet {
        std::vector<Mat22> elems;
        std::vector<std::array<uint8_t, 3>> perms; // parallel to elems
    };
    using SetKey = std::array<uint32_t, 3>; // { q, s1, s2 }
    using SetMap = std::map<SetKey, TransporterSet>;

    // Simplex lists of the model complex (copied; vertex lists sorted).
    std::vector<std::vector<std::vector<uint32_t>>> simplices_;
    // Edge {u,v} (u < v) -> index in simplices_[1].
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edge_index_;
    // Closed transporter tables.
    SetMap sets_;
    // Orbit representative per (q, simplex index).
    std::vector<std::vector<uint32_t>> orbit_rep_;
    // Exact vertex positions (7 entries).
    std::vector<QuadPoint> vertex_positions_;
    // Path templates keyed by (family, letter, w, w2).
    std::map<std::array<uint32_t, 4>, std::vector<ChainTerm1>> path_templates_;

    // Boundary subcomplex data (WP-4a): per-degree boundary simplex lists,
    // the line vertices v0..v3 in line order, the edges between them, T.
    uint32_t boundary_dim_ = 1;
    uint32_t boundary_period_ = 3;
    std::vector<std::vector<uint32_t>> boundary_simplices_; // [q] -> K indices
    std::vector<uint32_t> boundary_line_vertices_;         // v0, v1, v2, v3
    std::vector<uint32_t> boundary_line_edges_;            // e01, e12, e23
    Mat22 boundary_translation_ = Mat22::I();
    void build_boundary_subcomplex();

    // Seeds sets_ from the model relation table, +-I and the elliptic
    // stabilizers, then closes under composition, inverse and -I. Static so
    // validate_closure() can re-run it independently on a fresh copy.
    static SetMap build_closed_sets(
        const std::vector<std::vector<std::vector<uint32_t>>>& simplices);
    // One closure pass; returns true if anything was added.
    static bool closure_pass(SetMap& sets);
    // Insert with dedup + vertex-correspondence consistency check (throws on
    // an inconsistent duplicate). Returns true if newly inserted.
    static bool insert_elem(SetMap& sets, const SetKey& key, const Mat22& g,
                            const std::array<uint8_t, 3>& perm, uint32_t q);

    void compute_orbit_representatives();
    void build_path_templates();
    // Canonical-family template body: spoke(w) + tree(t(w), g . t(w2))
    // - g . spoke(w2), the tree path found in the (unique-path) subgraph
    // spanned by the tree edges of i(K) and g . i(K).
    std::vector<ChainTerm1> canonical_template(char letter, const Mat22& g,
                                               uint32_t w, uint32_t w2) const;
    // Exact boundary certification of one template; 0 on success.
    uint32_t certify_path_template(char letter, uint32_t w, uint32_t w2,
                                   const std::vector<ChainTerm1>& chain) const;

    // BFS (deterministic: ascending neighbours) vertex path a -> b in K.
    std::vector<uint32_t> bfs_vertex_path(uint32_t a, uint32_t b) const;
    // Appends the chain terms of a vertex path, left-translated by gamma.
    void append_path_chain(std::vector<ChainTerm1>& out,
                           const std::vector<uint32_t>& vertex_path,
                           const Mat22& gamma) const;

    std::vector<std::pair<char, int64_t>> word_impl(
        const Mat22& gamma, int64_t& max_intermediate_out) const;
};

} // namespace hecke

#endif // HECKE_SL2Z_TESSELLATION_TABLES_H
