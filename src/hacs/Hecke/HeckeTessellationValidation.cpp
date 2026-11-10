// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S2V -- INDEPENDENT validation of Stage S2 (tessellation bookkeeping).
// See HeckeTessellationValidation.h for the oracle list and independence
// statement.
//
// -----------------------------------------------------------------------
// Hand derivation of the expected tables (independent of the S2 code; done
// from the 8-row relation/generator table of [C, sec:model] /
// SL2ZRelationSeeds_7VertexModel(), which invariant I3 makes the single
// source of truth, and from elementary SL(2,Z) group theory):
//
// Generators (contract, [C, sec:model]): T = [[1,1],[0,1]], S = [[0,-1],[1,0]],
// ST = S*T = [[0,-1],[1,1]], TS = T*S = [[1,-1],[1,0]]. S^2 = (ST)^3 = -I,
// S^4 = (ST)^6 = I. A direct computation confirms TS^3 = -I, TS^6 = I as well
// (TS^2 = [[0,-1],[1,-1]], TS^3 = TS^2*TS = [[-1,0],[0,-1]] = -I).
//
// The seed table has EXACTLY 8 rows and touches only vertices {3,4,5,6} (as
// LHS or RHS of a q=0 relation) and edges {0,4},{3,5},{4,6},{5,6} (as LHS/RHS
// of a q=1 relation); there is NO q=2 (triangle) relation at all. Since every
// simplex's self-transporter (stabilizer) always contains {+-I} "for free"
// (-I acts trivially on H, so it identifies every simplex with itself; P4),
// and closure only ever composes existing transporter-set elements:
//
//   * Vertices 1, 2 never appear in any seed row => Stab(v1) = Stab(v2) = {+-I},
//     order 2; each is its own orbit (singleton).
//   * v0, v3: only relation is T: i(v3) = T*i(v0) (a DIFFERENT-simplex
//     transporter, not a self-relation). Composing S(v3,v0)*S(v0,v3) only
//     ever regenerates {+-I} (T*T^{-1} = I, etc.), so Stab(v0) = Stab(v3) =
//     {+-I}, order 2; orbit {0,3}.
//   * v6: seed row S: i(v6) = S*i(v6) is a genuine self-relation, so
//     <S> = {I,S,S^2=-I,S^3=-S} (order 4, already contains -I) sits inside
//     Stab(v6); no other seed row touches v6, so closure cannot add more:
//     Stab(v6) = <S>, order 4; orbit {6}.
//   * v4: seed row ST: i(v4) = ST*i(v4) gives <ST> = {I,ST,...,(ST)^5}, order
//     6 (already contains -I = (ST)^3), inside Stab(v4). The other row
//     touching v4, S: i(v4) = S*i(v5), only composes with its own inverse to
//     reproduce {+-I} inside Stab(v4) (no new elements: S(v4,v5)*S(v5,v4) =
//     {S,-S}*{S,-S} = {S^2,-S^2} = {-I,I}). So Stab(v4) = <ST>, order 6.
//   * v5: symmetric argument via TS: i(v5) = TS*i(v5) gives Stab(v5) = <TS>,
//     order 6 (consistency check: S^-1 = -S since S^2 = -I, so
//     S*(TS)*S^-1 = S*(TS)*(-S) = -(S*T*S)*S = -S*T*S^2 = -S*T*(-I) = S*T =
//     ST, i.e. conjugating the rho stabilizer by the S-transporter gives
//     exactly the rho^2 stabilizer <ST>, as it must since v4 = S*v5).
//     Orbit {4,5}.
//   * Every edge/triangle: no seed row is a *self*-relation for any edge or
//     triangle (the two edge rows relate DISTINCT edges {0,4}<->{3,5} and
//     {4,6}<->{5,6}; there are zero triangle rows). Composing a
//     cross-transporter with its own inverse again only ever regenerates
//     {+-I} (same argument as v0/v3 above). So every one of the 11 edges and
//     all 5 triangles has self-transporter exactly {+-I}, order 2.
//   * Edge orbits: {(0,4),(3,5)} (via T) and {(4,6),(5,6)} (via S); the
//     remaining 7 edges -- (0,1),(1,2),(1,4),(1,6),(2,3),(2,5),(2,6) -- are
//     untouched by any seed row, hence singleton orbits.
//   * Triangle orbits: all 5 triangles are singleton orbits (no seed rows).
//
// Cross transporter sets (oracle 2): S(sigma1,sigma2) = {g in Gamma :
// i(sigma1) = g*i(sigma2)}; if g0 works then so does g0*h for h in
// Stab(sigma2) = {+-I}, i.e. {g0,-g0}, and (since Stab(sigma2) has order
// exactly 2 by the above) no more. Hence:
//   S(v3,v0)          = {T,-T}   (seed row "T_v3_v0")
//   S({3,5},{0,4})    = {T,-T}   (seed row "T_e(0,4)_e(3,5)")
//   S({4,6},{5,6})    = {S,-S}   (seed row "S_e(5,6)_e(4,6)")
//
// Group-element sanity (oracle 3): every element of every transporter set is
// a product of copies of T, S, ST, TS, -I and their inverses -- all of which
// lie in SL(2,Z) (det = +1); det(-A) = (-1)^2 det(A) = det(A) for 2x2
// matrices, so negation never changes the determinant. Hence det(g) = 1 for
// EVERY g in EVERY transporter set, unconditionally.
// -----------------------------------------------------------------------

#include "HeckeTessellationValidation.h"

#include "SL2ZTessellationTables.h"
#include "ExactInt.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemUtils.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../EquivariantCohomologyOnK.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include <array>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kNotCovered = std::numeric_limits<uint32_t>::max();

// --- small helpers ----------------------------------------------------

bool mat22_set_equal(const std::vector<hecke::Mat22>& got,
                      const std::vector<hecke::Mat22>& expected) {
    if (got.size() != expected.size()) return false;
    for (const auto& e : expected) {
        bool found = false;
        for (const auto& g : got) {
            if (g == e) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

std::string mat22_to_string(const hecke::Mat22& m) {
    std::ostringstream oss;
    oss << "[[" << m.a << "," << m.b << "],[" << m.c << "," << m.d << "]]";
    return oss.str();
}

std::string mat22_set_to_string(const std::vector<hecke::Mat22>& v) {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < v.size(); ++i) {
        oss << mat22_to_string(v[i]);
        if (i + 1 < v.size()) oss << ", ";
    }
    oss << "}";
    return oss.str();
}

// --- Oracle 1: stabilizer orders ---------------------------------------

uint32_t check_stabilizer_orders(const SimplicialComplex& K,
                                  const hecke::SL2ZTessellationTables& tables) {
    uint32_t failures = 0;

    // q = 0: vertices. Expected orders derived by hand above.
    {
        const uint32_t num_v = K.get_num_q_simplices(0);
        if (num_v != 7) {
            std::cout << "HeckeTessellationValidation: expected 7 vertices in the model, got "
                      << num_v << "." << std::endl;
            return failures + 1;
        }
        std::vector<uint32_t> expected(num_v, kNotCovered);
        const std::vector<std::pair<uint32_t, uint32_t>> vertex_orders = {
            {0, 2}, {1, 2}, {2, 2}, {3, 2}, {4, 6}, {5, 6}, {6, 4}
        };
        for (const auto& vo : vertex_orders) {
            const uint32_t idx = find_simplex_idx(K, 0, { vo.first });
            expected[idx] = vo.second;
        }
        for (uint32_t s = 0; s < num_v; ++s) {
            if (expected[s] == kNotCovered) {
                std::cout << "HeckeTessellationValidation: vertex simplex idx " << s
                          << " not covered by the hand-derived expected table." << std::endl;
                ++failures;
                continue;
            }
            const uint32_t got = static_cast<uint32_t>(tables.transporter(0, s, s).size());
            if (got != expected[s]) {
                std::cout << "HeckeTessellationValidation: |Stab(vertex idx " << s
                          << ")| mismatch: expected " << expected[s] << ", got " << got
                          << "." << std::endl;
                ++failures;
            }
        }
    }

    // q = 1: edges. All 11 edges have self-transporter order 2 (see
    // derivation above -- no seed row is a self-relation on any edge).
    {
        const uint32_t num_e = K.get_num_q_simplices(1);
        if (num_e != 11) {
            std::cout << "HeckeTessellationValidation: expected 11 edges in the model, got "
                      << num_e << "." << std::endl;
            return failures + 1;
        }
        const std::vector<std::pair<uint32_t, uint32_t>> edges = {
            {0,1}, {0,4}, {1,2}, {1,4}, {1,6}, {2,3}, {2,5}, {2,6}, {3,5}, {4,6}, {5,6}
        };
        std::vector<uint32_t> expected(num_e, kNotCovered);
        for (const auto& e : edges) {
            const uint32_t idx = find_simplex_idx(K, 1, { e.first, e.second });
            expected[idx] = 2;
        }
        for (uint32_t s = 0; s < num_e; ++s) {
            if (expected[s] == kNotCovered) {
                std::cout << "HeckeTessellationValidation: edge simplex idx " << s
                          << " not covered by the hand-derived expected table." << std::endl;
                ++failures;
                continue;
            }
            const uint32_t got = static_cast<uint32_t>(tables.transporter(1, s, s).size());
            if (got != expected[s]) {
                std::cout << "HeckeTessellationValidation: |Stab(edge idx " << s
                          << ")| mismatch: expected " << expected[s] << ", got " << got
                          << "." << std::endl;
                ++failures;
            }
        }
    }

    // q = 2: triangles. All 5 triangles have self-transporter order 2 (no
    // triangle relation exists in the seed table at all).
    {
        const uint32_t num_t = K.get_num_q_simplices(2);
        if (num_t != 5) {
            std::cout << "HeckeTessellationValidation: expected 5 triangles in the model, got "
                      << num_t << "." << std::endl;
            return failures + 1;
        }
        const std::vector<std::vector<uint32_t>> triangles = {
            {0,1,4}, {1,2,6}, {1,4,6}, {2,3,5}, {2,5,6}
        };
        std::vector<uint32_t> expected(num_t, kNotCovered);
        for (const auto& t : triangles) {
            const uint32_t idx = find_simplex_idx(K, 2, t);
            expected[idx] = 2;
        }
        for (uint32_t s = 0; s < num_t; ++s) {
            if (expected[s] == kNotCovered) {
                std::cout << "HeckeTessellationValidation: triangle simplex idx " << s
                          << " not covered by the hand-derived expected table." << std::endl;
                ++failures;
                continue;
            }
            const uint32_t got = static_cast<uint32_t>(tables.transporter(2, s, s).size());
            if (got != expected[s]) {
                std::cout << "HeckeTessellationValidation: |Stab(triangle idx " << s
                          << ")| mismatch: expected " << expected[s] << ", got " << got
                          << "." << std::endl;
                ++failures;
            }
        }
    }

    return failures;
}

// --- Oracle 2: cross transporter sets -----------------------------------

uint32_t check_cross_transporters(const SimplicialComplex& K,
                                   const hecke::SL2ZTessellationTables& tables) {
    uint32_t failures = 0;
    const hecke::Mat22 T = hecke::Mat22::T();
    const hecke::Mat22 S = hecke::Mat22::S();

    {
        const uint32_t v3 = find_simplex_idx(K, 0, { 3 });
        const uint32_t v0 = find_simplex_idx(K, 0, { 0 });
        const auto& got = tables.transporter(0, v3, v0);
        const std::vector<hecke::Mat22> expected = { T, T.neg() };
        if (!mat22_set_equal(got, expected)) {
            std::cout << "HeckeTessellationValidation: S(v3,v0) mismatch: expected "
                      << mat22_set_to_string(expected) << ", got " << mat22_set_to_string(got)
                      << "." << std::endl;
            ++failures;
        }
    }
    {
        const uint32_t e35 = find_simplex_idx(K, 1, { 3, 5 });
        const uint32_t e04 = find_simplex_idx(K, 1, { 0, 4 });
        const auto& got = tables.transporter(1, e35, e04);
        const std::vector<hecke::Mat22> expected = { T, T.neg() };
        if (!mat22_set_equal(got, expected)) {
            std::cout << "HeckeTessellationValidation: S({3,5},{0,4}) mismatch: expected "
                      << mat22_set_to_string(expected) << ", got " << mat22_set_to_string(got)
                      << "." << std::endl;
            ++failures;
        }
    }
    {
        const uint32_t e46 = find_simplex_idx(K, 1, { 4, 6 });
        const uint32_t e56 = find_simplex_idx(K, 1, { 5, 6 });
        const auto& got = tables.transporter(1, e46, e56);
        const std::vector<hecke::Mat22> expected = { S, S.neg() };
        if (!mat22_set_equal(got, expected)) {
            std::cout << "HeckeTessellationValidation: S({4,6},{5,6}) mismatch: expected "
                      << mat22_set_to_string(expected) << ", got " << mat22_set_to_string(got)
                      << "." << std::endl;
            ++failures;
        }
    }
    return failures;
}

// --- Oracle 3: group-element sanity (det == 1) + hand-derived orbits ----

uint32_t check_group_sanity_and_orbits(const SimplicialComplex& K,
                                        const hecke::SL2ZTessellationTables& tables) {
    uint32_t failures = 0;

    // det == 1 for every g in every transporter set, over ALL (q, s1, s2).
    for (uint32_t q = 0; q <= 2; ++q) {
        const uint32_t num = K.get_num_q_simplices(q);
        for (uint32_t s1 = 0; s1 < num; ++s1) {
            for (uint32_t s2 = 0; s2 < num; ++s2) {
                const auto& set = tables.transporter(q, s1, s2);
                for (const auto& g : set) {
                    if (g.det() != 1) {
                        std::cout << "HeckeTessellationValidation: det(g) != 1 for g = "
                                  << mat22_to_string(g) << " in S(q=" << q << ", " << s1
                                  << ", " << s2 << ")." << std::endl;
                        ++failures;
                    }
                }
            }
        }
    }

    // Hand-derived orbit ids; same_groupoid_orbit(q,s1,s2) must equal
    // (orbit_id[s1] == orbit_id[s2]).
    {
        const uint32_t num_v = K.get_num_q_simplices(0);
        std::vector<uint32_t> orbit_id(num_v, kNotCovered);
        // Orbits: {0,3}, {1}, {2}, {4,5}, {6}
        const std::vector<std::vector<uint32_t>> vertex_orbits = {
            {0, 3}, {1}, {2}, {4, 5}, {6}
        };
        for (uint32_t oid = 0; oid < vertex_orbits.size(); ++oid) {
            for (uint32_t v : vertex_orbits[oid]) {
                orbit_id[find_simplex_idx(K, 0, { v })] = oid;
            }
        }
        for (uint32_t s1 = 0; s1 < num_v; ++s1) {
            for (uint32_t s2 = 0; s2 < num_v; ++s2) {
                const bool expected_same = (orbit_id[s1] == orbit_id[s2]);
                const bool got_same = tables.same_groupoid_orbit(0, s1, s2);
                if (got_same != expected_same) {
                    std::cout << "HeckeTessellationValidation: same_groupoid_orbit(0, " << s1
                              << ", " << s2 << ") mismatch: expected " << expected_same
                              << ", got " << got_same << "." << std::endl;
                    ++failures;
                }
            }
        }
    }
    {
        const uint32_t num_e = K.get_num_q_simplices(1);
        std::vector<uint32_t> orbit_id(num_e, kNotCovered);
        const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> edge_orbits = {
            { {0,4}, {3,5} },
            { {4,6}, {5,6} },
            { {0,1} },
            { {1,2} },
            { {1,4} },
            { {1,6} },
            { {2,3} },
            { {2,5} },
            { {2,6} },
        };
        for (uint32_t oid = 0; oid < edge_orbits.size(); ++oid) {
            for (const auto& e : edge_orbits[oid]) {
                orbit_id[find_simplex_idx(K, 1, { e.first, e.second })] = oid;
            }
        }
        for (uint32_t s = 0; s < num_e; ++s) {
            if (orbit_id[s] == kNotCovered) {
                std::cout << "HeckeTessellationValidation: edge idx " << s
                          << " not covered by the hand-derived orbit table." << std::endl;
                ++failures;
            }
        }
        for (uint32_t s1 = 0; s1 < num_e; ++s1) {
            for (uint32_t s2 = 0; s2 < num_e; ++s2) {
                const bool expected_same = (orbit_id[s1] == orbit_id[s2]);
                const bool got_same = tables.same_groupoid_orbit(1, s1, s2);
                if (got_same != expected_same) {
                    std::cout << "HeckeTessellationValidation: same_groupoid_orbit(1, " << s1
                              << ", " << s2 << ") mismatch: expected " << expected_same
                              << ", got " << got_same << "." << std::endl;
                    ++failures;
                }
            }
        }
    }
    {
        // Triangles: every one is a singleton orbit.
        const uint32_t num_t = K.get_num_q_simplices(2);
        for (uint32_t s1 = 0; s1 < num_t; ++s1) {
            for (uint32_t s2 = 0; s2 < num_t; ++s2) {
                const bool expected_same = (s1 == s2);
                const bool got_same = tables.same_groupoid_orbit(2, s1, s2);
                if (got_same != expected_same) {
                    std::cout << "HeckeTessellationValidation: same_groupoid_orbit(2, " << s1
                              << ", " << s2 << ") mismatch: expected " << expected_same
                              << ", got " << got_same << "." << std::endl;
                    ++failures;
                }
            }
        }
    }

    return failures;
}

// --- Oracle 4: relation-list structure + legacy bit-identity + I3 --------
//
// FIX-negI update: the base model now appends, after the 8 seed relations,
// one central self-relation f(sigma) = rho_n(-I) f(sigma) per simplex of K
// (required by the Psi isomorphism [C, thm:explicit_complex_equiv], since
// -I acts trivially on H; vacuous for even n, forces H^* = 0 for odd n,
// ell != 2). This oracle group therefore checks, hand-derived and
// independently of the S2 code:
//   (a) GenerateSL2ZRelations_7VertexModel and
//       GenerateSL2ZRelationsWithGroupData agree row-by-row on the FULL
//       list (single source of truth -- still true after the fix);
//   (b) the FIRST 8 rows are exactly the original {T,T,T,S,S,S,ST,TS} seed
//       relations (unchanged replica of the pre-fix list, hard-coded below);
//   (c) the list has exactly 8 + N_simplices rows (31 for the 7-vertex K:
//       7 vertices + 11 edges + 5 triangles), the trailing N_simplices being
//       self-relations (s_lhs == s_rhs) with g_r = -I and matrix
//       rho_n(-I) = ComputeSymmetricPowerMatrixFM((-I) mod ell, n), covering
//       EVERY simplex of every dimension exactly once;
//   (d) invariant I3 on every row: rel.M == independently recomputed
//       rho_n(g_r).

// Independent recomputation of rho_n(g) via the FlatMatrix / SM path, from
// the exact Mat22 g and modulus ell -- written directly against invariant
// I3's formula (rho_n(g) = ComputeSymmetricPowerMatrixFM(g mod ell, n)), NOT
// copied from LocalSystemSL2Z.cpp's internal rho_cache.
Matrix independent_rho_n(const hecke::Mat22& g, uint32_t n_sym, uint32_t ell) {
    const std::array<uint32_t, 4> gm = g.reduced_mod(ell);
    const FlatMatrix g_fm(2, 2, ell, std::vector<uint32_t>{ gm[0], gm[1], gm[2], gm[3] });
    return ComputeSymmetricPowerMatrixFM(g_fm, n_sym).to_matrix();
}

uint32_t check_legacy_bit_identity_and_I3(const SimplicialComplex& K,
                                           uint32_t n_sym, uint32_t ell) {
    uint32_t failures = 0;

    // Independent legacy-pipeline matrices (old Matrix-based
    // ComputeSymmetricPowerMatrix, exactly as TestSL2ZCohomologySymmetricPower
    // computes them), needed as arguments to the legacy 7-arg generator.
    Matrix T0(2, 2, ell, { {1, 1}, {0, 1} });
    Matrix S0(2, 2, ell, { {0, ell - 1}, {1, 0} });
    Matrix ST0 = S0 * T0;
    Matrix TS0 = T0 * S0;
    const Matrix M_T = ComputeSymmetricPowerMatrixFM(FlatMatrix(T0), n_sym).to_matrix();
    const Matrix M_S = ComputeSymmetricPowerMatrixFM(FlatMatrix(S0), n_sym).to_matrix();
    const Matrix M_ST = ComputeSymmetricPowerMatrixFM(FlatMatrix(ST0), n_sym).to_matrix();
    const Matrix M_TS = ComputeSymmetricPowerMatrixFM(FlatMatrix(TS0), n_sym).to_matrix();

    std::vector<LocalSystemSimplicialComplex::SimplexRelation> legacy;
    std::vector<SL2ZRelationDatum> with_group_data;
    try {
        legacy = GenerateSL2ZRelations_7VertexModel(K, n_sym, ell, M_T, M_S, M_ST, M_TS);
        with_group_data = GenerateSL2ZRelationsWithGroupData(K, n_sym, ell);
    } catch (const std::exception& e) {
        std::cout << "HeckeTessellationValidation: (n=" << n_sym << ", ell=" << ell
                  << ") relation generation threw: " << e.what() << std::endl;
        return failures + 1;
    }

    // (c) size: 8 seed rows + one -I self-relation per simplex of K.
    size_t num_simplices = 0;
    for (uint32_t q = 0; q <= K.get_dimension(); ++q) {
        num_simplices += K.get_num_q_simplices(q);
    }
    const size_t expected_size = 8 + num_simplices; // = 31 for the 7-vertex K
    if (legacy.size() != expected_size || with_group_data.size() != expected_size) {
        std::cout << "HeckeTessellationValidation: (n=" << n_sym << ", ell=" << ell
                  << ") expected " << expected_size << " relations, got legacy="
                  << legacy.size() << " with_group_data=" << with_group_data.size()
                  << "." << std::endl;
        return failures + 1;
    }

    // (a) row-by-row agreement of the two generators, over the FULL list.
    for (size_t r = 0; r < expected_size; ++r) {
        const auto& L = legacy[r];
        const auto& G = with_group_data[r].rel;
        if (G.q_dim != L.q_dim || G.s_lhs_idx_in_Kq != L.s_lhs_idx_in_Kq ||
            G.s_rhs_idx_in_Kq != L.s_rhs_idx_in_Kq || G.tag != L.tag || !(G.M == L.M)) {
            std::cout << "HeckeTessellationValidation: row " << r
                      << " mismatch between GenerateSL2ZRelationsWithGroupData and "
                         "GenerateSL2ZRelations_7VertexModel." << std::endl;
            ++failures;
        }
        // (d) I3 on every row: rel.M == independently recomputed rho_n(g_r).
        const Matrix rho = independent_rho_n(with_group_data[r].g_r, n_sym, ell);
        if (!(G.M == rho)) {
            std::cout << "HeckeTessellationValidation: row " << r
                      << " rel.M != independently recomputed rho_n(g_r) (invariant I3 "
                         "violation) for (n=" << n_sym << ", ell=" << ell << ")." << std::endl;
            ++failures;
        }
    }

    // (b) the first 8 rows are EXACTLY the original pre-fix seed relations
    // (unchanged replica, hard-coded independently of the S2 seed table).
    struct ExpectedSeedRow {
        uint32_t q;
        std::vector<uint32_t> lhs;
        std::vector<uint32_t> rhs;
        const char* tag;
        const Matrix* M;
    };
    const ExpectedSeedRow expected_rows[8] = {
        { 0, { 3 },    { 0 },    "T_v3_v0",          &M_T  },
        { 0, { 5 },    { 4 },    "T_v5_v4",          &M_T  },
        { 1, { 3, 5 }, { 0, 4 }, "T_e(0,4)_e(3,5)",  &M_T  },
        { 0, { 4 },    { 5 },    "S_v4_v5",          &M_S  },
        { 0, { 6 },    { 6 },    "S_v6_stabilizer",  &M_S  },
        { 1, { 4, 6 }, { 5, 6 }, "S_e(5,6)_e(4,6)",  &M_S  },
        { 0, { 4 },    { 4 },    "ST_v4_stabilizer", &M_ST },
        { 0, { 5 },    { 5 },    "TS_v5_stabilizer", &M_TS },
    };
    const hecke::Mat22 expected_g[8] = {
        hecke::Mat22::T(), hecke::Mat22::T(), hecke::Mat22::T(),
        hecke::Mat22::S(), hecke::Mat22::S(), hecke::Mat22::S(),
        hecke::Mat22::ST(), hecke::Mat22::TS(),
    };
    for (size_t r = 0; r < 8; ++r) {
        const auto& G = with_group_data[r].rel;
        const auto& E = expected_rows[r];
        if (G.q_dim != E.q || G.s_lhs_idx_in_Kq != find_simplex_idx(K, E.q, E.lhs) ||
            G.s_rhs_idx_in_Kq != find_simplex_idx(K, E.q, E.rhs) || G.tag != E.tag ||
            !(G.M == *E.M)) {
            std::cout << "HeckeTessellationValidation: seed row " << r
                      << " differs from the original pre-fix relation replica ('"
                      << E.tag << "')." << std::endl;
            ++failures;
        }
        if (!(with_group_data[r].g_r == expected_g[r])) {
            std::cout << "HeckeTessellationValidation: row " << r
                      << " g_r != expected generator dictionary entry." << std::endl;
            ++failures;
        }
    }

    // (c) trailing rows: -I self-relations covering every simplex exactly
    // once, with matrix rho_n(-I) recomputed independently.
    const hecke::Mat22 neg_I = hecke::Mat22::I().neg();
    const Matrix rho_neg_I = independent_rho_n(neg_I, n_sym, ell);
    std::vector<std::vector<bool>> covered(K.get_dimension() + 1);
    for (uint32_t q = 0; q <= K.get_dimension(); ++q) {
        covered[q].assign(K.get_num_q_simplices(q), false);
    }
    for (size_t r = 8; r < expected_size; ++r) {
        const auto& G = with_group_data[r].rel;
        if (!(with_group_data[r].g_r == neg_I)) {
            std::cout << "HeckeTessellationValidation: trailing row " << r
                      << " g_r != -I." << std::endl;
            ++failures;
            continue;
        }
        if (G.s_lhs_idx_in_Kq != G.s_rhs_idx_in_Kq) {
            std::cout << "HeckeTessellationValidation: trailing row " << r
                      << " is not a self-relation (s_lhs != s_rhs)." << std::endl;
            ++failures;
            continue;
        }
        if (!(G.M == rho_neg_I)) {
            std::cout << "HeckeTessellationValidation: trailing row " << r
                      << " matrix != rho_n(-I) for (n=" << n_sym << ", ell=" << ell
                      << ")." << std::endl;
            ++failures;
        }
        if (G.q_dim > K.get_dimension() ||
            G.s_lhs_idx_in_Kq >= K.get_num_q_simplices(G.q_dim)) {
            std::cout << "HeckeTessellationValidation: trailing row " << r
                      << " references an out-of-range simplex." << std::endl;
            ++failures;
            continue;
        }
        if (covered[G.q_dim][G.s_lhs_idx_in_Kq]) {
            std::cout << "HeckeTessellationValidation: trailing row " << r
                      << " covers simplex (q=" << G.q_dim << ", idx=" << G.s_lhs_idx_in_Kq
                      << ") twice." << std::endl;
            ++failures;
        }
        covered[G.q_dim][G.s_lhs_idx_in_Kq] = true;
    }
    for (uint32_t q = 0; q <= K.get_dimension(); ++q) {
        for (uint32_t s = 0; s < K.get_num_q_simplices(q); ++s) {
            if (!covered[q][s]) {
                std::cout << "HeckeTessellationValidation: simplex (q=" << q << ", idx="
                          << s << ") has no -I self-relation." << std::endl;
                ++failures;
            }
        }
    }

    return failures;
}

// --- Oracle 5: high-level cohomology unchanged ---------------------------

uint32_t check_cohomology_dimensions(const SimplicialComplex& K) {
    uint32_t failures = 0;
    const uint32_t n_sym = 10;
    const uint32_t ell = 5;
    const uint32_t d_V = n_sym + 1;

    // Method (b): independently reconstruct the equivariant cohomology from
    // the S2 relation data via the public EquivariantCohomologyOnK API (hard
    // assertion on exact dimensions).
    try {
        const auto data = GenerateSL2ZRelationsWithGroupData(K, n_sym, ell);
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        rels.reserve(data.size());
        for (const auto& d : data) rels.push_back(d.rel);

        LocalSystemSimplicialComplex lssc(K, rels, d_V, ell);
        EquivariantCohomologyOnK eq(&lssc);
        eq.compute_cohomology();
        const auto& dims = eq.get_H_eq_dimensions();

        if (dims.size() < 3) {
            std::cout << "HeckeTessellationValidation: expected H_eq_dimensions of size >= 3, "
                         "got " << dims.size() << "." << std::endl;
            ++failures;
        } else {
            // Eichler-Shimura: dim H^1(Sym^10) = 2*dim S_12 + 1 = 2*1 + 1 = 3
            // (dim S_12(SL(2,Z)) = 1, spanned by the cusp form Delta);
            // dim H^0 = 0 (Sym^10 has no SL(2,Z)-invariants for n > 0, SL(2,Z)
            // being Zariski-dense in SL(2)); dim H^2 = 0 for ell >= 5 (O14).
            const uint32_t expected_h0 = 0, expected_h1 = 3, expected_h2 = 0;
            if (dims[0] != expected_h0) {
                std::cout << "HeckeTessellationValidation: dim H^0(Sym^10,F_5) expected "
                          << expected_h0 << ", got " << dims[0] << "." << std::endl;
                ++failures;
            }
            if (dims[1] != expected_h1) {
                std::cout << "HeckeTessellationValidation: dim H^1(Sym^10,F_5) expected "
                          << expected_h1 << ", got " << dims[1] << "." << std::endl;
                ++failures;
            }
            if (dims[2] != expected_h2) {
                std::cout << "HeckeTessellationValidation: dim H^2(Sym^10,F_5) expected "
                          << expected_h2 << ", got " << dims[2] << "." << std::endl;
                ++failures;
            }
        }
    } catch (const std::exception& e) {
        std::cout << "HeckeTessellationValidation: EquivariantCohomologyOnK reconstruction "
                     "threw: " << e.what() << std::endl;
        ++failures;
    }

    // Method (a), secondary smoke check: the existing high-level driver
    // functions must run without throwing (they are void / stdout-only).
    try {
        TestSL2ZCohomologySymmetricPower(n_sym, ell);
    } catch (const std::exception& e) {
        std::cout << "HeckeTessellationValidation: TestSL2ZCohomologySymmetricPower(10,5) "
                     "threw: " << e.what() << std::endl;
        ++failures;
    }
    try {
        TestSL2Z_To_S1Cusp_RestrictionMorphism(n_sym, ell);
    } catch (const std::exception& e) {
        std::cout << "HeckeTessellationValidation: TestSL2Z_To_S1Cusp_RestrictionMorphism(10,5) "
                     "threw: " << e.what() << std::endl;
        ++failures;
    }

    return failures;
}

} // namespace

uint32_t HeckeTessellationValidation() {
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables tables(K);

        uint32_t failures = 0;

        failures += check_stabilizer_orders(K, tables);
        failures += check_cross_transporters(K, tables);
        failures += check_group_sanity_and_orbits(K, tables);

        const uint32_t legacy_cases[3][2] = { { 0, 5 }, { 2, 5 }, { 10, 691 } };
        for (const auto& c : legacy_cases) {
            failures += check_legacy_bit_identity_and_I3(K, c[0], c[1]);
        }

        failures += check_cohomology_dimensions(K);

        if (failures != 0) {
            std::cout << "HeckeTessellationValidation: " << failures
                      << " discrepancy(ies) found (see above)." << std::endl;
            return failures;
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cout << "HeckeTessellationValidation: exception: " << e.what() << std::endl;
        return 100;
    }
}
