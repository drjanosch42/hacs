// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "LocalSystemSL2Z.h"
#include "LocalSystemUtils.h"
#include "../ModArithmetic/EngineCheck.h"
#include "SimplicialCohomology.h" // For SimplicialComplex
#include "LocalSystemSimplicialComplex.h"
#include "EquivariantCohomologyOnK.h"
#include "EquivariantCochainComplexMorphismOnK.h"
#include "../ModArithmetic/FlatMatrix.h" // ComputeSymmetricPowerMatrixFM (invariant I3)
#include "LocalSystemPresets.h" // May not be used directly here, but good to have context
#include "Hecke/HeckeSettings.h" // hecke::RequireFieldCoefficientRing (slot-12 WP-1, carry F-1)

#include <array>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <type_traits>
#include <set>
#include <sstream>
#include <memory> // For std::make_unique or managing SCM

SimplicialComplex CreateSL2ZFundamentalDomainTriangulation() {
    std::vector<std::vector<uint32_t>> K0_vertices_list;
    for (uint32_t i = 0; i <= 6; ++i) {
        K0_vertices_list.push_back({ i });
    }

    std::set<std::vector<uint32_t>> K1_edges_set;
    std::vector<std::vector<uint32_t>> K2_faces_list;

    // Define 2-simplices (triangles)
    // These implicitly define the necessary 1-simplices (edges)
    K2_faces_list = {
        // Top strip (near cusp i*infinity), from left to right
        {0,1,4}, {1,2,6}, {2,3,5},       
        // 4 = rho^2 and 5=rho lie on the vertical lines connecting the top strip with the bottom "circle"
        // and are also connected via i=6, left to right
        {1,4,6}, {2,5,6}
    };

    // Collect all edges from the faces
    for (const auto& face : K2_faces_list) {
        K1_edges_set.insert(make_edge(face[0], face[1]));
        K1_edges_set.insert(make_edge(face[1], face[2]));
        K1_edges_set.insert(make_edge(face[0], face[2]));
    }

    std::vector<std::vector<uint32_t>> K1_edges_list(K1_edges_set.begin(), K1_edges_set.end());
    // Ensure edges are sorted internally (make_edge does this)
    // and the list of edges is sorted (std::set does this).

    // Sort faces internally for canonical representation
    for (auto& face : K2_faces_list) {
        std::sort(face.begin(), face.end());
    }
    // Sort the list of faces
    std::sort(K2_faces_list.begin(), K2_faces_list.end());

    std::vector<std::vector<std::vector<uint32_t>>> simplices_data = {
        K0_vertices_list,
        K1_edges_list,
        K2_faces_list
    };

    return SimplicialComplex(simplices_data, "K_SL2Z_Fund_Domain");
}

// ===========================================================================
// Stage S2 -- Model group data (invariant I3; [C, sec:model] contract box).
// ===========================================================================

// The 8 rows of the relation/generator table of [C, sec:model], in the exact
// legacy order (bit-identical q_dim / vertex lists / tags). K_SL2Z_Fund_Domain
// vertices: 0..3 on the cusp line Im z = Y (Re from -1/2 at v0 to +1/2 at v3),
// 4 = rho^2 (on Re z = -1/2), 5 = rho (on Re z = +1/2), 6 = i.
// Vertex correspondences (image_of_rhs): T: 0->3, 4->5; S: 5->4, 6->6;
// ST: 4->4; TS: 5->5. These are combinatorial model data (single source of
// truth alongside g_r), consumed by hecke::SL2ZTessellationTables to carry
// orientation signs by composition (invariant I8: never geometric).
const std::vector<SL2ZRelationSeed>& SL2ZRelationSeeds_7VertexModel() {
    static const std::vector<SL2ZRelationSeed> seeds = {
        // Vertex identifications for T (rows 1-2): i({3}) = T*i({0}), i({5}) = T*i({4})
        { 0, { 3 },    { 0 },    hecke::Mat22::T(),  "T_v3_v0",          { 3 } },
        { 0, { 5 },    { 4 },    hecke::Mat22::T(),  "T_v5_v4",          { 5 } },
        // Edge identification for T (row 3): i({3,5}) = T*i({0,4})
        { 1, { 3, 5 }, { 0, 4 }, hecke::Mat22::T(),  "T_e(0,4)_e(3,5)",  { 3, 5 } },
        // Vertex identifications for S (rows 4-5): i({4}) = S*i({5}), i({6}) = S*i({6})
        { 0, { 4 },    { 5 },    hecke::Mat22::S(),  "S_v4_v5",          { 4 } },
        { 0, { 6 },    { 6 },    hecke::Mat22::S(),  "S_v6_stabilizer",  { 6 } },
        // Edge identification for S (row 6): i({4,6}) = S*i({5,6})
        { 1, { 4, 6 }, { 5, 6 }, hecke::Mat22::S(),  "S_e(5,6)_e(4,6)",  { 4, 6 } },
        // Stabilizer relations at the elliptic points (rows 7-8)
        { 0, { 4 },    { 4 },    hecke::Mat22::ST(), "ST_v4_stabilizer", { 4 } },
        { 0, { 5 },    { 5 },    hecke::Mat22::TS(), "TS_v5_stabilizer", { 5 } },
    };
    return seeds;
}

std::vector<SL2ZRelationDatum> GenerateSL2ZRelationsWithGroupData(
    const SimplicialComplex& K, uint32_t n_sym, uint32_t ell)
{
    std::vector<SL2ZRelationDatum> data;
    try {
        // Cache rho_n(g) per distinct generator (T, S, ST, TS each occur
        // several times); derived per invariant I3 as
        // ComputeSymmetricPowerMatrixFM(g_r reduced mod ell, n_sym).
        std::vector<std::pair<hecke::Mat22, FlatMatrix>> rho_cache;
        for (const auto& seed : SL2ZRelationSeeds_7VertexModel()) {
            const FlatMatrix* M_cached = nullptr;
            for (const auto& entry : rho_cache) {
                if (entry.first == seed.g_r) { M_cached = &entry.second; break; }
            }
            if (M_cached == nullptr) {
                const std::array<uint32_t, 4> g_mod = seed.g_r.reduced_mod(ell);
                const FlatMatrix g_fm(2, 2, ell,
                    std::vector<uint32_t>{ g_mod[0], g_mod[1], g_mod[2], g_mod[3] });
                // (matrix-engine Phase 3) rho_n(g_r) is stored as the
                // FlatMatrix ComputeSymmetricPowerMatrixFM returns -- the
                // `.to_matrix()` copy is gone (SimplexRelation::M is a
                // FlatMatrix now).
                rho_cache.emplace_back(
                    seed.g_r, ComputeSymmetricPowerMatrixFM(g_fm, n_sym));
                M_cached = &rho_cache.back().second;
            }
            data.push_back(SL2ZRelationDatum{
                LocalSystemSimplicialComplex::SimplexRelation(
                    seed.q_dim,
                    find_simplex_idx(K, seed.q_dim, seed.lhs_vertices),
                    find_simplex_idx(K, seed.q_dim, seed.rhs_vertices),
                    *M_cached,
                    seed.tag),
                seed.g_r });
        }

        // -------------------------------------------------------------------
        // FIX-negI: central -I equivariance on EVERY simplex.
        //
        // -I is central in SL(2,Z) and acts trivially on the upper half
        // plane H, so in the universal-cover complex K^u we have
        // (-I) * i(sigma) = i(sigma) for EVERY oriented simplex sigma of K.
        // The Psi isomorphism
        //     H^*(K/R; L) = Hom_{Z[Gamma]}(C_*(K^u), V)
        // (Theorem "explicit_complex_equiv" in docs/latex/hecke_action.tex /
        // monodromy_v5.tex) therefore REQUIRES the equivariance constraint
        //     f(sigma) = rho_n(-I) * f(sigma)
        // for every simplex sigma of K -- not only at the elliptic vertices
        // v4/v5/v6, where the stabilizer seed relations above already contain
        // -I implicitly (S^2 = (ST)^3 = (TS)^3 = -I).
        //
        // For EVEN n this constraint is vacuous (rho_n(-I) = +I), so
        // appending these relations is a strict no-op. For ODD n and
        // ell != 2, rho_n(-I) = -I forces 2*f(sigma) = 0, i.e. f(sigma) = 0
        // and H^* = 0 -- the correct answer (Hecke oracle O2), where the
        // pre-fix 8-relation model wrongly gave dim H^1 = n + 1.
        //
        // The 8 seed relations above keep their original order; the -I
        // self-relations are appended, one per simplex of K (all dimensions
        // q = 0..dim K), each carrying g_r = -I with M derived as rho_n(-I)
        // per invariant I3. At the elliptic vertices this is redundant but
        // consistent; it is added uniformly for every simplex.
        // -------------------------------------------------------------------
        const hecke::Mat22 neg_I = hecke::Mat22::I().neg();
        const std::array<uint32_t, 4> nI_mod = neg_I.reduced_mod(ell);
        const FlatMatrix nI_fm(2, 2, ell,
            std::vector<uint32_t>{ nI_mod[0], nI_mod[1], nI_mod[2], nI_mod[3] });
        const FlatMatrix rho_neg_I = ComputeSymmetricPowerMatrixFM(nI_fm, n_sym);
        const auto& simplices_by_dim = K.get_simplices();
        for (uint32_t q = 0; q < simplices_by_dim.size(); ++q) {
            const uint32_t num_q = static_cast<uint32_t>(simplices_by_dim[q].size());
            for (uint32_t s = 0; s < num_q; ++s) {
                std::ostringstream tag;
                tag << "negI_q" << q << "_" << s;
                data.push_back(SL2ZRelationDatum{
                    LocalSystemSimplicialComplex::SimplexRelation(
                        q, s, s, rho_neg_I, tag.str()),
                    neg_I });
            }
        }
    }
    catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "ERROR in GenerateSL2ZRelationsWithGroupData for Sym^" << n_sym
            << "(F_" << ell << "): " << e.what();
        throw std::runtime_error(oss.str());
    }
    return data;
}

// hecke-extensions slot 6, WP-1 -- the contragredient relation list
// ([C, def:contragredient]); see the header.
std::vector<SL2ZRelationDatum> GenerateSL2ZContragredientRelationsWithGroupData(
    const SimplicialComplex& K, uint32_t n_sym, uint32_t ell)
{
    // The Sym^n list is the single source of truth for WHICH relations exist
    // (rows, order, simplices, tags, g_r); only the fibre matrix is replaced.
    std::vector<SL2ZRelationDatum> data = GenerateSL2ZRelationsWithGroupData(K, n_sym, ell);
    std::vector<std::pair<hecke::Mat22, FlatMatrix>> dual_cache;
    for (SL2ZRelationDatum& datum : data) {
        const FlatMatrix* M_cached = nullptr;
        for (const auto& entry : dual_cache) {
            if (entry.first == datum.g_r) { M_cached = &entry.second; break; }
        }
        if (M_cached == nullptr) {
            const std::array<uint32_t, 4> g_mod = datum.g_r.reduced_mod(ell);
            const FlatMatrix g_fm(2, 2, ell, std::vector<uint32_t>{ g_mod[0], g_mod[1], g_mod[2], g_mod[3] });
            dual_cache.emplace_back(datum.g_r, ComputeContragredientPowerMatrixFM(g_fm, n_sym));
            M_cached = &dual_cache.back().second;
        }
        ENGINE_CHECK_MSG(M_cached->get_rows() == datum.rel.M.get_rows() && M_cached->get_cols() == datum.rel.M.get_cols(),
                         "GenerateSL2ZContragredientRelationsWithGroupData: fibre shape differs from Sym^n");
        datum.rel = LocalSystemSimplicialComplex::SimplexRelation(
            datum.rel.q_dim, datum.rel.s_lhs_idx_in_Kq, datum.rel.s_rhs_idx_in_Kq, *M_cached, datum.rel.tag);
    }
    return data;
}

// hecke-extensions slot 11, WP-5 -- the quotient relation list; see the header.
std::vector<SL2ZRelationDatum> GenerateSL2ZQuotientLRelationsWithGroupData(
    const SimplicialComplex& K, uint32_t n_sym, uint32_t ell)
{
    // (slot-12 WP-1, carry F-1; [E] sec:ring (b1)) a purely F_ell routine with
    // no notion of a ring: DicksonThetaCell already refuses FIRST on every
    // production path (HeckeDicksonTheta.cpp), so this never fires there --
    // it closes the member for a caller that reaches this relation list
    // directly, naming itself rather than staying silent.
    hecke::RequireFieldCoefficientRing("GenerateSL2ZQuotientLRelationsWithGroupData");
    // The Sym^n list is the single source of truth for WHICH relations exist;
    // only the fibre matrix is replaced (as for the contragredient list).
    std::vector<SL2ZRelationDatum> data = GenerateSL2ZRelationsWithGroupData(K, n_sym, ell);
    std::vector<std::pair<hecke::Mat22, FlatMatrix>> q_cache;
    for (SL2ZRelationDatum& datum : data) {
        const FlatMatrix* M_cached = nullptr;
        for (const auto& entry : q_cache) {
            if (entry.first == datum.g_r) { M_cached = &entry.second; break; }
        }
        if (M_cached == nullptr) {
            const std::array<uint32_t, 4> g_mod = datum.g_r.reduced_mod(ell);
            const FlatMatrix g_fm(2, 2, ell, std::vector<uint32_t>{ g_mod[0], g_mod[1], g_mod[2], g_mod[3] });
            q_cache.emplace_back(datum.g_r, ComputeQuotientLMatrixFM(g_fm, n_sym));
            M_cached = &q_cache.back().second;
        }
        ENGINE_CHECK_MSG(M_cached->get_rows() == ell + 1 && M_cached->get_cols() == ell + 1,
                         "GenerateSL2ZQuotientLRelationsWithGroupData: fibre shape is not (ell+1) x (ell+1)");
        datum.rel = LocalSystemSimplicialComplex::SimplexRelation(
            datum.rel.q_dim, datum.rel.s_lhs_idx_in_Kq, datum.rel.s_rhs_idx_in_Kq, *M_cached, datum.rel.tag);
    }
    return data;
}

// Stage S13: element-width-generic reduction of a Mat22 mod ell.
template<class Elt>
static std::array<Elt, 4> mat22_reduced_mod_elt(const hecke::Mat22& g, Elt ell) {
    if constexpr (std::is_same_v<Elt, uint32_t>) {
        const std::array<uint32_t, 4> a = g.reduced_mod(static_cast<uint32_t>(ell));
        return { a[0], a[1], a[2], a[3] };
    } else {
        const std::array<uint64_t, 4> a = g.reduced_mod64(static_cast<uint64_t>(ell));
        return { a[0], a[1], a[2], a[3] };
    }
}

template<class Elt>
std::vector<typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation>
GenerateSL2ZRelationsElt(const SimplicialComplex& K, uint32_t n_sym, Elt ell)
{
    using Rel = typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation;
    std::vector<Rel> rels;
    try {
        // Seeds: M = rho_n(g_r) over FlatMatrixT<Elt> (invariant I3), cached per
        // distinct generator.
        std::vector<std::pair<hecke::Mat22, FlatMatrixT<Elt>>> rho_cache;
        for (const auto& seed : SL2ZRelationSeeds_7VertexModel()) {
            const FlatMatrixT<Elt>* M_cached = nullptr;
            for (const auto& entry : rho_cache) {
                if (entry.first == seed.g_r) { M_cached = &entry.second; break; }
            }
            if (M_cached == nullptr) {
                const std::array<Elt, 4> g_mod = mat22_reduced_mod_elt<Elt>(seed.g_r, ell);
                const FlatMatrixT<Elt> g_fm(2, 2, ell,
                    std::vector<Elt>{ g_mod[0], g_mod[1], g_mod[2], g_mod[3] });
                // (matrix-engine Phase 3) rho_n(g_r) is stored as the
                // FlatMatrix ComputeSymmetricPowerMatrixFM returns -- the
                // `.to_matrix()` copy is gone (SimplexRelation::M is a
                // FlatMatrix now).
                rho_cache.emplace_back(
                    seed.g_r, ComputeSymmetricPowerMatrixFM(g_fm, n_sym));
                M_cached = &rho_cache.back().second;
            }
            rels.push_back(Rel(
                seed.q_dim,
                find_simplex_idx(K, seed.q_dim, seed.lhs_vertices),
                find_simplex_idx(K, seed.q_dim, seed.rhs_vertices),
                *M_cached,
                seed.tag));
        }

        // FIX-negI: central -I equivariance on every simplex.
        const hecke::Mat22 neg_I = hecke::Mat22::I().neg();
        const std::array<Elt, 4> nI_mod = mat22_reduced_mod_elt<Elt>(neg_I, ell);
        const FlatMatrixT<Elt> nI_fm(2, 2, ell,
            std::vector<Elt>{ nI_mod[0], nI_mod[1], nI_mod[2], nI_mod[3] });
        const FlatMatrixT<Elt> rho_neg_I = ComputeSymmetricPowerMatrixFM(nI_fm, n_sym);
        const auto& simplices_by_dim = K.get_simplices();
        for (uint32_t q = 0; q < simplices_by_dim.size(); ++q) {
            const uint32_t num_q = static_cast<uint32_t>(simplices_by_dim[q].size());
            for (uint32_t s = 0; s < num_q; ++s) {
                std::ostringstream tag;
                tag << "negI_q" << q << "_" << s;
                rels.push_back(Rel(q, s, s, rho_neg_I, tag.str()));
            }
        }
    }
    catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "ERROR in GenerateSL2ZRelationsElt for Sym^" << n_sym
            << "(F_" << ell << "): " << e.what();
        throw std::runtime_error(oss.str());
    }
    return rels;
}

template std::vector<typename LocalSystemSimplicialComplexT<uint32_t>::SimplexRelation>
GenerateSL2ZRelationsElt<uint32_t>(const SimplicialComplex&, uint32_t, uint32_t);
template std::vector<typename LocalSystemSimplicialComplexT<uint64_t>::SimplexRelation>
GenerateSL2ZRelationsElt<uint64_t>(const SimplicialComplex&, uint32_t, uint64_t);

// ---------------------------------------------------------------------------
// hecke-extensions WP-4b -- the circle model as a library call (see the header).
// ---------------------------------------------------------------------------

SimplicialComplex CreateSL2ZBoundaryPathTriangulation() {
    // The demo's K_S1_Path_Repr_For_SL2Z data, verbatim: the boundary path
    // v0-v1-v2-v3 with K's vertex labels and the boundary_simplices(q) order.
    const std::vector<std::vector<std::vector<uint32_t>>> simplices = {
        { { 0 }, { 1 }, { 2 }, { 3 } },
        { make_edge(0, 1), make_edge(1, 2), make_edge(2, 3) }
    };
    return SimplicialComplex(simplices, "K_SL2Z_boundary_path_v0v1v2v3");
}

template<class Elt>
std::vector<typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation>
GenerateSL2ZBoundaryRelationsElt(const SimplicialComplex& K_bd, uint32_t n_sym, Elt ell)
{
    using Rel = typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation;
    std::vector<Rel> rels;
    try {
        // The cusp relation f(v3) = rho_n(T) f(v0), g_r = T ([C, rmk:bdrymodel]:
        // i(v3) = T . i(v0)); rho_n(T) derived, never entered by hand (I3).
        const std::array<Elt, 4> T_mod = mat22_reduced_mod_elt<Elt>(hecke::Mat22::T(), ell);
        const FlatMatrixT<Elt> T_fm(2, 2, ell, std::vector<Elt>{ T_mod[0], T_mod[1], T_mod[2], T_mod[3] });
        const FlatMatrixT<Elt> rho_T = ComputeSymmetricPowerMatrixFM(T_fm, n_sym);
        rels.push_back(Rel(0, find_simplex_idx(K_bd, 0, { 3 }), find_simplex_idx(K_bd, 0, { 0 }), rho_T, "cusp_T"));

        // FIX-negI on the boundary: central -I equivariance on every simplex.
        const hecke::Mat22 neg_I = hecke::Mat22::I().neg();
        const std::array<Elt, 4> nI_mod = mat22_reduced_mod_elt<Elt>(neg_I, ell);
        const FlatMatrixT<Elt> nI_fm(2, 2, ell, std::vector<Elt>{ nI_mod[0], nI_mod[1], nI_mod[2], nI_mod[3] });
        const FlatMatrixT<Elt> rho_neg_I = ComputeSymmetricPowerMatrixFM(nI_fm, n_sym);
        const auto& simplices_by_dim = K_bd.get_simplices();
        for (uint32_t q = 0; q < simplices_by_dim.size(); ++q) {
            const uint32_t num_q = static_cast<uint32_t>(simplices_by_dim[q].size());
            for (uint32_t s = 0; s < num_q; ++s) {
                std::ostringstream tag;
                tag << "negI_bd_q" << q << "_" << s;
                rels.push_back(Rel(q, s, s, rho_neg_I, tag.str()));
            }
        }
    }
    catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "ERROR in GenerateSL2ZBoundaryRelationsElt for Sym^" << n_sym
            << "(F_" << ell << "): " << e.what();
        throw std::runtime_error(oss.str());
    }
    return rels;
}

template std::vector<typename LocalSystemSimplicialComplexT<uint32_t>::SimplexRelation>
GenerateSL2ZBoundaryRelationsElt<uint32_t>(const SimplicialComplex&, uint32_t, uint32_t);
template std::vector<typename LocalSystemSimplicialComplexT<uint64_t>::SimplexRelation>
GenerateSL2ZBoundaryRelationsElt<uint64_t>(const SimplicialComplex&, uint32_t, uint64_t);

// Legacy generator for the 7-vertex model. Since Stage S2 this is a thin
// wrapper around GenerateSL2ZRelationsWithGroupData (single source of truth,
// invariant I3): it delegates and strips g_r, producing bit-identical output
// (same q_dim, s_lhs, s_rhs, tags, order and matrices) to the pre-S2
// implementation. The caller-supplied precomputed matrices are verified
// against the derived rho_n(g_r) so a hand-entered matrix can never silently
// diverge from the group datum ([C, sec:model] contract box).
std::vector<LocalSystemSimplicialComplex::SimplexRelation>
GenerateSL2ZRelations_7VertexModel(
    const SimplicialComplex& K_sl2z, // The K complex (assumed to be the 7-vertex model)
    uint32_t n_sym,
    uint32_t p_mod,
    const FlatMatrix& M_T, // Precomputed Sym^n(T0)
    const FlatMatrix& M_S, // Precomputed Sym^n(S0)
    const FlatMatrix& M_ST, // Precomputed Sym^n(ST0)
    const FlatMatrix& M_TS  // Precomputed Sym^n(TS0)
) {
    std::vector<SL2ZRelationDatum> data =
        GenerateSL2ZRelationsWithGroupData(K_sl2z, n_sym, p_mod);

    // Invariant I3 guard: the legacy caller-supplied matrices must equal the
    // derived rho_n(g_r), in the fixed row order of [C, sec:model]
    // (rows 1-3: T, rows 4-6: S, row 7: ST, row 8: TS).
    //
    // FIX-negI: since the central -I fix, GenerateSL2ZRelationsWithGroupData
    // appends one -I self-relation per simplex of K after the 8 seed
    // relations (see the comment there); this wrapper passes them through
    // unchanged, so it now returns 8 + N_simplices relations (31 for the
    // 7-vertex model). The old 8-relation output was mathematically
    // incomplete (missing the -I equivariance constraint on the free
    // simplices).
    const FlatMatrix* expected_by_row[8] = { &M_T, &M_T, &M_T, &M_S, &M_S, &M_S, &M_ST, &M_TS };
    size_t num_simplices = 0;
    for (const auto& level : K_sl2z.get_simplices()) num_simplices += level.size();
    if (data.size() != 8 + num_simplices) {
        throw std::runtime_error("GenerateSL2ZRelations_7VertexModel: expected exactly 8 seed relations plus one -I self-relation per simplex of K.");
    }
    for (size_t r = 0; r < 8; ++r) {
        if (!(data[r].rel.M == *expected_by_row[r])) {
            std::ostringstream oss;
            oss << "ERROR in GenerateSL2ZRelations_7VertexModel for Sym^" << n_sym
                << "(F_" << p_mod << "): caller-supplied matrix for relation '"
                << data[r].rel.tag << "' differs from the derived rho_n(g_r) (invariant I3).";
            throw std::runtime_error(oss.str());
        }
    }

    std::vector<LocalSystemSimplicialComplex::SimplexRelation> sl2z_relations;
    sl2z_relations.reserve(data.size());
    for (const auto& d : data) {
        sl2z_relations.push_back(d.rel);
    }
    return sl2z_relations;
}

namespace { // Anonymous namespace for helper struct and function

    struct SL2ZActionMatrices {
        FlatMatrix T_action;
        FlatMatrix S_action;
        FlatMatrix ST_action;
        FlatMatrix TS_action;
        // Add default constructor to allow easy creation before population
        SL2ZActionMatrices() : T_action(), S_action(), ST_action(), TS_action() {}
    };

    SL2ZActionMatrices ComputeSL2ZActionMatrices(uint32_t n_sym, uint32_t p_mod) {
        SL2ZActionMatrices matrices;

        // Define SL(2,Z) generator matrices in F_p
        FlatMatrix T0(2, 2, p_mod, std::vector<uint32_t>{ 1, 1, 0, 1 });
        FlatMatrix S0(2, 2, p_mod, std::vector<uint32_t>{ 0, p_mod - 1, 1, 0 }); // S = (0, -1; 1, 0)

        // Compute their symmetric powers
        matrices.T_action = ComputeSymmetricPowerMatrixFM(T0, n_sym);
        matrices.S_action = ComputeSymmetricPowerMatrixFM(S0, n_sym);

        // For stabilizers
        FlatMatrix ST0 = S0 * T0; // ST0 = (0 -1; 1 0)(1 1; 0 1) = (0 -1; 1 1)
        matrices.ST_action = ComputeSymmetricPowerMatrixFM(ST0, n_sym);

        FlatMatrix TS0 = T0 * S0; // TS0 = (1 1; 0 1)(0 -1; 1 0) = (1 -1; 1 0)
        matrices.TS_action = ComputeSymmetricPowerMatrixFM(TS0, n_sym);

        return matrices;
    }

} // anonymous namespace

void TestSL2ZCohomologySymmetricPower(uint32_t n_sym, uint32_t p_mod) {
    std::cout << "\n--- Testing SL(2,Z)\\H Cohomology with Sym^" << n_sym << "(F_" << p_mod << "^2) Local System ---" << std::endl;

    // 1. Create the base simplicial complex K
    SimplicialComplex K_sl2z = CreateSL2ZFundamentalDomainTriangulation();
    // std::cout << "  Base SimplicialComplex K for SL(2,Z) domain created:\n" << K_sl2z.to_string_offset(4); // Verbose

    // 2. Define parameters for the local system
    uint32_t d_V = n_sym + 1;
    std::cout << "  Local System Parameters: n_sym = " << n_sym << ", p = " << p_mod << ", d_V = " << d_V << std::endl;

    // 3. Define SL(2,Z) generator matrices in F_p
    FlatMatrix T0(2, 2, p_mod, std::vector<uint32_t>{ 1, 1, 0, 1 });
    FlatMatrix S0(2, 2, p_mod, std::vector<uint32_t>{ 0, p_mod - 1, 1, 0 }); // S = (0, -1; 1, 0)

    // 4. Compute their symmetric powers
    FlatMatrix M_T = ComputeSymmetricPowerMatrixFM(T0, n_sym);
    FlatMatrix M_S = ComputeSymmetricPowerMatrixFM(S0, n_sym);

    // std::cout << "  M_T (Sym^" << n_sym << "(T0)):\n" << M_T.to_string_offset(6); // Verbose
    // std::cout << "  M_S (Sym^" << n_sym << "(S0)):\n" << M_S.to_string_offset(6); // Verbose

    // For stabilizers
    FlatMatrix ST0 = S0 * T0; // ST0 = (0 -1; 1 0)(1 1; 0 1) = (0 -1; 1 1)
    FlatMatrix TS0 = T0 * S0; // TS0 = (1 1; 0 1)(0 -1; 1 0) = (1 -1; 1 0)
    FlatMatrix M_ST = ComputeSymmetricPowerMatrixFM(ST0, n_sym);
    FlatMatrix M_TS = ComputeSymmetricPowerMatrixFM(TS0, n_sym);

    // 5. Define SimplexRelations for SL(2,Z) identifications
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> sl2z_relations;
    try {
        // Vertex identifications for T
        sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 3 }), find_simplex_idx(K_sl2z, 0, { 0 }), M_T, "T_v3_v0");
        sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 5 }), find_simplex_idx(K_sl2z, 0, { 4 }), M_T, "T_v5_v4");

        // Edge identifications for T (path 0-4 with 3-5)
        // Edge (0,4) with (3,5) => f({3,5}) = M_T * f({0,4})
        sl2z_relations.emplace_back(1, find_simplex_idx(K_sl2z, 1, { 3,5 }), find_simplex_idx(K_sl2z, 1, { 0,4 }), M_T, "T_e(0,4)_e(3,5)");

        // S-identification: z -> -1/z. Identifies arc rho2-i with arc i-rho.
        // Arc rho^2-i is path 4-6. Arc i-rho is path 6-5 (reversed: 5-6).
        // S maps rho^2 (4) to rho (5) (S(e^{2pi i/3}) = e^{pi i/3}).
        // S maps i (6) to i (6) (S(i)=i).
        // S maps point on 4-6 to point on 5-6.
        //   rho^2 (4) <-> rho (5) (by S)
        //   i (6) <-> i (6) (by S, fixed point)
        // Rule: f(simplex_on_arc_i_rho) = M_S * f(simplex_on_arc_rho2_i)
        // Vertex identifications for S
        sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 4 }), find_simplex_idx(K_sl2z, 0, { 5 }), M_S, "S_v4_v5"); // rho from rho2
        // i is fixed by S, so f({10}) = M_S * f({10})
        sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 6 }), find_simplex_idx(K_sl2z, 0, { 6 }), M_S, "S_v6_stabilizer");

        // Edge identifications for S (path 4-6 with 5-6)
        // Edge (4,6) with (5,6) (S maps (rho^2,i) to (rho,i)) => f({4,6}) = M_S * f({5,6})
        sl2z_relations.emplace_back(1, find_simplex_idx(K_sl2z, 1, { 4,6 }), find_simplex_idx(K_sl2z, 1, { 5,6 }), M_S, "S_e(5,6)_e(4,6)");

        // Additional Stabilizer relations for orbifold points (not already covered by S acting on v6)
        // Point v4 (rho^2) has stabilizer ST. Relation: f(v4) = M_ST * f(v4).
        sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 4 }), find_simplex_idx(K_sl2z, 0, { 4 }), M_ST, "ST_v4_stabilizer");
        // Point v5 (rho) has stabilizer TS. Relation: f(v5) = M_TS * f(v5).
        sl2z_relations.emplace_back(0, find_simplex_idx(K_sl2z, 0, { 5 }), find_simplex_idx(K_sl2z, 0, { 5 }), M_TS, "TS_v5_stabilizer");

    }
    catch (const std::exception& e) {
        std::cerr << "ERROR setting up SL(2,Z) relations for Sym^" << n_sym << ": " << e.what() << std::endl;
        return;
    }

    // 6. Instantiate LocalSystemSimplicialComplex
    // std::cout << "  Instantiating LocalSystemSimplicialComplex for SL(2,Z)..." << std::endl; // Verbose
    LocalSystemSimplicialComplex lssc_sl2z(K_sl2z, sl2z_relations, d_V, p_mod);
    // std::cout << lssc_sl2z.to_string_offset(4); // Verbose

    // 7. Instantiate EquivariantCohomologyOnK
    // std::cout << "  Instantiating EquivariantCohomologyOnK for SL(2,Z)..." << std::endl; // Verbose
    EquivariantCohomologyOnK eq_cohom_sl2z(&lssc_sl2z);

    // 8. Compute Cohomology
    std::cout << "  Computing cohomology for SL(2,Z) with Sym^" << n_sym << "(F_" << p_mod << "^2)..." << std::endl;
    eq_cohom_sl2z.compute_cohomology();
    std::cout << "  Cohomology computed." << std::endl;

    // 9. Print Results
    std::cout << "\n--- Computed Cohomology H*(SL(2,Z)\\H; Sym^" << n_sym << "(F_" << p_mod << "^2)) ---" << std::endl;
    std::cout << eq_cohom_sl2z.to_string_offset(2) << std::endl;
}

void TestSL2Z_To_S1Cusp_RestrictionMorphism(uint32_t n_sym, uint32_t p_mod) {
    std::cout << "\n--- Testing SL(2,Z)\\H to S^1_cusp Restriction Morphism ---" << std::endl;
    std::cout << "    Sym^" << n_sym << "(F_" << p_mod << "^2) coefficients ---" << std::endl;

    // 1. Setup LSSC and Cohomology for SL(2,Z)\H
    SimplicialComplex K_sl2z = CreateSL2ZFundamentalDomainTriangulation();
    uint32_t d_V = n_sym + 1;
    std::cout << "  SL(2,Z) LSSC: K=" << K_sl2z.get_name() << ", n_sym=" << n_sym << ", p=" << p_mod << ", d_V=" << d_V << std::endl;

    // Get action matrices using the helper
    SL2ZActionMatrices sl2z_action_matrices = ComputeSL2ZActionMatrices(n_sym, p_mod);

    // Use the helper function to get relations for SL2Z LSSC
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> relations_sl2z =
        GenerateSL2ZRelations_7VertexModel(
            K_sl2z, n_sym, p_mod,
            sl2z_action_matrices.T_action,
            sl2z_action_matrices.S_action,
            sl2z_action_matrices.ST_action,
            sl2z_action_matrices.TS_action
        );

    LocalSystemSimplicialComplex lssc_sl2z(K_sl2z, relations_sl2z, d_V, p_mod);
    EquivariantCohomologyOnK eq_cohom_sl2z(&lssc_sl2z);
    std::cout << "  Computing SL(2,Z) cohomology..." << std::endl;
    eq_cohom_sl2z.compute_cohomology();
    std::cout << eq_cohom_sl2z.to_string_offset(4);

    // 2. Setup LSSC and Cohomology for S^1_cusp
    // S1_cusp uses the M_T_action (which is sl2z_action_matrices.T_action) for its loop monodromy.
    std::cout << "  S^1_cusp LSSC: K=K_Interval, monodromy=Sym^n(T0), d_V=" << d_V << std::endl;
    LocalSystemSimplicialComplex lssc_s1_cusp =
        LocalSystemPresets::CreateS1Generic(sl2z_action_matrices.T_action, p_mod, d_V, "S1_cusp_induced_by_T");
    EquivariantCohomologyOnK eq_cohom_s1_cusp(&lssc_s1_cusp); // This one uses the generic K_Interval from presets
    std::cout << "  Computing S^1_cusp cohomology (on generic K_Interval)..." << std::endl;
    eq_cohom_s1_cusp.compute_cohomology();
    // std::cout << eq_cohom_s1_cusp.to_string_offset(4); // Optional print

    // 3. Define K_S1_Path_Repr and its LSSC for the morphism domain
    std::vector<std::vector<std::vector<uint32_t>>> s1_path_simplices_data = {
        {{0},{1},{2},{3}},
        {make_edge(0,1), make_edge(1,2), make_edge(2,3)}
    };
    SimplicialComplex K_s1_path_representation(s1_path_simplices_data, "K_S1_Path_Repr_For_SL2Z");

    LocalSystemSimplicialComplex lssc_s1_cusp_on_path_for_morph(
        K_s1_path_representation,
        { LocalSystemSimplicialComplex::SimplexRelation(0,
            find_simplex_idx(K_s1_path_representation, 0, {3}),
            find_simplex_idx(K_s1_path_representation, 0, {0}),
            sl2z_action_matrices.T_action, // Use the same T_action matrix
            "S1_cusp_loop_on_path_for_morph") },
        d_V, p_mod
    );
    EquivariantCohomologyOnK eq_cohom_s1_cusp_on_path_for_morph(&lssc_s1_cusp_on_path_for_morph);
    std::cout << "  Computing S^1_cusp cohomology on specific K_S1_Path_Repr for morphism..." << std::endl;
    eq_cohom_s1_cusp_on_path_for_morph.compute_cohomology();
    std::cout << eq_cohom_s1_cusp_on_path_for_morph.to_string_offset(4);

    SimplicialComplexMorphism scm_s1_into_sl2z(&K_s1_path_representation, &K_sl2z);
    if (!scm_s1_into_sl2z.is_valid()) {
        std::cerr << "ERROR: SimplicialComplexMorphism for S1_path into K_SL2Z is not valid." << std::endl; return;
    }

    // 4. Define Fiber Map F: V_SL2Z -> V_S1_cusp. F is identity.
    FlatMatrix F_identity(d_V, p_mod);

    // 5. Create and Compute Morphism
    std::cout << "  Creating restriction morphism H*(SL2Z; L) -> H*(S1_cusp; L)..." << std::endl;
    EquivariantCochainComplexMorphismOnK restriction_morph(
        &eq_cohom_s1_cusp_on_path_for_morph,
        &eq_cohom_sl2z,
        &scm_s1_into_sl2z,
        F_identity
    );
    std::cout << "  Computing induced maps on cohomology..." << std::endl;
    restriction_morph.compute_induced_maps_on_cohomology();
    std::cout << restriction_morph.to_string_offset(2) << std::endl;

    std::cout << "--- TestSL2Z_To_S1Cusp_RestrictionMorphism for Sym^" << n_sym << " (p=" << p_mod << ") Finished ---" << std::endl;
}
