// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef LOCAL_SYSTEM_SL2Z_H
#define LOCAL_SYSTEM_SL2Z_H

#include "SimplicialCohomology.h" // For SimplicialComplex
#include "LocalSystemSimplicialComplex.h" // For SimplexRelation
#include "Hecke/ExactInt.h" // For hecke::Mat22 (exact group elements, Stage S2)

#include <vector>
#include <cstdint>

SimplicialComplex CreateSL2ZFundamentalDomainTriangulation();

// ===========================================================================
// Stage S2 -- Model group data (invariant I3; [C, sec:model] contract box in
// docs/latex/hecke_action.tex).
//
// The relation list of the 7-vertex model is the SINGLE SOURCE OF TRUTH for
// the pairs (relation r, group element g_r in SL(2,Z)). Every relation
// carries its g_r as exact integer 2x2 data, and its local-system matrix is
// DERIVED as rho_n(g_r) = ComputeSymmetricPowerMatrixFM(g_r mod ell, n) --
// never entered by hand.
// ===========================================================================

// One row of the model's relation/generator table ([C, sec:model]): the
// identification i(s_lhs) = g_r * i(s_rhs) of geometric simplices, at the
// group level (no coefficient system involved). `image_of_rhs[j]` is the
// model vertex that g_r sends `rhs_vertices[j]` to; as a set it equals
// `lhs_vertices` (this vertex correspondence is combinatorial model data,
// used by hecke::SL2ZTessellationTables to carry orientation signs by
// composition, never geometrically -- S2 risk note / invariant I8).
struct SL2ZRelationSeed {
    uint32_t q_dim;                       // dimension of both simplices
    std::vector<uint32_t> lhs_vertices;   // sorted vertex list of s_lhs
    std::vector<uint32_t> rhs_vertices;   // sorted vertex list of s_rhs
    hecke::Mat22 g_r;                     // exact generator: T, S, ST or TS
    const char* tag;                      // legacy relation tag (bit-identical)
    std::vector<uint32_t> image_of_rhs;   // g_r * rhs_vertices[j] = image_of_rhs[j]
};

// The 8 rows of [C, sec:model], in the exact legacy order of
// GenerateSL2ZRelations_7VertexModel: rows 1-3 |-> T, rows 4-6 |-> S, then
// ST (v4 stabilizer), TS (v5 stabilizer).
const std::vector<SL2ZRelationSeed>& SL2ZRelationSeeds_7VertexModel();

// A SimplexRelation together with its exact group element (invariant I3).
// rel.M is DERIVED as rho_n(g_r) -- see GenerateSL2ZRelationsWithGroupData.
struct SL2ZRelationDatum {
    LocalSystemSimplicialComplex::SimplexRelation rel; // matrix derived as rho_n(g_r)
    hecke::Mat22 g_r;                                  // exact generator: T, S, ST, TS or -I
};

// Generates the relations of the 7-vertex model with their group elements:
// first the 8 seed relations (in the exact legacy order; their list of
// q_dim, s_lhs, s_rhs, tags is identical to the pre-S2 output), then --
// FIX-negI -- one central self-relation f(sigma) = rho_n(-I) f(sigma) per
// simplex sigma of K (g_r = -I; tag "negI_q<q>_<idx>"), required by the Psi
// isomorphism of [C, thm:explicit_complex_equiv] since -I acts trivially on
// H (vacuous for even n; forces H^* = 0 for odd n, ell != 2). Each matrix is
// rho_n(g_r) = ComputeSymmetricPowerMatrixFM(g_r reduced mod ell, n_sym).
std::vector<SL2ZRelationDatum> GenerateSL2ZRelationsWithGroupData(
    const SimplicialComplex& K, uint32_t n_sym, uint32_t ell);

// hecke-extensions slot 6, WP-1 (phase E7a) -- THE CONTRAGREDIENT LOCAL
// SYSTEM D_n of [C, def:contragredient] on the same model.  Produces the SAME
// relation list as GenerateSL2ZRelationsWithGroupData -- the same rows in the
// same order, with the same q_dim, s_lhs, s_rhs, tag and g_r (the 8 seeds,
// then the FIX-negI self-relations) -- and only the matrix changed:
//     rel.M = rho_n^vee(g_r) = rho_n(adj g_r)^T   ( = rho_n(g_r^{-1})^T, g_r in SL(2,Z) ).
// The relation-by-relation sameness is ENGINE_CHECKed here and re-asserted by
// HeckeDualityValidation.  Uses ComputeContragredientPowerMatrixFM
// (LocalSystemUtils.h).
std::vector<SL2ZRelationDatum> GenerateSL2ZContragredientRelationsWithGroupData(
    const SimplicialComplex& K, uint32_t n_sym, uint32_t ell);

// hecke-extensions slot 11, WP-5 -- THE QUOTIENT LOCAL SYSTEM Q_n =
// Sym^{n+ell+1} / L . Sym^n on the same model (LocalSystemUtils.h,
// FibreRepresentation::QuotientL).  Same rows, order, q_dim, s_lhs, s_rhs,
// tag and g_r as GenerateSL2ZRelationsWithGroupData(K, n_sym, ell); only the
// matrix changes, to rel.M = rho_{Q_n}(g_r mod ell) = ComputeQuotientLMatrixFM
// ((ell+1) x (ell+1)).  n_sym is the n of Q_n.
std::vector<SL2ZRelationDatum> GenerateSL2ZQuotientLRelationsWithGroupData(
    const SimplicialComplex& K, uint32_t n_sym, uint32_t ell);

// Stage S13: element-width-generic relation generator for the extended
// coefficient regime (ell < 2^64). Returns the SimplexRelation list directly
// (seeds + FIX-negI, identical structure to GenerateSL2ZRelationsWithGroupData)
// with M = rho_n(g_r) over MatrixT<Elt>. For Elt = uint32_t it reproduces the
// relation list of GenerateSL2ZRelationsWithGroupData exactly.
template<class Elt>
std::vector<typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation>
GenerateSL2ZRelationsElt(const SimplicialComplex& K, uint32_t n_sym, Elt ell);

// ===========================================================================
// hecke-extensions slot 1, WP-4b (phase E4b) -- the CIRCLE MODEL of the
// boundary ([C, prop:circlemodel], [C, rmk:bdrymodel]; [C] = gl2_contract.tex,
// cited by label), lifted out of the demo TestSL2Z_To_S1Cusp_RestrictionMorphism
// (below, unchanged) as a library entry point for Hecke/HeckeBoundaryOperator.
//
// K_bd is the horocycle path v0 - v1 - v2 - v3 of the 7-vertex model, with
// K's OWN vertex labels {0, 1, 2, 3} (so that SimplicialComplexMorphism(&K_bd,
// &K) is the inclusion j : K_bd -> K of [C, sec:cone]) and with its q-simplex
// lists in the order of hecke::SL2ZTessellationTables::boundary_simplices(q)
// -- vertices {0},{1},{2},{3}; edges {0,1},{1,2},{2,3}. Byte-identical to the
// demo's K_S1_Path_Repr_For_SL2Z simplex data.
// ===========================================================================
SimplicialComplex CreateSL2ZBoundaryPathTriangulation();

// The relation list Rel_bd of the circle model C^*_bd = C^*_eq(K_bd, Rel_bd; L),
// L = Sym^n(F_ell^2): the single cusp relation f(v3) = rho_n(T) f(v0) of
// [C, prop:circlemodel] (tag "cusp_T", exactly as the demo spells it), then
// -- FIX-negI, mirroring GenerateSL2ZRelationsElt -- one central self-relation
// f(sigma) = rho_n(-I) f(sigma) per simplex of K_bd (tags "negI_bd_q<q>_<s>"),
// which makes C^*_bd literally Hom_{Z[Gamma]}(C_*(X_bd), V) = Hom_{Z[Gamma_inf]}
// (C_*(X_bd,inf), V^{-I}) of [C, prop:circlemodel](1) for EVERY n: vacuous
// for n even or ell = 2 (rho_n(-I) = I, the short-circuit skips it) and the
// zero complex for n odd at 2 in R^x ([C, rmk:bdryoddn]), exactly as the bulk
// relation list behaves. Element-width generic (Elt in {uint32_t, uint64_t}).
template<class Elt>
std::vector<typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation>
GenerateSL2ZBoundaryRelationsElt(const SimplicialComplex& K_bd, uint32_t n_sym, Elt ell);

// Legacy generator (declared here since S2 so validation stages can call it).
// Since Stage S2 this delegates to GenerateSL2ZRelationsWithGroupData and
// strips g_r; the caller-supplied matrices are checked against the derived
// rho_n(g_r) (exact equality) and a std::runtime_error is thrown on mismatch
// (invariant I3 guard; all pre-S2 callers pass exactly these matrices).
// Since FIX-negI it returns 8 + N_simplices relations (31 for the 7-vertex
// model): the 8 seed relations first, then the appended -I self-relations.
std::vector<LocalSystemSimplicialComplex::SimplexRelation>
GenerateSL2ZRelations_7VertexModel(
    const SimplicialComplex& K_sl2z,
    uint32_t n_sym,
    uint32_t p_mod,
    const FlatMatrix& M_T,
    const FlatMatrix& M_S,
    const FlatMatrix& M_ST,
    const FlatMatrix& M_TS);

/**
 * @brief Tests cohomology for SL(2,Z)\H with coefficients in Sym^n_sym(F_p^2).
 * @param n_sym The degree of the symmetric power (e.g., 0 for trivial, 1 for standard, 2 for Sym^2).
 * @param p_mod The prime modulus for the field F_p.
 */
void TestSL2ZCohomologySymmetricPower(uint32_t n_sym, uint32_t p_mod);

/**
 * @brief Tests the restriction morphism H*(SL(2,Z)\H; L_V) -> H*(S^1_cusp; L_V|_S^1).
 *        L_V is Sym^n_sym(F_p^2). L_V|_S^1 has monodromy Sym^n_sym(T_0).
 * @param n_sym The degree of the symmetric power.
 * @param p_mod The prime modulus.
 */
void TestSL2Z_To_S1Cusp_RestrictionMorphism(uint32_t n_sym, uint32_t p_mod);

#endif // LOCAL_SYSTEM_SL2Z_H
