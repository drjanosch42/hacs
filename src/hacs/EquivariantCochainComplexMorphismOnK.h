// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef EQUIVARIANT_COCHAIN_COMPLEX_MORPHISM_ON_K_H
#define EQUIVARIANT_COCHAIN_COMPLEX_MORPHISM_ON_K_H

#include "EquivariantCohomologyOnK.h" // Includes LSSC, SimplicialComplex, FlatMatrix
#include "SimplicialCohomology.h"     // For SimplicialComplexMorphism
#include "../ModArithmetic/FlatMatrix.h" // For FlatMatrix (per-simplex block data)
#include "../ModArithmetic/helper.h"
#include "../ModArithmetic/BlockMat.h"     // engine::BlockMat / BlockPool (Phase 3)

#include <memory>

#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>

// -----------------------------------------------------------------------------
// Stage S6 (the code-level fix of the "v5 bug", [C, rmk:nogo]): cochain maps
// must support per-simplex, MULTI-SOURCE block data. The middle map beta of the
// induced-module sequence CANNOT be realized by a single constant fiber matrix
// ([C, rmk:nogo] "(ii) No repair is possible"): a composite of three Gamma-module
// morphisms V -> Ind V -> Ind V -> V is scalar on cohomology, but T_2 on
// H^1(Sym^10) has eigenvalues {-24,-24,2049}, which is not scalar. The contract's
// resolution ([C, rmk:nogo] "Resolution adopted by this contract" and the remark
// "Recovering the three-map factorization of [v5]") realizes the middle map as an
// explicit cochain map given by PER-SIMPLEX BLOCK MATRICES built from the twisted
// chain datum tau: the value (Xi_g c)(sigma) is a SUM of matrices applied to the
// cochain values c(sigma_k) at OTHER simplices sigma_k designated by tau
// ([C, eq:xiexplicit]). A strictly diagonal table sigma -> F_sigma is therefore
// insufficient; the datum is a sparse block ROW per target simplex.
//
// PerSimplexBlock: a single contribution "F * c2(s2_idx_in_K2)" to the pullback
// value at a target simplex. F is a d_V1 x d_V2 FlatMatrix.
struct PerSimplexBlock {
    uint32_t s2_idx_in_K2; // source simplex index in K2 (codomain of the H*-map's underlying space)
    FlatMatrix F;          // d_V1 x d_V2 block; contributes F * c2(s2_idx_in_K2)
};

// FiberBlockTable: fiber_blocks[q][s1_idx] is the sparse block ROW of contributions
// to the pullback value at the target simplex s1_idx in K1 (degree q):
//
//     h1(s1) = sum over b in fiber_blocks[q][s1_idx] of  b.F * h2(b.s2_idx_in_K2)
//
// (all b.F are d_V1 x d_V2). The single-constant-fiber-matrix case is the sub-case
// of exactly one entry (f_K(s1), F_const) per simplex; a diagonal table is the
// sub-case s2 = f_K(s1) with varying F. This subsumes the SimplicialComplexMorphism
// data (which simplex of K2 each simplex of K1 draws from).
using FiberBlockTable = std::vector<std::vector<std::vector<PerSimplexBlock>>>;

// NOTE ON THE (CONTRAVARIANT) NAMING, kept deliberately unchanged (Stage S6 risk
// box): the induced map on cohomology is contravariant, so the object called
// "domain" here is the TARGET of the induced H*-map (cohomology of L1 on K1) and
// "codomain" is its SOURCE (cohomology of L2 on K2). This is confusing but is NOT
// "fixed" in this stage to avoid churn across the three existing callers.
class EquivariantCochainComplexMorphismOnK {
private:
    const EquivariantCohomologyOnK* domain_lssc_cohomology;    // Cohomology of LSSC_1 (target of map on H*)
    const EquivariantCohomologyOnK* codomain_lssc_cohomology;  // Cohomology of LSSC_2 (source of map on H*)
    const SimplicialComplexMorphism* simplicial_inclusion_K1_into_K2; // f_K: K1 -> K2 (may be null for the general ctor)
    FlatMatrix fiber_map_V2_to_V1; // F: V2 -> V1 (legacy single-constant-fiber-matrix case; kept for getters/to_string)

    // The per-simplex sparse block rows (the S6 generalization). Both constructors
    // populate this; compute_induced_maps_on_cohomology iterates it as the single
    // code path. Indexed [q][s1_idx] -> list of PerSimplexBlock.
    FiberBlockTable fiber_blocks_;
    std::string label_; // human-readable label (general ctor); empty for the legacy ctor

    uint32_t prime;
    uint32_t d_V1; // representation_dim of domain LSSC_1
    uint32_t d_V2; // representation_dim of codomain LSSC_2

    // (matrix-engine Phase 3) FlatMatrix storage. The accumulation order of
    // compute_induced_maps_on_cohomology (the S6 bit-identity contract B6) is
    // untouched; only the container of the finished columns changed.
    std::vector<FlatMatrix> induced_H_eq_maps; // H*(f_L): H*_eq(L2) -> H*_eq(L1)
    bool maps_computed;

    // (slot-9 WP-4) The coefficient ring (1 = F_ell) and the Z/ell^k induced maps.
    uint32_t ring_k_ = 1;
    std::vector<engine::DenseBuf<uint32_t>> induced_module_;
    void compute_induced_maps_module();

    // Helper to find the index of a simplex from K1 in K2
    // Returns -1 if not found (should not happen for inclusions if K1's simplices are literally in K2)
    int32_t find_simplex_K1_idx_in_K2(uint32_t q, uint32_t s1_idx_in_K1) const;

    // Builds the one-entry-per-simplex FiberBlockTable for the legacy constant
    // fiber-matrix case via find_simplex_K1_idx_in_K2 (single source of the s2 per
    // s1). Used only by the legacy constructor so that old callers execute
    // byte-identical arithmetic (one block, same multiply order).
    void build_constant_fiber_block_table();

    // -------------------------------------------------------------------------
    // (matrix-engine Phase 3 Stage B) The pullback operator as a block matrix.
    //
    // fiber_blocks_[q] IS a block-sparse matrix: the block row s1 holds the
    // blocks b.F at block columns b.s2_idx_in_K2, and
    //     h1 = Phi^{(q)} h2,   Phi^{(q)}[s1][s2] = sum of the F's naming s2,
    // in the flat layout documented at apply_to_flat_cochain (simplex-major,
    // fiber-minor) -- which is exactly BlockMat's L1 multi-vector layout
    // (ENGINE_API.md Sec. 9.2(4)) for the uniform row partition (d_V1, ..., d_V1).
    // So M1 and M3 are one `blk::gemm_dense` each, and the whole per-cell
    // extract/multiply/accumulate triple loop disappears.
    //
    // BIT-IDENTITY (contract B6, plan Phase 3 / 02 Sec. 6.5(7)). The old code
    // accumulated, per target component, the terms (b.F * h2(b.s2))_k with
    // ModArithmetic::mod_add in table order. Every term is an exact product of
    // canonical residues; the engine accumulates such products exactly in u64
    // and reduces once per k-block (ENGINE_API.md Sec. 5). A sum of exact
    // integers does not depend on its bracketing, so the GEMM reproduces the
    // old per-column result entry for entry -- which is what
    // HeckeMorphismRegressionValidation, HeckeMorphismGeneralizationValidation
    // and golden tier B check on real data.
    //
    // The grid is N_q(K1) x max(N_q(K2), largest s2 named by the table + 1),
    // so a block naming a source simplex outside K2's list keeps contributing
    // exactly what the old guarded loop gave it: its column of the operand is
    // zero, hence nothing.
    //
    // Built lazily and cached (fiber_blocks_ is immutable after construction);
    // the pool dedups identical F's -- the legacy constant-fiber table interns
    // its single matrix once for all N_q simplices.
    mutable std::shared_ptr<engine::BlockPool<uint32_t>> block_pool_;
    mutable std::vector<engine::BlockMat<uint32_t>> pullback_ops_;
    mutable std::vector<char> pullback_built_;
    const engine::BlockMat<uint32_t>& pullback_operator(uint32_t q) const;

    // Y = Phi * X on the block pattern (never a cell), skipping blocks whose
    // operand slice is identically zero -- see the .cpp for why that skip is
    // load-bearing for S8's one-basis-vector-at-a-time cross-check, and why it
    // cannot move a bit.
    void apply_blocks(const engine::BlockMat<uint32_t>& Phi,
                      engine::DenseView<const uint32_t> X,
                      engine::DenseView<uint32_t> Y) const;


public:
    // EXISTING constructor (UNCHANGED signature): single constant fiber matrix +
    // simplicial inclusion. Internally builds the one-entry-per-simplex
    // FiberBlockTable via find_simplex_K1_idx_in_K2 and delegates to the general
    // block-table code path -> old callers get byte-identical results.
    EquivariantCochainComplexMorphismOnK(
        const EquivariantCohomologyOnK* domain_lssc_cohomology,    // L1 on X1
        const EquivariantCohomologyOnK* codomain_lssc_cohomology,  // L2 on X2
        const SimplicialComplexMorphism* simplicial_inclusion_K1_into_K2, // Assumes K1 is domain of scm, K2 is codomain
        const FlatMatrix& fiber_map_V2_to_V1);

    // NEW constructor: general per-simplex block rows. The simplicial-map data is
    // subsumed by the table (each block names its source simplex s2 in K2). Pass a
    // nullptr / identity simplicial morphism for K1 = K2 = K (the Hecke case).
    EquivariantCochainComplexMorphismOnK(
        const EquivariantCohomologyOnK* domain_lssc_cohomology,    // L1 on X1 (target of H*-map)
        const EquivariantCohomologyOnK* codomain_lssc_cohomology,  // L2 on X2 (source of H*-map)
        FiberBlockTable fiber_blocks,
        std::string label);

    ~EquivariantCochainComplexMorphismOnK() = default;

    // --- Computation ---
    void compute_induced_maps_on_cohomology();

    // --- Cochain-level application (Stage S6; consumed by the S8 cross-check
    //     omega* . beta* . alpha* == Theta_g) ---
    //
    // Applies the block table to a flat q-cochain of the SOURCE local system L2 and
    // returns the flat q-cochain of the TARGET local system L1:
    //   out(s1) = sum over b in fiber_blocks[q][s1] of  b.F * c2(b.s2_idx_in_K2).
    //
    // FLAT COCHAIN LAYOUT (binding; S7/S8 MUST lay out C^q(K;V) the same way):
    //   A q-cochain c in C^q(K;V) is a column vector of length N_q * d_V, where
    //   N_q is the number of q-simplices in the stored SimplicialComplex order.
    //   The entry at row index  (s * d_V + k)  holds  c(sigma_s)_k, i.e. simplex
    //   index s is MAJOR (outer) and the fiber coordinate k = 0..d_V-1 is MINOR
    //   (inner), in the coefficient basis u_0 = x^n, ..., u_n = y^n. This is exactly
    //   the coordinate system consumed by
    //   EquivariantCohomologyOnK::get_H_eq_projectors_from_C_eq_flat_fm() and specified
    //   in the plan's Section 3, Level (a).
    //
    // c2_flat is (N_q(K2) * d_V2) x 1; the returned cochain is (N_q(K1) * d_V1) x 1.
    FlatMatrix apply_to_flat_cochain(uint32_t q, const FlatMatrix& c2_flat) const;

    // --- Getters ---
    const EquivariantCohomologyOnK* get_domain_lssc_cohomology() const { return domain_lssc_cohomology; }
    const EquivariantCohomologyOnK* get_codomain_lssc_cohomology() const { return codomain_lssc_cohomology; }
    const SimplicialComplexMorphism* get_simplicial_inclusion() const { return simplicial_inclusion_K1_into_K2; }
    const FlatMatrix& get_fiber_map() const { return fiber_map_V2_to_V1; }
    const FiberBlockTable& get_fiber_blocks() const { return fiber_blocks_; }
    const std::string& get_label() const { return label_; }
    bool are_maps_computed() const { return maps_computed; }
    const std::vector<FlatMatrix>& get_induced_H_eq_maps() const;

    // --- (slot-9 WP-4; E8 item 4b; [E] v1.10.0-pre.1 sec:ring (b1), (f)) Z/ell^k ---
    // The CONSTANT-FIBRE constructor honours a composite ring when both
    // cohomology objects are over the Z/ell^k in force (ell >= 5; the fibre map
    // over the same modulus ell^k); a ring mismatch between domain and codomain
    // is a wrong input (named std::logic_error), and the FiberBlockTable
    // constructor keeps refusing.  Over Z/ell^k the induced maps are the (c5)
    // matrices between the two cyclic-summand bases (column j = the target
    // coordinates of the pullback of b_j, entry i in [0, ell^{e_i})), and
    // get_induced_H_eq_maps() throws a named std::logic_error.
    uint32_t coefficient_ring_k() const { return ring_k_; }
    bool is_composite() const { return ring_k_ >= 2; }
    // The pullback on flat q-cochains, dense mod ell^k: (N_q(K1) d_V1) x (N_q(K2) d_V2).
    engine::DenseBuf<uint32_t> pullback_matrix_module(uint32_t q) const;
    const std::vector<engine::DenseBuf<uint32_t>>& get_induced_H_module_maps() const;

    // --- String Conversion ---
    std::string to_string() const;
    std::string to_string_offset(const uint32_t offset) const;

    // --- Compatibility Check (User Asserted for now, or basic version) ---
    // bool verify_compatibility(bool check_strict_relations = false) const;
};

#endif // EQUIVARIANT_COCHAIN_COMPLEX_MORPHISM_ON_K_H
