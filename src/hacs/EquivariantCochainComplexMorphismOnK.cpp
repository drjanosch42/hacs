// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * EquivariantCochainComplexMorphismOnK.cpp
 *
 * This file implements the `EquivariantCochainComplexMorphismOnK` class. Its
 * purpose is to compute the induced map on equivariant cohomology,
 *   H*(f_L): H*_eq(L2) -> H*_eq(L1)
 * arising from a morphism between two local systems, f_L: L1 -> L2.
 *
 * Stage S6 (the code-level fix of the "v5 bug", [C, rmk:nogo]): the class now
 * represents its cochain-level datum as a per-simplex sparse block ROW table
 * (`FiberBlockTable`), so that the pullback value at a simplex may be a SUM of
 * matrices applied to the source cochain values at OTHER simplices. This is the
 * datum required to represent the middle map beta_bullet / Xi_g of the contract,
 * which CANNOT be a single constant fiber matrix (see the header comment and
 * [C, rmk:nogo]). The single-constant-fiber-matrix case (alpha, omega, and the
 * three legacy preset/SL2Z callers) is recovered exactly by a one-entry-per-
 * simplex table, with byte-identical arithmetic.
 *
 * Defining the Morphism (legacy constant-fiber-matrix constructor):
 * A morphism of local systems is specified by:
 * 1. The source and target `EquivariantCohomologyOnK` objects, which contain
 *    the computed cohomology for local systems L2 and L1, respectively.
 * 2. A `SimplicialComplexMorphism` f_K: K1 -> K2, representing an inclusion ?(or
 *    map, but this is not yet implemented) between the base simplicial complexes.
 * 3. A linear `FlatMatrix` F: V2 -> V1, representing the map on the fibers
 *    (representation spaces) of the local systems.
 * The implementation assumes the user has ensured that f_K and F are compatible
 * with the relations defining L1 and L2 (i.e., F * M2 = M1 * F).
 *
 * Core Algorithm (`compute_induced_maps_on_cohomology`):
 * The induced map on cohomology is contravariant. The algorithm computes the
 * matrix representation of this map for each dimension q:
 *
 * 1. For each basis vector `h2_j` of the source cohomology `H^q(L2)`, its image
 *    under the pullback f* is constructed as a cochain `h_prime_1_j` in the
 *    target's cochain space `C^q_eq(K1, V1)`. This is done simplex-wise using the
 *    per-simplex block rows: h_prime_1_j(s1) = sum_b b.F * h2_j(b.s2). In the
 *    legacy single-constant-fiber case, this is the one term F( h2_j( f_K(s1) ) ).
 *
 * 2. The resulting cochain `h_prime_1_j` is "flattened" into a single column vector.
 *
 * 3. This vector is then projected onto the target cohomology group `H^q(L1)`
 *    using a pre-computed projection matrix from the `EquivariantCohomologyOnK`
 *    object for L1.
 *
 * 4. The resulting coordinate vector gives the j-th column of the matrix for
 *    the induced map H^q(f_L).
 *
 * The final matrices for each dimension are stored for retrieval.
 */

#include "EquivariantCochainComplexMorphismOnK.h"
#include "LocalSystemUtils.h" // For find_simplex_idx (if needed, or adapt)

// (matrix-engine Phase 3 Stage B) M1/M3 as block GEMMs -- see the
// pullback_operator() comment in the header for the contract-B6 argument.
#include "../ModArithmetic/BlockElim.h"
#include "../ModArithmetic/BlockKernels.h"
#include "EngineCheck.h"
#include "Hecke/HeckeSettings.h"
#include "Util/FieldDiet.h"

#include <algorithm>

// Legacy constructor: single constant fiber matrix + simplicial inclusion.
// Builds the one-entry-per-simplex FiberBlockTable and delegates to the general
// block-table code path so old callers execute byte-identical arithmetic.
EquivariantCochainComplexMorphismOnK::EquivariantCochainComplexMorphismOnK(
    const EquivariantCohomologyOnK* domain_lssc_cohomology_ptr,
    const EquivariantCohomologyOnK* codomain_lssc_cohomology_ptr,
    const SimplicialComplexMorphism* simplicial_inclusion,
    const FlatMatrix& fiber_map)
    : domain_lssc_cohomology(domain_lssc_cohomology_ptr),
    codomain_lssc_cohomology(codomain_lssc_cohomology_ptr),
    simplicial_inclusion_K1_into_K2(simplicial_inclusion),
    fiber_map_V2_to_V1(fiber_map),
    label_(),
    maps_computed(false) {
    // (slot-9 WP-4; E8 item 4b) The constant-fibre constructor honours Z/ell^k
    // when BOTH objects are over the ring in force (the restriction of the
    // pair); a domain / codomain ring mismatch is a wrong input.  Every other
    // combination keeps slot 8's refusal: [E] sec:ring (b).
    {
        const uint32_t kd = domain_lssc_cohomology_ptr ? domain_lssc_cohomology_ptr->coefficient_ring_k() : 1u;
        const uint32_t kc = codomain_lssc_cohomology_ptr ? codomain_lssc_cohomology_ptr->coefficient_ring_k() : 1u;
        const bool dc = kd >= 2, cc = kc >= 2;
        if (dc != cc || (dc && cc && (kd != kc || domain_lssc_cohomology_ptr->ring_context().p !=
                                                      codomain_lssc_cohomology_ptr->ring_context().p)))
            throw std::logic_error("EquivariantCochainComplexMorphismOnK: the domain (" + hecke::CoefficientRingText(kd) +
                                   ") and the codomain (" + hecke::CoefficientRingText(kc) +
                                   ") are over different rings -- a wrong input ([E] sec:ring (b1))");
        if (dc && hecke::CoefficientRingK() == kd) {
            const uint64_t ell = static_cast<uint64_t>(domain_lssc_cohomology_ptr->ring_context().ell);
            if (ell == 2 || ell == 3 || domain_lssc_cohomology_ptr->is_flat_route() || codomain_lssc_cohomology_ptr->is_flat_route())
                hecke::RefuseCoefficientRing("EquivariantCochainComplexMorphismOnK", kd,
                                             " at ell = " + std::to_string(ell) +
                                                 " (the restriction over Z/ell^k is honoured at ell >= 5 only; [E] sec:ring (f))");
            ring_k_ = kd;
        } else {
            // (slot-8 WP-3) [E] sec:ring (b): the induced maps are field (D5b) objects.
            hecke::RequireFieldCoefficientRing("EquivariantCochainComplexMorphismOnK", std::max(kd, kc));
        }
    }

    if (!domain_lssc_cohomology || !codomain_lssc_cohomology) {
        throw std::runtime_error("EquivariantCochainComplexMorphismOnK: LSSC cohomology wrapper pointers cannot be null.");
    }
    if (!simplicial_inclusion_K1_into_K2) {
        throw std::runtime_error("EquivariantCochainComplexMorphismOnK: SimplicialComplexMorphism pointer cannot be null.");
    }

    // (hecke-extensions slot 9, WP-9; slot-8 WP-12 review, m-10) get_prime()
    // now refuses on a composite object; the honouring branch just above
    // (ring_k_ >= 2) is exactly that case (E8 item 4b's Z/ell^k restriction,
    // ell >= 5), so it reads the ring-aware modulus instead -- the SAME
    // numeric value get_prime() returned before that guard existed, since
    // `prime` on a composite EquivariantCohomologyOnK is ell^k by
    // construction. The domain/codomain ring_context().p equality was already
    // asserted above when both are composite, so only the fiber map is
    // checked against it here.
    if (ring_k_ >= 2) {
        prime = static_cast<uint32_t>(domain_lssc_cohomology->ring_context().p);
        if (prime != fiber_map_V2_to_V1.get_p()) {
            throw std::runtime_error("EquivariantCochainComplexMorphismOnK: Prime moduli mismatch.");
        }
    } else {
        prime = domain_lssc_cohomology->get_prime();
        if (prime != codomain_lssc_cohomology->get_prime() || prime != fiber_map_V2_to_V1.get_p()) {
            throw std::runtime_error("EquivariantCochainComplexMorphismOnK: Prime moduli mismatch.");
        }
    }

    d_V1 = domain_lssc_cohomology->get_representation_dim();
    d_V2 = codomain_lssc_cohomology->get_representation_dim();

    if (fiber_map_V2_to_V1.get_rows() != d_V1 || fiber_map_V2_to_V1.get_cols() != d_V2) {
        std::ostringstream oss;
        oss << "EquivariantCochainComplexMorphismOnK: Fiber map F dimensions ("
            << fiber_map_V2_to_V1.get_rows() << "x" << fiber_map_V2_to_V1.get_cols()
            << ") not compatible with V1_dim=" << d_V1 << ", V2_dim=" << d_V2;
        throw std::runtime_error(oss.str());
    }

    // TODO: Add compatibility check F B = A F.
    // This is complex. For now, user asserts this.
    // Example: if LSSC has a "name" or "tag" for its relations, and K1, K2 are "the same K",
    // one could iterate through L1's relations, find matching tag in L2, and check.
    // For K1 proper subcomplex of K2, it's harder.

    // Build the one-entry-per-simplex FiberBlockTable (single source of the s2 per
    // s1 is find_simplex_K1_idx_in_K2), so compute_induced_maps_on_cohomology runs
    // the SAME single code path as the general constructor.
    build_constant_fiber_block_table();
}

// General constructor: per-simplex block rows. The simplicial-map data is
// subsumed by the table (each block names its source simplex s2 in K2).
EquivariantCochainComplexMorphismOnK::EquivariantCochainComplexMorphismOnK(
    const EquivariantCohomologyOnK* domain_lssc_cohomology_ptr,
    const EquivariantCohomologyOnK* codomain_lssc_cohomology_ptr,
    FiberBlockTable fiber_blocks,
    std::string label)
    : domain_lssc_cohomology(domain_lssc_cohomology_ptr),
    codomain_lssc_cohomology(codomain_lssc_cohomology_ptr),
    simplicial_inclusion_K1_into_K2(nullptr),
    fiber_map_V2_to_V1(),
    fiber_blocks_(std::move(fiber_blocks)),
    label_(std::move(label)),
    maps_computed(false) {
    // (slot-8 WP-3) [E] sec:ring (b): the induced maps are field (D5b) objects.
    hecke::RequireFieldCoefficientRing(
        "EquivariantCochainComplexMorphismOnK",
        std::max(domain_lssc_cohomology_ptr ? domain_lssc_cohomology_ptr->coefficient_ring_k() : 1u,
                 codomain_lssc_cohomology_ptr ? codomain_lssc_cohomology_ptr->coefficient_ring_k() : 1u));

    if (!domain_lssc_cohomology || !codomain_lssc_cohomology) {
        throw std::runtime_error("EquivariantCochainComplexMorphismOnK (block ctor): LSSC cohomology wrapper pointers cannot be null.");
    }

    prime = domain_lssc_cohomology->get_prime();
    if (prime != codomain_lssc_cohomology->get_prime()) {
        throw std::runtime_error("EquivariantCochainComplexMorphismOnK (block ctor): Prime moduli mismatch.");
    }

    d_V1 = domain_lssc_cohomology->get_representation_dim();
    d_V2 = codomain_lssc_cohomology->get_representation_dim();

    // Validate block dimensions: every F must be d_V1 x d_V2 over F_prime.
    for (uint32_t q = 0; q < fiber_blocks_.size(); ++q) {
        for (uint32_t s1 = 0; s1 < fiber_blocks_[q].size(); ++s1) {
            for (const PerSimplexBlock& blk : fiber_blocks_[q][s1]) {
                if (blk.F.get_rows() != d_V1 || blk.F.get_cols() != d_V2 || blk.F.get_p() != prime) {
                    std::ostringstream oss;
                    oss << "EquivariantCochainComplexMorphismOnK (block ctor): block F at q=" << q
                        << ", s1=" << s1 << " has dims " << blk.F.get_rows() << "x" << blk.F.get_cols()
                        << " (p=" << blk.F.get_p() << "), expected " << d_V1 << "x" << d_V2
                        << " (p=" << prime << ").";
                    throw std::runtime_error(oss.str());
                }
            }
        }
    }
}

// Helper to find index of K1's simplex in K2
// s1_idx_in_K1 is the index of the simplex in K1->get_simplices()[q]
int32_t EquivariantCochainComplexMorphismOnK::find_simplex_K1_idx_in_K2(uint32_t q, uint32_t s1_idx_in_K1) const {
    const SimplicialComplex* K1 = domain_lssc_cohomology->get_lssc()->get_base_K_complex();
    const SimplicialComplex* K2 = codomain_lssc_cohomology->get_lssc()->get_base_K_complex();

    if (q >= K1->get_simplices().size() || s1_idx_in_K1 >= K1->get_simplices()[q].size()) {
        throw std::out_of_range("find_simplex_K1_idx_in_K2: s1_idx_in_K1 out of bounds.");
    }
    const auto& simplex_s1_vertices = K1->get_simplices()[q][s1_idx_in_K1];

    if (q >= K2->get_simplices().size()) { // K2 might not have q-simplices if K1 does (e.g. K2 is smaller dim)
        return -1;
    }

    // (matrix-engine Phase 3, 02 Sec. 7.17) O(log N) lookup in K2's cached
    // ORDERED simplex index (never a hash map -- 02 Sec. 6.5(5) / R10), replacing
    // the O(N) std::find. Identical index: both are first-match-wins, and K1's
    // simplices are stored in ascending vertex order (validate_simplices()).
    const int64_t s2 = K2->find_simplex_index(q, simplex_s1_vertices);
    if (s2 < 0) {
        return -1; // Simplex from K1 not found in K2 (should not happen for true inclusion)
    }
    return static_cast<int32_t>(s2);
}

// Builds the one-entry-per-simplex FiberBlockTable for the legacy constant
// fiber-matrix case. For each target simplex s1 of K1, the single source simplex
// is find_simplex_K1_idx_in_K2(q, s1); if that returns -1 the block row is EMPTY
// (mirroring the old `continue`), otherwise it is a single block (s2, F_const).
void EquivariantCochainComplexMorphismOnK::build_constant_fiber_block_table() {
    const SimplicialComplex* K1 = domain_lssc_cohomology->get_lssc()->get_base_K_complex();
    const FlatMatrix& F_flat = fiber_map_V2_to_V1; // d_V1 x d_V2, the single constant block

    uint32_t num_degrees = static_cast<uint32_t>(K1->get_simplices().size());
    fiber_blocks_.assign(num_degrees, std::vector<std::vector<PerSimplexBlock>>());
    for (uint32_t q = 0; q < num_degrees; ++q) {
        uint32_t N_q_K1 = K1->get_num_q_simplices(q);
        fiber_blocks_[q].assign(N_q_K1, std::vector<PerSimplexBlock>());
        for (uint32_t s1_idx = 0; s1_idx < N_q_K1; ++s1_idx) {
            int32_t s2_idx = find_simplex_K1_idx_in_K2(q, s1_idx);
            if (s2_idx == -1) continue; // empty block row -- exactly the old `continue`
            fiber_blocks_[q][s1_idx].push_back(
                PerSimplexBlock{ static_cast<uint32_t>(s2_idx), F_flat });
        }
    }
}

// Main computation
void EquivariantCochainComplexMorphismOnK::compute_induced_maps_on_cohomology() {
    if (maps_computed) return;
    if (!domain_lssc_cohomology->is_cohomology_computed() || !codomain_lssc_cohomology->is_cohomology_computed()) {
        throw std::runtime_error("compute_induced_maps_on_cohomology: Cohomology of underlying LSSCs must be computed first.");
    }
    if (ring_k_ >= 2) {   // (slot-9 WP-4) Z/ell^k: the (c5) induced maps; the field path below is untouched
        compute_induced_maps_module();
        maps_computed = true;
        return;
    }

    const SimplicialComplex* K1 = domain_lssc_cohomology->get_lssc()->get_base_K_complex();

    const auto& H_eq_dims_L1 = domain_lssc_cohomology->get_H_eq_dimensions();
    const auto& H_eq_dims_L2 = codomain_lssc_cohomology->get_H_eq_dimensions();

    // (slot-24 G6, Util/FieldDiet.h) Under the diet the codomain's H basis is
    // read straight from its block form (H_eq_block(q): column j, entry
    // (s2 d_V2 + k) is h2_j's (s2, k) -- the layout get_H_eq_bases_fm()'s
    // per-cochain FlatMatrices are cut from), so the codomain's FlatMatrix list
    // cache -- EVERY C / Z / B / H basis vector of the bulk as its own FlatMatrix,
    // ~1.3 GB at n_sym = 1000, kept for the cohomology's lifetime -- is never
    // built for this read.  The same entries, so the same X: no value moves.
    const bool lean_bases = oc_util::field_diet_on();
    const std::vector<std::vector<FlatMatrix>> no_bases;
    const auto& H_eq_bases_L2 = lean_bases ? no_bases : codomain_lssc_cohomology->get_H_eq_bases_fm();
    const size_t n_bases_L2 = lean_bases ? H_eq_dims_L2.size() : H_eq_bases_L2.size();
    const auto& H_eq_projectors_L1 = domain_lssc_cohomology->get_H_eq_projectors_from_C_eq_flat_fm();

    uint32_t max_q = std::max(
        static_cast<uint32_t>(H_eq_dims_L1.empty() ? 0 : H_eq_dims_L1.size()),
        static_cast<uint32_t>(H_eq_dims_L2.empty() ? 0 : H_eq_dims_L2.size())
    );

    // Size the result vector to hold all possible maps.
    induced_H_eq_maps.assign(max_q, FlatMatrix::CreateZero(0, 0, prime));

    // Loop up to the maximum possible degree.
    for (uint32_t q = 0; q < max_q; ++q) {

        // Add robust boundary checks for all required resources at this degree q.
        if (q >= H_eq_projectors_L1.size() || q >= n_bases_L2) {
            // The map for this degree is trivially zero because either the domain (H^q(L2))
            // or the projection mechanism for the codomain (H^q(L1)) does not exist.
            // The induced_H_eq_maps vector is already initialized with zero matrices,
            // so we can simply skip to the next degree.
            continue;
        }

        uint32_t N_q_K1 = K1->get_num_q_simplices(q);
        uint32_t dim_Hq_L1 = H_eq_dims_L1[q];
        uint32_t dim_Hq_L2 = H_eq_dims_L2[q];

        if (dim_Hq_L2 == 0) {
            induced_H_eq_maps[q] = FlatMatrix::CreateZero(dim_Hq_L1, 0, prime);
            continue;
        }
        if (dim_Hq_L1 == 0) {
            induced_H_eq_maps[q] = FlatMatrix::CreateZero(0, dim_Hq_L2, prime);
            continue;
        }

        // (matrix-engine Phase 3) The projector IS a FlatMatrix now -- the
        // "convert projector to FlatMatrix" copy is gone (audit 02 Sec. 2.2).
        const FlatMatrix& projector_fm = H_eq_projectors_L1[q]; // This access is now safe.

        // 1. Pull back ALL dim_Hq_L2 basis cocycles at once: one block GEMM
        //    Phi^{(q)} * [h2_0 | ... | h2_{k-1}] instead of the old per-column,
        //    per-simplex, per-block extract / multiply / mod_add triple loop.
        //    See pullback_operator()'s header comment for the B6 argument.
        const engine::BlockMat<uint32_t>& Phi = pullback_operator(q);
        const uint32_t out_rows = N_q_K1 * d_V1;   // == Phi.rows()
        const uint32_t src_rows = Phi.cols();      // n_src_blocks * d_V2
        const uint32_t n_src_blocks = Phi.nbc();
        ENGINE_CHECK(Phi.rows() == out_rows);

        std::vector<uint32_t> pullback_buf(static_cast<size_t>(out_rows) * dim_Hq_L2, 0);
        if (out_rows > 0) {
            // The source basis as ONE (src_rows x dim_Hq_L2) row-major operand.
            // Column j is h2_j flattened simplex-major / fiber-minor; rows that
            // h2_j does not reach stay zero, which is exactly what the old
            // `s2_idx < h2_j.get_rows()` guard produced.
            engine::DenseBuf<uint32_t> X(src_rows, dim_Hq_L2);
            if (lean_bases) {
                const engine::BlockMat<uint32_t>& Hb = codomain_lssc_cohomology->H_eq_block(q);
                engine::DenseBuf<uint32_t> sH;
                const engine::DenseView<const uint32_t> Hv = engine::blk::multivector_view<uint32_t>(Hb, sH);
                const uint32_t N_q_K2 = codomain_lssc_cohomology->get_lssc()->get_base_K_complex()->get_num_q_simplices(q);
                ENGINE_CHECK_MSG(Hv.cols() == dim_Hq_L2 && Hv.rows() == N_q_K2 * d_V2,
                                 "compute_induced_maps_on_cohomology: codomain H basis shape");
                const uint32_t avail = std::min<uint32_t>(N_q_K2, n_src_blocks);
                for (uint32_t j_col = 0; j_col < dim_Hq_L2; ++j_col)
                    for (uint32_t s2 = 0; s2 < avail; ++s2)
                        for (uint32_t k = 0; k < d_V2; ++k) X.at(s2 * d_V2 + k, j_col) = Hv.at(s2 * d_V2 + k, j_col);
            } else {
            for (uint32_t j_col = 0; j_col < dim_Hq_L2; ++j_col) {
                const FlatMatrix& h2_j = H_eq_bases_L2[q][j_col];
                const uint32_t avail = std::min<uint32_t>(h2_j.get_rows(), n_src_blocks);
                for (uint32_t s2 = 0; s2 < avail; ++s2) {
                    for (uint32_t k = 0; k < d_V2; ++k) {
                        X.at(s2 * d_V2 + k, j_col) = h2_j.get_entry(s2, k);
                    }
                }
            }
            }
            engine::DenseBuf<uint32_t> Ypb(out_rows, dim_Hq_L2);
            apply_blocks(Phi, engine::cview(X), engine::view(Ypb));
            for (uint32_t r = 0; r < out_rows; ++r) {
                for (uint32_t j_col = 0; j_col < dim_Hq_L2; ++j_col) {
                    pullback_buf[static_cast<size_t>(r) * dim_Hq_L2 + j_col] = Ypb.at(r, j_col);
                }
            }
        }
        const FlatMatrix h_prime_all =
            pullback_buf.empty() ? FlatMatrix::CreateZero(out_rows, dim_Hq_L2, prime)
                                 : FlatMatrix(out_rows, dim_Hq_L2, prime, pullback_buf);

        // 2. Project -- one FlatMatrix multiply for the whole basis (the
        //    projector is applied to the columns independently, so batching is
        //    the same arithmetic column by column).
        if (projector_fm.get_cols() != h_prime_all.get_rows()) {
            std::ostringstream oss;
            oss << "compute_induced_maps_on_cohomology: Dimension mismatch for projector at q=" << q
                << ". Projector_L1_q (" << projector_fm.get_rows() << "x" << projector_fm.get_cols()
                << ") cannot be applied to pullback cochain (" << h_prime_all.get_rows() << "x1).";
            throw std::runtime_error(oss.str());
        }
        const FlatMatrix coords_fm = projector_fm * h_prime_all;

        // 3. The j-th column of coords_fm IS the j-th column of H^q(f_L).
        if (coords_fm.get_rows() != dim_Hq_L1) {
            throw std::runtime_error("Final coordinate vector has wrong dimension.");
        }
        std::vector<uint32_t> H_map_buf(static_cast<size_t>(dim_Hq_L1) * dim_Hq_L2, 0);
        for (uint32_t i_row = 0; i_row < dim_Hq_L1; ++i_row) {
            for (uint32_t j_col = 0; j_col < dim_Hq_L2; ++j_col) {
                H_map_buf[static_cast<size_t>(i_row) * dim_Hq_L2 + j_col] = coords_fm.get_entry(i_row, j_col);
            }
        }
        induced_H_eq_maps[q] =
            H_map_buf.empty() ? FlatMatrix::CreateZero(dim_Hq_L1, dim_Hq_L2, prime)
                              : FlatMatrix(dim_Hq_L1, dim_Hq_L2, prime, H_map_buf);
    }
    maps_computed = true;
}

// Cochain-level application of the block table (Stage S6; consumed by S8's
// omega* . beta* . alpha* == Theta_g cross-check). Flat layout is documented in
// the header: row (s * d_V + k) holds c(sigma_s)_k (simplex-major, fiber-minor).
FlatMatrix EquivariantCochainComplexMorphismOnK::apply_to_flat_cochain(uint32_t q, const FlatMatrix& c2_flat) const {
    if (c2_flat.get_p() != prime) {
        throw std::runtime_error("apply_to_flat_cochain: prime moduli mismatch on the input cochain.");
    }

    const SimplicialComplex* K1 = domain_lssc_cohomology->get_lssc()->get_base_K_complex();
    uint32_t N_q_K1 = K1->get_num_q_simplices(q);

    if (N_q_K1 == 0 || q >= fiber_blocks_.size()) {
        return FlatMatrix(N_q_K1 * d_V1, 1, prime);
    }

    // (matrix-engine Phase 3 Stage B) One block GEMV against the same cached
    // Phi^{(q)} that M1 uses -- same operator, same layout, same contract B6
    // argument (header). The old per-block d_V2-extract / d_V1-accumulate loop
    // is gone.
    const engine::BlockMat<uint32_t>& Phi = pullback_operator(q);
    const uint32_t out_rows = N_q_K1 * d_V1;
    const uint32_t src_rows = Phi.cols();
    ENGINE_CHECK(Phi.rows() == out_rows);
    if (out_rows == 0) return FlatMatrix(out_rows, 1, prime);

    // Zero-extend / truncate the input to Phi's column space: rows the source
    // cochain does not reach are zero, exactly as the old `r < in_rows` guard.
    const uint32_t in_rows = c2_flat.get_rows();
    engine::DenseBuf<uint32_t> X(src_rows, 1);
    for (uint32_t r = 0; r < src_rows && r < in_rows; ++r) X.at(r, 0) = c2_flat.get_entry(r, 0);

    engine::DenseBuf<uint32_t> Y(out_rows, 1);
    apply_blocks(Phi, engine::cview(X), engine::view(Y));

    std::vector<uint32_t> out_buf(out_rows, 0);
    for (uint32_t r = 0; r < out_rows; ++r) out_buf[r] = Y.at(r, 0);
    return FlatMatrix(out_rows, 1, prime, out_buf);
}

// ---------------------------------------------------------------------------
// Y = Phi * X for a dense multi-vector X (matrix-engine Phase 3 Stage B).
//
// This is blk::gemm_dense's job, and for a wide X (M1, which pulls back the
// whole H^q basis at once) that is exactly what it delegates to. The reason it
// is not an unconditional delegation is the SPARSE-OPERAND case: S8's
// cross-check drives apply_to_flat_cochain one STANDARD BASIS VECTOR at a time
// (HeckeCrossChecks.cpp:427/474 apply the composite to e_b), so nearly every
// block's operand slice is identically zero. The per-cell code this replaced
// short-circuited those for free -- its d_V2 x 1 operand stayed a scalar-zero
// FlatMatrix and the product returned a zero matrix without touching data --
// whereas a dense GEMM per block does the full d_V1 x d_V2 work regardless,
// which measured +13 % on `--bench hecke 64 --cross-checks`. So: iterate the
// block pattern (never a cell), skip a block whose operand slice is all zero
// (exact -- 0 contributes nothing), and for a single column go straight into
// the destination with gemv instead of gemm-into-a-temporary plus axpy.
//
// Bit-identity is unaffected: the arithmetic is the same exact products fed to
// the same kernels, and dropping a term that is identically zero cannot move a
// bit (the same argument FlatMatrix::operator*'s Phase-1 zero-skip rests on).
void EquivariantCochainComplexMorphismOnK::apply_blocks(
    const engine::BlockMat<uint32_t>& Phi, engine::DenseView<const uint32_t> X,
    engine::DenseView<uint32_t> Y) const {
    using engine::kernels::GemmOp;
    const engine::ModCtx<uint32_t>& ctx = Phi.ctx();
    const uint32_t w = X.cols();
    ENGINE_CHECK(X.rows() == Phi.cols() && Y.rows() == Phi.rows() && Y.cols() == w);
    for (uint32_t r = 0; r < Y.rows(); ++r)
        for (uint32_t c = 0; c < w; ++c) Y.at(r, c) = 0;
    if (w == 0) return;

    for (uint32_t br = 0; br < Phi.nbr(); ++br) {
        const uint32_t dr = Phi.row_dim(br);
        if (dr == 0) continue;
        engine::DenseView<uint32_t> y = Y.sub(Phi.row_start(br), dr, 0, w);
        for (uint32_t k = Phi.row_begin(br); k < Phi.row_end(br); ++k) {
            const uint32_t bc = Phi.col_at(k);
            const uint32_t dc = Phi.col_dim(bc);
            if (dc == 0) continue;
            const engine::DenseView<const uint32_t> x = X.sub(Phi.col_start(bc), dc, 0, w);
            bool x_is_zero = true;
            for (uint32_t i = 0; i < dc && x_is_zero; ++i)
                for (uint32_t c = 0; c < w; ++c)
                    if (x.at(i, c) != 0) { x_is_zero = false; break; }
            if (x_is_zero) continue;

            const engine::BlockRef<uint32_t>& ref = Phi.ref_at(k);
            if (ref.alpha != 0) {                       // the alpha*I part
                ENGINE_CHECK(dr == dc);
                engine::kernels::axpy<uint32_t>(ctx, ref.alpha, x, y, /*subtract=*/false);
            }
            if (ref.beta != 0) {                        // the beta*M part
                const engine::DenseView<const uint32_t> M = Phi.data_view(ref, dr, dc);
                if (ref.beta == 1) {
                    if (w == 1) engine::kernels::gemv<uint32_t>(ctx, M, x, y, GemmOp::Add);
                    else        engine::kernels::gemm<uint32_t>(ctx, M, x, y, GemmOp::Add);
                } else {
                    engine::DenseBuf<uint32_t> t(dr, w);
                    if (w == 1) engine::kernels::gemv<uint32_t>(ctx, M, x, engine::view(t), GemmOp::Set);
                    else        engine::kernels::gemm<uint32_t>(ctx, M, x, engine::view(t), GemmOp::Set);
                    engine::kernels::axpy<uint32_t>(ctx, ref.beta, engine::cview(t), y, false);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The block-sparse pullback operator Phi^{(q)} (matrix-engine Phase 3 Stage B).
// Contract and bit-identity argument: see the header.
// ---------------------------------------------------------------------------
const engine::BlockMat<uint32_t>&
EquivariantCochainComplexMorphismOnK::pullback_operator(uint32_t q) const {
    if (pullback_built_.size() <= q) {
        pullback_built_.resize(static_cast<size_t>(q) + 1, 0);
        pullback_ops_.resize(static_cast<size_t>(q) + 1);
    }
    if (pullback_built_[q]) return pullback_ops_[q];
    if (!block_pool_) block_pool_ = std::make_shared<engine::BlockPool<uint32_t>>();

    const SimplicialComplex* K1 = domain_lssc_cohomology->get_lssc()->get_base_K_complex();
    const SimplicialComplex* K2 = codomain_lssc_cohomology->get_lssc()->get_base_K_complex();

    const uint32_t N_q_K1 = K1->get_num_q_simplices(q);
    uint32_t n_src = (q < K2->get_simplices().size()) ? K2->get_num_q_simplices(q) : 0u;
    // Widen the source grid to every simplex index the table actually names
    // (a block beyond K2's list then multiplies a zero column, i.e. contributes
    // nothing -- what the old bounds-guarded loop did).
    if (q < fiber_blocks_.size()) {
        for (const std::vector<PerSimplexBlock>& row : fiber_blocks_[q]) {
            for (const PerSimplexBlock& b : row) {
                if (b.s2_idx_in_K2 + 1u > n_src) n_src = b.s2_idx_in_K2 + 1u;
            }
        }
    }

    const std::vector<uint32_t> row_dims(N_q_K1, d_V1);
    const std::vector<uint32_t> col_dims(n_src, d_V2);
    engine::BlockBuilder<uint32_t> bb(block_pool_, row_dims, col_dims, prime);

    if (q < fiber_blocks_.size() && d_V1 > 0 && d_V2 > 0) {
        const uint32_t n_rows_in_table =
            std::min<uint32_t>(N_q_K1, static_cast<uint32_t>(fiber_blocks_[q].size()));
        for (uint32_t s1 = 0; s1 < n_rows_in_table; ++s1) {
            for (const PerSimplexBlock& b : fiber_blocks_[q][s1]) {
                ENGINE_CHECK(b.s2_idx_in_K2 < n_src);
                b.F.populate_flat_data();
                const std::vector<uint32_t>& fd = b.F.get_flat_data_unsafe();
                ENGINE_CHECK(fd.size() == static_cast<size_t>(d_V1) * d_V2);
                engine::DenseBuf<uint32_t> Fbuf(const_cast<uint32_t*>(fd.data()), fd.size(),
                                                d_V1, d_V2, engine::Layout::RowMajor);
                // accumulate(): two table entries naming the SAME (s1, s2) sum,
                // which is what the old loop's repeated mod_add did.
                bb.accumulate(s1, b.s2_idx_in_K2,
                              bb.intern_dense(engine::cview(Fbuf), engine::Layout::RowMajor, true));
            }
        }
    }

    pullback_ops_[q] = bb.finalize();
    pullback_built_[q] = 1;
    return pullback_ops_[q];
}

// (slot-9 WP-4) The pullback as a dense matrix mod ell^k, from the SAME fibre
// block table the field path multiplies by (block row s1: sum of b.F at b.s2).
engine::DenseBuf<uint32_t> EquivariantCochainComplexMorphismOnK::pullback_matrix_module(uint32_t q) const {
    if (ring_k_ < 2) throw std::logic_error("pullback_matrix_module: a Z/ell^k accessor, but this morphism is over F_ell");
    const engine::ModCtx<uint32_t>& ctx = domain_lssc_cohomology->ring_context();
    const SimplicialComplex* K1 = domain_lssc_cohomology->get_lssc()->get_base_K_complex();
    const SimplicialComplex* K2 = codomain_lssc_cohomology->get_lssc()->get_base_K_complex();
    const uint32_t N1 = (q < K1->get_simplices().size()) ? K1->get_num_q_simplices(q) : 0u;
    const uint32_t N2 = (q < K2->get_simplices().size()) ? K2->get_num_q_simplices(q) : 0u;
    engine::DenseBuf<uint32_t> P(N1 * d_V1, N2 * d_V2);
    if (q >= fiber_blocks_.size()) return P;
    ENGINE_CHECK_MSG(fiber_blocks_[q].size() == N1 || fiber_blocks_[q].empty(), "pullback_matrix_module: table height");
    for (uint32_t s1 = 0; s1 < fiber_blocks_[q].size(); ++s1)
        for (const PerSimplexBlock& b : fiber_blocks_[q][s1]) {
            if (b.s2_idx_in_K2 >= N2) continue;   // the field path's guarded-loop semantics
            ENGINE_CHECK_MSG(b.F.get_rows() == d_V1 && b.F.get_cols() == d_V2 && b.F.get_p() == ctx.p,
                             "pullback_matrix_module: a fibre block is not d_V1 x d_V2 over ell^k");
            for (uint32_t a = 0; a < d_V1; ++a)
                for (uint32_t c = 0; c < d_V2; ++c) {
                    const uint32_t v = b.F.get_entry(a, c);
                    if (v == 0) continue;
                    uint32_t& e = P.at(s1 * d_V1 + a, b.s2_idx_in_K2 * d_V2 + c);
                    e = ctx.add(e, v);
                }
        }
    return P;
}

void EquivariantCochainComplexMorphismOnK::compute_induced_maps_module() {
    const engine::ModCtx<uint32_t>& ctx = domain_lssc_cohomology->ring_context();
    const uint32_t nq = std::min(domain_lssc_cohomology->num_degrees(), codomain_lssc_cohomology->num_degrees());
    induced_module_.assign(nq, engine::DenseBuf<uint32_t>());
    for (uint32_t q = 0; q < nq; ++q) {
        const engine::DenseBuf<uint32_t>& Hb = codomain_lssc_cohomology->H_module_basis(q);
        const uint32_t t1 = static_cast<uint32_t>(domain_lssc_cohomology->H_invariant_factors(q).size());
        engine::DenseBuf<uint32_t> M(t1, Hb.cols);
        if (Hb.cols > 0) {
            const engine::DenseBuf<uint32_t> P = pullback_matrix_module(q);
            ENGINE_CHECK_MSG(P.cols == Hb.rows, "compute_induced_maps_module: pullback width != the codomain's cochains");
            engine::DenseBuf<uint32_t> Y(P.rows, Hb.cols);
            for (uint32_t i = 0; i < P.rows; ++i)
                for (uint32_t l = 0; l < P.cols; ++l) {
                    const uint32_t pv = P.at(i, l);
                    if (pv == 0) continue;
                    for (uint32_t j = 0; j < Hb.cols; ++j)
                        if (Hb.at(l, j) != 0) Y.at(i, j) = ctx.add(Y.at(i, j), ctx.mul(pv, Hb.at(l, j)));
                }
            const auto co = domain_lssc_cohomology->H_class_coordinates(q, engine::cview(Y));
            for (uint32_t j = 0; j < Hb.cols; ++j)
                for (uint32_t i = 0; i < t1; ++i) M.at(i, j) = co[j][i];
        }
        induced_module_[q] = std::move(M);
    }
}

const std::vector<engine::DenseBuf<uint32_t>>& EquivariantCochainComplexMorphismOnK::get_induced_H_module_maps() const {
    if (ring_k_ < 2) throw std::logic_error("get_induced_H_module_maps: a Z/ell^k accessor, but this morphism is over F_ell");
    if (!maps_computed) throw std::runtime_error("get_induced_H_module_maps: Maps not computed yet.");
    return induced_module_;
}

const std::vector<FlatMatrix>& EquivariantCochainComplexMorphismOnK::get_induced_H_eq_maps() const {
    if (ring_k_ >= 2)
        throw std::logic_error("get_induced_H_eq_maps: a field (F_ell) accessor, but this morphism is over " +
                               hecke::CoefficientRingText(ring_k_) + " -- use get_induced_H_module_maps() ([E] sec:ring (b1))");
    if (!maps_computed) {
        throw std::runtime_error("get_induced_H_eq_maps: Maps not computed yet.");
    }
    return induced_H_eq_maps;
}

std::string EquivariantCochainComplexMorphismOnK::to_string() const {
    return to_string_offset(0);
}

std::string EquivariantCochainComplexMorphismOnK::to_string_offset(const uint32_t offset) const {
    std::ostringstream oss;
    std::string indent = offset_string(offset);
    std::string indent2 = offset_string(offset + 2);
    std::string indent4 = offset_string(offset + 4);

    oss << indent << "Equivariant Cochain Complex Morphism (on K):\n";
    oss << indent2 << "Domain LSSC (target for H* map):\n" << indent4
        << (domain_lssc_cohomology ? domain_lssc_cohomology->get_lssc()->get_base_K_complex()->get_name() : "null")
        << " (V1_dim=" << d_V1 << ")\n";
    oss << indent2 << "Codomain LSSC (source for H* map):\n" << indent4
        << (codomain_lssc_cohomology ? codomain_lssc_cohomology->get_lssc()->get_base_K_complex()->get_name() : "null")
        << " (V2_dim=" << d_V2 << ")\n";
    oss << indent2 << "Simplicial Inclusion K1 into K2:\n"
        << (simplicial_inclusion_K1_into_K2 ? simplicial_inclusion_K1_into_K2->to_string_offset(offset + 4) : indent4 + "null\n");
    oss << indent2 << "Fiber Map F: V2 -> V1 (dim " << fiber_map_V2_to_V1.get_rows() << "x" << fiber_map_V2_to_V1.get_cols() << ").\n";
    // oss << fiber_map_V2_to_V1.to_string_offset(offset + 4);
    if (!label_.empty()) {
        oss << indent2 << "Block-table label: " << label_ << "\n";
    }

    if (maps_computed) {
        oss << indent2 << "Induced Maps on Equivariant Cohomology H*_eq(L2) -> H*_eq(L1):\n";
        for (uint32_t q = 0; q < induced_H_eq_maps.size(); ++q) {
            oss << indent4 << "H^" << q << " map (dim "
                << induced_H_eq_maps[q].get_rows() << "x" << induced_H_eq_maps[q].get_cols() << "):\n"
                << induced_H_eq_maps[q].to_string_offset(offset + 6);
        }
    }
    else {
        oss << indent2 << "Induced maps on cohomology not yet computed.\n";
    }
    return oss.str();
}
