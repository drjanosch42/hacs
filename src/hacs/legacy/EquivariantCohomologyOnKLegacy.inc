// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// src/hacs/legacy/EquivariantCohomologyOnKLegacy.inc -- the LEGACY arm of compute_cohomology():
// the pre-Stage-B path (BlockFlatMatrix::flat_kernel + factor_basis_block + the flat/unflat/expand loops),
// compute_cohomology_legacy, adopt_legacy_result and the two legacy-arm-only assembly helpers
// (cochains_as_multivector, projector_as_block).  Run by ENGINE_BACKEND=legacy and
// EquivariantCohomologyOptions::ScopedLegacyCohomologyPath, and compared by the ENGINE_SHADOW_LEGACY build.
// Moved verbatim out of EquivariantCohomologyOnK.cpp by S24 CL-L2 (CLEANUP plan Sec. 2.2, C-12 step 2; owner ruling
// 1 of 2026-10-30).  Included by EquivariantCohomologyOnK.cpp (after the block arm, before the shadow .inc and the
// explicit instantiations) under `#if OC_ENABLE_LEGACY`; declared in EquivariantCohomologyOnK.h under the same guard.
// It prints the `[TIMING] A_Cq/A_Zq kernel` lines; an OFF build has none of it.
#if !OC_ENABLE_LEGACY
#error "EquivariantCohomologyOnKLegacy.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

#include "../../ModArithmetic/legacy/BlockFlatMatrix.h"

namespace {

// Inverse direction (legacy arm only): a list of N_q x d_V cochains as one
// column-major multi-vector over the uniform partition.
template<class Elt>
BlockMat<Elt> cochains_as_multivector(const std::shared_ptr<BlockPool<Elt>>& pool,
                                      const std::vector<FlatMatrixT<Elt>>& list, uint32_t N_q, uint32_t d, Elt p) {
    const std::vector<uint32_t> row_dims(N_q, d);
    const uint32_t dim = N_q * d, k = static_cast<uint32_t>(list.size());
    DenseBuf<Elt> buf(dim, k, Layout::ColMajor);
    for (uint32_t t = 0; t < k; ++t) {
        ENGINE_CHECK(list[t].get_rows() * list[t].get_cols() == dim);
        if (dim == 0) continue;
        list[t].populate_flat_data();
        std::memcpy(buf.data() + static_cast<size_t>(t) * buf.ld, list[t].get_flat_data_unsafe().data(),
                    static_cast<size_t>(dim) * sizeof(Elt));
    }
    return BlockMat<Elt>::from_flat(pool, std::move(buf), row_dims, p, /*detect=*/false);
}

template<class Elt>
BlockMat<Elt> projector_as_block(const std::shared_ptr<BlockPool<Elt>>& pool, const FlatMatrixT<Elt>& P,
                                 uint32_t N_q, uint32_t d, Elt p) {
    const uint32_t ambient = N_q * d;
    const uint32_t dim_H = P.get_rows();
    ENGINE_CHECK(P.get_cols() == ambient || (P.get_rows() == 0 && P.get_cols() == 0));
    DenseBuf<Elt> buf(dim_H, ambient, Layout::RowMajor);
    if (dim_H != 0 && ambient != 0) {
        P.populate_flat_data();
        std::memcpy(buf.data(), P.get_flat_data_unsafe().data(), static_cast<size_t>(dim_H) * ambient * sizeof(Elt));
    }
    return BlockMat<Elt>::from_dense(pool, std::move(buf), std::vector<uint32_t>{dim_H},
                                     std::vector<uint32_t>(N_q, d), p, /*detect=*/true);
}

} // namespace

// ===========================================================================
// THE LEGACY ARM (pre-Stage-B compute_cohomology, kept for ENGINE_BACKEND=legacy
// and the ENGINE_SHADOW_LEGACY comparison).  BlockFlatMatrix::flat_kernel and
// factor_basis_block themselves honour engine::elim::use_legacy(), so under
// ENGINE_BACKEND=legacy this is the pre-Phase-2 substrate end to end.  The
// scalar arithmetic of the three assembly helpers goes through engine::ModCtx
// (bit-identical to the mod_*_w primitives it replaces).
// ===========================================================================
namespace {

// Reshape an r x c FlatMatrix into the (r*c) x 1 flat column of layout L1 --
// row-major, so the buffer is literally the same bytes.
template<class Elt>
FlatMatrixT<Elt> legacy_reshape_to_flat_column(const FlatMatrixT<Elt>& m, Elt prime) {
    // (slot-14 WP-3, G1/H4) the product in 64 bits; `total` is a uint32 row
    // count, so rows*cols >= 2^32 is refused loudly instead of wrapping.
    const uint64_t total64 = static_cast<uint64_t>(m.get_rows()) * m.get_cols();
    ENGINE_CHECK_MSG(total64 <= UINT32_MAX, "legacy_reshape_to_flat_column: rows*cols exceeds the uint32 row domain");
    const uint32_t total = static_cast<uint32_t>(total64);
    Elt sv = 0;
    if (m.is_currently_scalar(sv)) {
        if (sv == 0) return FlatMatrixT<Elt>::CreateZero(total, 1, prime);
        std::vector<Elt> buf(total);
        const uint32_t c_n = m.get_cols();
        for (uint32_t r = 0; r < m.get_rows(); ++r)
            for (uint32_t c = 0; c < c_n; ++c)
                buf[static_cast<size_t>(r) * c_n + c] = m.get_entry(r, c);
        return FlatMatrixT<Elt>(total, 1, prime, buf);
    }
    m.populate_flat_data();
    return FlatMatrixT<Elt>(total, 1, prime, m.get_flat_data_unsafe());
}

// Unflatten basis vectors back to cochain matrices (N_q x d_V, row per simplex).
template<class Elt>
void legacy_unflatten_and_store_results(
    uint32_t q,
    const std::vector<FlatMatrixT<Elt>>& flat_basis,
    std::vector<FlatMatrixT<Elt>>& target_basis_storage,
    const SimplicialComplex* K,
    uint32_t representation_dim,
    Elt prime)
{
    target_basis_storage.clear();
    uint32_t N_q = K->get_num_q_simplices(q);

    for (const auto& flat_vec : flat_basis) {
        if (flat_vec.get_rows() == 0 && flat_vec.get_cols() == 1) {
            if (N_q == 0) {
                target_basis_storage.push_back(FlatMatrixT<Elt>::CreateZero(0, representation_dim, prime));
            }
            else {
                target_basis_storage.push_back(FlatMatrixT<Elt>::CreateZero(N_q, representation_dim, prime));
            }
        }
        else if (flat_vec.get_rows() > 0) {
            const size_t total = static_cast<size_t>(N_q) * representation_dim;
            std::vector<Elt> buf(total, 0);
            for (uint32_t r = 0; r < N_q; ++r)
                for (uint32_t c = 0; c < representation_dim; ++c)
                    buf[static_cast<size_t>(r) * representation_dim + c] =
                        flat_vec.get_entry(r * representation_dim + c, 0);
            target_basis_storage.push_back(
                total == 0 ? FlatMatrixT<Elt>::CreateZero(N_q, representation_dim, prime)
                           : FlatMatrixT<Elt>(N_q, representation_dim, prime, buf));
        }
    }
}

// Takes solutions for vector coefficients and expands them into full cochains.
template<class Elt>
std::vector<FlatMatrixT<Elt>> legacy_expand_coeffs_to_cochain_basis(
    uint32_t q,
    const std::vector<FlatMatrixT<Elt>>& solution_vectors_for_v_coeffs,
    const std::vector<FlatMatrixT<Elt>>& basis_Z1_q_K_Fp,
    const SimplicialComplex* K,
    uint32_t representation_dim,
    Elt prime)
{
    const engine::ModCtx<Elt> ctx(prime);
    std::vector<FlatMatrixT<Elt>> expanded_basis;
    uint32_t N_q = K->get_num_q_simplices(q);
    uint32_t k_q = static_cast<uint32_t>(basis_Z1_q_K_Fp.size());

    if (solution_vectors_for_v_coeffs.empty() || k_q == 0) {
        return expanded_basis;
    }

    for (const auto& sol_v_coeffs_vector : solution_vectors_for_v_coeffs) {
        std::vector<Elt> cochain_buf(static_cast<size_t>(N_q) * representation_dim, 0);

        for (uint32_t simplex_idx = 0; simplex_idx < N_q; ++simplex_idx) {
            std::vector<Elt> val_at_simplex_s(representation_dim, 0);

            for (uint32_t j = 0; j < k_q; ++j) {
                Elt beta_j_at_s = basis_Z1_q_K_Fp[j].get_entry(simplex_idx, 0);
                if (beta_j_at_s == 0) continue;

                for (uint32_t dv_comp = 0; dv_comp < representation_dim; ++dv_comp) {
                    Elt v_j_comp_val = sol_v_coeffs_vector.get_entry(j * representation_dim + dv_comp, 0);
                    Elt term = ctx.mul(beta_j_at_s, v_j_comp_val);
                    val_at_simplex_s[dv_comp] = ctx.add(val_at_simplex_s[dv_comp], term);
                }
            }
            for (uint32_t dv_comp = 0; dv_comp < representation_dim; ++dv_comp) {
                cochain_buf[static_cast<size_t>(simplex_idx) * representation_dim + dv_comp] =
                    val_at_simplex_s[dv_comp];
            }
        }
        expanded_basis.push_back(
            cochain_buf.empty() ? FlatMatrixT<Elt>::CreateZero(N_q, representation_dim, prime)
                                : FlatMatrixT<Elt>(N_q, representation_dim, prime, cochain_buf));
    }
    return expanded_basis;
}

template<class Elt>
FlatMatrixT<Elt> legacy_identity_minus_matrix_flat(const FlatMatrixT<Elt>& M, uint32_t d, Elt prime) {
    const engine::ModCtx<Elt> ctx(prime);
    std::vector<Elt> buf(static_cast<size_t>(d) * d);
    for (uint32_t r = 0; r < d; ++r) {
        for (uint32_t c = 0; c < d; ++c) {
            Elt v = ctx.sub(static_cast<Elt>(r == c ? 1 : 0), M.get_entry(r, c));
            buf[static_cast<size_t>(r) * d + c] = v;
        }
    }
    return FlatMatrixT<Elt>(d, d, prime, buf);
}

template<class Elt>
FlatMatrixT<Elt> legacy_negated_matrix_flat(const FlatMatrixT<Elt>& M, uint32_t d, Elt prime) {
    const engine::ModCtx<Elt> ctx(prime);
    std::vector<Elt> buf(static_cast<size_t>(d) * d);
    bool all_zero = true;
    for (uint32_t r = 0; r < d; ++r) {
        for (uint32_t c = 0; c < d; ++c) {
            Elt v = ctx.neg(M.get_entry(r, c));
            buf[static_cast<size_t>(r) * d + c] = v;
            if (v != 0) all_zero = false;
        }
    }
    if (all_zero) return FlatMatrixT<Elt>::CreateZero(d, d, prime);
    return FlatMatrixT<Elt>(d, d, prime, buf);
}

} // namespace

template<class Elt>
void EquivariantCohomologyOnKT<Elt>::compute_cohomology_legacy(FlatResult& out) const {
    const SimplicialComplex* K = lssc->get_base_K_complex();
    const auto& user_relations = effective_relations();
    const engine::ModCtx<Elt> ctx(prime);
    uint32_t max_dim_K = K->get_dimension();
    uint32_t report_bound = (max_dim_K == static_cast<uint32_t>(-1)) ? 0 : max_dim_K + 1;

    out = FlatResult();
    out.Z.assign(report_bound, {});
    out.B.assign(report_bound, {});
    out.H.assign(report_bound, {});
    out.dims.assign(report_bound, 0);
    out.C.assign(report_bound, {});
    out.P.resize(report_bound);

    // --- Pre-compute the bases for all C^q_eq spaces (legacy compute_C_eq_bases).
    std::vector<std::vector<FlatMatrixT<Elt>>> C_eq_bases_flat(report_bound + 1);
    for (uint32_t q = 0; q < report_bound; ++q) {
        uint32_t N_q = K->get_num_q_simplices(q);
        uint32_t num_vars_Cq = N_q * representation_dim;

        if (num_vars_Cq == 0) {
            C_eq_bases_flat[q] = {};
            continue;
        }

        const std::vector<const Relation<Elt>*> live = live_relations<Elt>(user_relations, q, representation_dim);
        uint32_t N_rel_q = static_cast<uint32_t>(live.size());

        if (N_rel_q == 0) {
            C_eq_bases_flat[q].clear();
            for (uint32_t i = 0; i < num_vars_Cq; ++i) {
                std::vector<Elt> e_i(num_vars_Cq, 0);
                e_i[i] = 1;
                C_eq_bases_flat[q].push_back(FlatMatrixT<Elt>(num_vars_Cq, 1, prime, e_i));
            }
        } else {
            std::vector<uint32_t> block_row_dims(N_rel_q, representation_dim);
            std::vector<uint32_t> block_col_dims(N_q, representation_dim);
            BlockFlatMatrixT<Elt> A_Cq_block(block_row_dims, block_col_dims, prime);

            uint32_t rel_idx = 0;
            for (const auto* rel_ptr : live) {
                const auto& rel = *rel_ptr;
                if (rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq) {
                    A_Cq_block.set_block_entry(rel_idx, rel.s_lhs_idx_in_Kq,
                        legacy_identity_minus_matrix_flat<Elt>(rel.M, representation_dim, prime));
                } else {
                    FlatMatrixT<Elt> I_dv(representation_dim, prime);
                    A_Cq_block.set_block_entry(rel_idx, rel.s_lhs_idx_in_Kq, I_dv);
                    A_Cq_block.set_block_entry(rel_idx, rel.s_rhs_idx_in_Kq,
                        legacy_negated_matrix_flat<Elt>(rel.M, representation_dim, prime));
                }
                rel_idx++;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            auto block_kernel_vecs = A_Cq_block.flat_kernel();
            auto t1 = std::chrono::high_resolution_clock::now();
            std::cout << "[TIMING] A_Cq kernel q=" << q << ": "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                      << " ms" << std::endl;

            C_eq_bases_flat[q].clear();
            C_eq_bases_flat[q].reserve(block_kernel_vecs.size());
            for (auto& bk_vec : block_kernel_vecs) {
                C_eq_bases_flat[q].push_back(bk_vec.to_flat_matrix());
            }
        }
    }
    for (uint32_t q = 0; q < report_bound; ++q) {
        legacy_unflatten_and_store_results<Elt>(q, C_eq_bases_flat[q], out.C[q], K, representation_dim, prime);
    }

    // This loop computes B^q, then Z^q, then H^q for each degree q.
    for (uint32_t q = 0; q < report_bound; ++q) {

        // === Step 1: Compute Z^q_eq (Cocycles) ===
        const auto& basis_Z1_q = K->template get_trivial_cocycle_basis_Z1_elt<Elt>(q);
        uint32_t k_q = static_cast<uint32_t>(basis_Z1_q.size());
        if (k_q == 0) {
            out.Z[q] = {};
        }
        else {
            uint32_t N_q = K->get_num_q_simplices(q);
            uint32_t num_vars_Zq = k_q * representation_dim;

            const std::vector<const Relation<Elt>*> live_relations_Z =
                live_relations<Elt>(user_relations, q, representation_dim);
            uint32_t N_rel_q = static_cast<uint32_t>(live_relations_Z.size());

            std::vector<FlatMatrixT<Elt>> sol_v_coeffs;

            if (N_rel_q == 0) {
                for (uint32_t i = 0; i < num_vars_Zq; ++i) {
                    std::vector<Elt> e_i(num_vars_Zq, 0);
                    e_i[i] = 1;
                    sol_v_coeffs.push_back(FlatMatrixT<Elt>(num_vars_Zq, 1, prime, e_i));
                }
            } else {
                std::vector<uint32_t> block_row_dims(N_rel_q, representation_dim);
                std::vector<uint32_t> block_col_dims(k_q, representation_dim);
                BlockFlatMatrixT<Elt> A_Zq_block(block_row_dims, block_col_dims, prime);

                uint32_t rel_idx = 0;
                for (const auto* rel_ptr_Z : live_relations_Z) {
                    const auto& rel = *rel_ptr_Z;
                    for (uint32_t j_idx = 0; j_idx < k_q; ++j_idx) {
                        Elt beta_lhs = (N_q > 0) ? basis_Z1_q[j_idx].get_entry(rel.s_lhs_idx_in_Kq, 0) : Elt(0);
                        Elt beta_rhs = (N_q > 0) ? basis_Z1_q[j_idx].get_entry(rel.s_rhs_idx_in_Kq, 0) : Elt(0);

                        if (beta_lhs == 0 && beta_rhs == 0) {
                            continue;
                        }

                        FlatMatrixT<Elt> block = FlatMatrixT<Elt>::CreateZero(representation_dim, representation_dim, prime);
                        if (beta_rhs == 0) {
                            block = FlatMatrixT<Elt>::CreateScalar(representation_dim, beta_lhs, prime);
                        } else {
                            std::vector<Elt> buf(static_cast<size_t>(representation_dim) * representation_dim);
                            bool all_zero = true;
                            for (uint32_t r = 0; r < representation_dim; ++r) {
                                for (uint32_t c = 0; c < representation_dim; ++c) {
                                    Elt term = ctx.mul(rel.M.get_entry(r, c), beta_rhs);
                                    Elt cur = (r == c) ? beta_lhs : Elt(0);
                                    Elt v = ctx.sub(cur, term);
                                    buf[static_cast<size_t>(r) * representation_dim + c] = v;
                                    if (v != 0) all_zero = false;
                                }
                            }
                            if (!all_zero) block = FlatMatrixT<Elt>(representation_dim, representation_dim, prime, buf);
                        }
                        A_Zq_block.set_block_entry(rel_idx, j_idx, block);
                    }
                    rel_idx++;
                }

                auto t0 = std::chrono::high_resolution_clock::now();
                auto block_kernel_vecs = A_Zq_block.flat_kernel();
                auto t1 = std::chrono::high_resolution_clock::now();
                std::cout << "[TIMING] A_Zq kernel q=" << q << ": "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                          << " ms" << std::endl;

                sol_v_coeffs.reserve(sol_v_coeffs.size() + block_kernel_vecs.size());
                for (auto& bk_vec : block_kernel_vecs) {
                    sol_v_coeffs.push_back(bk_vec.to_flat_matrix());
                }
            }
            out.Z[q] = legacy_expand_coeffs_to_cochain_basis<Elt>(q, sol_v_coeffs, basis_Z1_q, K, representation_dim, prime);
        }

        // === Step 2: Compute B^q_eq (Coboundaries) ===
        if (q > 0) {
            uint32_t N_qm1 = K->get_num_q_simplices(q - 1);
            std::vector<uint32_t> block_row_dims_qm1(N_qm1, representation_dim);

            std::vector<FlatMatrixT<Elt>> Z_qm1_flat_fm;
            Z_qm1_flat_fm.reserve(out.Z[q - 1].size());
            for (const auto& m : out.Z[q - 1]) {
                Z_qm1_flat_fm.push_back(legacy_reshape_to_flat_column<Elt>(m, prime));
            }

            const std::vector<FlatMatrixT<Elt>>& C_qm1_flat_fm = C_eq_bases_flat[q - 1];

            std::vector<FlatMatrixT<Elt>> W_qm1_flat_fm;
            if (!C_qm1_flat_fm.empty()) {
                auto [W_projector, W_pivots_out] = factor_basis_block<Elt>(
                    block_row_dims_qm1, Z_qm1_flat_fm, C_qm1_flat_fm, &W_qm1_flat_fm, prime);
                (void)W_projector; (void)W_pivots_out;
            }

            BlockFlatMatrixT<Elt> D1_block_qm1 = K->template coboundary_operator_block_elt<Elt>(q - 1, prime, representation_dim);
            uint32_t N_q_target = K->get_num_q_simplices(q);
            for (const auto& w_fm : W_qm1_flat_fm) {
                std::vector<uint32_t> w_block_row_dims(N_qm1, representation_dim);
                std::vector<uint32_t> w_block_col_dims = {1};
                BlockFlatMatrixT<Elt> w_block(w_block_row_dims, w_block_col_dims, prime);
                uint32_t offset = 0;
                for (uint32_t br = 0; br < N_qm1; ++br) {
                    FlatMatrixT<Elt> blk(representation_dim, 1, prime);
                    for (uint32_t r = 0; r < representation_dim; ++r)
                        blk.set_entry(r, 0, w_fm.get_entry(offset + r, 0));
                    w_block.set_block_entry(br, 0, blk);
                    offset += representation_dim;
                }
                BlockFlatMatrixT<Elt> dw_block = D1_block_qm1 * w_block;
                FlatMatrixT<Elt> dw_flat = dw_block.to_flat_matrix();
                const size_t total = static_cast<size_t>(N_q_target) * representation_dim;
                std::vector<Elt> dw_buf(total, 0);
                for (uint32_t r = 0; r < N_q_target; ++r)
                    for (uint32_t c = 0; c < representation_dim; ++c)
                        dw_buf[static_cast<size_t>(r) * representation_dim + c] =
                            dw_flat.get_entry(r * representation_dim + c, 0);
                out.B[q].push_back(
                    total == 0 ? FlatMatrixT<Elt>::CreateZero(N_q_target, representation_dim, prime)
                               : FlatMatrixT<Elt>(N_q_target, representation_dim, prime, dw_buf));
            }
        }

        // === Step 3: Compute H^q_eq and the Projector ===
        uint32_t N_q_step3 = K->get_num_q_simplices(q);
        std::vector<uint32_t> block_row_dims_q(N_q_step3, representation_dim);

        std::vector<FlatMatrixT<Elt>> Z_q_flat_fm, B_q_flat_fm;
        for (const auto& m : out.Z[q]) Z_q_flat_fm.push_back(legacy_reshape_to_flat_column<Elt>(m, prime));
        for (const auto& m : out.B[q]) B_q_flat_fm.push_back(legacy_reshape_to_flat_column<Elt>(m, prime));

        std::vector<FlatMatrixT<Elt>> H_q_flat_fm;

        auto [projector_fm, H_pivots_out] = factor_basis_block<Elt>(
            block_row_dims_q, B_q_flat_fm, Z_q_flat_fm, &H_q_flat_fm, prime);
        (void)H_pivots_out;

        out.P[q] = std::move(projector_fm);

        legacy_unflatten_and_store_results<Elt>(q, H_q_flat_fm, out.H[q], K, representation_dim, prime);
        out.dims[q] = static_cast<uint32_t>(out.H[q].size());
    }
    out.built = true;
}

// Legacy arm: the FlatMatrix result IS the cache; the block objects are
// rebuilt from it (same assembly helpers, from_flat / from_dense) so that the
// *_block accessors work under both backends.
template<class Elt>
void EquivariantCohomologyOnKT<Elt>::adopt_legacy_result(FlatResult&& legacy) {
    const SimplicialComplex* K = lssc->get_base_K_complex();
    const uint32_t nq = static_cast<uint32_t>(legacy.dims.size());
    const uint32_t d = representation_dim;
    const Elt p = prime;
    const auto& rels = effective_relations();

    pool_ = std::make_shared<BlockPool>();
    A_C_block_.assign(nq, BlockMat());
    A_Z_block_.assign(nq, BlockMat());
    C_eq_block_.assign(nq, BlockMat());
    Z_eq_block_.assign(nq, BlockMat());
    B_eq_block_.assign(nq, BlockMat());
    H_eq_block_.assign(nq, BlockMat());
    P_block_.assign(nq, BlockMat());
    for (uint32_t q = 0; q < nq; ++q) {
        const uint32_t N_q = K->get_num_q_simplices(q);
        const std::vector<const Relation<Elt>*> live = live_relations<Elt>(rels, q, d);
        const auto& Z1 = K->template get_trivial_cocycle_basis_Z1_elt<Elt>(q);
        A_C_block_[q] = assemble_A_Cq<Elt>(pool_, live, N_q, d, p);
        A_Z_block_[q] = assemble_A_Zq<Elt>(pool_, live, Z1, N_q, d, p);
        C_eq_block_[q] = cochains_as_multivector<Elt>(pool_, legacy.C[q], N_q, d, p);
        Z_eq_block_[q] = cochains_as_multivector<Elt>(pool_, legacy.Z[q], N_q, d, p);
        B_eq_block_[q] = cochains_as_multivector<Elt>(pool_, legacy.B[q], N_q, d, p);
        H_eq_block_[q] = cochains_as_multivector<Elt>(pool_, legacy.H[q], N_q, d, p);
        P_block_[q] = projector_as_block<Elt>(pool_, legacy.P[q], N_q, d, p);
    }
    H_eq_dimensions = legacy.dims;
    fm_ = std::move(legacy);
    fm_.built = true;
}

