// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// src/hacs/Hecke/legacy/HeckeOperatorOnKLegacy.inc -- the LEGACY (pre-Phase-3) dense field arm of
// HeckeOperatorOnKT::compute(): flatten_cochain_legacy, assemble_theta_matrix_legacy, compute_legacy and
// adopt_legacy_result (the Legacy* types are declared in HeckeOperatorOnK.h under the same guard).  Run by
// ENGINE_BACKEND=legacy and HeckeOperatorOptions::ScopedLegacyHeckeOperatorPath, and compared by the
// ENGINE_SHADOW_LEGACY build.  Moved verbatim out of HeckeOperatorOnK.cpp by S24 CL-L2 (CLEANUP plan Sec. 2.2, C-12
// step 2; owner ruling 1 of 2026-10-30).  Included by HeckeOperatorOnK.cpp (before the shadow .inc and the explicit
// instantiations) under `#if OC_ENABLE_LEGACY`.  An OFF build has none of it.
#if !OC_ENABLE_LEGACY
#error "HeckeOperatorOnKLegacy.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

// ===========================================================================
// THE LEGACY (pre-Phase-3) DENSE ARM.  Preserved verbatim as the
// ENGINE_BACKEND=legacy path and the -DENGINE_SHADOW_LEGACY oracle; the only
// change is that its scalar arithmetic goes through engine::ModCtx, which is
// bit-identical to the mod_* primitives it replaces (ModContextValidation).
// Nothing below runs on the default engine backend.
// ===========================================================================

// Reshape an N_q x d_V cochain into the simplex-major, fiber-minor flat layout
// (index s*d_V + k): the row-major buffer already IS that layout, so this is
// a reshape of the same bytes (populate_flat_data() materialises a
// scalar-flagged cochain -- zero, or alpha*I when N_q == d_V -- exactly as
// get_entry reads it).  No `s*d_V + k` loop: the layout's one implementation
// is BlockMat::to_flat / from_flat (plan 7.20).
template<class Elt>
FlatMatrixT<Elt> HeckeOperatorOnKT<Elt>::flatten_cochain_legacy(const FlatMatrixT<Elt>& cochain) const {
    // (slot-14 WP-3, G1/H4) the product in 64 bits: `total` becomes a uint32
    // row count, so a cochain with rows*cols >= 2^32 is refused loudly
    // instead of wrapping.
    const uint64_t total64 = static_cast<uint64_t>(cochain.get_rows()) * cochain.get_cols();
    ENGINE_CHECK_MSG(total64 <= UINT32_MAX, "flatten_cochain_legacy: rows*cols exceeds the uint32 row domain");
    const uint32_t total = static_cast<uint32_t>(total64);
    if (total == 0) return FlatMatrixT<Elt>::CreateZero(total, 1, ell_);
    cochain.populate_flat_data();
    return FlatMatrixT<Elt>(total, 1, ell_, cochain.get_flat_data_unsafe());
}

// Assembles the full (N_q*d_V)^2 dense matrix of Theta_g on C^q(K;V) in the
// flat layout, per-cell (the pre-Phase-3 assembly).  `fused` selects the
// route; `block_cache` is the item-7.2 memo keyed on (i, exact gamma_k), one
// per route.
template<class Elt>
FlatMatrixT<Elt> HeckeOperatorOnKT<Elt>::assemble_theta_matrix_legacy(
        uint32_t q, bool fused, LegacyBlockCache& block_cache) const {
    const uint32_t N_q = K()->get_num_q_simplices(q);
    const uint32_t dim = N_q * d_V_;
    if (dim == 0) return FlatMatrixT<Elt>(0, 0, ell_);
    const ModCtx<Elt> ctx(ell_);

    std::vector<Elt> data(static_cast<size_t>(dim) * dim, 0);
    auto key_of = [](uint32_t i, const Mat22& gamma) {
        return LegacyProductKey{ static_cast<int64_t>(i), gamma.a, gamma.b, gamma.c, gamma.d };
    };

    for (uint32_t sigma = 0; sigma < N_q; ++sigma) {
        for (uint32_t i = 1; i <= m_; ++i) {
            const std::vector<ChainTerm>& cts = tau_->tau(i, q, sigma);
            if (fused) {
                const FlatMatrixT<Elt>& omega_i = omega_of(i); // rho_n(s_i^{-1}) (in rep_)
                const std::vector<XiTermT<Elt>>& xts = tau_->xi_terms(i, q, sigma);
                ENGINE_CHECK_MSG(xts.size() == cts.size(),
                                 "HeckeOperatorOnK: xi_terms and tau term lists disagree in length.");
                for (size_t k_term = 0; k_term < xts.size(); ++k_term) {
                    const XiTermT<Elt>& xt = xts[k_term];
                    const Elt cmod = coeff_mod(xt.coeff, ell_);
                    if (cmod == 0) continue;
                    const LegacyProductKey key = key_of(i, cts[k_term].gamma);
                    auto it = block_cache.find(key);
                    if (it == block_cache.end()) {
                        // (slot-14 fix, WP-17 section 8.2) The factored pool holds
                        // rho_n (SymN) blocks: read it for a SymN operator only.
                        // A non-SymN fibre sharing the table (E7's op_D) reads its
                        // own rho_pool_dual_ through rho_of_term, as compute_block
                        // does -- before the fix it got the SymN block here.
                        if (rep_ == FibreRepresentation::SymN && tau_->rho_pool_is_low_rank()) {
                            // (S13 WP-16) A tau table built under the U_ell lever
                            // keeps its pool factored (RhoRefT::get() would throw):
                            // materialise the block as U V, exact over F_ell, so it
                            // is the very matrix the dense pool would hold (U V == R
                            // by construction; certification (f) re-derives it).
                            const lowrank::LowRankT<Elt>& lr = tau_->rho_pool_low_rank(xt.rho_g_gamma.index());
                            ENGINE_CHECK_MSG(lr.rows == d_V_ && lr.cols == d_V_,
                                             "HeckeOperatorOnK (legacy arm): factored rho block has the wrong shape.");
                            std::vector<Elt> dense(static_cast<size_t>(lr.rows) * lr.cols, Elt(0));
                            for (uint32_t a = 0; a < lr.rows; ++a)
                                for (uint32_t s = 0; s < lr.t; ++s) {
                                    const Elt u = lr.U[static_cast<size_t>(a) * lr.t + s];
                                    if (u == 0) continue;
                                    const Elt* vrow = &lr.V[static_cast<size_t>(s) * lr.cols];
                                    Elt* drow = &dense[static_cast<size_t>(a) * lr.cols];
                                    for (uint32_t b = 0; b < lr.cols; ++b) drow[b] = ctx.add(drow[b], ctx.mul(u, vrow[b]));
                                }
                            it = block_cache.emplace(key, omega_i * FlatMatrixT<Elt>(lr.rows, lr.cols, ell_, dense)).first;
                        } else {
                            it = block_cache.emplace(key, omega_i * rho_of_term(xt)).first;
                        }
                    }
                    const FlatMatrixT<Elt>& block = it->second;
                    const uint32_t col_base = xt.sigma_idx * d_V_;
                    for (uint32_t r = 0; r < d_V_; ++r) {
                        const size_t row_off = static_cast<size_t>(sigma * d_V_ + r) * dim;
                        for (uint32_t c = 0; c < d_V_; ++c) {
                            Elt& slot = data[row_off + col_base + c];
                            slot = ctx.add(slot, ctx.mul(cmod, block.get_entry(r, c)));
                        }
                    }
                }
            } else {
                const Mat22& delta_i = cosets_->delta(i); // s_i^{-1} g, exact
                for (const ChainTerm& ct : cts) {
                    const Elt cmod = coeff_mod(ct.coeff, ell_);
                    if (cmod == 0) continue;
                    const LegacyProductKey key = key_of(i, ct.gamma);
                    auto it = block_cache.find(key);
                    if (it == block_cache.end()) {
                        it = block_cache.emplace(key, rho_of_exact(delta_i * ct.gamma)).first;
                    }
                    const FlatMatrixT<Elt>& block = it->second;
                    const uint32_t col_base = ct.sigma_idx * d_V_;
                    for (uint32_t r = 0; r < d_V_; ++r) {
                        const size_t row_off = static_cast<size_t>(sigma * d_V_ + r) * dim;
                        for (uint32_t c = 0; c < d_V_; ++c) {
                            Elt& slot = data[row_off + col_base + c];
                            slot = ctx.add(slot, ctx.mul(cmod, block.get_entry(r, c)));
                        }
                    }
                }
            }
        }
    }
    return FlatMatrixT<Elt>(dim, dim, ell_, data);
}

template<class Elt>
void HeckeOperatorOnKT<Elt>::compute_legacy(LegacyResult& out) const {
    const SimplicialComplex* Kp = K();
    const ModCtx<Elt> ctx(ell_);

    std::vector<FlatMatrixT<Elt>> theta_flat(num_degrees_);
    std::vector<FlatMatrixT<Elt>> theta_eq(num_degrees_);
    std::vector<MatrixT<Elt>> A(num_degrees_);

    uint64_t relation_checks = 0;
    uint64_t cocycle_checks = 0;
    uint64_t theta_eq_solves = 0;
    Elt residual_max = 0;

    const auto& H_dims = coh_->get_H_eq_dimensions();
    const auto& H_bases = coh_->get_H_eq_bases_fm();
    const auto& C_eq_bases = coh_->get_C_eq_bases_fm();
    const auto& projectors = coh_->get_H_eq_projectors_from_C_eq_flat_fm();
    if (H_dims.size() < num_degrees_ || H_bases.size() < num_degrees_ ||
        C_eq_bases.size() < num_degrees_ || projectors.size() < num_degrees_) {
        throw std::runtime_error(
            "HeckeOperatorOnK::compute: cohomology object does not report all "
            "3 degrees of the 7-vertex model.");
    }
    const auto& relations = coh_->get_lssc()->get_user_defined_simplex_relations();

    LegacyBlockCache fused_block_cache;
    LegacyBlockCache route2_block_cache;

    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const uint32_t N_q = Kp->get_num_q_simplices(q);
        const uint32_t dim = N_q * d_V_;

        theta_flat[q] = assemble_theta_matrix_legacy(q, /*fused=*/true, fused_block_cache);
        const FlatMatrixT<Elt> theta_route2 =
            assemble_theta_matrix_legacy(q, /*fused=*/false, route2_block_cache);
        theta_flat[q].populate_flat_data();
        theta_route2.populate_flat_data();
        const std::vector<Elt>& fused_data = theta_flat[q].get_flat_data_unsafe();
        const std::vector<Elt>& route2_data = theta_route2.get_flat_data_unsafe();
        const size_t n_entries = static_cast<size_t>(dim) * dim;
        ENGINE_CHECK_MSG(fused_data.size() == n_entries && route2_data.size() == n_entries,
                         "HeckeOperatorOnK::compute: Theta buffer size mismatch.");
        if (n_entries != 0 &&
            std::memcmp(fused_data.data(), route2_data.data(), n_entries * sizeof(Elt)) != 0) {
            size_t bad = 0;
            while (bad < n_entries && fused_data[bad] == route2_data[bad]) ++bad;
            std::ostringstream oss;
            oss << "HeckeOperatorOnK::compute: O12(a) FAILED -- the fused "
                   "route rho_n(s_i^{-1})*rho_n(g gamma_k) and route 2 "
                   "rho_n((delta_i gamma_k) mod ell) differ at q=" << q
                << ", entry (" << (bad / dim) << "," << (bad % dim) << "): "
                << fused_data[bad] << " vs " << route2_data[bad] << ".";
            throw std::runtime_error(oss.str());
        }

        const uint32_t d_q = H_dims[q];
        if (static_cast<uint32_t>(H_bases[q].size()) != d_q) {
            throw std::runtime_error(
                "HeckeOperatorOnK::compute: H_eq basis size mismatch.");
        }
        std::vector<FlatMatrixT<Elt>> y_flat;
        y_flat.reserve(d_q);
        const FlatMatrixT<Elt> D1 = Kp->template coboundary_operator_elt<Elt>(q, ell_); // N_{q+1} x N_q
        for (uint32_t i_basis = 0; i_basis < d_q; ++i_basis) {
            const FlatMatrixT<Elt>& z_i = H_bases[q][i_basis]; // N_q x d_V, row per simplex
            if (z_i.get_rows() != N_q || z_i.get_cols() != d_V_) {
                throw std::runtime_error(
                    "HeckeOperatorOnK::compute: H_eq basis cochain has "
                    "unexpected shape.");
            }
            FlatMatrixT<Elt> y = theta_flat[q] * flatten_cochain_legacy(z_i); // dim x 1

            for (const auto& rel : relations) {
                if (rel.q_dim != q) continue;
                for (uint32_t r = 0; r < d_V_; ++r) {
                    Elt rhs_val = 0;
                    for (uint32_t c = 0; c < d_V_; ++c) {
                        rhs_val = ctx.add(rhs_val, ctx.mul(rel.M.get_entry(r, c),
                                                           y.get_entry(rel.s_rhs_idx_in_Kq * d_V_ + c, 0)));
                    }
                    const Elt lhs_val = y.get_entry(rel.s_lhs_idx_in_Kq * d_V_ + r, 0);
                    if (lhs_val != rhs_val) {
                        std::ostringstream oss;
                        oss << "HeckeOperatorOnK::compute: runtime assertion (i) "
                               "FAILED -- y_" << i_basis << " = Theta z_" << i_basis
                            << " violates relation '" << rel.tag << "' at q=" << q
                            << ", fiber row " << r << " (lhs " << lhs_val
                            << " != rhs " << rhs_val << ").";
                        throw std::runtime_error(oss.str());
                    }
                }
                ++relation_checks;
            }

            for (uint32_t t = 0; t < D1.get_rows(); ++t) {
                for (uint32_t k = 0; k < d_V_; ++k) {
                    Elt acc = 0;
                    for (uint32_t s = 0; s < D1.get_cols(); ++s) {
                        const Elt d1_ts = D1.get_entry(t, s);
                        if (d1_ts == 0) continue;
                        acc = ctx.add(acc, ctx.mul(d1_ts, y.get_entry(s * d_V_ + k, 0)));
                    }
                    if (acc != 0) {
                        std::ostringstream oss;
                        oss << "HeckeOperatorOnK::compute: runtime assertion (ii) "
                               "FAILED -- d(Theta z_" << i_basis << ") != 0 at q="
                            << q << ", target simplex " << t << ", fiber coord "
                            << k << " (value " << acc << ").";
                        throw std::runtime_error(oss.str());
                    }
                }
            }
            ++cocycle_checks;

            y_flat.push_back(std::move(y));
        }

        const uint32_t dim_C_eq = static_cast<uint32_t>(C_eq_bases[q].size());
        theta_eq[q] = FlatMatrixT<Elt>::CreateZero(dim_C_eq, dim_C_eq, ell_);
        std::vector<Elt> theta_eq_buf;
        if (dim_C_eq > 0) {
            theta_eq_buf.assign(static_cast<size_t>(dim_C_eq) * dim_C_eq, 0);
            std::vector<FlatMatrixT<Elt>> c_eq_flat;
            c_eq_flat.reserve(dim_C_eq);
            for (const FlatMatrixT<Elt>& c_j : C_eq_bases[q]) {
                if (c_j.get_rows() != N_q || c_j.get_cols() != d_V_) {
                    throw std::runtime_error(
                        "HeckeOperatorOnK::compute: C_eq basis cochain has "
                        "unexpected shape.");
                }
                c_eq_flat.push_back(flatten_cochain_legacy(c_j));
            }
            std::vector<FlatMatrixT<Elt>> w_all;
            w_all.reserve(dim_C_eq);
            for (uint32_t j = 0; j < dim_C_eq; ++j) {
                w_all.push_back(theta_flat[q] * c_eq_flat[j]);
            }
            std::vector<std::vector<Elt>> coords_all;
            try {
                coords_all = change_of_basis_fm_multi(c_eq_flat, w_all);
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "HeckeOperatorOnK::compute: runtime assertion (iii) "
                       "FAILED -- Theta_flat * (some C_eq basis vector) is not "
                       "in span(C_eq) at q=" << q << " (" << e.what() << ").";
                throw std::runtime_error(oss.str());
            }
            ENGINE_CHECK_MSG(coords_all.size() == dim_C_eq,
                             "HeckeOperatorOnK::compute: multi-RHS solve returned the "
                             "wrong number of coordinate vectors.");

            for (uint32_t j = 0; j < dim_C_eq; ++j) {
                const FlatMatrixT<Elt>& w = w_all[j];
                const std::vector<Elt>& coords = coords_all[j];
                for (uint32_t row = 0; row < dim; ++row) {
                    Elt acc = 0;
                    for (uint32_t rr = 0; rr < dim_C_eq; ++rr) {
                        if (coords[rr] == 0) continue;
                        acc = ctx.add(acc, ctx.mul(coords[rr], c_eq_flat[rr].get_entry(row, 0)));
                    }
                    const Elt res = ctx.sub(w.get_entry(row, 0), acc);
                    if (res > residual_max) residual_max = res;
                    if (res != 0) {
                        std::ostringstream oss;
                        oss << "HeckeOperatorOnK::compute: runtime assertion (iii) "
                               "FAILED -- nonzero exact residual " << res
                            << " in the Theta_eq solve at q=" << q << ", basis "
                               "vector " << j << ", flat row " << row << ".";
                        throw std::runtime_error(oss.str());
                    }
                }
                for (uint32_t row = 0; row < dim_C_eq; ++row) {
                    theta_eq_buf[static_cast<size_t>(row) * dim_C_eq + j] = coords[row];
                }
                ++theta_eq_solves;
            }
            theta_eq[q] = FlatMatrixT<Elt>(dim_C_eq, dim_C_eq, ell_, theta_eq_buf);
        }

        A[q] = MatrixT<Elt>(d_q, d_q, ell_);
        if (d_q > 0) {
            const FlatMatrixT<Elt>& P_fm = projectors[q];
            if (P_fm.get_rows() != d_q || P_fm.get_cols() != dim) {
                std::ostringstream oss;
                oss << "HeckeOperatorOnK::compute: projector P_" << q
                    << " has shape " << P_fm.get_rows() << "x" << P_fm.get_cols()
                    << ", expected " << d_q << "x" << dim << ".";
                throw std::runtime_error(oss.str());
            }
            for (uint32_t i_col = 0; i_col < d_q; ++i_col) {
                const FlatMatrixT<Elt> coords = P_fm * y_flat[i_col]; // d_q x 1
                for (uint32_t j_row = 0; j_row < d_q; ++j_row) {
                    A[q].data[j_row][i_col] = coords.get_entry(j_row, 0);
                }
            }
        }
    }

    out.theta_flat = std::move(theta_flat);
    out.theta_eq = std::move(theta_eq);
    out.A = std::move(A);
    out.relation_checks = relation_checks;
    out.cocycle_checks = cocycle_checks;
    out.theta_eq_solves = theta_eq_solves;
    out.residual_max = residual_max;
}

// ENGINE_BACKEND=legacy: publish the legacy arm's result and rebuild the block
// objects from it (one interned buffer per degree, structural zeros and alpha*I
// blocks detected), so every accessor works under both backends.
template<class Elt>
void HeckeOperatorOnKT<Elt>::adopt_legacy_result(LegacyResult&& L) {
    std::shared_ptr<BlockPool> pool = std::make_shared<BlockPool>();
    std::vector<BlockMat> theta_block(num_degrees_);
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const uint32_t N_q = K()->get_num_q_simplices(q);
        const std::vector<uint32_t> dims(N_q, d_V_);
        const DenseBuf<Elt> src = borrow_flat<Elt>(L.theta_flat[q]);
        theta_block[q] = BlockMat::from_dense(pool, cview(src), dims, dims, ell_, Layout::RowMajor,
                                              /*detect=*/true);
    }
    pool_ = std::move(pool);
    Theta_block_ = std::move(theta_block);
    Theta_flat_ = std::move(L.theta_flat);
    theta_flat_built_ = true;
    Theta_eq_ = std::move(L.theta_eq);
    A_ = std::move(L.A);
    routes_bit_identical_ = true;
    relation_checks_ = L.relation_checks;
    cocycle_checks_ = L.cocycle_checks;
    theta_eq_solves_ = L.theta_eq_solves;
    theta_eq_residual_max_ = L.residual_max;
    computed_ = true;
}

