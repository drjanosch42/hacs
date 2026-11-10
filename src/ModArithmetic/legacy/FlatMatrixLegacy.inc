// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ModArithmetic/legacy/FlatMatrixLegacy.inc -- the pre-Phase-2 scalar elimination route of FlatMatrixT (the
// ENGINE_BACKEND=legacy / engine::elim::ScopedLegacy backend, and the oracle of the ENGINE_SHADOW_LEGACY build).
// Moved out of FlatMatrix.cpp by S24 CL-L2 (CLEANUP plan Sec. 2.2, C-12 step 2; owner ruling 1 of 2026-10-30):
// _gaussian_elimination_core and the 2-d helpers verbatim; the RREF branch and the three `legacy_route` lambdas of
// get_rref_and_transform_matrix / compute_elimination_factor_and_apply / inverse became the private members
// _legacy_row_echelon_form / _legacy_rref_and_transform / _legacy_elimination_factor_and_apply / _legacy_inverse
// (bodies verbatim; the lambdas' captures are re-bound at the top of each member from the same expressions).
// Included by FlatMatrix.cpp at namespace scope, before the first member definition, under `#if OC_ENABLE_LEGACY`;
// declared in FlatMatrix.h under the same guard.  An OFF build has none of it: every elimination is the engine's.
#if !OC_ENABLE_LEGACY
#error "FlatMatrixLegacy.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

namespace {

template<class Elt>
std::vector<std::vector<Elt>> flat_to_2d(const std::vector<Elt>& flat, uint32_t rows, uint32_t cols) {
    std::vector<std::vector<Elt>> out(rows, std::vector<Elt>(cols));
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) out[i][j] = flat[static_cast<size_t>(i) * cols + j];
    return out;
}

template<class Elt>
void flat_from_2d(const std::vector<std::vector<Elt>>& src, uint32_t rows, uint32_t cols,
                  std::vector<Elt>& flat) {
    flat.resize(static_cast<size_t>(rows) * cols);
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) flat[static_cast<size_t>(i) * cols + j] = src[i][j];
}

} // namespace

template<class Elt>
void FlatMatrixT<Elt>::_gaussian_elimination_core(
    std::vector<std::vector<Elt>>& mat_2d,
    std::vector<uint32_t>& pivots,
    uint32_t num_mat_rows,
    uint32_t num_mat_cols) const
{
    pivots.clear();
    uint32_t lead = 0;

    for (uint32_t r = 0; r < num_mat_rows; ++r) {
        if (lead >= num_mat_cols) break;
        uint32_t i_pivot_row = r;
        while (lead < num_mat_cols && (i_pivot_row < num_mat_rows && mat_2d[i_pivot_row][lead] == 0)) {
            i_pivot_row++;
            if (i_pivot_row == num_mat_rows) { // Exhausted rows for current lead
                i_pivot_row = r; // Reset for next lead
                lead++;          // Try next column
            }
        }

        if (lead < num_mat_cols && i_pivot_row < num_mat_rows) { // Found a pivot
            std::swap(mat_2d[i_pivot_row], mat_2d[r]); // Bring pivot row to current row r

            Elt pivot_val = mat_2d[r][lead]; // Must be non-zero
            if (pivot_val == 0) throw std::logic_error("Gaussian elimination: zero pivot found unexpectedly.");


            Elt pivot_inv = ModArithmetic::mod_inv_w(pivot_val, p_);
            for (uint32_t j_col = lead; j_col < num_mat_cols; ++j_col) { // Normalize pivot row
                mat_2d[r][j_col] = ModArithmetic::mod_mul_w(mat_2d[r][j_col], pivot_inv, p_);
            }

            for (uint32_t k_row = 0; k_row < num_mat_rows; ++k_row) { // Eliminate other rows
                if (k_row != r) {
                    Elt factor = mat_2d[k_row][lead];
                    if (factor != 0) {
                        for (uint32_t j_col = lead; j_col < num_mat_cols; ++j_col) {
                            mat_2d[k_row][j_col] = ModArithmetic::mod_sub_w(mat_2d[k_row][j_col], ModArithmetic::mod_mul_w(factor, mat_2d[r][j_col], p_), p_);
                        }
                    }
                }
            }
            pivots.push_back(lead);
            lead++; // Move to next column for next pivot search
        }
        else { // No more pivots to find
            break;
        }
    }
}

template<class Elt>
void FlatMatrixT<Elt>::_legacy_row_echelon_form() const {
    // the caller (compute_row_echelon_form) has reset both caches and populated flat_data_
    std::vector<std::vector<Elt>> temp_2d_data = flat_to_2d<Elt>(flat_data_, rows_, cols_);
    _gaussian_elimination_core(temp_2d_data, *pivot_columns_cache_, rows_, cols_);
    flat_from_2d<Elt>(temp_2d_data, rows_, cols_, *row_echelon_form_cache_);
}

template<class Elt>
std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> FlatMatrixT<Elt>::_legacy_rref_and_transform() const {
    // Legacy scalar route: RREF([A | I]) through _gaussian_elimination_core.
    // the captures of the former lambda (get_rref_and_transform_matrix: rows_ > 0, cols_ > 0, flat data populated)
    uint32_t m = rows_;
    uint32_t n = cols_;
    uint32_t augmented_cols = n + m;
    std::vector<std::vector<Elt>> augmented_mat_2d(m, std::vector<Elt>(augmented_cols, 0));
    for (uint32_t r = 0; r < m; ++r) {
        for (uint32_t c_a = 0; c_a < n; ++c_a) {
            augmented_mat_2d[r][c_a] = this->get_entry(r, c_a);
        }
        augmented_mat_2d[r][n + r] = 1;
    }
    std::vector<uint32_t> augmented_pivots;
    _gaussian_elimination_core(augmented_mat_2d, augmented_pivots, m, augmented_cols);

    std::vector<Elt> rref_A_flat_data(static_cast<size_t>(m) * n);
    std::vector<Elt> C_transform_flat_data(static_cast<size_t>(m) * m);
    for (uint32_t r = 0; r < m; ++r) {
        for (uint32_t c_extract = 0; c_extract < augmented_cols; ++c_extract) {
            if (c_extract < n) {
                rref_A_flat_data[static_cast<size_t>(r) * n + c_extract] = augmented_mat_2d[r][c_extract];
            }
            else {
                C_transform_flat_data[static_cast<size_t>(r) * m + (c_extract - n)] = augmented_mat_2d[r][c_extract];
            }
        }
    }
    return { FlatMatrixT<Elt>(m, n, p_, rref_A_flat_data), FlatMatrixT<Elt>(m, m, p_, C_transform_flat_data) };
}

template<class Elt>
std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> FlatMatrixT<Elt>::_legacy_elimination_factor_and_apply(
    const FlatMatrixT<Elt>& pivot_rref_block) const {
    // Legacy route: gather L entrywise, then B - L*R through operator* / operator-.
    // the captures of the former lambda (compute_elimination_factor_and_apply's general case: rank_R > 0)
    uint32_t m_B = this->rows_;
    uint32_t k_R = pivot_rref_block.rows_; // R.rows
    Elt prime = this->p_;
    uint32_t rank_R = pivot_rref_block.rank();
    const std::vector<uint32_t>& R_pivot_cols_indices = pivot_rref_block.get_pivot_columns();
    FlatMatrixT<Elt> L_factor = FlatMatrixT<Elt>::CreateZero(m_B, k_R, prime);
    for (uint32_t r_B = 0; r_B < m_B; ++r_B) {
        for (uint32_t pivot_idx_in_R_rows = 0; pivot_idx_in_R_rows < rank_R; ++pivot_idx_in_R_rows) {
            // pivot_idx_in_R_rows is the effective row index of the pivot in R (0 to rank_R-1)
            // This corresponds to the (pivot_idx_in_R_rows)-th actual pivot row of R.
            uint32_t col_of_this_R_pivot = R_pivot_cols_indices[pivot_idx_in_R_rows];
            Elt val_from_B = this->get_entry(r_B, col_of_this_R_pivot);

            // L_factor's column 'pivot_idx_in_R_rows' corresponds to this pivot row of R.
            L_factor.set_entry(r_B, pivot_idx_in_R_rows, val_from_B);
        }
    }
    FlatMatrixT<Elt> B_new = *this - (L_factor * pivot_rref_block);
    return { B_new, L_factor };
}

template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::_legacy_inverse() const {
    std::vector<std::vector<Elt>> augmented_mat_2d(rows_, std::vector<Elt>(2 * cols_));
    for (uint32_t i = 0; i < rows_; ++i) {
        for (uint32_t j = 0; j < cols_; ++j) {
            // Use get_entry to be safe, though for general non-scalar, flat_data_ should be used
            augmented_mat_2d[i][j] = this->get_entry(i, j);
        }
        augmented_mat_2d[i][i + cols_] = 1; // Identity part
    }

    std::vector<uint32_t> augmented_pivots;
    // _gaussian_elimination_core is const, operates on the copy augmented_mat_2d
    _gaussian_elimination_core(augmented_mat_2d, augmented_pivots, rows_, 2 * cols_);

    FlatMatrixT<Elt> inv_matrix(rows_, cols_, p_); // Starts scalar 0
    inv_matrix.populate_flat_data(); // Becomes general 0-filled
    for (uint32_t i = 0; i < rows_; ++i) {
        for (uint32_t j = 0; j < cols_; ++j) {
            inv_matrix.flat_data_[static_cast<size_t>(i) * cols_ + j] = augmented_mat_2d[i][j + cols_];
        }
    }
    inv_matrix._determine_and_set_flags_from_data(); // Check if inverse became special
    return inv_matrix;
}
