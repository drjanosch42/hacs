// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "../pch.h" // If using precompiled headers
#include "BlockFlatMatrix.h"
#include <cstring>   // std::memcmp (legacy/ShadowLegacy.inc, ENGINE_SHADOW_LEGACY)
#include "../DenseBuf.h"      // matrix-engine Phase 2: block updates through the engine
#include "../DenseView.h"
#include "../DenseElim.h"
#include "../ModCtx.h"
#include "../ModArithmetic.h" // For modular operations
#include <sstream>
#include <iomanip>
#include <numeric> // For std::accumulate, std::iota
#include <algorithm> // For std::all_of, std::find_if, std::sort, std::copy etc.
#include <map>       // For temporarily finding pivot choices (original) or potentially new uses
#include "../EngineCheck.h" // ENGINE_CHECK / ENGINE_DEBUG_CHECK (NDEBUG policy)

// Assuming this helper is available (e.g., from FlatMatrixValidation or a utility header)
// For robust comparison without relying on FlatMatrix::operator== side effects or non-constness.
template<class Elt>
bool compare_flat_matrices_content_for_block(const FlatMatrixT<Elt>& fm1, const FlatMatrixT<Elt>& fm2) {
    if (fm1.get_rows() != fm2.get_rows() || fm1.get_cols() != fm2.get_cols() || fm1.get_p() != fm2.get_p()) {
        return false;
    }
    Elt val1, val2;
    bool is_scalar1 = fm1.is_currently_scalar(val1);
    bool is_scalar2 = fm2.is_currently_scalar(val2);
    if (is_scalar1 && is_scalar2) {
        return val1 == val2;
    }
    for (uint32_t r = 0; r < fm1.get_rows(); ++r) {
        for (uint32_t c = 0; c < fm1.get_cols(); ++c) {
            if (fm1.get_entry(r, c) != fm2.get_entry(r, c)) {
                return false;
            }
        }
    }
    return true;
}

// --- Private Helper Methods ---
template<class Elt>
void BlockFlatMatrixT<Elt>::_clear_cached_rref() const {
    rref_cache_.reset();
    fine_pivot_info_cache_.reset();
}

template<class Elt>
void BlockFlatMatrixT<Elt>::_compute_total_dims() {
    num_block_rows_ = static_cast<uint32_t>(block_row_dims_.size());
    num_block_cols_ = static_cast<uint32_t>(block_col_dims_.size());

    // (Phase 1, defect #8) The prefix sums are built in the same pass that
    // accumulates the totals, so they can never drift from the dimensions.
    block_row_offsets_.resize(static_cast<size_t>(num_block_rows_) + 1);
    total_rows_ = 0;
    for (uint32_t i = 0; i < num_block_rows_; ++i) {
        block_row_offsets_[i] = total_rows_;
        total_rows_ += block_row_dims_[i];
    }
    block_row_offsets_[num_block_rows_] = total_rows_;

    block_col_offsets_.resize(static_cast<size_t>(num_block_cols_) + 1);
    total_cols_ = 0;
    for (uint32_t j = 0; j < num_block_cols_; ++j) {
        block_col_offsets_[j] = total_cols_;
        total_cols_ += block_col_dims_[j];
    }
    block_col_offsets_[num_block_cols_] = total_cols_;
}

template<class Elt>
void BlockFlatMatrixT<Elt>::_validate_block_layout() const {
    if (block_layout_indices_.size() != num_block_rows_) {
        throw std::logic_error("Block layout rows mismatch block_row_dims size.");
    }
    for (uint32_t i = 0; i < num_block_rows_; ++i) {
        if (block_layout_indices_[i].size() != num_block_cols_) {
            throw std::logic_error("Block layout cols mismatch block_col_dims size for block_row " + std::to_string(i));
        }
        for (uint32_t j = 0; j < num_block_cols_; ++j) {
            size_t block_idx = block_layout_indices_[i][j];
            if (block_idx >= unique_blocks_.size()) {
                throw std::out_of_range("Block index out of range in layout.");
            }
            const FlatMatrixT<Elt>& block = unique_blocks_[block_idx];
            if (block.get_rows() != block_row_dims_[i]) {
                throw std::logic_error("Block at (" + std::to_string(i) + "," + std::to_string(j) + ") has incorrect row dimension.");
            }
            if (block.get_cols() != block_col_dims_[j]) {
                throw std::logic_error("Block at (" + std::to_string(i) + "," + std::to_string(j) + ") has incorrect col dimension.");
            }
            if (block.get_p() != p_) {
                throw std::logic_error("Block at (" + std::to_string(i) + "," + std::to_string(j) + ") has inconsistent prime modulus.");
            }
        }
    }
}

// Defect #14: the element hashes were taken through std::hash<uint32_t>, which
// for Elt = uint64_t truncates every entry to its low 32 bits.  That never made
// the class wrong -- _add_or_get_block_index still confirms a hit with an exact
// content comparison -- but in the extended regime any two blocks differing only
// above bit 31 collided, so the dedup pool degenerated into linear scans.  The
// hash is now taken at the element width, std::hash<Elt>.
template<class Elt>
size_t BlockFlatMatrixT<Elt>::_compute_block_hash(const FlatMatrixT<Elt>& block) {
    size_t h = std::hash<uint32_t>{}(block.get_rows()) ^ (std::hash<uint32_t>{}(block.get_cols()) << 16);
    Elt scalar_val;
    if (block.is_currently_scalar(scalar_val)) {
        h ^= std::hash<Elt>{}(scalar_val) * 2654435761u;
        return h;
    }
    const auto& data = block.get_flat_data_unsafe();
    for (size_t i = 0; i < data.size(); ++i) {
        h ^= std::hash<Elt>{}(data[i]) * (i + 1) * 2654435761u;
    }
    return h;
}

template<class Elt>
size_t BlockFlatMatrixT<Elt>::_add_or_get_block_index(const FlatMatrixT<Elt>& block) {
    if (block.get_p() != p_) {
        throw std::invalid_argument("Block to add has inconsistent prime modulus.");
    }
    size_t h = _compute_block_hash(block);
    auto it = block_hash_index_.find(h);
    if (it != block_hash_index_.end()) {
        for (size_t idx : it->second) {
            if (compare_flat_matrices_content_for_block(unique_blocks_[idx], block)) {
                return idx;
            }
        }
    }
    size_t new_idx = unique_blocks_.size();
    unique_blocks_.push_back(block);
    block_hash_index_[h].push_back(new_idx);
    return new_idx;
}

// --- Constructors ---
template<class Elt>
BlockFlatMatrixT<Elt>::BlockFlatMatrixT(Elt prime)
    : p_(prime), total_rows_(0), total_cols_(0), num_block_rows_(0), num_block_cols_(0),
      block_row_offsets_(1, 0), block_col_offsets_(1, 0) {}

template<class Elt>
BlockFlatMatrixT<Elt>::BlockFlatMatrixT(const std::vector<uint32_t>& block_row_dims,
    const std::vector<uint32_t>& block_col_dims,
    Elt prime)
    : block_row_dims_(block_row_dims), block_col_dims_(block_col_dims), p_(prime) {
    _compute_total_dims();
    block_layout_indices_.resize(num_block_rows_);

    if (num_block_rows_ > 0 || num_block_cols_ > 0) {
        std::map<std::pair<uint32_t, uint32_t>, size_t> zero_block_indices;
        for (uint32_t i = 0; i < num_block_rows_; ++i) {
            block_layout_indices_[i].resize(num_block_cols_);
            for (uint32_t j = 0; j < num_block_cols_; ++j) {
                uint32_t r_dim = block_row_dims_[i];
                uint32_t c_dim = block_col_dims_[j];
                auto key = std::make_pair(r_dim, c_dim);
                if (zero_block_indices.find(key) == zero_block_indices.end()) {
                    FlatMatrixT<Elt> zero_block = FlatMatrixT<Elt>::CreateZero(r_dim, c_dim, p_);
                    zero_block_indices[key] = _add_or_get_block_index(zero_block);
                }
                block_layout_indices_[i][j] = zero_block_indices[key];
            }
        }
    }
    // _validate_block_layout(); // Can be enabled for debugging constructor
}

template<class Elt>
BlockFlatMatrixT<Elt>::BlockFlatMatrixT(const std::vector<uint32_t>& block_row_dims,
    const std::vector<uint32_t>& block_col_dims,
    const std::vector<std::vector<FlatMatrixT<Elt>>>& block_data,
    Elt prime)
    : block_row_dims_(block_row_dims), block_col_dims_(block_col_dims), p_(prime) {
    _compute_total_dims();
    if (block_data.size() != num_block_rows_) {
        throw std::invalid_argument("block_data row count mismatch.");
    }
    block_layout_indices_.resize(num_block_rows_);
    for (uint32_t i = 0; i < num_block_rows_; ++i) {
        if (block_data[i].size() != num_block_cols_) {
            throw std::invalid_argument("block_data column count mismatch for block_row " + std::to_string(i));
        }
        block_layout_indices_[i].resize(num_block_cols_);
        for (uint32_t j = 0; j < num_block_cols_; ++j) {
            if (block_data[i][j].get_p() != p_) {
                throw std::invalid_argument("Block at (" + std::to_string(i) + "," + std::to_string(j) + ") has inconsistent prime modulus.");
            }
            if (block_data[i][j].get_rows() != block_row_dims_[i] || block_data[i][j].get_cols() != block_col_dims_[j]) {
                throw std::invalid_argument("Block at (" + std::to_string(i) + "," + std::to_string(j) + ") has mismatched dimensions.");
            }
            block_layout_indices_[i][j] = _add_or_get_block_index(block_data[i][j]);
        }
    }
    // _validate_block_layout(); // Can be enabled for debugging
}




template<class Elt>
BlockFlatMatrixT<Elt>::~BlockFlatMatrixT() {
    _clear_cached_rref();
}

// Copy Constructor
template<class Elt>
BlockFlatMatrixT<Elt>::BlockFlatMatrixT(const BlockFlatMatrixT<Elt>& other)
    : block_row_dims_(other.block_row_dims_), block_col_dims_(other.block_col_dims_),
    p_(other.p_), unique_blocks_(other.unique_blocks_), // Shallow copy of unique_blocks (they are const-behaving for BlockFlatMatrix)
    block_layout_indices_(other.block_layout_indices_),
    block_hash_index_(other.block_hash_index_),
    total_rows_(other.total_rows_), total_cols_(other.total_cols_),
    num_block_rows_(other.num_block_rows_), num_block_cols_(other.num_block_cols_),
    block_row_offsets_(other.block_row_offsets_), block_col_offsets_(other.block_col_offsets_) { // Caches are not copied (unique_ptr default-constructs to null)
}

// Copy Assignment
template<class Elt>
BlockFlatMatrixT<Elt>& BlockFlatMatrixT<Elt>::operator=(const BlockFlatMatrixT<Elt>& other) {
    if (this != &other) {
        _clear_cached_rref();
        block_row_dims_ = other.block_row_dims_;
        block_col_dims_ = other.block_col_dims_;
        p_ = other.p_;
        unique_blocks_ = other.unique_blocks_;
        block_layout_indices_ = other.block_layout_indices_;
        block_hash_index_ = other.block_hash_index_;
        total_rows_ = other.total_rows_;
        total_cols_ = other.total_cols_;
        num_block_rows_ = other.num_block_rows_;
        num_block_cols_ = other.num_block_cols_;
        block_row_offsets_ = other.block_row_offsets_;
        block_col_offsets_ = other.block_col_offsets_;
    }
    return *this;
}

// Move Constructor
template<class Elt>
BlockFlatMatrixT<Elt>::BlockFlatMatrixT(BlockFlatMatrixT<Elt>&& other) noexcept
    : block_row_dims_(std::move(other.block_row_dims_)),
    block_col_dims_(std::move(other.block_col_dims_)),
    p_(other.p_),
    unique_blocks_(std::move(other.unique_blocks_)),
    block_layout_indices_(std::move(other.block_layout_indices_)),
    block_hash_index_(std::move(other.block_hash_index_)),
    total_rows_(other.total_rows_), total_cols_(other.total_cols_),
    num_block_rows_(other.num_block_rows_), num_block_cols_(other.num_block_cols_),
    block_row_offsets_(std::move(other.block_row_offsets_)),
    block_col_offsets_(std::move(other.block_col_offsets_)),
    rref_cache_(std::move(other.rref_cache_)),
    fine_pivot_info_cache_(std::move(other.fine_pivot_info_cache_)) {

    other.total_rows_ = 0; other.total_cols_ = 0;
    other.num_block_rows_ = 0; other.num_block_cols_ = 0;
    other.block_row_offsets_.assign(1, 0); other.block_col_offsets_.assign(1, 0);
}

// Move Assignment
template<class Elt>
BlockFlatMatrixT<Elt>& BlockFlatMatrixT<Elt>::operator=(BlockFlatMatrixT<Elt>&& other) noexcept {
    if (this != &other) {
        _clear_cached_rref(); // Clear LHS caches

        block_row_dims_ = std::move(other.block_row_dims_);
        block_col_dims_ = std::move(other.block_col_dims_);
        p_ = other.p_;
        unique_blocks_ = std::move(other.unique_blocks_);
        block_layout_indices_ = std::move(other.block_layout_indices_);
        block_hash_index_ = std::move(other.block_hash_index_);
        total_rows_ = other.total_rows_; total_cols_ = other.total_cols_;
        num_block_rows_ = other.num_block_rows_; num_block_cols_ = other.num_block_cols_;
        block_row_offsets_ = std::move(other.block_row_offsets_);
        block_col_offsets_ = std::move(other.block_col_offsets_);

        rref_cache_ = std::move(other.rref_cache_);
        fine_pivot_info_cache_ = std::move(other.fine_pivot_info_cache_);

        other.total_rows_ = 0; other.total_cols_ = 0;
        other.num_block_rows_ = 0; other.num_block_cols_ = 0;
        other.block_row_offsets_.assign(1, 0); other.block_col_offsets_.assign(1, 0);
    }
    return *this;
}


template<class Elt>
uint32_t BlockFlatMatrixT<Elt>::get_block_row_dim(uint32_t block_r) const {
    if (block_r >= num_block_rows_) throw std::out_of_range("Block row index out of range.");
    return block_row_dims_[block_r];
}
template<class Elt>
uint32_t BlockFlatMatrixT<Elt>::get_block_col_dim(uint32_t block_c) const {
    if (block_c >= num_block_cols_) throw std::out_of_range("Block column index out of range.");
    return block_col_dims_[block_c];
}

template<class Elt>
const FlatMatrixT<Elt>& BlockFlatMatrixT<Elt>::get_block_entry(uint32_t block_r, uint32_t block_c) const {
    if (block_r >= num_block_rows_ || block_c >= num_block_cols_) {
        throw std::out_of_range("Block indices out of range in get_block_entry.");
    }
    return unique_blocks_[block_layout_indices_[block_r][block_c]];
}

template<class Elt>
void BlockFlatMatrixT<Elt>::set_block_entry(uint32_t block_r, uint32_t block_c, const FlatMatrixT<Elt>& block) {
    if (block_r >= num_block_rows_ || block_c >= num_block_cols_) {
        throw std::out_of_range("Block indices out of range in set_block_entry.");
    }
    if (block.get_rows() != block_row_dims_[block_r] || block.get_cols() != block_col_dims_[block_c]) {
        throw std::invalid_argument("Block dimensions mismatch for set_block_entry.");
    }
    if (block.get_p() != p_) {
        throw std::invalid_argument("Block prime modulus mismatch for set_block_entry.");
    }
    _clear_cached_rref();
    block_layout_indices_[block_r][block_c] = _add_or_get_block_index(block);
}

// (Phase 1, defect #8) The two O(#block rows) / O(#block cols) linear scans that
// used to walk the dimension vectors on EVERY scalar access are replaced by one
// binary search per axis over the cached prefix sums.  `upper_bound(...) - 1`
// selects the LAST block whose start offset is <= the index, which is exactly the
// block that contains it -- zero-dimensional blocks repeat their predecessor's
// offset and are skipped by that rule, matching the old `if (r < off + dim)` walk.
template<class Elt>
Elt BlockFlatMatrixT<Elt>::get_entry(uint32_t r, uint32_t c) const {
    if (r >= total_rows_ || c >= total_cols_) {
        throw std::out_of_range("Scalar indices out of range in get_entry.");
    }
    const uint32_t br = static_cast<uint32_t>(
        std::upper_bound(block_row_offsets_.begin(), block_row_offsets_.begin() + num_block_rows_ + 1, r)
        - block_row_offsets_.begin()) - 1;
    const uint32_t bc = static_cast<uint32_t>(
        std::upper_bound(block_col_offsets_.begin(), block_col_offsets_.begin() + num_block_cols_ + 1, c)
        - block_col_offsets_.begin()) - 1;
    ENGINE_DEBUG_CHECK(br < num_block_rows_ && bc < num_block_cols_);
    const FlatMatrixT<Elt>& block = unique_blocks_[block_layout_indices_[br][bc]];
    return block.get_entry(r - block_row_offsets_[br], c - block_col_offsets_[bc]);
}

template<class Elt>
FlatMatrixT<Elt> BlockFlatMatrixT<Elt>::to_flat_matrix() const {
    if (total_rows_ == 0 || total_cols_ == 0) {
        return FlatMatrixT<Elt>::CreateZero(total_rows_, total_cols_, p_);
    }
    std::vector<Elt> flat_data(static_cast<size_t>(total_rows_) * total_cols_);
    uint32_t current_global_row = 0;
    for (uint32_t br = 0; br < num_block_rows_; ++br) {
        uint32_t current_global_col = 0;
        for (uint32_t bc = 0; bc < num_block_cols_; ++bc) {
            const FlatMatrixT<Elt>& block = get_block_entry(br, bc);
            for (uint32_t r_in_block = 0; r_in_block < block.get_rows(); ++r_in_block) {
                for (uint32_t c_in_block = 0; c_in_block < block.get_cols(); ++c_in_block) {
                    flat_data[(current_global_row + r_in_block) * total_cols_ + (current_global_col + c_in_block)] =
                        block.get_entry(r_in_block, c_in_block);
                }
            }
            current_global_col += block.get_cols();
        }
        current_global_row += block_row_dims_[br];
    }
    return FlatMatrixT<Elt>(total_rows_, total_cols_, p_, flat_data);
}

template<class Elt>
std::string BlockFlatMatrixT<Elt>::to_string_offset(bool show_block_details, const uint32_t offset) const {
    std::ostringstream oss;
    oss << offset_string(offset) << "BlockFlatMatrix (" << num_block_rows_ << "x" << num_block_cols_ << " blocks, "
        << total_rows_ << "x" << total_cols_ << " total dims, p=" << p_ << ")\n";
    oss << offset_string(offset + 2) << "Block Row Dims: [ "; for (uint32_t d : block_row_dims_) oss << d << " "; oss << "]\n";
    oss << offset_string(offset + 2) << "Block Col Dims: [ "; for (uint32_t d : block_col_dims_) oss << d << " "; oss << "]\n";

    if (show_block_details) {
        for (uint32_t br = 0; br < num_block_rows_; ++br) {
            for (uint32_t bc = 0; bc < num_block_cols_; ++bc) {
                oss << offset_string(offset + 2) << "Block (" << br << "," << bc << ") - Index in unique_blocks: " << block_layout_indices_[br][bc] << "\n";
                oss << get_block_entry(br, bc).to_string_offset(offset + 4);
            }
        }
    }
    else {
        if (total_rows_ > 0 && total_cols_ > 0 && total_rows_ <= 20 && total_cols_ <= 20) {
            oss << to_flat_matrix().to_string_offset(offset + 2);
        }
        else if (total_rows_ > 0 || total_cols_ > 0) {
            oss << offset_string(offset + 2) << "(Expanded matrix too large to print by default)\n";
        }
    }
    return oss.str();
}
template<class Elt>
std::string BlockFlatMatrixT<Elt>::to_string(bool show_block_details) const { return to_string_offset(show_block_details, 0); }


template<class Elt>
bool BlockFlatMatrixT<Elt>::is_zero() const { // (defect #19)
    for (uint32_t br = 0; br < num_block_rows_; ++br) {
        for (uint32_t bc = 0; bc < num_block_cols_; ++bc) {
            const FlatMatrixT<Elt>& block = unique_blocks_[block_layout_indices_[br][bc]];
            if (!block.is_zero()) {
                return false;
            }
        }
    }
    _clear_cached_rref();
    return true;
}

namespace {
    template<class Elt>
    bool is_block_zero(const FlatMatrixT<Elt>& fm) {
        Elt val;
        return fm.is_currently_scalar(val) && val == 0;
    }
#ifdef ENGINE_SHADOW_LEGACY
#include "ShadowLegacy.inc"   // (S24 CL-L1) the shadow comparisons
#endif
}



template<class Elt>
BlockFlatMatrixT<Elt> BlockFlatMatrixT<Elt>::operator*(const BlockFlatMatrixT<Elt>& other) const {
    if (num_block_cols_ != other.num_block_rows_ || p_ != other.p_) {
        throw std::invalid_argument("BlockFlatMatrix::operator*: Block dimensions (inner) or prime mismatch for multiplication.");
    }
    for (size_t k = 0; k < num_block_cols_; ++k) {
        if (block_col_dims_[k] != other.block_row_dims_[k]) {
            throw std::invalid_argument("BlockFlatMatrix::operator*: Inner block dimensions mismatch at block index " + std::to_string(k));
        }
    }
    std::vector<std::vector<FlatMatrixT<Elt>>> prod_block_data(num_block_rows_);
    for (uint32_t br_A = 0; br_A < num_block_rows_; ++br_A) {
        prod_block_data[br_A].resize(other.num_block_cols_);
        for (uint32_t bc_B = 0; bc_B < other.num_block_cols_; ++bc_B) {
            // (Phase 1, defect #12) `sum_block = sum_block + (A*B)` allocated a
            // fresh FlatMatrix and re-ran _determine_and_set_flags_from_data()
            // over it once per k.  The partial sums now accumulate in one plain
            // buffer and the flags are derived once, at the end.  Same terms in
            // the same order, same mod_add_w, and an all-zero result still comes
            // back through CreateZero, so the representation is unchanged too.
            const uint32_t R_dim = block_row_dims_[br_A];
            const uint32_t C_dim = other.block_col_dims_[bc_B];
            std::vector<Elt> acc;
            bool acc_active = false;
            for (uint32_t k = 0; k < num_block_cols_; ++k) {
                const FlatMatrixT<Elt>& A_ik = get_block_entry(br_A, k);
                const FlatMatrixT<Elt>& B_kj = other.get_block_entry(k, bc_B);
                if (is_block_zero(A_ik) || is_block_zero(B_kj)) continue;
                if (!acc_active) { acc.assign(static_cast<size_t>(R_dim) * C_dim, Elt(0)); acc_active = true; }

                // (matrix-engine Phase 2, Stage B) When both blocks own a
                // buffer, engine::kernels::gemm accumulates A_ik * B_kj
                // DIRECTLY into acc (GemmOp::Add) -- no per-k temporary
                // FlatMatrix, no second entrywise pass, and no rebuilding of
                // the scalar flags for a value that is about to be summed
                // away. The exact k-blocked accumulation makes this the same
                // residue the two-step route produced (ENGINE_API.md Sec. 5).
                // A scalar-stored (alpha*I) block declines and falls through
                // to the legacy two-step route below, which for that case is
                // itself the cheaper one (operator*(scalar), not a gemm).
#ifdef ENGINE_SHADOW_LEGACY
                std::vector<Elt> shadow_acc = acc;
                const bool engine_did = flat_gemm_accumulate_engine<Elt>(A_ik, B_kj, shadow_acc,
                                                                         R_dim, C_dim);
#else
                if (flat_gemm_accumulate_engine<Elt>(A_ik, B_kj, acc, R_dim, C_dim)) continue;
#endif
                const FlatMatrixT<Elt> term = A_ik * B_kj;
                for (uint32_t r = 0; r < R_dim; ++r) {
                    for (uint32_t c = 0; c < C_dim; ++c) {
                        Elt& slot = acc[static_cast<size_t>(r) * C_dim + c];
                        slot = ModArithmetic::mod_add_w(slot, term.get_entry(r, c), p_);
                    }
                }
#ifdef ENGINE_SHADOW_LEGACY
                if (engine_did) shadow_check<Elt>("BlockFlatMatrix::operator*", shadow_acc, acc);
#endif
            }
            bool acc_all_zero = true;
            if (acc_active) {
                for (Elt v : acc) { if (v != 0) { acc_all_zero = false; break; } }
            }
            prod_block_data[br_A][bc_B] = (!acc_active || acc_all_zero)
                ? FlatMatrixT<Elt>::CreateZero(R_dim, C_dim, p_)
                : FlatMatrixT<Elt>(R_dim, C_dim, p_, acc);
        }
    }
    return BlockFlatMatrixT<Elt>(block_row_dims_, other.block_col_dims_, prod_block_data, p_);
}




// --- Linear Algebra ---

// Helper structure for the new _block_gaussian_elimination
template<class Elt>
struct EliminationBlockRow {
    std::vector<FlatMatrixT<Elt>> blocks; // One FlatMatrix per original block-column of 'this'
    uint32_t original_label;        // The initial block-row index from 'this'
    uint32_t current_row_dim;       // Scalar dimension of this logical row (can change on split)

    // Constructor from an existing BlockFlatMatrix row
    EliminationBlockRow(const BlockFlatMatrixT<Elt>& bfm_source, uint32_t block_r_idx_in_source)
        : original_label(block_r_idx_in_source), current_row_dim(bfm_source.get_block_row_dim(block_r_idx_in_source)) {
        blocks.reserve(bfm_source.get_num_block_cols());
        for (uint32_t bc = 0; bc < bfm_source.get_num_block_cols(); ++bc) {
            blocks.push_back(bfm_source.get_block_entry(block_r_idx_in_source, bc)); // Deep copy from source
        }
    }

    // Constructor for creating a new row from existing blocks (e.g., after a split)
    EliminationBlockRow(std::vector<FlatMatrixT<Elt>>&& row_blocks, uint32_t label, uint32_t dim)
        : blocks(std::move(row_blocks)), original_label(label), current_row_dim(dim) {}

    // Default constructor for placeholder or error states if needed
    EliminationBlockRow() : original_label(0), current_row_dim(0) {}
};


// Helper function to determine pivot priority (same as before)
template<class Elt>
int get_pivot_block_priority(const FlatMatrixT<Elt>& block, Elt p_mod) { // (defect #19) is_zero()/is_currently_scalar() are const
    if (block.is_zero()) { return 6; }
    if (block.get_rows() == block.get_cols() && block.get_rows() > 0) {
        Elt scalar_val;
        if (block.is_currently_scalar(scalar_val)) {
            if (scalar_val == 0) return 6;
            if (scalar_val == 1) return 1;
            if (scalar_val == p_mod - 1) return 2;
            return 3;
        }
    }
    return 5;
}

// (matrix-engine Phase 2) T - L*R for the two elimination loops below, as ONE
// in-place GEMM (GemmOp::Sub) on a dense copy of T -- no L*R temporary and no
// separate subtraction pass.  Exact arithmetic, so the values are those of
// `T - (L * R)`; the result goes through the DATA constructor, so its flags are
// the same pure function of the data as before.  L*R == 0 leaves T untouched
// (legacy: T - 0 is T with re-derived flags, i.e. the same object).
// ENGINE_BACKEND=legacy / ScopedLegacy takes the old operator* / operator-
// route; ENGINE_SHADOW_LEGACY runs both and compares.
template<class Elt>
static FlatMatrixT<Elt> block_sub_product(const FlatMatrixT<Elt>& T, const FlatMatrixT<Elt>& L,
                                          const FlatMatrixT<Elt>& R, Elt prime) {
    if (engine::elim::use_legacy()) return T - (L * R);
    const uint32_t m = T.get_rows(), n = T.get_cols(), k = L.get_cols();
    ENGINE_CHECK_MSG(L.get_rows() == m && R.get_rows() == k && R.get_cols() == n,
                     "block_sub_product: shape mismatch");
    if (L.is_zero() || R.is_zero() || m == 0 || n == 0) return T;
    // A general block's flat storage is used directly (one vector copy for T,
    // read-only wraps for L and R); only scalar-flagged blocks (alpha*I with
    // empty storage) are materialised entry by entry.  This is the innermost
    // call of factor_basis_block's elimination (millions of 1-wide blocks), so
    // the per-call overhead matters more than the arithmetic.
    auto general_flat = [](const FlatMatrixT<Elt>& X) -> const std::vector<Elt>* {
        Elt sv;
        const std::vector<Elt>& d = X.get_flat_data_unsafe();
        if (!X.is_currently_scalar(sv) && d.size() == static_cast<size_t>(X.get_rows()) * X.get_cols()) return &d;
        return nullptr;
    };
    auto materialise = [](const FlatMatrixT<Elt>& X, engine::DenseBuf<Elt>& holder) {
        holder = engine::DenseBuf<Elt>(X.get_rows(), X.get_cols());
        for (uint32_t i = 0; i < X.get_rows(); ++i)
            for (uint32_t j = 0; j < X.get_cols(); ++j) holder.at(i, j) = X.get_entry(i, j);
    };
    std::vector<Elt> t_data;
    if (const std::vector<Elt>* td = general_flat(T)) {
        t_data = *td;
    } else {
        t_data.resize(static_cast<size_t>(m) * n);
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < n; ++j) t_data[static_cast<size_t>(i) * n + j] = T.get_entry(i, j);
    }
    {
        const engine::ModCtx<Elt> ctx(prime);
        engine::DenseBuf<Elt> Tb(t_data.data(), t_data.size(), m, n, engine::Layout::RowMajor);
        engine::DenseBuf<Elt> Lb, Rb;
        if (const std::vector<Elt>* ld = general_flat(L)) {
            Lb = engine::DenseBuf<Elt>(const_cast<Elt*>(ld->data()), ld->size(), m, k, engine::Layout::RowMajor);
        } else {
            materialise(L, Lb);
        }
        if (const std::vector<Elt>* rd = general_flat(R)) {
            Rb = engine::DenseBuf<Elt>(const_cast<Elt*>(rd->data()), rd->size(), k, n, engine::Layout::RowMajor);
        } else {
            materialise(R, Rb);
        }
        engine::elim::sub_product<Elt>(ctx, engine::view(Tb), engine::cview(Lb), engine::cview(Rb));
    }
    FlatMatrixT<Elt> out(m, n, prime, t_data);
#ifdef ENGINE_SHADOW_LEGACY
    shadow_check_equal<Elt>(out, T - (L * R), "block_sub_product");
#endif
    return out;
}

// Core of RREF computations
template<class Elt>
void BlockFlatMatrixT<Elt>::_block_gaussian_elimination(
    BlockFlatMatrixT<Elt>& temp_bfm_output, // Matrix being modified to become RREF
    std::vector<std::pair<uint32_t, uint32_t>>& block_pivots_coords_out, // Coords in temp_bfm_output
    std::vector<typename BlockFlatMatrixT<Elt>::ScalarPivotInfoInternal>& fine_pivots_out, // Global scalar pivots in temp_bfm_output
    std::vector<std::pair<uint32_t, uint32_t>>& block_pivot_origins_out, // Original (br, bc) of blocks that became pivots
    std::map<std::pair<uint32_t, uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>>& original_block_fine_pivots_out // Fine pivots local to original P_block
) const {
    block_pivots_coords_out.clear();
    fine_pivots_out.clear();
    block_pivot_origins_out.clear(); // NEW
    original_block_fine_pivots_out.clear(); // NEW

    if (this->num_block_rows_ == 0 || this->num_block_cols_ == 0) {
        // temp_bfm_output is typically a copy of *this if called from compute_...
        // Ensure it's correctly representing *this state (e.g. if *this is empty).
        // The caller compute_block_row_echelon_form sets temp_bfm_output = *this initially.
        return;
    }

    std::vector<EliminationBlockRow<Elt>> work_matrix_list;
    work_matrix_list.reserve(this->num_block_rows_);
    for (uint32_t i = 0; i < this->num_block_rows_; ++i) {
        work_matrix_list.emplace_back(*this, i); // 'this' is the original matrix
    }

    std::vector<EliminationBlockRow<Elt>> processed_pivot_rows_list; // Rows that contain a chosen pivot

    uint32_t lead_block_col = 0;
    while (lead_block_col < this->num_block_cols_) {
        bool pivot_found_and_processed_in_column_pass;
        do {
            pivot_found_and_processed_in_column_pass = false;
            int best_candidate_ebr_idx = -1;
            int current_best_priority = 7;

            for (size_t ebr_idx = 0; ebr_idx < work_matrix_list.size(); ++ebr_idx) {
                const EliminationBlockRow<Elt>& candidate_row_ref = work_matrix_list[ebr_idx];
                if (candidate_row_ref.current_row_dim == 0) continue;
                const FlatMatrixT<Elt>& pivot_candidate_block_ref = candidate_row_ref.blocks[lead_block_col];
                int priority = get_pivot_block_priority(pivot_candidate_block_ref, this->p_);
                if (priority < current_best_priority) {
                    current_best_priority = priority;
                    best_candidate_ebr_idx = static_cast<int>(ebr_idx);
                }
            }

            if (best_candidate_ebr_idx == -1 || current_best_priority == 6) {
                break;
            }

            EliminationBlockRow<Elt> pivot_source_ebr = std::move(work_matrix_list[best_candidate_ebr_idx]);
            work_matrix_list.erase(work_matrix_list.begin() + best_candidate_ebr_idx);

            FlatMatrixT<Elt>& P_block = pivot_source_ebr.blocks[lead_block_col];
            std::pair<uint32_t, uint32_t> original_P_block_key = { pivot_source_ebr.original_label, lead_block_col }; // NEW

            auto [RREF_P, C_transform] = P_block.get_rref_and_transform_matrix();

            // RREF_P is already in reduced row echelon form. Compute its RREF cache
            // (which is itself) so that rank() and get_pivot_columns() don't recompute.
            RREF_P.compute_row_echelon_form();  // caches the RREF (trivial since already RREF)
            uint32_t k_rank_of_P = RREF_P.rank();
            const auto& local_piv_cols = RREF_P.get_pivot_columns();

            // Populate original_block_fine_pivots_out
            if (k_rank_of_P > 0) {
                std::vector<std::pair<uint32_t, uint32_t>> fine_pivs_for_this_orig_block;
                fine_pivs_for_this_orig_block.reserve(local_piv_cols.size());
                for (size_t i = 0; i < local_piv_cols.size(); ++i) {
                    fine_pivs_for_this_orig_block.push_back({ static_cast<uint32_t>(i), local_piv_cols[i] });
                }
                if (!fine_pivs_for_this_orig_block.empty()) {
                    original_block_fine_pivots_out[original_P_block_key] = fine_pivs_for_this_orig_block;
                }
            }

            if (k_rank_of_P == 0) {
                pivot_source_ebr.blocks[lead_block_col] = RREF_P; // which is zero
                for (uint32_t bc_apply = 0; bc_apply < this->num_block_cols_; ++bc_apply) {
                    if (bc_apply == lead_block_col) continue;
                    if (is_block_zero(pivot_source_ebr.blocks[bc_apply])) continue; // (Phase 1) C * 0 = 0
                    pivot_source_ebr.blocks[bc_apply] = C_transform * pivot_source_ebr.blocks[bc_apply];
                    const uint32_t z_rows = pivot_source_ebr.blocks[bc_apply].get_rows();
                    const uint32_t z_cols = pivot_source_ebr.blocks[bc_apply].get_cols();
                    if (pivot_source_ebr.blocks[bc_apply].is_zero()) pivot_source_ebr.blocks[bc_apply] = FlatMatrixT<Elt>::CreateZero(z_rows, z_cols, this->p_);
                }
                work_matrix_list.push_back(std::move(pivot_source_ebr));
                continue;
            }
            pivot_found_and_processed_in_column_pass = true;

            pivot_source_ebr.blocks[lead_block_col] = RREF_P;
            for (uint32_t bc_apply = 0; bc_apply < this->num_block_cols_; ++bc_apply) {
                if (bc_apply == lead_block_col) continue;
                if (is_block_zero(pivot_source_ebr.blocks[bc_apply])) continue; // (Phase 1) C * 0 = 0
                pivot_source_ebr.blocks[bc_apply] = C_transform * pivot_source_ebr.blocks[bc_apply];
                const uint32_t z_rows = pivot_source_ebr.blocks[bc_apply].get_rows();
                const uint32_t z_cols = pivot_source_ebr.blocks[bc_apply].get_cols();
                if (pivot_source_ebr.blocks[bc_apply].is_zero()) pivot_source_ebr.blocks[bc_apply] = FlatMatrixT<Elt>::CreateZero(z_rows, z_cols, this->p_);
            }

            std::vector<FlatMatrixT<Elt>> upper_part_row_blocks; upper_part_row_blocks.reserve(this->num_block_cols_);
            std::vector<FlatMatrixT<Elt>> lower_part_row_blocks; lower_part_row_blocks.reserve(this->num_block_cols_);
            uint32_t original_dim_of_pivot_source_row = pivot_source_ebr.current_row_dim;

            for (uint32_t bc_split = 0; bc_split < this->num_block_cols_; ++bc_split) {
                const FlatMatrixT<Elt>& current_block_in_row = pivot_source_ebr.blocks[bc_split];
                upper_part_row_blocks.push_back(current_block_in_row.extract_sub_matrix_rows(0, k_rank_of_P));
                if (original_dim_of_pivot_source_row > k_rank_of_P) {
                    uint32_t lower_rows_count = original_dim_of_pivot_source_row - k_rank_of_P;
                    FlatMatrixT<Elt> lower_block_segment = current_block_in_row.extract_sub_matrix_rows(k_rank_of_P, lower_rows_count);
                    if (lower_block_segment.is_zero()) {
                        lower_block_segment = FlatMatrixT<Elt>::CreateZero(lower_rows_count, current_block_in_row.get_cols(), this->p_);
                    }
                    lower_part_row_blocks.push_back(std::move(lower_block_segment));
                }
            }
            EliminationBlockRow<Elt> upper_ebr(std::move(upper_part_row_blocks), pivot_source_ebr.original_label, k_rank_of_P);

            // Store pivot coord relative to final position of processed_pivot_rows_list
            uint32_t pivot_block_row_in_rref = static_cast<uint32_t>(processed_pivot_rows_list.size());
            block_pivots_coords_out.push_back({ pivot_block_row_in_rref, lead_block_col });
            block_pivot_origins_out.push_back(original_P_block_key); // NEW: Store original block pivot location

            // Collect Fine Pivots from this upper_ebr segment (for fine_pivots_out)
            // upper_ebr.blocks[lead_block_col] == RREF_P (top k_rank_of_P rows)
            // Since RREF_P was already computed above, reuse its pivot columns directly.
            const std::vector<uint32_t>& local_scalar_pivot_cols_in_segment = local_piv_cols;
            for (size_t i = 0; i < local_scalar_pivot_cols_in_segment.size(); ++i) {
                uint32_t local_r = static_cast<uint32_t>(i);
                uint32_t local_c = local_scalar_pivot_cols_in_segment[i];
                fine_pivots_out.push_back({
                    pivot_block_row_in_rref,
                    lead_block_col,
                    local_r, local_c,
                    0, 0 // Global coords to be filled later by _block_gaussian_elimination after reconstruction
                    });
            }

            // A) Eliminate downwards in work_matrix_list (current 2e)
            const FlatMatrixT<Elt>& actual_pivot_block_for_elim = upper_ebr.blocks[lead_block_col];
            for (EliminationBlockRow<Elt>& target_ebr : work_matrix_list) {
                if (target_ebr.current_row_dim == 0) continue;
                FlatMatrixT<Elt>& block_in_target_at_pivot_col = target_ebr.blocks[lead_block_col];
                Elt bt_scalar_val;
                if (block_in_target_at_pivot_col.is_currently_scalar(bt_scalar_val)) {
                    if (bt_scalar_val == 0) continue;
                } else if (block_in_target_at_pivot_col.is_zero()) { // (Phase 1, defect #6) was a deep copy
                    continue;
                }

                auto [elim_block_new_val, L_factor] = block_in_target_at_pivot_col.compute_elimination_factor_and_apply(actual_pivot_block_for_elim);
                target_ebr.blocks[lead_block_col] = elim_block_new_val;
                const uint32_t z_rows = target_ebr.blocks[lead_block_col].get_rows();
                const uint32_t z_cols = target_ebr.blocks[lead_block_col].get_cols();
                if (target_ebr.blocks[lead_block_col].is_zero()) target_ebr.blocks[lead_block_col] = FlatMatrixT<Elt>::CreateZero(z_rows, z_cols, this->p_);

                for (uint32_t bc_elim_target = 0; bc_elim_target < this->num_block_cols_; ++bc_elim_target) {
                    if (bc_elim_target == lead_block_col) continue;
                    // (Phase 1) A zero block in the pivot row contributes
                    // L * 0 = 0, so the subtraction is the identity on the
                    // target.  Skipping it avoids two FlatMatrix constructions
                    // and an is_zero() scan per (pivot, target, column) triple --
                    // the dominant iteration count of factor_basis_block, whose
                    // augmented matrix is mostly zero blocks.  The scalar flags
                    // are a pure function of the data (every mutation ends in
                    // _determine_and_set_flags_from_data), so leaving the target
                    // untouched cannot change its pivot priority either.
                    if (is_block_zero(upper_ebr.blocks[bc_elim_target])) continue;
                    // (Phase 2) one fused GEMM: target -= L * pivot-row block
                    target_ebr.blocks[bc_elim_target] =
                        block_sub_product<Elt>(target_ebr.blocks[bc_elim_target], L_factor,
                                               upper_ebr.blocks[bc_elim_target], this->p_);
                    const uint32_t z_rows_bc = target_ebr.blocks[bc_elim_target].get_rows();
                    const uint32_t z_cols_bc = target_ebr.blocks[bc_elim_target].get_cols();
                    if (target_ebr.blocks[bc_elim_target].is_zero()) target_ebr.blocks[bc_elim_target] = FlatMatrixT<Elt>::CreateZero(z_rows_bc, z_cols_bc, this->p_);
                }
            }

            // B) Eliminate upwards in processed_pivot_rows_list
            const FlatMatrixT<Elt>& R_pivot_for_upward_elim = upper_ebr.blocks[lead_block_col];
            for (EliminationBlockRow<Elt>& existing_processed_ebr : processed_pivot_rows_list) {
                if (existing_processed_ebr.current_row_dim == 0) continue;
                FlatMatrixT<Elt>& B0_target_up = existing_processed_ebr.blocks[lead_block_col];
                Elt b0_scalar_val;
                if (B0_target_up.is_currently_scalar(b0_scalar_val)) {
                    if (b0_scalar_val == 0) continue;
                } else if (B0_target_up.is_zero()) { // (Phase 1, defect #6) was a deep copy
                    continue;
                }

                auto [B0_new_up, L_factor_up] = B0_target_up.compute_elimination_factor_and_apply(R_pivot_for_upward_elim);
                existing_processed_ebr.blocks[lead_block_col] = B0_new_up;
                const uint32_t z_rows = existing_processed_ebr.blocks[lead_block_col].get_rows();
                const uint32_t z_cols = existing_processed_ebr.blocks[lead_block_col].get_cols();
                if (existing_processed_ebr.blocks[lead_block_col].is_zero())
                    existing_processed_ebr.blocks[lead_block_col] = FlatMatrixT<Elt>::CreateZero(z_rows, z_cols, this->p_);

                for (uint32_t bc_apply_up = 0; bc_apply_up < this->num_block_cols_; ++bc_apply_up) {
                    if (bc_apply_up == lead_block_col) continue;
                    if (is_block_zero(upper_ebr.blocks[bc_apply_up])) continue; // (Phase 1) see above
                    // (Phase 2) one fused GEMM: target -= L * pivot-row block
                    existing_processed_ebr.blocks[bc_apply_up] =
                        block_sub_product<Elt>(existing_processed_ebr.blocks[bc_apply_up], L_factor_up,
                                               upper_ebr.blocks[bc_apply_up], this->p_);
                    const uint32_t z_rows_bc = existing_processed_ebr.blocks[bc_apply_up].get_rows();
                    const uint32_t z_cols_bc = existing_processed_ebr.blocks[bc_apply_up].get_cols();
                    if (existing_processed_ebr.blocks[bc_apply_up].is_zero())
                        existing_processed_ebr.blocks[bc_apply_up] = FlatMatrixT<Elt>::CreateZero(z_rows_bc, z_cols_bc, this->p_);
                }
            }

            // C) Move upper_ebr to processed_pivot_rows_list
            processed_pivot_rows_list.push_back(std::move(upper_ebr));

            // D) Add lower_part back to work_matrix_list if it has rows
            if (original_dim_of_pivot_source_row > k_rank_of_P) {
                EliminationBlockRow<Elt> lower_ebr(std::move(lower_part_row_blocks), pivot_source_ebr.original_label, original_dim_of_pivot_source_row - k_rank_of_P);
                work_matrix_list.push_back(std::move(lower_ebr));
            }
        } while (pivot_found_and_processed_in_column_pass);
        lead_block_col++;
    }

    // Reconstruct temp_bfm_output (remains the same as before)
    std::vector<uint32_t> final_block_row_dims;
    std::vector<std::vector<FlatMatrixT<Elt>>> final_block_data_layout;

    for (const auto& prow : processed_pivot_rows_list) {
        if (prow.current_row_dim > 0) {
            final_block_row_dims.push_back(prow.current_row_dim);
            final_block_data_layout.push_back(prow.blocks);
        }
    }
    std::sort(work_matrix_list.begin(), work_matrix_list.end(),
        [](const EliminationBlockRow<Elt>& a, const EliminationBlockRow<Elt>& b) {
            return a.original_label < b.original_label;
        });
    for (const auto& wrow : work_matrix_list) {
        if (wrow.current_row_dim > 0) {
            bool is_all_zero_blocks = true;
            for (const auto& block : wrow.blocks) { // (Phase 1, defect #6) was a deep copy per block
                if (!block.is_zero()) {
                    is_all_zero_blocks = false;
                    break;
                }
            }
            final_block_row_dims.push_back(wrow.current_row_dim);
            std::vector<FlatMatrixT<Elt>> current_row_blocks_for_layout;
            current_row_blocks_for_layout.reserve(wrow.blocks.size());
            for (const auto& block : wrow.blocks) { // (Phase 1, defect #6) was a deep copy per block
                if (is_all_zero_blocks || block.is_zero()) { // If entire row is zero, or this specific block is zero
                    current_row_blocks_for_layout.push_back(FlatMatrixT<Elt>::CreateZero(block.get_rows(), block.get_cols(), this->p_));
                }
                else {
                    current_row_blocks_for_layout.push_back(block);
                }
            }
            final_block_data_layout.push_back(current_row_blocks_for_layout);
        }
    }

    temp_bfm_output.block_row_dims_ = final_block_row_dims;
    temp_bfm_output.block_col_dims_ = this->block_col_dims_;
    temp_bfm_output.p_ = this->p_;
    temp_bfm_output._clear_cached_rref();
    temp_bfm_output.unique_blocks_.clear();
    temp_bfm_output.block_hash_index_.clear();
    temp_bfm_output._compute_total_dims();
    temp_bfm_output.block_layout_indices_.assign(temp_bfm_output.num_block_rows_, std::vector<size_t>());

    if (temp_bfm_output.num_block_rows_ > 0) {
        for (uint32_t br = 0; br < temp_bfm_output.num_block_rows_; ++br) {
            temp_bfm_output.block_layout_indices_[br].resize(temp_bfm_output.num_block_cols_);
            for (uint32_t bc = 0; bc < temp_bfm_output.num_block_cols_; ++bc) {
                if (br < final_block_data_layout.size() && bc < final_block_data_layout[br].size()) {
                    temp_bfm_output.block_layout_indices_[br][bc] = temp_bfm_output._add_or_get_block_index(final_block_data_layout[br][bc]);
                }
                else {
                    uint32_t r_dim_for_empty = (br < temp_bfm_output.block_row_dims_.size()) ? temp_bfm_output.block_row_dims_[br] : 0;
                    uint32_t c_dim_for_empty = (bc < temp_bfm_output.block_col_dims_.size()) ? temp_bfm_output.block_col_dims_[bc] : 0;
                    temp_bfm_output.block_layout_indices_[br][bc] = temp_bfm_output._add_or_get_block_index(FlatMatrixT<Elt>::CreateZero(r_dim_for_empty, c_dim_for_empty, this->p_));
                }
            }
        }
    }

    // Compute absolute global coordinates for fine_pivots_out (scalar pivots *in temp_bfm_output*)
    for (ScalarPivotInfoInternal& spi : fine_pivots_out) {
        // spi.block_row_in_rref_cache and spi.block_col_in_rref_cache are indices for temp_bfm_output's block structure
        uint32_t global_r_offset = 0;
        for (uint32_t br_idx = 0; br_idx < spi.block_row_in_rref_cache; ++br_idx) {
            if (br_idx < temp_bfm_output.block_row_dims_.size()) { // Check bounds
                global_r_offset += temp_bfm_output.block_row_dims_[br_idx];
            }
            else { /* Handle error or inconsistent state */ }
        }
        spi.global_r = global_r_offset + spi.local_row_in_block;

        uint32_t global_c_offset = 0;
        for (uint32_t bc_idx = 0; bc_idx < spi.block_col_in_rref_cache; ++bc_idx) {
            if (bc_idx < temp_bfm_output.block_col_dims_.size()) { // Check bounds
                global_c_offset += temp_bfm_output.block_col_dims_[bc_idx];
            }
            else { /* Handle error or inconsistent state */ }
        }
        spi.global_c = global_c_offset + spi.local_col_in_block;
    }
}

// --- compute_block_row_echelon_form (Orchestrator) ---
template<class Elt>
void BlockFlatMatrixT<Elt>::compute_block_row_echelon_form() const {
    // Check if all necessary caches are already populated
    if (rref_cache_ && fine_pivot_info_cache_) {
        return;
    }
    _clear_cached_rref(); // Clear all caches

    // std::cout << "  Computing BLOCK row echelon form for " << num_block_rows_ << "x" << num_block_cols_ << " block structure (" << get_total_rows() << "x" << get_total_cols() << ") ..." << std::endl;

    // Allocate all cache members
    rref_cache_ = std::make_unique<BlockFlatMatrixT<Elt>>(p_);
    *rref_cache_ = *this; // Make rref_cache_ a copy of the current matrix to be modified

    fine_pivot_info_cache_ = std::make_unique<std::vector<ScalarPivotInfoInternal>>();

    // (matrix-engine Phase 3) The block-pivot coordinate list and the two
    // pivot-origin maps were cached members only so that the validation-only
    // accessors get_block_pivot_coords_in_rref() / get_block_rref_intermediate_form()
    // could hand out pointers into them (audit defect #16, dangling on any
    // mutation).  Both accessors are gone; the elimination still fills these
    // out-parameters, so they are now plain locals -- three fewer cached
    // allocations per RREF, and the results are bit-identical.
    std::vector<std::pair<uint32_t, uint32_t>> block_pivot_coords;
    std::vector<std::pair<uint32_t, uint32_t>> block_pivot_origins;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>> original_block_fine_pivots;

    if (num_block_rows_ == 0 || num_block_cols_ == 0) {
        // For empty/0-dim matrices, *rref_cache_ (which is a copy of *this) is already the RREF.
        // The other caches remain empty, which is correct.
        return;
    }

    // Call the modified _block_gaussian_elimination
    _block_gaussian_elimination(*rref_cache_,
        block_pivot_coords,
        *fine_pivot_info_cache_,
        block_pivot_origins,
        original_block_fine_pivots
    );
    // rref_cache_ is now modified in-place to be the RREF form.

    // std::cout << "  Block row echelon form computed." << std::endl;
}


// Defect #32 (matrix-engine Phase 0): get_block_pivot_coords() was deleted here.
// It had zero callers (including in the validations) and duplicated
// get_block_pivot_coords_in_rref(), but returned a NON-const reference into
// the RREF cache (defect #17).

// This helper identifies definitive scalar pivots from rref_cache_ and their global coordinates.
// A definitive pivot is the first one encountered in a column when scanning rows top-to-bottom.
// (This is subtly different from the previous _compute_true_scalar_rref_and_pivots,
//  as we now directly use fine_pivot_info_cache_ and assume rref_cache_ is already "RREF-like")
// Output: vector of (global_r_in_rref_cache, global_c_idx_in_rref_cache)
// Also add debug prints to _get_definitive_global_scalar_pivots_from_rref_cache
template<class Elt>
std::vector<std::pair<uint32_t, uint32_t>> BlockFlatMatrixT<Elt>::_get_definitive_global_scalar_pivots_from_rref_cache() const {


    compute_block_row_echelon_form();

    if (!fine_pivot_info_cache_ || fine_pivot_info_cache_->empty()) {
        return {};
    }

    std::vector<ScalarPivotInfoInternal> sorted_fine_pivots = *fine_pivot_info_cache_;
    std::sort(sorted_fine_pivots.begin(), sorted_fine_pivots.end(),
        [](const ScalarPivotInfoInternal& a, const ScalarPivotInfoInternal& b) {
            if (a.global_c != b.global_c) {
                return a.global_c < b.global_c;
            }
            return a.global_r < b.global_r;
        });


    std::vector<std::pair<uint32_t, uint32_t>> definitive_pivots;
    if (sorted_fine_pivots.empty()) {
        return definitive_pivots;
    }

    definitive_pivots.push_back({ sorted_fine_pivots[0].global_r, sorted_fine_pivots[0].global_c });
    for (size_t i = 1; i < sorted_fine_pivots.size(); ++i) {
        if (sorted_fine_pivots[i].global_c > definitive_pivots.back().second) {
            definitive_pivots.push_back({ sorted_fine_pivots[i].global_r, sorted_fine_pivots[i].global_c });
        }
    }


    return definitive_pivots;
}


// (2c)
template<class Elt>
std::pair<FlatMatrixT<Elt>, std::vector<std::pair<uint32_t, uint32_t>>> BlockFlatMatrixT<Elt>::get_true_row_echelon_form_as_flat_matrix() const {


    compute_block_row_echelon_form(); // Ensures rref_cache_ and fine_pivot_info_cache_ are populated

    // (Debug printing for rref_cache_ and fine_pivot_info_cache_ can remain here if useful)

    if (!rref_cache_) {
        if (this->total_rows_ == 0 || this->total_cols_ == 0) {
            return { FlatMatrixT<Elt>::CreateZero(this->total_rows_, this->total_cols_, this->p_), {} };
        }
        throw std::logic_error("rref_cache_ not populated for get_true_row_echelon_form_as_flat_matrix.");
    }

    // Step 1: Get all definitive scalar pivots (global_r_in_rref_cache, global_c_in_rref_cache)
    // This list is already sorted by _get_definitive_global_scalar_pivots_from_rref_cache:
    // primary key global_c, secondary key global_r.
    std::vector<std::pair<uint32_t, uint32_t>> definitive_pivots_sorted_by_col = _get_definitive_global_scalar_pivots_from_rref_cache();


    uint32_t true_rank = static_cast<uint32_t>(definitive_pivots_sorted_by_col.size());
    std::vector<std::pair<uint32_t, uint32_t>> final_global_scalar_pivots_in_true_rref;
    final_global_scalar_pivots_in_true_rref.reserve(true_rank);

    // Special case: no definitive scalar pivots => the true RREF is the zero matrix.
    // Return the historical CreateZero(...) result UNCHANGED (scalar-zero, EMPTY flat_data_),
    // preserving byte-identical internal representation for this case. (A pre-zeroed buffer fed
    // to the DATA ctor would instead yield scalar-zero with a NON-empty all-zero flat_data_.)
    if (definitive_pivots_sorted_by_col.empty()) {
        return { FlatMatrixT<Elt>::CreateZero(this->total_rows_, this->total_cols_, this->p_),
                 final_global_scalar_pivots_in_true_rref };
    }

    // Step 2: Assemble the true RREF into a single pre-zeroed bulk buffer, then hand it to the
    // FlatMatrix DATA constructor ONCE. The old code called set_entry() for every cell, and each
    // set_entry() runs _determine_and_set_flags_from_data() (O(rows*cols)) => O(N^2), which stalls
    // at dim ~1000. Byte-identical to the old set_entry loop: same source values read the same way
    // (rref_cache_->get_entry) into the same target positions, a pre-zeroed background for all
    // non-pivot rows, the same mod-p normalization (the DATA ctor applies mod_add_w(x,0,p_)), and a
    // single _determine_and_set_flags_from_data() over the final data (null caches in both). The
    // order of definitive_pivots_sorted_by_col (sorted by column first) sets the pivot-column order;
    // the row taken for each pivot column is rref_cache_[pair.first].
    std::vector<Elt> true_rref_buf(static_cast<size_t>(this->total_rows_) * this->total_cols_, Elt(0));

    uint32_t current_true_rref_row_idx = 0;
    for (const auto& pivot_coord_pair : definitive_pivots_sorted_by_col) {
        uint32_t r_in_rref_cache_to_copy = pivot_coord_pair.first; // Row index from rref_cache_
        uint32_t c_pivot_in_this_row = pivot_coord_pair.second; // Global column of this pivot

        if (current_true_rref_row_idx >= this->total_rows_) { // Should not happen if rank <= total_rows
            break;
        }

        // Per-pivot data invariant: a definitive scalar pivot must read as a
        // non-zero entry. Cost scales with the rank, so it is debug-only.
        ENGINE_DEBUG_CHECK_MSG(rref_cache_->get_entry(r_in_rref_cache_to_copy, c_pivot_in_this_row) != 0,
                               "definitive pivot entry unexpectedly zero");

        // (Phase 1) One block-wise pass per pivot row instead of total_cols_
        // per-entry BFM::get_entry calls.  Same values, same positions, on the
        // same pre-zeroed background.
        const size_t dst_row_base = static_cast<size_t>(current_true_rref_row_idx) * this->total_cols_;
        rref_cache_->_copy_scalar_row_to_buffer(r_in_rref_cache_to_copy, true_rref_buf.data() + dst_row_base);
        final_global_scalar_pivots_in_true_rref.push_back({ current_true_rref_row_idx, c_pivot_in_this_row });
        current_true_rref_row_idx++;
    }

    // O(1) structural invariants -- always on (NDEBUG policy, EngineCheck.h):
    // rank never exceeds the row count; the pivot list length matches the
    // number of rows actually filled.
    ENGINE_CHECK_MSG(current_true_rref_row_idx <= this->total_rows_,
                     "true RREF pivot count exceeds total_rows_");
    ENGINE_CHECK(final_global_scalar_pivots_in_true_rref.size() == current_true_rref_row_idx);

    FlatMatrixT<Elt> true_rref_flat(this->total_rows_, this->total_cols_, this->p_, true_rref_buf);

    // (matrix-engine Phase 3, plan Sec. D5b) The block layer's RREF-idempotency
    // self-check, revived.  It sat behind `#ifdef DEBUG_BLOCKFLATMATRIX` -- a
    // macro commented out in BlockFlatMatrix.h and therefore defined in no
    // build -- with its only corrective action commented out on top of that, so
    // it was decoration in every configuration ever shipped.  The invariant it
    // states is real and load-bearing: the rows assembled here, one per
    // definitive pivot in ascending pivot-column order, must ALREADY be the
    // canonical RREF of *this, i.e. running FlatMatrix's own elimination over
    // them must be the identity.  If _block_gaussian_elimination ever left
    // rref_cache_ less than RREF-clean, every downstream basis (flat_kernel,
    // flat_image, factor_basis_block -> C_eq, Z_eq, W, B_eq, H_eq, P_q) would
    // change silently -- exactly the D5b failure mode.
    //
    // Cost is a second full elimination, O(m*n*rank), so it is an
    // ENGINE_DEBUG_CHECK: compiled out under NDEBUG (RelWithDebInfo / Release /
    // Bench), live in Debug and in the -fsanitize `Check` configuration.
    ENGINE_DEBUG_CHECK_MSG(
        [&]() {
            FlatMatrixT<Elt> probe = true_rref_flat;
            probe.rank();  // forces the RREF cache
            return probe.get_row_echelon_form_as_flat_matrix() == true_rref_flat;
        }(),
        "block-layer true RREF is not idempotent: rref_cache_ was not RREF-clean");

    return { true_rref_flat, final_global_scalar_pivots_in_true_rref };
}



// --- Kernel and Image related helpers and methods ---
// These helpers need to operate on the *final RREF structure*, usually rref_cache_

template<class Elt>
uint32_t BlockFlatMatrixT<Elt>::_get_block_start_row_idx(uint32_t block_row_layout_idx) const {
    // This helper should refer to the BlockFlatMatrix instance it's called on (e.g. rref_cache_ or *this*)
    if (block_row_layout_idx >= this->num_block_rows_) { // Use this->num_block_rows_ for consistency
        throw std::out_of_range("Block row layout index out of range in _get_block_start_row_idx.");
    }
    return this->block_row_offsets_[block_row_layout_idx]; // (Phase 1, defect #8) O(1)
}

template<class Elt>
uint32_t BlockFlatMatrixT<Elt>::_get_block_start_col_idx(uint32_t block_col_layout_idx) const {
    if (block_col_layout_idx >= this->num_block_cols_) {
        throw std::out_of_range("Block col layout index out of range in _get_block_start_col_idx.");
    }
    return this->block_col_offsets_[block_col_layout_idx]; // (Phase 1, defect #8) O(1)
}


// --- Phase 3: Linear Algebra Method Implementations ---

template<class Elt>
uint32_t BlockFlatMatrixT<Elt>::flat_rank() const { // (defect #19)
    // This method used to be rank(). Renamed for clarity.
    // It relies on the true scalar RREF.
    auto [_, final_global_scalar_pivots] = get_true_row_echelon_form_as_flat_matrix();
    return static_cast<uint32_t>(final_global_scalar_pivots.size());
}

template<class Elt>
bool BlockFlatMatrixT<Elt>::is_invertible() const { // (defect #19)
    if (this->total_rows_ != this->total_cols_ || this->total_rows_ == 0) {
        return false;
    }
    return this->flat_rank() == this->total_rows_;
}

template<class Elt>
FlatMatrixT<Elt> BlockFlatMatrixT<Elt>::flat_inverse() const {
    // (Phase 1, defect #6) The whole-BlockFlatMatrix deep copy made here purely to
    // reach the then-non-const is_invertible() is gone.
    if (!is_invertible()) {
        throw std::runtime_error("BlockFlatMatrix (when flattened) is not invertible.");
    }

    // Augment the original matrix *this* (flattened) with a scalar identity
    FlatMatrixT<Elt> M_flat = this->to_flat_matrix(); // Original matrix content
    if (M_flat.get_rows() == 0) { // Handle 0x0 case
        return FlatMatrixT<Elt>(0, 0, this->p_); // 0x0 identity
    }

    FlatMatrixT<Elt> I_flat = FlatMatrixT<Elt>(M_flat.get_rows(), this->p_); // Identity of same dimension

    // Create augmented matrix [M_flat | I_flat]
    FlatMatrixT<Elt> augmented_flat(M_flat.get_rows(), M_flat.get_cols() * 2, this->p_);
    augmented_flat.populate_flat_data(); // Make it general, 0-filled

    for (uint32_t r = 0; r < M_flat.get_rows(); ++r) {
        for (uint32_t c = 0; c < M_flat.get_cols(); ++c) {
            augmented_flat.set_entry(r, c, M_flat.get_entry(r, c));
        }
        augmented_flat.set_entry(r, M_flat.get_cols() + r, 1); // Identity part
    }

    // Compute RREF of the augmented FlatMatrix
    // FlatMatrix::get_row_echelon_form_as_flat_matrix() is the scalar RREF method
    FlatMatrixT<Elt> rref_augmented_flat = augmented_flat.get_row_echelon_form_as_flat_matrix();

    // Extract the inverse part
    FlatMatrixT<Elt> inverse_part = FlatMatrixT<Elt>::CreateZero(M_flat.get_rows(), M_flat.get_cols(), this->p_);
    inverse_part.populate_flat_data();
    for (uint32_t r = 0; r < M_flat.get_rows(); ++r) {
        for (uint32_t c = 0; c < M_flat.get_cols(); ++c) {
            inverse_part.set_entry(r, c, rref_augmented_flat.get_entry(r, M_flat.get_cols() + c));
        }
    }
    // The constructor of FlatMatrix from data (or set_entry) should handle flags.
    // Forcing a flag check for robustness (if set_entry is not fully robust).
    inverse_part.is_zero(); // This updates flags if it became scalar zero or some other scalar.

    return inverse_part;
}



template<class Elt>
std::vector<BlockFlatMatrixT<Elt>> BlockFlatMatrixT<Elt>::flat_kernel() const { // (defect #19)
    auto [true_rref_flat, final_global_scalar_pivots] = get_true_row_echelon_form_as_flat_matrix();

    std::vector<BlockFlatMatrixT<Elt>> kernel_basis_bfm;
    if (this->total_cols_ == 0) return kernel_basis_bfm;

    std::vector<uint32_t> global_scalar_free_col_indices;
    std::vector<bool> is_pivot_col(this->total_cols_, false);
    for (const auto& piv : final_global_scalar_pivots) {
        if (piv.second < this->total_cols_) is_pivot_col[piv.second] = true;
    }
    for (uint32_t c = 0; c < this->total_cols_; ++c) {
        if (!is_pivot_col[c]) global_scalar_free_col_indices.push_back(c);
    }

    // (matrix-engine Phase 2) All kernel vectors are assembled in ONE buffer by
    // engine::elim::kernel_basis (the D5b rule: one vector per free column
    // ascending, 1 at the free column, -RREF[i][j_free] at pivot i), and every
    // block of every vector is built by the DATA constructor once.  The old
    // code called set_entry() per coordinate -- each a full O(total_cols_) flag
    // rescan -- i.e. O(total_cols_^2) per vector.  Same values; an all-zero
    // block is CreateZero (scalar-0, empty storage), as the set_entry loop left
    // it.  The rows of the true RREF are one per pivot in ascending pivot-
    // column order (BlockFlatMatrix's assembly contract), so pivots[i] is the
    // pivot of row i, which is what kernel_basis assumes.
    std::vector<uint32_t> pivot_cols;
    pivot_cols.reserve(final_global_scalar_pivots.size());
    for (const auto& piv : final_global_scalar_pivots) {
        ENGINE_CHECK_MSG(piv.first == pivot_cols.size(), "flat_kernel: true RREF rows are not one per pivot in order");
        pivot_cols.push_back(piv.second);
    }
    const uint32_t k = static_cast<uint32_t>(pivot_cols.size());
    engine::DenseBuf<Elt> R(k, this->total_cols_);
    for (uint32_t i = 0; i < k; ++i)
        for (uint32_t j = 0; j < this->total_cols_; ++j) R.at(i, j) = true_rref_flat.get_entry(i, j);
    const engine::ModCtx<Elt> ctx(this->p_);
    const engine::DenseBuf<Elt> K = engine::elim::kernel_basis<Elt>(ctx, engine::cview(R), pivot_cols);
    ENGINE_CHECK(K.rows == global_scalar_free_col_indices.size());

    kernel_basis_bfm.reserve(K.rows);
    for (uint32_t t = 0; t < K.rows; ++t) {
        const Elt* kv = K.data() + static_cast<size_t>(t) * K.ld;
        // Kernel vector has M's original column structure as its row structure
        BlockFlatMatrixT<Elt> kv_bfm(this->block_col_dims_, { 1 } /*single scalar column*/, this->p_);
        uint32_t current_kv_row_offset = 0;
        for (uint32_t br_kv = 0; br_kv < kv_bfm.get_num_block_rows(); ++br_kv) {
            uint32_t block_h = kv_bfm.get_block_row_dim(br_kv);
            if (block_h == 0) continue;
            std::vector<Elt> block_data(kv + current_kv_row_offset, kv + current_kv_row_offset + block_h);
            bool all_zero = true;
            for (Elt v : block_data) if (v != 0) { all_zero = false; break; }
            if (!all_zero) {
                kv_bfm.set_block_entry(br_kv, 0, FlatMatrixT<Elt>(block_h, 1, this->p_, block_data));
            }   // else: the constructor's zero block already is CreateZero(block_h, 1)
            current_kv_row_offset += block_h;
        }
        kernel_basis_bfm.push_back(std::move(kv_bfm));
    }
    return kernel_basis_bfm;
}



template<class Elt>
std::vector<BlockFlatMatrixT<Elt>> BlockFlatMatrixT<Elt>::flat_image() const { // (defect #19)
    auto [_, final_global_scalar_pivots] = get_true_row_echelon_form_as_flat_matrix();

    std::vector<BlockFlatMatrixT<Elt>> image_basis_bfm;
    if (this->total_rows_ == 0) return image_basis_bfm;

    for (const auto& piv_info : final_global_scalar_pivots) {
        uint32_t gspc = piv_info.second; // Global scalar pivot column from true RREF context (column in *this*)
        FlatMatrixT<Elt> orig_scalar_col_fm = _get_original_scalar_column_as_flatmatrix(gspc); // from *this*

        // Convert orig_scalar_col_fm to BlockFlatMatrix
        // Image vector has M's original row structure
        BlockFlatMatrixT<Elt> im_vec_bfm(this->block_row_dims_, { 1 } /*single scalar column*/, this->p_);
        uint32_t current_im_row_offset = 0;
        for (uint32_t br_im = 0; br_im < im_vec_bfm.get_num_block_rows(); ++br_im) {
            uint32_t block_h = im_vec_bfm.get_block_row_dim(br_im);
            if (block_h == 0) continue;
            FlatMatrixT<Elt> block_data_for_im_bfm(block_h, 1, this->p_);
            for (uint32_t r_in_block = 0; r_in_block < block_h; ++r_in_block) {
                block_data_for_im_bfm.set_entry(r_in_block, 0, orig_scalar_col_fm.get_entry(current_im_row_offset + r_in_block, 0));
            }
            im_vec_bfm.set_block_entry(br_im, 0, block_data_for_im_bfm);
            current_im_row_offset += block_h;
        }
        image_basis_bfm.push_back(std::move(im_vec_bfm));
    }
    return image_basis_bfm;
}




// (Phase 1) Block-wise row copy.  `get_true_row_echelon_form_as_flat_matrix`
// used to materialise the whole scalar RREF through BFM::get_entry, one call per
// cell; even with the O(1) offsets of defect #8 that is a binary search, a block
// lookup and a FlatMatrix::get_entry per cell.  Here each block of the row is
// visited once and its `local_r`-th row is copied straight out of the block's
// own storage (or synthesised, for the scalar-block representation).
//
// `dst` points at total_cols_ PRE-ZEROED elements; zero blocks are therefore
// skipped entirely, which is what makes this cheap on the sparse block rows.
template<class Elt>
void BlockFlatMatrixT<Elt>::_copy_scalar_row_to_buffer(uint32_t global_r, Elt* dst) const {
    if (global_r >= total_rows_) {
        throw std::out_of_range("Row index out of range in _copy_scalar_row_to_buffer.");
    }
    const uint32_t br = static_cast<uint32_t>(
        std::upper_bound(block_row_offsets_.begin(), block_row_offsets_.begin() + num_block_rows_ + 1, global_r)
        - block_row_offsets_.begin()) - 1;
    const uint32_t local_r = global_r - block_row_offsets_[br];

    const std::vector<size_t>& layout_row = block_layout_indices_[br];
    for (uint32_t bc = 0; bc < num_block_cols_; ++bc) {
        const FlatMatrixT<Elt>& block = unique_blocks_[layout_row[bc]];
        const uint32_t width = block_col_dims_[bc];
        if (width == 0) continue;
        Elt* out = dst + block_col_offsets_[bc];

        Elt scalar_val;
        if (block.is_currently_scalar(scalar_val)) {
            // alpha * I: the only non-zero entry of row local_r is at column
            // local_r (alpha != 0 implies the block is square).  alpha == 0
            // leaves the pre-zeroed background untouched.
            if (scalar_val != 0 && local_r < width) out[local_r] = scalar_val;
            continue;
        }
        const std::vector<Elt>& data = block.get_flat_data_unsafe();
        if (data.size() == static_cast<size_t>(block.get_rows()) * width) {
            const Elt* src = data.data() + static_cast<size_t>(local_r) * width;
            std::copy(src, src + width, out);
        }
        else { // defensive: general block without materialised storage
            for (uint32_t c = 0; c < width; ++c) out[c] = block.get_entry(local_r, c);
        }
    }
}

template<class Elt>
FlatMatrixT<Elt> BlockFlatMatrixT<Elt>::_get_original_scalar_column_as_flatmatrix(uint32_t global_col_idx) const {
    if (global_col_idx >= total_cols_) {
        throw std::out_of_range("Global column index out of range in _get_original_scalar_column_as_flatmatrix.");
    }
    FlatMatrixT<Elt> scalar_col_fm = FlatMatrixT<Elt>::CreateZero(total_rows_, 1, p_);
    if (total_rows_ == 0) return scalar_col_fm;
    // scalar_col_fm needs to be mutable to use set_entry, CreateZero should prepare it.
    for (uint32_t r = 0; r < total_rows_; ++r) {
        scalar_col_fm.set_entry(r, 0, this->get_entry(r, global_col_idx)); // get_entry from *this*
    }
    return scalar_col_fm;
}



// ===========================================================================
// factor_basis_block
//
// (matrix-engine Phase 1, defect #22) Moved verbatim-in-behaviour out of the
// deprecated Matrix.cpp, where this block-layer routine had no business living:
// it never touched MatrixT at all.  Two densification fixes on top of the move
// (02 §2.1, the ~22 % of compute_cohomology spent outside its kernels):
//
//   (a) the relation / generator column blocks and the projector are assembled
//       into bulk buffers and handed to the FlatMatrix DATA constructor once,
//       instead of cell-by-cell through set_entry() -- each of which re-runs
//       _determine_and_set_flags_from_data() over the whole block (defect #4);
//   (b) the augmented RREF is no longer read back cell-by-cell (that read is now
//       block-wise inside get_true_row_echelon_form_as_flat_matrix itself).
//
// The block structure of `augmented`, the pivot classification and the
// projector's contents are unchanged, so the result is bit-identical.  The one
// representational nicety kept from the old code: an all-zero column block is
// built by CreateZero (scalar-0 with EMPTY storage) rather than by the DATA
// constructor (scalar-0 with an all-zero buffer), matching what the old
// set_entry loop produced.
// ===========================================================================
template<class Elt>
std::pair<FlatMatrixT<Elt>, std::vector<uint32_t>> factor_basis_block(
    const std::vector<uint32_t>& block_row_dims,
    const std::vector<FlatMatrixT<Elt>>& relations,
    const std::vector<FlatMatrixT<Elt>>& generators,
    std::vector<FlatMatrixT<Elt>>* factor_basis_out,
    Elt prime
) {
    // Edge case: no generators
    if (generators.empty()) {
        return { FlatMatrixT<Elt>(0, 0, prime), {} };
    }

    uint32_t n_block_rows = static_cast<uint32_t>(block_row_dims.size());
    uint32_t ambient_dim = 0;
    for (auto d : block_row_dims) ambient_dim += d;

    uint32_t n_rel = static_cast<uint32_t>(relations.size());
    uint32_t n_gen = static_cast<uint32_t>(generators.size());

    // Build block_col_dims: n_rel 1s + n_gen 1s + block_row_dims
    std::vector<uint32_t> block_col_dims;
    block_col_dims.reserve(n_rel + n_gen + n_block_rows);
    for (uint32_t j = 0; j < n_rel; ++j) block_col_dims.push_back(1);
    for (uint32_t j = 0; j < n_gen; ++j) block_col_dims.push_back(1);
    for (uint32_t br = 0; br < n_block_rows; ++br) block_col_dims.push_back(block_row_dims[br]);

    BlockFlatMatrixT<Elt> augmented(block_row_dims, block_col_dims, prime);

    // Fill relation columns: each relation is a column vector of length
    // ambient_dim, split into blocks.  Bulk-built, one FlatMatrix construction
    // per (block row, column) instead of `d` set_entry calls.
    auto make_column_block = [&](const FlatMatrixT<Elt>& src, uint32_t row_offset, uint32_t d) {
        std::vector<Elt> buf(d);
        bool all_zero = true;
        for (uint32_t r = 0; r < d; ++r) {
            Elt v = src.get_entry(row_offset + r, 0);
            buf[r] = v;
            if (v != 0) all_zero = false;
        }
        if (all_zero) return FlatMatrixT<Elt>::CreateZero(d, 1, prime);
        return FlatMatrixT<Elt>(d, 1, prime, buf);
    };

    uint32_t row_offset = 0;
    for (uint32_t br = 0; br < n_block_rows; ++br) {
        uint32_t d = block_row_dims[br];
        for (uint32_t j = 0; j < n_rel; ++j) {
            augmented.set_block_entry(br, j, make_column_block(relations[j], row_offset, d));
        }
        for (uint32_t j = 0; j < n_gen; ++j) {
            augmented.set_block_entry(br, n_rel + j, make_column_block(generators[j], row_offset, d));
        }
        // Fill identity: d x d identity block on the diagonal (scalar-1 storage, O(1))
        augmented.set_block_entry(br, n_rel + n_gen + br, FlatMatrixT<Elt>(d, prime));
        row_offset += d;
    }

    // Compute RREF
    auto [rref_flat, pivots] = augmented.get_true_row_echelon_form_as_flat_matrix();

    // Classify pivots
    uint32_t independent_relations = 0;
    std::vector<uint32_t> factor_basis_pivots;
    for (auto& [piv_row, piv_col] : pivots) {
        if (piv_col < n_rel) {
            independent_relations++;
        } else if (piv_col < n_rel + n_gen) {
            uint32_t gen_idx = piv_col - n_rel;
            factor_basis_pivots.push_back(gen_idx);
            if (factor_basis_out) {
                factor_basis_out->push_back(generators[gen_idx]);
            }
        }
        // identity region pivots: ignore
    }

    // Extract projector: dim_H x ambient_dim.  rref_flat is a dense FlatMatrix,
    // so the reads are O(1); the writes go into one bulk buffer.
    uint32_t dim_H = static_cast<uint32_t>(factor_basis_pivots.size());
    uint32_t identity_start_col = n_rel + n_gen;
    if (dim_H == 0 || ambient_dim == 0) {
        return { FlatMatrixT<Elt>::CreateZero(dim_H, ambient_dim, prime), factor_basis_pivots };
    }
    std::vector<Elt> projector_buf(static_cast<size_t>(dim_H) * ambient_dim);
    bool all_zero = true;
    for (uint32_t i = 0; i < dim_H; ++i) {
        uint32_t rref_row = independent_relations + i;
        const size_t base = static_cast<size_t>(i) * ambient_dim;
        for (uint32_t j = 0; j < ambient_dim; ++j) {
            Elt v = rref_flat.get_entry(rref_row, identity_start_col + j);
            projector_buf[base + j] = v;
            if (v != 0) all_zero = false;
        }
    }
    if (all_zero) {
        return { FlatMatrixT<Elt>::CreateZero(dim_H, ambient_dim, prime), factor_basis_pivots };
    }
    return { FlatMatrixT<Elt>(dim_H, ambient_dim, prime, projector_buf), factor_basis_pivots };
}

// ===========================================================================
// Explicit instantiations (Stage S13): uint32_t (primary regime, alias
// BlockFlatMatrix) and uint64_t (extended regime, alias BlockFlatMatrix64).
// ===========================================================================
template class BlockFlatMatrixT<uint32_t>;
template class BlockFlatMatrixT<uint64_t>;
template std::pair<FlatMatrixT<uint32_t>, std::vector<uint32_t>> factor_basis_block<uint32_t>(
    const std::vector<uint32_t>&, const std::vector<FlatMatrixT<uint32_t>>&,
    const std::vector<FlatMatrixT<uint32_t>>&, std::vector<FlatMatrixT<uint32_t>>*, uint32_t);
template std::pair<FlatMatrixT<uint64_t>, std::vector<uint32_t>> factor_basis_block<uint64_t>(
    const std::vector<uint32_t>&, const std::vector<FlatMatrixT<uint64_t>>&,
    const std::vector<FlatMatrixT<uint64_t>>&, std::vector<FlatMatrixT<uint64_t>>*, uint64_t);
