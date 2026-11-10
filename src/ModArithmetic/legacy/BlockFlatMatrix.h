// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef BLOCK_FLAT_MATRIX_H
#define BLOCK_FLAT_MATRIX_H

#include "../FlatMatrix.h"
#include "../ObjectCounters.h"  // (matrix-engine Phase 3) opt-in construction census
#include <vector>
#include <string>
#include <numeric> // For std::accumulate, std::iota
#include <stdexcept>
#include <map> // For std::map
#include <unordered_map>
#include <memory>

// ===========================================================================
// DEPRECATED -- LEGACY ARM ONLY (matrix-engine Phase 3).
//
// `BlockFlatMatrixT` is the pre-Phase-3 block container.  The live block model
// is `engine::BlockMat` / `engine::blk` (see ENGINE_API.md Sec. 9); this class
// survives only so that `ENGINE_BACKEND=legacy` still runs the pre-Phase-2
// substrate end to end, and it is REMOVED once Phase 4 ships.
//
// Do not add call sites and do not add members.  Its remaining surface is
// exactly what the legacy arm of `EquivariantCohomologyOnK::compute_cohomology`
// and `SimplicialComplex::coboundary_operator_block_elt` need, plus what the
// differential validations compare against:
//
//     ctor(row_dims, col_dims, p) / ctor(..., block_data, p),
//     set_block_entry, get_block_entry, get_entry, to_flat_matrix,
//     operator*(BlockFlatMatrixT), is_zero,
//     compute_block_row_echelon_form, get_true_row_echelon_form_as_flat_matrix,
//     flat_rank, is_invertible, flat_inverse, flat_kernel, flat_image,
//     factor_basis_block (free function),
//     dimension accessors.
//
// The ~20 members that existed only to be exercised by
// BlockFlatMatrixValidation -- CreateZero/CreateIdentity, swap, transpose,
// operator+/-/scalar/==/!=, the intermediate-RREF accessors, the block-shaped
// RREF/kernel/image/inverse, and the *_for_test getters -- were deleted in
// Phase 3 once BlockMatValidation / BlockEliminationValidation covered the same
// properties ENTRYWISE on the new model.
//
// The set of translation units allowed to include this header is enforced by
// docs/matrix-engine/tools/check_matrix_includes.sh (run by
// MatrixRetirementValidation).  A type-level [[deprecated]] attribute is
// deliberately NOT used: it was tried and measured -- `class [[deprecated]]
// BlockFlatMatrixT` produces 103 "-Wdeprecated-declarations" warnings on a
// clean build of the ModArithmetic library alone (its own member definitions,
// the explicit instantiations, and every legacy/validation use), and that is
// before cohomology_app's own TUs (which then carried -Wno-deprecated-
// declarations for the Matrix migration; slot-24 CL-M dropped that flag and the
// Matrix alias's attribute on this same precedent).  Burying
// real diagnostics under a hundred warnings is exactly defect #26; the
// include-list guard above enforces the same intent with zero noise.
// ===========================================================================

// ===========================================================================
// Stage S13 -- element-width generalization ([C, thm:backend] "Mod-ell
// backend"). `BlockFlatMatrixT<Elt>` is the block matrix of FlatMatrixT<Elt>
// entries over an element type Elt (uint32_t for the primary regime, uint64_t
// for the extended regime). Only the stored block values and the modulus
// become Elt; all block/scalar dimensions, indices, pivot coordinates and the
// deduplication layout stay uint32_t/size_t. The uint32_t instantiation is
// BYTE-IDENTICAL to the pre-S13 code.
//
//   using BlockFlatMatrix   = BlockFlatMatrixT<uint32_t>;  (default)
//   using BlockFlatMatrix64 = BlockFlatMatrixT<uint64_t>;  (extended regime)
// ===========================================================================

// Forward declaration for helper if needed outside, or define privately
// bool compare_flat_matrices_content_for_block(const FlatMatrix& fm1, const FlatMatrix& fm2);


template<class Elt>
class BlockFlatMatrixT {
private:
    std::vector<uint32_t> block_row_dims_; // Number of rows in each block-row
    std::vector<uint32_t> block_col_dims_; // Number of columns in each block-column
    Elt p_;                                // Prime modulus

    // Stores unique FlatMatrix objects
    std::vector<FlatMatrixT<Elt>> unique_blocks_;
    // Stores indices into unique_blocks_ to represent the block layout
    std::vector<std::vector<size_t>> block_layout_indices_;

    std::unordered_map<size_t, std::vector<size_t>> block_hash_index_;  // hash -> candidate indices in unique_blocks_

    uint32_t total_rows_;
    uint32_t total_cols_;
    uint32_t num_block_rows_;
    uint32_t num_block_cols_;

    // (matrix-engine Phase 1, defect #8) Prefix sums of block_row_dims_ /
    // block_col_dims_: block_row_offsets_[i] is the global scalar row at which
    // block-row i starts, and the vectors carry a trailing total (size
    // num_block_rows_ + 1 / num_block_cols_ + 1).  They turn
    // _get_block_start_{row,col}_idx into an O(1) array read and get_entry into
    // one binary search per axis instead of the two linear scans over the block
    // structure that dominated factor_basis_block.  They are a pure function of
    // the dimension vectors, rebuilt by _compute_total_dims() and carried by the
    // copy/move/swap paths alongside the dimensions themselves.
    std::vector<uint32_t> block_row_offsets_;
    std::vector<uint32_t> block_col_offsets_;

    // Cached RREF related data
    mutable std::unique_ptr<BlockFlatMatrixT> rref_cache_;

#ifdef ENGINE_COUNT_OBJECTS
    // (matrix-engine Phase 3) Construction census; see ObjectCounters.h. Present
    // ONLY under -DENGINE_COUNT_OBJECTS, so layout and sizeof are untouched by
    // default. Deliberately NOT named in any mem-init list: it is then default-
    // initialised by every constructor, copy and move included.
    engine::counters::BlockFlatMatrixCtorTick object_counter_tick_;
#endif


    // Structure for fine-grained scalar pivot information within rref_cache_
    struct ScalarPivotInfoInternal {
        uint32_t block_row_in_rref_cache; // The block-row index in the *rref_cache_*
        uint32_t block_col_in_rref_cache; // The block-col index in the *rref_cache_*
        uint32_t local_row_in_block;      // Row of scalar pivot *within* that block in rref_cache_
        uint32_t local_col_in_block;      // Col of scalar pivot *within* that block in rref_cache_

        // Global coordinates within rref_cache_
        uint32_t global_r;
        uint32_t global_c;
    };
    mutable std::unique_ptr<std::vector<ScalarPivotInfoInternal>> fine_pivot_info_cache_;

    // (matrix-engine Phase 3) The block-pivot coordinate list and the two
    // pivot-origin structures used to be cached members purely so the
    // validation-only accessors could hand out pointers into them (audit
    // defect #16 -- those pointers dangled on any mutation).  Both accessors
    // are gone; _block_gaussian_elimination still fills them, as locals of
    // compute_block_row_echelon_form().

    void _clear_cached_rref() const;
    void _compute_total_dims();
    void _validate_block_layout() const;

    static size_t _compute_block_hash(const FlatMatrixT<Elt>& block);
    size_t _add_or_get_block_index(const FlatMatrixT<Elt>& block);

    // Internal RREF logic, signature updated
    void _block_gaussian_elimination(
        BlockFlatMatrixT& temp_bfm_output, // Matrix being modified to become RREF
        std::vector<std::pair<uint32_t, uint32_t>>& block_pivots_coords_out, // Coords in temp_bfm_output
        std::vector<ScalarPivotInfoInternal>& fine_pivots_out, // Global scalar pivots in temp_bfm_output
        std::vector<std::pair<uint32_t, uint32_t>>& block_pivot_origins_out, // Original (br, bc) of blocks that became pivots
        std::map<std::pair<uint32_t, uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>>& original_block_fine_pivots_out // Fine pivots local to original P_block
    ) const;


    // Helper to get the starting global row index of a given block-row
    uint32_t _get_block_start_row_idx(uint32_t block_row_layout_idx) const;

    // Helper to get the starting global col index of a given block-column
    uint32_t _get_block_start_col_idx(uint32_t block_col_layout_idx) const;

    // Helper to get block layout indices and in-block coordinates from global scalar coordinates
    struct GlobalToBlockCoordsResult {
        uint32_t block_row_layout;
        uint32_t block_col_layout;
        uint32_t row_in_block;
        uint32_t col_in_block;
        bool success; // true if global coords are valid
    };
    GlobalToBlockCoordsResult _get_block_coords_from_global_scalar(uint32_t global_r, uint32_t global_c) const;

    FlatMatrixT<Elt> _get_original_scalar_column_as_flatmatrix(uint32_t global_col_idx) const;

    // (Phase 1) Copies scalar row `global_r` of *this into dst[0 .. total_cols_),
    // block by block, WITHOUT going through the per-entry accessor.  `dst` must
    // point at total_cols_ writable, pre-zeroed elements.
    void _copy_scalar_row_to_buffer(uint32_t global_r, Elt* dst) const;

    // Definitive scalar pivots of the RREF cache, (global_r, global_c), sorted
    // by column then row.  (The GlobalScalarPivotInfo struct that used to sit
    // here, and the two commented-out helper declarations that referenced it,
    // were dead in every build and are gone -- matrix-engine Phase 3.)
    std::vector<std::pair<uint32_t, uint32_t>> _get_definitive_global_scalar_pivots_from_rref_cache() const;

public:

    // Constructors
    BlockFlatMatrixT(Elt prime);
    BlockFlatMatrixT(const std::vector<uint32_t>& block_row_dims,
        const std::vector<uint32_t>& block_col_dims,
        Elt prime);
    BlockFlatMatrixT(const std::vector<uint32_t>& block_row_dims,
        const std::vector<uint32_t>& block_col_dims,
        const std::vector<std::vector<FlatMatrixT<Elt>>>& block_data,
        Elt prime);

    // Destructor
    ~BlockFlatMatrixT();

    // Copy/Move semantics
    BlockFlatMatrixT(const BlockFlatMatrixT& other);
    BlockFlatMatrixT& operator=(const BlockFlatMatrixT& other);
    BlockFlatMatrixT(BlockFlatMatrixT&& other) noexcept;
    BlockFlatMatrixT& operator=(BlockFlatMatrixT&& other) noexcept;

    // Accessors
    uint32_t get_num_block_rows() const { return num_block_rows_; }
    uint32_t get_num_block_cols() const { return num_block_cols_; }
    uint32_t get_total_rows() const { return total_rows_; }
    uint32_t get_total_cols() const { return total_cols_; }
    Elt get_p() const { return p_; }
    uint32_t get_block_row_dim(uint32_t block_r) const;
    uint32_t get_block_col_dim(uint32_t block_c) const;
    const FlatMatrixT<Elt>& get_block_entry(uint32_t block_r, uint32_t block_c) const;
    void set_block_entry(uint32_t block_r, uint32_t block_c, const FlatMatrixT<Elt>& block);
    Elt get_entry(uint32_t r, uint32_t c) const;

    // Conversion and Output
    FlatMatrixT<Elt> to_flat_matrix() const;
    std::string to_string(bool show_block_details = false) const;
    std::string to_string_offset(bool show_block_details, const uint32_t offset) const;

    // Operations
    // (matrix-engine Phase 1, defect #19) const: it only reads blocks and
    // touches their mutable scalar flags / the RREF cache.
    bool is_zero() const;
    // The only surviving arithmetic operator: the legacy arm of
    // compute_cohomology forms D1_block * w_block (EquivariantCohomologyOnK.cpp).
    BlockFlatMatrixT operator*(const BlockFlatMatrixT& other) const;

    // Linear Algebra (Methods to be revised in Phase 2 & 3)
    // This is the main orchestrator for RREF computation.
    void compute_block_row_echelon_form() const;

    // (2c) get_true_row_echelon_form_as_flat_matrix()
    // Returns: FlatMatrix (true scalar RREF, rows sorted) and globally sorted list of all scalar pivots.
    std::pair<FlatMatrixT<Elt>, std::vector<std::pair<uint32_t, uint32_t>>> get_true_row_echelon_form_as_flat_matrix() const;


    // Renamed for clarity, to use new RREF methods.
    // (defect #19) const -- see is_zero() above; this removes the whole-matrix
    // copies flat_inverse()/block_inverse() used to make just to call them.
    uint32_t flat_rank() const;
    bool is_invertible() const;
    FlatMatrixT<Elt> flat_inverse() const;  // scalar RREF on the augmented FlatMatrix

    std::vector<BlockFlatMatrixT> flat_kernel() const; // Kernel vectors as BFMs, from true scalar RREF
    std::vector<BlockFlatMatrixT> flat_image() const;  // Image vectors as BFMs, from true scalar RREF
};

using BlockFlatMatrix = BlockFlatMatrixT<uint32_t>;
using BlockFlatMatrix64 = BlockFlatMatrixT<uint64_t>;

// ---------------------------------------------------------------------------
// factor_basis_block  (moved here from Matrix.h / Matrix.cpp -- defect #22:
// this is block-layer code and had no business living in the deprecated
// `Matrix` translation unit).
//
// Computes a basis of  span(generators) / span(relations)  together with the
// projector onto it.  `block_row_dims` is the block structure of the ambient
// space; `relations` and `generators` are ambient_dim x 1 column vectors.
// Returns { projector (dim_H x ambient_dim), indices of the generators that
// survive }, and appends the surviving generators to `factor_basis_out`.
// ---------------------------------------------------------------------------
template<class Elt>
std::pair<FlatMatrixT<Elt>, std::vector<uint32_t>> factor_basis_block(
    const std::vector<uint32_t>& block_row_dims,
    const std::vector<FlatMatrixT<Elt>>& relations,
    const std::vector<FlatMatrixT<Elt>>& generators,
    std::vector<FlatMatrixT<Elt>>* factor_basis_out,
    Elt prime
);

#endif // BLOCK_FLAT_MATRIX_H
