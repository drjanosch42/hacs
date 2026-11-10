// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef MATRIX_H
#define MATRIX_H

#include "ModArithmetic.h"
#include "helper.h"
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <memory>   // For std::unique_ptr
#include <utility>  // For std::pair

// ===========================================================================
// Stage S13 -- element-width generalization ([C, thm:backend] "Mod-ell
// backend"). `MatrixT<Elt>` is the dense F_ell matrix over an element type Elt
// (uint32_t for the primary regime ell < 2^32, uint64_t for ell < 2^64). The
// mod-reduction inside picks mod_mul vs mod_mul64 by element width through the
// ModArithmetic::mod_*_w overloads, so the uint32_t instantiation is BYTE-
// IDENTICAL to the pre-S13 code. Dimensions/indices stay uint32_t; only the
// stored values and the modulus become Elt.
//
//   using Matrix   = MatrixT<uint32_t>;  (default, source-compatible)
//   using Matrix64 = MatrixT<uint64_t>;  (extended regime)
// ===========================================================================

template<class Elt> class FlatMatrixT; // Forward declaration
using FlatMatrix = FlatMatrixT<uint32_t>;

template<class Elt>
class MatrixT {
private:
    std::unique_ptr<std::vector<std::vector<Elt>>> row_echelon_form;
    std::unique_ptr<std::vector<uint32_t>> pivot_columns;
    // Defect #9 guard (matrix-engine Phase 0): `data` is public and is written
    // to directly on ~120 lines across the repo, so there is no mutator to hook
    // and no call-site-free way to be told the source changed.  Instead the
    // exact source that produced `row_echelon_form` / `pivot_columns` is kept
    // here, and every cache read first checks that `data` still equals it;
    // otherwise the cache is dropped and recomputed.  Cost is one O(rows*cols)
    // comparison per *cache read*, which is why the members of this class read
    // the cache once and bind a reference rather than re-querying in a loop.
    std::unique_ptr<std::vector<std::vector<Elt>>> row_echelon_form_source;

    // Drops row_echelon_form / pivot_columns if `data` no longer matches the
    // snapshot they were computed from.  No-op when no cache is present.
    void discard_row_echelon_form_if_stale();

public:
    uint32_t rows;
    uint32_t cols;
    Elt p;
    std::vector<std::vector<Elt>> data;

    MatrixT();
    MatrixT(uint32_t num_rows, uint32_t num_cols, Elt prime);
    MatrixT(uint32_t num_rows, uint32_t num_cols, Elt prime, const std::vector<std::vector<Elt>>& init_data);
    MatrixT(uint32_t num_rows, Elt prime); // creates identity matrix

        // The destructor can now be defaulted or removed!
    ~MatrixT() = default;

    // A proper copy constructor is needed because unique_ptr is not copyable.
    // We want to copy the data but not the cached results.
    MatrixT(const MatrixT & other);

    // A proper copy assignment is needed.
    MatrixT& operator=(const MatrixT & other);

    // Move constructor and move assignment for efficiency with std::vector
    MatrixT(MatrixT && other) noexcept = default;
    MatrixT& operator=(MatrixT && other) noexcept = default;

    // Swap function
    void swap(MatrixT& other);

    std::string to_string(void) const;
    std::string to_string_offset(const uint32_t offset) const;
    std::string to_string_transposed(void) const;
    std::string to_string_transposed_offset(const uint32_t offset) const;
    MatrixT transpose();

    // Elementary matrix operations
    bool is_zero(void) const;
    bool operator==(const MatrixT& other) const;
    bool operator!=(const MatrixT& other) const;
    MatrixT operator+(const MatrixT& other) const;
    MatrixT operator-(const MatrixT& other) const;
    MatrixT operator*(const MatrixT& other) const;
    MatrixT operator*(const Elt scalar) const;

    // basic statistics
    uint32_t nonzero_entries();

    // Linear algebra stage 1: Matrix normal form
    void gaussian_elimination(std::vector<uint32_t>& pivot_columns);
    void compute_row_echelon_form();    // Function to compute and store the row echelon form
    std::vector<std::vector<Elt>>& get_row_echelon_form(); // returns the intrinsic array for efficient computation
    MatrixT get_row_echelon_form_as_matrix(); // returns a new matrix containing the row echelon form
    std::vector<uint32_t>& get_pivot_columns();

    // Linear algebra stage 2
    uint32_t rank();
    bool is_invertible();
    MatrixT inverse();
    std::vector<MatrixT> kernel();
    std::vector<MatrixT> image();
};

// ---------------------------------------------------------------------------
// The aliases carry NO [[deprecated]] attribute (slot-24 CL-M, CODEAUDIT C-02),
// following the precedent at BlockFlatMatrix.h: the attribute produced
// hundreds of -Wdeprecated-declarations warnings on spellings that are not
// going away, and kept cohomology_app on -Wno-deprecated-declarations, which
// hid every real deprecation.  New code should still prefer FlatMatrix /
// engine::blk.  The set of translation units allowed to include this header
// directly is enforced by docs/matrix-engine/tools/check_matrix_includes.sh
// (run by MatrixRetirementValidation).
//
// Sanctioned MatrixT user (CODEAUDIT A9): HeckeExt's `ExtMatrix`
// (HeckeExt/HeckeLocalAlgebra.h, `using ExtMatrix = MatrixT<uint32_t>`) is the
// small dense F_ell algebra of the extensions programme ([X] Sec. 2.3); it is
// deliberate, not a work-around.  Beside it the live MatrixT users are
// HeckeModuleDecomposition / SubspaceOps / FpPoly / HeckeOperatorFamily and
// the read-only A^(q) consumers GoldenHarness and BenchDriver.
// ---------------------------------------------------------------------------
using Matrix = MatrixT<uint32_t>;
// Defect #23: Matrix64 is the same class template as Matrix and shares every
// one of its defects.
using Matrix64 = MatrixT<uint64_t>;

// (slot-24 CL-M) The deprecated free functions change_of_basis (x2),
// factor_basis, basis_to_string and the Matrix overload of
// ComputeSymmetricPowerMatrix were deleted: they had no caller outside
// Matrix.cpp and MatrixValidation.  Use change_of_basis_fm_multi,
// factor_basis_block / engine::blk::factor_basis and
// ComputeSymmetricPowerMatrixFM.

// (matrix-engine Phase 1, defect #22) factor_basis_block moved to
// BlockFlatMatrix.h/.cpp -- it is block-layer code and only ever operated on
// BlockFlatMatrixT; it was declared here purely for historical reasons.

#endif // MATRIX_H
