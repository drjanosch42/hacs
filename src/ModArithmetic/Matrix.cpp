// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * Matrix.cpp
 *
 * Implements the MatrixT<Elt> class template (Stage S13 element-width
 * generalization). See Matrix.h for the design note. The uint32_t
 * instantiation (alias Matrix) is byte-identical to the pre-S13 code: the
 * mod_*_w dispatchers forward verbatim to the uint32_t primitives.
 */

#include "pch.h"
#include "Matrix.h"
#include "FlatMatrix.h"

// for debugging
#include <stdio.h>
#include <iostream>

// Default constructor
template<class Elt>
MatrixT<Elt>::MatrixT()
    : row_echelon_form(nullptr),
    pivot_columns(nullptr),
    row_echelon_form_source(nullptr),
    rows(0),
    cols(0),
    p(0),   // Default to 0 to indicate "not ready for ops".
    data() { // Creates an empty vector of vectors
    // Body can be empty if all initialization is done in the initializer list
}

// Constructor
template<class Elt>
MatrixT<Elt>::MatrixT(uint32_t num_rows, uint32_t num_cols, Elt prime)
    : row_echelon_form(nullptr), pivot_columns(nullptr), row_echelon_form_source(nullptr), rows(num_rows), cols(num_cols), p(prime), data(num_rows, std::vector<Elt>(num_cols, 0)) {}

// Constructor from existing data
template<class Elt>
MatrixT<Elt>::MatrixT(uint32_t num_rows, uint32_t num_cols, Elt prime, const std::vector<std::vector<Elt>>& init_data)
    : row_echelon_form(nullptr), pivot_columns(nullptr), row_echelon_form_source(nullptr), rows(num_rows), cols(num_cols), p(prime), data(num_rows, std::vector<Elt>(num_cols))
{
    if (init_data.size() != num_rows || (num_rows > 0 && init_data[0].size() != num_cols)) {
        throw std::runtime_error("Invalid initial data dimensions.");
    }
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < num_cols; ++j) {
            data[i][j] = ModArithmetic::mod_add_w(init_data[i][j], Elt(0), p); // ensure elements are in [0, p-1]
        }
    }
}

// Creates an identity matrix
template<class Elt>
MatrixT<Elt>::MatrixT(uint32_t num_rows, Elt prime)
    : row_echelon_form(nullptr), pivot_columns(nullptr), row_echelon_form_source(nullptr), rows(num_rows), cols(num_rows), p(prime), data(num_rows, std::vector<Elt>(num_rows))
{
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < num_rows; ++j) {
            data[i][j] = (i == j) ? 1 : 0;
        }
    }
}

// Copy constructor implementation
template<class Elt>
MatrixT<Elt>::MatrixT(const MatrixT& other)
    : row_echelon_form(nullptr), // unique_ptr is initialized to null
    pivot_columns(nullptr),
    row_echelon_form_source(nullptr),
    rows(other.rows),
    cols(other.cols),
    p(other.p),
    data(other.data)
{}

// Copy assignment operator implementation
template<class Elt>
MatrixT<Elt>& MatrixT<Elt>::operator=(const MatrixT& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        p = other.p;
        data = other.data;
        // Reset the unique_ptrs, they will deallocate old memory automatically.
        row_echelon_form.reset();
        pivot_columns.reset();
        row_echelon_form_source.reset();
    }
    return *this;
}

// Swap function
template<class Elt>
void MatrixT<Elt>::swap(MatrixT& other) {
    std::swap(rows, other.rows);
    std::swap(cols, other.cols);
    std::swap(p, other.p);
    std::swap(data, other.data);
    std::swap(row_echelon_form, other.row_echelon_form);
    std::swap(pivot_columns, other.pivot_columns);
    std::swap(row_echelon_form_source, other.row_echelon_form_source);
}

// For standard output
template<class Elt>
std::string MatrixT<Elt>::to_string() const {
    return to_string_offset(0);
}


// For standard output
template<class Elt>
std::string MatrixT<Elt>::to_string_offset(const uint32_t offset) const {
    std::ostringstream oss;

    if (rows > 0 && cols > 0)
    {
        for (uint32_t i = 0; i < rows; ++i) {
            oss << offset_string(offset);
            oss << "[";
            for (uint32_t j = 0; j < cols; ++j) {
                oss << std::setw(2) << data[i][j]; // Adjust width (4) as needed
                if (j < cols - 1) {
                    oss << " ";
                }
            }
            oss << " ]\n";
        }
    } else {
        oss << offset_string(offset);
        oss << rows << " x " << cols << " matrix\n";
    }

    return oss.str();
}

// For standard output
template<class Elt>
std::string MatrixT<Elt>::to_string_transposed() const {
    return to_string_transposed_offset(0);
}

// For standard output
template<class Elt>
std::string MatrixT<Elt>::to_string_transposed_offset(const uint32_t offset) const {
    std::ostringstream oss;

    if (rows > 0 && cols > 0)
    {
        for (uint32_t i = 0; i < cols; ++i) {
            oss << offset_string(offset);
            oss << "[";
            for (uint32_t j = 0; j < rows; ++j) {
                oss << std::setw(2) << data[j][i]; // Adjust width (4) as needed
                if (j < rows - 1) {
                    oss << " ";
                }
            }
            oss << ((i == 0) ? " ]^t\n" : " ]\n");
        }
    } else {
        oss << offset_string(offset);
        oss << rows << " x " << cols << " matrix\n";
    }
    return oss.str();
}

// Transpose, returns a new matrix
template<class Elt>
MatrixT<Elt> MatrixT<Elt>::transpose() {
    MatrixT result(cols, rows, p); // Create a new matrix with swapped dimensions

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            result.data[j][i] = data[i][j]; // Copy elements, swapping rows and columns
        }
    }

    return result;
}

// Comparison with 0:

// Addition, returns a new matrix
template<class Elt>
MatrixT<Elt> MatrixT<Elt>::operator+(const MatrixT& other) const {
    if (rows != other.rows || cols != other.cols || p != other.p) {
        throw std::runtime_error("Matrix dimensions or prime moduli do not match for addition.");
    }

    MatrixT result(rows, cols, p);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            result.data[i][j] = ModArithmetic::mod_add_w(data[i][j], other.data[i][j], p);
        }
    }
    return result;
}


// Subtraction, returns a new matrix
template<class Elt>
MatrixT<Elt> MatrixT<Elt>::operator-(const MatrixT& other) const  {
    if (rows != other.rows || cols != other.cols || p != other.p) {
        throw std::runtime_error("Matrix dimensions or prime moduli do not match for subtraction.");
    }

    MatrixT result(rows, cols, p);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            result.data[i][j] = ModArithmetic::mod_sub_w(data[i][j], other.data[i][j], p);
        }
    }
    return result;
}

// Multiplication, returns a new matrix
template<class Elt>
MatrixT<Elt> MatrixT<Elt>::operator*(const MatrixT& other) const  {
    if (cols != other.rows || p != other.p) {
        throw std::runtime_error("Matrix dimensions or prime moduli do not match for multiplication.");
    }

    MatrixT result(rows, other.cols, p);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < other.cols; ++j) {
            for (uint32_t k = 0; k < cols; ++k) {
                result.data[i][j] = ModArithmetic::mod_add_w(
                    result.data[i][j],
                    ModArithmetic::mod_mul_w(data[i][k], other.data[k][j], p),
                    p
                );
            }
        }
    }
    return result;
}

template<class Elt>
MatrixT<Elt> MatrixT<Elt>::operator*(const Elt scalar) const  {

    MatrixT result(rows, cols, p);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            result.data[i][j] = ModArithmetic::mod_mul_w(data[i][j], scalar, p);
        }
    }
    return result;
}

// Comparison with 0
template<class Elt>
bool MatrixT<Elt>::is_zero(void) const {
    bool result = true;
    for(uint32_t i = 0; i < rows && result; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            if (data[i][j] != 0)
            {
                result = false;
                break;
            }
        }
    }
    return result;
}

// Equality Comparison
template<class Elt>
bool MatrixT<Elt>::operator==(const MatrixT& other) const {
    return (rows == other.rows && cols == other.cols && p == other.p && data == other.data);
}

template<class Elt>
bool MatrixT<Elt>::operator!=(const MatrixT& other) const {
    return !(*this == other);
}

template<class Elt>
uint32_t MatrixT<Elt>::nonzero_entries(void) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < rows; i++)
    {
        for (uint32_t j = 0; j < cols; j++)
            count += (data[i][j] != 0);
    }
    return count;
}

// Defect #9: the RREF cache used to be computed once and then kept forever,
// even though `data` is a public member that callers mutate in place; every
// `rank()` / `kernel()` / `image()` / `inverse()` after such a mutation silently
// answered for the OLD matrix.  The cache now carries the source it was built
// from and is dropped as soon as `data` differs from it.
template<class Elt>
void MatrixT<Elt>::discard_row_echelon_form_if_stale() {
    if (row_echelon_form == nullptr) return;
    if (row_echelon_form_source == nullptr || *row_echelon_form_source != data) {
        row_echelon_form.reset();
        pivot_columns.reset();
        row_echelon_form_source.reset();
    }
}

template<class Elt>
void MatrixT<Elt>::compute_row_echelon_form() {
    discard_row_echelon_form_if_stale();
    if (row_echelon_form == nullptr)
    {
        row_echelon_form_source = std::make_unique<std::vector<std::vector<Elt>>>(data);
        row_echelon_form = std::make_unique<std::vector<std::vector<Elt>>>(data);
        pivot_columns = std::make_unique<std::vector<uint32_t>>();
        gaussian_elimination(*pivot_columns); // Compute row echelon form
    }
}

template<class Elt>
void MatrixT<Elt>::gaussian_elimination(std::vector<uint32_t>& pivot_cols) {
    pivot_cols.clear();
    uint32_t lead = 0;

    for (uint32_t r = 0; r < rows; ++r) {
        if (lead >= cols) break;
        uint32_t i = r;
        while ((*row_echelon_form)[i][lead] == 0) {
            ++i;
            if (i == rows) {
                ++lead;
                if (lead == cols) break;
                i = r;
            }
        }
        if (lead < cols) {
            std::swap((*row_echelon_form)[i], (*row_echelon_form)[r]);

            Elt pivot_inv = ModArithmetic::mod_inv_w((*row_echelon_form)[r][lead], p);
            // for (uint32_t j = 0; j < cols; ++j) {
            for (uint32_t j = lead; j < cols; ++j) {
                (*row_echelon_form)[r][j] = ModArithmetic::mod_mul_w((*row_echelon_form)[r][j], pivot_inv, p);
            }

            for (uint32_t k = 0; k < rows; ++k) {
                if (k != r) {
                    Elt factor = (*row_echelon_form)[k][lead];
                    // for (uint32_t j = 0; j < cols; ++j) {
                    if(factor)
                        for (uint32_t j = lead; j < cols; ++j) {
                            (*row_echelon_form)[k][j] = ModArithmetic::mod_sub_w((*row_echelon_form)[k][j], ModArithmetic::mod_mul_w(factor, (*row_echelon_form)[r][j], p), p);
                        }
                }
            }
            pivot_cols.push_back(lead);
            ++lead;
        }
    }
}

template<class Elt>
std::vector<std::vector<Elt>>& MatrixT<Elt>::get_row_echelon_form() {
    compute_row_echelon_form();  // Compute if not already computed
    return *row_echelon_form;    // Return the computed row echelon form
}

// returns a new matrix
template<class Elt>
MatrixT<Elt> MatrixT<Elt>::get_row_echelon_form_as_matrix(void) {

    return MatrixT(rows, cols, p, get_row_echelon_form());    // Return the computed row echelon form
}

template<class Elt>
std::vector<uint32_t>& MatrixT<Elt>::get_pivot_columns() {
    compute_row_echelon_form(); // Compute if not already computed
    return *pivot_columns;       // Return the computed pivot columns
}

template<class Elt>
uint32_t MatrixT<Elt>::rank() {
    return get_pivot_columns().size(); // The rank is equal to the number of pivot columns
}

template<class Elt>
bool MatrixT<Elt>::is_invertible() {
    // Must be square
    if (rows != cols) {
        return false;
    }
    // For a square matrix, invertible iff rank equals dimension
    return rank() == rows;
}

// Matrix Inversion, returns new matrix
template<class Elt>
MatrixT<Elt> MatrixT<Elt>::inverse() {
    if (rows != cols) {
        throw std::runtime_error("Matrix is not square, inversion not possible.");
    }

    // Augment with identity matrix
    MatrixT augmented(rows, 2 * cols, p);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            augmented.data[i][j] = data[i][j];
        }
        augmented.data[i][i + cols] = 1; // Identity part
    }

    // Perform Gaussian elimination
    // Check for invertibility
    // (bind the caches once: each get_* now revalidates against `data`, so
    //  re-querying inside the loops would be quadratically expensive)
    const std::vector<uint32_t>& aug_pivots = augmented.get_pivot_columns();
    for (uint32_t i = 0; i < rows; ++i)
    {
        if (aug_pivots.size() <= i || aug_pivots[i] != i)
        {
            throw std::runtime_error("Matrix is not invertible.");
        }
    }

    // Extract inverse from the right half of the augmented matrix
    const std::vector<std::vector<Elt>>& aug_rref = augmented.get_row_echelon_form();
    MatrixT inv(rows, cols, p);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            inv.data[i][j] = aug_rref[i][j + cols];
        }
    }

    return inv;
}

// Matrix Kernel
template<class Elt>
std::vector<MatrixT<Elt>> MatrixT<Elt>::kernel() {
    std::vector<MatrixT> kernel_basis;
    std::vector<bool> is_pivot_column(cols, false);
    const std::vector<uint32_t> pivot_columns = get_pivot_columns();
    for (uint32_t col : pivot_columns)
    {
        is_pivot_column[col] = true;
    }

    const std::vector<std::vector<Elt>>& rref = get_row_echelon_form(); // bind once
    for (uint32_t j = 0; j < cols; ++j) {
        if (!is_pivot_column[j]) {
            MatrixT basis_vector(cols, 1, p);
            basis_vector.data[j][0] = 1;
            uint32_t pivot_index = 0;
            for (uint32_t i = 0; i < cols; ++i)
            {
                if (is_pivot_column[i])
                {
                    basis_vector.data[i][0] = ModArithmetic::mod_sub_w(Elt(0), rref[pivot_index][j], p);
                    pivot_index++;
                }
            }
            kernel_basis.push_back(basis_vector);
        }
    }

    return kernel_basis;
}

// Matrix Image
template<class Elt>
std::vector<MatrixT<Elt>> MatrixT<Elt>::image() {
    std::vector<MatrixT> image_basis;
    for (uint32_t col : get_pivot_columns())
    {
        MatrixT basis_vector(rows, 1, p);
        for (uint32_t i = 0; i < rows; i++)
        {
            basis_vector.data[i][0] = data[i][col];
        }
        image_basis.push_back(basis_vector);
    }

    return image_basis;
}

// Defect #32 (matrix-engine Phase 0): IdentityMatrix, flatten_to_column_vector
// and unflatten_from_column_vector were deleted here -- deprecated, with zero
// callers anywhere in the repository (pipeline, Hecke and validations alike).
// IdentityMatrix was a one-line forward to MatrixT(n, p).
// (slot-24 CL-M, CODEAUDIT C-02) change_of_basis (x2), factor_basis,
// basis_to_string and ComputeSymmetricPowerMatrix(MatrixT) were deleted the
// same way: deprecated, no caller outside this file and MatrixValidation.

// ===========================================================================
// Explicit instantiations (Stage S13): uint32_t (primary regime, alias Matrix)
// and uint64_t (extended regime, alias Matrix64).
// ===========================================================================
template class MatrixT<uint32_t>;
template class MatrixT<uint64_t>;

