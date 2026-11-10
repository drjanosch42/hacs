// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_BINARY_BITMATRIX_H
#define ENGINE_BINARY_BITMATRIX_H

// =====================================================================
// BitMatrix.h -- native F_2 linear algebra on bit-packed rows
// (hecke-extensions slot 22, package BIN-1; L23_master_plan.md Sec. 2, rung R0)
// =====================================================================
//
// A BitMatrix stores an m x n matrix over F_2 with one bit per entry: row i
// is W = ceil(n / 512) * 8 little-endian uint64_t words (bit j of the row is
// bit (j % 64) of word j / 64), so every row is a whole number of 512-bit
// lanes and the row operations need no tail.  Padding bits (columns >= n)
// are always zero -- every operation below preserves that invariant.
//
// Elimination is the "Method of the Four Russians" (M4RI, Bard): columns are
// walked left to right in strips of up to K = 8 pivots; the strip's pivot
// rows are reduced among themselves, a 256-entry table of all their XOR
// combinations is built (each entry = an earlier entry XOR one pivot row),
// and every other row is cleared in all K pivot columns by ONE table lookup
// and ONE row XOR.  The product C = A * B is M4RM: per 8-row slice of B one
// such table, per row of A one lookup (a byte of A's row) and one XOR.
//
// WHY THE RESULTS ARE THE ENGINE'S (ENGINE_API.md Sec. 8): RREF(A), its pivot
// columns (the column rank profile) and the D5b kernel basis are unique
// functions of A; over F_2 an exact XOR elimination is exact arithmetic, so a
// correct implementation reproduces engine::elim::rref / kernel_basis bit for
// bit at p = 2.  BinaryF2Validation proves it entrywise.
//
// ROW PATHS: the row XORs have an AVX-512 body, an AVX2 body (both compiled
// with target attributes, picked at runtime by __builtin_cpu_supports) and a
// portable body; a build on a CPU or compiler without them (aarch64, MSVC)
// gets the portable body only.  set_row_path() forces one (validation use);
// no path can change a bit.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../ModArithmetic/CpuFeatures.h"
#include "../ModArithmetic/DenseView.h"

namespace engine {
namespace binary {

class BitMatrix {
public:
    BitMatrix() = default;
    BitMatrix(uint32_t rows, uint32_t cols);          // zero matrix

    static BitMatrix identity(uint32_t n);

    uint32_t rows() const { return rows_; }
    uint32_t cols() const { return cols_; }
    uint32_t words_per_row() const { return W_; }     // multiple of 8

    uint64_t*       row(uint32_t i)       { return d_.data() + static_cast<size_t>(i) * W_; }
    const uint64_t* row(uint32_t i) const { return d_.data() + static_cast<size_t>(i) * W_; }

    bool get(uint32_t i, uint32_t j) const { return (row(i)[j >> 6] >> (j & 63)) & 1u; }
    void set(uint32_t i, uint32_t j, bool v) {
        const uint64_t m = uint64_t(1) << (j & 63);
        if (v) row(i)[j >> 6] |= m; else row(i)[j >> 6] &= ~m;
    }
    void flip(uint32_t i, uint32_t j) { row(i)[j >> 6] ^= uint64_t(1) << (j & 63); }
    void swap_rows(uint32_t a, uint32_t b);

    bool operator==(const BitMatrix& o) const { return rows_ == o.rows_ && cols_ == o.cols_ && d_ == o.d_; }
    bool operator!=(const BitMatrix& o) const { return !(*this == o); }

    // FNV-1a-64 over (rows, cols, every row's significant words) -- ModArithmetic/Fnv.h's constants.
    uint64_t fnv() const;
    uint32_t popcount() const;

    // Bridges to the engine's dense buffers.  from_dense reads entry & 1 of a CANONICAL F_2 view
    // (ENGINE_CHECKs every entry is 0 or 1); to_dense writes 0 / 1 into every entry of `dst`
    // (same shape).  Any layout; a row-contiguous uint32_t view takes the vector path.
    template<class Elt> static BitMatrix from_dense(DenseView<const Elt> src);
    template<class Elt> void to_dense(DenseView<Elt> dst) const;

private:
    uint32_t rows_ = 0, cols_ = 0, W_ = 0;
    std::vector<uint64_t> d_;
};

// Result of an elimination: rank and the pivot columns, ascending (= the column rank profile).
struct BitRref {
    uint32_t rank = 0;
    std::vector<uint32_t> pivots;
};

// In-place reduced row echelon form (D5b conventions: leading 1 in each pivot column, zero above
// and below, zero rows at the bottom).  M4RI, K = 8.
BitRref rref(BitMatrix& A);

// In-place row echelon form only (rows below each pivot cleared, rows above left alone): the
// pivot columns -- hence the rank -- are rref()'s, at about half the work.  A is left in an
// unspecified echelon form.
BitRref echelon(BitMatrix& A);

uint32_t rank(const BitMatrix& A);   // on a copy

// (slot-24 PAR-F2) echelon(A) bit for bit -- the same final matrix A, rank and pivots -- by the LAZY blocked M4RI: the
// serial strips (same pivot rows, same row swaps, same 256-entry subset tables) are kept, but the clears of up to eight
// consecutive strips are applied to each row below in ONE pass (each row receives exactly the serial table XORs, in strip
// order), over row tiles on the descent Smith pool (SmithPar.h) when smithpar::threads() > 1.  The pivot search reads a
// TRANSPOSED window of 512 columns of the reduced rows (one bitset over the rows per column, built at each flush and
// updated by every pivot), so a column without a pivot costs m / 64 words instead of a pass over every row (the descent's
// matrices: most columns, e.g. rank 31 489 of 88 200).  One pass over the matrix per up to 64 pivots instead of per 8.
// Single-threaded outside a ScopedEnable.  Extra memory: 8 tables (2048 W words) + the window (512 ceil(m / 64) words).
BitRref echelon_lazy(BitMatrix& A);

// (PAR-F2) dst[j / 64] bit j % 64 = src[j] & 1 for j < n (words [0, ceil(n / 64)) overwritten); the active row path's
// vector body (AVX-512 / AVX2) or the portable loop -- same bits.
void pack_low_bits(const uint32_t* src, uint32_t n, uint64_t* dst);
void pack_low_bits(const uint64_t* src, uint32_t n, uint64_t* dst);

// The D5b kernel basis from an RREF R (k >= rank rows) and its pivots: one ROW per free column
// ascending, 1 at the free column and R[i][j_free] (= -R[i][j_free] over F_2) at pivot i;
// shape (n - rank) x n.  Bit-identical to engine::elim::kernel_basis at p = 2.
BitMatrix kernel_basis(const BitMatrix& R, const std::vector<uint32_t>& pivots);

// The kernel basis of A directly (RREF on a copy, then kernel_basis).
BitMatrix kernel(const BitMatrix& A);

// C = A * B over F_2 by M4RM (8-row Gray tables of B).  ENGINE_CHECKs A.cols() == B.rows().
BitMatrix multiply_m4rm(const BitMatrix& A, const BitMatrix& B);

// Reference product (bit-by-bit row XORs, no table) -- the validation's cross-check.
BitMatrix multiply_naive(const BitMatrix& A, const BitMatrix& B);

// Reference elimination: Gauss--Jordan one pivot at a time with plain row XORs (no tables).
BitRref rref_naive(BitMatrix& A);

// ---------------------------------------------------------------------------
// Row paths (no path changes a value)
// ---------------------------------------------------------------------------
using RowPath = cpu::IsaPath;            // (S24 CL-5, C-06) the one enum (CpuFeatures.h)
void        set_row_path(RowPath p);      // Auto = the best the CPU supports at or below ENGINE_KERNEL (default)
RowPath     active_row_path();            // the path Auto resolves to, or the forced one
bool        row_path_available(RowPath p);
const char* row_path_name(RowPath p);

} // namespace binary
} // namespace engine

#endif
