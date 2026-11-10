// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_BINARY_TRITMATRIX_H
#define ENGINE_BINARY_TRITMATRIX_H

// =====================================================================
// TritMatrix.h -- native F_3 linear algebra on bit-sliced rows
// (hecke-extensions slot 23, package TER-1; L23_master_plan.md Sec. 3 (C), rung T0)
// =====================================================================
//
// A TritMatrix stores an m x n matrix over F_3 in two ONE-HOT bitplanes per row: bit j of plane
// `one` is [x_j == 1], bit j of plane `two` is [x_j == 2].  Each plane of a row is W = ceil(n / 512)
// * 8 uint64_t words (the BitMatrix layout of BIN-1), stored one after the other: row i is the
// 2W words [one | two].  Padding bits (columns >= n) are always zero in both planes, and no column
// has both bits set -- every operation below preserves both invariants.
//
// Arithmetic on 64 entries at once (x = (x1, x2), y = (y1, y2) one-hot):
//     x + y :  t = (x1 | y2) ^ (x2 | y1);  z1 = (x2 | y2) ^ t;  z2 = (x1 | y1) ^ t     (7 operations)
//     -x    :  (x2, x1)                                                                (swap the planes)
// (checked exhaustively on the 9 pairs by TernaryF3Validation).
//
// Elimination is Gauss--Jordan one pivot at a time (pivot = the topmost non-zero entry of the
// leftmost column with one), every row operation on whole bit-sliced words from the pivot's word
// on.  RREF, its pivot columns (the column rank profile) and the D5b kernel basis are unique
// functions of the input and the arithmetic is exact, so a correct implementation reproduces
// engine::elim::rref / kernel_basis at p = 3 bit for bit (ENGINE_API.md Sec. 8.3; the gate is
// TernaryF3Validation).
//
// ROW PATHS: the row bodies are portable, AVX2 and AVX-512 (target attributes, runtime
// dispatch).  They follow BIN-1's binary::set_row_path / active_row_path (one switch for both
// rungs); no path can change a value.
//
// The byte kernels f3_add_u8 / f3_sub_u8 / f3_neg_u8 serve callers that keep F_3 entries as
// uint8_t in {0, 1, 2} (the degree-1 amalgam reader's structural rho_n at ell = 3).  AVX-512
// needs avx512bw for them; without it the AVX2 body runs.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../ModArithmetic/DenseView.h"

namespace engine {
namespace binary {

class TritMatrix {
public:
    TritMatrix() = default;
    TritMatrix(uint32_t rows, uint32_t cols);          // zero matrix

    static TritMatrix identity(uint32_t n);

    uint32_t rows() const { return rows_; }
    uint32_t cols() const { return cols_; }
    uint32_t words_per_plane() const { return W_; }     // multiple of 8

    // Row i: plane `one` at row(i), plane `two` at row(i) + words_per_plane().
    uint64_t*       row(uint32_t i)       { return d_.data() + static_cast<size_t>(i) * 2 * W_; }
    const uint64_t* row(uint32_t i) const { return d_.data() + static_cast<size_t>(i) * 2 * W_; }

    uint32_t get(uint32_t i, uint32_t j) const {
        const uint64_t* r = row(i);
        const uint32_t w = j >> 6, s = j & 63;
        return static_cast<uint32_t>((r[w] >> s) & 1u) | (static_cast<uint32_t>((r[W_ + w] >> s) & 1u) << 1);
    }
    void set(uint32_t i, uint32_t j, uint32_t v);      // v in {0, 1, 2} (ENGINE_CHECK)
    void swap_rows(uint32_t a, uint32_t b);
    void negate_row(uint32_t i, uint32_t w0 = 0);      // swap the planes on words [w0, W)

    bool operator==(const TritMatrix& o) const { return rows_ == o.rows_ && cols_ == o.cols_ && d_ == o.d_; }
    bool operator!=(const TritMatrix& o) const { return !(*this == o); }

    // FNV-1a-64 over (rows, cols, every entry in {0, 1, 2} row-major) -- ModArithmetic/Fnv.h's constants.
    uint64_t fnv() const;

    // Bridges.  from_dense reads a CANONICAL F_3 view (ENGINE_CHECKs every entry is < 3); to_dense
    // writes 0 / 1 / 2 into every entry of `dst` (same shape).  Any layout.  uint8_t is the reader's
    // element (row-major, `ld` elements per row).
    template<class Elt> static TritMatrix from_dense(DenseView<const Elt> src);
    template<class Elt> void to_dense(DenseView<Elt> dst) const;
    static TritMatrix from_u8(const uint8_t* a, uint32_t rows, uint32_t cols, size_t ld);
    void to_u8(uint8_t* a, size_t ld) const;

private:
    uint32_t rows_ = 0, cols_ = 0, W_ = 0;
    std::vector<uint64_t> d_;
};

// Result of an elimination: rank and the pivot columns, ascending (= the column rank profile).
struct TritRref {
    uint32_t rank = 0;
    std::vector<uint32_t> pivots;
};

// In-place reduced row echelon form (D5b conventions: leading 1 in each pivot column, zero above
// and below, zero rows at the bottom).
TritRref rref(TritMatrix& A);

// In-place row echelon form (rows below each pivot cleared, rows above left alone; pivot rows NOT
// normalised): the pivot columns -- hence the rank -- are rref()'s, at about half the work.
TritRref echelon(TritMatrix& A);

// (slot-24 PAR-F2) echelon(A) bit for bit -- the same final matrix A, rank and pivots -- by the LAZY blocked Gauss: the
// rows below receive the pivots of a block (8..64 pivots) in one pass per row, each row exactly the serial operations in
// pivot order, over row tiles on the descent Smith pool (SmithPar.h) when smithpar::threads() > 1; the pivot search reads a
// transposed 512-column window of the reduced rows (two plane bitsets per column, updated by every pivot): m / 64 words.
TritRref echelon_lazy(TritMatrix& A);

uint32_t rank(const TritMatrix& A);   // on a copy

// The D5b kernel basis from an RREF R (k >= rank rows) and its pivots: one ROW per free column
// ascending, 1 at the free column and -R[i][j_free] at pivot i; shape (n - rank) x n.
// Bit-identical to engine::elim::kernel_basis at p = 3.
TritMatrix kernel_basis(const TritMatrix& R, const std::vector<uint32_t>& pivots);

// The kernel basis of A directly (RREF on a copy, then kernel_basis).
TritMatrix kernel(const TritMatrix& A);

// C = A * B over F_3: per non-zero A[i][t], one bit-sliced row addition (or subtraction) of B's row t.
TritMatrix multiply(const TritMatrix& A, const TritMatrix& B);

// References (validation use): entry-by-entry Gauss--Jordan / product through get / set only.
TritRref   rref_naive(TritMatrix& A);
TritMatrix multiply_naive(const TritMatrix& A, const TritMatrix& B);

// ---------------------------------------------------------------------------
// Byte kernels on canonical uint8_t F_3 entries (dst, src in {0, 1, 2}; outputs canonical)
// ---------------------------------------------------------------------------
void f3_add_u8(uint8_t* dst, const uint8_t* src, size_t n);   // dst += src
void f3_sub_u8(uint8_t* dst, const uint8_t* src, size_t n);   // dst -= src
void f3_neg_u8(uint8_t* dst, size_t n);                       // dst = -dst

// Which body the byte kernels / the plane kernels run on under the active row path (diagnostic).
const char* ternary_row_body_name();

} // namespace binary
} // namespace engine

#endif
