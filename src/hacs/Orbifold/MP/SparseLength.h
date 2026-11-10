// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_SPARSE_LENGTH_H
#define ORBIFOLD_MP_SPARSE_LENGTH_H

// ===========================================================================
// (hecke-extensions slot 22, MP-1b) The composition length of the row span of
// a sparse matrix over Z/ell^e (e >= 1; e = 1 is the rank over F_ell), for the
// MP nerve's coboundaries (MP_impl_plan.md v0.2 Sec. 2.3: "fallback a
// column-Markowitz mod-p elimination Orbifold/MP/SparseRank").
//
// ALGORITHM (unit-pivot elimination + a dense local residual).  Rows are
// reduced one at a time against the pivot rows found so far (a scatter
// accumulator and a column heap); a pivot row has a UNIT at its lead column
// and nothing before it.  A reduced row whose least column carries a unit
// becomes a pivot (normalised to 1 there); a reduced row whose least column
// carries a non-unit (only over Z/ell^e, e >= 2) is kept as a RESIDUAL row.
// At the end every residual row is re-reduced against all pivots, so the
// residual rows vanish on every pivot column.  Then
//     length(rowspan A) = e * #pivots + length(rowspan R)
// (the projection onto the pivot coordinates maps rowspan A onto (Z/ell^e)^P
// with kernel rowspan R), and length(rowspan R) is computed densely by
// engine::elim::length (Howell form, ModArithmetic/DenseElim) on the columns
// R actually meets.  Over F_ell R is empty.  Exact; no randomisation.
// ===========================================================================

#include <cstdint>
#include <vector>

namespace orbifold_mp {

// CSR, columns strictly increasing within a row, values in [1, m).
struct SparseRows {
    uint32_t ncols = 0;
    std::vector<uint64_t> start{0};   // rows + 1 offsets
    std::vector<uint32_t> col;
    std::vector<uint8_t> val;
    uint64_t rows() const { return start.size() - 1; }
    uint64_t nnz() const { return col.size(); }
};

struct LengthStats {
    uint64_t length = 0;          // composition length of the row span
    uint64_t rows = 0, cols = 0, input_nnz = 0;
    uint64_t pivots = 0;          // unit pivots
    uint64_t pivot_nnz = 0;       // stored entries of the pivot rows (the fill-in)
    uint64_t max_pivot_row = 0;   // longest pivot row
    uint64_t residual_rows = 0, residual_cols = 0;
    uint64_t residual_length = 0;
    uint64_t pivot_applications = 0;   // row operations performed
    uint64_t applied_entries = 0;      // entries touched by them (the work)
    double wall_s = 0.0;
};

// The order in which rows are reduced.  The length does not depend on it; the work does, by two orders of
// magnitude on the MP coboundaries: their rows come in string-extension order, and reducing the LAST-built
// strings first (Reversed, the default) keeps the fill-in near the input size (MP-1b's R3 measurement,
// MP_predictions.md: d^2 at n = 6 on M7, 372 001 x 17 787, fill 88 754 entries, 0.31 s; Natural 12.8 s at n = 2).
enum class RowOrder : uint8_t { Natural = 0, Reversed = 1 };

// ell prime, e >= 1, ell^e <= 255.  ENGINE_CHECK: the dense residual stays within dense_budget entries.
LengthStats RowSpanLength(const SparseRows& A, uint32_t ell, uint32_t e, uint64_t dense_budget = 200000000ull,
                          RowOrder order = RowOrder::Reversed);
// A^T (rows of A^T = columns of A, columns strictly increasing within a row).  (MP-2; replaces the dead
// ReverseColumns, F-MP1 F6.)
SparseRows Transpose(const SparseRows& A);

}  // namespace orbifold_mp

#endif
