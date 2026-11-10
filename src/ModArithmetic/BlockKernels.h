// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// BlockKernels.h -- pattern-driven kernels on BlockMat (Phase 3, plan D3/D4)
// =====================================================================
//
// Every kernel iterates the block-CSR pattern (never a dense grid), folds the
// (alpha, beta) kinds before touching data, and CSEs dense products on the
// data window: a product M * X of the same (pool window, X block) pair is
// computed once per call.  All arithmetic goes through engine::kernels, so
// the k-blocking overflow contract of ENGINE_API.md Sec. 5 holds and every
// result is exact -- the summation order across the pattern is fixed by the
// CSR order, but since every term is an exact residue the order cannot move a
// bit anyway.
//
// Preferred layouts (plan D3 layout policy, item 1):
//   gemm            A blocks row-major, B blocks column-major or row-major
//                   (the microkernel streams B's rows: a column-major B is
//                   fine through its transposed data view), C row-major.
//   gemm_dense      X (multi-vector) column-major, Y column-major.
//   to_flat/from_flat (BlockMat.h)  column-major multi-vectors.
// These are hints for the caller; every kernel is correct for any layout.

#include <cstdint>
#include <vector>

#include "BlockMat.h"
#include "DenseView.h"
#include "Kernels.h"

namespace engine {
namespace blk {

// C = A * B on the block level (C's grid: A's rows x B's cols; A.col_off must
// equal B.row_off).  Result blocks are interned in A's pool with layout
// `out_layout`; all-zero results are structural zeros, alpha*I results are
// Scalar refs.  Scalar-only products (alpha*I times beta*I) never touch data.
template<class Elt>
BlockMat<Elt> gemm(const BlockMat<Elt>& A, const BlockMat<Elt>& B,
                   Layout out_layout = Layout::RowMajor);

// Y op= A * X for a dense multi-vector X (A.cols() x k), Y (A.rows() x k).
// X's rows are partitioned by A's column offsets.  CSE on (data window, block
// column).  op: Set / Add / Sub.
template<class Elt>
void gemm_dense(const BlockMat<Elt>& A, DenseView<const Elt> X, DenseView<Elt> Y,
                kernels::GemmOp op = kernels::GemmOp::Set);

// C = A + s*B (s = 1 or p-1 for A - B) block-wise; same grid required.  Sums
// of refs sharing the data window fold; others are materialised and interned.
template<class Elt>
BlockMat<Elt> add(const BlockMat<Elt>& A, const BlockMat<Elt>& B, bool subtract = false,
                  Layout out_layout = Layout::RowMajor);

// alpha * A: touches no data (alpha, beta of every ref are scaled).
template<class Elt>
BlockMat<Elt> scale(const BlockMat<Elt>& A, Elt alpha);

// Equality AS MATHEMATICAL MATRICES: same scalar shape and entries, regardless
// of grids, kinds, pool ids, layouts.  Same-grid pairs are compared block by
// block (identical representations short-circuit); different grids are
// compared through dense row strips.
template<class Elt>
bool equal(const BlockMat<Elt>& A, const BlockMat<Elt>& B);
template<class Elt>
bool equal_dense(const BlockMat<Elt>& A, DenseView<const Elt> D);

// dst (nr x A.cols()) = scalar rows [r0, r0+nr) of A, materialised.
template<class Elt>
void materialise_rows(const BlockMat<Elt>& A, uint32_t r0, uint32_t nr, DenseView<Elt> dst);

// dst (A.rows() x nc) = scalar columns [c0, c0+nc) of A, materialised.
template<class Elt>
void materialise_cols(const BlockMat<Elt>& A, uint32_t c0, uint32_t nc, DenseView<Elt> dst);

struct GemmStats {
    uint64_t dense_products = 0;   // gemm calls issued
    uint64_t cse_hits       = 0;   // products served from the per-call cache
    uint64_t scalar_folds   = 0;   // block products resolved without data
    uint64_t axpys          = 0;
};
const GemmStats& gemm_stats();
void reset_gemm_stats();

} // namespace blk
} // namespace engine
