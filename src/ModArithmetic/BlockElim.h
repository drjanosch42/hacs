// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// BlockElim.h -- structured elimination on BlockMat (Phase 3, plan D5 / D5b)
// =====================================================================
//
// A panel driver over engine::elim (DenseElim) that walks the BLOCK COLUMNS
// of a BlockMat left to right and keeps one dense, FULLY REDUCED RREF prefix
// R (rank x n, row-major).  Every step is a GEMM or one DenseElim call on a
// dense panel; there is no per-cell work anywhere and no dense copy of the
// input.
//
// ALGORITHM (left-looking over block columns j = 0 .. nbc-1)
// ---------------------------------------------------------
// Invariant after processing block columns < j, with pivots P (columns,
// ascending) and R = RREF prefix (identity at the pivot columns):
//   for every row r of A:   A[r, C] - A[r, P] * R[:, C] == 0   on processed C.
// Because R has the identity at P, the multipliers of row r are just its
// ORIGINAL entries at the pivot columns -- gathered structurally from the
// BlockMat (a Scalar / I pivot block contributes a diagonal, a structural
// zero contributes nothing; no L factor is stored).  Per panel j (width d):
//   1. cand = active rows of the block rows that have a structural non-zero
//      at j or at any pivot block column (all other rows have S == 0 exactly);
//   2. S = A[cand, j] - A[cand, P] * R[:, j]            (one GEMM);
//   3. R_j = RREF(S), pivcols_j                          (elim::rref on the panel);
//      row-rank profile of S[:, pivcols_j] -> pivot rows prows, B = S[prows, pivcols_j];
//   4. new rows = [0 | R_j | B^{-1} (A[prows, trail] - A[prows, P] R[:, trail])]  (two GEMMs);
//   5. reduce above: R[0:rk, >= col_off[j]] -= R[0:rk, pivcols_j] * new[:, >= col_off[j]]  (one GEMM);
//   6. append, mark prows inactive, mark j a pivot block column.
// A proof that the invariant is preserved is in BlockElim.cpp above rref().
//
// WHY THE RESULTS ARE LEGACY-IDENTICAL (D5b)
// -----------------------------------------
// At the end every row of A lies in the row space of R and R is in reduced
// row echelon form, so R == RREF(A) -- unique -- and its pivot columns are
// the column rank profile, ascending, one row per pivot in ascending pivot
// order with no zero rows: exactly the "true scalar RREF row order" the
// legacy block layer emitted and factor_basis_block indexed arithmetically.
// Every basis below is then the fixed D5b function of (RREF, pivots, original
// columns), so a correct algorithm is automatically bit-identical to
// FlatMatrix / BlockFlatMatrix / factor_basis_block:
//   kernel   one vector per free column ascending, 1 at the free column,
//            -R[i][j_free] at pivot i           (== BlockFlatMatrix::flat_kernel)
//   image    original columns at the pivot indices ascending (alpha*e_j for
//            a Scalar block, Q11)               (== flat_image, engine image_basis)
//   factor   greedy leftmost-independent generators modulo the relations,
//            projector rows = identity-block rows independent_relations + i
//            of RREF([Rl | G | I])              (== factor_basis_block)
//   solve    coordinates read off RREF([B | Y]), legacy exception on an
//            out-of-span column                (== change_of_basis_fm_multi)
// Which rows are chosen as pivot rows, the panel widths and the GEMM blocking
// never enter the output.  BlockEliminationValidation checks all of it
// entrywise.
//
// STRUCTURE EXPLOITED (D5 budget: 20-30 %, the real win is GEMM-boundedness)
//   * structural zeros never enter a panel (cand), never a gather;
//   * I / alpha*I blocks are gathered as diagonals -- the identity
//     augmentation of factor_basis is NEVER materialised as input; its
//     columns are d_V-wide structural blocks;
//   * the dense operands of factor_basis / solve (relation, generator and
//     right-hand-side columns) are BORROWED (BlockPool::borrow), chunked into
//     `chunk`-wide column panels, not copied;
//   * the multipliers are the input's own blocks, so nothing is stored per
//     row besides R.
// Memory: R (rank x n) plus per-panel scratch (cand x d, cand x rank).  For
// factor_basis the trailing identity part of R is the transform U legacy
// also materialised (dim_H x a of it is the projector); it is inherent to the
// D5b projector contract (the rows must be reduced against the left-null
// rows), see the note in BlockElim.cpp.

#include <cstdint>
#include <memory>
#include <vector>

#include "BlockMat.h"
#include "DenseBuf.h"
#include "DenseView.h"

namespace engine {
namespace blk {

struct ElimParams {
    uint32_t chunk = 256;   // column-chunk width for dense operands of factor_basis / solve
};

struct ElimStats {
    uint64_t panels = 0, panels_with_pivots = 0, dense_rrefs = 0, gemms = 0;
    uint64_t panel_rows = 0;     // sum of cand over panels
    uint64_t gathered_blocks = 0, scalar_blocks_gathered = 0;
};
const ElimStats& elim_stats();
void reset_elim_stats();

template<class Elt>
struct BlockRrefResult {
    uint32_t rank = 0;
    std::vector<uint32_t> pivots;      // global pivot columns, ascending
    // (matrix-engine Phase 4) the ROW RANK PROFILE in pivot order: pivot_rows[i]
    // is the original row of A that supplied pivot i (row i of R is the
    // combination of rows pivot_rows[0..i] that the RREF requires), so
    // A[pivot_rows, pivots] is the rank x rank invertible pivot block and
    // R = A[pivot_rows, pivots]^{-1} . A[pivot_rows, :].  tape::factor_basis's
    // projector is built from it without the ambient-wide identity augmentation.
    std::vector<uint32_t> pivot_rows;
    DenseBuf<Elt> R;                   // rank x n, row-major: RREF(A) without its zero rows
};

// RREF of A (rank rows, no zero rows) and its pivot columns.
template<class Elt>
BlockRrefResult<Elt> rref(const BlockMat<Elt>& A);

template<class Elt>
uint32_t rank(const BlockMat<Elt>& A);

// RREF re-blocked as a BlockMat: one block row of height rank, A's column
// partition; structural zeros and alpha*I blocks detected.
template<class Elt>
BlockMat<Elt> rref_block(const BlockMat<Elt>& A, const BlockRrefResult<Elt>& res);

// Kernel basis as a multi-vector: cols() x (cols() - rank), row partition =
// A's column partition, `layout` = physical layout of the single backing
// buffer (column-major: every basis vector contiguous, the L1 layout).
template<class Elt>
BlockMat<Elt> kernel_basis(const BlockMat<Elt>& A, const BlockRrefResult<Elt>& res,
                           Layout layout = Layout::ColMajor);
template<class Elt>
BlockMat<Elt> kernel_basis(const BlockMat<Elt>& A, Layout layout = Layout::ColMajor);

// Image basis as a multi-vector: rows() x rank, row partition = A's row
// partition (the original columns of A at the pivot indices).
template<class Elt>
BlockMat<Elt> image_basis(const BlockMat<Elt>& A, const BlockRrefResult<Elt>& res,
                          Layout layout = Layout::ColMajor);
template<class Elt>
BlockMat<Elt> image_basis(const BlockMat<Elt>& A, Layout layout = Layout::ColMajor);

// factor_basis: generators G (a x ng) modulo relations Rl (a x nr), ambient
// space partitioned by row_dims (sum == a).  Result:
//   pivots                 indices into G of the chosen basis, ascending
//   independent_relations  rank of Rl
//   basis                  a x dim_H multi-vector (the chosen G columns), row partition row_dims
//   projector              dim_H x a row-major (legacy factor_basis_block projector)
// ng == 0 gives an empty result (dim_H = 0, projector 0 x a).
template<class Elt>
struct FactorBasisResult {
    std::vector<uint32_t> pivots;
    uint32_t independent_relations = 0;
    BlockMat<Elt> basis;
    DenseBuf<Elt> projector;
};
template<class Elt>
FactorBasisResult<Elt> factor_basis(std::shared_ptr<BlockPool<Elt>> pool, Elt p,
                                    const std::vector<uint32_t>& row_dims,
                                    DenseView<const Elt> Rl, DenseView<const Elt> G,
                                    const ElimParams& params = ElimParams());
// Multi-vector overload: Rl and G are nbc == 1 BlockMats over the same row
// partition (either may have zero columns).
template<class Elt>
FactorBasisResult<Elt> factor_basis(const BlockMat<Elt>& Rl, const BlockMat<Elt>& G,
                                    const ElimParams& params = ElimParams());

// solve_multi_rhs: coordinates of every column of Y (a x ny) in the columns
// of B (a x nb): result (ny x nb) row-major, result(t, j) = coefficient of
// column j of B in column t of Y.  Throws std::runtime_error with the legacy
// message when a column of Y is outside the span.
template<class Elt>
DenseBuf<Elt> solve_multi_rhs(std::shared_ptr<BlockPool<Elt>> pool, Elt p,
                              const std::vector<uint32_t>& row_dims,
                              DenseView<const Elt> B, DenseView<const Elt> Y,
                              const ElimParams& params = ElimParams());
template<class Elt>
DenseBuf<Elt> solve_multi_rhs(const BlockMat<Elt>& B, const BlockMat<Elt>& Y,
                              const ElimParams& params = ElimParams());

// Dense view of a multi-vector: zero-copy when it has a single backing
// window, otherwise materialised into `scratch`.
template<class Elt>
DenseView<const Elt> multivector_view(const BlockMat<Elt>& mv, DenseBuf<Elt>& scratch,
                                      Layout layout = Layout::ColMajor);

} // namespace blk
} // namespace engine
