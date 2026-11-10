// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_DENSE_RREF_H
#define HECKE_DENSE_RREF_H
// ===========================================================================
// (S24 CL-G2, CODEAUDIT-B B3 (b)) The in-place RREF of a row-major matrix over a prime field, on the engine's
// elimination (engine::elim::rref, ENGINE_API.md Sec. 8).  It replaced the hand-rolled Gauss--Jordan loops of
// HeckeLevelNSparse (Rref / FpRank / FpKernelBasis), HeckeTameFamily (tame::Rref, hence NullSpace / SolveSquare) and
// HeckeOrdinaryPart (FieldNullspace).  Value-identical: the RREF of a matrix over a field (leftmost pivots, leading 1,
// zero rows at the bottom) and its pivot columns are unique; CL-G2 compared the old bodies entrywise with this one on
// every call of the pinned validations before deleting them (docs/implementation/slot24/CLG2_acceptance.md).
// ===========================================================================

#include <cstdint>
#include <vector>

#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/ModCtx.h"

namespace hecke {

// A (rows x cols, row-major, any representatives) becomes RREF(A) over F_p with entries in [0, p); returns the pivot
// columns, ascending (the column rank profile).  p prime.
template <class Elt>
std::vector<uint32_t> RrefRowMajor(std::vector<Elt>& A, uint32_t rows, uint32_t cols, Elt p) {
    ENGINE_CHECK_MSG(A.size() == static_cast<size_t>(rows) * cols, "RrefRowMajor: store size != rows * cols");
    for (Elt& x : A) x %= p;
    if (rows == 0 || cols == 0) return {};
    const engine::ModCtx<Elt> ctx(p);
    engine::DenseBuf<Elt> buf(A.data(), A.size(), rows, cols, engine::Layout::RowMajor);
    return engine::elim::rref(ctx, engine::view(buf)).pivots;
}

} // namespace hecke

#endif // HECKE_DENSE_RREF_H
