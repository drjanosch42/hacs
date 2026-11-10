// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Hecke/FpRank.h -- the rank of a row-major matrix over F_p through the engine
// (S24 CL-3; CODEAUDIT B3 step (a); docs/implementation/slot24/CLEANUP_plan_1030.md)
// ===========================================================================
//
// Replaces the hand-rolled dense Gauss loops of HeckeLevelN (FieldRank),
// HeckeLevelEll (HRankModEll), HeckeLevelInfRes (RankRowsModEll) and
// HeckeOrdinaryPart (the rank-only FieldNullspace calls).  The rank is an
// invariant of the matrix, so no value can change; the engine call is the one
// HeckeReduction / HeckeInvolution already make
// (engine::elim::column_rank_profile on a field context; ENGINE_API.md).
// p must be prime.  Entries may be unreduced; they are reduced mod p here.
// ===========================================================================
#pragma once
#ifndef HECKE_FP_RANK_H
#define HECKE_FP_RANK_H

#include <cstdint>
#include <vector>

#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/ModCtx.h"

namespace hecke {

namespace detail {
template <class Elt, class Src>
uint32_t RankModPrimeAs(const Src* a, uint32_t rows, uint32_t cols, Elt p) {
    std::vector<Elt> buf(static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<Elt>(a[i] % p);
    const engine::ModCtx<Elt> ctx(p);
    engine::DenseBuf<Elt> m(buf.data(), buf.size(), rows, cols, engine::Layout::RowMajor);
    return engine::elim::column_rank_profile<Elt>(ctx, engine::view(m)).rank;
}
} // namespace detail

// Rank over F_p of the rows x cols matrix `a` (row-major).
template <class Src>
uint32_t RankModPrime(const Src* a, uint32_t rows, uint32_t cols, uint64_t p) {
    if (rows == 0 || cols == 0) return 0;
    if (p <= 0xFFFFFFFFull) return detail::RankModPrimeAs<uint32_t>(a, rows, cols, static_cast<uint32_t>(p));
    return detail::RankModPrimeAs<uint64_t>(a, rows, cols, p);
}

} // namespace hecke

#endif // HECKE_FP_RANK_H
