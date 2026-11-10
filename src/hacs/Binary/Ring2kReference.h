// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// Rung R6 of the Z/2^k ladder (slot-22 BIN-2): the in-house nf::BigInt reference.  ORACLE ONLY.
// Independent arithmetic: every entry is a non-negative BigInt reduced by floor_mod(2^k) after every
// operation; the inverse of a unit is the extended Euclidean algorithm (not Newton); the valuation is
// counted on the BigInt magnitude; exact division by 2^v is BigInt division.  The algorithms are the
// textbook ones: the Smith form by global minimal-valuation pivoting with BOTH the pivot's row and
// column cleared, and the Howell form by ENGINE_API.md Sec. 13.5's pass literally (active list,
// closure rows, reduce above one pivot at a time).  Compiled into cohomology_app only (the
// ModArithmetic library does not link NumberField).

#include <cstdint>
#include <vector>

#include "../../ModArithmetic/Binary/Ring2k.h"
#include "../NumberField/BigInt.h"

namespace engine {
namespace ring2k {
namespace ref6 {

struct BigMat {
    uint32_t rows = 0, cols = 0;
    std::vector<nf::BigInt> a;
    BigMat() = default;
    BigMat(uint32_t r, uint32_t c) : rows(r), cols(c), a(static_cast<size_t>(r) * c) {}
    nf::BigInt&       at(uint32_t i, uint32_t j)       { return a[static_cast<size_t>(i) * cols + j]; }
    const nf::BigInt& at(uint32_t i, uint32_t j) const { return a[static_cast<size_t>(i) * cols + j]; }
};

nf::BigInt from_words(const uint64_t* w, uint32_t nw);
void       to_words(const nf::BigInt& x, uint64_t* w, uint32_t nw);   // x >= 0
BigMat     from_wide(const WideMatrix& A);
WideMatrix to_wide(const BigMat& M, uint32_t k);
uint32_t   valuation(const nf::BigInt& x, uint32_t k);   // x canonical in [0, 2^k); k for 0

SmithResult smith(const WideMatrix& A);    // sigma ascending, each < k
WideHowell  howell(const WideMatrix& A);   // the unique Howell form

} // namespace ref6
} // namespace ring2k
} // namespace engine
