// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// RefLinAlg.h -- the naive in-tree oracle (Phase 2, plan D6)
// =====================================================================
//
// Schoolbook GEMM / GEMV / TRMM / transpose / RREF written with the LEGACY
// primitives ModArithmetic::mod_mul_w / mod_add_w and nothing else: straight
// triple loops, canonical reduction after every product, no delayed
// accumulation, no layout tricks. It exists only so KernelBitIdentityValidation
// (and, later, the elimination validations) have an independent answer to
// compare against entrywise. It must stay naive; do not optimise it.
//
// All routines address operands through DenseView::at(i, j), so any layout
// or transposition flag is exercised through the same index map the kernels
// resolve -- the oracle sees the logical matrix, never the bytes.
//
// RREF / kernel basis follow the legacy contract of plan D5b verbatim:
//   * RREF: leftmost pivot column, topmost non-zero row from r downward,
//     leading 1, eliminate above AND below, zero rows at the bottom.
//   * kernel basis: one vector per free column in ascending column index;
//     coordinate at the free column is 1; coordinate at the i-th pivot
//     column is  -RREF[i][j_free]; other coordinates 0.

#include <cstdint>
#include <vector>

#include "DenseView.h"

namespace engine {
namespace ref {

// C = A*B (op Set), C += A*B (Add), C -= A*B (Sub) -- same GemmOp encoding as
// the kernels but spelled as an int to keep this header free of Kernels.h:
// 0 = Set, 1 = Add, 2 = Sub.
template<class Elt>
void gemm(Elt p, DenseView<const Elt> A, DenseView<const Elt> B, DenseView<Elt> C, int op = 0);

template<class Elt>
void gemv(Elt p, DenseView<const Elt> A, DenseView<const Elt> x, DenseView<Elt> y, int op = 0);

// C = T*B with T unit upper (upper = true) / unit lower triangular; only the
// strict triangle of T is read.
template<class Elt>
void trmm(Elt p, DenseView<const Elt> T, DenseView<const Elt> B, DenseView<Elt> C, bool upper);

template<class Elt>
void transpose(DenseView<const Elt> src, DenseView<Elt> dst);

// In-place RREF of M; returns the pivot columns in ascending order.
template<class Elt>
std::vector<uint32_t> rref(Elt p, DenseView<Elt> M);

// Kernel basis of A (A is not modified): vectors of length A.cols(), one per
// free column ascending, per the D5b rule above.
template<class Elt>
std::vector<std::vector<Elt>> kernel_basis(Elt p, DenseView<const Elt> A);

// ---------------------------------------------------------------------------
// Z/ell^k (ENGINE_API.md Sec. 13.9; hecke-extensions slot 8 WP-1): the naive
// oracle of engine::elim::howell and invariant_factors.  It takes a bare
// (ell, k) pair -- it is not an entry point of the pipeline -- and runs on the
// legacy mod_*_w arithmetic with modulus ell^k.
//
// howell: Sec. 13.5's forward pass EXACTLY as written (valuation pivoting:
// minimal valuation, ties to the topmost active row; scale to ell^v; exact
// elimination; the closure row ell^{k-v} * pivot row appended at the BOTTOM
// of the active list, always), then the reduce-above pass.  k = 1 is allowed
// here (and only here); the result must then equal rref()'s non-zero rows.
// ---------------------------------------------------------------------------
template<class Elt>
struct RefHowell {
    std::vector<std::vector<Elt>> rows;     // the non-zero rows of the Howell form
    std::vector<uint32_t> pivots;           // leading columns, strictly increasing
    std::vector<uint32_t> pivot_v;          // row i leads with ell^{pivot_v[i]}
};
template<class Elt>
RefHowell<Elt> howell(Elt ell, uint32_t k, DenseView<const Elt> M);

// The exponents e_1 >= e_2 >= ... of M = (colspan G + colspan Rl) / colspan Rl
// by the NAIVE Smith form of a presentation chosen independently of the
// engine's: generators = the columns of G, relations = the G-coordinates of
// the kernel of [G | Rl] (from howell of [[G | Rl]^T | I]); Smith by the first
// minimal-valuation entry in ROW-major order (a different tie rule on
// purpose -- the exponents are unique).  Rl may have 0 columns.
template<class Elt>
std::vector<uint32_t> invariant_factors(Elt ell, uint32_t k, DenseView<const Elt> G, DenseView<const Elt> Rl);

// ---------------------------------------------------------------------------
// (hecke-extensions slot 11, WP-18a; ENGINE_API.md Sec. 13.12.3 / 13.12.4)
// The u128 oracle arithmetic: NAIVE and independent of ModCtx<u128>'s
// Montgomery-128 primitives -- a double-and-add mulmod and the extended
// Euclidean algorithm with sign-tracked u128 coefficients.  Modulus p odd or
// even, 2 <= p < 2^127; inputs canonical (< p).  gemm / gemv / transpose /
// howell / invariant_factors above are instantiated for u128 on exactly this
// arithmetic (rref and kernel_basis are field routes and are not).
// ---------------------------------------------------------------------------
namespace naive128 {
using u128 = unsigned __int128;
u128 add(u128 a, u128 b, u128 p);
u128 sub(u128 a, u128 b, u128 p);
u128 mul(u128 a, u128 b, u128 p);    // double-and-add, 128 steps
// a^{-1} mod p; throws std::runtime_error("Modular inverse does not exist.")
// when gcd(a, p) != 1 (a == 0 included).
u128 inv(u128 a, u128 p);
} // namespace naive128

} // namespace ref
} // namespace engine
