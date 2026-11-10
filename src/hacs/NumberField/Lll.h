// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Lll.h -- in-house exact LLL lattice reduction (hecke-extensions slot 21, T1;
// plan docs/implementation/slot21/00-slot21-plan.md "Talk": "integer LLL with
// exact rationals, size-reduction + Lovasz"; owner ruling of 2026-10-03: no
// external CAS). STL only, on nf::BigInt / nf::BigRational; no floating point.
//
// Source: H. Cohen, A Course in Computational Algebraic Number Theory, GTM 138,
// sec. 2.6 (the accepted substitute for LLL 1982, missing_literature.md
// LIT-09): Definition 2.6.1 (LLL-reduced), Algorithm 2.6.3 (LLL, exact
// rational Gram--Schmidt, in the Gram-matrix form of its Remark (2)) and
// Algorithm 2.6.7 (integral LLL: d_i = prod_{j<=i} B_j and
// lambda_{ij} = d_j mu_{ij}, all integers). Both make the same decisions (the
// nearest integer is floor(mu + 1/2) in both, Lovasz test strict "<"), so on
// an integral Gram matrix they return the same transformation matrix; the
// validation compares them.
//
// Conventions. A lattice is given by its Gram matrix G (n x n, symmetric,
// positive definite: the basis must be linearly independent -- a zero B_k is
// an NF_CHECK failure, no MLLL) or by n integer row vectors in Z^m. The
// result's H is unimodular, ROW convention: new b_i = sum_j H[i][j] old b_j,
// so the reduced Gram matrix is H G H^T. delta is rational, 1/4 < delta <= 1
// (Cohen's c, Remark (3); delta = 3/4 is the default, delta = 1 terminates but
// without the polynomial bound).
#include <cstdint>
#include <string>
#include <vector>

#include "BigInt.h"
#include "BigRational.h"

namespace nf {

using IntMat = std::vector<std::vector<BigInt>>;
using RatMat = std::vector<std::vector<BigRational>>;

struct LllResult {
    IntMat H;                     // unimodular, rows = new basis in old coordinates
    std::vector<BigRational> B;   // B_i = |b*_i|^2 of the reduced basis
    RatMat mu;                    // mu[i][j], j < i (strict lower triangle; the rest 0)
    uint64_t swaps = 0;           // SWAP calls
    uint64_t reductions = 0;      // RED calls that subtracted (q != 0)
};

BigRational lll_default_delta();  // 3/4

// Cohen Alg. 2.6.3 on a rational Gram matrix.
LllResult lll_gram_rational(const RatMat& G, const BigRational& delta = lll_default_delta());
// Cohen Alg. 2.6.7 on an integral Gram matrix (all-integer arithmetic).
LllResult lll_gram_integral(const IntMat& G, const BigRational& delta = lll_default_delta());

// Rows in Z^m (n <= m, independent): Gram, integral LLL, and the reduced rows H * rows.
struct LllBasisResult {
    IntMat basis;  // the reduced rows
    LllResult lll;
};
LllBasisResult lll_basis(const IntMat& rows, const BigRational& delta = lll_default_delta());

// Exact helpers.
IntMat gram_of_rows(const IntMat& rows);                          // rows * rows^T
IntMat int_mat_mul(const IntMat& A, const IntMat& B);             // A * B
IntMat transform_gram(const IntMat& H, const IntMat& G);          // H G H^T
RatMat transform_gram(const IntMat& H, const RatMat& G);
BigInt det_bareiss(IntMat A);                                     // exact, fraction-free

// Definition 2.6.1 checked exactly from scratch (fresh Gram--Schmidt on G):
// |mu_ij| <= 1/2 and B_i >= (delta - mu_{i,i-1}^2) B_{i-1}, G positive definite.
// On failure returns false and, if why != nullptr, the first violated condition.
bool lll_is_reduced(const RatMat& G, const BigRational& delta, std::string* why = nullptr);
bool lll_is_reduced(const IntMat& G, const BigRational& delta, std::string* why = nullptr);

}  // namespace nf
