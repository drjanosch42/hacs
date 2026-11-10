// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 10, WP-11 (Line 2, M1(a)): reading cochain-level
// operators on the isotropy-resolution model as operators on cohomology,
// basis-free.  See IntegralHecke.h.
#include "IntegralHecke.h"

#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Kernels.h"

#include <algorithm>
#include <stdexcept>

namespace integral_model {

using engine::DenseBuf;
using engine::ModCtx;

ModCtx<uint32_t> RingContext(uint32_t ell, uint32_t k) {
    ENGINE_CHECK_MSG(k >= 1, "RingContext: k >= 1");
    return (k == 1) ? ModCtx<uint32_t>(ell) : ModCtx<uint32_t>::prime_power(ell, k);
}

template<class Elt>
DenseBuf<Elt> MatMul(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& A, const DenseBuf<Elt>& B) {
    ENGINE_CHECK_MSG(A.cols == B.rows, "MatMul: inner dimensions differ");
    DenseBuf<Elt> C(A.rows, B.cols);
    if (A.rows == 0 || B.cols == 0 || A.cols == 0) return C;
    engine::kernels::gemm(ctx, engine::cview(A), engine::cview(B), engine::view(C));
    return C;
}

template<class Elt>
DenseBuf<Elt> LinComb(const ModCtx<Elt>& ctx, Elt a, const DenseBuf<Elt>& A, Elt b, const DenseBuf<Elt>& B) {
    ENGINE_CHECK_MSG(A.rows == B.rows && A.cols == B.cols, "LinComb: shapes differ");
    DenseBuf<Elt> C(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) C.at(i, j) = ctx.add(ctx.mul(a, A.at(i, j)), ctx.mul(b, B.at(i, j)));
    return C;
}

template<class Elt>
DenseBuf<Elt> AddScalar(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& A, Elt c) {
    ENGINE_CHECK_MSG(A.rows == A.cols, "AddScalar: square matrix expected");
    DenseBuf<Elt> C = A;
    for (uint32_t i = 0; i < A.rows; ++i) C.at(i, i) = ctx.add(C.at(i, i), c);
    return C;
}

template<class Elt>
DenseBuf<Elt> HStack(const DenseBuf<Elt>& A, const DenseBuf<Elt>& B) {
    ENGINE_CHECK_MSG(A.rows == B.rows, "HStack: row counts differ");
    DenseBuf<Elt> C(A.rows, A.cols + B.cols);
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) C.at(i, j) = A.at(i, j);
        for (uint32_t j = 0; j < B.cols; ++j) C.at(i, A.cols + j) = B.at(i, j);
    }
    return C;
}

template<class Elt>
Elt ReduceSigned(const ModCtx<Elt>& ctx, int64_t v) {
    const uint64_t p = static_cast<uint64_t>(ctx.p);
    const uint64_t mag = (v < 0) ? static_cast<uint64_t>(-(v + 1)) + 1u : static_cast<uint64_t>(v);
    const uint64_t r = mag % p;
    return static_cast<Elt>((v < 0 && r != 0) ? p - r : r);
}

namespace {

template<class Elt>
DenseBuf<Elt> Transposed(const DenseBuf<Elt>& A) {
    DenseBuf<Elt> T(A.cols, A.rows);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) T.at(j, i) = A.at(i, j);
    return T;
}

template<class Elt>
DenseBuf<Elt> Columns(const DenseBuf<Elt>& A, const std::vector<uint32_t>& idx) {
    DenseBuf<Elt> C(A.rows, static_cast<uint32_t>(idx.size()));
    for (uint32_t t = 0; t < idx.size(); ++t)
        for (uint32_t i = 0; i < A.rows; ++i) C.at(i, t) = A.at(i, idx[t]);
    return C;
}

// Rl with at least one column (a zero column when empty): the engine entry
// points are fed a non-degenerate relation block.
template<class Elt>
DenseBuf<Elt> NonEmpty(const DenseBuf<Elt>& Rl, uint32_t rows) {
    if (Rl.cols > 0) return Rl;
    return DenseBuf<Elt>(rows, 1);
}

// The right kernel of A as columns: a basis over a field (D5b kernel_basis),
// the Howell generators over Z/ell^k.
template<class Elt>
DenseBuf<Elt> KernelColumns(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& A) {
    if (A.cols == 0) return DenseBuf<Elt>(0, 0);
    if (A.rows == 0) {
        DenseBuf<Elt> I(A.cols, A.cols);
        for (uint32_t i = 0; i < A.cols; ++i) I.at(i, i) = 1;
        return I;
    }
    if (ctx.is_field()) {
        DenseBuf<Elt> R = A;
        const auto rr = engine::elim::rref(ctx, engine::view(R));
        const DenseBuf<Elt> K = engine::elim::kernel_basis(ctx, engine::cview(R), rr.pivots);
        return Transposed(K);
    }
    const auto h = engine::elim::kernel_module(ctx, engine::cview(A));
    return Transposed(h.H);
}

}  // namespace

template<class Elt>
uint32_t ColumnLength(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& A) {
    if (A.rows == 0 || A.cols == 0) return 0;
    if (ctx.is_field()) {
        DenseBuf<Elt> R = A;
        return engine::elim::rref(ctx, engine::view(R)).rank;
    }
    // |rowspan A| = |colspan A| over Z/ell^k (Smith normal form); the Howell
    // length of the row module is read on A itself.
    return engine::elim::length(ctx, engine::cview(A));
}

// ---------------------------------------------------------------------------
// ModelCohomology
// ---------------------------------------------------------------------------
template<class Elt>
ModelCohomology<Elt>::ModelCohomology(const IsotropyModel& M, const ModCtx<Elt>& ctx, uint32_t n)
    : ctx_(ctx), n_(n), q_max_(M.q_max()) {
    const uint32_t dV = n + 1;
    dims_.resize(q_max_ + 2);
    for (uint32_t q = 0; q <= q_max_ + 1; ++q) dims_[q] = M.rank(q) * dV;
    delta_.resize(q_max_ + 1);
    for (uint32_t q = 0; q <= q_max_; ++q) {
        delta_[q] = M.coboundary(ctx, n, q);
        ENGINE_CHECK_MSG(delta_[q].rows == dims_[q + 1] && delta_[q].cols == dims_[q],
                         "ModelCohomology: coboundary shape");
    }
    Z_.resize(q_max_ + 1);
    B_.resize(q_max_ + 1);
    lZ_.assign(q_max_ + 1, 0);
    lB_.assign(q_max_ + 1, 0);
    for (uint32_t q = 0; q <= q_max_; ++q) {
        Z_[q] = KernelColumns(ctx, delta_[q]);
        if (Z_[q].rows == 0) Z_[q] = DenseBuf<Elt>(dims_[q], 0);
        B_[q] = (q == 0) ? DenseBuf<Elt>(dims_[0], 0) : delta_[q - 1];
        // length Z^q = cols * k - length(im delta^q) (rank-nullity, Sec. 13.6),
        // read independently of the generators; the generators are checked
        // to span a module of that length.
        const uint32_t k = ctx.k;
        lZ_[q] = dims_[q] * k - ColumnLength(ctx, delta_[q]);
        ENGINE_CHECK_MSG(ColumnLength(ctx, Z_[q]) == lZ_[q], "ModelCohomology: cocycle generators span Z^q");
        ENGINE_CHECK_MSG(are_cocycles(q, Z_[q]), "ModelCohomology: generators are cocycles");
        lB_[q] = ColumnLength(ctx, B_[q]);
        ENGINE_CHECK_MSG(lB_[q] <= lZ_[q], "ModelCohomology: B^q in Z^q");
    }
}

template<class Elt>
bool ModelCohomology<Elt>::in_coboundaries(uint32_t q, const DenseBuf<Elt>& Y) const {
    ENGINE_CHECK_MSG(Y.rows == dims_.at(q), "in_coboundaries: row count");
    if (Y.cols == 0) return true;
    return ColumnLength(ctx_, HStack(B_.at(q), Y)) == lB_.at(q);
}

template<class Elt>
bool ModelCohomology<Elt>::induces_zero(uint32_t q_src, uint32_t q_dst, const DenseBuf<Elt>& P) const {
    ENGINE_CHECK_MSG(P.rows == dims_.at(q_dst) && P.cols == dims_.at(q_src), "induces_zero: shape");
    return in_coboundaries(q_dst, MatMul(ctx_, P, Z_.at(q_src)));
}

template<class Elt>
bool ModelCohomology<Elt>::are_cocycles(uint32_t q, const DenseBuf<Elt>& Y) const {
    const DenseBuf<Elt> D = MatMul(ctx_, delta_.at(q), Y);
    for (uint32_t i = 0; i < D.rows; ++i)
        for (uint32_t j = 0; j < D.cols; ++j)
            if (D.at(i, j) != 0) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Subquotients and induced operators
// ---------------------------------------------------------------------------
template<class Elt>
InducedOperator<Elt> Subquotient(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& G, const DenseBuf<Elt>& Rl) {
    ENGINE_CHECK_MSG(G.rows == Rl.rows, "Subquotient: row counts differ");
    InducedOperator<Elt> N;
    N.basis = DenseBuf<Elt>(G.rows, 0);
    if (G.cols == 0 || G.rows == 0) return N;
    const DenseBuf<Elt> R1 = NonEmpty(Rl, G.rows);
    if (ctx.is_field()) {
        const auto idx = engine::elim::complement_basis(ctx, engine::cview(R1), engine::cview(G),
                                                        static_cast<DenseBuf<Elt>*>(nullptr));
        N.basis = Columns(G, idx);
        N.exponents.assign(idx.size(), 1u);
    } else {
        const auto dec = engine::elim::cyclic_decomposition(ctx, engine::cview(G), engine::cview(R1));
        N.exponents = dec.exponents;
        N.basis = dec.basis;
        N.dec = dec;
    }
    return N;
}

template<class Elt>
DenseBuf<Elt> OperatorMatrix(const ModCtx<Elt>& ctx, const InducedOperator<Elt>& N, const DenseBuf<Elt>& Rl,
                             const DenseBuf<Elt>& Theta) {
    const uint32_t t = N.rank();
    DenseBuf<Elt> A(t, t);
    if (t == 0) return A;
    ENGINE_CHECK_MSG(Theta.cols == N.basis.rows && Theta.rows == N.basis.rows, "OperatorMatrix: shape");
    const DenseBuf<Elt> Y = MatMul(ctx, Theta, N.basis);
    const DenseBuf<Elt> R1 = NonEmpty(Rl, N.basis.rows);
    if (ctx.is_field()) {
        const auto coords = engine::elim::solve_multi_rhs(ctx, engine::cview(HStack(N.basis, R1)), engine::cview(Y));
        for (uint32_t j = 0; j < t; ++j)
            for (uint32_t i = 0; i < t; ++i) A.at(i, j) = coords[j][i];
        return A;
    }
    const auto c = engine::elim::cyclic_coordinates(ctx, N.dec, engine::cview(Y));
    for (uint32_t j = 0; j < t; ++j)
        for (uint32_t i = 0; i < t; ++i) A.at(i, j) = c[j][i];
    return A;
}

template<class Elt>
std::vector<Elt> CharpolyDivisionFree(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& A) {
    ENGINE_CHECK_MSG(A.rows == A.cols, "CharpolyDivisionFree: square matrix expected");
    const uint32_t t = A.rows;
    ENGINE_CHECK_MSG(t <= 7, "CharpolyDivisionFree: t <= 7 (Leibniz expansion)");
    std::vector<Elt> c(t + 1, 0);
    c[t] = 1;
    for (uint32_t mask = 1; mask < (1u << t); ++mask) {
        std::vector<uint32_t> S;
        for (uint32_t i = 0; i < t; ++i)
            if (mask & (1u << i)) S.push_back(i);
        const uint32_t s = static_cast<uint32_t>(S.size());
        std::vector<uint32_t> perm(s);
        for (uint32_t i = 0; i < s; ++i) perm[i] = i;
        Elt det = 0;
        do {
            uint32_t inv = 0;
            for (uint32_t i = 0; i < s; ++i)
                for (uint32_t j = i + 1; j < s; ++j)
                    if (perm[i] > perm[j]) ++inv;
            Elt term = 1;
            for (uint32_t i = 0; i < s; ++i) term = ctx.mul(term, A.at(S[i], S[perm[i]]));
            det = (inv % 2 == 0) ? ctx.add(det, term) : ctx.sub(det, term);
        } while (std::next_permutation(perm.begin(), perm.end()));
        // coefficient of x^{t-s} is (-1)^s * (sum of principal s-minors)
        c[t - s] = (s % 2 == 0) ? ctx.add(c[t - s], det) : ctx.sub(c[t - s], det);
    }
    return c;
}

template<class Elt>
DenseBuf<Elt> ReducedCocycles(const ModelCohomology<Elt>& big, uint32_t q, const ModCtx<Elt>& small) {
    ENGINE_CHECK_MSG(big.ctx().ell == small.ell || big.ctx().p == small.p, "ReducedCocycles: same prime");
    ENGINE_CHECK_MSG(big.ctx().p % small.p == 0, "ReducedCocycles: small modulus divides the big one");
    const DenseBuf<Elt>& Zb = big.Z(q);
    DenseBuf<Elt> R(Zb.rows, Zb.cols);
    for (uint32_t i = 0; i < Zb.rows; ++i)
        for (uint32_t j = 0; j < Zb.cols; ++j) R.at(i, j) = static_cast<Elt>(Zb.at(i, j) % small.p);
    return R;
}

#define INTEGRAL_HECKE_INSTANTIATE(Elt)                                                                          \
    template DenseBuf<Elt> MatMul<Elt>(const ModCtx<Elt>&, const DenseBuf<Elt>&, const DenseBuf<Elt>&);          \
    template DenseBuf<Elt> LinComb<Elt>(const ModCtx<Elt>&, Elt, const DenseBuf<Elt>&, Elt, const DenseBuf<Elt>&); \
    template DenseBuf<Elt> AddScalar<Elt>(const ModCtx<Elt>&, const DenseBuf<Elt>&, Elt);                         \
    template DenseBuf<Elt> HStack<Elt>(const DenseBuf<Elt>&, const DenseBuf<Elt>&);                               \
    template Elt ReduceSigned<Elt>(const ModCtx<Elt>&, int64_t);                                                  \
    template uint32_t ColumnLength<Elt>(const ModCtx<Elt>&, const DenseBuf<Elt>&);                                \
    template class ModelCohomology<Elt>;                                                                          \
    template InducedOperator<Elt> Subquotient<Elt>(const ModCtx<Elt>&, const DenseBuf<Elt>&, const DenseBuf<Elt>&); \
    template DenseBuf<Elt> OperatorMatrix<Elt>(const ModCtx<Elt>&, const InducedOperator<Elt>&,                   \
                                               const DenseBuf<Elt>&, const DenseBuf<Elt>&);                       \
    template std::vector<Elt> CharpolyDivisionFree<Elt>(const ModCtx<Elt>&, const DenseBuf<Elt>&);               \
    template DenseBuf<Elt> ReducedCocycles<Elt>(const ModelCohomology<Elt>&, uint32_t, const ModCtx<Elt>&);

INTEGRAL_HECKE_INSTANTIATE(uint32_t)

#undef INTEGRAL_HECKE_INSTANTIATE

}  // namespace integral_model
