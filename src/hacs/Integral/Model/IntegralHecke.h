// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_MODEL_INTEGRAL_HECKE_H
#define INTEGRAL_MODEL_INTEGRAL_HECKE_H

// ===========================================================================
// Line 2, M1(a) (hecke-extensions slot 10, WP-11): the integral Hecke action,
// BASIS-FREE ([IB] Sec. 6.2 M1(a), tier A).  [C] = docs/latex/gl2_contract.tex
// (v1.9.1), cited by label; [IB] = docs/implementation/integral_line_design_brief.md.
//
// WHAT THIS UNIT IS.  The reading of a cochain-level operator on the
// isotropy-resolution model C^._I(R) = Hom_{Z[Gamma]}(Iso_., V_R)
// ([C, def:isomodel], IsotropyModel) as an operator on cohomology
// H^q(C_I (x) R) = H^q(Gamma; V_R) ([C, prop:isomodel](3)), over any ModCtx
// (a field F_ell or a chain ring Z/ell^k).  The operators themselves are
// built elsewhere and only READ here:
//   * T_p = the coset transport Theta^_g of the strict lift tau_Iso
//     (StrictLift::theta_hat, WP-10), which induces cor o conj_g o res on
//     H^.(Gamma; V_R) for every R, in every degree ([C, prop:strictlift](3));
//   * E_q = e u - as the cochain map E_q^* : C^q -> C^{q+2}
//     (PeriodicityClass::matrix, WP-13; [C, lem:bulkperiodic]).
// Nothing here chooses an integral basis: every statement is an identity of
// operators on a module (an inclusion P(Z^q) in B^q), a composition length,
// an invariant factor, or a characteristic polynomial on a FREE module over
// Z/ell^m or a vector space over F_ell -- tier A of [IB] Sec. 3.3.
//
// THE TORSION READINGS (ell in {2, 3}; the UCT of [C, prop:certifiede](3)).
// Write H^q = H^q(Gamma; Sym^n Z^2), Lambda^q its free quotient, T^q_ell its
// ell-primary torsion, and red_{K' -> K} : H^q(C_I (x) Z/ell^K') ->
// H^q(C_I (x) Z/ell^K) the reduction.  The UCT sequence
//   0 -> H^q (x) Z/ell^K -> H^q(C_I (x) Z/ell^K) -> Tor(H^{q+1}, Z/ell^K) -> 0
// is natural in integral cochain maps, and red acts on the Tor term as
// multiplication by ell^{K'-K} on T^{q+1}[ell^K'] -> T^{q+1}[ell^K].  Hence,
// for K' - K >= exp T^{q+1}_ell:
//   IM(red_{K' -> K}) = H^q (x) Z/ell^K  inside H^q(C_I (x) Z/ell^K),
// a T_p-stable submodule (T_p = Theta^ is integral, so it commutes with red).
// Consequently, with e_q = exp T^q_ell:
//   (sub)   for q >= 2 (Lambda^q = 0) and K >= e_q: IM(red) = T^q_ell;
//   (quot)  for K >= e_{q+1}: H^q(C_I (x) Z/ell^K) / IM(red) = T^{q+1}_ell
//           (the UCT boundary map is the Bockstein, T_p-equivariant);
//   (lat)   for q = 1: ell^{e_1} IM(red_{K'->K}) = ell^{e_1} Lambda^1 / ell^K
//           ~= Lambda^1 (x) Z/ell^{K - e_1}, FREE of rank rank Lambda^1,
//           with T_p acting as on Lambda^1 (x) Z_ell (multiplication by
//           ell^{e_1} is a T_p-equivariant isomorphism).
// So T_p on T^q_ell (degrees <= 5, two readings for q >= 2) and on
// Lambda^1 (x) Z/ell^m is read off the model with no integral basis and no
// big integer.
//
// DEPENDENCIES: IsotropyModel.h and ModArithmetic/ only (the Integral/Model/
// rule).  No bignum, no __int128 (OD-3 (c)).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "IsotropyModel.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace integral_model {

// F_ell (k == 1) or Z/ell^k (k >= 2) as an engine context.
engine::ModCtx<uint32_t> RingContext(uint32_t ell, uint32_t k);

// Dense helpers (every product exact mod ctx.p through engine::kernels::gemm).
template<class Elt>
engine::DenseBuf<Elt> MatMul(const engine::ModCtx<Elt>& ctx, const engine::DenseBuf<Elt>& A,
                             const engine::DenseBuf<Elt>& B);
// a*A + b*B (same shape).
template<class Elt>
engine::DenseBuf<Elt> LinComb(const engine::ModCtx<Elt>& ctx, Elt a, const engine::DenseBuf<Elt>& A, Elt b,
                              const engine::DenseBuf<Elt>& B);
// A + c*I (A square).
template<class Elt>
engine::DenseBuf<Elt> AddScalar(const engine::ModCtx<Elt>& ctx, const engine::DenseBuf<Elt>& A, Elt c);
// [A | B] (same number of rows).
template<class Elt>
engine::DenseBuf<Elt> HStack(const engine::DenseBuf<Elt>& A, const engine::DenseBuf<Elt>& B);
// A signed integer reduced into [0, ctx.p).
template<class Elt>
Elt ReduceSigned(const engine::ModCtx<Elt>& ctx, int64_t v);

// The composition length of the column module of A: the rank over a field,
// the Howell length over Z/ell^k (|colspan| = ell^length; rowspan and colspan
// have the same order).
template<class Elt>
uint32_t ColumnLength(const engine::ModCtx<Elt>& ctx, const engine::DenseBuf<Elt>& A);

// ---------------------------------------------------------------------------
// The cochain complex C_I (x) R at one n, with the cocycle and coboundary
// modules of every degree q = 0 .. q_max cached.
// ---------------------------------------------------------------------------
template<class Elt>
class ModelCohomology {
public:
    ModelCohomology(const IsotropyModel& M, const engine::ModCtx<Elt>& ctx, uint32_t n);

    const engine::ModCtx<Elt>& ctx() const { return ctx_; }
    uint32_t n() const { return n_; }
    uint32_t q_max() const { return q_max_; }
    uint32_t dim(uint32_t q) const { return dims_.at(q); }                     // w_q (n + 1)
    const engine::DenseBuf<Elt>& delta(uint32_t q) const { return delta_.at(q); }   // C^q -> C^{q+1}
    // Z^q: a basis (field) or the Howell generators (Z/ell^k), as COLUMNS.
    const engine::DenseBuf<Elt>& Z(uint32_t q) const { return Z_.at(q); }
    // B^q = im delta^{q-1}: the columns of delta^{q-1} (q = 0: no column).
    const engine::DenseBuf<Elt>& B(uint32_t q) const { return B_.at(q); }
    uint32_t length_Z(uint32_t q) const { return lZ_.at(q); }
    uint32_t length_B(uint32_t q) const { return lB_.at(q); }
    uint32_t length_H(uint32_t q) const { return lZ_.at(q) - lB_.at(q); }

    // Every column of Y (dim(q) rows) lies in B^q.
    bool in_coboundaries(uint32_t q, const engine::DenseBuf<Elt>& Y) const;
    // P (dim(q_dst) x dim(q_src)) maps Z^{q_src} into B^{q_dst}: P induces the
    // zero map H^{q_src} -> H^{q_dst}.  The basis-free form of an operator
    // identity on cohomology.
    bool induces_zero(uint32_t q_src, uint32_t q_dst, const engine::DenseBuf<Elt>& P) const;
    // Every column of Y is a cocycle of degree q.
    bool are_cocycles(uint32_t q, const engine::DenseBuf<Elt>& Y) const;

private:
    engine::ModCtx<Elt> ctx_;
    uint32_t n_ = 0, q_max_ = 0;
    std::vector<uint32_t> dims_;
    std::vector<engine::DenseBuf<Elt>> delta_, Z_, B_;
    std::vector<uint32_t> lZ_, lB_;
};

// ---------------------------------------------------------------------------
// An operator on a subquotient N = (colspan G + colspan Rl) / colspan Rl of
// cochains: its cyclic summands (exponents; over a field all 1) and the matrix
// of the operator induced by Theta in the summand basis -- [E] sec:ring (c)'s
// rule over Z/ell^k (engine::elim::cyclic_decomposition / cyclic_coordinates),
// complement_basis over a field.  Column convention: Theta b_j = sum_i A_ij b_i,
// A_ij in [0, ell^{e_i}).  Throws std::runtime_error if Theta(G) leaves
// colspan G + colspan Rl (Theta must preserve N).
// ---------------------------------------------------------------------------
template<class Elt>
struct InducedOperator {
    std::vector<uint32_t> exponents;
    engine::DenseBuf<Elt> basis;    // cochains b_i as columns
    engine::elim::CyclicDecomposition<Elt> dec;   // Z/ell^k only: the (c1)-(c5) data (coordinates)
    uint32_t rank() const { return static_cast<uint32_t>(exponents.size()); }
};

// The summand structure of N (no operator yet).
template<class Elt>
InducedOperator<Elt> Subquotient(const engine::ModCtx<Elt>& ctx, const engine::DenseBuf<Elt>& G,
                                 const engine::DenseBuf<Elt>& Rl);
// The matrix of the operator Theta (a cochain map of the degree) on it, in
// Subquotient's basis; Rl as passed to Subquotient.
template<class Elt>
engine::DenseBuf<Elt> OperatorMatrix(const engine::ModCtx<Elt>& ctx, const InducedOperator<Elt>& N,
                                     const engine::DenseBuf<Elt>& Rl, const engine::DenseBuf<Elt>& Theta);

// Monic characteristic polynomial det(x I - A), coefficients low -> high, of a
// t x t matrix over ctx (field or chain ring), DIVISION-FREE (sum of signed
// principal minors, each by the Leibniz expansion); t <= 7 (ENGINE_CHECKed).
// Over Z/ell^m it is the characteristic polynomial of the endomorphism of the
// FREE module (Z/ell^m)^t -- a basis-free invariant there.
template<class Elt>
std::vector<Elt> CharpolyDivisionFree(const engine::ModCtx<Elt>& ctx, const engine::DenseBuf<Elt>& A);

// The generators of IM(red_{K' -> K}) on Z^q: the Howell generators of the
// cocycles over Z/ell^K' reduced entrywise mod ell^K (the note above).  Both
// contexts are chain rings (or K' = K, a field allowed) of the same ell.
template<class Elt>
engine::DenseBuf<Elt> ReducedCocycles(const ModelCohomology<Elt>& big, uint32_t q, const engine::ModCtx<Elt>& small);

}  // namespace integral_model

#endif
