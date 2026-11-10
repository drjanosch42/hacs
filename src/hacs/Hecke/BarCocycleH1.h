// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_BAR_COCYCLE_H1_H
#define HECKE_BAR_COCYCLE_H1_H

// Stage S8 -- Independent H^1(Gamma; Sym^n F_ell^2) via group (bar) cocycles,
// with the Hecke action of the CORRECTED group-cochain formula
// [C, prop:barformula]. This is a from-scratch validation route
// ([C], contract box after prop:barformula: "a validation oracle only"); it
// never touches the tau-table, the induced systems or the simplicial model's
// cochain spaces.
//
// Contract: docs/latex/hecke_action.tex ([C]):
//   * [C, prop:barformula]: with LEFT cosets Gamma g Gamma =
//     sqcup_j delta_j Gamma, delta_j = s_j^{-1} g ([C, prop:cosets](2)),
//     permutation gamma * j defined by gamma delta_j Gamma =
//     delta_{gamma*j} Gamma and kappa_j(gamma) :=
//     delta_{gamma*j}^{-1} gamma delta_j in Gamma:
//       (T_g phi)(gamma) = sum_j rho_n(delta_{gamma*j}) phi(kappa_j(gamma)).
//   * [C, rmk:barformula]: the [v5] Sec. 12.3 formula (without the coset
//     permutation) is NOT well defined; this corrected form is.
//
// BINDING PRESENTATION CHOICE (plan stage S8): Gamma = SL(2,Z) is the
// amalgam <S> *_{<S^2>} <ST> = Z/4 *_{Z/2} Z/6 (S^4 = (ST)^6 = 1,
// S^2 = (ST)^3 = -I central). Writing U := ST, a 1-cocycle
// phi : Gamma -> V (phi(ab) = phi(a) + rho_n(a) phi(b)) is determined by the
// pair (phi(S), phi(U)) in V^2 subject to the LINEAR conditions
//   N_S phi(S) = 0,   N_U phi(U) = 0,       N_g := sum_{j < ord(g)} rho_n(g)^j,
//   (1 + rho_n(S)) phi(S) = (1 + rho_n(U) + rho_n(U)^2) phi(U)
// (both sides compute phi(S^2) = phi(U^3) -- the amalgam matching), and
// conversely every such pair extends uniquely to a cocycle on the amalgam.
// Coboundaries: phi_v = (rho_n(S) v - v, rho_n(U) v - v). Torsion
// denominators 4, 6 are invertible for ell >= 5 (hypothesis (H3)); the
// constructor ENFORCES ell >= 5.
//
// Dimension cross-check (Mayer-Vietoris for the amalgam, asserted at
// construction): H^1(Gamma; V) = V^{S^2} / (V^S + V^U) -- for n even
// V^{S^2} = V, giving the plan's V/(V^S + V^{ST}); for n odd V^{S^2} = 0.
//
// Evaluation of phi on arbitrary kappa in Gamma: reduce kappa to a word in
// T, S via Stage S2's word_in_T_S ((P2), Euclid on the first column;
// encoding: optional leading ('-',1) for the -I factor, then ('T', a) /
// ('S', 1) pairs in product order), then expand with the cocycle rule
// phi(ab) = phi(a) + rho_n(a) phi(b), using
//   phi(T)      = rho_n(S^{-1}) (phi(U) - phi(S))        (T = S^{-1} U),
//   phi(T^{-1}) = -rho_n(T^{-1}) phi(T),
//   phi(-I)     = phi(S^2) = (1 + rho_n(S)) phi(S).
// The evaluator asserts rho_acc == rho_n(kappa mod ell) at the end and that
// the word reconstructs kappa exactly, so a word-reduction slip fails
// loudly.
//
// Comparison with the pipeline (BINDING, plan stage S8): charpoly AND
// minpoly of hecke_matrix() must equal those of Stage S7's A^{(1)}(T_p) --
// basis-independent similarity invariants, no explicit intertwiner. The
// division-free local helpers below serve both sides.
//
// Placement: consumes ModArithmetic + the S2/S3 exact layers; compiled into
// cohomology_app (NOT into the pure-integer HeckeCore static library).

#include <cstdint>
#include <vector>

#include "ExactInt.h"
#include "HeckeCosetTables.h"
#include "../../ModArithmetic/FlatMatrix.h"

namespace hecke {

// charpoly(A) = det(X I - A) over F_{A.p}, division-free (cofactor expansion
// on polynomial entries -- A is a small Hecke matrix on H^1, dim <= 5 in all
// S8 settings). Coefficients LOW -> HIGH degree (amendment A-N4), monic.
// The 0 x 0 matrix yields {1}. Self-checks (throw std::logic_error): monic,
// degree == dim, coeff of X^{dim-1} == -trace, Cayley-Hamilton
// (charpoly(A) evaluated at A is the zero matrix).
std::vector<uint32_t> CharPolyOfMatrix(const FlatMatrix& A);

// Minimal polynomial of A over F_{A.p} via the Krylov/linear-dependency
// method on I, A, A^2, ...: the first k with A^k in span(A^0..A^{k-1}) gives
// minpoly = X^k - sum c_j X^j. Coefficients LOW -> HIGH, monic; {1} for the
// 0 x 0 matrix. Self-checks (throw std::logic_error): minpoly(A) == 0 and
// minpoly divides charpoly exactly (division by a monic polynomial needs no
// inversion beyond the leading 1).
std::vector<uint32_t> MinPolyOfMatrix(const FlatMatrix& A);

class BarCocycleH1 {
public:
    // ell >= 5 enforced (throws std::invalid_argument otherwise; hypothesis
    // (H3) -- the amalgam torsion orders 4, 6 must be invertible). Builds
    // rho_n(S), rho_n(ST) (invariant I2: ComputeSymmetricPowerMatrixFM),
    // the cocycle space Z, the coboundary space B, a basis of H^1 = Z/B,
    // and asserts the Mayer-Vietoris dimension cross-check.
    BarCocycleH1(uint32_t ell, uint32_t n_sym);

    uint32_t dim_H1() const { return dim_H1_; }

    // T_p on THIS object's H^1 basis via [C, prop:barformula], with
    // delta_j = cosets.delta(j) (the S3 exact tables; cosets must have been
    // built with the same (ell, n_sym) -- checked). The permutation
    // gamma * j and kappa_j(gamma) are computed EXACTLY over Z
    // (kappa = g^{-1} s_k gamma s_j^{-1} g via the integral adj(g)/N trick;
    // uniqueness of gamma * j asserted). Column convention:
    // T_p[phi_i] = sum_j M_{ji} [phi_j]. Returns the 0 x 0 matrix when
    // dim_H1() == 0. Every (T_g phi) pair is asserted to satisfy the cocycle
    // conditions before projection (a loud whole-route self-check).
    FlatMatrix hecke_matrix(const HeckeCosetTables& cosets) const;


private:
    uint32_t ell_;
    uint32_t n_sym_;
    uint32_t d_; // n_sym + 1
    uint32_t dim_H1_ = 0;

    FlatMatrix rho_S_, rho_U_, rho_T_;
    FlatMatrix rho_S_inv_, rho_T_inv_, rho_neg_I_;

    // Cocycle conditions: 3d x 2d constraint matrix (rows: N_S | 0, 0 | N_U,
    // (1+rho_S) | -(1+rho_U+rho_U^2)); Z = ker.
    FlatMatrix constraint_;
    std::vector<FlatMatrix> Z_basis_; // 2d x 1 columns
    std::vector<FlatMatrix> B_basis_; // independent coboundary generators
    std::vector<FlatMatrix> H_reps_;  // Z-vectors completing B to a basis of Z

    // rho_n(gamma mod ell) of an exact integral matrix (invariant I2).
    FlatMatrix rho_of_exact(const Mat22& gamma) const;

    // phi(kappa) for the cocycle determined by (x, y) = (phi(S), phi(U)),
    // kappa in Gamma exact (det +1). Word reduction via Stage S2 (see the
    // header note); returns a d x 1 column.
    FlatMatrix eval_phi(const FlatMatrix& x, const FlatMatrix& y, const Mat22& kappa) const;

    // Coordinates of the cocycle z (2d x 1, must lie in Z) w.r.t.
    // [H_reps_ | B_basis_]; returns the first dim_H1_ coordinates. Throws if
    // z is not in span (i.e. not a cocycle -- cannot happen if the caller's
    // constraint assert passed).
    std::vector<uint32_t> project_to_H(const FlatMatrix& z) const;
};

} // namespace hecke

#endif // HECKE_BAR_COCYCLE_H1_H
