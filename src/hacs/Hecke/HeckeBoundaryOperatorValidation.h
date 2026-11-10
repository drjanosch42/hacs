// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_BOUNDARY_OPERATOR_VALIDATION_H
#define HECKE_BOUNDARY_OPERATOR_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 1, WP-4b -- validation of the boundary Hecke operator
// Theta^bd_g / T^bd_p (Hecke/HeckeBoundaryOperator) against
// docs/latex/gl2_contract.tex chapter sec:boundarydatum, cited by label:
// obl:taubdry(v)-(vii), def:bdryop / eq:bdrychain, prop:bdryop, prop:bdrychar
// (1)-(3), prop:bdrycosets(4) (the beta_j NEGATIVE control), prop:circlemodel
// (2)-(3) (the model recording), lem:ell2unbounded (what the circle model is
// NOT at ell = 2), cl:chi(2) (recorded), and sec:oraclesnew family (A).
//
// Independence discipline: every predicate is re-derived here from the
// contract and evaluated through public accessors only. Route B (the closed
// forms sum_delta rho_n(delta) on V^T and eq:bdrygroupform on V_T) is
// re-implemented in this file with its own class representatives / lifts;
// the invariant basis x^{n - b ell} Nm^b is built by our own homogeneous
// polynomial arithmetic in F_ell[x, y]; chi_b(p) by our own modular power;
// the bulk side of (vii) is the shipped HeckeOperatorOnK through the public
// restriction morphism EquivariantCochainComplexMorphismOnK (as
// HeckeIotaSignValidation's route (b) reads H^1(bd) off the shipped code).
// The operator's own compute()-time assertions are re-asserted to hold but
// never relied on for a verdict.
//
// Grid (uint32_t): (p, nu) in {(2,1), (3,1), (5,1), (7,1), (2,2), (3,2)} x
// ell in {2, 3, 5, 7, 11, 13, 101, 691, 134217757, 2^31 - 1} x
// n in {0, 2, ..., 16} (even; the bad cases p = ell and ell in {2, 3}
// included); uint64_t: ell = 2^61 - 1 x (p, nu) in {(2,1), (3,1)} x
// n in {2, 10, 16} (routes, oracles; no restriction -- the morphism class is
// uint32_t-only).
//
// Arms (GATING unless marked RECORDED; the return value is the number of
// failures):
//   (A) model recording: boundary_model = "circle",
//       boundary_is_group_cohomology = (ell != 2), q_max = 1 derived,
//       truncated = false, reported degrees {0, 1}; dim H^0 = dim H^1 =
//       dim V^T = floor(n / ell) + 1 = d_V - rank(rho_n(T) - I)
//       (prop:circlemodel(2), prop:bdrychar(1)) at EVERY ell; at ell = 2 the
//       dims are the circle-model values and NOT lem:ell2unbounded's
//       (2 dim V^T in degree >= 1; no degree 2 at all) -- asserted, and the
//       ell in {2, 3} cells are printed.
//   (B) route equality (obl:taubdry(v)): A_shapiro(q) == A(q) (nu = 1) and
//       A_groupform(q) == A(q) entrywise from the object, PLUS our own route B
//       (own M_0, M_1, own class representatives, own lifts, the stored P_q)
//       == A(q) entrywise; apply_direct(e_b) == column b of Theta_flat on the
//       full standard basis; the certificate bundle and the JSON agree.
//   (C) characters (obl:taubdry(vi), n even, p != ell, nu = 1): every
//       x^{n - b ell} Nm^b is rho_n(T)-invariant, they are a basis of V^T,
//       Theta^bd lift(w_b) = chi_b(p) lift(w_b) as cochains AND
//       A_bd^{(0)} [w_b] = chi_b(p) [w_b] -- ASSERTED; for n < ell,
//       A_bd^{(1)} [y^n] = (1 + p^{n+1}) [y^n] -- ASSERTED;
//       charpoly(A_bd^{(0)}) = prod_b (X - chi_b) -- ASSERTED; the general-n
//       H^1 multiset charpoly(A_bd^{(1)}) == prod_b (X - chi_b) (cl:chi(2))
//       -- RECORDED (count printed); the plan's two examples (6, 5, 2) ->
//       {4, 1} and (16, 7, 3) -> {6, 0, 1} asserted as arithmetic.
//   (D) the beta_j negative control (prop:bdrycosets(4), prop:bdrychar(2);
//       nu = 1, n even, p != ell): the naive right-coset sum acts on w_b by
//       chi_{b+1}(p) -- ASSERTED (it is a proved statement); on every cell
//       where the multisets {chi_b} and {chi_{b+1}} differ the control's H^0
//       matrix differs from A_bd^{(0)} -- ASSERTED, and at least one such
//       cell exists -- ASSERTED; on n < ell cells with a boundary-visible
//       bulk H^1 and chi_0 != chi_1, the right-coset scalar chi_1 BREAKS
//       r^1 A^{(1)} = A_bd^{(1)} r^1 -- ASSERTED (the control must fail).
//   (E) restriction equivariance (obl:taubdry(vii) = obl:coneexact(iii)):
//       r^q A^{(q)}(T_p) = A_bd^{(q)} r^q entrywise for q in {0, 1} (q = 2 is
//       vacuous, H^2(bd) = 0), r^q from EquivariantCochainComplexMorphismOnK
//       on the inclusion K_bd -> K with the identity fiber map, A^{(q)} from
//       HeckeOperatorOnK -- ASSERTED at ell >= 5 (nu in {1, 2}; p = ell
//       included), RECORDED at ell in {2, 3} (model objects; the tally is
//       printed).
// Prints one `BOUNDARY_OP cells=<k> checks=<n> failures=<f> ...` line.
uint32_t HeckeBoundaryOperatorValidation();

#endif // HECKE_BOUNDARY_OPERATOR_VALIDATION_H
