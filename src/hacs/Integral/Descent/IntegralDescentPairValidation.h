// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_PAIR_VALIDATION_H
#define INTEGRAL_DESCENT_PAIR_VALIDATION_H

// DESCENT milestone D4 (DESCENT_plan.md v0.4 Sec. 4 "D4"; [C] v1.23.0-pre.27 sec:descentpair and its code map (a)-(h)).
//
// IntegralDescentPairValidation (standard):
//   (T)  the oracle table docs/implementation/slot24/d4_pair_oracle.tsv (cd7_pair_check.py's model: the free tree resolution, the
//        minimal boundary resolution, the comparison solved over Z): H^q, H^q(Gamma_d; M_N), H^q_c and kernel / image / cokernel of
//        res^q, iota^q, pi^q, q <= 4, on every oracle cell, against AssembleDescentPair (Iso^(2), Iso^(3), the E8 pair at level 1,
//        the Bredon pair at level N);
//   (E)  exactness of eq:descentpairles at every reported spot as SUBGROUPS, on every piece (code map (e));
//   (B)  the boundary against the closed form from the cusp widths, |tors H^1(d)|_ell = prod_c n! w_c^n (code map (d)), and the
//        planted fault "the closed form without the widths";
//   (5)  prop:descentpair(5): H^0_c = 0, H^1_c torsion-free, rational ranks, 48 H^q_c = 0 (q >= 3), the 3-local identity, (5)(e);
//        the cone ranks also mod ell' on the same O_ell-model (code map (c));
//   (P)  q_max = 5: H^5_c = H^3_c, H^q(d) and H^q periodic (prop:descentpair(4), (5)(d));
//   (I)  model independence (lem:pairlifting(3)): the free Iso (model (a)) = Iso^(ell) (model (b)), and at level 1 the Bredon
//        model = the shipped E8 pair, on every group and map type;
//   (Q)  level 1, ell >= 5: the Z[1/6]-piece IS the shipped E8 item-4b pair after (x) Z/ell^K (prop:descentpair(1)(d)): eps^*
//        intertwines the coboundaries and the restriction and is onto, and the shipped ModuleCone's invariant factors of H^q_c are
//        the universal-coefficient image of the glued groups;
//   (S)  the strict datum: StrictLift::certify()'s boundary items for p in {2, 3, 5, 7, 11, 13} (tau_Iso(Isob) in Isob;
//        rmk:descentpairscope(iii)), with the lift's build / certify cost per p;
//   (3)  the (PC) arm of HeckeIsotropyModelEllValidation reproduced on D4's objects: Phi^*, Psi^* map the pair to the pair, the
//        groups of Iso^(3)'s pair equal Iso's, and T'_p = Psi^* T_p Phi^* equals T_p ON COHOMOLOGY (H^q_c and H^q(d), torsion
//        included), p in {2, 3, 5}, n in {0, 2, ..., 12}, mod 3^19;
//   (H)  Hecke on the pair (code map (f)): the pair lift's T~_Iso / T~_Isob, the strict identity entrywise, Phi_p a cochain map of
//        the cone, T~ at N = 1 equal to theta_hat; the characteristic polynomials of T_p on H^1_c and on H^1 equal the referee's
//        (tau(p) at n = 10, a_p(X_0(11)), Eisenstein 1 + p^{n+1}) mod 10007^2, 2^k and 3^k; T_p on H^0(d), H^1(d) is
//        (x - (1 + p^{n+1}))^{c(N)}; T_2 T_3 = T_3 T_2 on H^1_c;
//   (N)  negative controls, each REFUSED: a perturbed restriction entry; a false rank of res^1; K = 3 at 2; ell' = 5 at n = 8 (5 |
//        tors H_0) as a rank source (and ell' in {3, 5 <= n + 1} by rule); the RSS ceiling; q_max = 3; T_7 on the pair (level 11 and
//        level 1); T_2 at level 2 (p | N); the bulk-only lift on the cone.
// IntegralDescentPairSlowValidation (slow): the cost of the pair (DESCENT4_COST lines): the 2-adic piece on dense Iso^(2) and its
//   boundary alone on Isob along a ladder of level-N cells (DESCENT4_predictions.md, filed before the runs), the refusal above the
//   ceiling, and the level-1 pair to larger n.

#include <cstdint>

uint32_t IntegralDescentPairValidation();
uint32_t IntegralDescentPairSlowValidation();

#endif
