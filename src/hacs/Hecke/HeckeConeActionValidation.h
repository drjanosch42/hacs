// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_CONE_ACTION_VALIDATION_H
#define HECKE_CONE_ACTION_VALIDATION_H

// hecke-extensions slot 3, WP-6 (phase E5-3) -- validation of the cone HECKE
// ACTION (Hecke/HeckeConeAction) against docs/latex/gl2_contract.tex ([C],
// cited by label) chapter sec:cone: def:coneaction / eq:derivedh /
// eq:conemap, lem:conesign, lem:derivedhomotopy, thm:conewelldef(a),(b),(d),
// prop:coneindep, prop:coneambiguity, obl:eta(v),(vi) and
// obl:coneexact(i)-(iii), (vi), (viii)(a),(b).  The COMPLEX arms (the cone
// itself, obl:coneexact(iv)-(v)) are slot-3 WP-5's HeckeConeValidation, which
// this validation deliberately leaves byte-unchanged.  Registered `standard`.
//
// INDEPENDENCE.  Every predicate is re-derived here through public accessors
// and this TU's own arithmetic: an OWN coset transport of the stored eta-table
// (its own rho_n((delta_j gamma_t) mod ell) products, its own accumulation
// loop) for eq:derivedh, an OWN dense assembly of the residual bracket
// (d h + h d + r Theta - Theta^bd r) of lem:conesign from the simplicial
// coboundaries and the vertex-list restriction, OWN ranks and kernels, an OWN
// rank-one degree-(-1) cochain map xi for the negative control, and the
// operators the pipeline already computes (HeckeOperatorOnK's A^{(q)},
// HeckeBoundaryOperator's A_bd^{(q)}, HeckeOperatorFamily's INDEPENDENT
// double-coset route T_p_pow_direct / T_p_pow_primitive_direct).
//
// GATES (a failure of any is a returned failure count > 0):
//   A1  ConeActionCertificate::verify() == 0 on every cell (items 1..12;
//       item 12 is [OD] 2026-09-22 Q2's record-honesty clause).
//   A2  obl:eta(v): the stored h^q equals the OWN coset transport of the
//       stored eta-table, entrywise, in every computed degree -- and h^0 = 0,
//       h^q = 0 wherever eta is not built.
//   A3  obl:coneexact(i): the residual bracket of lem:conesign, assembled
//       DENSELY here, annihilates the stored C^q_eq basis in every degree;
//       d h + h d == Theta^bd r - r Theta on C^q_eq; Dcone Phi == Phi Dcone on
//       the equivariant cone cochains; Dcone^2 = 0.  The contract's complex is
//       Ceq(K, Rel; L) ([C, def:pair]) and the coset-transport formula
//       [C, eq:cosettransport] uses phi's Gamma-equivariance, so the FULL FLAT
//       identity is RECORDED, not asserted (it holds in 29 of 424 degrees on
//       the grid).  See the HeckeConeAction.h note "WHY C^*_eq AND NOT THE
//       FULL FLAT SPACE" and finding 1 of the WP-6 report.
//   A4  Two routes: the fused / direct bracketings of h agree (the object
//       asserts it; re-checked here by an own third assembly), and
//       Phi_flat[:, b] == apply_Phi(q, e_b) == apply_Phi_direct(q, e_b) on the
//       whole standard basis of every degree.
//   A5  obl:coneexact(ii): Dcone Phi z = 0 on the stored cocycle basis and
//       A_c^{(q)} == P_c Phi H_c recomputed here.
//   A6  obl:coneexact(iii): pi A_c == A pi, r A == A_bd r (ASSERTED at
//       6 in R^x, RECORDED at ell in {2, 3} -- [C, rmk:conerestrictsmallell],
//       slot-3 audit F8), iota A_bd == A_c iota; and the bulk / boundary
//       operators are the pipeline's own (HeckeOperatorOnK /
//       HeckeBoundaryOperator, the same objects HeckeBoundaryOperatorValidation
//       uses).
//   A7  obl:coneexact(vi): [A_c(T_p), A_c(T_p')] = 0 for every pair of the
//       cell's primes, and the nu = 2 recursion A_c(T_{p^2}) = A_c(T_p)^2 -
//       p^{n+1} I with the FULL T_{p^2} = T_{Delta'_{p^2}} + p^n I
//       ([C, sec:primfull]; the primitive operator is what g = diag(1, p^2)
//       gives, and using it unchanged is the err:e1 slip) -- asserted at
//       ell >= 5, recorded at ell in {2, 3}.  The bulk analogue is checked
//       against HeckeOperatorFamily::T_p_pow_direct, the INDEPENDENT
//       double-coset route, and carried to the cone through pi.
//   A8  obl:coneexact(viii)(a), INDEPENDENCE: the four combinations of
//       EtaVertexPolicy::{CanonicalTreePath, LegacyBfsPath} x
//       TauBoundaryVertexPolicy::{NearestLineVertex, NextLineVertex} give
//       BIT-IDENTICAL A_c^{(q)} in every degree (asserted at EVERY ell, since
//       [C, prop:coneindep](1),(2) hold over Z), while the eta- and
//       tau_bd-table hashes MOVE and h_flat / Theta^bd_flat move with them.
//       NON-VACUITY (slot-3 audit): the arm ASSERTS that the two eta policies
//       give different eta hashes and that the two tau_bd policies give
//       different tau_bd hashes, on every cell it uses; a policy that left a
//       table byte-identical would make the arm prove nothing.  The number of
//       nearest-vertex TIES in the canonical tau_bd construction is printed as
//       well (189 over the grid), so the alternative phrasing of the second
//       policy as a tie rule is measured rather than assumed.  The cone's own
//       bases and Dcone are bit-identical by construction (the cone carries no
//       Hecke datum) and are re-checked through the cone's golden digest.
//   A9  obl:coneexact(viii)(b), the NEGATIVE CONTROL: h' = h + xi with xi a
//       degree-(-1) cochain map built as in [C, prop:coneambiguity](2) (here
//       the rank-one xi^1 = v_0 u^T, v_0 a stored H^0(bd) cocycle and u a
//       functional vanishing on the EQUIVARIANT coboundaries B^1_eq = d C^0_eq
//       -- so d xi = 0 and xi d = 0 on the contract's complex, and xi_* is the
//       lambda of the proposition; annihilating the FLAT B^1 is impossible on
//       this model, K being contractible) must PASS certificate items
//       2..11 (i.e. obl:coneexact(i)-(vi)) and FAIL item 1 (obl:eta(v)), and
//       must CHANGE A_c^{(1)} at (10, 691), where
//       Hom(ker r^1, coker r^0) is 2-dimensional ([C, rmk:coneambiguity691] as
//       measured by slot-2 WP-7 / slot-3 WP-5).  The arm's failure to behave so
//       fails the validation.
//   A10 Tamper arm: one entry of the stored h fails item 1; one entry of the
//       stored A_c fails item 6.  (A single-entry tamper of h also breaks the
//       cone identity, so it fails item 2 as well -- the arm that fails
//       EXACTLY obl:eta(v) is A9's xi, which is what [C, obl:eta](v) names.)
//   A11 obl:eta(vi): eta_zero => cochain_commutes asserted; both recorded and
//       found false on every cell (the shipped tau is not a map of pairs,
//       [C, rmk:conestrict](1)).
//   A12 ell in {2, 3}: every clause asserted except (iii)'s r-half and (vi),
//       which are RECORDED per [C, rmk:conerestrictsmallell] /
//       [C, rmk:conetorsion]; the recorded outcomes are printed.  The record
//       (to_json, schema hecke_cone_action/v1) carries boundary_model = circle,
//       boundary_is_group_cohomology = (ell != 2), the per-complex degree
//       fields and h's SOURCE-complex fields ([C, sec:recordschema] item 5).
//   A13 The golden collectors (charpoly / rank / trace of A_c^{(q)} tier A;
//       the raw A_c^{(q)} -- tier B since slot-6 WP-5's relabel, the tier the
//       harness always published it under --, h_flat, Theta^bd_flat and the
//       three table hashes tier B) are DEFINED and their digest is stable
//       in-process.  (Since slot-5 WP-3 the golden harness emits them on the
//       ext subgrid; golden_digest() itself is printed nowhere.)
//   A14 (hecke-extensions slot 6, WP-4) obl:coneexact(vii), FIRST SENTENCE --
//       the STRICT-PAIR cross-route of [C, prop:coneroutes](1) (owner decision
//       OP3). For EVERY operator of the grid, hecke::HeckeConeStrictPairRoute
//       builds tau^s (TauVertexPolicy::BoundaryStrict, [C, prop:strictpair]:
//       its own certified integer table with its own fill search, int64 then
//       the widened tier, never mod ell), eta^s on (tau^s, tau_bd) -- which
//       must be IDENTICALLY ZERO --, Theta^s, and Phi^s = (Theta^s, Theta^bd)
//       on the same cone, and compares A_c^{(q)}(Phi^s) with the normative
//       A_c^{(q)} ENTRYWISE in every degree. At 6 in R^x (ell >= 5) a
//       disagreement or a build failure ABORTS (the route throws; the arm
//       fails); at ell in {2, 3} both are RECORDED and printed. The arm also
//       asserts: the plain constructor refuses BoundaryStrict; tau^s
//       re-certifies (clauses (a)-(g)); solve_regime "Z"; step (B) filled
//       exactly m * |K_bd| slots; h^s == 0 in every degree; and NON-VACUITY --
//       tau^s differs from the shipped tau wherever the shipped eta is
//       non-zero (i.e. the shipped tau is not itself a strict datum). The arm
//       has its own check log and prints its own lines (a `strict pair:`
//       accounting line and one `CONE_STRICT_PAIR ...` summary line after the
//       `CONE_ACTION` line), so every pre-existing printed line -- including
//       the CONE_ACTION check count -- is unchanged; its failures are added to
//       the validation's return value.
//   A15 (hecke-extensions slot 6, WP-5) obl:coneexact(vii), SECOND SENTENCE
//       -- ROUTE 3, the relative-complex cross-route of
//       [C, prop:coneroutes](2) (owner decision OP3; no hypothesis on ell).
//       For EVERY operator of the grid, hecke::HeckeConeRelativeRoute builds
//       ext_0 from the stored r_flat, E = ext_0 h, Theta' = Theta + dE + Ed
//       ([C, eq:route3]), ker r as the shipped EquivariantCohomologyOnK with
//       the seven boundary simplices forced to zero (the additive relative
//       sub-complex constructor), A' on H^*(ker r), and the correspondence
//       M : H^q(ker r) -> H^q_c, c |-> (c, 0), and compares M A' M^{-1} with
//       the normative A_c^{(q)} ENTRYWISE in every degree; a disagreement or
//       any failing route clause (R3.1-R3.6 of HeckeConeAction.h) ABORTS at
//       EVERY ell (the route throws; the arm fails).  Re-derived here: ext_0
//       as the transpose of the OWN vertex-list restriction, E, Theta' from
//       the OWN coboundaries, the intertwining A_c M == M A', H^0(ker r) = 0,
//       seven forced simplices.  NON-VACUITY: Theta' != Theta on C_eq wherever
//       the shipped Theta is not a map of pairs (every operator), and the
//       UNCORRECTED Theta is counted where it moves an H(ker r) cocycle off
//       ker r.  On three cells the relative sub-complex's LEGACY arm
//       (ScopedLegacyCohomologyPath) is compared with its block arm, C/Z/B/H/P
//       entrywise.  And the A_c^q relabel ([WP5-3] finding 5): every A_c^q
//       golden object carries tier 'B' in the unit itself.  Own check log,
//       own printed lines (a `route 3 (...)` accounting line after the strict
//       pair lines, one `CONE_ROUTE3 ...` summary line after
//       `CONE_STRICT_PAIR`); every pre-existing printed line is unchanged.
//
// GRID: ell in {2, 3, 5, 7, 691, 134217757, 2^31-1} x even n <= 16 with
// (p, nu) = (2, 1) and (3, 1); a sub-grid with (5, 1), (2, 2) and (3, 2);
// p = ell (U_ell) at the small ells only (the exact-integer coset reach of
// docs/KNOWN_ISSUES.md Sec. 3 is slot-3 WP-1's, not this package's); and the
// distinguished cell (10, 691) for the independence and negative-control arms.
// Prints one `CONE_ACTION cells=... checks=... failures=...` line (and, since
// slot-6 WP-4, one `CONE_STRICT_PAIR ...` line for A14; since slot-6 WP-5,
// one `CONE_ROUTE3 ...` line for A15).
//
// Environment: HECKE_CONE_ACTION_LARGE=<n> (optional: builds one cone action
// at (n, 134217757, p = 2) and prints its wall time and VmHWM -- recorded,
// never asserted).

#include <cstdint>

uint32_t HeckeConeActionValidation();

#endif // HECKE_CONE_ACTION_VALIDATION_H
