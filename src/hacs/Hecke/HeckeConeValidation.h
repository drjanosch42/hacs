// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_CONE_VALIDATION_H
#define HECKE_CONE_VALIDATION_H

// hecke-extensions slot 3, WP-5 (phase E5-2) -- validation of the equivariant
// mapping cone (EquivariantMappingCone, Hecke/SL2ZBoundaryPair) against
// docs/latex/gl2_contract.tex ([C], cited by label) chapter sec:cone:
// def:pair, eq:conediff, lem:conerelative(2)-(5), eq:coneles, eq:conedims,
// open:ell2 (riders (i)-(iii)), obl:coneexact(iv)-(v) and the Dcone^2 = 0
// half of (i).  The COMPLEX arms of [X] Sec. 5's HeckeConeValidation; the
// ACTION arms (Phi_g, h = eta~, obl:eta(v), obl:coneexact(i)-(iii), (vi),
// (viii)) are slot-3 WP-6's and extend this TU.  Registered `standard`.
//
// INDEPENDENCE.  Every predicate is re-derived here from the contract text
// through public accessors: an OWN dense cone differential assembled from
// the simplicial coboundaries and the vertex-list restriction (compared
// entrywise with the stored BlockMat), an OWN dense D^2, an OWN zig-zag for
// the connecting homomorphism of the short exact sequence 0 -> C_bd ->
// Cone -> C_eq[1] -> 0 (lift, apply D, read the boundary component, project)
// compared with the stored r^q AND with HeckeExt/HeckeBoundary::r(q) (a
// *Validation TU may include HeckeExt/), OWN ranks, OWN dim M_k / dim S_k.
//
// GATES (a failure of any is a returned failure count > 0):
//   G1  ConeCertificate::verify() == 0 on every cell (C1..C13 of the header).
//   G2  The stored D_c equals the own dense assembly of eq:conediff entrywise,
//       and the own dense D^2 = 0, in every built degree.
//   G3  Exactness at every asserted spot (every spot on the model complexes,
//       [C, obl:coneexact](iv)): composites zero and the rank identity, both
//       re-derived here from the stored maps; im(pi^1) = ker r^1 explicitly.
//   G4  The connecting map by the own zig-zag == the stored r^q == the
//       restriction morphism's induced map == HeckeBoundary::r(q), entrywise,
//       q in {0, 1} ([C, lem:conerelative](2): the connecting map is +r).
//   G5  Dimensions: H^0_c = 0 at EVERY ell and n ([C, lem:conerelative](4));
//       dim H^1_c = dim H^0(bd) - dim H^0_eq + dim ker r^1 at EVERY ell (finding
//       3 of the report: needs no H^2_eq = 0); dim H^2_c = dim H^1(bd) - rank r^1
//       where H^2_eq = 0; dim H^1_c - dim coker r^0 = dim H^1_! against
//       HeckeBoundary::dim_interior(); H^2_eq = 0 at ell >= 5 (asserted),
//       recorded at ell in {2, 3}.
//   G6  The measured (10, 691): dim H^1_c = 3 (slot-3 plan Sec. 2.3 item 3),
//       and the ten Kummer cells of [E] Claim 1.8: dim H^1_c = 1 + 2 dim S_{n+2},
//       dim H^2_c = 0, rank r^1 = 1 (WP-7's numbers reproduced through the cone).
//   G7  The record fields: bulk 2 / boundary 1 / cone 2, all derived, none
//       truncated, degrees [0..q_max] ([C, open:ell2] rider (i));
//       boundary_model = circle, boundary_is_group_cohomology = (ell != 2)
//       carried through to the cone's record; a set q_max on the finite model
//       rejected.
//   G8  The truncated branch of (iv), on a declared truncation of a finite
//       pair (the cylinder rel. both boundary circles, a test double for E9):
//       the cone truncated with q_max derived as min(q_max_K, q_max_bd + 1);
//       the spot at H^{q_max}(bd) NOT asserted, its rank r^{q_max} and coker
//       recorded; every other spot asserted and exact; the certificate green.
//   G9  Tamper arm ([C, obl:coneexact] "a failure aborts"): a corrupted entry
//       of a stored cone basis vector fails the basis items (C5, C6) and no
//       LES item; a corrupted entry of the stored r^1 fails the map item (C6)
//       and no basis item; the untampered certificate passes.
//   G10 The golden collectors' digest is equal across two in-process builds
//       and across two child processes at ENGINE_THREADS = 1 and 16 (the
//       slot-2 WP-7 pattern); the collectors are DEFINED, never emitted.
//   G11 ell in {2, 3}: every clause above asserted except where (v) says
//       "recorded" (H^2_eq), the model computes the orbifold-model relative
//       cohomology ([C, rmk:ell2window](1)) and is recorded, never refused.
//
// GRID: ell in {2, 3, 5, 7, 691, 134217757, 2^31-1} x even n <= 16, plus
// (20, 5), (10, 13) and the nine off-grid Kummer cells of [E] Claim 1.8
// ((10, 691) is on the grid); HeckeBoundary is built with (p, nu) in
// {(2, 1), (2, 2)} (diagnostics off) for the entrywise r^q cross-check and
// dim H^1_!.  Prints one `CONE cells=... checks=... failures=...` line.
//
// Environment: HECKE_CONE_CHILD=1 (the child mode of G10: prints
// `CONE_COLLECT digest=0x...` and exits); HECKE_CONE_LARGE=<n> (optional:
// builds one cone at (n, 134217757) and prints its wall time and VmHWM --
// the RSS record of [X] E5 (d), recorded, never asserted).

#include <cstdint>

uint32_t HeckeConeValidation();

#endif // HECKE_CONE_VALIDATION_H
