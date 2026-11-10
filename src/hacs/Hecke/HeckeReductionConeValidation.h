// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_REDUCTION_CONE_VALIDATION_H
#define HECKE_REDUCTION_CONE_VALIDATION_H

// ===========================================================================
// (hecke-extensions slot 9, WP-4; E8 item 4b) The boundary and cone arms over
// Z/ell^k.  Binding: slot-9 plan WP-4 scope 1-6, [E] v1.10.0-pre.1 sec:ring
// (f) (the objects, the certificates, the predicted values), (b1) (the closure
// rows), decision 3 (iii) (A) (route 1 only).
//
// Per cell (n, ell) and level K, for every Hecke prime p of the cell:
//   (4b-B)  the boundary over Z/ell^K: the two invariant-factor lists of H^0
//           and H^1 equal (universal coefficients on rho_n(T) - 1; hence the
//           lengths too), the lists against [E] (f)'s predicted table, and
//           bdry_bockstein_rank^0 == #{1 in bdry_invariant_factors_H^1} (from the
//           cell's Z/ell^2 boundary), with the coefficient-LES length bookkeeping;
//   (4b-C)  the cone's dense module route: ConeModuleCertificate::verify() == 0
//           once per level (C11 + C14; with H^0_c = 0, H^2 = 0 it forces the
//           image-length pattern (0, L0, B - L0, L1 - B + c2, B - c2, c2, 0, 0)
//           and l(H^1_c) = L1 - L0 + c2, [E] (f)'s consequence of exactness --
//           slot-9 WP-4t dropped those two rows and the coneexact_iv_certified
//           flag read, which ModuleConeT::compute() enforces by throwing),
//           H^0_c = 0, H^2 = 0, and [E] (f)'s complete rows at (10, 691) and
//           (16, 5); a tampered witness must fail item 14 (negative control, once);
//   (4b-H)  the Hecke action over the ring, route 1: T_p^bd (both bracketings,
//           (i)-(iv)) and the cone action (h in both bracketings, items 2, 4-9;
//           asserted by ModuleConeActionT::compute(), which throws -- WP-4t
//           dropped the routes_bit_identical flag read);
//   (4b-R)  the reductions to F_ell as MAP EQUALITIES ([X] F3): bdry_red_H^q
//           and cone_red_Hc^q are class maps; A^bd o red = red o A^bd,
//           A_c o red = red o A_c; every map of eq:coneles commutes with red;
//           beta o A = A o beta for the boundary Bockstein and the cone's
//           beta^1_c (validated, not an object); between levels K+1 -> K the
//           reductions are equivariant and functorial;
//   (4b-K)  (uint32_t cells) the CLASS route -- SL2ZBoundaryPair ->
//           EquivariantMappingCone -> HeckeConeAction -- agrees with the direct
//           template route on every length, invariant-factor list, image length
//           and A_c, and the pair's restriction equals RestrictionFlatRing;
//   (X4b)   the refusal closure moved by 4b: every member that now honours is
//           tested both ways (honours at ell >= 5; refuses at ell = 3 by name,
//           and on a ring mismatch), and the members that keep refusing (the
//           strict-pair and relative cross-routes on a composite normative
//           action, the FiberBlockTable morphism constructor, HeckeExt/
//           HeckeBoundary, cochain_commutes -- now reachable) each have a row.
//
// HeckeReductionConeArms (from HeckeReductionValidation, `standard`): the E8
// points but (60, 5) ((10, 691) at K = 2; (6, 5), (16, 5) at K = 2; (30, 5) at
// K = 2, 3; p in {2, 3} at ell = 5, {2, 3, 5} at 691), the Montgomery cell
// (10, 134217757) and the U64_Wide cell (10, 4294967291) over Z/ell^2 (p = 2;
// the F_ell cone class is uint32_t-only, so there the cone reductions are not
// run), and (X4b).
// HeckeReductionConeCoverageArms (from HeckeReductionCoverageSlowValidation,
// `slow`): first the E8 point (60, 5) at K = 2, 3, p in {2, 3}, with every arm
// above incl. (4b-K) (moved from the standard tier by slot-9 WP-4t, unchanged in
// content); then (4b-B), (4b-C), (4b-H), (4b-R) at K = 2, p = 2, on the E2 case
// list at ell >= 5 (the same 52 cells, the two large ones under the same interlock).
// ===========================================================================

#include <cstdint>

void HeckeReductionConeArms(uint64_t& checks, uint32_t& failures);
void HeckeReductionConeCoverageArms(uint64_t& checks, uint32_t& failures, bool large);

#endif // HECKE_REDUCTION_CONE_VALIDATION_H
