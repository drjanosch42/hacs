// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_BOUNDARY_TAU_VALIDATION_H
#define HECKE_BOUNDARY_TAU_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 1, WP-4a -- validation of the certified Gamma-level
// boundary datum tau_bd (Hecke/HeckeBoundaryDatum) against
// docs/latex/gl2_contract.tex chapter sec:boundarydatum, cited by label:
// obl:taubdry(i)-(iv), def:taubdry (T1bd)-(T3bd), lem:bdrycomponent,
// lem:taubdryexists(1)-(2), prop:bdrycosets(1)-(3), def:bdrygroup, and the
// boundary subcomplex hypotheses (B1)-(B3).
//
// Independence discipline: every predicate is RE-DERIVED here from the
// contract and evaluated through the PUBLIC accessors only -- the table's
// tau() / tau_on_translate() / boundary_cosets() / gamma_j() /
// slot_component(), SL2ZTessellationTables::canonical_key / transporter /
// boundary_simplices, HeckeCosetTables::s / s_inv / delta / class_index_of /
// perm / cocycle, and hecke::Mat22 / QuadPoint. The cusp arithmetic (P^1(Q)
// normalisation, the action of g^{-1}), the boundary operator, the (T1bd)
// translation rule g^{-1} h g = [[h.a, N h.b],[h.c / N, h.d]] and the
// left-coset canonical form are all re-implemented in this file; the table's
// own certify() is re-asserted to be 0 but never relied on for a verdict.
//
// Arms (all GATING; the returned value is the number of failures):
//   (A) certification grid, uint32_t: (p, nu) in {(2,1), (3,1), (5,1), (7,1),
//       (11,1), (2,2), (3,2)} x ell in {2, 3, 5, 7, 691, 134217757} x
//       n in {0, 2, 10} -- the bad cases p = ell and ell in {2, 3} included
//       (the table carries no hypothesis on ell); uint64_t: ell in
//       {2^61 - 1, 2^32 + 15} x (p, nu) in {(2,1), (3,1), (2,2)} x n in {0, 2}.
//       On every table: obl:taubdry(i) (single component; component map
//       == c -> g^{-1} c from g alone; degree 0 one simplex of coefficient 1),
//       (ii) (T2bd) through OUR boundary operator and OUR (T1bd) translation,
//       (iii) (T1bd) bookkeeping across the model's own relation i(v3) =
//       T i(v0) via HeckeCosetTables::perm/cocycle and on 40 fixed-seed random
//       words gamma per slot (also cross-checking tau_on_translate against
//       OUR translation), (iv) every coefficient in {+-1}, solve regime "Z",
//       integral = true; plus canonicity, det = 1, boundary support, the
//       certified bound by recount, and prop:bdrycosets(1)-(3): the m
//       canonical representatives, the nu = 1 table delta_inf = diag(1, p),
//       delta_t = [[p, t],[0, 1]] with the three matrix identities of (1),
//       the bijection j -> delta_j gamma_j Gamma_inf of (2), and the two
//       double cosets of (3) (T permutes the u = 1 cosets in one p-cycle and
//       fixes the u = 0 coset).
//   (B) ell-, n- and width-independence: for each (p, nu) the chains are
//       IDENTICAL (coeff, gamma, sigma, slot by slot) across every ell, n and
//       both element widths -- lem:taubdryexists(2): the table is a datum
//       over Z.
//   (C) tampered chains are REJECTED: certify_with_override with a negated
//       coefficient, a dropped term, a duplicated term, a vertex moved along
//       its line (T-shift), a vertex moved to another component, a
//       coefficient 2, and a two-term degree-0 chain, all return nonzero
//       codes, while the identical replacement returns 0.
//   (D) the boundary subcomplex (B1)-(B3) of SL2ZTessellationTables:
//       validate_boundary() == 0, the simplex lists are the horocycle path
//       v0-v1-v2-v3 and its three edges, the line coordinates and the
//       nearest-vertex rule at hand-computed points, and CuspKey::
//       reduce_to_infinity / apply against OUR cusp arithmetic.
// Prints the p = 2, nu = 1 table and one
// `BOUNDARY_TAU tables=<k> checks=<n> failures=<f>` line.
uint32_t HeckeBoundaryTauValidation();

#endif // HECKE_BOUNDARY_TAU_VALIDATION_H
