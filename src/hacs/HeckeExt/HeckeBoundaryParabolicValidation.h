// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_BOUNDARY_PARABOLIC_VALIDATION_H
#define HECKE_EXT_BOUNDARY_PARABOLIC_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 2, WP-7 (phase E4b) -- the validation of
// HeckeExt/HeckeBoundary (H^1_!, the Eisenstein line, Harder's criterion, the
// Harder split) and of Hecke/HeckeBoundaryDiagnostics (cochain_commutes, the
// cusp-support walk / eta_zero). [C] = docs/latex/gl2_contract.tex, [E] =
// docs/latex/hecke_extensions.tex, cited by label.
//
// INDEPENDENCE. Every gating oracle is re-derived here without the unit:
//   * the restriction maps r^0, r^1 are rebuilt from the stored H_eq bases of
//     K by restricting each cochain to the boundary simplices by VERTEX-LIST
//     matching and projecting with the boundary P_q, and compared entrywise
//     with the object's r^q;
//   * H^1_! is recomputed as the FlatMatrix kernel of that own r^1 (a
//     different elimination path from MatrixT's) and compared by span;
//   * T_p-stability A B = B A_! is re-multiplied on FlatMatrix;
//   * the Eisenstein line is re-derived from the stored A^{(1)} by FlatMatrix
//     stacking, and compared with the object's line (rank 1 together);
//   * the Kummer side is pinned against [E] Claim 1.8's published list
//     (k, ell) -- ten pairs -- and against the factorisations 174611 = 283 *
//     617, 77683 = 131 * 593, 236364091 = 103 * 2294797 and trial-division
//     primality of 691, 3617, 43867, 657931, so the object's numerator table
//     is checked by an independent source and not by itself;
//   * d_k = dim M_k and the Dickson semigroup predicate are written here.
//
// GATING (ell >= 5 unless said otherwise):
//   G1  certificate().verify() == 0 on every cell (every ell);
//   G2  own r^q == r(q) entrywise, own ker r^1 == H^1_! as a span,
//       dim H^1_! + rank r^1 = dim H^1, r^0 injective (every ell);
//   G3  H^1_! is T_p-stable for every built (p, nu) -- (2,1), (3,1), (5,1),
//       (7,1), (2,2) -- with A B = B A_! entrywise;
//   G4  the joint 1 + p^{n+1} eigenspace E re-derived agrees with the object's
//       (same span, same rank(r^1 E)); it is a LINE off the Dickson and
//       Kummer loci (n > 0, k <= 26) -- at every Kummer cell it is
//       2-dimensional (the reduced Eisenstein class and the congruent
//       cuspidal eigenline are not separable mod ell, [E] D6), RECORDED;
//       where rank r^1 = 1 and n < ell the line's image is the
//       boundary-visible class (route (b) of obl:iota(v));
//   G5  HARDER ([E] cl:harder): on every cell with n < ell - 1 (k <= 26) the
//       verdict decided on E (HeckeBoundaryCertificate::HarderVerdict) is
//       never `refuted` -- the ten Kummer cells (10,691), (14,3617),
//       (16,43867), (18,283), (18,617), (20,131), (20,593), (22,103),
//       (22,2294797), (24,657931) are all built, and on all ten r^1 vanishes
//       on the whole 2-dimensional E (verdict `consistent`), as it does on
//       no non-Kummer cell (the "only if": rank(r^1 E) = 1 on the line);
//       (10, 5) and (10, 7) lie outside the hypothesis and are RECORDED;
//   G6  the Harder split: charpoly(A^{(1)}) = charpoly(A_!) * charpoly(A_bd |
//       im r^1) per (p, nu); at n < ell with rank r^1 = 1 the quotient
//       H^1 / H^1_! carries chi_0(p) = 1 + p^{n+1}; where the line is not in
//       H^1_!, H^1 = H^1_! (+) Eis;
//   G7  the classical dimension: for ell > n + 1 (n > 0, k <= 26), Kummer
//       cells INCLUDED, dim H^1_! = 2 dim S_{n+2} = 2 (d_{n+2} - 1) and
//       rank r^1 = 1 -- so H^1_! != H^1 at every Kummer cell, against [E]
//       sec:tests' "H^1_! = H^1 (all)"; at ell <= n + 1 the dimensions are
//       RECORDED, at the Dickson loci (20, 5), (24, 5) the plan's
//       "dim H^1_! >= 2 dim S_{n+2}" is RECORDED (it fails on 3 of 6);
//   G8  the diagnostics' one consistency theorem: eta_zero implies
//       cochain_commutes ([C, rmk:conestrict](2)); the p = 2 walk reproduces
//       the shape of the canonical tau's docs/implementation/tau_table_p2.txt
//       (10 of the 12 vertex slots and 5 of the 9 edge slots leave the
//       boundary; [C, rmk:conestrict](1)'s "8 / 2" is the pre-re-baseline
//       count) -- a regression gate on the construction, not a branch on the
//       verdict;
//   G9  the tamper arm: a certificate copy with a wrong interior vector, a
//       wrong A_!, a flipped Kummer flag, a flipped in_interior flag, a zero
//       Eisenstein vector or a wrong split flag fails verify();
//   G10 the golden collectors: 14 named records per (p, nu), the digest equal
//       across two independent builds in-process and across two child
//       processes run with ENGINE_THREADS=1 and ENGINE_THREADS=16;
//   G11 to_json() carries the per-complex q_max / truncated / q_max_source /
//       degrees (slot-2 owner confirmation 1) and the schema string.
// RECORDED (printed, never gating): the verdicts cochain_commutes / eta_zero
// per (cell, p) -- `BOUNDARY_DIAG cells=... cochain_commutes=k/m eta_zero=k/m`
// -- the ell in {2, 3} behaviour of every gate above, the H^1 multiset
// divisibility of [C, cl:chi](2), and whether H^1_! = H^1 at the Kummer cells
// ([E] sec:tests, VERIFY).
//
// Returns 0 on success, otherwise the number of failing checks.
uint32_t HeckeBoundaryParabolicValidation(void);

#endif // HECKE_EXT_BOUNDARY_PARABOLIC_VALIDATION_H
