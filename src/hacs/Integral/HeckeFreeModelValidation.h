// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_FREE_MODEL_VALIDATION_H
#define HECKE_FREE_MODEL_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 9, WP-11 (Line 2, M0): the integral groups
// H^q(Gamma; Sym^n Z^2), n = 0 .. 20, q <= 5, assembled and compared with
// Appendix A of the integral-line brief EXACTLY (rank and elementary
// divisors; oracle O1).  [C] = docs/latex/gl2_contract.tex (v1.8.1), cited by
// label; [IB] = docs/implementation/integral_line_design_brief.md (its §6.2
// M0 items 1-8, read with [C]'s corrections).
//
// This name is [X]'s E9 validation, re-hosted in line 2 on the unrefined
// Xbulk ([IB] OD-8 (a)): tests (1), (2) and the negative control (6) live
// here; tests (3), (4), (5) belong to M1(a) -- concretely (slot 10, WP-8,
// m-16), test (3) is WP-13's HeckePeriodicityClassValidation and tests (4),
// (5) are HeckeIntegralActionValidation's arms (3c) and (3d).
//
// The assembly (arm (M), [C, def:ringsemantics](3)(a)-(b)):
//   * free ranks: rank_Q d^q of C_I(Z) at ONE good prime (10007),
//     [C, prop:certifiede](4); cross-checked against the shipped 7-vertex
//     pipeline's dim H^q over F_10007;
//   * 2- and 3-primary parts: sigma^{<= 6} C_I (x) Z/ell^K, ell in {2, 3}, K
//     certified PER DIFFERENTIAL by [C, prop:certifiede](3), (5) -- the rule
//     applied to the integral complex C_I, never to the Bredon object;
//   * ell >= 5 parts: C_7 (x) Z/ell^K at each Dickson prime of
//     [C, prop:badprimes](1) through the shipped pipeline's certified K-mode
//     (ladic_type; C_eq(Z/ell^K) = C_eq(Z) (x) Z/ell^K, [C, prop:ringinforce](1));
//     cross-checked against C_I at the same prime, and no ell-torsion at the
//     other primes 5 <= ell <= 23 (rank of every d^q over F_ell = rank_Q).
// Arms:
//   (M)  the assembled groups == Appendix A, every n <= 20, q <= 5 (O1);
//        the UCT prediction of [C, prop:certifiede](3) at f = K + 1 against
//        H^q(C_I (x) Z/ell^f) computed afresh (a held-out level);
//   (O2) n = 0: Z, 0, Z/12, 0, Z/12, 0;
//   (X3) eps^* (x) Z/ell^K an isomorphism on H^q, q <= 5, at every Dickson
//        cell: the mapping cone of eps^* : C_eq(Z/ell^K) -> C_I(Z/ell^K) has
//        zero cohomology through degree 5 (lengths over Z/ell^K);
//   (O9) H^2 ~= H^4, H^3 ~= H^5 (and tors H^6 = tors H^4) on the integral
//        groups ([C, lem:bulkperiodic]);
//   (O8) odd n: C_{7,Z} = 0 (C_eq has rank 0 at a good prime), the group
//        object from C_I is non-zero 2-primary torsion, and C_7 (x) F_2 != 0 is
//        the Bredon object -- each labelled;
//   (T1) [X] test (1): dim H^q(C_I (x) F_2) = dim H^q(C_I (x) F_3) = 1,
//        q = 0 .. 5, at n = 0;
//   (T2) [X] test (2): dim H^q = dim H^{q+2}, 2 <= q <= q_max - 2, at
//        ell in {2, 3}, every n <= 20;
//   (T6) [X] test (6), the negative control: the harness refuses (T1)/(T2)
//        on the shipped 7-vertex model (truncated = false) and at q_max < 4,
//        derived from the object's OWN degree count (num_degrees), not from
//        the caller's dispatch label (hecke-extensions slot 10, WP-8, m-6);
//        and asserts nothing there;
//   (R)  M0 items 3 and 7 (O7, the pipeline over Z/ell^K at 2 and 3; O5,
//        char_Z(T_2 | Lambda^1)) are HeckeBredonTreeRingValidation's and
//        HeckeIntegralCharpolyValidation's own independently registered
//        --list lines, cited rather than re-run in process (slot 10, WP-8,
//        m-7: the re-run duplicated both and silently coupled this
//        standard-tier name to HECKE_INTEGRAL_CAMPAIGN=1's ~30 s).
// Writes nothing; no golden object, no record key.  Returns the number of
// failed checks.
uint32_t HeckeFreeModelValidation();

#endif
