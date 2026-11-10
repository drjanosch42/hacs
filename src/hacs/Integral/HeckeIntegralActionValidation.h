// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_INTEGRAL_ACTION_VALIDATION_H
#define HECKE_INTEGRAL_ACTION_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 10, WP-11 (Line 2, M1(a): the integral Hecke action,
// basis-free; [IB] Sec. 6.2 M1(a) items 1-5, tier A).  T_p is the coset
// transport Theta^_g of the strict lift tau_Iso (Integral/Model/StrictLift,
// WP-10) on the isotropy-resolution model C_I, which induces
// cor o conj_g o res on H^.(Gamma; V_R) for every ring R and every degree
// ([C, prop:strictlift](3)); E_q = e u - is PeriodicityClass::matrix (WP-13;
// [C, lem:bulkperiodic]).  Every operator statement is read on cohomology
// through Integral/Model/IntegralHecke (an inclusion P(Z^q) in B^q, a length,
// a cyclic decomposition), never in an integral basis.  Arms, at
// n in {10, 12, 16}:
//   (P)  preconditions: the three lifts' models are one complex; the lifts'
//        and e's certificates hold (WP-10's (T1)/(S) carry "strictly
//        eps^*-compatible" over Z/ell^K at 2 and 3 -- CITED, not re-checked;
//        plan B-1 / decision 4: no Theta^s_g on the Bredon object is built);
//   (1)  O6 at n = 10: (T_2 - 2049)(T_2 + 24) = 0 and 3(T_3 - 252) =
//        256(T_2 + 24) on H^1(C_I (x) F_ell) = Lambda^1 (x) F_ell at good
//        primes, prime by prime; the O5 minimal identities at n = 12, 16; a
//        negative control with a wrong constant;
//   (2)  the 691-congruence module at n = 10 is Z/691: on line 1 (the shipped
//        E8 route, (10, 691) over Z/691^2 and Z/691^3, A^{(1)}(T_2), A^{(1)}(T_3))
//        and on line 2 (Theta^ on C_I (x) Z/691^k), by two readings each
//        (through T - b and through T - a);
//   (3)  T_2, T_3 (and T_5) on the 2- and 3-primary torsion in degrees <= 5:
//        X7 (T_p T_p' = T_p' T_p on H^q(C_I (x) Z/ell^K) and on F_ell);
//        the torsion modules T^q_ell read twice (as IM(red) in degree q and
//        as the Tor quotient of degree q - 1), their exponents equal to M0's
//        groups, the two readings' T_p mod ell equal; X6 (Theta^ E_q = E_q
//        Theta^ on cohomology, q = 0..3; HeckeFreeModelValidation's test (4),
//        hosted here) with its char-poly form; test (5) (charpoly A^{(3)}(T_p)
//        | charpoly A^{(1)}(T_p) over F_ell); negative controls;
//   (4)  the local type of Lambda^1 (x) Z_ell as a Z_ell[T_2]-module at
//        ell in {2, 3}: Lambda^1 (x) Z/ell^m read off the model, its
//        characteristic polynomials against O5 mod ell^m, O6 there, and the
//        Eisenstein-cusp congruence index (recorded; no further oracle);
//   (5)  X1 (a CONSISTENCY check, [IB] K6): the UCT prediction from M0's
//        groups against H^q(C_I (x) Z/ell^k), k < K_run, at 2 and 3 and the
//        Dickson primes, and against line 1's pipeline over Z/ell^k at the
//        Dickson primes; the Lambda^1 characteristic polynomial at two
//        precisions.
// Returns the number of failed checks.  `standard`.  Writes nothing.
uint32_t HeckeIntegralActionValidation();

#endif
