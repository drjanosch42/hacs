// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LVALUE_EISENSTEIN_MEASURE_VALIDATION_H
#define HECKE_LVALUE_EISENSTEIN_MEASURE_VALIDATION_H
// (hecke-extensions slot 21, P2) HeckeLValueEisensteinMeasureValidation (standard): the Eisenstein measure exactly in Lambda --
// EisensteinMeasure.{h,cpp}, spec docs/implementation/slot21/P2_spec.md (predictions committed first), Python oracle
// docs/implementation/slot21/p2/run_p2.py.  Arms:
//   (E0) Zagier's n! p_k^- satisfies 1+S, 1+U+U^2 as a Laurent polynomial (mod a 61-bit prime); the interior P^Eis alone does not;
//   (E1) the Manin-path symbol of n! p_k^- (singular monomial dropped) = -D(a,m) + (B_k/k) a/ell^m at every residue (Dedekind-Rademacher);
//   (E2) mu^Eis = -D(a,m) + ell^{k-2} D(a,m-1) is a distribution over Q and odd; NEG: the cusp term kept, WP-MEAS's mu^poly;
//   (E3) on units mu^Eis = -E_1 * iota E_{k-1} (the convolution);
//   (E4) sum_a omega^i(a) mu^Eis(a + ell^M) = -B_{1,omega^i} B_{k-1,omega^-i} / (k-1) exactly (TwistedBernoulliTable);
//   (E5) regular odd branches = -G_{a+1}(u(1+T)-1) G_{k-1-a}(u^{k-1}(1+T)^{-1}-1) (both KL routes), even branches 0;
//   (E6) pole branches: valuation d - M, minus the explicit polar term integral;
//   (E7) WP-MEAS's polynomial-symbol measure vs mu^Eis (- polar): depth exactly d;
//   (E8) the cusp forms at (30,37), (282,37), (226,103): the pole branches at every critical point with the polar term;
//   (E9) negative controls (no polar term, Res (1+ell), -Res, the wrong twist).
// Prints EISMEAS / EISMEAS_POLE lines and P2_ARMS checks=<c> failing=<f>.  Env (read-only): HECKE_LVALUE_EISMEAS_HEAVY=1 adds (30,37) at
// level 3 and (20,131) at level 2 (EFFECTIVE HeckeLValueEisensteinMeasureValidation arm=HEAVY cells=15; else arm=DEFAULT cells=13).
// Default arm also (P2x, spec Sec. 10) (E8L2)/(E9L2): Delta's own eigensymbol at (10,691),
// R = 2, level 2: (a) -s (F1 F2 [+ Res/(u^r-1)]) mod 691 at every critical point of every odd branch; (b) g^Delta = s (g^Eis - polar) in
// Z/691[C_691] on every odd branch (g^Eis by the E3 convolution through the branch homomorphism), E6 at level 2, NEGs; line EISMEAS_DELTA2.
// Writes nothing; no golden object.
#include <cstdint>

uint32_t HeckeLValueEisensteinMeasureValidation();

#endif // HECKE_LVALUE_EISENSTEIN_MEASURE_VALIDATION_H
