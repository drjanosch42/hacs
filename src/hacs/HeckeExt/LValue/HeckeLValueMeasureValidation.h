// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LVALUE_MEASURE_VALIDATION_H
#define HECKE_LVALUE_MEASURE_VALIDATION_H
// (hecke-extensions slot 20, WP-MEAS) HeckeLValueMeasureValidation (standard): the measure-valued evaluation of the linear
// route ([E] 1.20.0-pre.1 sec:lvmeasure, gate chk:lvmeas) -- the ell-stabilised Mazur-Tate-Teitelbaum measure of an ordinary
// eigensymbol (ModularSymbolMeasure.{h,cpp}), against R20-1's frozen values (MeasureFrozenR201.inc) and in-process oracles:
//   (M0) the route on the C++ module cone at (30, 37) over Z/37^4: measure() == the symbol-side measure, linearity, the
//        bilinear / congruence routes refuse, (10, 5) refuses "not ordinary", a non-eigenclass refuses;
//   (M1) prop:lvmtt (PROVED): three-term relation, distribution, parity, the moment identity against WP-LIN's EvAllJ, the
//        Riemann-sum moment == the group-ring branch value;
//   (M2) R20-1's frozen a_p, alpha, s, mu(2 + ell^2), FNV of the level-2 unit balls;
//   (M3) cl:lvmeascong(1)(2), rmk:lvmeasplus(i)(ii) at (30,37), (282,37), (226,103);
//   (M4) Kubota-Leopoldt: the E_{1,c} and the Coates-Wiles-layer group rings agree (Q7) and give -(1-ell^{k-1})B_k/k; the
//        congruence of measures at the critical moments and as group-ring elements; the located pole-branch points;
//   (M5) the (mu, lambda) tables; (M6) negative controls;
//   (M7) (slot-20 S1, [E] sec:lvlfunction) the record lfunction/v1: values, truncation against level M-1 (lem:lvlfunprec),
//        (mu, lambda) from the coefficients, determinism.
// Prints MEAS_CELL / MEAS_ROUTE / MEAS_ID / MEAS_EIS / MEAS_KL / MEAS_INV / MEAS_NEG / MEAS_LFUN lines and MEAS_ARMS checks=<c> failing=<f>.
// Env (read-only): HECKE_LVALUE_MEASURE_HEAVY=1 adds (226,103) at level 3, (282,37) at R = 5 with every character, and
// (30,37) with every character (EFFECTIVE HeckeLValueMeasureValidation arm=HEAVY cells=3).  (slot-20 S2) HECKE_LVALUE_MEASURE_691=1
// adds the 691 arm: Delta at (10, 691), R = 2, level 2, against Kubota-Leopoldt (M1, M3-M7 as at the Eisenstein cells, d = 1; a_p ==
// tau(p) mod 691^2; the oracle's values, MeasureFrozenS2.inc) (EFFECTIVE HeckeLValueMeasureValidation arm=ELL691 cells=1).
// Writes nothing; no golden object.
#include <cstdint>

uint32_t HeckeLValueMeasureValidation();

#endif // HECKE_LVALUE_MEASURE_VALIDATION_H
