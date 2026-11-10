// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_ALL_DEGREES_VALIDATION_H
#define ORBIFOLD_MP_ALL_DEGREES_VALIDATION_H

#include <cstdint>

// ===========================================================================
// (hecke-extensions slot 22, MP-2) OrbifoldMPAllDegreesValidation (standard).
//
// "All degrees" (MP_impl_plan.md v0.2 Sec. 2.4, review R1; gate values
// MP_predictions.md P4.1 / P4.2 and the "MP-2 predictions" detail):
//   * P4.1a C_I (IsotropyModel, q_max = 7) and the hybrid (LocalModel) H^0..H^7
//     over F_ell, ell in {2,3,5,7}, n = 0..12 (both parities) = the frozen
//     mv_check.out (primary oracle, MvCheckOracle.inc);
//   * P4.1b the periodicity E^* = e u - : H^q(C_I) -> H^{q+2}(C_I) an
//     isomorphism for q = 2..5 ([C, lem:bulkperiodic](2), PeriodicityClass);
//   * P4.1c NC: ell.E^* (the zero map) at ell in {2,3} is not an isomorphism;
//   * P4.2 the certificate line, per triangulation and ring:
//       MP_ALL_DEGREES model=<M7|R_Z> ring=F_ell nerve_to=2 periodic_on=C_I
//       from=2 via=thm:mpcomparison certified=<k>/<k> scope=bulk verdict=OK
//     certified = the n <= 3 cells where (1) E^* is an isomorphism on C_I and
//     (2) Phi^* : H^q(C_I) -> H^q(MP) is one for q <= nerve_to.  On the MP
//     complex the degrees above nerve_to are Thm 2.1 TRANSPORTED, not computed;
//     scope: Gamma = SL_2(Z), the bulk only (not the boundary, the pair or a
//     finite-index subgroup; [C, rmk:bulkperiodicscope]).
// Line: ORBIFOLD_MP OrbifoldMPAllDegreesValidation checks=<k> failing=0.
// Read-only; no golden object, no setting, digest-neutral.
// ===========================================================================
uint32_t OrbifoldMPAllDegreesValidation();

#endif
