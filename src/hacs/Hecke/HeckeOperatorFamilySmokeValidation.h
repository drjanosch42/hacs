// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_OPERATOR_FAMILY_SMOKE_VALIDATION_H
#define HECKE_OPERATOR_FAMILY_SMOKE_VALIDATION_H

#include <cstdint>

// Stage S9 -- HeckeOperatorFamily Definition-of-Done smoke validation.
//
// Asserts, for p in {2,3}, ell in {5,13,101}, n_sym in {0,2,10}:
//   * T_p_pow(p,1) == T_p(p)                         (r=1 recursion base);
//   * T_pp(p) == the scalar matrix p^n_sym * I per degree ([C, prop:diamond]);
//   * T_p_pow(p,2) == T_p_pow_direct(p,2)            (the corrected O9-style
//     identity: FULL recursion [C, thm:relations] == FULL determinant-p^2
//     double-coset route; both are the STANDARD Hecke operator T_{p^2},
//     equal for all ell >= 5);
//   * T_p_pow_direct(p,2) == T_p_pow_primitive_direct(p,2) + p^n_sym * I
//     (explicit witness of the primitive-vs-full double-coset decomposition
//     T_{p^2}^full = T'_{p^2} + p^n I; the primitive T'_{p^2} is the raw
//     Gamma diag(1,p^2) Gamma operator, which for r>=2 is NOT T_{p^2}).
//
// NOTE: distinct from the (not-yet-landed) Stage S9V validation
// `HeckeRelationsValidation` (the full independent O9/O10 battery -- see
// the plan's S9V section), which is written and registered separately by
// the independent S9V validation agent. S9V's O9/O10 must use the FULL
// operator (T_p_pow and the redefined T_p_pow_direct); the identity
// A(T_{p^2}) == A(T_p)^2 - p^{n+1} I now holds.
uint32_t HeckeOperatorFamilySmokeValidation();

#endif // HECKE_OPERATOR_FAMILY_SMOKE_VALIDATION_H
