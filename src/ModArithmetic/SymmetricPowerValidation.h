// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef SYMMETRIC_POWER_VALIDATION_H
#define SYMMETRIC_POWER_VALIDATION_H

#include <cstdint>

// ===========================================================================
// matrix-engine Phase 1, items 7.0 / 7.5 -- rho_n validation
// (docs/matrix-engine/00-implementation-plan.md, "Phase 1", "Validations").
//
// ComputeSymmetricPowerMatrixFM was replaced by closed forms for the
// structured 2x2 generators plus an O(d_V^2) two-term column recurrence.
// rho_n(A) is a mathematically UNIQUE matrix, so the new code must be
// BIT-IDENTICAL to the old one on every input; this harness proves it.
//
//   (a) exhaustive   -- every n_sym <= 48, all six structured generators,
//                       the [[1,0],[k,1]] family and 20 fixed-seed random
//                       2x2 matrices (deliberately including det != 1 and
//                       a == 0 mod ell), at ell in {2,3,5,691,134217757},
//                       compared entry by entry against
//                       ComputeSymmetricPowerMatrixFM_reference.
//   (b) spot         -- n_sym in {64, 96, 128, 160} x {T, S, generic} x
//                       ell in {5, 134217757}, same reference.
//   (c) structural   -- large n_sym (up to 1000): closed forms against
//                       independent Pascal/power oracles, plus
//                       rho_n(A) rho_n(B) = rho_n(AB) and
//                       rho_n(A)^{-1} = rho_n(A^{-1}) on fixed-seed pairs.
//                       No O(n^5) reference is involved here -- (c) is what
//                       actually covers the large-n regime.
//
// The reference implementation is super-cubic, which is why (a)/(b) stop at
// n_sym = 160; see the plan's rescoping note.
//
// Returns the number of FAILING cases (0 = pass). Each failing case prints a
// localising line.
// ===========================================================================
uint32_t SymmetricPowerValidation();

// Slow-tier companion: the same three sections at the grids that do not fit
// the ~2 minute budget of the `standard` tier (reference comparison up to
// n_sym = 224, structural identities at n_sym = 1000).
uint32_t SymmetricPowerValidationSlow();

#endif // SYMMETRIC_POWER_VALIDATION_H
