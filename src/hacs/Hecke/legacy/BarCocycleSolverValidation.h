// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_BAR_COCYCLE_SOLVER_VALIDATION_H
#define HECKE_BAR_COCYCLE_SOLVER_VALIDATION_H

#include <cstdint>

// ===========================================================================
// BarCocycleSolverValidation -- matrix-engine Phase 3 Stage B.
//
// WHAT IT GATES. Stage B absorbed the hand-rolled RREF-with-right-hand-side
// solver of BarCocycleH1.cpp (`solve_in_columns`, pre-Stage-B lines 119-163)
// into engine::elim::solve_multi_rhs. The plan makes that absorption
// CONDITIONAL (00-implementation-plan.md, Phase 3): it is allowed "only if the
// engine's solve reproduces their exact pivot rule and their Z-first /
// mod-ell-fallback policy". The written argument is in the comment above
// solve_in_columns in BarCocycleH1.cpp; this validation is its empirical pin.
//
// HOW. It keeps a byte-for-byte copy of the PRE-STAGE-B body as an oracle
// (`solve_in_columns_legacy` in the .cpp -- do not "simplify" it, it is a
// frozen reference) and compares, on every case:
//   * the boolean verdict (consistent / inconsistent), and
//   * the full coefficient vector when consistent, ENTRYWISE.
//
// CASES. Fixed-seed (deterministic, no time/RNG-state dependence) systems over
// ell in {5, 13, 691, 134217757, 2147483647, 4294967291} covering: full column
// rank; rank-deficient column sets (so the free-coordinate convention is
// exercised, which the two BarCocycleH1 call sites themselves never reach);
// right-hand sides in and out of the column span; wide (Kc > R) and tall
// (R > Kc) shapes; Kc = 0 with b = 0 and b != 0; R = 0; duplicated and zero
// columns. Returns 0 on success, nonzero on the first disagreement.
// ===========================================================================
uint32_t BarCocycleSolverValidation();

#endif // HECKE_BAR_COCYCLE_SOLVER_VALIDATION_H
