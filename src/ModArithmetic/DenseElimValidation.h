// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// DenseElimValidation -- the blocked elimination vs oracle vs legacy
// (matrix-engine Phase 2, plan D5 / D5b)
// =====================================================================
//
// ENTRYWISE (never span-only) comparison of everything engine::elim
// produces -- RREF, pivot columns / rank, kernel basis, image basis, the
// transform U, inverse, multi-RHS coordinates, complement basis + projector,
// and the block-elimination step (L, B - L*R) -- against
//   (a) the naive RefLinAlg oracle (engine::ref::rref / kernel_basis), and
//   (b) the OLD FlatMatrix / factor_basis_block / change_of_basis_fm_multi
//       code paths, reached through engine::elim::ScopedLegacy, as well as
//       the retrofitted FlatMatrix wrappers themselves (engine mode),
// on fixed-seed inputs: sizes {1, 2, 7, 16, 17, 64, 65, 129, 256, 517}
// (square plus tall / wide variants at the smaller sizes), ranks {0, 1,
// small, half, full, full-1}, both element widths, primes {2, 3, 134217757,
// 2^31-1, 2^32-5} and 2^61-1 for uint64_t; a sweep over the blocking
// parameters (panel / base / trsm_base) on top, which must not change a
// single bit.  The large sizes run a reduced prime / rank set to stay
// within the budget.  Returns 0 on success.
//
// The real pipeline matrices (A_C1 at n_sym = 128, 1677 x 1419 of rank 258)
// are covered by DenseElimPipelineValidation in src/hacs/,
// which can reach the SL(2,Z) model; this library cannot.

#include <cstdint>

uint32_t DenseElimValidation();

// (S24 VAL-IMPL) The slow part of the same grid: the u64 width at the five primes below 2^32 and the parameter sweeps at
// 2^32-5 and 2 (u32).  DenseElimValidation + DenseElimSlowValidation = the pre-split grid, case for case.
uint32_t DenseElimSlowValidation();
