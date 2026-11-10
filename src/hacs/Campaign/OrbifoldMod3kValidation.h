// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// (hecke-extensions slot 23, Z3K) OrbifoldMod3kValidation: the ell = 3 (Z) phase of the orbifold census on the
// Z/3^k rung engine::ring3k (Ring3k.h; CampaignOrbifold.h, OrbifoldMod3kEnabled) against the engine route it replaces.
//   (L) the rung itself: ring3k::smith / howell / kernel_module, and each variant (smith_levels, smith_lengths at
//       k <= 5, howell_blocked) at every k, against the unblocked references (smith_naive, howell_naive) and against
//       engine::elim (invariant_factors, howell, kernel_module over ModCtx::prime_power(3, k))
//       on random, zero, high-valuation, planted-Smith (P diag(3^sigma) Q, sigma including 0, k - 1 and k) and
//       rank-deficient matrices, k in {2, 3, 4, 5, 6, 7, 19} (and k = 1 against the reference; 5 / 6 straddle the variant switch), at the default and at tiny
//       panel / reduce-block sizes (every blocking boundary); the planted sigma are the oracle of the Smith arms;
//   (M) matrix level: on the real coboundaries delta^0 (Z/3^19), delta^1, delta^2 (Z/9) of the model C_I, the Smith
//       torsion exponents and the Howell kernel modules (rows, pivots, valuations, every entry) route on = off;
//   (R) record level: whole ell = 3 cells (ComputeOrbifoldCell) route on vs off -- the record bodies byte-identical
//       after masking only the sum-rule oracle's wall time (GPU_scout.md H-1, GL-1's record_compare.py rule), the log
//       identical after masking the PROFILE lap times, reader_path / sumrule_path equal; the route counters prove each
//       arm took the route it claims (non-vacuity), and an ell = 2 cell never reaches ring3k;
//   (C) negative controls: the mask, K = 1 / E = 1 refused, the non-canonical input refused, the EFFECTIVE line.
// Last line: ORBIFOLD_MOD3K ... checks= failing= verdict=OK.  Returns the failure count (0 on success).
#include <cstdint>
uint32_t OrbifoldMod3kValidation();
