// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// (hecke-extensions slot 23, Z2K) OrbifoldRing2kValidation: the ell = 2 (Z) phase of the orbifold census on the
// Z/2^k ladder engine::ring2k (CampaignOrbifold.h, OrbifoldRing2kEnabled) against the engine route it replaces.
//   (M) matrix level: on the real coboundaries delta^0 (Z/2^31), delta^1, delta^2 (Z/16) of the model C_I, the
//       Smith torsion exponents and the Howell kernel modules (rows, pivots, valuations, every entry) route on = off;
//   (R) record level: whole ell = 2 cells (ComputeOrbifoldCell) route on vs off -- the record bodies byte-identical
//       after masking only the sum-rule oracle's wall time "(x.xxx s)" (GPU_scout.md finding H-1), the log identical
//       after masking the PROFILE lap times; the route counters prove each arm took the route it claims (non-vacuity),
//       and an ell = 3 cell never reaches the ladder;
//   (C) negative controls: the masked comparator flags a one-character change of a body, and does not flag a change
//       of the masked timing; the EFFECTIVE line reports the scoped state.
// Last line: ORBIFOLD_RING2K ... checks= failing= verdict=OK.  Returns the failure count (0 on success).
#include <cstdint>
uint32_t OrbifoldRing2kValidation();
