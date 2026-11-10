// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_N_WIDE_VALIDATION_H
#define HECKE_LEVEL_N_WIDE_VALIDATION_H
// (hecke-extensions slot 17, WP-5; slot-16 WP-13 finding F-1) The width arms of HeckeLevelNSparseValidation
// (no new registered name; [C] sec:levelN, the reach of def:cosetrepsN's table at composite N):
//   (W5b) BIT-IDENTITY, a gate: on the slot-16 ladder's levels (N <= 30030) the forced-__int128 path
//         (ScopedLevelNWidth kForceWide) equals the shipped attempt-then-escalate path exactly -- the
//         orbit-wise generators of S and ST entry for entry, MonomialOf / OrbitWiseFixedSpace (dense routes,
//         small N), LevelChangeMap (restriction and trace) -- and the shipped path escalates NOWHERE there
//         (LevelNWideProducts() == 0: the old int64 control flow, so no level-N golden object can move);
//   (W5n) the NEGATIVE CONTROL: the old width (kForceNarrow) refuses loudly (hecke::HeckeOverflow) on a
//         synthetic product beyond int64 and on the real generators of Gamma_0(60060) at n = 0, where the
//         shipped path completes on the __int128 tier (escalations > 0) with the orbit counts of
//         prop:levelorbit (3); the synthetic product's residues equal an independent u128 computation.
// The ladder beyond 5.5e4 runs through the existing hooks HECKE_LEVEL_N_SPARSE_CELLS (arm (C)) and
// HECKE_LEVEL_N_DEGEN_CELLS (arm (G)), whose SPARSE_CELL / DEGEN_CELL lines carry level_wide_products=.
// Adds its checks to `checks` and returns its failure count.
#include <cstdint>

uint32_t HeckeLevelNWideArms(uint32_t& checks);

#endif // HECKE_LEVEL_N_WIDE_VALIDATION_H
