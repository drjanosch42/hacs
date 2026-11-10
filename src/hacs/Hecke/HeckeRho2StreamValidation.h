// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// HeckeRho2StreamValidation (slot-23 P4R, tier standard).
//
// P4R streams route 2 of HeckeOperatorOnK::compute_block() (the O12(a) cross-check
// rho_n((delta_i gamma_k) mod ell), [C, eq:heckechain]): one d x d accumulator per route-2
// block, the distinct reduced keys evaluated in small batches (each key once) and added
// into the accumulators that use them, so the per-degree memo rho2 (one d x d FlatMatrix
// per distinct key, ~1.4 GB at (n, ell, p) = (700, 13, 103)) is never built.
// The rollback HECKE_RHO2_STREAM=0 / ScopedRho2Streaming(false) is the materialised memo
// and its per-row "hecke.route2" programs, unchanged.  Value-neutral: route 2 only ever
// produces a verdict.  This validation gates the change entrywise:
//   (X) cross-check: per arm, the route-2 blocks of the streamed route and of the
//       materialised memo (captured through the validation-only ScopedRoute2Capture)
//       have the same key set and agree entrywise, and both equal the fused block of
//       Theta_block(q); Theta_flat, A^{(q)} and the O12(a) certificate agree on/off;
//   (V) non-vacuity: the streamed route ran (degree / key / block counters; every key
//       evaluated exactly once), the materialised one did not under the rollback;
//   (N) negative control: the entrywise comparator flags a flipped entry;
//   (M) mutation: ScopedRoute2TargetMutation perturbs ONE entry of the fused block route 2
//       is compared against; compute() must throw "O12(a) FAILED" naming exactly that entry,
//       streamed and materialised alike (at p = 31, audit F4, the first block of every
//       degree, so multi-batch degrees are mutated too);
//   (R) (audit F3) ScopedRho2StreamReduceEvery(1 / 2) forces the delayed accumulation's
//       in-loop reduction (dead in production unless k_max <= 64): it must run (counter)
//       and the blocks must equal the materialised ones;
//   (B) (audit F4) (n, ell, p) = (12, 101, 61), a degree with 257 keys (1 mod the batch
//       of 64): (X), (R) and (M) there.
// Arms: (n, ell) in {(50, 13), (42, 7), (64, 101)}, p in {2, 3, 31, ell} (p = ell is the
// U_ell lever, whose factored route 2 is not streamed: there the arm checks the counters
// stay zero), plus one memory-lean arm (the census path), plus (B).  Prints one line
// `P4R_RHO2_STREAM arms=<a> checks=<n> failing=<f> blocks=<b> mutations=<m> reduce_arms=<r> verdict=<OK|FAIL>`.
#include <cstdint>
uint32_t HeckeRho2StreamValidation();
