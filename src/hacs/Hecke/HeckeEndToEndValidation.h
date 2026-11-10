// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_END_TO_END_VALIDATION_H
#define HECKE_END_TO_END_VALIDATION_H

// Stage S12 -- End-to-end validation battery + benchmark driver.
//
// This is the ACCEPTANCE GATE for the whole integral-Hecke-action feature: one
// aggregated, independently-runnable battery that exercises every contract
// obligation ([C, obl:welldef], [C, obl:agreement], [C, obl:tau],
// [C, obl:algebra], [C, obl:integral], [C, obl:backend], [C, obl:fp],
// [C, obl:shapiro], [C, obl:oracle] -- [C] = docs/latex/hecke_action.tex) and
// every relevant numerical oracle (O1-O16 of
// docs/implementation/hecke_action_implementation_plan.md Sec. 2), with the
// binding gating rules established by the per-stage V-validations (S7V's
// amendment A-F4 / the FIX-negI O2 correction, S9V's A-F1 bad-prime gating,
// S10V/S11V structure), plus a performance smoke.
//
// The three registered validations (all return 0 on success, invariant I9):
//
//   * HeckeEndToEndValidation -- the full oracle matrix over
//       p    in {2,3,5,7}
//       ell  in {5,7,13,101,691,10007,134217757}
//       n    in {0,1,2,3,8,10,11,14,16,18,20,22,24}
//       q    in {0,1,2}
//     applying the BINDING GATING (A-F4 + Sec 0.2):
//       - assert dim (O14) and char-poly (O5/O7) for all ell >= 5 EXCEPT the
//         mod-ell Dickson exceptional loci (n>0, (ell+1)|n) -- those are
//         RECORDED, not asserted;
//       - assert odd-n => H^* = 0 for ell != 2 (O2, now correct post-FIX-negI);
//       - only hard-assert the auditor-verified a_p constants (weights 12/16/24
//         + the Eisenstein eigenvalue 1+p^{n+1} + the 691 congruence); weights
//         18/20/22/26 are RUN and RECORDED, never hard-asserted.
//     Fails fast PER CASE but runs ALL cases; the RETURN VALUE is the aggregate
//     failure count (0 = all pass). Prints an obligation->test-IDs coverage
//     table and a per-(ell,n) summary.
//
//   * HeckeBadPrimeValidation -- the O10 bad-prime settings (p = ell in {2,5}),
//     with A-F1 gating: ell=5 hard-asserts U_25 = U_5^2 and rank(U^r)
//     monotonicity; ell=2 asserts only pipeline-completion / well-typedness and
//     RECORDS the recursion / rank observations.
//
//   * HeckeBenchmarkSmoke -- (ell,n) = (134217757, 100), p = 2: build the
//     pipeline, compute A^{(1)}(T_2), run HeckeOperatorOnK::compute() (all
//     runtime certs) + a single-operator block decomposition + certify();
//     asserts it COMPLETES and CERTIFIES (NO wall-time assertion) and PRINTS
//     the wall time (baseline for Open Question Q8). Level-(a) matrix printing
//     is size-gated (n=100 => Theta_flat is not dumped).

#include <cstdint>

// The full oracle matrix (aggregate failure count as return value).
uint32_t HeckeEndToEndValidation();

// O10 bad-prime settings (p = ell in {2,5}), A-F1 gating.
uint32_t HeckeBadPrimeValidation();

// Performance smoke at (ell,n) = (134217757, 100), p = 2 (no time assertion).
uint32_t HeckeBenchmarkSmoke();

#endif // HECKE_END_TO_END_VALIDATION_H
