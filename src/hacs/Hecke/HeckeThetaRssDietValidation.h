// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// HeckeThetaRssDietValidation (slot-23 PERF-P4a, tier standard).
//
// P4a lowers the peak RSS of HeckeOperatorOnK::compute_block() without moving a value:
//   (1) program 1's executor (its arena: the fused-assembly temps, 1.45 GB at
//       (n, ell, p) = (700, 13, 103)) is released after its last reader (Theta6 / Theta7),
//       before route 2 materialises rho2; Theta8 / 9 run on a fresh executor;
//   (2) route 2's rho2 memo is released after the route-2 programs;
//   (3) the A-N3 residual (certificate C5) is chunked over the right-hand sides (512 at a
//       time: P = coords[t0..] . C_eq^T COMPAREd with W[:, t0..]^T) instead of one in-place
//       Sub against a dim_C_eq x dim zero temp; W and the C_eq scratch are released at their
//       last reader, before the Theta_eq copy.
// This validation gates (1)-(3) against the independent legacy route
// (ScopedLegacyHeckeOperatorPath, its own Theta_eq solve and residual), entrywise:
//   (T) the tape operator (Theta_eq certification on) == the legacy operator: Theta_flat,
//       Theta_eq and A^{(q)} in every degree, and the certificate counters;
//   (L) the memory-lean tape operator (certification off + ScopedMemoryLean, the streamed
//       pool / Sturm operators of the census path) == (T) on A^{(q)} and Theta_block(q);
//   (C) chunk coverage: at least one arm has dim C_eq^1 > 512 with a ragged last chunk, and at
//       least one has > 1024 (three chunks or more, ragged), so the chunked residual runs
//       full chunks, interior chunks and a short tail.
// Arms: (n, ell) in {(50, 13), (42, 7), (64, 101), (114, 13)}, p in {2, 3, 31, ell} (p = ell is U_ell,
// the low-rank lever's path).  Prints one line
// `P4A_RSS_DIET arms=<a> checks=<n> failing=<f> chunked_arms=<c> chunked3_arms=<c3> digest=0x<fnv of A, Theta_eq> verdict=<OK|FAIL>`.
//
// (slot-24 G6) The G6 arm gates the field-route RSS diet (HECKE_FIELD_DIET, Util/FieldDiet.h;
// HeckeOperatorOptions (6)) against its rollback, entrywise:
//   (P) program 1 in parts (a part budget of two shared products forces many parts at
//       (n, ell) in {(50, 13), (64, 101)}, p in {2, 31}, certified and memory-lean):
//       Theta_block / Theta_flat / Theta_eq / A^{(q)} and the certificate counters == the
//       single program (ScopedFieldDiet(false));
//   (F) HeckeOperatorFamily::release_level_a(): T_p is the same object with the same
//       entries, T_p_pow / T_n unchanged, theta_flat_fm of a released prime throws;
//   (M) the boundary morphism's induced H maps by the lean basis read == the FlatMatrix
//       list read;
//   (S) the cone certificate's span predicate by the unit-row route == the dense RREF ==
//       the elimination route on in-span / out-of-span / no-unit-row operands (the route
//       taken checked), and one cone per setting: equal golden digests (this also covers
//       the shared Z basis at a degree whose differential has no rows) and certificates
//       that verify.
// Prints a second line `G6_FIELD_DIET checks=<n> failing=<f> parted_degrees=<d> parts=<p>
// released_pipelines=<r> unit_row_routes=<u> fallbacks=<b> verdict=<OK|FAIL>`; the
// P4A line and its digest are unchanged.
#include <cstdint>
uint32_t HeckeThetaRssDietValidation();
