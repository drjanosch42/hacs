// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// HeckeIotaStreamValidation (slot-23 P4I, tier standard).
//
// P4I lowers the host RSS of the iota phase (HeckeInvolution::compute(), the
// binding RSS item of the HeckeExt pass at n >= 500): the tasks of
// check_obl_iii and build_split are admitted first come, first served under a byte
// budget instead of all at once, the flat +- split takes the column rank profile of one
// (X +- iota_C X)/2 buffer per sign built straight from the stored cochains
// (no X / Pmat / RREF-cache / image() copies), and the C / Z / B +- bases are
// stored once (in the certificate).  Rollback HECKE_IOTA_STREAM=0 /
// HeckeInvolution::ScopedIotaStreaming(false) = the pre-P4I computation.
// Value-neutral; this validation gates it entrywise:
//   (X) per arm, the streamed route (admission budgets: automatic = twice the
//       largest task, 1 byte = one task at a time, 40000 bytes = admission
//       boundaries inside a degree with tasks above the budget running alone,
//       unbounded) against the materialised route: iota^* on H^q, H^q_+-,
//       C/Z/B^q_+- entrywise, dims, certificate verify;
//   (V) non-vacuity / the bound: the streamed route ran (streamed families > 0
//       off ell = 2 and odd n), the budget in force, the admitted estimate
//       <= max(budget, largest task), one task at a time at 1 byte, a task
//       above the 40000-byte budget alone; the materialised route reports no
//       streaming;
//   (S) the serial path (one-worker consumer pool);
//   (L) the legacy elimination (engine::elim::ScopedLegacy on a one-worker
//       consumer pool): the streamed split's FlatMatrix branch, same objects;
//   (M) mutations (ScopedIotaStreamMutation 1 / 2 / 3: an entry of a + basis,
//       a pivot replaced by a non-pivot column, the last split task with real
//       work not run) must be applied and caught (compute() throws or
//       the comparison differs); the materialised route ignores them;
//   (N) negative control: the comparator flags a flipped entry;
//   (F) (audit F1) a thread-context installer that throws (validation-only,
//       registered by this file, armed per scope): on a 4-worker consumer pool,
//       once (the first installer call) and on every task thread, streamed and
//       materialised -- compute() returns within a watchdog's 300 s (else the
//       process exits with a FAIL line) and surfaces exactly the injected
//       error; a clean streamed run afterwards equals the reference.
// Arms: (n, ell) in {(12, 13), (24, 7), (30, 13), (40, 5), (18, 3), (20, 101),
// (13, 7) (odd n), (12, 2) (char 2: no split, obl(iii) admission only), (100, 13)}; (F) on (24, 7).  Prints one line
// `P4I_IOTA_STREAM arms=<a> checks=<n> failing=<f> streamed_families=<s> admissions=<t> mutations=<m> faults=<k> verdict=<OK|FAIL>`.
#include <cstdint>
uint32_t HeckeIotaStreamValidation();
