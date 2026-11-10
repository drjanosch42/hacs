// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleCostModelValidation -- NBCONFIRM S3 (slot 25, tier fast).
//
//   (A) Console/Core/CostModel's Estimate against the frozen Python reference
//       (Console/cost/expected_estimates.tsv, from docs/console/tools/
//       nbconfirm_expected.py): the 213 checked S2 cells at their own threads,
//       hw = 32, no profile; 72 rows with the frozen fixture profile
//       (Console/cost/test_device_profile.tsv) at threads 1/4/6/16, hw 6/32.
//       Floats to 1e-9 relative, flags and strings exactly.
//   (B) Decide (the design's rule, S2 Fix round 1) against the reference on
//       every (A) cell under five environments (1065 rows).
//   (C) the embedded model text = the tracked cost_models.tsv; every capped
//       verb has wall and RSS models, a class row, the safety rows, a calib set.
//   (D) the settings confirm / confirm_wall_s (defaults, refusals, engine
//       family, the mathematical fingerprint does not move).
//   (E) prompt-less byte identity: 41 calls just outside the silent envelopes
//       reproduce the PRE-S3 binary's stdout and exit code byte for byte
//       (Console/cost/refusals_pre_s3.tsv) under -c, under confirm = no, and
//       under a hostile host (terminal, 1 MiB MemAvailable, a slow profile);
//       a piped REPL refuses the same way.
//   (F) confirm = yes runs just-over cells (a `#~` note), and the hard
//       refusals (memory, C: <= 10 GB) still refuse.
//   (G) the terminal REPL policy in process (a fake prompt stream): silent
//       rule 4, normal accept / decline (nothing bound), strong needs a typed
//       `yes`, confirm = no refuses, a hard refusal never asks.
//   (H) the real binary's `--console` under a pseudo-terminal: a declined
//       prompt prints `not run` and binds nothing, an accepted one runs.
//   (I) the two typed refusals that replace ENGINE_CHECK leaks:
//       hecke_structure(1, 0) and hida_tower K < 2.
//   (J) the device-profile path, `none`, and the staleness rule.
//
// HACS_DEVICE_PROFILE is pinned (`none`, or the fixture) for the whole run.
// ===========================================================================
#pragma once
#ifndef CONSOLE_COST_MODEL_VALIDATION_H
#define CONSOLE_COST_MODEL_VALIDATION_H

#include <cstdint>

uint32_t ConsoleCostModelValidation();

#endif // CONSOLE_COST_MODEL_VALIDATION_H
