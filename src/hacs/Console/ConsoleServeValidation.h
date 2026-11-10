// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleServeValidation.h -- [P] v3.10 Sec. 18.11, `standard`
// (hecke-extensions slot 19, NB-2).
//
// Drives the REAL binary: fork/exec of its own executable with `--serve`, a
// whitelist environment (ENGINE_CUDA_DISABLE=1) and an empty scratch working
// directory, over pipes; the harness is a C++ NDJSON client (no Python).
//
//   (A) handshake: hello first, seq 0, oc-serve/1, json_version 1, counts =
//       the registry's tagged counts, settable = [P] Sec. 18.8's list; every
//       frame parses, no blank line, seq strictly +1;
//   (B) the transcripts -- the ConsoleDet guard: every numbered script of
//       Console/scripts/ (includes flattened by the harness) as ONE eval; the
//       concatenated `out` text, transcript-normalised as ConsoleScriptValidation
//       does, is byte for byte the `.expected` body; the `progress` frames are
//       the `#~` lines of the in-process run (digit runs normalised: timings);
//       no frame exceeds max_reply_bytes.  A script that calls an untagged
//       overload is in the explicit EXEMPT table and must be REJECTED as the
//       table states (an exempt script that passes fails: the table is
//       recomputed and must stay minimal);
//   (C) values: every `value` frame of (B) is a valid envelope whose canonical
//       is what the statement printed; hash(x) cross-check; and a determinism
//       arm: the (B) frame stream, volatile fields stripped, hashes alike at
//       ENGINE_THREADS 1 / 2 / 8 and with the program cache off;
//   (C2) the dynamic read_only audit: every tagged overload is called at least
//       once (the (B) run plus Console/serve/coverage.oc) under RLIMIT_FSIZE = 0,
//       the working directory empty before and after, `git status --porcelain`
//       of the source tree unchanged (probed: `git=absent` when not a checkout);
//   (D) error mapping, one cell per row of Sec. 18.7, the session surviving each;
//   (E) interrupt;  (F) protocol robustness;  (G) environment gate & hardening;
//   (H) the RSS cap (fatal rss_cap / exit 70, or an ordinary bad_alloc);
//   (I) chatter capture;  (J) completion / help show the served view only.
// Prints `EFFECTIVE ConsoleServeValidation arm=ALL cells=<k> scripts=<k>
// exempt=<j> ...`.  With OC_BUILD_NOTEBOOK=OFF: `[SKIP] ConsoleServeValidation:
// OC_BUILD_NOTEBOOK=OFF`, exit 0.
// ===========================================================================
#pragma once

#include <cstdint>

uint32_t ConsoleServeValidation();
