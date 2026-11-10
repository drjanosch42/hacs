// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleDeterminismValidation.h -- console phase C2b (hecke-extensions slot 7,
// WP-8): the determinism arms and the harness acceptance of
// docs/console/00-console-interface-plan.md v3.4 Sec. 10 C2, items (a)-(e).
//
// ONE registered name, `standard` ([P] Sec. 11: "ConsoleDeterminismValidation
// | standard | C2b/C4 | transcript digest identical across threads x program
// cache x device"; C2b ships its FIRST arm, threads, and C4 the rest):
//
//   (a) threads      every numbered `Console/scripts/*.oc`, run in process on
//                    a fresh session under `set threads = 1` and again under
//                    `set threads = 16`: the two transcripts (`#~` lines
//                    excluded) are byte-identical, script by script, and one
//                    FNV-1a-64 digest per arm is printed.
//   (b) backend      the C1 script set (001-016) under `set backend =
//                    "legacy"`: every `hash(...)` line -- indeed the whole
//                    transcript -- equals the engine arm's, unconditionally.
//   (c) run_validation("GoldenRegressionValidation") from inside a script
//                    returns 0; the script's transcript is exactly its own
//                    lines (the validation's chatter captured); a probe inside
//                    the lifted scope sees the host's engine state, not the
//                    session's `threads` / `device` overrides; and the
//                    overrides are back in force afterwards.
//   (d) run_campaign on the CampaignSmokeValidation spec (the very text,
//                    CampaignSmokeSpecText()) writes the same summary.tsv as
//                    campaign::RunCampaignSpec on the same spec, outside the
//                    two timing columns.
//   (e) bench("hecke", n = 48, ell = 134217757, p = 2).dims equals
//                    cohomology(...).dims (the CLI half of (e) is
//                    ConsoleCliPreservationValidation's).
//   plus two small arms for the rest of C2b: `--log` in script mode (the log
//   replays: its `> ` lines are the source, its other lines the transcript),
//   and gap G9 (`dickson_exceptional` == hecke::IsDicksonExceptional).
//
// (console C4, hecke-extensions slot 8, WP-10) [P] Sec. 10 C4 (a), "all arms":
//   (C4 a) product   the C1 script set (001-016) under every arm of threads in
//                    {1, 4, 16, 32} x program_cache in {on, off} x device in
//                    {cpu, hybrid, gpu} -- 24 arms, the 16 device arms [SKIP]
//                    without a usable CUDA device -- each on fresh sessions
//                    whose baseline carries the arm: every script exits 0 and
//                    every arm has the ONE transcript digest of the first.
//                    CONSOLE_DETERMINISM_ARMS=threads (read-only) runs only
//                    the four thread arms (cache on, cpu), the subset the
//                    CheckThread acceptance item names.
//
// Returns 0 on success, the number of failed checks otherwise.
// ===========================================================================
#pragma once
#ifndef CONSOLE_DETERMINISM_VALIDATION_H
#define CONSOLE_DETERMINISM_VALIDATION_H

#include <cstdint>

uint32_t ConsoleDeterminismValidation();

#endif // CONSOLE_DETERMINISM_VALIDATION_H
