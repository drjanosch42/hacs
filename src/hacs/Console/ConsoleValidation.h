// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleValidation.h -- console phase C1a acceptance gates.
//
// Console plan (docs/console/00-console-interface-plan.md v3) Sec. 10 "Phase
// C1a" acceptance (a)-(g) and Sec. 11, and work package WP-7 of
// docs/implementation/hecke_extensions_slot3_plan.md Sec. 3.
//
// TWO registered names, both `fast`:
//
//   ConsoleScriptValidation        -- the bindings and the smoke script set,
//                                     IN PROCESS.  Every binding of plan
//                                     Sec. 3.4-3.6 is registered; every
//                                     `scripts/NNN_*.oc` runs on a fresh
//                                     Session and must exit 0; every pin inside
//                                     those scripts is read from
//                                     GoldenReference.inc through
//                                     `golden(point, object)` -- never a copied
//                                     number -- at the DEFAULT mathematical
//                                     configuration (fingerprint 0, rule R19).
//                                     It also carries the exit-code table
//                                     0/1/2/3/4/5 including the typed
//                                     not-implemented outcome, the R3
//                                     state / resume / save_state verbs, the
//                                     narrowed "no decomposition transcript
//                                     before C1b" guard, and a re-run of
//                                     check_console_includes.sh (plan Sec. 2 C8:
//                                     "C1a re-runs the same arm so the guard
//                                     survives if C0's validation is ever
//                                     narrowed").
//
//   ConsoleCliPreservationValidation -- the CLI, through std::system on THIS
//                                     binary.  The stdout and exit code of
//                                     --help, --bench --help, --bench micro
//                                     --help, --campaign --help, --campaign
//                                     <spec> --dry-run and one validation name
//                                     are byte-compared against files frozen
//                                     BEFORE the console branch landed
//                                     (Console/scripts/cli_frozen/); --list is
//                                     compared MODULO APPENDED NAMES, because
//                                     every slot of the extension stage
//                                     registers validations (plan Sec. 11).
//                                     Then the console's own forms are driven:
//                                     -c, --script with parameters, stdin, and
//                                     the exit codes.
//
// Both return 0 on success, the number of failed checks otherwise.
// ===========================================================================
#pragma once
#ifndef CONSOLE_VALIDATION_H
#define CONSOLE_VALIDATION_H

#include <cstdint>

uint32_t ConsoleScriptValidation();
uint32_t ConsoleCliPreservationValidation();

// (console phase C1b, slot-4 WP-5) Regenerates the `.expected` transcripts of
// Console/scripts/.  OWNER-GATED exactly as GoldenEmit is: a no-op unless run
// as `CONSOLE_EMIT_EXPECTED=1 ./cohomology_app ConsoleScriptEmit`, so `--all`
// stays read-only.  The FIRST emission was part of the C1b package -- a new
// artefact, nothing re-baselined -- and every hash line of the emitted files
// is cross-checked by ConsoleScriptValidation against `golden(...)` at
// fingerprint 0.  Any later re-emission must explain its diff file by file.
uint32_t ConsoleScriptEmit();

#endif // CONSOLE_VALIDATION_H
