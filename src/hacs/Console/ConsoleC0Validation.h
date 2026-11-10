// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleC0Validation.h -- console phase C0 acceptance gate.
//
// Console plan (docs/console/00-console-interface-plan.md v2) Sec. 10, Phase
// C0, and work package WP-5 of docs/implementation/hecke_extensions_slot1_plan.md
// Sec. 3.  The plan's working name for this entry is
// `ConsoleLanguageValidation`; it is registered as `ConsoleC0Validation`
// because that is the name the slot-1 work package assigns it (the deviation is
// recorded in docs/implementation/slot1/WP-5_report.md).
//
// What it gates, over ConsoleCore ONLY (no matrix is constructed anywhere in
// this file):
//
//   1. lexing      -- every literal form, comments, continuation, reserved
//                     words, and every lexical error
//   2. parsing     -- every production of the Sec. 3.2 EBNF, the precedence
//                     table, terminator rules, caret positions
//   3. evaluation  -- Int overflow, exact division, for/if, loop shadowing,
//                     ';' suppression, include + include cycles, args
//   4. printing    -- canonical vs display form, caps, show, JSON escaping
//   5. registry    -- overload resolution order, keyword arguments, defaults,
//                     arity and type errors, members and methods
//   6. settings + handles -- the R1 settings layer (spec, families, env-only,
//                     validation, snapshot, baseline) and the R3 resumable
//                     contract; the declared -> computed | failed state
//                     machine, keep-alive edges, creation sites
//   7. the "no transcript freezing" guard -- C0 can express no pin, and the
//                     tree carries no frozen console transcript before E0b
//   8. the C0 (c) performance floor -- 10 000 statements parsed and evaluated
//                     in well under 0.5 s
//
// Tier: fast.  Returns 0 on success, the number of failed checks otherwise.
// ===========================================================================
#pragma once
#ifndef CONSOLE_C0_VALIDATION_H
#define CONSOLE_C0_VALIDATION_H

#include <cstdint>

uint32_t ConsoleC0Validation();

#endif // CONSOLE_C0_VALIDATION_H
