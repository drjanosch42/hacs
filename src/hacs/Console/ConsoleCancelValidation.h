// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleCancelValidation -- (NBCONFIRM S5) the cooperative cancel ([P] Sec. 7, Sec. 18.6).
//
//   (A) in process, per long verb (descent incl. its lazy read, hecke_structure, level_n, nebentypus,
//       hida_tower, hida_lambda, forms_algebra): the statement's cancellation points are traced; the
//       statement is then cancelled at EVERY traced point in turn (the test interlock `after:K`, Cancel.h),
//       each time in a fresh session: the stop names that point, nothing is bound (a lazy handle stays
//       declared and retries), the re-run in the same session prints exactly what a cold child process
//       prints (`-c`, no cancellation: the cache-poisoning check), and so does a fresh uncancelled session
//       after all the cancels;
//   (B) the REPL on a pseudo-terminal: Ctrl-C at a point (`wait:1:descent.charpoly`) prints `interrupted`,
//       the prompt returns, the re-read gives the cold digest; Ctrl-C at the idle prompt ends the process by
//       SIGINT (as before S5); a double Ctrl-C during a statement ends it by SIGINT too;
//   (C) --serve (OC_BUILD_NOTEBOOK=ON only, else [SKIP] of the arm): an `interrupt` sent while a lazy
//       descent read waits at a point ends the cell `interrupted`; the next cell's read gives the cold digest;
//       the stop latency is printed;
//   (D) scripts and -c install no handler and ignore the interlock (the transcripts are unchanged).
//
// standard tier; prints CONSOLE_CANCEL ... verdict=OK.
// ===========================================================================
#pragma once
#include <cstdint>

uint32_t ConsoleCancelValidation();
