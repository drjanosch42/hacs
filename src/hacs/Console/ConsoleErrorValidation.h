// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleErrorValidation.h -- console phase C4 (hecke-extensions slot 8,
// WP-10): the error model of docs/console/00-console-interface-plan.md, Sec. 8
// (the origin -> behaviour table) and Sec. 2 C7 (normative), gated row by row.
//
// [P] Sec. 10 C4 acceptance (b): "ConsoleErrorValidation 0: every row of Sec. 8
// reaches its exit code with file:line:col and, for a lazy failure, the
// creation site; after an induced throw (model mismatch, non-commuting
// operators) the same session recomputes a golden-grid object and matches its
// golden(...) value, repeated under device = gpu when a device exists."
//
// ONE registered name, `fast` ([P] Sec. 11).  Its arms:
//
//   (R) the Sec. 8 rows, each a real script FILE run through the CLI surface
//       (console::RunConsole(registry, {"--script", path}) -- the same code
//       path as `cohomology_app --script`), asserting the exit code, the
//       `path:line:col: kind:` prefix of the error line, the C++ class the C7
//       translator names, and -- for every lazy failure -- the handle's
//       creation site `created at path:line:col`.  Rows whose trigger no
//       shipped verb can reach on demand (HeckeOverflow, ENGINE_CHECK,
//       std::bad_alloc, non-commuting operators handed to the decomposition)
//       are raised by TEST-ONLY verbs registered on this validation's own
//       registry, never on the global one; the non-commuting row runs the
//       real hecke::HeckeModuleDecomposition on a non-commuting pair, so the
//       throw comes from the pipeline's own exactness certificate.
//   (U) the exit-1 rows of Sec. 6: a missing script file, an undeclared
//       `--set` key, an unrecognised console argument.
//   (L) the lazy-handle rules of Sec. 2 C2: a failed handle is bound in state
//       `failed`, rethrown on every later access naming access site and
//       creation site, and NEVER retried (a counting stub: one compute()).
//   (P) the REPL survives every error: a verb-time error binds nothing, an
//       out-of-memory error leaves the session usable.
//   (I) post-throw integrity: one session takes every induced throw above,
//       then recomputes the golden-grid objects of (n, ell) = (10, 691), p = 2
//       -- dim H^q, A(T_p)^q, Theta_flat^q and the decomposition invariants --
//       and matches golden(...); under `device = cpu`, and again under
//       `device = gpu` when a CUDA device is usable ([SKIP] otherwise).
//   (K) `coefficient_ring` (slot-8 WP-3), CONDITIONAL on the key being
//       registered: the reserved value `Z` is refused by the setting itself
//       and a non-default ring is refused by a pipeline verb, both exit 5
//       ([E] sec:ring (a), (b)).  Before WP-3 merges the arm prints [SKIP]
//       (WP-3 carries its own arm); no key count is hard-coded.
//       (slot-9 WP-12, decision 2 (ii) (A)) The `set`-statement refusal now
//       carries its file:line:col; `--set coefficient_ring=Z` is exit 5 (the
//       pin changed 1 -> 5 in that commit), an invalid `--set` value stays
//       exit 1; and [E] sec:ring (h)'s lift is checked member by member under
//       Z/ell^2 at (10, 691) -- honoured (exit 0), the field members' named
//       logic_error (exit 3), everything else refused (exit 5) -- and at the
//       uncertified (30, 5), K = 2.
//       (slot-10 WP-12, decision 6 (ii) (A)) PINS CHANGED 5 -> 0 for L.n,
//       L.d_V, T.p, T.r; L.coefficient_ring added (exit 0).
//   (M) (slot-9 WP-8c) a campaign spec refused for a ring reason is exit 5.
//   (W) (slot-10 WP-12, the C5 leftovers, [S10] decision 6 (A)) a member
//       getter's error and the type-level resume refusal carry file:line:col
//       (the one named Console/Core exception, two sites; codes unchanged),
//       on a test-only raw-getter stub and on the shipped bindings; the five
//       ring-handle members honour without forcing a computation, a member
//       read follows the handle's ring, a verb refuses a handle of another
//       ring naming both (exit 5); run_campaign / campaign_plan's handler:
//       controls at exit 3 / 0 and the recorded reachability (no input at the
//       base reaches the exit-5 branch; no test-only hook in production).
//
// Writes only under a per-process temporary directory it removes.  Returns 0
// on success, the number of failed checks otherwise.
// ===========================================================================
#pragma once
#ifndef CONSOLE_ERROR_VALIDATION_H
#define CONSOLE_ERROR_VALIDATION_H

#include <cstdint>

uint32_t ConsoleErrorValidation();

#endif // CONSOLE_ERROR_VALIDATION_H
