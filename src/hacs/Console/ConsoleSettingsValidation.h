// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsoleSettingsValidation.h -- console phase C2a (hecke-extensions slot 6,
// WP-13): settings and resume.
//
// Plan docs/console/00-console-interface-plan.md v3.4 Sec. 10 Phase C2,
// acceptance (f), (g), (h), and Sec. 11's row for this validation.  Tier fast.
//
//   (f) THE ENGINE KNOBS.  Each settable key changes its documented getter and
//       its default restores it; `set` on every env-only key errors naming
//       the variable; interleaving three keys then `reset_settings()` returns
//       every getter to its env default; the knobs are visible inside a
//       run_concurrently task; a handle created under `lean = true` computes
//       lean after `set lean = false`; a session that ends restores what it
//       moved.
//   (g) THE MATHEMATICAL SETTINGS.  Every key of hecke::MathSettings::keys()
//       -- the count and the list are PRINTED and reconciled against [X]
//       Sec. 1.5's table, never compared to a literal -- round-trips through
//       `settings()` with its `source`; a bad value's error text IS
//       HeckeSettings' own; an unknown key is refused; the fingerprint is 0 at
//       the defaults and moves (as does the record's settings block, the
//       ext-layer hash input) under every non-default key; a key set in the
//       session is visible inside a run_concurrently task; and the EXIT-5 ARM:
//       `set field_polynomial = "conway"` is accepted and a use that reaches
//       its consumer exits 5.
//   (h) RESUME-INVARIANCE on the two shipped console instances (Decomposition
//       -- the locality loop -- and Sturm): resumed == fresh modulo
//       budget_history, a lower budget refused with the named error, a
//       complete state refused without moving, a changed non-budget setting
//       refused.
//
// Returns 0 on success, the number of failed checks otherwise.
// ===========================================================================
#pragma once
#ifndef CONSOLE_SETTINGS_VALIDATION_H
#define CONSOLE_SETTINGS_VALIDATION_H

#include <cstdint>

uint32_t ConsoleSettingsValidation();

#endif // CONSOLE_SETTINGS_VALIDATION_H
