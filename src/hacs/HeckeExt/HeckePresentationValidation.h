// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_PRESENTATION_VALIDATION_H
#define HECKE_EXT_PRESENTATION_VALIDATION_H

// hecke-extensions phase E2 (slot 6, WP-2; OQ11) -- the validation of
// HeckeExt/HeckePresentation, the on-request presentation of [X] Sec. 2.2
// Stage 5, with [X] Sec. 5's arms EXCEPT the key arm:
//   * (10, 691): H^1_+ ~= T_m free of rank one (one generator, no relation),
//     reached through ExtDriverOptions::with_presentation;
//   * the unsplit H^1_m at (10, 691): t = 2 and exactly one minimal relation;
//   * constructed algebras (the shapes of HeckeExtAlgebraValidation) against
//     their known presentations, in the stored basis and conjugated;
//   * a tamper arm on verify();
//   * the record text (and the two per-sign records) byte-identical with the
//     option on and off -- the option must not leak into the record.
// DEFERRED to slot 7's changed-line re-baseline (slot-6 plan Sec. 2.3 item 2,
// Sec. 4.4 stage 1, Sec. 10 ruling 3): the `presentation = off | on` settings
// key, the record key, and the arm "record hash unchanged by turning it on".
//
// Returns 0 on success; nonzero = number of failing checks.  Fast tier.

#include <cstdint>

uint32_t HeckePresentationValidation(void);

#endif // HECKE_EXT_PRESENTATION_VALIDATION_H
