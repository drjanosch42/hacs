// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_MIXED_FINALITY_VALIDATION_H
#define HECKE_EXT_MIXED_FINALITY_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 13, WP-11 -- the evidence arm of the `mixed` finality
// proposal (docs/implementation/slot13/WP-11_mixed_finality_proposal.md,
// Proposition M).  `slow` tier (opt-in: not in --all); read-only, no record,
// golden line or digest is touched.  One MIXED_FIN line per H^1 ideal.
uint32_t HeckeMixedFinalityValidation();

#endif // HECKE_EXT_MIXED_FINALITY_VALIDATION_H
