// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LADIC_SPLIT_VALIDATION_H
#define HECKE_LADIC_SPLIT_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 18, WP-19b) The p != ell digit split: the
// hecke::ladic_cell::DigitCellSession (RunCell cut at the Hecke step) against
// fresh RunCell records, bit for bit (canonical_text and every serialised
// object), over every prime order, every width, both precision sources, the
// module consumer's keep_hecke / module_K and the refusals.  Read-only; no env.
uint32_t HeckeLadicSplitValidation();

#endif // HECKE_LADIC_SPLIT_VALIDATION_H
