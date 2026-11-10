// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_SMALL_CARRIES_VALIDATION_H
#define HECKE_SMALL_CARRIES_VALIDATION_H
// (hecke-extensions slot 17, WP-8) Negative tests of the small carries' refusals: the strict hook grammar and the rule-19
// pre-run RSS refusal behind HECKE_LEVEL_N_SPARSE_CELLS (WP-5 audit MINOR-2), and the grammar of HECKE_DUMP_SAT_CERT_CELLS
// (L17-2 audit item 3). The console's `u_ell` gate on hecke_ring_algebra is tested in ConsoleErrorValidation arm (HR).
// Pure, fast, no env read, no output file.
#include <cstdint>

uint32_t HeckeSmallCarriesValidation();

#endif // HECKE_SMALL_CARRIES_VALIDATION_H
