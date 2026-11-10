// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_RECORD_VALIDATION_H
#define HECKE_EXT_RECORD_VALIDATION_H

#include <cstdint>

// hecke-extensions phase E2a (slot 4, WP-1) -- the EXTENSION RECORD's own
// validation: the `hecke_ext/v2` key set of [E] `sec:schema`, the one
// tri-state `Final` and `is_final`'s truth table over the mathematical
// settings, the per-complex degree fields of [C] `sec:recordschema` item 5,
// the R3.1 item-6 sidecar, Stage 5's `HeckeModuleStructure`, the
// `commutant_budget` key with its R3 resume, and the campaign job id's
// `_s<fingerprint>` suffix.
//
// It builds exactly ONE real cell -- (n, ell) = (10, 691) under `u_ell = off`,
// so no U_691 is formed and the arm stays inside the `fast` tier's budget --
// and everything else is constructed.  It emits NO golden entry: its three
// collectors (`ext_ideals^q`, `ext_certify^q`, `ext_json_canonical^q`) are
// defined and called HERE and nowhere else, which is what keeps slot 4's
// "no golden value moves, no golden entry is emitted" true.
//
// Returns the number of failing checks (0 on success).
uint32_t HeckeExtRecordValidation();

#endif // HECKE_EXT_RECORD_VALIDATION_H
