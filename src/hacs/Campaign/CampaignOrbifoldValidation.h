// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// (hecke-extensions slot 22, L23-1) OrbifoldCampaignL23Validation: the orbifold section of the campaign
// (Campaign/CampaignOrbifold.h, `orbifold = c_i`, orbifold_cell/v1) end to end at ell = 2, 3, even and odd n <= 60,
// against the [C] sec:oddnltwo closed forms (as the record's own G gates), the frozen BIN-3 oracles (Bin3Oracle.inc:
// ladic2.py even n, oddn_l2.py odd n) and the frozen level-1 modular-forms sum rules (Bin3SumRules.inc); the spec-time
// refusals, the byte-identity of a spec without `orbifold`, the cell identity (fingerprint, `_s` suffix, the record
// hash input), resume, determinism and the fallback discipline.  Returns the failure count (0 on success).
#include <cstdint>
uint32_t OrbifoldCampaignL23Validation();
