// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_UP_VALIDATION_H
#define INTEGRAL_HECKEZ_UP_VALIDATION_H

#include <cstdint>

// slot-26 Z1 (UP-Z (a)): U_p, p | N, over Z_(ell), ell >= 5 -- the export's U_p against HZ-UTAG's chi_Q(U_p), a_p^2 = p^{k-2} on the
// p-new part, thread / ISA route identity, a tampered control, the recompute subset (15 golden level-1 + 3 anemic records hash-equal to
// HZ-DS edition 1) and the full level-N records' scan.  Gate line `HECKEZ_UP ... verdict=OK` (HeckeZUp.h; Z1_report.md).
uint32_t HeckeZUpValidation();

#endif
