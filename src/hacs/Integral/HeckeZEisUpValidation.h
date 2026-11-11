// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_EISUP_VALIDATION_H
#define INTEGRAL_HECKEZ_EISUP_VALIDATION_H

#include <cstdint>

// slot-26 HZ-EISUP: the Eisenstein U_p systems of the full algebra at N > 1 (HeckeZEisUp.h; HZEISUP_design.md v0.2 Sec. 5.2) -- ten cells,
// arms (a) the certificate C-E1 .. C-E5 with pinned digests, (b) the second route (the closed form), (c) HZ-UTAG's chi_Q(U_p) as an
// independent oracle, (d) four records through the record path (C-E7, final, pinned, scan clean), (e) bit identity (threads, reuse vs.
// recompute, ISA), (f) six tampers, (g) the HZ hash invariance, (h) the A4 schema tampers (scope and per-m completeness: code
// audit m4) and the re-scan of edition 2's 226 records, (i) the support word on a forged torsion read (code audit m2).
// Gate line `HECKEZ_EISUP cells=10 ... verdict=OK`.
uint32_t HeckeZEisUpValidation();

#endif
