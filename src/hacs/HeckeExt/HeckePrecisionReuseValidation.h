// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_PRECISION_REUSE_VALIDATION_H
#define HECKE_EXT_PRECISION_REUSE_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 23, G1) The gate of [C, obl:precisionreuse](v): one module run per cell, the
// precision levels below it derived ([C, lem:precisionreduction]; ringalg::LevelCache), against the per-K
// route (one run per level read, HECKE_PRECISION_REUSE=0: the oracle and rollback arm) -- the whole printed
// output (LOEWY_RING and SAT_DEFECT lines) byte for byte, K_run (the levels read) equal, timings and Lv_run
// excepted; the derived-module checks (iii) and their four negative controls.  See the .cpp for the grid.
//
// HeckePrecisionReuseValidation      (standard): the cheap grid; (108, 7) in place of (300, 7).
// HeckePrecisionReuseSlowValidation  (slow): (300, 7) (eps = [9,16,21,25,27,27]) and (300, 11) on the sat route,
//                                    with the per-K oracle (~ 378 s + 243 s single-thread on it alone).
uint32_t HeckePrecisionReuseValidation();
uint32_t HeckePrecisionReuseSlowValidation();

#endif // HECKE_EXT_PRECISION_REUSE_VALIDATION_H
