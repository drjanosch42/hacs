// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_LOCAL_PAR_VALIDATION_H
#define INTEGRAL_HECKEZ_LOCAL_PAR_VALIDATION_H

#include <cstdint>

// HeckeZLocalParValidation (standard) -- slot-26 Z9 HZ-LOC-PAR, the route bit-identity of Integral/HeckeZLocal's par route
// (docs/implementation/slot26/Z9_predictions.md Sec. 1) against the slot-25 serial route (the rollback lever HECKEZ_LOCAL_PAR=0):
//  (S) Howell spans over Z/ell^K (u32 and u64 moduli, valuation conflicts, annihilator rows, vectors in the span): the same insert
//      sequence under the serial reduce and the two-phase parallel reduce (forced below its size threshold) at threads 1 / 4 / 16
//      gives the same insert results, the same row lists entrywise and the same membership answers;
//  (C) AlgebraClosure on random generators: the par route (semi-naive worklist) spans the same module as the serial route (equal
//      length, Smith length, generator count, mutual containment); its row list is the same at threads 1 / 4 / 16; one natural
//      D = 128 closure (above the thresholds, not forced) at threads 1 vs 16, row lists entrywise;
//  (R) the 30 cells of HeckeZLocalValidation under the serial route and under the par route (forced) at threads 1 / 4 / 16: the
//      DIGEST over every LocalRead::text() and module status is the same, and each run's gate reads verdict=OK.
// Gate line: HECKEZ_LOCAL_PAR cells=<k> threads_equal=yes routes_equal=yes default_route=<par|serial(...)> verdict=OK
// Returns the number of failing checks.
uint32_t HeckeZLocalParValidation();

#endif
