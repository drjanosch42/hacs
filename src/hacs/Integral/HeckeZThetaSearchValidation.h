// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_THETA_SEARCH_VALIDATION_H
#define INTEGRAL_HECKEZ_THETA_SEARCH_VALIDATION_H

#include <cstdint>

// slot-27 TS-FIX / TS (docs/implementation/slot27/TS_design.md Sec. 9): the theta search of HZ-RAT (site 1, RationalStep) and of
// heckez.maximal_ideals (site 2, GeneratingTheta) under RationalSettings::theta_order.  (a) the text rule (explicit lex = the edition-1--3
// text byte for byte, lex_then_greedy = exactly two insertions, the spec key); (b), (c) the orders on the probe cells and the last-round
// rule; (c') (TS-FIX) the g_P predicate GPIsBound under a validation-only hook; (d) the continuations as pure functions on synthetic tables;
// (e) site 2 on records; (f) route identity at threads 1 vs 16.  Gate line `HECKEZ_THETA_SEARCH ... verdict=OK`.
uint32_t HeckeZThetaSearchValidation();

#endif
