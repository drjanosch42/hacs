// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKEZ_RATIONAL_VALIDATION_H
#define HECKEZ_RATIONAL_VALIDATION_H

#include <cstdint>

// slot-25 HZ-RAT (docs/implementation/slot25/00-slot25-plan.md Sec. 2.2; 06-work-packages.md "HZ-RAT"; the numbers of
// docs/implementation/slot24/heckez/02-toy-cases.md). `fast` tier. The library is Integral/HeckeZRational.{h,cpp} with the untrusted
// Integral/HeckeZFactorFinder.{h,cpp}.
//  T1  (1, 10), P = 2, 3, 5, 7: d = 2; theta = T_2, f = x^2 - 2025 x - 49176, disc f = 3^2 691^2; the gcd of the discriminants 691^2,
//      S_cand = {691} (disc_square); c_2(lambda) = 3 * 691, g_P = 691 from T_3 on, `upper_bound_certified`.
//  T2  (1, 22), P = 2, 3, 5, 7: d = 3; the running g_P 3^5 5^2 7^2 103 2294797 -> 5^2 103 2294797 -> 103 2294797 (the drop at T_5);
//      2294797 by a verified Pratt certificate; the cusp part x^2 - 1080 x - 20468736.
//  T5  (8, 10), anemic, P = 3, 5, 7, 11, 13: d = 13, d^an = 6, b1 = 22; (C1) with theta = T_3; S_cand subset {2, 3, 5, 7, 691}.
//  W44 (1, 42), P = 2, 3, 5, 7: d = 4; the discriminant gcd carries a 67-bit prime, labelled `probable` (Q1(a): above 2^64 no Pratt
//      check in C++), the cell `partial(unfactored)`.
//  Negative controls: a tampered factor list (one prime replaced) and a tampered Pratt certificate (a square as witness) are rejected.
//  Thread independence: the result digests of T1 and T2 at 1, 4 and 16 threads agree (the four cells run at 4 threads).
// Read-only arm: HECKEZ_RAT_CELLS="N:n,N:n,..." adds report-only cells (P = the first four primes of 2..13 prime to N, 4 threads),
// one ARM line each, then `EFFECTIVE HeckeZRationalValidation arm=cells cells=<k> refused=<r>`; it checks nothing and changes no value.
// Gate line: HECKEZ_RAT cells=4 mismatched=0 tampered_rejected=2 threads_equal=yes verdict=OK
// Returns the number of failing checks (0 on success).
uint32_t HeckeZRationalValidation();

#endif
