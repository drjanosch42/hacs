// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_BIGINT_NTT_VALIDATION_H
#define INTEGRAL_BIGINT_NTT_VALIDATION_H

#include <cstdint>

// slot-27 BIGNTT M1 (docs/implementation/slot27/BIGNTT_design.md v0.2 Sec. 7): the NTT product and the Barrett modular
// product of nf::bigntt against BigInt's schoolbook / Karatsuba / Knuth-D route, bit for bit, on the FORCED route
// (threshold as a parameter, audit M2). Arms P (products, length-class boundaries, adversarial operands), T (threads
// 1 / 4 / 16, the pool and the nested spawn fallback, exception propagation through the pool), B (Barrett mulmod /
// sqrmod and <= 64-bit-exponent powmods), MR (the finder's Miller--Rabin: Mersenne primes, composites, the isqrt
// branch, a strong pseudoprime to the first 12 prime bases). Lines `EFFECTIVE BigIntNttValidation ntt_products=<n>
// barrett_steps=<m> ...` and `BIGNTT_VALIDATION arms=<k> products=<n> mismatched=0 routes_equal=yes verdict=OK`.
uint32_t BigIntNttValidation();

#endif
