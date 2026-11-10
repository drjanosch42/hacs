// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_CHARPOLY_H
#define INTEGRAL_HECKEZ_CHARPOLY_H

// ===========================================================================
// HeckeZCharpoly -- slot-25 package HZ-RAT-PERF (docs/implementation/slot25/HZRATPERF_report.md, HZKIT_report.md Sec. 8.2 (b1)).
//
// The characteristic polynomial det(x I - A) of a b x b matrix over F_ell (ell an odd prime < 2^31) by a Hessenberg reduction
// (similarity transforms: row and column swaps, elimination below the subdiagonal) followed by the Hessenberg recurrence
// (Cohen, GTM 138, Alg. 2.2.9), O(b^3) word operations, every product reduced by Shoup's precomputed quotient (the row and column
// operations) or Barrett's (everything else) on 64-bit words; no division and no 128-bit remainder in a loop.
//
// VALUE-NEUTRAL: the characteristic polynomial over a field is unique, so the output equals hecke::level::HCharpoly's (Berkowitz,
// division-free, O(b^4) with a u128 remainder per multiply) coefficient for coefficient: low to high, monic, entries in [0, ell).
// HeckeZRationalValidation (arm CP) compares the two on random, sparse, nilpotent and block matrices mod several primes.
// ===========================================================================

#include <cstdint>
#include <vector>

namespace heckez {

// x * y mod p for x, y < p < 2^31 (Barrett with m = floor(2^64 / p)).
struct Barrett31 {
    uint64_t p = 0, m = 0;
    explicit Barrett31(uint64_t prime);
    uint64_t mul(uint64_t x, uint64_t y) const {
        const uint64_t t = x * y;
        const uint64_t q = static_cast<uint64_t>((static_cast<unsigned __int128>(t) * m) >> 64);
        uint64_t r = t - q * p;
        return r >= p ? r - p : r;
    }
    uint64_t reduce(uint64_t t) const {   // any t < 2^64
        const uint64_t q = static_cast<uint64_t>((static_cast<unsigned __int128>(t) * m) >> 64);
        uint64_t r = t - q * p;
        while (r >= p) r -= p;
        return r;
    }
    uint64_t pow(uint64_t a, uint64_t e) const;
    uint64_t inv(uint64_t a) const { return pow(a, p - 2); }   // p prime, a != 0
};

// det(x I - A) mod ell, low to high (b + 1 coefficients, monic). A row-major b x b with entries < ell (checked). ell odd prime < 2^31.
std::vector<uint64_t> CharpolyHessenberg(const std::vector<uint64_t>& A, uint32_t b, uint64_t ell);

}  // namespace heckez

#endif
