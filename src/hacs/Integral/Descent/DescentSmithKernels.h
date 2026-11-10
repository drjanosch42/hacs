// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_SMITH_KERNELS_H
#define INTEGRAL_DESCENT_DESCENT_SMITH_KERNELS_H

// PAR-D2: the inner kernels of the blocked Smith form (DescentSmith.cpp) over Z/ell^k, ell^k < 2^31.  Three bodies of the same
// exact map: scalar (the reference), AVX2, AVX-512 (runtime dispatch, __builtin_cpu_supports; target attributes, so the build
// flags stay unchanged).  Every body returns reduced representatives in [0, ell^k); the values are equal by exactness.
//   upd(row, U, f, nf, n):  row[j] := row[j] - sum_{x < nf} f[x] * U[x][j]  (mod ell^k), j < n; f[x], U[x][j], row[j] reduced.
//   first_odd / first_nondiv (the pivot search): the first index j in [0, n) with a[j] a unit, or n.
// Overflow: at ell = 2 the arithmetic is exact mod 2^32 and masked at the end; at odd ell the scalar body sums at most 3
// products (< 3 m^2 < 2^63.6) before a reduction, the SIMD bodies run Montgomery REDC on sums of at most 2 products
// (< 2 m^2 < m 2^32, the REDC precondition), m = ell^k < 2^31.

#include <cstdint>

#include "DescentMaps.h"

namespace integral_descent {
namespace maps {
namespace smithk {

struct KZ {
    bool two = true;
    uint32_t k = 1, ell = 2;
    uint32_t mask = 1;     // ell = 2
    uint32_t m = 2;        // ell^k
    uint32_t inv32 = 1;    // odd ell: ell^{-1} mod 2^32
    uint32_t lim = 0;      // odd ell: floor((2^32 - 1) / ell): x divisible by ell iff x * inv32 <= lim
    uint32_t mprime = 0;   // odd: -m^{-1} mod 2^32 (Montgomery)
    uint32_t r1 = 0;       // odd: 2^32 mod m
};
KZ MakeKZ(const ZMod& Z);

inline uint32_t Mul(const KZ& z, uint32_t a, uint32_t b) {
    return z.two ? ((a * b) & z.mask) : static_cast<uint32_t>(uint64_t(a) * b % z.m);
}
// x / ell^v for x divisible by ell^v (exact division)
inline uint32_t DivPowExact(const KZ& z, uint32_t x, uint32_t v) {
    if (z.two) return x >> v;
    for (uint32_t i = 0; i < v; ++i) x *= z.inv32;
    return x;
}
inline uint32_t Val(const KZ& z, uint32_t x) {   // x != 0, reduced
    if (z.two) return static_cast<uint32_t>(__builtin_ctz(x));
    uint32_t v = 0;
    while (x * z.inv32 <= z.lim) {
        x *= z.inv32;
        ++v;
    }
    return v;
}

using UpdFn = void (*)(const KZ& z, uint32_t* row, const uint32_t* const* U, const uint32_t* f, uint32_t nf, uint32_t n);
using FirstUnitFn = uint32_t (*)(const KZ& z, const uint32_t* a, uint32_t n);
struct Kernels {
    UpdFn upd = nullptr;
    FirstUnitFn first_unit = nullptr;
    int path = 0;
};
int Clamp(int simd);              // -1 auto -> the best the CPU has; else min(request, CPU)
const char* PathName(int path);   // "scalar" / "avx2" / "avx512"
Kernels Select(int simd);         // simd already clamped (or -1)

// The row scan of the pivot search on a[t .. c): the first unit (best = 0), else the first entry of minimal valuation (best =
// that valuation, k when the range is zero).
inline void ScanRow(const KZ& z, const Kernels& K, const uint32_t* a, uint32_t t, uint32_t c, uint32_t& best, uint32_t& col) {
    best = z.k;
    col = 0;
    if (t >= c) return;
    const uint32_t ju = t + K.first_unit(z, a + t, c - t);
    if (ju < c) {
        best = 0;
        col = ju;
        return;
    }
    for (uint32_t j = t; j < c; ++j) {
        const uint32_t x = a[j];
        if (!x) continue;
        const uint32_t v = Val(z, x);
        if (v < best) {
            best = v;
            col = j;
        }
    }
}

}  // namespace smithk
}  // namespace maps
}  // namespace integral_descent

#endif
