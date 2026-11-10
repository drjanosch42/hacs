// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Mont.h -- 64-bit and 32-bit Montgomery arithmetic and a deterministic 64-bit
// Miller--Rabin test for the NumberField library (slot-19 WP-NF2: the
// Coates--Wiles layer route, CwLayer.h).
//
// HPC portability (owner rule): the 64 x 64 -> 128 product uses unsigned
// __int128 only where the compiler has it and NF_NO_INT128 is not defined;
// otherwise, and always in mul128_portable(), four 32 x 32 products.
// NumberFieldValidation compares the two on every build.
#include <cstdint>

#include "NfCheck.h"

namespace nf {

inline void mul128_portable(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
    const uint64_t a0 = a & 0xffffffffu, a1 = a >> 32, b0 = b & 0xffffffffu, b1 = b >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
    lo = (p00 & 0xffffffffu) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

inline void mul128(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
#if defined(__SIZEOF_INT128__) && !defined(NF_NO_INT128)
    const unsigned __int128 t = static_cast<unsigned __int128>(a) * b;
    lo = static_cast<uint64_t>(t);
    hi = static_cast<uint64_t>(t >> 64);
#else
    mul128_portable(a, b, hi, lo);
#endif
}

// a * b mod m for any m < 2^63 through the 128-bit product (shift-subtract reduction, portable).
inline uint64_t mulmod_u64(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t hi, lo;
    mul128(a % m, b % m, hi, lo);
    if (hi == 0) return lo % m;
    // (hi * 2^64 + lo) mod m by 64 doublings of hi (hi < m).
    uint64_t r = hi % m;
    for (int i = 63; i >= 0; --i) {
        r = (r >= m - r) ? r - (m - r) : r + r;  // r = 2r mod m
        if ((lo >> i) & 1u) r = (r >= m - 1) ? r + 1 - m : r + 1;
    }
    return r;
}

inline uint64_t powmod_u64(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m, b = a % m;
    while (e) {
        if (e & 1u) r = mulmod_u64(r, b, m);
        e >>= 1;
        if (e) b = mulmod_u64(b, b, m);
    }
    return r;
}

// Deterministic for n < 2^64 (the seven bases of Jim Sinclair).
inline bool is_prime_u64(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        if (n % p == 0) return n == p;
    }
    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1u) == 0) {
        d >>= 1;
        ++s;
    }
    for (uint64_t a : {2ull, 325ull, 9375ull, 28178ull, 450775ull, 9780504ull, 1795265022ull}) {
        uint64_t x = powmod_u64(a % n, d, n);
        if (a % n == 0 || x == 1 || x == n - 1) continue;
        bool comp = true;
        for (unsigned r = 1; r < s; ++r) {
            x = mulmod_u64(x, x, n);
            if (x == n - 1) {
                comp = false;
                break;
            }
        }
        if (comp) return false;
    }
    return true;
}

// Montgomery arithmetic modulo an odd p < 2^62, R = 2^64. mul(a, b) = a b R^{-1} mod p for a, b < p.
struct Mont64 {
    uint64_t p = 0, pinv = 0, r2 = 0;  // pinv = p^{-1} mod 2^64; r2 = R^2 mod p

    Mont64() = default;
    explicit Mont64(uint64_t prime) : p(prime) {
        NF_CHECK_MSG((p & 1u) && p < (uint64_t(1) << 62), "Mont64: modulus must be odd and < 2^62");
        uint64_t inv = p;  // Newton: 5 steps double the 3 correct low bits to 96
        for (int i = 0; i < 5; ++i) inv *= 2 - p * inv;
        pinv = inv;
        const uint64_t r1 = (0 - p) % p;  // 2^64 mod p
        r2 = mulmod_u64(r1, r1, p);
    }
    inline uint64_t mul(uint64_t a, uint64_t b) const {
        uint64_t hi, lo, qh, ql;
        mul128(a, b, hi, lo);
        const uint64_t q = lo * pinv;
        mul128(q, p, qh, ql);
        (void)ql;  // == lo by construction
        return hi >= qh ? hi - qh : hi + p - qh;
    }
    inline uint64_t to(uint64_t a) const { return mul(a % p, r2); }
    inline uint64_t from(uint64_t a) const { return mul(a, 1); }
    inline uint64_t add(uint64_t a, uint64_t b) const {
        const uint64_t s = a + b;
        return s >= p ? s - p : s;
    }
    inline uint64_t sub(uint64_t a, uint64_t b) const { return a >= b ? a - b : a + p - b; }
    // normal-form power (a normal, result normal)
    uint64_t pow(uint64_t a, uint64_t e) const {
        uint64_t r = to(1), b = to(a);
        while (e) {
            if (e & 1u) r = mul(r, b);
            e >>= 1;
            if (e) b = mul(b, b);
        }
        return from(r);
    }
};

// Montgomery arithmetic modulo an odd M < 2^31, R = 2^32 (the moments' Horner loop).
struct Mont32 {
    uint32_t M = 0, minv = 0, r2 = 0;
    Mont32() = default;
    explicit Mont32(uint32_t mod) : M(mod) {
        NF_CHECK_MSG((M & 1u) && M < (1u << 31) && M > 1, "Mont32: modulus must be odd, 1 < M < 2^31");
        uint32_t inv = M;
        for (int i = 0; i < 4; ++i) inv *= 2 - M * inv;
        minv = inv;
        const uint64_t r1 = (uint64_t(1) << 32) % M;
        r2 = static_cast<uint32_t>(r1 * r1 % M);
    }
    inline uint32_t mul(uint32_t a, uint32_t b) const {
        const uint64_t t = static_cast<uint64_t>(a) * b;
        const uint32_t q = static_cast<uint32_t>(t) * minv;
        const uint32_t th = static_cast<uint32_t>(t >> 32);
        const uint32_t qh = static_cast<uint32_t>((static_cast<uint64_t>(q) * M) >> 32);
        return th >= qh ? th - qh : th + M - qh;
    }
    inline uint32_t to(uint32_t a) const { return mul(a % M, r2); }
    inline uint32_t from(uint32_t a) const { return mul(a, 1); }
};

}  // namespace nf
