// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// ModUtil.h -- word-size modular helpers of the NumberField library (slot-19 WP-NF).
//
// "Modular first" ([E] sec:lvnf, plan WP-NF): the heavy work runs mod ell^M on
// uint32_t with uint64_t products. A 64-bit modulus needs a 128-bit product;
// mulmod64() uses unsigned __int128 where the compiler has it and the portable
// mulmod64_portable() (shift-and-add, no 128-bit type) otherwise, or always
// when NF_NO_INT128 is defined. NumberFieldValidation compares the two, so the
// fallback is exercised on every build (aarch64 / HPC portability).
#include <cstdint>
#include <vector>

#include "NfCheck.h"

namespace nf {

inline uint64_t ipow_u64(uint64_t b, unsigned k) {
    uint64_t r = 1;
    for (unsigned i = 0; i < k; ++i) {
        NF_CHECK_MSG(b == 0 || r <= UINT64_MAX / b, "ipow_u64: overflow");
        r *= b;
    }
    return r;
}

inline uint32_t mulmod32(uint64_t a, uint64_t b, uint32_t m) {
    return static_cast<uint32_t>(((a % m) * (b % m)) % m);
}

inline uint32_t powmod32(uint64_t a, uint64_t e, uint32_t m) {
    uint64_t r = 1 % m, b = a % m;
    while (e) {
        if (e & 1u) r = r * b % m;
        e >>= 1;
        if (e) b = b * b % m;
    }
    return static_cast<uint32_t>(r);
}

// Inverse of a modulo m (any m >= 2); NF_CHECK(gcd(a, m) = 1).
inline uint32_t invmod32(uint64_t a, uint32_t m) {
    int64_t t = 0, nt = 1, r = m, nr = static_cast<int64_t>(a % m);
    while (nr) {
        const int64_t q = r / nr;
        int64_t tmp = t - q * nt;
        t = nt;
        nt = tmp;
        tmp = r - q * nr;
        r = nr;
        nr = tmp;
    }
    NF_CHECK_MSG(r == 1, "invmod32: not invertible");
    if (t < 0) t += m;
    return static_cast<uint32_t>(t);
}

inline uint64_t mulmod64_portable(uint64_t a, uint64_t b, uint64_t m) {
    a %= m;
    b %= m;
    uint64_t r = 0;
    while (b) {
        if (b & 1u) r = (r >= m - a) ? r - (m - a) : r + a;
        b >>= 1;
        a = (a >= m - a) ? a - (m - a) : a + a;
    }
    return r;
}

inline uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
#if defined(__SIZEOF_INT128__) && !defined(NF_NO_INT128)
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) % m);
#else
    return mulmod64_portable(a, b, m);
#endif
}

inline bool is_prime_u32(uint32_t n) {
    if (n < 2) return false;
    for (uint32_t p : {2u, 3u, 5u, 7u, 11u, 13u}) {
        if (n % p == 0) return n == p;
    }
    for (uint64_t d = 17; d * d <= n; d += 2) {
        if (n % d == 0) return false;
    }
    return true;
}

inline std::vector<uint32_t> prime_factors_u32(uint32_t n) {
    std::vector<uint32_t> f;
    for (uint32_t d = 2; static_cast<uint64_t>(d) * d <= n; ++d) {
        if (n % d == 0) {
            f.push_back(d);
            while (n % d == 0) n /= d;
        }
    }
    if (n > 1) f.push_back(n);
    return f;
}

// Smallest primitive root modulo a prime p.
inline uint32_t primitive_root_prime(uint32_t p) {
    NF_CHECK(is_prime_u32(p));
    if (p == 2) return 1;
    const std::vector<uint32_t> fac = prime_factors_u32(p - 1);
    for (uint32_t g = 2; g < p; ++g) {
        bool ok = true;
        for (uint32_t q : fac) {
            if (powmod32(g, (p - 1) / q, p) == 1) {
                ok = false;
                break;
            }
        }
        if (ok) return g;
    }
    NF_CHECK_MSG(false, "primitive_root_prime: none found");
    return 0;
}

// v_p(x) for x != 0.
inline unsigned valuation_u64(uint64_t x, uint64_t p) {
    NF_CHECK(x != 0 && p >= 2);
    unsigned v = 0;
    while (x % p == 0) {
        x /= p;
        ++v;
    }
    return v;
}

}  // namespace nf
