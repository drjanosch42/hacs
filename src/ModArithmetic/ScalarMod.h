// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_SCALAR_MOD_H
#define ENGINE_SCALAR_MOD_H
// =====================================================================
// ScalarMod.h -- ONE home for the scalar modular helpers (S24 CL-1;
// CODEAUDIT A2, B2, C-07; docs/implementation/slot24/CLEANUP_plan_1030.md)
// =====================================================================
//
// Header-only, STL-only, constexpr.  Written WITHOUT ModArithmetic's mod_*
// primitives, so it needs no raw_mod_calls.allow entry.  NumberField keeps its
// own ModUtil.h (its STL-only guard forbids including this file).
//
// About 90 file-local copies of these helpers exist repo-wide (powmod /
// mulmod / invmod / gcd / "ell^k or 0").  This header reproduces each of their
// VALUE semantics; later packages (CL-2, CL-3, CL-G1, CL-G3) migrate them.
// The copies differ in four ways, and every way stays selectable:
//
//   1. the product width.  "Narrow" copies form a*b in uint64_t and are exact
//      only for m < 2^32; the u64 overloads here use a 128-bit product (or the
//      portable shift-and-add below) and agree with them on that domain.
//   2. the inverse.  Three behaviours exist:
//        try_invmod      extended Euclid, any m >= 1; false iff gcd(a, m) != 1.
//                        m == 1 gives out = 0, true (the GammaH / InvModN copies
//                        return 0 there).  Copies that THROW on a non-unit keep
//                        their own text: ENGINE_CHECK_MSG(try_invmod(..), "...").
//        invmod_or_zero  the same, 0 for a non-unit.
//        invmod_fermat   powmod(a % p, p - 2, p) verbatim, p prime: 0 for a = 0
//                        mod p when p > 2, and 1 for every a when p == 2 (the
//                        unchecked Fermat copies; checked ones test a % p first).
//   3. "ell^k": powfit (0 when ell^k >= 2^64: PowFit / Pow64Fit) versus
//      pow_wrap (the product mod 2^64: PowU / IPow).
//   4. the signed reduction modi(x, m) of ModU(int64_t, m) (exact for every m;
//      the copies cast m to int64_t and are exact for m < 2^63).
//
// powmod(b, e, m) returns 1 % m at e == 0 (so 0 when m == 1) and reduces b
// first, as every copy does.
//
// The u64 product: unsigned __int128 where the compiler has it and
// OC_SCALAR_NO_INT128 is not defined, else mulmod_portable (the
// NF_NO_INT128 pattern of NumberField/ModUtil.h).  SharedHelpersValidation
// compares the two on every build.  The u128 overloads exist only where the
// compiler has unsigned __int128.
// =====================================================================

#include <cstdint>
#include <type_traits>

namespace engine {
namespace scalar {

#if defined(__SIZEOF_INT128__)
using u128 = unsigned __int128;
#endif

// ---- products ----------------------------------------------------------------
constexpr uint32_t mulmod(uint32_t a, uint32_t b, uint32_t m) {
    return static_cast<uint32_t>(static_cast<uint64_t>(a) * b % m);
}

// (a * b) mod m by shift-and-add, no 128-bit type (any m >= 1).
constexpr uint64_t mulmod_portable(uint64_t a, uint64_t b, uint64_t m) {
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

constexpr uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
#if defined(__SIZEOF_INT128__) && !defined(OC_SCALAR_NO_INT128)
    return static_cast<uint64_t>(static_cast<u128>(a) * b % m);
#else
    return mulmod_portable(a, b, m);
#endif
}

#if defined(__SIZEOF_INT128__)
// (a * b) mod m for m < 2^128, by shift-and-add (no 256-bit product).
constexpr u128 mulmod(u128 a, u128 b, u128 m) {
    a %= m;
    b %= m;
    u128 r = 0;
    while (b) {
        if (b & 1u) r = (r >= m - a) ? r - (m - a) : r + a;
        b >>= 1;
        a = (a >= m - a) ? a - (m - a) : a + a;
    }
    return r;
}
#endif

template <class T>
constexpr T addmod(T a, T b, T m) {   // a, b < m
    return a >= m - b ? a - (m - b) : a + b;
}
template <class T>
constexpr T submod(T a, T b, T m) {   // a, b < m
    return a >= b ? a - b : a + (m - b);
}

// ---- powers --------------------------------------------------------------------
template <class T>
constexpr T powmod(T b, uint64_t e, T m) {
    T r = static_cast<T>(1 % m);
    b %= m;
    while (e) {
        if (e & 1u) r = mulmod(r, b, m);
        b = mulmod(b, b, m);
        e >>= 1;
    }
    return r;
}

// b^k, or 0 when b^k >= 2^64 (the PowFit / Pow64Fit / FitsU64 copies; 1 at k = 0).
constexpr uint64_t powfit(uint64_t b, uint32_t k) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < k; ++i) {
        if (b != 0 && r > UINT64_MAX / b) return 0;
        r *= b;
    }
    return r;
}

// b^k mod 2^64 (the unchecked PowU / IPow copies).
constexpr uint64_t pow_wrap(uint64_t b, uint32_t k) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < k; ++i) r *= b;
    return r;
}

// ---- gcd, signed reduction -----------------------------------------------------------
template <class T>
constexpr T gcd(T a, T b) {
    while (b) {
        const T t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// x mod m in [0, m), m >= 1 (the ModU(int64_t, m) copies).
constexpr uint64_t modi(int64_t x, uint64_t m) {
    const uint64_t ax = x < 0 ? (~static_cast<uint64_t>(x) + 1u) : static_cast<uint64_t>(x);
    const uint64_t r = ax % m;
    return (x < 0 && r != 0) ? m - r : r;
}

// ---- inverses ------------------------------------------------------------------------
// The extended Euclid with the Bezout coefficient kept as a residue mod m, so no
// signed type and no overflow for any m >= 1.  True iff gcd(a mod m, m) == 1;
// then out = a^{-1} mod m (out = 0 when m == 1).
template <class T>
constexpr bool try_invmod(T a, T m, T& out) {
    T r0 = m, r1 = static_cast<T>(a % m);
    T s0 = 0, s1 = static_cast<T>(1 % m);   // s_i * a == r_i (mod m)
    while (r1) {
        const T q = r0 / r1;
        const T r2 = r0 - q * r1;
        const T s2 = submod(s0, mulmod(static_cast<T>(q % m), s1, m), m);
        r0 = r1;
        r1 = r2;
        s0 = s1;
        s1 = s2;
    }
    if (r0 != 1) return false;   // (m == 1 ends with r0 == 1, s0 == 0: Z/1's unit 0)
    out = s0;
    return true;
}

template <class T>
constexpr T invmod_or_zero(T a, T m) {
    T x = 0;
    return try_invmod(a, m, x) ? x : T(0);
}

// a^(p-2) mod p, verbatim (p prime by precondition; see point 2 above).
template <class T>
constexpr T invmod_fermat(T a, T p) {
    static_assert(sizeof(T) <= 8, "invmod_fermat: the exponent p - 2 must fit uint64_t");
    return powmod(static_cast<T>(a % p), static_cast<uint64_t>(p - 2), p);
}

} // namespace scalar
} // namespace engine

#endif // ENGINE_SCALAR_MOD_H
