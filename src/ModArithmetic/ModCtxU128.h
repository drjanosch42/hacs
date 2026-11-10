// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// ModCtxU128.h -- the composite-only u128 context (ENGINE_API.md Sec. 13.12;
// hecke-extensions slot 11, WP-18a: L1-X step 1, the engine layer)
// =====================================================================
//
// Included at the end of ModCtx.h; never included directly.
//
//   Elt = engine::u128 (unsigned __int128)   Z/ell^k, ell odd prime, k >= 2,
//         ell^k < 2^127                     accumulator engine::u256
//
// ModCtx<u128> is a FULL SPECIALISATION of ModCtx, so the uint32_t and
// uint64_t instantiations are not touched by a single token (Sec. 13.12.3,
// "Not touched").  It is COMPOSITE-ONLY (Sec. 13.12.2):
//   * the only way to obtain one is ModCtx<u128>::prime_power(ell, k); there
//     is no ModCtx<u128>(p) constructor and no init(p), so a field context
//     of this type does not compile;
//   * is_field() is false for every context; pow() and batch_inv() are
//     ABSENT (field-only, Sec. 13.3), so their misuse does not compile;
//   * engine::is_composite_only<u128> is true (and false for the other two).
//
// Regime U128_Montgomery (Sec. 13.12.2): R = 2^128, p = ell^k odd, p < 2^127
// (so the REDC result t < 2p fits a u128).  REPRESENTATION: canonical
// integers in [0, p) everywhere -- storage, views, DenseBuf, BlockPool
// content, hashing, the Sec. 13.4 members.  The Montgomery form never escapes
// a primitive:
//   mul(a, b)          = REDC(REDC(a*b) * r2)        one-shot product
//   mul_prep(w)        = REDC(w * r2) = w*R mod p    the reused-operand pair
//   mul_prepped(x, w') = REDC(x * w') = x*w mod p    (the Sec. 13.5 row AXPY)
//   reduce(T)          = REDC(REDC(T) * r2)          T < p*R (the Sec. 5 bound)
// so mul_prepped(x, mul_prep(w)) == mul(x, w) == (x*w) mod p bit for bit --
// the same interface as ModCtx<uint64_t>'s F-1 pair (slot-11 WP-12), which is
// what lets DenseElim.cpp's composite rows instantiate for u128 unchanged.
//
// inv(u) is UNIT-ONLY (Sec. 13.3's meaning): Hensel lifting from the
// uint64_t inverse of u mod ell, x <- x (2 - u x) mod p, ceil(log2 k) steps;
// ell | u throws std::runtime_error("Modular inverse does not exist.").
//
// The Sec. 5 bound for u128: a k-block of kb products of canonical residues
// is exact in u256 and REDC needs T < p*R, so
//   k_max = floor((p * 2^128 - 1) / (p - 1)^2)     (saturated to 2^64 - 1),
// computed exactly at construction (a 256-bit long division).  kb <=
// min(k_max, 256) is ENGINE_CHECKed at every block start by the kernels.

namespace engine {

// ---- the 256-bit accumulator ---------------------------------------------------
// Two u128 limbs, value = hi * 2^128 + lo.  The compiler has no 256-bit type;
// an exact 128x128 product costs four 64x64 -> 128 multiplies.
struct u256 {
    u128 lo = 0;
    u128 hi = 0;
    u256() = default;
    constexpr u256(u128 lo_) : lo(lo_), hi(0) {}          // NOLINT: implicit widening, like the builtin types
    constexpr u256(u128 hi_, u128 lo_, int) : lo(lo_), hi(hi_) {}
    u256& operator+=(const u256& x) {
        const u128 l = lo + x.lo;
        hi += x.hi + static_cast<u128>(l < lo);
        lo = l;
        return *this;
    }
    friend u256 operator+(u256 a, const u256& b) { a += b; return a; }
    friend bool operator==(const u256& a, const u256& b) { return a.lo == b.lo && a.hi == b.hi; }
    friend bool operator!=(const u256& a, const u256& b) { return !(a == b); }
    friend bool operator<(const u256& a, const u256& b) { return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo; }
    friend bool operator<=(const u256& a, const u256& b) { return !(b < a); }
};

namespace detail {

// the exact 128 x 128 -> 256 product
inline u256 mul_wide128(u128 a, u128 b) {
    const uint64_t a0 = static_cast<uint64_t>(a), a1 = static_cast<uint64_t>(a >> 64);
    const uint64_t b0 = static_cast<uint64_t>(b), b1 = static_cast<uint64_t>(b >> 64);
    const u128 p00 = static_cast<u128>(a0) * b0;
    const u128 p01 = static_cast<u128>(a0) * b1;
    const u128 p10 = static_cast<u128>(a1) * b0;
    const u128 p11 = static_cast<u128>(a1) * b1;
    // middle column: < 3 * 2^64, no wrap
    const u128 mid = (p00 >> 64) + static_cast<uint64_t>(p01) + static_cast<uint64_t>(p10);
    u256 r;
    r.lo = (mid << 64) | static_cast<uint64_t>(p00);
    r.hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    return r;
}

// 256-bit shifts / subtraction for the construction-time long division.
inline u256 shl1(const u256& x) { return u256(static_cast<u128>((x.hi << 1) | (x.lo >> 127)), static_cast<u128>(x.lo << 1), 0); }
inline u256 sub256(const u256& a, const u256& b) {   // a >= b
    return u256(static_cast<u128>(a.hi - b.hi - static_cast<u128>(a.lo < b.lo)), static_cast<u128>(a.lo - b.lo), 0);
}
// floor(n / d), d != 0: restoring binary long division, 256 steps.  Used at
// construction only (k_max), never on a value path.
inline u256 div256(const u256& n, const u256& d) {
    u256 q, r;
    for (int i = 255; i >= 0; --i) {
        const bool bit = (i >= 128) ? ((n.hi >> (i - 128)) & 1) : ((n.lo >> i) & 1);
        r = shl1(r);
        r.lo |= static_cast<u128>(bit);
        if (!(r < d)) {
            r = sub256(r, d);
            if (i >= 128) q.hi |= static_cast<u128>(1) << (i - 128);
            else          q.lo |= static_cast<u128>(1) << i;
        }
    }
    return q;
}

} // namespace detail

template<> struct ModTraits<u128> {
    using Accum = u256;
    static constexpr int bits = 128;
};

// Sec. 13.12.2: true for u128 alone.
template<class Elt> struct is_composite_only : std::false_type {};
template<> struct is_composite_only<u128> : std::true_type {};

template<>
struct ModCtx<u128> {
    using Elt   = u128;
    using Accum = u256;
    static constexpr bool is_u32 = false;

    // ---- constants ----------------------------------------------------------
    u128     p        = 0;
    u128     mont128_ninv = 0;   // -p^{-1} mod 2^128
    u128     mont128_r1   = 0;   // R mod p, R = 2^128 (Montgomery form of 1)
    u128     mont128_r2   = 0;   // R^2 mod p
    uint64_t k_max    = 0;       // Sec. 5 bound (see the header comment)
    Regime   regime   = Regime::U128_Montgomery;

    // ---- the declared ring (Sec. 13.1) ----------------------------------------
    u128     ell      = 0;       // the prime; < 2^64 (k >= 2 and ell^k < 2^127)
    uint32_t k        = 0;
    // ell >= 3 and ell^k < 2^127 force k <= 80; 127 entries hold ell^0 ..
    // ell^126 whatever the prime (Sec. 13.12.2: a fixed std::array, so the
    // context stays trivially copyable).
    static constexpr uint32_t kPowTable = 127;

    ModCtx() = default;

    // Z/ell^k, ell an odd prime (precondition, as for every prime_power),
    // k >= 2, ell^k < 2^127.  Admissible below 2^64 as well (the oracle rule,
    // Sec. 13.12.4, runs it there).  ENGINE_CHECKs the rest.
    static ModCtx prime_power(u128 ell_, uint32_t k_) {
        ENGINE_CHECK_MSG(ell_ >= 3 && (ell_ & 1) != 0,
                         "ModCtx<u128>::prime_power: ell must be an odd prime (U128_Montgomery needs an odd modulus)");
        ENGINE_CHECK_MSG(ell_ <= std::numeric_limits<uint64_t>::max(), "ModCtx<u128>::prime_power: ell must be < 2^64");
        ENGINE_CHECK_MSG(k_ >= 2, "ModCtx<u128>::prime_power: k must be >= 2 (u128 is composite-only)");
        ENGINE_CHECK_MSG(k_ < kPowTable, "ModCtx<u128>::prime_power: ell^k must be < 2^127");
        const u128 lim = static_cast<u128>(1) << 127;
        ModCtx c;
        c.ell_pow_.fill(0);
        c.ell_pow_[0] = 1;
        for (uint32_t v = 1; v <= k_; ++v) {
            ENGINE_CHECK_MSG(c.ell_pow_[v - 1] <= (lim - 1) / ell_, "ModCtx<u128>::prime_power: ell^k must be < 2^127");
            c.ell_pow_[v] = c.ell_pow_[v - 1] * ell_;
            ENGINE_CHECK_MSG(c.ell_pow_[v] < lim, "ModCtx<u128>::prime_power: ell^k must be < 2^127");
        }
        c.p = c.ell_pow_[k_];
        c.ell = ell_;
        c.k = k_;
        // Newton for p^{-1} mod 2^128 (p odd): p*p == 1 mod 8 (3 bits), each
        // step doubles the correct bits: 3 -> 6 -> 12 -> 24 -> 48 -> 96 -> 192.
        u128 inv = c.p;
        for (int i = 0; i < 6; ++i) inv *= static_cast<u128>(2) - c.p * inv;
        ENGINE_CHECK(c.p * inv == 1);
        c.mont128_ninv = static_cast<u128>(0) - inv;
        // R mod p = (2^128 - 1) mod p + 1, reduced
        c.mont128_r1 = (~static_cast<u128>(0)) % c.p + 1;
        if (c.mont128_r1 == c.p) c.mont128_r1 = 0;
        // R^2 mod p = r1 * 2^128 mod p by 128 modular doublings (construction only)
        u128 r2 = c.mont128_r1;
        for (int i = 0; i < 128; ++i) { r2 <<= 1; if (r2 >= c.p) r2 -= c.p; }   // r2 < p < 2^127: no wrap
        c.mont128_r2 = r2;
        // k_max = floor((p*R - 1) / (p - 1)^2), saturated to 2^64 - 1
        const u256 d = detail::mul_wide128(c.p - 1, c.p - 1);
        const u256 pr_minus_1(static_cast<u128>(c.p - 1), ~static_cast<u128>(0), 0);   // p*2^128 - 1
        const u256 q = detail::div256(pr_minus_1, d);
        c.k_max = (q.hi != 0 || q.lo > std::numeric_limits<uint64_t>::max())
                      ? std::numeric_limits<uint64_t>::max() : static_cast<uint64_t>(q.lo);
        c.regime = Regime::U128_Montgomery;
        return c;
    }

    // Sec. 13.12.2: every u128 context is composite.
    bool is_field() const { return false; }

    // ---- Montgomery-128 --------------------------------------------------------
    // REDC(T) = T * R^{-1} mod p, canonical, for T < p*R.
    u128 redc(const u256& T) const {
        ENGINE_DEBUG_CHECK(T.hi < p);                       // T < p*R
        const u128 m = T.lo * mont128_ninv;                 // mod 2^128
        const u256 mp = detail::mul_wide128(m, p);
        // T + m*p == 0 mod 2^128, so the low limbs carry exactly when T.lo != 0.
        u128 t = T.hi + mp.hi + static_cast<u128>(T.lo != 0);   // < 2p < 2^128
        if (t >= p) t -= p;
        return t;
    }

    // ---- canonical primitives: domain a, b in [0, p) -----------------------------
    Elt mul(Elt a, Elt b) const { return redc(detail::mul_wide128(redc(detail::mul_wide128(a, b)), mont128_r2)); }
    Elt add(Elt a, Elt b) const {
        Elt r = a + b;                                      // < 2^128: p < 2^127
        if (r >= p) r -= p;
        return r;
    }
    Elt sub(Elt a, Elt b) const { return (a >= b) ? (a - b) : (a + (p - b)); }
    Elt neg(Elt a) const { return (a == 0) ? 0 : p - a; }

    // the reused-operand pair (the Sec. 13.5 row AXPY; see the header comment)
    Elt mul_prep(Elt w) const { return redc(detail::mul_wide128(w, mont128_r2)); }
    Elt mul_prepped(Elt x, Elt w_prepped) const { return redc(detail::mul_wide128(x, w_prepped)); }

    // Reduce an exact accumulator (T < p*R, the Sec. 5 bound) to the canonical residue.
    Elt reduce(const Accum& T) const { return redc(detail::mul_wide128(redc(T), mont128_r2)); }

    Elt canonicalise(Elt v) const { return v >= p ? v - p : v; }

    // UNIT-ONLY inverse (Sec. 13.3 / 13.4): Hensel from the inverse mod ell.
    Elt inv(Elt u) const {
        u %= p;
        if (u % ell == 0) throw std::runtime_error("Modular inverse does not exist.");
        // uint64_t inverse of u mod ell by the extended Euclidean algorithm
        const uint64_t e = static_cast<uint64_t>(ell);
        Elt x = ext_inv64(static_cast<uint64_t>(u % ell), e);
        // x <- x (2 - u x): correct mod ell^(2^j) after j steps
        for (uint32_t prec = 1; prec < k; prec *= 2) x = mul(x, sub(static_cast<Elt>(2) % p, mul(u, x)));
        ENGINE_DEBUG_CHECK(mul(u, x) == 1);
        return x;
    }

    // ---- composite arithmetic (Sec. 13.4), u128 integer % and / by ell -------
    uint32_t valuation(Elt x) const {
        if (x == 0) return k;
        uint32_t v = 0;
        while (v < k && x % ell == 0) { x /= ell; ++v; }
        return v;
    }
    Elt unit_part(Elt x) const {
        if (x == 0) return 1;
        while (x % ell == 0) x /= ell;
        return x;
    }
    Elt ell_pow(uint32_t v) const {
        ENGINE_CHECK_MSG(v <= k, "ModCtx::ell_pow: v > k");
        return ell_pow_[v];
    }
    Elt divide_exact(Elt x, Elt d) const {
        uint32_t v = 0;
        while (v <= k && ell_pow_[v] != d) ++v;
        ENGINE_CHECK_MSG(v <= k, "ModCtx::divide_exact: d is not a power ell^v with v <= k");
        ENGINE_CHECK_MSG(v <= valuation(x), "ModCtx::divide_exact: ell^v does not divide x");
        return x / d;
    }

private:
    // a^{-1} mod m for gcd(a, m) = 1, 0 < a < m < 2^64 (signed __int128 Bezout
    // coefficients; |s| <= m keeps them in range).
    static u128 ext_inv64(uint64_t a, uint64_t m) {
        __int128 r0 = m, r1 = a, s0 = 0, s1 = 1;
        while (r1 != 0) {
            const __int128 q = r0 / r1;
            __int128 t = r0 - q * r1; r0 = r1; r1 = t;
            t = s0 - q * s1; s0 = s1; s1 = t;
        }
        if (r0 != 1) throw std::runtime_error("Modular inverse does not exist.");
        if (s0 < 0) s0 += m;
        return static_cast<u128>(s0);
    }

    std::array<u128, kPowTable> ell_pow_{};
};

using ModCtx128 = ModCtx<u128>;

// Sec. 13.12.2: trivially copyable, as for the other two element types.
static_assert(std::is_trivially_copyable<ModCtx128>::value, "ModCtx<u128> must stay trivially copyable");

} // namespace engine
