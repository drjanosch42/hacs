// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "ModContextValidation.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Kernels.h"
#include "ModArithmetic.h"
#include "ModCtx.h"

// See ModContextValidation.h.

namespace {

using engine::ModCtx;
using engine::u128;

struct Splitmix {
    uint64_t s;
    explicit Splitmix(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // uniform in [0, m) (m >= 1); the bias of % is irrelevant for a test
    uint64_t below(uint64_t m) { return m <= 1 ? 0 : next() % m; }
};

uint32_t g_fail = 0;

void fail(const std::string& what) {
    ++g_fail;
    if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
}

template<class T> std::string str(T v) { return std::to_string(v); }

// legacy oracles, width-dispatched
inline uint32_t L_mul(uint32_t a, uint32_t b, uint32_t p) { return ModArithmetic::mod_mul(a, b, p); }
inline uint64_t L_mul(uint64_t a, uint64_t b, uint64_t p) { return ModArithmetic::mod_mul64(a, b, p); }
inline uint32_t L_add(uint32_t a, uint32_t b, uint32_t p) { return ModArithmetic::mod_add(a, b, p); }
inline uint64_t L_add(uint64_t a, uint64_t b, uint64_t p) { return ModArithmetic::mod_add64(a, b, p); }
inline uint32_t L_sub(uint32_t a, uint32_t b, uint32_t p) { return ModArithmetic::mod_sub(a, b, p); }
inline uint64_t L_sub(uint64_t a, uint64_t b, uint64_t p) { return ModArithmetic::mod_sub64(a, b, p); }
inline uint32_t L_neg(uint32_t a, uint32_t p) { return ModArithmetic::mod_minus(a, p); }
inline uint64_t L_neg(uint64_t a, uint64_t p) { return ModArithmetic::mod_minus64(a, p); }
inline uint32_t L_inv(uint32_t a, uint32_t p) { return ModArithmetic::mod_inv(a, p); }
inline uint64_t L_inv(uint64_t a, uint64_t p) { return ModArithmetic::mod_inv64(a, p); }
inline uint32_t L_pow(uint32_t a, uint32_t e, uint32_t p) { return ModArithmetic::mod_pow(a, e, p); }
inline uint64_t L_pow(uint64_t a, uint64_t e, uint64_t p) { return ModArithmetic::mod_pow64(a, e, p); }

template<class Elt>
std::vector<Elt> corners(Elt p) {
    std::vector<Elt> c = {Elt(0), Elt(1) % p, Elt(p - 1)};
    if (p > 2) c.push_back(Elt(p - 2));
    c.push_back(Elt(p / 2));
    if (p > 3) c.push_back(Elt(p / 2 + 1));
    return c;
}

template<class Elt>
void check_pair(const ModCtx<Elt>& ctx, Elt a, Elt b, const char* tag) {
    const Elt p = ctx.p;
    if (ctx.mul(a, b) != L_mul(a, b, p)) fail(std::string(tag) + " mul p=" + str(p) + " a=" + str(a) + " b=" + str(b));
    if (ctx.add(a, b) != L_add(a, b, p)) fail(std::string(tag) + " add p=" + str(p) + " a=" + str(a) + " b=" + str(b));
    if (ctx.sub(a, b) != L_sub(a, b, p)) fail(std::string(tag) + " sub p=" + str(p) + " a=" + str(a) + " b=" + str(b));
    if (ctx.neg(a) != L_neg(a, p))       fail(std::string(tag) + " neg p=" + str(p) + " a=" + str(a));
    if constexpr (std::is_same<Elt, uint32_t>::value) {
        if (ctx.mod_mul_barrett(a, b) != L_mul(a, b, p))
            fail(std::string(tag) + " barrett p=" + str(p) + " a=" + str(a) + " b=" + str(b));
        if (ctx.shoup_available()) {
            const uint32_t bs = ctx.shoup_precompute(b);
            const uint32_t lazy = ctx.shoup_mul_lazy(a, b, bs);
            if (lazy >= 2 * static_cast<uint64_t>(p))
                fail(std::string(tag) + " shoup_lazy range p=" + str(p) + " a=" + str(a) + " b=" + str(b));
            if (ctx.canonicalise(lazy) != L_mul(a, b, p) || ctx.shoup_mul(a, b, bs) != L_mul(a, b, p))
                fail(std::string(tag) + " shoup p=" + str(p) + " a=" + str(a) + " b=" + str(b));
        }
    } else {
        if (ctx.has_montgomery) {
            const uint64_t aR = ctx.mont64_to(a), bR = ctx.mont64_to(b);
            if (ctx.mont64_from(aR) != a) fail(std::string(tag) + " mont round trip p=" + str(p) + " a=" + str(a));
            if (ctx.mont64_from(ctx.mont64_mul(aR, bR)) != L_mul(a, b, p))
                fail(std::string(tag) + " mont mul p=" + str(p) + " a=" + str(a) + " b=" + str(b));
        }
        // reduce() of a wide accumulator: a*b + a
        const u128 T = static_cast<u128>(a) * b + a;
        if (ctx.reduce(T) != static_cast<uint64_t>(T % p))
            fail(std::string(tag) + " reduce128 p=" + str(p) + " a=" + str(a) + " b=" + str(b));
    }
}

template<class Elt>
void validate_prime(Elt p, uint64_t n_samples, engine::Regime expected_regime) {
    ModCtx<Elt> ctx(p);
    std::cout << "  p = " << str(p) << "  regime " << engine::regime_name(ctx.regime)
              << "  k_max " << ctx.k_max << std::endl;
    if (ctx.regime != expected_regime)
        fail("regime p=" + str(p) + " got " + engine::regime_name(ctx.regime) +
             " expected " + engine::regime_name(expected_regime));
    if (ctx.p != p) fail("ctx.p");
    // ENGINE_API.md Sec. 13.1: ModCtx(p) declares the field F_p.
    if (!ctx.is_field() || ctx.ell != p || ctx.k != 1 || ctx.ell_pow(0) != 1 || ctx.ell_pow(1) != p)
        fail("ModCtx(p) must declare the field: is_field, ell == p, k == 1, p=" + str(p));
    {
        const u128 pp = static_cast<u128>(p) * p;
        const bool fits = pp <= static_cast<u128>(std::numeric_limits<Elt>::max());
        if (ctx.has_p2 != fits || (fits && ctx.p2 != static_cast<Elt>(pp))) fail("p2 p=" + str(p));
    }

    Splitmix rng(0xC0FFEE0000ULL + static_cast<uint64_t>(p));

    // --- corners x corners ---
    const auto cs = corners<Elt>(p);
    for (Elt a : cs) for (Elt b : cs) check_pair(ctx, a, b, "corner");

    // --- random pairs ---
    for (uint64_t t = 0; t < n_samples; ++t) {
        const Elt a = static_cast<Elt>(rng.below(p)), b = static_cast<Elt>(rng.below(p));
        check_pair(ctx, a, b, "random");
    }

    // --- inv / pow (legacy inv is an extended Euclid: fewer samples) ---
    for (uint64_t t = 0; t < n_samples / 10 + 8; ++t) {
        Elt a = static_cast<Elt>(rng.below(p));
        if (t < cs.size()) a = cs[t];
        if (a != 0 && ctx.inv(a) != L_inv(a, p)) fail("inv p=" + str(p) + " a=" + str(a));
        const Elt e = static_cast<Elt>(rng.next());
        if (ctx.pow(a, e) != L_pow(a, e, p)) fail("pow p=" + str(p) + " a=" + str(a) + " e=" + str(e));
    }
    bool threw = false;
    try { (void)ctx.inv(0); } catch (const std::runtime_error&) { threw = true; }
    if (!threw) fail("inv(0) must throw p=" + str(p));

    // --- reduce_u64 over the FULL uint64_t range (u32) ---
    if constexpr (std::is_same<Elt, uint32_t>::value) {
        const uint64_t specials[] = {0ull, 1ull, p - 1ull, p, p + 1ull, static_cast<uint64_t>(p) * p,
                                     static_cast<uint64_t>(p) * p - 1ull, ~0ull, ~0ull - 1ull,
                                     (~0ull) - (~0ull) % p, 1ull << 63, (1ull << 63) - 1ull};
        for (uint64_t x : specials)
            if (ctx.reduce_u64(x) != x % p) fail("reduce_u64 special p=" + str(p) + " x=" + str(x));
        for (uint64_t t = 0; t < n_samples; ++t) {
            const uint64_t x = rng.next();
            if (ctx.reduce_u64(x) != x % p) fail("reduce_u64 p=" + str(p) + " x=" + str(x));
            if (ctx.reduce(x) != x % p) fail("reduce p=" + str(p) + " x=" + str(x));
        }
        // shoup availability vs the ENGINE_CHECK
        const bool avail = (p <= (uint32_t(1) << 31));
        if (ctx.shoup_available() != avail) fail("shoup_available p=" + str(p));
        if (!avail) {
            bool caught = false;
            try { (void)ctx.shoup_precompute(1); } catch (const std::logic_error&) { caught = true; }
            if (!caught) fail("shoup_precompute must ENGINE_CHECK for 2p > 2^32, p=" + str(p));
        }
        // canonicalise on [0, 2p) when representable
        if (avail) {
            for (uint32_t t = 0; t < 1000; ++t) {
                const uint32_t v = static_cast<uint32_t>(rng.below(2ull * p));
                if (ctx.canonicalise(v) != v % p) fail("canonicalise p=" + str(p) + " v=" + str(v));
            }
        }
    } else {
        const bool mont = (p & 1) && (p < (uint64_t(1) << 63));
        if (ctx.has_montgomery != mont) fail("has_montgomery p=" + str(p));
        if (mont) {
            if (ctx.mont64_to(1) != ctx.mont_r1) fail("mont_r1 p=" + str(p));
            if (ctx.mont64_from(ctx.mont_r1) != 1) fail("mont from(R) p=" + str(p));
            if (static_cast<uint64_t>(p * ctx.mont_ninv) != ~0ull) fail("mont_ninv p=" + str(p));
            if (ctx.mont_r2 != static_cast<uint64_t>((static_cast<u128>(ctx.mont_r1) * ctx.mont_r1) % p))
                fail("mont_r2 p=" + str(p));
        }
    }

    // --- batch_inv ---
    for (uint32_t len : {1u, 2u, 3u, 17u, 100u}) {
        std::vector<Elt> x(len);
        for (auto& v : x) { do { v = static_cast<Elt>(rng.below(p)); } while (v == 0); }
        std::vector<Elt> out(len);
        ctx.batch_inv(x.data(), out.data(), len);
        for (uint32_t i = 0; i < len; ++i)
            if (out[i] != L_inv(x[i], p)) fail("batch_inv p=" + str(p) + " len=" + str(len) + " i=" + str(i));
        // in place
        std::vector<Elt> y = x;
        ctx.batch_inv(y);
        if (y != out) fail("batch_inv in place p=" + str(p) + " len=" + str(len));
        // with zeros
        std::vector<Elt> z = x;
        z[len / 2] = 0;
        bool caught = false;
        try { ctx.batch_inv(z.data(), out.data(), len); } catch (const std::runtime_error&) { caught = true; }
        if (!caught) fail("batch_inv Throw policy p=" + str(p));
        ctx.batch_inv(z.data(), out.data(), len, engine::ZeroPolicy::MapToZero);
        for (uint32_t i = 0; i < len; ++i) {
            const Elt want = (z[i] == 0) ? Elt(0) : L_inv(z[i], p);
            if (out[i] != want) fail("batch_inv MapToZero p=" + str(p) + " i=" + str(i));
        }
    }

    // --- k_max: tight bound and the overflow guard ---
    {
        const u128 d = static_cast<u128>(p - 1) * (p - 1);
        const u128 kmax = ctx.k_max;
        if constexpr (std::is_same<Elt, uint32_t>::value) {
            const u128 limit = ~static_cast<uint64_t>(0);
            if (kmax * d > limit) fail("k_max*(p-1)^2 overflows u64, p=" + str(p));
            if ((kmax + 1) * d <= limit) fail("k_max not tight (k_max+1 still fits), p=" + str(p));
        } else {
            // limit = 2^128 - 1: k*d <= limit  <=>  k <= limit / d
            const u128 limit = ~static_cast<u128>(0);
            if (d != 0 && kmax > limit / d) fail("k_max*(p-1)^2 overflows u128, p=" + str(p));
            if (ctx.k_max != ~0ull && d != 0 && (kmax + 1) <= limit / d)
                fail("k_max not tight (k_max+1 still fits), p=" + str(p));
        }
        if (ctx.k_max <= (1u << 20)) {
            const size_t k = static_cast<size_t>(ctx.k_max);
            std::vector<Elt> a(k + 1, Elt(p - 1)), b(k + 1, Elt(p - 1));
            // expected: k * (p-1)^2 mod p, via the legacy primitives
            const Elt kmod = static_cast<Elt>(ctx.k_max % p);
            const Elt want = L_mul(L_mul(kmod, Elt(p - 1), p), Elt(p - 1), p);
            if (engine::kernels::dot_delayed<Elt>(ctx, a.data(), b.data(), k) != want)
                fail("dot_delayed at k = k_max inexact, p=" + str(p));
            bool caught = false;
            try { (void)engine::kernels::dot_delayed<Elt>(ctx, a.data(), b.data(), k + 1); }
            catch (const std::logic_error&) { caught = true; }
            if (!caught) fail("dot_delayed at k = k_max + 1 must ENGINE_CHECK, p=" + str(p));
        }
    }
}

// ---------------------------------------------------------------------------
// hecke-extensions slot 8 WP-1 (E8 item 2): the COMPOSITE arms,
// ModCtx::prime_power(ell, k) = Z/ell^k (ENGINE_API.md Sec. 13.1-13.4).  The
// modulus-only primitives (Sec. 13.2) must equal the legacy mod_* functions on
// the composite modulus sample by sample, the Sec. 1 constants must be the
// ones init(ell^k) computes, the k_max bound must stay tight; pow and
// batch_inv must refuse (Sec. 13.3), inv must stay unit-only (F1).
// ---------------------------------------------------------------------------
template<class Elt>
void validate_composite(Elt ell, uint32_t k, uint64_t n_samples, engine::Regime expected_regime) {
    const ModCtx<Elt> ctx = ModCtx<Elt>::prime_power(ell, k);
    const Elt p = ctx.p;
    std::cout << "  Z/" << str(ell) << "^" << k << " = " << str(p) << "  regime " << engine::regime_name(ctx.regime)
              << "  k_max " << ctx.k_max << std::endl;
    if (ctx.is_field() || ctx.ell != ell || ctx.k != k) fail("prime_power ring members ell=" + str(ell) + " k=" + str(k));
    if (ctx.regime != expected_regime) fail("prime_power regime p=" + str(p));
    {
        ModCtx<Elt> bare;
        bare.init(p);
        if (bare.p2 != ctx.p2 || bare.has_p2 != ctx.has_p2 || bare.barrett_m != ctx.barrett_m || bare.k_max != ctx.k_max ||
            bare.has_montgomery != ctx.has_montgomery || bare.mont_ninv != ctx.mont_ninv || bare.mont_r1 != ctx.mont_r1 ||
            bare.mont_r2 != ctx.mont_r2 || bare.regime != ctx.regime)
            fail("prime_power: Sec. 1 constants differ from init(ell^k), p=" + str(p));
        ModCtx<Elt> reset = ctx;            // review m5: init() resets the ring
        reset.init(Elt(5));
        if (!reset.is_field() || reset.ell != 5 || reset.k != 1) fail("init(5) on a composite context must reset the ring");
    }
    Splitmix rng(0xC0FFEE1000ULL + static_cast<uint64_t>(p));
    const auto cs = corners<Elt>(p);
    for (Elt a : cs) for (Elt b : cs) check_pair(ctx, a, b, "composite corner");
    for (uint64_t t = 0; t < n_samples; ++t) {
        const Elt a = static_cast<Elt>(rng.below(p)), b = static_cast<Elt>(rng.below(p));
        check_pair(ctx, a, b, "composite random");
    }
    if constexpr (std::is_same<Elt, uint32_t>::value) {
        for (uint64_t t = 0; t < n_samples; ++t) {
            const uint64_t x = rng.next();
            if (ctx.reduce_u64(x) != x % p) { fail("composite reduce_u64 p=" + str(p)); break; }
        }
    }
    // inv: unit-only (F1) -- equal to the legacy extended Euclid on units, the
    // legacy runtime_error on non-units
    for (uint64_t t = 0; t < n_samples / 100 + 8; ++t) {
        const Elt a = static_cast<Elt>(rng.below(p));
        if (a % ell != 0) {
            if (ctx.inv(a) != L_inv(a, p) || ctx.mul(a, ctx.inv(a)) != 1) fail("composite inv on a unit p=" + str(p));
        } else {
            bool threw = false;
            try { (void)ctx.inv(a); } catch (const std::runtime_error&) { threw = true; }
            if (!threw) fail("composite inv on a non-unit must throw runtime_error p=" + str(p) + " a=" + str(a));
        }
    }
    // pow and batch_inv refuse (Sec. 13.3), for EVERY exponent
    for (Elt e : {Elt(0), Elt(1), Elt(2), Elt(p - 1), Elt(p)}) {
        bool caught = false;
        try { (void)ctx.pow(Elt(1), e); } catch (const std::logic_error&) { caught = true; }
        if (!caught) fail("composite pow must ENGINE_CHECK p=" + str(p) + " e=" + str(e));
    }
    {
        std::vector<Elt> x = {Elt(1), Elt(1)};
        bool caught = false;
        try { ctx.batch_inv(x); } catch (const std::logic_error&) { caught = true; }
        if (!caught) fail("composite batch_inv must ENGINE_CHECK p=" + str(p));
    }
    // k_max tight and the dot_delayed guard, exactly as for a prime
    {
        const u128 d = static_cast<u128>(p - 1) * (p - 1);
        const u128 kmax = ctx.k_max;
        if constexpr (std::is_same<Elt, uint32_t>::value) {
            const u128 limit = ~static_cast<uint64_t>(0);
            if (kmax * d > limit || (kmax + 1) * d <= limit) fail("composite k_max not tight p=" + str(p));
        } else {
            const u128 limit = ~static_cast<u128>(0);
            if (kmax > limit / d) fail("composite k_max overflows p=" + str(p));
            if (ctx.k_max != ~0ull && (kmax + 1) <= limit / d) fail("composite k_max not tight p=" + str(p));
        }
        if (ctx.k_max <= (1u << 20)) {
            const size_t kk = static_cast<size_t>(ctx.k_max);
            std::vector<Elt> a(kk + 1, Elt(p - 1)), b(kk + 1, Elt(p - 1));
            const Elt kmod = static_cast<Elt>(ctx.k_max % p);
            const Elt want = L_mul(L_mul(kmod, Elt(p - 1), p), Elt(p - 1), p);
            if (engine::kernels::dot_delayed<Elt>(ctx, a.data(), b.data(), kk) != want)
                fail("composite dot_delayed at k = k_max inexact, p=" + str(p));
        }
    }
}

} // namespace

uint32_t ModContextValidation() {
    g_fail = 0;
    std::cout << "ModContextValidation: ModCtx primitives vs legacy mod_* / mod_*64" << std::endl;
    using engine::Regime;

    // construction guards
    for (uint32_t bad : {0u, 1u}) {
        bool caught = false;
        try { engine::ModCtx32 c(bad); (void)c; } catch (const std::logic_error&) { caught = true; }
        if (!caught) fail("ModCtx(" + str(bad) + ") must ENGINE_CHECK");
    }

    const uint64_t N = 1000000;   // ~1e6 fixed-seed samples per prime

    // uint32_t regime boundaries
    validate_prime<uint32_t>(2u, N, Regime::U32_Delayed);
    validate_prime<uint32_t>(3u, N, Regime::U32_Delayed);
    validate_prime<uint32_t>(5u, N, Regime::U32_Delayed);
    validate_prime<uint32_t>(134217757u, N, Regime::U32_Delayed);       // k_max 1023
    validate_prime<uint32_t>(1518500213u, N, Regime::U32_Delayed);      // k_max 8, last Delayed below 2^30.5
    validate_prime<uint32_t>(1518500279u, N, Regime::U32_Shoup);        // k_max 7, first prime above 2^30.5
    validate_prime<uint32_t>(2147483647u, N, Regime::U32_Shoup);        // 2^31-1, k_max 4
    validate_prime<uint32_t>(2147483659u, N, Regime::U32_Barrett64);    // 2^31+11, k_max 3
    validate_prime<uint32_t>(4294967291u, N, Regime::U32_Barrett64);    // 2^32-5,  k_max 1
    {
        engine::ModCtx32 c(134217757u);
        if (c.k_max != 1023) fail("k_max(134217757) must be 1023, got " + str(c.k_max));
        engine::ModCtx32 c2(2147483647u);
        if (c2.k_max < 4) fail("k_max(2^31-1) must be >= 4");
    }

    // uint64_t
    validate_prime<uint64_t>(2ull, N, Regime::U64_Wide);
    validate_prime<uint64_t>(3ull, N, Regime::U64_Montgomery);
    validate_prime<uint64_t>(5ull, N, Regime::U64_Montgomery);
    validate_prime<uint64_t>(134217757ull, N, Regime::U64_Montgomery);
    validate_prime<uint64_t>(4294967291ull, N, Regime::U64_Montgomery);          // 2^32-5
    validate_prime<uint64_t>(4294967311ull, N, Regime::U64_Montgomery);          // 2^32+15
    validate_prime<uint64_t>(2305843009213693951ull, N, Regime::U64_Montgomery); // 2^61-1, k_max 64
    validate_prime<uint64_t>(9223372036854775783ull, N, Regime::U64_Montgomery); // 2^63-25, k_max 4

    // hecke-extensions slot 8 WP-1: the composite arms (ENGINE_API.md Sec. 13)
    std::cout << "  sizeof(ModCtx<uint32_t>) = " << sizeof(engine::ModCtx32)
              << ", sizeof(ModCtx<uint64_t>) = " << sizeof(engine::ModCtx64) << std::endl;
    validate_composite<uint32_t>(2u, 2, N, Regime::U32_Delayed);          // Z/4
    validate_composite<uint32_t>(5u, 3, N, Regime::U32_Delayed);          // Z/125
    validate_composite<uint32_t>(691u, 2, N, Regime::U32_Delayed);        // Z/691^2
    validate_composite<uint32_t>(3u, 20, N, Regime::U32_Barrett64);       // 3^20 ~ 2^31.7: eager Barrett-64
    validate_composite<uint32_t>(2u, 31, N, Regime::U32_Shoup);           // 2^31: the Shoup edge
    validate_composite<uint64_t>(134217757ull, 2, N, Regime::U64_Montgomery);
    validate_composite<uint64_t>(2ull, 40, N, Regime::U64_Wide);          // even modulus: exact __int128
    validate_composite<uint64_t>(3ull, 40, N, Regime::U64_Wide);          // 3^40 > 2^63: U64_Wide
    for (uint32_t bad_k : {0u, 1u}) {
        bool caught = false;
        try { (void)engine::ModCtx32::prime_power(5u, bad_k); } catch (const std::logic_error&) { caught = true; }
        if (!caught) fail("prime_power(5, " + str(bad_k) + ") must ENGINE_CHECK");
    }
    {
        bool caught = false;
        try { (void)engine::ModCtx32::prime_power(2u, 32); } catch (const std::logic_error&) { caught = true; }
        if (!caught) fail("prime_power<u32>(2, 32) must ENGINE_CHECK (overflow)");
    }

    std::cout << "ModContextValidation: " << (g_fail == 0 ? "PASS" : "FAIL") << " (" << g_fail << " failing checks)" << std::endl;
    return g_fail;
}
