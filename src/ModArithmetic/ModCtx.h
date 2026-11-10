// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// ModCtx.h -- the modulus context (matrix-engine Phase 2, plan D2)
// =====================================================================
//
// One ModCtx<Elt> is constructed per modulus p and passed BY REFERENCE to
// every kernel (docs/matrix-engine/00-implementation-plan.md, D2). It holds
// the constants the fast reductions need and the delayed-accumulation bound
// k_max, all computed at runtime from p -- nothing is hard-coded.
//
//   Elt = uint32_t   p in [2, 2^32)     accumulator uint64_t
//   Elt = uint64_t   p in [2, 2^64)     accumulator unsigned __int128
//
// EVERY canonical primitive here returns, on every input in its documented
// domain, exactly the value the legacy ModArithmetic::mod_*  /  mod_*64
// primitive returns. That is a definition, not a target: the arithmetic is
// exact integer arithmetic and the canonical residue in [0, p) is unique.
// ModContextValidation checks it against the legacy functions sample by
// sample at every regime boundary.
//
// Regimes (D2 table; the row 2^31 <= p < 2^32 is ALLOWED since Phase 0 fixed
// mod_inv / mod_add):
//
//   U32_Delayed    p < 2^30.5  (k_max >= 8)  u32 x u32 -> u64, delayed
//                  reduction, k_max = floor((2^64-1)/(p-1)^2)
//   U32_Shoup      2^30.5 <= p <= 2^31      k_max in [4, 8): Shoup for the
//                  reused operand (axpy / pivot factor); GEMM still k-blocks
//   U32_Barrett64  2^31 < p < 2^32           Shoup-32 unavailable (2p >= 2^32);
//                  eager Barrett-64, k_max in [1, 4]
//   U64_Montgomery 2^32 <= p < 2^63, p odd   64-bit Montgomery available
//   U64_Wide       every other uint64_t p   exact __int128 product + %
//
// The u32 GEMM path never needs the regime tag: it k-blocks at k_max in all
// three u32 regimes (a k-block of 4 terms still costs one Barrett per 4 MACs,
// which is cheaper than any eager scheme). The tag is informational and
// selects Shoup vs Barrett in the scalar-broadcast kernels (axpy).
//
// Wide arithmetic uses `unsigned __int128` (already the tree's convention in
// ModArithmetic.cpp for the S13 backend); MSVC is not a target of this layer.
//
// THE DECLARED RING (ENGINE_API.md Sec. 13.1, hecke-extensions slot 8 WP-1 /
// E8 item 2).  A context is either the FIELD F_p -- ModCtx(p), p prime by
// precondition, is_field() == true, ell == p, k == 1 -- or the chain ring
// Z/ell^k -- ModCtx::prime_power(ell, k), k >= 2, is_field() == false, the
// ONLY way to obtain a composite context.  The ring is declared, never
// inferred.  Every Sec. 1 constant (p2, barrett_m, mont_*, k_max, regime) is
// computed by the SAME init(p) for both, so the modulus-only layer (Sec. 13.2)
// sees nothing new; the new members are written by both constructors and are
// read only by the Sec. 13.3 guards and the Sec. 13.4/13.6 composite entry
// points -- never on a value path of the field route (Sec. 13.8 item 2).
// ModCtx stays TRIVIALLY COPYABLE (it is copied into every BlockMat, Program
// and tape Bindings): the ell^v table is a fixed std::array, never a vector
// (Sec. 13.1, review finding m6).

#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "EngineCheck.h"
#include "ModArithmetic.h"

namespace engine {

using u128 = unsigned __int128;

enum class Regime : uint8_t {
    U32_Delayed,
    U32_Shoup,
    U32_Barrett64,
    U64_Montgomery,
    U64_Wide,
    U128_Montgomery   // (slot-11 WP-18a) ModCtx<u128> only, ModCtxU128.h
};

inline const char* regime_name(Regime r) {
    switch (r) {
        case Regime::U32_Delayed:    return "U32_Delayed";
        case Regime::U32_Shoup:      return "U32_Shoup";
        case Regime::U32_Barrett64:  return "U32_Barrett64";
        case Regime::U64_Montgomery: return "U64_Montgomery";
        case Regime::U64_Wide:       return "U64_Wide";
        case Regime::U128_Montgomery: return "U128_Montgomery";
    }
    return "?";
}

// Zero-handling policy of batch_inv (Montgomery's trick).
//   Throw     -- any zero input throws std::runtime_error("Modular inverse
//                does not exist."), the legacy mod_inv behaviour. Default.
//   MapToZero -- zeros are skipped in the prefix products and receive 0 in
//                the output; every non-zero entry is inverted exactly.
enum class ZeroPolicy : uint8_t { Throw, MapToZero };

template<class Elt> struct ModTraits;
template<> struct ModTraits<uint32_t> {
    using Accum = uint64_t;          // exact accumulator of delayed sums
    static constexpr int bits = 32;
};
template<> struct ModTraits<uint64_t> {
    using Accum = u128;
    static constexpr int bits = 64;
};

namespace detail {

inline uint64_t mulhi64(uint64_t a, uint64_t b) {
    return static_cast<uint64_t>((static_cast<u128>(a) * b) >> 64);
}

// (2^64 mod p), (2^64 / p) for 2 <= p < 2^64 without a 128-bit literal.
inline uint64_t two64_div(uint64_t p)  { return static_cast<uint64_t>((static_cast<u128>(1) << 64) / p); }
inline uint64_t two64_mod(uint64_t p)  { return static_cast<uint64_t>((static_cast<u128>(1) << 64) % p); }

} // namespace detail

template<class Elt>
struct ModCtx {
    static_assert(std::is_same<Elt, uint32_t>::value || std::is_same<Elt, uint64_t>::value,
                  "ModCtx supports uint32_t and uint64_t elements only");
    using Accum = typename ModTraits<Elt>::Accum;
    static constexpr bool is_u32 = std::is_same<Elt, uint32_t>::value;

    // ---- constants -----------------------------------------------------
    Elt      p        = 0;
    Elt      p2       = 0;      // p^2 if it fits in Elt, else 0 (see has_p2)
    bool     has_p2   = false;
    uint64_t barrett_m = 0;     // u32: floor(2^64 / p).  u64: unused (0).
    bool     has_montgomery = false;  // u64, odd p < 2^63
    uint64_t mont_ninv = 0;     // -p^{-1} mod 2^64
    uint64_t mont_r1   = 0;     // R mod p,  R = 2^64  (Montgomery form of 1)
    uint64_t mont_r2   = 0;     // R^2 mod p
    // Largest k such that k*(p-1)^2 <= 2^64-1 (u32) resp. <= 2^128-1 (u64,
    // saturated to 2^64-1). Delayed accumulation of k <= k_max products of
    // canonical residues cannot overflow Accum.  1023 at p = 134217757.
    uint64_t k_max    = 0;
    Regime   regime   = Regime::U64_Wide;

    // ---- the declared ring (ENGINE_API.md Sec. 13.1) -----------------------
    // ell: the prime (== p for a field context); k: the exponent (== 1 for a
    // field context); p == ell^k.  Written by init() / prime_power(), never
    // read on a value path of the field route.
    Elt      ell      = 0;
    uint32_t k        = 1;
    // Capacity of the ell^v table: ell >= 2 and ell^k < 2^bits force
    // k <= bits - 1, so bits entries hold ell^0 .. ell^k.
    static constexpr uint32_t kPowTable = static_cast<uint32_t>(ModTraits<Elt>::bits);

    ModCtx() = default;
    explicit ModCtx(Elt prime) { init(prime); }

    // Z/ell^k, k >= 2 (Sec. 13.1).  ENGINE_CHECKs: ell >= 2, k >= 2, and ell^k
    // fits Elt (overflow-checked product).  The primality of ell is a
    // precondition, exactly as for ModCtx(p).  k == 1 is REJECTED: F_ell has
    // exactly one representation, ModCtx(ell).  The ring members are set
    // AFTER init(p), which resets them to the field values (review m5).
    static ModCtx prime_power(Elt ell_, uint32_t k_) {
        ENGINE_CHECK_MSG(ell_ >= 2, "ModCtx::prime_power: ell must be >= 2");
        ENGINE_CHECK_MSG(k_ >= 2, "ModCtx::prime_power: k must be >= 2 (F_ell is ModCtx(ell))");
        ENGINE_CHECK_MSG(k_ < kPowTable, "ModCtx::prime_power: ell^k does not fit the element type");
        std::array<Elt, kPowTable> pw{};
        pw[0] = 1;
        for (uint32_t v = 1; v <= k_; ++v) {
            const u128 nx = static_cast<u128>(pw[v - 1]) * ell_;
            ENGINE_CHECK_MSG(nx <= static_cast<u128>(std::numeric_limits<Elt>::max()),
                             "ModCtx::prime_power: ell^k does not fit the element type");
            pw[v] = static_cast<Elt>(nx);
        }
        ModCtx c;
        c.init(pw[k_]);
        c.ell = ell_;
        c.k = k_;
        c.field_ = false;
        c.ell_pow_ = pw;
        return c;
    }

    // (S24 CL-1, CODEAUDIT C-07) Z/ell^k for every k >= 1: ModCtx(ell) at k == 1
    // (the field, is_field() == true), prime_power(ell, k) for k >= 2.  The one
    // entry point for callers that want "Z/ell^k, k >= 1" -- prime_power()'s
    // k >= 2 rule is why they wrote their own rings.  Additive: neither
    // constructor changes.  ENGINE_CHECKs: k >= 1, plus those of the two
    // routes.  (ModCtx<u128> is composite-only and has no chain(); Sec. 13.12.)
    static ModCtx chain(Elt ell_, uint32_t k_) {
        ENGINE_CHECK_MSG(k_ >= 1, "ModCtx::chain: k must be >= 1");
        if (k_ == 1) {
            ENGINE_CHECK_MSG(ell_ >= 2, "ModCtx::chain: ell must be >= 2");
            return ModCtx(ell_);
        }
        return prime_power(ell_, k_);
    }

    // true for ModCtx(p), false for prime_power(ell, k) (Sec. 13.1).
    bool is_field() const { return field_; }

    void init(Elt prime) {
        ENGINE_CHECK_MSG(prime >= 2, "ModCtx: modulus must be >= 2");
        p = prime;
        // Sec. 13.1 / review m5: init() is public and is re-entered on a live
        // context (FlatMatrix.cpp's thread-local engine_ctx cache), so it
        // resets the ring to the field F_p itself; prime_power() overrides
        // these after calling it.
        ell = prime;
        k = 1;
        field_ = true;
        ell_pow_.fill(0);
        ell_pow_[0] = 1;
        ell_pow_[1] = prime;
        if constexpr (is_u32) {
            const uint64_t pp = static_cast<uint64_t>(p) * p;
            has_p2 = pp <= std::numeric_limits<uint32_t>::max();
            p2 = has_p2 ? static_cast<Elt>(pp) : 0;
            barrett_m = detail::two64_div(p);
            const uint64_t d = static_cast<uint64_t>(p - 1) * (p - 1);   // < 2^64
            k_max = (d == 0) ? std::numeric_limits<uint64_t>::max()
                             : std::numeric_limits<uint64_t>::max() / d;
            if (k_max >= 8)                       regime = Regime::U32_Delayed;
            else if (p <= (uint32_t(1) << 31))    regime = Regime::U32_Shoup;
            else                                  regime = Regime::U32_Barrett64;
            has_montgomery = false;
        } else {
            const u128 pp = static_cast<u128>(p) * p;
            has_p2 = pp <= std::numeric_limits<uint64_t>::max();
            p2 = has_p2 ? static_cast<Elt>(pp) : 0;
            barrett_m = 0;
            const u128 d = static_cast<u128>(p - 1) * (p - 1);
            const u128 kk = (d == 0) ? ~static_cast<u128>(0) : (~static_cast<u128>(0)) / d;
            k_max = kk > std::numeric_limits<uint64_t>::max()
                        ? std::numeric_limits<uint64_t>::max() : static_cast<uint64_t>(kk);
            has_montgomery = (p & 1) && (p < (uint64_t(1) << 63));
            if (has_montgomery) {
                // Newton iteration for p^{-1} mod 2^64 (p odd): p*p == 1 mod 8,
                // and each step doubles the number of correct bits.
                uint64_t inv = p;
                for (int i = 0; i < 5; ++i) inv *= 2 - p * inv;
                mont_ninv = 0 - inv;
                mont_r1 = detail::two64_mod(p);
                mont_r2 = static_cast<uint64_t>((static_cast<u128>(mont_r1) * mont_r1) % p);
                regime = Regime::U64_Montgomery;
            } else {
                regime = Regime::U64_Wide;
            }
        }
    }

    // ---- canonical primitives (== legacy mod_*_w bit for bit) ------------
    // Domain: a, b in [0, p) unless stated otherwise.
    Elt mul(Elt a, Elt b) const {
        if constexpr (is_u32) return reduce_u64(static_cast<uint64_t>(a) * b);
        else                  return static_cast<Elt>((static_cast<u128>(a) * b) % p);
    }
    Elt add(Elt a, Elt b) const {
        Elt r = a + b;
        if (r < a || r >= p) r -= p;
        return r;
    }
    Elt sub(Elt a, Elt b) const { return (a >= b) ? (a - b) : (a + (p - b)); }
    Elt neg(Elt a) const { return (a == 0) ? 0 : p - a; }
    // inv is UNIT-ONLY, not field-only (Sec. 13.3 "what is not guarded", F1):
    // the extended Euclidean algorithm is correct on units for every modulus
    // and throws std::runtime_error("Modular inverse does not exist.") when
    // gcd(a, p) > 1 -- exactly Sec. 13.4's meaning, with no code change.
    Elt inv(Elt a) const { return ModArithmetic::mod_inv_w(a, p); }      // throws if none
    // FIELD-ONLY, guarded WHOLESALE (Sec. 13.3): mod_pow reduces the exponent
    // modulo p - 1 (Fermat), which is wrong on Z/ell^k for e >= p - 1 (review
    // m4: a no-op below p - 1, but the guard is on the entry point, not on e).
    Elt pow(Elt a, Elt e) const {
        ENGINE_CHECK_MSG(field_, "ModCtx::pow: field-only; use the Z/ell^k entry points of ENGINE_API.md Sec. 13");
        return ModArithmetic::mod_pow_w(a, e, p);
    }

    // ---- composite arithmetic (Sec. 13.4) -----------------------------------
    // Integers are the canonical representatives in [0, ell^k).  On a field
    // context these are defined and consistent with k = 1 (valuation(x) = 0
    // for x != 0, unit_part(x) = x); no field-path code calls them.
    //
    // the ell-adic valuation: the largest v <= k with ell^v | x; valuation(0) = k.
    uint32_t valuation(Elt x) const {
        if (x == 0) return k;
        uint32_t v = 0;
        while (v < k && x % ell == 0) { x /= ell; ++v; }
        return v;
    }
    // x / ell^valuation(x): the unique u in [1, ell^{k-v}) prime to ell with
    // x = ell^v * u; unit_part(0) = 1 by convention (0 = ell^k * 1).
    Elt unit_part(Elt x) const {
        if (x == 0) return 1;
        while (x % ell == 0) x /= ell;
        return x;
    }
    // ell^v as an integer for v <= k, from the table built at construction.
    // ell_pow(k) == p is NOT a residue; it is returned for bookkeeping only.
    Elt ell_pow(uint32_t v) const {
        ENGINE_CHECK_MSG(v <= k, "ModCtx::ell_pow: v > k");
        return ell_pow_[v];
    }
    // For d = ell^v with 0 <= v <= valuation(x): the integer x / d in
    // [0, ell^{k-v}), the canonical one of the ell^v solutions y of d*y = x.
    Elt divide_exact(Elt x, Elt d) const {
        uint32_t v = 0;
        while (v <= k && ell_pow_[v] != d) ++v;
        ENGINE_CHECK_MSG(v <= k, "ModCtx::divide_exact: d is not a power ell^v with v <= k");
        ENGINE_CHECK_MSG(v <= valuation(x), "ModCtx::divide_exact: ell^v does not divide x");
        return x / d;
    }

    // Reduce an exact accumulator to the canonical residue.
    Elt reduce(Accum x) const {
        if constexpr (is_u32) return reduce_u64(x);
        else                  return static_cast<Elt>(x % p);
    }

    // ---- u32: Barrett -------------------------------------------------------
    // reduce_u64(x) = x mod p for EVERY x < 2^64 (not only x < p^2).
    // q = floor(x*m / 2^64) with m = floor(2^64/p) satisfies
    // floor(x/p) - 1 <= q <= floor(x/p), so x - q*p in [0, 2p): one correction.
    Elt reduce_u64(uint64_t x) const {
        static_assert(is_u32, "reduce_u64 is a uint32_t-regime primitive");
        const uint64_t q = detail::mulhi64(x, barrett_m);
        uint64_t r = x - q * static_cast<uint64_t>(p);
        if (r >= p) r -= p;
        return static_cast<Elt>(r);
    }
    Elt mod_mul_barrett(Elt a, Elt b) const {
        static_assert(is_u32, "mod_mul_barrett is a uint32_t-regime primitive");
        return reduce_u64(static_cast<uint64_t>(a) * b);
    }

    // ---- u32: Shoup (precomputed quotient for a REUSED operand w) -----------
    // Valid only when 2p <= 2^32 (the lazy representative r in [0, 2p) must
    // fit uint32_t): ENGINE_CHECKed. `a` may be any uint32_t; w in [0, p).
    bool shoup_available() const {
        if constexpr (is_u32) return p <= (uint32_t(1) << 31);
        else return false;
    }
    Elt shoup_precompute(Elt w) const {
        static_assert(is_u32, "shoup_* are uint32_t-regime primitives");
        ENGINE_CHECK_MSG(shoup_available(), "shoup_precompute: needs 2p <= 2^32");
        return static_cast<Elt>((static_cast<uint64_t>(w) << 32) / p);
    }
    // Lazy result in [0, 2p).
    Elt shoup_mul_lazy(Elt a, Elt w, Elt w_shoup) const {
        static_assert(is_u32, "shoup_* are uint32_t-regime primitives");
        const uint32_t q = static_cast<uint32_t>((static_cast<uint64_t>(a) * w_shoup) >> 32);
        return a * w - q * p;   // exact in uint32_t arithmetic: true value < 2p <= 2^32
    }
    // Canonical result (== mul(a, w) for a in [0, p)).
    Elt shoup_mul(Elt a, Elt w, Elt w_shoup) const {
        Elt r = shoup_mul_lazy(a, w, w_shoup);
        return r >= p ? r - p : r;
    }

    // ---- u64: Montgomery (R = 2^64), odd p < 2^63 ----------------------------
    // mont64_reduce(T) = T * R^{-1} mod p for T < p*R. mont64_mul(aR, bR) = abR.
    uint64_t mont64_reduce(u128 T) const {
        static_assert(!is_u32, "mont64_* are uint64_t-regime primitives");
        ENGINE_DEBUG_CHECK(has_montgomery);
        const uint64_t m = static_cast<uint64_t>(T) * mont_ninv;
        const u128 t = (T + static_cast<u128>(m) * p) >> 64;   // < 2p, no 128-bit wrap for p < 2^63
        uint64_t r = static_cast<uint64_t>(t);
        if (r >= p) r -= p;
        return r;
    }
    uint64_t mont64_mul(uint64_t aR, uint64_t bR) const {
        return mont64_reduce(static_cast<u128>(aR) * bR);
    }
    uint64_t mont64_to(uint64_t a) const   { return mont64_mul(a, mont_r2); }   // a -> aR
    uint64_t mont64_from(uint64_t aR) const { return mont64_reduce(aR); }       // aR -> a

    // ---- the reused-operand pair (slot-11 WP-12, F-1; ENGINE_API.md Sec. 13.5) --
    // For a factor w applied to many x (a row AXPY, a row scaling, a closure
    // row): w' = mul_prep(w) once, then mul_prepped(x, w') per entry.
    //   mul_prepped(x, mul_prep(w)) == mul(x, w)   for every x, w in [0, p),
    // bit for bit: both sides are the canonical residue of x*w.  The prepared
    // operand is an OPAQUE token -- it never escapes the caller's loop, is never
    // stored, hashed or compared.
    //   U64_Montgomery (odd p < 2^63): w' = w*R mod p (canonical, R = 2^64), and
    //     mont64_reduce(x * w') = x*w*R*R^{-1} = x*w mod p, canonical, one REDC
    //     per product instead of the __umodti3 of mul() (T = x*w' < p^2 < p*R,
    //     REDC's precondition).  The Shoup analogue for u64.
    //   every other regime (all of u32; U64_Wide: even p or p >= 2^63):
    //     w' = w and mul_prepped == mul -- the u32 code is unchanged.
    Elt mul_prep(Elt w) const {
        if constexpr (!is_u32) {
            if (has_montgomery) return mont64_to(w);
        }
        return w;
    }
    Elt mul_prepped(Elt x, Elt w_prepped) const {
        if constexpr (!is_u32) {
            if (has_montgomery) return mont64_reduce(static_cast<u128>(x) * w_prepped);
        }
        return mul(x, w_prepped);
    }

    // ---- lazy [0, 2p) -> canonical ----------------------------------------------
    Elt canonicalise(Elt v) const { return v >= p ? v - p : v; }

    // ---- batched inversion (Montgomery's trick) ---------------------------------
    // out[i] = x[i]^{-1} for all i, with 3(n-1) multiplications and ONE inversion.
    // Exact: out[i] == inv(x[i]) bit for bit (the inverse is unique).  x and out
    // may alias.  Zero handling per ZeroPolicy (see above).
    // FIELD-ONLY (Sec. 13.3): ZeroPolicy treats only 0 as non-invertible.  On a
    // composite modulus a non-zero non-unit would make the running product a
    // non-unit; inv(acc) would then throw (value-safe, review m13), but the
    // entry point is guarded as policy, before any work.
    void batch_inv(const Elt* x, Elt* out, size_t n, ZeroPolicy zp = ZeroPolicy::Throw) const {
        ENGINE_CHECK_MSG(field_, "ModCtx::batch_inv: field-only; use the Z/ell^k entry points of ENGINE_API.md Sec. 13");
        if (n == 0) return;
        std::vector<Elt> prefix(n);   // prefix[i] = product of non-zero x[0..i]
        Elt acc = 1 % p;
        for (size_t i = 0; i < n; ++i) {
            if (x[i] == 0) {
                if (zp == ZeroPolicy::Throw) throw std::runtime_error("Modular inverse does not exist.");
            } else {
                acc = mul(acc, x[i]);
            }
            prefix[i] = acc;
        }
        Elt inv_acc = inv(acc);   // acc != 0 unless p divides a product of units: impossible for prime p
        for (size_t i = n; i-- > 0;) {
            const Elt xi = x[i];
            if (xi == 0) { out[i] = 0; continue; }
            const Elt before = (i == 0) ? static_cast<Elt>(1 % p) : prefix[i - 1];
            out[i] = mul(inv_acc, before);      // (x_0...x_i)^{-1} * (x_0...x_{i-1}) = x_i^{-1}
            inv_acc = mul(inv_acc, xi);         // (x_0...x_{i-1})^{-1}
        }
    }
    void batch_inv(std::vector<Elt>& x, ZeroPolicy zp = ZeroPolicy::Throw) const {
        batch_inv(x.data(), x.data(), x.size(), zp);
    }

private:
    bool field_ = true;
    std::array<Elt, kPowTable> ell_pow_{};
};

using ModCtx32 = ModCtx<uint32_t>;
using ModCtx64 = ModCtx<uint64_t>;

// Sec. 13.1 (review m6): ModCtx is copied freely (BlockMat, Program, Bindings).
static_assert(std::is_trivially_copyable<ModCtx32>::value, "ModCtx<uint32_t> must stay trivially copyable");
static_assert(std::is_trivially_copyable<ModCtx64>::value, "ModCtx<uint64_t> must stay trivially copyable");

} // namespace engine

// (hecke-extensions slot 11, WP-18a; ENGINE_API.md Sec. 13.12) the third,
// composite-only element type: a full specialisation ModCtx<u128>, so the two
// instantiations above are untouched.
#include "ModCtxU128.h"
