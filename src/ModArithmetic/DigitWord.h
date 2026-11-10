// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// DigitWord.h -- the digit-width abstraction engine::DigitWord<W>
// (hecke-extensions slot 11, WP-18a; ENGINE_API.md Sec. 13.11 / 13.12,
//  the owner requirement of 2026-09-25: the (C2) induction is written ONCE,
//  generic over the digit word W in {uint32_t, uint64_t, u128})
// =====================================================================
//
// One trait per native word width, all with the SAME static interface, so
// that WP-18b's engine::ladic code is a template over W and nothing else:
//
//   using Word  = W;                       the digit word
//   using Ctx   = ModCtx<W>;               its context type
//   using Wide  = ...;                     the exact double-word (uint64_t,
//                                          u128, u256) -- informational
//   static constexpr uint32_t bits;        32 / 64 / 128
//   static constexpr uint32_t digit_bits;  bits - 1: every digit modulus
//                                          ell^m is < 2^{bits-1} (the
//                                          Montgomery bound of the width)
//
//   digit_exponent(ell)   m(W, ell): the largest m with ell^m < 2^{bits-1};
//                         0 when ell^2 >= 2^{bits-1} (the width cannot carry
//                         a composite digit for this prime -- e.g. W = u32 at
//                         ell = 134217757).  ell >= 2.
//   available(ell)        digit_exponent(ell) >= 2.
//   pow(ell, e)           ell^e as a W, ENGINE_CHECKed to be < 2^bits.
//   fits(ell, e)          ell^e < 2^bits (the native reach of W, which is
//                         < 2^127 for u128 -- see ctx()).
//   ctx(ell, e)           the composite context Z/ell^e of this width,
//                         ModCtx<W>::prime_power(ell, e), e >= 2 (the Sec. 13.6
//                         entry points refuse a field context).  For W = u128
//                         also ell odd and ell^e < 2^127 (ModCtxU128.h).
//
//   mul_wide(a, b, hi, lo)      the exact 2w-bit product a*b = hi*2^w + lo.
//   split(hi, lo, M, r)         (hi*2^w + lo) = q*M + r, 0 <= r < M; returns q.
//                               Precondition hi < M (so q < 2^w), M >= 1.
//   mac_split(a, b, c, M, r)    a*b + c = q*M + r, 0 <= r < M; returns q.
//                               Precondition a, b, c < M (so q < M) -- the
//                               digit-convolution step with carry by M = ell^m.
//   widen(x) / narrow(x)        W <-> u128 (narrow ENGINE_CHECKs the range):
//                               the oracle bridges between widths.
//
//   The word entry points, forwarding to the shipped engine (every one takes
//   a COMPOSITE Ctx; they are the Sec. 6 / Sec. 13.6 functions verbatim, so
//   "L = 1 digit" IS the direct route):
//     gemm(ctx, A, B, C, op)                     kernels::gemm<W>
//     howell(ctx, A)                             elim::howell<W>
//     kernel_module(ctx, A)                      elim::kernel_module<W>
//     image_module(ctx, A)                       elim::image_module<W>
//     solve_multi_rhs_module(ctx, B, Y)          elim::solve_multi_rhs_module<W>
//     length(ctx, A) / length(ctx, H)            elim::length<W>
//     invariant_factors(ctx, G, Rl) / (ctx, A)   elim::invariant_factors<W>
//     cyclic_decomposition(ctx, G, Rl)           elim::cyclic_decomposition<W>
//     cyclic_coordinates(ctx, d, Y)              elim::cyclic_coordinates<W>
//
// Costs: split / mac_split are one native division for W = uint32_t
// (uint64_t / uint32_t) and W = uint64_t (u128 / uint64_t, __udivti3); for
// W = u128 a normalised long division on 64-bit halves (Knuth D, two quotient
// digits; since f90a9ee), with the 128-step restoring loop kept as
// split_restoring, its oracle (ENGINE_API.md Sec. 13.11.6).
//
// This header is the whole of the abstraction; it adds no value path, no
// consumer and no settings key.

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "DenseElim.h"
#include "DenseView.h"
#include "EngineCheck.h"
#include "Kernels.h"
#include "ModCtx.h"

namespace engine {

namespace detail {

// The operations common to every width: the digit exponent, the context,
// and the forwarding entry points.
template<class W, class Derived>
struct DigitWordBase {
    using Word = W;
    using Ctx  = ModCtx<W>;
    static constexpr uint32_t bits       = static_cast<uint32_t>(ModTraits<W>::bits);
    static constexpr uint32_t digit_bits = bits - 1;

    // ell^e < 2^bits ?  (overflow-safe in u128 for every width)
    static bool fits(uint64_t ell, uint32_t e) {
        ENGINE_CHECK_MSG(ell >= 2, "DigitWord::fits: ell must be >= 2");
        const u128 lim_minus_1 = (bits >= 128) ? ~static_cast<u128>(0)
                                               : ((static_cast<u128>(1) << bits) - 1);
        u128 x = 1;
        for (uint32_t i = 0; i < e; ++i) {
            if (x > lim_minus_1 / ell) return false;
            x *= ell;
        }
        return true;
    }
    static W pow(uint64_t ell, uint32_t e) {
        ENGINE_CHECK_MSG(fits(ell, e), "DigitWord::pow: ell^e does not fit the word");
        u128 x = 1;
        for (uint32_t i = 0; i < e; ++i) x *= ell;
        return static_cast<W>(x);
    }
    // m(W, ell): the largest m with ell^m < 2^{bits-1}; 0 if ell^2 >= 2^{bits-1}.
    static uint32_t digit_exponent(uint64_t ell) {
        ENGINE_CHECK_MSG(ell >= 2, "DigitWord::digit_exponent: ell must be >= 2");
        const u128 lim = static_cast<u128>(1) << digit_bits;   // 2^{bits-1} <= 2^127
        u128 x = 1;
        uint32_t m = 0;
        while (x <= (lim - 1) / ell) { x *= ell; ++m; }        // x * ell < lim
        return m >= 2 ? m : 0;
    }
    static bool available(uint64_t ell) { return digit_exponent(ell) >= 2; }
    static Ctx ctx(uint64_t ell, uint32_t e) {
        ENGINE_CHECK_MSG(e >= 2, "DigitWord::ctx: e must be >= 2 (the Sec. 13.6 entry points are composite-only)");
        ENGINE_CHECK_MSG(fits(ell, e), "DigitWord::ctx: ell^e does not fit the word");
        return Ctx::prime_power(static_cast<W>(ell), e);
    }

    // the oracle bridges
    static u128 widen(W x) { return static_cast<u128>(x); }
    static W narrow(u128 x) {
        ENGINE_CHECK_MSG(bits >= 128 || x <= static_cast<u128>(std::numeric_limits<W>::max()),
                         "DigitWord::narrow: value does not fit the word");
        return static_cast<W>(x);
    }

    // a*b + c = q*M + r (a, b, c < M): the digit-convolution step.
    static W mac_split(W a, W b, W c, W M, W& r) {
        ENGINE_DEBUG_CHECK(a < M && b < M && c < M);
        W hi, lo;
        Derived::mul_wide(a, b, hi, lo);
        const W lo2 = static_cast<W>(lo + c);
        if (lo2 < lo) ++hi;                                     // carry; hi*2^w + lo2 < M^2 keeps hi < M
        return Derived::split(hi, lo2, M, r);
    }

    // ---- the word entry points (Sec. 6 / Sec. 13.6), forwarded ----------------
    static void gemm(const Ctx& c, DenseView<const W> A, DenseView<const W> B, DenseView<W> C,
                     kernels::GemmOp op = kernels::GemmOp::Set) {
        kernels::gemm<W>(c, A, B, C, op);
    }
    static elim::HowellResult<W> howell(const Ctx& c, DenseView<const W> A) { return elim::howell<W>(c, A); }
    static elim::HowellResult<W> kernel_module(const Ctx& c, DenseView<const W> A) { return elim::kernel_module<W>(c, A); }
    static elim::HowellResult<W> image_module(const Ctx& c, DenseView<const W> A) { return elim::image_module<W>(c, A); }
    static std::vector<std::vector<W>> solve_multi_rhs_module(const Ctx& c, DenseView<const W> B, DenseView<const W> Y) {
        return elim::solve_multi_rhs_module<W>(c, B, Y);
    }
    static uint32_t length(const Ctx& c, DenseView<const W> A) { return elim::length<W>(c, A); }
    static uint32_t length(const Ctx& c, const elim::HowellResult<W>& h) { return elim::length<W>(c, h); }
    static std::vector<uint32_t> invariant_factors(const Ctx& c, DenseView<const W> G, DenseView<const W> Rl) {
        return elim::invariant_factors<W>(c, G, Rl);
    }
    static std::vector<uint32_t> invariant_factors(const Ctx& c, DenseView<const W> A) {
        return elim::invariant_factors<W>(c, A);
    }
    static elim::CyclicDecomposition<W> cyclic_decomposition(const Ctx& c, DenseView<const W> G, DenseView<const W> Rl) {
        return elim::cyclic_decomposition<W>(c, G, Rl);
    }
    static std::vector<std::vector<W>> cyclic_coordinates(const Ctx& c, const elim::CyclicDecomposition<W>& d,
                                                         DenseView<const W> Y) {
        return elim::cyclic_coordinates<W>(c, d, Y);
    }
};

} // namespace detail

template<class W> struct DigitWord;   // W in {uint32_t, uint64_t, u128} only

template<>
struct DigitWord<uint32_t> : detail::DigitWordBase<uint32_t, DigitWord<uint32_t>> {
    using Wide = uint64_t;
    static void mul_wide(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
        const uint64_t t = static_cast<uint64_t>(a) * b;
        hi = static_cast<uint32_t>(t >> 32);
        lo = static_cast<uint32_t>(t);
    }
    static uint32_t split(uint32_t hi, uint32_t lo, uint32_t M, uint32_t& r) {
        ENGINE_CHECK_MSG(M >= 1 && hi < M, "DigitWord<u32>::split: needs hi < M");
        const uint64_t t = (static_cast<uint64_t>(hi) << 32) | lo;
        r = static_cast<uint32_t>(t % M);
        return static_cast<uint32_t>(t / M);
    }
};

template<>
struct DigitWord<uint64_t> : detail::DigitWordBase<uint64_t, DigitWord<uint64_t>> {
    using Wide = u128;
    static void mul_wide(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
        const u128 t = static_cast<u128>(a) * b;
        hi = static_cast<uint64_t>(t >> 64);
        lo = static_cast<uint64_t>(t);
    }
    static uint64_t split(uint64_t hi, uint64_t lo, uint64_t M, uint64_t& r) {
        ENGINE_CHECK_MSG(M >= 1 && hi < M, "DigitWord<u64>::split: needs hi < M");
        const u128 t = (static_cast<u128>(hi) << 64) | lo;
        r = static_cast<uint64_t>(t % M);
        return static_cast<uint64_t>(t / M);
    }
};

template<>
struct DigitWord<u128> : detail::DigitWordBase<u128, DigitWord<u128>> {
    using Wide = u256;
    static void mul_wide(u128 a, u128 b, u128& hi, u128& lo) {
        const u256 t = detail::mul_wide128(a, b);
        hi = t.hi;
        lo = t.lo;
    }
    // (slot-11 WP-18b) 256-by-128 division by normalised long division on
    // 64-bit halves (Knuth, TAOCP 4.3.1 Algorithm D with two quotient digits;
    // Hacker's Delight's divlu, Fig. 9-3, at twice the width).  Each quotient
    // digit is estimated by one native 128/64 division and corrected at most
    // twice.  Every intermediate fits a u128: the estimate q < 2^64 before any
    // product with a 64-bit half is formed (the short-circuit order below), and
    // the partial remainders are < M < 2^128.  split_restoring (below) is the
    // bit-by-bit oracle it is validated against (EngineLadicInductionValidation (A)).
    static u128 split(u128 hi, u128 lo, u128 M, u128& r) {
        ENGINE_CHECK_MSG(M >= 1 && hi < M, "DigitWord<u128>::split: needs hi < M");
        if (hi == 0) { r = lo % M; return lo / M; }
        const u128 b = static_cast<u128>(1) << 64, mask = b - 1;
        const int s = (M >> 64) ? __builtin_clzll(static_cast<uint64_t>(M >> 64))
                                : 64 + __builtin_clzll(static_cast<uint64_t>(M));
        if (s >= 64) {
            // M < 2^64: two native 128/64 steps (hi < M < 2^64).
            const uint64_t m64 = static_cast<uint64_t>(M);
            const u128 top = (hi << 64) | (lo >> 64);           // < M * 2^64
            const u128 q1 = top / m64, r1 = top % m64;
            const u128 bot = (r1 << 64) | (lo & mask);
            const u128 q0 = bot / m64;
            r = bot % m64;
            return (q1 << 64) | q0;
        }
        const u128 v = M << s;                                   // top bit set
        const u128 vn1 = v >> 64, vn0 = v & mask;
        const u128 un32 = (s == 0) ? hi : ((hi << s) | (lo >> (128 - s)));
        const u128 un10 = lo << s;
        const u128 un1 = un10 >> 64, un0 = un10 & mask;
        u128 q1 = un32 / vn1, rhat = un32 - q1 * vn1;
        while (q1 >= b || q1 * vn0 > ((rhat << 64) | un1)) {
            --q1;
            rhat += vn1;
            if (rhat >= b) break;
        }
        const u128 un21 = ((un32 << 64) | un1) - q1 * v;          // exact mod 2^128: the true value is < v
        u128 q0 = un21 / vn1;
        rhat = un21 - q0 * vn1;
        while (q0 >= b || q0 * vn0 > ((rhat << 64) | un0)) {
            --q0;
            rhat += vn1;
            if (rhat >= b) break;
        }
        r = (((un21 << 64) | un0) - q0 * v) >> s;
        return (q1 << 64) | q0;
    }
    // The original 256-by-128 restoring division, 128 steps (hi < M keeps the
    // quotient in 128 bits and the running remainder < 2M <= 2^129: tracked
    // with a carry bit) -- kept as the oracle of split.
    static u128 split_restoring(u128 hi, u128 lo, u128 M, u128& r) {
        ENGINE_CHECK_MSG(M >= 1 && hi < M, "DigitWord<u128>::split: needs hi < M");
        u128 rem = hi, q = 0;
        for (int i = 127; i >= 0; --i) {
            const bool top = (rem >> 127) != 0;                 // the 129th bit after the shift
            rem = (rem << 1) | ((lo >> i) & 1);
            q <<= 1;
            if (top || rem >= M) { rem -= M; q |= 1; }          // wraps correctly when top is set
        }
        r = rem;
        return q;
    }
};

} // namespace engine
