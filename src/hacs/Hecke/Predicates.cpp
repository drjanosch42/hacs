// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S1 -- Certified floating-point sign filter with exact fallback.
// See Predicates.h for the contract references ([C, thm:fp64],
// [C, cor:fptotal], [C, cor:fp32], [C, thm:backend], [C, obl:fp]).
//
// DO NOT ALTER the filter constants (2^-50 for FP64, 2^-21 for FP32), the
// STRICT threshold inequality |x_hat| > E(B), or the FP evaluation order
// fl(fl(a1*b1) +- fl(a2*b2)). They are the only admissible values per the
// binding contract box in [C, sec:fpfilter].

#include "Predicates.h"

#include "EngineCheck.h" // ENGINE_DEBUG_CHECK (NDEBUG policy)
#include <cmath>

namespace hecke {

namespace {

// thread_local (slot-8 WP-6): see Predicates.h, "PER THREAD".
thread_local PredicateStats g_stats_fp64;
thread_local PredicateStats g_stats_fp32;

inline int sign_of_int(int64_t x) { return (x > 0) - (x < 0); }

// Exact-integer evaluation of sign(a1*b1 + s*a2*b2), backend ladder of
// [C, thm:backend]: exact int64 first, escalate to __int128 on overflow.
// The __int128 tier is always exact here: |a_i|,|b_i| <= B <= 2^53 implies
// |x| <= 2 B^2 <= 2^107 < 2^127. Arbitrary precision is never required
// (invariant I5).
int sign_dot2_exact(int64_t a1, int64_t b1, int64_t a2, int64_t b2, int s) {
    // int64 tier (branch-free overflow probes; no UB on overflow).
    int64_t p1, p2, x;
    if (!__builtin_mul_overflow(a1, b1, &p1) &&
        !__builtin_mul_overflow(a2, b2, &p2)) {
        const bool ovf = (s == 1) ? __builtin_add_overflow(p1, p2, &x)
                                  : __builtin_sub_overflow(p1, p2, &x);
        if (!ovf) return sign_of_int(x);
    }
    // __int128 tier.
    const __int128 w1 = static_cast<__int128>(a1) * static_cast<__int128>(b1);
    const __int128 w2 = static_cast<__int128>(a2) * static_cast<__int128>(b2);
    const __int128 xx = (s == 1) ? (w1 + w2) : (w1 - w2);
    return (xx > 0) - (xx < 0);
}

// Hard preconditions of [C, thm:fp64] / [C, cor:fp32]; a violation would
// leave the FP verdict uncertified, so it aborts (throws), never corrupts
// ([C, obl:backend]).
inline void check_dot2_preconditions(uint64_t m_a1, uint64_t m_b1,
                                     uint64_t m_a2, uint64_t m_b2,
                                     int s, uint64_t B, uint64_t B_limit) {
    if (s != 1 && s != -1) {
        throw HeckeOverflow("hecke::sign_dot2: s must be +1 or -1");
    }
    if (B > B_limit) {
        throw HeckeOverflow("hecke::sign_dot2: bound B exceeds the exact-conversion limit");
    }
    if (m_a1 > B || m_b1 > B || m_a2 > B || m_b2 > B) {
        throw HeckeOverflow("hecke::sign_dot2: input magnitude exceeds the certified bound B");
    }
}

} // namespace

int sign_dot2(int64_t a1, int64_t b1, int64_t a2, int64_t b2, int s, uint64_t B) {
    // Preconditions of [C, thm:fp64]: |a_i|,|b_i| <= B <= 2^53 (the inputs
    // then convert to binary64 exactly).
    check_dot2_preconditions(uabs64(a1), uabs64(b1), uabs64(a2), uabs64(b2),
                             s, B, uint64_t(1) << 53);

#ifndef HECKE_PURE_INTEGER
    // FP64 fast path ([C, thm:fp64]). Evaluation order is EXACTLY
    // fl(fl(a1*b1) +- fl(a2*b2)); the volatile intermediates force one
    // IEEE-754 binary64 rounding per step (no FMA contraction, no wider
    // intermediate precision).
    volatile double p1 = static_cast<double>(a1) * static_cast<double>(b1);
    volatile double p2 = static_cast<double>(a2) * static_cast<double>(b2);
    const double x_hat = (s == 1) ? (p1 + p2) : (p1 - p2);

    // Threshold E(B) = 2^-50 * B^2 ([C, thm:fp64]; the only admissible FP64
    // constant). (double)B is exact for B <= 2^53; the single rounding of
    // B*B (relative error <= 2^-53) is absorbed by the factor >= 2 margin in
    // the theorem's constant (its proof shows |x_hat - x| <= 4 B^2 eps (1+eps)
    // < 2^-50 B^2 / 2), so the FP verdict below remains certified.
    const double dB = static_cast<double>(B);
    const double E = std::ldexp(dB * dB, -50);

    // STRICT threshold test: |x_hat| > E(B) certifies sgn(x) = sgn(x_hat).
    if (x_hat > E) {
        // Debug-only: re-running the exact predicate defeats the whole point
        // of the FP filter, so this cross-check is compiled out under NDEBUG.
        ENGINE_DEBUG_CHECK_MSG(sign_dot2_exact(a1, b1, a2, b2, s) == 1,
                               "FP64 filter verdict disagrees with exact sign");
        ++g_stats_fp64.fp_hits;
        return 1;
    }
    if (x_hat < -E) {
        // Debug-only: re-running the exact predicate defeats the whole point
        // of the FP filter, so this cross-check is compiled out under NDEBUG.
        ENGINE_DEBUG_CHECK_MSG(sign_dot2_exact(a1, b1, a2, b2, s) == -1,
                               "FP64 filter verdict disagrees with exact sign");
        ++g_stats_fp64.fp_hits;
        return -1;
    }

    // Totality threshold ([C, cor:fptotal]): for B <= 2^24 (then E(B) <=
    // 2^-2 < 1/2), |x_hat| <= E(B) certifies x = 0; the exact fallback is
    // provably unreachable.
    if (B <= (uint64_t(1) << 24)) {
        // Debug-only: re-running the exact predicate defeats the whole point
        // of the FP filter, so this cross-check is compiled out under NDEBUG.
        ENGINE_DEBUG_CHECK_MSG(sign_dot2_exact(a1, b1, a2, b2, s) == 0,
                               "FP64 totality verdict disagrees with exact sign");
        ++g_stats_fp64.fp_hits;
        return 0;
    }
#endif // HECKE_PURE_INTEGER

    // Exact bounded-integer fallback ([C, thm:backend]).
    ++g_stats_fp64.exact_fallbacks;
    return sign_dot2_exact(a1, b1, a2, b2, s);
}

int sign_det2(const Mat22& m) {
    // Shape (D) of [C, sec:fpfilter]: sign(ad - bc).
    uint64_t B = uabs64(m.a);
    if (uabs64(m.b) > B) B = uabs64(m.b);
    if (uabs64(m.c) > B) B = uabs64(m.c);
    if (uabs64(m.d) > B) B = uabs64(m.d);
    return sign_dot2(m.a, m.d, m.b, m.c, -1, B);
}

int sign_dot2_f32(int32_t a1, int32_t b1, int32_t a2, int32_t b2, int s, uint32_t B) {
    // Preconditions of [C, cor:fp32]: |a_i|,|b_i| <= B <= 2^24 (exact
    // conversion of the inputs to binary32).
    check_dot2_preconditions(uabs64(a1), uabs64(b1), uabs64(a2), uabs64(b2),
                             s, B, uint64_t(1) << 24);

#ifndef HECKE_PURE_INTEGER
    // FP32 fast path ([C, cor:fp32]); same shape as the FP64 filter with
    // eps = 2^-24 and threshold constant 2^-21 (the only admissible FP32
    // constant). Evaluation order fl(fl(a1*b1) +- fl(a2*b2)) in binary32.
    volatile float p1 = static_cast<float>(a1) * static_cast<float>(b1);
    volatile float p2 = static_cast<float>(a2) * static_cast<float>(b2);
    const float x_hat = (s == 1) ? (p1 + p2) : (p1 - p2);

    const float fB = static_cast<float>(B);
    const float E = std::ldexp(fB * fB, -21);

    // STRICT threshold test.
    if (x_hat > E) {
        // Debug-only: see the FP64 routine above.
        ENGINE_DEBUG_CHECK_MSG(sign_dot2_exact(a1, b1, a2, b2, s) == 1,
                               "FP32 filter verdict disagrees with exact sign");
        ++g_stats_fp32.fp_hits;
        return 1;
    }
    if (x_hat < -E) {
        // Debug-only: see the FP64 routine above.
        ENGINE_DEBUG_CHECK_MSG(sign_dot2_exact(a1, b1, a2, b2, s) == -1,
                               "FP32 filter verdict disagrees with exact sign");
        ++g_stats_fp32.fp_hits;
        return -1;
    }

    // Totality ([C, cor:fp32]): total iff B <= 2^10.
    if (B <= (uint32_t(1) << 10)) {
        // Debug-only: see the FP64 routine above.
        ENGINE_DEBUG_CHECK_MSG(sign_dot2_exact(a1, b1, a2, b2, s) == 0,
                               "FP32 totality verdict disagrees with exact sign");
        ++g_stats_fp32.fp_hits;
        return 0;
    }
#endif // HECKE_PURE_INTEGER

    // Exact fallback; int64 always suffices here (|x| <= 2 B^2 <= 2^49).
    ++g_stats_fp32.exact_fallbacks;
    return sign_dot2_exact(a1, b1, a2, b2, s);
}

PredicateStats& predicate_stats_fp64() { return g_stats_fp64; }
PredicateStats& predicate_stats_fp32() { return g_stats_fp32; }

void reset_predicate_stats() {
    g_stats_fp64 = PredicateStats{};
    g_stats_fp32 = PredicateStats{};
}

} // namespace hecke
