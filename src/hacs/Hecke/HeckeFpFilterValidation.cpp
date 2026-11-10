// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S1V -- independent FP-filter / exact-arithmetic validation.
//
// INDEPENDENCE NOTE (binding on this file): this validation was written by an
// agent that did NOT author Stage S1 (Predicates.h/.cpp, ExactInt.h/.cpp).
// Every reference computation below (the independent __int128 exact sign,
// the independent bit-identical FP64/FP32 evaluation-order replay, the
// threshold formulas E64(B) = 2^-50*B^2 and E32(B) = 2^-21*B^2) is derived
// directly from the binding contract docs/latex/hecke_action.tex
// ([C, thm:fp64], [C, cor:fptotal], [C, cor:fp32], [C, thm:backend]) and the
// S1V oracle O13 text in
// docs/implementation/hecke_action_implementation_plan.md. The only S1
// artifacts consulted are the PUBLIC headers Predicates.h / ExactInt.h (for
// call signatures). Hecke/HeckeExactIntSmokeValidation.cpp (the S1 author's
// own test) was read only to confirm registry wiring and was NOT used as a
// source of mathematical truth: every check here recomputes its own
// reference from first principles instead of trusting that file's oracle.
//
// Oracle O13 (see the plan, stage S1V):
//   Phase A: >= 10^7 uniform random samples per magnitude class
//            B in {2^10, 2^24, 2^26, 2^31}; filtered sign must equal an
//            independently hand-rolled __int128 exact sign in 100% of
//            samples; PredicateStats.exact_fallbacks == 0 for B <= 2^24
//            ([C, cor:fptotal]).
//   Phase B: adversarial near-threshold construction (exact Sterbenz-style
//            boundary cases hitting E(B) exactly, plus a directed search for
//            genuine-rounding near-cancellation cases at B = 2^26, 2^31).
//   Phase C: overflow-guard confirmation (HeckeOverflow, never UB/wraparound).
//   Phase D: sign_det2 shape-(D) spot check.
//   Phase E: sign_dot2_f32 spot check ([C, cor:fp32]).
//
// Both HECKE_PURE_INTEGER settings run this same deterministic (fixed-seed)
// test; if both exit 0, the sign decisions in both builds necessarily agree
// with each other (both equal the same independent reference for the same
// inputs), which is the operational meaning of [C, obl:fp] / invariant I4
// for this predicate layer.

#include "HeckeFpFilterValidation.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ExactInt.h"
#include "Predicates.h"

namespace {

using hecke::HeckeOverflow;
using hecke::Mat22;

uint32_t fail(uint32_t code, const std::string& what) {
    std::cout << "HeckeFpFilterValidation: FAILED check " << code << " (" << what << ")"
              << std::endl;
    return code;
}

// ---------------------------------------------------------------------------
// Independent fixed-seed xorshift64 stream (Marsaglia's canonical (13,7,17)
// triple-shift generator, as named by the O13 spec text). Implemented from
// scratch for this validation; seeds below are distinct from any used
// elsewhere in the repository.
// ---------------------------------------------------------------------------
struct XorShift64 {
    uint64_t state;
    explicit XorShift64(uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return x;
    }
    // Approximately uniform in [-B, B]; bias is irrelevant for this test's
    // purpose (coverage, not a statistical-uniformity study).
    int64_t in_range(uint64_t B) {
        const uint64_t span = 2 * B + 1;
        return static_cast<int64_t>(next() % span) - static_cast<int64_t>(B);
    }
};

// ---------------------------------------------------------------------------
// Independent exact reference: sign(a1*b1 + s*a2*b2) via __int128, hand-
// rolled per the S1V spec ("hand-roll the __int128 computation in your
// test"). Always exact for the magnitude classes used here
// (|a_i|,|b_i| <= 2^31 => |a_i*b_i| <= 2^62, sum <= 2^63, well inside
// __int128's range).
// ---------------------------------------------------------------------------
int IndependentExactSign(int64_t a1, int64_t b1, int64_t a2, int64_t b2, int s) {
    const __int128 x = static_cast<__int128>(a1) * static_cast<__int128>(b1) +
                       static_cast<__int128>(s) * static_cast<__int128>(a2) * static_cast<__int128>(b2);
    return (x > 0) - (x < 0);
}

// Independent replay of the FP64 evaluation order mandated by [C, thm:fp64]:
// fl(fl(a1*b1) +- fl(a2*b2)), one IEEE-754 binary64 rounding per operation
// (volatile forces this; no FMA contraction / no wider intermediate
// precision). Written independently from the theorem statement, not copied
// from Predicates.cpp.
double IndependentXHat64(int64_t a1, int64_t b1, int64_t a2, int64_t b2, int s) {
    volatile double p1 = static_cast<double>(a1) * static_cast<double>(b1);
    volatile double p2 = static_cast<double>(a2) * static_cast<double>(b2);
    return (s == 1) ? (p1 + p2) : (p1 - p2);
}

double IndependentE64(uint64_t B) {
    const double dB = static_cast<double>(B);
    return std::ldexp(dB * dB, -50); // E(B) = 2^-50 * B^2, [C, thm:fp64]
}

void print_case(int64_t a1, int64_t b1, int64_t a2, int64_t b2, int s, uint64_t B,
                int filtered, int expected) {
    std::cout << "    a1=" << a1 << " b1=" << b1 << " a2=" << a2 << " b2=" << b2
              << " s=" << s << " B=" << B << " filtered_sign=" << filtered
              << " exact_sign=" << expected << std::endl;
}

// ===========================================================================
// Phase A: bulk uniform sampling, >= 10^7 per magnitude class.
// ===========================================================================

struct ClassSpec {
    uint64_t B;
    uint64_t N;
    uint64_t seed;
    bool totality_required; // [C, cor:fptotal]: B <= 2^24
};

uint32_t RunBulkUniformClass(const ClassSpec& cls, uint32_t fail_base) {
    hecke::reset_predicate_stats();
    XorShift64 rng(cls.seed);
    for (uint64_t i = 0; i < cls.N; ++i) {
        const int s = (rng.next() & 1u) ? 1 : -1;
        const int64_t a1 = rng.in_range(cls.B);
        const int64_t b1 = rng.in_range(cls.B);
        const int64_t a2 = rng.in_range(cls.B);
        const int64_t b2 = rng.in_range(cls.B);

        const int filtered = hecke::sign_dot2(a1, b1, a2, b2, s, cls.B);
        const int expected = IndependentExactSign(a1, b1, a2, b2, s);
        if (filtered != expected) {
            std::cout << "  Phase A mismatch (B=" << cls.B << ", sample " << i
                       << "), violates [C, thm:fp64]/[C, thm:backend]:" << std::endl;
            print_case(a1, b1, a2, b2, s, cls.B, filtered, expected);
            return fail_base + 1;
        }
    }
#ifndef HECKE_PURE_INTEGER
    const hecke::PredicateStats st = hecke::predicate_stats_fp64();
    if (cls.totality_required) {
        // [C, cor:fptotal]: for B <= 2^24 the exact fallback is provably
        // unreachable -- every one of the N calls must be an fp_hit.
        if (st.exact_fallbacks != 0) {
            std::cout << "  Phase A: exact_fallbacks=" << st.exact_fallbacks
                       << " != 0 for totality class B=" << cls.B
                       << " -- violates [C, cor:fptotal]." << std::endl;
            return fail_base + 2;
        }
        if (st.fp_hits != cls.N) {
            std::cout << "  Phase A: fp_hits=" << st.fp_hits << " != N=" << cls.N
                       << " for totality class B=" << cls.B << std::endl;
            return fail_base + 3;
        }
    } else {
        if (st.fp_hits == 0) {
            std::cout << "  Phase A: fp_hits == 0 for class B=" << cls.B
                       << " -- FP fast path never ran; expected fp_hits > 0." << std::endl;
            return fail_base + 4;
        }
        if (st.fp_hits + st.exact_fallbacks != cls.N) {
            std::cout << "  Phase A: fp_hits+exact_fallbacks=" << (st.fp_hits + st.exact_fallbacks)
                       << " != N=" << cls.N << " for class B=" << cls.B << std::endl;
            return fail_base + 5;
        }
        std::cout << "  Phase A class B=" << cls.B << ": fp_hits=" << st.fp_hits
                   << " exact_fallbacks=" << st.exact_fallbacks << " (out of " << cls.N << ")"
                   << std::endl;
    }
#else
    // Pure-integer build: the FP fast path is compiled out entirely; every
    // call must be an exact_fallback (Predicates.h contract comment).
    const hecke::PredicateStats st = hecke::predicate_stats_fp64();
    if (st.fp_hits != 0 || st.exact_fallbacks != cls.N) {
        std::cout << "  Phase A (HECKE_PURE_INTEGER): fp_hits=" << st.fp_hits
                   << " exact_fallbacks=" << st.exact_fallbacks << " N=" << cls.N
                   << " -- expected fp_hits=0, exact_fallbacks=N." << std::endl;
        return fail_base + 6;
    }
#endif
    return 0;
}

uint32_t CheckBulkUniform() {
    const ClassSpec classes[] = {
        {uint64_t(1) << 10, 10000000ull, 0x1F2E3D4C5B6A7988ull, true},
        {uint64_t(1) << 24, 10000000ull, 0x2468ACE013579BDFull, true},
        {uint64_t(1) << 26, 10000000ull, 0x77216A1F9C3E5D0Bull, false},
        {uint64_t(1) << 31, 10000000ull, 0x3355AABB77CCEE11ull, false},
    };
    uint32_t base = 1000;
    for (const ClassSpec& cls : classes) {
        const uint32_t rc = RunBulkUniformClass(cls, base);
        if (rc != 0) return rc;
        base += 10;
    }
    return 0;
}

// ===========================================================================
// Phase B1: deterministic exact boundary construction (Sterbenz-exact: no FP
// rounding at all in this construction, so the independently computed x_hat
// equals the exact integer target x bit-for-bit, giving an unambiguous test
// of the STRICT threshold inequality and the zero/totality decision).
//
// Construction: a1 = B - margin, b1 = 1; for s=-1, a2 = a1 - delta; for
// s=+1, a2 = delta - a1; b2 = 1. Then a1*b1 +- a2*b2 = delta exactly, and
// (since a1, a2 are both close in magnitude, exactly representable doubles)
// fl(a1) -+ fl(a2) is also computed exactly by Sterbenz's lemma -- verified
// in-test, not assumed, by recomputing x_hat with IndependentXHat64 rather
// than trusting the algebra.
// ===========================================================================

uint32_t RunBoundaryCase(uint64_t B, int64_t margin, int64_t delta, int s, uint32_t code) {
    const int64_t a1 = static_cast<int64_t>(B) - margin;
    const int64_t a2 = (s == -1) ? (a1 - delta) : (delta - a1);
    const int64_t b1 = 1, b2 = 1;

    // b1 = b2 = 1, so the exact value is simply a1 +- a2 (safe in int64:
    // |a1|,|a2| <= B <= 2^31).
    const int64_t x_exact = (s == 1) ? (a1 * b1 + a2 * b2) : (a1 * b1 - a2 * b2);
    if (x_exact != delta) {
        std::cout << "  Phase B1: construction bug (my own), x_exact=" << x_exact
                   << " != delta=" << delta << " (a1=" << a1 << " a2=" << a2 << " s=" << s
                   << ")." << std::endl;
        return code; // self-check failure: our own construction, not S1.
    }

    const double x_hat = IndependentXHat64(a1, b1, a2, b2, s);
    const double E = IndependentE64(B);
    const bool totality = (B <= (uint64_t(1) << 24));

    const int filtered = hecke::sign_dot2(a1, b1, a2, b2, s, B);
    const int expected = IndependentExactSign(a1, b1, a2, b2, s);
    if (filtered != expected) {
        std::cout << "  Phase B1 mismatch, violates [C, thm:fp64]:" << std::endl;
        print_case(a1, b1, a2, b2, s, B, filtered, expected);
        return code;
    }

#ifndef HECKE_PURE_INTEGER
    hecke::reset_predicate_stats();
    (void)hecke::sign_dot2(a1, b1, a2, b2, s, B);
    const hecke::PredicateStats st = hecke::predicate_stats_fp64();
    const bool decisive = (x_hat > E) || (x_hat < -E);
    const bool expect_fallback = !decisive && !totality;
    if (expect_fallback) {
        if (st.exact_fallbacks != 1 || st.fp_hits != 0) {
            std::cout << "  Phase B1: branch mismatch at delta=" << delta << " B=" << B
                       << " x_hat=" << x_hat << " E=" << E
                       << " -- expected exact_fallback, got fp_hits=" << st.fp_hits
                       << " exact_fallbacks=" << st.exact_fallbacks
                       << " (violates strict-inequality / [C, cor:fptotal] threshold logic)."
                       << std::endl;
            return code + 1;
        }
    } else {
        if (st.fp_hits != 1 || st.exact_fallbacks != 0) {
            std::cout << "  Phase B1: branch mismatch at delta=" << delta << " B=" << B
                       << " x_hat=" << x_hat << " E=" << E << " totality=" << totality
                       << " -- expected fp_hit (decisive=" << decisive << "), got fp_hits="
                       << st.fp_hits << " exact_fallbacks=" << st.exact_fallbacks << std::endl;
            return code + 2;
        }
        // If B <= 2^24 (totality) and delta == 0, this must be the totality
        // zero-decision branch, not the strict-inequality decisive branch.
        if (totality && delta == 0 && decisive) {
            std::cout << "  Phase B1: unexpected decisive verdict for delta=0 within the"
                       << " totality regime at B=" << B << " (x_hat=" << x_hat << ")."
                       << std::endl;
            return code + 3;
        }
    }
#endif
    return 0;
}

uint32_t CheckAdversarialBoundaryFp64() {
    struct ClassBoundary {
        uint64_t B;
        int64_t margin;
        std::vector<int64_t> deltas;
    };
    const std::vector<ClassBoundary> classes = {
        {uint64_t(1) << 10, 5, {-2, -1, 0, 1, 2}},
        {uint64_t(1) << 24, 5, {-2, -1, 0, 1, 2}},
        {uint64_t(1) << 26, 10, {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6}},
        {uint64_t(1) << 31, 4110, {-4102, -4097, -4096, -4095, -4090, -1, 0, 1, 4090, 4095, 4096,
                                    4097, 4102}},
    };
    uint32_t code = 2000;
    for (const ClassBoundary& cb : classes) {
        for (int64_t delta : cb.deltas) {
            for (int s : {-1, 1}) {
                const uint32_t rc = RunBoundaryCase(cb.B, cb.margin, delta, s, code);
                if (rc != 0) return rc;
                code += 10;
            }
        }
    }
    return 0;
}

// ===========================================================================
// Phase B2: directed search for near-threshold cases WITH genuine binary64
// rounding (not just Sterbenz-exact cancellation), at B = 2^26 and B = 2^31
// (the two classes above the totality bound where the exact fallback is
// legitimately reachable, [C, thm:backend]).
//
// Construction: draw a2,b2 in [1,B-1]; P2 = a2*b2 (exact __int128). Choose
// a1 = round(P2/B) (nearest-integer division, __int128 exact) and
// b1 = -s*B. Then algebraically a1*b1 + s*a2*b2 = -s*(a1*B - P2) = -s*rem,
// where rem = a1*B - P2 is the exact residual of the rounding -- so |x| is
// controlled to be within about B/2 of 0, and since a1,B are both ~B in
// magnitude, fl(a1*B) genuinely rounds (ULP ~ B^2/2^52) whenever B^2 > 2^53
// (true for B=2^31; marginal/exact for B=2^26). Every hit is independently
// re-derived via the generic formula (not trusted from the algebra above)
// before being used.
// ===========================================================================

uint32_t RunRoundingSearchClass(uint64_t B, uint64_t seed, uint64_t trial_budget,
                                uint64_t want_hits, double window_mult, uint32_t code) {
    XorShift64 rng(seed);
    const double E = IndependentE64(B);
    double window = window_mult * E;
    if (window < 8.0) window = 8.0;
    const __int128 window128 = static_cast<__int128>(window) + 1;

    uint64_t hits = 0;
    uint64_t decisive_hits = 0, fallback_hits = 0;
    for (uint64_t trial = 0; trial < trial_budget && hits < want_hits; ++trial) {
        const int64_t a2 = 1 + static_cast<int64_t>(rng.next() % (B - 1));
        const int64_t b2 = 1 + static_cast<int64_t>(rng.next() % (B - 1));
        const int s = (rng.next() & 1u) ? 1 : -1;

        const __int128 P2 = static_cast<__int128>(a2) * static_cast<__int128>(b2);
        const __int128 Bw = static_cast<__int128>(B);
        __int128 a1_w = (P2 + Bw / 2) / Bw; // nearest-integer division
        if (a1_w > static_cast<__int128>(B)) a1_w = static_cast<__int128>(B);
        const int64_t a1 = static_cast<int64_t>(a1_w);
        const __int128 rem = a1_w * Bw - P2; // exact residual

        __int128 abs_rem = rem < 0 ? -rem : rem;
        if (abs_rem > window128) continue;

        const int64_t b1 = (s == 1) ? -static_cast<int64_t>(B) : static_cast<int64_t>(B);

        // Self-check: recompute x via the generic exact formula and compare
        // to the algebraic prediction -s*rem, to catch a construction bug on
        // our own side before it is (mis)attributed to S1.
        const __int128 x_generic = static_cast<__int128>(a1) * static_cast<__int128>(b1) +
                                   static_cast<__int128>(s) * static_cast<__int128>(a2) *
                                       static_cast<__int128>(b2);
        const __int128 x_predicted = (s == 1) ? -rem : rem; // -s*rem, s in {-1,1}
        if (x_generic != x_predicted) {
            std::cout << "  Phase B2: self-check failed (our construction), x_generic="
                       << static_cast<int64_t>(x_generic)
                       << " x_predicted=" << static_cast<int64_t>(x_predicted) << std::endl;
            return code;
        }

        ++hits;
        const double x_hat = IndependentXHat64(a1, b1, a2, b2, s);
        const int filtered = hecke::sign_dot2(a1, b1, a2, b2, s, B);
        const int expected = IndependentExactSign(a1, b1, a2, b2, s);
        if (filtered != expected) {
            std::cout << "  Phase B2 mismatch (real-rounding near-threshold case), violates"
                       << " [C, thm:fp64]:" << std::endl;
            print_case(a1, b1, a2, b2, s, B, filtered, expected);
            std::cout << "    x_hat(independent)=" << x_hat << " E(B)=" << E << std::endl;
            return code + 1;
        }

#ifndef HECKE_PURE_INTEGER
        hecke::reset_predicate_stats();
        (void)hecke::sign_dot2(a1, b1, a2, b2, s, B);
        const hecke::PredicateStats st = hecke::predicate_stats_fp64();
        const bool decisive = (x_hat > E) || (x_hat < -E);
        if (decisive) {
            ++decisive_hits;
            if (st.fp_hits != 1 || st.exact_fallbacks != 0) {
                std::cout << "  Phase B2: expected decisive fp_hit at B=" << B
                           << " x_hat=" << x_hat << " E=" << E << " but got fp_hits="
                           << st.fp_hits << " exact_fallbacks=" << st.exact_fallbacks
                           << std::endl;
                print_case(a1, b1, a2, b2, s, B, filtered, expected);
                return code + 2;
            }
        } else {
            ++fallback_hits;
            if (st.exact_fallbacks != 1 || st.fp_hits != 0) {
                std::cout << "  Phase B2: expected exact_fallback (below/at threshold) at B="
                           << B << " x_hat=" << x_hat << " E=" << E << " but got fp_hits="
                           << st.fp_hits << " exact_fallbacks=" << st.exact_fallbacks
                           << " -- fallback taken when theorem forbids it, or vice versa."
                           << std::endl;
                print_case(a1, b1, a2, b2, s, B, filtered, expected);
                return code + 3;
            }
        }
#endif
    }

    if (hits < want_hits) {
        std::cout << "  Phase B2: only found " << hits << " near-threshold hits (wanted "
                   << want_hits << ") for B=" << B << " within the trial budget; widening"
                   << " window_mult or trial_budget may be needed. Treating as non-fatal if"
                   << " hits > 0." << std::endl;
        if (hits == 0) return code + 4;
    }
    std::cout << "  Phase B2 class B=" << B << ": " << hits << " near-threshold hits ("
               << decisive_hits << " decisive fp_hit, " << fallback_hits << " exact_fallback)."
               << std::endl;
    return 0;
}

uint32_t CheckAdversarialRoundingSearch() {
    // window_mult = 20 (rather than a much wider net): keeps a non-trivial
    // fraction (~1/20) of found hits strictly inside the fallback region
    // |x| <= E(B), so the search actually exercises BOTH the decisive fp_hit
    // branch AND the exact-fallback branch with genuine binary64 rounding in
    // the products (not just Sterbenz-exact cancellation, cf. Phase B1).
    // B=2^26: E=4, window ratio to B is small, needs more trials.
    uint32_t rc = RunRoundingSearchClass(uint64_t(1) << 26, 0x9F86D081884C7D65ull, 120000000ull,
                                         150, 20.0, 3000);
    if (rc != 0) return rc;
    // B=2^31: E=4096, much larger absolute window, cheap to search.
    rc = RunRoundingSearchClass(uint64_t(1) << 31, 0x4B227777D4DD1FC6ull, 8000000ull, 200, 20.0,
                                3050);
    if (rc != 0) return rc;
    return 0;
}

// Extreme-magnitude cases forcing the int64 -> __int128 escalation inside
// the exact backend ([C, thm:backend]): at B=2^31, 2*B^2 = 2^63 exceeds
// INT64_MAX (2^63-1) by exactly 1 when both dot-product terms are maximal
// and same-signed.
uint32_t CheckInt64ToInt128Escalation() {
    const int64_t B = int64_t(1) << 31;
    // s=+1: x = 2*B^2 = 2^63, overflows int64 addition, needs __int128.
    {
        const int filtered = hecke::sign_dot2(B, B, B, B, 1, uint64_t(B));
        const int expected = IndependentExactSign(B, B, B, B, 1);
        if (filtered != expected || expected != 1) {
            std::cout << "  int64->int128 escalation case (s=+1) failed: filtered=" << filtered
                       << " expected=" << expected << std::endl;
            return 3101;
        }
    }
    // s=-1: x = 0 exactly, despite maximal intermediate products.
    {
        const int filtered = hecke::sign_dot2(B, B, B, B, -1, uint64_t(B));
        const int expected = IndependentExactSign(B, B, B, B, -1);
        if (filtered != expected || expected != 0) {
            std::cout << "  int64->int128 escalation case (s=-1) failed: filtered=" << filtered
                       << " expected=" << expected << std::endl;
            return 3102;
        }
    }
    // Same at B=2^26 (2*B^2 = 2^53, safely within int64, sanity control).
    const int64_t B2 = int64_t(1) << 26;
    {
        const int filtered = hecke::sign_dot2(B2, B2, B2, B2, 1, uint64_t(B2));
        const int expected = IndependentExactSign(B2, B2, B2, B2, 1);
        if (filtered != expected || expected != 1) return 3103;
    }
    return 0;
}

// ===========================================================================
// Phase C: overflow guard confirmation -- HeckeOverflow must be thrown
// (never UB / silent wraparound) on crafted out-of-bound inputs. Constants
// chosen independently of Hecke/HeckeExactIntSmokeValidation.cpp.
// ===========================================================================

uint32_t CheckOverflowGuard() {
    // (1) Mat22 product with ~2^41-scale entries: product terms ~2^82,
    // overflows int64 in Mat22::operator*.
    {
        const int64_t big = (int64_t(1) << 41) + 12345;
        const Mat22 X{big, -big, big + 7, big};
        bool threw = false;
        try {
            const Mat22 Y = X * X;
            (void)Y;
        } catch (const HeckeOverflow&) {
            threw = true;
        }
        if (!threw) return fail(3201, "Mat22 2^41-scale product did not throw HeckeOverflow");
    }
    // (2) Mat22::det() overflow on a different scale (2^42 diag-ish matrix).
    {
        const int64_t big = int64_t(1) << 42;
        bool threw = false;
        try {
            (void)Mat22{big, 3, 5, big}.det();
        } catch (const HeckeOverflow&) {
            threw = true;
        }
        if (!threw) return fail(3202, "Mat22::det() 2^42-scale did not throw HeckeOverflow");
    }
    // (3) sign_dot2 with B > 2^53 (precondition of [C, thm:fp64] violated).
    {
        bool threw = false;
        try {
            (void)hecke::sign_dot2(5, 5, 5, 5, 1, (uint64_t(1) << 53) + 987654321ull);
        } catch (const HeckeOverflow&) {
            threw = true;
        }
        if (!threw) return fail(3203, "sign_dot2 with B > 2^53 did not throw HeckeOverflow");
    }
    // (4) sign_dot2 with an actual |a_i| > B violation (different constants
    // from the S1 smoke test).
    {
        bool threw = false;
        try {
            (void)hecke::sign_dot2(777, 1, 1, 1, 1, 100);
        } catch (const HeckeOverflow&) {
            threw = true;
        }
        if (!threw) return fail(3204, "sign_dot2 with |a1| > B did not throw HeckeOverflow");
        threw = false;
        try {
            (void)hecke::sign_dot2(1, 1, 1, -900, 1, 100);
        } catch (const HeckeOverflow&) {
            threw = true;
        }
        if (!threw) return fail(3205, "sign_dot2 with |b2| > B did not throw HeckeOverflow");
    }
    // (5) sign_dot2 with s not in {-1,+1}.
    {
        bool threw = false;
        try {
            (void)hecke::sign_dot2(1, 1, 1, 1, 0, 10);
        } catch (const HeckeOverflow&) {
            threw = true;
        }
        if (!threw) return fail(3206, "sign_dot2 with s=0 did not throw HeckeOverflow");
        threw = false;
        try {
            (void)hecke::sign_dot2(1, 1, 1, 1, 7, 10);
        } catch (const HeckeOverflow&) {
            threw = true;
        }
        if (!threw) return fail(3207, "sign_dot2 with s=7 did not throw HeckeOverflow");
    }
    // (6) sign_dot2_f32 with B > 2^24 (precondition of [C, cor:fp32]).
    {
        bool threw = false;
        try {
            (void)hecke::sign_dot2_f32(1, 1, 1, 1, 1, (uint32_t(1) << 24) + 999u);
        } catch (const HeckeOverflow&) {
            threw = true;
        }
        if (!threw) return fail(3208, "sign_dot2_f32 with B > 2^24 did not throw HeckeOverflow");
    }
    // (7) Mat22::max_abs / neg on INT64_MIN (magnitude not representable);
    // distinct entries from the S1 smoke test.
    {
        bool threw = false;
        try {
            (void)Mat22{1, 2, 3, INT64_MIN}.max_abs();
        } catch (const HeckeOverflow&) {
            threw = true;
        }
        if (!threw) return fail(3209, "Mat22::max_abs on INT64_MIN did not throw");
        threw = false;
        try {
            (void)Mat22{1, 2, INT64_MIN, 4}.neg();
        } catch (const HeckeOverflow&) {
            threw = true;
        }
        if (!threw) return fail(3210, "Mat22::neg on INT64_MIN did not throw");
    }
    // (8) Escalation tier (Mat22W) must NOT throw for the same 2^41-scale
    // product that overflowed Mat22 above -- confirms the guard is
    // magnitude-specific, not spuriously firing.
    {
        const int64_t big = (int64_t(1) << 41) + 12345;
        const hecke::Mat22W XW{big, -big, big + 7, big};
        try {
            const hecke::Mat22W YW = XW * XW;
            (void)YW;
        } catch (const HeckeOverflow&) {
            return fail(3211, "Mat22W (int128 tier) spuriously threw on a value it must accept");
        }
    }
    return 0;
}

// ===========================================================================
// Phase D: sign_det2 (shape (D): sign(ad-bc)) spot check against an
// independently computed reference, across varied entry magnitudes.
// ===========================================================================

uint32_t CheckSignDet2Random() {
    XorShift64 rng(0x6B17D1F2E12C4247ull);
    const uint64_t magnitudes[] = {1000ull, uint64_t(1) << 20, uint64_t(1) << 26,
                                   uint64_t(1) << 30};
    const uint64_t per_class = 500000ull;
    for (uint64_t B : magnitudes) {
        for (uint64_t i = 0; i < per_class; ++i) {
            const int64_t a = rng.in_range(B);
            const int64_t b = rng.in_range(B);
            const int64_t c = rng.in_range(B);
            const int64_t d = rng.in_range(B);
            const Mat22 m{a, b, c, d};
            const int filtered = hecke::sign_det2(m);
            const __int128 det = static_cast<__int128>(a) * d - static_cast<__int128>(b) * c;
            const int expected = (det > 0) - (det < 0);
            if (filtered != expected) {
                std::cout << "  sign_det2 mismatch: a=" << a << " b=" << b << " c=" << c
                           << " d=" << d << " filtered=" << filtered << " expected=" << expected
                           << std::endl;
                return 3301;
            }
        }
        // Exact-zero determinant spot check (rank-deficient matrix), per class.
        const Mat22 degenerate{static_cast<int64_t>(B) / 2 + 3, static_cast<int64_t>(B) / 4 - 1,
                               (static_cast<int64_t>(B) / 2 + 3) * 2,
                               (static_cast<int64_t>(B) / 4 - 1) * 2};
        if (hecke::sign_det2(degenerate) != 0) {
            std::cout << "  sign_det2 zero-determinant spot check failed at B=" << B << std::endl;
            return 3310;
        }
    }
    return 0;
}

// ===========================================================================
// Phase E: sign_dot2_f32 spot check ([C, cor:fp32]): total for B <= 2^10,
// correct-or-fallback above it (up to the B <= 2^24 exact-conversion limit).
// ===========================================================================

uint32_t CheckFp32Class(uint32_t B, uint64_t N, uint64_t seed, bool totality_required,
                        uint32_t code) {
    hecke::reset_predicate_stats();
    XorShift64 rng(seed);
    for (uint64_t i = 0; i < N; ++i) {
        const int s = (rng.next() & 1u) ? 1 : -1;
        const int32_t a1 = static_cast<int32_t>(rng.in_range(B));
        const int32_t b1 = static_cast<int32_t>(rng.in_range(B));
        const int32_t a2 = static_cast<int32_t>(rng.in_range(B));
        const int32_t b2 = static_cast<int32_t>(rng.in_range(B));
        const int filtered = hecke::sign_dot2_f32(a1, b1, a2, b2, s, B);
        const int expected = IndependentExactSign(a1, b1, a2, b2, s);
        if (filtered != expected) {
            std::cout << "  sign_dot2_f32 mismatch, violates [C, cor:fp32]:" << std::endl;
            print_case(a1, b1, a2, b2, s, B, filtered, expected);
            return code;
        }
    }
#ifndef HECKE_PURE_INTEGER
    if (totality_required) {
        const hecke::PredicateStats st = hecke::predicate_stats_fp32();
        if (st.exact_fallbacks != 0) {
            std::cout << "  sign_dot2_f32: exact_fallbacks=" << st.exact_fallbacks
                       << " != 0 for totality class B=" << B << " -- violates [C, cor:fp32]."
                       << std::endl;
            return code + 1;
        }
        if (st.fp_hits != N) {
            std::cout << "  sign_dot2_f32: fp_hits=" << st.fp_hits << " != N=" << N
                       << " for totality class B=" << B << std::endl;
            return code + 2;
        }
    }
#endif
    return 0;
}

uint32_t CheckFp32SpotCheck() {
    // Totality regime (B <= 2^10, [C, cor:fp32]).
    uint32_t rc = CheckFp32Class(uint32_t(1) << 10, 500000ull, 0xAA55AA55C3C3C3C3ull, true, 3401);
    if (rc != 0) return rc;
    // Above totality, below the exact-conversion limit: correct-or-fallback.
    rc = CheckFp32Class(uint32_t(1) << 20, 300000ull, 0x0F0F0F0FF0F0F0F0ull, false, 3411);
    if (rc != 0) return rc;
    rc = CheckFp32Class(uint32_t(1) << 24, 300000ull, 0x1234567890ABCDEFull, false, 3421);
    if (rc != 0) return rc;

    // Boundary at B = 2^10 exactly (the corollary's totality edge: E32(2^10)
    // = 0.5 exactly -- not < 0.5, so the totality guarantee here rests on
    // the products/sums at this B staying exactly representable in binary32
    // (max |a_i*b_i| = 2^20 < 2^24), not on the generic error bound being
    // strict. Confirm both delta=0 and delta=+-1 explicitly.
    for (int32_t delta : {-1, 0, 1}) {
        for (int s : {-1, 1}) {
            const int32_t B = int32_t(1) << 10;
            const int32_t margin = 5;
            const int32_t a1 = B - margin;
            const int32_t a2 = (s == -1) ? (a1 - delta) : (delta - a1);
            const int32_t b1 = 1, b2 = 1;
            const int filtered = hecke::sign_dot2_f32(a1, b1, a2, b2, s, uint32_t(B));
            const int expected = IndependentExactSign(a1, b1, a2, b2, s);
            if (filtered != expected) {
                std::cout << "  sign_dot2_f32 totality-boundary mismatch at delta=" << delta
                           << " s=" << s << std::endl;
                return 3431;
            }
        }
    }
    return 0;
}

} // namespace

uint32_t HeckeFpFilterValidation() {
    uint32_t rc = CheckBulkUniform();
    if (rc != 0) return fail(rc, "Phase A: bulk uniform sampling (>=10^7/class) vs independent __int128 sign");
    std::cout << "  Phase A: 4 x 10^7 uniform samples, 100% agreement across all classes." << std::endl;

    rc = CheckAdversarialBoundaryFp64();
    if (rc != 0) return fail(rc, "Phase B1: exact near-threshold boundary construction");
    std::cout << "  Phase B1: exact boundary constructions (all 4 classes) passed." << std::endl;

    rc = CheckAdversarialRoundingSearch();
    if (rc != 0) return fail(rc, "Phase B2: directed search for genuine-rounding near-threshold cases");
    std::cout << "  Phase B2: directed near-threshold search (B=2^26, 2^31) passed." << std::endl;

    rc = CheckInt64ToInt128Escalation();
    if (rc != 0) return fail(rc, "int64->int128 exact-backend escalation at B=2^31");

    rc = CheckOverflowGuard();
    if (rc != 0) return fail(rc, "Phase C: overflow guard (HeckeOverflow, never UB)");
    std::cout << "  Phase C: overflow guard confirmed on all crafted inputs." << std::endl;

    rc = CheckSignDet2Random();
    if (rc != 0) return fail(rc, "Phase D: sign_det2 shape-(D) spot check");
    std::cout << "  Phase D: sign_det2 spot check (2 x 10^6 samples) passed." << std::endl;

    rc = CheckFp32SpotCheck();
    if (rc != 0) return fail(rc, "Phase E: sign_dot2_f32 spot check ([C, cor:fp32])");
    std::cout << "  Phase E: sign_dot2_f32 spot check passed." << std::endl;

    return 0;
}
