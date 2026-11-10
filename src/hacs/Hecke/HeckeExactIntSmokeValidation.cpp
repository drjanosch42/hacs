// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S1 smoke validation (see HeckeExactIntSmokeValidation.h).
//
// Exercised here (per the S1 Definition of Done in
// docs/implementation/hecke_action_implementation_plan.md):
//   1. Mat22 group identities S^4 = (ST)^6 = I, S^2 = (ST)^3 = -I, plus
//      inverse/reduction/negation sanity and Mat22W/GroupElt round trips.
//   2. Overflow ABORT (HeckeOverflow exception, never UB/corruption) on a
//      2^40 * 2^40 product and related overflow-checked operations.
//   3. >= 10^5 random sign_dot2 agreements between the FP-filtered path and
//      an independent exact __int128 reference (in a debug build the
//      predicate additionally self-asserts FP-vs-exact agreement on every
//      fast-path verdict); the same for sign_dot2_f32 and sign_det2.
//   4. QuadPoint moebius/predicate spot checks on exactly known points.

#include "HeckeExactIntSmokeValidation.h"

#include <cstdint>
#include <iostream>

#include "ExactInt.h"
#include "Predicates.h"
#include "QuadPoint.h"

namespace {

using hecke::Mat22;
using hecke::Mat22W;
using hecke::GroupElt;
using hecke::HeckeOverflow;
using hecke::QuadPoint;

uint32_t fail(uint32_t code, const char* what) {
    std::cout << "HeckeExactIntSmokeValidation: FAILED check " << code
              << " (" << what << ")" << std::endl;
    return code;
}

// Deterministic xorshift64 stream (fixed seed; no <random> to keep the
// stream reproducible across standard libraries).
struct XorShift64 {
    uint64_t state;
    explicit XorShift64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return x;
    }
    // Roughly uniform in [-B, B] (modulo bias irrelevant for this test).
    int64_t next_in(uint64_t B) {
        const uint64_t span = 2 * B + 1; // B <= 2^53, no overflow
        return static_cast<int64_t>(next() % span) - static_cast<int64_t>(B);
    }
};

// Independent exact reference: sign(a1*b1 + s*a2*b2) in __int128
// (always exact for |inputs| <= 2^53).
int reference_sign(int64_t a1, int64_t b1, int64_t a2, int64_t b2, int s) {
    const __int128 x = static_cast<__int128>(a1) * b1 +
                       static_cast<__int128>(s) * (static_cast<__int128>(a2) * b2);
    return (x > 0) - (x < 0);
}

uint32_t CheckMat22GroupIdentities() {
    const Mat22 I = Mat22::I();
    const Mat22 negI = I.neg();
    const Mat22 S = Mat22::S();
    const Mat22 ST = Mat22::ST();
    const Mat22 TS = Mat22::TS();
    const Mat22 T = Mat22::T();

    // Generator sanity: ST == S*T, TS == T*S, all unimodular.
    if (!(S * T == ST)) return 101;
    if (!(T * S == TS)) return 102;
    if (T.det() != 1 || S.det() != 1 || ST.det() != 1 || TS.det() != 1) return 103;

    // S^2 = (ST)^3 = -I ; S^4 = (ST)^6 = I.
    const Mat22 S2 = S * S;
    if (!(S2 == negI)) return 104;
    if (!(S2 * S2 == I)) return 105;
    const Mat22 ST3 = ST * ST * ST;
    if (!(ST3 == negI)) return 106;
    if (!(ST3 * ST3 == I)) return 107;
    const Mat22 TS3 = TS * TS * TS;
    if (!(TS3 == negI)) return 108;

    // inv_unimodular round trips, incl. a longer random word in {T, T^-1, S}.
    if (!(S * S.inv_unimodular() == I)) return 109;
    if (!(T.inv_unimodular() * T == I)) return 110;
    XorShift64 rng(0x53315331u);
    Mat22 gamma = I;
    for (int k = 0; k < 30; ++k) {
        switch (rng.next() % 3) {
            case 0: gamma = gamma * T; break;
            case 1: gamma = gamma * T.inv_unimodular(); break;
            default: gamma = gamma * S; break;
        }
    }
    if (gamma.det() != 1) return 111;
    if (!(gamma * gamma.inv_unimodular() == I)) return 112;
    if (!(gamma.neg().neg() == gamma)) return 113;
    if (gamma.max_abs() < 1) return 114;

    // reduced_mod canonical representatives in [0, m).
    const std::array<uint32_t, 4> s_mod5 = S.reduced_mod(5);
    if (!(s_mod5[0] == 0 && s_mod5[1] == 4 && s_mod5[2] == 1 && s_mod5[3] == 0)) return 115;
    const std::array<uint32_t, 4> t_mod2 = T.reduced_mod(2);
    if (!(t_mod2[0] == 1 && t_mod2[1] == 1 && t_mod2[2] == 0 && t_mod2[3] == 1)) return 116;

    // Mat22W wide tier mirrors Mat22.
    const Mat22W SW = Mat22W::from(S);
    if (!(SW * SW == Mat22W::from(negI))) return 117;
    if (SW.det() != 1) return 118;
    if (!(SW.inv_unimodular() == Mat22W::from(S.inv_unimodular()))) return 119;
    if (!(SW.reduced_mod(5) == s_mod5)) return 120;

    // GroupElt early-reduction triple (invariant I6).
    GroupElt g = GroupElt::from_exact(ST, 101u, 8u);
    if (!g.exact.has_value() || !(*g.exact == ST)) return 121;
    if (!(g.mod_ell == ST.reduced_mod(101u))) return 122;
    if (!(g.mod_N == ST.reduced_mod(8u))) return 123;
    g.forget_exact();
    if (g.exact.has_value()) return 124;

    return 0;
}

uint32_t CheckOverflowAborts() {
    const int64_t big = int64_t(1) << 40; // 2^40
    const Mat22 X{ big, big, big, big };

    // 2^40 * 2^40 = 2^80 overflows int64: the product must ABORT via
    // HeckeOverflow (exception, not UB / silent wrap).
    bool threw = false;
    try {
        const Mat22 Y = X * X;
        (void)Y;
    } catch (const HeckeOverflow&) {
        threw = true;
    }
    if (!threw) return 201;

    // det() of diag(2^40, 2^40) also overflows.
    threw = false;
    try {
        (void)Mat22{ big, 0, 0, big }.det();
    } catch (const HeckeOverflow&) {
        threw = true;
    }
    if (!threw) return 202;

    // max_abs / neg on an INT64_MIN entry must abort, never invoke UB.
    threw = false;
    try {
        (void)Mat22{ INT64_MIN, 0, 0, 1 }.max_abs();
    } catch (const HeckeOverflow&) {
        threw = true;
    }
    if (!threw) return 203;
    threw = false;
    try {
        (void)Mat22{ INT64_MIN, 0, 0, 1 }.neg();
    } catch (const HeckeOverflow&) {
        threw = true;
    }
    if (!threw) return 204;

    // The wide tier must ACCEPT the same 2^40-entry product exactly.
    try {
        const Mat22W XW = Mat22W::from(X);
        const Mat22W YW = XW * XW;
        const __int128 expected = 2 * (static_cast<__int128>(big) * big);
        if (!(YW.a == expected && YW.b == expected && YW.c == expected && YW.d == expected)) {
            return 205;
        }
    } catch (const HeckeOverflow&) {
        return 206;
    }

    // Predicate precondition violations must abort as well.
    threw = false;
    try {
        (void)hecke::sign_dot2(1, 1, 1, 1, 1, (uint64_t(1) << 53) + 1); // B > 2^53
    } catch (const HeckeOverflow&) {
        threw = true;
    }
    if (!threw) return 207;
    threw = false;
    try {
        (void)hecke::sign_dot2(100, 1, 1, 1, 1, 10); // |a1| > B
    } catch (const HeckeOverflow&) {
        threw = true;
    }
    if (!threw) return 208;

    return 0;
}

// One magnitude class of random FP64-filter vs exact agreements. Injects a
// deterministic proportion of exact zeros and near-zero perturbations to
// exercise the threshold region and the fallback/totality branches.
uint32_t CheckSignDot2Class(uint64_t B, uint64_t samples, uint64_t seed) {
    XorShift64 rng(seed);
    for (uint64_t i = 0; i < samples; ++i) {
        const int s = (rng.next() & 1u) ? 1 : -1;
        int64_t a1 = rng.next_in(B);
        int64_t b1 = rng.next_in(B);
        int64_t a2, b2;
        const uint64_t mode = rng.next() % 8u;
        if (mode == 0) {
            // Exact zero: a1*b1 + s*(-s*a1)*b1 = 0.
            a2 = -s * a1;
            b2 = b1;
        } else if (mode == 1 && B >= 2) {
            // Near zero: perturb the cancelling pair by delta in {-1,0,1}.
            a2 = -s * a1;
            const int64_t delta = static_cast<int64_t>(rng.next() % 3u) - 1;
            b2 = b1 + delta;
            if (b2 > static_cast<int64_t>(B)) b2 -= 2;
            if (b2 < -static_cast<int64_t>(B)) b2 += 2;
        } else {
            a2 = rng.next_in(B);
            b2 = rng.next_in(B);
        }
        const int got = hecke::sign_dot2(a1, b1, a2, b2, s, B);
        const int expected = reference_sign(a1, b1, a2, b2, s);
        if (got != expected) {
            std::cout << "  sign_dot2 mismatch: a1=" << a1 << " b1=" << b1
                      << " a2=" << a2 << " b2=" << b2 << " s=" << s
                      << " B=" << B << " got=" << got
                      << " expected=" << expected << std::endl;
            return 301;
        }
    }
    return 0;
}

uint32_t CheckSignDot2Random() {
    // >= 10^5 total agreements across magnitude classes, incl. B = 2^53
    // (exercises the int64 -> __int128 exact escalation) and the totality
    // classes B <= 2^24.
    struct ClassSpec { uint64_t B; uint64_t samples; uint64_t seed; };
    const ClassSpec total_classes[] = {
        { uint64_t(1) << 10, 30000, 0xA5A5A5A5DEADBEEFull },
        { uint64_t(1) << 24, 30000, 0x0123456789ABCDEFull },
    };
    const ClassSpec fallback_classes[] = {
        { uint64_t(1) << 26, 30000, 0xFEDCBA9876543210ull },
        { uint64_t(1) << 31, 30000, 0x13579BDF2468ACE0ull },
        { uint64_t(1) << 53, 30000, 0xC001D00DCAFEF00Dull },
    };

    hecke::reset_predicate_stats();
    for (const ClassSpec& cls : total_classes) {
        const uint32_t rc = CheckSignDot2Class(cls.B, cls.samples, cls.seed);
        if (rc != 0) return rc;
    }
#ifndef HECKE_PURE_INTEGER
    // Below the totality threshold B <= 2^24 the exact fallback is provably
    // dead code ([C, cor:fptotal]).
    if (hecke::predicate_stats_fp64().exact_fallbacks != 0) return 302;
    if (hecke::predicate_stats_fp64().fp_hits == 0) return 303;
#endif

    for (const ClassSpec& cls : fallback_classes) {
        const uint32_t rc = CheckSignDot2Class(cls.B, cls.samples, cls.seed);
        if (rc != 0) return rc;
    }

    // sign_det2 spot checks (shape (D)).
    if (hecke::sign_det2(Mat22::S()) != 1) return 304;
    if (hecke::sign_det2(Mat22{ 2, 3, 4, 5 }) != -1) return 305;
    if (hecke::sign_det2(Mat22{ 1, 2, 2, 4 }) != 0) return 306;

    return 0;
}

uint32_t CheckSignDot2F32Random() {
    struct ClassSpec { uint32_t B; uint64_t samples; uint64_t seed; };
    const ClassSpec classes[] = {
        { uint32_t(1) << 10, 20000, 0xB16B00B5F00DFACEull }, // total regime
        { uint32_t(1) << 20, 20000, 0xDEADC0DE12345678ull },
        { uint32_t(1) << 24, 20000, 0x0F1E2D3C4B5A6978ull },
    };
    hecke::reset_predicate_stats();
    bool first_class = true;
    for (const ClassSpec& cls : classes) {
        XorShift64 rng(cls.seed);
        for (uint64_t i = 0; i < cls.samples; ++i) {
            const int s = (rng.next() & 1u) ? 1 : -1;
            const int32_t a1 = static_cast<int32_t>(rng.next_in(cls.B));
            const int32_t b1 = static_cast<int32_t>(rng.next_in(cls.B));
            int32_t a2, b2;
            if (rng.next() % 8u == 0) {
                a2 = static_cast<int32_t>(-s) * a1; // exact zero
                b2 = b1;
            } else {
                a2 = static_cast<int32_t>(rng.next_in(cls.B));
                b2 = static_cast<int32_t>(rng.next_in(cls.B));
            }
            const int got = hecke::sign_dot2_f32(a1, b1, a2, b2, s, cls.B);
            const int expected = reference_sign(a1, b1, a2, b2, s);
            if (got != expected) {
                std::cout << "  sign_dot2_f32 mismatch: a1=" << a1 << " b1=" << b1
                          << " a2=" << a2 << " b2=" << b2 << " s=" << s
                          << " B=" << cls.B << " got=" << got
                          << " expected=" << expected << std::endl;
                return 401;
            }
        }
#ifndef HECKE_PURE_INTEGER
        if (first_class && hecke::predicate_stats_fp32().exact_fallbacks != 0) {
            return 402; // fallback must be dead code for B <= 2^10 ([C, cor:fp32])
        }
#endif
        first_class = false;
    }
    return 0;
}

uint32_t CheckQuadPoint() {
    const Mat22 S = Mat22::S();
    const Mat22 T = Mat22::T();

    // i = (0 + 1*sqrt(1) i)/1.
    const QuadPoint zi(0, 1, 1, 1);
    if (zi.cmp_abs_re_half() != -1) return 501; // |Re i| = 0 < 1/2
    if (zi.cmp_norm_one() != 0) return 502;     // |i| = 1
    if (!(zi.moebius(S) == zi)) return 503;     // S fixes i
    if (!(zi.moebius(T) == QuadPoint(1, 1, 1, 1))) return 504; // T.i = 1 + i

    // rho = (1 + sqrt(3) i)/2, fixed by TS; S.rho = (-1 + sqrt(3) i)/2.
    const QuadPoint rho(1, 1, 2, 3);
    if (rho.cmp_abs_re_half() != 0) return 505; // |Re rho| = 1/2
    if (rho.cmp_norm_one() != 0) return 506;    // |rho| = 1
    if (!(rho.moebius(S) == QuadPoint(-1, 1, 2, 3))) return 507;
    if (!(rho.moebius(Mat22::TS()) == rho)) return 508;

    // 2i: |2i| > 1, Re = 0; also exercises gcd normalization ((0,4,2,1) -> (0,2,1,1)).
    const QuadPoint twoi(0, 4, 2, 1);
    if (!(twoi == QuadPoint(0, 2, 1, 1))) return 509;
    if (twoi.cmp_norm_one() != 1) return 510;
    if (twoi.cmp_abs_re_half() != -1) return 511;
    // S.(2i) = i/2: |i/2| < 1.
    if (twoi.moebius(S).cmp_norm_one() != -1) return 512;

    // Positive-determinant non-unimodular action (Hecke translate):
    // diag(1,2).(2i) = i, diag(2,1).(i) = 2i.
    if (!(twoi.moebius(Mat22{ 1, 0, 0, 2 }) == zi)) return 513;
    if (!(zi.moebius(Mat22{ 2, 0, 0, 1 }) == twoi)) return 514;

    // sign_lin_sqrt3 table (sqrt(3) = 1.7320508...).
    if (QuadPoint::sign_lin_sqrt3(0, 0) != 0) return 515;
    if (QuadPoint::sign_lin_sqrt3(5, 0) != 1) return 516;
    if (QuadPoint::sign_lin_sqrt3(0, -2) != -1) return 517;
    if (QuadPoint::sign_lin_sqrt3(1, -1) != -1) return 518;  // 1 - sqrt(3) < 0
    if (QuadPoint::sign_lin_sqrt3(2, -1) != 1) return 519;   // 2 - sqrt(3) > 0
    if (QuadPoint::sign_lin_sqrt3(-5, 3) != 1) return 520;   // -5 + 3 sqrt(3) > 0
    if (QuadPoint::sign_lin_sqrt3(-7, 4) != -1) return 521;  // -7 + 4 sqrt(3) < 0
    if (QuadPoint::sign_lin_sqrt3(97, -56) != 1) return 522; // 97/56 > sqrt(3)
    if (QuadPoint::sign_lin_sqrt3(-97, 56) != -1) return 523;

    return 0;
}

} // namespace

uint32_t HeckeExactIntSmokeValidation() {
    uint32_t rc = CheckMat22GroupIdentities();
    if (rc != 0) return fail(rc, "Mat22 group identities");
    rc = CheckOverflowAborts();
    if (rc != 0) return fail(rc, "overflow abort behaviour");
    rc = CheckSignDot2Random();
    if (rc != 0) return fail(rc, "sign_dot2 FP-filter vs exact agreement");
    rc = CheckSignDot2F32Random();
    if (rc != 0) return fail(rc, "sign_dot2_f32 FP-filter vs exact agreement");
    rc = CheckQuadPoint();
    if (rc != 0) return fail(rc, "QuadPoint predicates");
    return 0;
}
