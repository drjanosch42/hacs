// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S1 -- Exact quadratic points of the upper half-plane (see
// QuadPoint.h for the contract references: [C, alg:tau] step 1,
// [C, sec:fpfilter]; and for the two tiers of [C, thm:backend] added by
// hecke-extensions slot 3 WP-1).

#include "QuadPoint.h"

#include <numeric>
#include <type_traits>

#include "Predicates.h"

namespace hecke {

namespace {

inline uint64_t max4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    uint64_t m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    if (d > m) m = d;
    return m;
}

// |x| as the matching unsigned type, INT_MIN-safe (two's complement), for
// either tier.  std::gcd is not portably available on __int128 (it requires
// std::is_integral, which is an implementation extension there), so the wide
// tier gets its own Euclid -- the SAME mathematical function, so the tiers
// normalize identically.
template <typename I>
struct Unsigned;
template <>
struct Unsigned<int64_t> { using type = uint64_t; };
template <>
struct Unsigned<__int128> { using type = unsigned __int128; };

template <typename I>
inline typename Unsigned<I>::type uabs_of(I x) {
    using U = typename Unsigned<I>::type;
    return (x < 0) ? (~static_cast<U>(x) + U(1)) : static_cast<U>(x);
}

template <typename I>
inline typename Unsigned<I>::type gcd_of(typename Unsigned<I>::type a,
                                         typename Unsigned<I>::type b) {
    if constexpr (std::is_same_v<I, int64_t>) {
        return std::gcd(a, b);
    } else {
        while (b != 0) {
            const typename Unsigned<I>::type t = a % b;
            a = b;
            b = t;
        }
        return a;
    }
}

} // namespace

template <typename I>
QuadPointT<I>::QuadPointT(I u_, I v_, I w_, int32_t D_)
    : u(u_), v(v_), w(w_), D(D_) {
    if (w <= 0) throw HeckeInternalError("QuadPoint: requires w > 0");
    if (v <= 0) throw HeckeInternalError("QuadPoint: requires v > 0 (upper half plane)");
    if (D != 1 && D != 3) throw HeckeInternalError("QuadPoint: requires D in {1, 3}");
    // gcd-normalize (uabs_of is INT_MIN-safe; g >= 1 since v > 0).
    const auto g = gcd_of<I>(gcd_of<I>(uabs_of<I>(u), uabs_of<I>(v)), uabs_of<I>(w));
    if (g > 1) {
        const I gs = static_cast<I>(g); // g <= |v| <= max(I) here
        u /= gs;
        v /= gs;
        w /= gs;
    }
}

template <typename I>
QuadPointT<I> QuadPointT<I>::moebius(const Mat22& g) const {
    const int64_t dt = g.det(); // overflow-checked
    if (dt <= 0) throw HeckeInternalError("QuadPoint::moebius: requires det g > 0");
    const I Dw = static_cast<I>(D);
    const I ga = static_cast<I>(g.a), gb = static_cast<I>(g.b);
    const I gc = static_cast<I>(g.c), gd = static_cast<I>(g.d);

    // r = a u + b w, t = c u + d w (numerator/denominator real parts, cleared).
    const I r = checked_add<I>(checked_mul<I>(ga, u), checked_mul<I>(gb, w));
    const I t = checked_add<I>(checked_mul<I>(gc, u), checked_mul<I>(gd, w));
    const I v2 = checked_mul<I>(v, v);

    // u' = r t + D a c v^2 ; v' = v w det(g) ; w' = t^2 + D c^2 v^2.
    const I u_new = checked_add<I>(checked_mul<I>(r, t),
                                   checked_mul<I>(Dw, checked_mul<I>(checked_mul<I>(ga, gc), v2)));
    const I v_new = checked_mul<I>(checked_mul<I>(v, w), static_cast<I>(dt));
    const I w_new = checked_add<I>(checked_mul<I>(t, t),
                                   checked_mul<I>(Dw, checked_mul<I>(checked_mul<I>(gc, gc), v2)));

    // Constructor re-validates v' > 0, w' > 0 and gcd-normalizes. (w' > 0:
    // t = c v = 0 simultaneously would force det g = 0; v' > 0 from v, w > 0
    // and det g > 0.)
    return QuadPointT<I>(u_new, v_new, w_new, D);
}

template <typename I>
int QuadPointT<I>::cmp_abs_re_half() const {
    // sign(2|u| - w).
    const I au = (u < 0) ? checked_neg<I>(u) : u;
    if constexpr (std::is_same_v<I, int64_t>) {
        // As the 2-term dot product 2*|u| - 1*w, through the certified filter.
        const uint64_t B = max4(2u, static_cast<uint64_t>(au), static_cast<uint64_t>(w), 1u);
        return sign_dot2(2, au, 1, w, -1, B);
    } else {
        const I two_au = checked_mul<I>(static_cast<I>(2), au);
        const I diff = checked_sub<I>(two_au, w);
        return (diff > 0) - (diff < 0);
    }
}

template <typename I>
int QuadPointT<I>::cmp_norm_one() const {
    // sign(u^2 + D v^2 - w^2) = sign((u-w)(u+w) + (D v) * v).
    const I a1 = checked_sub<I>(u, w);
    const I b1 = checked_add<I>(u, w);
    const I a2 = checked_mul<I>(static_cast<I>(D), v);
    const I b2 = v;
    if constexpr (std::is_same_v<I, int64_t>) {
        const uint64_t B = max4(uabs_of<I>(a1), uabs_of<I>(b1), uabs_of<I>(a2), uabs_of<I>(b2));
        return sign_dot2(a1, b1, a2, b2, 1, B);
    } else {
        const I s = checked_add<I>(checked_mul<I>(a1, b1), checked_mul<I>(a2, b2));
        return (s > 0) - (s < 0);
    }
}

template <typename I>
int QuadPointT<I>::sign_lin_sqrt3(int64_t A, int64_t B) {
    // Decided by the signs of A, B; mixed signs by A^2 vs 3 B^2 (exact
    // squaring as the 2-term dot product A*A - (3B)*B; note A^2 = 3 B^2 has
    // no nonzero integer solutions, consistent with A + B*sqrt(3) = 0 iff
    // A = B = 0).
    if (B == 0) return (A > 0) - (A < 0);
    if (A == 0) return (B > 0) - (B < 0);
    if (A > 0 && B > 0) return 1;
    if (A < 0 && B < 0) return -1;
    const int64_t threeB = checked_mul(int64_t(3), B);
    const uint64_t bound = max4(uabs64(A), uabs64(threeB), uabs64(B), 1u);
    const int cmp = sign_dot2(A, A, threeB, B, -1, bound); // sign(A^2 - 3 B^2)
    // A > 0 > B: positive term dominates iff A^2 > 3 B^2.
    // A < 0 < B: sign flips.
    return (A > 0) ? cmp : -cmp;
}

template struct QuadPointT<int64_t>;
template struct QuadPointT<__int128>;

} // namespace hecke
