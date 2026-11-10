// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_QUAD_POINT_H
#define HECKE_QUAD_POINT_H

// Stage S1 -- Exact quadratic points of the upper half-plane.
//
// Contract: docs/latex/hecke_action.tex ([C]), [C, alg:tau] step 1 (the
// standard fundamental-domain reduction) and [C, sec:fpfilter] (the two
// branch-decision shapes (D), (Q) on cleared denominators). All predicates
// route through the certified filter hecke::sign_dot2 / exact squaring;
// all arithmetic is overflow-checked exact (throws HeckeOverflow,
// invariant I5 / [C, obl:backend]).
//
// -------------------------------------------------------------------------
// TWO TIERS (hecke-extensions slot 3, WP-1; [C, thm:backend]'s escalation
// ladder FP64 -> int64 -> __int128).
// -------------------------------------------------------------------------
// QuadPointT is templated on its coordinate type I:
//
//   QuadPoint  = QuadPointT<int64_t>    the DEFAULT tier.  Byte-for-byte the
//                                       pre-WP-1 class: same members, same
//                                       operations in the same order, and
//                                       the same FP64-filtered predicates
//                                       (hecke::sign_dot2).
//   QuadPointW = QuadPointT<__int128>   the WIDE tier.  Same mathematics,
//                                       pure exact integer arithmetic (no FP
//                                       filter -- at this width the filter's
//                                       error bound no longer holds, and the
//                                       exact route is the ground truth the
//                                       filter falls back to anyway), so a
//                                       wide-tier sign is by construction the
//                                       sign the narrow tier computes
//                                       wherever the narrow tier fits.
//
// WHY THE WIDE TIER EXISTS.  [C, alg:tau] step 1 forms
// z = (g^{-1} s_i) . pos(v) through the projective adj(g) = diag(N, 1) trick,
// i.e. one Moebius step by a matrix of norm N = p^r followed by the
// fundamental-domain reduction.  ||z|| grows to ~ ||pos||^2 N^2 (measured:
// 144.8 N^2 on the shipped 7-vertex model), and QuadPoint::moebius squares
// its input, so the int64 tier needs ~ (145 N^2)^2 < 2^63, i.e. N < 4580.
// At N >= 4580 -- the `hecke: exact multiply overflow` of docs/KNOWN_ISSUES.md
// Sec. 3, first seen between ell = 3617 and ell = 5003 -- the int64 tier of
// [C, thm:backend] is EXHAUSTED (its own criterion: the certified bound B
// exceeds 2^31), and the contract's next tier is __int128.  Nothing STORED
// widens: the reduction's output is the transporter gamma_red (||.|| ~ N) and
// a vertex index, both int64-sized, so the escalation is transient and the
// tables it feeds are unchanged.
//
// Which tier runs is decided per step-1 call by hecke::ScopedExactTier /
// escalate_on_overflow (ExactInt.h): the narrow tier is attempted first and
// the wide tier is used only after a HeckeOverflow, so every input on which
// the narrow tier completes produces byte-identical output.

#include <cstdint>

#include "ExactInt.h"

namespace hecke {

// z = (u + v*sqrt(D)*i) / w with u,v,w integers, w > 0, v > 0 (upper half
// plane), D in {1, 3}, gcd(u,v,w) = 1 (gcd-normalized on construction).
// Closed under the Moebius action of Mat22 with positive determinant.
template <typename I>
struct QuadPointT {
    I u, v, w;
    int32_t D;

    // Validates w > 0, v > 0, D in {1,3} (throws HeckeOverflow otherwise)
    // and gcd-normalizes.
    QuadPointT(I u_, I v_, I w_, int32_t D_);

    // g.z = (a z + b)/(c z + d) for g = [[a,b],[c,d]] with det g > 0
    // (checked; throws otherwise). Exact, overflow-checked:
    //   u' = (a u + b w)(c u + d w) + D a c v^2,
    //   v' = v w det(g),
    //   w' = (c u + d w)^2 + D c^2 v^2,
    // then gcd-normalized. Preserves v > 0, w > 0, D. The transform is always
    // an int64 Mat22 (its entries are coset data, bounded by N); only the
    // POINT widens.
    QuadPointT moebius(const Mat22& g) const;

    // sign(2|u| - w): compares |Re z| with 1/2 (shape (Q), routed through
    // sign_dot2 in the narrow tier, exact in the wide tier).
    int cmp_abs_re_half() const;

    // sign(u^2 + D v^2 - w^2): compares |z| with 1. Routed through sign_dot2
    // via the 2-term form (u-w)(u+w) + (D v) v in the narrow tier, exact in
    // the wide tier.
    int cmp_norm_one() const;

    // sign of A + B*sqrt(3) (A, B int64), for cross-D nearest-vertex
    // comparisons: decided by the signs of A, B and the exact-squaring
    // comparison A^2 vs 3 B^2 (a 2-term dot product through sign_dot2).
    // Tier-independent by design: both tiers reduce (A, B) by their gcd and
    // require the result to fit int64 before calling this (a named exception
    // otherwise), so there is exactly ONE nearest-vertex sign rule.
    static int sign_lin_sqrt3(int64_t A, int64_t B);

    bool operator==(const QuadPointT& rhs) const {
        return u == rhs.u && v == rhs.v && w == rhs.w && D == rhs.D;
    }

    // Exact equality against a narrow-tier point (the shipped vertex
    // positions are int64 constants; the reduced point may be wide).
    bool equals_narrow(const QuadPointT<int64_t>& p) const {
        return u == static_cast<I>(p.u) && v == static_cast<I>(p.v) &&
               w == static_cast<I>(p.w) && D == p.D;
    }

    // Exact widening (never narrowing: the wide tier is an escalation).
    static QuadPointT widen(const QuadPointT<int64_t>& p) {
        return QuadPointT(static_cast<I>(p.u), static_cast<I>(p.v),
                          static_cast<I>(p.w), p.D);
    }
};

using QuadPoint  = QuadPointT<int64_t>;
using QuadPointW = QuadPointT<__int128>;

extern template struct QuadPointT<int64_t>;
extern template struct QuadPointT<__int128>;

} // namespace hecke

#endif // HECKE_QUAD_POINT_H
