// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_PREDICATES_H
#define HECKE_PREDICATES_H

// Stage S1 -- Certified floating-point sign filter with exact fallback.
//
// Contract: docs/latex/hecke_action.tex ([C]), section [C, sec:fpfilter]:
// [C, thm:fp64] (FP64 filter, threshold E(B) = 2^-50 * B^2),
// [C, cor:fptotal] (totality for B <= 2^24), [C, cor:fp32] (FP32 analogue,
// threshold 2^-21 * B^2, total iff B <= 2^10), and the binding contract box:
//
//   Priority order (binding): (1) FP64 filtered predicate; (2) exact
//   bounded-integer evaluation (int64/int128 per [C, thm:backend]);
//   (3) fixed-width wide integers. Arbitrary-precision arithmetic must not
//   be linked. The filter constants E(B) = 2^-50 B^2 (FP64) and 2^-21 B^2
//   (FP32) with the CERTIFIED B from [C, lem:bounds](5) are the only
//   admissible thresholds.
//
// Under the CMake option HECKE_PURE_INTEGER the FP fast path is compiled out
// entirely (exact integer path only); both builds must produce bit-identical
// results ([C, obl:fp], invariant I4).

#include <cstdint>

#include "ExactInt.h"

namespace hecke {

// sign(a1*b1 + s*a2*b2), s = +-1, with |a1|,|b1|,|a2|,|b2| <= B. Returns
// -1/0/+1, provably correct ([C, thm:fp64]).
//
// Requirements (hard-checked; violation throws HeckeOverflow, per the
// abort-rather-than-corrupt rule [C, obl:backend]):
//   * B <= 2^53 (exact conversion of the inputs to binary64),
//   * |a_i|, |b_i| <= B,
//   * s in {+1, -1}.
//
// FP64 fast path: x_hat computed in the EXACT evaluation order
// fl(fl(a1*b1) +- fl(a2*b2)); if |x_hat| > E(B) = 2^-50 * B^2 (STRICT), the
// sign of x_hat is certified equal to sign(x). For B <= 2^24 the filter is
// total ([C, cor:fptotal]): |x_hat| <= E(B) certifies x = 0. Otherwise the
// exact int64 -> __int128 fallback of [C, thm:backend] decides.
// In debug (asserting) builds every FP verdict is re-checked exactly.
int sign_dot2(int64_t a1, int64_t b1, int64_t a2, int64_t b2, int s, uint64_t B);

// Predicate shape (D) of [C, sec:fpfilter]: sign(det m) = sign(ad - bc),
// routed through sign_dot2 with B = max |entry| (requires B <= 2^53).
int sign_det2(const Mat22& m);

// FP32 variant ([C, cor:fp32]): threshold 2^-21 * B^2; requires B <= 2^24
// (exact conversion to binary32); total iff B <= 2^10. Provided for future
// GPU inner loops on the fixed model constants ([C, sec:implementation]
// item 7); otherwise prefer sign_dot2.
int sign_dot2_f32(int32_t a1, int32_t b1, int32_t a2, int32_t b2, int s, uint32_t B);

// Counters for the S1V dead-code/threshold validation (oracle O13):
// fp_hits counts calls decided by the FP fast path (including totality
// zero-decisions), exact_fallbacks counts calls decided by the exact
// integer path. Under HECKE_PURE_INTEGER every call is an exact_fallback.
//
// PER THREAD (hecke-extensions slot 8, WP-6).  The counters are thread_local:
// predicate_stats_fp64() / predicate_stats_fp32() return, and
// reset_predicate_stats() zeroes, the CALLING thread's counters.  Every
// reader (HeckeFpFilterValidation, HeckeExactIntSmokeValidation) calls the
// predicates and reads the counters on one thread, so each count it asserts
// is unchanged.  They were process-wide and non-atomic -- a data race as
// soon as two threads evaluate a predicate, which a parallel campaign ext pass
// does (SL2ZTessellationTables, HeckeInvolution, HeckeBoundaryDatum all call
// them outside hecke::tau_construction_mutex()).  A counter never feeds a
// computed value.
struct PredicateStats {
    uint64_t fp_hits = 0;
    uint64_t exact_fallbacks = 0;
};

// Mutable global stats, one per predicate family.
PredicateStats& predicate_stats_fp64();
PredicateStats& predicate_stats_fp32();
void reset_predicate_stats(); // zeroes both

} // namespace hecke

#endif // HECKE_PREDICATES_H
