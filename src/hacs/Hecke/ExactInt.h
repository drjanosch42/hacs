// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXACT_INT_H
#define HECKE_EXACT_INT_H

// Stage S1 -- Exact bounded-integer layer.
//
// Contract: docs/latex/hecke_action.tex ([C]), sections [C, sec:reduction]
// ([C, def:primitives], [C, lem:bounds]) and [C, thm:backend] /
// [C, obl:backend]. Backend ladder (binding, invariant I5): FP64-filtered
// predicate -> exact int64_t -> __int128. Arbitrary-precision arithmetic is
// BANNED and must not be linked. All exact 2x2 products are overflow-checked
// (__builtin_mul_overflow / __builtin_add_overflow); a violation of the
// certified bound ABORTS (throws hecke::HeckeOverflow) rather than corrupts.

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace hecke {

// Thrown whenever overflow-checked exact arithmetic would overflow its
// fixed-width type, or a hard arithmetic precondition is violated
// ([C, obl:backend]: abort rather than corrupt).
struct HeckeOverflow : std::runtime_error {
    explicit HeckeOverflow(const char* what_arg) : std::runtime_error(what_arg) {}
};
// A HeckeOverflow that is NOT a cap: a violated precondition (a singular matrix, (0 : 0), w <= 0, ...) or a failed internal
// consistency check (a reduction that does not terminate, a solver's self-verification).  Still a HeckeOverflow, so every
// existing handler behaves as before; a caller that must tell a genuine error from an integer cap (the pair's refusal codes,
// Integral/Descent DescentPairHecke ClassifyPairException, D4 follow-up of the D5 audit round) tests for this type first.
struct HeckeInternalError : HeckeOverflow {
    explicit HeckeInternalError(const char* what_arg) : HeckeOverflow(what_arg) {}
};

// --- Overflow-checked scalar helpers (int64_t and __int128 tiers) ----------
// Shared by ExactInt / Predicates / QuadPoint. T is int64_t or __int128.

template <typename T>
inline T checked_mul(T x, T y) {
    T r;
    if (__builtin_mul_overflow(x, y, &r)) throw HeckeOverflow("hecke: exact multiply overflow");
    return r;
}

template <typename T>
inline T checked_add(T x, T y) {
    T r;
    if (__builtin_add_overflow(x, y, &r)) throw HeckeOverflow("hecke: exact add overflow");
    return r;
}

template <typename T>
inline T checked_sub(T x, T y) {
    T r;
    if (__builtin_sub_overflow(x, y, &r)) throw HeckeOverflow("hecke: exact subtract overflow");
    return r;
}

template <typename T>
inline T checked_neg(T x) {
    // -INT64_MIN (resp. -INT128_MIN) is not representable; std::abs on it is UB,
    // so negation is always routed through this checked subtraction.
    return checked_sub<T>(T(0), x);
}

// Magnitude of an int64_t as uint64_t; safe for INT64_MIN (two's complement),
// avoiding the std::abs(INT64_MIN) UB.
inline uint64_t uabs64(int64_t x) {
    return (x < 0) ? (~static_cast<uint64_t>(x) + 1u) : static_cast<uint64_t>(x);
}

// --- Mat22: overflow-checked exact 2x2 integer matrix ----------------------
// ([C, thm:backend] (i),(ii); invariant I5.)

struct Mat22 {
    int64_t a, b, c, d; // row-major [[a,b],[c,d]]

    // Generators of SL(2,Z) used throughout ([C, sec:model]):
    //   T = [[1,1],[0,1]], S = [[0,-1],[1,0]], ST = S*T, TS = T*S.
    static Mat22 I();
    static Mat22 T();
    static Mat22 S();
    static Mat22 ST();
    static Mat22 TS();

    // Exact product; throws HeckeOverflow on int64 overflow (never corrupts).
    Mat22 operator*(const Mat22& rhs) const;

    // Inverse of a unimodular matrix; requires det == +-1 (asserted; hard
    // guard throws in release builds). Never used on det == p matrices
    // (invariant I7 / [C, obl:integral]).
    Mat22 inv_unimodular() const;

    // Determinant, overflow-checked.
    int64_t det() const;

    // ||.|| = max_ij |entry| of [C, lem:bounds]; throws if an entry is
    // INT64_MIN (magnitude not representable as int64_t).
    int64_t max_abs() const;

    bool operator==(const Mat22& rhs) const;

    // -A (for +-I bookkeeping); overflow-checked.
    Mat22 neg() const;

    // (hecke-extensions slot 6, WP-1) The adjugate [C, sec:contragredient]:
    // adj [[a,b],[c,d]] = [[d,-b],[-c,a]], overflow-checked.  Defined for
    // EVERY matrix (no det hypothesis), and satisfies [C, eq:adjrules]:
    // g adj g = adj g g = (det g) I, adj(gh) = adj(h) adj(g), adj adj g = g,
    // adj I = I -- so adj gamma = gamma^{-1} on SL(2,Z).  The contragredient
    // action rho_n^vee(g) := rho_n(adj g)^T ([C, def:contragredient]) is built
    // on it.  (The inline adjugates of HeckeTauTable.cpp / BarCocycleH1.cpp
    // are deliberately NOT migrated onto this -- slot-6 plan Sec. 3.15.)
    Mat22 adj() const;

    // Canonical representative of each entry in [0, m).
    std::array<uint32_t, 4> reduced_mod(uint32_t m) const;

    // Stage S13: canonical representative of each entry in [0, m) for a 64-bit
    // modulus m < 2^64 (coefficient prime ell in the extended regime). Exact
    // integer only (no ModArithmetic dependency; HeckeCore stays pure).
    std::array<uint64_t, 4> reduced_mod64(uint64_t m) const;
};

// --- Mat22W: wide escalation tier (__int128) --------------------------------
// Same operations as Mat22; used only if the certified bound B of
// [C, thm:backend] exceeds the int64 regime (2B^2 >= 2^63, i.e. B > 2^31).
// Next (unreachable, [C, thm:backend](3)) tier would be fixed-width 192-bit;
// arbitrary precision is banned.

struct Mat22W {
    __int128 a, b, c, d; // row-major [[a,b],[c,d]]

    static Mat22W I();
    static Mat22W from(const Mat22& m); // exact widening

    Mat22W operator*(const Mat22W& rhs) const; // throws HeckeOverflow on int128 overflow
    Mat22W inv_unimodular() const;             // requires det == +-1; asserts
    __int128 det() const;                      // overflow-checked
    __int128 max_abs() const;                  // throws if an entry is INT128_MIN
    bool operator==(const Mat22W& rhs) const;
    Mat22W neg() const;
    std::array<uint32_t, 4> reduced_mod(uint32_t m) const;
};

// --- The exact-tier switch ([C, thm:backend]'s ladder, made observable) -----
// (hecke-extensions slot 3, WP-1.)
//
// [C, thm:backend] escalates FP64-filtered -> exact int64 -> __int128. The
// first two steps are inside the predicates; the third was, until WP-1,
// unreachable code (`Mat22W` had no consumer). WP-1 gives it its one real
// consumer -- [C, alg:tau] step 1's fundamental-domain reduction, whose
// `QuadPoint` coordinates reach ~ 145 N^2 and are then squared, so the int64
// tier is exhausted at N >= ~4580 (docs/KNOWN_ISSUES.md Sec. 3).
//
// The policy is ATTEMPT-THEN-ESCALATE, never a precomputed threshold:
//   * `Auto` (the default) runs the narrow tier and falls back to the wide
//     tier ONLY after a HeckeOverflow. Every input on which the narrow tier
//     completes therefore produces byte-identical output -- bit-identity is a
//     property of the control flow, not of a test;
//   * `ForceWide` runs the wide tier unconditionally. It exists for
//     HeckeUEllReachValidation's forced-wide arm, which asserts the two tiers
//     agree entrywise where both complete;
//   * `ForceNarrow` refuses to escalate, so the pre-WP-1 failure is
//     reproducible on demand. It is the NEGATIVE CONTROL: at ell >= 4580 the
//     narrow tier must still throw HeckeOverflow, which is what proves the
//     wide tier is what buys the reach.
//
// The value is thread_local and is NOT propagated onto engine task threads:
// tau construction is deliberately off the tape (matrix engine,
// Phase 4-5), so the tier in force is always the constructing thread's.
// exact_tier_escalations() counts the Auto-mode escalations that actually
// fired, for the reach validation and the report.

enum class ExactTier : uint32_t { Auto = 0, ForceWide = 1, ForceNarrow = 2 };

ExactTier current_exact_tier();
void set_exact_tier(ExactTier t);

// Number of Auto-mode escalations on THIS thread since the last reset.
uint64_t exact_tier_escalations();
void reset_exact_tier_escalations();
void note_exact_tier_escalation();

class ScopedExactTier {
    ExactTier saved_;
public:
    explicit ScopedExactTier(ExactTier t) : saved_(current_exact_tier()) { set_exact_tier(t); }
    ~ScopedExactTier() { set_exact_tier(saved_); }
    ScopedExactTier(const ScopedExactTier&) = delete;
    ScopedExactTier& operator=(const ScopedExactTier&) = delete;
};

// Runs `narrow`; on HeckeOverflow (and only then) runs `wide`, counting the
// escalation. Under ExactTier::ForceWide `narrow` is not attempted at all;
// under ExactTier::ForceNarrow the HeckeOverflow propagates. Both callables
// must be side-effect free up to the throw -- the callers in HeckeTauTable
// write only local variables. A wide-tier overflow is never swallowed: it
// propagates as the named exception, so the failure stays loud
// ([C, obl:backend]).
template <typename NarrowFn, typename WideFn>
inline void escalate_on_overflow(NarrowFn&& narrow, WideFn&& wide) {
    const ExactTier tier = current_exact_tier();
    if (tier == ExactTier::ForceWide) {
        wide();
        return;
    }
    if (tier == ExactTier::ForceNarrow) {
        narrow();
        return;
    }
    try {
        narrow();
        return;
    } catch (const HeckeOverflow&) {
        note_exact_tier_escalation();
    }
    wide();
}

// --- The stored wide tier of [C, thm:backend] (slot-24 ELL-W) ----------------
//
// [C, lem:bounds](5) / [C, obl:backend](ii): the narrow tier certifies
// 2 B^2 < 2^63 (B <= 2^31 - 1, i.e. N <= 23170 when B = 4 N^2).  Above it the
// STORED transporters still fit int64 -- the recount of B_tab is exact in
// int64 and the wide clause 2 N B <= 2^63 - 1 gives B < 2^61 (see below) -- so the
// storage stays `Mat22`.  What the narrow criterion protected is the
// arithmetic on two stored-class operands: a 2 x 2 product or a determinant
// forms intermediates of magnitude <= ||A|| ||B|| <= B^2, which leaves int64
// once B > 2^31.  The wide tier forms exactly those operations in __int128:
//
//   * PROOF of exactness.  Under the wide clause (N >= 2) B < 2^61, so every
//     stored-class operand has ||.|| <= B < 2^61, each of the two products in
//     an entry (or in a determinant) has magnitude < 2^122 and their sum
//     < 2^123 < 2^127.  (An operand outside the stored class -- the (T1)
//     conjugate, ||.|| up to 2 N^2 B -- is not covered by this bound; the
//     Mat22W operations are themselves checked, so it can only abort):
//     the __int128 evaluation is exact (and is overflow-checked anyway).
//   * The RESULT is narrowed back to int64 with a check.  A result that does
//     not fit int64 throws HeckeOverflow ([C, obl:backend](iii): abort,
//     never wrap); no value is ever produced from a wrapped intermediate.
//   * BYTE IDENTITY.  mul_tiered / det_tiered evaluate the int64 expression
//     FIRST, exactly as `Mat22::operator*` / `Mat22::det` do, and run the
//     __int128 form only after that threw HeckeOverflow.  Wherever the old
//     code completed, the new code returns the same int64 value through the
//     same instructions -- byte identity is a property of the control flow.
//   * Tier policy (the slot-3 switch, unchanged): ForceNarrow rethrows (the
//     negative control: the pre-ELL-W failure stays reproducible), ForceWide
//     evaluates the __int128 form unconditionally (the forced-wide arm of
//     HeckeWideTauValidation compares it with the narrow form), Auto escalates
//     only after an overflow.
//   * The lever: HECKE_TAU_WIDE (read once per process; exactly "0" = the
//     rollback: no product escalation, and certify() clause (e) refuses every
//     table with B > 2^31 - 1 with code 51 as before).  Read-only and
//     value-neutral: it never changes a value, it only decides whether a table
//     with B > 2^31 - 1 is refused (code 51) or certified in the wide tier.
bool tau_wide_tier_enabled();
// One line "EFFECTIVE HeckeTauWide wide_tier=on|off HECKE_TAU_WIDE=<value|unset>".
const char* tau_wide_tier_effective_line();

// Escalations the wide tier actually performed on THIS thread (separate from
// exact_tier_escalations(), which counts step-1 escalations only and is pinned
// by HeckeUEllReachValidation's arms).
uint64_t wide_product_escalations();

// (ELL-W audit F3) Clause (e)'s tier assertion as a pure function of (N, B, wide allowed): 0 = certified (narrow iff
// B <= 2^31 - 1, else wide), 51 = B > 2^31 - 1 and the wide tier may not run, 52 = the wide clause 2 N B <= 2^63 - 1 fails.
// certify() and the constructor's pre-flight (B >= 4 N^2) both call it; HeckeWideTauValidation [5] tests its boundaries.
uint32_t tau_bound_clause(uint64_t N, unsigned __int128 B, bool wide_allowed);
void reset_wide_product_escalations();

// A * B on two stored-class operands (see above).
Mat22 mul_tiered(const Mat22& A, const Mat22& B);
// det(A) on a stored-class operand (intermediates ad, bc <= ||A||^2).
int64_t det_tiered(const Mat22& A);
// A^{-1} for det A == +-1, the determinant through det_tiered (the inverse's
// entries are A's own, so only the determinant can leave int64).
Mat22 inv_unimodular_tiered(const Mat22& A);

// --- GroupElt: early-reduction triple ---------------------------------------
// ([C, thm:backend], mandatory early-reduction scheme; invariant I6.)
// A group element is carried as (exact int64 2x2 [only while needed],
// gamma mod ell, gamma mod N), N = p_hecke^r. Exact integers appear only in
// (P2)/(P3)/predicates of [C, def:primitives]; all consumers see the
// reductions.

struct GroupElt {
    std::optional<Mat22> exact;      // dropped by forget_exact() when no longer needed
    std::array<uint32_t, 4> mod_ell; // gamma mod ell (coefficient prime)
    std::array<uint32_t, 4> mod_N;   // gamma mod N, N = p_hecke^r

    void forget_exact() { exact.reset(); }

    // Builds the triple from an exact matrix by eager reduction.
    static GroupElt from_exact(const Mat22& g, uint32_t ell, uint32_t N);
};

} // namespace hecke

#endif // HECKE_EXACT_INT_H
