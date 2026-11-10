// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_PREDICTIONS_H
#define HECKE_EXT_HECKE_PREDICTIONS_H

// ===========================================================================
// (hecke-extensions slot 4, WP-7 / phase E6) BERNOULLI NUMBERS MOD ell AND THE
// KUMMER / HARDER DIVISIBILITY PREDICATE
// ([X] Sec. 2.1 "Bernoulli numbers mod ell", Sec. 2.2 Stage 6 `HeckePredictions`;
//  [E] the Harder criterion cl:harder and the Kummer list of Claim 1.8.)
// ===========================================================================
//
// WHAT THIS UNIT IS FOR.  Slot-2 WP-7 shipped Harder's criterion against a
// PINNED table of num(B_k/k) limited to k <= 26 (HeckeExt/HeckeBoundary.cpp,
// `pinned_bernoulli_numerator`), which is exactly the range of [E] Claim 1.8's
// Kummer list.  This unit is what retires that pin: it computes B_w in F_ell
// directly, so the criterion becomes available at every k the model can reach.
//
// THE UNIT DOES NOT REWIRE HeckeBoundary.  The slot-4 plan (WP-7 scope item 1)
// keeps the pinned table in place, because swapping it would move
// HeckeBoundaryParabolicValidation's RECORDED `harder_applicable` for k > 26.
// That rewiring is slot 5's, together with the record integration.  What WP-7
// owes is the unit and the demonstration that it reproduces the pin entrywise
// where the two overlap -- HeckePredictionsValidation arm A.
//
// THE ARITHMETIC.  For 0 <= w < ell - 1 the Bernoulli number B_w is
// ell-INTEGRAL: by von Staudt-Clausen den(B_w) = prod_{(p-1) | w} p, and
// (ell - 1) | w is impossible for 0 < w < ell - 1 (and w = 0 gives B_0 = 1).
// So B_w mod ell is well defined and the standard recursion
//
//     sum_{j=0}^{w} C(w+1, j) B_j = 0,      i.e.   B_w = -(w+1)^{-1} sum_{j<w} C(w+1, j) B_j,
//
// may be run verbatim in F_ell: every B_j that enters has j <= w < ell - 1 and
// is therefore itself ell-integral, and the only division is by w + 1 <= ell - 1.
// No bignum, no rational arithmetic -- this is [X] Sec. 2.1's point, and it is
// what settles [PA A9]'s Bernoulli question.
//
// CONVENTION.  The recursion fixes B_1 = -1/2 (the "B_1^-" convention).  Only
// even w matters downstream, where the two conventions agree.
//
// OUTSIDE THE RANGE the unit REFUSES with the named error BernoulliOutOfRange
// rather than returning a wrong value: at w = ell - 1 von Staudt-Clausen puts
// ell in the denominator of B_w and there is no residue to return.
//
// THE KUMMER PREDICATE.  `kummer_divides(k, ell)` decides ell | num(B_k/k),
// the condition of [E] cl:harder ("r^1 = 0 on the Eisenstein class").  It is
// total on even k >= 2 and every prime ell >= 2, by three provable branches:
//
//   (ell - 1) | k          -> FALSE.  von Staudt-Clausen puts ell in den(B_k),
//                             so ell does not divide num(B_k), hence not
//                             num(B_k/k) either (which divides num(B_k)).
//   k < ell - 1            -> B_k/k mod ell computed directly ("direct").
//   otherwise              -> KUMMER CONGRUENCE.  With k' = k mod (ell - 1)
//                             (even, 2 <= k' <= ell - 3, non-zero by the first
//                             branch) one has B_k/k == B_{k'}/k' in F_ell, and
//                             B_k/k is ell-integral because (ell - 1) does not
//                             divide k.  So the verdict is read off B_{k'}/k'.
//
// and two degenerate ones: B_k = 0 for odd k >= 3 (so ell | 0, TRUE), and
// B_1/1 = -1/2 whose numerator is -1 (FALSE).  k = 0 is the one inapplicable
// input (B_0/0).
//
// PLACEMENT.  HeckeExt/, like every extension-layer unit; the consumers are
// E6's atlas and, in slot 5, HeckeBoundary itself.  The TU consumes NOTHING of
// the pipeline -- only engine::ModCtx and EngineCheck.h -- so it is built and
// validated in isolation, exactly like HeckeSturmBound.  (It is not named in
// check_heckeext_includes.sh's rule 3, which is HeckeSturmBound's own
// allow-list; rules 1, 2 and 4 bind it like every other file in the layer.)

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// The named error.  B_w has no residue mod ell outside 0 <= w < ell - 1.
// ---------------------------------------------------------------------------
struct BernoulliOutOfRange : std::runtime_error {
    explicit BernoulliOutOfRange(const std::string& msg)
        : std::runtime_error("hecke_ext: " + msg) {}
};

// The largest w for which bernoulli_mod(w, ell) is defined: ell - 2.
uint32_t bernoulli_max_index(uint32_t ell);

// B_0, B_1, ..., B_{w_max} in F_ell, in one O(w_max^2) pass.
// Throws BernoulliOutOfRange if w_max > ell - 2.
std::vector<uint32_t> bernoulli_mod_table(uint32_t w_max, uint32_t ell);

// B_w in F_ell.  Throws BernoulliOutOfRange if w >= ell - 1.
uint32_t bernoulli_mod(uint32_t w, uint32_t ell);

// ---------------------------------------------------------------------------
// The Kummer / Harder divisibility predicate, in its reporting form.
// ---------------------------------------------------------------------------
struct KummerVerdict {
    bool applicable = false;  // false only for k = 0
    bool divides = false;     // ell | num(B_k / k)
    uint32_t k = 0;
    uint32_t ell = 0;
    uint32_t k_reduced = 0;         // the representative actually evaluated
    uint32_t b_over_k_mod_ell = 0;  // B_{k_reduced} / k_reduced in F_ell, where evaluated
    // "direct" | "kummer_congruence" | "von_staudt_clausen" | "odd_k_vanishes"
    // | "k_eq_one" | "k_eq_zero"
    std::string reason;

    std::string to_string() const;
};

KummerVerdict kummer_verdict(uint32_t k, uint32_t ell);

// The plain predicate.  Throws BernoulliOutOfRange when the verdict is not
// applicable (k = 0 only).
bool kummer_divides(uint32_t k, uint32_t ell);

} // namespace hecke_ext

#endif // HECKE_EXT_HECKE_PREDICTIONS_H
