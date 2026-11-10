// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_CANONICAL_FIELD_H
#define HECKE_EXT_HECKE_CANONICAL_FIELD_H

// ===========================================================================
// (hecke-extensions slot 5, WP-7 / phase E6) THE PROJECT-CANONICAL SPLIT FIELD
// ([X] Sec. 2.2 Stage 6 `HeckeCanonicalField`; [E] `def:canon`, `cl:canon`,
//  `alg:split` (Algorithm 6.4); [OD] OQ9 (the defining polynomial) and OQ10
//  (the default report form).  Planned as slot-4 WP-10, which did not run; the
//  slot-4 text is kept verbatim in the slot-5 plan Sec. 3 WP-7.)
// ===========================================================================
//
// WHAT THIS UNIT IS.  A block of the primary decomposition carries its residue
// data in its OWN field: the witness polynomial pi_w in F_ell[X], irreducible
// of degree r = f, and one polynomial P_p per family prime with
// a_p = P_p(w-bar) in F_ell[X]/(pi_w) ([E] `cl:resfield`).  That form is free
// -- it is what the decomposition already computed -- but it is NOT comparable
// across blocks or across runs, because every block names its own field by its
// own pi_w.  Stage 6 re-embeds all of them into ONE field per (ell, r), fixed
// by (ell, r) alone:
//
//     C_{ell,r} := the CanonicalLess-least monic irreducible of degree r
//                  over F_ell,                                  [E] `def:canon`
//     F         := F_ell[Y]/(C_{ell,r}),
//
// and reports the eigenvalues as elements of F, together with their Frobenius
// orbit.  [E] `cl:canon` is then a statement anyone can re-check: C_{ell,r} is
// a deterministic function of (ell, r) and every step of its construction is
// certified.
//
// WHAT THIS UNIT IS NOT.  There is NO REPORT CONSUMER in this slot: neither
// `hecke_report/v2` nor `hecke_ext/v1` gains a key here, and nothing on any
// pipeline path calls into this file.  The consumer is E6's, in slot 6.  What
// slot 5 owes is the unit, its two settings keys and the measurement its
// performance gate needs -- the gate itself
// (restated by [OD] E6-1 as `canonical_field_s <= 0.10 * total_s` on every cell, [X] v1.10 Sec. 6.2
// E6 (d); the old `<= 2 * own_field_s` half deleted) is E6 acceptance's; no default flips here.
//
// WHY IT IS CHEAP.  Every block here has D <~ 200 and r = f is small (r = 1 on
// the whole golden grid; r = 30 in the constructed (2, 30) covering example of
// `HeckeLocalityValidation` arm 7, which is this unit's stress case).  The
// arithmetic is O(r^3 log ell) per block ([E] after `alg:split`) COUNTED IN
// F-OPERATIONS -- i.e. O(r^5 log ell) F_ell-multiplications with schoolbook
// arithmetic, which is what the slot-7 WP-4 operation counter measured on the
// pre-WP-4 route (fitted exponent 4.8-5.6 in r over r = 10..20) and what made
// it NOT negligible at r >= 11 (cells of the ell = 691 campaign).  Since
// slot-7 WP-4 the root finder runs in a flat quotient ring with the Frobenius
// precomputed (see `FindCanonicalRoot` in the .cpp): the same values, ~12-16x
// fewer F_ell-multiplications at r = 19 and ~40-70x less time.  It runs once
// per ideal in the report layer, and it never touches a result path.
//
// THE ONLY NEW ARITHMETIC is univariate polynomials over F (`FqPoly` below),
// built on top of this file's own F-element operations, which are in turn
// built on `hecke::FpPolyT<uint32_t>` and `engine::ModCtx<uint32_t>`.  [X]
// Sec. 2.2 Stage 6 prices it at "~200 lines on top of FpPolyT, confined to the
// report layer"; it is exposed in this header (rather than hidden in the .cpp)
// only so that `HeckeCanonicalFieldValidation` can exercise it directly
// against an independent oracle.
//
// DETERMINISM.  Nothing here is randomised.  `alg:split` step 1 uses a FIXED
// candidate sequence -- the elements of F in CanonicalLess order, i.e.
// `0, 1, ..., ell - 1, Y, 2Y, ...`, which is "s = 0, 1, Y, ..." of [E]
// Alg. 6.4 -- and the first candidate that splits is the one taken.  The
// search STARTS AT Y, skipping the `ell` constants, because no `s` in F_ell
// can ever split pi_w: the roots are a single Frobenius orbit, `x |-> x + s`
// commutes with Frobenius for `s` in F_ell, and both two-valued invariants the
// splitting polynomial uses (the quadratic character for odd ell, the absolute
// trace for ell = 2) are Frobenius-invariant, so the gcd is 1 or f.  That is a
// theorem, not a heuristic: the skipped candidates are PROVED useless and the
// order of what remains is untouched.  (It is also worth `ell` rejected
// candidates per splitting step -- 0.47 s on one `(160, 691)` cell before the
// skip and microseconds after.)  The root reported is `rho_canonical`, the
// CanonicalLess-least of the r conjugates, so the answer does not depend on
// which conjugate the search happened to reach.
//
// CERTIFY-AFTER-FIND, as everywhere in this repository: the search may use any
// route, the RESULT is re-derived.  `CanonicalFieldPolynomial` certifies its
// answer with `hecke::CertifyIrreducible`, and `CertifyCanonicalSplit`
// re-derives every clause of a `CanonicalSplit` -- including re-running the
// enumeration for C_{ell,r} -- from pi_w and the character alone.
//
// PLACEMENT AND DEPENDENCIES.  `namespace hecke_ext`, uint32_t throughout, and
// PIPELINE-FREE: this unit consumes `Hecke/FpPoly.h` (which for these entry
// points needs no `MatrixT`), `Hecke/HeckeSettings.h`, `ModCtx.h` and
// `EngineCheck.h`, and nothing else -- no cohomology, no coset table, no
// `Matrix`.  It is therefore bound by rules 1, 2 and 4 of
// `check_heckeext_includes.sh` like every other file in the layer, and is NOT
// named in rule 3 (that list is `HeckeSturmBound`'s own).

#include <cstdint>
#include <string>
#include <vector>

#include "../Hecke/FpPoly.h"
#include "../../ModArithmetic/ModCtx.h"

namespace hecke_ext {

// ---------------------------------------------------------------------------
// Elements of F = F_ell[Y]/(C_{ell,r}).
// ---------------------------------------------------------------------------
//
// A field element is its coefficient vector of length EXACTLY r, low -> high
// (amendment A-N4, the repository's one polynomial-coefficient order), entries
// canonical in [0, ell).  The zero element is the all-zero vector of length r
// -- deliberately NOT the empty vector, so that `FieldElt` has a fixed width
// and `less()` below is a total order on a single field.
using FieldElt = std::vector<uint32_t>;

// ---------------------------------------------------------------------------
// C_{ell,r} ([E] `def:canon`).
// ---------------------------------------------------------------------------
//
// The CanonicalLess-least monic irreducible of degree r over F_ell.
// `hecke::CanonicalLess` is "degree, then `a.coeffs < b.coeffs`", i.e.
// lexicographic on the LOW -> HIGH coefficient vector, so among the monic
// polynomials of degree r the CONSTANT coefficient c_0 is compared FIRST.
// Candidates Y^r + sum_{i<r} c_i Y^i are therefore walked as an odometer over
// (c_0, ..., c_{r-1}) whose rightmost digit turns fastest, and the first one
// `hecke::CertifyIrreducible` accepts is returned.  The one prune is the
// theorem "c_0 = 0 and r >= 2 => Y | f": that subtree is skipped whole, which
// is what makes (2, 30) reachable at all.
//
// r >= 1; r = 1 gives Y.  Throws std::logic_error (ENGINE_CHECK) on ell < 2 or
// r = 0, and on an exhausted enumeration (which cannot happen: monic
// irreducibles of degree r over F_ell exist for every r >= 1).
hecke::FpPoly CanonicalFieldPolynomial(uint32_t ell, uint32_t r);

// [X] Sec. 1.5's `field_polynomial` key, consulted.  Returns
// CanonicalFieldPolynomial(ell, r) under `least` (the default) and raises
// hecke::NotImplementedInThisBuild under `conway` -- a LEGAL value this build
// cannot honour, never a parse error and never silently degraded to `least`
// ([OD] OQ9: "Conway prepared for as a future option").
hecke::FpPoly FieldPolynomialFromSettings(uint32_t ell, uint32_t r);

// ---------------------------------------------------------------------------
// (slot-7 WP-4) THE DETERMINISTIC OPERATION COUNTER.
// ---------------------------------------------------------------------------
//
// [E] `alg:split` prices step 1 without a unit; the slot-7 diagnosis
// (hecke_extensions_slot7_plan.md Sec. 2.3 item 1) needs the count, not a
// timing.  Every counted routine adds to the counts of the CALLING THREAD's
// installed counter, if one is installed, and to nothing otherwise: the cost
// on the uninstrumented path is one thread-local load and one branch per
// counted CALL (never per inner-loop iteration -- the inner loops count into a
// local and publish once), against O(r^2) work per call.  Counts are a pure
// function of the inputs (no timing, no randomness, no thread count), so they
// are valid from any run on any box.
//
//   fl_mul        F_ell multiplications: every product of two residues, in the
//                 field multiply, its reduction fold, the quotient-ring
//                 kernels and the Frobenius matrices (reductions of an
//                 accumulator are NOT counted as multiplications)
//   f_mul         CanonicalField::mul calls (multiplications in F)
//   f_inv         CanonicalField::inv calls (an extended Euclid in F_ell[Y])
//   poly_mul      products of polynomials over F: FqMul calls, plus products
//                 in the quotient ring F[X]/(f) of the root finder
//   poly_divrem   FqDivRem calls (a division with remainder over F)
//   frob_apply    applications of a precomputed Frobenius map (F[X]/(f) or F)
//   candidates    evaluations of the splitting polynomial (one per candidate s
//                 of the fixed sequence; == CanonicalRoot::candidates_used
//                 summed over the counted calls)
struct CanonicalFieldOpCounts {
    uint64_t fl_mul = 0;
    uint64_t f_mul = 0;
    uint64_t f_inv = 0;
    uint64_t poly_mul = 0;
    uint64_t poly_divrem = 0;
    uint64_t frob_apply = 0;
    uint64_t candidates = 0;
};

// Installs `counts` as the calling thread's counter for the scope's lifetime
// (restoring whatever was installed before).  The counts ACCUMULATE: the
// caller zeroes them if it wants a fresh count.
class ScopedCanonicalFieldOpCounter {
public:
    explicit ScopedCanonicalFieldOpCounter(CanonicalFieldOpCounts& counts);
    ~ScopedCanonicalFieldOpCounter();
    ScopedCanonicalFieldOpCounter(const ScopedCanonicalFieldOpCounter&) = delete;
    ScopedCanonicalFieldOpCounter& operator=(const ScopedCanonicalFieldOpCounter&) = delete;

private:
    CanonicalFieldOpCounts* previous_;
};

// The calling thread's installed counter, or nullptr.
CanonicalFieldOpCounts* CanonicalFieldOpCounter();

// ---------------------------------------------------------------------------
// The field F itself.
// ---------------------------------------------------------------------------
class CanonicalField {
public:
    // F_ell[Y]/(C) with C = FieldPolynomialFromSettings(ell, r): C_{ell,r} of
    // `def:canon` under `field_polynomial = least` (the default); under `conway`
    // this throws hecke::NotImplementedInThisBuild (slot-7 WP-4).
    CanonicalField(uint32_t ell, uint32_t r);

    // F_ell[Y]/(C) for a caller-supplied C, which is CERTIFIED irreducible
    // here (the constructor throws otherwise).  This is what
    // `FieldPolynomialFromSettings` feeds, and what lets a validation build a
    // NON-canonical model of the same field as an independent oracle.
    CanonicalField(uint32_t ell, const hecke::FpPoly& C);

    uint32_t ell() const { return ell_; }
    uint32_t degree() const { return r_; }              // r = [F : F_ell]
    const hecke::FpPoly& defining_poly() const { return C_; }
    // True iff defining_poly() == CanonicalFieldPolynomial(ell, r).
    bool is_canonical() const;

    FieldElt zero() const { return FieldElt(r_, 0u); }
    FieldElt one() const;
    FieldElt generator() const;                          // Y
    FieldElt from_scalar(uint32_t a) const;
    FieldElt from_poly(const hecke::FpPoly& p) const;    // p mod C
    hecke::FpPoly to_poly(const FieldElt& a) const;      // trimmed, over F_ell

    bool is_zero(const FieldElt& a) const;
    FieldElt add(const FieldElt& a, const FieldElt& b) const;
    FieldElt sub(const FieldElt& a, const FieldElt& b) const;
    FieldElt neg(const FieldElt& a) const;
    FieldElt mul(const FieldElt& a, const FieldElt& b) const;
    FieldElt inv(const FieldElt& a) const;               // throws on zero
    FieldElt pow(const FieldElt& a, uint64_t e) const;   // 0^0 = 1
    FieldElt frobenius(const FieldElt& a) const;         // a^ell

    // P(x) for P in F_ell[X] and x in F (Horner).
    FieldElt eval(const hecke::FpPoly& P, const FieldElt& x) const;

    // THE FIXED CANDIDATE SEQUENCE of [E] Alg. 6.4 step 1: the elements of F
    // in CanonicalLess order -- the zero element, then the elements of degree
    // 0, 1, ..., r - 1, and inside each degree lexicographically on the
    // low -> high coefficient vector (so c_0 is the most significant digit and
    // the leading c_d runs over the ell - 1 non-zero values).  For ell = 2
    // that is 0, 1, Y, Y + 1, Y^2, ... -- "s = 0, 1, Y, ..." of [E] Alg. 6.4.
    // `index` must be < |F| = ell^r; ENGINE_CHECK fires otherwise.
    FieldElt element_at(uint64_t index) const;

    // CanonicalLess on field elements: by the degree of the reduced
    // representative (the zero element first), then lexicographically on the
    // low -> high coefficient vector.  This is `hecke::CanonicalLess` applied
    // to `to_poly()`, spelled out on the fixed-width vectors so that it needs
    // no allocation.  `a` and `b` must have the same length.
    static bool less(const FieldElt& a, const FieldElt& b);
    static bool equal(const FieldElt& a, const FieldElt& b) { return a == b; }

    // The canonical text `[c0,c1,...,c_{r-1}]` -- the repository's low -> high
    // coefficient-list convention, the same one every to_json() under Hecke/
    // uses.
    std::string to_text(const FieldElt& a) const;

    // (slot-7 WP-4) Read-only views for the root finder's flat quotient-ring
    // kernels: red_[i] = Y^{r+i} mod C (0 <= i < r), and the field's ModCtx.
    const std::vector<FieldElt>& reduction_rows() const { return red_; }
    const engine::ModCtx<uint32_t>& mod_ctx() const { return ctx_; }

private:
    void init();
    uint32_t ell_ = 0;
    uint32_t r_ = 0;
    hecke::FpPoly C_;
    // The modular-arithmetic context, built ONCE per field.  It used to be
    // constructed inside every add / sub / mul / neg, which is a division and
    // a regime decision per field operation -- measurable at the (198, 691)
    // cell, where the Stage-6 report form is the ext layer's dominant cost.
    engine::ModCtx<uint32_t> ctx_;
    // red_[i] holds Y^{r+i} mod C as a length-r vector, 0 <= i < r, so that a
    // product of two degree-<r polynomials reduces with one pass.
    std::vector<FieldElt> red_;
};

// ---------------------------------------------------------------------------
// Univariate polynomials over F -- the only new arithmetic of Stage 6.
// ---------------------------------------------------------------------------
//
// Coefficients low -> high, trimmed (no trailing zero element), every
// coefficient a FieldElt of the ambient field's width.  The zero polynomial is
// the empty vector.  The field is passed explicitly to every operation, so an
// FqPoly carries no back-pointer and copies freely.
struct FqPoly {
    std::vector<FieldElt> coeffs;

    bool is_zero() const { return coeffs.empty(); }
    // -1 for the zero polynomial.
    int64_t degree() const { return static_cast<int64_t>(coeffs.size()) - 1; }
};

FqPoly FqTrim(const CanonicalField& F, FqPoly a);
FqPoly FqFromFp(const CanonicalField& F, const hecke::FpPoly& p);   // coefficients in F_ell < F
FqPoly FqConstant(const CanonicalField& F, const FieldElt& c);
FqPoly FqAdd(const CanonicalField& F, const FqPoly& a, const FqPoly& b);
FqPoly FqSub(const CanonicalField& F, const FqPoly& a, const FqPoly& b);
FqPoly FqMul(const CanonicalField& F, const FqPoly& a, const FqPoly& b);
// a = q*b + rem with deg rem < deg b; throws on b == 0.
void    FqDivRem(const CanonicalField& F, const FqPoly& a, const FqPoly& b,
                 FqPoly& q, FqPoly& rem);
FqPoly FqMod(const CanonicalField& F, const FqPoly& a, const FqPoly& b);
FqPoly FqMonic(const CanonicalField& F, const FqPoly& a);          // throws on zero
FqPoly FqGcd(const CanonicalField& F, FqPoly a, FqPoly b);         // monic; gcd(0,0) = 0
// g^e mod f by square-and-multiply.
FqPoly FqPowMod(const CanonicalField& F, const FqPoly& g, uint64_t e, const FqPoly& f);
FieldElt FqEval(const CanonicalField& F, const FqPoly& a, const FieldElt& x);

// ---------------------------------------------------------------------------
// [E] `alg:split` step 1 -- a root of pi_w in F, and the canonical one.
// ---------------------------------------------------------------------------
struct CanonicalRoot {
    bool        ok = false;
    std::string failure;              // empty iff ok

    FieldElt rho;                     // the root the fixed sequence reached
    FieldElt rho_canonical;           // the CanonicalLess-least conjugate
    // rho_canonical^{ell^i} for i = 0 .. r - 1, in that order (so
    // conjugates[0] == rho_canonical).  These are the r distinct roots of pi_w.
    std::vector<FieldElt> conjugates;

    // How far into the fixed candidate sequence the search went, summed over
    // the recursion.  Recorded, never asserted: it is a property of the
    // sequence and of pi_w, and it is what makes a regression in the search
    // visible.
    uint32_t candidates_used = 0;
};

// pi_w must be monic, irreducible over F_ell and of degree exactly
// F.degree(); it then splits completely in F.  The caller's own pi_w always
// satisfies this (it is the block's certified witness polynomial); it is
// re-checked here and a violation is REPORTED in `failure`, not thrown, so a
// report layer can carry on with the own-field form.
CanonicalRoot FindCanonicalRoot(const CanonicalField& F, const hecke::FpPoly& pi_w);

// ---------------------------------------------------------------------------
// [E] `alg:split` steps 1-3 -- the split form of one block's residue data.
// ---------------------------------------------------------------------------
struct CanonicalSplit {
    bool        ok = false;
    std::string failure;

    uint32_t ell = 0;
    uint32_t r   = 0;                 // = deg pi_w = deg C = [F : F_ell]

    hecke::FpPoly field_polynomial;   // C_{ell,r}
    hecke::FpPoly witness_poly;       // pi_w

    FieldElt              rho_canonical;
    std::vector<FieldElt> conjugates;      // rho_canonical^{ell^i}, i = 0 .. r-1

    std::vector<uint64_t> primes;          // S, in the caller's order
    // a_q = P_q(rho_canonical), index-aligned with `primes`.
    std::vector<FieldElt> eigenvalues;
    // The full Frobenius orbit: orbit[i][j] = P_{primes[j]}(conjugates[i]);
    // orbit[0] == eigenvalues.  [E] `alg:split` step 2 reports all of it.
    std::vector<std::vector<FieldElt>> orbit;

    uint32_t candidates_used = 0;

    // `hecke_canonical_split/v1` -- a standalone fragment.  NO record consumes
    // it in this slot (see the header comment); it exists so that the
    // validation and `--bench ext` have one canonical rendering to hash and to
    // time.
    std::string to_json() const;
};

// The one entry point a report layer calls.  `character` is index-aligned with
// `primes` and each P_q must have degree < r.  Never throws on a mathematical
// refusal (a non-irreducible pi_w, a degree mismatch, a failed search): it
// returns `ok = false` with a named `failure`.  ENGINE_CHECK still fires on a
// programming error (mismatched array lengths, ell < 2, a P_q over the wrong
// modulus).
CanonicalSplit SplitInCanonicalField(uint32_t ell, const hecke::FpPoly& pi_w,
                                     const std::vector<uint64_t>& primes,
                                     const std::vector<hecke::FpPoly>& character);

// The same, with the field supplied (so a caller that splits many blocks of
// the same (ell, r) builds C_{ell,r} once).
CanonicalSplit SplitInCanonicalField(const CanonicalField& F, const hecke::FpPoly& pi_w,
                                     const std::vector<uint64_t>& primes,
                                     const std::vector<hecke::FpPoly>& character);

// ---------------------------------------------------------------------------
// The certificate: 0 iff every clause below re-derives.  Nothing is read back
// from `s` that is not re-computed from (s.ell, s.witness_poly, s.primes,
// s.character-as-given); in particular C_{ell,r} is re-enumerated, not
// trusted.
//
//   1  ell >= 2, r >= 1, and the stored r is deg pi_w
//   2  pi_w monic and CertifyIrreducible(pi_w) == 0
//   3  field_polynomial == FieldPolynomialFromSettings(ell, r)  (re-enumerated;
//      C_{ell,r} under `least`, NotImplementedInThisBuild under `conway`)
//   4  pi_w(rho_canonical) == 0 in F
//   5  conjugates.size() == r, conjugates[0] == rho_canonical,
//      conjugates[i+1] == conjugates[i]^ell, conjugates[r-1]^ell == conjugates[0]
//   6  the r conjugates are pairwise distinct and each is a root of pi_w
//   7  rho_canonical is the CanonicalLess-least conjugate
//   8  eigenvalues[j] == P_{primes[j]}(rho_canonical)
//   9  orbit[i][j] == P_{primes[j]}(conjugates[i]), and orbit[0] == eigenvalues
//
// A non-zero return is that clause's number.  `character` must be the same
// array `SplitInCanonicalField` was given.
// ---------------------------------------------------------------------------
uint32_t CertifyCanonicalSplit(const CanonicalSplit& s,
                               const std::vector<hecke::FpPoly>& character);

// ---------------------------------------------------------------------------
// The own-field report form of [E] `sec:field` tier (i) -- the SECONDARY
// form behind `split_field_report = own_field`, "always emitted as
// coefficient_field" whichever way the key points.  It is ZERO new arithmetic
// by construction: it renders pi_w and the P_p as they already stand.  It
// lives here so that `--bench ext` can time the two report forms through one
// interface, which is what the E6 performance gate compares.
// ---------------------------------------------------------------------------
std::string OwnFieldReportJson(uint32_t ell, uint32_t w_hat_exponent,
                               const hecke::FpPoly& pi_w,
                               const std::vector<uint64_t>& primes,
                               const std::vector<hecke::FpPoly>& character);

} // namespace hecke_ext

#endif // HECKE_EXT_HECKE_CANONICAL_FIELD_H
