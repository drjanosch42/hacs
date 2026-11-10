// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_IDENTITY_CERTIFICATE_H
#define HECKE_EXT_IDENTITY_CERTIFICATE_H

// hecke-extensions slot 1, WP-2 -- the E0a subset of [X, "hecke_extensions_
// implementation_plan.md"] Sec.1.3's `IdentityCertificate`; EXTENDED BY
// SLOT 5, WP-8 with identity class 5 (the Dickson multiplication L*).
// docs/implementation/hecke_extensions_slot1_plan.md Sec.3 WP-2 named four
// of the seven identity classes of [E, "hecke_extensions.tex"] Sec.5.1 --
// the ones that needed no machinery slot 1 did not build (Dickson
// multiplication L*, socle/Loewy duality (E7), the residue-character atlas
// (E6), all then unscheduled).  Slot 5's WP-8 adds the first of those three;
// slot 6 adds class 7 (socle/Loewy duality under PD, WP-3: see the
// pd_* members below) and class 6 (the residue atlas, WP-9: item 6 here):
//
//   1. Eichler-Shimura, in the ROBUST form that holds even at a Dickson-
//      exceptional locus: dim H^1 = dim M_{n+2} + dim S_{n+2} + dim H^0
//      ([C, thm:relations] background; the classical special case is
//      dim H^0 = 0, dim H^1 = 2 dim S_{n+2} + 1 = dim M_{n+2} + dim S_{n+2});
//   2. Hecke commutativity: [A_p, A_p'] = 0 ([C, obl:algebra]);
//   3. d_2 = 0: dim H^2 = 0 (the vanishing that stops [E Sec.5.1]'s complex
//      at H^1, n even, ell in R^x);
//   4. the p^2 recursion with the FULL T_{p^2} and coefficient p^{n+1}
//      ([C, thm:relations], the err:e1 correction that T_{p^2} is the FULL
//      operator, not the primitive double coset): A_{p^2} = A_p^2 - p^{n+1} I.
//   5. (slot 5, WP-8) DICKSON MULTIPLICATION, [E Sec.5.1] item 6's `L.` half
//      and [E, cl:Tlnil]'s "Hecke-equivariant up to a det-twist": with
//      m := n - ell - 1 and L_* the map H^1(Ga; M_m) -> H^1(Ga; M_n) induced
//      by multiplication by the Dickson invariant
//      (HeckeExt/HeckeDicksonMultiplication),
//
//          A^{(1)}_n(T_p) . L_*  ==  p . L_* . A^{(1)}_m(T_p)   in M(F_ell).
//
//      The scalar is p^1 and not p^j for any other j: the cochain operator
//      Theta_g of [C, eq:xiexplicit] is a sum of terms each carrying EXACTLY
//      ONE coefficient-module matrix of determinant p (the coset
//      representatives that flank it lie in Ga and have determinant 1), and
//      the fibre identity rho_n(A) L = det(A) L rho_m(A) therefore pulls out
//      one factor of det = p per term.  A consequence worth naming: the image
//      L_* H^1(M_m) is a T_S-SUBMODULE of X_4 -- the canonical position
//      [E, D6] reserves the `L_image` slot for -- carrying the theta-twist of
//      the T_S-module H^1(M_m), which is why the ideals it meets are the
//      twisted ones and not the same ones.
//
//   6. (slot 6, WP-9) THE RESIDUE ATLAS, [E Sec.5.1] item 6's atlas half
//      ("identity 6"; E1 identity class 6; the slot-6 plan Sec. 2.5 reserves
//      certificate code 6 for it).  The residue-character multiset of
//      H^1(Ga; M_n) equals the theta^{j_i}-twists of the weight-(n_i + 2)
//      systems over the Glover factors Sym^{n_i} (x) det^{j_i} of M_n, plus
//      the H^0-corrections.  Precisely: Hecke operators commute with the
//      connecting maps of the long exact sequence of a short exact sequence
//      of coefficient modules, the Hecke action on H^*(Ga; Sym^m (x) det^j)
//      is q^j times the one on H^*(Ga; Sym^m) (every coset matrix has
//      determinant q), and H^q vanishes for q > 2; so in the Grothendieck
//      group of T-modules
//
//        [H^1(M_n)] = sum_i theta^{j_i} [H^1(M_{n_i})]
//                     + [H^0(M_n)] - sum_i theta^{j_i} [H^0(M_{n_i})]
//                     - [H^2(M_n)] + sum_i theta^{j_i} [H^2(M_{n_i})],
//
//      a class being one Galois orbit of anemic systems (a_q)_{q in S} over
//      the canonical split field F_ell[Y]/(C_{ell,r}) ([E] `sec:field`
//      tier (ii)) with multiplicity dim_{F_ell}(block) / r, and theta^j
//      acting as a_q |-> q^j a_q.  The factor multiset (n_i, j_i) is
//      RE-DERIVED here by [E] `cl:glover`(4)'s recursion (own code, not
//      GloverFactors, so a shared bug cannot cancel), UNDER THE
//      STANDARD-MODULE CONVENTION, quoted verbatim from [E] `cl:glover`:
//
//        "Sym^n is the symmetric power of the standard representation --
//         the module rho_n of [C] def:rhon acts on, in which diag(a,d) has
//         entry(t,t) = a^{n-t} d^t -- and not of its dual; the dual would
//         shift every det-twist by -n."
//
//      `residue_convention = kContragredient` evaluates the SAME identity
//      under the abstract-contragredient reading (Sym^n(V)^* ~= Sym^n(V)
//      (x) det^{-n}: every j_i shifted by -n mod (ell - 1)).  That is the
//      NEGATIVE CONTROL of HeckeExtEndToEndValidation (audit B-1): on a
//      cell where the two readings give different right-hand sides it must
//      FAIL, which is what makes [E] `cl:glover`'s "a green validation suite
//      cannot detect the discrepancy" false for this one comparison.
//
// Certificate convention (HeckeModuleDecomposition::certify(),
// Hecke/HeckeModuleDecomposition.h): a struct holding exactly the data a
// third party needs, with a deterministic `verify()` that RE-DERIVES every
// value -- classical closed-form dimension formulas computed independently
// here (not shared code with any existing DimSk/dim-M_k copy, so a shared
// bug cannot cancel), matrix arithmetic recomputed from the stored
// A_p / A_p_prime / A_p2 -- never by re-calling the production code that
// produced the inputs. Applicability ([X Sec.1.3] "Applicability"): items 1
// and 3 gate on n > 0 even and ell >= 5 (Eichler-Shimura / d_2=0's own
// hypotheses); elsewhere they are `kNotApplicable`, never `kPassed`. Item 2
// (commutativity) and item 4 (the recursion) need only non-empty,
// commuting-shaped operator matrices -- [C, thm:relations]'s hypothesis
// 6 in R^x is a RECURSION-only requirement (see HeckeOperatorFamily.h's own
// header note); commutativity holds at every ell.
//
// Placement ([X Sec.1.1]): HeckeExt/, MatrixT<uint32_t> for the stored
// operator matrices (the plan's "MatrixT<uint32_t> for everything in
// HeckeExt/" decision -- docs/matrix-engine/tools/check_matrix_includes.sh
// carries this file with that reason). Consumes ONLY MatrixT and
// engine::ModCtx -- explicitly NOT HeckeCrossChecks ([X Sec.3 WP-2 risk]).

#include <cstdint>
#include <string>
#include <vector>

#include "../../ModArithmetic/Matrix.h"

namespace hecke_ext {

// --- identity class 6 data (slot 6, WP-9) ---------------------------------
//
// One Galois orbit of anemic Hecke systems with its multiplicity.  `orbit`
// holds the r Frobenius conjugates (any order; verify() sorts), each a vector
// index-aligned with IdentityCertificate::residue_primes whose entries are
// width-r coefficient vectors of the canonical split field
// F_ell[Y]/(C_{ell,r}).  Plain vectors, so this header stays free of
// HeckeCanonicalField (the certificate re-derives, it does not re-call).
// theta^j is coordinate-wise multiplication by the SCALAR q^j mod ell, which
// on that coefficient representation is entrywise -- no field arithmetic.
struct ResidueCharacterClass {
    uint32_t r = 0;
    std::vector<std::vector<std::vector<uint32_t>>> orbit;
    uint32_t multiplicity = 0;   // per conjugate: dim_{F_ell}(block) / r
};
using ResidueCharacterMultiset = std::vector<ResidueCharacterClass>;

// The two readings of `Sym^n` in [E] `cl:glover`.  kStandard is the one the
// pipeline's rho_n is in and the one the identity is asserted under.
enum class GloverConvention : uint32_t {
    kStandard = 0,
    kContragredient = 1,   // every j_i shifted by -n: the negative control
};

enum class IdentityStatus : uint32_t {
    kNotApplicable = 0,
    kPassed = 1,
    kFailed = 2,
};

struct IdentityItem {
    IdentityStatus status = IdentityStatus::kNotApplicable;
    std::string detail;
};

struct IdentityCertificate {
    // Applicability inputs.
    uint32_t n = 0;
    uint32_t ell = 0;

    // The record's own already-computed dimensions (level (a):
    // EquivariantCohomologyOnK::get_H_eq_dimensions() / HeckeReport).
    uint32_t dim_H0 = 0;
    uint32_t dim_H1 = 0;
    uint32_t dim_H2 = 0;

    // Two distinct Hecke primes (p != p_prime, both != ell -- this
    // certificate is about the GOOD-prime algebra, not U_ell) and their
    // computed degree-1 cohomology matrices A^{(1)}(T_p) / A^{(1)}(T_p'),
    // plus the FULL T_{p^2} -- supply
    // `HeckeOperatorFamily::T_p_pow_direct(p, 2)` here (the INDEPENDENT
    // double-coset construction), NOT `T_p_pow(p, 2)` (the recursion's own
    // output): checking the recursion against its own output would be
    // tautological. All are d_1 x d_1 (d_1 = dim_H1); a 0x0 matrix marks
    // "not supplied" and makes the corresponding item(s) `kNotApplicable`.
    uint64_t p = 0;
    uint64_t p_prime = 0;
    MatrixT<uint32_t> A_p;
    MatrixT<uint32_t> A_p_prime;
    MatrixT<uint32_t> A_p2;

    // --- identity class 5 (slot 5, WP-8) ---------------------------------
    //
    // The SOURCE cell of `L.`: m = n - ell - 1, its degree-1 operator
    // A^{(1)}_m(T_p) for the SAME prime p, and the induced map
    // L_* : H^1(Ga; M_m) -> H^1(Ga; M_n), a dim_H1 x dim H^1(M_m) matrix.
    // All three are optional: a 0x0 L_star (the default) makes the item
    // `kNotApplicable`, which is what every pre-WP-8 caller gets.
    uint32_t n_source = 0;
    MatrixT<uint32_t> A_p_source;
    MatrixT<uint32_t> L_star;

    // --- identity class 7 (slot 6, WP-3): socle <-> Loewy under PD --------
    //
    // [E Sec.5.1] item 5 = E1 identity class 7 = [C, cl:pd](3): for V =
    // H^1_c(Ga; M_n) = X_3 and W = H^1(Ga; D_n), paired by the Gram matrix of
    // Hecke/HeckePairing (dim V x dim W, <x_i, y_j> = G[i][j]),
    //
    //     rad^i(W) = (soc^i V)^perp        for every i >= 0,
    //
    // the socle series of V taken over the algebra generated by the V-side
    // operators and the radical (Loewy) series of W over the algebra
    // generated by the W-side operators, EACH COMPUTED INDEPENDENTLY from its
    // own matrices (the Jacobson radical of a commutative F_ell-algebra A is
    // the kernel of the F_ell-LINEAR Frobenius x -> x^(ell^e), ell^e >= dim:
    // (x + y)^ell = x^ell + y^ell and c^ell = c on F_ell).  Nothing here
    // assumes the adjointness: it is what makes the two series match, so a
    // failure of [C, cl:pd](2) shows up here too.  The pairing is required
    // to be perfect ([C, cl:pd](1)); a non-perfect Gram FAILS the item.
    //   pd_ops_V[k], pd_ops_W[k]: A_c^{(1)}(T_{p_k}) on V and A(D_n)^{(1)}(T_{p_k})
    //   on W, the same primes in the same order, column convention of
    //   [C, alg:assembly].  Empty (the default) makes the item kNotApplicable,
    //   which is what every pre-WP-3 caller gets.
    // Applicability: ell >= 5, n even, at least one operator on each side,
    // shapes (dim V)^2, (dim W)^2 and dim V x dim W.  Outputs (overwritten by
    // verify()): the socle dimensions dim soc^i V and the radical dimensions
    // dim rad^i W, i = 0 .. Loewy length.
    std::vector<MatrixT<uint32_t>> pd_ops_V;
    std::vector<MatrixT<uint32_t>> pd_ops_W;
    MatrixT<uint32_t> pd_pairing;
    std::vector<uint32_t> pd_socle_dims;
    std::vector<uint32_t> pd_radical_dims;
    // --- identity class 6 (slot 6, WP-9) ---------------------------------
    //
    // `residue_primes`: the anemic primes S (ascending, none == ell) the
    // classes are over.  `residue_cell[q]`: the classes of H^q(M_n),
    // q = 0, 1, 2.  `residue_weights[m][q]`: the classes of H^q(M_m) for
    // every m in [0, ell - 1] (the atlas cells; odd m may be left empty --
    // the recursion never asks for one at even n).  All optional:
    // `residue_cell` empty (the default) makes the item `kNotApplicable`,
    // which is what every pre-WP-9 caller gets.
    std::vector<uint64_t> residue_primes;
    std::vector<ResidueCharacterMultiset> residue_cell;
    std::vector<std::vector<ResidueCharacterMultiset>> residue_weights;
    GloverConvention residue_convention = GloverConvention::kStandard;

    IdentityItem eichler_shimura;
    IdentityItem commutation;
    IdentityItem d2_vanishes;
    IdentityItem p2_recursion;
    IdentityItem dickson_multiplication;
    IdentityItem residue_atlas;           // item code 6 (slot-6 WP-9)
    IdentityItem socle_loewy_pd;          // item code 7 (slot-6 WP-3)

    // Recomputes every item above (overwriting it) and returns 0 iff every
    // APPLICABLE item passed. Nonzero return identifies the first failing
    // clause, in this fixed order: 1 = eichler_shimura, 2 = commutation,
    // 3 = d2_vanishes, 4 = p2_recursion, 5 = dickson_multiplication,
    // 6 = residue_atlas, 7 = socle_loewy_pd. An
    // inapplicable item never contributes a failure code. Applicability:
    // items 1 and 3 need n > 0 even and ell >= 5; item 4 needs ell >= 5 (any
    // n); item 2 needs only non-empty, equal-shape square operator matrices
    // (Hecke commutativity holds at every ell); item 5 needs ell >= 3,
    // n >= ell + 1, n_source == n - ell - 1 and a non-empty L_star whose
    // shape matches (dim_H1, A_p_source.rows) -- the det-twist identity is
    // proved in characteristic ell with no hypothesis on p or on n, so it is
    // asserted wherever the objects exist, ell in {3} included.  Item 6
    // needs ell >= 5, n even, `residue_cell` with three degrees,
    // `residue_weights` with ell entries (three degrees at every even m),
    // and a non-empty `residue_primes` free of multiples of ell.
    uint32_t verify();

    std::string to_string_offset(uint32_t offset) const;
};

// Classical dimension formulas for SL(2,Z), level 1, even weight k:
// dim M_k = floor(k/12) + (k % 12 != 2 ? 1 : 0) for k >= 4 (0 for k < 4,
// the k=0,2 special cases this certificate never uses since n > 0 => k >= 4);
// dim S_k = dim M_k - 1 for k >= 4.
uint32_t DimModularForms(uint32_t k);
uint32_t DimCuspForms(uint32_t k);

// [E] `cl:glover`(4)'s factor multiset of M_n, re-derived by the
// certificate's own recursion: (n_i, j_i, multiplicity), j_i in
// Z/(ell - 1), sorted, equal pairs merged, under `conv`.  Exposed so that the
// validation can name the factors of the cell its negative control uses and
// compare them with HeckeExt/GloverFactors.  ell >= 3 odd.
struct IdentityGloverFactor {
    uint32_t sym = 0, det_twist = 0, multiplicity = 0;
};
std::vector<IdentityGloverFactor> IdentityGloverFactors(uint32_t n, uint32_t ell,
                                                        GloverConvention conv);

// RHS - LHS of identity 6 as the certificate evaluates it, under `conv`;
// empty iff the identity holds.  Exposed for the negative control's
// non-vacuity statement.  Requires the item-6 inputs to be well-formed
// (throws std::logic_error otherwise).
struct ResidueDifference {
    ResidueCharacterClass cls;   // `multiplicity` unused
    int64_t delta = 0;           // RHS - LHS
};
std::vector<ResidueDifference> IdentityResidueDifference(const IdentityCertificate& c,
                                                         GloverConvention conv);
// The signed RHS alone (the twisted Glover side), for comparing conventions.
std::vector<ResidueDifference> IdentityResidueRhs(const IdentityCertificate& c,
                                                  GloverConvention conv);

} // namespace hecke_ext

#endif // HECKE_EXT_IDENTITY_CERTIFICATE_H
