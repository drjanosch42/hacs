// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LOCAL_ALGEBRA_H
#define HECKE_EXT_LOCAL_ALGEBRA_H

// ===========================================================================
// hecke-extensions phase E1a (slot 2, WP-4) -- the LOCAL ALGEBRA A_S of a
// certified-local block, its invariants, and generation certificate (a).
//
// WHAT THIS COMPUTES.  [E] = docs/latex/hecke_extensions.tex v1.3.3,
// Sec. `sec:localalg` and Algorithm `alg:stage3`; [X] =
// docs/implementation/hecke_extensions_implementation_plan.md v1.5.1,
// Sec. 2.2 Stage 3 and Stage 4(a); [C] = docs/latex/gl2_contract.tex v1.3.3.
//
// [C] defines the Hecke algebra itself -- `obl:algebra` / `thm:relations`:
// under 6 in R^x the assignment of double cosets to operators is a RING
// homomorphism, so the T_N commute and T_S := F_ell[T_p : p in S] is a
// commutative subalgebra of End(H^q).  [C] then DELEGATES its local pieces to
// [E] (the introduction, "What remains in [E]: the Hecke algebra \Talg and its
// local pieces \Talg_\midl"); `cl:pd`(3) is the one contract clause that names
// \Talg_\midl.  So the objects below are [E]'s, built on [C]'s commutativity.
//
// Fix a block V = H_m of dimension D over F_ell that `hecke::CertifyLocality`
// ([E] `cert:localcert` / Cert. 5.2) has certified LOCAL, and let
// t_p := T_p|_V for p in S.  Then ([E] `alg:stage3`):
//
//   (1) A_S := F_ell[t_p : p in S] inside End(V), as an F_ell-SUBSPACE of
//       F_ell^{D^2}: start from {I}, close under right multiplication by each
//       t_p, keep the RREF coordinate matrix.  Commutativity ([C] obl:algebra)
//       is what makes the span of the monomials the whole algebra.
//   (2) The coefficient field K = F_ell[w_hat] and the generators
//       n_p := t_p - P_p(w_hat) for EVERY p, the witness included
//       ([E] `cl:resfield`).  w_hat and the P_p are NOT re-derived here: they
//       are exactly what CertifyLocality returns, which is why this unit calls
//       it (with force_matrix_path, so w_hat is formed even at k = 1) rather
//       than re-implementing the Teichmueller lift.  Then
//       m_S = (n_p)_p as an ideal of A_S (NO separate pi_w(w) generator --
//       pi_w(w) lies in (n_w) already, [E] `alg:stage3`(2)), its powers
//       m_S^i, the nilpotency index nu, the socle soc A_S = Ann_{A_S}(m_S),
//       the Gorenstein verdict (dim_{F_ell} soc A_S == f) and the Hilbert
//       function h(i) = dim(m_S^i / m_S^{i+1}) / f.
//   (3) The TANGENT TEST OVER K ([E] `alg:stage3`(3)): the K-span of a set of
//       n_p-bar inside m_S/m_S^2 is the F_ell-span of {w_hat^i n_p-bar :
//       0 <= i < f}, and embdim = dim_{F_ell}(m_S/m_S^2)/f.  Over F_ell ONE
//       K-direction would count as f, and with P_p(w) in place of P_p(w_hat)
//       the witness's own direction is LOST ([E] `cl:resfield`'s
//       F[eps]/eps^2, t = alpha + eps counterexample; [XA B2]).  Both traps
//       are pinned by HeckeExtAlgebraValidation's f = 2 arms.
//
// CERTIFICATE (a) -- CyclicityCertificate, [E] `cert:cyclic`, [X] Sec. 1.3.
// If dim_{F_ell} V/m_S V = f then V is cyclic over A_S (Nakayama), hence
// End_{A_S}(V) = A_S|_V; every further T_q commutes with A_S and preserves V,
// so T_m = A_S|_V -- NO FURTHER OPERATOR CAN CHANGE THE MODULE STRUCTURE AT m.
// That is the claim; `verify()` is what a third party runs on the emitted data
// alone.  Per [X] Sec. 1.3 R17 it RE-DERIVES every quantity it concludes from
// and reads none of them:
//   * the algebra is recomputed from the stored t_p and compared ENTRYWISE
//     with the stored basis (the RREF is canonical, so this is a bit test);
//   * each n_p is recomputed as t_p - P_p(w_hat) and re-tested for nilpotency;
//   * m_S is recomputed as the ideal (n_p)_p and compared entrywise;
//   * dim m_S V is recomputed as a rank, so dim V/m_S V is never read;
//   * the generator index is recomputed as the LEAST standard basis vector
//     outside m_S V, and A_S v = V is recomputed as a rank.
// A truncated basis, a decremented dimension or a perturbed matrix therefore
// fails, which is what HeckeExtAlgebraValidation's tamper arm asserts.
//
// NOT IN THIS UNIT.  Certificate (b) (the commutant, the socle bound, the
// CommutantState resume) is WP-5 / [X] Sec. 2.2 Stage 4(b); certificate (c)
// (the Sturm sweep) is E1b and rests on HeckeExt/HeckeSturmBound; Stage 5's
// module structure (Loewy / socle series of V, Jordan types, structure maps)
// is E2.  There is NO resumable state here: [X] Sec. 1.3.1 R3.3 lists five
// instances and the algebra construction is not one of them -- it is a fixed,
// unbudgeted O(k dim A_S D^3) closure.
//
// PLACEMENT.  HeckeExt/, MatrixT<uint32_t> throughout ([X] Sec. 1.1, "D <=
// ~200"); it consumes Hecke/SubspaceOps, Hecke/FpPoly and
// Hecke/HeckeModuleDecomposition and nothing else of the pipeline.  The
// HeckeExt/ include guard (docs/matrix-engine/tools/check_heckeext_includes.sh)
// forbids Console/, Cuda/, Engine/ and BlockFlatMatrix.h here, and forbids any
// include of this header from outside HeckeExt/ / Campaign/ / Golden/ / Bench/
// / main.cpp / *Validation.*.
// ===========================================================================

#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/SubspaceOps.h"

#include "../../ModArithmetic/Matrix.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace hecke_ext {

using ExtMatrix = MatrixT<uint32_t>;

// ---------------------------------------------------------------------------
// Certificate (a) -- [E] `cert:cyclic`, [X] Sec. 1.3 `CyclicityCertificate`.
// ---------------------------------------------------------------------------

// Everything a third party needs, and nothing that verify() takes on trust.
struct CyclicityCertificate {
    uint32_t ell = 0;
    uint32_t D   = 0;   // dim_{F_ell} V
    uint32_t f   = 0;   // the residue degree, [F_{ell^f} : F_ell]

    // The generators, ascending in p (the fixed operator order of
    // [X] Sec. 2.2 Stage 2), and the coefficient-field data they are reduced
    // against.  `character[j]` is P_{primes[j]}, of degree < f.
    std::vector<uint64_t>   primes;
    std::vector<ExtMatrix>  t;            // t_p|_V, D x D, index-aligned
    std::vector<hecke::FpPoly> character; // P_p, deg < f, index-aligned
    hecke::FpPoly           witness_poly; // pi_w, irreducible of degree f
    ExtMatrix               w_hat;        // the Teichmueller lift, D x D

    // The stored bases, each in the CANONICAL RREF order of
    // hecke::RrefRowSpan (so an entrywise comparison is meaningful).
    std::vector<ExtMatrix> a_basis;  // an F_ell-basis of A_S
    std::vector<ExtMatrix> m_basis;  // an F_ell-basis of the ideal m_S

    // The claim.
    bool     cyclic        = false;  // dim V/m_S V == f
    uint32_t dim_quotient  = 0;      // dim_{F_ell} V / m_S V
    uint32_t generator_index = 0;    // least j with e_j not in m_S V (iff cyclic)

    // 0 on success, else the index of the first failing clause.  Deterministic;
    // it re-derives every quantity it concludes from ([X] Sec. 1.3 R17).
    uint32_t verify() const;
    static const char* verify_message(uint32_t code);

    // Fixed key order, no pointers, no hash container.
    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// The Stage-3 object -- [X] Sec. 1.2 `LocalAlgebra`.
// ---------------------------------------------------------------------------

struct LocalAlgebra {
    // --- status ----------------------------------------------------------
    // `ok` is false exactly when the block did not certify local; then
    // `locality.failure` says why and every field below is empty.  A
    // non-local block is REPORTED, never thrown -- it is E0b's refinement
    // loop that acts on it ([X] Sec. 3.1).
    bool ok = false;
    hecke::LocalityVerdict locality;

    // --- the block -------------------------------------------------------
    uint32_t ell = 0;
    uint32_t D   = 0;     // dim_{F_ell} V
    uint32_t f   = 0;     // residue degree
    uint32_t N   = 0;     // the Teichmueller exponent, f | N and ell^N >= D

    std::vector<uint64_t>  primes;   // S, ascending
    std::vector<ExtMatrix> t;        // t_p|_V, index-aligned with `primes`
    std::vector<hecke::FpPoly> character;  // P_p, deg < f, index-aligned
    std::vector<ExtMatrix> n;        // n_p = t_p - P_p(w_hat), index-aligned
    hecke::FpPoly          witness_poly;   // pi_w
    ExtMatrix              w_hat;          // pi_w(w_hat) == 0 exactly

    // --- A_S -------------------------------------------------------------
    uint32_t               dim_A = 0;
    ExtMatrix              coords;    // dim_A x D^2, RREF, row-major vec
    std::vector<ExtMatrix> basis;     // the same rows as D x D matrices

    // --- the maximal ideal and its filtration ----------------------------
    // m_powers[i] is a basis of m_S^{i+1} (so m_powers[0] is m_S itself) and
    // m_powers_dim[i] its F_ell-dimension; only the NONZERO powers are stored,
    // so nu = m_powers.size() + 1 is the nilpotency index (m_S^nu = 0 !=
    // m_S^{nu-1}) and nu == 1 exactly when A_S is the field F_{ell^f}.
    uint32_t                            dim_m_S = 0;
    std::vector<std::vector<ExtMatrix>> m_powers;
    std::vector<uint32_t>               m_powers_dim;
    uint32_t                            nu = 0;

    // hilbert[i] = dim(m_S^i / m_S^{i+1}) / f for 0 <= i < nu (m_S^0 = A_S).
    std::vector<uint32_t> hilbert;

    std::vector<ExtMatrix> socle;      // a basis of soc A_S = Ann_{A_S}(m_S)
    uint32_t               dim_socle = 0;
    bool                   gorenstein = false;   // dim_socle == f

    // --- the tangent table, over K ---------------------------------------
    uint32_t              embdim = 0;   // dim_{F_ell}(m_S/m_S^2) / f
    std::vector<uint8_t>  active;       // per prime index: n_p-bar != 0
    std::vector<uint64_t> minimal_active;  // greedy in ascending p, K-spanning

    // --- certificate (a) --------------------------------------------------
    CyclicityCertificate cyclicity;

    // dim_{F_ell}(V / m_S V) / f -- the `t` of the socle bound of
    // [X] Sec. 2.2 Stage 4(b), which WP-5 consumes.
    uint32_t min_generators = 0;

    std::string to_json() const;
};

struct LocalAlgebraOptions {
    // Forwarded to hecke::LocalityOptions::max_attempts (0 = the shipped
    // witness budget).  `force_matrix_path` is NOT exposed: this unit always
    // forces it, because it needs the matrix w_hat that the k = 1 analytic
    // shortcut deliberately never forms.
    uint32_t max_attempts = 0;

    // (hecke-extensions slot 4, WP-2 / phase E2b) THE STAGE-3 FAST PATH of
    // [X] Sec. 2.2 Stage 3 ("w_hat and the P_p CONSUMED from the locality
    // verdict, never re-derived differently") and of the slot-4 plan
    // Sec. 2.3 item 2.
    //
    // Since E0b the decomposition's own refinement loop enforces exactly
    // [E] Cert. 5.2 and stores its result on the block (`local_certified`,
    // `witness_poly`, `w_hat_exponent`, `character`, `residue_witness`), so
    // re-running `hecke::CertifyLocality` here searches again for a witness
    // that is already proved.  When this pointer is non-null, Step 0 ADOPTS
    // the verdict instead of running the search; everything after Step 0 is
    // unchanged, including the three ENGINE_CHECKs that re-derive
    // pi_w(w_hat) = 0, the Teichmueller exponent and the nilpotency of every
    // n_p -- so an adopted verdict that is not a certificate still fails
    // loudly.  The verdict must outlive the call.
    //
    // A null pointer (the default) is the pre-E2b behaviour, which
    // HeckeExtRamanujanValidation keeps as the fast path's ORACLE: the two
    // routes must agree entrywise on every field of the LocalAlgebra.
    const hecke::LocalityVerdict* precomputed_locality = nullptr;
};

// Stage 3 + Stage 4(a) on ONE block of an already-computed decomposition.
//
// `ops` is the family the decomposition was built from, over F_ell, in any
// order (it is sorted ascending by prime here, the fixed order of
// [X] Sec. 2.2 Stage 2); `block` is one of its `PrimaryBlock`s.  Throws
// std::invalid_argument on a malformed argument (wrong modulus, non-square
// operator, empty family) and reports every mathematical outcome -- including
// "not local" -- through the returned object.
LocalAlgebra ComputeLocalAlgebra(const hecke::PrimaryBlock& block,
                                 const std::vector<std::pair<uint64_t, ExtMatrix>>& ops,
                                 uint32_t ell,
                                 const LocalAlgebraOptions& opts = LocalAlgebraOptions());

// ---------------------------------------------------------------------------
// Cross-block identity of maximal ideals -- [X] Sec. 2.2 Stage 2, [XA M5],
// [X] Sec. 1.3 `ExhaustivenessCertificate`.
// ---------------------------------------------------------------------------
//
// Blocks B and B' carry the SAME maximal ideal iff, with B's OWN witness
// coefficients c, its pi_w and its characters {P_p}:
//
//     w'      := sum_p c_p t_p|_{B'}
//     N'      := the least multiple of f with ell^{N'} >= dim B'
//     w_hat'  := (w')^{ell^{N'}}
//
// pi_w(w') is nilpotent AND every t_p|_{B'} - P_p(w_hat') is nilpotent.
//
// The exponent is taken for the TESTED block ([XA M5]'s correction: nilpotency
// must be bounded on B', not on B).  The test is deterministic, needs no shared
// candidate sequence -- v1.1's "the first witness in both" was ill-defined,
// the candidate sequence being seeded by the block dimension
// (HeckeModuleDecomposition.cpp:695) -- and needs no "up to Galois" clause: it
// passes exactly when B''s character is B's or a Frobenius conjugate of it,
// i.e. the same maximal ideal of T_S (a character phi' with phi'(w) a root of
// pi_w and phi'(t_p) = P_p(phi'(w)) has kernel m_B).
struct EvaluationMatch {
    bool     same_ideal = false;
    uint32_t failure    = 0;   // 0 iff same_ideal; see EvaluationFailure
    uint32_t N_other    = 0;   // the N' actually used
    uint64_t failing_prime = 0;  // meaningful iff failure == kEvalCharacterNotNilpotent
};

enum EvaluationFailure : uint32_t {
    kEvalOk = 0,
    kEvalWitnessNotNilpotent = 1,    // pi_w(w') is not nilpotent on B'
    kEvalCharacterNotNilpotent = 2,  // some t_p|_{B'} - P_p(w_hat') is not
};

// `t_other` are B''s own restricted operators, index-aligned with `B.primes`
// (same primes, same order).  Throws std::invalid_argument otherwise.
EvaluationMatch SameMaximalIdeal(const LocalAlgebra& B,
                                 const std::vector<ExtMatrix>& t_other);

} // namespace hecke_ext

#endif // HECKE_EXT_LOCAL_ALGEBRA_H
