// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_BOUNDARY_DATUM_H
#define HECKE_BOUNDARY_DATUM_H

// hecke-extensions slot 1, WP-4a (phase E4a) -- the certified Gamma-level
// boundary datum tau_bd of docs/latex/gl2_contract.tex ([C], cited by label),
// chapter sec:boundarydatum:
//
//   * [C, def:bdrygroup]     Gamma_inf = Stab_Gamma(infinity) = <T> x <-I>.
//   * [C, prop:bdrycosets]   B_N = { delta in Gamma g Gamma : delta upper
//     (1)-(3)                triangular }, its m = N + p^{nu-1} LEFT cosets
//                            delta Gamma_inf (canonical representatives
//                            [[p^u, b],[0, p^{nu-u}]], 0 <= b < p^u,
//                            primitive; at nu = 1: delta_inf = diag(1, p) and
//                            delta_t = [[p, t],[0, 1]]), and the bijection
//                            B_N / Gamma_inf <-> Gamma g Gamma / Gamma,
//                            delta_j Gamma_inf . gamma_j <-> delta_j Gamma with
//                            gamma_j . infinity = delta_j^{-1} . infinity
//                            (boundary_cosets(), boundary_coset_of_bulk(j),
//                            gamma_j(j); the bijection is certified at
//                            construction). The RIGHT cosets beta_b of
//                            [C, prop:bdrycosets](4) are deliberately NOT
//                            built here (they are WP-4b's negative control).
//   * [C, def:taubdry]       (T1bd)-(T3bd): tau_bd : C_*(X_bd) -> C_*(X_bd)
//                            additive, (T1bd) tau_bd(h.x) = (g^{-1} h g).tau_bd(x)
//                            for h in Gamma'' = Gamma_0(N), (T2bd) boundary
//                            tau_bd = tau_bd boundary, (T3bd) the induced map
//                            on pi_0(X_bd) = P^1(Q) is a map of sets, each
//                            component to a single component with total
//                            coefficient 1.
//   * [C, lem:bdrycomponent] the component map is FORCED: c -> g^{-1} c. It is
//                            not chosen anywhere below; certification (i)
//                            recomputes it from g alone (CuspKey::apply with
//                            adj(g) = N g^{-1}, projective, no rationals).
//   * [C, lem:taubdryexists] the construction (1): for each Gamma''-orbit
//                            representative vertex slot, ANY vertex of the
//                            forced target component; for each edge slot, the
//                            UNIQUE 1-chain of that component (a line, (B2))
//                            with the right boundary -- an oriented path;
//                            extension by (T1bd), well defined because every
//                            boundary stabiliser is {+-I} ((B3)); nothing in
//                            degrees >= 2. (2): coefficients in {0, +-1}, no
//                            linear solve anywhere, so the table IS a datum
//                            over Z for EVERY ell, and its solve regime is Z
//                            unconditionally ([C, rmk:taubdryintegral], owner
//                            decision OP2: no mod-ell fill can arise here).
//   * [C, obl:taubdry](i)-(iv), certified in the constructor -- see certify().
//
// ===========================================================================
// INVARIANT I8 (certification over geometry), inherited from HeckeTauTable:
// the geometric choice below (the nearest line vertex) only PROPOSES the
// table; certify(), called by the constructor, DISPOSES. A table failing any
// clause is REJECTED (the constructor throws) and can never exist in an
// uncertified state. The certification is never downgraded and never skipped.
// ===========================================================================
//
// The choice made in [C, lem:taubdryexists](1) ("choose any vertex w of the
// target component"). This implementation takes the geometric analogue of
// [C, alg:tau] step 1: z := (g^{-1} s_j) . pos(v) computed integrally through
// adj(g) = diag(N, 1) (projective, no rationals), lies on a horocycle around
// the forced cusp c' = g^{-1} s_j infinity; transport c' to infinity by the
// deterministic gamma_red of CuspKey::reduce_to_infinity, and take on the
// standard line X_bd,inf the vertex NEAREST to Re(gamma_red z) -- the
// vertical projection onto the horocycle at height Y -- ties going to the
// larger real part; w := gamma_red^{-1} . (that vertex). The rule is
// T-equivariant, so it does not depend on which gamma_red is used, and
// -I acts trivially; the result is one term with coefficient 1, so (T3bd)
// holds by inspection. Any other admissible choice is a homotopic datum
// ([C, lem:taubdryexists](3), over Z for every ell); this one keeps tau_bd
// close to z -> g^{-1} z, which is what makes the pair homotopy eta of
// [C, sec:cone] (phase E5) small. iota-symmetry of tau_bd is NOT asserted and
// not claimed: a vertex choice on the line cannot commute with iota at a tie
// on the mirror Re z = 0 (the tied pair {v1, v2} is a full iota-orbit with no
// fixed vertex on the line) -- recorded in docs/implementation/slot1/WP-4a_report.md.
//
// Slots and the single (T1bd) helper ([C, alg:tau] step 0, applied to X_bd).
// The table stores one chain per pair (j, sigma), j in [1, m] (S3's 1-based
// coset indices), sigma a boundary simplex of K (K_bd = the path v0-v1-v2-v3
// and its three edges, SL2ZTessellationTables::boundary_simplices). The
// pairs are NOT pairwise Gamma''-inequivalent (i(v3) = T i(v0)); exactly one
// slot per Gamma''-orbit is CONSTRUCTED and every other slot is DERIVED
// through the one shared helper tau_on_translate() -- which is also what
// certification uses for the right-hand sides of (ii) and (iii), and which
// itself routes its (P1) split and conjugation through
// HeckeCosetTables::twisted_conjugate, the function the bulk tau uses.
//
// Degree-generic ([C, sec:boundarydatum] code map, [C, open:ell2] rider (i)):
// the record store is indexed by every degree the boundary subcomplex
// reports (num_degrees() = boundary_dim() + 1) and nothing below is written
// for q in {0, 1, 2}; (B2) forces dim X_bd = 1 on every admissible model and
// the constructor refuses a boundary with cells above that dimension rather
// than silently truncating.
//
// What is NOT here (WP-4b, phase E4b): Theta^bd_g and T^bd_p, the Shapiro
// transcription tau_inf (route A), route B, the chi_b oracles, the beta_j
// control, the restriction equivariance, and the record fields
// boundary_model / boundary_is_group_cohomology of the circle model. The
// table is ell-independent and is built for every ell, including ell in
// {2, 3} ([C, lem:taubdryexists] carries no hypothesis on ell).
//
// Placement: like HeckeCosetTables / HeckeTauTable this class consumes the
// SimplicialComplex model and the templated HeckeCosetTablesT<Elt> (which
// carries FlatMatrix rho_n blocks), so it is compiled into cohomology_app,
// not into the pure-integer HeckeCore static library; it links HeckeCore for
// Mat22 / QuadPoint. It makes no ModArithmetic call of its own.

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ExactInt.h"
#include "HeckeCosetTables.h"
#include "HeckeTauTable.h" // hecke::ChainTerm -- the stored-chain format of [C, alg:tau]
#include "SL2ZTessellationTables.h"

class SimplicialComplex;

namespace hecke {

// The vertex choice of [C, lem:taubdryexists](1) ("choose ANY vertex w of the
// forced target component"), exposed as a POLICY (hecke-extensions slot 3,
// WP-6; slot-3 plan Sec. 2.3 item 5 with its [audit] note).  Correctness never
// depends on it: (1) admits any vertex of the target component and (3) makes
// any two such tables homotopic over Z, so the boundary operator Theta^bd and
// the cone operator A_c^{(q)} are the same ([C, prop:coneindep](2)).  It only
// selects WHICH admissible tau_bd is stored, which is what
// [C, obl:coneexact](viii)(a) ("two tau_bd-tables") needs.
enum class TauBoundaryVertexPolicy {
    // Binding default: the canonical [C, alg:taubdry] rule -- the line vertex
    // NEAREST to Re(gamma_red z), ties to the larger real part.  Every shipped
    // object (HeckeBoundaryTauValidation, tau_table_p2.txt, the golden grid)
    // is built with this and is byte-unchanged by the existence of the enum.
    NearestLineVertex,
    // The second admissible rule: the line vertex at position P + 1, P the
    // nearest one.  Still a vertex of the forced target component, so
    // [C, lem:taubdryexists](1) admits it; the rule is T-equivariant in the
    // same sense as the first (P is, and +1 commutes with the period shift),
    // so it does not depend on which gamma_red is used.  It is a NON-TIE rule
    // by construction, so it moves the table on EVERY constructed slot -- which
    // is what the slot-3 audit requires of the second policy, since a policy
    // that left the table byte-identical would make
    // [C, obl:coneexact](viii)(a)'s independence arm vacuous.  (num_vertex_ties()
    // below measures the alternative: a MIRROR tie rule would also be
    // non-vacuous here -- 189 constructed vertex slots over
    // HeckeConeActionValidation's grid are ties -- but it would move the table
    // only on those slots, and only the non-tie rule is uniformly non-vacuous.
    // docs/implementation/slot3/WP-6_report.md.)
    NextLineVertex,
};

const char* to_string(TauBoundaryVertexPolicy p);

// One left coset delta Gamma_inf of B_N ([C, prop:bdrycosets](1)) by its
// canonical representative delta = [[p^u, b],[0, p^{nu-u}]], 0 <= b < p^u,
// gcd(p^u, b, p^{nu-u}) = 1 (i.e. p does not divide b when 0 < u < nu).
struct BoundaryCoset {
    Mat22 delta;
    uint32_t u;
    int64_t b;
};

template<class Elt>
class HeckeBoundaryTauTableT {
public:
    // K must be the 7-vertex model; model / cosets must outlive this object
    // (cosets carries g = diag(1, N), the s_j and the (P1) class map). Builds
    // the B_N / Gamma_inf table, the tau_bd table by [C, lem:taubdryexists](1),
    // and runs the MANDATORY certification: the constructor calls certify()
    // and THROWS (std::runtime_error / HeckeOverflow) on any failure.
    HeckeBoundaryTauTableT(const SimplicialComplex& K, const SL2ZTessellationTables& model,
                           const HeckeCosetTablesT<Elt>& cosets,
                           TauBoundaryVertexPolicy policy = TauBoundaryVertexPolicy::NearestLineVertex);

    // --- shape ---------------------------------------------------------------
    uint32_t num_degrees() const { return static_cast<uint32_t>(records_.size()); } // boundary_dim + 1
    uint32_t num_cosets() const { return m_; }
    uint64_t N() const { return N_; }
    const Mat22& g() const { return cosets_->g(); }
    const HeckeCosetTablesT<Elt>& cosets() const { return *cosets_; }
    const SL2ZTessellationTables& model() const { return *model_; }
    // (slot 3, WP-6) The vertex policy this table was built with, and the
    // FNV-1a-64 (ModArithmetic/Fnv.h) over the whole stored table in slot
    // order -- (q, sigma, j, then each term's coeff, gamma, sigma_idx), the
    // same walk HeckePairHomotopyT::tau_boundary_table_fnv64() uses, so the
    // two agree on the same table.  [C, obl:eta]'s preamble names the table
    // by this hash; [C, obl:coneexact](viii)(a)'s independence arm asserts it
    // DIFFERS between the two policies.
    TauBoundaryVertexPolicy policy() const { return policy_; }
    uint64_t table_fnv64() const;
    // (slot 3, WP-6) Constructed vertex slots at which the canonical rule's
    // nearest-vertex choice was a TIE, i.e. the line vertex at position P - 1
    // is exactly as far from Re(gamma_red z) as the chosen one at P.  This is
    // the measurement that decides whether a second policy phrased as a
    // TIE RULE ("ties to the smaller real part") moves the table at all.
    // Measured over HeckeConeActionValidation's grid: 189 constructed vertex
    // slots are ties, so such a rule WOULD be non-vacuous -- but only on those
    // slots; the shipped second policy is the uniformly non-vacuous non-tie
    // NextLineVertex (slot-3 audit, "the real risk is vacuity").  Recorded,
    // gates nothing.
    uint32_t num_vertex_ties() const { return num_vertex_ties_; }
    // Indices (into K's q-simplex list) of the boundary q-simplices.
    const std::vector<uint32_t>& boundary_simplices(uint32_t q) const {
        return model_->boundary_simplices(q);
    }
    // (WP-4b, additive) K's simplex lists as copied at construction, so the
    // boundary operator can verify that its circle model K_bd lists exactly
    // these boundary simplices, in this order, by vertex-list comparison.
    const std::vector<std::vector<std::vector<uint32_t>>>& K_simplices() const { return simplices_; }

    // --- the stored table ----------------------------------------------------
    // tau_bd(s_j . i(sigma)) for j in [1, num_cosets()], q < num_degrees(),
    // sigma a boundary q-simplex of K (K's index). Throws std::out_of_range
    // on a bad index and std::invalid_argument on a non-boundary simplex. An
    // edge chain may be legitimately empty (both endpoints sent to the same
    // vertex). Every term is canonical (SL2ZTessellationTables::canonical_key),
    // has coefficient +-1, det gamma = 1 and a boundary sigma_k.
    const std::vector<ChainTerm>& tau(uint32_t j, uint32_t q, uint32_t sigma) const;
    // True iff the slot was filled through the (T1bd) helper rather than
    // constructed (reporting only; certification treats both alike).
    bool slot_derived(uint32_t j, uint32_t q, uint32_t sigma) const;

    // --- THE single (T1bd) helper --------------------------------------------
    // tau_bd(gamma . i(sigma)) for ARBITRARY gamma in Gamma, from the stored
    // slot of its Gamma''-class: gamma = h s_j via HeckeCosetTables::
    // twisted_conjugate ((P1), h in Gamma_0(N)), the chain is
    // (g^{-1} h g) . tau_bd(s_j . i(sigma)), canonicalised. Used by the
    // construction (slot derivation), by certify() (the right-hand sides of
    // (ii) and (iii)) and, later, by the boundary operator of WP-4b and the
    // eta-table of E5. Public so that validations can cross-check it against
    // an independent re-derivation.
    std::vector<ChainTerm> tau_on_translate(const Mat22& gamma, uint32_t q, uint32_t sigma) const;

    // --- components ([C, lem:bdrycomponent]) ---------------------------------
    // The cusp s_j . infinity, i.e. the component of every slot (j, sigma).
    CuspKey slot_component(uint32_t j) const;
    // The forced component map c -> g^{-1} c, recomputed from g alone
    // (adj(g) = N g^{-1} acting projectively).
    static CuspKey component_map(const Mat22& g, const CuspKey& c);
    // The component of a chain: the common cusp gamma_k . infinity of its
    // terms. single_out = false (and infinity returned) if the terms span
    // more than one component or the chain is empty.
    static CuspKey chain_component(const std::vector<ChainTerm>& terms, bool& single_out);

    // --- B_N / Gamma_inf ([C, prop:bdrycosets](1)-(3)) -----------------------
    // The m left cosets in the order (u = 0), (u = 1, b ascending), ...; at
    // nu = 1 that is delta_inf = diag(1, p) first, then delta_t, t = 0..p-1.
    const std::vector<BoundaryCoset>& boundary_cosets() const { return bcosets_; }
    // Index into boundary_cosets() of the coset delta_j gamma_j Gamma_inf
    // (the bijection of [C, prop:bdrycosets](2); certified bijective).
    uint32_t boundary_coset_of_bulk(uint32_t j) const;
    // gamma_j in Gamma with delta_j gamma_j in B_N, i.e.
    // gamma_j . infinity = delta_j^{-1} . infinity = g^{-1} s_j . infinity.
    const Mat22& gamma_j(uint32_t j) const;
    // The canonical left-coset representative of an upper-triangular
    // delta in B_N (sign to p^u > 0, then b mod p^u), and its index; throws
    // if delta is not in B_N.
    uint32_t boundary_coset_index_of(const Mat22& delta_upper) const;

    // --- record fields ([C, obl:taubdry](iv), [C, rmk:taubdryintegral]) ------
    // The solve regime of a tau_bd-table is Z for EVERY ell, by construction:
    // there is no fill to fall back from. Constant, and certification asserts
    // the coefficient statement behind it.
    const char* solve_regime() const { return "Z"; }
    bool integral() const { return true; }
    // B_tab = max ||gamma_k|| over the stored table and B = max(B_tab, 4 N^2)
    // ([C, lem:bounds](5), [C, obl:backend]); certified by a full recount.
    uint64_t certified_bound_B_tab() const { return bound_B_tab_; }
    uint64_t certified_bound_B() const { return bound_B_; }
    uint32_t num_constructed_slots() const { return num_constructed_; }
    uint32_t num_derived_slots() const { return num_derived_; }
    uint32_t num_slots() const { return num_constructed_ + num_derived_; }

    // --- the MANDATORY certification ([C, obl:taubdry](i)-(iv)) --------------
    // 0 iff ALL of the following hold EXACTLY over Z:
    //   (a) 11-16: every slot stored; every term has coeff != 0, det gamma = 1,
    //       is in canonical form, and names a boundary simplex of degree q;
    //   (iv) 17: every coefficient is +-1 (with 13: never 0);
    //   (i) 20: every degree-0 chain is one simplex with coefficient 1 (T3bd);
    //       21: every chain is supported in a single component;
    //       22: that component is g^{-1} (s_j . infinity), recomputed from g
    //       alone ([C, lem:bdrycomponent]);
    //   (ii) 30: boundary(tau_bd(s_j sigma)) == tau_bd(boundary(s_j sigma)) for
    //       every stored slot of degree >= 1, the right side evaluated ONLY
    //       through tau_on_translate;
    //   (iii) 40: for EVERY stored transporter t in S(sigma, sigma') between
    //       boundary simplices and every j, the slot (j, sigma) equals the
    //       sign-tracked (T1bd)-translate of the slot reached through s_j t
    //       (this includes the {+-I} self-constraints);
    //   (e) 50/51: B_tab matches a full recount and B = max(B_tab, 4 N^2)
    //       satisfies 2 B^2 < 2^63 (the int64 tier, [C, obl:backend]);
    //   (f) 60-66: the B_N / Gamma_inf table has m canonical, primitive,
    //       pairwise distinct upper-triangular representatives of det N, and
    //       j -> delta_j gamma_j Gamma_inf is a bijection onto it with
    //       gamma_j . infinity = g^{-1} s_j . infinity.
    // Called by the constructor, which throws on a nonzero result.
    uint32_t certify() const;
    // The same certification with ONE slot replaced by `replacement` (the
    // object itself is not modified). Exposed so that a validation can show
    // that a tampered chain is rejected; 0 iff the tampered table certifies.
    uint32_t certify_with_override(uint32_t j, uint32_t q, uint32_t sigma,
                                   const std::vector<ChainTerm>& replacement) const;

    std::string to_string() const { return to_string_offset(0); }
    std::string to_string_offset(uint32_t offset) const;

private:
    struct ChainRecord {
        std::vector<ChainTerm> terms;
        bool stored = false;
        bool derived = false;
    };
    // records_[q][local sigma][j-1], local sigma indexing boundary_simplices(q).
    using Records = std::vector<std::vector<std::vector<ChainRecord>>>;
    // Signed canonical accumulation map: {q, sigma_hat, a, b, c, d} -> coeff.
    using CanonMap = std::map<std::array<int64_t, 6>, int64_t>;

    const SimplicialComplex* K_;
    const SL2ZTessellationTables* model_;
    const HeckeCosetTablesT<Elt>* cosets_;
    TauBoundaryVertexPolicy policy_ = TauBoundaryVertexPolicy::NearestLineVertex;
    uint64_t N_;
    uint32_t m_;
    std::vector<std::vector<std::vector<uint32_t>>> simplices_; // K's simplex lists (copied)
    Records records_;

    std::vector<BoundaryCoset> bcosets_;
    std::vector<uint32_t> bcoset_of_bulk_; // size m_+1, index 0 unused
    std::vector<Mat22> gamma_j_;           // size m_+1, index 0 unused

    uint64_t bound_B_tab_ = 0;
    uint64_t bound_B_ = 0;
    uint32_t num_constructed_ = 0;
    uint32_t num_derived_ = 0;
    uint32_t num_vertex_ties_ = 0;

    // --- bookkeeping ---------------------------------------------------------
    uint32_t local_index(uint32_t q, uint32_t sigma) const; // throws on a non-boundary simplex
    void check_indices(uint32_t j, uint32_t q, uint32_t sigma) const;
    const ChainRecord& record(uint32_t j, uint32_t q, uint32_t sigma) const;
    ChainRecord& record_mut(uint32_t j, uint32_t q, uint32_t sigma);
    void accumulate_canonical(CanonMap& acc, const Mat22& gamma, uint32_t q, uint32_t sigma,
                              int64_t coeff) const;
    std::vector<ChainTerm> canon_map_to_terms(const CanonMap& acc, uint32_t q) const;
    CanonMap terms_to_canon_map(const std::vector<ChainTerm>& terms, uint32_t q) const;
    CanonMap boundary_of_terms(const std::vector<ChainTerm>& terms, uint32_t q) const;
    static bool canon_maps_equal(const CanonMap& x, const CanonMap& y);
    // face_f of the q-simplex sigma of K (index into K's (q-1)-simplex list).
    uint32_t face_simplex_index(uint32_t q, uint32_t sigma, uint32_t f) const;

    // The helper over an explicit record set (certify_with_override runs the
    // certification on a modified copy; tau_on_translate uses records_).
    std::vector<ChainTerm> eval_on_translate(const Records& recs, const Mat22& gamma, uint32_t q,
                                             uint32_t sigma) const;

    // --- construction --------------------------------------------------------
    void build_boundary_cosets();
    bool try_derive(uint32_t j, uint32_t q, uint32_t sigma);
    void construct_vertex(uint32_t j, uint32_t v);  // [C, lem:taubdryexists](1), vertices
    void construct_edge(uint32_t j, uint32_t e);    // [C, lem:taubdryexists](1), edges

    uint32_t certify_impl(const Records& recs) const;
};

using HeckeBoundaryTauTable = HeckeBoundaryTauTableT<uint32_t>;
using HeckeBoundaryTauTable64 = HeckeBoundaryTauTableT<uint64_t>;

} // namespace hecke

#endif // HECKE_BOUNDARY_DATUM_H
