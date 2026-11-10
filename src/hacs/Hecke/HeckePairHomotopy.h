// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_PAIR_HOMOTOPY_H
#define HECKE_PAIR_HOMOTOPY_H

// hecke-extensions slot 2, WP-8 (phase E5-1) -- the certified pair homotopy
// eta of docs/latex/gl2_contract.tex ([C], cited by label) chapter sec:cone,
// the ONLY new chain-level table that chapter asks for, and the first third of
// the Hecke datum of the pair (tau, tau_bd, eta) that the cone action
// Phi_g of [C, def:coneaction] consumes (E5 core, slot 3 on).
//
//   * [C, def:pairhomotopy]  eta : C_*(X_bd) (x) R -> C_{*+1}(X) (x) R additive
//                            of degree +1 with (T1) eta(h.x) = (g^{-1} h g).eta(x)
//                            for h in Gamma'' = Gamma_0(N), and
//
//                                boundary eta + eta boundary = j tau_bd - tau j
//                                                        ([C, eq:pairhomotopy])
//
//                            as maps C_*(X_bd) -> C_*(X). Sign convention: owner
//                            decision OP6 (the decision box after the
//                            definition); the derived cochain homotopy of the
//                            cone is h := eta~ with NO sign.
//   * [C, lem:etaexists]     the constructive existence over Z for EVERY ell
//                            (no stabiliser order is inverted: every boundary
//                            simplex has Gamma''-stabiliser {+-I}, (B3)):
//                            (V) vertices -- the 0-chain (j tau_bd - tau j)(x)
//                            has augmentation 1 - 1 = 0 ((T3bd), (T3)), hence
//                            bounds; choose ANY 1-chain eta(x) with that
//                            boundary; (E) edges -- z_e := (j tau_bd - tau j)(e)
//                            - eta(boundary e) is a 1-cycle, hence a boundary,
//                            and the 2-chain eta(e) with boundary z_e is UNIQUE
//                            (Z_1(X) = B_1(X), Z_2(X) = 0, [C, eq:contractibleuse]);
//                            (>= 2) nothing, C_q(X_bd) = 0 by (B2). The only
//                            choice is the vertex chain, and [C, lem:etaunique]
//                            says what it costs: two eta's differ by
//                            boundary L - L boundary for a UNIQUE degree-2 L.
//   * [C, obl:eta](i)-(iv)   certified in the constructor -- see certify().
//                            Clause (v) (the derived h is the transport of the
//                            stored eta) and (vi) beyond the eta_zero record
//                            belong to the ConeActionCertificate ([C, sec:cone]
//                            code map) and are NOT here.
//   * [C, open:ell2]         (owner decision OC-1, rider (i)): on the finite
//                            7-vertex model q_max is DERIVED; eta is built in
//                            every boundary degree q with q + 1 <= dim X
//                            (num_degrees() = min(taub.num_degrees(), dim X),
//                            i.e. q in {0, 1} here) and certified in every
//                            degree in which it is built; truncated = false,
//                            q_max_source = "derived", degrees() = [0 .. q_max].
//
// ===========================================================================
// INVARIANT I8 (certification over geometry), inherited from HeckeTauTable
// and HeckeBoundaryDatum: the vertex-chain policy below only PROPOSES the
// table; certify(), called by the constructor, DISPOSES. A table failing any
// clause is REJECTED (the constructor throws) and can never exist in an
// uncertified state. Never downgraded, never skipped.
// ===========================================================================
//
// The construction, slot by slot ([C, alg:tau] step 0 applied to X_bd, as in
// HeckeBoundaryDatum). The table stores one (q+1)-chain of X per pair
// (j, sigma), j in [1, m] (S3's 1-based coset indices), sigma a boundary
// q-simplex of K (SL2ZTessellationTables::boundary_simplices(q)); the pairs
// are NOT pairwise Gamma''-inequivalent (i(v3) = T i(v0)), so exactly one
// slot per Gamma''-orbit is CONSTRUCTED and every other slot is DERIVED
// through the one shared (T1) helper eta_on_translate() -- which is also what
// certify() uses for the right-hand sides of (ii) and (iii), and which routes
// its (P1) split and conjugation through HeckeCosetTables::twisted_conjugate,
// the function both tau and tau_bd use.
//
//   (V) a vertex slot (j, v): A := tau(s_j . i(v)) and B := tau_bd(s_j . i(v))
//       are single K^u-vertices with coefficient 1 ((T3), (T3bd)); eta(s_j v)
//       is the [C, alg:tau] step-2 edge path from A to B -- the (P2) word of
//       gamma_A^{-1} gamma_B concatenating the certified path templates of
//       SL2ZTessellationTables, whose boundary telescopes to B - A EXACTLY over
//       Z -- with the template family selected by EtaVertexPolicy (the
//       canonical tree-routed family by default; the legacy BFS family as the
//       second admissible policy, so that [C, obl:coneexact](viii)(a)'s
//       "two eta-tables built with different vertex-chain policies" has two
//       tables to compare). Empty when A == B. No linear solve.
//   (E) an edge slot (j, e): z := (j tau_bd - tau j)(s_j e) - eta(boundary(s_j e)),
//       the faces fetched through eta_on_translate; eta(s_j e) := the unique
//       2-chain with boundary z, found by the finite-neighbourhood integer
//       solve of [C, alg:tau] step 3 (HeckeIntegerSolve, the SAME solver the
//       bulk tau uses; neighbourhood = the translates in supp z, grown by one
//       ring of T / T^{-1} / S per retry). Uniqueness makes the neighbourhood
//       and the elimination order irrelevant to the output.
//   Degree-generic: the loop runs over every boundary degree the model
//   reports, the fill solver takes the degree as a parameter, and nothing
//   below is written for q in {0, 1, 2}.
//
// Solve regime (owner decision OP2, [C, obl:eta](iv)): EVERY fill is over Z
// and the recorded regime of every slot is "Z"; there is NO mod-ell (or
// mod-m) fallback in this unit -- a Z-infeasible or overflowing fill is a
// tier failure and ABORTS (throws), it is never solved modulo anything. The
// table is therefore a datum over Z for every ell, integral() == true
// unconditionally, and the tau it is built against must itself be integral
// (tau.used_mod_ell_solve() == false; certified, clause (i)).
//
// ell / n independence: the constructor takes no ell and no n_sym; the table
// depends on (p, nu, the vertex policy) and on the shipped tau / tau_bd
// tables only, which are themselves ell- and n-independent. It is built for
// every ell, ell in {2, 3} included ([C, lem:etaexists] carries no hypothesis
// on ell).
//
// Placement (hecke_extensions_slot2_plan.md Sec. 3 WP-8 scope 1, [X] Sec. 1.1):
// Hecke/, beside HeckeTauTable / HeckeBoundaryDatum, because it consumes both
// (templated on the element width like them) and the ext guard forbids the
// reverse direction. No FlatMatrix of its own, no MatrixT, no raw mod_* call,
// no unordered_ container.

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ExactInt.h"
#include "HeckeBoundaryDatum.h"
#include "HeckeCosetTables.h"
#include "HeckeTauTable.h" // hecke::ChainTerm -- the stored-chain format of [C, alg:tau]
#include "SL2ZTessellationTables.h"

namespace hecke {

// The vertex-chain policy of [C, lem:etaexists](V) ("choose ANY 1-chain").
// Correctness NEVER depends on it (I8; [C, lem:etaunique]; the cone operator
// is independent of eta, [C, prop:coneindep](1)); it only selects which
// admissible eta is stored.
enum class EtaVertexPolicy {
    // Binding default: the tree-routed canonical path templates of
    // SL2ZTessellationTables (PathTemplateFamily::Canonical), the same family
    // the canonical tau of [C, alg:tau] step 2 uses since v1.1.0.
    CanonicalTreePath,
    // The legacy BFS templates (PathTemplateFamily::LegacyBfs): a second
    // admissible eta, differing from the first by boundary L - L boundary
    // ([C, lem:etaunique]); the independence arm's second table.
    LegacyBfsPath,
};

const char* to_string(EtaVertexPolicy p);

template<class Elt>
class HeckePairHomotopyT {
public:
    // tau and taub must be built on the SAME HeckeCosetTables and the same
    // SL2ZTessellationTables (identity, certified) and must outlive this
    // object. Builds the eta-table by [C, lem:etaexists] and runs the
    // MANDATORY certification: the constructor calls certify() and THROWS
    // (std::runtime_error / HeckeOverflow) on any failure.
    HeckePairHomotopyT(const HeckeTauTableT<Elt>& tau, const HeckeBoundaryTauTableT<Elt>& taub,
                       EtaVertexPolicy policy = EtaVertexPolicy::CanonicalTreePath);

    // --- shape ---------------------------------------------------------------
    // The boundary degrees q in which eta is built: q < num_degrees(), where
    // num_degrees() = min(taub.num_degrees(), dim X) (a (q+1)-chain of X must
    // exist). 2 on this model: q in {0, 1}.
    uint32_t num_degrees() const { return static_cast<uint32_t>(records_.size()); }
    uint32_t num_cosets() const { return m_; }
    uint64_t N() const { return N_; }
    const Mat22& g() const { return cosets_->g(); }
    const HeckeCosetTablesT<Elt>& cosets() const { return *cosets_; }
    const SL2ZTessellationTables& model() const { return *model_; }
    const HeckeTauTableT<Elt>& tau() const { return *tau_; }
    const HeckeBoundaryTauTableT<Elt>& tau_boundary() const { return *taub_; }
    EtaVertexPolicy policy() const { return policy_; }
    // Indices (into K's q-simplex list) of the boundary q-simplices.
    const std::vector<uint32_t>& boundary_simplices(uint32_t q) const {
        return model_->boundary_simplices(q);
    }
    // K's simplex lists (as copied by taub at its construction).
    const std::vector<std::vector<std::vector<uint32_t>>>& K_simplices() const { return simplices_; }

    // --- the stored table ----------------------------------------------------
    // eta(s_j . i(sigma)) for j in [1, num_cosets()], q < num_degrees(), sigma
    // a boundary q-simplex of K (K's index): a (q+1)-chain of X in canonical
    // ChainTerm form (sigma_idx indexes K's (q+1)-simplex list; det gamma = 1;
    // coefficients in Z, not restricted to +-1). Throws std::out_of_range on a
    // bad index and std::invalid_argument on a non-boundary simplex. May be
    // legitimately empty (tau and tau_bd agree on the slot, or the fill is 0).
    const std::vector<ChainTerm>& eta(uint32_t j, uint32_t q, uint32_t sigma) const;
    // True iff the slot was filled through the (T1) helper rather than
    // constructed (reporting only; certification treats both alike).
    bool slot_derived(uint32_t j, uint32_t q, uint32_t sigma) const;

    // --- THE single (T1) helper ----------------------------------------------
    // eta(gamma . i(sigma)) for ARBITRARY gamma in Gamma, from the stored slot
    // of its Gamma''-class: gamma = h s_j via HeckeCosetTables::twisted_conjugate
    // ((P1), h in Gamma_0(N)), the chain is (g^{-1} h g) . eta(s_j . i(sigma)),
    // canonicalised in degree q + 1. Used by the construction (slot derivation,
    // the faces of an edge slot), by certify() (the right-hand sides of (ii)
    // and (iii)) and, later, by the cone action's derived h = eta~. Public so
    // that validations can cross-check it against an independent re-derivation.
    std::vector<ChainTerm> eta_on_translate(const Mat22& gamma, uint32_t q, uint32_t sigma) const;

    // The right-hand side of [C, eq:pairhomotopy] on a stored slot:
    // (j tau_bd - tau j)(s_j . i(sigma)) as a canonical q-chain of X, j being
    // the inclusion (a boundary simplex of K IS a simplex of K, so the
    // tau_bd-terms are re-used verbatim). Read from the two stored tables.
    std::vector<ChainTerm> defect(uint32_t j, uint32_t q, uint32_t sigma) const;
    // The same on an arbitrary translate gamma . i(sigma): both tau and
    // tau_bd satisfy (T1), so the difference does, and the value is
    // (g^{-1} h g) . defect(j, q, sigma) for gamma = h s_j.
    std::vector<ChainTerm> defect_on_translate(const Mat22& gamma, uint32_t q, uint32_t sigma) const;

    // The unique (q+1)-chain x of X with boundary z, for a q-chain z of X in
    // ChainTerm form (q + 1 <= dim X; q >= 1 for uniqueness, [C,
    // eq:contractibleuse]; for q = 0 it returns SOME 1-chain with boundary z).
    // The finite-neighbourhood integer solve of [C, alg:tau] step 3; throws
    // std::runtime_error if no fill exists within the ring budget (z not a
    // boundary) and HeckeOverflow on a tier failure. rings_out (optional)
    // receives the rings of growth used. This is what [C, lem:etaunique]'s L
    // is computed with.
    std::vector<ChainTerm> unique_fill(uint32_t q, const std::vector<ChainTerm>& z,
                                       uint32_t* rings_out = nullptr) const;

    // --- record fields ([C, obl:eta](iv), (vi) eta_zero; [C, open:ell2]) -----
    const char* solve_regime() const { return "Z"; } // every fill; no other regime exists here
    bool integral() const { return true; }
    // eta_zero := every stored chain is 0 ([C, obl:eta](vi), RECORDED, gates
    // nothing; equals "tau is a map of pairs", [C, rmk:conestrict]).
    bool eta_zero() const { return num_nonzero_ == 0; }
    uint32_t num_nonzero_slots() const { return num_nonzero_; }
    uint32_t num_nonzero_slots(uint32_t q) const;
    uint32_t num_terms() const { return num_terms_; }
    int64_t max_abs_coeff() const { return max_abs_coeff_; }
    // B_tab = max ||gamma_t|| over the stored table and B = max(B_tab, 4 N^2)
    // ([C, lem:bounds](5) recounted on the eta-chains, [C, obl:eta](iii));
    // certified by a full recount.
    uint64_t certified_bound_B_tab() const { return bound_B_tab_; }
    uint64_t certified_bound_B() const { return bound_B_; }
    uint32_t num_constructed_slots() const { return num_constructed_; }
    uint32_t num_derived_slots() const { return num_derived_; }
    uint32_t num_slots() const { return num_constructed_ + num_derived_; }
    uint32_t num_fills() const { return num_fills_; }       // edge (and higher) slots solved
    uint32_t max_rings_used() const { return max_rings_; }
    // [C, open:ell2] rider (i) on the finite model: q_max derived = the top
    // boundary degree eta is built in; truncated = false; the degree list.
    uint32_t q_max() const { return num_degrees() - 1; }
    bool truncated() const { return false; }
    const char* q_max_source() const { return "derived"; }
    std::vector<uint32_t> degrees() const;
    // FNV-1a-64 (ModArithmetic/Fnv.h) over the whole stored table in slot
    // order: (q, sigma, j, then each term's coeff, gamma, sigma_idx). Two
    // admissible eta's may differ here while the cone operator does not.
    uint64_t table_fnv64() const;
    // The same hash over the stored tau-table (every slot (j, q, sigma) of K)
    // and over the stored tau_bd-table: "together with the stored tau- and
    // tau_bd-tables it was built against" ([C, obl:eta], preamble) -- the
    // record names them by hash.
    uint64_t tau_table_fnv64() const;
    uint64_t tau_boundary_table_fnv64() const;
    // The three solve regimes of [C, obl:eta](vi): tau's ("Z", or "mod_ell"
    // had it any mod-ell fill -- refused by the constructor), tau_bd's ("Z"
    // unconditionally) and eta's ("Z").
    const char* solve_regime_tau() const;
    const char* solve_regime_tau_boundary() const { return taub_->solve_regime(); }
    // Vertex slots whose defect needed more than one A -> B path (0 on the
    // shipped tables, where every vertex value is a single vertex).
    uint32_t num_vertex_slots_multi_path() const { return num_vertex_pairs_beyond_one_; }

    // --- the MANDATORY certification ([C, obl:eta](i)-(iv)) -------------------
    // 0 iff ALL of the following hold EXACTLY over Z:
    //   (i)  10: every slot stored; 11: det gamma_t = 1; 12: every term in
    //        canonical form (SL2ZTessellationTables::canonical_key idempotent,
    //        sign +1); 13: coeff != 0; 14: sigma_t is a (q+1)-simplex of K;
    //        20: tau and tau_bd are built on the same coset datum and model
    //        (identity); 21: tau is integral (no mod-ell fill anywhere in it);
    //        22: num_degrees() == min(taub.num_degrees(), dim X);
    //   (ii) 30 (degree 0): boundary(eta(s_j sigma)) == (j tau_bd - tau j)(s_j sigma);
    //        31 (degree >= 1): boundary(eta(s_j sigma)) + eta(boundary(s_j sigma))
    //        == (j tau_bd - tau j)(s_j sigma) -- the faces ONLY through
    //        eta_on_translate, tau_bd through tau_on_translate, tau the stored
    //        slot (= its helper at h = I), compared as canonical maps;
    //   (iii) 40: for EVERY stored transporter t in S(sigma, sigma') between
    //        boundary simplices and every j, the slot (j, sigma) equals the
    //        sign-tracked (T1)-translate of the slot reached through s_j t
    //        (the {+-I} self-constraints included); 50/51: B_tab matches a
    //        full recount and B = max(B_tab, 4 N^2) satisfies 2 B^2 < 2^63;
    //   (iv) 60: the recorded regime of every slot is Z.
    // Called by the constructor, which throws on a nonzero result.
    uint32_t certify() const;
    // The same certification with ONE slot replaced by `replacement` (the
    // object itself is not modified); 0 iff the tampered table certifies.
    uint32_t certify_with_override(uint32_t j, uint32_t q, uint32_t sigma,
                                   const std::vector<ChainTerm>& replacement) const;

    std::string to_string() const { return to_string_offset(0); }
    std::string to_string_offset(uint32_t offset) const;
    // The record object of the table (hand-rolled JSON, no library):
    // schema "hecke_pair_homotopy/v1" -- (p, nu, N, m), the policy, the solve
    // regime and integral flag, degrees / q_max / truncated / q_max_source,
    // the slot counts, eta_zero, the bounds and the table hash.
    std::string to_json() const;

private:
    struct ChainRecord {
        std::vector<ChainTerm> terms; // a (q+1)-chain of X
        bool stored = false;
        bool derived = false;
        bool regime_Z = true;         // every fill; recorded per slot ([C, obl:eta](iv))
        uint32_t rings = 0;
    };
    // records_[q][local sigma][j-1], local sigma indexing boundary_simplices(q).
    using Records = std::vector<std::vector<std::vector<ChainRecord>>>;
    // Signed canonical accumulation map: {q, sigma_hat, a, b, c, d} -> coeff.
    using CanonMap = std::map<std::array<int64_t, 6>, int64_t>;

    const HeckeTauTableT<Elt>* tau_;
    const HeckeBoundaryTauTableT<Elt>* taub_;
    const SL2ZTessellationTables* model_;
    const HeckeCosetTablesT<Elt>* cosets_;
    EtaVertexPolicy policy_;
    uint64_t N_;
    uint32_t m_;
    std::vector<std::vector<std::vector<uint32_t>>> simplices_; // K's simplex lists
    Records records_;

    uint64_t bound_B_tab_ = 0;
    uint64_t bound_B_ = 0;
    uint32_t num_constructed_ = 0;
    uint32_t num_derived_ = 0;
    uint32_t num_fills_ = 0;
    uint32_t max_rings_ = 0;
    uint32_t num_nonzero_ = 0;
    uint32_t num_terms_ = 0;
    int64_t max_abs_coeff_ = 0;
    uint32_t num_vertex_pairs_beyond_one_ = 0;

    // --- bookkeeping ---------------------------------------------------------
    uint32_t local_index(uint32_t q, uint32_t sigma) const;
    void check_indices(uint32_t j, uint32_t q, uint32_t sigma) const;
    const ChainRecord& record(uint32_t j, uint32_t q, uint32_t sigma) const;
    ChainRecord& record_mut(uint32_t j, uint32_t q, uint32_t sigma);
    void accumulate_canonical(CanonMap& acc, const Mat22& gamma, uint32_t q, uint32_t sigma,
                              int64_t coeff) const;
    std::vector<ChainTerm> canon_map_to_terms(const CanonMap& acc, uint32_t q) const;
    CanonMap terms_to_canon_map(const std::vector<ChainTerm>& terms, uint32_t q) const;
    CanonMap boundary_of_terms(const std::vector<ChainTerm>& terms, uint32_t q) const;
    static bool canon_maps_equal(const CanonMap& x, const CanonMap& y);
    uint32_t face_simplex_index(uint32_t q, uint32_t sigma, uint32_t f) const;
    // (j tau_bd - tau j)(s_j . i(sigma)) as a canonical map.
    CanonMap defect_map(uint32_t j, uint32_t q, uint32_t sigma) const;
    // The helper over an explicit record set (certify_with_override runs the
    // certification on a modified copy; eta_on_translate uses records_).
    std::vector<ChainTerm> eval_on_translate(const Records& recs, const Mat22& gamma, uint32_t q,
                                             uint32_t sigma) const;
    // eta(boundary(s_j . i(sigma))) through the helper over `recs`, as a
    // canonical q-map (q >= 1).
    CanonMap eta_of_boundary(const Records& recs, uint32_t j, uint32_t q, uint32_t sigma) const;
    // The fill solver ([C, alg:tau] step 3 in degree q -> q+1).
    bool solve_fill(uint32_t q, const CanonMap& z, std::vector<ChainTerm>& x_out,
                    uint32_t& rings_out) const;

    // --- construction --------------------------------------------------------
    bool try_derive(uint32_t j, uint32_t q, uint32_t sigma);
    // Accumulates a 1-chain of X with boundary gamma_b . i(wb) - gamma_a . i(wa)
    // ([C, alg:tau] step 2's word + path-template device, family per policy_).
    void append_vertex_path(CanonMap& acc, const Mat22& ga, uint32_t wa, const Mat22& gb, uint32_t wb) const;
    void construct_vertex(uint32_t j, uint32_t v);          // [C, lem:etaexists](V)
    void construct_fill(uint32_t j, uint32_t q, uint32_t s); // [C, lem:etaexists](E) (degree-generic)
    void tally();

    uint32_t certify_impl(const Records& recs) const;
};

using HeckePairHomotopy = HeckePairHomotopyT<uint32_t>;
using HeckePairHomotopy64 = HeckePairHomotopyT<uint64_t>;

} // namespace hecke

#endif // HECKE_PAIR_HOMOTOPY_H
