// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_MODEL_STRICT_LIFT_H
#define INTEGRAL_MODEL_STRICT_LIFT_H

// ===========================================================================
// Line 2, L2-1b (1) (hecke-extensions slot 10, WP-10): the STRICT LIFT
// tau_Iso of the strict pair datum tau^s to the isotropy-resolution model
// R = Iso_. ([C, def:isomodel]), its coset transport Theta^_g on
// C^._I(R) = Hom_{Z[Gamma]}(R, V_R) over any engine::ModCtx, and the exact
// certificates of [C, prop:strictlift] (1).  [C] =
// docs/latex/gl2_contract.tex (v1.9.1), cited by label; [IB] =
// docs/implementation/integral_line_design_brief.md.
//
// WHAT IS BUILT.  For g = diag(1, N) (N = p, a double-coset representative of
// T_p) and Gamma'' = Gamma_0(N) ([C, prop:cosets] (3)), R is FREE over
// Z[Gamma''] on the basis { s_j e_c : j = 1..m, c a cell } (s_j the shipped
// coset representatives of Gamma'' \ Gamma, e_c the free Z[Gamma]-generators
// of R).  A map with (T1) is therefore a free choice of tau_Iso(s_j e_c) in R,
// extended by tau_Iso(h s_j e_c) := (g^{-1} h g) tau_Iso(s_j e_c), h in
// Gamma'' -- the one (T1) helper, hecke::HeckeCosetTables::twisted_conjugate,
// shared with the shipped tau / tau_bd tables.  The values are built by the
// induction of [C, lem:strictlifting], degree by degree, q = 0 .. q_max + 1:
//
//   * boundary summands first ([C, prop:strictlift] (1): P = R_d, f = tau_bd o
//     eps, P' = 0, giving tau^d_Iso; then P = R, f = tau^s o eps, P' = R_d);
//   * y_0 with eps(y_0) = f(x): for a cell at t = 0 (s = q <= 2) the stored
//     chain tau^s(s_j . i(sigma)) of the shipped BoundaryStrict table, written
//     term by term as gamma e_{rho, 0} (eps is the augmentation of free
//     generators, onto); y_0 = 0 at t >= 1, where eps = 0;
//   * z := tau_Iso(d x) - d y_0 is a cycle of the acyclic ker eps, and the
//     correction k with d k = z is produced by DESCENT THROUGH THE COLUMNS:
//     the top column's component of z is a cycle of the augmented periodic
//     column F^sigma, so the closed-form Z-linear contracting homotopy h of
//     that column ([IB] Sec. 3.2 item 1; the same h IsotropyModel builds d^(1),
//     d^(2) with) gives k_s = (-1)^s h(z_s) with d^(0) k_s = z_s; z -= d k_s
//     clears column s and moves the rest to lower columns; after column 0,
//     z = 0.  Every k_s lives at t >= 1, so eps(k) = 0.  No linear system is
//     solved: the plan's "HeckeIntegerSolve on the t = 0 layer" is not needed,
//     because eps is onto by construction at t = 0 and the columns carry
//     explicit homotopies (report: docs/implementation/slot10/WP-10_report.md).
//   * tau_Iso(x) := y_0 + k.  Every homotopy input is ENGINE_CHECKed to be a
//     cycle; the final z = 0 is ENGINE_CHECKed.
//
// CERTIFICATES (certify(): re-derived from the stored table, exactly over
// Z[Gamma] and on canonical chains of Xbulk -- nothing is trusted from the
// construction):
//   (T1)  on the Schreier generators h_i(x) = s_i x s_{i.x}^{-1}, x in {T, S},
//         of Gamma'' (a generating set, by Schreier's lemma): tau_Iso(h y) =
//         (g^{-1} h g) tau_Iso(y) for y = s_j e_c and y = s_j u e_c, u in
//         {T, S}, every cell of every degree; the conjugate g^{-1} h g is
//         formed independently of the helper (N | h.c checked);
//   (T2)  d tau_Iso(s_j e_c) = tau_Iso(d (s_j e_c)), every (j, c), q >= 1;
//   (T3)  aug eps tau_Iso(x) = aug eps(x) in degree 0;
//   (S)   eps tau_Iso(x) = tau^s eps(x) on every basis element, as canonical
//         chains of Xbulk (SL2ZTessellationTables::canonical_key) -- STRICTLY;
//   (B)   tau_Iso(R_d) in R_d, and eps tau^d_Iso(x) = tau_bd eps(x) against the
//         boundary table itself (so eta_pair == 0 is a pair homotopy,
//         [C, prop:strictlift] (1));
//   integer sizes (max |coefficient|, max |matrix entry|, max l1 norm, term
//   counts per degree) -- [C] bounds none of them (K1 of [IB]).
//
// THE COSET TRANSPORT ([C, eq:cosettransport] verbatim, [C, prop:strictlift]
// (3)): theta_hat(ctx, n, q) is the (w_q d_V)^2 matrix of
//   (Theta^ phi)(e_c) = sum_j rho_n(delta_j) phi(tau_Iso(s_j e_c))
//                     = sum_j sum_{(gamma, m) in tau_Iso(s_j e_c) at e_c'}
//                         m rho_n(delta_j gamma) phi(e_c'),
// delta_j = s_j^{-1} g, the integer product delta_j gamma formed exactly
// (det N, never inverted), then reduced by integral_model::rho_n over ctx --
// modulus-only, valid over every ModCtx (a field or Z/ell^k).  Same L1 layout
// as IsotropyModel::coboundary.
//
// THE det-p^2 DOUBLE COSETS (hecke-extensions slot 11, WP-11, M-3; appended,
// the T_p path unchanged).  The same class, the same induction and the same
// certificates serve
//   * g = diag(1, p^nu), nu in {1, 2, 3} (the PRIMITIVE double coset of
//     determinant p^nu, [C, prop:cosets] (1)): Gamma'' = Gamma_0(p^nu),
//     m = p^nu + p^{nu-1}, the shipped coset tables HeckeCosetTables(p, nu)
//     and the shipped BoundaryStrict table on them -- nothing in the
//     construction or in certify() depended on nu = 1 except the refusal;
//   * g = p I (the central double coset of T(p, p), [C, prop:diamond]):
//     Gamma'' = Gamma, m = 1, s_1 = I, delta_1 = p I, conj_g = id, and the
//     explicit datum tau = id of prop:diamond's proof in place of tau^s
//     (tau_bd = id likewise).  The induction runs with f = eps (y_0 = x at
//     t = 0; the descent at t >= 1), so tau_Iso is a lift of the identity --
//     homotopic to id, not necessarily equal to it (report WP-11 records which).
//     The certificates read: (T1) on the Schreier generators T, S of Gamma
//     (conj = h); (S) eps tau_Iso = eps; (B) likewise against tau_bd = id.
// Theta^ for p I is then rho_n(p I) = p^n times the transport of that lift.
// The FULL operators of [C, eq:fullassembly] are assembled by the caller:
//   T_{p^2} = Theta^_{diag(1,p^2)} + Theta^_{pI},
//   T_{p^3} = Theta^_{diag(1,p^3)} + p^n Theta^_{diag(1,p)}
// (the second term: the double coset p diag(1, p) uses diag(1, p)'s datum with
// delta_t scaled by p, [C, prop:fullassembly]'s proof, already at cochain level).
//
// DEPENDENCIES: IsotropyModel (L2-1a), the shipped coset tables, the shipped
// BoundaryStrict tau table tau^s ([C, prop:strictpair]; built on the boundary
// table tau_bd) and its tessellation tables -- all READ, none edited.  No
// HeckeExt/, no big integer, no __int128 (OD-3 (c)); exact int64 through
// hecke::Mat22 / checked arithmetic, throwing hecke::HeckeOverflow.
// ===========================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "IsotropyModel.h"
#include "../../Hecke/ExactInt.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/ModCtx.h"

class SimplicialComplex;

namespace hecke {
template<class Elt> class HeckeCosetTablesT;
template<class Elt> class HeckeTauTableT;
template<class Elt> class HeckeBoundaryTauTableT;
class SL2ZTessellationTables;
struct ChainTerm;
}  // namespace hecke

namespace integral_model {

// An element of Iso_q: one Z[Gamma]-coefficient per cell of degree q, the
// element being sum_c coeff_c e_c (IsotropyModel::cells(q) order).
struct IsoChain {
    std::vector<GroupRingElt> c;

    IsoChain() = default;
    explicit IsoChain(size_t w) : c(w) {}
    bool is_zero() const;
    void add(const IsoChain& o, int64_t scale = 1);    // this += scale * o (overflow-checked)
    bool operator==(const IsoChain& o) const { return c == o.c; }
    bool operator!=(const IsoChain& o) const { return !(*this == o); }
    uint64_t num_terms() const;                        // sum over cells of #(gamma, m)
    int64_t max_abs_coefficient() const;
    int64_t max_abs_entry() const;                     // max |matrix entry| of a group element
    uint64_t l1_norm() const;                          // sum |m|
    std::string to_string() const;
};

// u * x (left multiplication of every coefficient by the group element u).
IsoChain left_multiply(const hecke::Mat22& u, const IsoChain& x);

struct StrictLiftCertificate {
    bool t1_twisted_equivariance = false;   // (T1) on the Schreier generators of Gamma''
    bool t2_chain_map = false;              // (T2)
    bool t3_augmentation = false;           // (T3)
    bool strict_eps = false;                // (S) eps tau_Iso = tau^s eps, strictly
    bool boundary_preserved = false;        // (B) tau_Iso(R_d) in R_d
    bool boundary_eps = false;              // (B) eps tau^d_Iso = tau_bd eps (against the boundary table)
    uint64_t t1_checks = 0, t2_checks = 0, t3_checks = 0, strict_checks = 0, boundary_checks = 0;
    uint32_t schreier_generators = 0;
    // Integer sizes, per degree q = 0 .. q_top.
    std::vector<uint64_t> terms;            // total #(gamma, m) over all stored values
    std::vector<uint64_t> max_terms;        // max #(gamma, m) of one value
    std::vector<uint64_t> max_l1;           // max l1 norm of one value
    int64_t max_abs_coefficient = 0;
    int64_t max_abs_entry = 0;
    uint64_t distinct_group_elements = 0;
    std::string first_failure;              // empty when all hold
    bool all() const {
        return t1_twisted_equivariance && t2_chain_map && t3_augmentation && strict_eps && boundary_preserved &&
               boundary_eps;
    }
};

// Which double coset a lift / bundle is for (slot-11 WP-11).
struct DoubleCoset {
    bool central = false;   // g = p I (T(p, p)); else g = diag(1, p^nu)
    uint32_t nu = 1;        // the exponent of diag(1, p^nu) (1 = T_p); 2 for p I (det p^2)
    static DoubleCoset diag_one(uint32_t nu_) { return DoubleCoset{false, nu_}; }
    static DoubleCoset scalar() { return DoubleCoset{true, 2}; }
    std::string name() const;   // "diag(1,p)", "diag(1,p^2)", "diag(p,p)", ...
};

class StrictLift {
public:
    // Builds tau_Iso in degrees 0 .. R.q_top().  tau_s must be a BoundaryStrict
    // table built on `taub` and on `cosets` (checked), with its model tables;
    // all four objects must outlive this one.  Throws (ENGINE_CHECK /
    // hecke::HeckeOverflow) on any failed construction step.
    StrictLift(const IsotropyModel& R, const hecke::HeckeCosetTablesT<uint32_t>& cosets,
               const hecke::HeckeTauTableT<uint32_t>& tau_s, const hecke::HeckeBoundaryTauTableT<uint32_t>& taub);
    // Slot-11 WP-11: the central double coset g = p I (T(p, p)), tau = id
    // (the header note).  `tess` supplies the canonical chains (S) and (B)
    // are compared on; it must outlive this object.
    StrictLift(const IsotropyModel& R, const hecke::SL2ZTessellationTables& tess, uint64_t p_central);

    const IsotropyModel& model() const { return *R_; }
    // The shipped tables (diag(1, p^nu) only; ENGINE_CHECKed).
    const hecke::HeckeCosetTablesT<uint32_t>& cosets() const;
    const hecke::HeckeTauTableT<uint32_t>& tau_s() const;
    const hecke::HeckeBoundaryTauTableT<uint32_t>& taub() const;
    bool central() const { return central_; }
    const hecke::Mat22& g() const { return g_; }
    DoubleCoset double_coset() const;
    uint32_t q_top() const { return static_cast<uint32_t>(table_.size()) - 1; }
    uint32_t num_cosets() const { return m_; }
    // (S13 WP-12, slot-11 m-8) NB the central constructor sets N_ = p^2 = det g (not the diag(1, N) reading).
    uint64_t N() const { return N_; }

    // (slot-18 WP-7) The coset data, read-only, for a strict lift on ANOTHER model built from this
    // one's datum (RefinedStrictLift.h): s_j, delta_j = s_j^{-1} g, and the (T1) decomposition
    // gamma = h s_j -> g^{-1} h g (the shipped coset tables' helper; no second coset construction).
    hecke::Mat22 coset_representative(uint32_t j) const { return coset_s(j); }
    hecke::Mat22 coset_delta_matrix(uint32_t j) const { return coset_delta(j); }
    hecke::Mat22 twisted_decomposition(const hecke::Mat22& gamma, uint32_t& j) const { return twisted(gamma, j); }

    // tau_Iso(s_j e_cell), j in [1, m] (the shipped 1-based coset index).
    const IsoChain& value(uint32_t q, uint32_t j, uint32_t cell) const;
    // tau_Iso(gamma e_cell) for ARBITRARY gamma in Gamma, through (T1).
    IsoChain on_translate(const hecke::Mat22& gamma, uint32_t q, uint32_t cell) const;
    // tau_Iso of an arbitrary element of Iso_q.
    IsoChain apply(uint32_t q, const IsoChain& x) const;
    // The differential d_q : Iso_q -> Iso_{q-1} of the model on an element.
    IsoChain boundary(uint32_t q, const IsoChain& x) const;

    // Re-derives every certificate (the note above).  Never throws on a failed
    // identity; records it.
    StrictLiftCertificate certify() const;

    // Theta^_g on C^q_I over ctx, q <= q_top(): (w_q d_V) x (w_q d_V).
    template<class Elt>
    engine::DenseBuf<Elt> theta_hat(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const;

    // NEGATIVE-CONTROL HOOK (validation only): a copy whose value (q, j, cell)
    // has `scale * e_cell` added -- a single perturbed entry.
    StrictLift perturbed(uint32_t q, uint32_t j, uint32_t cell, int64_t scale = 1) const;
    // NEGATIVE-CONTROL HOOK (validation only; slot-24 D4, C-D7p referee m4): tau' = tau + d K + K d for the
    // homotopy K : Iso_q -> Iso_{q+1} with K(s_j e_cell) = k, extended by (T1) (K(h s_j e_cell) =
    // (g^{-1} h g) k, h in Gamma''), zero on every other basis element.  tau' keeps (T1)-(T3); only (S)
    // and the boundary items (B) can see it.
    StrictLift with_homotopy(uint32_t q, uint32_t j, uint32_t cell, const IsoChain& k) const;

private:
    const IsotropyModel* R_;
    const hecke::HeckeCosetTablesT<uint32_t>* cosets_;
    const hecke::HeckeTauTableT<uint32_t>* tau_s_;
    const hecke::HeckeBoundaryTauTableT<uint32_t>* taub_;
    const hecke::SL2ZTessellationTables* tess_;
    uint32_t m_ = 0;
    uint64_t N_ = 0;
    // Slot-11 WP-11: the central case (cosets_ / tau_s_ / taub_ null) and g.
    bool central_ = false;
    hecke::Mat22 g_;
    // Per s in {0,1,2}: K's q-simplex index of each representative, and for
    // every K q-simplex (q <= 2) its (representative, transport) with
    // K-simplex = transport * representative.
    std::vector<std::vector<uint32_t>> rep_kidx_;
    std::vector<std::vector<std::pair<uint32_t, hecke::Mat22>>> kidx_to_rep_;
    // Stabiliser powers g_sigma^0 .. g_sigma^{m-1} per (s, rep).
    std::vector<std::vector<std::vector<hecke::Mat22>>> pow_;
    // table_[q][j - 1][cell].
    std::vector<std::vector<std::vector<IsoChain>>> table_;

    void init_representatives();
    void build();
    // The coset data, for both kinds (slot-11 WP-11): s_j, delta_j, the
    // (T1) decomposition gamma = h s_j -> g^{-1} h g, the Schreier data, and
    // the stored chains of tau^s and tau_bd (tau = id in the central case).
    hecke::Mat22 coset_s(uint32_t j) const;
    hecke::Mat22 coset_delta(uint32_t j) const;
    hecke::Mat22 twisted(const hecke::Mat22& gamma, uint32_t& j) const;
    std::vector<hecke::ChainTerm> tau_s_chain(uint32_t j, uint32_t q, uint32_t kidx) const;
    std::vector<hecke::ChainTerm> taub_chain(uint32_t j, uint32_t q, uint32_t kidx) const;
    IsoChain chain_to_iso(uint32_t q, const std::vector<hecke::ChainTerm>& terms) const;
    IsoChain contract(uint32_t qz, const IsoChain& z) const;   // k in Iso_{qz+1}, d k = z, t >= 1
    IsoChain basis_boundary(uint32_t q, uint32_t j, uint32_t cell) const;   // d(s_j e_cell)
};

// The shipped table data of two tau tables (every slot, every term: coeff,
// exact gamma, sigma index) -- "the same integral datum" (the tables built at
// different (ell, n) must agree: their construction is over Z).
bool SameIntegralDatum(const hecke::HeckeTauTableT<uint32_t>& a, const hecke::HeckeTauTableT<uint32_t>& b);

// Everything the strict lift of T_p needs, owned: the 7-vertex K, its
// tessellation tables, the coset tables of g = diag(1, p), the boundary table
// tau_bd, the BoundaryStrict table tau^s (built at the reference (ell_ref,
// n_ref); the datum is integral and independent of both), the model R and the
// lift.  Non-copyable, non-movable (the objects point at each other).  This is
// the entry point WP-11 (M1(a)) and WP-13 build on.
class StrictLiftBundle {
public:
    explicit StrictLiftBundle(uint64_t p_hecke, uint32_t q_max = 5, uint32_t ell_ref = 10007, uint32_t n_ref = 0);
    // Slot-11 WP-11: any of the appended double cosets (diag(1, p^nu), nu <= 3,
    // or p I).  For p I the coset / tau tables are not built (their accessors
    // throw); K, tess, the model and the lift are.  The (p, q_max) form above
    // is diag(1, p), unchanged.
    StrictLiftBundle(uint64_t p_hecke, DoubleCoset dc, uint32_t q_max = 5, uint32_t ell_ref = 10007,
                     uint32_t n_ref = 0);
    ~StrictLiftBundle();
    StrictLiftBundle(const StrictLiftBundle&) = delete;
    StrictLiftBundle& operator=(const StrictLiftBundle&) = delete;

    uint64_t p() const { return p_; }
    const DoubleCoset& double_coset() const { return dc_; }
    const SimplicialComplex& K() const { return *K_; }
    const hecke::SL2ZTessellationTables& tess() const { return *tess_; }
    const hecke::HeckeCosetTablesT<uint32_t>& cosets() const;
    const hecke::HeckeBoundaryTauTableT<uint32_t>& taub() const;
    const hecke::HeckeTauTableT<uint32_t>& tau_s() const;
    const IsotropyModel& model() const { return *R_; }
    const StrictLift& lift() const { return *lift_; }

private:
    uint64_t p_;
    DoubleCoset dc_;
    std::unique_ptr<SimplicialComplex> K_;
    std::unique_ptr<hecke::SL2ZTessellationTables> tess_;
    std::unique_ptr<hecke::HeckeCosetTablesT<uint32_t>> cosets_;
    std::unique_ptr<hecke::HeckeBoundaryTauTableT<uint32_t>> taub_;
    std::unique_ptr<hecke::HeckeTauTableT<uint32_t>> tau_s_;
    std::unique_ptr<IsotropyModel> R_;
    std::unique_ptr<StrictLift> lift_;
};

}  // namespace integral_model

#endif
