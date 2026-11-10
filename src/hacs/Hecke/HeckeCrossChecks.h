// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_CROSS_CHECKS_H
#define HECKE_CROSS_CHECKS_H

// Stage S8 -- The three-map factorization cross-check route of
// [C, obl:agreement]: T_p = omega_* . beta_* . alpha_* realized at COCHAIN
// level and asserted BIT-IDENTICAL against the binding fused pipeline
// Theta_g = omega_bullet . Xi_g of Stage S7.
//
// Contract: docs/latex/hecke_action.tex ([C]):
//   * [C, prop:alpha]: alpha : V -> Ind_{Gamma'}^Gamma V, alpha(v)(gamma) =
//     rho_n(gamma) v; matrix = block COLUMN (rho_n(t_i))_{i=1..m}, an honest
//     morphism of local systems L -> L' (Gamma' = Gamma^0(N), representatives
//     t_i, top-row P^1(Z/N) classes).
//   * [C, prop:omega]: omega : Ind_{Gamma''}^Gamma V -> V, omega(f) =
//     sum_i rho_n(s_i)^{-1} f(s_i); matrix = block ROW (rho_n(s_i^{-1}))_i,
//     an honest morphism L'' -> L (Gamma'' = Gamma_0(N), representatives s_i).
//   * [C, rmk:nogo], remark "Recovering the three-map factorization of [v5]":
//     the middle map is NOT a morphism of local systems; it is the
//     per-simplex-block cochain map
//         (beta_bullet c')(sigma)_i
//             := rho_n(g) phi^{L'}_{c'}(tau(s_i . i(sigma)))  evaluated at 1,
//     where phi^{L'}_{c'} is the Ind_{Gamma'}V-valued equivariant cochain of
//     c' (phi^{L'}_{c'}(gamma . i(sigma)) = M_{Gamma'}(gamma) c'(sigma)) and
//     evaluation at 1 is the Gamma'-morphism Ind_{Gamma'}V -> V, f |-> f(1).
//   * [C, def:xi] / [C, eq:xiexplicit]: Xi_g and the fused Theta_g.
//
// ---------------------------------------------------------------------------
// BLOCKWISE DERIVATION OF beta_bullet (the load-bearing formula; the
// acceptance bit-identity below makes any slip here fail LOUDLY):
//
// tau(s_i . i(sigma)) = sum_k n_k gamma_k . i(sigma_k) (the S4 ChainTerm
// table, gamma_k in Gamma exact). Hence
//   phi^{L'}_{c'}(tau(s_i . i(sigma))) = sum_k n_k M_{Gamma'}(gamma_k) c'(sigma_k).
// In the ordered-basis coordinates of [C, def:inducedmodule] (t_1 = I, so
// "evaluation at 1" = block coordinate 1) and by [C, eq:MH], block ROW 1 of
// M_{Gamma'}(gamma_k) has its single nonzero block rho_n(h_1(gamma_k)) in
// block column j_k := 1 . gamma_k, where
//   h_1(gamma_k) = t_1 gamma_k t_{j_k}^{-1} = gamma_k t_{j_k}^{-1} in Gamma^0(N)
// (j_k = the t-index of the coset Gamma^0(N) gamma_k, i.e. the unique j with
// gamma_k t_j^{-1} upper-triangular mod N). Therefore
//
//   (beta_bullet c')(sigma)_i
//       = sum_k n_k rho_n(g * gamma_k t_{j_k}^{-1}) * [block j_k of c'(sigma_k)],
//
// i.e. the FiberBlockTable entry for (target sigma, source sigma') is the
// m*(n+1) x m*(n+1) matrix whose block (i, j) accumulates
// sum_{k : sigma_k = sigma', j_k = j} n_k * rho_n((g gamma_k t_{j_k}^{-1}) mod ell).
// The matrix g * h (det N) is formed EXACTLY over Z (overflow-checked Mat22,
// invariant I6) and is NEVER inverted (invariant I7 / [C, obl:integral]).
//
// Consistency with Xi_g ([C, eq:xiexplicit]) is the per-term monoid identity
// rho_n(g h_k) rho_n(t_{j_k}) = rho_n(g gamma_k) mod ell ([C, prop:symconv](2)),
// which is what makes the composites below BIT-identical, not merely equal
// in cohomology.
// ---------------------------------------------------------------------------
//
// AMENDMENT A-F3 (binding, plan Sec. 0.1 -- localizing assertions BEFORE the
// full triple identity):
//   (i)  beta_bullet alone maps L'-equivariant cochains to L''-equivariant
//        cochains: per-relation check (beta c')(sigma^-) =
//        M_{Gamma''}(g_r) (beta c')(sigma^+) over the FULL induced-system
//        relation list of L'', for a full basis of C^q_eq(K,R;L') (computed
//        here directly as the kernel of the L'-relation constraint matrix);
//   (ii) beta_bullet . alpha_bullet == Xi_g -- asserted bit-identically on
//        the FULL standard basis of C^q(K;V) (hence in particular on
//        im alpha_bullet), against an independent re-evaluation of
//        [C, eq:xiexplicit] from the S4 xi_terms table (S7's fused pre-omega
//        map, reconstructed here since HeckeOperatorOnK does not expose it).
// ACCEPTANCE (the S8 DoD identity; do NOT relax):
//   (iii) omega_bullet . beta_bullet . alpha_bullet == Theta_g bit-identically
//         on ALL of C^q(K;V) for q = 0,1,2, as matrices, against Stage S7's
//         get_Theta_flat_matrices().
//
// CONTRAVARIANT WIRING (read the Stage S6 note in
// EquivariantCochainComplexMorphismOnK.h before touching this): the class's
// "domain" is the TARGET of the induced H*-map and its fiber map F : V2 -> V1
// maps the fiber of the cochain SOURCE (codomain, L2) to the fiber of the
// cochain TARGET (domain, L1). Hence:
//   alpha_bullet : C(K;L) -> C(K;L')  has domain = L' cohomology wrapper,
//       codomain = L cohomology, F = the (m d_V) x d_V block column;
//   beta_bullet  : C(K;L') -> C(K;L'') has domain = L'', codomain = L';
//   omega_bullet : C(K;L'') -> C(K;L)  has domain = L, codomain = L'',
//       F = the d_V x (m d_V) block row.
// The L' / L'' wrappers are NOT computed (only their relation lists and
// dimensions are consumed); the checks are pure cochain-level identities.
//
// Flat cochain layout: simplex-major, fiber-minor, binding and identical
// across S6/S7/S8 (see EquivariantCochainComplexMorphismOnK.h). For the
// induced systems the fiber coordinate of block i (1-based coset index) and
// in-block coordinate k is (i-1)*(n_sym+1) + k, matching [C,
// def:inducedmodule]'s ordered basis and Stage S5's M_H layout.
//
// Placement (documented per the task's instructions): like S2-S7, this class
// consumes ModArithmetic and the model/cohomology layer, so it is compiled
// into the cohomology_app executable sources (NOT into the pure-integer
// HeckeCore static library), linking HeckeCore for hecke::Mat22.

#include <cstdint>
#include <memory>
#include <vector>

#include "ExactInt.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/BlockMat.h"

namespace hecke {

class HeckeCrossChecks {
public:
    // cohomology_of_L: the COMPUTED base system L = Sym^n(F_ell^2) on the
    // 7-vertex model; cosets/tau: the S3/S4 tables for g = diag(1, N) built
    // against the same (ell, n_sym). All three must outlive this object.
    // The constructor builds L' = L_{Gamma^0(N)} and L'' = L_{Gamma_0(N)}
    // (Stage S5), reconstructs the exact Gamma^0(N) representatives t_i per
    // the S3 closed-form rule (verified against cosets.rho_t(i) entrywise),
    // and assembles the alpha / beta / omega morphisms. Throws on any
    // inconsistency.
    HeckeCrossChecks(const EquivariantCohomologyOnK* cohomology_of_L,
                     const HeckeCosetTables* cosets, const HeckeTauTable* tau);

    // A-F3 (i). 0 on success (nonzero codes localize the failure; details on
    // std::cout).
    uint32_t check_beta_equivariance() const;

    // A-F3 (ii). 0 on success.
    uint32_t check_beta_alpha_equals_Xi() const;

    // The acceptance identity (iii) against Stage S7's fused Theta_g.
    // 0 on success.
    uint32_t check_triple_factorization(const HeckeOperatorOnK& op) const;

    // Runs (i), (ii), (iii) in the A-F3 order; 0 on success.
    uint32_t run_all(const HeckeOperatorOnK& op) const;

    // The three cochain maps (cochain application via
    // apply_to_flat_cochain; see the layout note above).
    //
    // (matrix-engine Phase 3 Stage B) The CHECKS below no longer route through
    // these objects: alpha/beta/omega/Xi are each held as ONE engine::BlockMat
    // per degree (see the block-representation note in the .cpp) and the three
    // identities are block GEMMs over the whole of C^q at once. The morphism
    // objects are kept because they are this class's published surface and the
    // only place the S6 FiberBlockTable form of beta_bullet is written down.
    // alpha_ / omega_ are still built eagerly (their fibre matrices are
    // d_ind x d_V resp. d_V x d_ind); beta() materialises its dense
    // d_ind x d_ind fibre blocks ON FIRST USE only -- eagerly building them
    // costs m x the memory of the block-sparse form (audit 02 Sec. 1.6 X1:
    // density <= 1/m) for data no check reads any more.
    const EquivariantCochainComplexMorphismOnK& alpha() const { return *alpha_; }
    const EquivariantCochainComplexMorphismOnK& beta() const;
    const EquivariantCochainComplexMorphismOnK& omega() const { return *omega_; }

    // Independent re-evaluation of Xi_g ([C, eq:xiexplicit]) on a flat
    // q-cochain of L, from the S4 xi_terms table (the fused pre-omega map).
    // Output: flat q-cochain of L'' (length N_q * m * d_V).
    FlatMatrix apply_Xi(uint32_t q, const FlatMatrix& c_flat) const;

private:
    const EquivariantCohomologyOnK* coh_L_;
    const HeckeCosetTables* cosets_;
    const HeckeTauTable* tau_;

    uint32_t ell_ = 0;
    uint32_t n_sym_ = 0;
    uint32_t d_V_ = 0;   // n_sym + 1
    uint32_t m_ = 0;     // num_cosets
    uint32_t d_ind_ = 0; // m * d_V (fiber dim of L' and L'')
    uint32_t N32_ = 0;   // N = p^r as uint32_t

    // Exact Gamma^0(N) representatives t_i (1-based; index 0 unused),
    // reconstructed per the S3 closed-form rule and verified against
    // cosets_->rho_t(i).
    std::vector<Mat22> t_;
    std::vector<Mat22> t_inv_;

    // Induced systems (Stage S5) and their (uncomputed) cohomology wrappers.
    std::unique_ptr<LocalSystemSimplicialComplex> lssc_Lp_;  // L' = L_{Gamma^0(N)}
    std::unique_ptr<LocalSystemSimplicialComplex> lssc_Lpp_; // L'' = L_{Gamma_0(N)}
    std::unique_ptr<EquivariantCohomologyOnK> coh_Lp_;
    std::unique_ptr<EquivariantCohomologyOnK> coh_Lpp_;

    // Identity simplicial morphisms for the legacy (single-fiber-matrix)
    // constructor used by alpha / omega ([C]: "alpha_*, omega_* via
    // EquivariantCochainComplexMorphismOnK as-is").
    std::unique_ptr<SimplicialComplexMorphism> id_alpha_;
    std::unique_ptr<SimplicialComplexMorphism> id_omega_;

    std::unique_ptr<EquivariantCochainComplexMorphismOnK> alpha_;
    mutable std::unique_ptr<EquivariantCochainComplexMorphismOnK> beta_; // lazy, see beta()
    std::unique_ptr<EquivariantCochainComplexMorphismOnK> omega_;

    // (matrix-engine Phase 3 Stage B) The block representation the three
    // A-F3 identities are actually evaluated in. One BlockMat per degree, all
    // over the SAME content-addressed pool, so every distinct rho_n(.) block is
    // stored exactly once:
    //   alpha_bm_[q] : (N_q*m) x N_q      blocks of d_V   (block (s*m+i-1, s) = rho_n(t_i))
    //   beta_bm_[q]  : (N_q*m) x (N_q*m)  blocks of d_V
    //   omega_bm_[q] : N_q x (N_q*m)      blocks of d_V   (block (s, s*m+i-1) = rho_n(s_i^-1))
    //   xi_bm_[q]    : (N_q*m) x N_q      blocks of d_V   ([C, eq:xiexplicit])
    // The block row/column index (s*m + (i-1), fibre coordinate k) is exactly
    // the flat index s*d_ind + (i-1)*d_V + k of the binding L1 layout, so the
    // grids and the flat cochain layout agree by construction.
    std::shared_ptr<engine::BlockPool<uint32_t>> pool_;
    std::vector<engine::BlockMat<uint32_t>> alpha_bm_;
    std::vector<engine::BlockMat<uint32_t>> beta_bm_;
    std::vector<engine::BlockMat<uint32_t>> omega_bm_;
    std::vector<engine::BlockMat<uint32_t>> xi_bm_;
    mutable std::vector<engine::BlockMat<uint32_t>> beta_alpha_bm_; // lazy cache of beta*alpha
    mutable std::vector<bool> beta_alpha_built_;

    const SimplicialComplex* K() const; // base 7-vertex model of coh_L_

    // rho_n(gamma_exact mod ell) via ComputeSymmetricPowerMatrixFM
    // (invariant I2); gamma_exact may have det N and is never inverted.
    FlatMatrix rho_of_exact(const Mat22& gamma_exact) const;

    // The unique j in [1, m] with gamma * t_j^{-1} in Gamma^0(N) (upper-right
    // entry == 0 mod N); asserts uniqueness (P^1(Z/N) partition) and returns
    // h_out = gamma * t_j^{-1} exactly. This is "1 . gamma" of [C, eq:MH]
    // for the Gamma^0(N) tables.
    uint32_t upper_index_and_cocycle(const Mat22& gamma, Mat22& h_out) const;

    void build_t_representatives();

    // Builds alpha_bm_ / beta_bm_ / omega_bm_ / xi_bm_ (see above).
    void build_block_maps();
    // beta_bm_[q] * alpha_bm_[q], computed once per degree on first use.
    const engine::BlockMat<uint32_t>& beta_alpha(uint32_t q) const;

    // The S6 FiberBlockTable form of beta_bullet, materialised from
    // beta_bm_ (block-sparse -> the dense d_ind x d_ind blocks the
    // EquivariantCochainComplexMorphismOnK interface requires).
    FiberBlockTable build_beta_table() const;

    // Basis of C^q_eq(K,R;L') as ONE column-major (N_q*d_ind) x dim buffer:
    // kernel of the L'-relation constraint matrix (out(s_lhs) - M_r *
    // out(s_rhs) = 0 stacked over the degree-q relations), assembled and
    // eliminated through the block model.
    engine::DenseBuf<uint32_t> equivariant_cochain_basis_Lp(uint32_t q) const;

    // Failure reporters: re-scan in the ORIGINAL (pre-Phase-3) loop order so
    // the diagnostic names the same first offending (basis vector, relation,
    // row) the column-at-a-time code named. Only ever called after a batched
    // comparison has already found a mismatch.
    uint32_t report_equivariance_failure(uint32_t q, const engine::DenseBuf<uint32_t>& out) const;
};

} // namespace hecke

#endif // HECKE_CROSS_CHECKS_H
