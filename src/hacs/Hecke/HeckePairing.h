// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_PAIRING_H
#define HECKE_PAIRING_H

// ===========================================================================
// hecke-extensions slot 6, WP-3 -- phase E7b: THE COHOMOLOGICAL PAIRINGS.
//
// [C] = docs/latex/gl2_contract.tex (cited by label); [X] = the extension
// implementation plan, Sec. 6.2 E7 (second half); the slot-6 plan Sec. 3 WP-3.
//
// WHAT THIS UNIT BUILDS.  [C, cl:pd] is OPEN: its coefficient-level input is
// [C, prop:pairing] (proved) and its cohomological content is what this unit
// lets the validation TEST ([C, rmk:pdscope](2): "Not yet available: a
// cochain-level pairing on the stored bases").  Here is that pairing:
//
//   1. THE RELATIVE FUNDAMENTAL 2-CHAIN of the model (ModelFundamentalChain).
//      [K] := sum_f eps_f f over the 2-simplices of K (each in increasing
//      vertex order).  eps is found COMBINATORIALLY: the sign vector for which
//      d[K] vanishes in the Gamma-coinvariants C_1(X)_Gamma away from the
//      horocycle -- the side edges {0,4} ~ T{0,4} = {3,5} and the arc edges
//      {5,6} ~ S{5,6} = {4,6} cancel through the degree-1 seed relations of
//      [C, sec:model] (orientation sign = the parity of the seed's vertex
//      correspondence, +1 on both) and every other interior edge cancels
//      against its neighbour -- and whose remainder d[K] is carried by the
//      boundary path K_bd with ONE common coefficient on every edge.  Exactly
//      two sign vectors qualify (+-eps), and the one whose horocycle
//      coefficient is -1 is taken: the vertices v0..v3 sit on the line
//      Im z = 2 with increasing real part (SL2ZTessellationTables::
//      vertex_position) and K lies below it, so the counter-clockwise
//      (positive, orientation of H) boundary runs along the horocycle from
//      v3 to v0.  The global sign is a convention: no verdict below depends
//      on it (a perfect pairing, an adjointness and the iota-ratio are all
//      invariant under negating it).  Every 2-simplex of X has stabiliser
//      {+-I}, which acts trivially, so no orbifold weight enters.
//
//   2. THE CUP-PRODUCT EVALUATION FORM (CupEvaluationForm).  K, K_bd and X
//      are ORDERED complexes compatibly with Gamma: every relation of the
//      model maps the increasing vertex order of its right-hand simplex to
//      that of its left-hand one (checked for the degree-1 seeds at build),
//      so the Alexander-Whitney cup product
//          (a u b)(v_0 .. v_k) = < a(v_0 .. v_i), b(v_i .. v_k) >
//      of an equivariant V-cochain a with an equivariant V^vee-cochain b is a
//      Gamma-INVARIANT F_ell-cochain ([C, prop:pairing](4) at det = 1), and
//      its value on a simplex of K reads only the stored values of the
//      front and back faces, which are simplices of K.  The fibre pairing is
//      [C, prop:pairing](2)'s Gram matrix (PairingGram, = I).
//
//   3. THE CONE-LEVEL PAIRING (the cochain level of [C, cl:pd](1), first
//      pairing).  For a cone cochain (c, c_bd) in Cc^t = C^t(K; M_n) (+)
//      C^{t-1}(K_bd; M_n) (EquivariantMappingCone's H_c-indexing) and
//      y in C^{2-t}(K; D_n),
//
//          P_t((c, c_bd), y) := < c u y , [K] >  +  < c_bd u r y , d[K] > ,
//
//      r the restriction to K_bd (identity fibre).  With d(a u b) =
//      da u b + (-1)^{|a|} a u db, Stokes on the invariant cochain
//      (<dw, [K]> = <r w, d[K]>) and d(d[K]) = 0 in coinvariants
//      (v3 = T v0), P_t vanishes on (cone cocycle, D-coboundary) and on
//      (cone coboundary, D-cocycle): it DESCENDS to
//          H^t_c(K, R; M_n) x H^{2-t}_eq(K, R; D_n) -> F_ell .
//      The descent is not taken on trust: PairingCertificate items 2-3
//      re-derive it on the stored Z / B bases, and the constructor aborts if
//      it fails (that would be a defect of THIS unit, never of [C]).
//
//   4. THE BOUNDARY PAIRING (cl:pd(1), second pairing): the same cup form on
//      the circle model, H^s(K_bd; M_n) x H^{1-s}(K_bd; D_n) -> F_ell over
//      the 1-cycle d[K].  The D_n circle model is built here: the relation
//      list of GenerateSL2ZBoundaryRelationsElt with every matrix rho_n(g)
//      replaced by rho_n^vee(g) (g = T for the cusp relation, -I for the
//      FIX-negI ones; each replacement ENGINE_CHECKed against rho_n(g)).
//
//   5. THE GRAMS in the stored bases: G^t = H_c(t)^T Q_t H_D(2-t) and
//      G_bd^s = H_bd,M(s)^T Q_bd,s H_bd,D(1-s), perfect iff square of full
//      rank.  PERFECTNESS IS NOT ASSERTED HERE: it is [C, cl:pd](1), OPEN, and
//      the validation is the oracle that tests it; this object only records
//      it (perfect(t) / boundary_perfect(s)).
//
//   6. ADJOINTNESS ([C, cl:pd](2)) in the column convention of
//      [C, alg:assembly] on both sides (T[z_i] = sum_j A_{ji} [z_j]):
//          <A_c x_i, y_j> = (A_c^T G)_{ij},  <x_i, A_D y_j> = (G A_D)_{ij},
//      so the identity is  A_c^{(t)T} G^t == G^t A(D_n)^{(2-t)}, no
//      determinant factor (PairingAdjoint; `factor` is for the negative
//      controls only).
//
// COMPARISON THROUGH THE PAIRING, NOT THROUGH Phi_n.  At n >= ell the groups
// H^*(D_n) and H^*(M_n) have different dimensions (WP-1's datum: at (6, 5)
// H^0 is 1 vs 0 and H^1 is 2 vs 1; at (12, 5) Phi_n (x) F_5 = 0), so nothing
// here reads Phi_n: the pairing relates H^t_c(M_n) to H^{2-t}(D_n) directly.
//
// NOT HERE.  The Hecke action on the cone is HeckeConeAction's and on D_n is
// HeckeOperatorOnK's (FibreRepresentation::ContragredientN); both are read,
// neither is modified (the cone is read-only for this package).  The
// socle <-> Loewy identity is HeckeExt/HeckeIdentityCertificate item 7; the
// iota-sign is RECORDED by HeckeDualityValidation (arm A11).
//
// Golden objects (tier A pairing_perfect^t, pd_adjoint^t; tier B
// pairing_matrix^t) are DEFINED by PairingGoldenObjects and NEVER called
// from Golden/GoldenHarness.cpp (slot-6 owner ruling 3: no emission).
//
// Placement: Hecke/ (pipeline-level; it consumes EquivariantMappingCone and
// the contragredient cell), uint32_t only, no Matrix.h include, no raw mod_*
// call, no unordered_ container, ENGINE_CHECK for invariants.
// ===========================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "HeckeContragredient.h"
#include "SL2ZBoundaryPair.h"
#include "../EquivariantCohomologyOnK.h"
#include "../EquivariantMappingCone.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"

namespace hecke {

// ---- 1. the relative fundamental 2-chain ----------------------------------
struct ModelFundamentalChain {
    std::vector<int32_t> face_coeff;        // [f] over K's 2-simplices, eps_f in {+1, -1}
    std::vector<int32_t> boundary_coeff;    // [e] over K_bd's 1-simplices: the coefficient of e in d[K]
    std::vector<uint32_t> bd_simplex_in_K[2];   // K_bd q-simplex -> its index in K (q = 0, 1)
    uint32_t qualifying_sign_vectors = 0;   // sign vectors satisfying the cycle condition (2 expected)
    int32_t horocycle_coefficient = 0;      // the common coefficient chosen (-1, see the header)
};

// Builds and CERTIFIES the chain (throws std::runtime_error when the cycle
// condition has no solution, more than the two +-eps, or a seed relation is
// not order-preserving).  K must be the 7-vertex model and K_bd its boundary
// path (CreateSL2ZBoundaryPathTriangulation).
ModelFundamentalChain BuildModelFundamentalChain(const SimplicialComplex& K, const SimplicialComplex& K_bd);

// ---- 2. the cup-product evaluation form ------------------------------------
// The (N_a(X) d_V) x (N_b(X) d_V) matrix Q with
//   Q[(front_a(s), i)][(back_b(s), j)] += coeff[s] * fibre_gram[i][j]
// summed over the (a + b)-simplices s of X (increasing vertex order; front_a
// = the first a + 1 vertices, back_b = the last b + 1), so that
// x^T Q y = < x u y, sum_s coeff[s] s > on flat cochains (L1 layout).
FlatMatrix CupEvaluationForm(const SimplicialComplex& X, uint32_t a, uint32_t b,
                             const std::vector<int32_t>& coeff, const FlatMatrix& fibre_gram, uint32_t ell);

// The flat restriction C^q(K; V) -> C^q(K_bd; V) (identity fibre), found by
// vertex lists: (N_q(K_bd) d_V) x (N_q(K) d_V).
FlatMatrix RestrictionFlat(const SimplicialComplex& K, const SimplicialComplex& K_bd, uint32_t q,
                           uint32_t d_V, uint32_t ell);

// ---- adjointness -----------------------------------------------------------
// A_c^T G == factor * G A_D over F_ell (A_c: dim_c x dim_c, G: dim_c x dim_D,
// A_D: dim_D x dim_D; shapes ENGINE_CHECKed).  `nonvacuous_out`, when given,
// is set to whether G A_D != 0 (the negative controls need it).
bool PairingAdjoint(const MatrixT<uint32_t>& A_c, const FlatMatrix& G, const MatrixT<uint32_t>& A_D,
                    uint32_t factor, uint32_t ell, bool* nonvacuous_out = nullptr);

// ---- the certificate ---------------------------------------------------------
// A plain copyable struct holding the stored matrices; verify() RE-DERIVES
// every item from them alone:
//   1  the fibre Gram is the identity ([C, prop:pairing](2));
//   2  cone-level descent: Z_c(t)^T Q_t B_D(2-t) = 0 and B_c(t)^T Q_t Z_D(2-t) = 0, every t;
//   3  boundary descent: Z_bM(s)^T Qb_s B_bD(1-s) = 0 and B_bM(s)^T Qb_s Z_bD(1-s) = 0, every s;
//   4  the Grams recomputed: G^t == H_c(t)^T Q_t H_D(2-t), G_bd^s == H_bM(s)^T Qb_s H_bD(1-s);
//   5  record honesty: the recorded perfect flags == (square and full rank), recomputed.
// Perfectness itself is NOT an item (cl:pd is OPEN; the validation tests it).
struct PairingCertificate {
    uint32_t ell = 0, n_sym = 0, d_V = 0;
    FlatMatrix fibre_gram;
    std::vector<FlatMatrix> Q, Zc, Bc, Hc;        // [t] t = 0..2 (cone H_c degree)
    std::vector<FlatMatrix> ZD, BD, HD;           // [q] q = 0..2 (bulk D_n)
    std::vector<FlatMatrix> Qb, ZbM, BbM, HbM;    // [s] s = 0..1 (circle, M_n)
    std::vector<FlatMatrix> ZbD, BbD, HbD;        // [s] s = 0..1 (circle, D_n)
    std::vector<FlatMatrix> G, Gb;                // the Grams
    std::vector<char> perfect, boundary_perfect;  // recorded

    uint32_t verify() const;
    uint32_t verify(std::vector<std::string>& failures) const;
    std::vector<uint32_t> failing_items() const;
};

// ---- one (n, ell) cell -------------------------------------------------------
class PoincareDualityPairing {
public:
    // `cell` must outlive this object.  Builds the M_n pair and its cone, the
    // D_n circle model, the forms and the Grams; runs the certificate and
    // THROWS std::runtime_error if it fails (a failing descent is a defect of
    // this unit).  Perfectness is recorded, never thrown on.
    explicit PoincareDualityPairing(const ContragredientCell& cell);
    ~PoincareDualityPairing();
    PoincareDualityPairing(const PoincareDualityPairing&) = delete;
    PoincareDualityPairing& operator=(const PoincareDualityPairing&) = delete;

    uint32_t ell() const { return ell_; }
    uint32_t n_sym() const { return n_; }
    const ContragredientCell& cell() const { return *cell_; }
    const SL2ZBoundaryPair& pair_M() const { return *pair_M_; }
    const EquivariantMappingCone& cone() const { return *cone_; }
    const ModelFundamentalChain& chain() const { return chain_; }

    // The cone level, t = 0..2 (the degree of H^*_c).
    const FlatMatrix& cone_form(uint32_t t) const;       // flat_dim_c(t) x N_{2-t}(K) d_V
    const FlatMatrix& gram(uint32_t t) const;            // dim H^t_c x dim H^{2-t}(D_n)
    bool perfect(uint32_t t) const;
    // The circle level, s = 0..1.
    const FlatMatrix& boundary_form(uint32_t s) const;   // N_s(K_bd) d_V x N_{1-s}(K_bd) d_V
    const FlatMatrix& boundary_gram(uint32_t s) const;   // dim H^s(bd; M_n) x dim H^{1-s}(bd; D_n)
    bool boundary_perfect(uint32_t s) const;

    const PairingCertificate& certificate() const { return cert_; }

    // Evaluates P_t on one cone cochain and one bulk D-cochain (flat columns),
    // straight from the cup-product definition (not through cone_form) -- the
    // second route the validation compares cone_form against.
    uint32_t evaluate_direct(uint32_t t, const FlatMatrix& cone_cochain, const FlatMatrix& d_cochain) const;

private:
    const ContragredientCell* cell_;
    uint32_t ell_ = 0, n_ = 0, d_V_ = 0;
    std::unique_ptr<SL2ZBoundaryPair> pair_M_;
    std::unique_ptr<EquivariantMappingCone> cone_;
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels_bd_D_;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc_bd_D_;
    std::unique_ptr<EquivariantCohomologyOnK> coh_bd_D_;
    ModelFundamentalChain chain_;
    PairingCertificate cert_;
};

// ---- golden objects (DEFINED, never emitted) ---------------------------------
// tier A pairing_perfect^t (t = 0..2; scalar 0/1), pd_adjoint^t (the AND of
// the adjointness verdicts over the supplied primes, one per t; scalar 0/1);
// tier B pairing_matrix^t (G^t, hashed as Golden/GoldenHarness.cpp's
// HashFlatInto: rows, cols, entries row-major).
std::vector<DualityGoldenObject> PairingGoldenObjects(const PoincareDualityPairing& pd,
                                                      const std::vector<char>& pd_adjoint_per_t);

} // namespace hecke

#endif // HECKE_PAIRING_H
