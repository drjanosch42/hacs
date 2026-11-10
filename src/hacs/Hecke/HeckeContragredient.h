// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_CONTRAGREDIENT_H
#define HECKE_CONTRAGREDIENT_H

// ===========================================================================
// hecke-extensions slot 6, WP-1 -- phase E7a: THE CONTRAGREDIENT SIDE.
//
// [C] = docs/latex/gl2_contract.tex (cited by label); [E] =
// docs/latex/hecke_extensions.tex; [X] = the extension implementation plan,
// Sec. 6.2 E7 (first half of its scope).
//
// What this unit provides, each a base change of a Z-datum ([C,
// sec:contragredient], architecture rule R2):
//
//   * rho_n^vee(g) = rho_n(adj g)^T         ComputeContragredientPowerMatrixFM
//                                           (LocalSystemUtils.h), on Mat22::adj
//                                           (HeckeCore);
//   * the contragredient local system D_n   GenerateSL2ZContragredientRelations-
//                                           WithGroupData (LocalSystemSL2Z.h);
//   * Phi_n : V -> D_n of [C, lem:symdual]  PhiNMatrix / PhiNFiberBlockTable --
//                                           antidiagonal, Phi_n(u_t) =
//                                           (-1)^t t! (n-t)! u_{n-t}^vee, a
//                                           morphism of local systems for EVERY
//                                           n, an isomorphism over F_ell iff
//                                           n < ell;
//   * the evaluation pairing of             PairingGram (== I) / EvaluationPairing
//     [C, prop:pairing];
//   * H^*(D_n) and A(D_n)(T_p)^q            ContragredientCell: the SHIPPED
//                                           pipeline (EquivariantCohomologyOnK,
//                                           HeckeOperatorOnK with
//                                           FibreRepresentation::ContragredientN)
//                                           on D_n, next to the same objects on
//                                           M_n = Sym^n, plus the induced map
//                                           P = H^*(Phi_n) : H^*(M_n) -> H^*(D_n).
//
// THE EXPONENT ([C, lem:symdual], PROVED; slot-6 audit finding B-1).  Theta_g
// carries exactly ONE fibre matrix per term ([C, eq:xiexplicit] with the omega
// row: each block is n_k rho(s_i^{-1}) rho(g gamma_k) = n_k rho(delta_i gamma_k)),
// and the D_n operator is the same walk with rho_n^vee in place of rho_n.
// Since Phi_n rho_n(h) = rho_n^vee(h) Phi_n for EVERY h in Mat_2(Z), with no
// determinant factor, block by block
//
//     Theta^{D}_g (1 (x) Phi_n) = (1 (x) Phi_n) Theta^{M}_g     on C^q(K; -),
//
// hence on cohomology  A(D_n)(T_p)^q P = P A(M_n)(T_p)^q  -- EXPONENT 0.  The
// p^{+n} / p^{-n} readings (the abstract contragredient rho(g^{-1})^T =
// det(g)^{-n} rho_n^vee(g), scaled to be integral) are negative controls that
// must FAIL wherever p^n != 1 mod ell and P A(M_n) != 0.
//
// NOT here: the cohomological pairings, perfectness and adjointness (WP-3),
// and [E] cl:glover's convention hazard, which concerns the ABSTRACT
// contragredient and bites in identity 6 (WP-9).
//
// Golden objects (tier A dim_H_Dn^q, charpoly_A_Dn(T_p)^q) are DEFINED by
// ContragredientGoldenObjects and NEVER called from Golden/GoldenHarness.cpp
// (slot-6 owner ruling 3: no emission in slot 6).
// ===========================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../LocalSystemUtils.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"

namespace hecke {

// Phi_n over F_ell, [C, lem:symdual]: the (n+1) x (n+1) matrix whose column t
// holds Phi_n(u_t) = (-1)^t t! (n-t)! u_{n-t}^vee in the dual basis, i.e. the
// transpose of the Gram matrix [C, eq:BnGram] of B_n.  Antidiagonal.
FlatMatrix PhiNMatrix(uint32_t n_sym, uint32_t ell);

// [C, lem:symdual]: Phi_n (x) F_ell is an isomorphism iff n! is a unit iff
// n < ell.  (The code's own statement; the validation compares it with the
// rank of PhiNMatrix.)
bool PhiNIsIsomorphism(uint32_t n_sym, uint32_t ell);

// Phi_n as the FiberBlockTable morphism M_n -> D_n on K = K' (one block per
// simplex, source simplex = target simplex, F = Phi_n) -- the S6 general
// constructor of EquivariantCochainComplexMorphismOnK, with the domain (the
// TARGET of the induced map) the D_n cohomology.
FiberBlockTable PhiNFiberBlockTable(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell);

// The Gram matrix of the evaluation pairing <v, phi> = phi(v) in the bases
// (u_j), (u_j^vee): the identity ([C, prop:pairing](2)).
FlatMatrix PairingGram(uint32_t n_sym, uint32_t ell);

// <v, phi> = sum_j v_j phi_j (v, phi given as (n+1) x 1 columns over F_ell).
uint32_t EvaluationPairing(const FlatMatrix& v, const FlatMatrix& phi);

// Whether A_D * P == factor * (P * A_M) exactly over F_ell (column
// convention of [C, alg:assembly] on both sides; P : H(M) -> H(D)).  Shapes
// must agree (ENGINE_CHECKed).  `nonvacuous_out`, when given, is set to
// whether P * A_M != 0 (the negative controls need it).
bool HeckeIntertwines(const MatrixT<uint32_t>& A_D, const FlatMatrix& P, const MatrixT<uint32_t>& A_M,
                      uint32_t factor, uint32_t ell, bool* nonvacuous_out = nullptr);

// One (n, ell) cell of E7a: M_n and D_n on the 7-vertex model, their
// cohomology, and the induced map P = H^*(Phi_n).  Heap-held members, so the
// raw pointers between them stay valid for the object's lifetime.
class ContragredientCell {
public:
    ContragredientCell(uint32_t n_sym, uint32_t ell);

    uint32_t n_sym() const { return n_; }
    uint32_t ell() const { return ell_; }
    const SimplicialComplex& K() const { return *K_; }
    const hecke::SL2ZTessellationTables& model() const { return *model_; }
    const std::vector<LocalSystemSimplicialComplex::SimplexRelation>& relations_D() const { return rels_D_; }
    const EquivariantCohomologyOnK& coh_M() const { return *coh_M_; }
    const EquivariantCohomologyOnK& coh_D() const { return *coh_D_; }
    const FlatMatrix& Phi() const { return Phi_; }
    // P^q = H^q(Phi_n): H^q(M_n) -> H^q(D_n), dim_H(D)^q x dim_H(M)^q.
    const std::vector<FlatMatrix>& induced_Phi() const;

    // The Hecke operator for g = diag(1, p) in one representation, computed.
    // Owns its coset / tau tables (coefficient-free integral data, shared by
    // the two representations of the same p when built through this cell).
    struct HeckePair {
        uint32_t p = 0;
        std::unique_ptr<HeckeCosetTables> cosets;
        std::unique_ptr<HeckeTauTable> tau;
        std::unique_ptr<HeckeOperatorOnK> op_M;   // FibreRepresentation::SymN
        std::unique_ptr<HeckeOperatorOnK> op_D;   // FibreRepresentation::ContragredientN
    };
    HeckePair hecke_pair(uint32_t p) const;

private:
    uint32_t n_ = 0, ell_ = 0;
    std::unique_ptr<SimplicialComplex> K_;
    std::unique_ptr<hecke::SL2ZTessellationTables> model_;
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels_M_, rels_D_;
    std::vector<Mat22> g_r_;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc_M_, lssc_D_;
    std::unique_ptr<EquivariantCohomologyOnK> coh_M_, coh_D_;
    FlatMatrix Phi_;
    std::unique_ptr<EquivariantCochainComplexMorphismOnK> morph_;
};

// The E7a golden objects ([X] Sec. 6.2 E7; slot-6 plan Sec. 4.4), hashed
// exactly as Golden/GoldenHarness.cpp's collectors hash (FNV-1a-64 scalar /
// poly), DEFINED here and never called from the harness:
//   tier A  dim_H_Dn^q                   (one per degree)
//   tier A  charpoly_A_Dn(T_<p>)^q       (one per degree and Hecke prime)
struct DualityGoldenObject {
    std::string object;
    char tier = 'A';
    uint64_t rows = 0, cols = 0;
    uint64_t hash = 0;
};
struct DualityHeckeInput {
    uint32_t p = 0;
    const std::vector<MatrixT<uint32_t>>* A_D = nullptr;   // A(D_n)(T_p)^q, q = 0, 1, 2
};
std::vector<DualityGoldenObject> ContragredientGoldenObjects(const EquivariantCohomologyOnK& coh_D,
                                                            const std::vector<DualityHeckeInput>& heckes);

} // namespace hecke

#endif // HECKE_CONTRAGREDIENT_H
