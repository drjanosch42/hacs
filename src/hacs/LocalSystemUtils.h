// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef LOCAL_SYSTEM_UTILS_H
#define LOCAL_SYSTEM_UTILS_H

#include "SimplicialCohomology.h"
#include <vector>
#include <cstdint>
#include <string> // For std::string in exception
#include <stdexcept> // For std::runtime_error
#include <sstream>   // For std::ostringstream
#include <algorithm> // For std::sort, std::find
#include <iterator>  // For std::distance

// Helper to create canonical edge (sorted tuple)
inline std::vector<uint32_t> make_edge(uint32_t u, uint32_t v) {
    if (u < v) return { u, v };
    return { v, u };
}

// Helper to find index of a simplex in K's list
uint32_t find_simplex_idx(const SimplicialComplex& K, uint32_t q_dim, const std::vector<uint32_t>& simplex_vertices);

// ===========================================================================
// hecke-extensions slot 6, WP-1 (phase E7a) -- THE FIBRE REPRESENTATION.
//
// [C] = docs/latex/gl2_contract.tex, cited by label.
//
// The pipeline's coefficient system has, since Stage S2, exactly one fibre
// builder: rho_n(A) = ComputeSymmetricPowerMatrixFM(A mod ell, n)
// ([C, def:rhon], [C, eq:rhondef]).  E7 adds the CONTRAGREDIENT
//
//     rho_n^vee(g) := rho_n(adj g)^T      ([C, def:contragredient]),
//
// the action on D_n = Hom_Z(Sym^n Z^2, Z) in the dual basis (u_0^vee, ...,
// u_n^vee).  It is a monoid homomorphism on ALL of Mat_2 (singular matrices
// included -- adj is an anti-automorphism of the monoid, [C, eq:adjrules]),
// and rho_n^vee(gamma) = rho_n(gamma^{-1})^T on Gamma ([C, prop:pairing](1)).
//
// NOTE the determinant: since adj g = det(g) g^{-1}, rho_n^vee(g) =
// det(g)^n rho_n(g^{-1})^T, i.e. [C]'s D_n is the ABSTRACT contragredient
// tensored with det^n.  That is what makes [C, lem:symdual]'s
// Phi_n rho_n(g) = rho_n^vee(g) Phi_n hold with NO determinant factor.
//
// FibreRepresentation names the builder.  It is selected by the OBJECT that
// carries the coefficient system (the relation list of the local system, and
// the HeckeOperatorOnK built on it) -- there is no global switch.  SymN is
// the default everywhere and is EXACTLY the pre-slot-6 call.
// ===========================================================================
enum class FibreRepresentation : uint32_t {
    SymN = 0,             // rho_n(A)          = ComputeSymmetricPowerMatrixFM(A, n)
    ContragredientN = 1,  // rho_n^vee(A)      = rho_n(adj A)^T
    QuotientL = 2,        // rho_{Q_n}(A)      = pi . rho_{n+ell+1}(A) . E   (slot-11 WP-5, below)
    // (slot-13 WP-5) Hida's lattice L_n ([C] rmk:stabconvention(3), def:hidalattice
    // variant (H)): rho^H_n(A) = rho_n(A^T)^T = D^{-1} rho_n(A) D, D = diag(binom(n, j)).
    // Integral and multiplicative on Mat_2(Z); honoured by the MODULE route of
    // HeckeOperatorOnK (Z/ell^k, k >= 2) only -- the field route refuses it.
    HidaLatticeH = 3,     // rho^H_n(A)        = rho_n(A^T)^T
};

// "sym" / "contragredient" (for reports; never part of a golden text).
const char* FibreRepresentationName(FibreRepresentation rep);

// adj A = [[d, -b], [-c, a]] of a 2x2 matrix over F_p (reduction commutes
// with adj, so this is (adj A_Z) mod p for any integral lift A_Z).
template<class Elt>
FlatMatrixT<Elt> AdjugateMod2x2(const FlatMatrixT<Elt>& A);

// rho_n^vee(A) = rho_n(adj A)^T over F_p, [C, def:contragredient].
template<class Elt>
FlatMatrixT<Elt> ComputeContragredientPowerMatrixFM(const FlatMatrixT<Elt>& A, uint32_t n_sym);

// The fibre builder, dispatched on the representation.  SymN returns
// ComputeSymmetricPowerMatrixFM(A, n_sym) itself (the same call, not a copy
// of its logic), so every SymN consumer is byte-identical by construction.
template<class Elt>
FlatMatrixT<Elt> ComputeFibreMatrixFM(FibreRepresentation rep, const FlatMatrixT<Elt>& A, uint32_t n_sym);

// ===========================================================================
// hecke-extensions slot 11, WP-5 -- THE QUOTIENT FIBRE Q_n = Sym^{n+ell+1} / L . Sym^n.
//
// Over F_ell (ell = the modulus of A), with m := n + ell + 1 and the Dickson
// multiplication L. : Sym^n -> Sym^m of HeckeExt/HeckeDicksonMultiplication.h
// (L = x^ell y - x y^ell; column j = u_{j+1} - u_{j+ell} in the basis
// u_k = x^{m-k} y^k of ComputeSymmetricPowerMatrixFM).
//
// THE COMPLEMENT BASIS ([C, def:Qn](1); one definition, QuotientLComplementRows): the ell + 1
// monomials u_0, ..., u_{ell-1}, u_m of Sym^m.  The column u_{j+ell} of L.
// carries its last non-zero entry at row j + ell (j = 0..n), so the rows
// ell..m-1 are the pivot rows of L. and the listed rows complement them.
// The quotient map pi (QuotientLReduce) is the corresponding normal form:
// for r = m-1 down to ell, add v[r] * (column r - ell of L.) -- i.e.
// v[r - ell + 1] += v[r], v[r] = 0 -- and read the complement rows.
//
// rho_{Q_n}(A) := pi . rho_m(A) . E  for EVERY 2x2 A over F_ell (singular
// included), E the inclusion of the complement monomials; the matrix is
// (ell+1) x (ell+1) and its row / column t is the t-th complement row.
// It is well defined because rho_m(A) L. = det(A) L. rho_n(A) for every A
// ([C, lem:dicksontwist]); HeckeExt/HeckeDicksonTheta re-checks it on samples.
// n_sym is the n of Q_n (NOT d_V - 1: d_V = ell + 1).
// ===========================================================================

// The complement rows, ascending: {0, ..., ell-1, n+ell+1}.
std::vector<uint32_t> QuotientLComplementRows(uint32_t n_sym, uint32_t ell);

// pi on one coefficient vector of Sym^{n+ell+1} (length n+ell+2, entries in
// [0, ell)): reduces v IN PLACE to its normal form (every pivot row zero) and
// returns the ell + 1 complement coordinates.
template<class Elt>
std::vector<Elt> QuotientLReduce(std::vector<Elt>& v, uint32_t n_sym, Elt ell);

// pi as an (ell+1) x (n+ell+2) matrix over F_ell.
template<class Elt>
FlatMatrixT<Elt> QuotientLProjectionMatrixFM(uint32_t n_sym, Elt ell);

// rho_{Q_n}(A) over F_ell, ell = A.get_p() (>= 2), (ell+1) x (ell+1).
template<class Elt>
FlatMatrixT<Elt> ComputeQuotientLMatrixFM(const FlatMatrixT<Elt>& A, uint32_t n_sym);

// The fibre dimension of a representation: n+1 for SymN / ContragredientN,
// ell+1 for QuotientL.
uint32_t FibreDimension(FibreRepresentation rep, uint32_t n_sym, uint64_t ell);

#endif // LOCAL_SYSTEM_UTILS_H
