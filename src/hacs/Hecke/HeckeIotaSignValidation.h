// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_IOTA_SIGN_VALIDATION_H
#define HECKE_IOTA_SIGN_VALIDATION_H

#include <cstdint>
#include <vector>

#include "ExactInt.h"
#include "../../ModArithmetic/FlatMatrix.h"

class SimplicialComplex;

// Obligation obl:iota(ii), (iii) and (v) of docs/latex/gl2_contract.tex
// chapter 9 (the involution iota), v1.2.0 -- the clauses that need iota^* as a
// map of cochain complexes, which HeckeIotaTauSymmetryValidation (clauses (i),
// (iv)) deliberately did not build. This file builds it.
//
// THE SIGN (owner decision OQ3, 2026-09-16; docs/implementation/oq3_resolution.md).
// def:iota(iv) defines iota^* := -iota^*_geo, where iota^*_geo is the geometric
// pullback (iota^*_geo f)(sigma) = rho_n(iota) f(iota_K sigma): "+" is BY
// DEFINITION the iota^*-eigenspace containing the Eisenstein classes, and the
// record field "iota_sign = eisenstein_in_H1_plus" names that normalisation
// (a third party reproduces a record by iota^* = -iota^*_geo). The v1.1.1 run
// of this validation had found the Eisenstein class in H^1_{eq,-} of iota^*_geo
// on all 74 signed cells of the grid (oq3_iota_sign_verification.md); the
// owner resolved it by negating the definition, not by a flip at a call site.
// def:iota still leaves NO sign free (the two lifts iota and -iota of
// z -> -conj(z) give rho_n(-iota) = (-1)^n rho_n(iota), identical for even n,
// and H^* = 0 for odd n, ell != 2), so "Eisenstein in H^1_{eq,+}" is now a
// DEFINITION-CONSISTENCY oracle and GATING: a '-' on any cell would mean the
// geometric sign is not uniform across cells, i.e. the definition is ill-posed.
// Both involutions are built and iota_C == -iota_C_geo is asserted entrywise;
// the geometric sign is RECORDED next to the normalised one.
//
// ---------------------------------------------------------------------------
// WHAT IS BUILT (def:iota, all four clauses, in the flat coordinates
// eq:flatindex, s*d_V + j)
// ---------------------------------------------------------------------------
//   iota      = diag(-1, 1)                                     def:iota(i)
//   rho_n(iota) = ComputeSymmetricPowerMatrixFM(iota mod ell, n) -- the LIVE
//               rho_n of the pipeline (LocalSystemSL2Z derives every relation
//               matrix through the same call), checked against the closed
//               form diag((-1)^{n-j})_{0<=j<=n} of eq:rhoniota  def:iota(ii)
//   iota_K    = vertex permutation 0<->3, 1<->2, 4<->5, 6 fixed, RE-DERIVED
//               from the shipped exact vertex coordinates (z -> -conj(z)),
//               with the orientation sign eps_sigma of every simplex
//               recomputed as the parity of the sort back into increasing
//               vertex order and compared with lem:iotamodel's table
//                                                               def:iota(iii)
//   iota_C_geo^(q) = the (N_q d_V)^2 signed block permutation with block
//               eps_sigma * rho_n(iota) at block position (sigma, |iota_K sigma|)
//               -- i.e. (iota^*_geo f)(sigma) = rho_n(iota) f(iota_K sigma), the
//               relation matrices acting on COLUMN vectors exactly as in
//               f(s_lhs) = M f(s_rhs)                    def:iota(iv), iota^*_geo
//   iota_C^(q) = -iota_C_geo^(q), i.e. block -eps_sigma * rho_n(iota): the
//               normalised involution iota^* := -iota^*_geo   def:iota(iv), v1.2.0
//   iota^*    = P_q * iota_C^(q) * H_eq^(q)  in the stored basis of H^q_eq,
//               column convention iota^*[z_i] = sum_j (iota^*)_{ji} [z_j]
//               (the convention of A^{(q)}, alg:assembly); the geometric
//               P_q * iota_C_geo^(q) * H_eq^(q) == -iota^* is asserted.
//
// ---------------------------------------------------------------------------
// GATING (any failure is counted and returned)
// ---------------------------------------------------------------------------
//  G1  obl:iota(i)   iota_K is lem:iotamodel's permutation with its sign table.
//  G2  eq:rhoniota   rho_n(iota) == diag((-1)^{n-j}); rho_n(-iota) == (-1)^n rho_n(iota).
//  G3  obl:iota(ii)  every relation row (lhs, rhs, gamma) of the model, central
//                    -I rows included, maps under (iota_K, c_iota) to the INVERSE
//                    of a row of the list, with equal orientation signs on both
//                    sides, and rho_n(iota) rho_n(gamma) == rho_n(iota gamma iota) rho_n(iota).
//  G4  obl:iota(iii) iota_C^2 == I; iota_C d == d iota_C (d = d_1 (x) I from the
//                    base K's coboundary) as full matrices in every degree;
//                    A_rel * iota_C * C_eq == 0 ENTRYWISE on all 31 relations
//                    (never assumed from G3); iota_C Z_eq, iota_C B_eq lie in
//                    their own spans (rank test).
//  G5  prop:iotacommutes(2) (iota^*)^2 == I on H^q_eq, and (2 in R^x)
//                    dim H^q_+ + dim H^q_- == dim H^q.
//  G6  obl:iota(v)   iota^* A^{(1)}(T_p) == A^{(1)}(T_p) iota^* for p in {2, 3}
//                    (and 5 where used), 6 in R^x throughout the grid; and the
//                    degree-0 anchor: at n = 0, H^0(Gamma; F_ell) = F_ell and
//                    iota^* = -1 on it (v1.2.0; geometric +1). The anchor is what
//                    pins the global sign -- iota_C_geo = -iota_C passes G1-G5, G7-G9
//                    as well.
//  G7  route (a)     the joint T_p-eigenspace of eigenvalue 1 + p^{n+1} (p in
//                    {2,3}, then 5 if needed), WHERE 1-dimensional, is an
//                    iota^*-eigenline; its eigenvalue eps_a in {+1, -1}; and
//                    (v1.2.0, GATING) eps_a == +1: the Eisenstein class lies in
//                    H^1_{eq,+}.
//  G8  route (b)     res: H^1_eq(Gamma) -> H^1(boundary) through the shipped
//                    restriction morphism to the cusp path v0-v1-v2-v3 with
//                    f(v3) = rho_n(T) f(v0); iota_C is built on that complex by
//                    the SAME code (0<->3, 1<->2, all three edges sign -1, the
//                    same global sign), and res iota^* == iota^*_bdry res is
//                    asserted (no hypothesis on ell: the path is iota-stable and
//                    iota normalises Gamma_inf). WHERE rank(res) == 1,
//                    iota^*_bdry acts on im(res) by a scalar eps_b in {+1, -1},
//                    res iota^* == eps_b res, and (v1.2.0, GATING) eps_b == +1.
//  G9  agreement     where both routes are defined, res(v_a) != 0 (the
//                    Eisenstein T_p-eigenline is boundary-visible) and eps_a == eps_b.
//  G10 consistency   every defined sign (eps_a or eps_b) on the whole grid is
//                    the SAME sign, and (v1.2.0) that sign is '+'. A '-' anywhere
//                    would mean the geometric sign is not uniform across cells,
//                    i.e. def:iota(iv)'s "+" is ill-posed -- a defect of the
//                    definition, not of a call site.
//  G11 lift          rho_n(-iota) == rho_n(iota) hence identical iota_C for
//                    every even-n cell (the (-1)^n of rmk:iotasign); and
//                    iota_C == -iota_C_geo entrywise in every degree, on H^q too.
//
// ---------------------------------------------------------------------------
// RECORDED (printed, never a failure -- obl:iota(v) second half, rmk:iotasign)
// ---------------------------------------------------------------------------
//  * the GEOMETRIC sign, i.e. the eigenspace of iota^*_geo = -iota^* containing
//    the Eisenstein class (minus, by the v1.1.1 run; the negation of the
//    gated normalised sign by construction), and the geometric H^0 anchor;
//  * dim H^1_+ / dim H^1_- per cell against the predicted split
//    (d_{n+2}, d_{n+2} - 1), with the Dickson locus of rmk:dickson flagged --
//    recorded because, given the gated Eisenstein sign, it is the even split
//    of the cuspidal part, the mod-ell Eichler-Shimura prediction obl:iota
//    tags CONJ;
//  * the cells where route (a) or (b) is undefined and why (eigenspace not
//    1-dimensional -- which happens for small ell by Fermat, since
//    1 + p^{n+1} == 1 + p^{n+1-(ell-1)} mod ell; rank(res) != 1).
//
// Grid: n in {2,4,6,8,10,12,14,16,20,24,30},
//       ell in {5,7,11,13,101,134217757,2^31-1}   (77 cells; all 6 in R^x).
// Prints one machine-readable line
//     IOTA_SIGN eisenstein=<plus|minus|mixed|undetermined> (geometric=<...>) cells=<k> ...
//
// Returns the number of GATING failures (0 = pass), per the ValidationFn
// convention.

// ---------------------------------------------------------------------------
// THE ORACLE, EXPORTED (hecke-extensions slot 4, WP-3 / phase E3).
// ---------------------------------------------------------------------------
// WP-3 builds a PRODUCTION iota^* (Hecke/HeckeInvolution) and must compare it
// ENTRYWISE against the involution this validation builds from def:iota alone.
// The builders below were the anonymous-namespace helpers of
// HeckeIotaSignValidation.cpp; slot 4 moves them, BODY UNCHANGED, into the
// named namespace hecke::iota_oracle so that HeckeInvolutionValidation can
// call them.  Nothing else changed: HeckeIotaSignValidation's own printed
// lines are byte-identical before and after the move (standing gate, slot-4
// plan audit F7/F8).
//
// INDEPENDENCE DISCIPLINE (slot-4 plan, WP-3 scope 5).  These are the ORACLE.
// Hecke/HeckeInvolution -- the production object -- must NEVER call them: it
// re-derives iota_K, the sign table and rho_n(iota) from def:iota itself, and
// the two constructions are compared.  A production call here would make the
// comparison vacuous.  Only *Validation translation units may include this
// header for the oracle.
namespace hecke {
namespace iota_oracle {

// iota = diag(-1, 1) in GL_2(Z)                                  def:iota(i)
Mat22 iota_mat();

// c_iota(gamma) = iota gamma iota^{-1} = iota gamma iota, exactly.
Mat22 c_iota(const Mat22& m);

// The LIVE rho_n of the pipeline: ComputeSymmetricPowerMatrixFM(g mod ell, n).
FlatMatrix rho_n(const Mat22& g, uint32_t n, uint32_t ell);

// eq:rhoniota, the closed form diag((-1)^{n-j})_{0<=j<=n}.
FlatMatrix rho_iota_closed_form(uint32_t n, uint32_t ell);

// iota_K on a stored complex (def:iota(iii)): the vertex permutation, and per
// degree the image index |iota_K sigma| and the orientation sign eps_sigma.
struct IotaOnComplex {
    std::vector<uint32_t> vperm;                  // vertex -> image vertex
    std::vector<std::vector<uint32_t>> image_idx; // [q][s] -> |iota_K sigma|
    std::vector<std::vector<int>> image_sign;     // [q][s] -> eps_sigma
};

// Index of the simplex with the given (unsorted) vertex set; throws if the
// image is not a stored simplex, i.e. if hypothesis (I1) fails.
uint32_t find_simplex_index(const std::vector<std::vector<std::vector<uint32_t>>>& simplices,
                            uint32_t q, std::vector<uint32_t> verts);

// Parity of the permutation sorting img back into increasing vertex order.
int sort_parity(const std::vector<uint32_t>& img);

IotaOnComplex build_iota(const SimplicialComplex& K, const std::vector<uint32_t>& vperm);

// lem:iotamodel's printed sign table, by VERTEX SET (the six minus signs).
const std::vector<std::vector<uint32_t>>& minus_sign_simplices_7vertex();

// iota_C^{(q)} in the flat coordinates eq:flatindex: the block
// (global_sign * eps_sigma) * rho at block position (sigma, |iota_K sigma|).
//   global_sign = +1 -> the GEOMETRIC pullback iota^*_geo;
//   global_sign = -1 -> def:iota(iv) since v1.2.0, iota^* := -iota^*_geo.
FlatMatrix iota_flat(const IotaOnComplex& io, uint32_t q, const FlatMatrix& rho,
                     uint32_t ell, int global_sign);

// D (x) I_{d_V} for the simplex-major flat layout.
FlatMatrix kron_identity(const FlatMatrix& D, uint32_t d_V, uint32_t ell);

// A list of N_q x d_V cochains as one dim_q x k multi-vector, column t =
// cochain t flattened to s*d_V + j.
FlatMatrix multivector(const std::vector<FlatMatrix>& basis, uint32_t dim_q, uint32_t ell);

} // namespace iota_oracle
} // namespace hecke

uint32_t HeckeIotaSignValidation();

#endif // HECKE_IOTA_SIGN_VALIDATION_H
