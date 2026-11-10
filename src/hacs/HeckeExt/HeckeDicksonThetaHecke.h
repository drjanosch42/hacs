// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_DICKSON_THETA_HECKE_H
#define HECKE_EXT_DICKSON_THETA_HECKE_H

// ===========================================================================
// hecke-extensions slot 11, WP-6 -- THE DICKSON / THETA LINE (2): the Hecke
// operators on the six terms of WP-5's sequence
//
//     0 -> H^0_n -theta-> H^0_m -pi-> H^0_Q -delta-> H^1_n -theta-> H^1_m -pi-> H^1_Q -> 0,
//
// m = n + ell + 1, over F_ell, ell >= 5 (HeckeDicksonTheta.h), and what the
// contract says about them.  Plan: docs/implementation/
// hecke_extensions_slot11_plan.md "### WP-6"; the spec is [E] sec:thetaobj
// and (6e)-(6g) of sec:identities (v1.12.0-pre.1); the statements are
// [C, prop:thetatwist], [C, thm:thetales](2),(3),(5), [C, prop:heckeH0],
// [C, lem:Qnfunctions](3), [C, lem:heckefibre] (v1.11.0-pre.1).
//
// RunThetaHecke(cell, opts) computes, for every prime p of opts.primes, the
// three operators A^{(q)}(T_p) (p = ell: U_ell) on Sym^n, Sym^m and Q_n
// through DicksonThetaCell::hecke_triple, and CHECKS (each a named check; a
// failure is reported with its detail, never repaired):
//
//   (T)   the coset representatives s_i and every chain tau(i, q, sigma) of
//         the tables built at n and at m agree entrywise -- the datum is not
//         keyed on n ([C, lem:heckefibre]: "one datum serves every fibre");
//   (F)   (opts.cochain_arm) on FLAT cochains, block by block:
//           Theta_m (1 (x) L) = det(g) (1 (x) L) Theta_n,
//           Theta_Q (1 (x) pi) = (1 (x) pi) Theta_m,          q = 0, 1, 2,
//         det(g) = p mod ell (0 for U_ell) -- [C, prop:thetatwist] at cochain
//         level, which needs (T);
//   (E)   p != ell, q = 0, 1: A_m theta = p theta A_n entrywise, the untwisted
//         identity FAILS wherever theta A_n != 0 (and p != 1 mod ell),
//         pi A_m = A_Q pi, delta A_Q = p A_n delta ([C, thm:thetales](2));
//   (U)   p = ell: A_m(U) theta = 0, pi A_m(U) = A_Q(U) pi, delta A_Q(U) = 0;
//   (H0)  the classes of L^a D^b: A(T_p) = (1 + p) p^a on Sym^n and Sym^m;
//         U_ell on H^0(Sym^d) (d = n, m, d >= 1): rank <= 1, U^2 = 0, and U = 0
//         unless (d, ell) = (120, 5) ([C, prop:heckeH0](2)-(4); every degree
//         this unit touches is <= 214, where (120, 5) is the only exception:
//         a non-zero U needs a pure D^b, b in {ell+1, ...}, i.e. d = 20 b with
//         b in {6, 11, 12, ...} at ell = 5 and d >= 336 at ell >= 7); at
//         (120, 5): U D^6 = -L^20, U L^20 = U L^10 D^3 = 0; at d = 0, U = 1;
//         H^0(Q_n): T_p = p + 1, U = 0 ([C, lem:Qnfunctions](3));
//   (K)   rank delta = 1: A_n(T_p) delta(1_n) = (1 + p^{-1}) delta(1_n)
//         ([C, thm:thetales](3));
//   (B)   (opts.block_arm) the primary decomposition of the anemic family
//         {T_p : p in opts.primes, p != ell} on H^1_n and H^1_m: theta of every
//         block B of H^1_n lies in ONE block C of H^1_m, dim C >= dim B -
//         dim(B cap ker theta), and B cap ker theta != 0 only for the block of
//         the system (1 + p^{-1})_p ([C, thm:thetales](5)).
//
// Nothing here is a record key, a settings key or golden ([E] sec:thetaobj
// (g)); F_ell only (DicksonThetaCell refuses the rest).
// ===========================================================================

#include "HeckeDicksonTheta.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hecke_ext {

// MatrixT -> ThetaMat (same entries, same modulus).
ThetaMat ThetaFromMatrix(const MatrixT<uint32_t>& M);
ThetaMat ThetaScaled(const ThetaMat& A, uint32_t c);
bool ThetaEqual(const ThetaMat& A, const ThetaMat& B);
// Column basis of  cap_i ker (ops[i] - eig[i] I)^D,  D = the common dimension.
ThetaMat ThetaGeneralisedEigenspace(const std::vector<ThetaMat>& ops, const std::vector<uint32_t>& eig);
// dim(colspan A cap colspan B).
uint32_t ThetaIntersectionDim(const ThetaMat& A, const ThetaMat& B);

struct ThetaHeckeCheck {
    std::string name;
    bool ok = false;
    std::string detail;
};

struct ThetaHeckeOptions {
    std::vector<uint64_t> primes;   // ascending; ell among them runs U_ell
    bool cochain_arm = false;       // (F): flat Theta identities (O(N^2 d^3))
    bool tau_arm = true;            // (T)
    bool block_arm = true;          // (B)
};

// One primary block of the anemic family on H^1 (basis in the stored H^1
// coordinates of its term).
struct ThetaBlock {
    ThetaMat basis;                 // d x dim
    uint32_t dim = 0, residue_degree = 0;
};

struct ThetaHeckeRun {
    uint32_t n = 0, ell = 0, m = 0;
    std::vector<uint64_t> primes;
    // A[p][term][q], term = ThetaTerm index, q = 0, 1, 2.
    std::map<uint64_t, std::array<std::array<ThetaMat, 3>, 3>> A;
    std::vector<ThetaHeckeCheck> checks;
    std::vector<ThetaBlock> blocks_n, blocks_m;   // (B), anemic family, H^1
    // Observations.
    uint32_t u_rank_h0_n = 0, u_rank_h0_m = 0;    // rank U_ell on H^0 (when U ran)
    uint32_t rank_delta = 0, rank_theta1 = 0;
    uint32_t tau_slots_compared = 0;

    const ThetaMat& op(ThetaTerm t, uint64_t p, uint32_t q) const;
    uint32_t failures() const;
    std::string first_failure() const;
};

ThetaHeckeRun RunThetaHecke(const DicksonThetaCell& cell, const ThetaHeckeOptions& opts);

// The default prime set of the unit: {2, 3, 5, 7} \ {ell}, plus ell (U_ell),
// ascending -- the `operators_computed` list of every published ext_first
// record at ell <= 13.
std::vector<uint64_t> ThetaDefaultPrimes(uint32_t ell);

} // namespace hecke_ext

#endif // HECKE_EXT_DICKSON_THETA_HECKE_H
