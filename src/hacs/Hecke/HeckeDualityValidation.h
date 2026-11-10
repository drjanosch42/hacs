// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_DUALITY_VALIDATION_H
#define HECKE_DUALITY_VALIDATION_H

// hecke-extensions slot 6, WP-1 (phase E7a) -- HeckeDualityValidation
// (tier `standard`).  [C] = docs/latex/gl2_contract.tex, cited by label.
//
// Arms (0 = success; the count of failed checks otherwise):
//   A1  Mat22::adj satisfies [C, eq:adjrules] on seeded matrices (det 1, p,
//       0, negative), and adj gamma = gamma^{-1} on SL(2,Z).
//   A2  rho_n^vee(g) = rho_n(adj g)^T ([C, def:contragredient]) against an
//       EXACT integer rho_n (the closed form [C, eq:rhondef], int64 with
//       overflow checks) reduced mod ell; rho_n^vee a monoid homomorphism
//       ([C, prop:pairing](1)) exactly over Z and mod ell on singular g;
//       rho_n^vee(gamma) = rho_n(gamma^{-1})^T on SL(2,Z); the SymN fibre
//       builder IS ComputeSymmetricPowerMatrixFM.
//   A3  the evaluation pairing over Z ([C, prop:pairing](2)-(4)): Gram = I;
//       <rho_n(g) v, phi> = <v, rho_n^vee(adj g) phi> with NO determinant
//       factor; the det^n factor of the twisted invariance (4), as a contrast.
//   A4  Phi_n ([C, lem:symdual]): B_n's Gram from the PERMUTATION definition
//       equals [C, eq:BnGram] (n <= 6); both equivariances and
//       Phi_n rho_n(g) = rho_n^vee(g) Phi_n exactly over Z; PhiNMatrix equals
//       Gram^T mod ell; the intertwining mod ell for n up to 40, singular g
//       included; rank(Phi_n (x) F_ell) = n + 1 iff n < ell iff
//       PhiNIsIsomorphism, on a grid.
//   A5  the contragredient relation list: relation by relation the same
//       (q_dim, s_lhs, s_rhs, tag, g_r) as Sym^n's, M = rho_n^vee(g_r) =
//       rho_n(g_r^{-1})^T, and Phi_n M_M = M_D Phi_n (a morphism of local
//       systems).
//   A6  H^*(D_n) through the shipped pipeline: for n < ell dim H^q(D_n) =
//       dim H^q(M_n) and P^q = H^q(Phi_n) invertible; for n >= ell the
//       dimensions and rank P^q are RECORDED; (1 (x) Phi_n) z satisfies every
//       D_n relation for every H^q(M_n) basis cocycle z.
//   A7  A(D_n)(T_p)^q for p in {2, 3, 5, 7}, every q: the cochain identity
//       Theta^D (1 (x) Phi) = (1 (x) Phi) Theta^M on all of C^q(K; V), and
//       A(D_n) P = P A(M_n) -- EXPONENT 0, pre-stated from [C, lem:symdual]
//       (printed DUALITY_EXPONENT lines) -- for n < ell and n >= ell alike;
//       NEGATIVE CONTROLS: the p^{+n} and p^{-n} readings FAIL on every
//       (cell, p, q) with p != ell, p^n != 1 mod ell and P A(M_n) != 0 (and
//       at least one such non-vacuous case exists).  Both routes of the D_n
//       operator agree on standard bases (apply_Theta vs apply_route2).
//   A8  the memo stamp carries the representation: a rho_n operator and a
//       rho_n^vee operator constructed at the SAME address give identities
//       that differ ONLY in `representation`, and a memo stamped by one is
//       refused by the other (ENGINE_CHECK -> std::logic_error); a
//       contragredient operator on the Sym^n local system is refused at
//       construction.
//   A9  the golden objects (tier A dim_H_Dn^q, charpoly_A_Dn(T_p)^q) are
//       DEFINED and well formed; they are never called from the harness.
//
// slot 6, WP-3 (phase E7b) -- on 20 cells at ell >= 5, n even (WP-1's ten,
// (10, 691) and the ten Kummer cells of HeckeExtKummerValidation):
//   A10 Hecke/HeckePairing: the fundamental chain (exactly +-eps qualify),
//       PairingCertificate::verify() == 0 (descent on the stored Z / B bases
//       of the cone, of H^*(D_n) and of both circle models), the stored form
//       == the cup-product definition (evaluate_direct) on every basis pair,
//       and a non-vacuity check of the descent.
//   A11 PERFECTNESS ([C, cl:pd](1), OPEN in [C]): dim H^t_c(M_n) =
//       dim H^{2-t}(D_n) and the Gram has full rank, t = 0..2; the circle
//       pairings likewise, s = 0..1.  A failure is a FINDING about [C].
//   A12 THE PD GATE FOR X_3 ([C, cl:pd](2), [C, obl:coneexact](vii) third
//       sentence): A_c^{(t)T} G^t = G^t A(D_n)^{(2-t)} for p in {2,3,5,7},
//       t = 1, 2, no determinant factor; the p^n-twisted reading FAILS on
//       every non-vacuous (cell, p, t).  A failure is a FINDING about [C].
//   A13 socle <-> Loewy ([C, cl:pd](3)): HeckeExt/HeckeIdentityCertificate
//       item 7 passed wherever X_3 != 0, not applicable where X_3 = 0; a
//       tampered (still perfect) pairing is rejected wherever the Loewy
//       length is >= 2.
//   A14 the iota-pairing sign <iota x, iota y> = eps <x, y>, RECORDED with
//       the prediction -(-1)^n of [C, rmk:iotapairing]; asserted only to be
//       +-1.
//   A15 the golden objects pairing_perfect^t, pd_adjoint^t (tier A),
//       pairing_matrix^t (tier B): DEFINED; since slot 7 WP-1 (stage 0) they
//       are emitted by Golden/GoldenHarness.cpp on the 31 ext-subgrid cells.
//
// slot 7, WP-1 (E7 acceptance (b)) -- COVERAGE on the E2 case list at ell >= 5
// (the 31 ext-subgrid cells of [X] Sec. 1.6 and the 23 M1 cells), beyond the
// 20 cells above:
//   A16 A11's dimension identity and perfectness (cone and circle), the
//       certificate, and A12's PD gate for p in {2,3,5,7} with its p^n-twisted
//       negative controls, on the 34 uncovered cells with n <= 34 (inside
//       HeckeDualityValidation), and on the M1 cells (58, 7), (58, 5),
//       (82, 7) in the separate HeckeDualityCoverageSlowValidation (slow);
//       (160, 691) and (198, 691) (~6 min together) only under
//       HECKE_DUALITY_COVERAGE_LARGE=1.

#include <cstdint>

uint32_t HeckeDualityValidation();
uint32_t HeckeDualityCoverageSlowValidation();

#endif // HECKE_DUALITY_VALIDATION_H
