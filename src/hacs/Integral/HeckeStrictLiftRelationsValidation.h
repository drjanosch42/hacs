// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_STRICT_LIFT_RELATIONS_VALIDATION_H
#define HECKE_STRICT_LIFT_RELATIONS_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 11, WP-11 (M-3 of the slot-10 review): the strict lift
// tau_Iso of [C, prop:strictlift] for the double cosets of determinant p^2
// (and p^3) -- g = diag(1, p^nu), nu in {2, 3}, and the central g = p I -- on
// the free model C_I (Integral/Model/StrictLift, appended), and the
// [C, thm:relations] identities (X7 of [IB]) on H^q(C_I (x) R).
//   (A) construction and every certificate of StrictLift::certify(), degrees
//       0 .. 6: (T1) on the Schreier generators of Gamma'' = Gamma_0(p^nu)
//       (Gamma for p I), (T2), (T3), eps tau_Iso = tau^s eps STRICTLY (tau = id
//       for p I, [C, prop:diamond]), tau_Iso(R_d) in R_d, eps tau^d_Iso =
//       tau_bd eps; integer sizes printed (K1).  p in {2, 3} (nu = 2, 3, pI),
//       p = 5 (nu = 2, pI);
//   (B) Theta^_g a cochain map over F_ell, Z/ell^k ([C, lem:cosettransport] (2));
//   (C) strict compatibility eps^* Theta^s_g = Theta^_g eps^* on C^q_eq,
//       q <= 2, for diag(1, p^2) on the rings with a shipped Theta^s_g
//       (F_ell) -- a consistency check of the transport code;
//   (D) T(p, p) on the model: Theta^_{pI} - p^n induces 0 on H^q, q <= 5,
//       every ring of (E) ([C, prop:diamond] through [C, prop:strictlift] (3));
//       whether tau_Iso^{pI} is the identity on the nose is printed;
//   (E) X7 in full ([C, prop:relationsmodel](2), PROVED for every ring R,
//       ell in {2, 3} included): with the FULL operators of [C, eq:fullassembly] on the
//       model (T_{p^2} = Theta^_{diag(1,p^2)} + Theta^_{pI}, T_{p^3} =
//       Theta^_{diag(1,p^3)} + p^n Theta^_{diag(1,p)}), the recursion
//       T_{p^{nu+1}} = T_p T_{p^nu} - p^{n+1} T_{p^{nu-1}}, nu in {1, 2}, induces
//       0 on H^q(C_I (x) R), q <= 5, basis-free (ModelCohomology::induces_zero),
//       p in {2, 3}, at the M1(a) cells n in {10, 12, 16} (and a sweep n <= 20
//       over F_2, F_3), over Z/2^8, Z/3^5, F_2, F_3 (p = ell included) and
//       F_5, F_7, F_691; non-vacuity (cochain-level failures counted) and two
//       negative controls (the primitive T_{p^2}, [C, err:e1]; the coefficient
//       (p + 1) p^n of [C, rmk:relcoeff]);
//   (F) the ell >= 5 consistency check (labelled): Theta^_{diag(1,p^nu)} on
//       H^1(C_I (x) F_ell) through eps^* equals the shipped primitive A^{(1)}
//       of HeckeOperatorOnK on HeckeCosetTables(p, nu) with the canonical tau,
//       nu in {2, 3} ([C, prop:strictlift] (4));
//   (G) negative control: a perturbed diag(1, p^2) lift is rejected by certify().
// Returns the number of failed checks.
uint32_t HeckeStrictLiftRelationsValidation();

#endif
