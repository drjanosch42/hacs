// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_STRICT_LIFT_VALIDATION_H
#define HECKE_STRICT_LIFT_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 10, WP-10 (Line 2, L2-1b (1)).  The strict lift
// tau_Iso of [C, prop:strictlift] (Integral/Model/StrictLift.{h,cpp}) and its
// coset transport Theta^_g, for T_p, p in {2, 3, 5}:
//   (A) construction and every certificate of StrictLift::certify(), degrees
//       0 .. q_max + 1 = 6: (T1) on the Schreier generators of Gamma'', (T2),
//       (T3), eps tau_Iso = tau^s eps STRICTLY, tau_Iso(R_d) in R_d and
//       eps tau^d_Iso = tau_bd eps; the integer sizes printed (K1);
//   (B) Theta^_g a cochain map on C_I over F_ell, Z/ell^k (ell >= 5) and over
//       Z/4, Z/8, Z/9, Z/27 ([C, lem:cosettransport] (2) on the free model);
//   (C) STRICT COMPATIBILITY eps^* o Theta^s_g = Theta^_g o eps^* entrywise on
//       C^q_eq, q = 0, 1, 2 ([C, prop:strictlift] (3)), n <= 20, on the rings
//       with a SHIPPED Theta^s_g (plan B-1): F_ell, ell in {2, 3, 5, 7}, and
//       Z/ell^k, k in {2, 3}, ell in {5, 7} -- Theta^s_g = HeckeOperatorOnK on
//       the BoundaryStrict table.  A consistency check of the transport code,
//       not of the lift (the lift is certified by (A) over Z); over Z/ell^k at
//       ell in {2, 3} the identity FOLLOWS from (A)'s exact Z-certificate and
//       is not separately checked (no shipped Theta^s_g there, decision 4);
//   (D) the operator at ell >= 5 ([C, prop:strictlift] (4), eps^* an
//       isomorphism on H^., [C, prop:isomodel] (4)): Theta^_g on H^1(C_I (x)
//       F_ell), read through eps^*, equals the shipped A^{(1)}(T_p) (the
//       canonical tau) at (10, 691), (12, 5), (16, 7), (20, 691) and more;
//   (E) independent oracle O5 ([IB] Sec. 3.4): char(Theta^ | H^1(C_I (x)
//       F_ell)), H^1 taken INTRINSICALLY on the free model, equals O5 mod ell
//       for n in {10, ..., 20}, p = 2, and (x - 177148)(x - 252)^2 for p = 3
//       at n = 10;
//   (F) negative control: one perturbed tau_Iso entry is caught by certify()
//       AND breaks the strict compatibility of (C).
// Returns the number of failed checks.
uint32_t HeckeStrictLiftValidation();

#endif
