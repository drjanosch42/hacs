// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LVALUE_WILD_CHARACTER_VALIDATION_H
#define HECKE_LVALUE_WILD_CHARACTER_VALIDATION_H
// (hecke-extensions slot 21, P4) HeckeLValueWildCharacterValidation (standard): wild characters on the L-side -- WildCharacter.{h,cpp},
// spec docs/implementation/slot21/P4_spec.md (predictions committed first), Python oracle docs/implementation/slot21/p4/.  Characters
// chi = omega^i psi_t of conductor ell^m (m = 2, 3) with values in O = Z_ell[zeta_{ell^{m-1}}].  Arms:
//   (W0) the characters (multiplicative, primitive, chi(u) = zeta^t); P4's O arithmetic = NumberField's CycloModRing; NEG t = ell;
//   (W1) F B_{r, omega^{b-r} psi_t} (definition) = -F r G_b(zeta^t u^r - 1) (both KL group-ring routes); the pole branch v_pi = -1; NEG zeta^{-t};
//   (W2) [E] prop:lvmtt(4) at wild chi (alpha^{-m}), the symmetry lambda_{chi,j} = -eps lambda_{chibar,n-j}; NEG chi for chibar, alpha^{-(m-1)};
//   (W3) branch elements at the wild points = the Riemann sums; v_pi(L_b(zeta - 1)) = e mu_b + lambda_b;
//   (W4) P2's Eisenstein measure at wild chi: -B_{1,chi} B_{k-1,chibar}/(k-1) exactly; the polar term at zeta^t has v_pi = e d - 1;
//   (W5) complex L(f, chi, j+1) (incomplete-Gamma series, nf::MpReal) against lambda_{chi,j}: the tau(chi) constant of prop:lvmtt's analytic
//        form for every primitive chi of conductor 25, 49, 121 (k = 12) and 25, 49 (k = 16); root number tau(chi)^2 / M; NEG tau(chibar), conductor ell M;
//   (W6) the cusp form's pole branches at wild points: -s (F1 F2 + Res/(zeta^{+-t} u^r - 1)) mod ell^d; NEG no polar, -Res;
//   (W7) non-ordinary: UnitRoot refuses at (10,5), (10,7).
// Prints WILD_* lines and P4_ARMS checks=<c> failing=<f>.  Env (read-only): HECKE_LVALUE_WILD_HEAVY=1 adds W5 (12; 125), (12; 169),
// W2/W3 (10,11) at m = 3 and W6 (30,37) at t = 2 (EFFECTIVE HeckeLValueWildCharacterValidation arm=HEAVY cells=<k>; else arm=DEFAULT).
// Writes nothing; no golden object.
#include <cstdint>

uint32_t HeckeLValueWildCharacterValidation();

#endif // HECKE_LVALUE_WILD_CHARACTER_VALIDATION_H
