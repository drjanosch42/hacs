// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LVALUE_IWASAWA_INVARIANTS_VALIDATION_H
#define HECKE_LVALUE_IWASAWA_INVARIANTS_VALIDATION_H
// (hecke-extensions slot 21, P5 (b), (c)) HeckeLValueIwasawaInvariantsValidation (standard): the Iwasawa invariants (mu, lambda) of the
// branches of the ell-stabilised MTT measure (WP-MEAS) and the tame functional equation.  Spec docs/implementation/slot21/P5_spec.md
// (v0.1, predictions committed before this code ran).  Arms:
//   (I0) Kubota-Leopoldt (mu, lambda)(G_b), every even b, by the group ring (KlGroupRingBernoulli) and by exact Bernoulli numbers alone
//        (Newton divided differences of -(1 - ell^{r-1}) B_r / r at r = b + s(ell - 1)); mu = 0; lambda = 1 exactly at the irregular pairs;
//   (I1) the tame functional equation lambda_{chi,j}(P) = -chi(-1)(-1)^j lambda_{chibar,n-j}(P), conductor 1 and ell, P in W (eigen and
//        generic); parity; NEG the flipped sign, NEG chi for chibar;
//   (I2) the branch functional equation g_b = -(-1)^b GrTwInv(g_{n-b}, n+1) mod ell^{min(M,R)}; NEG the flipped sign;
//   (I3) the (mu, lambda) table per cell and sign: invariants() against the values oracle (divided differences of the critical values
//        from EvAllJ, no ball table); (mu, lambda)_b = (mu, lambda)_{n-b}; lambda odd at even self-dual branches, even at odd ones; the
//        near-trivial zero where alpha == 1 mod ell; mu = 0 at the controls;
//   (I4) level-1 Greenberg-Vatsal at the Eisenstein cells: lambda^-_a = lambda(G_{a+1}) + lambda(G_{k-1-a}) off the pole branches; mu^+ = d;
//   (I5) the table against MeasureFrozenR201.inc / MeasureFrozenS2.inc where the cell is shared.
// Prints IWA_* lines and P5_ARMS checks=<c> failing=<f>.  Env (read-only): HECKE_LVALUE_IWASAWA_HEAVY=1 adds the KL table at 691 and the
// cells (10,691), (282,37), (226,103) (EFFECTIVE HeckeLValueIwasawaInvariantsValidation arm=HEAVY cells=<k>; else arm=DEFAULT).
// Writes nothing; no golden object.
#include <cstdint>

uint32_t HeckeLValueIwasawaInvariantsValidation();

#endif // HECKE_LVALUE_IWASAWA_INVARIANTS_VALIDATION_H
