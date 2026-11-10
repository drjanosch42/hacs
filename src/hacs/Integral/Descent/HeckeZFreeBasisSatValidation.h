// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_HECKE_Z_FREE_BASIS_SAT_VALIDATION_H
#define INTEGRAL_DESCENT_HECKE_Z_FREE_BASIS_SAT_VALIDATION_H

// ===========================================================================
// HeckeZFreeBasisSatValidation (standard; hecke-extensions slot 25, EXPORT-Z step 2 package EXZ2-P1b;
// docs/implementation/slot25/EXPORTZ2_design.md v2 Sec. 1.2-1.4, 6 arms (B), (E), (F); DescentFreeBasisZ.h).  Arms, in this order:
//   (M) measurement, FIRST (so its RSS growth is taken in a fresh process; P1a audit F4): default cell (2, 6);
//       HECKE_EXZ2_CELLS=N:n[:p1/p2],... replaces the cells (read-only, prints EFFECTIVE; the primes are ignored here).  One
//       EXZ2_P1B line per cell (pairs, sums, saturation, certificate, the calibrated RSS prediction against the sampled growth) and one
//       EXZ2_P1B_SAT line per saturation step.  Fix round 1: the sums alone are sampled too (after malloc_trim) and checked against
//       PredictSumsRss (never below); q0 / q0_confirm / q0_skipped printed beside the rule (audit F7).
//   (S) the sums (Sec. 1.2) on (1, 10), (2, 6), (11, 2), (1, 30), (5, 12), (9, 8): B_Sigma cocycles and Phi_Sigma killing
//       d^0 exactly; Delta_Sigma = |det(Phi_Sigma^T B_Sigma)| (Bareiss); |Delta_Sigma| bits = the P1a audit's
//       alt2 figures (Sec. 3.2: 1, 2, 1, 42, 29, 18; (8, 10) 33 in arm (K)), an independent Python run of the same rule.
//   (B) the certificate recomputed by the validation's own code: dense exact d^1 B = 0, Phi^T d^0 = 0, Bareiss det(Phi^T B) = +-1.
//   (K) the K-escalation orders (P1a audit F1): K = 5 on (8, 10) -- the five orders pairwise distinct, none of pi_1 .. pi_4 a rotation
//       by a multiple of D, every pair valid, Delta_Sigma(K = 5) | Delta_Sigma(K = 3), ExtendPairs 3 -> 5 = BuildPairs(K = 5); the
//       structural size reduction (the path above 48 relations) forced on K = 3 keeps Delta_Sigma; (1, 10): q0 | 6N and a forced rank
//       drop move to the next CrtPrimes prime with the same pairs (F10), small q0 = 5, 7, 11, 691 give valid pairs.
//       Fix round 1: the inline structural reduction is bit-identical to the post-hoc one (FNV of B_Sigma, Phi_Sigma; (8, 10), (9, 8)
//       forced structural) with a smaller peak (F1); the profile confirmation makes every small q0 give the rule's pairs (F7);
//       dixon_batch = 1, 8, 32 give identical pairs (F11); ExtendPairs above the ceiling is REFUSED [RssCeiling], not built (F5); the
//       K-ESCALATION LOOP of BuildFreeBasis executed on (1, 30) by the validation hooks (cofactor limit 2^6, trial bound 8): K_used = 5,
//       multi-word steps, splits > 0, certified; with the ceiling at the K = 3 prediction the escalation is a refusal (F3, F5).
//   (E) basis teeth: 7 b_1 (local mode), 2 b_1 (local, ell = 2), 2147483659 b_1 (composite mode) fail (B) and are repaired by
//       Saturate to det = +-1; a 70-bit composite multiplier is repaired in the multi-word mode and, with that mode disabled,
//       refused [SaturationCofactor]; B U (U unimodular) keeps the certificate.  Fix round 1 (audit F2, F4, F6, F8): the gcd split
//       (b_1 p1, b_2 p2 in composite64; b_1 P1, b_2 P2 in multi-word: splits >= 1 asserted), the side Phi (phi_1 p2, phi_1 P1 P2:
//       the replaced columns on side Phi asserted), B M with M triangular non-diagonal (det 504, local at 2, 3, 7), a B-only piece
//       runs no side-Phi pass (F8), Delta given too large stalls at once ([SaturationStall], 2 iterations), and a rank-(r - 1)
//       pair fails the rank witness (F6).
//   (F) the global oracle on (1, 10), (2, 6), (11, 2): an in-validation integer kernel (unimodular row reduction) gives Z^1 and
//       B^1_sat = (Q B^1) cap Z^1; det[B | B^1_sat] = +-1 in Z^1 coordinates (B is a Z-basis of L = Z^1 / B^1_sat), = +-7 for 7 b_1.
//       (F = C_B^{-1} F_oracle C_B waits for P2's F.)
// Gate line: HECKEZ_FREEBASIS_SAT arms=6 failing=0 checks=.. rule=iso_dixon/v3 saturation=det_pm1 verdict=OK.

#include <cstdint>

uint32_t HeckeZFreeBasisSatValidation();

#endif
