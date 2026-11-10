// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_HECKE_Z_FREE_BASIS_PAIRS_VALIDATION_H
#define INTEGRAL_DESCENT_HECKE_Z_FREE_BASIS_PAIRS_VALIDATION_H

// ===========================================================================
// HeckeZFreeBasisPairsValidation (standard; hecke-extensions slot 25, EXPORT-Z step 2 package EXZ2-P1a;
// docs/implementation/slot25/EXPORTZ2_design.md v2 Sec. 1.0, 1.1, 7; IsoTreeCarrierZ.h).  Arms:
//   (C) the carrier: d^0, d^1 over Z reduced mod two primes = TreeModelChain(0, .)::coboundary on the dense induced module
//       (entrywise); = the lane route d_mod on random columns; d^1 d^0 = 0 exactly; RhoNZ mod p = rho_n on random g (det 1, det p,
//       a = 0); rank d^1 + rank d^0 + b_1 = dim C^1.
//   (T) the Hecke lanes: terms_Z() reduced = the reduced terms (the same apply on random columns); T~_T is a cochain map in degrees
//       0 -> 1 -> 2 mod a lane (d T = T d); teeth: the BlockSign fault breaks it; the majorant rho_n(|E|)|mul| dominates the exact
//       |T~_T X| on 0/1 columns (symmetric residues below q/2).
//   (D) the Dixon pairs under iso_dixon/v3 (v2 until EXZ2-P1b): |P_s| = r; d^1 B_s = 0 and Phi_s^T d^0 = 0 by plain BigInt products (not the lanes);
//       supports, positivity, primitivity; on (1, 10), (2, 6), (11, 2) every column = the exact rational solution of an independent
//       Gauss-Jordan over Q, primitive-normalised; det(Phi_s^T B_s) != 0; q0-independence (the pairs again at the second CrtPrimes
//       prime: the same FNVs); the pinned FNVs of (1, 10) (s = 0 unchanged since v2, s = 1, 2 re-pinned for v3); teeth: b_1 + e_c and 7 b_1 fail.
//   (R) the size cap: a 1-byte ceiling refuses [RssCeiling] before anything is built.
//   (M) measurement (the EXZ2_P1A lines): default cells (1, 10), (11, 2); HECKE_EXZ2_CELLS=N:n[:p1/p2],... replaces them (read-only,
//       prints EFFECTIVE); HECKE_DUMP_EXZ2=<dir> (scratch, outside the source tree, else REFUSED) writes the carrier, the pairs and
//       H_s = Phi_0^T T~_T B_s (and T~_T on C^1 for dim C^1 <= 420) for docs/implementation/slot25/exz2p1a/exz2p1a_check.py.
// Gate line: HECKEZ_FREEBASIS_PAIRS arms=5 failing=0 checks=.. rule=iso_dixon/v3 verdict=OK.

#include <cstdint>

uint32_t HeckeZFreeBasisPairsValidation();

#endif
