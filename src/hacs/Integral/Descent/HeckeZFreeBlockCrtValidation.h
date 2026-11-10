// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_HECKEZ_FREE_BLOCK_CRT_VALIDATION_H
#define INTEGRAL_DESCENT_HECKEZ_FREE_BLOCK_CRT_VALIDATION_H

// HeckeZFreeBlockCrtValidation (standard; EXPORT-Z step 2, package EXZ2-P2; docs/implementation/slot25/EXZ2P2_report.md).
// The free block F_p over Z by RNS lanes and the proven CRT (DescentFreeBlockCrt.h) on the committed FIXTURE bases
// docs/implementation/slot25/exz2p2/fixtures/c<N>_<n>.txt (B_sat, Phi_0, Phi_cert and the model's F_p; P1b builds the real basis):
//   (X) each fixture re-certified by this binary: d^1 B = 0, Phi_0^T d^0 = 0, Phi_cert^T d^0 = 0 (plain BigInt), det(Phi_cert^T B) = +-1
//       (Bareiss), and R0, P = BuildPairs' pair 0 (iso_dixon pair 0, unchanged v2 -> v3).
//   (L) single lanes: F_p mod q = the model's F_p mod q at several admissible q (the Psi route; no CRT).
//   (A) F_p (proven) = the model's exact F_p, and char(F_p) (Berkowitz over Z) = CharpolyOverZ's char_Z(T_p | H^1/tors) (independent route).
//   (C) F_p F_q = F_q F_p over Z; (11, 0): char F_2 = (x - 3)(x + 2)^2; (1, 10): char F_2 = (x - 2049)(x + 24)^2.
//   (P) the proofs: a-priori only, a-posteriori only and both give the same F; a later prime window (skip_first) gives the same F;
//       teeth: the unsaturated basis 2 b_1 (F not integral) is NOT proven by the a-posteriori test within 3x the lanes (refusal), and a
//       1-lane cap refuses.
//   (D) bit identity: FNV(F) and the lane residues for GEMM paths portable / avx2 / avx512, Garner paths portable / avx2 / avx512, threads
//       1 / 4 / 16, tiles 1 / 3 / 128, and the GPU lane route (ENGINE_DEVICE=gpu with a device; else [SKIP]).
//   (T) thread safety (audit F11): EchelonModQ::solve / solve_transposed and SparseZ::max_row_abs_bits from 8 threads at once; the
//       carrier holds no lane TreeModule after a run (F7); run under CheckThread via tsan_run.sh for the TSan arm.
//   (R) [RssCeiling]: a 1-byte ceiling refuses before anything is built.
//   (M) measurement lines EXZ2_P2 (default: the fixture cells); HECKE_EXZ2P2_FIXTURES=<path>[,<path>...] (read-only, prints EFFECTIVE)
//       replaces the measured list by those fixture files (a file without Phi_cert is measured with lanes_only: timings and the bound).
// Gate line: HECKEZ_FREEBLOCK_CRT arms=9 failing=0 checks=.. cells=.. verdict=OK.

#include <cstdint>

uint32_t HeckeZFreeBlockCrtValidation();

#endif
