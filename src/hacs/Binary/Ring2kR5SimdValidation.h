// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

#include <cstdint>

// Slot-23 R5V gate (standard): the SoA / SIMD kernels of ring2k's R5 rung (Ring2kR5.cpp; Ring2k.h r5k::) are
// bit-identical to the scalar Traits<WordN<N>> bodies on every available kernel path (portable / AVX2 / AVX-512)
// and every N with a SIMD body:
//   (A) kernel fuzz: GEMM / AXPY (tails, zero / small / all-ones / random / single-limb factors, the delayed-carry
//       stress np >> TC with all-ones operands) and the row kernels add / sub / neg (chunk tails);
//   (B) end to end: smith / howell / kernel_module on R5 rungs, SIMD (each path) vs set_r5_kernel(Scalar), on random,
//       planted and amalgam matrices (the exact Sym^n coboundary [rho(S) - 1; rho(ST) - 1] and the cocycle relation
//       matrix of SL_2(Z) = Z/4 *_{Z/2} Z/6, computed in WordN arithmetic);
//   (C) the BIN-3 reader (l23::read_lattice) on an R5 rung, SIMD (each path) vs scalar: every reported field equal;
//   (D) mutation controls: r5k::set_fault 1 (dropped borrow chain), 2 (no carries between updates), 3 (no element
//       top-word mask) and a flipped output bit must each be caught by the comparisons above;
//   (E) timing (R5V_TIMING lines; reported, never gated).
// Prints one
//   RING2K_R5_SIMD Ns=... paths=... kernel=<k> e2e=<e> reader=<r> mutations=<caught>/<armed> checks=<c> failing=<f> verdict=OK|FAIL
// line.  Returns the number of failing checks.
uint32_t Ring2kR5SimdValidation();
