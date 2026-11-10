// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef KERNEL_BIT_IDENTITY_VALIDATION_H
#define KERNEL_BIT_IDENTITY_VALIDATION_H

#include <cstdint>

// ===========================================================================
// matrix-engine Phase 2 -- kernel bit-identity harness (plan D6/D7, "Phase 2
// / Validations").
//
// Every kernel of Kernels.h is compared ENTRYWISE against the naive in-tree
// oracle RefLinAlg (legacy mod_mul/mod_add, schoolbook loops) on fixed-seed
// inputs at sizes {1, 2, 7, 16, 17, 64, 65, 129, 256, 517}, for both element
// widths, all four (A, B) layout combinations, row- and column-major C,
// transposed views, padded leading dimensions, Set/Add/Sub, and k > k_max
// (k-blocking at k_max = 4 / 1 for p = 2^31-1 / 2^32-5 and at k_max = 4 for
// p = 2^63-25). gemm is additionally compared against the OLD
// FlatMatrixT<Elt>::operator* on the same data. When the CPU has AVX-512 the
// AVX-512 microkernel and the portable microkernel are BOTH run and must
// agree bit for bit with each other and with the oracle.
//
// Also covered: gemv, gemm_nt, trmm (unit-triangular by tag, garbage in the
// unread triangle), apply_signed_antidiag / apply_diagonal, transpose / copy
// across layouts, axpy / scal (Shoup and Barrett regimes), canonicalise
// (Lazy2p and Accum64), swap_rows, the ENGINE_CHECK guards (shape, aliasing,
// non-canonical operand), and the oracle's own RREF / kernel basis against
// FlatMatrix (the D5b contract).
//
// Set ENGINE_KERNEL_TIMING=1 to append a single-thread GEMM throughput table
// (Gmodmul/s at n in {64, 128, 256, 512}, p = 134217757, per kernel path).
//
// Returns the number of FAILING checks (0 = pass).
// ===========================================================================
uint32_t KernelBitIdentityValidation();

#endif // KERNEL_BIT_IDENTITY_VALIDATION_H
