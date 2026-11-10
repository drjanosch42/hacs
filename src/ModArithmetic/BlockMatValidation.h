// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// BlockMatValidation -- the block-sparse data model (matrix-engine Phase 3, D3)
// =====================================================================
//
// ENTRYWISE checks of BlockPool / BlockRef / BlockMat / BlockView /
// BlockBuilder and the pattern-driven block kernels against dense
// arithmetic on fixed-seed inputs: kinds (Zero / I / alpha*I / M / -M / I-M /
// pencils / transposed windows), pool dedup + refcounting + hash-collision
// handling, CSR order determinism, the builder's set / accumulate folding,
// non-uniform block dimensions including zero-width blocks, both element
// widths, primes {2, 3, 7, 23, 134217757, 2^31-1, 2^32-5} (+ 2^61-1 for
// uint64_t), the to_flat / from_flat round trip and the L1 index identity
// against a hand-written s*d_V + k loop, structural transpose, views, block
// GEMM (kind folding + CSE) vs the dense product, gemm_dense, add / scale,
// and equal() as mathematical matrices.  Returns 0 on success.

#include <cstdint>

uint32_t BlockMatValidation();
