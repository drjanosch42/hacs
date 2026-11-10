// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_REDUCTION_PIPELINE_VALIDATION_H
#define HECKE_REDUCTION_PIPELINE_VALIDATION_H

#include <cstdint>

// ===========================================================================
// HeckeReductionValidation's E8 ITEM 3 ARMS (hecke-extensions slot 8, WP-3):
// the pipeline over Z/ell^k, the rho_n repair and `coefficient_ring` with its
// refusals.  Called from HeckeReductionValidation() (Hecke/
// HeckeReductionValidation.cpp) after WP-1's elimination arms; kept in its
// own TU so that WP-4's reduction / Bockstein arms can reuse the cell runner.
//
//  (K)  the key: `coefficient_ring` registered at the registry's tail (11
//       keys), default `F_ell`, fingerprint 0 at the defaults; the grammar
//       F_ell | Z/ell^<k> (2 <= k <= 4096 since slot-11 WP-18c; 63 before); `Z` and `Z/<m>` raise
//       NotImplementedInThisBuild inside MathSettings::set(); every other text
//       std::invalid_argument; canonical text round trip; a non-default ring
//       makes a non-zero fingerprint.
//  (B)  Sec. 13.7's decision: BlockBuilder / BlockMat's ModCtx overload keeps a
//       composite context (is_field() == false) through finalize, zeros,
//       from_flat, transposed and blk::gemm / add / scale; the field overload
//       is unchanged; blk::rref and tape::rref REFUSE a composite BlockMat
//       (the Sec. 13.3 guards WP-1 could not test).
//  (P)  E8-P5, the rho_n repair: ComputeSymmetricPowerMatrixRing ==
//       ComputeSymmetricPowerMatrixRingOracle ENTRYWISE on every g the cells
//       touch (the seeds, -I, s_i^{-1}, g gamma_k, delta_i gamma_k for every
//       prime of the cell, U_5 at ell = 5 included), each route counted
//       (unit_a / unit_b / split / first_row_zero); the Z/ell^k matrix
//       reduces mod ell to ComputeSymmetricPowerMatrixFM's.
//  (C)  the cohomology over Z/ell^k at the pre-stated cells ([E] sec:ring (e),
//       slot-8 plan WP-3 scope 7): invariant factors of H^0 / H^1 / H^2, the
//       K-mode counts r_K(d^0), r_K(d^1) (51 != 52 at (60,5) K=2; rank C^0 - 1
//       at (30,5) K=2), ladic_type_certified, ladic_type^q; lengths; the
//       field accessors refusing on a composite object.
//  (T)  A^{(q)}(T_p) over Z/ell^k at p in {2, 3} (and 5 = U_5 at ell = 5, 5
//       at (10, 691)): O12(a) over the ring, (i)/(ii)/(iii) counters, the
//       free-quotient charpoly at every certified level (precision
//       K - e_max per the table) dividing the F_ell charpoly of A^{(1)}(T_p)
//       mod ell (EQUAL to it where H^1 is free), and the named logic_error at
//       every uncertified level.
//  (U)  the u64 arm: (10, 134217757) over Z/134217757^2 (correctness only).
//  (X)  the refusal closure: every surface and every refusing member raises
//       NotImplementedInThisBuild naming itself under Z/ell^2, and the
//       honouring objects refuse ell in {2, 3} ((10, 3) under Z/ell^2).
//
// Adds its checks and failures to the caller's counters.
// ===========================================================================
void HeckeReductionPipelineArms(uint64_t& checks, uint32_t& failures);

#endif // HECKE_REDUCTION_PIPELINE_VALIDATION_H
