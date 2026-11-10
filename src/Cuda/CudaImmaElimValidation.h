// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// CudaImmaElimValidation (slot-22 G2; ENGINE_API.md Sec. 11.9): the device-resident elimination
// cuda::imma_rref (int8 storage, cooperative base panels, IMMA updates with the int8 epilogues)
// against the host engine, entrywise:
//   (A) imma_rref directly vs engine::elim::rref: rank, pivots, the RREF bytes, and pivot_rows (the
//       full sequence against a scalar reference of the smallest-original-row rule on small inputs;
//       the set against the row rank profile -- the column rank profile of A^T -- on all), over
//       p in {2, 3, 5, 13, 127}, random / planted-rank / zero and duplicated columns at panel and base
//       boundaries / zero and duplicated rows / sparse +-1 / extreme residues / zero / identity / 1 x n /
//       m x 1, both base widths, multi-slab uploads, windowed (ld > n) sources and sinks;
//   (B) tape::rref and every driver entry point (kernel_basis, image_basis, factor_basis with projector,
//       solve_multi_rhs incl. the out-of-span verdict) under the thread-local Gpu override vs engine::blk,
//       pivot_rows included (the driver's per-panel order), with the device route asserted per case;
//   (C) the DenseElim hook: elim::rref (both layouts), column_rank_profile, inverse, solve_multi_rhs,
//       complement_basis (projector), rref_with_transform under Gpu vs Cpu, counters asserted;
//   (D) routing: ENGINE_GPU_ELIM=0 (a legacy arm since S24 CL-R2: OC_ENABLE_LEGACY=ON only), the Cpu override, p = 131, a Z/9 context, the size floor
//       ENGINE_GPU_ELIM_MIN and a uint64_t context never reach the device (counters 0, same values);
//   (E) launch counts: one cooperative launch per base, ceil(n / W) bases on a full-column-rank input,
//       IMMA launches == the elimination's GEMM count (non-vacuous);
//   (F) negative controls that must fire: a one-entry perturbation of the input is detected; the
//       device's raw pivot order differs from the panel driver's order on a blocked input (so the
//       per-panel sort is exercised); a deliberately wrong rule (largest original row) disagrees with
//       pivot_rows.
// Last line: CUDA_IMMA_ELIM checks=<c> failing=<f> device_calls=<k> bases=<b> gemms=<g> wall_s=<t>.
// [SKIP] without a usable device or an sm_80+ IMMA image (and with OC_ENABLE_CUDA=OFF).
#include <cstdint>

uint32_t CudaImmaElimValidation();
