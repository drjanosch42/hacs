// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef MOD_CONTEXT_VALIDATION_H
#define MOD_CONTEXT_VALIDATION_H

#include <cstdint>

// ===========================================================================
// matrix-engine Phase 2 -- ModCtx<Elt> validation (plan D2, "Phase 2 /
// Validations").
//
// Every primitive of ModCtx.h is compared SAMPLE BY SAMPLE against the legacy
// ModArithmetic::mod_* (uint32_t) / mod_*64 (uint64_t) functions on fixed-seed
// inputs, including the corner values 0, 1, p-1, at every regime boundary:
//
//   uint32_t: p in {2, 3, 5, 134217757, 1518500213 (k_max 8, last Delayed),
//                   1518500279 (k_max 7, first Shoup), 2^31-1, 2^31+11, 2^32-5}
//   uint64_t: p in {2, 3, 5, 134217757, 2^32-5, 2^32+15, 2^61-1, 2^63-25}
//
// Checked: mul/add/sub/neg/inv/pow, mod_mul_barrett and reduce_u64 on the
// FULL uint64_t range, shoup_precompute/shoup_mul(_lazy) (and that the
// ENGINE_CHECK fires where 2p > 2^32), mont64_to/from/mul/reduce round trips,
// batch_inv under both zero policies, canonicalise, the regime tag, and the
// k_max bound: k_max*(p-1)^2 fits the accumulator and (k_max+1)*(p-1)^2 does
// not; kernels::dot_delayed over exactly k_max terms of (p-1)^2 is exact and
// over k_max+1 terms throws the ENGINE_CHECK.
//
// hecke-extensions slot 8 WP-1 (E8 item 2, ENGINE_API.md Sec. 13): every
// ModCtx(p) above must declare the field (is_field, ell == p, k == 1); the
// COMPOSITE arms run ModCtx::prime_power at Z/4, Z/125, Z/691^2, Z/3^20
// (U32_Barrett64), Z/2^31 (U32_Shoup), Z/134217757^2 (U64_Montgomery), Z/2^40
// and Z/3^40 (U64_Wide): the Sec. 1 constants equal init(ell^k)'s, the
// modulus-only primitives equal the legacy mod_* sample by sample, k_max is
// tight, inv is unit-only, pow / batch_inv refuse, init() resets the ring,
// and prime_power(ell, k < 2) / overflow are rejected.
//
// Returns the number of FAILING checks (0 = pass).
// ===========================================================================
uint32_t ModContextValidation();

#endif // MOD_CONTEXT_VALIDATION_H
