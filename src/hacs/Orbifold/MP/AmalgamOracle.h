// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_AMALGAM_ORACLE_H
#define ORBIFOLD_MP_AMALGAM_ORACLE_H

// ===========================================================================
// (hecke-extensions slot 22, MP-1b) AmalgamOracle: a C++ re-implementation of
// docs/implementation/slot22/mp/audit/mv_check.py section (1) -- the
// Mayer--Vietoris sequence of SL_2(Z) = A *_C B, A = <S> (Z/4), B = <ST>
// (Z/6), C = <-I>:  dim H^q = dim ker r_q + dim coker r_{q-1},
// r_q = res_A - res_B : H^q(A) (+) H^q(B) -> H^q(C), with the cyclic-group
// cohomology as subquotients of M = Sym^n F_p (H^0 = M^g, odd = ker N /
// (g - 1)M, even >= 2 = M^g / N M; restriction = inclusion in even degree,
// 1 + g + ... + g^{k-1} in odd degree, k = [G : C]).  Own Sym^n (the
// script's x^(n-j) y^j basis), own elimination mod p.
//
// R5 (MP_impl_plan_review.md F7): the PRIMARY oracle of
// OrbifoldMPNerveValidation is the frozen output mv_check.out as an .inc
// (MvCheckOracle.inc); this re-implementation, by the same agent as the nerve
// code, is compared with the .inc as its own check, never used as the gate.
// ===========================================================================

#include <cstdint>
#include <vector>

namespace orbifold_mp {

// dim_{F_p} H^q(SL_2(Z); Sym^n F_p), q = 0 .. q_max (p prime, p < 2^16).
std::vector<uint32_t> AmalgamDims(uint32_t n, uint32_t p, uint32_t q_max);

// The frozen mv_check.out section (1): rows (ell, n, dims q = 0..7) for ell in {2, 3, 5, 7}, n = 0..12.
struct MvCheckRow {
    uint32_t ell, n;
    uint32_t dims[8];
};
const std::vector<MvCheckRow>& MvCheckOracle();

}  // namespace orbifold_mp

#endif
