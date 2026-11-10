// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_NERVE_H3_SLOW_H
#define ORBIFOLD_MP_NERVE_H3_SLOW_H

#include <cstdint>

// ===========================================================================
// (hecke-extensions slot 22, MP-2) OrbifoldMPNerveH3Slow (slow).
//
// The nerve in degree 3 (MP_impl_plan.md v0.2 Sec. 2.6; MP_predictions.md P4.4
// and its MP-2 detail): H^3 of the MP cochain complex over F_ell, ell in {2,3},
// n in {0,2}, on M7 and R_Z (d^3 : C^3 -> C^4 on the k = 4 strings, ~1.2 M
// non-degenerate ones), against the frozen mv_check.out; Phi^* : H^3(C_I) ->
// H^3(MP) an isomorphism (Phi_3 certified, Phi^*_3(z) a cocycle by d^3); and
// F-MP1 F3: the independent enumerator MPSet(D, 4) equals the interned S_4
// string for string.  Prints `EFFECTIVE OrbifoldMPNerveH3Slow arm=default
// cells=8`.  RSS <= 6 GB predicted (rule 19).
// Line: ORBIFOLD_MP OrbifoldMPNerveH3Slow checks=<k> failing=0.  model=mp.
// Read-only; no golden object, no setting, digest-neutral.
// ===========================================================================
uint32_t OrbifoldMPNerveH3Slow();

#endif
