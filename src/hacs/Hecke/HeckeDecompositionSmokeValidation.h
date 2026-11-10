// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_DECOMPOSITION_SMOKE_VALIDATION_H
#define HECKE_DECOMPOSITION_SMOKE_VALIDATION_H

#include <cstdint>

// Stage S10 -- Definition-of-Done smoke validation for the F_ell[X] toolkit
// (FpPoly) and the single-operator level-(c) block-diagonalization
// (HeckeModuleDecomposition). Returns 0 on success (invariant I9).
//
// Checks (per the plan's S10 "Definition of Done"):
//   (0) FpPoly corner batteries: Berkowitz charpoly(companion(q)) == q
//       (including ell = 2 < d, exercising division-freeness); certified
//       factorization of the inseparable corner f' = 0 (X^5 - 1 = (X-1)^5
//       over F_5, and X^4 + X^2 = X^2 (X+1)^2 over F_2, both entering the
//       ell-th-root path); the ell = 2 trace-map equal-degree split
//       ((X^2+X+1)(X^3+X+1)(X^3+X^2+1) over F_2); an extgcd Bezout identity.
//   (i) 200 fixed-seed pseudorandom matrices of dimension d <= 8 over
//       ell in {5, 13, 101, 134217757} (50 each), ASSEMBLED from known block
//       forms: a block-diagonal truth of companion blocks of pi^s (pi random
//       monic irreducible of degree <= 3, certified irreducible at
//       construction time), conjugated by a random invertible U0. The
//       recovered invariants (irreducible-factor multiset, generalized-Jordan
//       partitions, residue degrees, char/min polys) must match the
//       constructed truth, certify() must return 0, conjugated(p) must be
//       EXACTLY block-diagonal (re-verified here from change_of_basis()
//       independently of the class's own check), the reconstruction
//       U C U^{-1} == A must hold bit-exactly, and MinPolyKrylov(A) must
//       equal prod_j pi_j^{e_j} of the truth.
//  (ii) The companion matrix of (X^2+X+1)^2 over F_5 (X^2+X+1 irreducible
//       mod 5): exactly one block, Jordan partition (2), residue degree 2.
//
// On a constructed-truth recovery failure this prints the failing test index
// (seed), the constructed block form, and the recovered one -- the
// certification is never weakened.
uint32_t HeckeDecompositionSmokeValidation();

#endif // HECKE_DECOMPOSITION_SMOKE_VALIDATION_H
