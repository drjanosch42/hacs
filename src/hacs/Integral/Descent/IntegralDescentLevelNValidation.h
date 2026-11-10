// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_LEVEL_N_VALIDATION_H
#define INTEGRAL_DESCENT_LEVEL_N_VALIDATION_H

// DESCENT D1 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.3 Sec. 4 "D1", report
// DESCENT1_report.md): the level-N integral groups H^q(Gamma_0(N); Sym^n Z^2) by the descent
// (Integral/Descent/DescentLevelN), and the level-1 direct Smith route at ell >= 5.
//
// IntegralDescentLevelNValidation (standard, <= 20 s).  Arms:
//   (T) the independent oracle of record docs/implementation/slot24/descentN_oracle.tsv (descentN_oracle.py: the
//       coinduced module on the Bass--Serre tree, exact ell-adic Smith forms certified by exact ranks; 204 cells,
//       q = 0..5): every cell with D <= 100 here, every cell in the slow name;
//   (R) the closed-form Betti numbers b_0, b_1 == the shipped SparseCokernel at the good prime (counted);
//   (B) [C, prop:badprimesN](2) (C-D2): at every prime 5 <= ell <= n + 2 and every ell | N of sample cells (an
//       ell = N = n + 1 cell among them), the shipped F_ell SparseCokernel's h^0 - b_0 == delta / beta, and the Z/ell^K
//       sparse Smith finds exactly that many cyclic summands (zero at the zero-count primes: arm (Z));
//   (D) three routes at ell >= 5 agree: the sparse elimination, the same generators stacked densely (SmithLevelsEll),
//       and the direct Smith of d^0 of C_eq(M_N) (x) Z/ell^K ([C, prop:descentN](7)(a), (b)); SmithLevelsEll ==
//       elim::invariant_factors on test matrices;
//   (S) [C, prop:descentNthree]'s shortcut at ell = 3 (nu_3(N) = 0) == the dense Iso^(3) route; the gate == its hypothesis;
//   (Q) the Iso^(ell) rank sources (4)(c)(i) and (ii) give the same groups;
//   (L) level 1: AssembleDescentN(1, n) == AssembleDescent(n) (D0) and the direct Smith == the C_7 K-mode at ell >= 5;
//   (N) negative controls: K below the certified exponent (Z/25 at (N, n, ell) = (5, 20, 5)) is not certified and the
//       descent REFUSES with H empty; an RSS ceiling below the prediction REFUSES before the piece starts.
// IntegralDescentLevelNSlowValidation (slow): (T) on every oracle cell; DESCENT1_L1_COST at n in {100, 200, 300}
// (the ell >= 5 piece: K-mode, direct Smith, sparse; serial pinned pool, HECKE_DESCENT_REPEATS rotated repeats,
// default 5); DESCENT1_N_COST on the level-N cells (HECKE_DESCENT_N_CELLS=N:n[:pieces][,...] replaces the default
// cells, pieces a subset of "z32t" (t: the tree at 2, measured only), default all; EFFECTIVE line; RSS predicted and printed before each dense piece, a
// piece above the ceiling is refused, never started).  The hooks are READ-ONLY: they choose cells, never a value.

#include <cstdint>

uint32_t IntegralDescentLevelNValidation();
uint32_t IntegralDescentLevelNSlowValidation();

#endif
