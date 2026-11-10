// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 15, WP-8) HeckeLevelNSparseValidation -- the sparse cokernel
// at large N, the level-change and degeneracy maps at composite N, the cost model:
// [C] sec:levelN, rmk:levelcost (4), prop:levelorbit, prop:heilbronn,
// prop:levelcorrespondence (4)(b), (c).
//
// Arms: (S) sparse = dense ENTRYWISE at every dense-reachable cell -- pivots, free
// columns, every RREF row of Rel_N, h^0 and h^1, and every [H_{N,p}] matrix against
// WP-7's dense CokernelPresentation / OnCokernel; (F) the level-change fibre maps on the
// presentation: Gamma-equivariance entrywise, Rel preserved, omega o alpha^0 = index,
// omega o H_{N,q} o alpha^0 = index * H_{M,q} (q !| N) entrywise, and the cross-route
// charpoly of omega o T_q^{(N)} o alpha^0 against the model's tau-route; (D) the degeneracy
// maps on the model: omega o alpha_d = [Gamma_0(dM):Gamma_0(N)] [Gamma_0(M) g_d Gamma_0(M)]
// entrywise, composite d independent of the factor order entrywise, dim H^1 = 2 dim S_k +
// dim Eis_k, and the old-space rank against the classical dimension formulas; (R) (R3) on
// the sparse route against the frozen scan values (26 cells N <= 211, (1321, 11), (1571, 5));
// (C) the cost model against the measured fill / operations, the route choice, the Euler
// characteristic of prop:levelorbit (3) on the ladder.
// Env hook (read-only, not an emitter): HECKE_LEVEL_N_SPARSE_CELLS="N:n:p,..." -- the
// performance ladder over the listed cells (EFFECTIVE line; SPARSE_CELL lines).
// Standard tier; writes nothing; LEVEL_N_SPARSE lines on stdout.
#pragma once
#ifndef HECKE_LEVEL_N_SPARSE_VALIDATION_H
#define HECKE_LEVEL_N_SPARSE_VALIDATION_H

#include <cstdint>

uint32_t HeckeLevelNSparseValidation();

#endif
