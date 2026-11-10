// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 15, WP-7) HeckeLevelNHeilbronnValidation -- the
// Heilbronn--Merel action H_{N,p} on the cokernel presentation, and (LN3)
// closed on both halves: [C] sec:levelN, lem:merelpath, prop:heilbronn,
// rmk:heilbronnconv, rmk:levelNcheckable (LN3).
//
// Arms: (M) Merel's set -- the counts of rmk:heilbronncost (1), |Mer_p| <= p^2 + 1,
// the path condition of lem:merelpath on each of the p + 1 lattices of index p;
// (L) (LN3) at every cell: H_{N,p}(Rel_N) in Rel_N entrywise, dim M_N / Rel_N =
// dim H^1 of the model, and the characteristic polynomial of [H_{N,p}] on
// M_N / Rel_N = that of prop:levelcorrespondence (4)(d) (the tau-route, eq:levelcorr
// with def:cosetrepsN's pinned r_t) on H^1_eq(K, Rel; L_N), per operator and for a
// linear combination and a product of the family; the family commutes; p !| N and
// p | N, prime and composite N with mixed classes, F_P (characteristic-0 surrogate),
// F_ell with ell | N and ell !| N; (K) the classical charpolys of (LN3)'s list on
// the Heilbronn side; (N1) prop:heilbronn (3) at N = 1; (E) the N = ell plug point:
// Heilbronn on Gamma0LevelN(ell) against the tau-route on the SHIPPED level-ell
// table (HeckeCosetTables, slot-13 HeckeLevelEll); (Z) rmk:heilbronnconv (2)'s two
// mis-transcriptions as negative controls (both leave Rel_N at the four named cells;
// (a) = eq:heilbronn at weight 2).
// Standard tier; writes nothing; LEVEL_N_HEILBRONN lines on stdout.
#pragma once
#ifndef HECKE_LEVEL_N_HEILBRONN_VALIDATION_H
#define HECKE_LEVEL_N_HEILBRONN_VALIDATION_H

#include <cstdint>

uint32_t HeckeLevelNHeilbronnValidation();

#endif
