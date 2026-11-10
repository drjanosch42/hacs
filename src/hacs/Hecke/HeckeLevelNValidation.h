// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 14, WP-12) HeckeLevelNValidation -- level N on the
// fixed model, first build: [C] v1.14.0 sec:levelN (def:cosetrepsN,
// lem:cosetrepsN, def:levelNmodel, rmk:levelNcheckable (LN1), (LN2), the tau
// half of (LN3), (LN4)-(LN6) at the R3 cells within the dense route's reach)
// and sec:levelshapiro (prop:levelcorrespondence, prop:levelorbit).
//
// Arms: (T) the composite-N table (every clause of lem:cosetrepsN, prime powers
// = the shipped HeckeCosetTables, an independent P^1(Z/N) test by cross
// products); (D) (LN1) and prop:levelorbit (3) on the model; (O) (LN2)
// orbit-wise = dense fixed spaces, the cokernel dims = the model's; (X) the
// level-N operators are cochain maps into C_eq; (H) the Hecke ring at level N
// (commutativity, independence of the pinned r_t); (G) an INDEPENDENT group-
// cocycle route extended to composite N (own P^1(Z/N) table, own transversals,
// no tau, no model) -- dims and characteristic polynomials; (K) classical
// eigenvalues; (E) the plug-point regression at N = ell (and ell^2); (R3)
// Mazur's Eisenstein ideal at weight 2, prime N, F_p, p !| N: dim W_{N,p},
// Loewy length, Merel's criterion, against the frozen WP-4 scan.
// Standard tier; writes nothing; LEVEL_N / LEVEL_N_R3 lines on stdout.
#pragma once
#ifndef HECKE_LEVEL_N_VALIDATION_H
#define HECKE_LEVEL_N_VALIDATION_H

#include <cstdint>

uint32_t HeckeLevelNValidation();

#endif
