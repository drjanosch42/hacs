// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_FORMS_HECKE_ALGEBRA_VALIDATION_H
#define HECKE_FORMS_HECKE_ALGEBRA_VALIDATION_H

// (hecke-extensions slot 12, WP-7) HeckeFormsHeckeAlgebraValidation (standard):
// the forms-side T_m / ell of S_k(F_ell)_m from q-expansions
// (HeckeExt/HeckeFormsHeckeAlgebra) against the slot-11 WP-20 referee's
// out-of-tree datum ((82, 7): embedding dimension 3; (142, 7): Hilbert
// (1, 3, 2), Loewy length 3) and against the ext driver's cohomological A at
// the same cells (A a quotient of T_m / ell).  Writes nothing; one
// FORMS_ALGEBRA line per compared ideal on stdout.

#include <cstdint>

uint32_t HeckeFormsHeckeAlgebraValidation();

// (hecke-extensions slot 12, WP-14) HeckeBoundaryFinalityValidation (standard):
// the test coverage the WP-13 referee asked of the boundary finality proposal
// (docs/implementation/slot12/WP-12_finality_proposal.md v2).  Same TU as the
// forms validation (it reuses its census reader, the forms-side comparison and
// the finality instance); its own name so the WP-7 / WP-12 name's counts do not
// move.  Arms: (L) Lemma B' tested on the BOUNDARY operators A^bd(T_p) of
// Hecke/HeckeBoundaryOperator (F3); (I) the Eichler-Shimura dimension identity
// dim V = dim T(S_k)_m + dim T(M_k)_m as a gate on four census rows (F9);
// (X) the `mixed` negative control (34, 7) ideal 2 (F12 (c)); (H) under
// HECKE_BOUNDARY_FINALITY_ALL=1 (read-only, not an emitter) every
// sturm_heuristic census row at ell in {5, 7, 11, 13} (F12 (a); the census
// is HECKE_FORMS_CENSUS_TSV, default the tracked edition 1).  Writes nothing.
uint32_t HeckeBoundaryFinalityValidation();

#endif
