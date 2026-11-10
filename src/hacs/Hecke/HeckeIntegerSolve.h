// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_INTEGER_SOLVE_H
#define HECKE_INTEGER_SOLVE_H

// hecke-extensions slot 2, WP-8 (phase E5-1) -- the ONE hand-rolled integer
// linear solver behind every chain fill of the Hecke data ([C, alg:tau]
// step 3 for the bulk tau; [C, lem:etaexists](E) for the pair homotopy eta).
//
// Value-preserving refactor: solve_integer_system() below is the former
// file-local function of HeckeTauTable.cpp, moved here VERBATIM in its
// arithmetic (only the exception texts no longer name HeckeTauTable), so
// that Hecke/HeckePairHomotopy can fill its boundary-edge slots through the
// same solver instead of a second copy. Gated by the golden suite
// (GOLDEN_STRICT tier A/B unchanged) and by
// docs/implementation/tau_table_p2.txt byte-identical
// (HeckeTauTableSelfCheckValidation rewrites it).
//
// Contract ([C, alg:tau] step 3, invariant I5): no HNF/Smith library, no
// arbitrary precision; all arithmetic overflow-checked int64 / __int128, and
// an overflow is a TIER failure, thrown as HeckeOverflow, never a silent
// fallback -- the CALLER decides what a tier failure means (HeckeTauTable
// falls through to the permitted mod-ell solve; HeckePairHomotopy aborts,
// owner decision OP2 / [C, obl:eta](iv)).

#include <cstdint>
#include <vector>

namespace hecke {

// Solve A x = c over Z. Column-echelon reduction by unimodular column
// operations with a tracked transformation U (A U = H): per pivot row,
// Euclid across the remaining columns until at most one nonzero entry
// survives; then forced forward substitution (H y = c has a solution over
// Z iff the greedy pivot division is exact and every pivot-free row has zero
// residual -- complete, since the free components of y multiply zero
// columns of H). x = U y is verified against A x = c exactly before it is
// returned. Returns false iff the system is Z-infeasible; throws
// HeckeOverflow on a checked-arithmetic overflow or a non-terminating
// elimination (both tier failures). A_in is R x n (R rows, every row of
// length n); an empty column set is feasible iff c == 0.
bool solve_integer_system(const std::vector<std::vector<int64_t>>& A_in,
                          const std::vector<int64_t>& c,
                          std::vector<int64_t>& x_out);

// (hecke-extensions slot 6, WP-4; additive) The WIDENED integer tier of
// owner decision OP2 ([OD] OP2, [C, obl:eta](iv)): the SAME column-echelon
// algorithm with every working entry (the reduced A, the transformation U and
// the substituted y) carried in overflow-checked __int128 instead of int64.
// Input and output stay int64 (a chain coefficient is int32 anyway); a
// solution that does not fit int64, or a checked __int128 overflow, is a
// HeckeOverflow -- the tier failure of the widened tier, never a silent
// fallback. Returns false iff the system is Z-infeasible. Reached ONLY by the
// strict-pair tau^s (TauVertexPolicy::BoundaryStrict) after the int64 tier
// above threw, through hecke::escalate_on_overflow; solve_integer_system()
// itself is unchanged, so no shipped table can observe this function.
bool solve_integer_system_wide(const std::vector<std::vector<int64_t>>& A_in,
                               const std::vector<int64_t>& c,
                               std::vector<int64_t>& x_out);

} // namespace hecke

#endif // HECKE_INTEGER_SOLVE_H
