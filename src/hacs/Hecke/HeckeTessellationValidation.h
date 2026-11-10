// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_TESSELLATION_VALIDATION_H
#define HECKE_TESSELLATION_VALIDATION_H

#include <cstdint>

// Stage S2V -- INDEPENDENT validation of Stage S2
// (src/hacs/Hecke/SL2ZTessellationTables.{h,cpp} and the
// group-data extension of LocalSystemSL2Z.{h,cpp}), written from the
// mathematical spec of docs/implementation/hecke_action_implementation_plan.md
// ("### S2V") and the contract docs/latex/hecke_action.tex ([C, def:primitives]
// (P2)/(P4), [C, sec:model]) WITHOUT reading HeckeTessellationSmokeValidation.cpp
// or SL2ZTessellationTables.cpp's internal logic for expected numbers. All
// expected stabilizer orders, transporter sets and orbits below were derived
// by hand from the 8-row relation/generator table of [C, sec:model] (see the
// derivation comments in the .cpp) and are hard-coded as a committed expected
// table, independent of the S2 implementation.
//
// Returns 0 on success. Implements ALL 5 S2V oracle groups:
//   1. Stabilizer orders for every simplex (vertices, edges, triangles) of
//      the 7-vertex model, compared exactly against
//      SL2ZTessellationTables::transporter(q,s,s).size().
//   2. Cross transporter sets S(v3,v0), S({3,5},{0,4}), S({4,6},{5,6}),
//      compared by exact Mat22 set equality.
//   3. Group-element sanity: det(g) == 1 for every g in every non-empty
//      transporter set; same_groupoid_orbit induces exactly the
//      hand-derived orbits (vertices, edges, triangles).
//   4. Relation-list structure (updated for FIX-negI, the central -I fix):
//      GenerateSL2ZRelations_7VertexModel and
//      GenerateSL2ZRelationsWithGroupData agree row-by-row on the FULL list
//      for (n, ell) in {(0,5), (2,5), (10,691)}; the FIRST 8 rows are the
//      original (pre-fix) seed relations with g_r sequence
//      {T,T,T,S,S,S,ST,TS}; the list has exactly 8 + N_simplices (= 31)
//      rows, the trailing 23 being -I self-relations (s_lhs == s_rhs,
//      g_r = -I, M == rho_n(-I)) covering every simplex of every dimension
//      exactly once ([C, thm:explicit_complex_equiv]: -I acts trivially on
//      H, so f(sigma) = rho_n(-I) f(sigma) is required for EVERY sigma);
//      each relation matrix equals an independently recomputed
//      rho_n(g_r) = ComputeSymmetricPowerMatrixFM(g_r mod ell, n)
//      (invariant I3).
//   5. High-level cohomology unchanged: an independently constructed
//      EquivariantCohomologyOnK from GenerateSL2ZRelationsWithGroupData(K,10,5)
//      has dim H^0 = 0, dim H^1 = 3, dim H^2 = 0 (Eichler-Shimura:
//      2*dim S_12 + 1 = 3, dim S_12 = 1); TestSL2ZCohomologySymmetricPower(10,5)
//      and TestSL2Z_To_S1Cusp_RestrictionMorphism(10,5) additionally run
//      without throwing.
uint32_t HeckeTessellationValidation();

#endif // HECKE_TESSELLATION_VALIDATION_H
