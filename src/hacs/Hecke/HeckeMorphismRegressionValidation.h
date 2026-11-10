// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_MORPHISM_REGRESSION_VALIDATION_H
#define HECKE_MORPHISM_REGRESSION_VALIDATION_H

#include <cstdint>

// Stage S6V -- INDEPENDENT validation agent for Stage S6 (the per-simplex
// sparse-block-row generalization of EquivariantCochainComplexMorphismOnK,
// the code-level fix of the "v5 bug", [C, rmk:nogo]).
//
// PURPOSE: prove S6's refactor did NOT change any EXISTING induced-cohomology
// map that predates the refactor. This is a pure regression check -- it makes
// no independent mathematical claim about what the induced maps *should* be
// (that is S7/S8's job); it only asserts the refactor is behaviour-preserving.
//
// ORACLE, part 1 (cross-commit bit-identity): get_induced_H_eq_maps() (all
// degrees) of
//   (a) the three LocalSystemPresets S1 morphisms exercised by
//       LocalSystemCohomologyValidation() (CreateS1_UnipotentToTrivialMorphism,
//       CreateS1_TrivialToUnipotentMorphism) plus CreateS1_IdentityMorphism
//       (declared in LocalSystemPresets.h but not itself exercised by that
//       validation -- included here anyway for completeness, since it is
//       part of "the S1 preset morphisms" the oracle names);
//   (b) TestSL2Z_To_S1Cusp_RestrictionMorphism(n, ell)'s internal restriction
//       morphism, for (n, ell) in {(8,5), (10,5), (10,691), (100,5)},
// must be BIT-IDENTICAL to reference values captured on the PARENT (pre-S6)
// commit.
//
// CAPTURE METHOD (documented here and re-printed by the validation's stdout
// at runtime for auditability):
//   Parent commit:  41df5e278460f4c1a9f7ee3270878052a0802983
//                   ("S5 + S5V: induced local systems L_H on the fixed
//                   (K,R) (independently validated)", branch hecke-action).
//   S6 tree:        same commit as HEAD (41df5e278460f4c1a9f7ee3270878052a0802983)
//                   PLUS Stage S6's uncommitted working-tree edits, confined
//                   to EquivariantCochainComplexMorphismOnK.{h,cpp},
//                   ValidationRegistry.cpp and CMakeLists.txt (verified by
//                   `git diff 41df5e2 --stat` at the time this file was
//                   written -- no other tracked file differs).
//
//   Commands run by the S6V validation agent:
//     git worktree add --detach <scratch>/parent-worktree 41df5e2
//     # In the worktree ONLY (never committed, worktree removed afterwards):
//     #   add Hecke/S6VCapture.{h,cpp} (a throwaway driver, NOT part of the
//     #   permanent codebase) that reconstructs, via ONLY the PUBLIC API
//     #   (LocalSystemPresets::*, CreateSL2ZFundamentalDomainTriangulation,
//     #   ComputeSymmetricPowerMatrix, GenerateSL2ZRelations_7VertexModel --
//     #   NEVER TestSL2Z_To_S1Cusp_RestrictionMorphism itself, which is void
//     #   and cannot hand back its internal morphism object), the exact same
//     #   sequence of calls TestSL2Z_To_S1Cusp_RestrictionMorphism makes
//     #   internally, and dumps get_induced_H_eq_maps() to stdout in a
//     #   simple "CASE/NDEG/DEG ROWS COLS/<rows of entries>/ENDCASE" format;
//     #   registered as "S6VCapture" in that worktree's ValidationRegistry.
//     cmake -DHECKE_PURE_INTEGER=OFF . && make -j
//     ./cohomology_app S6VCapture > s6v_capture_raw.txt
//     git worktree remove <scratch>/parent-worktree
//   The captured values were then embedded VERBATIM as the kReference* tables
//   in HeckeMorphismRegressionValidation.cpp (a literal transcription, not a
//   runtime file read, so the check has no external-file dependency at
//   `./cohomology_app` run time). This file's reconstruction functions
//   (build_preset_*, build_sl2z_restriction) independently replicate, on the
//   S6 TREE, the identical public-API call sequence used by the capture
//   driver -- so a match certifies S6's legacy code path is bit-identical to
//   the parent, and any mismatch is a genuine regression (reported with the
//   offending case name, degree, and differing matrix entries; NOT patched
//   by this file).
//
// ORACLE, part 2 (S6-tree-only equivalence, 100 fixed seeds): builds the same
// morphism two ways on the S6 tree -- legacy ctor (constant Matrix fiber map +
// SimplicialComplexMorphism) vs. an explicitly-constructed one-entry
// FiberBlockTable -- for 100 fixed-seed random 2-dimensional local-system
// pairs on S^1, and asserts get_induced_H_eq_maps() are elementwise identical
// in every degree. (This is in addition to, and at 100 seeds a superset in
// scale of, Stage S6's OWN self-check HeckeMorphismGeneralizationValidation
// -- run as a SEPARATE registered validation -- and re-implements the
// comparison independently in this file rather than calling into it.)
//
// Returns 0 on success (== 0 failing cases across both oracle parts).
uint32_t HeckeMorphismRegressionValidation();

#endif // HECKE_MORPHISM_REGRESSION_VALIDATION_H
