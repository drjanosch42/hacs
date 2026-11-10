// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_MORPHISM_GENERALIZATION_VALIDATION_H
#define HECKE_MORPHISM_GENERALIZATION_VALIDATION_H

#include <cstdint>

// Stage S6 self-check: the per-simplex sparse-block-row generalization of
// EquivariantCochainComplexMorphismOnK (the code-level fix of the "v5 bug",
// [C, rmk:nogo]) must recover the legacy single-constant-fiber-matrix behaviour
// EXACTLY. On a few fixed-seed random 2-local-system pairs on S^1 (plus a couple
// of structured pairs with non-trivial cohomology), this builds the SAME morphism
// two ways -- (1) via the legacy constructor (constant Matrix fiber map +
// identity SimplicialComplexMorphism), and (2) via an explicitly-constructed
// one-entry FiberBlockTable (one PerSimplexBlock (s2, F) per target simplex, with
// s2 found by an INDEPENDENT vertex-list match, not the class's private helper) --
// and asserts get_induced_H_eq_maps() is elementwise identical in ALL degrees.
//
// Returns 0 on success (== number of mismatching cases; 0 = all identical).
uint32_t HeckeMorphismGeneralizationValidation();

#endif // HECKE_MORPHISM_GENERALIZATION_VALIDATION_H
