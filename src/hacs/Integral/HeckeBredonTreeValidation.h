// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_BREDON_TREE_VALIDATION_H
#define HECKE_BREDON_TREE_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 8, WP-8 (Line 2, proposal V-B; [IB] (=
// docs/implementation/integral_line_design_brief.md) Sec. 6.2 V-B, Sec. 2.5(a)).
//
// THE CLAIM (L2-C's Prop. 2.5(a), checked empirically here before the
// contract chapter is written). The 7-vertex model computes Bredon
// cohomology over the ring in force, so for every ring R (here R = F_ell,
// ell in {2, 3}) it is chain-homotopy equivalent to the Bass-Serre TREE
// model: (V_R^S (+) V_R^U -> V_R^{-I}), with S, U generating the tree's
// vertex stabilisers (orders 4 and 6) and -I the edge stabiliser:
//
//     dim H^0 = dim(V^S \cap V^U)
//     dim H^1 = dim V^{-I} - dim(V^S + V^U)
//     dim H^2 = 0
//
// with S = [[0,-1],[1,0]], U = [[0,-1],[1,1]] (S^2 = U^3 = -I in SL(2,Z), so
// V^S, V^U both lie inside V^{-I} and dim H^1 is never negative).
//
// INDEPENDENCE DISCIPLINE (K6, K12). rho_n(S), rho_n(U), rho_n(-I) are built
// by TreeRhoN below, a fresh implementation of [C, def:rhon]'s closed form
//     rho_n(A)_{t,j} = sum_s C(n-j,s) a^{n-j-s} c^s * C(j,t-s) b^{j-t+s} d^{t-s}
// using a self-contained Pascal-triangle table and plain uint64_t mod ell
// arithmetic -- it calls neither ComputeSymmetricPowerMatrixFM nor any
// ModArithmetic::mod_* primitive, so it shares no code with the production
// rho_n route the shipped pipeline (EquivariantCohomologyOnK on
// GenerateSL2ZRelationsWithGroupData) uses. Only FlatMatrixT::rank() (generic
// linear algebra, not the object under test) is shared.
//
// ORACLE (frozen; [IB] Sec. 2.5(a), audit finding B-1's table). The 16
// golden cells' dim_H^{0,1} at ell in {2, 3} (n in the golden grid {0, 1, 2,
// 4, 10, 16, 22, 24}; dim H^2 = 0 throughout). If TreeRhoN's tree-model
// dimensions disagree with this frozen table on any of the 16 cells, the
// validation stops immediately (failing) WITHOUT comparing against the
// shipped pipeline: per [X] plan WP-8, "It does not adjust the model."
//
// SCOPE. Beyond the 16 golden cells, for ell in {2, 3} and every
// n in {0, ..., 40} (even AND odd), the tree-model dimensions are compared
// against the shipped pipeline's dim H^q (EquivariantCohomologyOnK on the
// live 7-vertex model), entrywise, for every degree q the pipeline reports.
//
// Registered `fast` tier. Returns the number of GATING failures (0 = pass),
// per the ValidationFn convention.
uint32_t HeckeBredonTreeValidation();

#endif // HECKE_BREDON_TREE_VALIDATION_H
