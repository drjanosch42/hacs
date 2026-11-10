// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_INDUCED_LOCAL_SYSTEM_H
#define HECKE_INDUCED_LOCAL_SYSTEM_H

// Stage S5 -- Induced local systems L_H on the fixed (K,R) model.
//
// Contract: docs/latex/hecke_action.tex ([C]), [C, def:inducedls] (the
// induced local system L_H = (Ind_H^Gamma V, {M_H(g_r)})), [C, eq:MH] (the
// block matrix M_H(gamma) = sum_i E_{i,i.gamma} (x) rho_n(h_i(gamma)),
// dimension m*(n+1)), [C, prop:lhcompliant] (L_H is SimplexRelation-
// compliant: every M_H(g_r) is invertible and satisfies the cocycle
// condition, so EquivariantCohomologyOnK runs on L_H UNCHANGED), and
// [C, obl:shapiro] (Shapiro consistency -- the S5V oracle).
//
// This is needed only for the Shapiro cross-check (S5V) and the S8
// three-map factorization (alpha : L -> L' = L_{Gamma^0(N)}, omega :
// L'' = L_{Gamma_0(N)} -> L), NOT for the binding Hecke pipeline itself
// (Theta_g of [C, def:xi], S7, never constructs L' or L'').
//
// Both functions reuse EXACTLY the (q_dim, s_lhs, s_rhs, tag) relation list
// and order of GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)
// (LocalSystemSL2Z.h; invariant I3 single source of truth for the model
// group data); only the matrix M is replaced -- M_H(g_r) in place of
// rho_n(g_r). Representation dimension: m * (n_sym + 1),
// m = cosets.num_cosets(). `cosets` must have been constructed with the
// SAME ell and n_sym passed here (checked defensively: BuildMH verifies
// each rho block's prime and dimensions match).
//
// Block sparsity note ([C, sec:implementation] item 2 / Open Question Q8):
// M_H(g_r) has exactly one nonzero (n_sym+1)x(n_sym+1) block per block row
// (block row i, block column i.g_r) -- the natural BlockFlatMatrix dedup use
// case (repeated identical blocks across rows whenever h_i(g_r) coincides).
// This stage assembles a DENSE Matrix instead (correctness first --
// SimplexRelation::M is a plain Matrix, not a BlockFlatMatrix); a
// BlockFlatMatrix-backed fast path is a TODO performance follow-up, not
// implemented here (see the TODO comment at BuildMH in the .cpp).

#include "HeckeCosetTables.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"

#include <cstdint>

namespace hecke {

// L_{Gamma_0(N)}: block matrices M_H(g_r) built from cosets.perm(gen,i) /
// cosets.rho_cocycle(gen,i) (right cosets Gamma_0(N)\Gamma, bottom-row
// P^1(Z/N) class map; [C, prop:cosets] / [C, prop:blockmatrices]).
LocalSystemSimplicialComplex MakeInducedLocalSystemGamma0(
    const SimplicialComplex& K, const HeckeCosetTables& cosets, uint32_t ell, uint32_t n_sym);

// L_{Gamma^0(N)}: block matrices M_H(g_r) built from cosets.perm_upper(gen,i)
// / cosets.rho_cocycle_upper(gen,i) (right cosets Gamma^0(N)\Gamma, top-row
// P^1(Z/N) class map; contract box after [C, prop:blockmatrices]). Used by
// S8's alpha/omega factorization, not by the binding Hecke pipeline.
LocalSystemSimplicialComplex MakeInducedLocalSystemGamma0Upper(
    const SimplicialComplex& K, const HeckeCosetTables& cosets, uint32_t ell, uint32_t n_sym);

} // namespace hecke

#endif // HECKE_INDUCED_LOCAL_SYSTEM_H
