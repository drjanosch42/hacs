// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_NORMALITY_VALIDATION_H
#define HECKE_LEVEL_NORMALITY_VALIDATION_H
// (hecke-extensions slot 22, LES-1) HeckeLevelNormalityValidation (fast): item (LES1) of [C] v1.22.0-pre.3 rmk:lescheckable --
// the normality classification and the fibre identities of the level-change cone, on the tables (HeckeLevelPair.h).
// Spec docs/implementation/slot22/LES0_spec.md v0.1 (package LES-1); predictions LES_predictions.md section 1 and
// LES1_predictions.md (both committed before this validation first ran).  Arms:
//   (N1) prop:gamma0normal: Gamma_0(MN) normal in Gamma_0(N) iff M | gcd(N, 24), 1 <= N <= 60, 2 <= M <= 24 (1380 pairs);
//        on the normal pairs the quotient is Z/M by gamma -> a c / N mod M (bijective and additive on the coset table);
//   (N2) prop:hidanormalsteps, the general Gamma_1 statements, M | N' <= 24 (60 pairs): the two normal Z/M steps with
//        their quotient maps, and Gamma_1(MN') not normal in Gamma_1(N') (index M^2);
//   (N3) the p-tower table, p in {5, 7}: row 1 (N in {1, 2, 3}), rows 2-4 (r = 1, N in {1, 2, 3}; r = 2, N = 1), rows 5-8
//        (N = 1, r <= 3 at p = 5, r <= 2 at p = 7), Gamma^t_r cap Gamma_0(p^{r+1}) = Gamma_{H_{r+1,r}}(p^{r+1});
//   (N4) rmk:normalN1: Gamma^2, Gamma^3 (CongruenceLevel::Commutator), their index and content, normal with Z/2, Z/3;
//        [Gamma : Gamma^2 cap Gamma^3] = 6; Gamma(M), M in {2, 3, 4, 5, 7}: normal with |G| = |SL_2(Z/M)|;
//   (N5) prop:modeltorsion: the model tags of the eight groups of the predictions; the order-3 criterion on Gamma_0(N),
//        N <= 60; torsion-freeness of Gamma_1(N) (4 <= N <= 24), Gamma(3), Gamma(4), Gamma_1(4) cap Gamma_0(8),
//        Gamma_1(5) cap Gamma_0(25);
//   (L)  lem:levelresmorph (4): RestrictionFibreMap = LevelChangeMap entry for entry (Gamma_0, chi = 1, n in {0, 4}),
//        the trace likewise; TwistedMonomialMatrix = MonomialOf (chi = 1) and = MonomialChiOf (n = 0, chi = omega^e);
//   (F)  obl:levelconeexact (i) at every cell of LES_predictions.md sections 2-3 (the n-sets stated in LES1_predictions.md):
//        alpha M_A = M_B alpha, M_A omega = omega M_B (gamma = S, T, -I), omega alpha = [A : B], the retraction; in the
//        normal case R_sigma M_B = M_B R_sigma, R_sigma alpha = alpha, R_sigma^{ord sigma} = 1, R_sigma R_tau = R_{sigma tau},
//        R_{sigma b} = R_sigma (b in B) and alpha omega = Nm_G;
//   (NEG) obl:levelconeexact (ix), the fibre-level controls: res_wrong_scalar, res_no_block, incompatible_chi (refused
//        les_incompatible_character), sigma_not_normalising (refused les_not_normal; the unchecked formula is not
//        Gamma-equivariant), not_contained (les_not_contained), character_level (les_character_level).
// No env hook; read-only; no golden object, record or setting is touched.
#include <cstdint>

uint32_t HeckeLevelNormalityValidation();

#endif // HECKE_LEVEL_NORMALITY_VALIDATION_H
