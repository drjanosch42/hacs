// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_TAME_FAMILY_VALIDATION_H
#define HECKE_TAME_FAMILY_VALIDATION_H
// (hecke-extensions slot 22, WAKE-1) HeckeTameFamilyValidation (standard): the level Gamma_H(N) (HeckeGammaH.h) and the
// tame-family Hecke algebra of Hsu--Pozzi--Wake--Wang-Erickson, arXiv:2610.03401v1 (HeckeTameFamily.h), at (p, l_0) = (5, 11).
// Spec and predictions: docs/implementation/slot22/WAKE1_spec.md (v0.2, committed before any run).  Oracle (R-22b, printed
// field by field, never called): docs/implementation/slot22/wake/{msymb.py, wake_check.py, wake1_controls.py}.
// Arms:
//   (T)  the table (prop:nebgammaH (2) of the P10-C text): self-checks, the action formula on S, T, ST; H = G_N is
//        Gamma0LevelN's table and H = {1} Gamma1LevelN's entry for entry; PrimeToPart at the WAKE and the P10 levels;
//        N2 (-1 not in H refused by name gammaH_minus_one in the cusp routine);
//   (S)  H = G_N at N = 253 reproduces the shipped Heilbronn route (PresentationDigest, [T_2], [U_11] entrywise); the n = 2
//        blocks equal HeilbronnOperator's at N = 11; RelEscapes = 0 at Gamma_H(33), n = 2, F_7;
//   (M)  the model route at Gamma_H(33) over F_5 (eq:levelcorr with GammaHSubTransversal, the diamond fibre map):
//        dim H^1 and the characteristic polynomials of T_2, U_3, U_11, <2> and a combination equal the cokernel route's;
//   (W)  W-P1, W-P2, W-P3, W-P5 and the wrong-H control N1 (Gamma_0(N), +-Gamma_1(11)) by the m-parts;
//   (H)  the HPWWE cells l_1 = 23, 67 (EFFECTIVE ... arm=HPWWE cells=2): W-P4, W-P4b, W-P6a/b, W-P7, W-P8, W-P8w, the
//        identities I5 - I8 and the controls N3 - N7, each against the oracle's frozen value; l_1 = 43 printed only (INFO).
// Read-only hook: HECKE_TAMEFAMILY_HEAVY=1 adds l_1 = 263 (W-X2, the oracle's survey line; EFFECTIVE ... arm=HEAVY cells=1)
// and the INFO cell l_1 = 197.
#include <cstdint>

uint32_t HeckeTameFamilyValidation();

#endif // HECKE_TAME_FAMILY_VALIDATION_H
