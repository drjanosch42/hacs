// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_NEBENTYPUS_WILD_VALIDATION_H
#define HECKE_EXT_HECKE_NEBENTYPUS_WILD_VALIDATION_H
// (hecke-extensions slot 22, P10 (a)) HeckeNebentypusWildValidation (standard): W1 - W4 of
// docs/implementation/slot22/P10_spec.md on the chi_{i,t}-models of Gamma_0(ell^r) over O_{j,e} (Hecke/HeckeNebentypusWild):
// (NB7) the reduction mod pi, (NB8) the lengths of H^1, H^0 and the torsion at i = 0, (NB9) / (U20) the ordinary part free
// of rank 2d + 1 with T_p, U given by the tower's scalars at g -> zeta (HeckeExt/HeckeLambdaTower), and the refusals.
// Cells (t = 1): O_{1, 3(ell-1)} at r = 2, ell in {5, 7, 11, 13} (the even classes of section 7.5); O_{2,40} at (5, r = 3).
// Read-only hook HECKE_NEB_WILD_ALL=1 adds O_{2,42} at (7, r = 3) (EFFECTIVE line; lengths and W3).
#include <cstdint>

uint32_t HeckeNebentypusWildValidation();

#endif // HECKE_EXT_HECKE_NEBENTYPUS_WILD_VALIDATION_H
