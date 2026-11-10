// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_INDUCED_LOCAL_SYSTEM_SMOKE_VALIDATION_H
#define HECKE_INDUCED_LOCAL_SYSTEM_SMOKE_VALIDATION_H

#include <cstdint>

// Stage S5 smoke validation (S5 Definition of Done; the INDEPENDENT Shapiro
// dimension oracle ([C, obl:shapiro]) is S5V's HeckeShapiroValidation).
// Returns 0 on success:
//   1. HeckeCosetTables::self_check() and the NEW self_check_upper()
//      (Gamma_0(N) resp. Gamma^0(N) DoD batteries) both pass for the coset
//      table used below (p_hecke=2, r=1);
//   2. MakeInducedLocalSystemGamma0(K, cosets, ell, n_sym) builds, every
//      relation matrix M_H(g_r) is invertible ([C, prop:lhcompliant]),
//      EquivariantCohomologyOnK runs UNCHANGED on it (construction,
//      compute_cohomology(), dimension getters all succeed, returning a
//      dimension vector of length dim(K)+1 = 3), for
//      (n_sym, ell) in {(0,5), (2,5)};
//   3. (extra robustness, not part of the official S5 DoD but exercised here
//      to de-risk S8, which consumes the Gamma^0(N) variant): the same for
//      MakeInducedLocalSystemGamma0Upper at (n_sym, ell) = (0,5).
uint32_t HeckeInducedLocalSystemSmokeValidation();

#endif // HECKE_INDUCED_LOCAL_SYSTEM_SMOKE_VALIDATION_H
