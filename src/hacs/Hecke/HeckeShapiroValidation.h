// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_SHAPIRO_VALIDATION_H
#define HECKE_SHAPIRO_VALIDATION_H

#include <cstdint>

// Stage S5V -- INDEPENDENT Shapiro validation ([C, obl:shapiro],
// [C, prop:lhcompliant]) of Stage S5's induced local systems L_H
// (Hecke/InducedLocalSystem.h). Oracle O11 of
// docs/implementation/hecke_action_implementation_plan.md, amended by the
// Q2 ruling in Sec. 0.1 (assert, not merely record, dim H^1(Gamma_0(3),
// n=0) = 1 for ell >= 5).
//
// INDEPENDENCE STATEMENT: the expected dimensions asserted/recorded by
// this file are derived from Shapiro's lemma plus classical genus/cusp
// data of the modular curves X_0(2), X_0(3) (see the long derivation
// comment at the top of HeckeShapiroValidation.cpp) -- NOT read off
// HeckeInducedLocalSystemSmokeValidation.cpp (Stage S5's own smoke test),
// which this file's author did not consult for numeric values.
//
// Returns 0 on success (== 0 failing cases):
//   1. L_{Gamma_0(2)}, n=0, ell in {5,101}: HARD-ASSERT dim H^0 = 1 and
//      dim H^1 = 1; dim H^2 recorded (not asserted).
//   2. L_{Gamma_0(2)}, n=2, ell=5: run and RECORD all three dimensions
//      (regression anchor only -- see the .cpp derivation comment for why
//      this case is not independently certifiable to hard-assert).
//   3. L_{Gamma_0(3)}, n=0, ell in {5,101}: HARD-ASSERT dim H^0 = 1 and
//      dim H^1 = 1 (Q2 ruling); dim H^2 recorded.
uint32_t HeckeShapiroValidation();

#endif // HECKE_SHAPIRO_VALIDATION_H
