// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_LAMBDA_MODSYM_VALIDATION_H
#define HECKE_EXT_HECKE_LAMBDA_MODSYM_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 21, P8) the gate of HeckeExt/HeckeLambdaModSym: ordinary
// cohomology along the tower as a Lambda-module, truncated mod (ell^K, X^m)
// ([U] v0.5.0-pre.2 part 4, sec:u-msym; docs/implementation/slot21/P8_spec.md v0.2).
// Read-only; env arms HECKE_LAMBDA_MSYM_ALL=1 / HECKE_LAMBDA_MSYM_CELLS (read-only,
// with an EFFECTIVE line).  Returns 0 iff every gate of kind PROVED holds.
uint32_t HeckeLambdaModSymValidation();

#endif  // HECKE_EXT_HECKE_LAMBDA_MODSYM_VALIDATION_H
