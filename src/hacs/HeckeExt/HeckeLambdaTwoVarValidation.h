// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_LAMBDA_TWOVAR_VALIDATION_H
#define HECKE_EXT_HECKE_LAMBDA_TWOVAR_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 21, P9) the gate of HeckeExt/HeckeLambdaTwoVar: the two-variable
// measure mu(X; b + ell^M) and L_a(X, T) mod (ell^K, (X - x_c)^m, omega_{M-1}(T)) by
// interpolation of P8's normalised masses (docs/implementation/slot21/P9_spec.md).
// Read-only; env arms HECKE_LAMBDA_TWOVAR_ALL=1 / HECKE_LAMBDA_TWOVAR_CELLS (read-only,
// with an EFFECTIVE line).  Returns 0 iff every gate of kind PROVED holds.
uint32_t HeckeLambdaTwoVarValidation();

#endif  // HECKE_EXT_HECKE_LAMBDA_TWOVAR_VALIDATION_H
