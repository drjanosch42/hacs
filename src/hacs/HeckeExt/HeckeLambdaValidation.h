// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-HL) HeckeLambdaValidation: the truncated
// Lambda-adic ordinary Hecke algebra T^ord(a) (x) Lambda/(ell^K, X^m) of
// HeckeExt/HeckeHidaLambda against the R19-2 prototype's JSON
// (docs/implementation/slot19/r192/data/), [U] 0.3.0-pre.3 sec:u-lambda.
// Read-only: writes nothing, no golden object, no digest of record.
// Default arm: (37, 32), (103, 24); HECKE_LAMBDA_ALL=1 (read-only) adds the
// 691 classes 12 and 200 and the 691 catalogue rows.
#pragma once
#ifndef HECKE_LAMBDA_VALIDATION_H
#define HECKE_LAMBDA_VALIDATION_H

#include <cstdint>

uint32_t HeckeLambdaValidation();

#endif
