// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 21, P7) HeckeLambdaQExpValidation: the truncated
// Lambda-adic q-expansions of HeckeExt/HeckeLambdaQExp against an independent
// forms route at further weights (Hida's control), the eta-product of 11a at
// weight 2 and Ramanujan's tau.  Read-only: writes nothing, no golden object,
// no digest of record, no env hook.
#pragma once
#ifndef HECKE_LAMBDA_QEXP_VALIDATION_H
#define HECKE_LAMBDA_QEXP_VALIDATION_H

#include <cstdint>

uint32_t HeckeLambdaQExpValidation();

#endif
