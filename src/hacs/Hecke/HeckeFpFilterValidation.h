// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_FP_FILTER_VALIDATION_H
#define HECKE_FP_FILTER_VALIDATION_H

#include <cstdint>

// Stage S1V -- independent validation of the certified FP64/FP32 sign filter
// and its exact-integer fallback (src/hacs/Hecke/Predicates.h/.cpp,
// ExactInt.h/.cpp).
//
// Written from the S1V spec (oracle O13) in
// docs/implementation/hecke_action_implementation_plan.md and the binding
// contract docs/latex/hecke_action.tex ([C, sec:fpfilter]: thm:fp64,
// cor:fptotal, cor:fp32; [C, thm:backend]) ONLY. This is an INDEPENDENT
// validation agent's test: the reference sign computation, RNG streams, and
// adversarial constructions below are derived from the contract's theorem
// statements, not from Hecke/HeckeExactIntSmokeValidation.cpp (the S1
// author's own test), which was not used as a source of truth.
//
// Returns 0 on success (I9 convention); registered in
// ValidationRegistry::RegisterAllValidations().
uint32_t HeckeFpFilterValidation();

#endif // HECKE_FP_FILTER_VALIDATION_H
