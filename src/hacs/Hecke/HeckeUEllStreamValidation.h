// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_U_ELL_STREAM_VALIDATION_H
#define HECKE_U_ELL_STREAM_VALIDATION_H

#include <cstdint>

// (hecke-extensions slot 14, WP-10) The coset streaming lever
// (HeckeOperatorOptions::ScopedCosetStreaming) and the serial phase on the pool
// (ScopedParallelSetup): bit-identity with and without each lever, and composed
// with the slot-13 U_ell lever.  `slow`.  0 on success.  (S24 CL-R1) The levers
// are retired: the definition is Hecke/legacy/HeckeUEllStreamValidation.cpp, an
// OC_ENABLE_LEGACY=ON source; registered through OC_LEGACY_VALIDATION.
uint32_t HeckeUEllStreamValidation();

#endif // HECKE_U_ELL_STREAM_VALIDATION_H
