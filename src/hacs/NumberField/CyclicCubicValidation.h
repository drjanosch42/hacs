// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// NumberFieldCyclicCubicValidation (standard; slot-23 C3, the cyclic cubic field of prime conductor ell, h_K by the
// cyclotomic-unit index and the set Lambda-bar of arXiv:2610.17865: CyclicCubic.h). Compiled into cohomology_app,
// not into the STL-only NumberField library. Returns the number of failing checks (0 = pass).
#include <cstdint>

uint32_t NumberFieldCyclicCubicValidation();
