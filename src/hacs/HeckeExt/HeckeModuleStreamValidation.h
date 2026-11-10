// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, G2-A / G2-B; docs/implementation/slot23/G2_plan.md Sec. 2.7)
// HeckeModuleStreamValidation (standard) and HeckeModuleStreamSlowValidation (slow): the module route's
// stream levers (HECKE_MODULE_STREAM) against the legacy route, in process (engine::elim::ScopedModuleStream),
// entry for entry and byte for byte.  The cpp's header comment lists the arms.
#pragma once

#include <cstdint>

uint32_t HeckeModuleStreamValidation();
uint32_t HeckeModuleStreamSlowValidation();
uint32_t HeckeModuleMtValidation();   // (S24 MT-1) HECKE_MODULE_MT at pools 1 / 2 / 7 / 16 (Registry/Slot24/MT1.cpp)
uint32_t HeckeSatPerfValidation();    // (S24 SATPERF) HECKE_SAT_PERF at pools 1 / 2 / 4 vs the rollback (Registry/Slot24/SATPERF.cpp)
