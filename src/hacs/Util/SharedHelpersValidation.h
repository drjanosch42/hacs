// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#include <cstdint>

// S24 CL-1 (docs/implementation/slot24/CLEANUP_plan_1030.md): the shared helpers ScalarMod.h, ModCtx::chain,
// EnvLever.h, Engine/Threads.h, CpuFeatures.h and Util/{ValidationCheck,CellList,TextIo,Timing,ParallelFor}.h reproduce
// the copies they replace -- against those copies wherever one is callable, else against a table transcribed from them.
// `fast`.  Read-only: it sets and restores only its own probe variables and ENGINE_THREADS / ENGINE_GPU_IMMA around
// calls that re-read them per call (no cached reader is touched).
uint32_t SharedHelpersValidation();
