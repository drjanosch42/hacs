// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// EngineTrace.h -- process-wide diagnostic counters of the tape engine
// (matrix-engine Phase 4, performance closure)
// =====================================================================
//
// ENGINE_TRACE_NODES=1 makes the executor, the thread pool and compile()
// accumulate, per op and per thread class, the wall time of every pool
// region, the busy time of every task (summed over the workers) and the
// number of nodes / tasks, so the bench driver can print the true serial
// fraction (host time outside any region) and the idle fraction (workers
// waiting inside a region) of a run.  Diagnostic only: nothing here is read
// by a computation, and with the flag unset every hook is one predicated
// branch on a static bool.

#include <cstdint>

namespace engine {
namespace tape {

bool trace_nodes_enabled();   // ENGINE_TRACE_NODES

constexpr uint32_t kTraceOps = 15;      // == number of Op values (Program.h)
constexpr uint32_t kTraceClasses = 3;   // MemoryBound, ComputeBound, Serial (EngineCostModel.h order)

struct TraceTotals {
    // per op (index = static_cast<uint32_t>(Op))
    uint64_t op_nodes[kTraceOps] = {};
    uint64_t op_tasks[kTraceOps] = {};
    double   op_busy[kTraceOps]  = {};     // seconds, summed over the workers
    // per thread class: pool regions issued by the executor
    double   class_wall[kTraceClasses]  = {};   // seconds of region wall (serial: host wall)
    double   class_busy[kTraceClasses]  = {};   // seconds summed over the workers
    uint64_t class_regions[kTraceClasses] = {};
    uint64_t class_thread_sum[kTraceClasses] = {};   // sum over regions of the thread count used
    // executor / compiler
    double   execute_wall = 0.0;   uint64_t runs = 0;
    double   compile_wall = 0.0;   uint64_t compiles = 0;
    // every pool region, whoever issued it
    double   pool_wall = 0.0, pool_busy = 0.0; uint64_t pool_regions = 0, pool_thread_sum = 0;
    double   pool_wait = 0.0;      // seconds callers spent waiting for the region mutex
};

TraceTotals trace_totals();
void reset_trace_totals();
// Accumulators (all no-ops unless trace_nodes_enabled()).
void trace_add_op(uint32_t op, uint64_t nodes, uint64_t tasks, double busy);
void trace_add_class(uint32_t cls, double wall, double busy, uint32_t threads);
void trace_add_execute(double wall);
void trace_add_compile(double wall);
void trace_add_pool_region(double wall, double busy, uint32_t threads, double wait);

} // namespace tape
} // namespace engine
