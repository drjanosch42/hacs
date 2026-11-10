// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineTrace.cpp -- see EngineTrace.h.
#include "EngineTrace.h"

#include <cstdlib>
#include <mutex>

namespace engine {
namespace tape {

namespace {
std::mutex& trace_mutex() { static std::mutex m; return m; }
TraceTotals& totals() { static TraceTotals t; return t; }
}

bool trace_nodes_enabled() {
    static const bool on = std::getenv("ENGINE_TRACE_NODES") != nullptr;
    return on;
}

TraceTotals trace_totals() { std::lock_guard<std::mutex> lk(trace_mutex()); return totals(); }
void reset_trace_totals() { std::lock_guard<std::mutex> lk(trace_mutex()); totals() = TraceTotals(); }

void trace_add_op(uint32_t op, uint64_t nodes, uint64_t tasks, double busy) {
    if (!trace_nodes_enabled() || op >= kTraceOps) return;
    std::lock_guard<std::mutex> lk(trace_mutex());
    TraceTotals& t = totals();
    t.op_nodes[op] += nodes; t.op_tasks[op] += tasks; t.op_busy[op] += busy;
}
void trace_add_class(uint32_t cls, double wall, double busy, uint32_t threads) {
    if (!trace_nodes_enabled() || cls >= kTraceClasses) return;
    std::lock_guard<std::mutex> lk(trace_mutex());
    TraceTotals& t = totals();
    t.class_wall[cls] += wall; t.class_busy[cls] += busy; ++t.class_regions[cls]; t.class_thread_sum[cls] += threads;
}
void trace_add_execute(double wall) {
    if (!trace_nodes_enabled()) return;
    std::lock_guard<std::mutex> lk(trace_mutex());
    totals().execute_wall += wall; ++totals().runs;
}
void trace_add_compile(double wall) {
    if (!trace_nodes_enabled()) return;
    std::lock_guard<std::mutex> lk(trace_mutex());
    totals().compile_wall += wall; ++totals().compiles;
}
void trace_add_pool_region(double wall, double busy, uint32_t threads, double wait) {
    if (!trace_nodes_enabled()) return;
    std::lock_guard<std::mutex> lk(trace_mutex());
    TraceTotals& t = totals();
    t.pool_wall += wall; t.pool_busy += busy; ++t.pool_regions; t.pool_thread_sum += threads; t.pool_wait += wait;
}

} // namespace tape
} // namespace engine
