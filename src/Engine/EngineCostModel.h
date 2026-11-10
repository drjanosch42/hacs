// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// EngineCostModel.h -- THE ONE TABLE of measured constants (Phase 4, D4)
// =====================================================================
//
// Every number the planner (pass 6, device assignment) and the executor
// (per-kernel thread count) consult lives here, initialised from the
// measured figures of docs/matrix-engine/04-hardware-limits-and-prior-art.md
// (Secs. 1.3, 1.4, 1.5, 6.2, 6.4, 7.5) and ENGINE_API.md Sec. 6.  Re-measure,
// edit here, nothing else moves.  None of these constants can change a
// computed value: they steer WHERE and with HOW MANY threads a node runs,
// never WHAT it computes (D4 determinism guarantee).

#include <cstdint>

namespace engine {
namespace tape {

struct CostConstants {
    // Arithmetic rates, modular multiply-adds per second.
    double cpu_modmac_1t     = 27.0e9;    // ENGINE_API Sec. 6: gemm<uint32_t> 128..512, one thread, AVX-512
    double cpu_modmac_16t    = 3.35e11;   // ENGINE_API Sec. 10.8: dense 2048^3 through the tape, 16 threads (335 G)
    double cpu_modmac_16t_small = 2.44e11; // ENGINE_API Sec. 10.8: Theta-like 2000 x 129^3 batched, 16 threads (244 G)
    double cpu_modmac_u64_1t = 1.2e9;     // 04 Sec. 1.5 regime (e), one core
    double gpu_modmac        = 4.0e12;    // ENGINE_API Sec. 11.6: u32 gemm 2048^3, 82 % of the roofline (4064 G)
    double gpu_modmac_small  = 2.0e12;    // ENGINE_API Sec. 11.6: batched 2000 x 129^3, one launch (2067 G)
    double gpu_modmac_shoup  = 1.6e12;    // ENGINE_API Sec. 11.6: 2^31-1 (k_max = 4), 1628 G
    double gpu_modmac_barrett64 = 5.2e11; // ENGINE_API Sec. 11.6: 2^32-5 (k_max = 1), 523 G
    double gpu_modmac_u64    = 2.7e11;    // ENGINE_API Sec. 11.6: u64 gemm at 2^61-1, 274 G
    // Memory.
    double cpu_dram_bytes_s  = 37.6e9;    // 04 Sec. 1.3: DRAM STREAM triad
    double cpu_l3_bytes_s    = 829e9;     // 04 Sec. 1.3: L3-resident, 16 threads
    double pcie_bytes_s      = 55e9;      // 04 Sec. 1.4 / 6.4: pinned PCIe 5.0 (direct, pinned both ends)
    double h2d_bytes_s       = 18.0e9;    // ENGINE_API Sec. 11.6: pageable -> pinned pool -> device, 4 copy threads
    double d2h_bytes_s       = 14.4e9;    // ENGINE_API Sec. 11.6: device -> pinned pool -> pageable
    // Per-node overheads.
    double gpu_launch_s      = 0.8e-6;    // ENGINE_API Sec. 11.6: one node inside a CUDA graph (0.80 us)
    double gpu_stream_launch_s = 6.0e-6;  // ENGINE_API Sec. 11.6: one async stream launch (first, uncaptured run)
    double gpu_kernel_floor_s = 8.0e-6;   // ENGINE_API Sec. 11.7: a small (129^3) kernel is latency-bound at ~8 us
    double cpu_node_s        = 1.0e-6;    // thread-pool dispatch of one task (order of magnitude)
    double d2h_call_s        = 30.0e-6;   // one staged download: chunk handshake + event sync (measured order)
    double h2d_call_s        = 10.0e-6;   // one staged upload
    // Elimination crossover (04 Sec. 6.4): GPU pays when the rank r >~ 45.
    double gpu_rank_crossover = 45.0;
    // Threshold for the "dense" GPU rate: >= this many 64x64 output tiles
    // fills the 70 SMs twice (cuda::suggest_splits uses the same figure).
    double gpu_dense_tiles   = 140.0;
};

inline const CostConstants& cost_constants() {
    static const CostConstants c;
    return c;
}

// Per-kernel thread count (04 Sec. 7.5, plan D4): 16 = one per core for
// memory-bound kernels (SMT hurts L3 bandwidth: 477 vs 829 GB/s), 32 for
// compute-bound ones (1023 vs 847 GFLOP/s).  The executor clamps to the pool
// size (ENGINE_THREADS).
enum class ThreadClass : uint8_t { MemoryBound, ComputeBound, Serial };

constexpr uint32_t kThreadsMemoryBound  = 16;
constexpr uint32_t kThreadsComputeBound = 32;
constexpr uint32_t kMaxWorkers          = 32;

} // namespace tape
} // namespace engine
