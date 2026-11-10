// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// Executor.h -- runs a compiled Program on the CPU thread pool (Phase 4)
// =====================================================================
//
// run(program, bindings) executes the program's wavefronts in order.  Inside
// a wavefront every node is expanded into tasks that write DISJOINT memory
// (GEMM: output tiles -- never a k-split; batched nodes: instances; memory
// ops: row ranges) and the tasks run on the pool with the per-kernel thread
// count of EngineCostModel.h.  Nothing a task does depends on which thread
// runs it or when, and every node's output is an exact object, so the
// result is bit-identical under any ENGINE_THREADS / ENGINE_BATCH setting.
//
// The executor owns ONE arena (grown to the largest program it has run,
// reused across replays) and one set of result slots per run.  An Executor
// is used by one thread at a time; concurrent pipelines use separate
// executors and share the pool.  Nodes assigned Device::GPU by pass 6 run
// on the executor's DeviceBackend (Phase 5, DeviceBackend.h) when a CUDA
// device is available, and on the CPU otherwise (the assignment never
// affects a value).  With ENGINE_VERIFY_DEVICE (the CMake option, or
// set_verify_device(true)) every GPU node is re-executed on the CPU and the
// two outputs are compared entrywise: a difference throws std::logic_error.

#include <cstdint>
#include <memory>
#include <vector>

#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/DenseView.h"
#include "DeviceBackend.h"
#include "Program.h"
#include "Recorder.h"
#include "ThreadPool.h"

namespace engine {
namespace tape {

struct RunResult {
    std::vector<ResultSlot> slots;
    uint32_t levels = 0, tasks = 0;
    double   seconds = 0.0;
    double   serial_seconds = 0.0;   // wall time spent in host-serial nodes (PANEL_RREF / PIVOT_SEARCH)
};

struct ExecStats {
    uint64_t runs = 0, nodes = 0, tasks = 0, gemm_tiles = 0;
    uint64_t fallback_nodes = 0;    // GPU-assigned nodes run on the CPU (no device)
    uint64_t lean_host_runs = 0;    // (slot-23 P7) programs with GPU memory nodes only, run on the host (ENGINE_GPU_LEAN)
    double   seconds = 0.0;
};

// (slot-23 P7, ENGINE_API.md Sec. 12.9) The VRAM-lean rule. A compiled program whose GPU nodes are all
// memory-class (AXPY, SCAL, TRANSPOSE, SCATTER, GATHER, CANONICALIZE, zero fills: no GEMM-class, BATCHED_GEMM
// or TRSM-General node on the GPU) is run on the host -- exactly the no-device path (GPU-assigned nodes on the
// CPU, transfer nodes skipped) -- unless an elimination session is open on the executor. Such a program does
// no GEMM-class work (memory movement and O(size) modular updates: AXPY, SCAL, scalar / identity TRSM); on the
// device it costs one mirror per operand (slot-22 (13, 700): 3325 route-2 rho blocks, 6.2 GB) and a PCIe round
// trip, for work linear in the bytes it moves. ENGINE_GPU_LEAN (re-read per run): unset / not starting
// with '0' = on, '0...' = off (the rollback arm). Reached only by programs compiled under gpu / hybrid, so
// the cpu golden path never sees it; it decides where bytes are computed, never what they are.
// (S25 LEVERS-25) The env arm ENGINE_GPU_LEAN=0... is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1: EnvLever.h EnvArmRetiredLever); the old route stays via the Scoped override.
bool gpu_lean_enabled();
uint64_t gpu_lean_host_runs();   // process total (diagnostic)

#ifndef ENGINE_VERIFY_DEVICE
#define ENGINE_VERIFY_DEVICE 0
#endif

template<class Elt>
class Executor {
public:
    explicit Executor(ThreadPool* pool = nullptr);   // nullptr: ThreadPool::global()
    ~Executor();

    // Compiles the program if necessary (PassOptions::from_env()) and runs it.
    RunResult run(const Program<Elt>& prog, const Bindings<Elt>& b);
    RunResult run(Program<Elt>& prog, const Bindings<Elt>& b, const PassOptions& opts);

    // Poison the arena before every run (validation: proves no node reads
    // arena memory it did not write).
    void set_poison(bool on) { poison_ = on; }
    const ExecStats& stats() const { return stats_; }
    // Phase 5: the device arm.
    void set_verify_device(bool on) { verify_ = on; }
    bool device_usable() const;                       // a CUDA device is available to this executor
    const DeviceStats& device_stats() const;
    void release_device_state();                      // free every device mirror / graph held by this executor
    size_t device_bytes() const;
    // (slot-18 WP-19a, GPU G1) the elimination session of DeviceBackend.h:
    // residency across the programs of one tape elimination.  begin returns
    // false (and every other call is a no-op) without a usable device.
    bool elim_session_begin();
    bool elim_session_open() const;
    void elim_session_adopt(const DenseBuf<Elt>& host, bool keep);
    void elim_session_release(const Elt* base);
    void elim_session_end(bool download = true);
    ThreadPool& pool() { return *pool_; }
    size_t arena_capacity() const { return arena_.size(); }
    const Elt* arena_base() const { return arena_.data(); }

private:
    struct Impl;
    void execute(const Program<Elt>& prog, const Bindings<Elt>& b, RunResult& out);

    void verify_level(Impl& st, uint32_t level);

    ThreadPool*        pool_;
    std::vector<Elt>   arena_;
    std::vector<DenseBuf<Elt>> own_;   // ENGINE_ARENA=0: per-Temp storage
    bool               poison_ = false;
    bool               verify_ = (ENGINE_VERIFY_DEVICE != 0);
    ExecStats          stats_;
    // (Phase 4 Stage B) created LAZILY -- by device_usable() or by the first run of a
    // program with device work -- so a CPU-only pipeline never creates a CUDA
    // context (110 MB of RSS, memory profile 05 Sec. 0).
    mutable std::unique_ptr<DeviceBackend<Elt>> backend_;
};

} // namespace tape
} // namespace engine
