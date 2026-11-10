// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// DeviceBackend.h -- the tape's CUDA execution arm (Phase 5 Stage B)
// =====================================================================
//
// One DeviceBackend per Executor.  For every compiled program it runs it
// keeps a DeviceProgram: one DeviceBuf mirror per program buffer any GPU
// node or transfer node touches (allocated once, so the pointers a captured
// graph records stay valid across replays), the per-node device state
// (GemmBatch instance tables, split-K workspaces, scratch for scaled
// products, device copies of the GATHER index lists), and one CUDA graph
// per wavefront that has GPU nodes.  A run:
//
//   begin_run(prog, host_bufs)   rebind the mirrors to this run's host buffers
//                                (Inputs / Outputs: host newer; Temps: none)
//   per level L:
//     transfers(L)               the H2D / D2H nodes of L (staged copies,
//                                stream-ordered before the level's work)
//     launch(L)                  enqueue the GPU nodes of L: round-robin over
//                                the backend's three streams with fork / join
//                                events; direct launches on the first run of a
//                                program, a captured graph replayed afterwards.
//                                Consecutive GPU-only levels (no CPU node, no
//                                D2H) are one RUN: enqueued, captured and
//                                synchronised as a unit (stream order carries
//                                the inter-level dependencies)
//     [CPU tasks of L run here]
//     wait(L)                    host waits for the level (at the end of its
//                                run); the mirrors of the buffers the GPU wrote
//                                become device-newer
//     host_wrote(L)              the mirrors of the buffers the CPU wrote
//                                become host-newer
//   end_run()
//
// Residency at run time is the DeviceBuf version pair; the static plan of
// pass 7 is asserted against it (an H2D of a device-newer buffer is a plan
// error), and a transfer whose buffer is already Both is a no-op (that
// happens after an ENGINE_VERIFY_DEVICE re-execution).
//
// Determinism: every kernel here is the exact-integer twin of the CPU kernel
// (ENGINE_API.md Sec. 11), so a node's output bytes do not depend on the
// device; ENGINE_VERIFY_DEVICE re-executes every GPU node on the CPU and
// memcmps (Executor.cpp), and EngineCrossDeviceValidation gates it.
//
// Everything is a no-op when engine::cuda::available() is false: the
// executor then runs GPU-assigned nodes on the CPU (usable() == false).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/ModCtx.h"
#include "Program.h"

namespace engine {
namespace tape {

struct DeviceStats {
    uint64_t runs = 0;              // runs with device work
    uint64_t gpu_nodes = 0;         // GPU nodes executed (batches count once)
    uint64_t gpu_levels = 0;
    uint64_t merged_levels = 0;     // levels folded into a preceding GPU-only run (one graph, one sync)
    uint64_t direct_launches = 0;   // levels enqueued kernel by kernel
    uint64_t graph_launches = 0;    // levels replayed from a captured graph
    uint64_t captures = 0;
    uint64_t h2d = 0, d2h = 0, skipped_transfers = 0;
    uint64_t h2d_bytes = 0, d2h_bytes = 0;
    uint64_t programs_built = 0, programs_evicted = 0;
    uint64_t verified_nodes = 0;
    double   transfer_seconds = 0.0;   // host time inside transfers()
    double   wait_seconds = 0.0;       // host time inside wait()
    double   launch_seconds = 0.0;     // host time inside launch()
    // (slot-18 WP-19a, GPU G1) the elimination session: mirrors created for
    // adopted host buffers, planned H2D skipped because the session mirror
    // was already device-newer, epilogue D2H deferred (no CPU reader later in
    // the program), begin-of-run downloads for a CPU access, and the
    // downloads at release / session end.  Diagnostic, never on a value path.
    uint64_t sessions = 0, session_mirrors = 0, session_skipped_h2d = 0, session_deferred_d2h = 0;
    uint64_t session_syncs = 0, session_release_d2h = 0, session_release_d2h_bytes = 0;
    // (slot-22 G1) GPU GEMM-class and BATCHED_GEMM nodes given an IMMA workspace at
    // program build (eligible modulus, supported device, ENGINE_GPU_IMMA on); the IMMA blocks
    // allocated (one per stream slot, grown to powers of two); the cached programs whose graphs a
    // block growth dropped (re-captured on their next run). Diagnostic.
    uint64_t imma_nodes = 0, imma_blocks = 0, imma_graph_drops = 0;
};

template<class Elt>
class DeviceBackend {
public:
    DeviceBackend();
    ~DeviceBackend();
    DeviceBackend(const DeviceBackend&) = delete;
    DeviceBackend& operator=(const DeviceBackend&) = delete;

    static bool usable();   // engine::cuda::available()

    // Per run.  `host` = the executor's per-run DenseBufs (one per program
    // buffer, in BufId order).  Returns false when the program has no device
    // work (no GPU node, no transfer node).
    bool begin_run(const Program<Elt>& prog, std::vector<DenseBuf<Elt>>& host, const ModCtx<Elt>& ctx, bool verify);
    void transfers(uint32_t level);
    void launch(uint32_t level);
    void wait(uint32_t level);
    void host_wrote(uint32_t level);
    void end_run();

    bool level_has_gpu(uint32_t level) const;
    const std::vector<uint32_t>& gpu_nodes(uint32_t level) const;

    // ENGINE_VERIFY_DEVICE support (Executor.cpp drives the re-execution):
    //   verify_sync_host(L)  before launch(L): every buffer a GPU node of L
    //                        touches is made current on the host
    //   verify_snapshot(L)   after wait(L): the device copy of every buffer a
    //                        GPU node of L wrote (whole footprint)
    //   verify_mark_synced   after the comparison passed
    void verify_sync_host(uint32_t level);
    void verify_snapshot(uint32_t level, std::vector<std::pair<BufId, DenseBuf<Elt>>>& out);
    void verify_mark_synced(uint32_t level);

    // (slot-18 WP-19a, GPU G1) Elimination session: residency across the
    // programs of ONE tape elimination (ENGINE_API.md Sec. 12.8).  An ADOPTED
    // host buffer (keyed by its base pointer) gets one session mirror shared
    // by every program that binds it (or a prefix window of it: same base,
    // ld and layout) as an Input or Output; its version pair carries across
    // programs.  A planned H2D of a device-newer session buffer is skipped (no
    // host write happened: the caller's promise, see below), a D2H with no CPU
    // access later in the program is deferred, and a program with a CPU node
    // touching a device-newer session buffer downloads it at begin_run.  The
    // caller promises: no HOST code touches an adopted buffer between
    // session_adopt and session_release except through tape programs.
    // `keep`: session_release / session_end download a device-newer mirror
    // into the host buffer; otherwise the device bytes are dropped (scratch).
    void session_begin();
    bool session_open() const;
    void session_adopt(const DenseBuf<Elt>& host, bool keep);
    void session_release(const Elt* base);
    void session_end(bool download = true);
    // A run the executor keeps entirely on the host while a session is open
    // (no device work, or begin_run returned false): the adopted buffers it
    // binds are made current on the host, and those it writes host-newer.
    void session_host_run(const Program<Elt>& prog, std::vector<DenseBuf<Elt>>& host);

    void release();                 // drop every device program (frees mirrors, graphs, workspaces)
    size_t device_bytes() const;    // bytes currently held by the cached programs
    const DeviceStats& stats() const;

    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};

} // namespace tape
} // namespace engine
