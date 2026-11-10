// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// (slot-24 PAR-GPU) The opt-in GPU route of the 2-adic Smith form's LU (ring2k::echelon_blocked, rungs u32 / u64): the
// policy and the installation of src/ModArithmetic/Binary/Ring2kDevice.h's operations table over EngineCuda (CudaRing2k.h).
// docs/implementation/slot24/PARGPU_report.md.
//
// WHEN THE DEVICE RUNS.  All of:
//   * the calling thread's device policy is ENGINE_DEVICE=gpu (engine::tape::elim_pass_options(), so a
//     ScopedThreadDevice / ScopedPassOptions steers it exactly as the G2 device elimination);  the default (cpu) never
//     reaches CUDA;
//   * ENGINE_GPU_SMITH2K is not "0" (the rollback arm; read once; anything but 0 / 1 warns and stays on);
//   * m n >= ENGINE_GPU_SMITH2K_MIN entries (default 2^24, about 4096 x 4096; below it the host LU is as fast);
//   * engine::cuda::available();
//   * the device has memory for at least eight panels' worth of columns (else the host LU runs): all n columns
//     ("resident") when m n elem + m panel elem fits in the free memory less a reserve (512 MiB + 1/32 of the card;
//     ENGINE_GPU_SMITH2K_VRAM_MB caps the budget), else the largest multiple of the panel that fits ("tiled", in phases:
//     Ring2kDevice.h; the host keeps one phase's L factors, at most ENGINE_GPU_SMITH2K_HOST_MB (default 4096) MiB plus one
//     panel, and the route declines when /proc/meminfo's MemAvailable is below that plus 2 GiB).
// A read-only tuning lever: it chooses where the arithmetic runs, never a value.  ENGINE_GPU_SMITH2K_VERIFY=1 (or a
// -DENGINE_VERIFY_DEVICE=ON build) re-runs every device LU on the host and compares it entrywise (throws on a difference).
// When ENGINE_GPU_SMITH2K is set, or ENGINE_DEVICE=gpu, one line
//   EFFECTIVE GPU_SMITH2K mode=on|off policy_gpu_calls=<a> device=<b> resident=<c> tiled=<d> slabs=<e> replays=<f>
//   fallbacks=<g> declined_vram=<h> declined_host=<i> panels=<j> lookahead=<k> verified=<l> min_entries=<x>
//   t_s=open:<s>,fetch:<s>,panel:<s>,step:<s>,host:<s>,close:<s>   (wall seconds per phase: Ring2kDevice.h Stats)
// is printed on stderr at exit.

#include <cstdint>
#include <string>

namespace integral_descent {
namespace smith_gpu {

// Validation overrides (process-wide).  ScopedForce(true): the device route regardless of the policy, the lever and the
// size floor (still only with a device); ScopedForce(false): never.  max_cols > 0 caps the device window (forces the tiled
// route when below n); verify forces the host re-run; store_bytes > 0 replaces the L-store budget (small: many phases).
// ScopedSimulateError: the next session open throws a CudaError.
struct ScopedForce {
    explicit ScopedForce(bool on, uint32_t max_cols = 0, bool verify = false, uint64_t store_bytes = 0);
    ~ScopedForce();
    ScopedForce(const ScopedForce&) = delete;
    ScopedForce& operator=(const ScopedForce&) = delete;
private:
    int saved_on_;
    uint32_t saved_cols_;
    bool saved_verify_;
    uint64_t saved_store_;
};
struct ScopedSimulateError {
    ScopedSimulateError();
    ~ScopedSimulateError();
    ScopedSimulateError(const ScopedSimulateError&) = delete;
    ScopedSimulateError& operator=(const ScopedSimulateError&) = delete;
};

bool        installed();          // the table is installed (static initialisation of this TU)
bool        lever_on();           // ENGINE_GPU_SMITH2K
uint64_t    min_entries();        // ENGINE_GPU_SMITH2K_MIN
std::string effective_text();     // the EFFECTIVE line (no newline)

// (PAR-GPU audit L2, L3) The host RSS the device route adds to one 2-adic LU of an m x n matrix (elem bytes per entry), for the
// descent's RSS predictions (PredictTreeLocalRss, PredictLocalRss at ell = 2): 0 unless the route would be taken by the calling
// thread (ScopedForce(true), or the policy: ENGINE_DEVICE=gpu, ENGINE_GPU_SMITH2K on, m n >= ENGINE_GPU_SMITH2K_MIN, a device).
// Then: the tiled route's L store, min(ENGINE_GPU_SMITH2K_HOST_MB + one panel, the whole L) -- (slot-24 PANEL-IMPL) dropped when
// the whole m x n window fits the device budget now (op_plan would run it resident; the store is then never allocated) --, the
// driver's P and L panel buffers (2 m panel elem), the pinned staging (CudaRing2k.h staging_bytes), at the device LU's panel
// (smithpar::kDevicePanel = 128, not the host LU's size-dependent width); and, under ENGINE_GPU_SMITH2K_VERIFY=1 /
// ENGINE_VERIFY_DEVICE, the verify twin (the whole input copied for the host re-run, m n elem).  A function of the shape, the
// process's settings and the card's free memory at the call; never a value.
uint64_t PredictHostRssBytes(uint64_t m, uint64_t n, uint32_t elem);

} // namespace smith_gpu
} // namespace integral_descent
