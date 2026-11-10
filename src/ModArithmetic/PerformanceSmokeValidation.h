// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef PERFORMANCE_SMOKE_VALIDATION_H
#define PERFORMANCE_SMOKE_VALIDATION_H

#include <cstdint>
#include <string>

#include "DenseBuf.h"

// ===========================================================================
// matrix-engine Phase 2 -- arithmetic micro-benchmarks + performance smoke
// (docs/matrix-engine/00-implementation-plan.md, "Phase 2": the
// `PerformanceSmokeValidation` bullet under "Validations" and the "Exit
// criterion" paragraph, whose numbers come from
// docs/matrix-engine/03-profiling-baseline.md Sec. 7.2 / 7.3).
//
// WHY THIS LIVES IN ModArithmetic/ RATHER THAN Bench/
// --------------------------------------------------
// The micro-benchmarks measure ONLY the arithmetic substrate (FlatMatrix
// today, ModCtx/DenseBuf/Kernels/DenseElim tomorrow), so they must be
// callable from a validation registered in the `ModArithmetic` static
// library. `Bench/BenchDriver.cpp` belongs to the `cohomology_app` target,
// which links ModArithmetic and not the other way round -- a validation in
// this library therefore cannot call into the bench driver. The measurement
// core lives here and BenchDriver's `micro` mode is a thin CLI over it, so
// `--bench micro gemm 512` and `PerformanceSmokeValidation` measure the
// same code by construction.
//
// ---------------------------------------------------------------------------
// STAGE B: WHERE TO PLUG IN THE NEW ENGINE
// ---------------------------------------------------------------------------
// Every op has exactly ONE dispatch function in
// PerformanceSmokeValidation.cpp, of the shape
//
//     MicroResult RunGemm(...) {
//         switch (backend) {
//             case Backend::Legacy: return GemmLegacy(...);
//             case Backend::Engine: return NotAvailable("gemm", ...);
//         }
//     }
//
// Wiring the new kernels is therefore: write `GemmEngine(...)` (etc.)
// alongside `GemmLegacy(...)` and replace the `NotAvailable(...)` line of
// the corresponding `Run*` dispatcher. Five dispatchers, five lines:
// RunGemm, RunGemv, RunRref, RunKernel, RunTranspose. Nothing else in this
// file or in BenchDriver has to change; `--backend engine` and
// `ENGINE_BACKEND=engine ./cohomology_app PerformanceSmokeValidation` then
// measure the new path against the SAME fixed-seed inputs and the SAME
// checksums as the legacy one, which is what makes the comparison
// like-for-like (and the checksum equality a free bit-identity check).
// ===========================================================================

namespace micro {

// The prime used by every default and by the whole exit-criterion suite:
// 134217757 ~= 2^27, the benchmark prime of BUILDING.md and doc 03.
constexpr uint32_t kDefaultEll = 134217757u;

// Which implementation of the substrate to measure.
//   Legacy -- FlatMatrix as it stands (operator*, compute_row_echelon_form /
//             get_row_echelon_form_as_flat_matrix, rank, kernel, transpose).
//   Engine -- the Phase-2 ModCtx/DenseBuf/Kernels/DenseElim layer. Not wired
//             yet: every op reports `available == false`, which the CLI turns
//             into "not available" + exit code 2.
//   Gpu    -- (matrix-engine Phase 5) the engine::cuda device kernels. This
//             library does not link the CUDA layer (EngineCuda links
//             ModArithmetic, not the other way round), so the arm is reached
//             through the GpuHooks installed by cohomology_app at start-up
//             (engine::cuda::InstallMicroHooks(), src/Cuda/CudaBench.h);
//             without the hooks -- or without a usable device -- every op
//             reports `available == false`, exit code 2.
enum class Backend { Legacy, Engine, Gpu };

const char* BackendName(Backend b);
bool ParseBackend(const std::string& s, Backend& out);

// Reads the ENGINE_BACKEND environment variable ("legacy" / "engine" / "gpu");
// anything else (including unset) yields Backend::Engine.
Backend BackendFromEnv();

// One measured op. `min_s` is the reporting figure (least noise-contaminated
// estimate over the repeats); `median_s` is printed next to it.
struct MicroResult {
    bool available = true;       // false <=> the backend has no implementation
    std::string op;              // "gemm", "rref", ...
    std::string params;          // machine-readable "n=512 k=512 ell=..."
    double min_s = 0.0;
    double median_s = 0.0;
    // Modular multiplications performed by ONE repeat, for the rate figure.
    // 0.0 for ops whose cost is not a modmul count (transpose).
    double work_modmul = 0.0;
    uint64_t checksum = 0;       // FNV-1a-64 over the result, see .cpp
    uint32_t rank = 0;           // rref / kernel: rank actually found
    uint32_t nullity = 0;        // kernel: number of basis vectors returned
    std::string note;            // free-form, printed after the numbers

    double gmodmul_per_s() const {
        return (work_modmul > 0.0 && min_s > 0.0) ? work_modmul / min_s / 1e9 : 0.0;
    }
};

// ---------------------------------------------------------------------------
// The five ops. All inputs are fixed-seed pseudo-random (splitmix64, seeded
// from the shape + a per-op salt), so two runs -- and two BACKENDS -- see
// byte-identical inputs and must report equal checksums.
//
// `repeat` >= 1; setup (input generation) is never timed, and each repeat
// operates on a fresh copy of the input so that FlatMatrix's lazy RREF cache
// cannot make repeats 2..k free.
// ---------------------------------------------------------------------------

// C = A * B with A: n x k, B: k x n, C: n x n. work = n*n*k.
MicroResult RunGemm(uint32_t n, uint32_t k, uint32_t ell, Backend backend, uint32_t repeat);

// y = A * x with A: n x n, x: n x 1. work = n*n.
MicroResult RunGemv(uint32_t n, uint32_t ell, Backend backend, uint32_t repeat);

// RREF of a `rows x cols` matrix of prescribed rank `rank_target`
// (`rank_target == 0` or >= min(rows,cols) means full rank).
MicroResult RunRref(uint32_t rows, uint32_t cols, uint32_t rank_target, uint32_t ell,
                    Backend backend, uint32_t repeat);

// kernel() of the same matrix family.
MicroResult RunKernel(uint32_t rows, uint32_t cols, uint32_t rank_target, uint32_t ell,
                      Backend backend, uint32_t repeat);

// transpose() of an n x n matrix. work_modmul == 0 (it is a data movement op);
// the note carries the element rate.
MicroResult RunTranspose(uint32_t n, uint32_t ell, Backend backend, uint32_t repeat);

// The two ranks used by the exit-criterion suite for the 1677 x 1419 case.
//
// kSuiteRrefRank -- FULL rank. Doc 03 Sec. 7.3 measures its reference
//   elimination on RANDOM FULL-RANK matrices of exactly this shape (6.589 s
//   scalar / 1.108 s lazy) and the Phase-2 floor of 0.6 s is derived from
//   those numbers, so the exit-criterion row must use full rank to be
//   comparable with the floor it is judged against.
//
// kSuiteRealRank -- the rank of the REAL matrix. Doc 03 Sec. 2 tabulates the
//   1677 x 1419 `A_C1` at n_sym = 128 but records only its shape. Measured
//   here via `--bench cohom 128 134217757`, which now also reports
//   dim C^q_eq = dim ker(A_Cq): dim C^1_eq = 1161, so
//       rank(A_C1) = 1419 - 1161 = 258 = 2 * d_V = 2 * 129,
//   which is exactly what Sec. 2's finding S2 predicts (every block row of
//   A_Cq carries at most two non-zero blocks). The real matrix is therefore
//   MASSIVELY rank-deficient, not the full-rank case Sec. 7.3 models, and the
//   suite reports both: the full-rank row for the plan's floor, and the
//   real-rank rref/kernel rows for what the pipeline actually eliminates.
constexpr uint32_t kSuiteRrefRank = 1419u;
constexpr uint32_t kSuiteRealRank = 258u;

// ---------------------------------------------------------------------------
// Phase 5: the GPU arm. The three measured ops take the SAME fixed-seed
// inputs (EngineMatrix) and report the SAME checksum family (ChecksumBuf) as
// the engine arm, so equal `checksum=` values across `--backend engine` and
// `--backend gpu` are a bit-identity statement. `suite` prints the Phase-5
// throughput table (`--bench micro cuda`).
// ---------------------------------------------------------------------------
struct GpuHooks {
    MicroResult (*gemm)(uint32_t n, uint32_t k, uint32_t ell, uint32_t repeat, const std::string& params) = nullptr;
    MicroResult (*gemv)(uint32_t n, uint32_t ell, uint32_t repeat, const std::string& params) = nullptr;
    MicroResult (*transpose)(uint32_t n, uint32_t ell, uint32_t repeat, const std::string& params) = nullptr;
    int (*suite)(uint32_t repeat) = nullptr;
};
void SetGpuHooks(const GpuHooks& hooks);
const GpuHooks& GetGpuHooks();

// The fixed-seed engine-layout inputs of the engine arm (row-major, ld == cols).
engine::DenseBuf<uint32_t> EngineMatrix(uint32_t rows, uint32_t cols, uint32_t ell, uint64_t salt);
// FNV-1a-64 over (rows, cols, entries row-major) -- the checksum of MicroResult.
uint64_t ChecksumBuf(const engine::DenseBuf<uint32_t>& b);

} // namespace micro

// ---------------------------------------------------------------------------
// The registered validation (tier `standard`, name
// "PerformanceSmokeValidation"). Runs `gemm 512` and `rref 1677x1419` on the
// backend selected by ENGINE_BACKEND (default legacy) and checks them against
// COARSE floors -- deliberately loose enough for WSL2 noise, see the
// kFloor* constants in the .cpp for how they were derived from the measured
// legacy baseline. Prints measured vs floor for each. Returns 0 on success.
// ---------------------------------------------------------------------------
uint32_t PerformanceSmokeValidation();

#endif // PERFORMANCE_SMOKE_VALIDATION_H
