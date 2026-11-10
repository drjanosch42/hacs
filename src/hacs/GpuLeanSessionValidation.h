// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#include <cstdint>
// (hecke-extensions slot 23, PERF-P7) The VRAM-lean device session (ENGINE_API.md Sec. 12.9): under ENGINE_DEVICE=gpu a
// compiled program whose GPU nodes are all memory-class runs on the host (ENGINE_GPU_LEAN, default on), and the arena's
// soft-trim slack is a sixteenth of the card. This validation proves that neither moves a value and that both act:
//   L  engine level, p = 13 and p = 134217757: a memory-only program (zero / AXPY / AXPY-identity / SCAL / TRANSPOSE /
//      COMPARE) and a compute program (GEMM + AXPY) under cpu, gpu lean, gpu lean=0, each gpu arm with and without the
//      ENGINE_VERIFY_DEVICE re-execution: outputs and result slots identical to cpu; the memory-only program under lean
//      counts one lean host run, puts no node on the device and allocates no device byte; under lean=0 it runs on the
//      device (the contrast: not vacuous); the compute program is never lean; gpu_compute_nodes as compiled;
//   S  inside an elimination session the rule is off (the session's residency wins);
//   H  the Hecke operator (Theta_flat, Theta_eq, A) at (ell, n, p) = (13, 24, 7) and (7, 30, 5) under cpu, gpu lean and
//      gpu lean=0: identical; route 2's memory-only programs are lean host runs (> 0) under lean and none under lean=0; the
//      thread's DeviceTally separates compiled placement (gpu_programs) from host execution (lean_host_programs);
//   A  the arena: the slack is a sixteenth (lean) or a quarter (lean=0) of the card unless ENGINE_CUDA_ARENA_SLACK_MB sets
//      it; a churn of 72 distinct block sizes leaves at most the slack plus two blocks idle.
// Prints [SKIP] without a usable device (built without CUDA, or ENGINE_CUDA_DISABLE=1). One GPU_LEAN line. Returns the number
// of failing checks.
uint32_t GpuLeanSessionValidation();
