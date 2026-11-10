// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_RETIRED_DEVICE_LEVERS_H
#define ENGINE_RETIRED_DEVICE_LEVERS_H
// =====================================================================
// RetiredDeviceLevers.h -- the three GPU rollback levers retired by S24 CL-R2
// (slot-24 close item K2, docs/implementation/slot24/CLOSE_plan.md D2;
// the rollback-lever retirement rule of 2026-10-30; CLR2_acceptance.md)
// =====================================================================
//
// Header-only, STL-only (EnvLever.h), so EngineCuda (cuda_kernels.cu and the CPU stub CudaStub.cpp)
// and MatrixEngine (TapeElim.cpp) read ONE object per variable (function-local statics of inline
// functions: one instance per process).
//
//   ENGINE_ELIM_SESSION  (slot-18 WP-19a; ENGINE_API.md Sec. 12.8)   `0...` = no elimination session (per-program mirrors)
//   ENGINE_GPU_IMMA      (slot-22 G1;     ENGINE_API.md Sec. 11.8)   `0...` = the u32 device GEMM instead of the int8 IMMA route
//   ENGINE_GPU_ELIM      (slot-22 G2;     ENGINE_API.md Sec. 11.9)   `0...` = the host / tape-panel elimination instead of imma_rref
//   ENGINE_GPU_LEAN      (slot-23 P7;     ENGINE_API.md Sec. 12.9)   `0...` = memory-class-only programs on the device and a quarter-card
//                        arena slack (retired by S25 LEVERS-25, the slot-25 close; evidence slot25/LEVERS25_brief.md R2)
//
// The policy is the old copies' reading, byte for byte: re-read on every call (the validations switch them
// per arm), OFF iff the value STARTS WITH '0' ("0", "00", "0x", ...), unset / empty / anything else = ON,
// silently (Match::FirstChar, Empty::Unset, Cache::PerCall, no WARNING).
//
//   * OC_ENABLE_LEGACY=ON  -- RetiredLever is EnvLever: each lever behaves exactly as before CL-R2.
//   * OC_ENABLE_LEGACY=OFF -- enabled() is always true (the default route); a value starting with '0' is
//     REFUSED at the first read: `REFUSED ENGINE_GPU_ELIM=0: compiled out (OC_ENABLE_LEGACY=OFF)` (stderr,
//     exit 1). main() reads all three once at start-up (refuse_retired_device_levers_at_startup()), so
//     the refusal does not depend on the device policy or on which code path reads first.
//
// What stays in every build (the oracle / production exception of the rule): the host elimination, the
// u32 device GEMM and the per-program-mirror executor path are the default CPU route, the route for
// p > 127 / u64, and the path of every program outside an elimination session. Only the env arms retire.
// None of these switches can change a value (ENGINE_API.md Sec. 11, 12.8).
// =====================================================================

#include <string>

#include "EnvLever.h"

namespace engine {
namespace lever {

inline constexpr Policy kRetiredDeviceLeverPolicy{true, Empty::Unset, Match::FirstChar, Cache::PerCall, nullptr,
                                                  "default", "env", "env"};

inline const RetiredLever& elim_session_lever() {
    static const RetiredLever l("ENGINE_ELIM_SESSION", kRetiredDeviceLeverPolicy);
    return l;
}
inline const RetiredLever& gpu_imma_lever() {
    static const RetiredLever l("ENGINE_GPU_IMMA", kRetiredDeviceLeverPolicy);
    return l;
}
inline const RetiredLever& gpu_elim_lever() {
    static const RetiredLever l("ENGINE_GPU_ELIM", kRetiredDeviceLeverPolicy);
    return l;
}
// (S25 LEVERS-25) ENGINE_GPU_LEAN retires its ENV ARM only (EnvLever.h, EnvArmRetiredLever): the old readers
// (Executor.cpp gpu_lean_enabled(), per run; CudaCtx.cpp, the arena slack at context creation) were
// `!(e != nullptr && e[0] == '0')`, the same policy.  The old route (a memory-class-only program on the device, a
// quarter-card slack) stays in every build -- it is the route of every such program inside an elimination session --
// and EngineCrossDeviceValidation / GpuLeanSessionValidation reach it through the Scoped override
// (engine::lever::EnvLever::Scoped(gpu_lean_lever(), false)) instead of the environment.
inline const EnvArmRetiredLever& gpu_lean_lever() {
    static const EnvArmRetiredLever l("ENGINE_GPU_LEAN", kRetiredDeviceLeverPolicy);
    return l;
}

// One read of each of the four (OFF: a '0...' value exits 1 here; ON: a pure read, no state, no output).
inline void refuse_retired_device_levers_at_startup() {
    (void)elim_session_lever().enabled();
    (void)gpu_imma_lever().enabled();
    (void)gpu_elim_lever().enabled();
    (void)gpu_lean_lever().env_state();   // (S25 LEVERS-25) the env reading (a Scoped override would hide it)
}

// `EFFECTIVE <NAME> route=new|old source=retired|default|env` -- printed by the lever's validations.
inline std::string retired_device_lever_line(const RetiredLever& l) { return l.effective_line("route", "new", "old"); }

} // namespace lever
} // namespace engine

#endif // ENGINE_RETIRED_DEVICE_LEVERS_H
