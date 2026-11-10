// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_FINGERPRINT_DEVICE_BIT_VALIDATION_H
#define ENGINE_FINGERPRINT_DEVICE_BIT_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 10, WP-8: decision 10 (A), the ninth fingerprint bit
// (ENGINE_API.md Sec. 10.6). `pass_options_fingerprint()` used to give
// `hybrid` and `gpu` the same value (both set only bit 128), so a
// hybrid-compiled and a gpu-compiled program could share a ProgramCache slot.
// Since this package bit 256 is set iff the effective policy is Gpu
// specifically, so the pair (128, 256) reads (0, 0) cpu, (1, 0) hybrid,
// (1, 1) gpu -- three distinct fingerprints -- with no value moved (a device
// result is the CPU result's bytes by construction, ENGINE_API.md Sec. 11;
// this only changes which cache slot a compiled program occupies).
//
// Arms:
//   (A) cpu / hybrid / gpu (the three from_env() device mappings) give three
//       distinct fingerprints, with the exact (bit 128, bit 256) pattern
//       above, and the seven non-device bits unaffected by the policy alone;
//   (B) a tiny one-GEMM tape program, recorded identically each time, is
//       compiled once under each of the three policies: the first compile
//       under each is a fresh ProgramCache insert (never a hit of another
//       policy's slot), the second compile under the SAME policy replays it
//       (a hit), and the cache grows by exactly three slots. Needs no CUDA
//       device: a GPU-assigned node still runs (on the CPU) when none is
//       usable (src/Engine/Executor.cpp), so this arm never [SKIP]s.
// Writes nothing; no golden object, no record key.  Returns the number of
// failed checks.
uint32_t EngineFingerprintDeviceBitValidation();

#endif
