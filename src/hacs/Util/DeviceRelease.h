// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Util/DeviceRelease.h -- the between-validations release of device memory
// (hecke-extensions slot 14, WP-3).
//
// THE DEFECT ([C4] Sec. 2 on a 40 GB A100, [C7] headline 3 on a 94 GB H100).
// Under `ENGINE_DEVICE=gpu` a `--all` process's VRAM climbs monotonically to
// the card's capacity and the process OOMs in HeckeHidaLevelEllValidation /
// HeckeCensusLatticeValidation.  The cause is the CUDA arena
// (src/Cuda/CudaCtx.cpp): every device buffer is cut from a slab
// and, when freed, parked on an EXACT-SIZE free list; nothing was ever
// cudaFree'd.  Executors, device programs, mirrors and workspaces ARE freed
// at the end of every run (they are stack objects of the consumers), but
// their bytes stay reserved in the arena, and a later program whose buffer
// sizes differ by 256 bytes cannot reuse them.
//
// THE HOOK.  ReleaseDeviceMemoryAfterValidation() calls
// engine::cuda::release_device_caches() -- every slab with no live block is
// returned to the driver, and the pinned staging pool is freed -- whenever
// the CUDA context already exists, under EVERY device policy.  (WP-3 gated it
// on a non-cpu effective policy as well; slot-14 WP-16, audit S-3, dropped
// that gate: under the default cpu policy the scoped `gpu` arms of the device
// validations -- CudaBitIdentityValidation, EngineCrossDeviceValidation, the
// device arms of the determinism validations, ... -- create the context and
// reserve slabs too, and nothing returned them for the rest of the process.)
// A process that never created the context (every cpu-only validation, any
// run under ENGINE_CUDA_DISABLE=1 or without a device) sees a single atomic
// load and nothing else: no probe, no context creation.
// ValidationRegistry's RunGuarded calls it after every validation, on the
// normal and on the exception path.  The in-run defence lives in
// CudaCtx::dalloc (needed: HeckeHidaLevelEllValidation ALONE filled the
// 16 GB RTX before WP-3): a soft trim before a new slab once the idle bytes
// grew by ENGINE_CUDA_ARENA_SLACK_MB (default a quarter of the device), and
// one trim + retry when cudaMalloc reports an OOM.
//
// WHY NO VALUE CAN CHANGE.  Only slabs none of whose blocks is live are
// freed; which device address a buffer lives at is not an input of any
// kernel (ENGINE_API.md Sec. 11: every device kernel is the exact-integer
// twin of its CPU kernel), and a released pinned chunk is re-allocated
// lazily on the next staged copy.
//
// ENGINE_TRACE_DEVICE_RELEASE=1 prints one `DEVICE_RELEASE` line per call to stderr
// (bytes freed, slabs freed / kept, the arena's reserved bytes after).
// ===========================================================================
#pragma once
#ifndef OC_UTIL_DEVICE_RELEASE_H
#define OC_UTIL_DEVICE_RELEASE_H

#include <string>

// Returns the arena bytes it freed (0 without a context).  Never throws: a failure is printed and swallowed, so the release
// can never turn a passing validation into a failing one.
unsigned long long ReleaseDeviceMemoryAfterValidation(const std::string& validation_name);

#endif // OC_UTIL_DEVICE_RELEASE_H
