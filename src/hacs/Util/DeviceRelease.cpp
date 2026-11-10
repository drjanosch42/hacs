// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Util/DeviceRelease.cpp -- see DeviceRelease.h (hecke-extensions slot 14, WP-3).
#include "DeviceRelease.h"

#include <cstdlib>
#include <exception>
#include <iostream>

#include "../../Cuda/CudaCtx.h"

unsigned long long ReleaseDeviceMemoryAfterValidation(const std::string& validation_name) {
    namespace cu = engine::cuda;
    try {
        // (slot-14 WP-16, audit S-3) no device-policy gate: the default cpu policy's
        // scoped-gpu arms create the context and reserve slabs too.  context_created()
        // alone keeps a process that never touched the device untouched -- no probe, no
        // context creation (release_device_caches() re-checks it before CudaCtx::get()).
        if (!cu::context_created()) return 0;
        const cu::ReleaseStats rs = cu::release_device_caches();
        const char* tr = std::getenv("ENGINE_TRACE_DEVICE_RELEASE");
        if (tr != nullptr && tr[0] != '\0' && tr[0] != '0') {
            const cu::ArenaStats as = cu::CudaCtx::get().arena_stats();
            // stderr: a traced run must not move any frozen stdout transcript
            // (ConsoleCliPreservationValidation re-runs this binary as a child).
            std::cerr << "DEVICE_RELEASE after=" << validation_name
                      << " arena_freed_mib=" << (rs.arena_bytes_freed >> 20)
                      << " slabs_freed=" << rs.slabs_freed << " slabs_kept=" << rs.slabs_kept
                      << " pinned_freed_mib=" << (rs.pinned_bytes_freed >> 20)
                      << " reserved_after_mib=" << (as.reserved_bytes >> 20)
                      << " peak_reserved_mib=" << (as.peak_reserved_bytes >> 20)
                      << " soft_trims=" << as.soft_trims << " oom_retries=" << as.oom_retries << std::endl;
        }
        return static_cast<unsigned long long>(rs.arena_bytes_freed);
    } catch (const std::exception& e) {
        std::cout << "[WARN] device release after " << validation_name << " failed: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "[WARN] device release after " << validation_name << " failed" << std::endl;
    }
    return 0;
}
