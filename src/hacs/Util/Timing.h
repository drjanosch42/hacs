// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Util/Timing.h -- wall seconds since a steady_clock time point, ONE definition
// (S24 CL-1; CODEAUDIT B2; docs/implementation/slot24/CLEANUP_plan_1030.md)
// ===========================================================================
//
// 56 file-local copies (`Seconds`, `Since`, `SecondsSince`) are all the same
// body, duration<double>(steady_clock::now() - t0).count(); no copy differs.
// Header-only, STL only.
// ===========================================================================
#pragma once
#ifndef OC_UTIL_TIMING_H
#define OC_UTIL_TIMING_H

#include <chrono>

namespace oc_util {

using SteadyClock = std::chrono::steady_clock;

inline double Seconds(SteadyClock::time_point t0) {
    return std::chrono::duration<double>(SteadyClock::now() - t0).count();
}

} // namespace oc_util

#endif // OC_UTIL_TIMING_H
