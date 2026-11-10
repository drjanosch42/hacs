// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Util/Rss.h -- the process's resident-set figures, ONE definition.
//
// (hecke-extensions slot 7, WP-8 / console C2b, gap G4b of
// docs/console/00-console-interface-plan.md Sec. 14.2.)  Before this header
// `PeakRssMiB()` was defined twice, file-locally, in Bench/BenchDriver.cpp and
// Campaign/CampaignDriver.cpp (with `ResetPeakRss()` only in the latter), and
// the console's `rss()` / `timed(...)` would have been a third copy.  Bench,
// campaign and console now read the same function.  The bodies are the two
// former copies verbatim (they were already the same parse), so no printed
// figure moves.
//
// Header-only, STL only, Linux /proc; every reader returns -1.0 where /proc is
// unavailable, and every caller prints "n/a" for that.
// ===========================================================================
#pragma once
#ifndef OC_UTIL_RSS_H
#define OC_UTIL_RSS_H

#include <fstream>
#include <sstream>
#include <string>

namespace oc_util {

// A "<key>\t   123456 kB" line of /proc/self/status, in MiB.  `key` includes
// its colon and is six characters long for both keys used ("VmHWM:",
// "VmRSS:"), which is the offset the value is parsed from.
inline double ProcStatusMiB(const char* key) {
    std::ifstream st("/proc/self/status");
    if (!st) return -1.0;
    std::string line;
    while (std::getline(st, line)) {
        if (line.rfind(key, 0) == 0) {
            std::istringstream iss(line.substr(6));
            double kb = 0.0;
            if (iss >> kb) return kb / 1024.0;
            return -1.0;
        }
    }
    return -1.0;
}

// Peak resident set size (VmHWM, the kernel's high-water mark): the PROCESS
// peak since start or since the last successful ResetPeakRss(), never a
// single call's own.
inline double PeakRssMiB() { return ProcStatusMiB("VmHWM:"); }

// Current resident set size (VmRSS).
inline double CurrentRssMiB() { return ProcStatusMiB("VmRSS:"); }

// Resets the kernel's VmHWM counter (writes "5" to /proc/self/clear_refs), so
// the next PeakRssMiB() is the peak from here on.  Best effort: false where
// the kernel does not support it, and the caller must then report the figure
// as the process peak rather than pretend it is scoped.
inline bool ResetPeakRss() {
    std::ofstream cr("/proc/self/clear_refs");
    if (!cr) return false;
    cr << "5\n";
    cr.flush();
    return static_cast<bool>(cr);
}

} // namespace oc_util

#endif // OC_UTIL_RSS_H
