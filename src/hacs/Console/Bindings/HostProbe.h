// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// HostProbe.h -- NBCONFIRM S3: the device facts behind console::HostProbe
// (Console/Core/Confirm.h), which ConsoleCore cannot read itself (STL only).
//
//   prompt_tty         isatty(0) && isatty(2)
//   write_prompt       write(2, ...) -- the real stderr, past any ChatterCapture
//   mem_available_mib  /proc/meminfo MemAvailable
//   c_free_gb          statvfs("/mnt/c") (the C: drive under WSL, for the low-disk warning); unknown elsewhere
//   cpu_count          sched_getaffinity(0) (S2 audit F9), else hardware_concurrency
//   build_id           the NT_GNU_BUILD_ID note of /proc/self/exe (S2 audit F4)
//   binary_stat        size and mtime of /proc/self/exe
//
// Advisory facts only: none of them reaches a value, a record or a hash.
// ===========================================================================
#pragma once
#ifndef CONSOLE_BINDINGS_HOST_PROBE_H
#define CONSOLE_BINDINGS_HOST_PROBE_H

#include <cstdint>
#include <string>

#include "../Core/Confirm.h"

namespace console {
namespace host {

HostProbe RealHostProbe();
// Installs RealHostProbe() (main.cpp, before the console runs).
void InstallRealHostProbe();

unsigned AffinityCpuCount();
double MemAvailableMiB();
// The ELF build id of `path` as lowercase hex, "" when there is none (not ELF, no note).
std::string ElfBuildId(const std::string& path);
bool FileStat(const std::string& path, std::uint64_t& size, std::int64_t& mtime);
std::string SelfExe();   // the realpath of /proc/self/exe

} // namespace host
} // namespace console

#endif // CONSOLE_BINDINGS_HOST_PROBE_H
