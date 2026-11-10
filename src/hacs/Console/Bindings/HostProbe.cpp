// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HostProbe.cpp -- see HostProbe.h.

#include "HostProbe.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace console {
namespace host {

unsigned AffinityCpuCount() {
#ifdef CPU_COUNT
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0) return static_cast<unsigned>(n);
    }
#endif
    const unsigned h = std::thread::hardware_concurrency();
    return h == 0 ? 1 : h;
}

double MemAvailableMiB() {
    std::ifstream in("/proc/meminfo");
    std::string key;
    double kb = 0;
    std::string unit;
    while (in >> key >> kb) {
        std::getline(in, unit);
        if (key == "MemAvailable:") return kb / 1024.0;
    }
    return -1.0;
}

namespace {

double CFreeGb() {
    struct stat st;
    if (::stat("/mnt/c", &st) != 0 || !S_ISDIR(st.st_mode)) return -1.0;
    struct statvfs v;
    if (::statvfs("/mnt/c", &v) != 0 || v.f_blocks == 0) return -1.0;
    // df -BG: available blocks to an unprivileged user, in GiB
    return static_cast<double>(v.f_bavail) * static_cast<double>(v.f_frsize) / (1024.0 * 1024.0 * 1024.0);
}

// Reads `n` bytes at `off` (the binary is hundreds of MB: never read it whole).
bool ReadAt(std::ifstream& in, std::uint64_t off, std::size_t n, std::string& out) {
    out.assign(n, '\0');
    in.clear();
    in.seekg(static_cast<std::streamoff>(off));
    if (!in) return false;
    in.read(&out[0], static_cast<std::streamsize>(n));
    return static_cast<std::size_t>(in.gcount()) == n;
}

template <class T>
T Le(const std::string& d, std::size_t off, bool le) {
    T v = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        const std::uint64_t b = static_cast<unsigned char>(d[off + (le ? i : sizeof(T) - 1 - i)]);
        v = static_cast<T>(v | static_cast<T>(b << (8 * i)));
    }
    return v;
}

} // namespace

// A port of nbconfirm_calibrate.py::elf_build_id: walk the section headers, read the SHT_NOTE sections, return the GNU note.
std::string ElfBuildId(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) return std::string();
    std::string h;
    if (!ReadAt(in, 0, 64, h) || h.compare(0, 4, "\x7f" "ELF") != 0) return std::string();
    const bool c64 = h[4] == 2, le = h[5] == 1;
    const std::uint64_t shoff = c64 ? Le<std::uint64_t>(h, 0x28, le) : Le<std::uint32_t>(h, 0x20, le);
    const std::uint16_t shentsize = Le<std::uint16_t>(h, c64 ? 0x3A : 0x2E, le);
    const std::uint16_t shnum = Le<std::uint16_t>(h, c64 ? 0x3C : 0x30, le);
    if (shentsize < (c64 ? 0x28u : 0x18u)) return std::string();
    for (std::uint32_t i = 0; i < shnum; ++i) {
        std::string sh;
        if (!ReadAt(in, shoff + static_cast<std::uint64_t>(i) * shentsize, shentsize, sh)) return std::string();
        if (Le<std::uint32_t>(sh, 4, le) != 7) continue;   // SHT_NOTE
        const std::uint64_t off = c64 ? Le<std::uint64_t>(sh, 0x18, le) : Le<std::uint32_t>(sh, 0x10, le);
        const std::uint64_t size = c64 ? Le<std::uint64_t>(sh, 0x20, le) : Le<std::uint32_t>(sh, 0x14, le);
        if (size > (1u << 20)) continue;
        std::string d;
        if (!ReadAt(in, off, static_cast<std::size_t>(size), d)) continue;
        std::size_t p = 0;
        while (p + 12 <= d.size()) {
            const std::uint32_t nsz = Le<std::uint32_t>(d, p, le), dsz = Le<std::uint32_t>(d, p + 4, le),
                                typ = Le<std::uint32_t>(d, p + 8, le);
            const std::size_t q = p + 12 + ((nsz + 3u) & ~3u);
            if (q + dsz > d.size() || p + 12 + nsz > d.size()) break;
            std::string name = d.substr(p + 12, nsz);
            while (!name.empty() && name.back() == '\0') name.pop_back();
            if (typ == 3 && name == "GNU") {
                static const char* hex = "0123456789abcdef";
                std::string out;
                for (std::uint32_t k = 0; k < dsz; ++k) {
                    const unsigned char c = static_cast<unsigned char>(d[q + k]);
                    out += hex[c >> 4];
                    out += hex[c & 15];
                }
                return out;
            }
            p = q + ((dsz + 3u) & ~3u);
        }
    }
    return std::string();
}

bool FileStat(const std::string& path, std::uint64_t& size, std::int64_t& mtime) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return false;
    size = static_cast<std::uint64_t>(st.st_size);
    mtime = static_cast<std::int64_t>(st.st_mtime);
    return true;
}

std::string SelfExe() {
    char buf[PATH_MAX];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n <= 0) return std::string();
    buf[n] = '\0';
    return std::string(buf);
}

HostProbe RealHostProbe() {
    HostProbe h;
    h.prompt_tty = [] { return ::isatty(0) == 1 && ::isatty(2) == 1; };
    h.write_prompt = [](const std::string& s) {
        std::size_t done = 0;
        while (done < s.size()) {
            const ssize_t w = ::write(2, s.data() + done, s.size() - done);
            if (w <= 0) break;
            done += static_cast<std::size_t>(w);
        }
    };
    h.mem_available_mib = [] { return MemAvailableMiB(); };
    h.c_free_gb = [] { return CFreeGb(); };
    // (S3 Fix round 1, audit m2) HACS_HOST_PROBE_FIXTURE=1: the test fixture of ConsoleCostModelValidation's pty arm (H), set
    // only in that arm's child, so that `--all` does not depend on this host's memory or free space on C:.  It pins exactly
    // two facts: MemAvailable = 4096 MiB (a LOWER limit than any reference host: rule 1 refuses more, not less) and C: free =
    // 255 GB.  Read-only like every env hook: no value, record or hash input reads the probe.  Never set it to make a run pass.
    // (NBCONFIRM S4 fix round 1, audit L2) `--serve` refuses it unless the session's parent is this same binary (Serve.cpp).
    if (const char* fx = std::getenv("HACS_HOST_PROBE_FIXTURE")) {
        if (std::strcmp(fx, "1") == 0) {
            h.mem_available_mib = [] { return 4096.0; };
            h.c_free_gb = [] { return 255.0; };
        }
    }
    h.cpu_count = [] { return AffinityCpuCount(); };
    h.build_id = [] {
        static const std::string id = ElfBuildId("/proc/self/exe");
        return id;
    };
    h.binary_stat = [](std::uint64_t& size, std::int64_t& mtime) { return FileStat(SelfExe(), size, mtime); };
    return h;
}

void InstallRealHostProbe() { SetHostProbe(RealHostProbe()); }

} // namespace host
} // namespace console
