// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_THREADS_H
#define ENGINE_THREADS_H
// =====================================================================
// Threads.h -- ONE reader of ENGINE_THREADS (S24 CL-1; CODEAUDIT C-05, A3, B2;
// docs/implementation/slot24/CLEANUP_plan_1030.md)
// =====================================================================
//
// Header-only, STL + <sched.h>.  Seven copies parse ENGINE_THREADS today and they
// disagree (C-05).  configured(policy) reproduces each of them exactly; the
// presets below name the copy they reproduce, so a migration is
// behaviour-neutral (CL-2 / CL-3 / CL-5 migrate).  The differences:
//
//   preset           copy                                      parse            default count             caps (default / env)
//   kThreadPool      Engine/ThreadPool.cpp default_threads()   Strtoul          hardware                  32 / 32 (kMaxWorkers)
//   kSmithPar        Binary/SmithPar.cpp compute_default_...   Strict (warns)   affinity, else hardware   64 / 256 (kDefaultThreadCap / kMaxThreads)
//   kDescentParAsm   Integral/Descent/DescentParAsm.cpp        StrtolPositive   hardware                  256 / 256
//   kCap16           DescentHeckeCrt.cpp DefaultThreads,       StrtolPositive   hardware                  16 / 16
//                    HeckeAtlas.cpp ParallelFor, LValue
//                    ModularSymbolMeasure.cpp MeasureThreads
//   kGammaHTower     Hecke/HeckeGammaHTower.cpp TowerThreads   AtolClampFirst   hardware                  8 / 64
//
//   Strtoul         n = (uint32_t) strtoul(v) for a non-empty v (leading blanks, a sign and
//                   trailing junk accepted; "-1" wraps to 2^32-1 and is then capped; a value
//                   >= 2^32 is TRUNCATED to 32 bits first); n == 0 -> the default.
//   StrtolPositive  v = strtol(v); v > 0 -> n = (uint32_t) v (truncated as above), else the
//                   default; then max(1, min(cap, n)).
//   AtolClampFirst  v = atol(v); v >= 1 -> min(v, env cap) computed in `long` (no
//                   truncation), else the default.
//   Strict          strtoll, trailing blanks / tabs / newlines allowed, anything else, a
//                   negative value or an overflow -> the default plus a WARNING;
//                   0 -> the default; above the env cap -> the cap plus a WARNING.
//
// In every preset an unset variable, or a hardware count of 0, gives at least 1.
// The source tags are SmithPar's: "affinity" | "hardware" | "env" | "env-invalid".
// configured() reads the environment on every call and prints a Strict WARNING
// each time; a caller that read once (SmithPar) keeps its own static.
// =====================================================================

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#if defined(__linux__)
#include <sched.h>
#endif

namespace engine {
namespace threads {

enum class Parse : unsigned char { Strtoul, StrtolPositive, AtolClampFirst, Strict };
enum class Base : unsigned char { Hardware, Affinity };

struct Policy {
    Parse parse;
    Base base;
    uint32_t default_cap;   // applied to the default count
    uint32_t env_cap;       // applied to a count read from the variable
    const char* who;        // Strict only: the WARNING's subject ("the descent Smith pool")
};

constexpr Policy kThreadPool{Parse::Strtoul, Base::Hardware, 32, 32, nullptr};
constexpr Policy kSmithPar{Parse::Strict, Base::Affinity, 64, 256, "the descent Smith pool"};
constexpr Policy kDescentParAsm{Parse::StrtolPositive, Base::Hardware, 256, 256, nullptr};
constexpr Policy kCap16{Parse::StrtolPositive, Base::Hardware, 16, 16, nullptr};
constexpr Policy kGammaHTower{Parse::AtolClampFirst, Base::Hardware, 8, 64, nullptr};

struct Result {
    uint32_t threads = 1;
    const char* source = "hardware";
    std::string warning;    // the WARNING line (with its '\n'), "" when none
};

inline uint32_t hardware() { return std::thread::hardware_concurrency(); }

// The CPU affinity count of this process, 0 when unknown.
inline uint32_t affinity() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0) return static_cast<uint32_t>(n);
    }
#endif
    return 0;
}

namespace detail {
inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }
inline uint32_t max1(uint32_t a) { return a < 1 ? 1u : a; }
} // namespace detail

// The pure resolution of one raw value (nullptr = unset) given the machine's counts.  No I/O.
inline Result resolve(const char* raw, const Policy& p, uint32_t hw, uint32_t aff) {
    Result r;
    switch (p.parse) {
    case Parse::Strtoul: {
        uint32_t n = 0;
        if (raw != nullptr && raw[0] != '\0') n = static_cast<uint32_t>(std::strtoul(raw, nullptr, 10));
        if (n != 0) {
            r.source = "env";
            r.threads = detail::min_u(n, p.env_cap);
        } else {
            r.threads = detail::min_u(detail::max1(hw), p.default_cap);
        }
        return r;
    }
    case Parse::StrtolPositive: {
        if (raw != nullptr) {
            const long v = std::strtol(raw, nullptr, 10);
            if (v > 0) {
                r.source = "env";
                r.threads = detail::max1(detail::min_u(p.env_cap, static_cast<uint32_t>(v)));
                return r;
            }
        }
        r.threads = detail::max1(detail::min_u(p.default_cap, hw));
        return r;
    }
    case Parse::AtolClampFirst: {
        if (raw != nullptr) {
            const long v = std::strtol(raw, nullptr, 10);
            if (v >= 1) {
                r.source = "env";
                r.threads = static_cast<uint32_t>(v < static_cast<long>(p.env_cap) ? v : static_cast<long>(p.env_cap));
                return r;
            }
        }
        r.threads = detail::max1(detail::min_u(p.default_cap, hw ? hw : 1u));
        return r;
    }
    case Parse::Strict: {
        uint32_t def = p.base == Base::Affinity ? aff : 0;
        const char* def_src = p.base == Base::Affinity ? "affinity" : "hardware";
        if (def == 0) {
            def = hw;
            def_src = "hardware";
        }
        if (def == 0) def = 1;
        def = detail::min_u(def, p.default_cap);
        r.threads = def;
        r.source = def_src;
        if (raw == nullptr || raw[0] == '\0') return r;
        char* end = nullptr;
        errno = 0;
        const long long v = std::strtoll(raw, &end, 10);
        while (end && (*end == ' ' || *end == '\t' || *end == '\n')) ++end;
        if (end == raw || *end != '\0' || errno != 0 || v < 0) {
            r.warning = std::string("WARNING: ENGINE_THREADS=") + raw + " is not a non-negative integer; " + (p.who ? p.who : "the pool") +
                        " uses the default " + std::to_string(def) + " threads (" + def_src + ")\n";
            r.source = "env-invalid";
            return r;
        }
        if (v == 0) return r;   // 0 = the default
        r.source = "env";
        if (v > static_cast<long long>(p.env_cap)) {
            r.warning = std::string("WARNING: ENGINE_THREADS=") + raw + " exceeds " + std::to_string(p.env_cap) + "; " +
                        (p.who ? p.who : "the pool") + " is clamped to " + std::to_string(p.env_cap) + " threads\n";
            r.threads = p.env_cap;
            return r;
        }
        r.threads = static_cast<uint32_t>(v);
        return r;
    }
    }
    return r;
}

// ENGINE_THREADS now, under `p`; a Strict WARNING goes to stderr.
inline Result configured_result(const Policy& p) {
    Result r = resolve(std::getenv("ENGINE_THREADS"), p, hardware(), p.base == Base::Affinity ? affinity() : 0);
    if (!r.warning.empty()) std::fputs(r.warning.c_str(), stderr);
    return r;
}
inline uint32_t configured(const Policy& p) { return configured_result(p).threads; }

} // namespace threads
} // namespace engine

#endif // ENGINE_THREADS_H
