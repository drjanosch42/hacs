// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// NfCheck.h -- the NumberField library's check macros (slot-19 WP-NF).
//
// NumberField is STL-only (check_numberfield_includes.sh rule 1), so it may not
// include src/ModArithmetic/EngineCheck.h. These macros have exactly
// EngineCheck.h's semantics (ENGINE_CHECK always, ENGINE_DEBUG_CHECK not under NDEBUG); the
// precedent is ConsoleCore's CONSOLE_CHECK:
//   NF_CHECK(cond) / NF_CHECK_MSG(cond, msg)  -- always compiled in, O(1)
//       invariants; throw std::logic_error carrying file:line and the condition.
//   NF_DEBUG_CHECK(cond) / NF_DEBUG_CHECK_MSG -- compiled out under NDEBUG, for
//       checks whose cost scales with the input.
#include <sstream>
#include <stdexcept>
#include <string>

namespace nf {
namespace detail {
[[noreturn]] inline void check_failed(const char* file, int line, const char* cond, const std::string& msg) {
    std::ostringstream os;
    os << "NF_CHECK failed at " << file << ":" << line << ": " << cond;
    if (!msg.empty()) os << " -- " << msg;
    throw std::logic_error(os.str());
}
}  // namespace detail
}  // namespace nf

#define NF_CHECK(cond) \
    do { if (!(cond)) ::nf::detail::check_failed(__FILE__, __LINE__, #cond, std::string()); } while (0)
#define NF_CHECK_MSG(cond, msg) \
    do { if (!(cond)) ::nf::detail::check_failed(__FILE__, __LINE__, #cond, (msg)); } while (0)
#ifdef NDEBUG
#define NF_DEBUG_CHECK(cond) do { } while (0)
#define NF_DEBUG_CHECK_MSG(cond, msg) do { } while (0)
#else
#define NF_DEBUG_CHECK(cond) NF_CHECK(cond)
#define NF_DEBUG_CHECK_MSG(cond, msg) NF_CHECK_MSG(cond, msg)
#endif
