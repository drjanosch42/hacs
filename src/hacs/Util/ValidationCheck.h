// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Util/ValidationCheck.h -- the validations' check counter, ONE definition
// (S24 CL-1; CODEAUDIT A4, B2; docs/implementation/slot24/CLEANUP_plan_1030.md)
// ===========================================================================
//
// 56 validation TUs carry their own `void Check(bool ok, const std::string&)`
// with file-global g_checks / g_fails.  They differ in three ways, all kept
// per site so that no transcript line moves:
//   prefix       the full text before `what`, indentation included:
//                "  [FAIL] " (most), "    [FAIL] ", "  FAIL: ", "  *** FAIL: ",
//                "  *** FAIL [<tag>] " (census-cell-list, atlas, e2e, ...);
//   newline      std::endl (flush; most) or "\n";
//   print_limit  0 = print every failure; N = print only the first N (the
//                copies' `if (g_fails <= N)` after the increment: 25/40/60/80).
// operator() returns `ok` (some copies return it, most return void).
// Header-only, STL only.
// ===========================================================================
#pragma once
#ifndef OC_UTIL_VALIDATION_CHECK_H
#define OC_UTIL_VALIDATION_CHECK_H

#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>

namespace oc_util {

struct Checker {
    std::string prefix = "  [FAIL] ";
    bool endl = true;              // std::endl (true) or "\n" (false)
    uint32_t print_limit = 0;      // 0 = unlimited
    std::ostream* os = &std::cout;
    uint32_t checks = 0;
    uint32_t fails = 0;

    Checker() = default;
    explicit Checker(std::string prefix_, bool endl_ = true, uint32_t print_limit_ = 0, std::ostream* os_ = &std::cout)
        : prefix(std::move(prefix_)), endl(endl_), print_limit(print_limit_), os(os_) {}

    bool operator()(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (print_limit == 0 || fails <= print_limit) {
                *os << prefix << what;
                if (endl) *os << std::endl;
                else *os << "\n";
            }
        }
        return ok;
    }
    void reset() { checks = fails = 0; }
    // "<name>: <checks> checks, <fails> failing" -- the closing line most copies print.
    std::string summary(const std::string& name) const {
        return name + ": " + std::to_string(checks) + " checks, " + std::to_string(fails) + " failing";
    }
    uint32_t exit_code() const { return fails ? 1u : 0u; }
};

} // namespace oc_util

#endif // OC_UTIL_VALIDATION_CHECK_H
