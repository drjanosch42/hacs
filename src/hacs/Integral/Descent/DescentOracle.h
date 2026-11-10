// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_ORACLE_H
#define INTEGRAL_DESCENT_DESCENT_ORACLE_H

// The independent oracles of DESCENT D0 (docs/implementation/slot24/DESCENT_plan.md, D0 arms (A), (T); audit M7).
//   * AppendixAGroups(): the frozen O1 table, H^q(SL_2(Z); Sym^n Z^2), q = 0..6, n = 0..20 -- the "group" rows of
//     docs/implementation/integral_line_probe.py (the tree, exact integer SNF, independent Python), = [IB] Appendix A.
//     Moved verbatim from HeckeFreeModelValidation.cpp (one copy).
//   * LoadOracleTable(path): the table of record of the probe extended to n <= 128
//     (docs/implementation/slot24/descent_oracle_n128.tsv).

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../Model/IntegralGroups.h"

namespace integral_descent {

using integral_model::AbelianGroupType;

// Parses one "Z^r+Z/p^a+..." term (the probe's format) into a group type.
AbelianGroupType ParseProbeGroup(const std::string& term);
// table[n][q], n = 0..20, q = 0..6.
const std::vector<std::vector<AbelianGroupType>>& AppendixAGroups();
// The oracle table's form: "0", or [Z^r +] Z/p^k + ... (free part omitted when 0).
AbelianGroupType ParseOracleGroup(const std::string& term);

struct OracleTable {
    std::string path;
    bool found = false;
    std::string error;                                          // non-empty: the file was unreadable / malformed
    std::map<uint32_t, std::vector<AbelianGroupType>> groups;   // n -> H^0, H^1, ...
};
OracleTable LoadOracleTable(const std::string& path);

}  // namespace integral_descent

#endif
