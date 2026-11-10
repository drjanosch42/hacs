// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_FACTOR_FINDER_H
#define INTEGRAL_HECKEZ_FACTOR_FINDER_H

// ===========================================================================
// HeckeZFactorFinder -- the UNTRUSTED side of HZ-RAT (slot-25 plan Sec. 2.2, audit F2; 04 Sec. 1.5 "factor_budget").
// Everything here is a proposal: HeckeZRational re-checks every factor it accepts (exact re-division in 128 bits, Pratt certificates
// verified by VerifyPratt). Nothing returned here enters a result field unchecked; a wrong answer can only make a check fail, i.e.
// leave a cofactor `unfactored`.
//  * FindWordFactors: nf::factor_u64 (trial division < 1000, Pollard--Brent with seeds c = 1 .. 39, r <= 2^26 -- its fixed counts).
//  * FindPratt: a Pratt certificate for a prime q < 2^64 (q - 1 split by nf::factor_u64, witnesses 2, 3, ... up to witness_cap,
//    recursion up to node_cap certificates).
//  * BigProbablePrime: for an integer above 2^64 given by its decimal text, an nf::BigInt Miller--Rabin with fixed bases (the first
//    `rounds` primes) on the integer or on its exact square root; the label is `probable`, never `pratt` (Q1(a)).
// NumberField is used unchanged and only through this file (check_numberfield_includes.sh unaffected).
// ===========================================================================

#include "HeckeZRational.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace heckez {
namespace finder {

bool FindWordFactors(uint64_t c, std::vector<std::pair<uint64_t, unsigned>>& out);
bool FindPratt(uint64_t q, uint32_t witness_cap, uint32_t node_cap, PrattCert* out);

struct BigProbable {
    bool found = false;
    std::string root_decimal;        // q (decimal), with value = q^e
    uint32_t e = 0;
};
BigProbable BigProbablePrime(const std::string& value_decimal, uint32_t rounds);

}  // namespace finder
}  // namespace heckez

#endif
