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
//    `rounds` primes) on the integer or on its exact square root; the label is `probable`, never `pratt` (Q1(a)). Slot-27 BIGNTT
//    M1: its x y mod n is nf::bigntt::BarrettModulus (NTT + Barrett above the route threshold; exact, so every outcome is today's).
// NumberField is used only through this file (check_numberfield_includes.sh unaffected); slot-27 BIGNTT M1 adds nf::bigntt.
// ===========================================================================

#include "HeckeZRational.h"

namespace nf { class BigInt; }

#include <cstddef>
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
// threads: the NTT threads of the Barrett products (the caller's Garner thread count; values never depend on it).
// min_limbs: the NTT route threshold in limbs of the modulus; 0 = ENGINE_BIGNTT_MIN_LIMBS or nf::bigntt::kDefaultMinLimbs
// (slot-27 BIGNTT M1; a route knob, it cannot change a value).
BigProbable BigProbablePrime(const std::string& value_decimal, uint32_t rounds, unsigned threads = 1, std::size_t min_limbs = 0);
// The finder's Miller--Rabin itself, for BigIntNttValidation (same code path as BigProbablePrime).
bool MillerRabinBigForTest(const nf::BigInt& n, uint32_t rounds, std::size_t min_limbs, unsigned threads);
// The effective default route threshold (ENGINE_BIGNTT_MIN_LIMBS, else nf::bigntt::kDefaultMinLimbs).
std::size_t BigNttMinLimbs();

}  // namespace finder
}  // namespace heckez

#endif
