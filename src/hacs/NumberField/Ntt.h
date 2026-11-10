// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Ntt.h -- exact integer convolution reduced mod M (M < 2^32) by a multimodular
// number-theoretic transform (slot-19 WP-NF).
//
// Two or three NTT primes (998244353, 469762049, 167772161, all with primitive
// root 3) are chosen so that the product of the primes exceeds the largest
// possible exact coefficient min(len a, len b) * (M - 1)^2; the residues are
// recombined by Garner's algorithm directly mod M. Everything is uint64_t
// arithmetic (no __int128, no floating point), so the result is exact and
// identical on every platform. Transform length <= 2^23.
#include <cstdint>
#include <vector>

namespace nf {

// Linear convolution c[k] = sum_{i+j=k} a[i] b[j] mod M, length |a| + |b| - 1.
// Entries of a, b must be < M. Small inputs use the schoolbook product.
std::vector<uint32_t> convolve_mod(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, uint32_t M);

// The schoolbook reference (also used below the NTT threshold).
std::vector<uint32_t> convolve_mod_schoolbook(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b,
                                              uint32_t M);

// Number of NTT primes convolve_mod would use (0 = schoolbook), for reports.
int ntt_prime_count(std::size_t len_a, std::size_t len_b, uint32_t M);

}  // namespace nf
