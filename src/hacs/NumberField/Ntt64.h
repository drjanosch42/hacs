// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Ntt64.h -- large number-theoretic transforms over 62-bit primes (slot-19
// WP-NF2, the product engine of the Coates--Wiles layer route CwLayer.h).
//
// Primes p = c * 15 * 2^32 + 1 < 2^62 (largest c first), so every length
// L = f * 2^a with f in {1, 3, 5}, a <= 32, divides p - 1. A transform of
// length L is an outer radix-f pass (twiddled) followed by f power-of-two
// transforms of length M = L / f; M > 2^direct_log is done four-step
// (R x C, column DIF + twiddle omega_M^{n2 k1}, row DIF), so every table is
// O(sqrt L) and every sub-transform is cache-sized. The forward transform
// leaves its output in a fixed permutation of the frequency order; inverse()
// accepts exactly that permutation and returns L times the input, so a
// convolution is forward(a), forward(b), pointwise mul, inverse, scale.
// Arithmetic is Montgomery (Mont.h); data are in normal form, tables in
// Montgomery form. Deterministic: the result never depends on the thread count.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "Mont.h"

namespace nf {

// The first `count` NTT primes (count <= 32), largest first; each with a primitive root.
const std::vector<uint64_t>& ntt64_primes();
uint64_t ntt64_prime(std::size_t i);
uint64_t ntt64_primitive_root(uint64_t p);

// The smallest supported length f * 2^a (f in {1,3,5}) that is >= min_len.
std::size_t ntt64_length(std::size_t min_len);

class Ntt64Plan {
public:
    Ntt64Plan(uint64_t prime, std::size_t L, unsigned direct_log = 15);

    std::size_t length() const { return L_; }
    unsigned radix() const { return f_; }
    bool four_step() const { return R_ > 1; }
    const Mont64& mont() const { return mt_; }

    void forward(uint64_t* a, unsigned threads) const;
    void inverse(uint64_t* a, unsigned threads) const;

private:
    void pow2_forward(uint64_t* x, unsigned threads) const;
    void pow2_inverse(uint64_t* x, unsigned threads) const;
    inline uint64_t omegaM(std::size_t e, bool inv) const;
    inline uint64_t omegaL(std::size_t e, bool inv) const;

    Mont64 mt_;
    std::size_t L_ = 0, M_ = 0, R_ = 1, C_ = 0;
    unsigned f_ = 1, logM_ = 0, hM_ = 0, hL_ = 16;
    std::vector<uint64_t> twR_[2], twC_[2];          // tw[half + k] = w_{2 half}^{+-k}
    std::vector<uint64_t> aM_[2], bM_[2];            // omega_M^{+-e} = aM[e & mask] * bM[e >> hM]
    std::vector<uint64_t> aL_[2], bL_[2];            // omega_L^{+-e}, the same two-level split (hL = 16)
    std::vector<uint64_t> wf_[2];                    // omega_f^{+-j}
    std::vector<uint32_t> revR_;                     // bit reversal on log2 R bits
};

}  // namespace nf
