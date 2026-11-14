// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// BigNtt.h -- the NTT product and the Barrett modular product of nf::BigInt magnitudes (slot-27 BIGNTT M1,
// docs/implementation/slot27/BIGNTT_design.md v0.2 Sec. 2.2-2.3).
//
// An EXPLICIT API (design v0.2, audit M3): nothing here is reached from BigInt::mul_mag or BigInt::operator/,
// so no other NumberField consumer changes route or thread use. The only production caller is the UNTRUSTED
// big-integer factor finder (Integral/HeckeZFactorFinder.cpp): no multiprecision arithmetic of this file
// enters the certified build ([C] Q1(a)).
//
// Exactness. A magnitude (base 2^32 limbs) is split into 16-bit chunks; the cyclic convolution of length
// L >= len_a + len_b - 1 (chunks) over the 62-bit prime p = ntt64_prime(0) equals the integer convolution,
// since every coefficient is <= min(len) (2^16 - 1)^2 <= (L / 2) 2^32 <= 2^28 2^32 = 2^60 < 2^61 < p
// (NF_CHECK(L <= 2^29); M1b, code audit m3); a carry pass in one 64-bit word (no __int128) turns it into the
// unique product. So mul_ntt == mul_schoolbook for every
// input, and every Barrett result is the unique residue: values never depend on the route, the threshold or
// the thread count (Ntt64Plan is thread-count invariant; the pointwise and carry passes are range-owned).
//
// Threads: every entry point takes an explicit thread count, default 1 (audit M3).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "BigInt.h"

namespace nf {

class Ntt64Plan;

namespace bigntt {

using Mag = BigInt::Mag;

// The default route threshold in limbs of the modulus (M1 measurement: the Barrett NTT step overtakes the
// Karatsuba + Knuth-D step at ~ 30 kbit). A parameter of every caller, so a validation can force the route.
constexpr std::size_t kDefaultMinLimbs = 1024;

// The thread count actually used at transform length L for a requested count: 1 below 2^14 points, else at most
// one per 4 096 points (pool overhead; values never depend on it).
unsigned effective_threads(std::size_t L, unsigned threads);

// Exact products (any sizes, any thread count).
Mag mul_ntt(const Mag& a, const Mag& b, unsigned threads = 1);
Mag sqr_ntt(const Mag& a, unsigned threads = 1);

// Counters of the NTT products and Barrett steps actually taken (relaxed atomics; the validation's EFFECTIVE line).
uint64_t ntt_products();
uint64_t barrett_steps();

// The plan of length L over ntt64_prime(0) for a thread count, built once per process and shared (thread-safe).
std::shared_ptr<const Ntt64Plan> plan_for(std::size_t L, unsigned threads = 1);

// A modulus m >= 2 with Barrett's reduction (HAC 14.42, b = 2^32, k = limbs of m): mu = floor(b^{2k} / m)
// once (one Knuth-D division), then per product the square / product, q1 mu, q3 m (the transforms of mu and m
// are cached at one length L), r = (x - q3 m) mod b^{k+1} and at most two subtractions of m (NF_CHECK).
// For m = b^{k-1} (mu = b^{k+1} would not fit in k + 1 limbs; M1b, code audit m1) the reduction is the mask to
// the low k - 1 limbs. If m has fewer than min_limbs limbs the object uses BigInt's own route, (x y) % m, unchanged.
class BarrettModulus {
public:
    BarrettModulus(const BigInt& m, std::size_t min_limbs = kDefaultMinLimbs, unsigned threads = 1);

    const BigInt& modulus() const { return m_; }
    bool ntt_route() const { return ntt_; }
    unsigned threads() const { return threads_; }   // the effective count (effective_threads)

    // x y mod m and x^2 mod m for 0 <= x, y < m. A negative operand throws (NF_CHECK, both routes; code audit m2).
    BigInt mulmod(const BigInt& x, const BigInt& y) const;
    BigInt sqrmod(const BigInt& x) const;

private:
    Mag reduce(const Mag& x) const;                 // x < b^{2k}: x mod m
    Mag mul_cached(const Mag& a, const std::vector<uint64_t>& bhat, std::size_t keep_limbs) const;

    BigInt m_;
    bool ntt_ = false;
    bool pow_b_ = false;                            // m = b^{k-1}: reduce() is a mask
    unsigned threads_ = 1;
    std::size_t k_ = 0, L_ = 0;
    Mag mmag_, mu_;
    std::shared_ptr<const Ntt64Plan> plan_;
    std::vector<uint64_t> mu_hat_, m_hat_;          // forward transforms, pre-scaled for the pointwise product
};

// b^e mod M.m with the control flow of the finder's PowModBig (right to left over the bits of e; r = r b
// when the bit is set, then b = b^2), the primitive being M.mulmod / M.sqrmod. e >= 0; any sign of the base
// (reduced by floor_mod, so the result is the residue in [0, m) on both routes; code audit m2).
BigInt powmod(const BigInt& base, const BigInt& e, const BarrettModulus& M);

}  // namespace bigntt
}  // namespace nf
