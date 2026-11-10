// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// BigRational.h -- exact rationals over nf::BigInt, and the exact Bernoulli
// numbers ([E] sec:lvnf / rmk:lvgates: "exact rationals from WP-NF's BigInt").
// Always normalised: gcd(num, den) = 1, den > 0, zero = 0/1.
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "BigInt.h"

namespace nf {

class BigRational {
public:
    BigRational() : num_(0), den_(1) {}
    BigRational(int64_t v) : num_(v), den_(1) {}  // NOLINT(google-explicit-constructor)
    BigRational(const BigInt& v) : num_(v), den_(1) {}  // NOLINT(google-explicit-constructor)
    BigRational(const BigInt& num, const BigInt& den);  // NF_CHECK(den != 0)

    const BigInt& num() const { return num_; }
    const BigInt& den() const { return den_; }
    bool is_zero() const { return num_.is_zero(); }
    std::string to_string() const;  // "p/q" or "p"

    friend BigRational operator+(const BigRational& a, const BigRational& b);
    friend BigRational operator-(const BigRational& a, const BigRational& b);
    friend BigRational operator*(const BigRational& a, const BigRational& b);
    friend BigRational operator/(const BigRational& a, const BigRational& b);
    BigRational operator-() const;
    BigRational& operator+=(const BigRational& o) { return *this = *this + o; }
    BigRational& operator-=(const BigRational& o) { return *this = *this - o; }
    BigRational& operator*=(const BigRational& o) { return *this = *this * o; }
    BigRational& operator/=(const BigRational& o) { return *this = *this / o; }
    friend bool operator==(const BigRational& a, const BigRational& b) { return a.num_ == b.num_ && a.den_ == b.den_; }
    friend bool operator!=(const BigRational& a, const BigRational& b) { return !(a == b); }

    // v_p of the rational (num valuation minus den valuation); NF_CHECK(!is_zero()).
    int valuation(uint32_t p) const;
    // The residue mod p^k of a p-integral rational (v_p >= 0); NF_CHECK otherwise.
    uint64_t mod_prime_power(uint32_t p, unsigned k) const;

private:
    void normalise();
    BigInt num_, den_;
};

std::ostream& operator<<(std::ostream& os, const BigRational& x);

// B_0 .. B_N (B_1 = -1/2), exact, by the Akiyama--Tanigawa algorithm.
std::vector<BigRational> bernoulli_numbers(unsigned N);

}  // namespace nf
