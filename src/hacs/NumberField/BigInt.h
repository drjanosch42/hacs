// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// BigInt.h -- the in-house multiprecision integer of the NumberField library
// (slot-19 WP-NF; owner ruling O3: our own BigInt, no GMP, no CAS).
//
// Sign-magnitude, magnitude little-endian in base 2^32 (std::vector<uint32_t>,
// normalised: no leading zero limb; zero is the empty vector with neg = false).
// Products: schoolbook below kKaratsubaLimbs limbs, Karatsuba above (enough for
// the ~24 800-digit coefficients of eta_12 at 691, [E] sec:lvnf). Division:
// Knuth's Algorithm D. Every intermediate is a uint64_t -- no __int128, so the
// code is the same on x86-64 and aarch64 (HPC-portable).
//
// Division semantics are C++'s: quotient truncated toward zero, remainder with
// the sign of the dividend. floor_mod() gives the non-negative residue.
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace nf {

class BigInt {
public:
    static constexpr std::size_t kKaratsubaLimbs = 32;

    BigInt() = default;
    BigInt(int64_t v);  // NOLINT(google-explicit-constructor): integer literals are BigInts
    static BigInt from_u64(uint64_t v);
    static BigInt from_string(const std::string& decimal);  // optional leading '-'

    bool is_zero() const { return mag_.empty(); }
    bool is_neg() const { return neg_; }
    int sign() const { return mag_.empty() ? 0 : (neg_ ? -1 : 1); }
    std::size_t limbs() const { return mag_.size(); }
    std::size_t bit_length() const;
    bool fits_int64() const;
    int64_t to_int64() const;  // NF_CHECK(fits_int64())
    std::string to_string() const;
    std::size_t decimal_digits() const;  // of |x|; 1 for zero

    BigInt operator-() const;
    BigInt abs() const;

    BigInt& operator+=(const BigInt& o);
    BigInt& operator-=(const BigInt& o);
    BigInt& operator*=(const BigInt& o);
    BigInt& operator/=(const BigInt& o);
    BigInt& operator%=(const BigInt& o);

    friend BigInt operator+(BigInt a, const BigInt& b) { a += b; return a; }
    friend BigInt operator-(BigInt a, const BigInt& b) { a -= b; return a; }
    friend BigInt operator*(const BigInt& a, const BigInt& b);
    friend BigInt operator/(const BigInt& a, const BigInt& b);
    friend BigInt operator%(const BigInt& a, const BigInt& b);

    // Truncated division: a = q b + r, |r| < |b|, sign(r) = sign(a). NF_CHECK(b != 0).
    static void divmod(const BigInt& a, const BigInt& b, BigInt& q, BigInt& r);
    // Non-negative residue mod m (m > 0).
    BigInt floor_mod(const BigInt& m) const;
    uint32_t mod_u32(uint32_t m) const;  // non-negative residue, m > 0
    // Exact division by a small divisor; NF_CHECK that it is exact.
    BigInt div_exact_u32(uint32_t d) const;

    static BigInt pow(BigInt base, uint64_t e);
    static BigInt gcd(BigInt a, BigInt b);  // >= 0
    // v_p(x) for a prime (or any integer >= 2) p; NF_CHECK(x != 0).
    unsigned valuation(uint32_t p) const;

    friend int compare(const BigInt& a, const BigInt& b);
    friend bool operator==(const BigInt& a, const BigInt& b) { return a.neg_ == b.neg_ && a.mag_ == b.mag_; }
    friend bool operator!=(const BigInt& a, const BigInt& b) { return !(a == b); }
    friend bool operator<(const BigInt& a, const BigInt& b) { return compare(a, b) < 0; }
    friend bool operator>(const BigInt& a, const BigInt& b) { return compare(a, b) > 0; }
    friend bool operator<=(const BigInt& a, const BigInt& b) { return compare(a, b) <= 0; }
    friend bool operator>=(const BigInt& a, const BigInt& b) { return compare(a, b) >= 0; }

    // Magnitude primitives (exposed for the self-tests: schoolbook vs Karatsuba).
    using Mag = std::vector<uint32_t>;
    static Mag mul_schoolbook(const Mag& a, const Mag& b);
    static Mag mul_mag(const Mag& a, const Mag& b);  // Karatsuba above the threshold
    const Mag& magnitude() const { return mag_; }
    static BigInt from_magnitude(Mag m, bool neg);

private:
    void normalise();
    static int cmp_mag(const Mag& a, const Mag& b);
    static void add_mag(Mag& a, const Mag& b);          // a += b
    static void sub_mag(Mag& a, const Mag& b);          // a -= b, requires a >= b
    static uint32_t divmod_small(Mag& a, uint32_t d);   // a /= d, returns remainder
    static void divmod_mag(const Mag& a, const Mag& b, Mag& q, Mag& r);

    bool neg_ = false;
    Mag mag_;
};

std::ostream& operator<<(std::ostream& os, const BigInt& x);

}  // namespace nf
