// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// BigRational.cpp -- see BigRational.h (slot-19 WP-NF).
#include "BigRational.h"

#include <ostream>

#include "ModUtil.h"
#include "NfCheck.h"

namespace nf {

BigRational::BigRational(const BigInt& num, const BigInt& den) : num_(num), den_(den) {
    NF_CHECK_MSG(!den.is_zero(), "BigRational: zero denominator");
    normalise();
}

void BigRational::normalise() {
    if (den_.is_neg()) {
        num_ = -num_;
        den_ = -den_;
    }
    if (num_.is_zero()) {
        den_ = BigInt(1);
        return;
    }
    const BigInt g = BigInt::gcd(num_, den_);
    if (g != BigInt(1)) {
        num_ /= g;
        den_ /= g;
    }
}

std::string BigRational::to_string() const {
    if (den_ == BigInt(1)) return num_.to_string();
    return num_.to_string() + "/" + den_.to_string();
}

BigRational operator+(const BigRational& a, const BigRational& b) {
    return BigRational(a.num_ * b.den_ + b.num_ * a.den_, a.den_ * b.den_);
}
BigRational operator-(const BigRational& a, const BigRational& b) {
    return BigRational(a.num_ * b.den_ - b.num_ * a.den_, a.den_ * b.den_);
}
BigRational operator*(const BigRational& a, const BigRational& b) {
    return BigRational(a.num_ * b.num_, a.den_ * b.den_);
}
BigRational operator/(const BigRational& a, const BigRational& b) {
    NF_CHECK_MSG(!b.is_zero(), "BigRational: division by zero");
    return BigRational(a.num_ * b.den_, a.den_ * b.num_);
}
BigRational BigRational::operator-() const {
    BigRational r = *this;
    r.num_ = -r.num_;
    return r;
}

int BigRational::valuation(uint32_t p) const {
    NF_CHECK(!is_zero());
    return static_cast<int>(num_.valuation(p)) - static_cast<int>(den_.valuation(p));
}

uint64_t BigRational::mod_prime_power(uint32_t p, unsigned k) const {
    const uint64_t m = ipow_u64(p, k);
    NF_CHECK_MSG(m <= 0xffffffffull, "BigRational::mod_prime_power: p^k must fit in 32 bits");
    if (is_zero()) return 0;
    NF_CHECK_MSG(den_.mod_u32(p) != 0, "BigRational::mod_prime_power: not p-integral");
    const uint32_t M = static_cast<uint32_t>(m);
    return mulmod32(num_.mod_u32(M), invmod32(den_.mod_u32(M), M), M);
}

std::vector<BigRational> bernoulli_numbers(unsigned N) {
    // Akiyama--Tanigawa: a_m = 1/(m+1); a_{j-1} = j (a_{j-1} - a_j); B_m = a_0 (gives B_1 = +1/2).
    std::vector<BigRational> B(N + 1), a(N + 1);
    for (unsigned m = 0; m <= N; ++m) {
        a[m] = BigRational(BigInt(1), BigInt(static_cast<int64_t>(m) + 1));
        for (unsigned j = m; j >= 1; --j) {
            a[j - 1] = BigRational(static_cast<int64_t>(j)) * (a[j - 1] - a[j]);
        }
        B[m] = a[0];
    }
    if (N >= 1) B[1] = BigRational(BigInt(-1), BigInt(2));
    return B;
}

std::ostream& operator<<(std::ostream& os, const BigRational& x) { return os << x.to_string(); }

}  // namespace nf
