// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// BigInt.cpp -- see BigInt.h (slot-19 WP-NF, owner ruling O3).
#include "BigInt.h"

#include <algorithm>
#include <ostream>
#include <utility>

#include "NfCheck.h"

namespace nf {

namespace {
constexpr uint64_t kBase = 1ull << 32;
}

BigInt::BigInt(int64_t v) {
    if (v == 0) return;
    neg_ = v < 0;
    uint64_t m = neg_ ? (~static_cast<uint64_t>(v) + 1u) : static_cast<uint64_t>(v);
    while (m) {
        mag_.push_back(static_cast<uint32_t>(m));
        m >>= 32;
    }
}

BigInt BigInt::from_u64(uint64_t v) {
    BigInt r;
    while (v) {
        r.mag_.push_back(static_cast<uint32_t>(v));
        v >>= 32;
    }
    return r;
}

BigInt BigInt::from_magnitude(Mag m, bool neg) {
    BigInt r;
    r.mag_ = std::move(m);
    r.neg_ = neg;
    r.normalise();
    return r;
}

void BigInt::normalise() {
    while (!mag_.empty() && mag_.back() == 0) mag_.pop_back();
    if (mag_.empty()) neg_ = false;
}

BigInt BigInt::from_string(const std::string& s) {
    NF_CHECK_MSG(!s.empty(), "BigInt::from_string: empty string");
    std::size_t i = 0;
    bool neg = false;
    if (s[0] == '-' || s[0] == '+') {
        neg = s[0] == '-';
        i = 1;
    }
    NF_CHECK_MSG(i < s.size(), "BigInt::from_string: no digits");
    Mag m;
    while (i < s.size()) {
        // take up to 9 digits
        std::size_t take = std::min<std::size_t>(9, s.size() - i);
        uint32_t chunk = 0, mul = 1;
        for (std::size_t k = 0; k < take; ++k) {
            const char c = s[i + k];
            NF_CHECK_MSG(c >= '0' && c <= '9', "BigInt::from_string: not a decimal digit");
            chunk = chunk * 10u + static_cast<uint32_t>(c - '0');
            mul *= 10u;
        }
        i += take;
        uint64_t carry = chunk;
        for (auto& limb : m) {
            const uint64_t t = static_cast<uint64_t>(limb) * mul + carry;
            limb = static_cast<uint32_t>(t);
            carry = t >> 32;
        }
        if (carry) m.push_back(static_cast<uint32_t>(carry));
    }
    return from_magnitude(std::move(m), neg);
}

std::size_t BigInt::bit_length() const {
    if (mag_.empty()) return 0;
    uint32_t top = mag_.back();
    std::size_t b = 0;
    while (top) {
        ++b;
        top >>= 1;
    }
    return 32 * (mag_.size() - 1) + b;
}

bool BigInt::fits_int64() const {
    if (mag_.size() > 2) return false;
    uint64_t m = 0;
    for (std::size_t i = mag_.size(); i-- > 0;) m = (m << 32) | mag_[i];
    if (!neg_) return m <= static_cast<uint64_t>(INT64_MAX);
    return m <= static_cast<uint64_t>(INT64_MAX) + 1u;
}

int64_t BigInt::to_int64() const {
    NF_CHECK_MSG(fits_int64(), "BigInt::to_int64: out of range");
    uint64_t m = 0;
    for (std::size_t i = mag_.size(); i-- > 0;) m = (m << 32) | mag_[i];
    if (!neg_) return static_cast<int64_t>(m);
    return static_cast<int64_t>(~m + 1u);
}

std::string BigInt::to_string() const {
    if (mag_.empty()) return "0";
    Mag m = mag_;
    std::vector<uint32_t> chunks;  // base 10^9, little-endian
    while (!m.empty()) {
        chunks.push_back(divmod_small(m, 1000000000u));
    }
    std::string out = neg_ ? "-" : "";
    out += std::to_string(chunks.back());
    for (std::size_t i = chunks.size() - 1; i-- > 0;) {
        std::string c = std::to_string(chunks[i]);
        out.append(9 - c.size(), '0');
        out += c;
    }
    return out;
}

std::size_t BigInt::decimal_digits() const {
    const std::string s = to_string();
    return s.size() - (neg_ ? 1 : 0);
}

BigInt BigInt::operator-() const {
    BigInt r = *this;
    if (!r.mag_.empty()) r.neg_ = !r.neg_;
    return r;
}

BigInt BigInt::abs() const {
    BigInt r = *this;
    r.neg_ = false;
    return r;
}

int BigInt::cmp_mag(const Mag& a, const Mag& b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    for (std::size_t i = a.size(); i-- > 0;) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

int compare(const BigInt& a, const BigInt& b) {
    if (a.neg_ != b.neg_) return a.neg_ ? -1 : 1;
    const int c = BigInt::cmp_mag(a.mag_, b.mag_);
    return a.neg_ ? -c : c;
}

void BigInt::add_mag(Mag& a, const Mag& b) {
    if (a.size() < b.size()) a.resize(b.size(), 0);
    uint64_t carry = 0;
    std::size_t i = 0;
    for (; i < b.size(); ++i) {
        const uint64_t t = static_cast<uint64_t>(a[i]) + b[i] + carry;
        a[i] = static_cast<uint32_t>(t);
        carry = t >> 32;
    }
    for (; carry && i < a.size(); ++i) {
        const uint64_t t = static_cast<uint64_t>(a[i]) + carry;
        a[i] = static_cast<uint32_t>(t);
        carry = t >> 32;
    }
    if (carry) a.push_back(static_cast<uint32_t>(carry));
}

void BigInt::sub_mag(Mag& a, const Mag& b) {
    int64_t borrow = 0;
    std::size_t i = 0;
    for (; i < b.size(); ++i) {
        int64_t t = static_cast<int64_t>(a[i]) - b[i] - borrow;
        borrow = t < 0 ? 1 : 0;
        if (t < 0) t += static_cast<int64_t>(kBase);
        a[i] = static_cast<uint32_t>(t);
    }
    for (; borrow && i < a.size(); ++i) {
        int64_t t = static_cast<int64_t>(a[i]) - borrow;
        borrow = t < 0 ? 1 : 0;
        if (t < 0) t += static_cast<int64_t>(kBase);
        a[i] = static_cast<uint32_t>(t);
    }
    NF_DEBUG_CHECK(borrow == 0);
    while (!a.empty() && a.back() == 0) a.pop_back();
}

BigInt& BigInt::operator+=(const BigInt& o) {
    if (o.mag_.empty()) return *this;
    if (neg_ == o.neg_ || mag_.empty()) {
        if (mag_.empty()) neg_ = o.neg_;
        add_mag(mag_, o.mag_);
        return *this;
    }
    const int c = cmp_mag(mag_, o.mag_);
    if (c == 0) {
        mag_.clear();
        neg_ = false;
    } else if (c > 0) {
        sub_mag(mag_, o.mag_);
    } else {
        Mag t = o.mag_;
        sub_mag(t, mag_);
        mag_ = std::move(t);
        neg_ = o.neg_;
    }
    normalise();
    return *this;
}

BigInt& BigInt::operator-=(const BigInt& o) {
    if (o.mag_.empty()) return *this;
    BigInt n = -o;
    return *this += n;
}

BigInt::Mag BigInt::mul_schoolbook(const Mag& a, const Mag& b) {
    if (a.empty() || b.empty()) return Mag();
    Mag r(a.size() + b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        uint64_t carry = 0;
        const uint64_t ai = a[i];
        if (ai == 0) continue;
        for (std::size_t j = 0; j < b.size(); ++j) {
            const uint64_t t = ai * b[j] + r[i + j] + carry;
            r[i + j] = static_cast<uint32_t>(t);
            carry = t >> 32;
        }
        std::size_t k = i + b.size();
        while (carry) {
            const uint64_t t = static_cast<uint64_t>(r[k]) + carry;
            r[k] = static_cast<uint32_t>(t);
            carry = t >> 32;
            ++k;
        }
    }
    while (!r.empty() && r.back() == 0) r.pop_back();
    return r;
}

namespace {
// r += x << (32 * shift)
void add_shifted(BigInt::Mag& r, const BigInt::Mag& x, std::size_t shift) {
    if (x.empty()) return;
    if (r.size() < x.size() + shift) r.resize(x.size() + shift, 0);
    uint64_t carry = 0;
    std::size_t i = 0;
    for (; i < x.size(); ++i) {
        const uint64_t t = static_cast<uint64_t>(r[i + shift]) + x[i] + carry;
        r[i + shift] = static_cast<uint32_t>(t);
        carry = t >> 32;
    }
    for (std::size_t k = i + shift; carry; ++k) {
        if (k == r.size()) r.push_back(0);
        const uint64_t t = static_cast<uint64_t>(r[k]) + carry;
        r[k] = static_cast<uint32_t>(t);
        carry = t >> 32;
    }
}
BigInt::Mag slice(const BigInt::Mag& a, std::size_t lo, std::size_t hi) {
    hi = std::min(hi, a.size());
    if (lo >= hi) return BigInt::Mag();
    BigInt::Mag r(a.begin() + static_cast<std::ptrdiff_t>(lo), a.begin() + static_cast<std::ptrdiff_t>(hi));
    while (!r.empty() && r.back() == 0) r.pop_back();
    return r;
}
}  // namespace

BigInt::Mag BigInt::mul_mag(const Mag& a, const Mag& b) {
    if (a.empty() || b.empty()) return Mag();
    if (a.size() < kKaratsubaLimbs || b.size() < kKaratsubaLimbs) return mul_schoolbook(a, b);
    const std::size_t h = (std::max(a.size(), b.size()) + 1) / 2;
    if (b.size() <= h || a.size() <= h) {
        // unbalanced: split only the longer operand
        const Mag& L = a.size() >= b.size() ? a : b;
        const Mag& S = a.size() >= b.size() ? b : a;
        Mag r = mul_mag(slice(L, 0, h), S);
        add_shifted(r, mul_mag(slice(L, h, L.size()), S), h);
        while (!r.empty() && r.back() == 0) r.pop_back();
        return r;
    }
    const Mag a0 = slice(a, 0, h), a1 = slice(a, h, a.size());
    const Mag b0 = slice(b, 0, h), b1 = slice(b, h, b.size());
    Mag z0 = mul_mag(a0, b0);
    Mag z2 = mul_mag(a1, b1);
    Mag sa = a0, sb = b0;
    add_mag(sa, a1);
    add_mag(sb, b1);
    Mag z1 = mul_mag(sa, sb);
    sub_mag(z1, z0);
    sub_mag(z1, z2);
    Mag r = z0;
    add_shifted(r, z1, h);
    add_shifted(r, z2, 2 * h);
    while (!r.empty() && r.back() == 0) r.pop_back();
    return r;
}

BigInt operator*(const BigInt& a, const BigInt& b) {
    return BigInt::from_magnitude(BigInt::mul_mag(a.mag_, b.mag_), a.neg_ != b.neg_);
}

BigInt& BigInt::operator*=(const BigInt& o) {
    *this = *this * o;
    return *this;
}

uint32_t BigInt::divmod_small(Mag& a, uint32_t d) {
    NF_CHECK(d != 0);
    uint64_t rem = 0;
    for (std::size_t i = a.size(); i-- > 0;) {
        const uint64_t cur = (rem << 32) | a[i];
        a[i] = static_cast<uint32_t>(cur / d);
        rem = cur % d;
    }
    while (!a.empty() && a.back() == 0) a.pop_back();
    return static_cast<uint32_t>(rem);
}

// Knuth, TAOCP vol. 2, 4.3.1, Algorithm D (base 2^32), magnitudes only.
void BigInt::divmod_mag(const Mag& a, const Mag& b, Mag& q, Mag& r) {
    NF_CHECK_MSG(!b.empty(), "BigInt: division by zero");
    if (cmp_mag(a, b) < 0) {
        q.clear();
        r = a;
        return;
    }
    if (b.size() == 1) {
        q = a;
        const uint32_t rem = divmod_small(q, b[0]);
        r.clear();
        if (rem) r.push_back(rem);
        return;
    }
    // normalise so that the top limb of the divisor has its high bit set
    int s = 0;
    {
        uint32_t t = b.back();
        while (!(t & 0x80000000u)) {
            t <<= 1;
            ++s;
        }
    }
    const std::size_t n = b.size(), m = a.size() - b.size();
    Mag v(n), u(a.size() + 1);
    for (std::size_t i = n; i-- > 0;) {
        v[i] = (b[i] << s) | (s && i ? static_cast<uint32_t>(static_cast<uint64_t>(b[i - 1]) >> (32 - s)) : 0u);
    }
    u[a.size()] = s ? static_cast<uint32_t>(static_cast<uint64_t>(a.back()) >> (32 - s)) : 0u;
    for (std::size_t i = a.size(); i-- > 0;) {
        u[i] = (a[i] << s) | (s && i ? static_cast<uint32_t>(static_cast<uint64_t>(a[i - 1]) >> (32 - s)) : 0u);
    }
    q.assign(m + 1, 0);
    for (std::size_t j = m + 1; j-- > 0;) {
        const uint64_t num = (static_cast<uint64_t>(u[j + n]) << 32) | u[j + n - 1];
        uint64_t qhat = num / v[n - 1];
        uint64_t rhat = num % v[n - 1];
        while (qhat >= kBase || qhat * v[n - 2] > ((rhat << 32) | u[j + n - 2])) {
            --qhat;
            rhat += v[n - 1];
            if (rhat >= kBase) break;
        }
        // multiply and subtract
        int64_t borrow = 0;
        uint64_t carry = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const uint64_t p = qhat * v[i] + carry;
            carry = p >> 32;
            const int64_t t = static_cast<int64_t>(u[i + j]) - borrow - static_cast<int64_t>(p & 0xffffffffu);
            u[i + j] = static_cast<uint32_t>(t);
            borrow = t < 0 ? 1 : 0;
        }
        const int64_t t = static_cast<int64_t>(u[j + n]) - borrow - static_cast<int64_t>(carry);
        u[j + n] = static_cast<uint32_t>(t);
        if (t < 0) {
            // add back
            --qhat;
            uint64_t c = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const uint64_t s2 = static_cast<uint64_t>(u[i + j]) + v[i] + c;
                u[i + j] = static_cast<uint32_t>(s2);
                c = s2 >> 32;
            }
            u[j + n] = static_cast<uint32_t>(static_cast<uint64_t>(u[j + n]) + c);
        }
        q[j] = static_cast<uint32_t>(qhat);
    }
    while (!q.empty() && q.back() == 0) q.pop_back();
    // unnormalise the remainder
    r.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = (u[i] >> s) | (s ? static_cast<uint32_t>(static_cast<uint64_t>(u[i + 1]) << (32 - s)) : 0u);
    }
    while (!r.empty() && r.back() == 0) r.pop_back();
}

void BigInt::divmod(const BigInt& a, const BigInt& b, BigInt& q, BigInt& r) {
    NF_CHECK_MSG(!b.is_zero(), "BigInt: division by zero");
    Mag qm, rm;
    divmod_mag(a.mag_, b.mag_, qm, rm);
    q = from_magnitude(std::move(qm), a.neg_ != b.neg_);
    r = from_magnitude(std::move(rm), a.neg_);
}

BigInt operator/(const BigInt& a, const BigInt& b) {
    BigInt q, r;
    BigInt::divmod(a, b, q, r);
    return q;
}

BigInt operator%(const BigInt& a, const BigInt& b) {
    BigInt q, r;
    BigInt::divmod(a, b, q, r);
    return r;
}

BigInt& BigInt::operator/=(const BigInt& o) {
    *this = *this / o;
    return *this;
}

BigInt& BigInt::operator%=(const BigInt& o) {
    *this = *this % o;
    return *this;
}

BigInt BigInt::floor_mod(const BigInt& m) const {
    NF_CHECK(m.sign() > 0);
    BigInt r = *this % m;
    if (r.neg_) r += m;
    return r;
}

uint32_t BigInt::mod_u32(uint32_t m) const {
    NF_CHECK(m != 0);
    uint64_t rem = 0;
    for (std::size_t i = mag_.size(); i-- > 0;) rem = ((rem << 32) | mag_[i]) % m;
    if (neg_ && rem) rem = m - rem;
    return static_cast<uint32_t>(rem);
}

BigInt BigInt::div_exact_u32(uint32_t d) const {
    Mag m = mag_;
    const uint32_t rem = divmod_small(m, d);
    NF_CHECK_MSG(rem == 0, "BigInt::div_exact_u32: not exact");
    return from_magnitude(std::move(m), neg_);
}

BigInt BigInt::pow(BigInt base, uint64_t e) {
    BigInt r(1);
    while (e) {
        if (e & 1u) r *= base;
        e >>= 1;
        if (e) base = base * base;
    }
    return r;
}

BigInt BigInt::gcd(BigInt a, BigInt b) {
    a = a.abs();
    b = b.abs();
    while (!b.is_zero()) {
        BigInt t = a % b;
        a = std::move(b);
        b = std::move(t);
    }
    return a;
}

unsigned BigInt::valuation(uint32_t p) const {
    NF_CHECK_MSG(!is_zero(), "BigInt::valuation of zero");
    NF_CHECK(p >= 2);
    unsigned v = 0;
    Mag m = mag_;
    for (;;) {
        Mag t = m;
        if (divmod_small(t, p) != 0) break;
        m = std::move(t);
        ++v;
    }
    return v;
}

std::ostream& operator<<(std::ostream& os, const BigInt& x) { return os << x.to_string(); }

}  // namespace nf
