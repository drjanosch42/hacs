// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// MpReal.cpp -- see MpReal.h.
#include "MpReal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "NfCheck.h"

namespace nf {

BigInt bigint_shl(const BigInt& x, std::size_t bits) {
    if (x.is_zero() || bits == 0) return x;
    const BigInt::Mag& a = x.magnitude();
    const std::size_t L = bits / 32, b = bits % 32;
    BigInt::Mag r(a.size() + L + 1, 0);
    if (b == 0) {
        for (std::size_t i = 0; i < a.size(); ++i) r[i + L] = a[i];
    } else {
        uint32_t carry = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            r[i + L] = (a[i] << b) | carry;
            carry = a[i] >> (32 - b);
        }
        r[a.size() + L] = carry;
    }
    return BigInt::from_magnitude(std::move(r), x.is_neg());
}

BigInt bigint_shr(const BigInt& x, std::size_t bits) {
    if (x.is_zero() || bits == 0) return x;
    const BigInt::Mag& a = x.magnitude();
    const std::size_t L = bits / 32, b = bits % 32;
    if (L >= a.size()) return BigInt(0);
    BigInt::Mag r(a.size() - L, 0);
    if (b == 0) {
        for (std::size_t i = 0; i < r.size(); ++i) r[i] = a[i + L];
    } else {
        for (std::size_t i = 0; i < r.size(); ++i) {
            const uint32_t hi = (i + L + 1 < a.size()) ? a[i + L + 1] : 0u;
            r[i] = (a[i + L] >> b) | (hi << (32 - b));
        }
    }
    return BigInt::from_magnitude(std::move(r), x.is_neg());
}

MpReal mp_from_int(const BigInt& v) { return MpReal{v, 0}; }
MpReal mp_from_i64(int64_t v) { return MpReal{BigInt(v), 0}; }

MpReal mp_from_double(double v) {
    NF_CHECK_MSG(std::isfinite(v), "mp_from_double: not finite");
    if (v == 0.0) return MpReal{};
    int e = 0;
    const double m = std::frexp(v, &e);  // v = m 2^e, 0.5 <= |m| < 1
    const int64_t mi = static_cast<int64_t>(std::ldexp(m, 53));
    return MpReal{BigInt(mi), static_cast<int64_t>(e) - 53};
}

MpReal mp_norm(MpReal x, unsigned prec) {
    if (x.man.is_zero()) return MpReal{};
    const std::size_t bl = x.man.bit_length();
    if (bl > prec) {
        const std::size_t s = bl - prec;
        x.man = bigint_shr(x.man, s);
        x.exp += static_cast<int64_t>(s);
    }
    return x;
}

MpReal mp_neg(MpReal x) {
    x.man = -x.man;
    return x;
}

MpReal mp_abs(MpReal x) {
    x.man = x.man.abs();
    return x;
}

MpReal mp_ldexp(MpReal x, int64_t k) {
    if (!x.man.is_zero()) x.exp += k;
    return x;
}

namespace {
// The exponent of the top bit + 1 (x != 0): |x| in [2^(top-1), 2^top).
int64_t top_of(const MpReal& x) { return static_cast<int64_t>(x.man.bit_length()) + x.exp; }

// Truncate x so that its exponent is >= cut (no-op if it already is).
MpReal cut_below(const MpReal& x, int64_t cut) {
    if (x.man.is_zero() || x.exp >= cut) return x;
    const int64_t s = cut - x.exp;
    if (s >= static_cast<int64_t>(x.man.bit_length())) return MpReal{};
    return MpReal{bigint_shr(x.man, static_cast<std::size_t>(s)), cut};
}
}  // namespace

MpReal mp_add(const MpReal& a0, const MpReal& b0, unsigned prec) {
    if (a0.man.is_zero()) return mp_norm(b0, prec);
    if (b0.man.is_zero()) return mp_norm(a0, prec);
    const int64_t top = std::max(top_of(a0), top_of(b0));
    const int64_t cut = top - static_cast<int64_t>(prec) - 4;
    const MpReal a = cut_below(a0, cut), b = cut_below(b0, cut);
    if (a.man.is_zero()) return mp_norm(b, prec);
    if (b.man.is_zero()) return mp_norm(a, prec);
    const int64_t e = std::min(a.exp, b.exp);
    MpReal r;
    r.man = bigint_shl(a.man, static_cast<std::size_t>(a.exp - e)) + bigint_shl(b.man, static_cast<std::size_t>(b.exp - e));
    r.exp = e;
    return mp_norm(std::move(r), prec);
}

MpReal mp_sub(const MpReal& a, const MpReal& b, unsigned prec) { return mp_add(a, mp_neg(b), prec); }

MpReal mp_mul(const MpReal& a, const MpReal& b, unsigned prec) {
    if (a.man.is_zero() || b.man.is_zero()) return MpReal{};
    const MpReal x = mp_norm(a, prec + 2), y = mp_norm(b, prec + 2);
    return mp_norm(MpReal{x.man * y.man, x.exp + y.exp}, prec);
}

MpReal mp_mul_i64(const MpReal& a, int64_t k, unsigned prec) {
    return mp_norm(MpReal{a.man * BigInt(k), a.exp}, prec);
}

MpReal mp_div_u32(const MpReal& a, uint32_t d, unsigned prec) {
    NF_CHECK(d != 0);
    if (a.man.is_zero()) return MpReal{};
    const int64_t bl = static_cast<int64_t>(a.man.bit_length());
    const int64_t s = std::max<int64_t>(0, static_cast<int64_t>(prec) + 34 - bl);
    MpReal r{bigint_shl(a.man, static_cast<std::size_t>(s)) / BigInt(static_cast<int64_t>(d)), a.exp - s};
    return mp_norm(std::move(r), prec);
}

MpReal mp_div(const MpReal& a, const MpReal& b0, unsigned prec) {
    NF_CHECK_MSG(!b0.man.is_zero(), "mp_div: division by zero");
    if (a.man.is_zero()) return MpReal{};
    const MpReal b = mp_norm(b0, prec + 32);
    const int64_t s = std::max<int64_t>(0, static_cast<int64_t>(prec) + 32 + static_cast<int64_t>(b.man.bit_length()) -
                                               static_cast<int64_t>(a.man.bit_length()));
    MpReal r{bigint_shl(a.man, static_cast<std::size_t>(s)) / b.man, a.exp - s - b.exp};
    return mp_norm(std::move(r), prec);
}

int mp_cmp(const MpReal& a, const MpReal& b) {
    const int sa = a.man.sign(), sb = b.man.sign();
    if (sa != sb) return sa < sb ? -1 : 1;
    if (sa == 0) return 0;
    const int64_t ta = top_of(a), tb = top_of(b);
    if (ta != tb) return (ta < tb) == (sa > 0) ? -1 : 1;
    const int64_t e = std::min(a.exp, b.exp);
    return compare(bigint_shl(a.man, static_cast<std::size_t>(a.exp - e)), bigint_shl(b.man, static_cast<std::size_t>(b.exp - e)));
}

namespace {
// |x| = t * 2^k with t < 2^53 the top bits (as a double), x != 0.
void top53(const MpReal& x, double& t, int64_t& k) {
    const std::size_t bl = x.man.bit_length();
    const std::size_t s = bl > 53 ? bl - 53 : 0;
    const BigInt top = bigint_shr(x.man.abs(), s);
    t = static_cast<double>(top.to_int64());
    k = static_cast<int64_t>(s) + x.exp;
}
}  // namespace

double mp_log2_abs(const MpReal& x) {
    if (x.man.is_zero()) return -std::numeric_limits<double>::infinity();
    double t;
    int64_t k;
    top53(x, t, k);
    return std::log2(t) + static_cast<double>(k);
}

double mp_to_double(const MpReal& x) {
    if (x.man.is_zero()) return 0.0;
    double t;
    int64_t k;
    top53(x, t, k);
    if (k > 4000) return x.man.is_neg() ? -HUGE_VAL : HUGE_VAL;
    if (k < -4000) return 0.0;
    const double v = std::ldexp(t, static_cast<int>(k));
    return x.man.is_neg() ? -v : v;
}

BigInt mp_round(const MpReal& x, double* log2_dist) {
    if (log2_dist) *log2_dist = -std::numeric_limits<double>::infinity();
    if (x.man.is_zero()) return BigInt(0);
    if (x.exp >= 0) return bigint_shl(x.man, static_cast<std::size_t>(x.exp));
    const std::size_t k = static_cast<std::size_t>(-x.exp);
    const BigInt a = x.man.abs();
    BigInt q = bigint_shr(a, k);
    BigInt r = a - bigint_shl(q, k);  // 0 <= r < 2^k
    const BigInt half = bigint_shl(BigInt(1), k - 1);
    BigInt dist = r;
    if (r >= half) {
        q += BigInt(1);
        dist = bigint_shl(BigInt(1), k) - r;
    }
    if (log2_dist && !dist.is_zero()) *log2_dist = mp_log2_abs(MpReal{dist, -static_cast<int64_t>(k)});
    return x.man.is_neg() ? -q : q;
}

namespace {
// atan(1/n) * 2^w (fixed point, truncated terms), n >= 2.
BigInt atan_inv_fixed(uint32_t n, unsigned w) {
    BigInt X = bigint_shl(BigInt(1), w) / BigInt(static_cast<int64_t>(n));
    BigInt sum = X;
    const BigInt n2(static_cast<int64_t>(n) * n);
    for (uint64_t k = 1; !X.is_zero(); ++k) {
        X = X / n2;
        const BigInt term = X / BigInt(static_cast<int64_t>(2 * k + 1));
        if (k % 2 == 1) sum -= term;
        else sum += term;
    }
    return sum;
}
}  // namespace

MpReal mp_pi(unsigned prec) {
    const unsigned w = prec + 32;
    const BigInt v = BigInt(16) * atan_inv_fixed(5, w) - BigInt(4) * atan_inv_fixed(239, w);
    return mp_norm(MpReal{v, -static_cast<int64_t>(w)}, prec);
}

MpReal mp_ln2(unsigned prec) {
    // log 2 = 2 atanh(1/3) = 2 sum_k 3^{-(2k+1)} / (2k+1).
    const unsigned w = prec + 32;
    BigInt X = bigint_shl(BigInt(1), w) / BigInt(3);
    BigInt sum = X;
    for (uint64_t k = 1; !X.is_zero(); ++k) {
        X = X / BigInt(9);
        sum += X / BigInt(static_cast<int64_t>(2 * k + 1));
    }
    return mp_norm(MpReal{sum * BigInt(2), -static_cast<int64_t>(w)}, prec);
}

MpContext::MpContext(unsigned prec) : prec_(prec), pi_(mp_pi(prec + 64)), ln2_(mp_ln2(prec + 64)) {}

MpReal MpContext::exp_at(const MpReal& x, unsigned p) const {
    NF_CHECK(p <= prec_);
    if (x.man.is_zero()) return mp_from_i64(1);
    const double xd = mp_to_double(x);
    NF_CHECK_MSG(std::fabs(xd) < 1099511627776.0, "MpContext::exp: |x| >= 2^40");
    const int64_t k = static_cast<int64_t>(std::llround(xd / 0.69314718055994530942));
    const unsigned t = std::max(4u, static_cast<unsigned>(std::sqrt(static_cast<double>(p))));
    const unsigned w = p + t + 48;
    const MpReal r = mp_sub(x, mp_mul_i64(ln2_, k, w + 48), w);  // |r| <= ~0.35
    const MpReal y = mp_ldexp(r, -static_cast<int64_t>(t));
    MpReal s = mp_from_i64(1), term = mp_from_i64(1);
    for (uint32_t j = 1;; ++j) {
        term = mp_div_u32(mp_mul(term, y, w), j, w);
        if (term.man.is_zero() || mp_log2_abs(term) < -static_cast<double>(w) - 2.0) break;
        s = mp_add(s, term, w);
    }
    for (unsigned j = 0; j < t; ++j) s = mp_mul(s, s, w);
    return mp_norm(mp_ldexp(s, k), p);
}

MpReal MpContext::exp(const MpReal& x) const { return exp_at(x, prec_); }

MpReal MpContext::log(const MpReal& x) const {
    NF_CHECK_MSG(mp_sign(x) > 0, "MpContext::log: x <= 0");
    const MpReal one = mp_from_i64(1);
    MpReal y = mp_from_double(mp_log2_abs(x) * 0.69314718055994530942);
    std::vector<unsigned> sched;
    for (unsigned q = prec_ + 32; q > 48; q = q / 2 + 16) sched.push_back(std::min(q, prec_));
    std::reverse(sched.begin(), sched.end());
    sched.push_back(prec_);  // one more step at the full precision
    for (unsigned q : sched) {
        const MpReal e = exp_at(mp_neg(y), q);
        y = mp_add(y, mp_sub(mp_mul(x, e, q), one, q), q + 32);
    }
    return mp_norm(y, prec_);
}

MpComplex MpContext::expi(const MpReal& t0) const {
    const unsigned p = prec_;
    const double td = mp_to_double(t0);
    NF_CHECK_MSG(std::fabs(td) < 1099511627776.0, "MpContext::expi: |t| >= 2^40");
    const int64_t k = static_cast<int64_t>(std::llround(td / 6.28318530717958647693));
    const unsigned s = std::max(4u, static_cast<unsigned>(std::sqrt(static_cast<double>(p)) / 2));
    const unsigned w = p + s + 48;
    const MpReal t = mp_sub(t0, mp_mul_i64(mp_ldexp(pi_, 1), k, w + 48), w);  // |t| <= ~pi
    const MpReal u = mp_ldexp(t, -static_cast<int64_t>(s));
    const MpReal u2 = mp_mul(u, u, w);
    // cos u = sum (-1)^j u^{2j}/(2j)!, sin u = sum (-1)^j u^{2j+1}/(2j+1)!
    MpReal c = mp_from_i64(1), sn = u, tc = mp_from_i64(1), ts = u;
    for (uint32_t j = 1;; ++j) {
        tc = mp_neg(mp_div_u32(mp_mul(tc, u2, w), (2 * j - 1) * (2 * j), w));
        ts = mp_neg(mp_div_u32(mp_mul(ts, u2, w), (2 * j) * (2 * j + 1), w));
        const bool dc = tc.man.is_zero() || mp_log2_abs(tc) < -static_cast<double>(w) - 2.0;
        const bool ds = ts.man.is_zero() || mp_log2_abs(ts) < -static_cast<double>(w) - 2.0;
        if (!dc) c = mp_add(c, tc, w);
        if (!ds) sn = mp_add(sn, ts, w);
        if (dc && ds) break;
    }
    MpComplex z{c, sn};
    for (unsigned j = 0; j < s; ++j) {
        // (c + i s)^2 = (c^2 - s^2) + 2 c s i
        const MpReal re = mp_sub(mp_mul(z.re, z.re, w), mp_mul(z.im, z.im, w), w);
        const MpReal im = mp_ldexp(mp_mul(z.re, z.im, w), 1);
        z = MpComplex{re, im};
    }
    return MpComplex{mp_norm(z.re, p), mp_norm(z.im, p)};
}

MpComplex mp_cadd(const MpComplex& a, const MpComplex& b, unsigned prec) {
    return MpComplex{mp_add(a.re, b.re, prec), mp_add(a.im, b.im, prec)};
}
MpComplex mp_csub(const MpComplex& a, const MpComplex& b, unsigned prec) {
    return MpComplex{mp_sub(a.re, b.re, prec), mp_sub(a.im, b.im, prec)};
}
MpComplex mp_cmul(const MpComplex& a, const MpComplex& b, unsigned prec) {
    return MpComplex{mp_sub(mp_mul(a.re, b.re, prec + 8), mp_mul(a.im, b.im, prec + 8), prec),
                     mp_add(mp_mul(a.re, b.im, prec + 8), mp_mul(a.im, b.re, prec + 8), prec)};
}
MpComplex mp_cmul_real(const MpComplex& a, const MpReal& r, unsigned prec) {
    return MpComplex{mp_mul(a.re, r, prec), mp_mul(a.im, r, prec)};
}
MpReal mp_cnorm(const MpComplex& a, unsigned prec) {
    return mp_add(mp_mul(a.re, a.re, prec + 8), mp_mul(a.im, a.im, prec + 8), prec);
}

std::string mp_to_string(const MpReal& x, unsigned digits) {
    if (x.man.is_zero()) return "0";
    NF_CHECK(digits >= 1 && digits <= 4000);
    const unsigned prec = static_cast<unsigned>(digits * 3.33) + 64;
    const double l10 = mp_log2_abs(x) * 0.30102999566398119521;
    int64_t e10 = static_cast<int64_t>(std::floor(l10));
    // scaled = |x| * 10^(digits - 1 - e10), rounded; fix e10 if it has digits + 1 digits.
    for (int pass = 0; pass < 2; ++pass) {
        const int64_t sh = static_cast<int64_t>(digits) - 1 - e10;
        MpReal y = mp_abs(x);
        const BigInt p10 = BigInt::pow(BigInt(10), static_cast<uint64_t>(std::llabs(sh)));
        y = sh >= 0 ? mp_mul(y, mp_from_int(p10), prec) : mp_div(y, mp_from_int(p10), prec);
        const BigInt n = mp_round(y);
        std::string s = n.to_string();
        if (s.size() > digits && pass == 0) {
            ++e10;
            continue;
        }
        std::string out = x.man.is_neg() ? "-" : "";
        out += s.substr(0, 1);
        if (s.size() > 1) out += "." + s.substr(1);
        out += "e" + std::to_string(e10);
        return out;
    }
    return "?";
}

}  // namespace nf
