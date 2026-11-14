// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// BigNtt.cpp -- see BigNtt.h (slot-27 BIGNTT M1).
#include "BigNtt.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <utility>

#include "NfCheck.h"
#include "Ntt64.h"
#include "Parallel.h"

namespace nf {
namespace bigntt {

namespace {

std::atomic<uint64_t> g_products{0};
std::atomic<uint64_t> g_barrett{0};

constexpr std::size_t kMaxLength = std::size_t(1) << 29;   // coefficient bound: L (2^16 - 1)^2 < 2^61 < p

// The scale constant of the pointwise product: mul(mul(x, y), S) = x y L^{-1}, so the inverse (L times) gives x y.
uint64_t scale_const(const Ntt64Plan& pl) {
    const Mont64& mt = pl.mont();
    const uint64_t Linv = mt.pow(static_cast<uint64_t>(pl.length() % mt.p), mt.p - 2);
    return mt.to(mt.to(Linv));
}

// 16-bit chunks of a, zero-padded to L.
void to_chunks(const Mag& a, std::vector<uint64_t>& buf, std::size_t L, unsigned threads) {
    buf.assign(L, 0);
    parallel_ranges(a.size(), threads, [&](std::size_t b, std::size_t e, unsigned) {
        for (std::size_t i = b; i < e; ++i) {
            buf[2 * i] = a[i] & 0xffffu;
            buf[2 * i + 1] = a[i] >> 16;
        }
    });
}

// Carry pass: the chunk convolution c[0 .. nc) (each < p < 2^62) -> limbs; at most keep_limbs low limbs
// (0 = all). One 64-bit word, no __int128: s = c + carry < 2^62 + 2^47.
Mag from_chunks(const std::vector<uint64_t>& c, std::size_t nc, std::size_t keep_limbs) {
    std::size_t nchunks = nc + 4;   // the product of na and nb limbs has <= 2 (na + nb) = nc + 1 chunks
    if (keep_limbs) nchunks = std::min(nchunks, 2 * keep_limbs);
    Mag out((nchunks + 1) / 2, 0);
    uint64_t carry = 0;
    for (std::size_t i = 0; i < nchunks; ++i) {
        const uint64_t s = (i < nc ? c[i] : 0) + carry;
        out[i >> 1] |= static_cast<uint32_t>(s & 0xffffu) << ((i & 1u) ? 16 : 0);
        carry = s >> 16;
    }
    NF_CHECK_MSG(keep_limbs || carry == 0, "bigntt: carry beyond the product length");
    while (!out.empty() && out.back() == 0) out.pop_back();
    return out;
}

void pointwise(std::vector<uint64_t>& x, const std::vector<uint64_t>& y, const Mont64& mt, uint64_t S, bool scale,
               unsigned threads) {
    parallel_ranges(x.size(), threads, [&](std::size_t b, std::size_t e, unsigned) {
        if (scale) {
            for (std::size_t i = b; i < e; ++i) x[i] = mt.mul(mt.mul(x[i], y[i]), S);
        } else {
            for (std::size_t i = b; i < e; ++i) x[i] = mt.mul(x[i], y[i]);
        }
    });
}

}  // namespace

unsigned effective_threads(std::size_t L, unsigned threads) {
    // Below 2^14 points a pool round trip costs more than the transform itself (measured: a 2.3-kbit step 0.044 ms on one
    // thread, 0.52 ms on four); above it at most one thread per 4 096 points. Pool round trip (M1b, design Sec. 13, a
    // loaded box): ~ 4 / 10 / 34 / 49 us of overhead per parallel pass at T = 2 / 4 / 8 / 16 with a 20-us body (M1:
    // 82 / 92 / 95 / 103 us with 15 pool workers, 2x that with 31), ~ 25 passes per Barrett step. A thread count
    // never changes a value.
    if (threads <= 1 || L < (std::size_t(1) << 14)) return 1;
    return static_cast<unsigned>(std::min<std::size_t>(threads, L >> 12));
}

uint64_t ntt_products() { return g_products.load(std::memory_order_relaxed); }
uint64_t barrett_steps() { return g_barrett.load(std::memory_order_relaxed); }

std::shared_ptr<const Ntt64Plan> plan_for(std::size_t L, unsigned threads) {
    static std::mutex mu;
    static std::map<std::pair<std::size_t, unsigned>, std::shared_ptr<const Ntt64Plan>> cache;
    NF_CHECK_MSG(L >= 1 && L <= kMaxLength, "bigntt: transform length above 2^29 (coefficient bound)");
    // Four-step above 2^13 points when threaded (more independent rows / column blocks for the pool: measured
    // 2.6 -> 1.6 ms at L = 3 * 2^15, T = 16), above 2^15 (the Ntt64Plan default) single-threaded. Every operand of one
    // product goes through the same plan, so the plan's output permutation never matters; the product is exact.
    const unsigned dlog = threads > 1 ? 13u : 15u;
    std::lock_guard<std::mutex> lk(mu);
    const auto key = std::make_pair(L, dlog);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto pl = std::make_shared<const Ntt64Plan>(ntt64_prime(0), L, dlog);
    cache.emplace(key, pl);
    return pl;
}

Mag mul_ntt(const Mag& a, const Mag& b, unsigned threads) {
    if (a.empty() || b.empty()) return Mag();
    if (&a == &b || a == b) return sqr_ntt(a, threads);
    const std::size_t nc = 2 * a.size() + 2 * b.size() - 1;
    const std::size_t L = ntt64_length(nc);
    threads = effective_threads(L, threads);
    const auto pl = plan_for(L, threads);
    std::vector<uint64_t> fa, fb;
    to_chunks(a, fa, L, threads);
    to_chunks(b, fb, L, threads);
    pl->forward(fa.data(), threads);
    pl->forward(fb.data(), threads);
    pointwise(fa, fb, pl->mont(), scale_const(*pl), true, threads);
    pl->inverse(fa.data(), threads);
    g_products.fetch_add(1, std::memory_order_relaxed);
    return from_chunks(fa, nc, 0);
}

Mag sqr_ntt(const Mag& a, unsigned threads) {
    if (a.empty()) return Mag();
    const std::size_t nc = 4 * a.size() - 1;
    const std::size_t L = ntt64_length(nc);
    threads = effective_threads(L, threads);
    const auto pl = plan_for(L, threads);
    std::vector<uint64_t> fa;
    to_chunks(a, fa, L, threads);
    pl->forward(fa.data(), threads);
    pointwise(fa, fa, pl->mont(), scale_const(*pl), true, threads);
    pl->inverse(fa.data(), threads);
    g_products.fetch_add(1, std::memory_order_relaxed);
    return from_chunks(fa, nc, 0);
}

// ---------------------------------------------------------------------------------------------------------
BarrettModulus::BarrettModulus(const BigInt& m, std::size_t min_limbs, unsigned threads)
    : m_(m), threads_(std::max(1u, threads)) {
    NF_CHECK_MSG(!m.is_neg() && m >= BigInt(2), "BarrettModulus: modulus must be >= 2");
    mmag_ = m.magnitude();
    k_ = mmag_.size();
    ntt_ = k_ >= std::max<std::size_t>(min_limbs, 2);
    if (!ntt_) return;
    // m = b^{k-1}: mu = b^{k+1} has k + 2 limbs (code audit m1); x mod m is the low k - 1 limbs of x
    pow_b_ = mmag_.back() == 1u && std::all_of(mmag_.begin(), mmag_.end() - 1, [](uint32_t w) { return w == 0; });
    if (pow_b_) {
        L_ = ntt64_length(4 * k_ - 1);   // the product x y only
        threads_ = effective_threads(L_, threads_);
        return;
    }
    // mu = floor(b^{2k} / m) < b^{k+1}
    Mag b2k(2 * k_ + 1, 0);
    b2k.back() = 1;
    BigInt q, r;
    BigInt::divmod(BigInt::from_magnitude(b2k, false), m_, q, r);
    mu_ = q.magnitude();
    NF_CHECK(mu_.size() <= k_ + 1);
    // one length for the three products: x y (<= 4k - 1 chunks), q1 mu (<= 4k + 3), q3 m (<= 4k + 1)
    L_ = ntt64_length(4 * k_ + 3);
    threads_ = effective_threads(L_, threads_);
    plan_ = plan_for(L_, threads_);
    const Mont64& mt = plan_->mont();
    const uint64_t S = scale_const(*plan_);
    for (int w = 0; w < 2; ++w) {
        std::vector<uint64_t>& h = w == 0 ? mu_hat_ : m_hat_;
        to_chunks(w == 0 ? mu_ : mmag_, h, L_, threads_);
        plan_->forward(h.data(), threads_);
        parallel_ranges(L_, threads_, [&](std::size_t b, std::size_t e, unsigned) {
            for (std::size_t i = b; i < e; ++i) h[i] = mt.mul(h[i], S);   // pre-scaled
        });
    }
}

Mag BarrettModulus::mul_cached(const Mag& a, const std::vector<uint64_t>& bhat, std::size_t keep_limbs) const {
    if (a.empty()) return Mag();
    NF_CHECK(a.size() <= k_ + 1);
    std::vector<uint64_t> fa;
    to_chunks(a, fa, L_, threads_);
    plan_->forward(fa.data(), threads_);
    pointwise(fa, bhat, plan_->mont(), 0, false, threads_);
    plan_->inverse(fa.data(), threads_);
    g_products.fetch_add(1, std::memory_order_relaxed);
    return from_chunks(fa, std::min(L_, 2 * a.size() + 2 * (k_ + 1) - 1), keep_limbs);
}

Mag BarrettModulus::reduce(const Mag& x) const {
    NF_CHECK_MSG(x.size() <= 2 * k_, "BarrettModulus::reduce: operand at or above b^{2k}");
    g_barrett.fetch_add(1, std::memory_order_relaxed);
    if (pow_b_) {
        Mag r(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(std::min(x.size(), k_ - 1)));
        while (!r.empty() && r.back() == 0) r.pop_back();
        return r;
    }
    // q1 = floor(x / b^{k-1}), q3 = floor(q1 mu / b^{k+1})
    Mag q3;
    if (x.size() > k_ - 1) {
        const Mag q1(x.begin() + static_cast<std::ptrdiff_t>(k_ - 1), x.end());
        const Mag q2 = mul_cached(q1, mu_hat_, 0);
        if (q2.size() > k_ + 1) q3.assign(q2.begin() + static_cast<std::ptrdiff_t>(k_ + 1), q2.end());
    }
    // r = (x - q3 m) mod b^{k+1}, as a (k+1)-limb two's-complement difference
    const std::size_t W = k_ + 1;
    Mag r(W, 0);
    const Mag r2 = q3.empty() ? Mag() : mul_cached(q3, m_hat_, W);
    uint64_t borrow = 0;
    for (std::size_t i = 0; i < W; ++i) {
        const uint64_t xi = i < x.size() ? x[i] : 0;
        const uint64_t yi = (i < r2.size() ? r2[i] : 0) + borrow;
        r[i] = static_cast<uint32_t>(xi - yi);
        borrow = xi < yi ? 1 : 0;
    }
    while (!r.empty() && r.back() == 0) r.pop_back();
    BigInt R = BigInt::from_magnitude(std::move(r), false);
    int subs = 0;
    while (R >= m_) {
        R -= m_;
        ++subs;
        NF_CHECK_MSG(subs <= 2, "BarrettModulus::reduce: more than two corrections (HAC 14.42 violated)");
    }
    return R.magnitude();
}

BigInt BarrettModulus::mulmod(const BigInt& x, const BigInt& y) const {
    NF_CHECK_MSG(!x.is_neg() && !y.is_neg(), "BarrettModulus::mulmod: negative operand");   // O(1), both routes (m2)
    if (!ntt_) return (x * y) % m_;
    NF_DEBUG_CHECK(x < m_ && y < m_);
    return BigInt::from_magnitude(reduce(mul_ntt(x.magnitude(), y.magnitude(), threads_)), false);
}

BigInt BarrettModulus::sqrmod(const BigInt& x) const {
    NF_CHECK_MSG(!x.is_neg(), "BarrettModulus::sqrmod: negative operand");   // O(1), both routes (m2)
    if (!ntt_) return (x * x) % m_;
    NF_DEBUG_CHECK(x < m_);
    return BigInt::from_magnitude(reduce(sqr_ntt(x.magnitude(), threads_)), false);
}

BigInt powmod(const BigInt& base, const BigInt& e, const BarrettModulus& M) {
    NF_CHECK(!e.is_neg());
    BigInt r(1);
    BigInt b = base.floor_mod(M.modulus());   // [0, m) on both routes (code audit m2)
    const Mag& em = e.magnitude();
    const std::size_t nb = e.bit_length();
    for (std::size_t i = 0; i < nb; ++i) {
        if ((em[i >> 5] >> (i & 31u)) & 1u) r = M.mulmod(r, b);
        if (i + 1 < nb) b = M.sqrmod(b);
    }
    return r;
}

}  // namespace bigntt
}  // namespace nf
