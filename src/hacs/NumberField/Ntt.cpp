// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Ntt.cpp -- see Ntt.h (slot-19 WP-NF).
#include "Ntt.h"

#include <algorithm>

#include "ModUtil.h"
#include "NfCheck.h"

namespace nf {

namespace {

constexpr uint32_t kP0 = 998244353u;  // 119 * 2^23 + 1
constexpr uint32_t kP1 = 469762049u;  // 7 * 2^26 + 1
constexpr uint32_t kP2 = 167772161u;  // 5 * 2^25 + 1
constexpr std::size_t kSchoolbookMax = 48;

template <uint32_t P>
inline uint32_t mm(uint64_t a, uint64_t b) {
    return static_cast<uint32_t>(a * b % P);  // P is a compile-time constant: % becomes a multiply
}

template <uint32_t P>
inline uint32_t pw(uint64_t a, uint64_t e) {
    uint64_t r = 1;
    a %= P;
    while (e) {
        if (e & 1u) r = r * a % P;
        e >>= 1;
        a = a * a % P;
    }
    return static_cast<uint32_t>(r);
}

// Twiddle tables, cached per prime and direction: tw[half + k] = w_{2 half}^k (k < half), so one
// table of size N serves every transform length <= N. thread_local: no shared mutable state.
template <uint32_t P>
const std::vector<uint32_t>& twiddles(std::size_t n, bool invert) {
    thread_local std::vector<uint32_t> tab[2];
    std::vector<uint32_t>& t = tab[invert ? 1 : 0];
    if (t.size() < n) {
        std::vector<uint32_t> nt(n, 0);
        for (std::size_t half = 1; half < n; half <<= 1) {
            uint32_t wl = pw<P>(3, (P - 1) / (2 * half));
            if (invert) wl = pw<P>(wl, P - 2);
            nt[half] = 1;
            for (std::size_t k = 1; k < half; ++k) nt[half + k] = mm<P>(nt[half + k - 1], wl);
        }
        t.swap(nt);
    }
    return t;
}

template <uint32_t P>
void ntt(std::vector<uint32_t>& a, bool invert) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const std::vector<uint32_t>& tw = twiddles<P>(n, invert);
    for (std::size_t half = 1; half < n; half <<= 1) {
        const uint32_t* w = tw.data() + half;
        for (std::size_t i = 0; i < n; i += 2 * half) {
            uint32_t* x = a.data() + i;
            uint32_t* y = x + half;
            for (std::size_t k = 0; k < half; ++k) {
                const uint32_t u = x[k];
                const uint32_t v = mm<P>(y[k], w[k]);
                const uint32_t s = u + v;
                x[k] = s >= P ? s - P : s;
                y[k] = u >= v ? u - v : u + P - v;
            }
        }
    }
    if (invert) {
        const uint32_t ninv = pw<P>(static_cast<uint64_t>(n % P), P - 2);
        for (auto& x : a) x = mm<P>(x, ninv);
    }
}

template <uint32_t P>
std::vector<uint32_t> conv_prime(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, std::size_t N,
                                 bool same) {
    std::vector<uint32_t> fa(N, 0), fb;
    for (std::size_t i = 0; i < a.size(); ++i) fa[i] = a[i] % P;
    ntt<P>(fa, false);
    if (same) {
        for (std::size_t i = 0; i < N; ++i) fa[i] = mm<P>(fa[i], fa[i]);
    } else {
        fb.assign(N, 0);
        for (std::size_t i = 0; i < b.size(); ++i) fb[i] = b[i] % P;
        ntt<P>(fb, false);
        for (std::size_t i = 0; i < N; ++i) fa[i] = mm<P>(fa[i], fb[i]);
    }
    ntt<P>(fa, true);
    return fa;
}

}  // namespace

std::vector<uint32_t> convolve_mod_schoolbook(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b,
                                              uint32_t M) {
    if (a.empty() || b.empty()) return {};
    std::vector<uint64_t> acc(a.size() + b.size() - 1, 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!a[i]) continue;
        const uint64_t ai = a[i];
        for (std::size_t j = 0; j < b.size(); ++j) {
            uint64_t& t = acc[i + j];
            t += ai * b[j] % M;
            if (t >= (1ull << 62)) t %= M;
        }
    }
    std::vector<uint32_t> out(acc.size());
    for (std::size_t k = 0; k < acc.size(); ++k) out[k] = static_cast<uint32_t>(acc[k] % M);
    return out;
}

int ntt_prime_count(std::size_t la, std::size_t lb, uint32_t M) {
    if (std::min(la, lb) <= kSchoolbookMax) return 0;
    const long double bound = static_cast<long double>(std::min(la, lb)) * (M - 1.0L) * (M - 1.0L) * 2.0L;
    const long double p01 = static_cast<long double>(kP0) * kP1;
    if (bound < p01) return 2;
    NF_CHECK_MSG(bound < p01 * kP2, "convolve_mod: coefficient bound exceeds the three NTT primes");
    return 3;
}

std::vector<uint32_t> convolve_mod(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, uint32_t M) {
    NF_CHECK(M >= 2);
    if (a.empty() || b.empty()) return {};
    const int np = ntt_prime_count(a.size(), b.size(), M);
    if (np == 0) return convolve_mod_schoolbook(a, b, M);
    const std::size_t L = a.size() + b.size() - 1;
    std::size_t N = 1;
    while (N < L) N <<= 1;
    NF_CHECK_MSG(N <= (std::size_t(1) << 23), "convolve_mod: transform length above 2^23");
    const bool same = (&a == &b);
    const std::vector<uint32_t> r0 = conv_prime<kP0>(a, b, N, same);
    const std::vector<uint32_t> r1 = conv_prime<kP1>(a, b, N, same);
    std::vector<uint32_t> r2;
    if (np == 3) r2 = conv_prime<kP2>(a, b, N, same);
    // Garner: x = r0 + P0 t1 + P0 P1 t2.
    const uint64_t inv_p0_mod_p1 = invmod32(kP0 % kP1, kP1);
    const uint64_t p0p1_mod_p2 = static_cast<uint64_t>(kP0 % kP2) * (kP1 % kP2) % kP2;
    const uint64_t inv_p0p1_mod_p2 = np == 3 ? invmod32(p0p1_mod_p2, kP2) : 0;
    const uint64_t p0_mod_M = kP0 % M;
    const uint64_t p0p1_mod_M = static_cast<uint64_t>(kP0 % M) * (kP1 % M) % M;
    std::vector<uint32_t> out(L);
    for (std::size_t k = 0; k < L; ++k) {
        const uint64_t x0 = r0[k];
        const uint64_t t1 = (static_cast<uint64_t>(r1[k]) + kP1 - x0 % kP1) % kP1 * inv_p0_mod_p1 % kP1;
        uint64_t res = (x0 % M + p0_mod_M * (t1 % M)) % M;
        if (np == 3) {
            // (x0 + P0 t1) mod P2
            const uint64_t y = (x0 % kP2 + static_cast<uint64_t>(kP0 % kP2) * (t1 % kP2)) % kP2;
            const uint64_t t2 = (static_cast<uint64_t>(r2[k]) + kP2 - y) % kP2 * inv_p0p1_mod_p2 % kP2;
            res = (res + p0p1_mod_M * (t2 % M)) % M;
        }
        out[k] = static_cast<uint32_t>(res);
    }
    return out;
}

}  // namespace nf
