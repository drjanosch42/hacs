// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Ntt64.cpp -- see Ntt64.h (slot-19 WP-NF2).
#include "Ntt64.h"

#include <algorithm>
#include <mutex>

#include "NfCheck.h"
#include "Parallel.h"

namespace nf {

namespace {

constexpr uint64_t kQ0 = uint64_t(15) << 32;  // 15 * 2^32 divides p - 1
constexpr std::size_t kPrimeCount = 16;
constexpr std::size_t kColBlock = 16;

std::vector<uint64_t> small_prime_factors(uint64_t x) {
    std::vector<uint64_t> f;
    for (uint64_t d = 2; d * d <= x; ++d) {
        if (x % d == 0) {
            f.push_back(d);
            while (x % d == 0) x /= d;
        }
    }
    if (x > 1) f.push_back(x);
    return f;
}

void build_table(std::vector<uint64_t>& tw, std::size_t n, const Mont64& mt, uint64_t g, bool inv) {
    // tw[half + k] = w_{2 half}^k (Montgomery form), half = 1, 2, 4, ..., n / 2.
    tw.assign(std::max<std::size_t>(n, 2), 0);
    const uint64_t p = mt.p;
    for (std::size_t half = 1; half < n; half <<= 1) {
        uint64_t w = mt.pow(g, (p - 1) / (2 * half));
        if (inv) w = mt.pow(w, p - 2);
        const uint64_t wm = mt.to(w);
        uint64_t cur = mt.to(1);
        for (std::size_t k = 0; k < half; ++k) {
            tw[half + k] = cur;
            cur = mt.mul(cur, wm);
        }
    }
}

// The powers base^0 .. base^{count-1} in Montgomery form.
std::vector<uint64_t> powers(const Mont64& mt, uint64_t base_normal, std::size_t count) {
    std::vector<uint64_t> v(count);
    const uint64_t bm = mt.to(base_normal);
    uint64_t cur = mt.to(1);
    for (std::size_t i = 0; i < count; ++i) {
        v[i] = cur;
        cur = mt.mul(cur, bm);
    }
    return v;
}

inline void dif(uint64_t* a, std::size_t n, const uint64_t* tw, const Mont64& mt) {
    for (std::size_t half = n >> 1; half >= 1; half >>= 1) {
        const uint64_t* w = tw + half;
        for (std::size_t i = 0; i < n; i += 2 * half) {
            uint64_t* x = a + i;
            uint64_t* y = x + half;
            for (std::size_t k = 0; k < half; ++k) {
                const uint64_t u = x[k], v = y[k];
                x[k] = mt.add(u, v);
                y[k] = mt.mul(mt.sub(u, v), w[k]);
            }
        }
    }
}

inline void dit(uint64_t* a, std::size_t n, const uint64_t* tw, const Mont64& mt) {
    for (std::size_t half = 1; half < n; half <<= 1) {
        const uint64_t* w = tw + half;
        for (std::size_t i = 0; i < n; i += 2 * half) {
            uint64_t* x = a + i;
            uint64_t* y = x + half;
            for (std::size_t k = 0; k < half; ++k) {
                const uint64_t u = x[k], v = mt.mul(y[k], w[k]);
                x[k] = mt.add(u, v);
                y[k] = mt.sub(u, v);
            }
        }
    }
}

unsigned log2_exact(std::size_t x) {
    unsigned k = 0;
    while ((std::size_t(1) << k) < x) ++k;
    NF_CHECK((std::size_t(1) << k) == x);
    return k;
}

}  // namespace

const std::vector<uint64_t>& ntt64_primes() {
    static std::once_flag once;
    static std::vector<uint64_t> primes;
    std::call_once(once, [] {
        uint64_t c = ((uint64_t(1) << 62) - 2) / kQ0;
        while (primes.size() < kPrimeCount && c > 0) {
            const uint64_t p = c * kQ0 + 1;
            if (is_prime_u64(p)) primes.push_back(p);
            --c;
        }
    });
    return primes;
}

uint64_t ntt64_prime(std::size_t i) {
    const auto& ps = ntt64_primes();
    NF_CHECK_MSG(i < ps.size(), "ntt64_prime: index beyond the prime table");
    return ps[i];
}

uint64_t ntt64_primitive_root(uint64_t p) {
    const Mont64 mt(p);
    std::vector<uint64_t> qs = {2, 3, 5};
    for (uint64_t q : small_prime_factors((p - 1) / kQ0)) {
        if (std::find(qs.begin(), qs.end(), q) == qs.end()) qs.push_back(q);
    }
    for (uint64_t g = 2; g < 1000; ++g) {
        bool ok = true;
        for (uint64_t q : qs) {
            if (mt.pow(g, (p - 1) / q) == 1) {
                ok = false;
                break;
            }
        }
        if (ok) return g;
    }
    NF_CHECK_MSG(false, "ntt64_primitive_root: none below 1000");
    return 0;
}

std::size_t ntt64_length(std::size_t min_len) {
    std::size_t best = 0;
    for (std::size_t f : {1u, 3u, 5u}) {
        std::size_t L = f;
        while (L < min_len) L <<= 1;
        if (best == 0 || L < best) best = L;
    }
    NF_CHECK_MSG(best <= (std::size_t(5) << 32), "ntt64_length: above 5 * 2^32");
    return best;
}

Ntt64Plan::Ntt64Plan(uint64_t prime, std::size_t L, unsigned direct_log) : mt_(prime), L_(L) {
    NF_CHECK((prime - 1) % kQ0 == 0);
    f_ = (L % 5 == 0) ? 5 : (L % 3 == 0) ? 3 : 1;
    M_ = L / f_;
    logM_ = log2_exact(M_);
    NF_CHECK_MSG(logM_ <= 32, "Ntt64Plan: length above f * 2^32");
    const uint64_t g = ntt64_primitive_root(prime);
    const uint64_t p = prime;
    if (logM_ > direct_log) {
        R_ = std::size_t(1) << (logM_ / 2);
        C_ = M_ / R_;
    } else {
        R_ = 1;
        C_ = M_;
    }
    for (int inv = 0; inv < 2; ++inv) {
        build_table(twC_[inv], C_, mt_, g, inv != 0);
        if (R_ > 1) build_table(twR_[inv], R_, mt_, g, inv != 0);
    }
    if (R_ > 1) {
        const unsigned lr = log2_exact(R_);
        revR_.resize(R_);
        for (std::size_t i = 0; i < R_; ++i) {
            uint32_t r = 0;
            for (unsigned b = 0; b < lr; ++b)
                if ((i >> b) & 1u) r |= uint32_t(1) << (lr - 1 - b);
            revR_[i] = r;
        }
        hM_ = (logM_ + 1) / 2;
        const uint64_t wM = mt_.pow(g, (p - 1) / M_);
        const uint64_t wMi = mt_.pow(wM, p - 2);
        const uint64_t wMh = mt_.pow(wM, uint64_t(1) << hM_), wMhi = mt_.pow(wMi, uint64_t(1) << hM_);
        aM_[0] = powers(mt_, wM, std::size_t(1) << hM_);
        aM_[1] = powers(mt_, wMi, std::size_t(1) << hM_);
        bM_[0] = powers(mt_, wMh, (M_ >> hM_) + 1);
        bM_[1] = powers(mt_, wMhi, (M_ >> hM_) + 1);
    }
    if (f_ > 1) {
        const uint64_t wL = mt_.pow(g, (p - 1) / L_);
        const uint64_t wLi = mt_.pow(wL, p - 2);
        const std::size_t nlo = std::size_t(1) << hL_;
        aL_[0] = powers(mt_, wL, nlo);
        aL_[1] = powers(mt_, wLi, nlo);
        bL_[0] = powers(mt_, mt_.pow(wL, nlo), (L_ >> hL_) + 1);
        bL_[1] = powers(mt_, mt_.pow(wLi, nlo), (L_ >> hL_) + 1);
        const uint64_t wf = mt_.pow(g, (p - 1) / f_);
        wf_[0] = powers(mt_, wf, f_);
        wf_[1] = powers(mt_, mt_.pow(wf, p - 2), f_);
    }
}

inline uint64_t Ntt64Plan::omegaM(std::size_t e, bool inv) const {
    const int s = inv ? 1 : 0;
    return mt_.mul(aM_[s][e & ((std::size_t(1) << hM_) - 1)], bM_[s][e >> hM_]);
}

inline uint64_t Ntt64Plan::omegaL(std::size_t e, bool inv) const {
    const int s = inv ? 1 : 0;
    return mt_.mul(aL_[s][e & ((std::size_t(1) << hL_) - 1)], bL_[s][e >> hL_]);
}

// The power-of-two transforms of the nsub consecutive blocks x + s * M_ (s < nsub). Slot-27 BIGNTT M1 (audit B2):
// the blocks of all sub-transforms share one parallel pass per phase instead of one sequence of passes per
// sub-transform; every block and row is computed exactly as before, so the output is unchanged.
void Ntt64Plan::pow2_forward(uint64_t* x, unsigned nsub, unsigned threads) const {
    if (R_ == 1) {
        parallel_ranges(nsub, threads, [&](std::size_t s0, std::size_t s1, unsigned) {
            for (std::size_t s = s0; s < s1; ++s) dif(x + s * M_, M_, twC_[0].data(), mt_);
        });
        return;
    }
    const std::size_t nblk = (C_ + kColBlock - 1) / kColBlock;
    parallel_ranges(nsub * nblk, threads, [&](std::size_t b0, std::size_t b1, unsigned) {
        std::vector<uint64_t> sc(R_ * kColBlock);
        for (std::size_t sb = b0; sb < b1; ++sb) {
            uint64_t* xs = x + (sb / nblk) * M_;
            const std::size_t blk = sb % nblk;
            const std::size_t c0 = blk * kColBlock, B = std::min(kColBlock, C_ - c0);
            for (std::size_t r = 0; r < R_; ++r) {
                const uint64_t* row = xs + r * C_ + c0;
                for (std::size_t b = 0; b < B; ++b) sc[b * R_ + r] = row[b];
            }
            for (std::size_t b = 0; b < B; ++b) dif(sc.data() + b * R_, R_, twR_[0].data(), mt_);
            for (std::size_t r = 0; r < R_; ++r) {
                const std::size_t k1 = revR_[r];
                uint64_t* row = xs + r * C_ + c0;
                for (std::size_t b = 0; b < B; ++b) {
                    const std::size_t e = ((c0 + b) * k1) & (M_ - 1);
                    row[b] = mt_.mul(sc[b * R_ + r], omegaM(e, false));
                }
            }
        }
    });
    parallel_ranges(nsub * R_, threads, [&](std::size_t r0, std::size_t r1, unsigned) {
        for (std::size_t sr = r0; sr < r1; ++sr) dif(x + (sr / R_) * M_ + (sr % R_) * C_, C_, twC_[0].data(), mt_);
    });
}

void Ntt64Plan::pow2_inverse(uint64_t* x, unsigned nsub, unsigned threads) const {
    if (R_ == 1) {
        parallel_ranges(nsub, threads, [&](std::size_t s0, std::size_t s1, unsigned) {
            for (std::size_t s = s0; s < s1; ++s) dit(x + s * M_, M_, twC_[1].data(), mt_);
        });
        return;
    }
    parallel_ranges(nsub * R_, threads, [&](std::size_t r0, std::size_t r1, unsigned) {
        for (std::size_t sr = r0; sr < r1; ++sr) dit(x + (sr / R_) * M_ + (sr % R_) * C_, C_, twC_[1].data(), mt_);
    });
    const std::size_t nblk = (C_ + kColBlock - 1) / kColBlock;
    parallel_ranges(nsub * nblk, threads, [&](std::size_t b0, std::size_t b1, unsigned) {
        std::vector<uint64_t> sc(R_ * kColBlock);
        for (std::size_t sb = b0; sb < b1; ++sb) {
            uint64_t* xs = x + (sb / nblk) * M_;
            const std::size_t blk = sb % nblk;
            const std::size_t c0 = blk * kColBlock, B = std::min(kColBlock, C_ - c0);
            for (std::size_t r = 0; r < R_; ++r) {
                const std::size_t k1 = revR_[r];
                const uint64_t* row = xs + r * C_ + c0;
                for (std::size_t b = 0; b < B; ++b) {
                    const std::size_t e = ((c0 + b) * k1) & (M_ - 1);
                    sc[b * R_ + r] = mt_.mul(row[b], omegaM(e, true));
                }
            }
            for (std::size_t b = 0; b < B; ++b) dit(sc.data() + b * R_, R_, twR_[1].data(), mt_);
            for (std::size_t r = 0; r < R_; ++r) {
                uint64_t* row = xs + r * C_ + c0;
                for (std::size_t b = 0; b < B; ++b) row[b] = sc[b * R_ + r];
            }
        }
    });
}

void Ntt64Plan::forward(uint64_t* a, unsigned threads) const {
    if (f_ > 1) {
        parallel_ranges(M_, threads, [&](std::size_t n0, std::size_t n1, unsigned) {
            uint64_t v[5], y[5];
            for (std::size_t n2 = n0; n2 < n1; ++n2) {
                for (unsigned i = 0; i < f_; ++i) v[i] = a[i * M_ + n2];
                for (unsigned k = 0; k < f_; ++k) {
                    uint64_t s = v[0];
                    for (unsigned i = 1; i < f_; ++i) s = mt_.add(s, mt_.mul(v[i], wf_[0][(i * k) % f_]));
                    y[k] = s;
                }
                a[n2] = y[0];
                for (unsigned k = 1; k < f_; ++k) a[k * M_ + n2] = mt_.mul(y[k], omegaL(n2 * k, false));
            }
        });
    }
    pow2_forward(a, f_, threads);
}

void Ntt64Plan::inverse(uint64_t* a, unsigned threads) const {
    pow2_inverse(a, f_, threads);
    if (f_ > 1) {
        parallel_ranges(M_, threads, [&](std::size_t n0, std::size_t n1, unsigned) {
            uint64_t v[5], y[5];
            for (std::size_t n2 = n0; n2 < n1; ++n2) {
                y[0] = a[n2];
                for (unsigned k = 1; k < f_; ++k) y[k] = mt_.mul(a[k * M_ + n2], omegaL(n2 * k, true));
                for (unsigned i = 0; i < f_; ++i) {
                    uint64_t s = y[0];
                    for (unsigned k = 1; k < f_; ++k) s = mt_.add(s, mt_.mul(y[k], wf_[1][(i * k) % f_]));
                    v[i] = s;
                }
                for (unsigned i = 0; i < f_; ++i) a[i * M_ + n2] = v[i];
            }
        });
    }
}

}  // namespace nf
