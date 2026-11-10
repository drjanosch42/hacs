// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// CwLayer.cpp -- see CwLayer.h (slot-19 WP-NF2).
#include "CwLayer.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "ModUtil.h"
#include "NfCheck.h"
#include "Parallel.h"
#include "Sha256.h"

namespace nf {

namespace {

double now_secs() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t upow(uint64_t b, unsigned e) {
    uint64_t r = 1;
    for (unsigned i = 0; i < e; ++i) {
        NF_CHECK_MSG(r <= UINT64_MAX / b, "upow: overflow");
        r *= b;
    }
    return r;
}

// v_ell(j!) by Legendre: (j - s_ell(j)) / (ell - 1).
uint64_t vfact(uint64_t j, uint32_t ell) {
    uint64_t s = 0, x = j;
    while (x) {
        s += x % ell;
        x /= ell;
    }
    return (j - s) / (ell - 1);
}

// The unit part and valuation of t >= 1.
inline uint32_t unit_part(uint64_t t, uint32_t ell) {
    while (t % ell == 0) t /= ell;
    return static_cast<uint32_t>(t);
}

uint64_t bigint_mod_u64(const BigInt& x, uint64_t p) {
    const BigInt r = x.floor_mod(BigInt::from_u64(p));
    return static_cast<uint64_t>(r.to_int64());
}

// Low j digits of a ladic value as an integer (ell^j < 2^63).
uint64_t low_value(const LadicCtx& cx, const uint32_t* a, unsigned j) {
    uint64_t v = 0, pw = 1;
    for (unsigned s = 0; s < j; ++s) {
        v += cx.digit(a, s) * pw;
        pw *= cx.ell;
    }
    return v;
}

}  // namespace

// ================================================================== the ring
CwRing::CwRing(uint32_t ell, unsigned m, unsigned W, unsigned threads, unsigned direct_log)
    : ell_(ell), m_(m), threads_(std::max(1u, threads)), cx_(ell, W) {
    NF_CHECK_MSG(m >= 1, "CwRing: m >= 1");
    blk_ = upow(ell, m - 1);
    N_ = blk_ * ell;
    n_ = blk_ * (ell - 1);
    L_ = ntt64_length(2 * n_ - 1);
    const BigInt Q = cx_.modulus();
    const BigInt bound = BigInt(4) * BigInt::from_u64(n_) * Q * Q;
    BigInt prod(1);
    std::vector<uint64_t> ps;
    while (prod <= bound) {
        ps.push_back(ntt64_prime(ps.size()));
        prod *= BigInt::from_u64(ps.back());
    }
    const unsigned P = static_cast<unsigned>(ps.size());
    plans_.reserve(P);
    for (uint64_t p : ps) plans_.emplace_back(p, L_, direct_log);
    const BigInt OFF = BigInt(2) * BigInt::from_u64(n_) * Q * Q;  // == 0 mod ell^W
    betapow_m_.assign(P, std::vector<uint64_t>(cx_.K));
    off_mod_p_.resize(P);
    scale_m_.resize(P);
    garner_pk_m_.assign(P, std::vector<uint64_t>(P, 0));
    garner_inv_m_.resize(P);
    garner_C_.resize(P);
    BigInt pre(1);
    for (unsigned j = 0; j < P; ++j) {
        const Mont64& mt = plans_[j].mont();
        const uint64_t p = mt.p;
        uint64_t bp = 1;
        for (unsigned t = 0; t < cx_.K; ++t) {
            betapow_m_[j][t] = mt.to(bp);
            bp = mulmod_u64(bp, cx_.beta, p);
        }
        off_mod_p_[j] = bigint_mod_u64(OFF, p);
        const uint64_t Linv = mt.pow(L_ % p, p - 2);
        scale_m_[j] = mt.to(mt.to(Linv));
        for (unsigned k = 0; k < j; ++k) garner_pk_m_[j][k] = mt.to(ps[k] % p);
        const uint64_t pre_p = bigint_mod_u64(pre, p);
        garner_inv_m_[j] = mt.to(mt.pow(pre_p, p - 2));
        garner_C_[j] = cx_.from_bigint(pre);
        pre *= BigInt::from_u64(p);
    }
    tdigits_ = 1;
    {
        long double bpow = cx_.beta;
        while (bpow <= 4611686018427387904.0L) {  // 2^62 > every prime
            bpow *= cx_.beta;
            ++tdigits_;
        }
    }
}

uint64_t CwRing::planned_product_peak_bytes(uint32_t ell, unsigned m, unsigned W) {
    const LadicCtx cx(ell, W);
    const uint64_t blk = upow(ell, m - 1), n = blk * (ell - 1);
    const uint64_t L = ntt64_length(2 * n - 1);
    const unsigned P = planned_prime_count(ell, m, W);
    return 2 * n * cx.K * 4 + 2 * L * 8 + P * n * 8;
}

unsigned CwRing::planned_prime_count(uint32_t ell, unsigned m, unsigned W) {
    const LadicCtx cx(ell, W);
    const uint64_t n = upow(ell, m - 1) * (ell - 1);
    const BigInt Q = cx.modulus();
    const BigInt bound = BigInt(4) * BigInt::from_u64(n) * Q * Q;
    BigInt prod(1);
    unsigned P = 0;
    while (prod <= bound) prod *= BigInt::from_u64(ntt64_prime(P++));
    return P;
}

CwRing::Elt CwRing::one() const {
    Elt e = zero();
    cx_.from_u64(1, e.data());
    return e;
}

CwRing::Elt CwRing::from_i64(const std::vector<int64_t>& c) const {
    const unsigned K = cx_.K;
    std::vector<uint32_t> dense(N_ * K, 0);
    uint32_t t[kLadicMaxLimbs];
    for (std::size_t i = 0; i < c.size(); ++i) {
        cx_.from_i64(c[i], t);
        uint32_t* d = dense.data() + (i % N_) * K;
        cx_.add(d, t, d);
    }
    Elt out = zero();
    for (uint64_t i = 0; i < n_; ++i)
        cx_.sub(dense.data() + i * K, dense.data() + (n_ + i % blk_) * K, out.data() + i * K);
    return out;
}

void CwRing::fill_residues(const Elt& a, unsigned j, std::vector<uint64_t>& buf) {
    const Mont64& mt = plans_[j].mont();
    const unsigned K = cx_.K;
    const uint64_t* bp = betapow_m_[j].data();
    buf.resize(L_);
    uint64_t* out = buf.data();
    const uint64_t n = n_;
    parallel_ranges(L_, threads_, [&](std::size_t b, std::size_t e, unsigned) {
        for (std::size_t i = b; i < e; ++i) {
            if (i >= n) {
                out[i] = 0;
                continue;
            }
            const uint32_t* x = a.data() + i * K;
            uint64_t s = 0;
            for (unsigned t = 0; t < K; ++t)
                if (x[t]) s = mt.add(s, mt.mul(x[t], bp[t]));
            out[i] = s;
        }
    });
}

void CwRing::mul_inplace(Elt& a, const Elt& b) {
    const bool same = (&a == &b);
    if (same) {
        ++stats.squarings;
    } else {
        ++stats.products;
    }
    const unsigned P = prime_count(), K = cx_.K;
    const uint64_t n = n_, N = N_, blk = blk_;
    std::vector<std::vector<uint64_t>> F(P);
    std::vector<uint64_t> bufA, bufB;
    for (unsigned j = 0; j < P; ++j) {
        const Ntt64Plan& pl = plans_[j];
        const Mont64& mt = pl.mont();
        double t0 = now_secs();
        fill_residues(a, j, bufA);
        stats.io_secs += now_secs() - t0;
        t0 = now_secs();
        pl.forward(bufA.data(), threads_);
        ++stats.ntts;
        if (!same) {
            stats.ntt_secs += now_secs() - t0;
            t0 = now_secs();
            fill_residues(b, j, bufB);
            stats.io_secs += now_secs() - t0;
            t0 = now_secs();
            pl.forward(bufB.data(), threads_);
            ++stats.ntts;
            uint64_t* x = bufA.data();
            const uint64_t* y = bufB.data();
            parallel_ranges(L_, threads_, [&](std::size_t s, std::size_t e, unsigned) {
                for (std::size_t i = s; i < e; ++i) x[i] = mt.mul(x[i], y[i]);
            });
        } else {
            uint64_t* x = bufA.data();
            parallel_ranges(L_, threads_, [&](std::size_t s, std::size_t e, unsigned) {
                for (std::size_t i = s; i < e; ++i) x[i] = mt.mul(x[i], x[i]);
            });
        }
        pl.inverse(bufA.data(), threads_);
        ++stats.ntts;
        stats.ntt_secs += now_secs() - t0;
        t0 = now_secs();
        F[j].resize(n);
        uint64_t* f = F[j].data();
        const uint64_t* c = bufA.data();
        const uint64_t sc = scale_m_[j], off = off_mod_p_[j];
        const uint64_t last = 2 * n - 2;
        parallel_ranges(n, threads_, [&](std::size_t s, std::size_t e, unsigned) {
            for (std::size_t i = s; i < e; ++i) {
                uint64_t v = c[i];
                if (i + N <= last) v = mt.add(v, c[i + N]);
                v = mt.sub(v, c[n + i % blk]);
                f[i] = mt.add(mt.mul(v, sc), off);
            }
        });
        stats.io_secs += now_secs() - t0;
    }
    bufA.clear();
    bufA.shrink_to_fit();
    bufB.clear();
    bufB.shrink_to_fit();
    const double t0 = now_secs();
    // Garner: V = t_0 + p_0 t_1 + p_0 p_1 t_2 + ...  (exact, < prod p), then V mod ell^W in limbs.
    const unsigned D = tdigits_;
    const uint64_t beta = cx_.beta;
    const bool norm_each = static_cast<long double>(P) * D * (beta - 1.0L) * (beta - 1.0L) >= 18446744073709551615.0L;
    std::vector<const Mont64*> mts(P);
    for (unsigned j = 0; j < P; ++j) mts[j] = &plans_[j].mont();
    parallel_ranges(n, threads_, [&](std::size_t s, std::size_t e, unsigned) {
        uint64_t t[16];
        uint64_t acc[kLadicMaxLimbs];
        for (std::size_t i = s; i < e; ++i) {
            t[0] = F[0][i];
            for (unsigned j = 1; j < P; ++j) {
                const Mont64& mt = *mts[j];
                uint64_t h = t[j - 1] % mt.p;
                for (int k = static_cast<int>(j) - 2; k >= 0; --k) h = mt.add(mt.mul(h, garner_pk_m_[j][k]), t[k] % mt.p);
                t[j] = mt.mul(mt.sub(F[j][i], h), garner_inv_m_[j]);
            }
            for (unsigned q = 0; q < K; ++q) acc[q] = 0;
            for (unsigned j = 0; j < P; ++j) {
                uint64_t x = t[j];
                const uint32_t* C = garner_C_[j].data();
                for (unsigned d = 0; d < D && d < K; ++d) {
                    const uint64_t dig = x % beta;
                    x /= beta;
                    if (!dig) continue;
                    for (unsigned q = 0; q + d < K; ++q) acc[q + d] += dig * C[q];
                }
                if (norm_each) {
                    uint64_t carry = 0;
                    for (unsigned q = 0; q + 1 < K; ++q) {
                        const uint64_t v = acc[q] + carry;
                        acc[q] = v % beta;
                        carry = v / beta;
                    }
                    acc[K - 1] = (acc[K - 1] + carry) % cx_.topmod;
                }
            }
            cx_.normalize64(acc, a.data() + i * K);
        }
    });
    stats.garner_secs += now_secs() - t0;
}

void CwRing::pow_inplace(Elt& a, uint64_t e) {
    if (e == 0) {
        a = one();
        return;
    }
    if (e == 1) return;
    const Elt base = a;
    int top = 63;
    while (!((e >> top) & 1u)) --top;
    for (int bit = top - 1; bit >= 0; --bit) {
        sqr_inplace(a);
        if ((e >> bit) & 1u) mul_inplace(a, base);
    }
}

void CwRing::add_inplace(Elt& a, const Elt& b) const {
    const unsigned K = cx_.K;
    parallel_ranges(n_, threads_, [&](std::size_t s, std::size_t e, unsigned) {
        for (std::size_t i = s; i < e; ++i) cx_.add(a.data() + i * K, b.data() + i * K, a.data() + i * K);
    });
}

void CwRing::sub_inplace(Elt& a, const Elt& b) const {
    const unsigned K = cx_.K;
    parallel_ranges(n_, threads_, [&](std::size_t s, std::size_t e, unsigned) {
        for (std::size_t i = s; i < e; ++i) cx_.sub(a.data() + i * K, b.data() + i * K, a.data() + i * K);
    });
}

void CwRing::scal_inplace(Elt& a, const LVal& c) const {
    const unsigned K = cx_.K;
    parallel_ranges(n_, threads_, [&](std::size_t s, std::size_t e, unsigned) {
        for (std::size_t i = s; i < e; ++i) cx_.mul(a.data() + i * K, c.data(), a.data() + i * K);
    });
}

CwRing::Elt CwRing::sigma(const Elt& a, uint64_t c) const {
    NF_CHECK_MSG(c % ell_ != 0, "sigma: c must be prime to ell");
    const unsigned K = cx_.K;
    std::vector<uint32_t> dense(N_ * K, 0);
    for (uint64_t i = 0; i < n_; ++i) {
        const uint64_t pos = (i % N_) * (c % N_) % N_;
        std::copy(a.begin() + i * K, a.begin() + (i + 1) * K, dense.begin() + pos * K);
    }
    Elt out = zero();
    for (uint64_t i = 0; i < n_; ++i)
        cx_.sub(dense.data() + i * K, dense.data() + (n_ + i % blk_) * K, out.data() + i * K);
    return out;
}

uint32_t CwRing::residue(const Elt& a) const {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n_; ++i) s += a[i * cx_.K] % ell_;
    return static_cast<uint32_t>(s % ell_);
}

unsigned CwRing::content_val(const Elt& a) const {
    unsigned v = cx_.W;
    for (uint64_t i = 0; i < n_ && v > 0; ++i) v = std::min(v, cx_.valuation(a.data() + i * cx_.K));
    return v;
}

CwRing::Elt CwRing::schoolbook_mul(const Elt& a, const Elt& b) const {
    const unsigned K = cx_.K;
    std::vector<uint32_t> dense(N_ * K, 0);
    uint32_t t[kLadicMaxLimbs];
    for (uint64_t i = 0; i < n_; ++i) {
        if (cx_.is_zero(a.data() + i * K)) continue;
        for (uint64_t j = 0; j < n_; ++j) {
            cx_.mul(a.data() + i * K, b.data() + j * K, t);
            uint32_t* d = dense.data() + ((i + j) % N_) * K;
            cx_.add(d, t, d);
        }
    }
    Elt out = zero();
    for (uint64_t i = 0; i < n_; ++i)
        cx_.sub(dense.data() + i * K, dense.data() + (n_ + i % blk_) * K, out.data() + i * K);
    return out;
}

std::string CwRing::digits_sha16(const Elt& a) const {
    Sha256 h;
    const unsigned K = cx_.K;
    std::vector<uint8_t> buf;
    const std::size_t chunk = 1 << 16;
    buf.reserve(chunk * 8);
    for (unsigned s = 0; s < cx_.W; ++s) {
        for (uint64_t i0 = 0; i0 < n_; i0 += chunk) {
            const uint64_t i1 = std::min<uint64_t>(n_, i0 + chunk);
            buf.assign((i1 - i0) * 8, 0);
            for (uint64_t i = i0; i < i1; ++i) buf[(i - i0) * 8] = static_cast<uint8_t>(0);
            for (uint64_t i = i0; i < i1; ++i) {
                uint64_t dg = cx_.digit(a.data() + i * K, s);
                for (int q = 0; q < 8; ++q) buf[(i - i0) * 8 + q] = static_cast<uint8_t>(dg >> (8 * q));
            }
            h.update(buf.data(), buf.size());
        }
    }
    return h.hex().substr(0, 16);
}

// ================================================================== pi^n / ell
CwRing::Elt cw_pi_n_over_ell(CwRing& R, bool binomial) {
    const LadicCtx& cx = R.ctx();
    const uint32_t ell = R.ell();
    const uint64_t n = R.n(), blk = R.blk();
    const unsigned K = cx.K, W = cx.W;
    CwRing::Elt out = R.zero();
    if (!binomial) {
        CwRing Rp(ell, R.m(), W + 1, R.threads());
        CwRing::Elt pi = Rp.from_i64({1, -1});
        Rp.pow_inplace(pi, n);
        const LadicCtx& c1 = Rp.ctx();
        for (uint64_t i = 0; i < n; ++i) {
            const uint32_t* x = pi.data() + i * c1.K;
            NF_CHECK_MSG(c1.digit(x, 0) == 0, "pi^n not divisible by ell");
            uint32_t t[kLadicMaxLimbs];
            c1.div_lpow(x, 1, t);
            for (unsigned q = K; q < c1.K; ++q) NF_CHECK(t[q] == 0);
            std::copy(t, t + K, out.data() + i * K);
        }
        R.stats.squarings += Rp.stats.squarings;
        R.stats.products += Rp.stats.products;
        R.stats.ntts += Rp.stats.ntts;
        return out;
    }
    const LadicCtx c1(ell, W + 1);
    const unsigned K1 = c1.K;
    NF_CHECK(c1.beta == cx.beta);
    const unsigned T = R.threads();
    // P[t] = prod_{s <= t} u(s) mod ell^{W+1}, t = 0..n; then overwritten by its inverse.
    std::vector<uint32_t> Pv((n + 1) * K1, 0);
    c1.from_u64(1, Pv.data());
    const std::size_t nch = std::max<std::size_t>(1, std::min<std::size_t>(4 * T, n));
    std::vector<uint64_t> lo(nch), hi(nch);
    for (std::size_t c = 0; c < nch; ++c) {
        lo[c] = 1 + n * c / nch;
        hi[c] = n * (c + 1) / nch;
    }
    std::vector<LVal> G(nch), E(nch);
    parallel_ranges(nch, T, [&](std::size_t b, std::size_t e, unsigned) {
        for (std::size_t c = b; c < e; ++c) {
            LVal g{};
            c1.from_u64(1, g.data());
            for (uint64_t t = lo[c]; t <= hi[c]; ++t) c1.mul_u32(g.data(), unit_part(t, ell), g.data());
            G[c] = g;
        }
    });
    c1.from_u64(1, E[0].data());
    for (std::size_t c = 1; c < nch; ++c) c1.mul(E[c - 1].data(), G[c - 1].data(), E[c].data());
    parallel_ranges(nch, T, [&](std::size_t b, std::size_t e, unsigned) {
        for (std::size_t c = b; c < e; ++c) {
            LVal run = E[c];
            for (uint64_t t = lo[c]; t <= hi[c]; ++t) {
                c1.mul_u32(run.data(), unit_part(t, ell), run.data());
                std::copy(run.begin(), run.begin() + K1, Pv.begin() + t * K1);
            }
        }
    });
    LVal Pn{};
    std::copy(Pv.begin() + n * K1, Pv.begin() + (n + 1) * K1, Pn.begin());
    parallel_ranges(nch, T, [&](std::size_t b, std::size_t e, unsigned) {
        for (std::size_t c = b; c < e; ++c) {
            if (hi[c] < lo[c]) continue;
            const LVal inv = c1.inverse(Pv.data() + hi[c] * K1);
            std::copy(inv.begin(), inv.begin() + K1, Pv.begin() + hi[c] * K1);
            for (uint64_t t = hi[c]; t > lo[c]; --t)
                c1.mul_u32(Pv.data() + t * K1, unit_part(t, ell), Pv.data() + (t - 1) * K1);
        }
    });
    const uint64_t vn = vfact(n, ell);
    parallel_ranges(n, T, [&](std::size_t b, std::size_t e, unsigned) {
        uint32_t x[kLadicMaxLimbs], y[kLadicMaxLimbs];
        for (std::size_t i = b; i < e; ++i) {
            const uint64_t ev = vn - vfact(i, ell) - vfact(n - i, ell);
            for (unsigned q = 0; q < K1; ++q) x[q] = 0;
            if (ev <= W) {
                c1.mul(Pn.data(), Pv.data() + i * K1, x);
                c1.mul(x, Pv.data() + (n - i) * K1, x);
                if (ev) c1.mul_lpow(x, static_cast<unsigned>(ev), x);
                if (i & 1u) c1.neg(x, x);
            }
            // pi^n has the coefficient +1 at z^n (n even); z^n = -sum_{j < ell - 1} z^{j blk}.
            if (i % blk == 0 && i / blk < ell - 1) {
                c1.from_u64(1, y);
                c1.sub(x, y, x);
            }
            NF_CHECK_MSG(c1.digit(x, 0) == 0, "pi^n not divisible by ell");
            c1.div_lpow(x, 1, y);
            for (unsigned q = K; q < K1; ++q) NF_CHECK(y[q] == 0);
            std::copy(y, y + K, out.data() + i * K);
        }
    });
    return out;
}

// ================================================================== the log of a unit
CwRing::Elt cw_log_unit(CwRing& R, CwRing::Elt v, unsigned r, unsigned* cy_out, unsigned* terms_out, double* secs_pow,
                        double* secs_series) {
    const LadicCtx& cx = R.ctx();
    const uint32_t ell = R.ell();
    const unsigned W = cx.W;
    double t0 = now_secs();
    const uint32_t e0 = R.residue(v);
    NF_CHECK_MSG(e0 != 0, "cw_log_unit: not a unit");
    LVal e0v{};
    cx.from_u64(e0, e0v.data());
    const LVal teich = cx.pow(e0v.data(), BigInt::pow(BigInt(static_cast<int64_t>(ell)), W - 1));
    const LVal tinv = cx.inverse(teich.data());
    R.scal_inplace(v, tinv);  // principal unit w (same log)
    NF_CHECK_MSG(R.residue(v) == 1, "w is not a principal unit");
    for (unsigned i = 0; i < r; ++i) R.pow_inplace(v, ell);
    R.sub_inplace(v, R.one());  // y = w^{ell^r} - 1
    if (secs_pow) *secs_pow = now_secs() - t0;
    t0 = now_secs();
    const unsigned cy = R.content_val(v);
    NF_CHECK_MSG(cy >= 1, "y not divisible by ell: raise r");
    const unsigned terms = (W + cy - 1) / cy;
    NF_CHECK_MSG(terms < ell, "series needs j >= ell: raise r");
    auto coef = [&](unsigned j) {
        LVal jv{};
        cx.from_u64(j, jv.data());
        LVal c = cx.inverse(jv.data());  // exact inverse mod ell^W (j < ell)
        if (j % 2 == 0) cx.neg(c.data(), c.data());
        return c;
    };
    // log(1 + y) = sum_{j <= terms} (-1)^{j+1} y^j / j by Horner (two live elements).
    CwRing::Elt H = v;
    R.scal_inplace(H, coef(terms));
    for (unsigned j = terms - 1; j >= 1; --j) {
        const LVal c = coef(j);
        cx.add(H.data(), c.data(), H.data());
        R.mul_inplace(H, v);
    }
    if (cy_out) *cy_out = cy;
    if (terms_out) *terms_out = terms;
    if (secs_series) *secs_series = now_secs() - t0;
    return H;
}

// ================================================================== lambda_m
CwLog cw_lambda(CwRing& R, const CwLogOptions& opt) {
    const LadicCtx& cx = R.ctx();
    const uint32_t ell = R.ell();
    const unsigned m = R.m();
    CwLog L;
    L.r = opt.r ? opt.r : m + 1;
    double t0 = now_secs();
    CwRing::Elt v = cw_pi_n_over_ell(R, opt.binomial_pi);
    if (opt.perturb == 1) {
        R.mul_inplace(v, R.from_i64({1, 1}));  // times the unit u_2 = 1 + z
    } else if (opt.perturb == 2) {
        LVal one{};
        cx.from_u64(1, one.data());
        uint32_t* c1 = v.data() + 1 * cx.K;
        cx.add(c1, one.data(), c1);  // one digit changed (still a unit unless the residue hits 0)
        if (R.residue(v) == 0) cx.add(c1, one.data(), c1);
    }
    L.secs_pi = now_secs() - t0;
    CwRing::Elt H = cw_log_unit(R, std::move(v), L.r, &L.cy, &L.terms, &L.secs_pow, &L.secs_series);
    t0 = now_secs();
    LVal lm1{};
    cx.from_u64(ell - 1, lm1.data());
    R.scal_inplace(H, cx.inverse(lm1.data()));
    L.Lg = std::move(H);
    L.s = L.r + m - 1;
    L.secs_series += now_secs() - t0;
    return L;
}

// ================================================================== the measure
std::vector<CwMeasure> cw_measure(const LadicCtx& cx, unsigned level, const uint32_t* y, unsigned scale,
                                  const std::vector<uint32_t>& classes, unsigned Mout, unsigned threads) {
    const uint32_t ell = cx.ell;
    const unsigned j = level, K = cx.K;
    NF_CHECK_MSG(j >= 1, "cw_measure: level >= 1");
    NF_CHECK_MSG(cx.W >= scale + Mout, "cw_measure: precision too small: nu would be known mod less than ell^Mout");
    const uint64_t Nj = upow(ell, j), bj = upow(ell, j - 1);
    // tables t_i (i < j), size ell^i, and ell^{j-1} S_j (size ell^{j-1})
    std::vector<std::vector<uint32_t>> tab(j);
    auto x_at = [&](unsigned i, uint64_t u, uint32_t* out) {  // ell^{j-i} y[u ell^{j-i}]
        const uint64_t st = upow(ell, j - i);
        cx.mul_lpow(y + u * st * K, j - i, out);
    };
    for (unsigned i = 1; i < j; ++i) {
        const uint64_t Ni = upow(ell, i), ni = upow(ell, i - 1) * (ell - 1), bi = upow(ell, i - 1);
        std::vector<uint32_t> S(bi * K, 0);
        uint32_t t[kLadicMaxLimbs];
        for (uint64_t u = 0; u < ni; ++u) {
            x_at(i, u, t);
            cx.add(S.data() + (u % bi) * K, t, S.data() + (u % bi) * K);
        }
        tab[i].assign(Ni * K, 0);
        for (uint64_t c = 0; c < Ni; ++c) {
            uint32_t a[kLadicMaxLimbs], b[kLadicMaxLimbs];
            for (unsigned q = 0; q < K; ++q) a[q] = 0;
            if (c < ni) {
                x_at(i, c, a);
                cx.mul_lpow(a, i, a);
            }
            cx.mul_lpow(S.data() + (c % bi) * K, i - 1, b);
            cx.sub(a, b, tab[i].data() + c * K);
        }
    }
    std::vector<uint32_t> Sj(bj * K, 0);
    {
        const uint64_t nj = bj * (ell - 1);
        parallel_ranges(bj, threads, [&](std::size_t s, std::size_t e, unsigned) {
            for (std::size_t c = s; c < e; ++c) {
                uint32_t* d = Sj.data() + c * K;
                for (uint64_t u = c; u < nj; u += bj) cx.add(d, y + u * K, d);
                cx.mul_lpow(d, j - 1, d);
            }
        });
    }
    const uint64_t nj = bj * (ell - 1);
    auto N_at = [&](uint64_t a, uint32_t* out) {
        uint32_t t[kLadicMaxLimbs];
        for (unsigned q = 0; q < K; ++q) out[q] = 0;
        if (a < nj) cx.mul_lpow(y + a * K, j, out);
        cx.sub(out, Sj.data() + (a % bj) * K, out);
        uint64_t Ni = 1;
        for (unsigned i = 1; i < j; ++i) {
            Ni *= ell;
            const uint32_t* ti = tab[i].data() + (a % Ni) * K;
            for (unsigned q = 0; q < K; ++q) t[q] = ti[q];
            cx.add(out, t, out);
        }
    };
    // Teichmueller: tg = g^{ell^{W-1}} mod ell^W; delta_q = tg^q mod ell^j; omega^w(delta_q) = tg^{q w mod (ell-1)}.
    const uint32_t g = primitive_root_prime(ell);
    LVal gv{};
    cx.from_u64(g, gv.data());
    const LVal tg = cx.pow(gv.data(), BigInt::pow(BigInt(static_cast<int64_t>(ell)), cx.W - 1));
    const uint64_t tgj = low_value(cx, tg.data(), j);
    std::vector<uint64_t> deltas(ell - 1);
    std::vector<LVal> tgp(ell - 1);
    {
        uint64_t d = 1 % Nj;
        LVal cur{};
        cx.from_u64(1, cur.data());
        for (uint32_t q = 0; q + 1 < ell; ++q) {
            deltas[q] = d;
            tgp[q] = cur;
            d = static_cast<uint64_t>(mulmod_u64(d, tgj, Nj));
            cx.mul(cur.data(), tg.data(), cur.data());
        }
    }
    LVal inv{};
    {
        LVal lm1{};
        cx.from_u64(ell - 1, lm1.data());
        inv = cx.inverse(lm1.data());
    }
    const std::size_t C = classes.size();
    std::vector<CwMeasure> res(C);
    for (std::size_t ci = 0; ci < C; ++ci) {
        res[ci].w = classes[ci];
        res[ci].level = j;
        res[ci].nu.assign(bj, 0);
    }
    std::vector<uint8_t> bad(C, 0);
    const uint64_t Mmod = upow(ell, Mout);
    std::vector<std::vector<uint8_t>> badt(std::max(1u, threads), std::vector<uint8_t>(C, 0));
    parallel_ranges(bj, threads, [&](std::size_t s, std::size_t e, unsigned wk) {
        std::vector<uint32_t> Nv((ell - 1) * K);
        uint64_t a = static_cast<uint64_t>(powmod_u64(1 + ell, s, Nj));
        uint32_t acc[kLadicMaxLimbs], t[kLadicMaxLimbs];
        for (std::size_t i = s; i < e; ++i) {
            for (uint32_t q = 0; q + 1 < ell; ++q) N_at(mulmod_u64(deltas[q], a, Nj), Nv.data() + q * K);
            for (std::size_t ci = 0; ci < C; ++ci) {
                const uint32_t w = classes[ci] % (ell - 1);
                for (unsigned q = 0; q < K; ++q) acc[q] = 0;
                for (uint32_t q = 0; q + 1 < ell; ++q) {
                    cx.mul(tgp[(static_cast<uint64_t>(q) * w) % (ell - 1)].data(), Nv.data() + q * K, t);
                    cx.add(acc, t, acc);
                }
                cx.mul(acc, inv.data(), acc);
                if (!cx.is_zero(acc) && cx.valuation(acc) < scale) badt[wk][ci] = 1;
                cx.div_lpow(acc, scale, t);
                res[ci].nu[i] = static_cast<uint32_t>(low_value(cx, t, Mout) % Mmod);
            }
            a = mulmod_u64(a, 1 + ell, Nj);
        }
    });
    for (auto& bt : badt)
        for (std::size_t ci = 0; ci < C; ++ci) bad[ci] |= bt[ci];
    for (std::size_t ci = 0; ci < C; ++ci) res[ci].integral = !bad[ci];
    return res;
}

std::vector<uint32_t> cw_moments(const std::vector<uint32_t>& nu, uint32_t ell, unsigned M, uint32_t w, uint64_t rows,
                                 unsigned threads) {
    const uint64_t mod = upow(ell, M);
    NF_CHECK_MSG(mod < (uint64_t(1) << 31), "cw_moments: ell^M must be < 2^31");
    const Mont32 mt(static_cast<uint32_t>(mod));
    std::vector<uint32_t> num(nu.size());
    for (std::size_t i = 0; i < nu.size(); ++i) num[i] = mt.to(nu[i]);
    const uint64_t h = powmod_u64(1 + ell, ell - 1, mod);
    std::vector<uint32_t> out(rows);
    const uint32_t lm1 = static_cast<uint32_t>((ell - 1) % mod);
    const uint32_t M32 = static_cast<uint32_t>(mod);
    parallel_ranges((rows + 7) / 8, threads, [&](std::size_t s, std::size_t e, unsigned) {
        for (std::size_t blk = s; blk < e; ++blk) {
            const uint64_t t0 = blk * 8;
            const unsigned cnt = static_cast<unsigned>(std::min<uint64_t>(8, rows - t0));
            uint32_t X[8], acc[8];
            uint64_t x = powmod_u64(1 + ell, w + static_cast<uint64_t>(ell - 1) * t0, mod);
            for (unsigned r = 0; r < 8; ++r) {
                X[r] = mt.to(static_cast<uint32_t>(x));
                acc[r] = 0;
                x = x * h % mod;
            }
            for (std::size_t i = num.size(); i-- > 0;) {
                const uint32_t c = num[i];
                for (unsigned r = 0; r < 8; ++r) {
                    uint32_t v = mt.mul(acc[r], X[r]) + c;
                    acc[r] = v >= M32 ? v - M32 : v;
                }
            }
            for (unsigned r = 0; r < cnt; ++r)
                out[t0 + r] = static_cast<uint32_t>(static_cast<uint64_t>(mt.from(acc[r])) * lm1 % mod);
        }
    });
    return out;
}

// ================================================================== Bernoulli reference
std::vector<BigRational> bernoulli_over_k(unsigned kmax) {
    const unsigned nmax = kmax / 2;
    std::vector<BigRational> out(kmax + 1);
    if (nmax == 0) return out;
    std::vector<BigInt> T(nmax + 1);
    T[1] = BigInt(1);
    for (unsigned k = 2; k <= nmax; ++k) T[k] = BigInt(static_cast<int64_t>(k - 1)) * T[k - 1];
    for (unsigned k = 2; k <= nmax; ++k)
        for (unsigned j = k; j <= nmax; ++j)
            T[j] = BigInt(static_cast<int64_t>(j - k)) * T[j - 1] + BigInt(static_cast<int64_t>(j - k + 2)) * T[j];
    for (unsigned k = 1; k <= nmax; ++k) {
        const BigInt f = BigInt::pow(BigInt(4), k);
        BigRational v(T[k], f * (f - BigInt(1)));
        if (k % 2 == 0) v = -v;
        out[2 * k] = v;
    }
    return out;
}

CwBernoulliRef cw_bernoulli_ref(uint32_t ell, unsigned m, unsigned M, uint32_t w, uint64_t rows,
                                const std::vector<BigRational>& Bk, unsigned kexact) {
    CwBernoulliRef R;
    R.ref.assign(rows, 0);
    const uint64_t mod = upow(ell, M);
    if (w % 2 == 1) {
        R.exact_rows = rows;  // B_k = 0 for odd k > 1
        return R;
    }
    NF_CHECK_MSG(M < ell, "cw_bernoulli_ref: M < ell");
    auto ex = [&](uint64_t k) -> uint64_t {
        NF_CHECK(k < Bk.size());
        uint64_t v = Bk[k].mod_prime_power(ell, M);
        if (k - 1 < M) {
            const uint64_t f = (mod + 1 - upow(ell, static_cast<unsigned>(k - 1)) % mod) % mod;
            v = static_cast<uint64_t>(mulmod_u64(v, f, mod));
        }
        return v;
    };
    std::vector<uint64_t> base(M);
    for (unsigned i = 0; i < M; ++i) {
        const uint64_t k = cw_row_k(ell, m, w, i);
        NF_CHECK_MSG(k <= kexact, "cw_bernoulli_ref: the Kummer law needs M exact weights");
        base[i] = ex(k);
    }
    // Lagrange weights: prod_{j != i} (t - j) / (i - j).
    std::vector<uint64_t> den(M);
    for (unsigned i = 0; i < M; ++i) {
        int64_t d = 1;
        for (unsigned jj = 0; jj < M; ++jj)
            if (jj != i) d *= static_cast<int64_t>(i) - static_cast<int64_t>(jj);
        const uint64_t dm = static_cast<uint64_t>(((d % static_cast<int64_t>(mod)) + static_cast<int64_t>(mod)) % static_cast<int64_t>(mod));
        // inverse mod ell^M (d is a unit: |d| < M! with M < ell)
        uint64_t inv = 1, b = dm, e = mod / ell * (ell - 1) - 1;  // Euler: phi(ell^M) - 1
        while (e) {
            if (e & 1u) inv = mulmod_u64(inv, b, mod);
            b = mulmod_u64(b, b, mod);
            e >>= 1;
        }
        den[i] = inv;
    }
    auto law = [&](uint64_t t) -> uint64_t {
        uint64_t tot = 0;
        for (unsigned i = 0; i < M; ++i) {
            uint64_t num = 1;
            for (unsigned jj = 0; jj < M; ++jj)
                if (jj != i) num = mulmod_u64(num, (t + mod - jj % mod) % mod, mod);
            tot = (tot + mulmod_u64(mulmod_u64(num, den[i], mod), base[i], mod)) % mod;
        }
        return tot;
    };
    for (uint64_t t = 0; t < rows; ++t) {
        const uint64_t k = cw_row_k(ell, m, w, t);
        if (k <= kexact) {
            R.ref[t] = static_cast<uint32_t>(ex(k));
            ++R.exact_rows;
            if (t >= M) {
                ++R.law_checked;
                R.law_held += (law(t) == R.ref[t]);
            }
        } else {
            R.ref[t] = static_cast<uint32_t>(law(t));
        }
    }
    return R;
}

// ================================================================== one cell
CwCellResult cw_run_cell(const CwCellSpec& spec, unsigned threads) {
    const double T0 = now_secs();
    CwCellResult res;
    res.spec = spec;
    const uint32_t ell = spec.ell;
    const unsigned m = spec.m;
    NF_CHECK_MSG(m >= 1 && ell >= 3, "cw_run_cell: ell >= 3, m >= 1");
    res.r = spec.r ? spec.r : m + 1;
    res.W = spec.W ? spec.W : res.r + 3 * m;
    NF_CHECK_MSG(res.W + 1 >= res.r + 3 * m,
                 "cw_run_cell: working precision too small: W must be >= r + 3m - 1 (nu known mod ell^m)");
    res.planned_peak_bytes = cw_predict(ell, m, res.r, res.W, static_cast<unsigned>(spec.classes.size()), threads).peak_bytes;
    CwLog lg;
    CwRingStats st;
    unsigned P = 0, K = 0;
    uint64_t n = 0, L = 0;
    {
        CwRing R(ell, m, res.W, threads);
        CwLogOptions opt = spec.log;
        opt.r = res.r;
        lg = cw_lambda(R, opt);
        st = R.stats;
        P = R.prime_count();
        K = R.ctx().K;
        n = R.n();
        L = R.ntt_length();
        if (spec.hash) res.sha16 = R.digits_sha16(lg.Lg);
    }
    res.s = lg.s;
    res.cy = lg.cy;
    res.terms = lg.terms;
    res.P = P;
    res.K = K;
    res.n = n;
    res.L = L;
    res.stats = st;
    res.secs_pi = lg.secs_pi;
    res.secs_pow = lg.secs_pow;
    res.secs_series = lg.secs_series;
    const double tm = now_secs();
    const LadicCtx cx(ell, res.W);
    const std::vector<CwMeasure> main = cw_measure(cx, m, lg.Lg.data(), lg.s + m, spec.classes, m, threads);
    std::vector<CwMeasure> low;
    const bool ctl = spec.control && m >= 2;
    if (ctl) {
        const uint64_t nl = n / ell;
        std::vector<uint32_t> ylow(nl * K);
        for (uint64_t u = 0; u < nl; ++u) cx.mul_lpow(lg.Lg.data() + u * ell * K, 1, ylow.data() + u * K);
        low = cw_measure(cx, m - 1, ylow.data(), lg.s + m - 1, spec.classes, m, threads);
    }
    lg.Lg.clear();
    lg.Lg.shrink_to_fit();
    const uint64_t rows = upow(ell, m - 1);
    res.rows = rows;
    const uint64_t mod = upow(ell, m);
    std::vector<BigRational> Bk_own;
    if (spec.bern_kmax && !(spec.bern_table && spec.bern_table->size() > spec.bern_kmax))
        Bk_own = bernoulli_over_k(spec.bern_kmax);
    const std::vector<BigRational>& Bk = Bk_own.empty() ? *spec.bern_table : Bk_own;
    for (std::size_t ci = 0; ci < spec.classes.size(); ++ci) {
        CwClassResult cr;
        cr.w = spec.classes[ci];
        cr.integral = main[ci].integral;
        cr.psi = cw_moments(main[ci].nu, ell, m, cr.w, rows, threads);
        cr.zeros_count.assign(m + 1, 0);
        uint64_t zk = 0;
        for (uint64_t t = 0; t < rows; ++t) {
            uint64_t q = ell;
            for (unsigned jj = 1; jj <= m; ++jj, q *= ell) {
                if (cr.psi[t] % q == 0) ++cr.zeros_count[jj];
            }
            if (cr.psi[t] == 0) {
                zk = cw_row_k(ell, m, cr.w, t);
                if (cr.zeros_top.size() < 16) cr.zeros_top.push_back(zk);
            }
        }
        if (cr.zeros_count[m] == 1) cr.alpha = static_cast<int64_t>((powmod_u64(1 + ell, zk, mod) + mod - 1) % mod);
        uint64_t sum = 0;
        for (uint32_t v : main[ci].nu) sum = (sum + v) % mod;
        cr.leopoldt = static_cast<uint32_t>((mod - mulmod_u64(sum, ell - 1, mod)) % mod);
        if (ctl) {
            cr.control_run = true;
            cr.control_integral = low[ci].integral;
            const std::vector<uint32_t> pl = cw_moments(low[ci].nu, ell, m, cr.w, rows, threads);
            const uint64_t qlow = mod / ell;
            for (uint64_t t = 0; t < rows; ++t) {
                cr.control_right_low += (pl[t] % qlow == cr.psi[t] % qlow);
                cr.control_right_high += (pl[t] == cr.psi[t]);
            }
        }
        // m-th differences in t vanish mod ell^m (Psi is a degree < m polynomial in t: a self-check of the moments).
        {
            std::vector<int64_t> binom(m + 1, 1);
            for (unsigned i = 1; i <= m; ++i) binom[i] = binom[i - 1] * static_cast<int64_t>(m - i + 1) / i;
            bool ok = true;
            for (uint64_t t = 0; t + m < rows && ok; ++t) {
                int64_t s = 0;
                for (unsigned i = 0; i <= m; ++i) {
                    const int64_t term = binom[i] * static_cast<int64_t>(cr.psi[t + i]) % static_cast<int64_t>(mod);
                    s = ((m - i) % 2 ? s - term : s + term) % static_cast<int64_t>(mod);
                }
                ok = (s % static_cast<int64_t>(mod) == 0);
            }
            cr.delta_law = ok;
        }
        if (spec.bern_kmax) {
            const CwBernoulliRef br = cw_bernoulli_ref(ell, m, m, cr.w, rows, Bk, spec.bern_kmax);
            cr.bern_run = true;
            cr.bern_exact_rows = br.exact_rows;
            cr.bern_law_checked = br.law_checked;
            cr.bern_law_held = br.law_held;
            for (uint64_t t = 0; t < rows; ++t) cr.bern_rows_ok += (br.ref[t] == cr.psi[t]);
        }
        res.classes.push_back(std::move(cr));
    }
    res.secs_measure = now_secs() - tm;
    res.secs_total = now_secs() - T0;
    return res;
}

// ================================================================== cost model
CwCostPrediction cw_predict(uint32_t ell, unsigned m, unsigned r, unsigned W, unsigned nclasses, unsigned threads) {
    CwCostPrediction c;
    c.r = r ? r : m + 1;
    c.W = W ? W : c.r + 3 * m;
    const LadicCtx cx(ell, c.W), c1(ell, c.W + 1);
    c.K = cx.K;
    const uint64_t blk = upow(ell, m - 1);
    c.n = blk * (ell - 1);
    c.L = ntt64_length(2 * c.n - 1);
    c.P = CwRing::planned_prime_count(ell, m, c.W);
    unsigned bits = 0, pop = 0;
    for (uint32_t x = ell; x; x >>= 1) {
        ++bits;
        pop += x & 1u;
    }
    c.squarings = static_cast<uint64_t>(c.r) * (bits - 1);
    const unsigned cy = c.r;  // observed: content_val(y) = r at every R19-7 cell
    const unsigned J = (c.W + cy - 1) / cy;
    c.products = static_cast<uint64_t>(c.r) * (pop - 1) + (J - 1);
    c.ntts = c.P * (2 * c.squarings + 3 * c.products);
    c.ntt_work = static_cast<double>(c.ntts) * static_cast<double>(c.L) * std::log2(static_cast<double>(c.L));
    // Fitted constants (docs/implementation/slot19/wpnf2/README.md, "cost model"): seconds per unit of work at
    // one thread, measured on the reference box at (347, m=3, r=3, W=11), T = 1 / 8 / 16 (2593 / 435 / 341 s,
    // loaded box), checked on (691, 2), (103, 3), (37, 4); effective speed-up s(T) = T / (1 + kappa (T - 1)).
    const double a_ntt = 2.6e-9;   // per L log2 L of one transform
    const double a_io = 3.0e-11;   // per product x n x P (residues, fold, Garner)
    const double a_bin = 1.2e-7;   // per n (pi^n from the binomials)
    const double a_meas = 1.0e-7;  // per n x (classes + 1) (traces, e_w projection, control)
    const double a_mom = 1.5e-9;   // per row x representative x class (Horner moments)
    const double kappa = 0.05;
    const double T = std::max(1u, threads);
    const double speed = T / (1.0 + kappa * (T - 1.0));
    const double nn = static_cast<double>(c.n);
    const double prod_count = static_cast<double>(c.squarings + c.products);
    const double work = a_ntt * c.ntt_work + a_io * prod_count * nn * c.P + a_bin * nn + a_meas * nn * (1.0 + nclasses) +
                        a_mom * static_cast<double>(blk) * static_cast<double>(blk) * nclasses;
    c.secs = work / speed;
    const uint64_t prod_peak = CwRing::planned_product_peak_bytes(ell, m, c.W);
    const uint64_t binom_peak = (c.n + 1) * c1.K * 4 + c.n * c.K * 4;
    c.peak_bytes = std::max(prod_peak, binom_peak) + (uint64_t(64) << 20);
    return c;
}

}  // namespace nf
