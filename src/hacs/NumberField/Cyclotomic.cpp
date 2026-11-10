// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Cyclotomic.cpp -- see Cyclotomic.h (slot-19 WP-NF).
#include "Cyclotomic.h"

#include <algorithm>

#include "ModUtil.h"
#include "NfCheck.h"
#include "Ntt.h"

namespace nf {

std::map<uint32_t, int64_t> kummer_exponents(uint32_t ell, unsigned k, uint64_t i) {
    const uint64_t m64 = ipow_u64(ell, k);
    NF_CHECK(m64 <= 0xffffffffull);
    const uint32_t m = static_cast<uint32_t>(m64);
    const uint64_t phi = static_cast<uint64_t>(ell - 1) * (m / ell);
    const uint64_t e = (phi - i % phi) % phi;  // -i mod phi
    std::map<uint32_t, int64_t> f;
    for (uint32_t c = 2; c < m; ++c) {
        if (c % ell == 0) continue;
        int64_t v = powmod32(c, e, m);
        if (v > static_cast<int64_t>(m / 2)) v -= m;
        f[c] = v;
    }
    return f;
}

// ---------------------------------------------------------------- CycloModRing

CycloModRing::CycloModRing(uint32_t ell, unsigned k, uint32_t M) : ell_(ell), k_(k), M_(M) {
    NF_CHECK(is_prime_u32(ell) && ell >= 3 && k >= 1 && M >= 2);
    const uint64_t m64 = ipow_u64(ell, k);
    NF_CHECK(m64 <= 0xffffffffull);
    m_ = static_cast<uint32_t>(m64);
    s_ = m_ / ell;
    n_ = (ell - 1) * s_;
    prec_ = 0;
    {
        uint64_t x = M;
        unsigned p = 0;
        while (x % ell == 0) {
            x /= ell;
            ++p;
        }
        if (x == 1) prec_ = p;
    }
    if (prec_ > 0) {
        pascal_.assign(static_cast<std::size_t>(ell) * ell, 0);
        for (uint32_t i = 0; i < ell; ++i) {
            pascal_[static_cast<std::size_t>(i) * ell] = 1;
            for (uint32_t j = 1; j <= i; ++j) {
                pascal_[static_cast<std::size_t>(i) * ell + j] =
                    (pascal_[static_cast<std::size_t>(i - 1) * ell + j] + pascal_[static_cast<std::size_t>(i - 1) * ell + j - 1]) % ell;
            }
        }
    }
}

CycloModRing::Elt CycloModRing::one() const {
    Elt r(n_, 0);
    r[0] = 1 % M_;
    return r;
}

CycloModRing::Elt CycloModRing::pi() const {
    std::vector<uint32_t> d = {1 % M_, M_ - 1};
    return reduce(d);
}

CycloModRing::Elt CycloModRing::reduce(const std::vector<uint32_t>& dense) const {
    std::vector<uint64_t> b(m_, 0);
    for (std::size_t i = 0; i < dense.size(); ++i) {
        uint64_t& t = b[i % m_];
        t += dense[i];
        if (t >= M_) t -= M_;
        if (t >= M_) t %= M_;
    }
    Elt out(n_);
    for (uint32_t i = 0; i < n_; ++i) out[i] = static_cast<uint32_t>(b[i]);
    // z^{n + t} = - sum_{j < ell - 1} z^{j s + t}, t < s
    for (uint32_t t = 0; t < s_; ++t) {
        const uint32_t c = static_cast<uint32_t>(b[n_ + t] % M_);
        if (!c) continue;
        for (uint32_t j = 0; j + 1 < ell_; ++j) {
            uint32_t& x = out[j * s_ + t];
            x = x >= c ? x - c : x + (M_ - c);
        }
    }
    return out;
}

CycloModRing::Elt CycloModRing::from_signed(const std::vector<int64_t>& coeffs) const {
    std::vector<uint32_t> d(coeffs.size());
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        int64_t v = coeffs[i] % static_cast<int64_t>(M_);
        if (v < 0) v += M_;
        d[i] = static_cast<uint32_t>(v);
    }
    return reduce(d);
}

CycloModRing::Elt CycloModRing::from_bigint(const std::vector<BigInt>& coeffs) const {
    std::vector<uint32_t> d(coeffs.size());
    for (std::size_t i = 0; i < coeffs.size(); ++i) d[i] = coeffs[i].mod_u32(M_);
    return reduce(d);
}

CycloModRing::Elt CycloModRing::add(const Elt& a, const Elt& b) const {
    Elt r(n_);
    for (uint32_t i = 0; i < n_; ++i) {
        const uint64_t t = static_cast<uint64_t>(a[i]) + b[i];
        r[i] = static_cast<uint32_t>(t >= M_ ? t - M_ : t);
    }
    return r;
}

CycloModRing::Elt CycloModRing::sub(const Elt& a, const Elt& b) const {
    Elt r(n_);
    for (uint32_t i = 0; i < n_; ++i) r[i] = a[i] >= b[i] ? a[i] - b[i] : a[i] + (M_ - b[i]);
    return r;
}

CycloModRing::Elt CycloModRing::mul(const Elt& a, const Elt& b) const {
    ++products_;
    return reduce(convolve_mod(a, b, M_));
}

CycloModRing::Elt CycloModRing::sqr(const Elt& a) const {
    ++products_;
    return reduce(convolve_mod(a, a, M_));
}

CycloModRing::Elt CycloModRing::pow(const Elt& a, uint64_t e) const {
    if (e == 0) return one();
    Elt r, b = a;
    bool first = true;
    while (e) {
        if (e & 1u) {
            r = first ? b : mul(r, b);
            first = false;
        }
        e >>= 1;
        if (e) b = sqr(b);
    }
    return r;
}

CycloModRing::Elt CycloModRing::multiexp(const std::vector<std::pair<const Elt*, uint64_t>>& terms) const {
    unsigned nb = 0;
    for (const auto& t : terms) {
        unsigned b = 0;
        for (uint64_t x = t.second; x; x >>= 1) ++b;
        nb = std::max(nb, b);
    }
    Elt r;
    bool have = false;
    for (unsigned bit = nb; bit-- > 0;) {
        if (have) r = sqr(r);
        for (const auto& t : terms) {
            if ((t.second >> bit) & 1u) {
                if (!have) {
                    r = *t.first;
                    have = true;
                } else {
                    r = mul(r, *t.first);
                }
            }
        }
    }
    return have ? r : one();
}

CycloModRing::Elt CycloModRing::sigma(const Elt& a, uint64_t t) const {
    t %= m_;
    NF_CHECK_MSG(t % ell_ != 0, "sigma_t needs t prime to ell");
    std::vector<uint32_t> d(m_, 0);
    for (uint32_t i = 0; i < n_; ++i) {
        if (!a[i]) continue;
        d[static_cast<std::size_t>(static_cast<uint64_t>(i) * t % m_)] = a[i];  // i -> i t is injective mod m
    }
    return reduce(d);
}

CycloModRing::Elt CycloModRing::unit_u(uint64_t c) const {
    c %= m_;
    NF_CHECK(c % ell_ != 0);
    std::vector<uint32_t> d(m_, 0);
    for (uint64_t j = 0; j < c; ++j) d[j] = 1 % M_;
    return reduce(d);
}

CycloModRing::Elt CycloModRing::unit_u_inv(uint64_t c) const {
    c %= m_;
    NF_CHECK(c % ell_ != 0);
    const uint32_t d = invmod32(c, m_);
    std::vector<uint32_t> dense(m_, 0);
    for (uint64_t t = 0; t < d; ++t) {
        uint32_t& x = dense[static_cast<std::size_t>(t * c % m_)];
        x = (x + 1) % M_;
    }
    return reduce(dense);
}

CycloModRing::Elt CycloModRing::product_unit(const std::map<uint32_t, int64_t>& f) const {
    // Units sharing an exponent are multiplied first (f_c = c^{-i} takes each value gcd(i, phi) times),
    // then one Straus pass over the distinct exponents.
    std::map<int64_t, Elt> grouped;
    for (const auto& kv : f) {
        if (kv.first == 1 || kv.second == 0) continue;
        Elt u = kv.second > 0 ? unit_u(kv.first) : unit_u_inv(kv.first);
        auto it = grouped.find(kv.second);
        if (it == grouped.end()) {
            grouped.emplace(kv.second, std::move(u));
        } else {
            it->second = mul(it->second, u);
        }
    }
    std::vector<Elt> bases;
    std::vector<uint64_t> exps;
    bases.reserve(grouped.size());
    for (auto& kv : grouped) {
        bases.push_back(std::move(kv.second));
        exps.push_back(static_cast<uint64_t>(kv.first > 0 ? kv.first : -kv.first));
    }
    std::vector<std::pair<const Elt*, uint64_t>> terms;
    for (std::size_t i = 0; i < bases.size(); ++i) terms.emplace_back(&bases[i], exps[i]);
    return multiexp(terms);
}

std::pair<unsigned, uint32_t> CycloModRing::vpi_lead(const Elt& a) const {
    NF_CHECK_MSG(prec_ > 0, "vpi needs M = ell^P");
    // content valuation
    unsigned kc = prec_;
    for (uint32_t x : a) {
        if (!x) continue;
        unsigned v = 0;
        uint32_t y = x;
        while (v < prec_ && y % ell_ == 0) {
            y /= ell_;
            ++v;
        }
        kc = std::min(kc, v);
        if (kc == 0) break;
    }
    if (kc == prec_) return {prec_ * n_, 0};
    uint32_t div = 1;
    for (unsigned i = 0; i < kc; ++i) div *= ell_;
    // F_ell digits of a / ell^kc, laid out on the ell^k digit tensor
    std::vector<uint32_t> A(m_, 0);
    for (uint32_t i = 0; i < n_; ++i) A[i] = (a[i] / div) % ell_;
    // Taylor shift z = 1 - pi mod ell, axis by axis (Lucas: C(i,j) = prod_d C(i_d, j_d) mod ell)
    std::vector<uint32_t> col(ell_), out(ell_);
    uint32_t stride = 1;
    for (unsigned d = 0; d < k_; ++d) {
        const uint32_t block = stride * ell_;
        for (uint32_t base = 0; base < m_; base += block) {
            for (uint32_t inner = 0; inner < stride; ++inner) {
                bool nz = false;
                for (uint32_t i = 0; i < ell_; ++i) {
                    col[i] = A[base + inner + i * stride];
                    nz = nz || col[i];
                }
                if (!nz) continue;
                for (uint32_t j = 0; j < ell_; ++j) {
                    uint64_t s = 0;
                    for (uint32_t i = j; i < ell_; ++i) s += static_cast<uint64_t>(pascal_[static_cast<std::size_t>(i) * ell_ + j]) * col[i];
                    out[j] = static_cast<uint32_t>(s % ell_);
                }
                for (uint32_t j = 0; j < ell_; ++j) A[base + inner + j * stride] = out[j];
            }
        }
        stride = block;
    }
    for (uint32_t j = 0; j < m_; ++j) {
        if (A[j]) {
            NF_CHECK(j < n_);
            const uint32_t lead = (j % 2 == 0) ? A[j] : (ell_ - A[j]) % ell_;  // b_j = (-1)^j sum a_i C(i,j)
            return {n_ * kc + j, lead};
        }
    }
    NF_CHECK_MSG(false, "vpi_lead: nonzero content but zero F_ell image");
    return {0, 0};
}

std::pair<unsigned, uint32_t> CycloModRing::vone_lead(const Elt& a) const {
    Elt d = a;
    d[0] = d[0] >= 1 % M_ ? d[0] - 1 % M_ : d[0] + M_ - 1;
    return vpi_lead(d);
}

uint32_t CycloModRing::eval_at(const Elt& a, uint32_t r, uint32_t q) {
    uint64_t s = 0;
    for (std::size_t i = a.size(); i-- > 0;) s = (s * r + a[i] % q) % q;
    return static_cast<uint32_t>(s);
}

// ---------------------------------------------------------------- CycloZ

CycloZ::CycloZ(uint32_t ell, unsigned k) : ell_(ell), k_(k) {
    NF_CHECK(is_prime_u32(ell) && ell >= 3 && k >= 1);
    const uint64_t m64 = ipow_u64(ell, k);
    NF_CHECK(m64 <= 0xffffffffull);
    m_ = static_cast<uint32_t>(m64);
    s_ = m_ / ell;
    n_ = (ell - 1) * s_;
}

CycloZ::Elt CycloZ::one() const {
    Elt r(n_);
    r[0] = BigInt(1);
    return r;
}

CycloZ::Elt CycloZ::reduce(std::vector<BigInt> dense) const {
    std::vector<BigInt> b(m_);
    for (std::size_t i = 0; i < dense.size(); ++i) {
        if (!dense[i].is_zero()) b[i % m_] += dense[i];
    }
    for (uint32_t t = 0; t < s_; ++t) {
        const BigInt c = b[n_ + t];
        if (c.is_zero()) continue;
        for (uint32_t j = 0; j + 1 < ell_; ++j) b[j * s_ + t] -= c;
    }
    b.resize(n_);
    return b;
}

CycloZ::Elt CycloZ::mul(const Elt& a, const Elt& b) const {
    std::vector<BigInt> d(2 * n_ - 1);
    for (uint32_t i = 0; i < n_; ++i) {
        if (a[i].is_zero()) continue;
        for (uint32_t j = 0; j < n_; ++j) {
            if (b[j].is_zero()) continue;
            d[i + j] += a[i] * b[j];
        }
    }
    return reduce(std::move(d));
}

CycloZ::Elt CycloZ::pow(const Elt& a, uint64_t e) const {
    Elt r = one(), b = a;
    while (e) {
        if (e & 1u) r = mul(r, b);
        e >>= 1;
        if (e) b = mul(b, b);
    }
    return r;
}

CycloZ::Elt CycloZ::multiexp(const std::vector<std::pair<const Elt*, uint64_t>>& terms) const {
    unsigned nb = 0;
    for (const auto& t : terms) {
        unsigned b = 0;
        for (uint64_t x = t.second; x; x >>= 1) ++b;
        nb = std::max(nb, b);
    }
    Elt r;
    bool have = false;
    for (unsigned bit = nb; bit-- > 0;) {
        if (have) r = mul(r, r);
        for (const auto& t : terms) {
            if ((t.second >> bit) & 1u) {
                r = have ? mul(r, *t.first) : *t.first;
                have = true;
            }
        }
    }
    return have ? r : one();
}

CycloZ::Elt CycloZ::unit_u(uint64_t c) const {
    c %= m_;
    NF_CHECK(c % ell_ != 0);
    std::vector<BigInt> d(m_);
    for (uint64_t j = 0; j < c; ++j) d[j] = BigInt(1);
    return reduce(std::move(d));
}

CycloZ::Elt CycloZ::unit_u_inv(uint64_t c) const {
    c %= m_;
    NF_CHECK(c % ell_ != 0);
    const uint32_t dd = invmod32(c, m_);
    std::vector<BigInt> d(m_);
    for (uint64_t t = 0; t < dd; ++t) d[static_cast<std::size_t>(t * c % m_)] += BigInt(1);
    return reduce(std::move(d));
}

CycloZ::Elt CycloZ::product_unit(const std::map<uint32_t, int64_t>& f) const {
    std::vector<Elt> bases;
    std::vector<uint64_t> exps;
    for (const auto& kv : f) {
        if (kv.first == 1 || kv.second == 0) continue;
        bases.push_back(kv.second > 0 ? unit_u(kv.first) : unit_u_inv(kv.first));
        exps.push_back(static_cast<uint64_t>(kv.second > 0 ? kv.second : -kv.second));
    }
    std::vector<std::pair<const Elt*, uint64_t>> terms;
    for (std::size_t i = 0; i < bases.size(); ++i) terms.emplace_back(&bases[i], exps[i]);
    return multiexp(terms);
}

CycloZ::Elt CycloZ::sigma(const Elt& a, uint64_t t) const {
    t %= m_;
    NF_CHECK(t % ell_ != 0);
    std::vector<BigInt> d(m_);
    for (uint32_t i = 0; i < n_; ++i) d[static_cast<std::size_t>(static_cast<uint64_t>(i) * t % m_)] = a[i];
    return reduce(std::move(d));
}

long CycloZ::vpi_exact(const Elt& a) const {
    NF_CHECK_MSG(k_ == 1, "CycloZ::vpi_exact: k = 1 only (pi^{ell-1} ~ ell)");
    // binomials C(i, j) for i < n exactly
    long best = -1;
    std::vector<std::vector<BigInt>> C(n_, std::vector<BigInt>(n_));
    for (uint32_t i = 0; i < n_; ++i) {
        C[i][0] = BigInt(1);
        for (uint32_t j = 1; j <= i; ++j) C[i][j] = C[i - 1][j] + C[i - 1][j - 1];  // C[i-1][i] is 0
    }
    for (uint32_t j = 0; j < n_; ++j) {
        BigInt b(0);
        for (uint32_t i = j; i < n_; ++i) b += a[i] * C[i][j];
        if (b.is_zero()) continue;
        const long v = static_cast<long>(n_) * static_cast<long>(b.valuation(ell_)) + static_cast<long>(j);
        if (best < 0 || v < best) best = v;
    }
    return best;
}

uint32_t eval_at(const CycloZ::Elt& a, uint32_t r, uint32_t q) {
    uint64_t s = 0;
    for (std::size_t i = a.size(); i-- > 0;) s = (s * r + a[i].mod_u32(q)) % q;
    return static_cast<uint32_t>(s);
}

uint32_t eval_product_unit(const std::map<uint32_t, int64_t>& f, uint32_t r, uint32_t q) {
    const uint64_t inv = invmod32((1 + static_cast<uint64_t>(q) - r) % q, q);
    uint64_t s = 1;
    for (const auto& kv : f) {
        if (kv.first == 1 || kv.second == 0) continue;
        const uint64_t uc = (1 + static_cast<uint64_t>(q) - powmod32(r, kv.first, q)) % q * inv % q;
        int64_t e = kv.second % static_cast<int64_t>(q - 1);
        if (e < 0) e += q - 1;
        s = s * powmod32(uc, static_cast<uint64_t>(e), q) % q;
    }
    return static_cast<uint32_t>(s);
}

std::vector<uint32_t> split_primes(uint32_t m, unsigned count) {
    std::vector<uint32_t> out;
    for (uint64_t q = static_cast<uint64_t>(m) + 1; out.size() < count; q += m) {
        NF_CHECK(q <= 0xffffffffull);
        if (is_prime_u32(static_cast<uint32_t>(q))) out.push_back(static_cast<uint32_t>(q));
    }
    return out;
}

uint32_t root_of_order(uint32_t q, uint32_t m) {
    NF_CHECK((q - 1) % m == 0);
    const std::vector<uint32_t> pf = prime_factors_u32(m);
    for (uint32_t g = 2; g < q; ++g) {
        const uint32_t r = powmod32(g, (q - 1) / m, q);
        bool ok = r != 1;
        for (uint32_t p : pf) {
            if (powmod32(r, m / p, q) == 1) ok = false;
        }
        if (ok) return r;
    }
    NF_CHECK_MSG(false, "root_of_order: none");
    return 0;
}

}  // namespace nf
