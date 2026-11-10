// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// PolyAlg.cpp -- see PolyAlg.h (slot-19 WP-NF).
#include "PolyAlg.h"

#include <algorithm>
#include <utility>

#include "ModUtil.h"
#include "NfCheck.h"

namespace nf {

// ---------------------------------------------------------------- Newton polygon

std::vector<NewtonSegment> newton_polygon(const std::vector<BigInt>& f, uint32_t p) {
    struct Pt {
        int64_t x, y;
    };
    std::vector<Pt> pts;
    for (std::size_t i = 0; i < f.size(); ++i) {
        if (!f[i].is_zero()) pts.push_back({static_cast<int64_t>(i), static_cast<int64_t>(f[i].valuation(p))});
    }
    std::vector<Pt> hull;
    for (const Pt& q : pts) {
        while (hull.size() >= 2) {
            const Pt& o = hull[hull.size() - 2];
            const Pt& a = hull.back();
            const int64_t cross = (a.x - o.x) * (q.y - o.y) - (a.y - o.y) * (q.x - o.x);
            if (cross <= 0) {
                hull.pop_back();
            } else {
                break;
            }
        }
        hull.push_back(q);
    }
    std::vector<NewtonSegment> segs;
    for (std::size_t i = 1; i < hull.size(); ++i) {
        NewtonSegment s;
        s.x0 = hull[i - 1].x;
        s.y0 = hull[i - 1].y;
        s.x1 = hull[i].x;
        s.y1 = hull[i].y;
        segs.push_back(s);
    }
    return segs;
}

// ---------------------------------------------------------------- polynomials mod N

PolyMod poly_trim(PolyMod a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
    return a;
}

PolyMod poly_add(const PolyMod& a, const PolyMod& b, uint32_t N) {
    PolyMod r(std::max(a.size(), b.size()), 0);
    for (std::size_t i = 0; i < r.size(); ++i) {
        const uint64_t x = (i < a.size() ? a[i] : 0u) + static_cast<uint64_t>(i < b.size() ? b[i] : 0u);
        r[i] = static_cast<uint32_t>(x % N);
    }
    return poly_trim(r);
}

PolyMod poly_sub(const PolyMod& a, const PolyMod& b, uint32_t N) {
    PolyMod r(std::max(a.size(), b.size()), 0);
    for (std::size_t i = 0; i < r.size(); ++i) {
        const uint64_t x = (i < a.size() ? a[i] : 0u) + static_cast<uint64_t>(N) - (i < b.size() ? b[i] : 0u) % N;
        r[i] = static_cast<uint32_t>(x % N);
    }
    return poly_trim(r);
}

PolyMod poly_mul(const PolyMod& a, const PolyMod& b, uint32_t N) {
    if (a.empty() || b.empty()) return {};
    std::vector<uint64_t> acc(a.size() + b.size() - 1, 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) acc[i + j] = (acc[i + j] + static_cast<uint64_t>(a[i]) * b[j] % N) % N;
    }
    PolyMod r(acc.size());
    for (std::size_t i = 0; i < acc.size(); ++i) r[i] = static_cast<uint32_t>(acc[i]);
    return poly_trim(r);
}

void poly_divrem(const PolyMod& a, const PolyMod& b0, uint32_t N, PolyMod& q, PolyMod& r) {
    const PolyMod b = poly_trim(b0);
    NF_CHECK_MSG(!b.empty(), "poly_divrem: division by zero");
    const uint32_t linv = invmod32(b.back(), N);
    r = poly_trim(a);
    q.clear();
    if (r.size() < b.size()) return;
    q.assign(r.size() - b.size() + 1, 0);
    for (std::size_t top = r.size(); top >= b.size(); --top) {
        const std::size_t k = top - 1;  // the coefficient being cancelled
        const uint32_t c = mulmod32(r[k], linv, N);
        if (c) {
            const std::size_t sh = k - (b.size() - 1);
            q[sh] = c;
            for (std::size_t j = 0; j < b.size(); ++j) {
                const uint64_t sub = static_cast<uint64_t>(c) * b[j] % N;
                r[sh + j] = static_cast<uint32_t>((r[sh + j] + N - sub) % N);
            }
        }
    }
    r = poly_trim(r);
    q = poly_trim(q);
}

void poly_xgcd_coprime(const PolyMod& a, const PolyMod& b, uint32_t p, PolyMod& s, PolyMod& t) {
    NF_CHECK(is_prime_u32(p));
    PolyMod r0 = poly_trim(a), r1 = poly_trim(b);
    PolyMod s0 = {1}, s1 = {}, t0 = {}, t1 = {1};
    while (!r1.empty()) {
        PolyMod q, r;
        poly_divrem(r0, r1, p, q, r);
        PolyMod sn = poly_sub(s0, poly_mul(q, s1, p), p);
        PolyMod tn = poly_sub(t0, poly_mul(q, t1, p), p);
        r0 = std::move(r1);
        r1 = std::move(r);
        s0 = std::move(s1);
        s1 = std::move(sn);
        t0 = std::move(t1);
        t1 = std::move(tn);
    }
    NF_CHECK_MSG(r0.size() == 1, "poly_xgcd_coprime: not coprime");
    const uint32_t ci = invmod32(r0[0], p);
    s = poly_mul(s0, PolyMod{ci}, p);
    t = poly_mul(t0, PolyMod{ci}, p);
}

PolyMod poly_from_bigint(const std::vector<BigInt>& f, uint32_t N) {
    PolyMod r(f.size());
    for (std::size_t i = 0; i < f.size(); ++i) r[i] = f[i].mod_u32(N);
    return poly_trim(r);
}

HenselPair hensel_lift(const std::vector<BigInt>& f, const PolyMod& g0, const PolyMod& h0, uint32_t p, unsigned M) {
    NF_CHECK(is_prime_u32(p) && M >= 1);
    const uint64_t PM = ipow_u64(p, M);
    NF_CHECK_MSG(PM <= 0xffffffffull, "hensel_lift: p^M must fit in 32 bits");
    const uint32_t N = static_cast<uint32_t>(PM);
    NF_CHECK_MSG(!f.empty() && f.back() == BigInt(1), "hensel_lift: f must be monic");
    PolyMod g = poly_trim(g0), h = poly_trim(h0);
    NF_CHECK_MSG(!g.empty() && g.back() == 1, "hensel_lift: g must be monic");
    const PolyMod fp = poly_from_bigint(f, p);
    NF_CHECK_MSG(poly_sub(fp, poly_mul(g, h, p), p).empty(), "hensel_lift: f != g h mod p");
    PolyMod s, t;
    poly_xgcd_coprime(g, h, p, s, t);  // s g + t h = 1 mod p
    const PolyMod fN = poly_from_bigint(f, N);
    uint64_t pk = p;
    for (unsigned k = 1; k < M; ++k) {
        // e = (f - g h) / p^k mod p
        const PolyMod diff = poly_sub(fN, poly_mul(g, h, N), N);
        PolyMod e(diff.size());
        for (std::size_t i = 0; i < diff.size(); ++i) {
            NF_CHECK(diff[i] % pk == 0);
            e[i] = static_cast<uint32_t>((diff[i] / pk) % p);
        }
        e = poly_trim(e);
        // G = (t e) rem g, H = s e + qq h, where t e = qq g + G  (all mod p)
        PolyMod qq, G;
        poly_divrem(poly_mul(t, e, p), g, p, qq, G);
        PolyMod H = poly_add(poly_mul(s, e, p), poly_mul(qq, h, p), p);
        PolyMod Gs(G.size()), Hs(H.size());
        for (std::size_t i = 0; i < G.size(); ++i) Gs[i] = static_cast<uint32_t>(static_cast<uint64_t>(G[i]) * pk % N);
        for (std::size_t i = 0; i < H.size(); ++i) Hs[i] = static_cast<uint32_t>(static_cast<uint64_t>(H[i]) * pk % N);
        g = poly_add(g, Gs, N);
        h = poly_add(h, Hs, N);
        pk *= p;
    }
    NF_CHECK(poly_sub(fN, poly_mul(g, h, N), N).empty());
    return {g, h, N};
}

// ---------------------------------------------------------------- Q(alpha)

namespace {
using QPoly = std::vector<BigRational>;
QPoly qtrim(QPoly a) {
    while (!a.empty() && a.back().is_zero()) a.pop_back();
    return a;
}
QPoly qmul(const QPoly& a, const QPoly& b) {
    if (a.empty() || b.empty()) return {};
    QPoly r(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].is_zero()) continue;
        for (std::size_t j = 0; j < b.size(); ++j) r[i + j] += a[i] * b[j];
    }
    return qtrim(r);
}
QPoly qsub(const QPoly& a, const QPoly& b) {
    QPoly r(std::max(a.size(), b.size()));
    for (std::size_t i = 0; i < r.size(); ++i) {
        r[i] = (i < a.size() ? a[i] : BigRational()) - (i < b.size() ? b[i] : BigRational());
    }
    return qtrim(r);
}
void qdivrem(const QPoly& a, const QPoly& b0, QPoly& q, QPoly& r) {
    const QPoly b = qtrim(b0);
    NF_CHECK(!b.empty());
    r = qtrim(a);
    q.clear();
    if (r.size() < b.size()) return;
    q.assign(r.size() - b.size() + 1, BigRational());
    while (!r.empty() && r.size() >= b.size()) {
        const std::size_t sh = r.size() - b.size();
        const BigRational c = r.back() / b.back();
        q[sh] = c;
        for (std::size_t j = 0; j < b.size(); ++j) r[sh + j] -= c * b[j];
        r.pop_back();  // leading term cancelled exactly
        r = qtrim(r);
    }
    q = qtrim(q);
}
}  // namespace

NumberFieldQ::NumberFieldQ(std::vector<BigRational> minpoly) {
    f_ = qtrim(std::move(minpoly));
    NF_CHECK_MSG(f_.size() >= 2, "NumberFieldQ: degree >= 1 required");
    const BigRational lc = f_.back();
    for (auto& c : f_) c /= lc;
    d_ = static_cast<unsigned>(f_.size() - 1);
}

NumberFieldQ::Elt NumberFieldQ::reduce(std::vector<BigRational> a) const {
    QPoly q, r;
    qdivrem(a, f_, q, r);
    r.resize(d_);
    return r;
}

NumberFieldQ::Elt NumberFieldQ::from_int(int64_t v) const {
    Elt r(d_);
    r[0] = BigRational(v);
    return r;
}

NumberFieldQ::Elt NumberFieldQ::alpha() const {
    std::vector<BigRational> x = {BigRational(0), BigRational(1)};
    return reduce(x);
}

NumberFieldQ::Elt NumberFieldQ::add(const Elt& a, const Elt& b) const {
    Elt r(d_);
    for (unsigned i = 0; i < d_; ++i) r[i] = a[i] + b[i];
    return r;
}

NumberFieldQ::Elt NumberFieldQ::sub(const Elt& a, const Elt& b) const {
    Elt r(d_);
    for (unsigned i = 0; i < d_; ++i) r[i] = a[i] - b[i];
    return r;
}

NumberFieldQ::Elt NumberFieldQ::mul(const Elt& a, const Elt& b) const { return reduce(qmul(qtrim(a), qtrim(b))); }

NumberFieldQ::Elt NumberFieldQ::pow(const Elt& a, uint64_t e) const {
    Elt r = from_int(1), b = a;
    while (e) {
        if (e & 1u) r = mul(r, b);
        e >>= 1;
        if (e) b = mul(b, b);
    }
    return r;
}

bool NumberFieldQ::is_zero(const Elt& a) const {
    for (const auto& c : a) {
        if (!c.is_zero()) return false;
    }
    return true;
}

bool NumberFieldQ::equal(const Elt& a, const Elt& b) const { return is_zero(sub(a, b)); }

NumberFieldQ::Elt NumberFieldQ::inv(const Elt& a) const {
    NF_CHECK_MSG(!is_zero(a), "NumberFieldQ::inv of zero");
    QPoly r0 = f_, r1 = qtrim(a);
    QPoly s0 = {}, s1 = {BigRational(1)};
    while (r1.size() > 1) {
        QPoly q, r;
        qdivrem(r0, r1, q, r);
        QPoly sn = qsub(s0, qmul(q, s1));
        if (!r.empty()) {
            // monic remainders (s_i a == r_i mod f is kept): bounds the size of the reduced rationals
            const BigRational lc = r.back();
            for (auto& c : r) c /= lc;
            for (auto& c : sn) c /= lc;
        }
        r0 = std::move(r1);
        r1 = std::move(r);
        s0 = std::move(s1);
        s1 = std::move(sn);
    }
    NF_CHECK_MSG(r1.size() == 1, "NumberFieldQ::inv: element shares a factor with the minimal polynomial");
    const BigRational c = r1[0];
    for (auto& x : s1) x /= c;
    return reduce(s1);
}

}  // namespace nf
