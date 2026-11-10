// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 21, P4) wild characters on the L-side -- see WildCharacter.h and docs/implementation/slot21/P4_spec.md.
#include "WildCharacter.h"

#include "Bernoulli.h"

#include "NumberField/BigRational.h"

#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace hecke_ext {
namespace lvalue {
namespace wild {

namespace {

using engine::scalar::pow_wrap;

uint64_t PowMod(uint64_t b, uint64_t e, uint64_t q) {
    u128 r = 1 % q, x = b % q;
    while (e) {
        if (e & 1) r = r * x % q;
        x = x * x % q;
        e >>= 1;
    }
    return static_cast<uint64_t>(r);
}

uint32_t ValU(uint64_t x, uint64_t ell) {
    uint32_t v = 0;
    while (x && x % ell == 0) {
        x /= ell;
        ++v;
    }
    return v;
}

} // namespace

// ---- OArith ----------------------------------------------------------------------------------------------
OArith MakeOArith(uint64_t ell, uint32_t m1, uint32_t P) {
    ENGINE_CHECK_MSG(ell >= 5 && m1 >= 1 && P >= 1, "wild::MakeOArith: ell >= 5, m1 >= 1, P >= 1");
    ENGINE_CHECK_MSG(static_cast<double>(P) * std::log2(static_cast<double>(ell)) < 62.0, "wild::MakeOArith: ell^P >= 2^62");
    OArith O;
    O.ell = ell;
    O.m1 = m1;
    O.P = P;
    O.q = pow_wrap(ell, P);
    O.order = static_cast<uint32_t>(pow_wrap(ell, m1));
    O.s = static_cast<uint32_t>(pow_wrap(ell, m1 - 1));
    O.e = static_cast<uint32_t>((ell - 1) * O.s);
    return O;
}

OArith::Elt OArith::one() const {
    Elt x(e, 0);
    x[0] = 1 % q;
    return x;
}

OArith::Elt OArith::zeta_pow(uint64_t ex) const {
    std::vector<uint64_t> d(order, 0);
    d[ex % order] = 1 % q;
    return reduce(std::move(d));
}

OArith::Elt OArith::reduce(std::vector<uint64_t> d) const {
    ENGINE_CHECK_MSG(d.size() == order, "wild::OArith::reduce: dense length must be ell^{m1}");
    // z^{(ell-1)s + r} = -sum_{j < ell-1} z^{j s + r}
    for (uint32_t r = 0; r < s; ++r) {
        const uint64_t c = d[e + r] % q;
        if (c == 0) continue;
        for (uint64_t j = 0; j + 1 < ell; ++j) {
            uint64_t& y = d[j * s + r];
            y = (y % q + q - c) % q;
        }
    }
    Elt out(e);
    for (uint32_t x = 0; x < e; ++x) out[x] = d[x] % q;
    return out;
}

OArith::Elt OArith::add(const Elt& a, const Elt& b) const {
    Elt c(e);
    for (uint32_t i = 0; i < e; ++i) c[i] = (a[i] + b[i]) % q;
    return c;
}

OArith::Elt OArith::sub(const Elt& a, const Elt& b) const {
    Elt c(e);
    for (uint32_t i = 0; i < e; ++i) c[i] = (a[i] + q - b[i]) % q;
    return c;
}

OArith::Elt OArith::scal(uint64_t c, const Elt& a) const {
    Elt out(e);
    for (uint32_t i = 0; i < e; ++i) out[i] = mulq(c % q, a[i]);
    return out;
}

OArith::Elt OArith::mul(const Elt& a, const Elt& b) const {
    std::vector<u128> acc(order, 0);
    for (uint32_t i = 0; i < e; ++i) {
        if (a[i] == 0) continue;
        for (uint32_t j = 0; j < e; ++j) {
            if (b[j] == 0) continue;
            const uint32_t x = (i + j) % order;   // z^{order} = 1
            acc[x] = (acc[x] + static_cast<u128>(a[i]) * b[j]) % q;
        }
    }
    std::vector<uint64_t> d(order);
    for (uint32_t x = 0; x < order; ++x) d[x] = static_cast<uint64_t>(acc[x]);
    return reduce(std::move(d));
}

bool OArith::is_zero(const Elt& a) const {
    return std::all_of(a.begin(), a.end(), [&](uint64_t x) { return x % q == 0; });
}

bool OArith::eq_mod(const Elt& a, const Elt& b, uint64_t modulus) const {
    for (uint32_t i = 0; i < e; ++i)
        if ((a[i] % modulus) != (b[i] % modulus)) return false;
    return true;
}

uint32_t OArith::vpi(const Elt& a) const {
    // a = sum_j c_j z^j = sum_i pi^i (-1)^i sum_{j >= i} C(j, i) c_j  (z = 1 - pi, j < e: no reduction), and
    // v_pi(sum_i b_i pi^i) = min_i (e v(b_i) + i) since ell = pi^e * unit and the i are distinct mod e.
    uint32_t best = e * P;
    std::vector<uint64_t> col(e, 0);   // C(j, i) mod q for the current i, j = 0 .. e-1
    for (uint32_t j = 0; j < e; ++j) col[j] = 1 % q;   // i = 0
    for (uint32_t i = 0; i < e; ++i) {
        if (i > 0) {
            // C(j, i) = C(j-1, i) + C(j-1, i-1)
            std::vector<uint64_t> nc(e, 0);
            for (uint32_t j = 1; j < e; ++j) nc[j] = (nc[j - 1] + col[j - 1]) % q;
            col.swap(nc);
        }
        u128 b = 0;
        for (uint32_t j = i; j < e; ++j)
            if (a[j] && col[j]) b = (b + static_cast<u128>(a[j]) * col[j]) % q;
        uint64_t bb = static_cast<uint64_t>(b);
        if (i % 2 == 1) bb = (q - bb) % q;
        if (bb == 0) continue;
        best = std::min<uint32_t>(best, e * ValU(bb, ell) + i);
    }
    return best;
}

// ---- characters ------------------------------------------------------------------------------------------
std::vector<uint32_t> LogTable(uint64_t ell, uint32_t m) {
    const uint64_t F = pow_wrap(ell, m), G = pow_wrap(ell, m - 1), u = 1 + ell;
    std::vector<uint32_t> idx(F, 0), out(F, 0);
    std::vector<char> seen(F, 0);
    uint64_t x = 1;
    for (uint64_t i = 0; i < G; ++i) {
        idx[x] = static_cast<uint32_t>(i);
        seen[x] = 1;
        x = x * u % F;
    }
    for (uint64_t a = 1; a < F; ++a) {
        if (a % ell == 0) continue;
        const uint64_t w = TeichmullerPowMod(a % ell, 1, ell, m) % F;
        const uint64_t br = static_cast<uint64_t>(static_cast<u128>(a) * InvModU64(w, F) % F);
        ENGINE_CHECK_MSG(seen[br], "wild::LogTable: <a> is not a power of u");
        out[a] = idx[br];
    }
    return out;
}

WildChar MakeWildChar(uint64_t ell, uint32_t m, int64_t i, int64_t t, uint32_t P, const std::vector<uint32_t>& logs) {
    WildChar c;
    c.ell = ell;
    c.m = m;
    c.F = pow_wrap(ell, m);
    c.i = i;
    c.t = t;
    ENGINE_CHECK_MSG(logs.size() == c.F, "wild::MakeWildChar: log table of the wrong size");
    const int64_t L1 = static_cast<int64_t>(ell - 1);
    const uint32_t ie = static_cast<uint32_t>(((i % L1) + L1) % L1);
    const int64_t G = static_cast<int64_t>(pow_wrap(ell, m - 1));
    const int64_t tt = ((t % G) + G) % G;
    c.w.assign(c.F, 0);
    c.ex.assign(c.F, 0);
    for (uint64_t a = 1; a < c.F; ++a) {
        if (a % ell == 0) continue;
        c.w[a] = TeichmullerPowMod(a % ell, ie, ell, P);
        c.ex[a] = static_cast<uint32_t>((static_cast<u128>(tt) * logs[a]) % static_cast<u128>(G));
    }
    return c;
}

OArith::Elt WildChar::value(const OArith& O, uint64_t a) const {
    a %= F;
    if (a % ell == 0) return O.zero();
    return O.scal(w[a], O.zeta_pow(ex[a]));
}

OArith::Elt CharSum(const OArith& O, const WildChar& chi, const std::vector<uint64_t>& x) {
    std::vector<u128> acc(O.order, 0);
    for (uint64_t a = 0; a < x.size(); ++a) {
        const uint64_t r = a % chi.F;
        if (r % chi.ell == 0 || x[a] == 0) continue;
        u128& y = acc[chi.ex[r] % O.order];
        y = (y + static_cast<u128>(chi.w[r] % O.q) * (x[a] % O.q)) % O.q;
    }
    std::vector<uint64_t> d(O.order);
    for (uint32_t k = 0; k < O.order; ++k) d[k] = static_cast<uint64_t>(acc[k]);
    return O.reduce(std::move(d));
}

std::vector<OArith::Elt> WildBernoulliFB(const OArith& O, const WildChar& chi, uint32_t rmax, const std::vector<uint64_t>& ellB) {
    ENGINE_CHECK_MSG(ellB.size() > rmax, "wild::WildBernoulliFB: ell B_s needed for s <= rmax");
    const uint64_t q = O.q, F = chi.F;
    // S_q = sum_{a < F} chi(a) a^q, q = 0 .. rmax
    std::vector<OArith::Elt> S(rmax + 1);
    {
        std::vector<std::vector<u128>> acc(rmax + 1, std::vector<u128>(O.order, 0));
        for (uint64_t a = 1; a < F; ++a) {
            if (a % chi.ell == 0) continue;
            const uint32_t x = chi.ex[a] % O.order;
            uint64_t p = chi.w[a] % q;
            for (uint32_t r = 0; r <= rmax; ++r) {
                acc[r][x] = (acc[r][x] + p) % q;
                p = O.mulq(p, a % q);
            }
        }
        for (uint32_t r = 0; r <= rmax; ++r) {
            std::vector<uint64_t> d(O.order);
            for (uint32_t k = 0; k < O.order; ++k) d[k] = static_cast<uint64_t>(acc[r][k]);
            S[r] = O.reduce(std::move(d));
        }
    }
    // F^s B_s = ell^{m s - 1} (ell B_s), s >= 1
    std::vector<uint64_t> FB(rmax + 1, 0);
    FB[0] = 1 % q;
    for (uint32_t s = 1; s <= rmax; ++s) {
        const uint64_t ex = static_cast<uint64_t>(chi.m) * s - 1;
        if (ex >= O.P || ellB[s] == 0) continue;
        FB[s] = O.mulq(ellB[s] % q, pow_wrap(O.ell, static_cast<uint32_t>(ex)));
    }
    std::vector<OArith::Elt> out(rmax + 1, O.zero());
    std::vector<uint64_t> row{1 % q};   // C(r, s) mod q
    for (uint32_t r = 1; r <= rmax; ++r) {
        std::vector<uint64_t> nr(r + 1, 1 % q);
        for (uint32_t s = 1; s < r; ++s) nr[s] = (row[s - 1] + row[s]) % q;
        row.swap(nr);
        std::vector<u128> acc(O.e, 0);
        for (uint32_t s = 0; s <= r; ++s) {
            if (FB[s] == 0) continue;
            const uint64_t c = O.mulq(row[s], FB[s]);
            const OArith::Elt& Sv = S[r - s];
            for (uint32_t x = 0; x < O.e; ++x) acc[x] = (acc[x] + static_cast<u128>(c) * Sv[x]) % q;
        }
        OArith::Elt v(O.e);
        for (uint32_t x = 0; x < O.e; ++x) v[x] = static_cast<uint64_t>(acc[x]);
        out[r] = v;
    }
    return out;
}

OArith::Elt GroupRingAtWildPoint(const OArith& O, const std::vector<uint64_t>& g, int64_t t, uint64_t r) {
    const uint64_t q = O.q, u = 1 + O.ell;
    const int64_t ord = O.order;
    const int64_t tt = ((t % ord) + ord) % ord;
    const uint64_t ur = PowMod(u, r, q);
    std::vector<u128> acc(O.order, 0);
    uint64_t w = 1 % q;
    for (size_t i = 0; i < g.size(); ++i) {
        if (g[i] % q) {
            const uint32_t x = static_cast<uint32_t>((static_cast<u128>(tt) * i) % static_cast<u128>(ord));
            acc[x] = (acc[x] + static_cast<u128>(g[i] % q) * w) % q;
        }
        w = O.mulq(w, ur);
    }
    std::vector<uint64_t> d(O.order);
    for (uint32_t k = 0; k < O.order; ++k) d[k] = static_cast<uint64_t>(acc[k]);
    return O.reduce(std::move(d));
}

// ---- (P4x) the 128-bit twin ------------------------------------------------------------------------------
OArith128 MakeOArith128(uint64_t ell, uint32_t m1, uint32_t P) {
    ENGINE_CHECK_MSG(ell >= 5 && m1 >= 1 && P >= 2, "wild::MakeOArith128: ell >= 5, m1 >= 1, P >= 2");
    // the exact accumulators: a product sum of `order` (O.mul) or ell - 1 (the power sums) terms < q^2 must stay below q 2^128
    ENGINE_CHECK_MSG(static_cast<double>(P + m1 + 1) * std::log2(static_cast<double>(ell)) < 126.0, "wild::MakeOArith128: ell^{P + m1 + 1} >= 2^126");
    OArith128 O;
    O.ell = ell;
    O.m1 = m1;
    O.P = P;
    O.ctx = engine::ModCtx<u128>::prime_power(ell, P);
    O.q = O.ctx.p;
    O.order = static_cast<uint32_t>(pow_wrap(ell, m1));
    O.s = static_cast<uint32_t>(pow_wrap(ell, m1 - 1));
    O.e = static_cast<uint32_t>((ell - 1) * O.s);
    return O;
}

OArith128::Elt OArith128::zeta_pow(uint64_t ex) const {
    std::vector<u128> d(order, 0);
    d[ex % order] = 1;
    return reduce(std::move(d));
}

OArith128::Elt OArith128::reduce(std::vector<u128> d) const {
    ENGINE_CHECK_MSG(d.size() == order, "wild::OArith128::reduce: dense length must be ell^{m1}");
    for (uint32_t r = 0; r < s; ++r) {
        const u128 c = d[e + r] % q;
        if (c == 0) continue;
        for (uint64_t j = 0; j + 1 < ell; ++j) {
            u128& y = d[j * s + r];
            y = ctx.sub(y % q, c);
        }
    }
    Elt out(e);
    for (uint32_t x = 0; x < e; ++x) out[x] = d[x] % q;
    return out;
}

OArith128::Elt OArith128::add(const Elt& a, const Elt& b) const {
    Elt c(e);
    for (uint32_t i = 0; i < e; ++i) c[i] = ctx.add(a[i], b[i]);
    return c;
}

OArith128::Elt OArith128::scal(u128 c, const Elt& a) const {
    Elt out(e);
    for (uint32_t i = 0; i < e; ++i) out[i] = mulq(c, a[i]);
    return out;
}

OArith128::Elt OArith128::mul(const Elt& a, const Elt& b) const {
    std::vector<engine::u256> acc(order);
    for (uint32_t i = 0; i < e; ++i) {
        if (a[i] == 0) continue;
        for (uint32_t j = 0; j < e; ++j) {
            if (b[j] == 0) continue;
            acc[(i + j) % order] += engine::detail::mul_wide128(a[i], b[j]);
        }
    }
    std::vector<u128> d(order);
    for (uint32_t x = 0; x < order; ++x) d[x] = ctx.reduce(acc[x]);
    return reduce(std::move(d));
}

u128 OArith128::pow(u128 b, u128 ex) const {
    u128 r = 1 % q, x = b % q;
    while (ex) {
        if (ex & 1) r = ctx.mul(r, x);
        x = ctx.mul(x, x);
        ex >>= 1;
    }
    return r;
}

namespace {
// v_ell(x) and the unit part mod O.q (x != 0)
uint32_t ValUnit128(const nf::BigInt& x, const OArith128& O, u128& unit) {
    ENGINE_CHECK_MSG(!x.is_zero(), "wild::ValUnit128: zero has no valuation");
    nf::BigInt y = x;
    uint32_t v = 0;
    while (y.mod_u32(static_cast<uint32_t>(O.ell)) == 0) {
        y = y.div_exact_u32(static_cast<uint32_t>(O.ell));
        ++v;
    }
    u128 acc = 0;
    const u128 ten = 10 % O.q;
    for (char ch : y.to_string()) acc = O.ctx.add(O.ctx.mul(acc, ten), static_cast<u128>(ch - '0') % O.q);
    unit = acc;
    return v;
}
} // namespace

std::vector<std::vector<u128>> OmegaTable128(const OArith128& O) {
    const uint64_t ell = O.ell;
    const u128 ex = O.ctx.ell_pow(O.P - 1);
    std::vector<std::vector<u128>> om(ell - 1, std::vector<u128>(ell, 0));
    for (uint64_t r = 1; r < ell; ++r) {
        const u128 w = O.pow(r, ex);   // the Teichmuller lift of r mod ell^P
        om[0][r] = 1;
        for (uint32_t x = 1; x + 1 < ell; ++x) om[x][r] = O.ctx.mul(om[x - 1][r], w);
    }
    return om;
}

namespace {
std::vector<u128> EllBernoulli128(uint32_t smax, const OArith128& O) {
    const u128 q = O.q, ell = O.ell;
    std::vector<u128> out(smax + 1, 0);
    out[0] = ell % q;
    if (smax >= 1) out[1] = O.ctx.mul(O.ctx.neg(ell % q), O.ctx.inv(2));   // ell B_1 = -ell/2
    for (uint32_t s = 2; s <= smax; s += 2) {
        const BernoulliOverK b = ExactBernoulliOverK(s);                    // B_s / s = sign num / den
        u128 un = 0, ud = 0, us = 0;
        const uint32_t vn = ValUnit128(b.num, O, un), vd = ValUnit128(b.den, O, ud), vs = ValUnit128(nf::BigInt(s), O, us);
        ENGINE_CHECK_MSG(vn + vs + 1 >= vd, "wild::EllBernoulli128: v(B_s) < -1 contradicts von Staudt-Clausen");
        const uint32_t v = vn + vs + 1 - vd;
        if (v >= O.P) continue;
        u128 x = O.ctx.mul(O.ctx.mul(un, us), O.ctx.inv(ud));
        if (b.sign < 0) x = O.ctx.neg(x);
        out[s] = O.ctx.mul(x, O.ctx.ell_pow(v));
    }
    return out;
}
} // namespace

Lad128 Residue128(uint32_t k, const OArith128& O) {
    const u128 ell = O.ell;
    // log_ell(u)/ell = sum_{m>=1} (-1)^{m+1} ell^{m-1} / m, to ell^P
    u128 lg = 0;
    for (uint32_t m = 1; m < 8 * (O.P + 4) + 8; ++m) {
        const uint32_t vm = ValU(m, O.ell);
        const int32_t e = static_cast<int32_t>(m) - 1 - static_cast<int32_t>(vm);
        if (e >= static_cast<int32_t>(O.P)) continue;
        u128 t = O.ctx.mul(O.ctx.ell_pow(static_cast<uint32_t>(e)), O.ctx.inv(m / pow_wrap(O.ell, vm)));
        if (m % 2 == 0) t = O.ctx.neg(t);
        lg = O.ctx.add(lg, t);
    }
    const BernoulliOverK bk = ExactBernoulliOverK(k);
    u128 un = 0, ud = 0;
    const uint32_t vn = ValUnit128(bk.num, O, un), vd = ValUnit128(bk.den, O, ud);
    Lad128 r;
    r.v = static_cast<int32_t>(vn) - static_cast<int32_t>(vd);
    u128 x = O.ctx.mul(un, O.ctx.inv(ud));
    if (bk.sign < 0) x = O.ctx.neg(x);
    x = O.ctx.mul(x, O.ctx.sub(1, ell % O.q));
    x = O.ctx.mul(x, lg);
    x = O.ctx.mul(x, O.ctx.sub(1, O.pow(ell, k - 1)));
    r.unit = x;
    return r;
}

WildBernoulliFB128::WildBernoulliFB128(const OArith128& O, uint32_t rmax, const std::vector<uint32_t>& logs) : O_(O), m_(O.m1 + 1), rmax_(rmax), logs_(logs) {
    F_ = pow_wrap(O.ell, m_);
    ENGINE_CHECK_MSG(logs_.size() == F_, "wild::WildBernoulliFB128: log table of the wrong size");
    // F^s B_s = ell^{m s - 1} (ell B_s) vanishes mod ell^P once m s - 1 >= P
    const uint32_t smax = std::min(rmax, O.P / m_);
    ellB_ = EllBernoulli128(smax, O);
    FB_.assign(smax + 1, 0);
    FB_[0] = 1;
    for (uint32_t s = 1; s <= smax; ++s) FB_[s] = O.ctx.mul(ellB_[s], O.ctx.ell_pow(m_ * s - 1));
    om_ = OmegaTable128(O);
    const uint32_t R1 = rmax + 1;
    apow_.assign(F_ * R1, 0);
    for (uint64_t a = 1; a < F_; ++a) {
        if (a % O.ell == 0) continue;
        u128 p = 1;
        for (uint32_t r = 0; r <= rmax; ++r) {
            apow_[a * R1 + r] = p;
            p = O.ctx.mul(p, a % O.q);
        }
    }
}

const std::vector<u128>& WildBernoulliFB128::sums(uint32_t i) {
    auto it = S_.find(i);
    if (it != S_.end()) return it->second;
    const uint32_t R1 = rmax_ + 1;
    std::vector<engine::u256> acc(static_cast<size_t>(O_.order) * R1);
    for (uint64_t a = 1; a < F_; ++a) {
        if (a % O_.ell == 0) continue;
        const u128 w = om_[i][a % O_.ell];
        engine::u256* row = &acc[static_cast<size_t>(logs_[a] % O_.order) * R1];
        const u128* pw = &apow_[a * R1];
        for (uint32_t r = 0; r <= rmax_; ++r) row[r] += engine::detail::mul_wide128(w, pw[r]);
    }
    std::vector<u128> S(acc.size());
    for (size_t x = 0; x < acc.size(); ++x) S[x] = O_.ctx.reduce(acc[x]);
    return S_.emplace(i, std::move(S)).first->second;
}

OArith128::Elt WildBernoulliFB128::at(int64_t i, int64_t t, uint32_t r) {
    ENGINE_CHECK_MSG(r >= 1 && r <= rmax_, "wild::WildBernoulliFB128::at: 1 <= r <= rmax");
    const int64_t L1 = static_cast<int64_t>(O_.ell - 1), ord = O_.order;
    const std::vector<u128>& S = sums(static_cast<uint32_t>(((i % L1) + L1) % L1));
    const uint64_t tt = static_cast<uint64_t>(((t % ord) + ord) % ord);
    const uint32_t R1 = rmax_ + 1;
    std::vector<u128> dense(O_.order, 0);
    u128 binom = 1;   // C(r, s), exact (r a few hundred, s <= P / m)
    for (uint32_t s = 0; s < FB_.size() && s <= r; ++s) {
        if (s > 0) binom = binom * (r - s + 1) / s;
        if (FB_[s] == 0) continue;
        const u128 c = O_.ctx.mul(binom % O_.q, FB_[s]);
        for (uint32_t y = 0; y < O_.order; ++y) {
            const u128 v = S[static_cast<size_t>(y) * R1 + (r - s)];
            if (v == 0) continue;
            u128& z = dense[(tt * y) % O_.order];
            z = O_.ctx.add(z, O_.ctx.mul(c, v));
        }
    }
    return O_.reduce(std::move(dense));
}

OArith128::Elt GroupRingAtWildPoint128(const OArith128& O, const std::vector<u128>& g, int64_t t, uint64_t r) {
    const int64_t ord = O.order;
    const uint64_t tt = static_cast<uint64_t>(((t % ord) + ord) % ord);
    const u128 ur = O.pow(1 + O.ell, r);
    std::vector<u128> d(O.order, 0);
    u128 w = 1;
    for (size_t i = 0; i < g.size(); ++i) {
        if (g[i] % O.q) {
            u128& z = d[(tt * i) % O.order];
            z = O.ctx.add(z, O.ctx.mul(g[i] % O.q, w));
        }
        w = O.ctx.mul(w, ur);
    }
    return O.reduce(std::move(d));
}

// ---- the complex side: exact objects ---------------------------------------------------------------------
namespace {
using i128 = __int128;

nf::BigInt FromI128(i128 v) {
    const bool neg = v < 0;
    unsigned __int128 a = neg ? static_cast<unsigned __int128>(-v) : static_cast<unsigned __int128>(v);
    const uint64_t hi = static_cast<uint64_t>(a >> 64), lo = static_cast<uint64_t>(a);
    nf::BigInt r = nf::BigInt::from_u64(hi);
    r *= nf::BigInt::from_u64(uint64_t(1) << 32);
    r *= nf::BigInt::from_u64(uint64_t(1) << 32);
    r += nf::BigInt::from_u64(lo);
    return neg ? -r : r;
}

int Bits(i128 v) {
    unsigned __int128 a = v < 0 ? static_cast<unsigned __int128>(-v) : static_cast<unsigned __int128>(v);
    int b = 0;
    while (a) {
        a >>= 1;
        ++b;
    }
    return b;
}
} // namespace

std::vector<nf::BigInt> CuspQExp(uint32_t k, uint32_t mmax) {
    ENGINE_CHECK_MSG(k == 12 || k == 16, "wild::CuspQExp: k = 12 (Delta) or 16 (Delta E_4)");
    const uint32_t L = mmax;   // coefficients of prod (1 - q^n)^24 to q^{L-1}
    // eta^3 / q^{1/8} = sum_{j >= 0} (-1)^j (2j + 1) q^{j(j+1)/2}  (Jacobi), sparse
    std::vector<std::pair<uint32_t, int64_t>> e3;
    for (uint64_t j = 0;; ++j) {
        const uint64_t ex = j * (j + 1) / 2;
        if (ex >= L) break;
        e3.emplace_back(static_cast<uint32_t>(ex), (j % 2 ? -1 : 1) * static_cast<int64_t>(2 * j + 1));
    }
    std::vector<i128> c(L, 0);
    for (const auto& pe : e3) c[pe.first] = pe.second;
    for (int step = 1; step < 8; ++step) {
        int mb = 0;
        for (i128 x : c) mb = std::max(mb, Bits(x));
        ENGINE_CHECK_MSG(mb <= 100, "wild::CuspQExp: an intermediate coefficient exceeds 2^100");
        std::vector<i128> d(L, 0);
        for (uint32_t i = 0; i < L; ++i) {
            if (c[i] == 0) continue;
            for (const auto& pe : e3) {
                if (i + pe.first >= L) break;
                d[i + pe.first] += c[i] * pe.second;
            }
        }
        c.swap(d);
    }
    std::vector<i128> f(mmax + 1, 0);   // Delta: a_m = c_{m-1}
    for (uint32_t m = 1; m <= mmax; ++m) f[m] = c[m - 1];
    if (k == 16) {
        std::vector<i128> e4(mmax + 1, 0);
        e4[0] = 1;
        for (uint32_t m = 1; m <= mmax; ++m) {
            i128 s3 = 0;
            for (uint32_t dv = 1; dv * dv <= m; ++dv)
                if (m % dv == 0) {
                    s3 += static_cast<i128>(dv) * dv * dv;
                    const uint32_t d2 = m / dv;
                    if (d2 != dv) s3 += static_cast<i128>(d2) * d2 * d2;
                }
            e4[m] = 240 * s3;
        }
        int bf = 0, be = 0, bl = 0;
        for (i128 x : f) bf = std::max(bf, Bits(x));
        for (i128 x : e4) be = std::max(be, Bits(x));
        for (uint32_t x = mmax; x; x >>= 1) ++bl;
        ENGINE_CHECK_MSG(bf + be + bl <= 125, "wild::CuspQExp: Delta E_4 would overflow int128 at this length");
        std::vector<i128> g(mmax + 1, 0);
        for (uint32_t m = 1; m <= mmax; ++m) {
            i128 s = 0;
            for (uint32_t i = 1; i <= m; ++i) s += f[i] * e4[m - i];
            g[m] = s;
        }
        f.swap(g);
    }
    std::vector<nf::BigInt> out(mmax + 1);
    for (uint32_t m = 0; m <= mmax; ++m) out[m] = FromI128(f[m]);
    return out;
}

namespace {
std::vector<nf::BigInt> ExactRho(const birch::Mat2& A, uint32_t n) {
    const uint32_t N1 = n + 1;
    std::vector<nf::BigInt> M(static_cast<size_t>(N1) * N1, nf::BigInt(0));
    // binomials and powers
    std::vector<std::vector<nf::BigInt>> C(N1, std::vector<nf::BigInt>(N1, nf::BigInt(0)));
    for (uint32_t i = 0; i <= n; ++i) {
        C[i][0] = 1;
        for (uint32_t j = 1; j <= i; ++j) C[i][j] = C[i - 1][j - 1] + (j < i ? C[i - 1][j] : nf::BigInt(0));
    }
    auto pows = [&](int64_t x) {
        std::vector<nf::BigInt> p(N1, nf::BigInt(1));
        for (uint32_t i = 1; i <= n; ++i) p[i] = p[i - 1] * nf::BigInt(x);
        return p;
    };
    const std::vector<nf::BigInt> pa = pows(A.a), pb = pows(A.b), pc = pows(A.c), pd = pows(A.d);
    for (uint32_t j = 0; j <= n; ++j) {
        // (a + c y)^{n-j} (b + d y)^j, coefficient of y^t -> row t
        const uint32_t nj = n - j;
        for (uint32_t s = 0; s <= nj; ++s) {
            const nf::BigInt ps = C[nj][s] * pa[nj - s] * pc[s];
            if (ps.is_zero()) continue;
            for (uint32_t r = 0; r <= j; ++r) {
                const nf::BigInt qr = C[j][r] * pb[j - r] * pd[r];
                if (qr.is_zero()) continue;
                M[static_cast<size_t>(s + r) * N1 + j] += ps * qr;
            }
        }
    }
    return M;
}

std::vector<nf::BigInt> ExactApply(const birch::Mat2& A, uint32_t n, const std::vector<nf::BigInt>& v) {
    const std::vector<nf::BigInt> M = ExactRho(A, n);
    const uint32_t N1 = n + 1;
    std::vector<nf::BigInt> out(N1, nf::BigInt(0));
    for (uint32_t r = 0; r < N1; ++r)
        for (uint32_t c = 0; c < N1; ++c)
            if (!v[c].is_zero() && !M[static_cast<size_t>(r) * N1 + c].is_zero()) out[r] += M[static_cast<size_t>(r) * N1 + c] * v[c];
    return out;
}
} // namespace

std::vector<nf::BigInt> ExactTwistVector(const std::vector<nf::BigInt>& P, uint32_t n, int64_t a, int64_t M) {
    std::vector<nf::BigInt> phi(n + 1, nf::BigInt(0));
    for (const birch::Mat2& g : birch::ManinPath(a, M)) {
        const std::vector<nf::BigInt> x = ExactApply(g, n, P);
        for (uint32_t t = 0; t <= n; ++t) phi[t] -= x[t];
    }
    birch::Mat2 gm;
    gm.a = M;
    gm.b = -a;
    gm.c = 0;
    gm.d = 1;
    return ExactApply(gm, n, phi);
}

ExactEigensymbols ComputeExactEigensymbols(uint32_t k, const nf::BigInt& a2) {
    const uint32_t n = k - 2, N1 = n + 1;
    using Q = nf::BigRational;
    std::vector<std::vector<Q>> rows;
    auto addMat = [&](const std::vector<nf::BigInt>& Mx) {
        for (uint32_t r = 0; r < N1; ++r) {
            std::vector<Q> row(N1);
            for (uint32_t c = 0; c < N1; ++c) row[c] = Q(Mx[static_cast<size_t>(r) * N1 + c]);
            rows.push_back(row);
        }
    };
    auto matmul = [&](const std::vector<nf::BigInt>& X, const std::vector<nf::BigInt>& Y) {
        std::vector<nf::BigInt> Z(static_cast<size_t>(N1) * N1, nf::BigInt(0));
        for (uint32_t i = 0; i < N1; ++i)
            for (uint32_t l = 0; l < N1; ++l) {
                const nf::BigInt& x = X[static_cast<size_t>(i) * N1 + l];
                if (x.is_zero()) continue;
                for (uint32_t j = 0; j < N1; ++j) Z[static_cast<size_t>(i) * N1 + j] += x * Y[static_cast<size_t>(l) * N1 + j];
            }
        return Z;
    };
    auto addId = [&](std::vector<nf::BigInt> X, const nf::BigInt& c) {
        for (uint32_t i = 0; i < N1; ++i) X[static_cast<size_t>(i) * N1 + i] += c;
        return X;
    };
    birch::Mat2 S, U;
    S.a = 0, S.b = -1, S.c = 1, S.d = 0;
    U.a = 0, U.b = -1, U.c = 1, U.d = 1;
    const std::vector<nf::BigInt> rS = ExactRho(S, n), rU = ExactRho(U, n);
    addMat(addId(rS, nf::BigInt(1)));
    std::vector<nf::BigInt> rU2 = matmul(rU, rU), RU(static_cast<size_t>(N1) * N1);
    for (size_t i = 0; i < RU.size(); ++i) RU[i] = rU[i] + rU2[i];
    addMat(addId(RU, nf::BigInt(1)));
    // T_2 = rho(diag(1,2)) + sum_{b < 2} rho([[2,-b],[0,1]]) Phi({b/2, oo}),  Phi({b/2, oo}) = -sum_k rho(g_k) P
    birch::Mat2 D2;
    D2.a = 1, D2.b = 0, D2.c = 0, D2.d = 2;
    std::vector<nf::BigInt> T2 = ExactRho(D2, n);
    for (int64_t b = 0; b < 2; ++b) {
        birch::Mat2 gb;
        gb.a = 2, gb.b = -b, gb.c = 0, gb.d = 1;
        const std::vector<nf::BigInt> rg = ExactRho(gb, n);
        std::vector<nf::BigInt> sum(static_cast<size_t>(N1) * N1, nf::BigInt(0));
        for (const birch::Mat2& g : birch::ManinPath(b, 2)) {
            const std::vector<nf::BigInt> x = ExactRho(g, n);
            for (size_t i = 0; i < sum.size(); ++i) sum[i] += x[i];
        }
        const std::vector<nf::BigInt> prod = matmul(rg, sum);
        for (size_t i = 0; i < T2.size(); ++i) T2[i] -= prod[i];
    }
    addMat(addId(T2, -a2));
    // RREF over Q
    const size_t R = rows.size();
    std::vector<int> pivcol;
    size_t r = 0;
    for (uint32_t c = 0; c < N1 && r < R; ++c) {
        size_t p = r;
        while (p < R && rows[p][c].is_zero()) ++p;
        if (p == R) continue;
        std::swap(rows[p], rows[r]);
        const Q inv = Q(1) / rows[r][c];
        for (uint32_t j = 0; j < N1; ++j) rows[r][j] *= inv;
        for (size_t i = 0; i < R; ++i) {
            if (i == r || rows[i][c].is_zero()) continue;
            const Q f = rows[i][c];
            for (uint32_t j = 0; j < N1; ++j)
                if (!rows[r][j].is_zero()) rows[i][j] -= f * rows[r][j];
        }
        pivcol.push_back(static_cast<int>(c));
        ++r;
    }
    std::vector<char> isPiv(N1, 0);
    for (int c : pivcol) isPiv[c] = 1;
    std::vector<std::vector<Q>> ker;
    for (uint32_t fcol = 0; fcol < N1; ++fcol) {
        if (isPiv[fcol]) continue;
        std::vector<Q> v(N1, Q(0));
        v[fcol] = Q(1);
        for (size_t i = 0; i < pivcol.size(); ++i) v[pivcol[i]] = -rows[i][fcol];
        ker.push_back(v);
    }
    ExactEigensymbols out;
    out.kernel_dim = static_cast<uint32_t>(ker.size());
    auto primitive = [&](const std::vector<Q>& v) {
        nf::BigInt den(1);
        for (const Q& x : v) {
            const nf::BigInt g = nf::BigInt::gcd(den, x.den());
            den = den / g * x.den();
        }
        std::vector<nf::BigInt> iv(N1);
        nf::BigInt g(0);
        for (uint32_t t = 0; t < N1; ++t) {
            iv[t] = v[t].num() * (den / v[t].den());
            g = nf::BigInt::gcd(g, iv[t]);
        }
        int sgn = 0;
        for (uint32_t t = 0; t < N1 && sgn == 0; ++t) sgn = iv[t].sign();
        for (uint32_t t = 0; t < N1; ++t) {
            iv[t] = iv[t] / g;
            if (sgn < 0) iv[t] = -iv[t];
        }
        return iv;
    };
    for (const auto& v : ker) {
        std::vector<Q> ev(N1, Q(0)), od(N1, Q(0));
        bool nzE = false, nzO = false;
        for (uint32_t t = 0; t < N1; ++t) {
            if (t % 2 == 0) {
                ev[t] = v[t];
                nzE = nzE || !v[t].is_zero();
            } else {
                od[t] = v[t];
                nzO = nzO || !v[t].is_zero();
            }
        }
        if (nzE && out.Pplus.empty()) out.Pplus = primitive(ev);
        if (nzO && out.Pminus.empty()) out.Pminus = primitive(od);
    }
    return out;
}

} // namespace wild
} // namespace lvalue
} // namespace hecke_ext
