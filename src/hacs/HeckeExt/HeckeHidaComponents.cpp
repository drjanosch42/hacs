// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 15, WP-6) HeckeHidaComponents -- see the header.
#include "HeckeHidaComponents.h"

#include "HeckeHidaTower.h"

#include "../Hecke/CoefficientRing.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>

namespace hecke_ext {
namespace hidacomp {

using hecke::ordinary::RingMat;

namespace {

constexpr u128 kU128Half = u128(1) << 127;

// a * b mod m, a, b < m, m < 2^127.
inline u128 MulMod(u128 a, u128 b, u128 m) {
    if ((m >> 64) == 0) return (a * b) % m;   // a, b < 2^64: the product fits
    u128 r = 0;
    a %= m;
    while (b) {
        if (b & 1) {
            r += a;
            if (r >= m) r -= m;
        }
        a += a;
        if (a >= m) a -= m;
        b >>= 1;
    }
    return r;
}

inline u128 AddMod(u128 a, u128 b, u128 m) {
    u128 r = a + b;   // a, b < m < 2^127
    return r >= m ? r - m : r;
}

inline u128 SubMod(u128 a, u128 b, u128 m) { return a >= b ? a - b : a + (m - b); }

u128 PowModU(u128 b, uint64_t e, u128 m) {
    u128 r = 1 % m;
    b %= m;
    while (e) {
        if (e & 1) r = MulMod(r, b, m);
        b = MulMod(b, b, m);
        e >>= 1;
    }
    return r;
}

// the inverse of a unit mod ell^e (m = ell^e): mod ell by Fermat, then Newton.
u128 InvUnit(u128 a, uint64_t ell, u128 m) {
    a %= m;
    ENGINE_CHECK_MSG(a % ell != 0, "HeckeHidaComponents: InvUnit of a non-unit");
    u128 x = PowModU(a % ell, ell - 2, ell);   // mod ell
    for (int it = 0; it < 8; ++it) {           // precision doubles: 2^8 digits >= 127 bits
        const u128 ax = MulMod(a, x, m);
        x = MulMod(x, SubMod(2 % m, ax, m), m);
    }
    ENGINE_CHECK_MSG(MulMod(a, x, m) == 1 % m, "HeckeHidaComponents: InvUnit failed");
    return x;
}

// (unit part, valuation) of x > 0.
inline std::pair<uint64_t, uint32_t> Split(uint64_t x, uint64_t ell) {
    uint32_t v = 0;
    while (x % ell == 0) {
        x /= ell;
        ++v;
    }
    return {x, v};
}

uint32_t FloorLog(uint64_t x, uint64_t ell) {   // floor(log_ell x), x >= 1
    uint32_t r = 0;
    while (x >= ell) {
        x /= ell;
        ++r;
    }
    return r;
}

void CheckPrimeEll(uint64_t ell, const char* who) {
    ENGINE_CHECK_MSG(ell >= 5 && (ell & 1), (std::string(who) + ": needs an odd prime ell >= 5").c_str());
    ENGINE_CHECK_MSG(hecke::IsPrimeU64(ell), (std::string(who) + ": ell is not prime").c_str());
}

using engine::scalar::powfit;

[[noreturn]] void Refuse(const std::string& what) {
    throw std::runtime_error("HeckeHidaComponents: " + what + " refused");
}


u128 PowU128Fit(uint64_t ell, uint32_t e) {
    u128 r = 1;
    for (uint32_t i = 0; i < e; ++i) {
        if (r >= kU128Half / ell) return 0;
        r *= ell;
    }
    return r;
}
} // namespace

uint32_t ValU64(uint64_t x, uint64_t ell) {
    ENGINE_CHECK_MSG(x != 0, "HeckeHidaComponents: valuation of 0");
    return Split(x, ell).second;
}

uint32_t ValCapped(uint64_t x, uint64_t ell, uint32_t cap) {
    uint32_t v = 0;
    while (v < cap && x % ell == 0) {
        if (x == 0) return cap;
        x /= ell;
        ++v;
    }
    return v;
}

// ---- the direct route ---------------------------------------------------------------

namespace {
// the first t0 >= 1 with t - floor(log_ell(t + 1)) >= Q (non-decreasing in t)
uint64_t FirstVanishingT(uint64_t ell, uint32_t Q) {
    uint64_t t = 1;
    while (t - FloorLog(t + 1, ell) < Q) ++t;
    return t;
}
} // namespace

uint64_t BernoulliTable::PredictedWork(uint64_t ell, uint32_t Q, uint64_t W) {
    const u128 t0 = FirstVanishingT(ell, Q);
    // a MulMod is one multiplication while ell^Q < 2^64, and a double-and-add loop of up to 127
    // steps (~4 u128 ops each) above it: weight it by that cost, not by 1.
    const bool wide = PowU128Fit(ell, Q) == 0 || (PowU128Fit(ell, Q) >> 64) != 0;
    const u128 mulcost = wide ? 128 : 1;
    const u128 w = (u128(W) / 2 + 1) * (u128(ell) + 4 * t0) * mulcost;
    return w >> 64 ? ~uint64_t(0) : static_cast<uint64_t>(w);
}

BernoulliTable::BernoulliTable(uint64_t ell, uint32_t Q, uint64_t W, uint64_t budget) : ell_(ell), Q_(Q) {
    CheckPrimeEll(ell, "BernoulliTable");
    ENGINE_CHECK_MSG(Q >= 2 && W >= 2, "BernoulliTable: needs Q >= 2, W >= 2");
    const u128 M = PowU128Fit(ell, Q);
    if (M == 0) Refuse("BernoulliTable: ell^Q >= 2^127");
    // the beta_ vector is (W + 1) u128 = 16 (W + 1) bytes, allocated before any work: cap it.
    if (W >= kMaxBetaBytes / 16)
        Refuse("BernoulliTable: table memory " + std::to_string(static_cast<uint64_t>((u128(W) + 1) * 16 / (1 << 20))) +
               " MiB > cap " + std::to_string(kMaxBetaBytes >> 20) + " MiB");
    const uint64_t work = PredictedWork(ell, Q, W);
    if (work > budget)
        Refuse("BernoulliTable: work " + std::to_string(work) + " > budget " + std::to_string(budget));
    const uint64_t t0 = FirstVanishingT(ell, Q);
    // inverses of the unit parts of 1 .. t0 + 1, and the valuations
    std::vector<u128> inv_unit(t0 + 2, 0);
    std::vector<uint32_t> vt(t0 + 2, 0);
    for (uint64_t t = 1; t <= t0 + 1; ++t) {
        const auto s = Split(t, ell);
        vt[t] = s.second;
        inv_unit[t] = InvUnit(s.first, ell, M);
    }
    std::vector<u128> ellpow(Q + 1, 1);
    for (uint32_t i = 1; i <= Q; ++i) ellpow[i] = ellpow[i - 1] * ell;
    beta_.assign(W + 1, 0);
    beta_[0] = ell % M;
    beta_[1] = SubMod(0, MulMod(ell % M, (M + 1) / 2, M), M);   // ell * (-1/2)
    // r^m for r in [1, ell), stepped by r^2 over even m
    std::vector<u128> pw(ell), sq(ell);
    for (uint64_t r = 1; r < ell; ++r) {
        sq[r] = MulMod(r, r, M);
        pw[r] = sq[r];   // m = 2
    }
    for (uint64_t m = 2; m <= W; m += 2) {
        if (m > 2)
            for (uint64_t r = 1; r < ell; ++r) pw[r] = MulMod(pw[r], sq[r], M);
        u128 s = 0;
        for (uint64_t r = 1; r < ell; ++r) s = AddMod(s, pw[r], M);
        // subtract sum_{t >= 1} C(m, t) beta_{m-t} ell^{t - v(t+1)} / unit(t+1)
        u128 cu = 1;       // unit part of C(m, t)
        uint32_t cv = 0;   // valuation of C(m, t)
        const uint64_t tmax = std::min<uint64_t>(m, t0 - 1);
        for (uint64_t t = 1; t <= tmax; ++t) {
            const auto nf = Split(m - t + 1, ell);
            cu = MulMod(cu, nf.first % M, M);
            cu = MulMod(cu, inv_unit[t], M);
            cv = cv + nf.second - vt[t];
            const uint64_t j = m - t;
            if (j >= 3 && (j & 1)) continue;
            const uint64_t vtot = uint64_t(cv) + t - vt[t + 1];
            if (vtot >= Q) continue;
            u128 coef = MulMod(cu, inv_unit[t + 1], M);
            coef = MulMod(coef, ellpow[vtot] % M, M);
            s = SubMod(s, MulMod(coef, beta_[j], M), M);
        }
        beta_[m] = s;
    }
}

u128 BernoulliTable::beta(uint64_t m) const {
    ENGINE_CHECK_MSG(m < beta_.size(), "BernoulliTable: index beyond the table");
    return beta_[m];
}

uint64_t BernoulliTable::BOverK(uint64_t w, uint32_t K) const {
    ENGINE_CHECK_MSG(w >= 2 && (w & 1) == 0 && w < beta_.size(), "BernoulliTable::BOverK: w even, 2 <= w <= W");
    ENGINE_CHECK_MSG(w % (ell_ - 1) != 0, "BernoulliTable::BOverK: (ell - 1) | w (B_w is not ell-integral)");
    const auto ws = Split(w, ell_);
    ENGINE_CHECK_MSG(K >= 1 && K + 1 + ws.second <= Q_, "BernoulliTable::BOverK: precision K + 1 + v(w) > Q");
    const u128 d = PowU128Fit(ell_, 1 + ws.second);
    const u128 b = beta_[w];
    // B_w / w is ell-integral: beta_w = ell B_w is divisible by ell^{1 + v(w)} (checked, a certificate of the table)
    ENGINE_CHECK_MSG(b % d == 0, "BernoulliTable::BOverK: ell^{1+v(w)} does not divide ell B_w -- the table is wrong");
    const u128 q = PowU128Fit(ell_, K);
    const u128 r = MulMod((b / d) % q, InvUnit(ws.first, ell_, q), q);
    return static_cast<uint64_t>(r);
}

uint64_t BernoulliOverKDirect(uint64_t w, uint64_t ell, uint32_t K, uint64_t budget) {
    CheckPrimeEll(ell, "BernoulliOverKDirect");
    if (powfit(ell, K) == 0) Refuse("BernoulliOverKDirect: ell^K >= 2^64");
    const BernoulliTable t(ell, K + 1 + ValU64(w, ell), w, budget);
    return t.BOverK(w, K);
}

// ---- the Kummer-Mahler route ---------------------------------------------------------

namespace {
// binom(x, i) mod q = ell^K (q < 2^64)
uint64_t BinomMod(uint64_t x, uint32_t i, uint64_t ell, uint32_t K, uint64_t q) {
    if (i == 0) return 1 % q;
    if (x < i) return 0;
    u128 un = 1, ud = 1;
    int64_t v = 0;
    for (uint32_t s = 0; s < i; ++s) {
        const auto a = Split(x - s, ell);
        const auto b = Split(s + 1, ell);
        un = MulMod(un, a.first % q, q);
        ud = MulMod(ud, b.first % q, q);
        v += int64_t(a.second) - int64_t(b.second);
    }
    ENGINE_CHECK_MSG(v >= 0, "HeckeHidaComponents: binomial of negative valuation");
    if (v >= K) return 0;
    u128 r = MulMod(un, InvUnit(ud, ell, q), q);
    for (int64_t j = 0; j < v; ++j) r = MulMod(r, ell % q, q);
    return static_cast<uint64_t>(r);
}
} // namespace

uint64_t ClassSeries::Phi(uint64_t x) const {
    const uint64_t q = powfit(ell, K);
    u128 s = 0;
    for (uint32_t i = 0; i < K; ++i) {
        if (c[i] == 0) continue;
        s = AddMod(s, MulMod(c[i], BinomMod(x, i, ell, K, q), q), q);
    }
    return static_cast<uint64_t>(s);
}

ClassSeries MakeClassSeries(uint64_t ell, uint64_t k0, uint32_t K, uint64_t budget) {
    CheckPrimeEll(ell, "MakeClassSeries");
    ENGINE_CHECK_MSG(K >= 1 && k0 >= 2 && k0 <= ell - 3 && (k0 & 1) == 0,
                     "MakeClassSeries: needs K >= 1 and an even class 2 <= k0 <= ell - 3");
    const uint64_t q = powfit(ell, K);
    if (q == 0) Refuse("MakeClassSeries: ell^K >= 2^64");
    uint32_t vmax = 0;
    std::vector<uint64_t> kj(K);
    for (uint32_t j = 0; j < K; ++j) {
        const u128 w = u128(k0) + u128(ell - 1) * j;
        if (w >> 63) Refuse("MakeClassSeries: weight beyond 2^63");
        kj[j] = static_cast<uint64_t>(w);
        vmax = std::max(vmax, ValU64(kj[j], ell));
    }
    const BernoulliTable T(ell, K + 1 + vmax, std::max<uint64_t>(kj.back(), 2), budget);
    std::vector<u128> phi(K);
    for (uint32_t j = 0; j < K; ++j) {
        const u128 bk = T.BOverK(kj[j], K);
        const u128 lp = (kj[j] - 1 >= K) ? 0 : PowU128Fit(ell, static_cast<uint32_t>(kj[j] - 1)) % q;
        // phi = -(1 - ell^{k-1}) B_k / k
        phi[j] = SubMod(0, MulMod(SubMod(1 % q, lp, q), bk, q), q);
    }
    ClassSeries s;
    s.ell = ell;
    s.K = K;
    s.k0 = k0;
    s.c.assign(K, 0);
    for (uint32_t i = 0; i < K; ++i) {
        u128 acc = 0;
        u128 binom = 1;   // C(i, j), exact (i < 64 is guaranteed: ell^K < 2^64, ell >= 5 => K <= 27)
        for (uint32_t j = 0; j <= i; ++j) {
            if (j > 0) binom = binom * (i - j + 1) / j;
            const u128 term = MulMod(binom % q, phi[j], q);
            acc = ((i - j) & 1) ? SubMod(acc, term, q) : AddMod(acc, term, q);
        }
        s.c[i] = static_cast<uint64_t>(acc);
        // SANITY CHECK, not a proof: v(c_i) >= i for i < K is what the Iwasawa-function structure predicts
        // (Washington 1st ed. p. 122 + Thm 7.10; slot-16 WP-8), and a violation would expose a wrong table or a wrong class.
        // It does NOT certify the truncation of the Mahler series at i >= K (v(c_i) >= i >= K there),
        // which rests on the cited theorem alone; the direct-route cross-check (validation) is the evidence.
        const uint64_t qi = powfit(ell, std::min(i, K));
        ENGINE_CHECK_MSG(s.c[i] % qi == 0, "MakeClassSeries: v(c_i) < i -- the Mahler bound fails");
    }
    return s;
}

uint64_t BernoulliOverKFromSeries(const ClassSeries& s, uint64_t k) {
    ENGINE_CHECK_MSG(k >= 2 && (k & 1) == 0 && k % (s.ell - 1) == s.k0,
                     "BernoulliOverKFromSeries: k even and in the series' class");
    const uint64_t q = powfit(s.ell, s.K);
    const uint64_t x = (k - s.k0) / (s.ell - 1);
    const u128 phi = s.Phi(x);
    const u128 lp = (k - 1 >= s.K) ? 0 : PowU128Fit(s.ell, static_cast<uint32_t>(k - 1)) % q;
    const u128 den = SubMod(1 % q, lp, q);   // 1 - ell^{k-1}, a unit
    return static_cast<uint64_t>(SubMod(0, MulMod(phi, InvUnit(den, s.ell, q), q), q));
}

uint64_t BernoulliOverKModPower(uint64_t k, uint64_t ell, uint32_t K, uint64_t budget) {
    CheckPrimeEll(ell, "BernoulliOverKModPower");
    ENGINE_CHECK_MSG(k >= 2 && (k & 1) == 0, "BernoulliOverKModPower: k even, k >= 2");
    ENGINE_CHECK_MSG(k % (ell - 1) != 0, "BernoulliOverKModPower: (ell - 1) | k (B_k / k is not ell-integral)");
    if (powfit(ell, K) == 0) Refuse("BernoulliOverKModPower: ell^K >= 2^64");
    const ClassSeries s = MakeClassSeries(ell, k % (ell - 1), K, budget);
    return BernoulliOverKFromSeries(s, k);
}

CappedVal ValBernoulliOverK(uint64_t k, uint64_t ell, uint32_t cap) {
    const uint64_t r = BernoulliOverKModPower(k, ell, cap);
    CappedVal cv;
    cv.v = ValCapped(r, ell, cap);
    cv.lower_bound = cv.v >= cap;
    return cv;
}

std::vector<uint64_t> IrregularIndices(uint64_t ell, uint64_t budget) {
    CheckPrimeEll(ell, "IrregularIndices");
    std::vector<uint64_t> out;
    if (ell < 7) return out;   // no class 2 <= k0 <= ell - 3 at ell = 5 beyond k0 = 2 (B_2 = 1/6)
    const BernoulliTable T(ell, 2, ell - 3, budget);
    for (uint64_t k0 = 2; k0 + 3 <= ell; k0 += 2)
        if (T.BOverK(k0, 1) == 0) out.push_back(k0);
    return out;
}

KStar ComputeKStar(uint64_t ell, uint64_t k0, uint32_t prec) {
    ENGINE_CHECK_MSG(prec >= 2, "ComputeKStar: prec >= 2");
    const ClassSeries s = MakeClassSeries(ell, k0, prec);
    KStar r;
    r.ell = ell;
    r.k0 = k0;
    r.prec = prec;
    r.modulus = powfit(ell, prec - 1);
    r.irregular = (s.c[0] % ell == 0);
    if (!r.irregular) return r;
    uint64_t x = 0, step = 1;
    bool unique = true;
    for (uint32_t j = 0; j + 1 < prec; ++j) {
        const uint64_t qj = powfit(ell, j + 2);
        uint32_t found = 0;
        uint64_t y0 = 0;
        for (uint64_t y = 0; y < ell; ++y) {
            if (s.Phi(x + step * y) % qj == 0) {
                if (!found) y0 = y;
                ++found;
            }
        }
        if (found != 1) {
            unique = false;
            break;
        }
        x += step * y0;
        step *= ell;
    }
    r.lambda_one = unique;
    if (!unique) return r;
    r.x_star = x % r.modulus;
    r.k_star = static_cast<uint64_t>((u128(k0) + u128(ell - 1) * r.x_star) % r.modulus);
    return r;
}

DepthPred PredictedDepth(uint64_t k, uint64_t ell, uint32_t prec) {
    DepthPred d;
    const uint64_t k0 = k % (ell - 1);
    if (k0 == 0) {
        d.na = true;
        return d;
    }
    const KStar ks = ComputeKStar(ell, k0, prec);
    if (!ks.irregular) return d;   // L = 0
    ENGINE_CHECK_MSG(ks.lambda_one, "PredictedDepth: the class has no simple zero (lambda >= 2) -- a finding");
    const uint64_t diff = (k % ks.modulus + ks.modulus - ks.k_star) % ks.modulus;
    if (diff == 0) {
        d.L = prec;
        d.lower_bound = true;
        return d;
    }
    d.L = 1 + ValU64(diff, ell);
    return d;
}

// ---- polynomials ------------------------------------------------------------------

std::pair<Poly, uint64_t> SyntheticDivide(const Poly& f, uint64_t e, uint64_t q) {
    ENGINE_CHECK_MSG(f.size() >= 2, "SyntheticDivide: degree >= 1");
    const size_t d = f.size() - 1;
    Poly qt(d, 0);
    u128 carry = f[d] % q;   // leading
    for (size_t i = d; i-- > 0;) {
        qt[i] = static_cast<uint64_t>(carry);
        carry = AddMod(f[i] % q, MulMod(carry, e % q, q), q);
    }
    return {qt, static_cast<uint64_t>(carry)};
}

std::pair<Poly, bool> MonicSqrt(const Poly& f, uint64_t q) {
    ENGINE_CHECK_MSG(f.size() % 2 == 1 && f.back() % q == 1 % q, "MonicSqrt: a monic polynomial of even degree");
    const size_t g = (f.size() - 1) / 2;
    Poly r(g + 1, 0);
    r[g] = 1 % q;
    const u128 inv2 = (u128(q) + 1) / 2;
    for (size_t i = g; i-- > 0;) {
        u128 s = f[g + i] % q;
        for (size_t a = i + 1; a < g; ++a) {
            const size_t b = g + i - a;
            if (b <= i || b >= g) continue;
            s = SubMod(s, MulMod(r[a], r[b], q), q);
        }
        r[i] = static_cast<uint64_t>(MulMod(s, inv2, q));
    }
    Poly sq(2 * g + 1, 0);
    for (size_t a = 0; a <= g; ++a)
        for (size_t b = 0; b <= g; ++b) sq[a + b] = static_cast<uint64_t>(AddMod(sq[a + b], MulMod(r[a], r[b], q), q));
    bool ok = true;
    for (size_t i = 0; i < sq.size(); ++i) ok = ok && sq[i] == f[i] % q;
    return {r, ok};
}

// ---- the readings -------------------------------------------------------------------

namespace {

uint64_t EvalPoly(const Poly& f, uint64_t x, uint64_t q) {
    u128 s = 0;
    for (size_t i = f.size(); i-- > 0;) s = AddMod(MulMod(s, x % q, q), f[i] % q, q);
    return static_cast<uint64_t>(s);
}

const RingMat& OpOf(const ringalg::OpModule& N, uint64_t p) {
    for (size_t i = 0; i < N.primes.size(); ++i)
        if (N.primes[i] == p) return N.op[i];
    ENGINE_CHECK_MSG(false, "HeckeHidaComponents: prime not run");
    return N.op[0];
}

bool IsFree(const ringalg::OpModule& N, uint32_t K) {
    for (uint32_t e : N.exps)
        if (e != K) return false;
    return true;
}

std::string Law(const ComponentCell& c) {
    if (c.class_zero || c.pred.na) return "na";
    if (c.eis.kind != "exact" && c.eis.kind != "upper_bound") return "no_data";
    if (c.pred.lower_bound) return "capped";
    // v_bk is capped at K; L_pred at prec >= K
    const uint32_t vex = c.v_bk.v;
    if (c.v_bk.lower_bound) {
        if (c.pred.L < c.K) return "LAW_FAIL";
    } else if (c.pred.L != vex) {
        return "LAW_FAIL";
    }
    if (c.eis.kind == "exact") {
        if (c.v_bk.lower_bound) return c.eis.L >= c.K ? "capped" : "COH_FAIL";
        if (c.eis.capped) return c.eis.L <= vex ? "capped" : "COH_FAIL";
        return c.eis.L == vex ? "agree" : "COH_FAIL";
    }
    return c.eis.L >= vex ? "bound_ok" : "BOUND_FAIL";
}

std::string Vec(const std::vector<uint32_t>& v) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << "]";
    return o.str();
}

std::string Vec64(const std::vector<uint64_t>& v) {
    std::ostringstream o;
    o << "(";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << ")";
    return o.str();
}

} // namespace

ComponentCell ComputeComponentCell(uint32_t n, uint64_t ell, uint32_t K, const ComponentOptions& opts) {
    const auto t0 = std::chrono::steady_clock::now();
    CheckPrimeEll(ell, "ComputeComponentCell");
    ENGINE_CHECK_MSG(n >= 2 && (n & 1) == 0 && K >= 2, "ComputeComponentCell: n >= 2 even, K >= 2");
    const uint64_t q = powfit(ell, K);
    if (q == 0) Refuse("ComputeComponentCell: ell^K >= 2^64");
    ComponentCell c;
    c.n = n;
    c.ell = ell;
    c.K = K;
    c.k = uint64_t(n) + 2;
    std::vector<uint64_t> S = opts.primes.empty() ? std::vector<uint64_t>{2, 3, 5, 7} : opts.primes;
    S.push_back(ell);
    std::sort(S.begin(), S.end());
    S.erase(std::unique(S.begin(), S.end()), S.end());
    c.primes = S;
    const ringalg::OpModule N = ringalg::RunModule(n, ell, K, S, opts.route);
    c.exps = N.exps;

    // the ordinary Eisenstein system
    std::vector<uint64_t> eis_sys;
    for (uint64_t p : S) eis_sys.push_back(p == ell ? 1 : static_cast<uint64_t>((1 + PowModU(p, c.k - 1, ell)) % ell));
    const ringalg::Block B = ringalg::Localize(N, eis_sys);
    EisReading& E = c.eis;
    E.present = B.local.rank() > 0;
    if (!E.present) {
        E.kind = "absent";
    } else if (!IsFree(B.local, K)) {
        E.D = B.local.rank();
        E.kind = "torsion_block";
    } else {
        E.D = B.local.rank();
        E.free = true;
        E.g = (E.D - 1) / 2;
        E.eis_root = true;
        E.square_ok = true;
        for (uint64_t p : S) {
            if (p == ell) continue;
            E.primes.push_back(p);
            const Poly chi = hida::CharpolyFree(OpOf(B.local, p));
            const uint64_t e = static_cast<uint64_t>((1 + PowModU(p, c.k - 1, q)) % q);
            const auto dv = SyntheticDivide(chi, e, q);
            E.eis_root = E.eis_root && dv.second == 0;
            if (E.D == 1) {
                E.per_p.push_back(0);
                E.cusp_factor.push_back(Poly{1});
                continue;
            }
            if (E.D % 2 == 0) {   // not 1 + 2g: no square root reading
                E.square_ok = false;
                E.per_p.push_back(ValCapped(EvalPoly(dv.first, e, q), ell, K) / 2);
                E.cusp_factor.push_back(Poly{});
                continue;
            }
            const auto sr = MonicSqrt(dv.first, q);
            E.square_ok = E.square_ok && sr.second;
            E.cusp_factor.push_back(sr.first);
            E.per_p.push_back(sr.second ? ValCapped(EvalPoly(sr.first, e, q), ell, K)
                                        : ValCapped(EvalPoly(dv.first, e, q), ell, K) / 2);
        }
        E.kind = (E.D == 1 || (E.g == 1 && E.square_ok)) ? "exact" : "upper_bound";
        E.L = *std::min_element(E.per_p.begin(), E.per_p.end());
        E.capped = true;
        for (uint32_t v : E.per_p)
            if (v == E.L && v < K) E.capped = false;
    }

    // the Bernoulli side
    c.class_zero = (c.k % (ell - 1) == 0);
    if (!c.class_zero) {
        c.v_bk = ValBernoulliOverK(c.k, ell, K);
        uint32_t prec = std::max<uint32_t>(K, 4);
        while (prec > K && powfit(ell, prec) == 0) --prec;
        c.pred = PredictedDepth(c.k, ell, prec);
    } else {
        c.pred.na = true;
    }
    c.law = Law(c);

    // the g = 2 intra-block readings
    if (opts.intra) {
        size_t ell_idx = 0;
        for (size_t i = 0; i < S.size(); ++i)
            if (S[i] == ell) ell_idx = i;
        for (const ringalg::JointSystem& js : ringalg::RationalSystems(ringalg::ReduceModEll(N))) {
            if (js.dim != 4 || js.a == eis_sys || js.a[ell_idx] % ell == 0) continue;
            const ringalg::Block b = ringalg::Localize(N, js.a);
            if (!IsFree(b.local, K) || b.local.rank() != 4) continue;
            IntraBlock ib;
            ib.system = js.a;
            ib.D = 4;
            ib.g = 2;
            ib.square_ok = true;
            for (uint64_t p : S) {
                if (p == ell) continue;
                const Poly chi = hida::CharpolyFree(OpOf(b.local, p));
                const auto sr = MonicSqrt(chi, q);
                ib.square_ok = ib.square_ok && sr.second;
                ib.cusp_factor.push_back(sr.first);
                const Poly& r = sr.first;   // x^2 + r1 x + r0
                const u128 disc = SubMod(MulMod(r[1], r[1], q), MulMod(4 % q, r[0], q), q);
                const uint32_t v = ValCapped(static_cast<uint64_t>(disc), ell, K);
                ib.disc_v.push_back(v);
                if (v >= K) {
                    ib.kinds.push_back("capped");
                } else if (v & 1) {
                    ib.kinds.push_back("ramified");
                } else {
                    uint64_t u = static_cast<uint64_t>(disc);
                    for (uint32_t i = 0; i < v; ++i) u /= ell;
                    ib.kinds.push_back(PowModU(u % ell, (ell - 1) / 2, ell) == 1 ? "split" : "inert");
                }
            }
            bool unit = false;
            for (uint32_t v : ib.disc_v) unit = unit || v == 0;
            std::vector<std::string> ks;
            uint32_t lv = K;
            for (size_t i = 0; i < ib.kinds.size(); ++i)
                if (ib.kinds[i] != "capped") {
                    ks.push_back(ib.kinds[i]);
                    lv = std::min(lv, ib.disc_v[i]);
                }
            std::sort(ks.begin(), ks.end());
            ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
            if (!ib.square_ok) ib.verdict = "INCONSISTENT_not_a_square";
            else if (unit) ib.verdict = "INCONSISTENT_unit_disc";
            else if (ks.empty()) ib.verdict = "capped";
            else if (ks.size() > 1) ib.verdict = "INCONSISTENT_kinds";
            else ib.verdict = ks[0] == "ramified" ? "one_component_ramified"
                            : ks[0] == "inert"    ? "one_component_inert"
                                                  : "two_rational_eigenforms";
            ib.level2 = ks.empty() ? K : lv;
            c.intra.push_back(ib);
        }
    }
    c.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return c;
}

std::string ComponentLine(const ComponentCell& c) {
    std::ostringstream o;
    o << "HIDA_COMPONENT n=" << c.n << " ell=" << c.ell << " K=" << c.K << " k=" << c.k << " D=" << c.eis.D
      << " g=" << c.eis.g << " L_coh=" << (c.eis.capped ? ">=" : "") << c.eis.L << " kind=" << c.eis.kind << " v_bk=";
    if (c.class_zero) o << "na";
    else o << (c.v_bk.lower_bound ? ">=" : "") << c.v_bk.v;
    o << " L_pred=";
    if (c.pred.na) o << "na";
    else o << (c.pred.lower_bound ? ">=" : "") << c.pred.L;
    o << " law=" << c.law << " eis_root=" << (c.eis.eis_root ? "ok" : "EIS_ROOT_FAIL") << " primes=" << Vec64(c.eis.primes)
      << " per_p=" << Vec(c.eis.per_p) << " intra_blocks=" << c.intra.size();
    return o.str();
}

std::string IntraLine(const ComponentCell& c, const IntraBlock& b) {
    std::ostringstream o;
    o << "HIDA_INTRA n=" << c.n << " ell=" << c.ell << " K=" << c.K << " system=" << Vec64(b.system) << " D=" << b.D
      << " g=" << b.g << " disc_v=" << Vec(b.disc_v) << " kinds=[";
    for (size_t i = 0; i < b.kinds.size(); ++i) o << (i ? "," : "") << b.kinds[i];
    o << "] verdict=" << b.verdict << " level=" << b.level2 / 2 << (b.level2 % 2 ? ".5" : "");
    return o.str();
}

} // namespace hidacomp
} // namespace hecke_ext
