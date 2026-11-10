// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// IntPolyFactor.cpp -- see IntPolyFactor.h (Yun + Zassenhaus over Z for a monic polynomial; exact, verified).
#include "IntPolyFactor.h"

#include <algorithm>
#include <sstream>
#include <utility>

#include "ModUtil.h"
#include "NfCheck.h"
#include "PolyAlg.h"
#include "PureField.h"

namespace nf {

namespace {

using ZPoly = std::vector<BigInt>;

constexpr uint32_t kFirstPrime = 10007;   // the Zassenhaus primes q: odd, >= 10007, q^2 < 2^32 is not needed (per-term reduction)
constexpr unsigned kPrimeTrials = 5;      // good primes tried; the one with the fewest factors mod q is used
constexpr uint64_t kMaxSubsets = 200000;  // the recombination cap (subsets looked at, constant-term rejections included)

int deg(const ZPoly& a) { return static_cast<int>(a.size()) - 1; }
ZPoly ztrim(ZPoly a) {
    while (!a.empty() && a.back().is_zero()) a.pop_back();
    return a;
}
ZPoly zsub(const ZPoly& a, const ZPoly& b) {
    ZPoly r(std::max(a.size(), b.size()));
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = (i < a.size() ? a[i] : BigInt()) - (i < b.size() ? b[i] : BigInt());
    return ztrim(r);
}
ZPoly zderiv(const ZPoly& a) {
    ZPoly r;
    for (std::size_t i = 1; i < a.size(); ++i) r.push_back(a[i] * BigInt(static_cast<int64_t>(i)));
    return ztrim(r);
}
BigInt zcontent(const ZPoly& a) {
    BigInt g;
    for (const BigInt& c : a) g = BigInt::gcd(g, c);
    return g;
}
// The primitive part with a positive leading coefficient.
ZPoly zprimpart(ZPoly a) {
    a = ztrim(a);
    if (a.empty()) return a;
    BigInt g = zcontent(a);
    if (a.back().is_neg()) g = -g;
    if (g != BigInt(1))
        for (BigInt& c : a) c /= g;
    return a;
}
// A pseudo-remainder of a by b (a multiple of prem by a power of lc(b)); b != 0.
ZPoly zprem(ZPoly r, const ZPoly& b) {
    const BigInt lb = b.back();
    while (!r.empty() && deg(r) >= deg(b)) {
        const BigInt lr = r.back();
        const std::size_t sh = r.size() - b.size();
        for (BigInt& c : r) c *= lb;
        for (std::size_t j = 0; j < b.size(); ++j) r[sh + j] -= lr * b[j];
        r = ztrim(r);
        if (r.size() > 8) r = zprimpart(r);   // curb the growth; only the class up to units and content matters
    }
    return r;
}
// gcd over Q, primitive with positive leading coefficient (monic when a or b is monic: Gauss's lemma).
ZPoly zgcd(ZPoly a, ZPoly b) {
    a = zprimpart(a);
    b = zprimpart(b);
    if (deg(a) < deg(b)) std::swap(a, b);
    while (!b.empty()) {
        ZPoly r = zprimpart(zprem(a, b));
        a = std::move(b);
        b = std::move(r);
    }
    return a;
}
// a / b exactly, b monic; NF_CHECK that the remainder is zero.
ZPoly zdiv_monic(const ZPoly& a0, const ZPoly& b, bool* exact = nullptr) {
    NF_CHECK(!b.empty() && b.back() == BigInt(1));
    ZPoly r = ztrim(a0);
    if (deg(r) < deg(b)) {
        if (exact) *exact = r.empty();
        else NF_CHECK_MSG(r.empty(), "zdiv_monic: not exact");
        return {};
    }
    ZPoly q(r.size() - b.size() + 1);
    for (int k = deg(r); k >= deg(b); --k) {
        const BigInt c = r[k];
        const std::size_t sh = static_cast<std::size_t>(k - deg(b));
        q[sh] = c;
        if (!c.is_zero())
            for (std::size_t j = 0; j < b.size(); ++j) r[sh + j] -= c * b[j];
    }
    r = ztrim(r);
    if (exact) *exact = r.empty();
    else NF_CHECK_MSG(r.empty(), "zdiv_monic: not exact");
    return ztrim(q);
}
bool is_monic(const ZPoly& a) { return !a.empty() && a.back() == BigInt(1); }

// Yun: f = prod_i g_i^i (f monic).
std::vector<std::pair<ZPoly, unsigned>> yun(const ZPoly& f) {
    std::vector<std::pair<ZPoly, unsigned>> out;
    const ZPoly f1 = zderiv(f);
    ZPoly a = zgcd(f, f1);
    NF_CHECK(is_monic(a));
    ZPoly b = zdiv_monic(f, a), c = zdiv_monic(f1, a);
    ZPoly d = zsub(c, zderiv(b));
    for (unsigned i = 1; deg(b) > 0; ++i) {
        a = d.empty() ? b : zgcd(b, d);
        NF_CHECK(is_monic(a));
        if (deg(a) > 0) out.emplace_back(a, i);
        b = zdiv_monic(b, a);
        c = zdiv_monic(d, a);
        d = zsub(c, zderiv(b));
    }
    return out;
}

FqPoly fq_deriv(const FqPoly& a, uint32_t q) {
    FqPoly r;
    for (std::size_t i = 1; i < a.size(); ++i) r.push_back(static_cast<uint32_t>(static_cast<uint64_t>(a[i]) * (i % q) % q));
    return fq_trim(r);
}
ZPoly zfrom_fq(const FqPoly& a) {
    ZPoly r;
    for (uint32_t x : a) r.push_back(BigInt(static_cast<int64_t>(x)));
    return r;
}
ZPoly zmod(ZPoly a, const BigInt& M) {
    for (BigInt& c : a) c = c.floor_mod(M);
    return ztrim(a);
}
ZPoly zsym(ZPoly a, const BigInt& M) {
    const BigInt half = M / BigInt(2);
    for (BigInt& c : a) {
        c = c.floor_mod(M);
        if (c > half) c -= M;
    }
    return ztrim(a);
}

// F == g0 h0 (mod q), g0 monic, coprime mod q: (G, H) with F == G H (mod q^k), G monic of degree deg g0.
std::pair<ZPoly, ZPoly> binary_lift(const ZPoly& F, const FqPoly& g0, const FqPoly& h0, uint32_t q, unsigned k) {
    PolyMod s, t;
    poly_xgcd_coprime(g0, h0, q, s, t);   // s g0 + t h0 = 1 (mod q)
    ZPoly G = zfrom_fq(g0), H = zfrom_fq(h0);
    BigInt pj(static_cast<int64_t>(q));
    for (unsigned j = 1; j < k; ++j) {
        const ZPoly E = zsub(F, zpoly_mul(G, H));
        PolyMod e(E.size());
        for (std::size_t i = 0; i < E.size(); ++i) {
            BigInt quo, rem;
            BigInt::divmod(E[i], pj, quo, rem);
            NF_CHECK_MSG(rem.is_zero(), "binary_lift: F != G H (mod q^j)");
            e[i] = quo.mod_u32(q);
        }
        e = poly_trim(e);
        PolyMod qq, Gc;
        poly_divrem(poly_mul(t, e, q), g0, q, qq, Gc);
        const PolyMod Hc = poly_add(poly_mul(s, e, q), poly_mul(qq, h0, q), q);
        if (G.size() < Gc.size()) G.resize(Gc.size());
        for (std::size_t i = 0; i < Gc.size(); ++i) G[i] += pj * BigInt(static_cast<int64_t>(Gc[i]));
        if (H.size() < Hc.size()) H.resize(Hc.size());
        for (std::size_t i = 0; i < Hc.size(); ++i) H[i] += pj * BigInt(static_cast<int64_t>(Hc[i]));
        pj *= BigInt(static_cast<int64_t>(q));
    }
    // pj = q^k: the lift is exact modulo q^k
    for (const BigInt& c : zsub(F, zpoly_mul(G, H))) NF_CHECK(c.floor_mod(pj).is_zero());
    return {G, H};
}
FqPoly fq_prod(const std::vector<FqPoly>& v, std::size_t lo, std::size_t hi, uint32_t q) {
    FqPoly r = {1};
    for (std::size_t i = lo; i < hi; ++i) r = fq_mul(r, v[i], q);
    return r;
}
// F == prod u[lo..hi) (mod q): the lifted factors, in the order of u, modulo M = q^k.
void lift_tree(const ZPoly& F, const std::vector<FqPoly>& u, std::size_t lo, std::size_t hi, uint32_t q, unsigned k,
               const BigInt& M, std::vector<ZPoly>& out) {
    if (hi - lo == 1) {
        out.push_back(zmod(F, M));
        return;
    }
    const std::size_t mid = lo + (hi - lo) / 2;
    auto [G, H] = binary_lift(F, fq_prod(u, lo, mid, q), fq_prod(u, mid, hi, q), q, k);
    lift_tree(G, u, lo, mid, q, k, M, out);
    lift_tree(H, u, mid, hi, q, k, M, out);
}

// Zassenhaus on a monic squarefree h of degree >= 1: (factor, irreducible).
std::vector<std::pair<ZPoly, bool>> zassenhaus(const ZPoly& h) {
    if (deg(h) == 1) return {{h, true}};
    // the prime
    uint32_t best_q = 0;
    std::vector<FqPoly> best;
    unsigned good = 0;
    for (uint32_t q = kFirstPrime; good < kPrimeTrials; q += 2) {
        if (!is_prime_u32(q)) continue;
        const FqPoly hq = fq_from_bigint(h, q);
        if (fq_gcd(hq, fq_deriv(hq, q), q) != FqPoly{1}) continue;
        ++good;
        std::vector<FqPoly> fac = fq_factor_squarefree(hq, q, static_cast<unsigned>(deg(h)));
        if (best_q == 0 || fac.size() < best.size()) {
            best_q = q;
            best = std::move(fac);
        }
        if (best.size() == 1) return {{h, true}};   // irreducible mod q
    }
    const uint32_t q = best_q;
    // the bound: every coefficient of a factor of h is <= 2^d ||h||_2 <= 2^d (d + 1) max|a_j| =: B; q^k > 2 B
    BigInt amax;
    for (const BigInt& c : h) amax = std::max(amax, c.abs());
    const BigInt B = BigInt::pow(BigInt(2), static_cast<uint64_t>(deg(h))) * BigInt(static_cast<int64_t>(deg(h) + 1)) * amax;
    unsigned k = 1;
    BigInt M(static_cast<int64_t>(q));
    while (M <= BigInt(2) * B) {
        M *= BigInt(static_cast<int64_t>(q));
        ++k;
    }
    std::vector<ZPoly> L;
    lift_tree(h, best, 0, best.size(), q, k, M, L);
    // recombination
    std::vector<std::pair<ZPoly, bool>> out;
    ZPoly cur = h;
    std::vector<std::size_t> T(L.size());
    for (std::size_t i = 0; i < T.size(); ++i) T[i] = i;
    uint64_t looked = 0;
    bool capped = false;
    for (std::size_t s = 1; 2 * s <= T.size() && !capped;) {
        bool found = false;
        std::vector<std::size_t> idx(s);
        for (std::size_t i = 0; i < s; ++i) idx[i] = i;
        for (;;) {
            if (++looked > kMaxSubsets) {
                capped = true;
                break;
            }
            // constant-term test
            BigInt c0(1);
            for (std::size_t i : idx) c0 = (c0 * (L[T[i]].empty() ? BigInt() : L[T[i]][0])).floor_mod(M);
            if (c0 > M / BigInt(2)) c0 -= M;
            const BigInt& h0 = cur[0];
            bool pass = h0.is_zero() ? true : (!c0.is_zero() && (h0 % c0).is_zero());
            if (pass) {
                ZPoly g = {BigInt(1)};
                for (std::size_t i : idx) g = zmod(zpoly_mul(g, L[T[i]]), M);
                g = zsym(g, M);
                bool exact = false;
                ZPoly quo = zdiv_monic(cur, g, &exact);
                if (exact) {
                    out.emplace_back(g, true);
                    cur = std::move(quo);
                    std::vector<std::size_t> rest;
                    for (std::size_t i = 0, j = 0; i < T.size(); ++i) {
                        if (j < s && idx[j] == i) { ++j; continue; }
                        rest.push_back(T[i]);
                    }
                    T = std::move(rest);
                    found = true;
                    break;
                }
            }
            // the next s-subset of {0 .. |T| - 1}
            std::size_t i = s;
            while (i > 0 && idx[i - 1] == T.size() - s + (i - 1)) --i;
            if (i == 0) break;
            ++idx[i - 1];
            for (std::size_t j = i; j < s; ++j) idx[j] = idx[j - 1] + 1;
        }
        if (!found) ++s;   // after a find, the same size again on the smaller set
    }
    if (deg(cur) > 0) out.emplace_back(cur, !capped);
    return out;
}

// Sort key: degree, then -c_{d-1}, ..., -c_0 (linear factors X - r by r ascending).
bool factor_less(const ZFactor& a, const ZFactor& b) {
    if (a.poly.size() != b.poly.size()) return a.poly.size() < b.poly.size();
    for (std::size_t i = a.poly.size(); i-- > 0;) {
        if (a.poly[i] != b.poly[i]) return -a.poly[i] < -b.poly[i];
    }
    return a.multiplicity < b.multiplicity;
}

}  // namespace

std::vector<BigInt> zpoly_mul(const std::vector<BigInt>& a, const std::vector<BigInt>& b) {
    if (a.empty() || b.empty()) return {};
    ZPoly r(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].is_zero()) continue;
        for (std::size_t j = 0; j < b.size(); ++j) r[i + j] += a[i] * b[j];
    }
    return ztrim(r);
}

std::string zpoly_text(const std::vector<BigInt>& f0, const std::string& var) {
    const ZPoly f = ztrim(f0);
    if (f.empty()) return "0";
    std::string out;
    for (std::size_t k = f.size(); k-- > 0;) {
        if (f[k].is_zero()) continue;
        const bool neg = f[k].is_neg();
        const std::string c = f[k].abs().to_string();
        const std::string mon = k == 0 ? "" : (k == 1 ? var : var + "^" + std::to_string(k));
        const std::string term = k == 0 ? c : (c == "1" ? mon : c + mon);
        if (out.empty()) out = neg ? "-" + term : term;
        else out += (neg ? " - " : " + ") + term;
    }
    return out;
}

std::string ZFactorisation::text(const std::string& var) const {
    if (factors.size() == 1 && factors[0].multiplicity == 1) return zpoly_text(factors[0].poly, var);
    std::string out;
    for (const ZFactor& f : factors) {
        if (!out.empty()) out += " ";
        const std::string t = zpoly_text(f.poly, var);
        out += t == var ? t : "(" + t + ")";   // the factor X needs no parentheses
        if (f.multiplicity > 1) out += "^" + std::to_string(f.multiplicity);
    }
    return out;
}

ZFactorisation factor_monic_over_z(const std::vector<BigInt>& f0) {
    const ZPoly f = ztrim(f0);
    NF_CHECK_MSG(deg(f) >= 1 && is_monic(f), "factor_monic_over_z: f must be monic of degree >= 1");
    ZFactorisation R;
    for (const auto& [g, i] : yun(f)) {
        for (auto& [p, irr] : zassenhaus(g)) {
            ZFactor z;
            z.poly = p;
            z.multiplicity = i;
            z.irreducible = irr;
            R.complete = R.complete && irr;
            R.factors.push_back(std::move(z));
        }
    }
    std::sort(R.factors.begin(), R.factors.end(), factor_less);
    // verification: the product, exactly
    ZPoly prod = {BigInt(1)};
    for (const ZFactor& z : R.factors) {
        NF_CHECK(is_monic(z.poly) && deg(z.poly) >= 1);
        for (unsigned m = 0; m < z.multiplicity; ++m) prod = zpoly_mul(prod, z.poly);
    }
    NF_CHECK_MSG(prod == f, "factor_monic_over_z: the product of the factors is not f");
    R.verified = true;
    return R;
}

}  // namespace nf
