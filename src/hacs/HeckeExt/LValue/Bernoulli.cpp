// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-LV) Exact Bernoulli numbers -- see Bernoulli.h.
#include "Bernoulli.h"

#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"

#include <algorithm>
#include <stdexcept>

namespace hecke_ext {
namespace lvalue {

using u128 = unsigned __int128;

uint64_t ModU64(const nf::BigInt& x, uint64_t q) {
    ENGINE_CHECK_MSG(q >= 1, "ModU64: modulus must be >= 1");
    ENGINE_CHECK_MSG(!x.is_neg(), "ModU64: x must be >= 0");
    const nf::BigInt::Mag& limbs = x.magnitude();
    u128 r = 0;
    for (size_t i = limbs.size(); i-- > 0;) r = ((r << 32) | limbs[i]) % q;
    return static_cast<uint64_t>(r);
}

// ---- tangent numbers (Brent-Harvey) ----------------------------------------------
std::vector<nf::BigInt> TangentNumbers(uint32_t N) {
    using nf::BigInt;
    std::vector<BigInt> T(N);
    if (N == 0) return T;
    T[0] = BigInt(1);
    for (uint32_t k = 2; k <= N; ++k) T[k - 1] = T[k - 2] * BigInt(k - 1);
    for (uint32_t k = 2; k <= N; ++k)
        for (uint32_t j = k; j <= N; ++j) T[j - 1] = T[j - 2] * BigInt(j - k) + T[j - 1] * BigInt(j - k + 2);
    return T;
}

BernoulliOverK ExactBernoulliOverK(uint32_t k) {
    ENGINE_CHECK_MSG(k >= 2 && k % 2 == 0, "ExactBernoulliOverK: k must be even and >= 2");
    ENGINE_CHECK_MSG(k <= 20000, "ExactBernoulliOverK: k > 20000 is out of the exact route's range (use the Voronoi route)");
    BernoulliOverK b;
    b.k = k;
    b.sign = ((k / 2 - 1) % 2 == 0) ? 1 : -1;
    b.num = TangentNumbers(k / 2).back();   // T_{k-1}
    const nf::BigInt two_k = nf::BigInt::pow(nf::BigInt(2), k);
    b.den = two_k * (two_k - nf::BigInt(1));   // 2^k (2^k - 1)
    return b;
}

uint32_t ValuationAndUnitPart(const nf::BigInt& x, uint64_t ell, uint64_t q, uint64_t& unit_mod_q) {
    ENGINE_CHECK_MSG(!x.is_zero(), "ValuationAndUnitPart: zero has no valuation");
    ENGINE_CHECK_MSG(ell >= 2 && ell < (uint64_t(1) << 32), "ValuationAndUnitPart: ell must fit 32 bits");
    nf::BigInt y = x;
    uint32_t v = 0;
    while (y.mod_u32(static_cast<uint32_t>(ell)) == 0) {
        y = y.div_exact_u32(static_cast<uint32_t>(ell));
        ++v;
    }
    unit_mod_q = ModU64(y, q);
    return v;
}

// ---- small modular helpers ---------------------------------------------------------
uint64_t PowModU64(uint64_t b, uint64_t e, uint64_t q) { return engine::scalar::powmod<uint64_t>(b, e, q); }

uint64_t InvModU64(uint64_t a, uint64_t q) {
    uint64_t x = 0;
    if (!engine::scalar::try_invmod<uint64_t>(a, q, x)) throw std::logic_error("InvModU64: not a unit");
    return x;
}

uint32_t ValuationU64(uint64_t x, uint64_t ell) {
    ENGINE_CHECK_MSG(x > 0, "ValuationU64: zero has no valuation");
    uint32_t v = 0;
    while (x % ell == 0) {
        x /= ell;
        ++v;
    }
    return v;
}

uint64_t PrimitiveRootModPrime(uint64_t ell) {
    std::vector<uint64_t> f;
    uint64_t x = ell - 1;
    for (uint64_t p = 2; p * p <= x; ++p)
        if (x % p == 0) {
            f.push_back(p);
            while (x % p == 0) x /= p;
        }
    if (x > 1) f.push_back(x);
    for (uint64_t g = 2; g < ell; ++g) {
        bool ok = true;
        for (uint64_t p : f)
            if (PowModU64(g, (ell - 1) / p, ell) == 1) {
                ok = false;
                break;
            }
        if (ok) return g;
    }
    throw std::logic_error("PrimitiveRootModPrime: none found (ell not prime?)");
}

namespace {
uint64_t PowU64(uint64_t b, uint32_t e) {
    u128 r = 1;
    for (uint32_t i = 0; i < e; ++i) {
        r *= b;
        ENGINE_CHECK_MSG(r < (u128(1) << 62), "Bernoulli: ell^m must stay below 2^62");
    }
    return static_cast<uint64_t>(r);
}
void RequireRegularClass(uint64_t k, uint64_t ell) {
    ENGINE_CHECK_MSG(ell >= 5, "Bernoulli: ell >= 5 required");
    ENGINE_CHECK_MSG(k >= 2 && k % 2 == 0, "Bernoulli: k must be even and >= 2");
    ENGINE_CHECK_MSG(k % (ell - 1) != 0, "Bernoulli: (ell - 1) | k -- B_k / k is not ell-integral (von Staudt-Clausen)");
}
} // namespace

BernoulliReading BernoulliModEllPowerExact(uint32_t k, uint64_t ell, uint32_t m) {
    RequireRegularClass(k, ell);
    ENGINE_CHECK_MSG(m >= 1, "Bernoulli: m >= 1");
    const uint64_t q = PowU64(ell, m);
    const BernoulliOverK b = ExactBernoulliOverK(k);
    uint64_t un = 0, ud = 0;
    const uint32_t vn = ValuationAndUnitPart(b.num, ell, q, un);
    const uint32_t vd = ValuationAndUnitPart(b.den, ell, q, ud);
    ENGINE_CHECK_MSG(vn >= vd, "Bernoulli: B_k / k is not ell-integral (contradicts von Staudt-Clausen / Kummer)");
    BernoulliReading R;
    R.k = k;
    R.ell = ell;
    R.m = m;
    R.v = vn - vd;
    R.v_exact = true;
    u128 x = static_cast<u128>(un) * InvModU64(ud, q) % q;
    if (b.sign < 0) x = (q - x) % q;
    x = R.v >= m ? 0 : x * PowU64(ell, R.v) % q;
    R.residue = static_cast<uint64_t>(x);
    return R;
}

BernoulliReading BernoulliModEllPowerVoronoi(uint64_t k, uint64_t ell, uint32_t m) {
    RequireRegularClass(k, ell);
    ENGINE_CHECK_MSG(m >= 1, "Bernoulli: m >= 1");
    const uint32_t vk = ValuationU64(k, ell);
    const uint64_t N = PowU64(ell, m + vk);
    const uint64_t q = PowU64(ell, m);
    const uint64_t a = PrimitiveRootModPrime(ell);
    ENGINE_CHECK_MSG(N <= (uint64_t(1) << 40), "Bernoulli: the Voronoi route's sum has more than 2^40 terms");
    u128 S = 0;
    for (uint64_t j = 1; j < N; ++j) {
        if (j % ell == 0 && k - 1 >= m + vk) continue;   // j^{k-1} == 0 mod N
        const uint64_t fl = static_cast<uint64_t>((static_cast<u128>(j) * a) / N);
        if (fl == 0) continue;
        S = (S + static_cast<u128>(PowModU64(j, k - 1, N)) * (fl % N)) % N;
    }
    const uint64_t R0 = static_cast<uint64_t>(static_cast<u128>(PowModU64(a, k - 1, N)) * S % N);
    // (a^k - 1) B_k == k R0 (mod N), N = ell^{m + v(k)}  ==>  (a^k - 1) B_k / k == R0 (mod ell^m)
    const uint64_t den = (PowModU64(a, k, q) + q - 1) % q;
    BernoulliReading Rd;
    Rd.k = static_cast<uint32_t>(k);
    Rd.ell = ell;
    Rd.m = m;
    Rd.residue = static_cast<uint64_t>(static_cast<u128>(R0 % q) * InvModU64(den, q) % q);
    if (Rd.residue == 0) {
        Rd.v = m;
        Rd.v_exact = false;
    } else {
        Rd.v = ValuationU64(Rd.residue, ell);
        Rd.v_exact = true;
    }
    return Rd;
}

// ---- (slot 19, WP-LV2) twisted Bernoulli numbers --------------------------------------
uint64_t TeichmullerPowMod(uint64_t a, uint32_t j, uint64_t ell, uint32_t P) {
    const uint64_t q = PowU64(ell, P);
    ENGINE_CHECK_MSG(a % ell != 0, "Teichmuller: a must be a unit mod ell");
    const uint64_t e = PowU64(ell, P - 1);
    return PowModU64(PowModU64(a % q, e, q), j, q);
}

namespace {
void RequireTwist(uint32_t k, uint64_t ell, uint32_t j) {
    ENGINE_CHECK_MSG(ell >= 5 && PrimitiveRootModPrime(ell) > 0, "TwistedBernoulli: ell >= 5 prime");
    ENGINE_CHECK_MSG(k >= 1, "TwistedBernoulli: k >= 1");
    ENGINE_CHECK_MSG(j % (ell - 1) != 0, "TwistedBernoulli: chi = omega^j must be non-trivial (j != 0 mod ell - 1)");
    ENGINE_CHECK_MSG((j + k) % 2 == 0, "TwistedBernoulli: chi(-1) = (-1)^k is required (else B_{k,chi} = 0)");
}
// ell * B_i mod ell^P (an ell-adic integer: v(B_i) >= -1); i >= 1.
uint64_t EllTimesBernoulli(uint32_t i, uint64_t ell, uint32_t P) {
    const uint64_t q = PowU64(ell, P);
    if (i == 1) return static_cast<uint64_t>(static_cast<u128>(q - ell % q) * InvModU64(2, q) % q);   // -ell/2
    if (i % 2 == 1) return 0;
    const BernoulliOverK b = ExactBernoulliOverK(i);      // B_i / i = sign num / den
    uint64_t un = 0, ud = 0, ui = 0;
    const uint32_t vn = ValuationAndUnitPart(b.num, ell, q, un);
    const uint32_t vd = ValuationAndUnitPart(b.den, ell, q, ud);
    const uint32_t vi = ValuationAndUnitPart(nf::BigInt(i), ell, q, ui);
    ENGINE_CHECK_MSG(vn + vi + 1 >= vd, "TwistedBernoulli: v(B_i) < -1 contradicts von Staudt-Clausen");
    const uint32_t v = vn + vi + 1 - vd;
    if (v >= P) return 0;
    u128 x = static_cast<u128>(un) * ui % q * InvModU64(ud, q) % q;
    if (b.sign < 0) x = (q - x) % q;
    return static_cast<uint64_t>(x * PowU64(ell, v) % q);
}
} // namespace

BernoulliReading TwistedBernoulliOverK(uint32_t k, uint64_t ell, uint32_t j, uint32_t m) {
    RequireTwist(k, ell, j);
    ENGINE_CHECK_MSG(m >= 1, "TwistedBernoulli: m >= 1");
    const uint32_t vk = ValuationU64(k, ell);
    uint32_t Pmax = 0;
    for (u128 x = ell; x < (u128(1) << 62); x *= ell) ++Pmax;
    const uint32_t P = std::min<uint32_t>(Pmax, m + vk + 4);
    ENGINE_CHECK_MSG(P >= m + vk + 1, "TwistedBernoulli: ell^{m + v(k) + 1} must stay below 2^62");
    const uint64_t q = PowU64(ell, P);
    std::vector<uint64_t> chi(ell, 0);
    for (uint64_t a = 1; a < ell; ++a) chi[a] = TeichmullerPowMod(a, j, ell, P);
    auto S = [&](uint32_t r) {
        u128 s = 0;
        for (uint64_t a = 1; a < ell; ++a) s = (s + static_cast<u128>(chi[a]) * PowModU64(a, r, q)) % q;
        return s;
    };
    // T := ell B_{k,chi} = S_k + sum_{i >= 1} C(k,i) (ell B_i) ell^{i-1} S_{k-i}  (mod ell^P; i - 1 >= P vanish)
    u128 T = S(k);
    nf::BigInt C(1);                                      // C(k, i), exact
    const uint32_t imax = std::min<uint32_t>(k, P);
    for (uint32_t i = 1; i <= imax; ++i) {
        C = (C * nf::BigInt(k - i + 1)).div_exact_u32(i);
        const uint64_t eb = EllTimesBernoulli(i, ell, P);
        if (eb == 0) continue;
        u128 t = static_cast<u128>(ModU64(C, q)) * eb % q * S(k - i) % q;
        for (uint32_t s = 1; s < i; ++s) t = t * ell % q;
        T = (T + t) % q;
    }
    ENGINE_CHECK_MSG(T % ell == 0, "TwistedBernoulli: ell B_{k,chi} must vanish mod ell (S_k == 0 mod ell)");
    const uint64_t q1 = PowU64(ell, P - 1);
    const uint64_t B = static_cast<uint64_t>(T / ell) % q1;   // B_{k,chi} mod ell^{P-1}
    BernoulliReading R;
    R.k = k;
    R.ell = ell;
    R.m = m;
    const uint32_t prec = P - 1 - vk;                     // B_{k,chi}/k is known mod ell^prec
    R.v = prec;
    R.v_exact = false;
    R.residue = 0;
    if (B != 0) {
        const uint32_t vB = ValuationU64(B, ell);
        ENGINE_CHECK_MSG(vB >= vk, "TwistedBernoulli: B_{k,chi}/k is not ell-integral");
        uint64_t uk = 0;
        (void)ValuationAndUnitPart(nf::BigInt(k), ell, q1, uk);
        const uint64_t qp = PowU64(ell, prec);
        const uint64_t Bk = static_cast<uint64_t>(static_cast<u128>((B / PowU64(ell, vk)) % qp) * InvModU64(uk % qp, qp) % qp);
        if (Bk != 0) {
            R.v = ValuationU64(Bk, ell);
            R.v_exact = true;
        }
        R.residue = Bk % PowU64(ell, m);
    }
    return R;
}

BernoulliReading TwistedBernoulliKummer(uint32_t k, uint64_t ell, uint32_t j, uint32_t m, uint64_t* k_prime) {
    RequireTwist(k, ell, j);
    ENGINE_CHECK_MSG(m >= 1, "TwistedBernoulli: m >= 1");
    const uint64_t L = PowU64(ell, m - 1);
    const uint64_t a = (static_cast<uint64_t>(k) + j) % (ell - 1);
    uint64_t kp = 0;
    for (uint64_t x = k % L, it = 0;; x += L, ++it) {   // x == k mod ell^{m-1}; the first x >= 2 with x == a mod ell - 1
        if (x >= 2 && x % (ell - 1) == a) {
            kp = x;
            break;
        }
        ENGINE_CHECK_MSG(it < 2 * (ell - 1) + 2, "TwistedBernoulli: no k' found (CRT)");
    }
    if (k_prime) *k_prime = kp;
    const BernoulliReading V = BernoulliModEllPowerVoronoi(kp, ell, m);
    const uint64_t q = PowU64(ell, m);
    const uint64_t euler = (1 + q - PowModU64(ell, kp - 1, q)) % q;
    BernoulliReading R;
    R.k = k;
    R.ell = ell;
    R.m = m;
    R.residue = static_cast<uint64_t>(static_cast<u128>(V.residue) * euler % q);
    R.v = R.residue == 0 ? m : ValuationU64(R.residue, ell);
    R.v_exact = R.residue != 0;
    return R;
}

bool BernoulliEquals(uint32_t k, bool negative, const std::string& P_decimal, const std::string& Q_decimal) {
    const BernoulliOverK b = ExactBernoulliOverK(k);
    if ((b.sign < 0) != negative) return false;
    // B_k = sign * k * num / den == P / Q  <=>  k * num * Q == P * den
    const nf::BigInt lhs = b.num * nf::BigInt(k) * nf::BigInt::from_string(Q_decimal);
    const nf::BigInt rhs = nf::BigInt::from_string(P_decimal) * b.den;
    return lhs == rhs;
}

} // namespace lvalue
} // namespace hecke_ext
