// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZRational -- slot-25 HZ-RAT, the rational step (see the header for the mathematics and the trust boundary).

#include "HeckeZRational.h"

#include "HeckeZCharpoly.h"

#include "HeckeZFactorFinder.h"
#include "IntegralCharpolyCrt.h"
#include "Descent/DescentHecke.h"
#include "Descent/DescentRecord.h"
#include "../Hecke/HeckeLevelNSparse.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace heckez {

namespace h3 = integral_descent::hecke3;
using u64 = uint64_t;
using u128 = unsigned __int128;

// =====================================================================================================================================
// dimension formulas
// =====================================================================================================================================
namespace {
std::vector<std::pair<u64, uint32_t>> FactorSmall(u64 n) {
    std::vector<std::pair<u64, uint32_t>> f;
    for (u64 p = 2; p * p <= n; ++p) {
        uint32_t e = 0;
        while (n % p == 0) { n /= p; ++e; }
        if (e) f.emplace_back(p, e);
    }
    if (n > 1) f.emplace_back(n, 1);
    return f;
}
u64 Gcd(u64 a, u64 b) { while (b) { const u64 t = a % b; a = b; b = t; } return a; }
u64 Phi(u64 n) {
    u64 r = n;
    for (const auto& pe : FactorSmall(n)) r = r / pe.first * (pe.first - 1);
    return r;
}
int Legendre(int64_t a, u64 p) {   // p odd prime
    int64_t r = a % static_cast<int64_t>(p);
    if (r < 0) r += static_cast<int64_t>(p);
    if (r == 0) return 0;
    return PowModU64(static_cast<u64>(r), (p - 1) / 2, p) == 1 ? 1 : -1;
}
int64_t NuTwo(u64 N) {   // elliptic points of order 2
    int64_t r = 1;
    for (const auto& pe : FactorSmall(N)) {
        if (pe.first == 2) { if (pe.second >= 2) return 0; continue; }
        r *= 1 + Legendre(-1, pe.first);
    }
    return r;
}
int64_t NuThree(u64 N) {   // elliptic points of order 3
    int64_t r = 1;
    for (const auto& pe : FactorSmall(N)) {
        if (pe.first == 3) { if (pe.second >= 2) return 0; continue; }
        if (pe.first == 2) return 0;
        r *= 1 + Legendre(-3, pe.first);
    }
    return r;
}
std::vector<u64> Divisors(u64 n) {
    std::vector<u64> d;
    for (u64 t = 1; t <= n; ++t) if (n % t == 0) d.push_back(t);
    return d;
}
}  // namespace

u64 Gamma0Index(u64 N) {
    u64 r = N;
    for (const auto& pe : FactorSmall(N)) r = r / pe.first * (pe.first + 1);
    return r;
}
u64 Gamma0Cusps(u64 N) {
    u64 c = 0;
    for (u64 t : Divisors(N)) c += Phi(Gcd(t, N / t));
    return c;
}
int64_t DimSk(uint32_t k, u64 N) {
    ENGINE_CHECK_MSG(N >= 1, "DimSk: N >= 1");
    if (k < 2 || k % 2) return 0;
    const int64_t mu = static_cast<int64_t>(Gamma0Index(N)), c = static_cast<int64_t>(Gamma0Cusps(N));
    const int64_t e2 = NuTwo(N), e3 = NuThree(N);
    const int64_t g12 = 12 + mu - 3 * e2 - 4 * e3 - 6 * c;   // 12 g (genus of X_0(N))
    ENGINE_CHECK_MSG(g12 % 12 == 0, "DimSk: 12 g not divisible by 12");
    const int64_t g = g12 / 12;
    if (k == 2) return g;
    const int64_t K = static_cast<int64_t>(k);
    return (K - 1) * (g - 1) + (K / 2 - 1) * c + e2 * (K / 4) + e3 * (K / 3);
}
int64_t DimMk(uint32_t k, u64 N) {
    if (k == 0) return 1;
    if (k % 2) return 0;
    const int64_t c = static_cast<int64_t>(Gamma0Cusps(N));
    return DimSk(k, N) + (k >= 4 ? c : c - 1);
}
int64_t DimSkNew(uint32_t k, u64 N) {
    // dim S_k(N) = sum_{M | N} sigma_0(N / M) dim S_k^new(M), inverted recursively (as dan.py)
    int64_t s = DimSk(k, N);
    for (u64 M : Divisors(N))
        if (M < N) s -= static_cast<int64_t>(Divisors(N / M).size()) * DimSkNew(k, M);
    return s;
}
u64 EisAnemicCount(uint32_t k, u64 N) {
    // #{psi primitive : cond(psi)^2 | N} - [k = 2, N = 1]
    u64 cnt = 0;
    for (u64 c = 1; c * c <= N; ++c) {
        if (N % (c * c)) continue;
        u64 r = 1;
        for (const auto& pe : FactorSmall(c)) {
            const u64 p = pe.first;
            u64 pe_ = 1;
            for (uint32_t i = 0; i < pe.second; ++i) pe_ *= p;
            r *= (pe.second == 1) ? (p - 2) : (pe_ - 2 * (pe_ / p) + pe_ / p / p);
        }
        cnt += r;
    }
    if (k == 2 && N == 1) cnt -= 1;
    return cnt;
}
int64_t DimAnemic(uint32_t k, u64 N) {
    int64_t s = static_cast<int64_t>(EisAnemicCount(k, N));
    for (u64 M : Divisors(N)) s += DimSkNew(k, M);
    return s;
}

// =====================================================================================================================================
// word arithmetic
// =====================================================================================================================================
uint32_t BitLen64(u64 x) { uint32_t b = 0; while (x) { ++b; x >>= 1; } return b; }

u64 PowModU64(u64 a, u64 e, u64 m) {
    ENGINE_CHECK_MSG(m >= 1, "PowModU64: m >= 1");
    u128 r = 1 % m, b = a % m;
    while (e) {
        if (e & 1u) r = (r * b) % m;
        b = (b * b) % m;
        e >>= 1;
    }
    return static_cast<u64>(r);
}

namespace {
// The sieve of the small primes (<= B).
const std::vector<uint32_t>& SmallPrimes() {
    static const std::vector<uint32_t> primes = [] {
        std::vector<uint8_t> comp(kSmallBound + 1, 0);
        std::vector<uint32_t> out;
        for (u64 i = 2; i <= kSmallBound; ++i) {
            if (comp[i]) continue;
            out.push_back(static_cast<uint32_t>(i));
            for (u64 j = i * i; j <= kSmallBound; j += i) comp[j] = 1;
        }
        return out;
    }();
    return primes;
}
u64 InvMod(u64 a, u64 m) { return PowModU64(a % m, m - 2, m); }   // m prime

// |c| mod Q from mixed-radix digits (Horner from the top), Q >= 2; word mulmod beyond 2^32 ([C, prop:crtcharpolyN](4)).
u64 DigitsModQ(const std::vector<uint32_t>& digits, const std::vector<uint32_t>& primes, uint32_t upto, u64 Q) {
    if (Q != 0 && Q <= (u64(1) << 32)) {
        u64 acc = 0;
        for (uint32_t k = upto; k-- > 0;) acc = (acc * (primes[k] % Q) + digits[k]) % Q;
        return acc;
    }
    return h3::MixedRadixModQ(std::vector<uint32_t>(digits.begin(), digits.begin() + upto),
                              std::vector<uint32_t>(primes.begin(), primes.begin() + upto), false, Q);
}

// ---- polynomials over F_ell (low to high, normalised: no trailing zero) ----
using Poly = std::vector<u64>;
void Trim(Poly& a) { while (!a.empty() && a.back() == 0) a.pop_back(); }
int Deg(const Poly& a) { return static_cast<int>(a.size()) - 1; }
Poly Derivative(const Poly& a, u64 ell) {
    Poly d;
    for (size_t i = 1; i < a.size(); ++i) d.push_back(static_cast<u64>((u128(a[i]) * (i % ell)) % ell));
    Trim(d);
    return d;
}
void DivRem(const Poly& a, const Poly& b, u64 ell, Poly& q, Poly& r) {
    ENGINE_CHECK_MSG(!b.empty(), "DivRem: division by zero polynomial");
    r = a;
    Trim(r);
    q.assign(r.size() >= b.size() ? r.size() - b.size() + 1 : 0, 0);
    const u64 inv = InvMod(b.back(), ell);
    while (r.size() >= b.size() && !r.empty()) {
        const size_t sh = r.size() - b.size();
        const u64 c = (r.back() * inv) % ell;
        q[sh] = c;
        for (size_t i = 0; i < b.size(); ++i) r[sh + i] = (r[sh + i] + ell - (c * b[i]) % ell) % ell;
        Trim(r);
    }
    Trim(q);
}
Poly Monic(Poly a, u64 ell) {
    Trim(a);
    if (a.empty()) return a;
    const u64 inv = InvMod(a.back(), ell);
    for (auto& x : a) x = (x * inv) % ell;
    return a;
}
Poly PGcd(Poly a, Poly b, u64 ell) {
    Trim(a);
    Trim(b);
    while (!b.empty()) {
        Poly q, r;
        DivRem(a, b, ell, q, r);
        a = std::move(b);
        b = std::move(r);
    }
    return Monic(a, ell);
}
// rad(a) = a / gcd(a, a'), monic (ell > deg a, so a' = 0 only for constants)
Poly Radical(const Poly& a, u64 ell) {
    const Poly g = PGcd(a, Derivative(a, ell), ell);
    Poly q, r;
    DivRem(a, g, ell, q, r);
    ENGINE_CHECK_MSG(r.empty(), "Radical: gcd does not divide");
    return Monic(q, ell);
}
u64 Eval(const Poly& a, u64 x, u64 ell) {
    u64 acc = 0;
    for (size_t i = a.size(); i-- > 0;) acc = (acc * x + a[i]) % ell;
    return acc;
}
u64 LeadPow(u64 c, u64 e, u64 ell) { return PowModU64(c, e, ell); }
// Res(A, B) over F_ell (Euclid; Res(A, B) = (-1)^{mn} lc(B)^{m - deg R} Res(B, R), R = A mod B)
u64 Resultant(Poly A, Poly B, u64 ell) {
    Trim(A);
    Trim(B);
    if (A.empty() || B.empty()) return 0;
    u64 acc = 1;
    for (;;) {
        const int m = Deg(A), n = Deg(B);
        if (n == 0) return (acc * LeadPow(B[0], static_cast<u64>(m), ell)) % ell;
        Poly q, R;
        DivRem(A, B, ell, q, R);
        if (R.empty()) return 0;
        if ((static_cast<int64_t>(m) * n) % 2) acc = (ell - acc) % ell;
        acc = (acc * LeadPow(B.back(), static_cast<u64>(m - Deg(R)), ell)) % ell;
        A = std::move(B);
        B = std::move(R);
    }
}
u64 Discriminant(const Poly& f, u64 ell) {   // f monic
    const int d = Deg(f);
    u64 r = Resultant(f, Derivative(f, ell), ell);
    if ((static_cast<int64_t>(d) * (d - 1) / 2) % 2) r = (ell - r) % ell;
    return r;
}
}  // namespace

// =====================================================================================================================================
// ZExact
// =====================================================================================================================================
namespace {
// fn(i) for i in [from, to) on T threads in chunks of `chunk` (fn writes index-owned storage only: no value depends on T)
template <class F>
void ParallelForIdx(size_t from, size_t to, uint32_t T, size_t chunk, F fn) {
    if (T <= 1 || to - from <= chunk) {
        for (size_t i = from; i < to; ++i) fn(i);
        return;
    }
    std::atomic<size_t> next(from);
    auto work = [&]() {
        for (;;) {
            const size_t a = next.fetch_add(chunk);
            if (a >= to) break;
            const size_t b = std::min(to, a + chunk);
            for (size_t i = a; i < b; ++i) fn(i);
        }
    };
    std::vector<std::thread> pool;
    for (uint32_t w = 1; w < T; ++w) {
        try {
            pool.emplace_back(work);
        } catch (const std::system_error&) {
            break;
        }
    }
    work();
    for (auto& t : pool) t.join();
}
}  // namespace
namespace {
// (HZ-RAT-PERF) the thread count of the current RationalStep, read by the Garner pass of ZFromResidues (no value depends on it)
thread_local uint32_t t_garner_threads = 1;
struct ScopedGarnerThreads {
    uint32_t prev;
    explicit ScopedGarnerThreads(uint32_t t) : prev(t_garner_threads) { t_garner_threads = std::max<uint32_t>(t, 1); }
    ~ScopedGarnerThreads() { t_garner_threads = prev; }
};
}  // namespace

// (HZ-RAT-PERF) Garner's mixed-radix digits in ONE O(K^2) pass, the column-update form: for every later index j the partial value
// acc_j = sum_{m<k} d_m prod_{i<m} ell_i and pw_j = prod_{i<k} ell_i, both mod ell_j, so d_k = (r_k - acc_k) pw_k^{-1} mod ell_k --
// the inverse of Garner's constant falls out on the fly.  Products reduced by Barrett (no division), the updates of the indices beyond
// a block of digits run in parallel over j (each j its own accumulator).  The digits of a residue vector are unique, so they equal
// h3::GarnerDigits(res, primes, h3::GarnerInverses(primes)) digit for digit (HeckeZRationalValidation arm GA compares them).
std::vector<uint32_t> GarnerDigitsFast(const std::vector<uint32_t>& primes, const std::vector<uint32_t>& res, uint32_t threads) {
    const size_t K = primes.size();
    ENGINE_CHECK_MSG(res.size() == K, "GarnerDigitsFast: one residue per prime");
    std::vector<Barrett31> br;
    br.reserve(K);
    for (uint32_t p : primes) {
        ENGINE_CHECK_MSG(p > 2 && p % 2 == 1, "GarnerDigitsFast: odd primes");
        br.emplace_back(p);
    }
    std::vector<uint64_t> acc(K, 0), pw(K, 1);
    std::vector<uint32_t> d(K, 0);
    constexpr size_t kBlock = 512;
    auto apply = [&](size_t j, size_t k0, size_t k1) {   // digits [k0, k1) into index j (> every k)
        const Barrett31& b = br[j];
        uint64_t a = acc[j], w = pw[j];
        for (size_t k = k0; k < k1; ++k) {
            a = b.reduce(a + uint64_t(d[k]) * w);
            w = b.reduce(w * primes[k]);
        }
        acc[j] = a;
        pw[j] = w;
    };
    for (size_t k0 = 0; k0 < K; k0 += kBlock) {
        const size_t k1 = std::min(K, k0 + kBlock);
        for (size_t k = k0; k < k1; ++k) {
            const Barrett31& b = br[k];
            ENGINE_CHECK_MSG(pw[k] != 0, "GarnerDigitsFast: the primes must be pairwise distinct");
            const uint64_t r = res[k] % b.p;
            const uint64_t diff = r >= acc[k] ? r - acc[k] : r + b.p - acc[k];
            d[k] = static_cast<uint32_t>(b.mul(diff, b.inv(pw[k])));
            for (size_t j = k + 1; j < k1; ++j) apply(j, k, k + 1);
        }
        if (k1 < K) {
            const uint32_t T = (K - k1) >= 4096 ? threads : 1;
            ParallelForIdx(k1, K, T, 1024, [&](size_t j) { apply(j, k0, k1); });
        }
    }
    return d;
}

uint64_t CoveredBits(const std::vector<uint32_t>& primes, size_t k) {
    u64 acc = 0;
    for (size_t i = 0; i < k && i < primes.size(); ++i) acc += BitLen64(primes[i]) - 1u;
    return acc;
}

ZExact ZFromResidues(const std::vector<uint32_t>& primes, const std::vector<uint32_t>& residues, uint64_t bound_bits) {
    ENGINE_CHECK_MSG(primes.size() == residues.size() && !primes.empty(), "ZFromResidues: one residue per prime");
    ZExact z;
    z.primes = primes;
    z.bound_bits = bound_bits;
    z.covered_bits = CoveredBits(primes, primes.size());
    ENGINE_CHECK_MSG(z.covered_bits >= bound_bits + 1, "ZFromResidues: the primes do not cover the certified bound");
    for (size_t i = 0; i < primes.size(); ++i) ENGINE_CHECK_MSG(residues[i] < primes[i], "ZFromResidues: residue >= prime");
    // (HZ-RAT-PERF fix F10) the one-pass Barrett route needs odd primes below 2^31 (Barrett31: 3 <= p < 2^31); any other list (a 2, a
    // prime >= 2^31) takes the descent's two-pass route, as before HZ-RAT-PERF -- the digits are unique, so the route changes no value
    const bool word31 = *std::max_element(primes.begin(), primes.end()) < (1u << 31) &&
                        *std::min_element(primes.begin(), primes.end()) > 2u;
    z.digits = word31 ? GarnerDigitsFast(primes, residues, t_garner_threads)
                      : h3::GarnerDigits(residues, primes, h3::GarnerInverses(primes));
    z.negative = h3::MixedRadixNegative(z.digits, primes);
    if (!z.negative) {
        z.abs_digits = z.digits;
    } else {
        // |c| = P - c^+ = ((P - 1) - c^+) + 1: digit-wise (ell_k - 1 - d_k), then + 1 with carry
        z.abs_digits.resize(primes.size());
        for (size_t k = 0; k < primes.size(); ++k) z.abs_digits[k] = primes[k] - 1u - z.digits[k];
        for (size_t k = 0; k < primes.size(); ++k) {
            if (z.abs_digits[k] + 1u < primes[k]) { ++z.abs_digits[k]; break; }
            z.abs_digits[k] = 0;
        }
    }
    z.significant = 0;
    for (size_t k = 0; k < primes.size(); ++k)
        if (z.abs_digits[k]) z.significant = static_cast<uint32_t>(k + 1);
    return z;
}
uint64_t ZExact::abs_bits_upper() const {
    u64 b = 0;
    for (uint32_t k = 0; k < significant; ++k) b += BitLen64(primes[k]);
    return b;
}
uint64_t ZExact::abs_mod(uint64_t Q) const { return DigitsModQ(abs_digits, primes, significant, Q); }
bool ZExact::abs_u64(uint64_t* v) const {
    u128 acc = 0;
    for (uint32_t k = significant; k-- > 0;) {
        acc = acc * primes[k] + abs_digits[k];
        if (acc >> 64) return false;
    }
    *v = static_cast<u64>(acc);
    return true;
}
std::string ZExact::hex() const { return integral_descent::record::ExactHex(digits, primes, negative); }
std::string ZExact::abs_hex() const { return integral_descent::record::ExactHex(abs_digits, primes, false); }

// =====================================================================================================================================
// Pratt
// =====================================================================================================================================
bool IsSmallPrime(uint64_t q) {
    if (q < 2 || q > kSmallBound) return false;
    const auto& sp = SmallPrimes();
    return std::binary_search(sp.begin(), sp.end(), static_cast<uint32_t>(q));
}

namespace {
bool VerifyPrattRec(const PrattCert& c, uint32_t node_cap, uint32_t* nodes) {
    if (++*nodes > node_cap) return false;
    const u64 q = c.q;
    if (q < 3) return false;
    if (q <= kSmallBound) return IsSmallPrime(q);
    // q - 1 = prod r^e exactly (128-bit product, no overflow)
    u128 prod = 1;
    for (const auto& f : c.factors) {
        if (f.first < 2 || f.second == 0) return false;
        for (uint32_t i = 0; i < f.second; ++i) {
            prod *= f.first;
            if (prod > u128(q)) return false;
        }
    }
    if (prod != u128(q - 1)) return false;
    // the witness
    if (c.witness < 2 || c.witness >= q) return false;
    if (PowModU64(c.witness, q - 1, q) != 1) return false;
    size_t si = 0;
    for (const auto& f : c.factors) {
        if (PowModU64(c.witness, (q - 1) / f.first, q) == 1) return false;
        if (f.first <= kSmallBound) {
            if (!IsSmallPrime(f.first)) return false;
        } else {
            if (si >= c.sub.size() || c.sub[si].q != f.first) return false;
            if (!VerifyPrattRec(c.sub[si], node_cap, nodes)) return false;
            ++si;
        }
    }
    return si == c.sub.size();
}
}  // namespace

bool VerifyPratt(const PrattCert& c, uint32_t node_cap, uint32_t* nodes) {
    uint32_t local = 0;
    uint32_t* cnt = nodes ? nodes : &local;
    *cnt = 0;
    return VerifyPrattRec(c, node_cap, cnt);
}

std::string RationalSettings::canonical_text() const {
    std::ostringstream o;
    o << "hz_rat/v1;trial_division_B=" << kSmallBound << ";theta_cap=" << theta_cap << ";theta_coeff_max="
      << (theta_coeff_max ? std::to_string(theta_coeff_max) : std::string("binom(d,2)")) << ";c1_probe_primes=" << c1_probe_primes
      << ";rho=nf::factor_u64(untrusted;seeds=39;r<=2^26);pratt_nodes=" << pratt_nodes << ";pratt_witness_cap=" << pratt_witness_cap
      << ";big_mr_rounds=" << mr_rounds << "(untrusted;probable);word_primes=CrtPrimes(<2^30,sized_from_bound,cap="
      << (word_prime_cap == (1u << 21) ? std::string("2^21") : std::to_string(word_prime_cap)) << ")";
    return o.str();
}

// =====================================================================================================================================
// factorisation from residues
// =====================================================================================================================================
namespace {
std::string HexU64(u64 v) {
    std::ostringstream o;
    o << "0x" << std::hex << v;
    return o.str();
}
// decimal text of |z| (untrusted use only: the finder's input)
std::string AbsDecimal(const ZExact& z) {
    // Horner on the digits in base 10^9 limbs (word arithmetic; the finder's input only)
    std::vector<uint32_t> limbs;   // little-endian base 1e9
    for (uint32_t k = z.significant; k-- > 0;) {
        u64 carry = z.abs_digits[k];
        for (auto& l : limbs) {
            const u64 t = u64(l) * z.primes[k] + carry;
            l = static_cast<uint32_t>(t % 1000000000ull);
            carry = t / 1000000000ull;
        }
        while (carry) { limbs.push_back(static_cast<uint32_t>(carry % 1000000000ull)); carry /= 1000000000ull; }
    }
    if (limbs.empty()) return "0";
    std::ostringstream o;
    o << limbs.back();
    for (size_t i = limbs.size() - 1; i-- > 0;) { o.width(9); o.fill('0'); o << limbs[i]; }
    return o.str();
}
// canonical hex of a decimal text (word arithmetic)
std::string DecimalToHex(const std::string& dec) {
    std::vector<uint32_t> limbs;   // base 2^32
    for (char ch : dec) {
        u64 carry = static_cast<u64>(ch - '0');
        for (auto& l : limbs) {
            const u64 t = u64(l) * 10u + carry;
            l = static_cast<uint32_t>(t);
            carry = t >> 32;
        }
        if (carry) limbs.push_back(static_cast<uint32_t>(carry));
    }
    if (limbs.empty()) return "0x0";
    std::ostringstream o;
    o << "0x" << std::hex << limbs.back();
    for (size_t i = limbs.size() - 1; i-- > 0;) { o.width(8); o.fill('0'); o << std::hex << limbs[i]; }
    return o.str();
}

// (hex value) mod m, m < 2^64 (the canonical hex "0x..." of a nonnegative integer)
u64 HexModU64(const std::string& hex, u64 m) {
    u64 r = 0;
    for (size_t i = 2; i < hex.size(); ++i) {
        const char h = hex[i];
        const u64 v = static_cast<u64>(h <= '9' ? h - '0' : h - 'a' + 10);
        r = static_cast<u64>((u128(r) * 16 + v) % m);
    }
    return r;
}

// |D| mod ell' for the residue r of D (sign known from the exact value)
uint32_t AbsResidue(uint32_t r, bool negative, uint32_t ell) { return negative ? (r == 0 ? 0 : ell - r) : r; }

// v_q(|D|) for a prime q < 2^64, |D| > 0, by residues. Direct while q^j < 2^64, then the cofactor route (01 Sec. 4, R-3).
uint32_t Valuation(const ZExact& D, const std::vector<uint32_t>& residues, u64 q) {
    ENGINE_CHECK_MSG(!D.is_zero(), "Valuation: D = 0");
    // the largest jmax with q^jmax < 2^64
    uint32_t jmax = 0;
    u64 Qmax = 1;
    while (Qmax <= UINT64_MAX / q) { Qmax *= q; ++jmax; }
    // the direct route on an exact Z: the largest j <= jmax with q^j | Z (Z mod q^j by Horner on its digits)
    auto direct = [&](const ZExact& Z) {
        uint32_t j = 0;
        u64 Q = 1;
        while (j < jmax) {
            const u64 Qn = Q * q;
            if (Z.abs_mod(Qn) != 0) break;
            Q = Qn;
            ++j;
        }
        return j;
    };
    uint32_t j = direct(D);
    if (j < jmax) return j;
    // q^jmax | D: the cofactor route (01 Sec. 4, R-3), one certified step per q^jmax: X = Z q^{-jmax} from residues, accepted iff
    // jmax bitlen(q) + bits(|X|) <= Bc - 2 (then q^jmax X = Z exactly, both below P / 4), then the direct route on X.
    std::vector<uint32_t> pr, zr;
    for (size_t i = 0; i < D.primes.size(); ++i) {
        if (D.primes[i] == q) continue;
        pr.push_back(D.primes[i]);
        zr.push_back(AbsResidue(residues[i], D.negative, D.primes[i]));
    }
    const u64 Bc = CoveredBits(pr, pr.size());
    ENGINE_CHECK_MSG(D.bound_bits + 2 <= Bc, "Valuation: the cofactor route needs the bound inside the cover");
    uint32_t total = 0;
    for (;;) {
        std::vector<uint32_t> xr(pr.size());
        for (size_t i = 0; i < pr.size(); ++i)
            xr[i] = static_cast<uint32_t>((u128(zr[i]) * InvMod(PowModU64(q % pr[i], jmax, pr[i]), pr[i])) % pr[i]);
        const ZExact X = ZFromResidues(pr, xr, D.bound_bits);
        const u64 lhs = u64(jmax) * BitLen64(q) + X.abs_bits_upper();
        ENGINE_CHECK_MSG(!X.negative && !X.is_zero() && lhs + 2 <= Bc,
                         "Valuation: the cofactor route lacks precision (q^jmax | Z is known; extend the prime list)");
        total += jmax;
        const uint32_t jx = direct(X);
        if (jx < jmax) return total + jx;
        zr = std::move(xr);
    }
}

}  // namespace

// The production verifier of a claimed factorisation of a word-size c > 1 (the finder's output, or a validation control's claim):
// each claimed prime above B gets a Pratt certificate (found, then VerifyPratt), small ones are checked against the sieve; the
// certified part is re-divided out of c and every exponent must equal the claim, else the whole claim is discarded (rest = c).
void CertifyWordClaim(uint64_t c, const std::vector<std::pair<uint64_t, unsigned>>& claim, const RationalSettings& s,
                      std::vector<PrimePower>& certified, uint64_t* rest) {
    *rest = c;
    std::vector<std::pair<u64, unsigned>> ok_claim;
    for (const auto& f : claim) {
        if (f.first < 2 || f.second == 0) continue;
        if (f.first <= kSmallBound) {
            if (IsSmallPrime(f.first)) ok_claim.push_back(f);
            continue;
        }
        PrattCert pc;
        if (finder::FindPratt(f.first, s.pratt_witness_cap, s.pratt_nodes, &pc) && VerifyPratt(pc, s.pratt_nodes))
            ok_claim.push_back(f);
    }
    // re-divide: the certified part must divide c exactly, with the claimed exponents
    u64 rem = c;
    std::vector<PrimePower> out;
    for (const auto& f : ok_claim) {
        uint32_t e = 0;
        while (rem % f.first == 0) { rem /= f.first; ++e; }
        if (e != f.second) { rem = c; out.clear(); break; }   // a wrong claim: discard the finder's answer
        PrimePower pp;
        pp.q = f.first;
        pp.q_hex = HexU64(f.first);
        pp.bits = BitLen64(f.first);
        pp.e = e;
        pp.proof = f.first <= kSmallBound ? "small" : "pratt";
        out.push_back(pp);
    }
    certified.insert(certified.end(), out.begin(), out.end());
    *rest = rem;
}

namespace {
// The finder on a word-size cofactor c > 1: returns the certified prime powers (pratt) and the uncertified remainder.
void SplitWordCofactor(u64 c, const RationalSettings& s, std::vector<PrimePower>& certified, u64* rest) {
    *rest = c;
    std::vector<std::pair<u64, unsigned>> claim;
    if (!finder::FindWordFactors(c, claim)) return;
    CertifyWordClaim(c, claim, s, certified, rest);
}

// the cofactor test of 01 Sec. 4: no factor <= B (trial division done), c < B^3, Fermat-composite, not a square
bool SquarefreeBySize(u64 c) {
    const u64 B3 = kSmallBound * kSmallBound * kSmallBound;
    if (c <= kSmallBound || c >= B3) return false;
    if (PowModU64(2, c - 1, c) == 1) return false;   // no Fermat witness at 2: not decided
    u64 r = 0;
    { u64 lo = 0, hi = 1u << 31; while (lo < hi) { const u64 mid = (lo + hi + 1) / 2; if (mid * mid <= c) lo = mid; else hi = mid - 1; } r = lo; }
    return r * r != c;
}

void SortFactors(std::vector<PrimePower>& f) {
    std::stable_sort(f.begin(), f.end(), [](const PrimePower& a, const PrimePower& b) {
        const bool pa = a.proof == "probable", pb = b.proof == "probable";
        if (pa != pb) return !pa;
        if (a.bits != b.bits) return a.bits < b.bits;
        return a.q_hex < b.q_hex;
    });
}
}  // namespace

Factored FactorFromResidues(const std::vector<uint32_t>& primes, const std::vector<uint32_t>& residues, uint64_t bound_bits,
                            const RationalSettings& s, const std::vector<uint64_t>& pool) {
    Factored out;
    const ZExact D = ZFromResidues(primes, residues, bound_bits);
    out.value_hex = D.hex();
    out.value_bits = D.abs_bits_upper();
    if (D.is_zero()) {
        out.zero = true;
        out.cofactor_hex = "0x0";
        out.cofactor_bits = 0;
        out.cofactor_test = "unfactored";
        return out;
    }
    // 1. trial division to B, valuations by residues
    std::vector<std::pair<u64, uint32_t>> pe;
    for (uint32_t q : SmallPrimes()) {
        if (D.abs_mod(q) != 0) continue;
        pe.emplace_back(q, Valuation(D, residues, q));
    }
    std::vector<uint64_t> pool_hits;
    for (uint64_t q : pool) {
        ENGINE_CHECK_MSG(q > kSmallBound, "FactorFromResidues: pool primes lie above B");
        if (D.abs_mod(q) != 0) continue;
        pe.emplace_back(q, Valuation(D, residues, q));
        pool_hits.push_back(q);
    }
    // 2. the cofactor C = |D| / prod q^e by residues, certified: sum e bitlen(q) + bits(C) <= Bc - 2 (then prod q^e C = |D|)
    auto cofactor_of = [&](const std::vector<std::pair<u64, uint32_t>>& fs, ZExact* C) {
        std::vector<uint32_t> pr, cr;
        for (size_t i = 0; i < primes.size(); ++i) {
            bool skip = false;
            for (const auto& f : fs) if (f.first == primes[i]) skip = true;
            if (skip) continue;
            u64 r = AbsResidue(residues[i], D.negative, primes[i]);
            for (const auto& f : fs) r = (u128(r) * InvMod(PowModU64(f.first % primes[i], f.second, primes[i]), primes[i])) % primes[i];
            pr.push_back(primes[i]);
            cr.push_back(static_cast<uint32_t>(r));
        }
        const u64 Bc = CoveredBits(pr, pr.size());
        ENGINE_CHECK_MSG(bound_bits + 2 <= Bc, "FactorFromResidues: the cover is too small for the cofactor check");
        *C = ZFromResidues(pr, cr, bound_bits);
        u64 lhs = C->abs_bits_upper();
        for (const auto& f : fs) lhs += u64(f.second) * BitLen64(f.first);
        ENGINE_CHECK_MSG(!C->negative && lhs + 2 <= Bc, "FactorFromResidues: the cofactor identity is not certified (precision)");
    };
    ZExact C;
    cofactor_of(pe, &C);
    for (const auto& f : pe) {
        PrimePower pp;
        pp.q = f.first;
        pp.q_hex = HexU64(f.first);
        pp.bits = BitLen64(f.first);
        pp.e = f.second;
        pp.proof = f.first <= kSmallBound ? "small" : "pratt";   // pool primes: certified (Pratt) where they were found
        out.factors.push_back(pp);
    }
    // 3. the cofactor
    u64 cw = 0;
    if (C.abs_u64(&cw)) {
        if (cw == 1) {
            out.cofactor_hex = "0x1";
            out.cofactor_bits = 1;
            out.cofactor_test = "one";
        } else {
            std::vector<PrimePower> cert;
            u64 rest = cw;
            SplitWordCofactor(cw, s, cert, &rest);
            // the certified word factors, valuations re-read by residues (must equal the finder's: re-division above)
            for (auto& pp : cert) {
                const uint32_t v = Valuation(D, residues, pp.q);
                ENGINE_CHECK_MSG(v == pp.e, "FactorFromResidues: the finder's exponent differs from the residue valuation");
                out.factors.push_back(pp);
            }
            out.cofactor_hex = HexU64(rest);
            out.cofactor_bits = BitLen64(rest);
            out.cofactor_test = rest == 1 ? "one" : (SquarefreeBySize(rest) ? "squarefree_by_size" : "unfactored");
        }
    } else {
        out.cofactor_hex = C.abs_hex();
        out.cofactor_bits = C.abs_bits_upper();
        out.cofactor_test = "unfactored";
        // above 2^64: the untrusted BigInt Miller--Rabin, `probable` (Q1(a)); the identity q^e = C is checked by residues
        const finder::BigProbable bp = finder::BigProbablePrime(AbsDecimal(C), s.mr_rounds);
        if (bp.found) {
            bool ok = true;
            for (size_t i = 0; i < C.primes.size() && ok; ++i) {
                const uint32_t ell = C.primes[i];
                const u64 qr = integral::ReduceDecimalMod(bp.root_decimal, ell);
                const u64 cr = DigitsModQ(C.abs_digits, C.primes, C.significant, ell);
                ok = PowModU64(qr, bp.e, ell) == cr;
            }
            if (ok) {
                PrimePower pp;
                pp.q = 0;
                pp.q_hex = DecimalToHex(bp.root_decimal);
                {   // exact bit length from the leading hex digit
                    const char h = pp.q_hex[2];
                    const uint32_t v = static_cast<uint32_t>(h <= '9' ? h - '0' : h - 'a' + 10);
                    pp.bits = static_cast<uint32_t>((pp.q_hex.size() - 3) * 4) + BitLen64(v);
                }
                pp.e = bp.e;
                pp.proof = "probable";
                // q^e = C exactly: q^e < 2^{e bits(q)} and C < 2^{bound} both below P / 4, and q^e = C mod P (the residues above)
                if (u64(pp.e) * pp.bits + 2 <= C.covered_bits && bound_bits + 2 <= C.covered_bits) {
                    // a word-size root (e >= 2): certify it by Pratt; then C = q^e is a certified factorisation (audit F3)
                    PrattCert pc;
                    if (pp.bits <= 64 && pp.e >= 2) {
                        const u64 qw = std::stoull(pp.q_hex.substr(2), nullptr, 16);
                        if (qw > kSmallBound && finder::FindPratt(qw, s.pratt_witness_cap, s.pratt_nodes, &pc) &&
                            VerifyPratt(pc, s.pratt_nodes)) {
                            ENGINE_CHECK_MSG(Valuation(D, residues, qw) == pp.e,
                                             "FactorFromResidues: the certified root's valuation differs from q^e = C");
                            pp.q = qw;
                            pp.q_hex = HexU64(qw);
                            pp.proof = "pratt";
                            out.cofactor_hex = "0x1";
                            out.cofactor_bits = 1;
                            out.cofactor_test = "one";
                        }
                    }
                    out.factors.push_back(pp);
                }
            }
        }
    }
    SortFactors(out.factors);
    return out;
}

std::string FactoredText(const Factored& f) {
    if (f.zero) return "0";
    std::ostringstream o;
    bool first = true;
    for (const auto& pp : f.factors) {
        if (pp.proof == "probable") continue;
        if (!first) o << "*";
        first = false;
        o << (pp.q ? std::to_string(pp.q) : pp.q_hex);
        if (pp.e > 1) o << "^" << pp.e;
    }
    if (f.cofactor_test != "one") {
        if (!first) o << "*";
        first = false;
        o << "[c:" << f.cofactor_hex << ":" << f.cofactor_test;
        for (const auto& pp : f.factors)
            if (pp.proof == "probable") o << ":probable(" << pp.q_hex << (pp.e > 1 ? "^" + std::to_string(pp.e) : "") << ")";
        o << "]";
    }
    if (first) o << "1";
    return o.str();
}

// =====================================================================================================================================
// the gcd of several factored integers (only through word-size primes, Q1(a))
// =====================================================================================================================================
namespace {
struct Valued {
    ZExact D;
    std::vector<uint32_t> residues;
    Factored F;
};
Factored GcdOf(const std::vector<const Valued*>& xs, const RationalSettings& s) {
    Factored g;
    g.value_hex = "";
    ENGINE_CHECK_MSG(!xs.empty(), "GcdOf: empty");
    std::set<u64> Q;
    for (const Valued* x : xs) {
        ENGINE_CHECK_MSG(!x->F.zero, "GcdOf: a zero value");
        for (const auto& pp : x->F.factors) if (pp.proof != "probable") Q.insert(pp.q);
    }
    // the base: the value with the smallest cofactor (complete first), first in order on ties
    size_t jb = 0;
    for (size_t j = 1; j < xs.size(); ++j) {
        const Factored &a = xs[j]->F, &b = xs[jb]->F;
        if ((a.complete() && !b.complete()) || (a.complete() == b.complete() && a.cofactor_bits < b.cofactor_bits)) jb = j;
    }
    // the word-size cofactor of the base, reduced by gcds with every other value (word gcd of c with |D_j| mod c)
    const Factored& base = xs[jb]->F;
    std::vector<PrimePower> extra;
    u64 gc = 0;
    bool word_cofactor = false;
    if (base.cofactor_bits <= 64) {
        ZExact tmp;   // parse the base cofactor (hex, < 2^64)
        gc = std::stoull(base.cofactor_hex.substr(2), nullptr, 16);
        word_cofactor = true;
        for (const Valued* x : xs) if (gc > 1) gc = Gcd(gc, x->D.abs_mod(gc));
        for (u64 q : Q) while (gc > 1 && gc % q == 0) gc /= q;
        if (gc > 1) {
            u64 rest = gc;
            SplitWordCofactor(gc, s, extra, &rest);
            for (auto& pp : extra) Q.insert(pp.q);
            gc = rest;
        }
    }
    for (u64 q : Q) {
        uint32_t e = UINT32_MAX;
        for (const Valued* x : xs) e = std::min(e, Valuation(x->D, x->residues, q));
        if (e == 0) continue;
        PrimePower pp;
        pp.q = q;
        pp.q_hex = HexU64(q);
        pp.bits = BitLen64(q);
        pp.e = e;
        pp.proof = q <= kSmallBound ? "small" : "pratt";
        g.factors.push_back(pp);
    }
    if (word_cofactor) {
        g.cofactor_hex = HexU64(gc);
        g.cofactor_bits = BitLen64(gc);
        g.cofactor_test = gc == 1 ? "one" : (SquarefreeBySize(gc) ? "squarefree_by_size" : "unfactored");
    } else {
        // the base's cofactor C is the gcd's cofactor iff C | D_j for every j and no prime of Q divides C (then the gcd's part prime
        // to Q divides C and is divided by it). C | D_j by the residue cofactor route: X = |D_j| C^{-1} from residues, accepted iff
        // bits(C) + bits(X) + 2 <= covered bits (then C X = |D_j| exactly, both below P / 4). Else C is only an upper bound (audit F4).
        bool exact = true;
        for (u64 q : Q) if (exact && HexModU64(base.cofactor_hex, q) == 0) exact = false;
        for (const Valued* x : xs) {
            if (!exact) break;
            const ZExact& Dj = x->D;
            std::vector<uint32_t> pr, xr;
            for (size_t i = 0; i < Dj.primes.size() && exact; ++i) {
                const uint32_t ell = Dj.primes[i];
                const u64 cm = HexModU64(base.cofactor_hex, ell);
                if (cm == 0) { exact = false; break; }
                const u64 zr = AbsResidue(x->residues[i], Dj.negative, ell);
                pr.push_back(ell);
                xr.push_back(static_cast<uint32_t>((u128(zr) * InvMod(cm, ell)) % ell));
            }
            if (!exact) break;
            const u64 Bc = CoveredBits(pr, pr.size());
            const ZExact X = ZFromResidues(pr, xr, Dj.bound_bits);
            exact = !X.negative && !X.is_zero() && base.cofactor_bits + X.abs_bits_upper() + 2 <= Bc;
        }
        g.cofactor_hex = base.cofactor_hex;
        g.cofactor_bits = base.cofactor_bits;
        g.cofactor_test = exact ? "unfactored" : "upper_bound";
        if (exact)
            for (const auto& pp : base.factors) if (pp.proof == "probable") g.factors.push_back(pp);
    }
    SortFactors(g.factors);
    return g;
}
}  // namespace

// =====================================================================================================================================
// word-prime runs
// =====================================================================================================================================
std::vector<uint32_t> WordPrimes(uint32_t count, uint32_t N) {
    std::vector<uint32_t> out;
    uint32_t want = count;
    while (out.size() < count) {
        out.clear();
        for (uint32_t p : integral::CrtPrimes(want))
            if (N % p != 0) out.push_back(p);
        want += count - static_cast<uint32_t>(std::min<size_t>(out.size(), count)) + 1;
    }
    out.resize(count);
    return out;
}

void ExtendWordPrimes(std::vector<uint32_t>& v, size_t count, uint32_t N) {
    // CrtPrimes' order (descending from 2^30 - 1 over odd x, IsPrimeU32), the divisors of N skipped: v stays a prefix of WordPrimes
    if (v.size() >= count) return;
    uint32_t x = v.empty() ? (1u << 30) - 1u : v.back() - 2u;
    while (v.size() < count) {
        ENGINE_CHECK_MSG(x > (1u << 29), "ExtendWordPrimes: ran below 2^29 (count too large)");
        if (integral::IsPrimeU32(x) && N % x != 0) v.push_back(x);
        x -= 2u;
    }
}

PrimeRun RunAtPrime(uint32_t N, uint32_t n, const std::vector<uint32_t>& P, uint32_t ell) {
    ENGINE_CHECK_MSG(ell > n + 1 && ell >= 5 && N % ell != 0, "RunAtPrime: ell' > n + 1, ell' >= 5, ell' prime to N");
    const hecke::level::Gamma0LevelN lvl(N);
    const engine::ModCtx<uint32_t> ctx(ell);
    const auto mats = hecke::level::CokernelHeckeMatrices(lvl, n, ctx, P, hecke::level::CokernelRoute::kSparse);
    PrimeRun r;
    r.ell = ell;
    for (uint32_t p : P) {
        const auto it = mats.find(p);
        ENGINE_CHECK_MSG(it != mats.end(), "RunAtPrime: operator missing");
        const hecke::level::HMat& A = it->second;
        ENGINE_CHECK_MSG(A.field && A.rows == A.cols, "RunAtPrime: a square field matrix");
        if (r.mats.empty()) r.h1 = A.rows;
        ENGINE_CHECK_MSG(A.rows == r.h1, "RunAtPrime: operators of different size");
        std::vector<u64> M(A.a.size());
        for (size_t i = 0; i < M.size(); ++i) M[i] = A.a[i] % ell;
        r.mats.push_back(std::move(M));
    }
    return r;
}

uint32_t H1DimAtPrime(uint32_t N, uint32_t n, uint32_t ell) {
    ENGINE_CHECK_MSG(ell > n + 1 && ell >= 5 && N % ell != 0, "H1DimAtPrime: ell' > n + 1, ell' >= 5, ell' prime to N");
    const hecke::level::Gamma0LevelN lvl(N);
    const engine::ModCtx<uint32_t> ctx(ell);
    return hecke::level::SparseCokernel::Build(lvl, n, ctx, false).h1();
}

std::vector<uint64_t> CharpolyOfCombination(const PrimeRun& r, const std::vector<uint32_t>& coeffs) {
    ENGINE_CHECK_MSG(coeffs.size() == r.mats.size(), "CharpolyOfCombination: one coefficient per operator");
    const u64 ell = r.ell;
    std::vector<u64> A(size_t(r.h1) * r.h1, 0);
    for (size_t o = 0; o < coeffs.size(); ++o) {
        if (!coeffs[o]) continue;
        const u64 c = coeffs[o] % ell;
        const std::vector<u64>& M = r.mats[o];
        for (size_t i = 0; i < A.size(); ++i) A[i] = (A[i] + c * M[i]) % ell;
    }
    return CharpolyHessenberg(A, r.h1, ell);
}

std::vector<uint64_t> CharpolyOfCombinationBerkowitz(const PrimeRun& r, const std::vector<uint32_t>& coeffs) {
    ENGINE_CHECK_MSG(coeffs.size() == r.mats.size(), "CharpolyOfCombination: one coefficient per operator");
    hecke::level::HMat T;
    T.rows = T.cols = r.h1;
    T.modulus = T.ell = r.ell;
    T.field = true;
    T.row_exp.assign(r.h1, 1);
    T.a.assign(size_t(r.h1) * r.h1, 0);
    for (size_t o = 0; o < coeffs.size(); ++o) {
        if (!coeffs[o]) continue;
        const u64 c = coeffs[o] % r.ell;
        for (size_t i = 0; i < T.a.size(); ++i) T.a[i] = (T.a[i] + c * r.mats[o][i]) % r.ell;
    }
    if (r.h1 == 0) return {1};
    return hecke::level::HCharpoly(T);
}

namespace {
// runs at primes[from, to) in parallel (index-ordered storage; the thread count changes nothing)
void RunBatch(uint32_t N, uint32_t n, const std::vector<uint32_t>& P, const std::vector<uint32_t>& primes, size_t from, size_t to,
              uint32_t threads, std::vector<PrimeRun>& out) {
    out.resize(std::max(out.size(), to));
    std::atomic<size_t> next(from);
    std::vector<std::exception_ptr> errs(std::max<uint32_t>(threads, 1));
    auto work = [&](size_t w) {
        try {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= to) break;
                out[i] = RunAtPrime(N, n, P, primes[i]);
            }
        } catch (...) {
            errs[w] = std::current_exception();
        }
    };
    const uint32_t T = std::max<uint32_t>(threads, 1);
    if (T == 1) {
        work(0);
    } else {
        std::vector<std::thread> pool;
        for (uint32_t w = 0; w < T; ++w) pool.emplace_back(work, w);
        for (auto& t : pool) t.join();
    }
    for (auto& e : errs) if (e) std::rethrow_exception(e);
}

// fn(i) for i in [from, to) on T threads (dynamic schedule; fn writes index-owned storage only, so no value depends on T)
template <class F>
void ParallelFor(size_t from, size_t to, uint32_t T, F fn) {
    if (T <= 1 || to - from <= 1) {
        for (size_t i = from; i < to; ++i) fn(i);
        return;
    }
    std::atomic<size_t> next(from);
    std::vector<std::exception_ptr> errs(T);
    auto work = [&](uint32_t w) {
        try {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= to) break;
                fn(i);
            }
        } catch (...) {
            errs[w] = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    const uint32_t W = static_cast<uint32_t>(std::min<size_t>(T, to - from));
    for (uint32_t w = 1; w < W; ++w) {
        try {
            pool.emplace_back(work, w);
        } catch (const std::system_error&) {
            break;   // no thread available: the started workers and this one take every index (no value depends on W)
        }
    }
    work(0);
    for (auto& t : pool) t.join();
    for (auto& e : errs) if (e) std::rethrow_exception(e);
}

// the growable, capped word-prime list (HZ-RAT-PERF (a))
struct WordPrimeList {
    uint32_t N, cap;
    std::vector<uint32_t> v;
    WordPrimeList(uint32_t N_, uint32_t cap_) : N(N_), cap(cap_) {}
    void ensure(size_t count) {
        if (count <= v.size()) return;
        if (count > cap) throw WordPrimeCapRefusal(cap);
        ExtendWordPrimes(v, std::min<size_t>(cap, std::max(count, v.size() + v.size() / 2)), N);
    }
};

// an integer as 32-bit limbs (little endian) and a sign: the descent's chi_Z coefficients
struct BigZ {
    bool neg = false;
    std::vector<uint32_t> limbs;   // no leading zero limb; zero = {}
};
bool ParseBigZ(const std::string& t, BigZ* z) {
    *z = BigZ();
    size_t i = 0;
    if (i < t.size() && t[i] == '-') { z->neg = true; ++i; }
    if (i >= t.size()) return false;
    if (t.compare(i, 2, "0x") == 0) {
        i += 2;
        if (i >= t.size()) return false;
        for (size_t k = t.size(); k > i;) {   // 8 hex digits per limb, from the low end
            const size_t lo = k >= i + 8 ? k - 8 : i;
            uint32_t limb = 0;
            for (size_t j = lo; j < k; ++j) {
                const char c = t[j];
                uint32_t d;
                if (c >= '0' && c <= '9') d = c - '0';
                else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
                else return false;
                limb = (limb << 4) | d;
            }
            z->limbs.push_back(limb);
            k = lo;
        }
    } else {
        u64 v = 0;
        for (; i < t.size(); ++i) {
            if (t[i] < '0' || t[i] > '9') return false;
            const u64 d = static_cast<u64>(t[i] - '0');
            if (v > (~0ull - d) / 10) return false;   // the descent prints decimals below 2^63 only
            v = v * 10 + d;
        }
        if (v) z->limbs.push_back(static_cast<uint32_t>(v));
        if (v >> 32) z->limbs.push_back(static_cast<uint32_t>(v >> 32));
    }
    while (!z->limbs.empty() && z->limbs.back() == 0) z->limbs.pop_back();
    if (z->limbs.empty()) z->neg = false;
    return true;
}
// the coefficients mod ell (Horner on the limbs from the top, Barrett)
std::vector<u64> ChiModEll(const std::vector<BigZ>& c, u64 ell) {
    const Barrett31 br(ell);
    std::vector<u64> out(c.size());
    for (size_t j = 0; j < c.size(); ++j) {
        u64 r = 0;
        const std::vector<uint32_t>& L = c[j].limbs;
        for (size_t k = L.size(); k-- > 0;) r = br.reduce((r << 32) | L[k]);
        out[j] = (c[j].neg && r) ? ell - r : r;
    }
    return out;
}

Poly ToPoly(const std::vector<u64>& c, u64 ell) {
    Poly p(c.begin(), c.end());
    for (auto& x : p) x %= ell;
    Trim(p);
    return p;
}

std::string JoinHex(const std::vector<std::string>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
    return s + "]";
}
std::string FactoredFull(const Factored& f) {
    std::ostringstream o;
    o << "{value=" << f.value_hex << ";text=" << FactoredText(f) << ";factors=";
    for (const auto& pp : f.factors) o << "(" << pp.q_hex << "," << pp.e << "," << pp.proof << ")";
    o << ";cofactor=" << f.cofactor_hex << ";test=" << f.cofactor_test << "}";
    return o.str();
}
}  // namespace

std::string RationalResult::text() const {
    std::ostringstream o;
    o << "hz_rat/v1 N=" << N << " n=" << n << " algebra=" << algebra << " P=[";
    for (size_t i = 0; i < P.size(); ++i) o << (i ? "," : "") << P[i];
    o << "]";
    if (nU) o << " nU=" << nU;   // (S26 Z1) RationalStepFull only: the text of a RationalStep result is unchanged
    o << " b1=" << b1 << " e_k=" << e_k << " d=" << d << " d_an=" << d_an << " d_target=" << d_target << " d_source=" << d_source
      << " eis_rational=" << (eis_rational ? 1 : 0) << " C1=" << c1_status << " theta_index=" << theta_index << "\n";
    for (size_t i = 0; i < thetas.size(); ++i) {
        const ThetaEntry& t = thetas[i];
        o << "theta[" << i << "] c=[";
        for (size_t j = 0; j < t.coeffs.size(); ++j) o << (j ? "," : "") << t.coeffs[j];
        o << "] primitive=" << (t.primitive ? 1 : 0);
        if (t.primitive) {
            o << " lambda=" << t.lambda_hex << " f=" << JoinHex(t.f) << " disc=" << FactoredFull(t.disc);
            if (eis_rational) o << " eis_res=" << FactoredFull(t.eis_res);
        }
        o << "\n";
    }
    o << "discriminants_used=[";
    for (size_t i = 0; i < discriminants_used.size(); ++i) o << (i ? "," : "") << discriminants_used[i];
    o << "] disc_gcd=" << FactoredFull(disc_gcd) << " C2=" << c2_status << " S_cand=[";
    for (size_t i = 0; i < S_cand.size(); ++i) o << (i ? "," : "") << S_cand[i].ell << ":" << S_cand[i].v_disc << ":" << S_cand[i].reason;
    o << "]\n";
    o << "gP_ops=[";
    for (size_t i = 0; i < gP_ops.size(); ++i) o << (i ? "," : "") << gP_ops[i];
    o << "] gP_running=[";
    for (size_t i = 0; i < gP_running.size(); ++i) o << (i ? "," : "") << gP_running[i];
    o << "] gP=" << FactoredFull(gP) << " gP_status=" << gP_status << " cusp_part=" << JoinHex(cusp_part) << "\n";
    o << "settings=" << settings << "\n";
    return o.str();
}
uint64_t RationalResult::digest() const { return engine::FnvBytes(engine::kFnvOffset64, text()); }

// =====================================================================================================================================
// the rational step
// =====================================================================================================================================
namespace {
// (slot-26 Z1, UPZ_design.md Sec. 1.3 R6) the body of RationalStep on an OPERATOR LIST: P = the T_p (p !| N) followed by the last nU
// entries, the U_p (p | N, ascending; RationalStepFull).  nU = 0 is RationalStep verbatim: every statement below reads P exactly as
// the pre-Z1 body did (the 141-record recompute of the Z1 merge gate checks it, HZUP_RECOMPUTE).  nU > 0 (full level N): the target is
// d = dim M_k(Gamma_0(N)) (the full (C1), item 8 (d')), and no rational Eisenstein system is assumed (eis_rational = false: the
// Eisenstein eigenvalue formula 1 + p^{k-1} is that of T_p, not of U_p).  The word-prime residues of U_p are RunAtPrime's, the
// shipped Heilbronn operator ([C, prop:heilbronn](2), p | N stated); the eigenvalue bound |a| <= p^{k-1} < 1 + p^{k-1} holds for U_p
// on H^1 as for T_p, so the bit bounds below are unchanged.
RationalResult RationalStepImpl(uint32_t N, uint32_t n, const std::vector<uint32_t>& P, size_t nU, const RationalSettings& s,
                                const RationalInputs* in, std::vector<PrimeRun>* keep = nullptr) {
    if (N == 0) throw std::invalid_argument("RationalStep: N >= 1");
    if (n % 2) throw std::invalid_argument("RationalStep: n even (H^1 = 0 for n odd)");
    if (P.size() <= nU || P.size() - nU > 6) throw std::invalid_argument("RationalStep: 1 <= |P| <= 6");
    for (size_t i = 0; i < P.size(); ++i) {
        if (i + nU >= P.size()) {   // (Z1) a U_p: a prime dividing N, ascending
            if (!integral::IsPrimeU32(P[i]) || N % P[i] != 0) throw std::invalid_argument("RationalStepFull: U_p needs a prime p | N");
            if (i + nU > P.size() && P[i] <= P[i - 1]) throw std::invalid_argument("RationalStepFull: the U_p ascending, distinct");
            continue;
        }
        if (!integral::IsPrimeU32(P[i]) || P[i] > 97) throw std::invalid_argument("RationalStep: P of primes <= 97");
        if (N % P[i] == 0) throw std::invalid_argument("RationalStep: p | N (T_p only; U_p is HZ-UTAG)");
        for (size_t j = 0; j < i; ++j) if (P[i] == P[j]) throw std::invalid_argument("RationalStep: repeated prime");
    }
    const bool fullN = nU > 0;   // (Z1) the full algebra at N > 1
    const ScopedGarnerThreads garner_threads(s.threads);
    RationalResult R;
    R.N = N;
    R.n = n;
    R.P = P;
    R.nU = static_cast<uint32_t>(nU);
    R.settings = s.canonical_text();
    R.algebra = N == 1 || fullN ? "full" : "anemic";
    const uint32_t k = n + 2;
    R.d = DimMk(k, N);
    R.e_k = static_cast<int64_t>(Gamma0Cusps(N)) - (k == 2 ? 1 : 0);
    R.d_an = DimAnemic(k, N);
    R.d_target = N == 1 || fullN ? R.d : R.d_an;
    R.d_source = N == 1 || fullN ? "hES_N" : "hES_N_an";
    R.eis_rational = EisAnemicCount(k, N) == 1 && !fullN;
    const uint32_t D = static_cast<uint32_t>(R.d_target);
    ENGINE_CHECK_MSG(D >= 1, "RationalStep: d_target >= 1");

    // the word primes: CrtPrimes order without the divisors of N, prefix-stable, grown on demand (HZ-RAT-PERF (a)): sized from the
    // certified targets once they are known, never beyond s.word_prime_cap (then refused(word_prime_cap), WordPrimeCapRefusal)
    WordPrimeList primes(N, s.word_prime_cap);
    std::vector<PrimeRun> runs;
    size_t computed = 0;
    auto ensure = [&](size_t upto) {
        if (upto <= computed) return;
        size_t batch = std::max<size_t>(upto, computed + std::max<uint32_t>(s.threads, 1) * 2);
        if (batch > s.word_prime_cap) batch = std::max<size_t>(upto, s.word_prime_cap);   // the look-ahead never causes the refusal
        primes.ensure(batch);
        RunBatch(N, n, P, primes.v, computed, batch, s.threads, runs);
        for (size_t i = computed; i < batch; ++i) {
            if (i == 0) R.b1 = runs[0].h1;
            ENGINE_CHECK_MSG(runs[i].h1 == R.b1, "RationalStep: dim H^1 differs between word primes");
        }
        computed = batch;
    };
    ensure(s.c1_probe_primes);
    ENGINE_CHECK_MSG(static_cast<int64_t>(R.b1) + R.e_k == 2 * R.d,
                     "RationalStep: d != (b1 + e_k) / 2 ([C, prop:heckeZrat] (3) fails: model or formula)");

    // (HZ-RAT-PERF (b2)) chi_Z(T_p | H^1/tors) from the descent, when the caller hands it in: at every word prime ell' (> n + 1, prime
    // to 6N) H^1 over F_ell' is L (x) F_ell' ([C, prop:crtcharpolyN](2)), so chi(T_p mod ell') = chi_Z(T_p) mod ell'.  Compared with
    // the matrices' charpoly at every probe prime of every cell (a defect, never a value, if they differ); used for a unit theta = T_p.
    std::map<uint32_t, std::vector<BigZ>> chi;   // p -> b1 + 1 coefficients, low to high
    if (in) {
        for (const auto& [p, coeffs] : in->chi_z) {
            const auto it = std::find(P.begin(), P.end(), p);
            if (it == P.end() || coeffs.size() != size_t(R.b1) + 1) continue;
            std::vector<BigZ> c;
            bool ok = true;
            for (const std::string& t : coeffs) {
                BigZ z;
                if (!ParseBigZ(t, &z)) { ok = false; break; }
                c.push_back(std::move(z));
            }
            if (!ok || !(c.back().limbs.size() == 1 && c.back().limbs[0] == 1 && !c.back().neg)) continue;
            chi[p] = std::move(c);
        }
        for (const auto& [p, c] : chi) {
            const uint32_t o = static_cast<uint32_t>(std::find(P.begin(), P.end(), p) - P.begin());
            std::vector<uint32_t> unit(P.size(), 0);
            unit[o] = 1;
            for (uint32_t i = 0; i < s.c1_probe_primes; ++i) {
                const u64 ell = runs[i].ell;
                ENGINE_CHECK_MSG(ToPoly(ChiModEll(c, ell), ell) == ToPoly(CharpolyOfCombination(runs[i], unit), ell),
                                 "RationalStep: chi_Z(T_p) of the descent mod ell' differs from the charpoly of T_p over F_ell' (a defect)");
            }
        }
    }

    // ---- (C1): the theta search, count-capped, lexicographic ----
    const uint32_t m = static_cast<uint32_t>(P.size());
    const uint32_t cmax = s.theta_coeff_max ? s.theta_coeff_max : std::max<uint32_t>(1, D * (D - 1) / 2);
    std::vector<std::vector<uint32_t>> cands;
    for (uint32_t i = 0; i < m; ++i) {
        std::vector<uint32_t> c(m, 0);
        c[i] = 1;
        cands.push_back(c);
    }
    auto test_c1 = [&](const std::vector<uint32_t>& c) {
        int best = -1;
        for (uint32_t i = 0; i < s.c1_probe_primes; ++i) {
            const u64 ell = runs[i].ell;
            best = std::max(best, Deg(Radical(ToPoly(CharpolyOfCombination(runs[i], c), ell), ell)));
        }
        ENGINE_CHECK_MSG(best <= static_cast<int>(D), "RationalStep: deg rad > d_target (the a-priori rank is wrong)");
        return best == static_cast<int>(D);
    };
    bool any_unit = false;
    for (const auto& c : cands) {
        if (R.thetas.size() >= s.theta_cap) break;
        ThetaEntry t;
        t.coeffs = c;
        t.primitive = test_c1(c);
        any_unit = any_unit || t.primitive;
        R.thetas.push_back(t);
    }
    if (!any_unit && m > 1) {
        // combinations in lexicographic order over {0..cmax}^m, skipping 0 and the units
        std::vector<uint32_t> c(m, 0);
        for (;;) {
            size_t i = m;
            while (i > 0) {
                --i;
                if (c[i] < cmax) { ++c[i]; break; }
                c[i] = 0;
                if (i == 0) { i = m + 1; break; }
            }
            if (i == m + 1) break;
            uint32_t nz = 0, sum = 0;
            for (uint32_t x : c) { nz += x != 0; sum += x; }
            if (nz == 0 || (nz == 1 && sum == 1)) continue;
            if (R.thetas.size() >= s.theta_cap) break;
            ThetaEntry t;
            t.coeffs = c;
            t.primitive = test_c1(c);
            R.thetas.push_back(t);
            if (t.primitive) break;
        }
    }
    for (size_t i = 0; i < R.thetas.size(); ++i)
        if (R.thetas[i].primitive) {
            if (R.theta_index < 0) R.theta_index = static_cast<int32_t>(i);
            R.discriminants_used.push_back(static_cast<uint32_t>(i));
        }
    R.c1_status = R.theta_index >= 0 ? "certified" : "open(cap)";
    if (R.theta_index < 0) {
        R.c2_status = "open(C1)";
        R.gP_status = R.eis_rational ? "open(C1)" : "general_case";
        if (keep) { runs.resize(computed); *keep = std::move(runs); }   // (HZ-EISUP) the retained runs, value-neutral
        return R;
    }

    // ---- per primitive theta: good primes, residues, exact values ----
    // (HZ-RAT-PERF (b3)) one walk over the word primes for every primitive theta at once, in index-ordered batches computed in parallel
    // (the per-prime matrices, the charpoly, rad, the residues), the accept / skip decisions taken afterwards in index order: the same
    // good primes in the same order as the sequential walk, hence the same residues at any thread count.  A prime's matrices are dropped
    // after its batch.  Running sums replace the quadratic CoveredBits / prefix scans.
    struct Need {
        uint64_t rb = 0, b_disc = 0, b_f = 0, b_eis = 0, target = 0;
    };
    std::vector<Valued> discV(R.thetas.size()), eisV(R.thetas.size());
    struct Acc {
        uint32_t ti = 0;
        Need nd;
        int32_t chi_op = -1;            // the operator index of a unit theta served by chi_Z, else -1
        bool cusp = false;              // theta_index with eis_rational: the cusp part's residues
        bool done = false;
        std::vector<uint32_t> gp, rdisc, reis, rlam;
        std::vector<std::vector<uint32_t>> rf, rc;   // per coefficient, the first kk_f primes
        uint64_t covered = 0;           // CoveredBits(gp)
    };
    std::vector<Acc> accs;
    uint64_t max_target = 0;
    for (uint32_t ti : R.discriminants_used) {
        const ThetaEntry& t = R.thetas[ti];
        Acc a;
        a.ti = ti;
        Need& nd = a.nd;
        u64 csum = 0, X = 0;
        for (uint32_t o = 0; o < m; ++o) {
            csum += t.coeffs[o];
            if (t.coeffs[o]) X = std::max<u64>(X, u64(n + 1) * BitLen64(P[o]));
        }
        nd.rb = BitLen64(csum) + X;                       // R = sum c_p (1 + p^{n+1}) < 2^rb
        nd.b_disc = u64(D) * (D - 1) * (nd.rb + 1);       // |disc f| <= (2R)^{D(D-1)}
        nd.b_f = u64(D) + u64(D) * nd.rb;                 // |coef| <= binom(D, i) R^i
        nd.b_eis = u64(D > 0 ? D - 1 : 0) * (nd.rb + 1);  // |c(lambda)| <= (2R)^{D-1}
        const u64 B = std::max({nd.b_disc, nd.b_f, nd.b_eis, nd.rb + 1});
        nd.target = 2 * B + 192 + (2 * B + 192) / 16;     // the valuation / cofactor routes' slack (header note)
        if (csum == 1)
            for (uint32_t o = 0; o < m; ++o)
                if (t.coeffs[o] == 1 && chi.count(P[o])) a.chi_op = static_cast<int32_t>(o);
        a.cusp = R.eis_rational && static_cast<int32_t>(ti) == R.theta_index;
        a.rf.assign(D + 1, {});
        if (a.cusp) a.rc.assign(D, {});
        max_target = std::max(max_target, nd.target);
        accs.push_back(std::move(a));
    }
    // (a) the list sized from the bound: about 29 bits per word prime, ~3.5 % bad primes measured; capped, grown on demand
    primes.ensure(std::min<u64>(s.word_prime_cap, (11 * max_target + 289) / 290 + 64));
    struct Cand {
        int deg = -1;
        uint32_t disc = 0, lam = 0, eis = 0;
        bool eis_root = true, cusp_ok = true;
        std::vector<uint32_t> f, cq;
    };
    const uint32_t T = std::max<uint32_t>(s.threads, 1);
    const size_t BATCH = std::max<size_t>(64, size_t(16) * T);
    size_t idx = 0;
    for (;;) {
        bool any = false, mats = false;
        for (const Acc& a : accs)
            if (!a.done) { any = true; mats = mats || a.chi_op < 0; }
        if (!any) break;
        if (idx >= s.word_prime_cap) throw WordPrimeCapRefusal(s.word_prime_cap);
        const size_t end = std::min<size_t>(idx + BATCH, s.word_prime_cap);
        primes.ensure(end);
        std::vector<std::vector<Cand>> cand(end - idx, std::vector<Cand>(accs.size()));
        ParallelFor(idx, end, T, [&](size_t i) {
            const u64 ell = primes.v[i];
            PrimeRun local;
            const PrimeRun* r = nullptr;
            if (mats) {
                if (i < computed) {
                    r = &runs[i];
                } else {
                    local = RunAtPrime(N, n, P, static_cast<uint32_t>(ell));
                    ENGINE_CHECK_MSG(local.h1 == R.b1, "RationalStep: dim H^1 differs between word primes");
                    r = &local;
                }
            } else if (i >= computed) {
                // (fix round 1, audit F4) the matrix route's per-prime dimension check, kept on the chi_Z route: the presentation alone
                // (no Hecke matrix) at every word prime, dim H^1(Gamma_0(N); Sym^n F_ell') = b1 ([C, prop:crtcharpolyN] (2))
                ENGINE_CHECK_MSG(H1DimAtPrime(N, n, static_cast<uint32_t>(ell)) == R.b1,
                                 "RationalStep: dim H^1 differs between word primes (chi_Z route)");
            }
            for (size_t a = 0; a < accs.size(); ++a) {
                const Acc& A = accs[a];
                if (A.done) continue;
                const ThetaEntry& t = R.thetas[A.ti];
                const Poly f = A.chi_op >= 0 ? Radical(ToPoly(ChiModEll(chi.at(P[A.chi_op]), ell), ell), ell)
                                             : Radical(ToPoly(CharpolyOfCombination(*r, t.coeffs), ell), ell);
                Cand& c = cand[i - idx][a];
                c.deg = Deg(f);
                if (c.deg != static_cast<int>(D)) continue;   // ell | disc f_theta
                c.disc = static_cast<uint32_t>(Discriminant(f, ell));
                c.f.assign(D + 1, 0);
                for (uint32_t j = 0; j <= D; ++j) c.f[j] = static_cast<uint32_t>(j < f.size() ? f[j] : 0);
                u64 lam = 0;
                for (uint32_t o = 0; o < m; ++o)
                    lam = (lam + (t.coeffs[o] % ell) * ((1 + PowModU64(P[o], n + 1, ell)) % ell)) % ell;
                c.lam = static_cast<uint32_t>(lam);
                if (R.eis_rational) {
                    c.eis_root = Eval(f, lam, ell) == 0;
                    c.eis = static_cast<uint32_t>(Eval(Derivative(f, ell), lam, ell));
                }
                if (A.cusp) {
                    Poly q, rr;
                    DivRem(f, Poly{(ell - lam) % ell, 1}, ell, q, rr);
                    c.cusp_ok = rr.empty();
                    c.cq.assign(D, 0);
                    for (uint32_t j = 0; j < D; ++j) c.cq[j] = static_cast<uint32_t>(j < q.size() ? q[j] : 0);
                }
            }
        });
        for (size_t a = 0; a < accs.size(); ++a) {
            Acc& A = accs[a];
            if (A.done) continue;
            for (size_t i = idx; i < end; ++i) {
                if (A.covered >= A.nd.target) break;
                const Cand& c = cand[i - idx][a];
                if (c.deg != static_cast<int>(D)) continue;
                const uint32_t ell = primes.v[i];
                const u64 before = A.covered;
                A.gp.push_back(ell);
                A.covered += BitLen64(ell) - 1u;
                A.rdisc.push_back(c.disc);
                if (before < A.nd.b_f + 1)
                    for (uint32_t j = 0; j <= D; ++j) A.rf[j].push_back(c.f[j]);
                if (before < A.nd.rb + 2) A.rlam.push_back(c.lam);
                if (R.eis_rational) {
                    ENGINE_CHECK_MSG(c.eis_root, "RationalStep: the Eisenstein eigenvalue is not a root of f_theta");
                    A.reis.push_back(c.eis);
                }
                if (A.cusp && before < A.nd.b_f + 1) {
                    ENGINE_CHECK_MSG(c.cusp_ok, "RationalStep: (x - lambda) does not divide f");
                    for (uint32_t j = 0; j < D; ++j) A.rc[j].push_back(c.cq[j]);
                }
            }
            if (A.covered >= A.nd.target) A.done = true;
        }
        idx = end;
    }
    for (Acc& A : accs) {
        ThetaEntry& t = R.thetas[A.ti];
        const Need& nd = A.nd;
        const uint32_t ti = A.ti;
        const std::vector<uint32_t>& gp = A.gp;
        t.primes_used = static_cast<uint32_t>(gp.size());
        {
            const size_t kk = A.rf[0].size();
            const std::vector<uint32_t> pp(gp.begin(), gp.begin() + kk);
            for (uint32_t j = 0; j <= D; ++j) t.f.push_back(ZFromResidues(pp, A.rf[j], nd.b_f).hex());
        }
        if (R.eis_rational) {
            const size_t kk = A.rlam.size();
            t.lambda_hex = ZFromResidues(std::vector<uint32_t>(gp.begin(), gp.begin() + kk), A.rlam, nd.rb + 1).hex();
        }
        t.disc = FactorFromResidues(gp, A.rdisc, nd.b_disc, s);
        discV[ti].D = ZFromResidues(gp, A.rdisc, nd.b_disc);
        discV[ti].residues = std::move(A.rdisc);
        discV[ti].F = t.disc;
        if (R.eis_rational) {
            t.eis_res = FactorFromResidues(gp, A.reis, nd.b_eis, s);
            eisV[ti].D = ZFromResidues(gp, A.reis, nd.b_eis);
            eisV[ti].residues = std::move(A.reis);
            eisV[ti].F = t.eis_res;
            if (A.cusp) {
                const size_t kk = A.rc.empty() ? 0 : A.rc[0].size();
                const std::vector<uint32_t> pp(gp.begin(), gp.begin() + kk);
                for (uint32_t j = 0; j < D; ++j) R.cusp_part.push_back(ZFromResidues(pp, A.rc[j], nd.b_f).hex());
            }
        }
    }

    // ---- the pool: every certified prime above B of the cell, divided out of every value, iterated to a fixed point (audit F2):
    // a prime certified only during a pass (a cofactor that becomes word size once a pool prime is removed, or a Pratt-certified
    // root of a big cofactor) enters the next pass. It terminates: the pool only grows and is bounded by the primes of the values.
    {
        std::set<u64> pool_set, done;
        for (;;) {
            for (uint32_t ti : R.discriminants_used) {
                for (const auto& pp : R.thetas[ti].disc.factors) if (pp.proof == "pratt") pool_set.insert(pp.q);
                for (const auto& pp : R.thetas[ti].eis_res.factors) if (pp.proof == "pratt") pool_set.insert(pp.q);
            }
            if (pool_set == done) break;
            done = pool_set;
            const std::vector<u64> pool(pool_set.begin(), pool_set.end());
            for (uint32_t ti : R.discriminants_used) {
                ThetaEntry& t = R.thetas[ti];
                t.disc = FactorFromResidues(discV[ti].D.primes, discV[ti].residues, discV[ti].D.bound_bits, s, pool);
                discV[ti].F = t.disc;
                if (R.eis_rational) {
                    t.eis_res = FactorFromResidues(eisV[ti].D.primes, eisV[ti].residues, eisV[ti].D.bound_bits, s, pool);
                    eisV[ti].F = t.eis_res;
                }
            }
        }
    }

    // ---- (C2): the gcd of the discriminants over word-size primes ----
    {
        std::vector<const Valued*> xs;
        for (uint32_t ti : R.discriminants_used) xs.push_back(&discV[ti]);
        R.disc_gcd = GcdOf(xs, s);
        R.c2_status = R.disc_gcd.complete() ? "complete" : "partial(unfactored)";
        for (const auto& pp : R.disc_gcd.factors) {
            if (pp.proof == "probable") continue;
            R.S_cand.push_back({pp.q, pp.e, pp.e >= 2 ? "disc_square" : "disc_factor"});
        }
    }
    // ---- g_P: the Eisenstein partition ----
    if (!R.eis_rational) {
        R.gP_status = "general_case";
    } else {
        std::vector<const Valued*> xs;
        for (uint32_t ti : R.discriminants_used) {
            bool unit = false;
            for (uint32_t o = 0; o < m; ++o)
                if (R.thetas[ti].coeffs[o] == 1) {
                    uint32_t sum = 0;
                    for (uint32_t x : R.thetas[ti].coeffs) sum += x;
                    if (sum == 1) { unit = true; R.gP_ops.push_back(P[o]); }
                }
            if (!unit) continue;
            xs.push_back(&eisV[ti]);
            R.gP_running.push_back(FactoredText(GcdOf(xs, s)));
        }
        if (xs.empty()) {
            R.gP_status = "open(no primitive T_p)";
        } else {
            R.gP = GcdOf(xs, s);
            R.gP_status = R.gP.complete() ? "upper_bound_certified" : "partial(unfactored)";
        }
    }
    if (keep) { runs.resize(computed); *keep = std::move(runs); }   // (HZ-EISUP) the retained runs, value-neutral
    return R;
}
}  // namespace

RationalResult RationalStep(uint32_t N, uint32_t n, const std::vector<uint32_t>& P, const RationalSettings& s, const RationalInputs* in) {
    return RationalStepImpl(N, n, P, 0, s, in);
}

RationalResult RationalStepFull(uint32_t N, uint32_t n, const std::vector<uint32_t>& P, const std::vector<uint32_t>& U,
                                const RationalSettings& s, const RationalInputs* in, std::vector<PrimeRun>* keep) {
    if (N < 2) throw std::invalid_argument("RationalStepFull: N >= 2 (at N = 1 RationalStep is the full algebra)");
    if (U.empty()) throw std::invalid_argument("RationalStepFull: the U_p list is empty");
    std::vector<uint32_t> ops = P;
    ops.insert(ops.end(), U.begin(), U.end());
    return RationalStepImpl(N, n, ops, U.size(), s, in, keep);
}

}  // namespace heckez
