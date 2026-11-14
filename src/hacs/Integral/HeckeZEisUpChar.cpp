// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZEisUpChar -- slot-27 EISUPb: HZ-EISUP (b), the Eisenstein U_p systems with characters (see the header and HZEISUPb_design.md v0.4).

#include "HeckeZEisUpChar.h"

#include "HeckeZCharpoly.h"
#include "../NumberField/BigInt.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>

namespace heckez {
namespace eisup {

namespace {

using u64 = uint64_t;
using nf::BigInt;
using ZPoly = std::vector<BigInt>;   // low to high

// ---- exact integers ---------------------------------------------------------------------------------------------------------------
BigInt FromHex(const std::string& s) {
    size_t i = 0;
    const bool neg = !s.empty() && s[0] == '-';
    if (neg) i = 1;
    ENGINE_CHECK_MSG(s.size() >= i + 3 && s[i] == '0' && s[i + 1] == 'x', "HeckeZEisUpChar: not a canonical hex integer");
    BigInt v(0);
    const BigInt sixteen(16);
    for (size_t j = i + 2; j < s.size(); ++j) {
        const char c = s[j];
        int d = -1;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        ENGINE_CHECK_MSG(d >= 0, "HeckeZEisUpChar: not a canonical hex integer");
        v = v * sixteen + BigInt(d);
    }
    return neg ? -v : v;
}
std::string ToHex(const BigInt& v) {
    if (v.is_zero()) return "0x0";
    const BigInt::Mag& m = v.magnitude();
    std::ostringstream o;
    if (v.is_neg()) o << "-";
    o << "0x" << std::hex << m.back();
    for (size_t i = m.size() - 1; i-- > 0;) {
        o.width(8);
        o.fill('0');
        o << m[i];
    }
    return o.str();
}
ZPoly PolyFromHex(const std::vector<std::string>& h) {
    ZPoly p;
    for (const auto& s : h) p.push_back(FromHex(s));
    return p;
}
std::vector<std::string> PolyHex(const ZPoly& p) {
    std::vector<std::string> o;
    for (const auto& c : p) o.push_back(ToHex(c));
    return o;
}
ZPoly PolyMul(const ZPoly& a, const ZPoly& b) {
    ZPoly c(a.size() + b.size() - 1, BigInt(0));
    for (size_t i = 0; i < a.size(); ++i)
        if (!a[i].is_zero())
            for (size_t j = 0; j < b.size(); ++j) c[i + j] += a[i] * b[j];
    return c;
}
// a = q b + r for monic b; *rem_zero = (r == 0)
ZPoly PolyDivMonic(const ZPoly& a, const ZPoly& b, bool* rem_zero) {
    ENGINE_CHECK_MSG(!b.empty() && b.back() == BigInt(1), "HeckeZEisUpChar: division by a non-monic polynomial");
    if (a.size() < b.size()) {
        *rem_zero = true;
        for (const auto& c : a) *rem_zero = *rem_zero && c.is_zero();
        return ZPoly{BigInt(0)};
    }
    ZPoly r = a;
    ZPoly q(a.size() - b.size() + 1, BigInt(0));
    for (size_t i = q.size(); i-- > 0;) {
        const BigInt c = r[i + b.size() - 1];
        q[i] = c;
        if (c.is_zero()) continue;
        for (size_t j = 0; j < b.size(); ++j) r[i + j] -= c * b[j];
    }
    *rem_zero = true;
    for (size_t i = 0; i + 1 < b.size(); ++i) *rem_zero = *rem_zero && r[i].is_zero();
    return q;
}
uint64_t BitLen(const BigInt& v) { return v.is_zero() ? 0 : static_cast<uint64_t>(v.bit_length()); }
u64 ModOf(const BigInt& v, u64 q) {
    const BigInt r = v.floor_mod(BigInt::from_u64(q));
    if (r.is_zero()) return 0;
    const BigInt::Mag& m = r.magnitude();
    return m.size() == 1 ? m[0] : (static_cast<u64>(m[1]) << 32) | m[0];
}
uint32_t ValuationU64(const BigInt& x, uint64_t ell) {
    if (ell < (1ull << 32)) return x.valuation(static_cast<uint32_t>(ell));
    const BigInt L = BigInt::from_u64(ell);
    BigInt y = x, q, r;
    uint32_t v = 0;
    for (;;) {
        BigInt::divmod(y, L, q, r);
        if (!r.is_zero()) return v;
        y = q;
        ++v;
    }
}

// ---- arithmetic mod q < 2^31 --------------------------------------------------------------------------------------------------
u64 PowMod(u64 a, u64 e, u64 q) {
    u64 r = 1 % q;
    for (a %= q; e; e >>= 1, a = a * a % q)
        if (e & 1) r = r * a % q;
    return r;
}
u64 InvMod(u64 a, u64 q) { return PowMod(a, q - 2, q); }
bool IsPrime32(u64 x) {
    if (x < 2) return false;
    for (u64 d = 2; d * d <= x; ++d)
        if (x % d == 0) return false;
    return true;
}
std::vector<u64> PrimeFactors(u64 x) {
    std::vector<u64> o;
    for (u64 p = 2; p * p <= x; ++p)
        if (x % p == 0) {
            o.push_back(p);
            while (x % p == 0) x /= p;
        }
    if (x > 1) o.push_back(x);
    return o;
}
// the least primitive root mod a prime q; `skip` = how many primitive roots to pass first (0: the least)
u64 PrimitiveRoot(u64 q, uint32_t skip = 0) {
    if (q == 2) return 1;
    const std::vector<u64> fs = PrimeFactors(q - 1);
    for (u64 g = 2; g < q; ++g) {
        bool ok = true;
        for (u64 r : fs) ok = ok && PowMod(g, (q - 1) / r, q) != 1;
        if (ok) {
            if (skip == 0) return g;
            --skip;
        }
    }
    ENGINE_CHECK_MSG(false, "HeckeZEisUpChar: no primitive root");
    return 0;
}
std::vector<std::vector<u64>> Kernel(std::vector<u64> A, size_t rows, size_t cols, u64 q, std::vector<size_t>* free) {
    std::vector<size_t> pivc;
    size_t r = 0;
    for (size_t c = 0; c < cols && r < rows; ++c) {
        size_t piv = r;
        while (piv < rows && A[piv * cols + c] == 0) ++piv;
        if (piv == rows) continue;
        if (piv != r)
            for (size_t j = 0; j < cols; ++j) std::swap(A[piv * cols + j], A[r * cols + j]);
        const u64 inv = InvMod(A[r * cols + c], q);
        for (size_t j = 0; j < cols; ++j) A[r * cols + j] = A[r * cols + j] * inv % q;
        for (size_t i = 0; i < rows; ++i) {
            if (i == r) continue;
            const u64 f = A[i * cols + c];
            if (!f) continue;
            for (size_t j = 0; j < cols; ++j) A[i * cols + j] = (A[i * cols + j] + (q - f) * A[r * cols + j]) % q;
        }
        pivc.push_back(c);
        ++r;
    }
    std::vector<bool> isp(cols, false);
    for (size_t c : pivc) isp[c] = true;
    free->clear();
    for (size_t c = 0; c < cols; ++c)
        if (!isp[c]) free->push_back(c);
    std::vector<std::vector<u64>> basis;
    for (size_t f : *free) {
        std::vector<u64> v(cols, 0);
        v[f] = 1;
        for (size_t i = 0; i < pivc.size(); ++i) v[pivc[i]] = (q - A[i * cols + f]) % q;
        basis.push_back(std::move(v));
    }
    return basis;
}
std::vector<u64> MatMul(const std::vector<u64>& A, const std::vector<u64>& B, size_t b, u64 q) {
    std::vector<u64> C(b * b, 0);
    for (size_t i = 0; i < b; ++i)
        for (size_t l = 0; l < b; ++l) {
            const u64 a = A[i * b + l];
            if (!a) continue;
            for (size_t j = 0; j < b; ++j) C[i * b + j] = (C[i * b + j] + a * B[l * b + j]) % q;
        }
    return C;
}
// p(X) mod q by Horner (p low to high, entries already reduced)
std::vector<u64> PolyOfMatrix(const std::vector<u64>& p, const std::vector<u64>& X, size_t b, u64 q) {
    std::vector<u64> M(b * b, 0);
    for (size_t i = p.size(); i-- > 0;) {
        if (i + 1 < p.size()) M = MatMul(M, X, b, q);
        for (size_t x = 0; x < b; ++x) M[x * b + x] = (M[x * b + x] + p[i]) % q;
    }
    return M;
}
// the restriction of an operator X (b x b, column convention: X v in coordinates) to the kernel (basis, free columns); verified stable
std::vector<u64> Restrict(const std::vector<u64>& X, size_t b, const std::vector<std::vector<u64>>& basis, const std::vector<size_t>& fr,
                          u64 q, const char* what) {
    const size_t e = basis.size();
    std::vector<u64> R(e * e, 0);
    for (size_t t = 0; t < e; ++t) {
        std::vector<u64> w(b, 0);
        for (size_t i = 0; i < b; ++i) {
            u64 s = 0;
            for (size_t j = 0; j < b; ++j)
                if (basis[t][j]) s = (s + X[i * b + j] * basis[t][j]) % q;
            w[i] = s;
        }
        std::vector<u64> back(b, 0);
        for (size_t i = 0; i < e; ++i) {
            const u64 c = w[fr[i]];
            R[i * e + t] = c;
            if (c)
                for (size_t x = 0; x < b; ++x) back[x] = (back[x] + c * basis[i][x]) % q;
        }
        ENGINE_CHECK_MSG(back == w, what);
    }
    return R;
}
bool SeparableModQ(const std::vector<u64>& ev0, u64 q) {
    auto trim = [](std::vector<u64>& p) { while (!p.empty() && p.back() == 0) p.pop_back(); };
    std::vector<u64> a = ev0, b;
    for (size_t i = 1; i < ev0.size(); ++i) b.push_back(ev0[i] * (i % q) % q);
    trim(a);
    trim(b);
    while (!b.empty()) {
        while (a.size() >= b.size() && !a.empty()) {
            const u64 f = a.back() * InvMod(b.back(), q) % q;
            const size_t sh = a.size() - b.size();
            for (size_t i = 0; i < b.size(); ++i) a[sh + i] = (a[sh + i] + (q - f) * b[i]) % q;
            trim(a);
        }
        std::swap(a, b);
    }
    return a.size() == 1;
}
u64 EvalModQ(const std::vector<u64>& p, u64 x, u64 q) {
    u64 acc = 0;
    for (size_t i = p.size(); i-- > 0;) acc = (acc * x + p[i]) % q;
    return acc;
}

// ---- the characters mod s_N (Sec. 2.1) ---------------------------------------------------------------------------------------------
struct Chars {
    uint32_t N = 0, k = 0, s_N = 1, o_N = 1;
    std::vector<uint32_t> sp, gen;                 // primes of s_N, least primitive roots
    std::vector<std::vector<uint32_t>> ind;        // ind[i][x mod sp_i] (discrete log), x != 0
    std::vector<uint32_t> Up;                      // p | N ascending
    std::vector<uint32_t> vN;                      // v_p(N), p | N ascending
    // exponent of psi(x) in Z / o_N for psi of exponent vector a, x prime to cond(psi) (the primitive character)
    uint32_t exp_of(const std::vector<uint32_t>& a, u64 x) const {
        u64 t = 0;
        for (size_t i = 0; i < sp.size(); ++i) {
            if (!a[i]) continue;
            const uint32_t r = static_cast<uint32_t>(x % sp[i]);
            ENGINE_CHECK_MSG(r != 0, "HeckeZEisUpChar: psi evaluated at a divisor of its conductor");
            t += static_cast<u64>(a[i]) * ind[i][r] * (o_N / (sp[i] - 1));
        }
        return static_cast<uint32_t>(t % o_N);
    }
    uint32_t order(const std::vector<uint32_t>& a) const {
        uint32_t o = 1;
        for (size_t i = 0; i < sp.size(); ++i) {
            const uint32_t m = sp[i] - 1, g = std::gcd(a[i], m);
            const uint32_t oi = a[i] ? m / g : 1;
            o = o / std::gcd(o, oi) * oi;
        }
        return o;
    }
    uint32_t cond(const std::vector<uint32_t>& a) const {
        uint32_t c = 1;
        for (size_t i = 0; i < sp.size(); ++i)
            if (a[i]) c *= sp[i];
        return c;
    }
    std::vector<uint32_t> power(const std::vector<uint32_t>& a, u64 u) const {
        std::vector<uint32_t> b(a.size());
        for (size_t i = 0; i < sp.size(); ++i) b[i] = static_cast<uint32_t>(a[i] * u % (sp[i] - 1));
        return b;
    }
    std::vector<std::vector<uint32_t>> all() const {
        std::vector<std::vector<uint32_t>> out = {std::vector<uint32_t>(sp.size(), 0)};
        for (size_t i = 0; i < sp.size(); ++i) {
            std::vector<std::vector<uint32_t>> nx;
            for (const auto& v : out)
                for (uint32_t x = 0; x + 1 < sp[i]; ++x) {
                    auto w = v;
                    w[i] = x;
                    nx.push_back(w);
                }
            out = std::move(nx);
        }
        return out;
    }
    // Y_p(psi): 0, 1 (alpha), 2 (beta)
    std::vector<int32_t> types(const std::vector<uint32_t>& a, size_t ip) const {
        const uint32_t p = Up[ip];
        bool in_cond = false;
        for (size_t i = 0; i < sp.size(); ++i) in_cond = in_cond || (sp[i] == p && a[i] != 0);
        if (in_cond) return {0};
        if (vN[ip] == 1) return {1, 2};
        return {0, 1, 2};
    }
};
Chars MakeChars(uint32_t N, uint32_t k) {
    Chars C;
    C.N = N;
    C.k = k;
    uint32_t x = N;
    for (uint32_t p = 2; x > 1; ++p) {
        if (static_cast<u64>(p) * p > x) p = x;
        if (x % p == 0) {
            uint32_t v = 0;
            while (x % p == 0) {
                x /= p;
                ++v;
            }
            ENGINE_CHECK_MSG(v <= 2, "HeckeZEisUpChar C-B0: N is not cube-free with (C1) certified (contradicts prop:heckeZrat (4)(b): a defect)");
            C.Up.push_back(p);
            C.vN.push_back(v);
            if (v == 2 && p % 2 == 1) C.sp.push_back(p);
        }
    }
    for (uint32_t p : C.sp) {
        C.s_N *= p;
        C.o_N = C.o_N / std::gcd(C.o_N, p - 1) * (p - 1);
        const uint32_t g = static_cast<uint32_t>(PrimitiveRoot(p));
        C.gen.push_back(g);
        std::vector<uint32_t> ind(p, 0);
        u64 y = 1;
        for (uint32_t e = 0; e + 1 < p; ++e) {
            ind[y] = e;
            y = y * g % p;
        }
        C.ind.push_back(std::move(ind));
    }
    return C;
}

// a closed-form system's value of one operator modulo M, zeta -> z (zp[t] = z^t, t in [0, o_N))
struct Op {
    u64 p = 0;
    bool U = false;
    size_t ip = 0;   // index among p | N (U only)
};
u64 ValueMod(const Chars& C, const std::vector<uint32_t>& a, const std::vector<int32_t>& y, const Op& X, const std::vector<u64>& zp, u64 M) {
    const uint32_t o = C.o_N;
    if (!X.U) {
        const uint32_t t = C.exp_of(a, X.p);
        const u64 Q = PowMod(X.p % M, C.k - 1, M);
        return (zp[t] + zp[(o - t) % o] * Q) % M;
    }
    const int32_t ty = y[X.ip];
    if (ty == 0) return 0;
    const uint32_t t = C.exp_of(a, X.p);
    if (ty == 1) return zp[t];
    return zp[(o - t) % o] * PowMod(X.p % M, C.k - 1, M) % M;
}
// the symbolic root of X on the system (a, y): (kind, exponent); kind 0 = the value 0; distinct symbolic roots are distinct numbers
std::pair<int32_t, uint32_t> SymRoot(const Chars& C, const std::vector<uint32_t>& a, const std::vector<int32_t>& y, const Op& X) {
    if (!X.U) return {3, C.exp_of(a, X.p)};
    const int32_t ty = y[X.ip];
    if (ty == 0) return {0, 0};
    return {ty, C.exp_of(a, X.p)};
}

// a component of the closed form: an orbit representative, its type vector, the units of Z / ord
struct Comp {
    std::vector<uint32_t> a;
    std::vector<int32_t> y;
    uint32_t ord = 1, cond = 1;
    std::vector<uint32_t> units;   // u in (Z / ord)^x
};
std::vector<Comp> Components(const Chars& C) {
    std::vector<Comp> out;
    std::set<std::vector<uint32_t>> seen;
    for (const auto& a : C.all()) {
        if (seen.count(a)) continue;
        Comp c;
        c.a = a;
        c.ord = C.order(a);
        c.cond = C.cond(a);
        for (uint32_t u = 1; u <= c.ord; ++u)
            if (std::gcd(u, c.ord) == 1) {
                c.units.push_back(u % std::max<uint32_t>(c.ord, 1));
                seen.insert(C.power(a, u));
            }
        // every type vector
        std::vector<std::vector<int32_t>> ys = {{}};
        for (size_t ip = 0; ip < C.Up.size(); ++ip) {
            std::vector<std::vector<int32_t>> nx;
            for (const auto& v : ys)
                for (int32_t t : C.types(a, ip)) {
                    auto w = v;
                    w.push_back(t);
                    nx.push_back(w);
                }
            ys = std::move(nx);
        }
        for (const auto& y : ys) {
            if (C.k == 2 && c.ord == 1) {   // (1, (beta)_{p | N}) is excluded at k = 2
                bool all_beta = true;
                for (int32_t t : y) all_beta = all_beta && t == 2;
                if (all_beta) continue;
            }
            Comp d = c;
            d.y = y;
            out.push_back(d);
        }
    }
    return out;
}

// auxiliary primes M = 1 mod o_N below 2^31, descending; an element of exact order o_N mod M
struct AuxPrime {
    u64 M = 0;
    std::vector<u64> zp;
};
AuxPrime AuxAt(uint32_t o_N, size_t i) {
    static std::map<uint32_t, std::vector<AuxPrime>> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    auto& v = cache[o_N];
    u64 M = v.empty() ? ((1ull << 31) - 1) : v.back().M - 1;
    while (v.size() <= i) {
        // the largest prime below M with M = 1 mod o_N (and odd)
        M -= (M - 1) % o_N;
        while (!(M % 2 == 1 && IsPrime32(M))) M -= o_N;
        AuxPrime A;
        A.M = M;
        const std::vector<u64> rs = PrimeFactors(o_N);
        u64 z = 0;
        for (u64 x = 2; !z; ++x) {
            const u64 h = PowMod(x, (M - 1) / o_N, M);
            bool ok = true;
            for (u64 r : rs) ok = ok && PowMod(h, o_N / r, M) != 1;
            if (ok) z = h;
        }
        A.zp.resize(o_N);
        u64 w = 1;
        for (uint32_t t = 0; t < o_N; ++t) {
            A.zp[t] = w;
            w = w * z % M;
        }
        v.push_back(A);
        M -= 1;
    }
    return v[i];
}
// an integer polynomial (length len, |coeff| < 2^bits) from its residues fn(M, zp) at the auxiliary primes
ZPoly AuxCrt(uint32_t o_N, size_t len, uint64_t bits, const std::function<std::vector<u64>(u64, const std::vector<u64>&)>& fn) {
    std::vector<uint32_t> primes;
    std::vector<std::vector<u64>> res;
    uint64_t covered = 0;
    for (size_t i = 0; covered < bits + 1; ++i) {
        const AuxPrime A = AuxAt(o_N, i);
        primes.push_back(static_cast<uint32_t>(A.M));
        res.push_back(fn(A.M, A.zp));
        ENGINE_CHECK_MSG(res.back().size() == len, "HeckeZEisUpChar: auxiliary residues of the wrong length");
        covered += BitLen64(A.M) - 1u;
    }
    ZPoly out;
    for (size_t c = 0; c < len; ++c) {
        std::vector<uint32_t> r;
        for (const auto& v : res) r.push_back(static_cast<uint32_t>(v[c]));
        out.push_back(FromHex(ZFromResidues(primes, r, bits).hex()));
    }
    return out;
}
// prod_i (x - roots_i) mod M
std::vector<u64> ProdLinearMod(const std::vector<u64>& roots, u64 M) {
    std::vector<u64> p = {1};
    for (u64 r : roots) {
        std::vector<u64> q(p.size() + 1, 0);
        for (size_t i = 0; i < p.size(); ++i) {
            q[i + 1] = (q[i + 1] + p[i]) % M;
            q[i] = (q[i] + (M - r % M) * p[i]) % M;
        }
        p = std::move(q);
    }
    return p;
}
// sum_X c_X lambda(X) mod M on the system (a, y)
u64 ComboMod(const Chars& C, const std::vector<uint32_t>& a, const std::vector<int32_t>& y, const std::vector<Op>& ops,
             const std::vector<uint32_t>& coeffs, const std::vector<u64>& zp, u64 M) {
    u64 s = 0;
    for (size_t i = 0; i < ops.size(); ++i)
        if (coeffs[i]) s = (s + (coeffs[i] % M) * ValueMod(C, a, y, ops[i], zp, M)) % M;
    return s;
}
// the orbit product prod_{u} (x - lambda_{psi^u, y}(combo)); `units` the exponents u (the orbit, or (Z/o_N)^x for the tamper)
ZPoly OrbitPoly(const Chars& C, const std::vector<uint32_t>& a, const std::vector<int32_t>& y, const std::vector<uint32_t>& units,
                const std::vector<Op>& ops, const std::vector<uint32_t>& coeffs, uint64_t root_bits) {
    const uint64_t D = units.size();
    const uint64_t bits = D + D * std::max<uint64_t>(1, root_bits) + 64;
    return AuxCrt(C.o_N, units.size() + 1, bits, [&](u64 M, const std::vector<u64>& zp) {
        std::vector<u64> roots;
        for (uint32_t u : units) roots.push_back(ComboMod(C, C.power(a, u), y, ops, coeffs, zp, M));
        return ProdLinearMod(roots, M);
    });
}
// bit length of a bound on |lambda(combo)| (|psi| = 1)
uint64_t ComboBits(const Chars& C, const std::vector<Op>& ops, const std::vector<uint32_t>& coeffs) {
    BigInt B(0);
    for (size_t i = 0; i < ops.size(); ++i) {
        if (!coeffs[i]) continue;
        BigInt v = BigInt::pow(BigInt::from_u64(ops[i].p), C.k - 1);
        if (!ops[i].U) v += BigInt(1);
        B += BigInt::from_u64(coeffs[i]) * v;
    }
    return BitLen(B);
}
std::vector<Op> OpsOf(const Chars& C, const std::vector<uint32_t>& P) {
    std::vector<Op> o;
    for (uint32_t p : P) {
        Op X;
        X.p = p;
        X.U = C.N % p == 0;
        if (X.U) {
            const auto it = std::find(C.Up.begin(), C.Up.end(), p);
            ENGINE_CHECK_MSG(it != C.Up.end(), "HeckeZEisUpChar: U_p with p not a prime of N");
            X.ip = static_cast<size_t>(it - C.Up.begin());
        }
        o.push_back(X);
    }
    return o;
}

// the per-prime read: K_q = ker P^Eis(T_q0) and the restrictions of every operator (e x e) and of theta
struct PrimeRead {
    u64 q = 0;
    std::vector<std::vector<u64>> R;
    std::vector<u64> theta;
};
bool ReadAtPrime(const PrimeRun& run, size_t m, const std::vector<std::string>& Peis, size_t e, const std::vector<uint32_t>& theta,
                 PrimeRead* out) {
    const u64 q = run.ell;
    const size_t b = run.h1;
    std::vector<u64> pe;
    for (const auto& h : Peis) pe.push_back(ModOf(FromHex(h), q));
    const std::vector<u64> A = PolyOfMatrix(pe, run.mats[0], b, q);
    std::vector<size_t> fr;
    const auto basis = Kernel(A, b, b, q, &fr);
    if (basis.size() != e) return false;
    out->q = q;
    out->R.assign(m, {});
    for (size_t o = 0; o < m; ++o)
        out->R[o] = Restrict(run.mats[o], b, basis, fr, q, "HeckeZEisUpChar: K_q is not stable under an operator (T is commutative)");
    out->theta.assign(e * e, 0);
    for (size_t o = 0; o < m; ++o) {
        if (!theta[o]) continue;
        const u64 c = theta[o] % q;
        for (size_t x = 0; x < e * e; ++x) out->theta[x] = (out->theta[x] + c * out->R[o][x]) % q;
    }
    return true;
}
void RunPrimes(uint32_t N, uint32_t n, const std::vector<uint32_t>& ops, const std::vector<uint32_t>& primes, size_t from, size_t to,
               uint32_t threads, std::vector<PrimeRun>& out) {
    out.resize(std::max(out.size(), to));
    std::atomic<size_t> next(from);
    const uint32_t T = std::max<uint32_t>(1, std::min<uint32_t>(threads, static_cast<uint32_t>(to - from)));
    std::vector<std::exception_ptr> errs(T);
    auto work = [&](uint32_t w) {
        try {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= to) break;
                out[i] = RunAtPrime(N, n, ops, primes[i]);
            }
        } catch (...) {
            errs[w] = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    for (uint32_t w = 1; w < T; ++w) pool.emplace_back(work, w);
    work(0);
    for (auto& t : pool) t.join();
    for (auto& ex : errs)
        if (ex) std::rethrow_exception(ex);
}
bool PolyLess(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const BigInt x = FromHex(a[i]), y = FromHex(b[i]);
        if (x != y) return x < y;
    }
    return a.size() < b.size();
}
std::string JoinPoly(const std::vector<std::string>& p) {
    std::string s = "[";
    for (size_t i = 0; i < p.size(); ++i) s += (i ? "," : "") + p[i];
    return s + "]";
}

}  // namespace

bool InScopeChar(uint32_t N, uint32_t n) { return N > 1 && n % 2 == 0 && EisAnemicCount(n + 2, N) != 1; }

std::vector<uint32_t> EisUpCharResult::Uprimes() const { return std::vector<uint32_t>(ops.end() - nU, ops.end()); }

std::string EisUpCharResult::text() const {
    std::ostringstream o;
    o << "EISUPB N=" << N << " n=" << n << " ops=";
    for (size_t i = 0; i < ops.size(); ++i) o << (i ? "," : "") << (i + nU >= ops.size() ? "U_" : "T_") << ops[i];
    o << " theta=(";
    for (size_t i = 0; i < theta.size(); ++i) o << (i ? " " : "") << theta[i];
    o << ") e_k=" << e_k << " d=" << d << " q0=" << q0 << " s_N=" << s_N << " o_N=" << o_N << " primes=" << primes_used << " sep=" << sep_used
      << " skipped=[";
    for (size_t i = 0; i < skipped.size(); ++i) o << (i ? "," : "") << skipped[i];
    o << "] q*=" << id_prime << " S=[";
    for (size_t i = 0; i < S.size(); ++i) o << (i ? "," : "") << S[i];
    o << "]\nP_eis=" << JoinPoly(P_eis) << "\ne_theta=" << JoinPoly(e_theta) << "\n";
    const std::vector<uint32_t> U = Uprimes();
    for (size_t j = 0; j < comps.size(); ++j) {
        const EisCharComp& c = comps[j];
        o << "comp " << j << ": degree=" << c.degree << " psi_cond=" << c.psi_cond << " psi_order=" << c.psi_order << " minpoly=" << JoinPoly(c.minpoly)
          << " theta_poly=" << JoinPoly(c.theta_poly);
        if (c.degree == 1) {
            o << " lambda_theta=" << c.lambda_theta;
            for (size_t i = 0; i < U.size(); ++i) o << " U_" << U[i] << "=" << c.U[i];
        } else {
            for (size_t i = 0; i < U.size(); ++i) o << " U_" << U[i] << ".charpoly=" << JoinPoly(c.Ucharpoly[i]);
        }
        o << "\n";
    }
    o << "c_theta=" << JoinPoly(c_theta) << "\n";
    return o.str();
}

uint64_t EisUpCharResult::digest() const { return engine::FnvBytes(engine::kFnvOffset64, text()); }

EisUpCharResult EisUpCharStep(const EisUpCharInput& in) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(InScopeChar(in.N, in.n) || (in.force_scope && in.N > 1 && in.n % 2 == 0), "HeckeZEisUpChar: outside the scope (N > 1, e^an_k(N) != 1)");
    ENGINE_CHECK_MSG(in.nU >= 1 && in.nU < in.ops.size() && in.theta.size() == in.ops.size(), "HeckeZEisUpChar: the operator list");
    ENGINE_CHECK_MSG(in.e_k >= 1 && in.d >= in.e_k && in.f.size() == static_cast<size_t>(in.d) + 1, "HeckeZEisUpChar: e_k, d, f_theta");
    ENGINE_CHECK_MSG(in.N % in.ops[0] != 0, "HeckeZEisUpChar: q0 = the first operator must be a T_p, p !| N");
    const EisUpCharTamper tz = in.tamper ? *in.tamper : EisUpCharTamper{};
    const uint32_t N = in.N, n = in.n, k = n + 2;
    const size_t m = in.ops.size(), nT = m - in.nU;
    const size_t e = static_cast<size_t>(in.e_k + tz.ek_delta);
    ENGINE_CHECK_MSG(e >= 1, "HeckeZEisUpChar: e_k >= 1");

    // ---- C-B0: the characters, the closed form, P^Eis ----
    const Chars C = MakeChars(N, k);
    ENGINE_CHECK_MSG((C.s_N > 1 || in.force_scope) && EisAnemicCount(k, N) == [&] {
        u64 phi = 1;
        for (uint32_t p : C.sp) phi *= p - 1;
        return phi;
    }(), "HeckeZEisUpChar C-B0: e^an_k(N) != phi(s_N)");
    EisUpCharResult R;
    R.N = N;
    R.n = n;
    R.nU = in.nU;
    R.ops = in.ops;
    R.theta = in.theta;
    R.e_k = in.e_k;
    R.d = in.d;
    R.q0 = in.ops[0];
    R.s_N = C.s_N;
    R.o_N = C.o_N;
    R.sp = C.sp;
    R.gen = C.gen;
    ENGINE_CHECK_MSG(R.Uprimes() == C.Up, "HeckeZEisUpChar: the U_p must be the primes dividing N, ascending");
    for (size_t o = 0; o < nT; ++o) ENGINE_CHECK_MSG(N % in.ops[o] != 0, "HeckeZEisUpChar: a T_p of the list with p | N");
    const std::vector<Op> ops = OpsOf(C, in.ops);
    const uint32_t q0 = in.ops[0];
    {   // P^Eis_{q0}: the distinct values lambda_psi(T_q0) over all psi mod s_N (distinct exponents of psi(q0))
        std::set<uint32_t> ts;
        for (const auto& a : C.all()) ts.insert(C.exp_of(a, q0));
        const std::vector<uint32_t> tv(ts.begin(), ts.end());
        const BigInt Qb = BigInt::pow(BigInt::from_u64(q0), k - 1) + BigInt(1);
        const uint64_t D = tv.size();
        const ZPoly pe = AuxCrt(C.o_N, tv.size() + 1, D + D * BitLen(Qb) + 64, [&](u64 M, const std::vector<u64>& zp) {
            std::vector<u64> roots;
            const u64 Q = PowMod(q0 % M, k - 1, M);
            for (uint32_t t : tv) roots.push_back((zp[t] + zp[(C.o_N - t) % C.o_N] * Q) % M);
            return ProdLinearMod(roots, M);
        });
        R.P_eis = PolyHex(pe);
    }
    std::vector<Comp> comps = Components(C);
    if (tz.psi_squared)
        for (Comp& c : comps) {
            c.a = C.power(c.a, 2);
            c.ord = C.order(c.a);
            c.cond = C.cond(c.a);
        }
    if (tz.y_alpha_at) {   // (validation) a VALID wrong type: v_p = 2, p !| cond psi, y_p 0 -> alpha
        uint32_t hit = 0;
        for (Comp& c : comps)
            if (c.units.size() >= 2 && c.cond % static_cast<uint32_t>(tz.y_alpha_at) != 0)
                for (size_t ip = 0; ip < C.Up.size(); ++ip)
                    if (C.Up[ip] == static_cast<uint32_t>(tz.y_alpha_at) && c.y[ip] == 0) {
                        c.y[ip] = 1;
                        ++hit;
                    }
        ENGINE_CHECK_MSG(hit > 0, "HeckeZEisUpChar tamper y_alpha_at: vacuous (no degree >= 2 component with y_p = 0, p !| cond psi)");
    }
    if (tz.norm_full)
        for (Comp& c : comps) {
            c.units.clear();
            for (uint32_t u = 1; u <= C.o_N; ++u)
                if (std::gcd(u, C.o_N) == 1) c.units.push_back(u % C.o_N);
        }

    // ---- the bounds ----
    const uint64_t th_bits = ComboBits(C, ops, in.theta);
    const uint64_t E = static_cast<uint64_t>(e);
    uint64_t bound = E + E * std::max<uint64_t>(1, th_bits);
    for (size_t o = 0; o < m; ++o) {
        std::vector<uint32_t> unit(m, 0);
        unit[o] = 1;
        bound = std::max(bound, E + E * std::max<uint64_t>(1, ComboBits(C, ops, unit)));
    }
    const uint64_t need = bound + 1 + 64;

    // ---- C-B1: the word primes (the retained runs first, then the same list) ----
    const std::vector<PrimeRun>* kept = (in.runs && !in.force_recompute) ? in.runs : nullptr;
    std::vector<uint32_t> primes;
    if (kept)
        for (const PrimeRun& r : *kept) primes.push_back(r.ell);
    {
        std::vector<uint32_t> chk;
        ExtendWordPrimes(chk, primes.size(), N);
        ENGINE_CHECK_MSG(chk == primes, "HeckeZEisUpChar: the retained runs are not a prefix of the word-prime list");
    }
    std::vector<PrimeRun> extra;
    const size_t nkept = primes.size();
    const uint32_t T = std::max<uint32_t>(1, in.rs.threads);
    std::vector<PrimeRead> reads;
    std::vector<int64_t> read_of_index;   // per list index: the read's index, or -1 (skipped)
    auto run_at = [&](size_t i) -> const PrimeRun& {
        if (i < nkept) {
            ++R.reused;
            return (*kept)[i];
        }
        if (i - nkept >= extra.size() || extra[i - nkept].mats.empty()) {
            const size_t to = std::min<size_t>(std::max<size_t>(i + 1, i + T), in.rs.word_prime_cap);
            ExtendWordPrimes(primes, to, N);
            std::vector<uint32_t> tail(primes.begin() + nkept, primes.end());
            RunPrimes(N, n, in.ops, tail, i - nkept, to - nkept, T, extra);
            R.computed += static_cast<uint32_t>(to - i);
        }
        return extra[i - nkept];
    };
    auto next_read = [&]() -> bool {
        const size_t idx = read_of_index.size();
        if (idx >= in.rs.word_prime_cap) throw WordPrimeCapRefusal(in.rs.word_prime_cap);
        const PrimeRun& run = run_at(idx);
        ENGINE_CHECK_MSG(run.h1 == in.b1 && run.mats.size() == m, "HeckeZEisUpChar: dim H^1 at a word prime differs from b1");
        PrimeRead pr;
        const bool ok = ReadAtPrime(run, m, R.P_eis, e, in.theta, &pr);
        const uint32_t q = run.ell;
        if (idx >= nkept) extra[idx - nkept] = PrimeRun();
        if (!ok) {
            R.skipped.push_back(q);
            read_of_index.push_back(-1);
            return false;
        }
        read_of_index.push_back(static_cast<int64_t>(reads.size()));
        reads.push_back(std::move(pr));
        return true;
    };
    uint64_t covered = 0;
    while (covered < need)
        if (next_read()) covered += BitLen64(reads.back().q) - 1u;
    const size_t nread_eth = reads.size();
    R.primes_used = static_cast<uint32_t>(nread_eth);

    // ---- C-B2: e_theta by CRT, e_theta | f_theta, c_theta ----
    auto crt = [&](const std::vector<uint32_t>& ps, const std::vector<std::vector<u64>>& per_prime, uint64_t bits) {
        ZPoly out;
        for (size_t c = 0; c < per_prime[0].size(); ++c) {
            std::vector<uint32_t> res;
            for (const auto& v : per_prime) res.push_back(static_cast<uint32_t>(v[c]));
            out.push_back(FromHex(ZFromResidues(ps, res, bits).hex()));
        }
        return out;
    };
    ZPoly eth;
    {
        std::vector<uint32_t> ps;
        std::vector<std::vector<u64>> pp;
        for (size_t i = 0; i < nread_eth; ++i) {
            ps.push_back(static_cast<uint32_t>(reads[i].q));
            pp.push_back(CharpolyHessenberg(reads[i].theta, static_cast<uint32_t>(e), reads[i].q));
        }
        eth = crt(ps, pp, bound);
    }
    const ZPoly f = PolyFromHex(in.f);
    ENGINE_CHECK_MSG(f.back() == BigInt(1), "HeckeZEisUpChar: f_theta is not monic");
    bool rz = false;
    const ZPoly cth = PolyDivMonic(f, eth, &rz);
    ENGINE_CHECK_MSG(rz, "HeckeZEisUpChar C-B2: e_theta does not divide f_theta exactly over Z");
    ENGINE_CHECK_MSG(cth.size() == static_cast<size_t>(in.d - in.e_k) + 1, "HeckeZEisUpChar C-B2: deg c_theta != d - e_k");

    // ---- C-B3, C-B4: the proposal, its product ----
    const size_t nc = comps.size();
    std::vector<ZPoly> fac(nc);
    std::vector<std::vector<ZPoly>> gprop(nc, std::vector<ZPoly>(m));
    for (size_t j = 0; j < nc; ++j) {
        fac[j] = OrbitPoly(C, comps[j].a, comps[j].y, comps[j].units, ops, in.theta, th_bits);
        for (size_t o = 0; o < m; ++o) {
            std::vector<uint32_t> unit(m, 0);
            unit[o] = 1;
            gprop[j][o] = OrbitPoly(C, comps[j].a, comps[j].y, comps[j].units, ops, unit, ComboBits(C, ops, unit));
        }
    }
    if (tz.factor_plus_one >= 0 && static_cast<size_t>(tz.factor_plus_one) < nc) fac[static_cast<size_t>(tz.factor_plus_one)][0] += BigInt(1);
    if (tz.gprop_plus_one_op >= 0) {   // (validation, audit M2) the proposal g_{X_o,0} + 1 before C-B5; e_{theta,j} untouched
        ENGINE_CHECK_MSG(nc > 0 && static_cast<size_t>(tz.gprop_plus_one_op) < m, "HeckeZEisUpChar tamper gprop_plus_one_op: out of range");
        gprop[0][static_cast<size_t>(tz.gprop_plus_one_op)][0] += BigInt(1);
    }
    // (validation, audit M2) misalign_unweighted: the component jt, the T_p ops[ot] with theta_ot = 0, the unit vt (psi^vt(p) != psi(p))
    size_t jt = nc, ot = m;
    uint32_t vt = 1;
    if (tz.misalign_unweighted) {
        for (size_t j = 0; j < nc && jt == nc; ++j) {
            if (comps[j].units.size() < 2) continue;
            for (size_t o = 0; o < nT && jt == nc; ++o) {
                if (in.theta[o] != 0) continue;
                for (uint32_t u : comps[j].units)
                    if (u != 1 && C.exp_of(C.power(comps[j].a, u), in.ops[o]) != C.exp_of(comps[j].a, in.ops[o])) {
                        jt = j;
                        ot = o;
                        vt = u;
                        break;
                    }
            }
        }
        ENGINE_CHECK_MSG(jt < nc, "HeckeZEisUpChar tamper misalign_unweighted: vacuous (no degree >= 2 component and theta-weight-0 T_p)");
    }
    {
        ZPoly prod = {BigInt(1)};
        size_t sdeg = 0;
        for (size_t j = 0; j < nc; ++j) {
            prod = PolyMul(prod, fac[j]);
            sdeg += fac[j].size() - 1;
        }
        ENGINE_CHECK_MSG(sdeg == static_cast<size_t>(in.e_k), "HeckeZEisUpChar C-B4: the degrees of the proposed factors do not sum to e_k");
        ENGINE_CHECK_MSG(prod == eth, "HeckeZEisUpChar C-B4: e_theta != prod_j e_{theta,j} (the proposal is not the factorisation)");
    }

    // ---- C-B5, C-B8: g_{X,j} at the word primes where e_theta is separable ----
    auto ev_mod = [&](const ZPoly& p, u64 q) {
        std::vector<u64> v;
        for (const BigInt& c : p) v.push_back(ModOf(c, q));
        return v;
    };
    std::vector<uint32_t> sps;
    std::vector<std::vector<std::vector<std::vector<u64>>>> gres(nc, std::vector<std::vector<std::vector<u64>>>(m));   // [j][o][prime]
    uint64_t sep_cov = 0;
    bool any_sep = false;
    auto use_sep = [&](const PrimeRead& r) {
        const u64 q = r.q;
        if (!SeparableModQ(ev_mod(eth, q), q)) return;
        any_sep = true;
        std::vector<u64> tr(m, 0);
        for (size_t j = 0; j < nc; ++j) {
            const std::vector<u64> Ej = PolyOfMatrix(ev_mod(fac[j], q), r.theta, e, q);
            std::vector<size_t> fr;
            const auto basis = Kernel(Ej, e, e, q, &fr);
            ENGINE_CHECK_MSG(basis.size() == fac[j].size() - 1, "HeckeZEisUpChar C-B5: dim ker e_{theta,j}(theta) on K_q != deg e_{theta,j}");
            for (size_t o = 0; o < m; ++o) {
                const std::vector<u64> X = Restrict(r.R[o], e, basis, fr, q, "HeckeZEisUpChar: E_j mod q is not stable under an operator");
                const std::vector<u64> cp = CharpolyHessenberg(X, static_cast<uint32_t>(basis.size()), q);
                gres[j][o].push_back(cp);
                tr[o] = (tr[o] + cp[cp.size() - 2]) % q;   // - trace
            }
        }
        for (size_t o = 0; o < m; ++o) {   // C-B8: tr(X | K_q) = sum_j tr(X | E_j mod q)
            u64 t = 0;
            for (size_t x = 0; x < e; ++x) t = (t + r.R[o][x * e + x]) % q;
            ENGINE_CHECK_MSG((t + tr[o]) % q == 0, "HeckeZEisUpChar C-B8 (decomposition consistency): tr(X | K_q) != sum_j tr(X | E_j mod q)");
        }
        sps.push_back(static_cast<uint32_t>(q));
        sep_cov += BitLen64(q) - 1u;
    };
    for (const PrimeRead& r : reads) use_sep(r);
    while (sep_cov < need)
        if (next_read()) use_sep(reads.back());
    ENGINE_CHECK_MSG(any_sep, "HeckeZEisUpChar: e_theta separable at no word prime");
    R.sep_used = static_cast<uint32_t>(sps.size());
    std::vector<std::vector<ZPoly>> g(nc, std::vector<ZPoly>(m));
    for (size_t j = 0; j < nc; ++j)
        for (size_t o = 0; o < m; ++o) {
            g[j][o] = crt(sps, gres[j][o], bound);
            ENGINE_CHECK_MSG(g[j][o] == gprop[j][o], "HeckeZEisUpChar C-B5: g_{X,j} = charpoly(X | E_j) differs from the proposal");
        }

    // ---- S (Sec. 2.8) and q* (Sec. 2.7) ----
    std::vector<uint32_t> Tps(in.ops.begin(), in.ops.begin() + static_cast<long>(nT));
    R.S = ExtraPrimesS(N, Tps);
    for (uint32_t p : in.validation_extra_S) {
        ENGINE_CHECK_MSG(N % p != 0 && std::find(in.ops.begin(), in.ops.end(), p) == in.ops.end(), "HeckeZEisUpChar: validation_extra_S");
        R.S.push_back(p);
    }
    std::vector<uint32_t> opsS = in.ops;
    opsS.insert(opsS.end(), R.S.begin(), R.S.end());
    const std::vector<Op> allops = OpsOf(C, opsS);
    // the symbolic roots of every g_{X,j}: the distinct (kind, exponent) over the orbit
    std::vector<std::vector<std::vector<std::pair<int32_t, uint32_t>>>> sym(nc, std::vector<std::vector<std::pair<int32_t, uint32_t>>>(m));
    for (size_t j = 0; j < nc; ++j)
        for (size_t o = 0; o < m; ++o) {
            std::set<std::pair<int32_t, uint32_t>> s;
            for (uint32_t u : comps[j].units) s.insert(SymRoot(C, C.power(comps[j].a, u), comps[j].y, ops[o]));
            sym[j][o].assign(s.begin(), s.end());
        }
    auto sym_res = [&](const std::pair<int32_t, uint32_t>& r, const Op& X, const std::vector<u64>& zp, u64 q) -> u64 {
        const uint32_t o = C.o_N;
        if (r.first == 0) return 0;
        if (r.first == 1) return zp[r.second];
        const u64 Q = PowMod(X.p % q, k - 1, q);
        if (r.first == 2) return zp[(o - r.second) % o] * Q % q;
        return (zp[r.second] + zp[(o - r.second) % o] * Q) % q;
    };
    PrimeRead S;
    std::vector<u64> zs;
    std::vector<std::vector<u64>> Sread;   // the T_p~, p~ in S, restricted to K_q*
    for (size_t i = 0;; ++i) {
        if (i >= in.rs.word_prime_cap) throw WordPrimeCapRefusal(in.rs.word_prime_cap);
        bool extra_run = false;
        ExtendWordPrimes(primes, std::max(primes.size(), i + 1), N);
        const uint32_t q = primes[i];
        if (!IdPrimeArithmeticOk(q, C.o_N, k, R.S)) continue;
        if (tz.qstar_none) continue;   // (validation, audit m5) no candidate admitted: the chooser's own cap refusal
        PrimeRead pr;
        bool ok = false;
        std::vector<std::vector<u64>> sr;
        if (R.S.empty() && i < read_of_index.size()) {
            if (read_of_index[i] < 0) continue;
            pr = reads[static_cast<size_t>(read_of_index[i])];
            ok = true;
        } else {
            const PrimeRun run = RunAtPrime(N, n, opsS, q);
            ++R.computed;
            extra_run = true;
            ENGINE_CHECK_MSG(run.h1 == in.b1 && run.mats.size() == opsS.size(), "HeckeZEisUpChar: dim H^1 at q* differs from b1");
            ok = ReadAtPrime(run, m, R.P_eis, e, in.theta, &pr);
            if (ok) {
                std::vector<u64> pe;
                for (const auto& h : R.P_eis) pe.push_back(ModOf(FromHex(h), q));
                std::vector<size_t> fr;
                const auto basis = Kernel(PolyOfMatrix(pe, run.mats[0], run.h1, q), run.h1, run.h1, q, &fr);
                for (size_t s = 0; s < R.S.size(); ++s)
                    sr.push_back(Restrict(run.mats[m + s], run.h1, basis, fr, q, "HeckeZEisUpChar: K_q* is not stable under T_p~"));
            }
        }
        if (!ok) continue;
        if (!SeparableModQ(ev_mod(eth, q), q)) continue;
        const u64 gq = PrimitiveRoot(q, in.embed_next_root ? 1 : 0);
        const u64 z = PowMod(gq, (q - 1) / C.o_N, q);
        std::vector<u64> zp(C.o_N);
        for (uint32_t t = 0, w = 1; t < C.o_N; ++t) {
            zp[t] = w;
            w = static_cast<uint32_t>(w * z % q);
        }
        bool incong = true;
        for (size_t j = 0; j < nc && incong; ++j)
            for (size_t o = 0; o < m && incong; ++o) {
                std::set<u64> rs;
                for (const auto& r : sym[j][o]) rs.insert(sym_res(r, ops[o], zp, q));
                incong = rs.size() == sym[j][o].size();
            }
        if (!incong) continue;
        S = std::move(pr);
        zs = zp;
        Sread = std::move(sr);
        R.id_prime = q;
        R.id_prime_extra = extra_run ? 1 : 0;
        break;
    }

    // ---- C-B6: the identification at q* ----
    const u64 qs = S.q;
    std::vector<int64_t> line_psi_cond(nc, -1), line_psi_ord(nc, -1);
    std::vector<uint32_t> lines_per(nc, 0);
    {
        // every closed-form system (psi, y) with its theta-value mod q*
        std::vector<std::vector<uint32_t>> all = C.all();
        size_t nlines = 0;
        for (size_t j = 0; j < nc; ++j)
            for (uint32_t u : comps[j].units) {
                const std::vector<uint32_t> a = C.power(comps[j].a, u);
                const u64 mu = ComboMod(C, a, comps[j].y, ops, in.theta, zs, qs);
                std::vector<u64> Am = S.theta;
                for (size_t x = 0; x < e; ++x) Am[x * e + x] = (Am[x * e + x] + qs - mu) % qs;
                std::vector<size_t> fr;
                const auto ker = Kernel(Am, e, e, qs, &fr);
                ENGINE_CHECK_MSG(ker.size() == 1, "HeckeZEisUpChar C-B6: the theta-value of a closed-form system is no simple eigenvalue on K_q*");
                const std::vector<u64>& v = ker[0];
                size_t piv = 0;
                while (v[piv] == 0) ++piv;
                // the component of the line: the unique j' with e_{theta,j'}(mu) = 0 mod q*
                size_t jl = nc;
                for (size_t j2 = 0; j2 < nc; ++j2)
                    if (EvalModQ(ev_mod(fac[j2], qs), mu, qs) == 0) {
                        ENGINE_CHECK_MSG(jl == nc, "HeckeZEisUpChar C-B6: the theta-eigenvalue of a line is a root of two factors mod q*");
                        jl = j2;
                    }
                ENGINE_CHECK_MSG(jl < nc, "HeckeZEisUpChar C-B6: the theta-eigenvalue of a line is a root of no factor mod q*");
                auto scalar = [&](const std::vector<u64>& M) {
                    std::vector<u64> w(e, 0);
                    for (size_t r = 0; r < e; ++r)
                        for (size_t c = 0; c < e; ++c) w[r] = (w[r] + M[r * e + c] * v[c]) % qs;
                    const u64 s = w[piv] * InvMod(v[piv], qs) % qs;
                    for (size_t r = 0; r < e; ++r)
                        ENGINE_CHECK_MSG(w[r] == s * v[r] % qs, "HeckeZEisUpChar C-B6: an operator does not act by a scalar on a theta-eigenline");
                    return s;
                };
                // the joint system read: for X of the list, the unique root of g_{X,jl} in its class (symbolic)
                std::map<u64, uint32_t> texp;   // p (T_p, p !| N) -> exponent of psi(p) read
                std::vector<int32_t> yread(C.Up.size(), -1);
                std::vector<uint32_t> uexp(C.Up.size(), 0);
                for (size_t o = 0; o < m; ++o) {
                    const u64 s = scalar(S.R[o]);
                    int found = -1;
                    for (size_t t = 0; t < sym[jl][o].size(); ++t)
                        if (sym_res(sym[jl][o][t], ops[o], zs, qs) == s) found = static_cast<int>(t);
                    ENGINE_CHECK_MSG(found >= 0, "HeckeZEisUpChar C-B6: the scalar of an operator on a line is no root of g_{X,j} mod q*");
                    const auto& r = sym[jl][o][static_cast<size_t>(found)];
                    if (!ops[o].U) texp[ops[o].p] = r.second;
                    else {
                        yread[ops[o].ip] = r.first;
                        uexp[ops[o].ip] = r.second;
                    }
                }
                for (size_t s = 0; s < R.S.size(); ++s) {   // (6') the T_p~ at q* only: psi(p~) from its residue
                    const u64 sv = scalar(Sread[s]);
                    int found = -1;
                    const u64 Q = PowMod(R.S[s] % qs, k - 1, qs);
                    for (uint32_t t = 0; t < C.o_N; ++t)
                        if ((zs[t] + zs[(C.o_N - t) % C.o_N] * Q) % qs == sv) {
                            ENGINE_CHECK_MSG(found < 0, "HeckeZEisUpChar (6'): the residue of T_p~ does not determine psi(p~)");
                            found = static_cast<int>(t);
                        }
                    ENGINE_CHECK_MSG(found >= 0, "HeckeZEisUpChar (6'): the residue of T_p~ is no lambda_psi(T_p~)");
                    texp[R.S[s]] = static_cast<uint32_t>(found);
                }
                // (7) psi of the line: the unique character with these values
                const std::vector<uint32_t>* psi = nullptr;
                for (const auto& b : all) {
                    bool okb = true;
                    for (const auto& [p, t] : texp) okb = okb && C.exp_of(b, p) == t;
                    if (okb) {
                        ENGINE_CHECK_MSG(!psi, "HeckeZEisUpChar (7): the classes of P and S do not determine psi (they do not generate)");
                        psi = &b;
                    }
                }
                ENGINE_CHECK_MSG(psi, "HeckeZEisUpChar (7): no character mod s_N has the values read on a line");
                // compare with the closed form: the system (a, y) of this mu, operator by operator on the T_p of the list (implied by
                // *psi == a for the closed form itself; it is what sees a closed form misaligned on a T_p that theta does not weigh)
                bool tp_eq = true;
                for (size_t o = 0; o < nT; ++o) {
                    const std::vector<uint32_t> ao = (j == jt && o == ot) ? C.power(a, vt) : a;
                    const auto it = texp.find(ops[o].p);
                    tp_eq = tp_eq && it != texp.end() && it->second == C.exp_of(ao, ops[o].p);
                }
                ENGINE_CHECK_MSG(jl == j && *psi == a && tp_eq, "HeckeZEisUpChar C-B6: the character read on a line is not that of the closed form");
                for (size_t ip = 0; ip < C.Up.size(); ++ip) {
                    ENGINE_CHECK_MSG(yread[ip] == comps[j].y[ip], "HeckeZEisUpChar C-B6: the U_p type read on a line differs from the closed form");
                    if (yread[ip] != 0)
                        ENGINE_CHECK_MSG(uexp[ip] == C.exp_of(a, C.Up[ip]), "HeckeZEisUpChar C-B6: the U_p value read on a line differs");
                }
                const int64_t pc = C.cond(*psi), po = C.order(*psi);
                if (line_psi_cond[jl] < 0) {
                    line_psi_cond[jl] = pc;
                    line_psi_ord[jl] = po;
                }
                ENGINE_CHECK_MSG(line_psi_cond[jl] == pc && line_psi_ord[jl] == po, "HeckeZEisUpChar (7): cond / ord not constant on a component");
                ++lines_per[jl];
                ++nlines;
            }
        ENGINE_CHECK_MSG(nlines == e, "HeckeZEisUpChar C-B6: the lines of K_q* are not e_k");
    }
    if (tz.swap_labels) {   // (validation) swap the labels of two components of different order: the re-check below catches it
        for (size_t j = 0; j + 1 < nc; ++j)
            for (size_t j2 = j + 1; j2 < nc; ++j2)
                if (line_psi_ord[j] != line_psi_ord[j2]) {
                    std::swap(line_psi_cond[j], line_psi_cond[j2]);
                    std::swap(line_psi_ord[j], line_psi_ord[j2]);
                    j = nc;
                    break;
                }
    }
    for (size_t j = 0; j < nc; ++j) {   // the labels: phi(order) = degree, cond | s_N, order | o_N
        uint32_t phi = 0;
        for (uint32_t u = 1; u <= static_cast<uint32_t>(line_psi_ord[j]); ++u) phi += std::gcd(u, static_cast<uint32_t>(line_psi_ord[j])) == 1;
        ENGINE_CHECK_MSG(phi == fac[j].size() - 1 && lines_per[j] == phi && C.s_N % line_psi_cond[j] == 0 && C.o_N % line_psi_ord[j] == 0 &&
                             line_psi_cond[j] == comps[j].cond && line_psi_ord[j] == comps[j].ord,
                         "HeckeZEisUpChar C-B6: psi_cond / psi_order of a component inconsistent (phi(order) = degree, cond | s_N, order | o_N)");
    }
    if (tz.forge_U_deg2)
        for (size_t j = 0; j < nc; ++j)
            if (fac[j].size() > 2 && in.nU) {
                ZPoly x1 = {BigInt(1)};
                for (size_t t = 0; t + 1 < fac[j].size(); ++t) x1 = PolyMul(x1, ZPoly{BigInt(-1), BigInt(1)});
                g[j][nT] = x1;
                break;
            }
    for (size_t j = 0; j < nc; ++j)   // the written g_{U_p,j} are the certified ones (re-check after every step that may touch them)
        for (size_t o = nT; o < m; ++o)
            ENGINE_CHECK_MSG(g[j][o] == gprop[j][o], "HeckeZEisUpChar C-B5: a written U_p charpoly is not the certified g_{U_p,j}");

    // ---- the result: rows ----
    std::vector<EisCharComp> out;
    for (size_t j = 0; j < nc; ++j) {
        EisCharComp c;
        c.degree = static_cast<uint32_t>(fac[j].size() - 1);
        c.psi_cond = static_cast<uint32_t>(line_psi_cond[j]);
        c.psi_order = static_cast<uint32_t>(line_psi_ord[j]);
        c.theta_poly = PolyHex(fac[j]);
        c.a = comps[j].a;
        c.y = comps[j].y;
        {   // the minimal polynomial of T_q0 on E_j: the distinct values over the orbit
            std::set<uint32_t> ts;
            for (uint32_t u : comps[j].units) ts.insert(C.exp_of(C.power(comps[j].a, u), q0));
            std::vector<uint32_t> tv(ts.begin(), ts.end());
            const uint64_t D = tv.size();
            const ZPoly mp = AuxCrt(C.o_N, tv.size() + 1, D + D * BitLen(BigInt::pow(BigInt::from_u64(q0), k - 1) + BigInt(1)) + 64,
                                    [&](u64 M, const std::vector<u64>& zp) {
                                        std::vector<u64> roots;
                                        const u64 Q = PowMod(q0 % M, k - 1, M);
                                        for (uint32_t t : tv) roots.push_back((zp[t] + zp[(C.o_N - t) % C.o_N] * Q) % M);
                                        return ProdLinearMod(roots, M);
                                    });
            // (audit m1) minpoly against the certified g_{T_q0,j} (ops[0] = T_q0): minpoly | g and g | minpoly^degree, exactly over Z
            bool rz1 = false, rz2 = false;
            PolyDivMonic(g[j][0], mp, &rz1);
            ZPoly mpd = {BigInt(1)};
            for (size_t t = 0; t + 1 < fac[j].size(); ++t) mpd = PolyMul(mpd, mp);
            PolyDivMonic(mpd, g[j][0], &rz2);
            ENGINE_CHECK_MSG(rz1 && rz2, "HeckeZEisUpChar: minpoly(T_q0 | E_j) is not the radical of the certified g_{T_q0,j}");
            c.minpoly = PolyHex(mp);
        }
        for (size_t o = nT; o < m; ++o) c.Ucharpoly.push_back(PolyHex(g[j][o]));
        if (c.degree == 1) {
            c.lambda_theta = ToHex(-fac[j][0]);
            for (size_t o = nT; o < m; ++o) c.U.push_back(ToHex(-g[j][o][0]));
        }
        out.push_back(std::move(c));
    }
    std::sort(out.begin(), out.end(), [](const EisCharComp& a, const EisCharComp& b) {
        if ((a.degree == 1) != (b.degree == 1)) return a.degree == 1;
        if (a.degree == 1) {
            for (size_t i = 0; i < a.U.size(); ++i) {
                const BigInt x = FromHex(a.U[i]), y = FromHex(b.U[i]);
                if (x != y) return x < y;
            }
            return FromHex(a.lambda_theta) < FromHex(b.lambda_theta);
        }
        if (a.degree != b.degree) return a.degree < b.degree;
        if (a.psi_cond != b.psi_cond) return a.psi_cond < b.psi_cond;
        if (a.psi_order != b.psi_order) return a.psi_order < b.psi_order;
        return PolyLess(a.theta_poly, b.theta_poly);
    });
    R.comps = std::move(out);
    R.e_theta = PolyHex(eth);
    R.c_theta = PolyHex(cth);
    R.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return R;
}

std::vector<std::string> LambdaPolyOf(const EisUpCharResult& R, size_t j, const std::vector<uint32_t>& thP, const std::vector<uint32_t>& coeffs) {
    ENGINE_CHECK_MSG(j < R.comps.size() && coeffs.size() == thP.size(), "HeckeZEisUpChar LambdaPolyOf: arguments");
    const Chars C = MakeChars(R.N, R.n + 2);
    const std::vector<Op> ops = OpsOf(C, thP);
    const EisCharComp& c = R.comps[j];
    std::vector<uint32_t> units;
    const uint32_t ord = C.order(c.a);
    for (uint32_t u = 1; u <= ord; ++u)
        if (std::gcd(u, ord) == 1) units.push_back(u % ord);
    return PolyHex(OrbitPoly(C, c.a, c.y, units, ops, coeffs, ComboBits(C, ops, coeffs)));
}

std::vector<uint64_t> PolyModEll(const std::vector<std::string>& A, uint64_t ell) {
    std::vector<uint64_t> o;
    for (const auto& h : A) o.push_back(ModOf(FromHex(h), ell));
    return o;
}

std::vector<std::string> PolyDivExact(const std::vector<std::string>& A, const std::vector<std::string>& B) {
    bool rz = false;
    const ZPoly q = PolyDivMonic(PolyFromHex(A), PolyFromHex(B), &rz);
    ENGINE_CHECK_MSG(rz, "HeckeZEisUpChar: a polynomial division over Z with a remainder");
    return PolyHex(q);
}

namespace {
BigInt ResultantZ(const ZPoly& A, const ZPoly& B);
}
uint32_t ResultantValuation(const std::vector<std::string>& A0, const std::vector<std::string>& B0, uint64_t ell) {
    const BigInt det = ResultantZ(PolyFromHex(A0), PolyFromHex(B0));
    ENGINE_CHECK_MSG(ell >= 2 && ell < (1ull << 63), "HeckeZEisUpChar: ell < 2^63");
    return ValuationU64(det, ell);
}
uint32_t DiscValuation(const std::vector<std::string>& f0, uint64_t ell) {
    const ZPoly f = PolyFromHex(f0);
    ZPoly fp;
    for (size_t i = 1; i < f.size(); ++i) fp.push_back(f[i] * BigInt(static_cast<int64_t>(i)));
    return ValuationU64(ResultantZ(f, fp), ell);
}
namespace {
BigInt ResultantZ(const ZPoly& A, const ZPoly& B) {
    const size_t a = A.size() - 1, b = B.size() - 1, n = a + b;
    ENGINE_CHECK_MSG(n >= 1, "HeckeZEisUpChar: resultant of constants");
    // the Sylvester matrix (b rows of A, a rows of B), coefficients high to low
    std::vector<std::vector<BigInt>> M(n, std::vector<BigInt>(n, BigInt(0)));
    for (size_t r = 0; r < b; ++r)
        for (size_t i = 0; i <= a; ++i) M[r][r + i] = A[a - i];
    for (size_t r = 0; r < a; ++r)
        for (size_t i = 0; i <= b; ++i) M[b + r][r + i] = B[b - i];
    // Bareiss
    BigInt prev(1);
    bool neg = false;
    for (size_t c = 0; c + 1 < n; ++c) {
        size_t piv = c;
        while (piv < n && M[piv][c].is_zero()) ++piv;
        if (piv == n) {
            ENGINE_CHECK_MSG(false, "HeckeZEisUpChar: Res = 0 (two components share a root)");
        }
        if (piv != c) {
            std::swap(M[piv], M[c]);
            neg = !neg;
        }
        for (size_t r = c + 1; r < n; ++r) {
            for (size_t x = c + 1; x < n; ++x) {
                BigInt t = M[r][x] * M[c][c] - M[r][c] * M[c][x];
                BigInt q, rem;
                BigInt::divmod(t, prev, q, rem);
                ENGINE_CHECK_MSG(rem.is_zero(), "HeckeZEisUpChar: Bareiss division not exact");
                M[r][x] = q;
            }
            M[r][c] = BigInt(0);
        }
        prev = M[c][c];
    }
    const BigInt det = M[n - 1][n - 1];
    ENGINE_CHECK_MSG(!det.is_zero(), "HeckeZEisUpChar: Res = 0 (two components share a root)");
    return neg ? -det : det;
}
}  // namespace

ClosedFormSummary ClosedForm(uint32_t N, uint32_t k) {
    ClosedFormSummary S;
    const Chars C = MakeChars(N, k);
    for (const Comp& c : Components(C)) {
        std::ostringstream o;
        o << "deg=" << c.units.size() << " cond=" << c.cond << " ord=" << c.ord << " y=";
        for (int32_t t : c.y) o << (t == 0 ? "0" : t == 1 ? "a" : "b");
        S.comps.push_back(o.str());
        S.systems += static_cast<int64_t>(c.units.size());
    }
    std::sort(S.comps.begin(), S.comps.end());
    S.cusps_minus = static_cast<int64_t>(Gamma0Cusps(N)) - (k == 2 ? 1 : 0);
    return S;
}

std::vector<uint32_t> ExtraPrimesS(uint32_t N, const std::vector<uint32_t>& Tprimes) {
    const Chars C = MakeChars(N, 2);
    std::vector<uint32_t> S;
    if (C.sp.empty()) return S;
    // elements of (Z/s_N)^x as index vectors; the subgroup generated, by closure
    auto vec = [&](u64 x) {
        std::vector<uint32_t> v;
        for (size_t i = 0; i < C.sp.size(); ++i) v.push_back(C.ind[i][x % C.sp[i]]);
        return v;
    };
    u64 phi = 1;
    for (uint32_t p : C.sp) phi *= p - 1;
    std::set<std::vector<uint32_t>> H = {std::vector<uint32_t>(C.sp.size(), 0)};
    auto add_gen = [&](const std::vector<uint32_t>& g) {
        std::vector<std::vector<uint32_t>> todo(H.begin(), H.end());
        while (!todo.empty()) {
            auto h = todo.back();
            todo.pop_back();
            for (size_t i = 0; i < h.size(); ++i) h[i] = (h[i] + g[i]) % (C.sp[i] - 1);
            if (H.insert(h).second) todo.push_back(h);
        }
    };
    for (uint32_t p : Tprimes)
        if (N % p != 0) add_gen(vec(p));
    for (uint32_t p = 2; H.size() < phi; ++p) {
        if (N % p == 0 || !IsPrime32(p) || std::find(Tprimes.begin(), Tprimes.end(), p) != Tprimes.end()) continue;
        if (H.count(vec(p))) continue;
        S.push_back(p);
        add_gen(vec(p));
    }
    return S;
}

bool IdPrimeArithmeticOk(uint32_t q, uint32_t o_N, uint32_t k, const std::vector<uint32_t>& S) {
    if (o_N == 0 || (q - 1) % o_N != 0) return false;
    for (uint32_t p : S) {
        if (p == q) return false;
        if (PowMod(p % q, static_cast<u64>(k - 1) * o_N, q) == 1) return false;
    }
    return true;
}

uint32_t DirectionRead(uint32_t o_N, uint32_t t, uint32_t c, uint32_t q) {
    ENGINE_CHECK_MSG((q - 1) % o_N == 0 && std::gcd(c, o_N) == 1, "HeckeZEisUpChar DirectionRead: arguments");
    const u64 z = PowMod(PrimitiveRoot(q), (q - 1) / o_N, q);   // zeta = z mod Q
    uint32_t cinv = 1;
    while (static_cast<u64>(cinv) * c % o_N != 1) ++cinv;
    const u64 zc = PowMod(z, cinv, q);                          // zeta = z^{c^{-1}} mod sigma_c Q (sigma_c zeta = zeta^c = z mod sigma_c Q)
    const u64 r = PowMod(z, t, q);                               // the residue of the line of psi(2) = zeta^t at Q
    for (uint32_t s = 0; s < o_N; ++s)
        if (PowMod(zc, s, q) == r) return s;                    // read at sigma_c Q as zeta^s
    ENGINE_CHECK_MSG(false, "HeckeZEisUpChar DirectionRead: no exponent");
    return 0;
}

}  // namespace eisup
}  // namespace heckez
