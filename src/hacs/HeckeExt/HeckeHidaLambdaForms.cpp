// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-HL) the q-expansion input arm (Q) of the
// truncated Lambda-adic Hecke algebra: Z/ell^W arithmetic, the multimodular
// NTT series products, the Miller basis, T_p on q-expansions, the ordinary
// projector.  A port of docs/implementation/slot19/r192/forms.py (R19-2), whose
// matrices (r192/data/forms_*.json) HeckeLambdaValidation reproduces exactly.
// See HeckeHidaLambda.h for the contract references.
#include "HeckeHidaLambda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <sstream>

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "../Util/CancelPoint.h"   // (NBCONFIRM S5) cooperative cancellation points
#include "NumberField/BigInt.h"
#include "NumberField/BigRational.h"

namespace hecke_ext {
namespace lambda {

std::string U128Text(u128 x) {
    if (x == 0) return "0";
    std::string s;
    while (x) {
        s.push_back(static_cast<char>('0' + static_cast<int>(x % 10)));
        x /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

// ===========================================================================
// ZMod
// ===========================================================================
namespace {
constexpr unsigned kHalf = 42;
constexpr uint64_t kHalfMask = (uint64_t{1} << kHalf) - 1;
const u128 kTwo84 = static_cast<u128>(1) << 84;
}  // namespace

ZMod::ZMod(uint64_t ell_, uint32_t W_) : ell(ell_), W(W_) {
    ENGINE_CHECK_MSG(ell >= 2 && W >= 1, "ZMod: ell >= 2, W >= 1");
    pw.assign(1, 1);
    for (uint32_t i = 0; i < W; ++i) {
        ENGINE_CHECK_MSG(pw.back() < kTwo84 / ell, "ZMod: ell^W must be < 2^84 (HeckeHidaLambda.h, ARITHMETIC)");
        pw.push_back(pw.back() * ell);
    }
    M = pw.back();
}

u128 ZMod::mul(u128 a, u128 b) const {
    if (M <= (static_cast<u128>(1) << 64)) return (a * b) % M;
    const uint64_t a1 = static_cast<uint64_t>(a >> kHalf), a0 = static_cast<uint64_t>(a) & kHalfMask;
    const uint64_t b1 = static_cast<uint64_t>(b >> kHalf), b0 = static_cast<uint64_t>(b) & kHalfMask;
    const u128 hh = static_cast<u128>(a1) * b1;
    const u128 mid = static_cast<u128>(a1) * b0 + static_cast<u128>(a0) * b1;
    const u128 ll = static_cast<u128>(a0) * b0;
    u128 r = hh % M;
    r = ((r << kHalf) + mid) % M;
    return ((r << kHalf) + ll) % M;
}

u128 ZMod::pow(u128 a, uint64_t e) const {
    u128 r = 1 % M, b = a % M;
    while (e) {
        if (e & 1u) r = mul(r, b);
        e >>= 1;
        if (e) b = mul(b, b);
    }
    return r;
}

u128 ZMod::inv(u128 a) const {
    a %= M;
    ENGINE_CHECK_MSG(a % ell != 0, "ZMod::inv: not a unit mod ell");
    // inverse mod ell by Fermat, then Newton lifting x <- x (2 - a x)
    uint64_t x0 = 1;
    {
        // inverse mod ell by Fermat
        uint64_t b = static_cast<uint64_t>(a % ell), r1 = 1, ex = ell - 2;
        while (ex) {
            if (ex & 1u) r1 = static_cast<uint64_t>(static_cast<u128>(r1) * b % ell);
            ex >>= 1;
            b = static_cast<uint64_t>(static_cast<u128>(b) * b % ell);
        }
        x0 = r1;
    }
    u128 x = x0;
    for (uint32_t prec = 1; prec < W; prec *= 2) x = mul(x, sub(2 % M, mul(a, x)));
    ENGINE_CHECK(mul(a, x) == 1 % M);
    return x;
}

u128 ZMod::from_i64(int64_t v) const {
    if (v >= 0) return static_cast<u128>(static_cast<uint64_t>(v)) % M;
    const u128 r = static_cast<u128>(static_cast<uint64_t>(-(v + 1)) + 1) % M;
    return neg(r);
}

uint32_t ZMod::val(u128 x, uint32_t cap) const {
    if (x == 0) return cap;
    uint32_t e = 0;
    while (x % ell == 0) {
        x /= ell;
        ++e;
    }
    return e;
}

namespace {
// split-accumulated dot product: sum_i a_i b_i mod M (entries < M < 2^84)
struct Acc {
    u128 hh = 0, mid = 0, ll = 0;
    void add(u128 a, u128 b) {
        const uint64_t a1 = static_cast<uint64_t>(a >> kHalf), a0 = static_cast<uint64_t>(a) & kHalfMask;
        const uint64_t b1 = static_cast<uint64_t>(b >> kHalf), b0 = static_cast<uint64_t>(b) & kHalfMask;
        hh += static_cast<u128>(a1) * b1;
        mid += static_cast<u128>(a1) * b0 + static_cast<u128>(a0) * b1;
        ll += static_cast<u128>(a0) * b0;
    }
    u128 get(const ZMod& R) const {
        u128 r = hh % R.M;
        r = ((r << kHalf) + mid % R.M) % R.M;
        return ((r << kHalf) + ll % R.M) % R.M;
    }
};
}  // namespace

// ===========================================================================
// linear algebra
// ===========================================================================
Mat Ident(size_t n) {
    Mat I(n, Vec(n, 0));
    for (size_t i = 0; i < n; ++i) I[i][i] = 1;
    return I;
}

Mat MatMul(const Mat& A, const Mat& B, const ZMod& R) {
    const size_t n = A.size();
    if (n == 0) return {};
    const size_t inner = B.size();
    const size_t m = inner ? B[0].size() : 0;
    ENGINE_CHECK(A[0].size() == inner);
    // reduce once, B transposed for a contiguous inner loop
    Mat Ar(n, Vec(inner)), Bt(m, Vec(inner));
    for (size_t i = 0; i < n; ++i)
        for (size_t t = 0; t < inner; ++t) Ar[i][t] = A[i][t] < R.M ? A[i][t] : A[i][t] % R.M;
    for (size_t t = 0; t < inner; ++t)
        for (size_t j = 0; j < m; ++j) Bt[j][t] = B[t][j] < R.M ? B[t][j] : B[t][j] % R.M;
    Mat C(n, Vec(m, 0));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j) {
            Acc acc;
            const u128* a = Ar[i].data();
            const u128* b = Bt[j].data();
            for (size_t t = 0; t < inner; ++t) acc.add(a[t], b[t]);
            C[i][j] = acc.get(R);
        }
    return C;
}

Vec MatVec(const Mat& A, const Vec& v, const ZMod& R) {
    Vec out(A.size(), 0);
    for (size_t i = 0; i < A.size(); ++i) {
        Acc acc;
        for (size_t j = 0; j < v.size(); ++j) acc.add(A[i][j] % R.M, v[j] % R.M);
        out[i] = acc.get(R);
    }
    return out;
}

Vec Charpoly(const Mat& A0, const ZMod& R) {
    const size_t n = A0.size();
    if (n == 0) return Vec{1 % R.M};
    Mat A(n, Vec(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) A[i][j] = A0[i][j] % R.M;
    Vec C{1 % R.M, R.neg(A[0][0])};   // high -> low
    for (size_t r = 1; r < n; ++r) {
        Vec t{1 % R.M, R.neg(A[r][r])};
        Vec v(r);
        for (size_t i = 0; i < r; ++i) v[i] = A[i][r];
        for (size_t s = 0; s < r; ++s) {
            Acc acc;
            for (size_t i = 0; i < r; ++i) acc.add(A[r][i], v[i]);
            t.push_back(R.neg(acc.get(R)));
            Vec nv(r);
            for (size_t i = 0; i < r; ++i) {
                Acc a2;
                for (size_t j = 0; j < r; ++j) a2.add(A[i][j], v[j]);
                nv[i] = a2.get(R);
            }
            v.swap(nv);
        }
        Vec nC(r + 2, 0);
        for (size_t i = 0; i < r + 2; ++i) {
            Acc acc;
            for (size_t j = 0; j < C.size(); ++j)
                if (i >= j && i - j < t.size()) acc.add(t[i - j], C[j]);
            nC[i] = acc.get(R);
        }
        C.swap(nC);
    }
    std::reverse(C.begin(), C.end());
    return C;
}

Mat InvUnitMatrix(const Mat& Rm, const ZMod& R) {
    const size_t n = Rm.size();
    Mat A(n, Vec(2 * n, 0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) A[i][j] = Rm[i][j] % R.M;
        A[i][n + i] = 1 % R.M;
    }
    for (size_t c = 0; c < n; ++c) {
        size_t piv = n;
        for (size_t i = c; i < n; ++i)
            if (A[i][c] % R.ell) {
                piv = i;
                break;
            }
        ENGINE_CHECK_MSG(piv < n, "InvUnitMatrix: not invertible mod ell");
        std::swap(A[c], A[piv]);
        const u128 iv = R.inv(A[c][c]);
        for (auto& x : A[c]) x = R.mul(x, iv);
        for (size_t i = 0; i < n; ++i) {
            if (i == c || A[i][c] == 0) continue;
            const u128 f = A[i][c];
            for (size_t j = 0; j < 2 * n; ++j) A[i][j] = R.sub(A[i][j], R.mul(f, A[c][j]));
        }
    }
    Mat out(n, Vec(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) out[i][j] = A[i][n + j];
    return out;
}

std::vector<size_t> IndepColsModEll(const Mat& P, size_t ncols, uint64_t ell) {
    const size_t n = P.size();
    std::vector<size_t> cols;
    std::vector<std::pair<size_t, std::vector<uint64_t>>> ech;
    auto inv_l = [ell](uint64_t a) {
        uint64_t r = 1, b = a % ell, e = ell - 2;
        while (e) {
            if (e & 1u) r = static_cast<uint64_t>(static_cast<u128>(r) * b % ell);
            e >>= 1;
            b = static_cast<uint64_t>(static_cast<u128>(b) * b % ell);
        }
        return r;
    };
    for (size_t j = 0; j < ncols; ++j) {
        std::vector<uint64_t> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint64_t>(P[i][j] % ell);
        for (const auto& pr : ech) {
            const uint64_t f = v[pr.first];
            if (!f) continue;
            for (size_t i = 0; i < n; ++i)
                v[i] = (v[i] + ell - static_cast<uint64_t>(static_cast<u128>(f) * pr.second[i] % ell)) % ell;
        }
        size_t nz = n;
        for (size_t i = 0; i < n; ++i)
            if (v[i]) {
                nz = i;
                break;
            }
        if (nz == n) continue;
        const uint64_t iv = inv_l(v[nz]);
        for (auto& x : v) x = static_cast<uint64_t>(static_cast<u128>(x) * iv % ell);
        ech.emplace_back(nz, std::move(v));
        cols.push_back(j);
    }
    return cols;
}

// ===========================================================================
// multimodular NTT (62-bit primes, Montgomery), Garner directly mod ell^W
// ===========================================================================
namespace {

inline uint64_t MulModU64(uint64_t a, uint64_t b, uint64_t m) { return engine::scalar::mulmod(a, b, m); }
inline uint64_t PowModU64(uint64_t a, uint64_t e, uint64_t m) { return engine::scalar::powmod<uint64_t>(a, e, m); }
bool IsPrimeU64(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull})
        if (n % p == 0) return n == p;
    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1u) == 0) {
        d >>= 1;
        ++s;
    }
    for (uint64_t a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        uint64_t x = PowModU64(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (unsigned r = 1; r < s; ++r) {
            x = MulModU64(x, x, n);
            if (x == n - 1) {
                comp = false;
                break;
            }
        }
        if (comp) return false;
    }
    return true;  // deterministic for n < 3.3e24 with these bases
}

constexpr unsigned kNttLog = 25;   // supports transforms up to 2^25

struct NttPrime {
    uint64_t p = 0, pinv = 0, r2 = 0, one = 0;   // Montgomery, R = 2^64
    uint64_t g = 0;                              // primitive root
    double log2p = 0;
    uint64_t mul(uint64_t a, uint64_t b) const {
        const u128 t = static_cast<u128>(a) * b;
        const uint64_t m = static_cast<uint64_t>(t) * pinv;
        const uint64_t u = static_cast<uint64_t>((t + static_cast<u128>(m) * p) >> 64);
        return u >= p ? u - p : u;
    }
    uint64_t k42 = 0;                            // 2^42 R^2 mod p
    uint64_t to(uint64_t a) const { return mul(a % p, r2); }
    // Montgomery form of x < 2^84: x = hi 2^42 + lo with hi, lo < 2^42 < p
    uint64_t to128(u128 x) const {
        const uint64_t hi = static_cast<uint64_t>(x >> 42), lo = static_cast<uint64_t>(x) & ((uint64_t{1} << 42) - 1);
        return add(mul(hi, k42), mul(lo, r2));
    }
    uint64_t from(uint64_t a) const { return mul(a, 1); }
    uint64_t add(uint64_t a, uint64_t b) const { const uint64_t s = a + b; return s >= p ? s - p : s; }
    uint64_t sub(uint64_t a, uint64_t b) const { return a >= b ? a - b : a + p - b; }
    uint64_t powm(uint64_t a, uint64_t e) const {   // Montgomery in, Montgomery out
        uint64_t r = one;
        while (e) {
            if (e & 1u) r = mul(r, a);
            e >>= 1;
            a = mul(a, a);
        }
        return r;
    }
};

NttPrime MakePrime(uint64_t p) {
    NttPrime P;
    P.p = p;
    uint64_t inv = p;
    for (int i = 0; i < 6; ++i) inv *= 2 - p * inv;
    P.pinv = ~inv + 1;   // -p^{-1} mod 2^64
    const uint64_t r1 = static_cast<uint64_t>((static_cast<u128>(1) << 64) % p);
    P.r2 = MulModU64(r1, r1, p);
    P.one = r1;
    P.k42 = P.mul(P.to(uint64_t{1} << 42), P.r2);   // (2^42 R) R^2 R^{-1} = 2^42 R^2
    // primitive root: p - 1 = c 2^25
    uint64_t c = (p - 1) >> kNttLog;
    std::vector<uint64_t> fac{2};
    for (uint64_t d = 2; d * d <= c; ++d)
        if (c % d == 0) {
            if (d != 2) fac.push_back(d);
            while (c % d == 0) c /= d;
        }
    if (c > 1 && c != 2) fac.push_back(c);
    for (uint64_t g = 2;; ++g) {
        bool ok = true;
        for (uint64_t q : fac)
            if (PowModU64(g, (p - 1) / q, p) == 1) {
                ok = false;
                break;
            }
        if (ok) {
            P.g = g;
            break;
        }
    }
    P.log2p = std::log2(static_cast<double>(p));
    return P;
}

const std::vector<NttPrime>& Primes() {
    static const std::vector<NttPrime> primes = [] {
        std::vector<NttPrime> v;
        uint64_t c = ((uint64_t{1} << 62) - 1) >> kNttLog;
        while (v.size() < 4) {
            const uint64_t p = (c << kNttLog) + 1;
            if (IsPrimeU64(p)) v.push_back(MakePrime(p));
            --c;
        }
        return v;
    }();
    return primes;
}

void Ntt(std::vector<uint64_t>& a, bool invert, const NttPrime& P) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const uint64_t gm = P.to(P.g);
    std::vector<uint64_t> tw;
    for (size_t len = 2; len <= n; len <<= 1) {
        uint64_t w = P.powm(gm, (P.p - 1) / len);
        if (invert) w = P.powm(w, P.p - 2);
        const size_t h = len / 2;
        tw.resize(h);
        tw[0] = P.one;
        for (size_t j = 1; j < h; ++j) tw[j] = P.mul(tw[j - 1], w);
        for (size_t i = 0; i < n; i += len)
            for (size_t j = 0; j < h; ++j) {
                const uint64_t u = a[i + j], v = P.mul(a[i + j + h], tw[j]);
                a[i + j] = P.add(u, v);
                a[i + j + h] = P.sub(u, v);
            }
    }
    if (invert) {
        const uint64_t ninv = P.powm(P.to(n % P.p), P.p - 2);
        for (auto& x : a) x = P.mul(x, ninv);
    }
}

size_t NttPrimeCount(size_t minlen, const ZMod& R) {
    const double need = 2.0 * std::log2(static_cast<double>(R.M)) + std::log2(static_cast<double>(minlen)) + 2.0;
    double have = 0;
    for (size_t i = 0; i < Primes().size(); ++i) {
        have += Primes()[i].log2p - 0.01;
        if (have > need) return i + 1;
    }
    ENGINE_CHECK_MSG(false, "HeckeHidaLambda NTT: four 62-bit primes do not cover the product bound");
    return 0;
}

// Garner: residues (normal form) -> the integer mod ell^W.
struct Garner {
    size_t k = 0;
    std::vector<std::vector<uint64_t>> invm;  // invm[i][j] = p_j^{-1} R mod p_i (Montgomery form), j < i
    std::vector<u128> prefM;                  // prod_{j<i} p_j mod M
    bool small = false;                       // M < 2^62: v_i reduced by a 64-bit %
    Garner(size_t k_, const ZMod& R) : k(k_) {
        const auto& P = Primes();
        invm.assign(k, {});
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < i; ++j) invm[i].push_back(P[i].to(PowModU64(P[j].p % P[i].p, P[i].p - 2, P[i].p)));
        prefM.assign(k, 1 % R.M);
        for (size_t i = 1; i < k; ++i) prefM[i] = R.mul(prefM[i - 1], static_cast<u128>(P[i - 1].p) % R.M);
        small = R.M < (static_cast<u128>(1) << 62);
    }
    u128 operator()(const uint64_t* r, const ZMod& R) const {
        const auto& P = Primes();
        uint64_t v[4];
        for (size_t i = 0; i < k; ++i) {
            const NttPrime& Pi = P[i];
            uint64_t t = r[i];
            for (size_t j = 0; j < i; ++j) {
                uint64_t vj = v[j];
                while (vj >= Pi.p) vj -= Pi.p;   // v_j < p_j < 2^62 < 2 p_i
                t = Pi.sub(t, vj);
                t = Pi.mul(t, invm[i][j]);       // t p_j^{-1}, normal form
            }
            v[i] = t;
        }
        Acc acc;
        for (size_t i = 0; i < k; ++i) {
            const u128 vi = small ? static_cast<u128>(v[i] % static_cast<uint64_t>(R.M)) : static_cast<u128>(v[i]);
            acc.add(vi, prefM[i]);
        }
        return acc.get(R);
    }
};

Vec MulLowSchool(const Vec& a, const Vec& b, size_t L, const ZMod& R) {
    const size_t la = std::min(a.size(), L), lb = std::min(b.size(), L);
    const size_t lo = (la && lb) ? std::min(L, la + lb - 1) : 0;
    Vec out(lo, 0);
    for (size_t t = 0; t < lo; ++t) {
        Acc acc;
        const size_t i0 = t + 1 > lb ? t + 1 - lb : 0;
        for (size_t i = i0; i <= std::min(t, la - 1); ++i) acc.add(a[i], b[t - i]);
        out[t] = acc.get(R);
    }
    return out;
}

// A fixed operand transformed once (the d products h <- h R of the basis).
struct NttOperand {
    size_t nt = 0, len = 0, k = 0;
    std::vector<std::vector<uint64_t>> f;
};

NttOperand Prepare(const Vec& b, size_t nt, size_t k) {
    NttOperand op;
    op.nt = nt;
    op.len = b.size();
    op.k = k;
    op.f.resize(k);
    for (size_t i = 0; i < k; ++i) {
        const NttPrime& P = Primes()[i];
        op.f[i].assign(nt, 0);
        for (size_t t = 0; t < b.size(); ++t) op.f[i][t] = P.to128(b[t]);
        Ntt(op.f[i], false, P);
    }
    return op;
}

Vec MulWithPrepared(const Vec& a, const NttOperand& B, size_t L, const ZMod& R) {
    ENGINE_CHECK(a.size() + B.len - 1 <= B.nt);
    const size_t k = B.k;
    std::vector<std::vector<uint64_t>> fa(k);
    for (size_t i = 0; i < k; ++i) {
        const NttPrime& P = Primes()[i];
        fa[i].assign(B.nt, 0);
        for (size_t t = 0; t < a.size(); ++t) fa[i][t] = P.to128(a[t]);
        Ntt(fa[i], false, P);
        for (size_t t = 0; t < B.nt; ++t) fa[i][t] = P.mul(fa[i][t], B.f[i][t]);
        Ntt(fa[i], true, P);
    }
    const size_t lo = std::min(L, a.size() + B.len - 1);
    Vec out(lo);
    const Garner G(k, R);
    uint64_t r[4];
    for (size_t t = 0; t < lo; ++t) {
        for (size_t i = 0; i < k; ++i) r[i] = Primes()[i].from(fa[i][t]);
        out[t] = G(r, R);
    }
    return out;
}

size_t Pow2AtLeast(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

Vec MulLow(const Vec& a0, const Vec& b0, size_t L, const ZMod& R) {
    const Vec a(a0.begin(), a0.begin() + std::min(a0.size(), L));
    const Vec b(b0.begin(), b0.begin() + std::min(b0.size(), L));
    if (a.empty() || b.empty()) return {};
    if (std::min(a.size(), b.size()) <= 48) return MulLowSchool(a, b, L, R);
    const size_t nt = Pow2AtLeast(a.size() + b.size() - 1);
    ENGINE_CHECK_MSG(nt <= (size_t{1} << kNttLog), "HeckeHidaLambda NTT: transform too long");
    const size_t k = NttPrimeCount(std::min(a.size(), b.size()), R);
    return MulWithPrepared(a, Prepare(b, nt, k), L, R);
}

Vec InverseSeries(const Vec& f, size_t L, const ZMod& R) {
    ENGINE_CHECK(!f.empty());
    Vec g{R.inv(f[0])};
    size_t cur = 1;
    while (cur < L) {
        const size_t c2 = std::min(2 * cur, L);
        const Vec fc(f.begin(), f.begin() + std::min(f.size(), c2));
        Vec e = MulLow(fc, g, c2, R);
        e.resize(c2, 0);
        for (auto& x : e) x = R.neg(x);
        e[0] = R.add(e[0], 2 % R.M);
        g = MulLow(g, e, c2, R);
        g.resize(c2, 0);
        cur = c2;
    }
    return g;
}

Vec SigmaList(size_t L, unsigned e, const ZMod& R) {
    Vec s(L, 0);
    for (size_t d = 1; d < L; ++d) {
        const u128 de = R.pow(d % R.M, e);
        for (size_t n = d; n < L; n += d) s[n] = R.add(s[n], de);
    }
    return s;
}

struct Eis {
    Vec e4, e6, E12, Delta, R;
};

Eis MakeSeries(size_t L, const ZMod& Rg) {
    Eis S;
    const Vec s3 = SigmaList(L, 3, Rg), s5 = SigmaList(L, 5, Rg);
    S.e4.assign(L, 0);
    S.e6.assign(L, 0);
    S.e4[0] = 1 % Rg.M;
    S.e6[0] = 1 % Rg.M;
    for (size_t n = 1; n < L; ++n) {
        S.e4[n] = Rg.mul(240 % Rg.M, s3[n]);
        S.e6[n] = Rg.neg(Rg.mul(504 % Rg.M, s5[n]));
    }
    S.E12 = MulLow(MulLow(S.e4, S.e4, L, Rg), S.e4, L, Rg);
    S.E12.resize(L, 0);
    Vec e6sq = MulLow(S.e6, S.e6, L, Rg);
    e6sq.resize(L, 0);
    const u128 i1728 = Rg.inv(1728 % Rg.M);
    S.Delta.assign(L, 0);
    for (size_t n = 0; n < L; ++n) S.Delta[n] = Rg.mul(Rg.sub(S.E12[n], e6sq[n]), i1728);
    S.R = MulLow(S.Delta, InverseSeries(S.E12, L, Rg), L, Rg);
    S.R.resize(L, 0);
    return S;
}

uint32_t DimS(uint32_t k) {
    if (k < 12 || k % 2) return 0;
    return k / 12 - (k % 12 == 2 ? 1 : 0);
}

}  // namespace

// ===========================================================================
// self-checks
// ===========================================================================
bool SeriesSelfCheck(uint64_t ell, uint32_t W, uint32_t n) {
    const ZMod R(ell, W);
    const Eis S = MakeSeries(n, R);
    Vec prod(n, 0);
    prod[0] = 1;
    for (size_t m = 1; m < n; ++m)
        for (int rep = 0; rep < 24; ++rep)
            for (size_t t = n - 1; t >= m; --t) prod[t] = R.sub(prod[t], prod[t - m]);
    bool ok = S.Delta[0] == 0;
    for (size_t t = 1; t < n; ++t) ok = ok && S.Delta[t] == prod[t - 1];
    const Vec e4c = MulLowSchool(MulLowSchool(S.e4, S.e4, n, R), S.e4, n, R);
    const Vec e6s = MulLowSchool(S.e6, S.e6, n, R);
    for (size_t t = 0; t < n; ++t)
        ok = ok && R.sub(R.sub(e4c[t], e6s[t]), R.mul(1728 % R.M, S.Delta[t])) == 0;
    return ok;
}

bool NttSelfCheck(uint64_t ell, uint32_t W) {
    const ZMod R(ell, W);
    std::mt19937_64 rng(20261004);
    bool ok = true;
    for (size_t la : {49u, 200u, 1000u, 3001u}) {
        Vec a(la), b(la + 17);
        for (auto& x : a) x = ((static_cast<u128>(rng()) << 64) | rng()) % R.M;
        for (auto& x : b) x = ((static_cast<u128>(rng()) << 64) | rng()) % R.M;
        a.back() = R.M - 1;
        b.back() = R.M - 1;
        const size_t L = la + 5;
        ok = ok && MulLow(a, b, L, R) == MulLowSchool(a, b, L, R);
        const size_t full = a.size() + b.size() - 1;
        ok = ok && MulLow(a, b, full, R) == MulLowSchool(a, b, full, R);
    }
    return ok;
}

bool CharpolySelfCheck() {
    std::mt19937_64 rng(1);
    const ZMod R(10007, 4);
    bool ok = true;
    for (size_t n = 1; n <= 6; ++n) {
        std::vector<std::vector<int64_t>> A(n, std::vector<int64_t>(n));
        for (auto& r : A)
            for (auto& x : r) x = static_cast<int64_t>(rng() % 100) - 50;
        Mat Am(n, Vec(n));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) Am[i][j] = R.from_i64(A[i][j]);
        const Vec cp = Charpoly(Am, R);
        for (int64_t y = -3; y <= 3; ++y) {
            // exact det(y I - A) by fraction-free Gaussian elimination over BigRational
            std::vector<std::vector<nf::BigRational>> D(n, std::vector<nf::BigRational>(n));
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j) D[i][j] = nf::BigRational((i == j ? y : 0) - A[i][j]);
            nf::BigRational det(1);
            for (size_t c = 0; c < n; ++c) {
                size_t p = n;
                for (size_t i = c; i < n; ++i)
                    if (!D[i][c].is_zero()) {
                        p = i;
                        break;
                    }
                if (p == n) {
                    det = nf::BigRational(0);
                    break;
                }
                if (p != c) {
                    std::swap(D[p], D[c]);
                    det = -det;
                }
                det *= D[c][c];
                for (size_t i = c + 1; i < n; ++i) {
                    const nf::BigRational f = D[i][c] / D[c][c];
                    for (size_t j = c; j < n; ++j) D[i][j] -= f * D[c][j];
                }
            }
            u128 val = 0;
            for (size_t i = cp.size(); i-- > 0;) val = R.add(R.mul(val, R.from_i64(y)), cp[i]);
            const int64_t dv = det.num().to_int64();
            ok = ok && val == R.from_i64(dv);
        }
    }
    return ok;
}

// ===========================================================================
// the forms side of one weight (forms.py weight_data)
// ===========================================================================
FormsWeight ComputeFormsWeight(uint64_t ell, uint32_t k, uint32_t W, const std::vector<uint64_t>& primes) {
    const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    ENGINE_CHECK_MSG(ell >= 5 && k >= 4 && k % 2 == 0, "ComputeFormsWeight: ell >= 5, k >= 4 even");
    ENGINE_CHECK_MSG(k - 1 >= W, "ComputeFormsWeight: U = T_ell on e M_k mod ell^W needs k - 1 >= W (REFUSED)");
    for (uint64_t p : primes) ENGINE_CHECK_MSG(p != ell && p >= 2, "ComputeFormsWeight: primes must avoid ell");
    const ZMod R(ell, W);
    const uint32_t d = DimS(k);
    const uint32_t r = d + 1;
    const uint32_t D = d + 2;   // two extra q-coefficients: the stability check
    std::vector<uint64_t> pl(primes);
    pl.push_back(ell);
    const uint64_t pmax = *std::max_element(pl.begin(), pl.end());
    const size_t L = static_cast<size_t>(pmax) * D + 1;
    FormsWeight out;
    out.ell = ell;
    out.k = k;
    out.W = W;
    out.dim_M = r;
    out.dim_S = d;
    out.L = L;
    const Eis S = MakeSeries(L, R);
    // weight split k - 12 d = 4 a + 6 b
    const uint32_t w = k - 12 * d;
    const uint32_t b = (w % 4 == 2) ? 1 : 0;
    const uint32_t a = (w - 6 * b) / 4;
    ENGINE_CHECK(4 * a + 6 * b == w);
    Vec C{1 % R.M};
    for (uint32_t i = 0; i < a; ++i) C = MulLow(C, S.e4, L, R);
    if (b) C = MulLow(C, S.e6, L, R);
    Vec E12d{1 % R.M};
    {
        Vec base = S.E12;
        uint32_t e = d;
        while (e) {
            if (e & 1u) E12d = MulLow(E12d, base, L, R);
            e >>= 1;
            if (e) base = MulLow(base, base, L, R);
        }
    }
    Vec h = MulLow(C, E12d, L, R);
    h.resize(L, 0);
    // the index set: 0..D and p n (p in pl, n <= D)
    std::vector<int32_t> pos(L, -1);
    std::vector<size_t> idx;
    for (uint32_t n = 0; n <= D; ++n) pos[n] = 0;
    for (uint64_t p : pl)
        for (uint32_t n = 0; n <= D; ++n) pos[p * n] = 0;
    for (size_t i = 0; i < L; ++i)
        if (pos[i] == 0) {
            pos[i] = static_cast<int32_t>(idx.size());
            idx.push_back(i);
        }
    std::vector<Vec> F(r, Vec(idx.size()));
    {
        const size_t nt = Pow2AtLeast(2 * L - 1);
        const size_t kp = NttPrimeCount(L, R);
        const bool small = L <= 48;
        const NttOperand Rop = small ? NttOperand{} : Prepare(S.R, nt, kp);
        for (uint32_t j = 0; j < r; ++j) {
            oc_util::CancelPoint("hida_lambda.basis");   // (NBCONFIRM S5) per Miller basis form (stack-local results only)
            for (size_t t = 0; t < idx.size(); ++t) F[j][t] = h[idx[t]];
            if (j + 1 < r) {
                h = small ? MulLowSchool(h, S.R, L, R) : MulWithPrepared(h, Rop, L, R);
                h.resize(L, 0);
            }
        }
    }
    auto Fat = [&](uint32_t j, size_t i) -> u128 { return F[j][static_cast<size_t>(pos[i])]; };
    for (uint32_t j = 0; j < r; ++j) {
        ENGINE_CHECK_MSG(Fat(j, j) == 1 % R.M, "Miller basis not unitriangular");
        for (uint32_t n = 0; n < j; ++n) ENGINE_CHECK_MSG(Fat(j, n) == 0, "Miller basis not unitriangular");
    }
    // solve: coordinates of the q-coefficients c[0..D] in the basis f_0..f_{r-1}
    auto solve = [&](const Vec& c, bool& ok) {
        Vec x(r, 0);
        for (uint32_t n = 0; n < r; ++n) {
            Acc acc;
            for (uint32_t j = 0; j < n; ++j) acc.add(x[j], Fat(j, n));
            x[n] = R.sub(c[n], acc.get(R));
        }
        ok = true;
        for (uint32_t n = r; n <= D; ++n) {
            Acc acc;
            for (uint32_t j = 0; j < r; ++j) acc.add(x[j], Fat(j, n));
            if (acc.get(R) != c[n]) ok = false;
        }
        return x;
    };
    auto hecke = [&](uint64_t p, const std::function<u128(size_t)>& coeff) {
        const u128 pk = R.pow(p % R.M, k - 1);
        Vec c(D + 1);
        for (uint32_t n = 0; n <= D; ++n) {
            u128 v = coeff(p * n);
            if (n % p == 0) v = R.add(v, R.mul(pk, coeff(n / p)));
            c[n] = v;
        }
        return c;
    };
    bool stab_ok = true;
    Mat Tl(r, Vec(r, 0));
    for (uint32_t j = 0; j < r; ++j) {
        bool ok = true;
        const Vec x = solve(hecke(ell, [&](size_t i) { return Fat(j, i); }), ok);
        stab_ok = stab_ok && ok;
        for (uint32_t i = 0; i < r; ++i) Tl[i][j] = x[i];
    }
    for (uint32_t j = 1; j < r; ++j) ENGINE_CHECK_MSG(Tl[0][j] == 0, "T_ell does not preserve S_k");
    for (int sp = 0; sp < 2; ++sp) {
        oc_util::CancelPoint("hida_lambda.space");   // (NBCONFIRM S5) M_k, then S_k
        const uint32_t lo = static_cast<uint32_t>(sp);
        FormsSpace& FS = sp == 0 ? out.M : out.S;
        const size_t n = r - lo;
        Mat T(n, Vec(n));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) T[i][j] = Tl[i + lo][j + lo];
        // ordinary_basis
        Mat B;
        std::vector<size_t> rows;
        uint32_t N = 0;
        if (n > 0) {
            Mat P = T;
            N = 1;
            while (N < W * n) {
                P = MatMul(P, P, R);
                N *= 2;
            }
            const std::vector<size_t> cols = IndepColsModEll(P, n, ell);
            B.assign(n, Vec(cols.size()));
            for (size_t i = 0; i < n; ++i)
                for (size_t t = 0; t < cols.size(); ++t) B[i][t] = P[i][cols[t]];
            Mat Bt(cols.size(), Vec(n));
            for (size_t i = 0; i < n; ++i)
                for (size_t t = 0; t < cols.size(); ++t) Bt[t][i] = B[i][t];
            rows = IndepColsModEll(Bt, n, ell);
            if (rows.size() > cols.size()) rows.resize(cols.size());
        }
        const size_t s = B.empty() ? 0 : B[0].size();
        FS.s = static_cast<uint32_t>(s);
        FS.N = N;
        Mat Rinv;
        if (s) {
            Mat Rm(s, Vec(s));
            for (size_t i = 0; i < s; ++i)
                for (size_t t = 0; t < s; ++t) Rm[i][t] = B[rows[i]][t];
            Rinv = InvUnitMatrix(Rm, R);
        }
        // q-coefficients of the ordinary basis vectors g_t = sum_j B[j][t] f_{j+lo}
        std::vector<Vec> G(s, Vec(idx.size(), 0));
        for (size_t t = 0; t < s; ++t)
            for (size_t c = 0; c < idx.size(); ++c) {
                Acc acc;
                for (size_t j = 0; j < n; ++j) acc.add(B[j][t], F[j + lo][c]);
                G[t][c] = acc.get(R);
            }
        for (uint64_t p : pl) {
            oc_util::CancelPoint("hida_lambda.operator");   // (NBCONFIRM S5) per Hecke operator
            const std::string name = p == ell ? "U" : "T" + std::to_string(p);
            Mat A(s, Vec(s, 0));
            for (size_t t = 0; t < s; ++t) {
                bool ok = true;
                const Vec x = solve(hecke(p, [&](size_t i) { return G[t][static_cast<size_t>(pos[i])]; }), ok);
                stab_ok = stab_ok && ok;
                for (uint32_t i = 0; i < lo; ++i) ENGINE_CHECK_MSG(x[i] == 0, "T_p left S_k");
                Vec y(x.begin() + lo, x.end());
                Vec yr(s);
                for (size_t i = 0; i < s; ++i) yr[i] = y[rows[i]];
                const Vec coord = MatVec(Rinv, yr, R);
                for (size_t i = 0; i < n; ++i) {
                    Acc acc;
                    for (size_t u = 0; u < s; ++u) acc.add(B[i][u], coord[u]);
                    if (acc.get(R) != y[i]) stab_ok = false;
                }
                for (size_t u = 0; u < s; ++u) A[u][t] = coord[u];
            }
            FS.names.push_back(name);
            FS.mats[name] = A;
        }
        // U last in the names list (pl order already has ell last)
        bool comm = true;
        const auto it2 = FS.mats.find("T2");
        if (it2 != FS.mats.end())
            for (const auto& kv : FS.mats)
                comm = comm && MatMul(it2->second, kv.second, R) == MatMul(kv.second, it2->second, R);
        FS.comm_ok = comm;
    }
    out.stable_ok = stab_ok;
    for (uint64_t p : primes) out.eis["T" + std::to_string(p)] = R.add(1 % R.M, R.pow(p % R.M, k - 1));
    out.eis["U"] = 1 % R.M;
    out.wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
    return out;
}

}  // namespace lambda
}  // namespace hecke_ext
