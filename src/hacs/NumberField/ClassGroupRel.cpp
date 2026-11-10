// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ClassGroupRel.cpp -- see ClassGroupRel.h (NF-PRANK, slot 22).
#include "ClassGroupRel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "Lll.h"
#include "ModUtil.h"
#include "NfCheck.h"

namespace nf {

const char* prank_level_name(PRankLevel l) {
    switch (l) {
        case PRankLevel::CERTIFIED_EXACT: return "CERTIFIED_EXACT";
        case PRankLevel::UPPER_CERTIFIED_LOWER_ANALYTIC: return "UPPER_CERTIFIED_LOWER_ANALYTIC";
        case PRankLevel::GRH: return "GRH";
        case PRankLevel::REFUSED: return "REFUSED";
    }
    return "?";
}

// ===========================================================================
// 64-bit primality and factorisation
// ===========================================================================
namespace {

uint64_t powmod64(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1u) r = mulmod64(r, a, m);
        e >>= 1;
        if (e) a = mulmod64(a, a, m);
    }
    return r;
}

uint64_t gcd64(uint64_t a, uint64_t b) {
    while (b) {
        const uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

uint64_t rho(uint64_t n, uint64_t c) {
    // Pollard--Brent
    uint64_t y = 2, x = 2, g = 1, q = 1, ys = 2;
    const uint64_t m = 64;
    uint64_t r = 1;
    auto f = [&](uint64_t v) {
        const uint64_t s = mulmod64(v, v, n);
        return s >= n - c ? s - (n - c) : s + c;
    };
    do {
        x = y;
        for (uint64_t i = 0; i < r; ++i) y = f(y);
        uint64_t k = 0;
        do {
            ys = y;
            for (uint64_t i = 0; i < std::min(m, r - k); ++i) {
                y = f(y);
                q = mulmod64(q, x > y ? x - y : y - x, n);
            }
            g = gcd64(q, n);
            k += m;
        } while (k < r && g == 1);
        r *= 2;
        if (r > (static_cast<uint64_t>(1) << 26)) return 0;
    } while (g == 1);
    if (g == n) {
        do {
            ys = f(ys);
            g = gcd64(x > ys ? x - ys : ys - x, n);
        } while (g == 1);
    }
    return g == n ? 0 : g;
}

}  // namespace

bool is_prime_u64(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        if (n % p == 0) return n == p;
    }
    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1u) == 0) {
        d >>= 1;
        ++s;
    }
    for (uint64_t a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        uint64_t x = powmod64(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (unsigned i = 1; i < s; ++i) {
            x = mulmod64(x, x, n);
            if (x == n - 1) {
                comp = false;
                break;
            }
        }
        if (comp) return false;
    }
    return true;
}

bool factor_u64(uint64_t n, std::vector<std::pair<uint64_t, unsigned>>& out) {
    out.clear();
    if (n == 0) return false;
    std::map<uint64_t, unsigned> f;
    for (uint64_t p = 2; p < 1000 && p * p <= n; ++p) {
        while (n % p == 0) {
            ++f[p];
            n /= p;
        }
    }
    std::vector<uint64_t> stack;
    if (n > 1) stack.push_back(n);
    while (!stack.empty()) {
        const uint64_t m = stack.back();
        stack.pop_back();
        if (m == 1) continue;
        if (is_prime_u64(m)) {
            ++f[m];
            continue;
        }
        uint64_t d = 0;
        for (uint64_t c = 1; c < 40 && !d; ++c) d = rho(m, c);
        if (!d) return false;
        stack.push_back(d);
        stack.push_back(m / d);
    }
    for (const auto& kv : f) out.emplace_back(kv.first, kv.second);
    return true;
}

double pure_euler_residue(uint32_t N, uint32_t p, uint64_t X) {
    std::vector<uint8_t> comp(X + 1, 0);
    long double logres = 0;
    const bool p_sq = (N - 1) % (static_cast<uint64_t>(p) * p) == 0;
    for (uint64_t q = 2; q <= X; ++q) {
        if (comp[q]) continue;
        if (q * q <= X)
            for (uint64_t m = q * q; m <= X; m += q) comp[m] = 1;
        const long double iq = 1.0L / static_cast<long double>(q);
        long double fac;
        if (q == N) {
            fac = 1;  // one prime of norm N
        } else if (q == p) {
            fac = p_sq ? 1.0L / (1.0L - iq) : 1.0L;  // P_1 P_2^{p-1} (two primes of norm p) or P^p
        } else if (q % p == 1) {
            if (powmod32(N, (q - 1) / p, static_cast<uint32_t>(q)) == 1)
                fac = std::pow(1.0L - iq, static_cast<long double>(1) - p);  // p primes of degree 1
            else
                fac = (1.0L - iq) / (1.0L - std::pow(iq, static_cast<long double>(p)));  // inert
        } else {
            unsigned m = 1;
            for (uint64_t t = q % p; t != 1; t = t * (q % p) % p) ++m;  // order of q mod p
            // one root, (p - 1)/m primes of degree m
            fac = std::pow(1.0L - std::pow(iq, static_cast<long double>(m)), -static_cast<long double>((p - 1) / m));
        }
        logres += std::log(fac);
    }
    return static_cast<double>(std::exp(logres));
}

// ===========================================================================
// The relation engine
// ===========================================================================
namespace {

struct Relation {
    std::vector<std::pair<uint32_t, int64_t>> v;  // (factor-base index, valuation), sorted
    std::vector<int64_t> elt;                      // element g(alpha), g = elt (constant term first)
};

struct OverflowError : std::runtime_error {
    OverflowError() : std::runtime_error("integer overflow in the elimination") {}
};

inline int64_t chk_mul(int64_t a, int64_t b) {
    int64_t r;
    if (__builtin_mul_overflow(a, b, &r)) throw OverflowError();
    return r;
}
inline int64_t chk_sub(int64_t a, int64_t b) {
    int64_t r;
    if (__builtin_sub_overflow(a, b, &r)) throw OverflowError();
    return r;
}

struct AugRow {
    std::vector<std::pair<uint32_t, int64_t>> e;  // (column, value), sorted by column
    long double L[3] = {0, 0, 0};
    long double mag = 0;
    std::vector<int32_t> ch;  // characters mod p at the test primes
};

struct KernelVec {
    long double L[3];
    long double mag;
    std::vector<int32_t> ch;
};

void aug_axpy(AugRow& r, const AugRow& piv, int64_t f, uint32_t p) {
    // r -= f * piv
    std::vector<std::pair<uint32_t, int64_t>> out;
    out.reserve(r.e.size() + piv.e.size());
    std::size_t i = 0, j = 0;
    while (i < r.e.size() || j < piv.e.size()) {
        if (j == piv.e.size() || (i < r.e.size() && r.e[i].first < piv.e[j].first)) {
            out.push_back(r.e[i++]);
        } else if (i == r.e.size() || piv.e[j].first < r.e[i].first) {
            out.emplace_back(piv.e[j].first, chk_sub(0, chk_mul(f, piv.e[j].second)));
            ++j;
        } else {
            const int64_t v = chk_sub(r.e[i].second, chk_mul(f, piv.e[j].second));
            if (v) out.emplace_back(r.e[i].first, v);
            ++i;
            ++j;
        }
    }
    r.e.swap(out);
    const long double fl = static_cast<long double>(f);
    for (int k = 0; k < 3; ++k) r.L[k] -= fl * piv.L[k];
    r.mag += std::fabs(fl) * piv.mag;
    const int64_t fp = ((f % static_cast<int64_t>(p)) + p) % p;
    for (std::size_t k = 0; k < r.ch.size(); ++k) r.ch[k] = static_cast<int32_t>(((r.ch[k] - fp * piv.ch[k]) % static_cast<int64_t>(p) + p) % p);
}

int64_t coef(const AugRow& r, uint32_t c) {
    const auto it = std::lower_bound(r.e.begin(), r.e.end(), std::make_pair(c, std::numeric_limits<int64_t>::min()));
    return (it != r.e.end() && it->first == c) ? it->second : 0;
}

uint32_t rank_mod_p_dense(std::vector<std::vector<int64_t>> M, uint32_t p) {
    if (M.empty()) return 0;
    const std::size_t nc = M[0].size();
    for (auto& row : M)
        for (auto& x : row) x = ((x % static_cast<int64_t>(p)) + p) % p;
    uint32_t rk = 0;
    for (std::size_t c = 0; c < nc && rk < M.size(); ++c) {
        std::size_t piv = rk;
        while (piv < M.size() && M[piv][c] == 0) ++piv;
        if (piv == M.size()) continue;
        std::swap(M[rk], M[piv]);
        const int64_t inv = invmod32(static_cast<uint64_t>(M[rk][c]), p);
        for (auto& x : M[rk]) x = x * inv % p;
        for (std::size_t i = 0; i < M.size(); ++i) {
            if (i == rk || M[i][c] == 0) continue;
            const int64_t t = M[i][c];
            for (std::size_t j = 0; j < nc; ++j) M[i][j] = ((M[i][j] - t * M[rk][j]) % static_cast<int64_t>(p) + p) % p;
        }
        ++rk;
    }
    return rk;
}

// Smith normal form diagonal of an integer matrix (BigInt), all diagonal entries (>= 0).
std::vector<BigInt> smith_diagonal(std::vector<std::vector<BigInt>> A) {
    const std::size_t m = A.size(), n = m ? A[0].size() : 0;
    std::vector<BigInt> diag;
    std::size_t t = 0;
    while (t < std::min(m, n)) {
        // pivot: smallest nonzero |entry| in the trailing block
        std::size_t bi = m, bj = n;
        for (std::size_t i = t; i < m; ++i)
            for (std::size_t j = t; j < n; ++j)
                if (!A[i][j].is_zero() && (bi == m || A[i][j].abs() < A[bi][bj].abs())) {
                    bi = i;
                    bj = j;
                }
        if (bi == m) break;
        std::swap(A[t], A[bi]);
        for (auto& row : A) std::swap(row[t], row[bj]);
        bool done = false;
        while (!done) {
            done = true;
            for (std::size_t i = t + 1; i < m; ++i) {
                if (A[i][t].is_zero()) continue;
                const BigInt q = A[i][t] / A[t][t];
                for (std::size_t j = t; j < n; ++j) A[i][j] -= q * A[t][j];
                if (!A[i][t].is_zero()) {
                    std::swap(A[t], A[i]);
                    done = false;
                }
            }
            for (std::size_t j = t + 1; j < n; ++j) {
                if (A[t][j].is_zero()) continue;
                const BigInt q = A[t][j] / A[t][t];
                for (std::size_t i = t; i < m; ++i) A[i][j] -= q * A[i][t];
                if (!A[t][j].is_zero()) {
                    for (auto& row : A) std::swap(row[t], row[j]);
                    done = false;
                }
            }
            if (done) {
                // divisibility: A[t][t] must divide the trailing block
                for (std::size_t i = t + 1; i < m && done; ++i)
                    for (std::size_t j = t + 1; j < n; ++j)
                        if (!(A[i][j] % A[t][t]).is_zero()) {
                            for (std::size_t k = t; k < n; ++k) A[t][k] += A[i][k];
                            done = false;
                            break;
                        }
            }
        }
        diag.push_back(A[t][t].abs());
        ++t;
    }
    return diag;
}

// best rational approximation with denominator <= maxd (continued fractions + semiconvergent)
void limit_denominator(long double x, int64_t maxd, int64_t& num, int64_t& den) {
    int64_t p0 = 0, q0 = 1, p1 = 1, q1 = 0;
    long double r = x;
    for (int it = 0; it < 64; ++it) {
        const long double a = std::floor(r);
        const int64_t ai = static_cast<int64_t>(a);
        const int64_t q2 = q0 + ai * q1;
        if (q2 > maxd) {
            const int64_t k = (maxd - q0) / q1;
            const int64_t bp = p0 + k * p1, bq = q0 + k * q1;
            if (std::fabs(x - static_cast<long double>(p1) / q1) <= std::fabs(x - static_cast<long double>(bp) / bq)) {
                num = p1;
                den = q1;
            } else {
                num = bp;
                den = bq;
            }
            return;
        }
        const int64_t p2 = p0 + ai * p1;
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        const long double frac = r - a;
        if (std::fabs(frac) < 1e-15L) break;
        r = 1.0L / frac;
    }
    num = p1;
    den = q1;
}

using V2 = std::pair<long double, long double>;

void gauss_reduce(V2& b1, V2& b2) {
    for (int it = 0; it < 10000; ++it) {
        auto n2 = [](const V2& v) { return v.first * v.first + v.second * v.second; };
        if (n2(b1) > n2(b2)) std::swap(b1, b2);
        const long double mu = std::round((b1.first * b2.first + b1.second * b2.second) / n2(b1));
        if (mu == 0) return;
        b2 = V2(b2.first - mu * b1.first, b2.second - mu * b1.second);
    }
}

// Basis of the lattice generated by float vectors assumed to lie in a rank-2 lattice (the prototype's
// lattice_2d). Returns false if the vectors do not span rank 2.
bool lattice_2d(const std::vector<V2>& W0, V2& b1, V2& b2, std::size_t* skipped = nullptr) {
    std::vector<V2> W;
    for (const V2& w : W0)
        if (std::fabs(w.first) + std::fabs(w.second) > 1e-6L) W.push_back(w);
    if (W.empty()) return false;
    b1 = W[0];
    bool found = false;
    auto len = [](const V2& v) { return std::sqrt(v.first * v.first + v.second * v.second); };
    for (const V2& w : W)
        if (std::fabs(b1.first * w.second - b1.second * w.first) > 1e-6L * len(b1) * len(w)) {
            b2 = w;
            found = true;
            break;
        }
    if (!found) return false;
    gauss_reduce(b1, b2);
    for (const V2& w : W) {
        const long double det = b1.first * b2.second - b1.second * b2.first;
        const long double x = (w.first * b2.second - w.second * b2.first) / det;
        const long double y = (b1.first * w.second - b1.second * w.first) / det;
        // tolerance relative to the size of the coordinates (the logs carry ~1e-15 relative error)
        const long double tol = 1e-7L * (1 + std::fabs(x) + std::fabs(y));
        if (std::fabs(x - std::round(x)) < tol && std::fabs(y - std::round(y)) < tol) continue;
        int64_t xn, xd, yn, yd;
        limit_denominator(x, 1000, xn, xd);
        limit_denominator(y, 1000, yn, yd);
        // a refinement is accepted only if both coordinates ARE small-denominator rationals; otherwise the vector is
        // numerically unreliable and skipped (counted)
        if (std::fabs(x - static_cast<long double>(xn) / xd) > tol || std::fabs(y - static_cast<long double>(yn) / yd) > tol) {
            if (skipped) ++*skipped;
            continue;
        }
        const int64_t d = xd / std::gcd(xd, yd) * yd;
        std::vector<std::pair<int64_t, int64_t>> a = {{d, 0}, {0, d}, {xn * (d / xd), yn * (d / yd)}};
        auto key = [](const std::pair<int64_t, int64_t>& v) { return std::make_pair(v.first == 0, std::llabs(v.first)); };
        for (int guard = 0; guard < 10000; ++guard) {
            int nz = 0;
            for (const auto& v : a) nz += v.first != 0;
            if (nz <= 1) break;
            std::sort(a.begin(), a.end(), [&](const auto& u, const auto& v) { return key(u) < key(v); });
            for (std::size_t i = 1; i < a.size(); ++i) {
                if (!a[i].first) continue;
                // Python floor division
                int64_t q = a[i].first / a[0].first;
                if ((a[i].first % a[0].first != 0) && ((a[i].first < 0) != (a[0].first < 0))) --q;
                a[i].first -= q * a[0].first;
                a[i].second -= q * a[0].second;
            }
        }
        std::sort(a.begin(), a.end(), [&](const auto& u, const auto& v) { return key(u) < key(v); });
        int64_t g2 = 0;
        for (std::size_t i = 1; i < a.size(); ++i) g2 = std::gcd(g2, a[i].second);
        const long double c1x = static_cast<long double>(a[0].first) / d, c1y = static_cast<long double>(a[0].second) / d;
        const long double c2y = static_cast<long double>(g2) / d;
        V2 n1(c1x * b1.first + c1y * b2.first, c1x * b1.second + c1y * b2.second);
        V2 n2(c2y * b2.first, c2y * b2.second);
        b1 = n1;
        b2 = n2;
        gauss_reduce(b1, b2);
    }
    return true;
}

struct Engine {
    const PureFieldData& K;
    const FactorBase& FB;
    const PRankOptions& opt;
    PRankResult& R;
    uint32_t p, N;
    uint64_t B0;
    std::vector<int> col_of;                 // factor-base index -> dense column, -1 if not dense
    std::vector<uint32_t> chi;               // genus character values mod p
    std::vector<uint32_t> small_primes;      // primes <= B0
    std::vector<std::pair<uint32_t, uint32_t>> sieve_roots;  // distinct (q, r) of dense degree-1 ideals
    // test primes for the index certificate
    std::vector<uint32_t> tq, troot;
    std::vector<std::unordered_map<uint32_t, int32_t>> tdlog;
    std::complex<long double> emb[3];

    Engine(const PureFieldData& K_, const FactorBase& FB_, const PRankOptions& o, PRankResult& R_)
        : K(K_), FB(FB_), opt(o), R(R_), p(K_.p), N(K_.N), B0(0) {}

    bool is_dense(uint32_t idx) const { return col_of[idx] >= 0; }

    void setup(uint64_t dense_bound) {
        B0 = dense_bound;
        col_of.assign(FB.ideals.size(), -1);
        int c = 0;
        for (uint32_t i = 0; i < FB.ideals.size(); ++i) {
            if (static_cast<int>(i) == opt.drop_ideal) continue;
            if (FB.ideals[i].norm <= B0) col_of[i] = c++;
        }
        R.dense_ideals = static_cast<std::size_t>(c);
        // genus character
        const uint32_t g = primitive_root_prime(N);
        const uint32_t z = powmod32(g, (N - 1) / p, N);
        std::unordered_map<uint32_t, uint32_t> dl;
        uint64_t x = 1;
        for (uint32_t k = 0; k < p; ++k, x = x * z % N) dl[static_cast<uint32_t>(x)] = k;
        chi.assign(FB.ideals.size(), 0);
        for (uint32_t i = 0; i < FB.ideals.size(); ++i) {
            const PrimeIdeal& P = FB.ideals[i];
            if (P.q == N) continue;
            chi[i] = dl.at(powmod32(P.norm % N, (N - 1) / p, N));
        }
        // primes <= B0 and the sieve roots
        std::vector<uint8_t> comp(B0 + 1, 0);
        for (uint64_t q = 2; q <= B0; ++q) {
            if (comp[q]) continue;
            small_primes.push_back(static_cast<uint32_t>(q));
            for (uint64_t m = q * q; m <= B0; m += q) comp[m] = 1;
        }
        std::vector<std::pair<uint32_t, uint32_t>> roots;
        for (uint32_t i = 0; i < FB.ideals.size(); ++i) {
            const PrimeIdeal& P = FB.ideals[i];
            if (!is_dense(i) || P.deg != 1) continue;
            roots.emplace_back(P.q, P.root);
        }
        std::sort(roots.begin(), roots.end());
        roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
        sieve_roots = roots;
        // embeddings alpha_j = N^{1/p} e^{2 pi i j / p}, j = 0, 1, 2 (p = 5)
        const long double a0 = std::pow(static_cast<long double>(N), 1.0L / p);
        const long double pi = 3.14159265358979323846264338327950288L;
        for (int j = 0; j < 3; ++j) emb[j] = std::polar(a0, 2 * pi * j / p);
        // test primes q > M_K, q = 1 mod p, N a p-th power mod q
        if (opt.analytic) {
            uint64_t q = static_cast<uint64_t>(std::floor(K.minkowski)) + 1;
            while (tq.size() < opt.index_test_primes) {
                ++q;
                if (q % p != 1 || !is_prime_u32(static_cast<uint32_t>(q))) continue;
                const uint32_t qq = static_cast<uint32_t>(q);
                if (powmod32(N, (q - 1) / p, qq) != 1) continue;
                const auto facs = fq_factor_squarefree(fq_from_bigint(K.f, qq), qq, 1);
                NF_CHECK(!facs.empty());
                const uint32_t r = (qq - facs[0][0]) % qq;
                const uint32_t zz = powmod32(primitive_root_prime(qq), (q - 1) / p, qq);
                std::unordered_map<uint32_t, int32_t> d;
                uint64_t y = 1;
                for (uint32_t k = 0; k < p; ++k, y = y * zz % qq) d[static_cast<uint32_t>(y)] = static_cast<int32_t>(k);
                tq.push_back(qq);
                troot.push_back(r);
                tdlog.push_back(d);
            }
        }
    }

    void element_data(const std::vector<int64_t>& g, AugRow& row) const {
        for (int j = 0; j < 3; ++j) {
            std::complex<long double> s = 0, pw = 1;
            for (std::size_t i = 0; i < g.size(); ++i) {
                s += static_cast<long double>(g[i]) * pw;
                pw *= emb[j];
            }
            row.L[j] = std::log(std::abs(s));
        }
        row.mag = std::fabs(row.L[0]) + std::fabs(row.L[1]) + std::fabs(row.L[2]) + 1;
        row.ch.assign(tq.size(), 0);
        for (std::size_t t = 0; t < tq.size(); ++t) {
            const uint32_t q = tq[t];
            uint64_t v = 0, pw = 1;
            for (std::size_t i = 0; i < g.size(); ++i) {
                int64_t c = g[i] % static_cast<int64_t>(q);
                if (c < 0) c += q;
                v = (v + static_cast<uint64_t>(c) * pw) % q;
                pw = pw * troot[t] % q;
            }
            NF_CHECK_MSG(v != 0, "index certificate: test prime divides an element norm");
            row.ch[t] = tdlog[t].at(powmod32(v, (q - 1) / p, q));
        }
    }

    // Accept a relation if every ideal is dense.
    bool all_dense(const std::vector<std::pair<uint32_t, int64_t>>& v) const {
        for (const auto& iv : v)
            if (!is_dense(iv.first)) return false;
        return true;
    }

    void collect(std::vector<Relation>& rels, uint32_t sieveB) {
        rels.clear();
        R.candidates = R.relations_2term = R.relations_3term = R.relations_q = 0;
        const int64_t A = opt.sieve_A;
        const std::size_t W = static_cast<std::size_t>(2 * A + 1);
        std::vector<float> acc(W);
        std::vector<float> lq(sieve_roots.size());
        for (std::size_t i = 0; i < sieve_roots.size(); ++i) lq[i] = static_cast<float>(std::log(static_cast<double>(sieve_roots[i].first)));
        const double slack = 1.3 * std::log(static_cast<double>(B0));
        std::vector<std::pair<uint32_t, int64_t>> vals;
        std::vector<std::pair<uint32_t, unsigned>> fact;
        for (uint32_t b = 1; b <= sieveB; ++b) {
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (std::size_t i = 0; i < sieve_roots.size(); ++i) {
                const uint32_t q = sieve_roots[i].first, r = sieve_roots[i].second;
                // a = -A + idx = -b r (mod q)
                const int64_t target = ((-static_cast<int64_t>(b) * r) % static_cast<int64_t>(q) + q) % q;
                int64_t start = ((target + A) % static_cast<int64_t>(q) + q) % q;
                for (std::size_t idx = static_cast<std::size_t>(start); idx < W; idx += q) acc[idx] += lq[i];
            }
            const double nb = static_cast<double>(N) * std::pow(static_cast<double>(b), static_cast<double>(p));
            for (std::size_t idx = 0; idx < W; ++idx) {
                const int64_t a = static_cast<int64_t>(idx) - A;
                const double nn = std::fabs(std::pow(static_cast<double>(a), static_cast<double>(p)) + nb);
                if (acc[idx] < std::log(nn + 1.0) - slack) continue;
                if (std::gcd(std::llabs(a), static_cast<int64_t>(b)) != 1) continue;
                ++R.candidates;
                BigInt n = BigInt::pow(BigInt(a), p) + BigInt(static_cast<int64_t>(N)) * BigInt::pow(BigInt(static_cast<int64_t>(b)), p);
                n = n.abs();
                if (n.is_zero()) continue;
                fact.clear();
                for (const auto& qr : sieve_roots) {
                    const int64_t q = qr.first;
                    if (((a + static_cast<int64_t>(b) * qr.second) % q + q) % q != 0) continue;
                    if (!fact.empty() && fact.back().first == qr.first) continue;  // p^2 | N - 1: two ideals, same root
                    unsigned k = 0;
                    while (n.mod_u32(qr.first) == 0) {
                        n = n.div_exact_u32(qr.first);
                        ++k;
                    }
                    if (k) fact.emplace_back(qr.first, k);
                }
                if (n != BigInt(1)) continue;
                if (!pure_valuations(K, FB, {a, static_cast<int64_t>(b)}, fact, vals)) continue;
                if (!all_dense(vals)) continue;
                rels.push_back(Relation{vals, {a, static_cast<int64_t>(b)}});
                ++R.relations_2term;
            }
        }
        // 3-term elements a + b alpha + c alpha^2
        for (int a = -opt.box3_ab; a <= opt.box3_ab; ++a)
            for (int b = -opt.box3_ab; b <= opt.box3_ab; ++b)
                for (int c = 1; c <= opt.box3_c; ++c) {
                    if (std::gcd(std::gcd(std::abs(a), std::abs(b)), c) != 1) continue;
                    const std::vector<int64_t> g = {a, b, c};
                    const BigInt n = pure_norm(K, g);
                    if (n.is_zero()) continue;
                    fact.clear();
                    if (!factor_over_small(n, fact)) continue;
                    if (!pure_valuations(K, FB, g, fact, vals)) continue;
                    if (!all_dense(vals)) continue;
                    rels.push_back(Relation{vals, g});
                    ++R.relations_3term;
                }
        // (q) = prod P^e when every prime over q is dense
        for (const auto& kv : FB.by_q) {
            const uint32_t q = kv.first;
            if (!FB.all_over_q.at(q)) continue;
            bool ok = true;
            for (uint32_t i : kv.second) ok = ok && is_dense(i);
            if (!ok) continue;
            Relation rel;
            for (uint32_t i : kv.second) rel.v.emplace_back(i, static_cast<int64_t>(FB.ideals[i].e));
            std::sort(rel.v.begin(), rel.v.end());
            rel.elt = {static_cast<int64_t>(q)};
            rels.push_back(rel);
            ++R.relations_q;
        }
        R.relations = rels.size();
    }

    // n > 0 factored over the primes <= B0 (n = 1 at the end, or the cofactor a prime <= B0).
    bool factor_over_small(BigInt n, std::vector<std::pair<uint32_t, unsigned>>& fact) const {
        if (n.fits_int64()) {
            uint64_t m = static_cast<uint64_t>(n.to_int64());
            for (uint32_t q : small_primes) {
                if (static_cast<uint64_t>(q) * q > m) break;
                if (m % q) continue;
                unsigned k = 0;
                while (m % q == 0) {
                    m /= q;
                    ++k;
                }
                fact.emplace_back(q, k);
            }
            if (m > 1) {
                if (m > B0) return false;
                fact.emplace_back(static_cast<uint32_t>(m), 1);
                std::sort(fact.begin(), fact.end());
            }
            return true;
        }
        for (uint32_t q : small_primes) {
            if (n.mod_u32(q)) continue;
            unsigned k = 0;
            while (n.mod_u32(q) == 0) {
                n = n.div_exact_u32(q);
                ++k;
            }
            fact.emplace_back(q, k);
            if (n == BigInt(1)) break;
        }
        return n == BigInt(1);
    }

    // full factorisation of the cofactor m (uint64) appended to fact; false if it fails
    static bool append_factor(uint64_t m, std::vector<std::pair<uint32_t, unsigned>>& fact) {
        std::vector<std::pair<uint64_t, unsigned>> f;
        if (!factor_u64(m, f)) return false;
        for (const auto& pk : f) {
            if (pk.first >= (static_cast<uint64_t>(1) << 32)) return false;
            fact.emplace_back(static_cast<uint32_t>(pk.first), pk.second);
        }
        std::sort(fact.begin(), fact.end());
        return true;
    }

    // Descent row for ideal idx; returns true and fills row on success.
    bool descend(uint32_t idx, Relation& row, bool& by_q, bool allow_q = true) {
        const PrimeIdeal& P = FB.ideals[idx];
        by_q = false;
        // (q) itself when idx is the last prime over q and every prime over q is in S
        const auto& over = FB.by_q.at(P.q);
        if (allow_q && FB.all_over_q.at(P.q) && over.back() == idx && P.e == 1) {
            row.v.clear();
            for (uint32_t i : over) row.v.emplace_back(i, static_cast<int64_t>(FB.ideals[i].e));
            std::sort(row.v.begin(), row.v.end());
            row.elt = {static_cast<int64_t>(P.q)};
            by_q = true;
            return true;
        }
        const int64_t q = P.q;
        std::vector<std::pair<uint32_t, unsigned>> fact;
        std::vector<std::pair<uint32_t, int64_t>> vals;
        auto accept = [&](const std::vector<int64_t>& g, const BigInt& n0) {
            const BigInt n = n0.abs();
            if (n.is_zero()) return false;
            BigInt m = n;
            for (unsigned i = 0; i < P.deg; ++i) {
                if (m.mod_u32(P.q) != 0) return false;
                m = m.div_exact_u32(P.q);
            }
            if (m.mod_u32(P.q) == 0) return false;  // v_q(N beta) = deg exactly
            if (!m.fits_int64()) return false;
            fact.clear();
            if (!append_factor(static_cast<uint64_t>(m.to_int64()), fact)) return false;
            fact.emplace_back(P.q, P.deg);
            std::sort(fact.begin(), fact.end());
            for (const auto& lk : fact)
                if (lk.first != P.q && static_cast<uint64_t>(lk.first) >= P.norm) return false;
            if (!pure_valuations(K, FB, g, fact, vals)) return false;
            bool has = false;
            for (const auto& iv : vals) {
                if (iv.first == idx) {
                    if (iv.second != 1) return false;
                    has = true;
                } else if (iv.first > idx) {
                    return false;
                }
            }
            if (!has) return false;
            row.v = vals;
            row.elt = g;
            return true;
        };
        if (P.deg == 1) {
            // lattice {(a, b) : a + b r = 0 mod q}, basis (q, 0), (-r, 1), Gauss-reduced exactly
            int64_t u1 = q, u2 = 0, w1 = -static_cast<int64_t>(P.root), w2 = 1;
            for (int it = 0; it < 1000; ++it) {
                const long double n1 = static_cast<long double>(u1) * u1 + static_cast<long double>(u2) * u2;
                const long double n2 = static_cast<long double>(w1) * w1 + static_cast<long double>(w2) * w2;
                if (n1 > n2) {
                    std::swap(u1, w1);
                    std::swap(u2, w2);
                    continue;
                }
                const long double dot = static_cast<long double>(u1) * w1 + static_cast<long double>(u2) * w2;
                const int64_t mu = static_cast<int64_t>(std::llround(dot / n1));
                if (mu == 0) break;
                w1 -= mu * u1;
                w2 -= mu * u2;
            }
            for (int rad = 1; rad <= 40; ++rad)
                for (int i = -rad; i <= rad; ++i)
                    for (int j = 0; j <= rad; ++j) {
                        if (std::max(std::abs(i), j) != rad) continue;
                        if (j == 0 && i <= 0) continue;
                        int64_t a = i * u1 + j * w1, b = i * u2 + j * w2;
                        if (b < 0) {
                            a = -a;
                            b = -b;
                        }
                        if (b == 0 || std::gcd(std::llabs(a), b) != 1) continue;
                        const BigInt n = BigInt::pow(BigInt(a), p) + BigInt(static_cast<int64_t>(N)) * BigInt::pow(BigInt(b), p);
                        if (accept({a, b}, n)) return true;
                    }
            return false;
        }
        if (P.deg == 2) {
            // lattice {(a, b, c) : a = c h0, b = c h1 mod q}, LLL-reduced exactly
            IntMat rows = {{BigInt(q), BigInt(0), BigInt(0)},
                           {BigInt(0), BigInt(q), BigInt(0)},
                           {BigInt(static_cast<int64_t>(P.h[0])), BigInt(static_cast<int64_t>(P.h[1])), BigInt(1)}};
            const LllBasisResult L = lll_basis(rows);
            int64_t v[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) v[i][j] = L.basis[i][j].to_int64();
            for (int rad = 1; rad <= 5; ++rad)
                for (int i = -rad; i <= rad; ++i)
                    for (int j = -rad; j <= rad; ++j)
                        for (int k = 0; k <= rad; ++k) {
                            if (std::max(std::max(std::abs(i), std::abs(j)), k) != rad) continue;
                            std::vector<int64_t> g(3);
                            for (int t = 0; t < 3; ++t) g[t] = i * v[0][t] + j * v[1][t] + k * v[2][t];
                            if (std::gcd(std::gcd(std::llabs(g[0]), std::llabs(g[1])), std::llabs(g[2])) != 1) continue;
                            if (accept(g, pure_norm(K, g))) return true;
                        }
            return false;
        }
        return false;
    }
};

}  // namespace

// ===========================================================================
// nf_prank
// ===========================================================================
PRankResult nf_prank(uint32_t N, uint32_t p, const PRankOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    PRankResult R;
    R.N = N;
    R.p = p;
    auto finish = [&](PRankLevel lvl, const std::string& why) {
        R.level = lvl;
        if (!why.empty() && R.refusal.empty()) R.refusal = why;
        R.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return R;
    };
    if (!is_prime_u32(p) || p < 5 || !is_prime_u32(N) || N % p != 1) return finish(PRankLevel::REFUSED, "cell outside the domain (N prime = 1 mod p, p >= 5)");
    if (!p_is_regular(p)) return finish(PRankLevel::REFUSED, "p divides h_F (p irregular)");
    R.chev = chevalley_data(N, p);
    if (p != 5) return finish(PRankLevel::REFUSED, "relation arm implemented for p = 5 only (slot 22)");
    const PureFieldData K = pure_field(N, p, opt.force_zalpha);
    R.dK = K.dK;
    R.index = K.index;
    R.r1 = K.r1;
    R.r2 = K.r2;
    R.order_maximal = K.order_maximal;
    R.dedekind_p = K.dedekind_p;
    R.dedekind_N = K.dedekind_N;
    R.round2_iterations = K.round2_iterations;
    R.minkowski = K.minkowski;
    if (!K.refusal.empty()) return finish(PRankLevel::REFUSED, K.refusal);
    if (K.minkowski > static_cast<long double>(opt.max_minkowski)) return finish(PRankLevel::REFUSED, "Minkowski bound above max_minkowski (descent / GRH arm needed)");
    const uint64_t M = static_cast<uint64_t>(std::floor(K.minkowski)) + 1;  // +1: hardening, the only floating decision on the certified path (referee R2)
    const FactorBase FB = pure_factor_base(K, M);
    R.factor_base = FB.ideals.size();
    for (const PrimeIdeal& P : FB.ideals)
        if (P.deg >= 2) R.deg2plus.emplace_back(P.q, P.deg);
    std::sort(R.deg2plus.begin(), R.deg2plus.end());
    R.minkowski_complete = opt.drop_ideal < 0;
    const uint64_t B0 = opt.dense_bound ? std::min<uint64_t>(opt.dense_bound, M) : M;
    R.dense_bound = B0;
    R.euler_X = opt.euler_X;
    try {
        Engine E(K, FB, opt, R);
        E.setup(B0);
        for (uint32_t i = 0; i < FB.ideals.size(); ++i)
            if (E.is_dense(i) && E.chi[i] % p) R.genus_nontrivial = true;
        // descent of S \ S_0
        std::size_t genus_bad = 0;
        for (uint32_t i = 0; i < FB.ideals.size(); ++i) {
            if (E.is_dense(i) || static_cast<int>(i) == opt.drop_ideal) continue;
            Relation row;
            bool byq = false;
            if (!E.descend(i, row, byq)) {
                ++R.descent_failed;
                if (R.refusal.empty())
                    R.refusal = "descent failed at the ideal of norm " + std::to_string(FB.ideals[i].norm) + " over q = " + std::to_string(FB.ideals[i].q);
                continue;
            }
            ++R.descended;
            if (byq) ++R.descended_by_q;
            int64_t s = 0;
            for (const auto& iv : row.v) s += iv.second * static_cast<int64_t>(E.chi[iv.first]);
            if (s % p) ++genus_bad;
        }
        if (R.descent_failed) R.minkowski_complete = false;
        // relations and the linear algebra, widening the sieve if Z^{S_0}/R_0 is not finite
        std::vector<Relation> rels;
        uint32_t sieveB = opt.sieve_B;
        std::vector<KernelVec> kernel;
        for (unsigned widen = 0;; ++widen) {
            E.collect(rels, sieveB);
            R.sieve_B_used = sieveB;
            if (opt.corrupt_relation && !rels.empty()) {
                // G4 (b): add 1 to the valuation of an ideal with chi != 0 in the first relation
                for (uint32_t i = 0; i < FB.ideals.size(); ++i)
                    if (E.is_dense(i) && E.chi[i] % p) {
                        auto& v = rels[0].v;
                        auto it = std::lower_bound(v.begin(), v.end(), std::make_pair(i, std::numeric_limits<int64_t>::min()));
                        if (it != v.end() && it->first == i)
                            it->second += 1;
                        else
                            v.insert(it, std::make_pair(i, static_cast<int64_t>(1)));
                        break;
                    }
            }
            R.genus_bad = genus_bad;
            // weakly covered dense ideals (in at most one relation, e.g. a degree-2 ideal met only by (q)): one lattice
            // element of the ideal (the descent search, (q) excluded) is appended as an ordinary relation -- a genuine
            // principal ideal, so R_0 stays a set of relations; all its ideals have norm < N(Q) <= B_0, hence dense
            {
                std::vector<std::size_t> occ(R.dense_ideals, 0);
                for (const Relation& rel : rels)
                    for (const auto& iv : rel.v) ++occ[E.col_of[iv.first]];
                R.extra_rows = 0;
                for (uint32_t i = 0; i < FB.ideals.size(); ++i) {
                    if (!E.is_dense(i) || occ[E.col_of[i]] > 1) continue;
                    Relation row;
                    bool byq = false;
                    if (!E.descend(i, row, byq, false) || !E.all_dense(row.v)) continue;
                    rels.push_back(row);
                    ++R.extra_rows;
                }
                R.relations = rels.size();
            }
            for (const Relation& rel : rels) {
                int64_t s = 0;
                for (const auto& iv : rel.v) s += iv.second * static_cast<int64_t>(E.chi[iv.first]);
                if (s % p) ++R.genus_bad;
            }
            // augmented rows over the dense columns
            const std::size_t ncol = R.dense_ideals;
            std::vector<AugRow> rows(rels.size());
            for (std::size_t i = 0; i < rels.size(); ++i) {
                for (const auto& iv : rels[i].v) rows[i].e.emplace_back(static_cast<uint32_t>(E.col_of[iv.first]), iv.second);
                std::sort(rows[i].e.begin(), rows[i].e.end());
                if (opt.analytic) E.element_data(rels[i].elt, rows[i]);
            }
            kernel.clear();
            // sparse elimination on +-1 pivots
            std::vector<std::vector<uint32_t>> colrows(ncol);
            for (uint32_t i = 0; i < rows.size(); ++i)
                for (const auto& cv : rows[i].e) colrows[cv.first].push_back(i);
            std::vector<uint8_t> alive(rows.size(), 1), col_alive(ncol, 1);
            uint32_t pivots = 0;
            for (bool progress = true; progress;) {
                progress = false;
                std::vector<uint32_t> order;
                for (uint32_t c = 0; c < ncol; ++c)
                    if (col_alive[c]) order.push_back(c);
                std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return colrows[a].size() < colrows[b].size(); });
                for (uint32_t c : order) {
                    std::vector<uint32_t> live;
                    for (uint32_t r : colrows[c])
                        if (alive[r] && coef(rows[r], c) != 0) live.push_back(r);
                    std::sort(live.begin(), live.end());
                    live.erase(std::unique(live.begin(), live.end()), live.end());
                    colrows[c] = live;
                    int best = -1;
                    for (uint32_t r : live) {
                        const int64_t v = coef(rows[r], c);
                        if ((v == 1 || v == -1) && (best < 0 || rows[r].e.size() < rows[best].e.size())) best = static_cast<int>(r);
                    }
                    if (best < 0) continue;
                    const AugRow piv = rows[best];
                    const int64_t s = coef(piv, c);
                    for (uint32_t r : live) {
                        if (static_cast<int>(r) == best) continue;
                        const int64_t f = coef(rows[r], c) * s;
                        const std::size_t before = rows[r].e.size();
                        std::vector<uint32_t> oldcols;
                        oldcols.reserve(before);
                        for (const auto& cv : rows[r].e) oldcols.push_back(cv.first);
                        aug_axpy(rows[r], piv, f, p);
                        for (const auto& cv : rows[r].e)
                            if (!std::binary_search(oldcols.begin(), oldcols.end(), cv.first)) colrows[cv.first].push_back(r);
                        if (rows[r].e.empty()) {
                            alive[r] = 0;
                            kernel.push_back(KernelVec{{rows[r].L[0], rows[r].L[1], rows[r].L[2]}, rows[r].mag, rows[r].ch});
                        }
                    }
                    alive[best] = 0;
                    col_alive[c] = 0;
                    colrows[c].clear();
                    ++pivots;
                    progress = true;
                }
            }
            // rows that were empty from the start (none expected) are kernel vectors too
            for (uint32_t r = 0; r < rows.size(); ++r)
                if (alive[r] && rows[r].e.empty()) {
                    alive[r] = 0;
                    kernel.push_back(KernelVec{{rows[r].L[0], rows[r].L[1], rows[r].L[2]}, rows[r].mag, rows[r].ch});
                }
            // dense remainder
            std::vector<uint32_t> cols;
            std::vector<int> dcol(ncol, -1);
            for (uint32_t c = 0; c < ncol; ++c)
                if (col_alive[c]) {
                    dcol[c] = static_cast<int>(cols.size());
                    cols.push_back(c);
                }
            const std::size_t k = cols.size();
            struct DRow {
                std::vector<int64_t> v;
                long double L[3];
                long double mag;
                std::vector<int32_t> ch;
            };
            std::vector<DRow> pool;
            for (uint32_t r = 0; r < rows.size(); ++r) {
                if (!alive[r]) continue;
                DRow d;
                d.v.assign(k, 0);
                for (const auto& cv : rows[r].e) {
                    NF_CHECK(dcol[cv.first] >= 0);
                    d.v[dcol[cv.first]] = cv.second;
                }
                for (int t = 0; t < 3; ++t) d.L[t] = rows[r].L[t];
                d.mag = rows[r].mag;
                d.ch = rows[r].ch;
                pool.push_back(std::move(d));
            }
            auto daxpy = [&](DRow& a, const DRow& b, int64_t f) {
                for (std::size_t j = 0; j < k; ++j)
                    if (b.v[j]) a.v[j] = chk_sub(a.v[j], chk_mul(f, b.v[j]));
                for (int t = 0; t < 3; ++t) a.L[t] -= static_cast<long double>(f) * b.L[t];
                a.mag += std::fabs(static_cast<long double>(f)) * b.mag;
                const int64_t fp = ((f % static_cast<int64_t>(p)) + p) % p;
                for (std::size_t t = 0; t < a.ch.size(); ++t) a.ch[t] = static_cast<int32_t>(((a.ch[t] - fp * b.ch[t]) % static_cast<int64_t>(p) + p) % p);
            };
            std::vector<std::vector<int64_t>> echelon;
            std::vector<uint8_t> col_has_pivot(k, 0);
            for (std::size_t j = 0; j < k; ++j) {
                for (;;) {
                    int best = -1;
                    std::size_t cnt = 0;
                    for (std::size_t i = 0; i < pool.size(); ++i) {
                        if (!pool[i].v[j]) continue;
                        ++cnt;
                        if (best < 0 || std::llabs(pool[i].v[j]) < std::llabs(pool[best].v[j])) best = static_cast<int>(i);
                    }
                    if (cnt == 0) break;
                    if (cnt == 1) {
                        echelon.push_back(pool[best].v);
                        col_has_pivot[j] = 1;
                        pool.erase(pool.begin() + best);
                        break;
                    }
                    const DRow pv = pool[best];
                    for (std::size_t i = 0; i < pool.size(); ++i) {
                        if (static_cast<int>(i) == best || !pool[i].v[j]) continue;
                        daxpy(pool[i], pv, pool[i].v[j] / pv.v[j]);
                    }
                    // move zero rows to the kernel
                    std::vector<DRow> keep;
                    for (std::size_t i = 0; i < pool.size(); ++i) {
                        bool zero = true;
                        for (int64_t x : pool[i].v) zero = zero && x == 0;
                        if (zero)
                            kernel.push_back(KernelVec{{pool[i].L[0], pool[i].L[1], pool[i].L[2]}, pool[i].mag, pool[i].ch});
                        else
                            keep.push_back(std::move(pool[i]));
                    }
                    pool.swap(keep);
                }
            }
            // whatever is left in the pool has only zeros left of the processed columns: kernel
            for (const DRow& d : pool) kernel.push_back(KernelVec{{d.L[0], d.L[1], d.L[2]}, d.mag, d.ch});
            // group bound
            std::vector<std::vector<BigInt>> Z;
            for (const auto& row : echelon) {
                std::vector<BigInt> zr;
                for (int64_t x : row) zr.push_back(BigInt(x));
                Z.push_back(zr);
            }
            std::vector<BigInt> diag = smith_diagonal(Z);
            R.group_bound.clear();
            for (const BigInt& d : diag)
                if (d != BigInt(1)) R.group_bound.push_back(d);
            std::size_t free_rank = k - diag.size();
            R.free_columns.clear();
            for (std::size_t j = 0; j < k; ++j) {
                if (col_has_pivot[j]) continue;
                for (uint32_t i = 0; i < FB.ideals.size(); ++i)
                    if (E.col_of[i] == static_cast<int>(cols[j])) {
                        std::size_t occ = 0;
                        for (const Relation& rel : rels)
                            for (const auto& iv : rel.v) occ += iv.first == i;
                        R.free_columns += "(" + std::to_string(FB.ideals[i].q) + "," + std::to_string(FB.ideals[i].deg) + ",kind" +
                                          std::to_string(FB.ideals[i].kind) + ",in " + std::to_string(occ) + " relations)";
                    }
            }
            for (std::size_t i = 0; i < free_rank; ++i) R.group_bound.push_back(BigInt(0));
            // F_p-rank of R_0: unimodular pivots + rank of the echelon block mod p
            const uint32_t rk_rem = rank_mod_p_dense(echelon, p);
            R.rank_Fp_rel = pivots + rk_rem;
            R.rank_upper = static_cast<uint32_t>(ncol) - R.rank_Fp_rel;
            uint32_t from_snf = 0;
            for (const BigInt& d : R.group_bound)
                if (d.is_zero() || d.mod_u32(p) == 0) ++from_snf;
            NF_CHECK_MSG(from_snf == R.rank_upper, "p-rank from the Smith form != |S_0| - rank_Fp(R_0)");
            if (free_rank == 0 || widen >= opt.max_widen) break;
            sieveB *= 2;
        }
        for (const BigInt& d : R.group_bound)
            if (d.is_zero()) R.refusal = "relation lattice not of full rank (Z^{S_0}/R_0 has a free part)";
        R.genus_ok = R.genus_bad == 0 && R.genus_nontrivial;
        R.rank_lower = R.genus_ok ? 1 : 0;
        // analytic side
        if (opt.analytic) {
            std::vector<V2> Wv;
            std::vector<std::vector<int64_t>> chars;
            // most accurate (smallest accumulated magnitude) first
            std::stable_sort(kernel.begin(), kernel.end(), [](const KernelVec& a, const KernelVec& b) { return a.mag < b.mag; });
            for (const KernelVec& kv : kernel) {
                const long double s = kv.L[0] + 2 * kv.L[1] + 2 * kv.L[2];
                if (std::fabs(s) > 1e-12L * kv.mag + 1e-7L || kv.mag > 1e12L) {
                    ++R.rejected_numerics;
                    continue;
                }
                Wv.emplace_back(kv.L[0], 2 * kv.L[1]);
                chars.emplace_back(kv.ch.begin(), kv.ch.end());
            }
            R.kernel_vectors = kernel.size();
            V2 b1, b2;
            std::size_t skipped = 0;
            const bool lat = lattice_2d(Wv, b1, b2, &skipped);
            R.rejected_numerics += skipped;
            if (lat) {
                R.regulator_found = static_cast<double>(std::fabs(b1.first * b2.second - b1.second * b2.first));
                // F_p-rank of the characters on E' (rows = units, columns = test primes)
                R.index_rank = rank_mod_p_dense(chars, p);
                R.p_coprime_unit_index = R.index_rank == R.r1 + R.r2 - 1;
            }
            R.euler_res = pure_euler_residue(N, p, opt.euler_X);
            const long double pi = 3.14159265358979323846264338327950288L;
            R.hR_estimate = static_cast<double>(2.0L * std::sqrt(std::fabs(bigint_to_ld(K.dK))) /
                                                (std::pow(2.0L, static_cast<long double>(K.r1)) * std::pow(2 * pi, static_cast<long double>(K.r2))) *
                                                R.euler_res);
            if (R.regulator_found > 0) R.h_times_index = R.hR_estimate / R.regulator_found;
        }
    } catch (const OverflowError& e) {
        return finish(PRankLevel::REFUSED, e.what());
    }
    if (!R.minkowski_complete)
        return finish(PRankLevel::REFUSED, opt.drop_ideal >= 0 ? "factor base not Minkowski-complete (an ideal of norm <= M_K was removed)" : "");
    if (!R.refusal.empty()) return finish(PRankLevel::REFUSED, "");
    if (!R.genus_ok) return finish(PRankLevel::REFUSED, R.genus_nontrivial ? "genus character does not kill every relation" : "genus character trivial on S");
    if (R.rank_upper == R.rank_lower) return finish(PRankLevel::CERTIFIED_EXACT, "");
    return finish(PRankLevel::UPPER_CERTIFIED_LOWER_ANALYTIC, "");
}

}  // namespace nf
