// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// PureField.cpp -- see PureField.h (NF-PRANK, slot 22).
#include "PureField.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

#include "Lll.h"
#include "ModUtil.h"
#include "NfCheck.h"

namespace nf {

// ===========================================================================
// F_q polynomials
// ===========================================================================
FqPoly fq_trim(FqPoly a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
    return a;
}

FqPoly fq_from_int(const std::vector<int64_t>& c, uint32_t q) {
    FqPoly r(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
        int64_t v = c[i] % static_cast<int64_t>(q);
        if (v < 0) v += q;
        r[i] = static_cast<uint32_t>(v);
    }
    return fq_trim(r);
}

FqPoly fq_from_bigint(const IntPoly& c, uint32_t q) {
    FqPoly r(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) r[i] = c[i].mod_u32(q);
    return fq_trim(r);
}

FqPoly fq_mul(const FqPoly& a, const FqPoly& b, uint32_t q) {
    if (a.empty() || b.empty()) return {};
    std::vector<uint64_t> r(a.size() + b.size() - 1, 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!a[i]) continue;
        for (std::size_t j = 0; j < b.size(); ++j) r[i + j] = (r[i + j] + static_cast<uint64_t>(a[i]) * b[j] % q) % q;
    }
    FqPoly o(r.size());
    for (std::size_t i = 0; i < r.size(); ++i) o[i] = static_cast<uint32_t>(r[i]);
    return fq_trim(o);
}

FqPoly fq_sub(const FqPoly& a, const FqPoly& b, uint32_t q) {
    FqPoly r(std::max(a.size(), b.size()), 0);
    for (std::size_t i = 0; i < r.size(); ++i) {
        const uint64_t x = i < a.size() ? a[i] : 0, y = i < b.size() ? b[i] : 0;
        r[i] = static_cast<uint32_t>((x + q - y) % q);
    }
    return fq_trim(r);
}

void fq_divrem(const FqPoly& a0, const FqPoly& b0, uint32_t q, FqPoly& quo, FqPoly& rem) {
    FqPoly b = fq_trim(b0);
    NF_CHECK_MSG(!b.empty(), "fq_divrem: division by zero");
    std::vector<uint64_t> a(a0.begin(), a0.end());
    while (!a.empty() && a.back() == 0) a.pop_back();
    quo.clear();
    if (a.size() < b.size()) {
        rem = FqPoly(a.begin(), a.end());
        return;
    }
    const uint64_t inv = invmod32(b.back(), q);
    quo.assign(a.size() - b.size() + 1, 0);
    for (std::size_t s = a.size() - b.size() + 1; s-- > 0;) {
        const std::size_t k = s + b.size() - 1;
        const uint64_t c = a[k] % q * inv % q;
        quo[s] = static_cast<uint32_t>(c);
        if (c)
            for (std::size_t i = 0; i < b.size(); ++i) a[s + i] = (a[s + i] + q - c * b[i] % q) % q;
    }
    a.resize(b.size() - 1);
    rem = fq_trim(FqPoly(a.begin(), a.end()));
    quo = fq_trim(quo);
}

FqPoly fq_mod(const FqPoly& a, const FqPoly& b, uint32_t q) {
    FqPoly qq, r;
    fq_divrem(a, b, q, qq, r);
    return r;
}

FqPoly fq_monic(const FqPoly& a0, uint32_t q) {
    FqPoly a = fq_trim(a0);
    if (a.empty()) return a;
    const uint64_t inv = invmod32(a.back(), q);
    for (auto& x : a) x = static_cast<uint32_t>(x * inv % q);
    return a;
}

FqPoly fq_gcd(FqPoly a, FqPoly b, uint32_t q) {
    a = fq_trim(a);
    b = fq_trim(b);
    while (!b.empty()) {
        FqPoly r = fq_mod(a, b, q);
        a = std::move(b);
        b = std::move(r);
    }
    return fq_monic(a, q);
}

FqPoly fq_powmod(const FqPoly& base, uint64_t e, const FqPoly& m, uint32_t q) {
    FqPoly r = fq_mod(FqPoly{1}, m, q), b = fq_mod(base, m, q);
    while (e) {
        if (e & 1u) r = fq_mod(fq_mul(r, b, q), m, q);
        e >>= 1;
        if (e) b = fq_mod(fq_mul(b, b, q), m, q);
    }
    return r;
}

uint32_t fq_eval(const FqPoly& a, uint32_t x, uint32_t q) {
    uint64_t r = 0;
    for (std::size_t i = a.size(); i-- > 0;) r = (r * x + a[i]) % q;
    return static_cast<uint32_t>(r);
}

namespace {

unsigned deg(const FqPoly& a) { return a.empty() ? 0 : static_cast<unsigned>(a.size() - 1); }

FqPoly fq_x() { return FqPoly{0, 1}; }

bool lex_less(const FqPoly& a, const FqPoly& b) {
    if (a.size() != b.size()) return a.size() < b.size();
    for (std::size_t i = a.size(); i-- > 0;)
        if (a[i] != b[i]) return a[i] < b[i];
    return false;
}

// g = product of distinct monic irreducibles of degree d: split them.
void split_equal(const FqPoly& g, unsigned d, uint32_t q, std::mt19937_64& rng, std::vector<FqPoly>& out) {
    if (deg(g) == d) {
        out.push_back(fq_monic(g, q));
        return;
    }
    NF_CHECK(deg(g) % d == 0 && deg(g) > d);
    if (q == 2) {
        // brute force over the monic polynomials of degree d (each degree-d divisor is irreducible)
        NF_CHECK_MSG(d < 20, "split_equal: q = 2 brute force limited to d < 20");
        for (uint32_t m = 0; m < (1u << d); ++m) {
            FqPoly c(d + 1, 0);
            for (unsigned i = 0; i < d; ++i) c[i] = (m >> i) & 1u;
            c[d] = 1;
            FqPoly qq, r;
            fq_divrem(g, c, q, qq, r);
            if (r.empty()) {
                split_equal(c, d, q, rng, out);
                split_equal(qq, d, q, rng, out);
                return;
            }
        }
        NF_CHECK_MSG(false, "split_equal: no factor found (q = 2)");
    }
    for (int attempt = 0; attempt < 10000; ++attempt) {
        FqPoly a(deg(g));
        for (auto& x : a) x = static_cast<uint32_t>(rng() % q);
        a = fq_trim(a);
        if (a.empty()) continue;
        // t = a^{(q^d - 1)/2} = (a^{1 + q + ... + q^{d-1}})^{(q-1)/2} mod g
        FqPoly b = fq_mod(a, g, q), s = b;
        for (unsigned i = 1; i < d; ++i) {
            b = fq_powmod(b, q, g, q);
            s = fq_mod(fq_mul(s, b, q), g, q);
        }
        FqPoly t = fq_powmod(s, (q - 1) / 2, g, q);
        t = fq_sub(t, FqPoly{1}, q);
        const FqPoly g1 = fq_gcd(g, t, q);
        if (deg(g1) > 0 && deg(g1) < deg(g)) {
            FqPoly qq, r;
            fq_divrem(g, g1, q, qq, r);
            split_equal(g1, d, q, rng, out);
            split_equal(qq, d, q, rng, out);
            return;
        }
    }
    NF_CHECK_MSG(false, "split_equal: Cantor-Zassenhaus did not split");
}

}  // namespace

std::vector<FqPoly> fq_factor_squarefree(const FqPoly& f0, uint32_t q, unsigned maxdeg) {
    std::vector<FqPoly> out;
    FqPoly rest = fq_monic(f0, q);
    NF_CHECK_MSG(!rest.empty(), "fq_factor_squarefree: zero polynomial");
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL ^ (static_cast<uint64_t>(q) * 1000003ULL));
    FqPoly h = fq_mod(fq_x(), rest, q);
    unsigned d = 1;
    for (; deg(rest) >= 2 * d && d <= maxdeg; ++d) {
        h = fq_powmod(h, q, rest, q);  // x^{q^d} mod rest
        const FqPoly g = fq_gcd(rest, fq_sub(h, fq_x(), q), q);
        if (deg(g) > 0) {
            split_equal(g, d, q, rng, out);
            FqPoly qq, r;
            fq_divrem(rest, g, q, qq, r);
            NF_CHECK(r.empty());
            rest = fq_monic(qq, q);
            h = fq_mod(h, rest, q);
        }
    }
    if (deg(rest) > 0 && deg(rest) < 2 * d && deg(rest) <= maxdeg) out.push_back(rest);
    std::sort(out.begin(), out.end(), lex_less);
    return out;
}

std::vector<std::pair<FqPoly, unsigned>> fq_factor_with_multiplicity(const FqPoly& f0, uint32_t q) {
    FqPoly f = fq_monic(f0, q);
    NF_CHECK_MSG(!f.empty(), "fq_factor_with_multiplicity: zero polynomial");
    std::mt19937_64 rng(0x51ed27ULL ^ (static_cast<uint64_t>(q) * 7919ULL));
    std::vector<FqPoly> distinct;
    FqPoly h = fq_mod(fq_x(), f, q);
    for (unsigned d = 1; d <= deg(f); ++d) {
        h = fq_powmod(h, q, f, q);  // x^{q^d} mod f
        FqPoly P = fq_gcd(f, fq_sub(h, fq_x(), q), q);  // squarefree: all distinct irreducibles of degree | d
        for (const FqPoly& g : distinct) {
            if (d % deg(g) != 0) continue;
            FqPoly qq, r;
            fq_divrem(P, g, q, qq, r);
            if (r.empty()) P = qq;
        }
        P = fq_monic(P, q);
        if (deg(P) > 0) split_equal(P, d, q, rng, distinct);
    }
    std::sort(distinct.begin(), distinct.end(), lex_less);
    std::vector<std::pair<FqPoly, unsigned>> out;
    for (const FqPoly& g : distinct) {
        unsigned m = 0;
        for (;;) {
            FqPoly qq, r;
            fq_divrem(f, g, q, qq, r);
            if (!r.empty()) break;
            f = qq;
            ++m;
        }
        NF_CHECK(m >= 1);
        out.emplace_back(g, m);
    }
    NF_CHECK_MSG(deg(f) == 0, "fq_factor_with_multiplicity: incomplete factorisation");
    return out;
}

// ===========================================================================
// Dedekind and Round 2
// ===========================================================================
namespace {

IntPoly lift(const FqPoly& a) {
    IntPoly r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = BigInt(static_cast<int64_t>(a[i]));
    return r;
}

IntPoly int_mul(const IntPoly& a, const IntPoly& b) {
    if (a.empty() || b.empty()) return {};
    IntPoly r(a.size() + b.size() - 1, BigInt(0));
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) r[i + j] += a[i] * b[j];
    return r;
}

using RVec = std::vector<BigRational>;
using RMat = std::vector<RVec>;
using ZVec = std::vector<BigInt>;
using ZMat = std::vector<ZVec>;

// a b mod f, f monic of degree n, rational coefficients.
RVec rat_mulmod(const RVec& a, const RVec& b, const IntPoly& f) {
    const std::size_t n = f.size() - 1;
    RVec c(2 * n - 1, BigRational(0));
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i].is_zero()) continue;
        for (std::size_t j = 0; j < n; ++j)
            if (!b[j].is_zero()) c[i + j] += a[i] * b[j];
    }
    for (std::size_t k = 2 * n - 2; k >= n; --k) {
        if (!c[k].is_zero()) {
            const BigRational t = c[k];
            for (std::size_t i = 0; i < n; ++i)
                if (!f[i].is_zero()) c[k - n + i] -= t * BigRational(f[i]);
            c[k] = BigRational(0);
        }
        if (k == n) break;
    }
    c.resize(n);
    return c;
}

RMat rat_inverse(RMat A) {
    const std::size_t n = A.size();
    RMat I(n, RVec(n, BigRational(0)));
    for (std::size_t i = 0; i < n; ++i) I[i][i] = BigRational(1);
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t piv = c;
        while (piv < n && A[piv][c].is_zero()) ++piv;
        NF_CHECK_MSG(piv < n, "rat_inverse: singular");
        std::swap(A[c], A[piv]);
        std::swap(I[c], I[piv]);
        const BigRational inv = BigRational(1) / A[c][c];
        for (std::size_t j = 0; j < n; ++j) {
            A[c][j] *= inv;
            I[c][j] *= inv;
        }
        for (std::size_t r = 0; r < n; ++r) {
            if (r == c || A[r][c].is_zero()) continue;
            const BigRational t = A[r][c];
            for (std::size_t j = 0; j < n; ++j) {
                A[r][j] -= t * A[c][j];
                I[r][j] -= t * I[c][j];
            }
        }
    }
    return I;
}

BigRational rat_det(RMat A) {
    const std::size_t n = A.size();
    BigRational d(1);
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t piv = c;
        while (piv < n && A[piv][c].is_zero()) ++piv;
        if (piv == n) return BigRational(0);
        if (piv != c) {
            std::swap(A[c], A[piv]);
            d = -d;
        }
        d *= A[c][c];
        for (std::size_t r = c + 1; r < n; ++r) {
            if (A[r][c].is_zero()) continue;
            const BigRational t = A[r][c] / A[c][c];
            for (std::size_t j = c; j < n; ++j) A[r][j] -= t * A[c][j];
        }
    }
    return d;
}

RVec row_times(const RVec& x, const RMat& M) {
    const std::size_t n = M.size(), m = M[0].size();
    RVec r(m, BigRational(0));
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i].is_zero()) continue;
        for (std::size_t j = 0; j < m; ++j)
            if (!M[i][j].is_zero()) r[j] += x[i] * M[i][j];
    }
    return r;
}

BigInt to_int(const BigRational& x) {
    NF_CHECK_MSG(x.den() == BigInt(1), "round2: non-integral coordinate");
    return x.num();
}

// Row-style Hermite normal form of integer generators spanning a full-rank lattice in Z^n.
ZMat hnf_rows(ZMat rows, std::size_t n) {
    std::size_t r = 0;
    for (std::size_t col = 0; col < n && r < rows.size(); ++col) {
        for (;;) {
            std::size_t best = rows.size();
            for (std::size_t i = r; i < rows.size(); ++i)
                if (!rows[i][col].is_zero() && (best == rows.size() || rows[i][col].abs() < rows[best][col].abs())) best = i;
            if (best == rows.size()) break;
            std::swap(rows[r], rows[best]);
            bool other = false;
            for (std::size_t i = r + 1; i < rows.size(); ++i) {
                if (rows[i][col].is_zero()) continue;
                const BigInt t = rows[i][col] / rows[r][col];
                for (std::size_t j = 0; j < n; ++j) rows[i][j] -= t * rows[r][j];
                if (!rows[i][col].is_zero()) other = true;
            }
            if (!other) break;
        }
        if (r < rows.size() && !rows[r][col].is_zero()) {
            if (rows[r][col].sign() < 0)
                for (auto& x : rows[r]) x = -x;
            for (std::size_t i = 0; i < r; ++i) {
                const BigInt t = rows[i][col].floor_mod(rows[r][col]);
                const BigInt k = (rows[i][col] - t) / rows[r][col];
                if (!k.is_zero())
                    for (std::size_t j = 0; j < n; ++j) rows[i][j] -= k * rows[r][j];
            }
            ++r;
        }
    }
    NF_CHECK_MSG(r == n, "hnf_rows: lattice not of full rank");
    rows.resize(n);
    return rows;
}

// Basis of {v in F_q^r : v A = 0} for A (r x c over F_q).
std::vector<std::vector<uint32_t>> left_kernel(const std::vector<std::vector<uint32_t>>& A, std::size_t r, std::size_t c, uint32_t q) {
    // transpose: T (c x r), nullspace of T
    std::vector<std::vector<uint64_t>> T(c, std::vector<uint64_t>(r));
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) T[j][i] = A[i][j] % q;
    std::size_t row = 0;
    std::vector<int> where(r, -1);
    for (std::size_t col = 0; col < r && row < c; ++col) {
        std::size_t piv = row;
        while (piv < c && T[piv][col] == 0) ++piv;
        if (piv == c) continue;
        std::swap(T[row], T[piv]);
        const uint64_t inv = invmod32(T[row][col], q);
        for (auto& x : T[row]) x = x * inv % q;
        for (std::size_t i = 0; i < c; ++i) {
            if (i == row || T[i][col] == 0) continue;
            const uint64_t t = T[i][col];
            for (std::size_t j = 0; j < r; ++j) T[i][j] = (T[i][j] + q - t * T[row][j] % q) % q;
        }
        where[col] = static_cast<int>(row);
        ++row;
    }
    std::vector<std::vector<uint32_t>> ker;
    for (std::size_t free = 0; free < r; ++free) {
        if (where[free] >= 0) continue;
        std::vector<uint32_t> v(r, 0);
        v[free] = 1;
        for (std::size_t col = 0; col < r; ++col)
            if (where[col] >= 0) v[col] = static_cast<uint32_t>((q - T[where[col]][free] % q) % q);
        ker.push_back(v);
    }
    return ker;
}

}  // namespace

bool dedekind_maximal_at(const IntPoly& f, uint32_t q) {
    const FqPoly fb = fq_from_bigint(f, q);
    const auto facs = fq_factor_with_multiplicity(fb, q);
    FqPoly g{1}, h{1};
    for (const auto& gm : facs) {
        g = fq_mul(g, gm.first, q);
        for (unsigned i = 1; i < gm.second; ++i) h = fq_mul(h, gm.first, q);
    }
    const IntPoly gh = int_mul(lift(g), lift(h));
    IntPoly F(std::max(f.size(), gh.size()), BigInt(0));
    for (std::size_t i = 0; i < F.size(); ++i) {
        BigInt v = (i < f.size() ? f[i] : BigInt(0)) - (i < gh.size() ? gh[i] : BigInt(0));
        NF_CHECK_MSG(v.mod_u32(q) == 0, "dedekind: f - g h not divisible by q");
        F[i] = v / BigInt(static_cast<int64_t>(q));
    }
    const FqPoly Fb = fq_from_bigint(F, q);
    const FqPoly d = fq_gcd(fq_gcd(Fb, g, q), h, q);
    return deg(d) == 0;
}

Round2Result round2_at(const IntPoly& f, uint32_t q) {
    const std::size_t n = f.size() - 1;
    NF_CHECK(n >= 2 && f.back() == BigInt(1));
    RMat B(n, RVec(n, BigRational(0)));
    for (std::size_t i = 0; i < n; ++i) B[i][i] = BigRational(1);
    unsigned k = 1;
    for (uint64_t qk = q; qk < n; qk *= q) ++k;
    Round2Result res;
    for (unsigned iter = 0; iter < 64; ++iter) {
        const RMat Binv = rat_inverse(B);
        // multiplication table in omega coordinates
        std::vector<std::vector<ZVec>> T(n, std::vector<ZVec>(n));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i; j < n; ++j) {
                const RVec c = row_times(rat_mulmod(B[i], B[j], f), Binv);
                ZVec z(n);
                for (std::size_t t = 0; t < n; ++t) z[t] = to_int(c[t]);
                T[i][j] = z;
                T[j][i] = z;
            }
        auto mulq = [&](const std::vector<uint64_t>& x, const std::vector<uint64_t>& y) {
            std::vector<uint64_t> z(n, 0);
            for (std::size_t i = 0; i < n; ++i) {
                if (!x[i]) continue;
                for (std::size_t j = 0; j < n; ++j) {
                    if (!y[j]) continue;
                    const uint64_t xy = x[i] * y[j] % q;
                    for (std::size_t t = 0; t < n; ++t) z[t] = (z[t] + xy * T[i][j][t].mod_u32(q)) % q;
                }
            }
            return z;
        };
        // q-radical = kernel of Frobenius^k on O/qO
        std::vector<std::vector<uint32_t>> M(n, std::vector<uint32_t>(n));
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<uint64_t> v(n, 0);
            v[i] = 1;
            // power v^{q^k} by repeated q-th powers
            std::vector<uint64_t> cur = v;
            for (unsigned s = 0; s < k; ++s) {
                std::vector<uint64_t> acc;
                bool first = true;
                std::vector<uint64_t> b = cur;
                uint64_t e = q;
                while (e) {
                    if (e & 1u) {
                        acc = first ? b : mulq(acc, b);
                        first = false;
                    }
                    e >>= 1;
                    if (e) b = mulq(b, b);
                }
                cur = acc;
            }
            for (std::size_t t = 0; t < n; ++t) M[i][t] = static_cast<uint32_t>(cur[t]);
        }
        const auto K1 = left_kernel(M, n, n, q);
        ZMat gens;
        for (std::size_t i = 0; i < n; ++i) {
            ZVec e(n, BigInt(0));
            e[i] = BigInt(static_cast<int64_t>(q));
            gens.push_back(e);
        }
        for (const auto& v : K1) {
            ZVec z(n);
            for (std::size_t t = 0; t < n; ++t) z[t] = BigInt(static_cast<int64_t>(v[t]));
            gens.push_back(z);
        }
        const ZMat G = hnf_rows(gens, n);
        RMat Gr(n, RVec(n));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) Gr[i][j] = BigRational(G[i][j]);
        const RMat Ginv = rat_inverse(Gr);
        // x -> (multiplication by x on I / q I), O/qO -> F_q^{n x n}
        std::vector<std::vector<uint32_t>> Mm(n, std::vector<uint32_t>(n * n));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t l = 0; l < n; ++l) {
                RVec w(n, BigRational(0));
                for (std::size_t m = 0; m < n; ++m) {
                    if (G[l][m].is_zero()) continue;
                    for (std::size_t t = 0; t < n; ++t) w[t] += BigRational(G[l][m] * T[i][m][t]);
                }
                const RVec y = row_times(w, Ginv);
                for (std::size_t t = 0; t < n; ++t) Mm[i][l * n + t] = to_int(y[t]).mod_u32(q);
            }
        const auto K2 = left_kernel(Mm, n, n * n, q);
        ZMat gens2;
        for (std::size_t i = 0; i < n; ++i) {
            ZVec e(n, BigInt(0));
            e[i] = BigInt(static_cast<int64_t>(q));
            gens2.push_back(e);
        }
        for (const auto& v : K2) {
            ZVec z(n);
            for (std::size_t t = 0; t < n; ++t) z[t] = BigInt(static_cast<int64_t>(v[t]));
            gens2.push_back(z);
        }
        const ZMat U = hnf_rows(gens2, n);
        BigInt detU(1);
        for (std::size_t i = 0; i < n; ++i) detU *= U[i][i];
        detU = detU.abs();
        if (detU == BigInt::pow(BigInt(static_cast<int64_t>(q)), n)) break;  // q-maximal
        RMat NB(n, RVec(n, BigRational(0)));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t m = 0; m < n; ++m) {
                if (U[i][m].is_zero()) continue;
                for (std::size_t t = 0; t < n; ++t) NB[i][t] += BigRational(U[i][m], BigInt(static_cast<int64_t>(q))) * B[m][t];
            }
        B = NB;
        ++res.iterations;
        NF_CHECK_MSG(iter < 63, "round2: no convergence");
    }
    const BigRational d = rat_det(B);
    const BigRational inv = BigRational(1) / d;
    NF_CHECK_MSG(inv.den() == BigInt(1), "round2: index not integral");
    res.index = inv.num().abs();
    res.basis = B;
    return res;
}

// ===========================================================================
// The pure field
// ===========================================================================
long double bigint_to_ld(const BigInt& x) { return std::strtold(x.to_string().c_str(), nullptr); }

PureFieldData pure_field(uint32_t N, uint32_t p, bool force_zalpha) {
    PureFieldData K;
    K.N = N;
    K.p = p;
    if (!is_prime_u32(p) || p < 3) {
        K.refusal = "p is not an odd prime";
        return K;
    }
    if (!is_prime_u32(N) || N % p != 1) {
        K.refusal = "N is not a prime = 1 mod p";
        return K;
    }
    K.f.assign(p + 1, BigInt(0));
    K.f[0] = BigInt(-static_cast<int64_t>(N));
    K.f[p] = BigInt(1);
    K.disc_f = poly_discriminant(K.f);
    K.p_squared = ((N - 1) % (static_cast<uint64_t>(p) * p)) == 0;
    K.dedekind_p = dedekind_maximal_at(K.f, p);
    K.dedekind_N = dedekind_maximal_at(K.f, N);
    K.index = BigInt(1);
    K.order_maximal = true;
    for (uint32_t q : {p, N}) {
        const bool ok = (q == p) ? K.dedekind_p : K.dedekind_N;
        if (ok) continue;
        if (force_zalpha) {
            K.forced_zalpha = true;
            K.order_maximal = false;
            K.refusal = "Z[alpha] is not " + std::to_string(q) + "-maximal and Round 2 was suppressed (order not maximal)";
            continue;
        }
        const Round2Result R = round2_at(K.f, q);
        K.index *= R.index;
        K.round2_iterations += R.iterations;
    }
    K.dK = K.disc_f / (K.index * K.index);
    NF_CHECK(K.dK * K.index * K.index == K.disc_f);
    K.r1 = 1;
    K.r2 = (p - 1) / 2;
    long double fact = 1;
    for (uint32_t i = 2; i <= p; ++i) fact *= i;
    const long double pi = 3.14159265358979323846264338327950288L;
    K.minkowski = std::pow(4.0L / pi, static_cast<long double>(K.r2)) * fact / std::pow(static_cast<long double>(p), static_cast<long double>(p)) *
                  std::sqrt(std::fabs(bigint_to_ld(K.dK)));
    if (K.p_squared && K.order_maximal) {
        // the p-adic root u of x^p - N (u = 1 mod p), digit by digit, modulus p^{Kp + 2} < 2^63
        unsigned Kp = 0;
        uint64_t pk2 = 1;
        while (pk2 <= (static_cast<uint64_t>(1) << 62) / p) {
            pk2 *= p;
            ++Kp;
        }
        // pk2 = p^{Kp}; precision of u: Kp - 2
        const uint64_t Pmod = pk2;
        Kp -= 2;
        auto fval = [&](uint64_t u) {  // f(u) mod Pmod
            uint64_t r = 1;
            for (uint32_t i = 0; i < p; ++i) r = mulmod64(r, u, Pmod);
            return (r + Pmod - N % Pmod) % Pmod;
        };
        uint64_t u = 1, pj = p;
        for (unsigned j = 1; j < Kp; ++j) {
            const uint64_t next = pj * p;
            const uint64_t want = next * p;  // f(u) = 0 mod p^{j+2}
            bool found = false;
            for (uint32_t d = 0; d < p; ++d) {
                const uint64_t cand = u + d * pj;
                if (fval(cand) % want == 0) {
                    u = cand;
                    found = true;
                    break;
                }
            }
            NF_CHECK_MSG(found, "pure_field: p-adic root lifting failed");
            pj = next;
        }
        K.p_adic_root = u;
        K.p_adic_mod = pj;  // p^{Kp}
        NF_CHECK_MSG(K.dK.valuation(p) == p - 2, "pure_field: v_p(d_K) != p - 2 although p^2 | N - 1 (P_1 P_2^{p-1} shape)");
    }
    return K;
}

int FactorBase::find(uint32_t q, const FqPoly& h) const {
    if (h.size() == 2) {
        const auto it = deg1.find((static_cast<uint64_t>(q) << 32) | ((q - h[0]) % q));
        return it == deg1.end() ? -1 : static_cast<int>(it->second);
    }
    const auto it = by_q.find(q);
    if (it == by_q.end()) return -1;
    for (uint32_t i : it->second)
        if (ideals[i].kind == 0 && ideals[i].h == h) return static_cast<int>(i);
    return -1;
}

FactorBase pure_factor_base(const PureFieldData& K, uint64_t bound) {
    NF_CHECK_MSG(K.refusal.empty() || K.forced_zalpha, "pure_factor_base: refused field");
    NF_CHECK_MSG(bound < (static_cast<uint64_t>(1) << 32), "pure_factor_base: bound too large");
    const uint32_t p = K.p, N = K.N;
    FactorBase FB;
    std::vector<uint8_t> comp(bound + 1, 0);
    const FqPoly x{0, 1};
    for (uint64_t q = 2; q <= bound; ++q) {
        if (comp[q]) continue;
        for (uint64_t m = q * q; m <= bound; m += q) comp[m] = 1;
        const uint32_t qq = static_cast<uint32_t>(q);
        if (qq == N) {
            PrimeIdeal P;
            P.q = qq;
            P.e = p;
            P.h = x;
            P.root = 0;
            P.norm = q;
            FB.ideals.push_back(P);
            FB.all_over_q[qq] = true;
            continue;
        }
        if (qq == p) {
            if (!K.p_squared || !K.order_maximal) {
                PrimeIdeal P;
                P.q = qq;
                P.e = p;
                P.root = N % p;
                P.h = FqPoly{(p - P.root) % p, 1};
                P.norm = q;
                FB.ideals.push_back(P);
            } else {
                PrimeIdeal P1, P2;
                P1.q = P2.q = qq;
                P1.kind = 1;
                P2.kind = 2;
                P2.e = p - 1;
                P1.root = P2.root = 1;
                P1.norm = P2.norm = q;
                FB.ideals.push_back(P1);
                FB.ideals.push_back(P2);
            }
            FB.all_over_q[qq] = true;
            continue;
        }
        unsigned maxd = 0;
        for (uint64_t t = q; t <= bound && maxd < p; t *= q) ++maxd;
        const FqPoly fq = fq_from_bigint(K.f, qq);
        const auto facs = fq_factor_squarefree(fq, qq, maxd);
        unsigned sumdeg = 0;
        for (const FqPoly& h : facs) {
            PrimeIdeal P;
            P.q = qq;
            P.deg = deg(h);
            P.h = h;
            P.root = P.deg == 1 ? (qq - h[0]) % qq : 0;
            P.norm = ipow_u64(q, P.deg);
            sumdeg += P.deg;
            FB.ideals.push_back(P);
        }
        FB.all_over_q[qq] = (sumdeg == p);
    }
    std::stable_sort(FB.ideals.begin(), FB.ideals.end(), [](const PrimeIdeal& a, const PrimeIdeal& b) {
        if (a.norm != b.norm) return a.norm < b.norm;
        if (a.q != b.q) return a.q < b.q;
        if (a.kind != b.kind) return a.kind < b.kind;
        return lex_less(a.h, b.h);
    });
    for (uint32_t i = 0; i < FB.ideals.size(); ++i) {
        const PrimeIdeal& P = FB.ideals[i];
        FB.by_q[P.q].push_back(i);
        if (P.kind == 0 && P.deg == 1) FB.deg1[(static_cast<uint64_t>(P.q) << 32) | P.root] = i;
    }
    return FB;
}

bool pure_valuations(const PureFieldData& K, const FactorBase& FB, const std::vector<int64_t>& g,
                     const std::vector<std::pair<uint32_t, unsigned>>& norm_fact,
                     std::vector<std::pair<uint32_t, int64_t>>& out) {
    out.clear();
    const uint32_t p = K.p, N = K.N;
    for (const auto& qk : norm_fact) {
        const uint32_t q = qk.first;
        const unsigned k = qk.second;
        const auto it = FB.by_q.find(q);
        if (it == FB.by_q.end()) return false;
        if (q == N || (q == p && !(K.p_squared && K.order_maximal))) {
            out.emplace_back(it->second[0], k);
            continue;
        }
        if (q == p) {
            // v_{P_1} = v_p(g(u)), u the p-adic root to precision p^Kp
            const uint64_t M = K.p_adic_mod;
            uint64_t val = 0, up = 1;
            for (std::size_t i = 0; i < g.size(); ++i) {
                int64_t c = g[i] % static_cast<int64_t>(M);
                if (c < 0) c += static_cast<int64_t>(M);
                val = (val + mulmod64(static_cast<uint64_t>(c), up, M)) % M;
                up = mulmod64(up, K.p_adic_root, M);
            }
            if (val == 0) return false;  // valuation beyond the precision
            const unsigned v1 = valuation_u64(val, p);
            if (v1 > k) return false;
            int i1 = -1, i2 = -1;
            for (uint32_t i : it->second) {
                if (FB.ideals[i].kind == 1) i1 = static_cast<int>(i);
                if (FB.ideals[i].kind == 2) i2 = static_cast<int>(i);
            }
            if (i1 < 0 || i2 < 0) return false;
            if (v1) out.emplace_back(static_cast<uint32_t>(i1), v1);
            if (k - v1) out.emplace_back(static_cast<uint32_t>(i2), k - v1);
            continue;
        }
        const FqPoly gq = fq_from_int(g, q);
        const FqPoly fq = fq_from_bigint(K.f, q);
        const FqPoly hh = fq_gcd(gq, fq, q);
        const unsigned d = deg(hh);
        if (d == 0) return false;
        const int idx = FB.find(q, hh);
        if (idx < 0) return false;
        if (FB.ideals[idx].deg != d || k % d != 0) return false;
        out.emplace_back(static_cast<uint32_t>(idx), static_cast<int64_t>(k / d));
    }
    std::sort(out.begin(), out.end());
    return true;
}

BigInt pure_norm(const PureFieldData& K, const std::vector<int64_t>& g) {
    const uint32_t p = K.p;
    IntMat M(p, std::vector<BigInt>(p, BigInt(0)));
    for (uint32_t j = 0; j < p; ++j)
        for (std::size_t i = 0; i < g.size(); ++i) {
            if (!g[i]) continue;
            const std::size_t k = i + j;
            if (k < p)
                M[k][j] += BigInt(g[i]);
            else
                M[k - p][j] += BigInt(g[i]) * BigInt(static_cast<int64_t>(K.N));
        }
    return det_bareiss(M).abs();
}

}  // namespace nf
