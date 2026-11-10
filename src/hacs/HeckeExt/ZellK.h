// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_ZELLK_H
#define HECKE_EXT_ZELLK_H

// ===========================================================================
// ZellK.h -- ONE u64 Z/ell^K kit for the LValue / Hida modules (S24 CL-G1;
// CODEAUDIT_A_heckeext.md A5; docs/implementation/slot24/CLEANUP_plan_1030.md).
//
// Before CL-G1 the ring, the 2x2 integral matrix, the free-module matrix and the
// small dense routines (product, Berkowitz characteristic polynomial, rank mod
// ell) were written out separately in HidaPairing, BirchModularSymbol,
// EisensteinBernoulliRoute, HeckeHidaTower, HeckeLatticeModule and
// Gamma0NShapiro.  This header is their single home:
//
//   Ring       Z/ell^m, ell^m < 2^64, uint64_t residues, unsigned __int128
//              products (was birch::Ring and hidapair::Zq; both names are
//              now aliases of it);
//   M22        an integral 2x2 matrix (was hidapair::M22 and birch::Mat2);
//   FreeMat    a square matrix on a free Z/ell^K-module (was latmod::FreeMat
//              and lvalue::FreeMat) with its arithmetic;
//   Charpoly   the division-free Berkowitz characteristic polynomial over
//              Z/q, on any square accessor (was four copies);
//   FieldRank  rank (and a nullspace basis) over F_ell of a row list.
//
// VALUE CONTRACT.  Every routine returns canonical residues in [0, q), and
// every replaced copy did too, so the consolidation is value-identical: a
// characteristic polynomial, a product and a rank do not depend on the order of
// the reductions.  (FieldRank's nullspace basis is the RREF basis of the
// HeckeHidaTower copy it replaces, verbatim.)
//
// NOT MERGED, BY DESIGN: HeckeHidaLambda's u128 ZMod and HeckeHidaLambdaForms'
// u128 kit (genuinely wider); HeckeLambdaTower's u32 kit over hecke::level::ZK
// (u64 products of u32 residues with lazy reduction: 1.2-1.8x faster than this kit's
// u128 MatMul at n = 120, q = 37^6, CL-G1 micro-benchmark); and the
// oracles that check other routes (HeckeLatticeModule's RankModEllJoint, the
// Voronoi Bernoulli route, RhoDirectMatrix).
//
// Header-only, STL-only (plus EngineCheck.h); no state, no env, no file.
// ===========================================================================

#include "../../ModArithmetic/EngineCheck.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace hecke_ext {
namespace zk {

using u128 = unsigned __int128;
using Vec = std::vector<uint64_t>;

// ---- Z/ell^m -------------------------------------------------------------------------------------------
struct Ring {
    uint64_t ell = 0;
    uint32_t m = 0;
    uint64_t q = 1;   // a default Ring is Z/1 (hidapair::Zq's default; birch's rings are always assigned before use)

    Ring() = default;
    // The HidaPairing constructor (was hidapair::Zq(ell, M)): ell >= 2, m >= 1 and ell^m <= 2^62, the bound the
    // lazy u128 accumulations of hidapair rely on.  (MakeRing below admits ell^m < 2^64.)
    Ring(uint64_t ell_, uint32_t m_) : ell(ell_), m(m_), q(1) {
        ENGINE_CHECK_MSG(ell_ >= 2 && m_ >= 1, "hidapair::Zq: ell >= 2 and M >= 1");
        for (uint32_t i = 0; i < m_; ++i) {
            ENGINE_CHECK_MSG(q <= (uint64_t(1) << 62) / ell_, "hidapair::Zq: ell^M must stay below 2^62");
            q *= ell_;
        }
    }

    uint64_t mul(uint64_t a, uint64_t b) const { return static_cast<uint64_t>((static_cast<u128>(a) * b) % q); }
    uint64_t add(uint64_t a, uint64_t b) const { return a >= q - b ? a - (q - b) : a + b; }
    uint64_t sub(uint64_t a, uint64_t b) const { return a >= b ? a - b : a + (q - b); }
    uint64_t neg(uint64_t a) const { return a == 0 ? 0 : q - a; }
    bool unit(uint64_t a) const { return a % ell != 0; }

    // x mod q in [0, q) for every int64_t x (u64 arithmetic only; -(x + 1) cannot overflow at INT64_MIN).
    uint64_t from_int(int64_t x) const {
        if (x >= 0) return static_cast<uint64_t>(x) % q;
        const uint64_t r = static_cast<uint64_t>(-(x + 1)) % q;
        return r == q - 1 ? 0 : q - 1 - r;
    }
    uint64_t pow(uint64_t b, uint64_t e) const {
        uint64_t r = 1 % q;
        b %= q;
        while (e) {
            if (e & 1) r = mul(r, b);
            b = mul(b, b);
            e >>= 1;
        }
        return r;
    }
    // a an ell-unit (checked).
    uint64_t inv(uint64_t a) const {
        ENGINE_CHECK_MSG(unit(a), "zk::Ring::inv: not an ell-unit");
        __int128 t = 0, nt = 1, r = q, nr = a % q;
        while (nr != 0) {
            const __int128 qq = r / nr;
            __int128 x = t - qq * nt;
            t = nt;
            nt = x;
            x = r - qq * nr;
            r = nr;
            nr = x;
        }
        ENGINE_CHECK_MSG(r == 1, "zk::Ring::inv: gcd != 1");
        if (t < 0) t += q;
        return static_cast<uint64_t>(t);
    }
    // v_ell(a), m for a == 0 mod q.
    uint32_t val(uint64_t a) const {
        a %= q;
        if (a == 0) return m;
        uint32_t v = 0;
        while (a % ell == 0) {
            a /= ell;
            ++v;
        }
        return v;
    }
    // ell^v (v <= m; ell^m reads as 0).
    uint64_t ell_pow(uint32_t v) const {
        ENGINE_CHECK_MSG(v <= m, "birch::Ring::ell_pow: v > m");
        if (v == m) return 0;
        uint64_t r = 1;
        for (uint32_t i = 0; i < v; ++i) r *= ell;
        return r;
    }
    // The largest M <= cap with ell^M < 2^62 (the hidapair precision rule).
    static uint32_t MaxPrecision(uint64_t ell, uint32_t cap) {
        uint32_t M = 0;
        u128 q = 1;
        while (M < cap && q * ell < (static_cast<u128>(1) << 62)) {
            q *= ell;
            ++M;
        }
        return M;
    }
};

// ell >= 2, m >= 1, ell^m < 2^64 (checked).
inline Ring MakeRing(uint64_t ell, uint32_t m) {
    ENGINE_CHECK_MSG(ell >= 2 && m >= 1, "birch::MakeRing: ell >= 2, m >= 1");
    u128 acc = 1;
    for (uint32_t i = 0; i < m; ++i) {
        acc *= ell;
        ENGINE_CHECK_MSG(acc < (static_cast<u128>(1) << 64), "birch::MakeRing: ell^m >= 2^64");
    }
    Ring R;
    R.ell = ell;
    R.m = m;
    R.q = static_cast<uint64_t>(acc);
    return R;
}

// ---- integral 2x2 matrices -----------------------------------------------------------------------------
struct M22 {
    int64_t a = 1, b = 0, c = 0, d = 1;
};
inline M22 Mul(const M22& x, const M22& y) {
    return {x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d, x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d};
}
inline M22 Adj(const M22& x) { return {x.d, -x.b, -x.c, x.a}; }

// ---- small dense matrices as row lists ------------------------------------------------------------------
// C = A B over Z/q (A n x m, B m x p, rows of std::vector<uint64_t>; entries any representative).  Reads R.q only.
inline std::vector<Vec> MatMul(const Ring& R, const std::vector<Vec>& A, const std::vector<Vec>& B) {
    if (A.empty()) return {};
    const size_t n = A.size(), m = B.size(), p = B.empty() ? 0 : B[0].size();
    std::vector<Vec> C(n, Vec(p, 0));
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < m; ++k) {
            const uint64_t a = A[i][k];
            if (!a) continue;
            for (size_t j = 0; j < p; ++j) C[i][j] = R.add(C[i][j], R.mul(a, B[k][j]));
        }
    return C;
}

// A^e (A square), by squaring.
inline std::vector<Vec> MatPow(const Ring& R, std::vector<Vec> A, uint64_t e) {
    const size_t d = A.size();
    std::vector<Vec> P(d, Vec(d, 0));
    for (size_t i = 0; i < d; ++i) P[i][i] = 1 % R.q;
    while (e) {
        if (e & 1) P = MatMul(R, P, A);
        A = MatMul(R, A, A);
        e >>= 1;
    }
    return P;
}
// A v.
inline Vec MatVec(const Ring& R, const std::vector<Vec>& A, const Vec& v) {
    Vec o(A.size(), 0);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < v.size(); ++j)
            if (A[i][j] && v[j]) o[i] = R.add(o[i], R.mul(A[i][j], v[j]));
    return o;
}

// ---- the characteristic polynomial -----------------------------------------------------------------------
// Berkowitz, division-free: det(X - A) over Z/q, coefficients LOW -> HIGH (monic).  `at(i, j)` returns the
// entry (i, j) of the n x n matrix, any representative (it is reduced mod q here).  Reads R.q only.
template <class At>
Vec Charpoly(const Ring& R, size_t n, At at) {
    Vec a(n * n);   // the matrix reduced once, flat row-major
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) a[i * n + j] = static_cast<uint64_t>(at(i, j)) % R.q;
    Vec C(1, 1 % R.q);   // high -> low
    for (size_t r = 0; r < n; ++r) {
        Vec T(r + 2, 0);   // Toeplitz column: T[0] = 1, T[1] = -a_rr, T[k+2] = -R_row M^k S_col
        T[0] = 1 % R.q;
        T[1] = R.neg(a[r * n + r]);
        Vec Sv(r);
        for (size_t i = 0; i < r; ++i) Sv[i] = a[i * n + r];
        for (size_t kk = 0; kk < r; ++kk) {
            uint64_t s = 0;
            for (size_t i = 0; i < r; ++i) s = R.add(s, R.mul(a[r * n + i], Sv[i]));
            T[kk + 2] = R.neg(s);
            if (kk + 1 == r) break;
            Vec nS(r, 0);
            for (size_t i = 0; i < r; ++i)
                for (size_t j = 0; j < r; ++j) {
                    const uint64_t aij = a[i * n + j];
                    if (aij && Sv[j]) nS[i] = R.add(nS[i], R.mul(aij, Sv[j]));
                }
            Sv.swap(nS);
        }
        Vec Cn(r + 2, 0);
        for (size_t i = 0; i < r + 2; ++i)
            for (size_t j = 0; j < C.size() && j <= i; ++j) Cn[i] = R.add(Cn[i], R.mul(T[i - j], C[j]));
        C.swap(Cn);
    }
    return Vec(C.rbegin(), C.rend());
}

// ---- rank over F_ell ------------------------------------------------------------------------------------
// Rank over F_ell (ell prime) of the rows of M (each of length cols; entries any representative), by RREF;
// with null_basis, the RREF nullspace basis (one vector per free column, free entry 1).  (Was
// HeckeHidaTower's FieldRank, verbatim.)
inline uint32_t FieldRank(std::vector<Vec> M, uint32_t cols, uint64_t ell, std::vector<Vec>* null_basis = nullptr) {
    const Ring F = MakeRing(ell, 1);
    const uint32_t rows = static_cast<uint32_t>(M.size());
    for (auto& r : M)
        for (auto& x : r) x %= ell;
    std::vector<int> col_pivot(cols, -1);
    uint32_t r = 0;
    for (uint32_t c = 0; c < cols && r < rows; ++c) {
        uint32_t piv = rows;
        for (uint32_t i = r; i < rows; ++i)
            if (M[i][c]) {
                piv = i;
                break;
            }
        if (piv == rows) continue;
        std::swap(M[piv], M[r]);
        const uint64_t iv = F.pow(M[r][c], ell - 2);
        for (uint32_t j = 0; j < cols; ++j) M[r][j] = F.mul(M[r][j], iv);
        for (uint32_t i = 0; i < rows; ++i) {
            if (i == r || !M[i][c]) continue;
            const uint64_t f = M[i][c];
            for (uint32_t j = 0; j < cols; ++j) M[i][j] = F.sub(M[i][j], F.mul(f, M[r][j]));
        }
        col_pivot[c] = static_cast<int>(r);
        ++r;
    }
    if (null_basis) {
        null_basis->clear();
        for (uint32_t fc = 0; fc < cols; ++fc) {
            if (col_pivot[fc] >= 0) continue;
            Vec v(cols, 0);
            v[fc] = 1;
            for (uint32_t c = 0; c < cols; ++c)
                if (col_pivot[c] >= 0) v[c] = F.sub(0, M[static_cast<size_t>(col_pivot[c])][fc]);
            null_basis->push_back(v);
        }
    }
    return r;
}

// ---- square matrices on a free Z/ell^K-module ----------------------------------------------------------------
// Column convention (X b_j = sum_i a[i*r+j] b_i), row-major, entries in [0, ell^K), ell^K < 2^62.
struct FreeMat {
    uint64_t ell = 0;
    uint32_t K = 0;
    uint32_t r = 0;
    std::vector<uint64_t> a;
    uint64_t q() const {
        uint64_t x = 1;
        for (uint32_t i = 0; i < K; ++i) {
            ENGINE_CHECK_MSG(x <= (uint64_t(1) << 62) / ell, "HeckeLatticeModule: ell^K must stay below 2^62");
            x *= ell;
        }
        return x;
    }
    uint64_t at(uint32_t i, uint32_t j) const { return a[static_cast<size_t>(i) * r + j]; }
    bool operator==(const FreeMat& o) const { return ell == o.ell && K == o.K && r == o.r && a == o.a; }
    bool operator!=(const FreeMat& o) const { return !(*this == o); }
};

inline FreeMat Zero(uint64_t ell, uint32_t K, uint32_t r) {
    FreeMat M;
    M.ell = ell;
    M.K = K;
    M.r = r;
    M.a.assign(static_cast<size_t>(r) * r, 0);
    return M;
}
inline FreeMat Identity(uint64_t ell, uint32_t K, uint32_t r) {
    FreeMat I = Zero(ell, K, r);
    const uint64_t one = 1 % I.q();
    for (uint32_t i = 0; i < r; ++i) I.a[static_cast<size_t>(i) * r + i] = one;
    return I;
}
inline FreeMat Mul(const FreeMat& A, const FreeMat& B) {
    ENGINE_CHECK(A.ell == B.ell && A.K == B.K && A.r == B.r);
    const uint64_t q = A.q();
    FreeMat C = Zero(A.ell, A.K, A.r);
    const uint32_t r = A.r;
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) {
            // products < q^2 < 2^124: eight of them (plus a reduced carry) never wrap a u128.
            u128 acc = 0;
            for (uint32_t k = 0; k < r; ++k) {
                acc += static_cast<u128>(A.a[static_cast<size_t>(i) * r + k]) * B.a[static_cast<size_t>(k) * r + j];
                if ((k & 7) == 7) acc %= q;
            }
            C.a[static_cast<size_t>(i) * r + j] = static_cast<uint64_t>(acc % q);
        }
    return C;
}
// alpha A + beta B.
inline FreeMat Lin(const FreeMat& A, uint64_t alpha, const FreeMat& B, uint64_t beta) {
    ENGINE_CHECK(A.ell == B.ell && A.K == B.K && A.r == B.r);
    const uint64_t q = A.q();
    FreeMat C = Zero(A.ell, A.K, A.r);
    for (size_t t = 0; t < A.a.size(); ++t)
        C.a[t] = static_cast<uint64_t>((static_cast<u128>(A.a[t]) * (alpha % q) + static_cast<u128>(B.a[t]) * (beta % q)) % q);
    return C;
}
inline FreeMat Add(const FreeMat& A, const FreeMat& B) { return Lin(A, 1, B, 1); }
// c * A (c reduced mod ell^K).
inline FreeMat Scale(const FreeMat& A, uint64_t c) {
    FreeMat C = A;
    const uint64_t q = A.q();
    for (uint64_t& x : C.a) x = static_cast<uint64_t>(static_cast<u128>(x) * (c % q) % q);
    return C;
}
// A + c * 1.
inline FreeMat AddScalar(const FreeMat& A, int64_t c) {
    const uint64_t q = A.q();
    const int64_t cm = c % static_cast<int64_t>(q);
    const uint64_t cu = static_cast<uint64_t>(cm < 0 ? cm + static_cast<int64_t>(q) : cm);
    return Lin(A, 1, Identity(A.ell, A.K, A.r), cu);
}
// A - c * 1.
inline FreeMat MinusScalar(const FreeMat& A, uint64_t c) {
    const uint64_t q = A.q();
    return Lin(A, 1, Identity(A.ell, A.K, A.r), (q - c % q) % q);
}
// The same matrix over Z/ell^M, 1 <= M <= K.
inline FreeMat ReduceTo(const FreeMat& A, uint32_t M) {
    ENGINE_CHECK(M >= 1 && M <= A.K);
    FreeMat C = A;
    C.K = M;
    const uint64_t q = C.q();
    for (uint64_t& x : C.a) x %= q;
    return C;
}
inline FreeMat DirectSum(const FreeMat& A, const FreeMat& B) {
    ENGINE_CHECK(A.ell == B.ell && A.K == B.K);
    FreeMat C = Zero(A.ell, A.K, A.r + B.r);
    const uint32_t r = C.r;
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.r; ++j) C.a[static_cast<size_t>(i) * r + j] = A.at(i, j);
    for (uint32_t i = 0; i < B.r; ++i)
        for (uint32_t j = 0; j < B.r; ++j) C.a[static_cast<size_t>(A.r + i) * r + A.r + j] = B.at(i, j);
    return C;
}
// P(X) by Horner, coefficients low -> high, each taken mod ell^K.
inline FreeMat EvalPoly(const std::vector<uint64_t>& coeffs, const FreeMat& X) {
    FreeMat R = Zero(X.ell, X.K, X.r);
    const FreeMat I = Identity(X.ell, X.K, X.r);
    for (size_t k = coeffs.size(); k-- > 0;) R = Lin(Mul(R, X), 1, I, coeffs[k]);
    return R;
}

} // namespace zk
} // namespace hecke_ext

#endif // HECKE_EXT_ZELLK_H
