// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-1b) AmalgamOracle -- see AmalgamOracle.h.  A line-by-line port of
// mv_check.py's sym / rank / kernel / cyc_data / dims_mv (section (1)).
#include "AmalgamOracle.h"

#include <utility>

#include "../../../ModArithmetic/EngineCheck.h"

namespace orbifold_mp {

namespace {

using Mat = std::vector<std::vector<int64_t>>;
using Vec = std::vector<int64_t>;

int64_t Md(int64_t x, int64_t p) {
    x %= p;
    return x < 0 ? x + p : x;
}

int64_t PowMod(int64_t b, int64_t e, int64_t p) {
    int64_t r = 1 % p;
    b = Md(b, p);
    while (e > 0) {
        if (e & 1) r = r * b % p;
        b = b * b % p;
        e >>= 1;
    }
    return r;
}

int64_t Inv(int64_t a, int64_t p) { return PowMod(a, p - 2, p); }

int64_t Comb(int64_t n, int64_t k) {
    int64_t r = 1;
    for (int64_t i = 1; i <= k; ++i) r = r * (n - k + i) / i;
    return r;
}

// sym(g, n, p): basis x^(n-j) y^j, g.x = a x + c y, g.y = b x + d y; column j = image of basis j.
Mat Sym(const int64_t g[4], uint32_t n, int64_t p) {
    const int64_t a = g[0], b = g[1], c = g[2], d = g[3];
    Mat M(n + 1, Vec(n + 1, 0));
    for (int64_t j = 0; j <= n; ++j)
        for (int64_t i1 = 0; i1 <= static_cast<int64_t>(n) - j; ++i1) {
            const int64_t c1 = Md(Comb(n - j, i1), p) * PowMod(a, n - j - i1, p) % p * PowMod(c, i1, p) % p;
            for (int64_t i2 = 0; i2 <= j; ++i2) {
                const int64_t c2 = Md(Comb(j, i2), p) * PowMod(b, j - i2, p) % p * PowMod(d, i2, p) % p;
                M[i1 + i2][j] = (M[i1 + i2][j] + c1 * c2) % p;
            }
        }
    return M;
}

Mat Mm(const Mat& A, const Mat& B, int64_t p) {
    Mat C(A.size(), Vec(B[0].size(), 0));
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t k = 0; k < B.size(); ++k)
            for (size_t j = 0; j < B[0].size(); ++j) C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % p;
    return C;
}

Mat Madd(const Mat& A, const Mat& B, int64_t p, int64_t s = 1) {
    Mat C = A;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j) C[i][j] = Md(A[i][j] + s * B[i][j], p);
    return C;
}

Mat Eye(size_t d) {
    Mat I(d, Vec(d, 0));
    for (size_t i = 0; i < d; ++i) I[i][i] = 1;
    return I;
}

uint32_t Rank(std::vector<Vec> rows, int64_t p) {
    if (rows.empty()) return 0;
    const size_t ncol = rows[0].size();
    size_t r = 0;
    for (size_t c = 0; c < ncol; ++c) {
        size_t piv = r;
        while (piv < rows.size() && Md(rows[piv][c], p) == 0) ++piv;
        if (piv == rows.size()) continue;
        std::swap(rows[r], rows[piv]);
        const int64_t inv = Inv(Md(rows[r][c], p), p);
        for (auto& x : rows[r]) x = Md(x * inv, p);
        for (size_t i = 0; i < rows.size(); ++i)
            if (i != r && Md(rows[i][c], p)) {
                const int64_t f = Md(rows[i][c], p);
                for (size_t t = 0; t < ncol; ++t) rows[i][t] = Md(rows[i][t] - f * rows[r][t], p);
            }
        ++r;
    }
    return static_cast<uint32_t>(r);
}

// Basis of ker A (A acting on column vectors).
std::vector<Vec> Kernel(const Mat& A, int64_t p) {
    const size_t m = A.size(), d = A[0].size();
    Mat rows = A;
    std::vector<size_t> pivcols;
    size_t r = 0;
    for (size_t c = 0; c < d; ++c) {
        size_t piv = r;
        while (piv < m && Md(rows[piv][c], p) == 0) ++piv;
        if (piv >= m) continue;
        std::swap(rows[r], rows[piv]);
        const int64_t inv = Inv(Md(rows[r][c], p), p);
        for (auto& x : rows[r]) x = Md(x * inv, p);
        for (size_t i = 0; i < m; ++i)
            if (i != r && Md(rows[i][c], p)) {
                const int64_t f = Md(rows[i][c], p);
                for (size_t t = 0; t < d; ++t) rows[i][t] = Md(rows[i][t] - f * rows[r][t], p);
            }
        pivcols.push_back(c);
        ++r;
    }
    std::vector<Vec> basis;
    for (size_t fc = 0; fc < d; ++fc) {
        bool is_piv = false;
        for (size_t pc : pivcols) is_piv = is_piv || pc == fc;
        if (is_piv) continue;
        Vec v(d, 0);
        v[fc] = 1;
        for (size_t i = 0; i < pivcols.size(); ++i) v[pivcols[i]] = Md(-rows[i][fc], p);
        basis.push_back(v);
    }
    return basis;
}

std::vector<Vec> Image(const Mat& A) {   // the columns
    std::vector<Vec> cols(A[0].size(), Vec(A.size(), 0));
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j) cols[j][i] = A[i][j];
    return cols;
}

Vec Apply(const Mat& A, const Vec& v, int64_t p) {
    Vec out(A.size(), 0);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < v.size(); ++j) out[i] = (out[i] + A[i][j] * v[j]) % p;
    return out;
}

struct HData {
    std::vector<Vec> V, W;
    Mat res;
};

struct Cyc {
    Mat gm1, Np, Nk, I;
    HData H(uint32_t q, int64_t p) const {
        if (q == 0) return {Kernel(gm1, p), {}, I};
        if (q % 2 == 1) return {Kernel(Np, p), Image(gm1), Nk};
        return {Kernel(gm1, p), Image(Np), I};
    }
};

Cyc CycData(const int64_t g[4], uint32_t order, uint32_t k, uint32_t n, int64_t p) {
    const size_t d = n + 1;
    const Mat R = Sym(g, n, p), I = Eye(d);
    Cyc C;
    C.gm1 = Madd(R, I, p, -1);
    C.I = I;
    C.Np = Mat(d, Vec(d, 0));
    Mat P = I;
    for (uint32_t t = 0; t < order; ++t) {
        C.Np = Madd(C.Np, P, p);
        P = Mm(P, R, p);
    }
    C.Nk = Mat(d, Vec(d, 0));
    P = I;
    for (uint32_t t = 0; t < k; ++t) {
        C.Nk = Madd(C.Nk, P, p);
        P = Mm(P, R, p);
    }
    return C;
}

}  // namespace

std::vector<uint32_t> AmalgamDims(uint32_t n, uint32_t p32, uint32_t q_max) {
    ENGINE_CHECK_MSG(p32 >= 2 && p32 < 65536, "AmalgamDims: 2 <= p < 2^16");
    const int64_t p = p32;
    const int64_t S[4] = {0, -1, 1, 0}, U[4] = {0, -1, 1, 1}, NEGI[4] = {-1, 0, 0, -1};   // U = S T
    const Cyc HA = CycData(S, 4, 2, n, p), HB = CycData(U, 6, 3, n, p), HC = CycData(NEGI, 2, 1, n, p);
    std::vector<int64_t> rk, src, tgt;
    for (uint32_t q = 0; q <= q_max; ++q) {
        const HData A = HA.H(q, p), B = HB.H(q, p), C = HC.H(q, p);
        const int64_t dA = A.V.empty() ? 0 : static_cast<int64_t>(A.V.size()) - Rank(A.W, p);
        const int64_t dB = B.V.empty() ? 0 : static_cast<int64_t>(B.V.size()) - Rank(B.W, p);
        const int64_t dC = C.V.empty() ? 0 : static_cast<int64_t>(C.V.size()) - Rank(C.W, p);
        std::vector<Vec> imgs;
        for (const Vec& v : A.V) imgs.push_back(Apply(A.res, v, p));
        for (const Vec& v : B.V) imgs.push_back(Apply(B.res, v, p));
        const int64_t wc = C.W.empty() ? 0 : Rank(C.W, p);
        std::vector<Vec> all = imgs;
        for (const Vec& w : C.W) all.push_back(w);
        const int64_t r = (all.empty() ? 0 : Rank(all, p)) - wc;
        rk.push_back(r);
        src.push_back(dA + dB);
        tgt.push_back(dC);
    }
    std::vector<uint32_t> out;
    for (uint32_t q = 0; q <= q_max; ++q) {
        const int64_t ker = src[q] - rk[q];
        const int64_t cok = q > 0 ? tgt[q - 1] - rk[q - 1] : 0;
        out.push_back(static_cast<uint32_t>(ker + cok));
    }
    return out;
}

const std::vector<MvCheckRow>& MvCheckOracle() {
    static const std::vector<MvCheckRow> rows = {
#include "MvCheckOracle.inc"
    };
    return rows;
}

}  // namespace orbifold_mp
