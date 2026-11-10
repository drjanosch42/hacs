// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Rung R6 (nf::BigInt reference) of the Z/2^k ladder.  See Ring2kReference.h.

#include "Ring2kReference.h"

#include <algorithm>
#include <stdexcept>

namespace engine {
namespace ring2k {
namespace ref6 {

using nf::BigInt;

namespace {

const BigInt& two64() {
    static const BigInt t = BigInt::pow(BigInt(2), 64);
    return t;
}
const BigInt& pow2(uint32_t e) {
    static std::vector<BigInt> table;
    while (table.size() <= e) table.push_back(BigInt::pow(BigInt(2), table.size()));
    return table[e];
}

struct Ring {
    uint32_t k;
    BigInt mod;
    explicit Ring(uint32_t k_) : k(k_), mod(pow2(k_)) {}
    BigInt red(const BigInt& x) const { return x.floor_mod(mod); }
    BigInt mul(const BigInt& a, const BigInt& b) const { return red(a * b); }
    BigInt sub(const BigInt& a, const BigInt& b) const { return red(a - b); }
    // Extended Euclid: u * x == 1 mod 2^k (u odd).
    BigInt inv(const BigInt& u) const {
        BigInt r0 = mod, r1 = red(u), s0 = 0, s1 = 1;
        while (!r1.is_zero()) {
            BigInt q, r;
            BigInt::divmod(r0, r1, q, r);
            r0 = r1; r1 = r;
            BigInt s = s0 - q * s1;
            s0 = s1; s1 = s;
        }
        if (r0 != BigInt(1)) throw std::logic_error("ref6: inverse of a non-unit");
        return red(s0);
    }
};

} // namespace

BigInt from_words(const uint64_t* w, uint32_t nw) {
    BigInt x = 0;
    for (uint32_t i = nw; i-- > 0;) x = x * two64() + BigInt::from_u64(w[i]);
    return x;
}
void to_words(const BigInt& x, uint64_t* w, uint32_t nw) {
    if (x.is_neg()) throw std::logic_error("ref6::to_words: negative");
    const BigInt::Mag& m = x.magnitude();   // little-endian 32-bit limbs
    for (uint32_t i = 0; i < nw; ++i) {
        const uint64_t lo = 2 * i < m.size() ? m[2 * i] : 0;
        const uint64_t hi = 2 * i + 1 < m.size() ? m[2 * i + 1] : 0;
        w[i] = lo | (hi << 32);
    }
}
BigMat from_wide(const WideMatrix& A) {
    BigMat M(A.rows, A.cols);
    const Ring R(A.k);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) M.at(i, j) = R.red(from_words(A.entry(i, j), A.nw));
    return M;
}
WideMatrix to_wide(const BigMat& M, uint32_t k) {
    WideMatrix A(M.rows, M.cols, k);
    const Ring R(k);
    for (uint32_t i = 0; i < M.rows; ++i)
        for (uint32_t j = 0; j < M.cols; ++j) to_words(R.red(M.at(i, j)), A.entry(i, j), A.nw);
    return A;
}
uint32_t valuation(const BigInt& x, uint32_t k) {
    if (x.is_zero()) return k;
    const BigInt::Mag& m = x.magnitude();
    uint32_t v = 0;
    for (size_t i = 0; i < m.size(); ++i) {
        if (m[i] == 0) { v += 32; continue; }
        uint32_t limb = m[i];
        while ((limb & 1u) == 0) { limb >>= 1; ++v; }
        break;
    }
    return v < k ? v : k;
}

SmithResult smith(const WideMatrix& Aw) {
    const Ring R(Aw.k);
    const uint32_t k = Aw.k;
    BigMat A = from_wide(Aw);
    const uint32_t m = A.rows, n = A.cols;
    SmithResult res;
    res.k = k;
    for (uint32_t r = 0; r < m && r < n; ++r) {
        uint32_t bi = m, bj = n, bv = k;
        for (uint32_t i = r; i < m; ++i)
            for (uint32_t j = r; j < n; ++j) {
                const uint32_t v = valuation(A.at(i, j), k);
                if (v < bv) { bv = v; bi = i; bj = j; }
            }
        if (bi == m) break;
        for (uint32_t j = 0; j < n; ++j) std::swap(A.at(bi, j), A.at(r, j));
        for (uint32_t i = 0; i < m; ++i) std::swap(A.at(i, bj), A.at(i, r));
        const BigInt& d = pow2(bv);
        const BigInt ui = R.inv(A.at(r, r) / d);
        // clear the column (row operations) ...
        for (uint32_t i = r + 1; i < m; ++i) {
            if (A.at(i, r).is_zero()) continue;
            const BigInt f = R.mul(A.at(i, r) / d, ui);
            for (uint32_t j = r; j < n; ++j) A.at(i, j) = R.sub(A.at(i, j), f * A.at(r, j));
        }
        // ... and the row (column operations)
        for (uint32_t j = r + 1; j < n; ++j) {
            if (A.at(r, j).is_zero()) continue;
            const BigInt f = R.mul(A.at(r, j) / d, ui);
            for (uint32_t i = r; i < m; ++i) A.at(i, j) = R.sub(A.at(i, j), f * A.at(i, r));
        }
        for (uint32_t i = r + 1; i < m; ++i)
            if (!A.at(i, r).is_zero()) throw std::logic_error("ref6::smith: column not cleared");
        for (uint32_t j = r + 1; j < n; ++j)
            if (!A.at(r, j).is_zero()) throw std::logic_error("ref6::smith: row not cleared");
        res.sigma.push_back(bv);
    }
    std::sort(res.sigma.begin(), res.sigma.end());
    return res;
}

WideHowell howell(const WideMatrix& Aw) {
    const Ring R(Aw.k);
    const uint32_t k = Aw.k;
    const BigMat A = from_wide(Aw);
    const uint32_t n = A.cols;
    std::vector<std::vector<BigInt>> rows;
    for (uint32_t i = 0; i < A.rows; ++i) rows.emplace_back(A.a.begin() + static_cast<std::ptrdiff_t>(i) * n,
                                                            A.a.begin() + static_cast<std::ptrdiff_t>(i + 1) * n);
    std::vector<size_t> active;
    for (size_t i = 0; i < rows.size(); ++i) active.push_back(i);
    std::vector<size_t> prow;
    std::vector<uint32_t> pivots, pivot_v;
    for (uint32_t c = 0; c < n; ++c) {
        size_t best = active.size();
        uint32_t bv = k;
        for (size_t t = 0; t < active.size(); ++t) {
            const uint32_t v = valuation(rows[active[t]][c], k);
            if (v < bv) { bv = v; best = t; }
        }
        if (best == active.size()) continue;
        const size_t pid = active[best];
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(best));
        std::vector<BigInt>& P = rows[pid];
        const BigInt& d = pow2(bv);
        const BigInt ui = R.inv(P[c] / d);
        for (uint32_t j = 0; j < n; ++j) P[j] = R.mul(P[j], ui);
        if (P[c] != d) throw std::logic_error("ref6::howell: pivot not 2^v");
        for (size_t id : active) {
            std::vector<BigInt>& X = rows[id];
            if (X[c].is_zero()) continue;
            const BigInt f = X[c] / d;   // exact: v(X[c]) >= bv
            for (uint32_t j = 0; j < n; ++j) X[j] = R.sub(X[j], f * P[j]);
        }
        if (bv > 0) {
            std::vector<BigInt> Z(n);
            for (uint32_t j = 0; j < n; ++j) Z[j] = R.mul(P[j], pow2(k - bv));
            rows.push_back(std::move(Z));
            active.push_back(rows.size() - 1);
        }
        prow.push_back(pid);
        pivots.push_back(c);
        pivot_v.push_back(bv);
    }
    // Reduce above, one pivot at a time, left to right.
    for (size_t t = 0; t < prow.size(); ++t) {
        const std::vector<BigInt>& P = rows[prow[t]];
        const BigInt& d = pow2(pivot_v[t]);
        for (size_t i = 0; i < t; ++i) {
            std::vector<BigInt>& X = rows[prow[i]];
            const BigInt q = X[pivots[t]] / d;   // floor of a non-negative integer
            if (q.is_zero()) continue;
            for (uint32_t j = 0; j < n; ++j) X[j] = R.sub(X[j], q * P[j]);
        }
    }
    WideHowell h;
    BigMat H(static_cast<uint32_t>(prow.size()), n);
    for (size_t i = 0; i < prow.size(); ++i)
        for (uint32_t j = 0; j < n; ++j) H.at(static_cast<uint32_t>(i), j) = rows[prow[i]][j];
    h.H = to_wide(H, k);
    h.pivots = pivots;
    h.pivot_v = pivot_v;
    return h;
}

} // namespace ref6
} // namespace ring2k
} // namespace engine
