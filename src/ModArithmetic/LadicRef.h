// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// LadicRef.h -- the naive base-ell oracle for the (C2) digit induction
// (ENGINE_API.md Sec. 13.11.7; hecke-extensions slot 11, WP-18b)
// =====================================================================
//
// engine::ref::ladic: Z/ell^K on K single base-ell digits (uint32_t each,
// ell < 2^31), schoolbook arithmetic with uint64_t carries -- an arithmetic
// INDEPENDENT of engine::ladic's word digits, used where no native oracle
// exists (ell^K >= 2^127).  Like engine::ref (Sec. 7), it is naive and never
// optimised; it reads everything DIRECTLY off A:
//   * howell(A)          -- the Sec. 13.5 forward pass + closure + reduce-above;
//   * kernel_module(A)   -- Sec. 13.6's definition: the rows of
//                           howell([A^T | I_n]) with leading column >= m;
//   * smith_exponents(A) -- a two-sided minimal-valuation Smith elimination
//                           (row-major pivot search: a different tie rule on
//                           purpose; the exponents are unique).
// Elements are std::vector<uint32_t> of exactly K digits, little-endian --
// the serialisation LadicMat::ell_digits() returns.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "EngineCheck.h"

namespace engine {
namespace ref {
namespace ladic {

using Elem = std::vector<uint32_t>;
using Mat = std::vector<std::vector<Elem>>;   // row-major

struct Ring {
    uint64_t ell = 0;
    uint32_t K = 0;
    Ring(uint64_t e, uint32_t k) : ell(e), K(k) {
        ENGINE_CHECK(ell >= 2 && ell < (uint64_t(1) << 31) && K >= 1);
    }
    Elem zero() const { return Elem(K, 0); }
    Elem one() const { Elem e(K, 0); e[0] = 1; return e; }
    Elem from_u64(uint64_t v) const {
        Elem e(K, 0);
        for (uint32_t i = 0; i < K; ++i) { e[i] = static_cast<uint32_t>(v % ell); v /= ell; }
        return e;
    }
    bool is_zero(const Elem& a) const {
        for (uint32_t d : a) if (d) return false;
        return true;
    }
    Elem add(const Elem& a, const Elem& b) const {
        Elem z(K);
        uint64_t c = 0;
        for (uint32_t i = 0; i < K; ++i) {
            const uint64_t t = uint64_t(a[i]) + b[i] + c;
            z[i] = static_cast<uint32_t>(t % ell);
            c = t / ell;
        }
        return z;
    }
    Elem sub(const Elem& a, const Elem& b) const {
        Elem z(K);
        int64_t br = 0;
        for (uint32_t i = 0; i < K; ++i) {
            int64_t t = int64_t(a[i]) - int64_t(b[i]) - br;
            if (t < 0) { t += static_cast<int64_t>(ell); br = 1; } else br = 0;
            z[i] = static_cast<uint32_t>(t);
        }
        return z;
    }
    Elem neg(const Elem& a) const { return sub(zero(), a); }
    // schoolbook: every digit product < ell^2 < 2^62 is split at once into
    // (hi, lo) < ell each; a column sum of 2K such terms is < 2^64.
    Elem mul(const Elem& a, const Elem& b) const {
        std::vector<uint64_t> col(K + 1, 0);
        for (uint32_t i = 0; i < K; ++i) {
            if (!a[i]) continue;
            for (uint32_t j = 0; i + j < K; ++j) {
                if (!b[j]) continue;
                const uint64_t p = uint64_t(a[i]) * b[j];
                col[i + j] += p % ell;
                col[i + j + 1] += p / ell;
            }
        }
        Elem z(K);
        uint64_t c = 0;
        for (uint32_t i = 0; i < K; ++i) {
            const uint64_t t = col[i] + c;
            z[i] = static_cast<uint32_t>(t % ell);
            c = t / ell;
        }
        return z;
    }
    uint32_t valuation(const Elem& a) const {
        for (uint32_t i = 0; i < K; ++i) if (a[i]) return i;
        return K;
    }
    Elem shr(const Elem& a, uint32_t v) const {   // floor(a / ell^v)
        Elem z(K, 0);
        for (uint32_t i = 0; i + v < K; ++i) z[i] = a[i + v];
        return z;
    }
    Elem shl(const Elem& a, uint32_t v) const {   // a ell^v mod ell^K
        Elem z(K, 0);
        for (uint32_t i = v; i < K; ++i) z[i] = a[i - v];
        return z;
    }
    Elem ell_pow(uint32_t v) const { Elem z(K, 0); if (v < K) z[v] = 1; return z; }
    // inverse of a unit by Newton from the inverse mod ell (Fermat: a^(ell-2))
    Elem inv(const Elem& u) const {
        if (u[0] == 0) throw std::runtime_error("Modular inverse does not exist.");
        uint64_t base = u[0], e = ell - 2, r = 1;
        while (e) { if (e & 1) r = (r * base) % ell; base = (base * base) % ell; e >>= 1; }
        Elem y = zero();
        y[0] = static_cast<uint32_t>(r);
        const Elem two = from_u64(2);
        for (uint32_t prec = 1; prec < K; prec *= 2) y = mul(y, sub(two, mul(u, y)));
        const Elem chk = mul(u, y);
        ENGINE_CHECK(chk == one());
        return y;
    }
    Elem unit_part(const Elem& a) const {
        const uint32_t v = valuation(a);
        return v >= K ? one() : shr(a, v);
    }
};

struct HowellOut {
    Mat H;
    std::vector<uint32_t> pivots, pivot_v;
};

// Sec. 13.5, directly on the rows of A (n columns).
inline HowellOut howell(const Ring& R, Mat rows, uint32_t n) {
    HowellOut out;
    std::vector<size_t> active;
    for (size_t i = 0; i < rows.size(); ++i) active.push_back(i);
    std::vector<size_t> prow;
    for (uint32_t c = 0; c < n && !active.empty(); ++c) {
        size_t best = active.size();
        uint32_t bv = R.K;
        for (size_t t = 0; t < active.size(); ++t) {
            const uint32_t v = R.valuation(rows[active[t]][c]);
            if (v < bv) { bv = v; best = t; }
        }
        if (best == active.size()) continue;
        const size_t p = active[best];
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(best));
        const Elem ui = R.inv(R.unit_part(rows[p][c]));
        for (uint32_t j = c; j < n; ++j) rows[p][j] = R.mul(rows[p][j], ui);
        for (size_t r : active) {
            if (R.is_zero(rows[r][c])) continue;
            const Elem q = R.shr(rows[r][c], bv);
            for (uint32_t j = c; j < n; ++j) rows[r][j] = R.sub(rows[r][j], R.mul(q, rows[p][j]));
        }
        if (bv > 0) {
            std::vector<Elem> cl(n, R.zero());
            for (uint32_t j = c + 1; j < n; ++j) cl[j] = R.shl(rows[p][j], R.K - bv);
            rows.push_back(cl);
            active.push_back(rows.size() - 1);
        }
        prow.push_back(p);
        out.pivots.push_back(c);
        out.pivot_v.push_back(bv);
    }
    for (size_t p : prow) out.H.push_back(rows[p]);
    for (size_t i = 0; i < out.H.size(); ++i) {
        const uint32_t ji = out.pivots[i];
        for (size_t ip = 0; ip < i; ++ip) {
            const Elem q = R.shr(out.H[ip][ji], out.pivot_v[i]);
            if (R.is_zero(q)) continue;
            for (uint32_t j = ji; j < n; ++j) out.H[ip][j] = R.sub(out.H[ip][j], R.mul(q, out.H[i][j]));
        }
    }
    return out;
}

// Sec. 13.6's definition, verbatim: howell([A^T | I_n]), rows leading at >= m.
inline HowellOut kernel_module(const Ring& R, const Mat& A, uint32_t m, uint32_t n) {
    Mat aug(n, std::vector<Elem>(m + n, R.zero()));
    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t i = 0; i < m; ++i) aug[j][i] = A[i][j];
        aug[j][m + j] = R.one();
    }
    const HowellOut full = howell(R, aug, m + n);
    HowellOut out;
    for (size_t i = 0; i < full.pivots.size(); ++i) {
        if (full.pivots[i] < m) continue;
        out.H.emplace_back(full.H[i].begin() + m, full.H[i].end());
        out.pivots.push_back(full.pivots[i] - m);
        out.pivot_v.push_back(full.pivot_v[i]);
    }
    return out;
}

// The exponents s of the Smith form diag(ell^s) (+) 0, ascending: two-sided
// minimal-valuation elimination, pivot search ROW-major (topmost row, then
// leftmost column) -- a different tie rule from the engine's on purpose.
inline std::vector<uint32_t> smith_exponents(const Ring& R, Mat A, uint32_t m, uint32_t n) {
    std::vector<uint32_t> s;
    std::vector<char> ru(m, 0), cu(n, 0);
    for (;;) {
        uint32_t pr = m, pc = n, pv = R.K;
        for (uint32_t i = 0; i < m; ++i) {
            if (ru[i]) continue;
            for (uint32_t j = 0; j < n; ++j) {
                if (cu[j]) continue;
                const uint32_t v = R.valuation(A[i][j]);
                if (v < pv) { pv = v; pr = i; pc = j; }
            }
        }
        if (pr == m) break;
        const Elem ui = R.inv(R.unit_part(A[pr][pc]));
        for (uint32_t j = 0; j < n; ++j) if (!cu[j]) A[pr][j] = R.mul(A[pr][j], ui);
        for (uint32_t i = 0; i < m; ++i) {
            if (ru[i] || i == pr || R.is_zero(A[i][pc])) continue;
            const Elem q = R.shr(A[i][pc], pv);
            for (uint32_t j = 0; j < n; ++j) if (!cu[j]) A[i][j] = R.sub(A[i][j], R.mul(q, A[pr][j]));
        }
        // the pivot row's other entries are ell^pv multiples; clearing them
        // by column operations changes no other active row (column pc is zero
        // there), so the remaining block is already the Schur complement.
        ru[pr] = 1;
        cu[pc] = 1;
        s.push_back(pv);
    }
    std::sort(s.begin(), s.end());
    return s;
}

inline Mat multiply(const Ring& R, const Mat& A, const Mat& B) {
    const size_t m = A.size(), k = B.size(), n = k ? B[0].size() : 0;
    Mat C(m, std::vector<Elem>(n, R.zero()));
    for (size_t i = 0; i < m; ++i)
        for (size_t l = 0; l < k; ++l) {
            if (R.is_zero(A[i][l])) continue;
            for (size_t j = 0; j < n; ++j) C[i][j] = R.add(C[i][j], R.mul(A[i][l], B[l][j]));
        }
    return C;
}

} // namespace ladic
} // namespace ref
} // namespace engine
