// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZCharpoly -- slot-25 HZ-RAT-PERF (b1): det(x I - A) over F_ell by a Hessenberg reduction, O(b^3) (see the header).

#include "HeckeZCharpoly.h"

#include "../../ModArithmetic/EngineCheck.h"

#include <utility>

namespace heckez {

Barrett31::Barrett31(uint64_t prime) : p(prime), m(~0ull / prime) {
    ENGINE_CHECK_MSG(prime >= 3 && prime < (1ull << 31), "Barrett31: 3 <= p < 2^31");
}

uint64_t Barrett31::pow(uint64_t a, uint64_t e) const {
    uint64_t r = 1 % p, b = a % p;
    while (e) {
        if (e & 1u) r = mul(r, b);
        b = mul(b, b);
        e >>= 1;
    }
    return r;
}

namespace {
// Shoup's multiplication by a fixed w < p < 2^31: w' = floor(w 2^32 / p); for x < 2^32, x w - floor(x w' / 2^32) p lies in [0, 2p).
struct Shoup {
    uint64_t w, wp, p;
    Shoup(uint64_t w_, uint64_t p_) : w(w_), wp((w_ << 32) / p_), p(p_) {}
    uint32_t mul(uint32_t x) const {
        const uint64_t q = (static_cast<uint64_t>(x) * wp) >> 32;
        uint64_t r = static_cast<uint64_t>(x) * w - q * p;
        return static_cast<uint32_t>(r >= p ? r - p : r);
    }
};
inline uint32_t AddMod(uint32_t a, uint32_t b, uint32_t p) {
    const uint32_t s = a + b;   // a, b < p < 2^31: no wrap
    return s >= p ? s - p : s;
}
}  // namespace

std::vector<uint64_t> CharpolyHessenberg(const std::vector<uint64_t>& A, uint32_t b, uint64_t ell) {
    ENGINE_CHECK_MSG(ell >= 3 && ell % 2 == 1 && ell < (1ull << 31), "CharpolyHessenberg: an odd prime ell < 2^31");
    ENGINE_CHECK_MSG(A.size() == static_cast<size_t>(b) * b, "CharpolyHessenberg: a b x b matrix");
    if (b == 0) return {1};
    const Barrett31 br(ell);
    const uint32_t p = static_cast<uint32_t>(ell);
    std::vector<uint32_t> H(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        ENGINE_CHECK_MSG(A[i] < ell, "CharpolyHessenberg: entries reduced mod ell");
        H[i] = static_cast<uint32_t>(A[i]);
    }
    auto row = [&](uint32_t i) { return &H[static_cast<size_t>(i) * b]; };

    // ---- Hessenberg form by similarity: for each column j, a pivot at (j+1, j), zeros below it ----
    for (uint32_t j = 0; j + 2 < b; ++j) {
        uint32_t r = j + 1;
        while (r < b && row(r)[j] == 0) ++r;
        if (r == b) continue;
        if (r != j + 1) {   // P H P^{-1} for the transposition (r, j+1)
            uint32_t* x = row(r);
            uint32_t* y = row(j + 1);
            for (uint32_t k = 0; k < b; ++k) std::swap(x[k], y[k]);
            for (uint32_t k = 0; k < b; ++k) std::swap(row(k)[r], row(k)[j + 1]);
        }
        const uint64_t inv = br.inv(row(j + 1)[j]);
        const uint32_t* piv = row(j + 1);
        for (uint32_t i = j + 2; i < b; ++i) {
            uint32_t* ri = row(i);
            if (ri[j] == 0) continue;
            const uint64_t u = br.mul(ri[j], inv);   // != 0
            // E H with E = I - u e_i e_{j+1}^T: row_i -= u row_{j+1} (columns < j vanish in both rows)
            const Shoup neg(p - u, p);
            for (uint32_t k = j; k < b; ++k) ri[k] = AddMod(ri[k], neg.mul(piv[k]), p);
            // (E H) E^{-1}, E^{-1} = I + u e_i e_{j+1}^T: column_{j+1} += u column_i
            const Shoup pos(u, p);
            for (uint32_t k = 0; k < b; ++k) {
                uint32_t* rk = row(k);
                rk[j + 1] = AddMod(rk[j + 1], pos.mul(rk[i]), p);
            }
        }
    }

    // ---- the Hessenberg recurrence (Cohen Alg. 2.2.9), 0-indexed:
    // p_{m+1} = (x - h_mm) p_m - sum_{i=1}^{m} h_{m-i,m} (prod_{t=m-i+1}^{m} h_{t,t-1}) p_{m-i}
    std::vector<std::vector<uint64_t>> P(b + 1);
    P[0] = {1};
    for (uint32_t m = 0; m < b; ++m) {
        std::vector<uint64_t>& q = P[m + 1];
        const std::vector<uint64_t>& pm = P[m];
        q.assign(m + 2, 0);
        const uint64_t hmm = row(m)[m];
        for (uint32_t k = 0; k <= m; ++k) {
            q[k + 1] = q[k + 1] + pm[k];
            if (q[k + 1] >= p) q[k + 1] -= p;
            const uint64_t t = br.mul(hmm, pm[k]);
            q[k] = q[k] >= t ? q[k] - t : q[k] + p - t;
        }
        uint64_t t = 1;
        for (uint32_t i = 1; i <= m; ++i) {
            t = br.mul(t, row(m - i + 1)[m - i]);
            if (t == 0) break;   // every later product contains this zero subdiagonal entry
            const uint64_t c = br.mul(t, row(m - i)[m]);
            if (c == 0) continue;
            const std::vector<uint64_t>& pl = P[m - i];
            for (uint32_t k = 0; k < pl.size(); ++k) {
                const uint64_t s = br.mul(c, pl[k]);
                q[k] = q[k] >= s ? q[k] - s : q[k] + p - s;
            }
        }
    }
    return P[b];
}

}  // namespace heckez
