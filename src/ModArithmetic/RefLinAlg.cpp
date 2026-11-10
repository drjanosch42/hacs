// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "RefLinAlg.h"

#include "EngineCheck.h"
#include "ModArithmetic.h"

#include <algorithm>
#include <stdexcept>

// The oracle. Naive on purpose -- see RefLinAlg.h.

// ---------------------------------------------------------------------------
// (slot-11 WP-18a) the naive u128 arithmetic (RefLinAlg.h, naive128), and the
// legacy-spelled overloads the templates below call, so that the u128
// instantiations of the oracle run on it.  Internal to this TU.
// ---------------------------------------------------------------------------
namespace engine {
namespace ref {
namespace naive128 {

u128 add(u128 a, u128 b, u128 p) {
    // a, b < p < 2^127: no wrap
    const u128 r = a + b;
    return r >= p ? r - p : r;
}
u128 sub(u128 a, u128 b, u128 p) { return a >= b ? a - b : a + (p - b); }
u128 mul(u128 a, u128 b, u128 p) {
    u128 r = 0;
    for (int i = 127; i >= 0; --i) {
        r = add(r, r, p);
        if ((b >> i) & 1) r = add(r, a, p);
    }
    return r;
}
u128 inv(u128 a, u128 p) {
    // extended Euclid on (p, a); the Bezout coefficient of a as (magnitude, sign)
    u128 r0 = p, r1 = a % p;
    u128 s0 = 0, s1 = 1;
    bool n0 = false, n1 = false;          // true: the coefficient is negative
    while (r1 != 0) {
        const u128 q = r0 / r1;
        const u128 r2 = r0 - q * r1;
        // s2 = s0 - q * s1 with signs; |q s1| <= p keeps every magnitude < 2^128
        const u128 qs = q * s1;
        u128 s2;
        bool n2;
        if (n0 != n1) {                   // s0 and -q s1 have the same sign
            s2 = s0 + qs; n2 = n0;
        } else if (s0 >= qs) {
            s2 = s0 - qs; n2 = n0;
        } else {
            s2 = qs - s0; n2 = !n0;
        }
        r0 = r1; r1 = r2;
        s0 = s1; n0 = n1;
        s1 = s2; n1 = n2;
    }
    if (r0 != 1) throw std::runtime_error("Modular inverse does not exist.");
    const u128 m = s0 % p;
    return (n0 && m != 0) ? p - m : m;
}

} // namespace naive128
} // namespace ref
} // namespace engine

namespace ModArithmetic {
static inline unsigned __int128 mod_mul_w(unsigned __int128 a, unsigned __int128 b, unsigned __int128 p) { return engine::ref::naive128::mul(a, b, p); }
static inline unsigned __int128 mod_add_w(unsigned __int128 a, unsigned __int128 b, unsigned __int128 p) { return engine::ref::naive128::add(a, b, p); }
static inline unsigned __int128 mod_sub_w(unsigned __int128 a, unsigned __int128 b, unsigned __int128 p) { return engine::ref::naive128::sub(a, b, p); }
static inline unsigned __int128 mod_inv_w(unsigned __int128 a, unsigned __int128 p) { return engine::ref::naive128::inv(a, p); }
} // namespace ModArithmetic

namespace engine {
namespace ref {

namespace {

template<class Elt>
void combine(Elt p, Elt& c, Elt r, int op) {
    switch (op) {
        case 0: c = r; break;
        case 1: c = ModArithmetic::mod_add_w(c, r, p); break;
        default: c = ModArithmetic::mod_sub_w(c, r, p); break;
    }
}

} // namespace

template<class Elt>
void gemm(Elt p, DenseView<const Elt> A, DenseView<const Elt> B, DenseView<Elt> C, int op) {
    ENGINE_CHECK(A.cols() == B.rows() && C.rows() == A.rows() && C.cols() == B.cols());
    const uint32_t m = A.rows(), k = A.cols(), n = B.cols();
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            Elt s = 0;
            for (uint32_t t = 0; t < k; ++t) {
                s = ModArithmetic::mod_add_w(s, ModArithmetic::mod_mul_w(A.at(i, t), B.at(t, j), p), p);
            }
            combine(p, C.at(i, j), s, op);
        }
    }
}

template<class Elt>
void gemv(Elt p, DenseView<const Elt> A, DenseView<const Elt> x, DenseView<Elt> y, int op) {
    ENGINE_CHECK(x.cols() == 1 && y.cols() == 1);
    gemm<Elt>(p, A, x, y, op);
}

template<class Elt>
void trmm(Elt p, DenseView<const Elt> T, DenseView<const Elt> B, DenseView<Elt> C, bool upper) {
    const uint32_t m = T.rows(), n = B.cols();
    ENGINE_CHECK(T.cols() == m && B.rows() == m && C.rows() == m && C.cols() == n);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            Elt s = B.at(i, j);                         // unit diagonal
            const uint32_t k0 = upper ? i + 1 : 0;
            const uint32_t k1 = upper ? m : i;
            for (uint32_t t = k0; t < k1; ++t) {
                s = ModArithmetic::mod_add_w(s, ModArithmetic::mod_mul_w(T.at(i, t), B.at(t, j), p), p);
            }
            C.at(i, j) = s;
        }
    }
}

template<class Elt>
void transpose(DenseView<const Elt> src, DenseView<Elt> dst) {
    ENGINE_CHECK(dst.rows() == src.cols() && dst.cols() == src.rows());
    for (uint32_t i = 0; i < src.rows(); ++i)
        for (uint32_t j = 0; j < src.cols(); ++j)
            dst.at(j, i) = src.at(i, j);
}

template<class Elt>
std::vector<uint32_t> rref(Elt p, DenseView<Elt> M) {
    const uint32_t rows = M.rows(), cols = M.cols();
    std::vector<uint32_t> pivots;
    uint32_t r = 0;
    for (uint32_t c = 0; c < cols && r < rows; ++c) {
        // topmost non-zero row from r downward
        uint32_t piv = r;
        while (piv < rows && M.at(piv, c) == 0) ++piv;
        if (piv == rows) continue;
        if (piv != r) {
            for (uint32_t j = 0; j < cols; ++j) { Elt t = M.at(r, j); M.at(r, j) = M.at(piv, j); M.at(piv, j) = t; }
        }
        // leading 1
        const Elt inv = ModArithmetic::mod_inv_w(M.at(r, c), p);
        for (uint32_t j = 0; j < cols; ++j) M.at(r, j) = ModArithmetic::mod_mul_w(M.at(r, j), inv, p);
        // eliminate above and below
        for (uint32_t i = 0; i < rows; ++i) {
            if (i == r) continue;
            const Elt f = M.at(i, c);
            if (f == 0) continue;
            for (uint32_t j = 0; j < cols; ++j) {
                M.at(i, j) = ModArithmetic::mod_sub_w(M.at(i, j),
                                 ModArithmetic::mod_mul_w(f, M.at(r, j), p), p);
            }
        }
        pivots.push_back(c);
        ++r;
    }
    return pivots;
}

template<class Elt>
std::vector<std::vector<Elt>> kernel_basis(Elt p, DenseView<const Elt> A) {
    DenseBuf<Elt> R(A.rows(), A.cols());
    for (uint32_t i = 0; i < A.rows(); ++i)
        for (uint32_t j = 0; j < A.cols(); ++j) R.at(i, j) = A.at(i, j);
    DenseView<Elt> Rv(R);
    const std::vector<uint32_t> piv = rref<Elt>(p, Rv);

    std::vector<bool> is_pivot(A.cols(), false);
    for (uint32_t c : piv) is_pivot[c] = true;

    std::vector<std::vector<Elt>> basis;
    for (uint32_t jf = 0; jf < A.cols(); ++jf) {
        if (is_pivot[jf]) continue;
        std::vector<Elt> v(A.cols(), Elt(0));
        v[jf] = 1;
        for (uint32_t i = 0; i < piv.size(); ++i) {
            v[piv[i]] = ModArithmetic::mod_minus_w(R.at(i, jf), p);
        }
        basis.push_back(std::move(v));
    }
    return basis;
}

// ---------------------------------------------------------------------------
// Z/ell^k (see RefLinAlg.h).  Naive on purpose.
// ---------------------------------------------------------------------------
namespace {

template<class Elt>
Elt ref_ell_pow(Elt ell, uint32_t v) {
    unsigned __int128 x = 1;
    for (uint32_t i = 0; i < v; ++i) {
        // (slot-11 WP-18a) the bound is checked BEFORE the product, so a u128
        // Elt cannot wrap; for u32 / u64 the same condition, equivalently.
        ENGINE_CHECK_MSG(x <= static_cast<unsigned __int128>(static_cast<Elt>(~Elt(0))) / ell,
                         "ref::howell: ell^k does not fit the element type");
        x *= ell;
    }
    return static_cast<Elt>(x);
}

// the ell-adic valuation in [0, k]; valuation(0) = k
template<class Elt>
uint32_t ref_val(Elt x, Elt ell, uint32_t k) {
    if (x == 0) return k;
    uint32_t v = 0;
    while (x % ell == 0) { x /= ell; ++v; }
    return v;
}

template<class Elt>
void ref_row_sub(std::vector<Elt>& r, const std::vector<Elt>& s, Elt q, Elt p) {
    for (size_t j = 0; j < r.size(); ++j)
        r[j] = ModArithmetic::mod_sub_w(r[j], ModArithmetic::mod_mul_w(q, s[j], p), p);
}

} // namespace

template<class Elt>
RefHowell<Elt> howell(Elt ell, uint32_t k, DenseView<const Elt> M) {
    ENGINE_CHECK(ell >= 2 && k >= 1);
    const Elt p = ref_ell_pow<Elt>(ell, k);
    const uint32_t n = M.cols();
    std::vector<std::vector<Elt>> active;
    for (uint32_t i = 0; i < M.rows(); ++i) {
        std::vector<Elt> r(n);
        for (uint32_t j = 0; j < n; ++j) r[j] = M.at(i, j);
        active.push_back(r);
    }
    RefHowell<Elt> out;
    for (uint32_t c = 0; c < n; ++c) {
        // 1. minimal valuation, ties to the topmost active row
        size_t best = active.size();
        uint32_t bv = k;
        for (size_t t = 0; t < active.size(); ++t) {
            if (active[t][c] == 0) continue;
            const uint32_t v = ref_val<Elt>(active[t][c], ell, k);
            if (v < bv) { bv = v; best = t; }
        }
        if (best == active.size()) continue;
        std::vector<Elt> P = active[best];
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(best));
        // 2. scale by the inverse of the unit part
        Elt u = P[c];
        while (u % ell == 0) u /= ell;
        const Elt ui = ModArithmetic::mod_inv_w(u, p);
        for (uint32_t j = 0; j < n; ++j) P[j] = ModArithmetic::mod_mul_w(P[j], ui, p);
        const Elt lead = ref_ell_pow<Elt>(ell, bv);
        ENGINE_CHECK(P[c] == lead);
        // 3. eliminate every other active row
        for (auto& r : active) {
            if (r[c] == 0) continue;
            ENGINE_CHECK(r[c] % lead == 0);
            ref_row_sub<Elt>(r, P, static_cast<Elt>(r[c] / lead), p);
            ENGINE_CHECK(r[c] == 0);
        }
        // 4. the closure row, at the bottom, whenever v > 0
        if (bv > 0) {
            const Elt f = ref_ell_pow<Elt>(ell, k - bv);
            std::vector<Elt> cl(n);
            for (uint32_t j = 0; j < n; ++j) cl[j] = ModArithmetic::mod_mul_w(f, P[j], p);
            active.push_back(cl);
        }
        out.rows.push_back(P);
        out.pivots.push_back(c);
        out.pivot_v.push_back(bv);
    }
    // reduce above, pivot columns left to right
    for (size_t i = 0; i < out.rows.size(); ++i) {
        const uint32_t ji = out.pivots[i];
        const Elt d = ref_ell_pow<Elt>(ell, out.pivot_v[i]);
        for (size_t ip = 0; ip < i; ++ip) {
            const Elt q = static_cast<Elt>(out.rows[ip][ji] / d);
            if (q != 0) ref_row_sub<Elt>(out.rows[ip], out.rows[i], q, p);
        }
    }
    return out;
}

template<class Elt>
std::vector<uint32_t> invariant_factors(Elt ell, uint32_t k, DenseView<const Elt> G, DenseView<const Elt> Rl) {
    const Elt p = ref_ell_pow<Elt>(ell, k);
    const uint32_t m = G.rows(), ng = G.cols(), nr = Rl.cols();
    ENGINE_CHECK(Rl.rows() == m);
    const uint32_t w = ng + nr;
    // [[G | Rl]^T | I_w]   (w x (m + w))
    DenseBuf<Elt> T(w, m + w);
    for (uint32_t j = 0; j < w; ++j) {
        for (uint32_t i = 0; i < m; ++i) T.at(j, i) = (j < ng) ? G.at(i, j) : Rl.at(i, j - ng);
        T.at(j, m + j) = 1;
    }
    const RefHowell<Elt> H = howell<Elt>(ell, k, DenseView<const Elt>(T));
    // relations: the G-coordinates of the kernel rows (leading column >= m);
    // the Smith matrix S is ng x (#relations), one column per relation
    std::vector<std::vector<Elt>> rel;
    for (size_t i = 0; i < H.rows.size(); ++i) {
        if (H.pivots[i] < m) continue;
        std::vector<Elt> c(ng);
        for (uint32_t g = 0; g < ng; ++g) c[g] = H.rows[i][m + g];
        rel.push_back(c);
    }
    const size_t r = rel.size();
    std::vector<std::vector<Elt>> S(ng, std::vector<Elt>(r, Elt(0)));
    for (size_t c = 0; c < r; ++c)
        for (uint32_t g = 0; g < ng; ++g) S[g][c] = rel[c][g];
    std::vector<char> ru(ng, 0), cu(r, 0);
    std::vector<uint32_t> e;
    uint32_t npiv = 0;
    for (;;) {
        uint32_t bi = ng, bv = k;
        size_t bj = 0;
        for (uint32_t i = 0; i < ng; ++i) {             // ROW-major search, on purpose
            if (ru[i]) continue;
            for (size_t j = 0; j < r; ++j) {
                if (cu[j] || S[i][j] == 0) continue;
                const uint32_t v = ref_val<Elt>(S[i][j], ell, k);
                if (v < bv) { bv = v; bi = i; bj = j; }
            }
        }
        if (bi == ng) break;
        const Elt d = ref_ell_pow<Elt>(ell, bv);
        Elt u = S[bi][bj];
        while (u % ell == 0) u /= ell;
        const Elt ui = ModArithmetic::mod_inv_w(u, p);
        // clear the column first (row operations), then the row (column operations)
        for (uint32_t i = 0; i < ng; ++i) {
            if (i == bi || ru[i] || S[i][bj] == 0) continue;
            const Elt q = ModArithmetic::mod_mul_w(static_cast<Elt>(S[i][bj] / d), ui, p);
            for (size_t j = 0; j < r; ++j)
                S[i][j] = ModArithmetic::mod_sub_w(S[i][j], ModArithmetic::mod_mul_w(q, S[bi][j], p), p);
            ENGINE_CHECK(S[i][bj] == 0);
        }
        for (size_t j = 0; j < r; ++j) {
            if (j == bj || cu[j] || S[bi][j] == 0) continue;
            const Elt q = ModArithmetic::mod_mul_w(static_cast<Elt>(S[bi][j] / d), ui, p);
            for (uint32_t i = 0; i < ng; ++i)
                S[i][j] = ModArithmetic::mod_sub_w(S[i][j], ModArithmetic::mod_mul_w(q, S[i][bj], p), p);
            ENGINE_CHECK(S[bi][j] == 0);
        }
        ru[bi] = 1; cu[bj] = 1; ++npiv;
        if (bv >= 1) e.push_back(bv);
    }
    for (uint32_t i = npiv; i < ng; ++i) e.push_back(k);
    std::sort(e.begin(), e.end(), [](uint32_t a, uint32_t b) { return a > b; });
    return e;
}

#define ENGINE_REF_INSTANTIATE(Elt)                                                          \
    template void gemm<Elt>(Elt, DenseView<const Elt>, DenseView<const Elt>, DenseView<Elt>, int); \
    template void gemv<Elt>(Elt, DenseView<const Elt>, DenseView<const Elt>, DenseView<Elt>, int); \
    template void trmm<Elt>(Elt, DenseView<const Elt>, DenseView<const Elt>, DenseView<Elt>, bool); \
    template void transpose<Elt>(DenseView<const Elt>, DenseView<Elt>);                          \
    template std::vector<uint32_t> rref<Elt>(Elt, DenseView<Elt>);                               \
    template std::vector<std::vector<Elt>> kernel_basis<Elt>(Elt, DenseView<const Elt>);         \
    template RefHowell<Elt> howell<Elt>(Elt, uint32_t, DenseView<const Elt>);                    \
    template std::vector<uint32_t> invariant_factors<Elt>(Elt, uint32_t, DenseView<const Elt>,    \
                                                          DenseView<const Elt>);

ENGINE_REF_INSTANTIATE(uint32_t)
ENGINE_REF_INSTANTIATE(uint64_t)
#undef ENGINE_REF_INSTANTIATE

// (slot-11 WP-18a) u128, on naive128 (composite routes and the modulus-only
// kernels; rref / kernel_basis are field routes and are not instantiated).
using u128_ = unsigned __int128;
template void gemm<u128_>(u128_, DenseView<const u128_>, DenseView<const u128_>, DenseView<u128_>, int);
template void gemv<u128_>(u128_, DenseView<const u128_>, DenseView<const u128_>, DenseView<u128_>, int);
template void trmm<u128_>(u128_, DenseView<const u128_>, DenseView<const u128_>, DenseView<u128_>, bool);
template void transpose<u128_>(DenseView<const u128_>, DenseView<u128_>);
template RefHowell<u128_> howell<u128_>(u128_, uint32_t, DenseView<const u128_>);
template std::vector<uint32_t> invariant_factors<u128_>(u128_, uint32_t, DenseView<const u128_>, DenseView<const u128_>);

} // namespace ref
} // namespace engine
