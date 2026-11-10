// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// LadicInduction.h -- the (C2) ell-adic digit induction, engine::ladic
// (ENGINE_API.md Sec. 13.11; hecke-extensions slot 11, WP-18b)
// =====================================================================
//
// Z/ell^K for ANY K >= 2 on L = ceil(K/m) fixed-width digit words of type
// W (uint32_t, uint64_t, u128 -- the widths of WP-18a's DigitWord<W>),
// m = m(W, ell) the largest exponent with ell^m < 2^(w-1).  Composite-only,
// CPU, host, single-threaded, header-only; it records no program and has no
// device path (Sec. 13.11 status).  Nothing in the tree calls it before
// WP-18c's pipeline route, so it moves no existing value.
//
//  * DigitLayout<W>, Ring<W>   -- the digit layout and its arithmetic
//                                 (Sec. 13.11.2);
//  * LadicMat<W>               -- a matrix as L digit planes;
//  * smith_lift                -- (L1) realised level by level
//                                 (Sec. 13.11.3 - 13.11.4);
//  * howell, kernel_module, image_module, solve_multi_rhs_module, length,
//    invariant_factors, cyclic_decomposition, cyclic_coordinates -- the
//    Sec. 13.6 entry points DERIVED from the Smith data by the
//    canonicalisation pass (Sec. 13.11.5); entrywise equal to the direct
//    route wherever both apply (Sec. 13.11.7, the oracle rule);
//  * stopping_rule             -- [L] Sec. 2.4 on smith_lift's output.
//
// Every value is exact; no step has a tolerance.  Every intermediate bound
// is stated where it is used.  "No bignum" ([C] obl:backend(iv), Sec.
// 13.11.8): L is fixed per call from (ell, K, W), never grown; every
// primitive is a word operation or ONE double-word digit product.

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DenseBuf.h"
#include "DenseElim.h"
#include "DenseView.h"
#include "DigitWord.h"
#include "EngineCheck.h"
#include "ModCtx.h"

namespace engine {
namespace ladic {

// ---------------------------------------------------------------------------
// The width seam (Sec. 13.11.6): WP-18a's engine::DigitWord<W> (DigitWord.h)
// supplies the digit word's product with carry split (mul_wide, split,
// mac_split) and the word context; Word<W> adds the level context of the
// word-native pivot search and, for the two widths with a NATIVE double word
// (uint64_t for u32 digits, unsigned __int128 for u64 digits), the lazy-carry
// digit convolution.  The u128 width has no native double word: its products
// go through DigitWord<u128>::mul_wide / split (schoolbook, one split per digit
// product), its level context is ModCtx<u128>::prime_power (ell odd; a
// one-digit window k = 1 runs over Z/ell^2 and accepts only valuation-0 pivots).
// ---------------------------------------------------------------------------
template<class W> struct Word;

template<> struct Word<uint32_t> : DigitWord<uint32_t> {
    static constexpr bool native_wide = true;
    using NWide = uint64_t;
    static NWide mul(uint32_t a, uint32_t b) { return static_cast<NWide>(a) * b; }
    static constexpr NWide wide_max = ~NWide(0);
    using LevelCtx = ModCtx<uint32_t>;
    static uint32_t level_word_exp(uint32_t k) { return k; }
    static LevelCtx level_ctx(uint64_t ell, uint32_t k) {
        return k >= 2 ? LevelCtx::prime_power(static_cast<uint32_t>(ell), k) : LevelCtx(static_cast<uint32_t>(ell));
    }
};
template<> struct Word<uint64_t> : DigitWord<uint64_t> {
    static constexpr bool native_wide = true;
    using NWide = unsigned __int128;
    static NWide mul(uint64_t a, uint64_t b) { return static_cast<NWide>(a) * b; }
    static constexpr NWide wide_max = ~NWide(0);
    using LevelCtx = ModCtx<uint64_t>;
    static uint32_t level_word_exp(uint32_t k) { return k; }
    static LevelCtx level_ctx(uint64_t ell, uint32_t k) {
        return k >= 2 ? LevelCtx::prime_power(ell, k) : LevelCtx(ell);
    }
};
template<> struct Word<u128> : DigitWord<u128> {
    static constexpr bool native_wide = false;
    using NWide = u128;   // unused: no native double word
    using LevelCtx = ModCtx<u128>;
    static uint32_t level_word_exp(uint32_t k) { return k < 2 ? 2 : k; }
    static LevelCtx level_ctx(uint64_t ell, uint32_t k) { return LevelCtx::prime_power(ell, k); }
};

template<class W> constexpr bool is_digit_word_v =
    std::is_same<W, uint32_t>::value || std::is_same<W, uint64_t>::value || std::is_same<W, u128>::value;

// ---------------------------------------------------------------------------
// DigitLayout (Sec. 13.11.2)
// ---------------------------------------------------------------------------
template<class W> struct DigitLayout {
    uint64_t ell = 0;
    uint32_t K = 0, m = 0, L = 0, top_exp = 0;
    W B = 0, top_mod = 0;          // B = ell^m; the top digit is < ell^top_exp = top_mod

    // The largest m >= 1 with ell^m < 2^(w-1) (the Montgomery bound of the width).
    static uint32_t max_digit_exp(uint64_t ell) {
        ENGINE_CHECK_MSG(ell >= 2 && ell < (uint64_t(1) << 31),
                         "ladic::DigitLayout: ell must satisfy 2 <= ell < 2^31 (ENGINE_API.md Sec. 13.11.1)");
        const unsigned w = Word<W>::bits;
        const u128 lim = u128(1) << (w - 1);         // 2^(w-1) <= 2^127
        u128 p = 1;
        uint32_t m = 0;
        while (p <= (lim - 1) / ell) { p *= ell; ++m; }   // p * ell < 2^(w-1)
        ENGINE_CHECK(m >= 1);
        return m;
    }
    // m_override = 0: the word-maximal m(W, ell).  A smaller m' is TEST-ONLY
    // (Sec. 13.11.2): it changes the layout, never a value.
    static DigitLayout make(uint64_t ell, uint32_t K, uint32_t m_override = 0) {
        ENGINE_CHECK_MSG(K >= 2, "ladic::DigitLayout: K must be >= 2 (F_ell is ModCtx(ell), ENGINE_API.md Sec. 13.1)");
        if constexpr (std::is_same<W, u128>::value)
            ENGINE_CHECK_MSG(ell % 2 == 1, "ladic::DigitLayout<u128>: ell must be odd (ModCtx<u128> is U128_Montgomery, ENGINE_API.md Sec. 13.12.2)");
        DigitLayout d;
        d.ell = ell;
        d.K = K;
        const uint32_t mmax = max_digit_exp(ell);
        ENGINE_CHECK_MSG(m_override <= mmax, "ladic::DigitLayout: digit exponent override exceeds m(W, ell)");
        d.m = m_override ? m_override : mmax;
        d.L = (K + d.m - 1) / d.m;
        d.top_exp = K - d.m * (d.L - 1);
        W b = 1;
        for (uint32_t i = 0; i < d.m; ++i) b = static_cast<W>(b * ell);
        d.B = b;
        W t = 1;
        for (uint32_t i = 0; i < d.top_exp; ++i) t = static_cast<W>(t * ell);
        d.top_mod = t;
        return d;
    }
    bool operator==(const DigitLayout& o) const { return ell == o.ell && K == o.K && m == o.m; }
    bool operator!=(const DigitLayout& o) const { return !(*this == o); }
};

// Base-ell digits (K of them, little-endian, each < ell) <-> the L word
// digits of the layout: the width-independent serialisation (Sec. 13.11.2).
template<class W> void digits_from_ell(const DigitLayout<W>& lay, W* z, const uint32_t* a) {
    for (uint32_t d = 0; d < lay.L; ++d) {
        W w = 0;
        const uint32_t e0 = d * lay.m, e1 = std::min(lay.K, e0 + lay.m);
        for (uint32_t e = e1; e-- > e0;) {
            ENGINE_CHECK_MSG(a[e] < lay.ell, "ladic: base-ell digit >= ell");
            w = static_cast<W>(w * lay.ell + a[e]);
        }
        z[d] = w;
    }
}
template<class W> void digits_to_ell(const DigitLayout<W>& lay, uint32_t* a, const W* x) {
    for (uint32_t d = 0; d < lay.L; ++d) {
        W w = x[d];
        const uint32_t e0 = d * lay.m, e1 = std::min(lay.K, e0 + lay.m);
        for (uint32_t e = e0; e < e1; ++e) { a[e] = static_cast<uint32_t>(w % lay.ell); w = static_cast<W>(w / lay.ell); }
    }
}

// ---------------------------------------------------------------------------
// Ring<W>: Z/ell^K on L digits (Sec. 13.11.2).  Operands are W* of length L,
// canonical (digit i < B, top digit < top_mod).  Outputs may alias inputs.
// NOT thread-safe (scratch buffers): one Ring per call, single-threaded by
// design (Sec. 13.11 status).
// ---------------------------------------------------------------------------
template<class W> class Ring {
public:
    using Wide = typename Word<W>::NWide;

    explicit Ring(const DigitLayout<W>& lay) : lay_(lay), L(lay.L), m(lay.m), K(lay.K), B(lay.B), top(lay.top_mod),
                                               ell(lay.ell) {
        pw_.resize(m + 1);
        pw_[0] = 1;
        for (uint32_t e = 1; e <= m; ++e) pw_[e] = static_cast<W>(pw_[e - 1] * ell);
        ENGINE_CHECK(pw_[m] == B);
        // Lazy-carry bound (mul_acc / normalise).  Either no column can ever
        // reach half the double word (L B^2 <= max/4: no fold happens at
        // all), or each fold moves at most max/B into the next column, at most
        // L/2 + 1 folds per column (every product is < max/4, so two products
        // separate two folds), which stays <= max/4 when B >= 2L + 4.  Then an
        // accumulator never exceeds 3/4 max and acc + carry < max.  Checked,
        // not assumed (it holds for every layout of Sec. 13.11.2 with L < 2^14).
        if constexpr (Word<W>::native_wide) {
            const Wide mx = Word<W>::wide_max;
            const Wide b = static_cast<Wide>(B);
            const bool no_fold = static_cast<Wide>(L) <= (mx / 4) / (b * b);
            const bool small_fold = b >= 2 * static_cast<Wide>(L) + 4;
            ENGINE_CHECK_MSG(no_fold || small_fold, "ladic::Ring: too many digits for the lazy-carry bound");
        }
        acc_.assign(L + 1, Wide(0));
        tm_.assign(L, 0);
        t1_.assign(L, 0);
        t2_.assign(L, 0);
        t3_.assign(L, 0);
    }
    const DigitLayout<W>& layout() const { return lay_; }
    W pw(uint32_t e) const { return pw_[e]; }

    // --- basics ----------------------------------------------------------
    void zero(W* z) const { std::fill(z, z + L, W(0)); }
    void one(W* z) const { zero(z); z[0] = 1; }
    bool is_zero(const W* x) const {
        for (uint32_t i = 0; i < L; ++i) if (x[i]) return false;
        return true;
    }
    bool is_one(const W* x) const {
        if (x[0] != 1) return false;
        for (uint32_t i = 1; i < L; ++i) if (x[i]) return false;
        return true;
    }
    bool eq(const W* a, const W* b) const { return std::equal(a, a + L, b); }
    void copy(W* z, const W* x) const { if (z != x) std::copy(x, x + L, z); }

    // Digits are < B < 2^(w-1), so x + y + 1 < 2^w: W arithmetic suffices.
    void add(W* z, const W* x, const W* y) const {
        W carry = 0;
        for (uint32_t i = 0; i + 1 < L; ++i) {
            const W t = static_cast<W>(x[i] + y[i] + carry);
            if (t >= B) { z[i] = static_cast<W>(t - B); carry = 1; }
            else        { z[i] = t;                     carry = 0; }
        }
        const W t = static_cast<W>(x[L - 1] + y[L - 1] + carry);   // < 2 top + 1
        z[L - 1] = static_cast<W>(t % top);
    }
    void sub(W* z, const W* x, const W* y) const {
        W borrow = 0;
        for (uint32_t i = 0; i + 1 < L; ++i) {
            const W need = static_cast<W>(y[i] + borrow);   // <= B
            if (x[i] >= need) { z[i] = static_cast<W>(x[i] - need); borrow = 0; }
            else              { z[i] = static_cast<W>(x[i] + (B - need)); borrow = 1; }
        }
        const W need = static_cast<W>(y[L - 1] + borrow);   // <= top
        z[L - 1] = (x[L - 1] >= need) ? static_cast<W>(x[L - 1] - need)
                                      : static_cast<W>(x[L - 1] + (top - need));
    }
    void neg(W* z, const W* x) const {
        zero(t3_.data());
        sub(z, t3_.data(), x);
    }

    // --- products (lazy-carry digit convolution) --------------------------
    // acc_[c] += x_i y_j over i + j = c < L.  Before each add, a column at or
    // above half the double word is FOLDED (one division by B) into the next
    // column; each product is < B^2 < 2^(2w-2) = wide_max / 4, so an
    // accumulator never exceeds 3/4 of the double word.
    void mul_acc(const W* x, const W* y) const {
        const Wide half = Word<W>::wide_max / 2;
        for (uint32_t c = 0; c < L; ++c) {
            for (uint32_t i = 0; i <= c; ++i) {
                const W a = x[i], b = y[c - i];
                if (a == 0 || b == 0) continue;
                if (acc_[c] >= half) fold(c);
                acc_[c] += Word<W>::mul(a, b);
            }
        }
    }
    void fold(uint32_t c) const {
        const Wide q = acc_[c] / B;
        acc_[c] -= q * B;
        if (c + 1 < L) acc_[c + 1] += q;
    }
    void clear_acc() const { std::fill(acc_.begin(), acc_.end(), Wide(0)); }
    // Normalise acc_ into z (mod ell^K).  t = acc + carry < 3/4 max + max/B < max.
    void normalise(W* z) const {
        Wide carry = 0;
        for (uint32_t c = 0; c < L; ++c) {
            const Wide t = acc_[c] + carry;
            carry = t / B;
            z[c] = static_cast<W>(t - carry * B);
        }
        z[L - 1] = static_cast<W>(z[L - 1] % top);
    }
    void mul(W* z, const W* x, const W* y) const {
        if constexpr (Word<W>::native_wide) {
            clear_acc();
            mul_acc(x, y);
            normalise(z);
        } else {
            // Schoolbook with one carry split per digit product (no native
            // double word): x_i y_j + t + carry <= (B-1)^2 + 2(B-1) < B^2, so
            // the high word is < B (DigitWord::split's precondition).
            W* t = tm_.data();
            std::fill(t, t + L, W(0));
            for (uint32_t i = 0; i < L; ++i) {
                if (x[i] == 0) continue;
                W carry = 0;
                for (uint32_t j = 0; i + j < L; ++j) {
                    W hi, lo;
                    Word<W>::mul_wide(x[i], y[j], hi, lo);
                    W lo2 = static_cast<W>(lo + t[i + j]);
                    if (lo2 < lo) ++hi;
                    const W lo3 = static_cast<W>(lo2 + carry);
                    if (lo3 < lo2) ++hi;
                    W r;
                    carry = Word<W>::split(hi, lo3, B, r);
                    t[i + j] = r;
                }
            }
            t[L - 1] = static_cast<W>(t[L - 1] % top);
            std::copy(t, t + L, z);
        }
    }
    // z -= q * y ; z += q * y
    void submul(W* z, const W* q, const W* y) const {
        mul(t1_.data(), q, y);
        sub(z, z, t1_.data());
    }
    void addmul(W* z, const W* q, const W* y) const {
        mul(t1_.data(), q, y);
        add(z, z, t1_.data());
    }

    // --- the ell-adic structure -------------------------------------------
    uint32_t valuation(const W* x) const {
        for (uint32_t i = 0; i < L; ++i) {
            W w = x[i];
            if (w == 0) continue;
            uint32_t v = 0;
            while (w % ell == 0) { w = static_cast<W>(w / ell); ++v; }
            return i * m + v;
        }
        return K;
    }
    // floor(x / ell^v) as an integer (exact division when ell^v | x).
    void shr(W* z, const W* x, uint32_t v) const {
        if (v >= K) { zero(z); return; }
        const uint32_t q = v / m, r = v % m;
        for (uint32_t i = 0; i < L; ++i) {
            const uint32_t src = i + q;
            const W lo = (src < L) ? x[src] : W(0);
            if (r == 0) { z[i] = lo; continue; }
            const W hi = (src + 1 < L) ? x[src + 1] : W(0);
            z[i] = static_cast<W>(lo / pw_[r] + (hi % pw_[r]) * pw_[m - r]);   // < ell^(m-r) + (ell^r - 1) ell^(m-r) = B
        }
    }
    // x mod ell^v
    void low(W* z, const W* x, uint32_t v) const {
        copy(z, x);
        if (v >= K) return;
        const uint32_t q = v / m, r = v % m;
        z[q] = static_cast<W>(z[q] % pw_[r]);
        for (uint32_t i = q + 1; i < L; ++i) z[i] = 0;
    }
    // x * ell^v mod ell^K
    void shl(W* z, const W* x, uint32_t v) const {
        if (v >= K) { zero(z); return; }
        const uint32_t q = v / m, r = v % m;
        W* y = t2_.data();
        W carry = 0;
        for (uint32_t i = 0; i < L; ++i) {
            W rr;
            carry = Word<W>::mac_split(x[i], pw_[r], carry, B, rr);   // x_i, ell^r, carry < B
            y[i] = rr;
        }
        for (uint32_t i = L; i-- > 0;) z[i] = (i >= q) ? y[i - q] : W(0);
        z[L - 1] = static_cast<W>(z[L - 1] % top);
    }
    void unit_part(W* z, const W* x) const {
        const uint32_t v = valuation(x);
        if (v >= K) { one(z); return; }
        shr(z, x, v);
    }
    // The inverse of a unit (Newton from the inverse mod ell); a non-unit
    // throws the Sec. 13.4 runtime_error.
    void inv(W* z, const W* u) const {
        const uint64_t u0 = static_cast<uint64_t>(u[0] % ell);
        if (u0 == 0) throw std::runtime_error("Modular inverse does not exist.");
        W* y = t3_.data();
        zero(y);
        y[0] = static_cast<W>(inv_mod_small(u0, ell));
        W* t = t2_.data();
        std::vector<W> two(L, W(0));
        from_u64(two.data(), 2);
        for (uint32_t prec = 1; prec < K; prec *= 2) {
            mul(t, u, y);           // u y = 1 + e, e = 0 mod ell^prec
            sub(t, two.data(), t);  // 2 - u y
            mul(y, y, t);           // 1 - u y' = (1 - u y)^2
        }
        copy(z, y);
        ENGINE_DEBUG_CHECK((mul(t, u, z), is_one(t)));
    }
    static uint64_t inv_mod_small(uint64_t a, uint64_t p) {   // a unit mod p, p < 2^31
        int64_t t = 0, nt = 1, r = static_cast<int64_t>(p), nr = static_cast<int64_t>(a % p);
        while (nr != 0) {
            const int64_t q = r / nr;
            int64_t tmp = t - q * nt; t = nt; nt = tmp;
            tmp = r - q * nr; r = nr; nr = tmp;
        }
        ENGINE_CHECK(r == 1);
        if (t < 0) t += static_cast<int64_t>(p);
        return static_cast<uint64_t>(t);
    }

    // --- conversions --------------------------------------------------------
    void from_u64(W* z, uint64_t v) const {
        for (uint32_t i = 0; i < L; ++i) {
            z[i] = static_cast<W>(v % B);
            v /= B;
        }
        z[L - 1] = static_cast<W>(z[L - 1] % top);
    }
    // K base-ell digits, little-endian, each < ell  <->  L word digits.
    void from_ell_digits(W* z, const uint32_t* a) const { digits_from_ell(lay_, z, a); }
    void to_ell_digits(uint32_t* a, const W* x) const { digits_to_ell(lay_, a, x); }
    // floor(x / ell^c) mod ell^k, k <= m: one word (Sec. 13.11.4 (a)).
    W window(const W* x, uint32_t c, uint32_t k) const {
        ENGINE_CHECK_MSG(k >= 1 && k <= m, "ladic::Ring::window: the window must fit one word (k <= m)");
        if (c >= K) return 0;
        const uint32_t q = c / m, r = c % m;
        W v = x[q];
        if (r) {
            const W hi = (q + 1 < L) ? x[q + 1] : W(0);
            v = static_cast<W>(v / pw_[r] + (hi % pw_[r]) * pw_[m - r]);
        }
        return static_cast<W>(v % pw_[k]);
    }

    const DigitLayout<W> lay_;
    const uint32_t L, m, K;
    const W B, top;
    const uint64_t ell;

private:
    std::vector<W> pw_;
    mutable std::vector<Wide> acc_;
    mutable std::vector<W> tm_, t1_, t2_, t3_;
};

// ---------------------------------------------------------------------------
// LadicMat<W>: a rows x cols matrix over Z/ell^K as L digit planes.
// ---------------------------------------------------------------------------
template<class W> class LadicMat {
public:
    LadicMat() = default;
    LadicMat(const DigitLayout<W>& lay, uint32_t rows, uint32_t cols)
        : lay_(lay), rows_(rows), cols_(cols), d_(static_cast<size_t>(lay.L) * rows * cols, W(0)) {}

    const DigitLayout<W>& layout() const { return lay_; }
    uint32_t rows() const { return rows_; }
    uint32_t cols() const { return cols_; }
    uint32_t K() const { return lay_.K; }
    uint64_t ell() const { return lay_.ell; }
    // plane d: the rows x cols row-major matrix of the d-th digits
    const W* plane(uint32_t d) const { return d_.data() + static_cast<size_t>(d) * rows_ * cols_; }
    W* plane(uint32_t d) { return d_.data() + static_cast<size_t>(d) * rows_ * cols_; }

    void get(uint32_t i, uint32_t j, W* out) const {
        ENGINE_DEBUG_CHECK(i < rows_ && j < cols_);
        const size_t o = static_cast<size_t>(i) * cols_ + j, ps = static_cast<size_t>(rows_) * cols_;
        for (uint32_t d = 0; d < lay_.L; ++d) out[d] = d_[d * ps + o];
    }
    void set(uint32_t i, uint32_t j, const W* in) {
        ENGINE_DEBUG_CHECK(i < rows_ && j < cols_);
        const size_t o = static_cast<size_t>(i) * cols_ + j, ps = static_cast<size_t>(rows_) * cols_;
        for (uint32_t d = 0; d < lay_.L; ++d) d_[d * ps + o] = in[d];
    }
    // The width-independent serialisation (Sec. 13.11.2): K base-ell digits.
    std::vector<uint32_t> ell_digits(uint32_t i, uint32_t j) const {
        std::vector<W> t(lay_.L);
        get(i, j, t.data());
        std::vector<uint32_t> a(lay_.K);
        digits_to_ell(lay_, a.data(), t.data());
        return a;
    }
    void set_ell_digits(uint32_t i, uint32_t j, const std::vector<uint32_t>& a) {
        ENGINE_CHECK_MSG(a.size() == lay_.K, "ladic::LadicMat::set_ell_digits: need exactly K digits");
        std::vector<W> t(lay_.L);
        digits_from_ell(lay_, t.data(), a.data());
        set(i, j, t.data());
    }
    // All entries' serialisations, row-major (the comparison key of Sec. 13.11.7).
    std::vector<uint32_t> serialise() const {
        std::vector<uint32_t> out;
        out.reserve(static_cast<size_t>(rows_) * cols_ * lay_.K + 2);
        out.push_back(rows_);
        out.push_back(cols_);
        for (uint32_t i = 0; i < rows_; ++i)
            for (uint32_t j = 0; j < cols_; ++j) {
                const std::vector<uint32_t> a = ell_digits(i, j);
                out.insert(out.end(), a.begin(), a.end());
            }
        return out;
    }

    // ell^K must fit N (checked); the entries of A must be canonical (< ell^K).
    template<class N> static LadicMat from_word(const DigitLayout<W>& lay, DenseView<const N> A) {
        const u128 mod = modulus_checked<N>(lay);
        LadicMat M(lay, A.rows(), A.cols());
        std::vector<W> t(lay.L);
        for (uint32_t i = 0; i < A.rows(); ++i)
            for (uint32_t j = 0; j < A.cols(); ++j) {
                u128 v = A.at(i, j);
                ENGINE_CHECK_MSG(v < mod, "ladic::LadicMat::from_word: entry not canonical");
                for (uint32_t d = 0; d < lay.L; ++d) { t[d] = static_cast<W>(v % lay.B); v /= lay.B; }
                M.set(i, j, t.data());
            }
        return M;
    }
    template<class N> DenseBuf<N> to_word() const {
        (void)modulus_checked<N>(lay_);
        DenseBuf<N> out(rows_, cols_);
        std::vector<W> t(lay_.L);
        for (uint32_t i = 0; i < rows_; ++i)
            for (uint32_t j = 0; j < cols_; ++j) {
                get(i, j, t.data());
                u128 v = 0;
                for (uint32_t d = lay_.L; d-- > 0;) v = v * lay_.B + t[d];
                out.at(i, j) = static_cast<N>(v);
            }
        return out;
    }
    // ell^K as an integer, checked to fit N (and u128).
    template<class N> static u128 modulus_checked(const DigitLayout<W>& lay) {
        u128 p = 1;
        for (uint32_t e = 0; e < lay.K; ++e) {
            ENGINE_CHECK_MSG(p <= (~u128(0)) / lay.ell, "ladic::LadicMat: ell^K does not fit the native width");
            p *= lay.ell;
        }
        if constexpr (sizeof(N) * 8 < 128) {
            ENGINE_CHECK_MSG(p - 1 <= ((u128(1) << (sizeof(N) * 8)) - 1), "ladic::LadicMat: ell^K does not fit the native width");
        }
        return p;
    }

    // Reduction Z/ell^K -> Z/ell^k (2 <= k <= K): drop planes, reduce the top
    // digit (Sec. 13.11.2).  The digit exponent m is kept.
    LadicMat truncate(uint32_t k) const {
        ENGINE_CHECK_MSG(k >= 2 && k <= lay_.K, "ladic::LadicMat::truncate: need 2 <= k <= K");
        const DigitLayout<W> nl = DigitLayout<W>::make(lay_.ell, k, lay_.m);
        LadicMat out(nl, rows_, cols_);
        const size_t ps = static_cast<size_t>(rows_) * cols_;
        for (uint32_t d = 0; d < nl.L; ++d)
            for (size_t o = 0; o < ps; ++o) {
                W v = d_[d * ps + o];
                if (d + 1 == nl.L) v = static_cast<W>(v % nl.top_mod);
                out.d_[d * ps + o] = v;
            }
        return out;
    }
    LadicMat transpose() const {
        LadicMat out(lay_, cols_, rows_);
        const size_t ps = static_cast<size_t>(rows_) * cols_;
        for (uint32_t d = 0; d < lay_.L; ++d)
            for (uint32_t i = 0; i < rows_; ++i)
                for (uint32_t j = 0; j < cols_; ++j)
                    out.d_[d * ps + static_cast<size_t>(j) * rows_ + i] = d_[d * ps + static_cast<size_t>(i) * cols_ + j];
        return out;
    }
    bool operator==(const LadicMat& o) const {
        return lay_ == o.lay_ && rows_ == o.rows_ && cols_ == o.cols_ && d_ == o.d_;
    }
    bool operator!=(const LadicMat& o) const { return !(*this == o); }

private:
    DigitLayout<W> lay_;
    uint32_t rows_ = 0, cols_ = 0;
    std::vector<W> d_;
};

// Re-express a matrix in another width / digit exponent (same ell, K), through
// the width-independent serialisation.
template<class W2, class W1>
LadicMat<W2> convert(const LadicMat<W1>& A, uint32_t m_override = 0) {
    const DigitLayout<W2> lay = DigitLayout<W2>::make(A.ell(), A.K(), m_override);
    LadicMat<W2> out(lay, A.rows(), A.cols());
    for (uint32_t i = 0; i < A.rows(); ++i)
        for (uint32_t j = 0; j < A.cols(); ++j) out.set_ell_digits(i, j, A.ell_digits(i, j));
    return out;
}

// ---------------------------------------------------------------------------
// Results and options (Sec. 13.11.6)
// ---------------------------------------------------------------------------
struct Options {
    bool transforms = false;        // smith_lift: also U, V with U A V = diag(ell^s) (+) 0
    bool force_induction = false;   // TEST-ONLY: L = 1 runs the induction, not the Sec. 13.5 dispatch
    uint32_t level_exp = 0;         // TEST-ONLY: level window k_l <= level_exp (0 = m); no value changes
};

template<class W> struct Smith {
    uint32_t rows = 0, cols = 0;
    std::vector<uint32_t> s;                     // exponents s_t < K, non-decreasing, pivot order
    std::vector<uint32_t> pivot_row, pivot_col;
    std::vector<uint32_t> level_c, level_k, level_rank;   // the windows [c, c + k) and their pivot counts
    LadicMat<W> T;                               // r x cols: the pivot rows of U A, in pivot order
    bool has_transforms = false;
    LadicMat<W> U, V;                            // U A V = diag(ell^{s_t}) (+) 0, exactly
    uint32_t rank_K() const { return static_cast<uint32_t>(s.size()); }
};

template<class W> struct Howell {
    LadicMat<W> H;                               // h x n
    std::vector<uint32_t> pivots, pivot_v;
    uint32_t rows() const { return static_cast<uint32_t>(pivots.size()); }
};

template<class W> struct Cyclic {
    std::vector<uint32_t> exponents;             // (c4), descending
    LadicMat<W> basis;                           // m x t
    LadicMat<W> Gamma;                           // m x s   (c1)
    LadicMat<W> U;                               // s x s   (c2)-(c3)
    std::vector<uint32_t> position;
    uint32_t rows() const { return static_cast<uint32_t>(exponents.size()); }
};

struct StoppingRule {
    bool certified = false;
    uint32_t r0 = 0, r1 = 0, need0 = 0, need1 = 0;
};

namespace detail {

// An entry-major working matrix: entry (i, j) is L contiguous digits.
template<class W> struct Work {
    uint32_t rows = 0, cols = 0, L = 0;
    std::vector<W> d;
    Work() = default;
    Work(uint32_t r, uint32_t c, uint32_t l) : rows(r), cols(c), L(l), d(static_cast<size_t>(r) * c * l, W(0)) {}
    W* at(uint32_t i, uint32_t j) { return d.data() + (static_cast<size_t>(i) * cols + j) * L; }
    const W* at(uint32_t i, uint32_t j) const { return d.data() + (static_cast<size_t>(i) * cols + j) * L; }
    W* row(uint32_t i) { return at(i, 0); }
    const W* row(uint32_t i) const { return at(i, 0); }
};

template<class W> Work<W> to_work(const LadicMat<W>& A) {
    Work<W> w(A.rows(), A.cols(), A.layout().L);
    for (uint32_t i = 0; i < A.rows(); ++i)
        for (uint32_t j = 0; j < A.cols(); ++j) A.get(i, j, w.at(i, j));
    return w;
}
template<class W> LadicMat<W> from_work(const DigitLayout<W>& lay, const Work<W>& w) {
    LadicMat<W> A(lay, w.rows, w.cols);
    for (uint32_t i = 0; i < w.rows; ++i)
        for (uint32_t j = 0; j < w.cols; ++j) A.set(i, j, w.at(i, j));
    return A;
}
template<class W> void check_same_layout(const LadicMat<W>& a, const LadicMat<W>& b, const char* who) {
    ENGINE_CHECK_MSG(a.layout() == b.layout(), who);
}

// The level driver (Sec. 13.11.4) on a working matrix R (m x w) whose first
// `npiv` columns are pivot-eligible (the rest ride along: solve's Y).
template<class W> struct DriverOut {
    std::vector<uint32_t> s, prow, pcol;
    std::vector<uint32_t> level_c, level_k, level_rank;
    std::vector<char> row_is_pivot;
};

template<class W>
DriverOut<W> drive(const Ring<W>& ring, Work<W>& R, uint32_t npiv, Work<W>* U, const Options& opt) {
    using LCtx = typename Word<W>::LevelCtx;
    using LE = decltype(std::declval<LCtx>().p);
    const uint32_t K = ring.K, L = ring.L, m = R.rows, w = R.cols;
    ENGINE_CHECK(npiv <= w);
    DriverOut<W> out;
    std::vector<char> row_active(m, 1), col_active(npiv, 1);
    std::vector<W> q(L), uinv(L), up(L);
    std::vector<uint32_t> pnz;
    uint32_t c = 0;
    while (c < K) {
        uint32_t k = std::min(ring.m, K - c);
        if (opt.level_exp) k = std::min(k, opt.level_exp);
        std::vector<uint32_t> ar, ac;
        for (uint32_t i = 0; i < m; ++i) if (row_active[i]) ar.push_back(i);
        for (uint32_t j = 0; j < npiv; ++j) if (col_active[j]) ac.push_back(j);
        if (ar.empty() || ac.empty()) break;
        const uint32_t a = static_cast<uint32_t>(ar.size()), b = static_cast<uint32_t>(ac.size());
        // (a) the word-native problem: the window, eliminated over R_k.
        const uint32_t kw = Word<W>::level_word_exp(k);   // the level context's exponent (>= k)
        const LCtx ctx = Word<W>::level_ctx(ring.ell, kw);
        std::vector<LE> Wm(static_cast<size_t>(a) * b);
        bool any = false;
        for (uint32_t i = 0; i < a; ++i)
            for (uint32_t j = 0; j < b; ++j) {
                // the window mod ell^k (k <= m); the level context may be Z/ell^kw, kw >= k:
                // pivot decisions read valuations < k only, which are fixed mod ell^k
                const LE v = static_cast<LE>(ring.window(R.at(ar[i], ac[j]), c, k));
                Wm[static_cast<size_t>(i) * b + j] = v;
                any = any || v != 0;
            }
        uint32_t level_rank = 0;
        if (any) {
            std::vector<char> ra(a, 1), ca(b, 1);
            std::vector<std::pair<uint32_t, uint32_t>> piv;
            std::vector<uint32_t> sig;
            uint32_t v_floor = 0;
            for (;;) {
                // Smallest valuation, then leftmost column, then topmost row.
                // The minimal valuation never decreases under the row updates
                // (every update adds q * P_j with v(q P_j) >= the pivot's), so
                // the scan stops at the first entry of valuation v_floor: no
                // later entry can beat it, and it is the leftmost-topmost one.
                uint32_t pr = a, pc = b, pv = kw;
                for (uint32_t j = 0; j < b && pv > v_floor; ++j) {   // leftmost column first ...
                    if (!ca[j]) continue;
                    for (uint32_t i = 0; i < a; ++i) {     // ... topmost row; strict < keeps the first
                        if (!ra[i]) continue;
                        const LE e = Wm[static_cast<size_t>(i) * b + j];
                        if (e == 0) continue;
                        const uint32_t v = ctx.valuation(e);
                        if (v < pv) { pv = v; pr = i; pc = j; if (v == v_floor) break; }
                    }
                }
                ENGINE_DEBUG_CHECK(pr == a || pv >= v_floor);
                if (pr != a) v_floor = pv;
                if (pr == a || pv >= k) break;   // only valuations < k belong to this window
                piv.emplace_back(pr, pc);
                sig.push_back(pv);
                const LE d = ctx.ell_pow(pv);
                const LE ui = ctx.inv(ctx.unit_part(Wm[static_cast<size_t>(pr) * b + pc]));
                const LE* P = Wm.data() + static_cast<size_t>(pr) * b;
                for (uint32_t i = 0; i < a; ++i) {
                    if (!ra[i] || i == pr) continue;
                    LE* Ri = Wm.data() + static_cast<size_t>(i) * b;
                    if (Ri[pc] == 0) continue;
                    const LE qq = ctx.mul(ctx.divide_exact(Ri[pc], d), ui);
                    for (uint32_t j = 0; j < b; ++j)
                        if (ca[j] && P[j] != 0) Ri[j] = ctx.sub(Ri[j], ctx.mul(qq, P[j]));
                    ENGINE_DEBUG_CHECK(Ri[pc] == 0);
                }
                ra[pr] = 0;
                ca[pc] = 0;
            }
            // (b) the replay at precision K.
            for (size_t t = 0; t < piv.size(); ++t) {
                const uint32_t gi = ar[piv[t].first], gj = ac[piv[t].second], s = c + sig[t];
                W* P = R.row(gi);
                ENGINE_CHECK_MSG(ring.valuation(P + static_cast<size_t>(gj) * L) == s,
                                 "ladic::smith_lift: replayed pivot valuation differs from the word level (ENGINE_API.md Sec. 13.11.4 (b))");
                ring.unit_part(up.data(), P + static_cast<size_t>(gj) * L);
                if (!ring.is_one(up.data())) {
                    ring.inv(uinv.data(), up.data());
                    for (uint32_t j = 0; j < w; ++j) {
                        W* e = P + static_cast<size_t>(j) * L;
                        if (!ring.is_zero(e)) ring.mul(e, e, uinv.data());
                    }
                    if (U) {
                        W* Ur = U->row(gi);
                        for (uint32_t j = 0; j < m; ++j) {
                            W* e = Ur + static_cast<size_t>(j) * L;
                            if (!ring.is_zero(e)) ring.mul(e, e, uinv.data());
                        }
                    }
                }
                pnz.clear();
                for (uint32_t j = 0; j < w; ++j)
                    if (!ring.is_zero(P + static_cast<size_t>(j) * L)) pnz.push_back(j);
                for (uint32_t i = 0; i < m; ++i) {
                    if (!row_active[i] || i == gi) continue;
                    W* Ri = R.row(i);
                    W* x = Ri + static_cast<size_t>(gj) * L;
                    if (ring.is_zero(x)) continue;
                    ENGINE_CHECK_MSG(ring.valuation(x) >= s,
                                     "ladic::smith_lift: an entry below the pivot has smaller valuation (ENGINE_API.md Sec. 13.11.4 (b))");
                    ring.shr(q.data(), x, s);
                    for (uint32_t j : pnz)
                        ring.submul(Ri + static_cast<size_t>(j) * L, q.data(), P + static_cast<size_t>(j) * L);
                    ENGINE_CHECK(ring.is_zero(x));
                    if (U) {
                        W* Ui = U->row(i);
                        const W* Up = U->row(gi);
                        for (uint32_t j = 0; j < m; ++j) {
                            const W* pe = Up + static_cast<size_t>(j) * L;
                            if (!ring.is_zero(pe)) ring.submul(Ui + static_cast<size_t>(j) * L, q.data(), pe);
                        }
                    }
                }
                row_active[gi] = 0;
                col_active[gj] = 0;
                out.s.push_back(s);
                out.prow.push_back(gi);
                out.pcol.push_back(gj);
                ++level_rank;
            }
        }
        // (c) the level's exit check: the active block is divisible by ell^{c+k}.
        bool active_nonzero = false;
        for (uint32_t i = 0; i < m; ++i) {
            if (!row_active[i]) continue;
            for (uint32_t j = 0; j < npiv; ++j) {
                if (!col_active[j]) continue;
                const uint32_t v = ring.valuation(R.at(i, j));
                ENGINE_CHECK_MSG(v >= c + k,
                                 "ladic::smith_lift: level exit check failed (ENGINE_API.md Sec. 13.11.4 (c))");
                active_nonzero = active_nonzero || v < K;
            }
        }
        out.level_c.push_back(c);
        out.level_k.push_back(k);
        out.level_rank.push_back(level_rank);
        c += k;
        if (!active_nonzero) break;   // the active block is zero: no divisor is left
    }
    out.row_is_pivot.assign(m, 0);
    for (uint32_t i : out.prow) out.row_is_pivot[i] = 1;
    for (size_t t = 1; t < out.s.size(); ++t) ENGINE_CHECK(out.s[t - 1] <= out.s[t]);
    return out;
}

// M' = T with row t divided exactly by ell^{s_t}; back-substitution on its
// unit upper-triangular pivot part (pivot order): x_J = M'_J^{-1} b.
template<class W> struct Echelon {
    uint32_t r = 0, n = 0;
    std::vector<uint32_t> s, pcol;
    Work<W> Mp;                     // r x n
    std::vector<char> is_pcol;
    std::vector<std::vector<uint32_t>> nz;   // row t: the columns j != pcol[t] with M'[t][j] != 0 (sparsity only)
};
template<class W>
Echelon<W> echelon_of(const Ring<W>& ring, const Work<W>& R, const DriverOut<W>& d, uint32_t n) {
    Echelon<W> e;
    e.r = static_cast<uint32_t>(d.s.size());
    e.n = n;
    e.s = d.s;
    e.pcol = d.pcol;
    e.Mp = Work<W>(e.r, n, ring.L);
    e.is_pcol.assign(n, 0);
    for (uint32_t t = 0; t < e.r; ++t) {
        e.is_pcol[d.pcol[t]] = 1;
        for (uint32_t j = 0; j < n; ++j) {
            const W* x = R.at(d.prow[t], j);
            ENGINE_DEBUG_CHECK(ring.valuation(x) >= d.s[t]);
            ring.shr(e.Mp.at(t, j), x, d.s[t]);
        }
        ENGINE_DEBUG_CHECK(ring.is_one(e.Mp.at(t, d.pcol[t])));
    }
    e.nz.assign(e.r, {});
    for (uint32_t t = 0; t < e.r; ++t)
        for (uint32_t j = 0; j < n; ++j)
            if (j != d.pcol[t] && !ring.is_zero(e.Mp.at(t, j))) e.nz[t].push_back(j);
    return e;
}
// b (r digits-vectors, pivot order) -> x (n entries): x_F = xF given, x_J solved
// from M'_J x_J = b - M'_F x_F.
template<class W>
void back_substitute(const Ring<W>& ring, const Echelon<W>& e, std::vector<W>& b, std::vector<W>& x) {
    const uint32_t L = ring.L;
    for (uint32_t t = e.r; t-- > 0;) {
        W* bt = b.data() + static_cast<size_t>(t) * L;
        for (uint32_t j : e.nz[t]) {
            const W* xj = x.data() + static_cast<size_t>(j) * L;
            if (!ring.is_zero(xj)) ring.submul(bt, e.Mp.at(t, j), xj);
        }
        ring.copy(x.data() + static_cast<size_t>(e.pcol[t]) * L, bt);
    }
}
// Generators of the right kernel (Sec. 13.11.5), as rows of an entry-major store.
template<class W>
std::vector<W> kernel_generators(const Ring<W>& ring, const Echelon<W>& e, uint32_t& count) {
    const uint32_t L = ring.L, n = e.n, r = e.r;
    std::vector<W> store;
    count = 0;
    std::vector<W> b(static_cast<size_t>(r) * L), x(static_cast<size_t>(n) * L);
    for (uint32_t f = 0; f < n; ++f) {
        if (e.is_pcol[f]) continue;
        std::fill(b.begin(), b.end(), W(0));
        std::fill(x.begin(), x.end(), W(0));
        ring.one(x.data() + static_cast<size_t>(f) * L);
        // solve M' x = 0 with x_f = 1: b = 0, the x_f term enters through back_substitute
        back_substitute(ring, e, b, x);
        store.insert(store.end(), x.begin(), x.end());
        ++count;
    }
    for (uint32_t t = 0; t < r; ++t) {
        if (e.s[t] == 0) continue;
        std::fill(b.begin(), b.end(), W(0));
        std::fill(x.begin(), x.end(), W(0));
        std::vector<W> one(L);
        ring.one(one.data());
        ring.shl(b.data() + static_cast<size_t>(t) * L, one.data(), ring.K - e.s[t]);
        back_substitute(ring, e, b, x);
        store.insert(store.end(), x.begin(), x.end());
        ++count;
    }
    return store;
}

// The Sec. 13.5 algorithm in digit arithmetic (the canonicalisation pass,
// Sec. 13.11.5) on an entry-major row store of m rows x n columns.
template<class W>
Howell<W> howell_rows(const Ring<W>& ring, std::vector<W> store, uint32_t m, uint32_t n) {
    const uint32_t L = ring.L, K = ring.K;
    const size_t rs = static_cast<size_t>(n) * L;
    auto rowp = [&](uint32_t i) { return store.data() + static_cast<size_t>(i) * rs; };
    Howell<W> out;
    std::vector<uint32_t> active(m);
    for (uint32_t i = 0; i < m; ++i) active[i] = i;
    uint32_t n_rows = m;
    std::vector<uint32_t> piv_rows;
    std::vector<W> u(L), ui(L), q(L);
    for (uint32_t c = 0; c < n && !active.empty(); ++c) {
        size_t best = active.size();
        uint32_t best_v = K;
        for (size_t t = 0; t < active.size(); ++t) {
            const W* e = rowp(active[t]) + static_cast<size_t>(c) * L;
            if (ring.is_zero(e)) continue;
            const uint32_t v = ring.valuation(e);
            if (v < best_v) { best_v = v; best = t; if (v == 0) break; }
        }
        if (best == active.size()) continue;
        const uint32_t prow = active[best];
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(best));
        {
            W* P = rowp(prow);
            ring.unit_part(u.data(), P + static_cast<size_t>(c) * L);
            if (!ring.is_one(u.data())) {
                ring.inv(ui.data(), u.data());
                for (uint32_t j = c; j < n; ++j) {
                    W* e = P + static_cast<size_t>(j) * L;
                    if (!ring.is_zero(e)) ring.mul(e, e, ui.data());
                }
            }
        }
        for (uint32_t r : active) {
            W* Rr = rowp(r);
            W* x = Rr + static_cast<size_t>(c) * L;
            if (ring.is_zero(x)) continue;
            ring.shr(q.data(), x, best_v);
            const W* P = rowp(prow);
            for (uint32_t j = c; j < n; ++j) {
                const W* pe = P + static_cast<size_t>(j) * L;
                if (!ring.is_zero(pe)) ring.submul(Rr + static_cast<size_t>(j) * L, q.data(), pe);
            }
            ENGINE_DEBUG_CHECK(ring.is_zero(x));
        }
        if (best_v > 0) {
            store.resize(store.size() + rs, W(0));
            const W* P = rowp(prow);
            W* Cl = rowp(n_rows);
            for (uint32_t j = c + 1; j < n; ++j)
                ring.shl(Cl + static_cast<size_t>(j) * L, P + static_cast<size_t>(j) * L, K - best_v);
            active.push_back(n_rows++);
        }
        piv_rows.push_back(prow);
        out.pivots.push_back(c);
        out.pivot_v.push_back(best_v);
    }
    const uint32_t h = static_cast<uint32_t>(piv_rows.size());
    Work<W> H(h, n, L);
    for (uint32_t i = 0; i < h; ++i) std::copy(rowp(piv_rows[i]), rowp(piv_rows[i]) + rs, H.row(i));
    for (uint32_t i = 0; i < h; ++i) {
        const uint32_t ji = out.pivots[i];
        const W* Ri = H.row(i);
        for (uint32_t ip = 0; ip < i; ++ip) {
            W* Rp = H.row(ip);
            ring.shr(q.data(), Rp + static_cast<size_t>(ji) * L, out.pivot_v[i]);   // integer quotient
            if (ring.is_zero(q.data())) continue;
            for (uint32_t j = ji; j < n; ++j) {
                const W* pe = Ri + static_cast<size_t>(j) * L;
                if (!ring.is_zero(pe)) ring.submul(Rp + static_cast<size_t>(j) * L, q.data(), pe);
            }
        }
    }
    out.H = from_work(ring.layout(), H);
    return out;
}

// ---- the L = 1 dispatch to Sec. 13.5 - 13.6 (Sec. 13.11.4 last bullet) -----
template<class W> bool use_direct(const DigitLayout<W>& lay, const Options& opt) {
    return lay.L == 1 && !opt.force_induction && is_digit_word_v<W>;
}
template<class W> ModCtx<W> direct_ctx(const DigitLayout<W>& lay) {
    return ModCtx<W>::prime_power(static_cast<W>(lay.ell), lay.K);
}
template<class W> Howell<W> from_direct(const DigitLayout<W>& lay, const elim::HowellResult<W>& h) {
    Howell<W> out;
    out.H = LadicMat<W>::template from_word<W>(lay, cview(h.H));
    out.pivots = h.pivots;
    out.pivot_v = h.pivot_v;
    return out;
}

} // namespace detail

// ===========================================================================
// smith_lift (Sec. 13.11.3 - 13.11.4)
// ===========================================================================
template<class W>
Smith<W> smith_lift(const LadicMat<W>& A, const Options& opt = {}) {
    const DigitLayout<W>& lay = A.layout();
    const Ring<W> ring(lay);
    const uint32_t m = A.rows(), n = A.cols(), L = lay.L;
    detail::Work<W> R = detail::to_work(A);
    detail::Work<W> Uw;
    if (opt.transforms) {
        Uw = detail::Work<W>(m, m, L);
        for (uint32_t i = 0; i < m; ++i) ring.one(Uw.at(i, i));
    }
    const detail::DriverOut<W> d = detail::drive(ring, R, n, opt.transforms ? &Uw : nullptr, opt);
    Smith<W> out;
    out.rows = m;
    out.cols = n;
    out.s = d.s;
    out.pivot_row = d.prow;
    out.pivot_col = d.pcol;
    out.level_c = d.level_c;
    out.level_k = d.level_k;
    out.level_rank = d.level_rank;
    const uint32_t r = out.rank_K();
    detail::Work<W> T(r, n, L);
    for (uint32_t t = 0; t < r; ++t) std::copy(R.row(d.prow[t]), R.row(d.prow[t]) + static_cast<size_t>(n) * L, T.row(t));
    // non-pivot rows are zero (the last level's exit check at c >= K)
    for (uint32_t i = 0; i < m; ++i) {
        if (d.row_is_pivot[i]) continue;
        for (uint32_t j = 0; j < n; ++j) ENGINE_CHECK(ring.is_zero(R.at(i, j)));
    }
    if (opt.transforms) {
        // U: pivot rows first (pivot order), then the rest ascending.
        detail::Work<W> U2(m, m, L);
        uint32_t o = 0;
        for (uint32_t t = 0; t < r; ++t, ++o) std::copy(Uw.row(d.prow[t]), Uw.row(d.prow[t]) + static_cast<size_t>(m) * L, U2.row(o));
        for (uint32_t i = 0; i < m; ++i)
            if (!d.row_is_pivot[i]) { std::copy(Uw.row(i), Uw.row(i) + static_cast<size_t>(m) * L, U2.row(o)); ++o; }
        // V: column t = M'_J^{-1} e_t (x_F = 0); then the free-column kernel vectors.
        const detail::Echelon<W> e = detail::echelon_of(ring, R, d, n);
        detail::Work<W> V(n, n, L);
        std::vector<W> b(static_cast<size_t>(r) * L), x(static_cast<size_t>(n) * L);
        for (uint32_t t = 0; t < r; ++t) {
            std::fill(b.begin(), b.end(), W(0));
            std::fill(x.begin(), x.end(), W(0));
            ring.one(b.data() + static_cast<size_t>(t) * L);
            detail::back_substitute(ring, e, b, x);
            for (uint32_t i = 0; i < n; ++i) ring.copy(V.at(i, t), x.data() + static_cast<size_t>(i) * L);
        }
        uint32_t cnt = 0;
        const std::vector<W> kg = detail::kernel_generators(ring, e, cnt);   // free columns come first
        uint32_t col = r;
        for (uint32_t g = 0; g < cnt && col < n; ++g) {
            // only the free-column generators (the first n - r); the ell^{K-s} ones follow them
            for (uint32_t i = 0; i < n; ++i)
                ring.copy(V.at(i, col), kg.data() + (static_cast<size_t>(g) * n + i) * L);
            ++col;
        }
        ENGINE_CHECK(col == n);
        out.U = detail::from_work(lay, U2);
        out.V = detail::from_work(lay, V);
        out.has_transforms = true;
    }
    out.T = detail::from_work(lay, T);
    return out;
}

// ===========================================================================
// The derived entry points (Sec. 13.11.5)
// ===========================================================================
template<class W>
Howell<W> howell(const LadicMat<W>& A, const Options& opt = {}) {
    const DigitLayout<W>& lay = A.layout();
    if constexpr (is_digit_word_v<W>) {
        if (detail::use_direct(lay, opt)) {
            const DenseBuf<W> a = A.template to_word<W>();
            return detail::from_direct(lay, elim::howell<W>(detail::direct_ctx(lay), cview(a)));
        }
    }
    const Ring<W> ring(lay);
    const Smith<W> sm = smith_lift(A, opt);
    return detail::howell_rows(ring, detail::to_work(sm.T).d, sm.rank_K(), A.cols());
}

template<class W>
Howell<W> kernel_module(const LadicMat<W>& A, const Options& opt = {}) {
    const DigitLayout<W>& lay = A.layout();
    if constexpr (is_digit_word_v<W>) {
        if (detail::use_direct(lay, opt)) {
            const DenseBuf<W> a = A.template to_word<W>();
            return detail::from_direct(lay, elim::kernel_module<W>(detail::direct_ctx(lay), cview(a)));
        }
    }
    const Ring<W> ring(lay);
    const uint32_t m = A.rows(), n = A.cols();
    detail::Work<W> R = detail::to_work(A);
    const detail::DriverOut<W> d = detail::drive(ring, R, n, static_cast<detail::Work<W>*>(nullptr), opt);
    const detail::Echelon<W> e = detail::echelon_of(ring, R, d, n);
    uint32_t cnt = 0;
    std::vector<W> gens = detail::kernel_generators(ring, e, cnt);
    Howell<W> h = detail::howell_rows(ring, std::move(gens), cnt, n);
    uint32_t len_img = 0, len_ker = 0;
    for (uint32_t s : d.s) len_img += lay.K - s;
    for (uint32_t v : h.pivot_v) len_ker += lay.K - v;
    ENGINE_CHECK_MSG(len_ker + len_img == n * lay.K,
                     "ladic::kernel_module: length(ker) + length(im) != n K (ENGINE_API.md Sec. 13.11.5)");
    (void)m;
    return h;
}

template<class W>
Howell<W> image_module(const LadicMat<W>& A, const Options& opt = {}) {
    return howell(A.transpose(), opt);
}

template<class W>
uint32_t length(const Howell<W>& h) {
    uint32_t L = 0;
    for (uint32_t v : h.pivot_v) { ENGINE_CHECK(v < h.H.K()); L += h.H.K() - v; }
    return L;
}
template<class W>
uint32_t length(const LadicMat<W>& A, const Options& opt = {}) {
    const Smith<W> sm = smith_lift(A, opt);
    uint32_t L = 0;
    for (uint32_t s : sm.s) L += A.K() - s;
    return L;
}

// ny x nb: row t = x_t, reduced modulo kernel_module(B) (Sec. 13.6's rule).
template<class W>
LadicMat<W> solve_multi_rhs_module(const LadicMat<W>& B, const LadicMat<W>& Y, const Options& opt = {}) {
    detail::check_same_layout(B, Y, "ladic::solve_multi_rhs_module: B and Y must share one layout");
    const DigitLayout<W>& lay = B.layout();
    const uint32_t m = B.rows(), nb = B.cols(), ny = Y.cols();
    ENGINE_CHECK(Y.rows() == m);
    if constexpr (is_digit_word_v<W>) {
        if (detail::use_direct(lay, opt)) {
            const DenseBuf<W> b = B.template to_word<W>(), y = Y.template to_word<W>();
            const std::vector<std::vector<W>> x = elim::solve_multi_rhs_module<W>(detail::direct_ctx(lay), cview(b), cview(y));
            DenseBuf<W> X(ny, nb);
            for (uint32_t t = 0; t < ny; ++t)
                for (uint32_t j = 0; j < nb; ++j) X.at(t, j) = x[t][j];
            return LadicMat<W>::template from_word<W>(lay, cview(X));
        }
    }
    const Ring<W> ring(lay);
    const uint32_t L = lay.L, K = lay.K;
    LadicMat<W> out(lay, ny, nb);
    if (ny == 0) return out;
    detail::Work<W> R(m, nb + ny, L);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < nb; ++j) B.get(i, j, R.at(i, j));
        for (uint32_t j = 0; j < ny; ++j) Y.get(i, j, R.at(i, nb + j));
    }
    const detail::DriverOut<W> d = detail::drive(ring, R, nb, static_cast<detail::Work<W>*>(nullptr), opt);
    const detail::Echelon<W> e = detail::echelon_of(ring, R, d, nb);
    uint32_t cnt = 0;
    std::vector<W> kgens = detail::kernel_generators(ring, e, cnt);   // sets cnt (sequenced before its use)
    const Howell<W> Hk = detail::howell_rows(ring, std::move(kgens), cnt, nb);
    const detail::Work<W> Hw = detail::to_work(Hk.H);
    const uint32_t r = e.r;
    std::vector<W> b(static_cast<size_t>(r) * L), x(static_cast<size_t>(nb) * L), q(L);
    for (uint32_t t = 0; t < ny; ++t) {   // ascending t: the SMALLEST failing t throws (m8)
        bool ok = true;
        for (uint32_t i = 0; i < m && ok; ++i)
            if (!d.row_is_pivot[i] && !ring.is_zero(R.at(i, nb + t))) ok = false;
        for (uint32_t p = 0; p < r && ok; ++p) {
            const W* z = R.at(d.prow[p], nb + t);
            if (ring.valuation(z) < d.s[p]) ok = false;
            else ring.shr(b.data() + static_cast<size_t>(p) * L, z, d.s[p]);
        }
        if (!ok)
            throw std::runtime_error("Vector is not in image of basis (right-hand side " + std::to_string(t) + ").");
        std::fill(x.begin(), x.end(), W(0));
        detail::back_substitute(ring, e, b, x);
        // reduce modulo ker B, the kernel's pivot columns left to right
        for (uint32_t i = 0; i < Hk.rows(); ++i) {
            const uint32_t j = Hk.pivots[i];
            ring.shr(q.data(), x.data() + static_cast<size_t>(j) * L, Hk.pivot_v[i]);
            if (ring.is_zero(q.data())) continue;
            for (uint32_t jj = j; jj < nb; ++jj) {
                const W* he = Hw.at(i, jj);
                if (!ring.is_zero(he)) ring.submul(x.data() + static_cast<size_t>(jj) * L, q.data(), he);
            }
        }
        for (uint32_t j = 0; j < nb; ++j) out.set(t, j, x.data() + static_cast<size_t>(j) * L);
    }
    (void)K;
    return out;
}

template<class W>
std::vector<uint32_t> invariant_factors(const LadicMat<W>& A, const Options& opt = {}) {
    const DigitLayout<W>& lay = A.layout();
    if constexpr (is_digit_word_v<W>) {
        if (detail::use_direct(lay, opt)) {
            const DenseBuf<W> a = A.template to_word<W>();
            return elim::invariant_factors<W>(detail::direct_ctx(lay), cview(a));
        }
    }
    const Smith<W> sm = smith_lift(A, opt);
    std::vector<uint32_t> e;
    for (uint32_t s : sm.s) e.push_back(lay.K - s);
    std::sort(e.begin(), e.end(), [](uint32_t a, uint32_t b) { return a > b; });
    return e;
}

// M = (colspan G + colspan Rl) / colspan Rl = R^{ng} / pi(ker[G | Rl]).
template<class W>
std::vector<uint32_t> invariant_factors(const LadicMat<W>& G, const LadicMat<W>& Rl, const Options& opt = {}) {
    detail::check_same_layout(G, Rl, "ladic::invariant_factors: G and Rl must share one layout");
    const DigitLayout<W>& lay = G.layout();
    const uint32_t m = G.rows(), ng = G.cols(), nr = Rl.cols(), L = lay.L, K = lay.K;
    ENGINE_CHECK_MSG(Rl.rows() == m, "ladic::invariant_factors: G and Rl must have the same row count");
    if constexpr (is_digit_word_v<W>) {
        if (detail::use_direct(lay, opt)) {
            const DenseBuf<W> g = G.template to_word<W>(), rl = Rl.template to_word<W>();
            return elim::invariant_factors<W>(detail::direct_ctx(lay), cview(g), cview(rl));
        }
    }
    const Ring<W> ring(lay);
    std::vector<uint32_t> e;
    if (ng == 0) return e;
    detail::Work<W> N(m, ng + nr, L);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < ng; ++j) G.get(i, j, N.at(i, j));
        for (uint32_t j = 0; j < nr; ++j) Rl.get(i, j, N.at(i, ng + j));
    }
    detail::Work<W> R = N;
    const detail::DriverOut<W> d = detail::drive(ring, R, ng + nr, static_cast<detail::Work<W>*>(nullptr), opt);
    const detail::Echelon<W> ech = detail::echelon_of(ring, R, d, ng + nr);
    uint32_t cnt = 0;
    const std::vector<W> gens = detail::kernel_generators(ring, ech, cnt);
    LadicMat<W> Rel(lay, ng, cnt);
    for (uint32_t g = 0; g < cnt; ++g)
        for (uint32_t i = 0; i < ng; ++i) Rel.set(i, g, gens.data() + (static_cast<size_t>(g) * (ng + nr) + i) * L);
    const Smith<W> sr = smith_lift(Rel, opt);
    for (uint32_t s : sr.s) if (s >= 1) e.push_back(s);
    for (uint32_t i = sr.rank_K(); i < ng; ++i) e.push_back(K);
    std::sort(e.begin(), e.end(), [](uint32_t a, uint32_t b) { return a > b; });
    uint32_t sum = 0, lenN = 0, lenR = 0;
    for (uint32_t x : e) sum += x;
    for (uint32_t s : d.s) lenN += K - s;
    if (nr > 0) {
        const Smith<W> srl = smith_lift(Rl, opt);
        for (uint32_t s : srl.s) lenR += K - s;
    }
    ENGINE_CHECK_MSG(sum + lenR == lenN,
                     "ladic::invariant_factors: sum e_i != length(N) - length(colspan Rl) (ENGINE_API.md Sec. 13.6 step 4)");
    return e;
}

// (c1)-(c5) of DenseElim.h executed literally in digit arithmetic (Sec. 13.11.5).
template<class W>
Cyclic<W> cyclic_decomposition(const LadicMat<W>& G, const LadicMat<W>& Rl, const Options& opt = {}) {
    detail::check_same_layout(G, Rl, "ladic::cyclic_decomposition: G and Rl must share one layout");
    const DigitLayout<W>& lay = G.layout();
    const uint32_t m = G.rows(), ng = G.cols(), nr = Rl.cols(), L = lay.L, K = lay.K;
    ENGINE_CHECK_MSG(Rl.rows() == m, "ladic::cyclic_decomposition: G and Rl must have the same row count");
    Cyclic<W> out;
    if constexpr (is_digit_word_v<W>) {
        if (detail::use_direct(lay, opt)) {
            const DenseBuf<W> g = G.template to_word<W>(), rl = Rl.template to_word<W>();
            const elim::CyclicDecomposition<W> c = elim::cyclic_decomposition<W>(detail::direct_ctx(lay), cview(g), cview(rl));
            out.exponents = c.exponents;
            out.position = c.position;
            out.basis = LadicMat<W>::template from_word<W>(lay, cview(c.basis));
            out.Gamma = LadicMat<W>::template from_word<W>(lay, cview(c.Gamma));
            out.U = LadicMat<W>::template from_word<W>(lay, cview(c.U));
            return out;
        }
    }
    const Ring<W> ring(lay);
    // (c1)
    LadicMat<W> N(lay, m, nr + ng);
    std::vector<W> tmp(L);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < nr; ++j) { Rl.get(i, j, tmp.data()); N.set(i, j, tmp.data()); }
        for (uint32_t j = 0; j < ng; ++j) { G.get(i, j, tmp.data()); N.set(i, nr + j, tmp.data()); }
    }
    const Howell<W> Nh = image_module(N, opt);
    const uint32_t s = Nh.rows();
    out.Gamma = Nh.H.transpose();                               // m x s
    detail::Work<W> Uw(s, s, L);
    for (uint32_t i = 0; i < s; ++i) ring.one(Uw.at(i, i));
    if (s == 0) {
        out.U = detail::from_work(lay, Uw);
        out.basis = LadicMat<W>(lay, m, 0);
        return out;
    }
    const Howell<W> Kh = kernel_module(out.Gamma, opt);
    LadicMat<W> coords;
    if (nr > 0) coords = solve_multi_rhs_module(out.Gamma, Rl, opt);   // nr x s
    const uint32_t r = Kh.rows() + nr;
    detail::Work<W> Rm(s, r, L);
    for (uint32_t c = 0; c < Kh.rows(); ++c)
        for (uint32_t i = 0; i < s; ++i) Kh.H.get(c, i, Rm.at(i, c));
    for (uint32_t c = 0; c < nr; ++c)
        for (uint32_t i = 0; i < s; ++i) coords.get(c, i, Rm.at(i, Kh.rows() + c));
    detail::Work<W> Gm = detail::to_work(out.Gamma);
    std::vector<uint32_t> sigma;
    const uint32_t tmax = std::min(s, r);
    std::vector<W> uinv(L), up(L), q(L), sw(L);
    auto swap_e = [&](W* a, W* b) { std::swap_ranges(a, a + L, b); };
    for (uint32_t t = 0; t < tmax; ++t) {
        uint32_t pr = s, pc = r, pv = K;
        for (uint32_t j = t; j < r; ++j)
            for (uint32_t i = t; i < s; ++i) {
                const W* a = Rm.at(i, j);
                if (ring.is_zero(a)) continue;
                const uint32_t v = ring.valuation(a);
                if (v < pv) { pv = v; pr = i; pc = j; }
            }
        if (pr == s) break;
        if (pr != t) {
            for (uint32_t j = 0; j < r; ++j) swap_e(Rm.at(t, j), Rm.at(pr, j));
            for (uint32_t i = 0; i < m; ++i) swap_e(Gm.at(i, t), Gm.at(i, pr));
            for (uint32_t j = 0; j < s; ++j) swap_e(Uw.at(t, j), Uw.at(pr, j));
        }
        if (pc != t)
            for (uint32_t i = 0; i < s; ++i) swap_e(Rm.at(i, t), Rm.at(i, pc));
        ring.unit_part(up.data(), Rm.at(t, t));
        ring.inv(uinv.data(), up.data());
        if (!ring.is_one(uinv.data()))
            for (uint32_t i = t; i < s; ++i) ring.mul(Rm.at(i, t), Rm.at(i, t), uinv.data());
        ENGINE_DEBUG_CHECK(ring.valuation(Rm.at(t, t)) == pv);
        for (uint32_t j = t + 1; j < r; ++j) {                  // clear the pivot's ROW (column ops)
            if (ring.is_zero(Rm.at(t, j))) continue;
            ring.shr(q.data(), Rm.at(t, j), pv);
            for (uint32_t i = t; i < s; ++i)
                if (!ring.is_zero(Rm.at(i, t))) ring.submul(Rm.at(i, j), q.data(), Rm.at(i, t));
            ENGINE_DEBUG_CHECK(ring.is_zero(Rm.at(t, j)));
        }
        for (uint32_t i = t + 1; i < s; ++i) {                  // clear the pivot's COLUMN (row ops, mirrored)
            if (ring.is_zero(Rm.at(i, t))) continue;
            ring.shr(q.data(), Rm.at(i, t), pv);
            ring.zero(Rm.at(i, t));
            for (uint32_t j = 0; j < s; ++j)
                if (!ring.is_zero(Uw.at(t, j))) ring.submul(Uw.at(i, j), q.data(), Uw.at(t, j));
            for (uint32_t row = 0; row < m; ++row)
                if (!ring.is_zero(Gm.at(row, i))) ring.addmul(Gm.at(row, t), q.data(), Gm.at(row, i));
        }
        sigma.push_back(pv);
    }
    std::vector<std::pair<uint32_t, uint32_t>> summ;
    for (uint32_t i = 0; i < s; ++i) {
        if (i < sigma.size()) { if (sigma[i] >= 1) summ.emplace_back(sigma[i], i); }
        else summ.emplace_back(K, i);
    }
    std::stable_sort(summ.begin(), summ.end(),
                     [](const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b) { return a.first > b.first; });
    detail::Work<W> Bw(m, static_cast<uint32_t>(summ.size()), L);
    uint32_t sum = 0;
    for (uint32_t c = 0; c < summ.size(); ++c) {
        out.exponents.push_back(summ[c].first);
        out.position.push_back(summ[c].second);
        sum += summ[c].first;
        for (uint32_t i = 0; i < m; ++i) ring.copy(Bw.at(i, c), Gm.at(i, summ[c].second));
    }
    uint32_t lenN = 0, lenR = 0;
    for (uint32_t v : Nh.pivot_v) lenN += K - v;
    if (nr > 0) lenR = length(Rl, opt);
    ENGINE_CHECK_MSG(sum + lenR == lenN,
                     "ladic::cyclic_decomposition: sum e_i != length(N) - length(colspan Rl) ([E] sec:ring (c4))");
    out.basis = detail::from_work(lay, Bw);
    out.U = detail::from_work(lay, Uw);
    // out.Gamma stays the (c1) generators (the untransformed Gamma, as Sec. 13.6's output)
    return out;
}

// (c5): ny x t, row c = the coordinates of column c of Y on the b_i.
template<class W>
LadicMat<W> cyclic_coordinates(const Cyclic<W>& d, const LadicMat<W>& Y, const Options& opt = {}) {
    const DigitLayout<W>& lay = Y.layout();
    const uint32_t ny = Y.cols(), s = d.Gamma.cols(), t = d.rows(), L = lay.L;
    ENGINE_CHECK_MSG(Y.rows() == d.Gamma.rows(), "ladic::cyclic_coordinates: row count mismatch");
    LadicMat<W> out(lay, ny, t);
    if (ny == 0 || t == 0) {
        if (ny > 0 && s > 0) (void)solve_multi_rhs_module(d.Gamma, Y, opt);   // membership still checked
        return out;
    }
    const Ring<W> ring(lay);
    const LadicMat<W> x = solve_multi_rhs_module(d.Gamma, Y, opt);   // ny x s
    std::vector<W> acc(L), a(L), b(L);
    for (uint32_t c = 0; c < ny; ++c)
        for (uint32_t i = 0; i < t; ++i) {
            const uint32_t row = d.position[i];
            ring.zero(acc.data());
            for (uint32_t j = 0; j < s; ++j) {
                d.U.get(row, j, a.data());
                x.get(c, j, b.data());
                if (!ring.is_zero(a.data()) && !ring.is_zero(b.data())) ring.addmul(acc.data(), a.data(), b.data());
            }
            const uint32_t e = d.exponents[i];
            if (e < lay.K) ring.low(acc.data(), acc.data(), e);
            out.set(c, i, acc.data());
        }
    return out;
}

// [L] Sec. 2.4 on smith_lift's output (Sec. 13.11.5, last row).
template<class W>
StoppingRule stopping_rule(const Smith<W>& d0, const Smith<W>& d1, uint32_t dim_c0, uint32_t dim_c2) {
    StoppingRule r;
    r.r0 = d0.rank_K();
    r.r1 = d1.rank_K();
    r.need0 = dim_c0;
    r.need1 = dim_c2;
    r.certified = (r.r0 == dim_c0) && (r.r1 == dim_c2);
    return r;
}

// Dense product over Z/ell^K (test support and WP-18c's assembly helper): C = A B.
template<class W>
LadicMat<W> multiply(const LadicMat<W>& A, const LadicMat<W>& B) {
    detail::check_same_layout(A, B, "ladic::multiply: operands must share one layout");
    ENGINE_CHECK(A.cols() == B.rows());
    const DigitLayout<W>& lay = A.layout();
    const Ring<W> ring(lay);
    const detail::Work<W> a = detail::to_work(A), b = detail::to_work(B);
    detail::Work<W> c(A.rows(), B.cols(), lay.L);
    for (uint32_t i = 0; i < A.rows(); ++i)
        for (uint32_t k = 0; k < A.cols(); ++k) {
            const W* aik = a.at(i, k);
            if (ring.is_zero(aik)) continue;
            for (uint32_t j = 0; j < B.cols(); ++j) {
                const W* bkj = b.at(k, j);
                if (!ring.is_zero(bkj)) ring.addmul(c.at(i, j), aik, bkj);
            }
        }
    return detail::from_work(lay, c);
}

} // namespace ladic
} // namespace engine
