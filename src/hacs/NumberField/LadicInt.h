// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// LadicInt.h -- residues modulo ell^W as K limbs in base beta = ell^d < 2^30
// (slot-19 WP-NF2; the coefficient type of the Coates--Wiles layer ring).
//
// A value is K uint32 limbs, little-endian, limbs 0 .. K-2 in [0, beta), the top
// limb in [0, ell^{W - d(K-1)}): the residue ring Z/ell^W is "truncate and
// reduce the top limb", and the ell-adic digits are read off directly. Every
// operation is exact uint64 arithmetic (beta^2 < 2^60), no __int128.
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "BigInt.h"
#include "NfCheck.h"

namespace nf {

constexpr unsigned kLadicMaxLimbs = 8;
using LVal = std::array<uint32_t, kLadicMaxLimbs>;

struct LadicCtx {
    uint32_t ell = 0;
    unsigned W = 0;   // precision: residues mod ell^W
    unsigned d = 0;   // digits per limb
    unsigned K = 0;   // limbs
    uint32_t beta = 0, topmod = 0;
    std::vector<uint32_t> lpow;  // ell^0 .. ell^d

    LadicCtx() = default;
    LadicCtx(uint32_t l, unsigned w) : ell(l), W(w) {
        NF_CHECK_MSG(l >= 3 && w >= 1, "LadicCtx: ell >= 3 and W >= 1");
        d = 0;
        uint64_t b = 1;
        while (b * l < (uint64_t(1) << 30)) {
            b *= l;
            ++d;
        }
        NF_CHECK_MSG(d >= 1, "LadicCtx: ell too large for a 30-bit limb");
        beta = static_cast<uint32_t>(b);
        K = (W + d - 1) / d;
        NF_CHECK_MSG(K <= kLadicMaxLimbs, "LadicCtx: precision beyond kLadicMaxLimbs limbs");
        lpow.resize(d + 1);
        lpow[0] = 1;
        for (unsigned i = 1; i <= d; ++i) lpow[i] = lpow[i - 1] * l;
        topmod = lpow[W - d * (K - 1)];
    }

    // ---- scalars on raw limb pointers (K limbs each)
    inline void normalize64(const uint64_t* acc, uint32_t* out) const {
        uint64_t carry = 0;
        for (unsigned i = 0; i + 1 < K; ++i) {
            const uint64_t t = acc[i] + carry;
            out[i] = static_cast<uint32_t>(t % beta);
            carry = t / beta;
        }
        out[K - 1] = static_cast<uint32_t>((acc[K - 1] + carry) % topmod);
    }
    inline void add(const uint32_t* a, const uint32_t* b, uint32_t* out) const {
        uint32_t carry = 0;
        for (unsigned i = 0; i + 1 < K; ++i) {
            uint32_t t = a[i] + b[i] + carry;
            carry = t >= beta;
            out[i] = carry ? t - beta : t;
        }
        uint32_t t = a[K - 1] + b[K - 1] + carry;
        out[K - 1] = t >= topmod ? t - topmod : t;
    }
    inline void neg(const uint32_t* a, uint32_t* out) const {
        // Q - a (mod Q): limbs (beta - 1 - a_i), top (topmod - 1 - a_top), then + 1.
        uint32_t carry = 1;
        for (unsigned i = 0; i + 1 < K; ++i) {
            uint32_t t = (beta - 1 - a[i]) + carry;
            carry = t >= beta;
            out[i] = carry ? t - beta : t;
        }
        uint32_t t = (topmod - 1 - a[K - 1]) + carry;
        out[K - 1] = t >= topmod ? t - topmod : t;
    }
    inline void sub(const uint32_t* a, const uint32_t* b, uint32_t* out) const {
        uint32_t nb[kLadicMaxLimbs];
        neg(b, nb);
        add(a, nb, out);
    }
    inline void mul(const uint32_t* a, const uint32_t* b, uint32_t* out) const {
        uint64_t acc[kLadicMaxLimbs] = {0};
        for (unsigned i = 0; i < K; ++i) {
            if (!a[i]) continue;
            const uint64_t ai = a[i];
            for (unsigned j = 0; i + j < K; ++j) acc[i + j] += ai * b[j];
        }
        normalize64(acc, out);
    }
    inline void mul_u32(const uint32_t* a, uint32_t c, uint32_t* out) const {
        // c < 2^32: split c = c1 beta + c0 so every partial product stays below 2^61.
        const uint64_t c0 = c % beta, c1 = c / beta;
        uint64_t acc[kLadicMaxLimbs] = {0};
        for (unsigned i = 0; i < K; ++i) {
            acc[i] += a[i] * c0;
            if (c1 && i + 1 < K) acc[i + 1] += a[i] * c1;
        }
        normalize64(acc, out);
    }
    inline bool is_zero(const uint32_t* a) const {
        for (unsigned i = 0; i < K; ++i)
            if (a[i]) return false;
        return true;
    }
    inline uint32_t digit(const uint32_t* a, unsigned s) const { return (a[s / d] / lpow[s % d]) % ell; }
    // v_ell(a), W if a == 0.
    inline unsigned valuation(const uint32_t* a) const {
        for (unsigned i = 0; i < K; ++i) {
            if (a[i]) {
                uint32_t x = a[i];
                unsigned v = i * d;
                while (x % ell == 0) {
                    x /= ell;
                    ++v;
                }
                return v;
            }
        }
        return W;
    }
    // a * ell^e mod ell^W.
    void mul_lpow(const uint32_t* a, unsigned e, uint32_t* out) const {
        uint32_t t[kLadicMaxLimbs];
        for (unsigned i = 0; i < K; ++i) t[i] = a[i];
        while (e > 0) {
            const unsigned s = e < d ? e : d;
            mul_u32(t, lpow[s], t);
            e -= s;
        }
        for (unsigned i = 0; i < K; ++i) out[i] = t[i];
    }
    // floor(a / ell^e) (as an integer in [0, ell^W)), written mod ell^W.
    void div_lpow(const uint32_t* a, unsigned e, uint32_t* out) const {
        uint32_t t[kLadicMaxLimbs];
        for (unsigned i = 0; i < K; ++i) t[i] = a[i];
        while (e > 0) {
            const unsigned s = e < d ? e : d;
            const uint64_t q = lpow[s];
            uint64_t rem = 0;
            for (int i = static_cast<int>(K) - 1; i >= 0; --i) {
                const uint64_t cur = rem * beta + t[i];
                t[i] = static_cast<uint32_t>(cur / q);
                rem = cur % q;
            }
            e -= s;
        }
        for (unsigned i = 0; i < K; ++i) out[i] = t[i];
    }
    void from_u64(uint64_t v, uint32_t* out) const {
        uint64_t acc[kLadicMaxLimbs] = {0};
        for (unsigned i = 0; i + 1 < K && v; ++i) {
            acc[i] = v % beta;
            v /= beta;
        }
        acc[K - 1] += v % topmod;
        normalize64(acc, out);
    }
    void from_i64(int64_t v, uint32_t* out) const {
        if (v >= 0) {
            from_u64(static_cast<uint64_t>(v), out);
        } else {
            uint32_t t[kLadicMaxLimbs];
            from_u64(static_cast<uint64_t>(-(v + 1)) + 1, t);
            neg(t, out);
        }
    }
    LVal from_bigint(const BigInt& x) const {
        BigInt q = x.floor_mod(modulus());
        LVal out{};
        const BigInt B(static_cast<int64_t>(beta));
        for (unsigned i = 0; i < K; ++i) {
            BigInt qq, rr;
            BigInt::divmod(q, B, qq, rr);
            out[i] = static_cast<uint32_t>(rr.to_int64());
            q = qq;
        }
        return out;
    }
    BigInt to_bigint(const uint32_t* a) const {
        BigInt r(0);
        for (int i = static_cast<int>(K) - 1; i >= 0; --i) r = r * BigInt(static_cast<int64_t>(beta)) + BigInt(a[i]);
        return r;
    }
    BigInt modulus() const { return BigInt::pow(BigInt(static_cast<int64_t>(ell)), W); }
    // inverse of a unit (Newton from the residue mod ell).
    LVal inverse(const uint32_t* a) const {
        const uint32_t a0 = a[0] % ell;
        NF_CHECK_MSG(a0 != 0, "LadicCtx::inverse: not a unit");
        // x0 = a0^{-1} mod ell (Fermat)
        uint64_t x0 = 1, b = a0, e = ell - 2;
        while (e) {
            if (e & 1u) x0 = x0 * b % ell;
            b = b * b % ell;
            e >>= 1;
        }
        LVal x{}, two{}, t{}, u{};
        from_u64(x0, x.data());
        from_u64(2, two.data());
        for (unsigned prec = 1; prec < W; prec *= 2) {
            mul(a, x.data(), t.data());
            sub(two.data(), t.data(), u.data());
            mul(x.data(), u.data(), x.data());
        }
        return x;
    }
    LVal pow(const uint32_t* a, BigInt e) const;
    std::string to_string(const uint32_t* a) const { return to_bigint(a).to_string(); }
};

inline LVal LadicCtx::pow(const uint32_t* a, BigInt e) const {
    LVal r{}, b{};
    from_u64(1, r.data());
    for (unsigned i = 0; i < K; ++i) b[i] = a[i];
    const BigInt two(2);
    while (!e.is_zero()) {
        BigInt q, rem;
        BigInt::divmod(e, two, q, rem);
        if (!rem.is_zero()) mul(r.data(), b.data(), r.data());
        e = q;
        if (!e.is_zero()) mul(b.data(), b.data(), b.data());
    }
    return r;
}

}  // namespace nf
