// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 4, WP-7 (phase E6) -- HeckeExt/HeckePredictions.
// See the header for the mathematics, the range and the three Kummer branches.

#include "HeckePredictions.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"

#include <sstream>

namespace hecke_ext {

namespace {

// The inverses of 1, 2, ..., n in F_ell, in one batch (Montgomery's trick
// inside ModCtx).  out[i] = (i + 1)^{-1}.
std::vector<uint32_t> small_inverses(uint32_t n, const engine::ModCtx<uint32_t>& ctx) {
    std::vector<uint32_t> x(n);
    for (uint32_t i = 0; i < n; ++i) x[i] = (i + 1u) % ctx.p;
    // Every entry is a unit: i + 1 <= n <= ell - 1 by the caller's precondition.
    for (uint32_t i = 0; i < n; ++i)
        ENGINE_CHECK_MSG(x[i] != 0u, "hecke_ext::small_inverses: a non-unit reached the batch");
    ctx.batch_inv(x);
    return x;
}

} // namespace

uint32_t bernoulli_max_index(uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 2, "hecke_ext::bernoulli_max_index: ell < 2");
    return ell - 2u;
}

std::vector<uint32_t> bernoulli_mod_table(uint32_t w_max, uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 2, "hecke_ext::bernoulli_mod_table: ell < 2");
    if (w_max >= ell - 1u) {
        std::ostringstream os;
        os << "bernoulli_mod_table: w_max = " << w_max << " is out of range at ell = " << ell
           << " (von Staudt-Clausen: B_w has ell in its denominator once (ell - 1) | w, "
           << "so the recursion is valid only for w <= ell - 2 = " << (ell - 2u) << ")";
        throw BernoulliOutOfRange(os.str());
    }

    const engine::ModCtx<uint32_t> ctx(ell);
    // Inverses of 1 .. w_max + 1; all of them are units because
    // w_max + 1 <= ell - 1 < ell.
    const std::vector<uint32_t> inv = small_inverses(w_max + 1u, ctx);

    std::vector<uint32_t> b(w_max + 1u, 0u);
    b[0] = 1u % ell;

    for (uint32_t m = 1; m <= w_max; ++m) {
        // s = sum_{j < m} C(m + 1, j) b[j], the binomial row built in place by
        // C(N, j) = C(N, j - 1) * (N - j + 1) / j with N = m + 1 <= ell - 1, so
        // no Lucas reduction and no zero denominator can occur.
        const uint32_t N = m + 1u;
        uint32_t c = 1u % ell; // C(N, 0)
        uint32_t s = 0u;
        for (uint32_t j = 0; j < m; ++j) {
            s = ctx.add(s, ctx.mul(c, b[j]));
            // advance to C(N, j + 1)
            c = ctx.mul(ctx.mul(c, (N - j) % ell), inv[j]); // inv[j] = (j + 1)^{-1}
        }
        // b[m] = -s / (m + 1)
        b[m] = ctx.sub(0u, ctx.mul(s, inv[m])); // inv[m] = (m + 1)^{-1}
    }

    ENGINE_DEBUG_CHECK_MSG(b.size() == static_cast<size_t>(w_max) + 1u,
                           "hecke_ext::bernoulli_mod_table: table length");
    return b;
}

uint32_t bernoulli_mod(uint32_t w, uint32_t ell) {
    return bernoulli_mod_table(w, ell).back();
}

std::string KummerVerdict::to_string() const {
    std::ostringstream os;
    os << "KummerVerdict{k=" << k << " ell=" << ell
       << " applicable=" << (applicable ? "true" : "false")
       << " divides=" << (divides ? "true" : "false") << " reason=" << reason;
    if (reason == "direct" || reason == "kummer_congruence")
        os << " k'=" << k_reduced << " B_{k'}/k'=" << b_over_k_mod_ell;
    os << "}";
    return os.str();
}

KummerVerdict kummer_verdict(uint32_t k, uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 2, "hecke_ext::kummer_verdict: ell < 2");

    KummerVerdict v;
    v.k = k;
    v.ell = ell;

    if (k == 0u) {
        v.applicable = false;
        v.reason = "k_eq_zero";
        return v;
    }
    v.applicable = true;

    if (k == 1u) {
        // B_1/1 = -1/2, numerator -1; no prime divides it.
        v.divides = false;
        v.reason = "k_eq_one";
        return v;
    }
    if ((k % 2u) == 1u) {
        // B_k = 0 for odd k >= 3, so num(B_k/k) = 0 and every ell divides it.
        v.divides = true;
        v.reason = "odd_k_vanishes";
        return v;
    }

    const uint32_t period = ell - 1u; // >= 1
    if ((k % period) == 0u) {
        // von Staudt-Clausen: ell | den(B_k), hence ell does not divide
        // num(B_k), hence not num(B_k/k) either.
        v.divides = false;
        v.reason = "von_staudt_clausen";
        return v;
    }

    const uint32_t k_red = k % period; // 2 <= k_red <= ell - 3, even (ell is odd here)
    ENGINE_CHECK_MSG(k_red >= 1u && k_red < period,
                     "hecke_ext::kummer_verdict: reduced index out of range");
    ENGINE_CHECK_MSG((k_red % 2u) == 0u,
                     "hecke_ext::kummer_verdict: reduced index is odd -- ell - 1 must be even here");

    const engine::ModCtx<uint32_t> ctx(ell);
    const std::vector<uint32_t> b = bernoulli_mod_table(k_red, ell);
    v.k_reduced = k_red;
    v.b_over_k_mod_ell = ctx.mul(b[k_red], ctx.inv(k_red % ell));
    v.divides = (v.b_over_k_mod_ell == 0u);
    v.reason = (k_red == k) ? "direct" : "kummer_congruence";
    return v;
}

bool kummer_divides(uint32_t k, uint32_t ell) {
    const KummerVerdict v = kummer_verdict(k, ell);
    if (!v.applicable) {
        std::ostringstream os;
        os << "kummer_divides: k = " << k << " has no B_k/k (" << v.reason << ")";
        throw BernoulliOutOfRange(os.str());
    }
    return v.divides;
}

} // namespace hecke_ext
