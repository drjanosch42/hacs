// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LOW_RANK_H
#define HECKE_LOW_RANK_H

// (hecke-extensions slot 13, WP-4) The U_ell memory lever's factored blocks
// (HeckeOperatorOptions::ScopedUEllLowRank, HeckeOperatorOnK.h): a d x d block
// that is rho_n of a 2x2 matrix SINGULAR mod ell, stored as U V with at most
// min(k, n + 1) outer products over Z/ell^k (rank 1 over F_ell).  Header-only;
// used by HeckeTauTable (the factored rho pool) and HeckeOperatorOnK (the
// host assembly of the fused blocks, route 2, and the module route).

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/DenseElim.h"   // (S23 G2-C) module_stream_enabled
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Kernels.h"
#include "../../ModArithmetic/ModCtx.h"

namespace hecke {
namespace lowrank {

using engine::DenseBuf;
using engine::Layout;
using engine::ModCtx;
using engine::cview;
using engine::kernels::GemmOp;

// (hecke-extensions slot 13, WP-4) THE U_ell MEMORY LEVER's factoriser.
//
// R = U V exactly over Z/ell^k (k = 1: F_ell), U: rows x t (row-major),
// V: t x cols (row-major), by elimination with MINIMUM-VALUATION full
// pivoting: pick an entry pi = ell^v w (w a unit) of least valuation, take
// u = column / (ell^v w) (exact: every entry has valuation >= v), v = the
// pivot row, subtract u v^T (which zeroes that row and column), repeat until
// R = 0.  Every step subtracts exactly what it records, so U V == R by
// construction (re-derived under ENGINE_DEBUG_CHECK); t is the number of
// non-zero Smith invariants of R.  THE RANK CLAIM: for rho_n of a 2x2 matrix
// that is singular mod ell, t <= min(k, n + 1) -- ENGINE_CHECKed here against
// `max_terms`, per block.  R is consumed.
template<class Elt>
struct LowRankT {
    uint32_t rows = 0, cols = 0, t = 0;
    std::vector<Elt> U;   // rows x t, row-major
    std::vector<Elt> V;   // t x cols, row-major
};

// (S23 G2-C) The ell-adic valuation by multiplication instead of division, for the scan below: for odd ell and
// inv = ell^{-1} mod 2^W (W = the word), x is divisible by ell iff x inv mod 2^W <= (2^W - 1) / ell, and then
// x / ell = x inv mod 2^W exactly -- so it returns ctx.valuation(x) for every canonical x (an even ell, never on the
// module route, takes ctx.valuation itself).  HECKE_MODULE_STREAM=0 keeps ctx.valuation.
template<class Elt>
struct FastValuation {
    const ModCtx<Elt>& ctx;
    bool fast = false;
    Elt inv = 1, lim = 0;
    FastValuation(const ModCtx<Elt>& c, bool on) : ctx(c) {
        const Elt e = static_cast<Elt>(c.ell);
        fast = on && !c.is_field() && (e & 1) != 0;
        if (!fast) return;
        Elt x = e;                                    // Newton: x <- x (2 - e x) doubles the correct low bits
        for (int it = 0; it < 6; ++it) x = static_cast<Elt>(x * static_cast<Elt>(2 - static_cast<Elt>(e * x)));
        inv = x;
        lim = static_cast<Elt>(~Elt(0) / e);
    }
    uint32_t operator()(Elt x) const {
        if (!fast) return ctx.valuation(x);
        if (x == 0) return ctx.k;
        uint32_t v = 0;
        for (;;) {
            if (v >= ctx.k) return v;
            const Elt y = static_cast<Elt>(x * inv);
            if (y > lim) return v;
            x = y;
            ++v;
        }
    }
};

// (S23 G2-T) THE SCAN'S DIVISIBILITY TEST (HECKE_MODULE_EVAL, `fast_scan`): once a candidate of valuation best_v is
// held, an entry x can only replace it if v(x) < best_v, i.e. iff ell^{best_v} does not divide x -- for odd ell
// tested by ONE multiplication, x * (ell^{-best_v} mod 2^W) > floor((2^W - 1) / ell^{best_v}) -- and only then is
// v(x) computed.  The same strict "<" in the same row-major order, so the same pivot (pr, pc) and best_v as the
// per-entry valuation scan; the elimination below is untouched.  Off (the default for every other caller) and for
// even ell / a field: the per-entry scan.
template<class Elt>
struct PowDivisibility {
    bool on = false;
    std::vector<Elt> inv, lim;   // [v] = ell^{-v} mod 2^W, floor((2^W - 1) / ell^v), v = 0 .. k
    PowDivisibility(const ModCtx<Elt>& ctx, const FastValuation<Elt>& val, bool want) {
        on = want && val.fast;
        if (!on) return;
        inv.assign(ctx.k + 1, Elt(1));
        lim.assign(ctx.k + 1, static_cast<Elt>(~Elt(0)));
        Elt pw = 1;
        for (uint32_t v = 1; v <= ctx.k; ++v) {
            inv[v] = static_cast<Elt>(inv[v - 1] * val.inv);
            pw = static_cast<Elt>(pw * static_cast<Elt>(ctx.ell));   // ell^v <= ell^k = p, no wrap
            lim[v] = static_cast<Elt>(~Elt(0) / pw);
        }
    }
    bool divisible(Elt x, uint32_t v) const { return static_cast<Elt>(x * inv[v]) <= lim[v]; }
};

template<class Elt>
LowRankT<Elt> factor_low_rank(const ModCtx<Elt>& ctx, std::vector<Elt>&& R, uint32_t rows, uint32_t cols,
                              uint32_t max_terms, bool fast_scan = false) {
    const FastValuation<Elt> val(ctx, engine::elim::module_stream_enabled());
    const PowDivisibility<Elt> pdiv(ctx, val, fast_scan && !ctx.is_field());
    ENGINE_CHECK(R.size() == static_cast<size_t>(rows) * cols);
#ifndef NDEBUG
    const std::vector<Elt> R0 = R;
#endif
    LowRankT<Elt> out;
    out.rows = rows; out.cols = cols;
    std::vector<std::vector<Elt>> us, vs;
    // (S23 G2-T, fast_scan) the valuation floor: every entry of R has valuation >= the last pivot's (the pivot is a
    // minimum and the update u v^T, u = column / (ell^v w), v = the pivot row, has every entry of valuation >= v), so
    // an entry AT the floor is the first minimum in row-major order and the scan may stop there -- the same pivot.
    uint32_t floor_v = 0;
    for (;;) {
        uint32_t best_v = ctx.k + 1, pr = 0, pc = 0;
        if (pdiv.on) {
            bool any = false;   // R == 0 is the common last pass: one plain sweep, no valuation work
            for (size_t e = 0; e < R.size() && !any; ++e) any = R[e] != 0;
            if (any)
            for (uint32_t a = 0; a < rows && best_v > floor_v; ++a)
                for (uint32_t b = 0; b < cols; ++b) {
                    const Elt x = R[static_cast<size_t>(a) * cols + b];
                    if (x == 0) continue;
                    if (best_v <= ctx.k && pdiv.divisible(x, best_v)) continue;   // v(x) >= best_v: not a new pivot
                    const uint32_t v = val(x);
                    ENGINE_DEBUG_CHECK(v < best_v && v >= floor_v);
                    if (v < best_v) { best_v = v; pr = a; pc = b; if (v == floor_v) break; }
                }
            if (best_v <= ctx.k) floor_v = best_v;
        } else
        for (uint32_t a = 0; a < rows && best_v > 0; ++a)
            for (uint32_t b = 0; b < cols; ++b) {
                const Elt x = R[static_cast<size_t>(a) * cols + b];
                if (x == 0) continue;
                const uint32_t v = ctx.is_field() ? 0u : val(x);
                if (v < best_v) { best_v = v; pr = a; pc = b; if (v == 0) break; }
            }
        if (best_v > ctx.k) break;   // R == 0
        ENGINE_CHECK_MSG(out.t < max_terms,
                         "HeckeOperatorOnK (U_ell lever): a rho_n block of a matrix singular mod ell has more "
                         "outer-product terms than the rank claim allows (<= min(k, n + 1)).");
        const Elt pi = R[static_cast<size_t>(pr) * cols + pc];
        const Elt ellv = ctx.is_field() ? Elt(1) : ctx.ell_pow(best_v);
        const Elt w = ctx.is_field() ? pi : ctx.unit_part(pi);
        const Elt winv = ctx.inv(static_cast<Elt>(w % ctx.p));
        std::vector<Elt> u(rows), v(cols);
        for (uint32_t a = 0; a < rows; ++a) {
            const Elt x = R[static_cast<size_t>(a) * cols + pc];
            u[a] = ctx.mul(static_cast<Elt>(x / ellv), winv);   // exact: valuation(x) >= best_v
        }
        for (uint32_t b = 0; b < cols; ++b) v[b] = R[static_cast<size_t>(pr) * cols + b];
        if (pdiv.on) {
            // (S23 G2-T) R -= u v^T as ONE exact rank-1 kernels::gemm (GemmOp::Sub): every entry the canonical residue
            // of R(a, b) - u[a] v[b], as the loop below (where u[a] == 0 or v[b] == 0 it leaves R(a, b) as it is; so
            // does R - 0)
            DenseBuf<Elt> ub(u.data(), u.size(), rows, 1, Layout::RowMajor);
            DenseBuf<Elt> vb(v.data(), v.size(), 1, cols, Layout::RowMajor);
            DenseBuf<Elt> Rb(R.data(), R.size(), rows, cols, Layout::RowMajor);
            engine::kernels::gemm<Elt>(ctx, cview(ub), cview(vb), engine::view(Rb), GemmOp::Sub);
        } else
        for (uint32_t a = 0; a < rows; ++a) {
            if (u[a] == 0) continue;
            Elt* row = &R[static_cast<size_t>(a) * cols];
            for (uint32_t b = 0; b < cols; ++b)
                if (v[b] != 0) row[b] = ctx.sub(row[b], ctx.mul(u[a], v[b]));
        }
        ENGINE_DEBUG_CHECK_MSG(R[static_cast<size_t>(pr) * cols + pc] == 0,
                               "HeckeOperatorOnK (U_ell lever): elimination step did not clear its pivot");
        us.push_back(std::move(u)); vs.push_back(std::move(v));
        ++out.t;
    }
    out.U.assign(static_cast<size_t>(rows) * out.t, Elt(0));
    out.V.assign(static_cast<size_t>(out.t) * cols, Elt(0));
    for (uint32_t s = 0; s < out.t; ++s) {
        for (uint32_t a = 0; a < rows; ++a) out.U[static_cast<size_t>(a) * out.t + s] = us[s][a];
        std::copy(vs[s].begin(), vs[s].end(), out.V.begin() + static_cast<size_t>(s) * cols);
    }
#ifndef NDEBUG
    for (uint32_t a = 0; a < rows; ++a)
        for (uint32_t b = 0; b < cols; ++b) {
            Elt acc = 0;
            for (uint32_t s = 0; s < out.t; ++s)
                acc = ctx.add(acc, ctx.mul(out.U[static_cast<size_t>(a) * out.t + s], out.V[static_cast<size_t>(s) * cols + b]));
            ENGINE_DEBUG_CHECK_MSG(acc == R0[static_cast<size_t>(a) * cols + b],
                                   "HeckeOperatorOnK (U_ell lever): U V != R");
        }
#endif
    return out;
}

// ===========================================================================
// (hecke-extensions slot 14, G0) THE RANK-ONE CLOSED FORM.
//
// Over F_ell (ctx.is_field()), let g = [[a, b], [c, d]] be singular mod ell.
// rho_n(g)'s column j is (a + c y)^(n-j) (b + d y)^j (ComputeSymmetricPowerMatrixFM's
// convention, u_k = x^(n-k) y^k).  Singularity makes the two linear forms
// proportional, so rho_n(g) = w v^T with
//   (a, c) != 0:  (b, d) = lambda (a, c)  (lambda = b/a, or d/c when a = 0),
//                 w = coefficients of (a + c y)^n,  v_j = lambda^j  (0^0 = 1);
//   (a, c) == 0:  w = coefficients of (b + d y)^n,  v = e_n  (column n alone).
// w and v cost O(d) each, against the O(d^2) of the dense block and the O(d^2)
// scan of factor_low_rank.  factor_low_rank's normalisation is reproduced BYTE
// FOR BYTE: its pivot is the first non-zero entry of R in row-major order,
// i.e. (k0, j0) with k0 the first k with w_k != 0 and j0 the first j with
// v_j != 0; u = column j0 times the pivot's inverse; v = row k0; one step
// clears R (R - u v^T = 0 for a rank-1 R), so t = 1, or t = 0 when R = 0
// (n >= 1 and g == 0).  Every entry is a canonical residue computed by
// ctx.mul from the entries' unique values mod ell, so the bytes cannot depend
// on the route.
//
// THE RANK CLAIM stays ENGINE_CHECKed per key: det g == 0 mod ell is checked
// (O(1)), and with it the factorisation's proportionality (b, d) = lambda (a, c)
// -- the hypothesis under which t <= 1 = min(k, n + 1); a key that is not
// singular throws exactly where factor_low_rank's term-count check would.
// Under !NDEBUG (Check / CheckThread / Debug) every key is also re-derived by
// the dense route, ComputeSymmetricPowerMatrixFM + factor_low_rank, and the two
// (U, V) compared entrywise.  The independent dense witness of every tau-pool
// slot stays in HeckeTauTable::certify() clause (f), which recomputes rho_n
// from the exact gamma and compares it with U V (the certificate chain: route
// 2's blocks are COMPAREd with route 1's, which read the (f)-certified pool).
//
// `binom` is the row C(n, k) mod ell, k = 0..n (binomial_row_mod), shared by
// every key of one (n, ell): Lucas's theorem over factorial tables of size
// min(ell, n + 1) built with ctx (C(n, k) mod ell is a unique residue, so it is
// mod_nCr's value; no raw mod_* call, raw_mod_calls.sh).
template<class Elt>
std::vector<Elt> binomial_row_mod(const ModCtx<Elt>& ctx, uint32_t n) {
    ENGINE_CHECK_MSG(ctx.is_field(), "binomial_row_mod: F_ell only");
    const uint64_t ell = static_cast<uint64_t>(ctx.p);
    const uint64_t m = std::min<uint64_t>(ell - 1, n);   // every base-ell digit of n or k is <= m
    std::vector<Elt> fact(m + 1), inv_fact(m + 1);
    fact[0] = Elt(1);
    for (uint64_t i = 1; i <= m; ++i) fact[i] = ctx.mul(fact[i - 1], static_cast<Elt>(i));
    inv_fact[m] = ctx.inv(fact[m]);
    for (uint64_t i = m; i > 0; --i) inv_fact[i - 1] = ctx.mul(inv_fact[i], static_cast<Elt>(i));
    std::vector<Elt> out(static_cast<size_t>(n) + 1);
    for (uint32_t k = 0; k <= n; ++k) {
        Elt r = Elt(1);
        for (uint64_t nn = n, kk = k; nn > 0 || kk > 0; nn /= ell, kk /= ell) {
            const uint64_t a = nn % ell, b = kk % ell;
            if (b > a) { r = Elt(0); break; }
            r = ctx.mul(r, ctx.mul(fact[a], ctx.mul(inv_fact[b], inv_fact[a - b])));
        }
        out[k] = r;
    }
    return out;
}

template<class Elt>
LowRankT<Elt> rank_one_rho_factors(const ModCtx<Elt>& ctx, Elt a, Elt b, Elt c, Elt dd, uint32_t n,
                                   const std::vector<Elt>& binom) {
    ENGINE_CHECK_MSG(ctx.is_field(), "rank_one_rho_factors: F_ell only (the rank-1 closed form)");
    ENGINE_CHECK(binom.size() == static_cast<size_t>(n) + 1);
    ENGINE_CHECK(a < ctx.p && b < ctx.p && c < ctx.p && dd < ctx.p);
    ENGINE_CHECK_MSG(ctx.mul(a, dd) == ctx.mul(b, c),
                     "HeckeOperatorOnK (U_ell lever): a rho_n block of a matrix singular mod ell has more "
                     "outer-product terms than the rank claim allows (<= min(k, n + 1)) -- the key is not "
                     "singular mod ell (rank-1 closed form, G0).");
    const uint32_t d = n + 1;
    std::vector<Elt> w(d), v(d, Elt(0));
    // w_k = C(n, k) x0^(n-k) x1^k, the coefficients of (x0 + x1 y)^n (0^0 = 1).
    auto power_poly = [&](Elt x0, Elt x1) {
        std::vector<Elt> p0(d);
        p0[0] = Elt(1);
        for (uint32_t k = 1; k < d; ++k) p0[k] = ctx.mul(p0[k - 1], x0);
        Elt p1 = Elt(1);
        for (uint32_t k = 0; k < d; ++k) {
            w[k] = ctx.mul(binom[k], ctx.mul(p0[n - k], p1));
            p1 = ctx.mul(p1, x1);
        }
    };
    if (a != 0 || c != 0) {
        const Elt lambda = (a != 0) ? ctx.mul(b, ctx.inv(a)) : ctx.mul(dd, ctx.inv(c));
        ENGINE_CHECK_MSG(ctx.mul(lambda, a) == b && ctx.mul(lambda, c) == dd,
                         "rank_one_rho_factors: (b, d) is not proportional to (a, c) (internal error)");
        power_poly(a, c);
        Elt pl = Elt(1);
        for (uint32_t j = 0; j < d; ++j) { v[j] = pl; pl = ctx.mul(pl, lambda); }
    } else {
        power_poly(b, dd);
        v[n] = Elt(1);
    }
    LowRankT<Elt> out;
    out.rows = d; out.cols = d;
    uint32_t k0 = d, j0 = d;
    for (uint32_t k = 0; k < d; ++k) if (w[k] != 0) { k0 = k; break; }
    for (uint32_t j = 0; j < d; ++j) if (v[j] != 0) { j0 = j; break; }
    if (k0 == d || j0 == d) return out;   // R == 0: t = 0, U and V empty (factor_low_rank's shape)
    const Elt pi = ctx.mul(w[k0], v[j0]);
    const Elt pinv = ctx.inv(pi);
    out.t = 1;
    out.U.resize(d);
    out.V.resize(d);
    for (uint32_t r = 0; r < d; ++r) out.U[r] = ctx.mul(ctx.mul(w[r], v[j0]), pinv);   // R(r, j0) / pi
    for (uint32_t s = 0; s < d; ++s) out.V[s] = ctx.mul(w[k0], v[s]);                   // R(k0, s)
    return out;
}

// Dense row-major copy of a FlatMatrix (any storage kind).
template<class Elt>
std::vector<Elt> dense_rows_of(const FlatMatrixT<Elt>& M) {
    std::vector<Elt> out(static_cast<size_t>(M.get_rows()) * M.get_cols());
    for (uint32_t a = 0; a < M.get_rows(); ++a)
        for (uint32_t b = 0; b < M.get_cols(); ++b) out[static_cast<size_t>(a) * M.get_cols() + b] = M.get_entry(a, b);
    return out;
}

// dst (d x d, row-major view) = sum over the listed (coeff, L_j, V_j) of
// coeff * L_j V_j, L_j: d x t_j, V_j: t_j x d -- one GEMM over the stacked
// factors [c_1 L_1 | c_2 L_2 | ...] . [V_1; V_2; ...].  (S23 G2-C) `op` = GemmOp::Add accumulates into dst
// instead (the module route's grouped T_ell accumulation); every entry is an exact sum mod ell^k.
template<class Elt>
void sum_of_low_rank(const ModCtx<Elt>& ctx, uint32_t d,
                     const std::vector<std::tuple<Elt, const std::vector<Elt>*, const std::vector<Elt>*, uint32_t>>& terms,
                     engine::DenseView<Elt> dst, GemmOp op = GemmOp::Set) {
    uint32_t T = 0;
    for (const auto& tm : terms) T += std::get<3>(tm);
    if (T == 0) {
        if (op != GemmOp::Set) return;   // (S23 G2-C) accumulating nothing
        for (uint32_t a = 0; a < d; ++a) for (uint32_t b = 0; b < d; ++b) dst.at(a, b) = Elt(0);
        return;
    }
    DenseBuf<Elt> Lb(d, T, Layout::RowMajor), Vb(T, d, Layout::RowMajor);
    uint32_t off = 0;
    for (const auto& tm : terms) {
        const Elt c = std::get<0>(tm);
        const std::vector<Elt>& L = *std::get<1>(tm);
        const std::vector<Elt>& V = *std::get<2>(tm);
        const uint32_t t = std::get<3>(tm);
        for (uint32_t a = 0; a < d; ++a)
            for (uint32_t s = 0; s < t; ++s) Lb.at(a, off + s) = ctx.mul(c, L[static_cast<size_t>(a) * t + s]);
        for (uint32_t s = 0; s < t; ++s)
            for (uint32_t b = 0; b < d; ++b) Vb.at(off + s, b) = V[static_cast<size_t>(s) * d + b];
        off += t;
    }
    engine::kernels::gemm<Elt>(ctx, cview(Lb), cview(Vb), dst, op);
}

} // namespace lowrank
} // namespace hecke

#endif // HECKE_LOW_RANK_H
