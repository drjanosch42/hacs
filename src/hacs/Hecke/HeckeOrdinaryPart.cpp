// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 12, WP-4) The certified ordinary projector -- see the header.
#include "HeckeOrdinaryPart.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "HeckeSettings.h"   // NotImplementedInThisBuild
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/LadicInduction.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "DenseRref.h"
#include "FpRank.h"

namespace hecke {
namespace ordinary {

namespace {

std::string CellText(uint32_t n, uint64_t ell) {
    return "(" + std::to_string(n) + ", " + std::to_string(ell) + ")";
}

void RequireLevel(uint32_t n, uint64_t ell, uint32_t e) {
    if (engine::scalar::powfit(ell, e) == 0) {
        std::ostringstream o;
        o << "HeckeOrdinaryPart (torsion pre-pass): ell^" << e << " does not fit uint64_t at " << CellText(n, ell)
          << "; the engine::ladic path of the pre-pass is not in this build ([C] prop:torsionprepass(3))";
        throw NotImplementedInThisBuild(o.str());
    }
}

// Rank and a nullspace basis of a rows x cols matrix over F_ell (row-major): one vector per free column ascending, 1 at
// the free column, -RREF[i][free] at pivot i.  (S24 CL-G2, B3 (b)) the engine's RREF (DenseRref.h).
uint32_t FieldNullspace(std::vector<uint64_t> A, uint32_t rows, uint32_t cols, uint64_t ell,
                        std::vector<std::vector<uint64_t>>* basis) {
    const std::vector<uint32_t> piv = RrefRowMajor<uint64_t>(A, rows, cols, ell);
    std::vector<int64_t> row_of(cols, -1);
    for (size_t r = 0; r < piv.size(); ++r) row_of[piv[r]] = static_cast<int64_t>(r);
    std::vector<std::vector<uint64_t>> out;
    for (uint32_t fc = 0; fc < cols; ++fc) {
        if (row_of[fc] >= 0) continue;
        std::vector<uint64_t> v(cols, 0);
        v[fc] = 1;
        for (uint32_t c = 0; c < cols; ++c)
            if (row_of[c] >= 0) v[c] = (ell - A[static_cast<size_t>(row_of[c]) * cols + fc]) % ell;
        out.push_back(std::move(v));
    }
    if (basis) *basis = std::move(out);
    return static_cast<uint32_t>(piv.size());
}

engine::DenseBuf<uint64_t> JMatrix(uint32_t n, uint64_t ell, uint32_t E) {
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, E);
    const uint64_t m = ctx.p;
    const FlatMatrixT<uint64_t> S = ComputeSymmetricPowerMatrixRing<uint64_t>(ctx, 0, m - 1, 1, 0, n);
    const FlatMatrixT<uint64_t> T = ComputeSymmetricPowerMatrixRing<uint64_t>(ctx, 1, 1, 0, 1, n);
    const uint32_t d = n + 1;
    engine::DenseBuf<uint64_t> J(2 * d, d);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j) {
            const uint64_t one = (i == j) ? 1 : 0;
            // (slot-12 WP-11, A3) the sums in u128: m may lie in (2^63, 2^64).
            J.at(i, j) = static_cast<uint64_t>((static_cast<u128>(S.get_entry(i, j)) + m - one) % m);
            J.at(d + i, j) = static_cast<uint64_t>((static_cast<u128>(T.get_entry(i, j)) + m - one) % m);
        }
    return J;
}

uint32_t KernelLength(uint32_t n, uint64_t ell, uint32_t e) {
    const uint32_t d = n + 1;
    if (e == 1) {
        const engine::DenseBuf<uint64_t> J = JMatrix(n, ell, 2);
        std::vector<uint64_t> A(static_cast<size_t>(2) * d * d);
        for (uint32_t i = 0; i < 2 * d; ++i)
            for (uint32_t j = 0; j < d; ++j) A[static_cast<size_t>(i) * d + j] = J.at(i, j) % ell;
        return d - RankModPrime(A.data(), 2 * d, d, ell);
    }
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, e);
    const engine::DenseBuf<uint64_t> J = JMatrix(n, ell, e);
    const auto ker = engine::elim::kernel_module(ctx, engine::cview(J));
    return engine::elim::length(ctx, ker);
}

// ---- (slot-13 WP-5) the digit route ------------------------------------------
// Base-ell digit vectors of length E (little-endian), arithmetic mod ell^E.
using Digits = std::vector<uint32_t>;

Digits DigitsOf(uint64_t v, uint64_t ell, uint32_t E) {
    Digits d(E, 0);
    for (uint32_t i = 0; i < E && v; ++i) { d[i] = static_cast<uint32_t>(v % ell); v /= ell; }
    return d;
}
Digits DigitsAdd(const Digits& a, const Digits& b, uint64_t ell) {
    Digits c(a.size());
    uint64_t carry = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const uint64_t t = uint64_t(a[i]) + b[i] + carry;
        c[i] = static_cast<uint32_t>(t % ell);
        carry = t / ell;
    }
    return c;   // the carry out of digit E-1 is dropped: mod ell^E
}
Digits DigitsSub(const Digits& a, const Digits& b, uint64_t ell) {
    Digits c(a.size());
    int64_t borrow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        int64_t t = int64_t(a[i]) - int64_t(b[i]) - borrow;
        borrow = 0;
        if (t < 0) { t += static_cast<int64_t>(ell); borrow = 1; }
        c[i] = static_cast<uint32_t>(t);
    }
    return c;
}

void RequireDigitRoute(uint32_t n, uint64_t ell, uint32_t e) {
    if (ell >= (uint64_t(1) << 31)) {
        std::ostringstream o;
        o << "HeckeOrdinaryPart (torsion pre-pass): level ell^" << e << " at " << CellText(n, ell)
          << " needs the engine::ladic digit route, which requires ell < 2^31 (ENGINE_API.md Sec. 13.11.1) "
             "([C] prop:torsionprepass(3))";
        throw NotImplementedInThisBuild(o.str());
    }
}

engine::ladic::LadicMat<uint64_t> DigitJ(uint32_t n, uint64_t ell, uint32_t E, uint32_t m_override) {
    const std::vector<Digits> J = PrepassMatrixDigits(n, ell, E);
    const uint32_t d = n + 1;
    const auto lay = engine::ladic::DigitLayout<uint64_t>::make(ell, E, m_override);
    engine::ladic::LadicMat<uint64_t> M(lay, 2 * d, d);
    for (uint32_t i = 0; i < 2 * d; ++i)
        for (uint32_t j = 0; j < d; ++j) M.set_ell_digits(i, j, J[static_cast<size_t>(i) * d + j]);
    return M;
}

engine::ladic::Options DigitOptions(const PrepassOptions& opt) {
    engine::ladic::Options o;
    // A forced digit route runs the induction even at L = 1 (else L = 1 is the
    // Sec. 13.5 dispatch, i.e. the native route again).
    o.force_induction = (opt.route == PrepassRoute::Ladic);
    return o;
}

bool UseNative(const PrepassOptions& opt, uint64_t ell, uint32_t level) {
    if (opt.route == PrepassRoute::Native) return true;
    if (opt.route == PrepassRoute::Ladic) return false;
    return engine::scalar::powfit(ell, level) != 0;
}

} // namespace

std::vector<std::vector<uint32_t>> PrepassMatrixDigits(uint32_t n, uint64_t ell, uint32_t E) {
    ENGINE_CHECK_MSG(n >= 1 && E >= 1 && ell >= 2 && ell < (uint64_t(1) << 32),
                     "HeckeOrdinaryPart: PrepassMatrixDigits needs n >= 1, E >= 1, 2 <= ell < 2^32");
    const uint32_t d = n + 1;
    const Digits zero(E, 0), one = DigitsOf(1, ell, E), minus_one = DigitsSub(zero, one, ell);
    // rho_n(T): column j = coefficients of x^{n-j} (x + y)^j, entry (t, j) = binom(j, t)
    // (Pascal's rule, exact, mod ell^E); rho_n(S): entry (n - j, j) = (-1)^j
    // ([C] eq:rhondef; the native JMatrix's ComputeSymmetricPowerMatrixRing convention).
    std::vector<Digits> binom_prev(1, one);   // row j = 0: binom(0, 0)
    std::vector<std::vector<Digits>> T(d, std::vector<Digits>(d, zero));
    for (uint32_t j = 0; j < d; ++j) {
        std::vector<Digits> row(j + 1, zero);
        if (j == 0) row[0] = one;
        else
            for (uint32_t t = 0; t <= j; ++t) {
                const Digits& a = (t >= 1) ? binom_prev[t - 1] : zero;
                const Digits& b = (t < j) ? binom_prev[t] : zero;
                row[t] = DigitsAdd(a, b, ell);
            }
        for (uint32_t t = 0; t <= j; ++t) T[t][j] = row[t];
        binom_prev = row;
    }
    std::vector<Digits> J(static_cast<size_t>(2) * d * d, zero);
    for (uint32_t j = 0; j < d; ++j) J[static_cast<size_t>(n - j) * d + j] = (j % 2 == 0) ? one : minus_one;   // rho_n(S)
    for (uint32_t i = 0; i < d; ++i) J[static_cast<size_t>(i) * d + i] = DigitsSub(J[static_cast<size_t>(i) * d + i], one, ell);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j)
            J[static_cast<size_t>(d + i) * d + j] = (i == j) ? DigitsSub(T[i][j], one, ell) : T[i][j];
    return J;
}

uint32_t PrepassKernelLength(uint32_t n, uint64_t ell, uint32_t e, const PrepassOptions& opt) {
    ENGINE_CHECK_MSG(n >= 1 && e >= 1, "HeckeOrdinaryPart: PrepassKernelLength needs n >= 1, e >= 1");
    if (e == 1) {
        if (opt.route != PrepassRoute::Ladic) { RequireLevel(n, ell, 2); return KernelLength(n, ell, 1); }
        RequireDigitRoute(n, ell, 1);
        const uint32_t d = n + 1;
        const std::vector<Digits> J = PrepassMatrixDigits(n, ell, 1);
        std::vector<uint64_t> A(J.size());
        for (size_t i = 0; i < J.size(); ++i) A[i] = J[i][0];
        return d - RankModPrime(A.data(), 2 * d, d, ell);
    }
    if (UseNative(opt, ell, e)) { RequireLevel(n, ell, e); return KernelLength(n, ell, e); }
    RequireDigitRoute(n, ell, e);
    const auto J = DigitJ(n, ell, e, opt.digit_exp_override);
    return engine::ladic::length(engine::ladic::kernel_module(J, DigitOptions(opt)));
}

std::vector<uint32_t> PrepassTorsionAtLevel(uint32_t n, uint64_t ell, uint32_t E, const PrepassOptions& opt) {
    ENGINE_CHECK_MSG(n >= 1 && E >= 2, "HeckeOrdinaryPart: PrepassTorsionAtLevel needs n >= 1, E >= 2");
    std::vector<uint32_t> inv;
    if (UseNative(opt, ell, E)) {
        RequireLevel(n, ell, E);
        const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, E);
        const engine::DenseBuf<uint64_t> J = JMatrix(n, ell, E);
        inv = engine::elim::invariant_factors(ctx, engine::cview(J));
    } else {
        RequireDigitRoute(n, ell, E);
        inv = engine::ladic::invariant_factors(DigitJ(n, ell, E, opt.digit_exp_override), DigitOptions(opt));
    }
    ENGINE_CHECK_MSG(inv.size() == n + 1,
                     "HeckeOrdinaryPart: r_E(J_n) != n + 1 at a level above e_max (prop:torsionprepass(3))");
    std::vector<uint32_t> tors;
    for (uint32_t x : inv) {
        const uint32_t b = E - x;
        if (b > 0) tors.push_back(b);
    }
    std::sort(tors.begin(), tors.end(), [](uint32_t a, uint32_t b) { return a > b; });
    return tors;
}

// ---------------------------------------------------------------------------
std::vector<uint64_t> PrepassMatrix(uint32_t n, uint64_t ell, uint32_t E) {
    ENGINE_CHECK_MSG(n >= 1, "HeckeOrdinaryPart: the torsion pre-pass needs n >= 1 ([C] prop:torsionprepass; "
                             "H^0(Gamma; Z) != 0 at n = 0)");
    ENGINE_CHECK(E >= 2);
    RequireLevel(n, ell, E);
    const engine::DenseBuf<uint64_t> J = JMatrix(n, ell, E);
    std::vector<uint64_t> out;
    for (uint32_t i = 0; i < J.rows; ++i)
        for (uint32_t j = 0; j < J.cols; ++j) out.push_back(J.at(i, j));
    return out;
}

TorsionPrepass TorsionExponentPrepass(uint32_t n, uint64_t ell) {
    return TorsionExponentPrepass(n, ell, PrepassOptions{});
}

TorsionPrepass TorsionExponentPrepass(uint32_t n, uint64_t ell, const PrepassOptions& opt) {
    ENGINE_CHECK_MSG(n >= 1, "HeckeOrdinaryPart: the torsion pre-pass needs n >= 1 ([C] prop:torsionprepass; "
                             "H^0(Gamma; Z) != 0 at n = 0)");
    ENGINE_CHECK_MSG(ell >= 2, "HeckeOrdinaryPart: ell must be a prime");
    TorsionPrepass t;
    t.n_sym = n;
    t.ell = ell;
    t.lambda.push_back(0);
    // The stopping rule, prop:torsionprepass(3): the first e with lambda_{e+1} == lambda_e.
    // (slot-13 WP-5) each level on the route of `opt` (Auto: native where ell^e fits
    // uint64_t -- the slot-12 code path -- and the digit route beyond).
    t.lambda.push_back(PrepassKernelLength(n, ell, 1, opt));
    uint32_t e = 0;
    while (true) {
        if (t.lambda[e + 1] == t.lambda[e]) break;
        ++e;
        t.lambda.push_back(PrepassKernelLength(n, ell, e + 1, opt));
        ENGINE_CHECK_MSG(t.lambda[e + 1] - t.lambda[e] <= t.lambda[e] - t.lambda[e - 1],
                         "HeckeOrdinaryPart: lambda_{e+1} - lambda_e is not non-increasing (prop:torsionprepass(3))");
    }
    t.e_max = e;
    // The torsion multiset from the Smith rule at Z/ell^{e_max + 1} (the second form of the rule).
    const uint32_t E = std::max<uint32_t>(2, e + 1);
    t.torsion = PrepassTorsionAtLevel(n, ell, E, opt);   // ENGINE_CHECKs r_E(J_n) == n + 1
    t.rank_full_level = n + 1;
    ENGINE_CHECK_MSG(t.torsion.empty() ? t.e_max == 0 : t.torsion.front() == t.e_max,
                     "HeckeOrdinaryPart: max b_i != e_max (prop:torsionprepass(2)-(3))");
    return t;
}

std::vector<std::vector<uint64_t>> PrepassKernelGenerators(uint32_t n, uint64_t ell, uint32_t e) {
    ENGINE_CHECK_MSG(n >= 1 && e >= 1, "HeckeOrdinaryPart: n >= 1, e >= 1");
    RequireLevel(n, ell, std::max<uint32_t>(2, e));
    const uint32_t d = n + 1;
    std::vector<std::vector<uint64_t>> out;
    if (e == 1) {
        const engine::DenseBuf<uint64_t> J = JMatrix(n, ell, 2);
        std::vector<uint64_t> A(static_cast<size_t>(2) * d * d);
        for (uint32_t i = 0; i < 2 * d; ++i)
            for (uint32_t j = 0; j < d; ++j) A[static_cast<size_t>(i) * d + j] = J.at(i, j) % ell;
        FieldNullspace(A, 2 * d, d, ell, &out);
        return out;
    }
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, e);
    const engine::DenseBuf<uint64_t> J = JMatrix(n, ell, e);
    const auto ker = engine::elim::kernel_module(ctx, engine::cview(J));
    for (uint32_t i = 0; i < ker.rows(); ++i) {
        std::vector<uint64_t> v(d);
        for (uint32_t j = 0; j < d; ++j) v[j] = ker.H.at(i, j);
        out.push_back(v);
    }
    return out;
}

std::vector<uint64_t> ApplyTpOnH0(uint32_t n, uint64_t ell, uint32_t e, uint64_t p, const std::vector<uint64_t>& v) {
    ENGINE_CHECK_MSG(p != ell && p >= 2, "HeckeOrdinaryPart: ApplyTpOnH0 needs a good prime p != ell");
    const uint32_t E = std::max<uint32_t>(2, e);
    RequireLevel(n, ell, E);
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, E);
    const uint64_t M = ctx.p, m = engine::scalar::powfit(ell, e);
    const uint32_t d = n + 1;
    ENGINE_CHECK(v.size() == d);
    std::vector<uint64_t> out(d, 0);
    auto add = [&](uint64_t a, uint64_t b, uint64_t c, uint64_t dd) {
        const FlatMatrixT<uint64_t> R = ComputeSymmetricPowerMatrixRing<uint64_t>(ctx, a % M, b % M, c % M, dd % M, n);
        for (uint32_t i = 0; i < d; ++i) {
            // (slot-12 WP-11, A3) accumulate in u128: M may lie in (2^63, 2^64).
            u128 s = 0;
            for (uint32_t j = 0; j < d; ++j) s = (s + engine::scalar::mulmod(R.get_entry(i, j), v[j] % M, M)) % M;
            out[i] = static_cast<uint64_t>((static_cast<u128>(out[i]) + s) % M);
        }
    };
    for (uint64_t b = 0; b < p; ++b) add(p, b, 0, 1);
    add(1, 0, 0, p);
    for (uint64_t& x : out) x %= m;
    return out;
}

uint32_t FreeRankH1(uint32_t n) {
    ENGINE_CHECK_MSG(n >= 1, "HeckeOrdinaryPart: FreeRankH1 needs n >= 1 ([C] cor:ordprecision)");
    if (n % 2) return 0;
    const uint32_t k = n + 2;
    const uint32_t dk = k / 12 + (k % 12 == 2 ? 0u : 1u);
    return 2 * dk - 1;
}

uint32_t CertifiedWorkingPrecision(uint32_t n, uint64_t ell, uint32_t e_target) {
    RequirePrecisionUse(n, ell, e_target);
    return e_target + TorsionExponentPrepass(n, ell).e_max;
}

void RequirePrecisionUse(uint32_t n, uint64_t ell, uint32_t e_target) {
    ENGINE_CHECK_MSG(ell >= 5,
                     "HeckeOrdinaryPart: the torsion pre-pass is not a precision for the model at ell in {2, 3} "
                     "([C] prop:torsionprepass(6); cor:ordprecision is not asserted there) -- refused");
    ENGINE_CHECK_MSG(n >= 1, "HeckeOrdinaryPart: cor:ordprecision needs n >= 1 ([C] prop:torsionprepass)");
    ENGINE_CHECK_MSG(e_target >= 1, "HeckeOrdinaryPart: the target precision e' must be >= 1 ([C] cor:ordprecision)");
}

// ---------------------------------------------------------------------------
OrdinaryExponent OrdinaryExponentOf(uint32_t g, uint32_t K, uint64_t ell) {
    ENGINE_CHECK_MSG(g >= 1 && K >= 1 && ell >= 2, "HeckeOrdinaryPart: N_ord(g, K) needs g, K >= 1 ([C] lem:ordexponent)");
    OrdinaryExponent N;
    N.ell = ell;
    N.g = g;
    N.K = K;
    u128 t = 1;
    while (t < g) { t *= ell; ++N.c; }
    return N;
}

double OrdinaryExponent::log2() const {
    double s = static_cast<double>(ell_power()) * std::log2(static_cast<double>(ell));
    for (uint32_t f = 1; f <= g; ++f) s += static_cast<double>(f) * std::log2(static_cast<double>(ell));   // upper bound
    return s;
}

bool OrdinaryExponent::fits_u128() const {
    u128 v = 1;
    const u128 lim = ~static_cast<u128>(0);
    auto mul = [&](u128 x) { if (x != 0 && v > lim / x) return false; v *= x; return true; };
    for (uint32_t i = 0; i < ell_power(); ++i) if (!mul(ell)) return false;
    for (uint32_t f = 1; f <= g; ++f) {
        u128 q = 1;
        for (uint32_t i = 0; i < f; ++i) { if (q > lim / ell) return false; q *= ell; }
        if (!mul(q - 1)) return false;
    }
    return true;
}

u128 OrdinaryExponent::value() const {
    ENGINE_CHECK_MSG(fits_u128(), "HeckeOrdinaryPart: N_ord does not fit u128 (use its factors)");
    u128 v = 1;
    for (uint32_t i = 0; i < ell_power(); ++i) v *= ell;
    for (uint32_t f = 1; f <= g; ++f) {
        u128 q = 1;
        for (uint32_t i = 0; i < f; ++i) q *= ell;
        v *= (q - 1);
    }
    return v;
}

// ---------------------------------------------------------------------------
RingMat Identity(uint64_t ell, const std::vector<uint32_t>& row_exp) {
    RingMat I;
    I.ell = ell;
    I.row_exp = row_exp;
    const uint32_t n = I.n();
    I.a.assign(static_cast<size_t>(n) * n, 0);
    for (uint32_t i = 0; i < n; ++i) I.a[static_cast<size_t>(i) * n + i] = row_exp[i] ? 1 : 0;
    return I;
}

RingMat Canonical(RingMat M) {
    const uint32_t n = M.n();
    ENGINE_CHECK(M.a.size() == static_cast<size_t>(n) * n);
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t m = engine::scalar::powfit(M.ell, M.row_exp[i]);
        ENGINE_CHECK_MSG(m != 0, "HeckeOrdinaryPart: RingMat modulus beyond uint64_t");
        for (uint32_t j = 0; j < n; ++j) M.a[static_cast<size_t>(i) * n + j] %= m;
    }
    return M;
}

RingMat Mul(const RingMat& A, const RingMat& B) {
    ENGINE_CHECK_MSG(A.ell == B.ell && A.row_exp == B.row_exp, "HeckeOrdinaryPart: Mul on different modules");
    const uint32_t n = A.n();
    RingMat C = A;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t m = engine::scalar::powfit(A.ell, A.row_exp[i]);
        ENGINE_CHECK(m != 0);
        for (uint32_t j = 0; j < n; ++j) {
            u128 s = 0;
            for (uint32_t k = 0; k < n; ++k) s = (s + static_cast<u128>(A.at(i, k)) * B.at(k, j)) % m;
            C.a[static_cast<size_t>(i) * n + j] = static_cast<uint64_t>(s);
        }
    }
    return C;
}

RingMat Sub(const RingMat& A, const RingMat& B) {
    ENGINE_CHECK_MSG(A.ell == B.ell && A.row_exp == B.row_exp, "HeckeOrdinaryPart: Sub on different modules");
    const uint32_t n = A.n();
    RingMat C = A;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t m = engine::scalar::powfit(A.ell, A.row_exp[i]);
        for (uint32_t j = 0; j < n; ++j) C.a[static_cast<size_t>(i) * n + j] = static_cast<uint64_t>((static_cast<u128>(A.at(i, j) % m) + m - B.at(i, j) % m) % m);
    }
    return C;
}

RingMat PowU128(const RingMat& A, u128 e) {
    RingMat r = Identity(A.ell, A.row_exp), b = A;
    while (e) {
        if (e & 1) r = Mul(r, b);
        e >>= 1;
        if (e) b = Mul(b, b);
    }
    return r;
}

RingMat Pow(const RingMat& A, const OrdinaryExponent& N) {
    ENGINE_CHECK(A.ell == N.ell);
    RingMat X = A;
    for (uint32_t i = 0; i < N.ell_power(); ++i) X = PowU128(X, N.ell);
    for (uint32_t f = 1; f <= N.g; ++f) {
        // X^{ell^f - 1} = prod_{i < f} (X^{ell - 1})^{ell^i}
        RingMat Y = PowU128(X, N.ell - 1), acc = Identity(A.ell, A.row_exp);
        for (uint32_t i = 0; i < f; ++i) {
            acc = Mul(acc, Y);
            if (i + 1 < f) Y = PowU128(Y, N.ell);
        }
        X = acc;
    }
    return X;
}

RingMat Reduce(const RingMat& A, uint32_t K) {
    RingMat B = A;
    for (uint32_t& x : B.row_exp) x = std::min(x, K);
    return Canonical(B);
}

bool IsZero(const RingMat& A) {
    const RingMat C = Canonical(A);
    for (uint64_t x : C.a) if (x) return false;
    return true;
}

uint32_t RankModEll(const RingMat& A) {
    const uint32_t n = A.n();
    std::vector<uint64_t> M(A.a.size());
    for (size_t i = 0; i < M.size(); ++i) M[i] = A.a[i] % A.ell;
    return RankModPrime(M.data(), n, n, A.ell);
}

RingMat OrdinaryProjector(const RingMat& U) {
    const uint32_t g = U.n();
    if (g == 0) return U;
    const uint32_t K = *std::max_element(U.row_exp.begin(), U.row_exp.end());
    return Pow(U, OrdinaryExponentOf(g, std::max<uint32_t>(1, K), U.ell));
}

} // namespace ordinary
} // namespace hecke
