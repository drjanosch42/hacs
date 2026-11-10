// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineLadicInductionValidation.cpp -- see the header.
//
// hecke-extensions slot 11, WP-18b: engine::ladic (the (C2) ell-adic digit
// induction, ENGINE_API.md Sec. 13.11) against the width oracle matrix of
// Sec. 13.11.7.
#include "EngineLadicInductionValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/DenseView.h"
#include "../ModArithmetic/LadicInduction.h"
#include "../ModArithmetic/LadicRef.h"
#include "../ModArithmetic/ModCtx.h"
#include "EquivariantCohomologyOnK.h"
#include "Hecke/CoefficientRing.h"
#include "Hecke/HeckeSettings.h"
#include "LocalSystemSL2Z.h"
#include "LocalSystemSimplicialComplex.h"

namespace {

using engine::u128;
namespace lad = engine::ladic;
namespace rl = engine::ref::ladic;

struct Checker {
    uint32_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (fails <= 40) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

std::string Mod(uint64_t ell, uint32_t K) { return std::to_string(ell) + "^" + std::to_string(K); }
// An arm summary: printed whether it passes or not (the report reads these lines).
void Summ(Checker& g, bool ok, const std::string& what) {
    if (ok) std::cout << "    ok  " << what << "\n";
    g.check(ok, what);
}

// ell^K as u128 (only called where it fits)
u128 PowU128(uint64_t ell, uint32_t K) {
    u128 p = 1;
    for (uint32_t i = 0; i < K; ++i) p *= ell;
    return p;
}
bool FitsBits(uint64_t ell, uint32_t K, unsigned bits) {   // ell^K <= 2^bits - 1
    u128 p = 1;
    const u128 lim = (bits >= 128) ? ~u128(0) : ((u128(1) << bits) - 1);
    for (uint32_t i = 0; i < K; ++i) {
        if (p > lim / ell) return false;
        p *= ell;
    }
    return p <= lim;
}

// ---------------------------------------------------------------------------
// Inputs: width-independent, as base-ell digit matrices (the ref oracle's).
// ---------------------------------------------------------------------------
rl::Elem RandElem(const rl::Ring& R, std::mt19937_64& rng) {
    const uint32_t roll = static_cast<uint32_t>(rng() % 100);
    uint32_t v;
    if (roll < 45) v = 0;
    else if (roll < 82) v = 1 + static_cast<uint32_t>(rng() % (R.K > 1 ? R.K - 1 : 1));
    else return R.zero();
    rl::Elem e(R.K, 0);
    if (v >= R.K) return e;
    e[v] = 1 + static_cast<uint32_t>(rng() % (R.ell - 1));
    for (uint32_t i = v + 1; i < R.K; ++i) e[i] = static_cast<uint32_t>(rng() % R.ell);
    return e;
}
rl::Elem RandUniform(const rl::Ring& R, std::mt19937_64& rng) {
    rl::Elem e(R.K);
    for (uint32_t i = 0; i < R.K; ++i) e[i] = static_cast<uint32_t>(rng() % R.ell);
    return e;
}
rl::Mat RandMat(const rl::Ring& R, uint32_t m, uint32_t n, std::mt19937_64& rng) {
    rl::Mat A(m, std::vector<rl::Elem>(n));
    for (auto& row : A)
        for (auto& e : row) e = RandElem(R, rng);
    return A;
}
rl::Mat ZeroMat(const rl::Ring& R, uint32_t m, uint32_t n) { return rl::Mat(m, std::vector<rl::Elem>(n, R.zero())); }
// A unimodular matrix: (unit lower) x (unit upper), rows permuted.
rl::Mat Unimodular(const rl::Ring& R, uint32_t n, std::mt19937_64& rng) {
    rl::Mat Lw = ZeroMat(R, n, n), Up = ZeroMat(R, n, n);
    for (uint32_t i = 0; i < n; ++i) {
        Lw[i][i] = R.one();
        rl::Elem u = RandUniform(R, rng);
        if (u[0] == 0) u[0] = 1;
        Up[i][i] = u;
        for (uint32_t j = 0; j < i; ++j) Lw[i][j] = RandUniform(R, rng);
        for (uint32_t j = i + 1; j < n; ++j) Up[i][j] = RandUniform(R, rng);
    }
    rl::Mat P = rl::multiply(R, Lw, Up);
    std::shuffle(P.begin(), P.end(), rng);
    return P;
}

struct Case {
    std::string name;
    uint32_t m = 0, n = 0;
    rl::Mat A, G, Rl, X, Ybad, XG;
    rl::Mat Y, Yc;                      // A X (in the image), G XG (in colspan N)
    std::vector<uint32_t> planted;      // non-empty: the planted Smith exponents (ascending, < K)
};

void Finish(const rl::Ring& R, Case& c, std::mt19937_64& rng) {
    c.G = RandMat(R, c.m, 3, rng);
    c.Rl = RandMat(R, c.m, 2, rng);
    c.X = RandMat(R, c.n, 3, rng);
    c.Ybad = RandMat(R, c.m, 2, rng);
    c.XG = RandMat(R, 3, 2, rng);
    c.Y = rl::multiply(R, c.A, c.X);
    c.Yc = rl::multiply(R, c.G, c.XG);
}

std::vector<Case> Families(const rl::Ring& R, uint64_t seed, bool small) {
    std::mt19937_64 rng(seed);
    std::vector<Case> cs;
    auto add = [&](const std::string& nm, rl::Mat A) {
        Case c;
        c.name = nm;
        c.m = static_cast<uint32_t>(A.size());
        c.n = c.m ? static_cast<uint32_t>(A[0].size()) : 0;
        c.A = std::move(A);
        Finish(R, c, rng);
        cs.push_back(std::move(c));
    };
    const uint32_t a = small ? 4 : 7, b = small ? 5 : 9;
    add("random " + std::to_string(a) + "x" + std::to_string(b), RandMat(R, a, b, rng));
    add("tall", RandMat(R, small ? 6 : 10, small ? 3 : 5, rng));
    add("wide", RandMat(R, 3, small ? 6 : 10, rng));
    add("zero", ZeroMat(R, 4, 5));
    {
        rl::Mat I = ZeroMat(R, 5, 5);
        for (uint32_t i = 0; i < 5; ++i) I[i][i] = R.one();
        add("identity", I);
    }
    {
        rl::Mat A = RandMat(R, small ? 4 : 6, small ? 5 : 7, rng);
        for (auto& row : A)
            for (auto& e : row) e = R.shl(e, 1);
        add("ell-multiples", A);
    }
    {
        rl::Mat A = ZeroMat(R, 3, 4);
        A[0][0] = R.ell_pow(1); A[0][1] = R.one();
        A[1][1] = R.ell_pow(std::min<uint32_t>(2, R.K - 1)); A[1][2] = R.ell_pow(1); A[1][3] = R.one();
        A[2][0] = R.ell_pow(R.K - 1); A[2][3] = R.ell_pow(1);
        add("closure", A);
    }
    {
        const rl::Mat P = RandMat(R, small ? 5 : 8, 2, rng), Q = RandMat(R, 2, small ? 5 : 7, rng);
        add("rank-deficient", rl::multiply(R, P, Q));
    }
    {
        // planted: P diag(ell^s) Q with exponents spread over [0, K), plus zero diagonals
        const uint32_t m = small ? 5 : 7, n = small ? 5 : 8, r = small ? 4 : 5;
        std::vector<uint32_t> s;
        for (uint32_t i = 0; i < r; ++i) s.push_back(static_cast<uint32_t>(rng() % R.K));
        s[0] = 0;
        if (r > 1) s[1] = R.K - 1;
        std::sort(s.begin(), s.end());
        rl::Mat D = ZeroMat(R, m, n);
        for (uint32_t i = 0; i < r; ++i) D[i][i] = R.ell_pow(s[i]);
        const rl::Mat A = rl::multiply(R, rl::multiply(R, Unimodular(R, m, rng), D), Unimodular(R, n, rng));
        add("planted", A);
        cs.back().planted = s;
    }
    return cs;
}

// ---------------------------------------------------------------------------
// Records: named sections of serialised canonical data.
// ---------------------------------------------------------------------------
using Record = std::vector<std::pair<std::string, std::vector<uint32_t>>>;

template<class W> lad::LadicMat<W> ToLadic(const rl::Mat& A, uint32_t cols, const lad::DigitLayout<W>& lay) {
    const uint32_t m = static_cast<uint32_t>(A.size());
    lad::LadicMat<W> M(lay, m, cols);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < cols; ++j) M.set_ell_digits(i, j, A[i][j]);
    return M;
}
template<class N> engine::DenseBuf<N> ToNative(const rl::Mat& A, uint32_t cols, uint64_t ell) {
    const uint32_t m = static_cast<uint32_t>(A.size());
    engine::DenseBuf<N> B(m, cols);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < cols; ++j) {
            u128 v = 0;
            for (uint32_t e = static_cast<uint32_t>(A[i][j].size()); e-- > 0;) v = v * ell + A[i][j][e];
            B.at(i, j) = static_cast<N>(v);
        }
    return B;
}
std::vector<uint32_t> Cat(std::vector<uint32_t> a, const std::vector<uint32_t>& b) {
    a.push_back(0xFFFFFFFEu);
    a.insert(a.end(), b.begin(), b.end());
    return a;
}
template<class W> std::vector<uint32_t> SerH(const lad::Howell<W>& h) {
    return Cat(Cat(h.H.serialise(), h.pivots), h.pivot_v);
}
// native DenseBuf<N> -> the same serialisation, through a u64-width layout
template<class N> std::vector<uint32_t> SerN(const engine::DenseBuf<N>& B, uint64_t ell, uint32_t K) {
    const lad::DigitLayout<uint64_t> lay = lad::DigitLayout<uint64_t>::make(ell, K);
    return lad::LadicMat<uint64_t>::template from_word<N>(lay, engine::cview(B)).serialise();
}
template<class N> std::vector<uint32_t> SerNH(const engine::elim::HowellResult<N>& h, uint64_t ell, uint32_t K) {
    return Cat(Cat(SerN<N>(h.H, ell, K), h.pivots), h.pivot_v);
}

template<class W>
Record LadicRecord(const Case& c, const lad::DigitLayout<W>& lay, const lad::Options& o) {
    Record r;
    const lad::LadicMat<W> A = ToLadic<W>(c.A, c.n, lay), G = ToLadic<W>(c.G, 3, lay), Rl = ToLadic<W>(c.Rl, 2, lay);
    const lad::LadicMat<W> Y = ToLadic<W>(c.Y, 3, lay), Yb = ToLadic<W>(c.Ybad, 2, lay), Yc = ToLadic<W>(c.Yc, 2, lay);
    r.emplace_back("smith_s", lad::smith_lift(A, o).s);
    r.emplace_back("howell", SerH(lad::howell(A, o)));
    r.emplace_back("kernel", SerH(lad::kernel_module(A, o)));
    r.emplace_back("image", SerH(lad::image_module(A, o)));
    r.emplace_back("length", std::vector<uint32_t>{lad::length(A, o)});
    r.emplace_back("invf_A", lad::invariant_factors(A, o));
    r.emplace_back("invf_GRl", lad::invariant_factors(G, Rl, o));
    r.emplace_back("solve", lad::solve_multi_rhs_module(A, Y, o).serialise());
    {
        std::vector<uint32_t> s;
        try { s = lad::solve_multi_rhs_module(A, Yb, o).serialise(); s.push_back(0xFFFFFFFFu); }
        catch (const std::runtime_error& e) { s = {0xABCDu}; const std::string w = e.what(); s.insert(s.end(), w.begin(), w.end()); }
        r.emplace_back("solve_bad", s);
    }
    const lad::Cyclic<W> d = lad::cyclic_decomposition(G, Rl, o);
    r.emplace_back("cyc_exp", d.exponents);
    r.emplace_back("cyc_pos", d.position);
    r.emplace_back("cyc_basis", d.basis.serialise());
    r.emplace_back("cyc_gamma", d.Gamma.serialise());
    r.emplace_back("cyc_U", d.U.serialise());
    r.emplace_back("cyc_coords", lad::cyclic_coordinates(d, Yc, o).serialise());
    // the same with (A, Rl) as (G, Rl): the pipeline's shape, H = Z / B
    const lad::Cyclic<W> d2 = lad::cyclic_decomposition(A.transpose(), lad::LadicMat<W>(lay, c.n, 0), o);
    r.emplace_back("cyc2_exp", d2.exponents);
    r.emplace_back("cyc2_basis", d2.basis.serialise());
    r.emplace_back("cyc2_U", d2.U.serialise());
    return r;
}

template<class N>
Record NativeRecord(const Case& c, uint64_t ell, uint32_t K) {
    namespace el = engine::elim;
    Record r;
    const engine::ModCtx<N> ctx = engine::ModCtx<N>::prime_power(static_cast<N>(ell), K);
    const engine::DenseBuf<N> A = ToNative<N>(c.A, c.n, ell), G = ToNative<N>(c.G, 3, ell), Rl = ToNative<N>(c.Rl, 2, ell);
    const engine::DenseBuf<N> Y = ToNative<N>(c.Y, 3, ell), Yb = ToNative<N>(c.Ybad, 2, ell), Yc = ToNative<N>(c.Yc, 2, ell);
    using engine::cview;
    {
        std::vector<uint32_t> e = el::invariant_factors<N>(ctx, cview(A)), s;
        for (uint32_t x : e) s.push_back(K - x);
        std::sort(s.begin(), s.end());
        r.emplace_back("smith_s", s);
    }
    r.emplace_back("howell", SerNH<N>(el::howell<N>(ctx, cview(A)), ell, K));
    r.emplace_back("kernel", SerNH<N>(el::kernel_module<N>(ctx, cview(A)), ell, K));
    r.emplace_back("image", SerNH<N>(el::image_module<N>(ctx, cview(A)), ell, K));
    r.emplace_back("length", std::vector<uint32_t>{el::length<N>(ctx, cview(A))});
    r.emplace_back("invf_A", el::invariant_factors<N>(ctx, cview(A)));
    r.emplace_back("invf_GRl", el::invariant_factors<N>(ctx, cview(G), cview(Rl)));
    auto ser_solve = [&](const std::vector<std::vector<N>>& x, uint32_t ny) {
        engine::DenseBuf<N> X(ny, c.n);
        for (uint32_t t = 0; t < ny; ++t)
            for (uint32_t j = 0; j < c.n; ++j) X.at(t, j) = x[t][j];
        return SerN<N>(X, ell, K);
    };
    r.emplace_back("solve", ser_solve(el::solve_multi_rhs_module<N>(ctx, cview(A), cview(Y)), 3));
    {
        std::vector<uint32_t> s;
        try { s = ser_solve(el::solve_multi_rhs_module<N>(ctx, cview(A), cview(Yb)), 2); s.push_back(0xFFFFFFFFu); }
        catch (const std::runtime_error& e) { s = {0xABCDu}; const std::string w = e.what(); s.insert(s.end(), w.begin(), w.end()); }
        r.emplace_back("solve_bad", s);
    }
    const el::CyclicDecomposition<N> d = el::cyclic_decomposition<N>(ctx, cview(G), cview(Rl));
    r.emplace_back("cyc_exp", d.exponents);
    r.emplace_back("cyc_pos", d.position);
    r.emplace_back("cyc_basis", SerN<N>(d.basis, ell, K));
    r.emplace_back("cyc_gamma", SerN<N>(d.Gamma, ell, K));
    r.emplace_back("cyc_U", SerN<N>(d.U, ell, K));
    {
        const std::vector<std::vector<N>> x = el::cyclic_coordinates<N>(ctx, d, cview(Yc));
        engine::DenseBuf<N> X(2, d.rows());
        for (uint32_t t = 0; t < 2; ++t)
            for (uint32_t i = 0; i < d.rows(); ++i) X.at(t, i) = x[t][i];
        r.emplace_back("cyc_coords", SerN<N>(X, ell, K));
    }
    {
        engine::DenseBuf<N> At(c.n, c.m);
        for (uint32_t i = 0; i < c.m; ++i)
            for (uint32_t j = 0; j < c.n; ++j) At.at(j, i) = A.at(i, j);
        const engine::DenseBuf<N> none(c.n, 0);
        const el::CyclicDecomposition<N> d2 = el::cyclic_decomposition<N>(ctx, cview(At), cview(none));
        r.emplace_back("cyc2_exp", d2.exponents);
        r.emplace_back("cyc2_basis", SerN<N>(d2.basis, ell, K));
        r.emplace_back("cyc2_U", SerN<N>(d2.U, ell, K));
    }
    return r;
}

bool SameRecord(const Record& a, const Record& b, std::string& why) {
    if (a.size() != b.size()) { why = "section count"; return false; }
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].first != b[i].first || a[i].second != b[i].second) { why = a[i].first; return false; }
    return true;
}

// A LadicMat reduced mod ell as a native u32 matrix (digit 0 mod ell).
template<class W> uint32_t RankModEll(const lad::LadicMat<W>& M) {
    const uint32_t ell = static_cast<uint32_t>(M.ell());
    engine::DenseBuf<uint32_t> B(M.rows(), M.cols());
    const W* p0 = M.plane(0);
    for (uint32_t i = 0; i < M.rows(); ++i)
        for (uint32_t j = 0; j < M.cols(); ++j)
            B.at(i, j) = static_cast<uint32_t>(p0[static_cast<size_t>(i) * M.cols() + j] % ell);
    const engine::ModCtx<uint32_t> f(ell);
    return engine::elim::rref<uint32_t>(f, engine::view(B)).rank;
}

// (D) the Smith certificate: U A V = diag(ell^s) (+) 0 exactly; U, V unimodular.
template<class W>
void Certificate(Checker& g, const lad::LadicMat<W>& A, const lad::Options& o0, const std::string& tag) {
    lad::Options o = o0;
    o.transforms = true;
    const lad::Smith<W> sm = lad::smith_lift(A, o);
    const lad::LadicMat<W> D = lad::multiply(lad::multiply(sm.U, A), sm.V);
    lad::LadicMat<W> want(A.layout(), A.rows(), A.cols());
    for (uint32_t t = 0; t < sm.rank_K(); ++t) {
        std::vector<uint32_t> e(A.K(), 0);
        e[sm.s[t]] = 1;
        want.set_ell_digits(t, t, e);
    }
    g.check(D == want, "(D) " + tag + ": U A V == diag(ell^s) (+) 0 exactly");
    g.check(RankModEll(sm.U) == A.rows() && RankModEll(sm.V) == A.cols(), "(D) " + tag + ": U and V unimodular");
    // the T rows: row t = ell^{s_t} * (a 1 at pivot_col[t], 0 at earlier pivot columns)
    bool tshape = true;
    for (uint32_t t = 0; t < sm.rank_K(); ++t) {
        std::vector<uint32_t> e(A.K(), 0);
        e[sm.s[t]] = 1;
        tshape = tshape && sm.T.ell_digits(t, sm.pivot_col[t]) == e;
        for (uint32_t tp = 0; tp < t; ++tp)
            tshape = tshape && sm.T.ell_digits(t, sm.pivot_col[tp]) == std::vector<uint32_t>(A.K(), 0);
    }
    g.check(tshape, "(D) " + tag + ": T rows lead with exactly ell^{s_t}, zero at earlier pivot columns");
}

// ===========================================================================
// (A) the digit arithmetic
// ===========================================================================
template<class W>
std::vector<W> DigitsOf(const lad::Ring<W>& R, u128 v) {
    std::vector<W> d(R.L);
    for (uint32_t i = 0; i < R.L; ++i) { d[i] = static_cast<W>(v % R.B); v /= R.B; }
    return d;
}
template<class W>
u128 ValueOf(const lad::Ring<W>& R, const std::vector<W>& d) {
    u128 v = 0;
    for (uint32_t i = R.L; i-- > 0;) v = v * R.B + d[i];
    return v;
}
uint32_t ValU128(u128 x, uint64_t ell, uint32_t K) {
    if (x == 0) return K;
    uint32_t v = 0;
    while (x % ell == 0) { x /= ell; ++v; }
    return v;
}
template<class W>
void RingVsNative(Checker& g, uint64_t ell, uint32_t K, uint32_t m_override, const std::string& tag) {
    const lad::DigitLayout<W> lay = lad::DigitLayout<W>::make(ell, K, m_override);
    const lad::Ring<W> R(lay);
    const u128 p = PowU128(ell, K);
    std::mt19937_64 rng(0x1adc0000u ^ static_cast<uint32_t>(ell * 131 + K));
    uint32_t bad = 0, n = 0, throws_ok = 0, throws_n = 0;
    std::vector<W> z(R.L), w(R.L);
    for (uint32_t s = 0; s < 4000; ++s) {
        u128 a = ((static_cast<u128>(rng()) << 64) | rng()) % p, b = ((static_cast<u128>(rng()) << 64) | rng()) % p;
        if (s % 5 == 0) { a = (a / p) * p + (a % p); a = a - a % ell; }   // non-units too
        if (s % 7 == 0) b = p - 1;
        const std::vector<W> A = DigitsOf(R, a), B = DigitsOf(R, b);
        ++n;
        R.add(z.data(), A.data(), B.data());
        if (ValueOf(R, z) != (a + b) % p) ++bad;
        R.sub(z.data(), A.data(), B.data());
        if (ValueOf(R, z) != (a + p - b) % p) ++bad;
        R.mul(z.data(), A.data(), B.data());
        if (ValueOf(R, z) != (a * b) % p) ++bad;          // a, b < 2^64 here
        if (R.valuation(A.data()) != ValU128(a, ell, K)) ++bad;
        const uint32_t v = static_cast<uint32_t>(rng() % (K + 1));
        u128 ev = 1;
        for (uint32_t i = 0; i < v; ++i) ev *= ell;
        R.shr(z.data(), A.data(), v);
        if (ValueOf(R, z) != a / ev) ++bad;
        R.low(z.data(), A.data(), v);
        if (ValueOf(R, z) != a % ev) ++bad;
        R.shl(z.data(), A.data(), v);
        if (ValueOf(R, z) != (a * (ev % p)) % p) ++bad;
        const uint32_t c = static_cast<uint32_t>(rng() % K), k = 1 + static_cast<uint32_t>(rng() % lay.m);
        u128 ec = 1, ek = 1;
        for (uint32_t i = 0; i < c; ++i) ec *= ell;
        for (uint32_t i = 0; i < k; ++i) ek *= ell;
        if (static_cast<u128>(R.window(A.data(), c, k)) != (a / ec) % ek) ++bad;
        if (a % ell != 0) {
            R.inv(z.data(), A.data());
            if ((ValueOf(R, z) * a) % p != 1) ++bad;
        } else {
            ++throws_n;
            try { R.inv(z.data(), A.data()); } catch (const std::runtime_error&) { ++throws_ok; }
        }
    }
    g.check(bad == 0, "(A) " + tag + " Z/" + Mod(ell, K) + ": add/sub/mul/valuation/shr/low/shl/window/inv == native on " +
                          std::to_string(n) + " samples (" + std::to_string(bad) + " bad)");
    g.check(throws_ok == throws_n, "(A) " + tag + " Z/" + Mod(ell, K) + ": inv throws on every non-unit (" +
                                       std::to_string(throws_ok) + "/" + std::to_string(throws_n) + ")");
}
template<class W>
void RingVsRef(Checker& g, uint64_t ell, uint32_t K, uint32_t m_override, const std::string& tag) {
    const lad::DigitLayout<W> lay = lad::DigitLayout<W>::make(ell, K, m_override);
    const lad::Ring<W> R(lay);
    const rl::Ring Q(ell, K);
    std::mt19937_64 rng(0x2adc0000u ^ static_cast<uint32_t>(ell * 7 + K));
    uint32_t bad = 0, n = 0, thr = 0, thr_n = 0;
    std::vector<W> a(R.L), b(R.L), z(R.L);
    std::vector<uint32_t> out(K);
    auto eq = [&](const std::vector<W>& x, const rl::Elem& y) { R.to_ell_digits(out.data(), x.data()); return out == y; };
    for (uint32_t s = 0; s < 400; ++s) {
        const rl::Elem A = RandElem(Q, rng), B = (s % 3) ? RandUniform(Q, rng) : RandElem(Q, rng);
        R.from_ell_digits(a.data(), A.data());
        R.from_ell_digits(b.data(), B.data());
        ++n;
        R.add(z.data(), a.data(), b.data()); if (!eq(z, Q.add(A, B))) ++bad;
        R.sub(z.data(), a.data(), b.data()); if (!eq(z, Q.sub(A, B))) ++bad;
        R.mul(z.data(), a.data(), b.data()); if (!eq(z, Q.mul(A, B))) ++bad;
        R.neg(z.data(), a.data()); if (!eq(z, Q.neg(A))) ++bad;
        if (R.valuation(a.data()) != Q.valuation(A)) ++bad;
        const uint32_t v = static_cast<uint32_t>(rng() % (K + 1));
        R.shr(z.data(), a.data(), v); if (!eq(z, Q.shr(A, v))) ++bad;
        R.shl(z.data(), a.data(), v); if (!eq(z, Q.shl(A, v))) ++bad;
        if (A[0] != 0) {
            R.inv(z.data(), a.data());
            if (!eq(z, Q.inv(A))) ++bad;
        } else {
            ++thr_n;
            try { R.inv(z.data(), a.data()); } catch (const std::runtime_error&) { ++thr; }
        }
    }
    g.check(bad == 0, "(A) " + tag + " Z/" + Mod(ell, K) + ": ring == the base-ell oracle on " + std::to_string(n) +
                          " samples (" + std::to_string(bad) + " bad)");
    g.check(thr == thr_n, "(A) " + tag + " Z/" + Mod(ell, K) + ": inv throws on every non-unit");
}

// DigitWord<u128>::split (the normalised long division, WP-18b) against the
// restoring oracle split_restoring, on random and boundary operands.
void SplitArm(Checker& g) {
    using DW = engine::DigitWord<u128>;
    std::mt19937_64 rng(0x5b17u);
    auto r128 = [&] { return (static_cast<u128>(rng()) << 64) | rng(); };
    uint32_t bad = 0, n = 0;
    std::vector<u128> Ms = {1, 2, 3, 691, (u128(1) << 63) + 1, ~u128(0) >> 64, (u128(1) << 64) + 13, u128(1) << 100,
                            (u128(1) << 127) - 1, ~u128(0), PowU128(691, 13), PowU128(5, 54), PowU128(3, 80)};
    for (int i = 0; i < 400; ++i) Ms.push_back(r128() >> (rng() % 128));
    for (u128 M : Ms) {
        if (M == 0) continue;
        for (int t = 0; t < 250; ++t) {
            u128 hi = r128() % M, lo = r128();
            if (t == 0) { hi = M - 1; lo = ~u128(0); }
            if (t == 1) { hi = 0; lo = M - 1; }
            if (t == 2) { hi = M - 1; lo = 0; }
            u128 ra, rb;
            const u128 qa = DW::split(hi, lo, M, ra), qb = DW::split_restoring(hi, lo, M, rb);
            ++n;
            if (qa != qb || ra != rb || ra >= M) ++bad;
        }
    }
    g.check(bad == 0, "(A) DigitWord<u128>::split == the restoring oracle on " + std::to_string(n) + " (hi, lo, M) (" +
                          std::to_string(bad) + " bad)");
}

void LayoutArm(Checker& g) {
    using L32 = lad::DigitLayout<uint32_t>;
    using L64 = lad::DigitLayout<uint64_t>;
    g.check(L32::make(5, 20).m == 13 && L64::make(5, 20).m == 27 && L32::make(691, 6).m == 3 && L64::make(691, 6).m == 6 &&
                L32::make(134217757, 2).m == 1 && L64::make(134217757, 2).m == 2 && L32::make(2, 40).m == 30,
            "(A) layout: m(W, ell) as ENGINE_API.md Sec. 13.11.2 (5: 13 / 27, 691: 3 / 6, 134217757: 1 / 2, 2: 30)");
    g.check(L32::make(691, 20).L == 7 && L64::make(691, 20).L == 4 && L32::make(3, 100).L == 6 && L64::make(5, 60).L == 3,
            "(A) layout: L = ceil(K / m)");
    bool threw = false;
    try { (void)L32::make(5, 1); } catch (const std::logic_error&) { threw = true; }
    g.check(threw, "(A) layout: K = 1 refused (F_ell is ModCtx(ell))");
    threw = false;
    try { (void)L32::make(5, 10, 14); } catch (const std::logic_error&) { threw = true; }
    g.check(threw, "(A) layout: a digit exponent above m(W, ell) refused");
    // from_word / to_word round trip, truncation == native reduction
    std::mt19937_64 rng(77);
    const uint64_t ell = 691;
    const uint32_t K = 6;
    const u128 p = PowU128(ell, K);
    engine::DenseBuf<uint64_t> B(5, 4);
    for (uint32_t i = 0; i < 5; ++i)
        for (uint32_t j = 0; j < 4; ++j) B.at(i, j) = static_cast<uint64_t>(rng() % p);
    for (uint32_t mo : {0u, 1u, 2u}) {
        const auto M = lad::LadicMat<uint32_t>::from_word<uint64_t>(L32::make(ell, K, mo), engine::cview(B));
        const engine::DenseBuf<uint64_t> back = M.to_word<uint64_t>();
        bool same = true;
        for (uint32_t i = 0; i < 5; ++i)
            for (uint32_t j = 0; j < 4; ++j) same = same && back.at(i, j) == B.at(i, j);
        g.check(same, "(A) layout: from_word / to_word round trip at 691^6, m' = " + std::to_string(mo));
        for (uint32_t k = 2; k <= K; ++k) {
            const u128 pk = PowU128(ell, k);
            const engine::DenseBuf<uint64_t> t = M.truncate(k).to_word<uint64_t>();
            bool ok = true;
            for (uint32_t i = 0; i < 5; ++i)
                for (uint32_t j = 0; j < 4; ++j) ok = ok && t.at(i, j) == static_cast<uint64_t>(B.at(i, j) % pk);
            g.check(ok, "(A) layout: truncate(" + std::to_string(k) + ") == reduction mod 691^k, m' = " + std::to_string(mo));
        }
        const auto M64 = lad::convert<uint64_t>(M);
        g.check(M64.serialise() == M.serialise() && M64.to_word<uint64_t>().at(3, 2) == B.at(3, 2),
                "(A) layout: convert<u64> preserves the serialisation");
    }
}

// ===========================================================================
// (B) / (C) / (E) the elimination arms
// ===========================================================================
struct Variant {
    std::string name;
    unsigned width;       // 32, 64 or 128
    uint32_t m_override;
    lad::Options opt;
};

template<class W>
Record RunVariant(const Case& c, uint64_t ell, uint32_t K, const Variant& v) {
    return LadicRecord<W>(c, lad::DigitLayout<W>::make(ell, K, v.m_override), v.opt);
}
Record Run(const Case& c, uint64_t ell, uint32_t K, const Variant& v) {
    if (v.width == 32) return RunVariant<uint32_t>(c, ell, K, v);
    if (v.width == 64) return RunVariant<uint64_t>(c, ell, K, v);
    return RunVariant<u128>(c, ell, K, v);
}

std::vector<Variant> Variants(uint64_t ell, uint32_t K, bool cheap) {
    std::vector<Variant> vs;
    vs.push_back({"w32", 32, 0, {}});
    vs.push_back({"w64", 64, 0, {}});
    lad::Options fi;
    fi.force_induction = true;
    vs.push_back({"w64 force_induction", 64, 0, fi});
    vs.push_back({"w32 force_induction", 32, 0, fi});
    lad::Options le;
    le.level_exp = 1;
    if (!cheap || K <= 30) vs.push_back({"w32 level_exp=1", 32, 0, le});
    const uint32_t m32 = lad::DigitLayout<uint32_t>::max_digit_exp(ell);
    if (m32 >= 2) vs.push_back({"w32 m'=" + std::to_string((m32 + 1) / 2), 32, (m32 + 1) / 2, {}});
    const uint32_t m1 = 1;
    if (!cheap || K <= 40) vs.push_back({"w32 m'=1 (L = K)", 32, m1, {}});
    const uint32_t m64 = lad::DigitLayout<uint64_t>::max_digit_exp(ell);
    if (m64 >= 3) vs.push_back({"w64 m'=" + std::to_string(m64 / 3), 64, m64 / 3, {}});
    if (ell % 2 == 1) {   // the u128 width is U128_Montgomery: ell odd
        vs.push_back({"w128", 128, 0, {}});
        vs.push_back({"w128 force_induction", 128, 0, fi});
        if (!cheap || K <= 30) vs.push_back({"w128 level_exp=1", 128, 0, le});
        const uint32_t m128 = lad::DigitLayout<u128>::max_digit_exp(ell);
        if (m128 >= 4) vs.push_back({"w128 m'=" + std::to_string(m128 / 4), 128, m128 / 4, {}});
    }
    return vs;
}

// (B): 2 x 32 (and every variant) vs native u64 (and u32 below 2^32)
void NativeArm(Checker& g, uint64_t ell, uint32_t K, uint64_t seed) {
    const rl::Ring R(ell, K);
    const std::vector<Case> cases = Families(R, seed, false);
    const std::vector<Variant> vs = Variants(ell, K, false);
    const bool fits32 = FitsBits(ell, K, 32);
    uint32_t n_ok = 0, n = 0;
    for (const Case& c : cases) {
        const std::string tag = "Z/" + Mod(ell, K) + " " + c.name;
        Record nat;
        try { nat = NativeRecord<uint64_t>(c, ell, K); }
        catch (const std::exception& e) { g.check(false, "(B) " + tag + ": native u64 threw " + e.what()); continue; }
        if (fits32) {
            std::string why;
            const Record n32 = NativeRecord<uint32_t>(c, ell, K);
            g.check(SameRecord(n32, nat, why), "(B) " + tag + ": native u32 == native u64 (" + why + ")");
        }
        for (const Variant& v : vs) {
            ++n;
            std::string why;
            try {
                const Record lr = Run(c, ell, K, v);
                const bool ok = SameRecord(lr, nat, why);
                if (ok) ++n_ok;
                else g.check(false, "(B) " + tag + " [" + v.name + "] != native u64 at section " + why);
            } catch (const std::exception& e) {
                g.check(false, "(B) " + tag + " [" + v.name + "] threw " + e.what());
            }
        }
        if (!c.planted.empty()) {
            const auto A = ToLadic<uint32_t>(c.A, c.n, lad::DigitLayout<uint32_t>::make(ell, K));
            g.check(lad::smith_lift(A).s == c.planted, "(B) " + tag + ": the planted exponents recovered");
        }
        try {
            Certificate<uint32_t>(g, ToLadic<uint32_t>(c.A, c.n, lad::DigitLayout<uint32_t>::make(ell, K)), {}, "Z/" + Mod(ell, K) + " " + c.name + " w32");
            lad::Options fi;
            fi.force_induction = true;
            Certificate<uint64_t>(g, ToLadic<uint64_t>(c.A, c.n, lad::DigitLayout<uint64_t>::make(ell, K)), fi, "Z/" + Mod(ell, K) + " " + c.name + " w64");
        } catch (const std::exception& e) {
            g.check(false, "(D) " + tag + ": threw " + e.what());
        }
    }
    Summ(g, n_ok == n, "(B) Z/" + Mod(ell, K) + ": " + std::to_string(n_ok) + "/" + std::to_string(n) +
                           " (case, digit variant) records == native u64 entrywise (2 x 32 = the w32 rows)");
}

// (G): 4 x 32 and 2 x 64 (and 3 x 32, 1 x 128) vs NATIVE u128 (WP-18a).
void NativeU128Arm(Checker& g, uint64_t ell, uint32_t K, uint64_t seed) {
    const rl::Ring R(ell, K);
    const std::vector<Case> cases = Families(R, seed, true);
    const std::vector<Variant> vs = Variants(ell, K, true);
    const uint32_t L32 = lad::DigitLayout<uint32_t>::make(ell, K).L, L64 = lad::DigitLayout<uint64_t>::make(ell, K).L,
                   L128 = lad::DigitLayout<u128>::make(ell, K).L;
    uint32_t n_ok = 0, n = 0;
    for (const Case& c : cases) {
        const std::string tag = "Z/" + Mod(ell, K) + " " + c.name;
        Record nat;
        try { nat = NativeRecord<u128>(c, ell, K); }
        catch (const std::exception& e) { g.check(false, "(G) " + tag + ": native u128 threw " + e.what()); continue; }
        for (const Variant& v : vs) {
            ++n;
            std::string why;
            try {
                if (SameRecord(Run(c, ell, K, v), nat, why)) ++n_ok;
                else g.check(false, "(G) " + tag + " [" + v.name + "] != native u128 at section " + why);
            } catch (const std::exception& e) {
                g.check(false, "(G) " + tag + " [" + v.name + "] threw " + e.what());
            }
        }
        try {
            lad::Options fi;
            fi.force_induction = true;
            Certificate<u128>(g, ToLadic<u128>(c.A, c.n, lad::DigitLayout<u128>::make(ell, K)), fi, tag + " w128");
        } catch (const std::exception& e) {
            g.check(false, "(D) " + tag + ": threw " + e.what());
        }
    }
    Summ(g, n_ok == n, "(G) Z/" + Mod(ell, K) + ": " + std::to_string(n_ok) + "/" + std::to_string(n) +
                           " (case, variant) records == native u128 entrywise (" + std::to_string(L32) + " x 32, " +
                           std::to_string(L64) + " x 64, " + std::to_string(L128) + " x 128)");
}

// (C): beyond the words -- cross-width, the base-ell oracle, planted divisors.
void CrossWidthArm(Checker& g, uint64_t ell, uint32_t K, uint64_t seed, bool small) {
    const rl::Ring R(ell, K);
    const std::vector<Case> cases = Families(R, seed, small);
    const std::vector<Variant> vs = Variants(ell, K, true);
    uint32_t n_ok = 0, n = 0;
    for (const Case& c : cases) {
        const std::string tag = "Z/" + Mod(ell, K) + " " + c.name;
        Record base;
        try { base = Run(c, ell, K, vs[0]); }
        catch (const std::exception& e) { g.check(false, "(C) " + tag + ": w32 threw " + e.what()); continue; }
        for (size_t i = 1; i < vs.size(); ++i) {
            ++n;
            std::string why;
            try {
                if (SameRecord(Run(c, ell, K, vs[i]), base, why)) ++n_ok;
                else g.check(false, "(C) " + tag + " [" + vs[i].name + "] != [w32] at section " + why);
            } catch (const std::exception& e) {
                g.check(false, "(C) " + tag + " [" + vs[i].name + "] threw " + e.what());
            }
        }
        // the naive base-ell oracle, read directly on A
        const lad::DigitLayout<uint32_t> lay = lad::DigitLayout<uint32_t>::make(ell, K);
        const auto A = ToLadic<uint32_t>(c.A, c.n, lay);
        const lad::Howell<uint32_t> h = lad::howell(A);
        const rl::HowellOut rh = rl::howell(R, c.A, c.n);
        bool hs = h.pivots == rh.pivots && h.pivot_v == rh.pivot_v && h.rows() == rh.H.size();
        for (uint32_t i = 0; hs && i < h.rows(); ++i)
            for (uint32_t j = 0; hs && j < c.n; ++j) hs = h.H.ell_digits(i, j) == rh.H[i][j];
        g.check(hs, "(C) " + tag + ": howell == the base-ell oracle's Sec. 13.5 pass on A");
        const lad::Howell<uint32_t> k = lad::kernel_module(A);
        const rl::HowellOut rk = rl::kernel_module(R, c.A, c.m, c.n);
        bool ks = k.pivots == rk.pivots && k.pivot_v == rk.pivot_v && k.rows() == rk.H.size();
        for (uint32_t i = 0; ks && i < k.rows(); ++i)
            for (uint32_t j = 0; ks && j < c.n; ++j) ks = k.H.ell_digits(i, j) == rk.H[i][j];
        g.check(ks, "(C) " + tag + ": kernel_module == the base-ell oracle's howell([A^T | I]) rows");
        const lad::Smith<uint32_t> sm = lad::smith_lift(A);
        g.check(sm.s == rl::smith_exponents(R, c.A, c.m, c.n), "(C) " + tag + ": Smith exponents == the base-ell oracle's");
        if (!c.planted.empty()) g.check(sm.s == c.planted, "(C) " + tag + ": the planted exponents recovered");
        // (slot-14 WP-14b, m-2) howell / kernel_module / Smith were the only
        // checks against the base-ell oracle; add A x = y re-derived by the
        // independent oracle's own multiply (the review's other half of
        // m-2, the refusal on Ybad, is NOT added here: Ybad is a random
        // m x 2 matrix, and the review's premise that it is "not in the
        // image w.h.p." only holds when A is not surjective onto R^m -- it
        // fails for exactly the "wide"/"identity"/full-row-rank cases this
        // arm also covers, where EVERY Ybad is trivially in the image, so an
        // unconditional refusal check is wrong there; verified by running it
        // first. A correct version needs an independent colspan-membership
        // oracle -- not the one call the review estimated -- so it is
        // re-carried, not attempted here).
        {
            const lad::LadicMat<uint32_t> Yl = ToLadic<uint32_t>(c.Y, 3, lay);
            // solve_multi_rhs_module returns ny x nb (row t = x_t): TRANSPOSED
            // from c.X's n x ny convention, hence the swapped indices below.
            const lad::LadicMat<uint32_t> X = lad::solve_multi_rhs_module(A, Yl);
            rl::Mat Xp(c.n, std::vector<rl::Elem>(3));
            for (uint32_t i = 0; i < c.n; ++i)
                for (uint32_t j = 0; j < 3; ++j) Xp[i][j] = X.ell_digits(j, i);
            g.check(rl::multiply(R, c.A, Xp) == c.Y,
                    "(C) " + tag + ": A x = y re-derived by the base-ell oracle's own multiply (m-2)");
        }
        try {
            Certificate<uint32_t>(g, A, {}, tag + " w32");
            Certificate<uint64_t>(g, ToLadic<uint64_t>(c.A, c.n, lad::DigitLayout<uint64_t>::make(ell, K)), {}, tag + " w64");
        } catch (const std::exception& e) {
            g.check(false, "(D) " + tag + ": threw " + e.what());
        }
    }
    Summ(g, n_ok == n, "(C) Z/" + Mod(ell, K) + ": " + std::to_string(n_ok) + "/" + std::to_string(n) +
                           " (case, variant) records == the w32 record (cross-width agreement)");
}

// (E) truncation
void TruncationArm(Checker& g, uint64_t ell, uint32_t K, const std::vector<uint32_t>& ks, uint64_t seed) {
    const rl::Ring R(ell, K);
    const std::vector<Case> cases = Families(R, seed, true);
    uint32_t n_ok = 0, n = 0;
    for (const Case& c : cases) {
        const lad::DigitLayout<uint32_t> lay = lad::DigitLayout<uint32_t>::make(ell, K);
        const auto A = ToLadic<uint32_t>(c.A, c.n, lay);
        const lad::Smith<uint32_t> sK = lad::smith_lift(A);
        const lad::Howell<uint32_t> hK = lad::howell(A), iK = lad::image_module(A);
        for (uint32_t kp : ks) {
            ++n;
            const std::string tag = "Z/" + Mod(ell, K) + " -> k' = " + std::to_string(kp) + " " + c.name;
            // truncate the INPUT; run the digit route at k'; compare with native at k'
            const rl::Ring Rk(ell, kp);
            Case ck = c;
            auto cut = [&](rl::Mat& M) { for (auto& row : M) for (auto& e : row) e.resize(kp); };
            cut(ck.A); cut(ck.G); cut(ck.Rl); cut(ck.X); cut(ck.Ybad); cut(ck.XG); cut(ck.Y); cut(ck.Yc);
            std::string why;
            const Record nat = NativeRecord<uint64_t>(ck, ell, kp);
            lad::Options fi;
            fi.force_induction = true;
            const lad::LadicMat<uint32_t> At = A.truncate(kp);
            const bool trunc_ok = At.serialise() == ToLadic<uint32_t>(ck.A, c.n, At.layout()).serialise();
            const bool rec_ok = SameRecord(LadicRecord<uint32_t>(ck, At.layout(), fi), nat, why);
            // the Smith law and the re-canonicalised Howell / image forms
            std::vector<uint32_t> want;
            for (uint32_t s : sK.s) if (s < kp) want.push_back(s);
            const bool smith_ok = lad::smith_lift(At, fi).s == want;
            const lad::Howell<uint32_t> hre = lad::howell(hK.H.truncate(kp), fi), ire = lad::howell(iK.H.truncate(kp), fi);
            const bool howell_ok = SerH(hre) == SerH(lad::howell(At, fi)) && SerH(ire) == SerH(lad::image_module(At, fi));
            const bool ok = trunc_ok && rec_ok && smith_ok && howell_ok;
            if (ok) ++n_ok;
            else g.check(false, "(E) " + tag + ": trunc " + std::to_string(trunc_ok) + " record " + std::to_string(rec_ok) +
                                    " (" + why + ") smith " + std::to_string(smith_ok) + " howell " + std::to_string(howell_ok));
            (void)Rk;
        }
    }
    Summ(g, n_ok == n, "(E) Z/" + Mod(ell, K) + ": truncation law on " + std::to_string(n_ok) + "/" + std::to_string(n) +
                           " (case, k') pairs");
}

// ===========================================================================
// (F) the pipeline's own d^q
// ===========================================================================
struct RingCell {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    engine::ModCtx<uint64_t> ctx;
    std::unique_ptr<LocalSystemSimplicialComplexT<uint64_t>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<uint64_t>> coh;
};
std::unique_ptr<RingCell> BuildCell(uint32_t n, uint64_t ell, uint32_t k) {
    auto c = std::make_unique<RingCell>();
    c->ctx = hecke::CoefficientRingContext<uint64_t>("EngineLadicInductionValidation", ell, k);
    auto rels = hecke::GenerateSL2ZRelationsRing<uint64_t>(c->K, n, c->ctx);
    c->lssc = std::make_unique<LocalSystemSimplicialComplexT<uint64_t>>(c->K, std::move(rels), n + 1, c->ctx.p);
    c->coh = std::make_unique<EquivariantCohomologyOnKT<uint64_t>>(c->lssc.get());
    c->coh->compute_cohomology();
    return c;
}

template<class W>
bool CycEqNative(const lad::Cyclic<W>& d, const engine::elim::CyclicDecomposition<uint64_t>& e, uint64_t ell, uint32_t K) {
    return d.exponents == e.exponents && d.position == e.position && d.basis.serialise() == SerN<uint64_t>(e.basis, ell, K) &&
           d.Gamma.serialise() == SerN<uint64_t>(e.Gamma, ell, K) && d.U.serialise() == SerN<uint64_t>(e.U, ell, K);
}

void PipelineArm(Checker& g, uint32_t n, uint64_t ell, uint32_t k) {
    const std::string tag = "(" + std::to_string(n) + ", " + std::to_string(ell) + ") Z/" + Mod(ell, k);
    const uint32_t f0 = g.fails, c0 = g.checks;
    try {
        hecke::ScopedCoefficientRing ring(k);
        const auto cell = BuildCell(n, ell, k);
        const EquivariantCohomologyOnKT<uint64_t>& coh = *cell->coh;
        const lad::DigitLayout<uint32_t> lay = lad::DigitLayout<uint32_t>::make(ell, k);
        g.check(lay.L == 2, "(F) " + tag + ": the w32 layout has L = 2 digits (2 x 32 vs native u64)");
        std::vector<lad::LadicMat<uint32_t>> X;
        for (uint32_t q = 0; q < coh.num_degrees(); ++q)
            X.push_back(lad::LadicMat<uint32_t>::from_word<uint64_t>(lay, engine::cview(coh.module_degree(q).X)));
        std::vector<lad::Smith<uint32_t>> sm;
        for (uint32_t q = 0; q < coh.num_degrees(); ++q) {
            const auto& M = coh.module_degree(q);
            const uint32_t cq = M.C.cols;
            const lad::Howell<uint32_t> kz = lad::kernel_module(X[q]);
            g.check(kz.H.transpose().serialise() == SerN<uint64_t>(M.Zc, ell, k), "(F) " + tag + " q = " + std::to_string(q) + ": kernel_module(X_q) == Zc");
            const lad::LadicMat<uint32_t> Zc = lad::LadicMat<uint32_t>::from_word<uint64_t>(lay, engine::cview(M.Zc));
            const lad::LadicMat<uint32_t> Bq = (q > 0) ? X[q - 1] : lad::LadicMat<uint32_t>(lay, cq, 0);
            const lad::Cyclic<uint32_t> d = lad::cyclic_decomposition(Zc, Bq);
            g.check(CycEqNative(d, M.dec, ell, k), "(F) " + tag + " q = " + std::to_string(q) + ": cyclic_decomposition(Zc, X_{q-1}) == the stored decomposition");
            if (M.X.rows > 0 && cq > 0) g.check(lad::invariant_factors(X[q]) == M.dK, "(F) " + tag + " q = " + std::to_string(q) + ": invariant_factors(X_q) == dK");
            sm.push_back(lad::smith_lift(X[q]));
            if (q < 2) g.check(sm.back().rank_K() == coh.r_K(q), "(F) " + tag + " q = " + std::to_string(q) + ": rank_K == r_K(q)");
            if (X[q].rows() > 0) Certificate<uint32_t>(g, X[q], {}, tag + " X_" + std::to_string(q));
        }
        if (coh.ladic_applicable() && coh.num_degrees() >= 2) {
            const uint32_t dc0 = coh.module_degree(0).C.cols, dc2 = coh.module_degree(1).X.rows;
            const lad::StoppingRule st = lad::stopping_rule(sm[0], sm[1], dc0, dc2);
            std::cout << "    (F) " << tag << ": r_K(d^0) = " << st.r0 << " / dim C^0_eq = " << dc0 << ", r_K(d^1) = " << st.r1
                      << " / dim C^2_eq = " << dc2 << ": stopping rule certified = " << st.certified << "\n";
            g.check(st.certified == coh.ladic_type_certified(), "(F) " + tag + ": stopping_rule == ladic_type_certified() (" +
                                                                   std::to_string(st.certified) + ")");
        }
        const uint32_t nc = g.checks - c0;
        Summ(g, g.fails == f0, "(F) " + tag + ": " + std::to_string(nc) + " checks on the pipeline's d^q at 2 x 32 digits (Zc, cyclic decomposition, dK, r_K, stopping rule, certificate)");
    } catch (const std::exception& e) {
        g.check(false, "(F) " + tag + ": threw " + e.what());
    }
}

// LADIC_INDUCTION_PROBE_N128=1: indicative timing, no verdict.
void ProbeN128() {
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    const uint32_t n = 128, k = 26;
    const uint64_t ell = 5;
    hecke::ScopedCoefficientRing ring(k);
    auto t0 = clk::now();
    const auto cell = BuildCell(n, ell, k);
    auto t1 = clk::now();
    const auto& M1 = cell->coh->module_degree(1);
    const auto& M0 = cell->coh->module_degree(0);
    std::cout << "  PROBE (128, 5) Z/5^26: pipeline (native u64) " << secs(t0, t1) << " s; X_1 " << M1.X.rows << " x "
              << M1.X.cols << ", X_0 " << M0.X.rows << " x " << M0.X.cols << "\n";
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, k);
    t0 = clk::now();
    const auto nk = engine::elim::kernel_module<uint64_t>(ctx, engine::cview(M1.X));
    t1 = clk::now();
    {
        engine::DenseBuf<uint64_t> Zc(M1.X.cols, nk.rows());
        for (uint32_t c = 0; c < nk.rows(); ++c)
            for (uint32_t i = 0; i < M1.X.cols; ++i) Zc.at(i, c) = nk.H.at(c, i);
        const auto t2 = clk::now();
        const auto nd = engine::elim::cyclic_decomposition<uint64_t>(ctx, engine::cview(Zc), engine::cview(M0.X));
        const auto t3 = clk::now();
        std::cout << "  PROBE native u64 kernel_module(X_1): " << secs(t0, t1) << " s, cyclic_decomposition(Zc, X_0): "
                  << secs(t2, t3) << " s (" << nd.rows() << " summands)\n";
    }
    {
        lad::Options fi;
        fi.force_induction = true;
        const lad::DigitLayout<uint64_t> lay = lad::DigitLayout<uint64_t>::make(ell, k);
        const auto X1 = lad::LadicMat<uint64_t>::from_word<uint64_t>(lay, engine::cview(M1.X));
        const auto X0 = lad::LadicMat<uint64_t>::from_word<uint64_t>(lay, engine::cview(M0.X));
        t0 = clk::now();
        const auto sm = lad::smith_lift(X1, fi);
        t1 = clk::now();
        const auto kz = lad::kernel_module(X1, fi);
        auto t2 = clk::now();
        const auto d = lad::cyclic_decomposition(kz.H.transpose(), X0, fi);
        auto t3 = clk::now();
        std::cout << "  PROBE w64 L = 1 (force_induction): smith_lift(X_1) " << secs(t0, t1) << " s, kernel_module(X_1) "
                  << secs(t1, t2) << " s, cyclic_decomposition " << secs(t2, t3) << " s (" << d.rows() << ", rank " << sm.rank_K() << ")\n";
    }
    for (uint32_t mo : {0u, 7u}) {
        const lad::DigitLayout<uint32_t> lay = lad::DigitLayout<uint32_t>::make(ell, k, mo);
        const auto X1 = lad::LadicMat<uint32_t>::from_word<uint64_t>(lay, engine::cview(M1.X));
        const auto X0 = lad::LadicMat<uint32_t>::from_word<uint64_t>(lay, engine::cview(M0.X));
        t0 = clk::now();
        const auto sm = lad::smith_lift(X1);
        t1 = clk::now();
        const auto kz = lad::kernel_module(X1);
        auto t2 = clk::now();
        const auto Zc = kz.H.transpose();
        const auto d = lad::cyclic_decomposition(Zc, X0);
        auto t3 = clk::now();
        std::cout << "  PROBE w32 L = " << lay.L << ": smith_lift(X_1) " << secs(t0, t1) << " s (rank " << sm.rank_K()
                  << "), kernel_module(X_1) " << secs(t1, t2) << " s, cyclic_decomposition(Zc, X_0) " << secs(t2, t3)
                  << " s; kernel == native: " << (kz.H.serialise() == SerN<uint64_t>(nk.H, ell, k)) << "; H^1 exponents "
                  << d.exponents.size() << "\n";
    }
}

} // namespace

uint32_t EngineLadicInductionValidation() {
    Checker g;
    std::cout << "EngineLadicInductionValidation (slot-11 WP-18b; ENGINE_API.md Sec. 13.11)\n";
    auto guard = [&](const std::string& what, const std::function<void()>& f) {
        try { f(); } catch (const std::exception& e) { g.check(false, what + ": threw " + e.what()); }
    };
    // (A)
    guard("(A) layout", [&] { LayoutArm(g); });
    guard("(A) split", [&] { SplitArm(g); });
    for (auto mk : std::vector<std::pair<uint64_t, uint32_t>>{{691, 6}, {5, 20}, {3, 30}, {7, 20}, {134217757, 2}, {2, 40}}) {
        guard("(A)", [&] {
            RingVsNative<uint32_t>(g, mk.first, mk.second, 0, "w32");
            RingVsNative<uint64_t>(g, mk.first, mk.second, 0, "w64");
            RingVsNative<uint32_t>(g, mk.first, mk.second, 1, "w32 m'=1");
            if (mk.first % 2 == 1) RingVsNative<u128>(g, mk.first, mk.second, 0, "w128");
        });
    }
    for (auto mk : std::vector<std::pair<uint64_t, uint32_t>>{{691, 20}, {5, 60}, {3, 100}, {691, 12}, {2147483647, 10}, {2147483647, 40}, {2, 200}}) {
        guard("(A)", [&] {
            RingVsRef<uint32_t>(g, mk.first, mk.second, 0, "w32");
            RingVsRef<uint64_t>(g, mk.first, mk.second, 0, "w64");
            RingVsRef<uint32_t>(g, mk.first, mk.second, 1, "w32 m'=1");
            if (mk.first % 2 == 1) {
                RingVsRef<u128>(g, mk.first, mk.second, 0, "w128");
                RingVsRef<u128>(g, mk.first, mk.second, 1, "w128 m'=1");
            }
        });
    }
    std::cout << "  (A) digit arithmetic: " << g.checks << " checks, " << g.fails << " failing\n";
    // (B) 2 x 32 vs native u64 (+ 1 x 32 vs native u32 below 2^32)
    uint64_t seed = 0x5eed18b0u;
    for (auto mk : std::vector<std::pair<uint64_t, uint32_t>>{{691, 6}, {5, 20}, {3, 30}, {7, 20}, {134217757, 2}, {2, 40}, {5, 13}, {691, 3}})
        for (int rep = 0; rep < 2; ++rep) guard("(B)", [&] { NativeArm(g, mk.first, mk.second, seed++); });
    std::cout << "  (B)+(D) native arms done: " << g.checks << " checks, " << g.fails << " failing\n";
    // (C) cross-width beyond the words
    for (auto mk : std::vector<std::pair<uint64_t, uint32_t>>{{691, 12}, {5, 40}, {691, 20}, {5, 60}, {2147483647, 12}})
        guard("(C)", [&] { CrossWidthArm(g, mk.first, mk.second, seed++, false); });
    guard("(C)", [&] { CrossWidthArm(g, 3, 100, seed++, true); });
    guard("(C)", [&] { CrossWidthArm(g, 2, 200, seed++, true); });
    std::cout << "  (C)+(D) cross-width arms done: " << g.checks << " checks, " << g.fails << " failing\n";
    // (E) truncation
    guard("(E)", [&] { TruncationArm(g, 691, 20, {2, 3, 6}, seed++); });
    guard("(E)", [&] { TruncationArm(g, 5, 60, {2, 13, 20, 27}, seed++); });
    guard("(E)", [&] { TruncationArm(g, 3, 100, {2, 19, 40}, seed++); });
    // (F) the pipeline's d^q
    PipelineArm(g, 10, 691, 6);
    PipelineArm(g, 30, 5, 20);
    std::cout << "  (E)+(F) done: " << g.checks << " checks, " << g.fails << " failing\n";
    // (G) the w = 128 columns
    // (G) the w = 128 columns (WP-18a merged): 4 x 32 and 2 x 64 vs NATIVE u128;
    // 3 x 32 vs 2 x 64 vs 1 x 128 (mutual, and vs native u128).
    for (auto mk : std::vector<std::pair<uint64_t, uint32_t>>{{691, 12}, {5, 40}, {3, 76}, {7, 40}, {134217757, 4}, {691, 9}, {5, 35}})
        guard("(G)", [&] { NativeU128Arm(g, mk.first, mk.second, seed++); });
    std::cout << "  (G) w = 128 columns done: " << g.checks << " checks, " << g.fails << " failing\n";
    if (const char* p = std::getenv("LADIC_INDUCTION_PROBE_N128"); p && std::string(p) == "1") {
        try { ProbeN128(); } catch (const std::exception& e) { std::cout << "  PROBE threw " << e.what() << "\n"; }
    }
    std::cout << "EngineLadicInductionValidation: " << g.checks << " checks, " << g.fails << " failing\n";
    return g.fails;
}
