// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineCarriesS11Validation.cpp -- see the header.
//
// hecke-extensions slot 11, WP-12: F-1 (the Montgomery-64 reused-operand pair
// in the composite row updates) and F-2 (ExactKthRoot's primality by
// Miller-Rabin; the ring carried as (ell, k)).
#include "EngineCarriesS11Validation.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/DenseView.h"
#include "../ModArithmetic/ModCtx.h"
#include "../ModArithmetic/RefLinAlg.h"
#include "EquivariantCohomologyOnK.h"
#include "Hecke/CoefficientRing.h"
#include "Hecke/HeckeSettings.h"
#include "LocalSystemSL2Z.h"
#include "LocalSystemSimplicialComplex.h"

namespace {

using engine::u128;

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

std::string u64s(uint64_t v) { return std::to_string(v); }

uint64_t PowU64(uint64_t b, uint32_t e) {
    u128 acc = 1;
    for (uint32_t i = 0; i < e; ++i) acc *= b;
    return static_cast<uint64_t>(acc);
}

// ===========================================================================
// F-1
// ===========================================================================

// The old exact product: (u128)a*b % p -- ModCtx<uint64_t>::mul's body.
uint64_t OldMul(uint64_t a, uint64_t b, uint64_t p) { return static_cast<uint64_t>((static_cast<u128>(a) * b) % p); }

template<class Elt>
void ArithmeticArm(Checker& g, const engine::ModCtx<Elt>& ctx, bool want_mont, const std::string& tag, uint32_t samples) {
    const uint64_t p = ctx.p;
    g.check(ctx.has_montgomery == want_mont, "(A) " + tag + ": has_montgomery == " + std::to_string(want_mont));
    std::mt19937_64 rng(0x5eed0000u + static_cast<uint32_t>(p % 1000003u));
    std::vector<uint64_t> vals = {0, 1, 2 % p, p - 1, (p >= 2) ? p - 2 : 0};
    for (uint32_t i = 0; i < samples; ++i) vals.push_back(rng() % p);
    uint32_t bad = 0;
    for (size_t i = 0; i < vals.size(); ++i) {
        const Elt w = static_cast<Elt>(vals[i]);
        const Elt wp = ctx.mul_prep(w);
        for (size_t j = 0; j < 8 && j < vals.size(); ++j) {
            const Elt x = static_cast<Elt>(vals[(i * 7 + j * 13) % vals.size()]);
            const Elt a = ctx.mul_prepped(x, wp);
            const Elt b = ctx.mul(x, w);
            const uint64_t c = OldMul(x, w, p);
            if (!(a == b && static_cast<uint64_t>(a) == c && static_cast<uint64_t>(a) < p)) ++bad;
        }
        if (!want_mont && wp != w) ++bad;   // outside the Montgomery regime the token IS w
    }
    g.check(bad == 0, "(A) " + tag + ": mul_prepped(x, mul_prep(w)) == mul(x, w) == (u128)x*w % p on " +
                          std::to_string(vals.size() * 8) + " pairs (" + std::to_string(bad) + " bad)");
}

template<class Elt>
engine::DenseBuf<Elt> RandomValued(const engine::ModCtx<Elt>& ctx, uint32_t m, uint32_t n, std::mt19937_64& rng) {
    engine::DenseBuf<Elt> A(m, n);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t r = static_cast<uint32_t>(rng() % 10);
            if (r < 2) { A.at(i, j) = 0; continue; }
            uint32_t v = (r < 5) ? 0u : (r < 8 ? 1u : 2u);
            if (v >= ctx.k) v = ctx.k - 1;
            const u128 x = static_cast<u128>(ctx.ell_pow(v)) * (rng() % ctx.p);
            A.at(i, j) = static_cast<Elt>(x % ctx.p);
        }
    return A;
}

template<class Elt>
engine::DenseBuf<Elt> MatMul(const engine::ModCtx<Elt>& ctx, engine::DenseView<const Elt> A, engine::DenseView<const Elt> B) {
    engine::DenseBuf<Elt> C(A.rows(), B.cols());
    for (uint32_t i = 0; i < A.rows(); ++i)
        for (uint32_t j = 0; j < B.cols(); ++j) {
            Elt acc = 0;
            for (uint32_t t = 0; t < A.cols(); ++t) acc = ctx.add(acc, ctx.mul(A.at(i, t), B.at(t, j)));
            C.at(i, j) = acc;
        }
    return C;
}

template<class EA, class EB>
bool BufEq(const engine::DenseBuf<EA>& a, const engine::DenseBuf<EB>& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (static_cast<uint64_t>(a.at(i, j)) != static_cast<uint64_t>(b.at(i, j))) return false;
    return true;
}
template<class EA, class EB>
bool HowellEq(const engine::elim::HowellResult<EA>& a, const engine::elim::HowellResult<EB>& b) {
    return a.pivots == b.pivots && a.pivot_v == b.pivot_v && BufEq(a.H, b.H);
}
template<class EA, class EB>
bool VecVecEq(const std::vector<std::vector<EA>>& a, const std::vector<std::vector<EB>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); ++j)
            if (static_cast<uint64_t>(a[i][j]) != static_cast<uint64_t>(b[i][j])) return false;
    }
    return true;
}
template<class EA, class EB>
bool CycEq(const engine::elim::CyclicDecomposition<EA>& a, const engine::elim::CyclicDecomposition<EB>& b) {
    return a.exponents == b.exponents && a.position == b.position && BufEq(a.basis, b.basis) &&
           BufEq(a.Gamma, b.Gamma) && BufEq(a.U, b.U);
}
template<class Elt>
bool RefEq(const engine::elim::HowellResult<Elt>& a, const engine::ref::RefHowell<Elt>& r) {
    if (a.pivots != r.pivots || a.pivot_v != r.pivot_v || a.H.rows != r.rows.size()) return false;
    for (uint32_t i = 0; i < a.H.rows; ++i) {
        if (r.rows[i].size() != a.H.cols) return false;
        for (uint32_t j = 0; j < a.H.cols; ++j)
            if (a.H.at(i, j) != r.rows[i][j]) return false;
    }
    return true;
}

// Every composite entry point, on one input set, under one context.
template<class Elt>
struct ElimOut {
    engine::elim::HowellResult<Elt> h, ker, img;
    std::vector<std::vector<Elt>> sol;
    uint32_t len = 0;
    std::vector<uint32_t> inv;
    engine::elim::CyclicDecomposition<Elt> cyc;
    std::vector<std::vector<Elt>> coords;
};
template<class Elt>
ElimOut<Elt> RunAll(const engine::ModCtx<Elt>& ctx, const engine::DenseBuf<Elt>& A, const engine::DenseBuf<Elt>& Y,
                    const engine::DenseBuf<Elt>& G, const engine::DenseBuf<Elt>& Rl, const engine::DenseBuf<Elt>& YG) {
    using namespace engine;
    ElimOut<Elt> o;
    o.h = elim::howell<Elt>(ctx, cview(A));
    o.ker = elim::kernel_module<Elt>(ctx, cview(A));
    o.img = elim::image_module<Elt>(ctx, cview(A));
    o.sol = elim::solve_multi_rhs_module<Elt>(ctx, cview(A), cview(Y));
    o.len = elim::length<Elt>(ctx, cview(A));
    o.inv = elim::invariant_factors<Elt>(ctx, cview(G), cview(Rl));
    o.cyc = elim::cyclic_decomposition<Elt>(ctx, cview(G), cview(Rl));
    o.coords = elim::cyclic_coordinates<Elt>(ctx, o.cyc, cview(YG));
    return o;
}
template<class EA, class EB>
bool ElimEq(const ElimOut<EA>& a, const ElimOut<EB>& b) {
    return HowellEq(a.h, b.h) && HowellEq(a.ker, b.ker) && HowellEq(a.img, b.img) && VecVecEq(a.sol, b.sol) &&
           a.len == b.len && a.inv == b.inv && CycEq(a.cyc, b.cyc) && VecVecEq(a.coords, b.coords);
}

struct Shape { uint32_t m, n, ng, nr; };
const Shape kShapes[] = {{6, 9, 5, 2}, {14, 11, 9, 4}, {23, 31, 15, 6}};

// (B): the Montgomery context against the same context with the flag cleared
// (the pre-package %-path) and against ref::howell.
void ElimArmBody(Checker& g, uint64_t ell, uint32_t k) {
    using namespace engine;
    const ModCtx<uint64_t> ctx = ModCtx<uint64_t>::prime_power(ell, k);
    ModCtx<uint64_t> old = ctx;
    old.has_montgomery = false;   // mul_prep / mul_prepped == identity / mul: the old code, byte for byte
    const std::string tag = "Z/" + u64s(ell) + "^" + std::to_string(k) + (ctx.has_montgomery ? " (Montgomery)" : " (Wide)");
    std::mt19937_64 rng(0xF1u * ell + k);
    uint32_t n_ok = 0, n_ref = 0, n_cases = 0, n_nonunit = 0, n_torsion = 0;
    for (const Shape& s : kShapes) {
        for (uint32_t rep = 0; rep < 3; ++rep) {
            ++n_cases;
            const DenseBuf<uint64_t> A = RandomValued(ctx, s.m, s.n, rng);
            const DenseBuf<uint64_t> X = RandomValued(ctx, s.n, 3, rng);
            const DenseBuf<uint64_t> Y = MatMul(ctx, cview(A), cview(X));
            const DenseBuf<uint64_t> G = RandomValued(ctx, s.m, s.ng, rng);
            const DenseBuf<uint64_t> Rl = RandomValued(ctx, s.m, s.nr, rng);
            const DenseBuf<uint64_t> XG = RandomValued(ctx, s.ng, 2, rng);
            const DenseBuf<uint64_t> YG = MatMul(ctx, cview(G), cview(XG));
            const ElimOut<uint64_t> nw = RunAll(ctx, A, Y, G, Rl, YG);
            const ElimOut<uint64_t> od = RunAll(old, A, Y, G, Rl, YG);
            if (ElimEq(nw, od)) ++n_ok;
            if (RefEq(nw.h, ref::howell<uint64_t>(ell, k, cview(A)))) ++n_ref;
            for (uint32_t v : nw.h.pivot_v) if (v > 0) { ++n_nonunit; break; }
            for (uint32_t e : nw.cyc.exponents) if (e < k) { ++n_torsion; break; }
        }
    }
    // Coverage: the inputs reach the non-field branches (a non-unit pivot, so
    // the closure row and the ell^v scaling; a torsion summand in the Smith
    // elimination), not only unit pivots.
    g.check(n_nonunit > 0 && n_torsion > 0, "(B) " + tag + ": the inputs reach non-unit pivots (" +
                                                 std::to_string(n_nonunit) + " cases) and torsion summands (" +
                                                 std::to_string(n_torsion) + " cases)");
    g.check(n_ok == n_cases, "(B) " + tag + ": all eight composite entry points equal the %-path entrywise (" +
                                 std::to_string(n_ok) + "/" + std::to_string(n_cases) + ")");
    g.check(n_ref == n_cases, "(B) " + tag + ": howell == ref::howell (naive oracle) (" + std::to_string(n_ref) + "/" +
                                  std::to_string(n_cases) + ")");
}

// (C): u64 (Montgomery row updates) against u32 (unchanged) at ell^k < 2^32.
void WidthArmBody(Checker& g, uint32_t ell, uint32_t k) {
    using namespace engine;
    const ModCtx<uint32_t> c32 = ModCtx<uint32_t>::prime_power(ell, k);
    const ModCtx<uint64_t> c64 = ModCtx<uint64_t>::prime_power(ell, k);
    const std::string tag = "Z/" + std::to_string(ell) + "^" + std::to_string(k);
    g.check(c64.has_montgomery, "(C) " + tag + ": the u64 context is in the Montgomery regime");
    std::mt19937_64 rng(0xC0u * ell + k);
    auto narrow = [](const DenseBuf<uint64_t>& b) {
        DenseBuf<uint32_t> o(b.rows, b.cols);
        for (uint32_t i = 0; i < b.rows; ++i)
            for (uint32_t j = 0; j < b.cols; ++j) o.at(i, j) = static_cast<uint32_t>(b.at(i, j));
        return o;
    };
    uint32_t n_ok = 0, n_cases = 0;
    for (const Shape& s : kShapes) {
        for (uint32_t rep = 0; rep < 3; ++rep) {
            ++n_cases;
            const DenseBuf<uint64_t> A = RandomValued(c64, s.m, s.n, rng);
            const DenseBuf<uint64_t> X = RandomValued(c64, s.n, 3, rng);
            const DenseBuf<uint64_t> Y = MatMul(c64, cview(A), cview(X));
            const DenseBuf<uint64_t> G = RandomValued(c64, s.m, s.ng, rng);
            const DenseBuf<uint64_t> Rl = RandomValued(c64, s.m, s.nr, rng);
            const DenseBuf<uint64_t> XG = RandomValued(c64, s.ng, 2, rng);
            const DenseBuf<uint64_t> YG = MatMul(c64, cview(G), cview(XG));
            const ElimOut<uint64_t> w = RunAll(c64, A, Y, G, Rl, YG);
            const ElimOut<uint32_t> n = RunAll(c32, narrow(A), narrow(Y), narrow(G), narrow(Rl), narrow(YG));
            if (ElimEq(w, n)) ++n_ok;
        }
    }
    g.check(n_ok == n_cases, "(C) " + tag + ": u64 == u32 entrywise on all eight entry points (" + std::to_string(n_ok) +
                                 "/" + std::to_string(n_cases) + ")");
}

template<class Elt>
struct RingCell {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    engine::ModCtx<Elt> ctx;
    std::unique_ptr<LocalSystemSimplicialComplexT<Elt>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<Elt>> coh;
};
template<class Elt>
std::unique_ptr<RingCell<Elt>> BuildCell(uint32_t n, uint64_t ell, uint32_t k, bool declare) {
    auto c = std::make_unique<RingCell<Elt>>();
    c->ctx = hecke::CoefficientRingContext<Elt>("EngineCarriesS11Validation", ell, k);
    auto rels = hecke::GenerateSL2ZRelationsRing<Elt>(c->K, n, c->ctx);
    c->lssc = std::make_unique<LocalSystemSimplicialComplexT<Elt>>(c->K, std::move(rels), n + 1, c->ctx.p);
    if (declare) c->lssc->declare_coefficient_ring(ell, k);
    c->coh = std::make_unique<EquivariantCohomologyOnKT<Elt>>(c->lssc.get());
    c->coh->compute_cohomology();
    return c;
}
// A throw inside an arm (e.g. solve_multi_rhs_module's "not in image" on a
// wrong elimination) is a failed check, not an aborted suite.
void ElimArm(Checker& g, uint64_t ell, uint32_t k) {
    try { ElimArmBody(g, ell, k); }
    catch (const std::exception& e) { g.check(false, "(B) Z/" + u64s(ell) + "^" + std::to_string(k) + ": threw " + e.what()); }
}
void WidthArm(Checker& g, uint32_t ell, uint32_t k) {
    try { WidthArmBody(g, ell, k); }
    catch (const std::exception& e) { g.check(false, "(C) Z/" + std::to_string(ell) + "^" + std::to_string(k) + ": threw " + e.what()); }
}

template<class EA, class EB>
bool CohEq(const EquivariantCohomologyOnKT<EA>& a, const EquivariantCohomologyOnKT<EB>& b, std::string& why) {
    if (a.H_lengths() != b.H_lengths()) { why = "H_lengths"; return false; }
    for (uint32_t q = 0; q < 3; ++q) {
        const std::string at = " at q = " + std::to_string(q);
        if (a.H_invariant_factors(q) != b.H_invariant_factors(q)) { why = "invariant factors" + at; return false; }
        const auto& x = a.module_degree(q);
        const auto& y = b.module_degree(q);
        if (!BufEq(x.C, y.C) || !BufEq(x.X, y.X) || !BufEq(x.Zc, y.Zc) || !BufEq(x.H, y.H)) { why = "C / X / Zc / H" + at; return false; }
        if (!CycEq(x.dec, y.dec) || x.dK != y.dK || x.length_H != y.length_H) { why = "cyclic decomposition / dK" + at; return false; }
    }
    return true;
}

// (D): the pipeline's module route, u64 against u32.
void PipelineWidthArm(Checker& g, uint32_t n, uint32_t ell, uint32_t k) {
    const std::string tag = "(" + std::to_string(n) + ", " + std::to_string(ell) + ") Z/ell^" + std::to_string(k);
    try {
        hecke::ScopedCoefficientRing ring(k);
        const auto a = BuildCell<uint32_t>(n, ell, k, false);
        const auto b = BuildCell<uint64_t>(n, ell, k, false);
        g.check(b->ctx.has_montgomery, "(D) " + tag + ": the u64 ring context is in the Montgomery regime");
        std::string why;
        const bool eq = CohEq(*a->coh, *b->coh, why);
        g.check(eq, "(D) " + tag + ": u64 == u32 entrywise on the module route" + (eq ? "" : " (" + why + ")"));
    } catch (const std::exception& e) {
        g.check(false, "(D) " + tag + ": threw " + e.what());
    }
}

// ===========================================================================
// F-2
// ===========================================================================

bool TrialPrime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

// The slot-8 ExactKthRoot, verbatim but for the name (the oracle of (B)).
uint64_t OldExactKthRoot(uint64_t m, uint32_t k) {
    if (k < 2 || m < 4) return 0;   // k = 1: the slot-8 shift by 65 is UB (see CoefficientRing.cpp)
    auto pow_le = [k](uint64_t r, uint64_t bound, bool& fits) -> uint64_t {
        unsigned __int128 acc = 1;
        for (uint32_t i = 0; i < k; ++i) {
            acc *= r;
            if (acc > bound) { fits = false; return 0; }
        }
        fits = true;
        return static_cast<uint64_t>(acc);
    };
    uint64_t lo = 1, hi = (k >= 32) ? 4 : (uint64_t(1) << (64 / k + 1));
    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo + 1) / 2;
        bool fits = false;
        (void)pow_le(mid, m, fits);
        if (fits) lo = mid; else hi = mid - 1;
    }
    bool fits = false;
    const uint64_t v = pow_le(lo, m, fits);
    if (!(fits && v == m && lo >= 2)) return 0;
    for (uint64_t d = 2; d * d <= lo; ++d)
        if (lo % d == 0) return 0;
    return lo;
}

template<class F>
bool Throws(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
template<class F>
std::string WhatOf(F&& f) {
    try { f(); } catch (const std::exception& e) { return e.what(); }
    return "";
}

void DeclaredCellArm(Checker& g, uint32_t n, uint32_t ell, uint32_t k) {
    const std::string tag = "(" + std::to_string(n) + ", " + std::to_string(ell) + ") Z/ell^" + std::to_string(k);
    try {
        hecke::ScopedCoefficientRing ring(k);
        const auto u = BuildCell<uint32_t>(n, ell, k, false);
        const auto d = BuildCell<uint32_t>(n, ell, k, true);
        g.check(d->lssc->has_declared_coefficient_ring() && d->lssc->declared_ring_ell() == ell &&
                    d->lssc->declared_ring_k() == k && !u->lssc->has_declared_coefficient_ring(),
                "(D) " + tag + ": the declaration is recorded on one system only");
        g.check(d->coh->is_composite() && d->coh->ring_context().ell == ell && d->coh->ring_context().k == k,
                "(D) " + tag + ": the declared object is over Z/ell^k with the declared ell");
        std::string why;
        const bool eq = CohEq(*u->coh, *d->coh, why);
        g.check(eq, "(D) " + tag + ": declared == undeclared entrywise" + (eq ? "" : " (" + why + ")"));
    } catch (const std::exception& e) {
        g.check(false, "(D) " + tag + ": threw " + e.what());
    }
}

}  // namespace

uint32_t ModCtxMontgomeryPairValidation() {
    using namespace engine;
    Checker g;
    // ---- (A) arithmetic ------------------------------------------------------
    struct M64 { uint64_t p; bool mont; const char* name; };
    const M64 mods[] = {
        {PowU64(691, 6), true, "691^6"},
        {PowU64(5, 27), true, "5^27"},
        {PowU64(3, 39), true, "3^39"},
        {PowU64(134217757ull, 2), true, "134217757^2"},
        {4294967311ull, true, "2^32+15 (prime)"},
        {9223372036854775783ull, true, "2^63-25 (prime)"},
        {uint64_t(1) << 40, false, "2^40"},
        {uint64_t(1) << 63, false, "2^63"},
        {PowU64(3, 40), false, "3^40 (> 2^63)"},
        {18446744073709551557ull, false, "2^64-59 (prime)"},
    };
    for (const M64& m : mods) ArithmeticArm<uint64_t>(g, ModCtx<uint64_t>(m.p), m.mont, m.name, 20000);
    ArithmeticArm<uint64_t>(g, ModCtx<uint64_t>::prime_power(691, 6), true, "prime_power(691, 6)", 2000);
    for (uint32_t p32 : {134217757u, 4294967291u, 329939371u /* 691^3 */})
        ArithmeticArm<uint32_t>(g, ModCtx<uint32_t>(p32), false, "u32 " + std::to_string(p32), 20000);

    // ---- (B) composite entry points: Montgomery vs the %-path vs ref::howell -
    ElimArm(g, 691, 6);
    ElimArm(g, 5, 27);
    ElimArm(g, 3, 39);
    ElimArm(g, 134217757ull, 2);
    ElimArm(g, 2, 40);
    ElimArm(g, 2, 63);
    ElimArm(g, 3, 40);

    // ---- (C) u64 == u32 below 2^32 ---------------------------------------------
    WidthArm(g, 691, 3);
    WidthArm(g, 5, 13);
    WidthArm(g, 7, 11);

    // ---- (D) the pipeline's module route, u64 == u32 ----------------------------
    PipelineWidthArm(g, 10, 691, 3);
    PipelineWidthArm(g, 10, 5, 3);
    PipelineWidthArm(g, 22, 7, 3);

    std::cout << "ModCtxMontgomeryPairValidation: " << g.checks << " checks, " << g.fails << " failing\n";
    return g.fails;
}

uint32_t CoefficientRingKthRootValidation() {
    Checker g;
    // ---- (A) IsPrimeU64 == trial division --------------------------------------
    {
        uint32_t bad = 0;
        for (uint64_t n = 0; n < 200000; ++n)
            if (hecke::IsPrimeU64(n) != TrialPrime(n)) ++bad;
        g.check(bad == 0, "(A) IsPrimeU64 == trial division on n < 200000 (" + std::to_string(bad) + " bad)");
        std::mt19937_64 rng(0xF2);
        bad = 0;
        for (uint32_t i = 0; i < 20000; ++i) {
            const uint64_t n = rng() & 0xffffffffull;
            if (hecke::IsPrimeU64(n) != TrialPrime(n)) ++bad;
        }
        g.check(bad == 0, "(A) IsPrimeU64 == trial division on 20000 random 32-bit n (" + std::to_string(bad) + " bad)");
        // Strong pseudoprimes to the leading bases (OEIS A014233), Carmichael
        // numbers, squares of large primes: all composite.
        const uint64_t composites[] = {2047ull, 1373653ull, 25326001ull, 3215031751ull, 2152302898747ull,
                                       3474749660383ull, 341550071728321ull, 3825123056546413051ull,
                                       561ull, 1105ull, 1729ull, 2465ull, 41041ull, 825265ull, 321197185ull,
                                       4294967291ull * 4294967291ull, 4294967279ull * 4294967291ull,
                                       PowU64(691, 6), 18446744073709551615ull};
        bad = 0;
        for (uint64_t n : composites) if (hecke::IsPrimeU64(n)) ++bad;
        g.check(bad == 0, "(A) IsPrimeU64 rejects the strong pseudoprimes, Carmichael numbers and prime squares");
        const uint64_t primes[] = {2ull, 3ull, 37ull, 41ull, 691ull, 134217757ull, 4294967291ull, 4294967311ull,
                                   9223372036854775783ull, 18446744073709551557ull, 18446744073709551533ull};
        bad = 0;
        for (uint64_t n : primes) if (!hecke::IsPrimeU64(n)) ++bad;
        g.check(bad == 0, "(A) IsPrimeU64 accepts the listed primes (to 2^64 - 59)");
        g.check(!hecke::IsPrimeU64(18446744073709551555ull) && !hecke::IsPrimeU64(18446744073709551559ull),
                "(A) the odd neighbours of 2^64 - 59 are composite");
    }
    // ---- (B) ExactKthRoot == the slot-8 implementation ---------------------------
    {
        uint32_t bad = 0, cases = 0;
        for (uint64_t ell = 2; ell < 2000; ++ell) {
            if (!TrialPrime(ell)) continue;
            u128 acc = ell;
            for (uint32_t k = 2; k <= 63; ++k) {
                acc *= ell;
                if (acc > std::numeric_limits<uint64_t>::max()) break;
                const uint64_t m = static_cast<uint64_t>(acc);
                ++cases;
                if (hecke::ExactKthRoot(m, k) != ell || OldExactKthRoot(m, k) != ell) ++bad;
                // off by one, and the wrong k
                if (hecke::ExactKthRoot(m + 1, k) != OldExactKthRoot(m + 1, k)) ++bad;
                if (hecke::ExactKthRoot(m, k + 1) != OldExactKthRoot(m, k + 1)) ++bad;
                if (k >= 3 && hecke::ExactKthRoot(m, k - 1) != OldExactKthRoot(m, k - 1)) ++bad;
            }
        }
        g.check(bad == 0, "(B) ExactKthRoot == the slot-8 implementation on " + std::to_string(cases) +
                              " prime powers ell^k < 2^64, ell < 2000, and their neighbours (" + std::to_string(bad) + " bad)");
        bad = 0; cases = 0;
        for (uint64_t r : {4ull, 6ull, 9ull, 15ull, 25ull, 49ull, 1001ull, 65535ull}) {
            u128 acc = r;
            for (uint32_t k = 2; k <= 63; ++k) {
                acc *= r;
                if (acc > std::numeric_limits<uint64_t>::max()) break;
                ++cases;
                const uint64_t m = static_cast<uint64_t>(acc);
                if (hecke::ExactKthRoot(m, k) != 0 || OldExactKthRoot(m, k) != 0) ++bad;
            }
        }
        g.check(bad == 0, "(B) ExactKthRoot rejects " + std::to_string(cases) + " powers of composite roots (625 = 25^2 etc.)");
        std::mt19937_64 rng(0xF22);
        bad = 0;
        for (uint32_t i = 0; i < 20000; ++i) {
            const uint64_t m = rng() >> (rng() % 60);
            const uint32_t k = 1 + static_cast<uint32_t>(rng() % 63);
            if (hecke::ExactKthRoot(m, k) != OldExactKthRoot(m, k)) ++bad;
        }
        g.check(bad == 0, "(B) ExactKthRoot == the slot-8 implementation on 20000 random (m, k)");
        g.check(hecke::ExactKthRoot(0, 2) == 0 && hecke::ExactKthRoot(3, 2) == 0 && hecke::ExactKthRoot(16, 0) == 0 &&
                    hecke::ExactKthRoot(691, 1) == 0 && hecke::ExactKthRoot(4, 1) == 0 &&
                    hecke::ExactKthRoot(4, 2) == 2 && hecke::ExactKthRoot(uint64_t(1) << 63, 63) == 2,
                "(B) edge inputs (m < 4, k = 0, k = 1 refused, 2^2, 2^63)");
        const uint64_t big = 4294967291ull;   // the largest prime below 2^32
        g.check(hecke::ExactKthRoot(big * big, 2) == big && hecke::ExactKthRoot(big * 4294967279ull, 2) == 0,
                "(B) the k = 2 worst case of the word (4294967291^2) and a semiprime neighbour");
    }
    // ---- (C) declare_coefficient_ring ----------------------------------------------
    {
        SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const engine::ModCtx<uint32_t> ctx = engine::ModCtx<uint32_t>::prime_power(5, 3);
        LocalSystemSimplicialComplexT<uint32_t> l(K, hecke::GenerateSL2ZRelationsRing<uint32_t>(K, 4, ctx), 5, ctx.p);
        g.check(!l.has_declared_coefficient_ring() && l.declared_ring_ell() == 0 && l.declared_ring_k() == 1,
                "(C) undeclared by default");
        g.check(Throws([&] { l.declare_coefficient_ring(5, 1); }), "(C) k = 1 refused");
        g.check(Throws([&] { l.declare_coefficient_ring(1, 3); }), "(C) ell = 1 refused");
        g.check(Throws([&] { l.declare_coefficient_ring(5, 2); }), "(C) 5^2 != 125 refused");
        g.check(Throws([&] { l.declare_coefficient_ring(5, 4); }), "(C) 5^4 != 125 refused");
        g.check(Throws([&] { l.declare_coefficient_ring(7, 3); }), "(C) 7^3 != 125 refused");
        g.check(!l.has_declared_coefficient_ring(), "(C) a refused declaration records nothing");
        l.declare_coefficient_ring(5, 3);
        g.check(l.has_declared_coefficient_ring() && l.declared_ring_ell() == 5 && l.declared_ring_k() == 3,
                "(C) 5^3 == 125 recorded as (5, 3)");
    }
    // ---- (D) the declared ring through EquivariantCohomologyOnK ---------------------
    DeclaredCellArm(g, 10, 5, 2);
    DeclaredCellArm(g, 10, 691, 2);
    DeclaredCellArm(g, 16, 7, 3);
    {
        // A declaration of k = 4 under coefficient_ring = Z/ell^2 refuses exactly
        // as an undeclared 5^4 modulus does (ExactKthRoot(625, 2) = 0: 25 is not prime).
        SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const engine::ModCtx<uint32_t> ctx = engine::ModCtx<uint32_t>::prime_power(5, 4);
        LocalSystemSimplicialComplexT<uint32_t> lu(K, hecke::GenerateSL2ZRelationsRing<uint32_t>(K, 4, ctx), 5, ctx.p);
        LocalSystemSimplicialComplexT<uint32_t> ld(K, hecke::GenerateSL2ZRelationsRing<uint32_t>(K, 4, ctx), 5, ctx.p);
        ld.declare_coefficient_ring(5, 4);
        hecke::ScopedCoefficientRing ring(2);
        const std::string wu = WhatOf([&] { EquivariantCohomologyOnK c(&lu); });
        const std::string wd = WhatOf([&] { EquivariantCohomologyOnK c(&ld); });
        g.check(!wu.empty() && wu == wd, "(D) declared k = 4 under Z/ell^2 refuses with the undeclared refusal's text");
    }

    std::cout << "CoefficientRingKthRootValidation: " << g.checks << " checks, " << g.fails << " failing\n";
    return g.fails;
}
