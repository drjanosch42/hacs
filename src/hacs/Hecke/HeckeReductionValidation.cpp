// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "HeckeReductionValidation.h"
#include "HeckeReductionPipelineValidation.h"   // (slot-8 WP-3) the E8 item 3 arms
#include "HeckeReductionCoverageSlowValidation.h"   // (slot-8 WP-4) the E8 item 4a arms
#include "HeckeReductionConeValidation.h"           // (slot-9 WP-4) the E8 item 4b arms

// hecke-extensions slot 8, WP-1 (E8 item 2): HeckeReductionValidation's
// elimination arms, per src/ModArithmetic/ENGINE_API.md Sec. 13.9.
// See the header for the arm list.  The pipeline / reduction / Bockstein arms
// are E8 items 3-4 (WP-3 / WP-4) and are appended by those packages.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../../Engine/Executor.h"
#include "../../Engine/Passes.h"
#include "../../Engine/Recorder.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/ModArithmetic.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../ModArithmetic/RefLinAlg.h"

namespace {

using engine::DenseBuf;
using engine::DenseView;
using engine::ModCtx;
using engine::u128;
namespace elim = engine::elim;
namespace ref = engine::ref;

// ---- the harness -----------------------------------------------------------
uint32_t g_fail = 0;
uint64_t g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
    }
}

template<class F>
bool throws_logic(F&& f) {
    try { f(); } catch (const std::logic_error&) { return true; } catch (...) { return false; }
    return false;
}
template<class F>
bool throws_runtime_msg(F&& f, const std::string& msg) {
    try { f(); } catch (const std::logic_error&) { return false; }
    catch (const std::runtime_error& e) { return std::string(e.what()) == msg; }
    catch (...) { return false; }
    return false;
}

struct Splitmix {
    uint64_t s;
    explicit Splitmix(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    uint64_t below(uint64_t m) { return m <= 1 ? 0 : next() % m; }
};

template<class T> std::string str(T v) { return std::to_string(v); }

// ---- small matrix helpers ---------------------------------------------------
template<class Elt>
std::string tag_of(const ModCtx<Elt>& c) {
    return "Z/" + str(c.ell) + "^" + str(c.k) + (std::is_same<Elt, uint32_t>::value ? " (u32)" : " (u64)");
}

template<class Elt>
DenseBuf<Elt> transpose_of(DenseView<const Elt> A) {
    DenseBuf<Elt> T(A.cols(), A.rows());
    for (uint32_t i = 0; i < A.rows(); ++i)
        for (uint32_t j = 0; j < A.cols(); ++j) T.at(j, i) = A.at(i, j);
    return T;
}

template<class Elt>
DenseBuf<Elt> mul(Elt p, DenseView<const Elt> A, DenseView<const Elt> B) {
    DenseBuf<Elt> C(A.rows(), B.cols());
    ref::gemm<Elt>(p, A, B, engine::view(C), 0);
    return C;
}

// A random element of valuation-biased shape: ell^v * u, v uniform in [0, k]
// (v = k gives 0), u a random residue (so the valuation is >= v).
template<class Elt>
Elt rand_elt(Splitmix& rng, const ModCtx<Elt>& c) {
    const uint32_t v = static_cast<uint32_t>(rng.below(c.k + 1));
    if (v == c.k) return 0;
    const Elt u = static_cast<Elt>(rng.below(c.p));
    return c.mul(c.ell_pow(v), u);
}

template<class Elt>
DenseBuf<Elt> rand_mat(Splitmix& rng, const ModCtx<Elt>& c, uint32_t m, uint32_t n) {
    DenseBuf<Elt> A(m, n);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) A.at(i, j) = rand_elt(rng, c);
    return A;
}

// A random unimodular m x m matrix: a product of elementary operations
// (unit scalings, transvections, swaps) applied to I.
template<class Elt>
DenseBuf<Elt> rand_unimodular(Splitmix& rng, const ModCtx<Elt>& c, uint32_t m) {
    DenseBuf<Elt> U(m, m);
    for (uint32_t i = 0; i < m; ++i) U.at(i, i) = 1;
    if (m == 0) return U;
    for (uint32_t t = 0; t < 4 * m + 2; ++t) {
        const uint32_t i = static_cast<uint32_t>(rng.below(m)), j = static_cast<uint32_t>(rng.below(m));
        const uint32_t kind = static_cast<uint32_t>(rng.below(3));
        if (kind == 0) {                           // scale row i by a unit
            Elt u;
            do { u = static_cast<Elt>(rng.below(c.p)); } while (u == 0 || u % c.ell == 0);
            for (uint32_t q = 0; q < m; ++q) U.at(i, q) = c.mul(U.at(i, q), u);
        } else if (kind == 1 && i != j) {          // row i += f * row j
            const Elt f = static_cast<Elt>(rng.below(c.p));
            for (uint32_t q = 0; q < m; ++q) U.at(i, q) = c.add(U.at(i, q), c.mul(f, U.at(j, q)));
        } else {                                   // swap
            for (uint32_t q = 0; q < m; ++q) std::swap(U.at(i, q), U.at(j, q));
        }
    }
    return U;
}

template<class Elt>
bool same_howell(const elim::HowellResult<Elt>& a, const ref::RefHowell<Elt>& b, uint32_t n) {
    if (a.pivots != b.pivots || a.pivot_v != b.pivot_v || a.H.rows != b.rows.size() || a.H.cols != n) return false;
    for (uint32_t i = 0; i < a.H.rows; ++i)
        for (uint32_t j = 0; j < n; ++j)
            if (a.H.at(i, j) != b.rows[i][j]) return false;
    return true;
}
template<class Elt>
bool same_howell(const elim::HowellResult<Elt>& a, const elim::HowellResult<Elt>& b) {
    if (a.pivots != b.pivots || a.pivot_v != b.pivot_v || a.H.rows != b.H.rows || a.H.cols != b.H.cols) return false;
    for (uint32_t i = 0; i < a.H.rows; ++i)
        for (uint32_t j = 0; j < a.H.cols; ++j)
            if (a.H.at(i, j) != b.H.at(i, j)) return false;
    return true;
}

// Conditions 1-3 of Sec. 13.5, checked directly.
template<class Elt>
bool howell_conditions(const ModCtx<Elt>& c, const elim::HowellResult<Elt>& h) {
    for (uint32_t i = 0; i < h.rows(); ++i) {
        const uint32_t j = h.pivots[i];
        if (i > 0 && h.pivots[i - 1] >= j) return false;                   // 1. strictly increasing
        for (uint32_t q = 0; q < j; ++q) if (h.H.at(i, q) != 0) return false;   // 1. leading column
        if (h.pivot_v[i] >= c.k || h.H.at(i, j) != c.ell_pow(h.pivot_v[i])) return false;   // 2.
        for (uint32_t ip = 0; ip < i; ++ip)                                  // 3. reduced above
            if (h.H.at(ip, j) >= c.ell_pow(h.pivot_v[i])) return false;
    }
    return true;
}

// ============================================================================
// (R) the ring
// ============================================================================
template<class Elt>
void ring_arm(Elt ell, uint32_t k, engine::Regime want_regime, uint64_t n_samples) {
    const uint64_t f0 = g_fail, c0 = g_checks;
    const ModCtx<Elt> c = ModCtx<Elt>::prime_power(ell, k);
    u128 pp = 1;
    for (uint32_t i = 0; i < k; ++i) pp *= ell;
    const std::string t = tag_of(c);
    check(!c.is_field() && c.ell == ell && c.k == k && static_cast<u128>(c.p) == pp, t + ": declared ring members");
    check(c.regime == want_regime, t + ": regime " + engine::regime_name(c.regime) + " expected " +
                                       engine::regime_name(want_regime));
    {   // the same Sec. 1 constants as init(p) on the bare modulus
        ModCtx<Elt> bare; bare.init(c.p);
        check(bare.p2 == c.p2 && bare.has_p2 == c.has_p2 && bare.barrett_m == c.barrett_m && bare.k_max == c.k_max &&
              bare.has_montgomery == c.has_montgomery && bare.mont_ninv == c.mont_ninv &&
              bare.mont_r1 == c.mont_r1 && bare.mont_r2 == c.mont_r2 && bare.regime == c.regime,
              t + ": Sec. 1 constants == init(ell^k)'s");
        check(bare.is_field() && bare.k == 1 && bare.ell == c.p, t + ": init(p) declares a field (never inferred)");
    }
    for (uint32_t v = 0; v <= k; ++v) {
        u128 e = 1;
        for (uint32_t i = 0; i < v; ++i) e *= ell;
        check(static_cast<u128>(c.ell_pow(v)) == e, t + ": ell_pow(" + str(v) + ")");
    }
    check(throws_logic([&] { (void)c.ell_pow(k + 1); }), t + ": ell_pow(k + 1) must ENGINE_CHECK");

    Splitmix rng(0x5EED0000ULL ^ (static_cast<uint64_t>(ell) * 131 + k));
    const Elt p = c.p;
    for (uint64_t s = 0; s < n_samples; ++s) {
        const Elt a = static_cast<Elt>(rng.below(p)), b = static_cast<Elt>(rng.below(p));
        if (c.mul(a, b) != ModArithmetic::mod_mul_w(a, b, p) || c.add(a, b) != ModArithmetic::mod_add_w(a, b, p) ||
            c.sub(a, b) != ModArithmetic::mod_sub_w(a, b, p)) {
            check(false, t + ": mul/add/sub vs ModArithmetic a=" + str(a) + " b=" + str(b));
            break;
        }
        const typename ModCtx<Elt>::Accum x = static_cast<typename ModCtx<Elt>::Accum>(a) * b + a;
        if (c.reduce(x) != static_cast<Elt>(x % p)) { check(false, t + ": reduce a=" + str(a)); break; }
    }
    check(true, t + ": " + str(n_samples) + " mul/add/sub/reduce samples");

    // inv on units; the named error on non-units
    const std::string no_inv = "Modular inverse does not exist.";
    for (uint32_t s = 0; s < 2000; ++s) {
        Elt u;
        do { u = static_cast<Elt>(rng.below(p)); } while (u % ell == 0);
        const Elt ui = c.inv(u);
        if (c.mul(u, ui) != 1 || ui != ModArithmetic::mod_inv_w(u, p)) { check(false, t + ": inv on unit " + str(u)); break; }
    }
    if (static_cast<uint64_t>(p) <= 1000) {
        bool all = true;
        for (Elt x = 0; x < p; x += ell) all = all && throws_runtime_msg([&] { (void)c.inv(x); }, no_inv);
        check(all, t + ": inv throws the named error on EVERY non-unit (exhaustive)");
    } else {
        bool all = throws_runtime_msg([&] { (void)c.inv(0); }, no_inv);
        for (uint32_t v = 1; v < k; ++v)
            for (uint32_t s = 0; s < 100; ++s) {
                const Elt x = c.mul(c.ell_pow(v), static_cast<Elt>(rng.below(p)) | 1u);
                if (x == 0) continue;
                all = all && throws_runtime_msg([&] { (void)c.inv(x); }, no_inv);
            }
        check(all, t + ": inv throws the named error on 0 and on 100 non-units per valuation 1..k-1");
    }
    // valuation / unit_part / divide_exact against integer arithmetic
    bool ok = c.valuation(0) == k && c.unit_part(0) == 1;
    for (uint32_t s = 0; s < 20000 && ok; ++s) {
        const Elt x = (s < 64) ? static_cast<Elt>(s % p) : rand_elt(rng, c);
        if (x == 0) continue;
        uint32_t v = 0; u128 y = x;
        while (y % ell == 0) { y /= ell; ++v; }
        ok = ok && c.valuation(x) == v && static_cast<u128>(c.unit_part(x)) == y && (y % ell) != 0;
        for (uint32_t w = 0; w <= v && ok; ++w) {
            const Elt q = c.divide_exact(x, c.ell_pow(w));
            ok = ok && static_cast<u128>(q) * c.ell_pow(w) == static_cast<u128>(x) && q < c.ell_pow(k - w);
        }
        if (v + 1 <= k) ok = ok && throws_logic([&] { (void)c.divide_exact(x, c.ell_pow(v + 1)); });
    }
    check(ok, t + ": valuation / unit_part / divide_exact vs integer arithmetic");
    if (ell > 2) check(throws_logic([&] { (void)c.divide_exact(Elt(0), Elt(ell + 1)); }),
                       t + ": divide_exact by a non-power of ell must ENGINE_CHECK");
    check(c.divide_exact(Elt(0), c.ell_pow(k)) == 0, t + ": divide_exact(0, ell^k) == 0");
    // prime_power(ell, 1) rejected; init() resets the ring (review m5)
    check(throws_logic([&] { (void)ModCtx<Elt>::prime_power(ell, 1); }), t + ": prime_power(ell, 1) rejected");
    check(throws_logic([&] { (void)ModCtx<Elt>::prime_power(ell, 0); }), t + ": prime_power(ell, 0) rejected");
    {
        ModCtx<Elt> r = c;
        r.init(Elt(7));
        check(r.is_field() && r.ell == 7 && r.k == 1 && r.p == 7 && r.valuation(3) == 0 && r.unit_part(3) == 3,
              t + ": init(7) on a live composite context resets the ring to F_7");
    }
    std::cout << "  (R) " << t << "  regime " << engine::regime_name(c.regime) << "  k_max " << c.k_max
              << "  seed 0x" << std::hex << (0x5EED0000ULL ^ (static_cast<uint64_t>(ell) * 131 + k)) << std::dec
              << "  checks " << (g_checks - c0) << "  failing " << (g_fail - f0) << std::endl;
}

// Every Sec. 13.3 entry point throws logic_error on a composite context;
// every Sec. 13.6 entry point throws on a field context.
template<class Elt>
void guard_arm(const ModCtx<Elt>& comp, const ModCtx<Elt>& field) {
    const std::string t = tag_of(comp);
    DenseBuf<Elt> A(3, 3), R(3, 3), U(3, 3), Ai(3, 3), Y(3, 1), P;
    for (uint32_t i = 0; i < 3; ++i) A.at(i, i) = 1;
    Y.at(0, 0) = 1;
    const std::vector<uint32_t> piv = {0, 1, 2};
    // Sec. 13.3 on the composite context
    check(throws_logic([&] { (void)comp.pow(Elt(2), Elt(3)); }), t + ": ModCtx::pow must refuse (13.3, wholesale)");
    check(throws_logic([&] { (void)comp.pow(Elt(1), Elt(0)); }), t + ": ModCtx::pow refuses every exponent (e = 0)");
    check(throws_logic([&] { std::vector<Elt> x = {1, 1}; comp.batch_inv(x); }), t + ": ModCtx::batch_inv must refuse");
    check(throws_logic([&] { DenseBuf<Elt> B = A; (void)elim::rref<Elt>(comp, engine::view(B)); }), t + ": elim::rref must refuse");
    check(throws_logic([&] { DenseBuf<Elt> B = A; (void)elim::column_rank_profile<Elt>(comp, engine::view(B)); }),
          t + ": elim::column_rank_profile must refuse");
    check(throws_logic([&] { (void)elim::rref_with_transform<Elt>(comp, engine::cview(A), engine::view(R), engine::view(U)); }),
          t + ": elim::rref_with_transform must refuse");
    check(throws_logic([&] { (void)elim::kernel_basis<Elt>(comp, engine::cview(A), piv); }), t + ": elim::kernel_basis must refuse");
    check(throws_logic([&] { (void)elim::inverse<Elt>(comp, engine::cview(A), engine::view(Ai)); }), t + ": elim::inverse must refuse");
    check(throws_logic([&] { (void)elim::solve_multi_rhs<Elt>(comp, engine::cview(A), engine::cview(Y)); }),
          t + ": elim::solve_multi_rhs must refuse");
    check(throws_logic([&] { (void)elim::complement_basis<Elt>(comp, engine::cview(Y), engine::cview(A), &P); }),
          t + ": elim::complement_basis must refuse");
    // inv is unit-only, NOT guarded (13.3 / F1)
    check(!throws_logic([&] { (void)comp.inv(Elt(1)); }) && comp.inv(Elt(1)) == 1, t + ": ModCtx::inv is not field-only");
    // the same calls succeed on the field context (the guard is true there)
    check(!throws_logic([&] { (void)field.pow(Elt(2), Elt(3)); std::vector<Elt> x = {1, 1}; field.batch_inv(x);
                              DenseBuf<Elt> B = A; (void)elim::rref<Elt>(field, engine::view(B)); }),
          t + ": the field context passes the same guards");
    // Sec. 13.6 on the field context
    const std::string f = "F_" + str(field.p);
    check(throws_logic([&] { (void)elim::howell<Elt>(field, engine::cview(A)); }), f + ": elim::howell must refuse a field");
    check(throws_logic([&] { (void)elim::kernel_module<Elt>(field, engine::cview(A)); }), f + ": elim::kernel_module must refuse a field");
    check(throws_logic([&] { (void)elim::image_module<Elt>(field, engine::cview(A)); }), f + ": elim::image_module must refuse a field");
    check(throws_logic([&] { (void)elim::solve_multi_rhs_module<Elt>(field, engine::cview(A), engine::cview(Y)); }),
          f + ": elim::solve_multi_rhs_module must refuse a field");
    check(throws_logic([&] { (void)elim::length<Elt>(field, engine::cview(A)); }), f + ": elim::length must refuse a field");
    check(throws_logic([&] { (void)elim::length<Elt>(field, elim::HowellResult<Elt>()); }),
          f + ": elim::length(HowellResult) must refuse a field");
    check(throws_logic([&] { (void)elim::invariant_factors<Elt>(field, engine::cview(A)); }),
          f + ": elim::invariant_factors must refuse a field");
    check(throws_logic([&] { (void)elim::invariant_factors<Elt>(field, engine::cview(A), engine::cview(Y)); }),
          f + ": elim::invariant_factors(G, Rl) must refuse a field");
}

// Sec. 13.7: the replay check.  A program recorded under one ring never runs
// on a binding of the other; same-ring replays run and are exact.
template<class Elt>
void replay_arm(const ModCtx<Elt>& field, const ModCtx<Elt>& field2, const ModCtx<Elt>& comp, const ModCtx<Elt>& comp2) {
    using namespace engine::tape;
    const std::string t = tag_of(comp);
    Splitmix rng(0xBEEF0001ULL);
    const uint32_t m = 7, kk = 5, n = 4;
    DenseBuf<Elt> A(m, kk), B(kk, n);
    // entries < min modulus so that one input set is valid under all four rings
    Elt pmin = std::min(std::min(field.p, field2.p), std::min(comp.p, comp2.p));
    for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < kk; ++j) A.at(i, j) = static_cast<Elt>(rng.below(pmin));
    for (uint32_t i = 0; i < kk; ++i) for (uint32_t j = 0; j < n; ++j) B.at(i, j) = static_cast<Elt>(rng.below(pmin));
    auto run = [&](const ModCtx<Elt>& rec_ctx, const ModCtx<Elt>* bind_ctx, DenseBuf<Elt>& C) {
        Recorder<Elt> rec(rec_ctx);
        const BufId ba = rec.input(engine::cview(A)), bb = rec.input(engine::cview(B)), bc = rec.output(engine::view(C));
        rec.gemm(rec.whole(ba), rec.whole(bb), rec.whole(bc));
        Program<Elt> prog = rec.finish();
        compile(prog, PassOptions());
        Bindings<Elt> b = rec.bindings();
        if (bind_ctx != nullptr) b.set_ctx(*bind_ctx);
        Executor<Elt> ex;
        ex.run(static_cast<const Program<Elt>&>(prog), b);
    };
    auto expect = [&](const ModCtx<Elt>& c) { return mul<Elt>(c.p, engine::cview(A), engine::cview(B)); };
    auto equal = [](const DenseBuf<Elt>& x, const DenseBuf<Elt>& y) {
        for (uint32_t i = 0; i < x.rows; ++i) for (uint32_t j = 0; j < x.cols; ++j) if (x.at(i, j) != y.at(i, j)) return false;
        return true;
    };
    check(field.regime == comp.regime, t + ": replay arm needs one regime for all four contexts");
    {
        DenseBuf<Elt> C(m, n);
        check(throws_logic([&] { run(field, &comp, C); }), t + ": a FIELD program never replays on a composite binding (13.7)");
    }
    {
        DenseBuf<Elt> C(m, n);
        check(throws_logic([&] { run(comp, &field, C); }), t + ": a composite program never replays on a field binding (13.7)");
    }
    {
        DenseBuf<Elt> C(m, n);
        run(field, &field2, C);
        check(equal(C, expect(field2)), t + ": field -> field replay (F_" + str(field.p) + " -> F_" + str(field2.p) + ") still runs, exact");
    }
    {
        DenseBuf<Elt> C(m, n);
        run(comp, &comp2, C);
        check(equal(C, expect(comp2)), t + ": composite -> composite replay (" + tag_of(comp) + " -> " + tag_of(comp2) + ") runs, exact");
    }
    {
        DenseBuf<Elt> C(m, n);
        run(comp, nullptr, C);
        check(equal(C, expect(comp)), t + ": a composite program on its own context is exact (GEMM is modulus-only, 13.2)");
    }
}

// ============================================================================
// (H) Howell, (D) derived
// ============================================================================
struct Shape { uint32_t m, n; };

template<class Elt>
void howell_arm(Elt ell, uint32_t k, uint32_t n_random, uint64_t seed) {
    const uint64_t f0 = g_fail, c0 = g_checks;
    const ModCtx<Elt> c = ModCtx<Elt>::prime_power(ell, k);
    const std::string t = tag_of(c);
    Splitmix rng(seed);
    const Shape shapes[] = {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {3, 3}, {4, 2}, {2, 5}, {5, 5}, {6, 4}, {4, 7}, {8, 8}, {9, 5}, {3, 10}, {12, 12}, {16, 10}, {10, 16}};
    std::vector<DenseBuf<Elt>> cases;
    // structured
    cases.push_back(DenseBuf<Elt>(4, 5));                                   // zero
    { DenseBuf<Elt> I(5, 5); for (uint32_t i = 0; i < 5; ++i) I.at(i, i) = 1; cases.push_back(I); }   // identity
    { DenseBuf<Elt> E = rand_mat(rng, c, 5, 6);                            // ell-multiples
      for (uint32_t i = 0; i < 5; ++i) for (uint32_t j = 0; j < 6; ++j) E.at(i, j) = c.mul(E.at(i, j), c.ell_pow(1));
      cases.push_back(E); }
    for (uint32_t v = 1; v < k; ++v) {                                    // closure type [[ell^v, 1]] and a 2-row variant
        DenseBuf<Elt> C1(1, 2); C1.at(0, 0) = c.ell_pow(v); C1.at(0, 1) = 1; cases.push_back(C1);
        DenseBuf<Elt> C2(2, 3); C2.at(0, 0) = c.ell_pow(v); C2.at(0, 1) = 1; C2.at(1, 1) = c.ell_pow(k - v); C2.at(1, 2) = c.ell_pow(v);
        cases.push_back(C2);
    }
    {   // rank-deficient: (6 x 2)(2 x 7)
        const DenseBuf<Elt> L = rand_mat(rng, c, 6, 2), Rr = rand_mat(rng, c, 2, 7);
        cases.push_back(mul<Elt>(c.p, engine::cview(L), engine::cview(Rr)));
    }
    cases.push_back(rand_mat(rng, c, 12, 3));                              // tall
    cases.push_back(rand_mat(rng, c, 3, 12));                              // wide
    for (uint32_t r = 0; r < n_random; ++r) {
        const Shape s = shapes[rng.below(sizeof(shapes) / sizeof(shapes[0]))];
        cases.push_back(rand_mat(rng, c, s.m, s.n));
    }
    uint32_t n_diff = 0, n_uniq = 0, n_derived = 0, n_inv = 0, n_closure = 0, n_nonunit_piv = 0, n_mixed = 0;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        try {
            const DenseBuf<Elt>& A = cases[ci];
            const uint32_t m = A.rows, n = A.cols;
            const std::string cs = t + " case " + str(ci) + " (" + str(m) + "x" + str(n) + ")";
            // --- elim::howell == ref::howell entrywise; conditions 1-3
            const elim::HowellResult<Elt> H = elim::howell<Elt>(c, engine::cview(A));
            const ref::RefHowell<Elt> Hr = ref::howell<Elt>(ell, k, engine::cview(A));
            check(same_howell(H, Hr, n), cs + ": elim::howell == ref::howell entrywise");
            check(howell_conditions(c, H), cs + ": Howell conditions 1-3");
            ++n_diff;
            if (H.rows() > m) ++n_closure;                          // more rows than A: a closure row pivoted
            for (uint32_t v : H.pivot_v) if (v > 0) { ++n_nonunit_piv; break; }
            // --- uniqueness: row permutation, unimodular left factor, appended span rows
            {
                DenseBuf<Elt> Pm(m, n);
                std::vector<uint32_t> perm(m);
                for (uint32_t i = 0; i < m; ++i) perm[i] = i;
                for (uint32_t i = m; i > 1; --i) std::swap(perm[i - 1], perm[rng.below(i)]);
                for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) Pm.at(i, j) = A.at(perm[i], j);
                check(same_howell(elim::howell<Elt>(c, engine::cview(Pm)), H), cs + ": unique under a row permutation");
                const DenseBuf<Elt> U = rand_unimodular(rng, c, m);
                const DenseBuf<Elt> UA = mul<Elt>(c.p, engine::cview(U), engine::cview(A));
                check(same_howell(elim::howell<Elt>(c, engine::cview(UA)), H), cs + ": unique under a unimodular left factor");
                const DenseBuf<Elt> X = rand_mat(rng, c, 3, m);
                const DenseBuf<Elt> XA = mul<Elt>(c.p, engine::cview(X), engine::cview(A));
                DenseBuf<Elt> Ap(m + 3, n);
                for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) Ap.at(i, j) = A.at(i, j);
                for (uint32_t i = 0; i < 3; ++i) for (uint32_t j = 0; j < n; ++j) Ap.at(m + i, j) = XA.at(i, j);
                check(same_howell(elim::howell<Elt>(c, engine::cview(Ap)), H), cs + ": unique under appending rows of the span");
                ++n_uniq;
            }
            // --- kernel_module
            const elim::HowellResult<Elt> K = elim::kernel_module<Elt>(c, engine::cview(A));
            {
                bool zero = true;
                for (uint32_t r = 0; r < K.rows() && zero; ++r)
                    for (uint32_t i = 0; i < m && zero; ++i) {
                        Elt s = 0;
                        for (uint32_t j = 0; j < n; ++j) s = c.add(s, c.mul(A.at(i, j), K.H.at(r, j)));
                        zero = (s == 0);
                    }
                check(zero, cs + ": kernel_module rows satisfy A x = 0");
                const DenseBuf<Elt> At = transpose_of<Elt>(engine::cview(A));
                DenseBuf<Elt> Aug(n, m + n);
                for (uint32_t i = 0; i < n; ++i) { for (uint32_t j = 0; j < m; ++j) Aug.at(i, j) = At.at(i, j); Aug.at(i, m + i) = 1; }
                const ref::RefHowell<Elt> Ka = ref::howell<Elt>(ell, k, engine::cview(Aug));
                ref::RefHowell<Elt> Kr;
                for (size_t i = 0; i < Ka.rows.size(); ++i) {
                    if (Ka.pivots[i] < m) continue;
                    Kr.rows.emplace_back(Ka.rows[i].begin() + m, Ka.rows[i].end());
                    Kr.pivots.push_back(Ka.pivots[i] - m);
                    Kr.pivot_v.push_back(Ka.pivot_v[i]);
                }
                check(same_howell(K, Kr, n), cs + ": kernel_module == ref::howell([A^T | I]) restricted");
                check(same_howell(elim::howell<Elt>(c, engine::cview(K.H)), K), cs + ": kernel_module is a Howell form");
            }
            // --- image_module
            const elim::HowellResult<Elt> Im = elim::image_module<Elt>(c, engine::cview(A));
            {
                const DenseBuf<Elt> At = transpose_of<Elt>(engine::cview(A));
                check(same_howell(Im, ref::howell<Elt>(ell, k, engine::cview(At)), m), cs + ": image_module == ref::howell(A^T)");
                // rank-nullity, and |ker| |im| = |R^n| makes the kernel the WHOLE kernel
                check(elim::length<Elt>(c, K) + elim::length<Elt>(c, Im) == n * k, cs + ": length(ker) + length(im) == n k");
                check(elim::length<Elt>(c, engine::cview(A)) == elim::length<Elt>(c, H), cs + ": length(ctx, A) == length(howell(A))");
                // mutual membership: A's columns in span(Im rows) and Im rows in colspan A
                const DenseBuf<Elt> Gen = transpose_of<Elt>(engine::cview(Im.H));      // m x s
                bool ok = true;
                try {
                    (void)elim::solve_multi_rhs_module<Elt>(c, engine::cview(Gen), engine::cview(A));
                    (void)elim::solve_multi_rhs_module<Elt>(c, engine::cview(A), engine::cview(Gen));
                } catch (const std::exception&) { ok = false; }
                check(ok, cs + ": image_module and colspan A contain each other");
            }
            // --- solve_multi_rhs_module
            {
                const DenseBuf<Elt> X0 = rand_mat(rng, c, n, 3);
                const DenseBuf<Elt> Y = mul<Elt>(c.p, engine::cview(A), engine::cview(X0));
                const auto X = elim::solve_multi_rhs_module<Elt>(c, engine::cview(A), engine::cview(Y));
                bool exact = X.size() == 3, reduced = true;
                for (uint32_t t2 = 0; t2 < X.size() && exact; ++t2) {
                    for (uint32_t i = 0; i < m; ++i) {
                        Elt s = 0;
                        for (uint32_t j = 0; j < n; ++j) s = c.add(s, c.mul(A.at(i, j), X[t2][j]));
                        exact = exact && s == Y.at(i, t2);
                    }
                    for (uint32_t r = 0; r < K.rows(); ++r) reduced = reduced && X[t2][K.pivots[r]] < c.ell_pow(K.pivot_v[r]);
                }
                check(exact, cs + ": solve_multi_rhs_module: B X == Y exactly");
                check(reduced, cs + ": solve_multi_rhs_module: X reduced modulo ker B");
                // independence of the preimage: X0 + (kernel element) gives the same X
                if (K.rows() > 0) {
                    DenseBuf<Elt> X1 = X0;
                    for (uint32_t t2 = 0; t2 < 3; ++t2) {
                        const Elt f = static_cast<Elt>(rng.below(c.p));
                        const uint32_t r = static_cast<uint32_t>(rng.below(K.rows()));
                        for (uint32_t j = 0; j < n; ++j) X1.at(j, t2) = c.add(X1.at(j, t2), c.mul(f, K.H.at(r, j)));
                    }
                    const DenseBuf<Elt> Y1 = mul<Elt>(c.p, engine::cview(A), engine::cview(X1));
                    check(elim::solve_multi_rhs_module<Elt>(c, engine::cview(A), engine::cview(Y1)) == X,
                          cs + ": solve_multi_rhs_module independent of the preimage (unique reduced representative)");
                }
                ++n_derived;
            }
            // --- invariant_factors vs the oracle; sum e == length difference
            {
                const DenseBuf<Elt> Rl = rand_mat(rng, c, m, static_cast<uint32_t>(rng.below(3)));
                const auto e = elim::invariant_factors<Elt>(c, engine::cview(A), engine::cview(Rl));
                const auto er = ref::invariant_factors<Elt>(ell, k, engine::cview(A), engine::cview(Rl));
                check(e == er, cs + ": invariant_factors(G, Rl) == ref::invariant_factors");
                if (!e.empty() && e.front() != e.back()) ++n_mixed;
                const DenseBuf<Elt> none(m, 0);
                const auto e0 = elim::invariant_factors<Elt>(c, engine::cview(A));
                check(e0 == ref::invariant_factors<Elt>(ell, k, engine::cview(A), engine::cview(none)),
                      cs + ": invariant_factors(A) == ref::invariant_factors");
                uint32_t sum = 0;
                for (uint32_t x : e0) sum += x;
                check(sum == elim::length<Elt>(c, Im), cs + ": sum e_i == length(colspan A)");
                // Rl inside span G: M = colspan G / colspan Rl, length difference
                const DenseBuf<Elt> Z = rand_mat(rng, c, n, 2);
                const DenseBuf<Elt> Rin = mul<Elt>(c.p, engine::cview(A), engine::cview(Z));
                const auto e2 = elim::invariant_factors<Elt>(c, engine::cview(A), engine::cview(Rin));
                check(e2 == ref::invariant_factors<Elt>(ell, k, engine::cview(A), engine::cview(Rin)),
                      cs + ": invariant_factors(G, Rl in span G) == ref");
                uint32_t sum2 = 0;
                for (uint32_t x : e2) sum2 += x;
                check(sum2 + elim::length<Elt>(c, elim::image_module<Elt>(c, engine::cview(Rin))) == elim::length<Elt>(c, Im),
                      cs + ": sum e_i == length(G) - length(Rl)");
                bool sorted = std::is_sorted(e.begin(), e.end(), [](uint32_t a, uint32_t b) { return a > b; });
                for (uint32_t x : e) sorted = sorted && x >= 1 && x <= k;
                check(sorted, cs + ": exponents in [1, k], non-increasing");
                ++n_inv;
            }
        } catch (const std::exception& ex) {   // an unexpected throw is a counted failure, not an abort
            check(false, t + " case " + str(ci) + ": unexpected exception: " + ex.what());
        }
    }
    // --- the named throw at the SMALLEST failing column
    {
        DenseBuf<Elt> B(3, 2);
        for (uint32_t i = 0; i < 3; ++i) for (uint32_t j = 0; j < 2; ++j) B.at(i, j) = c.mul(c.ell_pow(1), rand_elt(rng, c));
        DenseBuf<Elt> Y(3, 4);                     // columns 1 and 3 have a unit entry: not in im B (im B in ell R^3)
        Y.at(0, 1) = 1; Y.at(2, 3) = 1;
        check(throws_runtime_msg([&] { (void)elim::solve_multi_rhs_module<Elt>(c, engine::cview(B), engine::cview(Y)); },
                                 "Vector is not in image of basis (right-hand side 1)."),
              t + ": solve_multi_rhs_module throws the Sec. 9.4 message for the SMALLEST failing t (= 1)");
        DenseBuf<Elt> B0(2, 0), Y0(2, 2);          // nb = 0: only y = 0 is in the image
        Y0.at(1, 1) = c.ell_pow(k - 1);
        check(throws_runtime_msg([&] { (void)elim::solve_multi_rhs_module<Elt>(c, engine::cview(B0), engine::cview(Y0)); },
                                 "Vector is not in image of basis (right-hand side 1)."),
              t + ": solve_multi_rhs_module with nb = 0 throws for a non-zero y");
    }
    std::cout << "  (H)(D) " << t << "  seed 0x" << std::hex << seed << std::dec << "  cases " << cases.size()
              << " (differential " << n_diff << ", uniqueness " << n_uniq << ", solve " << n_derived
              << ", invariant_factors " << n_inv << "; with a non-unit pivot " << n_nonunit_piv
              << ", h > m " << n_closure << ", mixed exponents " << n_mixed << ")  checks " << (g_checks - c0) << "  failing " << (g_fail - f0)
              << std::endl;
}

// The field path: ref::howell(ell, 1, M) == ref::rref(ell, M) and == the
// engine's RREF without its zero rows (Sec. 13.5 "At k = 1", 13.9).
template<class Elt>
void field_path_arm(Elt ell, uint32_t n_cases, uint64_t seed) {
    const uint64_t f0 = g_fail, c0 = g_checks;
    Splitmix rng(seed);
    const ModCtx<Elt> f(ell);
    for (uint32_t ci = 0; ci < n_cases; ++ci) {
        const uint32_t m = 1 + static_cast<uint32_t>(rng.below(9)), n = 1 + static_cast<uint32_t>(rng.below(9));
        DenseBuf<Elt> A(m, n);
        const bool deficient = (ci % 3 == 0);
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < n; ++j) A.at(i, j) = (deficient && rng.below(2)) ? 0 : static_cast<Elt>(rng.below(ell));
        const ref::RefHowell<Elt> H = ref::howell<Elt>(ell, 1, engine::cview(A));
        DenseBuf<Elt> R1 = A, R2 = A;
        const std::vector<uint32_t> piv = ref::rref<Elt>(ell, engine::view(R1));
        const elim::RrefResult er = elim::rref<Elt>(f, engine::view(R2));
        bool ok = H.pivots == piv && H.pivots == er.pivots;
        for (uint32_t v : H.pivot_v) ok = ok && v == 0;
        for (size_t i = 0; i < H.rows.size() && ok; ++i)
            for (uint32_t j = 0; j < n; ++j) ok = ok && H.rows[i][j] == R1.at(static_cast<uint32_t>(i), j) &&
                                                  H.rows[i][j] == R2.at(static_cast<uint32_t>(i), j);
        check(ok, "F_" + str(ell) + " case " + str(ci) + ": ref::howell(ell, 1) == ref::rref == elim::rref (non-zero rows)");
    }
    std::cout << "  (H) field path F_" << ell << (std::is_same<Elt, uint32_t>::value ? " (u32)" : " (u64)")
              << "  seed 0x" << std::hex << seed << std::dec << "  cases " << n_cases << "  checks " << (g_checks - c0)
              << "  failing " << (g_fail - f0) << std::endl;
}

// Review M1's arm, in Sec. 13.6's column convention.
void m1_arm() {
    const ModCtx<uint32_t> c = ModCtx<uint32_t>::prime_power(2, 2);   // Z/4
    DenseBuf<uint32_t> G(2, 1);                                        // [[2], [1]]
    G.at(0, 0) = 2; G.at(1, 0) = 1;
    const auto e = elim::invariant_factors<uint32_t>(c, engine::cview(G));
    check(e == std::vector<uint32_t>{2}, "M1: invariant_factors(Z/4, [[2],[1]]) == (2), i.e. Z/4");
    const auto Im = elim::image_module<uint32_t>(c, engine::cview(G));
    check(Im.pivot_v == std::vector<uint32_t>({1, 1}) && Im.pivots == std::vector<uint32_t>({0, 1}) &&
          Im.H.at(0, 0) == 2 && Im.H.at(0, 1) == 1 && Im.H.at(1, 0) == 0 && Im.H.at(1, 1) == 2,
          "M1 (negative): image_module([[2],[1]]) = howell([[2, 1]]) = [[2, 1], [0, 2]], pivot valuations (1, 1) -- "
          "they would misread as Z/2 + Z/2");
    const auto Hg = elim::howell<uint32_t>(c, engine::cview(G));
    check(Hg.pivots == std::vector<uint32_t>{0} && Hg.pivot_v == std::vector<uint32_t>{0},
          "M1 (the trap): howell of the 2x1 matrix itself has ONE pivot, of valuation 0");
    DenseBuf<uint32_t> Row(1, 2);                                      // Sec. 13.9's literal "[[2, 1]] |-> (2)"
    Row.at(0, 0) = 2; Row.at(0, 1) = 1;
    check(elim::invariant_factors<uint32_t>(c, engine::cview(Row)) == std::vector<uint32_t>{2},
          "13.9: invariant_factors(Z/4, [[2, 1]]) == (2) (its column module is all of Z/4)");
    check(elim::length<uint32_t>(c, engine::cview(Row)) == 2, "13.5: length(rowspan[[2, 1]]) == 2 (|Z/4| = 4)");
    std::cout << "  (D) M1 arm over Z/4: invariant_factors([[2],[1]]) = (" << (e.empty() ? 0u : e[0])
              << "), image_module pivot valuations (" << (Im.pivot_v.size() > 0 ? Im.pivot_v[0] : 9u) << ", "
              << (Im.pivot_v.size() > 1 ? Im.pivot_v[1] : 9u) << ")" << std::endl;
}

// ============================================================================
// (M) the M3 basis rule, directly (slot-8 WP-3r; WP-12 pass-2 review M-1):
// cyclic_decomposition / cyclic_coordinates on fixed-seed (G, Rl), against
// the module structure itself -- not only through the pipeline cells.
// ============================================================================
template<class Elt>
bool in_colspan(const ModCtx<Elt>& c, DenseView<const Elt> Rl, const std::vector<Elt>& y) {
    DenseBuf<Elt> Y(static_cast<uint32_t>(y.size()), 1);
    for (uint32_t i = 0; i < y.size(); ++i) Y.at(i, 0) = y[i];
    try { (void)elim::solve_multi_rhs_module<Elt>(c, Rl, engine::cview(Y)); return true; }
    catch (const std::runtime_error&) { return false; }
}

struct CycVerdict {
    bool shape = true, ident = true, order = true, gen = true, coords = true;
    bool all() const { return shape && ident && order && gen && coords; }
};

// Every property of (c1)-(c5) that a wrong U mirror, a wrong Gamma mirror or a
// wrong `position` would break:
//  ident  cyclic_coordinates(dec, dec.basis) is the (reduced) identity;
//  order  ell^{e_i} b_i in colspan Rl and ell^{e_i - 1} b_i not in it;
//  gen    every column g of G: g - sum c_i b_i in colspan Rl (c = its coordinates);
//  coords x = sum a_i b_i + Rl z has coordinates a_i mod ell^{e_i} (unique, (c5)).
template<class Elt>
CycVerdict verify_cyclic(const ModCtx<Elt>& c, DenseView<const Elt> G, DenseView<const Elt> Rl,
                         const elim::CyclicDecomposition<Elt>& dec, uint64_t seed) {
    CycVerdict v;
    const uint32_t m = G.rows(), t = dec.rows();
    v.shape = dec.basis.rows == m && dec.basis.cols == t && dec.position.size() == t;
    if (!v.shape) return v;
    auto red = [&](Elt a, uint32_t e) -> Elt { return e >= c.k ? a : static_cast<Elt>(a % c.ell_pow(e)); };
    auto col = [&](uint32_t i) {
        std::vector<Elt> b(m);
        for (uint32_t r = 0; r < m; ++r) b[r] = dec.basis.at(r, i);
        return b;
    };
    auto scaled = [&](std::vector<Elt> b, Elt s) {
        for (Elt& x : b) x = c.mul(x, s);
        return b;
    };
    try {
        const auto I = elim::cyclic_coordinates<Elt>(c, dec, engine::cview(dec.basis));
        for (uint32_t j = 0; j < t && v.ident; ++j)
            for (uint32_t i = 0; i < t; ++i)
                if (I[j][i] != (i == j ? Elt(1) : Elt(0))) { v.ident = false; break; }
    } catch (const std::exception&) { v.ident = false; }
    for (uint32_t i = 0; i < t && v.order; ++i) {
        const uint32_t e = dec.exponents[i];
        if (e < 1 || e > c.k) { v.order = false; break; }
        const std::vector<Elt> b = col(i);
        if (e < c.k && !in_colspan<Elt>(c, Rl, scaled(b, c.ell_pow(e)))) v.order = false;
        if (in_colspan<Elt>(c, Rl, scaled(b, c.ell_pow(e - 1)))) v.order = false;
    }
    try {
        const auto Cg = elim::cyclic_coordinates<Elt>(c, dec, G);
        for (uint32_t j = 0; j < G.cols() && v.gen; ++j) {
            std::vector<Elt> r(m);
            for (uint32_t row = 0; row < m; ++row) {
                Elt s = G.at(row, j);
                for (uint32_t i = 0; i < t; ++i) s = c.sub(s, c.mul(Cg[j][i], dec.basis.at(row, i)));
                r[row] = s;
            }
            if (!in_colspan<Elt>(c, Rl, r)) v.gen = false;
        }
    } catch (const std::exception&) { v.gen = false; }
    try {
        Splitmix rng(seed);
        const uint32_t nx = 3;
        DenseBuf<Elt> X(m, nx);
        std::vector<std::vector<Elt>> a(nx, std::vector<Elt>(t));
        for (uint32_t x = 0; x < nx; ++x) {
            for (uint32_t i = 0; i < t; ++i) {
                a[x][i] = static_cast<Elt>(rng.below(c.p));
                for (uint32_t row = 0; row < m; ++row)
                    X.at(row, x) = c.add(X.at(row, x), c.mul(a[x][i], dec.basis.at(row, i)));
            }
            for (uint32_t j = 0; j < Rl.cols(); ++j) {
                const Elt z = static_cast<Elt>(rng.below(c.p));
                for (uint32_t row = 0; row < m; ++row) X.at(row, x) = c.add(X.at(row, x), c.mul(z, Rl.at(row, j)));
            }
        }
        const auto Cx = elim::cyclic_coordinates<Elt>(c, dec, engine::cview(X));
        for (uint32_t x = 0; x < nx && v.coords; ++x)
            for (uint32_t i = 0; i < t; ++i)
                if (Cx[x][i] != red(a[x][i], dec.exponents[i])) { v.coords = false; break; }
    } catch (const std::exception&) { v.coords = false; }
    return v;
}

template<class Elt>
void cyclic_arm(Elt ell, uint32_t k, uint32_t n_random, uint64_t seed) {
    const uint64_t f0 = g_fail, c0 = g_checks;
    const ModCtx<Elt> c = ModCtx<Elt>::prime_power(ell, k);
    const std::string t = tag_of(c);
    Splitmix rng(seed);
    struct Case { DenseBuf<Elt> G, Rl; std::vector<uint32_t> known; bool has_known; };
    std::vector<Case> cases;
    // Known answer: G = U (unimodular), Rl = U diag(ell^{v_i}) V, so M = R^m /
    // colspan Rl = (+) R/ell^{v_i}: the exponents are the v_i >= 1 (v = k: a
    // zero diagonal entry, a free summand Z/ell^k), descending.
    for (uint32_t r = 0; r < 24; ++r) {
        const uint32_t m = 1 + static_cast<uint32_t>(rng.below(7));
        const DenseBuf<Elt> U = rand_unimodular(rng, c, m), V = rand_unimodular(rng, c, m);
        DenseBuf<Elt> D(m, m);
        std::vector<uint32_t> known;
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t vv = static_cast<uint32_t>(rng.below(c.k + 1));
            D.at(i, i) = (vv == c.k) ? Elt(0) : c.ell_pow(vv);
            if (vv >= 1) known.push_back(vv);
        }
        std::sort(known.begin(), known.end(), [](uint32_t a, uint32_t b) { return a > b; });
        const DenseBuf<Elt> UD = mul<Elt>(c.p, engine::cview(U), engine::cview(D));
        cases.push_back({U, mul<Elt>(c.p, engine::cview(UD), engine::cview(V)), known, true});
    }
    // Rl empty (M = colspan G); Rl inside span G; ell-multiples of such (torsion); Rl random.
    const Shape shapes[] = {{1, 1}, {2, 1}, {2, 2}, {3, 3}, {4, 2}, {5, 5}, {6, 4}, {4, 7}, {8, 6}, {9, 5}, {10, 10}};
    for (uint32_t r = 0; r < n_random; ++r) {
        const Shape s = shapes[rng.below(sizeof(shapes) / sizeof(shapes[0]))];
        const DenseBuf<Elt> G = rand_mat(rng, c, s.m, s.n);
        const uint32_t kind = static_cast<uint32_t>(rng.below(4)), nr = static_cast<uint32_t>(rng.below(5));
        DenseBuf<Elt> Rl(s.m, 0);
        if (kind == 1 || kind == 2) {
            const DenseBuf<Elt> Z = rand_mat(rng, c, s.n, nr);
            Rl = mul<Elt>(c.p, engine::cview(G), engine::cview(Z));
            if (kind == 2)
                for (uint32_t i = 0; i < Rl.rows; ++i)
                    for (uint32_t j = 0; j < Rl.cols; ++j)
                        Rl.at(i, j) = c.mul(Rl.at(i, j), c.ell_pow(1 + (i + j) % (c.k - 1)));
        } else if (kind == 3) {
            Rl = rand_mat(rng, c, s.m, nr);
        }
        cases.push_back({G, Rl, {}, false});
    }
    uint32_t n_sum = 0, n_mixed = 0, n_multi = 0;
    std::vector<elim::CyclicDecomposition<Elt>> decs;
    decs.reserve(cases.size());
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const std::string cs = t + " (M) case " + str(ci) + " (" + str(cases[ci].G.rows) + "x" + str(cases[ci].G.cols) +
                               ", Rl " + str(cases[ci].Rl.cols) + ")";
        try {
            const auto G = engine::cview(cases[ci].G);
            const auto Rl = engine::cview(cases[ci].Rl);
            decs.push_back(elim::cyclic_decomposition<Elt>(c, G, Rl));
            const auto& dec = decs.back();
            const auto e = elim::invariant_factors<Elt>(c, G, Rl);
            check(dec.exponents == e, cs + ": exponents == invariant_factors(G, Rl)");
            check(e == ref::invariant_factors<Elt>(ell, k, G, Rl), cs + ": invariant_factors == ref::invariant_factors");
            if (cases[ci].has_known) check(dec.exponents == cases[ci].known, cs + ": exponents == the known diag(ell^v) answer");
            const CycVerdict v = verify_cyclic<Elt>(c, G, Rl, dec, seed ^ (0x5151ULL * (ci + 1)));
            check(v.shape, cs + ": basis m x t, one position per summand");
            check(v.ident, cs + ": cyclic_coordinates(dec, basis) == I");
            check(v.order, cs + ": ell^{e_i} b_i in colspan Rl, ell^{e_i - 1} b_i not");
            check(v.gen, cs + ": the b_i generate colspan G modulo colspan Rl");
            check(v.coords, cs + ": coordinates of sum a_i b_i + Rl z are a_i mod ell^{e_i}");
            ++n_sum;
            if (dec.rows() >= 2) ++n_multi;
            if (!dec.exponents.empty() && dec.exponents.front() != dec.exponents.back()) ++n_mixed;
        } catch (const std::exception& ex) {
            check(false, cs + ": unexpected exception: " + ex.what());
            decs.push_back(elim::CyclicDecomposition<Elt>());
        }
    }
    // Negative controls: the verdict REJECTS a decomposition corrupted the way a
    // wrong U mirror (DenseElim.cpp, row_i -= q row_t), a wrong Gamma mirror
    // (gamma_t += q gamma_i) or a wrong `position` would corrupt it.
    uint32_t n_neg = 0;
    for (size_t ci = 0; ci < cases.size() && n_neg < 6; ++ci) {
        if (decs[ci].rows() < 2) continue;
        const auto G = engine::cview(cases[ci].G);
        const auto Rl = engine::cview(cases[ci].Rl);
        const uint64_t s2 = seed ^ 0xBADULL;
        {   // U: row position[0] += row position[1]
            elim::CyclicDecomposition<Elt> bad = decs[ci];
            const uint32_t a = bad.position[0], b = bad.position[1];
            for (uint32_t j = 0; j < bad.U.cols; ++j) bad.U.at(a, j) = c.add(bad.U.at(a, j), bad.U.at(b, j));
            check(!verify_cyclic<Elt>(c, G, Rl, bad, s2).all(), t + " (M) negative control: a wrong U mirror is caught");
        }
        {   // Gamma: b_0 += b_1
            elim::CyclicDecomposition<Elt> bad = decs[ci];
            for (uint32_t r = 0; r < bad.basis.rows; ++r) bad.basis.at(r, 0) = c.add(bad.basis.at(r, 0), bad.basis.at(r, 1));
            check(!verify_cyclic<Elt>(c, G, Rl, bad, s2).all(), t + " (M) negative control: a wrong Gamma mirror is caught");
        }
        {   // position: swap the first two
            elim::CyclicDecomposition<Elt> bad = decs[ci];
            std::swap(bad.position[0], bad.position[1]);
            check(!verify_cyclic<Elt>(c, G, Rl, bad, s2).all(), t + " (M) negative control: a wrong position is caught");
        }
        ++n_neg;
    }
    check(n_neg >= 1, t + " (M) at least one case with >= 2 summands for the negative controls");
    std::cout << "  (M) " << t << "  seed 0x" << std::hex << seed << std::dec << "  cases " << cases.size()
              << " (verified " << n_sum << ", >= 2 summands " << n_multi << ", mixed exponents " << n_mixed
              << ", negative-control cases " << n_neg << ")  checks " << (g_checks - c0) << "  failing " << (g_fail - f0)
              << std::endl;
}

} // namespace

uint32_t HeckeReductionValidation() {
    g_fail = 0;
    g_checks = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeReductionValidation: E8 item 2 -- Z/ell^k ring, Howell / Smith elimination "
                 "(ENGINE_API.md Sec. 13.9 elimination arms)" << std::endl;
    using engine::Regime;
    const uint64_t N = 1000000;

    // (R) the ring -- Sec. 13.9's (5,2), (5,3), (691,2), (134217757,2) plus WP-1's moduli
    ring_arm<uint32_t>(2u, 2, Regime::U32_Delayed, N);
    ring_arm<uint32_t>(2u, 3, Regime::U32_Delayed, N);
    ring_arm<uint32_t>(3u, 2, Regime::U32_Delayed, N);
    ring_arm<uint32_t>(5u, 2, Regime::U32_Delayed, N);
    ring_arm<uint32_t>(5u, 3, Regime::U32_Delayed, N);
    ring_arm<uint32_t>(691u, 2, Regime::U32_Delayed, N);
    ring_arm<uint64_t>(134217757ull, 2, Regime::U64_Montgomery, N);
    ring_arm<uint64_t>(2ull, 40, Regime::U64_Wide, N);
    {   // overflow of ell^k is rejected
        check(throws_logic([] { (void)ModCtx<uint32_t>::prime_power(2u, 32); }), "prime_power<u32>(2, 32) rejected (2^32 does not fit)");
        check(throws_logic([] { (void)ModCtx<uint32_t>::prime_power(691u, 4); }), "prime_power<u32>(691, 4) rejected");
        check(throws_logic([] { (void)ModCtx<uint64_t>::prime_power(134217757ull, 3); }), "prime_power<u64>(134217757, 3) rejected");
        check(throws_logic([] { (void)ModCtx<uint32_t>::prime_power(1u, 2); }), "prime_power(1, 2) rejected");
        const ModCtx<uint32_t> f(5u);
        check(f.is_field() && f.ell == 5 && f.k == 1 && f.valuation(0) == 1 && f.valuation(3) == 0 && f.unit_part(4) == 4 &&
              f.ell_pow(0) == 1 && f.ell_pow(1) == 5, "ModCtx(5): is_field, ell == p, k == 1, consistent Sec. 13.4 members");
        std::cout << "  (R) sizeof(ModCtx<uint32_t>) = " << sizeof(ModCtx<uint32_t>)
                  << ", sizeof(ModCtx<uint64_t>) = " << sizeof(ModCtx<uint64_t>)
                  << " (trivially copyable: static_assert in ModCtx.h)" << std::endl;
    }
    // every Sec. 13.3 entry refuses a composite context, every Sec. 13.6 entry a field
    guard_arm<uint32_t>(ModCtx<uint32_t>::prime_power(5u, 2), ModCtx<uint32_t>(5u));
    guard_arm<uint64_t>(ModCtx<uint64_t>::prime_power(134217757ull, 2), ModCtx<uint64_t>(134217757ull));
    guard_arm<uint64_t>(ModCtx<uint64_t>::prime_power(2ull, 40), ModCtx<uint64_t>(2ull));
    // Sec. 13.7 replay
    replay_arm<uint32_t>(ModCtx<uint32_t>(5u), ModCtx<uint32_t>(7u), ModCtx<uint32_t>::prime_power(5u, 2),
                         ModCtx<uint32_t>::prime_power(7u, 2));
    replay_arm<uint64_t>(ModCtx<uint64_t>(4294967311ull), ModCtx<uint64_t>(134217757ull),
                         ModCtx<uint64_t>::prime_power(134217757ull, 2), ModCtx<uint64_t>::prime_power(3ull, 30));
    std::cout << "  (R) guards + replay: checks so far " << g_checks << "  failing " << g_fail << std::endl;

    // (H) + (D)
    howell_arm<uint32_t>(2u, 2, 400, 0xE8A10004ULL);           // Z/4
    howell_arm<uint32_t>(2u, 3, 400, 0xE8A10008ULL);           // Z/8
    howell_arm<uint32_t>(3u, 2, 400, 0xE8A10009ULL);           // Z/9
    howell_arm<uint32_t>(5u, 2, 400, 0xE8A10019ULL);           // Z/25
    howell_arm<uint32_t>(5u, 3, 400, 0xE8A1007DULL);           // Z/125
    howell_arm<uint32_t>(691u, 2, 300, 0xE8A102B3ULL);         // Z/691^2 (u32)
    howell_arm<uint64_t>(134217757ull, 2, 200, 0xE8A1C0DEULL); // Z/134217757^2 (u64, Montgomery)
    howell_arm<uint64_t>(2ull, 40, 200, 0xE8A12040ULL);        // Z/2^40 (u64, U64_Wide)
    field_path_arm<uint32_t>(2u, 60, 0xF1E1D002ULL);
    field_path_arm<uint32_t>(5u, 60, 0xF1E1D005ULL);
    field_path_arm<uint32_t>(691u, 60, 0xF1E1D2B3ULL);
    field_path_arm<uint32_t>(134217757u, 60, 0xF1E1DC0DULL);
    field_path_arm<uint64_t>(134217757ull, 40, 0xF1E1DC0EULL);
    m1_arm();
    // (M) the M3 basis rule, directly (slot-8 WP-3r, review M-1)
    cyclic_arm<uint32_t>(2u, 2, 150, 0xE8C10004ULL);           // Z/4
    cyclic_arm<uint32_t>(5u, 2, 150, 0xE8C10019ULL);           // Z/25
    cyclic_arm<uint32_t>(5u, 3, 150, 0xE8C1007DULL);           // Z/125
    cyclic_arm<uint32_t>(691u, 2, 150, 0xE8C102B3ULL);         // Z/691^2
    cyclic_arm<uint64_t>(134217757ull, 2, 60, 0xE8C1C0DEULL);  // Z/134217757^2 (u64, Montgomery)
    cyclic_arm<uint64_t>(4294967291ull, 2, 60, 0xE8C1FFFBULL); // Z/4294967291^2 (u64, U64_Wide, >= 2^63)

    // (slot-8 WP-3, E8 item 3) the key, the BlockMat overload, the rho_n
    // repair, the pipeline over Z/ell^k at the pre-stated cells, the u64 arm
    // and the refusal closure (Hecke/HeckeReductionPipelineValidation.h).
    HeckeReductionPipelineArms(g_checks, g_fail);

    // (slot-8 WP-4, E8 item 4a) the reductions, the Bockstein, the bookkeeping
    // identity and higher_torsion_count at the five pre-stated cells, P3's
    // positive controls and the D6 arm (Hecke/HeckeReductionCoverageSlowValidation.h).
    HeckeReductionBulkArms(g_checks, g_fail);

    // (slot-9 WP-4, E8 item 4b) the boundary and the cone over Z/ell^k at the
    // E8 points but (60, 5) (in the `slow` sibling since slot-9 WP-4t) and the
    // two U64 regime cells, and the closure rows moved
    // by 4b (Hecke/HeckeReductionConeValidation.h).
    HeckeReductionConeArms(g_checks, g_fail);

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeReductionValidation: " << (g_fail == 0 ? "PASS" : "FAIL") << " (" << g_checks << " checks, "
              << g_fail << " failing, " << secs << " s)" << std::endl;
    return g_fail;
}
