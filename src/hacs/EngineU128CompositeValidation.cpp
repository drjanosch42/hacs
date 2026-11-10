// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "EngineU128CompositeValidation.h"

// hecke-extensions slot 11, WP-18a -- see the header for the arm list.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../ModArithmetic/BlockKernels.h"
#include "../ModArithmetic/BlockMat.h"
#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/DigitWord.h"
#include "../ModArithmetic/Kernels.h"
#include "../ModArithmetic/ModCtx.h"
#include "../ModArithmetic/RefLinAlg.h"

namespace {

using engine::DenseBuf;
using engine::DenseView;
using engine::ModCtx;
using engine::u128;
using engine::u256;
namespace elim = engine::elim;
namespace ref = engine::ref;
namespace kernels = engine::kernels;
namespace n128 = engine::ref::naive128;

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
    u128 below128(u128 m) {
        if (m <= 1) return 0;
        const u128 x = (static_cast<u128>(next()) << 64) | next();
        return x % m;
    }
};

// rand_below: for uint32_t / uint64_t EXACTLY HeckeReductionValidation's
// rng.below(p) (so its case generators replay bit for bit); u128 uses 128 bits.
template<class Elt> Elt rand_below(Splitmix& rng, Elt m) {
    if constexpr (std::is_same<Elt, u128>::value) return rng.below128(m);
    else return static_cast<Elt>(rng.below(static_cast<uint64_t>(m)));
}

std::string str128(u128 v) {
    if (v == 0) return "0";
    std::string s;
    while (v != 0) { s.push_back(static_cast<char>('0' + static_cast<int>(v % 10))); v /= 10; }
    std::reverse(s.begin(), s.end());
    return s;
}
template<class T> std::string str(T v) {
    if constexpr (std::is_same<T, u128>::value) return str128(v);
    else return std::to_string(v);
}

template<class Elt> const char* wname() {
    if constexpr (std::is_same<Elt, uint32_t>::value) return "u32";
    else if constexpr (std::is_same<Elt, uint64_t>::value) return "u64";
    else return "u128";
}
template<class Elt> std::string tag_of(const ModCtx<Elt>& c) {
    return "Z/" + str(c.ell) + "^" + std::to_string(c.k) + " (" + wname<Elt>() + ")";
}

u128 pow128(u128 ell, uint32_t k) { u128 x = 1; for (uint32_t i = 0; i < k; ++i) x *= ell; return x; }

// ---- matrix helpers ---------------------------------------------------------
template<class Elt>
DenseBuf<Elt> mul(Elt p, DenseView<const Elt> A, DenseView<const Elt> B) {
    DenseBuf<Elt> C(A.rows(), B.cols());
    ref::gemm<Elt>(p, A, B, engine::view(C), 0);
    return C;
}

// HeckeReductionValidation's generators, verbatim but for rand_below (which is
// its rng.below for u32 / u64).
template<class Elt>
Elt rand_elt(Splitmix& rng, const ModCtx<Elt>& c) {
    const uint32_t v = static_cast<uint32_t>(rng.below(c.k + 1));
    if (v == c.k) return 0;
    const Elt u = rand_below<Elt>(rng, c.p);
    return c.mul(c.ell_pow(v), u);
}
template<class Elt>
DenseBuf<Elt> rand_mat(Splitmix& rng, const ModCtx<Elt>& c, uint32_t m, uint32_t n) {
    DenseBuf<Elt> A(m, n);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) A.at(i, j) = rand_elt(rng, c);
    return A;
}
template<class Elt>
DenseBuf<Elt> rand_unimodular(Splitmix& rng, const ModCtx<Elt>& c, uint32_t m) {
    DenseBuf<Elt> U(m, m);
    for (uint32_t i = 0; i < m; ++i) U.at(i, i) = 1;
    if (m == 0) return U;
    for (uint32_t t = 0; t < 4 * m + 2; ++t) {
        const uint32_t i = static_cast<uint32_t>(rng.below(m)), j = static_cast<uint32_t>(rng.below(m));
        const uint32_t kind = static_cast<uint32_t>(rng.below(3));
        if (kind == 0) {
            Elt u;
            do { u = rand_below<Elt>(rng, c.p); } while (u == 0 || u % c.ell == 0);
            for (uint32_t q = 0; q < m; ++q) U.at(i, q) = c.mul(U.at(i, q), u);
        } else if (kind == 1 && i != j) {
            const Elt f = rand_below<Elt>(rng, c.p);
            for (uint32_t q = 0; q < m; ++q) U.at(i, q) = c.add(U.at(i, q), c.mul(f, U.at(j, q)));
        } else {
            for (uint32_t q = 0; q < m; ++q) std::swap(U.at(i, q), U.at(j, q));
        }
    }
    return U;
}

template<class E>
DenseBuf<u128> widen(const DenseBuf<E>& A) {
    DenseBuf<u128> W(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) W.at(i, j) = static_cast<u128>(A.at(i, j));
    return W;
}
template<class E>
bool same_buf(const DenseBuf<E>& a, const DenseBuf<u128>& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (static_cast<u128>(a.at(i, j)) != b.at(i, j)) return false;
    return true;
}
template<class E>
bool same_howell(const elim::HowellResult<E>& a, const elim::HowellResult<u128>& b) {
    return a.pivots == b.pivots && a.pivot_v == b.pivot_v && same_buf(a.H, b.H);
}
template<class E>
bool same_vv(const std::vector<std::vector<E>>& a, const std::vector<std::vector<u128>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); ++j) if (static_cast<u128>(a[i][j]) != b[i][j]) return false;
    }
    return true;
}
template<class E>
bool same_cyc(const elim::CyclicDecomposition<E>& a, const elim::CyclicDecomposition<u128>& b) {
    return a.exponents == b.exponents && a.position == b.position && same_buf(a.basis, b.basis) &&
           same_buf(a.Gamma, b.Gamma) && same_buf(a.U, b.U);
}
bool same_ref(const elim::HowellResult<u128>& a, const ref::RefHowell<u128>& b, uint32_t n) {
    if (a.pivots != b.pivots || a.pivot_v != b.pivot_v || a.rows() != b.rows.size()) return false;
    for (uint32_t i = 0; i < a.rows(); ++i)
        for (uint32_t j = 0; j < n; ++j) if (a.H.at(i, j) != b.rows[i][j]) return false;
    return true;
}

// Run f; capture either its value or its exception text.
template<class T, class F>
bool capture(F&& f, T& out, std::string& err) {
    try { out = f(); return true; }
    catch (const std::exception& e) { err = e.what(); return false; }
}

// ============================================================================
// (A) arithmetic
// ============================================================================
struct Mod { uint64_t ell; uint32_t k; };

// T = kk * D exactly (the caller guarantees the product is < 2^256).
u256 mul_u256_u64(const u256& D, uint64_t kk) {
    const u256 t1 = engine::detail::mul_wide128(D.lo, kk);
    return u256(static_cast<u128>(t1.hi + D.hi * kk), t1.lo, 0);
}

void arithmetic_arm(const Mod& md, uint32_t n_samples, uint64_t seed) {
    const uint64_t f0 = g_fail, c0 = g_checks;
    const ModCtx<u128> c = ModCtx<u128>::prime_power(md.ell, md.k);
    const u128 p = c.p;
    const std::string t = tag_of(c);
    check(p == pow128(md.ell, md.k), t + ": p == ell^k");
    check(!c.is_field() && c.regime == engine::Regime::U128_Montgomery && c.ell == md.ell && c.k == md.k,
          t + ": composite, U128_Montgomery, (ell, k) declared");
    check(std::string(engine::regime_name(c.regime)) == "U128_Montgomery", t + ": regime_name");
    check(c.p * (static_cast<u128>(0) - c.mont128_ninv) == 1, t + ": mont128_ninv == -p^{-1} mod 2^128");
    check(c.mont128_r1 == ((~static_cast<u128>(0)) % p + 1) % p, t + ": r1 == 2^128 mod p");
    check(c.mont128_r2 == n128::mul(c.mont128_r1, c.mont128_r1, p), t + ": r2 == r1^2 mod p (naive)");
    // ---- k_max tight ----
    const u256 D = engine::detail::mul_wide128(p - 1, p - 1);
    const bool saturated = c.k_max == std::numeric_limits<uint64_t>::max();
    {
        const u256 T = mul_u256_u64(D, c.k_max);
        check(T.hi < p, t + ": a block of k_max maximal products satisfies T < p*2^128");
        check(c.reduce(T) == static_cast<u128>(c.k_max) % p, t + ": reduce(k_max (p-1)^2) == k_max mod p");
        if (!saturated) {
            const u256 T1 = T + D;
            check(!(T1.hi < p), t + ": k_max + 1 maximal products violate T < p*2^128 (tight)");
        }
        if (!saturated && c.k_max <= 100000) {
            std::vector<u128> a(c.k_max + 1, p - 1), b(c.k_max + 1, p - 1);
            check(kernels::dot_delayed<u128>(c, a.data(), b.data(), c.k_max) == static_cast<u128>(c.k_max) % p,
                  t + ": dot_delayed over exactly k_max maximal products is exact");
            check(throws_logic([&] { (void)kernels::dot_delayed<u128>(c, a.data(), b.data(), c.k_max + 1); }),
                  t + ": dot_delayed refuses k_max + 1");
        }
    }
    // ---- primitives against naive128 ----
    Splitmix rng(seed);
    std::vector<u128> edge = {0, 1, 2, p - 1, p - 2, p / 2, p / 2 + 1};
    const u128 two64 = static_cast<u128>(1) << 64;
    for (u128 j : {static_cast<u128>(0), static_cast<u128>(1), static_cast<u128>(2), static_cast<u128>(977)}) {
        edge.push_back((two64 + j) % p);
        edge.push_back((two64 - 1 - j) % p);
    }
    edge.push_back((static_cast<u128>(1) << 63) % p);
    edge.push_back(((static_cast<u128>(1) << 127) - 1) % p);
    uint64_t bad_mul = 0, bad_pair = 0, bad_add = 0, bad_sub = 0, bad_neg = 0, bad_red = 0;
    auto one = [&](u128 a, u128 b) {
        const u128 want = n128::mul(a, b, p);
        if (c.mul(a, b) != want) ++bad_mul;
        if (c.mul_prepped(a, c.mul_prep(b)) != want) ++bad_pair;
        if (c.reduce(engine::detail::mul_wide128(a, b)) != want) ++bad_red;
        if (c.add(a, b) != n128::add(a, b, p)) ++bad_add;
        if (c.sub(a, b) != n128::sub(a, b, p)) ++bad_sub;
        if (c.neg(a) != n128::sub(0, a, p)) ++bad_neg;
    };
    for (u128 a : edge) for (u128 b : edge) one(a, b);
    for (uint32_t s = 0; s < n_samples; ++s) one(rng.below128(p), rng.below128(p));
    check(bad_mul == 0, t + ": mul == naive mulmod (" + std::to_string(n_samples) + " samples + boundary)");
    check(bad_pair == 0, t + ": mul_prepped(x, mul_prep(w)) == naive mulmod");
    check(bad_red == 0, t + ": reduce(a*b) == naive mulmod");
    check(bad_add == 0 && bad_sub == 0 && bad_neg == 0, t + ": add / sub / neg == naive");
    // below 2^64: also against ModCtx<uint64_t> (the shipped route)
    if (p <= std::numeric_limits<uint64_t>::max()) {
        const ModCtx<uint64_t> c64 = ModCtx<uint64_t>::prime_power(md.ell, md.k);
        uint64_t bad64 = 0;
        Splitmix r2(seed ^ 0x64ULL);
        for (uint32_t s = 0; s < n_samples / 4; ++s) {
            const uint64_t a = r2.below(static_cast<uint64_t>(p)), b = r2.below(static_cast<uint64_t>(p));
            if (c.mul(a, b) != c64.mul(a, b) || c.mul_prepped(a, c.mul_prep(b)) != c64.mul_prepped(a, c64.mul_prep(b)))
                ++bad64;
            if ((a % md.ell) != 0 && c.inv(a) != c64.inv(a)) ++bad64;
        }
        check(bad64 == 0, t + ": u128 == ModCtx<uint64_t> on mul, the pair and inv (below 2^64)");
    }
    // ---- inv: units, every valuation class ----
    uint64_t bad_inv = 0;
    for (uint32_t s = 0; s < 20000; ++s) {
        u128 u;
        do { u = rng.below128(p); } while (u % md.ell == 0);
        const u128 x = c.inv(u);
        if (x != n128::inv(u, p) || c.mul(u, x) != 1) ++bad_inv;
    }
    check(bad_inv == 0, t + ": inv(u) == naive extended Euclid, u * inv(u) == 1 (20000 units)");
    check(throws_runtime_msg([&] { (void)c.inv(0); }, "Modular inverse does not exist."), t + ": inv(0) throws");
    bool all_classes = true;
    for (uint32_t v = 1; v < md.k; ++v) {
        u128 u;
        do { u = rng.below128(p); } while (u % md.ell == 0);
        const u128 x = c.mul(c.ell_pow(v), u);
        if (c.valuation(x) != v) all_classes = false;
        if (!throws_runtime_msg([&] { (void)c.inv(x); }, "Modular inverse does not exist.")) all_classes = false;
        if (!throws_runtime_msg([&] { (void)n128::inv(x, p); }, "Modular inverse does not exist.")) all_classes = false;
    }
    check(all_classes, t + ": inv throws on every valuation class v = 1 .. k-1 (engine and naive)");
    // ---- Sec. 13.4 members ----
    {
        bool ok = c.valuation(0) == md.k && c.unit_part(0) == 1 && c.ell_pow(0) == 1 && c.ell_pow(md.k) == p;
        for (uint32_t v = 0; v < md.k && ok; ++v) {
            const u128 x = c.mul(c.ell_pow(v), 7 % md.ell == 0 ? 8 : 7);
            ok = c.valuation(x) == v && c.unit_part(x) * c.ell_pow(v) == x &&
                 c.divide_exact(x, c.ell_pow(v)) == x / c.ell_pow(v);
        }
        check(ok, t + ": valuation / unit_part / ell_pow / divide_exact");
        check(throws_logic([&] { (void)c.divide_exact(c.ell_pow(1), c.ell_pow(2)); }), t + ": divide_exact refuses ell^2 !| ell");
    }
    // ---- kernels against ref::gemm on naive128 ----
    {
        const uint32_t kdim = (c.k_max < 1000) ? 9 : 300;   // crosses k_max (small) or KC_MAX = 256
        DenseBuf<u128> A(7, kdim), B(kdim, 5), Bt(5, kdim);
        for (uint32_t i = 0; i < 7; ++i) for (uint32_t j = 0; j < kdim; ++j) A.at(i, j) = rng.below128(p);
        for (uint32_t i = 0; i < kdim; ++i) for (uint32_t j = 0; j < 5; ++j) { B.at(i, j) = rng.below128(p); Bt.at(j, i) = B.at(i, j); }
        bool ok = true;
        for (int op = 0; op < 3; ++op) {
            DenseBuf<u128> C1(7, 5), C2(7, 5), C3(7, 5);
            for (uint32_t i = 0; i < 7; ++i) for (uint32_t j = 0; j < 5; ++j) C1.at(i, j) = C2.at(i, j) = C3.at(i, j) = rng.below128(p);
            const kernels::GemmOp gop = op == 0 ? kernels::GemmOp::Set : (op == 1 ? kernels::GemmOp::Add : kernels::GemmOp::Sub);
            kernels::gemm<u128>(c, engine::cview(A), engine::cview(B), engine::view(C1), gop);
            kernels::gemm_nt<u128>(c, engine::cview(A), engine::cview(Bt), engine::view(C3), gop);
            ref::gemm<u128>(p, engine::cview(A), engine::cview(B), engine::view(C2), op);
            for (uint32_t i = 0; i < 7; ++i) for (uint32_t j = 0; j < 5; ++j)
                if (C1.at(i, j) != C2.at(i, j) || C3.at(i, j) != C2.at(i, j)) ok = false;
        }
        check(ok, t + ": kernels::gemm / gemm_nt (k = " + std::to_string(kdim) + ", Set/Add/Sub) == ref::gemm (naive128)");
        DenseBuf<u128> x(kdim, 1), y1(7, 1), y2(7, 1);
        for (uint32_t i = 0; i < kdim; ++i) x.at(i, 0) = rng.below128(p);
        kernels::gemv<u128>(c, engine::cview(A), engine::cview(x), engine::view(y1));
        ref::gemv<u128>(p, engine::cview(A), engine::cview(x), engine::view(y2), 0);
        bool okv = true;
        for (uint32_t i = 0; i < 7; ++i) if (y1.at(i, 0) != y2.at(i, 0)) okv = false;
        check(okv, t + ": kernels::gemv == ref::gemv");
        const size_t kk = std::min<uint64_t>(c.k_max, 64);
        std::vector<u128> a(kk), b(kk);
        u128 want = 0;
        for (size_t i = 0; i < kk; ++i) { a[i] = rng.below128(p); b[i] = rng.below128(p); want = n128::add(want, n128::mul(a[i], b[i], p), p); }
        check(kernels::dot_delayed<u128>(c, a.data(), b.data(), kk) == want, t + ": dot_delayed == naive");
        const u128 alpha = rng.below128(p);
        DenseBuf<u128> Y(7, kdim), Y2(7, kdim), S(7, kdim), Y0(7, kdim);
        bool oka = true, oks = true;
        for (uint32_t i = 0; i < 7; ++i) for (uint32_t j = 0; j < kdim; ++j) Y0.at(i, j) = Y.at(i, j) = Y2.at(i, j) = rng.below128(p);
        for (uint32_t i = 0; i < 7; ++i) for (uint32_t j = 0; j < kdim; ++j) S.at(i, j) = A.at(i, j);
        kernels::axpy<u128>(c, alpha, engine::cview(A), engine::view(Y), false);
        kernels::axpy<u128>(c, alpha, engine::cview(A), engine::view(Y2), true);
        kernels::scal<u128>(c, alpha, engine::view(S));
        for (uint32_t i = 0; i < 7; ++i) for (uint32_t j = 0; j < kdim; ++j) {
            const u128 pr = n128::mul(alpha, A.at(i, j), p);
            if (Y.at(i, j) != n128::add(Y0.at(i, j), pr, p) || Y2.at(i, j) != n128::sub(Y0.at(i, j), pr, p)) oka = false;
            if (S.at(i, j) != pr) oks = false;
        }
        check(oka, t + ": axpy (add, subtract) == naive");
        check(oks, t + ": scal == naive");
    }
    std::cout << "  (A) " << t << "  k_max " << c.k_max << (saturated ? " (saturated)" : "") << "  samples "
              << n_samples << "  checks " << (g_checks - c0) << "  failing " << (g_fail - f0) << std::endl;
}

void construction_refusals() {
    check(throws_logic([] { (void)ModCtx<u128>::prime_power(2, 70); }), "prime_power<u128>(2, 70) refused (ell = 2: no Montgomery form)");
    check(throws_logic([] { (void)ModCtx<u128>::prime_power(5, 1); }), "prime_power<u128>(5, 1) refused (composite-only)");
    check(throws_logic([] { (void)ModCtx<u128>::prime_power(3, 81); }), "prime_power<u128>(3, 81) refused (3^81 >= 2^127)");
    check(throws_logic([] { (void)ModCtx<u128>::prime_power(691, 14); }), "prime_power<u128>(691, 14) refused (691^14 >= 2^127)");
    check(throws_logic([] { (void)ModCtx<u128>::prime_power(static_cast<u128>(1) << 64 | 1, 2); }),
          "prime_power<u128>(ell >= 2^64, 2) refused");
    check(!ModCtx<u128>::prime_power(691, 13).is_field(), "prime_power<u128>(691, 13) (~2^122.5) admitted");
    check(engine::is_composite_only<u128>::value && !engine::is_composite_only<uint64_t>::value &&
          !engine::is_composite_only<uint32_t>::value, "is_composite_only: u128 alone");
    std::cout << "  (A) sizeof(ModCtx<u128>) = " << sizeof(ModCtx<u128>) << " (trivially copyable: static_assert)" << std::endl;
}

// ============================================================================
// (E) the oracle rule below 2^64: u128 == native, entrywise
// ============================================================================
struct Shape { uint32_t m, n; };

// Every Sec. 13.6 entry point on one matrix A (m x n) in the native width E
// and in u128.  `rng` supplies the right-hand sides and the relation columns.
template<class E>
void compare_entry_points(const ModCtx<E>& c, const ModCtx<u128>& cw, const DenseBuf<E>& A, Splitmix& rng,
                          const std::string& cs, uint32_t& n_cmp) {
    const DenseBuf<u128> Aw = widen(A);
    const uint32_t m = A.rows, n = A.cols;
    check(same_howell(elim::howell<E>(c, engine::cview(A)), elim::howell<u128>(cw, engine::cview(Aw))), cs + ": howell");
    check(same_howell(elim::kernel_module<E>(c, engine::cview(A)), elim::kernel_module<u128>(cw, engine::cview(Aw))),
          cs + ": kernel_module");
    check(same_howell(elim::image_module<E>(c, engine::cview(A)), elim::image_module<u128>(cw, engine::cview(Aw))),
          cs + ": image_module");
    check(elim::length<E>(c, engine::cview(A)) == elim::length<u128>(cw, engine::cview(Aw)), cs + ": length");
    check(elim::invariant_factors<E>(c, engine::cview(A)) == elim::invariant_factors<u128>(cw, engine::cview(Aw)),
          cs + ": invariant_factors(A)");
    // solves: an image right-hand side and a random one (both outcomes compared, the refusal text included)
    const DenseBuf<E> Z = rand_mat(rng, c, n, 2);
    const DenseBuf<E> Y1 = mul<E>(c.p, engine::cview(A), engine::cview(Z));
    const DenseBuf<E> Y2 = rand_mat(rng, c, m, 2);
    for (const DenseBuf<E>* Y : {&Y1, &Y2}) {
        std::vector<std::vector<E>> x;
        std::vector<std::vector<u128>> xw;
        std::string e1, e2;
        const bool ok1 = capture(
            [&] { return elim::solve_multi_rhs_module<E>(c, engine::cview(A), engine::cview(*Y)); }, x, e1);
        const DenseBuf<u128> Yw = widen(*Y);
        const bool ok2 = capture(
            [&] { return elim::solve_multi_rhs_module<u128>(cw, engine::cview(Aw), engine::cview(Yw)); }, xw, e2);
        check(ok1 == ok2 && (ok1 ? same_vv(x, xw) : e1 == e2), cs + ": solve_multi_rhs_module (value or refusal text)");
    }
    // cyclic decomposition of colspan A modulo a relation block Rl
    const DenseBuf<E> Rl = (rng.below(2) == 0) ? mul<E>(c.p, engine::cview(A), engine::cview(rand_mat(rng, c, n, 2)))
                                              : rand_mat(rng, c, m, 1 + static_cast<uint32_t>(rng.below(2)));
    const DenseBuf<u128> Rlw = widen(Rl);
    check(elim::invariant_factors<E>(c, engine::cview(A), engine::cview(Rl)) ==
              elim::invariant_factors<u128>(cw, engine::cview(Aw), engine::cview(Rlw)),
          cs + ": invariant_factors(G, Rl)");
    const auto d = elim::cyclic_decomposition<E>(c, engine::cview(A), engine::cview(Rl));
    const auto dw = elim::cyclic_decomposition<u128>(cw, engine::cview(Aw), engine::cview(Rlw));
    check(same_cyc(d, dw), cs + ": cyclic_decomposition (exponents, basis, Gamma, U, position)");
    check(same_vv(elim::cyclic_coordinates<E>(c, d, engine::cview(A)),
                  elim::cyclic_coordinates<u128>(cw, dw, engine::cview(Aw))),
          cs + ": cyclic_coordinates");
    // the naive oracle on the u128 side as well
    check(same_ref(elim::howell<u128>(cw, engine::cview(Aw)), ref::howell<u128>(cw.ell, cw.k, engine::cview(Aw)), n),
          cs + ": howell<u128> == ref::howell<u128>");
    ++n_cmp;
}

// HeckeReductionValidation's howell_arm case list (seed, shapes, order), replayed.
template<class E>
void howell_cases_arm(E ell, uint32_t k, uint32_t n_random, uint64_t seed) {
    const uint64_t f0 = g_fail, c0 = g_checks;
    const ModCtx<E> c = ModCtx<E>::prime_power(ell, k);
    const ModCtx<u128> cw = ModCtx<u128>::prime_power(ell, k);
    const std::string t = tag_of(c) + " vs u128";
    Splitmix rng(seed);
    const Shape shapes[] = {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {3, 3}, {4, 2}, {2, 5}, {5, 5}, {6, 4}, {4, 7}, {8, 8}, {9, 5}, {3, 10}, {12, 12}, {16, 10}, {10, 16}};
    std::vector<DenseBuf<E>> cases;
    cases.push_back(DenseBuf<E>(4, 5));
    { DenseBuf<E> I(5, 5); for (uint32_t i = 0; i < 5; ++i) I.at(i, i) = 1; cases.push_back(I); }
    { DenseBuf<E> Em = rand_mat(rng, c, 5, 6);
      for (uint32_t i = 0; i < 5; ++i) for (uint32_t j = 0; j < 6; ++j) Em.at(i, j) = c.mul(Em.at(i, j), c.ell_pow(1));
      cases.push_back(Em); }
    for (uint32_t v = 1; v < k; ++v) {
        DenseBuf<E> C1(1, 2); C1.at(0, 0) = c.ell_pow(v); C1.at(0, 1) = 1; cases.push_back(C1);
        DenseBuf<E> C2(2, 3); C2.at(0, 0) = c.ell_pow(v); C2.at(0, 1) = 1; C2.at(1, 1) = c.ell_pow(k - v); C2.at(1, 2) = c.ell_pow(v);
        cases.push_back(C2);
    }
    {
        const DenseBuf<E> L = rand_mat(rng, c, 6, 2), Rr = rand_mat(rng, c, 2, 7);
        cases.push_back(mul<E>(c.p, engine::cview(L), engine::cview(Rr)));
    }
    cases.push_back(rand_mat(rng, c, 12, 3));
    cases.push_back(rand_mat(rng, c, 3, 12));
    for (uint32_t r = 0; r < n_random; ++r) {
        const Shape s = shapes[rng.below(sizeof(shapes) / sizeof(shapes[0]))];
        cases.push_back(rand_mat(rng, c, s.m, s.n));
    }
    Splitmix aux(seed ^ 0x18A0ULL);
    uint32_t n_cmp = 0;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const std::string cs = t + " (H) case " + std::to_string(ci) + " (" + std::to_string(cases[ci].rows) + "x" +
                               std::to_string(cases[ci].cols) + ")";
        try { compare_entry_points<E>(c, cw, cases[ci], aux, cs, n_cmp); }
        catch (const std::exception& ex) { check(false, cs + ": unexpected exception: " + ex.what()); }
    }
    std::cout << "  (E) " << t << "  (H) cases seed 0x" << std::hex << seed << std::dec << ": " << cases.size()
              << " (compared " << n_cmp << ")  checks " << (g_checks - c0) << "  failing " << (g_fail - f0) << std::endl;
}

// HeckeReductionValidation's cyclic_arm case list (seed, order), replayed.
template<class E>
void cyclic_cases_arm(E ell, uint32_t k, uint32_t n_random, uint64_t seed) {
    const uint64_t f0 = g_fail, c0 = g_checks;
    const ModCtx<E> c = ModCtx<E>::prime_power(ell, k);
    const ModCtx<u128> cw = ModCtx<u128>::prime_power(ell, k);
    const std::string t = tag_of(c) + " vs u128";
    Splitmix rng(seed);
    struct Case { DenseBuf<E> G, Rl; };
    std::vector<Case> cases;
    for (uint32_t r = 0; r < 24; ++r) {
        const uint32_t m = 1 + static_cast<uint32_t>(rng.below(7));
        const DenseBuf<E> U = rand_unimodular(rng, c, m), V = rand_unimodular(rng, c, m);
        DenseBuf<E> D(m, m);
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t vv = static_cast<uint32_t>(rng.below(c.k + 1));
            D.at(i, i) = (vv == c.k) ? E(0) : c.ell_pow(vv);
        }
        const DenseBuf<E> UD = mul<E>(c.p, engine::cview(U), engine::cview(D));
        cases.push_back({U, mul<E>(c.p, engine::cview(UD), engine::cview(V))});
    }
    const Shape shapes[] = {{1, 1}, {2, 1}, {2, 2}, {3, 3}, {4, 2}, {5, 5}, {6, 4}, {4, 7}, {8, 6}, {9, 5}, {10, 10}};
    for (uint32_t r = 0; r < n_random; ++r) {
        const Shape s = shapes[rng.below(sizeof(shapes) / sizeof(shapes[0]))];
        const DenseBuf<E> G = rand_mat(rng, c, s.m, s.n);
        const uint32_t kind = static_cast<uint32_t>(rng.below(4)), nr = static_cast<uint32_t>(rng.below(5));
        DenseBuf<E> Rl(s.m, 0);
        if (kind == 1 || kind == 2) {
            const DenseBuf<E> Z = rand_mat(rng, c, s.n, nr);
            Rl = mul<E>(c.p, engine::cview(G), engine::cview(Z));
            if (kind == 2)
                for (uint32_t i = 0; i < Rl.rows; ++i)
                    for (uint32_t j = 0; j < Rl.cols; ++j)
                        Rl.at(i, j) = c.mul(Rl.at(i, j), c.ell_pow(1 + (i + j) % (c.k - 1)));
        } else if (kind == 3) {
            Rl = rand_mat(rng, c, s.m, nr);
        }
        cases.push_back({G, Rl});
    }
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const std::string cs = t + " (M) case " + std::to_string(ci);
        try {
            const DenseBuf<u128> Gw = widen(cases[ci].G), Rlw = widen(cases[ci].Rl);
            const auto d = elim::cyclic_decomposition<E>(c, engine::cview(cases[ci].G), engine::cview(cases[ci].Rl));
            const auto dw = elim::cyclic_decomposition<u128>(cw, engine::cview(Gw), engine::cview(Rlw));
            check(same_cyc(d, dw), cs + ": cyclic_decomposition (exponents, basis, Gamma, U, position)");
            check(elim::invariant_factors<E>(c, engine::cview(cases[ci].G), engine::cview(cases[ci].Rl)) ==
                      elim::invariant_factors<u128>(cw, engine::cview(Gw), engine::cview(Rlw)),
                  cs + ": invariant_factors(G, Rl)");
            check(same_vv(elim::cyclic_coordinates<E>(c, d, engine::cview(d.basis)),
                          elim::cyclic_coordinates<u128>(cw, dw, engine::cview(dw.basis))),
                  cs + ": cyclic_coordinates(dec, basis)");
            check(elim::invariant_factors<u128>(cw, engine::cview(Gw), engine::cview(Rlw)) ==
                      ref::invariant_factors<u128>(cw.ell, cw.k, engine::cview(Gw), engine::cview(Rlw)),
                  cs + ": invariant_factors<u128> == ref::invariant_factors<u128>");
        } catch (const std::exception& ex) {
            check(false, cs + ": unexpected exception: " + ex.what());
        }
    }
    std::cout << "  (E) " << t << "  (M) cases seed 0x" << std::hex << seed << std::dec << ": " << cases.size()
              << "  checks " << (g_checks - c0) << "  failing " << (g_fail - f0) << std::endl;
}

// Random cases at a u64 modulus not in HeckeReductionValidation.
void random_u64_arm(uint64_t ell, uint32_t k, uint32_t n_cases, uint64_t seed) {
    const uint64_t f0 = g_fail, c0 = g_checks;
    const ModCtx<uint64_t> c = ModCtx<uint64_t>::prime_power(ell, k);
    const ModCtx<u128> cw = ModCtx<u128>::prime_power(ell, k);
    const std::string t = tag_of(c) + " [" + engine::regime_name(c.regime) + "] vs u128";
    Splitmix rng(seed);
    const Shape shapes[] = {{1, 3}, {3, 3}, {4, 6}, {6, 4}, {7, 7}, {5, 9}, {10, 8}, {12, 12}};
    uint32_t n_cmp = 0;
    for (uint32_t r = 0; r < n_cases; ++r) {
        const Shape s = shapes[rng.below(sizeof(shapes) / sizeof(shapes[0]))];
        const DenseBuf<uint64_t> A = rand_mat(rng, c, s.m, s.n);
        const std::string cs = t + " random case " + std::to_string(r);
        try { compare_entry_points<uint64_t>(c, cw, A, rng, cs, n_cmp); }
        catch (const std::exception& ex) { check(false, cs + ": unexpected exception: " + ex.what()); }
    }
    std::cout << "  (E) " << t << "  random cases " << n_cases << "  checks " << (g_checks - c0) << "  failing "
              << (g_fail - f0) << std::endl;
}

// ============================================================================
// (B) beyond 2^64: the naive oracle and planted elementary divisors
// ============================================================================
void beyond_arm(uint64_t ell, uint32_t k, uint32_t n_random, uint64_t seed) {
    const uint64_t f0 = g_fail, c0 = g_checks;
    const ModCtx<u128> c = ModCtx<u128>::prime_power(ell, k);
    const u128 p = c.p;
    const std::string t = tag_of(c);
    Splitmix rng(seed);
    const Shape shapes[] = {{1, 2}, {2, 2}, {3, 3}, {4, 2}, {2, 5}, {5, 5}, {6, 4}, {4, 7}, {8, 8}, {9, 5}, {10, 12}};
    uint32_t n_ok = 0;
    for (uint32_t r = 0; r < n_random; ++r) {
        const Shape s = shapes[rng.below(sizeof(shapes) / sizeof(shapes[0]))];
        DenseBuf<u128> A = rand_mat(rng, c, s.m, s.n);
        if (r % 5 == 0 && s.m >= 2) {   // rank-deficient: (m x 1)(1 x n)
            const DenseBuf<u128> L = rand_mat(rng, c, s.m, 1), Rr = rand_mat(rng, c, 1, s.n);
            A = mul<u128>(p, engine::cview(L), engine::cview(Rr));
        }
        const std::string cs = t + " case " + std::to_string(r) + " (" + std::to_string(s.m) + "x" + std::to_string(s.n) + ")";
        try {
            const auto H = elim::howell<u128>(c, engine::cview(A));
            check(same_ref(H, ref::howell<u128>(ell, k, engine::cview(A)), s.n), cs + ": howell == ref::howell (naive128)");
            const auto K = elim::kernel_module<u128>(c, engine::cview(A));
            // A * K^T == 0
            DenseBuf<u128> KT(s.n, K.rows());
            for (uint32_t i = 0; i < K.rows(); ++i) for (uint32_t j = 0; j < s.n; ++j) KT.at(j, i) = K.H.at(i, j);
            const DenseBuf<u128> AK = mul<u128>(p, engine::cview(A), engine::cview(KT));
            bool zero = true;
            for (uint32_t i = 0; i < AK.rows; ++i) for (uint32_t j = 0; j < AK.cols; ++j) if (AK.at(i, j) != 0) zero = false;
            check(zero, cs + ": A * kernel_module(A)^T == 0 (ref::gemm)");
            const auto Im = elim::image_module<u128>(c, engine::cview(A));
            check(elim::length<u128>(c, K) + elim::length<u128>(c, Im) == s.n * k, cs + ": length(ker) + length(im) == n k");
            // solve B x = y for y in the image
            const DenseBuf<u128> Z = rand_mat(rng, c, s.n, 2);
            const DenseBuf<u128> Y = mul<u128>(p, engine::cview(A), engine::cview(Z));
            const auto X = elim::solve_multi_rhs_module<u128>(c, engine::cview(A), engine::cview(Y));
            bool solved = true;
            for (uint32_t tt = 0; tt < 2; ++tt) {
                DenseBuf<u128> x(s.n, 1), y(s.m, 1);
                for (uint32_t j = 0; j < s.n; ++j) x.at(j, 0) = X[tt][j];
                ref::gemm<u128>(p, engine::cview(A), engine::cview(x), engine::view(y), 0);
                for (uint32_t i = 0; i < s.m; ++i) if (y.at(i, 0) != Y.at(i, tt)) solved = false;
            }
            check(solved, cs + ": A * solve_multi_rhs_module(A, Y) == Y (ref::gemm)");
            const DenseBuf<u128> Rl = rand_mat(rng, c, s.m, 1 + static_cast<uint32_t>(rng.below(3)));
            check(elim::invariant_factors<u128>(c, engine::cview(A), engine::cview(Rl)) ==
                      ref::invariant_factors<u128>(ell, k, engine::cview(A), engine::cview(Rl)),
                  cs + ": invariant_factors(G, Rl) == ref::invariant_factors (naive128)");
            ++n_ok;
        } catch (const std::exception& ex) {
            check(false, cs + ": unexpected exception: " + ex.what());
        }
    }
    // planted: G = U, Rl = U diag(ell^{v_i}) V  =>  M = (+) Z/ell^{v_i}, v_i >= 1
    uint32_t n_planted = 0, n_mixed = 0;
    for (uint32_t r = 0; r < 16; ++r) {
        const uint32_t m = 1 + static_cast<uint32_t>(rng.below(6));
        const DenseBuf<u128> U = rand_unimodular(rng, c, m), V = rand_unimodular(rng, c, m);
        DenseBuf<u128> D(m, m);
        std::vector<uint32_t> known;
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t vv = static_cast<uint32_t>(rng.below(k + 1));
            D.at(i, i) = (vv == k) ? u128(0) : c.ell_pow(vv);
            if (vv >= 1) known.push_back(vv);
        }
        std::sort(known.begin(), known.end(), [](uint32_t a, uint32_t b) { return a > b; });
        const DenseBuf<u128> Rl = mul<u128>(p, engine::cview(mul<u128>(p, engine::cview(U), engine::cview(D))), engine::cview(V));
        const std::string cs = t + " planted case " + std::to_string(r) + " (m = " + std::to_string(m) + ")";
        try {
            const auto d = elim::cyclic_decomposition<u128>(c, engine::cview(U), engine::cview(Rl));
            check(d.exponents == known, cs + ": cyclic_decomposition exponents == the planted divisors");
            check(elim::invariant_factors<u128>(c, engine::cview(U), engine::cview(Rl)) == known,
                  cs + ": invariant_factors == the planted divisors");
            check(ref::invariant_factors<u128>(ell, k, engine::cview(U), engine::cview(Rl)) == known,
                  cs + ": ref::invariant_factors (naive128) == the planted divisors");
            // Smith of Rl itself: colspan Rl = U diag V, so R^m / colspan Rl has the same exponents
            const DenseBuf<u128> I = [&] { DenseBuf<u128> X(m, m); for (uint32_t i = 0; i < m; ++i) X.at(i, i) = 1; return X; }();
            check(elim::invariant_factors<u128>(c, engine::cview(I), engine::cview(Rl)) == known,
                  cs + ": R^m / colspan(U D V) has the planted exponents");
            const auto I2 = elim::cyclic_coordinates<u128>(c, d, engine::cview(d.basis));
            bool ident = true;
            for (uint32_t a = 0; a < I2.size(); ++a)
                for (uint32_t b = 0; b < I2[a].size(); ++b) if (I2[a][b] != (a == b ? 1u : 0u)) ident = false;
            check(ident, cs + ": cyclic_coordinates(dec, basis) == I");
            ++n_planted;
            if (known.size() >= 2 && known.front() != known.back()) ++n_mixed;
        } catch (const std::exception& ex) {
            check(false, cs + ": unexpected exception: " + ex.what());
        }
    }
    check(n_mixed >= 1, t + ": at least one planted case with mixed exponents");
    std::cout << "  (B) " << t << "  random cases " << n_ok << "/" << n_random << ", planted " << n_planted
              << " (mixed " << n_mixed << ")  checks " << (g_checks - c0) << "  failing " << (g_fail - f0) << std::endl;
}

// ============================================================================
// (K) the block model's composite surface
// ============================================================================
void block_arm(uint64_t ell, uint32_t k, uint64_t seed) {
    const uint64_t f0 = g_fail, c0 = g_checks;
    using BM = engine::BlockMat<u128>;
    const ModCtx<u128> c = ModCtx<u128>::prime_power(ell, k);
    const u128 p = c.p;
    const std::string t = tag_of(c);
    Splitmix rng(seed);
    auto pool = std::make_shared<engine::BlockPool<u128>>();
    DenseBuf<u128> A(6, 7), B(7, 5), A2(6, 7);
    for (uint32_t i = 0; i < 6; ++i) for (uint32_t j = 0; j < 7; ++j) { A.at(i, j) = rng.below128(p); A2.at(i, j) = rng.below128(p); }
    for (uint32_t i = 0; i < 7; ++i) for (uint32_t j = 0; j < 5; ++j) B.at(i, j) = rng.below128(p);
    // a scalar block and a zero block in A (the Scalar / Zero kinds)
    for (uint32_t i = 0; i < 3; ++i) for (uint32_t j = 0; j < 3; ++j) A.at(3 + i, 4 + j) = (i == j) ? (p - 5) : 0;
    for (uint32_t i = 0; i < 3; ++i) for (uint32_t j = 0; j < 4; ++j) A.at(i, j) = 0;
    const DenseBuf<u128> Ad = A, Bd = B, A2d = A2;
    const BM bA = BM::from_dense(pool, DenseBuf<u128>(A), {3, 3}, {4, 3}, c);
    const BM bB = BM::from_dense(pool, DenseBuf<u128>(B), {4, 3}, {2, 3}, c);
    const BM bA2 = BM::from_dense(pool, DenseBuf<u128>(A2), {3, 3}, {4, 3}, c);
    check(!bA.ctx().is_field() && bA.ctx().p == p, t + ": BlockMat::from_dense(ctx) carries the composite context");
    const BM bC = engine::blk::gemm<u128>(bA, bB);
    check(!bC.ctx().is_field() && bC.ctx().p == p && bC.ctx().ell == ell, t + ": blk::gemm result on A.ctx()");
    const DenseBuf<u128> Cref = mul<u128>(p, engine::cview(Ad), engine::cview(Bd));
    DenseBuf<u128> Ck(6, 5);
    kernels::gemm<u128>(c, engine::cview(Ad), engine::cview(Bd), engine::view(Ck));
    const DenseBuf<u128> Cb = bC.to_dense();
    bool ok = true, ok2 = true;
    for (uint32_t i = 0; i < 6; ++i) for (uint32_t j = 0; j < 5; ++j) { if (Cb.at(i, j) != Cref.at(i, j)) ok = false; if (Ck.at(i, j) != Cref.at(i, j)) ok2 = false; }
    check(ok, t + ": blk::gemm == ref::gemm (naive128)");
    check(ok2, t + ": kernels::gemm == ref::gemm");
    DenseBuf<u128> Yd(6, 5);
    engine::blk::gemm_dense<u128>(bA, engine::cview(Bd), engine::view(Yd));
    bool ok3 = true;
    for (uint32_t i = 0; i < 6; ++i) for (uint32_t j = 0; j < 5; ++j) if (Yd.at(i, j) != Cref.at(i, j)) ok3 = false;
    check(ok3, t + ": blk::gemm_dense == ref::gemm");
    const BM bS = engine::blk::add<u128>(bA, bA2, true);
    const DenseBuf<u128> Sd = bS.to_dense();
    bool ok4 = true;
    for (uint32_t i = 0; i < 6; ++i) for (uint32_t j = 0; j < 7; ++j) if (Sd.at(i, j) != n128::sub(Ad.at(i, j), A2d.at(i, j), p)) ok4 = false;
    check(ok4 && !bS.ctx().is_field(), t + ": blk::add (subtract) == naive, composite context kept");
    const u128 alpha = rng.below128(p);
    const DenseBuf<u128> Sc = engine::blk::scale<u128>(bA, alpha).to_dense();
    bool ok5 = true;
    for (uint32_t i = 0; i < 6; ++i) for (uint32_t j = 0; j < 7; ++j) if (Sc.at(i, j) != n128::mul(alpha, Ad.at(i, j), p)) ok5 = false;
    check(ok5, t + ": blk::scale == naive");
    check(engine::blk::equal<u128>(bA, bA) && !engine::blk::equal<u128>(bA, bA2), t + ": blk::equal");
    // BlockPool: the content hash reads both halves; equal content deduplicates
    {
        engine::BlockPool<u128> pl;
        const u128 lo = 12345, hi_only = lo + (static_cast<u128>(1) << 64);   // differ in the high limb only
        check(hi_only < p, t + ": the high-limb probe value is a residue");
        DenseBuf<u128> x(1, 2), y(1, 2), z(1, 2);
        x.at(0, 0) = lo; x.at(0, 1) = 7;
        y.at(0, 0) = hi_only; y.at(0, 1) = 7;
        z.at(0, 0) = lo; z.at(0, 1) = 7;
        check(engine::BlockPool<u128>::content_hash(engine::cview(x)) != engine::BlockPool<u128>::content_hash(engine::cview(y)),
              t + ": BlockPool content hash distinguishes entries differing only in the high 64 bits");
        const auto ix = pl.intern(DenseBuf<u128>(x));
        const auto iy = pl.intern(DenseBuf<u128>(y));
        const auto iz = pl.intern(DenseBuf<u128>(z));
        check(ix != iy && ix == iz, t + ": BlockPool dedups equal content, keeps the high-limb twin distinct");
    }
    std::cout << "  (K) " << t << "  checks " << (g_checks - c0) << "  failing " << (g_fail - f0) << std::endl;
}

// ============================================================================
// (W) DigitWord<W>
// ============================================================================
template<class W>
void digitword_split_arm(uint64_t ell, uint64_t seed) {
    using DW = engine::DigitWord<W>;
    const uint64_t f0 = g_fail, c0 = g_checks;
    const uint32_t m = DW::digit_exponent(ell);
    const std::string t = std::string("DigitWord<") + wname<W>() + "> ell=" + std::to_string(ell);
    if (m < 2) {
        check(!DW::available(ell), t + ": unavailable width reports m = 0");
        std::cout << "  (W) " << t << "  m = 0 (width unavailable)" << std::endl;
        return;
    }
    const W M = DW::pow(ell, m);
    const u128 lim = static_cast<u128>(1) << DW::digit_bits;
    check(static_cast<u128>(M) < lim && static_cast<u128>(M) > (lim - 1) / ell,   // M * ell >= lim, overflow-free
          t + ": m maximal with ell^m < 2^{w-1}");
    const auto ctx = DW::ctx(ell, m);
    check(!ctx.is_field() && static_cast<u128>(ctx.p) == static_cast<u128>(M) && ctx.k == m, t + ": ctx(ell, m) == Z/ell^m");
    Splitmix rng(seed);
    uint64_t bad = 0;
    for (uint32_t s = 0; s < 20000; ++s) {
        const W a = rand_below<W>(rng, M), b = rand_below<W>(rng, M), cc = rand_below<W>(rng, M);
        W r;
        const W q = DW::mac_split(a, b, cc, M, r);
        // exact check in u256: a*b + c == q*M + r, r < M
        u256 lhs = engine::detail::mul_wide128(static_cast<u128>(a), static_cast<u128>(b));
        lhs += u256(static_cast<u128>(cc));
        u256 rhs = engine::detail::mul_wide128(static_cast<u128>(q), static_cast<u128>(M));
        rhs += u256(static_cast<u128>(r));
        if (!(lhs == rhs) || !(r < M)) ++bad;
        // split with a random hi < M
        const W hi = rand_below<W>(rng, M), lo = static_cast<W>((static_cast<u128>(rng.next()) << 64) | rng.next());
        W r2;
        const W q2 = DW::split(hi, lo, M, r2);
        u256 l2;
        if constexpr (std::is_same<W, u128>::value) l2 = u256(hi, lo, 0);
        else l2 = u256((static_cast<u128>(hi) << DW::bits) | static_cast<u128>(lo));
        u256 r2v = engine::detail::mul_wide128(static_cast<u128>(q2), static_cast<u128>(M));
        r2v += u256(static_cast<u128>(r2));
        if (!(l2 == r2v) || !(r2 < M)) ++bad;
    }
    // the boundary: (M-1)*(M-1) + (M-1)
    {   // (M-1)^2 + (M-1) = (M-1) M  =>  q = M-1, r = 0
        W r;
        const W q = DW::mac_split(M - 1, M - 1, M - 1, M, r);
        if (!(q == M - 1 && r == 0)) ++bad;
    }
    check(bad == 0, t + " (m = " + std::to_string(m) + "): mac_split / split == exact u256 (40000 samples + boundary)");
    check(throws_logic([&] { W r; (void)DW::split(M, 0, M, r); }), t + ": split refuses hi >= M");
    check(DW::narrow(DW::widen(M - 1)) == M - 1, t + ": widen / narrow round trip");
    std::cout << "  (W) " << t << "  m " << m << "  checks " << (g_checks - c0) << "  failing " << (g_fail - f0) << std::endl;
}

void digitword_arm() {
    const uint64_t f0 = g_fail, c0 = g_checks;
    // The predicted table (committed in WP-18a's report before the run).
    struct Row { uint64_t ell; uint32_t m32, m64, m128; };
    const Row rows[] = {{3, 19, 39, 80}, {5, 13, 27, 54}, {7, 11, 22, 45}, {691, 3, 6, 13},
                        {134217757, 0, 2, 4}, {4294967291ULL, 0, 0, 3}};
    for (const Row& r : rows) {
        const bool ok = engine::DigitWord<uint32_t>::digit_exponent(r.ell) == r.m32 &&
                        engine::DigitWord<uint64_t>::digit_exponent(r.ell) == r.m64 &&
                        engine::DigitWord<u128>::digit_exponent(r.ell) == r.m128;
        check(ok, "DigitWord: m(w, " + std::to_string(r.ell) + ") == (" + std::to_string(r.m32) + ", " +
                      std::to_string(r.m64) + ", " + std::to_string(r.m128) + ") [got " +
                      std::to_string(engine::DigitWord<uint32_t>::digit_exponent(r.ell)) + ", " +
                      std::to_string(engine::DigitWord<uint64_t>::digit_exponent(r.ell)) + ", " +
                      std::to_string(engine::DigitWord<u128>::digit_exponent(r.ell)) + "]");
    }
    check(engine::DigitWord<uint32_t>::bits == 32 && engine::DigitWord<uint64_t>::bits == 64 &&
          engine::DigitWord<u128>::bits == 128 && engine::DigitWord<u128>::digit_bits == 127, "DigitWord: bits / digit_bits");
    check(std::is_same<engine::DigitWord<u128>::Ctx, ModCtx<u128>>::value &&
          std::is_same<engine::DigitWord<uint64_t>::Wide, u128>::value &&
          std::is_same<engine::DigitWord<u128>::Wide, u256>::value, "DigitWord: Ctx / Wide types");
    for (uint64_t ell : {5ULL, 691ULL, 134217757ULL}) {
        digitword_split_arm<uint32_t>(ell, 0xD1600032ULL ^ ell);
        digitword_split_arm<uint64_t>(ell, 0xD1600064ULL ^ ell);
        digitword_split_arm<u128>(ell, 0xD1600128ULL ^ ell);
    }
    // the three widths' word entry points agree entrywise at 691^3 (< 2^31: all three carry it)
    {
        using D32 = engine::DigitWord<uint32_t>;
        using D64 = engine::DigitWord<uint64_t>;
        using D128 = engine::DigitWord<u128>;
        const auto c32 = D32::ctx(691, 3);
        const auto c64 = D64::ctx(691, 3);
        const auto c128 = D128::ctx(691, 3);
        Splitmix rng(0xD16A691ULL);
        uint32_t n = 0;
        for (uint32_t r = 0; r < 60; ++r) {
            const uint32_t mm = 1 + static_cast<uint32_t>(rng.below(8)), nn = 1 + static_cast<uint32_t>(rng.below(8));
            const DenseBuf<uint32_t> A = rand_mat(rng, c32, mm, nn);
            DenseBuf<uint64_t> A64(mm, nn);
            for (uint32_t i = 0; i < mm; ++i) for (uint32_t j = 0; j < nn; ++j) A64.at(i, j) = A.at(i, j);
            const DenseBuf<u128> A128 = widen(A);
            const auto h32 = D32::howell(c32, engine::cview(A));
            const auto h64 = D64::howell(c64, engine::cview(A64));
            const auto h128 = D128::howell(c128, engine::cview(A128));
            const auto k32 = D32::kernel_module(c32, engine::cview(A));
            const auto k128 = D128::kernel_module(c128, engine::cview(A128));
            const bool ok = same_howell(h32, h128) && same_howell(h64, h128) && same_howell(k32, k128) &&
                            D32::invariant_factors(c32, engine::cview(A)) == D128::invariant_factors(c128, engine::cview(A128)) &&
                            D64::invariant_factors(c64, engine::cview(A64)) == D128::invariant_factors(c128, engine::cview(A128)) &&
                            D32::length(c32, engine::cview(A)) == D128::length(c128, h128);
            check(ok, "DigitWord<u32 | u64 | u128> at 691^3: word entry points equal entrywise (case " + std::to_string(r) + ")");
            // the word GEMM, all three
            DenseBuf<uint32_t> C32(mm, mm); DenseBuf<uint64_t> C64(mm, mm); DenseBuf<u128> C128(mm, mm);
            DenseBuf<uint32_t> At(nn, mm);
            for (uint32_t i = 0; i < mm; ++i) for (uint32_t j = 0; j < nn; ++j) At.at(j, i) = A.at(i, j);
            DenseBuf<uint64_t> At64(nn, mm); DenseBuf<u128> At128(nn, mm);
            for (uint32_t i = 0; i < nn; ++i) for (uint32_t j = 0; j < mm; ++j) { At64.at(i, j) = At.at(i, j); At128.at(i, j) = At.at(i, j); }
            D32::gemm(c32, engine::cview(A), engine::cview(At), engine::view(C32));
            D64::gemm(c64, engine::cview(A64), engine::cview(At64), engine::view(C64));
            D128::gemm(c128, engine::cview(A128), engine::cview(At128), engine::view(C128));
            bool okg = true;
            for (uint32_t i = 0; i < mm; ++i) for (uint32_t j = 0; j < mm; ++j)
                if (static_cast<u128>(C32.at(i, j)) != C128.at(i, j) || static_cast<u128>(C64.at(i, j)) != C128.at(i, j)) okg = false;
            check(okg, "DigitWord<u32 | u64 | u128> at 691^3: word GEMM equal (case " + std::to_string(r) + ")");
            ++n;
        }
        std::cout << "  (W) three widths at 691^3: " << n << " cases" << std::endl;
    }
    std::cout << "  (W) DigitWord  checks " << (g_checks - c0) << "  failing " << (g_fail - f0) << std::endl;
}

} // namespace

uint32_t EngineU128CompositeValidation() {
    g_fail = 0;
    g_checks = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "EngineU128CompositeValidation: slot-11 WP-18a -- the u128 engine layer (ENGINE_API.md Sec. 13.12, "
                 "step 1) and DigitWord<W>, under the Sec. 13.12.4 oracle rule" << std::endl;

    // (A) arithmetic: 6 moduli x 200000 samples >= 10^6
    construction_refusals();
    const uint32_t N = 200000;
    arithmetic_arm({4294967291ULL, 2}, N, 0xA128F001ULL);   // 2^64 - small
    arithmetic_arm({4294967311ULL, 2}, N, 0xA128F002ULL);   // 2^64 + small
    arithmetic_arm({691, 12}, N, 0xA128F003ULL);            // ~2^113.2
    arithmetic_arm({3, 80}, N, 0xA128F004ULL);              // ~2^126.8
    arithmetic_arm({5, 27}, N, 0xA128F005ULL);              // < 2^63: also vs ModCtx<uint64_t>
    arithmetic_arm({691, 6}, N, 0xA128F006ULL);             // < 2^63: also vs ModCtx<uint64_t>

    // (E) u128 == u32 / u64 on HeckeReductionValidation's elimination-arm
    // matrices (its (H) / (M) generators and seeds, its odd moduli)
    howell_cases_arm<uint32_t>(3u, 2, 400, 0xE8A10009ULL);             // Z/9
    howell_cases_arm<uint32_t>(5u, 2, 400, 0xE8A10019ULL);             // Z/25
    howell_cases_arm<uint32_t>(5u, 3, 400, 0xE8A1007DULL);             // Z/125
    howell_cases_arm<uint32_t>(691u, 2, 300, 0xE8A102B3ULL);           // Z/691^2
    howell_cases_arm<uint64_t>(134217757ull, 2, 200, 0xE8A1C0DEULL);   // Z/134217757^2 (u64 Montgomery)
    cyclic_cases_arm<uint32_t>(5u, 2, 150, 0xE8C10019ULL);
    cyclic_cases_arm<uint32_t>(5u, 3, 150, 0xE8C1007DULL);
    cyclic_cases_arm<uint32_t>(691u, 2, 150, 0xE8C102B3ULL);
    cyclic_cases_arm<uint64_t>(134217757ull, 2, 60, 0xE8C1C0DEULL);
    cyclic_cases_arm<uint64_t>(4294967291ull, 2, 60, 0xE8C1FFFBULL);   // U64_Wide (>= 2^63)
    random_u64_arm(5, 27, 60, 0xE8E50027ULL);                           // U64_Montgomery
    random_u64_arm(691, 6, 60, 0xE8E26906ULL);                          // U64_Montgomery
    random_u64_arm(3, 40, 60, 0xE8E30040ULL);                           // U64_Wide (3^40 ~ 2^63.4 >= 2^63)
    random_u64_arm(4294967291ULL, 2, 40, 0xE8E4FFFBULL);                // U64_Wide (>= 2^63)

    // (B) beyond 2^64
    beyond_arm(4294967311ULL, 2, 60, 0xB128A001ULL);   // ~2^64
    beyond_arm(5, 30, 60, 0xB128A002ULL);              // ~2^69.7
    beyond_arm(7, 40, 60, 0xB128A003ULL);              // ~2^112.3
    beyond_arm(691, 12, 60, 0xB128A004ULL);            // ~2^113.2
    beyond_arm(3, 80, 60, 0xB128A005ULL);              // ~2^126.8

    // (K) the block model
    block_arm(691, 12, 0xB10C0001ULL);
    block_arm(3, 80, 0xB10C0002ULL);

    // (W) DigitWord
    digitword_arm();

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "EngineU128CompositeValidation: " << (g_fail == 0 ? "PASS" : "FAIL") << " (" << g_checks << " checks, "
              << g_fail << " failing, " << secs << " s)" << std::endl;
    return g_fail;
}
